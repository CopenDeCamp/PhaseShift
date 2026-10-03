import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "prepare_draft_vocab", ROOT / "tools/quantization/prepare_draft_vocab.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareDraftVocabTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.notice = self.root / "NOTICE.txt"
        self.notice.write_text("source notice\n", encoding="utf-8")
        self.vocab = self.root / "ids.u32"
        self.vocab.write_bytes(b"".join(struct.pack("<I", token_id) for token_id in range(32)))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def tokenizer(self, path: Path, *, equivalent: bool = False, incompatible: bool = False) -> None:
        vocab = {"alpha": 0, "beta": 1, "gamma": 2, "delta": 3}
        if incompatible:
            vocab["beta"] = 7
        added = [
            {"id": 4, "content": "<special>"},
            {"id": 5, "content": "<extra>"},
        ]
        document = {
            "model": {"vocab": vocab},
            "added_tokens": added,
        }
        if equivalent:
            document = {
                "added_tokens": list(reversed(added)),
                "model": {"vocab": {key: vocab[key] for key in reversed(list(vocab))}},
            }
            path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        else:
            path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")

    def args_pack(
        self,
        tokenizer: Path,
        output: Path,
        profile: str = "qwen38-test-v1",
        vocab_file: Path | None = None,
    ):
        return type("Args", (), {
            "vocab_file": vocab_file or self.vocab,
            "tokenizer": tokenizer,
            "profile_id": profile,
            "model_vocab_size": 64,
            "hidden_size": 128,
            "source_manifest_sha256": "a" * 64,
            "notice": self.notice,
            "output_dir": output,
        })()

    def make_bundle(
        self,
        tokenizer: Path,
        name: str = "bundle",
        profile: str = "qwen38-test-v1",
        vocab_file: Path | None = None,
    ) -> Path:
        output = self.root / name
        MODULE.pack(self.args_pack(tokenizer, output, profile, vocab_file=vocab_file))
        return output

    def make_model(
        self,
        tokenizer: Path,
        *,
        nested: bool = True,
        hidden: int = 128,
        name: str = "model",
    ) -> Path:
        model = self.root / name
        model.mkdir()
        shape = {"vocab_size": 64, "hidden_size": hidden}
        config = {"text_config": shape} if nested else shape
        (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (model / "tokenizer.json").write_bytes(tokenizer.read_bytes())
        return model

    def test_canonical_digest_ignores_json_format_and_order(self) -> None:
        first = self.root / "tokenizer-first.json"
        second = self.root / "tokenizer-second.json"
        self.tokenizer(first)
        self.tokenizer(second, equivalent=True)
        self.assertNotEqual(hashlib.sha256(first.read_bytes()).digest(), hashlib.sha256(second.read_bytes()).digest())
        self.assertEqual(
            MODULE.canonical_token_map(first, 64), MODULE.canonical_token_map(second, 64)
        )

    def test_pack_and_install_rebinds_exact_target_tokenizer_hash(self) -> None:
        source_tokenizer = self.root / "source-tokenizer.json"
        target_tokenizer = self.root / "target-tokenizer.json"
        self.tokenizer(source_tokenizer)
        self.tokenizer(target_tokenizer, equivalent=True)
        bundle = self.make_bundle(source_tokenizer)
        metadata = json.loads((bundle / MODULE.METADATA_FILE).read_text(encoding="utf-8"))
        self.assertEqual(metadata["vocab_count"], 32)
        self.assertEqual(metadata["vocab_file"], MODULE.VOCAB_FILE)
        self.assertEqual(metadata["notice_file"], MODULE.NOTICE_FILE)
        self.assertEqual(metadata["source_notices"][0]["file"], MODULE.NOTICE_FILE)
        model = self.make_model(target_tokenizer)
        installed = MODULE.install(type("Args", (), {
            "bundle_dir": bundle, "model_dir": model, "overwrite": False,
        })())
        self.assertEqual(installed["tokenizer_sha256"], hashlib.sha256(target_tokenizer.read_bytes()).hexdigest())
        self.assertEqual(installed["tokenizer_vocab_sha256"], MODULE.canonical_token_map(target_tokenizer, 64))
        self.assertEqual((model / MODULE.VOCAB_FILE).read_bytes(), self.vocab.read_bytes())
        self.assertEqual((model / MODULE.NOTICE_FILE).read_text(encoding="utf-8"), "source notice\n")

    def test_same_id_different_tokens_is_allowed(self) -> None:
        tokenizer = self.root / "same-id.json"
        tokenizer.write_text(json.dumps({
            "model": {"vocab": {"a": 0, "b": 0}},
            "added_tokens": [{"content": "c", "id": 1}],
        }), encoding="utf-8")
        self.assertTrue(MODULE.canonical_token_map(tokenizer, 4))

    def test_same_token_different_ids_is_rejected(self) -> None:
        tokenizer = self.root / "conflict.json"
        tokenizer.write_text(json.dumps({
            "model": {"vocab": {"a": 0}},
            "added_tokens": [{"content": "a", "id": 1}],
        }), encoding="utf-8")
        with self.assertRaises(MODULE.DraftVocabError):
            MODULE.canonical_token_map(tokenizer, 4)

    def test_pack_rejects_duplicate_unaligned_and_out_of_range_ids(self) -> None:
        tokenizer = self.root / "tokenizer.json"
        self.tokenizer(tokenizer)
        for name, ids in (
            ("duplicate.u32", list(range(31)) + [30]),
            ("unaligned.u32", list(range(16))),
            ("out-of-range.u32", list(range(31)) + [64]),
        ):
            path = self.root / name
            path.write_bytes(b"".join(struct.pack("<I", token_id) for token_id in ids))
            with self.assertRaises(MODULE.DraftVocabError):
                MODULE.pack(self.args_pack(
                    tokenizer, self.root / name.removesuffix(".u32"), vocab_file=path
                ))

    def test_install_rejects_mapping_shape_and_payload_hash_errors(self) -> None:
        source_tokenizer = self.root / "source.json"
        target_tokenizer = self.root / "target.json"
        self.tokenizer(source_tokenizer)
        self.tokenizer(target_tokenizer, incompatible=True)
        bundle = self.make_bundle(source_tokenizer)
        model = self.make_model(target_tokenizer)
        with self.assertRaises(MODULE.DraftVocabError):
            MODULE.install(type("Args", (), {
                "bundle_dir": bundle, "model_dir": model, "overwrite": False,
            })())

        compatible_model = self.make_model(source_tokenizer, name="compatible-model")
        payload_path = bundle / MODULE.VOCAB_FILE
        payload_path.write_bytes(payload_path.read_bytes()[:-4] + struct.pack("<I", 63))
        with self.assertRaises(MODULE.DraftVocabError):
            MODULE.install(type("Args", (), {
                "bundle_dir": bundle, "model_dir": compatible_model, "overwrite": False,
            })())

    def test_install_preserves_other_model_files_and_rejects_different_profile(self) -> None:
        tokenizer = self.root / "tokenizer.json"
        self.tokenizer(tokenizer)
        bundle_a = self.make_bundle(tokenizer, "bundle-a", "profile-a")
        bundle_b = self.make_bundle(tokenizer, "bundle-b", "profile-b")
        model = self.make_model(tokenizer, nested=False)
        sentinel = model / "weights.safetensors"
        sentinel.write_bytes(b"preserve")
        MODULE.install(type("Args", (), {"bundle_dir": bundle_a, "model_dir": model, "overwrite": False})())
        with self.assertRaises(MODULE.DraftVocabError):
            MODULE.install(type("Args", (), {"bundle_dir": bundle_b, "model_dir": model, "overwrite": False})())
        MODULE.install(type("Args", (), {"bundle_dir": bundle_b, "model_dir": model, "overwrite": True})())
        self.assertEqual(sentinel.read_bytes(), b"preserve")
        self.assertEqual(json.loads((model / MODULE.METADATA_FILE).read_text())["profile_id"], "profile-b")

    def test_same_bundle_is_idempotent_but_same_profile_different_payload_requires_overwrite(self) -> None:
        tokenizer = self.root / "tokenizer-idempotent.json"
        self.tokenizer(tokenizer)
        bundle = self.make_bundle(tokenizer, "bundle-idempotent", "profile-same")
        model = self.make_model(tokenizer, name="idempotent-model")
        first = MODULE.install(type("Args", (), {
            "bundle_dir": bundle, "model_dir": model, "overwrite": False,
        })())
        first_bytes = {
            name: (model / name).read_bytes()
            for name in (MODULE.VOCAB_FILE, MODULE.METADATA_FILE, MODULE.NOTICE_FILE)
        }
        second = MODULE.install(type("Args", (), {
            "bundle_dir": bundle, "model_dir": model, "overwrite": False,
        })())
        self.assertEqual(first, second)
        self.assertEqual(
            first_bytes,
            {
                name: (model / name).read_bytes()
                for name in (MODULE.VOCAB_FILE, MODULE.METADATA_FILE, MODULE.NOTICE_FILE)
            },
        )

        different_vocab = self.root / "different.u32"
        different_vocab.write_bytes(
            b"".join(struct.pack("<I", token_id) for token_id in list(range(31)) + [32])
        )
        different_bundle = self.make_bundle(
            tokenizer, "bundle-different-content", "profile-same", vocab_file=different_vocab
        )
        with self.assertRaises(MODULE.DraftVocabError):
            MODULE.install(type("Args", (), {
                "bundle_dir": different_bundle, "model_dir": model, "overwrite": False,
            })())
        MODULE.install(type("Args", (), {
            "bundle_dir": different_bundle, "model_dir": model, "overwrite": True,
        })())

    def test_replace_failure_restores_existing_files_and_fresh_install_leaves_none(self) -> None:
        tokenizer = self.root / "tokenizer-recovery.json"
        self.tokenizer(tokenizer)
        original_bundle = self.make_bundle(tokenizer, "bundle-recovery-old", "profile-recovery")
        different_vocab = self.root / "recovery-different.u32"
        different_vocab.write_bytes(
            b"".join(struct.pack("<I", token_id) for token_id in list(range(31)) + [32])
        )
        different_bundle = self.make_bundle(
            tokenizer,
            "bundle-recovery-new",
            "profile-recovery",
            vocab_file=different_vocab,
        )
        model = self.make_model(tokenizer, name="recovery-model")
        MODULE.install(type("Args", (), {
            "bundle_dir": original_bundle, "model_dir": model, "overwrite": False,
        })())
        old_bytes = {
            name: (model / name).read_bytes()
            for name in (MODULE.VOCAB_FILE, MODULE.METADATA_FILE, MODULE.NOTICE_FILE)
        }
        original_replace = Path.replace
        calls = 0

        def fail_after_first(source: Path, target: Path):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected replace failure")
            return original_replace(source, target)

        with patch.object(Path, "replace", fail_after_first):
            with self.assertRaises(MODULE.DraftVocabError):
                MODULE.install(type("Args", (), {
                    "bundle_dir": different_bundle, "model_dir": model, "overwrite": True,
                })())
        self.assertEqual(
            old_bytes,
            {
                name: (model / name).read_bytes()
                for name in (MODULE.VOCAB_FILE, MODULE.METADATA_FILE, MODULE.NOTICE_FILE)
            },
        )
        self.assertFalse(list(model.glob(".dflash2-vocab-recovery.*")))

        fresh_model = self.make_model(tokenizer, name="recovery-fresh-model")
        calls = 0
        with patch.object(Path, "replace", fail_after_first):
            with self.assertRaises(MODULE.DraftVocabError):
                MODULE.install(type("Args", (), {
                    "bundle_dir": original_bundle, "model_dir": fresh_model, "overwrite": False,
                })())
        self.assertFalse(any((fresh_model / name).exists() for name in (
            MODULE.VOCAB_FILE, MODULE.METADATA_FILE, MODULE.NOTICE_FILE
        )))
        self.assertFalse(list(fresh_model.glob(".dflash2-vocab-recovery.*")))


if __name__ == "__main__":
    unittest.main()
