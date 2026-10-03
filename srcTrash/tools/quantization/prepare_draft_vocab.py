#!/usr/bin/env python3
"""DFlash2固定語彙artifactの検証、pack、model sidecar installを行う。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any


SCHEMA_VERSION = "phaseshift-dflash2-vocab-v1"
VOCAB_FILE = "dflash2-draft-vocab.u32"
METADATA_FILE = "dflash2-draft-vocab.json"
NOTICE_FILE = "DRAFT_VOCAB_NOTICE.txt"
SHA256_LENGTH = 64
MIN_VOCAB_COUNT = 32
ALIGNMENT = 16


class DraftVocabError(ValueError):
    """固定語彙artifactの契約違反。"""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        raise DraftVocabError(f"ファイルを読めません: {path}: {error}") from error


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise DraftVocabError(f"{label}は通常のファイルである必要があります: {path}")
    return path


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH:
        raise DraftVocabError(f"{label}はSHA-256文字列である必要があります")
    value = value.lower()
    if any(char not in "0123456789abcdef" for char in value):
        raise DraftVocabError(f"{label}はSHA-256 hexである必要があります")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DraftVocabError(f"{label}は正の整数である必要があります")
    return value


def _profile_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise DraftVocabError("profile_idは1〜128文字である必要があります")
    if "/" in value or "\\" in value or value in {".", ".."}:
        raise DraftVocabError("profile_idにパス区切り文字は使えません")
    return value


def read_vocab_ids(path: Path, model_vocab_size: int) -> list[int]:
    """little-endian uint32列を検証してglobal token ID列として返す。"""
    _regular_file(path, "語彙ファイル")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise DraftVocabError(f"語彙ファイルを読めません: {path}: {error}") from error
    if not payload or len(payload) % 4:
        raise DraftVocabError("語彙ファイルは空でないuint32列である必要があります")
    return validate_vocab_payload(payload, model_vocab_size)


def validate_vocab_payload(
    payload: bytes, model_vocab_size: int, expected_count: int | None = None
) -> list[int]:
    if not payload or len(payload) % 4:
        raise DraftVocabError("語彙payloadは空でないuint32列である必要があります")
    ids = [value[0] for value in struct.iter_unpack("<I", payload)]
    if len(ids) < MIN_VOCAB_COUNT or len(ids) % ALIGNMENT:
        raise DraftVocabError(
            f"語彙数は{MIN_VOCAB_COUNT}以上かつ{ALIGNMENT}の倍数である必要があります"
        )
    if expected_count is not None and len(ids) != expected_count:
        raise DraftVocabError("語彙payloadの語彙数がmetadataと一致しません")
    previous = -1
    for index, token_id in enumerate(ids):
        if token_id >= model_vocab_size:
            raise DraftVocabError(
                f"語彙IDがmodel vocabulary範囲外です: index={index} id={token_id}"
            )
        if token_id <= previous:
            raise DraftVocabError("語彙IDはstrictly increasingである必要があります")
        previous = token_id
    return ids


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DraftVocabError(f"{label}のJSONを読めません: {path}: {error}") from error
    if not isinstance(value, dict):
        raise DraftVocabError(f"{label}はJSON objectである必要があります")
    return value


def canonical_token_map(tokenizer_path: Path, model_vocab_size: int) -> str:
    """tokenizerのvocabとadded_tokensから形式非依存のmap digestを作る。"""
    document = _load_json(tokenizer_path, "tokenizer")
    model = document.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("vocab"), dict):
        raise DraftVocabError("tokenizer.model.vocabがありません")
    mapping: dict[str, int] = {}
    for token, token_id in model["vocab"].items():
        if not isinstance(token, str) or isinstance(token_id, bool) or not isinstance(token_id, int):
            raise DraftVocabError("tokenizer.model.vocabのtokenまたはIDが不正です")
        if token_id < 0 or token_id >= model_vocab_size:
            raise DraftVocabError(f"tokenizer.model.vocabのIDが範囲外です: {token_id}")
        previous = mapping.get(token)
        if previous is not None and previous != token_id:
            raise DraftVocabError(f"同じtokenに複数のIDがあります: {token!r}")
        mapping[token] = token_id

    added_tokens = document.get("added_tokens", [])
    if not isinstance(added_tokens, list):
        raise DraftVocabError("tokenizer.added_tokensはlistである必要があります")
    for index, entry in enumerate(added_tokens):
        if not isinstance(entry, dict):
            raise DraftVocabError(f"added_tokens[{index}]がobjectではありません")
        token = entry.get("content")
        token_id = entry.get("id")
        if not isinstance(token, str) or isinstance(token_id, bool) or not isinstance(token_id, int):
            raise DraftVocabError(f"added_tokens[{index}]のtokenまたはIDが不正です")
        if token_id < 0 or token_id >= model_vocab_size:
            raise DraftVocabError(f"added_tokens[{index}]のIDが範囲外です: {token_id}")
        previous = mapping.get(token)
        if previous is not None and previous != token_id:
            raise DraftVocabError(f"同じtokenに複数のIDがあります: {token!r}")
        mapping[token] = token_id

    canonical = bytearray()
    for token, token_id in sorted(mapping.items(), key=lambda item: (item[1], item[0].encode("utf-8"))):
        encoded = token.encode("utf-8")
        canonical.extend(struct.pack("<II", token_id, len(encoded)))
        canonical.extend(encoded)
    return _sha256_bytes(bytes(canonical))


def _source_notices(notice_sha256: str) -> list[dict[str, str]]:
    return [{"file": NOTICE_FILE, "sha256": notice_sha256}]


def _metadata(
    profile_id: str,
    vocab_count: int,
    target_vocab_size: int,
    target_hidden_size: int,
    vocab_sha256: str,
    tokenizer_sha256: str,
    tokenizer_vocab_sha256: str,
    source_manifest_sha256: str,
    notice_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile_id,
        "vocab_file": VOCAB_FILE,
        "vocab_count": vocab_count,
        "target_vocab_size": target_vocab_size,
        "target_hidden_size": target_hidden_size,
        "vocab_sha256": vocab_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer_vocab_sha256": tokenizer_vocab_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "notice_file": NOTICE_FILE,
        "source_notices": _source_notices(notice_sha256),
    }


def _validate_source_notices(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise DraftVocabError("source_noticesは空でないlistである必要があります")
    result: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            raise DraftVocabError(f"source_notices[{index}]が不正です")
        digest = _sha256(item.get("sha256"), f"source_notices[{index}].sha256")
        result.append({"file": item["file"], "sha256": digest})
    return result


def validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise DraftVocabError(f"schema_versionは{SCHEMA_VERSION}である必要があります")
    profile_id = _profile_id(metadata.get("profile_id"))
    if metadata.get("vocab_file") != VOCAB_FILE:
        raise DraftVocabError(f"vocab_fileは{VOCAB_FILE}である必要があります")
    if metadata.get("notice_file") != NOTICE_FILE:
        raise DraftVocabError(f"notice_fileは{NOTICE_FILE}である必要があります")
    count = _positive_int(metadata.get("vocab_count"), "vocab_count")
    if count < MIN_VOCAB_COUNT or count % ALIGNMENT:
        raise DraftVocabError("vocab_countのalignmentが不正です")
    result = dict(metadata)
    result["profile_id"] = profile_id
    result["vocab_count"] = count
    for key in ("target_vocab_size", "target_hidden_size"):
        result[key] = _positive_int(metadata.get(key), key)
    for key in (
        "vocab_sha256",
        "tokenizer_sha256",
        "tokenizer_vocab_sha256",
        "source_manifest_sha256",
    ):
        result[key] = _sha256(metadata.get(key), key)
    result["source_notices"] = _validate_source_notices(metadata.get("source_notices"))
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(_json_bytes(value))


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _bundle_files(bundle_dir: Path) -> tuple[Path, Path, Path]:
    if bundle_dir.is_symlink() or not bundle_dir.is_dir():
        raise DraftVocabError(f"bundle directoryがありません: {bundle_dir}")
    return (
        _regular_file(bundle_dir / VOCAB_FILE, VOCAB_FILE),
        _regular_file(bundle_dir / METADATA_FILE, METADATA_FILE),
        _regular_file(bundle_dir / NOTICE_FILE, NOTICE_FILE),
    )


def pack(args: argparse.Namespace) -> dict[str, Any]:
    vocab_file = _regular_file(args.vocab_file.resolve(), "語彙ファイル")
    tokenizer = _regular_file(args.tokenizer.resolve(), "tokenizer")
    notice = _regular_file(args.notice.resolve(), "notice")
    profile = _profile_id(args.profile_id)
    model_vocab_size = _positive_int(args.model_vocab_size, "model_vocab_size")
    hidden_size = _positive_int(args.hidden_size, "hidden_size")
    source_manifest = _sha256(args.source_manifest_sha256, "source_manifest_sha256")
    ids = read_vocab_ids(vocab_file, model_vocab_size)
    tokenizer_map_sha = canonical_token_map(tokenizer, model_vocab_size)
    notice_bytes = notice.read_bytes()
    metadata = _metadata(
        profile,
        len(ids),
        model_vocab_size,
        hidden_size,
        _sha256_file(vocab_file),
        _sha256_file(tokenizer),
        tokenizer_map_sha,
        source_manifest,
        _sha256_bytes(notice_bytes),
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise DraftVocabError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        shutil.copyfile(vocab_file, staging / VOCAB_FILE)
        (staging / NOTICE_FILE).write_bytes(notice_bytes)
        _write_json(staging / METADATA_FILE, metadata)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output_dir)
    except (OSError, DraftVocabError) as error:
        shutil.rmtree(staging, ignore_errors=True)
        raise DraftVocabError(f"bundle作成に失敗しました: {error}") from error
    return metadata


def _read_bundle(bundle_dir: Path) -> tuple[dict[str, Any], bytes, bytes]:
    vocab_path, metadata_path, notice_path = _bundle_files(bundle_dir.resolve())
    metadata = validate_metadata(_load_json(metadata_path, METADATA_FILE))
    vocab_bytes = vocab_path.read_bytes()
    notice_bytes = notice_path.read_bytes()
    if _sha256_bytes(vocab_bytes) != metadata["vocab_sha256"]:
        raise DraftVocabError("bundle語彙payloadのSHA-256がmetadataと一致しません")
    if len(vocab_bytes) // 4 != metadata["vocab_count"]:
        raise DraftVocabError("bundle語彙数がmetadataと一致しません")
    validate_vocab_payload(vocab_bytes, metadata["target_vocab_size"], metadata["vocab_count"])
    notice_digest = _sha256_bytes(notice_bytes)
    if not any(
        item["file"] == NOTICE_FILE and item["sha256"] == notice_digest
        for item in metadata["source_notices"]
    ):
        raise DraftVocabError("bundle noticeのSHA-256がmetadataと一致しません")
    return metadata, vocab_bytes, notice_bytes


def _config_shape(model_dir: Path) -> tuple[int, int]:
    config = _load_json(_regular_file(model_dir / "config.json", "config.json"), "config.json")
    shape = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    return (
        _positive_int(shape.get("vocab_size"), "config.vocab_size"),
        _positive_int(shape.get("hidden_size"), "config.hidden_size"),
    )


def _existing_artifact(model_dir: Path) -> tuple[bool, bytes | None, bytes | None, bytes | None]:
    paths = _managed_paths(model_dir)
    present = [path.exists() or path.is_symlink() for path in paths]
    if not any(present):
        return False, None, None, None
    for path, is_present in zip(paths, present):
        if is_present:
            _regular_file(path, path.name)
    if not all(present):
        return True, None, None, None
    try:
        metadata_bytes = paths[1].read_bytes()
        validate_metadata(json.loads(metadata_bytes.decode("utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError, DraftVocabError):
        metadata_bytes = None
    return True, paths[0].read_bytes(), metadata_bytes, paths[2].read_bytes()


def _managed_paths(model_dir: Path) -> list[Path]:
    return [model_dir / VOCAB_FILE, model_dir / METADATA_FILE, model_dir / NOTICE_FILE]


def _make_recovery_backup(model_dir: Path) -> tuple[Path, list[bool]]:
    paths = _managed_paths(model_dir)
    present = [path.exists() or path.is_symlink() for path in paths]
    for path, is_present in zip(paths, present):
        if is_present:
            _regular_file(path, path.name)
    backup = Path(tempfile.mkdtemp(prefix=".dflash2-vocab-recovery.", dir=model_dir))
    try:
        for path, is_present in zip(paths, present):
            if is_present:
                shutil.copyfile(path, backup / path.name)
    except OSError as error:
        shutil.rmtree(backup, ignore_errors=True)
        raise DraftVocabError(f"既存artifactのbackup作成に失敗しました: {error}") from error
    return backup, present


def _rollback_install(model_dir: Path, backup: Path, present: list[bool]) -> None:
    paths = _managed_paths(model_dir)
    failures: list[str] = []
    for path, was_present in zip(paths, present):
        try:
            if was_present:
                source = backup / path.name
                if not source.is_file():
                    raise OSError(f"backup file missing: {source}")
                temporary_fd, temporary_name = tempfile.mkstemp(
                    prefix=f".{path.name}.restore.", dir=model_dir
                )
                temporary = Path(temporary_name)
                try:
                    with open(temporary_fd, "wb", closefd=True) as stream:
                        stream.write(source.read_bytes())
                    os.replace(temporary, path)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            elif path.exists() or path.is_symlink():
                _regular_file(path, path.name)
                path.unlink()
        except (OSError, DraftVocabError) as error:
            failures.append(f"{path.name}: {error}")
    if failures:
        raise DraftVocabError(
            "rollbackに失敗しました。recovery backupを保持しています: "
            f"{backup}; " + "; ".join(failures)
        )
    try:
        shutil.rmtree(backup)
    except OSError as error:
        raise DraftVocabError(
            f"rollback後のrecovery backup削除に失敗しました。backupを保持しています: {backup}: {error}"
        ) from error


def install(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model_dir.resolve()
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise DraftVocabError(f"model directoryがありません: {model_dir}")
    metadata, vocab_bytes, notice_bytes = _read_bundle(args.bundle_dir)
    config_vocab, config_hidden = _config_shape(model_dir)
    if config_vocab != metadata["target_vocab_size"]:
        raise DraftVocabError(
            f"target vocab_size不一致: config={config_vocab} bundle={metadata['target_vocab_size']}"
        )
    if config_hidden != metadata["target_hidden_size"]:
        raise DraftVocabError(
            f"target hidden_size不一致: config={config_hidden} bundle={metadata['target_hidden_size']}"
        )
    tokenizer_path = _regular_file(model_dir / "tokenizer.json", "target tokenizer")
    target_tokenizer_sha = _sha256_file(tokenizer_path)
    target_map_sha = canonical_token_map(tokenizer_path, config_vocab)
    if target_map_sha != metadata["tokenizer_vocab_sha256"]:
        raise DraftVocabError("target tokenizerのcanonical token mapがbundleと一致しません")
    installed = dict(metadata)
    installed["tokenizer_sha256"] = target_tokenizer_sha
    installed = validate_metadata(installed)
    installed_metadata_bytes = _json_bytes(installed)
    present, existing_vocab_bytes, existing_metadata_bytes, existing_notice_bytes = _existing_artifact(model_dir)
    if present and not args.overwrite:
        if (
            existing_vocab_bytes != vocab_bytes
            or existing_metadata_bytes != installed_metadata_bytes
            or existing_notice_bytes != notice_bytes
        ):
            raise DraftVocabError(
                "既存artifactのprofileまたは内容が異なります。上書きには--overwriteが必要です"
            )
        return installed
    recovery_backup, existing_present = _make_recovery_backup(model_dir)
    staging = Path(tempfile.mkdtemp(prefix=".dflash2-vocab-install.", dir=model_dir))
    try:
        (staging / VOCAB_FILE).write_bytes(vocab_bytes)
        (staging / NOTICE_FILE).write_bytes(notice_bytes)
        _write_json(staging / METADATA_FILE, installed)
        for name in (VOCAB_FILE, NOTICE_FILE, METADATA_FILE):
            (staging / name).replace(model_dir / name)
        staging.rmdir()
        shutil.rmtree(recovery_backup)
    except (OSError, DraftVocabError) as error:
        shutil.rmtree(staging, ignore_errors=True)
        try:
            _rollback_install(model_dir, recovery_backup, existing_present)
        except DraftVocabError as rollback_error:
            raise DraftVocabError(
                f"model sidecar installに失敗しました: {error}; {rollback_error}"
            ) from error
        raise DraftVocabError(f"model sidecar installに失敗しました: {error}") from error
    return installed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack_parser = subparsers.add_parser("pack", help="verified vocabulary bundleを作成する")
    pack_parser.add_argument("--vocab-file", required=True, type=Path)
    pack_parser.add_argument("--tokenizer", required=True, type=Path)
    pack_parser.add_argument("--profile-id", required=True)
    pack_parser.add_argument("--model-vocab-size", required=True, type=int)
    pack_parser.add_argument("--hidden-size", required=True, type=int)
    pack_parser.add_argument("--source-manifest-sha256", required=True)
    pack_parser.add_argument("--notice", required=True, type=Path)
    pack_parser.add_argument("--output-dir", required=True, type=Path)
    install_parser = subparsers.add_parser("install", help="bundleをmodel directoryへinstallする")
    install_parser.add_argument("--bundle-dir", required=True, type=Path)
    install_parser.add_argument("--model-dir", required=True, type=Path)
    install_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = pack(args) if args.command == "pack" else install(args)
    except (DraftVocabError, OSError) as error:
        print(f"prepare_draft_vocab: FAIL: {error}")
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
