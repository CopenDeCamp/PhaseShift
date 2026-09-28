# Qwen3.5-4B Full Application E2E

4つのuser-facing application（compute / cli / bench / quantizer）の
subprocess E2E契約。

## 前提

- 外部model必須: `Qwen3.5-4B`（repository外）
  - CMake option: `PHASESHIFT_MODEL_DIR_4B`（default: `models/Qwen3.5-4B`）
  - 必須geometry: hidden=2560, intermediate=9216, layers=32, q_heads=16,
    kv_heads=4, head_dim=256, vocab=248320, GDN層=24 / full attention層=8
- E2Eは `OPTIONAL` + `EXTERNAL_FILES`。`PHASESHIFT_BUILD_OPTIONAL_TESTS=OFF`
  ではregistration / buildされない。`PHASESHIFT_BUILD_BENCHMARKS=ON` 必須。
- formal qualificationではmodel必須・SKIP禁止。

## Oracle fixture provenance

`fixtures/qwen35_4b_oracle.json` は独立Oracle（Hugging Face Transformers +
PyTorch）によるgreedy生成の正式contract。PhaseShiftコードをimportしない。

- dtype: float32 / attention: eager / greedy
- `local_files_only=True`（network accessなし）
- 全stepのtop1-top2 margin >= 0.10 をgateに採用（near-tie禁止）
- model fingerprint（config / tokenizer / safetensors の SHA-256）を保持
- raw caseはBF16 / FP8 KV双方でoracle一致が確認済みの候補を採用
- 再生成:

```bash
HIP_VISIBLE_DEVICES=0 python3 tests/e2e/generate_qwen35_4b_oracle.py \
  --model-dir <Qwen3.5-4B> \
  --output tests/e2e/fixtures/qwen35_4b_oracle.json \
  --device cuda:0
```

  出力はdeterministic（timestampなし）。`cmp` で同一性を確認可能。

## E2E app coverage

```text
phaseshift-compute
    BF16 KV / FP8 KV / raw token input / greedy
    GENERATED_IDS == Oracle exact（BF16 == FP8 == Oracle）
    --dump-logits（argmax == GENERATED_IDS / NaN 0 / Inf 0）
    resource cleanup（exit 0）

phaseshift-cli
    --prompt single-shot（decoded text == Oracle）
    interactive multi-turn（turn1 / turn2 exact）
    HF tokenizer / chat template（enable_thinking=False）

phaseshift-bench
    pp forward / pp greedy / tg forward / tg greedy（deterministic）
    gemm correctness（BF16 direct + --check）

phaseshift-quantizer
    quantize / verify / kld(self)
    KLD: nonfinite 0 / mean < 1e-7 / max < 1e-6
```

## execution

```bash
export ROCM_PATH="$(rocm-sdk path --root)"
MODEL4B=/path/to/Qwen3.5-4B
ARCH=gfx1201

cmake -S . -B build-e2e -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPHASESHIFT_ROCM_ROOT="${ROCM_PATH}" -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="${ROCM_PATH}" \
  -DCMAKE_HIP_ARCHITECTURES="${ARCH}" \
  -DPHASESHIFT_BUILD_TESTS=ON \
  -DPHASESHIFT_BUILD_OPTIONAL_TESTS=ON \
  -DPHASESHIFT_BUILD_BENCHMARKS=ON \
  -DPHASESHIFT_MODEL_DIR_4B="${MODEL4B}"

cmake --build build-e2e --target phaseshift-e2e-apps --parallel

ctest --test-dir build-e2e --output-on-failure -L e2e
```

quantizer suiteのwork dirは `PHASESHIFT_E2E_WORK_DIR`（既定 `/tmp` 配下、
free space >= 20 GiB要求）。failure時 `PHASESHIFT_E2E_KEEP_WORK=1` で
work dirを保持できる。
