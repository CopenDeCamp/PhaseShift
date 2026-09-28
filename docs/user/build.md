# Build

## 要件

- OS: Linux
- ROCm toolchain（clang++ / hipcc）。GCC・libstdc++は使用不可
- host libc++ 23: `/usr/lib/llvm-23/include/c++/v1`（`libc++-23-dev` / `libc++abi-23-dev`）。
  別箇所なら `-DPHASESHIFT_LIBCXX_LLVM_ROOT=<root>` で指定
- GPU: AMD Radeon AI PRO R9700（gfx1201）
- CMake >= 3.24
- Ninja
- OpenSSL（libcrypto）
- Go >= 1.26、make、git、curl、unzip、network access
  （PhaseShift patched LocalAI runtime を自前で build する場合のみ。release bundle 同梱の
  runtime を使う場合は不要）
- Python 3（`phaseshift-cli`、`phaseshift-server`、E2E）
  - `phaseshift-server` / backend: `transformers==5.14.1`、`grpcio`、`protobuf`、`xgrammar==0.2.5.post1`
  - server test: `openai`（OpenAI Python client E2E）
  - LocalAI v4.10.0 + PhaseShift patched runtime（`vendor/localai/patches/` の patch series）
    （`phaseshift-server` 起動時のみ）

`xgrammar==0.2.5.post1` は `phaseshift-server` の Structured Output（GBNF constrained decode）に
必須である。native constraint engine は vendored XGrammar v0.2.5 (commit `2ea71da`) を使うが、
TokenizerInfo sidecar の生成は Python xgrammar で行う。`phaseshift-compute` 単体は Python 非依存。

server dependency は次の version を required dependency として固定する。

```bash
pip install 'transformers==5.14.1' 'grpcio' 'protobuf' 'xgrammar==0.2.5.post1'
```

`transformers==5.14.1` は tool parser / chat template の regression を避けるため固定する。
LocalAI base は `v4.10.0` であり、`phaseshift-server` は PhaseShift fail-closed patch
適用済み runtime を要求する。patched runtime は `tools/build_localai_runtime.sh` で
LocalAI v4.10.0 の checkout から build する。

`rocm-sdk` CLIがある環境では `rocm-sdk path --root` でROCm rootを解決する。
なければ `-DPHASESHIFT_ROCM_ROOT=<path>` か環境変数 `ROCM_PATH` を使う。

## 手順

```bash
export ROCM_PATH="$(rocm-sdk path --root)"
ARCH=gfx1201

cmake -S . -B build-${ARCH} -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPHASESHIFT_ROCM_ROOT="${ROCM_PATH}" \
  -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="${ROCM_PATH}" \
  -DCMAKE_HIP_ARCHITECTURES="${ARCH}"
cmake --build build-${ARCH} --parallel
```

PhaseShiftが対象とするのは `gfx1201`（AMD Radeon AI PRO R9700）のみである。
`CMAKE_HIP_ARCHITECTURES` に他のarchを指定した場合、configureで `FATAL_ERROR` になる。

## LocalAI runtime の build

`phaseshift-server` は PhaseShift patch series 適用済みの LocalAI v4.10.0 runtime を要求する。
release bundle 同梱の runtime を使う場合、この手順は不要である。自前で build する場合のみ
Go >= 1.26 が必要である。

```bash
git clone --depth 1 --branch v4.10.0 \
    https://github.com/mudler/LocalAI.git /path/to/LocalAI

tools/build_localai_runtime.sh --source-dir /path/to/LocalAI
# 既定の出力: build/local-ai
```

- 要件: Go >= 1.26、make、git、curl、unzip、network access。LocalAI v4.10.0 の `go.mod` は
  `go 1.26.0` を要求する。
- script は既定で `GOTOOLCHAIN=local` を使う。installed Go が 1.26 未満の場合は Go 1.26+ を
  導入するか、`GOTOOLCHAIN=auto` を指定して toolchain を取得させる。
- build した runtime は `--localai-binary`、または `PHASESHIFT_LOCALAI_BINARY` で指定する。

## options

| option | 既定 | 説明 |
|---|---|---|
| `PHASESHIFT_BUILD_TESTS` | ON | test suite build |
| `PHASESHIFT_BUILD_OPTIONAL_TESTS` | OFF | Qwen3.5-4B E2E（`PHASESHIFT_BUILD_BENCHMARKS=ON` 必須） |
| `PHASESHIFT_BUILD_BENCHMARKS` | ON | benchmark executables（bench + E2E前提） |
| `PHASESHIFT_HIP_GRAPH` | OFF | HIP graph execution mode |
| `PHASESHIFT_MODEL_DIR_4B` | `models/Qwen3.5-4B` | E2E用外部model directory |
| `PHASESHIFT_LIBCXX_LLVM_ROOT` | `/usr/lib/llvm-23` | host libc++ 23 root |
| `CMAKE_HIP_ARCHITECTURES` | 必須 | 対象GPUアーキテクチャ（`gfx1201` のみ） |

`PHASESHIFT_HIP_GRAPH` は graph capture / replay による execution mode である。
有効にする場合:

```bash
cmake -S . -B build -DPHASESHIFT_HIP_GRAPH=ON
```

## 成果物

```text
build/phaseshift-compute
build/phaseshift-cli
build/phaseshift-server
build/phaseshift-server-lib/
build/phaseshift-quantizer
build/phaseshift-bench
```

`phaseshift-server` は Python launcher である。起動には LocalAI runtime が必要で、
release bundle 内・`--localai-binary`・`PHASESHIFT_LOCALAI_BINARY` の順に探索する。
