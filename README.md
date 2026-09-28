# PhaseShift

AMD RDNA4 GPU向けの単一ハードウェア推論ランタイムの研究プロジェクト。

目的は一つ:

> 単一ハードウェア（AMD RDNA4 / ROCm / HIP / Linux）で、最速の推論ランタイムを作る。

汎用性は目的ではない。性能が目的。

## Current scope

- ハードウェア: AMD RDNA4（gfx1201、R9700）/ ROCm / HIP / Linux
- Runtime family: Qwen3.5 系 runtime（コード上は `src/phaseshift/models/qwen35/`）
- 主要 performance target: **Qwen3.8-27B-PSQ**
- Speculative decode: **DFlash2** drafter（Qwen3.8-27B target + DFlash2）
- KVキャッシュ: paged KV、BF16 / FP8_E4M3
- GPU-MCU: Qwen model 非依存の low-level substrate（`src/phaseshift/runtime/gpu_mcu/`）

runtime の namespace は `qwen35` のままである。Qwen3.8-27B を「別 family へ rename
済み」ではない。コード上の Qwen3.5 runtime family を基盤として、現在の主要
performance target に Qwen3.8-27B-PSQ を使っている。

機能追加や性能値の正本は README ではなく各 docs を参照する。

## Current execution

- correctness mode + Qwen-local optimized auto dispatch
- `PrimitiveGraph` → `ProgramSet` → Qwen35 Executor
- continuous batching（prefill / decode mixed）
- DFlash2 speculative decode
- GPU-MCU low-level substrate（tests / bench 付き）
- offline quantization（FPX / self-contained quantized safetensors / KLD）

## Server

Qwen3.5 の chat template / reasoning contract を持つ Server path が存在する。
利用方法は [docs/user/server.md](docs/user/server.md)、Server の責務境界は
[docs/developer/architecture.md](docs/developer/architecture.md)、developer 向け入口は
[docs/developer/server_status.md](docs/developer/server_status.md) を参照。prefix cache の
詳細は [docs/developer/prefix_cache.md](docs/developer/prefix_cache.md) を参照。

## Entrypoints

| 名 | 用途 |
|---|---|
| `phaseshift-compute` | 単一request推論（BF16 / 量子化model、greedy、DFlash2 speculative decode） |
| `phaseshift-cli` | チャットUI（Python、HF AutoProcessor） |
| `phaseshift-server` | OpenAI互換chat serving（[docs/user/server.md](docs/user/server.md)） |
| `phaseshift-quantizer` | quantize / verify / convert / validate / kld |
| `phaseshift-bench` | prefill/decode E2E + kernel/operator micro-benchmarks（[docs/user/bench.md](docs/user/bench.md)） |

## Build

標準 build:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPHASESHIFT_ROCM_ROOT="$(rocm-sdk path --root)" \
  -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$(rocm-sdk path --root)" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake --build build --parallel
```

HIP graph execution mode を使う場合は別 option で指定する。

```bash
cmake -S . -B build -DPHASESHIFT_HIP_GRAPH=ON
```

要件・option 一覧: [docs/user/build.md](docs/user/build.md)。

## Tests

required（自己完結: repository + ROCm + 1 GPUのみ）:

```bash
ctest --test-dir build -L required --output-on-failure
```

optional: Qwen3.5-4B 全 application E2E（外部 model + committed oracle fixture、
exact token match）。`PHASESHIFT_BUILD_OPTIONAL_TESTS=ON` で build。
詳細: [docs/developer/testing.md](docs/developer/testing.md)。

## Documentation map

- [docs/user/](docs/user/README.md) — 現在ユーザーが使用する操作方法
- [docs/developer/](docs/developer/README.md) — 現在実装されている architecture / runtime / server / kernel contract
- [docs/perf/current.md](docs/perf/current.md) — 現在の性能値と再現条件
- [docs/perf/methodology.md](docs/perf/methodology.md) — どう測るか
- [docs/rnd/](docs/rnd/README.md) — 過去に何を試してどう判断したか
- [docs/references/](docs/references/README.md) — 外部資料

## License

MIT — [LICENSE](LICENSE)
