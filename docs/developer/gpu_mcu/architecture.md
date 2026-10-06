# GPU-MCU Production Architecture

本ドキュメントは GPU-MCU の **production architecture の正本** である。
低レイヤー機構（AQL packet / queue / kernarg / CU partition / control ring 等）の
契約と実測は [low_level.md](low_level.md) を参照し、本ドキュメントでは重複させない。

- 開発史・実験・意思決定の記録は `docs/rnd/gpu_mcu/` にある。現在の仕様ではない。
- 現在の性能値は `docs/perf/` を参照する。

## GPU-MCU とは

GPU 上の常駐コントローラ（persistent MCU）が decode plan を自己完結で走らせ、
Host が per-token / per-batch の進行に関与しない実行経路である。
Host は request を渡し、確定した token を非同期に consume する。

対象は RDNA4（gfx1201）+ ROCm + HIP のみ。他のプラットフォーム向けの抽象化を持たない。

## 責務境界

### GPU-MCU substrate（model-independent）

対象: `include/phaseshift/runtime/gpu_mcu/**` と `src/phaseshift/runtime/gpu_mcu/**`。

| 責務 | header |
|---|---|
| AQL packet / queue / kernarg region | `aql.h`, `retained_packet.h` |
| control ring（host ⇄ device SPSC） | `control_ring.h`, `request_ingress.h` |
| slot table / binding / runtime | `slot_table.h`, `slot_binding.h`, `slot_runtime.h` |
| batch substrate | `batch_planner.h`, `batch_binding.h`, `batch_commit.h` |
| KV page / sequence resource | `kv_page_allocator.h`, `sequence_resource.h` |
| output ring | `output_ring.h` |
| persistent controller | `persistent_mcu.h`, `execution_bridge.h` |
| micro FSM | `micro_fsm.h` |
| worker image / CU partition / wall clock | `worker_image.h`, `fsm_worker.h`, `cu_partition.h`, `wall_clock.h` |
| completion | `completion.h`, `device_completion.h` |

- substrate は Qwen3.5 を参照しない。`test_architecture_boundaries` が
  `include/phaseshift/runtime/**` と `src/phaseshift/runtime/**` を走査して検証する。
- 消費者は Qwen3.5 GPU-MCU backend、`phaseshift-bench` の GPU-MCU subcommand、
  GPU-MCU テストのみ。
- stream bridge は GPU-MCU 固有ではなく generic runtime 側にある
  （`include/phaseshift/runtime/stream_bridge.h`、`ps::runtime`）。GPU-MCU と
  Host runtime は同じ `stream_signal_alloc` / `stream_write_value32` /
  `stream_wait_value32` を使う。

### Qwen35 GPU-MCU backend

対象: `src/phaseshift/models/qwen35/runtime/mcu_*` と
Qwen3.5 kernel の standalone HSACO。

| 責務 | ファイル |
|---|---|
| plan compilation | `mcu_plan_compiler.h/.hip` |
| plan cache | `mcu_plan_cache.h/.hip` |
| kernel registry | `mcu_kernel_registry.h/.hip` |
| decode runtime | `mcu_decode_runtime.h/.hip` |
| decode state | `mcu_decode_state.h` |
| KV addressing の接続 | `mcu_kv_binding.h` |
| body range | `mcu_layer_range.h`, `mcu_plan_value.h` |
| backend 選択 policy | `decode_backend.h/.cpp` |
| host 側の MCU 起動・供給 | `executor.hip`（`ensure_mcu_state` / `preflight_mcu_plan` / `execute_mcu_hybrid` / `stage_mcu_commit_slots`） |

backend は substrate の AQL / FSM / persistent controller をそのまま使い、
Qwen3.5 固有のものは plan・kernel inventory・kernarg recipe・KV addressing に限られる。

### Host runtime との境界

- scheduler / ContinuousBatcher / `ScheduledBatch` は substrate を認識しない。
  GPU topology は executor 内部の execution strategy である。
- `Executor::config.backend`（`DecodeBackend::Host` / `GpuMcu`）が
  経路の入口。`decide_decode_backend()` が eligibility と `DecodeBackendReason` を返し、
  `GpuMcu` 要求で成立しない場合は `decode_backend_execution_error()` が
  `Status::unsupported` を返す。Host 経路へは落ちない。
- 判定材料は request 数・token 数・prefill 有無・KV dtype・Tensor Parallel・
  stochastic sampling の有無・speculative verify・
  imatrix / value trace / hidden tap の有無・stream wait 支援・`mcu_body_range` の有効性。
- capability 判定は preflight で行う。`create_model_executor()` が config 段
  （TP・hidden tap・KV dtype）を、`submit_co_batch()` が batch 段
  （`decide_decode_backend()` → `preflight_mcu_plan()`）を担当し、
  どちらもこの batch の program dispatch より前に `Status::unsupported` を返す。
  `preflight_mcu_plan()` は unsupported kernel / physical variant を
  plan compile で検出し、`execute_mcu_hybrid()` は preflight 済み plan を consume するだけ。
  plan compile は batch の device staging（request upload と `DeviceBatchContext` の H2D）
  の後ろに置く。`gdn_conv_dispatch` / `gdn_recurrence_dispatch` が host から
  `batch_context->requests[r]` を読むため、staging より前には置けない。
- persistent 準備前は prefill・verify・複数 request・単一 token 以外が対象外で、
  `GpuMcu` 要求なら error になる。persistent 準備後は prefill・verify・複数 request を
  GPU-MCU が受け、BF16 以外の KV・Tensor Parallel・stochastic sampling・
  imatrix・value trace・hidden tap・body range 不備は引き続き対象外。
- embedding storage は BF16 / PSQ8 両方を plan compile できる。Host resolver と
  selector を共有し、recipe は `kMcuKernargRecipeEmbeddingBf16` /
  `kMcuKernargRecipeEmbeddingPsq8` を選ぶ。selector 非対象 shape のみ
  Host prefix に残る。

## target 構成

```text
phaseshift_gpu_mcu            substrate（AQL / ring / slot / persistent controller / FSM）
  -> phaseshift_core / phaseshift_gpu / hip::host / hsa-runtime64
  source: 14 translation units、probe worker HSACO の blob を保持

phaseshift_qwen35_gpu_mcu     Qwen3.5 kernel HSACO の blob 所有者
  -> phaseshift_gpu_mcu (PUBLIC)
  source: embedded_kernels.cpp、生成 .inc を 19 件コンパイル

phaseshift_qwen35_runtime     Qwen3.5 runtime
  -> phaseshift_gpu_mcu, phaseshift_qwen35_gpu_mcu, qwen35 / kernels / runtime / gpu
```

- `phaseshift` aggregate は `phaseshift_gpu_mcu` を明示リストに含めない。
  しかし `phaseshift_qwen35_runtime` が PUBLIC で link するため、
  `phaseshift-compute` / `phaseshift-quantizer` は両 archive と `hsa-runtime64` を実 link する。
  production decode が HSA に依存するのはこのためである。
- GPU-MCU の HSACO blob は substrate target に置かない。
  blob の消費者は `mcu_kernel_registry.hip` だけであり、所有者も Qwen3.5 backend 側にある。

## persistent controller の役割

- `GpuMcuPersistentMcu` が mapped stop flag / heartbeat / started / iterations を所有し、
  control stream 上へ persistent kernel を launch する。
- kernel は control WGP に常駐し、stop request まで loop する。
- Host は `configure_*` で渡したテーブルと ring を触るだけで、進行に関与しない。
- `GpuMcuFsm` は Host 駆動の standalone 実行経路であり、production では起動しない。
  `gpu_mcu_fsm_kernel` → `mcu_control_loop` → `mcu_run_once` は維持するが、
  production は persistent loop 経路を使う。
- shutdown 順序は stop request → wait stopped → stream sync。destructor は最終 fallback。
- 同期 API の制約と実測は [low_level.md](low_level.md) の `persistent MCU lifecycle` を参照。

## AQL queue の役割

- HSA queue は substrate が所有し、`shutdown()` で明示的に解放する。
- packet は 64 byte、`hsa_kernel_dispatch_packet_t` と static_assert で照合する。
- publication は slot reserve → DW0 INVALID → body copy → fence → DW0 publish → doorbell の順。
- MCU は stage / commit を使い、static region の dispatch を先行 enqueue する。
- doorbell mode と queue backpressure、fence scope は
  [low_level.md](low_level.md) の `continuous AQL queue feeder` / `kernarg generation and fence scope` が正本。

## request ingress と slot lifecycle

```text
Host
  -> control ring (host->device)  GpuMcuControlCommand
  -> GpuMcuRequestDescriptor (128 byte) / GpuMcuRequestIngressEntry
  -> idle slot があれば claim、なければ GpuMcuPendingAdmission の backlog へ
  -> GPU 上で phase 遷移
  -> terminal（理由付き）-> output ring へ publish -> slot release
  -> backlog があるなら次の request を admitted
```

- `GpuMcuSlotState` は 64 byte aligned・standard layout・trivially copyable で、
  `gpu_mcu_slot_table_claim` / `set_phase` / `mark_terminal` / `release` の一連で遷移する。
- terminal は理由（`GpuMcuTerminalReason`）を伴い、
  `gpu_mcu_publish_slot_terminals` が output ring へ書き、
  `gpu_mcu_release_terminal_slots` が slot を返す。
- backlog は固定容量。空き slot が出た時点で admitted され、
  満杯なら reject ではなく待ちになる。
- control ring の slot ABI は `sequence` + 56 byte opaque payload のみで、
  request / cancel / epoch などは payload 上の command ABI として定義する。

## completion / output

- hot path の完了観測は device 側の `GpuMcuDeviceCompletion`（agent scope）。
  host から読まない。
- host-visible な `GpuMcuCompletion` は qualification / debug 用。
- static region 末尾の completion marker だけを MCU が待ち、
  region 内部の実 primitive は poll しない。
- token は `OutputRing` へ `GpuMcuOutputRecord`（56 byte）として積まれ、
  64 byte の `OutputRingSlot` を介して Host が consume する。
- backpressure は ring の full 判定で返し、block ではなく待機として扱う。

## Qwen35 plan compilation

```text
Program (lower_to_primitives の出力)
  -> compile_mcu_plan(program, context, options, McuCompiledPlan)
  -> McuPlanCache が fingerprint で hit/miss を判定
  -> McuCompiledPlan { nodes, variants, invocation, epilogue, dynamic binding }
  -> GpuMcuPersistentMcu へ configure
  -> mcu_run_once が plan node を解釈して AQL dispatch
```

- plan node は `McuPlanNode`（variant_id / next / kernarg_recipe / completion_slot 等）。
  interpreter は `variant_id` のみ解決し、`kernel_object` を持たない。
- verify acceptance は runtime transaction であり Program graph には入れない。
  `McuCompiledPlan::epilogue` として runtime が node を積む。
- plan を 1 つも compile できなければ compile の `Status` をそのまま返す。
  Host 経路へは戻らない。
  `McuDecodeState::full_plan` は full plan（`0..dispatch_count`）を選んだことを示す。

## kernel registry と embedded HSACO

```text
src/phaseshift/models/qwen35/kernels/optimized/*.hip
  -> hipcc --genco で standalone HSACO（phaseshift_add_hsaco）
  -> dd で CLANG_OFFLOAD_BUNDLE header を除去して ELF HSACO へ
  -> xxd -i で .inc 化（phaseshift_embed_gpu_mcu_hsaco）
  -> embedded_kernels.cpp が phaseshift_qwen35_gpu_mcu に blob を所有
  -> mcu_kernel_code_object(McuCompiledVariantKind)
       = blob 名 + stable extern "C" symbol + kernarg recipe
```

- kernel は mangled symbol ではなく stable `extern "C"` entrypoint で dispatch する。
- Host launcher と同じ entrypoint を使い、同一 kernel を比較する。
- `gpu_mcu_embedded_kernel(name)` は substrate namespace 上の name → blob lookup であり、
  blob の中身は Qwen3.5 の kernel inventory である。
- 生成フラグ・`--offload-arch`・code object 形式は
  [low_level.md](low_level.md) の `worker HSACO build` と CMake の
  `cmake/gpu_mcu/hsaco.cmake` が正本。
- production runtime は build directory の path を持たない（blob を内包するため）。

## production decode までのデータフロー

```text
ContinuousBatcher -> ScheduledBatch
  -> Executor::execute
  -> ensure_mcu_state()                    GpuMcu 要求時の最初の submit で作成
  -> decide_decode_backend()                 Host / GpuMcu
       decode_backend_execution_error()      GpuMcu + eligibility NG -> Status::unsupported
  -> batch staging                           token / request / DeviceBatchContext の H2D
  -> preflight_mcu_plan()
       compile_mcu_plan()                    Program -> McuCompiledPlan
                                             （unsupported kernel / variant はここで確定）
  -> execute_mcu_hybrid()
       mcu_plan_fingerprint() / plan cache   upload は fingerprint 不一致時のみ
       stage_mcu_commit_slots()              slot / binding を device へ用意
       configure_batch_plan / configure_execution / configure_commit_slots
       GpuMcuPersistentMcu::start()          batch ready epoch を公開して供給
  -> persistent loop（control WGP 常駐）
       admission -> plan 解釈 -> kernarg 生成 -> AQL stage/commit
       -> worker（worker CU）が実行 -> completion marker
  -> OutputRing -> Host が確定 token を consume
  -> verify / GDN restore は epilogue node で device 上に完結
```

## ヘッダーの分類

`include/phaseshift/runtime/gpu_mcu/` の全 header を、外部からの include 実績で分類した。
移動の可否は「どの public header が include しているか」で決め、
未使用だから private という推測では決めない。

| 分類 | header |
|---|---|
| 1. 外部 target が直接使う正式 contract | `aql.h`, `cu_partition.h`, `worker_image.h`, `fsm_worker.h`, `persistent_mcu.h`, `micro_fsm.h`, `completion.h`, `device_completion.h`, `retained_packet.h`, `embedded_kernels.h` |
| 2. substrate 内部 contract（他 header からのみ include） | `slot_table.h`, `slot_binding.h`, `slot_runtime.h`, `batch_planner.h`, `batch_binding.h`, `batch_commit.h`, `control_ring.h`, `request_ingress.h`, `output_ring.h`, `sequence_resource.h`, `kv_page_allocator.h`, `gdn_reset.h`, `wall_clock.h`, `execution_bridge.h` |
| 3. implementation detail | `src/phaseshift/runtime/gpu_mcu/detail/*.h`（`micro_fsm.hip` の内部構成。public include tree には置かない） |
| 4. Qwen3.5 backend 固有の contract を含む | `gdn_reset.h`（GDN state reset）、`embedded_kernels.h`（Qwen3.5 kernel blob）。`micro_fsm.h` は Qwen3.5 の kernarg recipe と invocation struct を含む |
| 5. test-only contract | **なし**。全 header が production か、それを include する public header から参照される |

この結果、public include tree から private へ移動すべき header は無い。
`plan_binder.h` は substrate の 1 translation unit と 3 テストからのみ参照されるが、
移動には substrate target への private include path 追加と include 形式の変更が要り、
便益がそれを上回らないため現状のままとする。

## kernel の配置規則

- Qwen3.5 の correctness kernel は `kernels/correctness/`（1 TU + `detail/*.inc`）に置く。
- optimized kernel は implementation family ごとに 1 ファイル。
- substrate 側で唯一 model semantics を持つ kernel（`gdn_reset.hip`）は
  `src/phaseshift/runtime/gpu_mcu/` に置く。FSM が recipe として参照するためで、
  Qwen3.5 kernels ツリーから外してはならない。

## tests の分類

GPU-MCU テストは `tests/unit/gpu_mcu/` の下に、保証している内容ごとに置く。

| ディレクトリ | 保証する内容 |
|---|---|
| `tests/unit/gpu_mcu/substrate/` | AQL packet / queue / stage-commit、ring、slot、ingress、completion、allocator、binding、worker code object、CU partition、kernarg region、LDS template |
| `tests/unit/gpu_mcu/controller/` | persistent MCU、micro FSM、continuous refill / turnover、append while running、autonomous loop、dispatch / feed、runtime lifecycle |
| `tests/unit/gpu_mcu/qwen35/` | Qwen3.5 単体 primitive の AQL 実行、kernel registry、plan compiler / cache、attention / GDN / verify の統合、decode backend policy |
| `tests/unit/gpu_mcu/acceptance/` | production decode、mixed batch 実行、plan binder、chain bridge、one layer / full body plan、end-to-end |

- 共有 fixture（`gpu_mcu_fsm_test_util.h` 等）は `tests/unit/gpu_mcu/` 直下に置く。
  カテゴリ間で共有されるため、いずれかのカテゴリには入れない。
- 取得方法:
  `ctest --test-dir <build> -L gpu_mcu -N`
- required acceptance の集合は `cmake/tests.cmake` と `cmake/gpu_mcu/tests.cmake` が正本。
  件数は build 構成で変わるためここには固定しない。
- GPU 不足による skip は開発者ローカルでは許容するが、required acceptance では失敗として扱う。

## R&D と production 仕様の境界

| 場所 | 内容 | 位置付け |
|---|---|---|
| `docs/developer/gpu_mcu/` | 現在の production contract | 正本 |
| `docs/rnd/gpu_mcu/` | 開発史、Gate 検証、採否判断、実験ログ | 歴史。現在の仕様ではない |
| `docs/perf/` | 現在の性能値と測定方法 | 正本 |
| `src/apps/bench/gpu_dispatch.hip` 等 | GPU-MCU の R&D 測定 subcommand | production runtime ではない |
| `tests/unit/gpu_mcu/` | 保証単位ごとのテスト | required acceptance の一部を含む |

`phaseshift-bench` の GPU-MCU subcommand は substrate を直接 link して機構を測るもので、
production decode 経路そのものを測定するものではない。
