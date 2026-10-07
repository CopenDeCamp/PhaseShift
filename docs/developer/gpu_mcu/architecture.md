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

## execution model

### controller ownership

`GpuMcuPersistentMcu::start()` 後の inference progression の所有者は
Persistent MCU のみである。Host scheduler は progression を所有しない。
Host は per-token / per-batch の forward を発射しない。

batch planning、binding、forward execution、commit、停止判定、
次 batch への遷移、idle 復帰はすべて GPU-MCU controller 内で完結する。

### Host が start 後に行ってよいこと

control:

- 新規 request の SUBMIT
- request の CANCEL
- explicit controller shutdown

observation:

- OutputRing からの token / terminal record の consume
- event ring の consume
- telemetry / counters の観測

observation は進行の prerequisite にしてはならない。
観測結果を次の batch 開始・次の token 生成・次の forward 発射の条件にしない。

### Host が start 後に行ってはいけないこと

production GPU-MCU backend では start 後に次を行ってはいけない。

- `compile_mcu_plan()`
- `preflight_mcu_plan()`
- `prepare_plan()`
- per-batch `configure_*`
- Host による `ready_epoch` increment
- `execute_batch()` による inference progression
- Host scheduler による次 batch 選択
- `batches_committed() == 1` を待って次 forward を発射すること
- batch ごとの `start()`
- batch ごとの `request_stop()`
- batch ごとの `wait_stopped()`

### ContinuousBatcher との境界

```text
ContinuousBatcher::step() -> schedule_requests() -> execute_batch()
```

は Host backend の execution model である。
GPU-MCU backend の execution driver として使用してはいけない。

GPU-MCU backend が既存 public API の `step()` を残す場合は、
`step()` は OutputRing drain / event drain / Host mirror 更新のみを行う
observation API とし、GPU inference を前進させてはいけない。

### request terminal と controller stop

```text
1 request terminal            != controller stop
all request terminal          != controller stop
runnable request == 0         -> idle
request_stop()                -> application / model runtime shutdown 専用
```

### static plan

Persistent MCU start 後に Host は compile / upload / switch を行わない。
必要な execution plan は controller start 前に準備する。

batch ごとの actual rows / request count / output count /
attention region rows / verification count / runtime zero-work は、
device-side binding / invocation patch / dynamic node binding で処理する。
実行中の batch 値を理由として Host が recompile してはいけない。

### zero-work との関係

compile-time batch geometry と runtime batch geometry が同じであることを
前提にしてはいけない。

長寿命 controller では start 後に batch geometry が何度も変化するため、
runtime zero-work は device-side binding の責務になる。
runtime skip mechanism（`McuDynamicNodeBinding::enabled` 等）を保存する。

`workgroup_count_x/y/z == 0` の「geometry override なし」という
sentinel semantics は変更しない。

## request payload lifetime

`GpuMcuRequestDescriptor` が device へ渡す handle の lifetime は次のとおりである。

| handle | 書き込み | 読み取り | 有効期間 |
| --- | --- | --- | --- |
| `prompt_tokens_handle` | claim 時に `GpuMcuSlotBinding` へ転送 | 毎 PREFILL bind（`gpu_mcu_bind_prefill_tokens`）。chunked prefill では `prefix_length + row_count <= prompt_length` の範囲で複数回 | request terminal |
| `sampling_params_handle` | claim 時に `GpuMcuSlotBinding` へ転送 | 毎 batch bind | request terminal |
| `stop_conditions_handle` | claim 時に `GpuMcuSlotBinding` へ転送 | 毎 token commit | model lifetime（単一インスタンスを共有） |

- SUBMIT 時の H2D upload は許可される Host operation である。
- SUBMIT command を control ring に書いた時点で、参照する payload は
  device-visible で安定していなければならない。
- prompt payload は claim された時点で解放してはならない。
  chunked prefill で request lifetime 中に再参照される。
- payload の確保と解放は per-token で `hipMalloc` / `hipFree` を行わない。
  固定容量 pool または長寿命 allocation を使う。
- model 共通の stop-condition table は model lifetime で共有する。

## 責務境界

### GPU-MCU の責務ディレクトリ

対象: `include/phaseshift/runtime/gpu_mcu/**` と `src/phaseshift/runtime/gpu_mcu/**`。
直下には責務ディレクトリのみを置く。雑然とした header / source を直下に置かない。

```text
GPU-MCU
│
├── Infrastructure
│   AQL / HSA / CU / completion
│
├── I/O
│   control ingress / output
│
├── Scheduling
│   slot / batch / sequence / KV resource
│
├── Binding
│   logical → physical invocation
│
├── Execution
│   Persistent MCU / FSM
│
├── Commit
│   lifecycle / token / terminal
│
└── Model Hooks
    temporary architectural debt
```

| ディレクトリ | 責務 | header |
|---|---|---|
| `infrastructure/` | AQL queue / packet publication / doorbell / kernarg region / CU-WGP partition / completion primitive / worker HSACO / wall clock。LLM・request・batch・token を知らない | `aql.h`, `retained_packet.h`, `completion.h`, `device_completion.h`, `cu_partition.h`, `wall_clock.h`, `worker_image.h`, `fsm_worker.h` |
| `io/` | Host ⇄ Persistent MCU の通信境界。「次に何を実行するか」は判断しない | `control_ring.h`, `request_ingress.h`, `output_ring.h` |
| `scheduling/` | 何を実行するか（what should run next）。AQL packet は発行しない | `slot_table.h`, `batch_planner.h`, `kv_page_allocator.h`, `sequence_resource.h` |
| `binding/` | logical work を physical execution へ変換する（runtime pointer・actual rows・shape・invocation index・kernarg source） | `slot_binding.h`, `batch_binding.h`, `plan_binder.h`, `plan_binding_contract.h` |
| `execution/` | GPU execution progression の owner。`src/` 側の `execution/detail/` は micro FSM の内部構成 | `persistent_mcu.h`, `micro_fsm.h`, `execution_bridge.h`, `fsm_contract.h`, `invocation_abi.h`, `kernarg_recipe.h` |
| `commit/` | kernel 実行完了後の状態更新（token commit・sequence advance・terminal decision・slot lifecycle・output publication） | `stop_conditions.h`, `slot_runtime.h`, `batch_commit.h` |
| `model_hooks/` | substrate に残るモデル固有処理の隔離先。**DO NOT ADD NEW MODEL HOOKS**。extension point ではなく既知の負債の隔離先であり、新規ファイル追加は `test_architecture_boundaries` が失敗させる | `gdn_reset.h` |

#### dependency matrix

依存は下位から上位へ。実際の責務に対して**禁止する依存**を明記するのが目的であり、
完全な layered architecture に適合させることではない。

| from \\ to | infrastructure | io | scheduling | binding | execution | commit | model_hooks |
|---|---|---|---|---|---|---|---|
| `infrastructure/` | 許容 | 禁止 | 禁止 | 禁止 | 禁止 | 禁止 | 禁止 |
| `io/` | 現状なし | 許容 | 許容 | 許容 | **禁止** | 現状なし | 現状なし |
| `scheduling/` | 現状なし | 現状なし | 許容 | 現状なし | **禁止** | 現状なし | 現状なし |
| `binding/` | 現状なし | 現状なし | 許容 | 許容 | **禁止** | 現状なし | 現状なし |
| `commit/` | 現状なし | 許容 | 許容 | 許容 | 現状なし | 許容 | 現状なし |
| `execution/` | 許容 | 許容 | 許容 | 許容 | 許容 | 許容 | 許容 |
| `model_hooks/` | 現状なし | 現状なし | 現状なし | 現状なし | 現状なし | 現状なし | 許容 |

- `execution/` はループの owner であり、上位すべてを知ってよい。
- `model_hooks/` は transitional な隔離先である。
- 「現状なし」は将来禁止を保証しないが、追加時はこの表を更新する。

Qwen35 参照は `infrastructure` `io` `scheduling` `binding` `commit` で禁止する。
これは `include/phaseshift/runtime/**` の再帰走査で担保しており、
`model_hooks/` を含む全ディレクトリに効く。

`test_architecture_boundaries` が検査する禁止依存:

- `infrastructure/` → 上位 6 ディレクトリ
- `scheduling/` → `execution/`
- `binding/` → `execution/`
- `io/` → `execution/`
- `model_hooks/` に allowlist (`gdn_reset.h` / `gdn_reset.hip`) 外のファイル追加

`binding/` → `execution/` はかつて存在した。
`plan_binder.h` が static plan と runtime binding の間の契約を
`execution/micro_fsm.h` から得ていたためである。
`binding/plan_binding_contract.h` へ分離して実体を無くし、禁止ルールで固定した。

#### contract の分層

`execution/` の header は FSM 自身と、それを使う契約に分かれている。

| header | 内容 | model 識別子 |
|---|---|---|
| `micro_fsm.h` | `GpuMcuFsmState` / `GpuMcuFsmRunContext` / `GpuMcuFsmConfig` / `GpuMcuFsm` と FSM public entry point | 容量定数にのみ |
| `fsm_contract.h` | `McuSupervisorState` / `McuFaultCode` / `McuDoorbellMode` / `McuLogEvent` / `McuLogRecord` / `McuDispatchRecord` / `McuPlanNode` / `McuKernelVariantDesc` / `McuDispatchTiming` と node・record・log の flag | **なし** |
| `invocation_abi.h` | `Mcu*Invocation` 24 構造体とその `static_assert` | 多数 |
| `kernarg_recipe.h` | `kMcuKernargRecipe*` の番号表（`None = 0` … `Count = 29`） | 多数 |
| `binding/plan_binding_contract.h` | `McuDynamicNodeBinding` / `McuInvocationPatchSource` / `McuInvocationPatch` | なし |

recipe ID と `static_assert` は ABI であり、値・型・member 順・padding を変えてはならない。

#### model-specific 識別子の所在

`Gdn` `Qwen` `RmsNorm` `Psq4` `Attention` `Embedding` `Rope` `KvAppend` `Argmax` を含む件数:

| location | 件数 |
|---|---|
| `execution/invocation_abi.h` | 208 |
| `src/execution/detail/micro_fsm_kernarg.h` | 111 |
| `execution/micro_fsm.h` | 45（容量定数） |
| `execution/kernarg_recipe.h` | 18 |
| `src/execution/detail/micro_fsm_host.h` | 18 |
| `model_hooks/gdn_reset.*` | 5 |
| `execution/fsm_contract.h` | **0** |

`invocation_abi.h` と `kernarg_recipe.h` は `runtime/gpu_mcu` 配下に置いたままとする。
generic FSM 実装が kernarg recipe を直接解釈しているため、
ここで `models/qwen35/` へ移すと `execution → models/qwen35` という、より悪い依存を作る。

#### コンポーネントの所属分類

責務ディレクトリの物理配置とは別に、各コンポーネントが
**どの層のものか**を判定した。本フェーズでは移動せず、分類のみを正本とする。

判定は次の4問に対する答えによる。

1. GPU-MCU という実行基盤に必須か？
2. LLM inference だから必要なのか？
3. Qwen35 だから必要なのか？
4. 他モデルを追加した場合そのまま再利用できるか？

| コンポーネント | 1. 基盤に必須 | 2. LLM だから必要 | 3. Qwen35 だから必要 | 4. 他モデルで再利用 | 分類 |
|---|---|---|---|---|---|
| `batch_planner` | いいえ | はい（batch / row / token の概念） | いいえ | 可 | Inference Runtime |
| `slot_table` | いいえ | はい（request lifecycle と terminal reason） | いいえ | 可 | Inference Runtime |
| `sequence_resource` | いいえ | はい（KV cache の block table と sequence） | いいえ | 可 | Inference Runtime |
| `kv_page_allocator` | いいえ | はい（KV page は LLM 由来の概念） | いいえ | 可 | Inference Runtime |
| `slot_runtime` | いいえ | はい（slot の terminal publish と release） | いいえ | 可 | Inference Runtime |
| `batch_commit` | いいえ | はい（token commit と sequence advance） | いいえ | 可 | Inference Runtime |
| `stop_conditions` | いいえ | いいえ（GPU-MCU の stop request に数え上げ先を与えるだけ） | いいえ | 可 | Inference Runtime |

**7 件すべてが Inference Runtime** である。GPU-MCU Core にも Qwen35 Backend にも該当しない。
Qwen35 固有の識別子を1件も含まないことを確認済みで、
他モデルを足してもそのまま再利用できる。

ディレクトリ単位では次の所属になる。

| 層 | 対象 |
|---|---|
| GPU-MCU Core | `infrastructure/`、`execution/micro_fsm.h`・`fsm_contract.h`、`io/control_ring.h`、`binding/` |
| Inference Runtime | `scheduling/`、`commit/`、`io/request_ingress.h`・`output_ring.h` |
| Qwen35 Backend | `model_hooks/`、Qwen35 runtime 全体 |
| transitional | `execution/invocation_abi.h`、`execution/kernarg_recipe.h`（generic FSM が recipe を直接解釈しているため runtime/gpu_mcu に残す） |

移動は本フェーズでは行わない。まず分類し、architecture decision を経てから動かす。

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
| embedded HSACO inventory | `gpu_mcu/embedded_kernels.h/.cpp` |
| controller 起動・static plan・request ingress・output の所有 | GPU-MCU production runtime owner（Qwen35 Executor/backend boundary。§既知の差異参照） |

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
- capability 判定は plan compile で行う。`create_model_executor()` が config 段
  （TP・hidden tap・KV dtype）を、static plan compile が unsupported kernel /
  physical variant を検出して `Status::unsupported` を返す。
  static plan compile は controller start 前に行い、
  request 到着後には compile / upload / switch を行わない。
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

### persistent controller の目標境界

本フェーズではコードを変更しない。目標として次を記載する。

```text
receive command/event
    ↓
select executable work
    ↓
bind execution data
    ↓
execute static plan
    ↓
observe completion
    ↓
publish completion/event
```

この中に以下を含めないことを目標とする。

- Qwen layer knowledge
- GDN semantics
- attention semantics
- tokenizer semantics

現状は `execution/invocation_abi.h` と `execution/kernarg_recipe.h` が
kernel field の意味を substrate 内に持つ。これが達成を妨げている。

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
  -> compile_mcu_plan(program, context, options, McuCompiledPlan)   （controller start 前）
  -> McuCompiledPlan { nodes, variants, invocation, epilogue, dynamic binding }
  -> GpuMcuPersistentMcu へ configure と upload                     （controller start 前）
  -> GpuMcuPersistentMcu::start()
  -> mcu_run_once が plan node を解釈して AQL dispatch
```

- compile と upload は controller start 前に一度だけ行う。
  start 後に Host が compile / upload / switch を行うことは禁止である。
- batch ごとの差は `DeviceBatchContext` の値、`McuInvocationPatch` の
  row-global overlay、`McuDynamicNodeBinding` で吸収する。
  static plan は最大 geometry を保持する
  （`gpu_mcu_bind_execution_plan()` の契約）。
- plan node は `McuPlanNode`（variant_id / next / kernarg_recipe / completion_slot 等）。
  interpreter は `variant_id` のみ解決し、`kernel_object` を持たない。
- verify acceptance は runtime transaction であり Program graph には入れない。
  `McuCompiledPlan::epilogue` として runtime が node を積む。
- plan を 1 つも compile できなければ compile の `Status` をそのまま返す。
  Host 経路へは戻らない。

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

初期化（controller start 前にすべて完了する）:

```text
Host
  -> model / executor resource 構築
  -> static MCU plan compile
  -> static plan upload
  -> request / control / output infrastructure 構築
  -> persistent MCU state を configure（一度だけ）
  -> GpuMcuPersistentMcu::start()（一度だけ）
```

運転:

```text
Host                                   Persistent MCU
  SUBMIT request      ----------------> control / input ring
                                            -> admission
                                            -> reconcile / resource reserve
                                            -> scheduler snapshot
                                            -> device batch planning
                                            -> device batch binding
                                            -> batch_ready_epoch 更新
                                            -> micro-FSM execution
                                            -> commit
                                            -> scheduler
                                            -> 次の runnable batch
                                            -> runnable == 0 -> idle / sleep
  CANCEL              ----------------> control ring
  output consume      <---------------- OutputRing（進行の prerequisite ではない）
  event consume       <---------------- event ring
  telemetry 観測      <---------------- counters

application shutdown
  -> request_stop()
  -> wait_stopped()
```

idle 中の次の SUBMIT は同一 controller を自力で復帰させる。
request terminal は controller stop を伴わない。

## 既知の差異

以下は現在の contract と実装の差である。

- production の Qwen35 GPU-MCU 経路は §execution model に未到達である。
  `configure_request_runtime()` を呼ばれず、control ring / ingress / output ring
  を構築しない。`DeviceBatchContext` と request descriptor を Host が構築して
  `configure_batch_plan()` へ渡し、`configure_commit_slots()`
  （`request_runtime_enabled == 0`）を使う。
  `ContinuousBatcher::step()` → `execute_batch()` → `enqueue_batch()` が
  `preflight_mcu_plan()` / `compile_mcu_plan()` / per-batch `configure_*` /
  `start()` / `batches_committed()` ポーリング / `request_stop()` /
  `wait_stopped()` を forward ごとに行う。
- `compile_mcu_plan()` は live batch 値（`ctx.actual_rows` /
  `ctx.actual_outputs` / `batch_context->requests` 等）を参照し、
  static plan として長寿命で使える構成になっていない。
- token / terminal の取得が OutputRing ではなく Host の sampled token buffer
  直読になっている。

## ヘッダーの分類

`include/phaseshift/runtime/gpu_mcu/` の全 header を、外部からの include 実績で分類した。
header は責務ディレクトリ配下に置く。移動の可否は「どの public header が include しているか」で決め、
未使用だから private という推測では決めない。

| 分類 | header |
|---|---|
| 1. 外部 target が直接使う正式 contract | `aql.h`, `cu_partition.h`, `worker_image.h`, `fsm_worker.h`, `persistent_mcu.h`, `micro_fsm.h`, `completion.h`, `device_completion.h`, `retained_packet.h` |
| 2. substrate 内部 contract（他 header からのみ include） | `slot_table.h`, `slot_binding.h`, `slot_runtime.h`, `batch_planner.h`, `batch_binding.h`, `batch_commit.h`, `control_ring.h`, `request_ingress.h`, `output_ring.h`, `sequence_resource.h`, `kv_page_allocator.h`, `gdn_reset.h`, `wall_clock.h`, `execution_bridge.h`, `plan_binding_contract.h`, `fsm_contract.h`, `invocation_abi.h`, `kernarg_recipe.h` |
| 3. implementation detail | `src/phaseshift/runtime/gpu_mcu/execution/detail/*.h`（`micro_fsm.hip` の内部構成。public include tree には置かない） |
| 4. Qwen3.5 backend 固有の contract を含む | `execution/invocation_abi.h` と `execution/kernarg_recipe.h`（kernel ABI と recipe 番号表）。`model_hooks/gdn_reset.h` は Qwen3.5 固有だが substrate の負債として隔離済み。`embedded_kernels.h` は Qwen3.5 runtime 側にある。`micro_fsm.h` と `fsm_contract.h` は該当なし |
| 5. test-only contract | **なし**。全 header が production か、それを include する public header から参照される |

この結果、public include tree から private へ移動すべき header は無い。
`plan_binder.h` は substrate の 1 translation unit と 3 テストからのみ参照されるが、
移動には substrate target への private include path 追加と include 形式の変更が要り、
便益がそれを上回らないため現状のままとする。

## kernel の配置規則

- Qwen3.5 の correctness kernel は `kernels/correctness/`（1 TU + `detail/*.inc`）に置く。
- optimized kernel は implementation family ごとに 1 ファイル。
- substrate 側で唯一 model semantics を持つ kernel（`gdn_reset.hip`）は
  `src/phaseshift/runtime/gpu_mcu/model_hooks/` に置く。FSM が recipe として参照するためで、
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
