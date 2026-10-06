# Runtime

## Execution path

```text
Qwen35TextConfig + weights
        ↓
lower_qwen35_to_primitives
        ↓
PrimitiveGraph
        ↓
build_program_set
        ↓
ProgramSet
        ↓
Executor (create_model_executor)
        ↓
program_executor (private: src/phaseshift/models/qwen35/runtime/)
        ↓
model_dispatch_correctness (ps::kernel, single launcher)
```

- `lower_qwen35_to_primitives(config, weights)` は
  `Qwen35LoweredPrimitives{graph, weights[], parameters[]}` を返す。
- `build_program_set` はfree function。bucketごとに `Program` を作り
  `ProgramSet` に格納する。
- `Program` は dispatch-only（`push()`）。host resolutionとGPU launchの
  分離は `program_executor.h`（private header）の
  `HostExecutionContext` / `HostResolvedValue` で行う。
- `HostExecutionContext::external_inputs` / `external_outputs` が指す配列は
  **ctx 自身の** `external_input_storage` / `external_output_storage` が所有する。
  構築関数のローカル配列へ pointer を持たせない。
  ctx は構築関数の返却後も呼び出し側で生き続けるため、ローカル配列だと
  stack が書き換わって不正な pointer が kernel に渡る。
- device側は `DeviceProgramView`（`runtime/program/device_program.h`）を
  kernel argumentとして渡す。
- sync semantics: opごとにstaging H2D（async）→ launch →
  `hipStreamSynchronize`（pool経由でnon-synchronous opはskip）。

## Decode backend

1つの `Executor` は Host 経路と GPU-MCU 経路を持ち、step ごとに選ぶ。
ownership は phase や backend で分けない。

```cpp
enum class DecodeBackend { Host = 0, GpuMcu = 1 };
```

- `Executor::config.backend` の既定は `DecodeBackend::Host`。
- `DecodeBackend` は要求を表す。`GpuMcu` は「GPU-MCU で実行する」要求であり、
  eligibility 判定で書き換えられない。
- `decide_decode_backend(DecodeBackendInputs)` が eligibility と
  `DecodeBackendReason` を返す。requested が `Host` なら reason は `BackendIsHost`。
- GPU-MCU が実行できない主な理由:
  `NotDecode` / `MultipleRequests` / `NotSingleToken` / `NotSingleRow` /
  `SpeculativeVerify` / `PrefillPresent` / `KvDtypeNotBf16` /
  `TensorParallel` / `StochasticSampling` /
  `ImatrixCollector` / `ValueTrace` / `TargetHiddenTaps` /
  `StreamWaitUnsupported` / `McuBodyRangeUnavailable`。
  `GpuMcu` 要求で成立しない場合は `decode_backend_execution_error()` が
  `Status::unsupported` を返し、Host 経路へは落ちない。
- `TensorParallel`（`tensor_parallel_configured`）と `StochasticSampling`
  （`stochastic_output_count`）は `persistent_ready` でも免除されない。
  GPU-MCU は stochastic sampling を argmax で代用しない（plan compile も
  `stochastic sampling has no mcu entrypoint` で fail-closed）。
- capability 判定は次の 2 段階で、いずれも batch の program dispatch より前に行う。
  1. `create_model_executor()` の config preflight（TP・hidden tap・KV dtype）と
     `submit_co_batch()` の `decide_decode_backend()`（execution feature）
  2. `preflight_mcu_plan()`（unsupported kernel / physical variant）
- `Executor::mcu_state`（`McuDecodeState`）は最初に `GpuMcu` が選ばれた時点で
  `ensure_mcu_state()` が作る。以降の step で再利用する。初期化・plan・実行が
  失敗した場合も Host へ fallback せず error を返す。
- GPU-MCU 経路の全体（plan compilation、persistent controller、slot / output）は
  [gpu_mcu/architecture.md](gpu_mcu/architecture.md) が正本。

## Public API

```cpp
ps::qwen35::create_model_executor(model, slot_pool, gdn_pool, kv_pool, arena, config)
ps::qwen35::executor_shutdown(executor)
ps::qwen35::submit_batch(executor, batch, stream)
ps::qwen35::complete_batch(executor, submission, stream)
ps::qwen35::execute_batch(executor, batch, stream)
```

## State ownership

| 要素 | 所有 | 説明 |
|---|---|---|
| `ScheduledBatch` | ContinuousBatcher / caller | 1 step分のdecode/prefill request集合（POD） |
| `ContinuousBatcher` | app | request queue / KV admission / Executor呼出。`submit` / `step` / `cancel_all` |
| `KVCapacityManager` | ContinuousBatcher | requestごとのpage reservation state |
| `KVBanker` | ContinuousBatcher | paged KVのadmission / release（pure algorithm、stateless） |
| `PagedKVPool` | app | paged KV GPU buffer（attention層 × head × page） |
| `GdnStatePool` | app | GDN linear attention state GPU buffer |
| `SequenceSlotPool` | app | sequence slot GPU buffer |
| `Executor` | app | ProgramSet / GPU scratch / submission state。RAII + 明示shutdown |

`ContinuousBatcher` は `Executor` / 各poolをreference保持。
destroy順序は app が決定的に管理（executor shutdown → pool → arena shutdown）。

## GpuArena

単一の連続virtual address range + bump allocator。物理device memoryは
VMM（virtual address reservation + 増分physical commit）でbackingする。

- `capacity()`：reserved virtual addressの上限（callerが要求した値）。
- `used()`：bump allocatorが払い出したlogical bytesのhigh-water mark。
- `committed()`：現在physical device memoryへmapされているbytes。

commitはchunk単位（commit quantum）で、`allocate_aligned()`の進みに応じて
前方へだけ進む。個別free / decommitは無い（bump allocatorはindividual
allocationのlifetimeを追跡しないため）。

## Scheduler

`schedule_requests(span<const RuntimeRequest*>, max_scheduled_tokens,
max_scheduled_requests, page_tokens, KVBankerState*)` はfree function
（`runtime/token_budget_scheduler.h`）。token budgetとKV capacityの制約下で
`SchedulePlan`（decode / prefill requestの順序付き選択）を返す。
`SchedulePlan` は `append_decode` / `append_prefill` / `finalize` で
`ScheduledBatch` を直接buildする。

## Cancellation

`ContinuousBatcher::cancel(uint64_t id)` は対象requestが `Queued` / `Active` のとき、
`RequestState::Finished` / `FinishReason::Cancelled` へ遷移し、割り当て済みの
sequence slot / GDN state / paged KV / capacity claimを解放する。`Finished` 済みの
requestへのcancelはno-op。

`Qwen35ComputeRuntime::cancel(uint64_t id)` は `ContinuousBatcher::cancel()` へ委譲する
だけの薄いAPIで、runtime layerに新しいcancel logicを持たない。

cancel granularityは `runtime.step()` の境界である。実行中のGPU kernelはpreemptせず、
device reset / `hipStreamDestroy` / signalをcancel目的に使用しない。cancel latencyは
最大で現在実行中の1 step。

`phaseshift-compute --serve-stdio` の service loopはGPU runtime ownerである単一thread上で、
`poll(2)` によりstdinを non-blockingに監視する。idle時はblocking poll、in-flight request中は
timeout 0 pollと1 stepを交互に行い、各stepの前後で `ping` / `cancel` / `shutdown` を
処理できる。stdout writerも同一threadで、mutexを持たない。

## Concurrency

`Qwen35RuntimeConfig::max_concurrent_requests`（compute CLI `--max-concurrent-requests`）は
最大in-flight request数であり、`SequenceSlotPool` / `GdnStatePool` /
`ExecutorConfig::max_scheduled_requests` / `ContinuousBatcher::max_scheduled_requests` に
同一値として伝播する。defaultは1で、single-shotとsingle-request parityを変えない。

論理concurrencyとphysical KV budgetは別契約である。

- `max_blocks_per_sequence = ceil_div(max_seq_len + 1, page_tokens) + 1`
- `physical_capacity_tokens = kv_cache_capacity_tokens == 0 ? max_seq_len + 1
  : kv_cache_capacity_tokens`
- `kv_pool_pages = ceil_div(physical_capacity_tokens, page_tokens) + 1`

`--kv-cache-capacity-tokens 0` は1 context分を意味する。KVを自動的にN倍しない。
physical capacityが不足する場合、KV BankerがActive requestのmax_claimを保証した上で
Queued requestをadmissionする。`max_concurrent_requests > 1` でも全requestが常にActiveに
なるとは限らない。

複数requestは同一`ContinuousBatcher`へ`submit`され、1回の`step()`で
`ScheduledBatch`としてco-batchされる。requestごとにHIP streamやExecutorを作らない。
`StepResult::num_requests` が1 stepでco-batchされたrequest数である。debug時は
`PHASESHIFT_BATCH_TRACE=1` でstderrへ `PHASESHIFT_BATCH_STEP requests=R tokens=T` を出す。

注意: 通常decodeのlinear kernelはbatch行数に応じてGEMV / GEMM configを選択する
（`linear_selector.cpp`）。このためco-batchされたrequestのgreedy tokenは、
single-request実行と一致しない場合がある（argmaxが僅差のとき）。これはcross-request
混入ではなく、kernel選択に依存する浮動小数の差である。batch行数に依らずbit-exactな
経路は `VerifyNumericMode::Exact`（speculative verify専用）に限られる。request間の
event demultiplexと `token* == done.generated_ids` はco-batchでも保証される。

## Request retirement

`ContinuousBatcher` は `std::list<RuntimeRequest>` と `unordered_map<uint64_t, iterator>` を
持ち、address安定とO(1) eraseを両立する。`finish_request()` は`Finished`遷移と
GPU state解放（sequence slot / GDN state / paged KV / capacity claim）のみを行い、
host-side request objectは解放しない。

`ContinuousBatcher::retire(uint64_t id)` は `Finished` requestのみを`requests_`と
`request_index_`から削除する。`Queued` / `Active`のretireは `invalid_state`、unknown IDは
`invalid_argument`。serverはterminal `done` を生成した後にretireする。これにより
`generated` をterminal JSONへ書いた後にhost memoryを解放できる。

`Qwen35ComputeRuntime::retire(uint64_t id)` は `ContinuousBatcher::retire()` への薄いdelegate。

## Multi-request terminal semantics

停止条件の共通意味論（優先順位・one-way latch・zero-work との分離）は
[stop_conditions.md](stop_conditions.md) を正本とする。

`--serve-stdio` の `ServeState` は `std::map<int64_t, ServeGeneration>` でin-flight requestを
保持し、request_idでeventをdemultiplexする。request間のevent順序は保証しないが、
1 request内では `token*` の後にterminal event（`done` または `error`）が必ず1個だけ出る。
`cancel` はrequest-specificで、対象requestのみを `cancelled` にする。直前にterminalへ
遷移したrequest IDはbounded history（256件）に保持し、遅延cancelをidempotent no-opとして
吸収する。shutdown / EOFでは全in-flight requestをcancelし、全terminalの後に
`{"event":"shutdown"}` を出す。

### DFlash2 speculative decode 経路

`--dflash2-model-dir` 指定時の serve は `ContinuousBatcher` を経由しない。
`ServeState.dflash_enabled` が true のとき `serve_step()` は `dflash2_spec_step()` を、
`serve_parse_generate()` は受付時に `dflash2_spec_prefill()` を呼ぶ。
`Qwen35ComputeRuntime::submit()` / `step()` / `request()` / `cancel()` / `retire()` は使わない。

- `ServeGeneration` が `generated` / `finish_reason` / `prompt_tokens` /
  `max_new_tokens` を保持する。`RuntimeRequest` を使わないため `internal_request_id` は未使用。
  `finish_reason` は `dflash2_spec_step()` が返す `bool finished` から EOS / 上限を判別して
  `FinishReason` へ写像する。
- `DFlashServeSession`（`PagedSequenceState` / `DFlash2ContextState` /
  `DFlash2SpecDecoder` / `DFlash2SpecTiming`）は request ごとに作成し、
  `serve_retire()` で decoder → context → sequence の順に解放する。
  `PagedSequenceState` は move assignment が delete であるため `unique_ptr` で保持する。
- `DFlashServeContext` が serve 間 `DFlash2Weights` と `DFlash2Executor` を保持する。
  `DFlash2Executor` は `const DFlash2Config*` / `const DFlash2Weights*` を指すため、
  両者は serve 中アドレスを安定させる。serve 終了時に `dflash2_executor_shutdown()` し、
  その後 `Qwen35ComputeRuntime::shutdown()` する。
- `max_concurrent_requests` は compute の指定値をそのまま使う（DFlash2 でも複数
  受付できる）。inflight は順に1つずつ round され、同時に in-flight な target
  submit は常に1つで、round ごとに完全 sync する。`temperature > 0` は
  request ごとに受け付ける。
- `--kv-cache-capacity-tokens` が未指定で concurrency > 1 のとき、server は
  `(concurrency + 1) × (max_seq_len + 1)` へ自動拡張する（DFlash は
  `ContinuousBatcher` を通らないため KV Banker の admission が効かない）。

詳細は [dflash2.md](dflash2.md) を参照。

## Prefix cache

GPU 実装は提供しない。protocol contract のみを保有する。詳細は
[prefix_cache.md](prefix_cache.md) を参照。
