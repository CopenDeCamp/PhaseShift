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
- device側は `DeviceProgramView`（`runtime/program/device_program.h`）を
  kernel argumentとして渡す。
- sync semantics: opごとにstaging H2D（async）→ launch →
  `hipStreamSynchronize`（pool経由でnon-synchronous opはskip）。

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
- `max_concurrent_requests` は 1。`temperature > 0` / `grammar` / `structural_tag` /
  `prefix_cache_checkpoint_position` を含む request は error event で拒否する
  （fail-closed、unconstrained へは fallback しない）。
- `prefix_cache_checkpoint_position` は prefix cache が有効なときだけ backend から送る。
  無効時は 0 である。

詳細は [dflash2.md](dflash2.md) を参照。

## Prefix cache

opt-in の GPU-resident prefix cache を持つ。`PrefixCache` は active とは別の KV pool と
GDN pool を専用に所有し、checkpoint を D2D で snapshot / restore する。active KV Banker・
sequence slot・GDN slot には影響しない。

request は任意の `prefix_cache_checkpoint_position` を指定できる。指定すると scheduler は
その位置を跨がないように prefill chunk を切り、chunk 実行後に `input_tokens[0:N]` の snapshot
を保存する。この場合 terminal checkpoint は保存しない。指定が無い request は従来どおり
terminal checkpoint を保存する。詳細は [prefix_cache.md](prefix_cache.md) を参照。

## Grammar constraints

GBNF grammarによるtoken制約をruntimeへ統合する。`token_constraint.h` /
`token_constraint.cpp` がXGrammar v0.2.5を薄く包み、XGrammar型をpublic headerへ露出しない。

- `TokenConstraintCompiler` はmodel lifetimeで1個。compiled grammarはcount / byte boundedな
  cacheで共有する。同一grammar bytesは再compileしない。
- `TokenConstraintState`（matcher）はrequestごとに1個。`RuntimeRequest.constraint` に保持し、
  retireで破棄する。constraint stateはserver lifetime中に蓄積しない。
- grammarは`generate` requestのoptional fieldとして受け取る。fieldが無ければunconstrained。
- compile failure / empty allowed set / matcher accept failureはrequest terminal errorであり、
  unconstrained fallbackしない（fail-closed）。
- constrained requestはMTP / speculative decodeを使わない。

maskは `ScheduledBatch.constraint_masks`（host, `[output_rows][ceil(vocab/32)]` uint32）と
`ScheduledRequest.token_constraint` でexecutorへ渡す。executorはconstrained rowが存在するstepだけ
maskをdevice bufferへ1回のbatched H2Dで転送し、`HostExecutionContext` / `DeviceBatchContext` の
`constraint_masks` / `constraint_mask_words` 経由でsampling kernelへ渡す。unconstrained batchは
mask生成もH2Dも行わない。
