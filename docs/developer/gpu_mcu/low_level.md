# GPU-MCU Low-Level Substrate

本ドキュメントは GPU-MCU 低レイヤー実行基盤の契約を記録する。
production architecture 全体（Qwen3.5 backend との接続、plan compilation、
kernel registry、データフロー）は [architecture.md](architecture.md) が正本である。

## 位置付け

```text
Host runtime -> Executor -> Qwen35 GPU-MCU backend    （接続済み）
    mcu_plan_compiler / mcu_kernel_registry / McuDecodeRuntime
GPU-MCU Low-Level Substrate
    CuPartition / AqlQueue / AqlCodeObject / WorkerImage
    MappedControlRing / PersistentMcu / Completion / SlotTable
```

- target `phaseshift_gpu_mcu` は source 上 Qwen3.5 を一切知らない
  （`test_architecture_boundaries` が検査する）。
- 依存は `phaseshift_core` / `phaseshift_gpu` / `hip::host` /
  `hsa-runtime64::hsa-runtime64` のみ。
- `phaseshift` aggregate の明示リストには追加しない。一方
  `phaseshift_qwen35_runtime` が PUBLIC で link するため、
  production binary は `phaseshift_gpu_mcu` と `hsa-runtime64` を実 link する。
  production decode が HSA 依存なのはこのためである。
- Qwen3.5 kernel の HSACO blob は substrate target に置かず、
  `phaseshift_qwen35_gpu_mcu` が所有する。
- link するものは Qwen3.5 GPU-MCU backend、低レイヤーテスト、
  `phaseshift-bench` の GPU-MCU subcommand。

## HSA / HIP agent mapping

- HIP device から `hipDeviceGetPCIBusId` で PCI BDF を取得する。
- HSA agent を iterate し、`gfx1201` かつ kernel dispatch 可能で、
  `HSA_AMD_AGENT_INFO_BDFID` が HIP の BDF と完全一致する agent のみを採用する。
- 一致しない場合は fail-closed（`no gfx1201 HSA agent matching the HIP device
  PCI BDF`）。「最初に見つかった gfx1201」は使用しない。
- 本機は gfx1201 が 4 基 + gfx1036 iGPU が 1 基見えるため、BDF 完全一致は
  TP=2 / multi-GPU の前提条件である。

## CU / WGP mask semantics

- `hipDeviceProp_t::multiProcessorCount` = 32（R9700）。
- HIP stream mask（`hipExtStreamCreateWithCUMask`）は **1 bit = 1 WGP**。
  full mask は 32 bit（`0xffffffff`）で、物理 64 CU 全体を覆う。
- `hipExtStreamGetCUMask` は要求 mask をそのまま返す（readback 一致）。
  単一 bit、非対称 bit も受理される。
- HSA queue mask（`hsa_amd_queue_cu_set_mask`）は **1 bit = 1 CU**。
  `HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT` = 64。
  even-index contiguous pairwise を要求する（`0x3` / `0xfffffffc` は受理、
  `0x1` / `0x2` / `0xfffffffe` は `HSA_STATUS_ERROR_INVALID_ARGUMENT`）。
- `hsa_amd_queue_cu_get_mask` は queue 制限ではなく agent-wide mask を返す。
  したがって readback equality を placement correctness の根拠にしない。
  placement は別途 mask unit 数スケーリングで観測する。

結論:

| 層 | unit | bit 数 | pairwise |
|---|---|---|---|
| HIP stream mask | WGP | 32 | 不要 |
| HSA queue mask | CU | 64 | 必須（even-index 2 連続） |

このため `GpuMcuCuPartition` は HIP 用 mask と HSA 用 CU mask を別々に持つ。

## Control WGP / Worker partition

- physical: 32 WGP / 64 CU
- control: 1 WGP = 2 physical CU
  - HIP mask: bit0
  - HSA CU mask: bit0 + bit1
- worker: 31 WGP = 62 physical CU
  - HIP mask: bit1..31
  - HSA CU mask: bit2..63

`GpuMcuCuPartition::create(device, control_wgps)` が両方を構築し、

- control / worker が重複しない
- 和集合が利用対象全体を覆う
- HSA CU mask が WGP pairwise を満たす
- HIP stream mask が readback 一致する

を検証する。control / worker stream は `hipExtStreamCreateWithCUMask` で作成する。

## AQL queue ownership

- `GpuMcuAqlQueue` が HSA queue / kernarg allocation を所有し、
  `shutdown()` で明示的に解放する。destructor は最終 fallback。
- cleanup は途中で 1 件失敗しても残りを解放し、最初の error を返す。
- `hsa_shut_down()` は呼ばない。HIP runtime が HSA runtime の lifetime を
  所有しており、共有 lifetime を壊さないため。
- HSA queue の write / read dispatch index は `hsa_queue_t` 先頭からの
  AMD 内部 offset（56 / 128）で取得する。offset は aql 層に閉じ込める。
- worker queue は worker CU mask で作成する。control WGP で producer が
  回っている間に worker を control WGP へ配置しないため。
- 追加で host producer 経路 `submit_host()`（`hsa_queue_add_write_index_scacq_screl`
  + doorbell signal store）を持つ。GPU producer 経路とは別の測定軸として使う。

## kernarg

- `GpuMcuAqlQueue::allocate_kernarg(slot_bytes, slot_count)` は
  `GpuMcuKernargRegion` を返す。
- 明示する項目: requested slot bytes / granule / actual allocation size /
  slot stride / slot count。
- slot address は `base + index * slot_stride` のみで求める。
  allocation 範囲外 slot は生成しない。
- launch metadata（gridDim / blockDim）は
  `align_up(kernarg_segment_size, 16)` に置く。slot stride は
  `align_up(kernarg_segment_size, 16) + 16` 以上を要求する。
- 全 slot は確保時に 0 で初期化する。

## worker HSACO build

- `src/phaseshift/runtime/gpu_mcu/worker_probe.hip` を `hipcc --genco` で
  standalone HSACO にし、`worker_image.cpp` に埋め込む。
- `hipcc --genco` の出力は固定 4096 byte の `__CLANG_OFFLOAD_BUNDLE__`
  header + bare ELF である。HSA reader が読むのは後半の ELF なので、
  `dd bs=4096 skip=1` で抽出する（CMake 内に局所化）。
- 生成物は build directory 下に置き、source tree へ書かない。
- `worker_image.h` は data pointer と byte 数のみを公開し、file path を
  runtime requirement にしない。
- probe worker は block/thread identity を output へ書き、supplied value を
  変換して書き、completion を更新するだけ。モデル計算は一切しない。

## packet publication ordering

```text
1. slot reserve
2. DW0 INVALID
3. body copy
4. required fence
5. DW0 publish (release)
6. doorbell
```

- DW0 を先に valid にしない。
- 64 byte layout は `hsa_kernel_dispatch_packet_t` と static_assert で照合する。
- magic offset（DW0 / body / kernarg / kernel_object）は aql 層に閉じ込める。
- doorbell 内部 ABI（`AmdSignalDoorbell`、hardware doorbell pointer と
  signal value の分岐）も aql 層の外へ漏らさない。

## completion / error

- `GpuMcuCompletion`（64 byte）は `sequence` / `state`
  (idle/running/complete/error) / `detail` / `aux` を持つ低レイヤー専用 primitive。
- device は system-scope release で publish し、host は mapped coherent memory を
  acquire load で観測する。
- `sequence` により stale completion を識別できる。
- モデル固有 status、KernelId、command ABI は入れない。
- `GpuMcuCompletionWord` が mapped memory を所有し、`shutdown()` で解放する。

## persistent MCU lifecycle

- `GpuMcuPersistentMcu` が mapped stop flag / heartbeat / started / iterations を
  所有し、control stream 上へ persistent kernel を launch する。
- kernel は control WGP 上で常駐し、stop request まで loop して
  `wall_clock64()` heartbeat と `__builtin_amdgcn_s_sleep(64)` backoff を行う。
  command semantics は持たない。
- `start()` は launch 後に started flag を bounded time で待つ。
- `request_stop()` は stop flag を release store し、`wait_stopped()` が
  started=0 を bounded time で待つ。
- `shutdown()` は stop request -> wait_stopped -> `hipStreamSynchronize(control)`
  の順。destructor は最終 fallback。

既知の注意点:

- persistent wave 稼働中に legacy/default stream 操作（`hipMemset` /
  `hipMemcpy` / `hipFree` 等）を行うと待ちが発生する。
  persistent 稼働中の GPU 操作は明示した control / worker stream 上で行い、
  shutdown 前に安易な device-wide sync を行わない。

同期 API の実測（FSM 生存中の挙動）:

| 呼び出し | 挙動 |
|---|---|
| `hipStreamQuery` | 返る（`hipErrorNotReady` / `hipSuccess`） |
| `hipStreamSynchronize(指定した stream)` | 返る（待つべき作業がある状態でも返る） |
| `hipEventSynchronize` | 完了済みなら返る。**未完了だと返らない** |
| `hipEventDestroy` | 返る（対象 event が完了している場合） |
| 同期 `hipMemcpy` / `hipMemcpyWithStream`（pageable dst） | **返らない** |

したがって:

- batch の完了待ちは `hipStreamSynchronize(stream)` で行う。`hipEventSynchronize` は
  event が未完了のときに待ちに入るため使わない。`hipEventDestroy` は完了後に呼ぶ。
- device→host の readback は submit 時に stream へ enqueue し、完了 event（または
  stream sync）で覆う。pageable な stack 変数へ同期コピーしてはならない。

`GpuMcuFsm::wait_stopped()` が poll する停止フラグのように、host-mapped memory を
host が直接読める経路は同期を伴わないので制約を受けない。

signal memory は host-readable であることが必須条件であり、
`stream_signal_alloc`（`include/phaseshift/runtime/stream_bridge.h`）は `hipHostMalloc` の mapped 系 2 種のみを試して
fail-closed する（plain `hipMalloc` へは落とさない）。host から読めない signal は
CPU 同期なしの受け渡しという設計自体を満たせない。

違反は build 時に弾く。`tools/check_mcu_sync.py` が mcu-aware な翻訳単位を走査し、
同期 `hipMemcpy`（`hipMemcpyDeviceToHost`）/ `hipMemcpyWithStream`
（`hipMemcpyDeviceToHost`）/ `hipDeviceSynchronize` を検出したら失敗する。
`cmake/targets.cmake` の `phaseshift_mcu_sync_check` target として
`phaseshift_gpu_mcu` と `phaseshift_qwen35_runtime` の依存に入っている。
contract を文書で守るのではなく、compile 前に落とす。

`hipEventSynchronize` の未完了待ちも同じ性質の hazard だが、文字列だけでは
「完了済みか」を判別できないため lint の対象外とする。batch 完了待ちは
`complete_batch` の `hipStreamSynchronize(stream)` で担保する。

## AQL stage / commit split

- `gpu_mcu_aql_stage_packet` / `gpu_mcu_aql_stage_retained_packet` reserve a slot,
  write DW0 INVALID, copy the body, and patch only `kernarg_address`.
- `gpu_mcu_aql_commit_packet` fences, publishes DW0, updates write index, and rings
  the doorbell. stage だけでは packet は CP に visible にならない。
- 既存の `gpu_mcu_aql_publish_streaming` は互換のため残し、MCU は stage / commit を使う。

## retained packet template

- `GpuMcuRetainedPacket`（64 byte、publish DW0 + 60 byte body）が LDS 常駐形。
- `make_retained_packet` が host 側 `GpuAqlPacketTemplate` から変換する。
- offset は `kAqlOff*` と static_assert で照合する。
- dispatch ごとに patch する field は `kernarg_address` のみ。

## device-local completion

- `GpuMcuDeviceCompletion`（64 byte）を worker が publish し、MCU が observe する。
  scope は device（agent）。hot path では host から読まない。
- `generation` が dispatch を識別し、`state` が running / complete / error を持つ。
- host-visible な `GpuMcuCompletion` は qualification / debug 用として残す。

## micro FSM

- supervisor state: boot / idle / running / fault / stopping。
- plan node ABI: `McuPlanNode`（variant_id / next / kernarg_recipe / completion_slot /
  value / element_count / input_slot / work / flags）。flags は dispatch / wait / end。
- kernel variant ABI: `McuKernelVariantDesc` が kernarg recipe・geometry・retained packet
  を持つ。interpreter は `variant_id` のみを解決し、kernel_object を hardcode しない。
- prepared dispatch は前 dispatch の completion 観測前に次 dispatch を stage する。
- kernarg は事前確保した ring（`dispatch_seq % kernarg_slot_count`）へ MCU が書く。

## kernarg generation and fence scope

- device が書いた kernarg を AQL dispatch された worker が読むのは
  producer -> consumer 依存である。
- packet は **Agent（または System）acquire / release** を持つこと。
  `kMcuAqlFastPolicy`（NONE/NONE）は pre-staged kernarg 専用であり、
  device 生成 kernarg に使うと worker が古い値を読む。
- host からの device memory 観測は device 内部の可視性の根拠にならない。

## continuous AQL queue feeder

- MCU は static dependency の dispatch を completion ごとに待たない。
  plan は `dispatch x N` + 末尾 `wait` + `end` で表現でき、先行 enqueue できる。
- 各 packet は barrier=1 / Agent acquire / Agent release を持つ。CP が先行 packet の
  complete を保証するため、先行 enqueue しても dispatch 順序と依存は保たれる。
- `queue_ahead_depth` は MCU が先行 publish する最大 packet 数。0 は ring size による
  bound。backpressure は `read_index` で判定し、queue full は fault ではなく正常な
  待ちとして扱う。先行 enqueue は unread slot を overwrite しない。
- doorbell:
  - `per_packet`: 各 packet で鳴らす。
  - `coalesce`: `doorbell_batch` 個ごと、および wait / plan end で tail を鳴らす。
    tail を鳴らす前に batch 内の全 DW0 を valid publish する。
  - `first_only`: plan の最初の packet のみ鳴らす探索用。production contract ではない。
- `issue_wait_running` は plan の 2 番目の dispatch を publish する前に先頭 dispatch の
  running を bounded に観測する。append-while-running の検証用で、通常は 0。
- completion slot を世代で使い回す場合、worker の publish は
  `state=idle -> generation -> state` の順で行う。generation を先に書くと、直前の
  complete と組み合わさって偽 complete に見える窓ができる。
- `McuDispatchRecord` と queue counter（append-ahead / empty / full / refill /
  doorbell / max_ahead / wrap）は debug 時のみ有効にする。hot path では host へ
  毎 dispatch publish しない。

## real primitive dispatch

- production kernel は **stable `extern "C"` entrypoint** で dispatch する。
  compiler の mangled symbol や anonymous template symbol に依存しない。
- Host launcher も同じ entrypoint を launch する。同一 kernel を比較する。
- variant は **dense id**（0 = completion marker、1 以降 = 実 kernel）。
  MCU は `variant_id` のみを識別し、`kernel_object` の switch を持たない。
- 実 kernel の runtime 値は plan node に埋めず、invocation descriptor table に置く。

| recipe | invocation struct | production args |
|---|---|---|
| `kMcuKernargRecipeProbe` | (probe worker args) | `GpuMcuFsmWorkerArgs` |
| `kMcuKernargRecipeRmsNormBf16PfOnePlus` | `McuRmsNormInvocation` | `RmsNormBf16PfOnePlusArgs` |
| `kMcuKernargRecipeActivationQuantizeA8` | `McuActivationQuantizeInvocation` | `ActivationQuantizeA8Args` |
| `kMcuKernargRecipePsq4Decode1Bf16U16/U8` | `McuPsq4Decode1Invocation` | `Psq4Decode1Bf16Args` |
| `kMcuKernargRecipePsq8Decode1Bf16U8` | `McuPsq4Decode1Invocation` | `Psq8Decode1Bf16Args` |
| `kMcuKernargRecipeEmbeddingBf16` | `McuEmbeddingBf16Invocation` | `EmbeddingBf16Args` |
| `kMcuKernargRecipeOutputGatherBf16` | `McuOutputGatherBf16Invocation` | `OutputGatherBf16McuArgs` |
| `kMcuKernargRecipeVerifyAcceptBatch` | `McuVerifyAcceptBatchInvocation` | `VerifyAcceptBatchArgs` |
| `kMcuKernargRecipeGdnSpecRestoreFromCounts` | `McuGdnSpecRestoreFromCountsInvocation` | `GdnSpecRestoreFromCountsArgs` |

- invocation struct は production args と同じレイアウト・サイズ（static_assert 対応）。
  `kernarg_segment_size` は必ず Code Object metadata の値を使い、`sizeof(args)` を
  allocation size に使わない。`build_aql_hidden_args` と `build_aql_launch_metadata` は維持する。

## embedding prefix

- token 列を受け取る BF16 embedding（optimized selector が効く hidden 2560 / 5120）は
  MCU region の**先頭 node** として compile する。Host からの embedding launch と
  embedding 直後の Host/GPU 同期境界を置かない。
- token 入力は graph 上 `I32`（`ValueDType::I32`）を contract とする。`EmbeddingBf16Args`
  は `table` / `token_ids` / `output` / `error_word` / `rows` / `vocab_size` /
  `hidden_size` / `output_row_stride` を持ち、`McuEmbeddingBf16Invocation` と
  `sizeof` / `alignof` / 全 `offsetof` を static_assert で固定する。
- selector / weight encoding / shape / pointer 解決は Host resolver
  （`resolve_embedding_physical`）と共有する。Host と MCU で selector 条件を別実装しない。
- invocation の `token_ids` は compile 時に resolve した安定 device pointer を既定値とし、
  kernarg 生成時に `state->batch_input != 0` なら device batch context の token buffer で
  上書きする。HostStream と external persistent の両経路に対応する。
- `rows` / `grid.x` は **actual launch rows** を使い、bucket max へ丸めない。
  stable kernel 側に `row >= args.rows` guard を残す。
- `EmbeddingStorage::Psq8` は MCU 対象外。BF16 以外・selector 非対象 shape は
  Host prefix に残す。この判定は executor の `ProgramMcuRange::region_begin` で行い、
  `region_begin == 0` のとき Host prefix launch は発行しない。

## lm-head plan extension

- `OUTPUT_GATHER`（`selected_hidden`）は MCU entrypoint を持ち、`output_rows` を読んで
  input の行を output 行へ gather する。grid は `(ceil(features / 256), num_outputs)`。
  `McuOutputGatherBf16Invocation` は `OutputGatherBf16McuArgs` と ABI を共有する。
- executor の MCU plan は `[0, dispatch_count)` を優先して compile し、成功すれば Host
  prefix / suffix を発行しない。compile できない dispatch がある場合のみ
  `[region_begin, body_end)` へ fallback する。`McuDecodeState::full_plan` が選択結果を持つ。
- lm-head の `LINEAR_BF16` は `ExactRows` の rows 1..16 を MCU catalog として持つ。
  `output_rows` domain の node には rows 1..16 の variant を登録し、
  `McuDynamicNodeBinding::variant_override` を `Bf16ExactRowsVariant` patch が loop ごとの
  `num_outputs` で上書きする。16 を超える output は invalid variant として fail-closed。

## runtime execution epilogue

- verify acceptance は model computation ではなく runtime transaction なので、
  `lower_to_primitives()` の Program graph には入れない。Program external I/O と fingerprint は
  不変のまま、`McuCompiledPlan::epilogue`（`McuRuntimeEpilogue`）として runtime が node を積む。
- node 順は model dispatch → sampling → `VERIFY_ACCEPT` → `GDN_SPEC_RESTORE` →
  completion marker。verify request が無い loop では各 kernel が non-verify request を no-op に
  するため、epilogue の有無で plan 形状は変わらない。
- `VERIFY_ACCEPT`（`phaseshift_gpu_mcu_verify_accept_batch`）は `DeviceBatchContext` の
  request 列を走査し、candidate を `context->token_ids + descriptor.row_begin`、
  candidate count を `descriptor.row_count`、sampled token row を
  `context->output_rows[descriptor.output_index + i]` から解決し、leading equal prefix を
  committed count とする。`execution_class != SPEC_VERIFY` の request は 0。
- committed count の出力先は runtime-owned の `McuDecodeState::verify_counts`
  （`configure_execution()` の `execution_verify_counts` と同一 buffer、Host readback なし）。
- `GDN_SPEC_RESTORE`（`phaseshift_gpu_mcu_gdn_spec_restore_from_counts`）は
  `committed_counts[request]` を history row、`context->row_sequence_slots[descriptor.row_begin]`
  を sequence slot とし、既存 Host の `restore_gdn_spec_history()` と同じ semantics で
  GDN state を accepted boundary へ戻す。

## mixed GDN history capture

- history capture の可否は batch-global な `ExecutionRole` ではなく
  `DeviceRequestDescriptor::execution_class` で決める。conv1d kernel は
  `row - descriptor.row_begin` を verify-local row とし、`capture_verify_only != 0` では
  `execution_class == SPEC_VERIFY` の row だけを history へ store する。
  `capture_verify_only == 0` は従来の `row < capture_rows`（dflash 経路）。
- `capture_verify_only` は conv1d / recurrence の args と invocation に `uint32_t` として
  入り、`VerifyRequests` patch が loop ごとに書く。`GdnConv1dAqlArgs` は 120 byte、
  `GdnRecurrenceArgs` は 192 byte のまま ABI を固定する。

## static region completion marker

- 実 primitive は MCU completion（`GpuMcuDeviceCompletion*` / `generation`）を書かない。
  production kernel ABI を MCU 専用に汚さないため。grid が複数 block のとき block 0 だけでは
  dispatch 全体完了にならないことも理由。
- static region の末尾に小さい **completion marker**（probe variant）を置き、
  MCU はその marker の completion だけを待つ。region 内部の各 primitive は poll しない。
- marker が実行開始する時点で、同一 queue 上の先行 real primitive は完了済みである
  （barrier=1）。

## kernarg lifetime

- static region の途中では kernarg slot を再利用しない。
  `region_seq` を plan 先頭で 0 にし、dispatch ごとに加算。`kernarg_slot_count` を超えると
  `kernarg_region_exhausted` で fail-closed。
- region 末尾の marker が完了した後、次の plan で slot を再利用する。

## host handoff boundary

- GPU 内部の primitive 間 dependency は Agent scope のままとする。
- 実 primitive variant: barrier=1 / Agent / Agent。
- final completion marker variant のみ System acquire/release。
  これは hot path を System へ戻すことでなく、static region 終了時の host 観測境界のみ。
- 3 primitive の間、MCU WAIT は入れない。region 末尾の marker だけを待つ。

## control ring transport

- mapped coherent host memory 上に 64 byte slot の SPSC ring を 2 本
  （host->device、device->host）持つ。
- slot ABI は `sequence`（transport 所有の publication sequence）+
  56 byte opaque payload のみ。command ABI は payload 上に定義する。
  request / cancel / epoch / FSM 等は ring へ埋め込まない。
- Vyukov-style bounded sequence:
  consumer は `sequence == position + 1` で pop し `position + capacity` を書く。
  producer は `sequence == position` で push し `position + 1` を書く。
- host は `std::atomic_ref` の acquire/release、device は
  `__scoped_atomic_load_n` / `__scoped_atomic_store_n`（system scope）を使う。
- device system-scope atomic が使えない場合は volatile へ fallback せず
  fail-closed（compile 不能なら build failure）。
- host native atomic 非対応も fail-closed
  （`hipDeviceAttributeHostNativeAtomicSupported`）。

## known limitations

- dependent dispatch（kernel B が kernel A の生成データを読む）は device 生成 kernarg を
  Agent acquire / release と barrier=1 で並べることで成立する。NONE/NONE policy は
  「same agent / pre-staged kernarg / packet body が publish 前に完成 / single GPU
  producer」に限定する。
- `EXT_KERNEL_DISPATCH` は header に存在するが、本環境の ROCr runtime は
  `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` で拒否する
  （`UNAVAILABLE_AT_RUNTIME`）。
- `hsa_amd_queue_cu_get_mask` の readback は queue 制限を返さないため、
  placement correctness は mask unit 数スケーリングで補助観測している。

## Persistent controller

persistent MCU が唯一の controller。Host は request を渡し、確定 token を非同期に
consume するだけで、per-token / per-batch の進行に関与しない。

- `mcu_run_once(GpuMcuFsmState*, GpuMcuRetainedPacket*) -> bool` は plan を 1 回
  完走する execution primitive であり、controller ではない。
- persistent loop は `mcu_run_once` と同じ翻訳単位（`micro_fsm.hip`）に置く。本ビルドは
  relocatable device code を使わないため、device 関数は翻訳単位を跨げない。
- `gpu_mcu_fsm_kernel` → `mcu_control_loop` → `mcu_run_once` は Host 駆動の standalone
  経路として維持する。production ではこの kernel を起動しない。
- `configure_execution(fsm, retained_count, batch_ready_epoch, sampled_tokens,
  sampled_capacity, verify_counts, verify_capacity)` は start 前のみ有効。
- `configure_commit_slots(slots, max_slots, bindings, binding_max_slots)` は commit が読む
  slot / binding table だけを設定し、request runtime と batch planner を enable しない。
  `request_runtime_enabled == 0` の間は `gpu_mcu_scheduler_boundary()` が早期 return するため、
  呼び出し側が構築した `DeviceBatchContext` は再構築されない。executor 経路がこれを使う。
- `batch_ready_epoch` が進んだときだけ dispatch する。`actual_rows == 0` は dispatch
  しない。`execution_epoch` の wrap は fail-closed で継続しない。
- commit は completion 成功時のみ。失敗 batch は参加 slot だけを `error` terminal に
  し、slot progress を進めない。stale handle へは commit しない。
- commit 後に `scheduler_dirty` を立て、次 loop が snapshot / plan / binding を
  再構築する。これが自走の条件。
- output は `OutputRing`（`hipHostMalloc(mapped|coherent)`、256 × 64 byte slot、
  sequence SPSC）へ device が push し、Host が `try_pop` する。ring full は該当 slot
  のみ `output_blocked` にし、global stop を作らない。
- boundary の順序は固定: output flush → terminal publish → release → admission →
  control ingest → reconcile → resource reserve → snapshot → plan → binding。
- `configure_control_drain(max_commands_per_boundary)`（start 前のみ、既定 1）が
  1 boundary に ingest する command 数の上限。
- **completion を書くのは probe（supervisor probe image）だけ**。real primitive は AQL
  packet の `completion_signal` を立てない（`make_kernel_variant` が 0 を入れ、FSM は
  kernarg address しか patch しない）。したがって wait node で completion を待つ plan は
  real primitive を **probe で挟む**。
- `plans_started` / `dispatches_committed` は plan 末尾または定期 publish で初めて host から
  見える。**0 は「dispatch していない」ことを意味しない**。

## per-loop invocation patch

- static plan は bucket 最大の geometry で compile する。loop ごとに変わる row 数は
  `McuInvocationPatch` の overlay で device 側の dispatch 直前に書く。Host は再 compile /
  再 upload しない。
- `McuInvocationPatchSource` は `ActualRows` / `NumRequests` / `NumOutputs` /
  `AttentionRegionRows` / `VerifyRequests` / `Bf16ExactRowsVariant`。`param` は source 固有で、
  `AttentionRegionRows` は低 16 bit = region の `row_begin`、高 16 bit = region の
  compile 時 `rows` を表し、batch context の request 列のうち region の row 範囲と重なる
  行数を返す。`VerifyRequests` は `context->num_verify_requests != 0` を 1 / 0 で返し、
  GDN conv1d / recurrence invocation の `capture_verify_only` を loop ごとに決める。
  `Bf16ExactRowsVariant` は `param`（rows 1 の variant id）に `num_outputs - 1` を足した
  variant id を返し、16 超では `0xFFFFFFFF` を返して fail-closed にする。
- attention region の grid は compile 時のまま最大形状で dispatch し、kernel 側の
  `row >= a.rows` guard が余分な block を return させる。Host 経路では `rows` が実 row 数
  なので guard は no-op。

## terminal / release contract

terminal slot は release 後に recycle されるため、**terminal は output ring の record で
観測する**。停止後の slot readback は観測手段にならない。

- `GpuMcuOutputRecord.flags` が record kind を持つ。token record は kind 0、terminal flag
  は terminal record だけが立てる。token を出さない終了（cancel / pre token error /
  `max_new_tokens == 0`）でも request あたり丁度 1 個の terminal record が出る。
- `GpuMcuSlotRuntimeState`（64 byte、slot id で index）が、固定 256 byte の slot state に
  入らない lifecycle bookkeeping を持つ（terminal publish 状態、release pending、
  resource blocked、scheduler epoch）。slot generation に自己 bind し、recycle された
  slot は claim path に触れずに初期化される。
- release eligibility は `terminal_reason != none`、`pending_count == 0`、
  `terminal_published != 0`、`release_pending != 0` を要求する。ring full で terminal を
  push できない slot は pending のまま release されない。
- resource を管理している場合、release で KV / GDN handle を解放してよく、tree / prefix
  handle は常に拒否する。管理していない場合、KV / GDN handle が残っていれば release
  しない。release は `gpu_mcu_binding_detach` で binding を切り離す。
- cancel は current transaction を途中で破棄しない。次 loop から除外し、cancelled
  terminal を出す。

## admission backlog

slot が満杯の submit を `no_idle_slot` で拒否せず、device 上に queue する。

- `GpuMcuPendingAdmission` は request id と descriptor handle / generation の組を持つ固定
  容量 FIFO。ingress は ready と claimed の間に queued 状態を持つ。
- `gpu_mcu_claim_entry` を直接 submit と queued descriptor の admit が共有する。
  admission は release pass の直後に backlog 先頭を 1 件だけ admit する。
- backlog 満杯は `admission_backlog_full` で reject する（黙って捨てない）。
- cancel は admission 前でも効く。`apply_*` の signature は増やさず、戻り値の
  `slot_exhausted` で deferral を伝える。

## device resource ownership

request の resource lifecycle は device（persistent MCU）が所有する。Host は resource を
確保・解放しない。

- `GpuMcuSlotState` の `kv_sequence_handle` / `gdn_state_handle` は **opaque な 64 bit
  handle** であり続ける。slot は handle を index として解釈しない。
- handle を解釈するのは sequence resource manager だけ。
  `handle = (resource_generation << 32) | (resource_index + 1)`、0 は「resource なし」。
- **resource generation は request slot generation と別管理**。release で resource
  generation が進むため、slot generation が動く前に古い handle が無効化される。
- `gpu_mcu_resource_resolve` は entry が active かつ generation が一致する場合のみ受理
  する。physical な sequence / GDN slot は現状 resource index と 1:1 で、Radix / prefix
  共有が入っても `GpuMcuSlotState` の ABI を変えずに分離できる。
- CLAIMED で resource を確保して handle を slot へ書く。release は **page を返してから
  resource を返す**。
- KV page の所有者は tag（loop 統合以降は resource handle）であり、slot ではない。
  stale / duplicate な free は「拒否分岐」ではなく tag 不一致で 1 枚も返さない。
  不変条件は `free_count + owned == capacity`。
- page 確保は incremental で、足りない分だけ取る。
- boundary の reserve pass が、各 live slot の次の step に必要な page を確保する。
  失敗した slot だけを `resource_blocked` にし、他 slot は継続する。`resource_blocked`
  の slot は snapshot から除外され、確保できるようになった loop で解除される。
  `kv_pages.capacity == 0`（未設定）なら reserve pass は無効。
- telemetry（`GpuMcuExecutionTelemetry`）に `resources_allocated` / `resources_released` /
  `kv_pages_reserved` / `kv_pages_released` / `resource_blocked_events` /
  `resource_unblocked_events` を持つ。resource lifecycle の assertion はこれらが 0 で
  ないことを確認して初めて検証になる。

### model の KV addressing の接続

- MCU は KV page を自前で確保するため、model の block table と page size を MCU に渡す。
  接続は `bind_mcu_kv_addressing(persistent, block_table, kv_bf16, kv_fp8)`
  （`src/phaseshift/models/qwen35/runtime/mcu_kv_binding.h`）で行い、persistent MCU を
  所有する session が `start()` の前に呼ぶ。
- 物理 sequence slot は model slot と 1:1 なので、resolve 結果がそのまま block table の
  row になる。KV Append / Paged Attention kernel は ResourceManager を知らない。
- `configure_kv_blocks` の直接呼び出しは整合確認用で、production 経路はこの関数を経由する。

### GDN state reset

- 新規 sequence の GDN state（conv history の bf16、recurrent の f32）は **MCU が dispatch
  する kernel** で zero 化する。request ごとの Host `hipMemset` は禁止。
- `GpuMcuGdnResetInvocation`（`gdn_reset.h`、48 byte）が conv base / recurrent base /
  batch context / slot stride / slot count を持つ。conv は bf16 を `uint16_t*` として見る。
- 発行契機は `requests[r].prefix_length == 0`。conv / recurrence kernel と同じ request
  descriptor を読み、新 sequence を開く request だけが該当する。scheduler は 1 sequence を
  1 batch にしか入れないため **毎 loop dispatch しても冪等**で、decode だけの loop では
  no-op になる。
- plan builder（`compile_mcu_plan`）が GDN state を触る range の先頭に 1 node 出す。
  先頭に BF16 embedding がある場合は embedding node の後ろに置く。
  対象 range に `STATEFUL_CAUSAL_CONV1D` / `GDN_RECURRENCE` が無ければ出さない。
  node は plan に焼き込まれるので、batch ごとの Host 判断を挟まない。
- 対象は layer あたり MB 規模なので、persistent loop 自身の thread では zero 化しない。

## GpuMcuSlotTable

MCU continuous batching の永続 request state。CPU runtime の
`RuntimeRequest` / `SequenceSlotPool` とは**別物**で、共有もしない。

- `GpuMcuSlotState` は 256 byte / align 64 の固定 ABI
  (`include/phaseshift/runtime/gpu_mcu/slot_table.h`)。`phase`（idle / prefill /
  decode / verify）と `terminal_reason`、`cancel_requested`、`output_blocked` は
  直交しており、phase へ混ぜない。
- slot index と `handle.slot` は一致する。`generation` は 0 を恒久的に invalid とし、
  **release で**進める（`UINT32_MAX` からの前進は拒否する）。request が終わった瞬間に
  古い handle が stale になる。claim は slot が既に持つ generation を使い、初回 claim
  だけ invalid 値から seed する。
- active request の判定は handle 一致だけでは行わない。
  `gpu_mcu_slot_matches_handle()`（phase != idle かつ handle 一致）を使う。
  release 済み slot に遅れて届いた cancel を active と誤認しないため。
- `verify_candidate_count` / `verify_committed_count` / `pending_count` は 32 上限を
  ABI contract とする。KV / GDN / tree / prefix の各 handle と
  `verify_candidate_ref` は opaque で、Radix 等の内部 ABI を焼き込まない。
- `gpu_mcu_slot_release()` は pending output が残っている間は拒否する。
- `max_slots` は create 時に固定し、実行中に resize しない。

## CPU / MCU parity principle

- CPU runtime は **Host-driven な reference / backend** として維持する。
- GPU MCU runtime は **autonomous continuous-batching backend** であり、
  実装方式は異なる。
- sampling / chunked prefill / stop conditions /
  request cancellation / KV・生成 token の正しさ など、固定した CPU execution model
  でも実現できる機能は原則 feature parity を維持する。**MCU 実装を理由に CPU 機能を
  削除・簡略化・置換しない。**
- device-side autonomous scheduling や Host-free な per-loop progress のような
  MCU 固有の execution property を CPU backend へ要求しない。

## Persistent MCU と他 work の排他

- persistent kernel は `create()` に渡された stream を**占有し続ける**。Host はその
  stream へ他の work を enqueue してはならない（常駐 kernel の後ろに並び、永久に
  実行されない）。補助 kernel やテスト用の処理は別 stream に置く。
- persistent kernel 稼働中に**同期 D2H コピー**（`hipMemcpy`）や null stream の
  同期操作を行うと返らない。slot table 等 device-only buffer を test で読む場合は、
  `request_stop()` + `wait_stopped()` で MCU を停止してから readback する。
- どちらも 9 分規模のハングとして現れた。この 2 点は stream 契約であり、
  実装の見直しではなく配置の見直しで解消する。
