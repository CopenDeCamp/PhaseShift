> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# Production Decode Executor Integration / Timing Truth Gate（進行中）

> Status: R&D record。D0-D11.3-4 は実装・検証済み。D12 の阻害要因（production の
> `complete_batch` が同期 D2H readback で待ちに入っていた）は §12.4 で修正し、
> 再発防止の build 時静的ゲートを §12.5 に入れた。D13-D17 は未着手。
> production の default backend は `Host` のまま。

## 1. 基点

| 項目 | 値 |
|---|---|
| baseline | `5ea62db4`（Full Transformer Serial Static Region Gate 完了地点） |
| branch | `poc/gpu-mcu-program-plan-one-layer` |
| worktree | `.worktrees/gpu-mcu-program-plan-one-layer` |
| 参照 | [full_transformer_static_region.md](full_transformer_static_region.md) |

## 2. Gate D0: Timing Truth

### 2.1 誤り

`wall_clock64()` の戻り値は tick であり、ns ではない。
初版は rate を仮定して `ticks / 1.0e6` で ms にしていたため **10 倍小さい値**を記録していた。

| 項目 | 値 |
|---|---|
| `hipDeviceAttributeWallClockRate` | **100000 kHz**（100 MHz） |
| 正しい換算 | `ms = ticks / wall_clock_rate_khz` |
| 旧換算（誤り） | `ticks / 1.0e6` |

検出の手掛かりは「GPU region 1.67 ms」と「host-observed 17.4 ms」が 10 倍乖離していたこと。
修正後は 16.74 ms と 17.48 ms で整合する。

### 2.2 helper

`include/phaseshift/runtime/gpu_mcu/wall_clock.h` + `.cpp`:

- `gpu_wall_clock_rate_khz(device)` — rate が 0 以下なら fail-closed
- `gpu_wall_clock_ticks_to_ms(ticks, rate_khz)`
- `gpu_wall_clock_ticks_to_us(ticks, rate_khz)`

device 側には換算を入れない。MCU は tick を記録するだけ。

### 2.3 cross-check

`test_gpu_mcu_wall_clock_conversion`（required、model 不要）が
device の tick 換算と HIP event を比較する。kernel 内で `wall_clock64()` を挟んだ
固定 compute loop を回し、その区間の tick と event elapsed を比べる。

```
wall_clock_rate_khz=100000
  probe iters=1000000 ticks=642444   device_ms=6.4244  event_ms=7.2887
  probe iters=4000000 ticks=2569436  device_ms=25.6944 event_ms=25.7145
  cross-check: device_ms/event_ms=0.9992
```

- 十分長い kernel では **0.08% 誤差**で一致する。
- 短い kernel では起動オーバーヘッドで 0.87 程度になる。これは換算誤差ではないので、
  cross-check は十分長い kernel で行う必要がある。

### 2.4 過去 document の修正

`full_transformer_static_region.md` の timing 節を修正値に置き換え、
誤換算だったことを明記した（履歴は削除せず注記する方針）。

`production_program_integration.md` の timing は HIP event 由来、
`real_primitive_chain.md` の timing は host `std::chrono` 由来で、
いずれも tick 換算していないため影響なし。

## 3. Gate D1: HIP Stream ↔ AQL Queue Bridge

### 3.1 能力確認

`hipDeviceAttributeCanUseStreamWaitValue` を起動時に確認する。
この環境では **1**（対応）。0 の場合は production MCU decode を unsupported とする。

### 3.2 signal memory

| 項目 | 結果 |
|---|---|
| `hipExtMallocWithFlags(hipMallocSignalMemory)` | **invalid argument**（この環境で未サポート） |
| 採用 | `hipHostMalloc(hipHostMallocMapped \| hipHostMallocCoherent)` |

**重要**: `GpuMcuFsm` は state を `hipHostMalloc` し、kernel へは
`hipHostGetDevicePointer` が返す **device VA** を渡す。したがって
**kernel が dereference する pointer は device VA でなければならない**。
host VA を config に入れると、device から見て不正なアドレスになり
「FSM が起動しない」という無言の失敗になる。

`GpuMcuSignalMemory` は `ptr`（host VA）と `device_ptr`（device VA）の両方を持つ。

### 3.3 契約

- **epoch**: batch ごとに増える 32bit。初回 1。
- **HIP → MCU**: prefix の後に `hipStreamWriteValue32(stream, start, epoch)`。
- **MCU**: `start_signal != nullptr` のとき idle 状態で `*start_signal` を
  System acquire で poll し、`run_consumed` を超えた値になったら region 開始。
  `run_consumed` は signal 経路では `= request`、legacy の `run_request` 経路では `+= 1`。
  **legacy は `+= 1` でなければならない**（後述）。
- **MCU → HIP**: region 成功時は `result_code = 0` → `done_signal = epoch`、
  fault 時は `result_code = fault_code` → `done_signal = epoch` の順で
  System release publish する。**done を先に見せない。fault でも必ず publish する。**
- **HIP wait**: `hipStreamWaitValue32(stream, done, epoch, Eq, 0xffffffff)`。

### 3.4 visibility sandwich

`test_gpu_mcu_hip_stream_bridge`（required）:

```
HIP prefix kernel      : a[i] = i * 0.5 + 1
  ↓ WRITE start = epoch
MCU AQL worker         : b[i] = a[i] * 2        (production の scale_f32 stable kernel)
  ↓ done_signal = epoch
HIP WAIT
HIP suffix kernel      : c[i] = b[i] + 1
```

比較: `c[i] == a[i] * 2 + 1` を byte-exact で確認（HIP → AQL と AQL → HIP の両方向を同時に検証）。

| 反復 | 結果 |
|---|---|
| 100 | `stale=0 mismatched=0 result_errors=0 mcu_dispatches=200 fault=0` |
| **10000** | `stale=0 mismatched=0 result_errors=0 mcu_dispatches=20000 fault=0` |

## 4. Gate D11.3-4: Fault でも HIP wait を解除する

`test_gpu_mcu_decode_fault_bridge`（required）が、不正な kernarg recipe を含む node を
投入して MCU を意図的に fault させる。

```
fault bridge: epoch=1 done=1 result_code=2 fault_code=2 dispatches=0 invalid_node=2
```

- fault でも `done_signal` が publish され、**HIP stream は deadlock しない**。
- `result_code` は `McuFaultCode` をそのまま運ぶ（この場合は `invalid_node = 2`）。
- fault は最初の dispatch 前なので `dispatches=0`。

## 5. Gate D4: Program Range Executor

- `execute_program_range(program, ctx, stream, begin, end)` を追加（半開区間）。
- `execute_program` は status reset（`hipMemsetAsync(error_word)`）を行ってから
  `execute_program_range(0, dispatches.size())` を呼ぶ wrapper にした。
  **status reset は range に入れない**（prefix / MCU / suffix で消してはいけない）。
- range 末尾を跨ぐ `consumed_dispatches` は `i + consumed > end` で fail-closed。
- executor 系 test 11/11 PASS。

## 6. Gate D3: Production Code Object Registry

Real HSACO を build path ではなく **embedded blob** として production binary から
ロードできるようにした。

| 項目 | 値 |
|---|---|
| embed した HSACO | rmsnorm / elementwise / bf16 / l2_normalize / gdn_conv1d / gdn_recurrence / rope / kv_append / attention_paged（9 個） |
| blob 実サイズ（合計） | 約 **900 KB** |
| `.inc` テキスト合計 | 5.56 MB |
| `libphaseshift_gpu_mcu.a` | 272,816 → **1,172,864 bytes（+879 KB）** |
| ロード API | `GpuMcuAqlCodeObject::load_memory()`（既存） |

- `gpu_mcu_embedded_kernel(name)` — blob を名前で引く generic レジストリ。
- `mcu_kernel_code_object(kind)` — `McuCompiledVariantKind` を
  `{blob, stable symbol, kernarg recipe, hidden args policy}` へ写す qwen35 側の registry。
  **MCU interpreter にはkernel名 switch を追加していない。**
- `CompletionMarker` は supervisor の probe image が担当するため unsupported を返す。
- `ActivationQuantizeE4m3K5120` / `Psq4Decode1Bf16U16`（W4A8 経路）は今回 embed せず
  fail-closed にする（transformer body decode では使われない）。

## 7. Gate D2: McuDecodeRuntime

test `Rig` への依存を production から外すため、runtime を分離した。

`src/phaseshift/models/qwen35/runtime/mcu_decode_runtime.{h,hip}`:

- 所有物: `GpuMcuCuPartition` / `GpuMcuAqlQueue` / `GpuMcuFsm` /
  `GpuMcuKernargRegion` / loaded code objects / plan・variant・retained・completion・
  record・output・timestamp の device buffer / 13 種の invocation table /
  start・done・result の signal memory。
- API: `create` / `shutdown` / `prepare_plan` / `enqueue_run` / `enqueue_wait` /
  `result_code` / `ready` / `fsm`。move-only、destructor は `shutdown()` の
  最終フォールバック。
- `prepare_plan` は registry 経由で code object をロードし、variant を組み、
  node / variant / retained / invocation を upload して FSM を configure + start する。
- shutdown 順序: FSM（stop → wait → shutdown）→ invocation table → device buffer →
  signal memory → probe code object → AQL queue。cleanup 途中で失敗しても残りを解放する。

`test_gpu_mcu_runtime_lifecycle`（required）:

```
runtime: nodes=4 variants=2 embedded_code_objects=1 start_memory=hipHostMalloc(coherent)
runtime lifecycle: cycles=100 epochs_per_cycle=3 faults=0 mismatched=0
                   result_errors=0 dispatches=600
```

- 1 cycle = runtime create → prepare_plan → 3 epoch 実行 → shutdown。
- **100 cycle で fault / mismatch / result code error が 0**（D16 の初期確認）。
- code object は **embedded blob から** ロードされている（D3 の検証を兼ねる）。

## 8. Gate D5: Transformer Body Range の保存

`create_model_executor()` は lowering 直後に PrimitiveGraph を破棄するため、
body range を executor 作成時に 1 回だけ確定して保存するようにした。

- `Executor::ProgramMcuRange { body_begin, body_end, valid }` を
  `ProgramSet::kRowBucketCount` 分だけ executor に持たせた。
- 確定は `find_dispatch_range_for_graph_prefix(lowered_graph, program, "L", ...)` を
  **executor 作成時に 1 回だけ**呼ぶ。submission ごとに文字列 heuristic を回さない。
- `begin < end` かつ `end <= dispatches.size()` を満たさない bucket は `valid=false`
  のままにする（MCU 経路は後段で `valid` を要求する）。
- `Executor` の move ctor / move assignment が `mcu_body_range` を運ぶ。
  `create_model_executor()` は local の `Executor` を return するため、move 経路に
  無いと production では常に `valid=false` になる（§13 の 11）。
- `find_dispatch_range_for_graph_prefix` は `ps::qwen35::runtime` にあるため、
  executor 側では `runtime::` 修飾が必要だった。

## 9. Gate D6 / D7: Backend Mode と Eligibility

production default は変更せず、experimental backend を追加した。

`src/phaseshift/models/qwen35/runtime/decode_backend.{h,cpp}`:

- `DecodeBackend { Host, GpuMcuAuto, GpuMcuRequired }`（default `Host`）を
  `ExecutorConfig::backend` として追加。
- 判断は **model 非依存の純関数** `decide_decode_backend(inputs)` に切り出した。
  GPU も Program も要らないため CPU-only test で全分岐を検証できる。
- `Auto` は eligible なら batch 全体を MCU hybrid path へ、そうでなければ
  **batch 全体を Host** へ回す（primitive 単位で混ぜない）。
  `Required` は eligible でなければ fallback せず fail-closed にする（E2E 用）。

eligibility 条件（D7）と reason:

| 条件 | reason |
|---|---|
| `ExecutionRole == Decode` | `NotDecode` |
| `num_requests == 1` | `MultipleRequests` |
| `num_tokens == 1` | `NotSingleToken` |
| `actual_rows == 1` | `NotSingleRow` |
| speculative verify でない | `SpeculativeVerify` |
| prefill を含まない | `PrefillPresent` |
| KV dtype が BF16 | `KvDtypeNotBf16` |
| imatrix collector なし | `ImatrixCollector` |
| value trace なし | `ValueTrace` |
| target hidden tap なし | `TargetHiddenTaps` |
| stream wait value 対応 | `StreamWaitUnsupported` |
| transformer body range が有効 | `McuBodyRangeUnavailable` |

計測用に `DecodeRuntimeCounters { plan_compile_count, plan_upload_count,
plan_switch_count, mcu_run_count, host_fallback_count }`（D9.5）も同じ header に置いた。

`test_gpu_mcu_decode_backend_policy`（cpu / required、GPU 不要）:

```
decode backend policy: cases=15
```

- default は Host、`Required` + 全条件成立で MCU 選択、`Auto` + 全条件成立で MCU 選択。
- 12 の rejection 条件それぞれで `Host` へ落ち、reason が一致し、`eligible=false`。

## 10. Gate D8: HostExecutionContext 構築の共通化

`launch_host_backend()` の中に埋まっていた `HostExecutionContext` の構築を
`build_host_execution_context(...)` として抽出した。

- 引数: executor / program / status_ptr / actual_rows / actual_outputs /
  actual_sampled_outputs / actual_stochastic_outputs / min_visible_tokens /
  max_visible_tokens / role / numeric_mode / gdn_spec_history / constraint_masks /
  constraint_mask_words / 出力先 ctx。
- `launch_host_backend` は helper を呼んでから `execute_program` するだけになった。
- **prefix / MCU body / suffix はこの 1 個の helper から同じ context を派生させる**
  （D8.1）。別々に pointer / stride / state を組み立てない。
- external binding は Program の external_input / external_output をそのまま埋める:
  input は token ids、output は hidden / logits / sampled_tokens / token_hidden /
  target hidden taps。**test fixture の pointer は一切使わない**（D8.2）。
- `decode_backend.h` は public header（`include/...`）へ移した。
  `executor.h` から include するため src 側に置けない。

## 11. Gate D9: Plan Cache と Physical Signature

毎 token `prepare_plan()` を呼ぶと FSM の stop / plan 再 upload / restart が走る。
D9 はこれを「物理署名が変化したときだけ」に制限する。

`src/phaseshift/models/qwen35/runtime/mcu_plan_cache.{h,hip}`:

- `McuPlanFingerprint { hash, dispatch_count, variant_count }`。
  `mcu_plan_fingerprint(plan)` は plan に焼き込まれた値（invocation 配列の
  device pointer / shape / grid、node、variant）を FNV-1a でハッシュする。
  **field を列挙しない**ので、plan に新しい値が増えても署名漏れが起きない。
  署名漏れは「古い plan を実行する」= 正しさの破壊なので、これは重要。
- `McuPlanCache::observe(shape, fingerprint)` が `{ needs_switch, needs_upload, reused }`
  を返す。`McuPlanShape { Direct, Split }` を跨ぐ切替も検出する。
- `DecodeRuntimeCounters`（§9）を cache が保持し、`plan_compile_count` /
  `plan_upload_count` / `plan_switch_count` / `mcu_run_count` / `host_fallback_count`
  を report できる。

**plan は CPU で毎 token 再 compile してよい。** compile は GPU に触れないため
「毎 token の stop/upload/restart」には当たらない。署名が同じなら upload を省く。

`test_gpu_mcu_plan_cache`（cpu / required、GPU 不要）:

```
plan cache: fingerprint=0xac1bfa9b3d3c1280
plan cache: compiles=6 uploads=4 switches=4
```

6 回 observe して compile 6・upload 4・switch 4。同一署名の 2 回目と、
shape も署名も同じ 5 回目は `reused` になり upload も switch も増えない。
ほかに「同一内容の独立 2 plan は同一署名」「device pointer / rows / grid / node 順の
差は別署名」「空 plan でも hash は非 0」「reset で counter が消える」を検証する。

### 11.1 padding の正規化

生 byte ハッシュは struct の padding に弱い。実際 `McuCompiledVariant`
（`uint8_t` enum + `uint32_t`×4）は `{}` 初期化でも padding が残り、
同一内容の 2 plan で hash が一致しなかった（offset 1 と 21 が 0x94 / 0x95）。

`mcu_plan_value.h` の `mcu_plan_value<T>()` を plan の値構築に使い、
`memset` で padding をゼロに固定する。plan compiler の invocation 構築 14 箇所を
これに置き換えた。すべての invocation の default member initializer は 0 なので
意味は変わらない。`McuPlanNode` は 28 byte で padding が無く、
`next = kMcuNoNext` の default を持つため対象外にした。

## 12. Gate D10 / D11.1-2: Production Hybrid Enqueue

production の submission path に MCU hybrid を接続した。default は `Host` のまま。

### 12.1 Executor の所有

`Executor` は `std::unique_ptr<runtime::McuDecodeState>` を持つ（`mcu_decode_state.h` =
`McuDecodeRuntime` + `McuPlanCache` + epoch + fault 記録）。**pimpl にした理由**:
`McuDecodeRuntime` は `micro_fsm.h` → `aql.h` を通して device builtin を引くため、
public header である `executor.h` から直接 include できない。`launch_host_backend` の
ときに `decode_backend.h` を public 側へ移したのと同じ制約。

`Executor()` は **out-of-line** にした。`= default` のままだと defaulted
constructor が member の destructor を要求し、`McuDecodeState` が incomplete な TU
（`kld.cpp` / `compute_runtime.hip`）で `unique_ptr` の `sizeof` static_assert に落ちる。

`shutdown()` / `is_shutdown()` / move ctor / move assignment へ配線し、shutdown では
`runtime.shutdown()` を呼んでから `mcu_state.reset()` する。

### 12.2 eligibility は launch 前に 1 回だけ確定する

`submit_co_batch` の中で `DecodeBackendInputs` を組み、`decide_decode_backend` を
1 回呼ぶ。`Auto` は batch 全体を Host か MCU のどちらかへ回し、`Required` は
not eligible なら **launch する前に** fail-closed にする。

```
Required + not eligible -> Status::unsupported（prefix/suffix の launch も行わない）
Auto     + not eligible -> batch 全体を Host
```

launch 後に失敗した場合は fallback せず poison する（prefix が既に stream に載って
いるため、Host へ差し替えると primitive を混ぜることになる）。

`ensure_mcu_state` は `backend != Host` のときだけ lazy に呼ぶ。`hipStreamWaitValue32`
非対応なら fail-closed。`Auto` で作成に失敗した場合のみ Host 継続を許す。

### 12.3 hot path

`enqueue_batch` は ctx を `build_host_execution_context` で 1 回だけ構築し、
prefix / MCU body / suffix が同じ context から派生する。

```
hipMemsetAsync(error_word)
prefix range  [0, body_begin)
enqueue_run(stream, epoch)     -- hipStreamWriteValue32(start, epoch)
enqueue_wait(stream, epoch)    -- hipStreamWaitValue32(done, epoch)
suffix range  [body_end, dispatch_count)
```

`hipStreamSynchronize` / `wait_plans` / CPU polling / `request_run` は使わない。
epoch は batch ごとに +1。`hipGraph` capture は MCU path と併用しない
（`graph_active = graph_eligible && !mcu_selected`）。

plan は毎 token CPU で compile し、D9 の cache が signature を比較する。signature が
同じなら upload も FSM restart も起きない。

### 12.4 E2E が検出した production 側の hazard

Qwen3.5-4B + production executor で、64 token prefill の後の 1 token decode を
`GpuMcuRequired` で流すと、eligibility は成立する（実測 `eligible=1 reason=Selected`）
が batch が完了しなかった。当初は「FSM が `done` を publish しない」と見ていたが、
`submit_batch` + 非同期観測の probe（`PHASESHIFT_MCU_PROBE=1`、30 × 0.5 秒上限）で
**MCU body 自体は正常**と判明した。

```
probe t=0 stream=0 epoch_seen=1 plans=1 disp=697 comp=1 node_count=699 fault=0 result=0
probe: stream completed after 1 polls
```

- `epoch_seen=1` — FSM は epoch を認識している
- `dispatches_committed=697` / `node_count=699` — production body plan を完走
- `fault_code=0` / `result_code=0`
- `hipStreamQuery` が 500 ms 以内に `hipSuccess` — **stream は完了している**

停止点は `complete_batch` の中だった。step 出力で 1 つずつ確認した結果:

| 段階 | FSM 生存中の挙動 |
|---|---|
| `hipEventSynchronize` / `hipEventDestroy` | 返る |
| `hipMemcpyWithStream`（pageable dst への D2H） | **返らない** |

原因は `execution_status` の readback が同期 D2H コピーだったこと。
`docs/developer/gpu_mcu/low_level.md` に元から書かれていた contract
（persistent wave 稼働中に legacy/default stream 操作を行うと待ちが発生する）に
production の `complete_batch` が違反していた。

修正:

- `Executor` に pinned (`hipHostMalloc`) の 4 byte status staging を持たせ、
  `hipMemcpyAsync` + `hipStreamSynchronize(stream)` で読む。
- `gpu_mcu_signal_load` の同期 D2H 経路を削除し、signal memory は
  `hipHostMalloc` の mapped 系のみ（host-readable を必須条件）として fail-closed
  にした。host から読めない signal は CPU 同期なしの受け渡しという設計自体を
  満たせない。

### 12.5 再発防止: build 時の静的ゲート

contract を文書で守る方式は機能しない（実際 `complete_batch` が踏んだ）ため、
build 時に落とす gate を入れた。

`tools/check_mcu_sync.py` が mcu-aware な翻訳単位（`src/phaseshift/runtime/gpu_mcu/`、
`mcu_decode_runtime.hip`、`mcu_plan_*.{h,hip}`、`mcu_kernel_registry.hip`、
`executor.hip`）を走査し、次のいずれかで失敗する。

- `hipDeviceSynchronize`
- 同期 `hipMemcpy` / `hipMemcpyWithStream` の `hipMemcpyDeviceToHost`

`cmake/targets.cmake` の `phaseshift_mcu_sync_check` target が `phaseshift_gpu_mcu` と
`phaseshift_qwen35_runtime` の依存に入っているため、compile より前に走り、違反が
あれば build が止まる。違反を故意に注入して build が FAILED になることを確認済み。

### 12.6 D11.1-2: fault を commit 前に検出する

`Executor::active_submission_mcu_epoch` で in-flight の MCU run を追跡する。
`complete_batch` は `hipEventSynchronize(done)` の後、**host の execution_status より
先に** `mcu_runtime.result_code()` を確認する。非 0 なら `poison_pending()` して
commit せずに返す。

```
mcu transformer body failed: result_code=<n>
```

result signal は host-mapped coherent memory なので、stream を同期した後に host から
そのまま読める。hot path では読まない。

## 13. Failures / Surprises

1. **`wall_clock64()` の tick を ns として扱っていた。** device rate は実測 100000 kHz。
   「GPU region と host-observed が 10 倍違う」が検出の手掛かり。
2. **host VA と device VA の混同。** FSM の state は `hipHostMalloc` + `hipHostGetDevicePointer`
   であり、kernel に渡す pointer は device VA でなければならない。症状は
   「signal が見えない」という無言の失敗で、`epoch_seen` を publish して初めて切り分けられた。
3. **null stream hazard。** persistent MCU kernel が動いている間、`hipMemcpy` などの
   null stream 同期は永久に返らない。検証 readback も `hipMemcpyAsync(..., stream)` +
   `hipStreamSynchronize(stream)` を使う。D1 の最初の hang はこれだった。
4. **`run_consumed = request` は legacy 経路を壊す。** `request_run()` が FSM 消費前に
   複数積まれる test（`device_completion` など）では、`run_consumed` が飛んで
   後続 region が永久に開始されない。症状は `disp=8 comp=8` での wait timeout。
   signal 経路は epoch ベースで `= request`、legacy は `+= 1` に分ける。
5. **`hipMallocSignalMemory` が使えない環境がある。** 確保方式は fallback chain にし、
   実際に使えた種類を報告する（本環境は `hipHostMalloc(coherent)`）。
6. **診断手法。** HIP WAIT が stream に載っている間は host から同期できない。
   mapped memory に制御フローのファネルを publish し、sleep 後に非同期観測する方式が有効。
7. **`GpuMcuFsmConfig.output_base` / `timestamps_base` を設定しないと probe worker が
   address 0 へ書く。** runtime を Rig から写すときこの 2 つを落とし、初回実行で
   illegal memory access になった。Rig は専用バッファを用意している。
8. **null stream hazard は runtime の test でも再発した。** persistent MCU kernel が
   動いている間の `hipMemcpy` は永久に返らない。検証 readback も
   `hipMemcpyAsync(..., stream)` + `hipStreamSynchronize(stream)` を使う。
9. **`test_gpu_mcu_real_rmsnorm_feed` は co-residency に敏感。** この test は
   「feeder が queue を枯らさない」ことを `queue_empty_count() <= plans` で検証する
   timing 依存の性質を持つ。同一 GPU を他の test と共有すると FSM の wave が
   deschedule され、1000 plan 中 1 回だけ余分に empty を観測して失敗する
   （単独実行では 3/3 PASS）。assertion を緩めず、ctest の `RUN_SERIAL` で
   GPU を排他にして解消した。`GPU_COST_GB` を上げる方法は runner が
   `used + cost <= budget(24)` で判定するため他 test を排除できず不十分だった。

10. **plan の padding が未初期化で署名が不安定だった。** `McuCompiledVariant` は
    `{}` 初期化しても padding byte が残る。生 byte ハッシュのため同一内容の 2 plan で
    hash が変わり、D9 の署名が常に「変化した」と判定されてしまう。検出は
    「同一内容の独立 2 plan の hash 比較」で、diff 位置（offset 1 / 21）から
    padding と確定した。`mcu_plan_value<T>()` で構成時にゼロ固定して解消した。
11. **D5 の `mcu_body_range` が move で運ばれていなかった。** `create_model_executor`
    は local の `ex` に body range を設定して `return ex` するため、move ctor /
    move assignment に含まれていないと **production では常に invalid** になる。
    D5 時点の構文検証では見つからず、D10 で「prefix の終端が分からない」段になって
    判明した。move 経路へ追加して修正した。回帰は E2E test の
    「mcu body range survives executor creation」で押さえる。
12. **`Required` の eligibility 判定が prefix 前に効くことを E2E で確認した。**
    prefill を `GpuMcuRequired` で流すと、prefill は eligible でないため
    `prepare_batch_descriptor` に到達する前に fail-closed する
    （`Auto` なら batch 全体が Host に回る）。

13. **signal memory の probe 失敗が sticky error として残り、host の kernel launch が
    「失敗」と誤報される。** `gpu_mcu_alloc_signal_memory` は
    `hipExtMallocWithFlags(hipMallocSignalMemory)` を最初に試す。この環境では
    `hipErrorInvalidValue` で失敗するので `hipHostMalloc` へ fallback して成功するが、
    **失敗した probe の error は thread に残る**。`launch_prepare_batch_descriptor` は
    `hipLaunchKernelGGL` の後で `hipGetLastError()` を見るため、launch 自体が成功して
    いても古い error を拾い `prepare_batch_descriptor: invalid argument` を返す。
    症状は「MCU runtime を作った後の batch だけ host 側の launch が失敗する」。
    probe 失敗の直後に `hipGetLastError()` でクリアして解消した。D1 の bridge test は
    signal memory 確保の後に host kernel を起動しないため検出できず、production の
    submission path を通って初めて出た。

14. **`RowBucket` を配列 index として使ってしまった。** `RowBucket` の enum 値は
    行数（`R16 = 16` … `R2048 = 2048`）であり、index ではない。
    `mcu_body_range[static_cast<std::size_t>(bucket)]` は 8 要素 `std::array` の
    範囲外読みになり、`valid` を偶然 0 として読むため
    「body range が無い」と**無言で**判定されていた。
    正しい key は program slot index（`program - &program_set.programs[0]`）で、
    D8 が `program_staging_meta` に使っている key と同じ。症状は
    `eligible=0 reason=McuBodyRangeUnavailable`。
    気づけたのは E2E の counters が全て 0 で「MCU を使わず Host 同士を比較した
    偽の PASS」になっていたため。**MCU 経路が走ったことを assert しない E2E は
    偽陽性を返す**という教訓。
15. **production の `complete_batch` が同期 D2H readback で待ちに入っていた。**
    症状は「MCU body は完走しているのに batch が完了しない」。`submit_batch` +
    非同期観測の probe で body の正常を確認し、`complete_batch` の step 計測で
    `hipMemcpyWithStream`（pageable dst への D2H）と確定した。
    `docs/developer/gpu_mcu/low_level.md` に元からあった contract に production が
    違反していた。pinned staging + `hipMemcpyAsync` + `hipStreamSynchronize(stream)`
    に修正。あわせて **contract は文書では守られない** という前提に立ち、
    `tools/check_mcu_sync.py` を build 依存に入れて compile 前に落とすようにした。
16. **probe を最初から使うべきだった。** 40 分の timeout を 2 回消費したのは、
    「FSM が `done` を publish しない」という推測で stream を待ち続けたため。
    `submit_batch` して待たずに mapped memory を観測すれば 15 秒で
    「body は正常・停止点は complete_batch」と分かった。D1 で確立済みの手法だった。
    E2E test の timeout も 2400 → 900 にした。

## 14. 検証状況

- required acceptance: **174 / 174 PASS**
- `ctest -L gpu_mcu`: **rc=0**
- 新規 required test: `test_gpu_mcu_wall_clock_conversion` /
  `test_gpu_mcu_hip_stream_bridge` / `test_gpu_mcu_decode_fault_bridge` /
  `test_gpu_mcu_runtime_lifecycle` / `test_gpu_mcu_decode_backend_policy` /
  `test_gpu_mcu_plan_cache`
- 新規 optional test: `test_gpu_mcu_production_decode`（`optional;external_files`）。
  Qwen3.5-4B で production executor を作り、body range が executor 作成後も有効な
  ことと、`backend=Host` の prefill + decode が通ることを検証する。
  MCU hybrid の比較は `PHASESHIFT_MCU_E2E=1` を要求する（既定では
  §12.4 の阻害要因により実行しない）。
- `execute_program` の挙動は executor 系 11 test で変化なし

## 15. 未着手（次以降）

| Gate | 内容 |
|---|---|
| ~~D2~~ | 完了（§7）。plan switch 時の FSM restart は v0 として未検証 |
| ~~D3~~ | 完了（§6） |
| ~~D5~~ | 完了（§8） |
| ~~D6 / D7~~ | 完了（§9）。`enqueue_batch` への接続は D10 |
| ~~D8~~ | 完了（§10） |
| ~~D9~~ | 完了（§11）。executor への接続は D10 |
| ~~D10~~ | 完了（§12）。`enqueue_batch` が backend aware になった |
| ~~D11.1-2~~ | 完了（§12.4） |
| D12 | production single decode の Host 一致。**§12.4 の hang 解消が前提** |
| D13-D14 | sequence decode / timing 比較（`wall_clock_rate_khz` 換算） |
| D15-D17 | endurance / lifecycle / default は Host のまま |
