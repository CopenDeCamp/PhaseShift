> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# CB11 Mixed Physical Batch Production Qualification（進行記録）

> Status: R&D record（進行中）。mixed batch の production execution qualification の
> 途中経過と、判明した blocker を記録する。現在も有効な contract は
> [docs/developer/gpu_mcu/low_level.md](../../developer/gpu_mcu/low_level.md) を参照。

## 1. revision

| | |
|---|---|
| baseline | `d2bcba74`（poc/gpu-mcu-program-plan-one-layer、CB10 + KV addressing + GDN reset） |
| worktree | `.worktrees/gpu-mcu-cb11-mixed-production` |
| branch | `feat/gpu-mcu-cb11-mixed-production` |

## 2. Gate の目的

1 個の `DeviceBatchContext` に PREFILL / DECODE / VERIFY を共存させ、actual Program Plan →
actual AQL dispatch → actual KV addressing → actual recurrent state → actual attention →
actual linear/GEMM → actual commit/scatter までを 1 physical batch として実行できることを
証明する。geometry が作れることではなく、1 個の MCU inference transaction の中で
class 固有の physical kernel へ分解し、結果を正しい request へ戻せることを示す。

## 3. この作業で完了した部分 — mixed paged attention の physical region 分解

### 3.1 背景

`should_use_paged_attention_prefill()` は batch 全体が単一 request のときだけ PREFILL
kernel を選ぶ。mixed batch では全 row が decode kernel で処理され、prefill kernel の
32-row tile 前提（tile 先頭 row の `row_sequence_slots` / tile 末尾 row の position を
tile 全体へ流用）と整合しない。そのため logical な `PAGED_ATTENTION` dispatch を
execution-class / request region 単位の physical dispatch へ分解した。

### 3.2 region 分解

`resolve_paged_attention_physical()` を拡張し、`batch_context->num_requests > 1` のとき
descriptor 列から region を決める。

- decode / verify request 群 → 先頭 PREFILL request の `row_begin` までの decode region。
  decode kernel は row 独立なので decode/verify を同一 region にできる。
- 各 PREFILL request → 1 prefill region。`row_count >= kPagedAttentionPrefillMinRows`
  （128）なら PREFILL kernel、未満なら decode kernel。request 境界を跨ぐ tile を作らない。

region の `PagedAttentionCommonArgs` は元の common を row offset でコピーして作る。
`q` は `q_row_stride`、`output` は output dtype の byte 幅、`row_sequence_slots` /
`row_positions` は 1 row 単位で offset する。`PagedAttentionCommonArgs` の ABI は変更しない。

`PhysicalPagedAttentionPlan` の `kernels` 容量を `2 + 16`（decode split/reduce + prefill
region 上限）へ拡張した。上限は compile-time で、超過時は `NotApplicable` へ fail-closed
する。

`launch_paged_attention_physical()` は従来 decode 系 variant で `return` しており
physical_count > 1 の plan を途中で打ち切っていた。全 variant を `break` へ揃え、
region を順に発行するようにした。

### 3.3 MCU plan への反映

`compile_mcu_plan()` の `PAGED_ATTENTION` case を、plan の全 kernel を走査して
1 kernel = 1 node へ展開する形に変更した。BF16 decode（`AttentionPagedBf16`）と
BF16 prefill（`AttentionPagedPrefillBf16`）は同じ `kMcuKernargRecipeAttentionPaged` /
`McuPagedAttentionInvocation` を使う。split / reduce も kernel 単位で node 化する。
plan の variant 数が `kMcuMaxVariants`（64）を超える場合は fail-closed する。

### 3.4 qualification

新規 required test `test_gpu_mcu_mixed_attention_regions`（`gpu1;gpu_mcu;required`）は
外部 model file を使わず、synthetic な Program + mixed `DeviceBatchContext` から
Host physical path と MCU AQL plan を同条件で実行し、全 row の output を byte 比較する。

- `decode_prefill_decode`: DECODE 1 + DECODE 1 + PREFILL 160 = 162 row、2 region。
- `prefill_prefill`: PREFILL 160 + PREFILL 128（threshold 超が 2 request）、2 region。
  tile 数が異なるため 2 variant。
- `decode_verify_prefill`: DECODE 1 + SPEC_VERIFY 8 + PREFILL 128 = 137 row、2 region。
  verify row が decode-compatible region に入ることを確認。

3 scenario とも output 全 row byte exact、dispatch 数 = region 数 + marker、fault 0。

## 4. 未完了の gap（判明した blocker）

### 4.1 real Program Plan を persistent MCU で走らせる bridge が無い

`compile_mcu_plan()` の実 plan を実行するのは `McuDecodeRuntime` であり、
`GpuMcuPersistentMcu`（autonomous continuous batching の本体）は synthetic な probe chain
しか実行できない。両者を繋ぐ API が無く、signal mode（`start_signal`）の所有権も競合する。
これが CB11 の最大の blocker で、3-class mixed production execution はこの bridge に依存する。

- 影響 kernel: FSM 実行経路全般（`src/phaseshift/runtime/gpu_mcu/micro_fsm.hip`,
  `persistent_mcu.hip`, `mcu_decode_runtime.hip`）
- 必要な変更: compiled `McuCompiledPlan` を未起動の `GpuMcuFsmState` へ upload し、
  `configure_execution` へ渡す API。`McuDecodeRuntime` から code object / kernarg /
  invocation table を引き渡す ownership 設計。
- 追加論点: 現行 plan は row 数 / grid を compile 時に固定するため、loop ごとに変わる
  mixed geometry をどう表現するか（bucket 固定 or per-loop patch）。

  → 本作業で GPU-side dynamic node binding（§7）を追加し、static skeleton の
  variant / grid / enabled を loop ごとに device 側で上書きできるようにした。
  bridge 本体（prepared FSM を persistent MCU が drive する経路）は未接続。

  → 続けて external persistent driver mode（§8）を追加し、`McuDecodeRuntime` が
  `fsm.start()` せずに prepared FSM を `GpuMcuPersistentMcu` へ貸し出せるようにした。
  real Program の full output closure と VERIFY / GDN transaction は未接続。

  → §12 で canonical 26-row mixed plan を persistent bridge で実行し、Host reference と
  byte exact に一致させた。bridge 自体は attention region については成立。
  full chain（linear / GDN / KV / lm-head / sampling）の 1 plan 統合と per-loop geometry
  change、commit 統合は未完了。

  → §13 で GPU-side invocation patch binder を追加し、prepared plan を再 prepare せずに
  row-global invocation の行数を loop ごとに device 側で書き換えられるようにした。
  executor からの patch 供給と mixed autonomous multi-loop は未接続。

  → §15 で canonical 26-row mixed batch の full chain（rmsnorm + quantize + linear +
  kv_append + attention + GDN + sampling）を 1 plan に統合し、persistent bridge で
  Host reference と byte exact に一致させ、`sampled_tokens` まで MCU 内で生成した。
  executor からの供給、commit 統合、mixed autonomous multi-loop は未接続。

  → §18 で request runtime + batch planner + batch binding + execution step + commit の
  経路を通し、canonical mixed batch が 1 loop で commit されることを確認した。
  残るは production executor からの供給と mixed autonomous multi-loop。

  → §19 で verify accepted count を plan 内の VERIFY_ACCEPT node として device 計算し、
  commit がその値を使うようにした。§18 の test を 2 loop に拡張し、loop ごとに
  sampling outputs / verify request count を再 binding して commit するようにした。

  → §20 で production executor の MCU 経路を persistent MCU へ接続し、executor が context /
  requests / commit slot / patches を供給するようにした。実 model の mixed batch
  （DECODE + PREFILL）が host と byte exact に一致する。

  → §22 で `OUTPUT_GATHER` の MCU entrypoint を追加し、executor の plan を program 全体へ
  拡張した。単一 output の decode は embedding から sampling まで 1 plan で走り、
  `sampled_tokens` が MCU 内で生成される。

  → §24 で attention region の `rows` を loop ごとに device 側で再 binding するようにし、
  mixed batch の composition が変わっても static plan のまま正しい region 形状で走るように
  した。残るは verify 供給と multi-output lm-head。

### 4.2 PSQ4 multi-row MCU entrypoint（row-block family は実装済み、2D は未実装）

`compile_mcu_plan()` の `LINEAR_PSQ4` は従来 `Decode1Bf16`（unroll 16/8）のみ受理していた。
今回 row-block family を実装した。

- `psq4.hip` の WMMA kernel body を `__device__` 関数へ切り出し、`RowBlock1`（policy kernel）と
  `RowBlock2/4/8`（`gemm_psq4_w4a8_wmma_kernel<MB>`）の `extern "C"` AQL wrapper
  （`phaseshift_qwen35_psq4_rowblock{1,2,4,8}_bf16`）を追加した。
- 共通 args `Psq4MultiRowBf16Args`（72 byte、static_assert 付き）と MCU invocation
  `McuPsq4MultiRowInvocation`（72 byte、cross static_assert 付き）、recipe
  `kMcuKernargRecipePsq4MultiRowBf16`、FSM writer を追加した。
- `PhysicalPsq4Variant` / `resolve_psq4_physical()` が row-block config を
  variant + grid + `grid_y` へ解決し、`compile_mcu_plan()` が 1 node を emit する。
- registry（`mcu_kernel_code_object`）が 4 kind を正しい symbol / recipe へ解決する。
- 新規 required test `test_gpu_mcu_real_psq4_rows_aql` が、selector が選んだ config を
  そのまま使い、rows = 2 / 8（RowBlock1）と 32 / 26（RowBlock2）で Host launcher と
  host AQL の output を byte exact 比較する（全 case diffs 0）。registry の
  symbol / recipe / blob も同 test で検証する。

未実装（documented coverage）:

- `Prefill2D_K64N64 / K128N64 / K64N128 / K128N128`（rows が 256 の倍数で大きい場合）。
  selector がこれらを選ぶ shape では `compile_mcu_plan()` が
  `psq4 physical variant has no mcu entrypoint` で fail-closed する。
- PSQ8 の multi-row。CB11 target が PSQ8 path を使う場合に同じ closure を行う。
- FSM recipe writer の end-to-end 実行。`decode_backend` eligibility が未だ
  `actual_rows != 1` を拒否するため、real model の mixed plan がまだ到達しない。

### 4.3 SPEC_VERIFY transaction（KV 側は CB11-VT sub-gate で完成）

CB11-VT（[cb11_verify_kv_transaction.md](cb11_verify_kv_transaction.md)）で KV 側の
resource transaction（candidate 予約、base 固定、accepted boundary への shrink、abort、
batch commit 統合）を完成させた。ここでは重複を避け、詳細は同ドキュメントを正本とする。

まだ device 側に欠けているもの:

- decode→verify の phase 遷移と `verify_candidate_ref` / `verify_candidate_count` を設定する
  production 経路（現状 test 専用）。
- verify logits から device 側で accepted count を生成する経路（Host 計算は禁止）。
  → accepted count kernel 自体は §9 で追加済み。skeleton への node 予約が未完了。
- accepted-boundary の GDN/recurrent restore を device 側で行う kernel（現状 Host の
  `restore_gdn_spec_history` のみ）。
  → restore kernel 自体は §10 で追加済み。accepted count 参照 → restore の node 予約が
  未完了。

したがって `validate_scheduled_batch` の guard はまだ解除しない。

`reserve pass の verify 未対応`（`persistent_mcu.h` の `gpu_mcu_resource_reserve_pass`）と
`page tail trim primitive の不在`（`sequence_resource.h`）は、transaction の成立に必須。

### 4.4 real-model mixed multi-loop qualification 未実施

上記 4.1〜4.3 が揃うまで、27B-PSQ mixed の compile / execute はできない。

## 5. この時点で確認できたこと

> mixed batch の paged attention を、logical 1 dispatch から request 境界を跨がない
> physical region へ分解し、MCU の AQL plan として実際に dispatch して Host path と
> byte exact に一致させられること。decode / verify は同一 region、prefill は request
> ごとの region になる。

これは CB11 の attention region 分解（Gate の一部）を満たすが、PSQ4 multi-row、
VERIFY transaction、persistent bridge が未完了のため Gate は閉じない。

> 加えて、PSQ4 row-block family（RowBlock1/2/4/8）が MCU の AQL entrypoint として
> 実際に dispatch でき、selector が選んだ config のまま Host launcher と byte exact に
> 一致すること（rows 2 / 8 / 26 / 32）。

> また、verify の candidate 数を考慮した KV page 予約と、accepted 位置までの
> page tail rollback（block table の page id を使った index 指定 trim）が device 側で
> 成立し、pool invariant を壊さないこと。

> canonical 26-row mixed batch（DECODE 1 + DECODE 1 + SPEC_VERIFY 8 + PREFILL 16）の
> attention region 分解が、prepared FSM を `GpuMcuPersistentMcu` が drive する経路で
> Host reference と byte exact に一致すること（§12）。

> static skeleton を再 prepare せずに、row-global invocation の行数を device 側の
> patch overlay で loop ごとに書き換えられること（§13）。

> multi-row の実 multi-operator chain（rmsnorm + e4m3 quantize + row-block linear）が
> 1 plan として persistent MCU で走り、Host reference と byte exact に一致し、
> 2 loop 目で行数を device 側で縮めても再 prepare なしで一致すること（§14）。

> canonical mixed batch で KV append + linear + attention（2 region）+ GDN（reset + conv1d +
> recurrence）+ sampling が同一 plan に共存し、persistent bridge でそれぞれ Host reference
> と byte exact に一致し、`sampled_tokens` まで MCU 内で生成されること（§15）。

> GDN（reset + conv1d + recurrence）が 1 plan として persistent bridge で実行され、
> Host reference と byte exact に一致すること（§16）。

> mixed batch の decode + decode + verify + prefill が 1 transaction で commit されること
> （§17）。

> canonical mixed batch が request runtime + batch planner + batch binding + execution
> step を通り、1 loop で commit され、decode / verify / prefill の slot が accepted count
> を含めて正しく進むこと（§18）。

> verify の accepted prefix が plan 内の VERIFY_ACCEPT node で device 計算され、その値が
> commit に使われること（§19）。

> canonical mixed batch が autonomous loop で 2 loop 走り、loop ごとに sampling outputs と
> verify request count が再 binding され、それぞれ commit されること（§18）。

> production executor の MCU 経路が persistent MCU を駆動し、executor が context /
> requests / commit slot / patches を供給して、実 model の mixed batch（DECODE + PREFILL）が
> host backend と byte exact に一致すること（§20）。

> `OUTPUT_GATHER` が MCU plan に統合され、単一 output の decode で embedding から sampling
> までが 1 plan として走り、`sampled_tokens` が MCU 内で生成され、host backend と一致する
> こと（§22）。

> attention region の `rows` が loop ごとに device 側で再 binding され、mixed batch の
> composition が変わっても static plan のまま正しい region 形状で走ること（§24）。

## 6. sync audit

production hot path への同期追加は無い。`tools/check_mcu_sync.py` clean。

## 7. dynamic node binding（static skeleton + GPU-side per-loop overlay）

§4.1 の「per-loop geometry」問題に対する第一段。static plan を再 compile / 再 upload
せずに、device memory 上の binding table で loop ごとの execution を上書きする。

### 7.1 binding ABI

`McuDynamicNodeBinding`（32 byte、`alignas(32)`、offset static_assert 付き）を
`micro_fsm.h` に追加した。plan node index と 1:1 で対応する。

- `enabled`: 0 の node は dispatch せず `next` へ進む。dispatch node を disable した
  場合、直後の対応 wait node も自動で読み飛ばす（plan 構造に依存しない）。
- `variant_id` / `invocation_index`: static node の値を上書きする。
- `workgroup_count_x/y/z`: 非ゼロなら static geometry を grid 上書きする。
  workgroup size（kernel 側 `workgroup_x/y/z`）は static のまま維持する。

`GpuMcuFsmState` / `GpuMcuFsmConfig` に `dynamic_node_bindings` と
`dynamic_node_binding_count` を追加し、`GpuMcuFsm::configure()` が device table を
そのまま参照する。Host は binding table の pointer を渡すだけで、loop ごとの
re-compile / re-upload を行わない。

### 7.2 dispatch 反映

`mcu_run_once()` は node fetch 後に binding を適用し、effective な node / variant /
geometry を解決する。kernarg builder（`mcu_build_kernarg` と各 recipe writer）は
`node.variant_id` から variant を引く代わりに、解決済み variant を引数で受け取る。

grid 上書きは retained packet の `grid_size_x/y/z`（本 AQL では global work items）と
kernarg launch metadata の `workgroup_count_*` を同時に patch する。
`expand_retained_packet()` / `gpu_mcu_aql_stage_retained_packet()` に optional な
workgroup count 引数を追加し、template の workgroup size を掛けて global work items を
書く。prefetch 経路（`prepared_enabled`）も同じ解決を通す。

### 7.3 qualification

新規 required test `test_gpu_mcu_dynamic_plan_binding`（`gpu1;gpu_mcu;required`）は、
1 個の static skeleton と device binding table を使い、Host plan re-upload なしで
連続 loop を実行する。binding の更新は test 内の device kernel で行う。

- static geometry: variant 0（wg 64 × 1）→ 64 thread のみ書く。
- variant switch: variant 1（wg 32）→ 32 thread のみ書く。
- grid override: variant 0 + count x 2 → 128 thread を書く。
- disabled: dispatch を skip し、dispatch count を増やさない。
- prepared prefetch: `prepared_enabled = 1` で 2 dispatch を走らせ、prefetch された
  dispatch の variant / grid 上書きが反映される（wg 32 × 4 = 128 thread）。

これは CB11 の dynamic binding 部（static skeleton + GPU-side per-loop binding）を
満たすが、prepared FSM を persistent MCU が drive する bridge、actual VERIFY execution、
GDN restore が未完了のため Gate は閉じない。

## 8. external persistent driver mode（bridge の前半）

§4.1 の「prepared plan を persistent MCU が drive できない」問題に対する第一段。

### 8.1 driver mode

`McuDecodeRuntimeOptions` に `McuPlanDriverMode`（`HostStream` / `ExternalPersistentMcu`）を
追加した。

- `HostStream`（既定）: 従来どおり `prepare_plan()` が `GpuMcuFsm::start()` まで行い、
  FSM kernel を起動する。`enqueue_run` / `enqueue_wait` は `start_signal` / `done_signal`
  を Host stream から駆動する。
- `ExternalPersistentMcu`: `prepare_plan()` は `GpuMcuFsm::create()` + `configure()` までで
  止め、`fsm.start()` を呼ばない。`start_signal` / `done_signal` / `result_code` も設定
  しない（persistent MCU は `run_request` を直接書いて `mcu_run_once` を呼ぶため、
  external epoch signal と併用できない）。`enqueue_run` / `enqueue_wait` は
  `invalid_state` を返す。

`McuDecodeRuntime::execution_state()` が `GpuMcuFsmState*` を公開し、
`GpuMcuPersistentMcu::configure_execution()` へそのまま渡せる。ownership は
`McuDecodeRuntime` が保持し、persistent MCU は borrow するだけ。shutdown は
persistent MCU stop → no execution in flight → runtime destroy の順を守る。

### 8.2 marker-only plan の修正

`prepare_plan()` の `needed_segment` が `CompletionMarker` variant の
`probe_meta_.kernarg_segment_size` を反映しておらず、marker だけの plan で kernarg slot
stride が不足していた（`kernarg slot stride does not cover a variant segment`）。
marker variant でも `needed_segment` を更新するよう修正した。

### 8.3 qualification

新規 required test `test_gpu_mcu_external_persistent_driver`（`gpu1;gpu_mcu;required`）は、
external モードで `McuDecodeRuntime::prepare_plan()` した marker-only plan を、
`GpuMcuPersistentMcu::configure_execution()` 経由で実際に drive する。

- `prepare_plan` 後に `fsm().running() == false`（Host controller kernel 未起動）。
- `enqueue_run` / `enqueue_wait` が reject される。
- persistent MCU を start し、batch context の `actual_rows` を立てて ready epoch を
  publish すると、`batches_dispatched >= 1` / `batches_completed >= 1` になる。
- その間 `fsm().running() == false` のまま（second controller kernel が無い）。
- 実行後 `GpuMcuFsmState::plans_started >= 1`、fault 0。

これは CB11 の persistent bridge の前半（prepared FSM の外部駆動）を満たす。
後半（real Program を full output まで MCU 内で閉じる、VERIFY / GDN transaction）は
未完了のため Gate は閉じない。

## 9. VERIFY accept count device kernel

§4.3 の「accepted count を device 側で生成する経路が無い」問題に対する第一段。
VERIFY の proposal engine 全体ではなく、accepted count の device 生成だけを追加した。

### 9.1 kernel

新規 MCU kernel `phaseshift_gpu_mcu_verify_accept_prefix`
（`src/phaseshift/models/qwen35/kernels/optimized/verify_accept.hip`）。

- input: `candidates`（request ごとに `candidate_stride` 間隔）、`sampled_tokens`（同じ
  配置の target sampled token 列）、`candidate_counts[request]`。
- output: `committed_counts[request]`。
- `accepted_count = candidate 先頭から target sampled token と一致する連続数`。
  `max_candidates`（32）で clamp する。
- 1 workgroup（`kVerifyAcceptPrefixThreads = 32`）で request を grid-stride する。
  Host 計算は行わない。

`VerifyAcceptPrefixArgs`（48 byte）と `McuVerifyAcceptPrefixInvocation`（48 byte）を
cross static_assert で一致させ、recipe `kMcuKernargRecipeVerifyAcceptPrefix`、FSM writer、
registry（`McuCompiledVariantKind::VerifyAcceptPrefix`、blob `verify_accept`）、
embedded kernel entry を追加した。

### 9.2 latent bug 修正

`GpuMcuFsm::configure()` が `psq4_multi_invocations` を state にコピーしておらず、
PSQ4 multi-row の FSM 実行が常に `nullptr` を見て失敗する状態だった。合わせて
`psq4_multi` と `verify_accept` の validation / copy を追加した。

### 9.3 qualification

新規 required test `test_gpu_mcu_verify_accept`（`gpu1;gpu_mcu;required`）は、
real kernel を plan 内で marker dispatch と組にして FSM から実行し、accepted count を
Host reference（leading match 数）と比較する。

- accepted 0 / 1 / 4 / 8（8 candidate）
- accepted 32（32 candidate、全一致）

Host readback は結果比較のみに使い、accepted の決定には使わない。

これは CB11 の VERIFY accepted count device 生成を満たす。skeleton への optional node
予約（sampling 後の node 化）と GDN restore は未完了のため Gate は閉じない。

## 10. GDN accepted-boundary restore device kernel

§4.3 の「accepted-boundary の GDN/recurrent restore を device 側で行う kernel が無い」
問題に対する第一段。既存 Host `restore_gdn_spec_history()` は
`history[accepted] → pool[slot]` の D2D copy（conv = bf16、recurrent = float）であり、
これを `hipMemcpyAsync` から AQL kernel へ置き換えた。

### 10.1 kernel

新規 MCU kernel `phaseshift_gpu_mcu_gdn_spec_restore`
（`src/phaseshift/models/qwen35/kernels/optimized/gdn/spec_restore.hip`）。

- input: conv/recurrent の pool base と history base、slot stride / history stride、
  `sequence_slot`、`history_row`（= accepted boundary）、conv/recurrent elems。
- output: live な conv state / recurrent state。
- dst = pool + slot * slot_stride、src = history + history_row * history_stride を
  grid-stride で D2D copy する。`history_row = accepted` という mapping は既存 Host と同一。
- Host `hipMemcpyAsync` を使わない。

`GdnSpecRestoreArgs`（80 byte）と `McuGdnSpecRestoreInvocation`（80 byte）を cross
static_assert で一致させ、recipe `kMcuKernargRecipeGdnSpecRestore`、FSM writer、
registry（`McuCompiledVariantKind::GdnSpecRestore`、blob `gdn_spec_restore`）、
embedded kernel entry を追加した。

### 10.2 qualification

新規 required test `test_gpu_mcu_gdn_verify_restore`（`gpu1;gpu_mcu;required`）は、
4 つの (slot, accepted) case（(0,0)/(1,1)/(2,4)/(3,8)）を 1 plan で実行し、
pool の結果を Host の leading-copy reference と byte 比較する。各 slot の未書き込み
tail が保持されることも確認する。

これは CB11 の GDN accepted-boundary restore を満たす。skeleton への restore node 予約
（accepted count 参照 → restore）は未完了のため Gate は閉じない。

## 11. sampling MCU entrypoint（greedy）

§4 の「sampled_tokens までを MCU/AQL で生成できない」問題に対する第一段。既存の
optimized sampling kernel を再利用し、greedy の MCU entrypoint を追加した。

### 11.1 kernel

`sampling.hip` の `argmax_single_impl` の body を `__device__ argmax_single_body()` へ
切り出し、既存 kernel と新規 AQL wrapper の両方が同じ body を呼ぶ。

新規 `extern "C" __global__ phaseshift_gpu_mcu_argmax_f32(ArgmaxF32Args args)`。
`ArgmaxF32Args` をそのまま AQL kernel 引数に使う。greedy は
`DeviceSamplingParams::mode = Greedy` で argmax 経路を通る。stochastic / top-p / top-k の
exhaustive support は CB14/F0 へ回す（既存 CPU/HIP backend は変更しない）。

`McuArgmaxF32Invocation`（56 byte）を `ArgmaxF32Args` と cross static_assert で一致させ、
recipe `kMcuKernargRecipeArgmaxF32`、FSM writer、registry
（`McuCompiledVariantKind::SamplingArgmaxF32`、blob `sampling`）、embedded kernel entry を
追加した。

### 11.2 qualification

新規 required test `test_gpu_mcu_sampling_argmax`（`gpu1;gpu_mcu;required`）は、
3 row / vocab 4096 の logits を greedy で sampling し、FSM の AQL path の
`sampled_tokens` を Host の argmax reference（tie-break は最小 token）と比較する。
tie case も含める。

これは CB11 の sampling/output closure の kernel 部分を満たす。skeleton への sampling
node 予約は §15 で完了した（`KernelId::SAMPLING` → `argmax_f32` node）。lm-head の明示
node 化と completion marker の順序保証は未完了のため Gate は閉じない。

## 12. canonical 26-row mixed plan を persistent bridge で実行

§4.1 の bridge blocker に対する統合第一段。canonical geometry（Slot0 DECODE 1 /
Slot1 DECODE 1 / Slot2 SPEC_VERIFY 8 / Slot3 PREFILL 16 = 26 rows）の mixed plan を、
`McuDecodeRuntime`（`ExternalPersistentMcu`）が準備し、`GpuMcuPersistentMcu` が
prepared FSM を drive して実行する経路を成立させた。

### 12.1 経路

1. `compile_mcu_plan()` が mixed attention を 2 physical region（decode+verify 10 rows、
   prefill 16 rows）へ分解する。prefill 16 rows は `kPagedAttentionPrefillMinRows` 未満の
   ため decode kernel（`Bf16FullHead`）で serving される。
2. `McuDecodeRuntime::prepare_plan()` が embedded registry から code object を load し、
   variant / retained packet / invocation table を device へ upload する。
   `ExternalPersistentMcu` では `fsm.start()` を呼ばない。
3. `GpuMcuPersistentMcu::configure_batch_plan()` + `configure_execution()` に
   `runtime.execution_state()` を渡し、`gpu_mcu_execution_step()` が
   `mcu_run_once(execution_fsm)` を呼んで plan 全体を 1 step で実行する。

Host は plan compile / upload のみ。実行中の kernel launch / 同期 / readback は無い。

### 12.2 qualification

新規 required test `test_gpu_mcu_mixed_batch_execution`（`gpu1;gpu_mcu;required`）は、
canonical 26-row の mixed plan を Host reference（`launch_paged_attention_physical`）と
byte exact 比較する。

- region 数 2、region rows 10 + 16 = 26
- persistent MCU の `batches_dispatched >= 1` / `batches_completed >= 1` /
  `batches_failed == 0`
- prepared FSM の `plans_started >= 1` / `dispatches_committed >= 3`（decode region +
  prefill region + completion marker）/ fault 0
- output は Host と byte exact（diffs 0）

これは CB11 の canonical mixed batch の attention region と persistent bridge を満たす。
full chain（linear / GDN / KV / lm-head / sampling を 1 plan に統合）と per-loop geometry
change、commit 統合（`batches_committed`）は未完了のため Gate は閉じない。

### 12.3 注意（legacy default stream）

persistent kernel 起動後の同期 `hipMemcpy`（legacy default stream）は persistent kernel と
暗黙同期してブロックする。readback は persistent を `request_stop` + `wait_stopped` して
から行う。

## 13. GPU-side invocation patch binder（per-loop geometry）

§4.1 の「loop ごとに変わる mixed geometry をどう表現するか」に対する device 側の overlay。
static skeleton は最大 geometry で準備したまま、row-global invocation の行数を loop ごとに
device 側で書き換える。Host は loop ごとの再 compile / 再 upload をしない。

### 13.1 binding ABI

`McuInvocationPatch`（16 byte、`alignas(16)`、offset static_assert 付き）を追加した。

- `target`: 書き換える u32 field の device address。
- `source`: `ActualRows` / `NumRequests` / `NumOutputs`（`McuInvocationPatchSource`）。

`gpu_mcu_bind_execution_plan(context, patches, patch_count)`
（`include/phaseshift/runtime/gpu_mcu/binding/plan_binder.h`）が各 patch に
`context` の該当値を書き、`__threadfence_system()` で publish する。host でも呼べるよう
`__host__ __device__`。

### 13.2 反映経路

- `GpuMcuPersistentState` / `GpuMcuPersistentStartConfig` に `execution_patches` /
  `execution_patch_count` を追加し、capture / restore へ反映。
- `GpuMcuPersistentMcu::configure_execution_patches()` で patch table を設定。
- `gpu_mcu_execution_step()` が `mcu_run_once()` の直前に
  `gpu_mcu_bind_execution_plan(context, execution_patches, execution_patch_count)`
  を呼ぶ。loop ごと（`batch_ready_epoch` が変わるたび）に適用される。
- `McuDecodeRuntime::prepare_plan()` が `psq4_multi` の `rows` と `gdn_conv1d` の
  `actual_rows` を対象に patch table を構築し、`row_global_patches()` /
  `row_global_patch_count()` で公開する。

### 13.3 qualification

新規 required test `test_gpu_mcu_plan_binder`（`gpu1;gpu_mcu;required`）。

- host unit: `McuInvocationPatch` 3 種（ActualRows / NumRequests / NumOutputs）を
  実 ABI で適用し、値と再 bind を確認。
- persistent integration: prepared marker FSM に対し、standalone な device `rows` field を
  対象とする patch を設定。loop 1 で `actual_rows = 26`、loop 2 で `actual_rows = 3` に
  更新し（`batch_ready_epoch` を進める）、再 prepare なしで field が 26 → 3 に
  書き換わることを確認。fault 0。

これは CB11 の per-loop geometry binding の device 経路を満たす。実 operator の patch を
executor から `configure_execution_patches` へ渡す統合、および mixed の autonomous
multi-loop は未完了のため Gate は閉じない。

## 14. multi-row real operator chain を persistent bridge で実行

§4.1 / §12 の統合を、attention だけでなく実 multi-operator に広げた。26-row の
RMS_NORM + ACTIVATION_QUANTIZE_W4A8 + LINEAR_PSQ4（row-block）を 1 plan として
compile / prepare し、`GpuMcuPersistentMcu` が drive する。

### 14.1 経路

1. `compile_mcu_plan()` が 3 dispatch を 3 physical dispatch（rmsnorm / e4m3 quantize /
   row-block linear）+ completion marker へ compile する。
2. `McuDecodeRuntime::prepare_plan()` が 3 kernel を embedded registry から load し、
   `psq4_multi` の `rows` を対象とする row-global patch を 1 件構築する。
3. `GpuMcuPersistentMcu` に batch plan / execution / patches を configure し、
   `gpu_mcu_execution_step()` が `mcu_run_once()` の前に patch を適用して実行する。

### 14.2 qualification

新規 required test `test_gpu_mcu_multirow_chain_bridge`（`gpu1;gpu_mcu;required`）。

- Host reference（`try_launch_rmsnorm` / `try_launch_activation_quantize_a8` /
  `try_launch_linear_psq4`）と persistent MCU の output を 26-row で byte exact 比較。
- 2 loop 目で `actual_rows = 13` に更新（`batch_ready_epoch` を進める）。device binder が
  `psq4_multi.rows` を 13 に書き換え、再 prepare なしで先頭 13 row が reference と一致。
- patch 1 件、`plans_started >= 2`、fault 0。

これは CB11 の「linear + rmsnorm + quantize を 1 plan で、per-loop geometry を device 側で
bind する」を満たす。attention / KV / GDN を同一 plan に統合した mixed canonical、および
executor からの供給と commit 統合は未完了のため Gate は閉じない。

## 15. mixed canonical に KV append + linear + attention + GDN + sampling を統合

§12（mixed attention）と §14（multi-row linear chain）と §16（GDN chain）を 1 つの Program
に統合し、さらに SAMPLING node を追加した。canonical 26-row mixed batch（DECODE 1 +
DECODE 1 + SPEC_VERIFY 8 + PREFILL 16）に対し、1 つの plan が RMS_NORM +
ACTIVATION_QUANTIZE_W4A8 + LINEAR_PSQ4 + KV_APPEND + PAGED_ATTENTION（2 region）+
GDN_RESET + STATEFUL_CAUSAL_CONV1D + GDN_RECURRENCE + SAMPLING を 10 physical dispatch
として実行し、`sampled_tokens` を device 側で生成する。

### 15.1 経路

- Program: rmsnorm → quantize → linear（1024 out）、KV_APPEND（k/v 1024）→ attention
  （q 4096、2 region）、GDN conv1d（conv_dim 8192）→ recurrence（q/k 2048、v 4096、
  a/b 32 → 4096）、logits（1024）→ SAMPLING（greedy）を同一 plan に持つ。state は
  KV_CACHE / GDN_CONV_STATE / GDN_RECURRENCE_STATE。
- `compile_mcu_plan()` の `KernelId::SAMPLING` case が `McuArgmaxF32Invocation` を emit し、
  `argmax_f32` MCU entrypoint（§11）を再利用する。sampling params は
  `ctx.batch_context->output_sampling_params`、`outputs` は `batch_context->num_outputs`。
- sampling の logits は `OUTPUT_ROWS`（compact）で、`sampled_tokens` は token row で index
  される。`McuArgmaxF32Invocation::output_rows` が output slot → token row を写像し、
  `sampled_tokens[output_rows[slot]]` へ書く（null なら identity）。commit の
  `gpu_mcu_commit_read_token()` は `sampled_tokens[output_rows[...]]` を読むため、この
  写像で両者が一致する。
- `compile_mcu_plan()` が GDN state を触る range に対して `GDN_RESET` node を先頭に emit。
- Host は plan compile / prepare のみ。実行は `GpuMcuPersistentMcu` が
  `gpu_mcu_execution_step()` で `mcu_run_once()` を呼ぶ。
- context / requests は host-mapped（§16.3）。GDN state は Host / MCU で別 buffer。

### 15.2 qualification

required test `test_gpu_mcu_mixed_linear_attention_bridge`（`gpu1;gpu_mcu;required`）。

- Host reference（`try_launch_*` + `launch_paged_attention_physical`）と persistent MCU の
  linear / attention / GDN conv1d / GDN recurrence output をそれぞれ byte exact 比較。
- `sampled_tokens` を CPU の greedy argmax reference と比較（Host readback で accepted を
  決めない）。
- logits は compact（`kOutputs = 11`）で、`output_rows = {0..9, 25}` により scattered な
  token row へ書く。CPU reference も同じ写像で照合する。
- physical dispatch 10（rmsnorm + quantize + linear + kv_append + attention 2 region +
  gdn reset + conv1d + recurrence + sampling）+ marker。
- fault 0。

これは CB11 の「KV append / linear / attention / GDN を 1 plan で実行し、`sampled_tokens`
までを MCU 内で生成する」を満たす。lm-head の明示 node 化、executor からの供給、commit
統合、mixed autonomous multi-loop は未完了のため Gate は閉じない。

## 16. GDN chain を persistent bridge で実行

§4.3 の GDN 部分を、compiler の GDN path ごと persistent bridge へ接続した。
canonical 26-row mixed batch に対し、`GDN_RESET` + `STATEFUL_CAUSAL_CONV1D` +
`GDN_RECURRENCE` を 1 plan として compile / prepare し、`GpuMcuPersistentMcu` が drive する。

### 16.1 経路

- Program は `GDN_CONV_STATE` / `GDN_RECURRENCE_STATE` を持ち、conv1d（input 8192 →
  output F32 8192）と recurrence（q/k F32 2048、v BF16 4096、a/b BF16 32 → out F32 4096）
  を dispatch する。
- `compile_mcu_plan()` は GDN state を触る range に対して `GDN_RESET` node を自動で
  先頭に emit する（`prefix_length == 0` の request の conv/recurrent state を zero 化）。
- `McuDecodeRuntime::prepare_plan()` が conv1d / recurrence / reset の code object を load
  し、conv1d の `actual_rows` を対象とする row-global patch を構築する。
- `GpuMcuPersistentMcu` が `gpu_mcu_execution_step()` で plan 全体を実行する。

### 16.2 qualification

新規 required test `test_gpu_mcu_gdn_chain_bridge`（`gpu1;gpu_mcu;required`）。

- Host reference（`try_launch_gdn_conv1d` / `try_launch_gdn_recurrence`）と persistent MCU
  の conv1d output / recurrence output をそれぞれ byte exact 比較。
- GDN state は Host / MCU で別 buffer に持つ。reset が zero 化する prefill slot は
  Host 側も zero に揃える。
- physical dispatch 4（reset + conv1d + recurrence + marker）。
- fault 0。

### 16.3 注意（device 参照される context / requests）

`GDN_RESET` と `GDN_CONV1D` は `DeviceBatchContext` / `DeviceRequestDescriptor` を device
で参照する。compile 時の resolver は host で同じ descriptor を読むため、context と requests
は host-mapped（managed）にする必要がある。raw `hipMalloc` の device buffer では
resolver の host 読みが、host の通常メモリでは kernel の device 読みが fault する。

これは CB11 の「GDN（conv1d + recurrence）が mixed plan で persistent bridge で実行される」
を満たす。lm-head / sampling の統合、executor からの供給、commit 統合、mixed autonomous
multi-loop は未完了のため Gate は閉じない。

## 17. mixed batch commit（decode + decode + verify + prefill を 1 transaction）

§15 の mixed plan が生成した `sampled_tokens` を、canonical の 4 request
（DECODE 1 + DECODE 1 + SPEC_VERIFY 8 + PREFILL 16 = 26 rows）に対して
**1 回の `gpu_mcu_commit_active_batch()`** で commit できることを確認した。

### 17.1 経路

- `gpu_mcu_commit_active_batch()` は `context->num_requests` 個の request を 1 transaction で
  走査し、request ごとに `execution_class` に応じて slot を進める。
  - DECODE: `row_count == 1` を要求し、1 token を stage する。
  - SPEC_VERIFY: `verify_committed_counts[i]` を accepted とし、
    `sequence_length = prefix_length + accepted`、`verify_committed_count` を記録、accepted
    個の token を stage する。accepted が `row_count` または
    `kGpuMcuMaxVerifyCandidates` を超える場合は fail closed。
  - PREFILL: `prefill_position` / `sequence_length` / `committed_position` を進め、
    prompt を満たした chunk が 1 token を stage して phase を decode へ移す。
- token は `output_rows[descriptor.output_index + j]` を経由して
  `sampled_tokens[row]` から読む（Host readback ではない）。
- 成功した commit は `decode_input_token` を slot binding へ書き戻し、次 loop の decode 入力を
  供給する。

### 17.2 qualification

required test `test_gpu_mcu_batch_commit`（`gpu1;gpu_mcu;required`）の mixed batch case。

- 4 slot を claim し、2 decode + 1 verify（accepted 8）+ 1 prefill（final chunk）を
  1 transaction で commit。
- `committed_tokens == 11`、`committed_requests == 4`。
- decode slot は `sequence_length == 17`、verify slot は `sequence_length == 16` と
  `verify_committed_count == 8`、prefill slot は `prefill_position == 16` で phase が decode へ。
- stage された token（decode は 2000/2001、verify は 2002..2009、prefill は 2010）と
  binding の `decode_input_token` を検証。
- fault 0。

これは CB11 の commit integration（mixed batch の accepted count を含む 1 transaction
commit）の commit 部分を満たす。execution step からの供給、mixed autonomous multi-loop、
lm-head の明示 node 化は未完了のため Gate は閉じない。

## 18. execution step からの mixed commit 統合

§15 の prepared plan を、request runtime + batch planner + batch binding + execution
step + commit の production 経路で実行し、canonical mixed batch を 2 loop commit する
ことを確認した。

### 18.1 経路

- 4 slot（DECODE / DECODE / SPEC_VERIFY / PREFILL）を request runtime に渡す。
  `gpu_mcu_scheduler_boundary()` が runnable snapshot を作り、
  `gpu_mcu_build_batch_geometry()` が 26-row context を構成し、
  `gpu_mcu_bind_batch_io()` が `num_outputs` / `output_rows` / token / sampling を
  束ねて `batch_ready_epoch` を進める。
- `gpu_mcu_execution_step()` が `gpu_mcu_bind_execution_plan()` で per-loop patch を
  device 側で適用してから `mcu_run_once()` を呼び、成功時に
  `gpu_mcu_commit_active_batch()` で slot を commit する。Host は loop ごとに plan を
  再 compile / 再 upload しない。
- loop ごとの再 binding は `rebuild_row_global_patches()` が用意する。psq4_multi /
  gdn_conv1d の rows は `ActualRows`、sampling の outputs は `NumOutputs`、verify_accept の
  request_count は `NumRequests` を loop ごとに書く。
- slot の `max_sequence_length` は batch planner の `sequence_length <= max_sequence_length`
  検査を満たす必要がある（decode の `committed_position` より大きいこと）。
- `max_new_tokens` を slot ごとに調整すると loop 数を決められる（decode/prefill 2、
  verify 9 で 2 loop 後に全 slot が terminal）。

### 18.2 qualification

required test `test_gpu_mcu_mixed_linear_attention_bridge`（`gpu1;gpu_mcu;required`）の
phase 2。

- loop 1 の geometry は 26 rows / 4 requests（decode 2、verify 1、prefill 1）/ 11 outputs。
- loop 1 で `committed_tokens == 11`、verify slot は `verify_committed_count == 8`。
- loop 2 は commit 後の 4 decode（4 rows）で `committed_tokens == 4`。
- `batches_committed == 2`、fault 0。decode slot は `sequence_length == 102`、
  verify だった slot は `109`、prefill だった slot は `17`。
- sampling node が書いた `sampled_tokens[output_rows[slot]]` を commit が読む。
- attention region の rows は `AttentionRegionRows` patch で loop ごとに再 binding され、loop 2 の
  decode region は 4 rows、prefill region は 0 rows になる（§24）。

これは CB11 の「PREFILL / DECODE / SPEC_VERIFY が 1 つの `DeviceBatchContext` に共存し、
1 plan として execution step で実行され、commit される」を満たす。accepted count は §19 で
plan 内の device 計算に置き換えた。production executor からの供給、mixed autonomous
multi-loop、lm-head の明示 node 化は未完了のため Gate は閉じない。

## 19. verify accepted count を plan 内で device 計算

§18 では accepted count を test が `execution_verify_counts` に与えていた。これを plan の
node として device 計算するようにした。

### 19.1 経路

- `KernelId::VERIFY_ACCEPT` を追加し、`compile_mcu_plan()` が
  `McuVerifyAcceptPrefixInvocation` を emit する。inputs は candidates / sampled_tokens /
  candidate_counts、output は committed_counts。`max_candidates` は `rotary_dim` で渡す。
- `VerifyAcceptPrefixArgs` に `requests`（`DeviceRequestDescriptor*`）と `output_rows` を
  追加し、`phaseshift_gpu_mcu_verify_accept_prefix` が
  `sampled_tokens[output_rows[descriptor.output_index + i]]` と `candidates[i]` を比較して
  accepted prefix を求める row-indexed path を持つ（null なら従来の compact stride path）。
- accepted count は `sampled_tokens[output_rows[...]]` を読む。これは sampling node が書いた
  場所と同じであり、commit が `gpu_mcu_commit_read_token()` で読む場所とも同じ。
- `output_index == 0xFFFFFFFF` または `execution_class != SPEC_VERIFY` の request は 0 を書く。
  `DeviceRequestDescriptor::output_index` の既定値が `0xFFFFFFFF` のため、この guard が無いと
  row-indexed path が out-of-bounds になる。

### 19.2 qualification

required test `test_gpu_mcu_mixed_linear_attention_bridge`（`gpu1;gpu_mcu;required`）。

- Program に VERIFY_ACCEPT dispatch（candidates / sampled_tokens / candidate_counts →
  committed_counts、`rotary_dim = 8`）を追加。physical dispatch は 9 + reset + marker。
- phase 2 は `execution_verify_counts` に plan の committed_counts output を渡す。
- `committed_host[2] == 8`（verify request の accepted prefix を device が計算）、
  非 verify request は 0。
- commit は `committed_tokens == 11`、`batches_committed == 1`。

これにより §18 の「test が accepted count を与える」状態を解消した。production executor からの
供給、mixed autonomous multi-loop、lm-head の明示 node 化は未完了のため Gate は閉じない。

## 20. production executor が persistent MCU へ batch を供給

§4.1 の blocker（real Program Plan を persistent MCU で走らせる bridge が無い）を解消した。
executor の MCU 経路が host stream controller ではなく persistent MCU を駆動する。

### 20.1 経路

- `McuDecodeRuntimeOptions::driver_mode = ExternalPersistentMcu` で executor の runtime を作る。
  `prepare_plan()` は FSM を start せず、prepared FSM を `GpuMcuPersistentMcu` へ貸し出す。
- `GpuMcuPersistentMcu::configure_commit_slots()` を追加し、request runtime / batch planner を
  enable せずに commit 用の `GpuMcuSlotState` / `GpuMcuSlotBinding` table だけを設定できるように
  した。`request_runtime_enabled == 0` のため `gpu_mcu_scheduler_boundary()` は早期 return し、
  executor が構築した `DeviceBatchContext` を上書きしない。
- `execute_mcu_hybrid()` は body range の plan を compile した後、
  `configure_batch_plan` / `configure_execution` / `configure_execution_patches` /
  `configure_commit_slots` で executor の buffer（context / requests / row slots / row
  positions / sampled tokens / patches）を供給し、`batch_ready_epoch` を進めて
  `start()` → commit 待ち → `request_stop()` → `wait_stopped()` する。
- commit slot table は `ScheduledBatch` の `DeviceRequestDescriptor` から stage する。
  `handle.slot` は physical sequence slot なので `SequenceSlotPool::max_sequences()` 分を確保する。
- `wait_stopped()` 成功時に `launched_` を clear し、同じ `GpuMcuPersistentMcu` を再 configure して
  次 batch で再 start できるようにした。
- `decide_decode_backend()` に `persistent_ready` を追加。persistent 実行が準備できている場合だけ
  MultipleRequests / NotSingleToken / NotSingleRow / PrefillPresent の rejection を外し、mixed
  batch を MCU 経路へ通す。`SpeculativeVerify` は executor の Program が VERIFY_ACCEPT を
  持たないため引き続き reject する。

### 20.2 qualification

optional test `test_gpu_mcu_production_decode`（`gpu1;optional;external_files`、
`PHASESHIFT_MCU_E2E=1`）。実 model（Qwen3.5-4B）で host backend と MCU backend を同条件で
実行し、`final_hidden` を byte 比較する。

- 単一 decode: host と MCU が byte exact、`mcu_run_count >= 1`、`host_fallback_count == 0`。
- mixed batch（DECODE 1 + PREFILL 64）: host と MCU が byte exact。
- MCU commit が `batches_committed >= 1` を満たす。

これにより production executor からの供給が成立した。残る gap は次節のとおり。

## 21. 未完了の gap（2026-09 時点）

- verify accepted count を production executor の Program から供給する経路。executor の Program は
  `VERIFY_ACCEPT` node を持たず、lowering にも verify accept が無い。したがって executor 経由の
  SPEC_VERIFY batch はまだ MCU 経路で正しく commit できない。
- attention region 形状の per-loop 化（§7 の dynamic node binding の `workgroup_count_*` を
  使った variant grid override と region kind の明示）。
- `scheduled_batch` の VERIFY guard 解除は上記 verify 供給が成立してから行う。

## 22. lm-head を MCU plan に統合（OUTPUT_GATHER）

`OUTPUT_GATHER` の MCU entrypoint を追加し、executor の MCU plan を layer body から
program 全体へ拡張できるようにした。単一 output の batch では embedding から sampling までが
1 plan で走り、`sampled_tokens` が MCU 内で生成される。

### 22.1 経路

- `output_gather.hip` に `phaseshift_gpu_mcu_output_gather_bf16(OutputGatherBf16McuArgs)` を
  追加。`OutputGatherBf16McuArgs` は input / output / output_rows / input_row_stride /
  output_row_stride / num_outputs / features を持ち、`McuOutputGatherBf16Invocation` と
  ABI が一致する（`compile_mcu_plan()` の cross static_assert）。
- `McuCompiledVariantKind::OutputGatherBf16`、`kMcuKernargRecipeOutputGatherBf16`、
  `mcu_write_output_gather_kernarg()`、`McuCompiledPlan::output_gather` を追加。
  grid は `(ceil(features / 256), num_outputs)`。
- HSACO は `gpu_mcu_hsaco_output_gather` として build し、`output_gather` blob として
  embed する。
- `execute_mcu_hybrid()` はまず `[0, dispatch_count)` を compile し、成功すれば Host
  prefix / suffix を出さない。失敗時は従来どおり `[region_begin, body_end)` に fallback する。
  `McuDecodeState::full_plan` がどちらを使ったかを示す。

### 22.2 qualification

optional test `test_gpu_mcu_production_decode`（`PHASESHIFT_MCU_E2E=1`）。

- 単一 decode: `full_plan == true`（embedding + layer + gather + final_norm + lm-head +
  sampling が 1 plan）、host と MCU の `final_hidden` が byte exact、`sampled_tokens` が一致。
- mixed batch（DECODE + PREFILL、2 output）: lm-head の `LINEAR_BF16` が `ExactRows`
  （`exact_rows == 1`）を要求するため full plan は compile できない。body range に fallback し、
  sampling は Host suffix のまま。host と MCU の `final_hidden` は byte exact。

## 23. 未完了の gap（2026-09 時点、更新）

- multi-output batch の lm-head。`LINEAR_BF16` の `ExactRows` が `rows == 1` を要求するため、
  output が 2 以上の batch では full plan にできない（§27 で扱う）。
- `scheduled_batch` の VERIFY guard 解除は verify 供給が成立してから行う。
- mixed batch の GDN recurrence history capture。recurrence の mixed 経路は
  `gdn_recurrence_wmma_body`（汎用）を使い、この body は per-row の history を materialize
  しない。conv1d 側の capture は §26 で対応済み。

## 24. attention region 形状の per-loop 化

§18 の 2 loop 目は commit 後の 4 decode（4 rows）になるが、attention region は compile 時の
26-row 形状のままだった。loop ごとに region の `rows` を device 側で再 binding するようにした。

### 24.1 経路

- `resolve_paged_attention_physical()` が各 region に `PhysicalPagedAttentionKernel::row_begin`
  を残す。`compile_mcu_plan()` は region ごとに
  `McuAttentionRegion{invocation_kind, invocation_index, row_begin, rows}` を plan へ記録する
  （kind 0 = `attention_paged`、1 = split、2 = reduce）。
- `McuInvocationPatchSource::AttentionRegionRows` を追加し、`McuInvocationPatch` に
  `param`（低 16 bit = `row_begin`、高 16 bit = `rows`）を持たせた。source は context の
  request 列のうち region の row 範囲と重なる行数を合計する。
- `McuDecodeRuntime::rebuild_row_global_patches()` が region ごとに invocation の `rows` field を
  指す patch を追加する。`mcu_run_once()` の dispatch 直前に `gpu_mcu_bind_execution_plan()` が
  loop ごとの値を書く。
- attention kernel（decode full-head / split / reduce / prefill）に `row >= a.rows` guard を追加。
  region の grid は compile 時のまま（最大形状）で、余分な block は guard で return する。
  Host 経路では `rows` が実 row 数なので guard は no-op。

### 24.2 qualification

- `test_gpu_mcu_plan_binder`（`gpu1;gpu_mcu;required`）: `AttentionRegionRows` が decode region
  10 / prefill region 16 を返し、context が 4 decode に縮むと decode region が 2、prefill region が
  0 になることを確認。
- `test_gpu_mcu_mixed_linear_attention_bridge` / `test_gpu_mcu_mixed_attention_regions` /
  `test_gpu_mcu_mixed_batch_execution` が回帰なく PASS（canonical 26-row では patch が
  compile 時と同じ値を書く）。

## 25. MCU runtime execution epilogue

CB11 最終設計（Decision 1 / 2）に従い、VERIFY acceptance を model graph ではなく
runtime epilogue として `McuCompiledPlan` へ付加する。

### 25.1 runtime epilogue

- `McuRuntimeEpilogue`（`verify_accept_capable` / `gdn_restore_capable` / variant /
  invocation / node index）を `McuCompiledPlan::epilogue` に持たせる。
- `compile_mcu_plan()` は model dispatch の後・completion marker の前に epilogue node を積む。
  `McuPlanCompileOptions` に `verify_sampled_tokens` / `verify_committed_counts` /
  `verify_max_candidates` / `verify_conv_history` / `verify_recurrent_history` と stride を追加し、
  runtime が供給したときだけ node を emit する。
- `lower_to_primitives()` は変更しない。Program external I/O 数も fingerprint も不変。

### 25.2 context-native VERIFY_ACCEPT

- `phaseshift_gpu_mcu_verify_accept_batch(VerifyAcceptBatchArgs)` を追加。
  args は `{context, sampled_tokens, committed_counts, max_candidates}`。
  candidate は `context->token_ids + descriptor.row_begin`、candidate count は
  `descriptor.row_count`、sampled token row は `context->output_rows[output_index + i]` から解決する。
  `execution_class != SPEC_VERIFY` の request は `committed_counts[request] = 0`。
- `McuVerifyAcceptBatchInvocation` / `kMcuKernargRecipeVerifyAcceptBatch` /
  `McuCompiledVariantKind::VerifyAcceptBatch` を追加。既存の
  `phaseshift_gpu_mcu_verify_accept_prefix` は不変。

### 25.3 committed_counts と GDN restore

- committed count の buffer は runtime-owned の `McuDecodeState::verify_counts`。
  `configure_execution()` の `execution_verify_counts` と同一 buffer で Host readback しない。
- `phaseshift_gpu_mcu_gdn_spec_restore_from_counts(GdnSpecRestoreFromCountsArgs)` を追加。
  accepted は `committed_counts[request]`、sequence slot は
  `context->row_sequence_slots[descriptor.row_begin]`。source は既存 Host の
  `restore_gdn_spec_history(pool, slot, history, accepted)` と一致する。

### 25.4 qualification

- `test_gpu_mcu_verify_accept`（`gpu1;gpu_mcu;required`）: DECODE + SPEC_VERIFY 8 + PREFILL の
  `DeviceBatchContext` を渡し、accepted 0 / 1 / 4 / 8 を device が算出し、非 verify request の
  count は 0 のままであることを確認（Host accepted injection なし）。
- `test_gpu_mcu_mixed_linear_attention_bridge` / `test_gpu_mcu_production_decode`
  （実モデル、`PHASESHIFT_MCU_E2E=1`）が回帰なく PASS。

## 26. mixed batch の GDN conv1d verify history capture

§33-35 に従い、batch-global `ExecutionRole` ではなく `DeviceRequestDescriptor::execution_class`
で history capture を判定する。

### 26.1 経路

- `McuInvocationPatchSource::VerifyRequests` を追加。`context->num_verify_requests != 0` を
  1 / 0 で返す。
- `GdnConv1dArgs` / `GdnConv1dAqlArgs` / `McuGdnConv1dInvocation` に `capture_verify_only` を
  追加（AQL offset 116、struct は 120 バイトのまま）。`GdnRecurrenceArgs` /
  `McuGdnRecurrenceInvocation` にも追加（args は 192 バイト）。
- conv1d kernel は `row - descriptor.row_begin` を verify-local row とし、
  `capture_verify_only != 0` のとき `execution_class == SPEC_VERIFY` の row だけを history へ
  store する。`capture_verify_only == 0` のときは従来どおり `row < capture_rows`（dflash 経路は
  不変）。recurrence の `decode_rows_exact` も同じ gating を持つ。
- `resolve_gdn_conv1d_physical()` / `resolve_gdn_recurrence_args()` は
  `verify_exact_active(ctx) || context->num_verify_requests != 0` のとき history を bind し、
  `capture_verify_only` を立てる。
- `McuDecodeRuntime::rebuild_row_global_patches()` が gdn_conv1d / gdn_recurrence の
  `capture_verify_only` を `VerifyRequests` source の patch として追加する。loop ごとに
  `gpu_mcu_bind_execution_plan()` が書く。

### 26.2 qualification

- `test_gpu_mcu_mixed_gdn_verify_history`（`gpu1;gpu_mcu;required`）: DECODE 1 + SPEC_VERIFY 8 +
  PREFILL 16 の mixed batch で conv1d を走らせ、`capture_verify_only = 1` の history local row i が
  legacy capture の batch row `i + 1` と一致し、capture 対象外の row が sentinel のままである
  ことを確認する。patch source が verify batch で 1、plain batch で 0 を返すことも確認する。
- `test_gpu_mcu_mixed_linear_attention_bridge` / `test_gpu_mcu_production_decode` が回帰なく PASS。
- `test_gpu_mcu_gdn_chain_bridge`（`gpu1;gpu_mcu;required`）: row-global patch 数が gdn conv の
  `actual_rows` + `capture_verify_only` と recurrence の `capture_verify_only` を反映するよう更新。

## 27. BF16 lm-head の output rows dynamic catalog

§20-24 に従い、multi-output batch の lm-head を full plan 化する。

### 27.1 経路

- `phaseshift_qwen35_gemm_bf16_exact_rows_2..16` の `extern "C"` entrypoint を追加し、
  symbol table `kBf16ExactRowsSymbols[16]` を公開する。kernel 本体は既存の
  `gemm_bf16_exact_rows_body<kRows>` を再利用する。
- `McuCompiledVariantKind::Bf16ExactRows2..16` と `kMcuBf16ExactRowsKinds[16]` を追加し、
  registry が kind ごとに `kBf16ExactRowsSymbols[rows-1]` を返す。
- `PhysicalBf16Launch::output_rows_domain`（`dst.row_domain == OUTPUT_ROWS`）を追加。
  compiler は `LINEAR_BF16` の `ExactRows` node について rows 1..16 の variant を
  catalog として plan へ登録し、`McuBf16VariantCatalog{node_index, invocation_index,
  variant_base}` を記録する。variant id は `variant_base + (rows - 1)` で連続していることを
  compile 時に検証し、崩れていれば fail-closed。
- `McuDynamicNodeBinding` に `variant_override`（既定 `0xFFFFFFFF`）を追加し、FSM は
  binding 適用時に override があれば `node.variant_id` を差し替える。
  `McuInvocationPatchSource::Bf16ExactRowsVariant` は `context->num_outputs`
  （0 のとき `actual_rows`）から `variant_base + rows - 1` を返し、16 を超えるときは
  `0xFFFFFFFF` を返して invalid variant として fail-closed にする。
- `McuDecodeRuntime::prepare_plan()` が plan の node 列から `McuDynamicNodeBinding` table を
  構築して `config.dynamic_node_bindings` へ渡し、`rebuild_row_global_patches()` が
  catalog ごとに binding の `variant_override` を指す patch を追加する。
- `OUTPUT_GATHER` の `num_outputs` も `NumOutputs` patch で loop ごとに更新する（gather kernel は
  `i >= args.num_outputs` guard を持つ）。

### 27.2 per-loop 値の source

`decide_decode_backend()` は `persistent_ready` のとき `NotDecode` と `SpeculativeVerify` を
適用しない。executor は batch に SPEC_VERIFY request が含まれるとき `ExecutionRole::Verify` を
立てる（§12 の legacy resolver fallback）。これにより multi-output lm-head が
`ExactRows` selector を通る。

### 27.3 qualification

- `test_gpu_mcu_dynamic_lm_head_rows`（`gpu1;gpu_mcu;required`）: `exact_rows` 1..16 の kernel 出力が
  M=1 GEMV reference と byte-exact であること、symbol table が重複しないこと、
  `Bf16ExactRowsVariant` source が outputs 1/2/4/8/11/16 を `variant_base + rows - 1` へ写像し
  16 超で fail-closed になること、BF16 lm-head の plan が catalog を 1 本 emit し
  rows 1..16 の kind が連続することを確認する。

## 28. SPEC_VERIFY guard の解除

§38 の解除条件（VERIFY epilogue、committed_counts 配線、GDN restore、KV transaction、
multi-output full plan）が揃ったため、`validate_scheduled_batch()` の
「SPEC_VERIFY requires transactional KV support」hard reject を削除する。

- 代わりに §16 の validation を追加する。SPEC_VERIFY は candidate 数が
  `kGpuMcuMaxVerifyCandidates` 以下、`num_output_rows == num_tokens`（candidate ごとに 1 output row）、
  `compute_logits` と `sample` を要求する。
- KV transaction は `GpuMcuBatchCommitView::verify_committed_counts`（= `McuDecodeState::verify_counts`）
  を読み、accepted boundary で `gpu_mcu_verify_resource_commit()` / `abort()` を実行して
  `slot.sequence_length = prefix_length + committed` を確定する。GDN は
  `phaseshift_gpu_mcu_gdn_spec_restore_from_counts` が accepted boundary へ戻す。
