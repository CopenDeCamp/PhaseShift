> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# CB5〜CB8 Autonomous Closed Loop（Gate 記録）

> Status: R&D record。`Execution-Ready Batch` から `Autonomous Closed Loop` までの
> 実装と Gate 検証の記録。現在も有効な contract は
> [docs/developer/gpu_mcu/low_level.md](../../developer/gpu_mcu/low_level.md) を参照。

## 1. revision

| | |
|---|---|
| baseline | `fb168b4f` |
| final | `1f0e65b5` |
| worktree | `.worktrees/gpu-mcu-program-plan-one-layer`（AGENTS.md の `.worktrees/` 配下） |

| commit | 内容 |
|---|---|
| `2d7c550d` | CB4 R/S/T 真因修正（persistent の binding 接続） |
| `9f01e175` | `mcu_run_once` 抽出（単一 controller の土台） |
| `cd1bbc69` | 2 controller framing の訂正 |
| `a38ab79d` | persistent execution 基盤 |
| `2cd374e4` | persistent loop を `micro_fsm.hip` へ移動 |
| `372e39ac` | CB5 dispatch |
| `9f9954ad` | GPU test runner の分散修正 |
| `a175b045` | CB6 transaction commit |
| `20aaefee` | CB7 output ring |
| `0584eb8b` | CB8 自走の再計画 |
| `045ca600` | CB8 統合 test |
| `1f0e65b5` | CB8 scenario 1 / 4 / 6 |

## 2. 前提として修正した 2 件

### 2.1 persistent binding が publish されていなかった

`gpu_mcu_scheduler_consume_one` が `gpu_mcu_apply_control_command` へ binding table を
渡しておらず、SUBMIT claim 時に `gpu_mcu_binding_attach` が呼ばれなかった。binder は
`stale_binding` で必ず不成立となり `batch_ready_epoch` が増えなかった。consume 経路へ
`slot_bindings` / `binding_max_slots` を渡して修正（`2d7c550d`）。

### 2.2 run context の既定値が `boot` だった

`McuSupervisorState` は `boot = 0`, `idle = 1`。`GpuMcuFsmRunContext` の
`supervisor` 既定値を `0` にしたため、集約リセット後の supervisor が `boot` になり、
`mcu_run_once` の idle 判定を外れて **plan を 1 つも dispatch せずに完走 publish**
していた。既定値を `idle` にし、`boot` を `idle` へ正規化して修正（`9f01e175`）。
既存 FSM regression が検出した。

## 3. architecture: controller は 1 つ

RDC（`-fgpu-rdc`）が無効なため device 関数は TU を跨げない。したがって persistent loop
を `mcu_run_once` と同じ `micro_fsm.hip` に置き、別 kernel を起動しない。

```text
Host: SUBMIT / CANCEL / output consume / shutdown

GPU MCU（gpu_mcu_persistent_loop_kernel、唯一の controller）
  boundary        : flush pending output → control ingest → reconcile
                    → snapshot → plan → binding → batch_ready_epoch++
  execution step  : batch_ready_epoch が進んだら fsm の run_request を更新し
                    同一 TU で mcu_run_once() を実行（別 kernel を起動しない）
  commit          : completion 成功時のみ slot progress を進める
  ⇒ scheduler_dirty = 1 で次 loop が再計画（これが自走の核）

mcu_run_once(): plan を 1 回完走する execution primitive。controller ではない
gpu_mcu_fsm_kernel → mcu_control_loop → mcu_run_once: 既存 Host 駆動経路（維持）
```

`GpuMcuFsmRunContext`（`run_consumed` / `write_base` / `dispatch_seq` / counter /
`supervisor`）を `GpuMcuFsmState` に移し、`mcu_control_loop` が entry で集約リセット
することで kernel 起動ごとの fresh を保つ。

### 3.1 却下した方式

「persistent scheduler と persistent FSM を 2 本の persistent kernel として共存させる」
案は採らない。同一 stream では HSA stream order で後発 kernel が永久に起動せず、
別 stream でも同一 CU mask 上で 2 workgroup が同時 resident になる保証が無い。
controller を 1 つにすればこの問題自体が消える。

## 4. CB5 — MCU Batch Dispatch

`configure_execution(fsm, retained_count, batch_ready_epoch, sampled_tokens,
sampled_capacity, verify_counts, verify_capacity)`（start 前のみ）。

`gpu_mcu_execution_step`:

- `batch_ready_epoch == active` → 何もしない
- `actual_rows == 0` → active を進めて idle（busy-spin しない）
- `execution_epoch == UINT64_MAX` → fail-closed（wrap して継続しない）
- `fsm->batch_input = ctx->token_ids` → `fsm->run_request = ++execution_epoch`
  → `mcu_run_once(fsm, lds=nullptr)` → telemetry

test `test_gpu_mcu_batch_dispatch`: dispatch 前後で slot 256 byte が `memcmp == 0`、
worker が束縛 token を消費、Host per-batch launch 0。

## 5. CB6 — Transaction Commit

`batch_commit.h` の `gpu_mcu_commit_active_batch(view)`。

| class | commit |
|---|---|
| PREFILL | `prefill_position` / `sequence_length` / `committed_position` 前進。final chunk で decode へ遷移 + token stage |
| DECODE | sequence +1 / `generated_tokens` +1 / token stage / `binding.decode_input_token` 更新 |
| SPEC_VERIFY | accepted 0..32。`> row_count` または `> 32` は error terminal（fail-closed） |
| stop | `max_new_tokens` / `max_sequence_length` を GPU 側で判定 |
| pending | 上限 32。overflow は該当 slot を `output_blocked` にして fail-closed |

失敗時は commit せず、参加 slot のみ `error` terminal へ遷移。stale handle は commit
しない。二重 commit は `last_committed_execution_epoch`（dispatch 1 回 = commit 1 回）
で構造的に防止。

test `test_gpu_mcu_batch_commit`: PREFILL chunk / final、DECODE、VERIFY 0/8/32 +
overflow、stale generation。

## 6. CB7 — Async Output Ring

`output_ring.h`: 56 byte の `GpuMcuOutputRecord`（`request_handle_bits` / `request_id` /
`token_id` / `token_index` / `flags` / `terminal_reason`）と 256 entry の sequence SPSC
ring（`hipHostMalloc(mapped|coherent)`、64 byte slot、ControlRing と同じ順序規約）。
device が push、host が `try_pop`（blocking しない）。

boundary 先頭で `gpu_mcu_flush_slot_outputs`:

- `pending_count == 0` → `output_blocked` を解除
- ring へ順に push、全成功 → pending 0 / blocked 0
- 途中で full → 残りを pending に保持し **該当 slot のみ** `output_blocked = 1`
- global stop は作らない

test `test_gpu_mcu_output_ring`: 順序・handle・request_id 保存、empty で即 false、
256 token で full → 1 slot のみ blocked、消費後に unblock。

## 7. CB8 — Autonomous Closed Loop

commit 後に `scheduler_dirty = 1` を立て、次 loop が snapshot / plan / binding を
再構築する。これが無いと plan と ready epoch が動かず 1 token で停止する。失敗 batch
でも dirty を立て、terminal slot を次 snapshot から落とす。

test `test_gpu_mcu_autonomous_loop`:

| scenario | 内容 |
|---|---|
| DECODE 自走 | Host は SUBMIT 1 回のみ。PREFILL → commit → DECODE ×3 → `max_new_tokens` terminal。token 列 `500+100n` は `decode_input_token` のフィードバック無しには進まない |
| 1 chunked PREFILL | `max_prefill_rows_per_slot = 1` で 3 token prompt を 3 prefill loop に分割 → decode |
| 4 delayed consumption | loop が収束するまで host は ring を drain しない |
| 6 cancel | 走行中に CANCEL → current transaction 完遂 → 次 loop から除外 → cancelled terminal |

telemetry（実測）: `autonomous_loops > 1`、`batches_dispatched > 1`、
`batches_completed == batches_dispatched`、`batches_committed == successful`、
`batches_failed == 0`。

## 8. sync audit

production hot path の同期は **0**。

- `hipDeviceSynchronize` / 同期 D2H `hipMemcpy` / `hipMemcpyWithStream` D2H:
  `tools/check_mcu_sync.py` clean（gpu_mcu 配下 + persistent test）
- `hipStreamSynchronize` は `GpuMcuPersistentMcu::shutdown()` と
  `GpuMcuFsm::shutdown()` の lifecycle 経路のみ（spec が許容）
- Host polling に inference progress を依存させていない（Host は `try_pop` を
  呼ぶだけ。呼ばなくても capacity 内で progress する）

## 9. tests

| label | 件数 |
|---|---|
| gpu_mcu | 59 / 59 PASS |
| required | 184 / 184 PASS |

新規 required test:

- `test_gpu_mcu_batch_dispatch`（CB5）
- `test_gpu_mcu_batch_commit`（CB6）
- `test_gpu_mcu_output_ring`（CB7）
- `test_gpu_mcu_autonomous_loop`（CB8）

副次: `tests/support/gpu_test_runner` が `budget_gb` 既定 24 のため `-j4` の全 test を
GPU 0 に詰め込んでいた。使用量が最少の device を選ぶよう修正し、4 枚に分散するように
した（`9f9954ad`）。co-residency 干渉の低減も狙う。

## 10. known deferrals

CB5〜CB8 で完了したのは **control loop が GPU 内で閉じることの証明**まで。以下は未着手。

- CB9: continuous admission 完成 / terminal slot release・reuse / generation++ turnover
- CB10: production mixed PREFILL + DECODE + VERIFY physical batch qualification
- CB11: scheduler policy（batch-size table、優先度、fairness）
- CB12: stop / cancel / failure exhaustive qualification
- CB13: PhaseShift-Server production integration
- Radix R0-R8（GPU Radix、refcount、eviction、Tree Sparse）

### 10.1 未検証項目

- CB8 scenario 3（VERIFY 0..32 の **統合**）。commit 層の 0/8/32 と overflow は
  `test_gpu_mcu_batch_commit` で実測済み。execution result からの accepted count
  取得の統合は未実施。
- CB6 の D（fault → ERROR、progress 0）。実装は入っているが test 未実施。

### 10.2 27B-PSQ のカーネル充足（本番 target との乖離）

**現在の MCU 実行系は BF16・単一 request・単一 token の DECODE（transformer body）
専用**であり、Qwen3.8-27B-PSQ では PREFILL / DECODE / SPEC_DECODE のいずれも
そのままでは動かない。詳細な調査結果は本記録とは別に扱うべき production 課題だが、
要点のみ記す。

| フェーズ | 状態 | 主因 |
|---|---|---|
| PREFILL | 不可 | executor の MCU 適格判定が prefill を拒否。MCU compiler も prefill attention を拒否 |
| DECODE (PSQ4) | 不可 | `ActivationQuantizeE4m3K5120` / `Psq4Decode1Bf16U16` の code object が embed されておらず `mcu_kernel_code_object` が unsupported |
| DECODE (BF16, 長 context) | 不可 | q_per_kv = 6 のため split/reduce が `Bf16SplitOther` / `Bf16ReduceOther` になり compiler が拒否 |
| SPEC_VERIFY | 不可 | `speculative_verify` は MCU 対象外。exact GDN variant に MCU kind が無い |
| SPEC_DRAFT (DFlash2) | 対象外 | 手書き executor。`Program` を作らない |

recipe の kernarg writer は 16 個すべて揃っており、カーネル本体（prefill 用 attention
含む）も存在する。欠けているのは **embed 済み code object、plan compiler の variant
受理条件、executor の適格判定、PREFILL の executor 配線**。

## 11. この Gate が証明したこと

> Host が request を 1 回渡した後、GPU MCU だけで
> plan → execute → commit → output → next token → next loop を繰り返せること。

production の性能最適化（queue depth / doorbell / batch-size / 優先度 / speculative
throughput）は CB9 以降。
