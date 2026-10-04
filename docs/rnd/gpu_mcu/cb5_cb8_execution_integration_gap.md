> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# CB5〜CB8 Execution Integration（方式訂正）

> **Status: SUPERSEDED**
> **Superseded by: [docs/rnd/gpu_mcu/continuous_batching_cb5_cb8.md](continuous_batching_cb5_cb8.md)**
>
> この文書は CB5 着手前の調査記録であり、「CB5 以降の実装は未着手」の時点の内容である。
> CB5〜CB8 は実装・検証済みで、結果は後継文書に記録している。当時の文面は書き換えず
> そのまま保持する。現在の contract は
> [docs/developer/gpu_mcu/low_level.md](../../developer/gpu_mcu/low_level.md) を参照。

## 1. baseline

| | |
|---|---|
| baseline revision | `fb168b4f` |
| CB4 前提修正 | `2d7c550d` |
| worktree | `.worktrees/gpu-mcu-program-plan-one-layer`（AGENTS.md の `.worktrees/` 配下） |
| branch | `poc/gpu-mcu-program-plan-one-layer` |

作業ツリーは clean。

## 2. CB5 の前提が成立していなかった（修正済み）

`gpu_mcu_scheduler_consume_one` が `gpu_mcu_apply_control_command(...)` へ
binding table を渡していなかった。

```text
persistent boundary
  -> gpu_mcu_apply_control_command(..., slots, max_slots, nullptr, 0u)   ← binding 未指定
  -> gpu_mcu_apply_submit は bindings == nullptr のとき binding_attach を呼ばない
  -> GpuMcuSlotBinding は zero のまま
  -> gpu_mcu_bind_batch_io の gpu_mcu_binding_valid_for が false（stale_binding）
  -> ready = false
  -> batch_ready_epoch は増えない
```

修正 `2d7c550d`: consume 経路へ `state->slot_bindings` /
`state->binding_max_slots` を渡す。`test_gpu_mcu_persistent_ingress` に
有効化経路の assertion を追加し PASS を実測。

```text
configure_batch_binding -> batch_binding_enabled
boundary bind           -> batch_ready_epoch >= 1
                        -> batch_binding_bound_requests >= 1
                        -> host-mapped token_ids[0] == prompt[0]
```

## 3. controller は 1 つ（初版の誤りの訂正）

初版は「persistent scheduler と persistent FSM が 2 本の persistent kernel
として共存する」ことを前提にし、その co-residency を STOP 条件とした。
これは誤りである。

| 実体 | 役割 | 位置 |
|---|---|---|
| `gpu_mcu_fsm_kernel` | **Host 駆動の packaging**。`<<<1,32>>>` | `micro_fsm.hip:1221` |
| `mcu_control_loop` | plan 実行ロジック**本体**（device function） | `micro_fsm.hip:628` |
| `gpu_mcu_persistent_loop_kernel` | 常駐 scheduler。`<<<1,32>>>` | `persistent_mcu.hip:119` |
| `gpu_mcu_scheduler_boundary` | 常駐 scheduler の 1 loop 分 | `persistent_mcu.h:317` |

`gpu_mcu_fsm_kernel` は `mcu_control_loop(state, lds)` を呼ぶだけの薄い
ラッパーである。Host が毎 plan キックする現行経路では、この包装が必要だった
にすぎない。

自律 runtime では **controller は persistent MCU ただ 1 つ**である。
plan 実行は controller が実行ロジックを**直接呼ぶ**。

```text
persistent MCU loop  (唯一の controller)
    ├─ boundary: consume / plan / bind
    ├─ execution step: 既存 plan 実行ロジックを直接呼ぶ
    ├─ commit
    └─ output
```

2 本目の persistent kernel は不要であり、作りもしない。

## 4. 2 本案を棄却する理由（記録）

初版が想定した「別 kernel を device kick する」案を採らない理由。

- 同一 stream では HSA stream order により後発 kernel が永久に起動しない。
- 別 stream にしても、同一 CU mask 上で 2 workgroup が同時 resident になる
  保証は HSA 仕様に無く、「たまたま動いた」を排除できない。
- そもそも controller を 2 つにする必要が無い（§3）。
- 手順書 §27 は「共存が不安定なら persistent MCU へ execution step を統合する」
  ことを明示的に許容している。

したがって co-residency の実測確認そのものが不要になる。

## 5. 採用方式: 単一 controller + execution step の抽出

`mcu_control_loop` は既に次の構造を持つ。

```text
for (;;) {
    supervisor == idle のとき start signal / run_request を 1 回読む
    request が無ければ sleep して continue
    plan を kMcuNodeEnd まで実行
    completion を待つ
    done / result を publish
    supervisor = idle
}
```

これは

```text
mcu_run_once(state, lds, ctx) -> bool   // 要求を 1 回消費して 1 plan を完走、done publish
mcu_control_loop(state, lds)            // for(;;) { mcu_run_once か idle_sleep }
```

へ機械的に分解できる。`gpu_mcu_fsm_kernel` は `mcu_control_loop` を呼び続ける
（既存 Host 駆動経路を完全維持）。persistent MCU は boundary で
`mcu_run_once` を直接呼ぶ（新規 controller 経路）。

- run を跨ぐ local（`run_consumed` / `write_base` / `dispatch_seq` /
  `region_seq` / 各種 counter）は `McuRunContext` へ移す。
- 既存 FSM test 群（36 本）が回帰網になる。`mcu_control_loop` の挙動を
  byte 単位で変えないことをここで担保する。
- LDS は persistent MCU 側で `retained_count * sizeof(GpuMcuRetainedPacket)`
  以上を確保する。

## 6. 実装順

1. `mcu_run_once` 抽出 + `McuRunContext` 導入。既存 FSM test 全件 PASS を確認。
2. `GpuMcuPersistentState` へ FSM state / retained LDS / `McuRunContext` /
   execution telemetry を追加。
3. boundary の binding 直後に execution step を接続（CB5）。slot progress は
   まだ commit しない。
4. CB7 output ring、CB6 commit、CB8 autonomous loop の順。

## 7. mcu_run_once 抽出（実施済み）

`mcu_control_loop` を `mcu_run_once` と薄い loop wrapper に分解した。

- `mcu_run_once(GpuMcuFsmState*, GpuMcuRetainedPacket*) -> bool` は要求を 1 回
  消費して plan を 1 回完走し、完了 publish を行う。
- `mcu_control_loop` は `for (;;) { stop 判定 -> mcu_run_once -> idle_sleep }`
  の wrapper として既存 Host 駆動経路を完全維持する。
- run を跨ぐ local（`run_consumed` / `write_base` / `dispatch_seq` /
  `region_seq` / 各種 counter / `supervisor`）は `GpuMcuFsmRunContext` として
  `GpuMcuFsmState` に移した。`mcu_control_loop` が entry で
  `GpuMcuFsmRunContext{}` を代入し、kernel 起動ごとの fresh を保つ。
- これにより plan 実行は「呼び出し可能な execution primitive」になった。
  persistent MCU から別 kernel を起動せずに呼べる。production 側の接続
  （CB5）はまだ行っていない。

### 7.1 抽出時に混入した不具合（FSM regression が検出）

`McuSupervisorState` は `boot = 0`, `idle = 1` である。`GpuMcuFsmRunContext` の
`supervisor` 既定値を `0` にしたため、集約リセット後の supervisor は `boot`
になり、`mcu_run_once` の idle 判定を外れた。その結果、初回呼び出しが plan を
1 つも dispatch せずに完走 tail へ落ち、`plans_started` を誤って publish して
いた（`dispatches` は 0 のまま）。

- 検出: 既存 FSM regression（`test_gpu_mcu_micro_fsm` ほか）。device printf で
  「要求消費の print より先に完走 tail の print が出る」ことを観測して確定。
- 修正: `supervisor` の既定値を
  `static_cast<uint32_t>(McuSupervisorState::idle)` にし、`mcu_run_once` で
  `boot` を `idle` へ正規化する。

## 8. 未着手のため deferred

CB5〜CB8 の実装そのものは未着手。以下は全て deferred。

- CB5 MCU batch dispatch（execution bridge）
- CB6 transaction commit（PREFILL / DECODE / VERIFY / failure / stale / cancel）
- CB7 async output ring / backpressure
- CB8 autonomous closed loop
- CB9 continuous turnover
- CB10 mixed physical batch qualification
- CB11 scheduler policy
- CB12 stop / cancel / failure exhaustive qualification
- CB13 server production integration
- Radix R0-R8

CB5〜CB8 が完了したという記述は、この文書のいずれの箇所にも無い。
