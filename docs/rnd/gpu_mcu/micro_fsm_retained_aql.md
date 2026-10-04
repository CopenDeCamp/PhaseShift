> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# GPU-MCU Micro-FSM / Retained AQL PoC

- 目的: GPU-MCU を机上設計ではなく実機で組み始め、どの設計が成立し、どこで問題が出るかを露出させる
- 対象: `phaseshift_gpu_mcu` の低レイヤー上に Micro-FSM / Retained AQL を追加
- 対象外: production Qwen executor への接続、qwen35 への変更、kernel fusion の復活、command ABI の汎用化
- 性能の GO/NO-GO 判定は行わない。成立可否と問題点の記録が目的
- 詳細契約: `docs/developer/gpu_mcu/low_level.md`

## 1. 基点

| 項目 | 値 |
|---|---|
| base revision | `82177ee97ee802acbb45e495e15514062803789f`（de-fusion primitive baseline） |
| worktree | `.worktrees/gpu-mcu-fsm-retained-aql` |
| branch | `poc/gpu-mcu-fsm-retained-aql` |

## 2. Gate 結果

| Gate | 内容 | 結果 |
|---|---|---|
| M0 | baseline instrumentation（既存 test / bench / resource） | PASS |
| M1 | AQL stage / commit 分離 | PASS |
| M2 | compact retained packet（64 byte） | PASS |
| M3 | retained template を LDS 常駐 | PASS |
| M4 | GPU による runtime kernarg 生成 | PASS（条件付き、§6） |
| M5 | minimal micro FSM | PASS |
| M6 | GPU-local completion | PASS |
| M7 | prepared dispatch | PASS |
| M8 | prepare / worker duration の実測 | 実測済み（§8） |
| M9 | LDS vs VRAM retained template A/B | 差なし（§8） |
| M10 | SGPR / VGPR / LDS resource audit | spill 0（§7） |
| M11 | lane0-only vs wave cooperative の境界 | 差なし（§8） |
| M12 | minimal kernel variant registry | PASS |
| M13 | 実 kernel 接続前の停止点 | 本ドキュメント |

## 3. MCU architecture

- supervisor state: boot / idle / running / fault / stopping
- fault code: invalid_plan / invalid_node / invalid_variant / kernarg_slot_unavailable /
  queue_stage_failure / completion_generation_mismatch
- node ABI: `McuPlanNode` 24 byte
  （variant_id / next / kernarg_recipe / completion_slot / value / element_count /
  input_slot / work / flags）。flags は dispatch / wait / end のみ。
  branch / loop / fanout / join は未実装。
- variant ABI: `McuKernelVariantDesc`。kernarg recipe・geometry・retained packet を持つ。
  interpreter は variant_id のみを解決する。
- prepared dispatch ABI: stage 済み packet + queue slot + generation + completion slot を
  MCU ローカルに保持する。ready queue は作らない。

## 4. Packet

- retained packet は **64 byte**（publish DW0 4 byte + body 60 byte）。
  従来の `GpuAqlPacketTemplate` は 80 byte なので 1 variant あたり 16 byte 削減。
- variant 数: test は 1〜2。`retained_count * 64 byte` を LDS budget と比較し、
  超える場合は configure が fail-closed（VRAM へ自動 fallback しない）。
- LDS 使用量: test は 8192 byte budget を指定（variant 2 で 128 byte のみ使用）。
  LDS は dynamic shared memory で確保し、`sharedMemPerBlock` を超える budget は拒否する。
- VRAM fallback: なし。`template_source` は test / bench 専用の A/B ノブ。
- dispatch ごとに patch する field: **kernarg_address のみ**。加えて MCU は
  HIP 暗黙引数用の launch metadata（grid / block）を variant から再書き込みする。

## 5. Kernarg

- slot: 事前確保した ring。`slot_index = dispatch_seq % kernarg_slot_count`。
- slot size / count: `aql_kernarg_slot_stride(kernarg_segment_size)`、test は 8 slot。
- allocation: `GpuMcuAqlQueue::allocate_kernarg`（HSA kernarg region）。
- GPU から直接 write できたか: **できた**。ただし packet の acquire/release scope が必須（§6）。
- generation 管理: `dispatch_seq + 1` を worker の `generation` に載せ、completion の
  generation と照合する。stale completion は generation 不一致で棄却される。
- host は起動前に slot を確保するだけで、dispatch ごとの `memcpy(kernarg)` を行わない。

## 6. 最重要の finding: packet の acquire / release scope

**現象**: device 側で kernarg を書き換えて publish すると、worker が**古い kernarg を読む**
ことがあった。host から kernarg region を読むと正しい値が見えるが、device 内部の worker は
古い値を読んでいた。ホスト観測はコヒーレンス回復後の値であり、根拠にならない。

**原因**: `build_aql_packet_template` に `kMcuAqlFastPolicy`（acquire=NONE / release=NONE）を
使っていた。NONE/NONE は RnD ドキュメント自身が「pre-staged kernarg 前提」としている policy で、
device が書いた kernarg を worker が読む producer -> consumer 依存には順序付けを与えない。

**修正**: packet を **Agent acquire / Agent release**（MCU と worker は同一 GPU agent）で構築する。
System でも成立するが、Agent で十分かつ安い。

**検証**: Agent で 1200 dispatch（400 plan × 3）+ recreate、10000 transition、
600 plan の device kernarg、2 variant、prepared dispatch がすべて PASS。
Agent / System の timing 差は本スケールでは観測できない（§8）。

この問題は「ホストから見えるから正しい」と判断すると見落とす。Gate M4 の中心的な収穫である。

## 7. Resources

| kernel | sgpr | sgpr spill | vgpr | vgpr spill | scratch | group | kernarg | code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline persistent loop（M0） | 28 | 0 | 35 | 0 | 0 | 0 | 8 | 6036 |
| micro FSM kernel | 70 | 0 | 100 | 0 | 0 | 0 | 264 | 15228 |
| FSM worker | 20 | 0 | 7 | 0 | 8 | 0 | 312 | - |
| legacy probe worker | 16 | 0 | 7 | 0 | 0 | 0 | 296 | - |

- **sgpr_spill / vgpr_spill はすべて 0**。scratch も実質 0（worker の 8 byte のみ）。
- FSM kernel は state / plan / variant / timing を扱うため SGPR / VGPR が増えるが spill しない。
- LDS は dynamic 確保。budget は `lds_budget_bytes`。

## 8. Timing

条件: gfx1201、Release、plan = 3 dispatch（dispatch -> wait を 3 回）、
`phaseshift-bench gpu-dispatch --variant device-aql-fsm[-lds] --iterations 200-300`。
値は p50（us）。`dependency_wait` は doorbell から dependency 成立までの時間で、
worker 起動遅延と worker duration を含む。

### template / scope A/B（work=0, barrier=1, plans=300）

| variant | scope | kernarg_build | packet_stage | prepare_total | commit | dependency_wait | prepare/wait |
|---|---|---:|---:|---:|---:|---:|---:|
| device-aql-fsm（VRAM） | agent | 1.64 | 2.32 | 3.96 | 1.56 | 17.60 | 0.225 |
| device-aql-fsm-lds | agent | 1.68 | 2.32 | 4.00 | 1.56 | 17.64 | 0.227 |
| device-aql-fsm-lds | system | 1.60 | 2.32 | 3.92 | 1.60 | 17.68 | 0.222 |
| device-aql-fsm-lds（copy-lanes 32） | agent | 1.64 | 2.32 | 3.96 | 1.60 | 17.60 | 0.225 |

### work 量スイープ（LDS, agent, barrier=1, plans=200）

| work | prepare_total | dependency_wait | prepare/wait |
|---:|---:|---:|---:|
| 0 | 4.00 | 17.64 | 0.227 |
| 200 | 3.84 | 52.96 | 0.073 |
| 2000 | 3.80 | 542.48 | 0.007 |
| 20000 | 3.76 | 5329.48 | 0.001 |

### 所見

- **LDS と VRAM で差なし**。この規模（variant 1、64 byte）では retained template の
  read は hot path の支配項ではない。LDS 化による resource pressure 増も観測されない。
- **Agent と System で差なし**。本スケールでは fence scope の差は wait に埋もれる。
- **copy-lanes 1 と 32 で差なし**。LDS template copy は起動時のみで支配項ではない。
- **prepare_total は約 4 us で一定**（kernarg build 1.6 + packet stage 2.3 + commit 1.6）。
- worker duration が ~15 us を超えると prepare_total は完全に隠蔽される。
- work=0 では queue が毎 dispatch drain するため **cold worker start 約 17.6 us** が
  dependency_wait を支配する。hot な連続 kick（約 3 us）とは別物。
- barrier=0 でも dependency_wait は変わらない（FSM が毎 dispatch completion を待つため）。

## 9. Failures / surprises

- 当初 400 plan 程度で worker が古い kernarg を読む現象が出た（§6）。原因は fence scope。
- host 観測では正しい値が見えるため、最初は「メモリには正しい値があるのに worker が古い値を読む」
  と誤認した。device 内部の可視性は host からは判定できない。
- LDS retained template は**有利でも不利でもなかった**。この規模では差が出ない。
- Agent / System の fence scope も timing 差が出なかった。
- MCU が毎 dispatch completion を待つ構成では cold worker start が支配的で、
  prepare の隠蔽余地は大きいが、そもそも worker 起動が支配的である。

## 10. 次 Gate への判断

- 維持する: stage/commit 分離、retained packet、device completion、micro FSM、
  prepared dispatch、Agent fence scope、kernarg ring、variant registry。
- 捨てる: source 側の debug 計装、System fence scope（Agent で十分）、
  reserve 方式の publish（streaming で成立）。
- 次に実 kernel で試す: 実 RMSNorm 単体、次いで実 primitive 2〜3 個の chain。
  いきなり Qwen full layer へは行かない。
- 未解決の論点: LDS が有利になる条件（variant 数 / template サイズ / reuse 回数）、
  prepared dispatch を活かした bubble 削減、completion wait 中の stop 応答性。

## 11. Acceptance

- `ctest -L gpu_mcu`: 15/15 PASS（既存 8 + 新規 7）
- `tests/run_required_acceptance.py`: 結果は本ドキュメントの commit メッセージ / 報告を参照
- skip は PASS 扱いしない
