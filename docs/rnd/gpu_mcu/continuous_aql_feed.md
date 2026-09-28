> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# GPU-MCU Continuous AQL Queue Feeder PoC

- 目的: MCU を「completion ごとに CP を再起動する launcher」から「CP 実行中に AQL ring へ
  仕事を継ぎ足す producer」へ変えられるかを実機で確定する
- 対象: `phaseshift_gpu_mcu` の micro FSM に queue feeder を追加
- 対象外: production Qwen executor 接続、MoE、multi-GPU、EXT_KERNEL_DISPATCH、packet
  cancellation、複数 AQL producer
- 性能改善そのものは GO/NO-GO 条件ではない。queue / CP 挙動の確定が目的
- 詳細契約: `docs/developer/gpu_mcu/low_level.md`

## 1. 基点

| 項目 | 値 |
|---|---|
| base revision | `886851947640aea857383dd451dff721d99a36a3` |
| worktree | `.worktrees/gpu-mcu-continuous-aql-feed` |
| branch | `poc/gpu-mcu-continuous-aql-feed` |

## 2. Gate 結果

| Gate | 内容 | 結果 |
|---|---|---|
| Q0 | 既存 Micro-FSM の再計測（inter-kernel gap 含む） | 計測済み |
| Q1 | append while running（A 実行中に B を追加） | PASS |
| Q2 | 複数 packet 後詰め（B〜F） | PASS |
| Q3 | no-re-doorbell 探索 | Case 1（後詰めだけでは実行されない）。後 doorbell で回復 |
| Q4 | doorbell coalescing | PASS |
| Q5 | ring wraparound | PASS（10000 dispatch、156 wrap、overwrite 0） |
| Q6 | queue-fed static chain | PASS |
| Q7 | queue ahead depth 1/2/4/8/16 | 計測済み |
| Q8 | continuous refill の証明 | PASS（refill 9991/10000、empty 1、max_ahead 8） |
| Q9 | drain vs feed A/B | 計測済み（§7） |
| Q10 | expected classification | A（queue-fed 成功） |
| Q11 | continuous feeder resource audit | spill 0（§8） |
| Q12 | stop / fault | 成立（新 packet 追加を停止、doorbell 済みは drain） |
| Q13 | 実 RMSNorm へ進む判定 | 次 Gate 候補は real RMSNorm |

## 3. Feeder 契約

- plan は `dispatch x N` + 末尾 `wait` + `end`。static dependency の dispatch を
  completion ごとに待たない。
- 各 packet は barrier=1 / Agent acquire / Agent release。CP が先行 packet の complete を
  保証するため、A 完了前に B を enqueue してよい。
- `queue_ahead_depth` = MCU が先行 publish する最大 packet 数。0 は ring size bound。
  backpressure は `read_index` で判定し、queue full は fault にしない。unread slot は
  overwrite しない。
- doorbell: `per_packet` / `coalesce`（`doorbell_batch` 個ごと + wait/plan end で tail）/
  `first_only`（探索用）。
- `issue_wait_running` は 2 番目の dispatch を publish する前に先頭 dispatch の running を
  観測する（append-while-running 検証専用）。

## 4. 最重要 finding: completion publish の race

同じ completion slot を世代で使い回す構成で、worker が `running` を
「generation を書いてから state を書く」順で publish していた。直前の dispatch が
`complete` を残しているため、`generation = N, state = complete(N-1)` という瞬間ができ、
末尾の `wait`（generation N の complete を待つ）が**早期復帰**した。

- 症状: 最後の dispatch が実行中なのに plan が完了扱いになり、出力の一部が未確定。
- 原因: 世代使い回し時の publish 順序。
- 修正: publish を `state=idle -> generation -> state` の順にする。idle を先に置くことで
  「新 generation + 旧 complete」の窓が消える。
- 影響: feed では全 dispatch が completion slot 0 を共有するため必ず踏む。per-dispatch
  slot + 直後 wait の従来構成では見えなかった。

`wait` の generation 比較だけでは不十分で、publish 側の順序が contract である。

## 5. Append while running（Q1 / Q2）

work=4000 の A 実行中に後続を publish した。`McuDispatchRecord.publish_ts` と worker
timestamp で確認。

- Q1: `A_start < B_publish < A_end`、`B_start >= A_end` を満たした。
- Q2: A_start より後に B〜F の publish がすべて A_end より前に収まった。
- `issue_wait_running=1` のときだけ「A running 観測後に publish」になる。0 のときは
  cold start 中（A_start 前）に publish され、AQL 上は先行 enqueue だが timestamp 条件は
  満たさない。

## 6. Doorbell behavior

- append + re-doorbell: `per_packet` で 12800/12800 鳴らし、全 packet 実行。
- batch-tail doorbell: `coalesce` batch=4 で 12800 dispatch に対し doorbell 3200
  （12800/4）。実行順序・出力一致。tail を鳴らす前に batch 内 DW0 は全 valid。
- append without re-doorbell（探索、required ではない）:
  `observed_case=case1_not_executed`。最初の doorbell 後、後詰め packet は追加 doorbell
  無しでは実行されなかった。後から doorbell を進めると回復して実行された。
  これは**実機で観測された実装挙動**であり、supported contract ではない。

## 7. Timing

worker timestamp から `inter_kernel_gap = worker_start[i+1] - worker_end[i]` を算出。
単位 us、p50。barrier=1、Agent、template=VRAM。

### work=0（plans=200）

| mode | depth | gap p50 | gap p95 | gap max | worker_duration p50 |
|---|---:|---:|---:|---:|---:|
| wait | - | 23.84 | 23.84 | 23.84 | 0.76 |
| feed | 1 | 22.48 | 23.00 | 23.48 | 0.92 |
| feed | 2 | 3.76 | 19.00 | 19.08 | 0.92 |
| feed | 4 | 3.00 | 16.88 | 17.12 | 0.88 |
| feed | 8 | 3.00 | 16.84 | 18.04 | 0.88 |
| feed | 16 | 3.00 | 17.04 | 17.68 | 0.88 |
| feed-batch | 4 (batch 4) | 3.04 | 26.28 | 26.60 | 0.88 |

### work=2000（plans=50, dispatch=16）

| mode | depth | gap p50 | gap p95 | gap max | worker_duration p50 |
|---|---:|---:|---:|---:|---:|
| wait | - | 23.52 | 23.52 | 23.52 | 541.88 |
| feed | 1 | 22.40 | 23.32 | 23.44 | 545.44 |
| feed | 2 | 3.36 | 3.52 | 3.72 | 545.32 |
| feed | 4 | 3.00 | 3.04 | 3.48 | 545.24 |
| feed | 8 | 3.04 | 3.12 | 3.60 | 546.60 |
| feed | 16 | 3.04 | 3.36 | 3.44 | 543.96 |
| feed-batch | 8 (batch 4) | 3.04 | 3.12 | 3.52 | 544.96 |

- depth=1 では改善しない。publish が 1 個先までしか進まず CP が drain する。
- depth>=2 で gap p50 が約 23.5 us から約 3.0 us に下がる。
- work=2000 では depth>=2 の gap は 3.0 us で安定し、p95 も 3.5 us 以下。これは CP の
  dispatch floor とみられる。
- work=0 では p95 が 17〜27 us に伸びる。worker が短すぎて CP が drain する瞬間が残る。
- batch-tail doorbell は work=2000 では gap に影響しない。work=0 では tail が
  26 us まで伸びる。

### counters（`--debug-records`、overhead を含む）

| mode | depth | dispatches | doorbells | empty | full | refill | max_ahead | wraps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wait (work0) | - | 600 | 600 | 600 | 0 | 599 | 1 | 0 |
| feed (work0) | 2 | 12800 | 12800 | 200 | 6201 | 12797 | 2 | 12 |
| feed (work0) | 4 | 12800 | 12800 | 200 | 2 | 12795 | 4 | 12 |
| feed (work0) | 16 | 12800 | 12800 | 200 | 2 | 12783 | 16 | 12 |
| feed-batch (work0) | 8 (batch4) | 12800 | 3200 | 200 | 2 | 12791 | 8 | 12 |
| feed (work2000) | 4 | 800 | 800 | 50 | 600 | 795 | 4 | 0 |

- wait は **600 dispatch 中 600 回 queue が空**。completion ごとに CP を再起動するため
  毎回 drain する。
- feed の empty は plan 数と一致する。各 plan 末尾の `wait` 中に queue が drain するため。
  plan 内部では empty が発生しない。
- 単一長 plan（後述 ring wrap、10000 dispatch、depth 8）では empty=1、refill=9991、
  max_ahead=8。steady-state で CP が飢えていない。
- depth=2 で full=6201 と backpressure が多い。depth>=4 では full はほぼ 0 で、MCU が
  CP より先に走っている。

## 8. Ring（Q5）

- queue size 64（HSA の最小制約。8 / 16 は `hsa_queue_create` が失敗した）。
- 10000 dispatch、156 wrap、overwrite 0、出力一致。
- counters: wraps=156、full=2、empty=1、refill=9991、max_ahead=8。
- ring wrap しても MCU は `read_index` で backpressure し、unread slot を overwrite しない。

## 9. Resources（Q11）

| kernel | sgpr | sgpr spill | vgpr | vgpr spill | scratch | group | kernarg | code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro FSM（前 PoC） | 70 | 0 | 100 | 0 | 0 | 0 | 264 | 15228 |
| continuous feeder FSM | 77 | 0 | 153 | 0 | 0 | 0 | 264 | 21780 |
| FSM worker | 20 | 0 | 7 | 0 | 8 | 0 | 312 | 736 |
| probe worker | 16 | 0 | 7 | 0 | 0 | 0 | 296 | 460 |

**sgpr_spill / vgpr_spill はすべて 0**。feeder 追加で VGPR と code size が増えたが spill は
出ていない。

## 10. Failures / surprises

- completion publish の race（§4）。feed を組むまで見えなかった。
- HSA queue は size 8 / 16 を作れなかった。ring wrap は size 64 で行った。
- depth=1 は feeder として機能しない。depth>=2 が必要。
- `first_only` は後詰め packet を実行しない（Case 1）。doorbell は CP への kick として
  必要であり、write_index 更新だけでは足りない。
- batch-tail doorbell は doorbell 回数を 1/4 に減らせるが、gap 改善には寄与しない。
  work=0 では tail 待ちで p95 が悪化する。

## 11. 次 Gate への判断（Q13）

- 判定: A（queue-fed 成功）。gap は depth>=2 で約 23.5 us → 約 3.0 us に低下し、
  correctness（順序・依存・wrap・10000 dispatch）も成立した。
- 維持: stage/commit、retained packet、device completion（publish 順序含む）、
  micro FSM、Agent fence scope、kernarg ring、variant registry、queue feeder、
  coalesced doorbell、debug record / counter。
- 捨てる: depth=1 運用、`first_only` の contract 化、System fence scope。
- 次候補: real RMSNorm 単体。その次に RMSNorm -> Activation Quantize -> PSQ GEMM の
  3 primitive chain を最初の micro-program 候補とする。
- 未解決: CP dispatch floor 約 3.0 us の正体、work=0 で残る p95 の伸び、
  per-plan 末尾 wait による drain、dynamic dependency の分離。

## 12. Acceptance

- `ctest -L gpu_mcu`: 22/22 PASS（required 21 + 探索 1）
- required acceptance: 結果は commit message / 最終報告を参照
- skip は PASS 扱いしない
