> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# MTP 最適化・Gate 検証履歴

`docs/rnd/optimization_findings.md` から分離した、MTP（内蔵 drafter）の correctness / acceptance / Gate 検証の履歴。見出し番号（7.x）は元 dump の通し番号を保持する。MTP の現状診断は同 directory の [mtp.md](mtp.md) を参照。

## 収録セクション

- 7.38 MTP head: correctness fallback と draft 連鎖の hidden 入力
- 7.39 MTP acceptance: prompt 長と K 依存、multi-row verify の等価性
- 7.60 MTP Gate 1: 孤立 attention の識別力と、累加順序差に由来する bf16 誤差の深度依存
- 7.61 MTP Gate 2: BF16 cache × 長列 softmax では argmax 完全一致は達成不能
- 7.62 MTP Gate 3: vLLM の draft alignment と MTP KV length
- 7.63 MTP Gate 4A: spec ON/OFF token mismatch は M 依存 kernel 選択が原因
- 7.64 MTP Gate 4A.1: per-tensor trace による M 依存 divergence の特定と VERIFY_EXACT
- 7.65 MTP Gate 4B: VERIFY_EXACT の verify role 限定と fallback 除去
- 7.66 MTP Gate 4C: verification D2H / sync の baseline
- 7.67 MTP Gate 4E: fixed-K qualification / break-even
- 7.69 MTP Gate 5A: draft cost profile と exact LM-head の評価
- 7.70 MTP Gate 5B: confidence calibration と oracle K
- 7.71 MTP Gate 5C: dynamic draft depth
- 7.72 MTP 全 stage 内訳（per emitted token）
- 7.73 MTP draft の Correctness fallback 除去
- 7.74 Verify 本体の M 依存の正体: partial accept の rerun forward
- 7.75 Gate 5 最終結果（shape 修正 + dynamic + discard）
- 7.76 all-accept 時の pending_hidden 行の誤り
- 7.77 n-gram tail draft

---

## 7.38 MTP head: correctness fallback と draft 連鎖の hidden 入力（2026-09-16）

### 背景

Qwen3.5-4B-ROCMFP-BALANCED で MTP head を実装直後、`phaseshift-bench mtp`
実測で head 1 call = **11.07 ms**（decode step 14.26 ms の 77.6%）。
draft は depth-1 で 76% 前後当たり、連鎖が全く儲からない水準だった。

### 原因（`PHASESHIFT_OP_DUMP=1 PHASESHIFT_OP_SYNC=1` と `PHASESHIFT_QWEN35_KERNEL_TRACE=1` で確定）

per-op 内訳（1 call、修正前）:

| op | ms | 状態 |
|---|---|---|
| LINEAR_ROCM_FP8 (lm_head 248320x2560) | 3.64 | Correctness |
| LINEAR_ROCM_FP4 (mtp.fc 2560x5120) | 0.82 | Correctness |
| PAGED_ATTENTION | 0.42→0.60 | Correctness、**系列長に比例して悪化** |
| CONCAT / ROPE x2 / OUTPUT_GATHER x2 / RMS_NORM | 各 0.03-0.55 | 一部 Correctness |

`KERNEL_TRACE_FALLBACK` は `ROPE 132` / `OUTPUT_GATHER 17` を示した。
原因は3つ、すべてselector側の記述欠落または未設定:

1. `kLinearShapeRules` に 4B 形状が無い（§7.31 と同じ失敗の 4B 版）。
   rule は family x out x k の完全一致なので、1形状の欠落がそのまま
   reference kernel 落ちになる。
2. `MtpExecutor` が `HostExecutionContext::min/max_visible_tokens` を
   0 のまま渡していた。`select_paged_attention_implementation` は
   `max_visible_tokens >= 1` を要求するため、MTP の attention は
   常に Correctness になっていた。
3. final block が N 行 rmsnorm + OUTPUT_GATHER x2 を実行していた。
   rmsnorm は行単位なので gather 後（OUTPUT_ROWS 1 行）に移動できる。

### 修正

- rule 追加: `RocmFp8W8A8 (248320,2560)` / `RocmFp4W4A8 (2560,5120)` /
  `Psq8W8A8 (248320,2560)` / `Psq4W4A8 (2560,5120)` / `Bf16 (2560,5120)`
- `run_mtp_rows`: `min_visible = position_start + 1`, `max_visible = position_start + rows`
- target / MTP: `layer_output` を gather してから rmsnorm（gather 1回に統合）

### 実測（4B-ROCMFP / context=64 / steps=40 / K=4）

| | 修正前 | 修正後 |
|---|---|---|
| MTP head | 11.07 ms/call (77.6%) | **1.99 ms/call (16.9%)** |
| target decode | 14.26 ms/step | 11.83 ms/step |
| tg forward (compute-logits 1) | 70.6 tok/s | **85.7 tok/s** |

残る MTP head 内訳の 1.19 ms は lm_head（int8 636 MB read、534 GB/s 相当）。
docs 実測（bf16 lm_head ~507 GB/s / psq8 ~460 GB/s）を上回っており、
rows=1 帯域律速としてほぼ床。

数値: LOGITS bit-identical（argmax=198 max=13.597682 sum=-440666.781）。
greedy stream も全長一致。ctest 78/78。

### draft 連鎖の hidden 入力は post-norm（実測で pre を棄却）

MTP graph に 4つ目の external output として `mtp.norm` 適用後
（`normed_hidden`、lm_head が入力として見た表現。既存値の露出のみで
追加カーネル無し）を追加し、step >=2 の入力を選ぶ `--chain pre|post` で A/B。

実テキスト corpus（`tools/gen_token_corpus.py` で生成した PSKLDTOK、
96 prompt token + 128 decode step、draft-k=4、Qwen3.5-4B-PSQ）:

| 累計受容 | chain=pre | chain=post |
|---|---|---|
| P(draft1) | 75.9% | 76.5% |
| P(>=2) | 42.6% | 45.1% |
| P(>=3) | 9.3% (5/54) | **17.6%** (9/51) |
| P(>=4) | 3.7% (2/54) | **9.8%** (5/51) |
| tokens/round (K=4) | 2.31 | **2.49** |
| MTP head | 1.886 ms | 1.887 ms |

同一コストで深い段が約2倍。既定は post。

段別からの見積（plain 10.63 ms/tok、draft 1.887 ms）:

| K | tokens/round | ms/token | speedup |
|---|---|---|---|
| 1 | 1.77 | 7.09 | 1.50x |
| 2 | 2.22 | **6.50** | **1.63x** |
| 3 | 2.39 | 6.81 | 1.56x |
| 4 | 2.49 | 7.30 | 1.46x |

**K=2 が最適**。verify 側（K+1 token 処理）の増分は未計上なので実測は
この下限になる。

### 否定的結果: MTP を bf16 にしても受容率は上がらない

draft は 1 層しかなく量子化誤差を吸収しない假説を検証するため、
`psq()` の MTP role を PSQ4 -> BF16 にしたモデル
（`Qwen3.5-4B-PSQ-MTP16`、target 量子化は同一）を作り A/B した。

| | MTP=psq4 | MTP=bf16 |
|---|---|---|
| P(draft1) | 75.9% | 73.6% |
| P(>=2) | 42.6% | 43.4% |
| tokens/round (K=4) | 2.31 | 2.36 (+2%) |
| MTP head | 1.888 ms | 2.385 ms (+26%) |

depth-1 は誤差範囲、+2% の受容より +26% の draft コストが勝つ。
**MTP 重みの量子化精度は受容率の犯人ではない。** 変更は採用せず
`profile.cpp` は revert 済み。

### 未対応（別件で判明した不整合）

`attention_prep` 融合が 4B で一度も発火していない（`attention_prep=0`,
`KERNEL_TRACE_FALLBACK ROPE 132`）。target は decode step 16 発の rope が
correctness で約 0.17 ms/発 = step の 23% 相当。MTP 固有でも量子化とも
無関係な全体最適点で、本節の修正とは別途。


---

## 7.39 MTP acceptance: prompt 長と K 依存、multi-row verify の等価性（2026-09-16）

### §7.38 の数字は条件が狭かった

§7.38 は context=96 / draft-k=4 / 51 ラウンドで測った。context=128 / draft-k=6 で
実テキスト corpus を測り直すと深い段の生き残りが大きく違う。

PSQ 4B / 128 decode steps / MTP head 1.887 ms/call（decode step の 17.6%）:

| 累計受容 P(≥k) | prose (en+ja) | code+JSON |
|---|---|---|
| 1 | 88.2% | 69.8% |
| 2 | 58.8% | 41.9% |
| 3 | 47.1% | 30.2% |
| 4 | 38.2% | 25.6% |
| 5 | 20.6% | 14.0% |
| 6 | 11.8% | 4.7% |
| tokens/round | 3.65 | 2.86 |

段別 K の見積（plain 10.67 ms/tok、verify 1 step + draft K 回、rollback 増分は未計上）:

| K | tokens/round | ms/token | speedup |
|---|---|---|---|
| 2 | 2.47 | 5.85 | 1.82x |
| 3 | 2.94 | 5.55 | 1.92x |
| 4 | 3.32 | 5.48 | **1.95x** |
| 5 | 3.53 | 5.70 | 1.87x |
| 6 | 3.65 | 6.03 | 1.77x |

**K=4 が山**。5 段超は draft コスト（1 回 1.89ms、うち 1.19ms は lm_head）が
受容増を上回る。§7.38 の「K=2 が最適」は prompt 長を嘗て過小評価していた。

教訓: **acceptance は prompt と continuation の種別に強く依存する**。
96 token の散文続きと 128 token では深い段の率が倍以上違うため、
単一条件の測定で K を決めるな。

### multi-row forward == sequential greedy（verify の等価性）

`DeviceRequestDescriptor` に `output_count`（`_pad` を消費、sizeof 32 維持）、
`ScheduledRequest::num_output_rows`、`ExecutorConfig::max_scheduled_output_rows` を追加し、
1 request で複数 output row を出すようにした（spec verify の前提）。

`bench mtp` の verify-check phase で、phase 1 の reference greedy stream を
GDN reset + prefix replay してから同一 window を multi-row forward にかけ、
行ごとの argmax を突き合わせた:

```
verify-check: window at pos 218, 6 rows
  row  expect  got    ok
    0    1542   1542   ok
    ...
verify-check: PASS (0/6 rows mismatched)
```

**multi-row forward の per-row argmax == sequential greedy**。KV を棄却行で
ahead に書くこと自体は無害（後で上書きされ、valid length 以上は読まれない）。

### 運用上の注意

verify-check は target sequence の GDN reset + replay を行うため、
**MTP の acceptance 測定より後に実行する**こと。先に走らせると acceptance が
76.5% -> 12.6% に化ける（実測）。

### 却下: MTP role を BF16 にする案（§7.36 記載の再確認）

draft を 4bit から bf16 にしても P(draft1) は 75.9% -> 73.6%（誤差範囲）で
tokens/round +2%、MTP head +26%。**採用せず revert**。


---

## 7.60 MTP Gate 1: 孤立 attention の識別力と、累加順序差に由来する bf16 誤差の深度依存（2026-09-19）

### 経緯

- Qwen3.5-4B の MTP 1 層 draft forward を vLLM 参照意味と突き合わせる
  Gate 1（詳細は mtp.md の Gate 記録）を締結。
- fixture 4 case（token 925 の position 0/1/127、token 5678）、
  31 中間 tensor の per-tensor trace 比較、draft argmax は完全一致を要求。
- 結果: 4 case 全 PASS、draft 770/770/770/883 一致、required acceptance 79/79 PASS。

### 検証入力の識別力（top1 一致は弱い証拠）

- MTP draft forward は KV 未永続・**可視トークン 1** の孤立 attention。
  このとき attention 出力は softmax が自明（1 key）になるため
  **厳密に V** と等しい。
- したがって position 0/1/127 で `q_rope`/`k_rope` は変わるが、
  draft top1 は 770 のまま変わらない。この不変性は bug ではなく数学的帰結。
- 帰結: **top1 / top10 一致だけを合格条件にすると RoPE 経路の破損を検出できない。**
  中間 tensor（`q_norm` / `k_norm` / `q_rope` / `k_rope` / `attention_core_out`）を
  per-tensor で比較し、top1/top10 は補助指標に留める必要がある。
- 同種の落とし穴は他の「可視長 1」検証でも起きる。
  入力選択が出力に現れる経路を列挙してから fixture を設計する。

### 累加順序差に由来する bf16 誤差の深度依存

- 参照（f64 計算・境界で bf16 丸め）と実 kernel（f32 累加・bf16 保存）の差は、
  層を下るほど増大する。f32 累加順序の違いが bf16 丸め境界をまたぐため。
- 実測 rel_l2: 前半 ≤ ~1e-3、後半は単調増加し `mlp_down_proj` 最大 5.3e-3、
  `logits` 2.0e-3（pos0）/ 3.7e-3（token_alt）。cosine はすべて ≈1.0。
- semantic 誤り（norm 式・split 順序・gate 位置・concat 順序・残差順序・
  tied 破損）は rel_l2 ≫ 10% を生むため、前半 1e-2 / 後半 2e-2 rel_l2 の
  閾値で 4〜10× の余裕をもって分離できる。
- 閾値は C++ テストと Python クロスチェックで同一値を持ち、
  **argmax は閾値対象外で完全一致**を要求する（緩めない）。

### wiring で確定した事実（Gate 1 で発見・修正した production bug）

1. embedding kernel は graph の external input ではなく
   `batch_context->token_ids` を読む。executor が graph に token を流すだけでは
   常に token 0 を lookup する。`batch_storage.token_ids` へ書く必要がある。
2. RoPE position と KV / 可視位置は分離が必要
   （`DeviceBatchContext::rope_positions` + `run_mtp_rows(..., kv_position_start)`）。
   body 経路は `rope_positions = null` で従来通り。
3. correctness パスに CONCAT dispatch が無かった（`exec_concat` を追加）。
4. safetensors reader/writer に I32 が無かった
   （fixture の INT32 保存・読み出し用に追加）。

### 参照

- 検証の詳細（閾値表・fixture 仕様・再現手順）は mtp.md の Gate 1 記録。
- 新規 required テスト: `test_qwen35_mtp_lowering`（175 checks、lowering の構造）、
  `test_qwen35_mtp_primitives`（29 checks、実 kernel の数式 + 負コントロール）。


---

## 7.61 MTP Gate 2: BF16 cache × 長列 softmax では argmax 完全一致は達成不能（2026-09-19）

### 経緯

- Gate 2 で MTP の persistent KV / state synchronization を検証（詳細は
  mtp.md の Gate 2 記録）。257 step の teacher-forced 列を production
  `mtp_forward_step` で再生し、cache を使わない参照（Python、履歴全体から
  毎 step attention を再計算）と比較した。
- state mechanics は完全一致: canonical K/V は serial と 16-token chunk で
  **bit-exact**、block boundary / reset / position offset / 2 sequence isolation
  も PASS。required acceptance は 81/81。

### 知見: argmax 一致は数値的に達成不能

- 参照との worst 相対誤差: `k_rope` 6.2e-3、`v_proj` 6.6e-3、
  `attention_core_out` 1.8e-2、`o_proj_out` 1.9e-2、`final_norm` 1.9e-2
  （cosine ≥ 0.9998）。
- 誤差源は **BF16 出力の丸め境界差**（bf16 は有効 3 桁 ≈ 0.4%）。kernel と参照が
  隣接 bf16 値へ丸まると要素差 1 ulp ≈ 0.8% になる。
- これが 257 key の softmax 重み付けで増幅し、logits 相対誤差 ≈ 2% になる。
- 結果、top1-top2 margin が 3.3% / 1.5% の 2 step（84 / 98）で draft argmax が
  flip した。margin < 誤差の near-tie であり、state/alignment の欠陥ではない
  （canonical K/V は bit-exact）。
- 教訓: 独立な f32 実装同士を BF16 I/O で突き合わせる検証では、
  **「全 step argmax 完全一致」を合否条件にしない**。合否は canonical K/V の
  一致（構造）と margin に対する誤差の大きさで判定する。
  Gate 1 の isolated T=1（可視 1 key、softmax 自明）で一致したのは増幅が
  無かったためで、長列へ一般化できない。
- 採用した規約: `margin_ref = ref_top1 - ref_top2`、
  `epsilon_inf = max|actual - reference|`（top10 上で実測）とし、
  `margin > 2*epsilon_inf` を stable、それ以外を `NUMERICALLY_AMBIGUOUS` とする。
  stable mismatch のみ FAIL。実測で flip 2 件はどちらも ambiguous
  （step 84: margin 0.033 ≤ 2×0.051、step 98: margin 0.018 ≤ 2×0.034）、
  stable mismatch 0。cosine は補助指標へ降格し、primary は relative L2 /
  max abs / canonical state / stable argmax とした。

### 参照

- 詳細（KV contract・state 契約・未実施項目）は mtp.md の Gate 2 記録。
- 新規 required テスト: `test_qwen35_mtp_kv_cache`（35 checks）、
  `test_qwen35_mtp_state`（51 checks）。

---


---

## 7.62 MTP Gate 3: vLLM の draft alignment と MTP KV length は `+accepted` ではない（2026-09-19）

### わかったこと

- vLLM `vllm/v1/spec_decode/llm_base_proposer.py` の default pathway では、
  MTP draft は `input_ids[t] = target_token_ids[t+1]`、
  `hidden_states[t] = target_hidden_states[t]`、
  `positions[t] = target_positions[t]` で走る。最後の slot は
  `input_ids[last] = next_token_ids`（pending token）であり、
  **MTP は (hidden[p], token[p+1]) を position p で消費して p+2 を予測する**。
- 次 draft step へ渡す hidden は model が返す hidden（Qwen3.5 は tuple 返しでない）。
  PhaseNonShift では final norm 前の `hidden_out` を使う。
- accept 後の MTP logical length は `mtp_before + min(accepted+1, K)`。
  全 accept のとき最後の draft は MTP に未消費なので `+K` で cap される。
  `mtp_before + accepted` を仮定すると、accept=K の場合に 1 つずれる。

### 実装

- `spec_decode.h/.cpp`: `spec_greedy_accept`、`SpecPhase` 状態機械、
  `SpecTransaction`（MTP logical rollback + target position/block rollback）、
  `spec_gdn_snapshot/restore`、`mtp_generate_drafts`。
- required: `test_qwen35_spec_verify`（pure 100 checks）、
  `test_qwen35_spec_transaction`（GPU 730 checks: forced accept 0..K、page boundary、
  rollback、GDN snapshot、100 iteration、2 sequence isolation）。
- required acceptance: **83/83 PASS**。

### 注意

- `PagedSequenceState` は move assignment が delete されている。
  再作成時は `runtime_install_sequence_state` を使う。
- `commit_sequence_append` は `committed_tokens == 0` を拒否する。
  `advance(0)` のような no-op は呼び出し側で早期 return する。

### 7.62 追記: end-to-end spec ON/OFF greedy 完全一致（2026-09-19）

`spec_decoder.cpp` で target verifier を接続し、同一 target の spec OFF（通常
greedy decode）と spec ON（MTP K draft + greedy verify + rollback）の
token stream を比較した。実 Qwen3.5-4B、32/128 token prompt、K=1/2/4 で
**全 token ID 完全一致**。rollback は 44 回（accept=0 が 23 回）発生した。

- MTP は prompt を teacher-force して KV を作る（`(hidden[p], token[p+1])` @ p）。
- partial accept は GDN snapshot へ戻し accepted prefix を再 forward して
  target state を作る（推測しない）。
- MTP logical length は `before + min(accepted+1, K)`。

### 7.62 追記2: Gate 2.5 完了（real target fixture + compare tool、2026-09-19）

- `test_qwen35_mtp_spec_real` の `PHASESHIFT_MTP_GATE2_REAL_DUMP` モードで実
  Qwen3.5-4B target の `token_hidden`（64 token real prompt）を dump し、
  exporter `--real-dir` で `real/` case を作る。実 target hidden を MTP persistent
  state へ teacher-force した結果は `MTP_GATE2_REAL: PASS`
  （argmax 63/64、1 件 ambiguous、k_rel 8.0e-3、v_rel 5.3e-3）。
- `tools/compare_mtp_gate2.py`: C++ dump を独立に検証。
  synthetic / real とも `MTP_GATE2_COMPARE: PASS`、
  `FIRST_STATE_MISMATCH_STEP=-1`、`FIRST_MISMATCH=<none>`。
- Gate 2 は `STATE CONTRACT: PASS` / `REFERENCE NUMERICS:
  PASS_WITH_AMBIGUOUS_NEAR_TIES`（stable mismatch 0）として締結。

### 7.62 追記3: K=0 path と acceptance statistics（2026-09-19）

- K=0（sync-only の単一 decode path）を end-to-end に追加。両 prompt で spec OFF と
  完全一致（`MTP_GATE3_K0: PASS`）。
- acceptance statistics（correctness とは別 metric、PASS 条件ではない）:
  iterations=155、drafts=346、accepted=121、avg accepted per draft=0.350、
  full-accept=35、reject-at-0=29。
- この synthetic prompt は acceptance が低め（avg 0.35）。最終出力の correctness は
  greedy verify が保証しており、acceptance は性能側（Gate 4）の指標。

### 7.62 追記4: Gate 2 の 3 seed 誤差分布（2026-09-19）

hidden/token seed を変えた 257 step を 3 種測った。worst rel_l2 は
attention_core 1.28–1.82e-2、final_norm 1.50–1.87e-2、o_proj 1.40–1.85e-2、
k_rope 6.21–6.32e-3、v 6.58–7.22e-3。argmax flip は 2 / 2 / 6 件で全て ambiguous、
stable mismatch は 0。Gate 1 の閾値（前半 1e-2、後半 2e-2）は 3 seed で成立し、
1 run に合わせた緩和ではない。

---


---

## 7.63 MTP Gate 4A: spec ON/OFF token mismatch は M 依存 kernel 選択が原因（2026-09-19）

### 4A baseline（optimized kernel mode、自然 prompt 8 件 × 128 token）

- attribution（K4）: verify 78% / draft 22% / GDN snapshot 0.02% / commit ~0、
  unattributed ≈0。correctness mode では verify 88% / draft 12%、`token_match=4/4`。
- `spec_off` 18.2 ms/token（54.9 tok/s）に対し K1 23.8 / K2 24.2 / K4 25.7 /
  K8 30.5 ms/token。**現状は全 K で target-only に負け**。
- optimized mode では `token_match` が 4–6/8。**Gate 4 §1 の blocker**。

### 原因

M（行数）で kernel 自体が変わるため、K 方向の浮動小数演算順序が変わる。

- `launch_gemm_bf16_wmma`（gfx1201）: `rows==1` は `gemv_bf16_kernel`
  （32 lane の tree reduction）、`rows>=2` は `gemm_bf16_wmma_kernel`
  （WMMA の K-tile 累加）。
- `paged_attention_dispatch.hip`: decode は `launch_attention_paged_bf16(_split)`、
  verify は `launch_attention_paged_prefill_bf16` と別 kernel。
- `gemm_qkv/gate_up/gdn_proj_bf16_wmma` は独自 kernel（generic gemm を経由しない）。

M=1 decode と M=K+1 verify で bf16 の数 ULP 差が蓄積し、near-tie で argmax が反転する。
vLLM も同種問題（M=1 vs M=2 の GEMM 差）を Closed as not planned としており、
PhaseNonShift の Gate 4 基準の方が厳しい。

### 実験: M=1 を WMMA 化（`PHASESHIFT_BF16_EXACT_SMALLM=1`）

M=1 でも `gemm_bf16_wmma_kernel` を通すフラグを追加（WMMA の K 還元は M 非依存）。

結果: `token_match` は **5–6/8 のまま**（改善せず）。generic bf16 GEMM は単独原因ではなく、
attention（decode vs prefill）と fused 射影が残る。副次的に `spec_off` が
18.2 → 23.8 ms/token に悪化（M=1 の gemv を失うコスト）。

### 次

target 側へ Gate 1 方式の per-tensor trace を適用し、M=1 と M=K+1 の
`FIRST_MISMATCH_KERNEL` を特定する。そのうえで
「行内 accumulation を M=1 decode と一致させ、weight 再利用は維持する」
`VERIFY_EXACT` kernel を作る（serial verifier へ全面 fallback しない）。

---


---

## 7.64 MTP Gate 4A.1: per-tensor trace による M 依存 divergence の特定と VERIFY_EXACT（2026-09-20）

### 何が問題だったか

optimized kernel mode で spec ON の target token 列が spec OFF と一致しなかった。
原因は M（行数）で kernel 自体が変わり、K 方向の浮動小数演算順序が変わること。

### 手法

target 側に per-tensor trace を追加した。

- `lower_qwen35_to_primitives` が各 node の `debug_name` を設定（`L{li}.{op}`）。
- `Executor::value_trace`（`ValueTraceSink*`）を追加し、`launch_host_backend` が
  `ctx.value_trace` へ伝搬。`nullptr` なら production は完全 no-op。
- trace 有効時は HIP graph capture を無効化（`imatrix_collector` と同じガード）。
- `test_qwen35_mtp_verify_divergence` が同じ prefill 状態から 2 経路を実行する。
  - Path A（SERIAL）: 同じ token 列を M=1 decode で 1 行ずつ。
  - Path B（VERIFY）: 同じ token 列を M=K+1 で 1 回。
  - 全 dispatch 出力を value_id 単位で bit 比較し、最初の差を
    `FIRST_MISMATCH_ROW/LAYER/OP` として報告する。
- trace 中は `PHASESHIFT_QWEN35_FUSION=0` で論理中間値を materialize する
  （融合 kernel が中間値を書かない場合があるため）。

### 特定した M 依存 divergence（順に）

1. **bf16 GEMM（最初の差）**: `launch_gemm_bf16_wmma`（gfx1201）は
   `rows==1` で `gemv_bf16_kernel`（32 lane tree reduction）、`rows>=2` で
   `gemm_bf16_wmma_kernel`（WMMA K-tile）を使う。K 還元順序が別物で、
   M=1 と M=K+1 で bf16 の 1 ULP 差が出る。
   最初の差は常に layer 0 の最初の射影（GDN 層なら `L0.gdn_qkv`）。
2. **GDN recurrence**: chunk 内の複数行をまとめて処理し、chunk 全体の
   decay `exp(Σ oe)` と in-chunk 三角 solve を使う。1 行ずつ処理した場合と
   数値が一致しない（row 1 以降で差）。
3. **fused projection / prepare / attention-prep**: 融合 kernel も同じ理由で
   M 依存。`LinearQkv` / `LinearGateUp` / `LinearGdnProj` / `GdnPrepare` /
   `AttentionPrep` が該当。

attention core（decode と prefill の別 kernel）と GDN conv、RMSNorm、RoPE、
residual、elementwise、sampling、lm_head は M 非依存（bit-exact）だった。

### VERIFY_EXACT の実装

`PHASESHIFT_VERIFY_EXACT=1`（default OFF）。

1. **exact bf16 GEMM** `gemv_bf16_exact_rows`（`gemm_bf16_wmma.gfx1201.hip`）:
   `1 < rows <= 16` のとき、各 (row, out_feature) の K 還元を M=1 の
   `gemv_bf16_kernel` と完全に同一順序で行う。block は 8 output feature を担当し、
   weight を 1 回ロードして全 row の accumulator を更新するため weight 再利用を保つ。
   M=1 は従来どおり gemv（最速）。
2. **exact GDN recurrence** `gdn_recurrence_wmma_impl<kLossy, kSerialRows=true>`:
   chunk 内で 1 行ずつ、nval=1 の数値契約で逐次処理する。M=1 の繰り返しと同じ。
3. **fused kernel の回避**: `LinearQkv` / `LinearGateUp` / `LinearGdnProj` /
   `GdnPrepare` / `AttentionPrep` を VERIFY_EXACT では使わず、
   個別の exact GEMM / optimized kernel へ落とす。

### 結果

`test_qwen35_mtp_verify_divergence`（fusion ON, VERIFY_EXACT, K=1/4/8）:
全 layer tensor が bit-exact、`MTP_VERIFY_DIVERGENCE: NONE`。

`test_qwen35_mtp_spec_perf`（optimized, 8 prompt × 128 token, VERIFY_EXACT）:

| K | token_match |
| --- | --- |
| 1 | 8/8 |
| 2 | 8/8 |
| 4 | 8/8 |
| 8 | 8/8 |

Gate 4 の blocker（spec ON != spec OFF）は解消。`spec_off` は
43.9 ms/token で、融合 kernel を多数無効化するため性能は大きく劣化する。
性能回復は Gate 4B/4C（exact 版の fused GEMM / attention-prep / gdn-prepare）で行う。

---


---

## 7.65 MTP Gate 4B: VERIFY_EXACT の verify role 限定と fallback 除去（2026-09-20）

### 問題

Gate 4A.1 の `PHASESHIFT_VERIFY_EXACT=1` は kernel dispatch が env を直接参照し、
通常の M=1 decode にも作用していた（spec_off 18.2 → 43.9 ms/token）。
また M 依存 fused kernel を無効化していたため correctness fallback が走っていた。

### 変更

- `ExecutionRole`（Decode/Prefill/Verify）と `VerifyNumericMode`（Fast/Exact）を
  `execution_types.h` に追加。`ScheduledBatch` が `speculative_verify` と
  `verify_numeric_mode` を持つ。
- `Executor` が batch から role/mode を決め、`HostExecutionContext` へ伝搬。
  dispatch は `verify_exact_active(ctx)` のみを見る（kernel dispatch から env 参照を排除）。
- SpecDecoder の verify batch と partial-accept rerun batch だけが Verify/Exact。
  M=1 decode と prefill は Fast のまま。
- VERIFY_EXACT は
  - 非 fused bf16 linear を `launch_gemm_bf16_wmma_verify_exact`（M=1 gemv と同一
    K 還元順序、weight は 1 回ロード）へ、
  - GDN recurrence を `launch_gdn_recurrence_f32_wmma_serial`（nval=1 逐次）へ
    置き換える。
- **fused kernel は無効化しない。** fused projection / GdnPrepare / AttentionPrep の
  M 依存は harness 上の stale intermediate 由来の誤検出で、実体は M 非依存だった。
  decode と verify が同じ fused kernel を使うため bit-exact になる。
- HIP graph cache key に `verify_exact` を追加（Fast/Exact の graph 混在を防止）。
- fused bf16 GEMM の M=1 oracle と generic GEMV は異なるため、fused を無効化して
  generic GEMV へ寄せるのではなく、fused をそのまま共有する方式を採る。

### 結果（同一 session、8 prompt × 128 token、optimized）

| K | FAST token_match | EXACT token_match | FAST ms/token | EXACT ms/token |
| --- | --- | --- | --- | --- |
| off | — | — | 17.84 | 17.86 |
| 1 | 4/8 | 8/8 | 23.44 | 19.67 |
| 2 | 3/8 | 8/8 | 23.41 | 19.54 |
| 4 | 5/8 | 8/8 | 25.82 | 21.71 |
| 8 | 4/8 | 8/8 | 30.54 | 28.00 |

- spec_off は FAST baseline を維持（FAST decode path 不変）。
- EXACT verify は fallback 0。exact GEMM（gemv 順序、weight 1 回読み）が
  small-M WMMA より速いため、EXACT の方が FAST より高速な K もある。
- `test_qwen35_mtp_verify_divergence`（A=FAST decode、B=EXACT verify、fusion ON）:
  全 layer boundary bit-exact、`MTP_VERIFY_DIVERGENCE: NONE`。
  （layer 内部の fused intermediate は materialize されず stale になるため、
  trace 比較は layer boundary のみ信頼する。）

### 追加 required test

- `test_gemm_bf16_verify_exact`: `rows in {2,3,5,6,9,16}`、実 geometry、
  K tail、stride>K について、exact rows kernel == M=1 を row ごとに個別実行した
  結果と **byte 単位で一致**（16 checks PASS）。
- `test_verify_exact_scope`: `verify_exact_active` の truth table
  （decode/prefill は Exact 設定でも exact を使わない、verify+Fast も使わない）。

### 未了（Gate 4C 以降）

- GPU 側 accept reduction（D2H 縮約）。
- fused fused exact kernel は不要と判明したため、4B の「exact fused projection」は
  実装せず（不要）。fallback attribution は fallback 0 のため自明。

---


---

## 7.66 MTP Gate 4C: verification D2H / sync の baseline（2026-09-20）

`test_qwen35_mtp_spec_perf` に 1 iteration あたりの D2H 回数と target forward 回数を
出力させ、VERIFY_EXACT で測定した（8 prompt × 128 token）。

| K | updates | d2h | d2h/iter | target fwd | fwd/iter |
| --- | --- | --- | --- | --- | --- |
| off | 1024 | 1024 | 1.00 | 1024 | 1.00 |
| 1 | 663 | 663 | 1.00 | 970 | 1.46 |
| 2 | 527 | 527 | 1.00 | 876 | 1.66 |
| 4 | 433 | 433 | 1.00 | 842 | 1.94 |
| 8 | 421 | 421 | 1.00 | 842 | 2.00 |

D2H は **1 iteration あたり 1 回**で、payload は `(K+1) * sizeof(int32)`（最大 36 byte）
のみ。logits や全 position 情報は host へ戻していない。blocking sync も 1 回。

したがって Gate 4C の「GPU 側 accept reduction」で削減できる余地は
数 byte 程度で、支配項ではない（GDN snapshot と同様）。

`SKIPPED — NON DOMINANT` と判断する。GPU accept-reduce kernel は実装しない。
将来 D2H が増えた場合（例: full logits を host で扱う構成）に再検討する。

---


---

## 7.67 MTP Gate 4E: fixed-K qualification / break-even（2026-09-20）

自然 corpus（8 カテゴリ × PP32/128/512/2048、53 prompt、TG256、VERIFY_EXACT、
optimized）で K=1/2/4/8 の cost と acceptance を分離測定した。
artifact は `artifacts/mtp_gate4e/`（raw_iterations.csv / aggregate.csv /
summary.md / acceptance.json / corpus_manifest.json / environment.json）。

### 集計（iteration 数で重み付け）

| K | updates | emitted/update | accepted/update | ms/update | ms/token | draft ms/update | verify ms/update |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8451 | 1.602 | 0.602 | 28.31 | 17.67 | 3.46 | 24.76 |
| 2 | 6740 | 2.011 | 1.011 | 36.44 | 18.12 | 6.70 | 29.60 |
| 4 | 5394 | 2.515 | 1.515 | 50.08 | 19.91 | 13.18 | 36.84 |
| 8 | 5259 | 2.580 | 1.580 | 66.80 | 25.89 | 26.13 | 40.66 |

target-only: PP32-2048 で 17.49〜17.84 ms/token。

### acceptance by draft position（pooled）

| K | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 60.2% | - | - | - | - | - | - | - |
| 2 | 65.2% | 35.9% | - | - | - | - | - | - |
| 4 | 78.0% | 47.6% | 20.9% | 5.0% | - | - | - | - |
| 8 | 80.8% | 49.5% | 21.0% | 5.4% | 1.2% | 0.2% | 0.0% | 0.0% |

条件付き: P(2|1)=61%, P(3|1,2)=43%, P(4|1..3)=25%, p5 以上は ~20% だが母数が
小さく、position 5-8 の寄与は 1% 程度。K8 の後半はほぼ無駄。

### break-even（speedup = target ms/token / spec ms/token）

| ctx | K1 | K2 | K4 | K8 |
| --- | --- | --- | --- | --- |
| 32 | 1.036 | 1.059 | 0.953 | 0.721 |
| 128 | 1.017 | 0.985 | 0.916 | 0.708 |
| 512 | 1.004 | 0.958 | 0.880 | 0.686 |
| 2048 | 0.934 | 0.893 | 0.801 | 0.619 |

- 短 context では K1/K2 が break-even を超える。長 context では全 K で負ける。
- カテゴリでは `repetitive` が突出（K1 1.175, K2 1.116, K4 1.130）。
- 必要 cost 削減率: ctx2048 K8 で 38.1%、ctx2048 K4 で 19.9%、ctx32 K4 で 4.7%。

### cost 分解

- draft ms/update は K にほぼ線形（K1 3.46 → K8 26.13）。
- verify ms/update は M=2→M=9 で 24.8 → 40.7（row 単価は逓減）。
- verify が常に支配項（K1 で 87%、K8 で 61%）。

### Gate 3 residual closure（`test_qwen35_mtp_gate3_replay`）

- target canonical KV serial replay: bit-exact PASS。
- GDN recurrent + conv state serial replay: bit-exact PASS。
- third real prompt spec ON/OFF: token exact PASS。
- real prompt EOS truncation: PASS。

---


---

## 7.69 MTP Gate 5A: draft cost profile と exact LM-head の評価（2026-09-20）

27B-PSQ（PP32、Gate 4E の aggregate）:

| K | draft ms/update | verify ms/update | draft ms/draft | emitted/update | ms/token |
| --- | --- | --- | --- | --- | --- |
| 1 | 7.60 | 52.18 | 7.60 | 1.61 | 37.1 |
| 2 | 14.45 | 62.88 | 7.22 | 1.99 | 38.9 |
| 4 | 27.87 | 74.82 | 6.97 | 2.53 | 40.6 |
| 8 | 54.46 | 85.26 | 6.81 | 2.83 | 49.4 |

target-only: 37.2 ms/token。

### MTP 1 draft step の内訳（27B、計算から）

- lm_head: tied embed [248320, 5120] bf16 = 2.543GB → DRAM 636GB/s で **4.0ms**。
  draft 1 step（6.8〜7.6ms）の **約 55〜58%**。
- 残り（MTP decoder layer + norm + host/launch gap）: 約 2.8ms。

### exact fused LM-head + argmax の評価

- 現行: lm_head が full logits[V] f32 を書く（1MB/row）→ argmax kernel が読む。
  materialize の増分は write 1MB + read 1MB = 2MB。
- 2MB / 2.543GB（weight read）= **0.08%**。
- したがって fused argmax による削減は測定 noise 以下で、**採用しない**（§15 の条件）。
- ただし dynamic policy の confidence（top1-top2 margin）には top2 が必要なため、
  `launch_argmax_f32_top2`（exact、既存 argmax と top1 完全一致）を実装した。

### 結論（27B）

draft cost の本命は lm_head の weight read でも full-logits materialize でもなく、
**不要な draft step の削減**（K=0 含む early stop）。verify が最大支配項であり、
K1 でもほぼ break-even なので、acceptance が低い iteration で draft を打ち切る効果が大きい。

---


---

## 7.70 MTP Gate 5B: confidence calibration と oracle K（27B-PSQ）（2026-09-20）

Gate 4E corpus（PP32、16 prompt、TG64）で static K1/2/4/8 を回し、各 draft step の
`top1-top2 margin` と accept を記録（`draft_margin_log.csv`）。

### margin → accept（pooled、6095 draft）

| margin | count | accept |
| --- | --- | --- |
| [0, 0.1) | 302 | 9.9% |
| [0.1, 0.25) | 476 | 14.5% |
| [0.25, 0.5) | 595 | 16.8% |
| [0.5, 1.0) | 922 | 20.3% |
| [1.0, 2.0) | 1145 | 29.2% |
| [2.0, 4.0) | 1213 | 43.8% |
| [4.0, inf) | 1442 | 65.4% |

### 27B の break-even

- target-only: 37.2 ms/token。
- draft 1 step: 6.8〜7.9ms。verify の row 追加単価: M=2 が target +15ms、
  M=2→3 +10.7ms、M=5→9 +2.6ms/row。
- position i を追加する条件（概算）: `P(accept_i) × 37 > draft + incremental_verify`
  → 後段で `P > ~0.31`。margin ≈ 2.0 が境界。
- static K: K1 37.84 / K2 39.14 / K4 40.06 / K8 48.74 ms/token → **oracle = K1**。

### 実装

- `launch_argmax_f32_top2`（exact top1 = 既存 argmax と完全一致、top2 = 次点値）。
- `MtpDraftPolicy { dynamic, stop_margin, min_drafts, enable_discard, discard_margin }`。
- `MtpDraftPolicy` は MTP core から分離し、`SpecDecoderConfig` 経由で渡す。
- `SpecDraftStep.margin` / `SpecIterationOutput.num_drafts_generated` を追加。
- `mtp_generate_drafts` は各 step 後に margin < stop_margin なら打ち切り。
  `enable_discard` なら初回 margin < discard_margin で 0 draft（K=0）へ落とし、
  transaction rollback 後に M=1 decode する。
- actual_K で verify rows = actual_K+1 に縮小（fixed-K と同一 semantics）。

---


---

## 7.71 MTP Gate 5C: dynamic draft depth（27B-PSQ）（2026-09-20）

### 実測（PP32、16 prompt、TG64、VERIFY_EXACT + PSQ4 batch invariant）

| mode | updates | emitted/update | ms/update | ms/token | tok/s | target比 | draft ms/update |
| --- | --- | --- | --- | --- | --- | --- | --- |
| target-only | — | 1.00 | 37.17 | 37.17 | 26.9 | 1.000 | — |
| static K1 | 633 | 1.600 | 60.65 | 37.90 | 26.4 | 0.981 | 7.98 |
| static K2 | 515 | 1.983 | 77.73 | 39.21 | 25.5 | 0.948 | 15.17 |
| static K4 | 402 | 2.572 | 103.22 | 40.13 | 24.9 | 0.926 | 29.26 |
| static K8 | 353 | 2.915 | 142.30 | 48.81 | 20.5 | 0.761 | 57.26 |
| dynamic (tau=2.0, Kmax=8, min=1) | 467 | 2.178 | 79.04 | 36.29 | 27.6 | 1.024 | 15.60 |
| **dynamic + discard (tau=2.0, discard=2.0)** | — | 1.965 | **67.80** | **34.50** | **29.0** | **1.077** | 15.47 |

- 全 mode で `TOKEN_MATCH=80/80`（exact 維持）。
- dynamic+discard で **target-only を 7.7% 上回る**（29.0 vs 26.9 tok/s）。
- discard は初回 margin < 2.0 のとき draft を破棄して K=0（verify M=1）へ落とす。
  draft を捨てても verify row が 1 減る（M=2→1 で約 15ms）ため正味で得。

### 設計

- policy は MTP core から分離（`MtpDraftPolicy`、`SpecDecoderConfig` 経由）。
- confidence は `launch_argmax_f32_top2` の top1-top2 margin。full logits は既存の
  argmax 経路のまま（fused 化は §7.69 のとおり効果がないため不採用）。
- actual_K で verify rows = actual_K+1。K=0 は transaction rollback → M=1 decode。
- static mode は維持（policy 無効時は完全に従来動作）。

### Gate 5A/5D

- 5A（exact fused LM-head）: draft の支配項は weight read で、materialize 削減は
  0.08% のため不採用。top2 kernel のみ confidence 用に採用。
- 5D（INT2 近似 head）: exact + dynamic で target-only を上回ったため、§67 に従い
  今回は導入しない（次の伸びしろとして残す）。

### dynamic correctness（27B-PSQ）

`test_qwen35_mtp_gate3_replay` に強制 policy を入れ、K/stop/discard を固定して
serial（M=1）と比較:

| 強制 policy | 結果 |
| --- | --- |
| stop after 1（K=1 相当） | PASS（tokens/KV/GDN replay bit-exact） |
| stop after 4（K=4 相当） | PASS |
| discard all（K=0 相当） | PASS |
| mixed（stop=2.0, discard=2.0） | PASS |
| EOS + K=0 | PASS（K=0 経路で EOS 判定を追加修正） |

- dynamic+discard の実測で `TOKEN_MATCH=80/80`（static K1/2/4/8 + dynamic）。
- top2 は policy 有効時のみ計算（`collect_top2`）。static mode は従来挙動のまま。

---


---

## 7.72 MTP 全 stage 内訳（27B-PSQ、per emitted token）（2026-09-20）

`spec_decoder_step` の wall clock を stage 別に集計（PP32、16 prompt、TG64、VERIFY_EXACT）。

| mode | E/update | draft ms/tok | verify ms/tok | state+launch ms/tok | **total ms/tok** | tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| static K1 | 1.600 | 4.76 | 32.90 | 0.006 | **37.67** | 26.5 |
| static K2 | 1.983 | 7.30 | 31.55 | 0.004 | 38.86 | 25.7 |
| static K4 | 2.572 | 10.85 | 28.76 | 0.003 | 39.61 | 25.3 |
| static K8 | 2.915 | 18.71 | 29.22 | 0.003 | 47.93 | 20.9 |
| **dynamic+discard** | 1.965 | **7.87** | **26.61** | 0.004 | **34.48** | 29.0 |

- `draft` = `mtp_generate_drafts`（MTP decoder + lm_head + argmax/top2 + sync）。
- `verify` = target forward（M=actual_K+1）+ sampling、および K=0 の M=1 decode。
- `state+launch` = transaction begin/rollback/commit + GDN snapshot/restore + 未帰属。
  **0.004 ms/token（0.01%）で、全 stage の和は total に一致する。**
- したがって drafter + target verify + sampling + state handling + launch overhead を
  全部足すと、K1 で **約 37.7ms**、dynamic+discard で **約 34.5ms**（≤ ~38ms）。
- launch overhead は draft/verify の wall clock に含まれており、別途の大きな隠れコストはない。
- verify が依然として最大項（DYN で 26.6 ms/token = 全体の 77%）。次の伸びしろは
  verify 本体の高速化、または INT2 近似 head（5D）で draft の weight read を削ること。

---


---

## 7.73 MTP draft の Correctness fallback 除去（27B-PSQ）（2026-09-20）

`select_linear_implementation` は `(family, out_features, k)` のホワイトリスト方式で、
載っていない形状は `LinearImplementation::Correctness` に落ちる。27B MTP 層の bf16 形状
`{out=5120, k=10240}` と `{out=12288, k=5120}` が漏れており、draft と MTP warmup が
`model_dispatch_kernel`（Correctness fallback、**約 2.5ms/op**）に落ちていた。

- `PHASESHIFT_LINEAR_DEBUG` で miss 形状を検出（target-only では出ない MTP 専用形状）。
- 上記 2 形状を `kLinearShapeRules` に追加。

| K | draft_ms/update | verify_ms/update | ms/token | 改善 |
| --- | --- | --- | --- | --- |
| 1 | 7.63 → **4.68** | 52.68 → 52.64 | 37.90 → **35.77** | -2.13 |
| 2 | 14.48 → **8.70** | 62.59 → 62.76 | 39.21 → **35.88** | -3.33 |
| 4 | 27.93 → **16.72** | 74.00 → 73.93 | 40.13 → **34.87** | -5.26 |
| 8 | 54.58 → **32.74** | 85.20 → 85.62 | 48.81 → **40.14** | -8.67 |

draft 1 step あたり約 2.7ms 削減。`TOKEN_MATCH=64/64`（exact 維持）。draft の数値は
変わるが target 検証があるため `spec ON == spec OFF` は不変。


---

## 7.74 Verify 本体の M 依存の正体: partial accept の rerun forward（27B-PSQ）

`test_qwen35_mtp_verify_divergence` に純粋な target forward のタイミングを追加して測定:

| forward | ms |
| --- | --- |
| M=1（Decode, Fast） | 37.05 |
| M=2（Verify, Exact） | 37.83（比 **1.021**） |

**target forward 自体は M にほぼ依存しない**（weight-bound、18.9GB / 636GB/s ≈ 29.8ms）。
一方 gate4e の `verify_ms`（K1）は 52.6ms。差分は `spec_decoder_step` 内の
**partial accept 時の rerun forward** である:

- all-accept: verify の `final_hidden` をそのまま使い commit（追加 forward なし）。
- partial: GDN の逐次 recurrent state を accepted prefix まで戻す必要があり、
  GDN snapshot を復元した後に **もう一度 target forward（prefix 長 = accepted+1）** を回す。
  KV は truncate で済むが GDN state は再計算が必要なため。
- K=1 は accept≈0.6 なので平均 `0.6×37.8 + 0.4×(37.8+37.0) = 52.6ms` と一致。
  追加コストは構成上 **約 15ms/update（K1 で約 9.4 ms/token、全体の約 25%）**。

### 緩和

- dynamic + discard policy は、初回 margin が低い iteration を K=0（M=1 decode、rerun 不要）
  に落とすため、partial accept の rerun を実質的に回避する（§7.71）。
- 恒久的な除去には **GDN の row-0 state を verify 中に保存**し、partial accept 時に
  その state を復元して rerun を省略する方式が考えられる（48 layer × 3.1MB ≈ 150MB の
  追加 snapshot と kernel 変更が必要）。今回は未実装（次の最適化候補）。


---

## 7.75 Gate 5 最終結果（27B-PSQ、shape 修正 + dynamic + discard）

PP32、16 prompt、TG64、`VERIFY_EXACT` + `PSQ4_BATCH_INVARIANT`、`TOKEN_MATCH=80/80`。

| mode | E/update | draft ms/update | verify ms/update | ms/token | tok/s | target-only 比 |
| --- | --- | --- | --- | --- | --- | --- |
| target-only | 1.000 | — | 37.17 | 37.17 | 26.9 | 1.000 |
| static K1 | 1.603 | 4.69 | 52.66 | 35.79 | 27.9 | 1.039 |
| static K2 | 1.992 | 8.71 | 62.80 | 35.90 | 27.9 | 1.035 |
| static K4 | 2.599 | 16.73 | 73.96 | 34.89 | 28.7 | 1.065 |
| static K8 | 2.949 | 32.77 | 85.68 | 40.18 | 24.9 | 0.925 |
| **dynamic + discard** | 1.967 | 9.72 | 52.08 | **31.42** | **31.8** | **1.183** |

- shape 修正前の dynamic+discard は 34.50 ms/token（29.0 tok/s）→ **-8.9%**。
- `verify` は依然として最大項（DYN で 52.1ms/update）。内訳は target forward 約 38ms +
  partial accept の rerun 約 14ms（§7.74）。

---


---

## 7.76 all-accept 時の pending_hidden 行の誤り（27B-PSQ）（2026-09-20）

`spec_decoder_step` の all-accept 分岐は verify の `final_hidden` の **row 0** を
`pending_hidden` にコピーしていた。verify は `num_output_rows = actual_k+1` 行を
出力するので、正しくは最後に accept した行 `final_hidden[actual_k]`（= 次の pending
トークンの直前トークンの hidden）。MTP draft の入力が stale になっていた。

partial 分岐は rerun（`num_tokens = accepted+1`, `output_rows=1`）の row 0 を使うため
元から正しい。all-accept のみが誤っていた。

env A/B（prose、PP32、TG64）:

| K | E/update (row 0) | E/update (last) | ms/token (row 0) | ms/token (last) |
| --- | --- | --- | --- | --- |
| 1 | 1.587 | **1.764** | 36.33 | **29.08** |
| 2 | 2.098 | **2.228** | 33.93 | **31.04** |
| 4 | 2.453 | **2.600** | 37.01 | **35.25** |
| 8 | 2.826 | 2.826 | 41.96 | 42.06 |

`TOKEN_MATCH=8/8`（exactness 不変。draft の入力のみ変わる）。
K1 で accept が大きく改善（E 1.59 → 1.76、ms/token -20%）。

### 7.76 反映後の最終値（27B-PSQ、PP32、TG64、TOKEN_MATCH=80/80）

| mode | E/update | draft ms/upd | verify ms/upd | ms/token | tok/s | target-only 比 |
| --- | --- | --- | --- | --- | --- | --- |
| target-only | 1.000 | — | 37.17 | 37.17 | 26.9 | 1.000 |
| **static K1** | 1.808 | 4.69 | 45.07 | **27.53** | **36.3** | **1.350** |
| static K2 | 2.271 | 8.71 | 58.09 | 29.42 | 34.0 | 1.263 |
| static K4 | 2.762 | 16.73 | 73.65 | 32.73 | 30.6 | 1.136 |
| static K8 | 2.966 | 32.78 | 85.70 | 39.96 | 25.0 | 0.930 |
| dynamic+discard | 2.075 | 10.06 | 52.27 | 30.04 | 33.3 | 1.237 |

- draft の質が上がり accept が改善（K1 の E 1.603 → 1.808、reject0 39.7% → 19.2%）。
- 結果として **static K1 が最良**（27.53 ms/token、target-only 比 1.350x）。
  dynamic policy は現行パラメータだと K>1 を選びすぎて K1 に劣るため再調整が必要。

---


---

## 7.77 n-gram tail draft（参照実装 Lossless dynamic MTP drafting 相当）（2026-09-20）

参照実装 vllm-mxfp4 の "Lossless dynamic MTP drafting" は
**per-request confidence gate + verbatim n-gram tail** で draft 深度を可変にする
（MTP draft forward の直列ループを早く止め、不足分を文脈中の一致 n-gram の続きで埋める）。
lossless（target が検証する）である点は本側の dynamic policy と同じ。

実装:
- `SpecDecoder.token_history`（prompt + 生成トークン、上限 8192）を保持。
- `MtpDraftPolicy` の early stop 後、`append_ngram_tail` が
  `history + pending + MTP drafts` の末尾 n トークンに一致する過去位置を探し、
  その続きを verbatim で draft に追加（上限 `ngram_max_tail`、MTP forward なし）。
- n-gram draft は MTP KV を進めないため、`SpecVerifyResult.num_mtp_drafts` を追加し
  `spec_transaction_commit` の MTP kept 数を `min(accepted+1, mtp_drafts)` に制限。

計測（27B-PSQ、dynamic tau=2.0、Kmax=8、`TOKEN_MATCH` 全一致）:

| 条件 | n-gram | E/update | ms/token | draft ms/update |
| --- | --- | --- | --- | --- |
| code PP32 | off | 2.932 | 28.18 | 15.59 |
| code PP32 | on(4,8) | 2.955 | 28.35 | 15.58 |
| json PP32 | off | 3.175 | 25.74 | 14.89 |
| json PP32 | on(4,8) | 3.175 | 25.79 | 14.89 |
| prose PP32 | off | 2.321 | 29.36 | 8.88 |
| prose PP32 | on(3,8) | 2.321 | 30.58 | 8.88 |
| **code PP512** | off | 3.722 | 21.63 | 16.09 |
| **code PP512** | **on(4,8)** | **3.938** | **21.40** | **15.56** |

- 短い文脈（ctx32）では履歴が足りず一致がほぼ発火しない（反復的な code/json でも
  E が伸びず、verify の M が増える分だけわずかに悪化）。
- 長い文脈（ctx512）では **E/update +5.8%、draft ms/update -3.3%、ms/token -1.1%**。
- 既定は無効（`SpecDecoderConfig::ngram_n = 0`）。長文脈・反復的な配信で有効化する。
