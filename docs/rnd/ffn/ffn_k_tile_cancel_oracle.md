# FFN down_proj K256 Cancellation-Aware Subset Oracle Gate

更新日: 2026-09-24
branch: `poc/ffn-k256-cancel-oracle`
harness: `tools/rnd/ffn_k_tile_oracle/`

## 1. Goal

前回 Gate の N64 × K256 は SMALL_L2（group 単体の `||c_g||²` の小さい順に skip）で
+2% 以内 9.38%、+5% 以内 18.10% だった。

今回の目的は、skip 集合 `S` の error vector

```text
e(S) = Σ_{g∈S} c_g
```

の cross-term / cancellation まで考慮して `||e(S)||²` を最小化する subset を選んだとき、
N64 × K256 の headroom が 25〜40% まで伸びるかを exact partial で確認すること。

predictor は実装しない。INT2 / activation-only / weight-norm / production sparse kernel /
runtime selector は対象外。構造的 headroom の有無のみを確定する。

## 2. Previous Oracle result

`docs/perf/ffn_k_tile_oracle.md`（revision `49245135`）より:

| method | tile | +2% 以内の最大 skip | +5% 以内の最大 skip |
| --- | --- | ---: | ---: |
| SMALL_L2 | N64 | 9.38% | 18.10% |
| SMALL_L2 | N16 | 14.90% | 25.05% |

SMALL_L2 は group を独立に選ぶため、`c_a ≈ -c_b` のような打ち消し合いを利用できない。

## 3. Why SMALL_L2 is insufficient

`y = Σ_g c_g` であり、skip 後の local error は `e = Σ_{g∈S} c_g` である。個々の
`||c_g||²` が小さくても、選んだ group の寄与が同じ方向に足し合わされば `||e||²` は
小さくならない。逆に `||c_g||²` が大きくても、複数 group が打ち消し合えば `||e||²` は
小さくなる。SMALL_L2 はこの後者を取り逃す。

## 4. Objective definition

各 (token, layer, N64 output tile) について 68 個の exact partial
`C = {c_0, ..., c_67}`、`c_g ∈ R^64` を得る。skip 集合 `S`（`|S| = k`）の local output error:

```text
e(S) = Σ_{g∈S} c_g
J(S) = ||e(S)||² = Σ_g ||c_g||² + 2 Σ_{g<h} c_g·c_h
```

最小化対象は `J(S)`。exact なのは `c_g` そのもので、subset search は heuristic である。
本 report では「Cancellation-Aware Subset Oracle（approximation）」と呼び、組合せ論的な
global exact optimum とは呼ばない。

## 5. Selector methods

| method | 内容 |
| --- | --- |
| `SMALL_L2` | `||c_g||²` の小さい順に k 個。前回方式の baseline |
| `RESIDUAL_GREEDY` | 残差 `e` に対し `||e + c_g||²` を最小化する group を逐次追加 |
| `PAIR_SEEDED_GREEDY` | 全 pair `C(68,2)=2278` の `||c_i+c_j||²` 上位 `seed_count=16` pair と SMALL_L2 最小 pair を seed に residual greedy を拡張し、最終 `||e||²` 最小の seed を採用 |
| `PAIR_GREEDY_SWAP` | 上記の後、selected `i` と unselected `j` の 1-swap を `||e-c_i+c_j||²` が改善する限り最大 8 回反復 |

すべて 1 行 = 1 (token, tile) の `C [R, 68, 64]` に対する torch 演算で実装し、group ごとの
GPU sync をしない。ranking には `||e||² + ||c_g||² + 2 e·c_g` を用いる。

## 6. Synthetic exhaustive validation

group 数 11、k=4、dim 8、batch 5 の random case で全 `C(11,4)=330` subset を列挙し、
`J / J_exact` を比較（4 seed 平均 / max）。

| method | mean J/exact | max J/exact |
| --- | ---: | ---: |
| SMALL_L2 | 3.967 | 4.560 |
| RESIDUAL_GREEDY | 1.483 | 1.755 |
| PAIR_SEEDED_GREEDY | 1.005 | 1.022 |
| PAIR_GREEDY_SWAP | 1.005 | 1.022 |

synthetic では pair-seeded greedy が exhaustive optimum にほぼ一致する。加えて
cancellation pair test（`c0=+10, c1=-10, k=2`）で SMALL_L2 が `{c2,c3}` を選ぶのに対し
PAIR_GREEDY_SWAP が `{c0,c1}` を選び `J=0` にできること、token ごと・tile ごとに独立な
pair を選ぶこと、同一入力で deterministic であること、skip=0 で exact full output に
一致することを unit test で確認する。

## 7. Pair cancellation statistics

sampled layer（0, 16, 28, 32, 48, 63）× 64 token × 全 80 tile の pair cosine と
best pair cancel ratio（`||c_i+c_j||² / (||c_i||²+||c_j||²)`、0 が完全打消し）。

| layer | cos mean | cos min | p01 | p05 | frac cos<-0.5 | frac cos<-0.7 | best ratio mean | best ratio min |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0201 | -0.982 | -0.291 | -0.199 | 1.19e-03 | 6.09e-04 | 0.5971 | 0.0286 |
| 16 | 0.0056 | -0.730 | -0.286 | -0.203 | 1.28e-05 | 1.71e-07 | 0.5936 | 0.3132 |
| 28 | 0.0043 | -0.614 | -0.291 | -0.207 | 1.75e-05 | 0.00e+00 | 0.5851 | 0.3997 |
| 32 | 0.0041 | -0.901 | -0.294 | -0.208 | 8.69e-05 | 5.66e-06 | 0.5828 | 0.1348 |
| 48 | 0.0025 | -0.956 | -0.292 | -0.207 | 1.09e-04 | 1.45e-05 | 0.5880 | 0.0656 |
| 63 | 0.0220 | -0.983 | -0.403 | -0.246 | 4.59e-03 | 1.12e-03 | 0.4823 | 0.0236 |

pair は平均ほぼ直交（cos mean ~0.005〜0.02）。強い打消し（cos<-0.5）は稀で最大 4.6e-3。
ただし best pair cancel ratio は平均 0.48〜0.60 程度まで下がり、最小では 0.02〜0.03 の
pair も存在する。すなわち「完全打消し pair は少数、弱い打消しは多数」である。

## 8. Diagnostic local-error comparison

sampled layer × 64 token × 80 tile（R=5120 rows/layer）の aggregate。
`improvement = mean J_small_l2 / mean J_method`。

| k | actual skip | SMALL_L2 mean J | RESIDUAL_GREEDY | PAIR_SEEDED_GREEDY | PAIR_GREEDY_SWAP | imp greedy | imp pair |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.1029 | 1.2121 | 0.8343 | 0.7732 | 0.7611 | 1.453 | 1.592 |
| 10 | 0.1471 | 2.0699 | 1.2565 | 1.1614 | 1.1226 | 1.647 | 1.844 |
| 14 | 0.2059 | 3.5406 | 1.8920 | 1.7506 | 1.6608 | 1.871 | 2.132 |
| 17 | 0.2500 | 4.9007 | 2.4450 | 2.2719 | 2.1270 | 2.004 | 2.304 |
| 20 | 0.2941 | 6.5104 | 3.0802 | 2.8737 | 2.6762 | 2.114 | 2.433 |
| 27 | 0.3971 | 11.304 | 5.0613 | 4.7890 | 4.4700 | 2.233 | 2.529 |

normalized 指標（layer 平均、k=17）:

| metric | SMALL_L2 | PAIR_GREEDY_SWAP | 比 |
| --- | ---: | ---: | ---: |
| mean R_output = J / ||y||² | 0.0991 | 0.0327 | 3.03 |
| mean R_group = J / Σ||c_g||² | 0.1287 | 0.0437 | 2.95 |
| median J | 0.2774 | 0.0859 | 3.23 |
| p90 J | 22.570 | 10.175 | 2.22 |
| p99 J | 50.291 | 20.911 | 2.41 |

**Stage A Gate: PASS**（k=17 で 2.30、k=20 で 2.43 ≥ 2.0）。full PPL へ進む。

## 9. Full PPL results

N64, K256, 2048-token teacher forcing, positions 2047。
baseline PPL = 7.056874、decomposed skip0 = 7.061929（+0.072%、parity PASS）。
物理 GPU2,3（logical 0,1）で測定。同一 device map では完全に再現する。

| skip groups | actual skip | SMALL_L2 dPPL% | RESIDUAL_GREEDY dPPL% | PAIR_GREEDY_SWAP dPPL% |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0000 | +0.072 | +0.104 | +0.072 |
| 7 | 0.1029 | +2.007 | +1.224 | +1.293 |
| 10 | 0.1471 | +3.792 | +1.454 | +1.695 |
| 14 | 0.2059 | +6.994 | +2.456 | +1.697 |
| 15 | 0.2206 | — | — | +2.273 |
| 16 | 0.2353 | — | — | +1.851 |
| 17 | 0.2500 | +9.312 | +3.442 | +2.885 |
| 20 | 0.2941 | +12.067 | +3.893 | +2.936 |
| 27 | 0.3971 | +23.662 | +5.721 | +4.714 |
| 28 | 0.4118 | — | — | +4.545 |
| 30 | 0.4412 | — | — | +6.857 |

PPL 実値（PAIR_GREEDY_SWAP）: k=7 7.148109, k=14 7.176660, k=17 7.260476,
k=20 7.264051, k=27 7.389538, k=28 7.377634, k=30 7.540788。
NaN / Inf は全 run で無し。

## 10. SMALL_L2 vs cancellation-aware comparison

補間による +2% / +5% 以内の最大 skip:

| method | +2% 以内 最大 skip | +5% 以内 最大 skip |
| --- | ---: | ---: |
| SMALL_L2 | 10.3% | 16.9% |
| RESIDUAL_GREEDY | 17.9% | 35.6% |
| PAIR_GREEDY_SWAP | 21.4%（最大測定 23.5% @ +1.85%） | 41.8% |

headroom 改善倍率: +2% で 2.1x、+5% で 2.5x。local error 改善（k=17 で 2.30x）と整合する。
k=27（39.71%）で SMALL_L2 +23.66% に対し PAIR_GREEDY_SWAP +4.71%（5.0x 改善）。
k=17（25.00%）で +9.31% に対し +2.885%（3.2x 改善）。

k=15/16 の非単調（+2.273 / +1.851）は device map 固定でも生じる選択境界の段差で、
+2% 近傍の headroom 推定は ±1pp 程度の不確かさを持つ。

## 11. Per-layer analysis

pair_greedy_swap の SMALL_L2 に対する improvement（per layer）:

| layer | k=14 | k=17 | k=20 | k=27 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.937 | 1.902 | 1.832 | 1.633 |
| 16 | 2.792 | 3.015 | 3.155 | 3.199 |
| 28 | 2.973 | 3.222 | 3.389 | 3.485 |
| 32 | 2.999 | 3.255 | 3.437 | 3.535 |
| 48 | 2.836 | 3.049 | 3.214 | 3.317 |
| 63 | 2.104 | 2.277 | 2.406 | 2.506 |

前回 worst cluster（L28〜L33）は今回むしろ改善が大きい（k=17 で 3.2x）。逆に L0 は
もともと error が小さく改善も 1.9x にとどまる。L63 は絶対 error が最大
（k=17 で SMALL_L2 28.14 → pair 12.36）で、集約 mean を支配する。層単位の崩壊はなく、
cancellation-aware 選択は全層で一貫して有効。

## 12. Gate decision

**HOLD**

根拠（実測値）:

* k=17（25.00% skip）で PAIR_GREEDY_SWAP は ΔPPL = **+2.885%**。GO 条件の
  「25% 以上 skip して ΔPPL ≤ +2%」を満たさない（+2% を 0.885pp 超過）。
* k=27（39.71% skip）で ΔPPL = **+4.714%** ≤ +5%。GO の「約 40% で +5% 以内」は満たす。
* したがって GO 条件の前半のみ未達。HOLD 定義「25% skip で +2% < ΔPPL ≤ +5% 程度」に
  一致する。
* SMALL_L2 に対する改善は大きく（+2% headroom 2.1x、+5% headroom 2.5x、k=27 で 5.0x の
  ΔPPL 改善）、cancellation-aware 選択の効果は明確。NO-GO 条件
  （改善が小さい / strong cancellation pair が少ないだけ / PPL が改善しない）には該当しない。

## 13. Runtime implications

* headroom は約 2 倍になった（+5% で 18% → 42%）。構造的には「K256 をまとめて消す」余地が
  存在することを確認した。
* ただし今回の selector は exact `c_g` 全件、`68×68` pair 行列、pair-seed greedy、
  1-swap refinement を使用し、production selector としては極めて高コスト。
* 25% 付近は +2.9%（わずかに +2% 超過）、40% 付近は +4.7%（+5% 未満）であり、
  selector overhead を回収できるかは break-even 次第。
* よって次に進む前に selector break-even を評価する。cheap predictor の研究はその後に限る。

## 14. Reproduction commands

```bash
W=.worktrees/ffn-k256-cancel-oracle
cd $W
python3 -m compileall tools/rnd/ffn_k_tile_oracle
python3 tools/rnd/ffn_k_tile_oracle/test_oracle.py
python3 tools/rnd/ffn_k_tile_oracle/test_cancel.py

# Stage A diagnostic
CUDA_VISIBLE_DEVICES=2,3 python3 tools/rnd/ffn_k_tile_oracle/run_cancel.py \
    --stage diagnostic --model-dir models/Qwen3.8-27B \
    --corpus artifacts/ffn_prune/corpus.psktok --mem-gib 27 \
    --k-group 256 --tile-width 64 --layers 0,16,28,32,48,63 \
    --skip-groups 7,10,14,17,20,27 --out-dir artifacts/ffn_k_tile_cancel_oracle

# Stage B full PPL
CUDA_VISIBLE_DEVICES=2,3 python3 tools/rnd/ffn_k_tile_oracle/run_cancel.py \
    --stage ppl --model-dir models/Qwen3.8-27B \
    --corpus artifacts/ffn_prune/corpus.psktok --mem-gib 27 \
    --k-group 256 --tile-width 64 --methods small_l2,residual_greedy,pair_greedy_swap \
    --skip-groups 0,7,10,14,17,20,27 --out-dir artifacts/ffn_k_tile_cancel_oracle
```

raw JSON / summary は `artifacts/ffn_k_tile_cancel_oracle/`（git 管理外）。
