> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# lm_head High-Entropy Fallback PoC（margin 検出, F0-F5）

## 0. 目的

Target fast path

```
INT2 coarse → coarse Top-N → PSQ8 exact rerank → next token
```

は自然分布では高精度だが、**高エントロピー / flat-logit 状態**では PSQ8 Top1 が candidate pool から
漏れる。そこで

```
confident  → PSQ8 candidate rerank
ambiguous  → full PSQ8
```

の二段構成とする。fallback 判定に full PSQ8 logits を使ってはならない（高速化の意味が消える）。
本 PoC は **coarse Top-N の margin** を flatness の proxy として使い、F0-F5 までを評価する。

数学的 certificate は使わない。

## 1. branch / revision

| 項目 | 値 |
| --- | --- |
| 分岐元 | `3a8564e5` |
| branch | `poc/target-lm-head-entropy-fallback` |
| GPU | AMD Radeon AI PRO R9700 (gfx1201)、物理 GPU3 のみ |
| target model | `models/Qwen3.8-27B-PSQ` |
| 診断実装 | `tests/unit/test_target_lm_coarse_rerank.hip`（per-row CSV） |
| 解析 | `tools/analyze_lm_head_margin_fallback.py` |

## 2. F1 — diagnostic capture

`test_target_lm_coarse_rerank.hip` が row ごとに CSV を出す。`PHASESHIFT_TARGET_LM_DIAG_CSV` と
`PHASESHIFT_TARGET_LM_DATASET` で指定する。列:

```
dataset,seg,split,row,match64,match127,top1,m2,m8,m32,m64,m127,oracle_rank
```

- `matchN` = INT2 coarse Top-N + PSQ8 rerank Top1 が full PSQ8 Top1 と一致したか。
- `mK` = coarse logits の rank1 - rankK（`m2` は top1-top2、`m127` は top1-top127）。
- `oracle_rank` = full PSQ8 Top1 が coarse Top-127 内で何位か（-1 は圏外）。
- split は segment 単位で `seg % 4`（0,1=calibration 50%、2=validation 25%、3=qualification 25%）。
  隣接 token は split を跨がない。

margin は既存 `coarse_topn` の出力（`top_values`）だけで計算でき、追加コストはほぼゼロ。
新しい full-vocab kernel は追加していない。

## 3. dataset / F0 baseline

`PHASESHIFT_TARGET_LM_SEG=512`。全 194,922 row。`hient` は temperature 1.3 / top_p 0.98 で生成した
高エントロピー stress set（多言語断片が混ざる劣化テキスト）。

| dataset | rows | N=64 mismatch | N=64 rate | N=127 mismatch | N=127 rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| prose（英文） | 2,600 | 2 | 7.69e-4 | 1 | 3.85e-4 |
| ja（日本語） | 12,000 | 2 | 1.67e-4 | 1 | 8.33e-5 |
| code（src） | 12,000 | 0 | 0 | 0 | 0 |
| json（structured） | 6,378 | 0 | 0 | 0 | 0 |
| math（数値・式） | 12,000 | 8 | 6.67e-4 | 1 | 8.33e-5 |
| mixed204k（KVPrune） | 137,944 | 80 | 5.80e-4 | 13 | 9.43e-5 |
| **hient（high-entropy）** | 12,000 | **808** | **6.73e-2** | **268** | **2.23e-2** |

## 4. F2 — failure separation

margin_64 の percentile（correct / mismatch）:

| dataset | correct p05 | correct p50 | mismatch p05 | mismatch p50 |
| --- | ---: | ---: | ---: | ---: |
| code | 4.82 | 11.79 | — | — |
| json | 6.06 | 13.63 | — | — |
| prose | 4.18 | 8.78 | 3.46 | 4.50 |
| ja | 4.25 | 8.75 | 4.09 | 4.32 |
| math | 5.25 | 12.39 | 4.05 | 5.32 |
| mixed204k | 4.11 | 7.96 | 2.71 | 4.12 |
| hient | 2.01 | 2.83 | 1.79 | 2.38 |

margin_127 でも同様（mixed204k: correct p05 5.03 / mismatch p05 3.31、hient: correct p05 2.36 /
mismatch p05 2.14）。

**観測**: mismatch row は確かに低 margin 側へ寄る（mixed204k・math では明確）が、
correct row の下位裾と広く重なる。hient は分布がほぼ完全に重なる。

## 5. F4/F5 — threshold sweep

`margin_N < threshold` を fallback 条件とする。threshold は **calibration split で fit** し、
validation+qualification で評価する（同じ row で fit と評価をしない）。

### global threshold（全 dataset の calibration から 1 本、N=64, margin_64）

| fallback | threshold | eval | fast | fallback rows | raw mismatch | caught | uncaught | final mismatch | est_us |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.11% | 1.784 | 96,128 | 96,024 | 104 | 405 | 15 | 390 | 390 | 1003 |
| 0.85% | 2.243 | 96,128 | 95,315 | 813 | 405 | 116 | 289 | 289 | 1019 |
| 5.22% | 3.200 | 96,128 | 91,111 | 5,017 | 405 | 355 | 50 | 50 | 1113 |
| 10.29% | 4.169 | 96,128 | 86,236 | 9,892 | 405 | 386 | 19 | 19 | 1221 |
| 20.58% | 5.282 | 96,128 | 76,346 | 19,782 | 405 | 401 | 4 | 4 | 1442 |

（est_us = 1000.7 + fallback_rate × 2144.4。full PSQ8 は 2144.4 us。）

### margin の選択（global threshold、uncaught 合計 / 括弧内 natural のみ）

| margin | 2% fallback | 5% | 10% | 20% |
| --- | ---: | ---: | ---: | ---: |
| m2 | 377 (45) | 331 (41) | 261 (35) | 156 (24) |
| m8 | 295 (42) | 209 (33) | 87 (21) | 11 (6) |
| m32 | 211 (44) | 66 (35) | 21 (20) | **3 (3)** |
| m64 | 196 (45) | 50 (39) | 19 (19) | 4 (4) |
| m127 | 184 (45) | 46 (40) | 26 (26) | **3 (3)** |

**大きい N の margin の方が良い検出器**である（m2 はほぼ無力）。

### natural のみ（hient を除外して fit）の場合

| margin | q | threshold | fallback | raw mismatch | uncaught | est_us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m32 | 5% | 3.312 | 5.42% | 46 | 20 | 1117 |
| m64 | 5% | 4.230 | 5.40% | 46 | 19 | 1117 |
| m127 | 5% | 5.156 | 5.45% | 46 | 23 | 1118 |

### per-dataset threshold（参考。production では dataset 既知でないため oracle 的）

| dataset | raw mismatch | 最小の final mismatch | そのときの fallback | est_us |
| --- | ---: | ---: | ---: | ---: |
| prose | 1 | **0** | 10.94% | 1235 |
| ja | 2 | **0** | 11.65% | 1250 |
| code | 0 | 0 | 0.03% | 1001 |
| json | 0 | 0 | 0.20% | 1005 |
| math | 4 | **0** | 10.43% | 1224 |
| mixed204k | 39 | 1 | 20.71% | 1445 |
| hient | 359 | 68 | 18.68% | 1401 |

## 6. 判定 — margin-only fallback は NO-GO

§49 の NO-GO 条件を両方満たす。

- **mismatch と correct の margin 分布が重なる**: mixed204k では mismatch p50 = 4.12 に対し
  correct p05 = 4.11。hient ではほぼ完全に重なる。単一 threshold では分離できない。
- **final mismatch を抑えるには fallback 率が過大**: global 20.58% でも uncaught 4、
  natural のみ 5.4% では uncaught 20/46。hient の mismatch を捕捉するには 93% 超が必要で、
  その場合 est_us ≈ 3000 us > full 2144 us となり高速化が消える。

ただし重要な限定:

- production の **greedy** 文脈では raw mismatch 自体が小さい（natural eval 46/96,128 = 4.8e-4）。
- closed-loop（16 prompt × 128 token）では natural で 0〜1 token、**hient でも 0/2,048**
  （`lm_head_int2_coarse_psq8_rerank_poc.md` §6）。hient の static mismatch は greedy では
  再現しない。
- したがって「margin で fallback して精度を保証する」用途には不足だが、
  「greedy production の素の fast path」としては実用域にある。

## 7. best (N, threshold) と closed-loop への GO/NO-GO

| 前提 | N | margin | threshold | fallback | final mismatch | est_us | speedup |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| natural のみ | 64 | m32 | 3.31 | 5.4% | 20 / 96,128 (2.1e-4) | 1117 | 1.92x |
| hient 込み | 64 | m32 | 4.27 | 20.6% | 3 / 96,128 (3.1e-5) | 1442 | 1.49x |

- margin-only で final mismatch 0 は **per-dataset threshold でしか達成できない**（および
  mixed204k は 1、hient は 68 が残る）。production の単一 threshold では 0 にできない。
- よって §49 に従い、次段は **複数 margin / Top-K entropy / learned confidence classifier** を
  検討する。F6（adaptive closed-loop）へは margin-only では進めない。

## 8. 再現

```
cmake --build build --target test_target_lm_coarse_rerank --parallel

export HIP_VISIBLE_DEVICES=3 PHASESHIFT_MODEL_DIR_DFLASH2_TARGET=models/Qwen3.8-27B-PSQ
export PHASESHIFT_TARGET_LM_SEG=512
for ds in prose ja code json math hient mixed204k; do
  PHASESHIFT_TARGET_LM_DIAG_CSV=/tmp/opencode/lm_head_margin/$ds.csv \
  PHASESHIFT_TARGET_LM_DATASET=$ds \
  PHASESHIFT_TARGET_LM_CORPUS=<corpus> \
  ./build/tests/test_target_lm_coarse_rerank
done

python3 tools/analyze_lm_head_margin_fallback.py '/tmp/opencode/lm_head_margin/*.csv'
```

## 9. その後（方針変更）

本 Gate の NO-GO を受けて、margin fallback は **採用しない**（複数 margin / Top-K entropy /
learned classifier も保留）。理由は「high-entropy hidden を検出できないこと」よりも
「**その high-entropy hidden に production greedy が実際に到達するのか**」が先だからである。

static corpus は **adversarial / stress qualification** として役割を変え、production 判断は
`lm_head_onpolicy_greedy_shadow_poc.md` の **on-policy greedy shadow**（baseline は full PSQ8
greedy、proxy は診断のみ）で行う。そこでは static より一桁低い mismatch が観測されている
（例: hient N=64 で static 6.73e-2 → on-policy 2.73e-4）。

