> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# lm_head On-Policy Greedy Shadow Qualification

## 0. 目的

Target fast path

```
INT2 coarse → coarse Top-N → PSQ8 exact rerank → next token
```

は **greedy decode（temperature = 0）専用** である。したがって本当に見るべき分布は
「full PSQ8 greedy が実際に到達する hidden」であり、teacher-forced な static corpus ではない。

本 Gate では

- baseline: **常に full PSQ8 で greedy 生成**し、実際の次 token には必ず full PSQ8 の oracle を使う
- 同じ step の final_norm hidden から proxy（INT2 coarse → Top-N → PSQ8 rerank）も計算し、
  oracle と比較する（proxy は診断のみ）

ことで、**production trajectory を一切汚染せずに** proxy mismatch 率を測る。
これは static corpus を「adversarial / stress qualification」、on-policy shadow を
「production qualification」として役割分離するものである。

## 1. branch / revision

| 項目 | 値 |
| --- | --- |
| branch | `poc/target-lm-head-entropy-fallback` |
| 実装 commit | `be2e848d` |
| GPU | R9700 (gfx1201) |
| target model | `models/Qwen3.8-27B-PSQ` |
| 実装 | `tests/unit/test_target_lm_greedy_shadow.hip` |

## 2. 方法

- prompt は複数 corpus（prose / ja / code / json / math / hient / mixed204k）から取り、
  4 本に 1 本を long context（1,024 token）、残りを 128 token にする。
- generation は全 prompt で `temperature = 0`。
- batch 16 の multi-request decode で throughput を確保する。
- 各 decode step で:
  - full PSQ8 の `sampled_tokens` を oracle とし、**その token を次の入力に使う**。
  - 同じ `final_hidden`（final RMSNorm ONE_PLUS 後）から proxy を計算し、
    N=64 / N=127 の rerank Top1 を oracle と比較する。
- `hient` は natural から分離して集計する（`corpus` 列で識別）。

## 3. 結果

100,000 decisions 級の run（400 prompt × 256 token = 102,400）と、
1,000,000 decisions 級を途中まで（437,211）実行したものを合算する。

**合計 539,611 decisions**:

| pool | mismatch | rate | natural のみ（hient 除外 462,687） |
| ---: | ---: | ---: | ---: |
| 64 | **68** | **1.260e-4** | 47 / 462,687 = 1.016e-4 |
| 127 | **15** | **2.780e-5** | 7 / 462,687 = 1.513e-5 |

corpus 別:

| corpus | rows | N=64 | N=127 |
| --- | ---: | ---: | ---: |
| code | 77,113 | 2 | 0 |
| json | 77,113 | 7 | 0 |
| math | 76,924 | 4 | 0 |
| ja | 77,179 | 16 | 2 |
| prose | 77,435 | 8 | 3 |
| train（mixed） | 76,923 | 10 | 2 |
| **hient（stress）** | 76,924 | **21** | **8** |

## 4. N=127 の失敗の正体

N=127 の 15 件は**全件 `oracle_rank = -1`**、すなわち full PSQ8 Top1 が
**coarse Top-127 の外側**にある prune miss である（rerank の誤りではない）。

内訳は hient 8 / prose 3 / train 2 / ja 1。うち 4 件は生成 step 0（prompt 直後）で発生している。

したがって N をさらに増やしても解消は保証されない（現状 `kDflash2Int2MaxPool = 128` で
127 が上限）。これらを消すには **coarse ranking 自体の改善**（learned PSQ2 等）が必要である。

## 5. static qualification との比較（本 Gate の主発見）

同じ corpus を teacher-forced で流した static の mismatch 率（`lm_head_margin_fallback_poc.md`）
と、on-policy のそれを比較する（N=64）。

| corpus | static N=64 | on-policy N=64 | 比 |
| --- | ---: | ---: | ---: |
| hient | 6.73e-2 | **2.73e-4** | **約 247 倍低い** |
| mixed204k / train | 5.80e-4 | 1.30e-4 | 約 4.5 倍低い |
| math | 6.67e-4 | 5.20e-5 | 約 13 倍低い |
| prose | 7.69e-4 | 1.03e-4 | 約 7.5 倍低い |
| ja | 1.67e-4 | 2.07e-4 | ほぼ同等 |
| code | 0 | 2.60e-5 | — |
| json | 0 | 9.08e-5 | — |

**static corpus には、greedy production では実際には到達しにくい hidden が大量に含まれる。**
特に `hient`（temperature 1.3 で別 token を選び続けて作った off-policy trajectory）は
意図的な stress set であり、greedy 専用 fast path の qualification には使えない。
static は adversarial / stress qualification として保存し、production 判断は本 Gate で行う。

## 6. 判定

- **N=64 は production 候補として有力**: on-policy 539,611 decisions で 68 件（1.26e-4）、
  natural のみでは 462,687 で 47 件（1.02e-4）。
- **N=127 はより安全側**: 15 件（2.78e-5）、natural のみ 7 件（1.51e-5）。
  ただし残る失敗はすべて coarse Top-127 圏外の prune miss であり、
  N を上げるだけでは保証できない。
- 失敗はいずれも **prune miss**（rerank は候補集合内で exact）であり、
  設計上の想定どおりである。
- `hient` は依然として最悪（N=64 で 21/76,924 = 2.7e-4）だが、これは
  off-policy stress であり、greedy production の代表値ではない。

margin fallback は F0-F5 で NO-GO のため採用しない。残る N=127 の 15 件のような
on-policy failure を対象に、必要になった時点で fallback detector を再検討する。

## 7. 再現

```
PHASESHIFT_TARGET_LM_SHADOW_CORPORA=<comma separated pskldtok>
PHASESHIFT_TARGET_LM_SHADOW_PROMPTS=4000
PHASESHIFT_TARGET_LM_SHADOW_TOKENS=256
PHASESHIFT_TARGET_LM_SHADOW_PROMPT=128
PHASESHIFT_TARGET_LM_SHADOW_PROMPT_LONG=1024
PHASESHIFT_TARGET_LM_SHADOW_BATCH=16
PHASESHIFT_TARGET_LM_SHADOW_CSV=/tmp/opencode/shadow_1m.csv
HIP_VISIBLE_DEVICES=<gpu> ./build/tests/test_target_lm_greedy_shadow
```

## 8. 未実施

- 1M の完走（本記録は 437,211 decisions で停止。539,611 は 100k run との合算）。
- on-policy で通過した N での closed-loop（baseline full PSQ8 と proxy の独立生成比較）。
- coarse Top-127 圏外の prune miss に対する fallback detector / coarse ranking 改善。
