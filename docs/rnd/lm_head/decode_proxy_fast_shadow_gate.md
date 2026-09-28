# Target Decode LmHeadCandidateProxy の fast / shadow 展開（Gate 3）

> Status: R&D record（Gate の経緯・採否判断。現在の contract は
> `docs/developer/qwen35.md` §7.1、現在の性能値は `docs/perf/current.md` を正本とする）

## 0. 目的

Verify 専用だった INT2 coarse → TopN → PSQ8 exact candidate rerank → argmax を、
通常の `ExecutionRole::Decode` greedy token selection へ展開する。
INT2 は candidate generation のみに使い、final decision は常に original PSQ8 exact rerank。

前提となる Gate 1 / Gate 2 は
[../dflash2/int2_head_gate1.md](../dflash2/int2_head_gate1.md) /
[../dflash2/radix_topn_gate2.md](../dflash2/radix_topn_gate2.md) を参照。

## 1. 実装した内容

### 1.1 適用可否

`lm_head_proxy_path(role, stochastic_outputs, constrained, mode)`
（`include/phaseshift/models/qwen35/runtime/lm_head_proxy.h`）へ分岐を 1 箇所に集約した。

- role は `Verify` か `Decode` のみ。`Prefill` は常に off。
- Decode は `actual_stochastic_outputs == 0`（greedy）に限定。
- constraint（`constraint_masks` / `constraint_mask_words`）は常に off。
- weight 前提（PSQ8 / preshuffled / `weight_scale_group == 32` / `k_padded % 32 == 0`）と
  pattern 前提（`ACTIVATION_QUANTIZE_W4A8` → `LINEAR_PSQ8` → `SAMPLING` の 3 連続）は据え置き。

### 1.2 mode の意味（`PHASESHIFT_TARGET_LM_HEAD_PROXY`）

| mode | 既定（未指定） | Verify | Decode greedy |
| --- | --- | --- | --- |
| 未指定 | — | Fast | **off** |
| 0 | — | off | off |
| 1 | — | Fast | Fast |
| 2 | — | off | Shadow |

- **既定値は既存挙動の維持**である（Verify のみ Fast、Decode は off）。Gate の途中で
  default を ON に変えていない。
- mode=2 の Decode は production output を一切 proxy に依存させない:
  `select_shadow()` が proxy 自有 buffer へ書き、3 dispatch は通常どおり実行され、
  `execute_program_range` の末尾で `compare_shadow()` が device counter に加算する。
  Host への退避と `TARGET_LM_HEAD_PROXY_SHADOW` 行の出力は shutdown のみ。

### 1.3 logits 外部利用との関係

Decode greedy は `sample requires compute_logits` のため常に full logits を external output
として持つ。Fast は full vocab logits を計算しないため、
`phaseshift-compute --dump-logits` とは併用できない。
これは model load 前に exit 2 で拒否する（DFlash2 + `--dump-logits` と同じ fail-closed）。
`phaseshift-bench tg --mode greedy` は logits を読まないため影響しない。

### 1.4 計測 instrument

- `PHASESHIFT_TARGET_LM_HEAD_PROXY_TIMING=1` → `TARGET_LM_HEAD_PROXY_TIMING` 行に
  quant / coarse / topn / rerank / argmax / total の running average（ms、round 単位）。
- `PHASESHIFT_OP_DUMP=1` + `PHASESHIFT_OP_SYNC=1` → dispatch 単位の実行時間。
  Fast 時は 3 dispatch が 1 エントリになるため full path との比較に使える。

### 1.5 test

- `tests/unit/test_lm_head_proxy_path.cpp`（cpu;required、13 check）
  - Decode greedy mode1 → Fast、mode2 → Shadow、mode0 → off
  - Decode stochastic → off、constraint → off
  - Verify mode1 → Fast（既存挙動）、mode0/2 → off
  - Prefill → off

## 2. 測定条件

| 項目 | 値 |
| --- | --- |
| GPU | AMD Radeon AI PRO R9700 (gfx1201)、device 3 に固定 |
| model | `models/Qwen3.8-27B-PSQ` |
| tg | `--mode greedy --context 2048 --tokens 128 --prefill-chunk 2048 --compute-logits 1 --page-tokens 16 --arena-gib 24 --warmup 2` |
| decode | `phaseshift-compute`、`--max-new-tokens N --max-seq-len 4096 --temperature 0` |

## 3. FAST performance（`phaseshift-bench tg`、3 rep 中央値）

| config | tok/s | vs baseline |
| --- | ---: | ---: |
| baseline（mode=0） | **27.47** | — |
| Fast pool=16 | 28.41 | +3.4% |
| Fast pool=24 | 28.32 | +3.1% |
| Fast pool=32 | 28.35 | +3.2% |
| Fast pool=48 | 28.38 | +3.3% |
| Fast pool=64 | 28.31 | +3.1% |

- **全 18 run で `GREEDY_TOKEN_SUM=2446188`**（baseline 一致、`docs/perf/current.md` の
  記載値とも一致）。
- pool による差は ±0.5% 以内。proxy cost の大部分は INT2 coarse head であり、
  Top-N の pool は E2E にほぼ効かない（Gate 2 の結論と整合）。


## 4. Shadow qualification（§3.8）

`PHASESHIFT_TARGET_LM_HEAD_PROXY=2` で production 出力を full path に固定し、同じ
final_norm から proxy token を計算して比較する（trajectory は汚染しない）。
corpus は prose / code / math の 3 種、各 4,000 token、greedy。

### 4.1 pool sweep（各 pool 11,997 decisions）

| pool | decisions | mismatches | mismatch rate | IDS vs baseline |
| ---: | ---: | ---: | ---: | --- |
| 16 | 11,997 | **1** | 8.3e-5 | 一致（mode=2 は出力に proxy を使わない） |
| 24 | 11,997 | 0 | 0 | 一致 |
| 32 | 11,997 | 0 | 0 | 一致 |
| 48 | 11,997 | 0 | 0 | 一致 |
| 64 | 11,997 | 0 | 0 | 一致 |

mismatch は pool=16 の prose 1 件のみ。mode=2 の `GENERATED_IDS` は全 15 run で
base（mode=0）と完全一致しており、**shadow が full path の出力を変更しないこと**を
同時に確認している。

### 4.2 deep（候補 pool=32、追加 12 run）

pool=32 を候補として decision 数を深めた（3 corpus × 4,000 token × 4 rep）。

| pool | decisions | mismatches | mismatch rate |
| ---: | ---: | ---: | ---: |
| 32 | **59,985** | **0** | 0 |

**合計 107,973 decisions / 1 mismatch**（唯一の mismatch は pool=16）。
pool=32 は 95% 信頼上限で約 5.0e-5。

### 4.3 closed-loop（mode=1、各 corpus 4,000 token）

| config | prose | code | math |
| --- | --- | --- | --- |
| base（mode=0） | `11d9886c10d1` | `6c70ef18ada8` | `56f7514ec9b9` |
| Fast pool=16 | **`0c572030bad9`（diverge）** | 一致 | 一致 |
| Fast pool=32 | 一致 | 一致 | 一致 |

**pool=16 は prose で closed-loop が divergence する**ため採用しない
（§4.1 の 1 件の mismatch と対応する）。
**pool=32 は 3 corpus × 4,000 token で生成列が完全一致**した。

短い `phaseshift-bench tg`（128 token）では pool によらず全 18 run で
`GREEDY_TOKEN_SUM=2446188` が一致しており、pool=16 の risk は長生成で顕在化する。

## 5. lm_head 部分の計測（rows=1、§3.11）

### 5.1 baseline（mode=0、`PHASESHIFT_OP_DUMP=1 PHASESHIFT_OP_SYNC=1`）

decode step のみを抽出（`ACTIVATION_QUANTIZE_W4A8` → `LINEAR_PSQ8` → `SAMPLING` の 3 連、7 step 平均）:

| 段 | ms |
| --- | ---: |
| activation quantize | 0.0289 |
| full PSQ8 lm_head | **2.1833** |
| greedy argmax（sampling） | 0.0394 |
| **合計** | **2.2516** |

### 5.2 proxy（mode=1、`PHASESHIFT_TARGET_LM_HEAD_PROXY_TIMING=1`、7 round 平均）

| pool | quant | coarse | topn | rerank | argmax | **total** | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.011 | 0.899 | 0.039 | 0.072 | 0.010 | **1.031** | 2.18x |
| 32 | 0.012 | 0.903 | 0.057 | 0.072 | 0.010 | **1.054** | 2.14x |
| 64 | 0.011 | 0.906 | 0.108 | 0.073 | 0.011 | **1.108** | 2.03x |

- **INT2 coarse head（0.90 ms）が支配的**で、proxy の 86% を占める。
- topn は pool に比例（0.039 → 0.108 ms）だが E2E にはほぼ効かない（§3）。
- pool=32 で 1 step あたり **1.198 ms を節約**。tg の 27.47 tok/s（36.4 ms/token）に対する
  理論改善は 3.3% で、実測の +3.2% と整合する。

## 6. fallback 確認（§3.12）

`PHASESHIFT_OP_DUMP=1 PHASESHIFT_OP_SYNC=1` で dispatch 単位の実行を確認した
（`--max-new-tokens 8`、plain decode）。

| case | 条件 | 結果 |
| --- | --- | --- |
| A | `PHASESHIFT_TARGET_LM_HEAD_PROXY=0` | `LINEAR_PSQ8` count=520 / `SAMPLING` count=8 → full path |
| A' | env 未指定（既存挙動） | 同上（Decode は既定で off） |
| B | mode=1 + `--temperature 0.8` | `LINEAR_PSQ8` count=520 / `SAMPLING` count=8 → full path |
| C | constraint | `lm_head_proxy_path` で off（`test_lm_head_proxy_path` が担保） |
| D | weight encoding 非対応 | fusion 内の weight 条件で off（既存チェック据え置き） |
| E | proxy 未初期化（mode=0） | `ctx.lm_head_proxy == nullptr` → not_applicable |
| F | role 対象外（Prefill / stochastic Decode） | `lm_head_proxy_path` で off（`test_lm_head_proxy_path` が担保） |

mode=1 のとき `LINEAR_PSQ8` は count=513（= 520 − 7）、`SAMPLING` は count=1（= 8 − 7）となり、
**decode の 7 step だけが fusion で消費され**、prefill は従来どおりであることを確認した。

`--dump-logits` は Fast と併用できないため model load 前に exit 2 で拒否する（§1.3）。

## 7. 判定（§3.14）

| 条件 | 結果 |
| --- | --- |
| required tests 全 PASS | **126/126 PASS** |
| baseline `GREEDY_TOKEN_SUM` 一致 | 全 18 run で `2446188` |
| closed-loop 生成一致 | pool=32 で 3 corpus × 4,000 token が完全一致 |
| shadow decision 数 | **107,973 decisions / 1 mismatch**（pool=32 は 59,985 / 0） |
| crash / NaN なし | 全 run rc=0 |
| full fallback 正常 | §6 の A/A'/B で full path を確認 |
| E2E tok/s 改善 | pool=32 で 27.47 → 28.35 tok/s（**+3.2%**） |

**GO（qualification 成立）**。ただし以下は変更しない。

- `PHASESHIFT_TARGET_LM_HEAD_PROXY` の**既定値は据え置き**（Verify のみ Fast、
 Decode は off）。mode=1 の既定 ON は別の commit で、containment をさらに積んでから。
- 推奨 pool は **32**（+3.2%、59,985 decisions で mismatch 0）。
 pool=16 は最速だが長生成で closed-loop divergence が観測されており不採用。
- pool=48 / 64 も 0 mismatch だが、E2E は pool によらず同程度で
 Top-N の pool は INT2 coarse head に対して影響が小さい。
