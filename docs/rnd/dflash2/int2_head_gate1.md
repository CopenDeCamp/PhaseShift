# DFlash2 INT2 proposal head 再qualification（Gate 1）

> Status: R&D record（Gate の経緯・採否判断。現在の contract は
> `docs/developer/dflash2.md`、現在の性能値は `docs/perf/current.md` を正本とする）

## 0. 目的

新しいコードを書く前に、CURRENT HEAD に存在する
**INT2 coarse + PSQ8 exact rerank** の proposal head が、現行 DFlash2 baseline に対して
正しさ / acceptance / latency / E2E tok/s で有効かを再測定する。
Gate 2（Radix Select Top-N）への入力はこの結果である。

## 1. 計測条件

| 項目 | 値 |
| --- | --- |
| revision | `1613544f`（Gate 1）／同一 revision + 本 Gate の実装（Gate 2） |
| GPU | AMD Radeon AI PRO R9700 (gfx1201) |
| device | Gate 1 は device 1、Gate 2 は device 3（計測中は同一 GPU に固定） |
| model | `models/Qwen3.8-27B-PSQ` + `models/Qwen3.8-27B-DFlash2-PSQ` |
| corpus | `build/dflash2-gate51/real/real_prose/prompt_ids.txt` |
| build | `build-gfx1201` / Release / HIP graph OFF |

方法論は `docs/perf/methodology.md` に従い、条件比較は同一 GPU 上で交互
（baseline → candidate → ...、開始順は rep ごとに回す）に測定した。

---

## 2. Gate 1: INT2 proposal head 再qualification

### 2.1 受入テスト

- `ctest -L required` = **124/124 PASS**（本 Gate 開始時）
- `ctest -R 'dflash2_int2|dflash2_coarse_topn|dflash2_psq8_rerank'` = **4/4 PASS**

### 2.2 baseline の実測

`PHASESHIFT_DFLASH2_INT2_HEAD=0`（drafter は full PSQ8 proposal head）。
docs/perf/current.md の 65.10 tok/s は古い revision の値であり、比較基準には使わず
**CURRENT HEAD の実測を baseline とした**。

interleaved A/B（8 rep、中央値）:

| config | tok/s | vs baseline | rounds | accepted |
| --- | ---: | ---: | ---: | ---: |
| full PSQ8 | **64.90** | — | 84 | 171 |
| INT2 pool=32 | 65.90 | +1.5% | 85 | 170 |
| INT2 pool=64 | **66.45** | **+2.4%** | 84 | 171 |
| INT2 pool=80 | 66.37 | +2.3% | 84 | 171 |

全 rep・全設定で `GENERATED_IDS` の sha1 先頭 12 桁 = `47aebe55d048` が baseline と一致。
crash / NaN / invalid token なし。

参考（初回 sweep、3 rep 中央値、同条件）:

| pool | 16 | 24 | 32 | 48 | 64 | 80 | 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tok/s | 65.41 | 66.15 | 66.03 | 65.85 | 66.49 | 66.42 | 66.31 |

`PHASESHIFT_DFLASH2_INT2_HEAD=1` は常に IDS 一致だった（pool によらず
proposal の差異は Verify で吸収される）。

### 2.3 INT2 pipeline timing（`PHASESHIFT_DFLASH2_INT2_TIMING=1`、rows=7、単位 ms/round）

| pool | quant | coarse | **topn** | rerank | top16 | pipe |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.009 | 0.888 | **0.080** | 0.071 | 0.047 | 1.127 |
| 24 | 0.010 | 0.889 | **0.101** | 0.067 | 0.047 | 1.146 |
| 32 | 0.009 | 0.889 | **0.123** | 0.072 | 0.047 | 1.172 |
| 48 | 0.010 | 0.889 | **0.165** | 0.073 | 0.047 | 1.215 |
| 64 | 0.009 | 0.888 | **0.209** | 0.073 | 0.047 | 1.258 |
| 80 | 0.010 | 0.890 | **0.253** | 0.075 | 0.048 | 1.308 |
| 128 | 0.010 | 0.890 | **0.385** | 0.081 | 0.049 | 1.447 |

**coarse head（INT2 全語彙 GEMV）が 0.89 ms で支配的**であり、topn は pool に比例して
0.080 → 0.385 ms に増える。これが Gate 2 の baseline である。

### 2.4 diagnostic（`PHASESHIFT_DFLASH2_INT2_HEAD=2` + `INT2_DIAG=1`、83-86 round / 577 row）

`contain` は「full PSQ8 top-16 が全て coarse pool に入る row の割合」、
`selcov` は「full path の選択 token が coarse pool に入る割合」、
`tokenmatch` は「INT2 proposal token が full proposal token と一致する割合」。

| pool | top1 | recall16 | contain | selcov | tokenmatch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.9393 | 0.6232 | 0.0017 | 0.8475 | 0.7366 |
| 24 | 0.9740 | 0.7458 | 0.0607 | 0.8925 | 0.7747 |
| 32 | 0.9861 | 0.8195 | 0.1334 | 0.9272 | 0.8475 |
| 64 | 0.9983 | 0.9413 | 0.4679 | 0.9705 | 0.9203 |

proposal token は pool=64 でも 8% が full proposal と異なるが、最終 IDS は全設定で
baseline 一致（Verify が吸収）。

### 2.5 副次発見: target Verify の proxy は既定で有効

`PHASESHIFT_TARGET_LM_HEAD_PROXY` は未指定時に mode=1 を返し、
`try_launch_lm_head_proxy_fusion` は Verify role で発火する。つまり
**現行既定の DFlash2 計測は「drafter full PSQ8 + target Verify は INT2 candidate proxy」**
という構成である。実測（同 GPU・2 rep）:

| 設定 | tok/s |
| --- | ---: |
| 既定（target proxy 有効） | 65.52 / 65.46 |
| `PHASESHIFT_TARGET_LM_HEAD_PROXY=0` | 63.86 / 63.80 |

どちらも IDS は `47aebe55d048` で一致。**Gate 1 の baseline は既定値を production
incumbent として採用した**（methodology §2 に従う）。

### 2.6 Gate 1 判定

```
BEST_POOL=64
FULL_PSQ8_TPS=64.90
INT2_TPS=66.45        (+2.4%)
INT2_TOPN_US=209      (pool=64, rows=7, per round)
```

**GO**。required PASS、IDS baseline 一致、E2E 改善が測定揺らぎを明確に超え、
acceptance 悪化（171→170）は利益を相殺していない。
候補 pool は「head だけ最速」ではなく E2E 最大の **pool=64** を採用。
