# FFN down_proj Dynamic K256 Tile-local Oracle Gate

更新日: 2026-09-24
branch: `poc/ffn-k256-tile-oracle`
harness: `tools/rnd/ffn_k_tile_oracle/`

## 1. Goal

Qwen3.8-27B / Qwen3.5 Dense の FFN `down_proj` decode1 を対象に、
**token × output tile × K256 group** ごとに独立な dynamic mask を持たせたとき、
寄与の小さい K256 group の計算を丸ごと省略できるかを **exact Oracle** で確定する。

predictor は実装しない。activation-only / activation×weight-norm / INT2 estimator /
sparse production kernel / runtime selector はこの Gate の対象外。

この Gate の目的は高速化コードではなく、

> N64 × K256 という GPU 実装可能な粗さでも、token 依存の計算 mass が本当に消せるのか

を確定することである。

## 2. Baseline / revision

| 項目 | 値 |
| --- | --- |
| base revision | `3b8d6f6d55e737dca6eefa1728ce705d3d87d0b5` (main) |
| GPU | AMD Radeon AI PRO R9700 (gfx1201) ×3 |
| ROCm / HIP | 7.15.26333 |
| torch | 2.13.0+rocm10.0.0 |
| transformers | 5.14.1 |
| model | `models/Qwen3.8-27B` (BF16, safetensors streaming load) |
| corpus | `artifacts/ffn_prune/corpus.psktok` (2048 token) |
| corpus SHA256 | `72392ef85f1dee247a2ee6259a6dc8460eebebd42fb7d663e89d9060b8ea593b` |
| N / output | 5120 |
| K / intermediate | 17408 |
| K group | 256 → 68 groups |

corpus / model / loader は既存 `docs/rnd/ffn/ffn_int2_prune_gate.md` と同一。
baseline は次節のとおり旧記録 `PPL = 7.063564`（positions 2047）と一致する。

### baseline PPL

teacher forcing、2048 token 単一 forward、positions 2047。

| run | PPL | mean NLL |
| --- | ---: | ---: |
| N64/N16 sweep の baseline | 7.056874 | 1.954002 |
| global 5120 sweep の baseline | 7.063697 | 1.954969 |
| 旧 ffn_int2_prune_gate 記録 | 7.063564 | — |

同一条件の別 process 実行で baseline が 7.0569〜7.0637（spread 約 0.10%）で変動した。
kernel selection に起因する再現ノイズであり、後述 parity の分解能限界でもある。

## 3. 既存 global G256 結果との違い

既存 `ffn_int2_prune_gate.md` は down_proj の K/channel を **全 5120 output 共通 mask**
で group 化していた（G256: +2% 以内は実質 0%、+5% で 12.5%）。本 Gate はこれを
**output tile ごとに独立な mask** へ拡張する。全 output 共通 mask へ退化させないことが
最重要条件である。

sanity のため `tile_width = 5120`（全 output 共通 mask 相当）を同一 harness で測定した。
このとき `score = sum_d c[g,d]^2 = ||W_g h_g||^2` となり、既存 G256 の true group energy
`T` と厳密に一致する。

| 目標 | 既存 global G256 | 本 sanity (tile_width=5120) |
| --- | ---: | ---: |
| +2% 以内の最大 prune | 0% | 8.75%（補間） |
| +5% 以内の最大 prune | 12.5% | 14.45%（補間） |

既存記録の測定 grid は 0% / 12.5% 刻みであり、12.5% で +3.76%（本 sanity 補間）、
18.75% で +5% 超となるため「+5% 以内 = 12.5%」と整合する。corpus / score 定義 / loader /
model / dynamic mask 実装は既存結果を再現している。

## 4. Method

1. safetensors shard を streaming し 3 GPU へ直接ロード（CPU RAM 全載せはしない）。
2. Hugging Face `Qwen3_5MLP.forward` を差し替える。Gate / Up / SwiGLU は exact のまま。
3. `h = SiLU(gate(x)) * up(x)` を計算し、`down_proj(h)` を
   `OracleKTileDown(h, W_down)` に置換する。
4. 各 token chunk について K256 group ごとの exact partial を計算し、score を出し、
   tile ごとの mask を作り、selected partial を加算して chunk output を確定する。
5. PPL を teacher forcing で測り、baseline からの ΔPPL% を出す。

partial workspace は token chunk 単位で破棄する。`2048 × ntile × 68 × tile_width` を
全 sequence で保持しない。CPU への自動 fallback は行わない。

## 5. exact partial 定義

```text
c[tile, g]      = W_down[tile, g*256:(g+1)*256] @ h[g*256:(g+1)*256]     (float32)
score[tile, g]  = sum_d c[tile, g, d]^2                                  (squared L2, primary)
keep[tile]      = score の上位 (68 - skip_count) group
y[tile]         = sum_{g in keep[tile]} c[tile, g]
```

* mask は token ごと・output tile ごとに独立。
* `skip_count = round(target_skip_fraction * 68)`。JSON には `target_skip_fraction` /
  `skip_group_count` / `actual_skip_fraction` を併記する。
* partial accumulation は float32。source weight / activation は BF16。
* `tile_width = 64` が primary（80 tiles）、`16` が diagnostic（320 tiles）、
  `5120` が global sanity。

## 6. N64 skip vs PPL

tile_width = 64（80 output tiles）。baseline PPL = 7.056874。

| target skip | groups skipped | actual skip | PPL | dPPL % |
| ---: | ---: | ---: | ---: | ---: |
| 0.0000 | 0/68 | 0.00000 | 7.061929 | +0.072 |
| 0.0500 | 3/68 | 0.04412 | 7.116386 | +0.843 |
| 0.0750 | 5/68 | 0.07353 | 7.169427 | +1.595 |
| 0.1000 | 7/68 | 0.10294 | 7.210855 | +2.182 |
| 0.1250 | 8/68 | 0.11765 | 7.244539 | +2.659 |
| 0.1500 | 10/68 | 0.14706 | 7.302566 | +3.482 |
| 0.2000 | 14/68 | 0.20588 | 7.488119 | +6.111 |
| 0.3000 | 20/68 | 0.29412 | 7.914752 | +12.157 |
| 0.4000 | 27/68 | 0.39706 | 8.738650 | +23.832 |
| 0.5000 | 34/68 | 0.50000 | 10.278225 | +45.648 |
| 0.6000 | 41/68 | 0.60294 | 14.250210 | +101.934 |

補間による最大 skip:

* ΔPPL ≤ +2%: **9.38%**
* ΔPPL ≤ +5%: **18.10%**

## 7. N16 skip vs PPL

tile_width = 16（320 output tiles）。baseline PPL = 7.056874。

| target skip | groups skipped | actual skip | PPL | dPPL % |
| ---: | ---: | ---: | ---: | ---: |
| 0.0000 | 0/68 | 0.00000 | 7.061929 | +0.072 |
| 0.0500 | 3/68 | 0.04412 | 7.060099 | +0.046 |
| 0.0750 | 5/68 | 0.07353 | 7.093811 | +0.523 |
| 0.1000 | 7/68 | 0.10294 | 7.126404 | +0.985 |
| 0.1250 | 8/68 | 0.11765 | 7.139749 | +1.174 |
| 0.1500 | 10/68 | 0.14706 | 7.191465 | +1.907 |
| 0.1750 | 12/68 | 0.17647 | 7.291535 | +3.325 |
| 0.2000 | 14/68 | 0.20588 | 7.298056 | +3.418 |
| 0.2500 | 17/68 | 0.25000 | 7.407568 | +4.970 |
| 0.3000 | 20/68 | 0.29412 | 7.612078 | +7.868 |
| 0.4000 | 27/68 | 0.39706 | 8.174157 | +15.833 |
| 0.5000 | 34/68 | 0.50000 | 9.346450 | +32.445 |
| 0.6000 | 41/68 | 0.60294 | 11.854514 | +67.985 |

補間による最大 skip:

* ΔPPL ≤ +2%: **14.90%**
* ΔPPL ≤ +5%: **25.05%**

global sanity（tile_width=5120、baseline PPL = 7.063697）:

| actual skip | dPPL % |
| ---: | ---: |
| 0.00000 | +0.023 |
| 0.10294 | +2.350 |
| 0.20588 | +8.914 |
| 0.29412 | +16.419 |
| 0.39706 | +31.150 |

補間: ΔPPL ≤ +2% は **8.75%**、≤ +5% は **14.45%**。

tile-local 化の効果（+2% 以内の最大 skip）:

| 粒度 | headroom |
| --- | ---: |
| global 5120 | 8.75% |
| N64 | 9.38% |
| N16 | 14.90% |

N64 は global に対し +0.6pp しか改善しない。N16 では +6.2pp 改善する。

## 8. contribution distribution

`score` からの group-energy proxy。`sum_g ||c_g||^2` は cross-term を含まないため
真の output energy `||sum_g c_g||^2` ではない。以下はすべて **group-energy proxy** の
統計である。64 layer の layer 平均。

| stat | N64 | N16 |
| --- | ---: | ---: |
| score_mean | 0.20984 | 0.05246 |
| score_p50 | 0.11748 | 0.02770 |
| score_p90 | 0.37464 | 0.09325 |
| score_p99 | 1.52301 | 0.39611 |
| top1_frac | 0.10882 | 0.11229 |
| top4_frac | 0.22653 | 0.23707 |
| top8_frac | 0.32298 | 0.34053 |
| bottom10_frac | 0.04385 | 0.03327 |
| bottom20_frac | 0.09832 | 0.07967 |
| bottom30_frac | 0.15102 | 0.12701 |
| bottom40_frac | 0.21909 | 0.19058 |
| retained proxy @ skip20% | 0.90168 | 0.92033 |
| retained proxy @ skip30% | 0.84898 | 0.87299 |
| retained proxy @ skip40% | 0.78091 | 0.80942 |

重要な帰結:

* 下位 20% group は group-energy proxy の約 9.8%（N64）しか持たないのに、
  これを skip すると PPL は +6.1%（N64）壊れる。group 間 cross-term（cancellation）が
  支配的で、`||c_g||^2` の大きさが output 重要度に対応しない。
* top1 group でも proxy の ~11% しかなく、mass は広く分散している。

## 9. per-layer profile

retained group-energy proxy（layer 平均ではなく各 layer）:

### N64

| layer | retained @ skip20% | @ skip30% | @ skip40% |
| ---: | ---: | ---: | ---: |
| best L0 | 0.95728 | 0.93207 | 0.89785 |
| median L21 | 0.89418 | 0.83873 | 0.76808 |
| worst L32 | 0.87700 | 0.81294 | 0.73153 |

worst 5（skip30%）: L32 0.81294, L33 0.81400, L28 0.81420, L30 0.81500, L29 0.81560。
best 5: L0 0.93207, L2 0.91570, L63 0.91340, L61 0.90310, L54 0.90000。

### N16

| layer | retained @ skip20% | @ skip30% | @ skip40% |
| ---: | ---: | ---: | ---: |
| best L0 | 0.96280 | 0.93935 | 0.90678 |
| median L21 | 0.91545 | 0.86598 | 0.79955 |
| worst L32 | 0.90284 | 0.84616 | 0.77094 |

worst layer は両粒度で L32、L28〜L33 が cluster をなす。best は L0 / L2 / L63 / L61。
平均では成立しても特定 layer だけ壊れる、という崩壊は観測されない（層間差は
retained proxy で最大 ~0.13 = 13pp 程度）。

## 10. decomposition parity

skip = 0 の decomposed path（全 group KEEP、partial 和 = full down_proj）と
original `down_proj` を比較する。

| run | basal PPL | decomposed skip0 PPL | ΔPPL % |
| --- | ---: | ---: | ---: |
| N64/N16 sweep | 7.056874 | 7.061929 | +0.072 |
| global sweep | 7.063697 | 7.065320 | +0.023 |

すべて |ΔPPL| ≤ 0.1% を満たす。ただし再現ノイズ（baseline spread ~0.10%）と同程度で、
0.1% は分解能限界に近い。

小 smoke test（corpus 先頭 128 token、skip=0）での original vs decomposed logits 差:

| 量 | 値 |
| --- | ---: |
| max abs diff | 0.9375 |
| mean abs diff | 0.02281 |
| logits abs mean | 2.5764 |

差は logits の BF16 丸め数 ULP 程度。tile_width 64 / 16 / 5120 で同一（skip=0 では
mask が全 KEEP のため当然）。partial accumulation の誤差は問題にならない範囲。

## 11. Gate 判定

**HOLD**

根拠（実測値）:

* N64（primary）で ΔPPL +2% 以内にできる skip は **9.38%**、+5% 以内は **18.10%**。
  GO 条件（skip ≥ 25% で +2%、skip ≥ 40% で +5%）は満たさない。
* N16 でも **14.90% / 25.05%** であり、CONDITIONAL-GO 条件（25% / 40%）は満たさない。
* N64 は global G256（+2% headroom 8.75%）に対し 9.38% と、ほぼ改善しない（+0.6pp）。
  GPU 実装可能な N64 粒度ではタイル局所化の利得が小さい。
* 一方で sparsity は存在し、token/tile 動的 mask で N16 は global 比 +6.2pp
  （8.75% → 14.90%）改善する。N64 でも +2% 付近までわずかに伸びる。
  したがって「sparsity が全く無い／tile-local が global と同一」という NO-GO ではない。
* headroom は小さく、selector cost 次第で成立しうる水準にとどまる（N64 で約 9%、
  N16 で約 15%）。NO-GO 帯の下限（N64 headroom 10% 未満）に近い。

decomposition は安定（parity ≤ 0.1%）。dynamic tile-local 化は global とほぼ同一では
ないが、N64 での利得は小さい。

## 12. 次工程

GO ではないため、この Gate では predictor study を開始しない。

* N64 × K256 の headroom は +2% で約 9% であり、selector（activation-only 等）の
  cost を正当化できる見込みが薄い。
* N16 の約 15% は N16 mask を毎 token / 毎 tile 決める cost を前提とし、
  selection 費用対効果の別途見積りが必要。
* したがって現時点では `activation-only predictor` Gate、INT2 estimator、
  sparse production kernel へは進まない。

## 13. 再現コマンド

```bash
W=.worktrees/ffn-k256-tile-oracle
cd $W
python3 -m compileall tools/rnd/ffn_k_tile_oracle
python3 tools/rnd/ffn_k_tile_oracle/test_oracle.py

CUDA_VISIBLE_DEVICES=0,1,2 python3 tools/rnd/ffn_k_tile_oracle/run_oracle.py \
    --model-dir models/Qwen3.8-27B \
    --corpus artifacts/ffn_prune/corpus.psktok \
    --k-group 256 --tile-width 64 --score l2 --max-tokens 2048 \
    --skip-fracs 0,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6 \
    --out artifacts/ffn_k_tile_oracle/oracle_n64.json

CUDA_VISIBLE_DEVICES=0,1,2 python3 tools/rnd/ffn_k_tile_oracle/run_oracle.py \
    --model-dir models/Qwen3.8-27B \
    --corpus artifacts/ffn_prune/corpus.psktok \
    --k-group 256 --tile-width 16 --score l2 --max-tokens 2048 \
    --skip-fracs 0,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.4,0.5,0.6 \
    --out artifacts/ffn_k_tile_oracle/oracle_n16.json

CUDA_VISIBLE_DEVICES=0,1,2 python3 tools/rnd/ffn_k_tile_oracle/run_oracle.py \
    --model-dir models/Qwen3.8-27B \
    --corpus artifacts/ffn_prune/corpus.psktok \
    --k-group 256 --tile-width 5120 --score l2 --max-tokens 2048 \
    --skip-fracs 0,0.1,0.2,0.3,0.4,0.5,0.6 \
    --out artifacts/ffn_k_tile_oracle/oracle_global.json
```

raw JSON / summary は `artifacts/ffn_k_tile_oracle/`（git 管理外）。
