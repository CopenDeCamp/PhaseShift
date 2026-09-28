> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# FFN latency PoC: PSQ4 据え置き vs INT2 → PSQ8

更新日: 2026-09-24
branch: `poc/ffn-int2-prune`（harness `tools/rnd/ffn_prune/` はこの branch にのみ存在）

## 1. 目的

FFN の decode（M=1）について、次のどちらが速いかを実測で比較する。

```text
A. 現行 PSQ4 据え置き
     gate/up PSQ4 + down PSQ4/PSQ8 混在（現行モデル）

B. INT2 candidate → PSQ8 exact two-stage
     INT2 proxy (全 channel) + candidate PSQ8 gate/up + final PSQ8 down
```

あわせて、B が成立する場合に「モデル全体の bpw を保つ」ことが可能かを検討する。

## 2. 実測（`phaseshift-bench gemm`, rows=1, gfx1201）

`--shapes` で FFN の実 shape（gate/up 融合 `34816:5120`、down `5120:17408`）。

| dtype | shape | p50 us | code+scale MB | 実効 GB/s |
| --- | --- | ---: | ---: | ---: |
| psq4 | 34816:5120 (gate+up) | 172.7 | 100.3 | 581 |
| psq8 | 34816:5120 (gate+up) | 308.3 | 189.4 | 614 |
| psq4 | 5120:17408 (down) | 83.4 | 50.1 | 601 |
| psq8 | 5120:17408 (down) | 153.2 | 94.7 | 618 |

PSQ8 は同一 shape で **1.79〜1.84 倍**遅い（bytes 比 1.89、bandwidth 効率は PSQ8 が僅かに高い）。

INT2 の FFN GEMV kernel は現存しない。bytes は
`2·(89.13M·2/8 + 89.13M/32·2) = 55.7 MB`（gate+up 合計）で、
PSQ4 実測 bandwidth 580GB/s を仮定すると **約 96us（推定、未実測）**。

## 3. FFN 1 layer あたりの latency

現行 down は 48 layer PSQ4 + 16 layer PSQ8 混在なので
`down_current = (48·83.4 + 16·153.2)/64 = 100.8us`。

```text
dense_current = 172.7 + 100.8 = 273.6 us
```

| variant | latency | vs dense_current |
| --- | ---: | ---: |
| dense PSQ4 (all psq4) | 256.1us | +6.8% |
| **dense current（A）** | **273.6us** | 0% |
| dense PSQ8 (all psq8) | 461.5us | -40.7% |
| two-stage INT2→PSQ8 c=0.6875 f=0.5（INT2 無料） | 288.5us | **-5.2%** |
| two-stage INT2→PSQ8 c=0.6875 f=0.5（INT2 推定） | 384.6us | **-28.9%** |
| two-stage INT2→PSQ8 c=0.75 f=0.5（INT2 推定） | 403.9us | **-32.3%** |
| two-stage INT2→PSQ4 c=0.6875 f=0.5（INT2 推定） | 256.5us | +6.7% |
| two-stage INT2→PSQ4 c=0.625 f=0.5（INT2 推定） | 245.7us | +11.3% |
| down-only, exact gate/up, f=0.5 | 223.1us | **+22.6%** |
| down-only, exact gate/up, f=0.4375 | 216.8us | **+26.2%** |

（正 = dense_current より速い）

## 4. 判定

### 4.1 PSQ4 据え置き vs INT2 → PSQ8

**PSQ4 据え置きの方が明確に速い。**

- INT2→PSQ8 は、たとえ INT2 proxy を完全に無料にしても
  c=0.6875/f=0.5 で 288.5us > 273.6us（-5.2%）。
  PSQ8 は PSQ4 の 1.8 倍遅く、20% 前後の削減では逆転しない。
- INT2 proxy の実コスト（推定 96us）を入れれば -29% で、論外。
- これは §break-even の byte 数と一致する。PSQ8 exact の FFN は
  dense PSQ4 の 1.76 倍の traffic を持ち、two-stage の削減率
  (c=0.6875, f=0.5 で 17.9%) では埋まらない。

### 4.2 モデル bpw を保てるか

保てない。

| 内訳 | 現行 | FFN 全 PSQ8 |
| --- | ---: | ---: |
| FFN (gate/up/down) | 10.339 GB | 18.182 GB |
| 非 FFN | 8.584 GB | 8.584 GB |
| payload 合計 | 18.923 GB (5.5411 bpw) | 26.765 GB |
| 差分 | — | **+7.843 GB (+41%)** |

- 現行 bpw を保つには非 FFN を 8.584 → 0.741 GB にする必要があり、
  これは非 FFN 10.21e9 param で **0.58 bpw** に相当する。2 bit 未満であり不可能。
- したがって「FFN を PSQ8、他を PSQ4」で bpw を保つ設計は成立しない。
  PSQ8 FFN は必ずモデルを約 41% 大きくし、FFN の per-token traffic も増える。

### 4.3 副次的知見: down-only pruning

INT2 proxy を使わず、exact Gate/Up（全 channel）で選択して Down の K だけ
final keep まで削る方が、はるかに有望である。

- f=0.5 で **+22.6%**、f=0.4375 で **+26.2%**（Down が FFN の約 37% を占めるため）。
- proxy も PSQ8 も不要。
- ただし dynamic K の channel 単位 skip（G1）が必要。G32 block skip では
  Exact Oracle の粒度結果（+2% 以内で 12.5% 上限）から精度が成立しない。
  G1 の K-gather を GPU で bandwidth 効率良く実現できるかが次の論点。

## 5. 結論

```text
PSQ4 据え置き > INT2 → PSQ8
```

- INT2→PSQ8 は速さ（-5〜-29%）でも容量（+41%）でも劣る。採用しない。
- INT2→PSQ4 two-stage は +6.7〜+11.3% だが、INT2 GEMV kernel と
  candidate row-gather が必要で、利得は小さい。
- 本命は down-only pruning（+22〜+26%）。exact Gate/Up をそのまま使い、
  Down の K のみ削る。ただし G1 K-skip の GPU 実現性検証が前提。

## 6. Addendum: main `da50f61d`（down 系を全 PSQ4 へ統一）時点の補正

main の `da50f61d` で FfnDown ほかの PSQ8 分岐が廃止され、down 系が全 layer
PSQ4 になった。同じ実測 GEMV 値（psq4 down 83.4us）で現行構成を引き直すと、
dense_current は `172.7 + 83.4 = 256.1us` になる。PSQ4 exact 前提の
two-stage はほぼ中立以下になり、結論はさらに強くなる。

| variant | latency | vs 256.1us |
| --- | ---: | ---: |
| dense PSQ4（現行構成） | 256.1us | 0% |
| dense PSQ8 | 461.5us | -44.5% |
| two-stage INT2→PSQ8 c=0.6875 f=0.5（INT2 無料） | 288.5us | -11.2% |
| two-stage INT2→PSQ8 c=0.6875 f=0.5（INT2 推定） | 384.6us | -33.4% |
| two-stage INT2→PSQ4 c=0.6875 f=0.5（INT2 推定） | 256.5us | -0.2% |
| two-stage INT2→PSQ4 c=0.625 f=0.5（INT2 推定） | 245.7us | +4.2% |
| down-only f=0.5 | 214.4us | **+19.5%** |
| down-only f=0.4375 | 209.2us | **+22.4%** |

## 7. 再現コマンド

```text
R=/opt/zen/wk/PhaseNonShift
for dt in psq4 psq8; do
  $R/build/phaseshift-bench gemm --dtype $dt --rows 1 --shapes 34816:5120,5120:17408
done
python3 $R/.worktrees/ffn-prune/tools/rnd/ffn_prune/latency_poc.py
```

raw 実測: `artifacts/ffn_prune/ffn_gemv_latency.txt`
