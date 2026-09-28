# FFN K256 cheap vector sketch Gate

## 1. Goal

前 Gate で N64 × K256 sparse down の実装上の break-even が確認された。

| 項目 | 実測 |
| --- | --- |
| 39.71% skip | PSQ4 82.45 → 52.30 us、PSQ8 152.71 → 93.49 us |
| gross saving | 2.395 ms/token |
| mask / branch overhead | ほぼ 0 |
| +5% E2E を残す selector budget | 39.71%: 約 9.95 us/layer、41.18%: 約 11.17 us/layer |

したがって今回の問いは次のとおりである。

> exact `c_g = W_tile,g x_g ∈ R64` を計算せず、10 us/layer 以内で
> cancellation-aware mask を生成できるか。

本 Gate は 2 段階に分離した。

- **Gate A（情報 Gate）**: exact `c_g` を一度計算した後、`z_g = R c_g` だけを selector に見せる。
  `c_g` を参照せず、d=2/4/8/16 への圧縮で cancellation 情報が残るかを確認する。
  selector latency は測らない。
- **Gate B（production feasibility）**: Gate A 通過後のみ、`B = R W` を offline 生成し、
  INT2/INT4 量子化・latency・E2E を評価する。

結論を先に記す。**Gate A は NO-GO**。d<=8 の linear sketch は cancellation 情報をほぼ保持せず、
magnitude のみの `SMALL_L2` と同等以下であった。仕様 §14「Gate A 通過後のみ」に従い
Gate B は実行していない。

## 2. Previous break-even result

baseline revision は `401eb6ec`（branch `poc/ffn-k256-break-even`）。
前 Gate の数値は `docs/perf/ffn_k256_break_even.md` を正本とする。

前 Gate の quality 基準（`docs/perf/ffn_k_tile_cancel_oracle.md`、2048-token、positions 2047）:

| selector | 35.29% | 39.71% | 41.18% |
| --- | ---: | ---: | ---: |
| `PAIR_GREEDY_SWAP`（exact, N64） | — | +4.714% | +4.545% |
| `SMALL_L2`（exact） | — | +23.662% | — |

なお前 Gate 文書の `SMALL_L2 +2.007%` は k=7（10.29%）の値であり、23.53% の値ではない。

## 3. Projection definition

学習は禁止。固定 deterministic projection を使う。

- basis: 64×64 Walsh-Hadamard（Sylvester 構成）。行を `1/8` で正規化し、`R R^T = I_d`。
- 各 (layer, tile) について splitmix64 hash で 64 basis 行の permutation と per-rank sign を生成。
- dimension `d` は permutation の先頭 `d` 行を使うため nested（d2 ⊂ d4 ⊂ d8 ⊂ d16）。
- primary seed = 0。

実装は `tools/rnd/ffn_k256_sketch/projection.py`。`R(W x) = (R W) x` は本 Gate では
projected weight を作らないため未検証（Gate B 項目）。

## 4. Information Gate

dataset: layer L0, L16, L28, L32, L48, L63、128 token、全 80 N64 tile。
skip groups `k ∈ {10,14,16,17,20,24,27,28}`。

selector が見るのは sketch `z` のみで、mask 評価には exact `c_g` を使う。
`J_exact(S) = ||Σ_{g∈S} c_g||²`。

単位は `mean J_exact / mean J_exact(exact PAIR_GREEDY_SWAP)`。小さいほど良い。

| selector | k=10 | k=14 | k=16 | k=17 | k=20 | k=24 | k=27 | k=28 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact `SMALL_L2` | 1.853 | 2.146 | 2.266 | 2.322 | 2.449 | 2.534 | 2.548 | 2.546 |
| exact `RESIDUAL_GREEDY` | 1.121 | 1.142 | 1.147 | 1.150 | 1.152 | 1.144 | 1.134 | 1.130 |
| exact `PAIR_GREEDY_SWAP` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| d2 sketch residual | 4.344 | 4.480 | 4.507 | 4.507 | 4.475 | 4.315 | 4.135 | 4.062 |
| d4 sketch residual | 3.304 | 3.510 | 3.568 | 3.590 | 3.623 | 3.561 | 3.441 | 3.395 |
| d4 sketch pair4 | 3.403 | 3.551 | 3.602 | 3.613 | 3.633 | 3.570 | 3.452 | 3.402 |
| d8 sketch residual | 2.667 | 2.873 | 2.940 | 2.969 | 3.014 | 2.986 | 2.909 | 2.873 |
| d8 sketch pair4 | 2.699 | 2.892 | 2.968 | 2.981 | 3.022 | 2.996 | 2.909 | 2.882 |
| d16 sketch pair4 | 2.181 | 2.375 | 2.429 | 2.458 | 2.508 | 2.487 | 2.430 | 2.408 |

読み取り:

- exact では cancellation-aware selector（residual / pair）が `SMALL_L2` を約 2.3x 改善する。
- d8 sketch は `SMALL_L2` すら上回れない（k=16 で 2.94 vs 2.27）。cancellation 情報はほぼ消えている。
- d16 でも 2.2〜2.5 で、exact pair には遠い。
- d4/d8 で sketch pair が sketch residual を改善しない（差 <0.03）。

## 5. Dimension sweep

d2 → d4 → d8 → d16 で単調に改善するが、改善幅は小さい。

| d | k=16 sketch pair/residual | exact pair 比 |
| ---: | ---: | ---: |
| 2 | 4.51 | 4.5x |
| 4 | 3.60 | 3.6x |
| 8 | 2.97 | 3.0x |
| 16 | 2.43 | 2.4x |

d16（64 次元中 16 次元 = 25%）でも exact の 2.4x にとどまる。dimension の問題ではなく、
cancellation 後の残差 `Σc` が小さいため、その二乗ノルムを低次元射影で相対精度よく推定できない
という構造的な限界である。

## 6. Selector comparison

`RESIDUAL_GREEDY` と pair-based（seed_count=4、swap 0/1）を比較した。

- exact では pair が residual を改善する（k=16 で 1.147 → 1.000）。
- sketch では pair の改善がほぼ無い（d8: 2.940 → 2.968、むしろ悪化）。
- swap 1 round は sketch で改善しない（d8 residual 比で +0.01〜0.02 悪化）。

したがって低次元 pair search に価値は認められない。full PPL では
residual と pair4 の両方を確認した（後述）。

## 7. Full PPL

2048-token teacher forcing、positions 2047、seed 0。baseline PPL = 7.063698。
NaN / Inf は全 run で無し。selector は sketch のみを参照し、skip 後 down output は
exact partial から作成した（projected weight は使わない）。

| method | d | skip | PPL | dPPL % |
| --- | ---: | ---: | ---: | ---: |
| exact_pair | 0 | 16 (23.53%) | 7.189854 | +1.786 |
| exact_pair | 0 | 24 (35.29%) | 7.363162 | +4.239 |
| exact_pair | 0 | 27 (39.71%) | 7.405491 | +4.839 |
| exact_pair | 0 | 28 (41.18%) | 7.426296 | +5.133 |
| `SMALL_L2` | 0 | 16 (23.53%) | 7.678605 | +8.705 |
| `SMALL_L2` | 0 | 24 (35.29%) | 8.302741 | +17.541 |
| `SMALL_L2` | 0 | 27 (39.71%) | 8.848750 | +25.271 |
| `SMALL_L2` | 0 | 28 (41.18%) | 8.922187 | +26.310 |
| sketch residual | 4 | 16 (23.53%) | 7.757643 | +9.824 |
| sketch residual | 4 | 24 (35.29%) | 8.597958 | +21.720 |
| sketch residual | 4 | 27 (39.71%) | 9.064056 | +28.319 |
| sketch residual | 4 | 28 (41.18%) | 9.383743 | +32.845 |
| sketch pair4 | 4 | 16 (23.53%) | 7.886375 | +11.647 |
| sketch pair4 | 4 | 24 (35.29%) | 8.527636 | +20.725 |
| sketch pair4 | 4 | 27 (39.71%) | 8.999548 | +27.406 |
| sketch pair4 | 4 | 28 (41.18%) | 9.131988 | +29.281 |
| sketch residual | 8 | 16 (23.53%) | 7.692909 | +8.908 |
| sketch residual | 8 | 24 (35.29%) | 8.426946 | +19.299 |
| sketch residual | 8 | 27 (39.71%) | 8.909117 | +26.125 |
| sketch residual | 8 | 28 (41.18%) | 8.880751 | +25.724 |
| sketch pair4 | 8 | 16 (23.53%) | 7.737831 | +9.544 |
| sketch pair4 | 8 | 24 (35.29%) | 8.369178 | +18.482 |
| sketch pair4 | 8 | 27 (39.71%) | 8.829697 | +25.001 |
| sketch pair4 | 8 | 28 (41.18%) | 8.840275 | +25.151 |
| sketch pair4 | 16 | 16 (23.53%) | 7.580757 | +7.320 |
| sketch pair4 | 16 | 24 (35.29%) | 8.044751 | +13.889 |
| sketch pair4 | 16 | 27 (39.71%) | 8.433980 | +19.399 |
| sketch pair4 | 16 | 28 (41.18%) | 8.591859 | +21.634 |

exact_pair の再現性は良好（前 Gate k=27 +4.714% に対し今回 +4.839%、選択境界の非単調により
±0.5pp 程度の差）。

+5% / +2% 以内の最大 skip（線形補間）:

| method | d | +2% | +5% |
| --- | ---: | ---: | ---: |
| exact_pair | 0 | 24.6% | 40.5% |
| `SMALL_L2` | 0 | 5.4% | 13.5% |
| sketch residual | 4 | 4.8% | 12.0% |
| sketch pair4 | 4 | 4.0% | 10.1% |
| sketch residual | 8 | 5.3% | 13.2% |
| sketch pair4 | 8 | 4.9% | 12.3% |
| sketch pair4 | 16 | 6.4% | 16.1% |

## 8. Gate A decision: NO-GO

仕様の判定基準に照らす。

| 基準 | 条件 | 結果 |
| --- | --- | --- |
| Strong GO | d<=8 で pair k>=27 かつ ΔPPL<=+5%、または residual k>=24 かつ ΔPPL<=+5% | 不成立（d8 pair4 k=27 は +25.0%） |
| GO | d<=8 で skip>=30% かつ ΔPPL<=+5% | 不成立（d8 は +5% で約 12%） |
| HOLD | d16 のみ成立、または d4/d8 が 25〜30% | 不成立（d16 も +5% で 16.1%、+2% で 6.4%） |
| NO-GO | d8 で cancellation advantage がほぼ消える | **該当** |

d8 sketch は magnitude のみの `SMALL_L2` と同等以下（+5% headroom 12.3% vs 13.5%）であり、
cancellation advantage は消失している。d16 でも Quality target（skip>=20% かつ ΔPPL<=+2.5%）
を満たさない。したがって **NO-GO**。

seed robustness（seed 1/2）は仕様 §12 により結果が境界の場合のみ実施するため、今回は未実施。
d8 と exact pair の差（+5% headroom 12% vs 40%）は seed 変更で埋まる幅ではない。

## 9. Projected weight derivation

**未実行**。Gate A 通過後のみ実施する仕様 §14 のため。`B = R W` は生成していない。

参考として仕様 §17 の a-priori traffic 見積り（実測ではない）: d4 INT2 約 1.39MB/layer
（DRAM 600GB/s で約 2.3us）、d8 INT2 約 2.79MB/layer（約 4.7us）、d16 INT2 約 5.57MB/layer
（約 9.3us）。d16 は weight read だけで +5% E2E budget をほぼ使い切る。

## 10. INT2 / INT4 quantization

**未実行**。Gate A の情報量段階で不成立のため、量子化を評価する意味がない。

## 11. Quantized PPL

**未実行**。

## 12. Sketch traffic

**未実行**。

## 13. GPU selector kernel / latency decomposition / fused latency

**未実行**。`SKETCH_ONLY` / `SELECT_ONLY` / `FUSED_SKETCH_SELECT` のいずれも実装していない。
selector が選択を誤る場合、latency が budget 内でも E2E の意味がないため、Gate A 不成立の
時点で打ち切った。

## 14. Sparse down combined latency

**未実行**。

## 15. E2E projection

**未実行**。前 Gate の baseline（27.09 tok/s, 36.91 ms/token）に対する投影は、
品質が成立しないため算出しない。

## 16. Gate

**NO-GO**

- d4/d8 の linear vector sketch は cancellation 情報を保持しない。
- d16 でも Quality / Fast target に届かない。
- INT2 化以前に float sketch の段階で不成立。

仕様 §33 の NO-GO 条件のうち、
「d8 でも cancellation 情報が保てない」「35% skip で +5% 品質を大きく超える」に該当する。

## 17. Key comparison tables

### Information quality

| selector | d | skip | exact J ratio | PPL delta |
| --- | ---: | ---: | ---: | ---: |
| exact pair | 64 | 23.53% | 1.000 | +1.786% |
| exact residual | 64 | — | 1.147 | — |
| `SMALL_L2` | 64 | 23.53% | 2.266 | +8.705% |
| sketch residual | 4 | 23.53% | 3.568 | +9.824% |
| sketch residual | 8 | 23.53% | 2.940 | +8.908% |
| sketch pair4 | 4 | 23.53% | 3.602 | +11.647% |
| sketch pair4 | 8 | 23.53% | 2.968 | +9.544% |
| sketch pair4 | 16 | 23.53% | 2.429 | +7.320% |

### Quantization / Latency / E2E

未実行（Gate A NO-GO）。

## 18. Answers to the Gate questions

1. exact 64D contribution を d4 にすると cancellation 情報は残るか → **残らない**（`SMALL_L2` より悪い）。
2. d8 ならどうか → **残らない**（`SMALL_L2` と同等以下）。
3. residual greedy だけで十分か → 不十分。sketch では pair も改善しない。
4. low-d pair search を使う価値はあるか → **ない**（pair4 ≈ residual、exact pair には遠い）。
5. projected weight を INT2 にしても mask quality は残るか → 未検証（float 段階で不成立）。
6. d4 INT2 selector は何 us/layer か → 未測定。
7. d8 INT2 selector は何 us/layer か → 未測定。
8. 35〜40% skip を +5% PPL 以内で維持できるか → **できない**（d8 は約 12%）。
9. selector 込みで E2E +5% が残るか → **残らない**。
10. production 実装へ進む価値があるか → **ない**。

## 19. Next step

NO-GO のため、linear vector sketch 案は終了候補とする。
projection 改善（学習・非線形）は本 Gate の範囲外であり、別 Gate で扱う。

## 20. Reproduction

```bash
git worktree add .worktrees/ffn-k256-cheap-sketch -b poc/ffn-k256-cheap-sketch 401eb6ec

# tests
python3 tools/rnd/ffn_k256_sketch/test_sketch.py
python3 tools/rnd/ffn_k_tile_oracle/test_oracle.py
python3 tools/rnd/ffn_k_tile_oracle/test_cancel.py
python3 -m compileall -q tools/rnd/ffn_k256_sketch

# information Gate
CUDA_VISIBLE_DEVICES=2,3 python3 tools/rnd/ffn_k256_sketch/gateA_diag.py \
    --layers 0,16,28,32,48,63 --n-tokens 128 --dims 2,4,8,16 \
    --skips 10,14,16,17,20,24,27,28 --seed 0 \
    --out artifacts/ffn_k256_sketch/diagnostic.json

# full PPL (float sketch)
CUDA_VISIBLE_DEVICES=2,3 python3 tools/rnd/ffn_k256_sketch/gateA_ppl.py \
    --configs exact_pair:0,small_l2:0,sk_residual:4,sk_pair4:4,sk_residual:8,sk_pair4:8,sk_pair4:16 \
    --skips 16,24,27,28 --out artifacts/ffn_k256_sketch/ppl_float_sketch.json

# tables
python3 tools/rnd/ffn_k256_sketch/analyze.py --dir artifacts/ffn_k256_sketch \
    --out artifacts/ffn_k256_sketch/info_gate_summary.md
```

artifacts: `artifacts/ffn_k256_sketch/`（git 管理外）。
