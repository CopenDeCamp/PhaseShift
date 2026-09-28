# FFN K256 candidate-prefiltered high-D cancellation selector Gate

## 1. Goal

前 Gate `Cheap Cancellation-Aware Vector Sketch Gate` では、全 68 K256 group を一律に
d4/d8/d16 へ薄く圧縮する方式が NO-GO となった。

- exact `PAIR_GREEDY_SWAP`: 23.5% skip → +1.8%、35.3% → +4.2%、39.7% → +4.8%、41.2% → +5.1%
- d8 sketch: `J / exact pair` ≈ 2.9〜3.0、+5% headroom ≈ 12%
- d16 sketch: `J / exact pair` ≈ 2.4、+5% headroom ≈ 16%

今回の仮説はこれと別である。

> cheap scalar で skip 候補 pool だけを 32〜48 group へ絞り、
> その候補にのみ 16D〜32D 程度の濃い direction 情報を使えば、
> cancellation headroom を維持しつつ runtime traffic を budget 内へ戻せるか。

段階を混ぜない。Stage A（pool 情報）→ Stage B（pool 内高次元 sketch）→ traffic →
Stage C（projected weight / INT2）→ Stage D（GPU）の順に判定し、
Stage A/B で NO-GO なら以降を実装しない。

結論: **Stage A HOLD、Stage B NO-GO、最終 Gate NO-GO**。
candidate narrowing 自体は headroom をほぼ維持するが、pool 内を高次元投影すると
cancellation geometry が再び崩壊し、35% skip を +5% 以内に保てない。

## 2. Previous sketch NO-GO

前 Gate の数値は `docs/perf/ffn_k256_sketch.md` を正本とする。全 68 group の線形 sketch は
`J / exact pair` が 2.9（d8）〜2.4（d16）まで悪化し、+5% headroom は 12〜16% だった。

## 3. Candidate pool hypothesis

各 group に cheap score `q_g` を計算し、`q` の小さい C 個を candidate pool とする。
pool 外は強制 KEEP。pool は「必ず skip する group」ではなく
「cancellation search に参加できる group」である。

## 4. Prefilter definitions

| 名前 | score | 性質 |
| --- | --- | --- |
| `exact_l2` | `q_g = ||c_g||²` | production 候補ではない上限。exact magnitude |
| `act_energy` | `q_g = a_g = Σ x_j²` | 68 scalar、全 N64 tile 共通。最安 |
| `act_wnorm` | `q_g = a_g × ||W[tile,g]||_F²` | weight norm は offline metadata。primary cheap 候補 |

weight norm metadata は `n[tile,g] = Σ W[tile,g]²`（80×68/layer、決定論的に生成）。

## 5. Exact restricted Oracle

pool 決定後、pool 内で exact 64D `PAIR_GREEDY_SWAP`（seed_count=16, swap 8）を実行し、
選ばれた mask を exact 64D contribution で採点する。unrestricted C68 も同一 run で再測定する。

## 6. Candidate recall

exact unrestricted が選んだ skip 集合 `S*` の pool 内 recall `|S* ∩ C| / |S*|`。
6 layer（L0,16,28,32,48,63）× 128 token × 80 tile の aggregate。

| prefilter | C | k=16 | k=24 | k=27 | rank median | rank p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_l2 | 32 | 0.77 | 0.69 | 0.67 | 20 | 49 |
| exact_l2 | 40 | 0.87 | 0.80 | 0.78 | 20 | 49 |
| exact_l2 | 48 | 0.93 | 0.89 | 0.87 | 20 | 49 |
| exact_l2 | 56 | 0.98 | 0.95 | 0.95 | 20 | 49 |
| act_energy | 48 | 0.91 | 0.87 | 0.86 | 22 | 51 |
| act_energy | 56 | 0.96 | 0.94 | 0.93 | 22 | 51 |
| act_wnorm | 48 | 0.91 | 0.87 | 0.86 | 22 | 51 |
| act_wnorm | 56 | 0.96 | 0.94 | 0.93 | 22 | 51 |

recall は C48 で 0.87〜0.91、C56 で 0.93〜0.96。exact が選ぶ group の prefilter rank は
median 20〜22、p90 49〜51 で、下位に集中してはいるが p90 は pool 外にも及ぶ。

## 7. Pool-size sweep

`J / unrestricted exact pair`（小さいほど良い）。

| prefilter | C | k=16 | k=20 | k=24 | k=27 | k=28 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unrestricted | 68 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_l2 | 32 | 1.224 | 1.414 | 1.660 | 1.888 | 1.978 |
| exact_l2 | 40 | 1.100 | 1.200 | 1.325 | 1.434 | 1.473 |
| exact_l2 | 48 | 1.038 | 1.084 | 1.146 | 1.201 | 1.220 |
| exact_l2 | 56 | 1.010 | 1.026 | 1.050 | 1.073 | 1.080 |
| act_energy | 48 | 1.068 | 1.122 | 1.193 | 1.254 | 1.277 |
| act_energy | 56 | 1.024 | 1.045 | 1.074 | 1.100 | 1.109 |
| act_wnorm | 48 | 1.068 | 1.122 | 1.194 | 1.255 | 1.277 |
| act_wnorm | 56 | 1.024 | 1.045 | 1.074 | 1.099 | 1.108 |

前 Gate の全 68 d8 sketch が 2.9〜3.0 だったのに対し、候補絞り込みでは C48 で 1.15〜1.26、
C56 で 1.05〜1.11 まで戻る。pool 制限は cancellation headroom をほぼ維持する。

## 8. Full PPL

2048-token teacher forcing、positions 2047。baseline PPL ≈ 7.0636。
pool 外は強制 KEEP、skip 後 down output は exact partial から作成（projected weight は使わない）。
PPL は 3 process に分けて測定した（run1: exact_l2、run2: act_wnorm、run3: act_energy）。
run1 と run3 は同一 process で `unrestricted` C68 を再測定した（run1 +2.08 / +3.53 / +4.06 / +5.60、
run3 +1.68 / +4.19 / +5.03 / +5.52）。run2 は `unrestricted` が完走しなかったため、
act_wnorm は run1/run3 の unrestricted を参照する。選択境界のため unrestricted は run 間で
±0.5〜1.0pp 揺れるが、Gate 判定は absolute ΔPPL で行う。

| prefilter | C | k=16 | k=24 | k=27 | k=28 |
| --- | ---: | ---: | ---: | ---: | ---: |
| unrestricted (run1) | 68 | +2.08 | +3.53 | +4.06 | +5.60 |
| unrestricted (run3) | 68 | +1.68 | +4.19 | +5.03 | +5.52 |
| exact_l2 | 40 | +3.32 | +7.54 | +9.83 | +11.82 |
| exact_l2 | 48 | +2.95 | +4.95 | +8.55 | +8.55 |
| exact_l2 | 56 | +2.01 | +4.82 | +5.87 | +7.21 |
| act_wnorm | 48 | +2.65 | +6.41 | +7.80 | +8.88 |
| act_wnorm | 56 | +1.76 | +5.07 | +6.38 | +6.05 |
| act_energy | 48 | +2.51 | +6.04 | +7.96 | +9.74 |
| act_energy | 56 | +2.69 | +4.43 | +5.50 | +6.85 |

+5% 以内の最大 skip（線形補間）:

| prefilter | C | +2% | +5% |
| --- | ---: | ---: | ---: |
| unrestricted | 68 | 23.5% | 39.7% |
| exact_l2 | 48 | 16.0% | 35.4% |
| exact_l2 | 56 | 23.4% | 36.0% |
| act_wnorm | 48 | 17.7% | 30.9% |
| act_wnorm | 56 | 24.4% | 35.1% |
| act_energy | 48 | 18.8% | 31.8% |
| act_energy | 56 | 17.5% | 37.7% |

## 9. Stage A decision: HOLD

| 基準 | 条件 | 結果 |
| --- | --- | --- |
| STRONG GO | cheap, C≤40, k≥24, ΔPPL≤+5% | 不成立（act_wnorm C40 は未測定だが、`J` 比 1.405 > exact_l2 C40 の 1.325 = +7.54% なので ≤+5% は不可能） |
| GO | act_wnorm / act_energy, C≤48, k≥24, ΔPPL≤+5% | 不成立（act_wnorm C48 k24 +6.41%、act_energy +6.04%） |
| HOLD | cheap prefilter で C48〜56 必要 | **該当**（C56 k24 = +4.4〜5.1%） |
| NO-GO | exact_l2 C48 でも k24 が +5% 超 | 不成立（+4.95%） |

candidate narrowing という発想は成立する。cheap prefilter は C56 で unrestricted に近い
+5% headroom（35〜38% vs 39.7%）を保つが、C≤48 では 31% 程度に落ち、GO 条件には届かない。

## 10. High-D pool sketch

Stage B は pool 決定後に候補のみを d 次元へ投影する。projection は前 Gate と同じ
deterministic Hadamard（per (layer,tile) hash permutation + sign、nested、seed 0）。

## 11. Dimension sweep

`J / pool-exact-pair`（pool 内 exact pair に対する比）。

| prefilter | C | d | selector | k=16 | k=24 | k=27 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| act_wnorm | 48 | 16 | residual | 2.20 | 2.00 | 1.86 |
| act_wnorm | 48 | 24 | residual | 1.91 | 1.76 | 1.65 |
| act_wnorm | 48 | 32 | residual | 1.68 | 1.56 | 1.48 |
| act_wnorm | 48 | 48 | residual | 1.35 | 1.29 | 1.25 |
| exact_l2 | 48 | 16 | residual | 2.25 | 2.05 | 1.91 |
| exact_l2 | 48 | 32 | residual | 1.72 | 1.60 | 1.51 |
| act_wnorm | 56 | 16 | residual | 2.33 | 2.27 | 2.16 |
| act_wnorm | 56 | 32 | residual | 1.76 | 1.72 | 1.66 |

pair4 は residual とほぼ同等（差 <0.02）。pool 内 C48 d16 の `J / unrestricted` は
k=24 で 2.35 となり、前 Gate の全 68 d16（2.43）とほぼ同じ。
pool 制限は sketch の geometry 崩壊を緩和しない。

## 12. Stage B PPL

同じ 2048-token。baseline PPL ≈ 7.0606。

| prefilter | C | d | selector | k=16 | k=24 | k=27 | k=28 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| act_wnorm | 48 | 16 | residual | +7.03 | +14.12 | +19.97 | +21.09 |
| act_wnorm | 48 | 24 | residual | +6.96 | +13.93 | +17.14 | +18.04 |
| act_wnorm | 48 | 32 | residual | +4.36 | +11.74 | +13.83 | +15.01 |
| exact_l2 | 48 | 16 | residual | +6.03 | +16.18 | +19.57 | +20.56 |
| act_wnorm | 48 | 24 | pair4 | +6.20 | +12.74 | +16.49 | +18.30 |

+5% headroom: d16 16.7%、d24 16.9%、d32 24.5%（いずれも 35% に遠い）。

## 13. Traffic lower bound

`traffic_model.json`。INT2 code `80 × C × d × 256 × 2/8` + scale（fp16, 1/row/K256）、
sustained 600 GB/s（break-even Gate 実測 578〜636）。

| C | d | format | code MB/L | total MB/L | lower-bound us/L | +5% (≤7us) |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| 32 | 16 | INT2 | 2.62 | 2.70 | 4.51 | yes |
| 40 | 16 | INT2 | 3.28 | 3.38 | 5.63 | yes |
| 48 | 16 | INT2 | 3.93 | 4.06 | 6.76 | yes |
| 32 | 24 | INT2 | 3.93 | 4.06 | 6.76 | yes |
| 40 | 24 | INT2 | 4.92 | 5.07 | 8.45 | tight |
| 32 | 32 | INT2 | 5.24 | 5.41 | 9.01 | tight |
| 48 | 24 | INT2 | 5.90 | 6.08 | 10.14 | no |
| 48 | 32 | INT2 | 7.86 | 8.11 | 13.52 | no |

Stage B が要求する `C48/d32` は lower bound だけで 13.5 us/L であり、
+5% E2E budget（≈10us/L）を超える。品質が成立しても traffic で NO-GO となる。

## 14. Stage B decision: NO-GO

| 基準 | 条件 | 結果 |
| --- | --- | --- |
| STRONG GO | cheap, d≤24, k≥24, ΔPPL≤+5% | 不成立（d24 k24 +13.9%） |
| GO | C≤40/d≤32 または C≤48/d≤24, k≥24, ΔPPL≤+5% | 不成立（C48/d24 +13.9%） |
| CONDITIONAL | d32/C48 以上で品質成立 | 不成立（C48/d32 k24 +11.7%） |
| NO-GO | d32 でも k24 で +5% を維持できない | **該当** |

pool 内 d32 でも k24 は +11.7%、+5% headroom は 24.5%。さらに d32 は traffic も 13.5us/L。
candidate を絞って dimension を上げても cancellation geometry は保存されない。

## 15. Projected weights

**未実行**。Stage B NO-GO のため、`B = R W` は生成していない。

## 16. INT2 / INT4 quantization

**未実行**。

## 17. GPU prefilter / sketch / select / fused

**未実行**。

## 18. Sparse-down combined

**未実行**。

## 19. E2E projection

**未実行**。品質が成立しないため投影しない。

## 20. Final Gate: NO-GO

- Stage A: HOLD（cheap prefilter は C48〜56 必要、candidate narrowing は成立）。
- Stage B: NO-GO（pool 内 d32 でも k24 +11.7%、+5% headroom 24.5%）。
- traffic: d32/C48 は 13.5 us/L で budget 超過。

spec §33 の NO-GO 条件「d32 でも k24 +5% を維持できない」に該当する。
`Candidate-prefiltered linear high-D sketch` は終了候補とする。

## 21. Required tables

### Table 1: Candidate pool

| prefilter | C | k | recall exact mask | J/exact | dPPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| unrestricted | 68 | 24 | 1.000 | 1.000 | +3.53 |
| exact_l2 | 32 | 24 | 0.69 | 1.660 | — |
| exact_l2 | 40 | 24 | 0.80 | 1.325 | +7.54 |
| exact_l2 | 48 | 24 | 0.89 | 1.146 | +4.95 |
| exact_l2 | 56 | 24 | 0.95 | 1.050 | +4.82 |
| act_energy | 48 | 24 | 0.87 | 1.193 | +6.04 |
| act_energy | 56 | 24 | 0.94 | 1.074 | +4.43 |
| act_wnorm | 48 | 24 | 0.87 | 1.194 | +6.41 |
| act_wnorm | 56 | 24 | 0.94 | 1.074 | +5.07 |

k=27/28 も §7/§8 に同形式で記載。

### Table 2: High-D sketch

| prefilter | C | d | selector | k | J/pool-exact | dPPL |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| act_wnorm | 48 | 16 | residual | 24 | 2.00 | +14.12 |
| act_wnorm | 48 | 24 | residual | 24 | 1.76 | +13.93 |
| act_wnorm | 48 | 32 | residual | 24 | 1.56 | +11.74 |
| act_wnorm | 48 | 24 | pair4 | 24 | 1.75 | +12.74 |

### Table 3: Traffic

§13 を参照。

### Table 4: Final

未実行（Stage B NO-GO）。

## 22. Answers to the Gate questions

1. exact が選ぶ group は scalar score 下位 32〜48 に集中するか → 部分的。recall C48 ≈ 0.87〜0.89、
   rank median 20〜22 だが p90 は 49〜51。
2. pool を絞るだけなら 40% 近い headroom は残るか → C56 で +5% headroom 35〜38%（unrestricted 39.7%）。
   C48 では 31% 前後。
3. ACT_WEIGHT_NORM 程度の cheap prefilter でも pool を作れるか → 作れる。exact_l2 比で
   `J` は僅かに悪いだけ（C48 k24 で 1.194 vs 1.146）。
4. C32/C40 へ絞れば d16 で direction 情報が復活するか → 復活しない（`J/pool-exact` ≈ 2.0）。
5. d24 ではどうか → 1.76 で不十分。
6. d32 まで必要か → d32 でも 1.56、PPL k24 +11.7%。d48 でも 1.29。
7. 必要 dimension × candidate count の traffic → C48/d32 で 8.11 MB/L、13.5us/L。
8. INT2 にしても品質は残るか → 未検証（float sketch 段階で不成立）。
9. prefilter + sketch + select は何 us/L か → 未測定（Stage B NO-GO）。
10. sparse down 込みで E2E +3〜5% が残るか → 残らない。
11. candidate-prefilter 方式を production へ進める価値があるか → ない。

## 23. Next step

spec §42 に従い、`Pilot N16 / Sparse N48 Online Selector Gate` へ切り替える。

- N64 を 4×N16 とみなし、最初の N16 を dense 計算して実 contribution 方向を得る。
- その方向で残り N48 のみを skip する。
- extra projected weight 不要、probe N16 は最終 output の一部、64D を人工圧縮しない。
- 制約: 最大 saving は残り 75% に限定、kernel 構造変更が大きい。

d48/d64 brute-force へは進まない。

## 24. Reproduction

```bash
git worktree add .worktrees/ffn-k256-candidate-selector \
    -b poc/ffn-k256-candidate-selector 7f3dcb0c

# tests
python3 tools/rnd/ffn_k256_candidate_selector/test_pool.py
python3 tools/rnd/ffn_k256_candidate_selector/test_stageb.py
python3 tools/rnd/ffn_k_tile_oracle/test_oracle.py
python3 tools/rnd/ffn_k_tile_oracle/test_cancel.py
python3 -m compileall -q tools/rnd/ffn_k256_candidate_selector

# Stage A
CUDA_VISIBLE_DEVICES=2,3 python3 tools/rnd/ffn_k256_candidate_selector/gateA_pool_diag.py \
    --layers 0,16,28,32,48,63 --n-tokens 128 --pools 32,40,48,56,68 \
    --skips 16,20,24,27,28 --out artifacts/ffn_k256_candidate_selector/pool_diag.json
CUDA_VISIBLE_DEVICES=2,3 python3 tools/rnd/ffn_k256_candidate_selector/gateA_pool_ppl.py \
    --configs unrestricted:68,exact_l2:40,exact_l2:48,exact_l2:56 \
    --skips 16,24,27,28 --out artifacts/ffn_k256_candidate_selector/pool_ppl_run1.json

# Stage B
CUDA_VISIBLE_DEVICES=2,3 python3 tools/rnd/ffn_k256_candidate_selector/gateB_pool_diag.py \
    --prefilters act_wnorm,exact_l2 --pools 48,56 --dims 16,24,32,48 \
    --methods residual,pair4 --skips 16,20,24,27,28 \
    --out artifacts/ffn_k256_candidate_selector/sketch_pool_diag.json
CUDA_VISIBLE_DEVICES=2,3 python3 tools/rnd/ffn_k256_candidate_selector/gateB_pool_ppl.py \
    --configs act_wnorm:48:16:residual,act_wnorm:48:24:residual,act_wnorm:48:32:residual \
    --skips 16,24,27,28 --out artifacts/ffn_k256_candidate_selector/sketch_pool_ppl.json

python3 tools/rnd/ffn_k256_candidate_selector/traffic_model.py
```

artifacts: `artifacts/ffn_k256_candidate_selector/`（git 管理外）。
