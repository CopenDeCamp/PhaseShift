> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# Qwen3.8-27B Quaternion-PSQ 圧縮 PoC

4x4 weight block を Quaternion 構造へ写像し、R1/R2/R3 の自由度削減で
2.0〜3.0 bpw の圧縮余地があるかを CPU 上で検証する。
production format (`WeightEncoding` / `QuantFormatId` / manifest schema /
PSQ4 kernel) は一切変更しない。

## Revision

| 項目 | 値 |
| --- | --- |
| baseline SHA | `ef98918c38cf7def1b63bddb845801280476beb5` |
| branch | `poc/quaternion-psq` |
| worktree | `.worktrees/quaternion-psq-poc` |
| tool | `tools/quantization/quaternion_psq.py`（解析・selftest） |
| 集計 / report | `analyze_quaternion_psq.py` / `report_quaternion_psq.py` |
| report | `docs/rnd/quantization/quaternion_psq_poc.md` |
| final | `2026-09-23` |

## Environment

| 項目 | 値 |
| --- | --- |
| model (参照 original) | `models/Qwen3.8-27B (Qwen/Qwen3.8-27B, 18 shard, 51.74 GiB)` |
| production PSQ | `models/Qwen3.8-27B-PSQ (production, 参照のみ)` |
| iMatrix | なし（`*.psim` が存在しないため未使用、通常 MSE で評価） |
| 対象 | PSQ4 FFN tensor 層方向 stratified サンプル 19 tensor。全 176 tensor 走査は 75/176 で打ち切り、残りは代表層で確認 |
| 対象 weight | 1,693,450,240 個（1693.5 M） |
| tile 数 | 105,840,640 |
| GPU | 不使用（CPU / numpy） |

参照 original は `Qwen/Qwen3.8-27B` の BF16 safetensors を直接読む。
PSQ4 baseline は同一 weight を `tools/quantization/psq4.py` の
CB10 (K32 + bf16 scale, 4.5 bpw) で量子化して測る。
`phaseshift_quantization.json` の encoding が `psq4` の FFN tensor のみを対象とした。

## Method

- `W[N,K]` を 4 output x 4 input の `W4x4` tile に分割する。
- Quaternion `q = a + b i + c j + d k` の左 Hamilton 積を実 4x4 行列
  `L(q) = H(a,b,c,d)`、右積を `R(q)` とする。
- 16 個の直交基底 `B[c*4+r] = L(e_r) R(e_c)` を生成し、tile の 16 要素を
  16 次元係数 `x = vec(W) @ B^T / 4` へ射影する。
- `R` は右基底 `e_c` のうち energy 上位 `R` 群のみを残した再構成誤差で評価する。
  `fixed` は `c=0..R-1`、`dyn` は tile ごとに energy 上位 `R` 群を選ぶ。
- R1 は Pure Quaternion (`H(a,b,c,d)`) と一致する。係数は Q1 まで FP32。
- `p50/p90/p99/worst` は tile 単位の captured energy の分布。
- `E/Q` は Quaternion 候補の relative Frobenius RMS を PSQ4 baseline の
  それで割った比。1.0 未満なら PSQ4 より高精度。

## Self-test

`quaternion_psq.py selftest` は全項目 PASS。
- Quaternion 積: `i*i=j*j=k*k=-1`, `i*j=k`, `j*k=i`, `k*i=j`, 反交換側も一致。
- Pure Quaternion 射影: 解析解が `numpy.linalg.lstsq` と max diff 1.1e-16 で一致。
  `project(H(a,b,c,d))` が元の係数を復元。
- 16 基底: rank 16、`M M^T = 4 I`（直交・等ノルム）。
- R4 再構成: max abs error 4.4e-16 (< 1e-6)。
- `quant_generic` が `psq4.quant_block` と scale / err まで一致。

## Gate Q0 — Pure Quaternion

| role | PSQ4 relRMS | PureQ captured | p50 | p90 | p99 | worst | E/Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ffn_gate | 0.08615 | 0.2500 | 0.2288 | 0.4512 | 0.6408 | 0.0000 | 10.05 |
| ffn_up | 0.08608 | 0.2500 | 0.2288 | 0.4517 | 0.6412 | 0.0000 | 10.06 |
| ffn_down | 0.08791 | 0.2500 | 0.2298 | 0.4482 | 0.6352 | 0.0001 | 9.85 |

Pure Quaternion は 25% 前後で飽和する。16 basis が直交変換であるため、
i.i.d. な 4x4 block では期待値が厳密に 1/4 になる。実測が
PSQ4 (relRMS 0.086, captured 99.3%) の約 10 倍の誤差であり、
Quaternion rank-1 部分空間は weight の主成分ではない。

## Gate Q1 — Quaternion Basis

| role | R1 | R2 | R3 | R4 relRMS | R1d | R2d | R3d | R2d E/Q | R3d E/Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ffn_gate | 0.2500 | 0.5001 | 0.7501 | 0.00000000 | 0.4416 | 0.7182 | 0.8987 | 6.16 | 3.70 |
| ffn_up | 0.2500 | 0.5000 | 0.7500 | 0.00000000 | 0.4420 | 0.7186 | 0.8989 | 6.16 | 3.69 |
| ffn_down | 0.2500 | 0.5001 | 0.7502 | 0.00000000 | 0.4379 | 0.7153 | 0.8971 | 6.07 | 3.65 |

### captured energy 分布 (dynamic)

| role | method | p50 | p90 | p95 | p99 | worst |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ffn_gate | r2d | 0.7153 | 0.8317 | 0.8618 | 0.9113 | 0.5005 |
| ffn_gate | r3d | 0.9022 | 0.9627 | 0.9742 | 0.9888 | 0.7503 |
| ffn_up | r2d | 0.7153 | 0.8317 | 0.8622 | 0.9113 | 0.5008 |
| ffn_up | r3d | 0.9022 | 0.9627 | 0.9742 | 0.9888 | 0.7505 |
| ffn_down | r2d | 0.7127 | 0.8293 | 0.8598 | 0.9093 | 0.5022 |
| ffn_down | r3d | 0.9008 | 0.9617 | 0.9738 | 0.9888 | 0.7513 |

fixed truncation は R1/R2/R3 = 25/50/75% と `R/4` に一致する。
これは基底が直交であることから、係数 energy が等方なときに
厳密に成立する値であり、weight 固有の低次元構造が存在しないことを示す。
dynamic top-R は 44/72/90% まで上がるが、これは order statistics
（大きい群を選ぶ効果）であり、直交変換後の係数が等方でも得られる。

## Control — real vs permutation vs iid

| tensor | R1 | R2 | R3 | R1d | R2d | R3d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.mlp.gate_proj.weight | 0.2495 | 0.4990 | 0.7496 | 0.4417 | 0.7182 | 0.8988 |
| 0.mlp.gate_proj.weight (perm) | 0.2497 | 0.5004 | 0.7500 | 0.4428 | 0.7191 | 0.8991 |
| 0.mlp.gate_proj.weight (iid) | 0.2501 | 0.5006 | 0.7506 | 0.4437 | 0.7197 | 0.8993 |
| 32.mlp.up_proj.weight | 0.2506 | 0.5004 | 0.7499 | 0.4410 | 0.7178 | 0.8985 |
| 32.mlp.up_proj.weight (perm) | 0.2505 | 0.4997 | 0.7502 | 0.4412 | 0.7179 | 0.8986 |
| 32.mlp.up_proj.weight (iid) | 0.2511 | 0.5015 | 0.7501 | 0.4436 | 0.7197 | 0.8993 |
| 63.mlp.down_proj.weight | 0.2501 | 0.5000 | 0.7503 | 0.4374 | 0.7150 | 0.8970 |
| 63.mlp.down_proj.weight (perm) | 0.2496 | 0.5000 | 0.7496 | 0.4394 | 0.7165 | 0.8977 |
| 63.mlp.down_proj.weight (iid) | 0.2498 | 0.4996 | 0.7498 | 0.4437 | 0.7201 | 0.8996 |

tile 内の 16 要素をランダム置換したもの、および i.i.d. 正規乱数は、
実 weight とほぼ同一の captured energy を与える。Quaternion 基底は
4x4 tile に対する Hadamard 的直交変換であり、実 weight には
その係数空間で他と区別できる structure が存在しない。

## Gate Q2 — Quaternion 係数の PSQ4 化

CB10 4bit + BF16 scale 共有 (S = 共有 tile 数) で係数を量子化した。
代表値 (layer 16 gate_proj, 128 row, PSQ4 relRMS 0.08582)。
bpw は code + selector + scale。

| mode | S4 bpw | S4 err | S8 bpw | S8 err | S16 bpw | S16 err | E/Q (R2d S8) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed R1 | 1.2500 | 0.86730 | 1.1250 | 0.86755 | 1.0625 | 0.86777 | 10.11 |
| dyn R1 | 1.3750 | 0.74782 | 1.2500 | 0.74825 | 1.1875 | 0.74859 | 8.72 |
| fixed R2 | 2.2500 | 0.71007 | 2.1250 | 0.71060 | 2.0625 | 0.71102 | 8.28 |
| dyn R2 | 2.4375 | 0.53412 | 2.3125 | 0.53499 | 2.2500 | 0.53565 | 6.23 |
| fixed R3 | 3.2500 | 0.50593 | 3.1250 | 0.50693 | 3.0625 | 0.50765 | 5.91 |
| dyn R3 | 3.3750 | 0.32807 | 3.2500 | 0.32974 | 3.1875 | 0.33095 | 3.84 |
| fixed R4 | 4.2500 | 0.09399 | 4.1250 | 0.10016 | 4.0625 | 0.10435 | 1.18 |
| dyn R4 | 4.3750 | 0.09399 | 4.2500 | 0.10016 | 4.1875 | 0.10435 | 1.18 |

- R2 (8 次元) は S8 dynamic で 2.3125 bpw、relRMS 0.535、E/Q 6.2。
- R3 (12 次元) は S16 dynamic で 3.1875 bpw、relRMS 0.331、E/Q 3.9。
- R4 (16 次元) は 4.06 bpw、relRMS 0.104、E/Q 1.2。完全基底で
  4bit 係数化した Hadamard codec は PSQ4 近傍だが 4 bpw であり、
  目標 2〜3 bpw の外側にある。

## Gate Q3 — Sparse Residual

`E = W - W_R2dyn` の |E| 上位 k 要素のみを残す oracle 評価。
position metadata は 4bit / residual、値は FP16 (下限)。

| residual | extra bpw (position+value, oracle FP16) | relRMS | E/Q |
| --- | ---: | ---: | ---: |
| R2d top-1 | +1.25 | 0.4681 | 5.45 |
| R2d top-2 | +2.50 | 0.3969 | 4.62 |
| R2d top-4 | +5.00 | 0.3040 | 3.54 |

- R2d-S8 (2.3125 bpw) + k=1 FP16 は 3.56 bpw で E/Q 5.45。
- position 4bit + 値 4bit + scale 共有 S8 でも R2 系で 2.94 bpw、
  E/Q 5.4 程度に留まる。
- E/Q を 1 に近づけるには R4 相当の全 16 次元が必要で、そこでは
  4 bpw を下回らない。

## 集約表 (手順書 §42 相当)

| Method | BPW | Error/PSQ4 |
| --- | ---: | ---: |
| PSQ4 | 4.50 | 1.00 |
| Pure Quaternion (R1 fixed, FP32) | (4 coeff/tile) | 10.1 |
| Q-R1 dyn (S8, 4bit coeff) | 1.25 | 8.7 |
| Q-R2 dyn (S8, 4bit coeff) | 2.31 | 6.2 |
| Q-R2 dyn + residual k1 (S8, oracle) | 2.94 | 5.4 |
| Q-R3 dyn (S16, 4bit coeff) | 3.19 | 3.9 |
| Q-R4 (S16, 4bit coeff) | 4.06 | 1.2 |

BPW は code + selector + scale を含む。residual は position metadata を含む。
係数を量子化しない Q1 の時点で既に R2 が E/Q 6.2、R3 が E/Q 3.7 であり、
4bit 係数化はこの構造誤差をほとんど増やさない。

## per-layer 分布

### ffn_gate

| layer | R2d captured | R3d captured | PSQ4 relRMS | R2d E/Q |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7187 | 0.8989 | 0.08589 | 6.18 |
| 8 | 0.7193 | 0.8992 | 0.08574 | 6.18 |
| 16 | 0.7191 | 0.8991 | 0.08584 | 6.17 |
| 24 | 0.7190 | 0.8991 | 0.08590 | 6.17 |
| 32 | 0.7182 | 0.8987 | 0.08648 | 6.14 |
| 40 | 0.7180 | 0.8986 | 0.08593 | 6.18 |
| 48 | 0.7174 | 0.8982 | 0.08658 | 6.14 |
| 56 | 0.7184 | 0.8988 | 0.08601 | 6.17 |
| 63 | 0.7165 | 0.8978 | 0.08676 | 6.14 |

### ffn_up

| layer | R2d captured | R3d captured | PSQ4 relRMS | R2d E/Q |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7193 | 0.8992 | 0.08578 | 6.18 |
| 8 | 0.7193 | 0.8993 | 0.08580 | 6.17 |
| 16 | 0.7192 | 0.8992 | 0.08585 | 6.17 |
| 24 | 0.7189 | 0.8991 | 0.08611 | 6.16 |
| 32 | 0.7182 | 0.8987 | 0.08645 | 6.14 |
| 40 | 0.7189 | 0.8991 | 0.08616 | 6.15 |
| 48 | 0.7182 | 0.8987 | 0.08636 | 6.15 |
| 56 | 0.7190 | 0.8991 | 0.08591 | 6.17 |
| 63 | 0.7172 | 0.8981 | 0.08610 | 6.18 |

### ffn_down

| layer | R2d captured | R3d captured | PSQ4 relRMS | R2d E/Q |
| ---: | ---: | ---: | ---: | ---: |
| 63 | 0.7153 | 0.8971 | 0.08791 | 6.07 |

## Quality (KLD/PPL)

NO-GO のため KLD/PPL は実施しない。手順書 §33 は CPU tensor error が
良好な場合のみ candidate model を生成するとしており、今回は該当しない。

## Conclusion

**NO-GO**

| 判定 | 条件 | 本結果 |
| --- | --- | --- |
| Strong GO | R2/R2+residual <= 3.0 bpw で production PSQ 近傍 | 不成立 |
| GO | <= 3.0 bpw、品質は回復余地あり | 不成立 |
| Borderline | 3.0〜3.3 bpw または品質劣化が明確 | R3 は 3.19 bpw だが E/Q 3.9 |
| NO-GO | >3.3 bpw 必要、または R3 でも精度不足 | **該当** |

理由:
- Quaternion 16 基底は 4x4 tile の直交変換であり、実 weight の係数 energy は
  i.i.d. と区別できない。低次元 (R2/R3) へ落とすと energy が一様に失われる。
- PSQ4 は 4.5 bpw で captured 99.3% を達成する。Quaternion で同等精度には
  16 次元（R4）が必要で、その時点で 4 bpw を下回らない。
- 2〜3 bpw 領域 (R2/R3) の誤差は PSQ4 の 4〜6 倍で、sparse residual を
  budget 内で足しても PSQ4 に近づかない。
- したがって Quaternion GEMV/GEMM kernel は実装しない（手順書 §34, §44）。

## 参考: 構造の解釈

16 基底 `B[c*4+r]` は符号付き置換行列であり、係数は tile 16 要素の
符号反転和 (Hadamard 変換) である。したがって本 PoC は
「4x4 tile の Hadamard spectrum がスパースか」を測ったことに等しく、
結果は flat spectrum（構造なし）だった。Quaternion という代数そのものは
任意の 4x4 を表現できるが、Qwen の FFN weight はその低次元部分空間に
集中していない。

