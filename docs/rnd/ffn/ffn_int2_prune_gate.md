> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# FFN Dynamic Sparse Gate（Exact Oracle → INT2 candidate → exact rerank）

更新日: 2026-09-24
branch: `poc/ffn-int2-prune`（harness `tools/rnd/ffn_prune/` はこの branch にのみ存在）

## 0. 目的と方針補正

Qwen3.8-27B（Qwen3.5 Dense 構成、intermediate 17408）の FFN について、
token ごとに intermediate channel を落とす dynamic sparse FFN の成立性を
PPL 劣化 ↔ keep ratio ↔ (将来の) tok/s の Pareto として測定する。

従来案 `INT2 → final mask → selected PSQ8` は、INT2 単独で final decision を
行う点が誤りだった。補正後は次の二段構成とする。

```text
INT2 proxy
    ↓
candidate prefilter
    ↓
candidate のみ exact Gate / Up / SwiGLU
    ↓
candidate 内部の exact 値だけで rerank
    ↓
final pruning mask
    ↓
selected Down
```

本ドキュメントは Gate 1（candidate containment）と Gate 2（candidate +
exact rerank の PPL）まで、および break-even の理論計算を対象とする。
native kernel / E2E tok/s は Gate 通過時のみ実施する。

## 1. 環境

| 項目 | 値 |
| --- | --- |
| GPU | AMD Radeon AI PRO R9700 (gfx1201) ×3 |
| ROCm / HIP | 7.15.26333 |
| BF16 source | `models/Qwen3.8-27B`（55.6GB） |
| 実 PSQ | `models/Qwen3.8-27B-PSQ`（gate/up psq4 4.5bpw、down psq4/psq8 混在 5.5bpw） |
| corpus | `artifacts/ffn_prune/corpus.psktok`（2048 token） |
| harness | `tools/rnd/ffn_prune/`（Python / torch / transformers 5.14.1） |

27B BF16 は host RAM に載らないため safetensors を shard 単位で streaming し、
`infer_auto_device_map` の配置で 3 GPU へ直接ロードする（`stream_load.py`）。
4B モデルで `from_pretrained` と logits bit-exact を確認済み。

## 2. 手法

`Qwen3_5MLP.forward` を差し替え、exact な Gate/Up/SwiGLU の intermediate
activation `h` に対し token ごとに channel を選択して落とす。Down は線形なので
channel を落とすことはその Down 寄与を 0 にすることと厳密に等価。

スコア（channel 単位、group 化する場合は group 内和）:

```text
A = Σ |h_i|
B = Σ |h_i| · ||W_down[:,i]||
C = Σ h_i^2 · ||W_down[:,i]||^2
T = ||W_down[:,group] · h[group]||^2   (true energy oracle)
```

`T` は `W_g^T W_g` を layer ごとに前計算（32×32）し `h^T M h` で全 token/layer
厳密に計算する。PPL は corpus 2048 token を単一 forward で teacher forcing。

INT2 proxy は実レシピに合わせ、Gate/Up weight を W32 BF16 scale 付き 4-level
codebook に量子化したもの（`int2` = 対称固定 codebook、`int2l` = weight から
重み付き Lloyd で学習した codebook）。Down は exact norm metadata のみ使用。

## 3. 確定事実

### 3.1 Baseline

```text
baseline perplexity = 7.063564   (positions 2047)
```

### 3.2 Exact Oracle（G1、channel 単位）

| keep | prune | A dPPL% | C dPPL% | T dPPL% |
| ---: | ---: | ---: | ---: | ---: |
| 0.7500 | 25.0% | +0.101 | -0.029 | -0.057 |
| 0.6250 | 37.5% | +0.298 | +0.265 | +0.260 |
| 0.5625 | 43.75% | +0.416 | +0.340 | +0.451 |
| 0.5000 | 50.0% | +0.840 | +0.688 | +0.600 |
| 0.4375 | 56.25% | +1.283 | +1.368 | +1.248 |
| 0.3750 | 62.5% | +2.095 | +2.398 | +2.245 |
| 0.3125 | 68.75% | +4.288 | +4.924 | +4.382 |

| 目標 | 最大 prune |
| --- | ---: |
| +1% 以内 | 50.0% |
| +2% 以内 | 56.25% |
| +5% 以内 | 68.75% |

### 3.3 粒度の影響（最重要）

| G | prune@+1% | prune@+2% | prune@+5% |
| ---: | ---: | ---: | ---: |
| 1 | 50.0% | 56.25% | 68.75% |
| 2 | 31.25% | 37.5% | 56.25% |
| 4 | 18.75% | 31.25% | 43.75% |
| 8 | 0% | 18.75% | 31.25% |
| 16 | 0% | 12.5% | 25.0% |
| 32 | 0% | 12.5% | 18.75% |
| 64 | 0% | 0% | 18.75% |
| 128 | 0% | 0% | 12.5% |
| 256 | 0% | 0% | 12.5% |

削減可能な mass は **channel 単位にのみ**存在する。G32（W32 / Down K-block）は
+2% 以内で 12.5% が上限。重要度順 permutation でも G1 には届かない。
`C` は group 化しても true energy と recall≈0.99 で、律速は ranking ではなく粒度。

### 3.4 INT2 direct final mask（NO-GO）

G1、INT2 proxy を直接 final mask に使う（candidate なし）。

| final keep | int2 対称 dPPL% | int2 学習 dPPL% | exact oracle dPPL% |
| ---: | ---: | ---: | ---: |
| 0.5000 | +6.382 | +4.035 | +0.744 |
| 0.4375 | +8.625 | +4.684 | +1.458 |

exact oracle（0.5 で +0.74%）に対し大きく劣化。INT2 単独 final decision は NO-GO。
ただし candidate + exact rerank に変えると大きく回復する（§5）。

## 4. Gate 1: INT2 candidate containment

final_keep = 0.50、score C、G1、sampled 8 layer × 64 position。指標:

- `containment` = |exact top-final ∩ proxy top-candidate| / final
- `weighted` = candidate 内に残った exact score 総量 / exact final の score 総量
- `true_weighted` = 同じを true group energy `T` で計算

| candidate keep | int2 cont mean/p05/min | int2 weighted | int2l cont mean/p05/min | int2l weighted |
| ---: | --- | ---: | --- | ---: |
| 0.5625 | 0.782 / 0.717 / 0.673 | 0.948 | 0.832 / 0.786 / 0.727 | 0.967 |
| 0.6250 | 0.827 / 0.767 / 0.724 | 0.959 | 0.873 / 0.829 / 0.777 | 0.976 |
| 0.6875 | 0.866 / 0.813 / 0.771 | 0.969 | 0.906 / 0.869 / 0.825 | 0.982 |
| 0.7500 | 0.901 / 0.857 / 0.820 | 0.977 | 0.933 / 0.903 / 0.866 | 0.987 |
| 0.8125 | 0.931 / 0.898 / 0.870 | 0.984 | 0.955 / 0.934 / 0.907 | 0.992 |
| 0.8750 | 0.958 / 0.937 / 0.918 | 0.990 | 0.974 / 0.960 / 0.940 | 0.995 |

score A / B / C の差は小さい（weighted は C が最良）。G32 では containment が
さらに高い（int2l ck=0.6875 で mean 0.954）が、これは group 粒度の劣化とは無関係。

per-layer（int2l、fk=0.5、ck=0.6875）:

| layer | mean | p05 | min |
| ---: | ---: | ---: | ---: |
| 0 | 0.928 | 0.918 | 0.908 |
| 24 | 0.898 | 0.884 | 0.858 |
| 40 | 0.877 | 0.852 | 0.825 |
| 56 | 0.916 | 0.906 | 0.858 |

**Gate 1 判定: GO（条件付き）**

- 素の containment は ck=0.75 でも 0.93 だが、weighted containment は 0.987。
  落ちた channel は exact 寄与の小さい channel であり、PPL が回復する（§5）。
- worst layer（L40）でも mean 0.877 / min 0.825 で、層単位の崩壊はない。
- 学習 codebook は対称固定 codebook より一貫して良い（containment +3〜5pp）。

## 5. Gate 2: candidate + exact rerank の PPL

candidate 内だけの exact 値で rerank し final を決める。G1、score C。

final_keep = 0.50（exact oracle +0.744%）:

| candidate keep | int2 対称 dPPL% | int2 学習 dPPL% |
| ---: | ---: | ---: |
| direct（candidate なし） | +6.382 | +4.035 |
| 0.5625 | +4.612 | +3.136 |
| 0.6250 | +3.810 | +2.455 |
| 0.6875 | +2.156 | **+1.752** |
| 0.7500 | **+1.914** | **+1.299** |
| 0.8125 | +1.335 | +0.981 |
| 0.8750 | +1.349 | +0.914 |

final_keep = 0.4375（exact oracle +1.458%）:

| candidate keep | int2 対称 dPPL% | int2 学習 dPPL% |
| ---: | ---: | ---: |
| direct | +8.625 | +4.684 |
| 0.6250 | +3.444 | +2.415 |
| 0.6875 | +2.738 | +2.432 |
| 0.7500 | +3.086 | **+1.914** |
| 0.8125 | +1.594 | +1.337 |

**Gate 2 判定: GO**

- `candidate ≤ 0.75, final = 0.50, PPL ≤ +2%` が成立する。
  - 学習 INT2: candidate 0.6875 で **+1.75%**、0.75 で +1.30%。
  - 対称 INT2: candidate 0.75 で +1.91%。
- 理想（+1% 前後）は candidate 0.8125（+0.98%）だが、これは candidate>0.75 のため
  performance headroom の観点で HOLD。candidate 0.75 でも +1.30% まで戻る。
- exact rerank により、direct INT2（+4.0〜+6.4%）から大幅に回復する。
  INT2 は final selector ではなく candidate generator として機能する。

## 6. Break-even（実フォーマット byte 数）

`breakeven.py` が PSQ manifest の実際の `codes` / `metadata1` shape から算出。
padding 0。

実 FFN byte 数（64 layer 合計）:

| tensor | bytes | bpw |
| --- | ---: | ---: |
| ffn_gate (psq4) | 3.209 GB | 4.500 |
| ffn_up (psq4) | 3.209 GB | 4.500 |
| ffn_down (psq4×48 + psq8×16) | 3.922 GB | 5.500 |
| **dense FFN** | **10.339 GB** | — |

INT2 proxy（2bit code + BF16 scale/W32、gate+up）= 3.565 GB。

```text
B_two_stage = B_int2_gateup + rc·B_gateup + rf·B_down
```

| rc | rf | bytes | saving | (PSQ8 exact なら) |
| ---: | ---: | ---: | ---: | ---: |
| 0.8750 | 0.5000 | 11.141 | **-7.76%** | +5.39% |
| 0.8125 | 0.5000 | 10.740 | **-3.88%** | +9.56% |
| 0.7500 | 0.5000 | 10.339 | **+0.00%** | +13.73% |
| 0.6875 | 0.5000 | 9.938 | +3.88% | +17.89% |
| 0.6250 | 0.5000 | 9.537 | +7.76% | +22.06% |
| 0.5625 | 0.5000 | 9.136 | +11.64% | +26.23% |
| 0.5625 | 0.4375 | 8.891 | +14.01% | +28.31% |

重要な帰結:

- 現行 PSQ4/5.5 exact では rc=0.75 が **byte 中立（saving 0%）**。rc≤0.6875 で
  はじめて正になるが +3.9% にとどまる。
- Gate の ±15% を満たすには rc≤0.5625（現行）または rc≤0.75（PSQ8 exact）が必要。
- INT2 proxy は全 channel を毎 token 読むため 2.5bpw の固定 overhead を持つ。
  exact 側が 4.5bpw では利得が薄い。**PSQ8 exact（8.5bpw）を仮定して初めて
  headroom が出る。**

### 6.1 Addendum: main `da50f61d`（down を全 PSQ4 へ統一）

main で down 系が全 layer PSQ4 になった場合、dense FFN は
`gate 3.209 + up 3.209 + down 3.209 = 9.626GB`（全 4.5bpw）。INT2 gate+up は
3.565GB のままなので

```text
saving(rc, rf) = 1 - (3.565 + rc·6.418 + rf·3.209) / 9.626
```

| rc | rf | saving |
| ---: | ---: | ---: |
| 0.7500 | 0.5000 | -3.7% |
| 0.6875 | 0.5000 | +0.5% |
| 0.6250 | 0.5000 | +4.6% |
| 0.5625 | 0.5000 | +8.8% |
| 0.5000 | 0.5000 | +13.0% |

PPL 制約を満たす rc=0.6875 では **+0.5%（ほぼ中立）**。rc=0.5 は final と同値で
二段構成の意味がない。したがって exact を PSQ4 に統一した場合、
INT2 two-stage の theoretical traffic 利得は消失する。

## 7. Latency model

```text
T_two_stage = T_int2_proxy + T_candidate_select + T_exact_gate_cand
            + T_exact_up_cand + T_swiglu_cand + T_exact_rerank
            + T_final_select + T_sparse_down + T_bookkeeping
T_dense_ffn = T_gate_dense + T_up_dense + T_swiglu_dense + T_down_dense
```

decode M=1 では memory-bound なので、traffic 削減率がほぼそのまま latency 比に
効く。したがって §6 の byte saving が第一近似となる。
exact rerank / select の bookkeeping は実装後に実測する。

## 8. 判定

### 8.1 Gate 結果

| Gate | 結果 | 根拠 |
| --- | --- | --- |
| Exact Oracle (G1) | 成立 | 50% prune で +0.74%、56.25% で +1.46% |
| Group 粒度 (G32) | 不成立 | +2% 以内は 12.5% が上限 |
| INT2 direct final mask | NO-GO | final 0.5 で +4.0〜+6.4% |
| Gate 1 containment | **GO（条件付き）** | ck=0.75 で weighted 0.987、worst layer 崩壊なし |
| Gate 2 two-stage PPL | **GO** | ck=0.6875/0.75, final 0.5 で +1.3〜+1.8% |
| Break-even (PSQ8 exact) | **GO** | ck=0.6875 で traffic saving +17.9% |
| Break-even (実 PSQ4/5.5 exact) | HOLD〜NO-GO | ck=0.6875 で +3.9%、ck=0.75 で 0% |

### 8.2 結論

- **INT2 は final selector としては NO-GO だが、candidate generator としては GO。**
  candidate を 0.6875〜0.75 に絞り、candidate 内 exact rerank を行うことで、
  `final = 0.50, PPL ≤ +2%` を満たせる。
- **break-even は exact 側の bpw に支配される。** INT2 proxy は全 channel を毎 token
  読む 2.5bpw の固定 overhead を持つため、exact が PSQ4（4.5bpw）では利得が
  出ない。exact を PSQ8（8.5bpw）で実行する前提でのみ、candidate 0.6875 で
  +17.9% の traffic saving が得られ、GO 条件（≥15%）を満たす。
- native kernel へ進む判断は、**exact FFN を PSQ8 で実行する** ことを前提とする。
  現行 PSQ4 のままでは theoretical break-even が gate を満たさない。

### 8.3 次の工程（Gate 通過時）

```text
1. Sparse Down microbench (M=1, PSQ8 Down, G1/G2/G4, keep 0.75/0.625/0.5/0.4375)
   - actual weight bytes / latency / effective GB/s / WMMA / VALU / VGPR / occupancy
   - zero masking + dense GEMM は禁止。weight load と compute を本当に消す。
2. selected Gate/Up kernel (candidate 0.625/0.6875/0.75)
3. INT2 controller kernel (candidate index 生成まで)
4. exact rerank kernel
5. Full two-stage E2E (PPL / tok/s / FFN GPU time)
```

E2E の tok/s 改善を確認するまで INT2 controller を production 化しない。
現行 PSQ4 exact のまま進める場合は、まず exact bpw を上げるか、proxy 側を
2.5bpw 未満（scale 削減 / 固定 codebook）にする追加検討が必要。

### 8.4 実測 latency PoC（追記）

`docs/rnd/ffn/ffn_psq4_vs_int2_psq8.md` で PSQ4/PSQ8 の decode GEMV を実測した。
per-layer FFN は dense_current 273.6us に対し、INT2→PSQ8 (c=0.6875,f=0.5) は
INT2 無料でも 288.5us（-5.2%）、INT2 推定込みで 384.6us（-28.9%）。
**PSQ4 据え置きの方が速い。** また FFN 全 PSQ8 は payload を +41% し、
bpw 中立化は不可能（非 FFN を 0.58bpw にする必要）。

一方、exact Gate/Up + down-only pruning は f=0.5 で +22.6%、f=0.4375 で +26.2%
となり、proxy も PSQ8 も不要。ただし G1 K-skip の GPU 実現性が前提。

## 9. 再現コマンド

```text
W=.worktrees/ffn-prune
R=/opt/zen/wk/PhaseNonShift
cd $W
# Exact Oracle / group / granularity
CUDA_VISIBLE_DEVICES=0,1,2 python3 tools/rnd/ffn_prune/run_oracle.py \
    --group-sizes 1,2,4,8,16,32 --scores A,B,C,T \
    --ratios 0.875,0.75,0.625,0.5,0.4375,0.375,0.3125,0.25 --max-tokens 2048 \
    --out $R/artifacts/ffn_prune/oracle_primary.json
# Gate 1 / Gate 2
CUDA_VISIBLE_DEVICES=0,1,2 python3 tools/rnd/ffn_prune/run_candidate.py \
    --group-sizes 1,32 --proxies int2,int2l --scores A,B,C \
    --final-keeps 0.5,0.4375 --candidate-keeps 0.5625,0.625,0.6875,0.75,0.8125,0.875 \
    --out $R/artifacts/ffn_prune/candidate_gate.json
python3 tools/rnd/ffn_prune/breakeven.py
```

raw JSON は `artifacts/ffn_prune/`（git 管理外）。
