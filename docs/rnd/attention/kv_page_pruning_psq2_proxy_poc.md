> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# PSQ2 K-Cache Proxy Selector PoC — Gate 2.75 レポート

> 注記: PSQ2 KV の実装（shape table・専用 kernel・selector・trainer）は
> リポジトリから削除済み。本記録は履歴として残す。

本ドキュメントは、BF16 min/max upper-bound selector を PSQ2 K proxy で置き換える
Offline PoC（Gate 2.75）の結果を記録する。

結論:

- **Gate 2.75: GO**
  128K context で、PSQ2 K proxy は **model-wide physical exact KV read ratio 0.607**
  （PSQ2 group-max）/ **0.613**（PSQ2 per-Q-head）を達成した。
  Oracle は 0.599 であり、**selector 品質のボトルネックは解消した**。
  BF16 min/max は group-max 0.849 / per-Q-head 0.862 であり、大きな差がある。
- 理想 total traffic（exact union + PSQ2 scan 0.082）は 128K で約 **0.69（1.45x）**。
  Gate 4 の 20% latency 目標（total <= 0.80）へ margin がある。
- Strong GO（union <= 0.60）にはわずかに届かず（0.607）。
- 重大な注意: per-Q-head の union は、budget が緩い層で physical read を大きく膨らませる
  （1/2 budget 層では per-head selected 0.53 に対し union 0.81）。
  **層ごとに per-Q-head / group-max を選択する**のが Gate 4 では有利。

---

## 1. revision / 環境

| 項目 | 値 |
|---|---|
| worktree | `.worktrees/poc-kv-psq2-proxy` |
| branch | `poc/kv-psq2-proxy` |
| baseline | `ba153347`（最新 main。Gate 2.5 は `a521a298` として統合済み） |
| GPU | R9700 (gfx1201) |
| model | `Qwen3.8-27B-PSQ` |
| full attention layer 数 | 16 |
| q_heads / kv_heads / head_dim / page_tokens | 24 / 4 / 256 / 16 |
| always-on | recent_tokens=4096, sink_pages=1, current partial page exact |
| query row 数 | 32（最終 prefill chunk） |
| corpus | Gate 1–2.5 と同一（約204K token） |

PSQ2 format は既存実装を再利用した。

```text
2bit code × 32
BF16 scale / W32
4bit shape ID / W32
= 2.625 bpw
kShapeTables.k のみ使用（V shape table は使用しない）
```

---

## 2. Host PSQ2 K encoder 一致検証

`kv_append.hip` の `op_kv_append_psq2::encode_block` と同一アルゴリズムを
host reference として実装した。

- 16 shape 全探索
- nearest 4-level code
- scale least-squares refinement
- BF16 round
- scale 確定後の code 再選択
- 2bit packing / 4bit shape ID

GPU `launch_kv_append_psq2_shape` と比較した結果:

```text
code_mismatch    = 0
scale_mismatch   = 0
shape_mismatch   = 0
max_recon_diff   = 0.000e+00
```

`phaseshift-bench paged-prune --validate-encoder` として実装し、
required test `test_psq2_proxy_encoder`（gpu1;required）に登録した。

---

## 3. Policy 定義

| policy | selector |
|---|---|
| `oracle` | exact BF16 QK の page logsumexp（理想） |
| `per_q_head` | BF16 min/max upper bound、Q head 単位 |
| `group_max` | BF16 min/max upper bound、KV head group の最大 |
| `psq2_per_q_head` | PSQ2 K proxy の QK logsumexp、Q head 単位 |
| `psq2_group_max` | PSQ2 K proxy の QK logsumexp、KV head group の最大 |

attention 本計算は全 policy で **exact BF16 K/V** を使用する。
PSQ2 proxy は page ranking のみに使用する。

---

## 4. safe budget（mean rel_l2 <= 1e-2）

各 layer / context で閾値を満たす最小 old-page 保持率。
`DENSE` は 1/2 でも閾値を満たさない。

### 64K

| layer | oracle | per_q_head | group_max | psq2_per_q_head | psq2_group_max |
|----:|----:|----:|----:|----:|----:|
| 0 | DENSE | DENSE | DENSE | DENSE | DENSE |
| 1 | DENSE | DENSE | DENSE | DENSE | DENSE |
| 2 | 1/2 | DENSE | DENSE | 1/2 | DENSE |
| 3 | 1/2 | DENSE | DENSE | 1/2 | DENSE |
| 4 | 1/2 | DENSE | DENSE | 1/2 | DENSE |
| 5 | 1/2 | DENSE | DENSE | 1/2 | DENSE |
| 6 | 1/8 | 1/4 | 1/4 | 1/8 | 1/4 |
| 7 | 1/8 | 1/4 | 1/4 | 1/8 | 1/4 |
| 8 | 1/8 | 1/4 | 1/4 | 1/8 | 1/8 |
| 9 | 1/16 | 1/4 | 1/2 | 1/16 | 1/8 |
| 10 | 1/8 | 1/2 | 1/2 | 1/8 | 1/8 |
| 11 | 1/8 | 1/2 | DENSE | 1/8 | 1/4 |
| 12 | 1/8 | 1/2 | DENSE | 1/8 | 1/4 |
| 13 | 1/4 | DENSE | DENSE | 1/4 | 1/2 |
| 14 | 1/4 | 1/2 | 1/2 | 1/4 | 1/4 |
| 15 | 1/4 | 1/2 | 1/2 | 1/4 | 1/4 |

### 128K

| layer | oracle | per_q_head | group_max | psq2_per_q_head | psq2_group_max |
|----:|----:|----:|----:|----:|----:|
| 0 | DENSE | DENSE | DENSE | DENSE | DENSE |
| 1 | 1/2 | DENSE | DENSE | 1/2 | DENSE |
| 2 | 1/2 | DENSE | DENSE | 1/2 | DENSE |
| 3 | 1/2 | DENSE | DENSE | 1/2 | DENSE |
| 4 | 1/2 | DENSE | DENSE | 1/2 | DENSE |
| 5 | 1/2 | DENSE | DENSE | 1/2 | DENSE |
| 6 | 1/4 | 1/2 | 1/2 | 1/4 | 1/4 |
| 7 | 1/4 | 1/2 | 1/2 | 1/4 | 1/4 |
| 8 | 1/4 | 1/2 | 1/2 | 1/4 | 1/4 |
| 9 | 1/4 | 1/2 | 1/2 | 1/4 | 1/4 |
| 10 | 1/4 | 1/2 | DENSE | 1/4 | 1/4 |
| 11 | 1/4 | DENSE | DENSE | 1/4 | 1/2 |
| 12 | 1/4 | DENSE | DENSE | 1/4 | 1/2 |
| 13 | 1/2 | DENSE | DENSE | 1/2 | 1/2 |
| 14 | 1/4 | 1/2 | DENSE | 1/4 | 1/2 |
| 15 | 1/8 | 1/2 | 1/2 | 1/4 | 1/4 |

### layer 数

| context | policy | DENSE | 1/2 | 1/4 | 1/8 | 1/16 |
|---|---|---:|---:|---:|---:|---:|
| 64K | oracle | 2 | 4 | 3 | 6 | 1 |
| 64K | per_q_head | 7 | 5 | 4 | 0 | 0 |
| 64K | group_max | 9 | 4 | 3 | 0 | 0 |
| 64K | psq2_per_q_head | 2 | 4 | 3 | 6 | 1 |
| 64K | psq2_group_max | 6 | 1 | 6 | 3 | 0 |
| 128K | oracle | 1 | 6 | 8 | 1 | 0 |
| 128K | per_q_head | 9 | 7 | 0 | 0 | 0 |
| 128K | group_max | 11 | 5 | 0 | 0 | 0 |
| 128K | psq2_per_q_head | 1 | 6 | 9 | 0 | 0 |
| 128K | psq2_group_max | 6 | 4 | 6 | 0 | 0 |

`psq2_per_q_head` は Oracle とほぼ一致（64K は完全一致）。

---

## 5. model-wide physical exact KV read ratio

`union_ratio`（physical に読む exact KV page / visible page）の layer 平均。

| context | oracle | per_q_head | group_max | psq2_per_q_head | psq2_group_max |
|---|---:|---:|---:|---:|---:|
| 64K | 0.531 | 0.751 | 0.751 | **0.534** | 0.554 |
| 128K | 0.599 | 0.862 | 0.849 | 0.613 | **0.607** |

### 理想 total traffic（selector scan 込み）

PSQ2 K scan = 2.625/32 = 0.082、BF16 min/max scan = 0.0625（KV 比）。

| context | oracle | per_q_head | group_max | psq2_per_q_head | psq2_group_max |
|---|---:|---:|---:|---:|---:|
| 64K total | 0.531 | 0.814 | 0.814 | 0.616 | 0.636 |
| 64K speedup | 1.88x | 1.23x | 1.23x | 1.62x | 1.57x |
| 128K total | 0.599 | 0.924 | 0.911 | 0.695 | 0.689 |
| 128K speedup | 1.67x | 1.08x | 1.10x | 1.44x | 1.45x |

これは selector 演算・Top-N・index 処理を含まない理想値である。
実際は selector kernel overhead が加わる。

---

## 6. 128K 詳細（safe budget 時）

| layer | policy | budget | union | qk | mass_min | rel_p95 | rel_p99 | cos_min | active_mean | oracle recall |
|----:|:--|:--|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | oracle | 1/2 | 0.811 | 0.636 | 0.896 | 2.61e-2 | 4.31e-2 | 0.99630 | 3.82 | 1.000 |
| 1 | psq2_per_q_head | 1/2 | 0.812 | 0.635 | 0.894 | 2.67e-2 | 4.38e-2 | 0.99642 | 3.81 | 0.950 |
| 4 | oracle | 1/2 | 0.822 | 0.628 | 0.941 | 1.86e-2 | 2.57e-2 | 0.99957 | 3.77 | 1.000 |
| 4 | psq2_per_q_head | 1/2 | 0.824 | 0.626 | 0.937 | 1.93e-2 | 2.67e-2 | 0.99953 | 3.76 | 0.950 |
| 6 | oracle | 1/4 | 0.487 | 0.563 | 0.917 | 2.32e-2 | 3.87e-2 | 0.99892 | 3.38 | 1.000 |
| 6 | group_max | 1/2 | 0.516 | 1.000 | 0.893 | 2.44e-2 | 4.41e-2 | 0.99781 | 6.00 | 0.719 |
| 6 | psq2_per_q_head | 1/4 | 0.493 | 0.556 | 0.912 | 2.40e-2 | 4.08e-2 | 0.99877 | 3.33 | 0.924 |
| 6 | psq2_group_max | 1/4 | 0.274 | 1.000 | 0.863 | 3.77e-2 | 6.43e-2 | 0.99682 | 6.00 | 0.723 |
| 7 | psq2_per_q_head | 1/4 | 0.473 | 0.579 | 0.874 | 1.99e-2 | 3.43e-2 | 0.99907 | 3.47 | 0.920 |
| 7 | psq2_group_max | 1/4 | 0.274 | 1.000 | 0.657 | 3.63e-2 | 5.79e-2 | 0.99636 | 6.00 | 0.746 |
| 10 | psq2_per_q_head | 1/4 | 0.420 | 0.652 | 0.870 | 2.68e-2 | 4.06e-2 | 0.99688 | 3.91 | 0.924 |
| 10 | psq2_group_max | 1/4 | 0.274 | 1.000 | 0.813 | 3.79e-2 | 6.29e-2 | 0.99738 | 6.00 | 0.795 |
| 15 | oracle | 1/8 | 0.267 | 0.573 | 0.724 | 2.34e-2 | 3.99e-2 | 0.99819 | 3.44 | 1.000 |

- `union` = physical exact KV page 比
- `qk` = union page のうち実際に QK 計算する (head, page) pair 比
- `active_mean` = union page を選択する平均 Q head 数（q_per_kv=6）
- `recall` = Oracle top-N page との一致率

nonfinite は全 1408 測定行で **0**。

---

## 7. 考察

### 7.1 selector gap の解消

BF16 min/max は model-wide 0.85（128K）で、Oracle 0.60 に対し selector 品質が
支配的なボトルネックだった。PSQ2 K proxy は 0.607 まで改善し、Oracle にほぼ一致した。
これは「K を高精度に再現する」ではなく「重要 page を candidate に残す」という
proxy の目的に PSQ2 が適していることを示す。

### 7.2 per-Q-head union の膨張

per-Q-head 選択は各 head が独立に page を選ぶため、union は per-head 平均より大きい。
特に budget が緩い層（1/2）では union 0.81 対 per-head sel 0.53 と 1.5 倍に膨らむ。
一方 `psq2_group_max` は 6 head 共有のため union = selected list となり、
1/2 層でも 0.516 に収まる。

したがって Gate 4 では **層ごとに policy を選ぶ**のが良い。例:

- layer 6–10: `psq2_group_max` KEEP_1_4（union 0.274、ただし qk 1.0）
- layer 11–14: `psq2_per_q_head` KEEP_1_4（union ~0.43、qk ~0.64）
- layer 1–5: `psq2_per_q_head` KEEP_1_2（union ~0.81）

単純な層別最良選択では model-wide 約 0.53 まで下がる見込みで、Oracle 0.60 を下回る。
これは per-Q-head の広い union でも、group_max が不得意な層を補うためである。

### 7.3 qk compute ratio

`psq2_per_q_head` は union が広い代わりに qk 計算量は 0.56〜0.66 に抑えられる。
`psq2_group_max` は union が狭いが qk 1.0。メモリ律速か compute 律速かで
優劣が変わるため、Gate 3/4 の実測で決める。

### 7.4 PSQ2 K proxy の shape table

既存 `psq2_kv_shapes_generated.h` の K table をそのまま使用した。
ranking-aware 再学習は本 Gate では行わない（§25 の方針）。

---

## 8. Gate 2.75 判定

| 条件（128K primary） | 結果 |
|---|---|
| model-wide union ratio <= 0.65 | psq2_group_max 0.607 / psq2_per_q_head 0.613 → PASS |
| mean rel_l2 <= 1e-2 | PASS |
| nonfinite = 0 | PASS |

- Strong GO（<= 0.60）: **該当せず**（0.607）
- **GO（<= 0.65）: 該当**
- Conditional GO: 不要

**Gate 2.75: GO。** Gate 3（GPU PSQ2 K proxy selector）へ進む。

---

## 9. 未実施 / 次段階

- Gate 3: GPU selector kernel（score / Top-N / union mask）、性能計測
- PSQ4 control（§24）: PSQ2 が GO のため今回は不要
- PSQ2 shape table の ranking-aware 再学習: 別 PoC
- Gate 4: dual-store runtime、sparse exact BF16 decode、実 model KLD、performance

本 PoC の host encoder / analyzer は R&D であり、
production へは Gate 3/4 で GPU kernel・runtime として再設計する。
