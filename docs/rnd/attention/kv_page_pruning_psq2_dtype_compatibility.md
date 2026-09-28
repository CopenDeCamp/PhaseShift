> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# PSQ2 Sparse Attention — KV DType Compatibility PoC（Gate 4Q-A）レポート

> 注記: PSQ2 KV の実装（shape table・専用 kernel・selector・trainer）は
> リポジトリから削除済み。本記録は履歴として残す。

本ドキュメントは、PSQ2 page selector が authoritative KV dtype を跨いで
成立するかを Offline で検証した Gate 4Q-A の結果を記録する。

結論:

- **Gate 4Q-A: PASS（4Q-B へ進む）**。
- PSQ2 selector の page 選択品質は authoritative KV dtype に実質依存しない。
  BF16 / FP8 / PSQ4 で **safe budget・union ratio・selector gap がほぼ一致**した。
- 誤差の支配項は dtype ごとに異なる。FP8 / PSQ4 では **KV quantization error が支配的**で、
  pruning error と selector gap は BF16 と同水準である。
- したがって「Sparse Attention は BF16 でしか効かない」という懸念は、
  **quality 面では否定された**。残る論点は performance（Gate 4Q-C）のみ。

---

## 1. 目的

検証する問い:

```text
PSQ2 K proxy selector は、
authoritative KV が FP8 / PSQ4 でも
重要 page を正しく残せるか。
```

Shape v0 は freeze 継続。selector algorithm も変更しない。
dtype 別 shape も導入しない。BF16 / FP8 / PSQ4 で同一の PSQ2 K proxy を使う。

---

## 2. revision / 環境

| 項目 | 値 |
|---|---|
| worktree | `.worktrees/poc-kv-psq2-proxy` |
| branch | `poc/kv-psq2-proxy` |
| GPU | R9700 (gfx1201)（本 Gate の analyzer は CPU offline reference） |
| probe | `/tmp/opencode/kv_page_prune/all_128k/layer_{0..15}`（BF16 dump） |
| context | 131072（128K） |
| full attention layer | 16 |
| q_heads / kv_heads / head_dim / page_tokens | 24 / 4 / 256 / 16 |
| always-on | recent_tokens=4096, sink_pages=1, current partial page exact |
| query row | 8（可視長 131072 の prefill chunk から等間隔抽出） |
| enumerator | `phaseshift-bench paged-prune --target-kv-dtype {bf16,fp8,psq4}` |

analyzer は host reference として実装した。R0 / R1 / R2 を GPU kernel の数値に
依存させず、quantization error と pruning error を分離するためである。

---

## 3. 誤差の定義

| 記号 | 構成 | 意味 |
|---|---|---|
| R0 | BF16 dense | model-level absolute reference |
| R1 | target dtype dense | KV quantization 単独の影響 |
| R2 | target dtype + target-dtype Oracle pruning | その dtype で達成可能な pruning 上限 |
| R3 | PSQ2 selector → target dtype sparse | production 候補 |

| 差分 | 意味 |
|---|---|
| `quant = R1 - R0` | FP8 / PSQ4 KV そのものの誤差 |
| `oracle_prune = R2 - R1` | その dtype で Sparse 化すること自体の誤差 |
| `selector_gap = R3 - R2` | PSQ2 proxy で page を選んだことで増えた誤差（今回の最重要値） |
| `total = R3 - R0` | production 構成全体の誤差 |

`rel_l2` は mean、`selector_gap` は R3 と R2 の出力 rel_l2。
R2 の page importance は **target dtype の K を decode した値**から計算した
（BF16 Oracle を流用しない）。

---

## 4. dtype の encode / decode semantics

二重量子化を避けるため、parallel generation とした。

```text
BF16 K ──┬──> authoritative KV encoder (BF16 / FP8 / PSQ4)
         └──> PSQ2 K proxy encoder          (常に BF16 K から)
```

| dtype | layout | encoder（既存実装を正とする） |
|---|---|---|
| BF16 | 16 bit/value | そのまま |
| FP8_E4M3 | 8 bit/value | per (token, head) scale = amax/448、`e4m3_encode_u8` |
| PSQ4_W32 | 4 bit code + BF16 W32 scale | per 32-block、`psq4_lsq1_encode_block`（Lsq1）、`cb10_code_to_f32` |

PSQ2 K proxy は全 dtype で **BF16 K から生成**し、2.625 bpw（K のみ）を維持する。

---

## 5. 必須 Table 1 — target-dtype safe budget（mean rel_l2 <= 1e-2）

各 layer / dtype / policy について、閾値を満たす最小 old-page 保持率。
`DENSE` は 1/2 でも閾値を満たさない。

| layer | bf16 oracle | fp8 oracle | psq4 oracle | bf16 gm | fp8 gm | psq4 gm | bf16 pqh | fp8 pqh | psq4 pqh |
| ----: | ----------: | ---------: | ----------: | ------: | -----: | ------: | -------: | ------: | -------: |
|     0 |       DENSE |      DENSE |       DENSE |   DENSE |  DENSE |   DENSE |    DENSE |   DENSE |    DENSE |
|     1 |         1/2 |        1/2 |         1/2 |   DENSE |  DENSE |   DENSE |      1/2 |     1/2 |      1/2 |
|     2 |         1/2 |        1/2 |         1/2 |   DENSE |  DENSE |   DENSE |      1/2 |     1/2 |      1/2 |
|     3 |         1/2 |        1/2 |         1/2 |   DENSE |  DENSE |   DENSE |      1/2 |     1/2 |      1/2 |
|     4 |         1/2 |        1/2 |         1/2 |   DENSE |  DENSE |   DENSE |      1/2 |     1/2 |      1/2 |
|     5 |         1/2 |        1/2 |         1/2 |   DENSE |  DENSE |   DENSE |      1/2 |     1/2 |      1/2 |
|     6 |         1/4 |        1/4 |         1/4 |     1/2 |    1/2 |     1/2 |      1/4 |     1/4 |      1/4 |
|     7 |         1/4 |        1/4 |         1/4 |     1/4 |    1/4 |     1/2 |      1/4 |     1/4 |      1/4 |
|     8 |         1/4 |        1/4 |         1/4 |     1/4 |    1/4 |     1/4 |      1/4 |     1/4 |      1/4 |
|     9 |         1/4 |        1/4 |         1/4 |     1/4 |    1/4 |     1/4 |      1/4 |     1/4 |      1/4 |
|    10 |         1/4 |        1/4 |         1/4 |     1/4 |    1/4 |     1/4 |      1/4 |     1/4 |      1/4 |
|    11 |         1/4 |        1/4 |         1/4 |     1/2 |    1/2 |     1/2 |      1/4 |     1/4 |      1/4 |
|    12 |         1/4 |        1/4 |         1/4 |     1/2 |    1/2 |     1/2 |      1/4 |     1/4 |      1/4 |
|    13 |         1/2 |        1/2 |         1/2 |     1/2 |    1/2 |     1/2 |      1/2 |     1/2 |      1/2 |
|    14 |         1/4 |        1/4 |         1/4 |     1/2 |    1/2 |     1/2 |      1/4 |     1/4 |      1/2 |
|    15 |         1/8 |        1/8 |         1/8 |     1/4 |    1/4 |     1/4 |      1/8 |     1/8 |      1/8 |

`gm` = `psq2_group_max`、`pqh` = `psq2_per_q_head`。

### layer 数

| dtype | policy | DENSE | 1/2 | 1/4 | 1/8 |
|---|---|---:|---:|---:|---:|
| bf16 | oracle | 1 | 6 | 8 | 1 |
| bf16 | psq2_group_max | 6 | 5 | 5 | 0 |
| bf16 | psq2_per_q_head | 1 | 6 | 8 | 1 |
| fp8 | oracle | 1 | 6 | 8 | 1 |
| fp8 | psq2_group_max | 6 | 5 | 5 | 0 |
| fp8 | psq2_per_q_head | 1 | 6 | 8 | 1 |
| psq4 | oracle | 1 | 6 | 8 | 1 |
| psq4 | psq2_group_max | 6 | 6 | 4 | 0 |
| psq4 | psq2_per_q_head | 1 | 7 | 7 | 1 |

### BF16 を基準とした dtype 差

| dtype | oracle | psq2_group_max | psq2_per_q_head |
|---|---|---|---|
| fp8 | 全 16 layer で一致 | 全 16 layer で一致 | 全 16 layer で一致 |
| psq4 | 全 16 layer で一致 | layer 7 のみ 1 段階 conservative | layer 14 のみ 1 段階 conservative |

手順書 §12 の条件「same budget、または 1 段階だけ conservative」を満たす。
PSQ4 で conservative になったのは 2 / 48（layer × policy）のみ。

---

## 6. 必須 Table 2 — 誤差分離（layer 平均）

| dtype | quant (R1-R0) | oracle_prune (R2-R1) | selector_gap (R3-R2) | total (R3-R0) |
| ----- | ------------: | -------------------: | -------------------: | ------------: |
| bf16  |     2.677e-04 |            3.659e-03 |            2.637e-03 |     5.984e-03 |
| fp8   |     1.856e-02 |            3.671e-03 |            2.647e-03 |     2.118e-02 |
| psq4  |     5.904e-02 |            3.782e-03 |            2.463e-03 |     5.934e-02 |

- `oracle_prune` と `selector_gap` は dtype によらず **3.7e-3 / 2.5e-3 前後で一致**した。
  すなわち PSQ2 page 選択は KV dtype に対して robust である。
- FP8 / PSQ4 の total を支配するのは **KV quantization error** であり、
  pruning / selector 起因ではない。これは pruning の問題ではなく KV format の特性である。
- bf16 の quant 2.677e-04 は、probe dump の dense 出力が BF16 round されていることによる
  **測定 floor** である（analyzer は float で再計算する）。FP8 / PSQ4 の quant はこれを大幅に上回る。
- 参考: 各誤差は独立ではないため二乗和の単純加算にはならない。

---

## 7. 必須 Table 3 — model-wide union ratio と selector gap（safe budget 時）

`union_ratio` = physical に読む exact KV page / visible page の layer 平均。

| dtype | policy | union | qk_ratio | page_recall | active_heads |
| ----- | ------ | ----: | -------: | ----------: | -----------: |
| bf16  | oracle | 0.5721 | 0.6301 | 1.0000 | 3.780 |
| bf16  | psq2_group_max | 0.3947 | 1.0000 | 0.8111 | 6.000 |
| bf16  | psq2_per_q_head | 0.5759 | 0.6250 | 0.9308 | 3.750 |
| fp8   | oracle | 0.5721 | 0.6300 | 1.0000 | 3.780 |
| fp8   | psq2_group_max | 0.3947 | 1.0000 | 0.8111 | 6.000 |
| fp8   | psq2_per_q_head | 0.5759 | 0.6250 | 0.9305 | 3.750 |
| psq4  | oracle | 0.5723 | 0.6297 | 1.0000 | 3.778 |
| psq4  | psq2_group_max | 0.4189 | 1.0000 | 0.8191 | 6.000 |
| psq4  | psq2_per_q_head | 0.5938 | 0.6320 | 0.9304 | 3.792 |

target-dtype Oracle の union は 0.572 で完全一致した。
PSQ2 側も FP8 は BF16 と一致、PSQ4 はごく僅かに広い（group_max +0.024、per_q_head +0.018）。

selector gap:

| dtype | psq2_group_max | psq2_per_q_head |
| ----- | -------------: | --------------: |
| bf16  |      2.637e-03 |       6.337e-04 |
| fp8   |      2.647e-03 |       6.373e-04 |
| psq4  |      2.463e-03 |       5.904e-04 |

dtype 間の差は 1 割未満であり、selector 品質は dtype 不変とみなせる。

---

## 8. 128K 詳細（safe budget 時の抜粋）

| layer | dtype | policy | budget | union | qk | mass_min | rel_p95 | rel_p99 | cos_min | recall | gap |
| ----: | ----- | ------ | -----: | ----: | --: | -------: | ------: | ------: | ------: | -----: | --: |
| 6 | bf16 | oracle | 1/4 | 0.4932 | 0.5557 | 0.991 | 2.50e-02 | 3.24e-02 | 0.99998 | 1.000 | 0 |
| 6 | psq4 | oracle | 1/4 | 0.4934 | 0.5555 | 0.991 | 2.58e-02 | 3.30e-02 | 0.99997 | 1.000 | 0 |
| 6 | bf16 | psq2_group_max | 1/2 | 0.5158 | 1.0000 | 0.995 | 1.33e-02 | 2.29e-02 | 0.99999 | 0.805 | 1.75e-03 |
| 6 | psq4 | psq2_group_max | 1/2 | 0.5158 | 1.0000 | 0.995 | 1.37e-02 | 2.43e-02 | 0.99999 | 0.804 | 1.85e-03 |
| 7 | bf16 | psq2_group_max | 1/4 | 0.2737 | 1.0000 | 0.984 | 3.37e-02 | 5.94e-02 | 0.99993 | 0.745 | 4.13e-03 |
| 7 | psq4 | psq2_group_max | 1/2 | 0.5158 | 1.0000 | 0.995 | 1.10e-02 | 2.25e-02 | 0.99999 | 0.831 | 1.50e-03 |
| 7 | bf16 | psq2_per_q_head | 1/4 | 0.4721 | 0.5808 | 0.991 | 2.10e-02 | 3.26e-02 | 0.99998 | 0.920 | 3.76e-04 |
| 7 | psq4 | psq2_per_q_head | 1/4 | 0.4721 | 0.5808 | 0.990 | 2.22e-02 | 3.40e-02 | 0.99998 | 0.917 | 4.18e-04 |
| 15 | bf16 | oracle | 1/8 | 0.2629 | 0.5822 | 0.938 | 2.02e-02 | 2.48e-02 | 0.99994 | 1.000 | 0 |
| 15 | psq4 | oracle | 1/8 | 0.2633 | 0.5814 | 0.936 | 2.05e-02 | 2.55e-02 | 0.99994 | 1.000 | 0 |

`nonfinite` は全 720 測定行で **0**。

---

## 9. 判定

| 条件（128K primary） | 結果 |
|---|---|
| target-dtype Oracle が dtype 間で同等 | BF16 0.5721 / FP8 0.5721 / PSQ4 0.5723 → PASS |
| PSQ2 safe budget が dtype 間で同等 | FP8 完全一致、PSQ4 は 2/48 で 1 段階 conservative → PASS |
| selector gap が dtype 間で同等 | 差 < 1 割 → PASS |
| mean rel_l2 <= 1e-2（safe budget 時） | 全 dtype で PASS |
| nonfinite = 0 | PASS |

**Gate 4Q-A: PASS。** PSQ2 selector は authoritative KV dtype を跨いで成立する。
FP8 / PSQ4 の sparse kernel（Gate 4Q-B）へ進む。

最終判定は model-level logit KLD（Gate 4Q-D）を優先する。
本 Gate は page 選択互換性のみを対象とする。

---

## 10. analyzer 実装（本 Gate で追加）

`phaseshift-bench paged-prune` に `--target-kv-dtype auto|bf16|fp8|psq4` を追加。

- BF16 dump から target dtype を offline 生成（FP8 / PSQ4 の semantics は既存 host reference を使用）。
- PSQ2 K proxy は常に BF16 K から生成。
- 追加 column: `dense_q_ref_rel_l2`（R1-R0）、`rel_l2_ref_mean`（R3-R0）、`oracle_gap_mean`（R3-R2）。

回帰: BF16 経路は旧 baseline と bit-identical
（layer 6, rows=32: oracle 0.5 → rel_l2 1.522471e-03 / union 0.774376、psq2_group_max 0.25 → 9.918043e-03 / 0.273673）。

---

## 11. Gate 4Q-B — sparse FP8 / PSQ4 kernel

### 11.1 実装

`attention_paged_sparse_split_bf16_impl` と同じ selected list semantics
（`packed = (logical_page << 6) | qmask`）で、FP8 / PSQ4 の sparse kernel を追加した。

- `launch_attention_paged_fp8_e4m3_sparse[_split]`
- `launch_attention_paged_psq4_sparse[_split]`

dtype 依存部は decode のみなので、compile-time の reader trait で表現した。

| reader | K/V の decode | score scale | V scale |
|---|---|---|---|
| `Fp8Reader` | `e4m3_decode_f32`（byte） | per (token, head) `k_scale` | per (token, head) `v_scale` |
| `Psq4Reader` | `cb10_code_to_f32` + BF16 W32 scale | 1（scale は decode に内包） | 1 |

selector output は dtype 非依存で、dispatch 側が BF16 / FP8 / PSQ4 kernel を選ぶ。
reduce は既存 `launch_attention_paged_bf16_split_reduce` を共用する（partials は float）。

### 11.2 all-pages / masked correctness

sparse を selected-all-pages で実行し、同 dtype の dense と比較した
（required test `test_sparse_paged_attention_dtype`）。

| dtype | 条件 | sparse (non-split) rel_l2 | sparse (split=4) rel_l2 |
|---|---|---|---|
| fp8 | all-pages | 2.984e-05 | 1.663e-06 |
| fp8 | masked (active heads) | 1.433e-05 | 1.764e-06 |
| psq4 | all-pages | 6.596e-06 | 1.057e-04 |
| psq4 | masked (active heads) | 6.991e-06 | 1.121e-04 |

`nonfinite = 0`。閾値 1e-2 に対し十分小さい。
sparse 化に伴う numerics の変更はない（既存 dense と同一の decode / accumulation 精度）。

---

## 12. Gate 4Q-C — dtype 別 performance

### 12.1 条件

`phaseshift-bench psq2-selector`、policy `group_max`、frac 0.25、layout `distinct-slot`、
32K / 64K / 128K / 256K（256K は rows 1,2）。dense と sparse は **同一 dtype** で比較する。

`dense_us` = 同 dtype の dense paged attention、
`selector_us` = PSQ2 page score + select、`sparse_us` = sparse attention (+ reduce)、
`total_us = selector_us + sparse_us`。

### 12.2 必須 Table 3 — 128K

| dtype | rows | dense_us | selector_us | sparse_us | total_us | total/dense | speedup |
| ----- | ---: | -------: | ----------: | --------: | -------: | ----------: | ------: |
| bf16 | 1 |  7480.89 |  518.02 |  722.65 |  1240.67 | 16.6% | 6.03x |
| bf16 | 2 |  7357.63 |  955.51 |  722.69 |  1678.20 | 22.8% | 4.38x |
| bf16 | 4 |  8806.05 | 2347.58 | 1858.83 |  4206.41 | 47.8% | 2.09x |
| bf16 | 8 | 12791.56 | 4422.61 | 3589.76 |  8012.37 | 62.6% | 1.60x |
| fp8  | 1 | 20158.67 |  515.50 | 1747.69 |  2263.19 | 11.2% | 8.91x |
| fp8  | 2 | 20769.29 |  930.76 | 1835.52 |  2766.28 | 13.3% | 7.51x |
| fp8  | 4 | 24214.18 | 2339.23 | 3395.03 |  5734.27 | 23.7% | 4.22x |
| fp8  | 8 | 41216.43 | 4405.82 | 6431.20 | 10837.02 | 26.3% | 3.80x |
| psq4 | 1 | 12863.15 |  514.83 |  884.14 |  1398.98 | 10.9% | 9.19x |
| psq4 | 2 | 12903.62 |  925.98 |  980.69 |  1906.67 | 14.8% | 6.77x |
| psq4 | 4 | 14958.26 | 2308.85 | 2231.40 |  4540.25 | 30.4% | 3.29x |
| psq4 | 8 | 25053.42 | 4410.98 | 4175.81 |  8586.79 | 34.3% | 2.92x |

全 dtype / 全 rows で Performance GO（total <= 0.80 x dense）。

参考（total/dense）:

| ctx | bf16 r1 | bf16 r8 | fp8 r1 | fp8 r8 | psq4 r1 | psq4 r8 |
| --: | ------: | ------: | -----: | -----: | ------: | ------: |
| 32K | 19.3% | 80.9% | 12.0% | 31.1% | 15.6% | 39.4% |
| 64K | 18.0% | 71.0% | 10.5% | 28.6% | 11.2% | 37.5% |
| 128K | 16.6% | 62.6% | 11.2% | 26.3% | 10.9% | 34.3% |
| 256K | 17.6% | (未測定) | 10.3% | (未測定) | 11.4% | (未測定) |

### 12.3 必須 Table 4 — crossover

| dtype | sparse_min_context | eligible layers | final verdict |
| ----- | -----------------: | --------------- | ------------- |
| bf16  | 32K（rows 8 のみ 64K） | all（union=1 層は DENSE fallback） | GO |
| fp8   | 32K | all | GO |
| psq4  | 32K | all | GO |
| 256K（rows 1,2, bf16/fp8/psq4） | 16-27% | — | GO |

dtype ごとの crossover 差は本測定では出なかった（全 dtype 32K で既に 0.80 を下回る）。
ただし次節の注意点により、dtype 別 crossover を固定値として扱わない。

### 12.4 考察

- PSQ4 は authoritative KV が小さいが、**dense PSQ4 自体が BF16 dense より遅い**
  （128K rows=1 で 12863us vs 7481us）。PSQ4 decode の compute が支配的なためである。
  結果として sparse の相対利得はむしろ大きい（rows=8 で 2.92x）。
- **FP8 dense が最も遅い**（128K rows=1 で 20159us、BF16 の 2.7 倍）。
  既存 FP8 dense kernel の decode / occupancy 特性によるもので、sparse 側の問題ではない。
  FP8 dense が将来高速化されれば FP8 sparse の相対利得は下がるため、
  **FP8 の crossover は FP8 dense kernel 改善後に再測定する**。
- rows=8 では selector が total の 50% 前後（128K psq4: 4411/8587us）を占める。
  §36 の temporal selector reuse（refresh=2/4/8）は PSQ4 で特に有効な余地がある。
- 本 bench の K/V は合成 random であり、`rel_l2` は pruning 対象データに構造がないため
  品質指標にはならない。ここでは performance のみを評価する。

---

## 13. Gate 4Q 総合判定

| dtype | Quality (4Q-A offline / 4Q-D E2E) | Performance (4Q-C) | 判定 |
| ----- | ------------------- | ------------------ | ---- |
| bf16  | GO / GO（128K selector gap 4.2e-3） | GO | **GO** |
| fp8   | GO / GO（128K selector gap 4.6e-3。dense kernel 改善後に再確認） | GO | **GO** |
| psq4  | GO（2/48 で 1 段階 conservative）/ GO（128K selector gap 3.0e-2） | GO | **GO** |

`sparse_supported != sparse_always_faster` の原則どおり、runtime では
`policy(kv_dtype, layer, visible_tokens, active_rows)` で DENSE fallback を常に残す。
Gate 4F で decode 専用の runtime 統合を実装し、Gate 4Q-D で model-level logit KLD を測定した。

---

## 14. Gate 4Q-D（dense 部分）— KV dtype の E2E

- `phaseshift-compute --kv-cache-dtype {bf16,fp8_e4m3,psq4} --dump-logits`
- model `Qwen3.8-27B-PSQ`、greedy（temperature 0）、合成 token id prompt
- logits 行の構造: prefill は 2048 token/chunk で 1 行/chunk、その後 decode 1 行/step。
  prefill 最終 chunk の行が次の token を決める。

### 1K context（1024 prompt / 8 generated）

3 dtype で生成列が完全一致（同一 trajectory）。

| dtype | KLD mean | KLD max | top1 agree | top5 containment |
| ----- | -------: | ------: | ---------: | ---------------: |
| fp8_e4m3 | 6.429e-03 | 2.214e-02 | 1.000 | 0.950 |
| psq4 | 2.248e-02 | 1.148e-01 | 1.000 | 0.875 |

### 8K context（8192 prompt）

- prefill 後の最初の decode 行（同一入力）:

| dtype | KLD | top1 | max \|dlogit\| |
| ----- | --: | ---: | --------------: |
| fp8_e4m3 | 1.343e-03 | 一致 | 0.269 |
| psq4 | 1.978e-03 | 一致 | 0.356 |

- 8K では fp8 / psq4 とも **生成 1 トークン目で greedy が分岐**した
  （1K では分岐せず）。長 context では KV quantization が greedy の trajectory を変え得る。
- prefill 行の dtype 差は context とともに増える（8K では先頭 chunk 行が一致し最終 chunk 行のみ差、
  128K では 63/63 の prefill 行が差）。

---

## 15. Gate 4F — dtype-aware runtime sparse policy

sparse exact attention を runtime に統合した。**decode 専用**（prefill は dense のまま）。

### 制御（env）

| env | 既定 | 意味 |
| --- | ---: | --- |
| `PHASESHIFT_PSQ2_SPARSE` | 0 | sparse 経路を有効化 |
| `PHASESHIFT_PSQ2_SPARSE_MIN_TOKENS` | 65536 | これ未満の max_visible では dense |
| `PHASESHIFT_PSQ2_SPARSE_BUDGET` | 0.25 | eligible page の選択率（group_max） |
| `PHASESHIFT_PSQ2_SPARSE_RECENT` | 4096 | always-on recent tokens |
| `PHASESHIFT_PSQ2_SPARSE_SINK` | 1 | always-on sink pages |
| `PHASESHIFT_PSQ2_SPARSE_SPLITS` | 16 | sparse split 数 |
| `PHASESHIFT_PSQ2_SPARSE_TRACE` | - | 1 で dispatch 判定を stderr に出力 |

### 構成

- **K proxy**: `PagedKProxyPSQ2Storage` を executor arena に確保し、`try_launch_kv_append` で
   authoritative KV append と並んで `launch_k_append_psq2_proxy` を毎回実行する。
   proxy は常に **量子化前 BF16 K** から生成する（FP8/PSQ4 K からは生成しない）。
- **selector**: `launch_psq2_page_score` → `launch_psq2_page_select`（GROUP_MAX, FRACTION）。
- **unified list**: 新規 `launch_psq2_unify_entries` が `[0, sink)` + 選択 eligible + `[hi, pc)`
  を **page 昇順**で結合する（always-on = sink / recent / current partial page）。
- **attention**: dtype 別 sparse split（bf16/fp8/psq4）→ 既存 `launch_attention_paged_bf16_split_reduce`。
- **DENSE fallback**: 次では sparse を使わない。head_dim != 256、q_heads != kv_heads*6、
  rows > 32、max_visible < min_tokens、kv_dtype == PSQ2、または plan が非 Optimized。
- workspace（score/entries/counts/unified/partials）は execution workspace 末尾に固定配置し、
  program 依存の scratch と重ならない。

### exactness

budget = 1.0（eligible を全選択）では unified list が全 page の昇順になり、
sparse 出力は dense と **bit-exact**（`recent=0,sink=0` でも確認）。
required test `test_psq2_sparse_unify`（unify list 内容 + full-selection sparse == dense）を追加。

---

## 16. Gate 4Q-D — sparse E2E（model quality）

### 方法

- `phaseshift-compute`、prompt は合成 token id、`--max-new-tokens 2`、greedy。
- rows = ceil(L/2048) prefill 行 + 1 decode 行。比較するのは **最初の decode 行**のみ
  （prefill は dense のため入力が config 間で同一。token0 が一致する config 間でのみ比較可能）。
- config: R0 = bf16 dense、R1 = target-dtype dense、R3 = target-dtype sparse
  （budget 0.25, min 4096, recent 4096, sink 1）。GPU1。

### L = 8192

| 比較 | KLD | top1 |
| --- | ---: | --- |
| R1 fp8 vs R0 | 2.2595e-02 | 不一致 |
| R1 psq4 vs R0 | 1.6702e-02 | 不一致 |
| R3 bf16 vs R0 | 1.7629e-02 | 不一致 |
| R3 fp8 vs R1 fp8 | 8.2917e-03 | - |
| R3 psq4 vs R1 psq4 | 1.4613e-01 | - |

8K は always-on（recent 4096）が context の約半分を占め、eligible が小さいため
budget 0.25 が相対的に厳しい。psq4 の 1.46e-1 は top1 が変わる外れ値位置。

### L = 32768

| 比較 | KLD | top1 |
| --- | ---: | --- |
| R1 fp8 vs R0 | 7.5659e-03 | 一致 |
| R1 psq4 vs R0 | 2.5244e-02 | 一致 |
| R3 bf16 vs R0 | 8.8981e-03 | 一致 |
| R3 fp8 vs R1 fp8 | 1.5511e-02 | 一致 |
| R3 psq4 vs R1 psq4 | 2.1739e-02 | 一致 |

全 config で top1 一致。

### L = 131072（本 Gate の主対象）

| 比較 | KLD | top1 |
| --- | ---: | --- |
| R1 fp8 vs R0 | 3.3078e-03 | 一致 |
| R3 bf16 vs R0 | 4.2091e-03 | 一致 |
| R3 fp8 vs R1 fp8 | 4.5780e-03 | 一致 |
| R3 psq4 vs R1 psq4 | 2.9729e-02 | 一致 |

psq4 は prefill 最終 chunk 行で token0 が変わるため bf16 とは同一入力で比較できない
（psq4 内 dense vs sparse のみ比較）。全 config で top1 一致。

### まとめ

- 128K では selector gap（sparse − dense、同一 dtype）が bf16 4.2e-3 / fp8 4.6e-3 /
  psq4 3.0e-2 で、**dtype の quant error と同程度以下**。top1 も保存される。
- context が長いほど selector gap は小さい（128K < 32K < 8K）。recent 4096 が重要な直近を
  常時保持するため、設計意図どおり長 context で効く。
- 制約: 各 L で比較可能な位置は最初の decode 1 点のみ。他 config と入力が一致しない行は除外。

---

## 17. Gate 4F 補足 — DFlash2 verify 併用

DFlash2 の target verify は rows = 1+M（2〜8）で decode 扱いになるため sparse 経路に入る。
`phaseshift-compute`（drafts=7, greedy, draft=`Qwen3.8-27B-DFlash2-PSQ4`, prose）で検証した。

### policy guard

eligible が空の条件（`max_visible <= recent + sink*page_tokens`）では sparse を使わない。
prune する page が無ければ利得がゼロで、sparse split kernel と dense kernel の数値差だけが
残るためである。短 context の verify はこの guard で dense と完全一致する。

### 短 context（prose 468 prompt / 256 generated）

| 経路 | sparse | GENERATED_IDS | rounds | mean accepted | reruns |
| --- | --- | --- | ---: | ---: | ---: |
| dense | OFF | 基準 | 78 | 2.269 | 0 |
| sparse 0.25 | ON（guard で無効） | dense と完全一致 | 78 | 2.269 | 0 |

### 長 context（prose×16 = 7488 prompt / 64 generated）

| 経路 | GENERATED_IDS | rounds | mean accepted | reruns | tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| dense | 基準 | 22 | 1.864 | 0 | 53.14 |
| sparse budget=1.0 | **dense と完全一致** | 22 | 1.864 | 0 | 48.02 |
| sparse budget=0.25 | token 11 で分岐 | 20 | 2.150 | 0 | 54.31 |

### 判定

- sparse 経路は budget=1.0 で dense と bit-exact（DFlash2 全体でも token 完全一致）。
  よって verify 統合は機構的に正しい。
- budget=0.25 では pruning が近似になるため、**target-only greedy との token exact 保証は
  成立しない**（dense DFlash2 からも分岐する）。reruns=0 は維持される。
- production で verify に sparse を入れる場合は、この exactness 契約の緩和を明記する。
- この長さ（約 7.5K）では pruning は mild で、利得は小さい。利得評価はより長い context で行う。

---

## 18. teacher forcing による複数位置 KLD

Gate 4Q-D は最初の decode 1 点/L のみ比較可能だった。全 position を同一入力で比較するため、
`phaseshift-compute` に **teacher forcing** を追加した。

### 実装

- `RuntimeRequest` に `forced_tokens` / `forced_index` を追加。
- `ContinuousBatcher::step()` の sampling 点で、forced が残っていれば sampled token を上書きする。
- `ContinuousBatcher::set_forced_tokens(id, tokens)` と `Qwen35ComputeRuntime::set_forced_tokens` を追加。
- CLI `--force-tokens-file PATH`（whitespace separated ids）。指定時は `max_new_tokens` を強制列長に一致させる。

これにより全 config が同一の continuation を処理し、全 decode position の logits が比較可能になる。

### 設定

- prompt = prose×16（7488 token）、forced = bf16 dense が生成した 64 token。
- rows = 4 prefill + 63 decode。decode position 4..66 を比較。
- control: forced した bf16 dense は非 forced の dense と **maxdiff 0**（teacher forcing が正しい）。

### 結果（decode 63 position、budget 0.25）

| 成分 | KLD mean | median | max | top1 agree |
| --- | ---: | ---: | ---: | ---: |
| prune bf16（sparse vs dense） | 7.887e-02 | 2.549e-04 | 2.733 | 59/63 |
| quant psq4（psq4 dense vs bf16 dense） | 2.289e-02 | 6.252e-04 | 5.843e-01 | 61/63 |
| prune psq4（psq4 sparse vs psq4 dense） | 5.804e-02 | 9.178e-04 | 2.413 | 61/63 |
| total psq4（psq4 sparse vs bf16 dense） | 7.895e-02 | 1.288e-03 | 3.217 | 62/63 |

- median は 2.5e-4〜1.3e-3 と非常に小さく、**大半の position では分布がほとんど変わらない**。
  mean は少数の高エントロピー位置（max 2〜3）に引っ張られる。
- top1 は 59〜62/63 で一致。
- prune の median（bf16 2.5e-4）は quant psq4 の median（6.3e-4）より小さい。
- この context（約 7.5K）は recent 4096 が半分近くを占め prune は mild。より長い context での
  評価は残課題。

---

## 19. Gate 4G — prefix cache companion proxy

prefix cache の restore は KV page を直接コピーし、KV append を走らせない。そのため restored
page の K proxy が stale になり、sparse selector が誤った page を選ぶ。これを解消するため、
prefix cache に **companion K proxy** を持たせた。

### 実装

- `PagedKProxyPSQ2Storage::make_view()` と `copy_k_proxy_page(dst, dst_page, src, src_page, stream)`
  を追加。page 単位で code/scale/shape を layer ごとに D2D コピーする。
- `PrefixCache::create(..., bool with_k_proxy)` が cache 側の `cache_k_proxy_` を生成。
  page は KV と 1:1（cache KV の page id をそのまま使う。独立 allocator は持たない）。
- `save` / `restore` が active proxy の view を受け取り、KV page と並べて proxy page をコピーする。
- batcher が `executor_.psq2_sparse` から view を構築して渡す。
- compute runtime は companion 生成に成功したら sparse を許可し、失敗したら
  **dense fallback**（`ExecutorConfig::psq2_sparse_allowed`）+ 警告。
- trace: `PREFIX_CACHE_K_PROXY_BYTES` / `PREFIX_CACHE_K_PROXY_SAVE` /
  `PREFIX_CACHE_K_PROXY_RESTORE`。

### コスト

- メモリ: cache 容量 × 5.25 KB/token。8192 token の cache で **44,040,192 B（約 42 MiB）**。
  KV cache 本体（512 MiB）と比べ小さい。
- 帯域: save / restore 各 ~42 MiB の D2D コピー。ホットパスではない。

### 検証

- required test `test_k_proxy_page_copy`: 異なる num_pages の 2 pool 間で page copy を行い、
  code/scale/shape が byte-exact であること、未コピー page が 0 であること、範囲外・layout 不一致を
  拒否することを確認。
- 統合: prefix 境界 checkpoint（prefill 由来）で、companion ありの restore 経路と
  prefix cache なしの full prefill が **GENERATED_IDS 完全一致**（restored=7488, prefill=8）。

### 注意

sparse decode で生成したトークンを含む checkpoint は、その生成時点の近似状態
（sparse で計算した hidden → K/V）を保存する。したがってそれを restore した場合、
同じ token 列を full prefill した結果とは一致しない。これは companion proxy の欠陥ではなく、
checkpoint 内容が近似であることに起因する。

---

## 20. 限界と次段階

- teacher forcing（§18）で複数位置の比較が可能になった。より長い context（128K 等）での
  per-position KLD は未測定。
- DFlash2 の target verify 併用は §17 で検証済み（budget=1.0 で token 完全一致、pruning 時は近似）。
- target-dtype Oracle の page 集合一致度（BF16 / FP8 / PSQ4 Oracle の Jaccard /
  retained target attention mass）は未集計。
- bf16 の quant floor（2.677e-04）は probe dump 出力の BF16 round に由来する。
- FP8 dense kernel の効率は本 Gate の範囲外。改善後は crossover を再測定する。
- prefix cache companion proxy（4G）は §19 で実装・検証済み。
- shadow mode（4I）は未実施。

