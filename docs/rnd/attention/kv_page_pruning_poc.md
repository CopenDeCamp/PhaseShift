> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# KV Page Pruning PoC — Gate 0 / Gate 1 / Gate 2 レポート

本ドキュメントは、paged attention に対する Query-aware KV Page Pruning の
PoC（Gate 0〜Gate 2）の結果を記録する。

対象は RDNA4 / ROCm / Qwen3.8-27B（full attention layer）である。

結論を先に述べる。

- **Gate 1（Oracle）: GO（条件付き）**
  64K / 128K context で、**old page を 1/4、最良層では 1/8 まで削っても
  平均 attention output 誤差 `rel_l2 <= 1e-2` を満たす case が存在する。**
  ただし **層依存が非常に大きい**。最初の full attention layer（layer 0）は密であり、
  1/2 を超える old page 保持を要求する。
- **Gate 2（BF16 min/max selector）: GO（条件付き）**
  Oracle 比で page budget を 1.5〜2 倍に増やせば Oracle 同等の閾値を満たせる。
  `GROUP_MAX` は `PER_Q_HEAD` より多少品質が落ちるが、実用範囲であり、
  metadata read を 6 分の 1 にできるため production では `GROUP_MAX` を優先する価値がある。
- **Gate 3 以降へ進む判断: 進む。**
  ただし per-layer budget と full attention layer 0 の扱いを必須条件とする。

---

## 1. 環境

| 項目 | 値 |
|---|---|
| worktree | `/opt/zen/wk/PhaseNonShift/.worktrees/poc-kv-page-pruning` |
| branch | `poc/kv-page-pruning` |
| 開始 revision（分岐元 origin/main） | `5a53fef2b82c3bfde4cf231d0b18b985700c6fdd` |
| 計測時 origin/main | `f4b0d68d06dd71c3834f533b2b18d17c0dd7b624` |
| GPU | AMD Radeon AI PRO R9700 (gfx1201) |
| HIP | 7.15.26333-0000000 |
| ROCm-SMI | 4.0.0+6b0e43f3 |
| model | `models/Qwen3.8-27B-PSQ` |
| KV dtype（probe） | BF16 |
| page_tokens | 16 |
| head_dim | 256 |
| q_heads | 24 |
| kv_heads | 4 |
| q_per_kv | 6 |
| full attention layer 数 | 16 |
| probe layer | 0, 7, 15 |

layer は full attention layer の通し番号（`state_index`）である。

---

## 2. Gate 0: Dense Baseline

### 2.1 required correctness

```text
test_paged_attention                      PASS
test_paged_attention_split_reduce_exact   PASS
```

### 2.2 dense decode latency（rows=1, out=f32, warmup=50, samples=50, launches=10）

FP8（optimized）:

| visible | p50 (us) | mean (us) |
|---:|---:|---:|
| 4096 | 575.40 | 575.40 |
| 16384 | 2396.63 | 2407.19 |
| 32768 | 5050.66 | 5069.56 |
| 65536 | 9937.15 | 9934.70 |
| 131072 | 19663.85 | 19657.01 |
| 262144 | 39701.26 | 39697.68 |

BF16（optimized, plain）:

| visible | p50 (us) | mean (us) |
|---:|---:|---:|
| 4096 | 184.82 | 184.83 |
| 16384 | 744.79 | 744.37 |
| 32768 | 1999.11 | 1999.02 |
| 65536 | 3762.19 | 3762.57 |
| 131072 | 7281.16 | 7265.17 |
| 262144 | 14219.56 | 14212.04 |

いずれも `--check`（correctness path との比較）は PASS。

注意: FP8 decode は BF16 decode より遅い。これは既存 FP8 decode kernel の
per-element 変換・vector 化の事情によるもので、page pruning とは独立した既知事象である。
本 PoC の probe は BF16 KV で行う。

---

## 3. 手法

### 3.1 probe dump

`try_launch_paged_attention` に env-gated の debug dump を追加した。

- `PHASESHIFT_PA_PROBE_DIR`
- `PHASESHIFT_PA_PROBE_LAYER`（カンマ区切り、複数層）
- `PHASESHIFT_PA_PROBE_CONTEXT`（最終 prefill chunk を狙う）

最終 prefill chunk の `PAGED_ATTENTION` 呼び出しで、以下を dump する。

```text
Q（post-RoPE, BF16）
K/V pool（対象 layer 全体, BF16）
block table / row positions / row sequence slots
scale
PAGED_ATTENTION の出力（参照用 dense_out）
```

これは R&D 専用であり、production 統合前に test-only へ移すか削除する。

### 3.2 corpus

`models/Qwen3.8-27B-PSQ` の tokenizer で以下を連結し、204049 token の
PSKLDTOK corpus を作成した。

- prose: Moby Dick（Project Gutenberg #2701）
- code: repository の `src/**/*.cpp`, `*.hip`, `include/**/*.h`
- reasoning / docs: `docs/**/*.md`
- structured: 生成 JSON
- long dependency: token 8030 / 30066 / 55103 に calibration code（needle）を配置

### 3.3 解析

`phaseshift-bench paged-prune` を追加し、dump に対して CPU（f64/f32）で以下を行う。

- page ごとの exact `logsumexp`（Oracle page importance）
- page ごとの BF16 `K_min/K_max` からの query-aware upper bound
- `always-on` = current partial page + recent window + sink pages
- `selected = always-on ∪ Top-N(eligible old full pages)`
- selected page のみを **既存 BF16 K/V** で exact attention 再計算
- dense attention output との比較（rel_l2 / cosine / retained mass）

`always-on` は初期値として `recent_tokens=4096`, `sink_pages=1` を用いた。

### 3.4 CPU 参照の妥当性

`dense_ref_check`（CPU 再構成 dense と model 出力の一致）は全条件で

```text
max_rel_l2 <= 1.65e-03
min_cosine >= 0.999999
```

であり、解析系は正しい。

---

## 4. Gate 1: Oracle Page Pruning

32 query row、`recent=4096`, `sink=1`。`old_keep` は eligible old page の残存比。

### 4.1 64K context（Oracle）

| layer | old_keep | rel_l2 mean | rel_l2 max | cos min | mass mean | mass min |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1/2 | 1.70e-02 | 8.26e-02 | 0.99659 | 0.9784 | 0.8532 |
| 0 | 1/4 | 4.84e-02 | 2.14e-01 | 0.97710 | 0.9415 | 0.6852 |
| 0 | 1/8 | 8.54e-02 | 3.67e-01 | 0.93429 | 0.8996 | 0.5496 |
| 7 | 1/2 | 8.58e-04 | 1.10e-02 | 0.99996 | 0.9990 | 0.9814 |
| 7 | 1/4 | 3.40e-03 | 3.90e-02 | 0.99953 | 0.9957 | 0.9348 |
| 7 | 1/8 | 7.86e-03 | 8.27e-02 | 0.99796 | 0.9897 | 0.8715 |
| 15 | 1/4 | 5.72e-03 | 4.75e-02 | 0.99888 | 0.9817 | 0.8315 |
| 15 | 1/8 | 1.10e-02 | 7.95e-02 | 0.99686 | 0.9639 | 0.7148 |

### 4.2 128K context（Oracle）

| layer | old_keep | rel_l2 mean | rel_l2 max | cos min | mass mean | mass min |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1/2 | 1.22e-02 | 4.95e-02 | 0.99886 | 0.9793 | 0.8815 |
| 0 | 1/4 | 3.10e-02 | 1.07e-01 | 0.99489 | 0.9347 | 0.7019 |
| 0 | 1/8 | 4.99e-02 | 1.80e-01 | 0.98576 | 0.8776 | 0.5303 |
| 7 | 1/4 | 5.52e-03 | 4.82e-02 | 0.99910 | 0.9914 | 0.8802 |
| 7 | 1/8 | 1.26e-02 | 9.40e-02 | 0.99713 | 0.9794 | 0.7726 |
| 15 | 1/4 | 5.04e-03 | 3.64e-02 | 0.99938 | 0.9725 | 0.8369 |
| 15 | 1/8 | 9.84e-03 | 6.55e-02 | 0.99819 | 0.9361 | 0.7241 |

### 4.3 最小安全 old page 比率（Oracle, 平均 rel_l2 <= 1e-2 基準）

| context | layer 0 | layer 7 | layer 15 |
|---|---:|---:|---:|
| 64K | > 1/2 | 1/8 | 1/4 |
| 128K | > 1/2 | 1/4 | 1/8 |

### 4.4 Gate 1 判定

- **old page 1/4 で平均 `rel_l2 <= 1e-2` を満たす case は存在する（layer 7, 15）。**
  → Gate 1 GO 条件「64K 以上で old pages <= 1/4」を満たす。
- ただし layer 0 は密で、old page を 1/2 残しても平均 `rel_l2` は 1e-2 を超える。
  → **per-layer budget は必須。**
- 全 head の worst-case（`rel_l2 max`, `cos_min`）まで 1e-2 / 0.9999 を課すと、
  1/4 では満たさない。閾値は平均基準で解釈し、worst-case は retained mass と
  downstream KLD で担保するのが妥当である。
- Gate 1 NO-GO 条件「50% 以上を残さないと品質維持できない」は
  全体としては成立しない。層 0 のみが該当する。

---

## 5. Gate 2: BF16 Min/Max Page Selector

`recent=4096`, `sink=1`, 32 query row。
`page_upper = scale * sum_d(q_d >= 0 ? q_d*K_max_d : q_d*K_min_d)` を page ranking のみに使用し、
attention 本計算は既存 BF16 K/V の exact で行う。

### 5.1 64K context, old_keep = 1/4

| policy | layer | rel_l2 mean | rel_l2 max | cos min | mass min | oracle page recall |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 7 | 3.40e-03 | 3.90e-02 | 0.99953 | 0.9348 | 1.000 |
| per_q_head | 7 | 6.96e-03 | 8.05e-02 | 0.99856 | 0.8349 | 0.711 |
| group_max | 7 | 9.80e-03 | 9.09e-02 | 0.99630 | 0.7620 | 0.655 |
| oracle | 15 | 5.72e-03 | 4.75e-02 | 0.99888 | 0.8315 | 1.000 |
| per_q_head | 15 | 1.04e-02 | 5.66e-02 | 0.99851 | 0.6689 | 0.619 |
| group_max | 15 | 1.23e-02 | 6.66e-02 | 0.99792 | 0.6441 | 0.578 |

### 5.2 128K context, old_keep = 1/4

| policy | layer | rel_l2 mean | rel_l2 max | cos min | mass min | oracle page recall |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 7 | 5.52e-03 | 4.82e-02 | 0.99910 | 0.8802 | 1.000 |
| per_q_head | 7 | 1.34e-02 | 1.03e-01 | 0.99706 | 0.7153 | 0.643 |
| group_max | 7 | 1.92e-02 | 1.40e-01 | 0.99055 | 0.4936 | 0.572 |
| oracle | 15 | 5.04e-03 | 3.64e-02 | 0.99938 | 0.8369 | 1.000 |
| per_q_head | 15 | 1.07e-02 | 7.74e-02 | 0.99745 | 0.6570 | 0.569 |
| group_max | 15 | 1.31e-02 | 9.43e-02 | 0.99588 | 0.5305 | 0.528 |

### 5.3 old_keep = 1/2（selector）

| context | policy | layer | rel_l2 mean | cos min | mass min |
|---|---|---:|---:|---:|---:|
| 64K | group_max | 7 | 3.05e-03 | 0.99767 | 0.9040 |
| 64K | group_max | 15 | 4.92e-03 | 0.99939 | 0.8226 |
| 128K | group_max | 7 | 6.38e-03 | 0.99831 | 0.7500 |
| 128K | group_max | 15 | 4.86e-03 | 0.99904 | 0.7443 |

### 5.4 Gate 2 判定

- BF16 min/max selector は、Oracle より page budget を概ね 1.5〜2 倍に増やすことで
  同じ品質閾値（平均 rel_l2 <= 1e-2）に到達できる。
  → **Gate 2 GO。**
- `GROUP_MAX` は `PER_Q_HEAD` に対し
  - 同じ budget では平均 rel_l2 が約 1.3〜1.6 倍
  - 同じ品質には約 2 倍の budget
  を要する。一方で production の metadata read を 6 分の 1 にできる。
- page 一致率（oracle page recall）は 0.5〜0.7 程度でも、最終 attention 品質は
  oracle に近い。これは「page 一致率そのものより retained mass / final output を
  優先する」という方針と整合する。

---

## 6. 考察

### 6.1 層依存

- layer 0（最初の full attention layer）は明確に密である。
  old page を 1/2 残しても平均 rel_l2 は 1e-2 を超える。
- layer 7 / 15（中盤・終盤）は疎であり、1/8〜1/4 の old page で平均 rel_l2 <= 1e-2。
- したがって **layer 一律 budget は不可**。per-layer budget を前提とする。

### 6.2 閾値の解釈

- 平均 rel_l2 <= 1e-2 は layer 7 / 15 で成立する。
- per-head worst-case まで 1e-2 / cos 0.9999 を課すのは、raw attention output に対して
  厳しすぎる。最終判断は実 model の logit KLD（Gate 7）で行うべきである。
- retained mass は 1/4 で layer 7: 0.93、layer 15: 0.83、layer 0: 0.69 程度。

### 6.3 production 構成への示唆

- metadata は BF16 K_min/K_max（1 page / 1 KV head = 1024 byte）で成立する。
  PSQ metadata 化（Gate 9）はこの後の最適化として妥当。
- selector は `GROUP_MAX` を第一候補とする。
- always-on は current partial page + recent window + sink pages。
  recent=4096 は 64K で約 6% であり妥当。sink=1 で十分機能する。
- layer 0 のような密な層は budget を大きくする。あるいは
  layer 0 のみ pruning を無効化する選択肢もある。

---

## 7. 再現手順

```bash
# build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPHASESHIFT_ROCM_ROOT="$(rocm-sdk path --root)" \
  -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$(rocm-sdk path --root)" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake --build build --target phaseshift-bench test_paged_attention \
  test_paged_attention_split_reduce_exact -j 6

# Gate 0
./build/phaseshift-bench paged-attention --variant optimized --kv-dtype fp8 \
  --out-dtype f32 --q-heads 24 --kv-heads 4 --head-dim 256 --page-tokens 16 \
  --rows 1 --visible 4096,16384,32768,65536,131072,262144 \
  --warmup 50 --samples 50 --launches 10 --check --out dense_fp8.csv

# probe dump
PHASESHIFT_PA_PROBE_DIR=/tmp/probe_64k \
PHASESHIFT_PA_PROBE_LAYER=0,7,15 \
PHASESHIFT_PA_PROBE_CONTEXT=65536 \
  ./build/phaseshift-bench tg --mode forward --context 65536 --tokens 1 \
  --model-dir models/Qwen3.8-27B-PSQ --arena-gib 24 \
  --tokens-file prune_corpus.pskldtok

# Gate 1 / Gate 2
./build/phaseshift-bench paged-prune --dir /tmp/probe_64k/layer_7 \
  --recent 4096 --sink 1 --rows 32 \
  --policies oracle,per_q_head,group_max \
  --fractions 0.5,0.25,0.125,0.0625 --budgets 256,512,1024 --out sweep.csv
```

---

## 8. 未実施項目

- Gate 3（GPU selector kernel）
- Gate 4（sparse exact attention kernel）
- Gate 5（metadata lifecycle / PagedKVPool 拡張 / prefix cache）
- Gate 6（runtime integration）
- Gate 7（実 model KLD、needle retrieval）
- Gate 8（selector / sparse attention の分離性能、crossover）
- Gate 9（PSQ metadata）
- Gate 10（selector unit test、sparse correctness、boundary）

本 PoC の probe dump フックは production 到達可能な実装ではない。
Gate 3 以降へ進む前に test-only へ隔離するか削除すること。
