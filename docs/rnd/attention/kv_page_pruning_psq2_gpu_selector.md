> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# PSQ2 K Proxy GPU Page Selector — Gate 3 / 3.5 / 4-thin / 3.6 レポート

> 注記: PSQ2 KV の実装（shape table・専用 kernel・selector・trainer）は
> リポジトリから削除済み。本記録は履歴として残す。

Gate 3 で PSQ2 K proxy selector を GPU へ移し、Gate 3.5 で selector kernel を最適化し、
Gate 4-thin で最小の sparse BF16 attention を実装し、
Gate 3.6 で production-like な batch layout の benchmark を追加した。

結論:

- selector correctness: GO。page score と selected set / qmask は host reference と一致。
- sparse attention correctness: GO。all-pages で dense と一致（rel_l2 3e-8）。
- **Gate 3.6: GO**。128K distinct-slot（production-like）で
  `selector + sparse` は rows=1 で dense の 18.7%、rows=8 でも **72.2%**。
  rows>=4 を NO-GO と断定していた旧判定は撤回する。
- **Gate 3.7: score kernel を 12-19% 改善**（reduction のインターリーブ + shape table の shared 化）。
  128K rows=8 は **64.46%**（Strong GO まで残り約 4.5 point）。
- 残る支配項は score kernel（rows=8 で総時間の約 54%）と sparse attention（約 45%）。

---

## 1. revision / 環境

| 項目 | 値 |
| --- | --- |
| worktree | `.worktrees/poc-kv-psq2-proxy` |
| branch | `poc/kv-psq2-proxy` |
| revision | `acb71f2d` |
| PSQ2-K-Shape-v0 | `423819bf3d5dd86b8dbdc264b8d9bd15ac370ea1951723b4ad00f2680771116c` |
| GPU | R9700 (gfx1201), idle |
| geometry | q_heads=24, kv_heads=4, q_per_kv=6, head_dim=256, page=16 |
| always-on | recent_tokens=4096, sink_pages=1 |

---

## 2. 実装

| 項目 | 値 |
| --- | --- |
| score kernel | `psq2_page_score_impl`（`psq2_page_selector.hip`） |
| select kernel | `psq2_page_select_impl`（同 file） |
| 共通 decode | `include/.../attention/psq2_device.h` |
| sparse attention | `attention_paged_sparse_split_bf16_impl`（`paged_decode.hip`） |
| bench | `phaseshift-bench psq2-selector` |
| tests | `test_psq2_page_selector`, `test_sparse_paged_attention`（gpu1;required） |

### Gate 3.5 最適化

初版 select は 707us だった。原因は次の 3 点。

1. compaction と tie-fill を thread 0 が逐次処理（固定費 約 290us）。
2. tie-fill が常に fill >= 1 のため毎回 E 回走っていた。
3. all-select 分岐が barrier を欠き compaction と race していた。

block-wide exclusive scan で並列化し、select は **707us → 34us**（GROUP_MAX, 128K）になった。

---

## 3. correctness

`test_psq2_page_selector`: page score max abs diff 1.192e-06、selected set / qmask 完全一致。

`test_sparse_paged_attention`: all-pages rel_l2 vs dense 3.182e-08、
masked active-heads rel_l2 3.375e-08、nonfinite 0。

回帰: test_paged_attention / test_kv_append / test_qwen35_psq2_kv_pool PASS。

---

## 4. Gate 3.6: production-like batch benchmark

bench に batch layout を追加した。

| layout | 内容 |
| --- | --- |
| `shared-slot` | 全 row が同一 slot / 同一 physical page（旧 benchmark、best-case） |
| `distinct-slot` | 各 row が独立 slot・独立 physical page range（**production-like, default**） |
| `shared-prefix` | prefix を共有し tail を row 固有に（`--shared-prefix-frac`） |

physical page は全 page を一度だけ PSQ2 encode し、row ごとの block table で解決する。
`---batch-shape mixed` で row ごとに context を変えられる。
percentage は bench 自身が同一 run 内の `dense_us` から計算する。

### 128K GROUP_MAX 1/4

| layout | rows | dense_us | score_us | select_us | sparse_us | total_us | %dense | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shared-slot | 1 | 7435.0 | 608.5 | 35.1 | 750.4 | 1394.1 | 18.75% | 5.33x |
| shared-slot | 2 | 7468.0 | 1371.4 | 35.0 | 781.1 | 2187.5 | 29.29% | 3.41x |
| shared-slot | 4 | 7802.0 | 2486.9 | 36.9 | 1706.7 | 4230.5 | 54.22% | 1.84x |
| shared-slot | 8 | 10469.1 | 4672.2 | 46.9 | 3456.7 | 8175.8 | 78.09% | 1.28x |
| **distinct-slot** | 1 | 7450.8 | 609.4 | 35.1 | 747.8 | 1392.2 | **18.69%** | **5.35x** |
| **distinct-slot** | 2 | 7297.3 | 1371.2 | 34.9 | 724.2 | 2130.2 | **29.19%** | **3.43x** |
| **distinct-slot** | 4 | 8772.0 | 2819.3 | 42.8 | 1862.4 | 4724.5 | **53.86%** | **1.86x** |
| **distinct-slot** | 8 | 12616.1 | 5382.7 | 51.0 | 3678.9 | 9112.7 | **72.23%** | **1.38x** |
| shared-prefix 0.5 | 4 | 8507.8 | 2450.3 | 42.0 | 1856.0 | 4348.3 | 51.11% | 1.96x |
| shared-prefix 0.5 | 8 | 11913.5 | 5290.0 | 50.2 | 3606.1 | 8946.3 | 75.09% | 1.33x |
| shared-prefix 0.9 | 4 | 7722.2 | 2451.3 | 37.8 | 1725.7 | 4214.8 | 54.58% | 1.83x |
| shared-prefix 0.9 | 8 | 11478.4 | 4690.1 | 49.5 | 3546.6 | 8286.3 | 72.19% | 1.39x |

### 64K GROUP_MAX 1/4

| layout | rows | dense_us | total_us | %dense | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| shared-slot | 1 | 3788.7 | 690.5 | 18.22% | 5.49x |
| shared-slot | 2 | 3887.0 | 1048.6 | 26.98% | 3.71x |
| shared-slot | 4 | 4071.7 | 2233.0 | 54.84% | 1.82x |
| shared-slot | 8 | 4670.9 | 4339.5 | 92.91% | 1.08x |
| distinct-slot | 1 | 3836.9 | 698.6 | 18.21% | 5.49x |
| distinct-slot | 2 | 3817.8 | 989.0 | 25.91% | 3.86x |
| distinct-slot | 4 | 3952.2 | 2177.1 | 55.08% | 1.82x |
| distinct-slot | 8 | 6049.0 | 4914.1 | 81.24% | 1.23x |

### PER_Q_HEAD 比較（128K distinct-slot 1/4）

| rows | dense_us | score_us | select_us | sparse_us | total_us | %dense | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7366.6 | 596.3 | 153.9 | 1875.5 | 2625.7 | 35.64% | 2.81x |
| 4 | 8772.8 | 2734.3 | 200.2 | 4192.2 | 7126.7 | 81.24% | 1.23x |

PER_Q_HEAD は union が広く（0.798）sparse の read/計算が増えるため、GROUP_MAX に劣る。

---

## 5. Gate 4-thin（旧 shared-slot synthetic 計測、参考）

旧 report の Gate 4-thin 表は shared-slot（全 row 同一 page）で、
dense と total を別 run から混ぜていたため比率が不整合だった。
同一 run で測り直した shared-slot の値は §4 のとおり。

128K shared-slot GROUP_MAX 1/4:

| rows | total_dense_pct | speedup |
| ---: | ---: | ---: |
| 1 | 18.75% | 5.33x |
| 4 | 54.22% | 1.84x |
| 8 | 78.09% | 1.28x |

**旧値（rows=8 83.13% / rows=4 53.79%）は使用しない。**
また `rows>=4 = NO-GO` の断定は撤回する。

---

## 6. Gate 3.6 判定

128K distinct-slot を primary とする（§10）。

| rows | total_dense_pct | 判定 |
| ---: | ---: | --- |
| 8 | **72.23%** | **GO（<= 0.80）** |
| 4 | 53.86% | 明確に勝つ |
| 2 | 29.19% | 明確に勝つ |
| 1 | 18.69% | 明確に勝つ |

**Gate 3.6: GO。** rows=8 でも `selector + sparse` は dense の 72% で、
条件（<= 0.80）を満たす。Gate 3.7 で score kernel を改善できれば Strong GO も狙える。

---

## 7. Gate 3.7: score kernel 最適化

順序 A→B→C→D→E（§18）。各段階で bench を保存した。

### A. adaptive splits（不採用）

bench に `--score-splits` を追加し、splits = 128 / 256 / 512 / 1024 / 2048 / auto を
64K・128K × rows 1/2/4/8 で sweep した。

score_us は splits に対しほぼフラット（±5-15%）で、auto が常に競争力があった。
**score は warp 並列度では支配されていない**ため、splits の adaptive 化は採用しない。
auto = `(block_table_stride + 6) / 7`（1 warp あたり約 7 page）を維持する。

### B/C. ボトルネック分解（一時変更による計測）

128K distinct, GROUP_MAX, splits=auto で内訳を計測した。

| 変種 | rows=1 score_us | rows=8 score_us |
| --- | ---: | ---: |
| baseline | 636 | 5445 |
| online logsumexp を max-only に置換 | 551 | 4338 |
| さらに warp reduction を除去 | 325 | 3087 |
| decode の load を除去（値は擬似） | 397 | 3207 |

寄与の概算: **warp reduction 約 40-50%、decode(load+table) 約 25-33%、
exp/log 約 15-20%**。

### 採用した最適化

1. **6 head 分の reduction をインターリーブ**（`perf(attention): interleave PSQ2 six-head warp reduction`）。
   6 本の butterfly を step ごとにまとめて発行し ILP を上げる。score 12-16% 改善。
2. **shape table を constant から shared へ**（`perf(attention): stage PSQ2 shape table in shared memory`）。
   `table[sid][code]` の divergent constant load による直列化を回避。score 8-10% 改善。

### 試行して不採用

- `__launch_bounds__(32, min_blocks)` による occupancy 強制（8/16 を試行）:
  register spill により rows=1/8 とも悪化。不採用。
- 2 warp で 6 head を 3+3 分割（両 warp が K を独立 decode）:
  correctness は通るが decode が 2 重になり rows=1/8 とも悪化（486→600、4336→5541）。不採用。
- 3 head のみ計算する timing 実験: rows=8 で 31% 減、rows=1 では変化なし。
  → rows=8 は head 数に部分依存、rows=1 は memory/latency 律速。

### Gate 3.7 後の全体（128K distinct-slot GROUP_MAX 1/4）

| rows | dense_us | score_us | select_us | sparse_us | total_us | %dense | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7377.9 | 475.5 | 34.9 | 750.0 | 1260.4 | 17.08% | 5.85x |
| 2 | 7302.3 | 927.3 | 34.8 | 725.6 | 1687.8 | 23.11% | 4.33x |
| 4 | 8893.0 | 2237.0 | 43.4 | 1833.9 | 4114.3 | 46.27% | 2.16x |
| 8 | 12502.5 | 4355.5 | 52.3 | 3651.1 | 8058.9 | **64.46%** | **1.55x** |

64K distinct-slot GROUP_MAX 1/4: rows=1 16.30%、rows=2 24.19%、rows=4 44.20%、rows=8 69.13%。

128K rows=8 の推移: Gate 3.6 72.23% → reduction インターリーブ 68.46% → shared table **64.46%**。
Strong GO（<=60%）まで残り約 4.5 point。

### D/E（未実施）

page_max / top2_lse の offline 評価（§16/§17）は未実施。
max-only 実験の上限改善は約 15-20% であり、品質（model-wide union ratio 差 <= 0.03、
mean rel_l2 <= 1e-2）が確認できるまで GPU 化しない。優先度は低い。

---

## 8. 残る課題

rows=8 の内訳（128K distinct, Gate 3.7 後）: score 4356us（54%）、sparse 3651us（45%）、select 52us。

1. score の decode load 最適化（code byte の vector load 等）。
2. sparse attention の index / split reduce overhead。
3. Gate 3.8: temporal / cross-layer selector reuse
   （refresh=2/4 で selector 実行回数を削減。連続 decode Q dump が必要）。
4. 品質評価（real model KLD, top1/top5, needle）は Gate 4 本実装で行う。
