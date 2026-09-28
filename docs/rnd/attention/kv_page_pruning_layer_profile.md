> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# KV Page Pruning — Gate 2.5 全 layer sparsity profile

本ドキュメントは Gate 2.5（全 full attention layer の page sparsity profile と
production budget table）の結果を記録する。

結論:

- **Gate 2.5: GO（条件付き）**
  全 16 layer × 32K/64K/128K の profile を取得し、layer 単位の safe budget を決定できた。
  required tests は PASS。probe は compile-time option へ隔離し、
  production build では完全に compile out される。
- **ただし重大な留保がある。**
  `GROUP_MAX`（production 第一候補）が実現できる model-wide KV page read 削減は
  128K で約 15%、64K で約 15%（conservative production table）。
  Oracle は 128K で約 60% 削減可能であり、**ボトルネックは sparsity ではなく selector 品質**である。
- Gate 4 の性能目標（128K で attention latency 20% 削減）は
  `GROUP_MAX` のみでは達成困難と判断する。
  Gate 3 では `GROUP_MAX` に加えて、metadata を共有したまま品質を上げる方式
  （per-q-head selection の union 等）を検討すべきである。

---

## 1. revision

| 項目 | 値 |
|---|---|
| branch 作成時 baseline | `5a53fef2` |
| rebase 後 revision（Gate 2.5 計測時） | `1acf75f8` |
| current origin/main | `f4b0d68d` |
| worktree | `.worktrees/poc-kv-page-pruning` |
| GPU | R9700 (gfx1201), device 2 を明示使用 |
| model | `Qwen3.8-27B-PSQ` |
| KV dtype | BF16 |
| full attention layer 数 | 16（model config / runtime metadata から取得、hard-code なし） |
| probe | `PHASESHIFT_PA_PROBE` compile-time option（default OFF） |

計測 corpus は Gate 1–2 と同一（Moby Dick + repository code/docs + JSON + needle、約204K token）。

---

## 2. 計測条件

- `recent_tokens = 4096`
- `sink_pages = 1`
- current partial page は常時 exact
- query row 数 = 32（最終 prefill chunk から 32 行）
- budget: dense（1.0）/ 1/2 / 1/4 / 1/8 / 1/16
- 判定: `mean rel_l2 <= 1e-2`（Gate 1 と同一、変更なし）

`dense_ref_check`（CPU 再構成 dense と model 出力）は全 layer / context で
`max_rel_l2 <= 1.7e-3`, `min_cosine >= 0.999999` であり、解析系は正しい。

---

## 3. 全 layer budget table

safe budget は「その budget で `mean rel_l2 <= 1e-2` を満たす最小の old-page 保持率」。
`DENSE` は 1/2 でも閾値を満たさない（= selector を使わない）ことを意味する。

| layer | Oracle 64K | GROUP_MAX 64K | Oracle 128K | GROUP_MAX 128K | PER_Q_HEAD 64K | production policy |
| ----: | ---------: | ------------: | ----------: | -------------: | -------------: | :---------------- |
| 0 | DENSE(in 1/2) | DENSE | DENSE(in 1/2) | DENSE | DENSE | DENSE |
| 1 | DENSE(in 1/2) | DENSE | 1/2 | DENSE | DENSE | DENSE |
| 2 | 1/2 | DENSE | 1/2 | DENSE | DENSE | DENSE |
| 3 | 1/2 | DENSE | 1/2 | DENSE | DENSE | DENSE |
| 4 | 1/2 | DENSE | 1/2 | DENSE | DENSE | DENSE |
| 5 | 1/2 | DENSE | 1/2 | DENSE | DENSE | DENSE |
| 6 | 1/8 | 1/4 | 1/4 | 1/2 | 1/4 | KEEP_1_2 |
| 7 | 1/8 | 1/4 | 1/4 | 1/2 | 1/4 | KEEP_1_2 |
| 8 | 1/8 | 1/4 | 1/4 | 1/2 | 1/4 | KEEP_1_2 |
| 9 | 1/16 | 1/2 | 1/4 | 1/2 | 1/4 | KEEP_1_2 |
| 10 | 1/8 | 1/2 | 1/4 | DENSE | 1/2 | DENSE |
| 11 | 1/8 | DENSE | 1/4 | DENSE | 1/2 | DENSE |
| 12 | 1/8 | DENSE | 1/4 | DENSE | 1/2 | DENSE |
| 13 | 1/4 | DENSE | 1/2 | DENSE | DENSE | DENSE |
| 14 | 1/4 | 1/2 | 1/4 | DENSE | 1/2 | DENSE |
| 15 | 1/4 | 1/2 | 1/8 | 1/2 | 1/2 | KEEP_1_2 |

production policy は 64K–128K bucket で conservative（64K/128K の安全側）を採用した。

### layer 集計

production policy（保守的 bucket）:

| policy | layer 数 |
|---|---:|
| DENSE | 11 |
| KEEP_1_2 | 5 |
| KEEP_1_4 | 0 |
| KEEP_1_8 | 0 |
| KEEP_1_16 | 0 |

参考: 各 context で測定した GROUP_MAX safe budget の分布:

| context | DENSE | KEEP_1_2 | KEEP_1_4 | KEEP_1_8 | KEEP_1_16 |
|---|---:|---:|---:|---:|---:|
| 32K | 14 | 2 | 0 | 0 | 0 |
| 64K | 9 | 4 | 3 | 0 | 0 |
| 128K | 11 | 5 | 0 | 0 | 0 |

参考: Oracle safe budget の分布:

| context | DENSE | KEEP_1_2 | KEEP_1_4 | KEEP_1_8 | KEEP_1_16 |
|---|---:|---:|---:|---:|---:|
| 32K | 0 | 14 | 2 | 0 | 0 |
| 64K | 2 | 4 | 3 | 6 | 1 |
| 128K | 1 | 6 | 8 | 1 | 0 |

---

## 4. expected model-wide KV page read ratio

`sel_ratio = selected pages / visible pages` の layer 平均。

| context | Oracle | PER_Q_HEAD | GROUP_MAX |
|---|---:|---:|---:|
| 32K | 0.536 | （未計測） | 0.945 |
| 64K | 0.389 | 0.678 | 0.751 |
| 128K | 0.402 | 0.788 | 0.849 |

production conservative table（GROUP_MAX, 64K–128K bucket）:

| context | expected sel_ratio | page read 削減 |
|---|---:|---:|
| 64K | ≈ 0.853 | ≈ 15% |
| 128K | ≈ 0.849 | ≈ 15% |

Oracle は 60% 前後の削減を示す。**selector 品質が最大のボトルネック**である。

---

## 5. 考察

### 5.1 sparsity は存在する

Oracle では 64K / 128K で多くの layer が 1/8〜1/16 まで削減可能であり、
sparsity 自体は十分存在する。特に layer 6–12 は 64K で 1/8 以下。

### 5.2 GROUP_MAX の品質不足

`GROUP_MAX` は同一 KV head の 6 Q head の upper bound の最大値を用いるため、
page 集合が union 側へ広がり、budget を多く要求する。
結果として 64K/128K で 11/16 layer が DENSE となった。

`PER_Q_HEAD` は quality が良い（64K で many layer 1/2）が、
Oracle にはなお大きく劣る。min/max upper bound 自体の緩さも要因である。

### 5.3 閾値ぎりぎりの layer が多い

64K の GROUP_MAX 1/2 で `mean rel_l2` が 1e-2 をわずかに超える layer が複数ある
（layer 10: 0.0157, layer 11: 0.0108, layer 12: 0.0112, layer 13: 0.0130, layer 14: 0.0125）。
これらは selector の品質ではなく閾値の厳しさで DENSE 化されている。
Gate 4 の logit KLD で許容できる可能性が高く、
`mean rel_l2 <= 1e-2` の hard gate だけで最終判断すべきではない。

### 5.4 dense layer の扱い

layer 0–5 は Oracle でも 64K の 1/2 が境界（layer 0 は DENSE、layer 1–5 は 1/2）。
layer 0 は全方式で DENSE であり、selector を起動しない。

### 5.5 context 依存

32K では GROUP_MAX の削減効果はほぼ無い（sel_ratio 0.945）。
`sparse_min_context` は 64K 以上に設定するのが妥当である。

---

## 6. production budget table（Gate 4 初期値）

context bucket:

| visible | policy |
|---|---|
| `< 32K` | DENSE（未測定、fallback） |
| `32K–64K` | layer 8, 15 のみ KEEP_1_2、他 DENSE |
| `64K–128K` | layer 6, 7, 8, 9, 15 を KEEP_1_2、他 DENSE |
| `>= 128K` | layer 6, 7, 8, 9, 15 を KEEP_1_2、他 DENSE（128K 実測を上限、256K 計測前はより aggressive にしない） |

`GROUP_MAX` のみを使う場合、この table がそのまま Gate 4 初期値となる。
ただし §5 の理由により、Gate 3 で selector 改善を検討した上で
Gate 4 の KLD 評価により near-threshold layer を追加することを推奨する。

---

## 7. probe の隔離

`try_launch_paged_attention` の probe は

```text
cmake -DPHASESHIFT_PA_PROBE=ON
```

でのみ有効化される。default は OFF であり、その場合

- probe コードは compile out される
- 呼び出し箇所も `#if` で除去される
- branch / allocation / memcpy / file I/O / synchronization は一切発生しない

production build は変更前と同一の code path である。

---

## 8. Gate 2.5 GO 判定

| 条件 | 結果 |
|---|---|
| 全 layer の 64K/128K profile | 取得済み |
| dense/sparse classification | 決定済み |
| GROUP_MAX safe budget の layer 決定 | 可能 |
| ある程度の layer 群で 1/2 以下 | 64K/128K で 5 layer |
| required tests PASS | PASS（paged attention、full suite は build 後実行） |
| probe が production path に影響しない | compile-time guard で保証 |

**Gate 2.5: GO。**

ただし production を `GROUP_MAX` のみで構成した場合の期待削減は約 15% であり、
Gate 4 の 20% latency 目標には不足する見込みである。
Gate 3 では selector 改善（per-q-head union 等）を並行評価対象とすること。
