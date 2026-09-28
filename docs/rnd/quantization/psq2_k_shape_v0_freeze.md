> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# PSQ2-K-Shape-v0 Freeze

> 注記: PSQ2 KV の実装（shape table・専用 kernel・selector・trainer）は
> リポジトリから削除済み。本記録は履歴として残す。

Gate 3 / Gate 4 の間、PSQ2 K proxy の shape table を再学習・変更しないための
freeze 宣言である。

---

## 1. Freeze 宣言

| 項目 | 値 |
| --- | --- |
| name | `PSQ2-K-Shape-v0` |
| model | `Qwen3.8-27B-PSQ` |
| worktree | `.worktrees/poc-kv-psq2-proxy` |
| branch | `poc/kv-psq2-proxy` |
| freeze revision | `2c8bffa927c18fbcd6b7a3b0508339ec788a9a06` |
| format | W32 / 2bit code / BF16 scale / 4bit Shape16 ID / 2.625 bpw |
| objective | K reconstruction MSE |
| sparse-attention-aware training | NO |
| ranking-aware training | NO |

---

## 2. 対象 artefact と SHA256

| artefact | path | SHA256 |
| --- | --- | --- |
| K shape table | `include/phaseshift/models/qwen35/kernels/optimized/attention/psq2_kv_shapes_generated.h` | `423819bf3d5dd86b8dbdc264b8d9bd15ac370ea1951723b4ad00f2680771116c` |
| trainer | `tools/rnd/train_psq2_kv_shapes.py` | `aa808377f657b7bf3e589543a861c37228ec434a11872cc314594411d133c86e` |

`psq2_kv_shapes_generated.h` の hash は required test
`test_psq2_k_shape_v0_freeze` により強制される。
hash が一致しない場合、ビルド後の required suite は FAIL する。

trainer の hash は記録のみであり、test では強制していない。
Gate 3 / Gate 4 では `tools/rnd/train_psq2_kv_shapes.py` を実行しない。

---

## 3. Gate 2.75 result

本 shape table を用いた PSQ2 K proxy selector の Gate 2.75 結果:

| context | policy | model-wide exact KV read ratio |
| --- | --- | --- |
| 64K | PSQ2 per-Q-head | 0.534 |
| 128K | PSQ2 group-max | 0.607 |
| 128K | PSQ2 per-Q-head | 0.613 |

参照値:

| context | Oracle |
| --- | --- |
| 64K | 0.531 |
| 128K | 0.599 |

PSQ2 K proxy は Oracle 近傍に到達しており、selector 品質の再学習は不要と判断した。
詳細は `docs/rnd/attention/kv_page_pruning_psq2_proxy_poc.md` を参照。

---

## 4. calibration corpus provenance

**currently not fully recorded / unknown**

`psq2_kv_shapes_generated.h` は `tools/rnd/train_psq2_kv_shapes.py` に
`--k` / `--v` で与えた BF16 K/V calibration dump から生成された。
dump 自体は runtime の `PHASESHIFT_KV_CALIB_DUMP`
（`src/phaseshift/models/qwen35/runtime/kv_calib_dump.cpp`）で取得できるが、
本 table の生成に使用した corpus、token 数、dump の SHA256 は
本 repository に記録されていない。

したがって次は保証できない。

- Shape v0 の calibration に使用した系列
- calibration が Gate 1〜2.75 の評価 corpus と独立であること
- 後続の blind holdout が shape training に対して完全に独立であること

Gate 4 の報告では、この点を limitation として明記する。

---

## 5. Accidental retrain 防止

Gate 3 / Gate 4 では `tools/rnd/train_psq2_kv_shapes.py` を実行しない。
仮に実行され header が書き換わった場合、
`test_psq2_k_shape_v0_freeze` が hash 不一致で FAIL する。

freeze を解除して良いのは、Gate 4 の終了報告が完了した後だけである。

---

## 6. PSQ2-K-Shape-v1 の扱い

新しい shape を検討する場合は、v0 を上書きしない。
別 branch / 別 PoC として `PSQ2-K-Shape-v1` を作成し、
v0 と v1 を別 table として比較する。

再学習を検討してよい条件は、selector 品質が明確なボトルネックになった場合のみである。
Gate 2.75 では Oracle 近傍であり、この条件を満たしていない。

---

## 7. Freeze guard test

| 項目 | 値 |
| --- | --- |
| test 名 | `test_psq2_k_shape_v0_freeze` |
| label | `cpu;required` |
| 実装 | `cmake/verify_psq2_k_shape_v0_freeze.cmake` |
| 登録 | `cmake/tests.cmake` |
| 判定 | `psq2_kv_shapes_generated.h` の SHA256 が v0 hash と一致しなければ FAIL |
