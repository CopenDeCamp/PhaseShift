# GDN compact-log commit PoC

## 目的

GDN (GatedDeltaNet) の speculative verify で accept prefix の recurrent state を
再構成するとき、full state snapshot（144 MiB/row）や full rerun を避けられるかを検証する。

現行:
- history mode: verify rows ごとに full recurrent state を保存（27B で約 144 MiB/row）。
  1.5 GiB guard により rows ≤ 10 に制限される。
- rerun mode: committed state を snapshot し、accept 後に accepted prefix を再実行
  （round あたり 22〜37 ms）。

## 数式（既存 kernel と同一）

`recurrence.hip` の chunk body はすでに

```
S_end = exp(G_last)·S_0 + Σ_i exp(G_last − G_i)·k_i·dT_i^T
```

を計算している（`dT` が `v_new`/`δ` に相当）。GDN の decay `a_t = exp(g_t)` と
`β_t` は V-head ごとの scalar なので、1 行の更新は

```
S_t = a_t·S_{t-1} + k_t·δ_t^T
δ_t = β_t·(v_t − k_t^T (a_t·S_{t-1}))
```

であり、accept = m の状態は閉形式で

```
S_m = A_m·S_0 + Σ_{i≤m} (A_m/A_i)·k_i·δ_i^T      A_i = Π_{j≤i} a_j
```

と書ける。`δ_i` は verify 時にすでに計算済みで、未来の token に依存しないため
prefix の再構成に再利用できる。よって compact log `{a_i, k_i, δ_i}` を保存すれば
full state snapshot は不要になる。

## PoC（`tests/unit/test_gdn_compact_commit_poc.hip`）

1 layer / 1 head（head dim 128）、N = 16 rows を fp32 の逐次 recurrence で回し、
`{a_i, k_i, δ_i}` を spill して、各 m の `S_m` を 2 方式で再構成し reference と比較した。

| 再構成方式 | max_abs | max_rel | bit-exact |
| --- | ---: | ---: | --- |
| 逐次 replay（δ から rank-1 更新を m 回） | 0 | 0 | **yes** |
| rank-m GEMM 一括（`A_m S_0 + Σ c_i k_i δ_i^T`） | 1.2e-7 | 1.1e-7 | no |

- 数式は厳密で、両方式とも fp32 丸め以内で reference と一致する。
- **逐次 replay は reference と bit-exact**。`S_t = a_t S_{t-1} + k_t δ_t^T` を
  S_0 から m 回適用するだけで、weight の再読み込み（rerun）は発生しない。
- rank-m GEMM 一括は丸め順が変わり **bit-exact でない**。NgramTail Gate 2 の
  attention split fallback（max_rel ≈ 8e-7）で parity が崩れた前例があり、
  exact verify の target parity を割る可能性がある。

## メモリ見積もり（27B, 48 GDN layers）

| 保存物 | 1 token / 1 layer | 15 token 合計 |
| --- | ---: | ---: |
| full state（現行 history） | 3 MiB | 45 MiB（×48 = 2.1 GiB） |
| compact log（δ 24 KiB + k 8 KiB + a 192 B） | 32.2 KiB | 0.48 MiB（×48 = 23 MiB） |
| conv 入力（BF16 shift register） | 20 KiB | 0.29 MiB（×48 = 14 MiB） |

compact log + conv で 40 MiB 級。full state snapshot の 2 GiB 級から大きく削減できる。

## 結論

- compact log 方式は成立する。数学的にも fp32 丸め以内。
- **parity 安全側は逐次 replay**。rank-m GEMM 一括は丸め差が出るため、
  target parity を測ってから判断する。
- これにより full state snapshot（1.5 GiB guard）と rerun（22〜37 ms/round）の
  両方を置き換えられる可能性がある。

## 本番 kernel への実装（2026-09-29）

decode1 kernel（`phaseshift_qwen35_gdn_recurrence_wmma_decode1`）と decode_rows_exact
kernel に compact log の optional spill を追加した。`GdnRecurrenceArgs` の
`compact_delta` / `compact_k` / `compact_a`（+ layer stride）が非 null のとき、各行の
`δ`（`s.delta`）、raw `k`、`a = dval` を書き出す。log は `[state_index][row][...]` で、
`compact_*_layer_stride` で層ごとに分離する。

commit は decode1 と同一の state fragment / WMMA 順序を mirror する専用 kernel
`phaseshift_qwen35_gdn_recurrence_wmma_commit` で行う。全 GDN 層（`num_gdn_states`）を
1 launch でループし、`S ← a_i·S + k_i·δ_i^T` を `compact_rows` 回適用する。

検証 `tests/kernels/optimized/test_gdn_compact_log_commit.hip`（27B geometry、2 層、
rows=16）:

- sequential decode1 で各行・各層を回して spill と参照軌跡を取得
- 各 m で `S_0` から commit し、参照 `S_m` と比較
- 結果: **全 m・全層で bit-exact**

## e2e 統合の試み（未完了）

decoder に `PHASESHIFT_DFLASH2_GDN_COMPACT_COMMIT=1` で compact commit 経路を接続する
実装を試した（conv は既存 history 機構を conv-only で併用、rec は snapshot + commit）。

- 最初の off-by-one（log 行数を `history_rows` で確保し、verify の `total_k+1` 行が
  隣層の row 0 を上書き）を修正し、parity は 0/32 → 16/32 まで改善した。
- しかし **kernel 単体では bit-exact であるにもかかわらず、e2e parity は content 依存で
  約半数失敗**する（json512 / reasoning2048 は合格、code / json2048 / prose2048 は不合格）。
- 原因は未特定。decoder 統合はいったん revert し、kernel 層（spill / commit / 層対応 /
  bit-exact テスト）のみ確定した。

次の調査候補: conv-only history の restore、snapshot 併用時の状態差、verify
Decode1Serial と commit の層インデックス対応。

## 次の作業

1. 本番 recurrence kernel に compact log の optional spill を追加する。→ **完了**（層対応含む）
2. spill した log からの逐次 replay が bit-exact かを確認。→ **完了**（2 層、bit-exact）
3. verify rows を増やして target parity を測る。→ **未完了**（e2e 統合で content 依存の
   残差を確認、原因調査が必要）
4. commit を次回 GDN kernel へ fusion できるか設計する（pending prefix commit）。

## 再現

```bash
cmake --build build-gfx1201 --target test_gdn_compact_commit_poc test_gdn_compact_log_commit
HIP_VISIBLE_DEVICES=0 ./build-gfx1201/tests/test_gdn_compact_commit_poc
HIP_VISIBLE_DEVICES=0 ./build-gfx1201/tests/test_gdn_compact_log_commit
```
