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
rows=16、lossy 既定）:

- sequential decode1 で各行・各層を回して spill と参照軌跡を取得
- 各 m で `S_0` から commit し、参照 `S_m` と比較 → **全 m・全層で bit-exact**
- `launch_gdn_recurrence_f32_wmma_decode1_serial`（rows=16）と
  `launch_gdn_recurrence_f32_wmma_decode_rows_exact`（rows=8）についても、
  それぞれの経路の state と per-row decode1 chain、commit 結果が三者一致（bit-exact）
- verify で使う 3 経路すべてで、層インデックス（`state_index` ↔ log layer）対応が正しい

## e2e 統合の切り分け（2026-09-30）

decoder に `PHASESHIFT_DFLASH2_GDN_COMPACT_COMMIT=1` で compact commit 経路を接続し、
以下の env で切り分けた。

- `..._COMPACT_FULL_HISTORY=1`: conv をフル history（conv+rec）と共有し、rec のみ commit
- `..._COMPACT_USE_HISTORY_REC=1`: compact spill は行うが rec は history から復元（control）
- `..._COMPACT_COMPARE=1`: commit 状態と history 状態を層ごとに比較

correctness（GEN=128、Exact、既定 lossy）での結果:

| 構成 | parity |
| --- | --- |
| control（配線 + spill + snapshot、rec は history） | **32/32** |
| full-history conv + commit | 24/32 |
| conv-only + commit | 24/32 |

- **control が 32/32**: conv の扱い、snapshot 併用、compact view/spill の追加は
  verify の数値結果に影響しない（配線は健全）。
- **full-history と conv-only は同一**: conv をフル history と共有しても結果は変わらず、
  **conv は原因ではない**（候補1の結論）。
- 残る差は **rec commit のみ**。比較診断では commit と history の差は特定の層
  （例: layer 1 / 3 / 43）に局在し、`max_abs` 1e-8〜3e-5、要素数 128〜768。
  大半の accept では bit-exact。
- 一方 kernel 単体（上記テスト）では decode1 / decode1_serial / decode_rows_exact の
  いずれに対しても commit は bit-exact。**実モデルのデータでのみ**微小差が生じる。
- `de0 = __expf(gval−gval)` を spill の k に掛ける実験は逆に差を拡大した（悪化）。

原因は未特定。e2e 統合は revert し、kernel 層（spill / commit / 層対応 / bit-exact
テスト）のみ確定した。

### 残差の切り分け（2026-09-30 追加）

kernel 単体テストを **history capture 有効**（`recurrent_history` を設定）に拡張し、
chain 状態 / history capture / commit の三者比較を追加 → **10/10 PASS**。
すなわち history capture の有無を問わず、kernel 単体では bit-exact。

e2e の比較診断を `(m, n)` 分布まで拡張した結果（correctness GEN=128、rows=8）:

| 位置 | diff | m 一意 | n 一意 | m の値 | n 範囲 |
| --- | ---: | ---: | ---: | --- | --- |
| layer 43 / v_head 16 | 128 | **1** | 128 | 93 | 0..127 |
| layer 1 / v_head 0 | 239 | **1** | 126 | 86 | 0..127 |
| layer 3 / v_head 30 | 768 | **2** | 256 | 18,126 | 0..127 |

- `hd = 128`, `hvv = 128` を確認（27B geometry）。
- 差分は **m（key 次元）方向に 1〜2 行のみ**に集中し、n（value 次元）は全 128 列に広がる。
  rank-1 更新 `S[m][n] = a·S_0[m][n] + k[m]·δ[n]` で、`δ` の誤りは列（n 一意）に、
  `a` の誤りは全要素に現れるため、**k 因子または S_0 の 1 行**が原因であることが確定。
- `de0`（k のスケール）は一様な乗算なので全 128 行に現れるはずで、**原因から除外**。
- **矛盾点**: spill の書き出しアドレス `qk_head*hd + d`、update の読み出しアドレス
  `qk_head*hd + (kt0+ki)*16 + lo`、commit の読み出しアドレスはすべて一致し、
  単一要素だけが食い違うことは構造上起こらない。

未解消の候補:

1. `S_0`（snapshot）と verify 開始状態の差が特定の (layer, v_head, m) 行にだけ存在
2. `compact_k` の 3 つの v_head（同 `qk_head`）による共有アドレスへの書込み競合
3. history buffer と pool state のアドレス重複

次の実験: compact log を v_head ごとに分離（共有書をなくす）して差が消えるか確認。

## 次の作業

1. 本番 recurrence kernel に compact log の optional spill を追加する。→ **完了**（層対応含む）
2. spill した log からの逐次 replay が bit-exact かを確認。→ **完了**
   （decode1 / decode1_serial / decode_rows_exact の 3 経路、全層）
3. verify rows を増やして target parity を測る。→ **未完了**（配線は健全、conv も無関係。
   rec commit の実モデルデータ依存の微小差が残る）
4. 実モデルデータ依存の rec 差の原因究明。候補: `de0` スケーリング、bf16 分解の
   境界値、`dval` の edge case、history capture 同時実行時の codegen。
5. commit を次回 GDN kernel へ fusion できるか設計する（pending prefix commit）。

## 再現

```bash
cmake --build build-gfx1201 --target test_gdn_compact_commit_poc test_gdn_compact_log_commit
HIP_VISIBLE_DEVICES=0 ./build-gfx1201/tests/test_gdn_compact_commit_poc
HIP_VISIBLE_DEVICES=0 ./build-gfx1201/tests/test_gdn_compact_log_commit
```

e2e 切り分け（要: target/draft model、GPU）:

```bash
HIP_VISIBLE_DEVICES=0 PHASESHIFT_TARGET_LM_HEAD_PROXY=0 \
  PHASESHIFT_DFLASH2_GDN_COMPACT_COMMIT=1 \
  PHASESHIFT_DFLASH2_GDN_COMPACT_FULL_HISTORY=1 \
  PHASESHIFT_DFLASH2_GDN_COMPACT_USE_HISTORY_REC=1 \
  PHASESHIFT_NGRAM_TAIL_GATE2_MODE=correctness PHASESHIFT_NGRAM_TAIL_GATE2_GEN=128 \
  ./build-gfx1201/tests/test_dflash2_ngram_tail_gate2
```
（注: 統合コードは未コミット。再現には decoder への再配線が必要。）
