# NgramTail / GDN compact-log commit 作業レポート

対象ブランチ: `poc/ngram-tail-gate1`
最終更新: 2026-09-30

本ファイルは作業中の現況メモであり、確定した contract ではない。数値・採否判断の
正本はそれぞれ `docs/rnd/spec_decode/ngram_tail_gate2.md`、
`docs/rnd/gdn/compact_commit_poc.md` に置く。

---

## 目的

NgramTail Gate 2 の wide-tail 再測定と、ユーザー提案の GDN compact-log commit 方式
（accept prefix の recurrent state を full snapshot / rerun なしで再構成する）の PoC。

compact-log が parity を保てれば、rerun の全廃（round あたり `rerun_ms` の削減）と
history の 2 GiB 級メモリ削減を同時に得られる見込みである。

---

## 前提（確定済みの設計判断）

- 既定 lossy は維持する。exact は `PHASESHIFT_GDN_RECURRENCE_EXACT=1` で opt-in。
- Gate 2 の測定条件: `VerifyNumericMode::Exact`、`PHASESHIFT_TARGET_LM_HEAD_PROXY=0`。
- compact-log の逐次 replay（保存 δ から rank-1 更新を m 回）は **bit-exact**。
  rank-m GEMM 一括は丸め順が変わり非 bit-exact（parity リスク）→ 安全側は逐次 replay。
- compact log の閉形式: `S_m = A_m·S_0 + Σ_{i≤m}(A_m/A_i)·k_i·δ_i^T`、`A_i = Π_{j≤i} a_j`。

---

## 完了（コミット済み）

| commit | 内容 |
| --- | --- |
| `8e0712ae` | PoC v1: fp32 での compact-log 再構成検証 |
| `5d72b199` | 本番 kernel への spill + commit kernel + 1 層 bit-exact テスト |
| `bcb95c17` | 全 GDN 層対応 + `decode_rows_exact` への spill + 2 層 bit-exact テスト |

### PoC v1（数式検証）

- 逐次 replay（保存 δ から rank-1 更新）: **bit-exact**
- rank-m GEMM 一括: 非 bit-exact（max_rel ≈ 1e-7）

### 本番 kernel（`test_gdn_compact_log_commit`）

- decode1 kernel と `decode_rows_exact` kernel に `{δ, raw k, a}` の optional spill を追加。
  log は `[state_index][row][...]` で、`compact_*_layer_stride` により層ごとに分離する。
- `phaseshift_qwen35_gdn_recurrence_wmma_commit` が全 GDN 層を 1 launch でループし、
  decode1 と同一の state fragment / WMMA 順序で `S ← a_i·S + k_i·δ_i^T` を適用する。
- 27B geometry（key_heads=16 / num_v_heads=48 / head dim=128）・2 層・rows=16 で
  **全 m・全層 bit-exact**。

---

## 未完了（e2e 統合）

decoder へ `PHASESHIFT_DFLASH2_GDN_COMPACT_COMMIT=1` で接続する実装を試したが、
**parity が成立しないため revert** した（作業ツリーには残していない）。

### 経過

1. compact log の行数の off-by-one を発見・修正。log を `history_rows`（≤63）で確保して
   いたため、verify の `total_k+1` 行（最大 64）の最終行が隣層の row 0 を上書きしていた。
   `kDFlash2SpecMaxVerifyRows` 行で確保するよう修正 → parity 0/32 → **16/32** まで改善。
2. しかし **kernel 単体では bit-exact であるにもかかわらず、e2e parity が content 依存で
   約半数失敗**する。

| workload | compact parity |
| --- | --- |
| json:512 | 合格 |
| reasoning:2048 | 合格 |
| code:512 / code:2048 | 不合格 |
| json:2048 | 不合格 |
| prose:2048 | 不合格 |

3. 原因は未特定。統合コードは revert し、kernel 層のみ確定した。

### 考察

kernel 単体が bit-exact である以上、差分は次のいずれかに潜んでいる。

- conv-only history の restore（conv は既存 history 機構を併用する設計）
- snapshot 併用時の状態差
- verify `Decode1Serial` の `state_index` と log 層の対応

content 依存の parity 崩壊は、以前の attention split（max_rel ≈ 8e-7 で parity 崩壊）と
同じ「微小な丸め差が near-tie を反転させる」パターンである。

---

## 参考: 速度ポテンシャル

同一条件の smoke（json512 C16、GEN=256）での比較。

| 経路 | tok/s |
| --- | ---: |
| rerun（現行・parity OK） | 69.05 |
| compact（parity NG） | 102.58 |

parity が成立すれば rerun 全廃による明確な高速化が期待できる。

---

## 次の候補

1. conv-only history の restore と snapshot 併用の切り分け
   （conv をフル history と共有し、rec のみ commit にする）。
2. verify `Decode1Serial` の `state_index` と log 層の対応確認。
3. 原因特定後の decoder 統合の再開。
4. commit を次回 GDN kernel へ fusion する設計（pending prefix commit）。

---

## 追記（2026-09-30）: 候補1・2 の切り分け結果

### 候補2: verify kernel の層対応（完了）

`test_gdn_compact_log_commit` を 2 層に拡張し、verify で使う 3 経路すべてで検証した。

- `decode1`（per-row chain）
- `launch_gdn_recurrence_f32_wmma_decode1_serial`（rows=16）
- `launch_gdn_recurrence_f32_wmma_decode_rows_exact`（rows=8）

各経路の state と per-row decode1 chain、commit 結果が三者一致（bit-exact）。
`state_index ∈ [0, num_gdn_states)` は dispatch で検証済みで、log 層との対応も正しい。
テストは 8/8 PASS。

### 候補1: conv 切り分け（完了）

decoder 統合を再適用し、env で切り分けた（`..._COMPACT_FULL_HISTORY` /
`..._COMPACT_USE_HISTORY_REC` / `..._COMPACT_COMPARE`）。correctness（GEN=128）での結果:

| 構成 | parity |
| --- | --- |
| control（配線 + spill + snapshot、rec は history から復元） | **32/32** |
| full-history conv + commit | 24/32 |
| conv-only + commit | 24/32 |

- control が 32/32 のため、conv の扱い・snapshot 併用・spill の追加は verify の数値結果に
  影響しない（配線は健全）。
- full-history と conv-only は同一。**conv は原因ではない**。
- 残差は **rec commit のみ**。比較診断では commit と history の差は特定の層
  （layer 1 / 3 / 43 など）に局在し、`max_abs` 1e-8〜3e-5、要素数 128〜768。
  大半の accept は bit-exact。
- kernel 単体では 3 経路すべて bit-exact だが、**実モデルのデータでのみ**微小差が出る。
  `de0 = __expf(gval−gval)` を spill の k に掛ける実験は悪化した。
- 追加の切り分け: kernel 単体テストを history capture 有効に拡張して三者比較 → **10/10 PASS**。
  e2e の `(m, n)` 分布診断で、差分は **m 方向 1〜2 行のみ**（n は全 128 列）であることを確認。
  → rank-1 の **k 因子または S_0 の 1 行**が原因。`de0` は一様乗算なので除外。
  → ただし spill / update / commit の k アドレスは構造上一致しており、単一要素の食い違いは
  未説明。候補は S_0 の行差・`compact_k` の共有書込み競合・buffer のアドレス重複。

### 状態

- 確定: kernel 層（spill / commit / 層対応 / 三者比較 10/10）と候補1・2。
- **除外済み**: conv、`S_0` snapshot、buffer アドレス重複、k の読み出し時点、`de0`、配線。
- 未確定: key 次元の 1 要素が e2e のみ食い違う原因（kernel 単体の合成データでは再現せず、
  決定論的）。history capture 側の可能性も未排除。
- decoder 統合の診断コード（`PHASESHIFT_DFLASH2_GDN_COMPACT_COMMIT=1` ほか 2 種の env、
  `[gdn-alloc]` / `[s0-check]` / `[compact-compare]` の m/n 分布診断）は**作業ツリーに未コミット**で残置。
- 詳細と次の実験は `docs/rnd/gdn/compact_commit_poc.md` を参照。

---

## 追記（2026-09-30 深夜）: 根因を特定し修正

### 実験1の結果

device で **update が実際に使った k（`k·de0`）と、既存 spill が書いた raw `a.k`** を直接比較した:

```
[kcheck] re-read_vs_used diff = 26880 / 18874368 (0.14%), max_abs ≈ 2.4e-7 ≒ 1 ULP
```

**⇒ log の k が update の実値と一致していなかった。**

### 根因

1. log の k は `qk_head` インデックス（GQA で 3 v_head が共有）だったが、update の実値
   `k·de0` は **v_head ごとに異なる** → 3 v_head が同一アドレスへ別値を書く競合。
2. 既存 spill は raw `a.k` を書くため、`k·de0` と約 1 ULP の差が生じていた。

rank-1 更新の性質上、k の 1 要素の誤りは **key 次元の 1 行 × 全 128 列**に現れるため、
観測されていた `dm=1, dn=128` の形状と完全に一致する。

### 修正と結果

- log の k を `qk_head` → **`v_head`** に分離（3 v_head の書込み競合を解消）、
  update 実値 `k·de0` を `compact_k` に保存（raw `a.k` の再読み出しを廃止）。

| 経路 | 修正前 | 修正後 |
| --- | --- | --- |
| correctness（rows=8, `DecodeRowsExact`） | 24/32 | **32/32 (failed=0)** |
| perf C16（rows=24, `Decode1Serial`） | ~50% | **4/4 (failed=0)** |

両経路とも残差診断はゼロ出力。json512 C16 は **97.25 tok/s**（rerun 69.05 比 **+41%**）。

### 次

1. compact 経路の性能測定（rerun / history との本比較、メモリ削減量）
2. commit の GDN kernel への fusion 設計

---

## 追記（2026-09-30）: K/T sweep と最良パラメータ

main は `68cbf6e0` のまま進んでおらず、ブランチは既に最新 main 上（rebase 不要）。

harness に `PHASESHIFT_NGRAM_TAIL_GATE2_GRID="K:T,..."`（K/T 自由指定）を追加し、
compact 経路（rerun 不使用）で K/T を sweep した。6 workload × 6 設定、
GEN=256 / RUNS=1、**parity 72/72（failed=0）**。
`artifacts/ngram_tail_gate2/kt_sweep/`、詳細は
`docs/rnd/spec_decode/ngram_tail_gate2.md` §6.6。

平均 tok/s:

| K/T | k1_t8 | k3_t8 | k5_t8 | k7_t0 | **k7_t8** | k7_t16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 平均 | 51.3 | 73.4 | 82.9 | 85.9 | **88.7** | 86.6 |

- **K=7 が最良かつ上限**: T=8 一定で K を上げるほど単調改善（51.3→73.4→82.9→88.7）。
  上限は `max_draft_tokens() = block_size - 1`（DFlash2 block_size=8）＝**7**。
- **T=8 が最良**: T=0 は 85.9、**T=8 は 88.7**、T=16 は 86.6。
  T は E/round を増やす（4.435→4.856→4.972）が verify rows も伸び
  （7.9→12.1→14.4）、山は T=8。T=0 比で平均 **+3.3%**、json PP512 は **+10.2%**。
- 推奨は従来どおり **DFlash2 K=7 + NgramTail T=8（N=5、window 2048）**。

ついでに、compact の history を conv-only 化（rec は compact log の責務）して
full history による arena 溢れ（5 設目で 30 GB 超）を解消。
等価性は候補1で確認済みで、correctness 32/32・perf 72/72・既定 32/32・kernel 10/10 を確認。

---

## 再現

```bash
cmake --build build-gfx1201 --target test_gdn_compact_commit_poc test_gdn_compact_log_commit
HIP_VISIBLE_DEVICES=0 ./build-gfx1201/tests/test_gdn_compact_commit_poc
HIP_VISIBLE_DEVICES=0 ./build-gfx1201/tests/test_gdn_compact_log_commit
```

GPU 非搭載環境では `hipGetDeviceCount` 不在時に SKIP (77) を返す。
