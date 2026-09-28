> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# GDN（gated delta net）カーネル最適化履歴

`docs/rnd/optimization_findings.md` から分離した、GDN recurrence / conv / chunked scan / WMMA 数値挙動の実験履歴。見出し番号（4.x / 7.x）は元 dump の通し番号を保持する。

## 収録セクション

- 4 GDN（gated delta net）一般則
- 4.1 27B (v_heads=48) の chunked GDN 不一致（本番は WMMA 版で回避）
- 7.15 GDN recurrence wmma の bf16 staging 損失（既知・受容）
- 7.43 GDN recurrence WMMA の FP32 semantic 化
- 7.51 GDN recurrence の optional lossy モード（4 WMMA -> 1 WMMA）
- 7.83 GDN recurrence decode: rows=1 専用 kernel と DRAM roofline
- 7.84 GDN recurrence decode: 律速点の再判定と dead work / 重複 fragment の除去
- 7.87 GDN recurrence prefill: grid quantization を LDS alias + VGPR 上限で解消
- 7.88 GDN recurrence prefill: VGPR 上限 240 の妥当性検証
- 7.95 gdn_conv1d の行方向 tiling

---

## 4. GDN（gated delta net）

- per-token 逐次 recurrence は state 更新が head_k 深さの依存鎖になり、f32 で ~15 GFLOP/s しか出ない。
- **chunked parallel scan**（chunk 内を並列化、cross-chunk は host が逐次launch）で **3.0〜3.3x**。
  - 構成: inclusive cumsum（`__shfl_up`）/ strict-lower A・L / unit-lower 前進代入 / rank-C state 更新。
- **前提条件: q/k の L2 正規化**。無いと C×C solve が不適条件になり発散する（benchの入力を契約に合わせる必要がある）。
- **multi-accumulator**（内側の縮約を4本に分割）でさらに **1.8x**。ループ運搬依存を短くするのが効く。
- parity: 参照実装と **max_rel 8.3e-7**（chunk=1/8/32、rows=32/128/512 で一致）。
- 並列度: workgroup数 = request × v_heads。**head_v 列の分割は idle thread を作るので逆効果**。
- decode（rows=1）は chunk の旨みがないので sequential のまま、prefill のみ chunked。

---


---

### 4.1 27B (v_heads=48) の chunked GDN 不一致（本番は WMMA 版で回避、chunked 自体は未修正）

> **状態**: 27B が速い GDN 経路を使えない問題は **libr4d 型 WMMA chunk scan への置換で解決**
> （下記「解決」節）。chunked(f32, 8 launch) カーネルは原因を確定できないまま
> **削除した**（runtime からも bench からも消えた。以下は経緯の記録）。
> 未確定のまま残った容疑は「8 launch × 層間 arena 再利用」。

27B（key_heads=16 / num_v_heads=48 / head_k=head_v=128, repeat=3）で
**chunked 経路だけが sequential/参照と不一致**（selector rule を入れていない = 正解経路のまま）。

```text
- bench（同一geometry: nh=48, chunk16, cols128, rows 8..128）は参照と max_rel 9.3e-07 で合格
- モデル経由だと greedy トークンが変わる（chunked 1,2918,2918,... / sequential・参照 1,271,248068,...）
- 層0は 出力・state とも参照と完全一致、層1から乖離
- kGdnPrefillChunkMinRows を上げて sequential(最適化) と比較 → chunked ≠ sequential
  = 数値差ではなくアルゴリズムのデータ依存バグ
- 切り分けに使った計測: args ダンプ(rows/nreq/geometry/strides/requests)、
  |q|/|k| チェック(ともに正常: |k|≈1, |q|=1/√128)、出力・state の要素ダンプ
```

未確認のまま残っている差分: チャンク間 state 引き継ぎ、repeat=3 の qk_head 割り当て、
`max_request_rows` と `req.row_count` の不一致時の挙動。

**追加で判明（原因の在処が変わった）**: chunk ごとに **入力そのもの**をダンプすると
```text
cb=0  v[0]=-0.034668 v[1]=-0.000254 v[100]=-0.012268 k[100]=0.000832
cb=16 v[0]=-0.035645 v[1]= 0.025879 v[100]=-0.002808 k[100]=0.018790
cb=32 v[0]=-0.035645 v[1]= 0.025879 v[100]=-0.002808 k[100]=0.018790  ← cb=16 と完全一致
cb=48 ... 同一 / cb=64 ... 同一 / cb=80 ... 同一
```
アドレス(vptr / c0 / v_row_stride=10240)は正しく進んでいるのに**中身が row16 のまま**。
= recurrence kernel のバグではなく、**chunked 経路では GDN への入力(q/k/v/a/b)が row16 以降で
row16 の値に潰れている**（sequential/参照経路では正しい）。8 launch に分かれることで
初めて露出する pipeline 側（buffer alias / 部分書き込み / dispatch）の疑いが濃い。

**追加（決定的に近い）**: recurrence launch 直後に `hipStreamSynchronize` を入れると**greedy トークンがまた別の列に変わる**（sync 無し `1,2918,...` / sync 有り `248044,16,15,16`）。
= **タイミング依存のレース**。さらに、chunked の層0出力は row0/row127 は参照一致だが
row16/32/48/112 が 0.000118/0.000129/0.000130/0.000130 と収束しており、
参照の **state([0][0]) は -0.003029 に対し chunked は -0.000200**（15倍違う）。
→ 「層1の入力が潰れる」は **層0の chunked 出力自体が既にズレている**ことの伝播。

**bench が検出できない理由**: bench の `--check` は
- 1 回だけ呼ぶ（state をリセットして単発）
- **出力しか比較しない**（state は比較しない）
→ 多層・state持ち越し・レースのどれも踏まない。bench の chunked を
「N 回連続呼び出し + 各回の state も比較」に拡張すれば再現できるはず。

**層0は完全に正しい（参照と一致）**: 層0出力を6点で参照と比較すると完全一致
```text
row0 0.000010189 / row16 0.000118342 / row32 0.000128884 / row48 0.000129687
row112 0.000129804 / row127 0.000129817   ← chunked と参照で同値
```
= 層0の recurrence は出力も state も正しい。にもかかわらず層1の入力が潰れる
→ 差は **chunked の 8 launch だけ**（single stream 順序は保たれている）。
producer(conv/qkvza/rmsnorm...) は両経路で同一なので、残る容疑は
**arena の buffer 再利用（lifetime 解析）が「recurrence = 1 logical op」前提で
層間の面を重ねている**こと。

現状: 27B は recurrence rule 無し（correctness）= 正解。PP128 238 / TG 8.02。

**最終状態（2026-09-13）**: chunked は **4B でも参照と不一致**（`--warmup 0 --tokens 8` で
sequential `1,1,1,1,1,1,1,1` に対し chunked `96227,25,25,313,13,13,13,13`）。
= chunked は**両モデルで未検証のまま本番に入っていた**。
対策として runtime は **sequential(検証済み・参照一致) 固定**にした（`e34e7e3`）。

```text
4B : PP128 845 / PP512 904（chunked 時代は 1792/3380 = -53%/-73%）TG 46.5
27B: PP128 255 / PP512 278（correctness 時代は 238/256 = +7..9%）TG 8.22
```

bench は以下を振っても再現しなかった（すべて OK）:
chunk16/cols128、key16/value48 と 32、rows 16..128、state ランダム/ゼロ、
q を 1/√head_dim スケール、v の行ストライドを 10240 にパディング。
→ bench とモデルの差は **入力データそのもの**。次の一手は
「モデルの実 GDN 入力を1層分そのままダンプして bench でリプレイ」。
（hipfire の一次実装 `gated_delta_net_f32_chunked.hip` とは cumsum / decay mask /
forward substitution / state carry まで一致を確認済み。expf は本家も我々も expf。）

次の一手: 層1の **qkvza GEMM 直後**と **conv1d 直後**で同領域をダンプし、どの段で既に潰れているかを見る。

**さらに判明（recurrence kernel はシロ）**: recurrence dispatch の**先頭**（launch 前）で
ホストに読み戻して確認すると、
```text
chunked : row0 -0.034668 / row16..112 -0.035645（同一）  ← 既に潰れている
sequential: row0 -0.005096 / row16 0.000507 / row32 0.000542 / row48 0.000610 / row112 0.000515（全部違う）
```
= **launch 前の時点で既に潰れている**。層0の chunked 出力(row0..112)自体は正常
(0.000010/0.000118/0.000129/0.000130/0.000130)。
ポインタを見ると GDN の作業バッファは**層ごとに回転して再利用**されている:
```text
si=0: q=...be0b1800 k=...bdfb1800 v=...bdbb3800 out=...be1b1800
si=1: q=...bdbb1800 k=...be0b1800 v=...be1b3800 out=...be7b1800   ← si=1 の q が si=0 の v 領域
```
→ **chunked の 8 launch と「層をまたぐ作業バッファ再利用」の相互作用**（stream 順序 or
alias）が本命。recurrence kernel 内のロジックではない。
**alias の実体（ポインタで確定）**:
```text
si=0 out = 0x...be1b1800 (stride 6144 floats, 128行 → 0x...be1b1800 + 0x300000)
si=1 v   = 0x...be1b3800   ← si=0 の out バッファの内側 (+0x2000)
```
つまり **層1の v 入力は層0の出力バッファに重なっている**（layer0 out の consumer が終わった後に
qkvza がその領域を再利用する設計）。順序は chunked/sequential で同じなので、
**8 launch 化で stream 順序 / 依存追跡が崩れている**のが疑わしい。
次の一手: 層1の **qkvza GEMM 直後**で同領域をダンプし、そこで既に潰れているかを見る
（そうなら GEMM/その dispatch、違えば GEMM の後段〜recurrence の間）。

→ 27B では rule を足さず correctness 経路（PP128 238 tok/s。rule を足すと 393 だが不正解）。

**解決（2026-09-13）: libr4d 型 WMMA chunk scan に置換して解決した。**

chunked 経路の欠陥（8 launch + 層をまたぐ arena 再利用の相互作用）を回避するため、
**1 launch・state を register 保持**のカーネルに置き換えた
（`kernels/gdn/gdn_recurrence_wmma.gfx1201.hip`）。
selector は 2 値（`Correctness` / `Optimized`）のままで、`Optimized` の内部戦略として
head_k=128 かつ head_v%64==0 なら WMMA、それ以外は f32 sequential を dispatch が選ぶ。
libr4d の `r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip` の構造（block が sequence 全体を
chunk loop し、state を WMMA accumulator に持ち、LDS に bf16 staging）を、
我々の検証済み math（expG / beta / D / forward substitution 内包）に載せた。

```text
形状: head_k=head_v=128 固定, BT=32, BV=64, 256 threads, grid=requests*nvh*(hv/64)
per chunk: U=K@S, O=Q@S, P=Q@K^T, PKK=K@K^T, O+=Mkq@delta, S=eg*S+K^T@delta
state: f32 register 保持 / bf16 は operand staging のみ
```

decay は D[i][j]=expf(G_i-G_j) を **f32 で直接**計算する（libr4d の cref 中点分割は
span>176 で bf16 の subnormal に落ちて 1.13 相対誤差を出したため不採用）。

実測（bench `gdn-recurrence`, rows=128, nr=1）:

| variant | 4B形状(nvh=32) | 27B形状(nvh=48) |
|---|---|---|
| optimized(sequential f32) | 4397 us | 4799 us |
| chunked(f32, 16 launch) | 2820 us | 4937 us |
| **wmma(bf16, 1 launch)** | **84 us** | **91 us** |

モデル実測（トークンは sequential と完全一致、決定的）:

```text
4B : PP128 2925 / PP512 3102 / TG 48.5   （sequential 時代 845/904/46.5 = +246%）
27B: PP128  470 / PP512  540 / TG 8.33   （correctness 時代 255/278/8.22 = +84%）
GREEDY 4B : 1,1,1,1,1,248046,198,248045       （sequential と同一）
GREEDY 27B: 271,248068,198,760,1156,682,3106,264（sequential と同一）
```

注意: **bench `--check` は bf16 では 1e-3 絶対許容を満たさない**（大きな要素の相対誤差は
0.2%=2^-9 だが、打ち消し合う小さい要素では相対誤差が 100% を超える）。
gross bug を検出できる粗い許容（max|ref| の 5%）を wmma variant のみに適用している。
本カーネルの正しさは **モデルの greedy トークン一致**で担保する。


---

## 7.15 GDN recurrence wmma の bf16 staging 損失（既知・受容）

`cli:interactive:turn2` が oracle（HF fp32 greedy）と一致しない件。結論は
**最適化パスの数値損失。文脈は壊れず、言い回しだけが変わる。受容する。**

### 切り分け

`optimized_dispatch` に per-kernel-id の参照パス落ちカウンタを足して bisect した結果、
`GDN_RECURRENCE` を参照実装に落とすと oracle と一致する。つまり
`gdn_recurrence_wmma.gfx1201.hip`（libr4d 移植、1 launch / state register 保持）が原因。

### 数値

同一 prompt（36 token, 2 chunk）の最初のサンプリング行の logits を fp32 参照と比較:

| 経路 | max|Δ| | mean|Δ| | rel | 生成 |
|---|---|---|---|---|
| correctness（fp32 逐次） | — | — | 0 | `27775,383` = "Based on"（oracle） |
| f32 sequential（optimized の非 wmma 経路） | 0.99 | 0.157 | 3.7% | `27775` ✓ |
| wmma chunked | 2.83 | 0.279 | 10.7% | `7525` ✗ |

誤差は **chunk 数ではなく chunk 内の行数** で増える:

| prompt 長 | chunk 数 | rel |
|---|---|---|
| 14 | 1 | 2.2% |
| 32 | 1 | 11.1% |
| 36 | 2 | 10.7% |

wmma の operand である `qbf / kbf / Sbf / Abf / Vbf` は bf16 staging 必須なので、
chunk を 64 にしても（36 行が 1 chunk になるだけ）改善しない見込み。
三角求解の RHS `Wb` を bf16→fp32 にした実験は 10.7%→10.2% で無効だった。

### 速度（wmma は外せない）

`phaseshift-bench gdn-recurrence`:

| rows=96 nr=1 | wmma | f32 sequential |
|---|---|---|
| p50 | 45.3us | 3366us |

**74x**。recurrence は 27B prefill の 3.0%（96ms/3190ms）なので、f32 に落とすと
prefill が 3.2s → 10s 級になる。f32 sequential は選択肢にならない。

### 実際の壊れ方（受容の根拠）

同一 prompt を 16 token 生成:

```text
wmma    : "Your favorite color is **blue**.\n"
fp32    : "Based on what you just told me, your favorite color is **blue**."
```

質問 "What is my favorite color?" に対し**両方とも blue と正答**。落ちているのは
1 トークン目の言い回し（"Based on…" vs "Your…"）と長さだけで、**文脈は保持されている**。
libr4d も同種の劣化を clamp(80) で緩和しており（自身の doc で "attenuated, not exact"）、
lossy 許容は同じ設計判断。

e2e は `tests/e2e/test_apps_qwen35_4b.py` の `KNOWN_LOSSY` に登録し、
gate は通すが `[KNOWN-LOSSY]` として理由付きで必ず出力する。

### 未着手の改善候補（やるなら）

```text
(a) state 更新だけ fp32 FMA に戻す      S_out = D31·S + Σ_t (de_t·k_t) ⊗ δ_t
    （文脈の運び手なので支配項の可能性が高い。262k FMA/block/chunk ≒ 1000 FMA/thread）
(b) 出力側だけ fp32 FMA に戻す
(c) state を fp32 で持ち、wmma operand を都度 bf16 化（丸め位置は同じなので効果薄の見込み）
```


---

## 7.43 GDN recurrence WMMA の FP32 semantic 化（BF16 2-way 4-term、gfx1151/gfx1201）（2026-09-17）

`docs/now` の仕様に沿い、GDN recurrence wmma を
「semantic 状態は FP32、BF16 は WMMA operand 搬送用の分解形式のみ」で書き直した。

### 数値設計

- BF16 2-way 分解: `x0 = bf16(x); x1 = bf16(x - f32(x0))`、全行列積を
  `Ah·Bh + Ah·Bl + Al·Bh + Al·Bl` の 4-term、accumulator はすべて fp32 登録。
- semantic FP32 のまま保持: Q/K（FP32 入力から fragment を都度生成＝staging ゼロ）、
  S（Sacc fp32 レジスタ + チャンク可視コピー）、A/L、W、delta、G/de/bt/oe/eg、de·K。
- 廃止: Sbf/Abf/Wb/Vbf の bf16 semantic staging（旧実装は状態そのものを bf16 に丸めていた）。
- K·Kᵀ（PK）の tile バグ修正: 旧実装は `pk_acc = wmma(knt, knt)` で A 側に pnt tile を使用し、
  pmt≠pnt の warp で L の行 tile が誤答だった（新実装は (aki=pmt, akj=pnt) を正しく分離）。

### fragment 生成コストが支配的

- 2-way 分解は要素あたり約 7 VALU 命令（bf16 丸め×2, 復元, 減算）。
  1 chunk/warp あたり ~1,300 要素の分解で issue-bound になる（WMMA 自体は 2.5% しかない）。
- 3-term（LL drop）実測: rows=96 nr=1 で 302→295us（2.5%）のみ。4-term を保持。
- cvt_pk_bf16 系の packed 変換 builtin は ROCm 7.15 に露出していない。

### P/PK と U/O のループ統合

- warp 割りを `mt = warp&1, nt = warp>>1` に再配置し、warp 0-3 の P/PK ループを
  U/O ループへ統合。Q/K 行 fragment（aq/ak）を共有し、decomp 作業量 ~22% 削減。
- rows=96 nr=1 で 301.8→236.6us（-22%）。全 8 warp が (mt, nt) ⊂ {0,1}×{0..3} の bijection。

### LDS vs L2（semantic scratch 配置）

| 配置 | LDS | rows=96 nr=1 | nr=8 | 備考 |
|---|---|---|---|---|
| S を LDS (61.4KB, 1 WG/CU) | 60.4KB | 231.8us | 538us | 採用 |
| S を recurrent_state 直接 (26.6KB) | 26.6KB | 400us | 842 | 負け |

- L2 側は `__launch_bounds__(256,2)` 強制で VGPR spill 207（Sacc 32 + solve dlt 32 +
  fragment でレジスタ上限超過）。2 WG/CU の residency が VGPR 壁で実現しないため、
  L2 配置は LDS より 1.6-1.7x 遅い。**VGPR 255-256 が issue-bound 状態の本当の上限**であり、
  residency 増強はVGPR 削減とセットでないと成立しない。

### リソース / 実測

- gfx1151: VGPR 256, spill 0, SGPR 68, LDS 61,444B（1 WG/CU）。
- gfx1201: VGPR 256, spill 2, LDS 61,444B。
- 精度（f32 sequential 参照）: output max_rel ≤ 1e-4、state max_rel ≤ 2e-4
  （旧 f16 WMMA は logit rel ~10.7% class）。正規化 Q/K 入力でのテスト 20 case 全 PASS。


---

## 7.51 GDN recurrence の optional lossy モード（4 WMMA → 1 WMMA）（2026-09-19）

### 目的

27B prefill の GDN recurrence は 179.6ms（全 kernel 時間 1016ms の 17.7%、PP の第 2 の
コスト）であり、`gdn_wmma2` が bf16 hi/lo 分解の 4 項（hh / hl / lh / ll）を
WMMA 4 回で計算している。cross 項と lo 項を落として **hh のみ 1 回**にし、
精度を犠牲にして時間を取る optional モードを追加した。

### 実装

- `PHASESHIFT_GDN_RECURRENCE_LOSSY=1` で有効化（既定 off、env-gated。
  `PHASESHIFT_QWEN35_KERNEL_MODE` 等と同じ流儀）。無効時は完全に従来経路。
- `gdn_decomp8<kLossy>`: lossy では lo 側の `cvt/sub/cvt` を計算しない。
- `gdn_wmma2<kLossy>`: lossy では `a.h × b.h` の 1 回のみ。
- 対象は gfx1201 のみ（gfx1200 の recurrence は別実装のため未対応）。

### 速度

`phaseshift-bench gdn-recurrence --variant optimized`（27B 幾何 kh=16 nvh=48 hk=128 hv=128）:

| rows / nr | exact p50 | lossy p50 | Δ |
|---|---|---|---|
| 512 / 1 | 731.9us | 526.8us | **−28.0%** |
| 2048 / 1 | 2845.4us | 2188.9us | **−23.1%** |

e2e PP（27B-PSQ、prompt 2048、3 ラウンド交互、per-invocation min）:

| mode | gpu_ms | tok/s |
|---|---|---|
| exact | 1050.5 / 1054.1 / 1057.2 | 1947 |
| lossy | 1014.0 / 1015.4 / 1020.2 | 2019 |

**−3.5%（+3.7% tok/s）**。recurrence 179.6ms の −23% ≒ −41ms と整合。

### 精度

`phaseshift-bench gdn-recurrence --check`（correctness reference 比）:

| mode | mismatches | max_rel | meanabs | state_max |
|---|---|---|---|---|
| exact | 0 | 5.4e-05 | 1.2e-07 | 1.3e-04 |
| lossy | 0 | **4.5e-02** | 8.1e-05 | **8.7e-02** |

出力の最大相対誤差は bf16 入力丸め相当（4.5%）まで増え、recurrent state の差は
8.7e-2（exact 1.3e-4）。ただし check 閾値は両者とも通る。

e2e 生成（4B、3 プロンプト、96 token）は exact / lossy で**完全一致**
（"Your favorite color is blue." / 1..20 列挙 / 平均速度の手順）。

### 決定・運用（2026-09-19 更新: 既定 lossy へ）

- **既定 lossy**（`PHASESHIFT_GDN_RECURRENCE_EXACT=1` で exact に戻す。
  `PHASESHIFT_GDN_RECURRENCE_LOSSY=0` でも off）。
- `test_gdn_recurrence` の受入基準をモード依存にした:
  lossy 時は out rel tol 2e-2 / state rel tol 1e-2（exact 時は従来の 2e-4 / 1e-3）。
  出力に `mode=lossy|exact` を表示し、どちらの基準で判定したか追跡できる。
  両モードとも PASS を確認済み。
- 有効化した状態で e2e gate を通す場合は `KNOWN_LOSSY` として理由付きで扱う
  （`tests/e2e/test_apps_qwen35_4b.py` の既存運用に合わせる）。
- さらなる候補（§7.15 の未着手案と同系統）: state 更新のみ fp32 FMA に戻す、
  出力側のみ fp32 に戻す、など。今回は hh のみ 1 WMMA の最小構成に留めた。


---

## 7.83 GDN recurrence decode: rows=1 専用 kernel と DRAM roofline（2026-09-21）

§7.82 で `gdn_recurrence` が pp の 13.1%（48 dispatch × 2717 µs）と判明した。
本節は decode 側の GDN recurrence（48 layer）を、PSQ4 decode と同じ手順
（bit-exact を維持したまま 1 Phase ずつ A/B）で最適化した記録である。

### 背景

現行の chunked kernel `gdn_recurrence_wmma_impl<true, false>` は `kWt=32` 行の
chunk を前提に、prefix scan + `A/L/W/dT` staging + 32 段の triangular solve を持つ。
decode（`nval=1`）では有効な行が 1 行しかないため solve は `L[0][0]=0` に縮退し、
演算の大半が無駄になる。ISA 8141 命令のうち `v_fma_f32` が 508（solve 由来）。

### Phase 1: rows=1 専用 decode1 kernel

`gdn_recurrence_wmma_decode1_lossy_impl<true>` を追加し、dispatch を
`max_request_rows <= 1 && head_k == 128 && head_v % 64 == 0` で切り替えた。
chunked kernel は prefill 用に残す。decode の代数は:

```
beta  = 1/(1+exp(-b))
g     = -exp(a_log) * log(1+exp(a + dt_bias))
d     = exp(g),  de0 = exp(g-g)
delta = beta * (v - d*u)
out   = d*o + A00*delta
S_new = d*S + k^T * delta
```

bit-exact のため `jt=1`（t=16..31）と `tstep=1` の零 fragment WMMA は残置した。

ISA（chunked → decode1）:

| | chunked | decode1 |
| --- | ---: | ---: |
| 命令 | 8141 | **5412** |
| 分岐 | 379 | 299 |
| VGPR | 256 | **169** |
| SGPR | 59 | 48 |
| spill / private | 2 / 12 B | **0 / 0** |
| LDS | 61444 B | 35076 B |
| WMMA | 42 | 34 |
| `v_fma_f32` | 508 | **3** |
| barrier | 7 | 2 |
| `ds_load` / `ds_store` | 306 / 94 | **36 / 18** |
| `s_delay_alu` | 588 | 224 |

bit-exact oracle `test_gdn_recurrence_decode1` を追加（nreq 1/2/4/8 ×
steps 1/2/16/128 + 零入力 case、output と state を各 step で語単位比較）:
**20/20 PASS**。

性能（p50 µs、warmup 100 / samples 200 / launches 20、`--device 2`）:

| nreq | chunked | **decode1** | 倍率 |
| ---: | ---: | ---: | ---: |
| 1 | 34.85 | **14.70** | 2.37x |
| 2 | 50.60 | **24.89** | 2.03x |
| 4 | 95.43 | **43.50** | 2.19x |
| 8 | 180.66 | **81.30** | 2.22x |

E2E（tg128、context 2048）: 38.22 → **36.97 / 37.00 / 37.03 ms/token**
（26.16 → 27.05 t/s、**-1.25 ms/token**）。`GREEDY_TOKEN_SUM=2649738` 一致、
fallback 0、acceptance 97/97。device 2 での再確認は 36.64 ms/token（27.30 t/s）。

rocprofv3 130 step trace: `gdn_recurrence_wmma_decode1_lossy_impl<true>` n=6240
（= 48 layer × 130 step）、avg 16.41 µs = **0.788 ms/token**。

### Phase 2: 128 threads / warp-local ownership → 棄却

256 → 128 threads にし、warp `w` が v 列 `[w*16, w*16+16)` と全 128 k 行を所有、
per-warp LDS scratch `S[4][128][16]`（= 32768 B、2 blocks/WGP）、`p` / `delta` /
`a00` を warp-local 化して barrier を 0 にする 2 variant を実装した。bit-exact は
40/40 PASS だが **全 shape で decode1 より遅い**:

| nreq | decode1 (256 thr) | x4 LDS 版 | x4 register 版 |
| ---: | ---: | ---: | ---: |
| 1 | 14.70 | 16.07 (+9.0%) | 16.35 (+10.9%) |
| 2 | 24.76 | 29.14 (+17.7%) | — |
| 4 | 51.04 | 59.32 (+16.2%) | 57.25 (+12.2%) |
| 8 | 98.65 | 120.23 (+21.9%) | 116.89 (+18.5%) |

原因は 128 threads 化で **warps/MP が半減**すること。256 thr × 1 block/WGP と
128 thr × 2 blocks/WGP はどちらも 8 warps/MP（= 1 warp/SIMD）で、後者は per-warp
の仕事量と LDS traffic だけが増える。

### Phase 3: LDS ≤ 32768 B（2 blocks/WGP）→ 棄却

`S[128][64] = 32768` ちょうどにし、`delta` / `a00` を register へ移す
（row-split ownership: warp が 64 行 × 16 列を所有）ことで 2 blocks/WGP を
達成した。ISA は 5238 命令 / VGPR 167 / spill 0、bit-exact 20/20 PASS。しかし:

| nreq | Phase 1 | **Phase 3** | 差 |
| ---: | ---: | ---: | ---: |
| 1 | 14.70 | 15.56 | +5.9% |
| 2 | 24.89 | 26.49 | +6.4% |
| 4 | 43.50 | 47.47 | +9.1% |
| 8 | 81.30 | 90.71 | +11.6% |

**LDS 32768 を達成したのに悪化**した。そこで `kSp` のみを変えて LDS 容量を
強制変更し（数値は不正、タイミングのみ。pitch は帯域・bank 挙動が同一になる値を
選択）、occupancy の効果を直接測った:

| LDS (B) | blocks/WGP | nreq=1 | nreq=4 | nreq=8 |
| ---: | ---: | ---: | ---: | ---: |
| 35076 | 1 | 14.70 | 43.50 | 81.30 |
| 32004 | 2 | 14.73 | 43.57 | 81.00 |
| 16644 | 3 | 14.64 | 43.66 | 80.39 |
| 8452 | 4+ | 14.83 | 43.54 | 79.82 |

**完全にフラット**。1 → 4+ blocks/WGP まで上げても一切変わらない。
**カーネルは occupancy 律速ではない。**

### 律速は DRAM 帯域 → **§7.84 で訂正**

> この節の「DRAM roofline 到達」という判定は誤りだった。standalone bench の
> state 3 MiB は 200 launch のあいだ L2 に残るため、ここで測っている 619 GB/s は
> DRAM ではなく L2/L3 の帯域である。§7.84 を参照。

state は request あたり 3 MiB、decode 1 回で read + write = 6 MiB。

| nreq | 時間 | state traffic | 実効帯域 | 636 GB/s 比 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14.70 µs | 6.0 MiB | 428 GB/s | 67% |
| 2 | 24.89 µs | 12.0 MiB | 506 GB/s | 80% |
| 4 | 43.50 µs | 24.0 MiB | 578 GB/s | 91% |
| 8 | 81.30 µs | 48.0 MiB | **619 GB/s** | **97%** |

nreq=8 は実測 streaming peak 636 GB/s の **97%**。**nreq ≥ 4 は DRAM roofline に
到達済み**で、LDS / pitch / occupancy / WMMA / ALU の最適化は原理的に効かない。
残るレバーは traffic 削減のみだが、state は本質的に read + write が必須である。

nreq=1 は 67% で余地があるが、それが occupancy でないことは上表で実証済み。
read → compute → write の依存鎖と block 分割が候補として残る。

### 測定上の注意（汚染）

GPU0 が別プロセス（`/tmp/opencode/coresid`）に占有されており、device 0 の
nreq ≥ 4 測定は **15〜21% 大きく出る**:

| nreq | device 0 | device 2（クリーン） | 差 |
| ---: | ---: | ---: | ---: |
| 1 | 14.75 | 14.70 | ±0% |
| 2 | 24.76 | 24.89 | ±0% |
| 4 | 51.04 | 43.50 | +17% |
| 8 | 98.65 | 81.30 | +21% |

device 2 の再現性は 0.3% 以内（3 reps）。**GDN の性能値は `--device 2` で取得して
いる。** なお Phase 2 / 3 の nreq=1 / 2 は汚染の影響がなく（±0%）、棄却判断は
有効である。

### 教訓

1. decode 特化 kernel の効き目は大きい（chunked の solve を持たないだけで
   2.0〜2.4x）。ISA 命令数 -34%、`v_fma_f32` 508 → 3。
2. しかし本カーネルは **DRAM 帯域律速**であり、occupancy を 1 → 4+ blocks/WGP に
   上げても 1% も変わらない。PSQ4 decode（K-chain レイテンシ律速）とは律速点が
   異なる。
3. 並行 workload のある GPU では **nreq ≥ 4 の測定が 15〜21% ずれる**。
   A/B は同一 device で行い、device を明記する。

### 再現方法

```bash
./build/phaseshift-bench gdn-recurrence --device 2 --variant optimized \
  --rows 8 --nreq 8 --key-heads 16 --value-heads 48 --head-k 128 --head-v 128 \
  --warmup 100 --samples 200 --launches 20
./build/phaseshift-bench gdn-recurrence --device 2 --variant optimized-decode1 \
  --rows 8 --nreq 8 --key-heads 16 --value-heads 48 --head-k 128 --head-v 128 \
  --warmup 100 --samples 200 --launches 20
./build/tests/test_gdn_recurrence_decode1
```

---


---

## 7.84 GDN recurrence decode: 律速点の再判定と dead work / 重複 fragment の除去（2026-09-21）

§7.83 で nreq=1 は DRAM 帯域の 67% と報告したが、その後の追加調査で
**律速点の判定が誤っていた**ことが判明した。nreq=1 を引き上げるための調査記録。

### 判定の誤り: bench の state は L2 に残る

standalone bench は同一 state（3 MiB）を 200 回 launch する。L2 は 8 MB なので
state は L2 に常駐し、DRAM には落ちない。マイクロベンチ（同一 ownership で
state を read/write するだけの kernel、`--device 2`、200 launch）:

| kernel | 時間 | launch 差引後 | 実効帯域 |
| --- | ---: | ---: | ---: |
| null 96 blocks | 2.48 µs | — | — |
| copy nreq=1（6 MiB） | 5.39 µs | 2.91 µs | 2.2 TB/s |
| read nreq=1（3 MiB） | 4.26 µs | 1.78 µs | 1.8 TB/s |
| copy nreq=8（48 MiB, L3 域） | 17.35 µs | 14.9 µs | 2.9 TB/s |

decode1 は 14.70 µs なので、**メモリは 2.9 µs 分しか使っていない**。
§7.83 で 619 GB/s を「DRAM peak の 97%」と読んだのは L2/L3 帯域の誤読だった。
なお本番モデルでは 48 layer × 3 MiB = 144 MiB が L3 を超えるため layer 間で
追い出され、そちらは DRAM 律速に近い（per layer 6 MiB / 636 GB/s = 9.9 µs、
実測 16.41 µs）。

### occupancy はレバーではない（確定）

rocprofv3 の kernel-trace が `LDS_Block_Size` / `VGPR_Count` を出すので、
実際の occupancy を確認できる（`--rows 1 --nreq 1`）:

| kSp | LDS (B) | VGPR | blocks/WGP | warps/WGP | nreq=1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 68 | 35328 | 224 | 1 | 8 | 14.70 µs |
| 32 | 16896 | 176 | 3 | 24 | 15.25 µs |
| 16 | 8704 | 176 | 4 | 32 | 15.42 µs |

warps/WGP を 8 → 32（1 → 4 warp/SIMD）にしても**わずかに悪化**する。
latency 律速でも帯域律速でもなく、issue と共有ユニットの throughput 律速である。
§7.83 の「LDS を下げても 1% も変わらない」は正しかったが、理由は DRAM
roofline ではなくこれである。

### Phase 4: dead work の除去

decode1 は `max_request_rows <= 1` 専用なので、`mt=1` の warp は絶対に出力に
寄与しない（`s.delta` / `s.a00` の書き込みは `mt == 0` と warp 0 で guard 済み）。
それでも全 8 warp が u/o/p loop と `A00*delta` WMMA を実行していた。

u/o/p loop・出力 WMMA を `mt == 0` で、`p` を warp 0 のみで実行するようにした。
`mt=1` の仕事（state staging と S update）は不変で、**bit-exact 20/20 PASS**。

| nreq | §7.83 | **Phase 4** | delta |
| ---: | ---: | ---: | ---: |
| 1 | 14.74 | **14.41** | **-2.2%** |
| 2 | 24.89 | **24.22** | **-2.7%** |

ISA は 5412 → 5445 命令（guard の分岐が増える）にもかかわらず高速化する。
**利得は命令数ではなく、block の global load と WMMA が半分になること**から来る。

### Phase 5: `akj` の再利用 → 追加で -12〜-15%

`p_acc`（A00 を作る）は `kpj` から `jok` predicate 付きで専用の `kjtmp`
fragment を組んでいた。しかし `p_acc[0]` を消費するのは **warp 0 だけ**
（`ppk` guard）で、その warp では 2 つの predicate が同じ条件になる:

```
ti = mt*16 + lo,  mt = 0  →  iok = (lo == 0)
tj = nt*16 + lo,  nt = 0  →  jok = (lo == 0)
```

つまり `akj` は `ak` と **bit 一致**する。`ak` をそのまま B operand として
再利用した（`p_acc = wmma2(p_acc, aq, ak)`）。`tj` / `jok` / `kpj` のアドレス
計算も消え、compiler が大きく簡約する:

| | Phase 4 | **Phase 5** |
| --- | ---: | ---: |
| 命令 | 5445 | **3917** |
| `global_load_b32` | 226 | **162** |
| bf16 変換ブロック | ~1538 | ~1162 |

| nreq | Phase 4 | **Phase 5** | §7.83 比 |
| ---: | ---: | ---: | ---: |
| 1 | 14.41 | **12.66** | **-14.1%** |
| 2 | 24.22 | **20.64** | **-17.1%** |

bit-exact **20/20 PASS**。E2E は 36.60 / 36.68 ms/token（27.32 / 27.27 t/s）で
ノイズ内（GDN は 0.79 ms/token しかないため）。

### Phase 6: k/q fragment の LDS 共有 → さらに -5.8% / -16.6%

`bf16` 変換の重複を狙った。`ak` / `aq` は `(lane, kk)` だけで決まり
（`ak` = `k[row][kk*16 .. +16]` を `lo == 0` でマスク、`aq` は `q` の同じ位置）、
**4 個の mt=0 warp が全く同じ 16 変換/kk を実行**していた。RN 変換は
`gfx1201` に packed 命令が無く 6 命令/要素なので、ここは重複が直接コストになる。

`kk` ごとに 1 回だけ変換する方式にした: warp `w` が `kk = w` を担当して
2 fragment を LDS（8 要素 × 2 B = 16 B/lane/fragment）に書き、mt=0 warp が
kk loop で読む。S staging の既存 `__syncthreads()` がそのまま hand-off を
保証するので **barrier は増えていない**。

| | Phase 5 | **Phase 6** |
| --- | ---: | ---: |
| 命令 | 3917 | **1716** |
| `global_load_b32` | 162 | **50** |
| bf16 変換ブロック | ~1162 | **~411** |
| LDS | 35076 B | 51460 B |

| nreq | Phase 5 | **Phase 6** | §7.83 比 |
| ---: | ---: | ---: | ---: |
| 1 | 12.66 | **11.93** | **-19.1%** |
| 2 | 20.64 | **17.21** | **-30.8%** |

bit-exact **20/20 PASS**。E2E は 36.60 → **36.52 ms/token**（27.32 → 27.39 t/s）。

### 残っている候補

1. **bf16 変換はまだ ~411 命令**（1716 の 24%）。`bs`（S operand）は warp ごとに
   異なる列なので共有できない。`ak` / `aq` は Phase 6 で共有済み。
2. `__float22bfloat162_rn` による packed bf16 化は逆効果だった
   （5412 → 5611 命令）。compiler は pair 変換を使わず move が増える。
3. `__shfl` で A operand を組む案は、Phase 5 で load が減ったため
   **shfl の追加分の方が大きくなり不成立**。
4. 128 threads 化（§7.83 Phase 2）と LDS ≤ 32768（同 Phase 3）は再確認しても
   棄却のまま。Phase 6 で LDS は 51460 B に増えたが、occupancy がレバーでない
   ことは §7.84 で確定しているため影響しない。

### 測定環境について

device 2 でも decode1 の nreq ≥ 4 は 43.5 → 48.8 µs と 12% ドリフトする
（同一時間帯の chunked kernel は 95.4 → 95.1 µs で安定）。co-tenant の L2/L3
汚染が decode1 だけに効いていると見られる。**decode1 の A/B は nreq=1/2
（再現性 0.5%）で判断する。**

### 再現方法

```bash
# occupancy の実測
rocprofv3 --kernel-trace -f csv -o /tmp/kt -- \
  ./build/phaseshift-bench gdn-recurrence --device 2 --variant optimized-decode1 \
  --rows 1 --nreq 1 --key-heads 16 --value-heads 48 --head-k 128 --head-v 128 \
  --warmup 5 --samples 10 --launches 5
```


---

## 7.87 GDN recurrence prefill: grid quantization を LDS alias + VGPR 上限で解消（2026-09-21）

§7.82 で `gdn_recurrence_wmma_impl` が pp の 13.1%（48 dispatch × 2717 µs）と判明した。
§7.83/§7.84 は decode 側（rows=1）を扱っており、**prefill 側（rows=2048、nreq=1）は
未着手**だった。本節はその律速を特定し、**-42%** まで縮めた記録である。

### 律速: grid の端数（grid quantization）

| | 値 |
| --- | --- |
| grid | **96 blocks** = `num_requests × num_v_heads × vgroups` = 1 × 48 × 2（**rows に依存しない**） |
| block | 256 threads（8 warp） |
| LDS | **61952 B** → `131072 / 61952 = 2.11` → **2 blocks/MPU = 64 並列スロット** |
| 帰結 | 96 ÷ 64 = **1.5 ラウンド → 稼働率 75%** |

同じ 2048 rows を request 数だけ変えて分割した実測（`--rows 2048 --nreq N`）。
grid は nreq に比例して増えるので、ラウンド数だけが変わる:

| nreq | blocks | ラウンド | p50 | ktok/s |
| ---: | ---: | ---: | ---: | ---: |
| **1** | 96 | 1.5 | **2319 µs** | 883 |
| 2 | 192 | 3（ちょうど） | **1863 µs** | 1099 |
| 4 | 384 | 6（ちょうど） | 1892 µs | 1083 |
| 8 | 768 | 12（ちょうど） | 1909 µs | 1073 |

**nreq=1 は grid の端数で 19.7% 損している。** DRAM でも compute でもない
（state 3 MiB の traffic は 2.6 GB/s、bf16 WMMA は peak の数 % しか使っていない）。

### 診断法: nreq=1 と nreq=2 の時間比

同時 resident blocks を `R`、1 block の処理時間を `T` とすると:

```text
R=2 (64 スロット): nreq=1 は 96/64 = 1.5 → 2T、nreq=2 は 3 ラウンド × T/2 = 1.5T  → 比 1.33
R=3 (96 スロット): nreq=1 は 96/96 = 1   → 1T、nreq=2 は 2 ラウンド × T/2 = 1T    → 比 1.00
```

実測比がそのまま `R` の判定になる。base は **1.259**（R=2）。

### 修正 1: `S` と `A/L/W/dT` の alias

LDS の中身（`recurrence.hip` の `gdngfx12` namespace）:

```text
ChunkSmem = A[32][34] + L[32][34] + W[32][68] + dT[64][34] + G/de/bt/oe + eg = 26628 B
LdsSmem  += S[128][68]                                                        = 34816 B
                                                                         計 61444 B（実測 61952）
```

**`S`（state tile）の生存期間は chunk の両端だけ**である:

```text
chunk 先頭  主積の kk ループが Sst を読む              ← S の唯一の読み
chunk 中間  A / L / W / dT を書いて読む                ← S は触らない
chunk 末尾  状態更新の後、Sacc(register) から Sst を書く ← S の唯一の書き
```

`A/L/W/dT` はこの中間でのみ live なので `S` と重ねられる。union にして
`ChunkSmem mid` と `float S[kWk][kSp]` を同一領域に置いた。

**競合を避けるため barrier を 2 本追加**した:

1. 主積の直後（全 warp が `Sst` の読みを終えるまで `A/L` を書かない）
2. 状態更新の `tstep` ループの直後（全 warp が `mid.dT` の読みを終えるまで `Sst` を書かない）

結果: **LDS 61952 → 35332 B**（`131072 / 35332 = 3.7`）。LDS 制約は 3 blocks/MPU に緩和。

### 修正 2: VGPR 上限 240

LDS を 35332 B にしても **occupancy は 2 のまま**だった。原因は register 側。

この GPU の register file を合成カーネル（VGPR を変えながら co-residency を実測）で
測ると:

| regs | blocks/MPU | waves/SIMD | 含意 |
| ---: | ---: | ---: | --- |
| 231 | 3 | 6 | file ≥ 6×32×231 = 44352 |
| 239 | 3 | 6 | file ≥ 6×32×239 = 45888 |
| **248** | **2** | 5 | file < 6×32×248 = 47616 |
| 256 | 2 | 5 | — |

したがって **register file は約 46080 dword/SIMD（180 KB）**で、
`hipOccupancyMaxActiveBlocksPerMultiprocessor` や compiler が仮定する 49152 ではない。
**VGPR 256 では 5 wave/SIMD しか立たず 2 blocks/MPU。**

3 blocks/MPU には `46080 / (6 wave × 32 lane) = 240` → **VGPR ≤ 240** が必要。

試した手段:

| 手段 | 結果 |
| --- | --- |
| `__launch_bounds__(256, 3)` | 効かない（compiler の occupancy モデルが 49152 のため 256 でよいと判断） |
| `#pragma unroll 1`（kk ループ） | 効かない（256 のまま） |
| `__attribute__((amdgpu_waves_per_eu(6,6)))` | 単独では効かない（256）。`__launch_bounds__(256,3)` と併用すると VGPR 240 になるが **nreq 比 1.263（R=2）のままで性能が出ない** |
| **`__attribute__((amdgpu_num_vgpr(240)))`** | **VGPR 240 になり、nreq 比 0.997（R=3）を達成** |

`amdgpu_waves_per_eu` は `amdgpu_num_vgpr` の後継として案内されるが、このカーネルでは
VGPR 数が同じ 240 でも occupancy が上がらなかった。`waves_per_eu` は backend の
occupancy モデル（register file を 49152 と仮定）から「6 waves なら何 VGPR まで許せるか」
を逆算するヒントで、そのモデルが実機とずれているため 256 のままでよいと判断される。
`amdgpu_num_vgpr` は occupancy 推論を経由せず register allocator に上限を直接渡すので効く。

**つまり `amdgpu_num_vgpr(240)` は hack ではなく、backend の誤った resource inference を
迂回する暫定 workaround** という位置付けである。deprecated の理由は API 設計上のもので、
機能が壊れているわけではない。deprecation warning はカーネル宣言の前後を
`#pragma clang diagnostic push/pop` で囲んで抑制している。

`gdn_decomp8` が lossy 経路で未使用の `f.l` を 0 初期化している点も疑ったが、
外しても VGPR/spill は変わらなかったため元に戻した。

### 結果

`nreq` 比が **1.259 → 0.997** になり、**R=3（96 スロット）到達**。96 blocks が
ちょうど 1 ラウンドになる。interleaved A/B（device 1、7 reps）:

| rows | base p50 | 修正後 p50 | delta | base ktok/s | 修正後 ktok/s | 比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 152.0 | 86.6 | **-43.0%** | 841.94 | 1477.72 | 1.755x |
| 512 | 516.7 | 295.6 | **-42.8%** | 990.94 | 1732.01 | 1.748x |
| 1024 | 1247.0 | 592.3 | **-52.5%** | 821.20 | 1728.94 | 2.105x |
| **2048** | **2368.0** | **1373.8** | **-42.0%** | **864.87** | **1490.76** | **1.724x** |

1 ラウンド化だけなら -25% だが、実際は **-42%**。wave が 4 → 6 per SIMD に増えて
latency 隠蔽も改善したため。

E2E（pp2048、4 run、t/s）:

| | run1 | run2 | run3 | run4 | 中央値 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 2070.27 | 2068.56 | 2064.17 | 2067.17 | 2068.56 |
| 修正後 | 2170.51 | 2172.05 | 2167.54 | 2165.73 | **2170.51** |

**pp2048 +4.93%**（4 run すべてで改善）。GDN の -42% × 13.1% = 5.5% と整合。

### 正しさ

- `--check`（reference 比較）: rows=1/33/128/512/2048 すべて `mismatches=0 OK`
- `test_gdn_recurrence` PASS / `test_gdn_recurrence_decode1` 20/20 / `test_gdn_conv1d` PASS
- required acceptance 97/97 PASSED

### 教訓

1. **`hipOccupancyMaxActiveBlocksPerMultiprocessor` と compiler の occupancy モデルは
   この GPU では信用できない。** 実測（co-residency）で register file 46080 dword/SIMD、
   LDS 128 KB/MPU を確定させたことが決定的だった。
2. **律速が occupancy のとき、LDS を削るだけでは足りない。** register 側が binding なら
   両方解かないと動かない。nreq 比は「R が上がったか」の安価で確実な判定法。
3. **生存期間が排他な LDS 領域は union で重ねられる。** barrier の追加位置は
   「読み終わり」と「書き始め」の間で、2 箇所だけだった。
4. grid が並列スロット数の倍数でないときは、occupancy を上げる以外に
   「grid を倍数にする」道もある（本件は kWv を 64 → 32 にすると 192 blocks になる）。
   ただし warp の tile 配置が `kWv` と結合しているため大改修になる。
5. **`amdgpu_waves_per_eu` が効かないのは backend の occupancy モデルが gfx1201 実機と
   ずれているため。** 恒久対応は LLVM 側の gfx1201 resource model を実測に合わせることだが、
   当面は `amdgpu_num_vgpr(240)` を隔離して使うのが安全。
   （§7.88 の再測定で、この節の「`waves_per_eu` を併用しても R=2」という記録は
   属性の配置が `__launch_bounds__` の後ろだったための誤りと判明した。）

### 再現方法

```bash
# 律速の確認: nreq を変えると同じ rows でラウンド数だけが変わる
./build/phaseshift-bench gdn-recurrence --variant optimized --rows 2048 --nreq 1 \
  --key-heads 16 --value-heads 48 --head-k 128 --head-v 128 \
  --warmup 10 --samples 20 --launches 5 --device 1
```


---

## 7.88 GDN recurrence prefill: VGPR 上限 240 の妥当性検証（2026-09-21）

§7.87 で `amdgpu_num_vgpr(240)` により R=3 を達成したが、次の 3 点が未検証だった:

1. **240 という境界が hardware の性質なのか**（block size や LDS に依存しないか）
2. **deprecated な `amdgpu_num_vgpr` を非 deprecated な手段で置き換えられるか**
3. **240 を超える原因は何か**

結論は「240 は hardware の性質」「置き換えは可能だが 4% 遅い」「超過は compiler の
occupancy モデルと、ループ外アドレスをループ内で保持しないことの組み合わせ」。

### 1. VGPR しきい値の詳細測定

合成カーネル（float8 accumulator 数で自然な VGPR を作り、host の flag で block を
滞留させて同時 resident block を数える）で VGPR・block size・LDS を振った。
`blocks/MPU` は peak counter ÷ `multiProcessorCount`(32)。

**VGPR を属性で固定（nt=256、LDS 256 B）:**

| regs | blocks/MPU | waves/MPU | waves/SIMD |
| ---: | ---: | ---: | ---: |
| 224 | 3 | 24 | 6 |
| 232 | 3 | 24 | 6 |
| 239 | 3 | 24 | 6 |
| **240** | **3** | **24** | **6** |
| **241** | **2** | **16** | **4** |
| 247 | 2 | 16 | 4 |
| 248 | 2 | 16 | 4 |
| 255 | 2 | 16 | 4 |
| 256 | 2 | 16 | 4 |

**境界はちょうど 240 / 241。** compiler と `hipOccupancyMaxActiveBlocksPerMultiprocessor`
は 256 を境界とみなすので、モデルが食い違っている。

**block size 依存（wave budget は block size に依存しない）:**

| regs | nt=32 | nt=64 | nt=128 | nt=256 |
| ---: | ---: | ---: | ---: | ---: |
| **240** | 24 blocks（24 waves） | 12（24） | 6（24） | 3（24） |
| **248** | 20 blocks（20 waves） | 10（20） | 5（20） | 2（16）※ |

※ nt=256 だけ 16 waves になるのは、wave budget 20 に対して 1 block = 8 wave なので
2 blocks（16 waves）しか入らないため。wave budget 自体は 24 / 20 で block size に依存しない。

**LDS 依存（16 KB / 32 KB でも境界は不変）:**

| regs | LDS 256 B | LDS 16 KB | LDS 32 KB |
| ---: | ---: | ---: | ---: |
| 240 | 3 blocks | 3 blocks | 3 blocks |
| 248 | 2 blocks | 2 blocks | 2 blocks |

**自然な VGPR（属性なし、scratch 0）でも同じ境界:**

| regs | blocks/MPU | regs | blocks/MPU |
| ---: | ---: | ---: | ---: |
| 166 | 4 | 231 | 3 |
| 183 | 4 | **238** | **3** |
| 199 | 3 | **247** | **2** |
| 214 | 3 | 254 | 2 |
| 223 | 3 | 256 | 2 |

**実効 VGPR file の逆算:**

```text
R=240 で 24 waves が入る     → 24 × 32 × 240 = 184320 ≤ file
R=248 で 24 waves が入らない → 24 × 32 × 248 = 190464 >  file
R=184 で 32 waves が入る     → 32 × 32 × 184 = 188416 ≤ file
```

→ **実効 file は 188416〜190464 dword/MPU（47104〜47616 dword/SIMD、約 184〜186 KB）**。
driver の `regsPerMultiprocessor = 196608`（= 49152/SIMD）より **3〜4% 少ない**。
この 3〜4% が VGPR 241〜256 で 3 blocks 目が入るかを反転させる。

### 2. 240 超過の原因

prefill の production 経路は `gdn_recurrence_wmma_impl<true, false>`（kLossy=true）である。
属性を外した自然値は:

| variant | vgpr | spill | private |
| --- | ---: | ---: | ---: |
| `<true,false>`（lossy、production） | 256 | 4 | 20 B |
| `<false,false>`（exact） | 256 | 0 | 0 B |

上限を振ると（`<true,false>` / `<false,false>`）:

| cap | `<true,false>` | `<false,false>` |
| ---: | --- | --- |
| 200 | 217 / 6 spill / 28 B | 217 / 17 spill / 72 B |
| 216 | 217 / 6 / 28 | 217 / 17 / 72 |
| 224 | 224 / 4 / 20 | 224 / 11 / 48 |
| 232 | 232 / 4 / 20 | 232 / 2 / 12 |
| 240 | 240 / 4 / 20 | **240 / 0 / 0** |
| 248 | 248 / 4 / 20 | 248 / 0 / 0 |
| 256 | 256 / 4 / 20 | 256 / 0 / 0 |

- **lossy 版の 4 spill は上限に依存しない**（224 以上で常に 4）。ISA を見ると、spill は
  main chunk loop の前（#533）に `scratch_store_b64` ×2、ループの後（#8109）に
  `scratch_load_b64` ×2 で、退避されるのは v53:54 と v55:56 の **address ペア**である。
  ループ内で使わないループ外アドレスをループの外へ追い出しており、occupancy には有利な
  正当な判断である。したがって **240 上限は spill を増やしていない**。
- 逆に **exact 版は 232 で 2 spill、240 で 0 spill** なので、真の必要量は 233〜240。
- ピーク圧力領域は main chunk loop 全体（命令 #700〜#5150 で v235〜239）。自然版では
  v244 以上を使う命令が 216 個あり、その多くは `v_dual_add_nc_u32` などの **global
  address 計算**である。scheduler が load/store のアドレス計算を hoist した結果である。

`-mllvm -amdgpu-print-max-reg-pressure-regusage-before/after-scheduler` はこの ROCm には
存在しない。register pressure printer は legacy pass `amdgpu-print-rp` としてのみ登録され、
`llc -run-pass` で動かすが、この SDK には `llc` が同梱されていない。そのため
`-mllvm -stop-after=machine-scheduler` で MIR を取り出し、ISA と併せて解析した。

### 3. 非 deprecated な代替

| 手段 | 結果 |
| --- | --- |
| `__launch_bounds__(256, 3)` 単独 | VGPR 256（compiler は `196608/(3×8×32)=256` と計算する） |
| `__launch_bounds__(256, 4)` 単独 | VGPR 256（min-blocks hint が VGPR を制限しない） |
| `amdgpu_waves_per_eu(N,N)` を `__launch_bounds__` の**後**に置く | VGPR 256（**無効**） |
| `amdgpu_waves_per_eu(N,N)` を `__launch_bounds__` の**前**に置く | **VGPR 240 / spill 2 / private 12 B**（N=5〜10 で同じ） |

`amdgpu_waves_per_eu` は **属性の配置に依存する**。`__launch_bounds__` の後ろに書くと
無視される。§7.87 の「併用しても R=2」という記録は配置が後ろだったための誤りである。

A/B（device 1、interleaved、rows=2048、p50 µs）:

| variant | nreq=1 | nreq=2 | 比 (=R) | VGPR / spill |
| --- | ---: | ---: | ---: | --- |
| **`amdgpu_num_vgpr(240)`（production）** | **1372.4–1375.6** | **1369.8–1371.7** | **1.00 (R=3)** | 240 / 4 |
| `waves_per_eu(6,6)` + `launch_bounds(…,1)` | 1422.0–1427.7 | 1414.5–1420.7 | 1.00 (R=3) | 240 / 2 |
| `waves_per_eu(6,6)` + `launch_bounds(…,3)` | 1422.9–1435.8 | 1414.7–1422.4 | 1.00 (R=3) | 240 / 2 |

**occupancy は同じ R=3 だが、`waves_per_eu` 版は約 4% 遅い。** spill はむしろ少ないので、
差は scheduler が occupancy target に合わせて schedule を変えたことによる。

### 結論

1. **240 は hardware の性質**であり、compiler の resource model（49152 dword/SIMD）が
   実測（47104〜47616）より 3〜4% 大きいために 256 を選んでしまう。恒久対応は LLVM 側。
2. **`amdgpu_num_vgpr(240)` は spill を増やさない**（自然 256 と同じ 4 spill）。したがって
   この workaround にコストはない。
3. **非 deprecated な `amdgpu_waves_per_eu(6,6)` でも 240 / R=3 は作れるが 4% 遅い**ので、
   現状は `amdgpu_num_vgpr(240)` を維持する。
4. 「自然に ≤240 へ落とす」には main chunk loop の圧力を 16 VGPR 以上下げる必要があり、
   単純な変更では達成できない。ピークの主因は address 計算の hoist である。

### 再現方法

```bash
# しきい値測定（合成カーネル）: /tmp/opencode/vgpr_sweep.cpp
#   float8 accumulator 数で VGPR を作り、host flag で block を滞留させて同時 resident 数を数える

# 240 超過の確認: 属性を外して資源量を見る
ROCM=/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel
$ROCM/lib/llvm/bin/clang++ --offload-arch=gfx1201 -O3 -std=gnu++20 -Iinclude -Isrc \
  -x hip --offload-device-only -c recurrence.hip -o nat.o
$ROCM/lib/llvm/bin/clang-offload-bundler --unbundle --type=o \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx1201 --input=nat.o --output=nat.elf
$ROCM/lib/llvm/bin/llvm-readelf -n nat.elf | grep -E "\.name:|vgpr_count|vgpr_spill"

# MIR 取得（register pressure 解析用）
$ROCM/lib/llvm/bin/clang++ --offload-arch=gfx1201 -O3 -std=gnu++20 -Iinclude -Isrc \
  -x hip --offload-device-only -c recurrence.hip -o rec.o \
  -mllvm -stop-after=machine-scheduler -save-temps
```


---

## 7.95 gdn_conv1d の行方向 tiling（2026-09-21）

GEMM 群を打ち上げた後、次の対象として gdn_conv1d を測った記録。
**pp2048 +1.86%（6/6 対）**、カーネル単体では **583 → 224 µs（−62%）**。
計測は power limit 210 W の状態で行った（§7.90 の 300 W とは条件が異なる）。

### 発見: 帯域が 34% しか出ていない

`gdn-conv1d` マイクロベンチ（conv_dim=10240, rows=2048, nreq=1）:

| | p50 | 達成帯域 |
| --- | ---: | ---: |
| 変更前 | 583 µs | 126 MB / 583 µs = **218 GB/s（DRAM ピーク 636 の 34%）** |

原因は grid の小ささ。カーネルは 1 thread = 1 channel で行方向に逐次ループする構造で、
grid は `(conv_dim/256, num_requests)` = prefill では **40 blocks しかない**
（64 CU に対して 0.6 blocks/CU）。

### 変更: 4-tap を利用した行方向の並列化

この conv は history=3 の causal conv なので、chunk の先頭 tile 以外は
**直前 3 行を入力バッファから直接読めば state が要らない**（chunk 内の行は同一系列で、
`t0 >= 3` なら `row0-3` は必ず chunk の内側）。そこで:

- grid を `(conv_dim/256, num_requests, tiles)` の 3D にし、1 block = `row_tile` 行を担当
- 初期 shift register は `t0 == 0` なら `conv_state`、それ以外は入力の `row0-3..row0-1`
- `conv_state` の書き戻しは各 request の**最終 tile だけ**が行う
- tile 数は `max_request_rows`（runtime が selector 用に既に計算している値）から決める。
  `actual_rows` から決めると短い request に無駄な block が出る（実測で nreq=16 のとき +41%）
- `GdnConv1dArgs.max_request_rows` を追加し、bench と runtime が設定
- 既定 `row_tile = 32`。`PHASESHIFT_GDN_CONV1D_TILE`（0 = 従来の 1 tile）で A/B 可能

### 正しさ（bit-exact）

要素ごとの計算（FMA の順序、shift される値）は不変。`test_gdn_conv1d` は
**out maxabs=0 / maxrel=0 / state_diff=0**（correctness 経路とビット一致）で全ケース PASS。
128 行の prefill ケースが tiling された経路を通る。

### 効果（rows 別、nreq=1）

| rows | 従来 | tiled (32) | 差 |
| ---: | ---: | ---: | ---: |
| 512 | 145.9 µs | 25.3 µs | −83% |
| 1024 | 193.2 | 48.2 | −75% |
| 2048 | 582.9 | 223.7 | −62% |
| 4096 | 1389.8 | 438.8 | −68% |
| 64 | 17.3 | 11.1 | −36% |
| 128 | 29.1 | 11.4 | −61% |

rows=2048 の達成帯域は 578 GB/s = **DRAM ピークの 91%**。tile 16/32/64 を振ると
16 が一部形状で最良だが rows=512 と多リクエストで劣化するため、**全ケースで従来を
下回らない 32 を既定**にした（e2e 差は 0.03% 程度）。

### e2e

| bench | 従来 | tiled | 差 |
| --- | ---: | ---: | ---: |
| pp2048（interleaved 6 対） | 2179.6 | **2220.1** | **+1.86%**（6/6 対） |
| tg（ctx 2048, 32 tok） | 27.16 | 27.15 | ±0%（decode は tiles=1 で不変） |

### 計測コマンド

```bash
./build/phaseshift-bench gdn-conv1d --rows 2048 --warmup 20 --samples 30 \
  --launches 20 --device 1
PHASESHIFT_GDN_CONV1D_TILE=0 ./build/phaseshift-bench pp \
  --model-dir models/Qwen3.8-27B-PSQ --prompt-tokens 2048 --mode forward \
  --page-tokens 16 --arena-gib 24 --runs 1 --warmup 0 --device 1
```
