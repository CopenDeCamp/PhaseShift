> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# Runtime / HIP Graph 実行オーバーヘッド履歴

`docs/rnd/optimization_findings.md` から分離した、HIP Graph capture・warm 判定・keep-alive・kernel 間 gap・execution overhead の実験履歴。見出し番号（7.x）は元 dump の通し番号を保持する。

## 収録セクション

- 7.14 prefill のコスト内訳（rocprofv3 kernel-trace）と参照パス潰し
- 7.27 HIP Graph 評価: 本 workload では net 効果なし（不採用）
- 7.29 HIP Graph 実装（compile-time option）と 210W power cap 環境への移行
- 7.53 HIP graph が PP で効かない理由: capture が一度も発火していない
- 7.54 HIP graph を PP で engage させた（warm 判定の修正）
- 7.56 現状ベースライン再計測（既定ビルド = HIP graph OFF）
- 7.57 decode（tg）の内訳と実効帯域
- 7.58 decode GEMV の実効帯域は既に roofline 近傍
- 7.59 小 kernel は launch 律速で、既存 fusion は本モデルに効かない
- 7.82 prefill のコスト内訳（PSQ4 最適化後）
- 7.89 pp の kernel 間 gap: CP の power-gating と keep-alive
- 7.90 power limit 300 W での pp/tg 再測定
- 7.97 keep-alive の production 化
- 7.101 HIP graph が optimized-only batch で engage しない gate バグ

---

## 7.14 prefill のコスト内訳（rocprofv3 kernel-trace）と参照パス潰し

PP2048、`rocprofv3 --kernel-trace` でカーネル別に集計。

**27B before（1 pass 3475ms = 589 t/s）**

| kernel | ms/pass | % | dispatch/pass |
|---|---|---|---|
| gemm_rocmfp4_w4a8 | 1140 | 32.8% | 336 |
| attention_paged_bf16 | 1134 | 32.7% | 16 |
| **model_dispatch_kernel（正しさ参照）** | **689** | **19.8%** | 313 |
| gemm_rocmfp8_w8a8 | 299 | 8.6% | 64 |
| gdn_recurrence_wmma | 86 | 2.5% | 48 |
| gdn_norm_gate | 45 | 1.3% | 45 |
| activation_quantize_e4m3 | 31 | 0.9% | 208 |
| その他 | ~50 | ~1.4% | — |

**4B before（1 pass 806ms）**: attention 47.5% / fp4 19.7% / gdn_prepare 11.4% /
参照パス 6.9% / fp8 5.7% / GDN 系 5.0% / その他 3.8%。

### 7.14.1 参照パス落ちの正体と潰し方（`c3975d2`, `795cecb`）

当初「最適化カーネルが無い op は ROPE / L2_NORMALIZE / OUTPUT_GATHER /
STATEFUL_CAUSAL_CONV1D の 4 つ」と推定したが、実 op 列を出すと**本当の原因は別**だった。
`PHASESHIFT_QWEN35_KERNEL_TRACE` に **per-kernel-id の参照パス落ちカウンタ**を足して
数えると（27B、PP512）:

```text
SWIGLU 64   ← 64 layer 全部
SILU 3 / RMS_NORM 3 / LINEAR_BF16 1 / OUTPUT_GATHER 1 / SAMPLING 1
```

**原因1: `GdnPrepare` fusion が 27B で全滅（48 層）**

op 列・matcher・selector は全部一致していた。落ちていたのは
`gdn_prepare_dispatch.hip` の `io_overlap` ガード:

```text
workspace planner が conv 入力バッファを SiLU 出力に再利用する
  → v_output == input + 2*qk_features*sizeof(bf16)（row stride も一致）
  → 汎用の io_overlap が「V 出力が入力を踏む」と判定して 48 層すべて bail
```

これは**安全な self-write**である。`gdn_prepare_bf16_impl` は conv の history を
register に持つので、各スレッドが同じ要素を同じ反復で read→write するだけ。
この厳密形（オフセットと stride が一致）だけ許可し、それ以外の重なりは従来どおり拒否。

```text
27B PP2048:
  model_dispatch_kernel  1377.7 -> 296.7 ms  (2 pass 合計)
  gdn_prepare_bf16          0.0 -> 410.7 ms
  kernel 合計             6949   -> 6381 ms   (1.09x)
  gdn_prepare 発火         0      -> 48/pass（4B も 22 -> 24 層）
```

**原因2: 27B の `SwigluBf16` selector の feature 数を間違えていた**

`elementwise_selector.cpp` の 27B ルールが `34816`（= 2 × intermediate_size）だが、
dispatcher が渡すのは**出力**の feature_count（= intermediate_size = 17408）。
よって SwiGLU 64 dispatch が全部参照パス落ち。17408 に修正。

```text
27B PP2048: 3300.9 -> 3200.2 ms (620.4 -> 640.0 t/s)
```

**結果（27B、1 pass）**

| kernel | ms/pass | % |
|---|---|---|
| gemm_rocmfp4_w4a8 | 1173 | 37.7% |
| attention_paged_bf16 | 1138 | 36.6% |
| gemm_rocmfp8_w8a8 | 305 | 9.8% |
| gdn_prepare_bf16 | 208 | 6.7% |
| gdn_recurrence_wmma | 96 | 3.1% |
| gdn_norm_gate | 52 | 1.7% |
| **model_dispatch_kernel（参照）** | **37** | **1.2%** |
| activation_quantize_e4m3 / a8 | 32+15 | 1.5% |
| swiglu_bf16 | 22 | 0.7% |
| その他 | ~30 | ~1% |

参照パス: **689 → 37 ms/pass**。kernel 合計 6949 → 6227ms（2 pass）= **1.12x**、
bench は 589 → **640 t/s（1.09x）**。残った参照落ちは LINEAR_BF16 1 /
RMS_NORM 3 / SILU 3 / OUTPUT_GATHER 1 / SAMPLING 1（単発系、計 1.2%）で ① は完了。

### 7.14.2 ② attention 36.6%（次の本命）

実効 ~6 TFLOPS（GEMM は 76）。`--page-tokens 16` の細かさが疑わしい。
prefill 用の fused attention（ページ統合 / ブロック処理）が本命。

### 7.14.3 ③ 4B の gdn_prepare

原因1 と同じ修正で 4B も 22 → 24 層全部発火するようになった（7.14.1）。

### その他

- 27B は `--arena-gib 24` を明示しないとモデルロードが OOM する（default 6GiB）。
- 27B prefill は記録の 463 t/s → **640 t/s（1.38x）**（7.11/7.12 の GEMM + MB + 本節）。


---

## 7.27 HIP Graph 評価: 本 workload では net 効果なし（不採用）（2026-09-15）

- e2e trace（rocprofv3 kernel_dispatch、101 step）から kernel 間 gap 計測:
  **4.47ms/token（wall の 12%）**。全 kernel クラスでほぼ均一
  （GEMM 後 4.09us ×502、fused EW 後 4.07us ×259、小 kernel 後 4.50us ×228、
  その他 6.36us ×59）。
- executor は 1 token あたりの全 kernel（~1038 個）を submit 時に一括
  enqueue し、PendingBatch 完了まで host は待つ構造 → CPU dispatch は
  GPU 実行中に隠れており、gap は GPU 側の kernel dispatch rate 由来。
- microbench A/B（back-to-back、graph 化前後、p50）:
  - empty kernel: plain 2.43us / graph 2.45us（GPU dispatch 下限）
  - psq4 gate/up（41us）: plain 41.2 / graph 39.7 → **graph −1.5us**
  - fused swiglu（12us）: plain 11.96 / graph 13.31 → **graph +1.35us**
  - hipGraphLaunch の CPU cost: 1038 ノードで 11.9us（無視できる）
- e2e mix（GEMM 502 : 小 kernel 540）に適用すると
  502×(−1.5) + 540×(×1.35) ≒ **±0.1ms（net zero）**。
  大きな GEMM の利得が小 kernel の graph ノードペナルティで相殺される。
- 判定: 採用しない。dispatch floor（4.47ms/tok）を下げるには
  kernel 数自体の削減（GEMM epilogue への residual_add 等の融合など）が
  本来の手。


---

## 7.29 HIP Graph 実装（compile-time option）と 210W power cap 環境への移行（2026-09-15）

### 経緯

- §7.27 で graph は micro 基準「net 効果なし・不採用」とした。
  e2e gap 4.5us/tok×kernel 数と、融合後のカーネル mix（GEMM 497 + 小 kernel ~360）
  なら符号が反転する可能性（−0.2〜−1.6ms/tok 見積）があるため、
  executor 内 capture/replay として実装して e2e A/B で最終判定する。
- 有効化は起動引数（env var）ではなく **CMake option `PHASESHIFT_HIP_GRAPH`
  （`#ifdef PHASESHIFT_HIP_GRAPH`、既定 OFF）**。A/B は 2 バイナリ交互で実施。

### 実装

- `submit_co_batch` の 1 submit 分（H2D upload 4 件 + prepare_batch_descriptor
  + program 全 dispatch）を `hipStreamBeginCapture/EndCapture`（Global mode）で
  キャプチャし `hipGraphExec` 化。以降は `hipGraphLaunch` のみ。
- キャッシュキーは **(row bucket, N, num_outputs, num_sampled)**。
  warmup（compute_logits=0）と計測（=1）で num_outputs が変わるため必須。
  キー不一致時は旧 exec を破棄して再キャプチャ。
- 毎 submit で変化する値（token_ids / request descriptors / batch ctx）は
  全て device memory 経由なので、pinned staging（executor 所有）への
  CPU memcpy + 固定ポインタの H2D で capture 合法性を担保。
- capture 不可能だった箇所と対応:
  - `launch_host_binding`（correctness パス）の staging pool upload 時の
    `hipStreamSynchronize` → **pool warm 前チェック**（executor 側で
    該当 bucket の全 slot upload 完了まで plain 実行。初回 submit = warm）。
  - 同関数の EMBEDDING_LOOKUP 後 sync → capture 中（via_pool）は skip。
    pool image は 1 回だけ device に上がっており host 再利用ハザードなし。
  - non-pool staging パス（共有 host バッファを dispatch ごとに refill）は
    構造的に capture 不可 → capture 中に到達したらエラー（現在到達しない）。
- 失敗時は `graph_failed` で以降 plain に自動 fallback（正しさは常に plain と同一）。

### 結果

- 正しさ: e2e LOGITS bit-identical（argmax=271 max=10.836942 sum=-845382.688）。
- **機材状態の日内変動（重要）**: 210W power limit と VLLM 負荷は
  1 日中不変だったが、e2e 絶対値は日中に 2 状態往復していた
  （本文書の A/B 履歴: 01:2x 44.3ms/tok → 08:1x-20:2x 36.0-37.8ms/tok
  → 21:0x- 40.9ms/tok）。per-kernel exec 比較で「40.9ms 状態」は
  午前 08:18 trace と同一（±1%）、「36.0ms 状態」では kernel busy が
  ~14% 短い（gap は両状態で ~2.0ms/tok 同一）= clock/電力 headroom
  由来の GPU 状態差。300W 実験（ユーザー側で limit 変更）は +2% のみ
  → power cap 単独が状態差の原因ではない。
  旧バイナリ（6bb6028）で同一数値を再現 → コード起因ではない。
  以降は**相対 A/B（同一ウィンドウ内）のみを受理**し、絶対値の
  通し比較はしない。
- e2e A/B（210W 環境、交互 3 ペア、tg_plain vs tg_graph）:
  - plain: 40.96 / 41.04 / 41.08 ms/tok
  - graph: 41.01 / 41.07 / 41.14 ms/tok
  - 差 **+50us/tok（graph 側が僅かに不利）**。§7.27 の micro 見積
    （大 GEMM −1.5us / 小 kernel +1.35us）の e2e 転移は起きておらず、
    micro 利得は e2e に転移しない件（GEMM variant 2 件の前科）と同じ挙動。
- 判定: **採用見送り（option 既定 OFF のまま保持）**。
  高速状態ウィンドウ（36ms 系）での再 A/B が有効性判定の
  機会になる（現在の低速状態では GPU 側利得が mask される）。


---

## 7.53 HIP graph が PP で効かない理由: capture が一度も発火していない（2026-09-19）

### 実測

`PHASESHIFT_HIP_GRAPH=ON` でビルドして 27B PP（prompt 2048）を計測:
**975.9ms / 2098 tok/s**（OFF は 979.2ms / 2086 tok/s）で差はノイズ内。
ユーザー観測どおり「効かない」。

### 原因（一時計装で確定）

`Executor::submit_co_batch` の graph 経路は

```cpp
if (!enqueued && !executor.graph_failed &&
    program_pool_warm(executor, *program, gslot_idx)) { ...capture... }
```

で capture を試行するが、**`program_pool_warm` が常に false**:

```
[graph] attempt N=2048 gslot=7 enqueued=0 warm=0 dispatches=1654
[graph] warm fail: i=0 slot=11578 bytes=240 uploaded=0
```

`program_pool_warm` は bucket program の**全 dispatch** について
`bytes[slot] == 0 || uploaded[slot] == 1` を要求する。一方 `uploaded[slot]` は
dispatch が実際に launch された時にしか 1 にならない（`program_executor.hip` の
staging pool upload 経路）。R2048 の prefill program は 1654 dispatch を持つが、
1 バッチで実行されるのは約 995 dispatch（trace 実測）で、**残りは条件で skip** される。
このため dispatch 0（bytes=240）等が永久に `uploaded=0` となり、warm 判定は
充足不能 → **capture は一度も試行されない**。

つまり「graph が効かない」のではなく **graph が一度も使われていない**。
しかも失敗は無言（`graph_failed` も立たない）なので、計測上は
「効果なし」に見えていた。

### 含意

- 修正するなら warm 判定を「実際に実行される dispatch」に限定する
  （probe バッチで実行済み dispatch の bitmap を取る、または capture 中に
  upload が必要になった slot を除外する）。
- ただし上限は小さい: PP の kernel 間 gap は約 43ms/1059ms（4%）で、
  その一部は GPU 側の kernel ramp/drain なので、graph で取れるのは数 % 程度。
- 少なくとも「graph が engage しなかった」ことをログに出すべき
  （無言フォールバックは計測を誤解させる）。

→ §7.54 で warm 判定を修正し、実際に engage させた。


---

## 7.54 HIP graph を PP で engage させた（warm 判定の修正）（2026-09-19）

§7.53 の原因（`program_pool_warm` が充足不能で capture が一度も試行されない）を修正し、
HIP graph を実際に使わせた。

### 修正内容

1. `DispatchStagingPool` に `run_seen` を追加し、`launch_host_binding` が
   staging pool 経由で dispatch を起動した時に `run_seen[slot] = 1` を立てる。
   最適化 dispatch は `launch_host_binding` を通らないため立たない。
2. `program_pool_warm` を「この bucket で一度でも起動された dispatch」だけを
   対象に変更。未起動 dispatch は capture 中も起動されない前提で判定から除外する。
   併せて 1 件以上の起動済み dispatch を要求し、probe バッチなしの初回 capture
   （未 upload の staging に当たって `graph_failed` に落ちる）を防ぐ。
3. cache key に `min_visible_tokens` / `max_visible_tokens` を追加。
   これらは paged attention の kernel 引数として graph に焼き込まれるため、
   可視範囲が異なる batch で replay すると誤った結果になる。
4. `PHASESHIFT_GRAPH_DEBUG=1` で skip 理由 / capture 成否 / replay を stderr に出す
   （無言フォールバックの解消）。

### 実測（27B-PSQ、prompt 2048、OFF/ON interleaved 3 pair、gpu_ms_min）

| pair | OFF ms | OFF tok/s | ON ms | ON tok/s |
|---|---|---|---|---|
| 1 | 980.34 | 2086.30 | 961.12 | 2130.19 |
| 2 | 983.92 | 2080.05 | 963.59 | 2123.99 |
| 3 | 987.44 | 2072.90 | 965.81 | 2117.99 |
| 平均 | 983.90 | 2079.75 | 963.51 | 2124.06 |

- **−20.4ms / −2.07%、+44.3 tok/s（+2.13%）**。
  3 pair とも ON が OFF を下回り、値域も重ならない（ON ≤ 965.8 < OFF ≥ 980.3）。
- kernel 間 gap 全体は約 43ms なので、その約半分を回収したことになる。
  残りは GPU 側の kernel ramp/drain と graph launch 自体のコスト。
- `PHASESHIFT_GRAPH_DEBUG=1` で `capture ok bucket=7 N=2048` → `replay` を確認。
  最初の 1 バッチのみ `skip: staging pool not warm`。

### 検証

- ctest（`PHASESHIFT_HIP_GRAPH=ON` ビルド）: 76/78 PASS、新規 failure 0。
  失敗 2 件は gfx1151 専用 halo test を本機（R9700）で実行していた既知事象で、
  本変更とは無関係（arch 非適合側を DISABLED 化した。docs/developer/testing.md）。
- 数値: `phaseshift-compute` に同一 prompt（2048 token、decode 4 step）を与え、
  OFF / ON で `GENERATED_IDS` と `--dump-logits` が完全一致（capture/replay 経路の確認）。

### 既知の制約

- decode は step ごとに可視範囲が変わるため cache key が毎 step 変わり、
  現状は capture をやり直す（正しさ優先の代償）。
  恒久対応は可視範囲を batch context 経由で渡し、kernel 引数に焼き込まないこと。
- PP の replay 経路の数値一致は bench が出力を検証しないため未確認
  （logits 比較は decode 経路のみ）。
- 既定は opt-in（`PHASESHIFT_HIP_GRAPH=ON`）のまま。


---

## 7.56 現状ベースライン再計測（2026-09-19、既定ビルド = HIP graph OFF）

`main`（PGN lossy 既定 + 2D prefill + HIP graph 修正込み）の既定ビルドで pp / tg を再計測した。
GPU は共有のため min/median を採用し、OOM 時はリトライした。

### pp（prompt 2048、`--mode forward --page-tokens 16 --arena-gib 24 --runs 8 --warmup 3`）

| run | gpu_ms_median | gpu_ms_min | gpu_tokens_per_sec |
|---|---|---|---|
| 1 | 980.77 | 979.15 | 2083.9 |
| 2 | 984.76 | 982.78 | 2081.8 |
| 3 | 987.35 | 985.39 | 2074.3 |

前回記録（979.2ms / 2085.6 tok/s）と同等（差は -0.2〜-0.8%、run 内ドリフトと同程度）。
参考: HIP graph ON では 963.5ms / 2124 tok/s（§7.54、別ビルドでの interleaved 計測）。

### tg（`--mode forward --prefill-chunk 128 --warmup 4 --compute-logits 1 --arena-gib 24`）

| context | tokens | gpu_ms | us/token | gpu_tokens_per_sec | 過去記録 |
|---|---|---|---|---|---|
| 256 | 64 | 2320.9 | 36263.7 | **27.58** | ctx128: 26.79 t/s |
| 2048 | 32 | 1187.5 | 37108.2 | **26.95** | ctx2048: 26.03 t/s |

過去記録と同等〜微改善で回帰なし。なお 37us/token は per-token weight traffic 7.93GB
（AGENTS.md）として **約 215 GB/s** であり、DRAM 640GB/s の roofline（12.4ms/token ≒ 80 tok/s）の
約 1/3。decode 側は prefill より余地が大きい（既知の GVM latency 問題）。

計測メモ: 共有 GPU で他プロセスが VRAM を掴むと `GpuArena::create failed`（OOM）で
失敗する。本計測でも tg が数回 OOM し、リトライで成功した（コード側の問題ではない）。


---

## 7.57 decode（tg）の内訳と実効帯域（2026-09-19）

decode の最適化余地を判断するため、tg 1 回の kernel trace を batch 単位に分解した
（`rocprofv3 --kernel-trace`、ctx256 / 16 decode step、batch 境界は
`prepare_batch_descriptor_kernel` をマーカーに使用）。

### per-token 内訳（16 step 平均、kernel 時間合計 34.3ms/token）

| us/token | % | n/token | us/call | kernel |
|---|---|---|---|---|
| 11887 | 34.6% | 144 | 82.5 | gemm_psq4_w4a8_wmma_kernel<1> |
| 8125 | 23.7% | 65 | 125.0 | gemm_psq8_w8a8_wmma_kernel<1> |
| 4335 | 12.6% | 112 | 38.7 | gemm_psq4_w4a8_wmma_block_splitk_kernel<8> |
| 4044 | 11.8% | 48 | 84.2 | gemm_psq4_w4a8_wmma_block_splitk_kernel<4> |
| 2165 | 6.3% | 48 | 45.1 | gdn_recurrence_wmma_impl<true> |
| 1047 | 3.1% | 128 | 8.2 | fused_rmsnorm_e4m3_kernel |
| 677 | 2.0% | 96 | 7.1 | gemv_bf16_kernel |
| 618 | 1.8% | 64 | 9.6 | fused_swiglu_e4m3_kernel |
| 483 | 1.4% | 48 | 10.1 | fused_gdn_norm_gate_e4m3_kernel |
| 303 | 0.9% | 16 | 18.9 | attention_paged_bf16_impl<true> |
| 156 | 0.5% | 48 | 3.2 | gdn_prepare_bf16_impl |

GEMV 系（psq4/psq8 <1> + splitk）で **58.8%**、残りが GDN/FN/attention 等。
実測 36.26ms/token（ctx256）に対し kernel 合計 34.3ms → **gap は約 2ms（5.5%）**。

### per-token weight traffic の訂正: 16.7GB（AGENTS.md の 7.93GB は誤り）

safetensors の header から tensor 別 payload を集計（総 18.92GB、5.54 bpw）:

| 対象 | GB |
|---|---|
| layer.mlp | 10.339 |
| layer.linear_attn (GDN) | 4.088 |
| layer.self_attn | 0.944 |
| lm_head | 1.351 |
| embed_tokens | 1.351（decode では 1 行のみ = 無視可） |
| mtp.* | 0.85（`--mtp` 時のみ） |

decode 1 token で読むのは mlp + linear_attn + self_attn + lm_head = **約 16.72 GB**。
よって実効帯域は 16.72GB / 36.26ms = **461 GB/s = DRAM ピーク 636GB/s の 72%**
（ctx2048 は 37.11ms → 450 GB/s = 71%）。

AGENTS.md の「per-token weight traffic = 7.93GB、天井 12.4ms/token ≒ 80 tok/s」は
係数が実測と合わない（実 payload からは 26.3ms/token ≒ 38 tok/s が DRAM 天井）。
結論（L3 ではなく DRAM 律速）自体は変わらないが、余地は 1/3 ではなく **約 1.28×**。

### 余地の内訳（優先順）

1. **GEMV 帯域 72% → 上振れ**（最大の絶対量）: 16.7GB を 90% で流せれば 29.2ms/token = 34 tok/s。
2. **gap 約 2ms（5.5%）**: 180 launch/token で約 11us/launch。
   decode は形状（可視範囲）が毎 step 変わるため HIP graph の cache key が毎回変わり
   capture し直す（§7.54）。可視範囲を batch context 経由にすれば graph が使え、gap を削れる。
3. **小 kernel の latency**:
   `fused_rmsnorm_e4m3_kernel` 128 call × 8.2us = 1.05ms/token（入力 10-20KB のみで launch/ramp 支配）、
   `gdn_ba`（n=48）は 12.6us/call（grid 3 block）で完全に latency bound。
4. `gdn_recurrence` 2.17ms/token（6.3%）は §7.51 の lossy が既定で既に削減済み。

※ 本節の数値は `perf/decode-dig` ブランチ時点の計測。実装は未着手。


---

## 7.58 decode GEMV の実効帯域は既に roofline 近傍（優先順 1 の打ち切り、2026-09-19）

§7.57 で「decode はピークの 72%」と報告したが、あれは **トークン全体**（非 GEMV kernel と
gap を含む）の値だった。GEMV kernel 自体の帯域を分離して測った結果、**既に 92%** に達して
おり、GEMV 帯域の改善余地はほぼ無い。以下、すべて実測。

### 単体計測（rows=1、`--variant gemm`、min of 20x10 launch、重みが L3 64MB を超える大きさ）

| dtype | n | k | 重み | min us | GB/s |
|---|---|---|---|---|---|
| psq4 | 17408 | 20480 | 178.3MB | 327.9 | 543 |
| psq4 | 32768 | 10240 | 167.8MB | 315.3 | 532 |
| psq4 | 16384 | 20480 | 167.8MB | 340.3 | 493 |
| psq4 | 8192 | 40960 | 167.8MB | 365.4 | 459 |
| psq4 | 2048 | 40960 | 41.9MB | 94.2 | 445 |
| psq8 | 17408 | 20480 | 356.5MB | 622.5 | 573 |
| bf16 | 17408 | 20480 | 713.0MB | 1141.7 | 625 |
| psq4 | 17408 | 5120 | 44.6MB（L3 内） | 38.3 | 1163 |
| psq4 | 5120 | 17408 | 44.6MB（L3 内） | 36.9 | 1209 |

- 冷えた重みでは psq4 が 543GB/s、psq8 が 573GB/s、bf16 GEMV が 625GB/s。
- L3 に載る大きさなら 1163-1209GB/s。DRAM の 636GB/s を超えるのは L3 ヒットのためで、
  decode の重み（16.7GB）は L3 に載らない。
- 同一バイト数で grid だけ変えると、n=8192（8 block/CU）459 → n=16384（16/CU）493 →
  n=32768（32/CU）532 と飽和する。n=2048（2/CU）は L3 内 41.9MB でも 445GB/s しか出ず、
  小 shape は parallel 不足で latency bound。

### 律速の切り分け（n=17408 k=20480、plain kernel、ベースライン 327.9us）

| 条件 | min us | 差 |
|---|---|---|
| ベースライン | 327.9 | - |
| `kUnroll` 4 → 8 | 352.3 | +7.4%（悪化） |
| psq4 decode（perm テーブル）を恒等写像に置換 | 330.1 | +0.7%（ALU は律速でない） |
| splitk S=2 / S=4 / S=8 | 332.3 / 329.8 / 346.7 | 改善なし |

MLP を増やしても（unroll 8）、ALU を消しても、K 分割で並列度を上げても改善しない。
543GB/s はこのアクセスパターンの実効上限。

### モデル内での分離（ctx256 / 16 step の trace、§7.57 の内訳を使用）

per-token の重みは safetensors header から **psq4 11.844GB + psq8 4.827GB = 16.671GB**。

| 経路 | 重み | kernel 時間 | 実効帯域 |
|---|---|---|---|
| psq4 plain（gate/up/z/out 等） | 7.079GB | 11887us | 595 GB/s |
| psq4 splitk（down / qkv / q / kv / o） | 4.765GB | 8602us | 554 GB/s |
| psq8（16 層分の down/qkv/z/out + lm_head） | 4.827GB | 8125us | 594 GB/s |
| 合計 | 16.671GB | 28614us | **583 GB/s = ピーク 636 の 91.6%** |

GEMV の理論床は 16.671GB / 636GB/s = 26.2ms で、実測 28.6ms。残差 2.4ms のうち
splitk が plain より 7% 低い分が約 0.6ms、残りは小 shape の parallel 不足。

### 結論と残りの余地

- **GEMV 帯域は打ち切り**（改善余地は全体の 4% 未満、kernel 書き換えの費用に見合わない）。
- decode 1 token = 34.3ms（kernel）+ 約 2ms（gap）の内訳は:
  - GEMV 系 28.6ms（roofline の 92%）
  - 非 GEMV な小 kernel 5.9ms（すべて latency bound）:
    `gdn_recurrence` 2.17ms / `fused_rmsnorm` 1.05ms（128 call x 8.2us、
    入力 10-20KB で launch/ramp 支配）/ `gemv_bf16` 0.68ms（96 call x 7.1us、
    重みは合計 1MB 未満）/ `fused_swiglu` 0.62ms / `gdn_norm_gate` 0.48ms /
    attention 系 0.38ms
  - gap 約 2ms（約 900 launch/token x 約 2.2us）
- したがって decode の次の実利は **gap（5.5%）** と **小 kernel の latency（合計 5.9ms の
  うち launch/ramp 分）** であり、GEMV 帯域ではない。


---

## 7.59 小 kernel は launch 律速で、既存 fusion は本モデルに効かない（2026-09-19）

§7.57 で残した「小 kernel の latency」を掘った。結論は **打ち切り**。

### 小 kernel のコストは work ではなく launch に支配される

`phaseshift-bench gemm --dtype bf16 --rows 1 --variant gemm --launches 20` で
n を振ると（k=5120、1 launch あたりの us）:

| n | 重み | us/launch |
|---|---|---|
| 8 | 80KB | 6.97 |
| 16 | 160KB | 7.02 |
| 24 | 240KB | 7.02 |
| 48 | 480KB | 6.99 |
| 128 | 1.3MB | 7.05 |
| 512 | 5.2MB | 7.06 |
| 5120 | 52MB | 27.11 |

work が 64 倍違っても 7.0us で一定（n=512 は L2 内で 714GB/s 相当）。つまり
**1 launch あたり約 7us の固定コスト**があり、decode の小 kernel の実測時間
（`fused_rmsnorm` 8.2us、`fused_swiglu` 9.6us、`gdn_norm_gate` 10.1us、
`gemv_bf16` 7.1us）はほぼこの固定コストで説明できる（`gdn_prepare` の 3.2us が例外）。

→ 小 kernel の削減は「本体を速くする」ではなく「launch 数を減らす」でしか効かない。

### 既存 fusion の適用可否（本モデルの shape で確認）

- **GateUp（gate+up+swiglu）**: 条件に完全一致（全 64 層で gate/up は psq4、n=17408、
  k=5120、直後に SWIGLU）。規則表は k=2560 のみなので k=5120 の規則を追加して実測。
- **Qkv**: 本モデルの full-attention 層（16 層）は psq8 なので psq4 融合カーネルは不可。
- **GdnProj（qkv+z+a+b）**: `try_launch_fusion_psq4` が全 projection に `encoding==3`
  （psq4）を要求するが、本モデルの `in_proj_a` / `in_proj_b` は bf16 → 却下される。

### GateUp 融合の実測: ビット一致は取れたが 10% 遅い

融合カーネル `fused_gemm_psq4_wmma_body` は WMMA 2 回を *別々の* accumulator で受けて
いたため plain MB=1 経路と丸めが一致せず、`phaseshift-compute` の logits が 2 行目で
食い違った。同一 accumulator に連結するよう直すと **GENERATED_IDS・logits が完全一致**
（bit-exact）した。

その上で tg を interleaved で測定（ctx256 / tokens64、base = 規則なし、gateup = 規則あり）:

| pair | base | gateup |
|---|---|---|
| 1 | 38.36ms / 26.07 tok/s | 42.92ms / 23.30 tok/s |
| 2 | 38.22ms / 26.17 tok/s | 43.10ms / 23.20 tok/s |
| 3 | 37.63ms / 26.58 tok/s | 40.96ms / 24.41 tok/s |

**3 組すべてで +7〜12% 遅い**（base の絶対値が §7.57 の 36.3ms より大きいのは GPU 共有の
contention による。同一 session 内の相対比較が有効）。1088 block の launch 2 本の方が、
2176 block に連結した 1 本より速い。原因は特定していない（grid 順序/DRAM stream の
interleave が疑わしい）が、計測が明確なので採用しない（バンクマークルール）。

production は元の状態へ revert 済み（`linear_fusion_selector.cpp` の規則追加と
`fused_gemm_psq4_wmma_detail.h` の accumulator 変更の両方）。規則追加のみの状態で
`fused gate_up=960/16 step` が発火することは `PHASESHIFT_QWEN35_KERNEL_TRACE=1` で確認済み。

### 結論

decode の小 kernel（合計約 3.0ms + gap の一部）は launch 固定コスト支配で、
本モデルに適用できる fusion は無い（Qkv/GdnProj は dtype 不一致、GateUp は遅くなる）。
残る実利は §7.58 のとおり **gap 約 2ms**（HIP graph。可視範囲を batch context 経由に
する恒久対応が必要）と **gdn_recurrence 2.17ms** であり、小 kernel 本体の最適化では
1% 程度しか動かない。

---


---

## 7.82 prefill のコスト内訳（2026-09-21、PSQ4 最適化後）

§7.81 の PSQ4 decode 最適化を入れた後の PP2048 を `rocprofv3 --kernel-trace` で
カーネル別に集計した。測定窓 994.8 ms、kernel 合計 896.7 ms、kernel 間 gap 98.2 ms。

### まず pp への寄与

§7.81 の 3 変更のうち prefill に効くのは **Phase 1 の codebook 簡約だけ**
（`psq4_cb10_low_high` は PSQ4 prefill_2d も共有する）。Phase 2〜4 は rows=1 専用。

interleaved A/B（legacy helper ↔ v2、各 2 ラウンド、ビルドを交互に切替）:

| build | pp2048 t/s (4 測定) |
| --- | --- |
| LEGACY | 2044.78 / 2043.81 / 2043.88 / 2044.25 |
| v2 | 2055.13 / 2054.74 / 2054.69 / 2055.63 |

pp2048 **2044.2 → 2055.0 t/s（+0.53%）**。LEGACY が以前記録した baseline
2044.8 t/s を再現するので A/B は妥当。PSQ4 prefill kernel 単体（rows=2048）でも
k5120/n17408 −0.80%、k17408/n5120 −0.77%、k5120/n10240 −0.63%、k5120/n12288 −0.73%。

### kernel 別内訳（1 pass、% は測定窓 994.8 ms 基準）

| kernel | ms | % | dispatch | avg us |
| --- | ---: | ---: | ---: | ---: |
| gemm_psq4_w4a8_wmma_prefill_2d `<false,128,128>` | 499.93 | 50.3 | 304 | 1644.5 |
| gemm_psq4_w4a8_wmma_prefill_2d `<false,64,64>` | 4.54 | 0.5 | 32 | 141.9 |
| gdn_recurrence_wmma_impl | 130.43 | 13.1 | 48 | 2717.3 |
| gemm_psq8_w8a8_wmma_prefill_2d | 82.21 | 8.3 | 64 | 1284.6 |
| gdn_conv1d_bf16_f32 | 31.49 | 3.2 | 48 | 656.1 |
| attention_paged_prefill_bf16 | 30.02 | 3.0 | 16 | 1876.0 |
| op_swiglu_bf16 | 22.16 | 2.2 | 64 | 346.3 |
| activation_quantize_e4m3_vec（3 種） | 17.83 | 1.8 | 257 | — |
| op_mul_f32_f32_bf16 | 16.24 | 1.6 | 64 | 253.7 |
| op_silu（2 種） | 14.57 | 1.5 | 96 | — |
| rmsnorm（3 種） | 13.92 | 1.4 | 208 | — |
| op_residual_add_bf16 | 12.09 | 1.2 | 128 | 94.5 |
| gemm_bf16_splitk_wmma | 9.52 | 1.0 | 96 | 99.2 |
| rope_f32_bf16_pair | 2.71 | 0.3 | 32 | 84.7 |
| l2_normalize_bf16_f32 | 2.70 | 0.3 | 96 | 28.1 |
| op_split_bf16 / sigmoid / scale / kv_append / embed | ~4.5 | 0.5 | — | — |
| **kernel 合計** | **896.7** | **90.1** | 1669 | — |
| **kernel 間 gap（host）** | **98.2** | **9.9** | — | **59 us/dispatch** |

グループ集計: **PSQ4 prefill 504.5 ms（50.7%）** / GDN 161.9 ms（16.3%）/
PSQ8 prefill 82.2 ms（8.3%）/ host gap 98.2 ms（9.9%）/ elementwise ~76 ms（7.6%）/
attention 30.0 ms（3.0%）/ quant+norm+rope+bf16 GEMM ~55 ms（5.5%）。

### PSQ4 prefill_2d の効率

WMMA 単体の天井は実測 ~230-240 TFLOPS（`f32_fp8_fp8`、§7.5 のマイクロベンチ）。

| shape | dispatch | ms | avg us | TFLOP/s | peak 比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| K5120 N17408 | 128 | 272.70 | 2130.5 | 171.4 | 73% |
| N5120（K17408 / K6144） | 96 | 137.31 | 1430.3 | 172.6 | 74% |
| K5120 N10240 | 32 | 40.83 | 1275.9 | 168.3 | 72% |
| K5120 N6144 | 32 | 24.96 | 780.0 | 165.1 | 70% |
| K5120 N12288 | 16 | 24.13 | 1508.2 | 170.8 | 73% |
| K5120 N1024（`<64,64>`） | 32 | 4.54 | 141.9 | 151.3 | 64% |
| **合計 / 平均** | **336** | **504.5** | — | **173.0** | **73%** |

PSQ8 prefill は 1.34e13 FLOP / 82.21 ms = **163 TFLOP/s（69%）**。

### 読み

1. pp の半分は PSQ4 prefill_2d で、既に WMMA peak の 73%。旧 fp4 GEMM（peak の 30%、
   §7.5）から大幅に改善済みだが、残り ~27% = 約 136 ms/pass（pp の 13.7%）が伸びしろ。
2. 2 位の gdn_recurrence は 13.1%。48 dispatch × 2717 us = decode の
   48 layer × 64.8 us/token とほぼ同じ per-token 速度で、**系列方向に並列化されていない**。
   decode 8.1% / pp 13.1% の両方に効く共通ターゲット。
3. host launch gap が 9.9%（1669 dispatch / 98 ms、59 us/dispatch）。GPU が遊んでいる。
4. attention prefill は 3.0% で、elementwise 群（7.6%）より小さい。

優先順位: ① PSQ4 prefill の残り 27% ② GDN recurrence ③ host gap。

### 計測上の注意

- `rocprofv3 --kernel-trace` の `Grid_Size_X` は**総 work-item 数**（block 数 × threads）。
  例: N=17408 の prefill_2d は block 136 × 256 = 34816。
- トレースには**重みロード相**（~130-180 ms の host gap が 60 回以上）が混ざる。
  測定窓は「最後の 5 ms 超 gap の直後」から取る。これをやらないと wall span が
  10 s を超えて % が壊れる。

### 再現方法

```bash
rocprofv3 --kernel-trace -f csv -o /tmp/pptr -- \
  ./build/phaseshift-bench pp --model-dir models/Qwen3.8-27B-PSQ \
  --prompt-tokens 2048 --mode forward --page-tokens 16 --arena-gib 24 \
  --runs 1 --warmup 0
```

---


---

## 7.89 pp の kernel 間 gap: CP の power-gating と keep-alive（2026-09-21）

§7.82 で pp2048 の kernel 間 gap が 97 ms（9.9%、1669 dispatch）と判明していた。
本節はその機構を特定し、**keep-alive で +5.45%** を得た記録である。

### 現象

rocprofv3 --kernel-trace の kernel 間 gap は二峰性である:

| gap | 件数 | 割合 |
| --- | ---: | ---: |
| 4-5 us | 1178 | 70.6% |
| 5-20 us | 159 | 9.5% |
| **100-1000 us** | **316** | **18.9%** |
| >1000 us | 5 | 0.3% |

長い側は **262.0-265.4 µs（p10-p90）** と極めて一定で、合計 89.7 ms。
直前 kernel の時間でバケット化すると閾値がはっきりする:

| 直前 kernel | n | avg gap | 長い gap の割合 |
| --- | ---: | ---: | ---: |
| <50 us | 522 | 16.5 | 1.1% |
| 100-200 us | 247 | 4.2 | 0.0% |
| 600-800 us | 104 | 13.1 | 0.0% |
| 800-1000 us | 56 | 12.9 | 0.0% |
| **1000-1500 us** | 62 | 246.3 | **93.5%** |
| **1500-2500 us** | 258 | 259.3 | **98.8%** |

### 合成カーネルでの再現（profiler 無し）

純 compute の spin カーネル（依存 chain の multiply）で、event 計測した 1 launch の
時間と N launch の wall 平均を比べた。**profiler 無しでも再現する**:

| kernel 時間 | blocks | 1 launch (event) | per-launch (wall) | 差 |
| ---: | ---: | ---: | ---: | ---: |
| 833 us | 64 | 833.3 | 866.0 | +32.6（窓で ~0 に収束） |
| **1312 us** | 64 | 1312.1 | 1559.6 | **+247.5** |
| 1562 us | 1 | 1578.9 | 1815.5 | **+236.6** |
| 1575 us | 2 | 1575.2 | 1812.7 | **+237.5** |
| 1663 us | 64 | 1663.1 | 1887.0 | **+223.9** |
| 400 us | 1 | 401.8 | 394.7 | -7.1 |
| 420 us | 64 | 419.8 | 415.0 | -4.8 |

**閾値は約 1 ms、コストは約 235 µs で一定、block 数（1 / 2 / 64）に依存しない。**

### 機構: CP の power-gating

「GPU が idle だから」ではなく「**CP（command processor）が idle だから**」である。
背景負荷を変えて切り分けた（長いカーネル = 約 1.3 ms、64 blocks）:

| 背景 | single (event) | per-launch | gap |
| --- | ---: | ---: | ---: |
| なし | 1644.0 | 1877.3 | **+233.3** |
| 1 block の長いカーネル（GPU は busy） | 2509.6 | 2759.6 | **+250.0** |
| 200 × 1 block の長いカーネル | 2435.7 | 2716.8 | **+281.1** |
| **tiny kernel の洪水** | 1730.4 | 1725.7 | **-4.7** |

長いカーネルを背景に置いても gap は残る（GPU は busy でも CP は待ちだけ）。
**dispatch の洪水を流すと完全に消える。** カーネル実行中に CP が待ちだけになると
power-gate し、次の dispatch で wake-up に ~235 µs 払っている。

### keep-alive の間隔

`hipEventRecord` を別ストリームに周期的に流すだけで足りる（GPU compute を消費しない）。
間隔を振った pp2048（t/s、2 サンプル）:

| 間隔 | 1 | 2 |
| ---: | ---: | ---: |
| off | 2154.6 | 2152.8 |
| 300 us | 2258.0 | 2284.8 |
| 500 us | 2266.6 | 2270.8 |
| 800 us | 2276.4 | 2274.0 |
| 950 us | 2276.9 | 2284.9 |
| 1100 us | 2272.2 | 2271.9 |
| 1400 us | 2263.3 | 2261.5 |
| 1800 us | 2259.6 | 2260.2 |

950-1100 us が最良。1800 us でも効果は残るので CP の timeout は 1.8 ms より長い。

### 効果

interleaved A/B（device 1、6 ペア、pp2048、`--gap-keepalive 1000`）:

| | 1 | 2 | 3 | 4 | 5 | 6 | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 2154.03 | 2153.35 | 2154.08 | 2168.77 | 2157.24 | 2152.68 | 2156.7 |
| 1000 us | 2274.99 | 2271.54 | 2279.20 | 2276.55 | 2271.41 | 2271.59 | **2274.2** |

**pp2048 +5.45%**（全 6 ペアで改善）。内訳（rocprofv3、`--gap-keepalive 500`）:

| | off | keep-alive |
| --- | ---: | ---: |
| 窓 | 952.73 ms | 906.67 ms |
| kernel 合計 | 855.45 ms (89.8%) | 887.79 ms (97.9%) |
| gap 合計 | 97.28 ms (10.2%) | **18.88 ms (2.1%)** |

gap は -78.4 ms、kernel は +32.3 ms（keep-alive の干渉で +3.8%）。差し引き -46.1 ms。

### event flag の検討

干渉を下げる目的で `hipEventCreateWithFlags` の flag を試した（interval 1000 us、t/s 2 サンプル）:

| flags | 1 | 2 | 判定 |
| --- | ---: | ---: | --- |
| `hipEventCreate`（-1） | 2278.27 | 2282.61 | 効果あり |
| **`hipEventDisableSystemFence`（0x20000000）** | **2283.88** | **2283.65** | **効果あり・最良** |
| `hipEventDisableTiming`（0x2） | 2158.57 | 2159.18 | **効果なし** |
| 両方（0x20000002） | 2157.70 | 2159.82 | **効果なし** |

**`hipEventDisableTiming` を付けると keep-alive 効果が完全に消える。** 間隔を 50 us まで
詰めても 2162-2166（baseline 2160-2161）で変わらない。つまり **CP を起こしているのは
timestamp の書き込みそのもの**であり、timing を切ると CP がする仕事が無くなる
（runtime が GPU 側 packet を出さない実装と推測される）。

`hipEventDisableSystemFence` 単独は効果を保ったまま僅かに良い（interleaved 6 ペア:
2277.9 → 2281.1、+0.14%、6 ペア中 5 で改善）。結果を host から観測しないので
fence は不要である。既定をこれにした。内訳（rocprofv3）:

| | off | `DisableSystemFence` |
| --- | ---: | ---: |
| 窓 | 952.73 ms | **902.13 ms** |
| kernel 合計 | 855.45 ms | 885.06 ms |
| gap 合計 | 97.28 ms | **17.08 ms (1.9%)** |

最終 A/B（interleaved 6 ペア、`--gap-keepalive 1000`、既定 flag）:

| | 1 | 2 | 3 | 4 | 5 | 6 | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 2158.43 | 2157.66 | 2155.93 | 2160.68 | 2160.24 | 2158.55 | 2158.6 |
| 1000 us | 2282.89 | 2280.95 | 2282.48 | 2279.08 | 2280.09 | 2277.19 | **2280.4** |

**pp2048 +5.65%**（全 6 ペアで改善）。

### tg（decode）には効果がない

tg の gap は性質が違う。`--context 2048 --tokens 16 --prefill-chunk 2048` の trace を
時間方向に 10 分割すると:

| 区間 | dispatch | 平均カーネル | gap p50 | gap p90 |
| --- | ---: | ---: | ---: | ---: |
| 0（prefill） | 2850 | 308.7 us | 4.2 | **260.7** |
| 1-9（decode） | 2850 × 9 | 19.7-20.5 us | 3.9 | 4.2 |

decode は **1 token あたり 1603 dispatch、平均カーネル 20 µs、gap は 3.9-4.2 µs で均一**
で、1 ms を超えるカーネルが 1 つも無い。CP は常に仕事を持っているため power-gate せず、
keep-alive の対象外である。実測でも差はない（interleaved、`--gap-keepalive 1000`）:

| | 1 | 2 | 3 | 4 | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 27.16 | 27.18 | 27.18 | 27.17 | 27.172 |
| 1000 us | 27.19 | 27.17 | 27.18 | 27.17 | 27.178 |

差は 0.02% で、測定誤差の範囲である。

decode の gap は **CP の通常の dispatch latency（約 4 µs）**であり、削るには dispatch 数
自体を減らす必要がある（別のレバー）。`--prefill-chunk 128`（既定）の trace では
gap が 14.5%（16733 dispatch × 4.35 µs）になるが、これは prefill を 128 token ずつに
切った分の dispatch が積み上がったもので、同じ約 4 µs の latency である。

### 棄却した案: prefill chunking

カーネルを 1 ms 未満に分割すれば gap は消える。`pp` に `--prefill-chunk` を追加して
検証したが、**重みの再読が上回る**:

| | off | chunk=512 |
| --- | ---: | ---: |
| kernel 合計 | 855.45 ms | 913.32 ms（**+57.9**） |
| gap 合計 | 97.28 ms | 45.80 ms（-51.5） |
| 窓 | 952.73 ms | 959.13 ms（**+6.4**） |

PSQ4 prefill が 513.64 → 568.15 ms（+54.5）。M（token 次元）を分けると各 block が
重みを読み直すため。純損失なので棄却。

### 位置付け

keep-alive は driver/hardware の power 挙動への workaround であり、数値は一切変えない
（bit-exact は不変）。現状は bench の `--gap-keepalive US` として実装している。
production へ入れる場合は executor の execution strategy として置くのが自然だが、
thread と stream のライフサイクルを持つため別途設計が要る。

理想形は HIP event ですらなく、**shader を dispatch せずに CP だけを起こす最小 packet**
を queue に流すことである。`hipEventDisableTiming` が効果を消した事実は
「CP に必要なのは timestamp 書き込みという GPU 側の仕事」を示しており、
AQL の barrier packet に signal 書き込みを伴わせる形が候補になる。
ただし既存の HIP queue へ生 packet を差し込むと runtime の管理と衝突するため、
queue を自前で持つ GPU-MCU 経路（docs/RnD_MCU.md）と組み合わせる話になる。

### 再現方法

```bash
./build/phaseshift-bench pp --model-dir models/Qwen3.8-27B-PSQ \
  --prompt-tokens 2048 --gap-keepalive 1000 --mode forward \
  --page-tokens 16 --arena-gib 24 --runs 1 --warmup 0 --device 1
```


---

## 7.90 power limit 300 W での pp/tg 再測定（2026-09-21）

power limit を 300 W（MIN 210 / MAX 300、GFX max 2350 MHz）にして pp2048 と tg を測り直した。
300 W 化前の値は §7.89 と同じ条件（`--device 1`、interleaved）で取った過去の計測である。

| bench | 旧 power | 300 W | 差 |
| --- | ---: | ---: | ---: |
| pp2048, keep-alive off | 2158.6 | 2404.4（2403.75 / 2403.98 / 2405.49） | **+11.4%** |
| pp2048, keep-alive 1000 us | 2280.4 | 2574.2（2574.53 / 2572.81 / 2575.25） | **+12.9%** |
| tg（ctx 2048, 32 tok） | 27.17 | 27.37（27.37 / 27.40 / 27.34） | +0.74% |

累積（旧 power, off → 300 W, on）は 2158.6 → 2574.2 = **+19.3%**。

### 機構

rocprofv3 の窓内集計（pp2048, keep-alive off）:

| | 窓 | kernel | gap |
| --- | ---: | ---: | ---: |
| 旧 power | 952.73 ms | 855.45 | 97.28 |
| 300 W | 861.31 ms | 763.72 | **97.59** |

kernel は −10.7%、gap は不変。gap 1 件あたりのコスト（約 235〜262 µs）は clock 非依存なので、
kernel が速くなるほど相対比率が上がる（10.2% → 11.3%）。これが keep-alive の相対効果が
+5.65% → +7.06% に増えた理由である。

kernel 別（300 W, keep-alive off）:

| kernel | 旧 power | 300 W | 差 |
| --- | ---: | ---: | ---: |
| `gemm_psq4_w4a8_wmma_prefill_2` | 513.64 ms | 447.83 ms | −12.8% |
| `gdn_recurrence_wmma_impl` | 72.08 ms | 63.32 ms | −12.1% |

### keep-alive on で kernel 時間が増える

| | kernel | gap | 窓 |
| --- | ---: | ---: | ---: |
| 300 W, off | 763.72 | 97.59 | 861.31 |
| 300 W, on | 784.90（**+2.8%**） | 17.05 | 801.95 |

gap を埋めて GPU を連続稼働させると power/clock governor が clock を下げるため、kernel は
+2.8% 遅くなる。それでも gap の削減が上回る（旧 power でも +3.5% と同じ傾向）。

### tg が効かない理由

tg は decode で DRAM 律速のため +0.74% にとどまる。isolated な GDN recurrence マイクロベンチも
p50 1387〜1397 µs（旧 1373.8）で有意差が無い。単体で実行すると 300 W を消費して cap に達する
（`amd-smi metric --power` で 297〜300 W を観測）ため、power limit の変化に対する感度が低い。
一方 in-situ では前段の重い prefill kernel が clock を決めるため −12% が現れる。

### 注意

旧 power の値は別セッションで取得したものであり、machine の競合状態が厳密には同一でない。
300 W 内での keep-alive A/B（+7.06%）は同一セッションの interleaved 計測なので信頼できる。

### 再現方法

```bash
./build/phaseshift-bench pp --model-dir models/Qwen3.8-27B-PSQ \
  --prompt-tokens 2048 --mode forward --page-tokens 16 --arena-gib 24 \
  --runs 1 --warmup 0 --device 1 --gap-keepalive 1000

./build/phaseshift-bench tg --model-dir models/Qwen3.8-27B-PSQ \
  --context 2048 --tokens 32 --prefill-chunk 128 --mode forward \
  --page-tokens 16 --arena-gib 24 --warmup 0 --device 1
```


---

## 7.97 keep-alive の production 化（2026-09-21）

§7.89 で機構とパラメータを特定し、bench でのみ有効だった CP keep-alive を executor に
組み込んだ。**pp2048 +6.07%（3/3 対）**、tg は +0.26%（回帰なし）。

### 実装

`Executor` が所有する（`include/phaseshift/models/qwen35/runtime/executor.h` には
不透明ポインタ 1 個だけを追加し、`<thread>` を public ヘッダへ持ち込まない）:

- `create_model_executor` の最後で開始: 専用 `hipStream` + `hipEventCreateWithFlags(
  hipEventDisableSystemFence)` を作り、1000 µs 間隔で `hipEventRecord` を打つ thread を起動
- `executor_shutdown` の**先頭**で停止（thread を join → stream synchronize → event/stream 破棄）。
  他のリソースより先に止めることで、破棄中の event record が残らない順序にしている
- move ctor / move assignment は所有権を移す（`other.keepalive = nullptr`）
- 開始に失敗しても executor は動作を続ける（最適化が使えないだけ）。破棄側の失敗は
  `executor_shutdown` の first_error として報告する
- `PHASESHIFT_KEEPALIVE=0` で無効化できる（既定は有効）

### 効果（210 W、interleaved 3 対）

| bench | off | on | 差 |
| --- | ---: | ---: | ---: |
| pp2048 | 2222.2 | **2357.0** | **+6.07%** |
| tg（ctx 2048, 32 tok） | 27.14 | 27.21 | +0.26% |

bench の `--gap-keepalive 1000` で測った値（+6.0%）と一致する。
カーネルは一切変更していないので bit-exact は自明に保たれる（required acceptance 97/97）。

### 計測コマンド

```bash
PHASESHIFT_KEEPALIVE=0 ./build/phaseshift-bench pp --model-dir models/Qwen3.8-27B-PSQ \
  --prompt-tokens 2048 --mode forward --page-tokens 16 --arena-gib 24 \
  --runs 1 --warmup 0 --device 1
```


---

## 7.101 HIP graph が optimized-only batch で engage しない gate バグと、plan 署名 key / SPEC_VERIFY 対応（2026-09-23）

branch `fix/hipgraph-pool-warm`（base `fb6bbaf5`）、build は Release / gfx1201 /
`PHASESHIFT_HIP_GRAPH=ON`、GPU1。

### 症状

`HIP_GRAPH=ON` build の pp / tg / DFlash2 は `[graph] skip: staging pool not warm` のみを出し、
`capture` / `replay` を一度も出さない。`PHASESHIFT_QWEN35_KERNEL_MODE=correctness` にすると
capture する。すなわち kernel 種別の問題ではなく **eligibility gate の誤判定**である。

### 原因 A: `program_pool_warm` の `any_run`

`program_executor.hip` の dispatch loop は optimized が成功すると `continue` し、
`launch_host_binding`（`staging_pool.run_seen[slot]=1` を書く唯一の場所）を通らない。
27B-PSQ は全 dispatch が optimized なので `run_seen` が立たず、`program_pool_warm` の
`any_run` が永久に false → capture 分岐に到達しない。

修正: per-bucket の「その bucket の program を一度実行したか」(`staging_pool.bucket_ran`) を
`program_pool_warm` の条件にする。staging を使わない program も 1 回実行済みなら warm とみなす。

### 原因 B: cache key が可視範囲の生値

key は `min_visible_tokens` / `max_visible_tokens` の生値だったが、

- `min_visible_tokens` はどこからも参照されていない（selector にも kernel にも渡らない）。
- `max_visible_tokens` は paged attention の selector（Optimized/Correctness）と `splits`
  （2048 以上で 16 split）の判定にのみ使われ、**kernel 引数には焼き込まれない**。
  kernel は device 上の `row_positions` から可視長を計算する。

したがって decode / verify は step / round ごとに key が変わり、毎回 capture し直していた
（この状態の tg は 4658 → 5836 ms、**-20%**）。

修正: key を paged attention の **plan 署名**（`{optimized, use_prefill, splits}`）に置換し、
`min/max_visible` を key から除去。`paged_attention_dispatch` に
`plan_paged_attention` / `paged_attention_plan_signature` を追加し、
`try_launch_paged_attention` の plan 判定と共有する。`use_prefill` は key 計算時点で
`batch_context` が未更新のため、`exec_role==Prefill && R==1 && N>=128` で明示的に与える。
plan 署名を計算できない場合は graph を無効化して従来経路に落とす。

### 原因 C: SPEC_VERIFY の除外

`graph_eligible` の `!batch.speculative_verify` により verify は対象外だった。
DFlash2 の verify は draft round の約 88%（`verify_gpu_ms` 3493 ms / round_ms 4040 ms）を占める。

修正: **GDN history を渡す通常 verify のみ**対象にする。history を渡さない rerun-reference と
MTP は `spec_verify_without_history` として従来どおり動的経路に固定する。
key に `exec_role` / `has_spec_history` / `capture_rows` / history pointer を追加した
（decode と verify が同一 bucket を共有しうるため）。

### 実測（OFF = 既存 `build`、ON = 本修正、3 rep interleaved、paired median）

| workload | OFF | ON | delta |
| --- | ---: | ---: | ---: |
| pp2048 | 877.69 ms / 2333.41 t/s | 877.24 ms / 2334.60 t/s | -0.04%（neutral） |
| tg128 | 4727.38 ms / 27.08 t/s | 4539.28 ms / 28.20 t/s | **+3.48%** |
| DFlash2 prose256 | 64.17 t/s | 64.49 t/s | +0.40% |

tg の paired は 3/3 正（+3.48 / +4.02 / +2.42%）。pp は `--runs 1 --warmup 0` のため
replay が 1 回も入らず neutral。DFlash2 は `verify_gpu_ms` が 3492.8 → 3458.2 ms（-1.0%）。
`GENERATED_IDS`（sha1 `7ba06220c093`）と `GREEDY_TOKEN_SUM=2446188` は全 rep で一致。

capture 回数（`PHASESHIFT_GRAPH_DEBUG=1`）:

- tg ctx2048 tokens16: capture 2（prefill R2048 + decode R16）、残りは replay。
- DFlash2 prose256: capture 3（prefill + 最初の decode M=1 + verify）、verify は約 83 回 replay。

### 検証

| 項目 | 結果 |
| --- | --- |
| `ctest -L required`（`PHASESHIFT_HIP_GRAPH=ON`） | **131/131 PASS**（gpu1 66 / gpu_mcu 8 含む） |
| `test_dflash2_gate10_cli` | PASS |
| `test_dflash2_gate11c_e2e`（history / rerun-reference 一致） | PASS |
| `test_constraint_graph`（server） | 10/10 PASS |
| DFlash2 code / math の ids | OFF と一致 |
| graph OFF build（`build-plain`） | compile PASS。tg `GREEDY_TOKEN_SUM=2446188`、DFlash2 ids 一致（plan 抽出は graph-OFF の挙動を変えない） |

### 効果の上限について

HIP graph replay が削るのは kernel 間の submission gap だけである。27B-PSQ の decode /
verify は weight streaming 律速（1 forward ≈ 15.12 GB）なので gap は数 % が上限で、
実測 +3.5%（tg）/ -1.0%（verify_gpu）はその範囲に収まる。より大きな改善は
kernel 自体（weight streaming 効率、fusion で dispatch 数を減らす）か、
DFlash2 なら acceptance（1 round の emitted 数）側にある。

### 既知の限界

- `graph_cache` は bucket あたり 1 slot。decode と verify が同一 bucket を共有する workload では
  両者が slot を奪い合い、交互 batch で再 capture が起きうる（DFlash2 単発では prefill +
  初回 decode + verify の 3 capture で収束する）。
- history pointer を key に含むため decoder を差し替えると再 capture する。DFlash2 は
  `--serve-stdio` 非対応で decoder はプロセス内に 1 個なので実害はない。
- MTP の spec verify は GDN history を渡さないため対象外のまま（従来どおり動的経路）。

---
