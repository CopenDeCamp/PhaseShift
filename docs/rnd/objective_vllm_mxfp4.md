> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# 目標リファレンス: vllm-mxfp4（One card, TP=1）

新たな性能目標は、[vllm-mxfp4](https://github.com/GGZ14/vllm-mxfp4) リポジトリの
**One card (TP=1)** プロファイル
（[該当セクション](https://github.com/GGZ14/vllm-mxfp4#one-card-tp1-1)）の
単一カード推論性能水準に、本リポジトリを到達させることである。

> 量子化方式の違い（PSQ vs MXFP4）により達成は難しい可能性があると判断している。
> 作業仮説は「精度で勝って、速度で負ける」（§6 参照）。

---

## 1. 参照リポジトリ（そのもの）

- **URL**: <https://github.com/GGZ14/vllm-mxfp4>
  （目標値のアンカー: [#one-card-tp1-1](https://github.com/GGZ14/vllm-mxfp4#one-card-tp1-1)）
- **名称**: vllm-radiance (MXFP4)
- **正体**: AMD Radeon AI PRO R9700（gfx1201 / RDNA4）向けのコンテナ化 vLLM
  推論サーバー。ROCm + PyTorch + Triton + AITER + vLLM スタックに RDNA4 固有の
  patch（Python patch スクリプト約 40 本をコンテナ起動時に pinned な vLLM イメージ
  へ適用）とカスタム kernel（libr4d）を組み込む構成。
- **対象モデル**: Qwen3.8-27B。ネイティブ 4-bit MXFP4
  （e2m1 code + e8m0 scale / 32 要素、4.25 bpw）+ FP8 推測 drafter
  （DFlash2、SPEC=7）でサーブする。
- **対象ハード**: 1-2 枚の R9700（本環境と同じカード）。
- **自己申告の品質**: WikiText-2 ppl 8.3708、GSM8K 500q greedy 97.8%。
- **自己申告の状態**: "Early dev, experimental. Not production hardened."

## 2. 目標数値（One card TP=1）

出典はすべて相手側の自己申告値（BetterBench `--quick`、2026-09-16、
1× R9700 @ 210 W cap / -75 mV、`serve-tp1.sh` 既定
（MAXSEQS=8、context 65,536、DFlash2 SPEC=7、fp8 KV、GDN state fp16/bf16、
ネイティブ MXFP4 checkpoint））。

### 2.1 シングルストリーム（推測デコード込み）

| Category | TTFT p50 (ms) | update p50 (ms) | tok/update | decode t/s |
| --- | --- | --- | --- | --- |
| code | 95.0 | 35.3 | 4.65 | 145.2 |
| json | 94.9 | 35.1 | 5.59 | 184.4 |
| math | 94.4 | 35.2 | 5.56 | 168.6 |
| summarization | 98.0 | 35.1 | 4.67 | 138.1 |
| file_edit | 99.6 | 35.3 | 5.63 | 166.1 |
| reasoning | 94.8 | 35.2 | 3.60 | 120.8 |
| chat | 96.9 | 34.9 | 2.75 | 81.4 |
| prose | 50.4 | 35.2 | 2.76 | 79.0 |

**Combined**: decode **137.7 t/s**、update p99 **35.7 ms**、TTFT p50 **95 ms**。
同一カードの直前値は 64 ms/update（75.4 t/s）であり、2026-09-16 のパスで
1.79x 改善した値。

### 2.2 並行

| Concurrent | Aggregate t/s | TTFT p50 (ms) | Per-request decode t/s |
| --- | --- | --- | --- |
| 1 | 116.9 | 94.8 | 146.3 |
| 2 | 207.6 | 136.6 | 138.3 |
| 4 | 319.0 | 147.8 | 108.4 |
| 8 | **416.3** | 172.9 | 79.4 |
| 16（MAXSEQS=16） | 486.6 | 533.4 | 47.8 |

### 2.3 Prefill

| Target depth | Prompt tokens | TTFT p50 (ms) | PP t/s |
| --- | --- | --- | --- |
| 2k | 1,514 | 553.6 | 2,737 |
| 8k | 5,892 | 2,163.9 | 2,720 |
| 16k | 11,800 | 4,245.9 | 2,778 |
| 32k | 23,548 | 8,880.3 | 2,651 |
| 64k | 47,014 | 18,949.9 | 2,480 |

電力ポリシーとのトレード（同一設定）: 300 W（undervolt 維持）では prefill
2k 3,552 / 64k 3,192 t/s、conc-8 471 t/s。一方シングルストリーム decode は
悪化（core が 3.3 GHz を超えると 42 ms/update）。

この設定での GSM8K 250q greedy: 97.6-98.4%。

### 2.4 Single-GPU プロファイルの設定（TP=1 のみ有効化）

- context 65,536（2 枚構成の既定 262,144 は単一カードに載らないため）
- attention KV cache: **fp8**（TP=2 と同じ）
- GDN recurrent state: **fp16**、conv window **bf16**（fp32 非許容 → GDN page
  1648 → 880 tokens、vLLM の attention block も連動して半分）
- decode GEMM の幅上限が unsharded `gate_up`（34816 幅）を覆い、decode kernel
  経路に載せる
- fp8 residual stream epilogue（all-reduce 不要）、fused GDN step（48 heads
  routing）、CHUNK 4096
- KV pin 6,535,819,798 bytes（計測済みプロファイル行 `1x7551-32624`）

## 3. 数値の比較可能性（重要）

1. **相手側の decode 数値は推測デコード込み**である。1 update（35.7 ms）=
   draft（7 tokens）+ verify（target の forward 1 回、M = batch×(SPEC+1) = 8）で、
   **tok/update 2.75-5.63** を産む。137.7 t/s は推測後のスループット
   （単位時間あたりのトークン数）であり、target モデルの裸 decode 速度
   ではない。
2. 本リポジトリは **draft モデルを持たない**（DFlash2 は範囲外と決裁済み）。
   MTP 重みのロード経路は存在するが、speculative ループは無い。
3. したがって「137.7 t/s に到達する」は現行スコープでは直接達成できない目標
   であり、目的は比較可能な指標に分解して設定すべきである:
   - **Prefill t/s**（量子化・推測の双方に非依存で直接比較できる）
   - **spec 無し step 時間 / t/s**（target 側の verify cost に対応。相手側は
     spec 無し数値を公表していないため、自前の絶対目標を設定し、相手側の
     1 update 内の target forward cost の推定値と突合する）
   - **並行 aggregate t/s**（スケジューラ / バッチリング能力）
   - **キャパシティ**（KV token 数、context 長、単一カード上の定常消費）
4. 相手側は 210 W cap / -75 mV という電力ポリシーで計測している。
   比較のためには本側のベンチマークも同一パワーグリッドに固定する必要がある
   （現行ベースラインの計測条件との差を確認すること）。

## 4. 取り入れられそうな変更

「相手側の報告値 × 本リポジトリへの適用性」の順。本側の現行状態は
2026-09-19 時点のコード確認による。

1. **GDN recurrent state の fp16 化（conv window bf16）**
   相手側: state 容量が半分（GDN page 1648→880 tokens）、decode の state
   トラフィック削減。
   本側: recurrent state は fp32（`gdn_recurrence_f32`、dispatch は
   F32 以外の state を拒否）、conv state は既に bf16。
   → 採用候補。f16 state + wmma recurrence 変種の追加と精度検証
   （logit cos / E2E）が要る。
2. **fp8 KV を既定化**
   相手側: TP=1/TP=2 とも fp8 KV（attention kernel の fp8 経路）。
   本側: BF16/FP8_E4M3 の paged decode/prefill kernel は揃っており、
   `--kv-cache-dtype fp8_e4m3` で opt-in（既定は BF16）。
   prefix cache の checkpoint は BF16 pool 限定。
   → 採用候補。精度確認のうえ既定切替 + prefix cache の fp8 対応。
3. **decode / prefill GEMM タイル分離と split-K の in-place fused reduction**
   相手側: decode tile（TM=ceil(M/16)、BK=128、split-K）/ prefill tile
   （BM=256、BK=64）を分離。split-K の reduce を atomic counter による
   in-place fused reduction にし、step 時間 -3.9%（bit 一致）。
   本側: `rows==1` の split-K（S=4/8/16）は存在するが、small M（2-64）への
   適用と reduce の経路（2nd launch 有無）は未確認。
   → 調査候補。decode の M 分布（MTP verify は M=batch×(k+1)）に合わせる
   small-M 経路の設計と、split-K reduce の launch 削減。
4. **decode GEMM の M 上限（small-M ルーティング）**
   相手側: M ≤ 64（MAXSEQS>8 なら 128）を decode kernel へ。prefill tile は
   M=5 で実効 row 5 に対して 51x の無駄 MAC を発するため。
   本側: mb tile dispatch は rows==1 中心。
   → 調査候補。small-M バンドの split-K タイルへの統合。
5. **R4D 型 transposed attention（S^T = K·Q^T）**
   相手側: wave32 matrix-core fragment で 1 lane 1 query row、softmax が
   lane 内で完結。prefill +14.6%（64K）、+4.1%（16K）。条件: head_dim 256、
   paged block 16、GQA 6、causal、bf16/fp8 KV。
   本側: 対象モデル（Qwen3.8-27B）の形状条件は確認済み
   （head_dim 256 / q 24 heads / kv 4 heads = GQA 6）。
   paged attention は自前 kernel（page 16 必須条件との整合は要確認）。
   → 調査候補。既存 paged decode/prefill への transposed score 適用の
   費用対効果を、prefill bench（256+ tokens）で測る。
6. **skinny GEMM（small-M projection の split-K）**
   相手側: M ∈ [6,64] の projection を split-K kernel へ。GDN `in_proj_ba`
   （480 KiB × 48 回/step）28.5 µs → 3.6 µs（約 8x）。
   本側: `gemm_gdn_proj_bf16_wmma` は存在。small-M 時の split-K 適用有無は
   未確認。
   → 調査候補。gdn_proj の rows=1-64 を bench して判定。
7. **KV メモリの計測 pin（calibration）**
   相手側: vLLM の保守的 profiling（一時アクティベーション peak 控除）に対し
   計測 pin で cache 5.7% 増。
   本側: arena 事前割当 + page 数計算。KV/weight/workspace の割当は静的。
   → 相当する作業は「arena 配分 rightsizing」。本側の KV 割当余白を
   実測で回収できるか確認。
8. **電力 / クロックポリシー**
   相手側: 210 W cap + -75 mV で decode 有利、300 W で prefill 有利
   （prefill 2k: 2,737 → 3,552 t/s）。decode は core 3.3 GHz 超で劣化。
   本側: ベースライン計測のパワーグリッド記録が未整備。
   → 採用候補（計測手法として）。比較ベンチの前提条件として固定し、
   必要なら運用設定として文書化。
9. **GDN conv1d の channel block 拡大**
   相手側: prefill conv の channel block 拡張で kernel 2.2x（bit 一致）。
   本側: conv は `gdn_prepare` 内（bf16 state、48 channel 系）。
   → 調査候補。conv 部分の occupancy 改善の有無を profile で確認。
10. **prefix caching + GDN state の境界 snapshot（align 相当）**
    相手側: GDN 再帰状態を block 境界で snapshot/restore し、shared prefix
    で TTFT 大幅削減（full recompute と bit 一致を確認済み）。
    本側: batcher 系の prefix cache（KV page + GDN state の checkpoint）は
    存在（BF16 KV 限定）。
    → ほぼ既存。fp8 KV 対応と GDN snapshot の bit 一致検証が残作業。
11. **fused RMSNorm+quant / GDN norm+gate epilogue**
    相手側: イメージ内で既定有効。
    本側: `rmsnorm_quantize` / `swiglu_quantize` / `gdn_norm_gate` の
    fusion dispatch 既存。
    → 既に対応。
12. **preshuffled native layout GEMM**
    相手側: preshuffled FP8 blockscale GEMM を実装。
    本側: preshuffle native layout + WMMA kernel 既存
    （2026-09-19 に `ps::weights::WeightLoadOptions` で共通層に統一済み）。
    → 既に対応（概念も同じ）。

### 取り入れられない / 範囲外

- **推測デコード（DFlash2 drafter + int2 draft head + verify ループ）**:
  相手側 137.7 t/s の最大項（tok/update 2.75-5.63）。draft モデルは範囲外。
  後段で MTP 重みを用いた spec ループを導入する場合に再検討。
- **TP=2/3、all-reduce、dummy head padding**: 本リポジトリは設計上
  単一 GPU。
- **NVFP4 → MXFP4 オンライン再量子化**: 相手側の checkpoint 固有の橋渡し
  （double rounding で 0.158 relRMS の損失）。本側は PSQ checkpoint を
  直接生成する。
- **MTP head の fp8 rewrite（`fp8_mtp.py`）**: 相手側 checkpoint の
  Quark exclude 記述不備への workaround。本側に該当なし。

## 5. 利点と欠点

### 本リポジトリ（PhaseNonShift）の利点

- ネイティブ HIP ランタイム: Python / PyTorch / Triton / コンテナ無し。
  相手側は ~40 GiB のダウンロード（image + checkpoint + drafter）、初回
  起動時の Triton/inductor コンパイル（数分）を毎回払う。
- 単一ターゲット特化（gfx1201 + Qwen3.5 系）: 抽象化税と patch チェーン無し。
  相手側は pinned vLLM イメージへの Python patch 約 40 本を起動時に適用する
  構成で、vLLM アップグレードは非公式（監査スクリプトのみ）。
- 量子化の精度設計（§6）: 同じ bit 級で scale 表現が細かい。
- 決定論的実行、arena 事前割当、明示的 ownership、ロード時検証
  （CRC / layout contract テスト）。
- preshuffle native layout 哲学は相手側と同一（WMMA 高速化のため）。

### 本リポジトリの欠点

- 推測デコードが無い（相手側 decode の最大項）。
- GDN recurrent state が fp32（相手側 fp16）→ state 容量・トラフィック 2x。
- KV 既定が BF16（相手側 fp8）→ KV 容量半分。
- small-M GEMM / attention の tuning 深度が相手側に劣する可能性
  （decode tile 分離、split-K fused reduction、transposed attention、
  skinny GEMM）— 未計測で断定できない。
- 電力 / クロックの計測ループが無い。
- モデル 1 系統・GPU 1 枚（設計上）。

### 相手側（vllm-mxfp4）の利点

- 計測されたエンジニアリング台帳（`PERFORMANCE.md`: 各最適化の寄与と
  gate 条件を 1 行ずつ記録）— 本リポジトリの bankmark ルールと同等の文化。
- 推測デコード（DFlash2 block diffusion + int2 draft head + 動的 draft
  深度）で decode 2.75-5.63 tok/update。
- サービング機能一式: 並行バッチ（conc-8 416 t/s）、prefix caching、
  KV calibration、OpenAI 互換 API。
- kernel ズーの広さ: R4D attention、skinny GEMM、GDN WMMA、preshuffle
  GEMM、all-reduce（TP 系）。
- 多フォーマット対応（MXFP4 / NVFP4 / ParoQuant int4・int5）。
- 単一カードでも 27B + drafter + 65k context の定常運用を成立させている。

### 相手側の欠点

- スタックの脆さ: pinned image + 起動時 patch chain。vLLM バージョンは
  固定、変更は監査スクリプトによる手作業。
- 損失近似の連鎖: e8m0 scale（2 のべき乗のみ）、fp8 KV、fp16 GDN state、
  NVFP4→MXFP4 の double rounding。精度を速度・容量のトレードに使っている。
- 自己申告 early dev / experimental、production 非耐性。
- 設定の複雑さ: `RADIANCE_*` 環境変数の集合、TP=3 の padding hack
  （TP=3 では既定の 2 つの production default が無効化され decode -3~6%）。
- 一部機能はバグのため既定 OFF（lazy GDN snapshots: multi-turn 破損）。
- 40 GiB 級のディスク消費、コンテナ運用。

## 6. 量子化方式の違い（PSQ vs MXFP4）

| | 相手側 MXFP4 (W4A8) | 本側 PSQ4 (W4A8) | 本側 PSQ8 (W8A8) |
| --- | --- | --- | --- |
| 重み code | e2m1（8 値 + sign） | 4-bit PSQ code（8 値 + sign、codebook 最適化） | 8-bit e4m3 code |
| 重み scale | e8m0 / 32 要素（2 のべき乗のみ） | f16 / 32 要素（任意値） | f16 / 32 要素 |
| 重み bpw | 4.25 | 4.5 | 8.5 |
| 活性化 | e4m3 + e8m0 scale / 32 | e4m3 + f32 scale / 32 | e4m3 + f32 scale / 32 |
| WMMA 命令 | fp8×fp8（16×16×16） | fp8×fp8（16×16×16） | fp8×fp8（16×16×16） |

（本側の WMMA 命令・scale 配置は
`src/phaseshift/models/qwen35/kernels/linear/gemm_psq4_w4a8_wmma.gfx1201.hip`
等のコード確認による。PSQ4 の 8 値 codebook は kernel 内の定数
`{0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5} × scale`。）

- **精度**: 32 要素グループごとに、相手側は 8-bit の 2 のべき乗 scale
  （e8m0）しか持たず、code も e2m1 の粗い格子（2-4 領域は 25% ステップ）。
  本側は f16/f32 の任意 scale + codebook 最適化の 8 値格子。同 bit 級で
  scale 表現が細いため、精度で勝つ構造はコード上で支持される。
- **速度**: GEMM の核心（WMMA 命令・presuffle・split-K）は同じ fp8×fp8
  クラスであり、**量子化方式由来の構造的な速度差は GEMM には存在しない**
  （kernel 間の head-to-head 未計測 — §7 で実施）。
- **「速度で負ける」の実体は量子化方式そのものではなく**:
  (a) 推測デコードの有無（137.7 t/s の最大項）、
  (b) decode tile / split-K reduce / attention / small-M の kernel tuning
  深度、
  (c) fp8 KV・fp16 GDN state による容量差とそれに伴うバッチ可能数、
  と考えられる。
- モデル全体 bpw: 本側 Qwen3.8-27B-PSQ は 5.5411 bpw（18 GiB、bf16 部
  込み）。相手側 MXFP4 は 27B 本体 4.25 bpw（~14.4 GiB）+ fp8 MTP head +
  2 GiB drafter。単一カード上の定常消費はほぼ同水準。

## 7. 未決項目 / 次の手

1. **目標の定義**: §3 の分解指標（prefill t/s / spec 無し step / 並行
   aggregate / キャパシティ）に対する絶対目標値。§8 のベースライン
   計測完了により、提案値を §8.2 に記述。
2. ~~27B-PSQ のベースライン計測~~ → §8 完了（2026-09-19）。
3. **GEMM head-to-head**: §8.4 の計測経路修復（per-op event / wall-clock）
   が先決。現状の gemm bench 数値は使用不可。
4. **精度検証の同一基準化**: 27B-PSQ で WikiText-2 ppl / GSM8K を
   相手側と同条件（greedy、温度 0）で計測し、「精度で勝つ」を数値で確認。
5. **採用候補の triage**（§8 を踏まえて更新）:
   1. paged attention decode の KV split（flash-decoding 型）—
      ctx 8k/32k の decode 劣化の直接対策（8k で 6.5ms、32k で 32.5ms/step）
   2. GDN 層 PSQ8→PSQ4 の精度評価 — 現在の PSQ8 は li%4==0 の 16 GDN 層
      （down/qkvz/out、3.27GB）のみ。全 GDN 層 PSQ4（P1）で
      15.12→13.49 GB → 天井 47.1 t/s（+12%）、+ lm_head PSQ4（P2）で
      12.85 GB → 49.5 t/s（+18%）。新 preset（psq4gdn / psq4all）で
      ppl/GSM8K ゲートを実施
   3. GDN state fp16 化（2.6ms/step）
   4. paged prefill の大 visible 効率（§8.5）— long-prompt TTFT
   5. fp8 KV 既定化は凍結候補（fp8 decode は bf16 の 2.75 倍遅い実測）

## 8. ベースライン計測: Qwen3.8-27B-PSQ / R9700（210 W cap, 2026-09-19）

計測条件: greedy・spec 無し・PwrCap 210W・`--prefill-chunk 2048`
（生産の row bucket R2048 に相当）。

### 8.1 計測結果

| 項目 | 値 | 備考 |
|---|---|---|
| decode ctx=128 | **26.79 t/s**（37.33 ms/tok） | |
| decode ctx=2048 | 25.67 t/s（38.96 ms/tok） | |
| decode ctx=8192 | 22.42 t/s（44.60 ms/tok） | |
| decode ctx=32768 | **14.33 t/s**（69.8 ms/tok） | §8.3 修正後 |
| prefill 2048（単発 batch） | 1718 t/s（582 µs/tok） | pp bench。8k/32k は row bucket 超過で単発不可 |
| conc=8（prefix 512, cold） | **272 tok/s** aggregate | step_tokens_saved 33.0%、greedy_match=YES |
| GDN recurrence rows=1 | 54.7 µs/層 → 2.6 ms/step | 48 層 |
| paged attention decode | vis8192: 405.6 µs/層、vis32768: 2.15 ms/層 | 16 層。vis に線形 |

### 8.2 decode は weight ストリーミング帯域律速

- decode step 当たりの weight 読取 bytes = **15.12 GB**
  （full attention 16 層 全 PSQ4 = 2.98 GB、GDN 48 層 = 10.87 GB
  （うち PSQ8 は 3.27 GB = li%4==0 の 16 層、`fpx/profile.cpp` の
  `FfnDown / GdnQkvza / GdnOut → (li % 4 == 0) ? PSQ8 : PSQ4`）、
  lm_head PSQ8 = 1.27 GB、embedding は lookup）
- DRAM 読取ピーク = **640 GB/s（nominal）/ 実測 636**（AGENTS.md R9700 spec。
  本セッション単体計測では 560 GB/s（3 手法一致）で、clock / cap 条件の差と
  推定。天井計算は 636 を採用）→ **floor 23.8 ms → 天井 42.1 t/s**
- 実測 37.33 ms/tok = 実効 405 GB/s = **ピークの 64%**（floor 比 64%）。
  overhead（~13.5ms）内訳: GDN recurrence 2.6ms、launch gap 等 2-3ms、
  PSQ8 効率差、elementwise/state R/W
- ctx 128→32768 の delta（+32.5 ms）は**ほぼ全て paged attention decode**
  （2.15 ms/層 × 16）。attention の実効帯域は 8k 時点でピーク ~12%
- 相手側 137.7 t/s は推測デコード込み（tok/update 2.75–5.63）。
  implicit decode は 24.5–50 t/s 帯 = **本側と同一バンド**。
  相手が上位なのは (a) spec の乗数、(b) bpw 差（MXFP4 4.25 vs 本 5.54）
- **天井を超える手段**（weight bytes 削減。PSQ8 実分布 = li%4==0 の
  16 GDN 層 3.27GB + embedding/lm_head 2.54GB、profile.cpp `psq()`）:
  - P1: GDN の down/qkvz/out を全層 PSQ4 → 13.49 GB → **天井 47.1 t/s（+12%）**
  - P2: P1 + lm_head PSQ4 → 12.85 GB → **天井 49.5 t/s（+18%）**
  - 精度ゲート（ppl/GSM8K）で判断（§7-5.2）。embedding は decode 無関係
- **提案目標値**（現行量子化のまま、efficiency 改善のみで達成可能な範囲）:
  - decode ctx≤8k: 26.8 → **30 t/s**（実効 405 → 454 GB/s = ピークの 71%）
  - decode ctx=32k: 14.3 → **24 t/s**（attention split-K で 128 ctx 相当に復帰）
  - prefill 2048: 1718 → **2400 t/s**（相手 2.5–2.8k へ接近）
  - spec 込み 137.7 t/s 到達には spec decoding（2.75–5.63× 乗数）が必須

### 8.3 発見: paged attention セレクタの無言 fallback（修正済み）

`kMeasuredMaxVisible = 32768`（docs/todo.md 既定の「evidence の上限」）
が、ctx=32768 で decode visible = 32769 になり **Correctness パスへ
無言 fallback**（32.2 s/tok ≈ optimized の 950 倍）。

- 境界 A/B: ctx32767 = 16.2 s/tok = (optimized 34ms + fallback 32s)/2、
  ctx32768 = 32.2 s/tok
- 修正: evidence を max_position_embeddings（262144）まで拡張
  （paged-attention bench `--check` で decode bf16/fp8・prefill bf16 全て
  PASS）し、(24,4) heads rule の cap を 262144 に引き上げ。
  fallback 発生時に 1-shot stderr warning を追加
  （`runtime/paged_attention_dispatch.hip`）

### 8.4 発見: bench の両端 event 計測が非決定論的（gemm 数値は無効）

この ROCm / gfx1201 環境では、複数 kernel バッチの GPU event 両端計測が
wall-clock 真値に対して 2× / 0.1× で非決定論的にずれる
（per-op event と wall-clock は一致）。

- `gemm psq4/psq8` bench の全数値（baseline_27b/gemm_*.txt）は使用不可。
  wall-clock 参考値: PSQ4 gate_up（47.8 MB, rows=1）b2b 55 µs
- tg / pp の数値は BW モデル・context スケーリングのクロス検証で整合
  （15.12 GB / 37.33 ms = 405 GB/s）するため信頼可能
- 後続の bankmark は per-op event か wall-clock で計測する

### 8.5 発見: paged prefill が大 visible で異常に遅い

rows=2048 の paged prefill（optimized）:

| visible | p50 / 層 | 備考 |
|---|---|---|
| 32768 | 49.3 ms | ~67 GFLOPS（効率 ~0.2%） |
| 262144 | 8842 ms | 線形予測比 22×（超線形劣化） |

32k chunked prefill（16 チャンク）の合計は ~7s でベースラインには
影響しないが、64k+ の long prompt では TTFT のボトルネックになる。
todo「長コンテキスト decode attention」の直列 page walk 問題の prefill 版。

---

## 9. 推測デコードの drafter: DFlash2（2026-09-21 追記）

§2 の decode 数値（tok/update 2.75〜5.63）がどの drafter で出ているのかを
参照リポジトリの README と drafter のモデルカードで確認した。

### 9.1 既定は MTP ではなく DFlash2

- `SPEC_METHOD` の既定は **`dflash`**。`mtp` は `--no-drafter` のときの fallback で、
  その場合 `SPEC=4`（dflash は `SPEC=7`）。
- drafter は `tcclaviger/Qwen3.8-27B-DFlash2-FP8`（2 GiB / 2B params）で、
  <https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2> のミラー。
  target 側の MTP head は fp8 へ rewrite して**ロードはする**（AMD checkpoint の
  `exclude` が module 名でなく tensor 名で書かれていて load に失敗するため。
  `fp8_mtp.py` が rewrite する）。
- DFlash2 の説明（モデルカード）: block-diffusion drafter。
  ブロック全体を**1 回の並列 forward** で生成し、各位置の上位候補を保持して
  軽量 selector が一貫したパスを 1 本通す。backbone の **two-tap dynamic
  convolution** が「ブロック末尾へ向かう draft の劣化」を防ぐ。
  lossless（greedy は target と完全一致）。

### 9.2 内蔵 MTP との同条件比較（モデルカード）

task ごとに 7 draft token、acceptance length = 生成トークン / 検証ステップ。
条件: SGLang・1× H200・target/draft とも FlashAttention 3、
**temperature 1.0 / top-p 0.95 / top-k 20**、最大生成 4096、reasoning effort `xhigh`。

| Task | MTP | DSpark | DFlash 2 |
| --- | --- | --- | --- |
| GSM8K | 5.02 | 4.36 | **5.46** |
| MATH-500 | 4.72 | 3.92 | **5.28** |
| HumanEval | 3.91 | 3.30 | **4.39** |
| MBPP | 3.99 | 3.51 | **4.79** |
| MT-Bench | 3.74 | 3.01 | **4.10** |

同じモデル（Qwen3.8-27B）の内蔵 MTP でも **3.74〜5.02 / step** 出ている。
つまり参照側の高い tok/update は「MTP を使いこなしている」からではなく、
**別学習の block-diffusion drafter に載せ替えている**ことが主因で、
内蔵 MTP を使った場合でもカテゴリ依存の同種の値が出る。

### 9.3 数値を読むときの注意

- **sampling 条件**。上表は temperature 1.0、BetterBench 側も temp 0.7。
  speculative decoding の受容は rejection sampling 則なので、
  **greedy の argmax 一致より受容長が高く出る**。
  本リポジトリの `SpecDecoder` は greedy のみ。
- **context 長**。参照側の prompt は 1.5k〜47k token。
- **タスク mix**。カテゴリ表の振れ幅（prose 2.76〜file_edit 5.63）を
  重み付き平均した値が "combined"。

### 9.4 DFlash 論文の主張

<https://arxiv.org/abs/2602.06036>（ICML 2026）。
自己回帰的な drafting は逐次性が残り speedup を制限する。
target 条件付けのない小型拡散 drafter は受容長が伸びない（PARD は ~3x 天井）。
target の hidden に条件付けた block diffusion が受容長と並列性を両立する
（"the target knows best"）。

### 9.5 本リポジトリに関係する README のノブ

| 変数 | 既定 | 内容 |
| --- | --- | --- |
| `FAST_DRAFT` | `1` | int2 draft head + exact rerank（mtp で +6.5% decode） |
| `RADIANCE_VERIFY_HEAD` | dflash `1` / mtp `0` | target の verify lm_head も int2 に |
| `RADIANCE_DYNAMIC_WIDTH` | `1` | 受容 EMA に基づく scheduler 側の per-request verify 幅 |
| `RADIANCE_GDN_LAZY` | `0`（2026-09-17 以降） | 「draft token ごとの snapshot」ではなく「base state 1 + candidate stash」で GDN 状態を保持。multi-turn を壊すバグのため既定 OFF |
| `disable_padded_drafter_batch` | — | mtp の single-stream で ~+50% と README に記載 |

`RADIANCE_GDN_LAZY` は本リポジトリの partial accept rerun（`docs/rnd/mtp/mtp.md` §5.1）
と同じ問題に対する参照側の答えである。
