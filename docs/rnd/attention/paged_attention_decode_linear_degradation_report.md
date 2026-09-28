> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# Paged Attention decode の線形劣化 問題レポート（27B / gfx1201）

- 作成: 2026-09-19
- 状態: **解決済み**（`7f74738`、`optimization_history.md` §7.48）
- 二次被害（selector の無言 fallback）: 解決済み（`d53f10f`、`../objective_vllm_mxfp4.md` §8.3）
- 対象: Qwen3.8-27B-PSQ（q24/kv4/hd256/pt16）、R9700 単一 GPU、BF16 KV

## 1. 概要

paged attention の decode カーネルの実行時間が **visible トークン数に線形** で
増大し（405.6 µs/層 @8k → 2.15 ms/層 @32k → 14.1 ms/層 @262k）、
decode スループットが ctx=128 の 26.79 t/s から ctx=32768 で 14.33 t/s
まで落ちていた。加えて、ctx=32768 では decode の visible（32769）が
selector の evidence cap（32768）を 1 超えるだけで **Correctness
参照実装へ無言 fallback** して 32.2 s/tok（optimized の約 950 倍）になる
「ハング」として観測されていた。

原因は 3 つの構造的要因の重なり:

1. **並列化不足** — grid = rows×q_heads = 24 block（bs=1）で、64 CU に対し 37.5%。
2. **直列 page walk** — 1 block が全 page を直列に walk するため、
   コンテキストが伸びても並列が増えず latency だけ線形に増加。
3. **GQA の 6× K/V 重複読取** — 24 個の q_head が同一の 4 個の KV を
   それぞれの block で再読取。

なお旧カーネルは帯域律速では**なく**、KV 固有トラフィックに対する
実効帯域は 8k 時点でピークの 8.6%（`../objective_vllm_mxfp4.md` §8.2）、
560 GB/s（実測ピーク）換算でも 11-15% 止まりであった。

対策は **KV split + reduce（flash-decoding 型 2 カーネル）** の追加で、
32k の attention/層は 5.4×、E2E decode は 14.33 → 22.65 t/s（1.58×）。

## 2. 現象

### 2.1 decode スループットの線形低下（修正前）

`bench tg` greedy 64 tok（`../objective_vllm_mxfp4.md` §8.1、修正前ベースライン）:

| context | t/s | ms/tok | attention 分（16 層） |
|---|---|---|---|
| 128 | 26.79 | 37.33 | ~0.3 ms |
| 2048 | 25.67 | 38.96 | 93.7 µs/層 × 16 = 1.5 ms |
| 8192 | 22.42 | 44.60 | 405.6 µs/層 × 16 = 6.5 ms |
| 32768 | 14.33 | 69.79 | 2.15 ms/層 × 16 = 34.4 ms |

paged-attention bench（rows=1、旧カーネル、p50 µs/層）は visible に
ほぼ正確に線形:

| visible | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 | 262144 |
|---|---|---|---|---|---|---|---|---|---|
| µs/層 | 27.8 | 50.2 | 93.7 | 205.8 | 362.0 | 730.6 | 2027.4 | 3842.4 | 14425.6 |

### 2.2 ctx=32768 の「ハング」（二次被害、修正済み）

- `kMeasuredMaxVisible = 32768`（evidence 上限、capability ではない）を
  超えると selector が `Correctness` を返す。**warning なし**に参照実装へ
  落ち、32.2 s/tok（optimized の約 950 倍）になる。
- 境界 A/B で確認: ctx32767 = 16.2 s/tok = (optimized 34ms +
  fallback 32s)/2 の半々、ctx32768 = 32.2 s/tok。
- 修正（`d53f10f`）: (24,4) heads rule の cap を max_position_embeddings
  （262144）へ拡張（bench `--check` で evidence 取得）、fallback 発生時に
  1-shot stderr warning を追加。

## 3. 原因分析

### 3.1 並列化不足（主因）

旧カーネルは grid = rows × q_heads、bs=1 で **24 block × 8 wave**。
gfx1201 は 64 CU なので、best case でも 24/64 = 37.5% の CU しか
稼働しない。しかもコンテキストが伸びても block 数は 24 のままで、
増加するのは各 block の serial 処理時間だけである。

「wave 数（MLP 深度）が律速」であることの直接証拠:

- **GQA group のみ** variant（block = (row, kv_head)、K/V 重複読取は
  解消したが split なし）: 4 block × 4 wave = **16 wave** で
  vis 32768 @ 5384 µs（旧 2145 µs の **2.5× 劣化**、実効 25 GB/s）。
  重複読取を消しても wave 数が減ると逆に遅くなる。
- **S sweep**（split 数 = block 数を 4 倍ずつ増加）: 8192 で
  S=4 → 279.8 µs、S=8 → 155.0 µs、S=16 → 106.0 µs。
  S が大きいほど常に最良（2048-262144 の全 visible で同様）。

### 3.2 直列 page walk

1 block が `block_table` を先頭から全 page 直列 walk する
（page=16 token、32k で 2048 page）。DRAM リクエストは 8 wave の
stride で出るが、block 内の完了は最後の page 待ちになるため
latency ∝ visible。実施順 1 の計測（`todo.md`）で
「visible に線形 = 直列 walk 起因」を確認済み。

### 3.3 GQA の 6× K/V 重複読取

q24/kv4 の GQA では 6 個の q_head が 1 個の kv_head を共有する。
旧カーネルは q_head 毎に block を立てるため、**同一 KV を 6 block で
重複読取**（32k で固有 134 MB/層 → 重複換算 804 MB/層）。
L2 で一部吸収されるが、DRAM トラフィックと in-flight 深度の両方を
劣化させる。

### 3.4 帯域律速ではないことの確認

KV 固有トラフィック = visible × 4 kv × 256 d × 2 B × 2 (K+V)
（8k: 33.5 MB、32k: 134 MB、262k: 1.07 GB/層）。

| visible | 560 GB/s 換算の下限 | 旧カーネル実測 | 下限到達率 |
|---|---|---|---|
| 8192 | 60 µs | 362-406 µs | 14.8% |
| 32768 | 239 µs | 2027-2145 µs | 11.8% |
| 262144 | 1.91 ms | 14.1-14.4 ms | 13.5% |

帯域が律速なら長くなるほど下限到達率は上がるはずだが、ほぼ一定の
10% 台で頭打ち = **occupancy / latency 律速**である。

## 4. 対策（実装済み、`7f74738`）

**KV split + reduce の 2 カーネル**（flash-decoding 型、
`optimization_history.md` §7.48 に詳細）:

- **kernel A** `attention_paged_split_bf16_impl`: grid = rows × kv_heads × S。
  1 block = (row, kv_head, split) が page の S 分割区間を扱い、
  GQA グループ内全 q_heads（27B: 6）の online softmax を
  4 wave（128 threads）で処理。K/V はトークン毎 1 回だけ読取
  （§3.3 の重複を構造から除去）。epilogue で 4-wave merge し
  partial (m, P, A[256]) を global に書込（正規化しない）。
- **kernel B** `attention_paged_split_reduce_bf16_impl`:
  grid = rows × kv_heads。g = max_s m_s、W_s = exp(m_s − g) として
  `out = Σ_s W_s·A_s / Σ_s W_s·P_s`。
- **dispatch**（`paged_attention_dispatch.hip` 内の execution strategy、
  selector / public API 変更なし）: BF16・decode（非 prefill）・
  `max_visible ≥ 2048` で **S=16 固定**。当初の 8192→4 / 16384→8 /
  ≥65536→16 の段階 S 案は計測で S=16 が常に最良（§3.1 の S sweep）
  のため廃止。8MiB 固定 partial workspace（`kDecodeAttnPartialBytes`）
  を超過する rows（27B で >21）は旧カーネルへ自動 fallback。
- 旧カーネル（visible < 2048 と prefill）は byte-identical に維持し、
  短コンテキストのベースラインを不変に保つ。

## 5. 結果

### 5.1 per-shape（bench paged-attention、rows=1、p50 µs/層）

| visible | 旧カーネル | split（S=16） | 向上 | 下限到達率（560 GB/s） |
|---|---|---|---|---|
| 2048 | 93.7 | 52.6 | 1.78× | 16.0% → 28.5% |
| 8192 | 362.0 | 106.0 | 3.41× | 16.6% → 56.5% |
| 16384 | 730.6 | 186.8 | 3.91× | 16.4% → 64.1% |
| 32768 | 2027.4 | 376.9 | **5.38×** | 11.8% → 63.6% |
| 65536 | 3842.4 | 708.3 | 5.43× | 12.5% → 67.7% |
| 262144 | 14425.6 | 3081.3 | 4.68× | 13.3% → 62.2% |

- rows=8: 8192 で 640.1 → 401.5（1.60×）、32768 で 2850.0 → 1948.9
  （1.46×、134 MB×8 = 1.07 GB の BW 下限 1.92 ms に到達）。
- 262144 で 5.4× に届かないのは、65536 比で実効帯域が
  67.7% → 62.2% とわずかに下がるため（原因は特定済み bug では無く、
  超大容量の単一ストリーム DRAM 読取が理論値より低い実測ピーク
  560 GB/s へ近づく領域）。重要なのは**超線形の崩れが解消された**こと:
  65536→262144（visible 4×）で時間は 4.35×、32768→262144（8×）で
  8.17× と、帯域律速の線形スケーリングに収まっている。

### 5.2 E2E（27B-PSQ、`bench tg` greedy 64 tok）

| context | 修正前 | 修正後 | 変化 |
|---|---|---|---|
| 128 | 26.79 t/s | 26.80 t/s | 不変（経路不変） |
| 2048 | 25.67 t/s | 26.03 t/s | 1.01×（SUM bit 一致） |
| 8192 | 22.42 t/s | 25.39 t/s | 1.13×（SUM bit 一致） |
| 32768 | 14.33 t/s | **22.65 t/s** | **1.58×** |

32768 の 25.7 ms/tok 削減 = 16 層 × (2027 − 377) µs の attention
差分と整合。

- 事前目標との対比: `../objective_vllm_mxfp4.md` §8.2 の提案目標
  「decode ctx=32k: 14.3 → 24 t/s（attention split-K で 128 ctx 相当に
  復帰）」に対し、実測 **22.65 t/s**（目標の 94%）。32k の step は
  weight floor 27.0 ms + attention 6.0 ms + GDN/elementwise/launch の
  overhead（~11 ms）で 128 ctx 相当（26.79 t/s = 37.33 ms/tok）には
  構造的に届かず、22.65 t/s（44.1 ms/tok）が attention 改善後の
  自然な到達点。

### 5.3 数値整合性

- bench `--check`（correctness reference 比 2e-3）:
  27B（rows 1/8 × vis 128..262144 × S 2/4/8/16）と
  4B（q_per_kv=4、rows 1/8 × vis 128..32768）全 PASS。
- 旧カーネルと split カーネルの直接比較（vis=32768、同一 input）:
  **max_abs = 4.3e-6、max_rel = 8.3e-7**（f32 丸め・softmax 計算
  順序差のみ）。
- ctx=32768 greedy は token 7 で 1 点分岐（248320 vocab の top-2
  near-tie がカーネル差 8e-7 で flip）。旧カーネル固定ビルドは
  ベースラインと bit 一致（SUM=464326）を再現しており、
  分岐は split 経路の計算差起因の統計的 flip であることを確認済み。

## 6. 残課題

1. **4B（q16/kv4）の selector cap は 32768 のまま**
   （`kMeasuredMaxVisible`）。4B で visible > 32768 の decode は
   今でも fallback（warning は出る）。27B 用の (24,4) rule だけ
   262144 になっている。
2. **fp8 KV decode の 2.75× 遅延**（旧カーネル自体の異常）が未調査。
   split 経路は BF16 のみ有効（fp8 は旧カーネル維持・凍結候補）。
3. **262k の production A/B（tg）は未実施**（prefill が数分）。
   カーネル自体は bench `--check`（262144、S=16）で検証済み。
4. attention は 62-68% 帯域到達率で頭打ち（旧カーネルの 11-16% と比べ
   4-5 倍の効率）。次の decode 改善は weight ストリーミングの削減で、
   P1: GDN の PSQ8 層（li%4==0 の 16 層、3.27 GB）を PSQ4 化 →
   13.49 GB → 天井 41.5 t/s（+12%）、P2: + lm_head PSQ4 → 12.85 GB →
   43.5 t/s（+17%）（`../objective_vllm_mxfp4.md` §8.2）。精度ゲート
   （ppl/GSM8K）が前提（ユーザー判断待ち）。

## 7. 参考: 外部の同型問題

vLLM 系でも「single-sequence decode が context length で collapse
する」のと同じ類型であり、flash-decoding（KV split + partial reduce）
が既知の解決形（`todo.md`「長コンテキスト decode attention」節）。
本実装はその paged-KV 版（page 単位の split）であり、
selector / public API を変更せずに executor 内部の
execution strategy として導入している（Topoロジー哲学に適合）。
