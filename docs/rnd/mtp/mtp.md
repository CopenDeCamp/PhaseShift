> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# MTP（speculative decoding）現状診断

27B-PSQ の MTP を「何が悪いのか」から洗い出した記録。

- 日付: 2026-09-21
- commit: `215f8396`（`perf/psq8-prefill` merge 後。作業ツリーは clean）
- device: 0 と 1（両者で同一の数値を確認。A/B は同一 device 内で取る）
- power: 210 W cap
- model: `models/Qwen3.8-27B-PSQ`（64 layer hybrid、full attention 16 / GDN 48、
  MTP 1 層、`tie_word_embeddings=true`）

production 経路は `SpecDecoder`（`spec_decoder_step`）で、その呼び出し元は
tests と `phaseshift-bench mtp --spec`。`phaseshift-cli` / `phaseshift-compute` に
MTP の入口は無い。

---

## Current outcome

- **production 採用**: MTP 自体は production serve path へは採用されていない。CLI から
  起動できる drafter は DFlash2 である。ただし `spec_greedy_accept` と
  `spec_gdn_snapshot` / `spec_gdn_restore` は MTP の研究で確立した共通 contract として
  DFlash2 の production path へ採用された。現在の仕様は
  [../../developer/mtp.md](../../developer/mtp.md)、性能は
  [../../perf/current.md](../../perf/current.md) を参照。
- **採用**: MTP の weight contract / lowering / persistent KV / K>1 draft /
  speculative transaction（greedy accept + GDN rollback）/ verify の Exact numeric mode /
  dynamic draft depth + discard。n-gram tail も実装した。
- **rejected された主要案**:
  - fused LM-head: logits materialize の削減分が約 0.08% のため不採用。
  - lazy GDN state（参照実装の `RADIANCE_GDN_LAZY` 相当）: correctness 問題で不採用。
    partial accept の rerun 除去は exact な per-row state checkpoint で行う。
  - INT2 近似 draft head: exact + dynamic が target-only を上回ったため今回は不導入。
  - `tg.hip` の draft chain を `hidden_out_normed` に変える案: カテゴリごとに勝敗が
    入れ替わり総合差が無いため production の `hidden_out` を維持。
  - dynamic draft policy の既定 ON: pending_hidden の all-accept 行修正後、static K1 が
    dynamic + discard を上回ったため既定 `dynamic=false`。
- **訂正済みの旧結論**: §3.1 の rerun 除去概算（1.5〜1.7x）は、その後の実測で
  verify 本体が partial accept rerun であり M 非依存でないことが判明した
  （[optimization_history.md](optimization_history.md) §7.74）。
  §3.6 の「既定では lossless でない」は §7 の修正で解消した。
  §3.6 の「bench harness の欠陥」は §8 の修正で解消した。

## Gate index

| Gate | 内容 | 結果 | 記録 |
| --- | --- | --- | --- |
| 0 | MTP 重み読み込み契約（名前・形状・dtype・意味・レイアウト） | PASS | [Gate 0](#gate-0-weight-loading) |
| 1 | MTP 1 層 draft forward の vLLM 参照一致 | PASS | [Gate 1](#gate-1-single-layer-draft-forward) / §7.60 |
| 2 | persistent MTP KV / teacher-forced state synchronization | STATE CONTRACT PASS / REFERENCE NUMERICS PASS_WITH_AMBIGUOUS_NEAR_TIES | [Gate 2](#gate-2-persistent-kv-and-state-synchronization) / §7.61 |
| 3 | K>1 autoregressive draft / speculative transaction / greedy verification / rollback | core PASS | [Gate 3](#gate-3-speculative-transaction) / §7.62-7.63 |
| 4A.1 | per-tensor trace による M 依存 divergence 特定と VERIFY_EXACT | PASS | §7.64 |
| 4B | VERIFY_EXACT の verify role 限定 | PASS | §7.65 |
| 4C | verify D2H / sync baseline | SKIPPED（non dominant） | §7.66 |
| 4E | fixed-K qualification / break-even | PASS | §7.67 |
| 5A-5D | draft cost profile / confidence calibration / dynamic draft depth | QUALIFIED PASS | §7.69-7.77 |

Gate log の詳細は本ファイル末尾「Gate 記録（historical）」と
[optimization_history.md](optimization_history.md) §7.60-7.77 を参照。

---

## 1. 計測条件

| 項目 | 値 |
| --- | --- |
| harness | `build/tests/test_qwen35_mtp_spec_perf`（optional test。`-DPHASESHIFT_BUILD_OPTIONAL_TESTS=ON`） |
| context | 32（PP32、`tests/data/mtp_perf/*.psktok`） |
| generation | 64 token |
| prompts | prose / code / json / reasoning × 各 N（下表は N=2） |
| numeric mode | `PHASESHIFT_VERIFY_EXACT=1`（27B-PSQ で lossless に必須。§7.68） |
| 参考（draft 単体） | `phaseshift-bench mtp`（prose ctx256、teacher-forced） |
| 参考（draft/accept のカテゴリ別） | `phaseshift-bench tg --mtp`（ctx256、TG48） |

`test_qwen35_mtp_spec_perf` の `avg_accept` は **draft 1 本あたり**の受容率、
`tok/update` は **1 update あたりの生成トークン**（= accepted/update + 1）である。
混同しやすいので本稿では両方を明示する。

---

## 2. 現状の数値

### 2.1 K 別（PP32、8 prompt、TG64）

| mode | ms/token | tok/s | tok/update | accepted/update | accepted/draft | full accept | reject0 | fwd/update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| spec OFF | 35.908 | 27.85 | 1.000 | — | — | — | — | 1.000 |
| K1 | 30.680 | 32.59 | 1.759 | 0.759 | 0.746 | 75% | 25% | 1.254 |
| K2 | 30.162 | 33.15 | 2.236 | 1.236 | 0.620 | 49% | 25% | 1.507 |
| K4 | 34.279 | 29.17 | 2.626 | 1.626 | 0.406 | 15% | 27% | 1.851 |
| K8 | 42.578 | 23.49 | 2.681 | 1.681 | 0.212 | 0% | 27% | 2.000 |

- `fwd/update` = target forward の回数 / update。**1 を超えた分が partial accept の rerun**。
- 最良は K1〜K2 で **約 1.17〜1.19x**。draft を 4→8 と倍にしても accepted/update は
  1.626 → 1.681 でほぼ増えない。

### 2.2 K4 の stage 内訳

| stage | ms | 割合 |
| --- | ---: | ---: |
| draft | 3069.6 | 17.9% |
| verify | 14066.8 | 82.1% |
| gdn_snap | 2.4 | 0.01% |
| gdn_restore | 0.8 | 0.005% |
| commit | 0.3 | 0.002% |
| unattributed | -1.0 | — |

verify 14.07s / 195 update = 72.1 ms/update、forward 数 1.851 → **1 forward ≈ 39 ms**。
1 forward は 15.12 GB の weight 読みで決まる（M 非依存。§7.74: M=1 37.05 ms に対し
M=2 37.83 ms = 1.02x）。

### 2.3 draft 1 step（prose ctx256、`phaseshift-bench mtp`）

- `mtp head: 3.787 ms/call` = decode step の 10.5%。
- weight 読みの内訳（計算値）:
  - tied lm_head = body の `embed_tokens`（**PSQ8**、248320×5120）= codes 1.27 GB
    + scale 0.08 GB → **2.12 ms**
  - MTP 層 8 linear（**BF16**）= 240 MB → 0.38 ms
  - floor 合計 2.5 ms → **実測は 66% 効率**。
- per-step の CPU-GPU 往復（`mtp_executor.hip:404-597`）:
  hidden D2D 1 + H2D 4（token / descriptor / rope / batch context）+
  `hipStreamSynchronize` 1 + D2H 3（status / sampled / top2）。
  draft loop は step ごとに完全に直列化される。

### 2.4 受容のプロファイル（近似）

異なる K の run は生成列が変わるため厳密な条件付き確率ではないが、
K1/K2/K4/K8 の accepted/update と full accept から逆算すると:

| 深さ | 条件付き受容 |
| --- | ---: |
| P(d1) | ≈ 0.75 |
| P(d2 \| d1) | ≈ 0.65 |
| P(d3 \| d1d2) | ≈ 0.49 |
| d4 以降 | 崩壊（K8 の d5〜d8 合計 ≈ 0.055） |

teacher-forced の上限（`phaseshift-bench mtp`、prose ctx256、正解 hidden を与える）:

| depth | 一致 |
| --- | ---: |
| 1 | 46.2% |
| 2 | 46.2% |
| 3 | 15.4% |
| 4 | 7.7% |
| 5 以上 | 0.0% |

`P(d2|d1) = 83.3%`、`P(d3|d1d2) = 20.0%`。

### 2.5 カテゴリ依存（`tg --mtp`、ctx256、TG48、K4、device 0/1 で一致）

| category | chain=pre accept | chain=pre ms/token | chain=post accept | chain=post ms/token |
| --- | ---: | ---: | ---: | ---: |
| code | **3.08** | 17.33 | 3.00 | 16.86 |
| json | **1.67** | 32.30 | 1.53 | 34.24 |
| reasoning | 1.63 | 32.94 | **1.78** | **31.02** |
| prose | 1.18 | 42.37 | **1.53** | **36.79** |

- **code では 3.0/update**。参照実装の内蔵 MTP（HumanEval 3.91、ただし sampling 条件。
  `docs/rnd/objective_vllm_mxfp4.md` §9）と同帯。
- prose/chat で低い。ここが block-diffusion drafter の強みが出る領域。
- draft chain の hidden 源（pre = `hidden_out` / post = `hidden_out_normed`）は
  **カテゴリごとに勝敗が入れ替わり、総合では差が無い**。production の `hidden_out` を
  変える根拠にはならない。

---

## 3. 悪いところ（効く順）

### 3.1 partial accept の rerun が verify の 20〜50% を食う ← 最大

> Superseded by §7.74: verify 本体が partial accept rerun であり M 非依存ではないため、
> 下記の「rerun を消した場合の概算」は最終結果ではない。exact な per-row state
> checkpoint で rerun を除去する方針は DFlash2 へ採用された。

`spec_decoder.cpp:431-479`。全 draft が accept されなかったとき、
GDN state を snapshot へ戻し **accept 済み prefix をもう一度 target へ forward** する。
KV は block table の truncate で済むが、逐次 recurrent な GDN state だけは再計算が要る。

| mode | full accept | rerun 率 | fwd/update | rerun が forward に占める割合 |
| --- | ---: | ---: | ---: | ---: |
| K1 | 75% | 25% | 1.254 | 20% |
| K2 | 49% | 51% | 1.507 | 34% |
| K4 | 15% | 85% | 1.851 | 46% |
| K8 | 0% | 100% | 2.000 | 50% |

K を増やすほど損が増えるため、accepted/update が伸びるのに K>1 が伸びない直接の原因。

**rerun を消した場合の概算**（1 forward = 39 ms、draft = 3.787 ms）:

| mode | 現状 ms/token | rerun 無し | 対 spec OFF |
| --- | ---: | ---: | ---: |
| K1 | 30.68 | 24.3 | 1.48x |
| K2 | 30.16 | 20.8 | 1.73x |
| K4 | 34.28 | 20.6 | 1.74x |

なお rerun の forward が作り直しているのは **GDN state だけ**である。
`pending_hidden` は verify の `final_hidden[accepted]` で足りる（causal なので
row `accepted` の出力は rerun と一致する）。

### 3.2 受容が深さで崩れる（accepted/update が 2.6 で飽和）

§2.4 のとおり d4 以降がほぼ伸びない。draft を 4→8 に倍やしても +0.055/update。
MTP head は 1 層自己回帰で、深さ 2 以降の入力が自分の出力（学習分布外）になるため。

### 3.3 verify = target forward そのもの

§2.2 のとおり verify が 82%。target forward は weight-bound（15.12 GB / 実効 405 GB/s
= ピーク 636 の 64%）で M にほぼ依存しない。したがって
**「1 update = 1 forward」の構造を活かすには accepted/update を上げるしかない**。

### 3.4 draft step の無駄

§2.3。weight floor 2.5 ms に対し実測 3.787 ms。
- tied lm_head が PSQ8 のまま（1.35 GB）。PSQ4 相当にすれば -1.1 ms/step の見込み。
  target が検証するので lossless は保たれる。
- per-step の host 往復（sync + D2H 3 + H2D 4）で draft loop が直列化。
- PSQ8 の `rows==1` GEMV は `gemm_psq8_w8a8_wmma_auto` 経由で使われている
  （`psq8.hip:597`）。MTP 層の bf16 は `gemm_bf16_exact_rows_kernel`（M=1〜16）。

### 3.5 既定では lossless でない／harness では完全一致しない

> Superseded by §7: 27B-PSQ の spec ON ≠ spec OFF と間欠 FAIL は §7 の修正
> （verify-Exact GDN と decode GDN の代数一致、`output_gather` のホスト読み除去）で解消した。
> 本節は修正前の診断記録である。

- `PHASESHIFT_VERIFY_EXACT=1` が無いと verify（M=K+1）の sample が M=1 decode とずれ、
  **spec ON ≠ spec OFF**（§7.68）。この env は `create_spec_decoder` でしか読まれず、
  既定 OFF。
- `PHASESHIFT_VERIFY_EXACT=1` でも本 harness では **token_match 4/8〜10/16**。
  §7.68 が警告する「27B では harness で state を使い回すと不一致が出る」に該当
  （`test_qwen35_mtp_spec_perf` は `MtpKvState` を K ごとに 1 個作って prompt 間で
  使い回す）。true な losslessness の gate には使えない。

### 3.6 production 入口が無い／周辺が古い

- `create_spec_decoder` / `spec_decoder_step` の呼び出し元は tests と
  `phaseshift-bench mtp --spec`。
- `MtpDraftPolicy` の閾値は §7.76 の pending_hidden 修正**前**の calibration。
  修正後は static K1 が dynamic+discard を上回っており、既定 `dynamic=false`。
- **bench harness の欠陥（2026-09-29 に修正。経緯は §8）**:
  - `src/apps/bench/mtp.hip` の `--spec` は独自の古い speculative decode loop を持ち、
    MTP KV の prefill をせず（`spec_mode` 相では `run_mtp_rows` を呼んでいない）、
    partial accept で **GDN restore をしなかった**。→ K1 0.78x /
    `greedy equivalence: FAIL` の原因。同条件の `tg.hip` が accept 0.65 のところ
    mtp bench は 0.39 と食い違った。修正後は production `SpecDecoder` を使う。
  - `test_qwen35_mtp_spec_perf`: arena を prompt ごとに bump 確保して解放しないため
    prompts=4 で K4 が `arena capacity exceeded` になる。

### 3.7 補足（バグではない）

MTP の `logical_length`（書き込み slot）は `min(accepted+1, K)` で進み、
RoPE position は `accepted+1` で進むため all-accept のたびに slot が position より
1 遅れる（`spec_transaction_commit`）。ただし attention の可視範囲も同じ counter から
作られ、entry は時刻順に詰まるため RoPE は正しいまま。vLLM の
`+min(accepted+1, K)` cap と同じ挙動であり、実装バグではない。

---

## 4. 参照リポジトリとの差（要約）

詳細と出典は `docs/rnd/objective_vllm_mxfp4.md` §9。

- 参照側の既定は MTP ではなく **DFlash2**（別学習 2B の block-diffusion drafter、
  `SPEC_METHOD=dflash`）。MTP は `--no-drafter` 時の fallback（`SPEC=4`）。
- block diffusion はブロック全体を 1 回の並列 forward で生成し、target の hidden に
  条件付け、selector で 1 本のパスを通す。backbone の two-tap dynamic convolution が
  「ブロック末尾の draft 劣化」を防ぐ = §3.2 の対策。
- ただし**同じモデルの内蔵 MTP でも 3.74〜5.02/step**（モデルカード、7 draft）出ている。
  本リポジトリの code での 3.08 と同帯であり、MTP 実装が壊れているわけではない。
- 数値差の一部は計測条件: 参照は sampling（temp 0.7〜1.0）、prompt は 1.5k〜47k、
  "combined" は code/json/math 寄りの重み付き平均。本リポジトリは greedy。

---

## 5. 次の手（見込み順）

1. **rerun の除去**（§3.1）→ 概算 1.5〜1.7x。方法は 2 案:
   - verify 中に GDN recurrent state を **row ごとに snapshot** し、
     `accepted` 行のものを restore する（(K+1) × 150 MB ≈ 750 MB の書き込み ~1.3 ms）。
   - GDN 層の入力を保存し、**GDN 層だけ replay** する（~5〜10 ms）。
   参照側の `RADIANCE_GDN_LAZY` が同種の解（ただし既定 OFF）。
2. **draft cost の削減**（§3.4）: lm_head の低 bit 化、per-step 往復の除去。
   参照実装の設計は `docs/references/spec_decode_loops.md`。
3. **verify 幅の動的制御の再調整**（§3.6）。参照側の `RADIANCE_DYNAMIC_WIDTH` 相当。
4. **production 化**: VERIFY_EXACT 相当の numeric mode を既定にし、CLI から有効化。
5. **harness の修正**（§3.6）→ 完了（§8）。1〜3 の効果は
   `phaseshift-bench mtp --spec` で正しく測れる。

---

## 6. 使用コマンド

```bash
# production 経路（SpecDecoder）。optional test が必要
cmake -S . -B build -DPHASESHIFT_BUILD_OPTIONAL_TESTS=ON
cmake --build build --target test_qwen35_mtp_spec_perf
PHASESHIFT_MODEL_DIR_MTP=models/Qwen3.8-27B-PSQ \
PHASESHIFT_MTP_PERF_DIR=tests/data/mtp_perf \
PHASESHIFT_MTP_PERF_CONTEXT=32 PHASESHIFT_MTP_PERF_GEN=64 \
PHASESHIFT_MTP_PERF_PROMPTS=2 PHASESHIFT_VERIFY_EXACT=1 \
  build/tests/test_qwen35_mtp_spec_perf

# draft 単体（teacher-forced の深さ別一致と accept 統計）
./build/phaseshift-bench mtp --model-dir models/Qwen3.8-27B-PSQ --device 1 \
  --arena-gib 24 --context 256 --steps 32 --draft-k 8 \
  --tokens-file tests/data/mtp_perf/prose.psktok --chain pre

# production SpecDecoder contract で draft / verify / accept（§8。既定 verify=Exact）
./build/phaseshift-bench mtp --model-dir models/Qwen3.8-27B-PSQ --device 1 \
  --arena-gib 24 --context 256 --steps 48 --draft-k 1 \
  --tokens-file tests/data/mtp_perf/prose.psktok --spec --spec-only

# draft/verify/rerun を通した accept（legacy。GDN restore あり）
./build/phaseshift-bench tg --model-dir models/Qwen3.8-27B-PSQ --device 1 \
  --context 256 --tokens 48 --tokens-file tests/data/mtp_perf/prose.psktok \
  --mode greedy --mtp --draft-k 4 --mtp-chain pre \
  --page-tokens 16 --arena-gib 24 --warmup 0 --device 1
```

注意: `phaseshift-bench mtp --spec` は §8 以降 production `SpecDecoder` contract を
使うので計測に使える。spec OFF / ON の token 完全一致が条件で、不一致なら
終了コード非 0。`tg --mtp` は verify numeric mode を `Fast` のまま使うため
27B-PSQ では greedy からずれる（§3.5 / §7.68）。
`test_qwen35_mtp_spec_perf` は arena と state 使い回しの問題があり、
K4/K8 を prompts 4 以上で回すと途中で落ちる。

---

## 7. 修正（2026-09-21）: 27B-PSQ の spec ON ≠ spec OFF と間欠 FAIL

### 7.1 症状

- `test_qwen35_mtp_spec_perf`（27B-PSQ、ctx128、GEN64、`PHASESHIFT_VERIFY_EXACT=1`）で
  `token_match=7/8`。不一致は K に依らず同一 prompt・同一 token index（prose、index 41）。
- `test_qwen35_mtp_gate3_replay`（27B-PSQ、new32）で
  `FAIL: GDN recurrent + conv state serial replay bit-exact`。
  差分は 381/37,748,736 要素、約 3e-5 の微小数値差。
- `test_qwen35_mtp_gate3_replay`（27B-PSQ、new16 K4）が **同一条件で間欠的**に
  `FAIL: sync`（20 回中 16 回）。エラーは
  `optimized_dispatch.hip:268: HIP error (output_gather optimized launch): invalid argument`。
- 4B-PSQ では再現しない（geometry 依存）。
- target forward 自体は `test_qwen35_mtp_verify_divergence`（27B、ctx128、K=1/4/8）で
  `FIRST_MISMATCH=none`（全 65 layer bit-exact）。すなわち数値ではなく state 管理の問題。

### 7.2 原因 A: verify-Exact GDN と decode GDN の代数不一致

- `16dadca2` で decode M=1 専用の `decode1` カーネルを導入。選択条件は
  `head_k==128 && head_v==128 && key_heads==16 && num_v_heads==48`（27B geometry のみ）。
- 一方 verify-Exact は `wmma_serial`（chunked カーネル + serial rows）を使っていた。
  両者は数学的に等価だが浮動小数の丸め順が異なるため、GDN recurrent state が
  少数要素だけずれる。near-tie で argmax が反転し spec ON ≠ spec OFF となる。
- 4B では `decode1_supported` が false で decode も `wmma_serial` と同じカーネルに
  なるため、不一致が出ない。これが geometry 依存の理由。

### 7.3 原因 B: `output_gather` が device `batch_context` をホストで読む

- `launch_output_gather` が `ctx->num_outputs`（device ポインタ）をホストで読んで
  `grid.y` を決めていた。
- MTP は `batch_context` を `hipMemcpyAsync` で更新した直後に `execute_program` するため、
  ホスト読みが非同期コピーに追いつかず garbage を `grid.y` にして
  `hipErrorInvalidValue` を返す。これが間欠 FAIL の直接原因。

### 7.4 修正

- A: `GdnRecurrenceArgs.row_override` を追加し、
  `launch_gdn_recurrence_f32_wmma_decode1_serial` を新設。`num_requests==1` のとき
  `actual_rows` 行を `decode1` で 1 行ずつ逐次起動する。dispatch の `verify_exact_active`
  分岐で decode1 geometry ならこれを使う。decode と同一カーネルなので bit-exact が
  構造的に保証される（geometry 非対応・multi-request は従来の `wmma_serial`）。
- B: `launch_output_gather` にホスト引数 `num_outputs` を追加し、呼び出し側が
  `ctx.actual_outputs` を渡す。device struct をホストで読まない。

### 7.5 検証

| 検証 | 結果 |
| --- | --- |
| `test_gdn_recurrence_decode1`（required） | 23/23（新規 `serial rows=2/4/8` を含む） |
| `gate3_replay` 27B new32 K1/K8 | GDN serial replay bit-exact → PASS |
| `spec_perf` 27B ctx128 Exact | 全 K `token_match=8/8`（修正前 7/8） |
| `gate3_replay` 27B new16 K4 ×20 | 20/20 PASS（修正前 16/20 FAIL） |
| MTP required 合成 7 本 | 全 PASS |
| 4B-PSQ `spec_real` / `spec_perf` Exact | PASS / 全 K 8/8 |

### 7.6 pp/tg（target-only、変更の影響なし）

- pp2048: 863.79 ms（2370.95 tok/s）
- tg ctx2048 tok32: 27.32 tok/s

---

## 8. 修正（2026-09-29）: `phaseshift-bench mtp --spec` を production SpecDecoder へ

### 8.1 症状

`phaseshift-bench mtp --spec` が `greedy equivalence: FAIL` となり、
同条件の `tg --mtp` が accept 0.65 を出す箇所で 0.39 しか出なかった。
harness が SpecDecoder を使わず、独自の古い speculative decode loop を持っていた。

### 8.2 原因

1. `--spec` の parse が `spec_mode = true; spec_only = true;` であり、
   MTP prompt KV の teacher-force prefill が `if (!spec_only)` の中にあった。
   そのため `--spec` では MTP prompt KV が構築されない。
2. それでも最初の draft は `run_mtp_head(mtp, hid, in_tok, P + k, stream)` を呼び、
   `prefix_length = context - 1` 相当を前提に attention していた
   （重大な state contract 違反）。
3. partial accept 時に `seq.position = P + 1` に戻すだけで、verify 前の
   GDN recurrent / conv state を restore していなかった。正規 `SpecDecoder` は
   `spec_gdn_snapshot` → verify → partial accept → `spec_gdn_restore` →
   accepted prefix rerun を行う。
4. `run_mtp_head(position)` ベースの loop は、
   `mtp_forward_step` + `MtpKvState.logical_length` による
   「logical KV index / absolute RoPE position の分離」を表現できない。
   accept 後の MTP logical length は `before + min(accepted + 1, K)` であり、
   absolute position と常に一致するわけではない。
5. verify batch が `speculative_verify = true` と `verify_numeric_mode` を設定せず、
   正規 `SpecDecoder` の Verify role / Exact numeric path を通っていなかった。

### 8.3 修正

- `src/apps/bench/mtp.hip` phase 4 を production `SpecDecoder` API へ置き換えた。
  target prompt prefill → `mtp_kv_reset` → `spec_decoder_sync_prompt` →
  `spec_decoder_step` のみで update を進める。harness 側の独自 draft / verify /
  accept / `sequence.position` rollback は削除した。
- `--spec` は phase 4 の有効化だけとし、legacy phase 2/3 の省略は
  `--spec-only` が担うように分離した。phase 4 は phase 2 に依存しない。
- `SpecIterationOutput` に `num_mtp_drafts` / `mtp_length_before` /
  `mtp_length_after` / `rerun` / `draft_tokens` / `verify_sampled` を追加した。
- `--spec-debug` を追加。prompt sync 直後の `logical_length == context - 1`、
  draft 開始前の `logical_length` / absolute RoPE position / `sequence.position`、
  draft 後の `kv_length_before` / `kv_length_after`、commit 後の
  `logical_length == before + min(accepted + 1, num_mtp_drafts)` を assert / log する。
  absolute position と logical KV index を等しいものとして assert しない。
- `--verify-mode exact|fast`（既定 `Exact`）を追加。
- spec OFF / ON の token 列完全一致を必須条件にした。不一致時は divergence index・
  pending token・MTP `logical_length`・`sequence.position`・accepted 数・draft 列・
  verify sample 列を表示し、終了コード非 0 で終了する。
- `--chain pre|post` は legacy acceptance sim 専用とし、SpecDecoder 経路の
  次 draft hidden は常に `hidden_out`（final norm 前）であることを出力する。

### 8.4 検証（27B-PSQ、prose ctx256、TG48、device 1、`--spec --spec-only`）

| K | greedy equivalence | rounds | emitted | accepted | mean accept/update | emitted/update | reject0 | full accept | rerun | draft ms/update | verify ms/update | ms/token | tok/s | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | PASS | 31 | 49 | 18 | 0.581 | 1.581 | 41.9% | 58.1% | 41.9% | 4.47 | 51.10 | 35.16 | 28.4 | 1.02x |
| 2 | PASS | 26 | 49 | 23 | 0.885 | 1.885 | 50.0% | 38.5% | 61.5% | 8.29 | 59.22 | 35.83 | 27.9 | 1.00x |
| 4 | PASS | 24 | 49 | 25 | 1.042 | 2.042 | 50.0% | 4.2% | 95.8% | 15.88 | 73.66 | 43.86 | 22.8 | 0.82x |
| 8 | PASS | 24 | 49 | 25 | 1.042 | 2.042 | 50.0% | 0.0% | 100.0% | 31.01 | 78.53 | 53.66 | 18.6 | 0.67x |

- target forward/update は K1 1.419 / K2 1.615 / K4 1.958 / K8 2.000。
  同条件の target-only greedy は 36.0 ms/token。
- `--verify-mode fast` にすると greedy equivalence が FAIL し、
  `tg --mtp` と同一の token 列・同一の rounds / reruns / accept
  （29 / 10 / 0.655 対 29 / 10 / 0.66）を示す。
  つまり `tg --mtp` の greedy からのずれは state ではなく verify の
  numeric mode に由来する。
- required acceptance と MTP required 合成テストは変更前後で PASS。

---

## Gate 記録（historical）

以下は完了した Gate の記録である。現在の contract は
[../../developer/mtp.md](../../developer/mtp.md) を正とする。

## Gate 0: weight loading

目的は実 checkpoint の MTP 重みが名前・形状・dtype・意味・レイアウトのすべてで
正しく GPU に載ることを証明することである。forward / KV / speculative decode は
対象外である。

### 検証した checkpoint

| checkpoint | body | MTP 15 tensor |
| --- | --- | --- |
| `models/Qwen3.5-4B` | BF16 safetensors | 全 15 個 BF16 |
| `models/Qwen3.5-4B-PSQ` | psq4 / psq8 | 全 15 個 BF16（量子化除外） |

両 checkpoint で MTP 各 tensor の SHA-256 が一致し、PSQ checkpoint でも MTP は BF16 で
保持されることを実 safetensors ヘッダ / manifest で確認した（推測ではない）。

### 確認した checkpoint 事実

`H=2560`、`I=9216`、`Q=4096`（16 head × 256）、`KV=1024`（4 head × 256）、
`head_dim=256`、`vocab=248320`、body 32 層。`mtp_num_hidden_layers=1`、
`mtp_use_dedicated_embeddings=false`、`tie_word_embeddings=true`、
`attn_output_gate=true`。

| tensor | 実形状 | 意味 |
| --- | --- | --- |
| `mtp.fc.weight` | `[2560,5120]` = `[H,2H]` | `Linear(2H→H)`、PyTorch `[out,in]` |
| `mtp.layers.0.mlp.gate_proj.weight` | `[9216,2560]` = `[I,H]` | |
| `mtp.layers.0.mlp.up_proj.weight` | `[9216,2560]` = `[I,H]` | |
| `mtp.layers.0.mlp.down_proj.weight` | `[2560,9216]` = `[H,I]` | |
| `mtp.layers.0.self_attn.q_proj.weight` | `[8192,2560]` = `[2Q,H]` | gated |
| `mtp.layers.0.self_attn.k_proj.weight` | `[1024,2560]` = `[KV,H]` | |
| `mtp.layers.0.self_attn.v_proj.weight` | `[1024,2560]` = `[KV,H]` | |
| `mtp.layers.0.self_attn.o_proj.weight` | `[2560,4096]` = `[H,Q]` | |
| `mtp.pre_fc_norm_embedding/hidden.weight`、`mtp.norm.weight`、`input/post_attention_layernorm`、`q/k_norm` | `[H]` / `[head_dim]` | RMSNorm / QK-norm |

loader は転置しない（`rows=shape[0]`、`cols=shape[1]`）。MTP 層数は 1 を要求し、
必須 tensor が欠けると tensor 名付きで拒否する。

### 検証ツール / テスト

- `tools/inspect_qwen35_mtp_weights.py`（stdlib のみ、GPU 不要）。
- `tests/unit/test_qwen35_mtp_weight_load.hip`（required、synthetic で name / shape /
  dtype / layout / 必須 tensor / 層数チェック）。
- `tests/unit/test_qwen35_mtp_weight_real.hip`（実 checkpoint、GPU payload を
  logical payload と bytes 比較）。

実 checkpoint は BF16 / PSQ とも 15/15 MATCH で PASS。詳細な閾値・wiring の知見は
§7.60 を参照。

## Gate 1: single-layer draft forward

目的は MTP 1 層 draft forward（embedding → pre-fc norm → concat → fc →
full-attention 1 層 → final norm → lm_head → greedy sampling）が vLLM 参照意味と一致し、
同じ draft token を生成することである。スコープは T=1 の単独 forward（KV 永続化なし）、
BF16、TP=1、greedy、`PHASESHIFT_QWEN35_KERNEL_MODE=correctness` である。

### fixture

`tools/reference/export_qwen35_mtp_gate1.py` が独立の vLLM 参照意味を torch CPU（f32 計算・
境界で bf16 丸め）で実装し、全中間 tensor / top1 / top10 / tensor 毎 SHA-256 を
`artifacts/mtp_gate1/` に生成する。case は 4 個:

| case | token | position | 狙い |
| --- | --- | --- | --- |
| `position_0` / `position_1` / `position_127` | 925 | 0 / 1 / 127 | RoPE position の違いが `q_rope`/`k_rope` にのみ現れる |
| `token_alt` | 5678 | 0 | 別 embedding 経路 |

### 比較閾値

f32 累加順序差に由来する bf16 丸めノイズが深度に沿って増大する（最大
`mlp_down_proj` rel_l2 ≈ 5.3e-3、`logits` ≈ 3.7e-3）ことを観察した上で設定した。

- `target_hidden` / `embedding_raw`: bit-exact。
- 前半: rel_l2 ≤ 1e-2、cosine ≥ 0.9999。
- 後半・logits: rel_l2 ≤ 2e-2、cosine ≥ 0.9999。
- draft argmax: 完全一致必須（閾値対象外、緩めない）。

semantic 誤りは rel_l2 ≫ 10% を生むため、この閾値で 4〜10× の余裕をもって分離できる。
結果は 4 case 全 PASS、draft 770/770/770/883 一致、Python クロスチェック
（`tools/compare_mtp_gate1.py`）も全 case PASS。required synthetic として
`test_qwen35_mtp_lowering.hip`（lowering 構造）と
`test_qwen35_mtp_primitives.hip`（実 kernel の数式 + 負コントロール）を追加した。
wiring で発見・修正した production bug は §7.60 を参照。

## Gate 2: persistent KV and state synchronization

目的は確定済み token 列を MTP へ teacher-forced で順次入力したとき、MTP 専用の
persistent KV / position / token-hidden alignment が正しく永続化され、cache を使わず
履歴全体から再計算する参照と同一の logical K/V 履歴を保持し、chunking・block boundary・
reset・sequence 切替に依存しないことを証明することである。

### KV geometry

- `num_kv_heads = 4`、`head_dim = 256`、KV width = 1024。
- cache する K は Q/K norm + RoPE 適用後（`k_rope`、BF16）。V は `v_proj`（BF16）。
- block size（tokens per page）= 16。Gate 2 テストは `num_pages = 24`（capacity 384）。
- layout は `page * elems_per_page + offset_in_page * elems_per_token`。
  canonical `[logical_token][kv_head][head_dim]` を `dump_mtp_kv_canonical` で復元する。

### 検証構成

| 要素 | 役割 |
| --- | --- |
| `tools/reference/export_qwen35_mtp_gate2.py` | Gate 1 の参照意味を再利用し、logical KV history を保持して attention を毎 step 履歴全体から再計算 |
| `tests/unit/test_qwen35_mtp_state_real.hip` | 実 checkpoint + production `mtp_forward_step` で列を再生、golden と比較 |
| `tests/unit/test_qwen35_mtp_kv_cache.hip`（required） | KV_APPEND + PAGED_ATTENTION の append/readback・per-head layout・block boundary・overwrite・GQA |
| `tests/unit/test_qwen35_mtp_state.hip`（required） | `MtpKvState` の lifecycle・chunk equivalence・position offset・2 sequence isolation |

### 結果（257 step synthetic、実 Qwen3.5-4B）

| 検証 | 結果 |
| --- | --- |
| logical KV 永続化・per-step state | PASS |
| block boundary（B-1, B, B+1, 2B-1, 2B, 2B+1） | PASS |
| chunk equivalence（serial vs 16-token chunk、canonical K/V bit-exact） | PASS |
| state sync（catch-up 中は最終 row のみ draft） | PASS |
| reset / reuse | PASS |
| position offset（RoPE position ≠ KV index） | PASS |
| draft argmax（257 step） | 255/257（2 flip: ambiguous 2、stable 0） |
| worst rel_l2 | `k_rope` 6.2e-3、`v` 6.6e-3、`attention_core_out` 1.8e-2、`o_proj_out` 1.9e-2、`final_norm` 1.9e-2 |

### 数値判定規約

- cosine は補助指標とし、primary は relative L2・max abs・canonical state・stable argmax。
- draft argmax は state correctness 条件から外す（drafter は target ではない）。
- `margin_ref = ref_top1 - ref_top2`、`epsilon_inf = max|actual - reference|` とし、
  `margin > 2*epsilon_inf` なら stable。stable な mismatch のみ FAIL。
- 実測 flip 2 件（step 84 / 98）はいずれも ambiguous near-tie（margin ≤ 2ε）で、
  canonical K/V は serial と bit-exact。

実 target-hidden fixture（64 token real prompt）でも `MTP_GATE2_REAL: PASS`
（argmax 63/64、1 ambiguous）。誤差の数値的原因と教訓は §7.61 を参照。

## Gate 3: speculative transaction

Gate 3 は性能 Gate ではなく、speculative decoding を有効化しても target-only greedy
decode と同じ token stream を生成することを証明する Gate である。実装に先立ち state の
時系列 contract を固定した。

### state timeline 設計

長さの概念を 1 つの `sequence_length` で説明せず、以下を別 field として扱う。

| field | 意味 |
| --- | --- |
| `emitted_length` | token stream へ append 済みの token 数（pending を含む） |
| `target_computed_length` | target の可変 state へ forward 済みの token 数 |
| `mtp_computed_length` | MTP logical KV に append 済みの token 数 |
| `pending_token` | emit 済みだが target へ未 forward の token |
| `committed_length` | accept 済みで state に確定した長さ |
| `tentative_length` | transaction 中に一時的に進めた長さ |

不変条件は `target_computed_length <= emitted_length`。K=4 / accept=2 の例では
draft 直後に `mtp_computed_length = L+4`、accept 後に `L+3` となる。これは step 0 が
pending token を append しているためで、`旧 + accepted` では求まらない。各 accept count の
expected state は serial replay oracle で求める。

### target rollback / correction / bonus

target の可変 state は full-attention KV（paged）・GDN recurrent・GDN conv・
sequence length・position・block table・pending token metadata を含む。
transaction 開始時に GDN snapshot を取り、verification は serial（draft を 1 個ずつ
target へ投入）で行う。reject 後は「開始 snapshot + accepted prefix の再 forward」で
target state を作る（final state から accepted prefix を推測しない）。

correction token は emit 済みだが target state へ未 forward（pending）であり、次 iteration
の step 0 の入力になる。all-accept 時は bonus token を emit し `pending = bonus` とする。
状態機械は `IDLE → DRAFTING → VERIFYING → COMMITTING | ROLLING_BACK → IDLE`。

### vLLM alignment

`vllm/v1/spec_decode/llm_base_proposer.py` の default pathway を確認し、MTP は
`(hidden[p], token[p+1])` を position `p` で消費して `p+2` を予測すること、
次 step の hidden は final norm 前の `hidden_out` であること、accept 後の MTP logical
length が `mtp_before + min(accepted+1, K)` であることを確定した。詳細は §7.62。

### 実装した production core

- `spec_decode.h` / `.cpp`: `SpecVerifyResult` / `spec_greedy_accept` / `SpecPhase` 状態機械 /
  `SpecTransaction`（begin / begin_verify / commit / rollback / abort）/
  `spec_gdn_snapshot` / `spec_gdn_restore` / `mtp_generate_drafts`。
- `spec_decoder.h` / `.cpp`: `SpecDecoder` が target `Executor` + `MtpExecutor` +
  `MtpKvState` + `PagedSequenceState` + `GdnStatePool` を orchestrate する。
  `spec_decoder_sync_prompt` が prompt を teacher-force して MTP KV を作り、
  `spec_decoder_step` が K draft → target verify → longest-prefix accept →
  all-accept は commit、partial は GDN を snapshot へ戻して accepted prefix を
  再 forward してから commit、を行う。capacity clamp と EOS 停止を持つ。

### 結果（実 Qwen3.5-4B、greedy）

| prompt | K | off/on | match | reject events |
| --- | --- | --- | --- | --- |
| 32 | 0 / 1 / 2 / 4 / 8 | 32 / 32 | 完全一致 | 0 / 1 / 1 / 8 / 9 |
| 128 | 0 / 1 / 2 / 4 / 8 | 24 / 24 | 完全一致 | 0 / 9 / 12 / 13 / 13 |

K=0 は sync-only の単一 decode path。bonus token は有効。acceptance statistics は
PASS 条件ではない。

### Gate 3 residual closure

`test_qwen35_mtp_gate3_replay` で target canonical KV serial replay と
GDN recurrent + conv state serial replay が bit-exact、third real prompt の ON/OFF が
token exact、real prompt EOS truncation が PASS であることを確認した。

### Gate 4/5

Gate 4A.1 / 4B / 4C / 4E は VERIFY_EXACT による target token equivalence、
Gate 5A-5D は draft cost profile と dynamic draft depth の研究であり、記録は
[optimization_history.md](optimization_history.md) §7.64-7.77 を正とする
（本ファイルには重複記載しない）。

verify-Exact 経路のみの変更であり、decode / prefill の通常経路は不変。

---

## 9. MTP first-position acceptance の原因調査

prose corpus、GPU 1、`Qwen3.8-27B-PSQ`、`--spec --spec-only`、K=1。条件が揃っていない
数値を比較しないこと。

### 9.1 target hidden の semantic A/B

target graph は `layer_input → OUTPUT_GATHER → selected_hidden → final RMSNorm → logits`
であり、external output の `selected_hidden` / `layer_input` は final RMSNorm **前**である。
MTP の入力にどちらを渡すかを A/B した。

| 条件 | PRE（現行） | POST（final RMSNorm 後） |
| --- | ---: | ---: |
| ctx256 / TG256 | 0.6623 (102/154) | 0.6842 (104/152) |
| ctx32 / TG150 | 0.7647 (65/85) | 0.7442 (64/86) |

差は ±2pt で ctx により符号が反転する。**target hidden の PRE / POST は原因ではない。**

### 9.2 vLLM semantic reference との数値 A/B

`tools/reference/export_qwen35_mtp_gate1.py` は vLLM Qwen3.5 MTP semantics を実装した
reference である。同 exporter の `attention_isolated` は KV 履歴のない T=1 forward なので、
`--context 1`（MTP KV が空）で同一 `(hidden, token, position=0)` を与えて比較した。31 サンプル。

| 項目 | 結果 |
| --- | ---: |
| cos(reference `final_residual`, PhaseShift `hidden_out`) | 0.999949（min 0.999825） |
| cos(reference `final_norm`, PhaseShift `hidden_out_normed`) | 0.999937（min 0.999822） |
| draft == reference top1 | 30/31 |
| P(draft == target) | 11/31 |
| P(reference top1 == target) | 11/31 |

MTP 層の forward は reference と一致し、MTP 側の PSQ8 `embed_tokens` / `lm_head` も
target 一致率を変えていない。**MTP 実装に原因はない。**

参考事実:

- `mtp.*` 15 tensor は `Qwen3.8-27B` と `Qwen3.8-27B-PSQ` で bit-identical（MTP 層は非量子化）
- `mtp.pre_fc_norm_hidden` / `pre_fc_norm_embedding` の weight 平均は負であり、
  Gemma 形 `(1+w)` でなければ負倍率になる。ONE_PLUS は正しい
- fc concat は `[embedding_norm, hidden_norm]`（`test_qwen35_mtp_lowering` / `_primitives`）
- `tie_word_embeddings: False` であり lm_head は独立 tensor

### 9.3 draft chain の A/B

`mtp_generate_drafts()` の step>=2 の入力 hidden を `hidden_out`（現行）と
`hidden_out_normed` で比較した。per-position は同一 run の accepted histogram から算出する。

| target hidden | chain | P(d1) | P(d2 \| d1) |
| --- | --- | ---: | ---: |
| PRE | `hidden_out` | 0.6612 | 0.6875 |
| PRE | `hidden_out_normed` | 0.6423 | 0.6835 |
| POST | `hidden_out` | 0.6557 | 0.6750 |
| POST | `hidden_out_normed` | 0.6557 | 0.6750 |

K=3（POST）は chain pre `0.6729 / 0.6806 / 0.5714`、chain post `0.6667 / 0.6806 / 0.5510`。
**`hidden_out_normed` は劣化するため採用しない。** vLLM と同じく `hidden_out` を維持する。

### 9.4 判定

first-position acceptance は ctx256 で 0.66、ctx32 で 0.76。同一条件の DFlash2 は 0.841
（`docs/rnd/dflash2/dflash2.md`）。MTP 実装は reference と一致しているため、差は
**BF16 checkpoint 用に訓練された MTP が PSQ 化された target を予測している**ことによると
判断する。残る 0.84 から 0.94-0.97 までの差は比較基準（checkpoint・量子化度・acceptance
の定義）の違いによる。

acceptance 統計は accepted histogram とそこから算出する conditional で記録する。
K の異なる run を引き算して per-position を推定しない。

---

## 10. LmHeadCandidateProxy と greedy equivalence

### 10.1 症状

`--tokens-offset 345 --context 1 --steps 4` で `greedy equivalence: FAIL`
（`emitted 1245` / `expected 4220`）となった。`PHASESHIFT_TARGET_LM_HEAD_PROXY=0` で
PASS することから `LmHeadCandidateProxy` が原因である。

proxy は INT2 coarse の上位を候補にして PSQ8 で rerank する**近似**であり、真の argmax が
候補外だと取りこぼす。`test_target_lm_proxy_error_bound` の保証は per-logit の
Cauchy-Schwarz 誤差境界であって argmax 保存ではない。

### 10.2 argmax 証明は不採用

`upper_v = coarse_v + activation_l2 * error_l2` を exact logit の厳密な上界として全語彙に
作り、上位 `pool+1` を取り、先頭 `pool` を候補に exact rerank、
`best > upper[pool+1]` のときだけ採用して外れたら exact 経路へ fallback する証明を
実装した（`test_target_lm_proxy_fallback` と同型）。greedy equivalence は回復したが、
計測で次の3点が判明し**採用しなかった**。

**(1) 証明は実データで一度も通らない**

`PHASESHIFT_QWEN35_KERNEL_TRACE` の `sampling` 最適化カウンタは、
`phaseshift-compute`（max-new-tokens 128）で `PROXY=0` と `PROXY=1 CERTIFY=1` の
**両方とも 43**。proxy が `SAMPLING` dispatch を消費していない＝`NotApplicable`＝
証明失敗で常に exact 経路へ落ちていることを意味する。
`test_target_lm_proxy_fallback` が `certified > 0` を満たすのは合成入力だけで、
実 lm_head では満たさない。

理由は上界の桁。`activation_l2` は final norm 後の hidden のノルムで約 107、
INT2（4値）近似の行誤差 `error_l2` を掛けると上界は十〜数百になる。
対して実際の logit margin は 0〜20。**Cauchy-Schwarz は INT2 の誤差に対して広すぎる。**

**(2) 証明失敗時も proxy の kernel は走る**

GPU busy は exact 1901.4 ms に対し証明ありで 1958.0 ms（+3.0%）。証明が通らず
discard されるためこれは丸ごと無駄になる。さらに forward ごとの D2H（42 回 × 約 23 ms）で
wall が 63.55 → 43.99 tok/s（−31%）まで落ちる。

証明用 kernel 自体は軽い（`activation_l2` 0.051 / `target_upper_logits` 0.022 /
`target_certified_argmax` 0.002 / `compact` 0.002 ms で計 0.077 ms/call）。
遅いのは kernel ではなく **D2H sync**。

**(3) 候補選択を `coarse` から `upper` に変えると品質が落ちる**

上界 `upper` は行ごとに `error_l2` が加算されるため、`upper` の上位に選ばれるのは
「error が大きい行」になり候補として劣化する。証明をスキップして proxy の答を
採用した場合の実測は `DFLASH2_ROUNDS/ACCEPTED` が `42/85` から **`68/59`** に崩れた。

### 10.3 採否

- `LmHeadCandidateProxy` は元の pipeline（候補を `coarse` で選ぶ）に戻す
- `PHASESHIFT_TARGET_LM_HEAD_PROXY` の既定を `1` → **`0`** に変更（exact 経路）
- `PHASESHIFT_TARGET_LM_HEAD_PROXY_CERTIFY` は削除（証明しないため）

device 側で証明を完結させても (1) と (2) のため回復しない。証明が通るようになることが
前提で、そのためには INT2 より狭い誤差境界が要る
（候補を `coarse` で選び、候補外を追加で exact rerank する二段方式が候補）。

### 10.4 既定 OFF の代償（DFlash2 spec decode）

`LmHeadCandidateProxy` は `batch.speculative_verify` の `ExecutionRole::Verify` で
効くため、影響は DFlash2 の verify に出る。

`phaseshift-compute --dflash2-drafts 7 --max-new-tokens 256`（GPU 1、3 rep）:

| 条件 | DECODE_TOKENS_PER_SEC | DFLASH2_VERIFY_GPU_MS |
| --- | ---: | ---: |
| proxy 既定 ON（`2b90b202` の元実装 / lossy） | **65.20** | 3355.65 |
| proxy 既定 OFF（exact・**採用**） | **63.55** | 3456.34 |

帰属の切り分け:

- **model 差ではない** — 同一バイナリで旧 model（MTP 全 BF16）を当てても 63.60。
  MTP は DFlash2 経路で使われない。
- **binary 差 = 既定変更** — `2b90b202` の proxy 実装を差し戻すと 65.20
  （3 rep: 65.20 / 65.22 / 65.16）が再現し、同じく `PROXY=0` は 63.64。
  `PHASESHIFT_TARGET_LM_HEAD_PROXY` の既定 `1` → `0` そのもの。

正しさ契約は完全に一致: `GREEDY_TOKEN_SUM=2446188`、
`GENERATED_IDS` sha1 `47aebe55d048`、`DFLASH2_ROUNDS=84`、
`DFLASH2_ACCEPTED_DRAFTS=171`。

ただしこの workload で proxy が生成結果を変えていないにすぎない。
greedy equivalence を破る実例（10.1）が別に存在するため、
**−2.38% は argmax を保証する代償**として受け入れた。

---

## 11. MTP quantumization の A/B

MTP（425M params、BF16 で 0.79 GiB）を量子化して draft を速くする検討。
`--preset psq` の profile（`src/phaseshift/quantization/fpx/profile.cpp` の `psq()`）を
変更し、`phaseshift-quantizer quantize --preset psq --backend hip --scope text-only`
で再生成して比較した。比較対象は `models/Qwen3.8-27B-PSQ`（MTP 全 BF16）と生成物。

非 MTP tensor は **1253 本すべてビット一致**であり、差分は MTP のみ。

### 11.1 encoding の決定

| `TensorRole` | 実体 | params | 初期案 | 採用 |
| --- | --- | ---: | --- | --- |
| `MtpOutput` | `mtp.fc.weight` (5120, 10240) | 52M | PSQ8 | **PSQ4** |
| `MtpAttention` | `self_attn.{q,k,v,o}_proj` | 68M | PSQ4 | PSQ4 |
| `MtpFfn` | `mlp.{gate,up,down}_proj` | 267M | PSQ4 | PSQ4 |
| `MtpNorm` | norm 7本 | 1M | PSQ4 | **BF16**（`load_quantized_small` が 1 次元の量子化を拒否） |
| `Output` | `lm_head.weight` | 1271M | PSQ8 | PSQ8（従来どおり） |

`test_dflash2_quantization_adapter` に新契約の assert を 4 件追加（74/74 PASS）。

### 11.2 Round 1: `mtp.fc` = PSQ8 — 否決

| 条件 | acceptance base / new | draft ms/update base / new |
| --- | --- | --- |
| ctx32 K1 | 0.765 / 0.744 | 4.456 / **5.794** |
| ctx32 K2 | 1.141 / 1.141 | 8.162 / **10.949** |
| ctx256 K1 | 0.662 / 0.673 | 4.474 / **5.962** |
| ctx256 K2 | 1.116 / 1.065 | 8.292 / **11.076** |

draft が **+30〜34% 遅化**。`PHASESHIFT_QWEN35_KERNEL_TRACE=1` の
`KERNEL_TRACE_FALLBACK LINEAR_PSQ8`（base は fallback 0）で原因を特定した。
`linear_selector.cpp` の `kLinearShapeRules` には
`{ 5120, 10240, kFamilyBf16 | kFamilyPsq4 }` があり **PSQ8 が未登録**。
`select_psq8_gemm_config` が `nullopt` を返し `SelectorRejected` で
`launch_model_dispatch_correctness`（参照実装）へ退避していた。
rocprofv3 では `model_dispatch_kernel` 89 dispatch / 207 ms（2.33 ms/draft）。

### 11.3 Round 2: MTP linear 8本すべて PSQ4 — 採用

fallback は 0、`greedy equivalence` は全条件 PASS。

| 条件 | acceptance base / new | draft ms/update base / new |
| --- | --- | --- |
| ctx32 K1 | 0.765 / 0.786 | 4.453 / **3.552**（−20%） |
| ctx32 K2 | 1.141 / 1.141 | 8.252 / **6.442**（−22%） |
| ctx256 K1 | 0.662 / 0.652 | 4.471 / **3.570**（−20%） |
| ctx256 K2 | 1.116 / 1.151 | 8.281 / **6.484**（−22%） |

acceptance の差は 4 条件すべて標本誤差内
（n = 85〜154 rounds、`mean accepted/update` の標準誤差は約 ±4〜5pt）。

- draft 速度: **−20〜22%**
- メモリ: MTP linear 387M が 774 MiB → 約 218 MiB（約 556 MiB 削減）
- required acceptance: **124/124 PASS**

### 11.4 計測条件

`phaseshift-bench mtp --device 1 --arena-gib 24 --draft-k {1,2} --spec --spec-only`、
corpus は `tests/data/mtp_perf/prose.psktok`、`--verify-mode` 既定（Exact）。

