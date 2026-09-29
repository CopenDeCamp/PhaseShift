# MTP（Multi-Token Prediction）

MTP は Qwen3.5 系 checkpoint に内蔵された 1 層の drafter である。target の hidden と
直近 token から次 token を自己回帰的に予測し、greedy speculative decode の draft 列を
生成する。

本書は現在実装されている contract のみを記述する。研究の経緯、採否判断、性能値は
[R&D 記録](../rnd/mtp/mtp.md) と
[MTP 最適化履歴](../rnd/mtp/optimization_history.md) を参照。

---

## Scope / current status

**MTP は production serve path には接続されていない。**

- `create_spec_decoder` / `spec_decoder_step`（MTP 用 `SpecDecoder`）の呼び出し元は
  tests のみである。`phaseshift-compute` / `phaseshift-cli` / `phaseshift-server` に
  MTP の入口は無い。
- CLI から起動できる speculative decode は **DFlash2** である。
  `phaseshift-compute` の `--dflash2-model-dir` が指定されたときだけ
  `run_dflash2_shot` が DFlash2 経路に入り、`phaseshift-cli` はこの option を転送する。
  target-only 経路は `ContinuousBatcher` のまま変わらない。
- `phaseshift-bench mtp`（draft 単体）と `phaseshift-bench tg --mtp`（accept 込み）は
  `MtpExecutor` を計測のために直接使う開発用 harness であり、serve path ではない。
- 一方、`spec_decode.h` の共有 helper（`spec_greedy_accept` /
  `spec_gdn_snapshot` / `spec_gdn_restore`）は DFlash2 production path から使われる。
  これらは MTP と DFlash2 で共通の contract である。

対象は Qwen3.5 系 runtime family（`src/phaseshift/models/qwen35/`）である。MTP 層は
1 層、target は BF16、draft / verify は greedy を前提とする。

---

## Weight contract

MTP 重みは target model の一部（`Qwen35ModelWeights::mtp`）として通常の weight loader で
読み込む。tensor 名と形状は loader が検証し、契約違反は拒否する。

### config

target の `Qwen35TextConfig` から次を用いる。

| 記号 | field | 意味 |
| --- | --- | --- |
| `H` | `hidden_size` | hidden size |
| `I` | `intermediate_size` | MLP intermediate size |
| `Q` | `num_attention_heads * attention_head_dim` | attention query features |
| `KV` | `num_key_value_heads * attention_head_dim` | attention KV features |
| `D` | `attention_head_dim` | head_dim |
| `eps` | `rms_norm_eps` | RMSNorm eps |
| — | `rope_theta` | RoPE theta |
| — | `partial_rotary_factor` | rotary 比率（本系は 0.25） |

### tensor

必須 tensor は次の 15 個である。

| tensor 名 | 形状 | 意味 |
| --- | --- | --- |
| `mtp.fc.weight` | `[H, 2H]` | `Linear(2H → H)` |
| `mtp.layers.0.mlp.gate_proj.weight` | `[I, H]` | MLP gate |
| `mtp.layers.0.mlp.up_proj.weight` | `[I, H]` | MLP up |
| `mtp.layers.0.mlp.down_proj.weight` | `[H, I]` | MLP down |
| `mtp.layers.0.self_attn.q_proj.weight` | `[2Q, H]` | gated q projection（`attn_output_gate`） |
| `mtp.layers.0.self_attn.k_proj.weight` | `[KV, H]` | k projection |
| `mtp.layers.0.self_attn.v_proj.weight` | `[KV, H]` | v projection |
| `mtp.layers.0.self_attn.o_proj.weight` | `[H, Q]` | o projection |
| `mtp.pre_fc_norm_embedding.weight` | `[H]` | RMSNorm |
| `mtp.pre_fc_norm_hidden.weight` | `[H]` | RMSNorm |
| `mtp.layers.0.input_layernorm.weight` | `[H]` | RMSNorm |
| `mtp.layers.0.post_attention_layernorm.weight` | `[H]` | RMSNorm |
| `mtp.norm.weight` | `[H]` | final RMSNorm |
| `mtp.layers.0.self_attn.q_norm.weight` | `[D]` | per-head Q norm |
| `mtp.layers.0.self_attn.k_norm.weight` | `[D]` | per-head K norm |

### loader 契約

- `MatrixWeight.rows = shape[0] = output_features`、`cols = shape[1] = input_features`。
  loader は転置しない。PyTorch の `[out, in]` と一致する。
- MTP 層数は **1** を要求する。checkpoint の MTP 層数が 1 以外なら読み込みを拒否する。
- 必須 tensor が 1 つでも欠けると tensor 名付きで拒否する。
- dtype / encoding は checkpoint の manifest に従う。現在の checkpoint は BF16 と
  PSQ の双方で MTP 15 tensor を BF16 で保持する（MTP は量子化対象外）。
- `lm_head` は `tie_word_embeddings=true` のとき `embed_tokens` を使う
  （target と同一 weight）。MTP 専用の embedding / lm_head は持たない。

---

## Lowering

`lower_qwen35_mtp_to_primitives(config, weights)` が MTP forward の `PrimitiveGraph` を
生成する。先頭で `validate_mtp_geometry` が上記形状契約を検証する。emit 順序は次の
とおりで、順序を変えない。

1. `EMBEDDING_LOOKUP`: `tokens` → embedding（tied `embed_tokens`）。
2. `RMS_NORM`（`pre_fc_norm_embedding`）を embedding に適用する。
3. `RMS_NORM`（`pre_fc_norm_hidden`）を入力 hidden に適用する。
4. `CONCAT`: `[embedding_norm, hidden_norm]` を feature 軸で連結し `[2H]` にする
   （順序は embedding が前半）。
5. `LINEAR`: `mtp.fc` で `2H → H`。
6. `RMS_NORM`（`input_layernorm`）。
7. `LINEAR`: `q_proj` / `k_proj` / `v_proj`。
8. `SPLIT`（`INTERLEAVED_HEADS`）: `q_proj` 出力 `[2Q]` を head ごとに前半 q / 後半 gate に
   分ける（head `h` の境界は `h*D`）。global の前半後半ではない。
9. `RMS_NORM`（`q_norm` / `k_norm`）: per-head。`group_size = D`、weight `[D]`、出力 f32。
10. `ROPE`: q と k に適用する。NeoX half-rotate、`rotary_dim = D * partial_rotary_factor`。
11. `KV_APPEND`: `k_rope` と `v_proj` を MTP KV state へ append する。
12. `PAGED_ATTENTION`: MTP KV state を参照する。scale は `1/sqrt(D)`。
13. `SIGMOID`: gate を活性化する。
14. `MUL`: attention core × `sigmoid(gate)` を o_proj の手前で掛ける。
15. `LINEAR`: `o_proj`。
16. `RESIDUAL_ADD`: `o_proj` 出力 + `mtp.fc` 出力。
17. `RMS_NORM`（`post_attention_layernorm`）。
18. `LINEAR`: `mlp.gate_proj` / `mlp.up_proj`。
19. `SWIGLU`: `silu(gate) * up`。
20. `LINEAR`: `mlp.down_proj`。
21. `RESIDUAL_ADD`: `down` + 手前の残差。
22. `OUTPUT_GATHER`: output row を選択する。
23. `RMS_NORM`（`mtp.norm`）。
24. `LINEAR`: lm_head（`tie_word_embeddings=true` なら tied `embed_tokens`）。出力は f32。
25. `SAMPLING`: greedy。token を 1 個出す。

すべての RMSNorm は Gemma 形式 `x * rsqrt(mean(x^2) + eps) * (1 + w)`（`ONE_PLUS`）で
計算する。Q/K norm だけが per-head（`group_size = D`）である。

external output は 4 個で、順に:

- `hidden_out`: output row の選択結果（final norm **前**、BF16）。次 draft step の hidden。
- `logits`: f32。
- `sampled`: i32。
- `hidden_out_normed`: final norm 後（BF16）。

---

## MTP state

`MtpKvState`（`mtp_kv_state.h`）が MTP 専用の persistent state を所有する。target の
KV / sequence state とは別 allocator・別 block table である。

### 生成と所有

- `create_mtp_kv_state(arena, config, sequence_id, stream)`:
  - `PagedKVPool`（`num_pages × page_tokens × num_attention_layers=1 × kv_heads × head_dim`）と
    `SequenceSlotPool`（1 sequence）を arena から確保する。
  - 全 page を allocate し、`block_to_page[b]` と block table を create 時に確定する。
  - `capacity_tokens = num_pages * page_tokens`、`logical_length = 0`、`step_index = 0`。
- `MtpExecutor` が `MtpKvState` を所有する。`create_mtp_executor` が生成し、
  `mtp_executor_shutdown` が `mtp_kv_shutdown` を呼ぶ。
- `mtp_kv_shutdown` は handle を release し、block table と長さを 0 に戻す。

### geometry

`MtpKvConfig` は `num_pages`、`page_tokens`、`num_attention_layers = 1`、
`kv_heads = num_key_value_heads`、`head_dim = attention_head_dim`、`dtype` を持つ。
現在の correctness path は **BF16 のみ**で、他 dtype は create 時に拒否する。

### 長さと slot

- `logical_length` は MTP KV に append 済みの token 数（= logical KV index）である。
- `step_index` は `mtp_kv_reserve` の呼び出し回数である。
- `mtp_kv_reserve(rows)` は `logical_length += rows`、`step_index += 1` を行う。
  capacity 超過は `out_of_range`。
- `mtp_kv_reset` は `logical_length` と `step_index` を 0 にするだけで、GPU buffer を
  消去しない。`logical_length` 外の stale entry は読まない。
- `mtp_kv_physical_slot(logical)` は
  `block = logical / page_tokens`、`off = logical % page_tokens` として
  `block_to_page[block] * page_tokens + off` を返す。

### KV の内容

append する K は `k_proj → k_norm（per-head）→ RoPE` の結果（`k_rope`）である。
V は `v_proj` の結果である。raw K は保持しない。

### view と dump

- `mtp_kv_state_view(state)` が attention / KV append へ渡す
  `ModelDispatchStateView` を返す。
- `mtp_kv_dump_canonical` は BF16 のまま logical 順
  `[logical_token][kv_head][head_dim]` へ復元する。test / debug 用である。

---

## Draft forward

`mtp_forward_step(executor, state, hidden, tokens, rows, absolute_position_start, stream, trace, state_trace)`
が 1 回の forward を行う。

- `rows` は `[1, max_rows]`。`max_rows` は `create_mtp_executor` で 16 に clamp する。
- logical KV index の base は `state.logical_length` である。append invariant
  `logical_length + rows <= capacity_tokens` を検証する。
- RoPE position は `absolute_position_start + t`（`t = 0..rows-1`）であり、
  logical KV index とは独立である。
- attention の可視範囲は `[logical_length + 1, logical_length + rows]`。
- execution class は `rows == 1` なら `DECODE`、`rows > 1` なら `PREFILL`。
- sampling は greedy、`compute_logits = true`、output row は 1（最終 row）。
- 成功時は `mtp_kv_reserve(rows)` で `logical_length` を進める。
- 返す `MtpStepOutcome` は `{draft_token, kv_length_before, kv_length_after, absolute_position, top1_logit, top2_logit, top2_token}`。

`run_mtp_rows(executor, hidden, tokens, rows, position_start, stream, trace, kv_position_start)`
は同じ forward を executor 所有の default KV state に対して実行する。`kv_position_start`
は既定で `position_start` であり、teacher-forced の prompt sync など
logical index と RoPE position を分離したいときに上書きする。

top-1 / top-2 logit は `executor.collect_top2` が true のときだけ
`launch_argmax_f32_top2` で計算する。それ以外は `top1_logit = top2_logit = 0`。

staging は staging pool と `program_staging_meta` を必要とする（stateful graph では
`launch_host_binding` が device の `refs_dev->states` を必要とするため）。
`MtpExecutor` は常にこれらを確保する。

---

## Multi-step drafting

`mtp_generate_drafts(executor, state, first_hidden, first_token, num_drafts, first_absolute_position, stream, trace, policy)`
が draft 列を生成する。

- 引数契約: state initialized、`first_hidden != nullptr`、`1 <= num_drafts <= max_rows`、
  `logical_length + num_drafts <= capacity_tokens`。
- step `k` は `mtp_forward_step(rows=1, absolute_position=first_absolute_position + k)` を
  呼ぶ。
- 入力 `(hidden, token)` は step 0 が `(first_hidden, first_token)`、step 1 以降は
  前 step の `executor.hidden_out` と draft token である。
- 各 step を `SpecDraftStep {input_token, absolute_position, kv_length_before,
  kv_length_after, draft_token, physical_slot, margin}` として記録する。
  `margin = top1_logit - top2_logit`。
- `policy` が指定されると:
  - `enable_discard`: step 0 の margin が `discard_margin` 未満なら draft を破棄して
    空列を返す。
  - `dynamic`: `min_drafts` 本を生成した後、margin が `stop_margin` 未満になった時点で
    打ち切る。
  - いずれかの flag が立つとき `executor.collect_top2` を有効にする。
- 戻り値 `SpecDraftSet` は `{steps, mtp_length_before, mtp_length_after}`。
  成功時は生成 step 数だけ MTP の `logical_length` が進んでいる。

`SpecDecoder` は `first_absolute_position = sequence->position - 1` を渡す。これは
MTP が `(hidden[p], token[p+1])` を position `p` で消費する alignment に対応する。

---

## Speculative transaction

### 状態機械

`SpecPhase` は `IDLE → DRAFTING → VERIFYING → COMMITTING | ROLLING_BACK → IDLE`
である。不正遷移（nested begin、begin 前の verify、二重 commit、IDLE での rollback）は
error にする。

snapshot:

- `SpecMtpSnapshot {logical_length, step_index}`（MTP 側）
- `SpecTargetSnapshot {position, block_count}`（target 側）

### API

- `spec_transaction_begin(txn, mtp, target)`: `IDLE` から `DRAFTING` へ。
  MTP と target の snapshot を取る。
- `spec_transaction_begin_verify(txn)`: `DRAFTING` → `VERIFYING`。
- `spec_transaction_commit(txn, result, mtp, target, stream)`: `VERIFYING` のみ。
  `result.num_committed_target_tokens == accepted + 1` を要求する。
  MTP は `mtp_kept = min(accepted + 1, mtp_drafts)`（`mtp_drafts > 0` のとき）だけ進め、
  `logical_length` / `step_index` を `before + mtp_kept` にする。
  target は `required_blocks = ceil(new_position / page_tokens)` 本へ余剰 page を返し、
  `position = before.position + num_committed_target_tokens` にする。
- `spec_transaction_rollback` / `spec_transaction_abort(txn, mtp, target, stream)`:
  MTP の `logical_length` / `step_index`、target の block table と `position` を
  snapshot へ戻す。

`mtp_kept` が `accepted + 1` で cap されるのは、全 accept のとき最後の draft が MTP に
未消費のためである。`+accepted` を仮定してはならない。

### accept 判定

`spec_greedy_accept(drafts, num_drafts, target_tokens, bonus_enabled, result)`:

- longest-prefix rule。`accepted` は `drafts[i] == target_tokens[i]` が最初に崩れる index。
- 全 accept: `first_reject_index = 0xFFFFFFFF`。`bonus_enabled` なら
  `bonus_token = target_tokens[num_drafts]`。
- 途中 reject: `correction_token = target_tokens[accepted]`。
- `emitted_tokens = drafts[0..accepted) + (correction または bonus)`。
- `num_committed_target_tokens = accepted + 1`（reject 位置の correction token を
  target が生成済みのため）。
- `num_drafts == 0`（sync-only）は emitted 0、committed 0 を返す。

correction / bonus token は emit 済みだが target state へ未 forward の pending token で
あり、次 iteration の step 0 の入力になる。correction token の KV / GDN state を
ここで commit しない。

### target 側 mutable state の rollback

target の可変 state は次を含む。

- full-attention K/V cache（paged、block table で trim する）
- GDN recurrent state（`GdnStatePool`）
- GDN conv state
- sequence length / position / block table

MTP `SpecDecoder` は correctness-first に、transaction 開始時に GDN の
conv / recurrent を `spec_gdn_snapshot`（D2D copy）し、partial accept では
`spec_gdn_restore` で snapshot へ戻してから accepted prefix を再 forward する。
「final state から accepted prefix state を推測」しない。

### SpecDecoder の step

`spec_decoder_step(decoder, pending_token)`:

1. draft 深さ `K = min(config.num_drafts, target_room - 1, mtp_room)` に clamp する。
   `K == 0` なら transaction を開かず `run_single_decode` で 1 token を生成する。
2. transaction begin、GDN snapshot、`mtp_generate_drafts`。
   draft が 0 本なら rollback して single decode に落とす。
3. optional の n-gram tail（`ngram_n > 0`）を draft 列へ append する。
4. target verify: `[pending, draft_0..draft_{K-1}]` の `actual_k + 1` 行を
   `speculative_verify = true`、`verify_numeric_mode = config.verify_numeric_mode`、
   `num_output_rows = actual_k + 1`、`prefix_tokens = sequence.position` で実行する。
5. `spec_transaction_begin_verify` の後、`spec_greedy_accept` で accept を決める。
6. 全 accept: verify の `final_hidden[actual_k]` を `pending_hidden` にして commit する。
7. partial accept: GDN を restore し `sequence.position` を戻し、`[pending, accepted
   drafts]` を再 forward して、その `final_hidden` を `pending_hidden` にして commit する。
8. emitted token と pending token を返す。EOS（`eos_token >= 0`）は emitted を最初の EOS
   まで truncate し `finished = true`。pending が EOS でも `finished = true`。

`spec_decoder_sync_prompt(decoder, prompt_tokens, prompt_hidden, count)` は prompt を
teacher-force して MTP KV を作る。prompt 長 `N` に対し `mtp_forward_step` を
`max_rows` ずつ chunk しながら、`(hidden[p], token[p+1])` を position `p`（`p = 0..N-2`）で
消費し、logical length `N-1` を作る。`pending_hidden = prompt_hidden[N-1]`、
token history を prompt で初期化する。

---

## Numeric contract

採用している numeric mode は `VerifyNumericMode::Fast` と `VerifyNumericMode::Exact` の
2 つである。`ExecutionRole`（`Decode` / `Prefill` / `Verify`）と組み合わせ、
`verify_exact_active(ctx) = (role == Verify && numeric_mode == Exact)` のときだけ exact
kernel へ切り替える。

- **draft**: `ExecutionRole::Decode` / `Fast`。drafter は target ではないため、draft の
  差は acceptance を変えるが target verification が正しければ最終出力を変えない。
- **decode / prefill**: `Fast`。batch-invariant な decode 経路（M=1 は exact-rows）を使う。
- **verify**:
  - `Fast`（既定）: 既存の fast path。
  - `Exact`: bf16 linear は `1 <= rows <= 64`（`kBf16GemmExactRowsMax`）で
    `Bf16GemmConfig{ExactRows, rows}` を選び、`gemm_bf16_exact_rows_kernel` で row ごとの
    K 還元順序を M=1 と一致させる。`launch_gemm_bf16_exact_rows()` は
    `kBf16GemmExactRowsKernelMax = 16` 行ずつ chunk launch し、chunk 内でも M=1 と
    bit-exact な還元を保つ。
    GDN recurrence の演算モードは `gdn_recurrence_lossy_enabled()` が決める（既定 lossy、
    `PHASESHIFT_GDN_RECURRENCE_EXACT=1` で exact）。verify-exact の lossy 時は geometry に
    応じて decode1 系（`decode1_serial`、条件が合えば multi-row 版）、lossy off 時は
    `wmma_serial` を選ぶ。どちらの serial path も GDN spec history を capture する。

`SpecDecoderConfig::verify_numeric_mode` の既定は `Fast` である。`create_spec_decoder`
は環境変数 `PHASESHIFT_VERIFY_EXACT=1` が設定されているときだけ `Exact` へ上書きする。
verify role の判定は `SpecDecoder` が `ScheduledBatch::speculative_verify = true` を
設定することで行われる。

（DFlash2 の `DFlash2SpecDecoderConfig::verify_numeric_mode` は既定 `Exact` である。
これは DFlash2 側の契約であり、MTP の既定とは異なる。）

---

## Runtime configuration

`MtpExecutorConfig`:

| field | 既定 | 意味 |
| --- | --- | --- |
| `num_pages` | 4 | MTP KV の page 数 |
| `page_tokens` | 16 | 1 page の token 数 |
| `max_rows` | 16 | 1 forward の最大 row 数（16 に clamp） |

MTP KV 容量は `num_pages * page_tokens`（既定 64 token）である。

`SpecDecoderConfig`:

| field | 既定 | 意味 |
| --- | --- | --- |
| `num_drafts` | 4 | 1 iteration の最大 draft 数 |
| `bonus_token_enabled` | true | all-accept 時に bonus token を emit する |
| `eos_token` | -1 | EOS（負値で無効） |
| `verify_numeric_mode` | `Fast` | verify の numeric mode |
| `draft_policy` | 無効 | dynamic / discard policy |
| `ngram_n` | 0 | n-gram tail の n（0 で無効） |
| `ngram_max_tail` | 0 | n-gram tail の最大長 |
| `ngram_window` | 2048 | n-gram 探索窓 |

`MtpDraftPolicy`: `dynamic`（既定 false）、`stop_margin`、`min_drafts`（既定 1）、
`enable_discard`（既定 false）、`discard_margin`。

環境変数:

- `PHASESHIFT_VERIFY_EXACT=1`: `create_spec_decoder` で verify を `Exact` にする。
  既定は off。

production CLI / server には MTP の起動 option が無い（serve path は DFlash2）。

---

## Known limitations

- MTP は production serve path に未接続である。利用可能な入口は tests と
  `phaseshift-bench mtp` / `phaseshift-bench tg --mtp` のみ。
- 対応する MTP 層数は **1**。checkpoint の層数が 1 以外なら load を拒否する。
- MTP KV は **BF16 のみ**。create 時に他 dtype を拒否する。
- `max_rows` は 16 に clamp される。`MtpExecutorConfig` に 16 超を指定しても 16 になる。
- MTP state は **1 sequence** のみ（`SequenceSlotPool` を 1 で作る）。
- MTP executor は staging pool と program metadata を前提とする。これらを持たない
  非 pool の staging path では stateful graph が壊れるため使えない。
- `mtp_kv_reset` は GPU buffer を消去しない。`logical_length` を超える stale entry を
  読まないことが前提である。
- draft / verify は greedy のみ。stochastic speculative decoding は実装していない。
- `run_mtp_rows_on_state` は 1 step ごとに host で status / sampled / top2 を読む。
  これは draft 列を自己回帰的に生成するためのデータ依存であり、1 step 内の並列化は
  行わない。

---

## Source map

### weights / lowering

- `src/phaseshift/models/qwen35/weights/model_weights.cpp`
  - `load_mtp_bf16` / `load_mtp_quantized`
  - `check_mtp_layer_count` / `check_mtp_required_tensors` / `mtp_required_tensors`
- `src/phaseshift/models/qwen35/model/lower_to_primitives.cpp`
  - `lower_qwen35_mtp_to_primitives`
  - `validate_mtp_geometry`
- `include/phaseshift/models/qwen35/weights/model_weights.h`
  - `Qwen35MtpWeights` / `Qwen35ModelWeights::mtp`

### executor / state

- `include/phaseshift/models/qwen35/runtime/mtp_executor.h`
- `src/phaseshift/models/qwen35/runtime/mtp_executor.hip`
  - `create_mtp_executor` / `mtp_executor_shutdown`
  - `run_mtp_rows` / `mtp_forward_step` / `run_mtp_head`
- `include/phaseshift/models/qwen35/runtime/mtp_kv_state.h`
- `src/phaseshift/models/qwen35/runtime/mtp_kv_state.cpp`
  - `create_mtp_kv_state` / `mtp_kv_reset` / `mtp_kv_reserve` / `mtp_kv_shutdown`
  - `mtp_kv_physical_slot` / `mtp_kv_state_view` / `mtp_kv_dump_canonical`

### speculative decode

- `include/phaseshift/models/qwen35/runtime/spec_decode.h`
  - `SpecPhase` / `SpecTransaction` / `SpecVerifyResult`
  - `spec_greedy_accept`
  - `spec_transaction_begin` / `spec_transaction_begin_verify` /
    `spec_transaction_commit` / `spec_transaction_rollback` / `spec_transaction_abort`
  - `spec_gdn_snapshot` / `spec_gdn_restore`
  - `SpecDraftStep` / `SpecDraftSet` / `MtpDraftPolicy` / `mtp_generate_drafts`
- `src/phaseshift/models/qwen35/runtime/spec_decode.cpp`
- `include/phaseshift/models/qwen35/runtime/spec_decoder.h`
  - `SpecDecoderConfig` / `SpecDecoderTiming` / `SpecDecoder` / `SpecIterationOutput`
- `src/phaseshift/models/qwen35/runtime/spec_decoder.cpp`
  - `create_spec_decoder` / `spec_decoder_shutdown`
  - `spec_decoder_sync_prompt` / `spec_decoder_step`

### numeric mode / dispatch

- `include/phaseshift/runtime/execution/execution_types.h`
  - `VerifyNumericMode` / `ExecutionRole` / `ExecutionClass`
- `src/phaseshift/models/qwen35/runtime/program_executor.h`
  - `verify_exact_active`
- `src/phaseshift/models/qwen35/runtime/linear_selector.cpp`
- `src/phaseshift/models/qwen35/runtime/gdn_recurrence_dispatch.hip`

### tests

required:

- `tests/unit/test_qwen35_mtp_weight_load.hip`
- `tests/unit/test_qwen35_mtp_lowering.hip`
- `tests/unit/test_qwen35_mtp_primitives.hip`
- `tests/unit/test_qwen35_mtp_kv_cache.hip`
- `tests/unit/test_qwen35_mtp_state.hip`

optional（実 checkpoint / env 駆動）:

- `tests/unit/test_qwen35_mtp_weight_real.hip`
- `tests/unit/test_qwen35_mtp_forward_real.hip`
- `tests/unit/test_qwen35_mtp_state_real.hip`
- `tests/unit/test_qwen35_mtp_spec_real.hip`
- `tests/unit/test_qwen35_mtp_gate3_replay.hip`
- `tests/unit/test_qwen35_mtp_multistep.hip`
- `tests/unit/test_qwen35_mtp_spec_perf.hip`
- `tests/unit/test_qwen35_mtp_verify_divergence.hip`
- `tests/unit/test_qwen35_mtp_gate4e.hip`

### tools / bench

- `tools/inspect_qwen35_mtp_weights.py`
- `tools/reference/export_qwen35_mtp_gate1.py` / `export_qwen35_mtp_gate2.py`
- `tools/compare_mtp_gate1.py` / `compare_mtp_gate2.py`
- `src/apps/bench/mtp.hip`（`phaseshift-bench mtp`）
- `src/apps/bench/tg.hip`（`phaseshift-bench tg --mtp`）
