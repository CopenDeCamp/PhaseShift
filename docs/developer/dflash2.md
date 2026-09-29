# DFlash2 speculative decoding

DFlash2 は target model の途中層 hidden を入力に取り、1 anchor block 分の draft token を
まとめて提案する drafter による greedy speculative decoding である。target が提案 block を
Exact 数値モードで verify し、一致した prefix だけを commit する。

対象:

- target: `models/Qwen3.8-27B-*`（Qwen3.8-27B、64 層）
- drafter: `models/Qwen3.8-27B-DFlash2`（BF16）または
  `models/Qwen3.8-27B-DFlash2-PSQ4`（backbone を PSQ4 化、指定は `--dflash2-model-dir`）
- `--dflash2-model-dir` を指定したときだけ DFlash2 経路に入る。未指定時は既存の
  `ContinuousBatcher` 経路を変更しない。

経緯・benchmark・採用判断は [DFlash2 実装記録](../rnd/dflash2/dflash2.md) を参照。
現在の性能値は [current performance](../perf/current.md) を参照。

---

## Scope

- greedy のみ（`temperature == 0`）。stochastic speculative decoding は実装しない。
- target verify の数値モードは **Exact** を既定とし、CLI も Exact を渡す。
- drafter は 1 round で anchor 1 row + draft `block_size - 1` row を陽に処理する
  block-parallel 構成である。提案 block 内は非 causal。
- target 層数・tap 位置などは drafter の config が宣言し、target の層数と shape に
  突き合わせる。27B の既定は target 64 層、tap `[5, 19, 33, 47, 61]`。
- runtime 統合は `phaseshift-compute` の `run_dflash2_shot()` にある。executor / pool /
  arena / stream は `Qwen35ComputeRuntime` の accessor から取得する。`phaseshift-cli` は
  `--dflash2-model-dir` / `--dflash2-drafts` / `--dflash2-stats` を
  `phaseshift-compute` に転送する。

---

## Checkpoint contract

### config

`config.json` は flat（`text_config` を持たない）。`read_dflash2_config()` が読む key:

| key | 意味 |
| --- | --- |
| `architectures[0]` | `DFlash2DraftModel` のみ許可 |
| `vocab_size` / `hidden_size` / `intermediate_size` / `num_hidden_layers` | 形状 |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | attention 形状 |
| `sliding_window` | sliding attention の窓幅 |
| `is_causal` | **false のみ許可**（draft block 内は非 causal） |
| `num_target_layers` | target の層数 |
| `rms_norm_eps` | RMSNorm の eps |
| `tie_word_embeddings` | **false のみ許可**（drafter は embedding/lm_head を持たない） |
| `dflash_config.block_size` | anchor + draft の合計 row 数（8 なら anchor 1 + draft 7） |
| `dflash_config.conv_group_size` / `conv_kernel_size` | dynamic convolution 形状 |
| `dflash_config.mask_token_id` | noise block の MASK token |
| `dflash_config.selector_rank` / `selector_top_k` | candidate selector 形状 |
| `dflash_config.target_layer_ids` | target から tap する layer（昇順、`num_target_layers` 未満） |
| `rope_parameters` | 任意。存在する場合 `rope_type == "default"` のみ許可 |
| `layer_types` | 任意。存在する場合すべて `sliding_attention` のみ許可 |

契約検証（違反はすべて load 時の error。既定値で黙って補わない）:

- `block_size ∈ [2, 16]`（`kMaxBlockSize`）。
- `target_layer_ids` は空でなく、最大 `kMaxTargetLayerIds`（8）個、strictly increasing、
  各要素 `< num_target_layers`。
- `hidden_size % conv_group_size == 0`。
- `num_attention_heads % num_key_value_heads == 0`。
- `mask_token_id < vocab_size`、`selector_top_k <= vocab_size`。
- 必須次元（vocab / hidden / intermediate / layers / attention / sliding_window /
  num_target_layers）は 0 でない。

導出値:

- `max_draft_tokens = block_size - 1`
- `conv_groups = hidden_size / conv_group_size`
- `conv_projection_rows = 2 * conv_kernel_size * conv_groups`
- `attention_q_rows = num_attention_heads * head_dim`
- `attention_kv_rows = num_key_value_heads * head_dim`
- `tap_feature_size = num_target_layer_ids * hidden_size`

`create_dflash2_executor()` は加えて次を要求する。

- `num_target_layer_ids == kDFlash2TargetTaps`（5）
- `conv_kernel_size == kDFlash2ConvKernelSize`（2）

### weight

- BF16 safetensors、または self-contained 量子化 safetensors（PSQ4）を受け付ける。
- tensor 名と shape は `dflash2_expected_tensors(config)` が唯一の定義。
- missing tensor / unexpected tensor / shape 不一致 / dtype 不一致はすべて error。
  error message には tensor 名を含める。
- embedding / lm_head は checkpoint に存在しない。target 側を再利用する。
- 27B 用 checkpoint は 81 tensor、3,848,808,960 bytes。
- 量子化 bundle は `is_quantized_model_dir()` で判定し、`QuantizedModelReader` で開く。

### 量子化 drafter（PSQ4）

backbone weight だけを PSQ4（`PSQ4_S32_BF16_V1`）にできる。proposal と acceptance は
PSQ4 化で変化してよいが、`GENERATED_IDS` は target-only greedy と一致しなければならない。

- 量子化する行列は backbone linear の 46 本のみ:
  `fc.weight` と各層の `self_attn.{q,k,v,o}_proj` / `mlp.{gate,up,down}_proj` /
  `attention_conv.kernel_projection` / `mlp_conv.kernel_projection`（9 本 × 5 層 + 1）。
- PSQ4 にしない tensor: 全 norm（`hidden_norm` / `norm` / `input_layernorm` /
  `post_attention_layernorm` / `q_norm` / `k_norm`）、conv の `base_kernel`、
  `candidate_selector.hidden_projection.weight`、両 codebook。これらは BF16 のまま。
- offline 側は `quantization/offline/dflash2_adapter.{h,cpp}`（`DFlash2DraftModel` の
  detect と tensor 分類）と `adapter_dispatch.{h,cpp}`（Qwen3.5 との振り分け）が担当する。
  quantizer は DFlash2 draft に対して `--preset psq` 以外と `--imatrix` を拒否する
  （silent fallback しない）。
- runtime 側の contract（`validate_dflash2_tensor_contract` / `load_dflash2_weights`）は
  manifest `architecture == "dflash2_draft"`、logical tensor set の完全一致、
  backbone 46 本の encoding が Psq4、それ以外が Bf16 であることを検証する。
  PSQ4 行列は preshuffle 済み（`preshuffled=true`）で `weight_scale_group=32`、
  `k_padded % 32 == 0` を要求する。load 後は `hipStreamSynchronize` する。
- executor は PSQ4 backbone の時に activation を e4m3（per-row scale）へ量子化して
  W4A8 WMMA kernel に渡す。同一 activation を共有する GEMM は 1 回の量子化で再利用する
  （q/k/v noise、gate/up、context の K/V 全層）。
- PSQ4 の geometry が allowlist に無い場合、または activation workspace が
  量子化時の形状と一致しない場合は、silent fallback せず error を返す。

---

## Target bridge

### hidden taps

DFlash2 drafter は target の途中層の出力を入力に使う。tap は
`Qwen35LowerOptions::hidden_taps` で明示指定し、**自動推測しない**。

値の定義:

- tap する値は target graph の各層 final output、すなわち residual 加算後の値である
  （Transformers の `hidden_states[layer_id + 1]` に相当）。graph 上では
  debug name `L<layer>.output` の node の出力がこれに一致する。

不変条件:

- tap を指定しない場合、target graph の external output は 4 個のままで、buffer 確保も
  kernel 追加も行わない。出力 token は一切変わらない。
- tap を指定した場合、external output の index `4..4+n-1` に tap が追加される。
- tap buffer は `Executor` が所有する `dflash_target_hidden[k]`（`[Npad, hidden]` BF16。
  `Npad` は executor の padded token-row capacity）で、`token_hidden` とは別 buffer に
  明示的に bind する（alias させない）。
- host backend の external I/O 静的 capacity は `kHostExternalCapacity = 16`。
- tap は layer の row 数分（token rows）を保持する。
- `kMaxTargetHiddenTaps = 8`。`target_hidden_tap_count` がこれを超える場合は executor
  create が error。

`test_dflash2_target_taps` が、同じ layer output を `ValueTraceSink` で捕捉し、
external tap と internal layer output の bit 一致を確認する。tap の有無で sampled token が
一致することも同時に確認する。

### target feature projection

`dflash2_project_target_features()` は次の 3 段を 1 回の呼び出しで enqueue する。

1. `launch_dflash2_feature_concat`: 5 本の tap `[rows, 5120]` を feature 軸へ連結し
   `target_concat [rows, 25600]` を作る。tap の順序は `target_layer_ids` の順。
2. `dflash2_linear`: `fc.weight [5120, 25600]` を掛けて `target_fc [rows, 5120]`。
   BF16 なら bf16 GEMM、PSQ4 なら W4A8。
3. `launch_dflash2_rmsnorm_direct_bf16`: `hidden_norm.weight` で
   `target_feature [rows, 5120]`。

RMSNorm は **DIRECT**（`ONE_PLUS` は使わない）、`group_size = hidden_size`、
`eps = rms_norm_eps`。公式実装の `Qwen3RMSNorm` と同じく、norm を bf16 に落としてから
weight を掛ける。

### target embedding / noise

- `DFlash2ExecutorConfig::target_embed_tokens` で target の `embed_tokens` を借りる。
  所有も複製もしない。`DFlash2Executor` は同 pointer を保持し shutdown で nullptr にする。
- `create_dflash2_executor()` 時点で `embed_tokens` の shape が
  `[vocab_size, hidden_size]` であることを検証する。
- `noise_token_ids [block_size] I32` と `noise_embedding [block_size, hidden] BF16` を
  create 時に確保する（`target_embed_tokens` 指定時のみ）。
- `launch_dflash2_noise_token_ids()` が row0=anchor、row1 以降に `mask_token_id` を書く。
  anchor は kernel argument で渡す（H2D memcpy を毎回しない）。
- `dflash2_prepare_noise_embedding()` が token id 生成と embedding lookup を行う。
  BF16 / PSQ8 の両 encoding を既存 `launch_embedding_bf16` / `launch_embedding_psq8` で
  処理する。新しい embedding kernel は作らない。未対応 encoding は error。

### target lm_head 共有

- DFlash2 専用の lm_head を作らない。target の `MatrixWeight` を
  `DFlash2ExecutorConfig::target_lm_head` で借りる。`DFlash2Executor` は参照のみで
  複製も所有もしない。
- PSQ8 の場合、`program.cpp` の LINEAR 経路と同じ契約で計算する:
  `launch_activation_quantize_e4m3`（活性は行ごとの `peak/448` scale、累算 F32）と
  `launch_gemm_psq8_w8a8_wmma_auto`。W8A8 の A 側は int8 ではなく **e4m3** である。
- proposal logits は **F32** で出す（bf16 へ落とさない。top-16 ordering が proposal に
  直結するため）。
- `create_dflash2_spec_decoder()` が target との shape 一致
  （`hidden_size` / `vocab_size` / `lm_head` shape）と target 層数 > 最大 tap を検証する。
- INT2 coarse head を使う場合も、この PSQ8 lm_head が唯一の source である。

### ownership

- target tap buffer は `Executor` が所有し、`create_model_executor` で arena から確保、
  `executor_shutdown` で解放する。
- DFlash2 drafter の workspace は `DFlash2Executor` が所有し、
  `create_dflash2_executor` で確保、`dflash2_executor_shutdown` で解放する
  （arena ownership のため shutdown は pointer を nullptr に戻すだけ）。
- DFlash2 の persistent sequence state は `DFlash2ContextState` が所有し、
  `create_dflash2_context_state` で確保、`dflash2_context_shutdown` で解放する。
- `DFlash2ExecutorConfig::target_lm_head` / `target_embed_tokens` は借りるだけで所有しない。

---

## Drafter execution

### layer execution

`dflash2_forward_layer_stateless()` は次の順で enqueue する。順序を変えない。

1. layer input norm: `input_layernorm.weight` で `layer_input_norm` を作る。
   PSQ4 で融合が eligible なときは residual+rmsnorm+quant 融合が置き換える（後述）。
2. `attention_prepared = conv_prepare(layer_input_norm, attention_conv)`
   （kernel_projection と phase 0 conv。phase 1 の dynamic を保持）。
3. `q_raw = q_proj(attention_prepared)`
4. `k_ctx_raw = k_proj(target_feature)`, `v_ctx = v_proj(target_feature)`
5. `k_noise_raw = k_proj(attention_prepared)`, `v_noise = v_proj(attention_prepared)`
6. `q_norm` / `k_ctx_norm` / `k_noise_norm`（RMSNorm head_dim 単位、V は norm しない）
7. RoPE: Q は block positions、K_ctx は context positions、K_noise は block positions
8. DFlash2 attention（context + noise の 2 segment）
9. `attention_o = o_proj(attention_context)`
10. `attention_finished = conv_finish(attention_o, attention_conv)`
11. `attention_residual = hidden_in + attention_finished`
12. `post_attention_norm = post_attention_layernorm(attention_residual)`
13. `mlp_prepared = conv_prepare(post_attention_norm, mlp_conv)`（`conv_dynamic` を上書き）
14. MLP: gate/up、SwiGLU、down
15. `mlp_finished = conv_finish(down, mlp_conv)`
16. `output = attention_residual + mlp_finished`

`conv_dynamic` は attention prepare → attention finish → MLP prepare → MLP finish の順で
使い回す単一 buffer である。attention finish より前に MLP prepare を enqueue してはならない
（single stream の順序で保証する）。

`dflash2_forward_backbone_stateless()` / `dflash2_forward_backbone_cached()` は
`num_hidden_layers` を 0→N-1 の順に直列 enqueue し、最後に final RMSNorm（`norm.weight`）を
掛けて caller の output pointer へ直接書く。

- 入力は `noise_embedding [block_rows, hidden]` と `target_feature [context_rows, hidden]`。
  target feature projection は **呼び出し側で 1 回だけ**実行し、全 layer で同じ pointer を
  共有する。
- layer 間の relay は **ping-pong buffer の pointer 渡し**のみ。D2D memcpy を挟まない。
  `backbone_hidden_a`（偶数 layer）/ `backbone_hidden_b`（奇数 layer）を使う。
- intermediate tensor へ書いた後に copy しない。final norm は output へ直接書く。
- 引数の contract: `block_rows <= min(max_rows, block_size)`、
  `context_rows <= max_context_rows`、`context_position_start + context_rows ==
  block_position_start`（stateless のみ。cached は `block_position_start ==
  context.next_position`）。
- stateless path は同一入力・同一 position の複数回実行が bit 一致する
  （conv_dynamic の残留、ping-pong lifetime、layer 間 state leakage が無い）。

host 側の layer loop は enqueue のみで、GPU→CPU の read も途中判断も無い。禁止するのは
「GPU → CPU へ token → CPU 判断 → 次 GPU」であり、noise token id や anchor は kernel
argument として渡す。

### GroupedDynamicCausalConv

- `kernel_projection` の出力 `[rows, 1280]` は `[rows, 2 (phase), 2 (offset), 320 (group)]`
  として解釈する。group が最内。
- `base_kernel [2, 2, 5120]` は `[phase][offset][feature]`。
- 出力の各要素は
  `Σ_offset (base[phase][offset][f] + dynamic[row][phase][offset][group]) * input[row - offset][f]`
  （`row - offset < 0` の項は 0）。
- dynamic は **`row` の行**を使う（`row - offset` ではない）。block-local causal conv であり、
  persistent conv state は持たない。
- `prepare` は kernel_projection を実行し、phase 0 の conv 結果を返し、phase 1 の dynamic を
  executor の `conv_dynamic` に保持する。
- `finish` は kernel_projection を再実行せず、保持した `conv_dynamic` と phase 1 の base を
  使う。`prepare` → `finish` の順で呼ぶこと。

### attention

- `is_causal = false`、mask 条件は `abs(query_position - key_position) < sliding_window`
  （strict）。2047 は visible、2048 は masked。
- block 内の future row も visible。通常の causal attention を流用してはならない。
- Q の position は block、K_ctx の position は context、K_noise の position は block。
- GQA: `kv_head = q_head / (q_heads / kv_heads)`。27B checkpoint は 32 / 8 = 4。
- 実装は 3 系統:
  - `launch_dflash2_attention_bf16`: stateless contiguous。context と noise を別 segment で
    受け取り、物理 concat buffer を作らない。1 workgroup = 1 query row × 1 q head。
    correctness oracle として残す。
  - `launch_dflash2_attention_ring_bf16`: persistent ring から slot 計算で context を読む
    参照実装。mask / GQA / softmax は stateless と同一。
  - `launch_dflash2_attention_ring_gqa_bf16`: production。1 WG = 1 block row × 1 KV head
    で 4 q head を処理し、K/V を LDS tile へ uint4 coalesced load して 4 head で共有する。
    数学は参照実装と **bit-exact**（online softmax には移行していない）。
- GQA kernel の制限（不成立時は `hipErrorNotSupported`）: `q_heads == 4 * kv_heads`、
  `head_dim == 128`、strides の 8 要素倍数、dynamic LDS ≤ 60KB。executor は既定で GQA を
  使い、NotSupported のとき ring 参照実装へフォールバックする。
  `PHASESHIFT_DFLASH2_ATTENTION_REFERENCE=1` で参照実装を固定できる。

### final normalization / dtype contract

公式実装は bf16 tensor 同士の演算で各 op ごとに丸める。PhaseNonShift は次の 3 か所で
同じ丸め列を再現する（`round_to_bf16_f32`）。

- GroupedDynamicCausalConv: `round(kernel * values)` → `round(out + ...)` → `addcmul`
- RoPE: cos/sin を bf16 に丸め、積と和も bf16 に丸める
- SwiGLU: `silu(gate)` を bf16 に丸めてから `up` と掛け、積も bf16 に丸める

RMSNorm は公式の `weight * norm.to(bf16)` と同じく、norm を bf16 に落としてから weight を
掛ける（`launch_dflash2_rmsnorm_direct_bf16`）。公式 forward の中間 tensor はすべて bf16、
attention_mask は bool である。

RoPE は **f32 の `inv_freq`** から `powf` で計算する。`Qwen3RotaryEmbedding` の `inv_freq`
は checkpoint に含まれない非 persistent buffer であり、model 全体を bf16 化すると丸まる。
角度は `position * inv_freq` なので、量子化誤差は position 倍で増幅する。

### primitive dispatch

DFlash2 executor は layer 内の各 logical primitive をすべて独立した kernel launch として
実行する。対象は layer input/post-attention RMSNorm、Q/K の RMSNorm と RoPE、
q/k/v projection GEMM、attention、Gate GEMM、Up GEMM、SwiGLU、Down GEMM、
grouped dynamic conv と residual、INT2 head path の rerank / Top-16 / pool-id remap である。
複数の logical primitive を 1 physical kernel へ結合する fusion は production には存在しない。

### hot-path contract

`dflash2_project_target_features` / `dflash2_grouped_conv_prepare` /
`dflash2_grouped_conv_finish` / `dflash2_forward_layer_*` / `dflash2_forward_backbone_*` /
`dflash2_select_draft_tokens` / `dflash2_propose_cached` / `dflash2_append_target_taps` /
`dflash2_prepare_noise_embedding` は非同期 enqueue API である。以下を含んではならない。

- `hipStreamSynchronize` / `hipDeviceSynchronize`
- `hipMemcpyDeviceToHost` / `hipMemcpyHostToDevice`
- ホットパスでの allocation（arena / hipMalloc）

buffer 確保は `create_dflash2_executor` のみで行う。host へ戻るのは
`dflash2_spec_step` の proposal / decision の D2H と、テスト側の比較・計測のときだけである。

### no-D2D relay contract

kernel の出力 buffer を次 kernel の入力 pointer として直接渡す。D2D memcpy による relay を
行わない。context append も wrap 対応の scatter kernel で書き、D2D memcpy を使わない。

---

## Persistent context state

`DFlash2ContextState` は target が確定した context の K/V を保持する sequence state である。

### ownership

- `DFlash2ContextState` は `DFlash2Executor` から分離した sequence state である。
  executor の scratch ではない。
- `create_dflash2_context_state()` が arena から ring を確保し、
  `dflash2_context_shutdown()` で解放する。destructor には依存しない。
- `dflash2_context_reset(state, next_position)` は GPU buffer を memset しない。
  `length = 0`、`next_position = requested` を設定するだけである。stale data は
  `length` 外なので読まない。

### layout / memory

| buffer | shape | dtype |
| --- | --- | --- |
| `k_ring` | `[num_layers, capacity, kv_features]` | BF16 |
| `v_ring` | `[num_layers, capacity, kv_features]` | BF16 |

- production では `capacity == config.sliding_window`（2048）を要求する。
- `kv_features = attention_kv_rows() = num_key_value_heads * head_dim`（27B は 8 × 128 = 1024）。
- 27B の実 allocation は `2 * 5 * 2048 * 1024 * 2 = 41,943,040 bytes`（40 MiB / sequence）。

### ring が保持する値

- K は raw K を保持しない。`target_feature -> k_proj -> k_norm(RMSNorm head_dim) ->
  RoPE(absolute position)` の結果を保存する。
- V は `target_feature -> v_proj` の結果を保存する。
- proposal ごとに k_norm / RoPE をやり直さない。

### physical mapping

- write index を別管理しない。physical slot は `absolute_position % capacity`。
- logical row `i` の absolute position は `context_start + i`、
  `context_start = next_position - length`。
- この `slot = position % capacity` を append kernel と ring attention kernel で共通に使う。

### next_position 契約

- `state.next_position` は target が次に consume する absolute position。
- `context.length <= capacity`、`context_start = next_position - length`。
- cached backbone は `block_position_start == state.next_position` を要求する
  （block row 0 が anchor の position と一致する）。

### append 契約

`dflash2_context_append(executor, state, target_feature, rows, position_start, stream)`:

- input は `target_feature [rows, hidden]`、`position_start == state.next_position`。
- layer ごとに `k_proj -> k_norm -> RoPE(effective_position)` と `v_proj` を実行し、
  `k_ctx_raw` / `k_ctx_norm` / `k_ctx_rope` / `v_ctx` scratch を一時 buffer として再利用する。
  これらの scratch は cached proposal では使わない。
- `rows > capacity` のときは最後の `capacity` row だけ生成する。
  `skip = max(0, rows - capacity)`、`effective_position = position_start + skip`。
  metadata の `next_position` は `position_start + rows` まで進める。
- append 後: `length = min(capacity, old_length + rows)`、`next_position += rows`。
- host metadata は kernel を enqueue した後に更新する。GPU completion 待ちはしない
  （`hipStreamSynchronize` を呼ばない）。
- `effective_rows > max_context_rows` は error。

### commit-only semantics

- ring に入るのは **target が consume して commit した row** だけである。
- noise / MASK block、DFlash draft hidden、reject された target row は入れない。
- DFlash2 側に rollback API は無い。accepted row だけを append する。
- prefill / target embedding からの自動構築はしない。呼び出し側が明示的に append する。

### cached attention / backbone

`dflash2_forward_layer_cached()` は stateless layer から context K/V 生成 4 段
(`k_ctx_proj` / `v_ctx_proj` / `k_ctx_norm` / `k_ctx_rope`) を除いた経路である。

- noise 側は stateless と同一。
- context は ring を直接読む。
- attention は ring 用 kernel（既定 GQA、フォールバック ring 参照）を使う。
- attention 以降の `o_proj -> residual -> post norm -> MLP -> residual` は stateless と
  共通の内部 helper を使う。

### proposal read-only

- `dflash2_forward_backbone_cached()` / `dflash2_forward_and_select_cached()` /
  `dflash2_propose_cached()` は context state を read-only として扱う。
  `length` / `next_position` / ring K/V を変更しない。
- 同一 context state に対する proposal は何回呼んでも同じ結果になる。
- noise K/V を ring へ commit する設計にはしない。

### prefix cache

DFlash path では target prefix cache を使わない。prefix cache の checkpoint に
`DFlash2ContextState` が含まれず、target context だけ復元して DFlash ring が空になる
silent 不整合を避けるためである。`--prefix-cache-capacity-tokens` は error にする。

---

## Proposal pipeline

### candidate selector API

公開 API は次の 4 つである。

- `dflash2_compute_target_logits()`: `final_hidden` から target lm_head の logits を計算する。
- `dflash2_run_candidate_selector()`: hidden_proj / top-k から draft token 列を確定する。
- `dflash2_select_draft_tokens()`: backbone output から draft token 列までを一括で行う。
- `dflash2_forward_and_select_stateless()`: stateless backbone + select の便宜 API。

`dflash2_select_draft_tokens()` の流れ:

```
final_hidden の anchor row を除く draft row
  -> selector hidden_projection (exact bf16 gemm)
  -> proposal head
  -> top-16
  -> CandidateSelector
  -> draft_tokens[draft_rows]
```

すべて device 上で実行する。D2H / H2D / D2D relay / allocation は無い。
`lm_head` の入力は `final_hidden + hidden_size`（row 1 以降）で、anchor row は渡さない。

### proposal head

- full PSQ8（既定）: `lm_head_logits()` が activation を e4m3 へ量子化し、PSQ8 W8A8 で
  `proposal_logits [draft_rows, vocab_size]` F32 を作る。
- INT2 coarse + PSQ8 rerank（後述）: `int2_select_path()` が full logits を materialize
  せず、coarse Top-N と exact rerank で top-16 を作る。

### top-16

`launch_dflash2_topk_f32()` は 2 stage 構成である。

1. 1 row = 1 workgroup。各 thread が自分の shard を走査して local top-16 を保持し、
   256 thread 分（256×16 候補）を scratch へ書く。
2. 1 row = 1 thread が 256×16 候補から top-k を選び、降順で出力する。

ordering は **logit 降順、同値なら token id 昇順**。f32 の同値判定は bit 等価で行う
（丸め誤差による擬似 tie は許容しない）。full sort はしない。

production は `launch_dflash2_topk_f32_optimized`（hierarchical reduction）。

- partition stage: grid = `rows * partitions`。1 WG が vocab partition を担当し、
  thread local top16 から WG 内 cooperative selection で partition top16 を求める。
- merge stage: grid = `rows`。1 WG が `partitions * 16` candidate から cooperative
  selection で top16 を求める。
- scratch は `[rows, kDflash2TopKScratchPerRow]`（`kDflash2TopKThreads *
  kDflash2TopKMaxK = 256 * 16 = 4096`）を再利用する。
- plan は `DFlash2TopKPlan`。production default は partitions=32 / threads=256。
- `dflash2_select_draft_tokens` は optimized を呼ぶ。`top_k != 16` または
  `PHASESHIFT_DFLASH2_TOPK_REFERENCE=1` のときは reference に切り替える。
- 契約は不変: logit descending、exact tie は token id ascending、NaN は既存 comparator を
  維持、出力 IDs は exact。

### CandidateSelector

```
pred_0 = anchor_token
for r in 0..draft_rows-1:
    score_j = top16_logit[r][j]
              + dot(predecessor_codebook[pred_r] * hidden_proj[r],
                    successor_codebook[top16_id[r][j]])
    best_j  = argmax_j score_j     # 同値なら candidate token id 小
    draft_token[r] = top16_id[r][best_j]
    pred_{r+1} = draft_token[r]
```

`hidden_proj[r] = hidden_projection(final_hidden[r + 1])` は selector の前に draft row 分を
一括で計算する（position loop 内で再計算しない）。

累算は F32。codebook と hidden_proj は bf16 load で、積は `pred * hidden * successor` を
F32 で足し込む。

selector は **1 workgroup で draft 列全体を確定する**。position ごとに kernel を分けたり、
token を host へ戻して次の position を決めたりしない。`draft_rows` は引数で受け、
固定 7 回 loop はしない。

### block_rows と draft_rows

`draft_rows = block_rows - 1`。

| block_rows | draft_rows |
| --- | --- |
| 2 | 1 |
| 4 | 3 |
| 8 | 7 |

### INT2 coarse head + PSQ8 exact rerank

proposer の full PSQ8 lm_head を、INT2 coarse 全語彙探索 → coarse Top-N → original PSQ8
lm_head による candidate rerank → exact Top-16 に置き換える。target の lm_head と Verify は
一切変更しない（DFlash proposal 専用の lossy head）。`PHASESHIFT_DFLASH2_INT2_HEAD`で
明示選択できる。固定語彙は`PHASESHIFT_DFLASH2_DRAFT_VOCAB=1`または明示FILEでのみ有効にする。

data:

- `DFlash2DraftHeadInt2` は generic な `MatrixEncoding` ではない。executor が
  `target_lm_head`（PSQ8, preshuffled）から **load 時に一度だけ**作る runtime object。
- 追加 resident は **INT2 codes のみ**（`((vocab+15)/16) * k_padded * 4` bytes。
  248320 × 5120 で 317,849,600 bytes ≈ 303.1 MiB）。weight scale は target PSQ8 の
  `storage_scale_stride_bytes` の BF16 をそのまま borrow し、追加を持たない。
- codebook（4 値）と 256-entry の map / expand LUT（計 1280 bytes）を executor が保持する。
- hot path allocation は 0。生成は startup のみ。
- INT2 object を作れない場合は silent fallback せず error を返す。

codebook:

- source は target PSQ8 lm_head。GPU 上で 256 bin の重み付き histogram
  （weight = 出現数 × block scale²、固定小数点 uint64）を作る。
- host 側で finite な E4M3 候補に対する weighted Lloyd-Max（4 centroid）を行い、
  最後に各 centroid を最寄りの valid E4M3 byte へ snap する。
  `PHASESHIFT_DFLASH2_INT2_CODEBOOK=symmetric` では {-b,-a,+a,+b} を weighted squared
  error で総当たりする。既定は Lloyd-Max。
- 決定論的（同じ head なら同じ 4 byte）。

packing:

- `psq8 e4m3 code -> nearest codebook index`（256 B の map LUT）を全要素へ適用し、
  4 code / byte（bits 1:0 = w0, 3:2 = w1, 5:4 = w2, 7:6 = w3）で pack する。
- layout は PSQ8 preshuffle をそのまま 1/4 に圧縮したもので、16-row tile / 32-k block
  あたり 512 B → 128 B。各 thread が読む 8 byte PSQ8 chunk は 2 byte INT2 chunk になる。
- GPU repack（target PSQ8 codes を CPU に戻さない）。map/expand LUT は startup に
  CPU で作り upload する。

coarse kernel:

- `launch_dflash2_int2_coarse_head` は PSQ8 W8A8 kernel と同一構造で、W 側だけを
  「packed byte → expand LUT（256 × uint32）で 4 E4M3 byte に展開」する。
  activation は E4M3 preshuffle を共有し、A/B の WMMA と per-row activation scale も
  PSQ8 kernel と同一。expand LUT は block 先頭で LDS に 1 KiB stage する。
- したがって coarse の誤差は **codebook 近似のみ**であり、kernel 自体は
  「INT2 を dequant した E4M3 行列」に対する PSQ8 W8A8 と bit-exact になる。

coarse Top-N:

- `launch_dflash2_coarse_topn` は 2 stage。stage 1 は `rows × partitions` の各 WG が担当
  区間を走査し、per-thread top-16（register）から exact top-pool を選んで scratch へ書く。
  stage 2 は 1 WG/row が `partitions × pool` を merge して exact top-pool を出す。
- `partitions = ceil(vocab / (kDflash2Int2TopnThreads * kDflash2Int2TopnPerThread))`
  （512 × 16 = 8192 要素/partition、最大 `kDflash2Int2MaxPartitions` = 128）。
  これにより 1 thread の担当要素数が 16 以下となり、per-thread top-16 が exact になる。
  launcher はこれを検証し、超える設定を `hipErrorInvalidValue` で拒否する
  （暗黙に近似しない）。
- tie は (logit desc, token id asc)。pool は `kDflash2TopKMaxK`(16) 以上
  `kDflash2Int2MaxPool`(128) 以下で、既定 32。

Radix Select Top-N:

- `launch_dflash2_radix_topn` は full sort ではなく
  threshold selection → exact N candidate extraction → small N sort である。
- float32 ordered bits の 4 pass radix（31:24 → 7:0）。各 pass は
  `rows × partitions` の histogram kernel と `rows` の select kernel。
  prefix を満たす element だけを histogram し、上位 bucket から累積して
  N 番目を含む bucket を確定する。4 pass 後に N 番目の exact float32 threshold が得られる。
- `count_gt`（threshold より大）と `count_eq`（threshold と同値）を出し
  `need = N - count_gt`。`count_eq > need` の場合にのみ、threshold と同一 float の
  element を対象に token ID の radix selection（vocab 幅の 1 走査 bitmap）を行い、
  同値が起きない通常ケースでは ID 側の追加走査を発生させない。
- candidate は出力 buffer へ compact されて件数が正確に N になり、
  `next_pow2(N)` の bitonic で key 降順に sort して出力する。
- ordering contract は `detail::topn_key()` と完全互換（score 降順、同値は token id 昇順、
  `+0/-0` 同値、NaN と `-INFINITY` の扱い、tail の `(id=0, -INFINITY)` 満たしを含む）。
- scratch は init 時に
  `dflash2_radix_topn_scratch_bytes(rows, partitions, vocab)` で決め、実行時に
  `hipMalloc` しない。
- 選択は `dflash2_radix_topn_preferred(pool)` が決める:
  `PHASESHIFT_DFLASH2_RADIX_TOPN=0` で常に従来実装、`=1` で常に radix、
  未指定では `pool >= kDflash2RadixTopnCrossoverPool`（= 64）で radix。
  従来実装 `launch_dflash2_coarse_topn` は oracle / fallback として残す。

PSQ8 exact rerank:

- `launch_dflash2_psq8_candidate_rerank` は PSQ8 W8A8 kernel の weight row を
  `candidate_ids[row][cand]` から直接引く（gather buffer を作らない）。activation 行は
  block の draft row を全 lane で共有し、k 順序・scale・WMMA は本家と同一。出力は候補位置
  `[rows, pool]` に書く。
- 同じ PSQ8 行列の同じ候補に対する full head の logit と **bit-exact** である。
- 1 WG = 1 warp、16 候補 × 1 draft row。pool=32 で 2 tile/row。

small Top-16 と remap:

- rerank logits 上の exact top-16 は `launch_dflash2_topk_f32_optimized` を vocab=pool で
  使う。tie 規則は本家と同一。
- **重要**: この kernel が返す id は rerank logits の index（= pool 内位置）である。
  `launch_dflash2_int2_remap_pool_ids` で `pool_ids[row][position]` を引いて token id に
  戻す。これを忘れると proposal が pool 位置を token id として返し、acceptance が 0 になる。
  rerank / Top-16 / remap はそれぞれ独立した kernel launch である。

env:

- `PHASESHIFT_DFLASH2_INT2_HEAD`（0 = full PSQ8、1 = INT2、2 = diag。未指定時は0。明示した固定語彙が適合すれば1）
- `PHASESHIFT_DFLASH2_DRAFT_RERANK`（pool。既定 32、[16, 128] に clamp）
- `PHASESHIFT_DFLASH2_INT2_CODEBOOK`（`symmetric` で対称 codebook）
- `PHASESHIFT_DFLASH2_INT2_DIAG` / `PHASESHIFT_DFLASH2_INT2_TIMING`（診断。既定 off）
- INT2 有効時も target-only / Verify は INT2 コードパスを通らない。

### 固定語彙profile

one-shotとserve-stdioの両方で、target model directoryの`dflash2-draft-vocab.json`と
`dflash2-draft-vocab.u32`を解決する。resolverはGPU allocationを行わず、選択したpathと
INT2 modeを`DFlash2ExecutorConfig`へ渡す。診断はstderrへ出し、serveのJSON stdoutを汚さない。

metadataは`schema_version=phaseshift-dflash2-vocab-v1`、`profile_id`、
`vocab_file=dflash2-draft-vocab.u32`、`vocab_count`、`target_vocab_size`、`target_hidden_size`、
`vocab_sha256`、`tokenizer_sha256`を持つ。payloadはlittle-endian uint32の昇順・重複なしID列で、
件数は16の倍数、IDはtarget vocab内、件数はrerank pool以上である。
未知schema、破損、部分配置、不正IDを黙って利用しない。

固定語彙は既定off。`PHASESHIFT_DFLASH2_DRAFT_VOCAB`の未指定・空文字・0では標準profileを読まない。
`INT2_HEAD=1`だけなら標準profileが配置済みでも全語彙INT2を使う。
`DRAFT_VOCAB=1`を明示し、形状・payload hash・target tokenizer.jsonの実SHAが一致すれば、INT2＋固定語彙を使う。
有効化時にprofileなし、または正常だが非適合なら、head未指定時はfull PSQ8、明示mode 1なら全語彙INT2を維持する。
明示mode 0/2は標準profileより優先する。明示`DRAFT_VOCAB_FILE`はそれ自体がopt-inであり、
raw ID列を受け付けmode 1へ接続する。FILEとmode 0/2または明示`DRAFT_VOCAB=0`の競合は拒否する。

compact INT2 codesとscaleを元PSQ8の選択行から直接作る。codebookは全語彙PSQ8から導出し、
coarseとTop-Nはlocal ID、rerankとCandidateSelectorはglobal IDを使う。Top-Nの直後にID mapで復元する。
縮小PSQ8行列は持たない。`PHASESHIFT_DFLASH2_DRAFT_VOCAB_CHECK=1`は明示診断としてfull INT2/PSQ8
対照へ照合し、追加buffer・同期・D2Hを伴うため性能測定では無効にする。

profile作成者は、検証済みID列とtokenizerから配布用directoryを作れる。

```sh
python3 tools/quantization/prepare_draft_vocab.py pack \
  --vocab-file /path/to/selected-ids.u32 --tokenizer /path/to/tokenizer.json \
  --profile-id qwen3.8-27b-98304-v1 --model-vocab-size 248320 --hidden-size 5120 \
  --source-manifest-sha256 <manifest-sha256> \
  --notice docs/references/qwen38-draft-vocab-notice.txt --output-dir /path/to/new-bundle
```

packはコーパスからIDを選ぶtoolではない。集計レシピで作ったID列を包み、生成元NOTICEを付ける。
installは`tokenizer_vocab_sha256`により表現形式に依存しないtoken→ID対応を確認し、配置先の
`tokenizer_sha256`へmetadataを結び直す。token map digestはmodel.vocabとadded_tokensを統合し、
`(id, UTF-8 token bytes)`順に`u32le(id) || u32le(byte length) || token bytes`をSHA-256へ入力する。
推論時はtokenizerをPythonで処理せず、配置済みmetadataと実ファイルのSHAを照合する。

---

## Speculative transaction

### prefill

`dflash2_spec_prefill()`:

- prompt を `target->config.max_scheduled_tokens` 単位で chunk 実行する。
- intermediate chunk: `compute_logits=false`, `sample=false`。
- final chunk: `compute_logits=true`, `sample=true`, `num_output_rows=1`。
- 各 chunk 実行直後に target tap を ring へ append する。
- final chunk の sampled token を pending token（anchor）として返す。
- prefill 後: `sequence.position == prompt_count`、`context.next_position == prompt_count`、
  `context.length == min(prompt_count, capacity)`。
- anchor の absolute position は `prompt_count` であり、まだ ring へ commit しない。

### step

`dflash2_spec_step(decoder, pending_token, remaining_tokens)`:

1. `K` を clamp する: `min(config.num_drafts, remaining_tokens - 1, room - 1)`。
   `room = sequence.max_seq_len - position`。`remaining <= 1` または `room < 2` なら `K = 0`。
2. `K == 0` のときは単一 target decode（`execution_class = DECODE`,
   `speculative_verify = false`, numeric mode Fast, `num_output_rows = 1`）を行い、
   sampled token を emit して target tap 1 row を ring へ append する。
3. `K > 0` のとき:
   - DFlash proposal（`dflash2_propose_cached`）を行い、draft 列を得る。
   - target verify: `[anchor, draft_0..draft_{K-1}]` の `K+1` 行を
     `execution_class = PREFILL`（`K+1 > 1` のとき）, `speculative_verify = true`,
     `verify_numeric_mode = config.verify_numeric_mode`（既定 **Exact**）,
     `num_output_rows = K+1`, `prefix_tokens = position` で実行する。
   - target samples から `spec_greedy_accept` で longest-prefix acceptance を行う。

### accept / reject / commit

`spec_greedy_accept(drafts, K, sampled, bonus_token_enabled=true)`:

- draft と target sampled token が先頭から何 row 一致するかを数える（longest prefix）。
- `accepted < K` のとき correction token（最初に reject した row の target token）を返す。
- `accepted == K` のとき bonus token（block 末尾で target が sample した次 token）を返す。

commit は target state が確定してから行う（commit-only）。DFlash ring は proposal / verify /
acceptance の間は変更しない。

- full accept（`accepted == K`）: target state は既に正しい。verify の target tap `K+1` 行を
  `position_start = position` で ring へ append し、余剰 KV page を
  `rollback_sequence_append(ceil(sequence.position / page_tokens))` で trim する。
  pending は bonus token。
- partial（`accepted < K`、既定）: `restore_gdn_spec_history(pool, slot, history, accepted)`
  で conv / recurrent を live へ戻し、`sequence.position = position + accepted + 1` に設定し、
  verify の target tap rows `0..accepted` を ring へ append し、KV page を trim する。
  pending は correction token。rerun は行わない。
- partial（`PHASESHIFT_DFLASH2_GDN_RERUN_REFERENCE=1`）: verify 前の base snapshot へ
  restore し、`sequence.position = position` に戻して `[anchor, accepted drafts]` の
  `accepted+1` 行を rerun し、その tap を append する。reference / 調査用である。

### GDN state history

partial reject 時に full target rerun を不要にするため、verify の各 row 直後の GDN state を
その場で history へ記録する。既定は history path である。

- `GdnSpecHistory` は decoder 専用の single-sequence checkpoint で、sequence slot の
  GDN state を row 数分保持する。`GdnStatePool` 自体へは埋め込まない。
  - `conv`: `[row][num_gdn_states][conv_history][conv_history_stride]` BF16
    （live slot と同一 layout）
  - `recurrent`: `[row][num_gdn_states][num_v_heads][head_k][head_v]` F32
    （live slot と同一 layout）
- rows = `config.num_drafts`（最大 K）。capture_rows は round の実 K。
- allocation は create 時の 1 回のみ。hot path allocation は行わない。
- create 時に `history bytes > 1.5 GiB` なら decoder create を FAIL する
  （FP16 化等で逃げない）。
- `DFlash2SpecDecoderConfig::num_drafts` は `1..max_draft_tokens()` でなければ error。

capture:

- `execute_batch` に `ExecuteBatchOptions`（`GdnSpecHistoryDeviceView`）を渡す。
  scheduler contract へは漏らさない。history の書き込みは `role == Verify`、
  `numeric_mode == Exact`、history pointer != nullptr のときだけ行う。
- GDN recurrence は `row_override` 番目の writeback で live state と同一の local offset へ
  mirror store する。GDN conv は row 更新直後に `s0/s1/s2` を BF16 で mirror store する。
  いずれも kernel 内で書くため per-row / per-layer の D2D copy を作らない。

restore:

- `restore_gdn_spec_history(pool, slot, history, history_row)` は conv / recurrent を
  各 1 回の D2D copy で live へ戻す。

target verify の GDN recurrence 本体:

- Exact verify では `gdn_recurrence_wmma_decode_rows_exact_impl<kRows>`
  （`kRows = 2 / 4 / 8`）を使い、`decode1` の exact algorithm をそのまま row loop で回す。
  演算順序・reassociation・K split は不変である。
- dispatch 条件は `verify_exact_active` ∧ 27B geometry（`head_k == 128` かつ
  `head_v % 64 == 0`）∧ `decode1_supported` ∧ `decode_rows_supported`。
  `actual_rows` が 2 / 4 / 8 のとき multirow kernel が選ばれる。
- `PHASESHIFT_GDN_RECURRENCE_MULTIROW=0` で旧 serial を強制できる。

### EOS

`config.eos_token >= 0` のとき、emit 列の最初の EOS で truncate して `finished = true`。
pending token が EOS でも `finished = true`。

### state invariant

各 step 終了時:

- `context.next_position == sequence.position`
- `context.length == min(sequence.position, capacity)`
- proposal は context を変更しない
- proposal / verify / acceptance の間、DFlash ring を変更しない

### hot path（transaction）

- per-round の host allocation をしない（verify/draft は固定長 `std::array`）。
- device bridge path では 1 round に 1 回だけ `hipStreamSynchronize` する（後述）。
- `dflash2_spec_step` は timing を `DFlash2SpecDecoder::timing` に記録する。

---

## Device-resident path

DFlash proposal を Host へ戻さず、そのまま target verify の入力にする。

### env

`PHASESHIFT_DFLASH2_DEVICE_TOKEN_BRIDGE`（既定 1）。

- `1`: candidate device-resident bridge。
- `0`: legacy Host bridge（A/B と regression 調査用）。

### ScheduledBatch の token source

`ScheduledBatch::token_ids_location`（`TokenIdsLocation::Host` / `Device`、既定 Host）で
memory location を明示する。Device のときも pointer 非 null を要求する。Host CPU が
`token_ids` を dereference することはない。

executor の upload は Host source なら `hipMemcpyHostToDevice`、Device source なら
`hipMemcpyDeviceToDevice` を使う。対象は `executor.input_ids` と
`batch_context_storage.token_ids` の 2 箇所である。

### graph cache

`GraphCacheSlot` は replay 条件として token count / output 数 / attention plan key /
verify exact / exec role / spec history / capture rows / history pointer に加えて、
Device source のときの source pointer（`token_device_src`）を保持する。Device source では
Host staging への `std::memcpy` を行わず、capture 内で device verify token buffer →
executor input buffer の D2D をそのまま capture する。

### device buffer

`create_dflash2_spec_decoder()` が arena から以下を確保する。decoder lifetime 中
アドレス不変で、shutdown では arena ownership のため free せず pointer を nullptr に戻す。

| buffer | bytes | 用途 |
| --- | ---: | --- |
| `verify_token_ids_device` | `kMaxBlockSize * 4` | `[pending, draft_0, ..., draft_{K-1}]` |
| `decision_staging_device` | `2 * kMaxBlockSize * 4` | 前半 drafts / 後半 target samples |

### round 契約

1. `pending_token` を `verify_token_ids_device[0]` へ H2D（4 byte）。
2. `dflash2_propose_cached` を同一 stream へ enqueue。proposal output は
   `decoder.draft->proposal_tokens` に残す。
3. `proposal_tokens[0:K]` → `verify_token_ids_device[1:K+1]` の D2D を enqueue。
   ここに sync を入れない。
4. verify 用 `ScheduledBatch` を `token_ids = verify_token_ids_device`、
   `token_ids_location = Device` で構築する。CPU は draft token の値を知らないまま
   `submit_batch` まで先行できる。
5. `submit_batch` 後、proposal tokens と sampled tokens を decision staging へ D2D。
6. `hipStreamSynchronize` を round 中唯一の Host synchronization point として実行。
7. decision staging を 1 回だけ D2H し、`spec_greedy_accept` を CPU で実行する。

proposal → target verify の間に `hipStreamSynchronize` も D2H も存在しない。

legacy Host bridge では proposal 直後に `hipStreamSynchronize` + D2H を行い、target
samples を別途 D2H する。

### timing

`DFlash2SpecTiming` は `draft_gpu_ms` / `proposal_wait_ms` / `proposal_copy_ms` /
`verify_gpu_ms` / `decision_wait_ms` / `decision_copy_ms` を持つ。legacy 比較用に
`draft_ms` / `draft_d2h_ms` / `verify_ms` / `round_ms` も持つ。

`draft_d2h_ms` は legacy metric である。`DFLASH2_D2H_MS` は純粋な PCIe 転送時間では
なく、先頭の `hipStreamSynchronize` のため draft completion wait を含む。
`proposal_wait_ms` と `proposal_copy_ms` は分離して測る。

---

## Runtime configuration

### CLI（`phaseshift-compute`）

- `--dflash2-model-dir PATH`: DFlash2 draft model directory（指定で DFlash mode）
- `--dflash2-drafts N`: 1 round の draft 数（default 7、`block_size - 1` 以下）
- `--dflash2-stats 0|1`: speculative decode 統計行の出力（default 1）

`phaseshift-cli` は上記 3 つを受け取り、`--dflash2-model-dir` 指定時のみ
`phaseshift-compute` に転送する。

### invalid combinations（model load 前に exit 2）

- `--temperature > 0`（greedy のみ）
- `--dump-logits`
- `--constraint-tokenizer-info`
- `--prefix-cache-capacity-tokens > 0`
- `--dflash2-drafts` が `block_size - 1` を超える

silent fallback は行わない。

### serve mode（`--serve-stdio`）

`--dflash2-model-dir` は `--serve-stdio` と併用できる。server backend 経由で
DFlash2 speculative decoding を有効化する唯一的な経路である。

- serve は `run_serve_stdio()` 内で drafter weight と `DFlash2Executor` を serve 間
  保持し、request ごとに `PagedSequenceState` / `DFlash2ContextState` /
  `DFlash2SpecDecoder` を作成・解放する。解放順序は decoder → context → sequence。
- drafter の `DFlash2Executor` は serve 終了時に `dflash2_executor_shutdown()` する。
- request 時に次の3つは fail-closed で拒否する（error event）。
  - `temperature > 0`
  - `grammar` / `structural_tag`（constraint）
  - `prefix_cache_checkpoint_position`
- `max_concurrent_requests` は 1。`ServeState.dflash_enabled` が true のとき
  `serve_step()` は `ContinuousBatcher::step()` の代わりに
  `dflash2_spec_step()` を呼ぶ。
- `generated_ids` / `finish_reason` / `prompt_tokens` は serve 側の
  `ServeGeneration` が保持する（`RuntimeRequest` を使わない）。

### target executor config（DFlash 有効時）

- `max_concurrent_requests = 1`
- `target_hidden_taps = dflash_config.target_layer_ids`、順序も config 通り
- `target_hidden_tap_count = dflash_config.num_target_layer_ids`
- `max_scheduled_output_rows = block_size`
- `max_scheduled_tokens = min(2048, max(prompt_tokens, block_size))`
- `max_scheduled_output_rows` は `max(max_scheduled_output_rows, max_scheduled_requests)`
  で切り上げられた後 `max_scheduled_tokens` で上限されるため、
  `block_size <= max_scheduled_tokens` を満たす必要がある。prefill は
  `max_scheduled_tokens` 単位で chunk 実行する。

### DFlash2 drafter executor config

- `max_rows = block_size`
- `max_context_rows = sliding_window`
- `target_lm_head = target.weights().lm_head`
- `target_embed_tokens = target.weights().embed_tokens`

### env 一覧

| env | 既定 | 意味 |
| --- | --- | --- |
| `PHASESHIFT_DFLASH2_DEVICE_TOKEN_BRIDGE` | 1 | 0 で legacy Host bridge |
| `PHASESHIFT_DFLASH2_GDN_RERUN_REFERENCE` | 0 | 1 で base snapshot + full target rerun の参照 path |
| `PHASESHIFT_DFLASH2_TOPK_REFERENCE` | 0 | 1 で reference top-k |
| `PHASESHIFT_DFLASH2_ATTENTION_REFERENCE` | 0 | 1 で ring attention 参照実装 |
| `PHASESHIFT_DFLASH2_INT2_HEAD` | 0 | 0でfull PSQ8、1でINT2、2でdiag。固定語彙の明示有効化時は未指定なら1 |
| `PHASESHIFT_DFLASH2_DRAFT_VOCAB` | 0 | 1で標準固定語彙profileを検証して有効化 |
| `PHASESHIFT_DFLASH2_DRAFT_VOCAB_FILE` | 未指定 | 独自のraw ID列を明示指定 |
| `PHASESHIFT_DFLASH2_DRAFT_RERANK` | 32 | rerank pool（[16, 128] に clamp） |
| `PHASESHIFT_DFLASH2_INT2_CODEBOOK` | lloyd | `symmetric` で対称 codebook |
| `PHASESHIFT_DFLASH2_INT2_DIAG` | off | INT2 診断出力 |
| `PHASESHIFT_DFLASH2_INT2_TIMING` | off | INT2 区間 timing 出力 |
| `PHASESHIFT_DFLASH2_RADIX_TOPN` | auto | 0 で従来 topn、1 で radix topn、未指定は pool crossover |

target verify の GDN recurrence は `PHASESHIFT_GDN_RECURRENCE_MULTIROW`（既定 on、0 で
serial 強制）で切り替わる。

### resource lifecycle

DFlash mode の生成後は次の順で解放する。

1. `DFlash2SpecDecoder` shutdown
2. `DFlash2ContextState` shutdown
3. `PagedSequenceState` release
4. `DFlash2Executor` shutdown

その後、共通経路で target `Executor` shutdown → stream destroy → arena shutdown を行う。
pool leak 検査（kv page / seq slot / gdn slot）は DFlash mode でも共通経路で実施する。

### 統計行

`--dflash2-stats 1` のとき以下を stdout に出力する。

```
DFLASH2_DRAFTS, DFLASH2_BLOCK_SIZE, DFLASH2_VERIFY_NUMERIC_MODE=Exact,
DFLASH2_ROUNDS, DFLASH2_FULL_ACCEPTS, DFLASH2_RERUNS, DFLASH2_PARTIAL_ACCEPTS,
DFLASH2_GDN_HISTORY_BYTES, DFLASH2_ACCEPTED_DRAFTS, DFLASH2_MEAN_ACCEPTED,
DFLASH2_FULL_ACCEPT_RATE, DFLASH2_EMITTED_PER_ROUND,
DFLASH2_DRAFT_MS, DFLASH2_D2H_MS, DFLASH2_DRAFT_GPU_MS, DFLASH2_PROPOSAL_WAIT_MS,
DFLASH2_PROPOSAL_COPY_MS, DFLASH2_VERIFY_MS, DFLASH2_VERIFY_GPU_MS,
DFLASH2_DECISION_WAIT_MS, DFLASH2_DECISION_D2H_MS, DFLASH2_DEVICE_TOKEN_BRIDGE,
DFLASH2_RERUN_MS, DFLASH2_GDN_SNAPSHOT_MS, DFLASH2_GDN_RESTORE_MS, DFLASH2_COMMIT_MS,
DFLASH2_ROUND_MS
```

`GENERATED_IDS=` 形式は target-only と完全に同一で、`phaseshift-cli` は変更なしで
DFlash mode の出力を扱える。

---

## Invariants

- proposal は context state を変更しない（read-only）。
- DFlash ring に入るのは target が consume して commit した row だけ（commit-only）。
- step 終了時に `context.next_position == sequence.position`、
  `context.length == min(sequence.position, capacity)`。
- `GENERATED_IDS` は target-only greedy と完全一致する。
- target verify は Exact 数値モードで行う。
- drafter kernel は D2D memcpy による relay を行わない。
- hot-path enqueue API は sync / D2H / H2D / allocation を行わない。
- PSQ4 drafter でも `GENERATED_IDS` は target-only greedy と一致する。
- すべての契約違反・非対応構成は明示的な error とし、silent fallback しない。
- buffer は所有側（`Executor` / `DFlash2Executor` / `DFlash2ContextState` /
  `DFlash2SpecDecoder`）が確保し、明示的な shutdown 経路を持つ。

---

## Known limitations

- greedy のみ。stochastic speculative decoding は未実装。
- target prefix cache / grammar constraint / `--dump-logits` との併用は非対応。
  `--serve-stdio` と `--kv-cache-dtype psq4` は serve mode と併用できる。
- target 層数 > 最大 tap、`num_target_layer_ids == 5`、`conv_kernel_size == 2` を要求する。
- context ring は sliding_window 分しか保持しない。`capacity == sliding_window` を要求する。
- attention GQA kernel は `q_heads == 4 * kv_heads`、`head_dim == 128` を要求し、それ以外は
  ring 参照実装へフォールバックする（不成立自体はエラーではない）。
- GDN history は `history bytes > 1.5 GiB` のとき create を FAIL する。
- INT2 coarse head は preshuffled な PSQ8 lm_head を要求する。作れない場合は error。
- PSQ4 drafter の activation 量子化は allowlist geometry を要求し、非対応時は error。
- `--dflash2-drafts` は `block_size - 1` 以下。`block_size <= max_scheduled_tokens` を
  満たす必要がある。

---

## Source map

### drafter

- `include/phaseshift/models/qwen35/dflash2/{config.h,weights.h,executor.h,context_state.h}`
- `src/phaseshift/models/qwen35/dflash2/{config.cpp,weights.cpp,executor.hip,context_state.hip}`

### kernels

- `include/phaseshift/models/qwen35/kernels/dflash2/` と
  `src/phaseshift/models/qwen35/kernels/dflash2/`
  - `feature_concat` / `grouped_dynamic_conv` / `rmsnorm` / `rope` /
    `swiglu` / `attention` / `kv_ring` / `noise_input`
  - `candidate_selector` / `topk` / `topk_optimized`
  - `draft_head_int2` / `coarse_topn` / `radix_topn`
  - `detail/`（`coarse_head_device.h` / `coarse_topn_device.h` /
    `psq8_rerank_device.h` / `rmsnorm_device.h` / `rope_device.h` / `swiglu_device.h`）

### runtime

- `include/phaseshift/models/qwen35/runtime/dflash2_spec_decoder.h`
- `src/phaseshift/models/qwen35/runtime/dflash2_spec_decoder.cpp`
- `src/phaseshift/models/qwen35/runtime/gdn_spec_history.hip` と
  `include/phaseshift/models/qwen35/runtime/gdn_spec_history.h`
- `src/phaseshift/models/qwen35/runtime/spec_decode.{h,cpp}`
- `include/phaseshift/models/qwen35/runtime/scheduled_batch.h`

### apps / quantization

- `src/apps/compute/main.hip`（`run_dflash2_shot`）
- `src/apps/compute/compute_runtime.{h,hip}`（tap / max output rows の引き継ぎ）
- `src/apps/cli/phaseshift_cli.py`
- `include/phaseshift/quantization/offline/dflash2_adapter.h` /
  `src/phaseshift/quantization/offline/dflash2_adapter.cpp`
- `src/phaseshift/quantization/offline/adapter_dispatch.cpp`

### build / tests

- `cmake/options.cmake`
- `cmake/targets.cmake` / `cmake/apps.cmake` / `cmake/tests.cmake`
- required: `test_dflash2_config` / `test_dflash2_quantization_adapter` /
  `test_dflash2_weight_contract` / `test_dflash2_feature_concat` /
  `test_dflash2_grouped_dynamic_conv` / `test_dflash2_bf16_geometry` /
  `test_dflash2_rope` / `test_dflash2_attention` / `test_dflash2_topk` /
  `test_dflash2_candidate_selector` / `test_dflash2_kv_ring` /
  `test_dflash2_attention_ring` / `test_dflash2_psq4_shapes` /
  `test_dflash2_int2_pack` / `test_dflash2_int2_coarse_head` /
  `test_dflash2_coarse_topn` / `test_dflash2_radix_topn` /
  `test_dflash2_psq8_rerank` /
  `test_gdn_spec_history` / `test_gdn_recurrence_decode1`
- perf（label `gpu1;perf`、正しさテストと分離）: `test_dflash2_radix_topn_perf`
- optional / external: `test_dflash2_weight_real` / `test_dflash2_gate*_real` /
  `test_dflash2_gate*_perf` / `test_dflash2_gate8_live` / `test_dflash2_gate9_e2e` /
  `test_dflash2_gate10_cli` / `test_dflash2_gate11*`
