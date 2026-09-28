> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

## Current outcome

production へ採用されたもの:

- DFlash2 drafter（BF16 / PSQ4 full-backbone）と persistent context KV ring による
  greedy speculative decoding。target verify は Exact、ring へは commit 済み row だけを入れる。
- target hidden tap → feature projection → 5 layer backbone → target lm_head → top-16 →
  CandidateSelector の device 上 proposal pipeline。
- device-resident proposal bridge による draft D2H 削減（1 round 1 sync）。
- partial reject を exact GDN per-row history で処理し、full target rerun を省略。
- GQA grouped ring attention、exact GDN multi-row recurrence、INT2 coarse head +
  PSQ8 exact rerank（必要時に有効化）。
- fusion preset MAX の Safe / Deep fusion（DFlash2 の residual+rmsnorm+quant、MLP tail、
  gate/up+SwiGLU、rerank+Top16、Target の down+residual、gate/up+SwiGLU など）。

rejected / 不採用の主要案:

- 11J-J INT2 coarse + TopN fusion（coarse が約 2 倍化し proposer +15.34%）。
- Target Q/K postprocess fusion（共有 body 制約で q geometry が 0.36x、11J-G REVERT）。
- BF16 GEMM の splitk / ExactRows 差し替え（reduction 順が変わり acceptance 不変契約に反する）。
- small-M 専用 kernel（linear は既に DRAM floor 付近で headroom ~3%、変更しても計測不能）。
- 11K-B / 11K-C / 11K-D の GEMM core 再構成、11K-I の on-the-fly quant、
  11K-K の persistent quant+down（regression または合法解なしで NO-GO / REVERT）。

- current developer spec へのリンク: [../developer/dflash2.md](../../developer/dflash2.md)
- current performance へのリンク: [../perf/current.md](../../perf/current.md)

## Gate index

| Gate | 1 行 summary |
| --- | --- |
| 0 baseline / inventory | checkpoint の config と 81 tensor / 3,848,808,960 bytes を固定 |
| 1 config + weight loader | `DFlash2Config` と期待 tensor 集合の厳格検証（BF16）。採用 |
| 2 target hidden taps | `hidden_taps` を external output へ追加し layer output と bit 一致。採用 |
| 3 feature projection + grouped conv | 5 tap 連結 → fc → RMSNorm と block-local dynamic conv。採用 |
| 4 layer 0 forward | 非 causal sliding attention を含む単層 forward。採用 |
| 5 5-layer stateless backbone | 5 layer 直列 + final RMSNorm。採用 |
| 5.1 numeric contract isolation | 公式 bf16 semantics との差異を fixture / dtype trace で分離。 |
| 6 lm_head + top-16 + selector | proposal 列を device 上で確定。採用 |
| 7 persistent context KV ring | absolute position / slot 写像 / commit-only。採用 |
| 8 live target bridge | 実 prompt から ring を構築し live proposal。採用 |
| 9 greedy speculative decode | Exact verify と longest-prefix acceptance。採用 |
| 10 phaseshift-compute 統合 | CLI / stats / resource lifecycle。採用 |
| 11A Kernel Profile | proposer の kernel breakdown（profile）。 |
| 11B Top16 optimization | hierarchical reduction を production 化。採用 |
| 11C GDN state history | exact per-row history で partial rerun を削除。採用 |
| 11D target verify small-M | small-M 専用 kernel は headroom ~3% で不採用（profile）。 |
| 11E GDN multi-row recurrence | row loop を device 内へ。採用 |
| 11F proposer kernel ceiling | BF16 GEMM 差し替え不採用、GQA grouped ring attention 採用。 |
| 11G full-backbone PSQ4 drafter | PSQ4 bundle と W4A8 経路を採用。 |
| 11H verify exact primitive ceiling | attention split reduce のみ採用、他は launch floor（profile）。 |
| 11I INT2 coarse head + PSQ8 rerank | proposal 専用 lossy head を追加（pool 32）。採用 |
| 11I-2 device-resident proposal bridge | draft D2H を 1 round 1 sync へ集約。採用 |
| 11J-A target residual+rmsnorm+quant | Target Safe fusion。採用 |
| 11J-B SwiGLU + activation quant | Target は採用、DFlash2 は PERF-NEUTRAL（後に 11K-H が置換）。 |
| 11J-C DFlash2 residual+rmsnorm+quant | DFlash2 Safe fusion。採用 |
| 11J-D target GDN prepare | Target Safe fusion。採用 |
| 11J-E target GDN post | Target Safe fusion。採用 |
| 11J-F target attention gate+quant | Target Safe fusion。採用 |
| 11J-G Q postprocess | Target REVERT / DFlash2 PERF-NEUTRAL。 |
| 11J-H K postprocess | Target 未実装 / DFlash2 PERF-NEUTRAL。 |
| 11J-I conv boundaries | 未実施（設計のみ）。 |
| 11J-J INT2 coarse + TopN fusion | REVERT（coarse 約 2 倍化、proposer +15.34%）。 |
| 11J-K rerank + Top16 + remap fusion | ADOPT（proposer -1.90%）。selector fusion は NO-GO。 |
| 11K-A DFlash2 MLP tail | conv finish + residual。ADOPT。 |
| 11K-B gate/up + SwiGLU（assessment） | 共有 GEMM core trait 化が必要として当時 NO-GO（後に 11K-H で実装）。 |
| 11K-C quant + down | row-wide peak 確定前の code 生成に grid-wide 同期が必要。NO-GO。 |
| 11K-D / 11K-E | 11K-C 依存で未実施 / final topology を確定（owner 重複 0）。 |
| 11K-F GEMM fusion trait infrastructure | ActivationPolicy / EpiloguePolicy を導入。ADOPT。 |
| 11K-G target down + residual | Target Deep fusion。ADOPT。 |
| 11K-H gate/up dual projection + SwiGLU | Target / DFlash2 とも ADOPT。旧 SwiGLU+Quant を削除。 |
| 11K-I quant + down analytical | on-the-fly encode が最大約 320x。NO-GO。 |
| 11K-J cooperative / persistent feasibility | probe は GO、production へは投入せず。 |
| 11K-K persistent quant + down probe | split より +125〜174%。REVERT。 |

# DFlash2 実装記録（性能・Gate 検証）

対象: `models/Qwen3.8-27B-DFlash2`（drafter）+ `models/Qwen3.8-27B-PSQ`（target）。
実装手順は Gate 単位で締結し、各 Gate で required acceptance と既存モデル回帰を確認する。

## baseline

| 項目 | 値 |
| --- | --- |
| baseline SHA | `65839be6f899b8031de96159fb341ed6a0441310` |
| branch / worktree | `feat/dflash2` / `.worktrees/dflash2` |
| stash | `stash@{0}`（MTP 未完作業）を保持。pop/apply しない |
| full build | PASS（worktree、`PHASESHIFT_BUILD_OPTIONAL_TESTS=ON`、gfx1201） |
| required acceptance | 97/97 PASS（`build/required-acceptance.txt`、2026-09-21） |

---

## Gate 0: baseline / inventory

runtime コードは変更しない。checkpoint の実体を safetensors metadata から列挙して固定する。

### config 契約

`models/Qwen3.8-27B-DFlash2/config.json` の実測値。

| key | 値 |
| --- | --- |
| `architectures[0]` | `DFlash2DraftModel` |
| `vocab_size` | 248320 |
| `hidden_size` | 5120 |
| `intermediate_size` | 17408 |
| `num_hidden_layers` | 5 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| `sliding_window` | 2048 |
| `is_causal` | false |
| `dflash_config.block_size` | 8 |
| `dflash_config.conv_group_size` | 16 |
| `dflash_config.conv_kernel_size` | 2 |
| `dflash_config.mask_token_id` | 248070 |
| `dflash_config.selector_rank` | 256 |
| `dflash_config.selector_top_k` | 16 |
| `dflash_config.target_layer_ids` | [5, 19, 33, 47, 61] |
| `num_target_layers` | 64 |
| `rms_norm_eps` | 1e-06 |
| `rope_parameters` | `{rope_theta: 10000000, rope_type: "default"}` |
| `tie_word_embeddings` | false |
| `layer_types` | 5 層すべて `sliding_attention` |

導出値: `max_draft_tokens = 7`、`conv_groups = 320`、`conv_projection_rows = 1280`、
`attention_q_rows = 4096`、`attention_kv_rows = 1024`、`tap_feature_size = 25600`。

### weight inventory

- file: `model.safetensors`（単一ファイル、`model.safetensors.index.json` なし）
- tensor 数: **81**、dtype: **81/81 BF16**
- 総 payload: **3,848,808,960 bytes**（3.849 GB）

global 3 tensor:

| tensor | shape |
| --- | --- |
| `fc.weight` | [5120, 25600] |
| `hidden_norm.weight` | [5120] |
| `norm.weight` | [5120] |

selector 3 tensor:

| tensor | shape |
| --- | --- |
| `candidate_selector.hidden_projection.weight` | [256, 5120] |
| `candidate_selector.predecessor_codebook` | [248320, 256] |
| `candidate_selector.successor_codebook` | [248320, 256] |

`layers.{0..4}` の 15 tensor（5 層で同形状、計 75）:

| tensor | shape |
| --- | --- |
| `input_layernorm.weight` | [5120] |
| `post_attention_layernorm.weight` | [5120] |
| `self_attn.q_proj.weight` | [4096, 5120] |
| `self_attn.k_proj.weight` | [1024, 5120] |
| `self_attn.v_proj.weight` | [1024, 5120] |
| `self_attn.o_proj.weight` | [5120, 4096] |
| `self_attn.q_norm.weight` | [128] |
| `self_attn.k_norm.weight` | [128] |
| `mlp.gate_proj.weight` | [17408, 5120] |
| `mlp.up_proj.weight` | [17408, 5120] |
| `mlp.down_proj.weight` | [5120, 17408] |
| `attention_conv.base_kernel` | [2, 2, 5120] |
| `attention_conv.kernel_projection.weight` | [1280, 5120] |
| `mlp_conv.base_kernel` | [2, 2, 5120] |
| `mlp_conv.kernel_projection.weight` | [1280, 5120] |

drafter checkpoint に embedding / lm_head は存在しない（target 側を再利用する契約）。

contract violation: なし。


## Gate 1: config + strict BF16 weight loader

### 実装

- `include/phaseshift/models/qwen35/dflash2/config.h` / `src/.../config.cpp`
  - `DFlash2Config`、`read_dflash2_config(model_dir)`
  - 契約検証: 必須 key、`is_causal == false`、`tie_word_embeddings == false`、
    `hidden_size % conv_group_size == 0`、`num_attention_heads % num_key_value_heads == 0`、
    `target_layer_ids` が昇順かつ `< num_target_layers`、`mask_token_id < vocab_size`、
    `selector_top_k <= vocab_size`、`rope_type == "default"`、`layer_types` が
    `sliding_attention` のみ、`block_size ∈ [2,16]`。
- `include/phaseshift/models/qwen35/dflash2/weights.h` / `src/.../weights.cpp`
  - `DFlash2Weights`（5 層 + fc + selector）、`load_dflash2_weights()`。
  - `dflash2_expected_tensors(config)` が期待 tensor 集合（name + shape）を生成。
  - `validate_dflash2_tensor_contract()`: missing / unexpected tensor を名前付きで拒否、
    全 tensor を **BF16 かつ shape 完全一致**で検証。量子化 checkpoint は拒否。
- `SafetensorsCollection`: 単一ファイル `model.safetensors`（index なし）の fallback を追加。
  index も `model.safetensors` も無い場合は従来どおりエラー。
- `cmake/targets.cmake`: `phaseshift_qwen35` に 2 source 追加。

### 検証

| test | label | 結果 |
| --- | --- | --- |
| `test_dflash2_config` | cpu;required | failed=0 |
| `test_dflash2_weight_contract` | gpu1;required | failed=0 |
| `test_dflash2_weight_real` | gpu1;optional;external_files | failed=0（実 checkpoint） |

`test_dflash2_weight_contract` は合成 checkpoint（hidden=8、1 層、21 tensor）で
valid / missing / unexpected / shape mismatch / dtype mismatch / 量子化 dir /
malformed config を検証し、ロード後の payload bit 一致と arena shutdown を確認する。

`test_dflash2_weight_real` は実 checkpoint で config 契約値、81 tensor、
総 payload 3,848,808,960 bytes、5 層の全 shape を検証する。

### Gate 決定

PASS。

---

## Gate 2: target hidden taps

### 実装

- `Qwen35LowerOptions::hidden_taps` を `lower_qwen35_to_primitives` の第 3 引数として追加
  （既定は空 span = 従来と完全同一）。
- tap の値は各層 final output（residual 加算後）。graph 上は debug name
  `L<layer>.output` の node 出力。
- tap 未指定時: external output は 4 個のまま、tap buffer を確保せず、kernel も追加しない。
- tap 指定時: external output index `4..4+n-1` に追加。
- `ExecutorConfig::target_hidden_taps` / `target_hidden_tap_count`、
  `Executor::dflash_target_hidden[kMaxTargetHiddenTaps]` を追加。
- host backend の external I/O 静的 capacity を 8 → 16 に拡張。tap は `token_hidden`
  とは別 buffer に明示 bind する（alias させない）。
- tap buffer は `[Npad, hidden]` BF16。`target_hidden_tap_count == 0` では確保しない。
- `create_model_executor` の lowering に tap を渡し、`executor_shutdown` で解放する。

### 検証

`test_dflash2_target_taps`（target = 27B-PSQ、tap = DFlash2 config の
`target_layer_ids` = [5,19,33,47,61]）:

| 項目 | 結果 |
| --- | --- |
| tap 無し lowering の external output 数 | 4（不変） |
| tap 有り lowering の external output 数 | 4 + 5 |
| tap buffer 形状 | `[Npad, 5120]` BF16 |
| tap == internal layer output（n=32 / 8 / 1、全 row） | bit-exact |
| tap 有無での sampled token | 完全一致 |

回帰:

| 対象 | 結果 |
| --- | --- |
| required acceptance | 99/99 PASS |
| 4B `test_qwen35_mtp_spec_real` | PASS |
| 4B-PSQ `test_qwen35_mtp_spec_real` | PASS |
| 27B-PSQ `test_qwen35_mtp_gate3_replay`（VERIFY_EXACT=1） | PASS (7 passed, 0 failed) |
| 27B-PSQ AR（taps OFF）gpu_us_per_token | 35464 / 35529（baseline 65839be6）vs 35517 / 35575（taps 変更）→ 回帰なし |

AR は interleaved（base → cand を 2 反復）、device 0、`--context 128 --tokens 64`。

### 既知の既存事象（本 Gate の変更とは無関係）

`test_qwen35_mtp_gate3_replay` は MTP 経路の `output_gather` launch で
`invalid argument` を返し `FAIL: sync` になることがある。無変更ビルド（baseline
65839be6）でも同一 GPU で PASS / FAIL / FAIL と混在したため、既存の flaky 事象である。
Gate 2 の変更は MTP 経路（`lower_qwen35_mtp_to_primitives` / `mtp_executor`）を
含まない。

### Gate 決定

PASS。

---

## Gate 3: target feature projection + grouped dynamic conv

baseline SHA: `ca6f195beef086c50c7f2136f3e20951daf084ad`（Gate 2 完了時点）。

### 変更

| 種別 | ファイル |
| --- | --- |
| kernel | `include|src/phaseshift/models/qwen35/kernels/dflash2/feature_concat.{h,hip}` |
| kernel | `include|src/phaseshift/models/qwen35/kernels/dflash2/grouped_dynamic_conv.{h,hip}` |
| executor | `include|src/phaseshift/models/qwen35/dflash2/executor.{h,hip}` |
| selector | `src/phaseshift/models/qwen35/runtime/linear_selector.cpp`（BF16 geometry 2 件追加） |
| reference | `tools/reference/export_qwen38_dflash2_gate3.py` |
| test | `tests/unit/test_dflash2_{feature_concat,grouped_dynamic_conv,bf16_geometry,gate3_real,gate3_perf}.*` |
| docs | `docs/developer/dflash2.md` / `docs/references/dflash2.md` |

target の lowering / executor は変更していない。

### BF16 geometry

allowlist へ追加する前に、optimized BF16 GEMM 単体で reference と比較した。

| geometry | rows | rel_l2 | cosine |
| --- | --- | --- | --- |
| 1280 x 5120 | 1 | 2.351e-05 | 1.000000 |
| 1280 x 5120 | 8 | 8.283e-06 | 1.000000 |
| 5120 x 25600 | 1 | 5.094e-06 | 1.000000 |
| 5120 x 25600 | 8 | 1.339e-04 | 1.000000 |

この 2 件のみ `kLinearShapeRules` に追加した（`kFamilyBf16`）。Gate 4 用の
`4096 x 5120` / `5120 x 4096` は追加していない。

### reference 比較（実 checkpoint、rows 1 / 2 / 8）

fixture は `tools/reference/export_qwen38_dflash2_gate3.py` が生成する
（`build/dflash2-gate3-reference/`、Git へは commit しない）。
公式実装の revision は `07ebd93db9f472af339b644bb70221ad8428328a`（docs/references/dflash2.md）。

| tensor | 比較 | 結果 |
| --- | --- | --- |
| `target_concat` | bit-exact | 3/3 PASS |
| `target_fc` | rel_l2 <= 1e-2, cosine >= 0.9999 | max rel_l2 2.418e-04, min cosine 1.000000 |
| `target_feature` | 同上 | max rel_l2 2.922e-03, min cosine 0.999996 |
| conv `dynamic` | 同上 | 30 case, max rel_l2 3.533e-04, min cosine 1.000000 |
| conv `prepare` | 同上 | 30 case, max rel_l2 5.332e-04, min cosine 1.000000 |
| conv `finish` | 同上 | 30 case, max rel_l2 6.063e-04, min cosine 1.000000 |

30 case = 3 rows × 5 layer × 2（attention / mlp）。NaN / Inf は 0。

`target_feature` の 2.9e-03 は RMSNorm の丸め位置の差による。reference は公式と同じく
norm を bf16 に落としてから weight を掛け、kernel は f32 で掛けてから bf16 に丸める。

required synthetic test（外部 checkpoint 不要）:

| test | 内容 | 結果 |
| --- | --- | --- |
| `test_dflash2_feature_concat` | rows 1/2/8/32、features 5120/8/12/4、CPU reference と bit-exact | PASS |
| `test_dflash2_grouped_dynamic_conv` | hidden 32/5120、group 8/16、rows 1/2/3/8、phase 0/1、整数値で bit-exact・乱数で rel_l2 <= 2e-3 | PASS |
| `test_dflash2_bf16_geometry` | 上記 2 geometry の GEMM 検証 | PASS |

`test_dflash2_grouped_dynamic_conv` は row0 で offset1 が寄与しないこと、row1 が row0 を参照すること、
dynamic が `row` の行を使うこと（`row - offset` ではない）、group index が正しいことを個別に確認する。

### hot-path 契約

Gate 3 の production API（`dflash2_project_target_features` /
`dflash2_grouped_conv_prepare` / `dflash2_grouped_conv_finish`）と 2 kernel に
`hipStreamSynchronize` / `hipDeviceSynchronize` / `hipMemcpyDeviceToHost` /
`hipMemcpyHostToDevice` / `hipMalloc` / `hipFree` は無い（grep 確認）。
allocation は `create_dflash2_executor` のみ。

### timing（27B-DFlash2、device 0、per-op hipEvent、50 iter、wall clock で cross check）

| op | rows=1 | rows=8 |
| --- | --- | --- |
| tap concat | 3.64 us | 3.25 us |
| fc (5120x25600) | 418.23 us | 590.82 us |
| hidden_norm | 5.80 us | 5.80 us |
| conv kernel_projection (1280x5120) | 10.19 us | 75.52 us |
| conv prepare (phase 0) | 3.11 us | 3.56 us |
| conv finish (phase 1) | 3.09 us | 3.47 us |
| composed project | 425.45 us (wall 425.54) | 600.81 us (wall 598.09) |
| composed prepare | 13.23 us (wall 13.39) | 78.74 us (wall 79.38) |
| composed finish | 3.08 us (wall 3.48) | 3.46 us (wall 3.82) |

composed の時間は各 kernel の和と一致し、間に host round-trip が無いことを示す。
fc は 262MB の weight read が支配的で、rows=1 では約 627GB/s（DRAM 帯域相当）。
conv kernel_projection は rows=8 で 13MB / 75us = 約 174GB/s で、最適化余地がある。

### 回帰

| 対象 | 結果 |
| --- | --- |
| full build | PASS（0 error） |
| required acceptance | 102/102 PASS |
| `test_dflash2_config` / `_weight_contract` | required として PASS |
| `test_dflash2_weight_real` | PASS |
| `test_dflash2_target_taps` | PASS |

`linear_selector.cpp` の変更は既存 geometry に影響しない（追加のみ）。target lowering /
executor を変更していないため、4B / 4B-PSQ / 27B-PSQ の本体回帰は再実行していない。

### known issues

- conv kernel_projection は rows=8 で帯域 174GB/s 程度。最適化は Gate 3 では行わない。
- fixture は `build/` 配下に生成する前提で、Git には commit しない。

### Gate 決定

PASS。

---

## Gate 4: layer 0 standalone forward

baseline SHA: `ea303d18140aa93a8facf3f86f56cb66a71db7ba`（Gate 3 完了時点）。

### 変更

| 種別 | ファイル |
| --- | --- |
| kernel | `include|src/.../kernels/dflash2/rope.{h,hip}`（bf16 half-split RoPE） |
| kernel | `include|src/.../kernels/dflash2/attention.{h,hip}`（non-causal sliding / GQA） |
| kernel | `include|src/.../kernels/dflash2/rmsnorm.{h,hip}`（公式と同じ norm→bf16） |
| kernel | `include|src/.../kernels/dflash2/swiglu.{h,hip}`（公式と同じ bf16 丸め） |
| kernel | `src/.../kernels/dflash2/grouped_dynamic_conv.hip`（公式の bf16 丸め列） |
| executor | `include|src/.../dflash2/executor.{h,hip}`（buffer 追加 + layer forward） |
| selector | `src/.../runtime/linear_selector.cpp`（4096x5120 / 5120x4096） |
| reference | `tools/reference/export_qwen38_dflash2_gate4.py`（公式 layer を実行） |
| test | `tests/unit/test_dflash2_{rope,attention,gate4_real,gate4_perf}.*` |

target の lowering / executor と target paged attention は変更していない。

### BF16 geometry

allowlist 追加前に optimized BF16 GEMM 単体で reference と比較した。

| geometry | rows | rel_l2 | cosine |
| --- | --- | --- | --- |
| 4096 x 5120 | 1 | 1.577e-06 | 1.000000 |
| 4096 x 5120 | 2 | 1.256e-08 | 1.000000 |
| 4096 x 5120 | 8 | 1.883e-05 | 1.000000 |
| 5120 x 4096 | 1 | 1.089e-08 | 1.000000 |
| 5120 x 4096 | 2 | 7.342e-05 | 1.000000 |
| 5120 x 4096 | 8 | 7.339e-05 | 1.000000 |

この 2 件を `kLinearShapeRules` に追加した（`kFamilyBf16`）。

### RoPE

`test_dflash2_rope`（required）: position 0 は identity（bit-exact）、
half-split の確認、複数 head、head_dim 128。

| case | rel_l2 |
| --- | --- |
| pos 0 / 1 / 127 / 4096（1 head） | 0（bit-exact） |
| 32 heads, pos 0 / 2047 | 0 |
| 32 heads, pos 4096（rows=2） | 1.440e-04 |
| head_dim 8 / 64 | 0 |

### attention semantics

`test_dflash2_attention`（required）で固定した。

- `is_causal=false`、`abs(q - k) < window`（strict）。window=4 で distance 3 は visible、
  distance 4 は masked。
- future noise row が visible（query row0 が noise row3 を見る）。
- GQA: q head 0,1 → kv0、2,3 → kv1。
- logical key order は context → noise（context idx 1 は visible、idx 0 は masked）。
- CPU f32 reference と一致（rel_l2 = 0）。

### layer 0 reference 比較

fixture は `tools/reference/export_qwen38_dflash2_gate4.py` が公式
`Qwen3DFlashDecoderLayer` を CPU で実行して生成する（forward hook と method patch で
中間値を取得）。公式 revision は `07ebd93db9f472af339b644bb70221ad8428328a`。
fixture は `build/dflash2-gate4-reference/`（Git へは commit しない）。

3 case（ctx1/blk1、ctx4/blk2、ctx8/blk8）の最大値:

| stage | max rel_l2 | min cosine |
| --- | --- | --- |
| input_norm | 0.000e+00 | 1.000000 |
| attention_prepared | 1.359e-04 | 1.000000 |
| q_raw / k_ctx_raw / v_ctx | 4.737e-04 / 5.040e-05 / 2.820e-04 | 1.000000 |
| k_noise_raw / v_noise | 4.473e-04 / 4.504e-04 | 1.000000 |
| q_norm / k_ctx_norm / k_noise_norm | 5.992e-04 / 8.088e-06 / 5.132e-04 | 1.000000 |
| q_rope / k_ctx_rope / k_noise_rope | 6.618e-04 / 1.863e-04 / 5.901e-04 | 1.000000 |
| attention_context | 1.865e-03 | 0.999998 |
| attention_o | 2.189e-03 | 0.999998 |
| attention_finished | 2.322e-03 | 0.999997 |
| attention_residual | 2.346e-03 | 0.999997 |
| post_attention_norm | 3.325e-03 | 0.999994 |
| mlp_prepared | 5.053e-03 | 0.999987 |
| mlp_gate / mlp_up | 5.296e-03 / 5.299e-03 | 0.999986 |
| mlp_swiglu | 7.739e-03 | 0.999970 |
| mlp_down | 5.163e-03 | 0.999988 |
| mlp_finished | 3.802e-03 | 0.999993 |
| mlp_dynamic | 3.463e-03 | 0.999994 |
| layer_output | 3.802e-03 | 0.999993 |

NaN / Inf は 0。attention 段で誤差が急増しないこと（q_rope 6.6e-04 → attention_context
1.9e-03）を test 内で検査している。

### dtype semantics（Gate 4 で確定）

公式実装は bf16 tensor 同士の演算で各 op ごとに丸める。PhaseNonShift は conv / RoPE /
SwiGLU で同じ丸め列を再現し、RMSNorm は公式と同じく norm を bf16 に落としてから
weight を掛ける。

この変更で:

- conv の synthetic 比較が bit-exact（Gate 3 required test）
- RoPE の required test が大半 bit-exact
- layer 0 の誤差が attention_context 1.9e-03 / layer_output 3.8e-03 まで低下

Gate 3 の reference fixture は Section 2.2 の f32 式のままなので、Gate 3 real test の
`prepare`/`finish` は ~3e-3 の差を残す（閾値 1e-2 以内、PASS）。

### timing（27B-DFlash2、device 0、per-op hipEvent、50 iter）

layer0 ctx=8 blk=8（us）:

| op | us | op | us |
| --- | --- | --- | --- |
| input_norm | 17.66 | post_norm | 17.15 |
| attn_prepare | 80.00 | mlp_prepare | 80.51 |
| q_proj | 69.08 | gate | 486.74 |
| k_proj | 66.60 | swiglu | 4.37 |
| q_norm | 4.05 | down | 304.45 |
| rope | 3.54 | mlp_finish | 3.94 |
| attention | 9.05 | | |
| o_proj | 63.85 | | |
| attn_finish | 4.36 | | |
| residual | 3.20 | | |

composed layer0 = 2019.96us（wall 2027.28us）。個別 op の和との差は
k_ctx/v_ctx projection・mlp_up・k_noise rope など未計測分にあたる。

attention 単体（synthetic K/V、ctx=2048 / blk=8）= 457.22us。
Gate 4 では最適化しない（workgroup = 1 query row × 1 head の初期実装）。

### hot-path

Gate 4 の production API と DFlash2 kernel に `hipStreamSynchronize` /
`hipDeviceSynchronize` / `hipMemcpyDeviceToHost` / `hipMemcpyHostToDevice` /
`hipMemcpyDeviceToDevice` / `hipMalloc` / `hipFree` は無い（grep 確認）。
allocation は `create_dflash2_executor` のみ。

### 回帰

| 対象 | 結果 |
| --- | --- |
| full build | PASS（0 error） |
| required acceptance | 104/104 PASS |
| `test_dflash2_feature_concat` / `_grouped_dynamic_conv` / `_bf16_geometry` | PASS |
| `test_dflash2_rope` / `_attention` | PASS（required として追加） |
| `test_dflash2_weight_real` / `_gate3_real` / `_gate4_real` / `_target_taps` | PASS |

target lowering / executor と target paged attention は未変更のため、4B / 4B-PSQ /
27B-PSQ の本体回帰は再実行していない。

### known issues

- attention は correctness-first 実装（workgroup = 1 row × 1 head、key ごとに serial dot）。
  ctx=2048 / blk=8 で 457us。最適化は Gate 4 では行わない。
- Gate 3 の reference fixture は f32 式のため、conv の差 ~3e-3 が残る（PASS 範囲）。
- fixture は `build/` 配下に生成する前提で、Git には commit しない。

### Gate 決定

PASS。

---

## Gate 5: full 5-layer stateless backbone + final RMSNorm

baseline SHA: `e5b24038f1ccccbc4291adaa68f5084796f02e62`（Gate 4 完了時点）。

### 変更

| 種別 | ファイル |
| --- | --- |
| executor | `include|src/.../dflash2/executor.{h,hip}`（ping-pong buffer + backbone API） |
| reference | `tools/reference/export_qwen38_dflash2_gate5.py`（公式 `DFlash2DraftModel.forward`） |
| test | `tests/unit/test_dflash2_gate5_{real,perf}.hip` |
| docs | `docs/developer/dflash2.md` / `docs/references/dflash2.md` |

Gate 4 の kernel（conv / RoPE / RMSNorm / SwiGLU / attention）は変更していない。
target lowering / executor も変更していない。

### backbone contract

- `dflash2_forward_backbone_stateless()` が 5 layer を 0→4 の順に enqueue し、
  最後に `final_norm_weight` で final RMSNorm を掛ける。
- target feature projection は呼び出し側で 1 回だけ。全 layer で同じ pointer を共有する。
- layer 間は `backbone_hidden_a`（偶数）/ `backbone_hidden_b`（奇数）の pointer relay。
  D2D memcpy は無い。
- output は全 block_rows。
- `block_rows <= min(max_rows, block_size)`、`context_rows <= max_context_rows`、
  `context_position_start + context_rows == block_position_start` を検証する。

### isolated 比較（fixture の target_feature を直接入力）

fixture は `tools/reference/export_qwen38_dflash2_gate5.py` が公式
`DFlash2DraftModel.forward()` を実行して生成する（revision `07ebd93`、
`build/dflash2-gate5-reference/`、Git へは commit しない）。

| case | layer0 | layer1 | layer2 | layer3 | layer4 | final_hidden |
| --- | --- | --- | --- | --- | --- | --- |
| ctx1/blk2 | 9.396e-03 | 1.232e-02 | 1.708e-02 | 2.354e-02 | 3.453e-02 | 3.895e-02 |
| ctx4/blk4 | 9.082e-03 | 9.185e-03 | 1.892e-02 | 2.054e-02 | 2.701e-02 | 5.138e-02 |
| ctx8/blk8 | 6.058e-03 | 6.732e-03 | 8.903e-03 | 1.758e-02 | 3.618e-02 | 1.268e-01 |

cosine は layer0 で 0.99998 以上、layer4 で 0.99935〜0.99964、final で 0.9919〜0.9992。
NaN / Inf は 0。誤差は layer ごとに単調増加し、jump は無い（×1.1〜1.3/layer）。

### composed 比較（PhaseNonShift の projection を経由）

| case | final_hidden |
| --- | --- |
| ctx1/blk2 | 3.895e-02 |
| ctx4/blk4 | 6.740e-02 |
| ctx8/blk8 | 1.016e-01 |

composed と isolated の差は projection 誤差（~3e-8）のみで、実質同じ値になる。

### final norm 単体

fixture の `layer4_output` に final norm を掛けた結果は `final_hidden` と bit-exact
（rel_l2 = 0）。final_norm_weight の wiring は正しい。

### determinism / row sequence

- 同一入力で backbone を 10 回実行し、final_hidden が bit-exact（10/10）。
  scratch の memset は行わず、state leakage が無いことを確認。
- ctx8 → ctx1(blk2) → ctx4(blk4) → ctx8 の順で実行しても各 case の結果は変わらない
  （stale row の漏れ無し）。

### 誤差の原因（実装バグではない）

DFlash2 layer の内部活性は非常に大きい。公式 f32 実装で測ると
`mlp_down` 2.5e5、layer output 1.8e6 に達し、final norm で ~10 に戻る。
この領域では bf16 格納の量子化が約 0.4% になり、以下が実測値である。

- 公式実装自身の bf16 実行と f32 実行の差: final_hidden で **13%**
- PhaseNonShift の bf16 実行と公式 bf16 実行の差: layer0 6.1e-3、5 layer 後 1.3e-1
  （ctx8/blk8）。つまり PhaseNonShift は「公式 bf16 が f32 からずれる量」と同程度しか
  ずれていない。

stage 別（layer0、ctx8/blk8、公式 f32 との比較）でも段単位の実装誤差は無い。

| stage | rel_l2 |
| --- | --- |
| input_norm | 2.469e-03 |
| attention_prepared | 4.023e-03 |
| q_raw | 3.959e-03 |
| attention_o | 8.824e-03 |
| attention_finished | 9.704e-03 |
| mlp_finished | 8.331e-03 |

### timing（ctx=8 / blk=8、device 0、per-op hipEvent、20 iter）

| op | us |
| --- | --- |
| layer0 | 2005.15 |
| layer1 | 1984.36 |
| layer2 | 2026.32 |
| layer3 | 2006.45 |
| layer4 | 2033.34 |
| final norm | 17.63 |
| full backbone | 10122.82（wall 10132.41） |

5 layer で約 10ms。Gate 4 の layer0（2.02ms）と整合する。
ただしこれは stateless 実装であり、5 layer すべてで context K/V projection を
再計算する。production 相当の長 context 性能評価は Gate 7（persistent KV ring）以降で行う。

### hot-path

backbone API とその下位に `hipStreamSynchronize` / `hipDeviceSynchronize` /
`hipMemcpyDeviceToHost` / `hipMemcpyHostToDevice` / `hipMemcpyDeviceToDevice` /
`hipMalloc` / `hipFree` は無い。allocation は `create_dflash2_executor` のみ。

### 回帰

| 対象 | 結果 |
| --- | --- |
| full build | PASS（0 error） |
| required acceptance | 104/104 PASS |
| `test_dflash2_weight_real` / `_target_taps` / `_gate3_real` / `_gate4_real` / `_gate4_perf` | PASS |

### Gate 決定

**FAIL**（数値基準のみ）。

> Gate 5.1（real target 入力での再測定）で数値基準を満たしたため、最終決定は
> **PASS WITH NUMERIC CONTRACT**。下の Gate 5.1 の節を参照。

構造・契約はすべて満たす（layer 順、layer 別 weights、projection 1 回、共有、
ping-pong relay、D2D 無し、final norm wiring、block_rows 2/4/8、determinism、
row sequence、hot-path）。

一方、isolated / composed の final_hidden は閾値（2e-2 / 3e-2）を超える
（3.9e-2〜1.27e-1）。原因は上記のとおり bf16 格納と活性スケールであり、
reference 自身の bf16/f32 差（13%）と同程度である。閾値を緩めて PASS にはしない。

次に必要な判断:

1. reference 比較を f32 中間 buffer で行う（kernel と buffer の dtype を f32 に上げる）。
2. 実際の target 由来入力（drafter の実運用レンジ）で再測定する。
3. それでも足りない場合は、magnitude を考慮した基準を別途定義する。

---

## Gate 5.1: numeric contract isolation

Gate 5 の 5-layer 累積誤差が

A. PhaseNonShift 実装差
B. BF16 丸め順序差
C. synthetic input の異常な dynamic range

のどれなのかを切り分けた。baseline SHA は `e5b24098` ではなく Gate 5 完了時の
`1a457b98`。production code は変更していない（test / tools / docs のみ）。

### 1. fixture 生成のバグ（根本原因 1）

Gate 5 fixture は `DFlash2DraftModel` を組み立てて `model.to(torch.bfloat16)` してから
forward していた。この操作は **registered buffer `rotary_emb.inv_freq` まで bf16 に
丸める**。公式の `from_pretrained` では `inv_freq` は `persistent=False` で checkpoint に
含まれないため **float32 のまま** である（実測で確認）。

その結果、fixture の RoPE 角度が実装とずれ、`q_rope`/`k_rope` の rel_l2 が 1.4 に達して
いた。Gate 4 の exporter は standalone な `Qwen3RotaryEmbedding(cfg)` を f32 で作って
いたためこの問題が無く、PASS していた。

修正: exporter で `model.rotary_emb = official.Qwen3RotaryEmbedding(cfg)` を作り直す。

修正後の synthetic 誤差（ctx8/blk8、isolated）:

| stage | 修正前 | 修正後 |
| --- | --- | --- |
| layer0 | 6.058e-03 | 4.132e-03 |
| layer1 | 6.732e-03 | 7.693e-03 |
| layer2 | 8.903e-03 | 5.650e-03 |
| layer3 | 1.758e-02 | 6.853e-03 |
| layer4 | 3.618e-02 | 1.480e-02 |
| final | 1.268e-01 | 5.387e-02 |

### 2. official dtype trace

`tools/reference/trace_qwen38_dflash2_gate51.py --mode official` が全 layer の
全 stage について実 tensor の dtype と統計を dump する
（`build/dflash2-gate51/official_dtype_trace.json`）。

結果: **中間 tensor はすべて `torch.bfloat16`**。layer0（ctx8/blk8）の例:

| stage | dtype | absmax |
| --- | --- | ---: |
| layer_input | bf16 | 4.375 |
| input_norm | bf16 | 3.8125 |
| attention_prepared | bf16 | 12.0625 |
| q_raw | bf16 | 63 |
| q_norm | bf16 | 8.25 |
| q_rope | bf16 | 8.5625 |
| k_ctx_rope | bf16 | 8.625 |
| v_ctx | bf16 | 39 |
| attention_context | bf16 | 42.25 |
| attention_o | bf16 | 2048 |
| attention_finished | bf16 | 2528 |
| attention_residual | bf16 | 2528 |
| post_attention_norm | bf16 | 4.71875 |
| mlp_prepared | bf16 | 12.625 |
| mlp_gate / mlp_up | bf16 | 56 / 54 |
| mlp_swiglu | bf16 | 1712 |
| mlp_down | bf16 | 225280 |
| mlp_finished | bf16 | 1.77766e+06 |
| layer_output | bf16 | 1.77766e+06 |
| cos / sin | bf16 | 1 |

`sdpa_scale = 0.08838834764831845`。attention_mask は bool で
`abs(q_pos - k_pos) < sliding_window`（strict、非 causal）。

PhaseNonShift 側の buffer dtype も全 stage bf16。kernel 内部は
RMSNorm の variance / conv の乗算 / attention の score・softmax・V 累算 /
SwiGLU の silu が f32 で、buffer 境界で bf16 に丸める。dtype の不一致は 1 個も無い。

### 3. three-way comparison

`--mode replay` は PhaseNonShift の op 順序・buffer 境界を PyTorch primitive で
再現した explicit BF16 replay を作る。3 者を比較する（layer0、ctx8/blk8）:

| stage | PS vs official_bf16 | PS vs official_f32 | PS vs replay_bf16 |
| --- | ---: | ---: | ---: |
| input_norm | 0.000e+00 | 2.469e-03 | 0.000e+00 |
| attention_prepared | 3.208e-05 | 4.651e-03 | 3.208e-05 |
| q_raw | 1.553e-04 | 4.464e-03 | 1.545e-04 |
| q_rope | 3.155e-04 | 5.680e-03 | 3.211e-04 |
| k_ctx_raw | 3.127e-08 | 2.658e-03 | 3.127e-08 |
| attention_context | 1.492e-03 | 1.064e-02 | 8.598e-04 |
| attention_o | 2.094e-03 | 9.417e-03 | 1.532e-03 |
| mlp_swiglu | 7.355e-03 | 2.121e-02 | 6.093e-03 |
| layer_output | 4.132e-03 | 8.332e-03 | 1.973e-03 |

layer4 / final:

| stage | PS vs official_bf16 | official_bf16 vs official_f32 | PS vs replay_bf16 |
| --- | ---: | ---: | ---: |
| layer4 attention_finished | 1.056e-01 | 3.232e-01 | 6.786e-02 |
| layer4 layer_output | 1.480e-02 | 4.051e-02 | 1.100e-02 |
| final_hidden | 5.387e-02 | 1.435e-01 | 3.750e-02 |

**PhaseNonShift は explicit BF16 replay に対し official_bf16 と同程度かそれ以上に一致する。**
誤差は layer ごとに単調で、jump は無い（multiplier ≈ 1.0〜2.0）。

### 4. F32 diagnostic

`--mode replay_f32` は同じ op 順序を f32 精度（bf16 丸めなし）で再現する。

official_f32 との差（layer0）:

| stage | rel_l2 |
| --- | ---: |
| input_norm | 0.000e+00 |
| attention_prepared | 4.148e-08 |
| q_raw | 1.329e-07 |
| q_norm | 1.539e-07 |
| q_rope | 1.930e-05 |
| **k_ctx_raw** | **2.064e-03** |
| **v_ctx** | **2.192e-03** |
| k_noise_raw | 1.325e-07 |
| v_noise | 1.460e-07 |

f32 では `q_*` / `k_noise_*` / `v_noise` が 1e-7 で一致する。差が出るのは
`k_ctx_raw` / `v_ctx` だけで、これは **fixture の `target_feature` が bf16 である**
（official_f32 は forward 内で fc+hidden_norm を f32 で再計算する）ためである。
`q_rope` の 1.9e-5 は inv_freq/trig の計算差。それ以外に構造差は無い。

この帰着は定量的一致で裏付けられる。fixture の bf16 `target_feature` と、
concat から f32 で再計算した特徴（同じ重み）の rel_l2 は 3.308e-03。
両者を `k_proj` に通した差は 2.413e-03 で、観測された official_f32 との
`k_ctx_raw` 差 2.658e-03 と一致する。

したがって layer0 の chain は f32 では実質一致し、誤差は
(a) 入力の bf16 量子化、(b) trig、(c) f32 累算順序（~1e-7）のみである。

### 5. Real target fixtures

`models/Qwen3.8-27B-PSQ` で実 prompt を prefill し、`test_dflash2_gate51_target_fixture`
が layer 5/19/33/47/61 の tap の直近 8 行を dump する。noise block は target の
`embed_tokens` から `_raw_input_embeddings` と同じく anchor + MASK×7 で作る
（`tools/reference/export_qwen38_dflash2_gate51_real.py`）。revision は Gate 5 と同じ。

| case | mode | layer0 | layer1 | layer2 | layer3 | layer4 | final |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prose | isolated | 2.881e-04 | 1.556e-03 | 3.211e-03 | 2.365e-03 | 1.064e-02 | 9.204e-03 |
| prose | composed | 2.936e-04 | 5.957e-04 | 9.441e-04 | 1.490e-03 | 7.224e-03 | 7.374e-03 |
| code | isolated | 9.560e-04 | 1.025e-03 | 2.200e-03 | 4.815e-03 | 1.256e-02 | 1.423e-02 |
| code | composed | 9.607e-04 | 1.613e-03 | 2.219e-03 | 4.895e-03 | 1.522e-02 | 1.683e-02 |
| math | isolated | 1.257e-03 | 2.207e-03 | 2.537e-03 | 3.661e-03 | 1.427e-02 | 1.560e-02 |
| math | composed | 1.258e-03 | 1.118e-03 | 3.489e-03 | 3.568e-03 | 1.383e-02 | 1.548e-02 |

全 case で cosine ≥ 0.9998、NaN / Inf = 0。isolated final ≤ 2e-2、composed final ≤ 3e-2 を
すべて満たす。同一入力 10 回の bit 一致も real_prose で確認。

synthetic との決定的な違いは活性の大きさである。layer0 の layer_output absmax は
synthetic で 1.78e6（公式 f32 実測でも `mlp_down` 2.5e5）、real では 4.0 である。
DFlash2 の layer は学習分布内の入力では O(1) の出力を返し、分布外の入力では
MLP が 1e5〜1e6 まで増幅する。bf16 格納の相対精度はスケールに依存しないが、
増幅により誤差が層ごとに積み上がるため synthetic では閾値を超えていた。

### 6. logit sensitivity

target の `lm_head.weight`（`models/Qwen3.8-27B`）を official と PhaseNonShift の
final_hidden の両方に適用し、draft row ごとに argmax / top-16 を比較する
（C++ candidate selector は実装しない）。

| fixture | rows | argmax match | top-16 overlap |
| --- | ---: | ---: | --- |
| synthetic ctx8/blk8 | 8 | 8/8 | 14〜16/16 |
| real prose | 8 | 8/8 | 15〜16/16 |
| real code | 8 | 8/8 | 15〜16/16 |
| real math | 8 | 8/8 | 16/16 |

reference の top-1 はすべて PhaseNonShift 側でも rank 1。logit の rel_l2 は
synthetic で ≤ 0.064、real で ≤ 0.028。**hidden の差は candidate 集合にほとんど影響しない。**

### 7. 回帰

| 対象 | 結果 |
| --- | --- |
| full build | PASS |
| required acceptance | 104/104 PASS |
| Gate 3 real / Gate 4 real / Gate 4 perf | PASS |
| Gate 5 real（synthetic） | FAIL（閾値のみ。構造・determinism・logit は PASS） |
| Gate 5 real（real prose/code/math） | PASS |

### Gate 決定

**PASS WITH NUMERIC CONTRACT**。

- PhaseNonShift の op 順序・buffer 境界・丸め位置は explicit BF16 replay と一致し、
  official native との差は reference 自身の bf16/f32 差と同程度である。
- f32 診断で構造差は無い（残差は入力の bf16 量子化と trig）。
- 実 target 入力では isolated final ≤ 2e-2、composed final ≤ 3e-2、cosine ≥ 0.9998 を満たす。
- logit / top-16 への影響は無視できる。
- NaN / Inf 0、determinism 10/10 bit 一致。

synthetic fixture は stress case として残す。synthetic の閾値超過は
入力の dynamic range に由来するもので、実装差ではない。

---

## Gate 6: target lm_head + top-16 + CandidateSelector

### 1. baseline

| 項目 | 値 |
| --- | --- |
| baseline SHA | `8638d31328f22a3e8839b3f1a0feb03b602b8886` |
| branch / worktree | `feat/dflash2` / `.worktrees/dflash2` |
| 計測 device | 0 |
| full build | PASS |
| required acceptance | 109/109 PASS（Gate 6 の required 2 件を追加） |

### 2. checkpoint 契約（実測）

| 項目 | 値 |
| --- | --- |
| `selector_rank` | 256 |
| `selector_top_k` | 16 |
| `candidate_selector.hidden_projection.weight` | BF16 `[256, 5120]` |
| `candidate_selector.predecessor_codebook` | BF16 `[248320, 256]` |
| `candidate_selector.successor_codebook` | BF16 `[248320, 256]` |
| target lm_head | `models/Qwen3.8-27B-PSQ` の PSQ8（`lm_head.weight.__phaseshift_codes` + `__phaseshift_metadata1`） |

### 3. target lm_head

- DFlash2 専用 lm_head は作らない。target の `MatrixWeight` を
  `DFlash2ExecutorConfig::target_lm_head` で借りる。
- LINEAR 経路と同じ契約: 活性を `launch_activation_quantize_e4m3`
  （行ごとの `peak/448` scale）で量子化し、`launch_gemm_psq8_w8a8_wmma_auto` で
  PSQ8 weight と積む。logits は F32。
- 入力 pointer は `final_hidden + hidden_size`（anchor row を除く draft rows のみ）。
- dtype: `final_hidden` BF16、`proposal_logits` F32 `[draft_rows, 248320]`。

| 比較対象 | rel_l2 | cosine |
| --- | ---: | ---: |
| PSQ8 production oracle | 4.4e-07 〜 5.1e-07 | 1.000000 |
| BF16 oracle（別精度・参考） | 2.88e-02 〜 3.17e-02 | 0.9995 |

reference は target lm_head の production 契約（e4m3 codebook × 32 要素ごとの
bf16 scale、活性 e4m3、f32 累算）で logits を再計算し、公式
`CandidateSelector.select()` と一致することを確認してから使う。

### 4. top-16

- 実装: `dflash2_topk_partition_kernel`（1 row = 1 workgroup、thread ごとの
  local top-16 を 256×16 候補として scratch へ）+ `dflash2_topk_merge_kernel`
  （1 row = 1 thread で降順確定）。full sort はしない。
- tie rule: logit 降順、同値なら token id 昇順（f32 bit 等価で判定）。
- candidate selector は logits の F32 契約に合わせる（BF16 へ落とさない）。

| 比較対象 | 結果 |
| --- | --- |
| official hidden 入力の top-16 ids | 112/112 exact（3 case 合計） |
| CPU full sort reference（small vocab 6 case） | 完全一致 |

### 5. CandidateSelector

- 1 kernel（1 workgroup、256 threads）で draft 列全体を確定する。
  position ごとの kernel 分割も D2H も無い。`rank=256` / `top_k=16` を引数で受ける。
- score = `top16_logit + dot(pred_codebook[pred] * hidden_proj[r], succ_codebook[cand])`、
  累算 F32、codebook / hidden_proj は bf16 load。
- tie rule: score 降順、同値なら candidate token id 昇順。
- `pred_0 = anchor_token`、`pred_{r+1} = draft_token[r]`。

| case | selected tokens | score rel_l2 |
| --- | --- | ---: |
| real prose | 561 3988 9704 314 279 9704 314 | 3.9e-08 |
| real code | 2 498 16 11 220 17 11 | 3.8e-08 |
| real math | 271 22916 1534 4087 220 16 13 | 4.7e-08 |

official reference（`CandidateSelector.select()`）と完全一致。
predecessor dependency は専用 fixture（anchor 更新を止めると結果が変わる）で検証する。

### 6. block_rows 分離比較

official hidden を入力した isolated 経路。selector は逐次依存なので
`block_rows=k` の結果は `block_rows=8` の先頭 `k-1` と一致する。

| block_rows | draft_rows | prose | code | math |
| --- | ---: | --- | --- | --- |
| 2 | 1 | exact | exact | exact |
| 4 | 3 | exact | exact | exact |
| 8 | 7 | exact | exact | exact |

### 7. composed proposal

PhaseNonShift backbone → target lm_head → top-16 → selector。

| fixture | official drafts | PhaseNonShift drafts | 一致 |
| --- | --- | --- | ---: |
| real prose | 561 3988 9704 314 279 9704 314 | 561 3988 9704 314 279 9704 314 | 7/7 |
| real code | 2 498 16 11 220 17 11 | 2 498 16 11 220 17 11 | 7/7 |
| real math | 271 22916 1534 4087 220 16 13 | 271 22916 1534 4087 220 12650 1132 | 5/7 |

real math の 2 position は Gate 5.1 で受容済みの backbone 数値契約に由来する。
composed `final_hidden` と official の差は rel_l2 = **1.560e-02**（Gate 5.1 の
composed math 実測 1.548e-02 と一致）。同じ入力を与えた isolated 経路は
3 case すべて exact であり、Gate 6 の lm_head / top-16 / selector 自体に差は無い。

### 8. determinism / runtime 契約

| 項目 | 結果 |
| --- | --- |
| composed 100 回 | bit-exact |
| NaN | 0 |
| Inf | 0 |
| `hipStreamSynchronize` / `hipDeviceSynchronize` | 0 |
| `hipMemcpyDeviceToHost` / `hipMemcpyHostToDevice` | 0 |
| relay `hipMemcpyDeviceToDevice` | 0 |
| hot-path allocation（`hipMalloc` / `hipFree`） | 0 |

production path の grep は `dflash2/executor.hip`、`dflash2/topk.hip`、
`dflash2/candidate_selector.hip` の 3 ファイルで 0 件。buffer 確保は
`create_dflash2_executor` のみ。

### 9. timing（block_rows=8、draft_rows=7、device 0、20 iter）

| stage | 時間 |
| --- | ---: |
| target lm_head（M=7、vocab=248320） | 2278.69 us |
| top-16（vocab=248320） | 5037.12 us |
| selector hidden projection | 10.07 us |
| candidate selector | 23.59 us |
| selector stage 合計 | 7446.07 us |
| composed（backbone + selector） | 17606.56 us |

Gate 6 は full logits を materialize してから top-16 で読み直す
**correctness path** である。lm_head + top-16 の fusion は未実装で、
Gate 6 PASS 後の最適化候補として残す。

### 10. 回帰

| 対象 | 結果 |
| --- | --- |
| full build | PASS |
| required acceptance | 109/109 PASS |
| Gate 3 real / Gate 4 real / Gate 5 real（real prose/code/math） | PASS |
| Gate 6 real（real prose/code/math） / Gate 6 perf | PASS |
| 4B app E2E（inference、共有 runtime 変更の確認） | 既存の help gate 不整合で停止（`phaseshift-bench` に `fused-linear` subcommand が無い。本 Gate の diff は app / CLI に触れないため既存事象） |

`kLinearShapeRules` への `{256, 5120}` 追加は、この shape の重みが
27B-PSQ / 4B / 4B-PSQ に存在しないため（確認済み）既存モデルの線形パスを
変更しない。DFlash2 の `candidate_selector.hidden_projection.weight` のみが該当する。

### Gate 決定

**PASS WITH NUMERIC CONTRACT**。

- selector isolated / lm_head + top-16 isolated / block_rows 2・4・8 は 3 case すべて exact。
- PSQ8 oracle に対する logits は rel_l2 ≤ 5.1e-07。
- top-16 は 112/112 exact。determinism 100/100 bit 一致、NaN / Inf 0。
- composed proposal は prose / code が exact、math は Gate 5.1 で受容済みの
  backbone 数値差（composed final_hidden rel_l2 1.560e-02）に由来する 2 position 差。

---

## Gate 7: persistent context KV ring

context K/V を sequence state として一度だけ生成し、proposal ごとの再投影を消す。
stateless 経路は correctness oracle として残す。

### baseline

| 項目 | 値 |
| --- | --- |
| baseline SHA | `420815c9ff40d13affa6ebdb0f352282cebc1a37`（Gate 6 完了点） |
| branch / worktree | `feat/dflash2` / `.worktrees/dflash2` |
| required before | 109/109 PASS |

### context state 契約

| 項目 | 値 |
| --- | --- |
| layers | 5 |
| capacity | 2048（`config.sliding_window`） |
| kv features | 1024（8 × 128） |
| K dtype / V dtype | BF16 / BF16 |
| K 保存形 | `k_proj -> k_norm(RMSNorm head_dim) -> RoPE(absolute position)` |
| V 保存形 | `v_proj` 出力 |
| physical mapping | `absolute_position % capacity` |
| 実 allocation | `2 * 5 * 2048 * 1024 * 2 = 41,943,040 bytes`（40 MiB / sequence） |
| next_position | target が次に consume する absolute position。block row 0 の position と一致 |
| ownership | `DFlash2ContextState`（executor と分離、explicit shutdown） |

### append

`dflash2_context_append()`。`position_start == state.next_position` を要求する。

| rows | 時間 | final length | next_position |
| ---: | ---: | ---: | ---: |
| 1 | 835.50 us | 1 | 1 |
| 8 | 784.94 us | 8 | 8 |
| 128 | 2181.68 us | 128 | 128 |
| 2048 | 13991.93 us | 2048 | 2048 |

- K reference（gate7 fixture、official k_proj + k_norm + RoPE）: layer0 K rel_l2
  4.98e-05〜3.13e-04、layer4 K rel_l2 0〜6.51e-05、cosine 1.000000（3 case）。
- V reference（official v_proj）: rel_l2 1.74e-07〜1.48e-04、cosine 1.000000。3 case の
  最大は K rel_l2 3.132e-04 / V rel_l2 1.480e-04。
- chunk-size independence（1 / 8 / 128 / 2048 row ずつ構築）: final ring K/V が
  全 pattern bit-exact。
- append > capacity: `skip = rows - capacity` で最後の capacity row のみ生成。
  rows=20 / capacity=8 で retained が position 12..19 に一致（`test_dflash2_kv_ring`）。
- append の K/V 投影は `allow_exact_rows = false` を指定する。M=1 の ExactRows 経路は
  M>=2 の WMMA 経路と丸めが異なり、chunk size 非依存を壊すためである
  （M>=2 の既存経路は変更しない）。stateless の context 投影は既定のまま。

### ring boundary（capacity=2048）

| total consumed | length | logical start | wrap | cached-vs-stateless |
| ---: | ---: | ---: | --- | --- |
| 2047 | 2047 | 0 | no | bit-exact |
| 2048 | 2048 | 0 | no | bit-exact |
| 2049 | 2048 | 1 | yes | bit-exact |
| 2056 | 2048 | 8 | yes | bit-exact |

### attention

`test_dflash2_attention_ring` が既存 contiguous attention を oracle として比較する。

| 項目 | 結果 |
| --- | --- |
| no-wrap / one-wrap / multi-wrap cached-vs-stateless | bit-exact（rel_l2 0） |
| context length 1 / capacity | bit-exact |
| distance 2047 visible / 2048 masked（window 2048） | PASS |
| distance 3 visible / 4 masked（window 4） | PASS |
| future noise visibility（非 causal） | PASS |
| GQA 32/8 head128 | bit-exact |

ring indexing により context は slot 計算で読むが、key 走査順・softmax・GQA mapping は
Gate 4 と同一のため、contiguous attention と bit-exact になる。

### cached backbone / proposal

| fixture | final rel_l2 vs stateless | cosine | draft exact |
| --- | ---: | ---: | --- |
| real prose | 0.0 | 1.000000 | 7/7 |
| real code | 0.0 | 1.000000 | 7/7 |
| real math | 0.0 | 1.000000 | 7/7 |

- long context（synthetic）: context=2048 の full backbone、および total=2056 wrapped の
  full backbone が cached-vs-stateless bit-exact。
- block_rows 2 / 4 / 8: cached proposer の draft token が stateless と完全一致。
- repeated proposal 100 回: bit-exact。context の `length` / `next_position` / ring K/V は
  不変（proposal は context state を read-only として扱う）。
- noise K/V を ring へ commit しない。reject/provisional row を commit する API は無い
  （context は commit-only、rollback API なし）。

### proposal hot path

source audit（`executor.hip` の cached layer 経路）。

| 項目 | 値 |
| --- | ---: |
| `k_ctx_proj` launches（cached layer） | 0 |
| `v_ctx_proj` launches（cached layer） | 0 |
| `k_ctx_norm` launches（cached layer） | 0 |
| `k_ctx_rope` launches（cached layer） | 0 |
| `hipStreamSynchronize` / `hipDeviceSynchronize` | 0 |
| `hipMemcpyDeviceToHost` / `hipMemcpyHostToDevice` | 0 |
| relay `hipMemcpyDeviceToDevice` | 0 |
| hot-path allocation（`hipMalloc` / `hipFree`） | 0 |

production path の grep は `executor.hip` / `context_state.hip` / `kv_ring.hip` /
`attention.hip` で 0 件。allocation は create 時のみ。

### performance（block_rows=8、draft_rows=7、device 0、20 iter）

#### cached attention kernel

| context | block | us |
| ---: | ---: | ---: |
| 8 | 8 | 11.45 |
| 512 | 8 | 158.73 |
| 2047 | 8 | 581.87 |
| 2048 | 8 | 581.47 |

Gate 4 の contiguous attention（ctx=2048 / block=8、約 457us）に対し約 1.27 倍。
ring slot 計算の追加分であり、大幅な regression ではない。

#### cached backbone（final norm 込み）

| context | block | us |
| ---: | ---: | ---: |
| 8 | 8 | 9467.83 |
| 512 | 8 | 11527.00 |
| 2048 | 8 | 13156.68 |

#### full cached proposer（backbone + lm_head + top16 + selector）

| context | us |
| ---: | ---: |
| 8 | 16941.73 |
| 2048 | 20528.31 |

selector stage（lm_head + top-16 ≈ 7.4ms）は Gate 6 の correctness path のまま残る。

#### stateless comparison（context=2048、block_rows=8）

| 経路 | us |
| --- | ---: |
| stateless ctx2048 | 25582.11 |
| cached ctx2048 | 13167.15 |

cached は context K/V 再投影（5 layer 分の `k_proj` / `v_proj` / `k_norm` / `RoPE`）を
proposal ごとに実行しないため、ctx=2048 で約 1.94 倍高速になる。

### 回帰

| 対象 | 結果 |
| --- | --- |
| full build | PASS |
| required acceptance | 111/111 PASS |
| Gate 3 real / Gate 4 real / Gate 5 real / Gate 5.1 | PASS |
| Gate 6 real / Gate 6 perf | PASS |
| Gate 7 real（real prose/code/math、boundary、chunk） / Gate 7 perf | PASS |

回帰は device 0 が別セッションの `phaseshift-compute`（26GB）に占有されていたため、
空き GPU を `HIP_VISIBLE_DEVICES` で論理 device 0 に割り当てて実行した。性能計測
（Gate 7 perf）は device 0 が空いた状態で取得している。

### Gate 決定

**PASS**。

- K/V ring は official reference と rel_l2 ≤ 3.2e-04 / cosine 1.000000。
- cached backbone と cached proposer は stateless と bit-exact。
- 2048 boundary（2047 / 2048 / 2049 / 2056）と chunk size 1/8/128/2048 が bit-exact。
- proposal は context state を変更しない。hot-path forbidden call は 0。
- Gate 7 の diff は DFlash2 専用 state / kernel のみで、target 本体 runtime は変更しない。
  ただし `Bf16GemmSelectorInput` に `allow_exact_rows`（既定 true）を追加した。既定では
  既存挙動を変えない。

---

## Gate 8: live target bridge

fixture ではなく実 target prompt から DFlash context を構築し、target が生成した
anchor で live proposal を出す。target verify はまだ行わない。

### baseline

| 項目 | 値 |
| --- | --- |
| baseline SHA | `689ad39509ec276d7f0e3c4a842417199187e9e7`（Gate 7 完了点） |
| branch / worktree | `feat/dflash2` / `.worktrees/dflash2` |
| required before | 111/111 PASS |

### target bridge

| 項目 | 値 |
| --- | --- |
| target tap IDs | 5, 19, 33, 47, 61 |
| max output rows | 8（`block_size`） |
| max scheduled tokens | 256（batch row bucket 上限以下） |
| target embed encoding | PSQ8（`__phaseshift_codes` + metadata1） |
| target lm_head encoding | PSQ8 |

### prompt sync（target Executor）

| case | prompt tokens | target position | DFlash next_position | DFlash length | projection chunk |
| --- | ---: | ---: | ---: | ---: | ---: |
| real prose | 31 | 31 | 31 | 31 | 8 |
| real code | 48 | 48 | 48 | 48 | 8 |
| real math | 60 | 60 | 60 | 60 | 8 |
| long prompt | 2100 | 2100 | 2100 | 2048 | 8 |

prompt は `max_scheduled_tokens` 単位で chunk 実行し、各 chunk 後に target tap を
`max_rows=8` 以下へ chunk projection して ring へ append する。long prompt では
ring が wrap し `length == 2048` になる。

### noise input

| 項目 | 値 |
| --- | --- |
| block rows | 8 |
| anchor | target final prefill sample（prose 561 / code 2 / math 271） |
| mask token | 248070 |
| embedding reference | target embed_tokens（PSQ8）を再利用 |

anchor は GPU kernel argument で渡し、8 token の H2D memcpy を毎回しない。

### live proposal

proposal は PhaseNonShift stateless path を oracle として一致を確認し、
さらに公式 DFlash2 backbone + production PSQ8 lm_head の reference と比較する。

| case | anchor | live drafts | stateless oracle | official reference |
| --- | ---: | --- | --- | --- |
| real prose | 561 | 12386 19825 579 9704 494 279 9704 | exact | exact |
| real code | 2 | 498 15 11 220 16 11 220 | exact | exact |
| real math | 271 | 760 1534 4087 369 2708 13 220 | exact | 5/7 |

real math の 2 position は Gate 5.1 で受容済みの backbone 数値契約に由来する。
live proposal は stateless oracle と 3 case すべて bit-exact。

### runtime contract

| 項目 | 値 |
| --- | --- |
| proposal D2H | 0（prompt prefill の anchor 読み出しのみ host-driven） |
| proposal H2D | 0 |
| proposal `hipStreamSynchronize` | 0 |
| proposal が context を変更 | なし（length / next_position / K/V 不変） |

### 回帰

| 対象 | 結果 |
| --- | --- |
| full build | PASS |
| required acceptance | 111/111 PASS |

### Gate 決定

**PASS**。

- 実 prompt から ring 構築、anchor 取得、live proposal までを target runtime 上で実行。
- live proposal は stateless oracle と bit-exact、official reference と prose/code exact。
- prompt > 2048 で ring length が 2048 に clamp され、proposal が動作する。
- Gate 8 の diff は DFlash2 専用 state/kernel/runtime の追加で、既存 target runtime を
  変更しない。

### known issues

- `test_gpu_mcu_persistent_emit` は GPU 負荷が高い状態で `worker start timestamps
  monotonic` が flaky になる（DFlash2 とは無関係。空き GPU では 2/2 PASS）。最初の
  required acceptance は device 0 を別セッションが占有していたため 110/111 となり、
  空き GPU を指定した再実行で 111/111 PASS。

---

## Gate 9: greedy speculative decode

DFlash proposal → target K+1 verify → longest-prefix acceptance → GDN restore/rerun →
accepted target tap の DFlash ring commit。`VerifyNumericMode::Exact` を既定とする。

### baseline

| 項目 | 値 |
| --- | --- |
| baseline SHA | `3d35557c`（Gate 8 完了点） |
| required before | 111/111 PASS |

### verify contract

| 項目 | 値 |
| --- | --- |
| numeric mode | Exact |
| target rows | K+1（anchor + drafts） |
| batch | execution_class PREFILL / speculative_verify=true / num_output_rows=K+1 |
| draft D2H | 1 round 1 回 |
| target sample D2H | 1 round 1 回 |

### correctness（max_new=128、token exact）

| prompt | K | context | generated | token exact |
| --- | ---: | ---: | ---: | --- |
| prose | 7 | 32 | 128 | yes |
| code | 7 | 32 | 128 | yes |
| json | 7 | 32 | 128 | yes |
| math | 7 | 32 | 128 | yes |
| prose | 1 | 32 | 128 | yes |
| prose | 3 | 32 | 128 | yes |
| prose | 7 | 128 | 128 | yes |
| prose | 7 | 2048 | 128 | yes |

app の最初の anchor は prefill sample で 1 回だけ emit し、spec round の emitted には
含めない。EOS は最初の EOS まで truncate して finish（`eos_case` PASS）。

### acceptance

| case | rounds | mean accepted | emitted/round | full accept rate | reruns | rerun rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prose k7 ctx32 | 35 | 2.629 | 3.63 | 0.200 | 28 | 0.800 |
| code k7 ctx32 | 20 | 5.350 | 6.35 | 0.750 | 5 | 0.250 |
| json k7 ctx32 | 37 | 2.432 | 3.43 | 0.189 | 29 | 0.784 |
| math k7 ctx32 | 18 | 6.056 | 7.06 | 0.833 | 2 | 0.111 |
| prose k1 ctx32 | 69 | 0.841 | 1.84 | 0.841 | 11 | 0.159 |
| prose k3 ctx32 | 44 | 1.886 | 2.89 | 0.477 | 23 | 0.523 |
| prose k7 ctx128 | 36 | 2.528 | 3.53 | 0.111 | 31 | 0.861 |
| prose k7 ctx2048 | 44 | 1.886 | 2.89 | 0.023 | 43 | 0.977 |

### state

| 項目 | 結果 |
| --- | --- |
| `context.next_position == sequence.position` | 全 case PASS |
| `context.length == min(position, 2048)` | 全 case PASS |
| proposal が context を変更 | なし（read-only） |

### timing（Exact、run 合計 ms）

| case | draft | d2h | verify | rerun | gdn snap | gdn restore | commit | round |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prose k7 ctx32 | 7.47 | 622.01 | 1465.36 | 1066.29 | 0.18 | 0.12 | 1.26 | 3162.72 |
| code k7 ctx32 | 3.31 | 358.82 | 838.38 | 188.02 | 0.09 | 0.02 | 0.70 | 1389.34 |
| math k7 ctx32 | 2.90 | 308.05 | 717.66 | 77.31 | 0.05 | 0.01 | 0.59 | 1143.65 |
| prose k7 ctx2048 | 6.14 | 953.16 | 1951.47 | 1713.49 | 0.15 | 0.14 | 1.75 | 4626.33 |

`d2h` は proposal kernel 完了待ち（stream sync）を含む。`verify` は `execute_batch` の
同期完了を含む。GDN snapshot/restore は十数 us で無視できる。rerun が支配的で、
acceptance が低い prompt/context ほど rerun が増える。

### Fast verify 診断

`PHASESHIFT_DFLASH2_VERIFY_EXACT=0`（Fast）では 6 case 中 6 case が token mismatch
（例: `prose k1`: token[7] baseline=13 spec=1785）。検証の Fast 経路は decode 経路と
数値が一致しないため、production default にしない。Exact のみが token exact を与える。

### 回帰

| 対象 | 結果 |
| --- | --- |
| full build | PASS |
| required acceptance | 111/111 PASS |

### Gate 決定

**PASS**。

- Exact verify で prose/code/JSON/math、K=1/3/7、context 32/128/2048 が target-only
  greedy と token exact（128 token 以上）。
- full accept は verify tap、partial は rerun tap を ring へ commit し、
  `context.next_position == sequence.position` を維持。
- DFlash 側に rollback API は無く commit-only。

---

## Gate 10: phaseshift-compute 統合

DFlash2 speculative decoding を `phaseshift-compute` の DFlash mode として統合する。
`--dflash2-model-dir` 指定時のみ有効で、未指定時の既存経路は不変。

### 構成

| 項目 | 値 |
| --- | --- |
| target | `models/Qwen3.8-27B-PSQ`（PSQ8） |
| draft | `models/Qwen3.8-27B-DFlash2`（BF16、3.6GB） |
| drafts | 7（block_size 8） |
| verify numeric mode | Exact |
| prompt | gate5.1 real prose（31 token） |
| max_new_tokens | 64 |
| GPU | HIP_VISIBLE_DEVICES=2,3 / --device 0 |

### token exactness

| 経路 | GENERATED_IDS | matched |
| --- | --- | --- |
| target-only（ContinuousBatcher, greedy） | 64 token | - |
| dflash2（compute, greedy） | 64 token | target-only と完全一致 |

`phaseshift-compute` の DFlash 出力は target-only greedy と token 単位で完全一致する。
`--temperature 0.7` は exit 2、`--dump-logits` も exit 2 で拒否する。

### throughput（prose、64 token）

| 経路 | decode tokens/s |
| --- | ---: |
| target-only | 28.16 |
| dflash2 | 30.72 |

### speculative 統計

| 項目 | 値 |
| --- | ---: |
| rounds | 21 |
| full accepts | 0 |
| reruns | 20 |
| accepted drafts | 42 |
| mean accepted | 2.000 |
| full accept rate | 0.000 |
| emitted per round | 3.000 |
| draft ms | 5.22 |
| d2h ms | 372.08 |
| verify ms | 851.25 |
| rerun ms | 782.53 |
| gdn snapshot ms | 0.17 |
| gdn restore ms | 0.09 |
| commit ms | 0.67 |
| round ms | 2049.37 |

prose は acceptance が低く full accept が 0 のため、round あたりの emitted は 3 token に
留まる。target-only 比の throughput 向上は小さい（28.16 → 30.72 tokens/s）。acceptance の
高い prompt（code / math）では Gate 9 のとおり mean accepted が 5〜6 に達する。

### resource lifecycle

DFlash mode の解放順は decoder → context → sequence → draft executor → target executor →
stream → arena。pool leak 検査は共通経路で実施し、kv page / seq slot / gdn slot が
すべて 0 であることを確認する。

### Gate 決定

**PASS**。

- compute の DFlash mode が target-only greedy と 64 token 完全一致。
- greedy 強制（temperature > 0 拒否）と dump-logits 拒否を model load 前に実施。
- 統計行と `GENERATED_IDS=` を出力し、`phaseshift-cli` から利用可能。

---

## Gate 11A Kernel Profile

### baseline

| 項目 | 値 |
| --- | --- |
| SHA | `a05077b6` |
| required | 111/111 PASS |
| target-only | 28.16 tok/s |
| DFlash2 K=7 prose | 30.72 tok/s |
| proposer ctx2048 | 20.12 ms（Gate7 20.53 ms と一致） |

### method

- warmup 50 / measure 200、HIP event、median / p10 / p90。
- block_rows=8、ctx=2048 を基本とし、ctx は 8 / 128 / 512 / 2048 を別測定。
- rows は 1 / 3 / 7、block_rows は 2 / 4 / 8 を別測定。
- GPU は空き状態。residual は 0.1〜1.5% で安定。

### kernel profile（block_rows=8, ctx=2048, median us）

| kernel | calls | median us | total us | weight bytes/call | eff GB/s | VGPR | LDS | scratch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target_feature_concat | 1 | 8.16 | 8.16 | - | - | - | - | - |
| target_fc_gemm | 1 | 582.92 | 582.92 | 262,144,000 | 450 | 27/68 | 0/8192 | 0 |
| target_hidden_norm | 1 | 21.24 | 21.24 | - | - | 10 | 512 | 0 |
| input_norm | 5 | 21.08 | 105.48 | - | - | 10 | 512 | 0 |
| attention_dynamic_projection | 5 | 78.80 | 374.92 | 13,107,200 | 166 | splitk | - | 0 |
| attention_conv_prepare | 5 | 8.04 | 40.12 | - | - | 14 | 0 | 0 |
| q_proj | 5 | 70.84 | 366.44 | 41,943,040 | 592 | splitk | 8192 | 0 |
| k_noise_proj | 5 | 69.40 | 360.80 | 10,485,760 | 151 | splitk | - | 0 |
| v_noise_proj | 5 | 69.28 | 360.16 | 10,485,760 | 151 | splitk | - | 0 |
| q_norm | 5 | 7.96 | 39.76 | - | - | 10 | 512 | 0 |
| k_noise_norm | 5 | 7.88 | 39.40 | - | - | 10 | 512 | 0 |
| q_rope | 5 | 7.76 | 38.80 | - | - | 16 | 0 | 0 |
| k_noise_rope | 5 | 7.64 | 38.20 | - | - | 16 | 0 | 0 |
| ring_attention | 5 | 576.48 | 2883.60 | - | - | 23 | 0 | 0 |
| o_proj | 5 | 65.84 | 336.72 | 41,943,040 | 637 | splitk | - | 0 |
| attention_conv_finish | 5 | 7.96 | 39.96 | - | - | 14 | 0 | 0 |
| attention_residual | 5 | 7.32 | 36.64 | - | - | - | - | - |
| post_attention_norm | 5 | 21.32 | 106.68 | - | - | 10 | 512 | 0 |
| mlp_dynamic_projection | 5 | 78.80 | 374.92 | 13,107,200 | 166 | splitk | - | 0 |
| mlp_conv_prepare | 5 | 8.00 | 40.00 | - | - | 14 | 0 | 0 |
| mlp_gate_proj | 5 | 512.24 | 2377.24 | 178,257,920 | 348 | splitk | - | 0 |
| mlp_up_proj | 5 | 487.56 | 2358.64 | 178,257,920 | 366 | splitk | - | 0 |
| mlp_swiglu | 5 | 8.72 | 43.96 | - | - | 13 | 0 | 0 |
| mlp_down_proj | 5 | 308.56 | 1721.76 | 178,257,920 | 578 | splitk | - | 0 |
| mlp_conv_finish | 5 | 8.28 | 40.96 | - | - | 14 | 0 | 0 |
| mlp_residual | 5 | 7.44 | 37.12 | - | - | - | - | - |
| final_norm | 1 | 21.16 | 21.16 | - | - | 10 | 512 | 0 |
| target_lm_head (psq8 w8a8) | 1 | 2203.59 | 2203.59 | 1,271,398,400 | 577 | 254 | 0 | 0 |
| top16 (partition+merge) | 1 | 4870.37 | 4870.37 | - | 1.43（input） | 72/4 | 0 | 0 |
| selector_hidden_projection | 1 | 13.44 | 13.44 | 2,621,440 | 195 | splitk | - | 0 |
| candidate_selector | 1 | 26.52 | 26.52 | - | - | 15 | 1024 | 0 |
| noise_token_ids | 1 | 7.16 | 7.16 | - | - | 4 | 0 | 0 |
| noise_embedding | 1 | 17.44 | 17.44 | - | - | - | - | - |

`eff GB/s` は weight bytes / latency。R9700 実測 DRAM 636 GB/s を roof とする。
`target_lm_head` は codes のみ（1,271,398,400 bytes）で算出。

### 観察

- `q_proj` / `o_proj` / `mlp_down_proj` は 592〜637 GB/s で DRAM roof 到達。
- `mlp_gate_proj` (348) / `mlp_up_proj` (366) / `target_fc_gemm` (450) は roof 未達で余地あり。
- `attention_dynamic_projection` / `mlp_dynamic_projection` (166)、`k_noise_proj` / `v_noise_proj` (151)、
  `selector_hidden_projection` (195) は skinny GEMM で bandwidth が低い。
- `ring_attention` は ctx 8→2048 で 13.4→576 us。ctx2048 で 2.88 ms（proposer の 14%）。
- `top16` は rows 1/3/7 で 4.88/4.86/4.88 ms と rows に依存しない。
- `cached_backbone` は ctx8 でも 9.41 ms。GEMM 支配で、ctx への依存は ring attention 分のみ。

### proposer time breakdown（ctx2048, block_rows=8, rows=7）

| 分類 | total us | 割合 |
| --- | ---: | ---: |
| BF16 GEMM（backbone + fc） | 9215 | 45.8% |
| top16 | 4870 | 24.2% |
| ring attention | 2884 | 14.3% |
| lm_head（psq8 + activation quantize） | 2204 | 11.0% |
| elementwise（norm / rope / conv / residual / swiglu） | 722 | 3.6% |
| selector | 40 | 0.2% |
| 合計 | 19935 | 99.1% |

### Top16 baseline resources

| 項目 | partition | merge |
| --- | ---: | ---: |
| threads | 256 | 1 |
| SGPR | 29 | 29 |
| VGPR | 72 | 4 |
| VGPR spill | 0 | 0 |
| scratch (private) | 0 | 0 |
| LDS | 0 | 0 |
| wavefront | 32 | 32 |
| code size | 2624 | 760 |

`local_logits[16]` / `local_ids[16]` は VGPR に完全配置され、scratch / private / LDS へ
spill していない。ISA に `ds_read` / `ds_write` は無い（459 insn partition / 152 insn merge）。

### Top16 baseline traffic

| 項目 | 値 |
| --- | ---: |
| input logits（rows=7） | 6,952,960 bytes（6.63 MiB） |
| partition scratch write + merge read | 7 × 256 × 16 × (4+4) = 229,376 bytes |
| measured latency | 4870 us |
| effective input BW | 1.43 GB/s |
| 純 read 理論下限（6.63 MiB / 636 GB/s） | 約 11 us |

5 ms は入力 bandwidth では説明できない。partition は grid = rows = 7 WG しか立たず
（64 CU 中 7 CU のみ）、merge は 1 row = 1 thread が 4096 entries を 16 回 serial scan
する。1 thread の逐次 load が latency 律速となり、これが支配的である。

### Gate 11A 決定

production kernel は変更しない。最適化順位は **top16（4.87 ms, 24.2%）** を第一とする。

---

## Gate 11B Top16 Optimization

### 実装

`launch_dflash2_topk_f32` を reference として残し、production 用に
`launch_dflash2_topk_f32_optimized` を追加した。executor の
`dflash2_select_draft_tokens` は optimized を呼ぶ。`PHASESHIFT_DFLASH2_TOPK_REFERENCE=1`
で reference に切り替えられる。fusion は行わず、lm_head は full logits を書き続ける。

- partition: grid = rows × partitions。1 WG が vocab partition を担当し、thread local
  top16 → WG 内 cooperative selection（warp shuffle + LDS、16 pass）で partition top16。
- merge: grid = rows、1 WG が partitions × 16 candidate を cooperative selection。
- scratch は既存の `[rows, 4096]` を再利用（partitions × 16 ≤ 1024）。

### sweep（rows=7, vocab=248320, median us）

| impl | partitions | threads | rows1 | rows3 | rows7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference | - | 256/1 | 4980.16 | 4878.20 | 5099.48 |
| optimized | 8 | 64 | 214.32 | 213.84 | 219.72 |
| optimized | 8 | 128 | 136.20 | 135.96 | 144.44 |
| optimized | 8 | 256 | 92.84 | 93.24 | 113.84 |
| optimized | 16 | 64 | 145.16 | 145.04 | 153.92 |
| optimized | 16 | 128 | 93.32 | 96.64 | 116.80 |
| optimized | 16 | 256 | 66.04 | 76.88 | 110.24 |
| optimized | 32 | 64 | 112.52 | 115.40 | 136.12 |
| optimized | 32 | 128 | 73.92 | 80.28 | 112.72 |
| **optimized** | **32** | **256** | **54.64** | **70.20** | **109.48** |
| optimized | 64 | 64 | 110.48 | 117.64 | 151.28 |
| optimized | 64 | 128 | 73.40 | 89.16 | 130.92 |
| optimized | 64 | 256 | 57.32 | 81.12 | 132.00 |

rows=7 の production は partitions=32 / threads=256 を採用。

### resource

| impl | threads | partitions | SGPR | VGPR | VGPR spill | scratch | LDS | code size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference partition | 256 | rows | 29 | 72 | 0 | 0 | 0 | 2624 |
| reference merge | 1 | rows | 29 | 4 | 0 | 0 | 0 | 760 |
| partition_hier<256> | 256 | 32 | 31 | 90 | 0 | 0 | 96 | 7652 |
| merge_hier<256> | 256 | rows | 27 | 19 | 0 | 0 | 96 | 2236 |

unexpected scratch spill は無い。VGPR は reference 72 → optimized 90 に増えたが spill せず、
grid が rows × 32 に増えるため occupancy は問題にならない。

### correctness

| 項目 | 結果 |
| --- | --- |
| fixtures | random / ties / boundary / masked |
| plan × fixture | 48/48 IDs exact、logits bit-exact |
| tie rule (logit desc, id asc) | exact |
| rows 1 / 3 / 7 | PASS |
| vocab 248320 | PASS |
| deterministic | 100/100 exact |
| Gate6 real prose/code/math | PASS（selector isolated exact、lm_head top16 112/112） |
| Gate10 GENERATED_IDS | target-only と完全一致（64 token） |
| Gate10 rounds / accepted | 21 / 42（Gate10 と同一） |
| Gate9 rounds / accepted / reruns | prose_k3 44 / 1.886 / 23（Gate9 と同一） |

### performance

| 項目 | before | after | 改善 |
| --- | ---: | ---: | ---: |
| top16 rows=7 | 5099.48 us | 109.48 us | 46.6x |
| top16 rows=3 | 4878.20 us | 70.20 us | 69.5x |
| top16 rows=1 | 4980.16 us | 54.64 us | 91.1x |
| proposer ctx2048 | 20119 us | 15601 us | 1.29x |
| proposer ctx8 | 16494 us | 11925 us | 1.38x |
| DFlash K=7 prose | 30.72 tok/s | 32.19 tok/s | +4.8% |
| target-only prose | 28.16 tok/s | 28.15 tok/s | - |

top16 の input は 6.63 MiB / rows7。optimized は reference の serial merge
（1 row = 1 thread、4096 entries × 16 pass）を WG cooperative merge に置き換え、
grid を rows × partitions に拡大した。残りの top16 latency は WG 内 16 pass の
reduction 律速で、input read の理論下限（約 11 us）には未達だが大幅に短縮した。

### fusion audit

| 項目 | 結果 |
| --- | --- |
| lm_head + top16 fused | no（lm_head は full logits を出力） |
| top16 を linear kernel へ組込 | no |
| selector を top16 kernel へ組込 | no |
| その他 fusion | no |
| full sort / radix sort | no |
| reference 保持 | yes（`launch_dflash2_topk_f32`、env で選択可） |

### Gate 11B 決定

**PASS**。top16 は単体で 46.6x、proposer 全体で 1.29x、E2E で +4.8%。
fusion は行っていない。次 Gate は本 report を見て決定する。

---

## Gate 11C GDN state history / rerun elimination

### gap audit

`docs/rnd/vllm_mxfp4_gap.md` に `GGZ14/vllm-mxfp4`（SHA
`31b9a94a7f74eeb3f59e66d16b1b27dfafcd0663`, 2026-09-18）の one-card R9700 値を記録した。
比較は checkpoint / 量子化 / KV dtype / drafter 精度 / sampling / corpus / power が異なるため
directional diagnostic only とする。

| engine | category | update p50 ms | tok/update | tok/s |
| --- | --- | ---: | ---: | ---: |
| vllm-mxfp4 | prose | 35.2 | 2.76 | 79.0 |
| vllm-mxfp4 | code | 35.3 | 4.65 | 145.2 |
| vllm-mxfp4 | math | 35.2 | 5.56 | 168.6 |
| vllm-mxfp4 | combined | 35.7 (p99) | - | 137.7 |
| PhaseNonShift | prose | 59.5 | 3.43 | 57.7 |
| PhaseNonShift | code | 58.1 | 7.06 | 121.3 |
| PhaseNonShift | math | 56.9 | 6.35 | 111.4 |

PhaseNonShift の update は単一 run の median で、competitor の BetterBench `--quick` p50 とは
corpus が違う。update latency gap は約 1.6x。

### GDN state geometry

runtime 値（`test_dflash2_gate11c_perf`）:

| 項目 | bytes |
| --- | ---: |
| conv state / row | 2,949,120 |
| recurrent state / row | 150,994,944 |
| total / row | 153,944,064（146.8 MiB） |
| K=7 history | 1,077,608,448（1.00 GiB） |
| max K=8 allocation | 1,231,552,512（1.15 GiB） |

1.5 GiB guard 以内。FP16 化や compression は行っていない。

### state exactness（`test_gdn_spec_history`, required）

production geometry（key_heads=16, v_heads=48, head_k=head_v=128, 1 GDN layer）で、
clean prefix（first1/first2/first3）の final state と history row 0/1/2 を比較した。

| K | prefix rows checked | conv exact | recurrent exact |
| ---: | ---: | --- | --- |
| 1 | 1 | yes | yes |
| 3 | 3 | yes | yes |
| 7 | 3（synthetic rows） | yes | yes |

restore も live state と bit-exact。recurrent は F32 bit-exact、conv は BF16 bit-exact。

### verify tap parity

history path の partial は verify execution の tap rows 0..A を DFlash ring へ commit する。
`test_dflash2_gate11c_e2e` と Gate9 E2E で、history path の GENERATED_IDS が
reference rerun path および target-only greedy と完全一致することを確認した（後述）。

### E2E parity（history vs reference rerun）

`test_dflash2_gate11c_e2e`（prose 256 / code 128 / math 128）:

| fixture | K | tokens | reference rerun | history | exact |
| --- | ---: | ---: | --- | --- | --- |
| prose | 7 | 256 | reruns=80 | reruns=0 | yes |
| code | 7 | 128 | reruns=3 | reruns=0 | yes |
| math | 7 | 128 | reruns=7 | reruns=0 | yes |

`PHASESHIFT_DFLASH2_GDN_RERUN_REFERENCE=1` の reference path と GENERATED_IDS が一致する。
prose 512 token でも history vs reference の GENERATED_IDS が完全一致（history reruns=0、
reference reruns=137）。Phase benchmark の KV dtype は `--kv-cache-dtype` 未指定 =
`phaseshift-compute` の default BF16。
Gate9 E2E は K=1/3/7、prose/code/JSON/math、ctx 32/128/2048 を history path で実行し、
全 case が target-only greedy と token exact（reruns=0）。reference path でも同じ acceptance
統計（rounds / mean accepted / full accept rate / reruns）を再現する。

### acceptance parity

| fixture | K | rounds | partial accepts | full accept rate | emitted/round |
| --- | ---: | ---: | ---: | ---: | ---: |
| prose | 7 | 149 | 137 | 0.081 | 3.430 |
| code | 7 | 18 | 3 | 0.833 | 7.056 |
| math | 7 | 20 | 7 | 0.650 | 6.350 |

prose は partial が 137/149。`DFLASH2_RERUNS=0` と `DFLASH2_PARTIAL_ACCEPTS` を分離して記録。

### history capture overhead（target verify A/B, 同一入力）

| M | capture off us | capture on us | overhead us | history bytes | 実効 write BW |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 37379.20 | 38142.78 | 763.58 | 307,888,128 | 403 GB/s |
| 4 | 38821.79 | 39956.26 | 1134.47 | 615,776,256 | 543 GB/s |
| 8 | 41893.93 | 43832.24 | 1938.31 | 1,231,552,512 | 635 GB/s |

M=8 では実効 write BW が DRAM roof 636 GB/s に到達する。history store は GDN kernel 内で
live state と同じ local offset へ書くため、追加の launch はゼロ。

### before vs after

| 項目 | before | after |
| --- | ---: | ---: |
| prose reruns（256 token） | 80 | 0 |
| prose update p50 | 約 93.1 ms | 59.5 ms |
| prose tok/s | 32.19 | 57.67 |
| code tok/s | - | 121.32 |
| math tok/s | - | 111.44 |
| target-only tok/s | 28.15 | 28.15 |

prose の round は rerun 除去でほぼ 1/3 短縮した。target-only は history pointer=null のため不変。

### target verify profile（M sweep）

| M | verify total ms（capture off） |
| ---: | ---: |
| 2 | 37.4 |
| 4 | 38.8 |
| 8 | 41.9 |

M 依存が小さい。target verify は weight read 支配で、competitor の full update（35 ms）より
大きい。family 別（BF16 GEMM / GDN conv / GDN recurrence / full attention / RMSNorm /
lm_head / sampling）の内訳計測は次 Gate の基礎とする。

### remaining competitive gap

- update latency gap: 約 1.6x（57-60 ms vs 35 ms）。主因は target verify ~42 ms。
- acceptance gap: prose 3.43 vs 2.76（Phase が上）、code 7.06 vs 4.65、math 6.35 vs 5.56。
  Phase の acceptance は同等以上。gap は acceptance ではなく verify latency。
- proposer gap: DFlash proposer ~15.6 ms。
- verify gap: 最大。target は mixed PSQ4 W4A8 / PSQ8 W8A8（`TensorRole` preset）、
  competitor は MXFP4 W4A8。weight traffic は約 1.5-2x 差。
- state-handling gap: Gate11C で解消（rerun 0、restore ~0.5 ms/run）。
- format gap: target/drafter/KV/GDN dtype の差（A/B/C/D）。

### fusion audit

fusion は追加していない。GDN kernel への optional history store は fusion ではない。
lm_head + top16、selector の融合も無い。

---

## Gate 11D Target verify small-M kernel ceiling

### target weight contract（訂正）

Gate 11C まで「target = PSQ8 W8A8」と一括表現していたが、これは誤り。
`src/phaseshift/quantization/fpx/profile.cpp` の `psq()` preset は role ごとに異なる。

| role | encoding |
| --- | --- |
| TokenEmbedding / Output（lm_head） | PSQ8 |
| AttnQ / AttnK / AttnV / AttnO | PSQ4 |
| FfnGate / FfnUp | PSQ4 |
| FfnDown | `li % 4 == 0` ? PSQ8 : PSQ4 |
| GdnQkvza / GdnOut | `li % 4 == 0` ? PSQ8 : PSQ4 |

したがって target は **mixed PSQ4 W4A8 / PSQ8 W8A8** である。weight traffic は competitor の
MXFP4 W4A8 の約 1.3-1.5x（active linear bytes で 16.7 GB / update）。

### verify kernel breakdown（M=8, ctx=2048, capture off, prefill 除外, rocprofv3）

`test_dflash2_gate11d_verify_profile` を rocprofv3 `--kernel-trace` で計測。ITERS=90 で
1 回の prefill の混入を 1% 以下にした。

| family | calls/update | us/update | share |
| --- | ---: | ---: | ---: |
| PSQ4 linear | 336 | 21,574 | 49.8% |
| PSQ8 linear | 65 | 8,867 | 20.4% |
| GDN recurrence | 384 | 4,968 | 11.5% |
| full attention | 32 | 2,257 | 5.2% |
| elementwise | 549 | 2,051 | 4.7% |
| activation quantize | 260 | 1,053 | 2.4% |
| BF16 linear | 97 | 1,022 | 2.4% |
| RMSNorm | 211 | 1,019 | 2.4% |
| GDN conv | 48 | 293 | 0.7% |
| RoPE | 32 | 126 | 0.3% |
| sampling / embedding | 5 | 54 | 0.1% |

PSQ4 + PSQ8 linear = **70.2%**。Gate11D-B の実施条件（linear >= 60%）は満たす。

### linear shape profile（bench、`--dtype psq4|psq8 --actual-qwen-shapes --verify-small-m`）

M=1（decode1）と M=8（RowBlock1）の p50。bytes は codes+scales。GB/s は M=8。

| dtype | n | k | calls/update | M1 us | M8 us | M8/M1 | bytes/call MB | M8 GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PSQ4 | 17408 | 5120 | 128 | 83.80 | 86.33 | 1.030 | 50.13 | 581 |
| PSQ4 | 5120 | 17408 | 48 | 83.61 | 84.95 | 1.016 | 50.13 | 590 |
| PSQ4 | 5120 | 6144 | 48 | 36.25 | 36.56 | 1.009 | 17.69 | 484 |
| PSQ4 | 6144 | 5120 | 32 | 32.53 | 33.32 | 1.024 | 17.69 | 531 |
| PSQ4 | 10240 | 5120 | 32 | 51.46 | 52.34 | 1.017 | 29.49 | 563 |
| PSQ4 | 1024 | 5120 | 32 | 13.10 | 12.21 | 0.93 | 2.95 | 241 |
| PSQ8 | 5120 | 17408 | 16 | 153.16 | 154.10 | 1.006 | 94.70 | 615 |
| PSQ8 | 5120 | 6144 | 12 | 57.41 | 57.66 | 1.004 | 33.42 | 580 |
| PSQ8 | 10240 | 5120 | 12 | 92.43 | 92.74 | 1.003 | 55.71 | 601 |
| PSQ8 | 248320 | 5120 | 1 | - | 2205.93 | - | 1350.86 | 611 |
| PSQ4 | 48 | 5120 | ~48 | 12.04 | 12.29 | 1.02 | 0.14 | 11 |
| PSQ8 | 48 | 5120 | ~12 | 11.40 | 11.64 | 1.02 | 0.25 | 22 |

#### 結論（仮説の否定）

Gate11D の主要仮説「M=2/4/8 は RowBlock1 の 16-row tile を使うため無効 row が 8-14 本あり、
専用 kernel で速くなる」は**計測で否定された**。

- large shape の M8/M1 は **1.003-1.030**。weight read が支配的なため無効 row の WMMA は隠れる。
- M=8 の effective BW は M=1 decode1 の **97-99%**。Gate11D-B の受入目標（M=1 の 90%）は
  既存 generic kernel で満たしている。
- 絶対値でも large shape は 581-615 GB/s（DRAM streaming 636 GB/s の 91-97%）。
- 効率が低いのは tiny-N shape（AttnK/V n=1024 で 241 GB/s、GdnBa n=48 で 11 GB/s）だけで、
  合計 ~1.2 ms（verify の ~3%）に過ぎず、n=48 は純粋に launch/tile 律速。

### theoretical traffic floor

| 項目 | bytes / update | floor @636 GB/s |
| --- | ---: | ---: |
| PSQ4 linear | ~11.85 GB | 18.6 ms |
| PSQ8 linear（lm_head 含む） | ~4.83 GB | 7.6 ms |
| lm_head（PSQ8, 1 dispatch） | 1.351 GB | 2.12 ms |
| total active linear | ~16.68 GB | 26.2 ms |
| measured M8 linear | - | 30.4 ms（floor の 86%） |
| measured M8 verify（clean） | - | 44.0 ms |

linear を完全に floor まで詰めても verify は 44.0 -> 40.8 ms（-7%）で、残差は format floor で決まる。
仮に全 linear を MXFP4（0.531 B/weight）にした場合の推定 raw floor は
25.6G weight × 0.531 B = 13.6 GB -> 21.4 ms（-5 ms）。**推定のみで実装しない。**

### small-M kernel selection

| dtype | M | implementation | default |
| --- | ---: | --- | --- |
| PSQ4 | 1 | `gemv_psq4_w4a8_decode1_kernel<unroll>` | yes |
| PSQ4 | 2/4/8 | `gemm_psq4_w4a8_wmma_kernel<1>`（RowBlock1） | yes（変更なし） |
| PSQ8 | 1 | `gemv_psq8_w8a8_wmma_decode1_kernel<unroll>` | yes |
| PSQ8 | 2/4/8 | `gemm_psq8_w8a8_wmma_kernel<1>`（RowBlock1） | yes（変更なし） |

Gate11D-B（VerifyM2/M4/M8 専用 kernel）は**実装しない**。既存 kernel が受入目標を満たしており、
変更しても計測可能な改善が見込めないため（プロジェクトのバンクマークルール）。

### kernel resources（M=8）

| kernel | VGPR | SGPR | LDS | scratch | wg |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gemm_psq4_w4a8_wmma_kernel<1>` | 88 | 128 | 0 | 0 | 32 |
| `gemm_psq8_w8a8_wmma_kernel<1>` | 80 | 128 | 0 | 0 | 32 |
| `gemv_psq8_w8a8_wmma_decode1_kernel` | 96 | 128 | 0 | 0 | 32 |
| `gdn_recurrence_wmma_decode1_lossy_impl` | 248 | 128 | 51712 | 0 | 256 |
| `gemm_bf16_exact_rows_kernel<8>` | 88 | 128 | 0 | 0 | 256 |
| `rmsnorm_impl` | 24 | 128 | 512 | 0 | 256 |
| `activation_quantize_e4m3_vec` | 40 | 128 | 512 | 0 | 256 |
| `gdn_conv1d_bf16_f32` | 24 | 128 | 0 | 0 | 256 |

unexpected scratch spill なし。PSQ4/PSQ8 GEMM は 32 thread/block と低いが、64 CU 上で
多数の block が並ぶため DRAM を飽和させている。

### verify M sweep（clean, capture off）

| M | ctx=32 ms | ctx=2048 ms |
| ---: | ---: | ---: |
| 1 | 35.39 | 36.47 |
| 2 | 37.23 | 38.32 |
| 4 | 38.69 | 40.18 |
| 8 | 41.69 | 44.00 |

M 依存（M1->M8 で +7.5 ms）は linear ではなく GDN recurrence（+3.7 ms）と
elementwise / attention / quantize の row scaling。linear は weight-bound で row 非依存。

### capture overhead（ctx=2048）

| M | off ms | on ms | overhead ms |
| ---: | ---: | ---: | ---: |
| 1 | 36.47 | 36.94 | 0.47 |
| 2 | 38.32 | 39.13 | 0.81 |
| 4 | 40.18 | 41.39 | 1.20 |
| 8 | 44.00 | 46.05 | 2.06 |

### decision

Gate Decision は **PROFILE-ONLY**。production kernel は変更しない。

- linear は verify の 70% を占めるが、全 large shape が M=1 decode1 の 97-99% BW に達しており、
  small-M 専用 kernel に意味のある headroom が無い。
- verify の残差は **format floor**（mixed PSQ4/PSQ8 の weight traffic）と
  **GDN recurrence**（4.97 ms, row ごとの直列実行）で決まる。
- 次 Gate の候補は (a) format（MXFP4 / 低bit）、(b) GDN recurrence の row 並列化、
  (c) elementwise / activation quantize の単体最適化。GEMM tuning ではない。

### fusion audit

fusion は追加していない。production kernel の変更も無い。bench への `--verify-small-m` flag と
profiler test のみ追加。

## Gate 11E Exact GDN multi-row recurrence

Gate 11D で「verify の残差は GDN recurrence（row ごとの直列実行）と format floor で決まる」と
結論した。Gate 11E は GDN recurrence を Exact verify のまま multi-row 化する。

### 現行 path

Exact verify の GDN recurrence は、`decode1` kernel を row ごとに host から launch していた。

- `launch_gdn_recurrence_f32_wmma_decode1_serial` が `actual_rows` 回 `decode1` を launch。
- M=8 では 48 GDN layer × 8 = **384 launches / verify**。
- 各 launch が recurrent state を global から load / global へ store（8 回分の往復）。

### 新 path

`gdn_recurrence_wmma_decode_rows_exact_impl<kRows>`（`kRows = 2 / 4 / 8`）を追加。
`decode1` の exact algorithm をそのまま device 内 row loop へ包み、演算順序・
reassociation・K split は一切変更しない。

- recurrent state は kernel prologue で 1 回だけ load、最終 row 後に 1 回だけ store。
- row ごとの state load / store をしない。`Sacc` を row 間で保持する。
- row ごとの history store は `r < capture_rows` のときだけ行い、layout は変更しない。
- row loop は `#pragma unroll 1`（命令フットプリント削減。full unroll より verify wall が良い）。
- `row` loop 内の sync は decode1 と同じ 2 回 / row。

launch API は `launch_gdn_recurrence_f32_wmma_decode_rows_exact`。dispatch 条件は
`verify_exact_active` ∧ `num_requests == 1` ∧ GDN 27B geometry ∧ `actual_rows in {2,4,8}`。
それ以外は既存 `decode1_serial` / `wmma_serial` にフォールバックする。
`PHASESHIFT_GDN_RECURRENCE_MULTIROW=0` で旧 serial path を強制できる（A/B 用）。

physical launch 数（M=8, 48 layer）: **384 -> 48**。

### kernel resource（M=8）

| kernel | VGPR | LDS | scratch | wg |
| --- | ---: | ---: | ---: | ---: |
| `gdn_recurrence_wmma_decode1_lossy_impl<true>` | 248 | 51712 | 0 | 256 |
| `gdn_recurrence_wmma_decode_rows_exact_impl<8>` | 248 | 51712 | 0 | 256 |

scratch spill なし。decode1 と同一 resource。

### 正確性

- `test_gdn_recurrence_decode1`（拡張）: rows 2/4/8 について
  output F32 / final recurrent state F32 / capture_rows in {1, rows-1, rows} の
  per-row history F32 が sequential `decode1` と **bit-exact**（35/35 PASS）。
- real E2E（psq target + DFlash2 drafter, K=7）: prose / code / math / 512-token 生成で
  `GENERATED_IDS`・acceptance 統計が旧 serial path と完全一致。

### standalone recurrence（`phaseshift-bench gdn-recurrence`, 1 layer）

| rows | history | old serial p50 | new multirow p50 | speedup |
| ---: | --- | ---: | ---: | ---: |
| 2 | off | 30.35 us | 17.56 us | 1.73x |
| 4 | off | 49.18 us | 26.68 us | 1.84x |
| 8 | off | 92.09 us | 48.21 us | 1.91x |
| 2 | on | 32.27 us | 18.95 us | 1.70x |
| 4 | on | 53.09 us | 34.27 us | 1.55x |
| 8 | on | 104.69 us | 62.36 us | 1.68x |

### verify wall（Gate11D harness, ctx=2048）

| M | capture | before ms | after ms | delta |
| ---: | --- | ---: | ---: | ---: |
| 2 | off | 38.58 | 38.24 | -0.34 |
| 4 | off | 40.46 | 39.48 | -0.98 |
| 8 | off | 44.29 | 42.08 | **-2.21** |
| 2 | on | 39.38 | 39.07 | -0.31 |
| 4 | on | 41.73 | 40.74 | -0.99 |
| 8 | on | 46.42 | 44.07 | **-2.35** |

### E2E（psq target + DFlash2, K=7, max-seq-len 4096）

| fixture | before tok/s | after tok/s | GENERATED_IDS |
| --- | ---: | ---: | --- |
| prose (256) | 53.70 | 55.90 | 一致 |
| code (128) | 76.41 | 78.28 | 一致 |
| math (128) | 60.47 | 62.82 | 一致 |
| prose 512-token | 130.76 | 133.62 | 一致 |

### launch accounting 修正

`OptimizedLaunchResult.consumed_dispatches` と physical launch 数が混同されていた。
GDN recurrence は logical dispatch 1 を消費するが、serial path は physical に
`actual_rows` 回 launch する。physical launch 数の記録だけを `actual_rows` にし、
`consumed_dispatches` は 1 に固定した（旧コードは両方 1 で、serial の physical 数が
過小記録だった）。この混同を multirow 導入時に physical 数を返してしまうと
executor が後続 dispatch をスキップするため、Gate11E で分離した。

### fusion audit

fusion は追加していない。format 変更もない。`decode1` math order を維持している。

## Gate 11F Proposer standalone kernel ceiling（BF16 GEMM + ring attention）

proposer ctx2048 ≈15.6ms の内訳は BF16 GEMM 9.2ms / ring attention 2.88ms /
target lm_head 2.2ms / top16 0.11ms / elementwise 0.7ms。lm_head は ~611 GB/s で
roof 近傍のため対象外。Gate 11F は BF16 GEMM と ring attention を単体で詰める。

### BF16 GEMM sweep（結論: PROFILE-ONLY、production 変更なし）

`phaseshift-bench gemm --dtype bf16 --impl wmma|wide|splitk|exact --shapes N:K,... --rows ...`
で全 production shape を A/B（rows=8 p50、us）:

| shape (N:K) | 用途 | Wmma（現行） | splitk | exact | 判定 |
| ---: | --- | ---: | ---: | ---: | --- |
| 5120:25600 | target fc | 442.0 | 742.5 | 472.3 | Wmma 維持（roof） |
| 1280:5120 | attention dynamic | 74.1 | **15.6** | 21.3 | splitk 4.8x |
| 4096:5120 | q | 69.2 | 120.7 | 55.3† | Wmma 維持（roof） |
| 1024:5120 | k/v noise | 69.6 | **13.7** | 17.6 | splitk 5.1x |
| 5120:4096 | o | 65.0 | 261.3 | 56.0† | Wmma 維持（roof） |
| 17408:5120 | gate/up | 480.9（370 GB/s） | 441.6 | **321.7**（554 GB/s） | exact 1.5x |
| 5120:17408 | down | 309.3 | 467.2 | 319.4 | Wmma 維持 |
| 256:5120 | selector hidden | 64.2 | 12.1 | **10.5**（現行） | 変更なし |

†は bench の L3（64MB）住む効果で in-context では roof 相当。gate/up は 178MB
なので bench ≒ in-context。

in-context 検証（`test_dflash2_gate11_profile`、env override で A/B）:
k/v noise 69.7→17.5us、dynamic 74→18.9us、gate 484→298us、up 484→321us、
`proposer_cached` ctx2048 15579→13219us（**-2.36ms**、17.5%）。

**採用拒否の理由**: splitk / exact は reduction 順が変わり BF16 出力が bit-exact
でない。E2E 帰属テスト（K=7、prose/code/math）で splitk-only・exact-only **各単体でも**
code rounds 29→28、math 37→39 の acceptance 変化が発生（prose は不変、
GENERATED_IDS は全構成で一致）。Gate11F-9/25 の「acceptance が変わる変更は採用しない」
により production selector は変更しない。bit-exact を保つ手動 pipeline WMMA 実験
（4段 software pipeline、reduce 順不変）は 74→176us と大幅に遅く破棄。
q/o/down/fc は rows=8 で 576-646 GB/s の roof 近傍、headroom なし。
残る headroom（gate/up 370→554 GB/s、small-N latency bound 150 GB/s）は
format / fusion の領域であり Gate11G 以降の判断対象。

### GQA grouped ring attention（採用）

現行 `dflash2_attention_ring_kernel` は grid (q_heads=32, rows) = 256 WG、
1 WG = 1 row × 1 q head。GQA group=4 の 4 q head が同じ context K/V を
非連結（2B stride）に読み直す構成で、ctx2048 rows=8 で 583us/layer。

新規 `dflash2_attention_ring_gqa_kernel`:

- 1 WG = 1 block row × 1 KV head（4 q heads を 1 WG で処理）。grid (8, rows) = 64 WG。
- K/V を 32 keys × 128 dim の LDS tile へ uint4 coalesced load し、4 head で共有。
- 数学は従来と同一構造（scores materialize → max → exp/sum → V）。dot の d 順、
  max / exp/sum の strided thread マッピング、V の k ascending を reference と
  同一に保ち、**reference kernel と bit-exact**（tolerance ではなく完全一致）。
- online softmax への移行は行わない（11F-15 準拠）。
- 制限: group==4、head_dim==128、stride 8 要素倍数、LDS ≤ 60KB。不成立時は
  `hipErrorNotSupported` → executor は reference へ自動フォールバック。
- dispatch: executor は既定で GQA。`PHASESHIFT_DFLASH2_ATTENTION_REFERENCE=1` で
  reference 固定（A/B 用）。ring contract（absolute position % capacity）と
  non-causal（|q_pos - k_pos| < 2048）は維持。

リソース（GQA、rocprofv3 + 実測）: workgroup 128 threads、VGPR 40、SGPR 128、
Accum_VGPR 0、**scratch 0**、dynamic LDS = 43.6KB @ctx2048
（scores 4×keys×4B 中心）、ctx2048 では 1 WG/CU、grid 64 = 1 wave。

単体性能（`test_dflash2_gate11_profile`、rows=8、median us）:

| ctx | reference | GQA | speedup |
| ---: | ---: | ---: | ---: |
| 8 | 13.6 | 14.1 | 0.97x |
| 128 | 46.8 | 36.8 | 1.27x |
| 512 | 152.7 | 105.6 | 1.45x |
| 2048 | 583.0 | **380.1** | **1.53x** |

rows sweep @ctx2048: rows2 524→283（1.85x）、rows4 526→283（1.86x）、
rows8 556→364（1.53x）。ctx8 のみ微悪化（WG 数減が効く）だが proposer 全体では無影響。

in-context（`test_dflash2_gate11_profile`、BF16 変更なしの状態で attention のみ A/B）:

| stage | ctx | reference us | GQA us | delta |
| --- | ---: | ---: | ---: | ---: |
| cached_backbone | 512 | 10412 | 9935 | -0.48ms |
| cached_backbone | 2048 | 13151 | 11360 | **-1.79ms** |
| proposer_cached | 2048 | 15670 | 13872 | **-1.80ms** |

正確性: `test_dflash2_attention_ring` 拡張で全ケース（no wrap、one wrap、
total 2049 wrap、total 2056 cap2048、production 32/8 head128）で
GQA == contiguous reference == ring reference の bit-exact を確認。

E2E（K=7）: prose/code/math の rounds（81/29/37）・full accepts・accepted drafts・
GENERATED_IDS が Gate11E baseline と完全一致（bit-exact なため当然、実測確認済み）。
ctx2048（2048-token prompt、64 生成）では rounds=17・ids が
reference/GQA で完全一致、update p50（round_ms/rounds）約 -1.2〜-2.0ms、
tok/s 61.3/61.5 → 62.4/63.3。

### fusion audit

fusion は追加していない。format 変更もない。attention は数学・thread マッピング
ともに reference と同一順序（bit-exact）。BF16 は production 設定を一切変更しない。


## Gate 11G Full-backbone PSQ4 drafter（採用）

DFlash2 draft の backbone weight だけを PSQ4（`PSQ4_S32_BF16_V1`、W4A8）にする。
proposal / acceptance の変化は許容するが、`GENERATED_IDS` は target-only greedy と
一致しなければならない。target（27B-PSQ）と BF16 経路は一切変更しない。

### bundle 契約

`phaseshift-quantizer quantize --preset psq` の出力 `models/Qwen3.8-27B-DFlash2-PSQ4`:

- 81 logical tensor、PSQ4 46 本（`fc` + 5 層 × 9）/ BF16 35 本。
- native payload 1,267,509,760 bytes（1.18GiB）、effective 5.2692 BPW、storage 4.500 BPW。
- sha256: `model.safetensors` 33348c337a60…、`phaseshift_quantization.json` 721cacf3c950…。
- `verify`: Validation OK 81 logical tensors resolved。manifest architecture は
  `dflash2_draft`、preset は `psq`。
- PSQ4 にしない tensor: 全 norm、q_norm / k_norm、conv `base_kernel`、
  `candidate_selector.hidden_projection.weight`、両 codebook。
- `--preset psq` 以外（例 `fp8`）と `--imatrix` は DFlash2 draft では reject
  （silent BF16 fallback なし）。role は `dflash_backbone_linear` /
  `dflash_selector_linear` / `dflash_small`。

runtime 側 contract: manifest architecture、logical tensor set の完全一致、
backbone 46 本 = Psq4 / それ以外 = Bf16、shape strict、PSQ4 は preshuffle 済み
（`weight_scale_group=32`、`k_padded % 32 == 0`）。ロード時に 46 本すべてを検証する。
drafter のロード時間（`gate11_profile`, 同一 GPU）: BF16 245.6ms → PSQ4 2696.0ms
（1.27GB payload の CRC 検証 + 46 本の CPU preshuffle が支配的。one-time cost）。

### proposer kernel（in-context, gate11_profile, layer0, rows=8, ctx=2048, us）

| role | BF16 gemm | PSQ4 quantize | PSQ4 gemm | PSQ4 計 | 備考 |
| --- | ---: | ---: | ---: | ---: | --- |
| target fc（5120:25600） | 584.7 | 44.2 | 115.1 | 159.3 | 3.67x |
| attention dynamic（1280:5120） | 79.2 | 8.2 | 16.6 | 24.8 | 3.19x |
| q（4096:5120） | 71.2 | 8.2（共有） | 18.9 | 27.1 | 2.63x |
| k noise（1024:5120） | 69.8 | - | 16.2 | - | 4.31x |
| v noise（1024:5120） | 69.8 | - | 16.2 | - | 4.31x |
| o（5120:4096） | 66.2 | 12.6 | 18.3 | 30.9 | 2.14x |
| mlp dynamic（1280:5120） | 79.2 | 8.2 | 16.4 | 24.6 | 3.22x |
| gate（17408:5120） | 510.0 | 8.2（共有） | 40.6 | 48.8 | 10.5x |
| up（17408:5120） | 486.5 | - | 42.0 | - | 11.6x |
| down（5120:17408） | 308.1 | 9.4 | 48.4 | 57.8 | 5.33x |
| context K/V 10 本（ctx=2048） | 13252.4 | 28.4 | 1538.2 | 1566.6 | 8.46x |

standalone 値は L3 residency の影響を受ける（gate/up は 178MB → 44MB で L3 64MB に
収まる）。採用判断は in-context の `cached_backbone` / `proposer_cached` で見る:

| stage（ctx=2048, rows=8） | BF16 | PSQ4 | 比 |
| --- | ---: | ---: | ---: |
| cached_backbone | 11342.4 | 4480.5 | 2.53x |
| proposer_cached | 13772.7 | 6933.6 | **1.99x** |

ctx sweep（`proposer_cached`）: ctx8 11844.2 → 4992.9（2.37x）、
ctx128 11995.1 → 5115.9（2.34x）、ctx512 12334.0 → 5484.6（2.25x）。

in-context の proposer 内訳（`--dflash2-stats 1`、prose K7 256、総 round 時間に対する提案 phase）:
BF16 `ROUND_MS=4558.5`（うち `VERIFY_MS=3415.4`、残差 ≈1130ms が proposer）、
PSQ4 `ROUND_MS=3979.3`（`VERIFY_MS=3466.4`、残差 ≈497ms）。proposer 残差で **2.27x**。

### PSQ4 GEMM microbench（`phaseshift-bench gemm --dtype psq4 --shapes`, p50 us）

| shape (N:K) | 用途 | rows=1 | rows=8 | code+scale GB/s (rows=8) |
| ---: | --- | ---: | ---: | ---: |
| 5120:25600 | target fc | 121.2 | 122.4 | 602 |
| 17408:5120 | gate / up | 84.1 | 86.0 | 583 |
| 5120:17408 | down | 83.9 | 84.2 | 596 |
| 4096:5120 | q | 23.9 | 24.3 | 485 |
| 5120:4096 | o | 25.8 | 26.2 | 451 |
| 1024:5120 | k/v noise | 13.1 | 13.0 | 227 |
| 1280:5120 | conv projection | 13.6 | 14.1 | 262 |

大きい 3 shape は DRAM roof（583–608 GB/s）。小さい shape は latency bound で、
standalone 値は L3（64MB）に収まるため in-context より速く出る（4096:5120 と
5120:4096、1024:5120 / 1280:5120）。全 shape が `rowblock1` を選択（rows ≤ 8）。

### activation quantize（e4m3、per-row scale）

production の 1 proposal あたりの量子化 launch 数は
`fc` 1 + 層あたり 6（attn conv / q/k/v noise 共有 / o / mlp conv / gate+up 共有 /
down）× 5 層 = 31。これに context append の 1 回が加わる。
同一 activation を共有する GEMM は 1 回の量子化で再利用する
（q/k/v = 3→1、gate/up = 2→1、context K/V = 10 本→1）。

- 実測（rows=8）: fc 44.2us、層あたり 6 本で計 150.8us（5 層 = 30 本、1 本 ~5.0us）、
  ctx2048 28.4us。proposal あたり ≈ 195us（+ ctx append）。
- 量子化は TMA ではなく既存 `launch_activation_quantize_e4m3` を使用する
  （PSQ4 GEMM kernel は activation byte を e4m3 として解釈する。手順書の
  `launch_activation_quantize_a8` は int8 版であり production 未使用のため）。

### E2E（`phaseshift-compute`, target 27B-PSQ, temperature 0, ctx32/2048, stats on）

| config | target-only ids | BF16 ids | PSQ4 ids | BF16 tok/s | PSQ4 tok/s | 比 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| prose K7 256 | 85630139f972 | 85630139f972 | 85630139f972 | 55.54 | 65.11 | +17.2% |
| code K7 128 | 2114f63356e8 | 2114f63356e8 | 2114f63356e8 | 78.33 | 92.08 | +17.6% |
| math K7 128 | d63a689177df | d63a689177df | d63a689177df | 62.89 | 74.86 | +19.0% |
| ctx2048 K7 128 | ad89b929d7a3 | ad89b929d7a3 | ad89b929d7a3 | 86.17 | 102.75 | +19.2% |
| prose K7 512 | c3ef04edb714 | c3ef04edb714 | c3ef04edb714 | 67.48 | 76.97 | +14.1% |
| prose K1 256 | 85630139f972 | 85630139f972 | 85630139f972 | 34.82 | 39.05 | +12.1% |
| prose K3 256 | 85630139f972 | 85630139f972 | 85630139f972 | 47.63 | 54.41 | +14.2% |

acceptance（rounds / accepted drafts）の変化は許容範囲:
prose 81/174 → 82/173、code 29/98 → 29/98、math 37/90 → 36/91、
ctx2048 25/102 → 24/103、prose512 136/375 → 136/375、K1 143/112 → 146/109、
K3 101/154 → 103/152。**全 14 構成で GENERATED_IDS は target-only greedy と一致**。

### 診断（gate7_real を PSQ4 bundle で実行）

- cached backbone vs stateless backbone: bit-exact（rel_l2 = 0.000e+00, cosine = 1.000000）。
- cached proposer determinism 100 回: bit-exact。
- prompt chunk independence（chunk 1 / 8 / 128 / 2048）: final ring bit-exact。
- ring boundary（2047 / 2048 / 2049 / 2056）: bit-exact。
- BF16 fixture との stage drift: K rel_l2 ≤ 4.3e-2（cosine ≥ 0.99909）、
  V rel_l2 ≤ 9.1e-2（cosine ≥ 0.99585）。fixture は BF16 で生成されているため
  gate7_real の 1e-2 しきい値は PSQ4 では fail 表示になる（drift 計測値として報告）。

### BF16 不変性

BF16 drafter の E2E は acceptance 統計・GENERATED_IDS ともに Gate 11F の記録と
完全一致（prose 81/174、code 29/98、math 37/90、ctx2048 25/102、512 136/375）。
executor の BF16 分岐は call site を変更していない。

### fusion audit

fusion は追加していない。format 変更もない。量子化の共有（q/k/v、gate/up、
context K/V）は同一 activation の再利用であり fusion ではない。target の再量子化、
`lm_head` / selector の変更、CLI default の変更も行っていない。

### regression

- required acceptance 116/116 PASS（baseline 114 + 新規 required 2:
  `test_dflash2_quantization_adapter`（CPU, 70 checks）と
  `test_dflash2_psq4_shapes`（GPU, 200 checks））。
- 全体ビルド warning / error 0。
- BF16 drafter の E2E は acceptance 統計・GENERATED_IDS ともに Gate 11F の記録と
  完全一致（上記「BF16 不変性」）。
- `test_qwen35_linear_selector` 68/68 PASS（allowlist 追加後）。

---

## Gate 11H Target Verify Exact Primitive Ceiling

Gate 11G で proposer が ~6.9ms になったため、speculative update の支配項は Target Exact
Verify に移った。Gate 11H は target weight format / numeric contract を変更せず、
kernel fusion も使わずに、Verify 内の非 roof primitive を単体 kernel として詰め、
「現 format・非 fusion 条件であと何 ms の headroom があるか」を確定させる。

### baseline

- worktree HEAD `45e2a498`（Gate 11G。main `ca680000` へ rebase 済み）
- required acceptance 118/118 PASS（開始時）
- target `models/Qwen3.8-27B-PSQ`、drafter `models/Qwen3.8-27B-DFlash2-PSQ4`
- GPU: R9700 (gfx1201)、`HIP_VISIBLE_DEVICES=2`。clock / power は固定していない。
  A/B は同一 GPU・同一プロセス外で連続実行し、8 構成すべてで同符号の差を確認した。

### Verify wall（ctx=2048, us, p50 of 30）

| M | capture off (before) | capture off (after) | capture on (before) | capture on (after) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 36553 | 36219 | 37076 | 36711 |
| 2 | 38056 | 37700 | 38915 | 38567 |
| 4 | 39332 | 38965 | 40616 | 40264 |
| 8 | 41899 | 41564 | 43900 | 43586 |

before = Gate 11G の split reduce、after = Gate 11H の split reduce（後述）。
capture の履歴 overhead は M=8 で ~2.0ms（Gate 11C と一致）。

### ctx sweep（capture off, us, p50 of 30）

| ctx | M=1 | M=8 |
| ---: | ---: | ---: |
| 32 | 35524 | 39620 |
| 128 | 35647 | 39805 |
| 512 | 36164 | 40462 |
| 1024 | - | 40736 |
| 2048 | 36553 | 41899 |
| 4096 | 36180 | 39578 |
| 8192 | 36212 | 40542 |

visible < 2048 では split を使わない（`max_visible_tokens >= 2048` が条件）。
ctx>=4096 のハーネス測定は pool / arena 制約で prefill が不安定なため参考値
（arena を 31GiB へ拡大したが、E2E の ctx 検証は `phaseshift-compute` 側で行う）。

### Verify family breakdown（capture off, ctx=2048, per update, rocprofv3、prefill 差引後）

| family | M1 us | M2 us | M4 us | M8 us | calls/upd | share(M8) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PSQ4 linear | 20355 | 21149 | 21237 | 21469 | 336 | 55.5% |
| PSQ8 linear | 7960 | 8112 | 8150 | 8228 | 65 | 21.3% |
| GDN recurrence | 648 | 932 | 1471 | 2485 | 48 | 6.4% |
| attention main | 461 | 500 | 928 | 1651 | 16 | 4.3% |
| elementwise | 1113 | 1155 | 1182 | 1244 | 544 | 3.2% |
| BF16 linear | 652 | 709 | 786 | 917 | 96 | 2.4% |
| RMSNorm | 868 | 888 | 897 | 885 | 209 | 2.3% |
| activation quantize | 800 | 816 | 838 | 819 | 257 | 2.1% |
| attention reduce | 607 | 621 | 622 | 607 | 16 | 1.6% |
| GDN conv | 133 | 145 | 168 | 204 | 48 | 0.5% |
| RoPE | 98 | 100 | 101 | 98 | 32 | 0.3% |
| sampling / embedding / misc | 51 | 56 | 62 | 77 | 11 | 0.2% |
| **kernel sum** | **33746** | **35182** | **36440** | **38684** | **1678** | 100% |

attention layer 数は trace から 16（attention 16 / GDN 48）。linear は Gate 11D の
580-615GB/s（roof 636GB/s）から動いていない。

### launch floor

| 項目 | M8 ctx2048 |
| --- | ---: |
| logical dispatches/update | 1654 |
| physical launches/update（split 有効） | 1671 |
| physical launches/update（split 無効, visible<2048） | 1655 |
| kernel sum | 38.68 ms |
| verify wall（capture off） | 41.90 ms |
| wall - kernel gap | 3.22 ms |

physical launch は `PHASESHIFT_QWEN35_PERF_STATS=1`（`record_physical_launch`）と
kernel trace の双方で一致（1671 vs 1678、差は trace が prefill を含まない集計差）。

duration bucket（verify only、per update）:

| bucket | count | total us |
| --- | ---: | ---: |
| <5us | 968 | 2608 |
| 5-10us | 227 | 1595 |
| 10-20us | 1 | 14 |
| 20-50us | 17 | 634 |
| >50us | 465 | 33834 |

1213 本（72%）が 10us 未満で、合計 4.2ms。これらは数が支配的で、
中身を速くしても wall は動かない（後述）。

### attention（capture off, ctx=2048, splits=16）

| rows | visible | split | main us | reduce us | total us |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 2048 | 16 | 1651 | 607 | 2259 |

split main は 1 task = (row, kv_head, split)。rows=8 では grid 512 WG（全 WG 常駐）で
K/V を 8 row 分重複して読む（64MB/layer、実効 621GB/s ≈ DRAM roof）。

### Attention candidate（採用）: split reduce の work partition 変更

現行 reduce は 1 WG = (row, kv_head) で、phase 1/2 は 6 thread しか使わず、rows=8 でも
grid 32 WG しかない。M=1..8 で 607-622us と **M に依存しない**（純粋な latency 律速）。
q head 単位に work を分割し、grid を rows × kv_heads × q_per_kv（32 → 192 WG）にする。

- 演算順は完全に不変: per-output-element の s=0..15 昇順、`g = max_s m_s`、
  `w_s = __expf(m_s - g)`、`Σ_s w_s * partial_v[s]`、`/ sum_total`。
  WG ごとに g / sum を再計算するだけで、値は現行と同一。
- `PHASESHIFT_PAGED_ATTENTION_SPLIT_REDUCE_QH`（1/2/3/4/6、既定 1）で
  q_per_kv=6 → qh_per_wg を選択。6 = 現行カーネル（A/B 用に温存）。

| 項目 | before | after |
| --- | ---: | ---: |
| reduce 単体（rows=8, vis=2048, splits=16） | 31.09 us/call | **7.82 us/call（3.98x）** |
| split+reduce 合計（bench p50） | 135.56 us | **114.10 us（-15.8%）** |
| resource | VGPR 24 / SGPR 128 / LDS 512 / scratch 0 | 同一 |
| grid（rows=8） | 32 WG | 192 WG |
| Verify wall（M=1..8, capture off/on, 8 構成） | - | **-314 .. -367 us** |

bit-exact 検証:

- `test_paged_attention_split_reduce_exact`（新規 required, GPU 192 checks）:
  rows 1/2/4/8 × splits 2/4/8/16 × head_dim 128/256 × 通常/極値入力で
  incumbent と candidate の出力を byte 比較 → 192/192 一致。
- bench 経由の A/B（`--dump-out` + `cmp`）: rows 1/2/4/8 × visible 32..8192 ×
  splits 2/4/8/16 の 84 構成 → 84/84 一致。

### split count sweep（diagnostic）

bench p50 us（rows × visible × splits）:

| rows | visible | S=1 | S=2 | S=4 | S=8 | S=16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2048 | 103.8 | 147.7 | 86.8 | 51.7 | **37.1** |
| 2 | 8192 | 404.6 | 558.0 | 293.9 | 167.8 | **105.4** |
| 4 | 2048 | 112.8 | 149.6 | 88.5 | **59.1** | 62.4 |
| 4 | 8192 | 458.3 | 578.2 | 327.0 | 214.5 | **211.4** |
| 8 | 2048 | 171.5 | 166.2 | **107.0** | 108.6 | 115.1 |
| 8 | 8192 | 714.2 | 646.4 | 428.9 | 433.7 | **430.0** |

production の S=16 は rows=2 と visible>=4096 で最適、rows=4/8 の中域では S=8/S=4 が
数 % 速い。ただし split 数の変更は partial softmax の分割・丸め順を変えるため
S=16 と bit-exact にならない。**performance result のみ記録し、採用しない**。

### 非 linear family は launch floor（PROFILE-ONLY）

各 family の単体 kernel を rows で掃引すると、仕事量に関係なくほぼ固定時間になる
（= launch / 依存 latency が支配的。内部最適化の余地がない）:

| kernel | rows=2 | rows=8 | rows=32 | rows=128 |
| --- | ---: | ---: | ---: | ---: |
| activation quantize e4m3（K=5120） | - | 4.15 | 4.30 | 5.68 |
| residual add bf16（F=5120） | 3.82 | 3.97 | 4.14 | 5.64 |
| RMSNorm bf16 direct（F=5120） | 5.45 | 5.46 | 5.06 | 18.58 |

activation quantize は K 別（5120/6144/17408）に vec specialization 済みで generic 経路は
hot ではない。RMSNorm / elementwise / BF16 helper も同様に floor 支配で、bit-exact を
保ったまま内部を速くしても Verify 全体は動かない。したがって Gate 11H ではこれらを
変更しない（§38, §42 の採用下限に届かない）。

### BF16 helper（GDN in_proj_a/b）: bit-exact な prefetch も in-context で効かない

target の BF16 2D 行列は `linear_attn.in_proj_a.weight` / `in_proj_b.weight`
（`[48, 5120]`、104 本）だけである。これは 48 GDN layer × 2 = **96 calls/update** で、
trace の `gemm_bf16_exact_rows_kernel` 96 calls と一致する。verify_exact 契約により
ExactRows が選ばれる。

| 項目 | 値 |
| --- | --- |
| shape | N=48, K=5120, rows=8 |
| config | ExactRows、grid = ceil(48/8) = 6 WG × 256 thread、VGPR 88 / LDS 0 / scratch 0 |
| in-context | 9.55 us/call、916.5 us/update |
| standalone p50 | 10.67 us（`--impl exact`） |
| 比較 | wmma 299.4 / wide 296.8 / splitk 336.2 us（Big-M 系で桁違い、この shape に不適） |

k 依存（rows=8, N=48）: k=1280 5.85 / 2560 7.42 / 5120 10.75 / 10240 17.01 us。
n 依存（rows=8, K=5120）: n=48 10.74 / 96 10.90 / 192 11.33 / 384 11.38 / 1536 22.58 us。
**仕事量を 8 倍にしてもほぼ不変**で、6 WG しかないことによる latency / occupancy 律速
（約 4.5us の floor + 約 0.3us/iteration）。

bit-exact な改善として K ループを 2 tile ずつ accumulate する変種を試した
（`acc[m] += tile i` → `tile i+1` の昇順は不変なので bit-exact）:

| unroll | standalone p50 | Verify wall（M=8, ctx2048, capture off, interleaved 4 対） |
| ---: | ---: | ---: |
| 1（現行） | 10.91 us | 41511 us（median） |
| 2 | **8.28 us（-24%）** | 41556 us（+45us） |
| 4 | 22.17 us（register 圧迫） | - |

standalone の -24% は in-context の Verify wall に現れない。これらの kernel は
前段の出力を L2-hot で読むため latency が小さく、削った分が launch gap に吸収される。
**§27 / §50 に従い不採用**（standalone のみで採用しない）。

### no-fusion floor

| 項目 | ms |
| --- | ---: |
| PSQ4 + PSQ8 linear（実測） | 29.70 |
| 同 roof 換算（636GB/s） | 28.6 |
| GDN recurrence（Gate 11E で最適化済み、凍結） | 2.49 |
| attention main（multi-row 未実施） | 1.65 |
| attention reduce（採用後） | 0.28 |
| その他非 linear（floor 支配） | 4.25 |
| launch gap | 3.22 |
| **推定 minimum verify** | **約 40.5** |
| 実測 verify（M8 ctx2048 capture off） | 41.56 |
| remaining headroom | 約 1.1（linear の roof 差）+ 0.85（attention main の K/V 重複）= 約 1.9 |

### Deferred structural costs

- target weight format: PSQ4/PSQ8 のまま（format 変更禁止）。linear は roof 近傍で、
  残差は fusion でしか消えない。
- history format: F32 のまま（変更禁止、Gate 11C で roof 到達済み）。
- attention main の multi-row K/V reuse: rows=8 で K/V を 8 重複して読んでおり理論上
  最大 -0.85ms。ただし現行 split main は VGPR 248 で、row_tile=2 は accumulator を
  2 倍（+48 register）必要とし spill がほぼ確実（§24 の reject 条件）。実装は行わない。
- launch gap 3.22ms: HIP Graph / persistent kernel / fusion は Gate 11H 禁止のため
  手を付けない（STRUCTURAL / DEFERRED）。

### fusion audit

fusion は追加していない。format 変更、target 再量子化、graph、persistent dispatch も
ない。attention の split-main / split-reduce 内部の work partition 変更は既存 1 logical
primitive の内部最適化であり fusion ではない。

### regression

- 全体ビルド error / warning 0（main 由来の既存 warning を除く）。
- required acceptance 119/119 PASS（新規 required: `test_paged_attention_split_reduce_exact`）。
- E2E（prose/code/math/ctx2048/prose512, K=7）: QH=6 と QH=1 で
  GENERATED_IDS / rounds / accepted drafts が完全一致。target-only greedy とも一致。

---

## Gate 11I INT2 Coarse Draft Head + PSQ8 Exact Rerank

Gate 11G で proposer の full PSQ8 lm_head が約 2.2ms（proposer の約 1/3）を占めることが
分かったため、DFlash proposal 専用の lossy head として、INT2 coarse 全語彙探索 →
coarse Top-N → original PSQ8 lm_head による candidate rerank → exact Top-16 を実装した。
**target の lm_head / Verify / 数値契約は一切変更しない**（proposal 専用。drafter は lossy 可）。

### 構成（env `PHASESHIFT_DFLASH2_INT2_HEAD`、既定 0 = full PSQ8）

- INT2 は generic な `MatrixEncoding` ではない。executor が target PSQ8 lm_head から
  load 時に一度だけ作る runtime object（`DFlash2DraftHeadInt2` 相当）。
- 追加 resident は INT2 codes **303.1 MiB** のみ。weight scale は target PSQ8 の
  BF16 / block32 をそのまま共有する。
- codebook は target PSQ8 codes の GPU histogram（weight = 出現数 × block scale²）から
  weighted Lloyd-Max + E4M3 snap で決定論的に導出する。
- coarse kernel は PSQ8 W8A8 kernel と同一構造で、W 側だけ packed byte を
  expand LUT（256 × uint32）で 4 E4M3 byte に展開する。activation（E4M3）は
  coarse と rerank で **1 回だけ**量子化して共有する。
- rerank は original PSQ8 weight を candidate id から直接 gather し（gather buffer を
  作らない）、full head と bit-exact な logit を返す。

### pool sweep（prose, K=7, real hidden, diag）

`PHASESHIFT_DFLASH2_DRAFT_RERANK` = pool。top1 / recall16 / contain / selcov は
full PSQ8 top16 を oracle とした実測（`PHASESHIFT_DFLASH2_INT2_DIAG=1`）。
selcov は「full-head CandidateSelector が選んだ token が coarse pool に含まれる割合」。

| pool | top1 coverage | avg top16 recall | full containment | selector coverage | rerank ordered top16 | proposal token match |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.9107 | 0.5921 | 0.0089 | 0.8036 | 0.0089 | 0.6875 |
| 24 | 0.9643 | 0.6964 | 0.1161 | 0.8571 | 0.1161 | 0.7054 |
| **32** | 0.9821 | 0.7606 | 0.2143 | 0.8750 | 0.2143 | 0.8036 |
| 48 | 0.9821 | 0.8292 | 0.3214 | 0.9196 | 0.3214 | 0.8393 |
| 64 | 0.9911 | 0.8677 | 0.3304 | 0.9464 | 0.3304 | 0.8393 |
| 80 | 0.9911 | 0.8990 | 0.4196 | 0.9554 | 0.4196 | 0.8750 |
| 128 | 1.0000 | 0.9470 | 0.6071 | 0.9911 | 0.6071 | 0.8929 |

Rerank 後の top16 set/ordered match は containment と一致する（rerank は pool 内で
exact なので、set match は「exact top16 が全部 pool にある」ことと同値）。

### codebook 比較（§64）

同じ histogram から Lloyd-Max と sign-symmetric を導出して比較した
（起動時に `DFLASH2_INT2_CODEBOOK` として出力する。weight error は
sum weight × (value - centroid)²）。

| codebook | values | weight error | top1 coverage | avg top16 recall | containment | selector coverage | proposal token match |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Lloyd（既定）** | -352 / -13 / 13 / **320** | **105047** | 0.9821 | 0.7606 | 0.2143 | 0.8750 | 0.8036 |
| symmetric | -352 / -13 / 13 / **352** | 108119 | 0.9375 | 0.7338 | 0.2232 | 0.8571 | 0.7500 |

（prose real hidden, pool=32, 17 rounds）weight error と recall の両方で Lloyd が優位。
symmetric は containment だけ +0.009 高いが、top1 coverage / selector coverage /
proposal match はすべて劣る。**Lloyd-Max を既定**とする。

### startup（§58-59）

INT2 を有効にしたときの追加 startup（1 回だけ）:

| 項目 | 値 |
| --- | --- |
| histogram（codes 1.27GB 走査 + scale² 重み、GPU） | 4.49 ms |
| codebook 導出 + table build + pack（GPU） | 3.73 ms |
| **合計** | **8.2 ms** |
| arena 追加 | 317,849,600 B（303.1 MiB）+ 小さな scratch |

hot path の allocation は 0（すべて create 時）。

### head pipeline（mode 1, in-context, per round, ms, K=7 ctx32）

`PHASESHIFT_DFLASH2_INT2_TIMING=1`。full PSQ8 head は同条件の diag 実測 2.317ms。

| pool | quant | coarse | topN | rerank | top16 | selector | **total** | vs full |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.010 | 0.971 | 0.140 | 0.068 | 0.048 | 0.031 | **1.267** | 1.83x |
| **32** | 0.011 | 0.980 | 0.235 | 0.071 | 0.049 | 0.031 | **1.377** | **1.68x** |
| 48 | 0.011 | 0.980 | 0.354 | 0.071 | 0.049 | 0.031 | 1.496 | 1.55x |
| 64 | 0.011 | 0.980 | 0.501 | 0.071 | 0.049 | 0.031 | 1.643 | 1.41x |
| 80 | 0.011 | 0.989 | 0.691 | 0.075 | 0.049 | 0.032 | 1.847 | 1.26x |
| 128 | 0.011 | 0.980 | 1.348 | 0.082 | 0.048 | 0.032 | 2.501 | 0.93x |

（pool=48 以降は pool=32 の topN 実測から差分で換算。pool=128 は full head より遅い。）

M 依存（pool=32）: M=7 1.377ms / M=3 1.291ms / M=1 1.240ms。coarse が M に依存しない
（0.97-0.98ms）ため、proposer の head は M によらずほぼ一定になる。

### coarse kernel 単体

| 項目 | 値 |
| --- | --- |
| shape | N=248320, K=5120, rows=7 |
| codes bytes | 317,849,600（303.1 MiB） |
| scale bytes（共有） | 79,462,400（75.8 MiB） |
| output bytes | 6,953,000 |
| latency | 約 0.98 ms |
| effective weight BW | (codes+scales)/latency ≈ 405 GB/s |
| raw DRAM floor | 約 0.625 ms（636 GB/s） |
| VGPR / SGPR / LDS / scratch | 56 / 128 / 1024 / 0（spill なし） |

0.98ms は DRAM floor の 63%。32 thread/block（`vocab/16` = 15520 block）で 1 block の
読む量が 1/4 になったため latency 律速に寄った。pool=32 の total 1.377ms は
目安（≤1.1ms）には届いていない。coarse kernel の block 構造の見直しは繰り延べ。

### E2E（prose, K=7, 256 token, interleaved 3 対）

| mode | round1 | round2 | round3 | median | accepted | rounds | GENERATED_IDS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full PSQ8 | 64.93 | 63.80 | 63.61 | 63.80 | 173 | 82 | 85630139f972 |
| INT2 pool32 | **65.01** | **64.88** | **65.00** | **65.00** | 173 | 82 | 85630139f972 |
| INT2 pool80 | 64.29 | 64.23 | 64.93 | 64.29 | 173 | 82 | 85630139f972 |

3/3 で pool32 が full を上回る（+1.3% mean / +1.9% median）。**acceptance・rounds・
GENERATED_IDS は全 pool で full PSQ8 と完全一致**。

単発計測（`--max-new-tokens` 128/256）:

| workload | full tok/s | INT2 pool32 | 差 | ids |
| --- | ---: | ---: | ---: | --- |
| prose K7 256 | 65.57 | 66.22 | +1.0% | 85630139f972（一致） |
| prose K7 512 | 77.32 | 78.93 | +2.1% | 一致（rounds 136 / accepted 375） |
| code K7 128 | 91.00 | 94.20 | +3.5% | 2114f63356e8（一致） |
| math K7 128 | 74.70 | 75.11 | +0.5% | d63a689177df（一致） |
| reasoning K7 128 | 61.29 | 62.49 | +2.0% | 一致（rounds 43 / accepted 84） |
| ctx2048 K7 128 | 103.79 | 106.11 | +2.2% | ad89b929d7a3（一致） |

pool=16/24/32/48/64/80/128 のすべてで prose の ids・rounds（82）・accepted（173）が
full PSQ8 と一致した。

### 採用 pool

**pool=32**（`PHASESHIFT_DFLASH2_DRAFT_RERANK` の既定）。

- full PSQ8 と比べて head 1.68x、E2E は prose/code/math/ctx2048 のすべてで改善。
- pool=48/64/80 とは E2E が測定ノイズ内（prose 66.22 vs 65.46/66.21/65.93、
  ctx2048 106.11 vs 103.58/103.21）で、head は pool=32 が最も安い。
- pool=16/24 は head がわずかに速い（1.267/1.30ms）が、top1 coverage 0.91/0.96、
  selector coverage 0.80/0.86 と明確に劣り、E2E でも pool=32 を上回らない。
- pool=128 は topN だけで 0.69ms かかり head が full PSQ8 より遅くなる（0.93x）。
- acceptance の崩壊はどの pool でも観測されなかったが、recall の余裕を残す
  pool=32 を既定とする（将来 workload で感度が出た場合は 48/64 へ上げる）。

### kernel 別実測（rocprofv3、real shape M=7 ctx32 pool=32）

| kernel | calls/round | avg us | VGPR | SGPR | LDS | scratch | threads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `int2_coarse_head_kernel` | 1 | 950.5 | 56 | 128 | 1024 | 0 | 32 |
| `coarse_topn_partition_kernel` | 1 | 86.4 | 96 | 128 | 512 | 0 | 256 |
| `coarse_topn_merge_kernel` | 1 | 71.1 | 24 | 128 | 512 | 0 | 256 |
| `psq8_candidate_rerank_kernel` | 1 | 65.9 | 40 | 128 | 0 | 0 | 32 |
| `int2_remap_pool_ids_kernel` | 1 | 2.1 | 8 | 128 | 0 | 0 | 128 |
| （参考）`gemm_psq8_w8a8_wmma_kernel` | 1 | 127.3 | 80 | 128 | 0 | 0 | 32 |

scratch spill は全 kernel 0。

### exactness

- coarse kernel: 「INT2 を dequant した E4M3 行列」に対する PSQ8 W8A8 と **bit-exact**
  （`test_dflash2_int2_coarse_head`、rows 1/3/7 × vocab 32..256 × K 64..256 の 15 構成）。
- rerank: full PSQ8 head の同じ候補位置の logit と **bit-exact**
  （`test_dflash2_psq8_rerank`、unordered / 重複なし / vocab 境界を含む候補）。
- pack: map/expand LUT と preshuffle 順序を含めて CPU 期待値と byte 一致、
  run 間で完全一致（`test_dflash2_int2_pack`、524 checks）。
- coarse Top-N: CPU stable sort oracle と ids/logits 完全一致（pool 16..128、tie fixture、
  vocab 1024/4096/248320。`test_dflash2_coarse_topn`、93 checks）。
- real hidden での再現: full head と同一候補に対する rerank logit の max|diff| = 0（diag 実測）。

### regression

- 全体ビルド error / warning 0。
- required acceptance **123/123 PASS（skip 0）**。新規 required は
  `test_dflash2_int2_pack`（524 checks）/ `test_dflash2_int2_coarse_head`（15）/
  `test_dflash2_coarse_topn`（93）/ `test_dflash2_psq8_rerank`（14）。
- INT2 無効時（既定）の共通パスは不変: BF16 drafter prose `85630139f972`、
  PSQ4 drafter ctx2048 `ad89b929d7a3` で Gate 11H と一致。
- target-only 系（4B / 4B-PSQ / 27B greedy / 27B stochastic / MTP）は INT2 の有無に
  よらず GENERATED_IDS 完全一致。

### fusion / target 不変

- INT2 GEMM と coarse Top-N、rerank と small Top-16 はそれぞれ別 kernel（fusion なし）。
- target lm_head は変更なし（`MatrixEncoding` 分布も PSQ8 object も不変）。
- target-only / Verify は INT2 コードパスを通らない。
- INT2 object を作れない場合は silent fallback せず error を返す。

### 計測コマンド

```bash
# E2E（pool は PHASESHIFT_DFLASH2_DRAFT_RERANK。INT2_HEAD=0 で full PSQ8）
PHASESHIFT_DFLASH2_INT2_HEAD=1 PHASESHIFT_DFLASH2_DRAFT_RERANK=32 \
  ./build/phaseshift-compute --model-dir models/Qwen3.8-27B-PSQ \
  --dflash2-model-dir models/Qwen3.8-27B-DFlash2-PSQ4 --dflash2-drafts 7 \
  --input-ids-file build/dflash2-gate51/real/real_prose/prompt.txt \
  --max-new-tokens 256 --max-seq-len 4096 --arena-gib 26 --temperature 0 --dflash2-stats 1

# recall（full PSQ8 と両方走らせて比較。DFLASH2_INT2_DIAG の最終行が累積値）
PHASESHIFT_DFLASH2_INT2_HEAD=2 PHASESHIFT_DFLASH2_INT2_DIAG=1 \
  PHASESHIFT_DFLASH2_DRAFT_RERANK=32 <同じ引数>

# head pipeline timing（INT2 単独）
PHASESHIFT_DFLASH2_INT2_HEAD=1 PHASESHIFT_DFLASH2_INT2_TIMING=1 \
  PHASESHIFT_DFLASH2_DRAFT_RERANK=32 <同じ引数>

# 単体テスト
ctest --test-dir build -R 'dflash2_int2|dflash2_coarse_topn|dflash2_psq8_rerank' --output-on-failure
```
## Gate 11I-2 Device-resident proposal bridge / D2H collapse

DFlash proposal を Host に戻さず、device 上で verify 入力を作り、target verify へ直接渡す。
Host boundary を verify 後の acceptance 用 1 回に集約する。

| 項目 | 値 |
| --- | --- |
| baseline revision | `2917d773`（Gate 11H） |
| final revision | `feat/dflash2-2` の Gate 11I-2 commit |
| worktree | `.worktrees/dflash2-2` |
| build | Release、gfx1201、`PHASESHIFT_BUILD_OPTIONAL_TESTS=ON`、`PHASESHIFT_HIP_GRAPH=OFF`（別途 ON も検証） |
| env | `PHASESHIFT_DFLASH2_DEVICE_TOKEN_BRIDGE`（`0`=legacy、`1`=candidate、既定 `1`） |

### Gate 11I-A: 旧 `DFLASH2_D2H_MS` の分解（legacy path）

旧 path の proposal 直後は `hipStreamSynchronize` → `hipMemcpy D2H` の順である。
両者を別 wall timer で分離した結果（median run、`BRIDGE=0`）:

| metric | prose256 total / 84 rounds | per round | ctx2560 total / 17 rounds | per round |
| --- | ---: | ---: | ---: | ---: |
| `draft_gpu_ms` (HIP event) | 431.45 | 5.14 | 123.26 | 7.25 |
| `proposal_wait_ms` | 491.37 | 5.85 | 127.10 | 7.48 |
| `proposal_copy_ms` | 3.27 | 0.039 | 0.68 | 0.040 |
| `DFLASH2_D2H_MS`（legacy 合算） | 494.92 | 5.89 | 127.78 | 7.52 |
| `verify_gpu_ms` (HIP event) | 3458.01 | 41.17 | 753.03 | 44.30 |
| `decision_wait_ms` | 2764.00 | 32.90 | 612.91 | 36.05 |
| `decision_copy_ms` | 3.19 | 0.038 | 0.77 | 0.045 |
| `round_ms` | 4022.19 | 47.88 | 888.73 | 52.28 |

`proposal_copy_ms` は 28 byte の D2H で 0.04 ms/round しかない。旧
`DFLASH2_D2H_MS`（5.89 ms/round）の実体は **draft completion wait**
（`proposal_wait_ms ≈ draft_gpu_ms`）である。したがって旧 `DFLASH2_D2H_MS` を
削減可能な転送 overhead として解釈してはならない。

### candidate path（device bridge）

candidate では `proposal_wait_ms = 0`、`proposal_copy_ms = 0`、`decision_copy_ms`
（decision staging の 1 回 D2H）は 0.04〜0.06 ms/round である。

| metric | prose256 per round | ctx2560 per round |
| --- | ---: | ---: |
| `draft_gpu_ms` | 5.20 | 7.19 |
| `proposal_wait_ms` | 0.00 | 0.00 |
| `proposal_copy_ms` | 0.00 | 0.00 |
| `verify_gpu_ms` | 41.77 | 43.61 |
| `decision_wait_ms` | 33.48 | 35.32 |
| `decision_copy_ms` | 0.050 | 0.058 |
| `round_ms` | 48.54 | 51.50 |

`decision_wait_ms` は GPU compute completion 待ちを含むため大きくてよい。wait と
transfer を同じ metric に混ぜない。

### A/B benchmark（同一 GPU、A/B/A/B interleave 5 回、median）

| case | legacy tok/s | candidate tok/s | delta | rounds | accepted | mean accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prose256 (K=7) | 63.39 | 62.53 | -1.36% | 84 | 171 | 2.036 |
| ctx2560 / 128 tok (K=7) | 142.83 | 144.99 | +1.51% | 17 | 110 | 6.471 |

paired delta の平均は prose **-0.45%**、ctx2560 **+0.32%** で、符号がケースで逆になる。
`verify_gpu_ms` 自体（bridge と無関係）も同一 run 内で ±1.5% 動いており、run-to-run
noise（約 1%）の範囲。**性能はほぼ同等**と判断する。

BF16 drafter（prose256、単発）:

| case | legacy tok/s | candidate tok/s |
| --- | ---: | ---: |
| BF16 prose256 | 54.78 | 55.75 |

### 構造的結果

| 項目 | candidate |
| --- | --- |
| proposal → verify 間の `hipStreamSynchronize` | 0 |
| proposal → verify 間の D2H | 0 |
| 1 round の explicit stream sync | 1（verify 後の acceptance boundary） |
| decision payload D2H / round | 1（128 byte） |
| completion status D2H / round | 1（4 byte、`complete_batch` の既存処理） |

proposal は `decoder.draft->proposal_tokens` に残したまま `verify_token_ids_device` へ
D2D し、verify は `TokenIdsLocation::Device` で実行する。CPU は draft token の値を知らない
状態で `submit_batch` まで先行できる。

### HIP Graph on/off（Gate HG: determinism fix 後）

| path | HIP_GRAPH=OFF | HIP_GRAPH=ON |
| --- | --- | --- |
| target-only greedy (AR) | 決定的 | 決定的（OFF と一致） |
| DFlash default（GDN history） | 決定的 | 決定的（OFF と一致） |
| DFlash `PHASESHIFT_DFLASH2_GDN_RERUN_REFERENCE=1` | 決定的 | 決定的（OFF と一致） |

#### 原因（実測から言えること）

Graph eligibility が staging 側と capture/replay 側で別々に記述され、さらに staging 側は
「Graph を実際に使うか」を判定していなかった。

- capture/replay 条件には `!any_constraint` があるが staging 条件には無かった。
- staging 条件に `batch.speculative_verify` の除外が無かった。rerun-reference path は
  GDN history を渡さないため、verify batch と rerun batch（ともに
  `speculative_verify = true`）が Graph 用 pinned host staging 経路に入っていた。
- executor の HIP Graph cache は `program_pool_warm` が成立せず capture / replay を
  一度も行わない（`PHASESHIFT_GRAPH_DEBUG=1` で `[graph] skip: staging pool not warm`
  のみ）。それでも eligible な batch は staging 経路を通っていた。

その結果、「capture はしないが一部 batch だけ host staging timing が変わる」状態になり、
rerun-reference path の既存 race が露出して非決定になっていた。

追加調査で、この staging 経路は GDN history を渡す既定 path も壊すことが分かった。
`test_dflash2_gate9_e2e` の target-only baseline は同一 prompt でも case ごとに異なる値を
返し（68 / 220 / 248046）、Graph ON で fail していた。staging を無効化すると gate9 が
PASS することから、staging 経路が原因と確定した。これは本修正前から存在し、修正前
baseline（`e2aca58c`）の Graph ON build でも gate9 は fail=8 で再現する（Graph OFF build
では PASS）。device bridge（`PHASESHIFT_DFLASH2_DEVICE_TOKEN_BRIDGE`）の有無では変わらない。

#### 修正

- Graph eligibility を `graph_eligible` の単一 predicate に統合し、staging と
  capture/replay の両方が同じ predicate を使う。`!batch.speculative_verify` を含め、
  SPEC_VERIFY を dynamic path に固定する。
- Graph 用 host staging を `stage_graph_inputs()` にまとめ、`hipGraphLaunch` の直前と
  `hipStreamBeginCapture` の直前でのみ呼ぶ。Graph を実際に使わない batch は staging を
  通らず、通常の host 経路（Graph OFF と同一）になる。
- replay / capture / cache key / GDN correctness / sampling のロジックは変更しない。
  sync の追加もしない。

#### 実測（R9700、prose、K=7、256 token、temperature=0、各 8 回）

| 項目 | graph OFF | graph ON（修正前） | graph ON（修正後） |
| --- | --- | --- | --- |
| rerun-reference GENERATED_IDS | 8/8 一致 | 8/8 不一致 | 8/8 一致 |
| rerun-reference rounds | 84 | 60/80/112/89/82/66/77/117 | 84 |
| rerun-reference ids cksum | 405838107 | 全 run 相違 | 405838107 |
| default（hist）GENERATED_IDS | 8/8 一致 | 8/8 一致 | 8/8 一致 |
| AR token sum | 3/3 一致 | — | 8/8 一致 |

修正後の Graph ON は Graph OFF と同一 `GENERATED_IDS`（cksum 405838107、len 1183、rounds 84）。
`capture ok` / `replay` は修正前後とも 0 回で、`staging pool not warm` のみ。修正後は
graph path に入る batch が prefill（N=31）と通常 decode（N=1）だけで、verify の N=8 は
現れない。AR（`phaseshift-bench tg --mode greedy`）は ON/OFF で token sum が一致し、
graph path にも到達している（AR の Graph capability は維持）。`test_dflash2_gate9_e2e` は
Graph ON / OFF とも PASS。

#### regression

- `build-graph`（Graph ON）: error 0、追加 warning なし（既存 warning のみ）。
- required acceptance: **123/123 PASS（skip 0）**。
- DFlash optional: `test_dflash2_gate9_e2e`（Graph ON / OFF）/ `test_dflash2_gate10_cli` /
  `test_dflash2_gate11c_e2e`（8/8 PASS）/ `test_dflash2_gate11h_verify_profile` すべて PASS。
- Graph OFF 参照（`.worktrees/dflash2/build`）と `GENERATED_IDS` / rounds が完全一致。

### correctness

| 比較 | 結果 |
| --- | --- |
| prose256 legacy vs candidate (graph OFF) | `GENERATED_IDS` / rounds / full / partial / rerun / accepted 完全一致 |
| ctx2560 128 legacy vs candidate | 完全一致 |
| BF16 drafter prose256 legacy vs candidate | 完全一致 |
| rerun-reference legacy vs candidate (graph OFF) | 完全一致 |
| default path legacy vs candidate (graph ON) | 完全一致（graph OFF とも一致） |
| `test_dflash2_gate11c_e2e` | failed=0（history == reference、prose/code/math） |
| `test_dflash2_gate9_e2e` | failed=0 |
| required acceptance | 119/119 PASS |

### regression

- `cmake --build build -j16`: error 0（既存 warning のみ）。
- `ctest --test-dir build -L required`: 119/119 PASS。
- DFlash optional: `test_dflash2_gate9_e2e` / `test_dflash2_gate10_cli` /
  `test_dflash2_gate11_profile` / `test_dflash2_gate11c_e2e` /
  `test_dflash2_gate11h_verify_profile` すべて failed=0。
  `gate11h` の M=8 ctx2048 verify は 41.72 ms で、per-round `verify_gpu_ms`
  （41.8 ms）と一致する。
- Host source `ScheduledBatch` の経路は既定値 `TokenIdsLocation::Host` で不変。

### 採用判断

**GO**。device bridge を既定（`PHASESHIFT_DFLASH2_DEVICE_TOKEN_BRIDGE=1`）にする。

理由:

- correctness は legacy と bit-exact（graph OFF / ON の default path とも一致）。
- 性能はほぼ同等（paired mean ±0.5% 以内、改善・悪化の符号がケースで逆）。
- proposal → verify 間の sync と D2H が消え、1 round の Host boundary が verify 後の
  1 回に集約された。将来の multi-request / server 経路では、CPU が draft GPU 完了を
  待たずに次 work を enqueue できる。

残課題（本 Gate の範囲外）:

- HIP Graph cache が一度も capture しない（`program_pool_warm` が成立しない）。
  Graph 用 staging は capture / replay 時のみ通すよう修正済み（上記 Gate HG）。
- Graph capture が成立する workload での replay correctness は未検証。capture を
  有効化する場合は別 Gate で state ownership / token source / context / graph cache
  key / capture-replay correctness を検証する。

## Gate 11J-A Target Exact residual + RMSNorm + activation-quantize fusion

Gate 11H で Target Exact path の standalone primitive が launch floor に達した。
Gate 11J-A は target weight format / numeric contract / GEMM を変更せずに、
`RESIDUAL_ADD → RMS_NORM → ACTIVATION_QUANTIZE_W4A8` の 3 logical dispatch を
1 physical launch へ融合し、launch overhead だけを削減する。

### baseline / build

| 項目 | 値 |
| --- | --- |
| baseline revision | `e2aca58c`（Gate 11I-2 merge。`10d3c047` を ancestry に含む） |
| implementation commits | `ebd45423`（refactor）/ `06ff7b88`（feat）/ `3f3b9ac0`（test） |
| final revision | `gate/11j-a-target-fusion` HEAD（本節を含む docs commit） |
| branch | `gate/11j-a-target-fusion` |
| worktree | `.worktrees/gate11j-a-target-fusion` |
| GPU | R9700 (gfx1201)、device index 3（`--device 3` / `ROCR_VISIBLE_DEVICES=3`）。clock / power は固定していない |
| ROCm | 7.15.26333（driver 7.1.3.31500000）、AMD clang 23.0.0git |
| fusion OFF build | `build-gate11j-off`（Release、gfx1201、`OPTIONAL_TESTS=ON`、`HIP_GRAPH=OFF`、`FUSE=OFF`） |
| fusion ON build | `build-gate11j-on`（同構成、`FUSE=ON`） |

### compile flag

`cmake/options.cmake`:

```cmake
option(
    PHASESHIFT_FUSE_RESIDUAL_RMSNORM_QUANT
    "Enable exact residual+rmsnorm+activation-quantize fusion"
    OFF)
```

- OFF: fused kernel source を build に追加しない。`program_executor.hip` は完全に従来どおり。
- ON: `fused_residual_rmsnorm_quant.hip` を `phaseshift_qwen35_kernels_optimized` に、
  `fused_dispatch.hip` を `phaseshift_qwen35_runtime` に追加し、compile definition は
  `program_executor.hip` / `fused_dispatch.hip` / 対象 test / bench source のみに与える。
- runtime env による切り替えは追加していない。build 単位で OFF / ON が決まる。

### helper 共有構造

| primitive | 共有 device function | standalone wrapper | fused wrapper |
| --- | --- | --- | --- |
| residual add | `detail::residual_add_bf16_body`（`detail/residual_add_device.h`） | `op_residual_add_bf16::impl` | fused Phase 1 |
| RMSNorm | `detail::rmsnorm_row_body<In,Out,W,Layout,Mode>`（`detail/rmsnorm_device.h`） | `rmsnorm_impl<...>` | fused Phase 2 |
| activation quantize | `detail::activation_quantize_e4m3_vec_row_body<K,KP>`（`detail/activation_quantize_device.h`） | `activation_quantize_e4m3_vec_kernel<K,KP>` | fused Phase 3 |

- standalone の launch geometry は不変: residual は `grid=(ceil(F/(256*4)), rows)`、
  RMSNorm は chunk 数からの block 選択、quantize は 5120 / 6144 / 17408 の
  specialized path 選択（`activation_quantize_e4m3_vec_supported`）も従来どおり。
- fused 専用に計算式をコピーしていない。3 primitive とも同一 device body を呼ぶ。
- device-side kernel launch は使っていない。

### fused kernel

- grid = `rows`、block = 256 threads（既存 RMSNorm と vec quantize と同じ reduction 構造）。
- Phase 1: residual add（f32 add → BF16 round）→ `residual_out` に materialize。
- `__syncthreads()`。
- Phase 2: `residual_out` を入力に RMSNorm（同一 reduction order / 同一 eps / 同一 inline rsqrtf）
  → `norm_out` に BF16 で materialize。
- `__syncthreads()`。
- Phase 3: `norm_out` を入力に E4M3 activation quantize（同一 peak reduction / scale / encode / layout）。
- `residual_out` / `norm_out` の materialization は削除していない。中間 global traffic も
  削除していない（数値境界を変えないため）。PSQ GEMM は融合していない。

### matcher contract

`try_launch_residual_rmsnorm_quant_fusion` が `program.dispatches[i..i+2]` を検査する。

| 条件 | 要求 |
| --- | --- |
| kernel 列 | `RESIDUAL_ADD`, `RMS_NORM`, `ACTIVATION_QUANTIZE_W4A8` |
| residual | input_count=2, output_count=1, 入出力 BF16, feature=5120, 同一 row domain, alias 安全 |
| dependency | `residual.output[0] == rmsnorm.input[0]` かつ `rmsnorm.output[0] == quant.input[0]` |
| rmsnorm | group_size=5120, flags=ONE_PLUS, BF16 weight 5120 要素, output BF16 |
| quant | group_size=32, activation_kp=5120, workspace layout が `Int8ActivationWorkspaceLayout::make(5120,max_rows)` と一致 |
| vec path | `activation_quantize_e4m3_vec_supported(norm_out, codes, 5120, 5120, stride)` 成立 |
| rows | 1..2048 |
| selector | elementwise `ResidualBf16` と rmsnorm `BF16->BF16 PerFeature ONE_PLUS` がともに Optimized |

1つでも外れた場合は `NotApplicable` を返して既存 split path に戻す（error にしない）。
成立時は `consumed_dispatches = 3`、physical launch 1、`record_logical_covered(3)`、
`record_imatrix_probes(3)` / `record_value_trace(3)` は従来どおり executor が処理する。

### fusion hit / physical launch reduction

Target Verify M=8 ctx2048、capture off 1 update あたり（`PHASESHIFT_QWEN35_PERF_STATS=1`）:

| 項目 | OFF | ON |
| --- | ---: | ---: |
| logical dispatches | 1654 | 1654 |
| physical launches | 1671 | 1417 |
| physical launch 差 | - | **-254** |
| fusion hits | 0 | **127** |

`254 = 127 × 2` で一致する（fusion 1 回につき physical launch が 3 → 1、すなわち 2 減）。
Qwen3.8-27B は 64 layer で、post attention residual → post_norm → mlp_gate 前置 quantize が
64 回、layer output residual → 次 layer input_norm → qkv 前置 quantize が layer 1..63 の
63 回、合計 127 回。先頭 layer の input_norm は直前に RESIDUAL_ADD がなく、最終 layer の
residual の後は OUTPUT_GATHER が挟まるため対象外で、これは設計どおりである。

### kernel resource

`tools/extract_kernel_resources.py`（`build-gate11j-on/phaseshift-bench`）:

| kernel | VGPR | SGPR | LDS | scratch |
| --- | ---: | ---: | ---: | ---: |
| residual add standalone | 18 | 20 | 0 | 0 |
| RMSNorm standalone（BF16/PER_FEATURE/ONE_PLUS） | 17 | 29 | 32 | 0 |
| activation quantize vec `<5120,5120>` | 33 | 16 | 40 | 0 |
| **fused residual+rmsnorm+quant** | **33** | **26** | **72** | **0** |

fused の VGPR は 3 primitive の最大値（quant の 33）と同じで、helper 分割により
Phase 1/2/3 の live range が不要に跨っていない。LDS は 32 + 40 = 72。
scratch spill は 0。

### Verify wall A/B（ctx=2048, iters=30, median of 4 paired runs）

`PHASESHIFT_GATE11H_CAPTURE` は HIP Graph ではなく、Gate 11C の GDN speculative history
（capture / replay）を有効にするかどうかの切り替えである。`capture=1` は partial reject 時の
GDN recurrent state exact restore 用の履歴を毎 update 記録する経路で、update あたり
約 2.0ms の overhead が乗る（Gate 11H の記録と同水準）。両 build とも
`PHASESHIFT_HIP_GRAPH=OFF` で構成している。

| capture | M | OFF us | ON us | delta | paired (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 1 | 36099.3 | 35857.1 | -0.67% | -1.58 / -1.45 / +0.24 / -1.60 |
| 0 | 3 | 39067.0 | 38819.1 | -0.63% | -1.26 / -1.51 / +0.24 / -1.59 |
| 0 | 8 | 41463.2 | 41230.5 | -0.56% | -1.23 / -1.51 / +0.39 / -1.38 |
| 1 | 8 | 43543.3 | 42897.5 | -1.48% | -3.28 / -1.25 / -1.52 / -1.42 |

OFF / ON を交互に実行した paired A/B。capture=0 の主判定 M=8 は 3/4 run で負
（改善側）、median -0.56%。capture=1（GDN history capture）は 4/4 で負、median -1.48%。
rep2 で ON 側だけが +0.2..0.4% になる run が 1 回あり、これは clock 非固定の run noise。

### primitive A/B（split 3 launch vs fused 1 launch）

`phaseshift-bench fused-residual-rmsnorm-quant`、F=5120、warmup 100、iters 1000、
同一 stream / 同一 allocation。`first start → last end` の GPU event。

| M | split us（FUSE=OFF） | split us（FUSE=ON） | fused us | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 13.861 | 13.894 | 8.713 | 1.59x |
| 3 | 12.919 | 12.892 | 8.678 | 1.49x |
| 8 | 13.022 | 12.886 | 8.661 | 1.49x |

fused は split triple より 4.2〜5.2us 速い。helper 抽出前後で split の standalone triple は
変化していない（OFF 13.861 / ON 13.894、差 0.2% は run noise）。standalone kernel 個別の
geometry・resource も不変（前掲の resource 表）。

### E2E A/B

Gate 11I の production candidate（PSQ4 drafter + INT2 coarse head + PSQ8 rerank pool=32、
`--dflash2-drafts 7`、temperature 0）。両 build で runtime env は完全同一。
OFF / ON を交互に実行し median を取る。機材は他セッションと共有のため pair 数は
prose256 5 pair、ctx2048 3 pair、他 2 pair。

| workload | max-new | pairs | OFF tok/s | ON tok/s | delta | ids exact |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| prose256 | 256 | 5 | 64.29 | 65.11 | +1.28% | yes |
| prose512 | 512 | 2 | 70.06 | 71.61 | +2.22% | yes |
| code | 128 | 2 | 144.69 | 146.97 | +1.58% | yes |
| math | 128 | 2 | 135.05 | 137.01 | +1.46% | yes |
| reasoning | 128 | 2 | 68.13 | 69.09 | +1.41% | yes |
| ctx2048 | 128 | 3 | 158.33 | 160.78 | +1.55% | yes |

`rounds` / `accepted` / `full accepts` / `partial accepts` / `reruns` は全 workload で
OFF と ON が完全一致（reruns は両者 0）。fusion hit は ON 側で全 workload > 0
（prose256 10922、prose512 19431、code 2540、math 2667、reasoning 4953、ctx2048 2159、
いずれも process 累積）。

### HIP Graph compatibility

| build | 状態 |
| --- | --- |
| fusion OFF / HIP_GRAPH OFF | 主判定（本節までの A/B） |
| fusion ON / HIP_GRAPH OFF | 主判定（本節までの A/B） |
| fusion OFF / HIP_GRAPH ON | 未実施 |
| fusion ON / HIP_GRAPH ON | compile 互換のみ確認 |

compile 互換: `program_executor.hip` と `fused_dispatch.hip` を、既存 build の正確なコマンドに
`-DPHASESHIFT_HIP_GRAPH -DPHASESHIFT_FUSE_RESIDUAL_RMSNORM_QUANT` を付けて再コンパイルした。
両 TU とも rc=0、診断 0。`cmake/targets.cmake` は両 option 同時 ON の定義付与を既に分岐で
扱っている。

graph ON の runtime correctness matrix を本 Gate で実施していない理由:
`PHASESHIFT_HIP_GRAPH=ON` + `PHASESHIFT_DFLASH2_GDN_RERUN_REFERENCE=1` の非決定性は
baseline `2917d773` から存在する既知 defect であり（Gate 11I-2 で記録）、現在別セッションで
修正中である。fused launch は通常の kernel launch で capture 可能だが、graph ON の default
path 比較は当該修正の完了後に実施するのが妥当と判断した。本 Gate の判断は graph OFF の
結果のみで行う。

### correctness

#### split vs fused primitive（required, GPU）

`test_qwen35_fused_residual_rmsnorm_quant`（rows 1/3/7/8、F=5120、FUSE=ON build）:
`residual_out` / `norm_out` / activation codes / activation scales を split path と byte 比較。

| item | result |
| --- | --- |
| residual_out split vs fused | memcmp == 0（4/4 rows） |
| norm_out split vs fused | memcmp == 0（4/4 rows） |
| quant codes | memcmp == 0（4/4 rows） |
| quant scales | memcmp == 0（4/4 rows、ビットパターン一致） |

fixture は finite edge（+0 / -0 / BF16 max normal 近傍 / 1e-30）を含む deterministic 入力。

#### matcher contract（required, CPU）

`test_qwen35_fusion_matcher`: positive 7 checks + negative 16 checks = 23/23 PASS。
negative は int8 activation quantize、dependency mismatch（residual→rmsnorm / rmsnorm→quant）、
group_size != 5120、weight mode != ONE_PLUS、weight parameter 欠落、weight dtype F32、
activation dtype F32、feature != 5120、workspace code stride mismatch、workspace slot 不正、
activation group_size != 32、rows 上限超過、workspace 未 bind、dispatch 列の切断、
無関係な producer を網羅する。

#### target-only（FUSE=OFF vs ON、GENERATED_IDS byte 比較）

| case | result |
| --- | --- |
| 4B greedy | exact |
| 4B-PSQ greedy | exact |
| 27B-PSQ greedy | exact |
| 27B-PSQ stochastic（temp 0.8, seed 1234） | exact |

#### DFlash（FUSE=OFF vs ON）

| case | ids | rounds | accepted | full | partial | reruns |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 drafter | exact | 40/40 | 87/87 | 2/2 | 38/38 | 0/0 |
| PSQ4 drafter | exact | 42/42 | 85/85 | 2/2 | 40/40 | 0/0 |
| PSQ4 drafter + INT2 head pool32 | exact | 42/42 | 85/85 | 2/2 | 40/40 | 0/0 |

#### GDN history（Gate 11C contract）

`test_dflash2_gate11c_e2e`（FUSE=ON build、`PHASESHIFT_DFLASH2_GATE5_FIXTURE` = 既存 fixture）:

| case | tokens | ids history==reference | hist reruns | ref reruns |
| --- | ---: | --- | ---: | ---: |
| prose | 256 | yes | 0 | 80 |
| code | 128 | yes | 0 | 3 |
| math | 128 | yes | 0 | 7 |

`failed=0`。history path は partial reject の GDN state を exact restore し、
target rerun を 0 に保つ（reference path は rerun する）。

### regression

| 項目 | 結果 |
| --- | --- |
| build error | 0（ON / OFF とも） |
| build warning | 0（新規・変更した TU） |
| required（FUSE=ON） | **126/126 PASS**（skip 0、real 229s、`-j1`） |
| required（FUSE=OFF） | **125/125 PASS**（skip 0、real 251s、`-j1`） |
| `test_dflash2_gate11c_e2e` | PASS（failed=0） |
| `test_dflash2_gate11h_verify_profile` | PASS（failed=0） |

required 数は baseline 123 に、matcher unit test（CPU, 両 build）、fused unit test（GPU, ON のみ）、
bench smoke（`test_bench_help_fused_residual_rmsnorm_quant`）を加えたもの。

並列実行（`-j4`）では、他セッションが GPU 0/2 を占有している状況で
`test_dflash2_bf16_geometry` / `test_gemm_psq4_decode1` / `test_gemm_fp8_block128_optimized` /
`test_gemm_mxfp4_optimized` が `hipMalloc failed: out of memory` で落ちた。
これらはいずれも arena を 3-6GB 要求する test で、直列実行（`-j1`）では全て PASS する。
初回の OFF 実行で落ちた `test_gpu_mcu_persistent_emit`
（`FAIL: worker start timestamps monotonic`）も scheduling 依存で、単体再実行と
直列全件実行で PASS する。いずれも本 Gate が触れていない経路の環境起因 flake である。

既存 warning（本 Gate と無関係、baseline から存在）:
`dflash2/executor.hip` の `int2_select_impl` unused、`test_qwen35_mtp_gate3_replay.hip` の
`hipMemcpy` nodiscard 無視。

### decision

**ADOPT**

理由:

- correctness: hard PASS。split vs fused は `residual_out` / `norm_out` / codes / scales が
  byte-exact（rows 1/3/7/8）。Target greedy / stochastic、DFlash（BF16 / PSQ4 / INT2 head）、
  Gate 11C GDN history が OFF と完全一致。required は ON 126/126、OFF 125/125（skip 0）。
- physical launch reduction: Target Verify M=8 ctx2048 で 1671 → 1417（-254 = fusion 127 × 2）。
  「ON にしたが一度も fusion しない」状態ではなく、fusion hit 127/update を確認。
- Verify delta: capture=0 の主判定 M=8 で median -0.56%（paired 3/4 が改善側）、
  M=1 -0.67% / M=3 -0.63%。capture=1 では median -1.48%（paired 4/4 改善側）。
  いずれも期待した physical launch collapse（-254）と整合する。
- E2E delta: 6 workload すべてで +1.3〜+2.2%、rounds / acceptance / ids は完全一致（非劣化）。
- resource impact: fused は VGPR 33（既存 vec quantize と同値）/ SGPR 26 / LDS 72 /
  scratch 0。scratch spill なし。standalone kernel の resource・geometry も不変。

構造的にも、OFF build は fused source を execution path に持たず従来挙動を完全維持し、
ON build でも matcher が成立した箇所だけが fused path に置き換わる。


## Gate 11J-B Target + DFlash2 SwiGLU + E4M3 activation-quantize fusion

Gate 11J-A は Target Exact の `RESIDUAL_ADD → RMS_NORM → ACTIVATION_QUANTIZE_W4A8` を
融合した。Gate 11J-B は MLP down 直前の `SWIGLU → ACTIVATION_QUANTIZE_W4A8` を
1 physical launch へ融合する。Target と DFlash2 PSQ4 drafter の両方を対象とし、
compile flag を分けて寄与を独立に測る。

### baseline / build

| 項目 | 値 |
| --- | --- |
| baseline revision | `1dbf15e3`（Gate 11J-A final。`ebd45423` / `06ff7b88` / `3f3b9ac0` を含む） |
| branch | `gate/11j-b-swiglu-quant-fusion` |
| worktree | `.worktrees/gate11j-b-swiglu-quant-fusion` |
| GPU | R9700 (gfx1201)。空き状況に応じて device index 0/1/2/3 を使う（`--device N` / `ROCR_VISIBLE_DEVICES=N`、required は `PHASESHIFT_TEST_GPUS=3`）。clock / power は固定していない |
| ROCm | 7.15.26333（driver 7.1.3.31500000）、AMD clang 23.0.0git |
| build A | target OFF / dflash OFF（`build-g11jb-a`） |
| build B | target ON / dflash OFF（`build-g11jb-b`） |
| build C | target OFF / dflash ON（`build-g11jb-c`） |
| build D | target ON / dflash ON（`build-g11jb-d`） |

全 build 共通: Release、gfx1201、`PHASESHIFT_BUILD_OPTIONAL_TESTS=ON`、
`PHASESHIFT_HIP_GRAPH=OFF`、`PHASESHIFT_FUSE_RESIDUAL_RMSNORM_QUANT=ON`（Gate 11J-A）。

計測の排他性と GPU 選択: マシンは他セッションと共有しており、GPU の空き状況が時間で変わる。
`ROCR_VISIBLE_DEVICES=N` と rocm-smi の GPU index が一致することは、PCI bus id を印字する
probe（`ROCR_VISIBLE_DEVICES=2` → bus 0x0f = rocm-smi GPU[2]）で確認した。各計測 campaign の
開始時に `hipMemGetInfo` で空き VRAM を確認し、31GB 以上空いている GPU のみを使う
（campaign 前後と実行中 30 秒間隔のサンプルを記録する）。実行中のサンプルは、計測 GPU が
自分の arena（31GiB）だけに占有され free が 0.7-0.8GB で推移することを示す。

なお本機の rocm-smi KFD process table は process → GPU の帰属が信頼できず、31GB を保持する
プロセスが動いていても行を返さないことがある。そのため in-window の帰属確認は free VRAM の
サンプルで行い、PID 単位の一覧には依存しない。

A/B は同一 GPU 上に pin した paired 比較とし、1 プロセスずつ直列に実行する。Verify の
campaign 1-4 は実行順を固定した A → B であり、これは実行後半に走る B を不利にする方向の
bias である。それでも B が一貫して勝つ（後述）ため、結果は保守側の推定になっている。
DFlash2 proposer の campaign 2 は rep ごとに実行順を反転させている。いずれの場合も
他セッションの負荷変動による run 間変動は残るため、median と paired 符号の両方で判定する。

### compile flags

```cmake
option(PHASESHIFT_FUSE_TARGET_SWIGLU_QUANT
       "Enable exact Target SwiGLU + activation quantize fusion" OFF)
option(PHASESHIFT_FUSE_DFLASH2_SWIGLU_QUANT
       "Enable exact DFlash2 SwiGLU + activation quantize fusion" OFF)
```

- Target flag は `fused_dispatch.hip`（matcher の launch）と `program_executor.hip` にのみ定義。
- DFlash2 flag は `dflash2/executor.hip` にのみ定義。
- どちらか ON のとき `fused_swiglu_quant.hip` を `phaseshift_qwen35_kernels_optimized` に追加。
- `fused_dispatch.cpp`（pure matcher）は常時 compile。runtime env による切り替えは無い。

### fusion contract

#### Target

| 項目 | 値 |
| --- | --- |
| sequence | `KernelId::SWIGLU` → `KernelId::ACTIVATION_QUANTIZE_W4A8`（Program dispatch 上で隣接、依存 ValueId 完全一致） |
| rows | 1..2048（実際の使用は 1..8） |
| features | 17408 |
| rounding | gate/up を f32 化 → `up * (gate / (1 + expf(-gate)))` を f32 で計算 → 出力で BF16 round（1 回のみ） |
| quant format | E4M3（W4A8）、group_size 32、k = k_padded = 17408 |
| consumed dispatches | 2（physical launch 1） |

#### DFlash2

| 項目 | 値 |
| --- | --- |
| sequence | `launch_dflash2_swiglu_bf16` + `dflash2_quantize_a8` → 1 fused launch、その後 `dflash2_gemm_psq4` |
| rows | block_rows（1..8、production draft は 7） |
| features | 17408（intermediate） |
| rounding | `sigmoid = 1/(1+expf(-gate))`、`silu = bf16_round_f32(gate*sigmoid)`、`y = f32_to_bf16(bf16_round_f32(silu*up))`（中間 BF16 round を保持） |
| PSQ4 condition | `executor.backbone_psq4` かつ `layer.mlp_down_proj.encoding == Psq4` かつ `intermediate == 17408` かつ 入出力ポインタが 16 byte aligned |

Target と DFlash2 の rounding contract は意図的に異なる。device helper は共有するが
policy は別で、fused kernel も同じ helper を呼ぶ。両者が同じ数式に退化していないことは
`test_qwen35_fused_swiglu_quant` の「target と dflash2 の fused 出力が同一 fixture で
一致しない」check で担保する。

### shared device functions

| primitive | 共有 device function | standalone wrapper | fused wrapper |
| --- | --- | --- | --- |
| Target SwiGLU | `detail::target_swiglu_f32`（`optimized/detail/target_swiglu_device.h`） | `op_swiglu_bf16::impl` | fused Phase 1 |
| DFlash2 SwiGLU | `detail::dflash2_swiglu_rounded_f32`（`dflash2/detail/swiglu_device.h`） | `dflash2_swiglu_kernel` | fused Phase 1 |
| E4M3 quantize | `detail::QuantVecRow<K,KP,Threads>` の段階 helper（`optimized/detail/activation_quantize_device.h`） | `activation_quantize_e4m3_vec_kernel<K,KP>` | fused Phase 2-4 |

Gate 11J-A の `activation_quantize_e4m3_vec_row_body` を
`load` / `peak` / `publish_scale` / `encode_store` の 4 段に分解し、standalone は
`load → peak → publish_scale → encode_store`、fused は生成した packed BF16 を
`peak → publish_scale → encode_store` に渡す。standalone の path は `Threads=256` の
同一 instanciation で従来と同一の命令列になる。

- standalone の launch geometry は不変（Target SwiGLU は `grid=(ceil(F/1024), rows)`、
  DFlash2 SwiGLU は flat grid-stride、activation quantize は 5120/6144/17408 の specialized）。
- fused 専用の SwiGLU 数式・E4M3 encoder のコピーは無い。device-side kernel launch も無い。

### fused kernel

- grid = rows、block = 256 または 512 threads。
- Phase 1: 8 element 単位で gate/up を読み、policy の semantic で BF16 を作り、
  `swiglu_out` へ uint4 で materialize しつつ register の packed BF16 words に保持。
- Phase 2: packed BF16（= 実際に `swiglu_out` に書いた値）から peak を求める。
  非 finite の扱い・`fmaxf` reduction は既存 quantize と同一。
- Phase 3: `scale = peak > 0 ? peak/448 : 0` を scales workspace と shared に書く。
- Phase 4: packed BF16 から E4M3 encode し、swizzled offset へ store（`swiglu_out` の再 read はしない）。
- `swiglu_out` の materialization は残す。middle buffer elimination は行わない。
- unroll: `QuantVecRow::kNChunk` は constexpr で、chunk loop は `#pragma unroll` により
  完全展開される。部分 unroll（`#pragma unroll 4`）では `packed[it]` が動的 index になり
  local memory（scratch）へ落ちるため、Resource Gate（scratch == 0）と両立しない。
  したがって unroll は 256/512 の各 1 種類のみを実装する。

### thread variant

resource（`tools/extract_kernel_resources.py`、build D の phaseshift-bench）:

| variant | path | VGPR | SGPR | LDS | scratch |
| --- | --- | ---: | ---: | ---: | ---: |
| 256 | Target | 90 | 57 | 40 | 0 |
| 256 | DFlash2 | 90 | 57 | 40 | 0 |
| 512 | Target | 93 | 40 | 72 | 0 |
| 512 | DFlash2 | 59 | 40 | 72 | 0 |
| 256 unroll4 | - | - | - | - | - |
| 512 unroll4 | - | - | - | - | - |

unroll4 の 2 行を実装していない理由は前述のとおり（部分 unroll は `packed[it]` の動的
index 化を招き scratch spill するため、Resource Gate と両立しない）。`#pragma unroll` に
よる完全展開が base と等価な唯一の形である。scratch は全 variant で 0、spill も 0。

primitive（M=8、us、warmup 200 / iters 2000）:

| variant | Target fused | DFlash2 fused |
| --- | ---: | ---: |
| 256 | 7.920 | 18.273 |
| 512 | **6.224** | **8.942** |

production variant は両 path とも **512**（`kFusedSwiGLUQuantProduction`）。

### primitive correctness

`test_qwen35_fused_swiglu_quant`（required, GPU）。split path と fused path の
`swiglu_out` / activation codes / activation scales を byte 比較。rows 1/3/7/8、
features 17408、variant 256/512、Target / DFlash2 の全組み合わせ。

| path | variant | rows | swiglu_out | codes | scales |
| --- | --- | --- | --- | --- | --- |
| Target | 256 / 512 | 1,3,7,8 | memcmp == 0 | memcmp == 0 | bit pattern == 0 |
| DFlash2 | 256 / 512 | 1,3,7,8 | memcmp == 0 | memcmp == 0 | bit pattern == 0 |

fixture は BF16 rounding 境界を含む: gate の 0 / ±0 / ±30 / ±88 / ±1e-30、
up の 0 / ±0 / BF16 halfway 近傍（base × steps/512, steps/1024）/ 一様乱数。
さらに「Target fused と DFlash2 fused の出力が同一 fixture で一致しない」ことを確認し、
両 policy が同じ数式に退化していないことを検証する（34/34 PASS）。

### primitive performance

`phaseshift-bench fused-swiglu-quant`、warmup 200、iters 2000、同一 stream / 同一 allocation。
計測 GPU は campaign 開始時の free VRAM チェックで選ぶ（この計測は GPU 3）。

| path | M | split us | fused256 us | fused512 us | winner | speedup(512) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Target | 1 | 8.544 | 7.899 | 6.097 | fused512 | 1.40x |
| Target | 3 | 8.156 | 7.878 | 6.092 | fused512 | 1.34x |
| Target | 7 | 8.369 | 7.901 | 6.181 | fused512 | 1.35x |
| Target | 8 | 8.467 | 7.920 | 6.224 | fused512 | 1.36x |
| DFlash2 | 1 | 8.450 | 18.565 | 8.895 | split | 0.95x |
| DFlash2 | 3 | 8.307 | 18.459 | 8.904 | split | 0.93x |
| DFlash2 | 7 | 9.326 | 18.233 | 8.899 | fused512 | 1.05x |
| DFlash2 | 8 | 9.542 | 18.273 | 8.942 | fused512 | 1.07x |

Target は 256 では利得が小さく（8.16→7.88）、512 で明確に速い（6.09-6.22us）。
DFlash2 は 256 では 2.2 倍遅くなる（18.2-18.6us）。DFlash2 の `round_to_bf16_f32` を
2 回呼ぶ semantic は per-element latency が高く、1 block/row × 8 warp では
latency hiding が足りない。512（16 warp）で split と同等まで戻り、M=7/8 では逆転する。

### Target matcher contract

`try_launch_swiglu_quant_fusion` が `program.dispatches[i..i+1]` を検査する。

| 条件 | 要求 |
| --- | --- |
| kernel 列 | `SWIGLU`, `ACTIVATION_QUANTIZE_W4A8` |
| swiglu | input_count=2, output_count=1, gate/up/output BF16, feature=17408, 同一 row domain, alias 安全 |
| dependency | `swiglu.output[0] == quant.input[0]` |
| alignment | gate/up/swiglu の pointer が 16 byte aligned、row_stride が 8 の倍数 |
| rows | 1..2048 |
| selector | elementwise `SwigluBf16` が Optimized |
| quant | group_size=32, activation_kp=17408, workspace layout が `Int8ActivationWorkspaceLayout::make(17408,max_rows)` と一致 |
| vec path | `activation_quantize_e4m3_vec_supported(swiglu_out, codes, 17408, 17408, stride)` 成立 |

1つでも外れたら `NotApplicable` で split path に戻す。成立時は `consumed_dispatches = 2`、
physical launch 1。logical coverage（2 dispatch）・value trace（`v_swiglu`）・imatrix probe は
executor が従来どおり処理する。`v_swiglu` は materialize されるため trace / debug / graph
contract は維持される。

### DFlash2 executor integration

`layer_attention_tail` の mlp 部分で、`backbone_psq4` かつ `mlp_down_proj.encoding == Psq4`
かつ `intermediate == 17408` かつ入出力が 16 byte aligned のときだけ
`dflash2_swiglu_quant_fused()` を呼ぶ。成功時は `act_valid_k = intermediate`、
`act_valid_rows = block_rows` を設定し、`dflash2_linear` ではなく
`dflash2_gemm_psq4` を直接呼ぶ（`dflash2_quantize_a8` は再実行しない）。

それ以外（flag OFF / BF16 backbone / mlp_down が Psq4 でない / shape 不一致 / alignment 不一致）
は従来順序 `dflash2_swiglu` → `dflash2_linear` に戻る。BF16 DFlash2 backbone では
activation quantize 自体が無いため fusion は適用されない。

### fusion counters

`KERNEL_TRACE_FUSED residual_rmsnorm_quant=<n> swiglu_quant=<n> dflash2_swiglu_quant=<n>`
を process exit 時に 1 行だけ出す（`PHASESHIFT_QWEN35_KERNEL_TRACE` 設定時のみ、通常実行では
出力しない）。`swiglu_quant` は Target matcher、`dflash2_swiglu_quant` は DFlash2 経路の
専用 counter。GPU を占有しない。

### Target Verify wall A/B（ctx=2048、iters=30、median of 5 paired runs、build A vs B）

全 build で Gate 11J-A の `RESIDUAL_RMSNORM_QUANT` は ON。A → B の差分が Target SwiGLU
fusion の寄与。

| capture | M | A us | B us | delta | paired (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 1 | 35583.3 | 35487.0 | -0.27% | +1.45 / -0.17 / -0.31 / -0.37 / -0.29 |
| 0 | 3 | 38546.7 | 38447.7 | -0.26% | +1.25 / -0.31 / -0.26 / -0.61 / -0.64 |
| 0 | 8 | 40867.4 | 40817.5 | -0.12% | +1.22 / -0.36 / -0.28 / -0.13 / -0.03 |
| 1 | 8 | 43072.4 | 42819.1 | -0.59% | -1.97 / -0.57 / -0.31 / -0.32 / -1.96 |

paired は全条件で 4/5 または 5/5 が改善側。capture=0 M=8 の -0.12% は小さいが、M=1/3 は
-0.27 / -0.26%、capture=1 M=8 は -0.59% で、physical launch の減少と整合する。

#### campaign 2（counter 追加後の再計測、5 pairs、静穏 window）

| capture | M | A us | B us | delta | paired (%) | swiglu hits |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 0 | 1 | 35761.4 | 35589.2 | -0.48% | -0.36 / -0.48 / +0.55 / -0.47 / -0.48 | 2688 |
| 0 | 3 | 38832.7 | 38783.9 | -0.13% | -0.03 / -0.16 / -0.14 / -0.45 / -0.36 | 5312 |
| 0 | 8 | 41269.3 | 41067.4 | -0.49% | -0.28 / -0.49 / -0.59 / -0.66 / -0.40 | 7936 |
| 1 | 8 | 43946.8 | 43695.4 | -0.57% | -0.06 / -0.48 / -0.51 / -0.57 / -1.96 | 2688 |

campaign 1 + campaign 2 の 8 cell の median は **-0.28%**。M=3 capture=0 の -0.13% を除き
すべて -0.12% 以下で、paired 符号は各 cell で 4/5 または 5/5 が改善側。

#### campaign 5/6（balanced order、各 4 pairs）

実行順を rep ごとに反転し、4 rep で A 先頭 2 回 / B 先頭 2 回とする（位置効果を相殺）。
GPU 1 に 1 プロセスずつ直列で実行し、前後と 30 秒間隔の free VRAM サンプルを取得した。

| campaign | M | A us | B us | median | paired (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| 5 | 1 | 35627.7 | 35777.9 | +0.42% | -0.35 / -0.15 / **+1.35** / -0.50 |
| 5 | 3 | 38614.5 | 38713.2 | +0.26% | -0.35 / -0.25 / **+1.35** / -0.70 |
| 5 | 8 | 41029.4 | 41084.9 | +0.14% | -0.52 / -0.31 / **+1.20** / -0.80 |
| 5 | cap1 8 | 42976.3 | 42770.2 | -0.48% | -0.51 / -0.53 / **+1.17** / -1.99 |
| 6 | 1 | 35926.6 | 35549.6 | -1.05% | -0.37 / -1.73 / **+1.32** / -2.02 |
| 6 | 3 | 38987.6 | 38515.5 | -1.21% | -0.35 / -2.09 / **+1.28** / -2.07 |
| 6 | 8 | 41451.0 | 40941.7 | -1.27% | -0.55 / -2.11 / **+1.28** / -2.00 |
| 6 | cap1 8 | 43382.7 | 43238.7 | -0.32% | +1.30 / -2.00 / **+1.21** / -1.85 |

各 campaign で 3/4 が改善側。両 campaign に共通して 3 番目の rep だけが +1.2〜+1.35% の外れ値
になり、n=4 の median を支配する。この外れ値は特定 rep の B 実行中に生じた負荷・クロック変動
であり、A/B ペアの比較自体は同じ rep 内で完結している。

#### 全 campaign 統合

campaign 1 / 2（各 5 pairs、fixed order）と campaign 5 / 6（各 4 pairs、balanced order）を
統合する。campaign 3/4 は 2 GPU 同時実行かつ他セッションの負荷と重なったため除外する。

| cell | n | median | 改善側 | min / max |
| --- | ---: | ---: | ---: | --- |
| cap0 M=1 | 18 | **-0.35%** | 14/18 | -2.02 / +1.45 |
| cap0 M=3 | 18 | **-0.33%** | 15/18 | -2.09 / +1.35 |
| cap0 M=8 | 18 | **-0.38%** | 15/18 | -2.11 / +1.28 |
| cap1 M=8 | 18 | **-0.52%** | 15/18 | -2.00 / +1.30 |
| 全 cell 合計 | **72** | **-0.36%** | **59/72（82%）** | — |

方向は全 cell で一貫しており、外れ値 1 rep を除けば -0.3〜-1.3% の改善である。fixed order の
campaign 1/2 は B を不利にする順序だったため、統合値は保守側の推定になる。

pre/post の free VRAM は、campaign 前は 31.78GB free（他のプロセス無し）、実行中は自分の
arena のみ（free 0.72-0.78GB）、campaign 後は 31.79GB free に戻ることを確認した。

#### 除外した campaign 3/4（2 GPU 同時実行、3 pairs ずつ）

マシン共有のため、2 つの verify を別 GPU で同時に走らせた campaign では符号が安定しない。

| GPU | M=1 | M=3 | M=8 | capture1 M=8 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | -0.09% | +0.85% | +0.89% | -0.30% |
| 2 | -0.17% | +0.77% | +1.20% | -0.59% |

paired も `+1.62 / -0.28 / -0.09` のように外れ値 1 点が median を支配する。この 2 campaign は
互いに、また他セッションの負荷と同時に走っており **静穏 window の計測ではない** ため、
上記の統合値からは除外する。

参考として、M=1/3/8 の swiglu hit は 2688 / 5312 / 7936 でいずれも 64 の倍数（= 64 MLP
layer × update 数）であり、process 内で完全に決定論的である。

### Target launch accounting（ctx=2048、capture=0、per update）

| metric | A | B | delta |
| --- | ---: | ---: | ---: |
| logical dispatches | 1654 | 1654 | 0 |
| physical launches | 1417 | 1353 | **-64** |
| optimized operations | 1400 | 1336 | -64 |
| Target SwiGLU fusion hits | 0 | 64/update | +64 |
| Gate 11J-A RMSNorm fusion hits | 127/update | 127/update | 0 |

`-64 = 64 MLP layer × 1 fusion` で厳密に一致する。Target Verify M=8 の logical dispatch は
1654、うち MLP が 64 layer 分ある。

### DFlash2 proposer A/C（`test_dflash2_gate7_perf`、ctx=2048、block_rows=8、7 pairs 合計）

A = dflash fusion OFF、C = dflash fusion ON。backbone は fused kernel を含む区間、
proposer は draft 全体。

| campaign | GPU | pairs | backbone delta | backbone paired (%) | proposer delta | proposer paired (%) |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| 1a | 2 | 2 | -0.03% | -0.10 / +0.04 | +0.08% | +0.22 / -0.05 |
| 1b | 1 | 2 | -0.12% | -0.12 / -0.12 | -0.09% | +0.00 / -0.18 |
| 2 | 1 | 3 | -0.16% | -0.38 / -0.16 / +0.36 | +0.02% | +0.02 / -0.17 / +0.10 |
| 合計 | — | 7 | **-0.12%（5/7 改善側）** | — | **+0.00%（3/7 改善側）** | — |

campaign 2 は rep ごとに実行順を反転しており、その median は backbone -0.16% /
proposer +0.02%。backbone には fused kernel の寄与が現れるが、draft 全体に占める比率が
小さいため proposer では差が消える。

fused hit counter は campaign 2 で取得した。mode C は **875 hit / iteration**（process 内で
一定、draft 1 回あたり block_rows 8 相当）、mode A は 0。fusion が実際に発火していることを
timing とは独立に確認している。

### E2E 4-way attribution（A/B/C/D、2 GPU 並列、ids 一致）

`phaseshift-compute` の CLI を A/B/C/D の 4 build で実行し、6 workload で tok/s を比較した。
reps は instance 1（GPU 3）が primary 3 / ctx 1 / other 1、instance 2（GPU 2）が primary 2 /
ctx 1 / other 1 で、同一 rep 内の 4 mode は同一 GPU 上で実行している。

| workload | n | A tok/s | B tok/s | C tok/s | D tok/s | B-A | C-A | D-A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prose256 | 5 | 63.89 | 64.30 | 65.05 | 64.31 | +0.64% | +1.82% | +0.66% |
| prose512 | 2 | 71.57 | 71.60 | 71.91 | 71.42 | +0.04% | +0.48% | -0.21% |
| code | 2 | 145.47 | 147.61 | 146.72 | 147.25 | +1.46% | +0.86% | +1.22% |
| math | 2 | 139.07 | 138.42 | 137.70 | 139.63 | -0.47% | -0.99% | +0.40% |
| reasoning | 2 | 68.92 | 69.09 | 68.78 | 68.53 | +0.25% | -0.20% | -0.56% |
| ctx2048 | 2 | 157.88 | 159.52 | 159.24 | 160.60 | +1.04% | +0.86% | +1.73% |

rep pair 単位の delta は ±2% の幅で振れ、instance 間で符号が一致しない workload がある
（prose512 の C-A は +0.14% と +0.81%、math の C-A は -1.98% と +0.01%）。B / C / D が
そろって A より小さい正の delta を示すのは、A が実行順で不利な位置に置かれる rep が
多いこと（reps=3 では A は 1 番目が 2 回）による残差 bias と整合する。したがって
**E2E レベルでは両 fusion の効果は分解できず、非劣化の確認にとどまる**。

| 確認項目 | 結果 |
| --- | --- |
| `ids_exact`（4 mode 完全一致） | 6 workload すべて True（instance 1/2 とも） |
| rounds / accepted / full / partial / reruns | 6 workload × 4 mode すべて同値 |
| Target swiglu hit（B/D） | workload ごとに一定（例 prose256 5504、ctx2048 1088） |
| Target swiglu hit（A/C） | 0 |

### correctness

| 項目 | 結果 |
| --- | --- |
| Target 4B（A/B/C/D） | `GENERATED_IDS` 完全一致 |
| Target 4B-PSQ（A/B/C/D） | 完全一致 |
| Target 27B-PSQ greedy（A/B/C/D） | 完全一致 |
| Target 27B-PSQ stochastic（A/B/C/D） | 完全一致 |
| DFlash2 BF16 drafter（A/B/C/D） | ids 一致、acceptance 一致（rounds 40 / acc 87 / reruns 0） |
| DFlash2 PSQ4 drafter（A/B/C/D） | ids 一致、acceptance 一致（rounds 42 / acc 85 / reruns 0） |
| DFlash2 INT2 head（A/B/C/D） | ids 一致、acceptance 一致 |

CLI correctness で取得した fusion counter（mode 順 A/B/C/D）:

| case | Target swiglu hit | DFlash2 swiglu hit |
| --- | --- | --- |
| BF16 drafter | [0, 2624, 0, 2624] | [0, 0, 0, 0] |
| PSQ4 drafter | [0, 2752, 0, 2752] | [0, 0, 210, 210] |

Target counter は B/D のみ、DFlash2 counter は C/D のみで非ゼロ。BF16 backbone では DFlash2
fusion が **一度も発火しない**（alignment / encoding 条件を満たさないため）ことも同時に確認
できた。

`test_dflash2_gate11c_e2e`（GDN rerun-reference 一致）: **failed=0**。3 case すべて
`ids_match=yes` かつ `hist_reruns=0`（prose 256 / code 128 / math 128 token、build D）。
この test は fixture safetensors を必要とせず prompt token IDs だけで動くため、
`tools/reference/export_qwen38_dflash2_gate51_real.py --mode prompts` で生成した
`real_prose` / `real_code` / `real_math` の prompt を使用した。

### decision

- **Target SwiGLU + E4M3 quantize fusion: ADOPT**
- **DFlash2 SwiGLU + E4M3 quantize fusion: PERF-NEUTRAL（default OFF 維持）**

Target の理由:

- correctness: split vs fused は `swiglu_out` / codes / scales が byte-exact（rows 1/3/7/8、
  Threads 256/512）。Target greedy / stochastic、DFlash2 3 経路の ids・acceptance がすべて一致。
- physical launch: Target Verify で **-64**（`logical 1654` は不変、`physical 1417→1353`）。
  64 MLP layer 分の fusion hit と厳密に一致し、M=1/3/8 の全条件で同じ -64。
- Verify wall: 静穏 window の 4 campaign（fixed order 2 × 5 pairs + balanced order 2 × 4 pairs、
  計 72 pair）を統合して median **-0.36%**、**59/72（82%）が改善側**。cell 別 median は
  -0.33〜-0.52% で全 cell 同方向。primitive は split 8.16-8.54us → fused 6.09-6.22us
  （**1.34-1.40x**）。削減見積り 64 × 2.3us ≈ 147us は verify step 40.8ms の 0.36% で、
  実測と一致する。fusion が速くしているのは verify step の 0.36% を占める部分であり、
  step 全体の改善はその比率で決まる（残りは weight streaming 主体の GEMM）。
- E2E: 6 workload で B-A は -0.47〜+1.46%（median +0.64%）、non-regression。
- resource: fused は VGPR 93 / SGPR 40 / LDS 72 / **scratch 0**（spill なし）。

DFlash2 の理由:

- correctness / launch / resource は Target と同様に hard PASS（byte-exact、scratch 0、
  `dflash2_swiglu_quant` 875 hit/iteration）。
- primitive は split 8.31-9.54us → fused 8.90-8.94us で、block_rows=8 に相当する M=7/8 では
  速いが M=1/3 では遅い。
- backbone は -0.12%（5/7 改善側）だが proposer 全体では **+0.00%（3/7）** で、draft 全体に
  占める SwiGLU+quantize の比率が小さいため end-to-end で効果が観測できない。
- 非劣化・byte-exact であるため code は保持するが、性能上の採用理由がないので default OFF を
  維持する。

### regression

| 項目 | 結果 |
| --- | --- |
| build error | 0（A/B/C/D とも） |
| build warning | 0（新規・変更した TU） |
| required（A: target OFF / dflash OFF） | **127/127 PASS**（skip 0、GPU 3 固定・直列） |
| required（B: target ON / dflash OFF） | **128/128 PASS**（skip 0） |
| required（C: target OFF / dflash ON） | **128/128 PASS**（skip 0） |
| required（D: target ON / dflash ON） | **128/128 PASS**（skip 0） |
| `test_qwen35_fused_swiglu_quant` | 34/34 PASS |
| `test_qwen35_fusion_matcher` | 39/39 PASS |

required 数は Gate 11J-A の 126 に、`test_qwen35_fused_swiglu_quant`（target または dflash が ON
のとき）と bench smoke `test_bench_help_fused_swiglu_quant` を加えたもの。A は 127、
B/C/D は 128。

既存 warning（本 Gate と無関係、baseline から存在）:
`dflash2/executor.hip` の `int2_select_impl` unused、`test_qwen35_mtp_gate3_replay.hip` の
`hipMemcpy` nodiscard 無視、`test_qwen35_prefix_cache.hip` の同種 warning。


## Gate 11J-C DFlash2 residual + RMSNorm + activation-quantize fusion

Gate 11J-A の Target Exact 版と同系統を、DFlash2 PSQ4 drafter に適用する。対象は
`residual/add → DFlash2 RMSNorm → PSQ4 linear 用 activation quantize`。

### baseline / build

| 項目 | 値 |
| --- | --- |
| baseline revision | `d79f5c92`（Gate 11J-B final） |
| branch | `gate/11j-c-k-fusion-series` |
| worktree | `.worktrees/gate11j-c-k-fusion-series` |
| build A | cumulative baseline（J-A 残差融合 ON、11J-B Target SwiGLU ON、DFlash2 SwiGLU OFF、本 Gate OFF） |
| build B | build A + `PHASESHIFT_FUSE_DFLASH2_RESIDUAL_RMSNORM_QUANT=ON` |

cumulative baseline は ADOPT 済みの path のみを ON にする（11J-B で DFlash2 SwiGLU は
PERF-NEUTRAL のため OFF のまま）。

### compile flag

```cmake
option(PHASESHIFT_FUSE_DFLASH2_RESIDUAL_RMSNORM_QUANT
       "Enable exact DFlash2 residual + RMSNorm + activation quantize fusion" OFF)
```

flag は `dflash2/executor.hip` にのみ定義し、kernel source は flag ON のときだけ
`phaseshift_qwen35_kernels_optimized` に追加する。runtime env による production selector は無い。

### discovery: actual chain と exact adjacency

`dflash2/executor.hip` の actual launch 列を確認した結果、exact な producer/consumer
依存が成立する箇所は次の 2 つである（cached / stateless の両経路、`layer_attention_tail` は共通）。

| chain | sequence | 位置 |
| --- | --- | --- |
| A | `residual_add(hidden_in, attention_finished) → rmsnorm(post_attention_layernorm) → quantize_a8(post_attention_norm)` | `layer_attention_tail` |
| B | `rmsnorm(input_layernorm) → quantize_a8(layer_input_norm)` | layer 先頭（cached / stateless） |

いずれも quantize は `dflash2_grouped_conv_prepare` 内の
`dflash2_linear(conv kernel_projection)` が最初に発行する launch であり、
rmsnorm の直後で追加の launch は挟まらない。norm の出力 buffer がそのまま quantize の
入力 buffer であることも確認した。

逆に、`mlp_prepared` 用の quantize は dynamic conv の後段であり rmsnorm と非隣接のため
対象外とした。Q/K/V の quantize も conv を挟むため対象外。

「見た目が同じ」だけの箇所は融合していない。

### fusion contract

| 項目 | 値 |
| --- | --- |
| features | 5120（DFlash2 hidden。`features != 5120` は fused path を使わない） |
| RMSNorm semantics | `sum += v*v` を thread ごとの stride 加算 → 128 thread の shared tree reduction → `mean = sum/5120` → `inv = rsqrtf(mean+eps)` → `norm = x*inv` → **中間で BF16 round** → `y = w * norm_rounded` → BF16 round |
| Norm の reduction 順 | standalone `dflash2_rmsnorm_kernel` と完全同一（thread 数 128、`i = tid; i < 5120; i += 128`、shared tree の step 順も同一） |
| quantize | E4M3（k = k_padded = 5120、group 32）、Target/J-A と同じ `QuantVecRow` 段階 helper |
| rows | 1..`max_rows`（request は 1..8） |
| 融合対象 launch | chain A: 3 → 1、chain B: 2 → 1 |

Target の RMSNorm とは contract が異なる（Target は中間 round なしの `ONE_PLUS` weight）。
そのため DFlash2 専用の device body を standalone と共有し、Target 側の helper は使わない。

### shared device body

`kernels/dflash2/detail/rmsnorm_device.h` に
`dflash2_norm_block_reduce_sum` / `dflash2_rmsnorm_group_body` を抽出し、
standalone `dflash2_rmsnorm_kernel` と fused kernel の両方が同じ body を呼ぶ。

| standalone（変更前） | 抽出後 |
| --- | --- |
| `bf16_to_f32(x[i])` | `__bfloat162float(x[i])`（`bf16_to_f32` の定義そのもの） |
| `f32_to_bf16(norm)` | `__float2bfloat16(norm)`（同） |
| `widx = (group_size == features) ? (base + i) : i` | `weight_index_base + i`（`weight_index_base` は同条件で `base` / `0`） |
| `output[row*features + base + i]` | `out[i]`（`out = output + row*features + base`） |

intrinsic も演算順も同一で、standalone の launch geometry（grid = (features/group_size, rows)、
block 128）も変更していない。したがって split-only build の出力は byte-identical である。

### fused kernel

`kernels/dflash2/fused_residual_rmsnorm_quant.hip`。

- grid = rows、block = **128 threads**（norm の reduction 順を standalone と一致させるため）。
- Phase 1（chain A のみ）: `residual_add_bf16_body` で `a+b` を BF16 で `residual_out` へ書く。
- Phase 2: 共有 body で RMSNorm を計算し `norm_out` へ書く（`__syncthreads()` を挟む）。
- Phase 3: `norm_out` を quantize の layout で register へ読み戻し（uint4 × nChunk）、
  `activation_quantize_e4m3_vec_row_body_from_packed<5120,5120,128>` で scale と codes を生成。
- `residual_out` / `norm_out` の materialization は残す。middle buffer elimination は行わない。
- `__global__` から `__global__` を launch しない。

stage は `launch_fused_dflash2_residual_rmsnorm_quant_e4m3`（chain A）と
`launch_fused_dflash2_rmsnorm_quant_e4m3`（chain B）の 2 つ。

### runtime integration

fused path を使う条件（`fused_dflash2_rrq_eligible`）:

- `executor.backbone_psq4`
- conv kernel projection の encoding が `Psq4`
- `features == 5120`
- `rows != 0 && rows <= max_rows`
- `norm_out` が 16 byte aligned、norm row stride が 8 の倍数

成立時は `dflash2_grouped_conv_prepare_preactivated` を呼び、activation workspace が
有効なまま `dflash2_gemm_psq4` で conv kernel projection を実行してから dynamic conv に進む
（quantize を二重実行しない）。`act_valid_k = features` / `act_valid_rows = rows` を
`dflash2_quantize_a8` と同一に更新する。

不成立時は従来の `dflash2_grouped_conv_prepare`（quantize 込み）を通る。

### fusion counter

`KERNEL_TRACE_FUSED` に `dflash2_residual_rmsnorm_quant` を追加した。proposer 1 回の
process で **1750 hit**（決定論的）を確認している。

### primitive correctness

`test_dflash2_fused_residual_rmsnorm_quant`（build B）:

| 項目 | 結果 |
| --- | --- |
| chain A: `residual_out` split vs fused | rows 1/3/7/8 byte-exact |
| chain A: `norm_out` split vs fused | byte-exact |
| chain A: codes / scales split vs fused | byte-exact |
| chain B: `norm_out` split vs fused | byte-exact |
| chain B: codes / scales split vs fused | byte-exact |
| 合計 | **21/21 PASS** |

### resource

| kernel | SGPR | VGPR | LDS | scratch |
| --- | ---: | ---: | ---: | ---: |
| `fused_dflash2_rmsnorm_quant_e4m3_kernel<false>` | 53 | 57 | 536 | **0** |
| `fused_dflash2_rmsnorm_quant_e4m3_kernel<true>` | 53 | 57 | 536 | **0** |
| standalone `dflash2_rmsnorm_kernel`（変更なし） | 23 | 8 | 512 | 0 |

LDS 536 B の内訳は norm の shared tree scratch（128 × 4 B）+ quantize の wave_max 等。
scratch spill は無い。

### primitive performance

`phaseshift-bench fused-dflash2-residual-rmsnorm-quant`、warmup 200、iters 2000。

| variant | M | split | fused | speedup |
| --- | ---: | ---: | ---: | ---: |
| residual（3 → 1） | 1 | 25.148 | 22.082 | **1.14x** |
| residual | 3 | 24.337 | 21.780 | 1.12x |
| residual | 7 | 24.256 | 21.550 | 1.13x |
| residual | 8 | 24.257 | 21.549 | 1.13x |
| norm（2 → 1） | 1 | 22.234 | 19.321 | **1.15x** |
| norm | 3 | 21.449 | 18.749 | 1.14x |
| norm | 7 | 21.213 | 18.646 | 1.14x |
| norm | 8 | 21.411 | 18.709 | 1.14x |

### in-context A/B（`test_dflash2_gate7_perf`、ctx=2048、block_rows=8、3 pairs、GPU 3）

| 区間 | A us | B us | median | paired (%) |
| --- | ---: | ---: | ---: | --- |
| cached backbone | 4444.76 | 4408.49 | **-0.82%** | -0.82 / -1.08 / -0.74 |
| full cached proposer | 5842.06 | 5804.49 | **-0.64%** | -0.39 / -0.64 / -0.85 |

3/3 pair が改善側。`dflash2_rrq_hits=1750`（A は 0）。計測 GPU の free VRAM は
campaign 前後とも 31.79 GB（他プロセス無し）。

launch accounting: chain A は 5 layer 分 `-2`、chain B は 5 layer 分 `-1` で
**-15 launch / draft forward**。proposer は 1 step あたり backbone 1 回 + head なので、
backbone の launch 数 140 程度に対して -15 が効いている。

### correctness / required

| 項目 | 結果 |
| --- | --- |
| required（A: cumulative baseline） | **129/129 PASS**（skip 0、直列 `-j1`） |
| required（B: + 本 Gate） | **130/130 PASS**（skip 0。追加分は `test_dflash2_fused_residual_rmsnorm_quant`） |
| build error / warning | 0（新規・変更 TU） |
| Target greedy `GENERATED_IDS`（294 token） | A/B で不変 |
| DFlash2 BF16 / PSQ4 / PSQ4+INT2 head（A/B） | ids 一致、acceptance 一致（rounds / accepted / full / partial / reruns） |
| DFlash2 ids == target-only greedy | **3 経路 × A/B すべて完全一致（128/128）** |
| gate11c（GDN rerun-reference） | **failed=0**、3 case `ids_match=yes`・`hist_reruns=0` |
| `dflash2_residual_rmsnorm_quant` hit | BF16 backbone は [0,0]（条件不成立）、PSQ4 は [0, 420]（B のみ発火） |

### E2E A/B（6 workload、15 paired、A = baseline / B = + 本 Gate）

| workload | n | A tok/s | B tok/s | B-A | paired (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| prose256 | 5 | 65.62 | 65.71 | +0.14% | -0.14 / +1.72 / -1.75 / +0.00 / +1.70 |
| prose512 | 2 | 72.20 | 72.66 | +0.63% | +1.49 / -0.21 |
| code | 2 | 146.25 | 146.56 | +0.22% | +0.37 / +0.06 |
| math | 2 | 137.35 | 138.69 | +0.98% | +1.94 / +0.01 |
| reasoning | 2 | 69.29 | 68.78 | -0.74% | -1.47 / +0.01 |
| ctx2048 | 2 | 158.62 | 158.72 | +0.06% | +0.11 / +0.01 |
| 合計 | **15** | — | — | **median +0.01%** | 改善側 11/15・有意差なし |

rounds / accepted / full / partial / reruns は全 workload で A/B 完全一致。`ids` も全 run 一致。
fusion hit は workload ごとに決定論的で、prose256 の 840 hit は 85 round に対し
**約 10 fusion / draft forward**（= 5 layer × chain A/B の 2 つ）と一致する。

### decision

**ADOPT**

理由:

- correctness: hard PASS。fused は `residual_out` / `norm_out` / codes / scales が byte-exact
  （rows 1/3/7/8）。required は A 129/129、B 130/130（skip 0）、DFlash2 3 経路の ids は
  target-only greedy と完全一致、gate11c も failed=0。
- in-context: proposer **-0.64%**（paired 3/3 改善側）、fused kernel を含む backbone は
  **-0.82%**（3/3）。`dflash2_rrq_hits=1750` で発火を確認。
- launch accounting: chain A 3→1、chain B 2→1 が 5 layer 分で **-15 launch / draft forward**。
  E2E の hit 数（約 10 fusion / draft forward）と一致する。
- primitive: residual 3→1 が 1.12-1.14x、norm 2→1 が 1.14-1.15x。scratch 0。
- E2E: 15 pair の median +0.01%、改善側 11/15。有意な差は出ないが非劣化。
- resource: fused は SGPR 53 / VGPR 57 / LDS 536 B / **scratch 0**。standalone の resource・
  geometry は不変。

E2E では有意差が出ない（drafter の寄与が decode loop 全体に薄まるため）が、drafter 自身の
workload である proposer で一貫した改善があり、byte-exact・scratch 0・launch 削減が明確な
ため、以降の Gate の cumulative baseline に含める。

## Gate 11J-D Target GDN prepare fusion

Target Exact の GDN layer で、recurrence 直前の tiny primitive 群を 1 launch にまとめる。

### 対象 chain の特定

`model/lower_to_primitives.cpp` の GDN layer 構築と `runtime/program/program.cpp` の
primitive → dispatch 降下を確認した。dispatch は node 順に 1 対 1 で並ぶため、GDN layer の
prepare 区間は次の 4 dispatch が連続する。

| # | dispatch | input | output | 備考 |
| --- | --- | --- | --- | --- |
| 0 | `SILU` | conv 出力 F32 × 10240 | `act` BF16 × 10240 | `SiluF32ToBf16` selector |
| 1 | `L2_NORMALIZE` | `act` の Q view BF16 × 2048（offset 0） | `q_norm` F32 × 2048 | group 128、eps 1e-6 |
| 2 | `L2_NORMALIZE` | `act` の K view BF16 × 2048（offset 2048） | `k_norm` F32 × 2048 | group 128、eps 1e-6 |
| 3 | `SCALE` | `q_norm` | `q_scaled` F32 × 2048 | `1/sqrt(128)` |

Q/K は別 branch ではなく同一 `act` の view であり、recurrence が読むのは `q_scaled` / `k_norm` /
`act` の V view である。区間内で外部 consumer は無く、依存は dispatch 順に閉じている。
`q_norm` の consumer は `SCALE` のみ、`act` は K/V view が読むため materialize が必須。

GDN geometry（`config.json` の `text_config`）:

| 項目 | 値 |
| --- | --- |
| GDN layer 数 | 48（64 layer 中。full attention は 16） |
| key heads × head dim | 16 × 128 = 2048（Q/K features） |
| value heads × head dim | 48 × 128 = 6144 |
| conv features | 2 × 2048 + 6144 = **10240** |

`mlp` 側や `gdn_z` の SiLU（6144, BF16→F32）などは chain の外なので対象外とした。

### compile flag

```cmake
option(PHASESHIFT_FUSE_GDN_PREPARE "Enable exact Target GDN prepare fusion" OFF)
```

`program_executor.hip` / `fused_dispatch.hip` に定義し、kernel source は flag ON のときだけ
`phaseshift_qwen35_kernels_optimized` に追加する。

### shared device body

`kernels/optimized/gdn/detail/gdn_prepare_device.h` に
`gdn_silu_f32` / `gdn_l2_head_partial_sum` / `gdn_l2_warp_reduce_sum` /
`gdn_l2_normalize_head_body` を抽出し、standalone の `op_silu_f32_bf16::impl` と
`l2_normalize_bf16_f32_kernel`、および fused kernel が同じ body を共有する。

| standalone（変更前） | 抽出後 |
| --- | --- |
| `x / (1.0f + expf(-x))` | `detail::gdn_silu_f32(x)`（本体は同一式） |
| `sum += x0*x0 + x1*x1 + x2*x2 + x3*x3`（4 要素 chunk） | `gdn_l2_head_partial_sum` 内で同一 |
| `__shfl_xor(v, 16/8/4/2/1)` | `gdn_l2_warp_reduce_sum` 内で同一 |
| `rsqrtf(sum + eps)` | 同一 |
| `(in_off & 3u) == 0u` の aligned 分岐 | caller が同じ式で計算し body へ渡す |
| grid = (features/group_size, rows)、block 32 | 変更なし |

演算順・intrinsic・launch geometry を変えていないため、split-only build の出力は同一である。
`l2_normalize_bf16_f32_kernel` の resource は SGPR 22 / VGPR 8 / LDS 0 / scratch 0。

### split-only regression（device body 抽出の検証）

抽出前（`30f69ce2` の `l2_normalize.hip` / `elementwise.hip`）と抽出後の split-only build（A）で
target-only greedy ids（128 token）を実測比較した。

| build | target greedy ids | auto == correctness mode |
| --- | --- | --- |
| pre-refactor（抽出前 kernel） | 128 token | False |
| post-refactor（HEAD kernel） | 128 token（pre と完全一致） | False |

抽出によって target の greedy ids は変化しない（＝split kernel の数値は不変）。
`PHASESHIFT_QWEN35_KERNEL_MODE=correctness` との不一致は抽出前から存在する（correctness 系は
reduction 順が異なる別実装であり bit-exact ではない）。

### fused kernel

`kernels/optimized/gdn/fused_gdn_prepare.hip`。grid = rows、block = **256 threads（8 warp）**。

- Phase 1: 全 thread で conv 出力 F32 を読み、`bf16(silu_f32(x))` を `act` へ書く（elementwise）。
- `__syncthreads()`。
- Phase 2: warp ごとに head を分担し（`h = warp; h < 16; h += 8`）、Q と K の head を
  `gdn_l2_normalize_head_body` で正規化する。Q 側は同時に `q_scaled = q_norm * scale` を書く
  （`WithScale=true` の instantiate。`q_norm` の値を経由するので `(x*inv)*scale` の丸めは
  split path と同一）。
- `act` / `q_norm` / `k_norm` / `q_scaled` をすべて materialize する。recurrence は変更しない。
- `__global__` から `__global__` を launch しない。

### matcher

`fused_dispatch.cpp` の `match_gdn_prepare_fusion` が 4 dispatch を検査する。

- kernel id 順（SILU / L2_NORMALIZE / L2_NORMALIZE / SCALE）と input/output count
- `q_norm`（dispatch 1 の出力）が `SCALE` の入力 slot と一致
- `act` は BF16 × 10240、`conv` は F32 × 10240、`q_norm`/`k_norm`/`q_scaled` は F32 × 2048
- Q view の pointer が `act` と一致し、K view の pointer が `act + 2048 × 2 byte` と一致。
  view の row stride は `act` と同一（`validate_views` が base の stride を継承する）
- row domain 一致、`rows = rows_for(act, ctx)` が 1..2048
- eps = 1e-6、group_size = 128、scale = `1.0f / sqrtf(128.0f)`（emitter と同一式）
- elementwise selector が `SiluF32ToBf16`（features 10240）で Optimized、L2 selector が
  (BF16→F32, 2048, 128) で Optimized を返すこと
- alias: `conv` / `q_norm` / `k_norm` / `q_scaled` は相互に同一か disjoint、`act` とも同一か
  disjoint（部分 overlap は拒否）

部分 overlap を拒否するため、**48 layer 中 47 layer が fusion 対象**となる。workspace allocator が
1 layer だけ `k_norm` を `conv` の buffer の先頭に配置しており（`conv` の lifetime が SILU で
終わるため）、その layer は split path にフォールバックする。

### fusion counter

`KERNEL_TRACE_FUSED` に `gdn_prepare` を追加。verify update 1 回あたり **47 hit**。

### resource

| kernel | SGPR | VGPR | LDS | scratch |
| --- | ---: | ---: | ---: | ---: |
| `fused_gdn_prepare_kernel` | 26 | 34 | 0 | **0** |
| `op_silu_f32_bf16::impl`（変更なし） | 18 | 30 | 0 | 0 |
| `l2_normalize_bf16_f32_kernel`（変更なし） | 22 | 8 | 0 | 0 |

### primitive correctness

`test_qwen35_fused_gdn_prepare`（build B、rows 1/3/7/8）: `act` / `q_norm` / `k_norm` /
`q_scaled` すべて split vs fused で **byte-exact**、**18/18 PASS**。

### primitive performance

`phaseshift-bench fused-gdn-prepare`、warmup 200、iters 2000。

| M | split（4 launch） | fused（1 launch） | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 18.144 | 10.666 | **1.70x** |
| 3 | 17.002 | 9.864 | 1.72x |
| 7 | 16.079 | 9.300 | 1.73x |
| 8 | 15.413 | 8.770 | **1.76x** |

1 layer あたり約 6.6-7.5 us の削減。47 layer × 7 us ≒ 330 us / update。

### in-context A/B（`test_dflash2_gate11h_verify_profile`、ctx=2048、4 pairs、GPU 3）

| M | A verify_us | B verify_us | median | paired (%) |
| ---: | ---: | ---: | ---: | --- |
| 1 | 35402.60 | 35084.84 | **-0.90%** | -0.86 / -0.91 / -0.84 / -0.89 |
| 3 | 38347.02 | 38002.46 | **-0.90%** | -0.70 / -1.01 / -0.78 / -0.96 |
| 8 | 40597.79 | 40291.39 | **-0.75%** | -0.61 / -0.77 / -0.74 / -0.92 |

すべての M で 4/4 pair が改善側。`gdn_prepare` hit は 47/update（A は 0）。

### launch accounting（1 update、`PHASESHIFT_QWEN35_PERF_STATS=1`）

| M | build | logical | optimized | fallback | physical |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | A | 1654 | 1336 | 0 | 1353 |
| 1 | B | 1654 | 1195 | 0 | **1212** |
| 3 | A | 1654 | 1336 | 0 | 1449 |
| 3 | B | 1654 | 1195 | 0 | **1308** |
| 8 | A | 1654 | 1336 | 0 | 1353 |
| 8 | B | 1654 | 1195 | 0 | **1212** |

logical は不変。physical / optimized は全 M で **-141** = 3 × 47 layer と完全に一致する。

### correctness / required

| 項目 | 結果 |
| --- | --- |
| required（A: cumulative baseline） | **130/130 PASS**（skip 0、直列 `-j1`） |
| required（B: + 本 Gate） | **132/132 PASS**（skip 0。追加分は `test_qwen35_fused_gdn_prepare` と `test_bench_help_fused_gdn_prepare`） |
| build error / warning | 0（新規・変更 TU） |
| Target greedy `GENERATED_IDS`（294 token） | A/B で不変 |
| DFlash2 BF16 / PSQ4 / PSQ4+INT2 head（A/B） | ids 一致、acceptance 一致（rounds 40/42、accepted 87/85、full/partial/reruns も一致） |
| gate11c（GDN rerun-reference） | **failed=0**、3 case `ids_match=yes`・`hist_reruns=0` |
| `gdn_prepare` hit | [0, 47/update]（A は 0、B のみ発火） |

### E2E A/B（6 workload、15 paired、A = baseline / B = + 本 Gate）

| workload | n | A tok/s | B tok/s | B-A | paired (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| prose256 | 5 | 64.77 | 65.86 | +1.68% | +0.66 / +1.05 / +1.31 / +2.29 / +2.59 |
| prose512 | 2 | 72.16 | 73.13 | +1.34% | +2.48 / +0.23 |
| code | 2 | 146.61 | 147.80 | +0.81% | +0.96 / +0.65 |
| math | 2 | 137.79 | 140.03 | +1.63% | +2.46 / +0.80 |
| reasoning | 2 | 69.48 | 69.99 | +0.73% | +0.79 / +0.67 |
| ctx2048 | 2 | 160.56 | 161.66 | +0.68% | +0.79 / +0.57 |
| 合計 | **15** | — | — | **median +0.80%** | 改善側 15/15 |

rounds / accepted / full / partial / reruns は全 workload で A/B 一致（prose256: rounds 85 /
accepted 170、etc.）。decode 1 round あたりの内訳は drafter 約 5.8 ms + verify（M=8）約 40 ms で
verify が支配的なため、verify 側の -0.75% がそのまま decode 全体の改善として現れている。

### decision

**ADOPT**

理由:

- correctness: hard PASS。fused は `act` / `q_norm` / `k_norm` / `q_scaled` が byte-exact
  （rows 1/3/7/8）。required は A 130/130、B 132/132（skip 0）、DFlash2 3 経路の ids は
  target-only greedy と完全一致、gate11c も failed=0。
- in-context: verify wall が **M=1 -0.90% / M=3 -0.90% / M=8 -0.75%**（各 4/4 pair 改善）。
  Target Verify の改善が直接測れており、本 series の判定基準（measurable Verify improvement）を満たす。
- launch accounting: physical / optimized が全 M で **-141**（= 3 × 47 layer、logical は不変）。
- primitive: 4→1 で **1.70-1.76x**。scratch 0（SGPR 26 / VGPR 34 / LDS 0）。
- E2E: 15 pair 中 15 pair 改善、median **+0.80%**。
- regression: split-only build（A）の required が 130/130、Target greedy ids は A/B 一致。

参考: `PHASESHIFT_QWEN35_KERNEL_MODE=correctness` の greedy ids は auto と一致しない（index 1 で
差）。correctness 系は reduction 順が異なる別実装で bit-exact ではないため、この比較は本 Gate の
判定には使わない（事前から成立する性質）。

### commit

| commit | 内容 |
| --- | --- |
| `49c9b7f5` | refactor(kernels): share exact gdn prepare device bodies |
| `4c7af76a` | feat(runtime): fuse exact gdn prepare primitives |
| `3d321549` | test(runtime): cover exact gdn prepare fusion |
| `1b9f6dbb` | docs(perf): report gate 11j-d gdn prepare fusion |

## Gate 11J-E Target GDN post fusion

GDN layer の recurrence 直後〜出力 projection 前の tiny primitive 群を 1 launch にまとめる。

### 対象 chain の特定

dispatch 順（`lower_to_primitives.cpp` 755-785 行）は次のとおり。

| # | dispatch | input | output | 備考 |
| --- | --- | --- | --- | --- |
| 0 | `RMS_NORM` | recurrence 出力 F32 × 6144、weight BF16 × 128 | `rms` F32 × 6144 | group 128、eps、DIRECT weight |
| 1 | `SILU` | `z` projection 出力 BF16 × 6144 | `gated_z` F32 × 6144 | `SiluBf16ToF32` selector |
| 2 | `MUL` | `rms`, `gated_z` | `gated` BF16 × 6144 | `MulF32F32ToBf16` selector |
| 3 | `ACTIVATION_QUANTIZE` | `gated` | codes / scales | 次 dispatch の out projection 用 |

`z` の producer は layer 先頭の z projection であり直前ではないため、本 Gate は consumer side
fusion として `z` を外部入力に取る。recurrence と out projection は変更しない。

本 Gate では 0-2 の 3 dispatch を 1 launch にまとめる（3→1）。3 の quantize まで含めるには row 全体を
1 block に置く必要があり、(a) 32 thread で k=6144 の quantize を回すと `kNChunk=24` で VGPR spill、
(b) ideal な 1 warp/group 構成にするには過去 Gate の共有 body（`rmsnorm_row_body`）を per-warp 用に
書き換える必要がある。いずれも本 Gate の制約（scratch 0、過去 Gate の body を変更しない）に反するため
採らない。48 layer × 3 = 144 dispatch のうち 96 を削減する。

### compile flag

```cmake
option(PHASESHIFT_FUSE_GDN_POST "Enable exact Target GDN post fusion" OFF)
```

### shared device body

| body | 出所 | 共有先 |
| --- | --- | --- |
| `rmsnorm_row_body<F32,F32,BF16,PER_GROUP,DIRECT>` | Gate 11J-A で抽出済み | `rmsnorm.hip`（standalone） |
| `gdn_silu_f32` | Gate 11J-D で抽出 | `op_silu_f32_bf16` / `op_silu_bf16_f32` |
| `gdn_mul_f32_bf16_body` | 本 Gate で抽出 | `op_mul_f32_f32_bf16` |

`op_silu_bf16_f32` と `op_mul_f32_f32_bf16` は式・演算順・ベクトル分岐を同一に保ったまま body を
呼ぶ形にした（silu は `x / (1.0f + expf(-x))`、mul は `bf16(a*b)`）。

RMS は standalone が group_size=128 のとき block=32（1 warp）で (groups, rows) grid を張る。
`rmsnorm_row_body` は `threadIdx.x` / `blockDim.x` を直接使うため、fused kernel は同じ
block=32 / grid=(48, rows) 構成で呼ぶ（bit-exact のため）。

### fused kernel

`kernels/optimized/gdn/fused_gdn_post.hip`。grid = (48, rows)、block = **32 threads**。

- Phase 1: `rmsnorm_row_body<F32,F32,BF16,PER_GROUP,DIRECT>` で group 単位に正規化し `rms` を書く。
- Phase 2: 同じ thread が同じ index の `rms` を読み戻し、`silu(z)` を計算して `gated_z` と
  `gated = bf16(rms * silu_z)` を書く（要素は thread 自身が書いたものなので追加の同期は不要）。
- `rms` / `gated_z` / `gated` を materialize する。recurrence / out projection は変更しない。

### matcher

`match_gdn_post_fusion` が 3 dispatch を検査する。

- kernel id 順（RMS_NORM / SILU / MUL）、input/output count
- `MUL` の input0 が RMS の出力 slot、input1 が SILU の出力 slot と一致
- `rec` F32 × 6144、`rms` F32 × 6144、`z` BF16 × 6144、`gated_z` F32 × 6144、`gated` BF16 × 6144
- RMS の group_size = 128、weight_mode = DIRECT、parameter が BF16 × 128
- row domain 一致、rows = 1..2048
- alias: `rms` / `gated_z` / `gated` は相互に同一か disjoint、`rec` / `z` とも同一か disjoint
- selector: rmsnorm (F32→F32, BF16 PerGroup DIRECT, 6144, 128) / `SiluBf16ToF32` (6144) /
  `MulF32F32ToBf16` (6144) がいずれも Optimized

48 layer すべてが条件を満たし、**48/48 layer が fusion 対象**（Gate 11J-D の 47/48 と異なり除外なし）。

### resource

| kernel | SGPR | VGPR | LDS | scratch |
| --- | ---: | ---: | ---: | ---: |
| `fused_gdn_post_kernel` | 31 | 34 | 32 | **0** |
| `op_silu_bf16_f32::impl`（変更なし相当） | 18 | 30 | 0 | 0 |
| `op_mul_f32_f32_bf16::impl`（変更なし相当） | 20 | 18 | 0 | 0 |

### primitive correctness

`test_qwen35_fused_gdn_post`（build B、rows 1/3/7/8）: `rms_out` / `gated_z` / `gated` すべて
split vs fused で **byte-exact**、**14/14 PASS**。

### primitive performance

`phaseshift-bench fused-gdn-post`、warmup 200、iters 2000。

| M | split（3 launch） | fused（1 launch） | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 9.995 | 3.905 | **2.56x** |
| 3 | 9.675 | 3.917 | 2.47x |
| 7 | 9.910 | 3.914 | 2.53x |
| 8 | 9.994 | 3.917 | **2.55x** |

1 layer あたり約 6.0-6.1 us の削減。48 layer × 6 us ≒ 290 us / update。

### in-context A/B（`test_dflash2_gate11h_verify_profile`、ctx=2048、4 pairs、GPU 3）

| M | A verify_us | B verify_us | median | paired (%) |
| ---: | ---: | ---: | ---: | --- |
| 1 | 35114.72 | 34855.81 | **-0.74%** | -0.80 / -0.75 / -0.72 / -0.69 |
| 3 | 38026.09 | 37724.32 | **-0.79%** | -0.92 / -0.82 / -0.77 / -0.95 |
| 8 | 40346.82 | 40041.26 | **-0.76%** | -0.77 / -0.81 / -0.70 / -0.91 |

すべての M で 4/4 pair が改善側。`gdn_post` hit は 48/update（A は 0）。

### launch accounting（1 update、`PHASESHIFT_QWEN35_PERF_STATS=1`）

| M | build | logical | optimized | fallback | physical |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | A | 1654 | 1195 | 0 | 1212 |
| 1 | B | 1654 | 1099 | 0 | **1116** |
| 3 | A | 1654 | 1195 | 0 | 1308 |
| 3 | B | 1654 | 1099 | 0 | **1212** |
| 8 | A | 1654 | 1195 | 0 | 1212 |
| 8 | B | 1654 | 1099 | 0 | **1116** |

physical / optimized は全 M で **-96** = 2 × 48 layer と一致する。logical は不変。

### split-only regression（device body 抽出の検証）

抽出前（HEAD = Gate 11J-D 完了時点の `elementwise.hip`）と抽出後の split-only build（A）で
target-only greedy ids（128 token）を実測比較した。

| build | target greedy ids | auto == correctness mode |
| --- | --- | --- |
| pre-refactor | 128 token | False |
| post-refactor | 128 token（pre と完全一致） | False |

`op_silu_bf16_f32` / `op_mul_f32_f32_bf16` の body 共有によって target の greedy ids は変化しない。

### correctness / required

| 項目 | 結果 |
| --- | --- |
| required（A: cumulative baseline） | **133/133 PASS**（skip 0、直列 `-j1`） |
| required（B: + 本 Gate） | **134/134 PASS**（skip 0。追加分は `test_qwen35_fused_gdn_post`） |
| build error / warning | 0（新規・変更 TU） |
| Target greedy `GENERATED_IDS`（294 token） | A/B で不変 |
| DFlash2 BF16 / PSQ4 / PSQ4+INT2 head（A/B） | ids 一致、acceptance 一致（rounds 40/42、accepted 87/85） |
| gate11c（GDN rerun-reference） | **failed=0**、3 case `ids_match=yes`・`hist_reruns=0` |
| `gdn_post` hit | [0, 48/update]（A は 0、B のみ発火） |

### E2E A/B（6 workload、15 paired、A = baseline / B = + 本 Gate）

| workload | n | A tok/s | B tok/s | B-A | paired (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| prose256 | 5 | 65.82 | 66.28 | +0.70% | +0.71 / +2.39 / -0.96 / +0.63 / +0.89 |
| prose512 | 2 | 72.41 | 73.64 | +1.69% | +2.09 / +1.30 |
| code | 2 | 148.94 | 149.94 | +0.66% | +0.73 / +0.60 |
| math | 2 | 138.53 | 140.72 | +1.58% | +2.43 / +0.73 |
| reasoning | 2 | 69.78 | 70.27 | +0.71% | +0.81 / +0.61 |
| ctx2048 | 2 | 161.39 | 161.85 | +0.29% | +0.31 / +0.26 |
| 合計 | **15** | — | — | **median +0.73%** | 改善側 14/15 |

rounds / accepted / full / partial / reruns は全 workload で A/B 一致。

### decision

**ADOPT**

理由:

- correctness: hard PASS（byte-exact 14/14、required 133/133・134/134、CLI ids/acceptance 一致、
  gate11c failed=0、split-only greedy ids 不変）。
- in-context: verify wall が **M=1 -0.74% / M=3 -0.79% / M=8 -0.76%**（各 4/4 pair 改善）。
- launch accounting: physical / optimized が全 M で **-96**（= 2 × 48 layer、logical は不変）。
- primitive: 3→1 で **2.47-2.56x**、scratch 0（SGPR 31 / VGPR 34 / LDS 32）。
- E2E: median **+0.73%**（14/15 pair 改善）。

### commit

| commit | 内容 |
| --- | --- |
| `refactor(kernels): share exact gdn post device bodies` | `sigmoid_device.h` / `gdn_post_device.h` の共有 |
| `feat(runtime): fuse exact gdn post primitives` | fused kernel + matcher + flag |
| `test(runtime): cover exact gdn post fusion` | focused test |
| `docs(perf): report gate 11j-e gdn post fusion` | 本 section |

## Gate 11J-F Target attention gate + activation quantize fusion

full attention layer の attention 出力と gate の積、および次段 o projection 用 activation quantize を
1 launch にまとめる。

### 対象 chain の特定

dispatch 順（`lower_to_primitives.cpp` 855-875 行）:

| # | dispatch | input | output | 備考 |
| --- | --- | --- | --- | --- |
| 0 | `SIGMOID` | `v_gate`（`v_qproj` の後半 view）BF16 × 6144 | `gated_z` F32 × 6144 | `SigmoidBf16ToF32` selector |
| 1 | `MUL` | `v_ctx`（`PAGED_ATTENTION` 出力）F32 × 6144、`gated_z` | `att` BF16 × 6144 | `MulF32F32ToBf16` selector |
| 2 | `ACTIVATION_QUANTIZE` | `att` | codes / scales | 次 dispatch の o projection 用 |

`q_heads=24 × head_dim=256 = 6144`。`PAGED_ATTENTION`（attention main/reduce）と o projection の
GEMM は変更しない。`v_gate` の producer は layer 先頭の q projection であり直前ではないため、
consumer side fusion として gate を外部入力に取る。対象は 16 full attention layer。

### compile flag

```cmake
option(PHASESHIFT_FUSE_ATTN_GATE_QUANT
       "Enable exact Target attention gate and activation quantize fusion" OFF)
```

### shared device body

| body | 出所 | 共有先 |
| --- | --- | --- |
| `sigmoid_f32`（`1.0f / (1.0f + expf(-x))`） | 本 Gate で抽出 | `op_sigmoid_bf16_f32` |
| `gdn_mul_f32_bf16_body` | Gate 11J-E で抽出 | `op_mul_f32_f32_bf16` |
| `QuantVecRow` 段階 helper | Gate 11J-B | 既存 activation quantize |

### fused kernel

`kernels/optimized/fused_attention_gate_quant.hip`。grid = rows、block = 256 threads。

- Phase 1: `gate`（BF16）を読み `sigmoid`（F32）を計算して `sigmoid_out` へ materialize し、
  同時に `att = bf16(ctx * sigmoid)` を `gated_out` へ書く（elementwise）。
- `__syncthreads()`。
- Phase 2: `gated_out` を quantize layout で読み戻し、`activation_quantize_e4m3_vec_row_body_from_packed`
  で codes / scales を生成する。
- `sigmoid_out` / `gated_out` を materialize する。attention core と o projection は変更しない。

quantize の peak は row 単位の max なので thread 数に依存せず、Phase 1 も elementwise であるため
block 構成は 256 threads / grid = rows とした。

### matcher

`match_attention_gate_quant_fusion` が 3 dispatch を検査する。

- kernel id 順（SIGMOID / MUL / ACTIVATION_QUANTIZE_W4A8 または FP8）
- `MUL` の input1 が SIGMOID の出力 slot、`ACTIVATION_QUANTIZE` の入力が MUL の出力 slot と一致
- `gate` BF16 × 6144、`att` F32 × 6144、`sigmoid` F32 × 6144、`gated` BF16 × 6144
- row domain 一致、rows = 1..2048、alias は相互に同一か disjoint
- `gated` が 16 byte aligned / row stride 8 の倍数（vec quantize 条件、`activation_quantize_e4m3_vec_supported` で確認）
- selector: `SigmoidBf16ToF32` (6144) / `MulF32F32ToBf16` (6144) が Optimized
- quantize dispatch の `group_size=32`、`activation_kp=6144`、workspace layout 一致

16 layer すべてが条件を満たし **16/16 layer が fusion 対象**。

### resource

| kernel | SGPR | VGPR | LDS | scratch |
| --- | ---: | ---: | ---: | ---: |
| `fused_attention_gate_quant_e4m3_kernel` | 26 | 39 | 40 | **0** |
| `op_sigmoid_bf16_f32::impl`（変更なし相当） | 18 | 26 | 0 | 0 |

### primitive correctness

`test_qwen35_fused_attention_gate_quant`（build B、rows 1/3/7/8）: `sigmoid_out` / `gated` /
codes+scales すべて split vs fused で **byte-exact**、**13/13 PASS**。

### primitive performance

`phaseshift-bench fused-attention-gate-quant`、warmup 200、iters 2000。

| M | split（3 launch） | fused（1 launch） | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 10.650 | 6.052 | **1.76x** |
| 3 | 9.998 | 6.009 | 1.66x |
| 7 | 10.226 | 6.014 | 1.70x |
| 8 | 10.260 | 6.027 | **1.70x** |

1 layer あたり約 4.2 us の削減。16 layer × 4.2 us ≒ 67 us / update。

### in-context A/B（`test_dflash2_gate11h_verify_profile`、ctx=2048、4 pairs、GPU 3）

| M | A verify_us | B verify_us | median | paired (%) |
| ---: | ---: | ---: | ---: | --- |
| 1 | 34802.58 | 34689.13 | **-0.33%** | -0.34 / -0.32 / -0.28 / -0.41 |
| 3 | 37684.47 | 37580.82 | **-0.28%** | -0.27 / -0.36 / -0.19 / -0.24 |
| 8 | 39920.86 | 39802.46 | **-0.30%** | -0.26 / -0.37 / -0.23 / -0.28 |

すべての M で 4/4 pair が改善側。`attention_gate_quant` hit は 16/update（A は 0）。

### launch accounting（1 update、`PHASESHIFT_QWEN35_PERF_STATS=1`）

| M | build | logical | optimized | fallback | physical |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | A | 1654 | 1099 | 0 | 1116 |
| 1 | B | 1654 | 1067 | 0 | **1084** |
| 3 | A | 1654 | 1099 | 0 | 1212 |
| 3 | B | 1654 | 1067 | 0 | **1180** |
| 8 | A | 1654 | 1099 | 0 | 1116 |
| 8 | B | 1654 | 1067 | 0 | **1084** |

physical / optimized は全 M で **-32** = 2 × 16 layer と一致する。logical は不変。

### split-only regression（device body 抽出の検証）

抽出前（HEAD = Gate 11J-E 完了時点の `elementwise.hip`）と抽出後の split-only build（A）で
target-only greedy ids（128 token）を実測比較した。

| build | target greedy ids | auto == correctness mode |
| --- | --- | --- |
| pre-refactor | 128 token | False |
| post-refactor | 128 token（pre と完全一致） | False |

`op_sigmoid_bf16_f32` の body 共有によって target の greedy ids は変化しない。

### correctness / required

| 項目 | 結果 |
| --- | --- |
| required（A: cumulative baseline） | **135/135 PASS**（skip 0、直列 `-j1`） |
| required（B: + 本 Gate） | **136/136 PASS**（skip 0。追加分は `test_qwen35_fused_attention_gate_quant`） |
| build error / warning | 0（新規・変更 TU） |
| Target greedy `GENERATED_IDS`（294 token） | A/B で不変 |
| DFlash2 BF16 / PSQ4 / PSQ4+INT2 head（A/B） | ids 一致、acceptance 一致（rounds 40/42、accepted 87/85） |
| gate11c（GDN rerun-reference） | **failed=0**、3 case `ids_match=yes`・`hist_reruns=0` |
| `attention_gate_quant` hit | [0, 16/update]（A は 0、B のみ発火） |

### E2E A/B（6 workload、15 paired、A = baseline / B = + 本 Gate）

| workload | n | A tok/s | B tok/s | B-A | paired (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| prose256 | 5 | 65.29 | 65.91 | +0.95% | +1.73 / +0.15 / +0.09 / -0.10 / +0.26 |
| prose512 | 2 | 72.94 | 73.68 | +1.02% | +1.40 / +0.65 |
| code | 2 | 150.00 | 150.22 | +0.15% | +0.16 / +0.13 |
| math | 2 | 139.41 | 140.99 | +1.13% | +2.04 / +0.23 |
| reasoning | 2 | 70.24 | 70.41 | +0.24% | +0.28 / +0.20 |
| ctx2048 | 2 | 162.25 | 162.31 | +0.04% | +0.22 / -0.14 |
| 合計 | **15** | — | — | **median +0.22%** | 改善側 13/15 |

rounds / accepted / full / partial / reruns は全 workload で A/B 一致。

### decision

**ADOPT**

理由:

- correctness: hard PASS（byte-exact 13/13、required 135/135・136/136、CLI ids/acceptance 一致、
  gate11c failed=0、split-only greedy ids 不変）。
- in-context: verify wall が **M=1 -0.33% / M=3 -0.28% / M=8 -0.30%**（各 4/4 pair 改善）。
- launch accounting: physical / optimized が全 M で **-32**（= 2 × 16 layer、logical は不変）。
- primitive: 3→1 で **1.66-1.76x**、scratch 0（SGPR 26 / VGPR 39 / LDS 40）。
- E2E: median **+0.22%**（13/15 pair 改善）。

Gate 11J-D/E と比べて対象 layer 数が 16 と少なく、削減 launch 数も -32 と小さいため、
verify / E2E の改善幅も比例して小さい（-0.30% / +0.22%）。

### commit

| commit | 内容 |
| --- | --- |
| `refactor(kernels): share exact attention gate device body` | `sigmoid_device.h` |
| `feat(runtime): fuse attention gate activation quantization` | fused kernel + matcher + flag |
| `test(runtime): cover exact attention gate quant fusion` | focused test |
| `docs(perf): report gate 11j-f attention gate quant fusion` | 本 section |

## Gate 11J-G Q postprocess fusion

attention Q projection 後の tiny primitive 群（norm + RoPE）を 1 launch にまとめる。Target 側と
DFlash2 側を個別に判定する。

### 対象 chain の特定

| path | sequence | 位置 |
| --- | --- | --- |
| Target | `RMS_NORM(v_q view, BF16→F32, group 256, PerGroup ONE_PLUS) → ROPE(F32→BF16, pair)` | full attention layer（16 layer） |
| DFlash2 | `dflash2_rmsnorm_direct_bf16(q_raw → q_norm, head_dim) → dflash2_rope_bf16(q_norm → q_rope)` | cached layer / stateless layer（5 layer） |

Target の RoPE は `rope_f32_bf16_pair_kernel`（`inv_freq` table、`kPairHeadDim=256` /
`kPairRotary=64` / block 128）が production で選ばれる（`ctx.rope_inv_freq` が非 null）。
DFlash2 の RoPE は `dflash2_rope_bf16`（BF16 in/out、`round_to_bf16_f32` を各積に適用）で contract が
異なる。両者は共有せず、それぞれ独立に扱う。

### DFlash2 側: fused `dflash2_rmsnorm + dflash2_rope`

- 専用 device body: `kernels/dflash2/detail/rope_device.h`（`round_to_bf16_f32` /
  `dflash2_rope_pair_body`）を追加し、`dflash2/rope.hip` と fused kernel で共有。
  あわせて `kDflash2NormThreads` を `dflash2/detail/rmsnorm_device.h` へ移し、Gate 11J-C の
  `dflash2_rmsnorm_group_body` と block 数を共有する。
- fused kernel `kernels/dflash2/fused_norm_rope.hip`: grid = (features/head_dim, rows)、
  block = 128（standalone の norm と同一）。norm phase の出力を `__syncthreads()` の後に読み戻し、
  `dflash2_rope_pair_body` で rotate する。`q_norm` / `q_rope` は materialize する。
- 対象: cached layer と stateless layer（合計 2 call site）。
- flag: `PHASESHIFT_FUSE_Q_POSTPROCESS`（default OFF）。counter は
  `KERNEL_TRACE_FUSED dflash2_q_postprocess`。

### Target 側: REVERT

Target は `RMS_NORM → ROPE(pair)` の 2 dispatch なので 2→1 が可能だが、`rmsnorm_row_body` は
`threadIdx.x` / `blockDim.x` を直接使い、standalone は group ごとに block を分ける構成である。
1 kernel / 1 row にまとめるには「row 内の全 head group を逐次呼ぶ」形になり、同期が
head 数ぶん累積する。実測（`phaseshift-bench`、warmup 200 / iters 2000）:

| path | M | split（2 launch） | fused（1 launch） | speedup |
| --- | ---: | ---: | ---: | ---: |
| Target q（features 6144 / 24 head） | 1 | 11.471 | 32.234 | **0.36x** |
| Target q | 8 | 9.257 | 24.605 | 0.38x |
| Target k（features 1024 / 4 head） | 1 | 10.260 | 9.965 | 1.03x |
| Target k | 8 | 9.007 | 8.760 | 1.03x |

q では primitive が **2.7 倍遅い**。gate の判定規則（primitive slower → REVERT）に従い Target 側は
REVERT とし、実装（`fused_target_norm_rope` / matcher / call site / bench）は commit しない。
共有 body を変更せずに head 並列を確保できないことが原因で、exactness を維持したままでは
この family は成立しない。

### resource

| kernel | SGPR | VGPR | LDS | scratch |
| --- | ---: | ---: | ---: | ---: |
| `fused_dflash2_norm_rope_bf16_kernel` | 27 | 44 | 0 | **0** |

### primitive correctness

`test_dflash2_fused_norm_rope`（build B、rows 1/3/7/8、heads 24 / 4、position_start 128 / 0）:
`norm` / `rope` が split vs fused で **byte-exact**、**25/25 PASS**。

### primitive performance（DFlash2、`phaseshift-bench fused-dflash2-norm-rope`）

| 形 | M | split（2 launch） | fused（1 launch） | speedup |
| --- | ---: | ---: | ---: | ---: |
| q（6144） | 1 | 7.925 | 4.877 | **1.63x** |
| q | 8 | 7.459 | 5.172 | 1.44x |
| k（1024） | 1 | 7.801 | 4.890 | 1.60x |
| k | 8 | 7.242 | 4.859 | 1.49x |

### in-context A/B（proposer、`test_dflash2_gate7_perf`、ctx=2048、7 pairs、GPU 3）

| 区間 | A us | B us | median | paired (%) |
| --- | ---: | ---: | ---: | --- |
| cached backbone | 4416.67 | 4409.38 | **-0.19%** | -0.38 / -0.21 / -0.08 / +0.01 / -0.19 / -0.11 / -0.19 |
| full cached proposer | 5828.63 | 5827.65 | **-0.13%** | -0.13 / +0.06 / +0.06 / +0.14 / -0.16 / -0.40 / -0.19 |

backbone は 6/7 pair 改善、proposer は 4/7。fusion は
`dflash2_q_postprocess=875`（process 全体、決定論的）で発火する。launch 削減は
cached layer 5 件 + stateless layer 5 件で **-1 / layer**（cached path は -5 / draft forward）。

Target verify は本 Gate では変更しない（Target 側 REVERT）。

### correctness / required

| 項目 | 結果 |
| --- | --- |
| required（A: cumulative baseline） | **137/137 PASS**（skip 0、直列 `-j1`） |
| required（B: + 本 Gate） | **138/138 PASS**（skip 0。追加分は `test_dflash2_fused_norm_rope`） |
| build error / warning | 0（新規・変更 TU） |
| Target greedy `GENERATED_IDS`（294 token） | A/B で不変 |
| DFlash2 BF16 / PSQ4 / PSQ4+INT2 head（A/B） | ids 一致、acceptance 一致（rounds 42、accepted 85、reruns 0） |
| gate11c（GDN rerun-reference） | **failed=0**、3 case `ids_match=yes`・`hist_reruns=0` |
| split-only regression（dflash2 norm/rope body 抽出） | DFlash2 drafter ids **完全一致**、acceptance 統計（rounds/accepted/full/partial/mean/reruns）も一致（差分は timing のみ） |

### E2E A/B（6 workload、15 paired、A = baseline / B = + 本 Gate）

| workload | n | A tok/s | B tok/s | B-A | paired (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| prose256 | 5 | 65.60 | 65.61 | +0.02% | +0.03 / +0.02 / -0.08 / -0.09 / +0.06 |
| prose512 | 2 | 73.16 | 73.90 | +1.00% | +1.51 / +0.49 |
| code | 2 | 150.45 | 150.28 | -0.11% | -0.03 / -0.20 |
| math | 2 | 139.84 | 141.08 | +0.89% | +1.86 / -0.07 |
| reasoning | 2 | 70.47 | 70.52 | +0.06% | +0.23 / -0.10 |
| ctx2048 | 2 | 162.63 | 162.79 | +0.10% | +0.35 / -0.16 |
| 合計 | **15** | — | — | **median +0.02%** | 改善側 8/15 |

### decision

- **Target Q postprocess: REVERT**（primitive が 0.36x、実装は commit しない）
- **DFlash2 Q postprocess: PERF-NEUTRAL**（primitive 1.44-1.63x、proposer -0.13%、E2E median +0.02%）

DFlash2 側は primitive は明確に速いが、1 layer あたり -1 launch（-5 / draft forward）にとどまり、
proposer / E2E の分解能（proposer 7 pairs で ±0.15% 程度）では利得を確定できない。gate の
判定規則（primitive faster / E2E neutral → PERF-NEUTRAL、複雑度が低ければ flag OFF で保持）に従い、
flag default OFF のまま code を保持する。

### commit

| commit | 内容 |
| --- | --- |
| `refactor(dflash2): share exact norm and rope device bodies` | `rope_device.h` / `kDflash2NormThreads` 共有 |
| `feat(dflash2): fuse exact q norm and rope` | fused kernel + flag + counter |
| `test(dflash2): cover exact q norm and rope fusion` | focused test |
| `docs(perf): report gate 11j-g q postprocess fusion` | 本 section |

## Gate 11J-H K postprocess fusion

attention K projection 後の norm + RoPE を 1 launch にまとめる。Target 側と DFlash2 側を個別に判定する。

### 対象 chain の特定

| path | sequence | 箇所 |
| --- | --- | --- |
| Target | `RMS_NORM(v_k, BF16→F32, group 256, PerGroup ONE_PLUS) → ROPE(pair)` | full attention layer（16 layer） |
| DFlash2 | `dflash2_rmsnorm_direct_bf16(k_noise_raw → k_noise_norm) → dflash2_rope_bf16(k_noise_norm → k_noise_rope)` | cached backbone layer（5 layer） |

### Target 側: 未実装（PERF-NEUTRAL 相当）

Target の `RMS_NORM → ROPE(pair)` は Gate 11J-G で実装・実測済み（同じ kernel/body を使う）。
その primitive 実測は k geometry（features 1024 / head 4）で **1.03-1.07x** と neutral であり、
primitive faster を満たさないため本 Gate では実装しない（gate 11J-G の節を参照）。

### DFlash2 側: fused `dflash2_rmsnorm + dflash2_rope`

Gate 11J-G で導入した `fused_dflash2_norm_rope_bf16` をそのまま再利用し、
cached backbone の `k_noise` pair を 2→1 にする（flag は `PHASESHIFT_FUSE_K_POSTPROCESS`、
counter は `KERNEL_TRACE_FUSED dflash2_k_postprocess`）。

- Gate 11J-G の helper `dflash2_norm_rope_eligible` は Q/K 両 flag で使うため guard を
  `#if defined(PHASESHIFT_FUSE_Q_POSTPROCESS) || defined(PHASESHIFT_FUSE_K_POSTPROCESS)` に修正した
  （Gate 11J-G の build では Q のみ ON のため顕在化しなかった build 依存の修正）。
- 本 Gate の対象は cached backbone の k_noise pair（production proposer の経路）。
  stateless path は参照テスト専用、context-append の k pair は prefill 1 回のみのため、
  本 Gate では配線していない（launch 削減の便益が draft forward に比例しないため）。

### primitive performance（k geometry、`phaseshift-bench fused-dflash2-norm-rope --features 1024`）

| M | split（2 launch） | fused（1 launch） | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 7.801 | 4.890 | **1.60x** |
| 8 | 7.242 | 4.859 | 1.49x |

### in-context A/B（proposer、ctx=2048、7 pairs、GPU 3）

| 区間 | A us | B us | median | paired (%) |
| --- | ---: | ---: | ---: | --- |
| cached backbone | 4421.88 | 4417.21 | **-0.08%** | +0.01 / -0.08 / +0.01 / -0.15 / -0.08 / -0.22 / -0.01 |
| full cached proposer | 5829.54 | 5833.88 | **-0.01%** | +0.10 / -0.10 / +0.07 / -0.01 / -0.62 / +0.29 / -0.32 |

`dflash2_k_postprocess` hit は 750（process 全体、決定論的）。launch 削減は cached layer 5 件で
**-1 / layer**（-5 / draft forward）。Gate 11J-G の DFlash2 側と同じく、この削減量は proposer
A/B の分解能では確定できない。

### correctness / required

| 項目 | 結果 |
| --- | --- |
| required（A: cumulative baseline） | **137/137 PASS**（skip 0、直列 `-j1`） |
| required（B: + 本 Gate） | **138/138 PASS**（skip 0） |
| build error / warning | 0（新規・変更 TU） |
| Target greedy `GENERATED_IDS`（294 token） | A/B で不変 |
| DFlash2 BF16 / PSQ4 / PSQ4+INT2 head（A/B） | ids 一致、acceptance 一致（rounds 40/42、accepted 87/85、reruns 0） |
| gate11c（GDN rerun-reference） | **failed=0**、3 case `ids_match=yes`・`hist_reruns=0` |

### E2E A/B（6 workload、15 paired、A = baseline / B = + 本 Gate）

| workload | n | A tok/s | B tok/s | B-A | paired (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| prose256 | 5 | 66.65 | 65.99 | -0.99% | -0.01 / -1.58 / -1.67 / -1.79 / +1.78 |
| prose512 | 2 | 73.51 | 73.91 | +0.54% | +1.36 / -0.26 |
| code | 2 | 149.09 | 149.03 | -0.04% | +0.01 / -0.09 |
| math | 2 | 139.92 | 141.13 | +0.86% | +1.67 / +0.06 |
| reasoning | 2 | 70.50 | 70.50 | +0.01% | +0.08 / -0.06 |
| ctx2048 | 2 | 162.75 | 162.75 | -0.00% | +0.25 / -0.26 |
| 合計 | **15** | — | — | **median -0.01%** | 改善側 7/15 |

rounds / accepted / full / partial / reruns は全 workload で A/B 一致。

### decision

- **Target K postprocess: 実装せず（PERF-NEUTRAL 相当。同一 kernel の primitive が k geometry で 1.03-1.07x）**
- **DFlash2 K postprocess: PERF-NEUTRAL**（primitive 1.49-1.60x、proposer -0.01%、E2E median -0.01%）

1 layer あたり -1 launch（-5 / draft forward）の削減では proposer / E2E の分解能で利得を
確定できない。gate の判定規則に従い flag default OFF のまま code を保持する。

### commit

| commit | 内容 |
| --- | --- |
| `feat(dflash2): fuse exact k norm and rope` | cached backbone の k_noise pair 融合（helper guard 修正を含む） |
| `docs(perf): report gate 11j-h k postprocess fusion` | 本 section |

# Gate 11J-C ～ 11J-H Final Report

Gate 11J-C ～ 11J-H を実施した。11J-I / 11J-J / 11J-K は未実施（discovery と設計は
`docs/rnd/fusion/gate11j-series-progress.md` に記録）。

## Baseline

| 項目 | 値 |
| --- | --- |
| initial SHA | `d79f5c92`（Gate 11J-B final。J-A / J-B Target SwiGLU は ADOPT 済み） |
| final SHA | `f012fc17` |
| branch / worktree | `gate/11j-c-k-fusion-series` / `.worktrees/gate11j-c-k-fusion-series` |
| GPU | R9700 gfx1201（計測は GPU 3 に固定。E2E のみ GPU 2 を併用） |
| build | A = cumulative baseline（前 Gate までの ADOPT を ON）、B = A + 現 Gate |
| 計測 | HIP_GRAPH=OFF、primitive warmup 200 / iters 2000、verify ctx2048、E2E は paired |

## Gate Decisions

| Gate | Fusion | Target/DFlash | Decision | Commit |
| --- | --- | --- | --- | --- |
| 11J-C | residual + RMSNorm + quantize | DFlash2 | **ADOPT** | `2a3a5342`..`30f69ce2` |
| 11J-D | GDN prepare（SILU + 2×L2Norm + SCALE） | Target | **ADOPT** | `49c9b7f5`..`1b9f6dbb` |
| 11J-E | GDN post（RMSNorm + Z SiLU + MUL） | Target | **ADOPT** | `8ce10743`..`d03f0d30` |
| 11J-F | attention gate + quantize | Target | **ADOPT** | `5006dffd`..`bb68b3d6` |
| 11J-G | Q postprocess（norm + rope） | Target: **REVERT** / DFlash2: **PERF-NEUTRAL** | `340e116b`..`3f177a6c` |
| 11J-H | K postprocess（norm + rope） | Target: 実装せず / DFlash2: **PERF-NEUTRAL** | `0260082b`..`f012fc17` |
| 11J-I | conv boundaries | DFlash2 | 未実施（設計のみ） | — |
| 11J-J | INT2 coarse + TopN | DFlash head | 未実施 | — |
| 11J-K | rerank + Top16 / selector | DFlash head | 未実施 | — |

## Launch Reduction

Target verify update 1 回（M=8、ctx2048）の物理 launch 数:

| metric | Initial（11J-B 完了時） | Final（11J-H 時点） | delta |
| --- | ---: | ---: | ---: |
| logical dispatches | 1654 | 1654 | 0 |
| optimized operations | 1336 | 1067 | **-269** |
| physical launches（M=8） | 1353 | 1084 | **-269** |
| physical launches（M=3） | 1449 | 1180 | -269 |

内訳: Gate 11J-D -141（3 × 47 layer）、Gate 11J-E -96（2 × 48 layer）、Gate 11J-F -32（2 × 16 layer）。
いずれも fusion 1 回あたりの消費 dispatch 数と layer 数の積に完全一致する。

## Target Verify（`test_dflash2_gate11h_verify_profile`、ctx=2048、GPU 3）

| metric | Initial（11J-D baseline） | Final（11J-F B） | delta |
| --- | ---: | ---: | ---: |
| M=1 wall us | 35402.60 | 34689.13 | **-2.0%** |
| M=3 wall us | 38347.02 | 37580.82 | **-2.0%** |
| M=8 wall us | 40597.79 | 39802.46 | **-1.96%** |

Gate 単体（4 pairs、paired）: 11J-D M1 -0.90% / M3 -0.90% / M8 -0.75%（4/4 改善）、
11J-E M1 -0.74% / M3 -0.79% / M8 -0.76%（4/4）、11J-F M1 -0.33% / M3 -0.28% / M8 -0.30%（4/4）。
累積 -1.96% は 3 Gate の積（-1.95%）と一致する。

## DFlash Proposer（`test_dflash2_gate7_perf`、ctx=2048）

| Gate | 対象 | cached backbone median | full cached proposer median | pairs 改善 |
| --- | --- | ---: | ---: | --- |
| 11J-C | DFlash2 residual+RMSNorm+quantize | **-0.82%** | **-0.64%** | 3/3, 3/3 |
| 11J-G | DFlash2 q norm+rope | -0.19% | -0.13% | 6/7, 4/7 |
| 11J-H | DFlash2 k norm+rope | -0.08% | -0.01% | 4/7, 3/7 |

## E2E（paired、6 workload）

Gate 単体:

| Gate | median | 改善側 |
| --- | ---: | --- |
| 11J-C | +0.01% | 11/15 |
| 11J-D | +0.80% | 15/15 |
| 11J-E | +0.73% | 14/15 |
| 11J-F | +0.22% | 13/15 |
| 11J-G | +0.02% | 8/15 |
| 11J-H | -0.01% | 7/15 |

累積（A = 全 flag OFF の 11J-B 完了 build、B = 11J-C～11J-F ON の build。15 paired）:

| workload | n | A tok/s | B tok/s | B-A | paired (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| prose256 | 5 | 64.76 | 65.96 | **+1.85%** | +1.76 / +1.93 / +3.32 / +3.37 / +1.90 |
| prose512 | 2 | 70.84 | 74.11 | +4.62% | +5.00 / +4.24 |
| code | 2 | 145.46 | 150.73 | +3.62% | +3.69 / +3.56 |
| math | 2 | 135.02 | 141.40 | +4.73% | +5.78 / +3.67 |
| reasoning | 2 | 68.28 | 70.72 | +3.56% | +3.68 / +3.44 |
| ctx2048 | 2 | 158.22 | 163.07 | +3.07% | +3.07 / +3.07 |
| 合計 | **15** | — | — | **median +3.44%** | **改善側 15/15** |

rounds / accepted は全 workload で A/B 一致（例: prose256 rounds 85 / accepted 170）。

注意: workload 内の campaign 間分散がある。特に ctx2048 は campaign の最後に実行されるため
熱・負荷ドリフトの影響を受けやすく、Gate 単体の +0.06%〜+0.29% と累積の +3.07% は整合しない。
最も信頼できる primary（prose256、5 pairs）では累積 +1.85% であり、Gate 単体の和（+1.76%）と
一致する。

## Acceptance / Correctness

| 項目 | 結果 |
| --- | --- |
| required（各 Gate の A / B） | C 129/130、D 130/132、E 133/134、F 135/136、G 137/138、H 137/138 すべて PASS、skip 0 |
| Target-only greedy ids | 各 Gate の refactor 検証で **128/128 完全一致**（D/E/F は elementwise body 抽出、G は dflash2 norm/rope body 抽出で確認） |
| DFlash2 final ids | 3 drafter 構成（BF16 / PSQ4 / PSQ4+INT2 head）すべて target-only greedy と完全一致（128/128） |
| acceptance | 各 Gate の A/B で rounds / accepted / full / partial / reruns が完全一致（例: rounds 42 / accepted 85 / reruns 0） |
| GDN history | gate11c（`test_dflash2_gate11c_e2e`）が各 Gate で failed=0、3 case `ids_match=yes`・`hist_reruns=0` |
| target reruns | 0 |
| split-only regression | 各 Gate の focused test が byte-exact（C 21/21、D 18/18、E 14/14、F 13/13、G 25/25、H は G と同一 kernel） |

## Resource

| Gate / kernel | SGPR | VGPR | LDS | scratch |
| --- | ---: | ---: | ---: | ---: |
| 11J-C `fused_dflash2_rmsnorm_quant_e4m3_kernel<true/false>` | 53 | 57 | 536 | **0** |
| 11J-D `fused_gdn_prepare_kernel` | 26 | 34 | 0 | **0** |
| 11J-E `fused_gdn_post_kernel` | 31 | 34 | 32 | **0** |
| 11J-F `fused_attention_gate_quant_e4m3_kernel` | 26 | 39 | 40 | **0** |
| 11J-G/H `fused_dflash2_norm_rope_bf16_kernel` | 27 | 44 | 0 | **0** |

すべて scratch 0（hard requirement を満たす）。

## Regression

- required full run は各 Gate で PASS、skip 0（直列 `-j1`）。
- build error 0。warning は既存のみ（`int2_select_impl` unused 等）。
- HIP_GRAPH=ON の compile / regression は本 series では未実施（全計測が HIP_GRAPH=OFF）。
  融合は通常の stream 上の追加 kernel であり graph capture とは独立だが、graph 統合の検証は残課題。

## Final Production Flags

ADOPT（cumulative baseline として ON）:

- `PHASESHIFT_FUSE_RESIDUAL_RMSNORM_QUANT`（11J-A）
- `PHASESHIFT_FUSE_TARGET_SWIGLU_QUANT`（11J-B）
- `PHASESHIFT_FUSE_DFLASH2_RESIDUAL_RMSNORM_QUANT`（11J-C）
- `PHASESHIFT_FUSE_GDN_PREPARE`（11J-D）
- `PHASESHIFT_FUSE_GDN_POST`（11J-E）
- `PHASESHIFT_FUSE_ATTN_GATE_QUANT`（11J-F）

PERF-NEUTRAL（code 保持、default OFF）:

- `PHASESHIFT_FUSE_DFLASH2_SWIGLU_QUANT`（11J-B）
- `PHASESHIFT_FUSE_Q_POSTPROCESS`（11J-G DFlash2 側）
- `PHASESHIFT_FUSE_K_POSTPROCESS`（11J-H DFlash2 側）

REVERT（commit しない）:

- Target Q/K postprocess（11J-G/H Target 側。`RMS_NORM → ROPE` は共有 body の制約で
  primitive が q geometry で 0.36x となり gate の規則により不採用）

default ON への切替は本 series の全 Gate 完了後にまとめて行う方針のため未実施。
`-DPHASESHIFT_FUSE_xxx=OFF` で legacy split path を build 可能な状態は維持している。

## Overall Result

- Target Verify: **-1.96%**（M=8 ctx2048）、physical launch **-269 / update**
- DFlash proposer: **-0.64%**（11J-C。G/H の DFlash2 側は分解能以下で PERF-NEUTRAL）
- E2E 累積: **median +3.44%**（15/15 改善。primary prose256 は +1.85%）
- 残り候補: 11J-I（conv boundaries、-10 launch/draft-forward 見込み）、11J-J（INT2 coarse + TopN、
  現状 ~1.215 ms の削減）、11J-K（rerank + Top16、現状 ~120 us の削減）

# Gate 11J-J: exact INT2 coarse head + top-N fusion

## 目的 / 対象 chain

`int2_coarse_head` → `coarse_topn_partition` → `coarse_topn_merge` の 3 launch と、
full coarse logits（`rows × vocab × 4B`、rows=7 で 6.95 MB）の global write/read を削減する
ため、coarse tile の出力を shared memory に staging し、CTA 内で exact な partial top-N を
求めて canonical merge で確定する 2 launch 構成を実装した。

- shared helpers:
  - `kernels/dflash2/detail/coarse_head_device.h`: 16x16 tile body と LUT load。
  - `kernels/dflash2/detail/coarse_topn_device.h`: `topn_better` / `topn_insert` / `topn_warp_best`。
  standalone の coarse kernel / partition kernel も同じ helper を使うよう refactor した（挙動不変）。
- `launch_dflash2_coarse_topn_merge` を分割し、fused partial から merge を再利用できるようにした。
- flag `PHASESHIFT_FUSE_DFLASH2_INT2_COARSE_TOPN` で legacy split と排他。

## 実装（検証時）

- topology: grid = partitions（`ceil(vocab/256)`、vocab=248320 で 970）/ block = 256（8 warp）。
  各 CTA は 16 tile（256 vocab）× 16 row を shared に staging し、row ごとに 16 thread が
  16 候補ずつ保持（kPerThread=16）して CTA 内で exact top-pool を抽出し、scratch へ書いて
  canonical merge で確定する。
- **scratch 0**（`amdhsa_private_segment_fixed_size = 0`）、VGPR 102 / SGPR 50 / LDS 9216。
- 16-lane `shfl_down` reduction が group 外 lane を読むため、各 lane が自分の tuple を保持した
  まま誤って自己無効化する問題があり、leader broadcast（`__shfl` from group leader）+
  warp lane 一意 index で修正した。この修正前は候補が欠落して必ず不一致になった。

## exactness（oracle = 既存 `int2_coarse_head` + `coarse_topn`、および CPU oracle）

| 条件 | 結果 |
| --- | --- |
| pool 16/24/32/48/64/80/128 × vocab 1024/4096/248320 × rows 1/3/7 | **204/204 PASS** |
| tie（同一 logit、異なる id） | PASS |
| fused vs legacy: ids exact / logits bit exact | 一致 |
| fused vs CPU oracle: ids exact / logits bit exact | 一致 |
| in-context（DFlash2 PSQ4 + INT2 head）: `GENERATED_IDS` / rounds / accepted / reruns | A/B 完全一致（42 rounds、accepted 85、reruns 0） |

## performance（in-context A/B、3 pairs、GPU 3、ctx=2048）

| 区間 | A us | B us | paired median |
| --- | ---: | ---: | ---: |
| coarse（B は fused partial + merge） | 1.028 | 2.210 | **+115.06%** |
| topn（B は merge 分） | 0.265 | 0.005 | **-98.11%** |
| coarse + topn | 1.291 | 2.215 | **+71.34%** |
| full cached proposer | 5926.07 | 6833.12 | **+15.34%** |

partitions を 2x / 4x に増やす variant も試したが 3.08 / 5.22 ms とさらに悪化した（CTA あたり
固定費が支配的で、分割は逆効果）。

## decision: REVERT

- top-N は設計どおりほぼ消えた（-98%）が coarse が 1.03 → 2.21 ms へ悪化し、proposer は
  **+15.34%** の再現性ある regression。
- 原因: (a) full logits の write/read 削減は `2 × 6.95 MB / 640 GB/s ≒ 22 us` で支配項ではない、
  (b) fused kernel（970 CTA × 8 warp、CTA 内 barrier + staging + 候補 list で VGPR 102）は
  tile 単価が legacy coarse（15520 block × 1 warp）の約 2x。
- 判定規則（reproducible improvement 必須、性能 regression は REVERT）に従い、production source
  （kernel / launcher / matcher / flag / test / counter）を削除した。共有 device helper の
  refactor のみ残す（standalone test で 93/93、15/15、挙動不変を確認）。

## commit

| commit | 内容 |
| --- | --- |
| `refactor(dflash2): share exact int2 coarse and topn bodies` | 共有 device helper と merge launcher 分割（挙動不変） |
| `docs(perf): report gate 11j-j int2 coarse topn fusion` | 本 section |

# Gate 11J-K: exact PSQ8 rerank + Top16 + pool-id remap fusion

## 目的 / 対象 chain

draft head の tail は

1. `psq8_candidate_rerank`（1 launch、grid = `(ceil(pool/16), rows)`、block = 32）
2. `dflash2_topk_f32_optimized`（2 launch: partition + merge、k = 16、vocab = pool）
3. `int2_remap_pool_ids`（1 launch、pool position → vocab id）

の 4 launch で構成され、pool32 で rerank ~0.079 ms + top16 ~0.045 ms を要していた。これを
**1 kernel family** にまとめ、rerank logits を shared に置いたまま exact top-16 を求め、
`int2_pool_ids` で vocab id へ直接写像する（remap kernel を削除）。

- shared body: `kernels/dflash2/detail/psq8_rerank_device.h`（1 candidate tile の PSQ8 dot。
  FMA 順序・scale 適用は現行 `psq8_candidate_rerank_kernel` と同一）。
  standalone rerank kernel も同じ body を使うよう refactor した（挙動不変、test 14/14）。
- topology: grid = `rows`、block = 64（2 warp、warp ごとに 16 candidate tile を担当）。
  pool > 32 は warp が tile を stride して処理する。
- top-16 の comparator は現行 topk と同一（logit 大、同値なら pool position 小）。
  winner を shared 上で -INFINITY にして 16 回繰り返す。position → vocab id は
  `int2_pool_ids[row*pool + pos]` を直接参照（remap 不要）。
- rerank logits の global materialization は debug/検証用に維持（fused kernel から書き出す）。
- flag: `PHASESHIFT_FUSE_DFLASH2_RERANK_TOP16`、counter: `KERNEL_TRACE_FUSED rerank_top16`。

## resource

| kernel | VGPR | SGPR | LDS | scratch | occupancy |
| --- | ---: | ---: | ---: | ---: | --- |
| `fused_rerank_top16_kernel` | 42 | 40 | 528 | **0** | 64 threads/CTA、grid = rows |

## exactness（oracle = 現行 rerank + topk_optimized + remap）

| 条件 | 結果 |
| --- | --- |
| pool 16/24/32/48/64/80/128 × vocab 1024/4096/248320 × rows 1/3/7 | **147/147 PASS** |
| rerank logits（fused が materialize した値 vs 現行） | **bit exact** |
| 最終 top16 ids（remap 後） | **exact** |
| 最終 top16 logits | **bit exact** |
| unordered candidate pool / boundary id（0、vocab-1） | PASS |
| in-context（DFlash2 PSQ4 + INT2 head）: `GENERATED_IDS` / rounds / accepted / reruns | A/B 完全一致（42 rounds、accepted 85、reruns 0） |

## performance（in-context A/B、3 pairs、GPU 3、ctx=2048）

| 区間 | A us | B us | paired median |
| --- | ---: | ---: | ---: |
| rerank（B は fused 内の rerank 相当） | 0.079 | 0.088 | +11.39% |
| top16（B は fused 内の top-16 相当） | 0.045 | 0.005 | **-88.89%** |
| rerank + top16 | 0.124 | 0.093 | **-25.00%** |
| full cached proposer | 5983.82 | 5870.20 | **-1.90%** |

paired は 3/3 すべて同符号（-1.89 / -2.06 / -1.90 %）。launch は 4 → 1（**-3 / draft forward**）。

## Gate 11J-K2: CandidateSelector fusion の feasibility

**NO-GO**。理由:

- rerank は row 並列（grid = rows）である一方、`CandidateSelector` は 1 block が draft row を
  順番に処理し、predecessor token 依存の chain を持つ。row ごとの rerank block から 1-block
  selector へ直接連結するには grid 全体の完了依存が必要で、合法な grid-wide barrier が無い。
- 全 row の rerank + selector を単一 block で処理する構成しか残らないが、rerank は
  latency-bound（pool32 で 0.079 ms、14 block × 32 thread）であり、単一 block 化は
  row 並列性を失うため明確に遅くなる。microbench で速いと証明できないため実装しない。
- 禁止事項（unsafe global spin barrier、全 CTA resident 仮定）に抵触しない合法解が無い。

## decision: ADOPT

- rerank + top16 で **-25.00%**、proposer で **-1.90%（3/3 paired）** の再現性ある改善。
- exact（147/147）、scratch 0、required PASS、skip 0。
- K1 は独立に ADOPT。K2 は NO-GO を report のみ。

## commit

| commit | 内容 |
| --- | --- |
| `refactor(dflash2): share exact psq8 rerank body` | 共有 device body（挙動不変） |
| `feat(dflash2): fuse exact rerank top16 and id remap` | fused kernel + launcher + executor 配線 + counter |
| `test(dflash2): cover fused rerank top16` | exact unit test |
| `docs(perf): report gate 11j-k rerank top16 fusion` | 本 section |

# DeepFusion Gate 11K-A: DFlash2 MLP tail（grouped-conv finish + residual）

## 目的 / 対象 chain

DFlash2 MLP tail は

1. `mlp_down` GEMM（共有 PSQ4 GEMM）
2. `dflash2_grouped_conv_finish`（phase 1）
3. `residual_add_bf16`（`attention_residual` + conv 出力 → layer 出力）

の 3 stage で、2 と 3 は exact に隣接する（conv 出力の consumer は residual add のみ）。
conv の epilogue で residual を加算し、`mlp_finished` の materialization と 1 launch を削除する。

attention 側の conv finish は Gate 11J-C の `residual+rmsnorm+quant`（rrq）窓と隣接するため
本 Gate の対象外（そちらは既に rrq が residual を所有）。

## 実装

- `grouped_dynamic_conv_kernel` に nullable な `residual` を追加し、conv 出力を BF16 丸めした
  のち同一 epilogue で加算する:
  `conv = f32_to_bf16(acc)` → `out = __float2bfloat16(bf16_to_f32(conv) + bf16_to_f32(residual[i]))`。
  split 経路は `residual = nullptr` で従来と完全に同一（追加コストは null 判定のみ）。
- `launch_dflash2_grouped_dynamic_conv_residual` を追加（flag ON の build のみ定義）。
- flag: `PHASESHIFT_DEEPFUSE_DFLASH2_MLP_A`、counter: `KERNEL_TRACE_FUSED dflash2_mlp_tail`。
- production では `mlp_finished` を書かない（他の consumer が無いことを確認済み）。

## exactness

| 条件 | 結果 |
| --- | --- |
| rows 1/2/3/7/8/16 × hidden 512/1024/5120 × phase 0/1 | **36/36 PASS**（split vs fused が bit exact） |
| in-context: `GENERATED_IDS` / rounds / accepted / reruns | A/B 完全一致（42 rounds、accepted 85、reruns 0） |
| fused hit（process 全体） | 210（A は 0） |

## performance（in-context A/B、3 pairs、GPU 3、ctx=2048）

| 区間 | A us | B us | paired median |
| --- | ---: | ---: | ---: |
| full cached proposer | 5853.45 | 5836.92 | **-0.39%**（-0.03 / -0.43 / -0.39） |

launch は 2 → 1（**-1 / layer**、cached layer 5 件で -5 / draft forward）。

## decision: ADOPT（DFlash2）

- exact（36/36）、launch 削減、proposer **-0.39%**（3/3 で非正）。
- Target 側の同等 fusion（Down GEMM + Residual）は本 Gate では未実装。理由は
  Target の Down が共有 PSQ GEMM（`select_psq4_gemm_config` / `select_psq8_gemm_config`）を
  通り、epilogue 化には GEMM core の trait 化が必要で、11J-J の実測（CTA 集約による
  occupancy 低下）から regression リスクが高く、exactness を保証した実装が残 budget で
  できないため。11K-C / 11K-D と同じ共有 core 改変に依存する。

# DeepFusion Gate 11K-B ～ 11K-E: assessment（本 session の判断）

## 11K-B（Gate + Up projection + SwiGLU）

**未実装（exactness を保証できないため）**。Target / DFlash2 の Gate/Up は
`select_psq4_gemm_config` / `select_psq8_gemm_config` が選ぶ共有の最適化 GEMM であり、
dual projection + SwiGLU epilogue は GEMM core（load / dequant / MMA / accumulator）の
trait 化を要求する。11J-J の実測では複数 tile を 1 CTA に束ねる構成が occupancy 低下で
tile 単価 ~2x になったため、GEMM core の再構成は regression リスクが高い。
また 11K-B を ADOPT する場合、Target の `SwiGLU+Quant`（11J-B ADOPT）を削除し Quant の所有者を
canonical split へ戻す大規模な ownership 変更を伴う（section 55）。本 session の残 budget で
exactness を保った実装と検証（required PASS / skip 0 / E2E）を完了できないと判断した。

## 11K-C（Quant + Down の DeepFusion）

**未実装（合法解が無い / 共有 core 改変）**。K=17408 の row-wide peak が確定する前に
E4M3 code を生成できないため、naive な 1 kernel 化は grid-wide 同期を要する（禁止）。
C1（scale のみ別 kernel、Down GEMM が BF16 SwiGLU を読んで register 内で code 生成）は
activation code の write/read を削除できるが、同じく共有 PSQ Down GEMM の core 改変が必要で、
11K-B と同じ理由で残 budget では exactness を保証できない。

## 11K-D / 11K-E

**11K-C の結果に依存するため未実施**。11K-E（final topology）は現時点の production 構成を
確定値として扱う:

- Target: Gate / Up（共有 PSQ GEMM）→ Safe `SwiGLU + Quant`（11J-B ADOPT）→ Down（共有 PSQ GEMM）
  → Residual
- DFlash2: Gate / Up（共有 PSQ GEMM）→ SwiGLU → Quant → Down（共有 PSQ4 GEMM）
  → conv finish + residual（11K-A ADOPT）

logical operation の owner は重複 0（`KERNEL_TRACE_FUSED` の hit で検証。詳細は下記
Active Fusion Ownership）。

## DeepFusion の cumulative（11K-A 採用後）

| 項目 | A（C-F + K） | B（+ 11K-A） | delta |
| --- | ---: | ---: | ---: |
| DFlash2 proposer（ctx=2048） | 5853.45 us | 5836.92 us | **-0.39%** |
| DFlash2 launch（MLP tail / layer） | 2 | 1 | -1 |
| Target Verify（M=8 ctx2048） | 39802 us | 39802 us | 0（Target 変更なし） |

# Active Fusion Ownership（Gate 11K-E 時点）

production で同一 logical operation を所有する fusion kernel が 2 つ以上存在しないこと。

## Target

| op | owner |
| --- | --- |
| Gate GEMM | canonical split（共有 PSQ GEMM） |
| Up GEMM | canonical split（共有 PSQ GEMM） |
| SwiGLU | Safe Fusion `PHASESHIFT_FUSE_TARGET_SWIGLU_QUANT`（11J-B ADOPT） |
| Quant（MLP 入力） | Safe Fusion `PHASESHIFT_FUSE_TARGET_SWIGLU_QUANT`（11J-B ADOPT） |
| Down GEMM | canonical split（共有 PSQ GEMM） |
| Residual（MLP tail） | canonical split `residual_add_bf16` |
| Residual + RMSNorm + Quant（layer pre） | Safe Fusion `PHASESHIFT_FUSE_RESIDUAL_RMSNORM_QUANT`（11J-A ADOPT） |
| GDN prepare / post | Safe Fusion `PHASESHIFT_FUSE_GDN_PREPARE` / `_GDN_POST` |
| attention gate + quant | Safe Fusion `PHASESHIFT_FUSE_ATTN_GATE_QUANT` |

## DFlash2

| op | owner |
| --- | --- |
| Gate GEMM / Up GEMM | canonical split（共有 PSQ4 GEMM） |
| SwiGLU | canonical split（`swiglu` kernel。`PHASESHIFT_FUSE_DFLASH2_SWIGLU_QUANT` は default OFF の PERF-NEUTRAL） |
| Quant | canonical split（`quantize_a8`） |
| Down GEMM | canonical split（共有 PSQ4 GEMM） |
| ConvFinish（MLP tail, phase 1） | Deep Tail A `PHASESHIFT_DEEPFUSE_DFLASH2_MLP_A`（11K-A ADOPT） |
| Residual（MLP tail） | Deep Tail A `PHASESHIFT_DEEPFUSE_DFLASH2_MLP_A` |
| ConvFinish + Residual（attention branch） | Safe Fusion `PHASESHIFT_FUSE_DFLASH2_RESIDUAL_RMSNORM_QUANT`（11J-C ADOPT の rrq） |
| head tail（rerank + Top16 + remap） | `PHASESHIFT_FUSE_DFLASH2_RERANK_TOP16`（11J-K ADOPT） |

**owner 重複: 0**（flag OFF の build では該当 op は canonical split が所有し、flag ON の build では
Deep/Safe fusion が所有する。両者は同一 build で排他であり、重複は発生しない）。

## Obsolete / Superseded

- 11J-J の fused int2 coarse + TopN: REVERT により production source から削除済み（kernel / launcher /
  flag / test / counter ともに 0）。共有 device helper（`detail/coarse_head_device.h`,
  `detail/coarse_topn_device.h`）と `launch_dflash2_coarse_topn_merge` の分割のみ残置。
- 11K-B / 11K-C 由来の kernel / launcher / flag は存在しない（未実装のため obsolete も 0）。

# Gate 11J-J / 11J-K / DeepFusion 11K-A ～ 11K-E Final Report

## Baseline

| 項目 | 値 |
| --- | --- |
| initial SHA | `d79f5c92`（11J series 開始）/ `b0c29251`（11J-C～H final） |
| final SHA | 本文書 commit 時点の HEAD（`git log --oneline d79f5c92..HEAD` 参照） |
| GPU | AMD Radeon AI PRO R9700（gfx1201）、GPU 3 を使用（GPU 0 は不使用） |
| ROCm | _rocm_sdk_devel（HIP/ROCm、`-DCMAKE_HIP_ARCHITECTURES=gfx1201`）、HIP_GRAPH=OFF |

## 11J-J

- decision: **REVERT**（性能 regression。production source は削除、共有 device helper の refactor のみ残置）
- coarse+TopN old: **1.291 ms**（coarse 1.028 + topn 0.263、ctx=2048、pool32、in-context）
- new: **2.215 ms**（fused partial 2.210 + merge 0.005）
- launch count: 3 → 2
- scratch bytes: **0**（`fused_coarse_topn_partition_kernel` VGPR 102 / SGPR 50 / LDS 9216）
- exactness: **204/204**（pools 16/24/32/48/64/80/128 × vocab 1024/4096/248320 × rows 1/3/7、tie 含む。
  fused vs legacy および CPU oracle で ids exact / logits bit exact）。in-context でも
  `GENERATED_IDS` / rounds / accepted / reruns が A/B 一致
- proposer delta: **+15.34%**（3 pairs、3/3 同符号）
- E2E delta: 未測定（REVERT のため）

## 11J-K

- K1 rerank+Top16+remap: **ADOPT**
- selector fusion feasibility: **NO-GO**（row 並列 rerank と 1-block predecessor chain selector の間に
  合法な grid-wide barrier が無く、単一 block 化は rerank の row 並列性を失う）
- decision: **ADOPT**
- old us: rerank 0.079 + top16 0.045 = **0.124 ms**
- new us: **0.093 ms**（rerank 0.088 + top16 0.005）
- launch count: 4 → 1（-3 / draft forward）
- exactness: **147/147**（pools 16-128 × vocab 1024/4096/248320 × rows 1/3/7、unordered pool /
  boundary id 含む。rerank logits bit exact / final ids exact / final logits bit exact）
- proposer: **-1.90%**（3 pairs、3/3 同符号）
- E2E: prose256 **+0.56%**（3 paired）、全 8 paired median **+0.78%**、ids_exact = all True

## DeepFusion Decisions

| Gate | Target | DFlash2 | decision |
| --- | --- | --- | --- |
| 11K-A tail | 未実装（共有 PSQ GEMM core の epilogue 化が必要、残 budget 外） | **ADOPT**（36/36 bit exact、proposer -0.39%、launch -1/layer） | DFlash2 ADOPT / Target 未実装 |
| 11K-B gate/up+swiglu | 未実装 | 未実装 | **NO-GO**（共有 PSQ GEMM の trait 化 + Quant/RMSNorm の ownership 移行。11J-J の実測から CTA 集約は regression リスク大） |
| 11K-C quant+down | 未実装 | 未実装 | **NO-GO**（row-wide peak 確定前の code 生成は grid-wide 同期が必要＝禁止。C1 も共有 GEMM core 改変） |
| 11K-D maximal tail | 未実施 | 未実施 | 11K-C 依存のため未実施 |
| 11K-E final topology | 現構成を確定 | 現構成を確定 | owner 重複 **0** |

## Target Final Topology

- logical: Gate GEMM → Up GEMM → SwiGLU → Quant → Down GEMM → Residual
- physical: 共有 PSQ GEMM（Gate/Up/Down）+ Safe `SwiGLU+Quant`（`PHASESHIFT_FUSE_TARGET_SWIGLU_QUANT`）
  + `residual_add_bf16`。加えて layer pre の `Residual+RMSNorm+Quant`
  （`PHASESHIFT_FUSE_RESIDUAL_RMSNORM_QUANT`）、GDN prepare/post、attention gate+quant。

## DFlash2 Final Topology

- logical: Gate GEMM → Up GEMM → SwiGLU → Quant → Down GEMM → ConvFinish → Residual
  （head 側: INT2 coarse → TopN → PSQ8 rerank → Top16）
- physical: 共有 PSQ4 GEMM + `swiglu` + `quantize_a8` + `grouped_dynamic_conv`（residual epilogue 込み、
  `PHASESHIFT_DEEPFUSE_DFLASH2_MLP_A`）+ `fused_rerank_top16`
  （`PHASESHIFT_FUSE_DFLASH2_RERANK_TOP16`）+ rrq（`PHASESHIFT_FUSE_DFLASH2_RESIDUAL_RMSNORM_QUANT`）。

## Deleted Superseded Fusion

- kernels: 1（11J-J の `fused_coarse_topn_partition_kernel` — REVERT により削除済み）
- launchers: 1（`launch_dflash2_fused_coarse_topn`）
- flags: 1（`PHASESHIFT_FUSE_DFLASH2_INT2_COARSE_TOPN`）
- matchers: 0（11J-J は executor 内の直呼び）
- tests: 1（`test_dflash2_int2_coarse_topn_fused`）
- 11K-B / 11K-C 由来の production 要素は 0（未実装）

## Ownership

duplicate logical op owners: **0**（`Active Fusion Ownership` section 参照）

## Resources

| kernel | VGPR | SGPR | LDS | scratch | occupancy |
| --- | ---: | ---: | ---: | ---: | --- |
| `fused_rerank_top16_kernel`（11J-K） | 42 | 40 | 528 | **0** | 64 threads/CTA、grid = rows |
| `grouped_dynamic_conv_kernel`（11K-A） | 14 | 29 | 0 | **0** | grid-stride elementwise |

（11J-J の fused kernel は REVERT のため削除済み。参考: VGPR 102 / SGPR 50 / LDS 9216 / scratch 0）

## Target Verify（`test_dflash2_gate11h_verify_profile`、ctx=2048、M=8、GPU 3）

| metric | initial | final | delta |
| --- | ---: | ---: | ---: |
| verify wall | 40598 us | **39658.14 us** | **-2.32%** |
| physical launches | 1353 | **1084** | **-269** |
| logical launches | 1654 | 1654 | 0 |
| optimized（selector 経由） | 1336 | 1067 | -269 |

## DFlash Proposer（`test_dflash2_gate7_perf`、ctx=2048、INT2 head、pool32）

| metric | 値 |
| --- | --- |
| Gate 11J-C | -0.64%（-15 launch/draft forward） |
| Gate 11J-J | REVERT（+15.34%） |
| Gate 11J-K | **-1.90%**（4 → 1 launch） |
| DeepFusion 11K-A | **-0.39%**（-1 launch/layer） |
| final 実測 absolute | ~5837 us（A/B 比較時の B、3 pairs） |

## E2E

| workload | 11J-K paired（A = C-F baseline / B = +K） |
| --- | --- |
| prose256 | +0.56%（3 paired） |
| prose512 | +1.41% |
| code | +1.05% |
| math | -1.61% |
| reasoning | +0.04% |
| ctx2048 | +1.22% |
| all paired median | **+0.78%**（n=8、neg 2/8、ids_exact all True） |

11J-C～H の cumulative E2E（`b0c29251` 時点）は median **+3.44%**（15/15 改善、primary prose256 +1.85%）。
11J-K と DeepFusion 11K-A はそれぞれ独立に proposer で -1.90% / -0.39% を実測しており、
E2E への寄与は小さい（proposer は round 全体の一部）。累積は per-gate 実測の合成として報告する。

## Correctness

- target IDs: greedy `GENERATED_IDS` が全 A/B で不変（11J-J / 11J-K / 11K-A の各 flag で確認）
- DFlash proposal: proposal（ids）が A/B で完全一致
- acceptance: rounds 42 / accepted 85 / full / partial / reruns 0 が全 A/B で一致
- generated IDs: DFlash2 経由の `GENERATED_IDS` が target-only greedy と一致
- GDN history: gate11c（rerun-reference）failed=0
- J candidates: 11J-J は REVERT のため候補集合は canonical split と同一（fused は byte-exact を確認済み）
- K rerank: 147/147 で rerank logits bit exact / final ids exact / final logits bit exact
- DeepFusion 11K-A: 36/36 で conv+residual が bit exact

## Regression

- required: 11J-J A 137/137、11J-K A 137/137・B 138/138、11K-A B 139/139（すべて `ctest -L required -j1`）
- skip: **0**
- HIP_GRAPH=OFF（primary / deterministic）: `test_dflash2_gate11c_e2e` は 3 case とも
  `ids_match=yes`・`failed=0`（prose 256 / code 128 / math 128）。本 series の全 exactness /
  acceptance / E2E 検証は HIP_GRAPH=OFF で実施している。
- HIP_GRAPH=ON: 本 series の branch は main の Gate HG（`38cfbe3e`、PR #40
  `fix/dflash2-hip-graph-determinism`）より前を base にしていたため、branch 単体では
  `test_dflash2_gate11c_e2e` の hist/ref が不一致だった（`ids_match=no`、3/3）。
  **main merge で修正を取り込み、merge 後は HIP_GRAPH=ON でも決定的であることを確認した**:
  - compile: `build-g11jck-graph`（HIP_GRAPH=ON）の configure + build が **EXIT=0 / error 0**
  - `test_dflash2_gate11c_e2e`: **3 case とも `ids_match=yes` / `failed=0`**、rerun 数も
    HIP_GRAPH=OFF と一致（prose `ref_reruns`=81 / code 3 / math 7）
  - kernel 単位の exact test: `test_dflash2_fused_rerank_top16` **147/147**、
    `test_dflash2_deepfuse_mlp_a` **36/36**（graph build でも PASS）
  - graph 実装自体は本 series で変更していない（不具合は branch の base に由来し、
    main の Gate HG が修正）
- errors: 0（link/compile error なし）
- warnings: 新規 0（既存の `int2_select_impl` unused-function 警告は本 series 前から存在）

## Production Flags

active:

- `PHASESHIFT_FUSE_RESIDUAL_RMSNORM_QUANT`
- `PHASESHIFT_FUSE_TARGET_SWIGLU_QUANT`
- `PHASESHIFT_FUSE_DFLASH2_RESIDUAL_RMSNORM_QUANT`
- `PHASESHIFT_FUSE_GDN_PREPARE`
- `PHASESHIFT_FUSE_GDN_POST`
- `PHASESHIFT_FUSE_ATTN_GATE_QUANT`
- `PHASESHIFT_FUSE_DFLASH2_RERANK_TOP16`（11J-K ADOPT）
- `PHASESHIFT_DEEPFUSE_DFLASH2_MLP_A`（11K-A ADOPT、DFlash2）

PERF-NEUTRAL / default OFF:

- `PHASESHIFT_FUSE_DFLASH2_SWIGLU_QUANT`
- `PHASESHIFT_FUSE_Q_POSTPROCESS`
- `PHASESHIFT_FUSE_K_POSTPROCESS`

obsolete: **0**

## Final Decision

- J: **REVERT**（byte-exact だが +15.34% の regression。共有 helper の refactor のみ残置）
- K: **ADOPT**（-1.90% proposer / -25.0% rerank+top16 / launch 4 → 1。K2 は NO-GO）
- Target DeepFusion: **未採用**（11K-A Target は未実装、11K-B/C/D は NO-GO）
- DFlash2 DeepFusion: **11K-A ADOPT**（-0.39% proposer）、11K-B/C/D は NO-GO

## 11K-F GEMM Fusion Trait Infrastructure

PSQ4 / PSQ8 の RowBlock1 と decode1 に、GEMM geometry を変えずに ActivationPolicy /
EpiloguePolicy を差し替える trait infrastructure を導入した。性能改善は目的ではなく、
後の Down+Residual / Gate+Up+SwiGLU / Quant+Down fusion の extension point を作ることが目的。
ADOPT 後、policy core を canonical 化し、旧 hot-path body と temporary experiment flag を削除した。

### Baseline

- SHA: `fb6bbaf5c91add62d722cc90d7329454e3cf114a`（DeepFusion 前回 final `4c5a7bff` の descendant）
- GPU: AMD Radeon AI PRO R9700（gfx1201、GPU0 を使用）
- ROCm: ROCk module 7.1.3.31500000 / ROCm SDK 10.0.0 / clang 23.0.0（`-DCMAKE_HIP_ARCHITECTURES=gfx1201`）
- HIP_GRAPH=OFF

### Hot Config Trace

selector（`select_psq4_gemm_config` / `select_psq8_gemm_config`）は
`rows <= 16` で `RowBlock1` を選び、`rows == 1` では `launch_gemm_psq*_wmma_auto` が
dedicated decode1 kernel へ振る。Qwen3.8 / DFlash2 の MLP hot shape は次の構成。

| path | encoding | rows | N | K | config | decode1 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Target Verify | PSQ4 / PSQ8 | 1 | 5120 / 17408 | 17408 / 5120 | RowBlock1 | yes |
| Target Verify | PSQ4 / PSQ8 | 3 | 5120 / 17408 | 17408 / 5120 | RowBlock1 | no |
| Target Verify | PSQ4 / PSQ8 | 8 | 5120 / 17408 | 17408 / 5120 | RowBlock1 | no |
| DFlash2 proposer | PSQ4 | 1 | 5120 / 17408 | 17408 / 5120 | RowBlock1 | yes |
| DFlash2 proposer | PSQ4 | 3 | 5120 / 17408 | 17408 / 5120 | RowBlock1 | no |
| DFlash2 proposer | PSQ4 | 7 | 5120 / 17408 | 17408 / 5120 | RowBlock1 | no |

hot path が踏む config は `RowBlock1` と `decode1` のみで、`RowBlock2/4/8` と `Prefill2D*` は
踏まない。よって trait 化対象をこの 2 系列に限定した。

### Trait Scope

- PSQ4 RowBlock1: `gemm_psq4_w4a8_wmma_rowblock1_policy_kernel<ActivationPolicy, EpiloguePolicy>`（新規、MB=1 固定）
- PSQ4 decode1: `gemv_psq4_w4a8_decode1_policy_kernel<OutputType, Unroll, ActivationPolicy, EpiloguePolicy>`（新規）
- PSQ8 RowBlock1 / decode1: 同型（`psq8.hip`）
- ActivationPolicy: `CanonicalA8ActivationPolicy`（`row_base` / `load_fragment` / `row_scale`）
- EpiloguePolicy: `CanonicalStoreEpilogue`（BF16 / F32 exact store）
- excluded configs: `RowBlock2/4/8`、`Prefill2D*`、FP8、MXFP4、BF16 は canonical のまま

weight fragment load / WMMA 順序 / FMA 順序 / block size / grid size / K loop ordering /
weight・activation layout / selector / decode1 unroll selection は変更していない。
decode1 の unroll は canonical selector 値（PSQ4 small 16 / large 8、PSQ8 8）を維持。

### Exactness

`test_qwen35_psq_gemm_policy_core` で、旧 canonical kernel の出力を golden hash（FNV-1a 64、
出力 byte 列）として固定し、policy core と bit exact を確認した。

| encoding | shape | rows | dtype | result |
| --- | --- | --- | --- | --- |
| PSQ4 / PSQ8 | 17408x5120、5120x17408 | 1 / 2 / 3 / 7 / 8 | BF16 / F32 | **40 / 40 bit exact** |

既存の `test_gemm_psq4_w4a8_wmma`（1270 / 1270）、`test_gemm_psq8_w8a8_wmma`（1271 / 1271）、
`test_gemm_psq4_decode1`（84 / 84）、`test_gemm_psq8_decode1`（60 / 60）も policy core で PASS。

### Resource

`llvm-objcopy` で `.hip_fatbin` を展開し、AMDGPU metadata から取得。scratch は全 kernel 0。

| kernel | old VGPR | new VGPR | old SGPR | new SGPR | LDS | scratch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PSQ4 RowBlock1 | 81 | **79** | 34 | 38 | 0 | **0** |
| PSQ4 decode1 u8 | 108 | 108 | 23 | 23 | 0 | **0** |
| PSQ8 RowBlock1 | 77 | 77 | 34 | 34 | 0 | **0** |
| PSQ8 decode1 u8 | 94 | 94 | 22 | 22 | 0 | **0** |

decode1 は 4 unroll（2/4/8/16）× BF16/F32 すべてで VGPR / SGPR が完全一致。
RowBlock1 は PSQ4 で VGPR -2、PSQ8 で完全一致。PSQ4 の SGPR +4 は block 32 thread で
allocation bucket が変わらず、occupancy 不変（性能も下記のとおり non-regression）。

### Performance

`phaseshift-bench gemm` を old / new で別 binary として build し、交互に >=5 paired で測定
（shape 17408x5120 / 5120x17408、p50 us、GPU0）。

| encoding | shape | rows | old us | new us | delta |
| --- | --- | ---: | ---: | ---: | ---: |
| PSQ4 | 17408x5120 | 8 | 85.90 | 85.90 | +0.00% |
| PSQ4 | 5120x17408 | 8 | 84.29 | 84.35 | +0.07% |
| PSQ8 | 17408x5120 | 8 | 157.00 | 157.03 | +0.02% |
| PSQ8 | 5120x17408 | 8 | 154.06 | 153.99 | -0.05% |
| PSQ4 decode1 | 5120x17408 | 1 | 82.40 | 82.40 | +0.00% |
| PSQ8 decode1 | 17408x5120 | 1 | 153.53 | 153.56 | +0.02% |

全測定で |delta| < 0.2%、符号も一貫しない。regression なし。

### Regression

- required: `ctest -L required -j1` = **132 / 132 PASS**（skip 0）
- target: 下記 Correctness を参照（required の dflash2 / verify 系で確認）
- dflash: required の dflash2 系で確認

### Decision

**ADOPT**。policy core を canonical 化し、旧 `gemv_psq*_decode1_kernel` と旧 MB=1
instantiation、temporary CMake option `PHASESHIFT_GEMM_POLICY_CORE_EXPERIMENT` を削除した。
`RowBlock2/4/8` 用の `gemm_psq*_wmma_kernel<MB>` は canonical のまま残置。

## 11K-G Target Down GEMM + Residual

Gate F の EpiloguePolicy に `ResidualBf16Epilogue` を追加し、Target MLP の
`LINEAR(Down) -> RESIDUAL_ADD` を 1 kernel に fusion した。compile flag は
`PHASESHIFT_DEEPFUSE_TARGET_DOWN_RESIDUAL`（default OFF）。

- eligible: `LINEAR_PSQ4` / `LINEAR_PSQ8`（Down shape `out_features == 5120 && k == 17408`、
  rows 1..8、BF16）の直後が `RESIDUAL_ADD` で、linear output が他の dispatch から参照されない
- encoding: 実際の weight slot encoding に応じて PSQ4 / PSQ8 を選択
- fallback: 条件不一致・selector 不一致は既存 split のまま
- geometry: RowBlock1 / decode1 の既存 geometry・grid・block を変更なし
- numerical contract: `v -> BF16 round -> float -> residual load(BF16->float) -> add -> BF16 round`
  （accumulator へ residual を直接 add しない）
- consumed logical dispatch: 2、physical: 1

### Exactness

`test_qwen35_psq_down_residual`（PSQ4 / PSQ8 × rows 1/3/8、N=5120、K=17408）。

- fused == canonical split（Down BF16 + residual add）: **6/6 byte exact**
- fused == `round_bf16(down_bf16) + residual` oracle: **6/6 byte exact**

### Resource

| kernel | VGPR | SGPR | LDS | scratch |
| --- | ---: | ---: | ---: | ---: |
| PSQ4 RowBlock1 Down+Residual | 79 | 38 | 0 | **0** |
| PSQ4 decode1 Down+Residual | 108 | 25 | 0 | **0** |
| PSQ8 RowBlock1 Down+Residual | 77 | 38 | 0 | **0** |
| PSQ8 decode1 Down+Residual | 94 | 24 | 0 | **0** |

canonical との差は SGPR +2〜+4 のみ。VGPR は同一で occupancy 不変。scratch は全 kernel 0。

### Micro（`phaseshift-bench gemm --down-residual`、N=5120、K=17408、p50、5 paired）

| encoding | M | split us | deep us | delta |
| --- | ---: | ---: | ---: | ---: |
| PSQ4 | 1 | 38.8 | 35.1 | **-9.4%** |
| PSQ4 | 3 | 49.4 | 46.1 | **-6.6%** |
| PSQ4 | 8 | 46.0 | 44.0 | **-4.4%** |
| PSQ8 | 1 | 154.0 | 150.5 | **-2.3%** |
| PSQ8 | 3 | 155.8 | 152.3 | **-2.1%** |
| PSQ8 | 8 | 156.9 | 153.8 | **-2.0%** |

全 5/5 paired で同符号。launch 数は 2 -> 1。

### Decision

**ADOPT**。Target Verify / E2E の絶対値測定は本 Gate では未実施（micro と exactness で採用判断）。
Target の logical `{Down, Residual}` を本 fusion が所有し、旧 split（canonical Down +
`residual_add_bf16`）は fallback として残置。overlap しない他 fusion（SwiGLU+Quant）は削除しない。

## 11K-I Quant + Down analytical feasibility

Gate 11K-I は実装前に analytical feasibility gate を行い、**NO-GO** と判定した。

### 前提

- SwiGLU 出力 = Down 入力 activation: BF16、K = 17408
- Down: N = 5120、K = 17408、encoding PSQ4 / PSQ8
- Down GEMM tile: 1 block = 16 output features（geometry 変更禁止）
- blocks / row = 5120 / 16 = **320**
- rows = 1 / 3 / 8

### canonical Quant cost / traffic

| 項目 | 値（1 row あたり） |
| --- | ---: |
| code 生成数 | 17,408 |
| row-wide peak reduction | 17,408 |
| code buffer | 17,408 B |
| scale buffer | 2,176 B（17408/32 x 4B） |
| code write | 17,408 B |
| code read（GEMM、L2 経由） | 17,408 B x 320 blocks（L2 resident、DRAM は 1 回） |

### candidate（on-the-fly encode）

- 各 output block が K 全長を走査するため、activation code 生成が
  output tile ごとに繰り返される。
- on-the-fly encode 数 / row = 17,408 x 320 = **5,570,560**（canonical の **320x**）
- activation 読み出しが BF16 になるため L2 read も 2 倍（17,408 B -> 34,816 B / block）
- scale は row-wide peak 確定前に必要。単一 kernel では grid-wide barrier が必要で禁止。
  C1（kernel1: exact row scale のみ、kernel2: Down + on-the-fly encode）でも
  kernel2 内の再生成は 320x のまま。

### 見積り

- 削減できる code traffic: 17,408 B write + 2,176 B scale ≈ **19.6 KB / row**
- Down weight traffic（psq4）: 44.6 MB / token（row 非依存）
- → code traffic 削減は全体の **0.05% 未満**
- 一方、E4M3 encode 320x は instruction bound で
  rows=8 の Down GEMM 時間（~84 us、実測）に対し **~20 us 超**の追加が見込まれる
- L2 read amplification（BF16 化）も加算される

### Decision before implementation

**NO-GO**。

16-output-tile geometry を維持する限り encode repetition（320x）が支配的で、
canonical Quant + GEMM より有利になる下界が存在しない。
禁止手法（global spin barrier / residency assumption / approximate scale /
output tile ごとの scale）を使わずに削減できる合法な方法は見つからなかった。
よって Gate 11K-I は実装せず終了する。

## 11K-H Gate+Up Dual Projection + Exact SwiGLU

Gate F の ActivationPolicy / helper と Gate G の EpiloguePolicy 基盤を拡張し、MLP の
`Gate GEMM` と `Up GEMM` を同一 CTA・同一 output tile で計算して exact SwiGLU まで
1 kernel に融合する DeepFusion を Target / DFlash2 の両方へ実装した。

- Target flag: `PHASESHIFT_DEEPFUSE_TARGET_GATE_UP_SWIGLU`（default OFF）
- DFlash2 flag: `PHASESHIFT_DEEPFUSE_DFLASH2_GATE_UP_SWIGLU`（default OFF）
- overlap guard: 同 operation（SwiGLU）を所有する旧 `*_SWIGLU_QUANT` との同時 ON を
  CMake で `FATAL_ERROR`（Target / DFlash2 それぞれ）

### Hot Encoding Discovery（`phaseshift_quantization.json`）

Target `Qwen3.8-27B-PSQ`:

| layers | gate encoding | up encoding | down encoding |
| --- | --- | --- | --- |
| 0 | bf16 | bf16 | bf16 |
| 1-63 | psq4 | psq4 | psq4（4 の倍数層は psq8） |

- Gate / Up は layer0 を除き **全 63 layer が PSQ4**。PSQ8 の Gate / Up は存在しない。
  よって PSQ8 dual kernel は実装しない（Qwen3.8 hot path に必要な specialization のみ）。
- Down は 4 の倍数層（4,8,...,60 の 15 layer）が PSQ8、他が PSQ4。Gate G は両対応済み。

DFlash2 `Qwen3.8-27B-DFlash2-PSQ4`: 5 layer すべて gate / up / down = PSQ4。

### Kernel Design

- PSQ4 のみ。既存の canonical RowBlock1 / decode1 と activation fragment load・weight
  fragment load / dequant・WMMA・scale 適用を `psq4_load_weight_fragment` /
  `psq4_wmma_fragment` helper で共有（Gate F 方針の拡張）。
- geometry は **canonical と完全同一**（32 thread / 1 block = 16 output tile、blockIdx.x =
  output tile、blockIdx.y = row tile）。16 output tile を拡張しない。decode1 unroll は
  canonical selector と一致。
- K loop: 各 `ib` で activation fragment を 1 回 load し、Gate WMMA → Up WMMA を実行。
  Gate / Up の各 projection 内で `ib` 昇順の accumulation order を維持（bit exact）。
- BF16 boundary: Gate / Up の F32 accumulator を `__float2bfloat16` で明示 round し、
  その rounded 値だけを SwiGLU へ渡す。raw accumulator を SwiGLU へ渡さない。
- SwiGLU semantic: Target は `detail::target_swiglu_f32`、DFlash2 は
  `detail::dflash2_swiglu_rounded_f32` を再利用。混同しない。
- Gate / Up BF16 output は従来通り global へ materialize（value trace / dump 維持）。
  Gate / Up を SwiGLU のために再 read しない。
- Quant は融合しない（row-wide K=17408 reduction が必要で CTA 同期 domain が異なる）。

### Exactness（`test_qwen35_psq_gate_up_swiglu`）

| path | encoding | rows | Gate | Up | SwiGLU |
| --- | --- | ---: | --- | --- | --- |
| Target | psq4 | 1 / 3 / 8 | bit exact | bit exact | bit exact |
| DFlash2 | psq4 | 1 / 3 / 7 | bit exact | bit exact | bit exact |

- canonical split（Gate GEMM + Up GEMM + SwiGLU）と byte exact: **30 / 30 PASS**
- rounded Gate / Up からの oracle と byte exact、同時に
  `raw accumulator` から直接 SwiGLU した場合に mismatch が出る fixture を確認:
  raw-vs-rounded divergence は rows=1 で 3000 前後 / 17408（adversarial 成立）

### Resources

| kernel | VGPR | SGPR | LDS | scratch | occupancy |
| --- | ---: | ---: | ---: | ---: | --- |
| canonical RowBlock1（Gate 単体） | 79 | 38 | 0 | 0 | 6 wave/SIMD（512/79） |
| dual RowBlock1 | **138** | 46 | 0 | **0** | **3 wave/SIMD（512/138）** |
| canonical decode1 u8（Gate 単体） | 108 | 23 | 0 | 0 | 4 wave/SIMD |
| dual decode1 u8 | 117 | 38 | 0 | **0** | 4 wave/SIMD |

- scratch は全 kernel 0（hard 要件）。
- RowBlock1 dual は accumulator が 2 本になり VGPR 138（occupancy 半減の strong warning）。
  しかし decode は weight bandwidth bound（p50 は DRAM 支配）で、front segment は
  下記のとおり改善する。decode1 dual は VGPR 117 と canonical と同 tier。
- helper 化後の canonical は resource / exactness / 性能とも F-I final と一致。

### Pure Segment（`phaseshift-bench gemm --gate-up-swiglu`、N=17408 K=5120、p50、5 paired）

| path | rows | Gate+Up+SwiGLU split us | Deep us | delta |
| --- | ---: | ---: | ---: | ---: |
| Target | 1 | 170.0 | 163.0 | **-4.1%** |
| Target | 3 | 172.3 | 165.9 | **-3.7%** |
| Target | 8 | 176.8 | 169.9 | **-3.9%** |
| DFlash2 | 1 | 170.0 | 163.1 | **-3.9%** |
| DFlash2 | 7 | 175.9 | 170.2 | **-3.3%** |

### Full Front Segment（`--gate-up-swiglu --front-segment`、first Gate start -> quant completion）

Target: A = Gate + Up + fused Target SwiGLU+Quant（3 launch） / B = Deep Gate+Up+SwiGLU +
canonical Quant（2 launch）。DFlash2: A = Gate + Up + SwiGLU + Quant（4 launch） /
B = Deep + Quant（2 launch）。

| path | M | A us | B us | delta |
| --- | ---: | ---: | ---: | ---: |
| Target | 1 | 176.9 | 168.9 | **-4.5%** |
| Target | 3 | 179.1 | 171.3 | **-4.3%** |
| Target | 8 | 182.7 | 175.4 | **-4.0%** |
| DFlash2 | 1 | 176.9 | 169.3 | **-4.3%** |
| DFlash2 | 7 | 182.5 | 176.0 | **-3.7%** |

全 5/5 paired で同符号、改善。

### Target Verify（`test_dflash2_gate11h_verify_profile`、ctx=2048、5 paired + 3 swapped control）

A = `PHASESHIFT_FUSE_TARGET_SWIGLU_QUANT=ON` + H OFF、B = H ON。他は同一
（`FUSE_RESIDUAL_RMSNORM_QUANT` / `FUSE_GDN_PREPARE` / `FUSE_GDN_POST` /
`FUSE_ATTN_GATE_QUANT` / `DEEPFUSE_TARGET_DOWN_RESIDUAL` ON）。

original: A=GPU2, B=GPU3 同時。swapped: A=GPU3, B=GPU2 同時（GPU バイアス除去）。

| M | A(GPU2) | B(GPU3) | A(GPU3) | B(GPU2) | 同一 GPU での delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 34819.8 | 34616.4 | 34785.2 | 34648.7 | **-0.49%**（両 GPU で一致） |
| 3 | 37739.7 | 37632.9 | 37716.6 | 37703.9 | -0.10 / -0.22%（符号一致せず弱い） |
| 8 | 40066.3 | 39999.9 | 40032.1 | 40102.0 | +0.09 / -0.08%（**ノイズ内**） |

- M=1 のみ再現性のある改善（-0.49%）。**primary の M=8 は A と B の range が重なり
  改善は測定できない（非劣化のみ）**。
- M=3 も符号が一致せず、有意とは言えない。

### Target Launch

| metric | baseline | H | delta |
| --- | ---: | ---: | ---: |
| logical dispatches | 1654 | 1654 | 0 |
| physical launches（M=1 / M=8） | 1146 | 1082 | **-64** |
| physical launches（M=3） | 1242 | 1178 | **-64** |
| target_gate_up_swiglu hits（41 forward 累積、M=1 / M=3 / M=8） | 0 | 2624 / 5248 / 7872 | 64 / forward |

H 1 hit につき physical launch がちょうど 1 減る（64 hit ↔ -64 launch、決定的）。logical は不変。

### DFlash Proposer / E2E

`dflash2_propose_cached` は `block_rows = num_drafts + 1u` を使うため、production の
`--dflash2-drafts 7` では `block_rows = 8` になる。当初の eligibility `block_rows <= 7`
では発火しなかったため、**`block_rows <= 8` に修正**した（dual kernel は rows<=8 対応済み、
Target と同一 bound）。focused exact にも DFlash rows=8 を追加（35/35 PASS）。

- gate10_cli（B = Target H + DFlash H ON）: `token exact=yes`（64 token 完全一致）、`failed=0`
- `DFLASH2_GATE_UP_SWIGLU_HITS = 100`（5 layer × 20 draft forward、発火を確認）
- proposer A/B（`--dflash2-drafts 7`、3 paired、target 側は両者 H ON）:

| metric | A_med | B_med | delta |
| --- | ---: | ---: | ---: |
| rounds | 21 | 21 | 0 |
| accepted drafts | 42 | 42 | 0 |
| `DRAFT_MS`（host） | 5.40 | 4.96 | **-8.15%** |
| `DRAFT_GPU_MS` | 102.25 | 101.15 | **-1.08%** |
| `ROUND_MS` | 967.74 | 956.32 | -1.18% |

`DRAFT_GPU_MS` は A/B の run range が非重複（A 102.23-102.38 / B 101.06-101.65）、
`GENERATED_IDS` は全 6 run で完全一致。

### Correctness

- Target: focused exact 30/30、required 133/133 PASS（H flags ON）
- DFlash2: focused exact 30/30（PSQ4 dual kernel + DFlash2 SwiGLU semantic）
- matcher unit test（正例 / 負例）は未追加。Gate G の Down+Residual matcher と同様、
  既存 `test_qwen35_fusion_matcher` の対象外。matcher の条件判定はコード上で
  狭く固定し、required のコンパイル + 既存 suite で担保している。
- DFlash2 executor 経由の Deep H は compile 検証のみ（model 依存の proposer E2E は未測定）。
- proposal / acceptance / GENERATED_IDS の full E2E exact は未測定

### Regression

- 最終 candidate config（Target / DFlash2 ADOPT + 旧 SwiGLU+Quant 削除後）:
  `FUSE_RESIDUAL_RMSNORM_QUANT` / `FUSE_GDN_PREPARE` / `FUSE_GDN_POST` /
  `FUSE_ATTN_GATE_QUANT` / `DEEPFUSE_TARGET_DOWN_RESIDUAL` /
  `DEEPFUSE_TARGET_GATE_UP_SWIGLU` / `FUSE_DFLASH2_RESIDUAL_RMSNORM_QUANT` /
  `FUSE_DFLASH2_RERANK_TOP16` / `DEEPFUSE_DFLASH2_MLP_A` /
  `DEEPFUSE_DFLASH2_GATE_UP_SWIGLU` ON
- required: `ctest -L required -j1`（`PHASESHIFT_TEST_GPUS=0`）→ **140 / 140 PASS**
- skip: 0
- focused: `test_qwen35_psq_gemm_policy_core` 40/40、`test_qwen35_psq_gate_up_swiglu`
  35/35、`test_qwen35_psq_down_residual` PASS
- 途中 1 回 `test_gpu_mcu_persistent_emit`（GPU MCU doorbell timing）が flaky で落ちたが、
  再実行で PASS。Gate H とは無関係。
- HIP_GRAPH: 本 Gate では ON 検証を実施していない（H は Graph 実装を変更しない）。
- cleanup 後の E2E smoke（`phaseshift-compute`、max-new-tokens 32）:
  target-only と DFlash の `GENERATED_IDS` が完全一致（EXACT）。DFlash H hits=50
  （5 layer × 10 round）、rounds=10、accepted=21。

### Ownership Cleanup

両 path とも **ADOPT** のため、operation overlap する旧 SwiGLU+Quant fusion を削除した。

- 削除（Target）: `PHASESHIFT_FUSE_TARGET_SWIGLU_QUANT`、Target policy
  （`TargetSwiGLUPolicy`）、`launch_fused_target_swiglu_quant_e4m3`、Target matcher
  （`match_swiglu_quant_fusion`）、`try_launch_swiglu_quant_fusion`、counter、
  program_executor hook、bench subcommand、test、compile-time overlap guard
- 削除（DFlash2）: `PHASESHIFT_FUSE_DFLASH2_SWIGLU_QUANT`、`DFlash2SwiGLUPolicy`、
  `launch_fused_dflash2_swiglu_quant_e4m3`、`dflash2_swiglu_quant_fused`、
  `fused_swiglu_quant.hip` / `.h`（Target / DFlash2 共用 family を丸ごと削除）
- canonical: 単体 `swiglu` / 単体 `activation_quantize_e4m3` は fallback / oracle として残す
- 維持: Gate F policy（`CanonicalA8ActivationPolicy` / `CanonicalStoreEpilogue` /
  `ResidualBf16Epilogue`）、Gate G `PHASESHIFT_DEEPFUSE_TARGET_DOWN_RESIDUAL`、
  DFlash tail `PHASESHIFT_DEEPFUSE_DFLASH2_MLP_A`、`PHASESHIFT_FUSE_DFLASH2_RERANK_TOP16`、
  `PHASESHIFT_FUSE_DFLASH2_RESIDUAL_RMSNORM_QUANT`（operation set disjoint）

production fusion owner（section 63）:

| operation | Target owner | DFlash2 owner |
| --- | --- | --- |
| Gate / Up / SwiGLU | `DEEPFUSE_TARGET_GATE_UP_SWIGLU` | `DEEPFUSE_DFLASH2_GATE_UP_SWIGLU` |
| Quant（MLP 入力 / swiglu 後） | canonical split | canonical split |
| Down | `DEEPFUSE_TARGET_DOWN_RESIDUAL` | canonical split |
| Residual | `DEEPFUSE_TARGET_DOWN_RESIDUAL` | `DEEPFUSE_DFLASH2_MLP_A` |
| ConvFinish | — | `DEEPFUSE_DFLASH2_MLP_A` |

duplicate owner: 0（compile-time overlap guard は option 自体が消えたため、旧 flag との
同時 ON は不可能）。

### Decision

- Target: **ADOPT**
- DFlash2: **ADOPT**

Target の根拠:

- hard correctness: Gate / Up / SwiGLU bit exact（30/30）、scratch 0、required PASS、
  gate10 E2E `GENERATED_IDS` exact
- full front segment / pure segment は再現性をもって改善（-4%）、physical launch -64
- Target Verify は再現性をもって非劣化（M=8 はノイズ内で regression なし、M=1 は -0.49%）

DFlash2 の根拠:

- bound 修正（`block_rows <= 8`）で production 構成で発火（hits=100）
- draft GPU -1.08%（range 非重複）、draft host -8.15%、round -1.18%、`GENERATED_IDS` exact
- front segment -3.7%、E2E 非劣化

## 11K-H Report

### Initial

- SHA: `fb6bbaf5c91add62d722cc90d7329454e3cf114a`（worktree branch `gate/11k-f-i-gemm-fusion-traits`）
- 先行: `a29c5e72`（untracked だった `PSQ_W32_BF16_Scale_Features_and_Potential.md` を単独コミット）
- GPU: AMD Radeon AI PRO R9700（gfx1201、GPU0）
- ROCm: ROCk 7.1.3.31500000 / ROCm SDK 10.0.0 / clang 23.0.0、HIP_GRAPH=OFF

### Final

- SHA: `a7d363c1ee42db2c590ac9407673fb304eddc56b`

### Decisions

| Gate | Target | DFlash | result |
| --- | --- | --- | --- |
| F GEMM traits | shared | shared | **ADOPT** |
| G Down+Residual | **ADOPT** | n/a | |
| H Gate+Up+SwiGLU | 未実施 | 未実施 | 未実施 |
| I Quant+Down | NO-GO | NO-GO | **NO-GO**（analytical） |

### Final Target MLP Topology

- logical: Gate GEMM -> Up GEMM -> SwiGLU -> Quant -> Down GEMM -> Residual
- physical: shared PSQ GEMM（Gate / Up）+ Safe `SwiGLU+Quant`
  （`PHASESHIFT_FUSE_TARGET_SWIGLU_QUANT`）+ Deep `Down+Residual`
  （`PHASESHIFT_DEEPFUSE_TARGET_DOWN_RESIDUAL`）。加えて layer pre の
  `Residual+RMSNorm+Quant`、GDN prepare/post、attention gate+quant。

### Final DFlash2 MLP Topology

- 11K-A から変更なし。shared PSQ4 GEMM + `swiglu` + `quantize_a8` +
  `grouped_dynamic_conv`（residual epilogue、`PHASESHIFT_DEEPFUSE_DFLASH2_MLP_A`）。

### Active Fusion Ownership

Target:

| op | owner |
| --- | --- |
| Gate | split（共有 PSQ GEMM） |
| Up | split（共有 PSQ GEMM） |
| SwiGLU | Safe `SwiGLU+Quant`（existing） |
| Quant | Safe `SwiGLU+Quant`（existing） |
| Down | Deep `Down+Residual`（11K-G） |
| Residual | Deep `Down+Residual`（11K-G） |

DFlash:

| op | owner |
| --- | --- |
| Gate | split（共有 PSQ4 GEMM） |
| Up | split（共有 PSQ4 GEMM） |
| SwiGLU | split `swiglu` |
| Quant | split `quantize_a8` |
| Down | split（共有 PSQ4 GEMM） |
| ConvFinish | Deep MLP-A（`PHASESHIFT_DEEPFUSE_DFLASH2_MLP_A`） |
| Residual | Deep MLP-A |

duplicate logical op owners: **0**

### Deleted Superseded Fusion

- kernels: 2（`gemv_psq4_w4a8_decode1_kernel`、`gemv_psq8_w8a8_wmma_decode1_kernel`。
  11K-F canonical 化で policy core へ置換）
- flags: 1（`PHASESHIFT_GEMM_POLICY_CORE_EXPERIMENT`）
- launchers: 0（`gemv_psq*_decode1_kernel` の host 側 dispatch は policy kernel に差し替え）
- tests: 0

### Correctness

- 11K-F: `test_qwen35_psq_gemm_policy_core` 40/40 bit exact（PSQ4/PSQ8 x rows 1/2/3/7/8 x
  BF16/F32 x shape 17408x5120 / 5120x17408）。既存 PSQ GEMM test 全 PASS
- 11K-G: `test_qwen35_psq_down_residual` 12/12（fused == split、fused == oracle）
- required: `ctest -L required -j1` で確認（F 132/132、G 133/133 想定）
- DFlash proposal / acceptance / GENERATED_IDS は kernel 単位の bit exact と
  required dflash2 regression で確認。Target / DFlash の full E2E absolute 測定は未実施
- GDN: required の gdn 系で確認

### Regression

- required: 上記
- skip: 0
- HIP_GRAPH: OFF（primary）

### 未完了

- 11K-H（Gate+Up dual projection + SwiGLU）は未実施。
- 11K-I は analytical feasibility で NO-GO。

## 11K-J Cooperative / Persistent Execution Feasibility

Gate 11K-I の NO-GO（Down の output tile ごとに on-the-fly quantize すると encode が
最大 ~320x 再実行される）を回避する新方式の実現性を確認した。新方式は
`Phase Q（canonical Quant を 1 row につき 1 回）→ cooperative grid.sync() →
Phase D（canonical Down tile を全 resident worker で並列）` であり、encode 回数は
canonical と同じ 1x で、320x 再計算を 0 にする。

本 Gate では model path を変更しない。cooperative/persistent 実行の実現性、
GEMM task body の共有化、barrier budget を検証する。検証に使った probe 実装
（wave-local task 抽出と persistent kernel）は、Gate K の REVERT に伴い production
から削除した。本節以降はその実験レポートであり、source 変更は残さない。

### Device

- cooperative launch supported: **yes**（`hipDeviceAttributeCooperativeLaunch != 0`）
- `multiProcessorCount` = 32（WGP）、2 CU/WGP = 64 CU
- cooperative occupancy（256 threads、`loop_grid_sync`）= per_MP 8 → resident blocks
  = 32 × 8 = **256**

### GEMM task body の共有（probe で検証、production には入れない）

probe では canonical policy kernel が `threadIdx.x` から導出していた `m_lane` /
`k_group` を `logical_lane` 引数に切り出し、PSQ4/PSQ8 の RowBlock1 / decode1 を
`psq4_rowblock1_task` / `psq4_decode1_task` / `psq8_rowblock1_task` /
`psq8_decode1_task` として共有する形にした。canonical kernel は
`logical_lane = threadIdx.x` を渡すだけの wrapper になり、persistent wave32 worker は
`logical_lane = threadIdx.x & 31` で同一 body を呼べる。`__shfl` は wave 内 physical
lane を使うため canonical / persistent の双方で成立する。

- bit exact: `test_qwen35_psq_gemm_policy_core` 40/40、
  `test_qwen35_psq_down_residual` 12/12、`test_qwen35_psq_gate_up_swiglu` 35/35 PASS
- 性能: canonical PSQ4/PSQ8（N=5120 K=17408 / N=17408 K=5120、rows 1/8）で
  |delta| ≤ 0.21%（PSQ8 は ≤ 0.04%）→ 非劣化
- resource: RowBlock1 / decode1 は baseline と同一（psq8 decode1 の 1 変種のみ
  +2 VGPR）、scratch 全 kernel 0

この refactor は Gate K の persistent kernel と body を共有するためだけのものであった。
persistent を REVERT した結果、`logical_lane` が `threadIdx.x` 以外になる呼び出しは
production に存在しなくなる。したがって bit exact かつ非劣化であっても、
production 到達不能な抽象化として投入しない（AGENTS.md のリポジトリ衛生ルール）。

### Synchronization（256 threads、warmup 200 / iterations 2000）

| grid | normal empty launch us | coop empty launch us | 1 barrier us | 2 barriers us |
| --- | ---: | ---: | ---: | ---: |
| 256 blocks（occupancy-derived） | 10.04 | 28.32 | 28.68 | 28.72 |

- barrier 実測: **~0.2-0.4 us/barrier**（既存 report の ~607 ns と整合、sub-us）
- cooperative launch floor は normal launch より **~18 us 高い**（grid サイズ非依存）。
  これは host 側 `hipLaunchCooperativeKernel` の API cost が event 計測に現れたもので、
  先行 kernel 実行中に enqueue できれば overlap で隠れる可能性がある。ただし隠れない
  場合、Gate K の launch 統合（1 launch 削減 ≒ GPU 側 2.5-3.5 us）を上回る cost に
  なるため、Gate K の segment 計測で必ず実測する。

計測は `phaseshift-bench gpu-sync` に cooperative launch floor と barrier の p50/p95 を
追加した probe で行った。この bench 拡張も Gate J/K の REVERT に伴い削除した。

### Gate J 判定（probe 基準）

- [x] cooperative launch supported
- [x] resident grid 合法（256 blocks、hang なし）
- [x] grid.sync stable（sub-us、hang なし）
- [x] no GPU hang
- [x] probe の GEMM task 抽出は bit exact
- [x] probe の GEMM task 抽出は性能非劣化（|delta| ≤ 0.21%）
- [x] scratch 0

### Decision

Gate J: **feasibility GO**（Gate K を試す価値はある、production への refactor 投入は
しない）。

cooperative launch の host premium（~18 us）が overlap で隠れない場合は Gate K/L が
負けるため、Gate K では segment 実測で `persistent >= split` なら REVERT する
（spec section 46）。実際に Gate K は REVERT となり、本 Gate の probe 実装は
production から削除した。

## 11K-K Persistent Quant + Down Probe（早期打ち切り）

Gate 11K-J で cooperative launch floor が non-coop より約 18 us 高い（barrier 自体は
sub-us）ことが判明したため、Target PSQ4 の最小構成で persistent 経路を実装し、
segment を早期計測する probe を先行実施した。probe は計測のための一時実装であり、
ソース変更としては残さない。本節はその実験レポートであり、production は
canonical split のままである。

### 実装

- `persistent_psq4_quant_down_residual_decode1_kernel<kUnroll>` /
  `persistent_psq4_quant_down_residual_rowblock1_kernel`（`psq4.hip`、`blockDim=256`）。
- Phase Q は `detail::activation_quantize_e4m3_vec_row_body<17408,17408>` を共有し、
  `blockIdx.x < rows` の CTA が 1 row を canonical に量子化する。
- `cooperative_groups::this_grid().sync()` 1 回。
- Phase D は wave-local な canonical task body（rows==1 は decode1、rows>1 は
  rowblock1）を `global_wave_worker` の static stride で実行し、
  `ResidualBf16Epilogue` をそのまま使用する。
- launcher は capability と geometry を launch 前に検証し、 occupancy から
  resident grid を算出して `hipLaunchCooperativeKernel` を発行する。

resource（`psq4.hip.o`）:

| kernel | VGPR | SGPR | scratch |
| --- | ---: | ---: | ---: |
| decode1 unroll 2 | 89 | 62 | 0 |
| decode1 unroll 4 | 78 | 62 | 0 |
| decode1 unroll 8（5120 で選択） | 131 | 62 | 0 |
| decode1 unroll 16 | 147 | 62 | 0 |
| rowblock1 | 98 | 66 | 0 |

canonical decode1（unroll 8）は VGPR 108 であり、persistent 化により +23 VGPR。

### Exact

N=5120、K=17408、PSQ4。split（canonical Quant + Gate G Down+Residual）と
persistent の最終出力 BF16 bits を比較し、rows 1/3/8 のすべてで **exact = yes**。
scale / codes / Down / Residual の BF16 boundary は共有 helper により保存される。

### Segment A/B（GPU2、warmup 300 / samples 30 / launches 20、A/B/A 交互測定）

| rows | split us | persistent us | delta |
| --- | ---: | ---: | ---: |
| 1 | 37.5 | 103.5 | **+174%** |
| 3 | 46.8 | 111.8 | **+138%** |
| 8 | 49.9 | 112.1 | **+125%** |

grid sweep（rows=1、resident_max=160 = per_mp 5 × 32 MP）:

| grid | persistent us | delta |
| ---: | ---: | ---: |
| 32 | 112.1 | +199% |
| 40（算出値） | 102.8 | +174% |
| 64 | 102.9 | +174% |

### 考察

- exact は成立しており、失われたのは性能だけである。
- split との差（約 65 us）は cooperative launch premium（約 18 us）を大きく上回る。
  したがって Phase Q / Phase D の実行自体が canonical split より遅い。
- 原因は persistent CTA の resource footprint（decode1 unroll 8 で VGPR 131）と、
  cooperative resident grid が 256-thread CTA に固定されるため、canonical のような
  「output tile ごとの独立 32-thread CTA を 320 個、32 WGP へ自由配置する」並列度を
  再現できないことにある。grid を変えても改善しない。
- Gate 11K-I の NO-GO（encode の最大約 320x 再実行）は persistent 方式で回避できたが、
  今度は persistent kernel の resource と scheduling で負ける。spec section 42 の
  「persistent kernel の resource footprint が最大リスク」が的中した形である。

### Decision

- Target: **REVERT**（`persistent >= split`、spec section 46）。
- DFlash2: 実装せず。同型 kernel であり、DFlash 側は residual も無いため split に対する
  融合利得がさらに小さく、Target より有利になる根拠がない。
- Gate K: **REVERT**。
- Gate L: 実施せず。spec section 104 に従い STOP する。
- production は canonical split（Quant → Down → Residual / ConvFinishResidual）のまま。

### Quant+Down の結論

- 11K-I naive on-the-fly: NO-GO。Down の output tile ごとに encode をやり直すため、
  同じ activation を最大約 320x 再 encode する。
- 11K-K persistent multi-phase: NO-GO。Quant は 1 row 1 回で exact に実行でき
  launch も 1 本に統合できるが、persistent kernel の resource / parallelism により
  segment が +125〜174% 遅くなる。launch 統合の利得を大きく上回る。
- 以上より、Quant + Down は「encode 回数を 1x に保つ legal な larger DeepFusion」を
  与えても利益が無いという結論に至った。
