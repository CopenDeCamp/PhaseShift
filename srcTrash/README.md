# srcTrash

削除予定コードの一時保管場所。

- isolation date: 2026-10-03
- isolation 前の commit SHA: `7e27739ceb2d2fc6228728fb1fd81fb188a10fa0`
  (`docs(refactor): 全機能棚卸しの監査資料を追加する`, branch `refactor/src-trash-isolation` 起点)

## ルール

- srcTrash は削除予定コードの一時保管場所である。
- 元の relative path を維持する(`foo/bar.cpp` → `srcTrash/foo/bar.cpp`)。
- `srcTrash/` は CMake target / include path / test target に**追加しない**。
- srcTrash 内のコードをコンパイルされる状態にしない。
- restore する場合は Git history または srcTrash から戻す。
- `srcTrash/fragments/` には、独立ファイルではなく既存ファイルの一部だけを
  隔離した断片を `srcTrash/fragments/<元パス>/<feature名>.txt` として保存する。
- `docs/rnd/` は履歴資料としてリポジトリに残している(移動しない)。
- production build は `srcTrash` を一切参照しない
  (`rg "srcTrash" CMakeLists.txt cmake src include tests` が 0 件)。

## 隔離一覧

### 1. R&D tools

- **reason**: tests / CMake / `docs/user` / `docs/developer` から参照が 0 件の
  単発 Gate・解析 script。AGENTS.md の tools ルール(R&D 記録からしか参照されない
  script を残さない)に抵触していた。
- **original path**: `tools/rnd/`(43ファイル)、`tools/poc/clock_residency/`(4ファイル)、
  `tools/quantization/{psq4.py, analyze_su_channel_scaling.py, analyze_e4m0_codebook.py,
  analyze_scale_schemes.py, analyze_format_accuracy.py}`、
  `tools/extract_kernel_resources.py`、`tools/gen_token_corpus.py`
- **fallback path**: Git history / `docs/rnd/**` の記述
- **moved tests**: なし
- **removed CLI/env**: なし
- **removal rationale**: production から隔離。`docs/rnd/README.md` に中央注記を追加
  (個別の rnd 記述本文は履歴として変更していない)。
- **restore note**: `git mv` 済み。元 path から戻せる。

### 2. PA probe / paged-prune

- **reason**: KV page pruning 研究は runtime 未統合。probe は **call site が 0 件**
  (`probe_dump_inputs` / `probe_dump_output` は定義のみ)で、`PHASESHIFT_PA_PROBE=ON`
  にしても dump は生成されなかった。
- **original path**: `src/apps/bench/paged_prune.cpp`(642行)、
  `src/phaseshift/models/qwen35/runtime/paged_attention_dispatch.hip` の probe block
  (include 9行 + 実装 221行)
- **fallback path**: fragment に pre-image を保存
  (`srcTrash/fragments/src/phaseshift/models/qwen35/runtime/paged_attention_dispatch_PA_PROBE.txt`)
- **moved tests**: なし(`paged-prune` は `test_bench_help_*` の対象外)
- **removed CLI/env**: `phaseshift-bench paged-prune` subcommand、
  `PHASESHIFT_PA_PROBE` CMake option、`PHASESHIFT_PA_PROBE_DIR` / `_LAYER` / `_CONTEXT`
- **removal rationale**: 既定ビルドと同一挙動(既定 OFF かつ未呼び出し)。
- **restore note**: fragment + Git history。

### 3. NgramTail

- **reason**: 既定 OFF、既存実測で 8 workload 平均 **−1.10%**(改善2/悪化6、
  最悪 −3.93%)。既定 ON は NO-GO 決定済み(`docs/rnd/spec_decode/ngram_tail_gate3.md`)。
- **original path**: `include/phaseshift/models/qwen35/runtime/ngram_tail.h`、
  `src/phaseshift/models/qwen35/runtime/ngram_tail.cpp`、
  DFlash2/CLI/docs からの統合部
- **fallback path**: `dflash2_spec_decoder.{h,cpp}` / `main.hip` の該当行は Git diff、
  `docs/rnd/spec_decode/ngram_tail_*.md` は履歴として残存
- **moved tests**: `test_ngram_tail.cpp`、`test_qwen35_ngram_tail_gate1.hip`、
  `test_dflash2_ngram_tail_gate2.hip`
- **removed CLI/env**: `--dflash2-ngram-tail` / `--dflash2-ngram-n`、
  統計行 `DFLASH2_NGRAM_*` / `DFLASH2_TAIL_*`
- **restored docs**: `docs/now_ngram.md` → `srcTrash/docs/now_ngram.md`
- **restore note**: DFlash2 の通常 proposal → target verify 経路は無変更。

### 4. DecodeBackend::GpuMcu stub

- **reason**: production 到達不能な future placeholder(常に `Status::unsupported`)。
- **original path**: `decode_backend.{h,cpp}` の該当分岐、`main.hip` の help/表示、
  ctest の gpu-mcu case、docs/user・docs/developer の記述
- **fallback path**: `srcTrash/fragments/src/phaseshift/models/qwen35/runtime/decode_backend_GpuMcu.txt`
- **moved tests**: `test_compute_decode_backend_gpu_mcu`(ctest case)。
  `test_decode_backend_contract.cpp` は host contract のみに縮小
- **removed CLI/env**: `--decode-backend gpu-mcu` の受理
- **removal rationale**: AGENTS.md「production 到達不能な future placeholder を作らない」。
  GPU-MCU 構想の破棄ではなく stub の除去。
- **restore note**: fragment + Git history。

### 5. message_codec dead helper

- **reason**: `message_codec.resolve_tool_choice` は call site 0(production は
  `tool_constraint.resolve_tool_choice` を使用)。
- **original path**: `src/apps/server/backend/message_codec.py`(208行 → 180行)
- **fallback path**: `srcTrash/fragments/src/apps/server/backend/message_codec_resolve_tool_choice.py.txt`
- **moved tests**: なし(`test_tool_constraint_builder` は tool_constraint 版を使用)
- **removed CLI/env**: なし
- **restore note**: `tool_constraint.resolve_tool_choice` は production 実装として無変更。

### 6. MTP runtime stack

- **reason**: production entrypoint(compute / cli / server)から call 0 件。
  `docs/developer/mtp.md` が contract 化していた「serve path 未接続」状態の整理。
- **original path**: `include/.../runtime/{mtp_executor.h, mtp_kv_state.h, spec_decoder.h}`、
  `src/.../runtime/{mtp_executor.hip, mtp_kv_state.cpp, spec_decoder.cpp}`、
  `src/apps/bench/mtp.hip`、`tg.hip` の MTP 統合(123箇所)、
  MTP tools(`compare_mtp_gate1/2.py`、`inspect_qwen35_mtp_weights.py`、
  `tools/reference/export_qwen35_mtp_gate1/2.py`)、`docs/developer/mtp.md`
- **fallback path**: `spec_decode.{h,cpp}` の MTP 専用部は fragment
  (`srcTrash/fragments/{include,src}/phaseshift/models/qwen35/runtime/spec_decode_mtp.*.txt`)。
  `docs/rnd/mtp/**` は履歴として残存
- **moved tests**: 12件(`mtp_forward_real` / `gate3_replay` / `gate4e` / `kv_cache` /
  `multistep` / `primitives` / `spec_perf` / `spec_real` / `state` / `state_real` /
  `verify_divergence` / `spec_transaction`)+ `tests/support/mtp_kv_harness.h`
- **removed CLI/env**: `phaseshift-bench mtp`、`tg --mtp` / `--draft-k` / `--mtp-chain`、
  `PHASESHIFT_MTP_NGRAM_*`(NgramTail 連携)
- **KEEP(手順書からの逸脱・承認済み)**:
  - `lower_qwen35_mtp_to_primitives` / `validate_mtp_geometry` は loader 幾何の
    contract として残す。完全削除は MTP tensor skip-load と同梱の別 ticket。
  - `test_qwen35_mtp_weight_load` / `test_qwen35_mtp_weight_real` /
    `test_qwen35_mtp_lowering` は残す。
  - `spec_decode` の `SpecVerifyResult` / `spec_greedy_accept` /
    `spec_gdn_snapshot` / `spec_gdn_restore` / `spec_gdn_*_bytes` は DFlash2 が使用するため KEEP。
  - `test_qwen35_spec_verify.cpp` は KEEP(test 内の `SpecTransaction` 4ブロックのみ fragment 化)。
- **追加変更(承認済み)**: MTP 層ロードを
  `Qwen35LoadOptions::load_mtp_layers`(**既定 OFF** の opt-in)へ変更。
  production は MTP tensor を読み込まなくなった。
  KEEP tests は `load_mtp_layers = true` を明示設定。
- **restore note**: required は128件(隔離前134件、うち MTP 4件を削除)。

### 7. Target approximate LM-head proxy

- **reason**: INT2 coarse → Top-N 候補 prune による argmax は **非 exact(確率的)**。
  greedy equivalence 破りの実測あり。既定 OFF だったが、無効時の VRAM 割当も
  発生していた。
- **original path**: `include/.../runtime/lm_head_proxy.h`(172行)、
  `src/.../runtime/lm_head_proxy.hip`(894行)、shadow/certification 関連
- **fallback path**: **ExactCandidates は新モジュールへ救出**
  (`include/.../runtime/constraint_lm_head_exact.h` +
  `src/.../runtime/constraint_lm_head_exact.hip`、env は
  `PHASESHIFT_CONSTRAINT_LM_HEAD_EXACT`、既定 OFF のまま)。
  MaskedCoarse / proxy テストは `srcTrash/fragments/tests/unit/test_constraint_candidates_masked_topn.txt`
  に縮小保存
- **moved tests**: 12件(`test_lm_head_proxy_path`、`test_target_lm_proxy_{certificate,
  error_bound, fallback, real_hidden, real_perf, real_probe, upper_topn}`、
  `test_verify_lm_proxy_{direct, real_verify}`、`test_lm_head_residual_survey`、
  `test_lm_head_constraint_perf`)+ `tests/support/target_lm_proxy_harness.h`
- **removed CLI/env**: `PHASESHIFT_TARGET_LM_HEAD_PROXY` / `_POOL` /
  `_CONSTRAINT_THRESHOLD` / `_TIMING`
- **measured result(既存docs値)**: proxy 有効時 DFlash2 verify 65.20 tok/s、
  停止 63.55 tok/s(約 2.4% の差を「argmax 保証の代償」として受入)。
  認証カーネルは不採用理由として GPU busy +3.0% / wall −31%。
- **restore note**: exact path(constraint LM head)は active code に残存。

### 8. DFlash2 INT2 coarse head

- **reason**: opt-in の提案側近似(既定 OFF)。production default では
  full PSQ8 → top-K → candidate selector 経路のみが使われる。
- **original path**: `include/.../kernels/dflash2/draft_head_int2.h`、
  `src/.../kernels/dflash2/{draft_head_int2.hip, coarse_topn.hip}`、
  `detail/coarse_head_device.h`、DFlash2 executor の INT2 経路
  (`dflash2/executor.hip` 2856行 → 1833行)
- **fallback path**: 残すべき宣言は専用 header へ分離済み
  - radix: `include/.../kernels/dflash2/radix_topn.h`(新規、production の
    stochastic radix top-k 経路が使用)
  - PSQ8 rerank: `include/.../kernels/optimized/psq8_candidate_rerank.h` +
    `src/.../kernels/optimized/psq8_candidate_rerank.hip`(新規、ExactCandidates が使用)
- **moved tests**: `test_dflash2_coarse_topn`、`test_dflash2_int2_coarse_head`、
  `test_dflash2_int2_pack`。radix 系
  (`test_dflash2_radix_topn{,_perf}`、`test_stochastic_topk`)は KEEP
  (legacy topn 比較部分のみ fragment 相当の削除)
- **removed CLI/env**: `PHASESHIFT_DFLASH2_INT2_HEAD` / `_INT2_CODEBOOK` /
  `_INT2_DIAG` / `_INT2_TIMING` / `_DRAFT_RERANK`
- **measured result(既存docs値)**: INT2 化で +2.4%(`docs/rnd/dflash2/int2_head_gate1.md`)。
  既定 OFF のため削除時の既定性能影響はゼロ。
- **KEEP**: `radix_topn.hip` は `sampling_dispatch.hip` の stochastic top-k
  production path が使うため削除禁止(確認: `launch_dflash2_radix_topn` の
  production caller が残っている)。
- **restore note**: `detail/coarse_topn_device.h` は radix が使うため active に残存。

### 9. DFlash2 fixed draft vocabulary

- **reason**: INT2 head とセットの opt-in 機能(既定 off、既定化は却下済み)。
- **original path**: `include/.../dflash2/draft_vocab_profile.h`、
  `src/.../dflash2/draft_vocab_profile.cpp`、`tools/quantization/prepare_draft_vocab.py`
- **fallback path**: Git history + `docs/rnd/dflash2/fixed_vocab_default.md`(履歴)
- **moved tests**: `test_dflash2_draft_vocab_profile.cpp`、`test_prepare_draft_vocab.py`
- **removed CLI/env**: `PHASESHIFT_DFLASH2_DRAFT_VOCAB` / `_DRAFT_VOCAB_FILE` /
  `_DRAFT_VOCAB_CHECK`、`DFLASH2_DRAFT_VOCAB=` stderr 出力
- **measured result(既存docs値)**: 約 +4%(coverage 依存、既定化却下)。
- **restore note**: DFlash2 executor の proposal 経路は full PSQ8 のまま無変更。

## 統計

- 隔離した独立ファイル: tools 61 / tests 34 / src 11 / include 7 / docs 3
- 隔離した断片(fragment): 7
- required acceptance(実測): 隔離前 **134件 PASS** → 隔離後 **117件 PASS**
  - Phase 3 (NgramTail): `test_ngram_tail` −1 → 133
  - Phase 4 (GpuMcu): `test_compute_decode_backend_gpu_mcu` −1 → 132
  - Phase 5 (MTP): `mtp_primitives` / `mtp_kv_cache` / `mtp_state` / `spec_transaction` −4 → 128
  - Phase 6 (LM head proxy): `target_lm_proxy_{error_bound,upper_topn,certificate,fallback}` /
    `verify_lm_proxy_direct` / `lm_head_proxy_path` −6 → 122
  - Phase 7 (INT2 / fixed vocab): `dflash2_draft_vocab_profile` / `prepare_draft_vocab` /
    `dflash2_int2_pack` / `dflash2_int2_coarse_head` / `dflash2_coarse_topn` −5 → 117
- 正しさ契約の再現(Phase 8 実機確認):
  - `phaseshift-compute`(DFlash2) `GENERATED_IDS` sha1 `47aebe55d048`、
    `DFLASH2_ROUNDS=84`、`DFLASH2_ACCEPTED_DRAFTS=171`、`DFLASH2_GDN_MODE=compact` が一致
  - `phaseshift-bench tg` `GREEDY_TOKEN_SUM=2446188` が一致
  - `tests/server/test_compute_constraints.py` 18/18 PASS(`invalid-grammar-no-fallback` 含む)
