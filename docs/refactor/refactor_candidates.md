# リファクタリング候補(4段階整理)

**本ドキュメントはまだ何も変更しない。** 棚卸し結果に基づく候補整理のみ。

区分:

- **KEEP**: 現行機能として維持する
- **SIMPLIFY**: 機能は維持するが実装を単純化する余地がある
- **ISOLATE**: production 経路から分離(ターゲット分割・モジュール分離)する価値がある
- **REMOVE CANDIDATE**: 削除候補(削除には意思決定と代替確認が必要)

判断根拠はすべて call site / CMake / default / runtime 分岐 / tests の確認結果。
性能値は既存docs値のみ。

---

## KEEP

| 対象 | 理由 |
| --- | --- |
| model load / weights loader / safetensors | Basic。削除で起動不能 |
| HF chat template + Transformers response parser(`codec.py`) | Basic。prompt 生成と tool parse の本体 |
| prefill / decode / sampling / paged KV | Basic |
| Chat Completions + SSE + tool loop A〜E | Basic(OpenCode の最小構成、[opencode_minimum.md](opencode_minimum.md) 参照) |
| JSONL serve transport(`--serve-stdio`) | Basic |
| LocalAI patches 0001・0003(fail-closed / tool policy transport) | Basic handshake。無しは tool request が 400 |
| optimized kernels + selector/dispatch 群 | Performance の本体。correctness への live fallback も兼ねる |
| correctness kernels(TU + detail/*.inc) | 正しさの最後の砦。allowlist 外 shape の fallback 本体。分離不可 |
| continuous batching / `ScheduledBatch` | Performance/Basic |
| concurrency / cancel | Advanced、OpenCode 実運用で使用 |
| prefix cache | Advanced(server 既定 ON) |
| strict tool calling / structured output / Structural Tag | Advanced。OpenCode 必須ではないが製品機能として有効 |
| reasoning(+ composition) | Advanced |
| Responses API | Advanced(OpenCode には不要だが Codex 向け。ISOLATE との二択) |
| DFlash2 speculative decode 本体(candidate selector + target verify) | Performance の主要成果(既存docs値 27.51 → 63.15 tok/s) |
| GDN recurrence lossy + compact commit(既定 ON 系) | Performance。ただし誤差契約・parity の docs 整合を追従させる |
| stochastic radix top-k | Performance。削除で top_k 設定時に 7.6〜585倍の遅化リスク(既存docs値) |
| quantizer 全体(quantize/verify/kld/imatrix/ppl) | 現在サポートしている artifact 生成に必須 |
| `phaseshift-bench` の E2E/micro benchmark 本体 | 繰り返し実施する benchmark に必須 |
| `tools/reference/*`、`tools/build_localai_runtime.sh`、`tools/quantization/{prepare_draft_vocab,build_phaseshift_imatrix_corpus,prepare_kld_corpus}.py` | tests / docs/user / docs/developer から参照あり。tools ルール適合 |
| vendor(LocalAI / XGrammar / nlohmann / transformers / OpenSSL) | 対応機能の依存本体 |

## SIMPLIFY

| 対象 | 理由 |
| --- | --- |
| `run_dflash2_shot` と `serve_dflash_open_session` の重複構築(`main.hip:569-625` vs `:1204-1293`) | 同じ sequence/context/decoder 構築が2本。片方の修正がもう片方に反映されない事象のリスク。single-shot を serve session の1リクエスト版として共通化可能 |
| `phaseshift_backend.py` の decision tree(backend.py:306-326)と `tool_constraint.py` の分担 | constraint 判定・composition・reasoning envelope の分岐が Python 側に分散。[constraint_audit.md](constraint_audit.md) のレイヤ表と合わせて単純化余地を評価 |
| dispatch ループ内の無条件インストルメント(`program_executor.hip:473-475`) | `kv_calib_dump_maybe` / `OpDumpTimer` / semantic timing を毎 dispatch 構築。既定は no-op だが overhead 実測が未了(NEEDS_INVESTIGATION) |
| 未文書化 env 約20件(`PSQ_DEBUG_HIDDEN` ほか) | 現行 current docs に記載なし。debug env を docs 化または整理 |
| README / `apps.cmake` / `tests.cmake` の stale コメント(D1・D4) | subcommand 名・required 件数が実装と不一致 |
| `docs/user/bench.md` の subcommand 欠落(D6) | `batch`/`gpu-memory` 等が docs 未記載 |

## ISOLATE

| 対象 | 理由 |
| --- | --- |
| **correctness/standalone/*.hip(5ファイル)** | production から呼ばれない(参照元は tests/bench のみ)。`phaseshift_qwen35_kernels` から test/bench 専用ターゲットへ分離できる |
| **PHASESHIFT_PA_PROBE probe 本体(221行)+ `apps.cmake:112-114` の無意味 define** | call site 0 件。ON でも未使用。R&D 機能として production TU から分離 |
| **`bench/paged_prune.cpp`(PoC 642行)** | bench 内でも R&D PoC。subcommand として production benchmark から分離 |
| **LM head proxy 全体(env 専用の近似経路)** | `lm_head_proxy.hip` / `constraint_candidates.hip` / 認証カーネル群。既定 OFF。production と同等の性能監査対象として分離 |
| **DFlash2 提案側 opt-in 経路(INT2 head / 固定語彙 / NgramTail)** | すべて既定 OFF・env/flag 専用。DFlash2 本体と物理的に分離 |
| **MTP スタック(production 未接続)** | `mtp_executor.hip`/`mtp_kv_state.cpp`/`spec_decoder.cpp`(shared helper は DFlash2 使用のため分離注意)。まず接続方針の意思決定が先行 |
| **HIP Graph コード(#ifdef 約323行)** | 既定ビルドで未コンパイル。メンバ・move 管理は無条件維持されているため、完全隔離を検討 |
| **gate 系 tests(25件)+ optional 外部 model 依存 tests(39件)** | R&D Gate 由来。required から独立したまま整理単位を作る |
| **`decode_perf_stats` / `kv_calib_dump`** | production dispatch に埋め込まれた env-gated 計測系。bench 専用ターゲットへ |
| **Responses API 関連(LocalAI patch 0002/0005/0006 + `test_server_responses*`)** | OpenCode には不要(Codex 向け)。必要なら独立 subsystem として分離 |

## REMOVE CANDIDATE

以下は削除候補。実施には各項目の「代替経路」「影響 tests」の確認と意思決定が必要。

### R1. `PHASESHIFT_PA_PROBE` probe dump 本体

- **削除対象ファイル**: `src/phaseshift/models/qwen35/runtime/paged_attention_dispatch.hip:13-21, 30-250`(221行)、`cmake/options.cmake` の該当 option、`cmake/apps.cmake:112-114` の bench への無意味 define
- **削除対象 symbol**: `probe_dump_inputs`(`:133`)、`probe_dump_output`(`:215`)
- **影響する tests**: 該当 test なし(`PHASESHIFT_PA_PROBE=ON` でも call site 0 件のため dump は生成されない)
- **代替経路**: なし(現状でも dump 生成経路が存在しない)
- **OpenCode への影響**: なし(既定ビルドと同一挙動)
- **根拠**: option 既定 OFF(`cmake/options.cmake:223`)、call 0 件。ただし `docs/rnd/attention/kv_page_pruning_poc.md` は dump 手順を記載しており、再利用判断が必要(NEEDS_INVESTIGATION)

### R2. `tools/rnd/` 全43ファイル

- **削除対象ファイル**: `tools/rnd/**`(`ffn_k_tile_oracle`、`ffn_k256_{sketch,candidate_selector,break_even}`、`analyze_psq4_kv.py`、`psq4_entropy_poc.py`、`psq4_perm_search.py`、`psq4_weight_sharing.py`、`psq4e_rans.cpp` ほか)
- **削除対象 symbol**: 各スクリプト本体
- **影響する tests**: なし(tests / CMake / `docs/user` / `docs/developer` から参照 0 件)
- **代替経路**: Git history / `docs/rnd/**` の記述
- **OpenCode への影響**: なし
- **根拠**: AGENTS.md「tools は R&D 記録からしか参照されない script を残さない」に抵触。`docs/rnd/ffn/*` 側のリンクは併せて整理必要

### R3. `tools/poc/clock_residency/`(4ファイル)

- **削除対象ファイル**: `tools/poc/clock_residency/{clock_probe.hip,poc_clock_state.sh,gate1.sh,gate1_fixed.sh}`
- **影響する tests**: なし
- **代替経路**: `docs/rnd/runtime/gpu_clock_residency_poc.md` の記述 + Git history
- **OpenCode への影響**: なし

### R4. `tools/quantization/` の単発解析 script(5件)

- **削除対象ファイル**: `tools/quantization/{psq4.py,analyze_su_channel_scaling.py,analyze_e4m0_codebook.py,analyze_scale_schemes.py,analyze_format_accuracy.py}`
- **影響する tests**: なし(`prepare_draft_vocab.py` / `build_phaseshift_imatrix_corpus.py` / `prepare_kld_corpus.py` は **残す**)
- **代替経路**: `docs/rnd/quantization/*` + Git history
- **OpenCode への影響**: なし

### R5. `tools/extract_kernel_resources.py` / `tools/gen_token_corpus.py`

- **削除対象ファイル**: 上記2件
- **影響する tests**: なし
- **代替経路**: `docs/rnd` の記述。ただし corpus 再生成 workflow の継続有無を要判断(NEEDS_INVESTIGATION)
- **OpenCode への影響**: なし

### R6. PoC 残骸 tests(2件)

- **削除対象ファイル**: `tests/unit/test_gdn_compact_commit_poc.hip`、`tests/kernels/optimized/test_gdn_compact_log_commit.hip`
- **削除対象 symbol**: `cmake/tests.cmake:186-187` の登録(`if(PHASESHIFT_BUILD_OPTIONAL_TESTS)` の**外**で常時 register されている点が問題)
- **影響する tests**: 上記2件のみ(いずれも optional ラベル)
- **代替経路**: GDN compact commit は production 既定経路のため、本家の `test_dflash2_*` 回帰で担保されているか要確認
- **OpenCode への影響**: なし(登録テスト数が2件減る)

### R7. MTP スタックの production 分離(または削除)

- **削除対象ファイル**: `src/phaseshift/models/qwen35/runtime/mtp_executor.hip`(696行)、`mtp_kv_state.cpp`(190行)、`include/phaseshift/models/qwen35/runtime/{mtp_executor,mtp_kv_state}.h`、`spec_decode.cpp` の MTP 専用部分(**`spec_greedy_accept` / `spec_gdn_snapshot` / `spec_gdn_restore` は DFlash2 production が使うため分離不可**)
- **削除対象 symbol**: `create_mtp_executor`、`mtp_executor_shutdown`、`create_mtp_kv_state`、`mtp_kv_dump_canonical`、`create_spec_decoder` / `spec_decoder_step`(bench/tests のみ)
- **影響する tests**: required 5件(`cmake/tests.cmake:93,129,131,133,135`)+ optional MTP gate 系多数 + `phaseshift-bench mtp` / `tg --mtp` subcommand
- **代替経路**: DFlash2(現行の唯一の production spec decode)
- **OpenCode への影響**: なし(production entrypoint から call 0 件。`docs/developer/mtp.md:15-31` が contract 化)
- **注意**: 「MTP を production に接続するか / 放棄するか」の意思決定が先行。required acceptance の件数が変わる

### R8. `phaseshift-bench paged-prune`(PoC)

- **削除対象ファイル**: `src/apps/bench/paged_prune.cpp`(642行)、`cmake/apps.cmake:97`、`src/apps/bench/main.cpp:14,46,88`
- **影響する tests**: `test_bench_help*` の subcommand 一覧
- **代替経路**: なし(KV page pruning 研究は runtime 未統合)
- **OpenCode への影響**: なし
- **依存**: R1(PA probe dump)と併せて判断

### R9. LM head proxy 全体(Fast / Shadow / constrained / 認証カーネル)

- **削除対象ファイル**: `src/phaseshift/models/qwen35/runtime/lm_head_proxy.hip`(894行)、`include/phaseshift/models/qwen35/runtime/lm_head_proxy.h`(172行)、`src/phaseshift/models/qwen35/kernels/optimized/constraint_candidates.hip`(73行)、`include/.../constraint_candidates.h`、`draft_head_int2.hip` の認証 kernel 部分(`:437-537, 928-1071`)、`tests/support/target_lm_proxy_harness.h`
- **削除対象 symbol**: `try_launch_lm_head_proxy_fusion`、`select_lm_head_constrained`、`lm_head_proxy_path`、`target_lm_head_proxy_*()`、`launch_dflash2_target_certified_argmax` ほか
- **影響する tests**: `test_lm_head_proxy_path`(required, cpu)、`test_target_lm_proxy_*(required, gpu1)` 5件、`test_constraint_candidates.hip`、`test_verify_lm_proxy_*`、`test_lm_head_residual_survey`(optional)
- **代替経路**: full PSQ8 + 全語彙 argmax / constrained full PSQ8(いずれも常に到達可能)
- **OpenCode への影響**: **なし**(既定 OFF。DFlash2 時は compute が `PHASESHIFT_TARGET_LM_HEAD_PROXY=0` を強制)
- **性能影響**: 既定への影響ゼロ。有効時の効果は既存docs値(verify proxy +2.4%、ExactCandidates 8〜9×)
- **付随 NEEDS_INVESTIGATION**: `executor.hip:1828` の初期化条件(`target_lm_head_proxy_mode() != 0`、既定1)により、非 spec の PSQ8 モデルで使用前にバッファ割当が発生し得る点。削除するとこの問題も同時に解消

### R10. NgramTail

- **削除対象ファイル**: `src/phaseshift/models/qwen35/runtime/ngram_tail.cpp`(110行)、`include/.../ngram_tail.h`、`main.hip:1803-1814` の CLI、`dflash2_spec_decoder.cpp:805-840` 統合、`spec_decoder.cpp:191-219` 統合
- **影響する tests**: `test_qwen35_ngram_tail_gate1`、`test_dflash2_ngram_tail_gate2`(いずれも optional)
- **代替経路**: DFlash2 提案のみ
- **OpenCode への影響**: なし(既定 0=無効、server/cli は flag を転送しない)
- **根拠**: 既定 ON は NO-GO 決定済み(`docs/rnd/spec_decode/ngram_tail_gate3.md:183-188`、既存docs値で平均 −1.10%)

### R11. `DecodeBackend::GpuMcu` stub

- **削除対象ファイル**: `src/phaseshift/models/qwen35/runtime/decode_backend.cpp:19-39` の `gpu-mcu` 受理部分、`docs/user/compute.md:42-43` の記述
- **削除対象 symbol**: `gpu-mcu` 識別子
- **影響する tests**: `test_decode_backend_contract`、`test_compute_decode_backend_gpu_mcu`(contract 化済み)
- **代替経路**: `host` のみ
- **OpenCode への影響**: なし
- **根拠**: AGENTS.md「production 到達不能な future placeholder を作らない」

### R12. `docs/now_mtp.md` / `docs/now_ngram.md`

- **削除対象ファイル**: `docs/now_mtp.md`、`docs/now_ngram.md`
- **代替経路**: `docs/rnd/mtp/mtp.md`(正本と明記済み)
- **OpenCode への影響**: なし
- **根拠**: AGENTS.md docs lifecycle(user/developer/perf/rnd/references のみ)

### R13. 生成物・未使用

- **削除対象ファイル**: `src/apps/**/__pycache__/*.pyc`(9件)、`tools/reference/__pycache__/*.pyc`、空ディレクトリ `models2/`
- **影響する tests**: なし
- **OpenCode への影響**: なし

### R14. dead code: `message_codec.resolve_tool_choice`

- **削除対象ファイル**: `src/apps/server/backend/message_codec.py:183-208`
- **削除対象 symbol**: `resolve_tool_choice`(message_codec 版。実際は `tool_constraint.resolve_tool_choice` が使用)
- **影響する tests**: なし(call site 0 件を確認)
- **代替経路**: `tool_constraint.resolve_tool_choice`
- **OpenCode への影響**: なし

### R15. LM Head constraint 最適化(constraint 側からの視点)

- **削除対象ファイル**: `src/phaseshift/models/qwen35/kernels/optimized/constraint_candidates.hip`(73行)、`include/.../constraint_candidates.h`(20行)、`lm_head_proxy.hip` の `select_constrained_exact`(`:482-501`)/ `select_constrained_masked`(`:580-597`)、閾値 env(`lm_head_proxy.hip:743-756`)
- **削除対象 symbol**: `launch_constraint_mask_to_candidates`、`select_lm_head_constrained`、`target_lm_head_proxy_constraint_threshold`
- **影響する tests**: `test_constraint_candidates`(gpu1/**required**)、`test_lm_head_constraint_perf`(optional)
- **代替経路**: constrained full PSQ8 + sampling filter(R9 の LM head proxy 削除と**セット**)。正しさはレイヤ 8〜13 が担保
- **OpenCode への影響**: なし(既定 OFF、DFlash2 時は compute が env=0 を強制)
- **注意**: R9 と同様、素朴な分岐削除は fail-open になるため `not_applicable` への置換が必須

### R16. dead code / 単機能の constraint 断片

- **削除対象**: `message_codec.resolve_tool_choice`(`message_codec.py:183-208`、call site 0 件 = R14 と同一)、DFlash2 proposal mask(`dflash2_spec_decoder.cpp:732-756`、env 既定 true)は「ISOLATE して acceptance 未計測を片付ける」か「削除」かの判断(verify 側 mask が正しさを担保)
- **影響する tests**: `test_dflash2_constraint_mask`(required)の該当 assertion
- **OpenCode への影響**: なし
- **詳細**: [constraint_audit.md](constraint_audit.md) §1 レイヤ16、§3-B

---

## 判断が先に必要な項目(削除の前に意思決定)

| # | 論点 | 影響 |
| --- | --- | --- |
| 1 | MTP を production に接続するか、放棄するか | R7、required tests 5件、bench subcommand |
| 2 | Responses API を維持するか(Codex 向け) | ISOLATE か KEEP か、LocalAI patch 0002/0005/0006 |
| 3 | HIP Graph を本命候補として維持するか | `PHASESHIFT_HIP_GRAPH` の存続、`test_constraint_graph` |
| 4 | KV page pruning 研究の終了宣言 | R1 + R8 |
| 5 | `tools/gen_token_corpus.py` の継続利用 | R5 |
| 6 | Python xgrammar を「constraint 使用時のみ」に条件化する改修の要否 | [opencode_minimum.md](opencode_minimum.md) §2 |

## NEEDS_INVESTIGATION(R&D 残骸判定の前に確認が必要)

1. `PHASESHIFT_PA_PROBE=ON` でも dump が生成されない経緯(呼び出しが revert で落ちたか)。
2. GDN compact commit の現行 parity(`docs/now_ngram.md` の revert 記録と `docs/developer/dflash2.md` の一致記載の矛盾)。

---

## 実行状況(2026-10-03, branch `refactor/src-trash-isolation`)

棚卸しの後、以下を srcTrash へ隔離した。詳細・理由・restore 手順は
[srcTrash/README.md](../../srcTrash/README.md) を参照。

| Phase | commit 内容 | 対象 | 結果 |
| --- | --- | --- | --- |
| 1 | archive obsolete rnd tools | `tools/rnd/` ほか 61ファイル | build のみ(PASS) |
| 2 | archive paged attention pruning poc | `paged_prune.cpp` + PA probe 221行 | 134/134 PASS |
| 3 | remove ngram tail from active runtime | NgramTail 一式 + CLI 2本 | 133/133 PASS |
| 4 | remove gpu mcu placeholder | `DecodeBackend::GpuMcu` + dead helper | 132/132 PASS |
| 5 | archive unused mtp runtime | MTP runtime / bench / tests / docs | 128/128 PASS |
| 6 | split exact constrained lm head path | proxy 隔離 + ExactCandidates 分離 | 122/122 PASS |
| 7 | archive dflash2 int2 and fixed vocab paths | INT2 / 固定語彙 + radix/psq8 分離 | 117/117 PASS |

### 承認済みの逸脱

1. **MTP lowering は KEEP**:`lower_qwen35_mtp_to_primitives` /
   `validate_mtp_geometry` は loader 幾何の contract として残置。
   完全削除は MTP tensor skip-load と同梱の別 ticket。
2. **MTP 層ロードを opt-in 化**:`Qwen35LoadOptions::load_mtp_layers`
   (既定 OFF)。production は MTP tensor を読まなくなった。
3. `tests/support/dflash2_int2_test_common.h` は ExactCandidates の required test
   が使うため KEEP(R7/R8 相当の test support)。
4. `test_qwen35_mtp_lowering` は lowering KEEP に合わせて残置。

### Phase 8 検証(実機)

- required acceptance **117/117 PASS**(0 skip / 0 fail)
- `phaseshift-compute`(DFlash2)正しさ契約: `GENERATED_IDS` sha1 `47aebe55d048`、
  `DFLASH2_ROUNDS=84`、`DFLASH2_ACCEPTED_DRAFTS=171`、`DFLASH2_GDN_MODE=compact` が一致
- `phaseshift-bench tg` `GREEDY_TOKEN_SUM=2446188` が一致
  (既存docs値との比較であり性能の再計測は行っていない)
- `tests/server/test_compute_constraints.py` 18/18 PASS
  (`invalid-grammar-no-fallback` を含む = fail-open なし)
- `ctest -L e2e` は optional tests 未 build(`PHASESHIFT_BUILD_OPTIONAL_TESTS=OFF`)かつ
  fixture `models/Qwen3.5-4B` 不在のため**未実施**

### 残存する削除候補(今回の隔離対象外)

- ISOLATE 判定のまま: HIP Graph(`PHASESHIFT_HIP_GRAPH`)、Responses API、
  `decode_perf_stats` / `kv_calib_dump` の env 計測系
- 判断待ち: MTP の完全削除(lowering / weight loader を含む)、Python xgrammar の
  条件化、`tools/bench_server_reasoning_constraints.py` の存続
