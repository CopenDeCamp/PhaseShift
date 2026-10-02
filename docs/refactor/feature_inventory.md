# PhaseShift 全機能インベントリ

PhaseShift リポジトリの全機能棚卸し。すべてコード側(call site / CMake / default / runtime 分岐 / tests)で確認し、docs 記述のみの機能は「現行機能」と判定していない。

- 調査方式: read-only。ソース・CMake・tests は変更していない。
- 性能値は既存 `docs/perf/` `docs/rnd/` の引用(既存docs値)。新規 benchmark は未実施。
- 関連資料: [opencode_minimum.md](opencode_minimum.md) / [constraint_audit.md](constraint_audit.md) / [approx_audit.md](approx_audit.md) / [refactor_candidates.md](refactor_candidates.md)

## 規模感(実測)

| 項目 | 値 |
| --- | --- |
| `src/` (C++/HIP) | 62,366 行 |
| `src/apps` Python | 5,916 行 |
| `include/` | 11,847 行 |
| `tests/` | 62,338 行(src とほぼ同規模) |
| `docs/rnd/` | 61 ファイル |
| CMake test 登録数 | 既定 build 137(required 134 / optional 2 / perf 1)、optional ON で 179 |

---

## 1. Basic — OpenCode backend を成立させるために必須

| Feature | Category | Public Surface | Main Entry Point | Implementation Files | Default Enabled | OpenCode Required | Fallback | Dependencies | Tests | Configuration | Complexity | Removal Impact | Recommended Action | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model load(safetensors / quantized) | Basic | `--model-dir` | `phaseshift-compute` / server `--model-dir` | `io/safetensors_reader.cpp`、`weights/weight_loader.cpp`、`models/qwen35/weights/model_weights.cpp` | Yes | Yes | なし | `phaseshift_io` `phaseshift_weights` | required 多数 | `--verify-weights` | 大 | 起動不能 | KEEP | High |
| tokenizer / chat template(HF) | Basic | server/backend | `codec._render` | `src/apps/common/phaseshift_chat/codec.py` | Yes | Yes | なし | `transformers==5.14.1` | `test_server_chat` `test_reasoning_*` | `use_tokenizer_template: true` | 中 | prompt 生成不能 | KEEP | High |
| prefill / decode 実行 | Basic | `--input-ids-file` / JSONL `generate` | `run_single_shot` / `run_serve_stdio`(`main.hip:350,1572`) | `runtime/executor.hip`、`program_executor.hip`、`continuous_batcher.cpp` | Yes | Yes | correctness kernel fallback | `phaseshift_qwen35_runtime` | required GPU 系全般 | `--max-new-tokens` 等 | 大 | 推論不能 | KEEP | High |
| sampling(greedy / temp / top-k / top-p / seed) | Basic | `--temperature` `--top-k` `--top-p` `--seed` | `SAMPLING` dispatch | `kernels/optimized/sampling.hip`、`stochastic_sampling.hip`、`sampling_params.cpp` | Yes(temp=0 で greedy) | Yes | correctness SAMPLING | `phaseshift_qwen35_kernels_optimized` | `test_compute_service`、bench tg | CLI flag | 中 | 生成不能 | KEEP | High |
| paged KV state(BF16/FP8/PSQ4/PSQ8) | Basic | `--kv-cache-dtype` `--page-tokens` | `Qwen35ComputeRuntime` | `state/paged_kv_pool.cpp`、`paged_sequence_state.cpp`、`sequence_slot_pool.cpp` | Yes(bf16) | Yes | なし | `phaseshift_qwen35_state` | required GPU 系 | `--kv-cache-capacity-tokens` | 大 | 推論不能 | KEEP | High |
| OpenAI Chat Completions | Basic | `POST /v1/chat/completions` | LocalAI + `phaseshift_backend.py` | `vendor/localai/patches/*`、`backend/phaseshift_backend.py` | Yes | **Yes** | なし | LocalAI patched binary | `test_server_chat` `test_server_public_surface` | `--host/--port` | 大 | OpenCode 接続不能 | KEEP | High |
| streaming(SSE) | Basic | `stream:true` | `PredictStream` | `phaseshift_backend.py:456-499` | Yes(client opt-in、無効化 flag なし) | Yes | 非 stream | gRPC server-streaming | `test_server_stream*` | なし | 中 | OpenCode の通常動作が壊れる | KEEP | High |
| tool 定義の prompt 投入(A) | Basic | `tools` | `codec._render(tools=)` | `codec.py:142-151` | Yes | Yes | なし | HF template | `test_server_tools` | なし | 小 | tool 使用不能 | KEEP | High |
| tool call の parser(C)/ OpenAI 変換(D) | Basic | `tool_calls` / `finish_reason=tool_calls` | `parse_assistant_message` | `codec.py:359-575`、`phaseshift_backend.py:434-446` | Yes | Yes | streaming 一括 fallback(`phaseshift_backend.py:624-633`) | Transformers parser | `test_server_tools` `test_server_stream_tools` | なし | 中 | tool loop 崩壊 | KEEP | High |
| tool result round-trip(E) | Basic | `role:"tool"` | `proto_messages_to_hf` | `message_codec.py:84-124` | Yes | Yes | なし | なし | `test_server_tools:96-119`、`test_agent_opencode` | なし | 小 | agent の2往復目以降が壊れる | KEEP | High |
| server → compute transport | Basic | `--serve-stdio` | `ComputeClient` | `compute_client.py`、`main.hip:1537-1569`(ping/generate/cancel/shutdown) | Yes | Yes | なし | JSONL stdio | `test_compute_service` | `PHASESHIFT_COMPUTE` | 中 | serve 不能 | KEEP | High |
| `tool_choice` 解決(auto/none/required/named) | Basic→Advanced | request field | `tool_constraint.resolve_tool_choice` | `tool_constraint.py:39-88` | Yes(auto 既定) | auto は Basic | fail-closed 400 | LocalAI patch 0003 | `test_server_strict_tools:146-234` | なし | 小 | required/named のみ喪失 | KEEP | High |
| cancel(断線) | Basic | `finish_reason=cancelled` | `should_cancel` | `phaseshift_backend.py:393,519,600`、`compute_client.py:223-227` | Yes | Yes | なし | なし | `test_compute_cancel` `test_server_stream_cancel` | なし | 小 | 割り込み不能 | KEEP | High |
| context / max_tokens fail-closed | Basic | HTTP 400 | `_prepare` | `phaseshift_backend.py:357-368` | Yes | Yes(OpenCode の `limit` 設定前提) | なし | なし | `test_server_context_boundary` | `--max-seq-len` `--default-max-output-tokens` | 小 | overflow 時の挙動不明 | KEEP | High |

## 2. Advanced — 動作はするが製品機能として意味があるもの

| Feature | Category | Public Surface | Main Entry Point | Implementation Files | Default Enabled | OpenCode Required | Fallback | Dependencies | Tests | Configuration | Complexity | Removal Impact | Recommended Action | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict tool calling(`strict:true`) | Advanced | request field | `validate_strict_parameters` | `tool_constraint.py:108-216, 295-309` | OFF(明示 true のみ) | No | non-strict(any_text) | constraint stack | `test_server_strict_tools*` | なし | 中 | strict のみ喪失、best-effort は存続 | KEEP | High |
| Structural Tag constraint(H) | Advanced | compute `structural_tag` | `build_structural_tag` | `tool_constraint.py:312-352`、`token_constraint.cpp` | 条件付き(required/named・parallel=false・strict) | No | `structural_tag=None`(unconstrained) | XGrammar | `test_compute_structural_tag` `test_composite_structural_constraint` | なし | 中 | strict/required/named のみ | KEEP | High |
| structured output(response_format / text.format) | Advanced | Chat `response_format` / Responses `text.format` | LocalAI patch 0001/0002 → `PredictOptions.Grammar` | `main.hip:1429-1432`、`token_constraint.cpp` | 指定時のみ | No | **なし(fail-closed、400)** | XGrammar | `test_server_structured*` `test_localai_fail_closed` | 起動時 probe | 大 | JSON 出力制約が消える | KEEP | High |
| structured + tools composition | Advanced | 同時指定 | `build_composite_structural_tag` | `tool_constraint.py:322-352`、patch 0004 | 指定時のみ | No | 両方指定で error | なし | `test_server_structured_tools*` `test_tool_composition_builder` | なし | 中 | 同時指定のみ喪失 | KEEP(SIMPLIFY 候補) | High |
| reasoning / thinking | Advanced | `reasoning_effort` / `message.reasoning` | `resolve_reasoning_policy` | `codec.py:187-248`、`phaseshift_backend.py:307-309` | **OFF**(opt-in) | No | なし | `enable_thinking` template | `test_server_reasoning*` | request field | 中 | thinking 出力のみ喪失 | KEEP | High |
| reasoning + constraint 合成 | Advanced | envelope | `build_reasoning_structural_tag` | `tool_constraint.py:411-422` | reasoning 使用時のみ | No | reasoning off | Structural Tag | `test_compute_reasoning_constraint` `test_server_reasoning_composition` | なし | 中 | reasoning+tools 同時使用のみ | KEEP(ISOLATE 候補) | High |
| concurrency(`--max-concurrent-requests`) | Advanced | server flag | LocalAI thread pool → `ComputeClient` semaphore | `phaseshift_server.py:252,331,335`、`phaseshift_backend.py:984-989` | Yes(既定4) | Yes | serial(1) | なし | `test_compute_concurrency` `test_server_concurrency` | flag | 中 | 並列性喪失(動作はする) | KEEP | High |
| cancellation | Advanced | (上記 Basic 扱い) | — | — | Yes | — | — | — | — | — | — | — | (Basic 参照) | High |
| prefix cache | Advanced | `--prefix-cache-capacity-tokens` | `prefix_cache.cpp` | `runtime/prefix_cache.cpp`、`codec.py:160-184`(boundary 計算) | server 既定 ON(16384/8)、compute 既定 OFF | Yes(multi-turn 速度) | 無効(=毎回 prefill) | なし | `test_compute_prefix_cache` `test_server_prefix_cache*` `test_server_agent_prefix_cache` | server flag | 中 | multi-turn が毎回 full prefill | KEEP | High |
| DFlash2 speculative decode | Advanced→Performance | `--dflash2-model-dir` | `run_dflash2_shot` / `serve_dflash_open_session` | `dflash2/executor.hip`、`dflash2_spec_decoder.cpp`、`kernels/dflash2/*` | OFF(flag 指定時のみ) | No | 非 spec decode | draft model | `test_compute_dflash2` `test_server_dflash2` | flag `--dflash2-drafts` | 大 | decode 約2倍速(既存docs値 27.51→63.15 tok/s)を失う | KEEP | High |
| `/v1/responses` Responses API | Advanced | `POST /v1/responses` | LocalAI openresponses + patch 0002/0005/0006 | `vendor/localai/patches/0002,0005,0006` | Yes(到達可能) | **No**(Codex CLI 向け) | Chat Completions | patched LocalAI | `test_server_responses*` `test_agent_codex` | なし | 大 | Codex 互換のみ喪失、OpenCode 無影響 | KEEP または ISOLATE | Medium |
| LocalAI fail-closed capability probe | Advanced | 起動時 abort | `probe_structured_fail_closed` | `phaseshift_server.py:145-228` | Yes(unconditional) | No | 起動拒否なし | patched LocalAI | `test_localai_fail_closed` `test_server_localai_capability` | `--startup-timeout` | 小 | 不正 structured が静かに unconstrained へ落ちる | KEEP | High |
| decode backend `gpu-mcu`(stub) | Legacy候補 | `--decode-backend` | `decode_backend.cpp` | `decode_backend.cpp:19-39`(`validate` 常時 unsupported) | host のみ有効 | No | host | なし | `test_decode_backend_contract` `test_compute_decode_backend_*` | flag | 小 | 未来の placeholder を失うだけ | REMOVE CANDIDATE(要判断) | High |

## 3. Performance — 外部仕様を変えずに速くするもの

| Feature | Category | Public Surface | Main Entry Point | Implementation Files | Default Enabled | OpenCode Required | Fallback | Dependencies | Tests | Configuration | Complexity | Removal Impact | Recommended Action | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| optimized kernels + selector/dispatch | Performance | 内部 | `execute_program_range` | `kernels/optimized/*`(36ファイル)、`runtime/*_selector.cpp`(12)+ `*_dispatch.hip`(13) | Yes(Auto) | Yes | **correctness kernel が live fallback** | `phaseshift_qwen35_kernels_optimized` | `tests/kernels/*` | `PHASESHIFT_QWEN35_KERNEL_MODE` | 大 | 全性能喪失 | KEEP | High |
| correctness kernels(参照実装) | Performance兼正しさ | 内部 / env | `launch_model_dispatch_correctness` | `kernels/correctness/*`(TU+inc 約1,420行) | Yes(fallback) | Yes(間接) | なし(これが fallback 本体) | `phaseshift_qwen35_kernels` | required 系、`test_qwen35_kernel_mode` | `PHASESHIFT_QWEN35_KERNEL_MODE=correctness` | 大 | allowlist 外 shape が動かなくなる | KEEP(Basic 兼 Performance) | High |
| correct/optimized の `standalone/*` 参照 | Tooling | 内部 | tests / bench のみ | `kernels/correctness/standalone/*.hip`(5) | Yes(コンパイルされる) | No | optimized | なし | `cmake/tests.cmake:150-173`、`bench/gemm.hip` | なし | 小 | 単体 GEMM/RMSNorm 参照 test が消える | ISOLATE(test/bench 専用 target へ) | High |
| continuous batching(`ScheduledBatch`) | Performance | 内部 | `ContinuousBatcher::step` | `continuous_batcher.cpp`、`scheduled_batch.cpp`、`token_budget_scheduler.cpp` | Yes | Yes | serial | なし | `test_compute_concurrency` | `--max-concurrent-requests` | 大 | throughput 崩壊 | KEEP | High |
| prefix cache(see Advanced) | Performance | — | — | — | — | — | — | — | — | — | — | — | — | High |
| DFlash2(see Advanced) | Performance | — | — | — | — | — | — | — | — | — | — | — | — | High |
| stochastic radix top-k | Performance | 内部 | `sampling_dispatch.hip:70-121` | `kernels/dflash2/radix_topn.hip` + `sampling_selector.cpp:23-34` | 条件成立時 ON | Yes(temp>0 + top_k) | `StochasticSingleBlock` / `Correctness` | なし | `test_dflash2_radix_topn*`、sampling 系 | なし | 中 | top_k 指定時に 7.6〜585倍の遅化リスク(既存docs値) | KEEP | High |
| GDN recurrence lossy | Performance(近似) | 内部 | `gdn_recurrence_dispatch.hip:235-261` | `kernels/optimized/gdn/recurrence.hip:1190-1199` | **ON(既定)** | Yes | exact 経路(env) | なし | `test_gdn_*` | `PHASESHIFT_GDN_RECURRENCE_EXACT/LOSSY` | 大 | PP −3.7%(既存docs値) | KEEP だが誤差契約を docs 化 | High |
| GDN compact commit | Performance | 内部 | `dflash2_spec_decoder.cpp:238-243` | commit kernel `recurrence.hip:812-920`、`gdn_spec_history.hip` | **ON(既定)** | Yes(DFlash2 時) | history path / rerun | DFlash2 | `test_gdn_compact_*`、`test_dflash2_*` | `PHASESHIFT_DFLASH2_GDN_COMPACT_COMMIT` | 中 | メモリ +716.8 MiB(既存docs値) | KEEP(-parity 要再確認) | Medium |
| row bucket / ProgramSet / arena | Performance | 内部 | `program.cpp` `arena.hip` | `runtime/program/*`、`core/memory/arena.hip` | Yes | Yes | なし | なし | required | `PHASESHIFT_ARENA_VMM` | 大 | launch/alloc 劣化 | KEEP | High |
| keep-alive thread | Performance | 内部 | `executor.hip:1128-1135` | 同上 | **ON(既定)** | Yes | `PHASESHIFT_KEEPALIVE=0` | なし | — | env | 小 | 起動オーバーヘド挙動が変わる | KEEP(根拠 docs 要確認) | Medium |
| constraint LM head 最適化(ExactCandidates / MaskedCoarse) | Performance | env のみ | `select_lm_head_constrained` | `lm_head_proxy.h:139-170`、`constraint_candidates.hip` | **OFF(env のみ)** | No | constrained full PSQ8 | PSQ8 preshuffled | `test_constraint_candidates.hip` | `PHASESHIFT_TARGET_LM_HEAD_PROXY=1` | 中 | 既定への影響ゼロ(有効時 8〜9×、既存docs値) | ISOLATE → REMOVE CANDIDATE 候補 | High |
| candidate pruning(非制約 LM head proxy Fast) | Performance/Experimental | env のみ | `try_launch_lm_head_proxy_fusion` | `lm_head_proxy.hip` 894行ほか | **OFF(env のみ)** | No | full PSQ8 + 全語彙 argmax | 同上 | `test_lm_head_proxy_path` ほか | `PHASESHIFT_TARGET_LM_HEAD_PROXY` | 大 | 既定への影響ゼロ | REMOVE CANDIDATE 候補 | High |
| DFlash2 proposal-side 高速化(INT2 / 固定語彙) | Performance/Experimental | env のみ | `dflash2/executor.hip` | `draft_head_int2.*`、`coarse_topn.hip`、`draft_vocab_profile.cpp` | **OFF(env のみ)** | No | drafter full PSQ8 | なし | `test_dflash2_gate11*`、`test_dflash2_draft_vocab_profile` | `PHASESHIFT_DFLASH2_INT2_HEAD` 等 | 大 | 既定への影響ゼロ(+2.4%、既存docs値) | ISOLATE(env 専用) | High |

## 4. Experimental — 近似・PoC・build option・env 専用

| Feature | Category | Public Surface | Main Entry Point | Implementation Files | Default Enabled | OpenCode Required | Fallback | Dependencies | Tests | Configuration | Complexity | Removal Impact | Recommended Action | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HIP Graph capture/replay | Experimental | CMake option | `PHASESHIFT_HIP_GRAPH` | `executor.hip`(#ifdef 約293行)、`program_executor.hip`(#ifdef 約30行) | **OFF** | No | 通常 enqueue | 再 configure | `test_constraint_graph.py`(OFF なら skip) | `-DPHASESHIFT_HIP_GRAPH=ON` | 大 | 既定ビルドは無変化 | ISOLATE(監査対象) | High |
| LM head proxy Shadow(mode=2) | Experimental | env のみ | `lm_head_proxy.hip:392-395` | 同上 | **OFF(env のみ)** | No | full path | なし | proxy shadow 系 | `PHASESHIFT_TARGET_LM_HEAD_PROXY=2` | 小 | なし | REMOVE CANDIDATE | High |
| 認証(upper-bound)カーネル | Experimental | **test 専用** | `draft_head_int2.hip:437-537, 1044-1071` | 同上 + harness | test のみ | No | 不採用 | なし | `test_target_lm_proxy_*(required)` | `PHASESHIFT_TARGET_LM_CERT_RATE` | 中 | 本番なし(tests と一体) | REMOVE CANDIDATE(tests と併走) | High |
| PA probe(`PHASESHIFT_PA_PROBE`) | Experimental | CMake option | **call site 0 件** | `paged_attention_dispatch.hip:13-21, 30-250`(221行) | **OFF** | No | なし | `paged_prune` dump 読み取り | `test_constraint_graph` なし(該当 test なし) | `-DPHASESHIFT_PA_PROBE=ON` | 中 | **ON でも未使用** | REMOVE CANDIDATE | High |
| `phaseshift-bench paged-prune`(PoC) | Experimental | bench subcommand | `bench/main.cpp:46,88` | `bench/paged_prune.cpp`(642行) | 有効(登録) | No | なし | PA probe dump | なし | subcommand | 中 | PoC のみ | REMOVE CANDIDATE | High |
| DFlash2 proposal constraint 近似 | Experimental(既定ON) | 内部 | `dflash2_spec_decoder.cpp:732-753` | 同上 | **ON(env 既定 true)** | Yes | verify の mask が担保 | constraint | `test_dflash2_constraint_mask.cpp` | `PHASESHIFT_DFLASH2_CONSTRAINT_PROPOSAL` | 小 | 削除可(verify が担保) | ISOLATE(accept 率未計測) | Medium |
| NgramTail | Experimental | CLI flag | `ngram_tail.cpp` | `runtime/ngram_tail.cpp`(110行) | **OFF**(flag 未指定=0) | No | なし | DFlash2 | `test_qwen35_ngram_tail_gate1` `test_dflash2_ngram_tail_gate2`(optional) | `--dflash2-ngram-tail` | 小 | 既定ゼロ(平均 −1.10%、既存docs値) | REMOVE CANDIDATE(NO-GO 決定済み) | High |
| MTP draft policy(dynamic/discard) | Experimental | bench flag | `spec_decode.h:129-136` | `spec_decode.cpp:368-383` | **OFF** | No | 固定 K | MTP 全体 | `bench/mtp.hip` | bench flag | 小 | なし | (MTP と一体で判断) | High |
| `DecodeBackend::GpuMcu` stub | Experimental | `--decode-backend` | `decode_backend.cpp:34-39` | 同上 | 常時 reject | No | host | なし | `test_decode_backend_contract` | flag | 小 | placeholder 消失 | REMOVE CANDIDATE | High |
| KV calib dump | Experimental | env のみ | `program_executor.hip:473` から毎呼出し | `kv_calib_dump.cpp`(env 未設定なら即 return) | **OFF** | No | なし | なし | なし(docs 未記載) | `PHASESHIFT_KV_CALIB_DUMP` | 小 | なし | REMOVE CANDIDATE(R&D 由来) | High |
| decode perf stats | Experimental | env のみ | 各 dispatch の `record_*` | `decode_perf_stats.cpp`、`.h` | **OFF** | No | なし | dump は bench のみ | なし | `PHASESHIFT_QWEN35_PERF_STATS` | 小 | なし | ISOLATE(bench 専用へ) | High |

## 5. Tooling — serving runtime そのものではないもの

| Feature | Category | Public Surface | Main Entry Point | Implementation Files | Default Enabled | OpenCode Required | Fallback | Dependencies | Tests | Configuration | Complexity | Removal Impact | Recommended Action | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `phaseshift-compute` | Tooling兼Basic | entrypoint | `main.hip` | `src/apps/compute/*` | Yes | **Yes(serve-stdio の本体)** | なし | `phaseshift` aggregate | required | 多数の flag | 大 | — | KEEP | High |
| `phaseshift-cli` | Tooling | entrypoint | `phaseshift_cli.py` | `src/apps/cli/` | Yes | No | server | `codec.py` | e2e optional | なし | 中 | 対話 UI のみ | KEEP | High |
| `phaseshift-server` | Tooling兼Basic | entrypoint | `phaseshift_server.py` | `src/apps/server/*` | Yes | **Yes** | なし | LocalAI patched | `tests/server/*`(64) | flag 多数 | 大 | — | KEEP | High |
| `phaseshift-quantizer` | Tooling | entrypoint | `main.cpp`(quantize/verify/kld/imatrix/ppl) | `src/apps/quantizer/*`、`quantization/offline/*` | Yes | No | なし | `phaseshift_quantizer_core` | optional quantizer 系 | `--preset` 等 | 大 | 量子化 pipeline が消える | KEEP | High |
| `phaseshift-bench` | Tooling | entrypoint | `bench/main.cpp` | `src/apps/bench/*`(19 TU) | Yes(`PHASESHIFT_BUILD_BENCHMARKS` ON) | No | なし | `phaseshift` aggregate | `test_bench_help*` 15件 | flag 多数 | 大 | 計測不能 | KEEP(ただし PoC subcommand は削除候補) | High |
| quantizer(KLD / imatrix / PPL) | Tooling | quantizer subcommand | `commands/{kld,imatrix,ppl}.cpp` | `quantization/offline/*` | Yes | No | なし | OpenSSL | optional | flag | 大 | offline 品質管理喪失 | KEEP | High |
| `tools/quantization/prepare_draft_vocab.py` 等 | Tooling | script | tests / docs/user | `tools/quantization/{prepare_draft_vocab,build_phaseshift_imatrix_corpus,prepare_kld_corpus}.py` | — | No | なし | なし | `test_prepare_draft_vocab.py`(required) | なし | 小 | DFlash2 固定語彙 workflow 喪失 | KEEP | High |
| `tools/reference/*`(gate fixture 再生成) | Tooling | script | CMake comment / docs/developer | `tools/reference/*.py`(7) | — | No | なし | なし | fixture 依存 tests | なし | 小 | fixture 再生成不能 | KEEP | High |
| `tools/build_localai_runtime.sh` | Tooling | script | docs/user/build.md | 同上 | — | No | なし | LocalAI source | なし | なし | 中 | patched runtime build 手順喪失 | KEEP | High |
| **`tools/rnd/` 全43ファイル** | Legacy候補 | script | **docs/rnd のみ参照** | `tools/rnd/**` | — | No | Git history | なし | **tests/CMake/docs(user,developer) から 0 参照** | なし | 中 | R&D 再実行のみ | REMOVE CANDIDATE(AGENTS tools ルール抵触) | High |
| **`tools/poc/clock_residency/`** | Legacy候補 | script | docs/rnd のみ | 同上(4) | — | No | Git history | なし | 0 参照 | なし | 小 | PoC 再実行のみ | REMOVE CANDIDATE | High |
| `tools/quantization/{psq4,analyze_*}.py`(5) | Legacy候補 | script | docs/rnd のみ | 同上 | — | No | Git history | なし | 0 参照 | なし | 小 | 単発解析のみ | REMOVE CANDIDATE | High |
| `tools/{extract_kernel_resources,gen_token_corpus}.py` | Legacy候補 | script | docs/rnd のみ | 同上 | — | No | Git history | なし | 0 参照 | なし | 小 | 単発生成のみ | REMOVE CANDIDATE(継続有無を要判断) | Medium |
| `tools/bench_server_reasoning_constraints.py` | Tooling(要判断) | script | docs/developer/testing.md:91(「manifest へ入れない」) | 同上 | — | No | なし | なし | run_server_regression のコメントのみ | なし | 小 | reasoning benchmark のみ | NEEDS_INVESTIGATION | Medium |

## 6. Vendor — 外部依存

| Feature | Category | Public Surface | Main Entry Point | Implementation Files | Default Enabled | OpenCode Required | Fallback | Dependencies | Tests | Configuration | Complexity | Removal Impact | Recommended Action | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LocalAI v4.10.0 + patch 0001-0006 | Vendor | HTTP frontend | `phaseshift_server.py:322` | `vendor/localai/patches/*`、`backend.proto` | Yes | **Yes**(HTTP/gRPC frontend) | なし | `tools/build_localai_runtime.sh` | `test_localai_*`、`test_server_*` | `PHASESHIFT_LOCALAI_BINARY` | 大 | serving 全体が消える | KEEP | High |
| XGrammar v0.2.5(C++) | Vendor | grammar engine | `TokenConstraintCompiler` | `vendor/xgrammar/` | Yes | **現状は LoadModel の hard requirement** | unconstrained(要改修) | Python `xgrammar==0.2.5.post1` | `test_constraint_dependency` `test_xgrammar_cxx20` | `--constraint-tokenizer-info` | 中 | structured/strict 喪失 | KEEP(constraint_audit 参照) | High |
| nlohmann/json | Vendor | 内部 | quantized manifest 読み書き | `quantization/fpx/*`、`quantization/offline/kld.cpp` | Yes | Yes(quantized model load) | なし | `vendor/nlohmann` | required | なし | 小 | quantized safetensors 読込不能 | KEEP | High |
| `transformers`(Python) | Vendor | chat template / parser | `codec.py` | 外部 pip | Yes | **Yes** | なし | `transformers==5.14.1` | `test_reasoning_*` | pip 要件 | 中 | prompt/parser 不能 | KEEP | High |
| OpenSSL::Crypto | Vendor | 内部 | manifest CRC/FIPS 計算 | `phaseshift_fpx_format` | Yes | Yes | なし | `find_package(OpenSSL)` | required | なし | 小 | manifest 検証不能 | KEEP | High |

## 7. Legacy / Redundant 候補(詳細は refactor_candidates.md)

| Feature | Category | Public Surface | Main Entry Point | Implementation Files | Default Enabled | OpenCode Required | Fallback | Dependencies | Tests | Configuration | Complexity | Removal Impact | Recommended Action | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **MTP SpecDecoder スタック** | Legacy候補 | **production から到達不能** | `bench/mtp.hip` / tests のみ | `mtp_executor.hip`(696) `spec_decoder.cpp`(501) `mtp_kv_state.cpp`(190) + headers | Yes(コンパイル) | No | DFlash2 | `phaseshift_qwen35_runtime` | required 5件 + optional 多数 | `PHASESHIFT_MODEL_DIR_MTP` | 大 | production なし(required tests / bench 変化) | REMOVE CANDIDATE(意思決定先行) | High |
| `docs/now_mtp.md` / `docs/now_ngram.md` | Legacy候補 | docs 直下 | — | 2ファイル | — | No | `docs/rnd/mtp/mtp.md` | なし | なし | なし | 小 | docs lifecycle 違反 | REMOVE(統合) | High |
| `__pycache__/*.pyc` + `models2/` | Legacy候補 | — | — | pyc 10 + 空 dir | — | No | 再生成 | なし | なし | なし | 小 | なし | REMOVE | High |
| `message_codec.resolve_tool_choice` | Legacy候補 | — | **call site 0 件** | `message_codec.py:183-208` | — | No | `tool_constraint.resolve_tool_choice` | なし | なし | なし | 小 | なし | REMOVE | High |
| gate 系 tests(25件)+ PoC tests | Legacy候補 | ctest | `cmake/tests.cmake` | `tests/unit/test_*gate*` 等 | optional 40件は既定 OFF で未生成 | No | 回帰 test | 外部 model | — | `PHASESHIFT_MODEL_DIR_*` | 中 | required への影響は限定的 | ISOLATE / REMOVE CANDIDATE | Medium |

## 8. docs と code の不整合(棚卸しの副産物)

| # | 内容 | 判定 |
| --- | --- | --- |
| D1 | README / `apps.cmake` は quantizer を「quantize/verify/**convert/validate**/kld」と記載。実際は `quantize/verify/kld/imatrix/ppl`(`src/apps/quantizer/main.cpp:35-39`) | stale |
| D2 | README は KV dtype を「BF16 / FP8_E4M3」と記載。実際は psq4/psq8 も対応 | 過小記載 |
| D3 | `main.hip:74-75` の usage「--dflash2-model-dir requires --temperature 0」は実装に温度チェックなし、docs 側は stochastic 対応を主張 | 要確認 |
| D4 | `cmake/tests.cmake:4`「required: 14」は実際 134 件 | stale |
| D6 | `docs/user/bench.md` に `batch`/`l2-normalize`/`gdn-conv1d`/`paged-prune`/`gpu-memory` の記載なし | docs 過小 |
| C1 | `kv_calib_dump`(env)は production に link されているが docs 皆無 | 未文書化 |
| C2 | `decode_perf_stats`(env)は production dispatch に埋め込まれているが current docs なし | 未文書化 |
| C5 | 未文書化 debug/tunable env 約20件(`PSQ_DEBUG_HIDDEN` ほか) | 未文書化 |

## 9. NEEDS_INVESTIGATION

1. `test_dflash2_draft_vocab_profile` が required label だが `phaseshift-required-tests` の DEPENDS に不在(`cmake/tests.cmake:95` vs `:227-342`)。
2. 非 DFlash2 の PSQ8 モデル既定実行で `lm_head_proxy` バッファが割り当てられる量と実測影響(`executor.hip:1828` の条件)。
3. `PHASESHIFT_PA_PROBE=ON` でも dump が生成されない(call site 0 件)の経緯。
4. `tools/gen_token_corpus.py` / `tools/bench_server_reasoning_constraints.py` の継続利用有無。
5. `docs/rnd/quantization/kernel_optimization_history.md:481` の dangling 参照(`analyze_scale_factorization.py` が存在しない)。
6. MTP を production に接続する方針か、放棄する方針か(roadmap にも記載なし)。
7. `PHASESHIFT_KEEPALIVE` 既定 ON の根拠(docs/perf に記載なし)。
