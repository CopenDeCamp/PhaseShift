# Constraint subsystem 監査

> Status: historical R&D record（server 簡素化・constraint / LocalAI 削除 **前** の状態を記録している。現在の仕様ではない。現在の仕様は docs/developer/ を参照）


PhaseShift の Constraint を1機能として扱わず、レイヤごとに分解した資料。
すべて call site / CMake / default / runtime 分岐 / tests で確認済み。
設計の正本だった `docs/developer/structured_generation.md` は削除済み。Git history を参照。

## 0. 全体構成(証拠)

```text
OpenAI request
  → LocalAI v4.10.0 + patch 0001..0006   (structured fail-closed / tool policy / composition / reasoning)
  → gRPC PredictOptions{ Grammar, ToolChoice, Metadata }   vendor/localai/backend.proto:315,335,338
  → phaseshift_backend.py::_prepare  decision tree :306-326
      build_reasoning_structural_tag / build_composite_structural_tag / build_structural_tag / grammar
  → compute_client.py JSONL { grammar | structural_tag }   compute_client.py:207-210
  → main.hip:1429-1513 (排他検証 :1462) → TokenConstraintCompiler
  → TokenConstraintState (XGrammar GrammarMatcher)
  → ContinuousBatcher::step で CPU bitmask 生成 :370-410
  → executor.hip で H2D :914-921 → sampling kernel の constraint filter
  → (opt-in) lm_head_proxy constrained 経路 / (DFlash2) proposal mask + draft tree mask
```

---

## 1. レイヤ別監査表

| # | レイヤ | 何をするか | entry point | production call site | 既定・種別 | LOC | 削除影響 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1a | tool policy parser | `tool_choice` / `parallel_tool_calls` / `tools` を fail-closed に parse | `resolve_tool_choice` `tool_constraint.py:39`、`parse_parallel_tool_calls:91`、`collect_tools:108` | `phaseshift_backend.py:276,288` | production・常時 | `tool_constraint.py` 422 | tool_choice 解析不能 |
| 1b | strict schema 検証 | OpenAI strict contract(`additionalProperties:false` / required 全列挙) | `validate_strict_parameters tool_constraint.py:134` | `collect_tools:128-130`(`strict is True` のみ) | **default OFF** | 同 134-216 | strict のみ喪失 |
| 1c | chat codec | prompt render / tool call parse / reasoning split | `codec.py:120,359,373,485,528` | `phaseshift_backend.py:185,328,415,581` | production・常時 | `codec.py` 575 | 全機能が壊れる(Basic) |
| 1d | message codec | LocalAI proto ↔ HF message、tools JSON parse/正規化 | `proto_messages_to_hf:84`、`parse_tools:127` | `phaseshift_backend.py:274-275` | production・常時 | `message_codec.py` 208 | tool round-trip 崩壊(Basic) |
| 1e | `message_codec.resolve_tool_choice` | (dead code) | `message_codec.py:183-208` | **call site 0 件** | dead | 26 | なし |
| 2 | tool constraint builder | Structural Tag 文書を決定論的 JSON で生成 | `build_structural_tag:312`、`build_composite_structural_tag:322`、`build_reasoning_structural_tag:411` | `phaseshift_backend.py:309,316,320,324` | conditional(既定 request では非活性 `:218-226`) | ≒205 | required/named/strict/reasoning 合成が消える |
| 3 | JSON Schema / response grammar transport | response_format→GBNF を LocalAI 側で変換し `PredictOptions.Grammar` で配送、失敗は 400 | patch `0001:17-101`、`0002:114-160` | `phaseshift_backend.py:292,313,326` → `compute_client.py:207-208` → `main.hip:1429-1445` | production・常時(起動 probe が必須化) | 0001 102 / 0002 411 | structured 消滅 + 起動 probe fail |
| 4 | Structural Tag + reasoning 合成 | `or(grammar, tool)` 合成、reasoning envelope を1つの Structural Tag に畳み込む | `build_composite_structural_tag:322`、`build_reasoning_envelope:367`、`build_reasoning_structural_tag:411` | `phaseshift_backend.py:309,316,320` | conditional(reasoning / composition marker 時のみ) | 101 | reasoning+tools・structured+tools の同時指定が 400 化 |
| 5a | LocalAI patch 0001 / 0002 | structured fail-closed(400)、Responses `text.format` 統一 | `vendor/localai/patches/0001,0002` | 起動 probe `phaseshift_server.py:180-228` が必須化 | production・常時 | 513 | structured 消滅 + server 起動不能(要改修) |
| 5b | LocalAI patch 0003(tool policy) | `phaseshift.tool_policy_transport` marker 押印、`parallel_tool_calls` フィールド追加、INVALID_ARGUMENT→400 | `vendor/localai/patches/0003` | `phaseshift_backend.py:283-287` が marker 不在なら 400 | **production・OpenCode Basic に必須** | 207 | **tool calling 全滅** |
| 5c | LocalAI patch 0004(composition) | structured+tools 同時指定許可、marker、grammar rule 順序安定化(cache 向け) | `patches/0004` | `phaseshift_backend.py:291-295` | conditional(composition marker 時) | 206 | 同時指定 400 + cache miss 増 |
| 5d | LocalAI patch 0005 / 0006 | Responses reasoning forward / reasoning item 開始順序 | `patches/0005,0006` | Responses reasoning 経路 | conditional(reasoning 時) | — | Responses reasoning 崩壊 |
| 6 | backend transport(decision tree) | `grammar` と `structural_tag` のどちらを送るかを policy×reasoning×composition で決定、metadata 検証、sidecar 生成/後始末 | `_prepare phaseshift_backend.py:266`、tree `:306-326`、sidecar `:810-847` | `Predict:386-397`、`_stream_text:514`、`_stream_tools:595` | production・常時 | ≒150 | marker 分岐削除で spoof 可能、`grammar` 分岐削除で fail-open、sidecar 削除が xgrammar 依存の唯一の解除点 |
| 7 | compute JSONL transport | `grammar`/`structural_tag` を受信・1MiB 制限・**相互排他**検証、空なら無制約 | `main.hip:1429-1445`、排他 `:1462-1464`、`create_constraint_state:1264-1277` | `compute_client.py:207-210`、`compute_runtime.hip:238-306` | request-conditional(両方空なら compiler を呼ばない `:293-295`) | 該当 85+18+68 | 欄削除で該当機能不可。空時経路(unconstrained)は無傷 |
| 8 | TokenConstraintCompiler / State | compiled grammar を bounded cache(64MiB / 256 entry FIFO)で共有、request 単位 matcher 生成 | `create token_constraint.cpp:248`、cache `:188-241`、`create_gbnf_state:280`、`create_structural_tag_state:297` | `compute_runtime.hip:257,281,302-305` → `main.hip:1265` | conditional(sidecar あり + constraint 指定時) | .cpp 339 + .h 98 = 437 | **全 constraint の心臓**。GBNF だけ削るなら `create_gbnf_state` のみで Structural Tag は生き残る |
| 9 | XGrammar | native constraint engine(v0.2.5)+ Python TokenizerInfo sidecar 生成・pin | `vendor/xgrammar/cpp/grammar_matcher.cc`、`structural_tag.cc`；sidecar `phaseshift_backend.py:810-847` | LoadModel `:201-210` → `--constraint-tokenizer-info`(`main.hip:1749`) → `compute_runtime.hip:106-123` | **production・hard requirement**(version 不一致で LoadModel fail `:819-822`) | vendor 全体(grammar_matcher 1338 / structural_tag 2440) | 全削除には backend の条件化が前提。structured/strict/reasoning/DFlash2 constraint が消滅 |
| 10 | CPU token mask generation | bitmask を output row 単位で CPU 生成、draft 分は `TraverseDraftTree` で K+1 行生成 | `fill_next_mask token_constraint.cpp:94-106`、`fill_draft_tree_masks:114-154` | `continuous_batcher.cpp:370-410`(`:382`)、DFlash2 `dflash2_spec_decoder.cpp:98-99` | request-conditional(`request->constraint != nullptr`) | 該当 40 | mask 供給不能 → 全 constraint が fail-closed に失敗。draft tree も同時に壊れる |
| 11 | allowed token count | mask 生成直後に host で popcount 総和を取り、LM Head 経路選択と trace に使用 | `constraint_allowed_count token_constraint.h:23-40`、`scheduled_batch.h:38`(既定 `UINT32_MAX`) | `continuous_batcher.cpp:388-393` → `executor.hip:675-693` → `lm_head_proxy.hip:824` | conditional + env trace(`PHASESHIFT_CONSTRAINT_TRACE`、既定無効) | ≒60 | **Performance 専用入力**。削除で layer 15 が full path に落ちる(正しさ無傷、追加転送なし) |
| 12 | H2D constraint mask | constrained 行がある step だけ1回の batched H2D | 判定 `executor.hip:666-669`、upload `:914-921`、buffer 確保 `:1856-1862` | `enqueue_batch` 内 | request-conditional(unconstrained では転送なし。constrained batch は HIP Graph 不可 `:878`) | ≒70 | 消すと mask 無視(fail-open)。unconstrained 性能は不変 |
| 13 | constrained sampling | greedy は allowed のみ argmax、stochastic は各段で disallowed 除外。許可0は EOS にせず request error | `make_sampling_constraint sampling.hip:14-25`、適用 `:87-93,125-135`、`stochastic_sampling.hip:19-49`、判定 `sampling_common.h:67-71` | `sampling_dispatch.hip:114-115,134-135`、correctness `kernels/correctness/detail/output.inc:55-80` | conditional(`reserved[0]&1u`) | 884 | **correctness レイヤ**。単独では外せない。削除で fail-open |
| 14 | constraint candidate extraction | mask を token id 昇順の candidate ID 配列へ展開 | `constraint_mask_to_candidates_kernel constraint_candidates.hip:11-52` | **唯一の caller = `lm_head_proxy.hip:547`**(他は test のみ) | **env のみ・既定 OFF** | 93 | layer 15 とセットで完全孤立。正しさ・既定性能に影響ゼロ |
| 15 | LM Head constraint opt(ExactCandidates / MaskedCoarse) | allowed 件数で exact 列挙か INT2 coarse→Top-N かを選び full vocab matmul を回避 | `select_lm_head_constrained lm_head_proxy.h:139-170`(閾値 128)、`select_constrained_exact:482`、`select_constrained_masked:580` | `try_launch_lm_head_proxy_fusion:758-892`、分岐 `:771-773` | **env のみ・既定 OFF**(decode mode 既定 0、DFlash2 時は `main.hip:1854-1856` で強制 0) | 1066 | `not_applicable` への置換なら full PSQ8 + sampling filter に退化して正しさ不変。既定性能影響ゼロ |
| 16 | DFlash2 constraint | verify 行ごとに mask 生成、drafter 提案 logits に適用、matcher accept で採否保証 | `build_constraint_masks dflash2_spec_decoder.cpp:88-110`、set `:333-356`、提案 mask `:732-756` | `main.hip:1275-1277`、mask 呼び出し `:98-99,460,867`、kernel `dflash2/executor.hip:2300-2309`、`kernels/dflash2/topk.hip:113-146` | conditional(`--dflash2-model-dir` 時)+ env `PHASESHIFT_DFLASH2_CONSTRAINT_PROPOSAL` **既定 true** | 該当 ≒180 | proposal mask のみ削除可(verify が担保、受入率のみ低下)。`fill_draft_tree_masks` 削除は DFlash2+constraint が fail-closed |
| 17 | constrained stochastic sampling の分岐 | constraint がある行を radix stochastic 高速経路から外し従来実装へ | `executor.hip:645-648,658-663`、`sampling_selector.cpp:23-34` | `sampling_dispatch.hip:114-115` | conditional(radix は「constraint なし」が選択条件 `:27-33`) | ≒70 | 分岐を消すと radix が mask を無視して不正採样。正しい整理は radix の constraint 対応(要実装) |
| 18 | constraint tests | native 単体 + server E2E | `cmake/tests.cmake:123,144-147,569`、`tests/server/run_server_regression.py:43-129` | required に登録(`tests.cmake:262,336-339`) | **test**: `test_constraint_mask`(cpu/required)、`test_dflash2_constraint_mask`(cpu/required)、`test_structural_tool_constraint`、`test_composite_structural_constraint`、`test_constraint_candidates`(gpu1/required)、`test_lm_head_constraint_perf`(optional) | native 計 ≒2,144 / server constraint 系 計 ≒1,737 | レイヤ削除時は対応 group(constraints 6 / tools 7 / structured 6 / composition 6 / reasoning-tools 8)と required ラベルの同時削除が必要 |

---

## 2. 分類表(レイヤ × 必要な機能)

○ = 必要 / △ = 条件付き・部分的 / − = 不要

| レイヤ | OpenCode Basic | strict tool のみ | structured のみ | reasoning のみ | DFlash2 のみ | Performance のみ | 備考 |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | --- |
| 1a tool policy parse | ○ | ○ | − | − | − | − | `tool_choice` の fail-closed 解析(auto 既定) |
| 1b strict schema 検証 | − | ○ | − | − | − | − | `strict is True` のみ |
| 1c chat codec | ○ | ○ | ○ | ○ | − | − | 全機能が共有 |
| 1d message codec | ○ | ○ | − | − | − | − | proto↔HF、tools 正規化 |
| 1e `message_codec.resolve_tool_choice` | − | − | − | − | − | − | **dead code** |
| 2 builder | △ | ○ | △ | ○ | − | − | 既定 request では非活性 |
| 3 grammar transport | − | − | ○ | △ | − | − | 起動 probe が必須化 |
| 4 Structural Tag + reasoning 合成 | △ | ○ | △ | ○ | − | − | 該当時のみ |
| 5a patch 0001 / 0002 | − | − | ○ | △ | − | − | fail-closed structured |
| 5b **patch 0003(tool policy)** | **○** | ○ | △ | − | − | − | **marker 不在だと tool 全滅 = Basic に必須** |
| 5c patch 0004(composition) | − | △ | ○ | △ | − | − | rule 順序安定化は cache 向け |
| 5d patch 0005 / 0006 | △ | △ | − | **○** | − | − | Responses reasoning + tools stream |
| 6 backend decision tree | ○ | ○ | ○ | ○ | − | − | marker 検証 `:283-287` は Basic 必須 |
| 7 compute JSONL | △ | ○ | ○ | ○ | ○ | − | 両方空なら無制約経路 |
| 8 TokenConstraintCompiler/State | − | ○ | ○ | ○ | ○ | − | 全 constraint の心臓 |
| 9 XGrammar + sidecar | − | ○ | ○ | ○ | ○ | △ | grammar cache(64MiB/256)は付加的性能 |
| 10 CPU mask | − | ○ | ○ | ○ | ○ | − | unconstrained では生成されない |
| 11 allowed token count | − | − | − | − | − | **○** | LM Head gate と trace 専用 |
| 12 H2D mask | − | ○ | ○ | ○ | ○ | − | unconstrained では転送なし |
| 13 constrained sampling filter | − | ○ | ○ | ○ | ○ | − | correctness、単独では外せない |
| 14 constraint candidates | − | − | − | − | − | **○** | env 既定 OFF、caller は 15 のみ |
| 15 LM Head constraint opt | − | − | − | − | − | **○** | env 既定 OFF |
| 16 DFlash2 constraint | − | − | − | − | **○** | △ | proposal mask 既定 true、DFlash2 時のみ |
| 17 radix stochastic 分岐 | − | − | − | − | △ | **○** | constraint 時のみ高速経路が無効化 |
| 18 constraint tests | △ | △ | △ | △ | △ | △ | group 別に分割済み、required 6件 |

**要点**: OpenCode Basic に必要なのは **1a / 1c / 1d + patch 0003 + backend の marker 検証**だけで、constraint engine(8〜14)は Basic に対して非活性。Performance の専業は **11, 14, 15**(+17 の分岐)、DFlash2 専業は **16**。

---

## 3. 集約影響分析

### A. constraint stack 全抜きでも best-effort tool calling は動くか

**YES(コード経路上)。ただし constraint stack と独立した必須条件が3つある。**

1. `tool_choice=auto` + `parallel_tool_calls=true`(既定)+ 全 tool が non-strict なら `needs_structural_constraint` が false(`tool_constraint.py:218-226`)→ `build_structural_tag` が `None`(`:275-276,317-319`)。テスト固定: `test_tool_constraint_builder.py:124-126`「builder-loose-auto-none」。
2. backend は `grammar=""` / `structural_tag=None`(`phaseshift_backend.py:322-324`)、compute は両方空なら compiler を呼ばない(`compute_runtime.hip:293-295`)→ mask 生成も H2D も無し。
3. tool call の生成・parse は constraint 非依存(`codec.py:81-101` の宣言的 `TOOL_RESPONSE_TEMPLATE`、`:359-361`、`:373-387`)。実測: `test_server_strict_tools.py:196` の `loose-auto-call` は structural tag なしでも 1 call 検出。`test_server_unconstrained_oracle.py:62` は `--constraint-tokenizer-info` なし compute が oracle と bit-exact。

**残る必須条件(constraint とは独立)**:
- (i) Python xgrammar は LoadModel が無条件に sidecar を書き、失敗で model load が失敗(`phaseshift_backend.py:201-210, 810-847`)。外すには backend 改修が前提(現状その分岐は存在しない)。
- (ii) tool policy transport marker(`phaseshift.tool_policy_transport=="1"`)が無いと tool request は 400(`phaseshift_backend.py:283-287`)。LocalAI patch 0003 由来 = constraint engine とは独立。
- (iii) HF chat template + Transformers response parser の capability(`codec.py:390-460`)。

### B. GPU constraint 最適化(14, 15)だけ削除した場合

- **残るもの**: CPU mask 生成(10)、H2D(12)、sampling filter(13)、allowed count(11)、XGrammar/compiler(8, 9)、DFlash2 verify mask(16)。**strict / structured / reasoning / DFlash2 の正しさは全部残る**。
- 「制約を無視する fallback は無い」(`docs/developer/qwen35.md:324`)を壊さない — constrained 行が `not_applicable` を返して full PSQ8 + sampling filter に退化するだけ。
- **既定設定での遅化 = ゼロ**(14/15 は env 依存・既定 OFF)。既存docs値は `PHASESHIFT_TARGET_LM_HEAD_PROXY=1` 時のみで full PSQ8 2362us → 0.26〜0.30ms(**8〜9倍**、`docs/rnd/lm_head/constraint_lm_head_hybrid_gate4.md:97-107`)。
- **必須対応**: `try_launch_lm_head_proxy_fusion` の constrained 分岐(`:771-773, 821-872`)を `return not_applicable` 相当に置換。素朴に分岐を消すと unconstrained 用 `select/select_shadow`(`:873-891`)が mask を無視して **fail-open** になる。required テスト `test_constraint_candidates` の削除もセット。

### C. structured output だけ削除した場合

- **削除対象**: patch 0001(102) + 0002(411)、起動 probe(`phaseshift_server.py:180-228`)、backend の `request.Grammar` 分岐(`:292,313,326`)、compute の `grammar` フィールド、`create_gbnf_state`、composite/reasoning 内の `{"type":"grammar"}`。tests: `structured`(6) + `composition`(6) + `constraints` group の GBNF 系。
- **生き残るもの**: **strict tool calling 全部**(Structural Tag は `CompileStructuralTag` を使い GBNF と別)、reasoning(structural tag 経路)、レイヤ 8〜14・16・17(共用)、**OpenCode Basic は完全無傷**。
- **罠**: patch 0003/0004 は `Requires: 0001,0002` のため、0001/0002 を外すと **0003(=OpenCode Basic に必須)も貼れなくなる**。patch の作り直しが要る。

### D. strict tool calling だけ削除した場合

- **削除対象**: `validate_strict_parameters` 群(`tool_constraint.py:134-216`)、`_tool_tag` の `qwen_xml_parameter` 分岐(`:296-301`)、`needs_structural_constraint` の strict 項目(`:226`)。tests: `test_server_strict_tools*`(3)、`test_server_responses_strict_tools`、`test_tool_constraint_builder.py` strict 系。
- **影響**: `strict:true` request が **silent fail-open 化**(現状は 400、`docs/user/server.md:449-452`)。AGENTS の fail-closed 方針と衝突する点が唯一の実害。
- **無傷**: structural tag 自体(`required`/named/`parallel=false` は strict 無しでも生成)、structured output、reasoning、DFlash2、mask/H2D/sampling、GPU 最適化、OpenCode Basic。

### E. 「完全削除」の4シナリオ

| シナリオ | 影響範囲 |
| --- | --- |
| Constraint を完全削除 | structured / strict / required / named / reasoning 合成 / DFlash2 constraint が全滅。OpenCode の best-effort tool calling は上記3条件(i〜iii)さえ残れば動作。ただし patch 0003 は constraint と独立なので残す必要がある。patch chain(0001-0004 の Requires)の作り直しが必要 |
| Constraint の GPU 最適化だけ削除 | 正しさ・機能は無傷(レイヤ 8〜13 が full-vocab 経路で担保)。既定性能への影響ゼロ。`test_constraint_candidates`(required)のみ削除対象 |
| Structured Output だけ削除 | strict/reasoning/Basic は無傷。patch 0001/0002 が 0003/0004 の前提のため patch 再構成が必要。起動 probe の縮小も必要 |
| Strict Tool Calling だけ削除 | レイヤ最小。strict:true のみ fail-open 化。structural tag・全他機能は無傷 |

---

## 4. NEEDS_INVESTIGATION

1. commit `d87507d4`(allowed token count)は read-only tools では `.git` 内に該当 hash が確認できず、実装内容は symbol(`constraint_allowed_count`)からの逆推定。
2. constraint stack 全抜きでの best-effort tool calling を **単独で検証する test が存在しない**(現状はフルスタック起動下での `loose-auto-call` 検証のみ)。
3. `PHASESHIFT_DFLASH2_CONSTRAINT_PROPOSAL=0` 時の受入率影響(既存docsに数値なし)。
4. grammar cache(64MiB / 256 entry)の実効 cache hit 率(既存docsに数値なし)。
