# Minimal OpenCode Backend

> Status: historical R&D record（server 簡素化・constraint / LocalAI 削除 **前** の状態を記録している。現在の仕様ではない。現在の仕様は docs/developer/ を参照）


PhaseShift を OpenCode の coding agent バックエンドとして成立させる最小機能集合の棚卸し。
すべての記述はコード側(call site / CMake / default / runtime 分岐 / tests)で確認したもの。
docs の記述だけでの判定は行っていない。

## 1. 現在の接続経路(実測)

```text
OpenCode (opencode run -m phaseshift/phaseshift)
  │  @opencode/ai/providers/openai-compatible → HTTP POST /v1/chat/completions (SSE)
  ▼
phaseshift-server (src/apps/server/phaseshift_server.py)
  │  LocalAI を subprocess 起動・model YAML 生成・readiness poll・fail-closed probe
  │  phaseshift_server.py:322-338 (Popen), :87-122 (YAML), :180-228 (probe)
  ▼
LocalAI v4.10.0 + PhaseShift patch 0001..0006 (vendor/localai/patches/)
  │  private gRPC: backend.Backend (vendor/localai/backend.proto)
  ▼
phaseshift backend (src/apps/server/backend/phaseshift_backend.py)
  │  HF chat template → token IDs、tool constraint 構築、出力 parse
  │  phaseshift_backend.py:266-355 (_prepare)
  ▼
compute_client.py ── 永続 JSONL stdio (ops: ping/generate/cancel/shutdown)
  │  main.hip:1537-1569
  ▼
phaseshift-compute --serve-stdio (src/apps/compute/main.hip)
  ▼
Qwen35 runtime → HIP kernels → GPU
```

- `phaseshift_server.py` は HTTP プロキシではなく LocalAI のランチャー兼 orchestrator(`phaseshift_server.py:322-338`)。
- backend → compute は JSONL(`compute_client.py:197-214`)。tool policy / constraint / reasoning の transport は gRPC `Metadata` と `ChatDelta`。
- LocalAI patches: 0001(fail-closed structured)/ 0002(Responses structured)/ 0003(**tool policy transport**)/ 0004(composition)/ 0005・0006(Responses reasoning)。
- Python 依存: `transformers==5.14.1`、`grpcio`、`protobuf`、`xgrammar==0.2.5.post1`。

## 2. Tool Calling の分解判定

### A〜E: 基本 tool loop(OpenCode に必須)

| 項目 | 実装 | 証拠 | default |
| --- | --- | --- | --- |
| A. tools を prompt へ | HF chat template に `tools=` をそのまま渡すだけ | `codec.py:142-151`(`_render`)、呼び出し `phaseshift_backend.py:328-330` | default-on |
| B. model が tool call を生成 | Qwen XML 系 markup、`TOOL_RESPONSE_TEMPLATE` が宣言的テンプレート | `codec.py:59-101` | default-on |
| C. tool call を parse | **Transformers の `processor.parse_response` / `get_response_parser`**(自作 regex ではない) | `codec.py:359-361, 373-387, 528-575` | default-on |
| D. OpenAI 形式へ変換 | parse 結果 → OpenAI dict → gRPC `ChatDelta.tool_calls`、HTTP 表現は LocalAI 側 | `codec.py:507-525`、`phaseshift_backend.py:434-446, 679-697`、`test_server_tools.py:76-79` | default-on |
| E. tool result 再入力 | `message_codec.proto_messages_to_hf`(assistant `tool_calls` / `role:"tool"`、system/developer マージ) | `message_codec.py:84-124`、`test_server_tools.py:96-119`、`test_agent_opencode.py:149-158` | default-on |

### F〜K: constraint stack(A〜E とは独立)

| 項目 | 実装 | default | OpenCode への位置づけ |
| --- | --- | --- | --- |
| F. `strict:true` | `tool_constraint.py:108-131, 134-216`(OpenAI strict contract 検証、違反は 400) | **OFF**(`strict is True` のみ、`tool_constraint.py:126`) | strict tool calling のみ |
| G. `tool_choice` required/named | `tool_constraint.py:39-88`(fail-closed、silent auto fallback なし) | `auto` 既定 | Advanced |
| H. Structural Tag | `tool_constraint.py:312-352`、decision tree `phaseshift_backend.py:306-326` | **条件付き**(`needs_structural_constraint` が required/named・parallel=false・strict のいずれかでのみ生成。`auto`+`parallel=true`+non-strict なら `None`) | Advanced |
| I. XGrammar | native `vendor/xgrammar`、Python sidecar `_write_constraint_tokenizer_info` | **現状 hard requirement**(LoadModel が無条件に sidecar を書き、失敗で LoadModel 失敗 `phaseshift_backend.py:810-822`) | 現状は必須だが「使わないなら外せる code path は未存在」→ 要改修 |
| J. token bitmask | CPU `GrammarMatcher::FillNextTokenBitmask`、`continuous_batcher.cpp:370-407` | request-conditional(constraint がある request のみ) | Advanced |
| K. GPU constrained sampling | `sampling_common.h:67`、`kernels/optimized/sampling.hip:14-25` | request-conditional(`reserved[0]&1u` で off 時はフィルタなし) | Advanced |

### 「constraint 全抜きでも best-effort tool calling は動くか」

**概ね YES**(コード経路上)。根拠:

1. `tool_choice=auto` + `parallel_tool_calls=true`(既定)+ 全 tool が non-strict → `needs_structural_constraint` が false → `build_structural_tag` が `None`(`tool_constraint.py:275-276, 317-319`)。
2. backend は `tools_requested` 分岐で `grammar=""` / `structural_tag=None`(`phaseshift_backend.py:322-324`)。
3. compute は空 constraint なら constraint compiler を呼ばない(`compute_runtime.hip:293-295`)。
4. parse は constraint 独立(`phaseshift_backend.py:414-431` は `prepared.tools` のみ見る)。
5. 実測: `test_server_strict_tools.py:188-197` の `loose-auto-call` — structural tag が `None`(完全 unconstrained)でも 1 call 検出。
6. constraint stack 全抜きの compute が oracle と bit-exact: `test_server_unconstrained_oracle.py:39-53`(`--constraint-tokenizer-info` なしで起動)。

**それでも constraint stack とは独立に残る必須条件**:

- (i) **Python xgrammar の hard requirement**: LoadModel が無条件に sidecar を書き、失敗すると model 読み込み自体が失敗(`phaseshift_backend.py:201-210, 811-822`)。constraint を使わない運用にするには**この呼び出しを外す改修が必要**(現状その code path は存在しない)。
- (ii) **tool policy transport marker**: tool request で `phaseshift.tool_policy_transport == "1"` を必須とし、無ければ `INVALID_ARGUMENT`(`phaseshift_backend.py:283-287`)。LocalAI patch 0003 由来。素の LocalAI では tool calling 自体が 400(`test_tool_policy_transport.py:122-142`)。
- (iii) HF chat template + Transformers response parser の capability(`codec.py:390-460` probe)。

→ 「constraint stack を外した状態で best-effort tool calling を直接検証する test は存在しない」→ NEEDS_INVESTIGATION。

## 3. Minimal OpenCode Backend(残す必要がある機能)

OpenCode は `@opencode/ai/providers/openai-compatible` 経由で Chat Completions + SSE + tool loop を使う(`test_agent_opencode.py:2,25,54`)。

| # | 機能 | 必須根拠 | 現状 |
| --- | --- | --- | --- |
| 1 | `POST /v1/chat/completions`(非 stream) | `test_agent_opencode.py:149-158` が複数 request の tool loop を確認 | 実装済み |
| 2 | SSE streaming(`stream:true` → `delta.content` / `delta.tool_calls` / `[DONE]`) | OpenCode は通常 stream | 実装済み(`phaseshift_backend.py:456-499`) |
| 3 | `GET /v1/models`(discovery / readiness) | `test_server_chat.py:43-47` | 実装済み |
| 4 | tools を chat template へ注入(A) | `codec.py:142-151` | 実装済み |
| 5 | tool call 生成 + parse(B, C) | `codec.py:81-101, 359-361, 528-575` | 実装済み |
| 6 | OpenAI `tool_calls` 変換 + `finish_reason="tool_calls"`(D) | `phaseshift_backend.py:434-446` | 実装済み(HTTP 変換は LocalAI) |
| 7 | tool 結果 round-trip(E) | `message_codec.py:84-124` | 実装済み |
| 8 | `tool_choice` auto/none/required/named(fail-closed) | `tool_constraint.py:39-88` | 実装済み |
| 9 | tool policy transport handshake(LocalAI patch 0003) | `phaseshift_backend.py:283-287` | 必須(patch 前提、constraint stack とは独立) |
| 10 | context / max_tokens の fail-closed | `phaseshift_backend.py:357-368` | 実装済み(OpenCode の `limit` 設定が前提) |
| 11 | 断線による cancel | `phaseshift_backend.py:393, 519, 600` | 実装済み |
| 12 | concurrency(`--max-concurrent-requests` 既定4) | `phaseshift_server.py:252, 331, 335` | 実装済み |
| 13 | multi-turn prefix cache | `test_agent_opencode.py:166-172` | server 既定 ON(16384/8) |
| 14 | HF Transformers response parser capability | `codec.py:390-460` | 実装済み |
| 15 | Python `xgrammar==0.2.5.post1` | `phaseshift_backend.py:810-822` | **現状 hard requirement**(要改修で外せる) |

推論エンジン側の Basic(model load / tokenizer / prefill / decode / sampling / KV state / JSONL transport)は `phaseshift-compute --serve-stdio` が担う。

## 4. 現在構成 vs Minimal 構成

| サブシステム | 現在 | Minimal での扱い |
| --- | --- | --- |
| Chat Completions + SSE + tool loop A〜E | 実装済み | **保持(Basic)** |
| tool_choice / strict / Structural Tag(F〜H) | 実装済み、default は緩い | **保持(Advanced)**。ただし OpenCode の通常経路では非活性 |
| structured output(response_format / text.format + fail-closed probe) | 実装済み | Advanced。OpenCode 必須ではない |
| **Responses API(独立 route)** | 実装済み(`test_server_responses*.py`) | **OpenCode には不要**(openai-compatible provider は使わない。Codex CLI 向け) |
| reasoning / thinking envelope | 実装済み、opt-in/既定 OFF | Advanced |
| structured + tools composition | 実装済み | Advanced |
| **XGrammar C++ + Python sidecar** | **LoadModel の hard requirement** | **削除可能性の大きい subsystem**(constraint を使わない運用にするには backend の sidecar 書き出しを条件化する改修が前提) |
| token bitmask / H2D / GPU constrained sampling | 実装済み(request-conditional) | Advanced(strict / structured のために必要) |
| constraint LM head 最適化(ExactCandidates / MaskedCoarse) | env のみ・既定 OFF | **Performance**(削除しても constraint 自体は full-vocab 経路で動作) |
| LM head proxy(Fast / Shadow) | env のみ・既定 OFF | **Performance / Experimental**(OpenCode 無影響) |
| DFlash2 speculative decode | `--dflash2-model-dir` 指定時のみ | Performance(既定は非 spec) |
| prefix cache | server 既定 ON | Advanced(無効でも動作する) |
| MTP SpecDecoder スタック | **production 未接続**(bench/tests のみ) | **候補として最大の削除可能 subsystem** |
| HIP Graph | CMake option 既定 OFF(既定ビルドで未コンパイル) | Experimental |
| PA probe / paged-prune | option 既定 OFF・call site 0 件 | Experimental / R&D |
| gate 系 tests(25 件)・optional tests(42 件) | 外部 model 必須 | Tooling(R&D 由来) |
| `tools/rnd/` `tools/poc/` | docs/rnd のみが参照 | Tooling(R&D 残骸候補) |

## 5. 削除可能性の大きい subsystem(上位)

1. **MTP スタック**(`mtp_executor.hip` 696 + `spec_decoder.cpp` 501 + `mtp_kv_state.cpp` 190 + headers): production entrypoint からの call 0 件(`docs/developer/mtp.md:15-31` が contract 化)。ただし `spec_decode.cpp` の shared helper は DFlash2 が使うため分離不可。
2. **XGrammar 側の「constraint を使わない場合の強制要求」**(backend 改修が前提。dependency としては structured/strict のために残る)。
3. **LM head proxy 全体**(Fast / Shadow / constrained / 認証カーネル + tests): 既定 OFF。
4. **DFlash2 提案側の特殊経路**(INT2 coarse / 固定語彙 / NgramTail): すべて既定 OFF・opt-in。
5. **HIP Graph / PA probe**: 既定ビルドで未コンパイルまたは未呼び出し。
6. **R&D 由来 tests と `tools/rnd/`・`tools/poc/`**: production と無関係。

## 6. NEEDS_INVESTIGATION

1. `/v1/completions`、`/health` 等の LocalAI 上流 route が patched binary で実際に立つか(リポジトリ内 0 hit、LocalAI source は vendored していないため確認不能)。
2. constraint stack 全抜きでの best-effort tool calling を直接検証する test が存在しない。
3. `message_codec.resolve_tool_choice`(`message_codec.py:183-208`)は呼び出し元 0 件 = dead code 候補。
4. `probe_tool_calling` の streaming capability 判定(`phaseshift_backend.py:478-482`)は指定 transformers では常に available なため事実上の死に経路。
