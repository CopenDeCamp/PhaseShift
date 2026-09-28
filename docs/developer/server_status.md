# Server

PhaseShift Server (`phaseshift-server`) の開発者向け入口である。server の対応 surface と、
詳細 contract を持つ current doc へのリンクを示す。個別仕様は各 doc を正とする。

## Scope

OpenAI 互換の serving product であり、inference engine ではない。LocalAI v4.10.0 を hidden API
frontend として起動し、external gRPC backend が `phaseshift-compute --serve-stdio` を所有する。

```text
OpenAI client
    -> phaseshift-server (launcher)
    -> LocalAI v4.10.0
    -> phaseshift backend (Python)
    -> phaseshift-compute --serve-stdio   (token IDs in / token IDs out)
```

## Supported interfaces

- `/v1/chat/completions`: 非stream / stream、tool calling、streaming tool calls、tool result round-trip
- `/v1/responses`: 非stream / stream、function tool、function_call_output round-trip
- `/v1/models`
- tool policy: `auto` / `none` / `required` / named function、`parallel_tool_calls`、`strict:true`
- Grammar-Constrained Decode: GBNF / XGrammar / Structural Tag
- structured text + tools の合成（1 つの Structural Tag）
- reasoning（opt-in）、reasoning + tools / structured
- concurrent serving / Continuous Batching / request cancellation
- GPU-resident prefix cache（server default 有効）

## Architecture

責務境界と layer 構成は [architecture.md](architecture.md) を参照。

## Detailed contracts

- grammar / structured generation / generation stop token: [structured_generation.md](structured_generation.md)
- reasoning transport と semantics: [reasoning.md](reasoning.md)
- GPU-resident prefix cache: [prefix_cache.md](prefix_cache.md)
- server regression suite: [testing.md](testing.md)

## User guide

利用方法・起動 options・API 例は [../user/server.md](../user/server.md) を参照。

## Known limitations

- `Full JSON Schema semantics` は保証しない。LocalAI converter が理解する JSON Schema subset のみを
  対象とする。conversion failure は `Chat structured fail-closed`（Responses も同じ）として
  HTTP 400 で拒否し、unconstrained fallback しない。
- embeddings / vision / audio / realtime API / WebSocket Responses / background Responses /
  stored Responses / MCP server execution / server-side tool execution / authentication /
  TLS termination / multi-model hot loading は対象外。
- 初期 audit では Responses structured transport が BLOCKED_EXTERNAL と記録された。現在は
  fail-closed で supported である。経緯は
  [../rnd/server/server_history.md](../rnd/server/server_history.md) を参照。
