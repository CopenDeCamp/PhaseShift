# Server

PhaseShift Server (`phaseshift-server`) の開発者向け入口である。server の対応 surface と、
詳細 contract を持つ current doc へのリンクを示す。個別仕様は各 doc を正とする。

## Scope

OpenAI 互換の serving product であり、inference engine ではない。
`phaseshift-server` 自身が aiohttp で OpenAI Chat Completions の subset を提供し、
`phaseshift-compute --serve-stdio` へ直接 JSONL stdio で接続する。

```text
OpenAI client
    │ HTTP / SSE
    ▼
phaseshift-server
    ├── aiohttp
    ├── openai_protocol.py   request validation / wire encoding
    ├── chat_service.py      HF messages -> chat template -> compute -> response parser
    └── compute_client.py    asyncio native JSONL client
    │ persistent JSONL stdio
    ▼
phaseshift-compute --serve-stdio   (token IDs in / token IDs out)
```

起動順は model processor load → compute subprocess 起動 → `ping` → capability 取得 →
HTTP listen である。server が ready になるまで port を accept しない。

## Supported

- `GET /v1/models`
- `POST /v1/chat/completions`（`stream=false` / `stream=true` の SSE）
- sampling: `temperature` / `top_p` / `top_k` / `seed` / `max_tokens` /
  `max_completion_tokens`
- tools: `tools`、`tool_choice="auto"` / `"none"`、`assistant.tool_calls`、
  `role=tool` の round-trip、OpenCode agent loop
- best-effort tool calling（生成は Transformers の Qwen response parser で解析する。
  parse に失敗した output を捏造・修復しない）
- concurrent serving（複数 request が Continuous Batching へ到達する）
- request cancellation（HTTP client disconnect → JSONL cancel → ContinuousBatcher cancel）
- DFlash2 speculative decode（constraint なしの通常経路）

## Unsupported

以下は明示的に HTTP 400（`code=unsupported_parameter`）で拒否する。silent ignore はしない。

- `response_format`（Structured Output）
- `reasoning_effort` / `reasoning`（reasoning transport）
- `text`
- `tool_choice="required"` / named function
- `strict:true`（tool 単位で拒否）
- `parallel_tool_calls=false`
- `n != 1` / `logit_bias` / `logprobs`

以下は route 自体を持たない。

- `POST /v1/responses` → HTTP 404 `code=unsupported_endpoint`
- `POST /v1/completions` など未登録 path → HTTP 404 `code=not_found`

## Prefix cache

protocol contract のみを保有し、GPU 実装は提供しない。

- server は chat template の stable prefix boundary を計算する
- `capabilities.prefix_cache == false` のため常に
  `prefix_cache_checkpoint_position = 0` を送る（fail-closed）
- compute は `0` 以外を `invalid_argument` で拒否する
- `done` event は `prefill_tokens` / `restored_tokens` / `cache_checkpoint_tokens` を返す
  （現在は `restored_tokens = 0` / `cache_checkpoint_tokens = 0`）

詳細は [prefix_cache.md](prefix_cache.md) を参照。

## Architecture

責務境界と layer 構成は [architecture.md](architecture.md) を参照。

## Detailed contracts

- generation stop token: [qwen35.md](qwen35.md)
- sampling: [sampling.md](sampling.md)
- prefix cache の protocol contract: [prefix_cache.md](prefix_cache.md)
- server regression suite: [testing.md](testing.md)

## User guide

利用方法・起動 options・API 例は [../user/server.md](../user/server.md) を参照。

## Known limitations

- `Full JSON Schema semantics` は保証しない。tool の JSON Schema は chat template へ
  tool 説明として渡すだけで、生成時の schema enforcement は行わない。
- embeddings / vision / audio / realtime API / authentication / TLS termination /
  multi-model hot loading は対象外。
