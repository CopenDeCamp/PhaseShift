# phaseshift-server

OpenAI互換のchat servingアプリ。model directoryを指定して起動すると、
`/v1` 配下でOpenAI Chat Completions APIのsubsetを提供する。

```bash
./build/phaseshift-server \
    --model-dir /path/to/Qwen3.8-27B-PSQ \
    --port 8000
```

起動すると次のエンドポイントが利用できる。

```text
http://127.0.0.1:8000/v1
```

対応範囲は次のとおり。

* `GET /v1/models`
* `POST /v1/chat/completions`（非stream / stream）

embeddings、vision、audio、Responses API、認証、TLS、複数modelは対象外。

依存は Python / `aiohttp` / `transformers==5.14.1` のみである。
LocalAI・gRPC・protobuf・XGrammar は要らない。

## 起動

```bash
./build/phaseshift-server \
    --model-dir /path/to/Qwen3.8-27B-PSQ
```

起動順は model processor load → `phaseshift-compute` subprocess 起動 → `ping` →
capability 取得 → HTTP listen である。model load は server lifetime 中に 1 回だけ行われ、
request ごとに compute process を起動しない。server が ready になるまで port は accept しない。

起動完了時、stderr に以下が表示される。

```text
PhaseShift Server

Model:        phaseshift (Qwen3.8-27B-PSQ)
Endpoint:     http://127.0.0.1:8000/v1
Context:      32768
Concurrency:  4
KV dtype:     bf16
KV capacity:  auto
Prefix cache: protocol only (capabilities.prefix_cache=false)
Default output: 4096

Supported:   Chat Completions, SSE, tools, sampling, concurrency, cancel
Unsupported: Responses, structured output, strict tools, reasoning

Ready.
```

## options

```text
--model-dir PATH       model directory (required)
--model NAME           served model name (default phaseshift)
--host HOST            bind address (default 127.0.0.1)
--port N               bind port (default 8000)
--compute PATH         phaseshift-compute path (default: next to this script)

--max-seq-len N        context window (default 32768)
--max-concurrent-requests N  max in-flight requests (default 4)

--arena-gib N          GPU arena size in GiB (default 16)
--device N             GPU device (default 0)
--kv-cache-dtype TYPE  bf16 | fp8_e4m3 | psq4 | psq8 (default bf16)
--page-tokens N        paged KV page size in tokens (default 16)
--kv-cache-capacity-tokens N physical KV token capacity (0 = one context、
                     DFlash2 で concurrency > 1 のときは自動拡張)

--default-max-output-tokens N output cap when the request specifies none (default 4096)

--dflash2-model-dir PATH  DFlash2 draft model directory (speculative decoding を有効化)
--dflash2-drafts N     speculative draft tokens per round (default 7)

--verify-weights       verify quantized payload CRC32 on load
```

不正な値は起動時に拒否される。`--max-concurrent-requests` / `--max-seq-len` /
`--default-max-output-tokens` / `--page-tokens` / `--dflash2-drafts` は 1 以上である。

`--kv-cache-dtype psq4` / `psq8` は `head_dim == 256` を要求する。

## DFlash2 speculative decoding

`--dflash2-model-dir` を指定すると DFlash2 を有効化する。詳細は
[../developer/dflash2.md](../developer/dflash2.md) を参照。

指定時の挙動:

- `--max-concurrent-requests` は指定値を尊重する。DFlash2 でも複数 request を受付できる。
  `--kv-cache-capacity-tokens` が未指定のときは
  `(concurrency + 1) × (max_seq_len + 1)` へ自動拡張する。
- tool calling / sampling / cancel は DFlash2 経路でも利用できる。

```bash
./build/phaseshift-server \
    --model-dir models/Qwen3.8-27B-PSQ \
    --dflash2-model-dir models/Qwen3.8-27B-DFlash2-PSQ \
    --port 8001 \
    --device 0 \
    --arena-gib 31 \
    --max-seq-len 4096 \
    --kv-cache-dtype psq4
```

## 使い方

### chat completions

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "phaseshift",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0,
    "max_tokens": 32
  }'
```

### streaming

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "phaseshift",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "stream": true,
    "temperature": 0,
    "max_tokens": 32
  }'
```

末尾に `data: [DONE]` が出る。stream chunk を連結した文字列は、同じ request の非 stream 結果と一致する。

### sampling

受け付ける request parameter:

```text
model / messages / stream
max_tokens / max_completion_tokens（両方指定で値が異なる場合は reject）
temperature / top_p / top_k / seed
tools / tool_choice
```

`top_k` は PhaseShift extension である。指定がない値の既定は
`temperature=0`（greedy）/ `top_p=1.0` / `top_k=0` / `seed=0` である。

### tools

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "phaseshift",
    "messages": [
      {"role": "user", "content": "大阪の現在の気温をget_weatherを使って確認して"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 128
  }'
```

応答は OpenAI 形式の `tool_calls` になる。function arguments は JSON object ではなく
JSON string である。

```json
{
  "choices": [
    {
      "finish_reason": "tool_calls",
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_0",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\": \"大阪\"}"
            }
          }
        ]
      }
    }
  ]
}
```

PhaseShift Server は tool を実行しない。実際の tool 実行は client 側（OpenCode 等）が行う。
client が tool 結果を `role: "tool"` message として返すと、PhaseShift はそれを読んで次の
assistant 応答を生成する。

`tool_choice` は `auto` / `none` のみ受け付ける。`none` は tool 定義を model へ渡さない。

tool calling は best-effort である。生成された tool call は Transformers の Qwen response
parser で解析し、schema の enforce や生成後の修復は行わない。parser が tool-call-like な
出力を解析できない場合は parse failure として扱い、生の markup を content として捏造しない。

streaming 中の tool call にも対応する。OpenAI 形式の `delta.tool_calls` として返り、
model 固有の `<tool_call>` 等の markup は client へ漏れない。

### concurrency と cancel

`--max-concurrent-requests N` は最大 in-flight request 数であり、同時に 1 executor step で
scheduler が扱える最大 request 数でもある。physical KV capacity
（`--kv-cache-capacity-tokens`）が不足する場合、一部の request は Queued となり、
KV Banker が Active 完了後に admission する。`--kv-cache-capacity-tokens 0`（default）は
1 context 分（`max_seq_len + 1`）を意味する。

stream client が connection を切断すると、in-flight generation は cancel される
（`finish_reason="cancelled"`）。cancel は request-specific であり、他の in-flight
request の KV / GDN / token stream には影響しない。

## サポート外の API

以下は明示的に HTTP 400（`code=unsupported_parameter`）で拒否する。silent ignore はしない。

```text
response_format
reasoning_effort / reasoning
text
tool_choice="required" / named function
tools[].function.strict=true
parallel_tool_calls=false
n != 1 / logit_bias / logprobs
```

以下は route 自体を持たない。

```text
POST /v1/responses   -> HTTP 404 {"error": {"code": "unsupported_endpoint"}}
その他の未登録 path -> HTTP 404 {"error": {"code": "not_found"}}
```

## Prefix Cache

prefix cache の GPU 実装は提供しない。protocol contract のみを保有する。

- server は chat template の stable prefix boundary を計算する
- compute の capability が `false` のため、server は常に
  `prefix_cache_checkpoint_position = 0` を送る
- raw JSONL client が `0` を超える値を送った場合は `invalid_argument` で拒否する

将来 prefix cache を再実装したときは `capabilities.prefix_cache` が `true` に変わり、
server も boundary を要求するようになる。詳細は
[../developer/prefix_cache.md](../developer/prefix_cache.md) を参照。

## エージェントからの利用

agent frontend は OpenAI 互換 client として接続する。ユーザーが起動するのは
`phaseshift-server` だけである。

### OpenCode

OpenCode の project config（`opencode.jsonc`）に OpenAI-compatible provider を追加する。

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "phaseshift/phaseshift",
  "providers": {
    "phaseshift": {
      "name": "PhaseShift",
      "env": ["PHASESHIFT_API_KEY"],
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {"baseURL": "http://127.0.0.1:8000/v1"},
      "models": {"phaseshift": {
        "name": "PhaseShift",
        "limit": {"context": 32768, "output": 8192}
      }}
    }
  }
}
```

`limit` は server の `--max-seq-len` に合わせる。OpenCode は既定で 32000 の output
budget を要求するが、server は明示 `max_tokens` が残り context window を超える request を
fail-closed で拒否するため、`limit` を指定しないと agent prompt で HTTP 400 になる。
`limit.output` は `--max-seq-len` から実際の prompt 長を引いた残りに収まる必要がある。

`PHASESHIFT_API_KEY` は任意の dummy 値でよい（PhaseShift は認証を行わない）。
model は `phaseshift/phaseshift` を選ぶ。

Codex CLI は Responses API を使うため対象外である。

## runtime dependency

```bash
pip install 'aiohttp' 'transformers==5.14.1'
```

server / compute の責務境界は
[../developer/architecture.md](../developer/architecture.md) を参照する。
ユーザーは `phaseshift-compute` を個別に起動する必要はない。

## エラー挙動

- 起動: 不正な option 値、model directory 不在、`phaseshift-compute` 不在は、
  stderr へ理由を表示して non-zero で終了する。
- request: unsupported な parameter、不正な message role、不正な tool arguments、
  未知の model name は HTTP 400（model は 404）であり、生成を開始しない。
- context: prompt が `--max-seq-len` を超える場合、および明示 `max_tokens` が残り context を
  超える場合は HTTP 400 となる。
- cancel: stream client の切断などで cancel された generation は `finish_reason="cancelled"`
  となる。

## テスト

server / compute の E2E は `tests/server/` の Python suite で検証する。model directory は
`PHASESHIFT_MODEL_DIR` で指定する。

```bash
PHASESHIFT_MODEL_DIR=models/Qwen3.8-27B-PSQ \
python3 tests/server/run_server_regression.py --all
```

`tests/server/run_server_regression.py` が canonical な server regression runner である。
non-zero return は失敗として扱う。`--list` で group と test の一覧を表示できる。
suite の詳細は [../developer/testing.md](../developer/testing.md) を参照する。
