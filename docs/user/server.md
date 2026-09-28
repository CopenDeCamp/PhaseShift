# phaseshift-server

OpenAI互換のchat servingアプリ。Qwen3.5 Dense model directoryを指定して起動すると、
`/v1` 配下でOpenAI Chat Completions APIを提供する。

```bash
./build/phaseshift-server \
    --model-dir /path/to/Qwen3.5-4B \
    --port 8000
```

起動すると次のエンドポイントが利用できる。

```text
http://127.0.0.1:8000/v1
```

対応範囲は次のとおり。

* `/v1/chat/completions`（非stream / stream、tools / tool result round-trip）
* `/v1/responses`（非stream / stream、tools / function_call_output round-trip）
* `/v1/models`

embeddings、vision、audio、MCP、server-side tool実行、認証、TLS、複数modelは対象外。
Responses APIのbackground / storage / WebSocketは対象外。

## 起動

```bash
./build/phaseshift-server \
    --model-dir /path/to/Qwen3.5-4B
```

model loadはserver lifetime中に1回だけ行われる。requestごとにcompute processを
起動しない。

起動完了時、stderrに以下が表示される。

```text
PhaseShift Server

Model:        phaseshift (Qwen3.5-4B)
Endpoint:     http://127.0.0.1:8000/v1
Context:      32768
Concurrency:  4
KV dtype:     bf16
KV capacity:  auto
Prefix cache: 16384 tokens / 8 entries
Default output: 4096

Capabilities:
  Chat:         yes
  Streaming:    yes
  Tool calling: yes
  Responses API:yes
  Structured:   fail-closed (Chat + Responses)
  Reasoning:    yes (opt-in)

Compatible:
  OpenAI SDK
  OpenCode
  Codex CLI

Ready.
```

defaultで32K context、concurrency 4、GPU prefix cache、4096 output budgetが有効である。
OpenCode / Codex CLIは追加のserver flagなしで利用できる。

debug / repro用には保守的profileを明示指定できる。

```bash
./build/phaseshift-server \
    --model-dir /path/to/Qwen3.5-4B \
    --max-concurrent-requests 1 \
    --prefix-cache-capacity-tokens 0
```

## options

```text
--model-dir PATH       Qwen3.5 model directory (required)
--host HOST            bind address (default 127.0.0.1)
--port N               bind port (default 8000)
--max-seq-len N        context window (default 32768)
--arena-gib N          GPU arena size in GiB (default 16)
--device N             GPU device (default 0)
--kv-cache-dtype TYPE  bf16 | fp8_e4m3 | psq4 | psq8 (default bf16)
--page-tokens N        paged KV page size in tokens (default 16)
--max-concurrent-requests N  max in-flight requests (default 4)
--kv-cache-capacity-tokens N physical KV token capacity (0 = one context)
--prefix-cache-capacity-tokens N GPU prefix cache KV token capacity (0 = disabled, default 16384)
--prefix-cache-max-entries N max cached prefixes / GDN snapshots (default 8)
--default-max-output-tokens N output cap when the request specifies none (default 4096)
--dflash2-model-dir PATH  DFlash2 draft model directory (speculative decoding を有効化)
--dflash2-drafts N     speculative draft tokens per round (default 7)
--compute PATH         phaseshift-compute path (default: next to this script)
--model-name NAME      served model name (default phaseshift)
--verify-weights       verify quantized payload CRC32 on load
--startup-timeout N    readiness wait timeout in seconds (default 120)
```

不正な値は起動時に拒否される。`--max-concurrent-requests` と
`--default-max-output-tokens` は1以上、prefix cache有効時の `--prefix-cache-max-entries` は
1以上でなければならず、違反時はエラーメッセージを表示して終了する。

`--kv-cache-dtype psq4` / `psq8` は `head_dim == 256` を要求する。prefix cache と
併用でき、cache pool も同じ KV dtype で作られる。

## DFlash2 speculative decoding

`--dflash2-model-dir` を指定すると DFlash2 を有効化する。詳細は
[../developer/dflash2.md](../developer/dflash2.md) を参照。

指定時の挙動:

- `--prefix-cache-capacity-tokens 0` を強制する（target prefix cache は非対応）。
- `--max-concurrent-requests 1` を強制する。
- Structured Output（GBNF / json_schema）と tool calling は**利用できない**。
  該当する request は gRPC `INVALID_ARGUMENT` で拒否する。
- `temperature > 0` の request は拒否する（greedy のみ）。

起動ログの `Capabilities` はこれらを反映して表示する。

```bash
./build/phaseshift-server \
    --model-dir models/Jackrong/Qwopus3.8-27B-Flash-V2-PSQ \
    --port 8001 \
    --device 1 \
    --arena-gib 26 \
    --max-seq-len 512 \
    --kv-cache-dtype psq4 \
    --max-concurrent-requests 1 \
    --dflash2-model-dir models/z-lab/Qwen3.8-27B-DFlash2-PSQ
```

developer option:

```text
--localai-binary PATH  LocalAI runtime override
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

stream chunkを連結した文字列は、同じrequestの非stream結果と一致する。

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
    "temperature": 0
  }'
```

応答はOpenAI形式の `tool_calls` になる。

```json
{
  "choices": [
    {
      "finish_reason": "tool_calls",
      "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
          {
            "index": 0,
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

PhaseShift Serverはtoolを実行しない。実際のtool実行はclient側（OpenCode等）が行う。
clientがtool結果を `role: "tool"` messageとして返すと、PhaseShiftはそれを読んで
次のassistant応答を生成する。

`tool_choice` は `auto` / `none` / `required` / named function指定を扱う。tool call数と
function nameはQwen3.5のStructural Tag制約でdecode時に保証される。`none` はtool定義を
modelへ渡さず、tool callを生成させない。詳細は
[Strict Function Tool Calling](#strict-function-tool-calling)を参照。

streaming中のtool callにも対応する。OpenAI形式の `delta.tool_calls` として返り、
model固有の `<tool_call>` 等のmarkupはclientへ漏れない。stream chunkから再構成した
tool callは、同じrequestの非stream結果と意味的に一致する。

### concurrency と cancel

`--max-concurrent-requests N` は最大in-flight request数であり、同時に1 executor
stepでschedulerが扱える最大request数でもある。これはN requestが常にActiveになる
ことを意味しない。physical KV capacity（`--kv-cache-capacity-tokens`）が不足する
場合、一部のrequestはQueuedとなり、KV BankerがActive完了後にadmissionする。
`--kv-cache-capacity-tokens 0`（default）は1 context分（`max_seq_len + 1`）を意味する。
defaultは `--max-concurrent-requests 4` であり、Continuous Batchingが有効である。

stream clientがconnectionを切断すると、in-flight generationはcancelされる
（`finish_reason="cancelled"`）。cancelはrequest-specificであり、他のin-flight
requestのKV / GDN / token streamには影響しない。

### Responses API

`/v1/responses` はOpenAI Responses API互換のsurfaceを提供する。

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "phaseshift",
    "input": "Reply with exactly: hello",
    "stream": false
  }'
```

function tool:

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "phaseshift",
    "input": "大阪の天気をget_weatherで調べて",
    "tools": [
      {
        "type": "function",
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    ],
    "tool_choice": "auto"
  }'
```

応答の `output` に `type=function_call` が返る。`call_id` / `name` / `arguments`
（valid JSON）をclientで使い、tool実行後に `function_call_output` として返す。

```json
{
  "input": [
    {"role": "user", "content": "大阪の天気をget_weatherで調べて"},
    {"type": "function_call", "call_id": "fc_x", "name": "get_weather",
     "arguments": "{\"city\": \"大阪\"}"},
    {"type": "function_call_output", "call_id": "fc_x",
     "output": "{\"temperature\":22,\"unit\":\"celsius\"}"}
  ]
}
```

`stream=true` では `response.created` / `response.output_text.delta` /
`response.completed`（tool callでは `response.output_item.done` のfunction_call）
が流れる。`max_output_tokens` は生成の上限として適用される。

Responses APIの `store=false` は受け付ける。`background` / `previous_response_id` /
WebSocket Responses / response storage は対象外。

## Reasoning (thinking)

Qwen3.5のthinkingをopt-inで利用できる。requestで `reasoning_effort` を指定すると、modelの
`<think>` ブロックがreasoningとして分離され、最終回答はcontentに入る。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "phaseshift",
    "messages": [{"role": "user", "content": "..."}],
    "reasoning_effort": "high"
  }'
```

- 値は `none` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max`。`none` はthinking off、
  それ以外はon。
- 未指定はdefault offであり、thinkingを有効化しない。
- Qwen3.5ではeffortごとのtoken budgetを保証しない。`minimal`/`low`/`medium`/`high`/`xhigh`/`max`
  はすべて「thinkingを有効化する」という意味だけを持つ。PhaseShift独自のbudget変換は行わない。
- 未知の値はHTTP 400。silentにon/offしない。

Responses APIでは `reasoning: {"effort": "high"}` を指定する。

output:

- Chat非stream: `choices[0].message.reasoning` にthinking、`message.content` に最終回答。
- Chat stream: `delta.reasoning` と `delta.content`。thinkingが先に流れ、途中でcontentから
  reasoningへ戻らない。
- Responses: `output` 先頭に `type: "reasoning"` itemが付き、その後にmessage itemが続く。

thinking tokenも `max_tokens` / `max_output_tokens`（既定4096）を共有する。thinkingが長いと同じ
予算で最終回答が短くなる。reasoningのdelimiter（`<think>` / `</think>`）自体はclientへ漏れない。

reasoningはtools / structured outputと併用できる。`reasoning_effort` が `none` 以外でも、tools、`response_format`、
あるいはその両方を指定したrequestは通常どおり動作する。thinkingの後にtool callまたはconstrained
final outputが生成される。tools / structured constraintはthinking終了後に適用される。

## Structured Output

### Chat Structured Output

Chat Completionsの `response_format` を利用できる。

- `{"type":"json_object"}`: 任意のJSONを強制する。
- `{"type":"json_schema","json_schema":{...,"schema":{...}}}`: schemaから変換された
  grammarを強制する。

schemaはGBNFへ変換され、decode制約として適用される。grammar違反tokenはsampling不能に
なるため、形式を「お願い」するのではなく物理的に強制する。streamingでもfinal textは
grammar準拠である。

保証範囲は次のとおり。

- Grammar enforcement: strict。有効なGBNFが生成された場合、PhaseShiftはそのgrammarを
  decode時に厳密に守る。
- Chat schema transport / conversion: fail-closed。schemaをGBNFへ変換できない場合、
  requestはHTTP 400で失敗し、生成を開始しない。silentにunconstrained生成へ落ちない。
- Full JSON Schema semantics: 保証しない。converterが理解するJSON Schema subsetのみを
  対象とする。任意のJSON Schemaが必ず変換成功することは保証しない。

unsupportedなschemaはHTTP 400として返る。エラーbodyはOpenAI互換のJSON errorである。

### Responses Structured Output

Open Responsesの `text.format` を利用できる。

```json
{
  "model": "phaseshift",
  "input": "Answer with a JSON object.",
  "text": {
    "format": {
      "type": "json_schema",
      "name": "result",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
        "required": ["answer"],
        "additionalProperties": false
      }
    }
  }
}
```

`text.format` がcanonicalなsurfaceである。legacy aliasの `text_format` も受け付けるが
推奨しない。両方を同時に指定したrequestはHTTP 400で失敗する。

- `{"type":"text"}`: unconstrained。通常のtext Responsesと同じ。
- `{"type":"json_object"}`: 任意のJSONを強制する。
- `{"type":"json_schema",...}`: schemaから変換されたgrammarを強制する。

保証範囲はChatと同じである。conversion failureはHTTP 400であり、unconstrained生成へは
fallbackしない。structured `text.format` と `tools` は同時指定できる。詳細は
[Structured Output + Tools](#structured-output--tools)を参照。`text` と `tools` の同時指定は
通常のtool callingとして扱う。

応答の `text.format.type` はrequestで解決されたtypeをechoする。

### Strict Function Tool Calling

Chat CompletionsとResponses APIのfunction tool callingは、Qwen3.5のStructural Tag制約で
function name、tool call数、arguments schemaをdecode時に保証する。

`strict: true` のfunctionはargumentsがdecode制約される。`strict` 省略または `false` のfunctionは
従来どおりbest-effortであり、function nameのみを制約する。

`tool_choice` の挙動:

- `"auto"`: plain textまたはtool call(s)。toolを呼ばないことは正しい。複数callは
  `parallel_tool_calls` に従う。
- `"none"`: 制約なしのplain generation。
- `"required"`: 1 call以上。`parallel_tool_calls=false` ならちょうど1 call。
- named `{"type":"function","function":{"name":"..."}}` (Chat) /
  `{"type":"function","name":"..."}` (Responses): 指定functionをちょうど1 call。

`parallel_tool_calls` のdefaultは `true` である。

strict toolのparametersはOpenAI strict contractを満たす必要がある。各object schemaで
`additionalProperties: false` を要求し、`properties` の全keyが `required` に含まれることを
要求する。optional fieldはpropertyを `required` に含めたうえで `["string","null"]` のように
nullableにする。違反したrequestはHTTP 400で失敗する。

`strict:true` で `parameters` を省略した場合は空引数functionとして扱う。

constraint compile failure、unknown named function、duplicate function name、不正な
`tool_choice` はすべてHTTP 400であり、generationを開始しない。

### Structured Output + Tools

`response_format`（Chat）/ `text.format`（Responses）のstructured outputと `tools` を
同時指定できる。decode制約は1つのStructural Tagへ合成され、outcomeは `tool_choice` ごとに
固定される。

- `"none"`: structured textのみ。toolsはpromptへ渡さない。
- `"auto"`: structured text または pure tool call(s)。
- `"required"`: tool call(s)のみ。
- named: 指定toolのみ。

tool branchが選ばれた場合、assistant contentは空で `tool_calls` のみを返す。unstructuredな
leading textは生成されない。`parallel_tool_calls` はtool branch内で適用される
（falseなら最大1 call、trueなら複数可）。

autoでの推奨flow:

```text
request: structured format + tools + tool_choice=auto
    ↓
tool_call
    ↓
tool result を返す
    ↓
request: 同じstructured format + tools + tool_choice=none
    ↓
final structured response
```

通常のtool request（structured outputなし）の挙動は変わらない。

## Prefix Cache

GPU-resident の Agent prefix cache は server では default で有効である。multi-turn で共通
prefix を持つ request は、前回の prompt-boundary snapshot を D2D で restore し、残りの
prefill だけを実行する。KV / GDN payload は Host へ退避しない。詳細は
[../developer/prefix_cache.md](../developer/prefix_cache.md) を参照。

```bash
phaseshift-server --model-dir models/Qwen3.5-4B
```

default は 16384 token / 8 entry である。低 VRAM や benchmark 用途では無効化できる。

```bash
phaseshift-server \
  --model-dir models/Qwen3.5-4B \
  --prefix-cache-capacity-tokens 0
```

- `--prefix-cache-capacity-tokens N`: GPU 上に保持する prefix KV token capacity。
  `0` で完全 disabled（default 16384）。
- `--prefix-cache-max-entries N`: cache できる prefix / GDN snapshot 数（default 8）。
  `capacity > 0` で `max_entries = 0` は invalid である。

server は `add_generation_prompt=False` の会話履歴を turn-stable な prompt boundary として
使い、通常の Chat multi-turn、tool round-trip、Responses function-call output で hit する。
boundary が full prompt の exact prefix でない場合は boundary を使わず通常 generation を続ける。

prompt boundary まで prefill 済みなら、その後 generation が cancel されても snapshot は保持
される。boundary 到達前の未完成 state は cache しない。

lookup は exact longest prefix、重複は統合、eviction は LRU である。

**VRAM**: prefix cache は active KV capacity（`--kv-cache-capacity-tokens`）とは別に
arena を消費する。有効にすると予約量が増え、arena に入らない設定は起動失敗する。

Agent の推奨 flow では、前 turn の assistant 出力（tool call を含む）を次 turn の
messages に含めることで prefix が一致する。

## エージェントからの利用

agent frontendはOpenAI互換clientとして接続する。ユーザーが起動するのは
`phaseshift-server` だけである。

agent promptはtool schema・repository instructions・conversation・file contentsを含む。
defaultの `--max-seq-len 32768` で基本的に不足しないため、通常は追加指定不要である。
より長い会話や大きなrepositoryでは必要に応じて `--max-seq-len` を上げる。

```bash
./build/phaseshift-server \
    --model-dir /path/to/Qwen3.5-4B
```

### OpenCode

OpenCodeのproject config（`opencode.jsonc`）にOpenAI-compatible providerを追加する。

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
      "models": {"phaseshift": {"name": "PhaseShift"}}
    }
  }
}
```

`PHASESHIFT_API_KEY` は任意のdummy値でよい（PhaseShiftは認証を行わない）。
modelは `phaseshift/phaseshift` を選ぶ。

### Codex CLI

`~/.codex/config.toml` にproviderを追加する。

```toml
model = "phaseshift"
model_provider = "phaseshift"
model_reasoning_effort = "none"

[model_providers.phaseshift]
name = "PhaseShift"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
request_max_retries = 0
stream_max_retries = 0
```

Codex CLIはResponses API（HTTP SSE）を使用する。WebSocketは使用しない。
PhaseShiftはユーザーのglobal configを自動編集しない。

## runtime

`phaseshift-server` は PhaseShift 向けの bundled canonical LocalAI runtime を必要とする。
release bundle 同梱の runtime を使う場合は追加作業は不要である。custom runtime を build
する場合は [build.md](build.md) を参照する。runtime は `--localai-binary` または
`PHASESHIFT_LOCALAI_BINARY` でも指定できる。

Structured Output を fail-closed で扱えない runtime（素の LocalAI v4.10.0 や必要な
PhaseShift 側処理を含まない runtime）は起動時に拒否され、`Ready.` を表示せずに終了する。

server / backend / compute の責務境界は [../developer/architecture.md](../developer/architecture.md)
を参照する。ユーザーは LocalAI、gRPC backend、`phaseshift-compute` を個別に起動する必要は
ない。

## エラー挙動

- 起動: 不正な option 値、incompatible な LocalAI runtime、model directory 不在、
  `phaseshift-compute` 不在は、stderr へ理由を表示して non-zero で終了する。
- request: 不正または unsupported な structured schema、未知の `reasoning_effort`、
  strict tool contract 違反、`text.format` と `text_format` の同時指定、constraint compile
  failure、未知の named function、不正な `tool_choice` は HTTP 400 であり、生成を開始しない。
- context: prompt が `--max-seq-len` を超える場合、および `max_tokens` が残り context を
  超える場合は request error となる。
- cancel: stream client の切断などで cancel された generation は `finish_reason="cancelled"`
  となる。
- 対象外の機能（embeddings / vision / audio / WebSocket Responses 等）に対応する endpoint
  は提供しない。

## テスト

server / LocalAI / backend / compute の E2E は `tests/server/` の Python suite で検証する。
model と LocalAI runtime が必要であり、`PHASESHIFT_LOCALAI_BINARY` に利用可能な runtime を
指定する。

```bash
python3 tests/server/run_server_regression.py --all
python3 tests/server/run_server_regression.py --group reasoning --group reasoning-tools
```

`tests/server/run_server_regression.py` が canonical な server regression runner である。
non-zero return（prerequisite 不足を含む）は失敗として扱う。
`--list` で group と test の一覧を表示できる。suite の詳細は
[../developer/testing.md](../developer/testing.md) を参照する。
