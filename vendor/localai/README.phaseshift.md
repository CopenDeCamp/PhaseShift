# PhaseShift LocalAI vendor metadata

Base:

```text
repo:   https://github.com/mudler/LocalAI
tag:    v4.10.0
commit: 7ad0cbf259f0c7bf9920fe2438fc3630ecd6c672
```

PhaseShift patch series (apply in order):

```text
patches/0001-chat-response-format-fail-closed.patch
  Chat Completions response_format fail-closed

patches/0002-responses-structured-transport.patch
  Open Responses text.format fail-closed structured transport
  (depends on 0001)

patches/0003-tool-policy-transport.patch
  tool policy transport: parallel_tool_calls metadata, tool-policy
  transport marker, explicit Responses strict preservation, and
  Chat structured response_format + tools rejection
  (depends on 0001 and 0002)

patches/0004-structured-tools-composition-transport.patch
  structured text + tools transport: composition metadata marker,
  response Grammar preservation, LocalAI function grammar overwrite
  prevention, and removal of the structured + tools rejection
  (depends on 0001, 0002 and 0003)

patches/0005-responses-reasoning-transport.patch
  Responses reasoning transport: forward reasoning.effort to the backend
  like the Chat Completions path, prefer backend ChatDelta reasoning when
  building the non-stream reasoning item, and let ChatDelta reasoning open
  the streaming reasoning item
  (depends on 0001, 0002, 0003 and 0004)

patches/0006-responses-reasoning-tools-stream.patch
  Responses tool-stream reasoning ordering: let backend ChatDelta reasoning
  open the reasoning item on the tool-calling stream path too, so the
  reasoning item precedes the function_call item
  (depends on 0001, 0002, 0003, 0004 and 0005)
```

Modified behavior:

```text
json_schema grammar conversion failure returns HTTP 400
  (instead of logging and continuing unconstrained)

unsupported / unknown response_format type returns HTTP 400
malformed response_format payload returns HTTP 400
empty generated grammar returns HTTP 400

Open Responses text.format routes through the same fail-closed grammar helper
  unsupported / malformed text.format returns HTTP 400
  text.format and text_format together return HTTP 400
  structured text.format combined with tools returns HTTP 400

tool policy transport (0003)
  parallel_tool_calls forwarded via reserved metadata
  reserved tool-policy transport marker always overwritten
  explicit Responses strict tools preserved
  Chat structured response_format combined with tools returns HTTP 400
  backend INVALID_ARGUMENT maps to HTTP 400 (stream: invalid_request error)

structured text + tools composition transport (0004)
  Chat structured response_format + tools accepted
  Responses structured text.format + tools accepted
  response Grammar preserved for the backend
  LocalAI function grammar does not overwrite the response Grammar
  reserved composition marker always overwritten
```

Unmodified:

```text
tool/function grammar body
backend.proto
gRPC protocol
sampling
model loading
```

`text.format` is the canonical Responses surface; `text_format` is kept only as
a legacy alias and is rejected when combined with `text.format`.

`VERSION` stays at the upstream base (`v4.10.0`). Patch information is tracked
here and in `patches/`, so the patch can be dropped when upgrading to an upstream
release that already rejects unconvertible response formats.

## Build

PhaseShift は LocalAI の source tree を vendor しない。patched runtime は
`tools/build_localai_runtime.sh` で LocalAI v4.10.0 の checkout から build する。

```bash
tools/build_localai_runtime.sh \
    --source-dir /path/to/LocalAI-v4.10.0 \
    --output build/local-ai
```

この build には Go >= 1.26、make、git、curl、unzip、および Go modules / protoc 取得用の
network access が必要である。LocalAI v4.10.0 の `go.mod` は `go 1.26.0` を要求する。
script は既定で `GOTOOLCHAIN=local` を使うため、installed Go が 1.26 未満の場合は
Go 1.26+ を導入するか `GOTOOLCHAIN=auto` を指定する。

`phaseshift-server` は起動時に fail-closed capability probe を実行し、patch の無い
LocalAI runtime を reject する。Chat Completions と Responses の両方で structured
output が fail-closed であることを要求する。加えて tool request では backend が
`phaseshift.tool_policy_transport` marker を要求するため、`0003` を持たない runtime の
tool request は runtime 側で reject される。structured output + tools は `0004` を
持たない runtime では HTTP 400 で fail-closed になる（安全側）。

つまり、単なる `LocalAI v4.10.0`、`0001`、`0002`、`0003` までしか適用していない
runtime では Gate 9B の composition capability を満たさない。
