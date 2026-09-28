# Reasoning / Thinking

この文書は PhaseShift Server の Qwen3.5 thinking transport を記述する。

## 原則

thinking は通常の token generation である。したがって PhaseShift が追加するのは、

```
request policy
chat-template switch
output parsing
API transport
```

だけである。GPU runtime・KV layout・GDN layout・scheduler・executor・sampling kernel は変更しない。

## transport contract

LocalAI v4.10.0 は external backend へ `PredictOptions.Metadata` を渡す。probe で確認した keys:

```
enable_thinking        "true" / "false"
reasoning_effort       none | minimal | low | medium | high | xhigh | max | (未知値)
chat_template_kwargs   JSON blob（enable_thinking / reasoning_effort を内包）
```

backend は次で決定する。

```
explicit enable_thinking
    >
reasoning_effort
    >
server default（off）
```

- `enable_thinking` が explicit ならそれを使う。
- それ以外で `reasoning_effort` があれば `none` → off、その他 → on。
- どちらも無ければ off。
- `chat_template_kwargs` は standalone key が無い場合の fallback source としてのみ読む。
- `reasoning_effort` が未知値なら HTTP 400。silent fallback しない。

LocalAI v4.10.0 の Chat Completions は `ApplyReasoningEffort` を通して上記 metadata を作る。
Responses は素のままでは reasoning effort を落とすため、patch `0005` で同じ処理を追加する。

`xhigh` / `max` の transport:

```
LocalAI は値の検証・drop を行わず reasoning_effort をそのまま backend へ渡す
enable_thinking は付かない（ApplyReasoningEffort の switch に対応 case が無い）
    → metadata 例: {"reasoning_effort": "xhigh"}（enable_thinking なし）
backend は reasoning_effort だけから thinking ON を判定できる
    → REASONING_EFFORTS に xhigh / max を追加するだけで成立
    → LocalAI patch 0007 は不要
```

`xhigh` / `max` も `none` 以外は `enable_thinking=true` とするだけで、PhaseShift 独自の
level 別 token budget は与えない。未知値は引き続き HTTP 400 である。

## rendering

`enable_thinking` は full prompt（`add_generation_prompt=True`）と stable prompt
（`add_generation_prompt=False`）の両方へ渡す。stable prompt は generation prompt tail を
含まないため thinking on/off で同一であり、prompt-boundary prefix cache の node は変わらない。
full prompt は tail が異なるため、cache は自然に別 prefix として扱う（metadata key を追加しない）。

Qwen3.5 template の挙動:

```
enable_thinking=false  →  ...<|im_start|>assistant\n<think>\n\n</think>\n\n
enable_thinking=true   →  ...<|im_start|>assistant\n<think>\n
```

thinking on では `<think>` opener が prefill されるため、generated token 列は

```
reasoning text
</think>
final answer
```

となる（opening tag は含まれない場合が多い）。parser は両方の形を扱う。

## history

assistant history の `reasoning_content` は保持して chat template へ渡す。Qwen3.5 template は
last user query より後の assistant についてのみ `<think>` を復元するため、通常の multi-turn
history では reasoning は prompt に再展開されない（template の semantics）。codec は
`reasoning_content` を content へ連結せず、捨てない。

## parsing

```
reasoning_content = </think> より前
content           = </think> より後
```

- `</think>` が無い場合は全生成を `reasoning_content` とし、`content` は空にする
  （thinking を content として漏らさない）。
- delimiter 自体はどちらにも含めない。
- reasoning disabled では parser を通さず、従来と exact な parsing を維持する。

streaming は毎 chunk で accumulated text を再分割する。`</think>` が decode chunk を跨ぐ場合に
備えて、`len("</think>") - 1` 文字を保留し、delimiter が確定するまで reasoning として emit
しない。retraction が起きても `_suffix_delta` が新規 suffix のみを返す。

## output transport

```
Chat non-stream   pb.ChatDelta(reasoning_content=..., content=...)
Chat stream       pb.ChatDelta(reasoning_content=delta, content=delta)  per chunk
```

LocalAI は chunk 単位の `ChatDelta` を読み、reasoning が現れた時点で backend の分類を
採用する（`preferAutoparser`）。Chat の wire field は `message.reasoning` / `delta.reasoning`
である（`reasoning_content` は入力 alias）。

Responses は patch `0005`（非 tool stream）と `0006`（tool stream）により `type: "reasoning"` item として出力する。

## Interleaved reasoning + constraints

reasoning と tool / structured constraint は 1 個の Structural Tag で合成する。GPU runtime は
変更しない。

concept:

```
prefilled <think>
    ↓
generated free-form reasoning      (any_text)
    ↓
</think>\n\n
    ↓
constrained final phase            (existing tool / structured format)
```

Structural Tag:

```
sequence(
  tag(begin="", content=any_text, end="</think>\n\n"),
  FINAL_FORMAT
)
```

- `begin=""` でよい。generation prompt が `<think>\n` を prefill 済みで、generated token 列は
  reasoning block の途中から始まる。
- `any_text` は `excludes` として enclosing tag の end を検出するため、`</think>\n\n` を飲み込まない。
- `FINAL_FORMAT` は既存 format をそのまま再利用する。structured なら `grammar`、required /
  named tool なら tool branch、structured + tools auto なら `or(grammar, tool)`。
- 新しい GBNF parser や XGrammar primitive は追加しない。

constraint matrix:

```
reasoning off                          従来と exact
reasoning on + plain                   constraint なし
reasoning on + tools (constraint 必要)  envelope + existing tool format
reasoning on + structured only         envelope + grammar format (grammar="" で送る)
reasoning on + structured + tools      envelope + existing composite final format
```

compute へ送る constraint は常に `grammar` か `structural_tag` の片方だけである。

structural tag matcher は generated token 1 から active になり、reasoning 期間も constrained
decode の mask path（CPU matcher、mask 生成、batched mask H2D）を通る。`additional H2D = 0`
とは主張しない。

## 非目標

```
reasoning 専用 token budget 強制
GPU-side reasoning phase switch
deferred constraint activation
新 sampling kernel / XGrammar 変更 / backend.proto 変更
```

reasoning と constraint の併用で制約を外す（silent constraint disable）ことはない。
