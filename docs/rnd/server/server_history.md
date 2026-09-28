# PhaseShift Server Gate History

> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

PhaseShift Server の capability がどの Gate で構築され、どのような採否判断があったかの R&D 記録で
ある。current contract ではない。現在の contract は `docs/developer/`、利用方法は
`docs/user/server.md` を参照する。

canonical branch は `feat/phaseshift-server`。最終状態は
**Feature Complete after Gate 11C (Server Closure / Compatibility Audit)** である。

## Gate 系譜

### Gate 8A: native grammar-constrained decode
- GBNF engine、compute grammar protocol、constrained sampling、concurrent constraints を実装。
- Chat structured transport と streaming は PASS。Responses structured transport は当時
  BLOCKED_EXTERNAL、strict schema fail-closed は未実装。
- 完了 commit は a2548c2d。

### Gate 8B: Chat structured fail-closed transport
- invalid `json_schema` / unknown `response_format` type を HTTP 400 とし、backend と compute を
  呼ばない。SSE 未開始で拒否する。
- LocalAI patch 0001 を追加。起動時 capability probe は patched を PASS、vanilla を REJECT。

### Gate 8C: Responses structured fail-closed transport
- canonical `text.format` を採用し、legacy `text_format` alias を許容、同時指定は HTTP 400。
- LocalAI patch 0002 を追加。probe は chat-only / vanilla を REJECT。

### Gate 9A: strict function tool calling
- `strict:true` の arguments を `qwen_xml_parameter` で schema 制約し、non-strict は
  best-effort（function name のみ）。
- `tool_choice` auto / none / required / named、`parallel_tool_calls` transport を実装。
- LocalAI patch 0003 を追加。0003 なし runtime の tool request は handshake で拒否。

### Gate 9B: structured text + tool composition
- structured output と tools を 1 つの Structural Tag へ合成。grammar と structural_tag は排他。
- LocalAI patch 0004 を追加。

### Gate 10A: GPU-resident prefix cache foundation
- 専用 GPU KV pool / GDN pool を持ち、save / restore は D2D のみ。Host payload 退避なし。
- cache dtype は bf16 と fp8_e4m3。terminal checkpoint（Eos / MaxNewTokens）のみで default off。

### Gate 10B: Agent-ready defaults / prompt-boundary prefix cache
- server default を context 32768 / concurrency 4 / KV bf16 / prefix cache 16384 tokens 8 entries /
  output budget 4096 とした。
- `add_generation_prompt=False` の会話履歴を prompt boundary とし、full prompt の exact prefix で
  あるときだけ使う。boundary 到達後は cancel されても snapshot を保持する。

### Gate 11A: reasoning / thinking foundation
- LocalAI metadata の `enable_thinking` / `reasoning_effort` を transport し、precedence と effort
  alias を定めた。
- assistant reasoning history を保持し、Chat / Responses の reasoning 出力を実装。
- 当初は reasoning + tools / structured を HTTP 400 とした（Gate 11B で解除）。

### Gate 11B: interleaved reasoning + tools / constraints
- reasoning envelope を 1 つの Structural Tag として final format の前に付ける。
- compute へ渡す制約は grammar / structural_tag の片方のみ。parser は reasoning を final parser
  から分離する。

### Gate 11C: closure / compatibility audit
- `reasoning_effort` に `xhigh` / `max` を追加した。LocalAI patch 0007 は不要で、backend が
  `REASONING_EFFORTS` に追加するだけで成立する。
- canonical regression runner `tests/server/run_server_regression.py` と手動 closure soak を整備。
- この時点で Server API の機能追加を通常の開発優先順位から外し、GPU-MCU / quantization /
  runtime・kernel performance 側へ主開発を戻すと判断した。

## 採否判断

- XGrammar v0.2.5 の C++20/libc++ 互換 patch を最小 backport として保持する。PhaseShift native は
  例外なく C++20 で build する。
- runtime EOS と tokenizer EOS の divergence は正常状態として許可する。現在の contract は
  `docs/developer/structured_generation.md` の generation stop token を参照。
- deferred constraint activation（reasoning 開始時の constraint 遅延 activation）は未実装。
  Gate 11C の測定で normalized overhead が negligible と確認されたため、当時は正当化されなかった。
- Feature Complete 後の「Server へ戻る条件」は actual client incompatibility / actual correctness
  bug / actual measurable bottleneck / explicitly required API feature のいずれかが実際に発生した
  場合のみとした。

## Gate 11C 測定: reasoning + constraint overhead

Structural Tag matcher が reasoning 期間も active であることによる overhead を、同じ server
process で測定した（最適化はしていない）。

環境: RDNA4 R9700 / gfx1201、temperature 0、`reasoning_effort=high`、`max_tokens=512`、
warmup 3、measured samples 10、server default profile（context 32768 / concurrency 4 /
prefix cache 16384 tokens / 8 entries）。測定 script は
`tools/bench_server_reasoning_constraints.py`。

| case | TTFT | total wall | output | ms / completion token |
| --- | --- | --- | --- | --- |
| plain | 155.3 ms | 13865.2 ms | reasoning 2006 chars / 512 tok | 27.08 |
| structured | 155.3 ms | 13908.3 ms | reasoning 1994 chars / 512 tok | 27.16 |
| strict-tool | 154.2 ms | 1846.8 ms | tool call 2 / 157 chars | n/a |

classification:

```text
structured vs plain   ms/token +0.3%, TTFT 0.0%
    -> negligible (< 5%)
strict-tool           TTFT -0.7%, wall 1847 ms
    -> output が 13x 短いため char 正規化は比較不能。overhead の signal は無い。
```

`plain` と `structured` は同じ 512 token を decode しており、constraint 有無だけが異なる。
したがって matcher が reasoning 期間 active であることの normalized overhead は計測上
negligible である。この測定のために GPU runtime / kernel / instrumentation は変更していない。

## EOS oracle 証拠

- model config eos（generation stop token）: 248044 `<|endoftext|>`
- tokenizer eos metadata: 248046 `<|im_end|>`
- oracle terminal token: 248044
- transformers oracle（`chat_single`）の generated_ids は `[3793, 248046, 198, 248044]` であり、
  248046 の後も生成が継続し 248044 で停止する。したがって model config eos と tokenizer eos の
  divergence は検証済みの正常状態である。
