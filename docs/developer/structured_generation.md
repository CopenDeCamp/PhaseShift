# Grammar Constrained Generation

PhaseShiftはGBNF grammarによるtoken制約decodeを行う。

```
GBNF grammar
    ↓
XGrammar v0.2.5 (C++ static library, vendored)
    ↓
token constraint matcher (request単位)
    ↓
compressed token bitmask (32 token / uint32 word)
    ↓
HIP sampling kernel
    ↓
grammar違反tokenはsampling不能
```

## 境界

`phaseshift-compute` が理解するのはtoken IDs、sampling、GBNF grammarまでである。
OpenAI API schema、JSON Schema、tool schemaはserver / control-planeの責務であり、
computeへは入れない。PhaseShift独自のJSON Schema parserやResponses parserも持たない。

## XGrammar

- native constraint engine: XGrammar `v0.2.5`（tag commit `2ea71da`）を `vendor/xgrammar/` に固定vendorする。
- native C++ static libraryとしてbuildする。Python bindings / CUDA kernelは使わない。
- matcherはCPUでbitmaskを生成する。PhaseShift側のsampling kernelへbitmaskを統合する。
- TokenizerInfo producer: Python `xgrammar 0.2.5.post1`。tokenizer vocabularyをmodel load時に
  1回生成し、sidecar JSONへ書き出す。computeは起動時に1回だけ読む。
- serialized TokenizerInfo の native / Python 互換は acceptance test で検証する。
- `phaseshift-server` では Python xgrammar を required dependency とする。無い場合は
  model load が明確な error で失敗する。`phaseshift-compute` 単体は Python 非依存。

XGrammar v0.2.5 の public header には、`NamedGrammar` が不完全型の段階で
`std::vector<NamedGrammar>{}` を default argument として構築する C++20/libc++ 互換問題がある。
PhaseShift は upstream XGrammar main の `EmptyNamedGrammars()` 修正を最小 backport として
vendor patch で保持する。

```
vendor/xgrammar/patches/0001-cxx20-named-grammar-incomplete-type.patch
vendor/xgrammar/README.phaseshift.md
```

これにより PhaseShift native code は例外なく C++20 で統一される。C++17 の source-level
exception は存在しない。`VERSION` は upstream base (`v0.2.5` / `2ea71da`) のまま変更しない。

## generation stop token

generation 停止判定の source of truth は model config の `eos_token_id` である。
`tokenizer.eos_token_id` metadata は chat template の別トークン（`<|im_end|>` 等）を指すことが
あり、両者は一致しない場合がある。chat turn の途中で現れる tokenizer EOS で生成を止めず、
generation stop token のみで停止する。XGrammar の stop token にも generation stop token を使い、
tokenizer metadata の EOS は使わない。両者が異なることは検証済みの正常状態である。

## request単位の状態

- `TokenConstraintCompiler` はmodel lifetimeで1個。compiled grammarはbounded cacheで共有する。
- `TokenConstraintState`（matcher）はgeneration requestごとに1個。共有しない。
- `RuntimeRequest.constraint` がnullptrならunconstrained。
- grammar compile failureはfail-closedであり、unconstrained generationへは絶対にfallbackしない。

## sampling

- greedy: allowed tokenのみを対象にmax logitを選ぶ。
- stochastic: temperature / top-k / top-pの全段階でdisallowed tokenを除外する。
- allowed tokenが0個の場合はEOSへfallbackせずrequest errorとする。
- sampled tokenはmatcherへacceptする。accept失敗はrequest errorとする。
- EOSはgrammar完了後に許可し、既存の `FinishReason::Eos` へ到達させる。

## GPU統合

- unconstrained batchではmask生成もmask H2Dも行わない。
- constrained output rowが存在するstepだけ、compressed maskを1回のbatched H2Dで転送する。
- maskはoutput row単位で `[rows][ceil(vocab_size/32)]` のuint32。
- sampling kernelは `DeviceSamplingParams.reserved[0]` のconstraint flagとmask base pointerを読む。
- 追加D2H、追加hipStreamSynchronize、requestごとのHIP streamは無い。
- constrained requestはMTPを使わない。DFlash2 speculative decode ではconstraintが
  target verify の各行に適用され、drafter の proposalにも現在の grammar 状態を適用する
  （[dflash2.md](dflash2.md) を参照）。
- constrained batchはHIP Graph replay/captureを行わない（dynamic execution path）。
  unconstrained batchは既存のHIP Graph pathをそのまま使う。constrained request後も
  unconstrained requestでGraphを再利用できる。constrained batchのHIP Graph対応は未実装である。

## 計測

`PHASESHIFT_CONSTRAINT_TRACE=1` でstderrへ次を出す。production defaultは無効。

```
CONSTRAINT_COMPILE_US=...
CONSTRAINT_CACHE_HIT=0/1
CONSTRAINT_MASK_WORDS=...
CONSTRAINT_ROWS=...
MASK_BYTES=...
```

## 保証範囲

```
Grammar enforcement:                    strict
Chat schema transport/conversion:       fail-closed
Responses text.format transport:        fail-closed
Full JSON Schema semantic coverage:     not guaranteed
```

Chat Completionsの `response_format`（`json_object` / `json_schema`）と、Responsesの
canonicalな `text.format` はLocalAIがGBNFへ変換し、backendの `request.Grammar` として
到達する。**LocalAIが有効なGBNFを生成してbackendへ渡した場合、PhaseShiftはそのgrammarを
decode時に厳密に強制する。**

Chat CompletionsとResponsesのどちらも、structured transportはfail-closedである。

```
LocalAIがgrammarへ変換できるschema
    -> grammarをbackendへ渡し、decode時に強制する

LocalAIがgrammarへ変換できないschema / 不正なresponse_format / text.format
    -> HTTP 400。backendとcomputeを呼ばずgenerationを開始しない
```

Responsesの `text_format` はlegacy aliasであり、`text.format` と同時指定したrequestは
HTTP 400とする。structured `text.format` と `tools` の同時指定は1つのStructural Tagへ
合成して扱う。

unconstrained fallbackは行わない。以下は引き続き保証しない。

- 任意のJSON Schemaが必ずGBNFへ変換成功すること（LocalAI converter subsetのみ）。
- 変換できたgrammarがJSON Schemaの意味論を完全に表現すること。

Responses parser / prompt injection / unconstrained fallbackを追加しない。

## Composite constraint (structured text + tools)

structured outputとtool callingを同時指定した場合、PhaseShift backendは2つの独立した
constraintをcomputeへ送らない。代わりに1つのStructural Tagへ合成する。

```
response Grammar
      +
tool format
      ↓
one Structural Tag (or)
      ↓
one TokenConstraintState / one token mask
```

- `tool_choice="none"`: response Grammarのみ。
- `tool_choice="required"` / named: tool-only constraint。
- `tool_choice="auto"`: `or(response Grammar, pure tool branch)`。

tool branchは`tags_with_separator`であり、unstructured leading contentを許可しない。
つまりoutcomeは「structured text」または「pure tool call(s)」のどちらかである。
通常のtool request（structured outputなし）はbest-effort behaviorを維持する。

computeはcomposition semanticsを知らない。backendが`grammar`または`structural_tag`の
片方だけを送る既存contractを維持する。

reasoningを有効にした場合も同じ1つのStructural Tagへ合成する。reasoning envelopeを外側に
1回だけ被せ、内側は上記final formatをそのまま再利用する。

```
sequence(
  tag(begin="", content=any_text, end="</think>\n\n"),
  existing composite final format
)
```

`tool_choice="none"`は`envelope + grammar`、`required`/namedは`envelope + tool-only`、
`auto`は`envelope + or(grammar, tool)`となる。reasoningの詳細は
[reasoning.md](reasoning.md)を参照。

LocalAI response Grammarのrule順序は安定化されており、同一schema・同一tools・同一policyの
requestは同一Structural Tag bytesを生成する（native constraint cacheがhitする）。

## LocalAI runtime

Chat / Responses structured fail-closedはfrontend transportの契約であり、LocalAI runtime側の
patch seriesを必要とする。

```
LocalAI v4.10.0
  + vendor/localai/patches/0001-chat-response-format-fail-closed.patch
  + vendor/localai/patches/0002-responses-structured-transport.patch
  + vendor/localai/patches/0003-tool-policy-transport.patch
  + vendor/localai/patches/0004-structured-tools-composition-transport.patch
  + vendor/localai/patches/0005-responses-reasoning-transport.patch
  + vendor/localai/patches/0006-responses-reasoning-tools-stream.patch
```

`phaseshift-server` は起動時にfail-closed capability probeを実行する。ChatとResponsesの
両方がfail-closedでないruntime、および `--localai-binary` で指定されたvanilla v4.10.0は
起動時にrejectされる。patched runtimeは `tools/build_localai_runtime.sh` でbuildする。
このbuildには Go >= 1.26（加えて make、git、curl、unzip、network access）が必要である。

server が実際に受け付ける surface は [../user/server.md](../user/server.md) を参照。
