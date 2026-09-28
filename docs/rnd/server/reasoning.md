# Reasoning (thinking) R&D record

> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は docs/developer/、現在の性能は docs/perf/ を参照）

この文書は、PhaseShift Server に Qwen3.5 の thinking（reasoning）を導入したときの
検証記録である。user manual にはユーザー操作として必要な範囲だけを残し、そこから
分離した R&D 由来の知見をここへ置く。

現在の reasoning contract は [../../developer/reasoning.md](../../developer/reasoning.md)、
現在の性能値は [../../perf/README.md](../../perf/README.md) を参照する。

## transport の検証

- reasoning は request の `reasoning_effort` で opt-in する。未指定は default off であり、
  reasoning 導入前（thinking off）の serving 挙動をそのまま維持する。
- LocalAI v4.10.0 は effort 値を検証・drop せず、`reasoning_effort` をそのまま backend へ
  渡す。このため `xhigh` / `max` は backend 側に受け口を追加するだけで成立し、LocalAI 側の
  追加処理は不要だった。
- 未知の effort 値は HTTP 400 とし、silent に on/off へ落とさない方針を維持した。
- Responses API の `reasoning.effort` は `reasoning_effort` と同じ値域を共有する。

## 挙動の知見

- effort level は「thinking を有効化する」という意味だけを持ち、level ごとの token budget は
  与えない。`minimal` / `low` / `medium` / `high` / `xhigh` / `max` の差は生成 budget に
  反映されない。
- thinking token は final answer と同じ `max_tokens` / `max_output_tokens` budget を共有する。
  thinking が長いと、その分だけ final answer が短くなる。
- reasoning の delimiter（`<think>` / `</think>`）は client へ漏らさない。`</think>` が
  現れない場合は全生成を reasoning として扱い、content を空にする。
- reasoning は tools / structured output と併用できる。constraint は thinking 終了後に適用
  され、reasoning 中は free-form 生成である。併用時に constraint を silent に外さない。

## 関連

- 現在の contract: [../../developer/reasoning.md](../../developer/reasoning.md)
- structured output の contract: [../../developer/structured_generation.md](../../developer/structured_generation.md)
- user 向け操作: [../../user/server.md](../../user/server.md)
