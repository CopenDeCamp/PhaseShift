# phaseshift-cli

チャットUI。Python実装で、token化はHF `AutoProcessor`
（`enable_thinking=False`）を使用する。
推論本体は `phaseshift-compute` をsubprocessで呼ぶ。

## options

```text
--model-dir MODEL_DIR      Qwen3.5-4B model directory (required)
--binary BINARY            phaseshift-compute path (default: next to this script)
--prompt PROMPT            single-shot prompt (omit for interactive mode)
--max-new-tokens N
--max-seq-len N
--arena-gib N
--device N
--dflash2-model-dir PATH   DFlash2 draft model directory (speculative decoding)
--dflash2-drafts N         speculative draft tokens per round (default 7)
--dflash2-stats 0|1        print speculative decode statistics (default 1)
-h, --help                 show this help message and exit
```

`--dflash2-model-dir` 指定時のみ DFlash2 経路に入る。greedy のみ
（`--temperature` は 0 固定）で、`--prefix-cache-capacity-tokens` とは併用できない。
spec decode の構成は `phaseshift-compute` の既定（`--dflash2-drafts 7`、
NgramTail 無効）を使う。NgramTail を有効化する場合は `phaseshift-compute` を直接起動し、
`--dflash2-ngram-tail 8 --dflash2-ngram-n 5` を指定する（cli は ngram option を転送しない）。
詳細は [../developer/dflash2.md](../developer/dflash2.md) を参照。

## 使い方

single-shot:

```bash
./build/phaseshift-cli --model-dir /path/to/Qwen3.5-4B --prompt "こんにちは"
```

interactive multi-turn:

```bash
./build/phaseshift-cli --model-dir /path/to/Qwen3.5-4B
```

`assistant: ` 行の繰り返しで応答が表示される。EOFで終了。
