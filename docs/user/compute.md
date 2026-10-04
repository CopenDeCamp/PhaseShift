# phaseshift-compute

単一request推論アプリ。greedy / temperature / top-k / top-p デコード。BF16 modelと量子化model
（`phaseshift_quantization.json` による自動検出）の両方に対応。

## options

```text
--model-dir PATH       BF16 or quantized model directory (auto-detected via
                       phaseshift_quantization.json) (required)
--input-ids-file PATH  whitespace separated prompt token ids
--max-new-tokens N     max tokens to generate (default 32)
--max-seq-len N        max sequence length (default 512)
--arena-gib N          arena capacity in GiB (default 16). Specifies the
                       logical/virtual arena capacity. Physical VRAM is
                       committed incrementally as allocations are made.
--page-tokens N        paged KV page size in tokens (default 16)
--device N             GPU device (default 0)
--kv-cache-dtype TYPE  bf16 | fp8_e4m3 | psq4 | psq8 (default bf16)
--verify-weights 0|1   verify quantized payload CRC32 on load (default 0)
--dump-logits PATH     append per-step sampled logits rows (raw f32) to PATH
--temperature F        sampling temperature (default 0 = greedy)
--top-p F              nucleus sampling threshold in (0, 1] (default 1)
--top-k N              keep the N highest logits (default 0 = disabled)
--seed N               sampling seed (default 0)
--dflash2-model-dir PATH  DFlash2 draft model directory（指定で DFlash mode）
--dflash2-drafts N     speculative draft tokens per round (default 7)
--dflash2-stats 0|1    print speculative decode statistics (default 1)
--serve-stdio          serve JSON Lines generation requests on stdin/stdout
--help                 show this help
```

`--model-dir` は必須。`--input-ids-file` なしでは既定prompt
（`248041, 77091`）を使用。

`--kv-cache-dtype psq4` / `psq8` は `head_dim == 256` を要求する。prefix cache と
併用でき、cache pool も同じ KV dtype で作られる。

### DFlash2 speculative decoding

`--dflash2-model-dir` 指定時のみ DFlash 経路に入る。未指定時は既存の
`ContinuousBatcher` 経路を変更しない。詳細は
[../developer/dflash2.md](../developer/dflash2.md) を参照。

併用できない option（いずれも model load 前に exit 2）:

- `--dump-logits`
- `--dflash2-drafts` が `block_size - 1` を超える

constrained な LM head の候補展開最適化（`PHASESHIFT_CONSTRAINT_LM_HEAD_EXACT`）は
既定 0（無効）である。有効時も制約の正しさは full path の sampling filter が担保する。

`--serve-stdio` と `--kv-cache-dtype psq4` は併用できる。serve mode では
`grammar` / `structural_tag` / `prefix_cache_checkpoint_position` / `temperature > 0`
を DFlash2 経路でも受け付ける。constraint は target verify の各行に適用され、
generation は grammar 準拠である。prefix cache の checkpoint は prompt boundary
（prompt 全体）で保存する。

## serve-stdio

`--serve-stdio` はserver backendから利用されるengine modeである。modelとruntimeを
一度だけ初期化し、stdinからrequestを読んでstdoutへeventを返す。stdoutはJSON Lines
protocol専用で、`MODEL_*`・debug・load statusは出力しない（diagnosticはstderr）。
このmodeのstdoutは1行1 JSON objectである。

request:

```json
{"op":"generate","request_id":1,"input_ids":[1,2,3],"max_new_tokens":32,"temperature":0.0,"top_p":1.0,"top_k":0,"seed":0}
```

event:

```json
{"event":"token","request_id":1,"token_id":1234}
{"event":"done","request_id":1,"generated_ids":[10,20,30],"finish_reason":"max_tokens"}
{"event":"error","request_id":1,"message":"..."}
```

`token` eventは生成token順に送られ、その連結は `done.generated_ids` とexact一致する。

ping / shutdown:

```json
{"op":"ping"}      -> {"event":"pong"}
{"op":"shutdown"}  -> {"event":"shutdown"}
```

single-shot modeの出力契約（`GENERATED_IDS` 等）は変更されない。

## sampling

`--temperature 0`（既定）は従来どおり greedy で、結果は現行と完全に同一。
sampling は GPU 上で完結し、host は logits を読まない。logits の D2H も
vocab の host sort も行わない。

| 指定 | 挙動 |
| --- | --- |
| `--temperature 0` | deterministic greedy。`--top-p` / `--top-k` は無視 |
| `--temperature 0.7 --top-p 1` | temperature softmax からの categorical sample |
| `--temperature 0.7 --top-p 0.9` | nucleus sampling |
| `--temperature 0.7 --top-p 0.9 --top-k 50` | top-k と nucleus の積集合 |
| `--temperature 0.7 --top-p 1 --top-k 1` | 最上位 token のみ |

`--top-k` は `0` で無効。`--top-k` を小さくする、または `--top-p` を小さくすると
rejection の試行回数が増える。

同じ prompt・同じ `--seed`・同じ sampling 設定なら `GENERATED_IDS` は一致する。
batch 構成や prefix cache の状態には依存しない。

```bash
./build/phaseshift-compute \
  --model-dir /path/to/Qwen3.5-4B \
  --input-ids-file prompt_ids.txt \
  --max-new-tokens 32 \
  --temperature 0.7 --top-p 0.9 --top-k 50 --seed 42
```

## 出力契約

stdoutは `KEY=VALUE` 行。推論結果は以下:

```text
PROMPT_TOKENS=<prompt token数>
GENERATED_TOKENS=<生成token数>
KV_CACHE_DTYPE=bf16 | fp8_e4m3
KV_POOL_RESERVED_BYTES=<KV pool予約バイト数>
GENERATED_IDS=<id1>,<id2>,...
```

`GENERATED_IDS` が生成tokenのexact契約。

`--dump-logits` 指定時は追加で:

```text
LOGITS_VOCAB=<vocab size>
LOGITS_STEP_ROWS=<stepごとのlogits行数（カンマ区切り）>
LOGITS_TOTAL_ROWS=<合計行数>
```

logitsはraw f32、1行 = 1 token step（prefill行が先頭）。
`row k` は `GENERATED_IDS[k]` と対応。

## 例

```bash
./build/phaseshift-compute \
  --model-dir /path/to/Qwen3.5-4B \
  --input-ids-file prompt_ids.txt \
  --max-new-tokens 32 \
  --kv-cache-dtype fp8_e4m3
```
