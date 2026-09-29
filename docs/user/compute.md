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
--decode-backend TYPE  host | gpu-mcu (default host)
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

`--decode-backend gpu-mcu` は model load 前に exit 2 で拒否される。
backend contract は `host` / `gpu-mcu` の2値で、GPU-MCU implementation はこの build に
含まれない。指定しても `host` へ fallback しない。

### DFlash2 speculative decoding

`--dflash2-model-dir` 指定時のみ DFlash 経路に入る。未指定時は既存の
`ContinuousBatcher` 経路を変更しない。詳細は
[../developer/dflash2.md](../developer/dflash2.md) を参照。

併用できない option（いずれも model load 前に exit 2）:

- `--temperature > 0`（greedy のみ）
- `--dump-logits`
- `--constraint-tokenizer-info`
- `--prefix-cache-capacity-tokens > 0`
- `--dflash2-drafts` が `block_size - 1` を超える

`--serve-stdio` と `--kv-cache-dtype psq4` は併用できる。serve mode では
`temperature > 0` / `grammar` / `structural_tag` / `prefix_cache_checkpoint_position`
を含む request を fail-closed で拒否する。

### 固定draft語彙の配置と明示有効化

固定語彙はopt-inであり、profileを配置しただけではDFlash2の既定動作を変えない。
`PHASESHIFT_DFLASH2_DRAFT_VOCAB=1`を指定すると、対応する標準profileを検証してINT2＋固定語彙を使う。
通常利用者はprofile配布物だけを用意し、SWE-chat等の生成元コーパスを取得する必要はない。
配置にはPython 3の標準ライブラリだけを使い、推論時のPython依存は追加しない。

Qwen3.8-27B用の[語彙profileを取得](https://github.com/jyohukuchan/PhaseShift/releases/download/draft-vocab-qwen38-v1/qwen38-draft-vocab-98304-v1.tar.gz)して展開する。
SHA-256は`b7bd94c9131c3ef3573c27c92a512b5463bc9676c57a1a70ac0223fc2f8df246`。
この配布物はcontributorのforkで提供する。

```sh
python3 tools/quantization/prepare_draft_vocab.py install \
  --bundle-dir /path/to/extracted-vocabulary-profile \
  --model-dir /path/to/target-PSQ-model
```

配置先は`--model-dir`のtarget側であり、`--dflash2-model-dir`ではない。
toolはmodel形状とtokenizerのtoken→ID対応を照合し、次の3ファイルを配置する。

- `dflash2-draft-vocab.u32`
- `dflash2-draft-vocab.json`
- `DRAFT_VOCAB_NOTICE.txt`

異なる既存内容を置き換える場合だけ`--overwrite`を指定する。同梱NOTICEは保持する。
未指定時はprofileの有無・内容によらずfull PSQ8 headを使い、標準profileを読まない。
明示有効化時にprofileがない、または正常だが別tokenizer/形状向けなら、理由をstderrへ示して
従来経路を使う。明示有効化時の部分配置、破損・不正なprofileはエラーになる。

有効化と比較の指定:

```sh
PHASESHIFT_DFLASH2_DRAFT_VOCAB=1 ./build-gfx1201/phaseshift-compute ...
PHASESHIFT_DFLASH2_INT2_HEAD=1 ./build-gfx1201/phaseshift-compute ...
PHASESHIFT_DFLASH2_INT2_HEAD=0 ./build-gfx1201/phaseshift-compute ...
```

順に固定語彙INT2、全語彙INT2、full PSQ8を選ぶ。
`INT2_HEAD=1`だけではprofile配置済みでも全語彙INT2を使う。
`INT2_HEAD=0/2`は標準profileの有効化より優先する。
明示的な独自語彙は`PHASESHIFT_DFLASH2_DRAFT_VOCAB_FILE`で指定でき、INT2経路を選ぶ。
この明示ファイルと`INT2_HEAD=0/2`または`DRAFT_VOCAB=0`の同時指定はエラーになる。

profile作成者向けの`pack`手順と契約は[開発者文書](../developer/dflash2.md#固定語彙profile)を参照する。

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
