# phaseshift-quantizer

offline quantizationツール。サブコマンド:

```text
quantize   produce a self-contained PhaseShift quantized safetensors model
verify     verify a self-contained quantized model directory
kld        evaluate KLD between BF16 and quantized model
imatrix    collect calibration activation statistics into a PSIM file
ppl        evaluate teacher-forced perplexity on a corpus
```

`phaseshift-quantizer <command> --help` で各コマンドのoptions。

## quantize

```text
phaseshift-quantizer quantize --input <model-dir> --output <quant-dir> \
  --preset psq --backend cpu|hip \
  [--scope text-only] [--imatrix <file.psim>] [--imatrix-policy strict|best-effort] \
  [--max-shard-size 4GiB] [--overwrite] [--no-verify]
```

出力はself-containedな量子化safetensors model directory。
`phaseshift-compute` が `--model-dir` として直接読める。

`--imatrix` は `imatrix` サブコマンドが生成したPSIM fileをiMatrix inputとして使用する。

## verify

```text
phaseshift-quantizer verify <quant-dir>
```

self-contained量子化model directoryを検証。

## kld

```text
phaseshift-quantizer kld --input <dir> --tokens <file> --report <path> [options]
  --input <dir>            original BF16 model directory
  --bundle <dir>           quantized model directory
  --tokens <file>          PSKLDTOK corpus file
  --report <path>          JSON report output path
  --window N               window size (default 512)
  --stride N               stride (default 512)
  --max-eval-tokens N      max tokens to evaluate (default all)
  --cache-dir DIR          cache directory (default /tmp/phaseshift-kld)
  --keep-shadow            keep shadow model directory
  --keep-logit-cache       keep baseline logit cache
  --self                   self-test mode (no --bundle required)
  --batch-positions N      batch positions (default 16)
  --arena-gib N            arena size in GiB (default 20)
```

## imatrix

校正corpusをmodelに流し、量子化対象siteのactivation統計（per-channel Σx²）を収集して
PSIM fileへ書き出す。収集対象は layer別に full-attention qkv入力 / attention o_proj入力 /
GDN projection入力 / GDN out_proj入力 / MLP gate+up入力 / MLP down入力、および lm_head入力
（最終norm出力）の7 site。

```text
phaseshift-quantizer imatrix --input <bf16-dir> --tokens <file> --output <path> [options]
  --input <dir>                fingerprint算出用の元BF16 model directory
  --model-dir <dir>            実行するmodel directory (default: --input)
  --tokens <file>              PSKLDTOK corpus file
  --output <path>              .psim output path
  --window N                   window長 (default 512)
  --stride N                   stride (default 512)
  --max-calibration-tokens N   校正token上限 (default 全token)
  --flush-windows N            device→host flush間隔 (window数, default 16)
  --arena-gib N                arena size in GiB (default 24)
  --device N                   HIP device index (default 0)
```

corpusは `tools/quantization/build_phaseshift_imatrix_corpus.py` で生成し、
`tools/quantization/prepare_kld_corpus.py` でPSKLDTOKへ変換する。

校正modelは `--model-dir` に量子化済みmodelを指定できる。27B級はBF16重みがVRAMに
収まらないため、`--input` にBF16 directory（fingerprint用）、`--model-dir` に量子化model
（実行用）を指定する。

## ppl

corpusに対する teacher-forced NLL / perplexity を評価する。量子化レシピ間の相対比較用
（絶対値はcorpus依存）。1 tokenずつdecodeし、各位置のlogitsから次tokenのNLLを積む。

```text
phaseshift-quantizer ppl --model-dir <dir> --tokens <file> [options]
  --model-dir <dir>  評価するmodel directory
  --tokens <file>    PSKLDTOK corpus file
  --max-tokens N     評価token数 (default 2048)
  --arena-gib N      arena size in GiB (default 24)
  --device N         HIP device index (default 0)
```
