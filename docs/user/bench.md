# phaseshift-bench

ベンチマークアプリ。サブコマンドはカテゴリごと:

```text
E2E:
  pp                 prefill benchmark
  tg                 decode / token-generation benchmark

Linear:
  activation-quantize A8 activation quantization micro-benchmark
  gemm               linear GEMV/GEMM micro-benchmark

Normalization:
  rmsnorm            RMSNorm micro-benchmark

Attention/KV:
  kv-append          KV append benchmark
  paged-attention    paged-attention benchmark

GDN:
  gdn-recurrence     GDN recurrence benchmark

Auxiliary:
  elementwise        Qwen elementwise benchmark
  embedding          embedding lookup benchmark
  sampling           greedy argmax sampling benchmark
```

`--model-dir` は pp / tg で必須。他は model 不要。

GEMV/GEMM are compute kinds. WMMA is an internal GEMM implementation
detail and is not a `--variant` value.

各サブコマンドの全 option は `phaseshift-bench <sub> --help` を参照。

## pp

```text
--prompt-tokens N            (default 128)
--mode forward|greedy        (default forward)
--device N                   (default 0)
--model-dir PATH             (required)
--page-tokens N              (default 16)
--arena-gib N                (default 6)
--runs N                     (default 3)
--warmup N                   (default 0)
--output PATH                (default none)
```

## tg

```text
--mode forward|greedy        (default forward)
--context N                  (default 128)
--tokens N                   (default 8)
--prefill-chunk N            (default 128)
--warmup N                   (default 0)
--compute-logits 0|1         (default 1)
--device N                   (default 0)
--model-dir PATH             (required)
--page-tokens N              (default 16)
--arena-gib N                (default 6)
--output PATH                (default none)
--mtp                        MTP speculative decode を実行 (greedy のみ)
--draft-k N                  verify 1 ラウンドあたりの MTP draft 数 (default 4)
--mtp-chain pre|post         draft step >=2 への hidden 入力 (default post)
```

greedy mode は `--compute-logits 1` が必要。

`--mtp` は MTP draft/verify/accept ループで decode を計測する
（`--mode greedy --context >= 1` と MTP 重みを伴う model が必要）。
report に `tokens_emitted` / `mtp_rounds` / `mtp_reruns` / `mtp_mean_accept`
を追加して出力する。per-token 指標は `tokens_emitted` 基準。

## activation-quantize

```text
--variant correctness|optimized   (default optimized)
```

A8 activation quantization (production correctness path vs optimized kernel)。

## gemm

```text
--dtype bf16|psq4|psq8            (default bf16)
--variant correctness|gemv|gemm|auto  (default auto)
--rows 1,16,32,64,128             (comma list, default 1,16,32,64,128)
--n N                            (default 5120)
--k N                            (default 5120)
--model-dir PATH                 (with --actual-qwen-shapes: read Qwen config)
--actual-qwen-shapes             (benchmark all deduped Qwen linear shapes)
--warmup N                       (default 100)
--samples N                      (default 50)
--launches N                     (default 20)
--seed N                         (default 1234)
--device N                       (default 0)
--check                          (verify selected path vs the correctness path)
--out PATH                       (append CSV rows)
```

`--variant`:

```text
correctness   production correctness path
gemv          force Small-M GEMV compute kind (selector bypass)
gemm          force Big-M GEMM compute kind (selector bypass)
auto          production linear selector (default)
```

`auto` の場合、選択された compute kind (correctness/gemv/gemm) を
`requested_variant` と `selected_variant` として stdout/CSV に出力する。

CSV 列:

```text
dtype,requested_variant,selected_variant,shape_name,rows,n,k,
warmup,samples,launches_per_sample,min_us,p50_us,p95_us,mean_us
```

## rmsnorm

```text
--variant correctness|optimized|auto   (default optimized)
```

## kv-append

```text
--variant correctness|optimized|auto   (default optimized)
```

## paged-attention

```text
--variant correctness|optimized|auto   (default optimized)
```

## gdn-recurrence

```text
--variant correctness|optimized   (default optimized)
```

## elementwise

```text
--variant correctness|optimized   (default optimized)
```

## embedding

```text
--variant correctness|optimized   (default optimized)
```

embedding lookup (bf16 / psq8)。

## sampling

```text
--variant correctness|single|partitioned   (default correctness)
```

greedy argmax sampling。

## 例

```bash
./build/phaseshift-bench pp --model-dir /path/to/Qwen3.5-4B --prompt-tokens 512
./build/phaseshift-bench tg --model-dir /path/to/Qwen3.5-4B --mode greedy --compute-logits 1
./build/phaseshift-bench gemm --check --rows 1,16,128
./build/phaseshift-bench gemm --dtype psq8 --variant auto --actual-qwen-shapes --model-dir /path/to/Qwen3.5-4B --out gemm.csv
```
