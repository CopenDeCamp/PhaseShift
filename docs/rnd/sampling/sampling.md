> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# Target LM sampling

baseline SHA: `3424ef47`（main、sampling 実装前）

## 変更

| 種別 | ファイル |
| --- | --- |
| public contract | `include/phaseshift/models/qwen35/runtime/sampling_params.h` / `.cpp` |
| device contract | `include|src/phaseshift/runtime/batch/device_batch_context.{h,hip}` |
| kernel | `include|src/.../kernels/optimized/{sampling_common.h,stochastic_sampling.h,stochastic_sampling.hip}` |
| greedy kernel | `include|src/.../kernels/optimized/sampling.{h,hip}`（sample flag を sampling params へ） |
| correctness | `src/.../kernels/correctness/{detail/output.inc,detail/dispatch_env.h,model_dispatch_correctness.hip}` |
| runtime | `scheduled_batch` / `schedule_plan` / `runtime_request` / `token_budget_scheduler` / `continuous_batcher` / `executor` / `program_executor` / `sampling_dispatch` / `sampling_selector` / `mtp_executor` / `spec_decoder` |
| apps | `phaseshift-compute`（`--temperature --top-p --top-k --seed`）、`phaseshift-cli`、`phaseshift-bench sampling --variant stochastic` |
| test | `test_qwen35_sampling_params` / `test_qwen35_sampling_rng` / `test_stochastic_sampling` |

`sample_greedy` は削除し、`DeviceRequestDescriptor::sampling`（`DeviceSamplingParams`）へ
一本化した。descriptor は 32 → 64 byte。

## required acceptance

| | |
| --- | --- |
| before | 97 |
| after | **100 / 100 PASS** |

追加した required test は 3 件（params / RNG / stochastic sampling）。

## greedy regression

device 0、`phaseshift-bench sampling`、vocab=248320。

| variant | before (p50) | after (p50) |
| --- | ---: | ---: |
| partitioned out=1 P=64 | 10.08us / 10.13us | 10.11us / 10.11us |
| single out=1 | 175.93us | 175.47us |

変化は noise 範囲。`temperature == 0` は従来の partitioned argmax をそのまま通り、
stochastic kernel には入らない。

## stochastic latency

device 0、`phaseshift-bench sampling --variant stochastic`、vocab=248320、
`--logit-pattern random`（logits は [-20, 20] の一様分布 = ほぼ平坦）、
warmup 200 / samples 100 / launches 50。

| outputs | T | top_p | top_k | p50 | p95 | mean | avg attempts |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 | 1.0 | 0 | 596.75us | 649.21us | 603.49us | 1.00 |
| 1 | 0.8 | 0.95 | 0 | 586.93us | 617.50us | 591.94us | 1.00 |
| 1 | 0.7 | 0.90 | 0 | 588.71us | 626.24us | 594.74us | 1.00 |
| 1 | 0.7 | 0.90 | 50 | 3747.64us | 3767.76us | 3745.94us | 12.00 |
| 8 | 1.0 | 1.0 | 0 | 636.13us | 668.07us | 640.79us | 1.00 |
| 8 | 0.8 | 0.95 | 0 | 622.85us | 666.79us | 632.83us | 1.00 |
| 8 | 0.7 | 0.90 | 0 | 966.15us | 976.80us | 965.97us | 1.12 |
| 8 | 0.7 | 0.90 | 50 | 63862.72us | 64016.20us | 63776.72us | 101.88 |

### 読み方

- stochastic kernel は仕様どおり **1 workgroup = 1 output row（256 threads）** の
  correctness-first 実装。vocab 全体を 1 workgroup で走査するため、1 row あたり
  約 600us（max / Z / Gumbel / mass の 4 scan）。既存の `single` argmax variant
  （1 workgroup、175us/scan）と同じ特性で、partitioned greedy（10us）とは比較対象が違う。
- `top_k=50` が極端に遅いのは、bench の logits が平坦なため top-50 の mass が
  約 0.008 しかなく、rejection の accept 確率がそれだけ低いため。試行回数は
  幾何分布になり、out=8 の平均は 101.9 回になった。実 LM の分布はもっと尖っており、
  `top_p=0.9` 程度では attempts ≈ 1 になる。
- stochastic path には hard performance gate を置かない（仕様どおり）。将来の最適化は
  scan の partition 化が候補。

## correctness / optimized

`test_stochastic_sampling` の case N で、同じ logits と同じ sampling params を
correctness dispatch と optimized kernel の両方へ通し、sampled token が一致することを
確認している（`sampling_sample_row` を共有しているため bit-exact）。

## E2E

### 4B（BF16、oracle 比較）

`tests/e2e/fixtures/qwen35_4b_oracle.json` の `raw_compute`（input_ids 14 token、
`max_new_tokens=3`）で比較。

| 実行 | GENERATED_IDS |
| --- | --- |
| default greedy | `220,19,11` |
| `--temperature 0 --top-p 0.9 --seed 123` | `220,19,11` |
| oracle | `220,19,11` |

greedy は oracle と完全一致。`--top-p` / `--seed` を与えても temperature 0 では
結果が変わらない。

### 4B / 4B-PSQ / 27B-PSQ（default prompt、32 token）

`--model-dir models/<model>`、既定 prompt（248041, 77091）。

| model | default == explicit temp=0 | stochastic 決定性（同 seed 2 回） | stochastic vs greedy |
| --- | --- | --- | --- |
| Qwen3.5-4B | YES | YES | different |
| Qwen3.5-4B-PSQ | YES | YES | different |
| Qwen3.8-27B-PSQ | YES | YES | different |

`--temperature 0.7 --top-p 0.9 --top-k 50 --seed 123` を 2 回実行した
`GENERATED_IDS` は 3 model すべてで一致した（batch 構成に依存しない）。

生成例（27B-PSQ、32 token）:

```text
greedy    : 72,60,271,550,220,97036,128158,271,248041,96009,735,220,735,220,...
stochastic: 20053,60,271,248041,153223,5205,118716,15303,32317,15303,96527,...
```

### phaseshift-cli

`--temperature` / `--top-p` / `--top-k` / `--seed` を `phaseshift-compute` へそのまま渡す。

prompt `What is 2+2?`、32 token、device 0。

| 実行 | reply |
| --- | --- |
| greedy | `The answer to **2 + 2** is **4**.` |
| `T=0.7 p=0.9 k=50 seed=123` | `The answer is **4**.\n\nIn standard arithmetic, adding two numbers together gives you their sum. Since $2 + 2 = 4$, the result` |
| `T=0.7 p=0.9 k=50 seed=7` | `The answer is **4**.` |

CLI 経由の `GENERATED_IDS` は同じ条件の `phaseshift-compute` 直接実行と一致する。
`seed=123` の 2 回実行も一致した。

`--top-p 0.9` と `--top-k 50` を同時に与えた場合と `--top-k` なしの場合で
`GENERATED_IDS` が一致した。nucleus が top-50 の部分集合になっているため。

### E2E suite

`python3 tests/e2e/test_apps_qwen35_4b.py --suite inference ...` は
`help:bench-fused-linear` で停止する。`phaseshift-bench fused-linear --help` は
**変更前バイナリでも rc=2**（bench に `fused-linear` / `attention-prep` /
`gdn-prepare` / `gdn-norm-gate` という subcommand が無い）で、sampling とは無関係な
既存の破損。そのため suite 全体は実行できず、上記の直接比較で回帰を確認した。

## Regression

| 対象 | 結果 |
| --- | --- |
| full build | PASS |
| required acceptance | 100 / 100 PASS |
| MTP required（primitives / kv_cache / state / spec_verify / spec_transaction） | PASS |
| 4B / 4B-PSQ / 27B-PSQ default greedy | PASS（4B は oracle 一致） |
| greedy latency | 変化なし |

## Known issues

- stochastic kernel は 1 workgroup = 1 output row。1 row あたり約 600us で、
  partitioned greedy（10us）より遅い。scan の partition 化が将来の最適化候補。
- `top_k` を小さくする、または `top_p` を小さくすると rejection の試行回数が増える。
  分布が平坦なときは特に顕著（bench の最悪ケースで attempts 101.9）。
- top-k と top-p は full softmax 上で独立に定義した積集合。HF の
  「top-k で切ってから再正規化した分布に top-p」とは厳密には一致しない
  （片方だけ使う場合は標準定義と一致する）。`docs/developer/sampling.md` 参照。
- `tests/e2e/test_apps_qwen35_4b.py` の bench subcommand 一覧が実際の
  `phaseshift-bench` と食い違っており、変更前から suite が完走しない。
- `tests/kernels/optimized/test_output_gather.hip` が `launch_output_gather` へ
  `num_outputs` を渡しておらず、ヘッダ変更で再コンパイルされて初めてリンク/コンパイル
  エラーになった（変更前からの破損）。本作業で引数を追加して修正した。

## 停止

Sampling Gate PASS。min-p / typical-p / repetition penalty / DFlash CandidateSelector /
stochastic speculative decoding / DFlash persistent KV / Gate5.1 再開には進まない。
