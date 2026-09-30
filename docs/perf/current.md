# Current performance baseline

この文書は PhaseShift の最新の qualified performance baseline を記録する。
ここに載る値は下記 measurement revision での計測結果であり、それ以後に追加された
未計測の変更に対する性能を保証しない。

測定手順の原則は [methodology.md](methodology.md) を正本とする。

## 計測断面

| 項目 | 値 |
| --- | --- |
| model | `models/Qwen3.8-27B-PSQ`（main performance target） |
| draft model | `models/Qwen3.8-27B-DFlash2-PSQ` |
| GPU | AMD Radeon AI PRO R9700 (gfx1201) / ROCm / HIP / Linux |
| measurement revision | commit `2b90b202` |
| date | 2026-09-28 |
| build | Release / gfx1201 / benchmarks ON / optional tests OFF |
| build directory | `build-gfx1201` |
| HIP graph | OFF（`PHASESHIFT_HIP_GRAPH` 既定） |
| arena | bench 24 GiB / compute 26 GiB |
| page-tokens | 16 |
| keep-alive | 有効（executor 既定。`PHASESHIFT_KEEPALIVE=0` で無効化） |
| device | 1 |

build condition で変わるのは `PHASESHIFT_HIP_GRAPH` の ON/OFF のみである。
Fusion / DeepFusion は一時的に build option が撤去されている。
本 baseline は既定値である HIP graph OFF で計測する。

model は `--preset psq --backend hip` で再生成できる
（27B は `--scope text-only` が必要。手順は [../user/quantizer.md](../user/quantizer.md)）。

## Target prefill（pp2048）

`--prompt-tokens 2048 --mode forward`、外側 3 rep × 内側 runs 3 / warmup 1 の中央値。
単位は gpu_tokens_per_sec。

| rep | gpu_ms_median | tok/s |
| --- | ---: | ---: |
| 1 | 876.79 | 2335.79 |
| 2 | 879.85 | 2327.66 |
| 3 | 881.08 | 2324.43 |
| **中央値** | | **2327.66** |

run 間のばらつきは 0.49%。

## Target non-spec decode（tg128・ctx2048）

`--context 2048 --tokens 128 --mode greedy --compute-logits 1 --warmup 2`、3 rep 中央値。
単位は gpu_tokens_per_sec。

| rep | tok/s |
| --- | ---: |
| 1 | 27.60 |
| 2 | 27.58 |
| 3 | 27.56 |
| **中央値** | **27.58** |

run 間のばらつきは 0.15%。
`GREEDY_TOKEN_SUM=2446188` は全 rep で一致する。

## DFlash2 speculative decode（prose K7 256）

`--dflash2-drafts 7`、256 new tokens、3 rep 中央値。
単位は DECODE_TOKENS_PER_SEC。

| rep | tok/s | DFLASH2_VERIFY_GPU_MS |
| --- | ---: | ---: |
| 1 | 65.10 | 3360.58 |
| 2 | 65.09 | 3362.66 |
| 3 | 65.11 | 3360.77 |
| **中央値** | **65.10** | **3360.77** |

run 間のばらつきは 0.03%。

> このセクションは Gate3（[../rnd/spec_decode/ngram_tail_gate3.md](../rnd/spec_decode/ngram_tail_gate3.md)、
> target lm_head proxy の停止と NgramTail option の追加）**以前**の revision で
> 測った値である。DFlash2 有効時は lm_head proxy が既定で停止するため、
> acceptance / tok/s はこの表から変わる。現行既定は未再計測のため、この表は
> 現行既定の性能を保証しない。正本 model が計測環境にない場合は数値を書き換えない。

## 正しさ契約

速度の数値は生成結果が一致している rep からのみ採用する。

- `phaseshift-bench tg` — `GREEDY_TOKEN_SUM=2446188` が全 rep で一致。
- `phaseshift-compute`（DFlash2）— `GENERATED_IDS` の sha1 先頭 12 桁
  `47aebe55d048`、`DFLASH2_ROUNDS=84`、`DFLASH2_ACCEPTED_DRAFTS=171` が全 rep で一致。
- `phaseshift-bench pp` — `--mode forward` のため生成結果を持たない。

`GENERATED_IDS` の sha1 は `GENERATED_IDS=` 行の値部分（行末改行を含む）に対する
ものである。次のコマンドで再現できる。

```sh
grep '^GENERATED_IDS=' <output> | sed 's/^GENERATED_IDS=//' | sha1sum
```

## 計測コマンド

```sh
# pp2048
./build-gfx1201/phaseshift-bench pp --model-dir models/Qwen3.8-27B-PSQ \
  --prompt-tokens 2048 --mode forward --page-tokens 16 --arena-gib 24 \
  --runs 3 --warmup 1 --device 1

# tg128
./build-gfx1201/phaseshift-bench tg --model-dir models/Qwen3.8-27B-PSQ --mode greedy \
  --context 2048 --tokens 128 --prefill-chunk 2048 --compute-logits 1 \
  --page-tokens 16 --arena-gib 24 --warmup 2 --device 1

# DFlash2 prose K7 256
./build-gfx1201/phaseshift-compute --model-dir models/Qwen3.8-27B-PSQ \
  --dflash2-model-dir models/Qwen3.8-27B-DFlash2-PSQ --dflash2-drafts 7 \
  --input-ids-file build/dflash2-gate51/real/real_prose/prompt_ids.txt \
  --max-new-tokens 256 --max-seq-len 4096 --arena-gib 26 --temperature 0 \
  --dflash2-stats 1 --device 1
```

`prompt_ids.txt` は `tools/reference/export_qwen38_dflash2_gate51_real.py --mode prompts`
の tokenize 結果（`prompt.txt`）を whitespace 区切りに並べたものである。

## Caveat

- GPU0 は別 workload と共有していることがある。再現時は `--device 1` 等で
  隔離された GPU を使う。
- 初回プロセスは GPU クロックのランプアップで pp が大きく遅く出るため、
  warmup で除外している。
- 絶対値は single-run ではぶれる。run 間安定性（中央値）で判定する。
- ここに無い性能値が必要な場合は、勝手に推測せず
  [methodology.md](methodology.md) の手順で新規計測し、日付・revision・commit とともに
  追記する。
