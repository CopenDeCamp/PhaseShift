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
| measurement revision | commit `a33b0bb1` |
| date | 2026-09-30 |
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
MTP linear 8本は PSQ4、norm 7本は BF16。

## Target prefill（pp2048）

`--prompt-tokens 2048 --mode forward`、外側 3 rep × 内側 runs 3 / warmup 1 の中央値。
単位は gpu_tokens_per_sec。

| rep | gpu_ms_median | tok/s |
| --- | ---: | ---: |
| 1 | 888.45 | 2305.13 |
| 2 | 889.81 | 2301.60 |
| 3 | 889.16 | 2303.29 |
| **中央値** | | **2303.29** |

run 間のばらつきは 0.15%。

## Target non-spec decode（tg128・ctx2048）

`--context 2048 --tokens 128 --mode greedy --compute-logits 1 --warmup 2`、3 rep 中央値。
単位は gpu_tokens_per_sec。

| rep | tok/s |
| --- | ---: |
| 1 | 27.51 |
| 2 | 27.51 |
| 3 | 27.47 |
| **中央値** | **27.51** |

run 間のばらつきは 0.15%。
`GREEDY_TOKEN_SUM=2446188` は全 rep で一致する。

## GPU-MCU decode backend（pp2048・tg128）

`--decode-backend gpu-mcu` を有効にした revision で 2026-10-06 に計測した。
Host と交互3rep（methodology §2）、内側の runs / warmup / shape は上表と同じ。
GPU[1] を隔離して使用し、Host 値は同一セッションで再現した
（pp 2309.31、tg 27.56、いずれも既存 baseline との差 +0.3% 以内）。
この表は「計測断面」の measurement revision とは異なる revision の値である。

### pp2048（gpu_tokens_per_sec、外側3rep 中央値）

| backend | rep1 | rep2 | rep3 | 中央値 | Host 比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| host | 2333.70 | 2307.84 | 2309.31 | 2309.31 | 100% |
| gpu-mcu | 1923.67 | 1917.44 | 1907.72 | **1917.44** | **83.0%** |

### tg128 / ctx2048（gpu_tokens_per_sec、外側3rep 中央値）

| backend | rep1 | rep2 | rep3 | 中央値 | Host 比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| host | 27.60 | 27.56 | 27.56 | 27.56 | 100% |
| gpu-mcu | 18.92 | 18.91 | 18.92 | **18.92** | **68.7%** |

run 間のばらつきはいずれも 0.1% 未満。

GPU-MCU の値は PSQ8 / PSQ4 の Prefill2D・embedding・BF16 Wmma・multi-row
entrypoint を通る plan compile と実行を含む。正しさは
`test_gpu_mcu_production_decode` の Host との byte 比較と AQL 単体検証で担保する。
tg の Host 比が pp より低いのは、decode の dispatch 数あたりの GPU 時間が短く
MCU submission overhead（`queue_ahead_depth=4` と packet ごとの doorbell）の
比率が大きくなるためである。この構造は
[../rnd/gpu_mcu/full_transformer_static_region.md](../rnd/gpu_mcu/full_transformer_static_region.md)
の region starvation 計測と一致する。

## DFlash2 speculative decode（prose K7 256）

`--dflash2-drafts 7`、256 new tokens、3 rep 中央値。
単位は DECODE_TOKENS_PER_SEC。

| rep | tok/s | DFLASH2_VERIFY_GPU_MS |
| --- | ---: | ---: |
| 1 | 63.15 | 3479.58 |
| 2 | 63.14 | 3479.88 |
| 3 | 63.15 | 3479.66 |
| **中央値** | **63.15** | **3479.66** |

run 間のばらつきは 0.02%。

前回 baseline（commit `2b90b202`、65.10 tok/s）から **−3.00%**。

うち約 2.4% は verify の `LmHeadCandidateProxy` が効かなくなったことによる。
proxy は int2 coarse の top-P 候補からのみ argmax を決め、精度を保証できず
greedy equivalence を破る（実例は [../rnd/mtp/mtp.md](../rnd/mtp/mtp.md) §10.1）。
`phaseshift-compute` は DFlash2 有効時に mode を `0` へ強制する
（[../developer/qwen35.md](../developer/qwen35.md) §7.1）ため、
この経路は exact のままである。速度より正しさを採用した。
`2b90b202` の proxy 実装を差し戻すと 65.20 tok/s が再現することを
計測で確認している（差の主因が proxy の代替である根拠）。

残る約 0.6% は原因を断定していない。ただし pp で同時刻交互計測により
**コード寄与ゼロ**を確認済み（下記 caveat）であり、これも同種の日時差と推定する。
DFlash2 自体は A/B を行っていないので推定にとどめる。

この計測は全 GPU idle の単独環境で行った。

> このセクションは Gate3（[../rnd/spec_decode/ngram_tail_gate3.md](../rnd/spec_decode/ngram_tail_gate3.md)、
> target lm_head proxy の停止と NgramTail option の追加）**以前**の revision で
> 測った値である。DFlash2 有効時は lm_head proxy が既定で停止するため、
> acceptance / tok/s はこの表から変わる。現行既定は未再計測のため、この表は
> 現行既定の性能を保証しない。正本 model が計測環境にない場合は数値を書き換えない。

## 正しさ契約

速度の数値は生成結果が一致している rep からのみ採用する。

- `phaseshift-bench tg` — `GREEDY_TOKEN_SUM=2446188` が全 rep で一致。
- `phaseshift-bench tg --decode-backend gpu-mcu` — 同じく `GREEDY_TOKEN_SUM=2446188`
  と `GREEDY_FIRST_TOKENS` が Host と全 rep で一致する（2026-10-06 計測）。
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

# pp2048 / tg128（gpu-mcu）— 上記に --decode-backend gpu-mcu を付ける
./build-gfx1201/phaseshift-bench pp --model-dir models/Qwen3.8-27B-PSQ \
  --prompt-tokens 2048 --mode forward --page-tokens 16 --arena-gib 24 \
  --runs 3 --warmup 1 --device 1 --decode-backend gpu-mcu

./build-gfx1201/phaseshift-bench tg --model-dir models/Qwen3.8-27B-PSQ --mode greedy \
  --context 2048 --tokens 128 --prefill-chunk 2048 --compute-logits 1 \
  --page-tokens 16 --arena-gib 24 --warmup 2 --device 1 --decode-backend gpu-mcu

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
- **絶対値は計測日の GPU 状態にも依存する。** 同一セッションで `2b90b202`（元の
  baseline）と現行 main+3 を交互に pp2048 計測したところ、中央値は
  **2300.9 / 2301.0 でコード差はゼロ**だった。一方 2026-09-28 計測の 2327.66 と
  2026-09-30 の 2300.9 は **−1.15%** 差があり、その間 GPU[1] の junction 温度は
  31°C → 57°C に上がっている。約 1% 程度の振れはコードではなく蓄熱・クロック
  による日時差と考えること。日付違いの値を直接比較しないこと。
- ここに無い性能値が必要な場合は、勝手に推測せず
  [methodology.md](methodology.md) の手順で新規計測し、日付・revision・commit とともに
  追記する。
