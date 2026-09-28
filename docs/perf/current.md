# Current performance baseline

この文書は PhaseShift の最新の qualified performance baseline を記録する。
ここに載る値は下記 measurement revision での計測結果であり、それ以後に追加された
未計測の変更に対する性能を保証しない。

数値の出典は `docs/rnd/primitive_baseline.md` の 2026-09-23 追記
（measurement revision `ba153347`、device 1）である。

## 計測断面

| 項目 | 値 |
| --- | --- |
| model | `Qwen3.8-27B-PSQ`（main performance target） |
| GPU | AMD Radeon AI PRO R9700 (gfx1201) / ROCm / HIP / Linux |
| measurement revision | commit `ba153347` |
| date | 2026-09-23 |
| build | Release / gfx1201 / benchmarks ON / optional tests OFF |
| arena | bench 24 GiB / compute 26 GiB |
| page-tokens | 16 |
| keep-alive | 有効（executor 既定。`PHASESHIFT_KEEPALIVE=0` で無効化） |
| device | 1 |

4 build 条件（HIP graph ON/OFF × Fusion ON/OFF）の E2E 断面。

## historical measurement 注記

この節の「融合 ON」条件は、production から Fusion /
DeepFusion を撤去する以前に計測した **historical measurement** である。当時は
`-DPHASESHIFT_FUSION_PRESET=MAX` が展開する次の 10 flag を明示指定して configure して
いた。撤去後はこれらの build option は存在せず、同じ build を再現できない。
現在サポートされる build condition は HIP graph ON/OFF のみである。

```text
PHASESHIFT_FUSE_RESIDUAL_RMSNORM_QUANT
PHASESHIFT_FUSE_GDN_PREPARE
PHASESHIFT_FUSE_GDN_POST
PHASESHIFT_FUSE_ATTN_GATE_QUANT
PHASESHIFT_FUSE_DFLASH2_RESIDUAL_RMSNORM_QUANT
PHASESHIFT_FUSE_DFLASH2_RERANK_TOP16
PHASESHIFT_DEEPFUSE_TARGET_DOWN_RESIDUAL
PHASESHIFT_DEEPFUSE_TARGET_GATE_UP_SWIGLU
PHASESHIFT_DEEPFUSE_DFLASH2_MLP_A
PHASESHIFT_DEEPFUSE_DFLASH2_GATE_UP_SWIGLU
```

`PHASESHIFT_FUSE_Q_POSTPROCESS` / `PHASESHIFT_FUSE_K_POSTPROCESS` は perf-neutral の
ため MAX には含まれなかった。`PHASESHIFT_HIP_GRAPH` は Fusion ではなく execution mode
であり、preset とは独立した option。

## Target prefill（pp2048）

`--prompt-tokens 2048 --mode forward`、外側 3 rep × 内側 runs 3 / warmup 1 の中央値。
単位は gpu_tokens_per_sec。

| build condition | 中央値 | vs 基準 |
| --- | ---: | ---: |
| graph OFF / fusion OFF（基準） | 2302.87 | — |
| graph OFF / fusion ON | 2336.21 | +1.45% |
| graph ON / fusion OFF | 2313.08 | +0.44% |
| graph ON / fusion ON | **2340.15** | **+1.62%** |

## Target non-spec decode（tg128・ctx2048）

`--context 2048 --tokens 128 --mode greedy --compute-logits 1 --warmup 2`、3 rep 中央値。
単位は gpu_tokens_per_sec。

| build condition | 中央値 | vs 基準 |
| --- | ---: | ---: |
| graph OFF / fusion OFF（基準） | 27.50 | — |
| graph OFF / fusion ON | 28.69 | +4.33% |
| graph ON / fusion OFF | 28.44 | +3.42% |
| graph ON / fusion ON | **29.36** | **+6.76%** |

`GREEDY_TOKEN_SUM=2446188` は全条件・全 rep で一致。

## DFlash2 speculative decode（prose K7 256）

`--dflash2-drafts 7`、256 new tokens、3 rep 中央値。単位は DECODE_TOKENS_PER_SEC。

| build condition | 中央値 | vs 基準 |
| --- | ---: | ---: |
| graph OFF / fusion OFF（基準） | 63.55 | — |
| graph OFF / fusion ON | 64.56 | +1.59% |
| graph ON / fusion OFF | 63.84 | +0.46% |
| graph ON / fusion ON | **65.34** | **+2.82%** |

`GENERATED_IDS` sha1 先頭 12 桁 `47aebe55d048`、`rounds=84` / `accepted=171` は
全条件・全 rep で一致。`DFLASH2_VERIFY_GPU_MS` 中央値は基準 3451.4 →
fusion 3390.1 / graph 3421.0 / 両方 ON 3335.6。

## 計測コマンド

```sh
# pp2048
./build/phaseshift-bench pp --model-dir models/Qwen3.8-27B-PSQ \
  --prompt-tokens 2048 --mode forward --page-tokens 16 --arena-gib 24 \
  --runs 3 --warmup 1 --device 1

# tg128
./build/phaseshift-bench tg --model-dir models/Qwen3.8-27B-PSQ --mode greedy \
  --context 2048 --tokens 128 --prefill-chunk 2048 --compute-logits 1 \
  --page-tokens 16 --arena-gib 24 --warmup 2 --device 1

# DFlash2 prose K7 256
./build/phaseshift-compute --model-dir models/Qwen3.8-27B-PSQ \
  --dflash2-model-dir models/Qwen3.8-27B-DFlash2-PSQ4 --dflash2-drafts 7 \
  --input-ids-file build/dflash2-gate51/real/real_prose/prompt_ids.txt \
  --max-new-tokens 256 --max-seq-len 4096 --arena-gib 26 --temperature 0 \
  --dflash2-stats 1 --device 1
```

## Caveat

- GPU0 は別 workload と共有している。再現時は `--device 1` 等で隔離された GPU を使う。
- 初回プロセスは GPU クロックのランプアップで pp が大きく遅く出るため、
  warmup で除外している。
- 絶対値は single-run ではぶれる。run 間安定性（中央値）で判定する。
- ここに無い性能値が必要な場合は、勝手に推測せず `docs/perf/methodology.md` の
  手順で新規計測し、日付・revision・commit とともに追記する。
