> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# 27B-PSQ standalone primitive baseline (2026-09-20)

## 目的

Fusion を使わずに、Qwen3.8-27B-PSQ の主要 primitive を standalone optimized
kernel で実行し、correctness fallback に依存しない状態の基準値を保存する。
この baseline の上で、次タスクとして Fusion の要否を評価する。

対象: AMD Radeon AI PRO R9700 (gfx1201) / ROCm / HIP / Linux。

## 環境

| 項目 | 値 |
| --- | --- |
| commit | `c5a64470`（本 doc 追加前） |
| build | Release / gfx1201 / benchmarks ON / optional tests OFF |
| arena | 24 GiB |
| page-tokens | 16 |

## E2E A/B（base = `940d10b9` vs 本 baseline、3 組、old/new 交互）

`pp2048` は `--prompt-tokens 2048 --mode forward`、`tg128` は
`--context 2048 --tokens 128 --mode greedy --compute-logits 1 --warmup 2`。

### pp2048

| 組 | base t/s | new t/s |
| --- | ---: | ---: |
| 1 | 900.11 | 1998.96 |
| 2 | 899.79 | 1995.25 |
| 3 | 896.93 | 1986.55 |
| 中央値 | 899.79 | 1995.25 |

**2.22x**（`gpu_ms_median` 2276 → 1025）。

### tg128

| 組 | base t/s | base ms/tok | new t/s | new ms/tok |
| --- | ---: | ---: | ---: | ---: |
| 1 | 7.60 | 131.66 | 23.27 | 42.97 |
| 2 | 7.58 | 131.97 | 23.23 | 43.05 |
| 3 | 7.56 | 132.29 | 23.03 | 43.43 |
| 中央値 | 7.58 | 131.97 | 23.23 | 43.05 |

**3.06x**。

## KERNEL_TRACE（tg128）

```
KERNEL_TRACE rmsnorm_optimized=27376 kv_append=2096 paged_attention=2096 \
  gdn_recurrence=6288 elementwise=56592 embedding=131 sampling=128 \
  l2_normalize=12576 gdn_conv=6288 rope=4192 output_gather=128
```

`KERNEL_TRACE_FALLBACK` の出力は**無し**（全 KernelId で correctness fallback = 0）。
主要 6 KernelId（SILU / SCALE / RMS_NORM / L2_NORMALIZE /
STATEFUL_CAUSAL_CONV1D / ROPE）に加え OUTPUT_GATHER / SAMPLING / LINEAR_PSQ8 も 0。

## perf stats（tg128, `--perf-stats-output`）

```json
{
  "tokens": 128,
  "logical_dispatches": 211712,
  "optimized_operations": 211712,
  "correctness_fallbacks": 0,
  "physical_kernel_launches": 213888
}
```

- `physical_kernel_launches > logical_dispatches` の差分 +2176 は
  paged-attention decode（1 dispatch あたり 2 launch）と sampling partitioned
  （同 2 launch）の内部 multi-launch 分。融合（複数 dispatch を 1 launch が
  consume）ではない。融合なら physical < logical になる。
- 各 standalone kernel は `consumed_dispatches == 1`。

## standalone kernel 性能（rows=1、p50/p95、warmup 100 / samples 200 / launches 20）

| Kernel | shape | correctness p50 | optimized p50 | optimized p95 | speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| SILU F32→BF16 | F=10240 | 139.59 µs | 3.55 µs | 3.63 µs | 39.4x |
| SILU BF16→F32 | F=6144 | 140.56 µs | 3.57 µs | 3.68 µs | 39.4x |
| SCALE F32 | F=2048 | 141.52 µs | 3.44 µs | 3.55 µs | 41.2x |
| RMSNorm F32→F32 PG DIRECT | F=6144 G=128 | 489.82 µs | 4.01 µs | 4.18 µs | 122.1x |
| RMSNorm BF16→F32 PG ONE_PLUS | F=6144 G=256 | 499.54 µs | 4.07 µs | 4.19 µs | 122.9x |
| L2 normalize BF16→F32 | F=2048 G=128 | 271.62 µs | 4.03 µs | 4.40 µs | 67.4x |
| Conv1d BF16→F32 | conv_dim=10240 | 137.45 µs | 3.97 µs | 4.21 µs | 34.7x |
| RoPE Q F32→BF16 | F=6144 H=256 R=64 | 138.14 µs | 10.56 µs | 11.38 µs | 13.1x |
| RoPE K F32→BF16 | F=1024 H=256 R=64 | 139.09 µs | 6.52 µs | 7.01 µs | 21.3x |

全 shape で optimized が correctness を上回る。全 kernel の `--check` は PASS
（conv1d と RoPE は bit-exact、L2 は max_rel 3.6e-7、RMSNorm は既存 tolerance 内）。

## correctness / token について

- `PHASESHIFT_QWEN35_KERNEL_MODE=correctness` の出力は base commit と bit 一致
  （`GREEDY_TOKEN_SUM=181755`、先頭トークン列も同一）。correctness path は不変。
- `auto` の生成トークンは base auto から分岐する
  （base auto も既に base correctness と分岐済み）。原因は最適化 kernel の
  f32 丸め差で greedy の near-tie が flip するためで、各 kernel は `--check` PASS。
  f32 丸め差による near-tie flip は topic 別履歴に記録がある既知挙動
  （例: [gdn 履歴](gdn/optimization_history.md)）。
- 全 fallback 0 のため、`record_correctness_fallback` が一度も呼ばれず、
  kernel trace が出ない問題を修正（`try_launch_optimized` 先頭で登録）。

## 再現コマンド

```sh
# pp2048
./build/phaseshift-bench pp --model-dir models/Qwen3.8-27B-PSQ \
  --prompt-tokens 2048 --mode forward --page-tokens 16 --arena-gib 24 \
  --runs 1 --warmup 0

# tg128 + trace + stats
PHASESHIFT_QWEN35_KERNEL_TRACE=1 PHASESHIFT_QWEN35_PERF_STATS=1 \
./build/phaseshift-bench tg --model-dir models/Qwen3.8-27B-PSQ --mode greedy \
  --context 2048 --tokens 128 --prefill-chunk 2048 --compute-logits 1 \
  --page-tokens 16 --arena-gib 24 --warmup 2 \
  --perf-stats-output /tmp/tg_stats.json

# primitive micro-bench（例）
./build/phaseshift-bench l2-normalize --variant optimized --features 2048 \
  --group-size 128 --rows 1 --check --warmup 100 --samples 200 --launches 20
```

## 次のタスク

この baseline を出発点に、standalone（Conv / SILU / L2 / Scale）と
fused GDN prepare の比較を行い、Fusion による差と primitive 未最適化による差を
分離して評価する。

## 追記（2026-09-20）

上記 snapshot 後、RoPE optimized kernel を head_dim=256/rotary=64 専用 pair kernel
+ head loop パイプライン化（§7.79）、その後の Phase 2（GPU inv_freq table による
hot path からの pow 除去、§7.80）へ更新。
rows=1 の RoPE p50 は Q 10.56 → 6.51 → **5.27 µs**、K 6.52 → 5.69 → **4.25 µs**。
bit-exact（double pow + double sincos の BF16 落し込み）は Phase 2 でも維持
（pair vs generic 22 case・freq table identity 2 case すべて mismatch=0）。

続けて PSQ4 decode GEMV を §7.81 の手順で更新した。codebook 展開の代数簡約
（Phase 1）、rows=1 専用 decode1 kernel（Phase 2）、unroll 再調整（Phase 3/4）を
経て、production 7 shape の p50 は 5120/17408 84.04→83.97 µs、
17408/5120 84.06→82.69、5120/12288 60.70→59.82、5120/10240 51.31→50.07、
6144/5120 35.85→32.04、5120/6144 32.27→31.63、5120/1024 12.05→7.88 µs。
PSQ4 合計 20.54→20.08 ms/token、E2E decode 39.24→38.22 ms/token（25.5→26.2 t/s）。
N=1024 の 8-output/wave 実験（Phase 5）は全 shape で悪化するため棄却した。

続けて GDN recurrence の decode を §7.83 の手順で更新した。rows=1 専用
decode1 kernel の追加（Phase 1）で、VGPR 169 / spill 0 / LDS 35076 B となり、
p50 は nreq=1 34.85→**14.70 µs**（2.37x）、nreq=2 50.60→24.89、nreq=4
95.43→43.50、nreq=8 180.66→81.30。E2E decode 38.22→**36.97 ms/token**
（26.2→27.1 t/s）。128 threads 化（Phase 2、-9〜-22%）と LDS ≤ 32768 B の
2 blocks/WGP 化（Phase 3、-6〜-12%）はどちらも棄却した。

本カーネルは occupancy ではなく **DRAM 帯域律速**である: nreq=8 で
619 GB/s = 実測 peak 636 GB/s の 97%。LDS を 35076 → 8452 B（1 → 4+ blocks/WGP）
にしても性能は 1% も変わらない。

測定上の注意: GPU0 は別 workload と共有しており、device 0 の nreq ≥ 4 は
15〜21% 大きく出る。上記の値は `--device 2` で取得し、再現性は 0.3% 以内。

さらに §7.84 で GDN recurrence decode1 の律速点を再判定した。standalone bench の
state 3 MiB は L2 に残るため、このカーネルは DRAM ではなく compute（issue と共有
ユニット）律速である。occupancy を 1 → 4 blocks/WGP（warps/WGP 8 → 32）にしても
わずかに悪化する。

その上で dead work 除去（Phase 4）、`akj` fragment の再利用（Phase 5）、k/q
fragment の LDS 共有（Phase 6）を行った。命令数は 5412 → 1716（-68%）、
rows=1 の p50 は nreq=1 14.70 → **11.93 µs**（-19%）、nreq=2 24.89 → **17.21 µs**
（-31%）。bit-exact は 20/20 で維持し、E2E decode は 36.64 → **36.52 ms/token**。

## 追記（2026-09-21）: 現在の pp / tg

上記 primitive 最適化の後、PSQ8 prefill 2d GEMM、GDN conv1d の row tiling、
CP keep-alive の production 化（§7.96 / §7.97）まで反映した断面の E2E を再取得した。

| 項目 | 値 |
| --- | --- |
| commit | `215f8396`（`perf/psq8-prefill` merge 後） |
| device | 1 |
| arena | 24 GiB |
| page-tokens | 16 |
| keep-alive | 有効（executor 既定。§7.97） |

### pp2048

`--prompt-tokens 2048 --mode forward`、3 組。

| 組 | gpu_ms | t/s |
| --- | ---: | ---: |
| 1 | 872.17 | 2348.17 |
| 2 | 871.38 | 2350.29 |
| 3 | 871.56 | 2349.80 |
| 中央値 | 871.56 | **2349.80** |

初版 baseline の 1995.25 t/s（`gpu_ms_median` 1025）から **+17.8%**。
`../runtime/execution_overhead.md` §7.97 の 2357.0 t/s と 0.3% 以内で一致する。

### tg128

`--context 2048 --tokens 128 --mode greedy --compute-logits 1 --warmup 2`、3 組。

| 組 | gpu_ms | us/tok | t/s |
| --- | ---: | ---: | ---: |
| 1 | 4710.09 | 36797.61 | 27.18 |
| 2 | 4712.93 | 36819.77 | 27.16 |
| 3 | 4714.90 | 36835.13 | 27.15 |
| 中央値 | 4712.93 | 36819.77 | **27.16**（36.82 ms/tok） |

初版 baseline の 23.23 t/s（43.05 ms/tok）から **+16.9%**。
[runtime 履歴](runtime/execution_overhead.md) §7.97 の 27.21 t/s と一致する。

### 再現コマンド

```sh
./build/phaseshift-bench pp --model-dir models/Qwen3.8-27B-PSQ \
  --prompt-tokens 2048 --mode forward --page-tokens 16 --arena-gib 24 \
  --runs 1 --warmup 0 --device 1

./build/phaseshift-bench tg --model-dir models/Qwen3.8-27B-PSQ --mode greedy \
  --context 2048 --tokens 128 --prefill-chunk 2048 --compute-logits 1 \
  --page-tokens 16 --arena-gib 24 --warmup 2 --device 1
```

`GREEDY_TOKEN_SUM=2446188`。pp / tg とも keep-alive は executor 既定で有効
（`PHASESHIFT_KEEPALIVE=0` で無効化可能）。

## 追記（2026-09-23）: 4 ビルド条件 × pp / tg / DFlash2

HIP_GRAPH ON/OFF と融合カーネル ON/OFF の 4 条件で pp / tg / DFlash2 E2E を
GPU1 で再取得した。

### 環境

| 項目 | 値 |
| --- | --- |
| commit | `ba153347` |
| device | 1 |
| build | Release / gfx1201 |
| arena | bench 24 GiB / compute 26 GiB |
| page-tokens | 16 |
| keep-alive | 有効（executor 既定） |

### ビルド条件

| build dir | HIP_GRAPH | 融合 |
| --- | --- | --- |
| `build` | OFF | OFF |
| `build-fuse` | OFF | ON |
| `build-hipgraph` | ON | OFF |
| `build-fuse-hipgraph` | ON | ON |

融合 ON は Gate 11K 以降の正式集合（`docs/rnd/dflash2/dflash2.md` regression
candidate config）である 10 フラグ。`FUSE_Q_POSTPROCESS` / `FUSE_K_POSTPROCESS`
は perf-neutral のため OFF のまま。

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

注意: Gate 11K で `PHASESHIFT_FUSE_TARGET_SWIGLU_QUANT` /
`PHASESHIFT_FUSE_DFLASH2_SWIGLU_QUANT` は削除され、`PHASESHIFT_DEEPFUSE_*` 4 個が
追加された。古い `build-fuse*` キャッシュはこの集合とズレているため、本次数は
再 configure（上のフラグを明示指定）+ rebuild で計測している。

### pp2048（gpu_tokens_per_sec、外側 3 rep × 内側 runs 3 / warmup 1 の中央値）

| 条件 | r1 | r2 | r3 | 中央値 | vs 基準 |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph OFF / fuse OFF（基準） | 2325.50 | 2302.87 | 2294.05 | 2302.87 | — |
| graph OFF / fuse ON | 2355.37 | 2336.21 | 2322.54 | **2336.21** | **+1.45%** |
| graph ON / fuse OFF | 2315.61 | 2313.08 | 2305.12 | 2313.08 | +0.44% |
| graph ON / fuse ON | 2355.80 | 2340.15 | 2325.02 | **2340.15** | **+1.62%** |

### tg128・ctx2048（gpu_tokens_per_sec、3 rep）

| 条件 | r1 | r2 | r3 | 中央値 | vs 基準 |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph OFF / fuse OFF（基準） | 27.54 | 27.48 | 27.50 | 27.50 | — |
| graph OFF / fuse ON | 28.50 | 28.71 | 28.69 | **28.69** | **+4.33%** |
| graph ON / fuse OFF | 28.45 | 28.44 | 28.34 | 28.44 | **+3.42%** |
| graph ON / fuse ON | 29.29 | 29.39 | 29.36 | **29.36** | **+6.76%** |

`GREEDY_TOKEN_SUM=2446188` は全条件・全 rep で一致（追記（2026-09-21）の値とも一致）。

### DFlash2 prose K7 256（DECODE_TOKENS_PER_SEC、3 rep）

| 条件 | r1 | r2 | r3 | 中央値 | vs 基準 |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph OFF / fuse OFF（基準） | 63.63 | 63.55 | 62.50 | 63.55 | — |
| graph OFF / fuse ON | 64.56 | 64.57 | 64.53 | **64.56** | **+1.59%** |
| graph ON / fuse OFF | 63.85 | 63.84 | 63.84 | 63.84 | +0.46% |
| graph ON / fuse ON | 65.34 | 66.35 | 65.27 | **65.34** | **+2.82%** |

`GENERATED_IDS`（sha1 先頭 12 桁 `47aebe55d048`）と `rounds=84` /
`accepted=171` は全条件・全 rep で完全一致。
`DFLASH2_VERIFY_GPU_MS` 中央値は基準 3451.4 → fuse 3390.1 / graph 3421.0 /
両方 ON 3335.6。

### 読み取り

- tg では fuse（+4.33%）と graph（+3.42%）がほぼ直交的に重畳し、両方 ON で
  +6.76%（乗算予測 +7.9% に対してややサブ乗算だが、全 rep で最速・順序一貫）。
- pp と DFlash2 では fuse が確実（3/3 rep で基準を上回る）、graph 単独効果は
  +0.44% / +0.46% でノイズに近い。pp2048（= prefill 1 発）は prefill bucket が
  `staging pool not warm` のままで capture されず、§7.101 の記述どおり graph が
  効かない。DFlash2 は verify が約 83 回 replay するが、round 時間に占める
  launch overhead 削減が小さく +0.5% 程度。
- 絶対値は 2026-09-21 断面（pp 2349.8 / tg 27.16）と同程度の分散内。
  なお初回プロセスは GPU クロックのランプアップで pp が約 1.5 倍遅く出るため、
  `--warmup 1`（pp）/ `--warmup 2`（tg）で除外している。

### 再現コマンド

```sh
# pp2048（<dir> は build / build-fuse / build-hipgraph / build-fuse-hipgraph）
./<dir>/phaseshift-bench pp --model-dir models/Qwen3.8-27B-PSQ \
  --prompt-tokens 2048 --mode forward --page-tokens 16 --arena-gib 24 \
  --runs 3 --warmup 1 --device 1

# tg128
./<dir>/phaseshift-bench tg --model-dir models/Qwen3.8-27B-PSQ --mode greedy \
  --context 2048 --tokens 128 --prefill-chunk 2048 --compute-logits 1 \
  --page-tokens 16 --arena-gib 24 --warmup 2 --device 1

# DFlash2 prose K7 256
./<dir>/phaseshift-compute --model-dir models/Qwen3.8-27B-PSQ \
  --dflash2-model-dir models/Qwen3.8-27B-DFlash2-PSQ4 --dflash2-drafts 7 \
  --input-ids-file build/dflash2-gate51/real/real_prose/prompt_ids.txt \
  --max-new-tokens 256 --max-seq-len 4096 --arena-gib 26 --temperature 0 \
  --dflash2-stats 1 --device 1
```

融合 ON ビルドの再 configure は上記 10 フラグを `-D` で明示して行う
（`-DPHASESHIFT_HIP_GRAPH=ON` は graph 条件のみ）。
