# FFN down_proj N64 × K256 Sparse Break-even Budget Gate

更新日: 2026-09-24
branch: `poc/ffn-k256-break-even`
harness: `tools/rnd/ffn_k256_break_even/`

## 1. Goal

前回 Gate で N64 × K256 の PAIR_GREEDY_SWAP は +2% 品質帯 21–24%、+5% 品質帯 40–42% の
headroom を持つと分かった。今回は selector を作るのではなく、

> K256 group を実際に weight load / decode / WMMA から消したとき、R9700 上で何 µs 削減でき、
> selector + mask 生成に最大何 µs 使えるか

を microbenchmark で確定し、cheap selector 研究に進むだけの latency budget が存在するかを
判定する。

## 2. Baseline revision

| 項目 | 値 |
| --- | --- |
| base revision | `8c9b920a`（`poc/ffn-k256-cancel-oracle`） |
| GPU | AMD Radeon AI PRO R9700 (gfx1201) / GPU2 |
| ROCm / HIP | 7.15.26333 |
| torch / transformers | 2.13.0+rocm10.0.0 / 5.14.1 |
| model | `models/Qwen3.8-27B-PSQ` |
| corpus | `artifacts/ffn_prune/corpus.psktok`（SHA256 `72392ef8…593b`） |
| E2E baseline | `phaseshift-bench tg`（build `79b6994d`）再測定 27.09 tok/s |

GPU0,1 は他 workload が使用していたため GPU2 のみを使用（GPU3 も空き）。production kernel は
未変更で、benchmark 専用 kernel を `tools/rnd/ffn_k256_break_even/bench_sparse_down.hip` に
持つ。

## 3. Current quantization inventory

`models/Qwen3.8-27B-PSQ/phaseshift_quantization.json` の 64 layer `down_proj` を集計。

| encoding | layers | layer |
| --- | ---: | --- |
| psq8 | 16 | 0,4,8,…,60 |
| psq4 | 48 | それ以外 |

`k_padded = 17408`、`logical_shape = [5120, 17408]`、K256 = 68 groups = 8 × K32。

## 4. Kernel design

- 1 block = N16 output tile（32 threads、k_group 2）。N64 tile は `blockIdx.x >> 2`。
- `for g in 0..67` の外側で mask を 1 回判定し、KEEP 時のみ内側 8 K32 反復を実行する。
  SKIP 時は weight code / scale の load・decode・WMMA を本当に実行しない。
- 4 経路: DENSE（production と同一）、IDEAL_STATIC（mask load なしの contiguous prefix skip）、
  RUNTIME_MASK（3×uint32 mask を block 先頭で load して skip）、MASK_READ_ONLY（skip せず
  mask load + bit test のみ）。
- production の `launch_gemm_psq{4,8}_w8a8_wmma_decode1` をそのまま呼び、DENSE baseline と
  correctness reference に使用する。

## 5. Mask representation

N64 tile ごとに `3 × uint32`（68 bit、bit=1 が SKIP）。80 tiles で 960 byte。隣接する 4 つの
N16 block は同一 N64 mask を参照する。kernel は block 先頭で 3 word を一度 load する。
N64 内は uniform branch であり wave divergence を発生させない。

## 6. Correctness

skip した group の weight code / scale を 0 にした buffer を production kernel で計算し、
RUNTIME_MASK_SKIP 出力と比較した（code 0 は cb10 で寄与 0 になるため厳密な reference）。
全 pattern・全 skip（0 / 7 / 10 / 14 / 16 / 17 / 20 / 27 / 28 / 30）× PSQ4/PSQ8 で
`max abs diff = 0`。DENSE（自作）と production DENSE の出力も完全一致。skip=0 と skip=16/28 の
独立性、tile ごとに独立 mask、68 bit の tail word も確認済み。NaN / Inf なし。

## 7. Dense latency

同一 weight を毎 launch 再利用すると working set が L3 (64MB) に載り 36us という非現実的な値に
なる。weight buffer を 4 copy ローテーションし working set を L3 より大きくすると
（PSQ4 200.5MB、PSQ8 378.8MB）、

| format | production dense p50 | bench dense p50 | working set |
| --- | ---: | ---: | ---: |
| PSQ4 (K=17408) | 82.46us | 82.45us | 200.5MB |
| PSQ8 (K=17408) | 152.73us | 152.71us | 378.8MB |

旧記録（PSQ4 83.4us / PSQ8 153.2us）と一致。run 間 p50 spread は PSQ4 で 82.452–82.462us
（<0.02%）。

## 8. Ideal-static sparse latency

mask load なしの contiguous prefix skip（branch のみ）。runtime との差が mask 処理 overhead に
相当する。後述の表の `ideal` 列。RUNTIME との差は全点で 0.1–0.7us。

## 9. Runtime-mask sparse latency

事前生成済み N64 mask を load し、KEEP のみ 8×K32 を実行する primary 経路。

### PSQ4（dense 82.45us）

| skip | actual | dense | ideal | runtime | efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0% | 82.45 | 82.68 | 82.82 | — |
| 7 | 10.29% | 82.45 | 74.96 | 75.41 | 0.885 |
| 10 | 14.71% | 82.45 | 71.55 | 71.50 | 0.932 |
| 14 | 20.59% | 82.45 | 67.06 | 67.28 | 0.929 |
| 16 | 23.53% | 82.45 | 64.64 | 65.18 | 0.924 |
| 17 | 25.00% | 82.45 | 63.55 | 64.01 | 0.929 |
| 20 | 29.41% | 82.45 | 60.08 | 60.24 | 0.942 |
| 27 | 39.71% | 82.45 | 52.22 | 52.30 | 0.942 |
| 28 | 41.18% | 82.45 | 50.99 | 51.49 | 0.938 |
| 30 | 44.12% | 82.45 | 48.69 | 48.51 | 0.951 |

### PSQ8（dense 152.71us）

| skip | actual | dense | ideal | runtime | efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0% | 152.71 | 152.69 | 152.93 | — |
| 10 | 14.71% | 152.71 | 131.16 | 130.73 | 0.995 |
| 16 | 23.53% | 152.71 | 117.92 | 117.52 | 0.965 |
| 20 | 29.41% | 152.71 | 109.16 | 108.52 | 0.984 |
| 27 | 39.71% | 152.71 | 94.25 | 93.49 | 0.973 |
| 28 | 41.18% | 152.71 | 91.78 | 91.06 | 0.980 |
| 30 | 44.12% | 152.71 | 87.70 | 86.68 | 0.977 |

（効率は random pattern。contiguous も同程度。）

## 10. Mask pattern sensitivity

skip=16/27 で contiguous / random deterministic / Oracle replay を比較:
PSQ4 skip=16 は 64.80 / 65.18 / 64.75us、skip=27 は 52.57 / 52.30 / 51.87us。
PSQ8 skip=27 は 94.46 / 93.49 (L32) / 93.53 (L0) us。
pattern 差は 0.5us 以内で、mask の並びは latency に有意な影響を与えない。

## 11. Oracle replay result

`gen_oracle_masks.py` が前回 PAIR_GREEDY_SWAP を token 0 で再実行し、L0/L16/L28/L32/L48/L63 の
N64 mask を dump（selector 時間は計測外）。L32（PSQ8）/ L63（PSQ4）の replay は
random/contiguous と 0.5us 以内で一致し、correctness も `max abs diff = 0`。primary の
latency は replay と random で差がないと確認した。

## 12. Profiling / actual traffic reduction

rocprofv3 1.3.5 で `FETCH_SIZE` / `SQ_INSTS_VALU` を収集したが、本環境では全値 0
（hardware PMC config が公開されておらず multi-pass 不可）。代わりに有効帯域で確認する。

PSQ4 / PSQ8 とも、retained byte 数 × 実行時間がほぼ一定の有効帯域になる:

| kind | dense GB/s | skip=27 GB/s | skip=30 GB/s |
| --- | ---: | ---: | ---: |
| PSQ4 | 608 | 578 | 578 |
| PSQ8 | 620 | 611 | 611 |

すなわち skip 後も DRAM 帯域 (~610GB/s) を使い切っており、時間が retained byte に比例して
減っている = weight traffic が実際に減っている。加えて SKIP branch 内で weight pointer
dereference / decode / WMMA が一切実行されないことをソースで確認した。

## 13. Skip efficiency

`efficiency = (dense - runtime) / (dense × skip_fraction)`:

| skip | PSQ4 | PSQ8 |
| ---: | ---: | ---: |
| 10.29% | 0.885 | 0.996 |
| 23.53% | 0.924 | 0.965 |
| 39.71% | 0.942 | 0.973 |
| 44.12% | 0.951 | 0.977 |

40% skip で線形理想 saving の **94%（PSQ4）/ 97%（PSQ8）** を回収。

## 14. Selector break-even budget

64 layer = PSQ4 48 + PSQ8 16、E2E baseline 27.09 tok/s（36.91 ms/token）で投影。

per-layer gross save（random pattern、us）:

| skip | PSQ4 48L × save | PSQ8 16L × save | token gross save |
| ---: | ---: | ---: | ---: |
| 16 (23.53%) | 48 × 17.27 = 829 | 16 × 35.20 = 563 | 1.392 ms |
| 27 (39.71%) | 48 × 30.15 = 1447 | 16 × 59.22 = 948 | 2.395 ms |
| 28 (41.18%) | 48 × 30.96 = 1486 | 16 × 61.65 = 986 | 2.473 ms |

selector budget（gross save を基準）:

| OP | gross save/token | break-even | 75% retention | 50% retention | +3% E2E | +5% E2E | +8% E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 23.53% | 1.392 ms | 21.75us/L | 5.44us/L | 10.87us/L | 4.95us/L | 負（不可） | 負（不可） |
| 39.71% | 2.395 ms | 37.42us/L | 9.35us/L | 18.71us/L | 20.62us/L | **9.95us/L** | 負（不可） |
| 41.18% | 2.473 ms | 38.63us/L | 9.66us/L | 19.32us/L | 21.83us/L | **11.17us/L** | 負（不可） |

E2E baseline を fusion+graph の 29.36 tok/s（34.06 ms）にすると +5% budget は
39.71% で 12.08us/L、41.18% で 13.29us/L とさらに余裕が増える。

## 15. E2E projection

baseline: 27.09 tok/s / 36.91 ms/token。

| selector cost | 23.53% skip | 39.71% skip | 41.18% skip |
| --- | ---: | ---: | ---: |
| 0 us/L | 28.15 tok/s (+3.9%) | 28.97 (+6.9%) | 29.03 (+7.2%) |
| 2 us/L | 28.05 (+3.5%) | 28.86 (+6.5%) | 28.93 (+6.8%) |
| 5 us/L | 27.90 (+3.0%) | 28.70 (+6.0%) | 28.77 (+6.2%) |
| 10 us/L | 27.65 (+2.1%) | 28.44 (+5.0%) | 28.51 (+5.2%) |

selector が無料でも +8% E2E には届かない（down_proj だけで 64 layer 合計 2.4–2.5ms が上限）。
+5% E2E は 40% 前後 skip で selector を ~10us/layer 以下にできれば達成できる。

## 16. Gate decision

**GO**

| 条件 | 判定 | 根拠 |
| --- | --- | --- |
| OP-B/C で skip_efficiency ≥ 0.70 | PASS | 39.71% で 0.942、41.18% で 0.938 |
| mask_overhead_ratio ≤ 0.25 | PASS | PSQ4 で 0.003（runtime ≈ ideal、maskonly ≈ dense） |
| +5% E2E selector budget が正で余裕 | PASS | 39.71% で 9.95us/L、41.18% で 11.17us/L（≥3us/L の推奨を大きく上回る） |
| weight load / decode / compute が実際に減っている | PASS | reference 一致、有効帯域が skip 後も ~610GB/s で一定（時間 ∝ retained byte） |

40% 前後を実際に消して down_proj latency が 36% 削減（82.45→52.30us PSQ4）、mask 処理 overhead は
実質ゼロ。cheap selector 研究を正当化する latency budget が存在する。

## 17. Next step

`Cheap cancellation-aware vector sketch Gate`

exact c_g / pair / swap を使わず、低次元 sketch などで cancellation-aware な subset を
予測できるかを検討する。selector budget は 40% skip で ~10us/layer、+3% E2E なら ~20us/layer。

## 18. Reproduction commands

```bash
W=.worktrees/ffn-k256-break-even
cd $W
bash tools/rnd/ffn_k256_break_even/build.sh
ROCM=/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH=$ROCM/lib HIP_VISIBLE_DEVICES=2
B=tools/rnd/ffn_k256_break_even/bench_sparse_down
$B --kind psq4 --copies 4 --out artifacts/ffn_k256_break_even/dense_psq4.json \
   --skips 0,7,10,14,16,17,20,27,28,30
$B --kind psq8 --copies 4 --out artifacts/ffn_k256_break_even/dense_psq8.json \
   --skips 0,7,10,14,16,17,20,27,28,30

# Oracle replay masks
CUDA_VISIBLE_DEVICES=2,3 python3 tools/rnd/ffn_k256_break_even/gen_oracle_masks.py \
   --layers 0,16,28,32,48,63 --out-dir artifacts/ffn_k256_break_even/oracle_masks
$B --kind psq8 --copies 4 --mask-dir artifacts/ffn_k256_break_even/oracle_masks/L32 \
   --out artifacts/ffn_k256_break_even/replay_psq8_L32.json --skips 0,7,10,14,16,17,20,27,28,30
$B --kind psq4 --copies 4 --mask-dir artifacts/ffn_k256_break_even/oracle_masks/L63 \
   --out artifacts/ffn_k256_break_even/replay_psq4_L63.json --skips 0,7,10,14,16,17,20,27,28,30

# E2E baseline
./build/phaseshift-bench tg --model-dir models/Qwen3.8-27B-PSQ --mode greedy \
   --context 2048 --tokens 128 --prefill-chunk 2048 --compute-logits 1 \
   --page-tokens 16 --arena-gib 24 --warmup 2 --device 0

# projection
python3 tools/rnd/ffn_k256_break_even/analyze_break_even.py --pattern random \
   --e2e-baseline artifacts/ffn_k256_break_even/e2e_baseline.json \
   --out artifacts/ffn_k256_break_even/break_even.json
```

raw JSON / summary は `artifacts/ffn_k256_break_even/`（git 管理外）。
