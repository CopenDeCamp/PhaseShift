> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# vllm-mxfp4 competitive gap audit

## Audit provenance

| 項目 | 値 |
| --- | --- |
| repository | `GGZ14/vllm-mxfp4` |
| cloned at | 2026-09-22 |
| audited SHA | `31b9a94a7f74eeb3f59e66d16b1b27dfafcd0663` |
| audited commit date | 2026-09-18 |
| method | `git clone --filter=blob:none --depth=1` を `/tmp` に取得。repo は vendor しない |
| primary reference | `README.md#performance`（BetterBench `--quick`, corpus v1.0, 2026-09-16） |

一次情報は repo 内の `README.md`、`PERFORMANCE.md`、`serve-tp1.sh`、`serve-mxfp4.sh`、
各 `patch_*.py`、`quantize_dflash_mxfp4.py`。

## Competitor configuration（source から確認）

| 項目 | vllm-mxfp4 |
| --- | --- |
| target checkpoint | `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` + fp8 MTP head rewrite（`Qwen3.8-27B-MXFP4-mtpfp8`） |
| target quantization | native MXFP4 W4A8、fp8-WMMA（`RADIANCE_MXFP4_W4A8`） |
| target size | 9.4 GiB/GPU（TP=2）。TP=1 では全 weight を 1 card（~19 GiB source） |
| drafter | `tcclaviger/Qwen3.8-27B-DFlash2-FP8` |
| drafter precision | FP8 e4m3 per-channel。`RADIANCE_FAST_DRAFT` が decoder projection を signed symmetric int4（4.25 bits/weight）+ draft head 2-bit exact rerank に再pack |
| drafter size | 2 GiB |
| attention KV dtype | `--kv-cache-dtype fp8`（serve default） |
| GDN recurrent dtype | fp16 ssm cache（rx9 narrow-state GDN decode kernel） |
| GDN conv dtype | bf16 |
| SPEC width | `SPEC=7`（dflash）。`SPEC=5` も測定 |
| dynamic width | `RADIANCE_DYNAMIC_DRAFT` は MTP の serial draft loop 用。dflash + CUDA graph では depth 固定で inert。verify side の dynamic width は `f68d215` |
| verify head | int2 target verify head（`f882ea2`, +2.9%） |
| small-M / skinny GEMM | `RADIANCE_SKINNY_GEMM`: R4D split-K bf16 for M in [6,64]。MXFP4 decode kernel `RADIANCE_MXFP4_DECODE_MAX_M=64`、split-K |
| graph | torch.compile（inductor）+ Triton、vLLM cudagraph。AITER JIT core を image build 時に pre-bake |
| fusion | RMSNorm+quant、silu-mul-quant、GDN conv+recurrent（`9a84208`）、GDN in_proj merge（`588d5e6`）、residual+norm+rotate+quant、fp8 residual epilogue 等 |
| prefix cache | ON（`--enable-prefix-caching --mamba-cache-mode=align`） |
| GDN spec state strategy | vLLM mamba align-mode snapshot（block boundary の eager checkpoint）。lazy GDN snapshot（`RADIANCE_GDN_LAZY`）は 2026-09-17 に correctness 問題で default OFF |

## Public one-R9700 reference（vllm-mxfp4）

BetterBench `--quick`, 2026-09-16, one R9700, 210 W cap, -75 mV, `MAXSEQS=8`, 65536 context,
native MXFP4 checkpoint, `dflash` `SPEC=7`, temp 0.7 / top_p 0.95 / top_k 20, cold prefix cache。

| category | update p50 ms | tok/update | decode t/s |
| --- | ---: | ---: | ---: |
| code | 35.3 | 4.65 | 145.2 |
| json | 35.1 | 5.59 | 184.4 |
| math | 35.2 | 5.56 | 168.6 |
| summarization | 35.1 | 4.67 | 138.1 |
| file_edit | 35.3 | 5.63 | 166.1 |
| reasoning | 35.2 | 3.60 | 120.8 |
| chat | 34.9 | 2.75 | 81.4 |
| prose | 35.2 | 2.76 | 79.0 |
| combined | update p99 35.7 | - | 137.7 |

## Comparability notes

PhaseNonShift と vllm-mxfp4 は **directly comparable ではない**。

- checkpoint difference: PSQ8 W8A8（PhaseNonShift）vs MXFP4 W4A8（competitor）。weight 量子化精度が違う。
- target quantization difference: 8-bit vs 4-bit。decode weight traffic が約 2x 違う。
- drafter precision difference: DFlash2 BF16 3.6GB（PhaseNonShift）vs FP8 2GB + int4/int2 pack（competitor）。
- KV difference: BF16（PhaseNonShift default）vs FP8（competitor）。
- GDN state difference: F32 recurrent + BF16 conv（PhaseNonShift）vs fp16 recurrent + bf16 conv（competitor）。
- sampling difference: greedy temperature 0（PhaseNonShift）vs temp 0.7 / top_p 0.95 / top_k 20（competitor）。
- corpus difference: gate5.1 real prose/code/math（PhaseNonShift）vs BetterBench corpus v1.0（competitor）。
- power/clock difference: competitor は 210 W cap / -75 mV 固定。PhaseNonShift の計測条件は同一でない。
- metric: update p50 / tok/update を primary とする。tok/s は acceptance で大きく振れる。
- acceptance: competitor の SPEC=7 は code/json で 4.65-5.63 tok/update、prose で 2.76。
  PhaseNonShift の Gate10 prose は emitted/update 3.0 で近い帯。

よって本比較は **directional diagnostic only** とする。勝敗を断定しない。

## Feature gap matrix

| area | PhaseNonShift | vllm-mxfp4 | class | action |
| --- | --- | --- | --- | --- |
| target weight format | PSQ8 W8A8（~18-20GB） | MXFP4 W4A8（9.4GiB/GPU TP2） | FORMAT | 監査のみ（Gate11C 対象外） |
| drafter weight format | DFlash2 BF16（3.6GB） | FP8 + int4 decoder / int2 head | FORMAT | 監査のみ |
| attention KV dtype | BF16 | FP8 | FORMAT | 監査のみ（B） |
| GDN recurrent dtype | F32 | fp16 | FORMAT | 監査のみ（B） |
| GDN conv dtype | BF16 | BF16 | - | parity |
| GDN spec state handling | exact per-row history（本 Gate で追加） | eager block-boundary snapshot。lazy は OFF | STATE | **Gate11C で解決** |
| partial reject handling | full target rerun（本 Gate で除去） | vLLM align-mode snapshot | STATE | **Gate11C で解決** |
| dynamic verify width | fixed K=7 | dynamic verify width（MTP）/ dflash depth fixed | SCHEDULER | 監査のみ（F） |
| target small-M GEMM | BF16 WMMA splitk / exact_rows | MXFP4 W4A8 decode kernel、split-K | KERNEL | 監査のみ（A） |
| drafter skinny GEMM | BF16 WMMA splitk | R4D split-K bf16 M in [6,64] | KERNEL | 監査のみ（E） |
| verify lm_head | PSQ8 W8A8 + full logits + top16 | int2 target verify head | KERNEL | 監査のみ（G） |
| DFlash lm_head | PSQ8 W8A8 + top16 | FP8 / int4 decoder | KERNEL | 監査のみ |
| graph execution | none（direct HIP launch） | torch.compile + cudagraph | FUSION | **DEFERRED UNTIL KERNEL CEILING** |
| prefix cache | なし | mamba align-mode APC | SCHEDULER | 監査のみ |
| kernel fusion | なし | 多数 | FUSION | **DEFERRED UNTIL KERNEL CEILING** |

FUSION に分類された差は本 Gate では実装しない。単体 kernel の ceiling を計測した後に Gate12 以降で検討する。

## Gate 11C で解決する gap

STATE: partial reject 時の full target rerun を、exact per-row GDN state history に置き換える。
competitor の lazy GDN mode は correctness 問題で default OFF であるためコピーせず、
exact な per-row checkpoint を使う。
