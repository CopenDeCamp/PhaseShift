> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# TP-Exec-2: TP=2 Optimized Shape Coverage（Phase A）

## 目的

Qwen3.8-27B の TP=2 では local geometry が半分になり、optimized kernel の
validated allowlist 外になって Linear / Paged Attention / GDN Recurrence が
correctness fallback に落ちていた。TP=2 が optimized path で実行されるようにする。

## 背景（変更前）

TP=2 の local geometry:

```text
intermediate         17408 -> 8704
query heads             24 -> 12
key/value heads          4 -> 2
gdn key heads           16 -> 8
gdn value heads         48 -> 24
hidden                5120（分割しない）
```

これに対し selector は

- `linear_selector`: `(out_features, k)` の完全一致 allowlist
- `paged_attention_selector`: `q_heads ∈ {16,24}` かつ `kv_heads == 4`
- `gdn_recurrence_selector`: `key_heads == 16` かつ `num_v_heads ∈ {32,48}`

を要求するため、TP=2 は軒並み Correctness に落ちていた。

### enum と family bit の注意

`LinearComputeFamily` の enum 順は

```text
Psq4W4A8 = 0, Psq8W8A8 = 1, Bf16 = 2, Fp8W8A8 = 3, Mxfp4W4A8 = 4
```

であり、`kFamily*` の bit 順（Bf16=bit0, Psq4=bit1, Psq8=bit2 ...）とは一致しない。
`linear_debug_miss` が出す `family=N` は **enum 値**である。TP=2 実行時に必要な
family は、この enum 値で観測したものだけを追加した。

## 追加した geometry と family

`src/phaseshift/models/qwen35/runtime/linear_selector.cpp`:

| shape (out,k) | family |
|---|---|
| 8704, 5120 | Psq4 |
| 5120, 8704 | Psq4, Psq8 |
| 3072, 5120 | Psq4, Psq8 |
| 512, 5120 | Psq4 |
| 5120, 3072 | Psq4, Psq8 |
| 5120, 5120 | Psq4, Psq8 |
| 24, 5120 | Bf16 |

`paged_attention_selector.cpp`: BF16 KV に `(q_heads=12, kv_heads=2, head_dim=256,
page=16)` を追加。

`gdn_recurrence_selector.cpp`: `(key_heads=8, num_v_heads=24, head_k=128, head_v=128)`
を追加。

実行時に必要なかった family は追加していない（未知 shape の自動許可もしない）。

## 検証

- selector unit test: `test_qwen35_linear_selector` / `test_qwen35_paged_attention_selector`
  / `test_qwen35_gdn_recurrence_selector` に TP=2 の positive / negative（境界含む）を追加。
- kernel test（CPU reference 比較）:
  - `test_gemm_bf16_wmma` に `(24,5120)` ほか
  - `test_gemm_psq4_w4a8_wmma` に `(512,5120)`, `(5120,5120)` ほか rows=64
  - `test_gemm_psq8_w8a8_wmma` に `(5120,8704)`, `(3072,5120)`, `(5120,3072)`, `(5120,5120)`
  - `test_paged_attention` に `qh=12 kvh=2 hd=256 page=16` の decode / prefill / long decode
  - `test_gdn_recurrence` に `(8,24,128,128)` の decode / prefill / multi-request
- runtime（`PHASESHIFT_LINEAR_DEBUG=1`, `PHASESHIFT_QWEN35_KERNEL_TRACE=1`）:
  - `LINEAR_SELECT_MISS` = 0
  - `paged-attention: outside measured envelope` = 0
  - `KERNEL_TRACE paged_attention=128 gdn_recurrence=384`、`KERNEL_TRACE_FALLBACK LINEAR_*` = 0
- correctness gate 4/4 PASS（`test_tp_reduction` / `test_tp_transport` /
  `test_qwen35_tp_execution` / `test_qwen35_tp_e2e`、TP=1 vs TP=2 greedy token 一致）。
- required acceptance 124/124 PASS（skip 0, error 0）。

## 結果（Qwen3.8-27B-PSQ, greedy, 投機 OFF, features=5120/bf16）

変更前後（同一 harness、context=64 / 2048）:

```text
context   metric     before      after
64        prefill    6.93 s      0.202 s
64        decode     1.59 tok/s  7.50 tok/s
2048      prefill    218.7 s     1.494 s
2048      decode     0.38 tok/s  7.49 tok/s
```

参考: TP=1（単一 GPU）の non-spec decode は context=2048 で 27.55 tok/s。
TP=2 decode は context 非依存（7.5 tok/s）で、attention ではなく固定 per-token
コストが支配的になった。

## 残り（Phase A スコープ外）

`KERNEL_TRACE_FALLBACK` には引き続き KV_APPEND / ROPE / GDN conv1d /
L2_NORMALIZE / RMS_NORM / elementwise（SILU / SWIGLU / MUL / SCALE / SPLIT /
SIGMOID）が残る。これらは Phase A の対象外。GDN recurrence 自体は optimized で
あるため、Phase B（GDN decode1 の TP=2 開放）は今回の decode ボトルネックでは
ないと判断し、見送った。

## 変更ファイル

```text
src/phaseshift/models/qwen35/runtime/linear_selector.cpp
tests/unit/test_qwen35_linear_selector.cpp
src/phaseshift/models/qwen35/runtime/paged_attention_selector.cpp
tests/unit/test_qwen35_paged_attention_selector.cpp
tests/kernels/optimized/test_paged_attention.hip
src/phaseshift/models/qwen35/runtime/gdn_recurrence_selector.cpp
tests/unit/test_qwen35_gdn_recurrence_selector.cpp
tests/kernels/optimized/test_gemm_bf16_wmma.hip
tests/kernels/optimized/test_gemm_psq4_w4a8_wmma.hip
tests/kernels/optimized/test_gemm_psq8_w8a8_wmma.hip
tests/kernels/optimized/test_gdn_recurrence.hip
```
