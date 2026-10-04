> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# TP-Exec-2: TP=2 Optimized Shape Coverage

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

同一 harness（context=64 / 2048、new_tokens=32）での変化:

```text
context   metric     before     round1      round2      round3
64        prefill    6.93 s     0.202 s     0.0589 s    0.0605 s
64        decode     1.59       7.50        36.54       37.74 tok/s
2048      prefill    218.7 s    1.494 s     0.573 s     0.573 s
2048      decode     0.38       7.49        35.62       36.83 tok/s
```

round2 後は TP=1 を上回る:

```text
context   metric     TP=1        TP=2 (round3)
64        prefill    0.0587 s    0.0605 s
64        decode     28.05       37.74 tok/s
2048      prefill    0.870 s     0.573 s
2048      decode     27.48       36.83 tok/s
```

## 第3回: GDN decode1 の TP=2 開放（Phase B step1）

### 背景

`gdn_recurrence_decode1_supported()` は

```cpp
args.key_heads == 16u && args.num_v_heads == 48u
```

と TP1 geometry のみを許可していたため、TP=2（8/24）は selector が
Optimized でも rows=1 の専用 `WmmaDecode1` ではなく通常の `Wmma` に落ちていた。

一方 `phaseshift_qwen35_gdn_recurrence_wmma_decode1` は `key_heads` /
`num_v_heads` / `repeat` を実行時に計算しており geometry 自体は汎用である
（`repeat = num_v_heads / key_heads` は TP1・TP2 とも 3 で同じ）。

### 変更

- `src/phaseshift/models/qwen35/kernels/optimized/gdn/recurrence.hip` の
  `gdn_recurrence_decode1_supported()` に `(8, 24)` を追加。
- `tests/kernels/optimized/test_gdn_recurrence_decode1.hip` に
  `Shapes{key_heads=8, num_v_heads=24, head_k=128, head_v=128}` を追加し、
  `run_sequence`（nreq=1/4, steps=1/16, zero-inputs 含む） /
  `run_serial_equiv`（rows=2/4/8）/ `run_multirow_equiv`（rows=2/8）を実行。

`decode1` は production lossy WMMA と bit-exact で一致してから開放している。

### 結果（A/B, 同一 harness）

```text
context   decode      before (generic)   after (decode1)
64        36.54       37.74 tok/s        +3.3%
2048      35.62       36.83 tok/s        +3.4%
```

generic 34.85us → decode1 14.70us は 48 の GDN 層で約 −0.96ms/token 相当であり、
observed（27.2ms → 26.5ms/token）と整合する。prefill は不変。

- correctness gate 7/7 PASS（TP=1 vs TP=2 greedy token 一致、GDN recurrence 全テスト）。
- required acceptance 124/124 PASS。

## 第2回: 残りの correctness fallback（rope / kv_append / l2 / gdn_conv / rmsnorm / elementwise）

round1 後も `KERNEL_TRACE_FALLBACK` が残っていたため、各 dispatch に
`PHASESHIFT_SELECTOR_DEBUG` による一時的な入力出力を入れて実測し、
測定後にその debug 出力を除去してから rule を追加した。

実測した TP=2 の selector 入力（enum 値は `ValueDType{BF16=0, F32=1}`、
`RmsNormSelectorWeightLayout{None, PerFeature, PerGroup}`、
`ElementwiseProfile` の enum 順に注意）:

| selector | 追加した rule |
|---|---|
| rope | `F32->BF16, features=3072 / 512, head_dim=256, rotary=64, rows 1..2048` |
| kv_append | `BF16, kv_heads=2, head_dim=256, page=16, rows 1..2048` |
| l2_normalize | `BF16->F32, features=1024, group=128, rows 1..2048` |
| gdn_conv | `BF16->F32, conv_dim=5120, history=3, rows/requests 1..2048` |
| rmsnorm | `F32->F32 PG mode=1 F=3072 G=128`、`BF16->F32 PG mode=0 F=512 G=256`、`F=3072 G=256` |
| elementwise | `Swiglu 8704` / `SiluB2F 3072` / `SiluF2B 5120` / `Sigmoid 3072` / `Mul 3072` / `Scale 1024` / `SplitBf16InterleavedHeads 3072 aux=256` |

いずれも TP=1 の既存 rule がちょうど倍半分になった値に対応し、
実行時に観測した組合せだけを追加している。

検証:

- selector unit test 6 件（`rope` / `kv_append` / `l2_normalize` / `gdn_conv` /
  `rmsnorm` / `elementwise`）に TP=2 の positive と境界（rows 2049 → Correctness）を追加。
- kernel test 6 件（`test_rope` / `test_kv_append` / `test_l2_normalize` /
  `test_gdn_conv1d` / `test_rmsnorm` / `test_elementwise`）に TP=2 の実 shape を追加。
  `test_rmsnorm` は PER_GROUP の場合 `Config.weight_elements` に
  `G` を与えないと weight ベクタが空になりクラッシュするため、この点を明示した。
- runtime（`PHASESHIFT_LINEAR_DEBUG=1` / `PHASESHIFT_QWEN35_KERNEL_TRACE=1`）:
  `KERNEL_TRACE_FALLBACK` が **0 件**、`LINEAR_SELECT_MISS` 0、
  paged-attention の envelope 警告 0。
- correctness gate 4/4 PASS、required acceptance 124/124 PASS。

## 第4回: decode unroll / Prefill2D tile sweep（step2〜4、変更なし）

`phaseshift-bench gemm` の `--psq4-unroll` / `--psq8-unroll` /
`--psq4-prefill2d` / `--psq8-prefill2d` を使い、TP=2 の実 shape を測定した。
いずれも **既定の auto が全形状で最優勝**であったため selector は変更していない。
5反復（決定用の 3072 は20回・交互ペア）の p50 中央値で、値は µs。

### step2: PSQ4 decode1 unroll（rows=1）

| N:K | auto | u=2 | u=4 | u=8 | u=16 | auto の値 |
|---|---:|---:|---:|---:|---:|---|
| 512:5120 | 8.56 | 17.03 | 12.96 | 9.66 | **8.51** | 16（=最適） |
| 3072:5120 | 13.87 | 26.05 | 18.01 | 13.92 | **13.72** | 8（u16 が −2%） |
| 5120:3072 | **13.00** | 18.39 | 14.31 | 13.25 | 13.53 | 8（=最適） |
| 5120:5120 | 26.97 | 31.25 | 27.66 | 26.96 | 28.18 | 8（=最適） |
| 5120:8704 | **43.23** | 46.08 | 43.52 | 43.24 | 43.76 | 8（=最適） |
| 8704:5120 | **43.00** | 47.10 | 43.93 | 43.00 | 44.26 | 8（=最適） |

追加で閾値候補を測定:

```text
N=2560 (K=4096/5120/9216)  u8 の方が 1.0〜2.8% 速い
N=4096 (K=2560/5120)       u8 の方が 1.3〜3.7% 速い
N=3072 (K=5120)            20回ペア測定で u16 が 19/20 勝ち、中央値差 −0.72%
```

`out_features` 単独では 2560→u8 / 3072→u16 と非単調になるため、閾値では表現できない。
また 3072 の優位は 0.14 µs/13.9 µs（0.7〜2%）で、該当 shape は token あたり16回
出現するため E2E への影響は約 0.004% と測定誤差以下。**変更していない**。

u=2 は全形状で −8〜−98%、u=4 も全面的に劣後。既定は u=16（N≤2048）/ u=8（それ以外）のまま。

### step3: PSQ8 decode1 unroll（rows=1）

現在 `psq8_decode1_unroll()` は shape を見ずに 8 を返すが、これが最適だった。

```text
3072:5120   u2 34.84  u4 30.88  u8 30.52  u16 30.36   → 差 0.5% 以下
5120:3072   u2 33.63  u4 30.77  u8 30.42  u16 30.54   → 8 が最良
5120:5120   u2 52.14  u4 48.30  u8 47.58  u16 47.66   → 8 が最良
5120:8704   u2 78.02  u4 77.67  u8 77.63  u16 77.61   → 実質同値
```

**変更なし。**

### step4: Prefill2D tile（rows=2048 = ctx2048 prefill）

PSQ4（cfg 0=64x64 / 1=K128N64 / 2=K64N128 / 3=K128N128 / -1=auto）:

```text
N:K          auto      c0       c1       c2       c3     最良
3072:5120    451.9   493.5    485.8    461.0    453.9    auto(=c3)
512:5120      77.3    80.8     77.2     93.9     83.3    auto(=c1 K128N64)
5120:3072    467.9   508.5    500.9    472.9    467.9    auto(=c3)
5120:5120    761.2   826.2    819.3    769.0    765.1    auto(=c3)
5120:8704   1393.0  1468.6   1437.5   1423.0   1394.8    auto(=c3)
8704:5120   1409.8  1487.3   1452.5   1433.4   1413.9    auto(=c3)
```

PSQ8（cfg 0=64x64 / 1=K64N128 / 2=K128N128 / -1=auto）:

```text
N:K          auto      c0       c1       c2     最良
3072:5120    458.1   476.0    460.3    457.5    実質 auto=c2
5120:3072    470.6   491.3    473.0    471.7    auto(=c2)
5120:5120    769.7   793.1    771.9    769.5    実質 auto=c2
5120:8704   1433.6  1451.6   1436.3   1434.3    auto(=c2)
```

本番 selector と bench auto が一致することも確認済み
（`kPsqGemmPrefill2dMinOutFeatures = 512` なので 512:5120 でも両者とも K128N64、
それ以外は両者とも K128N128）。**変更なし。**

### 第4回の結論

「TP2 geometry が未チューニング」という仮説は、
**decode1 unroll / Prefill2D tile という3つの knob については成立しない**。
既定値は TP2 実 shape で既に最良（または 0.7% 以内の僅差）。
残る価値は step5 の TP combine 実測と step6 の collective 削減 / overlap にある。

### 現在残るもの

`KV_APPEND / ROPE / GDN conv1d / L2_NORMALIZE / RMS_NORM / elementwise` の
correctness fallback は 0 で、GDN decode1 も TP=2 へ開放済み。
`RcclTpTransport` は従来どおり未導入。

次の候補（未着手）:

1. TP combine の総時間計測（最大 64 層 × 2 = 128 回/token、rows=1 で
   約 88us/combine なら約 11ms/token の見込み）。
2. その後、collective 数削減（2 collective/layer → 1 など）と
   compute-communication overlap。decode の 26.5ms/token に対し
   collective 推計 11ms は約 4割で、現時点で最大の残課題。
3. RCCL backend（`RcclTpTransport`）。

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
src/phaseshift/models/qwen35/runtime/rope_selector.cpp
tests/unit/test_qwen35_rope_selector.cpp
src/phaseshift/models/qwen35/runtime/kv_append_selector.cpp
tests/unit/test_qwen35_kv_append_selector.cpp
src/phaseshift/models/qwen35/runtime/l2_normalize_selector.cpp
tests/unit/test_qwen35_l2_normalize_selector.cpp
src/phaseshift/models/qwen35/runtime/gdn_conv_selector.cpp
tests/unit/test_qwen35_gdn_conv_selector.cpp
src/phaseshift/models/qwen35/runtime/rmsnorm_selector.cpp
tests/unit/test_qwen35_rmsnorm_selector.cpp
src/phaseshift/models/qwen35/runtime/elementwise_selector.cpp
tests/unit/test_qwen35_elementwise_selector.cpp
tests/kernels/optimized/test_rope.hip
tests/kernels/optimized/test_kv_append.hip
tests/kernels/optimized/test_l2_normalize.hip
tests/kernels/optimized/test_gdn_conv1d.hip
tests/kernels/optimized/test_rmsnorm.hip
tests/kernels/optimized/test_elementwise.hip
src/phaseshift/models/qwen35/kernels/optimized/gdn/recurrence.hip
tests/kernels/optimized/test_gdn_recurrence_decode1.hip
```
