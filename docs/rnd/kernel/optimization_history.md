> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# 単独 kernel 最適化履歴（RoPE 等）

`docs/rnd/optimization_findings.md` から分離した、他 topic に属さない単独 kernel の実験履歴。見出し番号（7.x）は元 dump の通し番号を保持する。なお §7.80 は元 dump 内で RoPE Phase 2 と FP8/MXFP4 計測の 2 箇所に重複して付番されていた（FP8/MXFP4 側は quantization 履歴に収録）。

## 収録セクション

- 7.79 RoPE standalone: head_dim=256/rotary=64 専用 pair kernel + head loop パイプライン化
- 7.80 RoPE Phase 2: GPU inv_freq table（hot path からの pow 除去）
- 7.99 standalone kernel の高速化は in-context wall に現れないことがある

---

## 7.79 RoPE standalone: head_dim=256/rotary=64 専用 pair kernel + head loop パイプライン化（2026-09-20）

### 背景

RoPE optimized（F32→BF16、theta=1e7）は correctness fallback と bit-exact であることが
必須（double pow + double sincos の結果を BF16 へ落とした値が既存結果と bit 一致）。
float trig 置換・近似・recurrence は禁止。

generic kernel（`rope_f32_bf16_kernel`、256 threads / 8 waves）の課題:

- `head_col = idx % head_dim`（runtime modulo、逆数ベース整数除算列）
- rotary 64 elements に対して x load 64 + partner load 64 = 128 float loads/head
- cos/sin を LDS（`s_cos` / `s_sin`）に置くため `__syncthreads()` 必須
- Q rows=1 p50 ≈ 10.6 µs

### Phase 1: 専用 pair kernel

`rope_f32_bf16_pair_kernel`（128 threads / 4 waves）を追加。
適用条件は `head_dim == 256 && rotary == 64 && features % 256 == 0` と
pointer / stride の alignment（F32 in: 8B aligned + even stride、BF16 out: 4B aligned + even stride）。

- tid 0..31（rotary wave）: prologue で double trig を 1 回だけ計算し、
  cos/sin を **VGPR に保持**（LDS なし、barrier なし）。head ごとに
  `a = input[tid]`、`b = input[tid + 32]` を読み、
  `y0 = a*cs - b*sn`、`y1 = b*cs + a*sn` を計算。
  rotary の input traffic を 128 loads/head → 64 loads/head に削減。
- tid 32..127（non-rotary wave）: head ごと 192 elements を 96 threads × 2 で担当。
  `float2`（8-bit load）+ BF16 2 個を uint32 に pack して 32-bit store。
  trig を一切実行せず、rotary wave と待たずに head loop を開始。

選択: `launch_rope_f32_bf16` が条件を満たせば pair、でなければ generic。
`launch_rope_f32_bf16_generic` は public として残存（bench A/B と fallback）。
alignment は host 側の shape/pointer チェックで保証（workspace 値は 256B aligned、
F32 row_stride は 4 の倍数 / BF16 は 8 の倍数（`aligned_row_stride`））。
kernel 内の runtime branch は無し。

#### 丸め順序の罠

`a*cs - b*sn` の FMA 縮約はコンパイラが文脈によって別形を選ぶ
（2 積のうちどちらを先に丸めるかが逆転）。rows=128（786432 要素）で
BF16 1-2 要素が 1 ULP ずれたため、generic kernel の ISA と同一の
演算順序（partner 積を先に RN、主積は FMA 内で丸めなし）を
`__fmaf_rn(x, cs, ±__fmul_rn(p, sn))` で明示固定した。

#### ISA 検証（gfx1201）

- `ds_*` / `s_barrier_*` / `v_mul_hi_u32`（modulo）: **0**
- `group_segment_fixed_size = 0`、spill なし、VGPR 34 / SGPR 30
- pow / sincos の f64 ソフト実装は prologue のみ（rotary wave が実行）。
  cos/sin は head loop 中に再計算・再ロードされず VGPR で跨ぎ保持

#### correctness

- pair vs generic の bit-wise 比較: **mismatch 0**
  （rows 1/2/16/128 × features 1024/6144 × positions {0,1,2,31,128,1000,8191,32767}）
- 既存 case（correctness fallback 比較含む）も全 mismatch 0

### Phase 1 単体の測定: rows=1 では高速化せず

| rows | Q (f=6144) generic→pair | K (f=1024) generic→pair |
| --- | --- | --- |
| 1 | 9.84 → 10.08 µs（0.98x） | 6.06 → 5.92 µs（1.02x） |
| 128 | 19.24 → 18.13 µs（1.06x） | 14.36 → 13.54 µs（1.06x） |
| 512 | 49.96 → 42.19 µs（1.18x） | 39.06 → 36.31 µs（1.08x） |

ISA で原因を確認: head loop が反復ごとに
`global_load → s_wait_loadcnt 0x0（その場で待機）→ cvt → store` となり、
Q の 24 反復が全て full load latency（~450cy）を serial に支払う
（~5µs の wall）。f64 trig prologue（~200 f64 ops、CU 単 pipe）は
generic と同一の ~2µs 級固定コスト。LDS/barrier 除去の効きは rows ≥ 128 から。

### Phase 1b: head loop パイプライン化

wave 分岐を loop 外へ hoist + `#pragma unroll 4`。コンパイラが
software pipelined chunk を生成:

- non-rotary: `global_load_b64` 4 連続発行（offset -2048/-1024/0/+1024、
  末尾は次 chunk の prefetch 相当）、その後各要素は**自分の load のみ**を
  待って変換 + store（`s_wait_loadcnt 0x3 → 0x2 → 0x1 → 0x0`）。
  残りの load は in-flight のまま。
- rotary: `global_load_b32` 8 連続発行 + 同様の staged wait。
- 余り（`heads % 4`）は別 loop。VGPR 34 で不変、spill なし。

correctness: bit-wise 比較を再実行し mismatch 0 を維持。

### 最終性能（rows=1、warmup 100 / samples 200 / launches 20）

| shape | generic | pair（Phase 1） | pair（Phase 1b） | speedup |
| --- | ---: | ---: | ---: | ---: |
| Q F=6144 | 10.07 µs | 10.08 µs | **6.51 µs** | **1.55x** |
| K F=1024 | 6.35 µs | 5.92 µs | **5.69 µs** | **1.12x** |

| rows | Q generic→pair（1b） | K generic→pair（1b） |
| --- | --- | --- |
| 128 | 19.13 → 14.26 µs（1.34x） | 14.32 → 12.89 µs（1.11x） |
| 512 | 49.99 → 37.94 µs（1.32x） | 39.14 → 35.39 µs（1.11x） |

rows=1 の残りの床（K 5.69 µs）は f64 trig prologue + launch overhead。
次の削減対象は plan 既定の Phase 2（GPU inv_freq table で pow 除去、
f64 prologue ~75% 削減）と Phase 3（position trig cache で f64 全除去、
VRAM = max_pos × 32 × sizeof(float2)）。

### E2E（27B-PSQ）

- tg128: 23.23 → 23.39 t/s（中央値 43.05 → 42.75 ms/token、+0.7%）。
  rope は 4192 dispatch × ~6.5 µs ≈ 0.3 ms/token と decode の ~0.8% なので、
  E2E への取り分は期待通り小さい。
- pp2048: 1996.7 t/s（baseline ~1995、regression なし）
- required acceptance 72/72 PASS（0 skipped）、KERNEL_TRACE_FALLBACK 0、
  `GREEDY_TOKEN_SUM=2649738`（Phase 1 前と同一で決定的）


---

## 7.80 RoPE Phase 2: GPU inv_freq table（hot path からの pow 除去）（2026-09-20）

### 変更

- `rope_inv_freq_init_kernel`（32 threads）を追加。既存 hot kernel と**同一の
  double 表現** `1.0 / pow(theta, (double)i / (double)32)` で
  `double inv_freq[32]`（256 B）を GPU 上で 1 回だけ生成。
  CPU libm は bit pattern が異なるため CPU precompute は不可、GPU 生成が条件。
- `rope_f32_bf16_pair_kernel` の trig prologue を
  `freq = inv_freq[tid]`（1 回の global_load_b64）+ `angle = (double)pos * freq`
  + `sincos(angle)` に変更。hot kernel からは pow が完全除去。
- `launch_rope_f32_bf16_pair` / `launch_rope_f32_bf16` に `const double* inv_freq`
  を追加（pair 選択の条件に `inv_freq != nullptr` を追加。nullptr の場合は
  generic へ安全にフォールバック）。
- `HostExecutionContext` に `rope_inv_freq` フィールドを追加。
  Executor / MtpExecutor が `gpu::Tensor rope_inv_freq`（32 doubles、arena）を
  所有し、model setup 時に `upload_stream` 上で init kernel を 1 回 launch
  （末尾の `hipStreamSynchronize` で完了保証）。shutdown / move 経路に解放を
  追加。theta は `model.text_config().rope_theta`（`<= 0` の場合 1e7、
  lowering と dispatch の正規化と同一）。

### correctness

- test_rope 35/35 PASS:
  - correctness 比較 11 case（auto launcher、pair 実選択）mismatch=0
  - **freq identity 2 case**: init kernel が生成した表の各要素を、別 TU の
    verify kernel が同一 pow 表現で再計算し bit 比較（θ=1e7 / 5e5）→ mismatch=0。
    表の double がインライン pow の bit pattern と全一致であることを直接検証。
  - bit-exact（pair vs generic）22 case mismatch=0
- E2E: `GREEDY_TOKEN_SUM=2649738`（Phase 1 と同一で決定的）、
  KERNEL_TRACE_FALLBACK 0、required acceptance 83/83 PASS（0 skipped）

### ISA（pair kernel）

- f64 命令: Phase 1（pow + sincos）~590 件 → Phase 2（sincos のみ）~107 件。
  すべて prologue に集中（head loop 以降 f64 命令 0 件）。
- 消えた命令（pow の exp/log 実装 + 64-bit 除算）:
  `v_div_scale_f64`（6→0）、`v_rcp_f64`（5→0）、`v_div_fmas_f64`（3→0）、
  `v_div_fixup_f64`（3→0）、`v_frexp_mant/exp`（4→0）、`v_trunc_f64`（4→0）、
  `v_ldexp_f64`（9→2、残 2 件は sincos 内部）。
- 残る f64: `v_trig_preop_f64` ×3 + sincos 多項体（~100 件）+
  `v_mul_f64` ×1（angle）+ `v_cvt_f32_f64` ×2（cos/sin → float）。
- trig prologue の実体:
  `v_cvt_f64_u32`（pos→double）→ `global_load_b64`（inv_freq[tid]）→
  `s_wait_loadcnt 0x0` → `v_mul_f64`（angle）→ sincos。
- head loop は Phase 1b と同様の software pipeline
  （8×global_load_b32 連続発行 + staged wait `0x7 → 0x0`、4× unroll）。
  bit-exact の丸め順序（partner 積を RN、主積は FMA 内で丸めなし）は
  `v_mul_f32` + `v_dual_fmac_f32`（RDNA4 packed FMA）で維持。
- メタデータ: VGPR 34 / SGPR 29 / private 0 / group_segment_fixed_size 0 /
  kernarg 48。LDS・barrier なし（不変）。

### 性能（warmup 100 / samples 200 / launches 20、p50）

| rows | Q f=6144 generic→pair(2) | K f=1024 generic→pair(2) |
| --- | --- | --- |
| 1 | 10.11 → **5.27 µs**（1.92x） | 6.39 → **4.25 µs**（1.50x） |
| 128 | 17.37 → 6.85 µs（2.54x） | 12.62 → 5.67 µs（2.23x） |
| 512 | 49.35 → 18.27 µs（2.70x） | 38.95 → 18.32 µs（2.13x） |

rows=1 の詳細（p10/p50/p90）:

| shape | generic | pair(1b) | pair(2) |
| --- | --- | --- | --- |
| Q f=6144 | 9.87 / 10.11 / 11.03 | 6.25 / 6.51 / 7.12 | **4.94 / 5.27 / 5.64** |
| K f=1024 | 6.08 / 6.39 / 6.77 | 5.37 / 5.69 / 6.14 | **4.04 / 4.25 / 4.37** |

Phase 1b 比: Q 1.23x、K 1.34x。pow 除去で rows=1 は ~1.2-1.4 µs 削減。
rows が増えるほど効きが大きい（各 row = 1 block で pow を 1 回ずつ払っていた）。

### E2E（27B-PSQ）

- tg128 3 連測: 42.75 / 42.73 / 42.77 ms/token（中央値 42.75、23.39 t/s）。
  Phase 1b と同一レベル（regression なし）。rope の削減量
  （~2 µs × 4192 dispatch ≈ 0.08 ms/token）は decode の ~0.2% で
  単回測ではノイズ内に吸収される。
- pp2048: 1995.6 t/s（Phase 1b 1996.7、ノイズ範囲）

### 残っているボトルネック / Phase 3 判断

rows=1 pair 5.27 µs の内訳（推定）:

- launch overhead + event 計測の床: ~2-2.5 µs
- head loop: 24 heads の load latency 連鎖（4 waves しかなく隠蔽が限定的）
- sincos prologue: ~0.5 µs 以下（f64 ~107 件）

sincos は rows=1 でも**もはや支配的でない**。Phase 3（position trig cache、
VRAM = max_pos × 32 × 16 B = 32768 場合 16 MB）で得られる上限は
prologue の ~0.5 µs / dispatch で、E2E 影響は <0.1%。
plan のゲート（「sincos がまだ支配的である場合のみ」）を満たさないため
**Phase 3 は実施しない**方針。


---

## 7.99 standalone kernel の高速化は in-context wall に現れないことがある（2026-09-22）

Gate 11H で、BF16 ExactRows（GDN `in_proj_a/b`、N=48, K=5120, rows=8）の K ループを
2 tile ずつ accumulate する bit-exact 変種を試した。加算順は tile 昇順で完全に同一。

| unroll | standalone p50 | Verify wall（M=8, ctx2048, capture off, interleaved 4 対） |
| ---: | ---: | ---: |
| 1（現行） | 10.91 us | 41511 us（median） |
| 2 | **8.28 us（-24%）** | 41556 us（+45 us） |
| 4 | 22.17 us（register 圧迫） | - |

standalone では -24% だが、in-context の Verify wall は改善しない（むしろ微増）。
理由は、これらの kernel が前段の出力を L2-hot で読み latency が小さく、削った分が
launch gap に吸収されるため。**primitive 単体の bench だけで採用を決めてはならない**。

この効果は kernel の大きさ依存である。attention split reduce（1 WG = (row, kv_head)、
grid 32 WG、M 非依存で 607-622us）は grid が律速だったため、standalone の 3.98x が
そのまま Verify wall -0.33ms になった。**単体 bench の改善が wall に出るかは、
その kernel が launch floor 由来か occupancy 由来かで決まる。**

### 計測コマンド

```bash
PHASESHIFT_BF16_EXACT_UNROLL=1 ./build/phaseshift-bench gemm --dtype bf16 \
  --shapes 48:5120 --rows 8 --variant gemm --impl exact --samples 30 --warmup 50
```
