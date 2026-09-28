> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# W4A4 R&D 記録

PSQ4 系の量子化フォーマットで「K32 group ごとの任意 scale を runtime から消す」ための
調査記録。結論を先に書く。

## 結論

**K32 group ごとの任意 scale を与える限り、inner loop から scale 乗算は消せない。**
精度を保ったまま scale を粗くする（K64/K128/K256 で共有する、低 rank 化する、
2 枚の I4 に分解する、E4M3 residual にする）方向は、いずれも実測で不成立だった。

**ただし、そもそも scale 乗算は律速ではない。** main tree の §7.23（2026-09-14、
DCE 制御済み probe）が「scale 作業を完全にゼロにしたときの天井」を実測しており、
それは **1.12x** である。つまり scale 適用はカーネルの **11%** しか無く、
これを消しても 12% しか速くならない。

本ドキュメントの §4・§5（部分積 batching、batched dot）が効果ゼロだったのは、
命令数の削減には成功したが**削減対象が律速ではなかった**ためである。
両者は独立に同じ結論へ到達した。

| 方向 | 結果 | 根拠 |
| --- | --- | --- |
| M7 を K64/K128/K256 で共有 | 不成立 | rel_rms +6%（K64）で KLD が 2.9% -> 7.3% に増幅 |
| dual-I4 分解（2 枚の I4 WMMA） | Gate 不合格 | 表現力は十分（上界 0.0139）だが HALO >= 95% が必要 |
| E4M3 residual | 不成立 | bounded-uniform な residual に指数分布 level は不適合（SSE 2.6 倍） |
| group 軸 SmoothQuant | 不成立 | scale に rank-1 構造が無い（p50 0.21 bit = 1.16 倍） |
| 部分積 batching | 命令数不変 | `mad/WMMA` が全 variant で 8.0 のまま |
| batched dot（`v_dot2_f32_bf16`） | 不成立 | BF16 pack が 1 変換 5 命令、削減分の 3 倍 |
| scale fold（E8M0 / eighth） | 却下済み（main tree §7.23） | 天井 1.12x、format 変更を正当化できない |

## 律速はどこか

main tree §7.23 の実測（PSQ4 W4A8、n=17408 k=5120、rows=2048）:

| 構成 | 時間 | 比 |
| --- | --- | --- |
| 現行（per-32 bf16 scale + FMA） | 2908us | 1.00x |
| scale 作業ゼロ probe（天井） | 2590us | **1.12x** |
| FOLD E8M0 | 2644us | 1.10x |

**k-loop 本体（decode perm + WMMA issue + load）が支配的**である。
よって scale 粒度の変更は「11% の取り分」の話にしかならない。

## 現行構造（参考: 命令ミックス）

PSQ4_ADV(INT7+Po2) の inner loop:

```text
for blk:
    for sub (K32 group):
        WMMA  -> c
        acc[j] = v_mad_i32_i24(c[j], M7, acc[j])   x 8
```

1 WMMA は 8192 MAC を 1 issue で処理するが、直後の `v_mad_i32_i24` が
8 issue を占める。命令ミックスとしては scale が 8/9 を占めるが、
**実測の律速はそこではない**（上記 1.12x）。`v_mad_i32_i24` は
WMMA の int32 出力をそのまま食える（変換ゼロ）ため、latency 面でも有利。

## 各方向の詳細

### 1. scale を span で共有する（§7.23）

`W = q * M7 * 2^E` の M7 を K64/K128/K256 で共有すると、inner loop の
mad は 8/span まで落ちる（K128 なら 8 mad / 4 WMMA）。

実測（Qwen3.5-4B、2 layer、0.226 G params、`wq-eval`）:

| format | bpw | rel_rms | 対現行 |
| --- | --- | --- | --- |
| psq4_adv_int7po2（現行） | 4.500 | 0.08462 | — |
| z4_po2_m7_64 | 4.484 | 0.08973 | +6.0% |
| z4_po2_m7_128 | 4.430 | 0.09953 | +17.6% |
| z4_po2_m7_256 | 4.402 | 0.10859 | +28.3% |
| z4_po2_32（M=1） | 4.375 | 0.10329 | +22.1% |

**KLD（QDQ shadow、4B、1020 positions）:**

| format | KLD mean | ppl | ppl 相対 |
| --- | --- | --- | --- |
| psq4_adv_int7po2 | 0.04022 | 3.5411 | +2.90% |
| z4_po2_m7_64 | **0.07519** | 3.6933 | **+7.32%** |

weight の rel_rms は +6.0% だったが、**KLD は 1.87 倍**まで悪化した。
weight 誤差は E2E で増幅されるので、weight rel_rms の小さな差を許容と
判断してはいけない。

### 2. dual-I4 分解（§7.24）

`W = CB10(code) * BF16_scale` を teacher とし、

```text
student: W ~= 2^E * (q0 + q1 / 2^r)     q0, q1 は signed INT4
```

へ factorize する。保存は 4bit code 1 個 + K32 selector + exponent のみ。

| control | 内容 | bpw | rel_rms | HALO% |
| --- | --- | --- | --- | --- |
| A `dense_upper` | q0/q1 自由（bit 制約無視） | 8.00 | **0.0139** | 100% |
| B `codebook_pair` | (T,code)->(q0,q1) learned | 4.16 | **0.0925** | 99% |
| C `direct_q0_e32` | q0 = signed4(code) 固定 | 4.41 | 0.1140 | 100% |
| （参考）production PSQ4 -> BF16 | | 4.50 | 0.0862 | — |

**表現力は十分**（上界 0.0139 ≪ 0.0862）。しかし 4bit code + selector の
制約下で production 精度を保つには **HALO >= 95%** が必要で、理想上限は 415 TF。
HALO penalty を入れると急峻に崩れる:

| λ | rel_rms | HALO% | 理想 WMMA |
| --- | --- | --- | --- |
| 0 | 0.0932 | 99.4% | 1.99 |
| 1e-4 | 0.1353 | 41.2% | 1.41 |
| 1e-3 | 0.1672 | 25.2% | 1.25 |

**E4M3 residual は I4 residual に負ける。** same base / same selector の
統制比較（T=32）で `i4_e4m3_fullk` 0.152 vs `direct_q0_fullk` 0.135。

理由は residual の分布にある。`q0 = clamp4(x/unit)` の後の residual は
**bounded-uniform**（99% が [-0.49, +0.49]）で wide dynamic range ではない。
同じ 16 level 予算の統制実験（200k synthetic weights）で
I4 uniform SSE = 3.2e3 に対し E4M3 free = 8.4e3 と **2.6 倍悪い**。
E4M3 の指数分布 level は 0 付近に無駄に密で端が粗い。

### 3. group 軸 SmoothQuant（§7.25）

`A'[g] = A[g] * D[g]`、`W'[n,g] = W[n,g] / D[g]` は `A W^T = A' (W')^T` が
厳密に成り立つ変数変換。これが成立すれば scale 処理を output 側の
`M x N x K/32` から activation 側の `M x K` へ移せ、`D[g] = 2^e[g]` なら
A4 quantizer の exponent 処理へ fold できる。

成立条件は `S[n,g] ~= 2^Rn * 2^Cg`。実測（12 iter、単位 bit）:

| tensor | rank-1 p50 | rank-1 p95 | po2 p95 | prof32 p95 |
| --- | --- | --- | --- | --- |
| layer0 gate_proj | **0.212** | **0.630** | 0.631 | 0.602 |
| layer0 down_proj | 0.216 | 0.636 | 0.639 | 0.623 |
| layer0 in_proj_qkv | 0.221 | 0.674 | 0.915 | 0.641 |
| layer3 q_proj | 0.218 | 0.656 | 0.661 | 0.631 |

**中央値でも 0.21 bit（1.16 倍）の補正が必要。** 倍率誤差は p95 で 1.55 倍。
判定基準（有望 p95=0.12、厳しい p95=0.5）を大きく外れる。

原因は production PSQ4 の scale 決定則。`quantize_psq4_block` は
`s0 = max_abs / 10` から ±0.5 を探索して **その group 単体の SSE 最小**を
選ぶので、`S[n,g]` は本質的に独立な自由度であり低ランク近似の余地がない。

profile を 32 まで増やしても 0.630 -> 0.602 と 5% しか改善しない。
Po2 制約は悪化要因ではない（rank-1 0.630 vs po2 0.631）。

### 4. 部分積 batching（§7.26）

`Y = sum_g P_g D_g` なので、数式上は 1 WMMA ごとに scale を当てる必要はない。

```text
N 個の WMMA 部分積 -> 1 回の group 軸 ScaleReduce
```

5 variant で実測。ISA が決定的:

| variant | WMMA | mad24 | **mad24/WMMA** | VGPR |
| --- | --- | --- | --- | --- |
| A（現行） mb=1 | 2 | 16 | **8.0** | 41 |
| B2（VGPR 保持） mb=1 | 2 | 16 | **8.0** | 41 |
| B4（VGPR 保持） mb=1 | 4 | 32 | **8.0** | 60 |
| A mb=2 | 4 | 32 | **8.0** | 67 |
| B4 mb=2 | 8 | 64 | **8.0** | 95 |

**`mad24 / WMMA` が全 variant で 8.0 のまま。** 部分積を保持しても
合計 mad24 は 1 つも減らない。batching が変えるのは命令の順序だけ。

性能（n=k=4096 rows=4096 mb=2）: A 204.2 / B2 205.2 / B4 **217.6** /
B2L 203.5 / B4L 97.9 TF。B4 の +6.6% は VGPR 41 -> 95 と引き換え。
LDS 版は往復コストで不利。

### 5. batched dot（§7.27）

gfx1201 で実機確認した dot 系命令:

| builtin | 状態 |
| --- | --- |
| `fdot2_f32_bf16` | **使える** -> `v_dot2_f32_bf16` |
| `sdot2` / `udot2` | 不可（`dot2-insts` が必要） |
| `sdot4` / `udot4` | 不可（`dot1-insts` が必要） |
| `fdot2c_f32_bf16` | 不可（`dot13-insts` が必要） |
| `fdot4_f32_fp8` | clang に存在しない |

`v_dot2_f32_bf16` は `a0*b0 + a1*b1 + c` を 1 命令で行うので、partial と
scale を 2 group 分ずつ BF16 pair に詰めれば 2 group の reduce が 1 命令になる。
group 軸の命令数は K4096 で 1024 -> **512 に半減**する。

しかし実測は 1/3 の速度:

| variant | rows=4096 mb=2 |
| --- | --- |
| A（mad24） | 205.4 TF |
| B4 | 219.7 TF |
| **D2（dot2）** | **65.2 TF** |

ISA が原因を説明する:

| kernel | 総命令 | mad24 | dot2 | cvt_f32_i32 |
| --- | --- | --- | --- | --- |
| A mb=1 | 95 | 16 | 0 | 8 |
| **D2 mb=1** | **315** | 0 | 16 | **36** |

D2 の内訳: `v_bfe_u32` 36 / `v_or_b32` 36 / `v_add3_u32` 36 /
`v_cndmask_b64` 33 / `v_cmp_u_f32` 31 / `v_dot2_f32_bf16` 16。

`__float2bfloat16` が **1 変換あたり約 5 命令**の bit 操作（丸め処理込み）に
展開される。32 変換で約 150 命令になり、削減した 16 mad24 をはるかに上回る。

## なぜすべての方向が失敗するのか

production PSQ4 の scale は group 単体の SSE 最小化で決まる。したがって

- row と group の積で近似できない（rank-1 不成立、p50 0.21 bit）
- group 間で共有すると必ず精度が落ちる（KLD 1.87 倍）
- residual は bounded-uniform なので整数格子が最適（E4M3 は 2.6 倍悪い）

**scale の自由度を削る方向は、どれも E2E で増幅されて破綻する。**
加えて、そもそも scale は律速ではない（天井 1.12x）。つまり
**精度を犠牲にしても取り分が無い**という二重の意味で成立しない。

## OCP_FP8 経路への移植性

「OCP_FP8」= 通常の FP8（E4M3/E5M2）weight 経路。main tree の実装は:

- `gemm_psq4_w4a8_wmma.gfx1201.hip`（CB10 weight を E4M3 へ展開 + A8 E4M3）
- `gemm_psq8_w8a8_wmma.gfx1201.hip`（I8 affine weight + A8 E4M3）
- WMMA: `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`
- 27B-FP8 のロード経路がこれを使う

### 構造は同一

```cpp
c = wmma_fp8(a0, w0, c);
c = wmma_fp8(a1, w1, c);
for (j = 0; j < 8; ++j) acc[t][j] = fmaf(c[j], ws, acc[t][j]);   // 8 FMA / 2 WMMA
```

**PSQ4_ADV の `8 mad / WMMA` と同じ形**である。main tree の §7.23 の
命令数記録でも `gemm_rocmfp4_w4a8` は「180 命令 / 8 WMMA、46 が scale FMA」、
`gemm_rocmfp8_w8a8` は「163 命令 / 8 WMMA、34 f32 scale + 32 cvt」とある。

### 移植できるか: 結論は「できるが、移植する価値が無い」

| 本ドキュメントの知見 | OCP_FP8 経路への適用 |
| --- | --- |
| scale を K64/K128/K256 で共有 | **同じく精度が落ちる**（scale 決定則が同じため） |
| rank-1 SmoothQuant | **同じく不成立**（scale は group 単体最適で決まる） |
| dual-I4 分解 | FP8 WMMA では 2 枚目が FP8 になり、I4 の 2 枚目と事情が異なる。ただし bounded-uniform residual への指数分布 level という不利は共通 |
| 部分積 batching | **同じく命令数不変**（FP8 でも `fma/WMMA` は変わらない） |
| batched dot | FP8 には `v_dot2_f32_fp8` が**存在しない**ため、より不利 |
| 律速は scale ではない | **そのまま当てはまる**（下記）

**律速の話はそのまま移植できる。** FP8 経路でも scale 適用は
カーネルの一部でしかなく、main tree §7.23 の scale ゼロ probe が
1.12x 天井であることは PSQ4 と FP8 の両方に効く。

さらに参照実装（当時参照した hipfire snapshot の
`gemm_qkv_hfp4g32_wmma_fp8.gfx12.hip`）が独立に同じ結論を書いている:

> The raw FP8 WMMA throughput on gfx1201 is 1.87x FP16-WMMA. But the
> per-32-block UE8M0 scale forces post-WMMA `acc[j] += partial[j] * scale[j]`
> chains on the F32 accumulator. The FMA chain + cross-lane shuffle to gather
> per-output-row scales costs roughly what the WMMA-throughput advantage
> delivers, netting close to parity.
>
> Lifting this would require either (a) baking row_scale * UE8M0 into the FP8
> weight at dequant time (risks E4M3 saturation) or (b) a new quant format with
> row-scale-only (no per-block UE8M0).

つまり **3 者が独立に「per-block scale が FP8 の利得を食い潰す」に到達**している。
ただし main tree §7.23 の実測天井 1.12x は「その FMA を消しても 12% しか
速くならない」ことを示しており、参照実装の見立て（利得がほぼ消える）は
op 数としては正しいが、**そもそも WMMA 自体が律速でない**ため
FMA を消しても parity 前後から動かない、というのが正確な理解である。

### 移植するなら何をすべきか

scale を消す方向ではなく、**k-loop 本体**へ投資する:

1. WMMA の issue 間隔（`s_waitcnt` / dependency）
2. weight の decode（`v_perm` / LUT packing）
3. load（coalescing / LDS staging）
4. MB と occupancy のトレードオフ

これは本ドキュメントの §4（B4 の +6.6%、VGPR 41 -> 95）と同じ結論に行き着く。

### 実測（main tree の build 済みオブジェクトを ISA 解析）

`gemm_psq4_w4a8_wmma.gfx1201.hip` のカーネル本体（gfx1201）:

| kernel | 総命令 | WMMA | scale FMA | decode perm | fma/wmma |
| --- | --- | --- | --- | --- | --- |
| main | 2712 | 96 | 197 | 72 | 2.05 |
| main | 1723 | 80 | 162 | 120 | 2.02 |
| main | 1113 | 40 | 86 | 120 | 2.15 |
| main | 714 | 20 | 46 | 120 | 2.30 |
| splitk | 284 | 10 | 24 | 60 | 2.40 |

**`fma/wmma ≈ 2.0`** で、PSQ4_ADV の `mad/wmma = 8.0` より scale 比率が低い。
FP8 WMMA は K16 なので K32 group あたり 2 WMMA、その直後に 8 FMA だが、
unroll で WMMA が先にまとめて発行されるため実効比が下がる。

一方 **decode perm が 72〜120 命令**と、WMMA と同程度に重い。
これは main tree §7.23 の「律速は k-loop 本体（decode perm + WMMA issue）」
と一致する。

**移植性の結論（実測裏付け）**: FP8 経路では scale 比率が既に 2.0 と低く、
かつ decode perm が支配的である。したがって本ドキュメントの
「scale を消す方向」の知見を FP8 へ移植しても取り分はさらに小さい。
移植するなら decode perm / WMMA issue の側である。

### PSQ4 で f16 dot は使えるか（実測、2026-09-20）

PSQ4 の CB10（`{0,1,2,3,4,6,8,10}`）× 符号）は **全て FP16 で厳密に表現できる**
（main tree §7.24 で「CB10 の全値は E4M3 で厳密」と記録済み。FP16 はさらに広い）。
そこで weight を FP16 として `v_dot2_f32_f16` が使えるかを実機で検証した。

**gfx1201 で使える dot 系命令（実機確認）:**

| builtin | 状態 |
| --- | --- |
| `fdot2`（FP16） | **使える** → `v_dot2_f32_f16`（3*5+4*6=39 で動作確認） |
| `fdot2_f32_bf16` | 使える → `v_dot2_f32_bf16` |
| `fdot4_f32_fp8` | **clang に存在しない** |
| `fdot4_f32_bf8` | **clang に存在しない** |
| `sdot4` / `udot4` / `sdot2` / `udot2` | 不可（`dot1-insts` / `dot2-insts` が必要） |

**FP8 dot は無い。** したがって「PSQ4 は FP8 だから DOT4 が使える」は成立しない。
FP16 dot（`v_dot2_f32_f16`）なら使えるので、そちらを検証した。

PSQ4 相当（A4 int4 × CB10 fp16）で `v_dot2_f32_f16` は数値的に正しく動作する
（3*6+4*8=50 が厳密に出る）。しかし命令数で負ける:

8 accumulator × 64 group の scale reduce:

| 方式 | 総命令 | 積和 | 変換 |
| --- | --- | --- | --- |
| **現行 `v_mad_i32_i24`** | **1104** | 1024 | **0** |
| f16 dot2 | 2055 | 256 | **1440** |

f16 dot2 の変換内訳:

    s_cvt_f32_i32    576   (int32 -> f32)
    s_cvt_f16_f32    576   (f32 -> f16)
    s_pack_ll_b32_b16 288   (2 x f16 -> 1 b32)
    = 1440

`v_dot2_f32_f16` が積和を 4 分の 1 にするが、**変換が 1440 命令**かかる。
丸めが不要な「正確な変換」でも int32 → f32 → f16 → pack の 3 段が必要。

**現行が変換ゼロなのは、`v_mad_i32_i24` が WMMA の int32 出力を
int32 scale で直接食えるためである。** dot 系は必ず FP16/BF16/FP8 を要求するので、
この「変換ゼロ」という最大の利点を壊す。§7.27 の BF16 dot2 と同じ結論。

参考: hipfire の `gemm_gate_up_hfq4g256_dot2.hip` は FP16 activation
（`_Float16` X）を前提にしており、dequantize 済みの FP16 weight との dot に
dot2 を使っている。**activation が最初から FP16 である場合**には変換が
不要なので成立する。PSQ4 の A4（int4）はこれに当たらない。

### A8 activation 側の移植について

A4 activation producer（`activation_quantize_a4`）は A8 producer と
同じ構造（row scale + fragment-order 配置）なので、**A8 側の知見は
そのまま流用できる**。ただし A4 は A8 より scale 粒度が粗い
（INT4 row-scale only）ため、精度は A8 > A4 である。
OCP_FP8 経路は A8 を使うので、**A8 が既にあるなら A4 へ落とす理由は無い**
（W4A4 にしたい動機は WMMA の I32 累算器が速いことだが、§7.23 の
天井 1.12x を踏まえると優先度は低い）。

## 残っている道

1. **k-loop 本体の改善**（WMMA issue / decode / load / occupancy）。
   これが唯一、天井 1.12x を超えられる方向。
2. **A4 activation 側の改善**（row affine + calibrated clipping）。
   weight 側の構造を変えずに精度を稼げる。
3. **現行 int7po2 を production に載せる**。A4 producer は実装済み
   （CPU reference と bit 一致）。loader / dispatch が残り。
   KLD 0.04022 / ppl +2.90% は production PSQ（0.03359 / +2.81%）に近い。

## 再現方法

```bash
# 量子化フォーマットの offline 比較
phaseshift-quantizer wq-eval --input <model> --formats <...> --layers N

# E2E（QDQ shadow、format override で候補を差し替え）
phaseshift-quantizer kld --input <model> --bundle <bundle> --tokens <corpus> \
    --weight-format-override <format>

# scale の rank-1 診断
phaseshift-quantizer smooth --input <model> --tensors <...> --profiles 1,2,4,8,16,32

# dual-I4 分解の offline 探索
phaseshift-quantizer halo-i4 --input <model> --controls <...> --tables N --lambda X

# WMMA 部分積 batching / dot2 の PoC
./build/tests/test_gemm_psq4_scale_matrix_poc
```

実装は branch `exp/psq4-halo-i4matrix`（worktree
`.worktrees/exp_psq4_halo_i4matrix`）にある。

## 参照

- `docs/rnd/quantization/kernel_optimization_history.md` §7.23〜§7.27（詳細な計測記録）
- `docs/PSQ4_ADV_design_v0.1.md` §3（O1M7E8 contract）
- `docs/developer/integer_wmma_gfx12.md`（WMMA operand layout）
