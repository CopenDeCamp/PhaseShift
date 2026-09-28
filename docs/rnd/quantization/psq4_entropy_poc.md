> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# PSQ4-E 圧縮 PoC（Gate E0〜E2）

Qwen3.8-27B-PSQ の PSQ4 weight を lossless permutation + entropy coding で圧縮し、
GPU 上で on-the-fly decode して既存 PSQ4 GEMM へ流せるかを検証する RnD。
**production format は変更しない。**

---

## Revision / Environment

| 項目 | 値 |
| --- | --- |
| baseline SHA | `2c6130d9`（main） |
| branch | `poc/psq4-entropy-storage` |
| worktree | `.worktrees/psq4-entropy-storage` |
| build | `build-gfx1201`、Release、gfx1201、`PHASESHIFT_BUILD_OPTIONAL_TESTS=ON` |
| GPU | AMD Radeon AI PRO R9700（gfx1201） |
| ROCm | 7.15.26333（`_rocm_sdk_devel`） |
| model | `models/Qwen3.8-27B-PSQ`（`phaseshift_quantization.json` を source of truth） |

PSQ4 payload は `encoding == "psq4"` の 336 tensor、**21,055.4M weights**（10.5 GiB）。

---

## 1. PSQ4 canonical layout（確認事項）

file は canonical row-major で、preshuffle は loader が in-memory で行う
（`weight_loader.cpp` の `preshuffle_native`）。analyzer は canonical を読む。

- codes: `[m, k_padded/2]` U8。block = 32 weights = 16 byte。row stride = 2560 byte。
- scale: `[m, k_padded/16]` U8（BF16 2 byte / block、row stride = 320 byte）。
- nibble 順: byte `j` の low nibble = k `ib*32 + j`、high nibble = k `ib*32 + 16 + j`
  （`quantize_psq4_row` / `dequantize_psq4_row` と一致）。
- CRC 検証: manifest の `codes.crc32` / `metadata1.crc32` と実 byte が一致。

---

## 2. Baseline PSQ4 decode1 resource（書き換えない）

`tools/extract_kernel_resources.py build-gfx1201/phaseshift-bench gemv_psq4_w4a8_decode1`。

| variant | SGPR | SGPR spill | VGPR | VGPR spill | private | LDS/group fixed | kernarg | wavefront | code size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode1`<BF16,2>` | 22 | 0 | 43 | 0 | 0 | 0 | 48 | 32 | 1440 |
| decode1`<BF16,4>` | 23 | 0 | 67 | 0 | 0 | 0 | 48 | 32 | 1920 |
| **decode1`<BF16,8>`** | **23** | **0** | **108** | **0** | **0** | **0** | 48 | 32 | 2812 |
| decode1`<BF16,16>` | 23 | 0 | 108 | 0 | 0 | 0 | 48 | 32 | 4700 |

過去資料の U=8 VGPR 108 / SGPR 23 / spill 0 を現 HEAD で再取得して一致を確認した。
`decode1` は LDS 0。runtime occupancy API による active wave 計測は未実施（下記 §6）。

---

## 3. Gate E0 — PSQ4 entropy（336 tensor 実測）

`tools/rnd/psq4_entropy_poc.py`。symbol = code nibble 0..15。

| role | tensors | weights (M) | H0 | Hk | Hrow | H2d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ffn_gate | 64 | 5704.3 | 3.8948 | 3.8936 | 3.8948 | 3.8935 |
| ffn_up | 64 | 5704.3 | 3.8943 | 3.8932 | 3.8943 | 3.8931 |
| ffn_down | 48 | 4278.2 | 3.8931 | 3.8918 | 3.8931 | 3.8918 |
| gdn_qkvza | 64 | 2684.4 | 3.8927 | 3.8913 | 3.8924 | 3.8910 |
| attn_q | 16 | 1006.6 | 3.8916 | 3.8903 | 3.8895 | 3.8881 |
| gdn_out | 32 | 1006.6 | 3.8925 | 3.8912 | 3.8924 | 3.8910 |
| attn_o | 16 | 503.3 | 3.8899 | 3.8885 | 3.8896 | 3.8881 |
| attn_k | 16 | 83.9 | 3.8734 | 3.8707 | 3.8710 | 3.8677 |
| attn_v | 16 | 83.9 | 3.8806 | 3.8783 | 3.8798 | 3.8770 |
| **ALL** | **336** | **21055.4** | **3.8936** | **3.8923** | **3.8934** | **3.8921** |

- H0 = 3.89 bpw（raw 4.0）。つまり **code symbol だけで既に 0.11 bpw しか余裕がない**。
- 条件付き entropy（K_PREV / ROW_PREV / 2D）は H0 から **0.0015 bpw しか下がらない**。
  隣接 code 間に実質的な相関は無い。
- symbol histogram は codebook のmag/sign 構造を反映したほぼ対称な分布で、
  最頻 2 symbol（mag1 の ±）でも 9.1% 程度。

---

## 4. Gate E1 — block-preserving permutation

`tools/rnd/psq4_perm_search.py`。32-channel block を壊さない範囲で
Stage A（block 順序）+ Stage B（block 内 32 channel 順序）を greedy NN
（16-bin histogram の L1）で決め、entropy を再計測した。layer 1。

| tensor | variant | H0 | Hk | Hrow | H2d |
| --- | --- | ---: | ---: | ---: | ---: |
| ffn_gate | before | 3.8971 | 3.8959 | 3.8971 | 3.8959 |
| ffn_gate | block only | 3.8971 | 3.8959 | 3.8971 | 3.8959 |
| ffn_gate | within only | 3.8971 | 3.8959 | 3.8971 | 3.8959 |
| ffn_gate | both | 3.8971 | 3.8959 | 3.8971 | 3.8959 |
| ffn_up | both | 3.8967 | 3.8956 | 3.8967 | 3.8956 |
| ffn_down | within only | 3.8948 | 3.8934 | 3.8948 | 3.8933 |
| ffn_down | both | 3.8948 | 3.8934 | 3.8948 | 3.8933 |

**permutation は entropy を変えない**（小数 4 桁まで同一）。
histogram ベース greedy は block/row の histogram がほぼ同一のため
本質的に任意順序を返し、相関を作れない。ROW_PREV / 2D を GPU へ持ち込む
根拠（§16 の 0.20 bpw 改善）は存在しない。

---

## 5. Gate E2 — CPU rANS codec

`tools/rnd/psq4e_rans.cpp`（第三者 library なし、32-bit rANS、scale_bits=12、
static frequency、`S` interleaved states）。encode→decode を実データで往復し
mismatch=0 を確認（89.1M symbol、S1/S2/S4/S8）。

圧縮率（ffn_gate layer 1 = 17408×5120、S interleave 別、NONE / K_PREV）:

| S | transform | payload bpw | state header bpw | offset bpw | perm bpw | scale bpw | **total bpw** |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | NONE | 3.89708 | 0.00625 | 0.01289 | 0.00160 | 0.50 | **4.41782** |
| 1 | K_PREV | 3.89648 | 0.00625 | 0.01289 | 0.00160 | 0.50 | **4.41723** |
| 2 | NONE | 3.89708 | 0.01250 | 0.01289 | 0.00160 | 0.50 | **4.42407** |
| 2 | K_PREV | 3.89648 | 0.01250 | 0.01289 | 0.00160 | 0.50 | **4.42348** |
| 4 | NONE | 3.89708 | 0.02500 | 0.01289 | 0.00160 | 0.50 | **4.43657** |
| 4 | K_PREV | 3.89649 | 0.02500 | 0.01289 | 0.00160 | 0.50 | **4.43598** |
| 8 | NONE | 3.89708 | 0.05000 | 0.01289 | 0.00160 | 0.50 | **4.46157** |
| 8 | K_PREV | 3.89649 | 0.05000 | 0.01289 | 0.00160 | 0.50 | **4.46098** |

ffn_down layer 1（5120×17408）: NONE 3.89479 / K_PREV 3.89414（payload）、
total は scale 0.5 + header/offset 込みで同様に 4.41〜4.46。

- header は output row あたり `S*4` byte（1 lane = k_padded weights）。
- offset は 16-row tile あたり 33×uint32。
- **metadata / state / header をすべて含めた total で報告している。**
- 最高の S1+K_PREV でも **4.417 bpw**（baseline PSQ4 = 4.5 bpw）。
  削減はわずか **1.8%** で、S を増やすほど header が増えて悪化する。

判定基準（§14）:

| 目標 | 結果 |
| --- | --- |
| STRONG GO `total <= 3.0` | 未達 |
| GO `total <= 3.3` | 未達 |
| BORDERLINE `3.3 < total <= 3.5` | 未達 |
| **NO-GO `total > 3.5`** | **該当（4.42〜4.46）** |

---

## 6. Cache / metadata traffic（§38）

| 項目 | baseline PSQ4 | PSQ4-E（想定） |
| --- | --- | --- |
| code read / 32 weights | 16 byte（warp で 256 byte 連続） | payload ≈ 15.6 byte |
| scale | 2 byte | 2 byte |
| state header | なし | S×4 byte / lane stream |
| offset table | なし | 33×4 byte / 16-row tile（coalesced 可） |
| entropy table | なし | 16 symbol。L1/constant 常駐可能 |

layout の懸念: baseline は 1 warp が 1 block あたり 256 byte を連続 read できる。
per-lane rANS stream にすると 32 lane が独立 stream を読み、各 lane の
byte 消費量がずれるため cache line が 32 本に分散し得る。
payload がほぼ減らない（3.897 vs 4.0）以上、この cache 増幅は正当化できない。

---

## 7. 結論

```
NO-GO（compression）
```

理由（数字）:

- PSQ4 code symbol の H0 = **3.8936 bpw**（raw 4.0）。条件付き entropy は
  **0.0015 bpw** しか下がらない。
- block-preserving permutation（block / within / both）は entropy を
  **完全に変えない**（§4）。
- 実 rANS の total は最良でも **4.417 bpw**、baseline 4.5 bpw からの削減は
  **1.8%**。目標 3.0〜3.3 bpw には **1.1 bpw 以上足りない**。
- したがって §43 の仮説「HBM traffic を 25〜35% 減らす」は成立しない。

### Gate E3 以降を実施しない理由

§3 の規約「前段が成立しなければ後段へ進まない」に従う。本 Gate の核心は
圧縮率であり、E2 で NO-GO（4.42 bpw ≫ 3.3 bpw）が確定した。この状態で
GPU decoder / GEMV 融合 / M 評価 / FFN E2E を実装しても、入力 byte 数が
ほぼ減らないため M=1 の break-even（§28）に到達できない。GPU 側 resource と
cache line の追加コストだけが残る。よって E3〜E6 は実行しない。

### 副次的な知見

- PSQ4 code は codebook 設計（mag 8 値 + sign）によりほぼ一様で、
  entropy coder で削れる余地が構造的に存在しない。
- 4.5 bpw のうち scale が 0.5 bpw を占める。payload を削るより scale 側
  （per-32 scale の共有・再設計）を見る方が余地は大きいが、それは
  「量子化誤差を追加しない」制約の外側であり本 PoC の範囲外。

### 未実施（明示）

- E3 GPU decoder microbench、E4 GEMV 融合、E5 M=2..16、E6 FFN / E2E。
- runtime occupancy API による active wave 計測（baseline は static resource のみ）。
- PoC 用 ctest 追加（`test_psq4e_rans_roundtrip` / `test_psq4e_gpu_decode_roundtrip` 等）。
  圧縮 NO-GO により GPU 側実装を行わないため。

### required regression

production code / format は変更していない。変更は `tools/rnd/` の追加のみ。
`test_psq_payload` / `test_psq4_codebook` / `test_gemm_psq4_w4a8_wmma` /
`test_gemm_psq4_decode1` は未変更の production code 上で成立する。
