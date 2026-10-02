# PSQ Canonical SoA Payload / GPU Native Preshuffle Specification

この文書は quantized weight の **canonical file payload** と、GPU 実行時の
**native preshuffle contract** を定義する。

量子化アルゴリズムそのもの（codebook、scale 探索、丸め、誤差関数）は本仕様の対象外。
実装の正本は `include/phaseshift/quantization/**`、`src/phaseshift/quantization/**`、
`include/phaseshift/weights/**`、`src/phaseshift/weights/**` である。

対象 format は BF16 / PSQ4 / PSQ8 / FP8 E4M3 block128 / MXFP4 の 5 つ。

`canonical` は「GPU が直接読む形式」を意味しない。

```text
canonical = backend 非依存・冗長性なし・CPU で解釈可能な SoA
native    = kernel の load/gather と bank mapping に合わせて preshuffle 済みの SoA
```

GPU kernel が canonical payload を直接読む production path を禁止する。

---

## 1. 設計原則

優先順位は以下とする。

1. **常駐 VRAM を増やさない**
2. **GPU 実行前の preshuffle を必須とする**
3. **file payload は必ず SoA とする**
4. **GPU native payload も SoA とする**
5. canonical file には backend / kernel 固有の shuffle を保存しない
6. metadata を複製しない
7. 不要な reserved byte / padding を追加しない
8. canonical payload だけで CPU dequantize 可能にする
9. ロード時間の増加は許容する

---

## 2. 用語

### Stream

canonical レイアウトは次の stream からなる。

```text
codes
metadata1
metadata2
metadata3
metadata4
```

- `codes`：重みコード配列（常に 1 本）。format 別の code bytes/Block。
- `metadata1..4`：descriptor の `meta[4]` に対応する metadata 配列。
  未使用 slot は size-0／省略（0 byte ダミー領域は予約しない）。

format ごとに使う stream が異なる。PSQ4 / PSQ8 / FP8_BLOCK128 / MXFP4 は
`codes` と `metadata1` を使い、`metadata2..4` は size-0 である。

### Metadata mode

metadata の配置は `quantization/quant_format.h` の `MetaCountMode` が決める。

| mode | 要素数 |
|---|---|
| `None` | なし |
| `PerBlock` | (row, K block) ごと |
| `PerSuperBlock` | (row, superblock) ごと |
| `MatrixBlock` | (N block, K block) tile ごと。tile 内の全 row で共有する |

`QuantMetaDesc` は `element_bytes`（1 item の byte 数）、`count_mode`、
`MatrixBlock` 用の `block_n` / `block_k` を持つ。現在使用する format が
`PerBlock` / `MatrixBlock` / `None` を使う。

---

## 3. 共通 shape 規則

各 row の logical K を `logical_k` とする。

```text
padded_k       = align_up(logical_k, block_elements)
blocks_per_row = padded_k / block_elements
```

- padding weight は 0 として量子化する。
- `block_elements` は format 定義（BF16 = 16、PSQ4 / PSQ8 / MXFP4 = 32、
  FP8_BLOCK128 = 128）。
- metadata は row boundary をまたがない。row 間で scale / exponent を共有しない。

---

## 4. Canonical file layout

量子化 tensor の file payload は必ず SoA とする。

```text
BF16
  data
  (codes / metadata なし)

PSQ4
  codes
  metadata1

PSQ8
  codes
  metadata1

FP8_BLOCK128
  codes
  metadata1   (FP32 scales)

MXFP4
  codes
  metadata1   (E8M0 scales)
```

AoS は禁止。

```text
禁止:
  [block0 code][block0 meta][block1 code][block1 meta]...
```

- stream はそれぞれ独立した連続領域として扱う。
  Safetensors 等の外側コンテナでは各 stream を独立 tensor として格納する。
- 外側コンテナが region alignment を要求することは許容するが、その alignment は
  quantization payload の情報量には含めない。block / tile ごとの padding は追加しない。
- file の `codes` / `metadata1..4` は `QuantizedTensorMetadata` の同名 ref に対応する。

---

## 5. BF16 canonical format

BF16 は量子化ではないが、weight storage / compute format として同じ descriptor 機構で扱う。

```text
block_elements       = 16
code_bytes_per_block = 32   (= 16 element * 2 bytes)
metadata             = なし
```

- on-disk では `data` ref に row-major BF16 tensor を格納する。
  `codes` / `metadata1..4` は持たない。
- `k_padded == logical_k`。
- preshuffle を行わず、row-major BF16 のまま GPU へ配置する。
- `codes` stream の BF16 descriptor は compute view 用で、file 上に canonical
  codes stream を持つわけではない。

---

## 6. PSQ4 canonical format

### 6.1 Parameters

```text
block size            = 32 weights
codebook              = CB10
codes bytes/block     = 16
metadata1 bytes/block = 2   (BF16, PerBlock)
```

### 6.2 codes stream

Block（32 weight）の 4-bit code を 16 byte に格納する。
canonical file 上では **logical weight 順の sequential nibble packing** とする。

```text
codes byte 0: low = q[0],  high = q[1]
codes byte 1: low = q[2],  high = q[3]
...
codes byte15: low = q[30], high = q[31]
```

式:

```cpp
dst[j] = q[2 * j] | (q[2 * j + 1] << 4);
```

`q[j] | (q[j + 16] << 4)` のような kernel-oriented packing は canonical file に保存しない。
必要な場合は native preshuffle 時に作る。

### 6.3 metadata1 stream

metadata1 は Block-local BF16 scale。

```text
metadata1[block] = bf16 scale
2 bytes / block
little-endian raw BF16 bits
```

復元:

```text
w[i] ~= CB10[q[i]] * scale
```

### 6.4 Payload density

```text
codes     = 16 bytes / 32 weights
metadata1 =  2 bytes / 32 weights
Total     = 18 bytes / 32 weights = 4.5 bpw
```

---

## 7. PSQ8 canonical format

### 7.1 Parameters

```text
block size            = 32 weights
code format           = OCP E4M3
codes bytes/block     = 32
metadata1 bytes/block = 2   (BF16, PerBlock)
```

### 7.2 codes stream

codes は logical weight 順に格納する。

```text
codes[0]  = e4m3(q[0])
codes[1]  = e4m3(q[1])
...
codes[31] = e4m3(q[31])
```

1 byte / weight、32 bytes / block。

### 7.3 metadata1 stream

metadata1 は Block-local BF16 scale。

```text
metadata1[block] = bf16 scale
2 bytes / block
little-endian raw BF16 bits
```

復元:

```text
w[i] ~= e4m3_decode(codes[i]) * scale
```

### 7.4 Payload density

```text
codes     = 32 bytes / 32 weights
metadata1 =  2 bytes / 32 weights
Total     = 34 bytes / 32 weights = 8.5 bpw
```

---

## 8. FP8 E4M3 block128 canonical format

### 8.1 Parameters

```text
block_elements (K)    = 128
code format           = OCP E4M3
codes bytes/block     = 128   (1 byte / weight)
metadata              = metadata1, FP32, MatrixBlock 128 x 128
```

### 8.2 codes stream

codes は row-major の E4M3（1 byte / weight）を logical weight 順に格納する。

```text
codes[row][k] = e4m3(q[row * padded_k + k])
padded_k      = align_up(logical_k, 128)
```

### 8.3 metadata1 stream

metadata1 は **FP32 scale** で、`MatrixBlock`（`block_n = 128`、`block_k = 128`）を
使う。1 個の scale は 128 出力 row × 128 K の tile 内の全 weight が共有する。

```text
scale_count = ceil(rows / 128) * (padded_k / 128)
4 bytes / scale
```

- 2D matrix `[N, K]` では `ceil(N / 128) * (padded_k / 128)` 個。
- batch 次元を持つ場合、128 x 128 scale tile は matrix 境界をまたがない。
  batch ごとに別 tile を確保する。
- 復元:

```text
w[row][k] ~= e4m3_decode(codes[row][k]) * scale[row / 128][k / 128]
```

### 8.4 Payload density

codes は 1 byte / weight。scale は 128 x 128 tile あたり 4 bytes を tile 内の
16384 weight で割るため、weight あたり 1/4096 byte に相当する。

---

## 9. MXFP4 canonical format

### 9.1 Parameters

```text
block_elements (K)    = 32
code format           = OCP MXFP4 (E2M1)
codes bytes/block     = 16    (2 weights / byte)
metadata1 bytes/block = 1     (E8M0, PerBlock, block_k = 32)
```

### 9.2 codes stream

codes は row-major の E2M1（2 weights / byte）を格納する。

```text
nibble (0..15): bit 3 = sign, bits 0..2 = kE2M1Magnitudes の index
padded_k = align_up(logical_k, 32)
```

### 9.3 metadata1 stream

metadata1 は E8M0 scale（1 byte / block、`PerBlock`）である。

```text
metadata1[row][block] = e8m0 scale
scale_value = 2^(byte - 127)   // byte 255 は NaN
```

復元:

```text
w[i] ~= e2m1_decode(nibble) * e8m0_decode(metadata1[row][i / 32])
```

### 9.4 Payload density

```text
codes     = 16 bytes / 32 weights
metadata1 =  1 byte  / 32 weights
Total     = 17 bytes / 32 weights = 4.25 bpw
```

---

## 10. Stream size formulas

`rows` を row 数、`blocks = rows * (padded_k / block_elements)` とする。

### BF16

```text
data bytes = rows * logical_k * 2
```

### PSQ4

```text
codes bytes     = blocks * 16
metadata1 bytes = blocks * 2
```

### PSQ8

```text
codes bytes     = blocks * 32
metadata1 bytes = blocks * 2
```

### FP8_BLOCK128

```text
codes bytes     = rows * padded_k
metadata1 bytes = ceil(rows / 128) * (padded_k / 128) * 4
```

batch 次元を持つ場合は `ceil(rows / 128)` を batch ごとに計算する。

### MXFP4

```text
codes bytes     = rows * (padded_k / 2)
metadata1 bytes = rows * (padded_k / 32)
```

---

## 11. File stream naming / descriptor contract

format は固定数の Meta stream を持たない。
必要な stream だけを descriptor / manifest で参照する。

実装の descriptor（`phaseshift/quantization/quant_format.h`）:

```cpp
enum class QuantFormatId : uint8_t {
    None = 0, Bf16 = 1, Psq4 = 2, Psq8 = 3, Fp8Block128 = 4, Mxfp4 = 5,
};

static_assert(static_cast<uint8_t>(QuantFormatId::None) == 0);
static_assert(static_cast<uint8_t>(QuantFormatId::Bf16) == 1);
static_assert(static_cast<uint8_t>(QuantFormatId::Psq4) == 2);
static_assert(static_cast<uint8_t>(QuantFormatId::Psq8) == 3);
static_assert(static_cast<uint8_t>(QuantFormatId::Fp8Block128) == 4);
static_assert(static_cast<uint8_t>(QuantFormatId::Mxfp4) == 5);

enum class MetaCountMode : uint8_t {
    None = 0, PerBlock = 1, PerSuperBlock = 2, MatrixBlock = 3,
};

struct QuantMetaDesc {
    uint32_t element_bytes = 0;   // 1 item あたりの byte 数（可変: 1 / 2 / 4）
    MetaCountMode count_mode = MetaCountMode::None;
    uint32_t block_n = 1;         // MatrixBlock の N 方向 tile（他は 1）
    uint32_t block_k = 0;         // MatrixBlock の K 方向 tile（format block は 0）
};

struct QuantFormatDesc {
    uint32_t block_elements = 0;        // Block の weight 数
    uint32_t blocks_per_superblock = 1; // 現行 format はすべて 1
    uint32_t code_bytes_per_block = 0;  // BF16 = 32, PSQ4 = 16, PSQ8 = 32, FP8 = 128, MXFP4 = 16
    QuantMetaDesc meta[4];              // metadata1..4（未使用は None / 0 byte）
};
```

format descriptor:

```text
BF16
  block_elements = 16, code_bytes_per_block = 32, meta なし

PSQ4
  block_elements = 32, code_bytes_per_block = 16
  metadata1 : element_bytes = 2, PerBlock

PSQ8
  block_elements = 32, code_bytes_per_block = 32
  metadata1 : element_bytes = 2, PerBlock

FP8_BLOCK128
  block_elements = 128, code_bytes_per_block = 128
  metadata1 : element_bytes = 4, MatrixBlock, block_n = 128, block_k = 128

MXFP4
  block_elements = 32, code_bytes_per_block = 16
  metadata1 : element_bytes = 1, PerBlock, block_n = 1, block_k = 32
```

numeric value は in-memory の identifier のみで、file / manifest には保存しない。
on-disk の encoding は `phaseshift_quantization.json` の文字列名で管理される。

```text
bf16                  / Bf16
psq4                  / PSQ4
psq8                  / PSQ8
fp8_e4m3_block128_f32 / Fp8Block128
mxfp4_e2m1_e8m0_s32   / Mxfp4
```

`QuantizedEncoding`（`fpx/quantized_manifest.h`）と `WeightEncoding`（`fpx/types.h`）は
同じ番号体系（BF16 = 0, PSQ4 = 3, PSQ8 = 4, FP8_BLOCK128 = 5, MXFP4 = 6）を持つ。

既存 manifest / tensor reference を拡張する場合も、`scales` という 1 本の opaque stream に
複数の意味を埋め込む設計には戻さない。

---

## 12. Mandatory GPU native preshuffle

GPU backend では preshuffle を **必須処理** とする。

```text
Canonical file SoA
        |
        v
Canonical host view
        |
        v
MANDATORY native-layout builder / preshuffle
        |
        v
GPU Native SoA
        |
        v
Kernel
```

以下の経路は禁止。

```text
Canonical SoA -> GPU kernel
```

preshuffle は optional optimization ではなく GPU representation を成立させる変換である。

format 別の builder:

- PSQ4 / PSQ8 : `quantization::psq::preshuffle_native`
- FP8_BLOCK128 : `quantization::fp8::preshuffle_fp8_block128_native`
- MXFP4 : `quantization::mxfp4::preshuffle_mxfp4_native`
- BF16 : preshuffle なし（row-major BF16 のまま）

Kernel API は canonical view を受け取れない構造にすることを推奨する。

---

## 13. Preshuffle の目的: bank-aware physical mapping

preshuffle の目的は単なる転置ではない。

kernel の 1 load / gather 命令で必要になる codes / metadata が、対象 RDNA4 kernel の
cache / memory bank mapping 上で不要に多数の bank へ分散しないように物理配置を作る。

特に codes stream は必ず kernel access pattern に合わせて preshuffle する。
metadata も同じ native-layout builder で kernel-native order に変換する。

各 kernel について最低限以下の対応関係を定義できなければならない。

```text
lane
logical quant index
canonical byte offset
native byte offset
nibble position (PSQ4 / MXFP4)
metadata index
bank mapping
```

「correctness が通る」だけでは native layout の完了条件にならない。
期待する bank mapping をテストまたは profiler / ISA 解析で確認する。

canonical 順を直接 gather する fallback を production path に残してはならない。

---

## 14. GPU Native SoA

native representation も SoA を維持する。
codes と metadata を Block 単位で interleave した AoS resident layout は禁止。

stream 内部の順序は kernel / RDNA4 bank mapping に合わせて変更してよい。
canonical file の順序と native の順序が同一である必要はない。

native preshuffle は 16 出力 row を 1 tile とする。
`rows` は 16 の倍数へ pad し、pad 分は 0 で埋める。

### PSQ4 native

```text
codes     : [tile16][nb][16 outputs * 16 bytes]      (nb = padded_k / 32)
metadata1 : [tile16][nb][16 outputs * 2 bytes]
```

### PSQ8 native

```text
codes     : [tile16][nb][ [half=0: 16 outputs * 16 bytes][half=1: 16 outputs * 16 bytes] ]
metadata1 : [tile16][nb][16 outputs * 2 bytes]
```

各 (tile, block) の 512 byte chunk は、32-wide block の前半 16 code と後半 16 code を
half として分けた WMMA consumer layout である。

### FP8_BLOCK128 native

```text
codes     : 16 outputs x 32 K の chunk 順
            codes_row_stride_bytes = nb * 512   (nb = padded_k / 32)
metadata1 : row-major [ceil(rows / 128)][padded_k / 128] の FP32
```

canonical の codes は row-major だが、native では WMMA consumer の chunk 順へ並べ替える。
pad row は 0 埋めする。

### MXFP4 native

```text
codes     : 16 outputs x 32 K の chunk 順
            各 output は 16 bytes。byte b の low nibble が code[k=b]、
            high nibble が code[k=16+b]
            codes_row_stride_bytes = nb * 256   (nb = padded_k / 32)
metadata1 : row-major [rows][padded_k / 32] の E8M0
```

---

## 15. VRAM contract

同じ logical tensor shape / quantization type に対して、本仕様への移行によって
**resident VRAM を増加させてはならない**。

特に禁止:

- unused metadata stream の確保
- per-Block reserved byte
- per-Block / per-tile alignment padding
- codes / metadata の AoS 化に伴う padding
- canonical GPU copy と native GPU copy の二重常駐
- 高速化のための metadata duplication

推奨ロード経路:

```text
Disk canonical SoA
       |
       v
CPU canonical buffers
       |
       v
CPU native preshuffle / packing
       |
       v
H2D
       |
       v
GPU native SoA only
```

GPU 上で preshuffle する場合は temporary canonical buffer を変換後に必ず解放する。
peak VRAM が問題になる場合は chunked transform または in-place transform を使用する。

native layout は情報の permutation を基本とし、raw quantization payload の情報量を
増やさない。native preshuffle の 16-row tile padding は kernel が要求する tile 境界で
あり許容するが、今回の format 統一を理由に新たな常駐 padding を増やしてはならない。

---

## 16. Serialization / quantization / native layout の責務分離

責務は以下の 3 層に分ける。

```text
Quantization math
  - code selection
  - scale search
  - CB10 / E4M3 / E2M1 encode
  - scale 最適化

Canonical serialization
  - sequential codes packing
  - metadata stream 生成
  - SoA file write/read
  - shape / size validation

GPU native layout
  - mandatory preshuffle
  - WMMA / lane mapping
  - bank-aware physical ordering
```

format 統一で quantization math を変更してはならない。

canonical disk format は **TP-independent** である。tensor parallel を使う場合も
disk 上の canonical payload は global tensor のままで、rank-local partition は
load 時に行われ、その rank-local canonical の後に native preshuffle が実行される。
native payload を disk contract にしない現在の方針を維持する。
順序 contract は [tensor_partition.md](tensor_partition.md) を参照する。

---

## 17. Loader contract

この contract の実装は generic weights layer（`ps::weights::load_quantized_matrix` /
`load_bf16_matrix`、`include/phaseshift/weights/weight_loader.h`、target `phaseshift_weights`）。
model 側（例: Qwen35）は tensor name → model field の binding のみを行い、
encoding の分岐を持たない。

loader は以下を順番に行う。

1. format / version を検証する
2. logical shape / padded K を検証する
3. codes / metadata stream size を検証する
4. canonical SoA を読む
5. mandatory native preshuffle を行う（`WeightLoadOptions.preshuffle = false` の場合を除く）
6. native SoA（canonical の場合は canonical SoA）を GPU へ配置する
7. kernel には native view のみ渡す

preshuffle のスイッチは generic layer（`ps::weights::WeightLoadOptions`、
`load_bf16_matrix` / `load_quantized_matrix`）に属する。model 側（Qwen35 等）は
これを自 LoadOptions に組み込むだけで、独自に preshuffle フラグを持たない。

`preshuffle = false` の意味:

- PSQ4 / PSQ8 : canonical SoA をそのまま GPU へ配置する。optimized WMMA GEMM は
  dispatch 時に `weight not preshuffled to native layout` で失敗する。
  correctness / reference 経路でのみ消費できる。
- FP8_BLOCK128 : canonical SoA をそのまま配置する。optimized GEMM は
  未 preshuffle の weight を拒否する。
- MXFP4 : canonical SoA をそのまま配置する。optimized GEMM は
  未 preshuffle の weight を拒否する。
- BF16 : preshuffle の対象外（常に row-major BF16）。

preshuffle を呼び忘れても動作する API を作らない。
kernel dispatch 時に native layout であることを validate する。

---

## 18. Versioning

canonical packing または stream 構成を変更した場合は layout / version を更新する。
on-disk manifest の `format_version` は現在 3 である（`kQuantizedSafetensorsFormatVersion`）。

旧形式を新形式として silent に読み込んではならない。

旧形式をサポートする場合は:

```text
legacy file
    |
    v
legacy reader
    |
    v
canonical SoA
    |
    v
mandatory native preshuffle
    |
    v
GPU native SoA
```

とする。

writer は原則として最新 canonical format のみを書き出す。

---

## 19. Validation requirements

reader / verifier は最低限以下を検証する。

```text
logical_k > 0
padded_k >= logical_k
padded_k % block_elements == 0
known encoding
supported format version
codes size == expected
metadata1 size == expected
```

format 別の追加条件:

```text
PSQ4 / PSQ8 : k_padded % 32 == 0
FP8_BLOCK128: k_padded % 128 == 0, metadata1 は FP32 で ceil(rows / 128) * (padded_k / 128) 個
MXFP4       : k_padded % 32 == 0, metadata1 は E8M0 で rows * (padded_k / 32) 個
BF16        : k_padded == logical_k, codes / metadata を持たない
```

---

## 20. Required tests

### 20.1 Canonical pack / unpack

PSQ4 の nibble order を既知 code 列で検証する。

```text
q -> canonical pack -> canonical unpack -> q
```

bit-exact で一致すること。

### 20.2 Quantization round-trip

各 format について:

```text
FP32/BF16
  -> quantize
  -> canonical SoA
  -> CPU dequantize
```

を検証する。少なくとも以下を含む。

```text
all zero
positive / negative
mixed sign
small / large values
K = 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129
```

### 20.3 FP8 block128 tile boundary

N が 128 の倍数でない multi-row tensor を用意し、128 x 128 scale tile が
matrix 境界をまたがないことを検証する。

### 20.4 Native preshuffle correctness

```text
canonical
  -> native preshuffle
  -> test-only inverse transform
  -> canonical
```

を bit-exact で確認する。

### 20.5 Bank mapping

kernel の主要 load / gather について、logical index -> native offset -> bank mapping を検証する。
少なくとも codes stream は必須。metadata も kernel access pattern に応じて検証する。

### 20.6 VRAM regression

同一 tensor shape について resident allocation bytes を変更前後で比較し、
増加していないことを確認する。

raw payload の基準値:

```text
PSQ4      = 18 bytes / 32 weights
PSQ8      = 34 bytes / 32 weights
MXFP4     = 17 bytes / 32 weights
FP8_BLOCK128 = 1 byte / weight + FP32 scale per 128 x 128 tile
BF16      = 2 bytes / weight
```

---

## 21. Final format summary

```text
PSQ4
-----------------------------------------------
codes     : 16 bytes / 32 weights, 4-bit CB10
metadata1 :  2 bytes / 32 weights, BF16 scale
Raw       : 18 bytes / 32 weights
```

```text
PSQ8
-----------------------------------------------
codes     : 32 bytes / 32 weights, E4M3
metadata1 :  2 bytes / 32 weights, BF16 scale
Raw       : 34 bytes / 32 weights
```

```text
FP8_BLOCK128
-----------------------------------------------
codes     : 1 byte / weight, E4M3
metadata1 : 4 bytes / 128 x 128 tile, FP32 scale
Raw       : 1 byte / weight + scale tile
```

```text
MXFP4
-----------------------------------------------
codes     : 16 bytes / 32 weights, E2M1 (2 weights / byte)
metadata1 :  1 byte  / 32 weights, E8M0 scale
Raw       : 17 bytes / 32 weights = 4.25 bpw
```

```text
BF16
-----------------------------------------------
data      : 2 bytes / weight, row-major
Raw       : 2 bytes / weight
```

GPU execution:

```text
Canonical SoA
    |
    v
MANDATORY RDNA4 bank-aware preshuffle
    |
    v
Native SoA
    |
    v
Kernel
```

**canonical payload を production GPU kernel が直接参照することは禁止する。**
