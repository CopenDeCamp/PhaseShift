> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# PSQ4 次世代フォーマット検討メモ

更新日: 2026-09-21

## 1. 目的

現行 PSQ4 の精度上の利点を維持しながら、scale metadata と scale 適用コストを削減し、
prefill / decode の双方でより高いスループットを狙う。

現行 PSQ4 はすでに十分高速であり、全面的なフォーマット刷新を前提としない。

特に重要な前提:

- R9700 では native 4bit matrix path を前提にできない。
- PSQ4 は 4bit code を `v_perm` 等で FP8 E4M3 へ展開し、native FP8 WMMA を使う。
- CB10 decode 自体は、prefill では大部分が WMMA / memory latency の裏に隠れている。
- 現時点で最も明確な PSQ4 固有コストは per-K32 BF16 scale 側にある。
- よって、まず CB10 を捨てるのではなく、scale 構造を軽量化する。

---

## 2. 現行 PSQ4

概念的には以下。

```text
weight code:
    4bit CB10

decode:
    CB10 -> FP8 E4M3

scale:
    BF16 / (output row × K32)

compute:
    FP8 WMMA
    + block scale application
```

近似式:

```text
W[n,k] ~= S[n, k/32] * CB10(code[n,k])
```

scale overhead:

```text
BF16 16bit / 32 weights
= 0.5 bit / weight
```

したがって総量は概ね:

```text
4.0 bit code
+ 0.5 bit scale
= 4.5 bpw
```

---

## 3. 実測から分かったこと

### 3.1 Decode

同一 shape:

```text
n = 34816
k = 5120
rows = 1
```

実測:

| dtype | p50 us | weight bytes | 実効帯域 |
| --- | ---: | ---: | ---: |
| PSQ4 | 176.5 | 100.3 MB | 568 GB/s |
| PSQ8 | 308.5 | 189.4 MB | 614 GB/s |
| FP8 | 289.7 | 178.3 MB | 615 GB/s |
| MXFP4 | 152.2 | 94.7 MB | 622 GB/s |

PSQ4 は MXFP4 より遅いが、差は極端ではない。

MXFP4 と同じ 622 GB/s で 100.3 MB を読むだけなら:

```text
100.3 MB / 622 GB/s ~= 161.3 us
```

実測は 176.5 us なので、

```text
追加コスト ~= 15.2 us
```

しかない。

つまり CB10 decode + BF16 scale 処理 + scale layout の追加コストは、
decode 全体に対して約 8.6% 程度。

現行 kernel はすでにかなりよくチューニングされている。

### 3.2 Prefill

prefill では W load + CB10 decode の削除がほとんど性能改善にならないケースが確認されている。

一方で scale FMA を大きく削ると約 10% 級の改善が出たケースがある。

したがって prefill の次の主戦場は:

```text
CB10 decode
```

ではなく、

```text
scale metadata
scale load
scale address calculation
scale application FMA
```

である。

---

## 4. 設計方針

新 PSQ4 は次を基本方針とする。

```text
CB10 は維持する
    ↓
FP8 E4M3 へ軽量 decode
    ↓
native FP8 WMMA

ただし scale を再設計する
```

つまり、

```text
codebook を変えて高速化する
```

より先に、

```text
scale 表現を圧縮して高速化する
```

ことを狙う。

---

# 5. 候補A: N16 × K16 格子 scale

各 16 output × 16 K の 256 weights で 1 scale を共有する。

```text
W[n,k] ~= S[n/16, k/16] * CB10(code[n,k])
```

BF16 scale の overhead:

```text
16 bit / 256 weights
= 0.0625 bpw
```

総量:

```text
4.0
+ 0.0625
= 4.0625 bpw
```

現行 4.5 bpw から大幅に軽くなる。

### カーネル上の利点

RDNA4 の 16×16 WMMA tile と一致する。

概念的には:

```text
N16 × K16 packed PSQ4
        ↓
CB10 -> E4M3
        ↓
FP8 WMMA 16x16x16
        ↓
tile_scale
        ↓
accumulate
```

scale load / address / VGPR は大幅に減る。

### 問題

16 output 行の dynamic range を同一 scale が背負う。

一つの row に outlier があるだけで他の row の量子化効率が悪化する可能性がある。

精度面では現行 `N1 × K32` より厳しい。

---

# 6. 候補B: N16 × K32 格子 scale

より重要な候補。

```text
W[n,k] ~= S[n/16, k/32] * CB10(code[n,k])
```

512 weights につき BF16 scale 1個。

overhead:

```text
16 bit / 512
= 0.03125 bpw
```

総量:

```text
4.03125 bpw
```

scale metadata は現行の 1/16。

### 最大の利点

K32 を 2個の K16 WMMA として処理する場合、

```text
tmp0 = WMMA(K0..15)
tmp1 = WMMA(K16..31)

acc += scale * (tmp0 + tmp1)
```

とできる可能性がある。

つまり、

```text
2 WMMA に対して scale 1回
```

にできる。

これは metadata 削減だけでなく、
scale FMA の頻度そのものを減らせる可能性がある。

### 問題

N方向16行で scale を共有するため、row間 dynamic range 差に弱い。

---

# 7. 候補C: row_gain × N16 × K32 grid scale

現時点で最も有望な構造。

近似式:

```text
W[n,k]
~= row_gain[n]
 * grid_scale[n/16, k/32]
 * CB10(code[n,k])
```

構成:

```text
CB10 code:
    4bit / weight

grid_scale:
    BF16 / N16×K32

row_gain:
    BF16 / output row
```

### metadata

grid scale:

```text
0.03125 bpw
```

row_gain:

K=5120 の場合、

```text
16 / 5120
= 0.003125 bpw
```

合計:

```text
4.000000
+ 0.031250
+ 0.003125
----------------
= 4.034375 bpw
```

現行:

```text
4.5 bpw
```

に対して非常に軽い。

### 意味

row_gain が行ごとの global dynamic range 差を吸収する。

grid_scale は局所的な K方向変動だけを担当する。

つまり単純な N16×K32 grid よりも、

```text
row間のscale共有による精度悪化
```

を大きく緩和できる可能性がある。

### Runtime

row_gain は K-loop に置く必要がない。

```text
GEMM result
    ↓
output rowごとに row_gain を適用
```

とできるため、epilogueへ追い出せる。

K-loop 内は:

```text
CB10 decode
+ FP8 WMMA
+ grid_scale
```

だけに近づけられる。

---

# 8. 候補D: N4 / N8 × K32

N16共有で精度が落ちすぎる場合の中間案。

| scale unit | BF16 scale overhead |
| --- | ---: |
| N1 × K32 | 0.50000 bpw |
| N4 × K32 | 0.12500 bpw |
| N8 × K32 | 0.06250 bpw |
| N16 × K32 | 0.03125 bpw |

まず offline quantization で sweep すべき。

候補:

```text
N1 × K32
N2 × K32
N4 × K32
N8 × K32
N16 × K32
```

必要なら row_gain を各候補に追加する。

---

# 9. 格子化に意味がある理由

格子化の主目的は精度ではない。

主目的は:

```text
WMMA tile structure
と
scale ownership
を一致させる
```

ことである。

得られる利点:

1. scale metadata 削減
2. scale global load 削減
3. address calculation 削減
4. VGPR pressure 削減
5. cache pressure 削減
6. broadcast しやすい
7. 2 WMMA / 1 scale の構造を作れる可能性
8. epilogue へ row correction を追い出せる

一方で欠点は:

```text
N方向でscaleを共有することで量子化誤差が増える
```

ことである。

したがって、

```text
単純grid
```

より、

```text
row_gain × grid
```

が重要になる。

---

# 10. CB10を維持する理由

現時点では CB10 を捨てる理由は弱い。

理由:

- decode kernel では追加コストは限定的。
- prefill では CB10 decode がほぼ隠れているケースがある。
- CB10 は E2M1 より柔軟な非一様 codebook を持てる。
- 現行 PSQ4 の精度上の利点を保持しやすい。
- 4bit -> FP8 E4M3 decode は `v_perm` ベースで十分軽くできている。

したがって優先順位は:

```text
1. scale structure
2. scale application
3. scale layout
4. decode pipeline
5. codebook変更
```

とする。

---

# 11. 性能上の狙い

## Decode

現行:

```text
PSQ4:
176.5 us
100.3 MB
568 GB/s
```

grid scale で scale metadata を削ると、

```text
codes ~= 89.1 MB

N16×K32 BF16 scale:
~0.7 MB

total ~= 89.8 MB
```

程度まで下げられる。

row_gain を足しても増加は小さい。

理論的には MXFP4 の 94.7 MB より weight bytes は小さくできる。

ただし最終性能は:

```text
CB10 decode
scale apply
memory transaction efficiency
tile geometry
```

に依存する。

## Prefill

prefillでは memory bytes よりも:

```text
FP8 WMMA throughput
scale application dependency
A-side pipeline
occupancy
```

の重要度が高い。

特に N16×K32 によって:

```text
2×WMMA
↓
1×scale application
```

が成立すれば、大きな価値がある。

---

# 12. Offline Gate

production kernel を変更する前に、
quantizer だけで次を比較する。

```text
A. current PSQ4
   CB10 + BF16 N1×K32

B. CB10 + BF16 N4×K32

C. CB10 + BF16 N8×K32

D. CB10 + BF16 N16×K32

E. CB10 + BF16 N16×K16

F. row_gain × CB10 + BF16 N16×K32

G. row_gain × CB10 + BF16 N16×K16
```

評価対象:

```text
gate_proj
up_proj
down_proj
q_proj
k_proj / qkv
o_proj
```

複数 layer を使用する。

---

# 13. Offline metrics

最低限:

```text
RMS error
relative RMS
max error

block error:
p50
p90
p95
p99

per-row RMS
per-row max

scale distribution
row_gain distribution
```

加えて、

```text
current PSQ4 に対する error ratio
plain MXFP4 に対する error ratio
```

を出す。

特に N16共有では:

```text
同一N16 tile内の row max / RMS のばらつき
```

を必ず測定する。

---

# 14. 重要な追加解析

## 14.1 row_gain が有効か

各 row の既存 PSQ4 scale:

```text
S[n,g]
```

について、

```text
log2(S[n,g])
```

を調べる。

rowごとに共通のoffsetが存在するなら、

```text
log2(S[n,g])
~= row_offset[n]
 + grid_component[n/16,g]
```

と分解できる可能性がある。

これが成立するほど:

```text
row_gain × grid_scale
```

が有効。

## 14.2 N方向共有可能性

各 N16×K32 tile について:

```text
各rowの最適scale
```

の分散を測る。

例えば:

```text
max(scale) / min(scale)
std(log2(scale))
```

を記録する。

これにより N16共有が無理なlayerを特定できる。

---

# 15. Kernel Gate

offline精度が通った候補のみkernel化する。

測定:

```text
decode:
rows=1
n=34816
k=5120

prefill:
rows=1024
rows=2048
代表shape
```

取得:

```text
latency
effective GB/s
effective TFLOPS
VGPR
LDS
occupancy
v_perm count
VALU count
WMMA count
scale load count
scale FMA count
```

---

# 16. 特に確認すべき仮説

## 仮説1

```text
CB10 decodeは維持しても十分高速
```

すでにかなり支持されている。

## 仮説2

```text
PSQ4の主要な残存コストはscale側
```

prefill ablationから強く支持されている。

## 仮説3

```text
N16×K32にするとscale trafficだけでなく
scale application回数も削減できる
```

kernel PoCが必要。

## 仮説4

```text
row_gainを追加すればN16共有による精度低下を回収できる
```

offline quantizationで確認する。

---

# 17. 現時点の第一候補

```text
PSQ4-GRID

code:
    CB10 4bit

grid scale:
    BF16 / N16×K32

row correction:
    BF16 row_gain / output row

decode:
    CB10 -> FP8 E4M3

compute:
    FP8 WMMA ×2 / K32

scale:
    ideally one application after two K16 WMMA

epilogue:
    row_gain
```

概念:

```text
             K32
      ┌─────────────────┐
N16   │  512 CB10 codes │
      └─────────────────┘
              │
       grid_scale 1個
              │
     ┌────────┴────────┐
     │                 │
  K16 WMMA          K16 WMMA
     │                 │
     └────── add ──────┘
              │
         × grid_scale
              │
          accumulate
              │
             ...
              │
        × row_gain[n]
              │
            output
```

---

# 18. 期待される特性

現行 PSQ4:

```text
4.5 bpw
高いscale自由度
高精度
scale metadata重め
```

PSQ4-GRID:

```text
~4.034 bpw
CB10維持
scale metadata 約1/16
row_gainでrow差を吸収
WMMA tileにscale ownershipを合わせる
scale apply頻度削減の可能性
```

狙いは、

```text
MXFP4に近い、またはそれ以上のデータ効率
+
PSQ4の量子化精度
+
FP8 WMMAで高throughput
```

である。

---

# 19. 判定方針

この案は、まず速度ではなく精度でGateする。

### Gate 1

`N16×K32` 単体で current PSQ4 に十分近い。

→ そのままkernel PoC。

### Gate 2

`N16×K32` は悪化するが `row_gain × N16×K32` で回復。

→ row_gain版をkernel PoC。

### Gate 3

N16が厳しい。

→ N8、N4へ戻す。

### Gate 4

どのgridも精度を維持できない。

→ 現行 N1×K32 を維持し、scale dtype / layoutのみ再設計する。

---

# 20. 現時点の結論

現行 PSQ4 はすでに十分高速であり、
CB10 decode を根本から捨てる必要性は低い。

次世代化で最も価値があるのは:

```text
per-row K32 scale
        ↓
WMMA tile oriented grid scale
        +
row-level correction
```

への移行である。

特に:

```text
row_gain × N16×K32 grid_scale × CB10
```

は、

```text
精度
metadata量
kernel構造
scale処理回数
```

のバランスが最も良い第一候補。

まず offline quantization で成立性を確認し、
成立した場合のみ production kernel へ進む。
