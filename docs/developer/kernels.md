# Kernels

カーネルソースは `src/phaseshift/models/qwen35/kernels/` に置く。
配置と分割の詳細は `src/phaseshift/models/qwen35/kernels/README.md` を参照。

```text
kernels/
├── correctness/   # 遅いが正確な参照実装（E2E token exactness の基準）
├── optimized/     # gfx1201 向け高速実装
└── README.md
```

## 分割ルール

- `KernelId` は論理IDであり、ファイル構成を決めない。
- 1 ファイル = 1 実装 family（同じ最適化問題）。複数の `__global__` / launcher を置いてよい。
- dtype は variant。同じ family の BF16 / FP8 は同じファイルに置く。
- `1 KernelId = 1 file` / `1 __global__ = 1 file` / `1 dtype = 1 file` は原則NG。
- 細切れの `silu.hip` / `sigmoid.hip` / `scale.hip` / `mul.hip` / `residual_add.hip` /
  `swiglu.hip` は作らず、`elementwise.hip` にまとめる。
- attention は `attention_paged_bf16.hip` / `attention_paged_fp8.hip` ではなく
  `paged_decode.hip` / `paged_prefill.hip` に BF16 / FP8 variant を内包する。

## Correctness

`correctness/model_dispatch_correctness.hip` が唯一の `__global__` dispatch + KernelId switch +
launcher を持ち、`detail/*.inc` を include して 1 つの HIP translation unit を構成する。

```cpp
#include "detail/dispatch_env.h"
#include "detail/elementwise.inc"
#include "detail/normalization.inc"
#include "detail/linear.inc"
#include "detail/attention.inc"
#include "detail/gdn.inc"
#include "detail/embedding.inc"
#include "detail/output.inc"
```

これにより source 上は分割しつつ、次を満たす。

- KernelId switch は 1 箇所
- device helper の共有が容易
- RDC を要求しない
- correctness reference が 1 つに保たれる

`correctness/standalone/` は kernel 単体テスト用の参照実装
（`silu` / `swiglu` / `rmsnorm` / `gemm_bf16`）。E2E correctness と同じ階層に
二重実装を増やさない。可能なら `DispatchBinding` を作って
`model_dispatch_correctness` 経由で検証し、この階層を縮小する。

## Optimized

optimized variantは `src/phaseshift/models/qwen35/kernels/optimized/` に実装する。
対象platformはgfx1201のみであり、arch suffixは付けない。ソースは
`cmake/targets.cmake` に明示列挙する。

- correctness referenceはarch-independentで1つ。
- public launcher は semantic operator / optimization family ごとに置き、
  shape / dtype / algorithm variant の選択は launcher 内部に閉じ込め、runtime側へ漏らさない。
- WMMA kernelはgfx12 builtin（`_gfx12`）を使う。
- small-M（M=1..16）のbatch-invariant順序は `linear/bf16.hip` のexact-rows
  経路が担う。MTP verifyはこの順序に依存する。
