#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROCM=/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel
CLANG="$ROCM/lib/llvm/bin/clang++"
cd "$ROOT"
"$CLANG" \
  -DNDEBUG -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
  -I"$ROOT/include" -I"$ROOT/src" \
  -O3 -std=gnu++20 --offload-arch=gfx1201 -stdlib=libc++ \
  -stdlib++-isystem /usr/lib/llvm-23/include/c++/v1 \
  -Wno-unused-command-line-argument \
  -x hip \
  -o tools/rnd/ffn_k256_break_even/bench_sparse_down \
  tools/rnd/ffn_k256_break_even/bench_sparse_down.hip \
  src/phaseshift/models/qwen35/kernels/optimized/linear/psq4.hip \
  src/phaseshift/models/qwen35/kernels/optimized/linear/psq8.hip
echo "built $ROOT/tools/rnd/ffn_k256_break_even/bench_sparse_down"
