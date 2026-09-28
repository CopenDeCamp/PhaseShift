#include <phaseshift/quantization/fpx/layout.h>
#include <cstdio>
#include <cstdint>

using ps::quantization::fpx::align_up64;
using ps::quantization::fpx::compute_region_sizes;
using ps::quantization::fpx::element_count;
using ps::quantization::fpx::kPsq4CodesBytesPerBlock;
using ps::quantization::fpx::kPsq8CodesBytesPerBlock;
using ps::quantization::fpx::kPsqScalesBytesPerBlock;
using ps::quantization::fpx::kRegionAlignment;
using ps::quantization::fpx::padded_k;
using ps::quantization::fpx::row_count;
using ps::quantization::fpx::WeightEncoding;

namespace {
int g_passed = 0;
int g_failed = 0;

void check(bool cond, const char* msg) {
    if (cond) { ++g_passed; } else { ++g_failed; std::printf("FAIL: %s\n", msg); }
}

int run() {
    check(align_up64(0) == 0, "align64(0)==0");
    check(align_up64(1) == 64, "align64(1)==64");
    check(align_up64(64) == 64, "align64(64)==64");
    check(align_up64(65) == 128, "align64(65)==128");

    check(padded_k(1024, WeightEncoding::BF16) == 1024, "bf16 k_padded==k");
    check(padded_k(1024, WeightEncoding::PSQ4) == 1024, "psq4 k_padded 1024");
    check(padded_k(100, WeightEncoding::PSQ8) == 128, "psq8 k_padded 100->128");
    check(padded_k(33, WeightEncoding::PSQ4) == 64, "psq4 k_padded 33->64");

    // PSQ4: shape [rows, k]
    {
        const std::vector<int64_t> shape = {8, 100};  // rows=8, k=100 -> kp=128, blocks/row=4
        const auto r = compute_region_sizes(shape, WeightEncoding::PSQ4);
        const uint64_t blocks = 8 * 4;
        check(r.codes_bytes == blocks * kPsq4CodesBytesPerBlock, "psq4 codes bytes");
        check(r.scales_bytes == blocks * kPsqScalesBytesPerBlock, "psq4 scales bytes");
        check(r.data_bytes == 0, "psq4 no data region");
        check(r.codes_bytes == blocks * 16, "psq4 codes == num_blocks*16");
        check(r.scales_bytes == blocks * 2, "psq4 scales == num_blocks*2");
    }
    // PSQ8
    {
        const std::vector<int64_t> shape = {8, 100};
        const auto r = compute_region_sizes(shape, WeightEncoding::PSQ8);
        const uint64_t blocks = 8 * 4;
        check(r.codes_bytes == blocks * kPsq8CodesBytesPerBlock, "psq8 codes bytes");
        check(r.codes_bytes == blocks * 32, "psq8 codes == num_blocks*32");
        check(r.scales_bytes == blocks * kPsqScalesBytesPerBlock, "psq8 scales bytes");
        check(r.scales_bytes == blocks * 2, "psq8 scales == num_blocks*2");
    }
    // PSQ8 single block: 32 weights -> 32 B codes + 2 B scale = 34 B = 8.5 bpw
    {
        const std::vector<int64_t> shape = {1, 32};
        const auto r = compute_region_sizes(shape, WeightEncoding::PSQ8);
        check(r.codes_bytes == 32, "psq8 1-block codes == 32");
        check(r.scales_bytes == 2, "psq8 1-block scales == 2");
        check(r.data_bytes == 0, "psq8 1-block no data region");
        const double bpw = static_cast<double>(r.codes_bytes + r.scales_bytes) * 8.0 / 32.0;
        check(bpw == 8.5, "psq8 1-block bpw == 8.5");
    }
    // BF16: contiguous, no padding
    {
        const std::vector<int64_t> shape = {3, 1024};
        const auto r = compute_region_sizes(shape, WeightEncoding::BF16);
        check(r.codes_bytes == 0 && r.scales_bytes == 0, "bf16 no codes/scales");
        check(r.data_bytes == 3 * 1024 * 2, "bf16 data bytes");
    }
    // 1D tensor
    {
        const std::vector<int64_t> shape = {1024};
        check(row_count(shape) == 1, "1d row_count==1");
        const auto r = compute_region_sizes(shape, WeightEncoding::BF16);
        check(r.data_bytes == 1024 * 2, "1d bf16 data bytes");
        check(element_count(shape) == 1024, "1d element_count");
    }
    // 3D conv1d [6144,1,4] as BF16
    {
        const std::vector<int64_t> shape = {6144, 1, 4};
        check(row_count(shape) == 6144, "3d row_count");
        const auto r = compute_region_sizes(shape, WeightEncoding::BF16);
        check(r.data_bytes == 6144 * 4 * 2, "3d bf16 data bytes");
    }
    // kRegionAlignment is 64
    check(kRegionAlignment == 64, "region alignment == 64");
    return 0;
}
}  // namespace

int main() {
    run();
    std::printf("=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
