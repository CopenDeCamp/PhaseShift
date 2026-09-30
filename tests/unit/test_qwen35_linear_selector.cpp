#include <phaseshift/models/qwen35/runtime/linear_selector.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        passed++;
    } else {
        failed++;
        std::printf("FAIL: %s\n", name);
    }
}

using ps::kernel::Bf16GemmConfig;
using ps::kernel::Bf16GemmConfigId;
using ps::kernel::Fp8Block128GemmConfig;
using ps::kernel::Fp8Block128GemmConfigId;
using ps::kernel::Mxfp4GemmConfig;
using ps::kernel::Mxfp4GemmConfigId;
using ps::kernel::Psq4GemmConfig;
using ps::kernel::Psq4GemmConfigId;
using ps::kernel::Psq8GemmConfig;
using ps::kernel::Psq8GemmConfigId;
using ps::qwen35::runtime::Bf16GemmSelectorInput;
using ps::qwen35::runtime::Fp8Block128GemmSelectorInput;
using ps::qwen35::runtime::Mxfp4GemmSelectorInput;
using ps::qwen35::runtime::Psq4GemmSelectorInput;
using ps::qwen35::runtime::Psq8GemmSelectorInput;
using ps::qwen35::runtime::select_bf16_gemm_config;
using ps::qwen35::runtime::select_fp8_block128_gemm_config;
using ps::qwen35::runtime::select_mxfp4_gemm_config;
using ps::qwen35::runtime::select_psq4_gemm_config;
using ps::qwen35::runtime::select_psq8_gemm_config;

std::optional<Bf16GemmConfig> sel_bf16(uint32_t rows, uint32_t out_features, uint32_t k,
                                       uint32_t stride, bool weight_aligned, bool input_aligned,
                                       bool verify_exact) {
    Bf16GemmSelectorInput in;
    in.rows = rows;
    in.out_features = out_features;
    in.k = k;
    in.input_row_stride = stride != 0u ? stride : k;
    in.weight_aligned16 = weight_aligned;
    in.input_aligned16 = input_aligned;
    in.verify_exact = verify_exact;
    return select_bf16_gemm_config(in);
}

std::optional<Psq4GemmConfig> sel_psq4(uint32_t rows, uint32_t out_features, uint32_t k,
                                       uint32_t k_padded) {
    return select_psq4_gemm_config(Psq4GemmSelectorInput{rows, out_features, k, k_padded});
}

std::optional<Psq8GemmConfig> sel_psq8(uint32_t rows, uint32_t out_features, uint32_t k,
                                       uint32_t k_padded) {
    return select_psq8_gemm_config(Psq8GemmSelectorInput{rows, out_features, k, k_padded});
}

std::optional<Fp8Block128GemmConfig> sel_fp8(uint32_t rows, uint32_t out_features, uint32_t k,
                                             uint32_t k_padded) {
    return select_fp8_block128_gemm_config(
        Fp8Block128GemmSelectorInput{rows, out_features, k, k_padded});
}

std::optional<Mxfp4GemmConfig> sel_mxfp4(uint32_t rows, uint32_t out_features, uint32_t k,
                                         uint32_t k_padded) {
    return select_mxfp4_gemm_config(
        Mxfp4GemmSelectorInput{rows, out_features, k, k_padded});
}

bool is_fp8(const std::optional<Fp8Block128GemmConfig>& c, Fp8Block128GemmConfigId id) {
    return c.has_value() && c->id == id;
}

bool is_mxfp4(const std::optional<Mxfp4GemmConfig>& c, Mxfp4GemmConfigId id) {
    return c.has_value() && c->id == id;
}

bool is_bf16(const std::optional<Bf16GemmConfig>& c, Bf16GemmConfigId id,
             uint8_t exact_rows) {
    return c.has_value() && c->id == id &&
           (id != Bf16GemmConfigId::ExactRows || c->exact_rows == exact_rows);
}

bool is_psq4(const std::optional<Psq4GemmConfig>& c, Psq4GemmConfigId id) {
    return c.has_value() && c->id == id;
}

bool is_psq8(const std::optional<Psq8GemmConfig>& c, Psq8GemmConfigId id) {
    return c.has_value() && c->id == id;
}

void test_bf16() {
    check(!sel_bf16(1, 123, 456, 456, true, true, false).has_value(),
          "bf16 unknown geometry -> no config");
    check(!sel_bf16(0, 1024, 2560, 2560, true, true, false).has_value(),
          "bf16 rows=0 -> no config");

    check(is_bf16(sel_bf16(1, 1024, 2560, 2560, true, true, false),
                  Bf16GemmConfigId::ExactRows, 1u),
          "bf16 rows=1 aligned -> ExactRows(1)");
    check(is_bf16(sel_bf16(1, 1024, 2560, 2560, false, true, false),
                  Bf16GemmConfigId::Wmma, 0u),
          "bf16 rows=1 weight unaligned -> Wmma");

    bool exact_ok = true;
    for (uint32_t rows = 2u; rows <= ps::kernel::kBf16GemmExactRowsMax; ++rows) {
        exact_ok = exact_ok &&
                   is_bf16(sel_bf16(rows, 1024, 2560, 2560, true, true, true),
                           Bf16GemmConfigId::ExactRows, static_cast<uint8_t>(rows));
    }
    check(exact_ok, "bf16 verify_exact rows 2..64 aligned -> ExactRows(rows)");

    check(is_bf16(sel_bf16(8, 1024, 2560, 2560, false, true, true),
                  Bf16GemmConfigId::Wmma, 0u),
          "bf16 verify_exact weight unaligned -> Wmma");
    check(is_bf16(sel_bf16(8, 1024, 2560, 2559, true, true, true),
                  Bf16GemmConfigId::Wmma, 0u),
          "bf16 verify_exact row stride unaligned -> Wmma");
    check(is_bf16(sel_bf16(65, 1024, 2560, 2560, true, true, true),
                  Bf16GemmConfigId::WmmaWide, 0u),
          "bf16 verify_exact rows=65 -> WmmaWide");
    check(is_bf16(sel_bf16(17, 1024, 2560, 2560, true, true, true),
                  Bf16GemmConfigId::ExactRows, 17u),
          "bf16 verify_exact rows=17 -> ExactRows(17)");
    check(is_bf16(sel_bf16(64, 1024, 2560, 2560, true, true, true),
                  Bf16GemmConfigId::ExactRows, 64u),
          "bf16 verify_exact rows=64 -> ExactRows(64)");

    check(is_bf16(sel_bf16(2, 32, 2560, 2560, true, true, false),
                  Bf16GemmConfigId::WmmaKPartition, 0u),
          "bf16 small output geometry -> WmmaKPartition");
    check(is_bf16(sel_bf16(65, 1024, 2560, 2560, true, true, false),
                  Bf16GemmConfigId::WmmaWide, 0u),
          "bf16 large rows -> WmmaWide");
    check(is_bf16(sel_bf16(2, 1024, 2560, 2560, true, true, false),
                  Bf16GemmConfigId::Wmma, 0u),
          "bf16 default geometry -> Wmma");
}

void test_psq4_row_blocks() {
    check(!sel_psq4(100, 123, 456, 456).has_value(),
          "psq4 unknown geometry -> no config");
    check(!sel_psq4(0, 1024, 2560, 2560).has_value(),
          "psq4 rows=0 -> no config");

    check(is_psq4(sel_psq4(1, 1024, 2560, 2560), Psq4GemmConfigId::RowBlock1),
          "psq4 rows=1 -> RowBlock1");
    check(is_psq4(sel_psq4(16, 1024, 2560, 2560), Psq4GemmConfigId::RowBlock1),
          "psq4 rows=16 -> RowBlock1");
    check(is_psq4(sel_psq4(17, 1024, 2560, 2560), Psq4GemmConfigId::RowBlock2),
          "psq4 rows=17 -> RowBlock2");
    check(is_psq4(sel_psq4(63, 1024, 2560, 2560), Psq4GemmConfigId::RowBlock2),
          "psq4 rows=63 -> RowBlock2");
    check(is_psq4(sel_psq4(64, 1024, 2560, 2560), Psq4GemmConfigId::RowBlock4),
          "psq4 rows=64 -> RowBlock4");
    check(is_psq4(sel_psq4(127, 1024, 2560, 2560), Psq4GemmConfigId::RowBlock4),
          "psq4 rows=127 -> RowBlock4");
    check(is_psq4(sel_psq4(128, 1024, 2560, 2560), Psq4GemmConfigId::RowBlock8),
          "psq4 rows=128 -> RowBlock8");
    check(is_psq4(sel_psq4(255, 1024, 2560, 2560), Psq4GemmConfigId::RowBlock8),
          "psq4 rows=255 -> RowBlock8");
    check(is_psq4(sel_psq4(256, 1024, 2560, 2560), Psq4GemmConfigId::RowBlock8),
          "psq4 rows=256 -> RowBlock8");
    check(is_psq4(sel_psq4(200, 32, 2560, 2560), Psq4GemmConfigId::RowBlock1),
          "psq4 small out_features -> RowBlock1");
}

void test_psq4_prefill_2d() {
    check(is_psq4(sel_psq4(512, 1024, 2560, 2560), Psq4GemmConfigId::Prefill2D_K128N64),
          "psq4 2d auto rows=512 out=1024 -> K128N64");
    check(is_psq4(sel_psq4(1024, 2560, 4096, 4096), Psq4GemmConfigId::Prefill2D_K128N128),
          "psq4 2d auto rows=1024 out=2560 -> K128N128");
    check(is_psq4(sel_psq4(256, 1024, 2560, 2560), Psq4GemmConfigId::RowBlock8),
          "psq4 rows=256 < 2d threshold -> RowBlock8");

    setenv("PHASESHIFT_PSQ_PREFILL_2D", "bk64bn64", 1);
    check(is_psq4(sel_psq4(512, 1024, 2560, 2560), Psq4GemmConfigId::Prefill2D_K64N64),
          "psq4 override bk64bn64 -> K64N64");
    setenv("PHASESHIFT_PSQ_PREFILL_2D", "bk128", 1);
    check(is_psq4(sel_psq4(512, 1024, 2560, 2560), Psq4GemmConfigId::Prefill2D_K128N64),
          "psq4 override bk128 -> K128N64");
    setenv("PHASESHIFT_PSQ_PREFILL_2D", "bn128", 1);
    check(is_psq4(sel_psq4(512, 1024, 2560, 2560), Psq4GemmConfigId::Prefill2D_K64N128),
          "psq4 override bn128 -> K64N128");
    setenv("PHASESHIFT_PSQ_PREFILL_2D", "bk128bn128", 1);
    check(is_psq4(sel_psq4(512, 1024, 2560, 2560), Psq4GemmConfigId::Prefill2D_K128N128),
          "psq4 override bk128bn128 -> K128N128");
    setenv("PHASESHIFT_PSQ_PREFILL_2D", "bk128", 1);
    check(is_psq4(sel_psq4(512, 1024, 2560, 64), Psq4GemmConfigId::Prefill2D_K64N64),
          "psq4 invalid override falls back -> K64N64");
    setenv("PHASESHIFT_PSQ_PREFILL_2D", "bk128", 1);
    check(is_psq4(sel_psq4(512, 1024, 2560, 32), Psq4GemmConfigId::RowBlock8),
          "psq4 invalid override not divisible -> RowBlock8");
    unsetenv("PHASESHIFT_PSQ_PREFILL_2D");
    check(is_psq4(sel_psq4(512, 1024, 2560, 2560), Psq4GemmConfigId::Prefill2D_K128N64),
          "psq4 override cleared -> K128N64");
}

void test_psq8() {
    check(!sel_psq8(100, 999, 8888, 8896).has_value(),
          "psq8 unknown geometry -> no config");
    check(!sel_psq8(0, 1024, 2560, 2560).has_value(),
          "psq8 rows=0 -> no config");

    check(is_psq8(sel_psq8(1, 1024, 2560, 2560), Psq8GemmConfigId::RowBlock1),
          "psq8 rows=1 -> RowBlock1");
    check(is_psq8(sel_psq8(16, 1024, 2560, 2560), Psq8GemmConfigId::RowBlock1),
          "psq8 rows=16 -> RowBlock1");
    check(is_psq8(sel_psq8(17, 1024, 2560, 2560), Psq8GemmConfigId::RowBlock2),
          "psq8 rows=17 -> RowBlock2");
    check(is_psq8(sel_psq8(64, 1024, 2560, 2560), Psq8GemmConfigId::RowBlock4),
          "psq8 rows=64 -> RowBlock4");
    check(is_psq8(sel_psq8(128, 1024, 2560, 2560), Psq8GemmConfigId::RowBlock8),
          "psq8 rows=128 -> RowBlock8");
    check(is_psq8(sel_psq8(256, 1024, 2560, 2560), Psq8GemmConfigId::RowBlock8),
          "psq8 rows=256 -> RowBlock8");
    check(is_psq8(sel_psq8(512, 1024, 2560, 2560), Psq8GemmConfigId::Prefill2D_K128N128),
          "psq8 2d rows=512 out=1024 -> Prefill2D_K128N128");
    check(is_psq8(sel_psq8(1024, 2560, 4096, 4096), Psq8GemmConfigId::Prefill2D_K128N128),
          "psq8 2d rows=1024 out=2560 -> Prefill2D_K128N128");
    check(is_psq8(sel_psq8(512, 1024, 2560, 2624), Psq8GemmConfigId::Prefill2D_K64N128),
          "psq8 2d k%128!=0 -> Prefill2D_K64N128");
    check(is_psq8(sel_psq8(511, 1024, 2560, 2560), Psq8GemmConfigId::RowBlock8),
          "psq8 rows=511 < 2d threshold -> RowBlock8");
}

void test_fp8_block128() {
    check(!sel_fp8(100, 999, 8888, 8896).has_value(),
          "fp8 unknown geometry -> no config");
    check(!sel_fp8(0, 1024, 2560, 2560).has_value(),
          "fp8 rows=0 -> no config");
    check(!sel_fp8(1, 1024, 2560, 2561).has_value(),
          "fp8 k_padded not a multiple of 128 -> no config");

    check(is_fp8(sel_fp8(1, 1024, 2560, 2560), Fp8Block128GemmConfigId::RowBlock1),
          "fp8 rows=1 -> RowBlock1");
    check(is_fp8(sel_fp8(16, 1024, 2560, 2560), Fp8Block128GemmConfigId::RowBlock1),
          "fp8 rows=16 -> RowBlock1");
    check(is_fp8(sel_fp8(17, 1024, 2560, 2560), Fp8Block128GemmConfigId::RowBlock2),
          "fp8 rows=17 -> RowBlock2");
    check(is_fp8(sel_fp8(64, 1024, 2560, 2560), Fp8Block128GemmConfigId::RowBlock4),
          "fp8 rows=64 -> RowBlock4");
    check(is_fp8(sel_fp8(128, 1024, 2560, 2560), Fp8Block128GemmConfigId::RowBlock8),
          "fp8 rows=128 -> RowBlock8");
    check(is_fp8(sel_fp8(256, 1024, 2560, 2560), Fp8Block128GemmConfigId::RowBlock8),
          "fp8 rows=256 -> RowBlock8");
    check(is_fp8(sel_fp8(512, 1024, 2560, 2560), Fp8Block128GemmConfigId::Prefill2D),
          "fp8 2d rows=512 out=1024 -> Prefill2D");
    check(is_fp8(sel_fp8(511, 1024, 2560, 2560), Fp8Block128GemmConfigId::RowBlock8),
          "fp8 rows=511 < 2d threshold -> RowBlock8");
    check(is_fp8(sel_fp8(512, 32, 2560, 2560), Fp8Block128GemmConfigId::RowBlock1),
          "fp8 small out_features -> RowBlock1");
}

void test_mxfp4() {
    check(!sel_mxfp4(100, 999, 8888, 8896).has_value(),
          "mxfp4 unknown geometry -> no config");
    check(!sel_mxfp4(0, 1024, 2560, 2560).has_value(),
          "mxfp4 rows=0 -> no config");
    check(!sel_mxfp4(1, 1024, 2560, 2559).has_value(),
          "mxfp4 k_padded not a multiple of 32 -> no config");

    check(is_mxfp4(sel_mxfp4(1, 1024, 2560, 2560), Mxfp4GemmConfigId::RowBlock1),
          "mxfp4 rows=1 -> RowBlock1");
    check(is_mxfp4(sel_mxfp4(17, 1024, 2560, 2560), Mxfp4GemmConfigId::RowBlock2),
          "mxfp4 rows=17 -> RowBlock2");
    check(is_mxfp4(sel_mxfp4(64, 1024, 2560, 2560), Mxfp4GemmConfigId::RowBlock4),
          "mxfp4 rows=64 -> RowBlock4");
    check(is_mxfp4(sel_mxfp4(128, 1024, 2560, 2560), Mxfp4GemmConfigId::RowBlock8),
          "mxfp4 rows=128 -> RowBlock8");
    check(is_mxfp4(sel_mxfp4(512, 1024, 2560, 2560), Mxfp4GemmConfigId::Prefill2D),
          "mxfp4 2d rows=512 out=1024 -> Prefill2D");
    check(is_mxfp4(sel_mxfp4(511, 1024, 2560, 2560), Mxfp4GemmConfigId::RowBlock8),
          "mxfp4 rows=511 < 2d threshold -> RowBlock8");
    check(is_mxfp4(sel_mxfp4(512, 32, 2560, 2560), Mxfp4GemmConfigId::RowBlock1),
          "mxfp4 small out_features -> RowBlock1");
}

}  // namespace

int main() {
    test_bf16();
    test_psq4_row_blocks();
    test_psq4_prefill_2d();
    test_psq8();
    test_fp8_block128();
    test_mxfp4();

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
