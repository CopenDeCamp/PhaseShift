#include <phaseshift/models/qwen35/runtime/rmsnorm_selector.h>

#include <climits>
#include <cstdio>

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

using ps::qwen35::runtime::RmsNormImplementation;
using ps::qwen35::runtime::RmsNormSelectorWeightLayout;
using ps::qwen35::runtime::RmsNormSelectorInput;
using DT = ::ps::runtime::ValueDType;

ps::qwen35::runtime::RmsNormImplementation
sel(DT in, DT out, DT wt, RmsNormSelectorWeightLayout layout, uint32_t mode,
    uint32_t features, uint32_t group, uint32_t rows) {
    return ps::qwen35::runtime::select_rmsnorm_implementation(
        {in, out, wt, layout, mode, features, group, rows});
}

}  // namespace

int main() {
    check(sel(DT::BF16, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerFeature,
              0, 2560, 2560, 1) == RmsNormImplementation::Optimized,
          "hidden norm BF16->BF16 PF ONE_PLUS rows=1 -> Optimized");
    check(sel(DT::BF16, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerFeature,
              0, 2560, 2560, 2048) == RmsNormImplementation::Optimized,
          "hidden norm rows=2048 -> Optimized");
    check(sel(DT::BF16, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerFeature,
              0, 2560, 2560, 0) == RmsNormImplementation::Correctness,
          "hidden norm rows=0 -> Correctness");

    check(sel(DT::BF16, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 4096, 256, 1) == RmsNormImplementation::Optimized,
          "Q norm BF16->F32 PG ONE_PLUS F=4096 rows=1 -> Optimized");
    check(sel(DT::BF16, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 1024, 256, 1) == RmsNormImplementation::Optimized,
          "K norm BF16->F32 PG ONE_PLUS F=1024 rows=1 -> Optimized");
    check(sel(DT::F32, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              1, 4096, 128, 1) == RmsNormImplementation::Optimized,
          "GDN norm F32->F32 PG DIRECT rows=1 -> Optimized");
    check(sel(DT::BF16, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerFeature,
              0, 2560, 2560, 2049) == RmsNormImplementation::Correctness,
          "rows=2049 (unmeasured) -> Correctness");

    check(sel(DT::BF16, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 2560, 2560, 1) == RmsNormImplementation::Correctness,
          "wrong layout (PG for hidden PF profile) -> Correctness");
    check(sel(DT::BF16, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerFeature,
              1, 2560, 2560, 1) == RmsNormImplementation::Correctness,
          "wrong mode (DIRECT for ONE_PLUS profile) -> Correctness");
    check(sel(DT::F32, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerFeature,
              0, 2560, 2560, 1) == RmsNormImplementation::Correctness,
          "unsupported dtype (F32->BF16) -> Correctness");

    check(sel(DT::BF16, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerFeature,
              0, 1234, 1234, 1) == RmsNormImplementation::Correctness,
          "unknown features 1234 -> Correctness");
    check(sel(DT::BF16, DT::BF16, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 2560, 999, 1) == RmsNormImplementation::Correctness,
          "unknown group 999 -> Correctness");

    check(sel(DT::F32, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              1, 6144, 128, 1) == RmsNormImplementation::Optimized,
          "27B GDN norm F32->F32 PG DIRECT F=6144 G=128 rows=1 -> Optimized");
    check(sel(DT::F32, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              1, 6144, 128, 2048) == RmsNormImplementation::Optimized,
          "27B GDN norm rows=2048 -> Optimized");
    check(sel(DT::F32, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              1, 6144, 128, 2049) == RmsNormImplementation::Correctness,
          "27B GDN norm rows=2049 -> Correctness");
    check(sel(DT::BF16, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 6144, 256, 1) == RmsNormImplementation::Optimized,
          "27B q_norm BF16->F32 PG ONE_PLUS F=6144 G=256 rows=1 -> Optimized");
    check(sel(DT::BF16, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 6144, 256, 2048) == RmsNormImplementation::Optimized,
          "27B q_norm rows=2048 -> Optimized");
    check(sel(DT::BF16, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              0, 6144, 128, 1) == RmsNormImplementation::Correctness,
          "27B q_norm wrong group -> Correctness");
    check(sel(DT::F32, DT::F32, DT::BF16, RmsNormSelectorWeightLayout::PerGroup,
              1, 6144, 256, 1) == RmsNormImplementation::Correctness,
          "27B GDN norm wrong group -> Correctness");

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
