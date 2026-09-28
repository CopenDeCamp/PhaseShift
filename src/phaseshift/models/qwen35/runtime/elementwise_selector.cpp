#include <phaseshift/models/qwen35/runtime/elementwise_selector.h>

namespace ps::qwen35::runtime {
namespace {

struct Rule {
    ElementwiseProfile profile;
    uint32_t features;
    uint32_t aux;
    uint32_t min_rows;
    uint32_t max_rows;
};

constexpr Rule kRules[] = {
    {ElementwiseProfile::ResidualBf16, 2560u, 0u, 1u, 2048u},
    {ElementwiseProfile::SwigluBf16, 9216u, 0u, 1u, 2048u},
    {ElementwiseProfile::SigmoidBf16ToF32, 4096u, 0u, 1u, 2048u},
    {ElementwiseProfile::MulF32F32ToBf16, 4096u, 0u, 1u, 2048u},
    {ElementwiseProfile::SplitBf16InterleavedHeads, 4096u, 256u, 1u, 2048u},
    {ElementwiseProfile::ResidualBf16, 5120u, 0u, 1u, 2048u},
    {ElementwiseProfile::SwigluBf16, 17408u, 0u, 1u, 2048u},
    {ElementwiseProfile::SigmoidBf16ToF32, 6144u, 0u, 1u, 2048u},
    {ElementwiseProfile::MulF32F32ToBf16, 6144u, 0u, 1u, 2048u},
    {ElementwiseProfile::SplitBf16InterleavedHeads, 6144u, 256u, 1u, 2048u},
    {ElementwiseProfile::SiluF32ToBf16, 10240u, 0u, 1u, 2048u},
    {ElementwiseProfile::SiluF32ToBf16, 8192u, 0u, 1u, 2048u},
    {ElementwiseProfile::SiluBf16ToF32, 6144u, 0u, 1u, 2048u},
    {ElementwiseProfile::SiluBf16ToF32, 4096u, 0u, 1u, 2048u},
    {ElementwiseProfile::ScaleF32, 2048u, 0u, 1u, 2048u},
};

}  // namespace

ElementwiseImplementation select_elementwise_implementation(
    const ElementwiseSelectorInput& in) {
    for (const auto& r : kRules) {
        if (r.profile != in.profile) continue;
        if (r.features != in.features) continue;
        if (r.aux != in.aux) continue;
        if (in.rows < r.min_rows || in.rows > r.max_rows) continue;
        return ElementwiseImplementation::Optimized;
    }
    return ElementwiseImplementation::Correctness;
}

}  // namespace ps::qwen35::runtime
