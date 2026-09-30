#include <phaseshift/models/qwen35/runtime/lm_head_proxy.h>

#include <cstdint>
#include <cstdio>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* label) {
    if (cond) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", label);
    }
}

using ps::qwen35::runtime::LmHeadConstrainedPath;
using ps::qwen35::runtime::LmHeadProxyPath;
using ps::qwen35::runtime::lm_head_proxy_path;
using ps::qwen35::runtime::select_lm_head_constrained;
using ps::runtime::ExecutionRole;

}  // namespace

int main() {
    const ExecutionRole decode = ExecutionRole::Decode;
    const ExecutionRole verify = ExecutionRole::Verify;
    const ExecutionRole prefill = ExecutionRole::Prefill;
    const bool unconstrained = false;
    const bool constrained = true;

    check(lm_head_proxy_path(decode, 0u, unconstrained, 1u) == LmHeadProxyPath::Fast,
          "decode greedy mode1 is fast");
    check(lm_head_proxy_path(decode, 0u, unconstrained, 2u) == LmHeadProxyPath::Shadow,
          "decode greedy mode2 is shadow");
    check(lm_head_proxy_path(decode, 0u, unconstrained, 0u) == LmHeadProxyPath::None,
          "decode mode0 is off");
    check(lm_head_proxy_path(decode, 1u, unconstrained, 1u) == LmHeadProxyPath::None,
          "decode stochastic mode1 is off");
    check(lm_head_proxy_path(decode, 3u, unconstrained, 2u) == LmHeadProxyPath::None,
          "decode stochastic mode2 is off");
    check(lm_head_proxy_path(decode, 0u, constrained, 1u) == LmHeadProxyPath::None,
          "decode constrained mode1 is off");
    check(lm_head_proxy_path(decode, 0u, constrained, 2u) == LmHeadProxyPath::None,
          "decode constrained mode2 is off");
    check(lm_head_proxy_path(verify, 0u, unconstrained, 1u) == LmHeadProxyPath::Fast,
          "verify mode1 stays fast");
    check(lm_head_proxy_path(verify, 0u, unconstrained, 0u) == LmHeadProxyPath::None,
          "verify mode0 is off");
    check(lm_head_proxy_path(verify, 0u, unconstrained, 2u) == LmHeadProxyPath::None,
          "verify mode2 never consumes proxy output");
    check(lm_head_proxy_path(verify, 0u, constrained, 1u) == LmHeadProxyPath::None,
          "verify constrained is off");
    check(lm_head_proxy_path(prefill, 0u, unconstrained, 1u) == LmHeadProxyPath::None,
          "prefill is off");
    check(lm_head_proxy_path(prefill, 0u, unconstrained, 2u) == LmHeadProxyPath::None,
          "prefill shadow is off");

    const uint32_t kThreshold = 32u;
    const uint32_t kCoarsePool = 32u;
    const uint32_t kUnconstrained = UINT32_MAX;

    {
        const uint32_t counts[] = {7u, 12u, 30u};
        const auto sel = select_lm_head_constrained(0u, 3u, 3u, counts, 3u, kThreshold, kCoarsePool);
        check(sel.path == LmHeadConstrainedPath::ExactCandidates,
              "all small rows take exact candidates");
        check(sel.candidate_capacity == 30u, "capacity is max allowed count");
    }
    {
        const uint32_t counts[] = {33u, 40u, 64u};
        const auto sel = select_lm_head_constrained(0u, 3u, 3u, counts, 3u, kThreshold, kCoarsePool);
        check(sel.path == LmHeadConstrainedPath::MaskedCoarse,
              "all large rows take masked coarse");
        check(sel.candidate_capacity == 0u, "masked coarse has no candidate capacity");
    }
    {
        const uint32_t counts[] = {8u, 64u};
        const auto sel = select_lm_head_constrained(0u, 2u, 2u, counts, 2u, kThreshold, kCoarsePool);
        check(sel.path == LmHeadConstrainedPath::None, "mixed small and large falls back");
    }
    {
        const uint32_t counts[] = {8u, kUnconstrained};
        const auto sel = select_lm_head_constrained(0u, 2u, 2u, counts, 2u, kThreshold, kCoarsePool);
        check(sel.path == LmHeadConstrainedPath::None,
              "mixed constrained and unconstrained falls back");
    }
    {
        const uint32_t counts[] = {0u, 8u};
        const auto sel = select_lm_head_constrained(0u, 2u, 2u, counts, 2u, kThreshold, kCoarsePool);
        check(sel.path == LmHeadConstrainedPath::None, "zero allowed count falls back");
    }
    {
        const uint32_t counts[] = {8u};
        check(select_lm_head_constrained(1u, 1u, 1u, counts, 1u, kThreshold, kCoarsePool).path ==
                  LmHeadConstrainedPath::None,
              "stochastic constrained falls back");
        check(select_lm_head_constrained(0u, 0u, 1u, counts, 1u, kThreshold, kCoarsePool).path ==
                  LmHeadConstrainedPath::None,
              "unsampled rows fall back");
        check(select_lm_head_constrained(0u, 1u, 1u, nullptr, 1u, kThreshold, kCoarsePool).path ==
                  LmHeadConstrainedPath::None,
              "missing counts fall back");
        check(select_lm_head_constrained(0u, 1u, 1u, counts, 0u, kThreshold, kCoarsePool).path ==
                  LmHeadConstrainedPath::None,
              "zero rows fall back");
        check(select_lm_head_constrained(0u, 1u, 1u, counts, 1u, 0u, kCoarsePool).path ==
                  LmHeadConstrainedPath::None,
              "zero threshold falls back");
    }
    {
        const uint32_t counts[] = {32u};
        check(select_lm_head_constrained(0u, 1u, 1u, counts, 1u, kThreshold, kCoarsePool).path ==
                  LmHeadConstrainedPath::ExactCandidates,
              "count equal to threshold stays exact");
        const uint32_t counts2[] = {33u};
        check(select_lm_head_constrained(0u, 1u, 1u, counts2, 1u, kThreshold, kCoarsePool).path ==
                  LmHeadConstrainedPath::MaskedCoarse,
              "count above threshold goes masked coarse");
    }
    {
        const uint32_t counts[] = {17u, 20u};
        check(select_lm_head_constrained(0u, 2u, 2u, counts, 2u, 16u, 32u).path ==
                  LmHeadConstrainedPath::None,
              "allowed count below coarse pool falls back");
        const uint32_t counts2[] = {40u, 50u};
        check(select_lm_head_constrained(0u, 2u, 2u, counts2, 2u, 16u, 32u).path ==
                  LmHeadConstrainedPath::MaskedCoarse,
              "allowed count above coarse pool takes masked coarse");
        check(select_lm_head_constrained(0u, 2u, 2u, counts2, 2u, 16u, 0u).path ==
                  LmHeadConstrainedPath::None,
              "zero coarse pool falls back");
    }

    std::printf("test_lm_head_proxy_path: passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
