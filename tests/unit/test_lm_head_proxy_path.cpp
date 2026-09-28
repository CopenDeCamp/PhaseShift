#include <phaseshift/models/qwen35/runtime/lm_head_proxy.h>

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

using ps::qwen35::runtime::LmHeadProxyPath;
using ps::qwen35::runtime::lm_head_proxy_path;
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

    std::printf("test_lm_head_proxy_path: passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
