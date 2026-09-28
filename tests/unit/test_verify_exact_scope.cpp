#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <cstdio>

int main() {
    using ::ps::runtime::ExecutionRole;
    using ::ps::runtime::VerifyNumericMode;
    using ::ps::qwen35::runtime::HostExecutionContext;
    using ::ps::qwen35::runtime::verify_exact_active;

    int failed = 0;
    auto check = [&](bool cond, const char* name) {
        std::printf("%s: %s\n", cond ? "PASS" : "FAIL", name);
        if (!cond) ++failed;
    };

    HostExecutionContext ctx{};
    ctx.role = ExecutionRole::Decode;
    ctx.numeric_mode = VerifyNumericMode::Exact;
    check(!verify_exact_active(ctx),
          "decode role never uses verify-exact (M=1 fast path preserved)");

    ctx.role = ExecutionRole::Prefill;
    ctx.numeric_mode = VerifyNumericMode::Exact;
    check(!verify_exact_active(ctx), "prefill role never uses verify-exact");

    ctx.role = ExecutionRole::Verify;
    ctx.numeric_mode = VerifyNumericMode::Fast;
    check(!verify_exact_active(ctx), "verify role with Fast mode uses fast kernels");

    ctx.role = ExecutionRole::Verify;
    ctx.numeric_mode = VerifyNumericMode::Exact;
    check(verify_exact_active(ctx), "verify role with Exact mode uses exact kernels");

    std::printf("VERIFY_EXACT_SCOPE: %s\n", failed == 0 ? "PASS" : "FAIL");
    return failed == 0 ? 0 : 1;
}
