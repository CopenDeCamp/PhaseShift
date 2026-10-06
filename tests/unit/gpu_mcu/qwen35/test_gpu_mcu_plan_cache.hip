#include <phaseshift/models/qwen35/runtime/mcu_plan_cache.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_value.h>

#include <cstdint>
#include <cstdio>
#include <utility>

namespace {

namespace q35 = ::ps::qwen35::runtime;
namespace mcu = ::ps::runtime::gpu_mcu;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

q35::McuCompiledPlan make_plan(uint64_t base, uint32_t rows,
                               uint32_t grid) {
    q35::McuCompiledPlan plan{};
    for (uint32_t i = 0; i < 2u; ++i) {
        auto inv = q35::mcu_plan_value<mcu::McuKvAppendInvocation>();
        inv.k_input = base + 0x1000ull * i;
        inv.v_input = base + 0x2000ull * i;
        inv.k_pool = base + 0x3000ull * i;
        inv.v_pool = base + 0x4000ull * i;
        inv.row_sequence_slots = base + 0x5000ull;
        inv.row_positions = base + 0x6000ull;
        inv.block_tables = base + 0x7000ull;
        inv.rows = rows;
        inv.layer = i;
        inv.num_pages = 8u;
        inv.head_dim = 128u;
        plan.kv_append.push_back(inv);

        auto ew = q35::mcu_plan_value<mcu::McuElementwiseInvocation>();
        ew.in0 = base + 0x8000ull * i;
        ew.out0 = base + 0x9000ull * i;
        ew.rows = rows;
        ew.features = 5120u;
        plan.elementwise.push_back(ew);

        auto node = q35::mcu_plan_value<mcu::McuPlanNode>();
        node.variant_id = static_cast<uint16_t>(i);
        node.next = mcu::kMcuNoNext;
        node.kernarg_recipe = mcu::kMcuKernargRecipeKvAppend;
        node.invocation_index = i;
        node.value = grid;
        plan.nodes.push_back(node);

        auto variant = q35::mcu_plan_value<q35::McuCompiledVariant>();
        variant.kind = q35::McuCompiledVariantKind::KvAppendBf16;
        variant.grid = grid;
        variant.grid_y = 1u;
        variant.grid_z = 1u;
        variant.workgroup = 64u;
        plan.variants.push_back(variant);
    }
    plan.dispatch_count = 2u;
    plan.logical_dispatch_count = 2u;
    plan.physical_dispatch_count = 2u;
    plan.kernarg_slots_required = 2u;
    return plan;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    const q35::McuCompiledPlan plan_a = make_plan(0x100000ull, 1u, 4u);
    const q35::McuCompiledPlan plan_a_again = make_plan(0x100000ull, 1u, 4u);
    const q35::McuCompiledPlan plan_b = make_plan(0x200000ull, 1u, 4u);
    const q35::McuCompiledPlan plan_rows = make_plan(0x100000ull, 2u, 4u);
    const q35::McuCompiledPlan plan_grid = make_plan(0x100000ull, 1u, 8u);

    const q35::McuPlanFingerprint fp_a = q35::mcu_plan_fingerprint(plan_a);
    const q35::McuPlanFingerprint fp_a_again =
        q35::mcu_plan_fingerprint(plan_a_again);
    const q35::McuPlanFingerprint fp_b = q35::mcu_plan_fingerprint(plan_b);
    const q35::McuPlanFingerprint fp_rows =
        q35::mcu_plan_fingerprint(plan_rows);
    const q35::McuPlanFingerprint fp_grid =
        q35::mcu_plan_fingerprint(plan_grid);

    check(fp_a.hash != 0u, "fingerprint is not trivial");
    check(fp_a == fp_a_again, "identical plans hash the same");
    check(fp_a != fp_b, "different device pointers hash differently");
    check(fp_a != fp_rows, "different rows hash differently");
    check(fp_a != fp_grid, "different variant grid hashes differently");
    check(fp_a.dispatch_count == 2u, "dispatch count is carried over");
    check(fp_a.variant_count == 2u, "variant count is carried over");

    {
        q35::McuCompiledPlan reordered = plan_a;
        const mcu::McuPlanNode tmp = reordered.nodes[0];
        reordered.nodes[0] = reordered.nodes[1];
        reordered.nodes[1] = tmp;
        check(q35::mcu_plan_fingerprint(reordered) != fp_a,
              "node order changes the fingerprint");
    }
    {
        q35::McuCompiledPlan empty{};
        const q35::McuPlanFingerprint fp = q35::mcu_plan_fingerprint(empty);
        check(fp.hash != 0u, "empty plan still hashes");
        check(fp.dispatch_count == 0u, "empty plan has no dispatch");
    }

    q35::McuPlanCache cache{};
    check(!cache.has_active(), "cache starts empty");

    cache.note_compile();
    const q35::McuPlanCacheStep first =
        cache.observe(q35::McuPlanShape::Direct, fp_a);
    check(first.needs_switch && first.needs_upload, "first plan installs");
    check(!first.reused, "first plan is not reused");
    check(cache.has_active(), "cache becomes active");
    check(cache.counters().plan_compile_count == 1u, "compile counted 1");
    check(cache.counters().plan_upload_count == 1u, "upload counted 1");
    check(cache.counters().plan_switch_count == 1u, "switch counted 1");

    const q35::McuPlanCacheStep second =
        cache.observe(q35::McuPlanShape::Direct, fp_a);
    check(second.reused, "identical signature is reused");
    check(!second.needs_switch && !second.needs_upload,
          "reuse does not switch or upload");
    check(cache.counters().plan_compile_count == 1u,
          "catalog reuse does not compile");
    check(cache.counters().plan_upload_count == 1u,
          "reuse does not upload the plan");
    check(cache.counters().plan_switch_count == 1u,
          "reuse does not restart the mcu plan");

    cache.note_compile();
    const q35::McuPlanCacheStep third =
        cache.observe(q35::McuPlanShape::Direct, fp_b);
    check(third.needs_switch && third.needs_upload,
          "changed pointer switches the plan");
    check(cache.counters().plan_upload_count == 2u, "upload counted 2");
    check(cache.counters().plan_switch_count == 2u, "switch counted 2");

    cache.note_compile();
    const q35::McuPlanCacheStep fourth =
        cache.observe(q35::McuPlanShape::Split, fp_b);
    check(fourth.needs_switch && fourth.needs_upload,
          "changed plan shape switches the plan");
    check(cache.counters().plan_upload_count == 3u, "upload counted 3");
    check(cache.counters().plan_switch_count == 3u, "switch counted 3");
    check(cache.active_shape() == q35::McuPlanShape::Split,
          "active shape follows the switch");

    const q35::McuPlanCacheStep fifth =
        cache.observe(q35::McuPlanShape::Split, fp_b);
    check(fifth.reused, "same shape and signature is reused");
    check(cache.counters().plan_upload_count == 3u,
          "reuse keeps the upload count");
    check(cache.counters().plan_switch_count == 3u,
          "reuse keeps the switch count");

    cache.note_compile();
    const q35::McuPlanCacheStep sixth =
        cache.observe(q35::McuPlanShape::Split, fp_rows);
    check(sixth.needs_switch, "row change switches the plan");
    check(cache.counters().plan_switch_count == 4u, "switch counted 4");

    const uint64_t compiles_before_observe =
        cache.counters().plan_compile_count;
    (void)cache.observe(q35::McuPlanShape::Split, fp_rows);
    check(cache.counters().plan_compile_count == compiles_before_observe,
          "observe alone does not count compiles");
    check(cache.counters().plan_compile_count == 4u,
          "compile count tracks real compiles only");

    cache.note_mcu_run();
    cache.note_mcu_run();
    cache.note_host_fallback();
    check(cache.counters().mcu_run_count == 2u, "mcu run counted");
    check(cache.counters().host_fallback_count == 1u, "fallback counted");

    const q35::DecodeRuntimeCounters observed = cache.counters();
    cache.reset();
    check(!cache.has_active(), "reset clears the active plan");
    check(cache.counters().plan_compile_count == 0u, "reset clears counters");

    std::printf("plan cache: fingerprint=0x%llx\n",
                static_cast<unsigned long long>(fp_a.hash));
    std::printf("plan cache: compiles=%llu uploads=%llu switches=%llu\n",
                static_cast<unsigned long long>(observed.plan_compile_count),
                static_cast<unsigned long long>(observed.plan_upload_count),
                static_cast<unsigned long long>(observed.plan_switch_count));
    if (g_failed != 0) {
        std::printf("test_gpu_mcu_plan_cache: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_plan_cache: PASS\n");
    return 0;
}
