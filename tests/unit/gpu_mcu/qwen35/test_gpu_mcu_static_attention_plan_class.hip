#include <phaseshift/models/qwen35/runtime/paged_attention_dispatch.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/execution/execution_types.h>
#include <phaseshift/runtime/program/program.h>

#include <cstdint>
#include <cstdio>

namespace {

namespace rt = ps::runtime;
namespace q35 = ps::qwen35::runtime;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kQHeads = 16u;
constexpr uint32_t kKvHeads = 4u;
constexpr uint32_t kPageTokens = 16u;
constexpr uint32_t kQFeatures = kQHeads * kHeadDim;
constexpr uint32_t kNumPages = 64u;
constexpr uint32_t kNumLayers = 4u;
constexpr uint32_t kBlockTableStride = 4096u;
constexpr uint32_t kOutOfEnvelopeVisible = 40000u;

uint16_t g_q_storage[kQFeatures] = {};
float g_out_storage[kQFeatures] = {};
uint32_t g_row_slots[8] = {};
uint32_t g_row_positions[8] = {};
uint32_t g_block_tables[8] = {};
uint8_t g_partials[64] = {};
rt::DeviceRequestDescriptor g_requests[1] = {};

void* g_external_inputs[1] = {g_q_storage};
void* g_external_outputs[1] = {g_out_storage};

const char* class_name(const q35::McuStaticAttentionPlanClass& c) {
    if (!c.has_paged_attention) return "None";
    if (!c.supported) return "Unsupported";
    if (c.splits >= 16u) return "Split16";
    return "Direct";
}

void build_program(rt::Program& program, bool with_paged_attention) {
    program = rt::Program{};
    program.row_bucket = rt::RowBucket::R256;
    program.execution_class = rt::ExecutionClass::DECODE;
    if (!with_paged_attention) return;

    rt::ValueBinding q_value{};
    q_value.logical_value = rt::ValueId{0u};
    q_value.storage = rt::ValueStorage::EXTERNAL_INPUT;
    q_value.dtype = rt::ValueDType::BF16;
    q_value.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    q_value.feature_count = kQFeatures;
    q_value.row_stride = kQFeatures;
    program.values.push_back(q_value);

    rt::ValueBinding out_value{};
    out_value.logical_value = rt::ValueId{1u};
    out_value.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    out_value.dtype = rt::ValueDType::F32;
    out_value.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    out_value.feature_count = kQFeatures;
    out_value.row_stride = kQFeatures;
    program.values.push_back(out_value);

    rt::StateBinding binding{};
    binding.state = rt::StateId{0u};
    binding.kind = rt::PrimitiveStateKind::KV_CACHE;
    binding.state_index = 0u;
    binding.slot = 0u;
    program.states.push_back(binding);

    rt::DispatchBinding attention{};
    attention.kernel_id = rt::KernelId::PAGED_ATTENTION;
    attention.descriptor_slot = 1u;
    attention.input_count = 1u;
    attention.input_slots[0] = 0u;
    attention.output_count = 1u;
    attention.output_slots[0] = 1u;
    attention.group_size = kHeadDim;
    attention.scalar_a = 1.0f;
    program.dispatches.push_back(attention);
    program.dispatch_source_node.push_back(0u);
    program.external_input_count = 1u;
    program.external_output_count = 1u;
}

q35::HostExecutionContext make_context(rt::ExecutionRole role,
                                       uint32_t live_rows,
                                       uint32_t visible_tokens,
                                       bool with_prefill_request) {
    q35::HostExecutionContext ctx{};
    ctx.external_inputs = g_external_inputs;
    ctx.external_input_count = 1u;
    ctx.external_outputs = g_external_outputs;
    ctx.external_output_count = 1u;
    ctx.actual_rows = live_rows;
    ctx.max_visible_tokens = visible_tokens;
    ctx.role = role;
    ctx.row_sequence_slots = g_row_slots;
    ctx.row_positions = g_row_positions;
    ctx.decode_attn_partials = g_partials;
    ctx.decode_attn_partial_bytes = 0xFFFFFFFFu;

    ctx.model_state.kv_dtype = 0u;
    auto& bf16 = ctx.model_state.kv_bf16;
    bf16.k_pool = reinterpret_cast<decltype(bf16.k_pool)>(g_q_storage);
    bf16.v_pool = reinterpret_cast<decltype(bf16.v_pool)>(g_out_storage);
    bf16.page_tokens = kPageTokens;
    bf16.num_pages = kNumPages;
    bf16.num_attention_layers = kNumLayers;
    bf16.kv_heads = kKvHeads;
    bf16.head_dim = kHeadDim;
    bf16.elems_per_token = kKvHeads * kHeadDim;
    bf16.elems_per_page = kPageTokens * kKvHeads * kHeadDim;
    bf16.elems_per_layer = kNumPages * kPageTokens * kKvHeads * kHeadDim;

    auto& block_table = ctx.model_state.block_table;
    block_table.block_tables = g_block_tables;
    block_table.max_sequences = 8u;
    block_table.block_table_stride = kBlockTableStride;

    if (with_prefill_request) {
        g_requests[0] = rt::DeviceRequestDescriptor{};
        g_requests[0].execution_class = rt::ExecutionClass::PREFILL;
        g_requests[0].row_begin = 0u;
        g_requests[0].row_count = live_rows;
        ctx.staged_requests = g_requests;
        ctx.staged_request_count = 1u;
    }
    return ctx;
}

}  // namespace

int main() {
    rt::Program program{};
    build_program(program, true);

    std::printf("=== [6-A] capacity driven classification ===\n");
    for (uint32_t rows : {127u, 128u, 129u}) {
        q35::HostExecutionContext ctx =
            make_context(rt::ExecutionRole::Decode, rows, 512u, false);
        const auto r =
            q35::classify_static_attention_plan(program, ctx, rows);
        if (!r.ok()) {
            std::printf("FAIL: rows=%u classify: %s\n", rows,
                        r.status().message().c_str());
            ++g_failed;
            continue;
        }
        const auto& c = r.value();
        std::printf("[attn-class] rows=%u key=%u class=%s canonical=%u\n",
                    rows, c.key, class_name(c), c.canonical_max_visible);
        check(c.supported, "rows classify supported");
    }

    std::printf("=== [6-B] classification ignores current execution class ===\n");
    for (uint32_t rows : {127u, 128u, 129u}) {
        for (uint32_t visible : {512u, 4096u}) {
            const q35::HostExecutionContext decode_ctx =
                make_context(rt::ExecutionRole::Decode, rows, visible, false);
            const q35::HostExecutionContext prefill_ctx =
                make_context(rt::ExecutionRole::Prefill, rows, visible, true);
            const auto d =
                q35::classify_static_attention_plan(program, decode_ctx, rows);
            const auto p =
                q35::classify_static_attention_plan(program, prefill_ctx, rows);
            if (!d.ok() || !p.ok()) {
                std::printf("FAIL: rows=%u visible=%u classify role pair\n",
                            rows, visible);
                ++g_failed;
                continue;
            }
            std::printf(
                "[attn-class] rows=%u visible=%u decode{key=%u class=%s} "
                "prefill{key=%u class=%s}\n",
                rows, visible, d.value().key, class_name(d.value()),
                p.value().key, class_name(p.value()));
            check(d.value().key == p.value().key,
                  "execution class does not change the static plan key");
            check(d.value().splits == p.value().splits,
                  "execution class does not change the split class");
            check(d.value().canonical_max_visible ==
                      p.value().canonical_max_visible,
                  "execution class does not change the canonical selector input");
            check(d.value().supported == p.value().supported,
                  "execution class does not change support");
        }
    }

    std::printf("=== [6-C]/[6-D]/[6-E] visible boundary ===\n");
    const uint32_t direct_visible[3] = {512u, 1024u, 2047u};
    const uint32_t split_visible[2] = {2048u, 4096u};
    uint32_t direct_key = 0u;
    uint32_t split_key = 0u;
    bool direct_key_ok = true;
    bool split_key_ok = true;
    for (uint32_t visible : direct_visible) {
        q35::HostExecutionContext ctx =
            make_context(rt::ExecutionRole::Decode, 128u, visible, false);
        const auto r = q35::classify_static_attention_plan(program, ctx, 128u);
        if (!r.ok()) {
            std::printf("FAIL: visible=%u classify: %s\n", visible,
                        r.status().message().c_str());
            ++g_failed;
            continue;
        }
        const auto& c = r.value();
        std::printf("[attn-class] visible=%-5u key=%u class=%s canonical=%u\n",
                    visible, c.key, class_name(c), c.canonical_max_visible);
        check(c.supported, "direct visible supported");
        check(c.splits == 1u, "direct class uses one region");
        check(c.canonical_max_visible == 1u,
              "direct canonical selector input is 1");
        if (direct_key == 0u) {
            direct_key = c.key;
        } else if (direct_key != c.key) {
            direct_key_ok = false;
        }
    }
    for (uint32_t visible : split_visible) {
        q35::HostExecutionContext ctx =
            make_context(rt::ExecutionRole::Decode, 128u, visible, false);
        const auto r = q35::classify_static_attention_plan(program, ctx, 128u);
        if (!r.ok()) {
            std::printf("FAIL: visible=%u classify: %s\n", visible,
                        r.status().message().c_str());
            ++g_failed;
            continue;
        }
        const auto& c = r.value();
        std::printf("[attn-class] visible=%-5u key=%u class=%s canonical=%u\n",
                    visible, c.key, class_name(c), c.canonical_max_visible);
        check(c.supported, "split visible supported");
        check(c.splits == 16u, "split class uses sixteen regions");
        check(c.canonical_max_visible == q35::kPagedAttentionSplitMinVisible,
              "split canonical selector input is the split threshold");
        if (split_key == 0u) {
            split_key = c.key;
        } else if (split_key != c.key) {
            split_key_ok = false;
        }
    }
    check(direct_key_ok, "direct visible values share one structural key");
    check(split_key_ok, "split visible values share one structural key");
    check(direct_key != 0u && split_key != 0u && direct_key != split_key,
          "direct key differs from split key");

    std::printf("=== [13] selector geometry is capacity ===\n");
    {
        q35::HostExecutionContext ctx =
            make_context(rt::ExecutionRole::Decode, 4096u, 512u, false);
        const auto r = q35::classify_static_attention_plan(program, ctx, 128u);
        if (r.ok()) {
            std::printf("[attn-class] live=4096 capacity=128 class=%s\n",
                        class_name(r.value()));
            check(r.value().supported,
                  "selector geometry follows row capacity not live rows");
        } else {
            check(false, "capacity driven classification resolves");
        }
    }
    {
        q35::HostExecutionContext ctx =
            make_context(rt::ExecutionRole::Decode, 128u, 512u, false);
        const auto r = q35::classify_static_attention_plan(program, ctx, 4096u);
        if (r.ok()) {
            std::printf("[attn-class] live=128 capacity=4096 class=%s\n",
                        class_name(r.value()));
            check(!r.value().supported,
                  "out of envelope capacity is not optimized");
        } else {
            check(false, "capacity driven classification resolves");
        }
    }

    std::printf("=== [14] program without paged attention ===\n");
    {
        rt::Program plain{};
        build_program(plain, false);
        q35::HostExecutionContext ctx =
            make_context(rt::ExecutionRole::Decode, 128u, 512u, false);
        const auto r =
            q35::classify_static_attention_plan(plain, ctx, 128u);
        if (!r.ok()) {
            check(false, "missing paged attention still classifies");
        } else {
            const auto& c = r.value();
            std::printf("[attn-class] no-attention has=%d supported=%d key=%u\n",
                        c.has_paged_attention ? 1 : 0, c.supported ? 1 : 0,
                        c.key);
            check(!c.has_paged_attention, "no paged attention detected");
            check(c.supported, "program without paged attention stays supported");
            check(c.key == 0u, "no paged attention key is zero");
        }
    }

    std::printf("=== [15] outside optimized envelope ===\n");
    {
        q35::HostExecutionContext ctx =
            make_context(rt::ExecutionRole::Decode, 128u,
                         kOutOfEnvelopeVisible, false);
        const auto r = q35::classify_static_attention_plan(program, ctx, 128u);
        if (!r.ok()) {
            check(false, "envelope classification resolves");
        } else {
            const auto& c = r.value();
            std::printf("[attn-class] visible=%u class=%s supported=%d\n",
                        kOutOfEnvelopeVisible, class_name(c),
                        c.supported ? 1 : 0);
            check(!c.supported,
                  "live batch outside envelope is not optimized away");
            check(c.canonical_max_visible == 0u,
                  "unsupported class has no canonical selector input");
        }
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_static_attention_plan_class: %d failures\n",
                    g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_static_attention_plan_class: PASS\n");
    return 0;
}
