#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/runtime/mcu_decode_runtime.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/models/qwen35/runtime/gdn_conv_dispatch.h>
#include <phaseshift/models/qwen35/runtime/gdn_recurrence_dispatch.h>
#include <phaseshift/models/qwen35/runtime/kv_append_dispatch.h>
#include <phaseshift/models/qwen35/runtime/optimized_dispatch.h>
#include <phaseshift/models/qwen35/runtime/paged_attention_dispatch.h>
#include <phaseshift/models/qwen35/runtime/physical_launch.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/gpu_mcu/execution/persistent_mcu.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

namespace mcu = ps::runtime::gpu_mcu;
namespace rt = ps::runtime;
namespace q35 = ps::qwen35::runtime;
namespace kernel = ps::kernel;
namespace chain = gpu_mcu_chain;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kRows = 26u;
constexpr uint32_t kOutputs = 11u;
constexpr uint32_t kSlots = 4u;
constexpr uint32_t kVerifyStride = 8u;
constexpr uint32_t kVerifyElems = kSlots * kVerifyStride;
constexpr uint32_t kFeatures = chain::kFeatures;
constexpr uint32_t kN = chain::kN;
constexpr uint32_t kKp = chain::kKp;
constexpr uint32_t kNb = chain::kNb;
constexpr uint32_t kTiles = chain::kTiles;

constexpr uint32_t kPageTokens = 16u;
constexpr uint32_t kBlockTableStride = 16u;
constexpr uint32_t kNumPages = 64u;
constexpr uint32_t kMaxSequences = 4u;
constexpr uint32_t kLayers = 1u;
constexpr uint32_t kQHeads = 16u;
constexpr uint32_t kKvHeads = 4u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kMaxVisible = 256u;
constexpr uint32_t kQFeatures = kQHeads * kHeadDim;
constexpr uint32_t kElemsPerToken = kKvHeads * kHeadDim;
constexpr uint32_t kElemsPerPage = kPageTokens * kElemsPerToken;
constexpr uint32_t kElemsPerLayer = kNumPages * kElemsPerPage;
constexpr size_t kPoolBytes = static_cast<size_t>(kElemsPerLayer) * sizeof(uint16_t);

constexpr uint32_t kGdnConvDim = 8192u;
constexpr uint32_t kGdnConvHistory = 3u;
constexpr uint32_t kGdnConvHistoryStride = kGdnConvDim;
constexpr uint32_t kGdnConvLayerStride = kGdnConvHistory * kGdnConvHistoryStride;
constexpr uint32_t kGdnConvSlotStride = kGdnConvLayerStride;
constexpr uint32_t kGdnKeyHeads = 16u;
constexpr uint32_t kGdnNumVHeads = 32u;
constexpr uint32_t kGdnHeadK = 128u;
constexpr uint32_t kGdnHeadV = 128u;
constexpr uint32_t kGdnQkDim = kGdnKeyHeads * kGdnHeadK;
constexpr uint32_t kGdnValueDim = kGdnNumVHeads * kGdnHeadV;
constexpr uint32_t kGdnRecurrentLayerStride = kGdnNumVHeads * kGdnHeadK * kGdnHeadV;
constexpr uint32_t kGdnRecurrentSlotStride = kGdnRecurrentLayerStride;

const rt::Int8ActivationWorkspaceLayout& act_layout() {
    static const rt::Int8ActivationWorkspaceLayout l =
        rt::Int8ActivationWorkspaceLayout::make(kFeatures, kRows);
    return l;
}

uint64_t rms_out_bytes() {
    return static_cast<uint64_t>(kRows) * kFeatures * sizeof(uint16_t);
}

uint64_t workspace_bytes() {
    return rms_out_bytes() + act_layout().total_bytes;
}

void build_program(rt::Program& program) {
    const auto& layout = act_layout();
    program = rt::Program{};
    program.row_bucket = rt::RowBucket::R64;
    program.execution_class = rt::ExecutionClass::PREFILL;
    program.workspace.total_bytes = workspace_bytes();
    program.workspace.ranges.push_back(
        rt::WorkspaceRange{rms_out_bytes(), layout.total_bytes, 0u, 0u});
    program.weight_slot_count = 1u;
    program.parameter_slot_count = 4u;

    rt::ValueBinding in{};
    in.logical_value = rt::ValueId{0u};
    in.storage = rt::ValueStorage::EXTERNAL_INPUT;
    in.dtype = rt::ValueDType::BF16;
    in.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    in.slot = 0u;
    in.feature_count = kFeatures;
    in.row_stride = kFeatures;
    program.values.push_back(in);

    rt::ValueBinding rms_out{};
    rms_out.logical_value = rt::ValueId{1u};
    rms_out.storage = rt::ValueStorage::WORKSPACE;
    rms_out.dtype = rt::ValueDType::BF16;
    rms_out.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    rms_out.slot = 0u;
    rms_out.feature_count = kFeatures;
    rms_out.row_stride = kFeatures;
    rms_out.offset = 0u;
    rms_out.bytes = rms_out_bytes();
    program.values.push_back(rms_out);

    rt::ValueBinding lin_out{};
    lin_out.logical_value = rt::ValueId{2u};
    lin_out.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    lin_out.dtype = rt::ValueDType::BF16;
    lin_out.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    lin_out.slot = 0u;
    lin_out.feature_count = kN;
    lin_out.row_stride = kN;
    program.values.push_back(lin_out);

    rt::ValueBinding q{};
    q.logical_value = rt::ValueId{3u};
    q.storage = rt::ValueStorage::EXTERNAL_INPUT;
    q.dtype = rt::ValueDType::BF16;
    q.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    q.slot = 1u;
    q.feature_count = kQFeatures;
    q.row_stride = kQFeatures;
    program.values.push_back(q);

    rt::ValueBinding attn_out{};
    attn_out.logical_value = rt::ValueId{4u};
    attn_out.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    attn_out.dtype = rt::ValueDType::F32;
    attn_out.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    attn_out.slot = 1u;
    attn_out.feature_count = kQFeatures;
    attn_out.row_stride = kQFeatures;
    program.values.push_back(attn_out);

    rt::ValueBinding k_in{};
    k_in.logical_value = rt::ValueId{5u};
    k_in.storage = rt::ValueStorage::EXTERNAL_INPUT;
    k_in.dtype = rt::ValueDType::BF16;
    k_in.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    k_in.slot = 2u;
    k_in.feature_count = kElemsPerToken;
    k_in.row_stride = kElemsPerToken;
    program.values.push_back(k_in);

    rt::ValueBinding v_in{};
    v_in.logical_value = rt::ValueId{6u};
    v_in.storage = rt::ValueStorage::EXTERNAL_INPUT;
    v_in.dtype = rt::ValueDType::BF16;
    v_in.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    v_in.slot = 3u;
    v_in.feature_count = kElemsPerToken;
    v_in.row_stride = kElemsPerToken;
    program.values.push_back(v_in);

    rt::ValueBinding gconv_in{};
    gconv_in.logical_value = rt::ValueId{7u};
    gconv_in.storage = rt::ValueStorage::EXTERNAL_INPUT;
    gconv_in.dtype = rt::ValueDType::BF16;
    gconv_in.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    gconv_in.slot = 4u;
    gconv_in.feature_count = kGdnConvDim;
    gconv_in.row_stride = kGdnConvDim;
    program.values.push_back(gconv_in);

    rt::ValueBinding gconv_out{};
    gconv_out.logical_value = rt::ValueId{8u};
    gconv_out.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    gconv_out.dtype = rt::ValueDType::F32;
    gconv_out.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    gconv_out.slot = 2u;
    gconv_out.feature_count = kGdnConvDim;
    gconv_out.row_stride = kGdnConvDim;
    program.values.push_back(gconv_out);

    rt::ValueBinding gq{};
    gq.logical_value = rt::ValueId{9u};
    gq.storage = rt::ValueStorage::EXTERNAL_INPUT;
    gq.dtype = rt::ValueDType::F32;
    gq.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    gq.slot = 5u;
    gq.feature_count = kGdnQkDim;
    gq.row_stride = kGdnQkDim;
    program.values.push_back(gq);

    rt::ValueBinding gk{};
    gk.logical_value = rt::ValueId{10u};
    gk.storage = rt::ValueStorage::EXTERNAL_INPUT;
    gk.dtype = rt::ValueDType::F32;
    gk.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    gk.slot = 6u;
    gk.feature_count = kGdnQkDim;
    gk.row_stride = kGdnQkDim;
    program.values.push_back(gk);

    rt::ValueBinding gv{};
    gv.logical_value = rt::ValueId{11u};
    gv.storage = rt::ValueStorage::EXTERNAL_INPUT;
    gv.dtype = rt::ValueDType::BF16;
    gv.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    gv.slot = 7u;
    gv.feature_count = kGdnValueDim;
    gv.row_stride = kGdnValueDim;
    program.values.push_back(gv);

    rt::ValueBinding ga{};
    ga.logical_value = rt::ValueId{12u};
    ga.storage = rt::ValueStorage::EXTERNAL_INPUT;
    ga.dtype = rt::ValueDType::BF16;
    ga.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    ga.slot = 8u;
    ga.feature_count = kGdnNumVHeads;
    ga.row_stride = kGdnNumVHeads;
    program.values.push_back(ga);

    rt::ValueBinding gb{};
    gb.logical_value = rt::ValueId{13u};
    gb.storage = rt::ValueStorage::EXTERNAL_INPUT;
    gb.dtype = rt::ValueDType::BF16;
    gb.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    gb.slot = 9u;
    gb.feature_count = kGdnNumVHeads;
    gb.row_stride = kGdnNumVHeads;
    program.values.push_back(gb);

    rt::ValueBinding gdn_out{};
    gdn_out.logical_value = rt::ValueId{14u};
    gdn_out.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    gdn_out.dtype = rt::ValueDType::F32;
    gdn_out.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    gdn_out.slot = 3u;
    gdn_out.feature_count = kGdnValueDim;
    gdn_out.row_stride = kGdnValueDim;
    program.values.push_back(gdn_out);

    rt::ValueBinding logits{};
    logits.logical_value = rt::ValueId{15u};
    logits.storage = rt::ValueStorage::EXTERNAL_INPUT;
    logits.dtype = rt::ValueDType::F32;
    logits.row_domain = rt::ValueRowDomain::OUTPUT_ROWS;
    logits.slot = 10u;
    logits.feature_count = kN;
    logits.row_stride = kN;
    program.values.push_back(logits);

    rt::ValueBinding sampled{};
    sampled.logical_value = rt::ValueId{16u};
    sampled.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    sampled.dtype = rt::ValueDType::I32;
    sampled.row_domain = rt::ValueRowDomain::OUTPUT_ROWS;
    sampled.slot = 4u;
    sampled.feature_count = 1u;
    sampled.row_stride = 1u;
    program.values.push_back(sampled);

    rt::ValueBinding candidates{};
    candidates.logical_value = rt::ValueId{17u};
    candidates.storage = rt::ValueStorage::EXTERNAL_INPUT;
    candidates.dtype = rt::ValueDType::I32;
    candidates.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    candidates.slot = 11u;
    candidates.feature_count = kVerifyStride;
    candidates.row_stride = kVerifyStride;
    program.values.push_back(candidates);

    rt::ValueBinding candidate_counts{};
    candidate_counts.logical_value = rt::ValueId{18u};
    candidate_counts.storage = rt::ValueStorage::EXTERNAL_INPUT;
    candidate_counts.dtype = rt::ValueDType::I32;
    candidate_counts.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    candidate_counts.slot = 12u;
    candidate_counts.feature_count = kSlots;
    candidate_counts.row_stride = kSlots;
    program.values.push_back(candidate_counts);

    rt::ValueBinding committed{};
    committed.logical_value = rt::ValueId{19u};
    committed.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    committed.dtype = rt::ValueDType::I32;
    committed.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    committed.slot = 5u;
    committed.feature_count = kSlots;
    committed.row_stride = kSlots;
    program.values.push_back(committed);

    rt::StateBinding kv{};
    kv.state = rt::StateId{0u};
    kv.kind = rt::PrimitiveStateKind::KV_CACHE;
    kv.state_index = 0u;
    kv.slot = 0u;
    program.states.push_back(kv);

    rt::StateBinding gdn_conv_state{};
    gdn_conv_state.state = rt::StateId{1u};
    gdn_conv_state.kind = rt::PrimitiveStateKind::GDN_CONV_STATE;
    gdn_conv_state.state_index = 0u;
    gdn_conv_state.slot = 1u;
    program.states.push_back(gdn_conv_state);

    rt::StateBinding gdn_rec_state{};
    gdn_rec_state.state = rt::StateId{2u};
    gdn_rec_state.kind = rt::PrimitiveStateKind::GDN_RECURRENCE_STATE;
    gdn_rec_state.state_index = 0u;
    gdn_rec_state.slot = 2u;
    program.states.push_back(gdn_rec_state);

    rt::DispatchBinding rms{};
    rms.kernel_id = rt::KernelId::RMS_NORM;
    rms.input_count = 1u;
    rms.input_slots[0] = 0u;
    rms.output_count = 1u;
    rms.output_slots[0] = 1u;
    rms.parameter_index0 = 0u;
    rms.group_size = kFeatures;
    rms.scalar_a = 1e-6f;
    program.dispatches.push_back(rms);
    program.dispatch_source_node.push_back(0u);

    rt::DispatchBinding quant{};
    quant.kernel_id = rt::KernelId::ACTIVATION_QUANTIZE_W4A8;
    quant.input_count = 1u;
    quant.input_slots[0] = 1u;
    quant.output_count = 0u;
    quant.group_size = 32u;
    quant.workspace_slot = 0u;
    quant.activation_max_rows = kRows;
    quant.activation_kp = kFeatures;
    quant.activation_code_stride = layout.code_row_stride_bytes;
    quant.activation_scale_stride = layout.scale_row_stride_bytes;
    program.dispatches.push_back(quant);
    program.dispatch_source_node.push_back(1u);

    rt::DispatchBinding psq4{};
    psq4.kernel_id = rt::KernelId::LINEAR_PSQ4;
    psq4.input_count = 0u;
    psq4.output_count = 1u;
    psq4.output_slots[0] = 2u;
    psq4.weight_index = 0u;
    psq4.workspace_slot = 0u;
    psq4.compute_spec = 3u;
    psq4.activation_max_rows = kRows;
    psq4.activation_kp = kFeatures;
    psq4.activation_code_stride = layout.code_row_stride_bytes;
    psq4.activation_scale_stride = layout.scale_row_stride_bytes;
    program.dispatches.push_back(psq4);
    program.dispatch_source_node.push_back(1u);

    rt::DispatchBinding kv_append{};
    kv_append.kernel_id = rt::KernelId::KV_APPEND;
    kv_append.descriptor_slot = 1u;
    kv_append.input_count = 2u;
    kv_append.input_slots[0] = 5u;
    kv_append.input_slots[1] = 6u;
    kv_append.output_count = 0u;
    kv_append.group_size = kHeadDim;
    program.dispatches.push_back(kv_append);
    program.dispatch_source_node.push_back(2u);

    rt::DispatchBinding attn{};
    attn.kernel_id = rt::KernelId::PAGED_ATTENTION;
    attn.descriptor_slot = 1u;
    attn.input_count = 1u;
    attn.input_slots[0] = 3u;
    attn.output_count = 1u;
    attn.output_slots[0] = 4u;
    attn.group_size = kHeadDim;
    attn.scalar_a = 1.0f / 1.6e1f;
    program.dispatches.push_back(attn);
    program.dispatch_source_node.push_back(2u);

    rt::DispatchBinding gconv{};
    gconv.kernel_id = rt::KernelId::STATEFUL_CAUSAL_CONV1D;
    gconv.descriptor_slot = 2u;
    gconv.input_count = 1u;
    gconv.input_slots[0] = 7u;
    gconv.output_count = 1u;
    gconv.output_slots[0] = 8u;
    gconv.parameter_index0 = 1u;
    program.dispatches.push_back(gconv);
    program.dispatch_source_node.push_back(2u);

    rt::DispatchBinding grec{};
    grec.kernel_id = rt::KernelId::GDN_RECURRENCE;
    grec.descriptor_slot = 3u;
    grec.input_count = 5u;
    grec.input_slots[0] = 9u;
    grec.input_slots[1] = 10u;
    grec.input_slots[2] = 11u;
    grec.input_slots[3] = 12u;
    grec.input_slots[4] = 13u;
    grec.output_count = 1u;
    grec.output_slots[0] = 14u;
    grec.parameter_index0 = 2u;
    grec.parameter_index1 = 3u;
    program.dispatches.push_back(grec);
    program.dispatch_source_node.push_back(2u);

    rt::DispatchBinding sampling{};
    sampling.kernel_id = rt::KernelId::SAMPLING;
    sampling.input_count = 1u;
    sampling.input_slots[0] = 15u;
    sampling.output_count = 1u;
    sampling.output_slots[0] = 16u;
    program.dispatches.push_back(sampling);
    program.dispatch_source_node.push_back(2u);

    rt::DispatchBinding verify{};
    verify.kernel_id = rt::KernelId::VERIFY_ACCEPT;
    verify.input_count = 3u;
    verify.input_slots[0] = 17u;
    verify.input_slots[1] = 16u;
    verify.input_slots[2] = 18u;
    verify.output_count = 1u;
    verify.output_slots[0] = 19u;
    verify.rotary_dim = kVerifyStride;
    program.dispatches.push_back(verify);
    program.dispatch_source_node.push_back(2u);

    program.external_input_count = 13u;
    program.external_output_count = 6u;
}

bool wait_for(const mcu::GpuMcuPersistentMcu& persistent, uint64_t target,
              uint32_t timeout_ms) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (persistent.batches_completed() < target) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return true;
}

bool bytes_equal(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
                 size_t* first, size_t* diffs) {
    *first = 0u;
    *diffs = 0u;
    for (size_t i = 0u; i < a.size(); ++i) {
        if (a[i] == b[i]) continue;
        if (*diffs == 0u) *first = i;
        ++(*diffs);
    }
    return *diffs == 0u;
}

bool map_alloc(std::size_t bytes, void** host, void** dev) {
    void* p = nullptr;
    if (hipMallocManaged(&p, bytes) != hipSuccess) return false;
    *host = p;
    *dev = p;
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }

    rt::Program program;
    build_program(program);
    check(program.dispatches.size() == 9u, "program has nine dispatches");

    const size_t lin_bytes = static_cast<size_t>(kRows) * kN * sizeof(uint16_t);
    const size_t attn_bytes = static_cast<size_t>(kRows) * kQFeatures * sizeof(float);
    const size_t q_bytes = static_cast<size_t>(kRows) * kQFeatures * sizeof(uint16_t);
    const size_t kv_bytes =
        static_cast<size_t>(kRows) * kElemsPerToken * sizeof(uint16_t);
    const size_t input_bytes =
        static_cast<size_t>(kRows) * kFeatures * sizeof(uint16_t);
    const size_t rms_weight_bytes = static_cast<size_t>(kFeatures) * sizeof(uint16_t);
    const size_t w_codes_bytes = static_cast<size_t>(kTiles) * kKp * 8u;
    const size_t w_scales_bytes = static_cast<size_t>(kTiles) * kNb * 32u;
    const size_t ws_bytes = static_cast<size_t>(workspace_bytes());
    const size_t gconv_in_bytes = static_cast<size_t>(kRows) * kGdnConvDim * sizeof(uint16_t);
    const size_t gconv_out_bytes = static_cast<size_t>(kRows) * kGdnConvDim * sizeof(float);
    const size_t gq_bytes = static_cast<size_t>(kRows) * kGdnQkDim * sizeof(float);
    const size_t gv_bytes = static_cast<size_t>(kRows) * kGdnValueDim * sizeof(uint16_t);
    const size_t gab_bytes = static_cast<size_t>(kRows) * kGdnNumVHeads * sizeof(uint16_t);
    const size_t gdn_out_bytes = static_cast<size_t>(kRows) * kGdnValueDim * sizeof(float);
    const size_t gconv_weight_bytes =
        static_cast<size_t>(kGdnConvDim) * 4u * sizeof(uint16_t);
    const size_t gsmall_bytes = static_cast<size_t>(kGdnNumVHeads) * sizeof(uint16_t);
    const size_t gconv_state_bytes =
        static_cast<size_t>(kMaxSequences) * kGdnConvSlotStride * sizeof(uint16_t);
    const size_t grec_state_bytes =
        static_cast<size_t>(kMaxSequences) * kGdnRecurrentSlotStride * sizeof(float);
    const size_t logits_bytes = static_cast<size_t>(kOutputs) * kN * sizeof(float);
    const size_t sampled_bytes = static_cast<size_t>(kRows) * sizeof(int32_t);
    const size_t output_rows_bytes =
        static_cast<size_t>(kOutputs) * sizeof(uint32_t);
    const size_t cand_bytes = static_cast<size_t>(kVerifyElems) * sizeof(int32_t);
    const size_t cand_counts_bytes =
        static_cast<size_t>(kSlots) * sizeof(int32_t);
    const size_t committed_bytes =
        static_cast<size_t>(kSlots) * sizeof(int32_t);
    const size_t sampling_bytes =
        static_cast<size_t>(kRows) * sizeof(rt::DeviceSamplingParams);

    void* input_dev = nullptr;
    void* rms_weight_dev = nullptr;
    void* w_codes_dev = nullptr;
    void* w_scales_dev = nullptr;
    void* q_dev = nullptr;
    void* k_dev = nullptr;
    void* v_dev = nullptr;
    void* ws_host = nullptr;
    void* ws_mcu = nullptr;
    void* lin_host = nullptr;
    void* lin_mcu = nullptr;
    void* attn_host = nullptr;
    void* attn_mcu = nullptr;
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    void* row_slots = nullptr;
    void* row_pos = nullptr;
    void* block_tables = nullptr;
    void* gconv_in = nullptr;
    void* gconv_weight = nullptr;
    void* gdt_bias = nullptr;
    void* galog = nullptr;
    void* gq = nullptr;
    void* gk = nullptr;
    void* gv = nullptr;
    void* ga = nullptr;
    void* gb = nullptr;
    void* gconv_out_host = nullptr;
    void* gconv_out_mcu = nullptr;
    void* gdn_out_host = nullptr;
    void* gdn_out_mcu = nullptr;
    void* gconv_state_host = nullptr;
    void* gconv_state_mcu = nullptr;
    void* grec_state_host = nullptr;
    void* grec_state_mcu = nullptr;
    void* logits_dev = nullptr;
    void* sampled_dev = nullptr;
    void* output_rows_dev = nullptr;
    void* cand_dev = nullptr;
    void* cand_counts_dev = nullptr;
    void* committed_dev = nullptr;
    void* sampling_dev = nullptr;
    if (hipMalloc(&input_dev, input_bytes) != hipSuccess ||
        hipMalloc(&rms_weight_dev, rms_weight_bytes) != hipSuccess ||
        hipMalloc(&w_codes_dev, w_codes_bytes) != hipSuccess ||
        hipMalloc(&w_scales_dev, w_scales_bytes) != hipSuccess ||
        hipMalloc(&q_dev, q_bytes) != hipSuccess ||
        hipMalloc(&k_dev, kv_bytes) != hipSuccess ||
        hipMalloc(&v_dev, kv_bytes) != hipSuccess ||
        hipMalloc(&ws_host, ws_bytes) != hipSuccess ||
        hipMalloc(&ws_mcu, ws_bytes) != hipSuccess ||
        hipMalloc(&lin_host, lin_bytes) != hipSuccess ||
        hipMalloc(&lin_mcu, lin_bytes) != hipSuccess ||
        hipMalloc(&attn_host, attn_bytes) != hipSuccess ||
        hipMalloc(&attn_mcu, attn_bytes) != hipSuccess ||
        hipMalloc(&k_pool, kPoolBytes) != hipSuccess ||
        hipMalloc(&v_pool, kPoolBytes) != hipSuccess ||
        hipMalloc(&row_slots, kRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&row_pos, kRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&block_tables,
                  kMaxSequences * kBlockTableStride * sizeof(uint32_t)) !=
            hipSuccess ||
        hipMalloc(&gconv_in, gconv_in_bytes) != hipSuccess ||
        hipMalloc(&gconv_weight, gconv_weight_bytes) != hipSuccess ||
        hipMalloc(&gdt_bias, gsmall_bytes) != hipSuccess ||
        hipMalloc(&galog, gsmall_bytes) != hipSuccess ||
        hipMalloc(&gq, gq_bytes) != hipSuccess ||
        hipMalloc(&gk, gq_bytes) != hipSuccess ||
        hipMalloc(&gv, gv_bytes) != hipSuccess ||
        hipMalloc(&ga, gab_bytes) != hipSuccess ||
        hipMalloc(&gb, gab_bytes) != hipSuccess ||
        hipMalloc(&gconv_out_host, gconv_out_bytes) != hipSuccess ||
        hipMalloc(&gconv_out_mcu, gconv_out_bytes) != hipSuccess ||
        hipMalloc(&gdn_out_host, gdn_out_bytes) != hipSuccess ||
        hipMalloc(&gdn_out_mcu, gdn_out_bytes) != hipSuccess ||
        hipMalloc(&gconv_state_host, gconv_state_bytes) != hipSuccess ||
        hipMalloc(&gconv_state_mcu, gconv_state_bytes) != hipSuccess ||
        hipMalloc(&grec_state_host, grec_state_bytes) != hipSuccess ||
        hipMalloc(&grec_state_mcu, grec_state_bytes) != hipSuccess ||
        hipMalloc(&logits_dev, logits_bytes) != hipSuccess ||
        hipMalloc(&sampled_dev, sampled_bytes) != hipSuccess ||
        hipMalloc(&output_rows_dev, output_rows_bytes) != hipSuccess ||
        hipMalloc(&cand_dev, cand_bytes) != hipSuccess ||
        hipMalloc(&cand_counts_dev, cand_counts_bytes) != hipSuccess ||
        hipMalloc(&committed_dev, committed_bytes) != hipSuccess ||
        hipMalloc(&sampling_dev, sampling_bytes) != hipSuccess) {
        std::printf("FAIL: buffer alloc\n");
        return 1;
    }

    {
        std::vector<uint16_t> in(static_cast<size_t>(kRows) * kFeatures, 0u);
        for (uint32_t r = 0u; r < kRows; ++r)
            for (uint32_t c = 0u; c < kFeatures; ++c) {
                const uint32_t h = (r * 131u + c) * 2654435761u + 12345u;
                const int base = static_cast<int>(h % 2001u) - 1000;
                in[static_cast<size_t>(r) * kFeatures + c] =
                    chain::bf16_from_float(static_cast<float>(base) / 1000.0f).data;
            }
        std::vector<uint16_t> wt(kFeatures, 0u);
        for (uint32_t c = 0u; c < kFeatures; ++c) {
            const uint32_t h = c * 104729u + 11u;
            const int base = static_cast<int>(h % 2001u) - 1000;
            wt[c] = chain::bf16_from_float(static_cast<float>(base) / 1000.0f).data;
        }
        uint32_t s = 0x1234567u + kKp * 131u + kN;
        std::vector<uint8_t> codes(w_codes_bytes);
        for (auto& v : codes) v = chain::code_byte(s);
        std::vector<uint8_t> scales(w_scales_bytes, 0u);
        for (size_t i = 0u; i + 1u < scales.size(); i += 2u) {
            const uint16_t v = chain::scale_bf16(s);
            scales[i] = static_cast<uint8_t>(v & 0xFFu);
            scales[i + 1u] = static_cast<uint8_t>(v >> 8u);
        }
        (void)hipMemcpy(input_dev, in.data(), input_bytes, hipMemcpyHostToDevice);
        (void)hipMemcpy(rms_weight_dev, wt.data(), rms_weight_bytes,
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(w_codes_dev, codes.data(), w_codes_bytes,
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(w_scales_dev, scales.data(), w_scales_bytes,
                        hipMemcpyHostToDevice);
    }
    {
        std::vector<uint16_t> h_q(static_cast<size_t>(kRows) * kQFeatures);
        for (size_t i = 0u; i < h_q.size(); ++i)
            h_q[i] = static_cast<uint16_t>(0x3A00u + ((i * 2654435761u) % 1900u));
        std::vector<uint16_t> h_pool(kElemsPerLayer);
        for (size_t i = 0u; i < h_pool.size(); ++i)
            h_pool[i] = static_cast<uint16_t>(0x3900u + ((i * 40503u) % 1700u));
        std::vector<uint32_t> h_bt(kMaxSequences * kBlockTableStride);
        for (uint32_t slot = 0u; slot < kMaxSequences; ++slot)
            for (uint32_t b = 0u; b < kBlockTableStride; ++b)
                h_bt[slot * kBlockTableStride + b] = slot * kBlockTableStride + b;
        (void)hipMemcpy(q_dev, h_q.data(), q_bytes, hipMemcpyHostToDevice);
        std::vector<uint16_t> h_kv(static_cast<size_t>(kRows) * kElemsPerToken);
        for (size_t i = 0u; i < h_kv.size(); ++i)
            h_kv[i] = static_cast<uint16_t>(0x3800u + ((i * 2246822519u) % 1200u));
        (void)hipMemcpy(k_dev, h_kv.data(), kv_bytes, hipMemcpyHostToDevice);
        for (size_t i = 0u; i < h_kv.size(); ++i)
            h_kv[i] = static_cast<uint16_t>(0x3700u + ((i * 3266489917u) % 1300u));
        (void)hipMemcpy(v_dev, h_kv.data(), kv_bytes, hipMemcpyHostToDevice);
        (void)hipMemcpy(k_pool, h_pool.data(), kPoolBytes, hipMemcpyHostToDevice);
        for (size_t i = 0u; i < h_pool.size(); ++i)
            h_pool[i] = static_cast<uint16_t>(0x3B00u + ((i * 2654435761u) % 1500u));
        (void)hipMemcpy(v_pool, h_pool.data(), kPoolBytes, hipMemcpyHostToDevice);
        (void)hipMemcpy(block_tables, h_bt.data(), h_bt.size() * sizeof(uint32_t),
                        hipMemcpyHostToDevice);
    }

    std::vector<uint32_t> slots(kRows, 1u);
    std::vector<uint32_t> positions(kRows, 0u);
    slots[0] = 0u;
    positions[0] = 100u;
    slots[1] = 1u;
    positions[1] = 100u;
    for (uint32_t r = 0u; r < 8u; ++r) {
        slots[2u + r] = 2u;
        positions[2u + r] = 100u + r;
    }
    for (uint32_t r = 0u; r < 16u; ++r) {
        slots[10u + r] = 3u;
        positions[10u + r] = r;
    }
    (void)hipMemcpy(row_slots, slots.data(), kRows * sizeof(uint32_t),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(row_pos, positions.data(), kRows * sizeof(uint32_t),
                    hipMemcpyHostToDevice);
    (void)hipMemset(ws_host, 0xAB, ws_bytes);
    (void)hipMemset(ws_mcu, 0xAB, ws_bytes);
    (void)hipMemset(lin_host, 0xAB, lin_bytes);
    (void)hipMemset(lin_mcu, 0xAB, lin_bytes);
    (void)hipMemset(attn_host, 0xAB, attn_bytes);
    (void)hipMemset(attn_mcu, 0xAB, attn_bytes);
    {
        std::vector<uint16_t> gi(static_cast<size_t>(kRows) * kGdnConvDim);
        for (size_t i = 0u; i < gi.size(); ++i)
            gi[i] = static_cast<uint16_t>(0x3C00u + ((i * 2654435761u) % 900u));
        std::vector<uint16_t> gw(static_cast<size_t>(kGdnConvDim) * 4u);
        for (size_t i = 0u; i < gw.size(); ++i)
            gw[i] = static_cast<uint16_t>(0x3B00u + ((i * 40503u) % 800u));
        std::vector<uint16_t> gd(kGdnNumVHeads, 0x3A00u);
        std::vector<uint16_t> gl(kGdnNumVHeads, 0x3D00u);
        (void)hipMemcpy(gconv_in, gi.data(), gconv_in_bytes, hipMemcpyHostToDevice);
        (void)hipMemcpy(gconv_weight, gw.data(), gconv_weight_bytes,
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(gdt_bias, gd.data(), gsmall_bytes, hipMemcpyHostToDevice);
        (void)hipMemcpy(galog, gl.data(), gsmall_bytes, hipMemcpyHostToDevice);
        std::vector<float> gqf(static_cast<size_t>(kRows) * kGdnQkDim);
        for (size_t i = 0u; i < gqf.size(); ++i)
            gqf[i] = static_cast<float>((i % 97u) - 48u) * 0.01f;
        (void)hipMemcpy(gq, gqf.data(), gq_bytes, hipMemcpyHostToDevice);
        for (size_t i = 0u; i < gqf.size(); ++i)
            gqf[i] = static_cast<float>((i % 89u) - 44u) * 0.02f;
        (void)hipMemcpy(gk, gqf.data(), gq_bytes, hipMemcpyHostToDevice);
        std::vector<uint16_t> gvv(static_cast<size_t>(kRows) * kGdnValueDim);
        for (size_t i = 0u; i < gvv.size(); ++i)
            gvv[i] = static_cast<uint16_t>(0x3C00u + ((i * 2246822519u) % 700u));
        (void)hipMemcpy(gv, gvv.data(), gv_bytes, hipMemcpyHostToDevice);
        std::vector<uint16_t> gab(static_cast<size_t>(kRows) * kGdnNumVHeads);
        for (size_t i = 0u; i < gab.size(); ++i)
            gab[i] = static_cast<uint16_t>(0x3C00u + ((i * 3266489917u) % 400u));
        (void)hipMemcpy(ga, gab.data(), gab_bytes, hipMemcpyHostToDevice);
        for (size_t i = 0u; i < gab.size(); ++i)
            gab[i] = static_cast<uint16_t>(0x3D00u + ((i * 668265263u) % 300u));
        (void)hipMemcpy(gb, gab.data(), gab_bytes, hipMemcpyHostToDevice);
        std::vector<uint16_t> gcs(static_cast<size_t>(kMaxSequences) * kGdnConvSlotStride);
        for (size_t i = 0u; i < gcs.size(); ++i)
            gcs[i] = static_cast<uint16_t>(0x3900u + ((i * 2654435761u) % 600u));
        for (size_t i = 3u * kGdnConvSlotStride; i < 4u * kGdnConvSlotStride; ++i)
            gcs[i] = 0u;
        (void)hipMemcpy(gconv_state_host, gcs.data(), gconv_state_bytes,
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(gconv_state_mcu, gcs.data(), gconv_state_bytes,
                        hipMemcpyHostToDevice);
        std::vector<float> grs(static_cast<size_t>(kMaxSequences) *
                               kGdnRecurrentSlotStride);
        for (size_t i = 0u; i < grs.size(); ++i)
            grs[i] = static_cast<float>((i % 53u) - 26u) * 0.001f;
        for (size_t i = 3u * kGdnRecurrentSlotStride;
             i < 4u * kGdnRecurrentSlotStride; ++i)
            grs[i] = 0.0f;
        (void)hipMemcpy(grec_state_host, grs.data(), grec_state_bytes,
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(grec_state_mcu, grs.data(), grec_state_bytes,
                        hipMemcpyHostToDevice);
    }
    (void)hipMemset(gconv_out_host, 0xAB, gconv_out_bytes);
    (void)hipMemset(gconv_out_mcu, 0xAB, gconv_out_bytes);
    (void)hipMemset(gdn_out_host, 0xAB, gdn_out_bytes);
    (void)hipMemset(gdn_out_mcu, 0xAB, gdn_out_bytes);
    std::vector<int32_t> expected_sampled(kRows, -1);
    std::vector<uint32_t> output_rows_host(kOutputs, 0u);
    {
        for (uint32_t j = 0u; j < kOutputs; ++j) {
            output_rows_host[j] = j < 10u ? j : (kRows - 1u);
        }
        (void)hipMemcpy(output_rows_dev, output_rows_host.data(),
                        output_rows_bytes, hipMemcpyHostToDevice);
        std::vector<float> hl(static_cast<size_t>(kOutputs) * kN);
        for (uint32_t j = 0u; j < kOutputs; ++j) {
            for (uint32_t c = 0u; c < kN; ++c)
                hl[static_cast<size_t>(j) * kN + c] =
                    -1.0f - static_cast<float>(c) * 1e-4f;
            const uint32_t winner = (j * 7u + 3u) % kN;
            hl[static_cast<size_t>(j) * kN + winner] =
                5.0f + static_cast<float>(j);
            expected_sampled[output_rows_host[j]] =
                static_cast<int32_t>(winner);
        }
        (void)hipMemcpy(logits_dev, hl.data(), logits_bytes,
                        hipMemcpyHostToDevice);
        std::vector<rt::DeviceSamplingParams> hs(kOutputs);
        for (auto& s : hs) s.mode = 1u;
        (void)hipMemcpy(sampling_dev, hs.data(),
                        kOutputs * sizeof(rt::DeviceSamplingParams),
                        hipMemcpyHostToDevice);
        std::vector<int32_t> hcand(kVerifyElems, 0);
        for (uint32_t i = 0u; i < kVerifyStride; ++i) {
            hcand[2u * kVerifyStride + i] =
                static_cast<int32_t>(((2u + i) * 7u + 3u) % kN);
        }
        (void)hipMemcpy(cand_dev, hcand.data(), cand_bytes,
                        hipMemcpyHostToDevice);
        std::vector<int32_t> hcounts(kSlots, 0);
        hcounts[2] = static_cast<int32_t>(kVerifyStride);
        (void)hipMemcpy(cand_counts_dev, hcounts.data(), cand_counts_bytes,
                        hipMemcpyHostToDevice);
    }
    (void)hipMemset(committed_dev, 0, committed_bytes);
    (void)hipMemset(sampled_dev, 0, sampled_bytes);

    std::vector<rt::DeviceRequestDescriptor> request_seed;
    {
        rt::DeviceRequestDescriptor d0{};
        d0.execution_class = rt::ExecutionClass::DECODE;
        d0.request_handle = rt::RequestHandle{0u, 1u};
        d0.row_begin = 0u;
        d0.row_count = 1u;
        d0.prefix_length = 100u;
        d0.sequence_length = 101u;
        d0.output_index = 0u;
        d0.output_count = 1u;
        rt::DeviceRequestDescriptor d1{};
        d1.execution_class = rt::ExecutionClass::DECODE;
        d1.request_handle = rt::RequestHandle{1u, 1u};
        d1.row_begin = 1u;
        d1.row_count = 1u;
        d1.prefix_length = 100u;
        d1.sequence_length = 101u;
        d1.output_index = 1u;
        d1.output_count = 1u;
        rt::DeviceRequestDescriptor dv{};
        dv.execution_class = rt::ExecutionClass::SPEC_VERIFY;
        dv.request_handle = rt::RequestHandle{2u, 1u};
        dv.row_begin = 2u;
        dv.row_count = 8u;
        dv.prefix_length = 100u;
        dv.sequence_length = 108u;
        dv.output_index = 2u;
        dv.output_count = 8u;
        rt::DeviceRequestDescriptor dp{};
        dp.execution_class = rt::ExecutionClass::PREFILL;
        dp.request_handle = rt::RequestHandle{3u, 1u};
        dp.row_begin = 10u;
        dp.row_count = 16u;
        dp.prefix_length = 0u;
        dp.sequence_length = 16u;
        dp.output_index = 10u;
        dp.output_count = 1u;
        request_seed = {d0, d1, dv, dp};
    }
    void* requests_host = nullptr;
    void* requests_dev = nullptr;
    if (!map_alloc(request_seed.size() * sizeof(rt::DeviceRequestDescriptor),
                   &requests_host, &requests_dev)) {
        std::printf("FAIL: request map alloc\n");
        return 1;
    }
    std::memcpy(requests_host, request_seed.data(),
                request_seed.size() * sizeof(rt::DeviceRequestDescriptor));
    rt::DeviceBatchContext bc{};
    bc.actual_rows = kRows;
    bc.num_requests = static_cast<uint32_t>(request_seed.size());
    bc.num_decode_requests = 2u;
    bc.num_verify_requests = 1u;
    bc.num_prefill_requests = 1u;
    bc.requests =
        static_cast<const rt::DeviceRequestDescriptor*>(requests_dev);
    bc.row_sequence_slots = static_cast<uint32_t*>(row_slots);
    bc.row_positions = static_cast<uint32_t*>(row_pos);
    bc.num_outputs = kOutputs;
    bc.output_rows = static_cast<uint32_t*>(output_rows_dev);
    bc.output_sampling_params =
        static_cast<rt::DeviceSamplingParams*>(sampling_dev);

    void* ctx_host_mem = nullptr;
    void* ctx_dev_mem = nullptr;
    if (!map_alloc(sizeof(rt::DeviceBatchContext), &ctx_host_mem, &ctx_dev_mem)) {
        std::printf("FAIL: context map alloc\n");
        return 1;
    }
    std::memcpy(ctx_host_mem, &bc, sizeof(bc));
    rt::DeviceBatchContext* ctx_dev =
        static_cast<rt::DeviceBatchContext*>(ctx_dev_mem);

    rt::StaticParameterSlot params[4];
    params[0].device_ptr = rms_weight_dev;
    params[0].elements = kFeatures;
    params[0].dtype = rt::ValueDType::BF16;
    params[1].device_ptr = gconv_weight;
    params[1].elements = static_cast<uint64_t>(kGdnConvDim) * 4u;
    params[1].dtype = rt::ValueDType::BF16;
    params[2].device_ptr = gdt_bias;
    params[2].elements = kGdnNumVHeads;
    params[2].dtype = rt::ValueDType::BF16;
    params[3].device_ptr = galog;
    params[3].elements = kGdnNumVHeads;
    params[3].dtype = rt::ValueDType::BF16;
    rt::WeightSlot weights[1];
    std::memset(weights, 0, sizeof(weights));
    weights[0].rows = kN;
    weights[0].cols = kFeatures;
    weights[0].k_padded = kKp;
    weights[0].encoding = 3u;
    weights[0].compute_spec = 3u;
    weights[0].codes = w_codes_dev;
    weights[0].scales = w_scales_dev;
    weights[0].weight_scale_group = 32u;
    weights[0].storage_scale_stride_bytes = chain::kWeightScaleStride;
    weights[0].preshuffled = true;

    const void* ext_inputs[13] = {input_dev, q_dev, k_dev, v_dev, gconv_in, gq,
                                  gk,        gv,    ga,    gb,    logits_dev,
                                  cand_dev,  cand_counts_dev};
    void* ext_out_host[6] = {lin_host,      attn_host, gconv_out_host,
                             gdn_out_host,  sampled_dev, committed_dev};
    void* ext_out_mcu[6] = {lin_mcu,      attn_mcu, gconv_out_mcu,
                            gdn_out_mcu,  sampled_dev, committed_dev};

    kernel::ModelDispatchStateView ms{};
    ms.block_table = ::ps::qwen35::SequenceBlockTableDeviceView{
        .block_tables = static_cast<uint32_t*>(block_tables),
        .max_sequences = kMaxSequences,
        .block_table_stride = kBlockTableStride};
    ms.kv_bf16 = ::ps::qwen35::PagedKVPoolBF16View{
        .k_pool = static_cast<ps::bf16_t*>(k_pool),
        .v_pool = static_cast<ps::bf16_t*>(v_pool),
        .page_tokens = kPageTokens,
        .num_pages = kNumPages,
        .num_attention_layers = kLayers,
        .kv_heads = kKvHeads,
        .head_dim = kHeadDim,
        .elems_per_token = kElemsPerToken,
        .elems_per_page = kElemsPerPage,
        .elems_per_layer = kElemsPerLayer};
    ms.kv_dtype = static_cast<uint32_t>(::ps::qwen35::KVCacheDType::BF16);
    ms.gdn.conv_base = static_cast<ps::bf16_t*>(gconv_state_host);
    ms.gdn.recurrent_base = static_cast<float*>(grec_state_host);
    ms.gdn.num_gdn_states = 1u;
    ms.gdn.max_sequences = kMaxSequences;
    ms.gdn.conv_slot_stride = kGdnConvSlotStride;
    ms.gdn.conv_layer_stride = kGdnConvLayerStride;
    ms.gdn.conv_history_stride = kGdnConvHistoryStride;
    ms.gdn.recurrent_slot_stride = kGdnRecurrentSlotStride;
    ms.gdn.recurrent_layer_stride = kGdnRecurrentLayerStride;
    ms.gdn.conv_dim = kGdnConvDim;
    ms.gdn.conv_dim_padded = kGdnConvDim;
    ms.gdn.conv_history = kGdnConvHistory;
    ms.gdn.num_v_heads = kGdnNumVHeads;
    ms.gdn.head_k = kGdnHeadK;
    ms.gdn.head_v = kGdnHeadV;
    kernel::ModelDispatchStateView ms_mcu = ms;
    ms_mcu.gdn.conv_base = static_cast<ps::bf16_t*>(gconv_state_mcu);
    ms_mcu.gdn.recurrent_base = static_cast<float*>(grec_state_mcu);

    q35::HostExecutionContext ctx_host{};
    ctx_host.batch_context = ctx_dev;
    ctx_host.device_requests = bc.requests;
    ctx_host.device_output_rows = bc.output_rows;
    ctx_host.output_sampling_params = bc.output_sampling_params;
    ctx_host.staged_requests =
        static_cast<const rt::DeviceRequestDescriptor*>(requests_host);
    ctx_host.staged_request_count = bc.num_requests;
    ctx_host.staged_verify_request_count = bc.num_verify_requests;
    ctx_host.workspace = static_cast<uint8_t*>(ws_host);
    ctx_host.workspace_bytes = ws_bytes;
    ctx_host.external_inputs = ext_inputs;
    ctx_host.external_input_count = 13u;
    ctx_host.external_outputs = ext_out_host;
    ctx_host.external_output_count = 6u;
    ctx_host.host_parameters = params;
    ctx_host.host_parameter_count = 4u;
    ctx_host.weights = weights;
    ctx_host.weight_count = 1u;
    ctx_host.actual_rows = kRows;
    ctx_host.actual_outputs = 6u;
    ctx_host.row_positions = static_cast<const uint32_t*>(row_pos);
    ctx_host.row_sequence_slots = static_cast<const uint32_t*>(row_slots);
    ctx_host.max_visible_tokens = kMaxVisible;
    ctx_host.decode_attn_partials = nullptr;
    ctx_host.decode_attn_partial_bytes = 0u;
    ctx_host.model_state = ms;

    q35::HostExecutionContext ctx_mcu = ctx_host;
    ctx_mcu.workspace = static_cast<uint8_t*>(ws_mcu);
    ctx_mcu.external_outputs = ext_out_mcu;
    ctx_mcu.model_state = ms_mcu;

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL: stream\n");
        return 1;
    }
    const auto r1 =
        q35::try_launch_rmsnorm(program, program.dispatches[0], ctx_host, stream);
    const auto r2 = q35::try_launch_activation_quantize_a8(
        program, program.dispatches[1], ctx_host, stream);
    const auto r3 =
        q35::try_launch_linear_psq4(program, program.dispatches[2], ctx_host, stream);
    const auto r3b =
        q35::try_launch_kv_append(program, 3u, ctx_host, stream);
    q35::PhysicalPagedAttentionPlan host_attn{};
    const auto r4 =
        q35::resolve_paged_attention_physical(program, 4u, ctx_host, host_attn);
    const auto r5 = q35::try_launch_gdn_conv1d(program, 5u, ctx_host, stream);
    const auto r6 = q35::try_launch_gdn_recurrence(program, 6u, ctx_host, stream);
    if (!(r1.ok() && r1.value().kind == q35::OptimizedLaunchKind::Launched) ||
        !(r2.ok() && r2.value().kind == q35::OptimizedLaunchKind::Launched) ||
        !(r3.ok() && r3.value().kind == q35::OptimizedLaunchKind::Launched) ||
        !(r3b.ok() && r3b.value().kind == q35::OptimizedLaunchKind::Launched) ||
        !(r4.ok() && r4.value() == q35::PhysicalResolveStatus::Applicable) ||
        !(r5.ok() && r5.value().kind == q35::OptimizedLaunchKind::Launched) ||
        !(r6.ok() && r6.value().kind == q35::OptimizedLaunchKind::Launched) ||
        q35::launch_paged_attention_physical(host_attn, stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
        std::printf(
            "FAIL: host reference chain r1=%d r2=%d r3=%d r3b=%d r4=%d r5=%d "
            "r6=%d\n",
            static_cast<int>(r1.ok()), static_cast<int>(r2.ok()),
            static_cast<int>(r3.ok()), static_cast<int>(r3b.ok()),
            static_cast<int>(r4.ok()), static_cast<int>(r5.ok()),
            static_cast<int>(r6.ok()));
        if (!r3b.ok()) std::printf("  kv: %s\n", r3b.status().message().c_str());
        if (!r4.ok()) std::printf("  attn: %s\n", r4.status().message().c_str());
        if (!r5.ok()) std::printf("  gconv: %s\n", r5.status().message().c_str());
        if (!r6.ok()) std::printf("  grec: %s\n", r6.status().message().c_str());
        return 1;
    }
    check(host_attn.physical_count == 2u, "host attention decomposes to two regions");

    q35::McuPlanCompileOptions opts{};
    opts.dispatch_begin = 0u;
    opts.dispatch_end = 9u;
    opts.marker_grid = 1u;
    opts.marker_workgroup = 64u;
    q35::McuCompiledPlan plan;
    const q35::McuStaticPlanCompileContext ctx_mcu_static =
        q35::make_static_plan_compile_context(
            ctx_mcu, ctx_mcu.actual_rows, ctx_mcu.actual_outputs,
            ctx_mcu.staged_request_count,
            ctx_mcu.actual_stochastic_outputs);
    const ps::Status plan_status = q35::compile_mcu_plan(program,ctx_mcu_static, opts, plan);
    if (!plan_status.ok()) {
        std::printf("FAIL: plan: %s\n", plan_status.message().c_str());
        return 1;
    }
    check(plan.psq4_multi.size() == 1u, "row-block linear emitted");
    check(plan.kv_append.size() == 1u, "kv append emitted");
    check(plan.attention_paged.size() == 2u, "two attention regions emitted");
    check(plan.gdn_conv1d.size() == 1u, "gdn conv1d emitted");
    check(plan.gdn_recurrence.size() == 1u, "gdn recurrence emitted");
    check(plan.gdn_reset.size() == 1u, "gdn reset emitted");
    check(plan.argmax_f32.size() == 1u, "sampling emitted");
    check(plan.verify_accept.size() == 1u, "verify accept emitted");

    q35::McuDecodeRuntimeOptions options{};
    options.device = 0;
    options.record_entries = 0u;
    options.driver_mode = q35::McuPlanDriverMode::ExternalPersistentMcu;
    auto runtime_result = q35::McuDecodeRuntime::create(options);
    if (!runtime_result.ok()) {
        std::printf("FAIL: runtime create: %s\n",
                    runtime_result.status().message().c_str());
        return 1;
    }
    q35::McuDecodeRuntime runtime = runtime_result.release();
    const ps::Status prepared = runtime.prepare_plan(plan);
    if (!prepared.ok()) {
        std::printf("FAIL: prepare_plan: %s\n", prepared.message().c_str());
        return 1;
    }

    uint64_t* ready_dev = nullptr;
    uint64_t* plan_epoch_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&ready_dev), sizeof(uint64_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&plan_epoch_dev), sizeof(uint64_t)) !=
            hipSuccess) {
        std::printf("FAIL: context alloc\n");
        return 1;
    }
    (void)hipMemset(ready_dev, 0, sizeof(uint64_t));
    (void)hipMemset(plan_epoch_dev, 0, sizeof(uint64_t));

    hipStream_t persistent_stream = nullptr;
    if (hipStreamCreate(&persistent_stream) != hipSuccess) {
        std::printf("FAIL: persistent stream\n");
        return 1;
    }
    auto persistent_result = mcu::GpuMcuPersistentMcu::create(0, persistent_stream);
    if (!persistent_result.ok()) {
        std::printf("FAIL: persistent create\n");
        return 1;
    }
    mcu::GpuMcuPersistentMcu persistent = persistent_result.release();

    check(persistent
              .configure_batch_plan(ctx_dev,
                                    static_cast<rt::DeviceRequestDescriptor*>(
                                        requests_dev),
                                    static_cast<uint32_t*>(row_slots),
                                    static_cast<uint32_t*>(row_pos),
                                    static_cast<uint32_t>(request_seed.size()),
                                    kRows, 16u, plan_epoch_dev)
              .ok(),
          "configure batch plan");
    check(persistent
              .configure_execution(runtime.execution_state(),
                                   runtime.plan_variant_count(), ready_dev, nullptr,
                                   0u, nullptr, 0u)
              .ok(),
          "configure execution");
    check(persistent
              .configure_execution_patches(runtime.row_global_patches(),
                                           runtime.row_global_patch_count())
              .ok(),
          "configure execution patches");
    const uint64_t ready = 1u;
    (void)hipMemcpy(ready_dev, &ready, sizeof(ready), hipMemcpyHostToDevice);
    check(persistent.start().ok(), "persistent start");
    check(wait_for(persistent, 1u, 15000u), "mixed linear+attention loop completed");
    check(persistent.batches_failed() == 0u, "no fault");

    check(persistent.request_stop().ok(), "persistent stop requested");
    check(persistent.wait_stopped(5000u).ok(), "persistent stopped");
    check(persistent.shutdown().ok(), "persistent shutdown");

    std::vector<uint8_t> lin_h(lin_bytes, 0u);
    std::vector<uint8_t> lin_m(lin_bytes, 0u);
    std::vector<uint8_t> attn_h(attn_bytes, 0u);
    std::vector<uint8_t> attn_m(attn_bytes, 0u);
    (void)hipMemcpy(lin_h.data(), lin_host, lin_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(lin_m.data(), lin_mcu, lin_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(attn_h.data(), attn_host, attn_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(attn_m.data(), attn_mcu, attn_bytes, hipMemcpyDeviceToHost);
    size_t first = 0u;
    size_t diffs = 0u;
    check(bytes_equal(lin_h, lin_m, &first, &diffs),
          "linear output host vs persistent mcu byte exact");
    if (diffs != 0u) std::printf("  linear diff count=%zu first=%zu\n", diffs, first);
    check(bytes_equal(attn_h, attn_m, &first, &diffs),
          "attention output host vs persistent mcu byte exact");
    if (diffs != 0u) std::printf("  attention diff count=%zu first=%zu\n", diffs, first);
    std::vector<uint8_t> gch(gconv_out_bytes, 0u);
    std::vector<uint8_t> gcm(gconv_out_bytes, 0u);
    std::vector<uint8_t> ggh(gdn_out_bytes, 0u);
    std::vector<uint8_t> ggm(gdn_out_bytes, 0u);
    (void)hipMemcpy(gch.data(), gconv_out_host, gconv_out_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(gcm.data(), gconv_out_mcu, gconv_out_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(ggh.data(), gdn_out_host, gdn_out_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(ggm.data(), gdn_out_mcu, gdn_out_bytes, hipMemcpyDeviceToHost);
    check(bytes_equal(gch, gcm, &first, &diffs),
          "gdn conv1d output host vs persistent mcu byte exact");
    if (diffs != 0u) std::printf("  gconv diff count=%zu first=%zu\n", diffs, first);
    check(bytes_equal(ggh, ggm, &first, &diffs),
          "gdn recurrence output host vs persistent mcu byte exact");
    if (diffs != 0u) std::printf("  grec diff count=%zu first=%zu\n", diffs, first);
    std::vector<int32_t> sampled(kRows, -1);
    (void)hipMemcpy(sampled.data(), sampled_dev, sampled_bytes,
                    hipMemcpyDeviceToHost);
    for (uint32_t j = 0u; j < kOutputs; ++j) {
        const uint32_t r = output_rows_host[j];
        if (sampled[r] != expected_sampled[r]) {
            std::printf("FAIL: sampled[%u]=%d expected=%d\n", r, sampled[r],
                        expected_sampled[r]);
            ++g_failed;
        }
    }
    check(runtime.execution_state()->fault_code ==
              static_cast<uint32_t>(mcu::McuFaultCode::none),
          "prepared fsm has no fault");
    std::printf(
        "mixed full layer: regions=%u dispatches=%llu plans=%llu\n",
        plan.physical_dispatch_count,
        static_cast<unsigned long long>(
            runtime.execution_state()->dispatches_committed),
        static_cast<unsigned long long>(runtime.execution_state()->plans_started));

    // Phase 2: drive the prepared plan through the execution step with a
    // request runtime and commit the canonical mixed batch.
    {
        auto ring_result = mcu::GpuMcuControlRing::create(0);
        check(ring_result.ok(), "phase2 control ring");
        if (!ring_result.ok()) return 1;
        mcu::GpuMcuControlRing ring = ring_result.release();
        auto arena_result = mcu::GpuMcuRequestIngressArena::create(0, 8u);
        check(arena_result.ok(), "phase2 ingress arena");
        if (!arena_result.ok()) return 1;
        mcu::GpuMcuRequestIngressArena arena = arena_result.release();

        void* slots_mem = nullptr;
        void* slots_dev = nullptr;
        check(map_alloc(kSlots * sizeof(mcu::GpuMcuSlotState), &slots_mem,
                        &slots_dev),
              "phase2 slots alloc");
        void* runnable_mem = nullptr;
        void* runnable_dev = nullptr;
        check(map_alloc(kSlots * sizeof(rt::RequestHandle), &runnable_mem,
                        &runnable_dev),
              "phase2 runnable alloc");
        void* bind_mem = nullptr;
        void* bind_dev = nullptr;
        check(map_alloc(kSlots * sizeof(mcu::GpuMcuSlotBinding), &bind_mem,
                        &bind_dev),
              "phase2 bindings alloc");
        void* tokens_mem = nullptr;
        void* tokens_dev = nullptr;
        check(map_alloc(kRows * sizeof(int32_t), &tokens_mem, &tokens_dev),
              "phase2 tokens alloc");
        void* ready2_mem = nullptr;
        void* ready2_dev = nullptr;
        check(map_alloc(sizeof(uint64_t), &ready2_mem, &ready2_dev),
              "phase2 ready alloc");
        void* srt_mem = nullptr;
        void* srt_dev = nullptr;
        check(map_alloc(kSlots * sizeof(mcu::GpuMcuSlotRuntimeState), &srt_mem,
                        &srt_dev),
              "phase2 slot runtime alloc");
        void* prompt_mem = nullptr;
        void* prompt_dev = nullptr;
        check(map_alloc(16u * sizeof(int32_t), &prompt_mem, &prompt_dev),
              "phase2 prompt alloc");
        void* samp2_mem = nullptr;
        void* samp2_dev = nullptr;
        check(map_alloc(sizeof(rt::DeviceSamplingParams), &samp2_mem,
                        &samp2_dev),
              "phase2 sampling alloc");
        void* cand_mem = nullptr;
        void* cand_dev = nullptr;
        check(map_alloc(8u * sizeof(int32_t), &cand_mem, &cand_dev),
              "phase2 candidates alloc");

        std::memset(slots_mem, 0, kSlots * sizeof(mcu::GpuMcuSlotState));
        std::memset(bind_mem, 0, kSlots * sizeof(mcu::GpuMcuSlotBinding));
        std::memset(srt_mem, 0, kSlots * sizeof(mcu::GpuMcuSlotRuntimeState));
        auto* slots = static_cast<mcu::GpuMcuSlotState*>(slots_mem);
        auto* p2bind = static_cast<mcu::GpuMcuSlotBinding*>(bind_mem);
        static_cast<uint64_t*>(ready2_mem)[0] = 0u;

        for (uint32_t i = 0u; i < kSlots; ++i) {
            slots[i].handle.slot = i;
            slots[i].handle.generation = 1u;
            slots[i].request_id = 9000u + i;
            slots[i].max_sequence_length = 256u;
            slots[i].max_new_tokens = 2u;
            p2bind[i].handle = slots[i].handle;
            p2bind[i].sampling_params_handle =
                mcu::gpu_mcu_encode_device_address(samp2_dev);
            p2bind[i].decode_input_token = 4000 + static_cast<int32_t>(i);
            p2bind[i].decode_input_valid = 1u;
        }
        for (uint32_t i = 0u; i < 2u; ++i) {
            slots[i].phase =
                mcu::gpu_mcu_slot_phase_value(mcu::GpuMcuSlotPhase::decode);
            slots[i].committed_position = 100u;
            slots[i].sequence_length = 100u;
        }
        slots[2].phase =
            mcu::gpu_mcu_slot_phase_value(mcu::GpuMcuSlotPhase::verify);
        slots[2].committed_position = 100u;
        slots[2].sequence_length = 100u;
        slots[2].verify_candidate_count = 8u;
        slots[2].verify_candidate_ref =
            mcu::gpu_mcu_encode_device_address(cand_dev);
        slots[2].max_new_tokens = 9u;
        slots[3].phase =
            mcu::gpu_mcu_slot_phase_value(mcu::GpuMcuSlotPhase::prefill);
        slots[3].prompt_length = 16u;
        slots[3].prefill_position = 0u;
        slots[3].sequence_length = 0u;
        p2bind[3].prompt_tokens_handle =
            mcu::gpu_mcu_encode_device_address(prompt_dev);

        for (uint32_t i = 0u; i < 16u; ++i)
            static_cast<int32_t*>(prompt_mem)[i] = 700 + static_cast<int32_t>(i);
        static_cast<rt::DeviceSamplingParams*>(samp2_mem)[0].mode = 1u;
        for (uint32_t i = 0u; i < 8u; ++i)
            static_cast<int32_t*>(cand_mem)[i] = 500 + static_cast<int32_t>(i);

        auto persistent2_result =
            mcu::GpuMcuPersistentMcu::create(0, persistent_stream);
        check(persistent2_result.ok(), "phase2 persistent create");
        if (!persistent2_result.ok()) return 1;
        mcu::GpuMcuPersistentMcu persistent2 = persistent2_result.release();

        auto runtime2_result = q35::McuDecodeRuntime::create(options);
        check(runtime2_result.ok(), "phase2 runtime create");
        if (!runtime2_result.ok()) return 1;
        q35::McuDecodeRuntime runtime2 = runtime2_result.release();
        const ps::Status prepared2 = runtime2.prepare_plan(plan);
        check(prepared2.ok(), "phase2 prepare plan");
        if (!prepared2.ok()) return 1;

        check(persistent2
                  .configure_request_runtime(
                      ring.device_input(), ring.device_events(),
                      arena.device_entries(), 8u,
                      static_cast<mcu::GpuMcuSlotState*>(slots_dev), kSlots,
                      static_cast<rt::RequestHandle*>(runnable_dev), kSlots)
                  .ok(),
              "phase2 configure request runtime");
        check(persistent2
                  .configure_batch_plan(
                      ctx_dev,
                      static_cast<rt::DeviceRequestDescriptor*>(requests_dev),
                      static_cast<uint32_t*>(row_slots),
                      static_cast<uint32_t*>(row_pos),
                      static_cast<uint32_t>(request_seed.size()), kRows, 16u,
                      plan_epoch_dev)
                  .ok(),
              "phase2 configure batch plan");
        check(persistent2
                  .configure_batch_binding(
                      static_cast<mcu::GpuMcuSlotBinding*>(bind_dev), kSlots,
                      static_cast<int32_t*>(tokens_dev),
                      static_cast<uint32_t*>(output_rows_dev),
                      static_cast<rt::DeviceSamplingParams*>(sampling_dev),
                      kRows, static_cast<uint64_t*>(ready2_dev))
                  .ok(),
              "phase2 configure batch binding");
        check(persistent2
                  .configure_execution(
                      runtime2.execution_state(), runtime2.plan_variant_count(),
                      static_cast<uint64_t*>(ready2_dev),
                      static_cast<const int32_t*>(sampled_dev),
                      kRows, static_cast<const uint32_t*>(committed_dev), kSlots)
                  .ok(),
              "phase2 configure execution");
        check(persistent2
                  .configure_execution_patches(runtime2.row_global_patches(),
                                               runtime2.row_global_patch_count())
                  .ok(),
              "phase2 configure execution patches");
        check(persistent2
                  .configure_slot_runtime(
                      static_cast<mcu::GpuMcuSlotRuntimeState*>(srt_dev))
                  .ok(),
              "phase2 configure slot runtime");

        check(persistent2.start().ok(), "phase2 persistent start");
        check(wait_for(persistent2, 2u, 15000u),
              "phase2 mixed execution completes twice");
        check(persistent2.request_stop().ok(), "phase2 stop requested");
        check(persistent2.wait_stopped(5000u).ok(), "phase2 stopped");

        check(persistent2.batches_failed() == 0u, "phase2 no fault");
        std::vector<int32_t> committed_host(kSlots, -1);
        (void)hipMemcpy(committed_host.data(), committed_dev, committed_bytes,
                        hipMemcpyDeviceToHost);
        std::printf(
            "phase2: committed=%llu tokens=%llu decode0=%u decode1=%u "
            "verify=%u accepted=%u prefill=%u\n",
            (unsigned long long)persistent2.batches_committed(),
            (unsigned long long)persistent2.committed_tokens(),
            slots[0].sequence_length, slots[1].sequence_length,
            slots[2].sequence_length, slots[2].verify_committed_count,
            slots[3].prefill_position);
        check(slots[2].verify_committed_count == 8u,
              "phase2 verify records the accepted count");
        check(committed_host[2] == 0,
              "phase2 verify accept sees no verify request in the second loop");
        check(persistent2.batches_committed() == 2u,
              "phase2 commits the mixed batch twice");
        check(slots[0].sequence_length == 102u, "phase2 decode0 advances twice");
        check(slots[1].sequence_length == 102u, "phase2 decode1 advances twice");
        check(slots[2].sequence_length == 109u,
              "phase2 verify then decode advances");
        check(slots[3].sequence_length == 17u,
              "phase2 prefill then decode advances");
        check(slots[0].generated_tokens == 2u && slots[1].generated_tokens == 2u &&
                  slots[3].generated_tokens == 2u,
              "phase2 decode and prefill emit two tokens");
        check(slots[2].generated_tokens == 9u,
              "phase2 verify emits the accepted prefix twice");

        check(persistent2.shutdown().ok(), "phase2 shutdown");

        check(runtime2.shutdown().ok(), "phase2 runtime shutdown");
        check(ring.shutdown().ok(), "phase2 control ring shutdown");
        check(arena.shutdown().ok(), "phase2 ingress arena shutdown");
        check(hipFree(slots_mem) == hipSuccess, "phase2 slots free");
        check(hipFree(runnable_mem) == hipSuccess, "phase2 runnable free");
        check(hipFree(bind_mem) == hipSuccess, "phase2 bindings free");
        check(hipFree(tokens_mem) == hipSuccess, "phase2 tokens free");
        check(hipFree(ready2_mem) == hipSuccess, "phase2 ready free");
        check(hipFree(srt_mem) == hipSuccess, "phase2 slot runtime free");
        check(hipFree(prompt_mem) == hipSuccess, "phase2 prompt free");
        check(hipFree(samp2_mem) == hipSuccess, "phase2 sampling free");
        check(hipFree(cand_mem) == hipSuccess, "phase2 candidates free");
    }

    check(runtime.shutdown().ok(), "runtime shutdown");

    (void)hipFree(plan_epoch_dev);
    (void)hipFree(ready_dev);
    check(hipFree(ctx_host_mem) == hipSuccess, "context map free");
    check(hipFree(requests_host) == hipSuccess, "request map free");
    check(hipStreamDestroy(persistent_stream) == hipSuccess,
          "persistent stream destroy");
    (void)hipStreamDestroy(stream);
    (void)hipFree(grec_state_mcu);
    (void)hipFree(grec_state_host);
    (void)hipFree(gconv_state_mcu);
    (void)hipFree(gconv_state_host);
    (void)hipFree(gdn_out_mcu);
    (void)hipFree(gdn_out_host);
    (void)hipFree(gconv_out_mcu);
    (void)hipFree(gconv_out_host);
    (void)hipFree(gb);
    (void)hipFree(ga);
    (void)hipFree(gv);
    (void)hipFree(gk);
    (void)hipFree(gq);
    (void)hipFree(galog);
    (void)hipFree(gdt_bias);
    (void)hipFree(gconv_weight);
    (void)hipFree(gconv_in);
    (void)hipFree(sampling_dev);
    (void)hipFree(sampled_dev);
    (void)hipFree(output_rows_dev);
    (void)hipFree(cand_dev);
    (void)hipFree(cand_counts_dev);
    (void)hipFree(committed_dev);
    (void)hipFree(logits_dev);
    (void)hipFree(block_tables);
    (void)hipFree(row_pos);
    (void)hipFree(row_slots);
    (void)hipFree(v_pool);
    (void)hipFree(k_pool);
    (void)hipFree(attn_mcu);
    (void)hipFree(attn_host);
    (void)hipFree(lin_mcu);
    (void)hipFree(lin_host);
    (void)hipFree(ws_mcu);
    (void)hipFree(ws_host);
    (void)hipFree(v_dev);
    (void)hipFree(k_dev);
    (void)hipFree(q_dev);
    (void)hipFree(w_scales_dev);
    (void)hipFree(w_codes_dev);
    (void)hipFree(rms_weight_dev);
    (void)hipFree(input_dev);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_mixed_linear_attention_bridge: %d failures\n",
                    g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_mixed_linear_attention_bridge: PASS\n");
    return 0;
}
