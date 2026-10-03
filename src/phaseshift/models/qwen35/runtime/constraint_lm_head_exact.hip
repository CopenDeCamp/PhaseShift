#include <phaseshift/models/qwen35/runtime/constraint_lm_head_exact.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/models/qwen35/runtime/optimized_dispatch.h>
#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/models/qwen35/kernels/optimized/constraint_candidates.h>
#include <phaseshift/models/qwen35/kernels/optimized/psq8_candidate_rerank.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ps::qwen35::runtime {

namespace {

__global__ void pool_argmax_remap_kernel(
    const float* __restrict__ rerank_logits,
    const int32_t* __restrict__ pool_ids,
    int32_t* __restrict__ out_ids,
    const uint32_t pool) {
    const uint32_t row = blockIdx.x;
    const float* logits = rerank_logits + static_cast<std::size_t>(row) * pool;
    const int32_t* ids = pool_ids + static_cast<std::size_t>(row) * pool;
    float best_value = -INFINITY;
    int32_t best_pos = 0;
    for (uint32_t c = 0; c < pool; ++c) {
        const float v = logits[c];
        if (v > best_value || (v == best_value && ids[c] < ids[best_pos])) {
            best_value = v;
            best_pos = static_cast<int32_t>(c);
        }
    }
    out_ids[row] = ids[best_pos];
}

}  // namespace

ConstraintLmHeadExact::~ConstraintLmHeadExact() noexcept {
    (void)shutdown();
}

void ConstraintLmHeadExact::move_from(ConstraintLmHeadExact& other) noexcept {
    weight_codes_ = other.weight_codes_;
    weight_scales_ = other.weight_scales_;
    vocab_ = other.vocab_;
    cols_ = other.cols_;
    k_padded_ = other.k_padded_;
    scale_stride_ = other.scale_stride_;
    max_rows_ = other.max_rows_;
    candidate_capacity_ = other.candidate_capacity_;
    act_code_stride_ = other.act_code_stride_;
    act_scale_stride_ = other.act_scale_stride_;
    act_codes_ = other.act_codes_;
    act_scales_ = other.act_scales_;
    cand_ids_ = other.cand_ids_;
    rerank_ = other.rerank_;

    other.weight_codes_ = nullptr;
    other.weight_scales_ = nullptr;
    other.vocab_ = 0u;
    other.cols_ = 0u;
    other.k_padded_ = 0u;
    other.scale_stride_ = 0u;
    other.max_rows_ = 0u;
    other.candidate_capacity_ = 0u;
    other.act_code_stride_ = 0u;
    other.act_scale_stride_ = 0u;
    other.act_codes_ = nullptr;
    other.act_scales_ = nullptr;
    other.cand_ids_ = nullptr;
    other.rerank_ = nullptr;
}

ConstraintLmHeadExact::ConstraintLmHeadExact(ConstraintLmHeadExact&& other) noexcept {
    move_from(other);
}

ConstraintLmHeadExact& ConstraintLmHeadExact::operator=(ConstraintLmHeadExact&& other) noexcept {
    if (this != &other) {
        (void)shutdown();
        move_from(other);
    }
    return *this;
}

Status ConstraintLmHeadExact::shutdown() noexcept {
    auto free_one = [](void* p) {
        if (p != nullptr)
            (void)hipFree(p);
    };
    free_one(act_codes_);
    free_one(act_scales_);
    free_one(cand_ids_);
    free_one(rerank_);
    weight_codes_ = nullptr;
    weight_scales_ = nullptr;
    vocab_ = 0u;
    cols_ = 0u;
    k_padded_ = 0u;
    scale_stride_ = 0u;
    max_rows_ = 0u;
    candidate_capacity_ = 0u;
    act_code_stride_ = 0u;
    act_scale_stride_ = 0u;
    act_codes_ = nullptr;
    act_scales_ = nullptr;
    cand_ids_ = nullptr;
    rerank_ = nullptr;
    return Status::make_ok();
}

Status ConstraintLmHeadExact::init(const ConstraintLmHeadWeightView& weight, uint32_t max_rows,
                                   hipStream_t stream) {
    (void)stream;
    if (weight.codes == nullptr || weight.scales == nullptr || weight.vocab == 0u ||
        weight.cols == 0u || weight.k_padded == 0u || weight.scale_stride_bytes == 0u) {
        return Status::invalid_argument("constraint lm_head weight view invalid", __FILE__,
                                        __LINE__);
    }
    if (max_rows == 0u || max_rows > 64u) {
        return Status::invalid_argument("constraint lm_head max rows out of range", __FILE__,
                                        __LINE__);
    }
    if ((weight.k_padded % 32u) != 0u) {
        return Status::invalid_argument("constraint lm_head k_padded not 32-aligned", __FILE__,
                                        __LINE__);
    }
    (void)shutdown();

    weight_codes_ = weight.codes;
    weight_scales_ = weight.scales;
    vocab_ = weight.vocab;
    cols_ = weight.cols;
    k_padded_ = weight.k_padded;
    scale_stride_ = weight.scale_stride_bytes;
    max_rows_ = max_rows;
    candidate_capacity_ = kConstraintLmHeadExactMaxAllowed;

    const auto layout = ::ps::runtime::Int8ActivationWorkspaceLayout::make(cols_, max_rows_);
    if (layout.k_padded != k_padded_) {
        (void)shutdown();
        return Status::invalid_argument("constraint lm_head activation layout mismatch", __FILE__,
                                        __LINE__);
    }
    act_code_stride_ = layout.code_row_stride_bytes;
    act_scale_stride_ = layout.scale_row_stride_bytes;

    bool ok = true;
    auto alloc = [&ok](void** p, std::size_t bytes) {
        ok = ok && (hipMalloc(p, bytes) == hipSuccess);
    };
    alloc(reinterpret_cast<void**>(&act_codes_),
          static_cast<std::size_t>(act_code_stride_) * ((max_rows_ + 15u) / 16u) * 16u);
    alloc(reinterpret_cast<void**>(&act_scales_),
          static_cast<std::size_t>(act_scale_stride_) * max_rows_);
    alloc(reinterpret_cast<void**>(&cand_ids_),
          static_cast<std::size_t>(max_rows_) * candidate_capacity_ * sizeof(int32_t));
    alloc(reinterpret_cast<void**>(&rerank_),
          static_cast<std::size_t>(max_rows_) * candidate_capacity_ * sizeof(float));
    if (!ok) {
        (void)shutdown();
        return Status::hip_error("constraint lm_head allocation failed", "hipMalloc", __FILE__,
                                 __LINE__);
    }
    return Status::make_ok();
}

Status ConstraintLmHeadExact::select_exact(const bf16_t* normed, uint32_t rows,
                                           uint32_t row_stride, const uint32_t* masks,
                                           uint32_t mask_words, uint32_t candidate_capacity,
                                           int32_t* out_ids, hipStream_t stream) {
    if (!valid()) {
        return Status::invalid_state("constraint lm_head not initialized", __FILE__, __LINE__);
    }
    if (normed == nullptr || out_ids == nullptr || masks == nullptr || rows == 0u ||
        rows > max_rows_) {
        return Status::invalid_argument("constraint lm_head select arguments invalid", __FILE__,
                                        __LINE__);
    }
    if (mask_words == 0u || candidate_capacity == 0u ||
        candidate_capacity > candidate_capacity_ ||
        mask_words < (vocab_ + 31u) / 32u) {
        return Status::invalid_argument("constraint lm_head capacity invalid", __FILE__,
                                        __LINE__);
    }
    hipError_t err = ps::kernel::launch_activation_quantize_e4m3(
        normed, act_codes_, act_scales_, rows, cols_, k_padded_, row_stride, act_code_stride_,
        act_scale_stride_, stream);
    if (err != hipSuccess) {
        return Status::hip_error("constraint lm_head activation quantize failed",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    err = ps::kernel::launch_constraint_mask_to_candidates(
        masks, rows, mask_words, vocab_, candidate_capacity, cand_ids_, stream);
    if (err != hipSuccess) {
        return Status::hip_error("constraint lm_head candidate extraction failed",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    err = ps::kernel::launch_dflash2_psq8_candidate_rerank(
        weight_codes_, weight_scales_, act_codes_, act_scales_, cand_ids_, rerank_, rows,
        candidate_capacity, k_padded_, scale_stride_, act_code_stride_, act_scale_stride_,
        candidate_capacity, stream);
    if (err != hipSuccess) {
        return Status::hip_error("constraint lm_head rerank failed", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    pool_argmax_remap_kernel<<<rows, 1, 0, stream>>>(rerank_, cand_ids_, out_ids,
                                                     candidate_capacity);
    err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("constraint lm_head argmax failed", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

bool constraint_lm_head_exact_enabled() {
    const char* env = std::getenv("PHASESHIFT_CONSTRAINT_LM_HEAD_EXACT");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

Result<OptimizedLaunchResult> try_launch_constraint_lm_head_exact(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    OptimizedLaunchResult not_applicable{};
    if (ctx.constraint_lm_head == nullptr || !ctx.constraint_lm_head->valid())
        return not_applicable;
    if (!constraint_lm_head_exact_enabled())
        return not_applicable;
    const bool constrained =
        ctx.constraint_masks != nullptr || ctx.constraint_mask_words != 0u;
    if (!constrained)
        return not_applicable;
    if (ctx.role != ::ps::runtime::ExecutionRole::Decode)
        return not_applicable;
    if (dispatch_index + 2u >= program.dispatches.size())
        return not_applicable;
    const auto& actq = program.dispatches[dispatch_index];
    const auto& linear = program.dispatches[dispatch_index + 1u];
    const auto& sampling = program.dispatches[dispatch_index + 2u];
    if (actq.kernel_id != ::ps::runtime::KernelId::ACTIVATION_QUANTIZE_W4A8 ||
        linear.kernel_id != ::ps::runtime::KernelId::LINEAR_PSQ8 ||
        sampling.kernel_id != ::ps::runtime::KernelId::SAMPLING) {
        return not_applicable;
    }
    if (actq.input_count != 1u || linear.output_count != 1u || sampling.output_count != 1u)
        return not_applicable;
    if (linear.weight_index == ::ps::runtime::kNoWeight ||
        linear.weight_index >= ctx.weight_count) {
        return not_applicable;
    }
    const ::ps::runtime::WeightSlot weight = ctx.weights[linear.weight_index];
    if (weight.encoding != 4u || !weight.preshuffled || weight.weight_scale_group != 32u ||
        (weight.k_padded % 32u) != 0u) {
        return not_applicable;
    }
    const uint32_t rows = ctx.actual_outputs != 0u ? ctx.actual_outputs : ctx.actual_rows;
    if (rows == 0u || rows > ctx.constraint_lm_head->max_rows())
        return not_applicable;
    const uint32_t candidate_capacity = select_constraint_lm_head_exact_capacity(
        ctx.actual_stochastic_outputs, ctx.actual_sampled_outputs, ctx.actual_outputs,
        ctx.constraint_allowed_counts, rows);
    if (candidate_capacity == 0u)
        return not_applicable;

    auto input = resolve_host_value(program, ::ps::runtime::ValueId{actq.input_slots[0]}, ctx);
    if (!input.ok())
        return input.status();
    auto output = resolve_host_value(program, ::ps::runtime::ValueId{sampling.output_slots[0]},
                                     ctx);
    if (!output.ok())
        return output.status();
    const HostResolvedValue in = input.value();
    const HostResolvedValue out = output.value();
    if (in.ptr == nullptr || out.ptr == nullptr)
        return not_applicable;
    if (in.dtype != ::ps::runtime::ValueDType::BF16 ||
        out.dtype != ::ps::runtime::ValueDType::I32) {
        return not_applicable;
    }
    if (in.feature_count != weight.cols)
        return not_applicable;

    Status cst = ctx.constraint_lm_head->select_exact(
        static_cast<const bf16_t*>(in.ptr), rows, in.row_stride, ctx.constraint_masks,
        ctx.constraint_mask_words, candidate_capacity, static_cast<int32_t*>(out.ptr), stream);
    if (!cst.ok())
        return cst;
    OptimizedLaunchResult constrained_result{};
    constrained_result.kind = OptimizedLaunchKind::Launched;
    constrained_result.consumed_dispatches = 3u;
    return constrained_result;
}

}  // namespace ps::qwen35::runtime
