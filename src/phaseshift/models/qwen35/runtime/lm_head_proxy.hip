#include <phaseshift/models/qwen35/runtime/lm_head_proxy.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/models/qwen35/runtime/optimized_dispatch.h>
#include <phaseshift/models/qwen35/kernels/dflash2/draft_head_int2.h>
#include <phaseshift/models/qwen35/kernels/dflash2/topk.h>
#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/models/qwen35/kernels/optimized/constraint_candidates.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

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

__global__ void shadow_compare_kernel(
    const int32_t* __restrict__ full_ids,
    const int32_t* __restrict__ proxy_ids,
    uint32_t rows,
    uint64_t* __restrict__ counters) {
    __shared__ uint32_t dec[64];
    __shared__ uint32_t mis[64];
    const uint32_t tid = threadIdx.x;
    uint32_t d = 0u;
    uint32_t m = 0u;
    for (uint32_t r = tid; r < rows; r += blockDim.x) {
        ++d;
        if (full_ids[r] != proxy_ids[r])
            ++m;
    }
    dec[tid] = d;
    mis[tid] = m;
    __syncthreads();
    if (tid == 0u) {
        uint32_t td = 0u;
        uint32_t tm = 0u;
        for (uint32_t t = 0u; t < blockDim.x; ++t) {
            td += dec[t];
            tm += mis[t];
        }
        counters[0] += td;
        counters[1] += tm;
    }
}

bool proxy_timing_enabled() {
    const char* env = std::getenv("PHASESHIFT_TARGET_LM_HEAD_PROXY_TIMING");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

struct ProxyTimers {
    double quant_ms = 0.0;
    double coarse_ms = 0.0;
    double topn_ms = 0.0;
    double rerank_ms = 0.0;
    double argmax_ms = 0.0;
    double total_ms = 0.0;
    uint64_t rounds = 0u;
};

struct ConstraintProxyTimers {
    double quant_ms = 0.0;
    double extract_ms = 0.0;
    double rerank_ms = 0.0;
    double argmax_ms = 0.0;
    double total_ms = 0.0;
    uint64_t rounds = 0u;
};

ConstraintProxyTimers& constraint_proxy_timers() {
    static ConstraintProxyTimers t;
    return t;
}

ProxyTimers& proxy_timers() {
    static ProxyTimers t;
    return t;
}

float proxy_event_ms(const hipEvent_t& a, const hipEvent_t& b) {
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, a, b);
    return ms;
}

void constraint_proxy_timing_report(const hipEvent_t* ev) {
    ConstraintProxyTimers& t = constraint_proxy_timers();
    t.quant_ms += proxy_event_ms(ev[0], ev[1]);
    t.extract_ms += proxy_event_ms(ev[1], ev[2]);
    t.rerank_ms += proxy_event_ms(ev[2], ev[3]);
    t.argmax_ms += proxy_event_ms(ev[3], ev[4]);
    t.total_ms += proxy_event_ms(ev[0], ev[4]);
    ++t.rounds;
    const double rounds = static_cast<double>(t.rounds);
    std::printf(
        "TARGET_LM_HEAD_PROXY_CONSTRAINT_TIMING rounds=%llu quant=%.3f extract=%.3f "
        "rerank=%.3f argmax=%.3f total=%.3f\n",
        static_cast<unsigned long long>(t.rounds), t.quant_ms / rounds,
        t.extract_ms / rounds, t.rerank_ms / rounds, t.argmax_ms / rounds,
        t.total_ms / rounds);
}

void proxy_timing_report(const hipEvent_t* ev) {
    ProxyTimers& t = proxy_timers();
    t.quant_ms += proxy_event_ms(ev[0], ev[1]);
    t.coarse_ms += proxy_event_ms(ev[1], ev[2]);
    t.topn_ms += proxy_event_ms(ev[2], ev[3]);
    t.rerank_ms += proxy_event_ms(ev[3], ev[4]);
    t.argmax_ms += proxy_event_ms(ev[4], ev[5]);
    t.total_ms += proxy_event_ms(ev[0], ev[5]);
    ++t.rounds;
    const double rounds = static_cast<double>(t.rounds);
    std::printf(
        "TARGET_LM_HEAD_PROXY_TIMING rounds=%llu quant=%.3f coarse=%.3f topn=%.3f "
        "rerank=%.3f argmax=%.3f total=%.3f\n",
        static_cast<unsigned long long>(t.rounds), t.quant_ms / rounds, t.coarse_ms / rounds,
        t.topn_ms / rounds, t.rerank_ms / rounds, t.argmax_ms / rounds, t.total_ms / rounds);
}

}  // namespace

LmHeadCandidateProxy::~LmHeadCandidateProxy() noexcept { (void)shutdown(); }

LmHeadCandidateProxy::LmHeadCandidateProxy(LmHeadCandidateProxy&& other) noexcept {
    move_from(other);
}

LmHeadCandidateProxy& LmHeadCandidateProxy::operator=(LmHeadCandidateProxy&& other) noexcept {
    if (this != &other) {
        (void)shutdown();
        move_from(other);
    }
    return *this;
}

void LmHeadCandidateProxy::move_from(LmHeadCandidateProxy& other) noexcept {
    weight_codes_ = other.weight_codes_;
    weight_scales_ = other.weight_scales_;
    vocab_ = other.vocab_;
    cols_ = other.cols_;
    k_padded_ = other.k_padded_;
    scale_stride_ = other.scale_stride_;
    max_rows_ = other.max_rows_;
    pool_ = other.pool_;
    candidate_capacity_ = other.candidate_capacity_;
    partitions_ = other.partitions_;
    radix_partitions_ = other.radix_partitions_;
    radix_scratch_bytes_ = other.radix_scratch_bytes_;
    scratch_stride_ = other.scratch_stride_;
    act_code_stride_ = other.act_code_stride_;
    act_scale_stride_ = other.act_scale_stride_;
    int2_codes_ = other.int2_codes_;
    tables_ = other.tables_;
    act_codes_ = other.act_codes_;
    act_scales_ = other.act_scales_;
    coarse_ = other.coarse_;
    pool_ids_ = other.pool_ids_;
    pool_logits_ = other.pool_logits_;
    cand_ids_ = other.cand_ids_;
    rerank_ = other.rerank_;
    scratch_ids_ = other.scratch_ids_;
    scratch_values_ = other.scratch_values_;
    radix_scratch_ = other.radix_scratch_;
    shadow_ids_ = other.shadow_ids_;
    shadow_counters_ = other.shadow_counters_;
    other.int2_codes_ = nullptr;
    other.tables_ = nullptr;
    other.act_codes_ = nullptr;
    other.act_scales_ = nullptr;
    other.coarse_ = nullptr;
    other.pool_ids_ = nullptr;
    other.pool_logits_ = nullptr;
    other.cand_ids_ = nullptr;
    other.rerank_ = nullptr;
    other.scratch_ids_ = nullptr;
    other.scratch_values_ = nullptr;
    other.radix_scratch_ = nullptr;
    other.shadow_ids_ = nullptr;
    other.shadow_counters_ = nullptr;
}

Status LmHeadCandidateProxy::shutdown() noexcept {
    if (shadow_counters_ != nullptr) {
        uint64_t counters[2] = {0ull, 0ull};
        const hipError_t cerr =
            hipMemcpy(counters, shadow_counters_, sizeof(counters), hipMemcpyDeviceToHost);
        if (cerr == hipSuccess && counters[0] != 0ull) {
            std::printf(
                "TARGET_LM_HEAD_PROXY_SHADOW pool=%u decisions=%llu mismatches=%llu "
                "mismatch_rate=%.6g\n",
                pool_, static_cast<unsigned long long>(counters[0]),
                static_cast<unsigned long long>(counters[1]),
                static_cast<double>(counters[1]) / static_cast<double>(counters[0]));
        }
    }
    Status first_error = Status::make_ok();
    auto free_ptr = [&first_error](void*& p) {
        if (p != nullptr) {
            const hipError_t err = hipFree(p);
            if (first_error.ok() && err != hipSuccess) {
                first_error = Status::hip_error("hipFree lm_head proxy buffer",
                                                hipGetErrorString(err), __FILE__, __LINE__);
            }
            p = nullptr;
        }
    };
    free_ptr(reinterpret_cast<void*&>(int2_codes_));
    free_ptr(reinterpret_cast<void*&>(tables_));
    free_ptr(reinterpret_cast<void*&>(act_codes_));
    free_ptr(reinterpret_cast<void*&>(act_scales_));
    free_ptr(reinterpret_cast<void*&>(coarse_));
    free_ptr(reinterpret_cast<void*&>(pool_ids_));
    free_ptr(reinterpret_cast<void*&>(pool_logits_));
    free_ptr(reinterpret_cast<void*&>(cand_ids_));
    free_ptr(reinterpret_cast<void*&>(rerank_));
    free_ptr(reinterpret_cast<void*&>(scratch_ids_));
    free_ptr(reinterpret_cast<void*&>(scratch_values_));
    free_ptr(reinterpret_cast<void*&>(radix_scratch_));
    free_ptr(reinterpret_cast<void*&>(shadow_ids_));
    free_ptr(reinterpret_cast<void*&>(shadow_counters_));
    weight_codes_ = nullptr;
    weight_scales_ = nullptr;
    candidate_capacity_ = 0u;
    return first_error;
}

Status LmHeadCandidateProxy::init(const LmHeadProxyWeightView& weight, uint32_t max_rows,
                                  uint32_t pool, hipStream_t stream) {
    if (weight.codes == nullptr || weight.scales == nullptr || weight.vocab == 0u ||
        weight.cols == 0u || weight.k_padded == 0u || weight.scale_stride_bytes == 0u) {
        return Status::invalid_argument("lm_head proxy weight view invalid", __FILE__, __LINE__);
    }
    if (max_rows == 0u || max_rows > 64u) {
        return Status::invalid_argument("lm_head proxy max rows out of range", __FILE__, __LINE__);
    }
    if (pool == 0u || pool > ps::kernel::kDflash2Int2MaxPool) {
        return Status::invalid_argument("lm_head proxy pool out of range", __FILE__, __LINE__);
    }
    if ((weight.k_padded % 32u) != 0u) {
        return Status::invalid_argument("lm_head proxy k_padded not 32-aligned", __FILE__,
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
    pool_ = pool;
    partitions_ = ps::kernel::dflash2_int2_default_partitions(vocab_);
    scratch_stride_ = partitions_ * ps::kernel::kDflash2Int2MaxPool;
    candidate_capacity_ = ps::kernel::kDflash2Int2MaxPool;

    const auto layout = ::ps::runtime::Int8ActivationWorkspaceLayout::make(cols_, max_rows_);
    if (layout.k_padded != k_padded_) {
        (void)shutdown();
        return Status::invalid_argument("lm_head proxy activation layout mismatch", __FILE__,
                                        __LINE__);
    }
    act_code_stride_ = layout.code_row_stride_bytes;
    act_scale_stride_ = layout.scale_row_stride_bytes;

    const std::size_t code_bytes =
        ((static_cast<std::size_t>(vocab_) + 15u) / 16u) * k_padded_ * 4u;
    bool ok = true;
    auto alloc = [&ok](void** p, std::size_t bytes) {
        ok = ok && (hipMalloc(p, bytes) == hipSuccess);
    };
    alloc(reinterpret_cast<void**>(&int2_codes_), code_bytes);
    alloc(reinterpret_cast<void**>(&tables_), 1280u);
    alloc(reinterpret_cast<void**>(&act_codes_),
          static_cast<std::size_t>(act_code_stride_) * ((max_rows_ + 15u) / 16u) * 16u);
    alloc(reinterpret_cast<void**>(&act_scales_),
          static_cast<std::size_t>(act_scale_stride_) * max_rows_);
    alloc(reinterpret_cast<void**>(&coarse_),
          static_cast<std::size_t>(max_rows_) * vocab_ * sizeof(float));
    alloc(reinterpret_cast<void**>(&pool_ids_),
          static_cast<std::size_t>(max_rows_) * candidate_capacity_ * sizeof(int32_t));
    alloc(reinterpret_cast<void**>(&pool_logits_),
          static_cast<std::size_t>(max_rows_) * candidate_capacity_ * sizeof(float));
    alloc(reinterpret_cast<void**>(&cand_ids_),
          static_cast<std::size_t>(max_rows_) * candidate_capacity_ * sizeof(int32_t));
    alloc(reinterpret_cast<void**>(&rerank_),
          static_cast<std::size_t>(max_rows_) * candidate_capacity_ * sizeof(float));
    alloc(reinterpret_cast<void**>(&scratch_ids_),
          static_cast<std::size_t>(max_rows_) * scratch_stride_ * sizeof(int32_t));
    alloc(reinterpret_cast<void**>(&scratch_values_),
          static_cast<std::size_t>(max_rows_) * scratch_stride_ * sizeof(float));
    radix_partitions_ = ps::kernel::dflash2_radix_default_partitions(vocab_);
    radix_scratch_bytes_ =
        ps::kernel::dflash2_radix_topn_scratch_bytes(max_rows_, radix_partitions_, vocab_);
    alloc(reinterpret_cast<void**>(&radix_scratch_), radix_scratch_bytes_);
    alloc(reinterpret_cast<void**>(&shadow_ids_),
          static_cast<std::size_t>(max_rows_) * sizeof(int32_t));
    alloc(reinterpret_cast<void**>(&shadow_counters_), 2u * sizeof(uint64_t));
    if (!ok) {
        (void)shutdown();
        return Status::hip_error("lm_head proxy allocation failed", "hipMalloc", __FILE__,
                                 __LINE__);
    }
    if (hipMemset(shadow_counters_, 0, 2u * sizeof(uint64_t)) != hipSuccess) {
        (void)shutdown();
        return Status::hip_error("lm_head proxy shadow counter reset", "hipMemset", __FILE__,
                                 __LINE__);
    }

    uint64_t* histogram = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&histogram), 256u * sizeof(uint64_t)) != hipSuccess) {
        (void)shutdown();
        return Status::hip_error("lm_head proxy histogram allocation failed", "hipMalloc",
                                 __FILE__, __LINE__);
    }
    hipError_t herr = hipMemset(histogram, 0, 256u * sizeof(uint64_t));
    herr = herr == hipSuccess
               ? ps::kernel::launch_dflash2_int2_code_histogram(
                     weight_codes_, weight_scales_, vocab_, k_padded_, scale_stride_, histogram,
                     stream)
               : herr;
    std::vector<uint64_t> host_hist(256u, 0ull);
    if (herr == hipSuccess) {
        herr = hipMemcpy(host_hist.data(), histogram, 256u * sizeof(uint64_t),
                         hipMemcpyDeviceToHost);
    }
    (void)hipFree(histogram);
    if (herr != hipSuccess) {
        (void)shutdown();
        return Status::hip_error("lm_head proxy histogram failed", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    }
    const ps::kernel::Dflash2Int2Codebook codebook =
        ps::kernel::dflash2_int2_derive_codebook(host_hist.data(), false);

    std::vector<uint8_t> host_tables(1280u, 0u);
    std::vector<uint8_t> map_table(256u, 0u);
    ps::kernel::dflash2_int2_build_tables(
        codebook, map_table.data(),
        reinterpret_cast<uint32_t*>(host_tables.data() + 256u));
    std::memcpy(host_tables.data(), map_table.data(), 256u);
    herr = hipMemcpy(tables_, host_tables.data(), host_tables.size(), hipMemcpyHostToDevice);
    if (herr != hipSuccess) {
        (void)shutdown();
        return Status::hip_error("lm_head proxy table upload failed", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    }
    herr = ps::kernel::launch_dflash2_int2_pack(weight_codes_, tables_, int2_codes_, vocab_,
                                                k_padded_, stream);
    if (herr == hipSuccess) {
        herr = hipStreamSynchronize(stream);
    }
    if (herr != hipSuccess) {
        (void)shutdown();
        return Status::hip_error("lm_head proxy int2 pack failed", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status LmHeadCandidateProxy::select(const bf16_t* normed, uint32_t rows, uint32_t row_stride,
                                    int32_t* out_ids, hipStream_t stream) {
    return run_select(normed, rows, row_stride, out_ids, stream);
}

Status LmHeadCandidateProxy::select_shadow(const bf16_t* normed, uint32_t rows,
                                           uint32_t row_stride, hipStream_t stream) {
    return run_select(normed, rows, row_stride, shadow_ids_, stream);
}

Status LmHeadCandidateProxy::run_select(const bf16_t* normed, uint32_t rows,
                                        uint32_t row_stride, int32_t* out_ids,
                                        hipStream_t stream) {
    if (!valid()) {
        return Status::invalid_state("lm_head proxy not initialized", __FILE__, __LINE__);
    }
    if (normed == nullptr || out_ids == nullptr || rows == 0u || rows > max_rows_) {
        return Status::invalid_argument("lm_head proxy select arguments invalid", __FILE__,
                                        __LINE__);
    }
    const bool timing = proxy_timing_enabled();
    hipEvent_t ev[6] = {};
    struct EventGuard {
        hipEvent_t* events;
        bool active;
        ~EventGuard() {
            if (!active)
                return;
            for (uint32_t i = 0u; i < 6u; ++i)
                (void)hipEventDestroy(events[i]);
        }
    } guard{ev, false};
    if (timing) {
        for (auto& e : ev)
            (void)hipEventCreate(&e);
        guard.active = true;
        (void)hipEventRecord(ev[0], stream);
    }
    hipError_t err = ps::kernel::launch_activation_quantize_e4m3(
        normed, act_codes_, act_scales_, rows, cols_, k_padded_, row_stride, act_code_stride_,
        act_scale_stride_, stream);
    if (err != hipSuccess) {
        return Status::hip_error("lm_head proxy activation quantize failed",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    if (timing)
        (void)hipEventRecord(ev[1], stream);
    const uint32_t* expand_table = reinterpret_cast<const uint32_t*>(tables_ + 256u);
    err = ps::kernel::launch_dflash2_int2_coarse_head(
        int2_codes_, weight_scales_, act_codes_, act_scales_, expand_table, coarse_, rows, vocab_,
        k_padded_, scale_stride_, act_code_stride_, act_scale_stride_, vocab_, stream);
    if (err != hipSuccess) {
        return Status::hip_error("lm_head proxy int2 coarse head failed", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    if (timing)
        (void)hipEventRecord(ev[2], stream);
    if (ps::kernel::dflash2_radix_topn_preferred(pool_)) {
        err = ps::kernel::launch_dflash2_radix_topn(
            coarse_, rows, vocab_, vocab_, pool_, pool_ids_, pool_logits_,
            radix_partitions_, radix_scratch_, radix_scratch_bytes_, stream);
    } else {
        err = ps::kernel::launch_dflash2_coarse_topn(
            coarse_, rows, vocab_, vocab_, pool_, pool_ids_, pool_logits_, scratch_ids_,
            scratch_values_, scratch_stride_, partitions_, stream);
    }
    if (err != hipSuccess) {
        return Status::hip_error("lm_head proxy coarse topn failed", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    if (timing)
        (void)hipEventRecord(ev[3], stream);
    err = ps::kernel::launch_dflash2_psq8_candidate_rerank(
        weight_codes_, weight_scales_, act_codes_, act_scales_, pool_ids_, rerank_, rows, pool_,
        k_padded_, scale_stride_, act_code_stride_, act_scale_stride_, pool_, stream);
    if (err != hipSuccess) {
        return Status::hip_error("lm_head proxy candidate rerank failed", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    if (timing)
        (void)hipEventRecord(ev[4], stream);
    pool_argmax_remap_kernel<<<rows, 1, 0, stream>>>(rerank_, pool_ids_, out_ids, pool_);
    err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("lm_head proxy argmax failed", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    if (timing) {
        (void)hipEventRecord(ev[5], stream);
        (void)hipEventSynchronize(ev[5]);
        proxy_timing_report(ev);
    }
    return Status::make_ok();
}

Status LmHeadCandidateProxy::select_constrained_exact(const bf16_t* normed, uint32_t rows,
                                                      uint32_t row_stride,
                                                      const uint32_t* masks,
                                                      uint32_t mask_words,
                                                      uint32_t candidate_capacity,
                                                      int32_t* out_ids, hipStream_t stream) {
    return run_select_constrained_exact(normed, rows, row_stride, masks, mask_words,
                                        candidate_capacity, out_ids, stream);
}

Status LmHeadCandidateProxy::select_constrained_exact_shadow(const bf16_t* normed,
                                                             uint32_t rows,
                                                             uint32_t row_stride,
                                                             const uint32_t* masks,
                                                             uint32_t mask_words,
                                                             uint32_t candidate_capacity,
                                                             hipStream_t stream) {
    return run_select_constrained_exact(normed, rows, row_stride, masks, mask_words,
                                        candidate_capacity, shadow_ids_, stream);
}

Status LmHeadCandidateProxy::run_select_constrained_exact(
    const bf16_t* normed, uint32_t rows, uint32_t row_stride, const uint32_t* masks,
    uint32_t mask_words, uint32_t candidate_capacity, int32_t* out_ids, hipStream_t stream) {
    if (!valid()) {
        return Status::invalid_state("lm_head proxy not initialized", __FILE__, __LINE__);
    }
    if (normed == nullptr || out_ids == nullptr || masks == nullptr || rows == 0u ||
        rows > max_rows_) {
        return Status::invalid_argument("lm_head proxy constrained select arguments invalid",
                                        __FILE__, __LINE__);
    }
    if (mask_words == 0u || candidate_capacity == 0u ||
        candidate_capacity > candidate_capacity_ ||
        mask_words < (vocab_ + 31u) / 32u) {
        return Status::invalid_argument("lm_head proxy constrained capacity invalid",
                                        __FILE__, __LINE__);
    }
    const bool timing = proxy_timing_enabled();
    hipEvent_t ev[5] = {};
    struct EventGuard {
        hipEvent_t* events;
        bool active;
        ~EventGuard() {
            if (!active)
                return;
            for (uint32_t i = 0u; i < 5u; ++i)
                (void)hipEventDestroy(events[i]);
        }
    } guard{ev, false};
    if (timing) {
        for (auto& e : ev)
            (void)hipEventCreate(&e);
        guard.active = true;
        (void)hipEventRecord(ev[0], stream);
    }
    hipError_t err = ps::kernel::launch_activation_quantize_e4m3(
        normed, act_codes_, act_scales_, rows, cols_, k_padded_, row_stride, act_code_stride_,
        act_scale_stride_, stream);
    if (err != hipSuccess) {
        return Status::hip_error("lm_head proxy activation quantize failed",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    if (timing)
        (void)hipEventRecord(ev[1], stream);
    err = ps::kernel::launch_constraint_mask_to_candidates(
        masks, rows, mask_words, vocab_, candidate_capacity, cand_ids_, stream);
    if (err != hipSuccess) {
        return Status::hip_error("lm_head proxy constraint candidate extraction failed",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    if (timing)
        (void)hipEventRecord(ev[2], stream);
    err = ps::kernel::launch_dflash2_psq8_candidate_rerank(
        weight_codes_, weight_scales_, act_codes_, act_scales_, cand_ids_, rerank_, rows,
        candidate_capacity, k_padded_, scale_stride_, act_code_stride_, act_scale_stride_,
        candidate_capacity, stream);
    if (err != hipSuccess) {
        return Status::hip_error("lm_head proxy constrained rerank failed",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    if (timing)
        (void)hipEventRecord(ev[3], stream);
    pool_argmax_remap_kernel<<<rows, 1, 0, stream>>>(rerank_, cand_ids_, out_ids,
                                                     candidate_capacity);
    err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("lm_head proxy constrained argmax failed",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    if (timing) {
        (void)hipEventRecord(ev[4], stream);
        (void)hipEventSynchronize(ev[4]);
        constraint_proxy_timing_report(ev);
    }
    return Status::make_ok();
}

Status LmHeadCandidateProxy::select_constrained_masked(const bf16_t* normed, uint32_t rows,
                                                       uint32_t row_stride,
                                                       const uint32_t* masks,
                                                       uint32_t mask_words, int32_t* out_ids,
                                                       hipStream_t stream) {
    return run_select_constrained_masked(normed, rows, row_stride, masks, mask_words, out_ids,
                                         stream);
}

Status LmHeadCandidateProxy::select_constrained_masked_shadow(const bf16_t* normed,
                                                              uint32_t rows,
                                                              uint32_t row_stride,
                                                              const uint32_t* masks,
                                                              uint32_t mask_words,
                                                              hipStream_t stream) {
    return run_select_constrained_masked(normed, rows, row_stride, masks, mask_words,
                                         shadow_ids_, stream);
}

Status LmHeadCandidateProxy::run_select_constrained_masked(
    const bf16_t* normed, uint32_t rows, uint32_t row_stride, const uint32_t* masks,
    uint32_t mask_words, int32_t* out_ids, hipStream_t stream) {
    if (!valid()) {
        return Status::invalid_state("lm_head proxy not initialized", __FILE__, __LINE__);
    }
    if (normed == nullptr || out_ids == nullptr || masks == nullptr || rows == 0u ||
        rows > max_rows_) {
        return Status::invalid_argument("lm_head proxy masked select arguments invalid",
                                        __FILE__, __LINE__);
    }
    if (mask_words == 0u || mask_words < (vocab_ + 31u) / 32u || pool_ == 0u) {
        return Status::invalid_argument("lm_head proxy masked capacity invalid", __FILE__,
                                        __LINE__);
    }
    const bool timing = proxy_timing_enabled();
    hipEvent_t ev[6] = {};
    struct EventGuard {
        hipEvent_t* events;
        bool active;
        ~EventGuard() {
            if (!active)
                return;
            for (uint32_t i = 0u; i < 6u; ++i)
                (void)hipEventDestroy(events[i]);
        }
    } guard{ev, false};
    if (timing) {
        for (auto& e : ev)
            (void)hipEventCreate(&e);
        guard.active = true;
        (void)hipEventRecord(ev[0], stream);
    }
    hipError_t err = ps::kernel::launch_activation_quantize_e4m3(
        normed, act_codes_, act_scales_, rows, cols_, k_padded_, row_stride, act_code_stride_,
        act_scale_stride_, stream);
    if (err != hipSuccess) {
        return Status::hip_error("lm_head proxy activation quantize failed",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    if (timing)
        (void)hipEventRecord(ev[1], stream);
    const uint32_t* expand_table = reinterpret_cast<const uint32_t*>(tables_ + 256u);
    err = ps::kernel::launch_dflash2_int2_coarse_head(
        int2_codes_, weight_scales_, act_codes_, act_scales_, expand_table, coarse_, rows, vocab_,
        k_padded_, scale_stride_, act_code_stride_, act_scale_stride_, vocab_, stream);
    if (err != hipSuccess) {
        return Status::hip_error("lm_head proxy int2 coarse head failed", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    err = ps::kernel::launch_dflash2_apply_constraint_mask(coarse_, rows, vocab_, vocab_, masks,
                                                           mask_words, stream);
    if (err != hipSuccess) {
        return Status::hip_error("lm_head proxy constraint mask apply failed",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    if (timing)
        (void)hipEventRecord(ev[2], stream);
    if (ps::kernel::dflash2_radix_topn_preferred(pool_)) {
        err = ps::kernel::launch_dflash2_radix_topn(
            coarse_, rows, vocab_, vocab_, pool_, pool_ids_, pool_logits_,
            radix_partitions_, radix_scratch_, radix_scratch_bytes_, stream);
    } else {
        err = ps::kernel::launch_dflash2_coarse_topn(
            coarse_, rows, vocab_, vocab_, pool_, pool_ids_, pool_logits_, scratch_ids_,
            scratch_values_, scratch_stride_, partitions_, stream);
    }
    if (err != hipSuccess) {
        return Status::hip_error("lm_head proxy masked topn failed", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    if (timing)
        (void)hipEventRecord(ev[3], stream);
    err = ps::kernel::launch_dflash2_psq8_candidate_rerank(
        weight_codes_, weight_scales_, act_codes_, act_scales_, pool_ids_, rerank_, rows, pool_,
        k_padded_, scale_stride_, act_code_stride_, act_scale_stride_, pool_, stream);
    if (err != hipSuccess) {
        return Status::hip_error("lm_head proxy masked rerank failed", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    if (timing)
        (void)hipEventRecord(ev[4], stream);
    pool_argmax_remap_kernel<<<rows, 1, 0, stream>>>(rerank_, pool_ids_, out_ids, pool_);
    err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("lm_head proxy masked argmax failed", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    if (timing) {
        (void)hipEventRecord(ev[5], stream);
        (void)hipEventSynchronize(ev[5]);
        proxy_timing_report(ev);
    }
    return Status::make_ok();
}

Status LmHeadCandidateProxy::compare_shadow(const int32_t* full_ids, uint32_t rows,
                                            hipStream_t stream) {
    if (shadow_counters_ == nullptr || shadow_ids_ == nullptr || full_ids == nullptr ||
        rows == 0u || rows > max_rows_) {        return Status::invalid_argument("lm_head proxy shadow compare invalid", __FILE__,
                                        __LINE__);
    }
    shadow_compare_kernel<<<1, 64, 0, stream>>>(full_ids, shadow_ids_, rows,
                                                shadow_counters_);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("lm_head proxy shadow compare launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

uint32_t target_lm_head_proxy_mode() {
    const char* env = std::getenv("PHASESHIFT_TARGET_LM_HEAD_PROXY");
    if (env == nullptr || env[0] == '\0')
        return 1u;
    const long v = std::strtol(env, nullptr, 10);
    if (v <= 0)
        return 0u;
    if (v >= 2)
        return 2u;
    return 1u;
}

uint32_t target_lm_head_proxy_decode_mode() {
    const char* env = std::getenv("PHASESHIFT_TARGET_LM_HEAD_PROXY");
    if (env == nullptr || env[0] == '\0')
        return 0u;
    return target_lm_head_proxy_mode();
}

uint32_t target_lm_head_proxy_pool() {
    const char* env = std::getenv("PHASESHIFT_TARGET_LM_HEAD_PROXY_POOL");
    uint32_t pool = 32u;
    if (env != nullptr && env[0] != '\0') {
        const long v = std::strtol(env, nullptr, 10);
        if (v > 0)
            pool = static_cast<uint32_t>(v);
    }
    if (pool > ps::kernel::kDflash2Int2MaxPool)
        pool = ps::kernel::kDflash2Int2MaxPool;
    return pool;
}

uint32_t target_lm_head_proxy_constraint_threshold() {
    const char* env = std::getenv("PHASESHIFT_TARGET_LM_HEAD_PROXY_CONSTRAINT_THRESHOLD");
    uint32_t threshold = ps::kernel::kDflash2Int2MaxPool;
    if (env != nullptr && env[0] != '\0') {
        const long v = std::strtol(env, nullptr, 10);
        if (v > 0)
            threshold = static_cast<uint32_t>(v);
    }
    if (threshold == 0u)
        threshold = 1u;
    if (threshold > ps::kernel::kDflash2Int2MaxPool)
        threshold = ps::kernel::kDflash2Int2MaxPool;
    return threshold;
}

Result<OptimizedLaunchResult> try_launch_lm_head_proxy_fusion(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    OptimizedLaunchResult not_applicable{};
    if (ctx.lm_head_proxy == nullptr || !ctx.lm_head_proxy->valid())
        return not_applicable;
    const bool constrained =
        ctx.constraint_masks != nullptr || ctx.constraint_mask_words != 0u;
    const uint32_t mode = ctx.role == ::ps::runtime::ExecutionRole::Verify
                              ? target_lm_head_proxy_mode()
                              : target_lm_head_proxy_decode_mode();
    if (constrained) {
        if (ctx.role != ::ps::runtime::ExecutionRole::Decode || mode == 0u)
            return not_applicable;
    } else {
        const LmHeadProxyPath path =
            lm_head_proxy_path(ctx.role, ctx.actual_stochastic_outputs, false, mode);
        if (path == LmHeadProxyPath::None)
            return not_applicable;
    }
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
    const uint32_t rows = ctx.actual_outputs != 0u ? ctx.actual_outputs : ctx.actual_rows;
    if (rows == 0u || rows > ctx.lm_head_proxy->max_rows())
        return not_applicable;
    if (constrained) {
        const LmHeadConstrainedSelection selection = select_lm_head_constrained(
            ctx.actual_stochastic_outputs, ctx.actual_sampled_outputs, ctx.actual_outputs,
            ctx.constraint_allowed_counts, rows, target_lm_head_proxy_constraint_threshold(),
            ctx.lm_head_proxy->coarse_pool());
        if (selection.path == LmHeadConstrainedPath::None)
            return not_applicable;
        if (selection.path == LmHeadConstrainedPath::ExactCandidates) {
            if (mode == 2u) {
                Status cst = ctx.lm_head_proxy->select_constrained_exact_shadow(
                    static_cast<const bf16_t*>(in.ptr), rows, in.row_stride,
                    ctx.constraint_masks, ctx.constraint_mask_words,
                    selection.candidate_capacity, stream);
                if (!cst.ok())
                    return cst;
                ctx.lm_head_shadow_pending = true;
                ctx.lm_head_shadow_full = static_cast<const int32_t*>(out.ptr);
                ctx.lm_head_shadow_rows = rows;
                return not_applicable;
            }
            Status cst = ctx.lm_head_proxy->select_constrained_exact(
                static_cast<const bf16_t*>(in.ptr), rows, in.row_stride, ctx.constraint_masks,
                ctx.constraint_mask_words, selection.candidate_capacity,
                static_cast<int32_t*>(out.ptr), stream);
            if (!cst.ok())
                return cst;
            OptimizedLaunchResult constrained_result{};
            constrained_result.kind = OptimizedLaunchKind::Launched;
            constrained_result.consumed_dispatches = 3u;
            return constrained_result;
        }
        if (mode == 2u) {
            Status mst = ctx.lm_head_proxy->select_constrained_masked_shadow(
                static_cast<const bf16_t*>(in.ptr), rows, in.row_stride, ctx.constraint_masks,
                ctx.constraint_mask_words, stream);
            if (!mst.ok())
                return mst;
            ctx.lm_head_shadow_pending = true;
            ctx.lm_head_shadow_full = static_cast<const int32_t*>(out.ptr);
            ctx.lm_head_shadow_rows = rows;
            return not_applicable;
        }
        Status mst = ctx.lm_head_proxy->select_constrained_masked(
            static_cast<const bf16_t*>(in.ptr), rows, in.row_stride, ctx.constraint_masks,
            ctx.constraint_mask_words, static_cast<int32_t*>(out.ptr), stream);
        if (!mst.ok())
            return mst;
        OptimizedLaunchResult masked_result{};
        masked_result.kind = OptimizedLaunchKind::Launched;
        masked_result.consumed_dispatches = 3u;
        return masked_result;
    }
    if (mode == 2u) {
        Status st = ctx.lm_head_proxy->select_shadow(
            static_cast<const bf16_t*>(in.ptr), rows, in.row_stride, stream);
        if (!st.ok())
            return st;
        ctx.lm_head_shadow_pending = true;
        ctx.lm_head_shadow_full = static_cast<const int32_t*>(out.ptr);
        ctx.lm_head_shadow_rows = rows;
        return not_applicable;
    }
    Status st = ctx.lm_head_proxy->select(
        static_cast<const bf16_t*>(in.ptr), rows, in.row_stride,
        static_cast<int32_t*>(out.ptr), stream);
    if (!st.ok())
        return st;
    OptimizedLaunchResult result{};
    result.kind = OptimizedLaunchKind::Launched;
    result.consumed_dispatches = 3u;
    return result;
}

}  // namespace ps::qwen35::runtime
