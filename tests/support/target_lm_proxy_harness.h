#pragma once

#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/dflash2/draft_head_int2.h>
#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/quantization/fpx/e4m3.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>
#include <phaseshift/weights/matrix_weight.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace target_lm_proxy {

__global__ void gather_candidates_kernel(
    const int32_t* __restrict__ top_ids,
    int32_t* __restrict__ out_ids,
    const uint32_t total,
    const uint32_t pool,
    const uint32_t top_stride) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const uint32_t row = idx / pool;
    const uint32_t col = idx - row * pool;
    out_ids[idx] = top_ids[static_cast<std::size_t>(row) * top_stride + col];
}

struct Proxy {
    const ps::weights::MatrixWeight* weight = nullptr;
    uint32_t vocab = 0u;
    uint32_t k_padded = 0u;
    uint32_t cols = 0u;
    uint32_t scale_stride = 0u;
    uint32_t max_rows = 0u;
    uint32_t max_pool = 0u;
    uint32_t partitions = 0u;
    uint32_t scratch_stride = 0u;
    uint32_t act_code_stride = 0u;
    uint32_t act_scale_stride = 0u;
    uint32_t top_stride = 0u;

    ps::kernel::Dflash2Int2Codebook codebook{};
    uint8_t* int2_codes = nullptr;
    uint8_t* tables = nullptr;
    float* error_table = nullptr;
    float* error_l2 = nullptr;

    uint8_t* act_codes = nullptr;
    float* act_scales = nullptr;
    float* act_l2 = nullptr;
    float* coarse = nullptr;
    float* upper = nullptr;
    int32_t* top_ids = nullptr;
    float* top_values = nullptr;
    int32_t* cand_ids = nullptr;
    float* rerank = nullptr;
    int32_t* scratch_ids = nullptr;
    float* scratch_values = nullptr;
    int32_t* best_ids = nullptr;
    float* best_logits = nullptr;
    float* gaps = nullptr;
    uint8_t* certified = nullptr;
    int32_t* proposed_dev = nullptr;
    float* exact_proposed = nullptr;
    float* max_other_upper = nullptr;

    const uint32_t* expand_table() const {
        return reinterpret_cast<const uint32_t*>(tables + 256u);
    }

    bool init(const ps::weights::MatrixWeight& w, uint32_t rows, uint32_t pool,
              hipStream_t stream, std::string& err) {
        weight = &w;
        if (w.encoding != ps::weights::MatrixEncoding::Psq8 || !w.preshuffled ||
            w.weight_scale_group != 32u || (w.k_padded % 32u) != 0u) {
            err = "lm_head is not a preshuffled psq8 w32 weight";
            return false;
        }
        if (pool == 0u || pool >= ps::kernel::kDflash2Int2MaxPool) {
            err = "pool out of range";
            return false;
        }
        if (rows == 0u || rows > 16u) {
            err = "rows must be in [1,16]";
            return false;
        }
        vocab = w.rows;
        k_padded = w.k_padded;
        cols = w.cols;
        scale_stride = w.storage_scale_stride_bytes;
        max_rows = rows;
        max_pool = pool;
        partitions = ps::kernel::dflash2_int2_default_partitions(vocab);
        scratch_stride = partitions * ps::kernel::kDflash2Int2MaxPool;

        const auto layout = ps::runtime::Int8ActivationWorkspaceLayout::make(cols, rows);
        if (layout.k_padded != k_padded) {
            err = "activation layout k_padded mismatch";
            return false;
        }
        act_code_stride = layout.code_row_stride_bytes;
        act_scale_stride = layout.scale_row_stride_bytes;

        const std::size_t code_bytes =
            ((static_cast<std::size_t>(vocab) + 15u) / 16u) * k_padded * 4u;
        bool ok = true;
#define TP_ALLOC(p, bytes) ok = ok && (hipMalloc(&(p), (bytes)) == hipSuccess)
        TP_ALLOC(int2_codes, code_bytes);
        TP_ALLOC(tables, 1280u);
        TP_ALLOC(error_table, 256u * sizeof(float));
        TP_ALLOC(error_l2, vocab * sizeof(float));
        TP_ALLOC(act_codes, static_cast<std::size_t>(act_code_stride) * ((rows + 15u) / 16u) * 16u);
        TP_ALLOC(act_scales, static_cast<std::size_t>(act_scale_stride / 4u) * rows * 4u);
        TP_ALLOC(act_l2, rows * sizeof(float));
        TP_ALLOC(coarse, static_cast<std::size_t>(rows) * vocab * sizeof(float));
        TP_ALLOC(upper, static_cast<std::size_t>(rows) * vocab * sizeof(float));
        TP_ALLOC(top_ids, static_cast<std::size_t>(rows) * (pool + 1u) * sizeof(int32_t));
        TP_ALLOC(top_values, static_cast<std::size_t>(rows) * (pool + 1u) * sizeof(float));
        TP_ALLOC(cand_ids, static_cast<std::size_t>(rows) * pool * sizeof(int32_t));
        TP_ALLOC(rerank, static_cast<std::size_t>(rows) * pool * sizeof(float));
        TP_ALLOC(scratch_ids, static_cast<std::size_t>(rows) * scratch_stride * sizeof(int32_t));
        TP_ALLOC(scratch_values,
                 static_cast<std::size_t>(rows) * scratch_stride * sizeof(float));
        TP_ALLOC(best_ids, rows * sizeof(int32_t));
        TP_ALLOC(best_logits, rows * sizeof(float));
        TP_ALLOC(gaps, rows * sizeof(float));
        TP_ALLOC(certified, rows);
        TP_ALLOC(proposed_dev, rows * sizeof(int32_t));
        TP_ALLOC(exact_proposed, rows * sizeof(float));
        TP_ALLOC(max_other_upper, rows * sizeof(float));
#undef TP_ALLOC
        if (!ok) {
            err = "device allocation failed";
            return false;
        }
        top_stride = pool + 1u;

        uint64_t* histogram = nullptr;
        if (hipMalloc(&histogram, 256u * sizeof(uint64_t)) != hipSuccess) {
            err = "histogram allocation failed";
            return false;
        }
        hipError_t herr = hipMemset(histogram, 0, 256u * sizeof(uint64_t));
        herr = herr == hipSuccess
                   ? ps::kernel::launch_dflash2_int2_code_histogram(
                         w.codes.data<uint8_t>(), w.scales.data<uint8_t>(), vocab, k_padded,
                         scale_stride, histogram, stream)
                   : herr;
        std::vector<uint64_t> host_hist(256u, 0ull);
        if (herr == hipSuccess)
            herr = hipMemcpy(host_hist.data(), histogram, 256u * sizeof(uint64_t),
                             hipMemcpyDeviceToHost);
        (void)hipFree(histogram);
        if (herr != hipSuccess) {
            err = "int2 histogram failed";
            return false;
        }
        codebook = ps::kernel::dflash2_int2_derive_codebook(host_hist.data(), false);

        std::vector<uint8_t> host_tables(1280u, 0u);
        std::vector<uint8_t> map_table(256u, 0u);
        ps::kernel::dflash2_int2_build_tables(
            codebook, map_table.data(),
            reinterpret_cast<uint32_t*>(host_tables.data() + 256u));
        std::memcpy(host_tables.data(), map_table.data(), 256u);
        std::vector<float> host_error(256u, 0.0f);
        ps::kernel::dflash2_int2_build_error_table(codebook, map_table.data(),
                                                   host_error.data());
        if (hipMemcpy(tables, host_tables.data(), host_tables.size(), hipMemcpyHostToDevice) !=
                hipSuccess ||
            hipMemcpy(error_table, host_error.data(), host_error.size() * sizeof(float),
                      hipMemcpyHostToDevice) != hipSuccess) {
            err = "table upload failed";
            return false;
        }

        herr = ps::kernel::launch_dflash2_int2_pack(
            w.codes.data<uint8_t>(), tables, int2_codes, vocab, k_padded, stream);
        herr = herr == hipSuccess
                   ? ps::kernel::launch_dflash2_int2_row_error_l2(
                         w.codes.data<uint8_t>(), w.scales.data<uint8_t>(), error_table, vocab,
                         k_padded, scale_stride, error_l2, stream)
                   : herr;
        if (herr != hipSuccess) {
            err = "int2 pack / error l2 failed";
            return false;
        }
        return true;
    }

    void shutdown() {
#define TP_FREE(p) (void)hipFree(p)
        TP_FREE(int2_codes);
        TP_FREE(tables);
        TP_FREE(error_table);
        TP_FREE(error_l2);
        TP_FREE(act_codes);
        TP_FREE(act_scales);
        TP_FREE(act_l2);
        TP_FREE(coarse);
        TP_FREE(upper);
        TP_FREE(top_ids);
        TP_FREE(top_values);
        TP_FREE(cand_ids);
        TP_FREE(rerank);
        TP_FREE(scratch_ids);
        TP_FREE(scratch_values);
        TP_FREE(best_ids);
        TP_FREE(best_logits);
        TP_FREE(gaps);
        TP_FREE(certified);
        TP_FREE(proposed_dev);
        TP_FREE(exact_proposed);
        TP_FREE(max_other_upper);
#undef TP_FREE
    }

    bool prepare(const ps::bf16_t* normed, uint32_t rows, uint32_t normed_row_stride,
                 hipStream_t stream, std::string& err) {
        if (rows > max_rows) {
            err = "rows exceed proxy max_rows";
            return false;
        }
        hipError_t herr = ps::kernel::launch_activation_quantize_e4m3(
            normed, act_codes, act_scales, rows, cols, k_padded, normed_row_stride,
            act_code_stride, act_scale_stride, stream);
        herr = herr == hipSuccess
                   ? ps::kernel::launch_dflash2_activation_l2(
                         act_codes, act_scales, rows, k_padded, act_code_stride,
                         act_scale_stride, act_l2, stream)
                   : herr;
        herr = herr == hipSuccess
                   ? ps::kernel::launch_dflash2_int2_coarse_head(
                         int2_codes, weight->scales.data<uint8_t>(), act_codes, act_scales,
                         expand_table(), coarse, rows, vocab, k_padded, scale_stride,
                         act_code_stride, act_scale_stride, vocab, stream)
                   : herr;
        herr = herr == hipSuccess
                   ? ps::kernel::launch_dflash2_target_upper_logits(
                         coarse, act_l2, error_l2, upper, rows, vocab, stream)
                   : herr;
        if (herr != hipSuccess) {
            err = "proxy prepare launch failed";
            return false;
        }
        return true;
    }

    bool topn(uint32_t rows, uint32_t topn_pool, hipStream_t stream, std::string& err) {
        return topn_on(upper, rows, topn_pool, stream, err);
    }

    bool topn_on(const float* logits, uint32_t rows, uint32_t topn_pool, hipStream_t stream,
                 std::string& err) {
        if (topn_pool == 0u || topn_pool > top_stride ||
            topn_pool > ps::kernel::kDflash2Int2MaxPool) {
            err = "topn_pool out of range";
            return false;
        }
        const hipError_t herr = ps::kernel::launch_dflash2_coarse_topn(
            logits, rows, vocab, vocab, topn_pool, top_ids, top_values, scratch_ids,
            scratch_values, scratch_stride, partitions, stream);
        if (herr != hipSuccess) {
            err = "proxy topn launch failed";
            return false;
        }
        return true;
    }

    bool rerank_pool(uint32_t rows, uint32_t pool, uint32_t topn_pool, hipStream_t stream,
                     std::string& err) {
        if (pool == 0u || pool > max_pool || topn_pool == 0u || topn_pool > top_stride) {
            err = "rerank pool out of range";
            return false;
        }
        const uint32_t total = rows * pool;
        const uint32_t threads = 128u;
        const uint32_t blocks = (total + threads - 1u) / threads;
        gather_candidates_kernel<<<blocks, threads, 0, stream>>>(
            top_ids, cand_ids, total, pool, topn_pool);
        hipError_t herr = hipGetLastError();
        herr = herr == hipSuccess
                   ? ps::kernel::launch_dflash2_psq8_candidate_rerank(
                         weight->codes.data<uint8_t>(), weight->scales.data<uint8_t>(),
                         act_codes, act_scales, cand_ids, rerank, rows, pool, k_padded,
                         scale_stride, act_code_stride, act_scale_stride, pool, stream)
                   : herr;
        if (herr != hipSuccess) {
            err = "proxy rerank launch failed";
            return false;
        }
        return true;
    }

    bool certify(uint32_t rows, uint32_t pool, uint32_t topn_pool, float abs_margin,
                 float rel_margin, hipStream_t stream, std::string& err) {
        if (pool == 0u || pool > max_pool || topn_pool == 0u || topn_pool > top_stride) {
            err = "certify pool out of range";
            return false;
        }
        const hipError_t herr = ps::kernel::launch_dflash2_target_certified_argmax(
            rerank, cand_ids, top_values + pool, topn_pool, abs_margin, rel_margin, best_ids,
            best_logits, gaps, certified, rows, pool, stream);
        if (herr != hipSuccess) {
            err = "proxy certify launch failed";
            return false;
        }
        return true;
    }

    bool verify_direct(uint32_t rows, const int32_t* proposed, float abs_margin,
                       float rel_margin, hipStream_t stream, std::string& err) {
        if (rows == 0u || rows > max_rows) {
            err = "verify rows out of range";
            return false;
        }
        hipError_t herr = hipMemcpy(proposed_dev, proposed, rows * sizeof(int32_t),
                                    hipMemcpyHostToDevice);
        herr = herr == hipSuccess
                   ? ps::kernel::launch_dflash2_psq8_selected_token_logits(
                         weight->codes.data<uint8_t>(), weight->scales.data<uint8_t>(),
                         act_codes, act_scales, proposed_dev, exact_proposed, rows, k_padded,
                         scale_stride, act_code_stride, act_scale_stride, stream)
                   : herr;
        herr = herr == hipSuccess
                   ? ps::kernel::launch_dflash2_verify_max_other_upper(
                         coarse, act_l2, error_l2, proposed_dev, max_other_upper, rows, vocab,
                         stream)
                   : herr;
        herr = herr == hipSuccess
                   ? ps::kernel::launch_dflash2_verify_direct_certificate(
                         exact_proposed, max_other_upper, abs_margin, rel_margin, certified,
                         rows, stream)
                   : herr;
        if (herr != hipSuccess) {
            err = "proxy verify direct launch failed";
            return false;
        }
        return true;
    }
};

}  // namespace target_lm_proxy
