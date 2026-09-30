#include "support/dflash2_int2_test_common.h"

#include <phaseshift/models/qwen35/kernels/dflash2/topk.h>
#include <phaseshift/models/qwen35/kernels/optimized/constraint_candidates.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq8.h>
#include <phaseshift/quantization/fpx/e4m3.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

using dflash2_int2_test::Rng;

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

std::size_t psq8_bytes(uint32_t rows, uint32_t k_padded) {
    return ((static_cast<std::size_t>(rows) + 15u) / 16u) *
           static_cast<std::size_t>(k_padded) * 16u;
}

uint32_t mask_words_for(uint32_t vocab) {
    return (vocab + 31u) / 32u;
}

void set_allowed(std::vector<uint32_t>& mask, uint32_t token) {
    mask[token >> 5u] |= (1u << (token & 31u));
}

std::vector<uint32_t> build_mask(uint32_t vocab, const std::vector<uint32_t>& allowed) {
    std::vector<uint32_t> mask(mask_words_for(vocab), 0u);
    for (uint32_t token : allowed) {
        if (token < vocab)
            set_allowed(mask, token);
    }
    return mask;
}

std::vector<uint32_t> expected_candidates(const std::vector<uint32_t>& allowed,
                                          uint32_t capacity) {
    std::vector<uint32_t> sorted = allowed;
    std::sort(sorted.begin(), sorted.end());
    std::vector<uint32_t> out;
    const uint32_t n = static_cast<uint32_t>(std::min<std::size_t>(sorted.size(), capacity));
    for (uint32_t i = 0u; i < n; ++i)
        out.push_back(sorted[i]);
    if (out.empty())
        out.push_back(0u);
    const uint32_t fill = out[0];
    while (out.size() < capacity)
        out.push_back(fill);
    return out;
}

bool run_extraction_case(uint32_t vocab, const std::vector<std::vector<uint32_t>>& rows_allowed,
                         const char* label) {
    const uint32_t rows = static_cast<uint32_t>(rows_allowed.size());
    const uint32_t words = mask_words_for(vocab);
    uint32_t capacity = 1u;
    for (const auto& allowed : rows_allowed)
        capacity = std::max<uint32_t>(capacity, static_cast<uint32_t>(allowed.size()));

    std::vector<uint32_t> masks(static_cast<std::size_t>(rows) * words, 0u);
    for (uint32_t r = 0u; r < rows; ++r) {
        const std::vector<uint32_t> row_mask = build_mask(vocab, rows_allowed[r]);
        std::copy(row_mask.begin(), row_mask.end(),
                  masks.begin() + static_cast<std::size_t>(r) * words);
    }

    uint32_t* dev_masks = nullptr;
    int32_t* dev_ids = nullptr;
    const std::size_t mask_bytes = masks.size() * sizeof(uint32_t);
    const std::size_t id_bytes =
        static_cast<std::size_t>(rows) * capacity * sizeof(int32_t);
    if (hipMalloc(&dev_masks, mask_bytes) != hipSuccess ||
        hipMalloc(&dev_ids, id_bytes) != hipSuccess) {
        if (dev_masks) (void)hipFree(dev_masks);
        if (dev_ids) (void)hipFree(dev_ids);
        return false;
    }
    (void)hipMemcpy(dev_masks, masks.data(), mask_bytes, hipMemcpyHostToDevice);
    (void)hipMemset(dev_ids, 0xEE, id_bytes);

    const hipError_t err = ps::kernel::launch_constraint_mask_to_candidates(
        dev_masks, rows, words, vocab, capacity, dev_ids, nullptr);
    if (err != hipSuccess) {
        std::printf("FAIL: %s launch err=%s\n", label, hipGetErrorString(err));
        (void)hipFree(dev_masks);
        (void)hipFree(dev_ids);
        return false;
    }
    (void)hipDeviceSynchronize();

    std::vector<int32_t> got(static_cast<std::size_t>(rows) * capacity, 0);
    (void)hipMemcpy(got.data(), dev_ids, id_bytes, hipMemcpyDeviceToHost);
    (void)hipFree(dev_masks);
    (void)hipFree(dev_ids);

    bool exact = true;
    for (uint32_t r = 0u; r < rows && exact; ++r) {
        const std::vector<uint32_t> want =
            expected_candidates(rows_allowed[r], capacity);
        for (uint32_t c = 0u; c < capacity; ++c) {
            const int32_t g = got[static_cast<std::size_t>(r) * capacity + c];
            if (g != static_cast<int32_t>(want[c])) {
                exact = false;
                std::printf("  detail %s row=%u col=%u got=%d want=%u\n", label, r, c, g,
                            want[c]);
                break;
            }
        }
    }
    check(exact, label);
    return true;
}

struct ExactCaseSpec {
    uint32_t rows = 1u;
    uint32_t vocab = 512u;
    uint32_t k_padded = 128u;
    uint32_t seed = 0x9e3779b9u;
    std::vector<uint32_t> counts;
    bool tie_logits = false;
};

bool run_exact_case(const ExactCaseSpec& spec, const char* label) {
    const uint32_t rows = spec.rows;
    const uint32_t vocab = spec.vocab;
    const uint32_t k_padded = spec.k_padded;
    const uint32_t words = mask_words_for(vocab);
    const auto layout = ps::runtime::Int8ActivationWorkspaceLayout::make(k_padded, rows);
    const uint32_t act_code_stride = layout.code_row_stride_bytes;
    const uint32_t act_scale_stride = layout.scale_row_stride_bytes;
    const uint32_t weight_scale_stride = k_padded;

    Rng rng;
    rng.state = spec.seed;

    std::vector<uint8_t> psq8(psq8_bytes(vocab, k_padded), 0u);
    const std::vector<uint8_t> finite = {0x00u, 0x30u, 0x34u, 0x38u, 0x3Cu, 0xB0u, 0xB8u, 0xBCu};
    for (uint32_t o = 0u; o < vocab; ++o) {
        for (uint32_t k = 0u; k < k_padded; ++k)
            psq8[dflash2_int2_test::psq8_code_index(o, k, k_padded)] =
                spec.tie_logits ? 0x00u
                                : finite[rng.below(static_cast<uint32_t>(finite.size()))];
    }
    const std::size_t scale_count =
        ((static_cast<std::size_t>(vocab) + 15u) / 16u) * (weight_scale_stride / 2u);
    std::vector<uint16_t> scales(scale_count, 0u);
    for (uint32_t o = 0u; o < vocab; ++o) {
        for (uint32_t ib = 0u; ib < k_padded / 32u; ++ib)
            scales[dflash2_int2_test::psq8_scale_index(o, ib * 32u, weight_scale_stride)] =
                dflash2_int2_test::bf16_bits(spec.tie_logits ? 0.05f
                                                             : 0.05f + 0.4f * rng.unit());
    }
    std::vector<uint8_t> act_codes(act_code_stride * ((rows + 15u) / 16u) * 16u, 0u);
    std::vector<float> act_scales(static_cast<std::size_t>(act_scale_stride / 4u) * rows, 0.0f);
    for (uint32_t r = 0u; r < rows; ++r) {
        for (uint32_t k = 0u; k < k_padded; ++k)
            act_codes[dflash2_int2_test::act_code_index(r, k, act_code_stride)] =
                finite[rng.below(static_cast<uint32_t>(finite.size()))];
        act_scales[static_cast<std::size_t>(r) * (act_scale_stride / 4u)] = 0.3f + rng.unit();
    }

    std::vector<std::vector<uint32_t>> rows_allowed;
    rows_allowed.reserve(rows);
    for (uint32_t r = 0u; r < rows; ++r) {
        const uint32_t count =
            spec.counts.empty() ? 8u : spec.counts[r % spec.counts.size()];
        std::vector<uint32_t> allowed;
        if (count == 1u) {
            if (r % 2u == 0u) {
                allowed.push_back(0u);
            } else {
                allowed.push_back(vocab - 1u);
            }
        } else {
            std::vector<uint32_t> all(vocab);
            for (uint32_t i = 0u; i < vocab; ++i)
                all[i] = i;
            for (uint32_t i = vocab; i > 1u; --i) {
                const uint32_t j = rng.below(i);
                const uint32_t t = all[i - 1u];
                all[i - 1u] = all[j];
                all[j] = t;
            }
            const uint32_t n = std::min<uint32_t>(count, vocab);
            allowed.assign(all.begin(), all.begin() + n);
            std::sort(allowed.begin(), allowed.end());
        }
        rows_allowed.push_back(allowed);
    }

    uint32_t capacity = 1u;
    for (const auto& allowed : rows_allowed)
        capacity = std::max<uint32_t>(capacity, static_cast<uint32_t>(allowed.size()));

    std::vector<uint32_t> masks(static_cast<std::size_t>(rows) * words, 0u);
    for (uint32_t r = 0u; r < rows; ++r) {
        const std::vector<uint32_t> row_mask = build_mask(vocab, rows_allowed[r]);
        std::copy(row_mask.begin(), row_mask.end(),
                  masks.begin() + static_cast<std::size_t>(r) * words);
    }

    uint8_t* dev_psq8 = nullptr;
    uint8_t* dev_scale_bytes = nullptr;
    uint8_t* dev_act = nullptr;
    float* dev_act_scales = nullptr;
    uint32_t* dev_masks = nullptr;
    int32_t* dev_ids = nullptr;
    float* dev_full = nullptr;
    float* dev_rerank = nullptr;
    const std::size_t full_elems = static_cast<std::size_t>(rows) * vocab;
    const std::size_t cand_elems = static_cast<std::size_t>(rows) * capacity;
    bool ok = true;
    ok = ok && hipMalloc(&dev_psq8, psq8.size()) == hipSuccess;
    ok = ok && hipMalloc(&dev_scale_bytes, scale_count * 2u) == hipSuccess;
    ok = ok && hipMalloc(&dev_act, act_codes.size()) == hipSuccess;
    ok = ok && hipMalloc(&dev_act_scales, act_scales.size() * sizeof(float)) == hipSuccess;
    ok = ok && hipMalloc(&dev_masks, masks.size() * sizeof(uint32_t)) == hipSuccess;
    ok = ok && hipMalloc(&dev_ids, cand_elems * sizeof(int32_t)) == hipSuccess;
    ok = ok && hipMalloc(&dev_full, full_elems * sizeof(float)) == hipSuccess;
    ok = ok && hipMalloc(&dev_rerank, cand_elems * sizeof(float)) == hipSuccess;
    if (!ok) {
        for (void* p : {reinterpret_cast<void*>(dev_psq8),
                        reinterpret_cast<void*>(dev_scale_bytes),
                        reinterpret_cast<void*>(dev_act),
                        reinterpret_cast<void*>(dev_act_scales),
                        reinterpret_cast<void*>(dev_masks), reinterpret_cast<void*>(dev_ids),
                        reinterpret_cast<void*>(dev_full), reinterpret_cast<void*>(dev_rerank)}) {
            if (p) (void)hipFree(p);
        }
        return false;
    }
    (void)hipMemcpy(dev_psq8, psq8.data(), psq8.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_scale_bytes, scales.data(), scale_count * 2u, hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_act, act_codes.data(), act_codes.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_act_scales, act_scales.data(), act_scales.size() * sizeof(float),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_masks, masks.data(), masks.size() * sizeof(uint32_t),
                    hipMemcpyHostToDevice);

    ps::kernel::Psq8GemmConfig cfg;
    cfg.id = ps::kernel::Psq8GemmConfigId::RowBlock1;
    hipError_t err = ps::kernel::launch_gemm_psq8_w8a8_wmma(
        cfg, dev_psq8, dev_scale_bytes, reinterpret_cast<const int8_t*>(dev_act),
        dev_act_scales, dev_full, ps::kernel::Psq8GemmOutputDType::F32, rows, vocab, k_padded,
        weight_scale_stride, act_code_stride, act_scale_stride, vocab, nullptr);
    if (err == hipSuccess) {
        err = ps::kernel::launch_constraint_mask_to_candidates(
            dev_masks, rows, words, vocab, capacity, dev_ids, nullptr);
    }
    if (err == hipSuccess) {
        err = ps::kernel::launch_dflash2_psq8_candidate_rerank(
            dev_psq8, dev_scale_bytes, dev_act, dev_act_scales, dev_ids, dev_rerank, rows,
            capacity, k_padded, weight_scale_stride, act_code_stride, act_scale_stride,
            capacity, nullptr);
    }
    if (err != hipSuccess) {
        std::printf("FAIL: %s launch err=%s\n", label, hipGetErrorString(err));
        ok = false;
    }
    (void)hipDeviceSynchronize();

    std::vector<float> full(full_elems, 0.0f);
    std::vector<int32_t> ids(cand_elems, 0);
    std::vector<float> rerank(cand_elems, 0.0f);
    if (ok) {
        (void)hipMemcpy(full.data(), dev_full, full_elems * sizeof(float),
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(ids.data(), dev_ids, cand_elems * sizeof(int32_t),
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(rerank.data(), dev_rerank, cand_elems * sizeof(float),
                        hipMemcpyDeviceToHost);
    }
    (void)hipFree(dev_psq8);
    (void)hipFree(dev_scale_bytes);
    (void)hipFree(dev_act);
    (void)hipFree(dev_act_scales);
    (void)hipFree(dev_masks);
    (void)hipFree(dev_ids);
    (void)hipFree(dev_full);
    (void)hipFree(dev_rerank);
    if (!ok)
        return false;

    bool match = true;
    bool candidates_allowed = true;
    for (uint32_t r = 0u; r < rows; ++r) {
        const std::vector<uint32_t>& allowed = rows_allowed[r];
        const float* row_full = full.data() + static_cast<std::size_t>(r) * vocab;
        int32_t want_token = -1;
        float want_value = -INFINITY;
        for (uint32_t token = 0u; token < vocab; ++token) {
            const bool is_allowed =
                std::binary_search(allowed.begin(), allowed.end(), token);
            if (!is_allowed)
                continue;
            const float value = row_full[token];
            if (want_token < 0 || value > want_value ||
                (value == want_value && token < static_cast<uint32_t>(want_token))) {
                want_value = value;
                want_token = static_cast<int32_t>(token);
            }
        }
        int32_t got_token = -1;
        float got_value = -INFINITY;
        for (uint32_t c = 0u; c < capacity; ++c) {
            const int32_t token = ids[static_cast<std::size_t>(r) * capacity + c];
            if (token < 0 || static_cast<uint32_t>(token) >= vocab) {
                candidates_allowed = false;
                continue;
            }
            if (!std::binary_search(allowed.begin(), allowed.end(),
                                    static_cast<uint32_t>(token)))
                candidates_allowed = false;
            const float value = rerank[static_cast<std::size_t>(r) * capacity + c];
            if (got_token < 0 || value > got_value ||
                (value == got_value && token < got_token)) {
                got_value = value;
                got_token = token;
            }
        }
        if (got_token != want_token) {
            match = false;
            std::printf("  detail %s row=%u got=%d want=%d\n", label, r, got_token,
                        want_token);
        }
    }
    check(match, label);
    check(candidates_allowed, "candidates contain only allowed tokens");
    return true;
}

struct MaskedTopnSpec {
    uint32_t vocab = 1024u;
    uint32_t pool = 32u;
    uint32_t allowed_count = 64u;
    uint32_t seed = 0xabcdef01u;
    bool use_radix = true;
};

bool run_masked_topn_case(const MaskedTopnSpec& spec, const char* label) {
    const uint32_t rows = 1u;
    const uint32_t vocab = spec.vocab;
    const uint32_t pool = spec.pool;
    const uint32_t words = mask_words_for(vocab);

    Rng rng;
    rng.state = spec.seed;
    std::vector<float> logits(vocab);
    for (uint32_t i = 0u; i < vocab; ++i)
        logits[i] = rng.sym() * 6.0f;
    for (uint32_t i = 0u; i < vocab; i += 97u)
        logits[i] = logits[i > 0u ? i - 1u : 0u];

    std::vector<uint32_t> all(vocab);
    for (uint32_t i = 0u; i < vocab; ++i)
        all[i] = i;
    for (uint32_t i = vocab; i > 1u; --i) {
        const uint32_t j = rng.below(i);
        const uint32_t t = all[i - 1u];
        all[i - 1u] = all[j];
        all[j] = t;
    }
    const uint32_t allowed_n = std::min<uint32_t>(spec.allowed_count, vocab);
    std::vector<uint32_t> allowed(all.begin(), all.begin() + allowed_n);
    std::sort(allowed.begin(), allowed.end());
    const std::vector<uint32_t> mask = build_mask(vocab, allowed);

    float* dev_logits = nullptr;
    uint32_t* dev_mask = nullptr;
    int32_t* dev_ids = nullptr;
    float* dev_values = nullptr;
    void* dev_scratch = nullptr;
    const uint32_t partitions = ps::kernel::dflash2_radix_default_partitions(vocab);
    const std::size_t radix_bytes = ps::kernel::dflash2_radix_topn_scratch_bytes(
        rows, partitions, vocab);
    const uint32_t legacy_partitions = ps::kernel::dflash2_int2_default_partitions(vocab);
    const uint32_t legacy_stride =
        legacy_partitions * ps::kernel::kDflash2Int2MaxPool;
    const std::size_t scratch_bytes =
        spec.use_radix ? radix_bytes
                       : static_cast<std::size_t>(legacy_stride) *
                             (sizeof(int32_t) + sizeof(float));
    bool ok = true;
    ok = ok && hipMalloc(&dev_logits, vocab * sizeof(float)) == hipSuccess;
    ok = ok && hipMalloc(&dev_mask, words * sizeof(uint32_t)) == hipSuccess;
    ok = ok && hipMalloc(&dev_ids, pool * sizeof(int32_t)) == hipSuccess;
    ok = ok && hipMalloc(&dev_values, pool * sizeof(float)) == hipSuccess;
    ok = ok && hipMalloc(&dev_scratch, scratch_bytes) == hipSuccess;
    if (!ok) {
        if (dev_logits) (void)hipFree(dev_logits);
        if (dev_mask) (void)hipFree(dev_mask);
        if (dev_ids) (void)hipFree(dev_ids);
        if (dev_values) (void)hipFree(dev_values);
        if (dev_scratch) (void)hipFree(dev_scratch);
        return false;
    }
    (void)hipMemcpy(dev_logits, logits.data(), vocab * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_mask, mask.data(), words * sizeof(uint32_t), hipMemcpyHostToDevice);
    (void)hipMemset(dev_ids, 0, pool * sizeof(int32_t));
    (void)hipMemset(dev_values, 0, pool * sizeof(float));
    (void)hipMemset(dev_scratch, 0, scratch_bytes);

    hipError_t err = ps::kernel::launch_dflash2_apply_constraint_mask(
        dev_logits, rows, vocab, vocab, dev_mask, words, nullptr);
    if (err == hipSuccess) {
        if (spec.use_radix) {
            err = ps::kernel::launch_dflash2_radix_topn(
                dev_logits, rows, vocab, vocab, pool, dev_ids, dev_values, partitions,
                dev_scratch, radix_bytes, nullptr);
        } else {
            float* scratch_values = static_cast<float*>(dev_scratch);
            int32_t* scratch_ids = reinterpret_cast<int32_t*>(
                static_cast<std::byte*>(dev_scratch) +
                static_cast<std::size_t>(legacy_stride) * sizeof(float));
            err = ps::kernel::launch_dflash2_coarse_topn(
                dev_logits, rows, vocab, vocab, pool, dev_ids, dev_values, scratch_ids,
                scratch_values, legacy_stride, legacy_partitions, nullptr);
        }
    }
    if (err != hipSuccess) {
        std::printf("FAIL: %s launch err=%s\n", label, hipGetErrorString(err));
        (void)hipFree(dev_logits);
        (void)hipFree(dev_mask);
        (void)hipFree(dev_ids);
        (void)hipFree(dev_values);
        (void)hipFree(dev_scratch);
        return false;
    }
    (void)hipDeviceSynchronize();

    std::vector<int32_t> ids(pool, 0);
    std::vector<float> values(pool, 0.0f);
    (void)hipMemcpy(ids.data(), dev_ids, pool * sizeof(int32_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(values.data(), dev_values, pool * sizeof(float), hipMemcpyDeviceToHost);
    (void)hipFree(dev_logits);
    (void)hipFree(dev_mask);
    (void)hipFree(dev_ids);
    (void)hipFree(dev_values);
    (void)hipFree(dev_scratch);

    bool containment = true;
    bool ordering = true;
    for (uint32_t c = 0u; c < pool; ++c) {
        const int32_t token = ids[c];
        if (token < 0 || static_cast<uint32_t>(token) >= vocab ||
            !std::binary_search(allowed.begin(), allowed.end(),
                                static_cast<uint32_t>(token))) {
            containment = false;
            std::printf("  detail %s col=%u id=%d not allowed\n", label, c, token);
            break;
        }
        if (c > 0u) {
            const float prev = values[c - 1u];
            const float cur = values[c];
            if (cur > prev ||
                (cur == prev && ids[c - 1u] >= ids[c])) {
                ordering = false;
            }
        }
    }
    if (allowed_n >= pool) {
        check(containment, label);
        check(ordering, "masked topn ordering score desc token asc");
    }
    return true;
}

}  // namespace

int main() {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess)
        return 1;

    {
        std::vector<std::vector<uint32_t>> rows;
        rows.push_back({0u});
        run_extraction_case(70u, rows, "extract token0 only vocab70");

        rows.clear();
        rows.push_back({69u});
        run_extraction_case(70u, rows, "extract vocab tail only vocab70");

        rows.clear();
        rows.push_back({0u, 32u, 69u});
        run_extraction_case(70u, rows, "extract sparse vocab70");

        rows.clear();
        rows.push_back({64u, 65u, 66u, 67u, 68u, 69u});
        run_extraction_case(70u, rows, "extract contiguous tail vocab70");

        const uint32_t counts[] = {1u, 2u, 3u, 7u,  16u, 31u,
                                   32u, 63u, 64u, 96u, 127u, 128u};
        for (uint32_t count : counts) {
            const uint32_t vocab = 1024u;
            std::vector<uint32_t> allowed;
            for (uint32_t i = 0u; i < count; ++i)
                allowed.push_back(i * 3u % vocab);
            std::sort(allowed.begin(), allowed.end());
            allowed.erase(std::unique(allowed.begin(), allowed.end()), allowed.end());
            std::vector<std::vector<uint32_t>> rows(1u, allowed);
            char label[96];
            std::snprintf(label, sizeof(label), "extract allowed=%u vocab=%u",
                          static_cast<uint32_t>(allowed.size()), vocab);
            run_extraction_case(vocab, rows, label);
        }

        {
            std::vector<std::vector<uint32_t>> rows;
            std::vector<uint32_t> a;
            for (uint32_t i = 0u; i < 5u; ++i)
                a.push_back(i);
            std::vector<uint32_t> b;
            for (uint32_t i = 0u; i < 40u; ++i)
                b.push_back(i * 7u);
            std::vector<uint32_t> c;
            for (uint32_t i = 0u; i < 128u; ++i)
                c.push_back(i * 5u % 1024u);
            rows.push_back(a);
            rows.push_back(b);
            rows.push_back(c);
            run_extraction_case(1024u, rows, "extract mixed row counts");
        }

        {
            const uint32_t vocab = 65u;
            std::vector<std::vector<uint32_t>> rows(1u);
            for (uint32_t i = 0u; i < 65u; ++i)
                rows[0].push_back(i);
            run_extraction_case(vocab, rows, "extract full vocab 65");
        }
    }

    {
        ExactCaseSpec spec;
        spec.rows = 1u;
        spec.vocab = 512u;
        spec.k_padded = 128u;
        spec.counts = {1u};
        spec.seed = 0x1234567u;
        run_exact_case(spec, "exact allowed=1 rows=1");
    }
    {
        ExactCaseSpec spec;
        spec.rows = 1u;
        spec.vocab = 512u;
        spec.k_padded = 128u;
        spec.counts = {128u};
        spec.seed = 0x23456789u;
        run_exact_case(spec, "exact allowed=128 rows=1");
    }
    for (uint32_t count : {2u, 7u, 16u, 31u, 32u, 64u, 96u}) {
        ExactCaseSpec spec;
        spec.rows = 4u;
        spec.vocab = 512u;
        spec.k_padded = 128u;
        spec.counts = {count};
        spec.seed = 0x3456789au + count;
        char label[64];
        std::snprintf(label, sizeof(label), "exact allowed=%u rows=4", count);
        run_exact_case(spec, label);
    }
    {
        ExactCaseSpec spec;
        spec.rows = 4u;
        spec.vocab = 512u;
        spec.k_padded = 128u;
        spec.counts = {1u, 5u, 40u, 128u};
        spec.seed = 0x456789abu;
        run_exact_case(spec, "exact mixed row counts rows=4");
    }
    {
        ExactCaseSpec spec;
        spec.rows = 8u;
        spec.vocab = 1024u;
        spec.k_padded = 128u;
        spec.counts = {128u};
        spec.seed = 0x56789abcu;
        run_exact_case(spec, "exact allowed=128 rows=8 vocab=1024");
    }
    {
        ExactCaseSpec spec;
        spec.rows = 4u;
        spec.vocab = 256u;
        spec.k_padded = 64u;
        spec.counts = {1u, 2u, 3u, 64u};
        spec.seed = 0x6789abcdU;
        spec.tie_logits = true;
        run_exact_case(spec, "exact tie logits lower token wins");
    }
    {
        ExactCaseSpec spec;
        spec.rows = 1u;
        spec.vocab = 65u;
        spec.k_padded = 64u;
        spec.counts = {33u};
        spec.seed = 0x789abcdeu;
        run_exact_case(spec, "exact vocab 65 unpadded");
    }

    for (bool use_radix : {true, false}) {
        const char* kind = use_radix ? "radix" : "legacy";
        for (uint32_t pool : {32u, 64u}) {
            for (uint32_t allowed : {32u, 33u, 64u, 128u, 512u}) {
                if (allowed < pool)
                    continue;
                MaskedTopnSpec spec;
                spec.vocab = 1024u;
                spec.pool = pool;
                spec.allowed_count = allowed;
                spec.seed = 0x10203040u + pool * 7u + allowed;
                spec.use_radix = use_radix;
                char label[96];
                std::snprintf(label, sizeof(label), "masked topn %s allowed=%u pool=%u", kind,
                              allowed, pool);
                run_masked_topn_case(spec, label);
            }
        }
    }

    std::printf("test_constraint_candidates: passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
