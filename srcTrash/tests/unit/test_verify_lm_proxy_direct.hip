#include "support/dflash2_int2_test_common.h"

#include <phaseshift/models/qwen35/kernels/dflash2/draft_head_int2.h>
#include <phaseshift/quantization/fpx/e4m3.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using dflash2_int2_test::Rng;
using ps::kernel::Dflash2Int2Codebook;

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

const std::vector<uint8_t> kExactCodes = {0x00u, 0x30u, 0x38u, 0x3Cu};
const std::vector<uint8_t> kPerturbCodes = {0x00u, 0x30u, 0x34u, 0x38u, 0x3Cu, 0xB0u, 0xB8u};

Dflash2Int2Codebook fixed_codebook() {
    Dflash2Int2Codebook cb;
    cb.values = {0x00u, 0x30u, 0x38u, 0x3Cu};
    for (uint32_t i = 0u; i < 4u; ++i)
        cb.decoded[i] = ps::quantization::fpx::e4m3_decode_f32(cb.values[i]);
    return cb;
}

float bf16_to_float(uint16_t bits) {
    const uint32_t u = static_cast<uint32_t>(bits) << 16;
    return *reinterpret_cast<const float*>(&u);
}

std::size_t psq8_bytes(uint32_t rows, uint32_t k_padded) {
    return ((static_cast<std::size_t>(rows) + 15u) / 16u) *
           static_cast<std::size_t>(k_padded) * 16u;
}

uint32_t align16(uint32_t v) { return (v + 15u) & ~15u; }

void run_chain(const char* tag, uint32_t vocab, uint32_t k_padded, uint32_t rows, uint32_t seed,
               bool exact_mode) {
    const Dflash2Int2Codebook codebook = fixed_codebook();
    std::vector<uint8_t> map_table(256u, 0u);
    std::vector<uint32_t> expand_table(256u, 0u);
    ps::kernel::dflash2_int2_build_tables(codebook, map_table.data(), expand_table.data());
    float error_table[256];
    ps::kernel::dflash2_int2_build_error_table(codebook, map_table.data(), error_table);

    const std::vector<uint8_t>& codes = exact_mode ? kExactCodes : kPerturbCodes;
    const uint32_t weight_scale_stride = k_padded;
    const uint32_t act_code_stride = k_padded;
    const uint32_t act_scale_stride = align16((k_padded / 32u) * 4u);
    const std::size_t scale_count =
        ((static_cast<std::size_t>(vocab) + 15u) / 16u) * (k_padded / 2u);

    Rng rng;
    rng.state = seed;

    std::vector<uint8_t> w_codes(psq8_bytes(vocab, k_padded), 0u);
    std::vector<uint16_t> w_scales(scale_count, 0u);
    for (uint32_t v = 0u; v < vocab; ++v) {
        for (uint32_t k = 0u; k < k_padded; ++k)
            w_codes[dflash2_int2_test::psq8_code_index(v, k, k_padded)] =
                codes[rng.below(static_cast<uint32_t>(codes.size()))];
        for (uint32_t ib = 0u; ib < k_padded / 32u; ++ib)
            w_scales[dflash2_int2_test::psq8_scale_index(v, ib * 32u, k_padded)] =
                dflash2_int2_test::bf16_bits(0.01f + 0.9f * rng.unit());
    }

    std::vector<uint8_t> a_codes(static_cast<std::size_t>(act_code_stride) *
                                     ((rows + 15u) / 16u) * 16u,
                                 0u);
    std::vector<float> a_scales(static_cast<std::size_t>(act_scale_stride / 4u) * rows, 0.0f);
    for (uint32_t m = 0u; m < rows; ++m) {
        for (uint32_t k = 0u; k < k_padded; ++k)
            a_codes[dflash2_int2_test::act_code_index(m, k, act_code_stride)] =
                kPerturbCodes[rng.below(static_cast<uint32_t>(kPerturbCodes.size()))];
        a_scales[static_cast<std::size_t>(m) * (act_scale_stride / 4u)] = 0.3f + rng.unit();
    }

    const auto weight_exact = [&](uint32_t v, uint32_t k) {
        const uint8_t c = w_codes[dflash2_int2_test::psq8_code_index(v, k, k_padded)];
        const uint16_t sb =
            w_scales[dflash2_int2_test::psq8_scale_index(v, (k >> 5u) * 32u, k_padded)];
        return static_cast<double>(ps::quantization::fpx::e4m3_decode_f32(c)) *
               static_cast<double>(bf16_to_float(sb));
    };
    const auto weight_proxy = [&](uint32_t v, uint32_t k) {
        const uint8_t c = w_codes[dflash2_int2_test::psq8_code_index(v, k, k_padded)];
        const uint16_t sb =
            w_scales[dflash2_int2_test::psq8_scale_index(v, (k >> 5u) * 32u, k_padded)];
        return static_cast<double>(codebook.decoded[map_table[c]]) *
               static_cast<double>(bf16_to_float(sb));
    };
    const auto activation = [&](uint32_t m, uint32_t k) {
        const uint8_t c = a_codes[dflash2_int2_test::act_code_index(m, k, act_code_stride)];
        return static_cast<double>(ps::quantization::fpx::e4m3_decode_f32(c)) *
               static_cast<double>(a_scales[static_cast<std::size_t>(m) * (act_scale_stride / 4u)]);
    };

    std::vector<std::vector<double>> exact(rows, std::vector<double>(vocab, 0.0));
    std::vector<std::vector<double>> coarse(rows, std::vector<double>(vocab, 0.0));
    std::vector<double> al2(rows, 0.0);
    std::vector<double> err(vocab, 0.0);
    std::vector<double> a2(rows, 0.0);
    std::vector<double> e2(vocab, 0.0);
    for (uint32_t m = 0u; m < rows; ++m) {
        for (uint32_t k = 0u; k < k_padded; ++k) {
            const double a = activation(m, k);
            a2[m] += a * a;
        }
        al2[m] = std::sqrt(a2[m]);
    }
    for (uint32_t v = 0u; v < vocab; ++v) {
        for (uint32_t k = 0u; k < k_padded; ++k) {
            const uint8_t c = w_codes[dflash2_int2_test::psq8_code_index(v, k, k_padded)];
            const uint16_t sb =
                w_scales[dflash2_int2_test::psq8_scale_index(v, (k >> 5u) * 32u, k_padded)];
            const double d = static_cast<double>(error_table[c]) *
                             static_cast<double>(bf16_to_float(sb));
            e2[v] += d * d;
        }
        err[v] = std::sqrt(e2[v]);
    }
    for (uint32_t m = 0u; m < rows; ++m) {
        for (uint32_t v = 0u; v < vocab; ++v) {
            double ze = 0.0;
            double zp = 0.0;
            for (uint32_t k = 0u; k < k_padded; ++k) {
                const double a = activation(m, k);
                ze += a * weight_exact(v, k);
                zp += a * weight_proxy(v, k);
            }
            exact[m][v] = ze;
            coarse[m][v] = zp;
        }
    }

    std::vector<int32_t> proposed(rows, 0);
    std::vector<int32_t> oracle_argmax(rows, 0);
    for (uint32_t m = 0u; m < rows; ++m) {
        int32_t best = 0;
        for (uint32_t v = 1u; v < vocab; ++v)
            if (exact[m][v] > exact[m][best]) best = static_cast<int32_t>(v);
        oracle_argmax[m] = best;
        if (exact_mode && (m & 1u) == 0u) {
            proposed[m] = best;
        } else if (exact_mode) {
            proposed[m] = static_cast<int32_t>(rng.below(vocab));
        } else {
            proposed[m] = (m & 1u) == 0u ? best : static_cast<int32_t>(rng.below(vocab));
        }
        if (proposed[m] == best && (m & 1u) == 1u && exact_mode)
            proposed[m] = static_cast<int32_t>((best + 1) % vocab);
    }

    const float abs_margin = 1e-3f;
    const float rel_margin = 1e-5f;
    std::vector<float> cpu_max_other(rows, 0.0f);
    std::vector<uint8_t> cpu_certified(rows, 0u);
    for (uint32_t m = 0u; m < rows; ++m) {
        double best_other = -std::numeric_limits<double>::infinity();
        for (uint32_t v = 0u; v < vocab; ++v) {
            if (static_cast<int32_t>(v) == proposed[m]) continue;
            best_other = std::max(best_other, coarse[m][v] + al2[m] * err[v]);
        }
        cpu_max_other[m] = static_cast<float>(best_other);
        const double scale = std::max(std::fabs(exact[m][proposed[m]]), std::fabs(best_other));
        cpu_certified[m] =
            exact[m][proposed[m]] > best_other + (abs_margin + rel_margin * scale) ? 1u : 0u;
    }

    uint8_t* dev_w_codes = nullptr;
    uint8_t* dev_w_scales = nullptr;
    float* dev_error_table = nullptr;
    uint8_t* dev_a_codes = nullptr;
    float* dev_a_scales = nullptr;
    float* dev_err = nullptr;
    float* dev_al2 = nullptr;
    float* dev_coarse = nullptr;
    int32_t* dev_proposed = nullptr;
    float* dev_exact = nullptr;
    float* dev_max_other = nullptr;
    uint8_t* dev_certified = nullptr;
    bool ok = true;
    ok = ok && hipMalloc(&dev_w_codes, w_codes.size()) == hipSuccess;
    ok = ok && hipMalloc(&dev_w_scales, w_scales.size() * 2u) == hipSuccess;
    ok = ok && hipMalloc(&dev_error_table, 256u * sizeof(float)) == hipSuccess;
    ok = ok && hipMalloc(&dev_a_codes, a_codes.size()) == hipSuccess;
    ok = ok && hipMalloc(&dev_a_scales, a_scales.size() * sizeof(float)) == hipSuccess;
    ok = ok && hipMalloc(&dev_err, vocab * sizeof(float)) == hipSuccess;
    ok = ok && hipMalloc(&dev_al2, rows * sizeof(float)) == hipSuccess;
    ok = ok && hipMalloc(&dev_coarse, static_cast<std::size_t>(rows) * vocab * sizeof(float)) ==
                   hipSuccess;
    ok = ok && hipMalloc(&dev_proposed, rows * sizeof(int32_t)) == hipSuccess;
    ok = ok && hipMalloc(&dev_exact, rows * sizeof(float)) == hipSuccess;
    ok = ok && hipMalloc(&dev_max_other, rows * sizeof(float)) == hipSuccess;
    ok = ok && hipMalloc(&dev_certified, rows) == hipSuccess;
    check(ok, tag);
    if (!ok) {
        std::printf("FAIL: %s allocation\n", tag);
        return;
    }

    (void)hipMemcpy(dev_w_codes, w_codes.data(), w_codes.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_w_scales, w_scales.data(), w_scales.size() * 2u, hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_error_table, error_table, 256u * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_a_codes, a_codes.data(), a_codes.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_a_scales, a_scales.data(), a_scales.size() * sizeof(float),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_proposed, proposed.data(), rows * sizeof(int32_t), hipMemcpyHostToDevice);
    std::vector<float> coarse_flat(static_cast<std::size_t>(rows) * vocab);
    for (uint32_t m = 0u; m < rows; ++m)
        for (uint32_t v = 0u; v < vocab; ++v)
            coarse_flat[static_cast<std::size_t>(m) * vocab + v] = static_cast<float>(coarse[m][v]);
    (void)hipMemcpy(dev_coarse, coarse_flat.data(), coarse_flat.size() * sizeof(float),
                    hipMemcpyHostToDevice);

    hipError_t herr = ps::kernel::launch_dflash2_int2_row_error_l2(
        dev_w_codes, dev_w_scales, dev_error_table, vocab, k_padded, weight_scale_stride,
        dev_err, nullptr);
    check(herr == hipSuccess, tag);
    herr = ps::kernel::launch_dflash2_activation_l2(dev_a_codes, dev_a_scales, rows, k_padded,
                                                    act_code_stride, act_scale_stride, dev_al2,
                                                    nullptr);
    check(herr == hipSuccess, tag);
    herr = ps::kernel::launch_dflash2_psq8_selected_token_logits(
        dev_w_codes, dev_w_scales, dev_a_codes, dev_a_scales, dev_proposed, dev_exact, rows,
        k_padded, weight_scale_stride, act_code_stride, act_scale_stride, nullptr);
    check(herr == hipSuccess, tag);
    herr = ps::kernel::launch_dflash2_verify_max_other_upper(dev_coarse, dev_al2, dev_err,
                                                             dev_proposed, dev_max_other, rows,
                                                             vocab, nullptr);
    check(herr == hipSuccess, tag);
    herr = ps::kernel::launch_dflash2_verify_direct_certificate(
        dev_exact, dev_max_other, abs_margin, rel_margin, dev_certified, rows, nullptr);
    check(herr == hipSuccess, tag);
    (void)hipDeviceSynchronize();

    std::vector<float> gpu_err(vocab), gpu_al2(rows), gpu_exact(rows), gpu_max_other(rows);
    std::vector<uint8_t> gpu_certified(rows);
    (void)hipMemcpy(gpu_err.data(), dev_err, vocab * sizeof(float), hipMemcpyDeviceToHost);
    (void)hipMemcpy(gpu_al2.data(), dev_al2, rows * sizeof(float), hipMemcpyDeviceToHost);
    (void)hipMemcpy(gpu_exact.data(), dev_exact, rows * sizeof(float), hipMemcpyDeviceToHost);
    (void)hipMemcpy(gpu_max_other.data(), dev_max_other, rows * sizeof(float),
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(gpu_certified.data(), dev_certified, rows, hipMemcpyDeviceToHost);

    bool err_match = true;
    bool al2_match = true;
    bool exact_match = true;
    bool other_match = true;
    bool certify_match = true;
    uint32_t false_certificates = 0u;
    uint32_t certified_rows = 0u;
    for (uint32_t v = 0u; v < vocab; ++v) {
        if (std::fabs(gpu_err[v] - err[v]) > 1e-4 * (1.0 + err[v])) err_match = false;
    }
    for (uint32_t m = 0u; m < rows; ++m) {
        if (std::fabs(gpu_al2[m] - al2[m]) > 1e-4 * (1.0 + al2[m])) al2_match = false;
        const double want_exact = exact[m][proposed[m]];
        if (std::fabs(gpu_exact[m] - want_exact) > 1e-3 * (1.0 + std::fabs(want_exact)))
            exact_match = false;
        if (std::fabs(gpu_max_other[m] - cpu_max_other[m]) >
            1e-3 * (1.0 + std::fabs(cpu_max_other[m])))
            other_match = false;
        if (gpu_certified[m] != cpu_certified[m]) {
            const double boundary =
                std::fabs(static_cast<double>(gpu_exact[m]) - gpu_max_other[m] -
                          (abs_margin + rel_margin *
                                          std::max(std::fabs(gpu_exact[m]),
                                                   std::fabs(gpu_max_other[m]))));
            if (boundary > 1e-3) certify_match = false;
        }
        if (gpu_certified[m] != 0u) {
            ++certified_rows;
            if (oracle_argmax[m] != proposed[m]) ++false_certificates;
        }
    }
    check(err_match, tag);
    check(al2_match, tag);
    check(exact_match, tag);
    check(other_match, tag);
    check(certify_match, tag);
    check(false_certificates == 0u, tag);
    std::printf(
        "VERIFY_DIRECT tag=%s rows=%u vocab=%u k=%u mode=%s certified=%u false_cert=%u "
        "err=%d al2=%d exact=%d other=%d cert=%d\n",
        tag, rows, vocab, k_padded, exact_mode ? "exact" : "perturb", certified_rows,
        false_certificates, err_match ? 1 : 0, al2_match ? 1 : 0, exact_match ? 1 : 0,
        other_match ? 1 : 0, certify_match ? 1 : 0);

    (void)hipFree(dev_w_codes);
    (void)hipFree(dev_w_scales);
    (void)hipFree(dev_error_table);
    (void)hipFree(dev_a_codes);
    (void)hipFree(dev_a_scales);
    (void)hipFree(dev_err);
    (void)hipFree(dev_al2);
    (void)hipFree(dev_coarse);
    (void)hipFree(dev_proposed);
    (void)hipFree(dev_exact);
    (void)hipFree(dev_max_other);
    (void)hipFree(dev_certified);
}

void run_decision_case(const char* label, const std::vector<int32_t>& drafts,
                       const std::vector<uint8_t>& state,
                       const std::vector<int32_t>& certified_tokens, uint32_t num_drafts,
                       uint32_t expect_ready, uint32_t expect_accepted,
                       uint32_t expect_correction_valid, int32_t expect_correction,
                       uint32_t expect_bonus_valid, int32_t expect_bonus,
                       int32_t expect_first_unknown) {
    const uint32_t total = num_drafts + 1u;
    int32_t* dev_drafts = nullptr;
    uint8_t* dev_state = nullptr;
    int32_t* dev_cert = nullptr;
    uint32_t* dev_ready = nullptr;
    uint32_t* dev_accepted = nullptr;
    uint32_t* dev_corr_valid = nullptr;
    int32_t* dev_corr = nullptr;
    uint32_t* dev_bonus_valid = nullptr;
    int32_t* dev_bonus = nullptr;
    int32_t* dev_first = nullptr;
    bool ok = hipMalloc(&dev_drafts, drafts.size() * sizeof(int32_t)) == hipSuccess &&
              hipMalloc(&dev_state, state.size()) == hipSuccess &&
              hipMalloc(&dev_cert, certified_tokens.size() * sizeof(int32_t)) == hipSuccess &&
              hipMalloc(&dev_ready, sizeof(uint32_t)) == hipSuccess &&
              hipMalloc(&dev_accepted, sizeof(uint32_t)) == hipSuccess &&
              hipMalloc(&dev_corr_valid, sizeof(uint32_t)) == hipSuccess &&
              hipMalloc(&dev_corr, sizeof(int32_t)) == hipSuccess &&
              hipMalloc(&dev_bonus_valid, sizeof(uint32_t)) == hipSuccess &&
              hipMalloc(&dev_bonus, sizeof(int32_t)) == hipSuccess &&
              hipMalloc(&dev_first, sizeof(int32_t)) == hipSuccess;
    check(ok, label);
    if (!ok) return;
    (void)hipMemcpy(dev_drafts, drafts.data(), drafts.size() * sizeof(int32_t),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_state, state.data(), state.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_cert, certified_tokens.data(),
                    certified_tokens.size() * sizeof(int32_t), hipMemcpyHostToDevice);
    const hipError_t err = ps::kernel::launch_dflash2_verify_decision_sufficiency(
        dev_drafts, dev_state, dev_cert, num_drafts, dev_ready, dev_accepted, dev_corr_valid,
        dev_corr, dev_bonus_valid, dev_bonus, dev_first, nullptr);
    check(err == hipSuccess, label);
    uint32_t ready = 0u, accepted = 0u, corr_valid = 0u, bonus_valid = 0u;
    int32_t corr = 0, bonus = 0, first = 0;
    (void)hipMemcpy(&ready, dev_ready, sizeof(uint32_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(&accepted, dev_accepted, sizeof(uint32_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(&corr_valid, dev_corr_valid, sizeof(uint32_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(&corr, dev_corr, sizeof(int32_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(&bonus_valid, dev_bonus_valid, sizeof(uint32_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(&bonus, dev_bonus, sizeof(int32_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(&first, dev_first, sizeof(int32_t), hipMemcpyDeviceToHost);
    check(ready == expect_ready, label);
    check(accepted == expect_accepted, label);
    check(corr_valid == expect_correction_valid, label);
    check(corr == expect_correction, label);
    check(bonus_valid == expect_bonus_valid, label);
    check(bonus == expect_bonus, label);
    check(first == expect_first_unknown, label);
    (void)total;
    (void)hipFree(dev_drafts);
    (void)hipFree(dev_state);
    (void)hipFree(dev_cert);
    (void)hipFree(dev_ready);
    (void)hipFree(dev_accepted);
    (void)hipFree(dev_corr_valid);
    (void)hipFree(dev_corr);
    (void)hipFree(dev_bonus_valid);
    (void)hipFree(dev_bonus);
    (void)hipFree(dev_first);
}

void run_decision_cases() {
    using ps::kernel::kDflash2VerifyRowAcceptCertified;
    using ps::kernel::kDflash2VerifyRowArgmaxCertified;
    using ps::kernel::kDflash2VerifyRowUnknown;

    run_decision_case("decision full accept", {10, 11, 12},
                      {kDflash2VerifyRowAcceptCertified, kDflash2VerifyRowAcceptCertified,
                       kDflash2VerifyRowAcceptCertified, kDflash2VerifyRowArgmaxCertified},
                      {0, 0, 0, 99}, 3u, 1u, 3u, 0u, -1, 1u, 99, -1);
    run_decision_case("decision mismatch at 1", {10, 11, 12},
                      {kDflash2VerifyRowAcceptCertified, kDflash2VerifyRowArgmaxCertified,
                       kDflash2VerifyRowUnknown, kDflash2VerifyRowUnknown},
                      {0, 77, 0, 0}, 3u, 1u, 1u, 1u, 77, 0u, -1, -1);
    run_decision_case("decision argmax match continues", {10, 11, 12},
                      {kDflash2VerifyRowAcceptCertified, kDflash2VerifyRowArgmaxCertified,
                       kDflash2VerifyRowArgmaxCertified, kDflash2VerifyRowArgmaxCertified},
                      {0, 11, 12, 55}, 3u, 1u, 3u, 0u, -1, 1u, 55, -1);
    run_decision_case("decision unknown at 0", {10, 11, 12},
                      {kDflash2VerifyRowUnknown, kDflash2VerifyRowAcceptCertified,
                       kDflash2VerifyRowAcceptCertified, kDflash2VerifyRowArgmaxCertified},
                      {0, 0, 0, 99}, 3u, 0u, 0u, 0u, -1, 0u, -1, 0);
    run_decision_case("decision bonus unknown", {10, 11, 12},
                      {kDflash2VerifyRowAcceptCertified, kDflash2VerifyRowAcceptCertified,
                       kDflash2VerifyRowAcceptCertified, kDflash2VerifyRowUnknown},
                      {0, 0, 0, 0}, 3u, 0u, 3u, 0u, -1, 0u, -1, 3);
    run_decision_case("decision mismatch at 0", {10, 11, 12},
                      {kDflash2VerifyRowArgmaxCertified, kDflash2VerifyRowUnknown,
                       kDflash2VerifyRowUnknown, kDflash2VerifyRowUnknown},
                      {42, 0, 0, 0}, 3u, 1u, 0u, 1u, 42, 0u, -1, -1);
    run_decision_case("decision single draft accept", {10},
                      {kDflash2VerifyRowAcceptCertified, kDflash2VerifyRowArgmaxCertified},
                      {0, 7}, 1u, 1u, 1u, 0u, -1, 1u, 7, -1);
}

void run_certificate_cases() {
    struct Case {
        float exact;
        float other;
        float abs_margin;
        float rel_margin;
        uint8_t expect;
    };
    const Case cases[] = {
        {5.0f, 5.0f, 0.0f, 0.0f, 0u},
        {5.001f, 5.0f, 0.0f, 0.0f, 1u},
        {5.001f, 5.0f, 1e-2f, 0.0f, 0u},
        {5.2f, 5.0f, 1e-2f, 0.0f, 1u},
        {1000.0f, 999.95f, 0.0f, 1e-5f, 1u},
        {1000.0f, 999.95f, 0.0f, 1e-4f, 0u},
    };
    const uint32_t rows = static_cast<uint32_t>(sizeof(cases) / sizeof(cases[0]));
    std::vector<float> exact(rows), other(rows);
    std::vector<float> abs_m(rows), rel_m(rows);
    std::vector<uint8_t> expect(rows);
    for (uint32_t i = 0u; i < rows; ++i) {
        exact[i] = cases[i].exact;
        other[i] = cases[i].other;
        abs_m[i] = cases[i].abs_margin;
        rel_m[i] = cases[i].rel_margin;
        expect[i] = cases[i].expect;
    }
    for (uint32_t i = 0u; i < rows; ++i) {
        float* dev_exact = nullptr;
        float* dev_other = nullptr;
        uint8_t* dev_certified = nullptr;
        (void)hipMalloc(&dev_exact, sizeof(float));
        (void)hipMalloc(&dev_other, sizeof(float));
        (void)hipMalloc(&dev_certified, 1u);
        (void)hipMemcpy(dev_exact, &exact[i], sizeof(float), hipMemcpyHostToDevice);
        (void)hipMemcpy(dev_other, &other[i], sizeof(float), hipMemcpyHostToDevice);
        const hipError_t err = ps::kernel::launch_dflash2_verify_direct_certificate(
            dev_exact, dev_other, abs_m[i], rel_m[i], dev_certified, 1u, nullptr);
        uint8_t got = 0u;
        (void)hipMemcpy(&got, dev_certified, 1u, hipMemcpyDeviceToHost);
        check(err == hipSuccess, "certificate launch");
        char label[64];
        std::snprintf(label, sizeof(label), "certificate case %u", i);
        check(got == expect[i], label);
        (void)hipFree(dev_exact);
        (void)hipFree(dev_other);
        (void)hipFree(dev_certified);
    }
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

    run_chain("exact-8x48x64", 48u, 64u, 8u, 0x1234abcdu, true);
    run_chain("perturb-8x48x64", 48u, 64u, 8u, 0x51ab37c1u, false);
    run_chain("perturb-4x96x128", 96u, 128u, 4u, 0x0f1e2d3cu, false);
    run_chain("perturb-2x33x32", 33u, 32u, 2u, 0x77777777u, false);
    run_chain("exact-7x64x128", 64u, 128u, 7u, 0xdeadbee1u, true);

    run_certificate_cases();

    run_decision_cases();

    std::printf("test_verify_lm_proxy_direct: passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
