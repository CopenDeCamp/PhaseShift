#include <phaseshift/models/qwen35/kernels/optimized/attention/paged_attention.h>
#include <phaseshift/quantization/psq/psq4_kv.h>
#include <phaseshift/core/gpu/gfx_arch.h>
#include <phaseshift/models/qwen35/state/kv_cache_types.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/quantization/fpx/e4m3_detail.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static bool g_wmma12 = false;

static hipError_t launch_paged_bf16(
    const ps::kernel::PagedAttentionCommonArgs& c, const ps::bf16_t* k_pool,
    const ps::bf16_t* v_pool, hipStream_t stream) {
    if (g_wmma12)
        return ps::kernel::launch_attention_paged_bf16(c, k_pool, v_pool, stream);
    return ps::kernel::launch_attention_paged_bf16(c, k_pool, v_pool, stream);
}

namespace {

using bf16 = __hip_bfloat16;
using fp8 = ::ps::qwen35::fp8e4m3_storage_t;
using bf16t = ps::bf16_t;

float e4m3_decode_host(uint8_t x) {
    const int sign = (x & 0x80u) ? -1 : 1;
    const int exp = (x >> 3) & 0x0Fu;
    const int man = x & 0x07u;
    const float val = (exp == 0)
        ? (man * 0.125f) * std::ldexp(1.0f, -6)
        : (1.0f + man * 0.125f) * std::ldexp(1.0f, exp - 7);
    return sign * val;
}

int passed = 0;
int failed = 0;
int skipped_cases = 0;

void check(bool cond, const char* name) {
    if (cond) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", name);
    }
}

uint32_t h32(uint64_t x) {
    uint32_t h = static_cast<uint32_t>(x * 2654435761u + 12345u);
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h;
}

uint16_t bf16_bits(double x) {
    return ps::f32_to_bf16_rne(static_cast<float>(x));
}

int32_t rng1k(uint64_t x) {
    return static_cast<int32_t>(h32(x) % 2001u) - 1000;
}

float f32_from_bf16(uint16_t bits) {
    uint32_t ui = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &ui, 4);
    return f;
}

constexpr float kFp8E4M3Max = 448.0f;

struct Config {
    uint32_t q_heads = 8;
    uint32_t kv_heads = 4;
    uint32_t head_dim = 64;
    uint32_t page_tokens = 4;
    uint32_t num_pages = 8;
    uint32_t num_layers = 2;
    uint32_t layer = 1;
    uint32_t block_table_stride = 8;
    uint32_t max_sequences = 4;
    uint32_t visible_override = 0;
    uint32_t rows = 8;
    bool fp8 = false;
    bool prefill = false;
    float scale = 0.125f;
    uint32_t seed = 1;
};

struct Dev {
    void* q = nullptr;
    void* out = nullptr;
    void* row_seq = nullptr;
    void* row_pos = nullptr;
    void* block_tables = nullptr;
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    void* k_scale = nullptr;
    void* v_scale = nullptr;
};

void free_dev(Dev& d) {
    if (d.q) ps::gpu::discard_cleanup_result(hipFree(d.q));
    if (d.out) ps::gpu::discard_cleanup_result(hipFree(d.out));
    if (d.row_seq) ps::gpu::discard_cleanup_result(hipFree(d.row_seq));
    if (d.row_pos) ps::gpu::discard_cleanup_result(hipFree(d.row_pos));
    if (d.block_tables) ps::gpu::discard_cleanup_result(hipFree(d.block_tables));
    if (d.k_pool) ps::gpu::discard_cleanup_result(hipFree(d.k_pool));
    if (d.v_pool) ps::gpu::discard_cleanup_result(hipFree(d.v_pool));
    if (d.k_scale) ps::gpu::discard_cleanup_result(hipFree(d.k_scale));
    if (d.v_scale) ps::gpu::discard_cleanup_result(hipFree(d.v_scale));
    d = Dev{};
}

bool run_case(const Config& cfg, const char* label) {
    const uint32_t q_features = cfg.q_heads * cfg.head_dim;
    const uint32_t ept = cfg.kv_heads * cfg.head_dim;
    const uint32_t epp = cfg.page_tokens * ept;
    const uint32_t epl = cfg.num_pages * epp;
    const uint64_t pool_elems = static_cast<uint64_t>(cfg.num_layers) * epl;
    const uint32_t scale_elems = cfg.num_layers * cfg.num_pages * cfg.page_tokens * cfg.kv_heads;
    const size_t q_elems = static_cast<size_t>(cfg.rows) * q_features;
    const size_t bt_elems = static_cast<size_t>(cfg.max_sequences) * cfg.block_table_stride;
    const uint32_t q_per_kv = cfg.q_heads / cfg.kv_heads;

    std::vector<uint32_t> row_seq(cfg.rows), row_pos(cfg.rows), block_tables(bt_elems, 0);
    for (uint32_t r = 0; r < cfg.rows; ++r) {
        row_seq[r] = h32(cfg.seed * 1000u + r) % cfg.max_sequences;
        const uint32_t lb = r / cfg.page_tokens;
        uint32_t pos = lb * cfg.page_tokens;
        if (cfg.visible_override != 0u) {
            const uint32_t cap = cfg.page_tokens * cfg.num_pages;
            pos = cfg.visible_override > 1u ? cfg.visible_override - 1u : 0u;
            if (pos > cap - 1u) pos = cap - 1u;
        }
        row_pos[r] = pos;
    }
    for (uint32_t slot = 0; slot < cfg.max_sequences; ++slot)
        for (uint32_t lb = 0; lb < cfg.block_table_stride; ++lb)
            block_tables[static_cast<size_t>(slot) * cfg.block_table_stride + lb] = lb % cfg.num_pages;

    std::vector<uint16_t> q(q_elems);
    for (size_t i = 0; i < q_elems; ++i)
        q[i] = bf16_bits((rng1k(cfg.seed * 9000u + i)) / 128.0);

    std::vector<uint8_t> k_pool, v_pool;
    std::vector<float> k_scale, v_scale;
    const uint8_t init_byte = cfg.fp8 ? 0xEE : 0x3F;
    if (cfg.fp8) {
        k_pool.assign(pool_elems, init_byte);
        v_pool.assign(pool_elems, init_byte);
        k_scale.assign(scale_elems, 0.0f);
        v_scale.assign(scale_elems, 0.0f);
        for (uint32_t page = 0; page < cfg.num_pages; ++page) {
            for (uint32_t off = 0; off < cfg.page_tokens; ++off) {
                for (uint32_t h = 0; h < cfg.kv_heads; ++h) {
                    const uint64_t hb = static_cast<uint64_t>(cfg.layer) * epl
                        + static_cast<uint64_t>(page) * epp
                        + static_cast<uint64_t>(off) * ept
                        + static_cast<uint64_t>(h) * cfg.head_dim;
                    const size_t si = (static_cast<size_t>(cfg.layer) * cfg.num_pages + page)
                        * cfg.page_tokens + off;
                    const size_t sih = si * cfg.kv_heads + h;
                    float k_amax = 0.f, v_amax = 0.f;
                    for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                        k_amax = std::fmax(k_amax, std::fabs(f32_from_bf16(
                            bf16_bits((rng1k(cfg.seed * 2000u + hb + d)) / 128.0))));
                        v_amax = std::fmax(v_amax, std::fabs(f32_from_bf16(
                            bf16_bits((rng1k(cfg.seed * 3000u + hb + d)) / 128.0))));
                    }
                    const float ks = k_amax == 0.f ? 1.f : k_amax / kFp8E4M3Max;
                    const float vs = v_amax == 0.f ? 1.f : v_amax / kFp8E4M3Max;
                    k_scale[sih] = ks;
                    v_scale[sih] = vs;
                    for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                        const float kf = f32_from_bf16(bf16_bits((rng1k(cfg.seed * 2000u + hb + d)) / 128.0));
                        const float vf = f32_from_bf16(bf16_bits((rng1k(cfg.seed * 3000u + hb + d)) / 128.0));
                        k_pool[hb + d] = static_cast<uint8_t>(__hip_cvt_float_to_fp8(kf / ks, __HIP_SATFINITE, __HIP_E4M3));
                        v_pool[hb + d] = static_cast<uint8_t>(__hip_cvt_float_to_fp8(vf / vs, __HIP_SATFINITE, __HIP_E4M3));
                    }
                }
            }
        }
    } else {
        k_pool.assign(pool_elems * 2, init_byte);
        v_pool.assign(pool_elems * 2, init_byte);
        for (uint32_t page = 0; page < cfg.num_pages; ++page) {
            for (uint32_t off = 0; off < cfg.page_tokens; ++off) {
                for (uint32_t h = 0; h < cfg.kv_heads; ++h) {
                    const uint64_t hb = static_cast<uint64_t>(cfg.layer) * epl
                        + static_cast<uint64_t>(page) * epp
                        + static_cast<uint64_t>(off) * ept
                        + static_cast<uint64_t>(h) * cfg.head_dim;
                    for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                        const uint16_t kb = bf16_bits((rng1k(cfg.seed * 2000u + hb + d)) / 128.0);
                        const uint16_t vb = bf16_bits((rng1k(cfg.seed * 3000u + hb + d)) / 128.0);
                        k_pool[hb * 2 + d * 2] = kb & 0xFF;
                        k_pool[hb * 2 + d * 2 + 1] = kb >> 8;
                        v_pool[hb * 2 + d * 2] = vb & 0xFF;
                        v_pool[hb * 2 + d * 2 + 1] = vb >> 8;
                    }
                }
            }
        }
    }

    std::vector<float> ref(static_cast<size_t>(cfg.rows) * q_features, 0.0f);
    for (uint32_t r = 0; r < cfg.rows; ++r) {
        const uint32_t slot = row_seq[r];
        const uint32_t visible = row_pos[r] + 1;
        for (uint32_t qh = 0; qh < cfg.q_heads; ++qh) {
            const uint32_t kv_head = qh / q_per_kv;
            std::vector<float> qv(cfg.head_dim);
            for (uint32_t d = 0; d < cfg.head_dim; ++d)
                qv[d] = f32_from_bf16(q[static_cast<size_t>(r) * q_features + qh * cfg.head_dim + d]);
            float max_score = -3.402823466e+38f;
            std::vector<float> scores(visible);
            for (uint32_t p = 0; p < visible; ++p) {
                const uint32_t lb = p / cfg.page_tokens;
                const uint32_t off = p % cfg.page_tokens;
                const uint32_t page = block_tables[static_cast<size_t>(slot) * cfg.block_table_stride + lb];
                const uint64_t hb = static_cast<uint64_t>(cfg.layer) * epl
                    + static_cast<uint64_t>(page) * epp
                    + static_cast<uint64_t>(off) * ept
                    + static_cast<uint64_t>(kv_head) * cfg.head_dim;
                float dot = 0.f;
                if (cfg.fp8) {
                    const size_t si = (static_cast<size_t>(cfg.layer) * cfg.num_pages + page) * cfg.page_tokens + off;
                    const size_t sih = si * cfg.kv_heads + kv_head;
                    const float ks = k_scale[sih];
                    for (uint32_t d = 0; d < cfg.head_dim; ++d)
                        dot += qv[d] * e4m3_decode_host(k_pool[hb + d]);
                    scores[p] = dot * ks * cfg.scale;
                } else {
                    for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                        const uint16_t kb = k_pool[hb * 2 + d * 2] | (k_pool[hb * 2 + d * 2 + 1] << 8);
                        dot += qv[d] * f32_from_bf16(kb);
                    }
                    scores[p] = dot * cfg.scale;
                }
                max_score = std::fmax(max_score, scores[p]);
            }
            float sum_exp = 0.f;
            std::vector<float> acc(cfg.head_dim, 0.f);
            for (uint32_t p = 0; p < visible; ++p) {
                const float pv = std::exp(scores[p] - max_score);
                sum_exp += pv;
                const uint32_t lb = p / cfg.page_tokens;
                const uint32_t off = p % cfg.page_tokens;
                const uint32_t page = block_tables[static_cast<size_t>(slot) * cfg.block_table_stride + lb];
                const uint64_t hb = static_cast<uint64_t>(cfg.layer) * epl
                    + static_cast<uint64_t>(page) * epp
                    + static_cast<uint64_t>(off) * ept
                    + static_cast<uint64_t>(kv_head) * cfg.head_dim;
                if (cfg.fp8) {
                    const size_t si = (static_cast<size_t>(cfg.layer) * cfg.num_pages + page) * cfg.page_tokens + off;
                    const size_t sih = si * cfg.kv_heads + kv_head;
                    const float w = pv * v_scale[sih];
                    for (uint32_t d = 0; d < cfg.head_dim; ++d)
                        acc[d] += w * e4m3_decode_host(v_pool[hb + d]);
                } else {
                    for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                        const uint16_t vb = v_pool[hb * 2 + d * 2] | (v_pool[hb * 2 + d * 2 + 1] << 8);
                        acc[d] += pv * f32_from_bf16(vb);
                    }
                }
            }
            for (uint32_t d = 0; d < cfg.head_dim; ++d)
                ref[static_cast<size_t>(r) * q_features + qh * cfg.head_dim + d] = acc[d] / sum_exp;
        }
    }

    Dev d;
    bool ok = true;
    const size_t pool_bytes = cfg.fp8 ? pool_elems : pool_elems * 2;
    if (hipMalloc(&d.q, q_elems * 2) != hipSuccess) ok = false;
    if (hipMalloc(&d.out, q_elems * 4) != hipSuccess) ok = false;
    if (hipMalloc(&d.row_seq, cfg.rows * 4) != hipSuccess) ok = false;
    if (hipMalloc(&d.row_pos, cfg.rows * 4) != hipSuccess) ok = false;
    if (hipMalloc(&d.block_tables, bt_elems * 4) != hipSuccess) ok = false;
    if (hipMalloc(&d.k_pool, pool_bytes) != hipSuccess) ok = false;
    if (hipMalloc(&d.v_pool, pool_bytes) != hipSuccess) ok = false;
    if (cfg.fp8) {
        if (hipMalloc(&d.k_scale, scale_elems * 4) != hipSuccess) ok = false;
        if (hipMalloc(&d.v_scale, scale_elems * 4) != hipSuccess) ok = false;
    }
    if (!ok) { std::printf("FAIL: %s alloc\n", label); free_dev(d); check(false, label); return false; }

    if (hipMemcpy(d.q, q.data(), q_elems * 2, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemset(d.out, 0, q_elems * 4) != hipSuccess) ok = false;
    if (hipMemcpy(d.row_seq, row_seq.data(), cfg.rows * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.row_pos, row_pos.data(), cfg.rows * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.block_tables, block_tables.data(), bt_elems * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.k_pool, k_pool.data(), pool_bytes, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.v_pool, v_pool.data(), pool_bytes, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (cfg.fp8) {
        if (hipMemcpy(d.k_scale, k_scale.data(), scale_elems * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
        if (hipMemcpy(d.v_scale, v_scale.data(), scale_elems * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    }
    if (!ok) { std::printf("FAIL: %s upload\n", label); free_dev(d); check(false, label); return false; }

    ::ps::kernel::PagedAttentionCommonArgs c;
    c.q = static_cast<const bf16t*>(d.q);
    c.output = d.out;
    c.output_dtype = ::ps::kernel::PagedAttentionOutputDType::F32;
    c.q_row_stride = q_features;
    c.output_row_stride = q_features;
    c.row_sequence_slots = static_cast<const uint32_t*>(d.row_seq);
    c.row_positions = static_cast<const uint32_t*>(d.row_pos);
    c.block_tables = static_cast<const uint32_t*>(d.block_tables);
    c.block_table_stride = cfg.block_table_stride;
    c.max_sequences = cfg.max_sequences;
    c.rows = cfg.rows;
    c.layer = cfg.layer;
    c.page_tokens = cfg.page_tokens;
    c.num_pages = cfg.num_pages;
    c.num_attention_layers = cfg.num_layers;
    c.q_heads = cfg.q_heads;
    c.kv_heads = cfg.kv_heads;
    c.head_dim = cfg.head_dim;
    c.elems_per_token = ept;
    c.elems_per_page = epp;
    c.elems_per_layer = epl;
    c.scale = cfg.scale;

    hipError_t h = hipSuccess;
    if (cfg.prefill) {
        h = ::ps::kernel::launch_attention_paged_prefill_bf16(
            c, static_cast<const bf16t*>(d.k_pool), static_cast<const bf16t*>(d.v_pool), 0);
    } else if (cfg.fp8) {
        if (!g_wmma12) {
            std::printf("SKIP: %s (fp8 attention requires gfx12xx)\n", label);
            ++skipped_cases;
            return true;
        }
        h = ::ps::kernel::launch_attention_paged_fp8_e4m3(
            c, static_cast<const fp8*>(d.k_pool), static_cast<const fp8*>(d.v_pool),
            static_cast<const float*>(d.k_scale), static_cast<const float*>(d.v_scale), 0);
    } else {
        h = launch_paged_bf16(
            c, static_cast<const bf16t*>(d.k_pool), static_cast<const bf16t*>(d.v_pool), 0);
    }
    if (h == hipErrorNotSupported) {
        std::printf("SKIP: %s (wmma kernel not yet implemented for this arch)\n", label);
        free_dev(d);
        ++skipped_cases;
        return true;
    }
    ok = (h == hipSuccess) && (hipStreamSynchronize(0) == hipSuccess);
    if (!ok) {
        std::printf("FAIL: %s launch/sync (%s)\n", label, hipGetErrorString(h));
        free_dev(d);
        check(false, label);
        return false;
    }

    std::vector<float> got(q_elems);
    if (hipMemcpy(got.data(), d.out, q_elems * 4, hipMemcpyDeviceToHost) != hipSuccess) ok = false;
    free_dev(d);
    if (!ok) { check(false, label); return false; }

    double max_rel = 0.0;
    size_t mm = 0;
    const double tol = cfg.prefill ? 1e-2 : 2e-3;
    for (size_t i = 0; i < q_elems; ++i) {
        const double denom = std::fmax(1.0, std::fabs(static_cast<double>(ref[i])));
        const double rel = std::fabs(got[i] - ref[i]) / denom;
        if (rel > max_rel) max_rel = rel;
        if (rel > tol) ++mm;
    }
    if (mm != 0) ok = false;
    if (!ok) std::printf("  (max_rel=%.3e mm=%zu)\n", max_rel, mm);
    check(ok, label);
    return ok;
}

namespace psq4_attn {

using ::ps::quantization::psq::cb10_code_to_f32;

float decode_value(const std::vector<uint8_t>& code, const std::vector<uint16_t>& scale,
                   uint64_t bi, uint32_t d) {
    const uint32_t block = d / 32u;
    const uint32_t lane = d % 32u;
    const uint64_t b = bi + block;
    const uint8_t byte = code[b * 16u + (lane & 15u)];
    const uint32_t c = (lane < 16u) ? (static_cast<uint32_t>(byte) & 0xFu)
                                    : (static_cast<uint32_t>(byte) >> 4u);
    const float sc = f32_from_bf16(scale[b]);
    return sc * cb10_code_to_f32(static_cast<uint8_t>(c));
}

bool run_case_psq4(const Config& cfg, const char* label, bool prefill = false) {
    const uint32_t q_features = cfg.q_heads * cfg.head_dim;
    const uint32_t ept = cfg.kv_heads * cfg.head_dim;
    const uint32_t epp = cfg.page_tokens * ept;
    const uint32_t epl = cfg.num_pages * epp;
    const size_t q_elems = static_cast<size_t>(cfg.rows) * q_features;
    const size_t bt_elems = static_cast<size_t>(cfg.max_sequences) * cfg.block_table_stride;
    const uint32_t q_per_kv = cfg.q_heads / cfg.kv_heads;
    const uint32_t blocks_per_head = cfg.head_dim / 32u;
    const uint32_t blocks_per_token = cfg.kv_heads * blocks_per_head;
    const uint64_t total_blocks = static_cast<uint64_t>(cfg.num_layers) * cfg.num_pages
        * cfg.page_tokens * blocks_per_token;
    const size_t code_bytes = static_cast<size_t>(total_blocks) * 16u;

    std::vector<uint32_t> row_seq(cfg.rows), row_pos(cfg.rows), block_tables(bt_elems, 0);
    for (uint32_t r = 0; r < cfg.rows; ++r) {
        row_seq[r] = prefill ? 0u : (h32(cfg.seed * 1000u + r) % cfg.max_sequences);
        const uint32_t lb = r / cfg.page_tokens;
        uint32_t pos = prefill ? r : (lb * cfg.page_tokens + (r % cfg.page_tokens));
        if (cfg.visible_override != 0u) {
            const uint32_t cap = cfg.page_tokens * cfg.num_pages;
            pos = cfg.visible_override > 1u ? cfg.visible_override - 1u : 0u;
            if (pos > cap - 1u) pos = cap - 1u;
        }
        row_pos[r] = pos;
    }
    for (uint32_t slot = 0; slot < cfg.max_sequences; ++slot)
        for (uint32_t lb = 0; lb < cfg.block_table_stride; ++lb)
            block_tables[static_cast<size_t>(slot) * cfg.block_table_stride + lb] = lb % cfg.num_pages;

    std::vector<uint16_t> q(q_elems);
    for (size_t i = 0; i < q_elems; ++i)
        q[i] = bf16_bits((rng1k(cfg.seed * 9000u + i)) / 128.0);

    std::vector<uint8_t> k_code(code_bytes), v_code(code_bytes);
    std::vector<uint16_t> k_scale(total_blocks), v_scale(total_blocks);
    for (uint64_t bi = 0; bi < total_blocks; ++bi) {
        for (uint32_t by = 0; by < 16u; ++by) {
            k_code[bi * 16 + by] = static_cast<uint8_t>(h32(cfg.seed * 11u + bi * 16u + by) & 0xFFu);
            v_code[bi * 16 + by] = static_cast<uint8_t>(h32(cfg.seed * 13u + bi * 16u + by) & 0xFFu);
        }
        const float ks = 0.02f + static_cast<float>(h32(cfg.seed * 17u + bi) % 1000u) / 20000.0f;
        const float vs = 0.02f + static_cast<float>(h32(cfg.seed * 19u + bi) % 1000u) / 20000.0f;
        k_scale[bi] = bf16_bits(ks);
        v_scale[bi] = bf16_bits(vs);
    }

    std::vector<float> ref(q_elems, 0.0f);
    for (uint32_t r = 0; r < cfg.rows; ++r) {
        const uint32_t slot = row_seq[r];
        const uint32_t visible = row_pos[r] + 1u;
        for (uint32_t qh = 0; qh < cfg.q_heads; ++qh) {
            const uint32_t kv_head = qh / q_per_kv;
            std::vector<float> qv(cfg.head_dim);
            for (uint32_t d = 0; d < cfg.head_dim; ++d)
                qv[d] = f32_from_bf16(q[static_cast<size_t>(r) * q_features + qh * cfg.head_dim + d]);
            float max_score = -3.402823466e+38f;
            std::vector<float> scores(visible);
            for (uint32_t p = 0; p < visible; ++p) {
                const uint32_t lb = p / cfg.page_tokens;
                const uint32_t off = p % cfg.page_tokens;
                const uint32_t page = block_tables[static_cast<size_t>(slot) * cfg.block_table_stride + lb];
                const uint64_t tbb = ((static_cast<uint64_t>(cfg.layer) * cfg.num_pages + page)
                                      * cfg.page_tokens + off) * blocks_per_token
                                     + static_cast<uint64_t>(kv_head) * blocks_per_head;
                float dot = 0.0f;
                for (uint32_t d = 0; d < cfg.head_dim; ++d)
                    dot += qv[d] * decode_value(k_code, k_scale, tbb, d);
                scores[p] = dot * cfg.scale;
                max_score = std::fmax(max_score, scores[p]);
            }
            float sum_exp = 0.0f;
            std::vector<float> acc(cfg.head_dim, 0.0f);
            for (uint32_t p = 0; p < visible; ++p) {
                const float pv = std::exp(scores[p] - max_score);
                sum_exp += pv;
                const uint32_t lb = p / cfg.page_tokens;
                const uint32_t off = p % cfg.page_tokens;
                const uint32_t page = block_tables[static_cast<size_t>(slot) * cfg.block_table_stride + lb];
                const uint64_t tbb = ((static_cast<uint64_t>(cfg.layer) * cfg.num_pages + page)
                                      * cfg.page_tokens + off) * blocks_per_token
                                     + static_cast<uint64_t>(kv_head) * blocks_per_head;
                for (uint32_t d = 0; d < cfg.head_dim; ++d)
                    acc[d] += pv * decode_value(v_code, v_scale, tbb, d);
            }
            for (uint32_t d = 0; d < cfg.head_dim; ++d)
                ref[static_cast<size_t>(r) * q_features + qh * cfg.head_dim + d] = acc[d] / sum_exp;
        }
    }

    Dev d;
    bool ok = true;
    if (hipMalloc(&d.q, q_elems * 2) != hipSuccess) ok = false;
    if (hipMalloc(&d.out, q_elems * 4) != hipSuccess) ok = false;
    if (hipMalloc(&d.row_seq, cfg.rows * 4) != hipSuccess) ok = false;
    if (hipMalloc(&d.row_pos, cfg.rows * 4) != hipSuccess) ok = false;
    if (hipMalloc(&d.block_tables, bt_elems * 4) != hipSuccess) ok = false;
    if (hipMalloc(&d.k_pool, code_bytes) != hipSuccess) ok = false;
    if (hipMalloc(&d.v_pool, code_bytes) != hipSuccess) ok = false;
    if (hipMalloc(&d.k_scale, total_blocks * 2) != hipSuccess) ok = false;
    if (hipMalloc(&d.v_scale, total_blocks * 2) != hipSuccess) ok = false;
    if (!ok) { std::printf("FAIL: %s alloc\n", label); free_dev(d); check(false, label); return false; }

    if (hipMemcpy(d.q, q.data(), q_elems * 2, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemset(d.out, 0, q_elems * 4) != hipSuccess) ok = false;
    if (hipMemcpy(d.row_seq, row_seq.data(), cfg.rows * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.row_pos, row_pos.data(), cfg.rows * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.block_tables, block_tables.data(), bt_elems * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.k_pool, k_code.data(), code_bytes, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.v_pool, v_code.data(), code_bytes, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.k_scale, k_scale.data(), total_blocks * 2, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.v_scale, v_scale.data(), total_blocks * 2, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (!ok) { std::printf("FAIL: %s upload\n", label); free_dev(d); check(false, label); return false; }

    ::ps::kernel::PagedAttentionCommonArgs c;
    c.q = static_cast<const bf16t*>(d.q);
    c.output = d.out;
    c.output_dtype = ::ps::kernel::PagedAttentionOutputDType::F32;
    c.q_row_stride = q_features;
    c.output_row_stride = q_features;
    c.row_sequence_slots = static_cast<const uint32_t*>(d.row_seq);
    c.row_positions = static_cast<const uint32_t*>(d.row_pos);
    c.block_tables = static_cast<const uint32_t*>(d.block_tables);
    c.block_table_stride = cfg.block_table_stride;
    c.max_sequences = cfg.max_sequences;
    c.rows = cfg.rows;
    c.layer = cfg.layer;
    c.page_tokens = cfg.page_tokens;
    c.num_pages = cfg.num_pages;
    c.num_attention_layers = cfg.num_layers;
    c.q_heads = cfg.q_heads;
    c.kv_heads = cfg.kv_heads;
    c.head_dim = cfg.head_dim;
    c.elems_per_token = ept;
    c.elems_per_page = epp;
    c.elems_per_layer = epl;
    c.scale = cfg.scale;

    hipError_t h = hipSuccess;
    if (prefill) {
        h = ::ps::kernel::launch_attention_paged_prefill_psq4(
            c, static_cast<const uint8_t*>(d.k_pool), static_cast<const uint8_t*>(d.v_pool),
            reinterpret_cast<const ps::bf16_t*>(d.k_scale),
            reinterpret_cast<const ps::bf16_t*>(d.v_scale), blocks_per_head, 0);
    } else {
        h = ::ps::kernel::launch_attention_paged_psq4(
            c, static_cast<const uint8_t*>(d.k_pool), static_cast<const uint8_t*>(d.v_pool),
            reinterpret_cast<const ps::bf16_t*>(d.k_scale),
            reinterpret_cast<const ps::bf16_t*>(d.v_scale), blocks_per_head, 0);
    }
    ok = (h == hipSuccess) && (hipStreamSynchronize(0) == hipSuccess);
    if (!ok) {
        std::printf("FAIL: %s launch/sync (%s)\n", label, hipGetErrorString(h));
        free_dev(d);
        check(false, label);
        return false;
    }

    std::vector<float> got(q_elems);
    if (hipMemcpy(got.data(), d.out, q_elems * 4, hipMemcpyDeviceToHost) != hipSuccess) ok = false;
    free_dev(d);
    if (!ok) { check(false, label); return false; }

    double max_rel = 0.0;
    size_t mm = 0;
    const double tol = prefill ? 1e-2 : 2e-3;
    for (size_t i = 0; i < q_elems; ++i) {
        const double denom = std::fmax(1.0, std::fabs(static_cast<double>(ref[i])));
        const double rel = std::fabs(got[i] - ref[i]) / denom;
        if (rel > max_rel) max_rel = rel;
        if (rel > tol) ++mm;
    }
    if (mm != 0) ok = false;
    if (!ok) std::printf("  (psq4 max_rel=%.3e mm=%zu)\n", max_rel, mm);
    check(ok, label);
    return ok;
}

}  // namespace psq4_attn

}  // namespace

int main() {
    g_wmma12 = true;
    Config c;
    c.rows = 8;
    run_case(c, "bf16 GQA qh=8 kvh=4 hd=64 rows=8 causal");
    c = Config{};
    c.rows = 1;
    c.q_heads = 4;
    c.kv_heads = 4;
    c.head_dim = 128;
    run_case(c, "bf16 MHA qh=kvh=4 hd=128 decode rows=1");
    c = Config{};
    c.rows = 16;
    c.q_heads = 8;
    c.kv_heads = 2;
    c.head_dim = 32;
    c.page_tokens = 8;
    run_case(c, "bf16 GQA qh=8 kvh=2 hd=32 page=8 rows=16");
    c = Config{};
    c.rows = 8;
    c.fp8 = true;
    run_case(c, "fp8 GQA qh=8 kvh=4 hd=64 rows=8 causal");
    c = Config{};
    c.rows = 1;
    c.fp8 = true;
    c.q_heads = 4;
    c.kv_heads = 2;
    c.head_dim = 128;
    run_case(c, "fp8 GQA qh=4 kvh=2 hd=128 decode rows=1");
    c = Config{};
    c.rows = 24;
    c.q_heads = 16;
    c.kv_heads = 4;
    c.head_dim = 64;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.block_table_stride = 16;
    c.max_sequences = 8;
    run_case(c, "bf16 real-ish qh=16 kvh=4 hd=64 page=16 rows=24");
    c = Config{};
    c.rows = 4;
    c.q_heads = 4;
    c.kv_heads = 2;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.block_table_stride = 16;
    c.max_sequences = 8;
    c.prefill = true;
    run_case(c, "bf16 PREFILL qh=4 kvh=2 hd=256 page=16 rows=4");
    c = Config{};
    c.rows = 64;
    c.q_heads = 4;
    c.kv_heads = 2;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.block_table_stride = 16;
    c.max_sequences = 8;
    c.prefill = true;
    run_case(c, "bf16 PREFILL qh=4 kvh=2 hd=256 page=16 rows=64");
    c = Config{};
    c.rows = 1;
    c.q_heads = 16;
    c.kv_heads = 4;
    c.head_dim = 256;
    c.page_tokens = 64;
    c.num_pages = 128;
    c.num_layers = 1;
    c.layer = 0;
    c.block_table_stride = 128;
    c.max_sequences = 1;
    c.visible_override = 8192;
    run_case(c, "bf16 DECODE long visible=8192 hd=256 rows=1");
    c = Config{};
    c.rows = 1;
    c.q_heads = 16;
    c.kv_heads = 4;
    c.head_dim = 256;
    c.page_tokens = 64;
    c.num_pages = 128;
    c.num_layers = 1;
    c.layer = 0;
    c.block_table_stride = 128;
    c.max_sequences = 1;
    c.visible_override = 4097;
    run_case(c, "bf16 DECODE long visible=4097 hd=256 rows=1");
    c = Config{};
    c.rows = 1;
    c.q_heads = 16;
    c.kv_heads = 4;
    c.head_dim = 256;
    c.page_tokens = 64;
    c.num_pages = 512;
    c.num_layers = 1;
    c.layer = 0;
    c.block_table_stride = 512;
    c.max_sequences = 1;
    c.visible_override = 32768;
    run_case(c, "bf16 DECODE long visible=32768 hd=256 rows=1");

    // Qwen3.8-27B TP=2 local geometry: q_heads=12, kv_heads=2, head_dim=256, page=16.
    c = Config{};
    c.rows = 1;
    c.q_heads = 12;
    c.kv_heads = 2;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.num_layers = 1;
    c.layer = 0;
    c.block_table_stride = 16;
    c.max_sequences = 1;
    c.visible_override = 64;
    run_case(c, "bf16 TP2 DECODE qh=12 kvh=2 hd=256 page=16 vis=64 rows=1");
    c = Config{};
    c.rows = 64;
    c.q_heads = 12;
    c.kv_heads = 2;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.num_layers = 1;
    c.layer = 0;
    c.block_table_stride = 16;
    c.max_sequences = 8;
    c.prefill = true;
    run_case(c, "bf16 TP2 PREFILL qh=12 kvh=2 hd=256 page=16 rows=64");
    c = Config{};
    c.rows = 1;
    c.q_heads = 12;
    c.kv_heads = 2;
    c.head_dim = 256;
    c.page_tokens = 64;
    c.num_pages = 64;
    c.num_layers = 1;
    c.layer = 0;
    c.block_table_stride = 64;
    c.max_sequences = 1;
    c.visible_override = 2048;
    run_case(c, "bf16 TP2 DECODE qh=12 kvh=2 hd=256 page=64 vis=2048 rows=1");

    c = Config{};
    c.rows = 8;
    c.q_heads = 16;
    c.kv_heads = 4;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.block_table_stride = 16;
    c.max_sequences = 4;
    psq4_attn::run_case_psq4(c, "psq4 decode qh=16 kvh=4 hd=256 rows=8");
    c = Config{};
    c.rows = 4;
    c.q_heads = 24;
    c.kv_heads = 4;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.block_table_stride = 16;
    c.max_sequences = 4;
    psq4_attn::run_case_psq4(c, "psq4 decode qh=24 kvh=4 hd=256 rows=4");
    c = Config{};
    c.rows = 1;
    c.q_heads = 16;
    c.kv_heads = 4;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 128;
    c.num_layers = 1;
    c.layer = 0;
    c.block_table_stride = 128;
    c.max_sequences = 1;
    c.visible_override = 2048;
    psq4_attn::run_case_psq4(c, "psq4 decode long visible=2048 hd=256 rows=1");
    c = Config{};
    c.rows = 32;
    c.q_heads = 16;
    c.kv_heads = 4;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.block_table_stride = 16;
    c.max_sequences = 1;
    psq4_attn::run_case_psq4(c, "psq4 PREFILL qh=16 kvh=4 hd=256 rows=32", true);
    c = Config{};
    c.rows = 64;
    c.q_heads = 16;
    c.kv_heads = 4;
    c.head_dim = 256;
    c.page_tokens = 16;
    c.num_pages = 16;
    c.block_table_stride = 16;
    c.max_sequences = 1;
    psq4_attn::run_case_psq4(c, "psq4 PREFILL qh=16 kvh=4 hd=256 rows=64", true);

    std::printf("\nResults: %d passed, %d failed, %d skipped\n", passed, failed, skipped_cases);
    if (failed == 0 && passed == 0)
        return 77;
    return failed > 0 ? 1 : 0;
}
