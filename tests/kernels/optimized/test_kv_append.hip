#include <phaseshift/models/qwen35/kernels/optimized/kv_append.h>
#include <phaseshift/quantization/psq/psq4_kv.h>
#include <phaseshift/models/qwen35/state/kv_cache_types.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using bf16 = __hip_bfloat16;
using fp8 = ::ps::qwen35::fp8e4m3_storage_t;
using bf16t = ps::bf16_t;

int passed = 0;
int failed = 0;

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

float f32_from_bf16(uint16_t bits) {
    uint32_t ui = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &ui, 4);
    return f;
}

constexpr float kFp8E4M3Max = 448.0f;

struct Config {
    uint32_t kv_heads = 4;
    uint32_t head_dim = 256;
    uint32_t page_tokens = 16;
    uint32_t num_pages = 8;
    uint32_t num_layers = 4;
    uint32_t layer = 2;
    uint32_t block_table_stride = 6;
    uint32_t max_sequences = 4;
    uint32_t rows = 8;
    bool fp8 = false;
    bool multi_seq = false;
    uint32_t seed = 1;
};

struct Dev {
    void* k_in = nullptr;
    void* v_in = nullptr;
    void* row_seq = nullptr;
    void* row_pos = nullptr;
    void* block_tables = nullptr;
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    void* k_scale = nullptr;
    void* v_scale = nullptr;
};

void free_dev(Dev& d) {
    if (d.k_in) ps::gpu::discard_cleanup_result(hipFree(d.k_in));
    if (d.v_in) ps::gpu::discard_cleanup_result(hipFree(d.v_in));
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
    const uint32_t ept = cfg.kv_heads * cfg.head_dim;
    const uint32_t epp = cfg.page_tokens * ept;
    const uint32_t epl = cfg.num_pages * epp;
    const uint64_t pool_elems = static_cast<uint64_t>(cfg.num_layers) * epl;
    const uint32_t scale_elems = cfg.num_layers * cfg.num_pages * cfg.page_tokens * cfg.kv_heads;
    const size_t input_elems = static_cast<size_t>(cfg.rows) * ept;
    const size_t bt_elems = static_cast<size_t>(cfg.max_sequences) * cfg.block_table_stride;

    std::vector<uint32_t> row_seq(cfg.rows), row_pos(cfg.rows), block_tables(bt_elems, 0);
    std::vector<uint16_t> k_in(input_elems), v_in(input_elems);
    if (cfg.multi_seq) {
        for (uint32_t r = 0; r < cfg.rows; ++r) {
            row_seq[r] = r % cfg.max_sequences;
            row_pos[r] = 0;
        }
        for (uint32_t slot = 0; slot < cfg.max_sequences; ++slot)
            for (uint32_t lb = 0; lb < cfg.block_table_stride; ++lb)
                block_tables[static_cast<size_t>(slot) * cfg.block_table_stride + lb] = slot % cfg.num_pages;
    } else {
        for (uint32_t r = 0; r < cfg.rows; ++r) {
            row_seq[r] = h32(cfg.seed * 1000u + r) % cfg.max_sequences;
            const uint32_t lb = r / cfg.page_tokens;
            const uint32_t off = r % cfg.page_tokens;
            row_pos[r] = lb * cfg.page_tokens + off;
        }
        for (uint32_t slot = 0; slot < cfg.max_sequences; ++slot)
            for (uint32_t lb = 0; lb < cfg.block_table_stride; ++lb)
                block_tables[static_cast<size_t>(slot) * cfg.block_table_stride + lb] = lb % cfg.num_pages;
    }
    for (size_t i = 0; i < input_elems; ++i) {
        const double k = (h32(cfg.seed * 5000u + i) % 2001 - 1000) / 128.0;
        const double v = (h32(cfg.seed * 6000u + i) % 2001 - 1000) / 128.0;
        k_in[i] = bf16_bits(k);
        v_in[i] = bf16_bits(v);
    }

    const uint8_t bf16_sentinel = 0x3F;
    const uint8_t fp8_sentinel = 0xEE;
    const float scale_sentinel = 0.0f;
    const uint64_t pool_byte_len = cfg.fp8 ? pool_elems : pool_elems * 2;

    std::vector<uint8_t> exp_k_pool, exp_v_pool;
    std::vector<float> exp_k_scale, exp_v_scale;
    const uint8_t init_byte = cfg.fp8 ? fp8_sentinel : bf16_sentinel;
    exp_k_pool.assign(pool_byte_len, init_byte);
    exp_v_pool.assign(pool_byte_len, init_byte);
    if (cfg.fp8) {
        exp_k_scale.assign(scale_elems, scale_sentinel);
        exp_v_scale.assign(scale_elems, scale_sentinel);
    }

    for (uint32_t r = 0; r < cfg.rows; ++r) {
        const uint32_t slot = row_seq[r];
        const uint32_t pos = row_pos[r];
        const uint32_t lb = pos / cfg.page_tokens;
        const uint32_t off = pos % cfg.page_tokens;
        const uint32_t page = block_tables[static_cast<size_t>(slot) * cfg.block_table_stride + lb];
        const uint64_t base = static_cast<uint64_t>(cfg.layer) * epl +
                              static_cast<uint64_t>(page) * epp +
                              static_cast<uint64_t>(off) * ept;
        for (uint32_t h = 0; h < cfg.kv_heads; ++h) {
            const uint64_t hb = base + static_cast<uint64_t>(h) * cfg.head_dim;
            if (cfg.fp8) {
                float k_amax = 0.f, v_amax = 0.f;
                for (uint32_t i = 0; i < cfg.head_dim; ++i) {
                    const float kf = f32_from_bf16(k_in[static_cast<size_t>(r) * ept + h * cfg.head_dim + i]);
                    const float vf = f32_from_bf16(v_in[static_cast<size_t>(r) * ept + h * cfg.head_dim + i]);
                    k_amax = std::fmax(k_amax, std::fabs(kf));
                    v_amax = std::fmax(v_amax, std::fabs(vf));
                }
                const float k_scale = k_amax == 0.f ? 1.f : k_amax / kFp8E4M3Max;
                const float v_scale = v_amax == 0.f ? 1.f : v_amax / kFp8E4M3Max;
                const size_t si = (static_cast<size_t>(cfg.layer) * cfg.num_pages + page) * cfg.page_tokens + off;
                const size_t sih = si * cfg.kv_heads + h;
                exp_k_scale[sih] = k_scale;
                exp_v_scale[sih] = v_scale;
                for (uint32_t i = 0; i < cfg.head_dim; ++i) {
                    const float kf = f32_from_bf16(k_in[static_cast<size_t>(r) * ept + h * cfg.head_dim + i]);
                    const float vf = f32_from_bf16(v_in[static_cast<size_t>(r) * ept + h * cfg.head_dim + i]);
                    exp_k_pool[hb + i] = static_cast<uint8_t>(
                        __hip_cvt_float_to_fp8(kf / k_scale, __HIP_SATFINITE, __HIP_E4M3));
                    exp_v_pool[hb + i] = static_cast<uint8_t>(
                        __hip_cvt_float_to_fp8(vf / v_scale, __HIP_SATFINITE, __HIP_E4M3));
                }
            } else {
                for (uint32_t i = 0; i < cfg.head_dim; ++i) {
                    const uint16_t kb = k_in[static_cast<size_t>(r) * ept + h * cfg.head_dim + i];
                    const uint16_t vb = v_in[static_cast<size_t>(r) * ept + h * cfg.head_dim + i];
                    exp_k_pool[hb * 2 + i * 2] = kb & 0xFF;
                    exp_k_pool[hb * 2 + i * 2 + 1] = kb >> 8;
                    exp_v_pool[hb * 2 + i * 2] = vb & 0xFF;
                    exp_v_pool[hb * 2 + i * 2 + 1] = vb >> 8;
                }
            }
        }
    }

    Dev d;
    bool ok = true;
    const size_t pool_bytes = cfg.fp8 ? pool_elems : pool_elems * 2;
    if (hipMalloc(&d.k_in, input_elems * 2) != hipSuccess) ok = false;
    if (hipMalloc(&d.v_in, input_elems * 2) != hipSuccess) ok = false;
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

    if (hipMemcpy(d.k_in, k_in.data(), input_elems * 2, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.v_in, v_in.data(), input_elems * 2, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.row_seq, row_seq.data(), cfg.rows * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.row_pos, row_pos.data(), cfg.rows * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.block_tables, block_tables.data(), bt_elems * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemset(d.k_pool, init_byte, pool_bytes) != hipSuccess) ok = false;
    if (hipMemset(d.v_pool, init_byte, pool_bytes) != hipSuccess) ok = false;
    if (cfg.fp8) {
        if (hipMemset(d.k_scale, 0, scale_elems * 4) != hipSuccess) ok = false;
        if (hipMemset(d.v_scale, 0, scale_elems * 4) != hipSuccess) ok = false;
    }
    if (!ok) { std::printf("FAIL: %s upload\n", label); free_dev(d); check(false, label); return false; }

    ::ps::kernel::KvAppendCommonArgs c;
    c.k_input = static_cast<const bf16t*>(d.k_in);
    c.v_input = static_cast<const bf16t*>(d.v_in);
    c.k_row_stride = ept;
    c.v_row_stride = ept;
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
    c.kv_heads = cfg.kv_heads;
    c.head_dim = cfg.head_dim;
    c.elems_per_token = ept;
    c.elems_per_page = epp;
    c.elems_per_layer = epl;

    hipError_t h = hipSuccess;
    if (cfg.fp8) {
        h = ::ps::kernel::launch_kv_append_fp8_e4m3(
            c, static_cast<fp8*>(d.k_pool), static_cast<fp8*>(d.v_pool),
            static_cast<float*>(d.k_scale), static_cast<float*>(d.v_scale), 0);
    } else {
        h = ::ps::kernel::launch_kv_append_bf16(
            c, static_cast<bf16t*>(d.k_pool), static_cast<bf16t*>(d.v_pool), 0);
    }
    ok = (h == hipSuccess) && (hipStreamSynchronize(0) == hipSuccess);
    if (!ok) {
        std::printf("FAIL: %s launch/sync (%s)\n", label, hipGetErrorString(h));
        free_dev(d);
        check(false, label);
        return false;
    }

    std::vector<uint8_t> got_k_pool(pool_byte_len), got_v_pool(pool_byte_len);
    if (hipMemcpy(got_k_pool.data(), d.k_pool, pool_byte_len, hipMemcpyDeviceToHost) != hipSuccess) ok = false;
    if (hipMemcpy(got_v_pool.data(), d.v_pool, pool_byte_len, hipMemcpyDeviceToHost) != hipSuccess) ok = false;
    std::vector<float> got_ks, got_vs;
    if (cfg.fp8) {
        got_ks.resize(scale_elems);
        got_vs.resize(scale_elems);
        if (hipMemcpy(got_ks.data(), d.k_scale, scale_elems * 4, hipMemcpyDeviceToHost) != hipSuccess) ok = false;
        if (hipMemcpy(got_vs.data(), d.v_scale, scale_elems * 4, hipMemcpyDeviceToHost) != hipSuccess) ok = false;
    }
    free_dev(d);
    if (!ok) { check(false, label); return false; }

    size_t k_mm = 0, v_mm = 0;
    for (uint64_t i = 0; i < pool_byte_len; ++i) {
        if (got_k_pool[i] != exp_k_pool[i]) ++k_mm;
        if (got_v_pool[i] != exp_v_pool[i]) ++v_mm;
    }
    if (k_mm != 0 || v_mm != 0) ok = false;

    size_t ks_mm = 0, vs_mm = 0;
    if (cfg.fp8) {
        for (uint32_t i = 0; i < scale_elems; ++i) {
            if (got_ks[i] != exp_k_scale[i]) ++ks_mm;
            if (got_vs[i] != exp_v_scale[i]) ++vs_mm;
        }
        if (ks_mm != 0 || vs_mm != 0) ok = false;
    }

    if (!ok) std::printf("  (k_mm=%zu v_mm=%zu ks_mm=%zu vs_mm=%zu)\n", k_mm, v_mm, ks_mm, vs_mm);
    check(ok, label);
    return ok;
}

namespace psq4_test {

constexpr uint32_t kBlocksPerHead = ::ps::qwen35::kPsq4BlocksPerHead;
constexpr uint32_t kCodeBytesPerBlock = ::ps::qwen35::kPsq4CodeBytesPerBlock;

using Est = ::ps::qwen35::Psq4ScaleEstimator;

float pattern_value(int pattern, uint64_t seed, uint64_t index, bool alt_positive) {
    const double r = static_cast<double>(h32(seed) % 2001) / 128.0 - 7.8;
    const double sign = alt_positive ? 1.0 : ((h32(seed + 777u) & 1u) ? 1.0 : -1.0);
    switch (pattern) {
        case 0: return static_cast<float>(r);
        case 1: return 0.0f;
        case 2: return static_cast<float>(std::fabs(r));
        case 3: return static_cast<float>(-std::fabs(r));
        case 4: return static_cast<float>(sign * std::fabs(r));
        case 5: return (index % 32u) == 0u ? static_cast<float>(sign * 30.0) : static_cast<float>(r * 0.05);
        case 6: return static_cast<float>(r * 1e-4);
        case 7: return static_cast<float>(sign * std::fabs(r));
        default: return static_cast<float>(r);
    }
}

bool run_psq4_case(const Config& cfg, const char* label, int pattern, bool multi_seq, Est est) {
    const uint32_t blocks_per_token = cfg.kv_heads * kBlocksPerHead;
    const uint64_t total_blocks = static_cast<uint64_t>(cfg.num_layers) * cfg.num_pages
        * cfg.page_tokens * blocks_per_token;
    const size_t code_bytes = static_cast<size_t>(total_blocks) * kCodeBytesPerBlock;
    const size_t scale_elems = static_cast<size_t>(total_blocks);
    const size_t input_elems = static_cast<size_t>(cfg.rows) * cfg.kv_heads * cfg.head_dim;
    const size_t bt_elems = static_cast<size_t>(cfg.max_sequences) * cfg.block_table_stride;

    std::vector<uint32_t> row_seq(cfg.rows), row_pos(cfg.rows), block_tables(bt_elems, 0);
    std::vector<float> k_in(input_elems), v_in(input_elems);
    std::vector<uint16_t> k_bits(input_elems), v_bits(input_elems);
    if (multi_seq) {
        for (uint32_t r = 0; r < cfg.rows; ++r) {
            row_seq[r] = r % cfg.max_sequences;
            row_pos[r] = 0;
        }
        for (uint32_t slot = 0; slot < cfg.max_sequences; ++slot)
            for (uint32_t lb = 0; lb < cfg.block_table_stride; ++lb)
                block_tables[slot * cfg.block_table_stride + lb] = slot % cfg.num_pages;
    } else {
        for (uint32_t r = 0; r < cfg.rows; ++r) {
            row_seq[r] = h32(cfg.seed * 1000u + r) % cfg.max_sequences;
            const uint32_t lb = r / cfg.page_tokens;
            const uint32_t off = r % cfg.page_tokens;
            row_pos[r] = lb * cfg.page_tokens + off;
        }
        for (uint32_t slot = 0; slot < cfg.max_sequences; ++slot)
            for (uint32_t lb = 0; lb < cfg.block_table_stride; ++lb)
                block_tables[slot * cfg.block_table_stride + lb] = lb % cfg.num_pages;
    }
    for (size_t i = 0; i < input_elems; ++i) {
        k_bits[i] = bf16_bits(pattern_value(pattern, cfg.seed * 5000u + i, i, (i & 1u) == 0u));
        v_bits[i] = bf16_bits(pattern_value(pattern, cfg.seed * 6000u + i + 999u, i, (i & 1u) != 0u));
        k_in[i] = f32_from_bf16(k_bits[i]);
        v_in[i] = f32_from_bf16(v_bits[i]);
    }

    const uint8_t code_sentinel = 0xA5;
    std::vector<uint8_t> k_code(code_bytes, code_sentinel);
    std::vector<uint8_t> v_code(code_bytes, code_sentinel);
    std::vector<uint16_t> k_scale(scale_elems, 0xFFFFu);
    std::vector<uint16_t> v_scale(scale_elems, 0xFFFFu);
    auto written = std::vector<uint8_t>(total_blocks, 0);

    for (uint32_t r = 0; r < cfg.rows; ++r) {
        const uint32_t slot = row_seq[r];
        const uint32_t pos = row_pos[r];
        const uint32_t lb = pos / cfg.page_tokens;
        const uint32_t off = pos % cfg.page_tokens;
        const uint32_t page = block_tables[slot * cfg.block_table_stride + lb];
        for (uint32_t h = 0; h < cfg.kv_heads; ++h) {
            const size_t row_base = static_cast<size_t>(r) * cfg.kv_heads * cfg.head_dim
                + static_cast<size_t>(h) * cfg.head_dim;
            const uint64_t token_block_base =
                ((static_cast<uint64_t>(cfg.layer) * cfg.num_pages + page) * cfg.page_tokens + off)
                * blocks_per_token + static_cast<uint64_t>(h) * kBlocksPerHead;
            for (uint32_t blk = 0; blk < kBlocksPerHead; ++blk) {
                const uint64_t bi = token_block_base + blk;
                written[bi] = 1;
                float kx[32];
                float vx[32];
                for (int i = 0; i < 32; ++i) {
                    kx[i] = k_in[row_base + blk * 32u + static_cast<size_t>(i)];
                    vx[i] = v_in[row_base + blk * 32u + static_cast<size_t>(i)];
                }
                float ks = 0.0f, vs = 0.0f;
                uint8_t kcodes[32], vcodes[32];
                uint8_t kbytes[16], vbytes[16];
                if (est == Est::MaxAbs) {
                    ::ps::quantization::psq::psq4_maxabs_encode_block(kx, 32u, ks, kcodes);
                    ::ps::quantization::psq::psq4_maxabs_encode_block(vx, 32u, vs, vcodes);
                } else {
                    ::ps::quantization::psq::psq4_lsq1_encode_block(kx, 32u, ks, kcodes);
                    ::ps::quantization::psq::psq4_lsq1_encode_block(vx, 32u, vs, vcodes);
                }
                ::ps::quantization::psq::psq4_pack_codes(kcodes, kbytes);
                ::ps::quantization::psq::psq4_pack_codes(vcodes, vbytes);
                k_scale[bi] = bf16_bits(ks);
                v_scale[bi] = bf16_bits(vs);
                for (int b = 0; b < 16; ++b) {
                    k_code[bi * 16 + static_cast<size_t>(b)] = kbytes[b];
                    v_code[bi * 16 + static_cast<size_t>(b)] = vbytes[b];
                }
            }
        }
    }

    Dev d;
    bool ok = true;
    if (hipMalloc(&d.k_in, input_elems * 2) != hipSuccess) ok = false;
    if (hipMalloc(&d.v_in, input_elems * 2) != hipSuccess) ok = false;
    if (hipMalloc(&d.row_seq, cfg.rows * 4) != hipSuccess) ok = false;
    if (hipMalloc(&d.row_pos, cfg.rows * 4) != hipSuccess) ok = false;
    if (hipMalloc(&d.block_tables, bt_elems * 4) != hipSuccess) ok = false;
    if (hipMalloc(&d.k_pool, code_bytes) != hipSuccess) ok = false;
    if (hipMalloc(&d.v_pool, code_bytes) != hipSuccess) ok = false;
    if (hipMalloc(&d.k_scale, scale_elems * 2) != hipSuccess) ok = false;
    if (hipMalloc(&d.v_scale, scale_elems * 2) != hipSuccess) ok = false;
    if (!ok) { std::printf("FAIL: %s alloc\n", label); free_dev(d); check(false, label); return false; }

    if (hipMemcpy(d.k_in, k_bits.data(), input_elems * 2, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.v_in, v_bits.data(), input_elems * 2, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.row_seq, row_seq.data(), cfg.rows * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.row_pos, row_pos.data(), cfg.rows * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemcpy(d.block_tables, block_tables.data(), bt_elems * 4, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (hipMemset(d.k_pool, code_sentinel, code_bytes) != hipSuccess) ok = false;
    if (hipMemset(d.v_pool, code_sentinel, code_bytes) != hipSuccess) ok = false;
    if (hipMemset(d.k_scale, 0xFF, scale_elems * 2) != hipSuccess) ok = false;
    if (hipMemset(d.v_scale, 0xFF, scale_elems * 2) != hipSuccess) ok = false;
    if (!ok) { std::printf("FAIL: %s upload\n", label); free_dev(d); check(false, label); return false; }

    ::ps::kernel::KvAppendCommonArgs c;
    c.k_input = static_cast<const bf16t*>(d.k_in);
    c.v_input = static_cast<const bf16t*>(d.v_in);
    c.k_row_stride = cfg.kv_heads * cfg.head_dim;
    c.v_row_stride = cfg.kv_heads * cfg.head_dim;
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
    c.kv_heads = cfg.kv_heads;
    c.head_dim = cfg.head_dim;
    c.elems_per_token = cfg.kv_heads * cfg.head_dim;
    c.elems_per_page = cfg.page_tokens * c.elems_per_token;
    c.elems_per_layer = cfg.num_pages * c.elems_per_page;

    hipError_t h = ::ps::kernel::launch_kv_append_psq4(
        c, static_cast<uint8_t*>(d.k_pool), static_cast<uint8_t*>(d.v_pool),
        reinterpret_cast<ps::bf16_t*>(d.k_scale), reinterpret_cast<ps::bf16_t*>(d.v_scale),
        kBlocksPerHead, est, 0);
    if (h == hipSuccess) h = hipDeviceSynchronize();
    if (h != hipSuccess) {
        std::printf("FAIL: %s launch: %s\n", label, hipGetErrorString(h));
        free_dev(d);
        check(false, label);
        return false;
    }

    std::vector<uint8_t> got_k_code(code_bytes, 0);
    std::vector<uint8_t> got_v_code(code_bytes, 0);
    std::vector<uint16_t> got_k_scale(scale_elems, 0);
    std::vector<uint16_t> got_v_scale(scale_elems, 0);
    hipMemcpy(got_k_code.data(), d.k_pool, code_bytes, hipMemcpyDeviceToHost);
    hipMemcpy(got_v_code.data(), d.v_pool, code_bytes, hipMemcpyDeviceToHost);
    hipMemcpy(got_k_scale.data(), d.k_scale, scale_elems * 2, hipMemcpyDeviceToHost);
    hipMemcpy(got_v_scale.data(), d.v_scale, scale_elems * 2, hipMemcpyDeviceToHost);

    uint64_t written_count = 0;
    uint64_t code_mismatch = 0;
    uint64_t scale_mismatch = 0;
    bool sentinel_ok = true;
    for (uint64_t bi = 0; bi < total_blocks; ++bi) {
        if (written[bi] == 0) {
            if (got_k_scale[bi] != 0xFFFFu || got_v_scale[bi] != 0xFFFFu) sentinel_ok = false;
            for (int b = 0; b < 16; ++b) {
                if (got_k_code[bi * 16 + static_cast<size_t>(b)] != code_sentinel) sentinel_ok = false;
                if (got_v_code[bi * 16 + static_cast<size_t>(b)] != code_sentinel) sentinel_ok = false;
            }
            continue;
        }
        ++written_count;
        if (got_k_scale[bi] != k_scale[bi]) ++scale_mismatch;
        if (got_v_scale[bi] != v_scale[bi]) ++scale_mismatch;
        bool mismatch = false;
        for (int b = 0; b < 16; ++b) {
            if (got_k_code[bi * 16 + static_cast<size_t>(b)] != k_code[bi * 16 + static_cast<size_t>(b)]) mismatch = true;
            if (got_v_code[bi * 16 + static_cast<size_t>(b)] != v_code[bi * 16 + static_cast<size_t>(b)]) mismatch = true;
        }
        if (mismatch) ++code_mismatch;
    }
    check(sentinel_ok, "psq4 unwritten blocks keep sentinel");
    if (est == Est::MaxAbs) {
        check(scale_mismatch == 0 && code_mismatch == 0, "psq4 maxabs matches cpu reference");
    } else {
        const bool close = scale_mismatch * 100u <= written_count * 2u &&
                           code_mismatch * 100u <= written_count * 5u;
        if (!close) {
            std::printf("  (%s scale_mismatch=%llu code_mismatch=%llu written=%llu)\n", label,
                        static_cast<unsigned long long>(scale_mismatch),
                        static_cast<unsigned long long>(code_mismatch),
                        static_cast<unsigned long long>(written_count));
        }
        check(close, "psq4 lsq1 close to cpu reference");
    }

    bool recon_ok = true;
    for (uint32_t r = 0; r < cfg.rows && recon_ok; ++r) {
        const uint32_t slot = row_seq[r];
        const uint32_t pos = row_pos[r];
        const uint32_t lb = pos / cfg.page_tokens;
        const uint32_t off = pos % cfg.page_tokens;
        const uint32_t page = block_tables[slot * cfg.block_table_stride + lb];
        for (uint32_t h = 0; h < cfg.kv_heads && recon_ok; ++h) {
            const size_t row_base = static_cast<size_t>(r) * cfg.kv_heads * cfg.head_dim
                + static_cast<size_t>(h) * cfg.head_dim;
            const uint64_t token_block_base =
                ((static_cast<uint64_t>(cfg.layer) * cfg.num_pages + page) * cfg.page_tokens + off)
                * blocks_per_token + static_cast<uint64_t>(h) * kBlocksPerHead;
            for (uint32_t blk = 0; blk < kBlocksPerHead; ++blk) {
                const uint64_t bi = token_block_base + blk;
                const float gks = f32_from_bf16(got_k_scale[bi]);
                const float gvs = f32_from_bf16(got_v_scale[bi]);
                const float rks = f32_from_bf16(k_scale[bi]);
                const float rvs = f32_from_bf16(v_scale[bi]);
                for (int i = 0; i < 32; ++i) {
                    const float x = k_in[row_base + blk * 32u + static_cast<size_t>(i)];
                    const float y = v_in[row_base + blk * 32u + static_cast<size_t>(i)];
                    const uint8_t kb = got_k_code[bi * 16 + static_cast<size_t>(i & 15)];
                    const uint8_t vb = got_v_code[bi * 16 + static_cast<size_t>(i & 15)];
                    const uint32_t kc = (i < 16) ? (kb & 0xFu) : (kb >> 4u);
                    const uint32_t vc = (i < 16) ? (vb & 0xFu) : (vb >> 4u);
                    const float kdec = gks * ::ps::quantization::psq::cb10_code_to_f32(static_cast<uint8_t>(kc));
                    const float vdec = gvs * ::ps::quantization::psq::cb10_code_to_f32(static_cast<uint8_t>(vc));
                    const float kref = rks * ::ps::quantization::psq::cb10_code_to_f32(
                                                  static_cast<uint8_t>((k_code[bi * 16 + static_cast<size_t>(i & 15)] >> ((i < 16) ? 0u : 4u)) & 0xFu));
                    const float vref = rvs * ::ps::quantization::psq::cb10_code_to_f32(
                                                  static_cast<uint8_t>((v_code[bi * 16 + static_cast<size_t>(i & 15)] >> ((i < 16) ? 0u : 4u)) & 0xFu));
                    if (std::fabs(kdec - kref) > 1e-3f * std::fmax(1.0f, std::fabs(kref)) ||
                        std::fabs(vdec - vref) > 1e-3f * std::fmax(1.0f, std::fabs(vref))) {
                        recon_ok = false;
                        break;
                    }
                }
                if (!recon_ok) break;
            }
        }
    }
    if (!recon_ok) std::printf("  (%s reconstruction mismatch)\n", label);
    check(recon_ok, "psq4 gpu reconstruction matches reference");

    free_dev(d);
    return sentinel_ok && recon_ok;
}

}  // namespace psq4_test

}  // namespace

int main() {
    Config c;
    c.rows = 8;
    run_case(c, "bf16 real geometry rows=8 multi-seq page-boundary");
    c.rows = 4;
    c.multi_seq = true;
    c.max_sequences = 4;
    run_case(c, "bf16 multi-sequence slot->page rows=4");
    c = Config{};
    c.rows = 1;
    run_case(c, "bf16 decode rows=1");
    c = Config{};
    c.rows = 16;
    c.kv_heads = 1;
    c.head_dim = 64;
    c.page_tokens = 4;
    c.num_pages = 4;
    c.num_layers = 2;
    c.layer = 1;
    c.max_sequences = 8;
    run_case(c, "bf16 small geometry hd=64 page=4 rows=16");
    c = Config{};
    c.rows = 8;
    c.fp8 = true;
    run_case(c, "fp8 real geometry rows=8 multi-seq");
    c = Config{};
    c.rows = 1;
    c.fp8 = true;
    run_case(c, "fp8 decode rows=1");

    for (int est = 0; est < 2; ++est) {
        const auto e = est == 0 ? psq4_test::Est::MaxAbs : psq4_test::Est::Lsq1;
        const char* en = est == 0 ? "maxabs" : "lsq1";
        c = Config{};
        c.rows = 8;
        char label[96];
        std::snprintf(label, sizeof label, "psq4 %s random rows=8 page-boundary", en);
        psq4_test::run_psq4_case(c, label, 0, false, e);
        c = Config{};
        c.rows = 4;
        c.max_sequences = 4;
        std::snprintf(label, sizeof label, "psq4 %s multi-sequence rows=4", en);
        psq4_test::run_psq4_case(c, label, 0, true, e);
        c = Config{};
        c.rows = 3;
        std::snprintf(label, sizeof label, "psq4 %s all-zero", en);
        psq4_test::run_psq4_case(c, label, 1, false, e);
        c = Config{};
        c.rows = 3;
        std::snprintf(label, sizeof label, "psq4 %s positive-only", en);
        psq4_test::run_psq4_case(c, label, 2, false, e);
        c = Config{};
        c.rows = 3;
        std::snprintf(label, sizeof label, "psq4 %s negative-only", en);
        psq4_test::run_psq4_case(c, label, 3, false, e);
        c = Config{};
        c.rows = 3;
        std::snprintf(label, sizeof label, "psq4 %s symmetric", en);
        psq4_test::run_psq4_case(c, label, 4, false, e);
        c = Config{};
        c.rows = 3;
        std::snprintf(label, sizeof label, "psq4 %s strong-outlier", en);
        psq4_test::run_psq4_case(c, label, 5, false, e);
        c = Config{};
        c.rows = 3;
        std::snprintf(label, sizeof label, "psq4 %s tiny-values", en);
        psq4_test::run_psq4_case(c, label, 6, false, e);
        c = Config{};
        c.rows = 3;
        std::snprintf(label, sizeof label, "psq4 %s alternating-sign", en);
        psq4_test::run_psq4_case(c, label, 7, false, e);
    }

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
