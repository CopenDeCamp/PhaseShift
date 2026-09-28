#pragma once

namespace ps::kernel::detail {

__device__ __forceinline__ ProgramStatus exec_linear_weight(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    ResolvedValue in{};
    ResolvedValue out{};
    if (!resolve_value(env, b.input_slots[0], in))
        return ProgramStatus::INVALID_BINDING;
    if (!resolve_value(env, b.output_slots[0], out))
        return ProgramStatus::INVALID_BINDING;
    if (b.weight_index >= env.program->weight_slot_count ||
        env.program->weight_table == nullptr)
        return ProgramStatus::INVALID_BINDING;
    const auto* wtable = static_cast<const WeightSlot*>(env.program->weight_table);
    const WeightSlot e = wtable[b.weight_index];
    const uint32_t out_features = e.rows;
    const uint32_t kdim = e.cols;
    if (out_features != out.features || kdim != in.features)
        return ProgramStatus::INVALID_BINDING;
    if (out.dtype != ValueDType::BF16 && out.dtype != ValueDType::F32)
        return ProgramStatus::INVALID_BINDING;
    if (e.encoding == 0u) {
        if (e.device_ptr == nullptr || e.codes != nullptr || e.scales != nullptr)
            return ProgramStatus::INVALID_PROGRAM;
        const bf16* w = static_cast<const bf16*>(e.device_ptr);
        const uint32_t rows = value_rows(env, out);
        const size_t total = static_cast<size_t>(rows) * out_features;
        const uint32_t stride = global_stride(env);
        for (size_t idx = global_tid(env); idx < total; idx += stride) {
            const uint32_t m = static_cast<uint32_t>(idx / out_features);
            const uint32_t o = static_cast<uint32_t>(idx % out_features);
            float acc = 0.f;
            const bf16* wrow = w + static_cast<size_t>(o) * kdim;
            const size_t xin_base = static_cast<size_t>(m) * in.row_stride;
            for (uint32_t kk = 0; kk < kdim; ++kk)
                acc += __bfloat162float(wrow[kk]) * load_elem(in, xin_base + kk);
            store_elem(out, idx, acc);
        }
        return ProgramStatus::COMPLETE;
    }
    return ProgramStatus::INVALID_PROGRAM;
}

__device__ __forceinline__ float psq_e8m0_to_f32(uint8_t byte) {
    return (byte == 0xFFu) ? 0.0f : ldexpf(1.0f, static_cast<int>(byte) - 127);
}

__device__ __forceinline__ float psq_e2m1_nib_to_e4m3(uint8_t nibble) {
    const uint8_t mags[8] = {0x00u, 0x30u, 0x38u, 0x3Cu, 0x40u, 0x44u, 0x48u, 0x4Cu};
    const uint8_t mag = mags[nibble & 7u];
    const uint8_t code = (nibble & 8u) ? static_cast<uint8_t>(mag | 0x80u) : mag;
    return ::ps::quantization::fpx::device::e4m3_decode_f32(code);
}

__device__ __forceinline__ ProgramStatus exec_linear_psq(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    ResolvedValue out{};
    if (!resolve_value(env, b.output_slots[0], out))
        return ProgramStatus::INVALID_BINDING;
    if (b.weight_index >= env.program->weight_slot_count ||
        env.program->weight_table == nullptr)
        return ProgramStatus::INVALID_BINDING;
    if (b.workspace_slot >= env.program->workspace_range_count)
        return ProgramStatus::INVALID_PROGRAM;
    if (b.activation_kp < 32u || (b.activation_kp & 31u) != 0u ||
        b.activation_max_rows == 0u)
        return ProgramStatus::INVALID_PROGRAM;
    const auto* wtable = static_cast<const WeightSlot*>(env.program->weight_table);
    const WeightSlot e = wtable[b.weight_index];
    const bool psq4 = e.encoding == 3u;
    const bool psq8 = e.encoding == 4u;
    const bool fp8 = e.encoding == 5u;
    const bool mxfp4 = e.encoding == 6u;
    if (!psq4 && !psq8 && !fp8 && !mxfp4)
        return ProgramStatus::INVALID_PROGRAM;
    if (e.codes == nullptr || e.scales == nullptr)
        return ProgramStatus::INVALID_PROGRAM;
    if (e.weight_scale_group != (fp8 ? 128u : 32u))
        return ProgramStatus::INVALID_PROGRAM;
    if (e.rows != out.features)
        return ProgramStatus::INVALID_BINDING;
    if (out.dtype != ValueDType::BF16 && out.dtype != ValueDType::F32)
        return ProgramStatus::INVALID_BINDING;
    const uint32_t kdim = e.cols;
    const uint32_t wkp = e.k_padded;
    if (wkp < kdim || (wkp & (fp8 ? 127u : 31u)) != 0u)
        return ProgramStatus::INVALID_PROGRAM;
    const uint32_t sstride = e.storage_scale_stride_bytes;
    if (fp8) {
        if (sstride != (wkp >> 7) * 4u)
            return ProgramStatus::INVALID_PROGRAM;
    } else if (mxfp4) {
        if (sstride != (wkp >> 5))
            return ProgramStatus::INVALID_PROGRAM;
    } else {
        if (sstride < wkp || (sstride & 15u) != 0u)
            return ProgramStatus::INVALID_PROGRAM;
    }
    const uint32_t akp = b.activation_kp;
    if (akp != wkp)
        return ProgramStatus::INVALID_PROGRAM;
    const uint32_t nb = wkp >> 5;
    const uint32_t code_stride = b.activation_code_stride;
    const uint32_t scale_stride = b.activation_scale_stride;
    if (code_stride < akp || (code_stride & 15u) != 0u)
        return ProgramStatus::INVALID_PROGRAM;
    if (scale_stride < 4u || (scale_stride & 15u) != 0u)
        return ProgramStatus::INVALID_PROGRAM;
    const uint64_t ws_off = env.program->workspace_ranges[b.workspace_slot].offset;
    const uint8_t* acodes = reinterpret_cast<const uint8_t*>(env.workspace_base + ws_off);
    const uint8_t* asbase = reinterpret_cast<const uint8_t*>(env.workspace_base + ws_off +
        a8_ws_scales_offset(code_stride, b.activation_max_rows));
    const uint32_t rows = value_rows(env, out);
    if (rows > b.activation_max_rows)
        return ProgramStatus::INVALID_PROGRAM;
    if (rows == 0u)
        return ProgramStatus::COMPLETE;
    const uint8_t* wcodes = static_cast<const uint8_t*>(e.codes);
    const uint8_t* wscales = static_cast<const uint8_t*>(e.scales);
    const uint32_t out_features = e.rows;
    const bool nibble_weight = psq4 || mxfp4;
    const uint32_t tile_code_bytes = nibble_weight ? nb * 256u : nb * 512u;
    const uint32_t block_code_bytes = nibble_weight ? 256u : 512u;
    const size_t total = static_cast<size_t>(rows) * out_features;
    const uint32_t stride = global_stride(env);
    for (size_t idx = global_tid(env); idx < total; idx += stride) {
        const uint32_t m = static_cast<uint32_t>(idx / out_features);
        const uint32_t o = static_cast<uint32_t>(idx % out_features);
        const uint8_t* wrow = wcodes +
            static_cast<size_t>(o >> 4) * tile_code_bytes +
            static_cast<size_t>(o & 15u) * 16u;
        const uint8_t* wsrow = wscales +
            static_cast<size_t>(o >> 4) * sstride + static_cast<size_t>(o & 15u) * 2u;
        const float* wsrow_fp8 = reinterpret_cast<const float*>(wscales) +
            static_cast<size_t>(o >> 7) * (wkp >> 7);
        const uint8_t* wsrow_mx = wscales + static_cast<size_t>(o) * sstride;
        const uint8_t* arow = acodes +
            static_cast<size_t>(m >> 4) * (static_cast<size_t>(code_stride) * 16u) +
            static_cast<size_t>(m & 15u) * 16u;
        const float* asrow = reinterpret_cast<const float*>(
            asbase + static_cast<size_t>(m) * scale_stride);
        const float as0 = asrow[0];
        float acc = 0.f;
        for (uint32_t ib = 0; ib < nb; ++ib) {
            float ws;
            if (fp8) {
                ws = wsrow_fp8[ib >> 2];
            } else if (mxfp4) {
                ws = psq_e8m0_to_f32(wsrow_mx[ib]);
            } else {
                const uint16_t sb =
                    static_cast<uint16_t>(wsrow[static_cast<size_t>(ib) * 32u]) |
                    (static_cast<uint16_t>(wsrow[static_cast<size_t>(ib) * 32u + 1u]) << 8u);
                ws = __bfloat162float(__ushort_as_bfloat16(sb));
            }
            const uint8_t* wp = wrow + static_cast<size_t>(ib) * block_code_bytes;
            const uint8_t* ap = arow + static_cast<size_t>(ib) * 512u;
            float p = 0.f;
            for (uint32_t j = 0; j < 16u; ++j) {
                float w0;
                float w1;
                if (psq4) {
                    const uint8_t byte = wp[j];
                    w0 = psq_cb10_value(byte & 0x0fu);
                    w1 = psq_cb10_value(byte >> 4u);
                } else if (mxfp4) {
                    const uint8_t byte = wp[j];
                    w0 = psq_e2m1_nib_to_e4m3(static_cast<uint8_t>(byte & 0x0Fu));
                    w1 = psq_e2m1_nib_to_e4m3(static_cast<uint8_t>(byte >> 4u));
                } else {
                    w0 = ::ps::quantization::fpx::device::e4m3_decode_f32(wp[j]);
                    w1 = ::ps::quantization::fpx::device::e4m3_decode_f32(wp[256u + j]);
                }
                p += w0 * ::ps::quantization::fpx::device::e4m3_decode_f32(ap[j]);
                p += w1 * ::ps::quantization::fpx::device::e4m3_decode_f32(ap[256u + j]);
            }
            acc += p * ws * as0;
        }
        store_elem(out, idx, acc);
    }
    return ProgramStatus::COMPLETE;
}

__device__ __forceinline__ ProgramStatus exec_activation_quantize_fp8(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    ResolvedValue in{};
    if (!resolve_value(env, b.input_slots[0], in))
        return ProgramStatus::INVALID_BINDING;
    if (in.dtype != ValueDType::BF16)
        return ProgramStatus::INVALID_PROGRAM;
    if (b.workspace_slot >= env.program->workspace_range_count)
        return ProgramStatus::INVALID_PROGRAM;
    if (in.features == 0u)
        return ProgramStatus::INVALID_PROGRAM;
    if (b.activation_kp < in.features || (b.activation_kp & 31u) != 0u ||
        b.activation_max_rows == 0u)
        return ProgramStatus::INVALID_PROGRAM;
    const uint32_t rows = value_rows(env, in);
    if (rows > b.activation_max_rows)
        return ProgramStatus::INVALID_PROGRAM;
    if (rows == 0u)
        return ProgramStatus::COMPLETE;
    const uint32_t k = in.features;
    const uint32_t kp = b.activation_kp;
    const uint32_t code_stride = b.activation_code_stride;
    const uint32_t scale_stride = b.activation_scale_stride;
    if (code_stride < kp || (code_stride & 15u) != 0u)
        return ProgramStatus::INVALID_PROGRAM;
    if (scale_stride < 4u || (scale_stride & 15u) != 0u)
        return ProgramStatus::INVALID_PROGRAM;
    const uint64_t ws_off = env.program->workspace_ranges[b.workspace_slot].offset;
    int8_t* acodes = reinterpret_cast<int8_t*>(env.workspace_base + ws_off);
    float* ascales = reinterpret_cast<float*>(env.workspace_base + ws_off +
        a8_ws_scales_offset(code_stride, b.activation_max_rows));
    const uint32_t stride = global_stride(env);
    const bf16* xin = static_cast<const bf16*>(in.ptr);
    for (uint32_t m = global_tid(env); m < rows; m += stride) {
        float peak = 0.f;
        for (uint32_t i = 0; i < k; ++i) {
            const float v =
                __bfloat162float(xin[static_cast<size_t>(m) * in.row_stride + i]);
            if (std::isfinite(v) && std::fabs(v) > peak) peak = std::fabs(v);
        }
        const float scale = peak > 0.f ? peak / 127.0f : 0.f;
        const float inv = scale > 0.f ? 1.0f / scale : 0.f;
        int8_t* codes = acodes +
            static_cast<size_t>(m >> 4) * (code_stride * 16u) +
            static_cast<size_t>(m & 15u) * 16u;
        for (uint32_t i = 0; i < kp; ++i) {
            float v = 0.f;
            if (i < k) {
                const float x =
                    __bfloat162float(xin[static_cast<size_t>(m) * in.row_stride + i]);
                v = std::isfinite(x) ? x : 0.f;
            }
            int8_t c = 0;
            if (scale > 0.f) {
                float ci = std::nearbyint(v * inv);
                if (ci > 127.0f) ci = 127.0f;
                if (ci < -127.0f) ci = -127.0f;
                c = static_cast<int8_t>(ci);
            }
            codes[static_cast<size_t>(i >> 5) * 512u +
                  static_cast<size_t>((i & 31u) >> 4) * 256u + (i & 15u)] = c;
        }
        ascales[static_cast<size_t>(m) * (scale_stride / 4u)] = scale;
    }
    return ProgramStatus::COMPLETE;
}

__device__ __forceinline__ ProgramStatus exec_activation_quantize_w4a8(
    ModelDispatchEnv& env, const DispatchBinding& b) {
    ResolvedValue in{};
    if (!resolve_value(env, b.input_slots[0], in))
        return ProgramStatus::INVALID_BINDING;
    if (in.dtype != ValueDType::BF16)
        return ProgramStatus::INVALID_PROGRAM;
    if (b.workspace_slot >= env.program->workspace_range_count)
        return ProgramStatus::INVALID_PROGRAM;
    if (in.features == 0u)
        return ProgramStatus::INVALID_PROGRAM;
    if (b.activation_kp < in.features || (b.activation_kp & 31u) != 0u ||
        b.activation_max_rows == 0u)
        return ProgramStatus::INVALID_PROGRAM;
    const uint32_t rows = value_rows(env, in);
    if (rows > b.activation_max_rows)
        return ProgramStatus::INVALID_PROGRAM;
    if (rows == 0u)
        return ProgramStatus::COMPLETE;
    const uint32_t k = in.features;
    const uint32_t kp = b.activation_kp;
    const uint32_t code_stride = b.activation_code_stride;
    const uint32_t scale_stride = b.activation_scale_stride;
    if (code_stride < kp || (code_stride & 15u) != 0u)
        return ProgramStatus::INVALID_PROGRAM;
    if (scale_stride < 4u || (scale_stride & 15u) != 0u)
        return ProgramStatus::INVALID_PROGRAM;
    const uint64_t ws_off = env.program->workspace_ranges[b.workspace_slot].offset;
    uint8_t* acodes = reinterpret_cast<uint8_t*>(env.workspace_base + ws_off);
    uint8_t* ascales = reinterpret_cast<uint8_t*>(env.workspace_base + ws_off +
        a8_ws_scales_offset(code_stride, b.activation_max_rows));
    const uint32_t stride = global_stride(env);
    const bf16* xin = static_cast<const bf16*>(in.ptr);
    for (uint32_t m = global_tid(env); m < rows; m += stride) {
        float peak = 0.f;
        for (uint32_t i = 0; i < k; ++i) {
            const float v =
                __bfloat162float(xin[static_cast<size_t>(m) * in.row_stride + i]);
            if (std::isfinite(v) && std::fabs(v) > peak) peak = std::fabs(v);
        }
        const float scale = peak > 0.f ? peak / 448.0f : 0.f;
        const float inv = scale > 0.f ? 1.0f / scale : 0.f;
        uint8_t* codes = acodes +
            static_cast<size_t>(m >> 4) * (code_stride * 16u) +
            static_cast<size_t>(m & 15u) * 16u;
        for (uint32_t i = 0; i < kp; ++i) {
            float v = 0.f;
            if (i < k) {
                const float x =
                    __bfloat162float(xin[static_cast<size_t>(m) * in.row_stride + i]);
                v = std::isfinite(x) ? x : 0.f;
            }
            codes[static_cast<size_t>(i >> 5) * 512u +
                  static_cast<size_t>((i & 31u) >> 4) * 256u + (i & 15u)] =
                scale > 0.f
                    ? ::ps::quantization::fpx::device::e4m3_encode_u8(v * inv)
                    : 0u;
        }
        *reinterpret_cast<float*>(ascales + static_cast<size_t>(m) * scale_stride) = scale;
    }
    return ProgramStatus::COMPLETE;
}

}  // namespace ps::kernel::detail
