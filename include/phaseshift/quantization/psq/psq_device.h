#pragma once

#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__) || defined(__HIP_DEVICE_COMPILE__)
#define PSQ_DEVICE_HD __host__ __device__
#else
#define PSQ_DEVICE_HD
#endif

namespace ps::quantization::psq {

inline constexpr float kPsq4Magnitudes[8] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 10.0f,
};

PSQ_DEVICE_HD inline constexpr float cb10_code_to_f32(uint8_t code) {
    const float mag = kPsq4Magnitudes[code & 7u];
    return (code & 8u) ? -mag : mag;
}

PSQ_DEVICE_HD inline constexpr uint8_t cb10_magnitude_index(float a) {
    return (a <= 0.5f) ? 0u
         : (a <= 1.5f) ? 1u
         : (a <= 2.5f) ? 2u
         : (a <= 3.5f) ? 3u
         : (a <= 5.0f) ? 4u
         : (a <= 7.0f) ? 5u
         : (a <= 9.0f) ? 6u
                       : 7u;
}

PSQ_DEVICE_HD inline constexpr uint8_t cb10_encode(float x, float inv_scale) {
    const float a = (x < 0.0f ? -x : x) * inv_scale;
    return static_cast<uint8_t>(cb10_magnitude_index(a) | (x < 0.0f ? 8u : 0u));
}

}  // namespace ps::quantization::psq
