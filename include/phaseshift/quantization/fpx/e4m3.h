#pragma once
#include <cstdint>

namespace ps::quantization::fpx {

float e4m3_decode_f32(uint8_t x);
uint8_t e4m3_encode_u8(float x);

}  // namespace ps::quantization::fpx
