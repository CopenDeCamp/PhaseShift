#pragma once
#include <phaseshift/quantization/fpx/types.h>
#include <cstdint>
#include <optional>
#include <string_view>

namespace ps::quantization::fpx {

WeightEncoding choose_weight_encoding(
    FpxPreset preset,
    Architecture architecture,
    TensorRole role,
    uint32_t layer_index,
    uint32_t num_layers);

const char* to_string(WeightEncoding e);
const char* to_string(FpxPreset p);
const char* to_string(FpxLayout l);
const char* to_string(TensorRole r);
const char* to_string(Architecture a);

std::optional<TensorRole> try_parse_tensor_role(std::string_view s);
std::optional<WeightEncoding> try_parse_weight_encoding(std::string_view s);
std::optional<FpxLayout> try_parse_fpx_layout(std::string_view s);

}  // namespace ps::quantization::fpx
