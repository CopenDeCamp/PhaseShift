#pragma once

#include <cstddef>
#include <cstring>

namespace ps {
namespace qwen35 {
namespace runtime {

template <typename T>
T mcu_plan_value() {
    T value{};
    auto* bytes = reinterpret_cast<volatile unsigned char*>(&value);
    for (std::size_t i = 0; i < sizeof(T); ++i) bytes[i] = 0;
    return value;
}

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
