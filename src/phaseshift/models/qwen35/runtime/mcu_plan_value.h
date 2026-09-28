#pragma once

#include <cstring>

namespace ps {
namespace qwen35 {
namespace runtime {

template <typename T>
T mcu_plan_value() {
    T value{};
    std::memset(reinterpret_cast<unsigned char*>(&value), 0, sizeof(T));
    return value;
}

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
