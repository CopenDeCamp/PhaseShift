#pragma once

#include <phaseshift/core/memory/tensor.h>

namespace ps {
namespace qwen35 {

struct GDNState {
    gpu::Tensor conv_state;
    gpu::Tensor recurrent_state;
};

}
}
