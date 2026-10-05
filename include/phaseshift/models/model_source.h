#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/resident/resident_client.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace ps {
namespace models {

constexpr const char* kModelHostSocketEnv = resident::kSocketEnv;
constexpr const char* kDisableResidentEnv = resident::kDisableEnv;

std::optional<std::string> model_host_socket_from_env();

struct ModelLoadContext {
    std::string qwen_model_dir;
    std::string dflash2_model_dir;
    qwen35::Qwen35LoadOptions qwen_options;
    ps::weights::WeightLoadOptions dflash2_options;
    int device = 0;
    std::uint32_t tp_size = 1;
    std::uint32_t tp_rank = 0;
    std::optional<std::string> model_host_socket;
};

class ModelSource {
 public:
    static Result<std::unique_ptr<ModelSource>> create(const ModelLoadContext& context);

    ModelSource() = default;
    virtual ~ModelSource() = default;

    ModelSource(const ModelSource&) = delete;
    ModelSource& operator=(const ModelSource&) = delete;

    virtual bool resident() const noexcept = 0;
    virtual std::size_t persistent_bytes() const noexcept = 0;

    std::size_t ephemeral_bytes(std::size_t requested_bytes) const noexcept;

    virtual Result<qwen35::Qwen35Model> load_qwen35(
        gpu::GpuArena& arena,
        hipStream_t stream,
        const qwen35::Qwen35LoadOptions& options) = 0;

    virtual Result<qwen35::dflash2::DFlash2Weights> load_dflash2(
        const qwen35::dflash2::DFlash2Config& config,
        gpu::GpuArena& arena,
        hipStream_t stream,
        const ps::weights::WeightLoadOptions& options) = 0;
};

}
}
