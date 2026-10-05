#include <phaseshift/models/model_source.h>

#include <phaseshift/resident/resident_client.h>
#include <phaseshift/resident/resident_model_key.h>

#include <cstdio>
#include <cstdlib>
#include <utility>

namespace ps {
namespace models {
namespace {

std::uint32_t qwen_load_flags(const qwen35::Qwen35LoadOptions& options) {
    return resident::resident_load_flags(options);
}

class SafetensorsModelSource final : public ModelSource {
 public:
    explicit SafetensorsModelSource(ModelLoadContext context)
        : context_(std::move(context)) {}

    bool resident() const noexcept override { return false; }
    std::size_t persistent_bytes() const noexcept override { return 0; }

    Result<qwen35::Qwen35Model> load_qwen35(
        gpu::GpuArena& arena,
        hipStream_t stream,
        const qwen35::Qwen35LoadOptions& options) override {
        if (context_.tp_size > 1) {
            return qwen35::Qwen35Model::load_tensor_parallel_from_safetensors(
                context_.qwen_model_dir, context_.tp_size, context_.tp_rank, arena,
                stream, options);
        }
        return qwen35::Qwen35Model::load_from_safetensors(
            context_.qwen_model_dir, arena, stream, options);
    }

    Result<qwen35::dflash2::DFlash2Weights> load_dflash2(
        const qwen35::dflash2::DFlash2Config& config,
        gpu::GpuArena& arena,
        hipStream_t stream,
        const ps::weights::WeightLoadOptions& options) override {
        return qwen35::dflash2::load_dflash2_weights(
            context_.dflash2_model_dir, config, arena, stream, options);
    }

 private:
    ModelLoadContext context_;
};

class ResidentModelSource final : public ModelSource {
 public:
    ResidentModelSource(resident::ResidentModelAttachment target,
                        resident::ResidentModelAttachment draft)
        : target_(std::move(target)), draft_(std::move(draft)) {}

    bool resident() const noexcept override { return true; }

    std::size_t persistent_bytes() const noexcept override
    {
        std::size_t out = target_.persistent_bytes();
        if (draft_.persistent_bytes() > out) {
            out = draft_.persistent_bytes();
        }
        const std::size_t own = target_.resident_bytes() + draft_.resident_bytes();
        return own > out ? own : out;
    }

    Result<qwen35::Qwen35Model> load_qwen35(
        gpu::GpuArena&, hipStream_t, const qwen35::Qwen35LoadOptions&) override {
        if (!target_.valid()) {
            return Status::invalid_state("resident target model not attached",
                                         __FILE__, __LINE__);
        }
        return target_.take_qwen35();
    }

    Result<qwen35::dflash2::DFlash2Weights> load_dflash2(
        const qwen35::dflash2::DFlash2Config&, gpu::GpuArena&, hipStream_t,
        const ps::weights::WeightLoadOptions&) override {
        if (!draft_.valid()) {
            return Status::invalid_state("resident dflash2 model not attached",
                                         __FILE__, __LINE__);
        }
        return draft_.take_dflash2(nullptr);
    }

    resident::ResidentModelAttachment& target_attachment() { return target_; }
    resident::ResidentModelAttachment& draft_attachment() { return draft_; }

 private:
    resident::ResidentModelAttachment target_;
    resident::ResidentModelAttachment draft_;
};

}

std::optional<std::string> model_host_socket_from_env() {
    const char* disable = std::getenv(kDisableResidentEnv);
    if (disable != nullptr && disable[0] != '\0' &&
        std::string(disable) != "0") {
        return std::nullopt;
    }
    const char* socket = std::getenv(kModelHostSocketEnv);
    if (socket == nullptr || socket[0] == '\0') {
        return std::nullopt;
    }
    return std::string(socket);
}

std::size_t ModelSource::ephemeral_bytes(std::size_t requested_bytes) const noexcept
{
    const std::size_t persistent = persistent_bytes();
    if (persistent == 0) {
        return requested_bytes;
    }
    if (requested_bytes <= persistent) {
        return 0;
    }
    return requested_bytes - persistent;
}

Result<std::unique_ptr<ModelSource>> ModelSource::create(
    const ModelLoadContext& context) {
    if (!context.model_host_socket.has_value() ||
        context.model_host_socket->empty()) {
        return std::unique_ptr<ModelSource>(new SafetensorsModelSource(context));
    }

    const int device = context.device;
    resident::ResidentModelAttachment target;
    resident::ResidentModelAttachment draft;

    if (!context.qwen_model_dir.empty()) {
        const auto key = resident::ResidentModelKey::make(
            context.qwen_model_dir, resident::ModelKind::Qwen35Target,
            qwen_load_flags(context.qwen_options), context.tp_size,
            context.tp_rank, device);
        auto attachment = resident::acquire_resident_model(
            key, device, *context.model_host_socket);
        if (!attachment.ok()) {
            return attachment.status();
        }
        target = attachment.release();
    }

    if (!context.dflash2_model_dir.empty()) {
        const auto key = resident::ResidentModelKey::make(
            context.dflash2_model_dir, resident::ModelKind::DFlash2Draft,
            resident::resident_load_flags(context.dflash2_options), 1, 0, device);
        auto attachment = resident::acquire_resident_model(
            key, device, *context.model_host_socket);
        if (!attachment.ok()) {
            return attachment.status();
        }
        draft = attachment.release();
    }

    std::unique_ptr<ModelSource> source(
        new ResidentModelSource(std::move(target), std::move(draft)));
    std::fprintf(stderr, "model source: resident persistent_bytes=%zu socket=%s\n",
                 source->persistent_bytes(), context.model_host_socket->c_str());
    return source;
}

}
}
