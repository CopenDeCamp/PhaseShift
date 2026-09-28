#include <phaseshift/quantization/offline/kld_shadow_model.h>
#include <phaseshift/io/safetensors_writer.h>
#include <phaseshift/quantization/fpx/layout.h>
#include <phaseshift/quantization/fpx/quantized_model_reader.h>
#include <phaseshift/quantization/fpx/runtime_resolution.h>
#include <phaseshift/quantization/offline/model_fingerprint.h>
#include <phaseshift/quantization/psq/psq.h>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <linux/fs.h>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

namespace ps::quantization::fpx {

namespace {

namespace fs = std::filesystem;

struct MaterializedTensor {
    std::string name;
    QuantizedTensorView view;
    std::size_t rows = 0;
    std::size_t k = 0;
    std::size_t padded_k = 0;
};

uint16_t fp32_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

Status filesystem_failure(const std::string& operation, const std::error_code& error) {
    return Status::invalid_argument((operation + ": " + error.message()).c_str(), __FILE__, __LINE__);
}

Status remove_path(const fs::path& path) {
    std::error_code error;
    fs::remove_all(path, error);
    if (error) return filesystem_failure("remove " + path.string(), error);
    return Status::make_ok();
}

Status cleanup_after_failure(Status primary, const fs::path& staging) {
    const Status cleanup = remove_path(staging);
    if (primary.ok()) return cleanup;
    if (cleanup.ok()) return primary;
    const std::string message = primary.message() + "; cleanup failed: " + cleanup.message();
    return Status::invalid_state(message.c_str(), __FILE__, __LINE__);
}

bool is_path_prefix(const fs::path& prefix, const fs::path& path) {
    auto prefix_it = prefix.begin();
    auto path_it = path.begin();
    for (; prefix_it != prefix.end() && path_it != path.end(); ++prefix_it, ++path_it) {
        if (*prefix_it != *path_it) return false;
    }
    return prefix_it == prefix.end();
}

Result<fs::path> weakly_canonical_path(const fs::path& path) {
    std::error_code error;
    fs::path canonical = fs::weakly_canonical(path, error);
    if (error) return filesystem_failure("canonicalize " + path.string(), error);
    return canonical;
}

Status validate_output_path(const fs::path& input, const fs::path& quantized, const fs::path& output) {
    auto input_path = weakly_canonical_path(input);
    if (!input_path.ok()) return input_path.status();
    auto quantized_path = weakly_canonical_path(quantized);
    if (!quantized_path.ok()) return quantized_path.status();
    auto output_path = weakly_canonical_path(output);
    if (!output_path.ok()) return output_path.status();
    const fs::path& resolved_output = output_path.value();
    if (is_path_prefix(input_path.value(), resolved_output) || is_path_prefix(resolved_output, input_path.value()) ||
        is_path_prefix(quantized_path.value(), resolved_output) || is_path_prefix(resolved_output, quantized_path.value())) {
        return Status::invalid_argument("shadow output overlaps input or quantized model", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status validate_directory(const fs::path& path, const char* name) {
    std::error_code error;
    if (!fs::is_directory(path, error)) {
        if (error) return filesystem_failure(std::string(name) + " directory", error);
        return Status::invalid_argument((std::string(name) + " directory not found").c_str(), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status checked_shape(const QuantizedTensorView& view, std::size_t& rows, std::size_t& k,
                     std::size_t& padded_k) {
    if (view.logical_shape.empty()) return Status::invalid_argument("shadow tensor shape is empty", __FILE__, __LINE__);
    rows = 1;
    for (std::size_t index = 0; index + 1 < view.logical_shape.size(); ++index) {
        const int64_t dimension = view.logical_shape[index];
        if (dimension <= 0 || static_cast<uint64_t>(dimension) > std::numeric_limits<std::size_t>::max() / rows) {
            return Status::invalid_argument("shadow tensor shape is invalid", __FILE__, __LINE__);
        }
        rows *= static_cast<std::size_t>(dimension);
    }
    const int64_t logical_k = view.logical_shape.back();
    if (logical_k <= 0) return Status::invalid_argument("shadow tensor K dimension is invalid", __FILE__, __LINE__);
    k = static_cast<std::size_t>(logical_k);
    if (view.k_padded <= 0) return Status::invalid_argument("shadow tensor padded K is invalid", __FILE__, __LINE__);
    padded_k = static_cast<std::size_t>(view.k_padded);
    if (rows > std::numeric_limits<std::size_t>::max() / k) {
        return Status::overflow("shadow tensor element count overflow", __FILE__, __LINE__);
    }
    const std::size_t elements = rows * k;
    if (view.encoding == QuantizedEncoding::Bf16) {
        if (elements > std::numeric_limits<std::size_t>::max() / sizeof(uint16_t) ||
            padded_k != k || !view.data.data || view.data.size != elements * sizeof(uint16_t)) {
            return Status::invalid_argument("shadow BF16 tensor layout is invalid", __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    if (view.encoding != QuantizedEncoding::Psq4 &&
        view.encoding != QuantizedEncoding::Psq8) {
        return Status::invalid_argument("shadow tensor encoding is invalid", __FILE__, __LINE__);
    }
    if (padded_k < k || padded_k % kFpBlockSize != 0) {
        return Status::invalid_argument("shadow quantized tensor padded K is invalid", __FILE__, __LINE__);
    }
    const std::size_t blocks = padded_k / kFpBlockSize;
    const std::size_t code_bytes_per_block =
        view.encoding == QuantizedEncoding::Psq4 ? kPsq4CodesBytesPerBlock
        : kPsq8CodesBytesPerBlock;
    const std::size_t scale_bytes_per_block = kPsqScalesBytesPerBlock;
    if (blocks > std::numeric_limits<std::size_t>::max() / code_bytes_per_block ||
        rows > std::numeric_limits<std::size_t>::max() / (blocks * code_bytes_per_block) ||
        blocks > std::numeric_limits<std::size_t>::max() / scale_bytes_per_block ||
        rows > std::numeric_limits<std::size_t>::max() / (blocks * scale_bytes_per_block)) {
        return Status::overflow("shadow quantized tensor payload size overflow", __FILE__, __LINE__);
    }
    const std::size_t codes_per_row = blocks * code_bytes_per_block;
    const std::size_t scales_per_row = blocks * scale_bytes_per_block;
    if (!view.codes.data || !view.metadata1.data || view.codes.size != rows * codes_per_row ||
        view.metadata1.size != rows * scales_per_row) {
        return Status::invalid_argument("shadow quantized tensor payload is invalid", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Result<fs::path> create_staging_directory(const fs::path& output) {
    const std::string base = output.string() + ".partial." + std::to_string(static_cast<unsigned long long>(getpid()));
    for (unsigned int attempt = 0; attempt != 1000; ++attempt) {
        fs::path candidate = base + "." + std::to_string(attempt);
        std::error_code error;
        if (fs::exists(candidate, error)) {
            if (error) return filesystem_failure("check staging directory", error);
            continue;
        }
        if (fs::create_directories(candidate, error)) return candidate;
        if (error) return filesystem_failure("create staging directory", error);
    }
    return Status::invalid_state("cannot allocate unique shadow staging directory", __FILE__, __LINE__);
}

Status exchange_paths(const fs::path& first, const fs::path& second) {
#ifdef SYS_renameat2
    if (::syscall(SYS_renameat2, AT_FDCWD, first.c_str(), AT_FDCWD, second.c_str(), RENAME_EXCHANGE) != 0) {
        const int saved_errno = errno;
        return Status::invalid_argument(
            ("exchange shadow output: " + std::string(std::strerror(saved_errno))).c_str(), __FILE__, __LINE__);
    }
    return Status::make_ok();
#else
    return Status::unsupported("atomic shadow replacement is unavailable", __FILE__, __LINE__);
#endif
}

Status publish_staging(const fs::path& staging, const fs::path& output) {
    std::error_code error;
    const fs::file_status output_status = fs::symlink_status(output, error);
    if (error && error != std::errc::no_such_file_or_directory)
        return cleanup_after_failure(filesystem_failure("inspect shadow output", error), staging);
    if (output_status.type() == fs::file_type::not_found) {
        fs::rename(staging, output, error);
        if (!error) return Status::make_ok();
        return cleanup_after_failure(filesystem_failure("publish shadow output", error), staging);
    }

    const fs::path backup = output.string() + ".backup." +
        std::to_string(static_cast<unsigned long long>(getpid())) + ".0";
    const fs::file_status backup_status = fs::symlink_status(backup, error);
    if (error && error != std::errc::no_such_file_or_directory) {
        return cleanup_after_failure(filesystem_failure("inspect shadow backup", error), staging);
    }
    error.clear();
    if (backup_status.type() != fs::file_type::not_found) {
        return cleanup_after_failure(
            Status::invalid_argument("shadow backup path is occupied", __FILE__, __LINE__), staging);
    }

    Status exchange_status = exchange_paths(staging, output);
    if (!exchange_status.ok()) return cleanup_after_failure(exchange_status, staging);
    return remove_path(staging);
}

Status copy_source_file(const fs::path& source, const fs::path& destination) {
    std::error_code error;
    fs::copy_file(source, destination, fs::copy_options::overwrite_existing, error);
    if (error) return filesystem_failure("copy source file", error);
    return Status::make_ok();
}

Result<ShadowModelResult> materialize_shadow(const ShadowModelOptions& options) {
    if (options.input_dir.empty() || options.quantized_dir.empty() || options.output_dir.empty()) {
        return Status::invalid_argument("shadow model paths are required", __FILE__, __LINE__);
    }
    const fs::path input(options.input_dir);
    const fs::path quantized(options.quantized_dir);
    const fs::path output(options.output_dir);
    Status status = validate_directory(input, "input model");
    if (!status.ok()) return status;
    status = validate_directory(quantized, "quantized model");
    if (!status.ok()) return status;
    status = validate_output_path(input, quantized, output);
    if (!status.ok()) return status;
    const fs::path config = input / "config.json";
    std::error_code error;
    if (!fs::is_regular_file(config, error)) {
        if (error) return filesystem_failure("check source config", error);
        return Status::invalid_argument("source config.json not found", __FILE__, __LINE__);
    }

    auto reader_result = QuantizedModelReader::open(options.quantized_dir);
    if (!reader_result.ok()) return reader_result.status();
    QuantizedModelReader reader = reader_result.release();
    status = validate_quantized_manifest_contract(reader.manifest());
    if (!status.ok()) return status;
    const std::string fingerprint = compute_model_fingerprint(options.input_dir);
    if (fingerprint.empty() || reader.manifest().source_model_fingerprint != fingerprint) {
        return Status::invalid_argument("quantized model source fingerprint mismatch", __FILE__, __LINE__);
    }
    auto names_result = reader.list_logical_tensors();
    if (!names_result.ok()) return names_result.status();
    std::vector<MaterializedTensor> tensors;
    tensors.reserve(names_result.value().size());
    for (const std::string& name : names_result.value()) {
        auto view_result = reader.resolve(name);
        if (!view_result.ok()) return view_result.status();
        MaterializedTensor tensor;
        tensor.name = name;
        tensor.view = view_result.release();
        status = checked_shape(tensor.view, tensor.rows, tensor.k, tensor.padded_k);
        if (!status.ok()) return status;
        tensors.push_back(std::move(tensor));
    }

    auto staging_result = create_staging_directory(output);
    if (!staging_result.ok()) return staging_result.status();
    const fs::path staging = staging_result.release();
    try {
        status = copy_source_file(config, staging / "config.json");
        if (!status.ok()) return cleanup_after_failure(status, staging);
        auto writer_result = ps::io::SafetensorsWriter::create((staging / "model.safetensors-00001-of-00001.safetensors").string());
        if (!writer_result.ok()) return cleanup_after_failure(writer_result.status(), staging);
        ps::io::SafetensorsWriter writer = writer_result.release();
        for (const MaterializedTensor& tensor : tensors) {
            std::vector<std::size_t> shape;
            shape.reserve(tensor.view.logical_shape.size());
            for (const int64_t dimension : tensor.view.logical_shape) shape.push_back(static_cast<std::size_t>(dimension));
            status = writer.plan_tensor(tensor.name, ps::io::SType::BF16, shape);
            if (!status.ok()) return cleanup_after_failure(status, staging);
        }
        auto header = writer.write_header();
        if (!header.ok()) return cleanup_after_failure(header.status(), staging);
        ShadowModelResult result;
        result.shadow_dir = options.output_dir;
        result.source_model_fingerprint = fingerprint;
        result.total_tensors = tensors.size();
        for (const MaterializedTensor& tensor : tensors) {
            std::vector<uint16_t> output_data(tensor.rows * tensor.k);
            const QuantizedTensorView& view = tensor.view;
            if (view.encoding == QuantizedEncoding::Bf16) {
                std::memcpy(output_data.data(), view.data.data, view.data.size);
            } else {
                const std::size_t blocks = tensor.padded_k / kFpBlockSize;
                const std::size_t code_bytes_per_block =
                    view.encoding == QuantizedEncoding::Psq4 ? kPsq4CodesBytesPerBlock
                    : kPsq8CodesBytesPerBlock;
                const std::size_t scale_bytes_per_block = kPsqScalesBytesPerBlock;
                const std::size_t codes_per_row = blocks * code_bytes_per_block;
                const std::size_t scales_per_row = blocks * scale_bytes_per_block;
                std::vector<float> reconstructed(tensor.padded_k);
                for (std::size_t row = 0; row < tensor.rows; ++row) {
                    const auto* codes = static_cast<const uint8_t*>(view.codes.data) + row * codes_per_row;
                    const auto* scale_bytes = static_cast<const uint8_t*>(view.metadata1.data) + row * scales_per_row;
                    if (view.encoding == QuantizedEncoding::Psq4) {
                        psq::dequantize_psq4_row(
                            std::span<const uint8_t>(codes, codes_per_row),
                            std::span<const uint8_t>(scale_bytes, scales_per_row),
                            reconstructed, tensor.k, tensor.padded_k);
                    } else {
                        psq::dequantize_psq8_row(
                            std::span<const uint8_t>(codes, codes_per_row),
                            std::span<const uint8_t>(scale_bytes, scales_per_row),
                            reconstructed, tensor.k, tensor.padded_k);
                    }
                    for (std::size_t column = 0; column < tensor.k; ++column) {
                        output_data[row * tensor.k + column] = fp32_to_bf16(reconstructed[column]);
                    }
                }
                ++result.patched_tensors;
            }
            status = writer.write_tensor(tensor.name, output_data.data(), output_data.size() * sizeof(uint16_t));
            if (!status.ok()) return cleanup_after_failure(status, staging);
        }
        auto finished = writer.finish();
        if (!finished.ok()) return cleanup_after_failure(finished.status(), staging);
        std::ofstream metadata(staging / "phaseshift_qdq_shadow.json", std::ios::trunc);
        metadata << "{\"format\":\"phaseshift-fpx-qdq-shadow\",\"version\":1,\"source_model_fingerprint\":\""
                 << fingerprint << "\",\"semantics\":\"reference-dequantized shadow model; not native FPX execution\"}\n";
        if (!metadata) return cleanup_after_failure(Status::invalid_argument("cannot write shadow metadata", __FILE__, __LINE__), staging);
        metadata.flush();
        if (!metadata) return cleanup_after_failure(Status::invalid_argument("cannot flush shadow metadata", __FILE__, __LINE__), staging);
        metadata.close();
        if (!metadata) return cleanup_after_failure(Status::invalid_argument("cannot close shadow metadata", __FILE__, __LINE__), staging);
        status = publish_staging(staging, output);
        if (!status.ok()) return status;
        return result;
    } catch (const std::filesystem::filesystem_error& exception) {
        return cleanup_after_failure(Status::invalid_argument(exception.what(), __FILE__, __LINE__), staging);
    } catch (const std::bad_alloc& exception) {
        return cleanup_after_failure(Status::insufficient_memory(exception.what(), __FILE__, __LINE__), staging);
    } catch (const std::exception& exception) {
        return cleanup_after_failure(Status::invalid_argument(exception.what(), __FILE__, __LINE__), staging);
    }
}

}

Result<ShadowModelResult> create_qdq_shadow_model(const ShadowModelOptions& options) {
    try {
        return materialize_shadow(options);
    } catch (const std::filesystem::filesystem_error& error) {
        return Status::invalid_argument(error.what(), __FILE__, __LINE__);
    } catch (const std::bad_alloc& error) {
        return Status::insufficient_memory(error.what(), __FILE__, __LINE__);
    } catch (const std::exception& error) {
        return Status::invalid_argument(error.what(), __FILE__, __LINE__);
    }
}

}
