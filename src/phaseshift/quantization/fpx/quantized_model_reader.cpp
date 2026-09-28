#include <phaseshift/quantization/fpx/quantized_model_reader.h>
#include <phaseshift/quantization/fpx/crc32.h>
#include <phaseshift/quantization/fpx/layout.h>
#include <phaseshift/quantization/fpx/quantized_manifest.h>
#include <phaseshift/quantization/fpx/runtime_resolution.h>
#include <phaseshift/quantization/quant_format.h>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace ps::quantization::fpx {

namespace fs = std::filesystem;

namespace {

Status read_file_text(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return Status::invalid_argument("cannot open file", __FILE__, __LINE__);
    }
    out.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return Status::make_ok();
}

Status validate_shard_path(const std::string& shard) {
    const fs::path path(shard);
    if (shard.empty() || path.is_absolute() || path.has_root_name() || path.has_root_directory()) {
        return Status::invalid_argument("quantized shard path must be relative", __FILE__, __LINE__);
    }
    for (const fs::path& component : path) {
        if (component == "..") {
            return Status::invalid_argument("quantized shard path escapes bundle root", __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

Result<fs::path> weakly_canonical_path(const fs::path& path) {
    std::error_code error;
    fs::path canonical = fs::weakly_canonical(path, error);
    if (error) {
        return Status::invalid_argument(
            ("cannot resolve quantized path: " + path.string() + ": " + error.message()).c_str(),
            __FILE__, __LINE__);
    }
    return canonical;
}

bool is_path_prefix(const fs::path& prefix, const fs::path& path) {
    auto prefix_it = prefix.begin();
    auto path_it = path.begin();
    for (; prefix_it != prefix.end() && path_it != path.end(); ++prefix_it, ++path_it) {
        if (*prefix_it != *path_it) return false;
    }
    return prefix_it == prefix.end();
}

Result<uint32_t> parse_crc32(const std::string& text) {
    if (text.size() != 10 || text[0] != '0' || text[1] != 'x') {
        return Status::invalid_argument("quantized tensor CRC32 has invalid format", __FILE__, __LINE__);
    }
    uint32_t value = 0;
    for (std::size_t index = 2; index < text.size(); ++index) {
        const char digit = text[index];
        uint32_t nibble = 0;
        if (digit >= '0' && digit <= '9') nibble = static_cast<uint32_t>(digit - '0');
        else if (digit >= 'a' && digit <= 'f') nibble = static_cast<uint32_t>(digit - 'a' + 10);
        else if (digit >= 'A' && digit <= 'F') nibble = static_cast<uint32_t>(digit - 'A' + 10);
        else return Status::invalid_argument("quantized tensor CRC32 has invalid format", __FILE__, __LINE__);
        value = (value << 4) | nibble;
    }
    return value;
}

Status checked_product(uint64_t left, uint64_t right, uint64_t& result) {
    if (right != 0 && left > std::numeric_limits<uint64_t>::max() / right) {
        return Status::overflow("quantized tensor byte count overflow", __FILE__, __LINE__);
    }
    result = left * right;
    return Status::make_ok();
}

}

struct QuantizedModelReader::Impl {
    std::string output_dir;
    fs::path bundle_root;
    QuantizedManifest manifest;

    std::map<std::string, std::string> shard_for_tensor;
    std::set<std::string> shard_files;

    mutable std::mutex shard_mutex;
    mutable std::map<std::string, std::unique_ptr<ps::io::SafetensorsReader>> shard_readers;

    Result<fs::path> resolve_shard_path(const std::string& shard) const {
        Status path_status = validate_shard_path(shard);
        if (!path_status.ok()) return path_status;
        auto resolved_result = weakly_canonical_path(fs::path(output_dir) / shard);
        if (!resolved_result.ok()) return resolved_result.status();
        if (!is_path_prefix(bundle_root, resolved_result.value())) {
            return Status::invalid_argument("quantized shard path escapes bundle root", __FILE__, __LINE__);
        }
        return resolved_result.release();
    }

    Status open_and_validate_shard(const std::string& shard) {
        auto resolved_result = resolve_shard_path(shard);
        if (!resolved_result.ok()) return resolved_result.status();
        const fs::path full = resolved_result.release();
        std::lock_guard<std::mutex> lk(shard_mutex);
        auto it = shard_readers.find(shard);
        if (it == shard_readers.end()) {
            auto r = ps::io::SafetensorsReader::open(full.string());
            if (!r.ok()) {
                return r.status();
            }
            it = shard_readers.emplace(
                shard, std::make_unique<ps::io::SafetensorsReader>(r.release())).first;
        }

        const auto& md = it->second->metadata();
        auto fmt = md.find("phaseshift.format");
        if (fmt == md.end() || fmt->second != kQuantizedSafetensorsFormat) {
            return Status::invalid_argument(
                ("shard safetensors metadata phaseshift.format mismatch: " + shard).c_str(),
                __FILE__, __LINE__);
        }
        auto ver = md.find("phaseshift.format_version");
        if (ver == md.end() || ver->second != std::to_string(kQuantizedSafetensorsFormatVersion)) {
            return Status::invalid_argument(
                ("shard safetensors metadata phaseshift.format_version mismatch: " + shard).c_str(),
                __FILE__, __LINE__);
        }
        auto meta = md.find("phaseshift.quantization_metadata");
        if (meta == md.end() || meta->second != kQuantizedMetadataFile) {
            return Status::invalid_argument(
                ("shard safetensors metadata phaseshift.quantization_metadata mismatch: " + shard).c_str(),
                __FILE__, __LINE__);
        }
        return Status::make_ok();
    }

    Result<const ps::io::SafetensorsReader*> shard_reader(const std::string& physical) const {
        auto sit = shard_for_tensor.find(physical);
        if (sit == shard_for_tensor.end()) {
            return Status::invalid_argument("physical tensor not in weight_map", __FILE__, __LINE__);
        }
        const std::string shard = sit->second;
        auto resolved_result = resolve_shard_path(shard);
        if (!resolved_result.ok()) return resolved_result.status();
        const fs::path full = resolved_result.release();

        std::lock_guard<std::mutex> lk(shard_mutex);
        auto it = shard_readers.find(shard);
        if (it == shard_readers.end()) {
            auto r = ps::io::SafetensorsReader::open(full.string());
            if (!r.ok()) {
                return r.status();
            }
            auto inserted = shard_readers.emplace(shard, std::make_unique<ps::io::SafetensorsReader>(r.release()));
            it = inserted.first;
        }
        return it->second.get();
    }

    Status validate_physical_tensor(const QuantizedTensorRef& reference,
                                    const ps::io::SafetensorsReader& reader,
                                    ps::io::SType expected_dtype,
                                    const std::vector<std::size_t>& expected_shape,
                                    uint64_t expected_bytes,
                                    bool verify_crc) const {
        auto spec_result = reader.tensor_spec(reference.tensor);
        if (!spec_result.ok()) return spec_result.status();
        const ps::io::StTensorSpec& spec = spec_result.value();
        if (spec.dtype != expected_dtype || spec.shape != expected_shape) {
            return Status::invalid_argument(
                ("quantized physical tensor dtype or shape mismatch: " + reference.tensor).c_str(),
                __FILE__, __LINE__);
        }
        if (spec.data_end - spec.data_begin != expected_bytes) {
            return Status::invalid_argument(
                ("quantized physical tensor byte count mismatch: " + reference.tensor).c_str(),
                __FILE__, __LINE__);
        }
        if (expected_bytes > std::numeric_limits<std::size_t>::max()) {
            return Status::overflow("quantized physical tensor byte count overflow", __FILE__, __LINE__);
        }
        std::size_t actual_bytes = 0;
        auto data_result = reader.tensor_data(reference.tensor, actual_bytes);
        if (!data_result.ok()) return data_result.status();
        if (actual_bytes != expected_bytes) {
            return Status::invalid_argument(
                ("quantized physical tensor byte count mismatch: " + reference.tensor).c_str(),
                __FILE__, __LINE__);
        }
        auto expected_crc = parse_crc32(reference.crc32);
        if (!expected_crc.ok()) return expected_crc.status();
        if (verify_crc && crc32_iso_hdlc(data_result.value(), actual_bytes) != expected_crc.value()) {
            return Status::invalid_argument(
                ("quantized physical tensor CRC32 mismatch: " + reference.tensor).c_str(),
                __FILE__, __LINE__);
        }
        return Status::make_ok();
    }

    Status validate_physical_tensors(bool verify_crc) const {
        for (const auto& entry : manifest.tensors) {
            const QuantizedTensorMetadata& tensor = entry.second;
            if (tensor.logical_shape.empty()) {
                return Status::invalid_argument("quantized tensor shape is empty", __FILE__, __LINE__);
            }
            uint64_t rows = 1;
            for (std::size_t index = 0; index + 1 < tensor.logical_shape.size(); ++index) {
                if (tensor.logical_shape[index] <= 0) {
                    return Status::invalid_argument("quantized tensor shape is invalid", __FILE__, __LINE__);
                }
                Status product_status = checked_product(rows, static_cast<uint64_t>(tensor.logical_shape[index]), rows);
                if (!product_status.ok()) return product_status;
            }
            const int64_t logical_k = tensor.logical_shape.back();
            if (logical_k <= 0) return Status::invalid_argument("quantized tensor K is invalid", __FILE__, __LINE__);
            const uint64_t k = static_cast<uint64_t>(logical_k);

            auto validate_ref = [&](const QuantizedTensorRef& reference,
                                    ps::io::SType dtype,
                                    uint64_t physical_k,
                                    uint64_t element_size) -> Status {
                std::vector<std::size_t> shape;
                shape.reserve(tensor.logical_shape.size());
                for (const int64_t dimension : tensor.logical_shape) {
                    if (dimension <= 0 || static_cast<uint64_t>(dimension) > std::numeric_limits<std::size_t>::max()) {
                        return Status::invalid_argument("quantized tensor shape is invalid", __FILE__, __LINE__);
                    }
                    shape.push_back(static_cast<std::size_t>(dimension));
                }
                if (physical_k > std::numeric_limits<std::size_t>::max()) {
                    return Status::overflow("quantized physical tensor shape overflow", __FILE__, __LINE__);
                }
                shape.back() = static_cast<std::size_t>(physical_k);
                uint64_t expected_bytes = 0;
                Status bytes_status = checked_product(rows, physical_k, expected_bytes);
                if (!bytes_status.ok()) return bytes_status;
                bytes_status = checked_product(expected_bytes, element_size, expected_bytes);
                if (!bytes_status.ok()) return bytes_status;
                auto reader_result = shard_reader(reference.tensor);
                if (!reader_result.ok()) return reader_result.status();
                return validate_physical_tensor(reference, *reader_result.value(), dtype, shape, expected_bytes, verify_crc);
            };

            auto validate_ref_exact = [&](const QuantizedTensorRef& reference,
                                          ps::io::SType dtype,
                                          const std::vector<std::size_t>& shape,
                                          uint64_t expected_bytes) -> Status {
                auto reader_result = shard_reader(reference.tensor);
                if (!reader_result.ok()) return reader_result.status();
                return validate_physical_tensor(reference, *reader_result.value(), dtype, shape, expected_bytes, verify_crc);
            };

            if (tensor.encoding == QuantizedEncoding::Bf16) {
                Status ref_status = validate_ref(*tensor.data, ps::io::SType::BF16, k, sizeof(uint16_t));
                if (!ref_status.ok()) return ref_status;
            } else if (tensor.encoding == QuantizedEncoding::Fp8Block128) {
                std::vector<std::size_t> logical;
                logical.reserve(tensor.logical_shape.size());
                for (const int64_t dimension : tensor.logical_shape) {
                    logical.push_back(static_cast<std::size_t>(dimension));
                }
                if (logical.size() < 2) {
                    return Status::invalid_argument("fp8 block128 requires a matrix shape", __FILE__, __LINE__);
                }
                if (!tensor.codes) {
                    return Status::invalid_argument("quantized tensor is missing the codes stream", __FILE__, __LINE__);
                }
                if (!tensor.metadata1) {
                    return Status::invalid_argument("quantized tensor is missing the scale stream", __FILE__, __LINE__);
                }
                uint64_t batch = 1;
                for (std::size_t i = 0; i + 2 < logical.size(); ++i) {
                    Status product_status = checked_product(batch, logical[i], batch);
                    if (!product_status.ok()) return product_status;
                }
                const uint64_t n = logical[logical.size() - 2];
                const uint64_t kp = tensor.k_padded;
                if (kp < k || (kp % 128) != 0) {
                    return Status::invalid_argument("fp8 block128 k_padded invalid", __FILE__, __LINE__);
                }
                std::vector<std::size_t> codes_shape = logical;
                codes_shape.back() = static_cast<std::size_t>(kp);
                uint64_t codes_bytes = 0;
                Status code_bytes_status = checked_product(batch, n, codes_bytes);
                if (!code_bytes_status.ok()) return code_bytes_status;
                code_bytes_status = checked_product(codes_bytes, kp, codes_bytes);
                if (!code_bytes_status.ok()) return code_bytes_status;
                Status code_status = validate_ref_exact(*tensor.codes, ps::io::SType::U8, codes_shape, codes_bytes);
                if (!code_status.ok()) return code_status;

                const uint64_t scale_n = (n + 127u) / 128u;
                const uint64_t scale_k = kp / 128u;
                std::vector<std::size_t> scale_shape(logical.begin(), logical.end() - 2);
                scale_shape.push_back(static_cast<std::size_t>(scale_n));
                scale_shape.push_back(static_cast<std::size_t>(scale_k));
                uint64_t scale_count = 0;
                Status scale_status = checked_product(batch, scale_n, scale_count);
                if (!scale_status.ok()) return scale_status;
                scale_status = checked_product(scale_count, scale_k, scale_count);
                if (!scale_status.ok()) return scale_status;
                uint64_t scale_bytes = 0;
                scale_status = checked_product(scale_count, sizeof(float), scale_bytes);
                if (!scale_status.ok()) return scale_status;
                return validate_ref_exact(*tensor.metadata1, ps::io::SType::F32, scale_shape, scale_bytes);
            } else if (tensor.encoding == QuantizedEncoding::Mxfp4) {
                if (!tensor.codes) {
                    return Status::invalid_argument("quantized tensor is missing the codes stream", __FILE__, __LINE__);
                }
                if (!tensor.metadata1) {
                    return Status::invalid_argument("quantized tensor is missing the scale stream", __FILE__, __LINE__);
                }
                const uint64_t kp = tensor.k_padded;
                if (kp < k || (kp % 32) != 0) {
                    return Status::invalid_argument("mxfp4 k_padded invalid", __FILE__, __LINE__);
                }
                std::vector<std::size_t> codes_shape(tensor.logical_shape.begin(), tensor.logical_shape.end());
                codes_shape.back() = static_cast<std::size_t>(kp / 2u);
                uint64_t codes_bytes = 0;
                Status code_bytes_status = checked_product(rows, kp / 2u, codes_bytes);
                if (!code_bytes_status.ok()) return code_bytes_status;
                Status code_status = validate_ref_exact(*tensor.codes, ps::io::SType::U8, codes_shape, codes_bytes);
                if (!code_status.ok()) return code_status;

                std::vector<std::size_t> scale_shape(tensor.logical_shape.begin(), tensor.logical_shape.end());
                scale_shape.back() = static_cast<std::size_t>(kp / 32u);
                uint64_t scale_bytes = 0;
                Status scale_status = checked_product(rows, kp / 32u, scale_bytes);
                if (!scale_status.ok()) return scale_status;
                return validate_ref_exact(*tensor.metadata1, ps::io::SType::U8, scale_shape, scale_bytes);
            } else {
                const ps::quantization::QuantFormatId fid =
                    (tensor.encoding == QuantizedEncoding::Psq4)
                        ? ps::quantization::QuantFormatId::Psq4
                    : (tensor.encoding == QuantizedEncoding::Psq8)
                        ? ps::quantization::QuantFormatId::Psq8
                        : ps::quantization::QuantFormatId::None;
                const ps::quantization::QuantFormatDesc* desc = ps::quantization::quant_format_desc(fid);
                if (desc == nullptr)
                    return Status::invalid_argument("unknown quantized encoding", __FILE__, __LINE__);
                if (!tensor.codes)
                    return Status::invalid_argument("quantized tensor is missing the codes stream", __FILE__, __LINE__);
                const uint64_t kp = tensor.k_padded;
                const Status code_status = validate_ref(*tensor.codes, ps::io::SType::U8,
                    ps::quantization::quant_codes_bytes(1, kp, *desc), 1);
                if (!code_status.ok()) return code_status;
                const std::optional<QuantizedTensorRef>* meta_ref[4] = {
                    &tensor.metadata1, &tensor.metadata2, &tensor.metadata3, &tensor.metadata4};
                const uint64_t meta_bytes[4] = {
                    ps::quantization::quant_metadata1_bytes(1, kp, *desc),
                    ps::quantization::quant_metadata2_bytes(1, kp, *desc),
                    ps::quantization::quant_metadata3_bytes(1, kp, *desc),
                    ps::quantization::quant_metadata4_bytes(1, kp, *desc)};
                for (int s = 0; s < 4; ++s) {
                    if (desc->meta[s].count_mode == ps::quantization::MetaCountMode::None)
                        continue;
                    if (!(*meta_ref[s]))
                        return Status::invalid_argument("quantized tensor is missing a required metadata stream", __FILE__, __LINE__);
                    const Status st = validate_ref(**meta_ref[s], ps::io::SType::U8, meta_bytes[s], 1);
                    if (!st.ok()) return st;
                }
            }
        }
        return Status::make_ok();
    }
};

Result<QuantizedModelReader> QuantizedModelReader::open(
    const std::string& output_dir, bool verify_payload_crc) {
    const fs::path dir(output_dir);
    const fs::path manifest_path = dir / kQuantizedMetadataFile;
    if (!fs::exists(manifest_path)) {
        return Status::invalid_argument("phaseshift_quantization.json not found", __FILE__, __LINE__);
    }

    std::string manifest_text;
    auto mres = read_file_text(manifest_path.string(), manifest_text);
    if (!mres.ok()) return mres;
    auto manifest_result = parse_quantized_manifest(manifest_text);
    if (!manifest_result.ok()) return manifest_result.status();
    QuantizedManifest manifest = std::move(manifest_result.value());
    Status contract_status = validate_quantized_manifest_contract(manifest);
    if (!contract_status.ok()) return contract_status;

    auto pimpl = std::make_unique<Impl>();
    pimpl->output_dir = output_dir;
    auto root_result = weakly_canonical_path(dir);
    if (!root_result.ok()) return root_result.status();
    pimpl->bundle_root = root_result.release();
    pimpl->manifest = std::move(manifest);

    const fs::path index_path = dir / "model.safetensors.index.json";
    if (fs::exists(index_path)) {
        std::string index_text;
        auto ires = read_file_text(index_path.string(), index_text);
        if (!ires.ok()) return ires;
        nlohmann::json idx;
        try {
            idx = nlohmann::json::parse(index_text);
        } catch (const nlohmann::json::exception& e) {
            return Status::invalid_argument(
                ("bad safetensors index: " + std::string(e.what())).c_str(), __FILE__, __LINE__);
        }
        if (!idx.contains("weight_map") || !idx["weight_map"].is_object()) {
            return Status::invalid_argument("safetensors index missing weight_map", __FILE__, __LINE__);
        }
        for (auto it = idx["weight_map"].begin(); it != idx["weight_map"].end(); ++it) {
            if (!it.value().is_string()) {
                return Status::invalid_argument("safetensors index shard path is not a string", __FILE__, __LINE__);
            }
            const std::string shard = it.value().get<std::string>();
            Status path_status = validate_shard_path(shard);
            if (!path_status.ok()) return path_status;
            pimpl->shard_for_tensor[it.key()] = shard;
            pimpl->shard_files.insert(shard);
        }
    } else {
        const fs::path single = dir / "model.safetensors";
        if (fs::exists(single)) {
            const std::string shard = "model.safetensors";
            pimpl->shard_files.insert(shard);
        } else {
            return Status::invalid_argument("no model.safetensors or index found", __FILE__, __LINE__);
        }
    }

    for (const std::string& shard : pimpl->shard_files) {
        Status path_status = validate_shard_path(shard);
        if (!path_status.ok()) return path_status;
    }

    for (const std::string& shard : pimpl->shard_files) {
        Status st = pimpl->open_and_validate_shard(shard);
        if (!st.ok()) {
            return st;
        }
    }

    for (const auto& kv : pimpl->manifest.tensors) {
        const QuantizedTensorMetadata& t = kv.second;
        std::vector<std::string> physical;
        if (t.codes) physical.push_back(t.codes->tensor);
        if (t.metadata1) physical.push_back(t.metadata1->tensor);
        if (t.metadata2) physical.push_back(t.metadata2->tensor);
        if (t.metadata3) physical.push_back(t.metadata3->tensor);
        if (t.metadata4) physical.push_back(t.metadata4->tensor);
        if (t.data) physical.push_back(t.data->tensor);
        for (const auto& p : physical) {
            if (!pimpl->shard_for_tensor.count(p)) {
                if (pimpl->shard_files.count("model.safetensors")) {
                    pimpl->shard_for_tensor[p] = "model.safetensors";
                } else {
                    return Status::invalid_argument(
                        ("physical tensor missing from index: " + p).c_str(), __FILE__, __LINE__);
                }
            }
        }
    }

    Status physical_status = pimpl->validate_physical_tensors(verify_payload_crc);
    if (!physical_status.ok()) return physical_status;

    QuantizedModelReader reader;
    reader.pimpl_ = pimpl.release();
    return reader;
}

QuantizedModelReader::~QuantizedModelReader() noexcept {
    delete pimpl_;
}

QuantizedModelReader::QuantizedModelReader(QuantizedModelReader&& other) noexcept : pimpl_(other.pimpl_) {
    other.pimpl_ = nullptr;
}

QuantizedModelReader& QuantizedModelReader::operator=(QuantizedModelReader&& other) noexcept {
    if (this != &other) {
        delete pimpl_;
        pimpl_ = other.pimpl_;
        other.pimpl_ = nullptr;
    }
    return *this;
}

const QuantizedManifest& QuantizedModelReader::manifest() const {
    return pimpl_->manifest;
}

Result<std::string> QuantizedModelReader::resolve_alias(const std::string& name) const {
    std::string cur = name;
    std::set<std::string> seen;
    while (true) {
        auto it = pimpl_->manifest.aliases.find(cur);
        if (it == pimpl_->manifest.aliases.end()) break;
        if (!seen.insert(cur).second) {
            return Status::invalid_argument("alias cycle detected", __FILE__, __LINE__);
        }
        cur = it->second;
    }
    return cur;
}

Result<std::vector<std::string>> QuantizedModelReader::list_logical_tensors() const {
    std::vector<std::string> out;
    out.reserve(pimpl_->manifest.tensors.size());
    for (const auto& kv : pimpl_->manifest.tensors) {
        out.push_back(kv.first);
    }
    return out;
}

Result<QuantizedTensorView> QuantizedModelReader::resolve(const std::string& logical_name) const {
    std::string canonical = logical_name;
    auto alias_res = resolve_alias(logical_name);
    if (!alias_res.ok()) return alias_res.status();
    canonical = alias_res.value();

    auto it = pimpl_->manifest.tensors.find(canonical);
    if (it == pimpl_->manifest.tensors.end()) {
        return Status::invalid_argument("logical tensor not found", __FILE__, __LINE__);
    }
    const QuantizedTensorMetadata& t = it->second;

    QuantizedTensorView view;
    view.name = canonical;
    view.encoding = t.encoding;
    view.logical_shape = t.logical_shape;
    view.k_padded = static_cast<int64_t>(t.k_padded);

    auto fetch = [&](const std::string& physical, std::size_t& out_bytes)
        -> Result<const void*> {
        auto rres = pimpl_->shard_reader(physical);
        if (!rres.ok()) return rres.status();
        const ps::io::SafetensorsReader* reader = rres.value();
        auto sres = reader->tensor_data(physical, out_bytes);
        if (!sres.ok()) return sres.status();
        return sres.value();
    };

    if (t.codes) {
        std::size_t nb = 0;
        auto r = fetch(t.codes->tensor, nb);
        if (!r.ok()) return r.status();
        view.codes = ByteSpan{r.value(), nb};
    }
    if (t.metadata1) {
        std::size_t nb = 0;
        auto r = fetch(t.metadata1->tensor, nb);
        if (!r.ok()) return r.status();
        view.metadata1 = ByteSpan{r.value(), nb};
    }
    if (t.metadata2) {
        std::size_t nb = 0;
        auto r = fetch(t.metadata2->tensor, nb);
        if (!r.ok()) return r.status();
        view.metadata2 = ByteSpan{r.value(), nb};
    }
    if (t.metadata3) {
        std::size_t nb = 0;
        auto r = fetch(t.metadata3->tensor, nb);
        if (!r.ok()) return r.status();
        view.metadata3 = ByteSpan{r.value(), nb};
    }
    if (t.metadata4) {
        std::size_t nb = 0;
        auto r = fetch(t.metadata4->tensor, nb);
        if (!r.ok()) return r.status();
        view.metadata4 = ByteSpan{r.value(), nb};
    }
    if (t.data) {
        std::size_t nb = 0;
        auto r = fetch(t.data->tensor, nb);
        if (!r.ok()) return r.status();
        view.data = ByteSpan{r.value(), nb};
    }
    return view;
}

uint64_t QuantizedModelReader::payload_bytes() const {
    return pimpl_->manifest.payload_bytes;
}

}
