#include <phaseshift/quantization/offline/quantizer.h>
#include <phaseshift/quantization/offline/adapter_dispatch.h>
#include <phaseshift/quantization/offline/gpu_quantizer.h>
#include <phaseshift/quantization/offline/qwen35_adapter.h>
#include <phaseshift/quantization/offline/shard_resolver.h>
#include <phaseshift/io/safetensors_reader.h>
#include <phaseshift/io/safetensors_writer.h>
#include <phaseshift/quantization/fpx/layout.h>
#include <phaseshift/quantization/fpx/profile.h>
#include <phaseshift/quantization/fpx/quantize_source.h>
#include <phaseshift/quantization/fpx/quantized_model_reader.h>
#include <phaseshift/quantization/psq/psq.h>
#include <phaseshift/quantization/psq/quant_canonical.h>
#include <phaseshift/quantization/mxfp4/mxfp4.h>
#include <phaseshift/quantization/fp8/block128.h>
#include <phaseshift/quantization/quant_format.h>
#include <algorithm>
#include <span>
#include <phaseshift/quantization/offline/imatrix_format.h>
#include <phaseshift/quantization/offline/model_fingerprint.h>
#include <nlohmann/json.hpp>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <filesystem>

namespace ps::quantization::fpx {

namespace fs = std::filesystem;

namespace {

std::string crc_hex(uint32_t c) {
    char buf[12];
    std::snprintf(buf, sizeof(buf), "0x%08x", c);
    return std::string(buf);
}

struct PhysicalTensor {
    std::string name;
    ps::io::SType dtype = ps::io::SType::BF16;
    std::vector<int64_t> shape;
    uint64_t bytes = 0;
};

struct PlannedTensor {
    std::string name;
    TensorInfo info;
    SourceView sv;
    WeightEncoding enc = WeightEncoding::BF16;
    uint64_t k = 0;
    uint64_t kp = 0;
    uint64_t rows = 0;
    uint64_t rb = 0;
    uint64_t payload_bytes = 0;
    PhysicalTensor codes;
    PhysicalTensor metadata1;
    PhysicalTensor data;
    uint32_t codes_crc = 0;
    uint32_t metadata1_crc = 0;
    uint32_t data_crc = 0;
};

struct Shard {
    std::vector<PlannedTensor> tensors;
    uint64_t payload_bytes = 0;
};

std::vector<int64_t> replace_last_dim(const std::vector<int64_t>& shape, int64_t last) {
    std::vector<int64_t> out = shape;
    if (!out.empty()) out.back() = last;
    return out;
}

std::vector<std::size_t> to_size_shape(const std::vector<int64_t>& shape) {
    std::vector<std::size_t> out;
    out.reserve(shape.size());
    for (int64_t d : shape) out.push_back(static_cast<std::size_t>(d));
    return out;
}

std::vector<int64_t> fp8_scale_shape(const std::vector<int64_t>& shape, uint64_t kp) {
    std::vector<int64_t> out(shape.begin(), shape.end() - 1);
    if (out.empty()) out.push_back(1);
    else out.back() = (out.back() + 127) / 128;
    out.push_back(static_cast<int64_t>(kp / 128));
    return out;
}

void build_physical(PlannedTensor& p) {
    const MatrixShape ms = matrix_shape(p.sv.shape);
    const uint64_t rows = p.rows;
    const uint64_t kp = p.kp;
    if (p.enc == WeightEncoding::PSQ4) {
        p.codes.name = p.name + kCodesSuffix;
        p.codes.dtype = ps::io::SType::U8;
        p.codes.shape = replace_last_dim(p.sv.shape, static_cast<int64_t>(kp / 2));
        p.codes.bytes = rows * (kp / 2);
        p.metadata1.name = p.name + kMetadata1Suffix;
        p.metadata1.dtype = ps::io::SType::U8;
        p.metadata1.shape = replace_last_dim(p.sv.shape, static_cast<int64_t>(kp / 16));
        p.metadata1.bytes = rows * (kp / 16);
        p.payload_bytes = p.codes.bytes + p.metadata1.bytes;
    } else if (p.enc == WeightEncoding::PSQ8) {
        p.codes.name = p.name + kCodesSuffix;
        p.codes.dtype = ps::io::SType::U8;
        p.codes.shape = replace_last_dim(p.sv.shape, static_cast<int64_t>(kp));
        p.codes.bytes = rows * kp;
        p.metadata1.name = p.name + kMetadata1Suffix;
        p.metadata1.dtype = ps::io::SType::U8;
        p.metadata1.shape = replace_last_dim(p.sv.shape, static_cast<int64_t>(kp / 16));
        p.metadata1.bytes = rows * (kp / 16);
        p.payload_bytes = p.codes.bytes + p.metadata1.bytes;
    } else if (p.enc == WeightEncoding::FP8_BLOCK128) {
        p.codes.name = p.name + kCodesSuffix;
        p.codes.dtype = ps::io::SType::U8;
        p.codes.shape = replace_last_dim(p.sv.shape, static_cast<int64_t>(kp));
        p.codes.bytes = rows * kp;
        p.metadata1.name = p.name + kMetadata1Suffix;
        p.metadata1.dtype = ps::io::SType::F32;
        p.metadata1.shape = fp8_scale_shape(p.sv.shape, kp);
        p.metadata1.bytes = ms.batch * ((ms.n + 127) / 128) * (kp / 128) * 4;
        p.payload_bytes = p.codes.bytes + p.metadata1.bytes;
    } else if (p.enc == WeightEncoding::MXFP4) {
        p.codes.name = p.name + kCodesSuffix;
        p.codes.dtype = ps::io::SType::U8;
        p.codes.shape = replace_last_dim(p.sv.shape, static_cast<int64_t>(kp / 2));
        p.codes.bytes = rows * (kp / 2);
        p.metadata1.name = p.name + kMetadata1Suffix;
        p.metadata1.dtype = ps::io::SType::U8;
        p.metadata1.shape = replace_last_dim(p.sv.shape, static_cast<int64_t>(kp / 32));
        p.metadata1.bytes = rows * (kp / 32);
        p.payload_bytes = p.codes.bytes + p.metadata1.bytes;
    } else {
        p.data.name = p.name;
        p.data.dtype = ps::io::SType::BF16;
        p.data.shape = p.sv.shape;
        p.data.bytes = rows * p.k * 2;
        p.payload_bytes = p.data.bytes;
    }
}

bool is_excluded_asset(const std::string& filename) {
    if (filename == "weights.bin" || filename == "manifest.json") return true;
    if (filename == "model.safetensors.index.json") return true;
    if (filename.size() >= 12 &&
        filename.compare(filename.size() - 12, 12, ".safetensors") == 0) return true;
    if (filename.compare(0, 14, "pytorch_model") == 0 &&
        filename.size() >= 4 && filename.compare(filename.size() - 4, 4, ".bin") == 0) return true;
    if (filename.size() >= 5 &&
        filename.compare(filename.size() - 5, 5, ".gguf") == 0) return true;
    return false;
}

Status copy_assets(const std::string& src_dir, const std::string& dst_dir) {
    fs::create_directories(dst_dir);
    for (const auto& entry : fs::recursive_directory_iterator(src_dir)) {
        const fs::path p = entry.path();
        const fs::path rel = p.lexically_relative(src_dir);
        if (fs::is_directory(p)) {
            fs::create_directories(dst_dir / rel);
            continue;
        }
        if (is_excluded_asset(p.filename().string())) continue;
        const fs::path dest = dst_dir / rel;
        fs::create_directories(dest.parent_path());

        std::error_code ec;
        fs::path src = p;
        if (fs::is_symlink(p)) {
            src = fs::weakly_canonical(p, ec);
            if (ec || !fs::is_regular_file(src)) {
                std::fprintf(stderr, "WARNING: skipping broken/external symlink: %s\n", p.string().c_str());
                continue;
            }
        }
        if (!fs::is_regular_file(src)) continue;
        fs::copy_file(src, dest, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            return Status::invalid_argument(
                ("asset copy failed: " + p.string() + ": " + ec.message()).c_str(),
                __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

Result<std::string> read_text(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        return Status::invalid_argument("cannot open file", __FILE__, __LINE__);
    }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

}  // namespace

Result<uint64_t> parse_byte_size(const std::string& s) {
    if (s.empty()) {
        return Status::invalid_argument("empty byte size", __FILE__, __LINE__);
    }
    size_t i = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
    if (i == 0) {
        return Status::invalid_argument("invalid byte size", __FILE__, __LINE__);
    }
    const uint64_t num = std::stoull(s.substr(0, i));
    std::string suffix = s.substr(i);
    for (char& c : suffix) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    uint64_t mult = 1;
    if (suffix == "b" || suffix.empty()) mult = 1;
    else if (suffix == "kb" || suffix == "k") mult = 1000ull;
    else if (suffix == "kib") mult = 1024ull;
    else if (suffix == "mb" || suffix == "m") mult = 1000ull * 1000ull;
    else if (suffix == "mib") mult = 1024ull * 1024ull;
    else if (suffix == "gb" || suffix == "g") mult = 1000ull * 1000ull * 1000ull;
    else if (suffix == "gib") mult = 1024ull * 1024ull * 1024ull;
    else return Status::invalid_argument("unknown byte size suffix", __FILE__, __LINE__);
    if (num > (UINT64_MAX / mult)) {
        return Status::invalid_argument("byte size overflow", __FILE__, __LINE__);
    }
    return num * mult;
}

Result<QuantizedModelReader> validate_model_dir(const std::string& output_dir) {
    return QuantizedModelReader::open(output_dir);
}

namespace {

Result<QuantizeSummary> run_quantize(const QuantizeOptions& opts) {
    const fs::path config_path = fs::path(opts.input_dir) / "config.json";
    if (!fs::exists(config_path)) {
        return Status::invalid_argument("config.json not found", __FILE__, __LINE__);
    }
    auto config_text_res = read_text(config_path);
    if (!config_text_res.ok()) return config_text_res.status();
    const std::string config_text = config_text_res.value();

    nlohmann::json config;
    try {
        config = nlohmann::json::parse(config_text);
    } catch (...) {
        return Status::invalid_argument("bad config.json", __FILE__, __LINE__);
    }
    if (config.contains("phaseshift_quantization")) {
        return Status::invalid_argument(
            "input is already a PhaseShift quantized model; V1 does not requantize",
            __FILE__, __LINE__);
    }
    if (fs::exists(fs::path(opts.input_dir) / kQuantizedMetadataFile)) {
        return Status::invalid_argument(
            "input is already a PhaseShift quantized model; V1 does not requantize",
            __FILE__, __LINE__);
    }

    auto arch_result = detect_quantization_architecture(config_text);
    if (!arch_result.ok()) return arch_result.status();
    const Architecture arch = arch_result.value();

    if (arch == Architecture::DFlash2Draft && opts.preset != FpxPreset::Psq) {
        return Status::invalid_argument(
            "DFlash2 draft models support only --preset psq", __FILE__, __LINE__);
    }
    if (arch == Architecture::DFlash2Draft && !opts.imatrix_path.empty()) {
        return Status::invalid_argument(
            "DFlash2 draft models do not support --imatrix yet", __FILE__, __LINE__);
    }

    ImatrixCoverage imatrix_cov;
    std::map<std::string, ps::quantization::imatrix::PsimEntryDesc> imatrix_entries;
    ps::quantization::imatrix::PsimFile psim_file;
    if (!opts.imatrix_path.empty()) {
        auto psim_res = ps::quantization::imatrix::read_psim(opts.imatrix_path);
        if (!psim_res.ok()) {
            return Status::invalid_argument(
                ("imatrix load failed: " + psim_res.status().message()).c_str(),
                __FILE__, __LINE__);
        }
        psim_file = std::move(psim_res.value());
        const auto& h = psim_file.header;
        if (h.model.architecture != to_string(arch)) {
            return Status::invalid_argument("imatrix architecture mismatch", __FILE__, __LINE__);
        }
        const std::string fp = compute_model_fingerprint(opts.input_dir);
        if (h.model.fingerprint != fp) {
            return Status::invalid_argument("imatrix model fingerprint mismatch", __FILE__, __LINE__);
        }
        for (const auto& e : h.entries) {
            imatrix_entries[e.name] = e;
        }
        imatrix_cov.used = true;
        imatrix_cov.policy = opts.imatrix_policy;
    }

    const bool tie = config.value("tie_word_embeddings", false);
    auto num_layers_result = quantization_num_layers(arch, config_text);
    if (!num_layers_result.ok()) return num_layers_result.status();
    const uint32_t num_layers = num_layers_result.value();

    auto resolver_result = ShardResolver::open(opts.input_dir);
    if (!resolver_result.ok()) return resolver_result.status();
    ShardResolver resolver = resolver_result.release();

    std::vector<PlannedTensor> planned;
    std::map<std::string, int> dtype_counts;
    bool has_visual = false;
    uint64_t skipped_tensors = 0;

    std::map<std::string, std::unique_ptr<ps::io::SafetensorsReader>> readers;
    for (const auto& shard : resolver.shard_files()) {
        auto r = ps::io::SafetensorsReader::open(shard);
        if (!r.ok()) return r.status();
        readers[shard] = std::make_unique<ps::io::SafetensorsReader>(r.release());
    }

    for (const auto& tname : resolver.tensor_names()) {
        auto shard_res = resolver.shard_path(tname);
        if (!shard_res.ok()) return shard_res.status();
        const std::string shard = shard_res.value();
        auto* reader = readers[shard].get();
        auto spec_res = reader->tensor_spec(tname);
        if (!spec_res.ok()) return spec_res.status();
        const auto& spec = spec_res.value();
        std::size_t nbytes = 0;
        auto data_res = reader->tensor_data(tname, nbytes);
        if (!data_res.ok()) return data_res.status();

        PlannedTensor p;
        p.name = tname;
        p.sv.data = static_cast<const uint8_t*>(data_res.value());
        p.sv.dtype = spec.dtype;
        p.sv.shape = std::vector<int64_t>(spec.shape.begin(), spec.shape.end());
        if (spec.dtype == ps::io::SType::BF16) p.sv.byte_stride = 2;
        else if (spec.dtype == ps::io::SType::F16) p.sv.byte_stride = 2;
        else p.sv.byte_stride = 4;

        p.info = classify_quantization_tensor(arch, tname, p.sv.shape);
        if (p.info.malformed_name) {
            return Status::invalid_argument(
                ("malformed layer index in tensor name: " + tname).c_str(),
                __FILE__, __LINE__);
        }
        if (!p.info.is_visual && p.info.layer_index >= 0) {
            const uint32_t li = static_cast<uint32_t>(p.info.layer_index);
            const bool is_main_layer =
                tname.rfind(quantization_main_layer_prefix(arch), 0) == 0;
            if (is_main_layer && li >= num_layers) {
                return Status::invalid_argument(
                    ("tensor layer index out of range: " + tname).c_str(),
                    __FILE__, __LINE__);
            }
        }
        if (p.info.is_visual) {
            has_visual = true;
            ++skipped_tensors;
            continue;
        }
        if (p.info.role == TensorRole::Unknown) {
            return Status::invalid_argument(
                ("Unsupported quantizable tensor: " + tname).c_str(),
                __FILE__, __LINE__);
        }
        if (tie && p.info.role == TensorRole::Output) {
            return Status::invalid_argument(
                "tied checkpoint unexpectedly contains an Output-role tensor (lm_head.weight)",
                __FILE__, __LINE__);
        }
        const uint32_t li = p.info.layer_index >= 0 ? static_cast<uint32_t>(p.info.layer_index) : 0;
        p.enc = choose_weight_encoding(opts.preset, arch, p.info.role, li, num_layers);

        p.k = k_dim(p.sv.shape);
        p.kp = padded_k(p.k, p.enc);
        p.rows = row_count(p.sv.shape);
        p.rb = (p.enc == WeightEncoding::FP8_BLOCK128) ? (p.kp / 128) : (p.kp / 32);
        build_physical(p);

        if (imatrix_cov.used && p.enc != WeightEncoding::BF16) {
            ++imatrix_cov.required;
            auto it = imatrix_entries.find(tname);
            if (it == imatrix_entries.end()) {
                if (opts.imatrix_policy == ImatrixPolicy::Strict) {
                    return Status::invalid_argument(
                        ("imatrix missing entry (strict): " + tname).c_str(),
                        __FILE__, __LINE__);
                }
                ++imatrix_cov.missing;
                std::fprintf(stderr, "WARNING: imatrix missing: %s falling back to unweighted %s\n",
                             tname.c_str(), to_string(p.enc));
            } else if (it->second.k != p.k) {
                return Status::invalid_argument(
                    ("imatrix K mismatch: " + tname).c_str(), __FILE__, __LINE__);
            } else {
                ++imatrix_cov.found;
            }
        }
        const std::string dt = (spec.dtype == ps::io::SType::BF16) ? "bf16"
                              : (spec.dtype == ps::io::SType::F16) ? "f16" : "f32";
        dtype_counts[dt]++;
        planned.push_back(std::move(p));
    }

    if (has_visual && !opts.text_only_scope) {
        return Status::invalid_argument(
            "Qwen3.5 multimodal conversion is not supported by FPX V1. "
            "Use --scope text-only to convert the language model only.",
            __FILE__, __LINE__);
    }

    if (imatrix_cov.used && imatrix_cov.required > 0) {
        imatrix_cov.coverage = static_cast<double>(imatrix_cov.found) /
                               static_cast<double>(imatrix_cov.required);
    }

    const fs::path input_abs = fs::absolute(opts.input_dir);
    const fs::path output_abs = fs::absolute(opts.output_dir);
    if (output_abs == input_abs ||
        output_abs.string().rfind(input_abs.string() + "/", 0) == 0 ||
        input_abs.string().rfind(output_abs.string() + "/", 0) == 0) {
        return Status::invalid_argument(
            "output directory must not be inside the input directory", __FILE__, __LINE__);
    }
    if (fs::exists(opts.output_dir) && !opts.overwrite) {
        return Status::invalid_argument(
            "output directory already exists; use --overwrite to replace", __FILE__, __LINE__);
    }

    std::vector<Shard> shards;
    {
        Shard cur;
        for (auto& p : planned) {
            if (!cur.tensors.empty() &&
                cur.payload_bytes + p.payload_bytes > opts.max_shard_size) {
                shards.push_back(std::move(cur));
                cur = Shard{};
            }
            cur.payload_bytes += p.payload_bytes;
            cur.tensors.push_back(std::move(p));
        }
        if (!cur.tensors.empty()) shards.push_back(std::move(cur));
    }

    const std::string staging = opts.output_dir + ".partial." + std::to_string(getpid());
    const auto cleanup = [&]() {
        std::error_code ec;
        fs::remove_all(staging, ec);
    };
    cleanup();
    fs::create_directories(staging);

    std::string scope_str = opts.text_only_scope ? "text-only" : "full";
    std::map<std::string, std::string> weight_map;
    uint64_t total_bits = 0;
    uint64_t total_logical = 0;
    uint64_t payload_total = 0;
    uint64_t padding_total = 0;

    const bool multi = shards.size() > 1;
    for (size_t si = 0; si < shards.size(); ++si) {
        const std::string shard_name = multi
            ? "model-" + std::to_string(si + 1).insert(0, 5 - std::to_string(si + 1).size(), '0') +
              "-of-" + std::to_string(shards.size()).insert(0, 5 - std::to_string(shards.size()).size(), '0') +
              ".safetensors"
            : "model.safetensors";
        const std::string shard_path = (fs::path(staging) / shard_name).string();

        auto wres = ps::io::SafetensorsWriter::create(shard_path);
        if (!wres.ok()) { cleanup(); return wres.status(); }
        ps::io::SafetensorsWriter writer = wres.release();

        for (const auto& p : shards[si].tensors) {
            if (p.enc != WeightEncoding::BF16) {
                auto s1 = writer.plan_tensor(p.codes.name, p.codes.dtype, to_size_shape(p.codes.shape));
                auto s2 = writer.plan_tensor(p.metadata1.name, p.metadata1.dtype, to_size_shape(p.metadata1.shape));
                if (!s1.ok()) { cleanup(); return s1; }
                if (!s2.ok()) { cleanup(); return s2; }
            } else {
                auto s1 = writer.plan_tensor(p.data.name, p.data.dtype, to_size_shape(p.data.shape));
                if (!s1.ok()) { cleanup(); return s1; }
            }
        }
        writer.set_metadata("format", "pt");
        writer.set_metadata("phaseshift.format", kQuantizedSafetensorsFormat);
        writer.set_metadata("phaseshift.format_version", std::to_string(kQuantizedSafetensorsFormatVersion));
        writer.set_metadata("phaseshift.quantization_metadata", kQuantizedMetadataFile);

        auto hres = writer.write_header();
        if (!hres.ok()) { cleanup(); return hres.status(); }

        for (const auto& p : shards[si].tensors) {
            if (p.enc != WeightEncoding::BF16) {
                weight_map[p.codes.name] = shard_name;
                weight_map[p.metadata1.name] = shard_name;
            } else {
                weight_map[p.data.name] = shard_name;
            }
        }

        GpuQuantizer gpu;
        bool gpu_created = false;
        if (opts.backend == ConvertBackend::Hip) {
            std::size_t max_input = 0, max_codes = 0, max_scales = 0, max_qw = 0, max_cr = 0;
            for (const auto& p : shards[si].tensors) {
                if (p.enc == WeightEncoding::BF16) continue;
                const uint64_t row_bytes = p.k * p.sv.byte_stride;
                uint64_t cr = 0;
                uint64_t cpb = 32;
                uint64_t scale_row = 0;
                if (p.enc == WeightEncoding::FP8_BLOCK128) {
                    const MatrixShape ms = matrix_shape(p.sv.shape);
                    cr = 128;
                    if (cr > ms.n) cr = ms.n;
                    cpb = 128;
                    scale_row = p.rb * 4u;
                } else {
                    cr = kGpuQuantChunkTargetInputBytes / row_bytes;
                    if (cr < 1) cr = 1;
                    if (cr > kGpuQuantChunkMaxRows) cr = kGpuQuantChunkMaxRows;
                    if (cr > p.rows) cr = p.rows;
                    if (p.enc == WeightEncoding::PSQ4) cpb = 16;
                    else if (p.enc == WeightEncoding::MXFP4) cpb = 16;
                    scale_row = (p.enc == WeightEncoding::MXFP4) ? p.rb : p.rb * 2u;
                }
                const uint64_t scales_bytes = (p.enc == WeightEncoding::FP8_BLOCK128)
                    ? p.rb * 4u : cr * scale_row;
                max_input = std::max(max_input, static_cast<std::size_t>(cr * row_bytes));
                max_codes = std::max(max_codes, static_cast<std::size_t>(cr * p.rb * cpb));
                max_scales = std::max(max_scales, static_cast<std::size_t>(scales_bytes));
                max_qw = std::max(max_qw, static_cast<std::size_t>(p.kp) * sizeof(float));
                max_cr = std::max(max_cr, static_cast<std::size_t>(cr));
            }
            if (max_input > 0) {
                const Status st = gpu_quantizer_create(gpu, max_input, max_codes, max_scales,
                                                       max_qw, max_cr);
                if (!st.ok()) { cleanup(); return st; }
                gpu_created = true;
            }
        }
        const auto cleanup_gpu = [&]() {
            if (gpu_created) { gpu_quantizer_destroy(gpu); gpu_created = false; }
        };

        for (auto& p : shards[si].tensors) {
            std::vector<float> qw;
            bool weighted = false;
            if (imatrix_cov.used && p.enc != WeightEncoding::BF16) {
                auto dit = imatrix_entries.find(p.name);
                if (dit != imatrix_entries.end()) {
                    qw = build_quant_weights(
                        reinterpret_cast<const double*>(psim_file.payload.data() + dit->second.data_offset),
                        dit->second.count, p.k, p.kp);
                    if (qw.empty()) {
                        cleanup_gpu(); cleanup();
                        return Status::invalid_argument(
                            ("imatrix invalid weights: " + p.name).c_str(), __FILE__, __LINE__);
                    }
                    weighted = true;
                }
            }

            Crc32Accumulator codes_crc, metadata1_crc, data_crc;
            if (p.enc == WeightEncoding::PSQ4 || p.enc == WeightEncoding::PSQ8) {
                const bool psq4 = p.enc == WeightEncoding::PSQ4;
                const uint64_t rb = p.rb;
                const uint64_t cpb = psq4 ? 16u : 32u;
                const uint64_t scale_row = rb * 2u;
                if (opts.backend == ConvertBackend::Hip) {
                    if (weighted) {
                        const Status st = gpu_quantizer_upload_quant_weights(gpu, qw.data(), qw.size());
                        if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                    }
                    const uint64_t row_bytes = p.k * p.sv.byte_stride;
                    uint64_t chunk_rows = kGpuQuantChunkTargetInputBytes / row_bytes;
                    if (chunk_rows < 1) chunk_rows = 1;
                    if (chunk_rows > kGpuQuantChunkMaxRows) chunk_rows = kGpuQuantChunkMaxRows;
                    if (chunk_rows > p.rows) chunk_rows = p.rows;
                    GpuTensorSource gsrc;
                    gsrc.dtype = p.sv.dtype;
                    gsrc.stride = p.sv.byte_stride;
                    gsrc.logical_k = p.k;
                    gsrc.padded_k = p.kp;
                    gsrc.blocks_per_row = rb;
                    gsrc.rows = p.rows;
                    const GpuQuantKind kind = psq4 ? GpuQuantKind::Psq4 : GpuQuantKind::Psq8;
                    auto flush_chunk = [&](int slot, uint64_t begin, uint64_t end) -> Status {
                        const uint64_t pn = end - begin;
                        const uint8_t* hc = static_cast<const uint8_t*>(gpu_quantizer_host_codes(gpu, slot));
                        const uint8_t* hs = static_cast<const uint8_t*>(gpu_quantizer_host_scales(gpu, slot));
                        codes_crc.update(hc, pn * rb * cpb);
                        Status st = writer.write_tensor_chunk(p.codes.name, hc, begin * rb * cpb, pn * rb * cpb);
                        if (!st.ok()) return st;
                        metadata1_crc.update(hs, pn * scale_row);
                        st = writer.write_tensor_chunk(p.metadata1.name, hs, begin * scale_row, pn * scale_row);
                        return st;
                    };
                    uint64_t pos = 0;
                    int submitted = 0, pending_slot = 0;
                    uint64_t pending_begin = 0, pending_end = 0;
                    bool have_pending = false;
                    while (pos < p.rows) {
                        const uint64_t r_end = std::min(p.rows, pos + chunk_rows);
                        const int slot = submitted % kGpuQuantSlots;
                        const uint64_t in_bytes = (r_end - pos) * row_bytes;
                        std::memcpy(gpu_quantizer_host_input(gpu, slot), p.sv.data + pos * row_bytes, in_bytes);
                        Status st = gpu_quantize_chunk_submit(gpu, slot, kind, weighted,
                                                              gpu_quantizer_host_input(gpu, slot), gsrc, pos, r_end);
                        if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                        ++submitted;
                        if (have_pending) {
                            st = gpu_quantize_chunk_sync(gpu, pending_slot);
                            if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                            st = flush_chunk(pending_slot, pending_begin, pending_end);
                            if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                        }
                        pending_slot = slot;
                        pending_begin = pos;
                        pending_end = r_end;
                        have_pending = true;
                        pos = r_end;
                    }
                    if (have_pending) {
                        Status st = gpu_quantize_chunk_sync(gpu, pending_slot);
                        if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                        st = flush_chunk(pending_slot, pending_begin, pending_end);
                        if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                    }
                } else {
                    std::vector<float> row(p.kp);
                    std::vector<uint8_t> codes(rb * cpb);
                    std::vector<uint8_t> scales(scale_row);
                    for (uint64_t r = 0; r < p.rows; ++r) {
                        read_row_f32(p.sv, r, p.k, p.kp, row.data());
                        if (psq4) {
                            ps::quantization::psq::quantize_psq4_row(
                                std::span<const float>(row),
                                std::span<uint8_t>(codes),
                                std::span<uint8_t>(scales), p.k, p.kp);
                        } else {
                            ps::quantization::psq::quantize_psq8_row(
                                std::span<const float>(row),
                                std::span<uint8_t>(codes),
                                std::span<uint8_t>(scales), p.k, p.kp);
                        }
                        codes_crc.update(codes.data(), rb * cpb);
                        metadata1_crc.update(scales.data(), scale_row);
                        Status st = writer.write_tensor_chunk(p.codes.name, codes.data(), r * rb * cpb, rb * cpb);
                        if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                        st = writer.write_tensor_chunk(p.metadata1.name, scales.data(), r * scale_row, scale_row);
                        if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                    }
                }
                total_bits += p.payload_bytes * 8;
            } else if (p.enc == WeightEncoding::MXFP4) {
                const uint64_t rb = p.rb;
                const uint64_t codes_row = rb * ps::quantization::mxfp4::kCodesBytesPerBlock;
                const uint64_t scale_row = rb;
                if (opts.backend == ConvertBackend::Hip) {
                    const uint64_t row_bytes = p.k * p.sv.byte_stride;
                    uint64_t chunk_rows = kGpuQuantChunkTargetInputBytes / row_bytes;
                    if (chunk_rows < 1) chunk_rows = 1;
                    if (chunk_rows > kGpuQuantChunkMaxRows) chunk_rows = kGpuQuantChunkMaxRows;
                    if (chunk_rows > p.rows) chunk_rows = p.rows;
                    GpuTensorSource gsrc;
                    gsrc.dtype = p.sv.dtype;
                    gsrc.stride = p.sv.byte_stride;
                    gsrc.logical_k = p.k;
                    gsrc.padded_k = p.kp;
                    gsrc.blocks_per_row = rb;
                    gsrc.rows = p.rows;
                    for (uint64_t pos = 0; pos < p.rows; pos += chunk_rows) {
                        const uint64_t end = std::min(p.rows, pos + chunk_rows);
                        const uint64_t pn = end - pos;
                        std::memcpy(gpu_quantizer_host_input(gpu, 0), p.sv.data + pos * row_bytes,
                                    pn * row_bytes);
                        Status st = gpu_quantize_chunk_submit(gpu, 0, GpuQuantKind::Mxfp4, false,
                                                              gpu_quantizer_host_input(gpu, 0),
                                                              gsrc, pos, end);
                        if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                        st = gpu_quantize_chunk_sync(gpu, 0);
                        if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                        const uint8_t* hc = static_cast<const uint8_t*>(gpu_quantizer_host_codes(gpu, 0));
                        const uint8_t* hs = static_cast<const uint8_t*>(gpu_quantizer_host_scales(gpu, 0));
                        codes_crc.update(hc, pn * codes_row);
                        st = writer.write_tensor_chunk(p.codes.name, hc, pos * codes_row, pn * codes_row);
                        if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                        metadata1_crc.update(hs, pn * scale_row);
                        st = writer.write_tensor_chunk(p.metadata1.name, hs, pos * scale_row, pn * scale_row);
                        if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                    }
                } else {
                    std::vector<float> row(p.kp);
                    std::vector<uint8_t> codes(codes_row);
                    std::vector<uint8_t> scales(scale_row);
                    for (uint64_t r = 0; r < p.rows; ++r) {
                        read_row_f32(p.sv, r, p.k, p.kp, row.data());
                        ps::quantization::mxfp4::quantize_mxfp4_row(
                            std::span<const float>(row),
                            std::span<uint8_t>(codes),
                            std::span<uint8_t>(scales), p.k, p.kp);
                        codes_crc.update(codes.data(), codes_row);
                        metadata1_crc.update(scales.data(), scale_row);
                        Status st = writer.write_tensor_chunk(p.codes.name, codes.data(), r * codes_row, codes_row);
                        if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                        st = writer.write_tensor_chunk(p.metadata1.name, scales.data(), r * scale_row, scale_row);
                        if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                    }
                }
                total_bits += p.payload_bytes * 8;
            } else if (p.enc == WeightEncoding::FP8_BLOCK128) {
                const MatrixShape ms = matrix_shape(p.sv.shape);
                const uint64_t n = ms.n;
                const uint64_t kp = p.kp;
                const uint64_t scale_n = (n + 127u) / 128u;
                const uint64_t scale_k = kp / 128u;
                if (opts.backend == ConvertBackend::Hip) {
                    const uint64_t row_bytes = p.k * p.sv.byte_stride;
                    GpuTensorSource gsrc;
                    gsrc.dtype = p.sv.dtype;
                    gsrc.stride = p.sv.byte_stride;
                    gsrc.logical_k = p.k;
                    gsrc.padded_k = kp;
                    gsrc.blocks_per_row = scale_k;
                    gsrc.rows = n;
                    for (uint64_t m = 0; m < ms.batch; ++m) {
                        for (uint64_t r0 = 0; r0 < n; r0 += 128u) {
                            const uint64_t g = std::min(static_cast<uint64_t>(128), n - r0);
                            const uint64_t global_row = m * n + r0;
                            std::memcpy(gpu_quantizer_host_input(gpu, 0),
                                        p.sv.data + global_row * row_bytes, g * row_bytes);
                            Status st = gpu_quantize_chunk_submit(gpu, 0, GpuQuantKind::Fp8Block128,
                                                                  false, gpu_quantizer_host_input(gpu, 0),
                                                                  gsrc, 0, g);
                            if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                            st = gpu_quantize_chunk_sync(gpu, 0);
                            if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                            const uint8_t* hc = static_cast<const uint8_t*>(gpu_quantizer_host_codes(gpu, 0));
                            const uint8_t* hs = static_cast<const uint8_t*>(gpu_quantizer_host_scales(gpu, 0));
                            const uint64_t codes_len = g * kp;
                            const uint64_t codes_off = global_row * kp;
                            codes_crc.update(hc, codes_len);
                            st = writer.write_tensor_chunk(p.codes.name, hc, codes_off, codes_len);
                            if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                            const uint64_t scale_len = scale_k * sizeof(float);
                            const uint64_t scale_off = (m * scale_n + r0 / 128u) * scale_len;
                            metadata1_crc.update(hs, scale_len);
                            st = writer.write_tensor_chunk(p.metadata1.name, hs, scale_off, scale_len);
                            if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                        }
                    }
                } else {
                    std::vector<float> tile;
                    std::vector<uint8_t> codes;
                    std::vector<float> scales;
                    for (uint64_t m = 0; m < ms.batch; ++m) {
                        for (uint64_t r0 = 0; r0 < n; r0 += 128u) {
                            const uint64_t r1 = std::min(n, r0 + 128u);
                            const uint64_t g = r1 - r0;
                            tile.assign(g * kp, 0.0f);
                            for (uint64_t r = r0; r < r1; ++r) {
                                read_row_f32(p.sv, m * n + r, p.k, kp, tile.data() + (r - r0) * kp);
                            }
                            codes.assign(g * kp, 0u);
                            scales.assign(scale_k, 0.0f);
                            ps::quantization::fp8::quantize_fp8_block128_matrix(
                                tile.data(), kp, g, p.k, kp, codes.data(), scales.data());
                            const uint64_t codes_len = g * kp;
                            const uint64_t codes_off = m * n * kp + r0 * kp;
                            codes_crc.update(codes.data(), codes_len);
                            Status st = writer.write_tensor_chunk(p.codes.name, codes.data(), codes_off, codes_len);
                            if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                            const uint64_t scale_off = (m * scale_n + r0 / 128u) * scale_k * sizeof(float);
                            const uint64_t scale_len = scale_k * sizeof(float);
                            const uint8_t* scale_bytes = reinterpret_cast<const uint8_t*>(scales.data());
                            metadata1_crc.update(scale_bytes, scale_len);
                            st = writer.write_tensor_chunk(p.metadata1.name, scale_bytes, scale_off, scale_len);
                            if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                        }
                    }
                }
                total_bits += p.payload_bytes * 8;
            } else {
                std::vector<uint16_t> b16(p.k);
                for (uint64_t r = 0; r < p.rows; ++r) {
                    bf16_row_bytes(p.sv, r, p.k, b16, &data_crc);
                    Status st = writer.write_tensor_chunk(p.data.name, b16.data(), r * p.k * 2, p.k * 2);
                    if (!st.ok()) { cleanup_gpu(); cleanup(); return st; }
                }
                total_bits += p.data.bytes * 8;
            }
            if (p.enc != WeightEncoding::BF16) {
                p.codes_crc = codes_crc.value();
                p.metadata1_crc = metadata1_crc.value();
            } else {
                p.data_crc = data_crc.value();
            }
            total_logical += element_count(p.sv.shape);
            payload_total += p.payload_bytes;
        }
        cleanup_gpu();
        auto fres = writer.finish();
        if (!fres.ok()) { cleanup(); return fres.status(); }
    }

    std::vector<PlannedTensor> all_tensors;
    for (const auto& s : shards) {
        for (const auto& t : s.tensors) all_tensors.push_back(t);
    }

    QuantizedManifest m;
    m.format = kQuantizedSafetensorsFormat;
    m.format_version = kQuantizedSafetensorsFormatVersion;
    m.architecture = to_string(arch);
    m.preset = to_string(opts.preset);
    m.scope = scope_str;
    m.source_dtype = dominant_dtype(dtype_counts);
    m.source_model_fingerprint = compute_model_fingerprint(opts.input_dir);
    m.logical_parameter_count = total_logical;
    m.payload_bytes = payload_total;
    m.padding_bytes = padding_total;
    m.effective_payload_bpw = total_logical > 0
        ? static_cast<double>(total_bits) / static_cast<double>(total_logical) : 0.0;

    if (tie) {
        for (const auto& p : all_tensors) {
            if (p.info.role == TensorRole::TokenEmbedding) {
                m.aliases["lm_head.weight"] = p.name;
                break;
            }
        }
    }

    if (imatrix_cov.used) {
        QuantizedImatrixMetadata im;
        im.enabled = true;
        im.policy = opts.imatrix_policy == ImatrixPolicy::Strict ? "strict" : "best-effort";
        std::ifstream psim_in(opts.imatrix_path, std::ios::binary);
        std::string psim_bytes((std::istreambuf_iterator<char>(psim_in)),
                               std::istreambuf_iterator<char>());
        im.sha256 = sha256_hex(psim_bytes.data(), psim_bytes.size());
        im.model_fingerprint = psim_file.header.model.fingerprint;
        im.corpus_sha256 = psim_file.header.calibration.corpus_sha256;
        im.required = imatrix_cov.required;
        im.matched = imatrix_cov.found;
        m.imatrix = im;
    }

    for (const auto& p : all_tensors) {
        QuantizedTensorMetadata tm;
        tm.role = to_string(p.info.role);
        tm.encoding = (p.enc == WeightEncoding::PSQ4) ? QuantizedEncoding::Psq4
                     : (p.enc == WeightEncoding::PSQ8) ? QuantizedEncoding::Psq8
                     : (p.enc == WeightEncoding::FP8_BLOCK128) ? QuantizedEncoding::Fp8Block128
                     : (p.enc == WeightEncoding::MXFP4) ? QuantizedEncoding::Mxfp4
                     : QuantizedEncoding::Bf16;
        if (p.enc == WeightEncoding::PSQ4) tm.codebook = "cb10";
        if (p.enc == WeightEncoding::FP8_BLOCK128) tm.layout = kFp8Block128LayoutName;
        if (p.enc == WeightEncoding::MXFP4) tm.layout = kMxfp4LayoutName;
        tm.logical_shape = p.sv.shape;
        tm.k_padded = p.kp;
        if (p.enc != WeightEncoding::BF16) {
            const char* metadata_dtype = (p.enc == WeightEncoding::FP8_BLOCK128) ? "F32" : "U8";
            tm.codes = QuantizedTensorRef{p.codes.name, "U8", crc_hex(p.codes_crc)};
            tm.metadata1 = QuantizedTensorRef{p.metadata1.name, metadata_dtype, crc_hex(p.metadata1_crc)};
        } else {
            tm.data = QuantizedTensorRef{p.data.name, "BF16", crc_hex(p.data_crc)};
        }
        m.tensors[p.name] = std::move(tm);
    }

    (void)weight_map;

    auto asset_st = copy_assets(opts.input_dir, staging);
    if (!asset_st.ok()) { cleanup(); return asset_st; }

    nlohmann::json out_config = config;
    out_config["phaseshift_quantization"] = {
        {"format", kQuantizedSafetensorsFormat},
        {"format_version", kQuantizedSafetensorsFormatVersion},
        {"metadata_file", kQuantizedMetadataFile},
    };
    {
        std::ofstream cf(fs::path(staging) / "config.json", std::ios::trunc);
        cf << out_config.dump(2) << "\n";
    }

    if (multi) {
        uint64_t total_size = 0;
        for (const auto& p : all_tensors) total_size += p.payload_bytes;
        nlohmann::json idx;
        idx["metadata"]["total_size"] = total_size;
        idx["weight_map"] = weight_map;
        std::ofstream of(fs::path(staging) / "model.safetensors.index.json", std::ios::trunc);
        of << idx.dump(2) << "\n";
    }

    auto ser = serialize_quantized_manifest(m);
    if (!ser.ok()) { cleanup(); return ser.status(); }
    {
        std::ofstream mf(fs::path(staging) / kQuantizedMetadataFile, std::ios::trunc);
        mf << ser.value();
    }

    if (opts.verify) {
        auto vr = QuantizedModelReader::open(staging);
        if (!vr.ok()) { cleanup(); return vr.status(); }
        auto vres = vr.value().list_logical_tensors();
        if (!vres.ok()) { cleanup(); return vres.status(); }
        for (const auto& lname : vres.value()) {
            auto rres = vr.value().resolve(lname);
            if (!rres.ok()) { cleanup(); return rres.status(); }
        }
    }

    if (fs::exists(opts.output_dir)) {
        std::error_code ec;
        fs::remove_all(opts.output_dir, ec);
    }
    std::error_code ec;
    fs::rename(staging, opts.output_dir, ec);
    if (ec) {
        cleanup();
        return Status::invalid_argument(
            ("rename to output failed: " + ec.message()).c_str(), __FILE__, __LINE__);
    }

    QuantizeSummary summary;
    summary.manifest = m;
    summary.shards = shards.size();
    summary.logical_parameter_count = total_logical;
    summary.padding_bytes = padding_total;
    summary.skipped_tensors = skipped_tensors;
    summary.imatrix = imatrix_cov;
    summary.effective_bpw = m.effective_payload_bpw;
    summary.self_contained = true;
    for (const auto& p : all_tensors) {
        FormatSummary& fs = summary.formats[p.enc];
        fs.tensors += 1;
        fs.parameters += element_count(p.sv.shape);
        if (p.enc == WeightEncoding::BF16) {
            fs.codes_bytes += p.data.bytes;
        } else {
            fs.codes_bytes += p.codes.bytes;
            fs.metadata_bytes += p.metadata1.bytes;
        }
    }
    return summary;
}

}  // namespace

Result<QuantizeSummary> quantize_model(const QuantizeOptions& opts) {
    const bool block_scaled =
        opts.preset == FpxPreset::Fp8 || opts.preset == FpxPreset::Mxfp4;
    if (block_scaled && !opts.imatrix_path.empty()) {
        return Status::invalid_argument(
            "iMatrix weighting is not supported for the fp8/mxfp4 presets",
            __FILE__, __LINE__);
    }
    return run_quantize(opts);
}

}
