#include <phaseshift/quantization/offline/quantizer.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace ps::quantization::fpx;

namespace {

void usage(FILE* f) {
    std::fprintf(f,
        "Usage: phaseshift-quantizer quantize --input <model-dir> --output <quant-dir> \\\n"
        "      --preset psq|fp8|mxfp4 --backend cpu|hip \\\n"
        "      [--scope text-only] [--imatrix <file.psim>] [--imatrix-policy strict|best-effort] \\\n"
        "      [--max-shard-size 4GiB] [--overwrite] [--no-verify]\n"
        "  phaseshift-quantizer verify <quant-dir>\n");
}

bool valid_preset(const std::string& s) {
    return s == "psq" || s == "fp8" || s == "mxfp4";
}

std::string with_commas(uint64_t v) {
    std::string s = std::to_string(v);
    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) {
        s.insert(static_cast<size_t>(i), ",");
    }
    return s;
}

std::string human_bytes(uint64_t v) {
    static const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double d = static_cast<double>(v);
    int i = 0;
    while (d >= 1024.0 && i < 4) {
        d /= 1024.0;
        ++i;
    }
    char buf[32];
    if (i == 0) std::snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(v));
    else std::snprintf(buf, sizeof(buf), "%.2f %s", d, kUnits[i]);
    return buf;
}

void print_summary(const QuantizeSummary& s) {
    const auto& m = s.manifest;
    std::printf("\nPhaseShift Quantization Complete\n\n");
    std::printf("  %-13s phaseshift-fpx-safetensors v%u\n", "Format", m.format_version);
    std::printf("  %-13s %s\n", "Architecture", m.architecture.c_str());
    std::printf("  %-13s %s\n", "Preset", m.preset.c_str());
    std::printf("  %-13s %s\n", "Scope", m.scope.c_str());
    std::printf("  %-13s %llu\n", "Shards", static_cast<unsigned long long>(s.shards));
    std::printf("\n");

    struct Row {
        const char* label;
        WeightEncoding enc;
        bool is_bf16;
    };
    const Row rows[] = {
        {"BF16", WeightEncoding::BF16, true},
        {"PSQ4", WeightEncoding::PSQ4, false},
        {"PSQ8", WeightEncoding::PSQ8, false},
        {"FP8", WeightEncoding::FP8_BLOCK128, false},
        {"MXFP4", WeightEncoding::MXFP4, false},
    };

    const std::string sep(96, '-');
    std::printf("  %-12s %8s %15s %15s %15s %15s %8s\n",
                "Encoding", "Tensors", "Parameters", "Codes (B)", "Scales (B)", "Total (B)", "BPW");
    std::printf("  %s\n", sep.c_str());
    uint64_t total_tensors = 0;
    uint64_t total_params = 0;
    uint64_t total_codes = 0;
    uint64_t total_metadata = 0;
    uint64_t total_bf16 = 0;
    uint64_t quantized_params = 0;
    for (const Row& r : rows) {
        FormatSummary fs;
        auto it = s.formats.find(r.enc);
        if (it != s.formats.end()) fs = it->second;
        const uint64_t codes = r.is_bf16 ? 0 : fs.codes_bytes;
        const uint64_t scales = r.is_bf16 ? 0 : fs.metadata_bytes;
        const uint64_t total = r.is_bf16 ? fs.codes_bytes : fs.total_bytes();
        char bpw[16];
        if (fs.parameters > 0) {
            std::snprintf(bpw, sizeof(bpw), "%.3f",
                          static_cast<double>(total) * 8.0 / static_cast<double>(fs.parameters));
        } else {
            std::snprintf(bpw, sizeof(bpw), "-");
        }
        const std::string codes_s = codes > 0 ? with_commas(codes) : std::string("-");
        const std::string scales_s = scales > 0 ? with_commas(scales) : std::string("-");
        std::printf("  %-12s %8s %15s %15s %15s %15s %8s\n",
                    r.label,
                    with_commas(fs.tensors).c_str(),
                    with_commas(fs.parameters).c_str(),
                    codes_s.c_str(),
                    scales_s.c_str(),
                    with_commas(total).c_str(),
                    bpw);
        total_tensors += fs.tensors;
        total_params += fs.parameters;
        if (r.is_bf16) {
            total_bf16 += fs.codes_bytes;
        } else {
            total_codes += fs.codes_bytes;
            total_metadata += fs.metadata_bytes;
            quantized_params += fs.parameters;
        }
    }
    const uint64_t total_bytes = total_codes + total_metadata + total_bf16;
    char total_bpw[16];
    if (total_params > 0) {
        std::snprintf(total_bpw, sizeof(total_bpw), "%.3f",
                      static_cast<double>(total_bytes) * 8.0 / static_cast<double>(total_params));
    } else {
        std::snprintf(total_bpw, sizeof(total_bpw), "-");
    }
    std::printf("  %s\n", sep.c_str());
    std::printf("  %-12s %8s %15s %15s %15s %15s %8s\n",
                "Total",
                with_commas(total_tensors).c_str(),
                with_commas(total_params).c_str(),
                with_commas(total_codes).c_str(),
                with_commas(total_metadata).c_str(),
                with_commas(total_bytes).c_str(),
                total_bpw);
    std::printf("\n");

    const double quantized_bpw = quantized_params > 0
        ? static_cast<double>(total_codes + total_metadata) * 8.0 /
              static_cast<double>(quantized_params)
        : 0.0;
    std::printf("  %-23s %s\n", "Logical parameters", with_commas(s.logical_parameter_count).c_str());
    std::printf("  %-23s %s  (%s)\n", "Native payload",
                with_commas(m.payload_bytes).c_str(), human_bytes(m.payload_bytes).c_str());
    std::printf("  %-23s %.4f\n", "Effective payload BPW", s.effective_bpw);
    std::printf("  %-23s %.3f\n", "Quantized storage BPW", quantized_bpw);
    if (s.imatrix.used) {
        std::printf("  %-23s enabled (%llu / %llu)\n", "iMatrix",
                    static_cast<unsigned long long>(s.imatrix.found),
                    static_cast<unsigned long long>(s.imatrix.required));
    } else {
        std::printf("  %-23s disabled\n", "iMatrix");
    }
    std::printf("  %-23s %s\n", "Self-contained", s.self_contained ? "YES" : "NO");
    std::printf("  %-23s %s\n", "Original BF16 required", "NO");
    std::printf("\n");
}

}

int run_quantize(int argc, char** argv) {
    QuantizeOptions opts;
    std::string preset = "psq";
    std::string backend = "cpu";
    std::string max_shard = "4GiB";
    bool no_verify = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--input") { const char* v = next("--input"); if (!v) return 2; opts.input_dir = v; }
        else if (a == "--output") { const char* v = next("--output"); if (!v) return 2; opts.output_dir = v; }
        else if (a == "--preset") {
            const char* v = next("--preset");
            if (!v) return 2;
            if (!valid_preset(v)) {
                std::fprintf(stderr, "unsupported preset: %s\n", v);
                return 2;
            }
            preset = v;
        }
        else if (a == "--backend") { const char* v = next("--backend"); if (!v) return 2; backend = v; }
        else if (a == "--scope") { const char* v = next("--scope"); if (!v) return 2; opts.text_only_scope = std::string(v) == "text-only"; }
        else if (a == "--imatrix") { const char* v = next("--imatrix"); if (!v) return 2; opts.imatrix_path = v; }
        else if (a == "--imatrix-policy") { const char* v = next("--imatrix-policy"); if (!v) return 2; opts.imatrix_policy = std::string(v) == "best-effort" ? ImatrixPolicy::BestEffort : ImatrixPolicy::Strict; }
        else if (a == "--max-shard-size") { const char* v = next("--max-shard-size"); if (!v) return 2; max_shard = v; }
        else if (a == "--overwrite") { opts.overwrite = true; }
        else if (a == "--no-verify") { no_verify = true; }
        else if (a == "-h" || a == "--help") { usage(stdout); return 0; }
        else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }

    if (opts.input_dir.empty() || opts.output_dir.empty()) {
        usage(stderr);
        return 2;
    }

    if (preset == "fp8") opts.preset = FpxPreset::Fp8;
    else if (preset == "mxfp4") opts.preset = FpxPreset::Mxfp4;
    else opts.preset = FpxPreset::Psq;
    if (backend == "hip") opts.backend = ConvertBackend::Hip;
    else opts.backend = ConvertBackend::Cpu;
    opts.verify = !no_verify;
    auto size_res = parse_byte_size(max_shard);
    if (!size_res.ok()) {
        std::fprintf(stderr, "bad --max-shard-size: %s\n", size_res.status().message().c_str());
        return 2;
    }
    opts.max_shard_size = size_res.value();

    auto result = quantize_model(opts);
    if (!result.ok()) {
        std::fprintf(stderr, "quantization failed: %s\n", result.status().message().c_str());
        return 1;
    }
    print_summary(result.value());
    return 0;
}

int run_verify(int argc, char** argv) {
    if (argc < 2) {
        usage(stderr);
        return 2;
    }
    if (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0) {
        usage(stdout);
        return 0;
    }
    auto r = validate_model_dir(argv[1]);
    if (!r.ok()) {
        std::fprintf(stderr, "validation failed: %s\n", r.status().message().c_str());
        return 1;
    }
    auto list = r.value().list_logical_tensors();
    if (!list.ok()) {
        std::fprintf(stderr, "validation failed: %s\n", list.status().message().c_str());
        return 1;
    }
    for (const auto& n : list.value()) {
        auto res = r.value().resolve(n);
        if (!res.ok()) {
            std::fprintf(stderr, "validation failed resolving %s: %s\n",
                         n.c_str(), res.status().message().c_str());
            return 1;
        }
    }
    std::printf("Validation OK: %zu logical tensors resolved\n", list.value().size());
    return 0;
}
