#include <phaseshift/quantization/offline/imatrix_collect.h>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

void usage(FILE* out) {
    std::fprintf(out,
        "Usage: phaseshift-quantizer imatrix --input <dir> --tokens <file> --output <path> [options]\n"
        "  --input <dir>                original BF16 model directory (fingerprint source)\n"
        "  --model-dir <dir>            model directory to run (default: --input)\n"
        "  --tokens <file>              PSKLDTOK corpus file\n"
        "  --output <path>              .psim output path\n"
        "  --window N                   window size (default 512)\n"
        "  --stride N                   stride (default 512)\n"
        "  --max-calibration-tokens N   max tokens to calibrate (default all)\n"
        "  --flush-windows N            device flush period in windows (default 16)\n"
        "  --arena-gib N                arena size in GiB (default 24)\n"
        "  --device N                   HIP device index (default 0)\n"
        "  --help                       show this help\n");
}

}

int run_imatrix(int argc, char** argv) {
    ps::quantization::imatrix::ImatrixOptions opts;
    bool have_input = false;
    bool have_tokens = false;
    bool have_output = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--input") {
            const char* v = need_value("--input");
            if (!v) { usage(stderr); return 2; }
            opts.input_dir = v;
            have_input = true;
        } else if (arg == "--model-dir") {
            const char* v = need_value("--model-dir");
            if (!v) { usage(stderr); return 2; }
            opts.model_dir = v;
        } else if (arg == "--tokens") {
            const char* v = need_value("--tokens");
            if (!v) { usage(stderr); return 2; }
            opts.tokens_path = v;
            have_tokens = true;
        } else if (arg == "--output") {
            const char* v = need_value("--output");
            if (!v) { usage(stderr); return 2; }
            opts.output_path = v;
            have_output = true;
        } else if (arg == "--window") {
            const char* v = need_value("--window");
            if (!v) { usage(stderr); return 2; }
            opts.window = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
        } else if (arg == "--stride") {
            const char* v = need_value("--stride");
            if (!v) { usage(stderr); return 2; }
            opts.stride = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
        } else if (arg == "--max-calibration-tokens") {
            const char* v = need_value("--max-calibration-tokens");
            if (!v) { usage(stderr); return 2; }
            opts.max_calibration_tokens = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
        } else if (arg == "--flush-windows") {
            const char* v = need_value("--flush-windows");
            if (!v) { usage(stderr); return 2; }
            opts.flush_windows = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
        } else if (arg == "--arena-gib") {
            const char* v = need_value("--arena-gib");
            if (!v) { usage(stderr); return 2; }
            opts.arena_gib = std::strtod(v, nullptr);
        } else if (arg == "--device") {
            const char* v = need_value("--device");
            if (!v) { usage(stderr); return 2; }
            opts.device = static_cast<int>(std::strtol(v, nullptr, 10));
        } else if (arg == "--help" || arg == "-h") {
            usage(stdout);
            return 0;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            usage(stderr);
            return 2;
        }
    }
    if (!have_input || !have_tokens || !have_output) {
        usage(stderr);
        return 2;
    }
    if (opts.window < 2 || opts.stride == 0 || opts.flush_windows == 0 || opts.arena_gib <= 0.0) {
        std::fprintf(stderr, "--window/--stride/--flush-windows/--arena-gib invalid\n");
        usage(stderr);
        return 2;
    }
    auto result = ps::quantization::imatrix::collect_imatrix(opts);
    if (!result.ok()) {
        std::fprintf(stderr, "imatrix calibration failed: %s\n", result.status().message().c_str());
        return 1;
    }
    const auto& summary = result.value();
    std::printf("positions %llu\n", static_cast<unsigned long long>(summary.positions));
    std::printf("windows %llu\n", static_cast<unsigned long long>(summary.windows));
    std::printf("entries %llu\n", static_cast<unsigned long long>(summary.entries));
    std::printf("output %s\n", opts.output_path.c_str());
    return 0;
}
