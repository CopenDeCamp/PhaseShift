#include <phaseshift/quantization/offline/ppl.h>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

void usage(FILE* out) {
    std::fprintf(out,
        "Usage: phaseshift-quantizer ppl --model-dir <dir> --tokens <file> [options]\n"
        "  --model-dir <dir>   model directory to evaluate\n"
        "  --tokens <file>     PSKLDTOK corpus file\n"
        "  --max-tokens N      tokens to evaluate (default 2048)\n"
        "  --arena-gib N       arena size in GiB (default 24)\n"
        "  --device N          HIP device index (default 0)\n"
        "  --help              show this help\n");
}

}

int run_ppl(int argc, char** argv) {
    ps::quantization::fpx::PplOptions opts;
    bool have_model = false;
    bool have_tokens = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--model-dir") {
            const char* v = need_value("--model-dir");
            if (!v) { usage(stderr); return 2; }
            opts.model_dir = v;
            have_model = true;
        } else if (arg == "--tokens") {
            const char* v = need_value("--tokens");
            if (!v) { usage(stderr); return 2; }
            opts.tokens_path = v;
            have_tokens = true;
        } else if (arg == "--max-tokens") {
            const char* v = need_value("--max-tokens");
            if (!v) { usage(stderr); return 2; }
            opts.max_tokens = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
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
    if (!have_model || !have_tokens) {
        usage(stderr);
        return 2;
    }
    if (opts.max_tokens < 2 || opts.arena_gib <= 0.0) {
        std::fprintf(stderr, "--max-tokens/--arena-gib invalid\n");
        usage(stderr);
        return 2;
    }
    auto result = ps::quantization::fpx::evaluate_ppl(opts);
    if (!result.ok()) {
        std::fprintf(stderr, "ppl evaluation failed: %s\n", result.status().message().c_str());
        return 1;
    }
    const auto& summary = result.value();
    std::printf("positions %llu\n", static_cast<unsigned long long>(summary.positions));
    std::printf("mean_nll %.6f\n", summary.mean_nll);
    std::printf("perplexity %.6f\n", summary.perplexity);
    std::printf("argmax_first %d\n", summary.argmax_first);
    return 0;
}
