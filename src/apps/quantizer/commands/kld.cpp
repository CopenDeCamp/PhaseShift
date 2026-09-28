#include <phaseshift/quantization/offline/kld.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

void usage(FILE* out) {
    std::fprintf(out,
        "Usage: phaseshift-quantizer kld --input <dir> --tokens <file> --report <path> [options]\n"
        "  --input <dir>            original BF16 model directory\n"
        "  --bundle <dir>           quantized model directory\n"
        "  --tokens <file>          PSKLDTOK corpus file\n"
        "  --report <path>          JSON report output path\n"
        "  --window N               window size (default 512)\n"
        "  --stride N               stride (default 512)\n"
        "  --max-eval-tokens N      max tokens to evaluate (default all)\n"
        "  --cache-dir DIR          cache directory (default /tmp/phaseshift-kld)\n"
        "  --keep-shadow            keep shadow model directory\n"
        "  --keep-logit-cache       keep baseline logit cache\n"
        "  --self                   self-test mode (no --bundle required)\n"
        "  --batch-positions N      batch positions (default 16)\n"
        "  --arena-gib N            arena size in GiB (default 20)\n"
        "  --help                   show this help\n");
}

}

int run_kld(int argc, char** argv) {
    ps::quantization::fpx::KldOptions opts;
    bool have_input = false;
    bool have_tokens = false;
    bool have_report = false;
    bool have_bundle = false;
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
        } else if (arg == "--bundle") {
            const char* v = need_value("--bundle");
            if (!v) { usage(stderr); return 2; }
            opts.bundle_dir = v;
            have_bundle = true;
        } else if (arg == "--tokens") {
            const char* v = need_value("--tokens");
            if (!v) { usage(stderr); return 2; }
            opts.tokens_path = v;
            have_tokens = true;
        } else if (arg == "--report") {
            const char* v = need_value("--report");
            if (!v) { usage(stderr); return 2; }
            opts.report_path = v;
            have_report = true;
        } else if (arg == "--window") {
            const char* v = need_value("--window");
            if (!v) { usage(stderr); return 2; }
            opts.window = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
        } else if (arg == "--stride") {
            const char* v = need_value("--stride");
            if (!v) { usage(stderr); return 2; }
            opts.stride = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
        } else if (arg == "--max-eval-tokens") {
            const char* v = need_value("--max-eval-tokens");
            if (!v) { usage(stderr); return 2; }
            opts.max_eval_tokens = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
        } else if (arg == "--cache-dir") {
            const char* v = need_value("--cache-dir");
            if (!v) { usage(stderr); return 2; }
            opts.cache_dir = v;
        } else if (arg == "--keep-shadow") {
            opts.keep_shadow = true;
        } else if (arg == "--keep-logit-cache") {
            opts.keep_logit_cache = true;
        } else if (arg == "--self") {
            opts.self_test = true;
        } else if (arg == "--batch-positions") {
            const char* v = need_value("--batch-positions");
            if (!v) { usage(stderr); return 2; }
            opts.batch_positions = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
        } else if (arg == "--arena-gib") {
            const char* v = need_value("--arena-gib");
            if (!v) { usage(stderr); return 2; }
            opts.arena_gib = std::strtod(v, nullptr);
        } else if (arg == "--help" || arg == "-h") {
            usage(stdout);
            return 0;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            usage(stderr);
            return 2;
        }
    }
    if (!have_input || !have_tokens || !have_report) {
        usage(stderr);
        return 2;
    }
    if (!opts.self_test && !have_bundle) {
        usage(stderr);
        return 2;
    }
    if (opts.window == 0 || opts.stride == 0 || opts.batch_positions == 0 || opts.arena_gib <= 0.0) {
        std::fprintf(stderr, "--window/--stride/--batch-positions/--arena-gib must be > 0\n");
        usage(stderr);
        return 2;
    }
    auto result = ps::quantization::fpx::evaluate_kld(opts);
    if (!result.ok()) {
        std::fprintf(stderr, "KLD evaluation failed: %s\n", result.status().message().c_str());
        return 1;
    }
    const auto& summary = result.value();
    const auto& kld = summary.kld;
    std::printf("KLD mean %.6g p50 %.6g p90 %.6g p99 %.6g p99.9 %.6g max %.6g\n",
        kld.mean, kld.p50, kld.p90, kld.p99, kld.p99_9, kld.max);
    std::printf("report %s\n", opts.report_path.c_str());
    std::printf("evaluated_positions %zu\n", summary.evaluated_positions);
    return 0;
}
