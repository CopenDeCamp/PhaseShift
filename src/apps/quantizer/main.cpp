#include <cstdio>
#include <cstring>
#include <string>

namespace {

void usage(FILE* f) {
    std::fprintf(f,
        "Usage: phaseshift-quantizer <command> [options]\n"
        "\n"
        "Commands:\n"
        "  quantize   produce a self-contained PhaseShift quantized safetensors model\n"
        "  verify     verify a self-contained quantized model directory\n"
        "  kld        evaluate KLD between BF16 and quantized model\n"
        "  imatrix    collect calibration activation statistics into a .psim file\n"
        "  ppl        teacher-forced NLL / perplexity of a model on a corpus\n"
        "\n"
        "Run 'phaseshift-quantizer <command> --help' for command options.\n");
}

}  // namespace

int run_quantize(int argc, char** argv);
int run_verify(int argc, char** argv);
int run_kld(int argc, char** argv);
int run_imatrix(int argc, char** argv);
int run_ppl(int argc, char** argv);

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(stderr);
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "quantize") return run_quantize(argc - 1, argv + 1);
    if (cmd == "verify") return run_verify(argc - 1, argv + 1);
    if (cmd == "kld") return run_kld(argc - 1, argv + 1);
    if (cmd == "imatrix") return run_imatrix(argc - 1, argv + 1);
    if (cmd == "ppl") return run_ppl(argc - 1, argv + 1);
    if (cmd == "-h" || cmd == "--help") {
        usage(stdout);
        return 0;
    }
    std::fprintf(stderr, "unknown command: %s\n", cmd.c_str());
    usage(stderr);
    return 2;
}
