#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

std::string exe_dir() {
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return std::string(".");
    buf[n] = '\0';
    const std::string p(buf);
    const auto pos = p.find_last_of('/');
    return pos == std::string::npos ? std::string(".") : p.substr(0, pos);
}

int run_capture(const std::string& cmd, std::string& out) {
    out.clear();
    FILE* f = popen((cmd + " 2>&1").c_str(), "r");
    if (f == nullptr) return -1;
    char line[8192];
    while (std::fgets(line, sizeof(line), f) != nullptr) out += line;
    const int status = pclose(f);
    if (status == -1) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

std::string extract_line(const std::string& out, const std::string& key) {
    const auto pos = out.find(key);
    if (pos == std::string::npos) return std::string();
    const auto start = pos + key.size();
    const auto end = out.find('\n', start);
    return out.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

std::vector<int32_t> read_ids_file(const std::string& path) {
    std::vector<int32_t> ids;
    std::ifstream in(path);
    if (!in) return ids;
    std::string token;
    while (in >> token) {
        std::size_t begin = 0;
        while (begin <= token.size()) {
            const auto comma = token.find(',', begin);
            const std::string piece = token.substr(
                begin, comma == std::string::npos ? std::string::npos : comma - begin);
            if (!piece.empty()) ids.push_back(std::atoi(piece.c_str()));
            if (comma == std::string::npos) break;
            begin = comma + 1;
        }
    }
    return ids;
}

}  // namespace

int main() {
    const char* draft_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2");
    const char* target_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2_TARGET");
    if (draft_dir == nullptr || draft_dir[0] == '\0' || target_dir == nullptr ||
        target_dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2 and _TARGET\n");
        return 77;
    }
    const char* fix_env = std::getenv("PHASESHIFT_DFLASH2_GATE5_FIXTURE");
    const std::string fixture_root =
        (fix_env != nullptr && fix_env[0] != '\0') ? std::string(fix_env)
                                                   : std::string("build/dflash2-gate51/real");
    const char* bin_env = std::getenv("PHASESHIFT_COMPUTE_BIN");
    const std::string compute_bin =
        (bin_env != nullptr && bin_env[0] != '\0') ? std::string(bin_env)
                                                   : (exe_dir() + "/../phaseshift-compute");

    struct Case {
        std::string name;
        std::string prompt_path;
        uint32_t max_new;
    };
    std::vector<Case> cases;
    cases.push_back({"prose", fixture_root + "/real_prose/prompt.txt", 256u});
    cases.push_back({"code", fixture_root + "/real_code/prompt.txt", 128u});
    cases.push_back({"math", fixture_root + "/real_math/prompt.txt", 128u});

    for (const Case& c : cases) {
        const std::vector<int32_t> ids = read_ids_file(c.prompt_path);
        if (ids.empty()) {
            std::printf("SKIP: prompt missing %s\n", c.prompt_path.c_str());
            return 77;
        }
        char prompt_path[256];
        std::snprintf(prompt_path, sizeof(prompt_path), "/tmp/phaseshift_g11c_%s_%ld.txt",
                      c.name.c_str(), static_cast<long>(getpid()));
        {
            FILE* f = std::fopen(prompt_path, "w");
            if (f == nullptr) {
                std::printf("FAIL cannot write %s\n", prompt_path);
                return 1;
            }
            for (int32_t id : ids) std::fprintf(f, "%d\n", id);
            std::fclose(f);
        }
        const std::string common = "'" + compute_bin + "' --model-dir '" + target_dir +
                                   "' --dflash2-model-dir '" + draft_dir +
                                   "' --dflash2-drafts 7 --input-ids-file '" + prompt_path +
                                   "' --max-new-tokens " + std::to_string(c.max_new) +
                                   " --max-seq-len 4096 --arena-gib 26 --temperature 0 --device 0";

        std::string hist_out;
        const int hist_rc = run_capture(common, hist_out);
        std::string ref_out;
        const int ref_rc =
            run_capture("PHASESHIFT_DFLASH2_GDN_RERUN_REFERENCE=1 " + common, ref_out);
        check(hist_rc == 0, c.name + " history exit");
        check(ref_rc == 0, c.name + " reference exit");
        const std::string hist_ids = extract_line(hist_out, "GENERATED_IDS=");
        const std::string ref_ids = extract_line(ref_out, "GENERATED_IDS=");
        const std::string hist_reruns = extract_line(hist_out, "DFLASH2_RERUNS=");
        const std::string ref_reruns = extract_line(ref_out, "DFLASH2_RERUNS=");
        const bool match = !hist_ids.empty() && hist_ids == ref_ids;
        check(match, c.name + " GENERATED_IDS history==reference");
        check(hist_reruns == "0", c.name + " history reruns=0");
        std::printf("GATE11C_E2E case=%s tokens=%u ids_match=%s hist_reruns=%s ref_reruns=%s\n",
                    c.name.c_str(), c.max_new, match ? "yes" : "no", hist_reruns.c_str(),
                    ref_reruns.c_str());
        if (!match) {
            std::printf("  hist=%s\n  ref =%s\n", hist_ids.c_str(), ref_ids.c_str());
        }
        std::remove(prompt_path);
    }

    std::printf("test_dflash2_gate11c_e2e: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
