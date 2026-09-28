#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

bool parse_generated(const std::string& out, std::vector<int32_t>& ids) {
    ids.clear();
    const std::string key = "GENERATED_IDS=";
    const auto pos = out.find(key);
    if (pos == std::string::npos) return false;
    const auto start = pos + key.size();
    const auto end = out.find('\n', start);
    const std::string payload =
        out.substr(start, end == std::string::npos ? std::string::npos : end - start);
    std::size_t i = 0;
    while (i < payload.size()) {
        while (i < payload.size() && (payload[i] == ',' || payload[i] == ' ')) ++i;
        if (i >= payload.size()) break;
        char* e = nullptr;
        const long v = std::strtol(payload.c_str() + i, &e, 10);
        if (e == payload.c_str() + i) break;
        ids.push_back(static_cast<int32_t>(v));
        i = static_cast<std::size_t>(e - payload.c_str());
    }
    return !ids.empty();
}

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

const int32_t kPromptIds[] = {
    760, 3712, 314, 279, 12386, 19825, 42824, 777, 1056, 264, 15706,
    1578, 11, 494, 1141, 26814, 34063, 383, 279, 13558, 314, 279,
    345, 8365, 310, 279, 4255, 314, 88047, 1125, 13};

}  // namespace

int main() {
    const char* draft_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2");
    const char* target_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2_TARGET");
    if (draft_dir == nullptr || draft_dir[0] == '\0' || target_dir == nullptr ||
        target_dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2 and _TARGET\n");
        return 77;
    }
    const char* bin_env = std::getenv("PHASESHIFT_COMPUTE_BIN");
    const std::string compute_bin =
        (bin_env != nullptr && bin_env[0] != '\0')
            ? std::string(bin_env)
            : (exe_dir() + "/../phaseshift-compute");

    char prompt_path[256];
    std::snprintf(prompt_path, sizeof(prompt_path), "/tmp/phaseshift_gate10_ids_%ld.txt",
                  static_cast<long>(getpid()));
    {
        FILE* f = std::fopen(prompt_path, "w");
        if (f == nullptr) {
            std::printf("FAIL cannot write prompt file %s\n", prompt_path);
            return 1;
        }
        for (std::size_t i = 0; i < sizeof(kPromptIds) / sizeof(kPromptIds[0]); ++i) {
            std::fprintf(f, "%d\n", kPromptIds[i]);
        }
        std::fclose(f);
    }

    const std::string max_new = "64";
    const std::string common = "'" + compute_bin + "' --model-dir '" + target_dir +
                               "' --input-ids-file '" + prompt_path + "' --max-new-tokens " +
                               max_new + " --max-seq-len 512 --arena-gib 24 --device 0";
    const std::string dflash_common = common + " --dflash2-model-dir '" + draft_dir +
                                      "' --dflash2-drafts 7";

    std::string target_out;
    const int target_rc = run_capture(common + " --temperature 0", target_out);
    std::vector<int32_t> target_ids;
    const bool target_ok = parse_generated(target_out, target_ids);
    check(target_rc == 0, "target-only exit code");
    check(target_ok, "target-only GENERATED_IDS");
    if (target_rc != 0 || !target_ok) {
        std::printf("--- target-only output ---\n%s\n", target_out.c_str());
    }

    std::string dflash_out;
    const int dflash_rc = run_capture(dflash_common + " --temperature 0 --dflash2-stats 1",
                                      dflash_out);
    std::vector<int32_t> dflash_ids;
    const bool dflash_ok = parse_generated(dflash_out, dflash_ids);
    check(dflash_rc == 0, "dflash exit code");
    check(dflash_ok, "dflash GENERATED_IDS");
    check(contains(dflash_out, "DFLASH2_ROUNDS="), "dflash stats rounds");
    check(contains(dflash_out, "DFLASH2_VERIFY_NUMERIC_MODE=Exact"),
          "dflash stats verify mode");
    if (dflash_rc != 0 || !dflash_ok) {
        std::printf("--- dflash output ---\n%s\n", dflash_out.c_str());
    }

    if (target_ok && dflash_ok) {
        bool exact = (target_ids == dflash_ids);
        std::size_t first = 0u;
        if (!exact) {
            while (first < target_ids.size() && first < dflash_ids.size() &&
                   target_ids[first] == dflash_ids[first]) {
                ++first;
            }
            std::printf("  FIRST_MISMATCH=token[%zu] target=%d dflash=%d\n", first,
                        first < target_ids.size() ? target_ids[first] : -1,
                        first < dflash_ids.size() ? dflash_ids[first] : -1);
        }
        std::printf("gate10 token exact=%s target_len=%zu dflash_len=%zu\n",
                    exact ? "yes" : "no", target_ids.size(), dflash_ids.size());
        check(exact, "dflash matches target-only greedy");
        check(dflash_ids.size() == static_cast<std::size_t>(64), "dflash max_new_tokens");
    }

    std::string bad_out;
    const int bad_temp_rc = run_capture(dflash_common + " --temperature 0.7", bad_out);
    check(bad_temp_rc == 2, "temperature>0 rejected with exit 2");

    std::string logits_out;
    const int bad_logits_rc = run_capture(
        dflash_common + " --temperature 0 --dump-logits /tmp/phaseshift_gate10_logits.bin",
        logits_out);
    check(bad_logits_rc == 2, "dump-logits rejected with exit 2");

    std::remove(prompt_path);
    std::printf("test_dflash2_gate10_cli: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
