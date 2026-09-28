#include <phaseshift/models/qwen35/runtime/ngram_tail.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL: %s\n", msg.c_str());
    }
}

using ps::qwen35::runtime::NgramTailConfig;
using ps::qwen35::runtime::NgramTailProposal;
using ps::qwen35::runtime::propose_ngram_tail;

NgramTailConfig make_config(uint32_t n, uint32_t max_tail, uint32_t window) {
    NgramTailConfig cfg;
    cfg.n = n;
    cfg.max_tail = max_tail;
    cfg.window = window;
    return cfg;
}

ps::Result<NgramTailProposal> run(
    const std::vector<int32_t>& history,
    const NgramTailConfig& cfg) {
    return propose_ngram_tail(std::span<const int32_t>(history), cfg);
}

void expect_no_leak(const std::vector<int32_t>& history, const NgramTailProposal& p) {
    check(p.candidate_start < p.current_position, "leak: candidate_start < current_position");
    check(p.candidate_end <= p.current_position, "leak: candidate_end <= current_position");
    check(p.candidate_end + p.tokens.size() <= p.current_position,
          "leak: continuation stays inside history");
    bool verbatim = true;
    for (std::size_t i = 0u; i < p.tokens.size(); ++i) {
        if (p.candidate_end + i >= history.size()) {
            verbatim = false;
            break;
        }
        if (history[p.candidate_end + i] != p.tokens[i]) verbatim = false;
    }
    check(verbatim, "leak: tokens copied verbatim from committed history");
    check(p.candidate_start + p.seed_length == p.candidate_end, "seed length");
}

void test_exact_hit() {
    const std::vector<int32_t> history{1, 2, 3, 4, 5, 9, 8, 7, 1, 2, 3, 4, 5};
    auto r = run(history, make_config(5u, 16u, 2048u));
    check(r.ok(), "test1: ok");
    if (!r.ok()) return;
    const NgramTailProposal& p = r.value();
    check(p.hit, "test1: hit");
    check(p.match_position == 0u, "test1: match_position");
    check(p.seed_length == 5u, "test1: seed_length");
    const std::vector<int32_t> expected{9, 8, 7, 1, 2, 3, 4, 5};
    check(p.tokens == expected, "test1: tokens");
    expect_no_leak(history, p);
}

void test_no_hit() {
    const std::vector<int32_t> history{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto r = run(history, make_config(4u, 16u, 2048u));
    check(r.ok(), "test2: ok");
    if (!r.ok()) return;
    check(!r.value().hit, "test2: no hit");
    check(r.value().tokens.empty(), "test2: empty tokens");
}

void test_recent_match_wins() {
    const std::vector<int32_t> history{
        1, 2, 3, 4, 5, 11, 12, 1, 2, 3, 4, 5, 21, 22, 1, 2, 3, 4, 5};
    auto r = run(history, make_config(5u, 16u, 2048u));
    check(r.ok(), "test3: ok");
    if (!r.ok()) return;
    const NgramTailProposal& p = r.value();
    check(p.hit, "test3: hit");
    check(p.match_position == 7u, "test3: most recent occurrence");
    const std::vector<int32_t> expected{21, 22, 1, 2, 3, 4, 5};
    check(p.tokens == expected, "test3: tokens from recent occurrence");
    expect_no_leak(history, p);
}

void test_window() {
    std::vector<int32_t> history{1, 2, 3, 4, 5};
    for (int32_t i = 0; i < 30; ++i) history.push_back(100 + i);
    for (int32_t t : {1, 2, 3, 4, 5}) history.push_back(t);
    check(history.size() == 40u, "test4: history size");

    auto outside = run(history, make_config(5u, 16u, 32u));
    check(outside.ok(), "test4: window32 ok");
    if (outside.ok()) {
        check(!outside.value().hit, "test4: occurrence outside window is ignored");
        check(outside.value().tokens.empty(), "test4: window32 empty tokens");
    }

    auto inside = run(history, make_config(5u, 16u, 2048u));
    check(inside.ok(), "test4: window2048 ok");
    if (inside.ok()) {
        check(inside.value().hit, "test4: occurrence inside window is used");
        check(inside.value().match_position == 0u, "test4: match_position");
        check(inside.value().tokens.size() == 16u, "test4: capped at max_tail");
        expect_no_leak(history, inside.value());
    }
}

void test_max_tail() {
    std::vector<int32_t> history{1, 2, 3, 4, 5};
    std::vector<int32_t> continuation;
    for (int32_t i = 0; i < 32; ++i) {
        continuation.push_back(1000 + i);
        history.push_back(1000 + i);
    }
    for (int32_t t : {1, 2, 3, 4, 5}) history.push_back(t);

    auto r = run(history, make_config(5u, 8u, 2048u));
    check(r.ok(), "test5: ok");
    if (!r.ok()) return;
    const NgramTailProposal& p = r.value();
    check(p.hit, "test5: hit");
    check(p.tokens.size() == 8u, "test5: capped at max_tail");
    check(std::vector<int32_t>(p.tokens.begin(), p.tokens.end()) ==
              std::vector<int32_t>(continuation.begin(), continuation.begin() + 8),
          "test5: tokens");
    expect_no_leak(history, p);
}

void test_no_future_leakage() {
    const std::vector<int32_t> periodic{1, 2, 1, 2, 1, 2};
    auto r = run(periodic, make_config(2u, 32u, 2048u));
    check(r.ok(), "test6: ok");
    if (!r.ok()) return;
    const NgramTailProposal& p = r.value();
    check(p.hit, "test6: hit");
    check(p.candidate_end == 4u, "test6: candidate_end");
    check(p.candidate_end + p.tokens.size() == p.current_position,
          "test6: extension stops at history end");
    const std::vector<int32_t> expected{1, 2};
    check(p.tokens == expected, "test6: tokens");
    expect_no_leak(periodic, p);

    const std::vector<int32_t> tail_bound{7, 7, 7, 3, 7, 7, 7};
    auto r2 = run(tail_bound, make_config(3u, 32u, 2048u));
    check(r2.ok(), "test6: tail bound ok");
    if (r2.ok()) expect_no_leak(tail_bound, r2.value());
}

void test_n_larger_than_history() {
    const std::vector<int32_t> history{1, 2, 3};
    auto r = run(history, make_config(5u, 16u, 2048u));
    check(r.ok(), "test7: ok");
    if (!r.ok()) return;
    check(!r.value().hit, "test7: no hit");
    check(r.value().tokens.empty(), "test7: empty tokens");
}

void test_n_zero() {
    const std::vector<int32_t> history{1, 2, 3, 4, 5};
    auto r = run(history, make_config(0u, 16u, 2048u));
    check(!r.ok(), "test8: rejected");
    if (!r.ok()) {
        check(r.status().code() == ps::Status::Code::invalid_argument,
              "test8: invalid_argument");
    }
}

}  // namespace

int main() {
    test_exact_hit();
    test_no_hit();
    test_recent_match_wins();
    test_window();
    test_max_tail();
    test_no_future_leakage();
    test_n_larger_than_history();
    test_n_zero();
    std::printf("test_ngram_tail: passed=%d failed=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
