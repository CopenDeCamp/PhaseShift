#include <phaseshift/models/qwen35/runtime/token_constraint.h>

#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what, int line) {
    if (!condition) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
        ++failures;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

constexpr uint32_t kVocabSize = 70u;
constexpr int32_t kYes = 10;
constexpr int32_t kNo = 11;
constexpr int32_t kMaybe = 12;
constexpr int32_t kStop = 13;

std::vector<std::string> make_encoded_vocab() {
    std::vector<std::string> encoded(kVocabSize);
    for (uint32_t i = 0; i < kVocabSize; ++i) {
        encoded[i] = "T" + std::to_string(i);
    }
    encoded[kYes] = "YES";
    encoded[kNo] = "NO";
    encoded[kMaybe] = "MAYBE";
    encoded[kStop] = "<stop>";
    return encoded;
}

bool row_empty(const uint32_t* row, uint32_t words) {
    for (uint32_t w = 0; w < words; ++w) {
        if (row[w] != 0u) return false;
    }
    return true;
}

}  // namespace

int main() {
    using namespace ps::qwen35::runtime;

    xgrammar::TokenizerInfo info(make_encoded_vocab(), xgrammar::VocabType::RAW,
                                 static_cast<int>(kVocabSize),
                                 std::vector<int32_t>{kStop});
    auto compiler_result = TokenConstraintCompiler::create(info.SerializeJSON(), kVocabSize,
                                                           ps::qwen35::StopTokens{kStop});
    CHECK(compiler_result.ok());
    if (!compiler_result.ok()) return 1;
    auto& compiler = *compiler_result.value();
    const uint32_t words = compiler.mask_words();

    const std::string grammar = "root ::= \"YES\" \"NO\" \"YES\"";

    auto tree_state_result = compiler.create_gbnf_state(grammar);
    CHECK(tree_state_result.ok());
    if (!tree_state_result.ok()) return 1;
    auto& tree_state = *tree_state_result.value();

    auto walk_state_result = compiler.create_gbnf_state(grammar);
    CHECK(walk_state_result.ok());
    if (!walk_state_result.ok()) return 1;
    auto& walk_state = *walk_state_result.value();

    std::vector<uint32_t> root_mask(words, 0u);
    CHECK(walk_state.fill_next_mask(root_mask.data(), words).ok());

    const std::vector<int32_t> drafts = {kYes, kNo, kYes};
    const uint32_t draft_count = static_cast<uint32_t>(drafts.size());
    std::vector<uint32_t> rows(static_cast<std::size_t>(draft_count + 1u) * words, 0u);

    CHECK(tree_state.fill_draft_tree_masks(drafts.data(), draft_count, rows.data(), words)
              .ok());

    CHECK(!row_empty(root_mask.data(), words));
    for (uint32_t w = 0; w < words; ++w) {
        CHECK(rows[w] == root_mask[w]);
    }

    for (uint32_t i = 1u; i <= draft_count; ++i) {
        CHECK(walk_state.accept_token(drafts[i - 1u]));
        std::vector<uint32_t> expected(words, 0u);
        CHECK(walk_state.fill_next_mask(expected.data(), words).ok());
        const uint32_t* row = rows.data() + static_cast<std::size_t>(i) * words;
        CHECK(!row_empty(row, words));
        for (uint32_t w = 0; w < words; ++w) {
            CHECK(row[w] == expected[w]);
        }
    }

    std::vector<uint32_t> root_mask_after(words, 0u);
    CHECK(tree_state.fill_next_mask(root_mask_after.data(), words).ok());
    for (uint32_t w = 0; w < words; ++w) {
        CHECK(root_mask_after[w] == root_mask[w]);
    }

    const std::vector<int32_t> bad_drafts = {kYes, kYes, kYes};
    std::vector<uint32_t> bad_rows(static_cast<std::size_t>(draft_count + 1u) * words, 0u);
    CHECK(tree_state
              .fill_draft_tree_masks(bad_drafts.data(), draft_count, bad_rows.data(), words)
              .ok());
    CHECK(!row_empty(bad_rows.data(), words));
    CHECK(row_empty(bad_rows.data() + 2u * words, words));

    std::vector<uint32_t> root_mask_final(words, 0u);
    CHECK(tree_state.fill_next_mask(root_mask_final.data(), words).ok());
    for (uint32_t w = 0; w < words; ++w) {
        CHECK(root_mask_final[w] == root_mask[w]);
    }

    CHECK(tree_state.fill_draft_tree_masks(nullptr, draft_count, rows.data(), words).ok() ==
          false);
    CHECK(tree_state.fill_draft_tree_masks(drafts.data(), 0u, rows.data(), words).ok() ==
          false);
    CHECK(tree_state.fill_draft_tree_masks(drafts.data(), draft_count, rows.data(),
                                           words + 1u)
              .ok() == false);

    auto walk_state_result2 = compiler.create_gbnf_state(grammar);
    CHECK(walk_state_result2.ok());
    if (walk_state_result2.ok()) {
        auto& second = *walk_state_result2.value();
        std::vector<uint32_t> again(words, 0u);
        CHECK(second.fill_next_mask(again.data(), words).ok());
        for (uint32_t w = 0; w < words; ++w) {
            CHECK(again[w] == root_mask[w]);
        }
        CHECK(second.accept_token(kYes));
        CHECK(!second.accept_token(kYes));
        CHECK(second.accept_token(kNo));
        CHECK(second.accept_token(kYes));
        std::vector<uint32_t> terminal(words, 0u);
        CHECK(second.fill_next_mask(terminal.data(), words).ok());
        CHECK(constraint_token_allowed(terminal.data(), kVocabSize, kStop));
    }

    if (failures == 0) {
        std::printf("dflash2-constraint-mask: passed\n");
    }
    return failures == 0 ? 0 : 1;
}
