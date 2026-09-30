#include <phaseshift/models/qwen35/runtime/token_constraint.h>

#include <dlpack/dlpack.h>
#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
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

xgrammar::TokenizerInfo make_tokenizer_info() {
    std::vector<std::string> encoded(kVocabSize);
    for (uint32_t i = 0; i < kVocabSize; ++i) {
        encoded[i] = "T" + std::to_string(i);
    }
    encoded[kYes] = "YES";
    encoded[kNo] = "NO";
    encoded[kMaybe] = "MAYBE";
    encoded[kStop] = "<stop>";
    return xgrammar::TokenizerInfo(
        encoded, xgrammar::VocabType::RAW, static_cast<int>(kVocabSize),
        std::vector<int32_t>{kStop});
}

}  // namespace

int main() {
    using namespace ps::qwen35::runtime;

    xgrammar::TokenizerInfo info = make_tokenizer_info();
    const std::string serialized = info.SerializeJSON();

    auto compiler_result = TokenConstraintCompiler::create(
        serialized, kVocabSize, kStop);
    CHECK(compiler_result.ok());
    if (!compiler_result.ok()) {
        return 1;
    }
    auto& compiler = *compiler_result.value();
    CHECK(compiler.vocab_size() == kVocabSize);
    CHECK(compiler.mask_words() == (kVocabSize + 31u) / 32u);

    auto state_result = compiler.create_gbnf_state("root ::= \"YES\" | \"NO\"");
    CHECK(state_result.ok());
    if (!state_result.ok()) {
        return 1;
    }
    auto& state = *state_result.value();

    std::vector<uint32_t> mask(compiler.mask_words(), 0u);
    auto fill_status = state.fill_next_mask(mask.data(), compiler.mask_words());
    CHECK(fill_status.ok());

    // PhaseShift helper honours the XGrammar bit convention.
    CHECK(constraint_token_allowed(mask.data(), kVocabSize, kYes));
    CHECK(constraint_token_allowed(mask.data(), kVocabSize, kNo));
    CHECK(!constraint_token_allowed(mask.data(), kVocabSize, kMaybe));
    // Vocabulary tail and out-of-range ids are never allowed.
    for (uint32_t id = kVocabSize; id < kVocabSize + 8u; ++id) {
        CHECK(!constraint_token_allowed(mask.data(), kVocabSize, id));
    }

    // Cross-check the full mask against the XGrammar reference bitmask.
    DLTensor tensor;
    tensor.data = mask.data();
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim = 1;
    int64_t shape[1] = {static_cast<int64_t>(compiler.mask_words())};
    tensor.shape = shape;
    tensor.dtype = xgrammar::GetBitmaskDLType();
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
    std::vector<int> rejected;
    xgrammar::_DebugGetMaskedTokensFromBitmask(
        &rejected, tensor, static_cast<int>(kVocabSize), 0);
    auto is_rejected = [&](int id) {
        for (int r : rejected) {
            if (r == id) return true;
        }
        return false;
    };
    for (uint32_t id = 0; id < kVocabSize; ++id) {
        if (id % 32u == 0u || id % 32u == 31u) {
            CHECK(constraint_token_allowed(mask.data(), kVocabSize, id) == !is_rejected(id));
        }
    }
    for (uint32_t boundary : {0u, 31u, 32u, 33u, 63u, 64u}) {
        CHECK(constraint_token_allowed(mask.data(), kVocabSize, boundary) ==
              !is_rejected(static_cast<int>(boundary)));
    }
    CHECK(is_rejected(kMaybe));
    CHECK(!is_rejected(kYes));
    CHECK(!is_rejected(kNo));

    // Accepting a sampled allowed token advances the matcher.
    CHECK(state.accept_token(kYes));
    CHECK(state.accept_token(kStop));
    CHECK(state.is_terminated());

    uint32_t manual_allowed = 0u;
    for (uint32_t id = 0; id < kVocabSize; ++id) {
        if (constraint_token_allowed(mask.data(), kVocabSize, id)) {
            ++manual_allowed;
        }
    }
    CHECK(constraint_allowed_count(mask.data(), compiler.mask_words(), kVocabSize) ==
          manual_allowed);

    std::vector<uint32_t> synthetic(3, 0xFFFFFFFFu);
    CHECK(constraint_allowed_count(synthetic.data(), 3u, 70u) == 70u);
    CHECK(constraint_allowed_count(synthetic.data(), 3u, 64u) == 64u);
    CHECK(constraint_allowed_count(synthetic.data(), 3u, 96u) == 96u);
    CHECK(constraint_allowed_count(synthetic.data(), 3u, 33u) == 33u);
    CHECK(constraint_allowed_count(synthetic.data(), 3u, 1u) == 1u);
    CHECK(constraint_allowed_count(synthetic.data(), 3u, 0u) == 0u);
    CHECK(constraint_allowed_count(nullptr, 3u, 96u) == 0u);
    CHECK(constraint_allowed_count(synthetic.data(), 0u, 96u) == 0u);

    std::vector<uint32_t> sparse(3, 0u);
    sparse[0] = 0x00000001u;
    sparse[1] = 0x80000000u;
    sparse[2] = 0x00000003u;
    CHECK(constraint_allowed_count(sparse.data(), 3u, 64u) == 2u);
    CHECK(constraint_allowed_count(sparse.data(), 3u, 65u) == 3u);
    CHECK(constraint_allowed_count(sparse.data(), 3u, 96u) == 4u);

    // Invalid grammar is a fail-closed error, never a silent unconstrained path.
    auto invalid = compiler.create_gbnf_state("root ::= [");
    CHECK(!invalid.ok());

    // Cross-language TokenizerInfo compatibility (Python sidecar -> native engine).
    if (const char* sidecar = std::getenv("PHASESHIFT_CONSTRAINT_SIDECAR")) {
        int32_t stop_token = 248044;
        if (const char* raw = std::getenv("PHASESHIFT_CONSTRAINT_STOP_TOKEN")) {
            stop_token = std::atoi(raw);
        }
        std::ifstream stream(sidecar, std::ios::binary);
        CHECK(stream.good());
        std::string json((std::istreambuf_iterator<char>(stream)),
                         std::istreambuf_iterator<char>());
        auto native = TokenConstraintCompiler::create(json, 248320u, stop_token);
        CHECK(native.ok());
        if (native.ok()) {
            auto& native_compiler = *native.value();
            CHECK(native_compiler.vocab_size() == 248320u);
            CHECK(native_compiler.mask_words() == 7760u);
            auto native_state = native_compiler.create_gbnf_state("root ::= \"YES\" | \"NO\"");
            CHECK(native_state.ok());
            if (native_state.ok()) {
                std::vector<uint32_t> native_mask(native_compiler.mask_words(), 0u);
                CHECK(native_state.value()
                          ->fill_next_mask(native_mask.data(),
                                           native_compiler.mask_words())
                          .ok());
                uint32_t allowed = 0u;
                int32_t chosen = -1;
                for (uint32_t id = 0; id < native_compiler.vocab_size(); ++id) {
                    if (constraint_token_allowed(native_mask.data(),
                                                 native_compiler.vocab_size(), id)) {
                        ++allowed;
                        if (chosen < 0) chosen = static_cast<int32_t>(id);
                    }
                }
                CHECK(allowed > 0u);
                CHECK(chosen >= 0);
                if (chosen >= 0) CHECK(native_state.value()->accept_token(chosen));
            }
            CHECK(!native_compiler.create_gbnf_state("root ::= [").ok());
        }
    }

    if (failures == 0) {
        std::printf("constraint-mask: passed\n");
    }
    return failures == 0 ? 0 : 1;
}
