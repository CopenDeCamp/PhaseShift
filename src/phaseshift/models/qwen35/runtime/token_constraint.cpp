#include <phaseshift/models/qwen35/runtime/token_constraint.h>

#include <xgrammar/xgrammar.h>

#include <dlpack/dlpack.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace ps {
namespace qwen35 {
namespace runtime {

namespace {

constexpr std::size_t kMaxCacheEntries = 256;

struct CompiledConstraint {
    explicit CompiledConstraint(xgrammar::CompiledGrammar grammar)
        : compiled(std::move(grammar)),
          bytes(compiled.MemorySizeBytes()) {}

    xgrammar::CompiledGrammar compiled;
    std::size_t bytes = 0;
};

DLTensor make_bitmask_tensor(uint32_t* mask_words, uint32_t word_count) {
    DLTensor tensor;
    tensor.data = mask_words;
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim = 1;
    tensor.dtype = xgrammar::GetBitmaskDLType();
    tensor.shape = nullptr;
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
    (void)word_count;
    return tensor;
}

DLTensor make_bitmask_grid_tensor(uint32_t* mask_rows, int64_t rows, int64_t word_count,
                                  int64_t* shape) {
    shape[0] = rows;
    shape[1] = word_count;
    DLTensor tensor;
    tensor.data = mask_rows;
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim = 2;
    tensor.dtype = xgrammar::GetBitmaskDLType();
    tensor.shape = shape;
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
    return tensor;
}

DLTensor make_index_tensor(int64_t* data, int64_t count) {
    DLTensor tensor;
    tensor.data = data;
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim = 1;
    tensor.dtype = DLDataType{kDLInt, 64, 1};
    tensor.shape = nullptr;
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
    (void)count;
    return tensor;
}

std::vector<int> stop_token_ids(const StopTokens& stop_tokens) {
    std::vector<int> ids;
    ids.reserve(stop_tokens.size());
    for (const int32_t token : stop_tokens) {
        ids.push_back(static_cast<int>(token));
    }
    return ids;
}

class XGrammarConstraintState final : public TokenConstraintState {
 public:
    XGrammarConstraintState(std::shared_ptr<const CompiledConstraint> compiled,
                            uint32_t mask_words,
                            const StopTokens& generation_stop_tokens)
        : compiled_(std::move(compiled)),
          mask_words_(mask_words),
          matcher_(compiled_->compiled, stop_token_ids(generation_stop_tokens), false) {}

    Status fill_next_mask(uint32_t* mask_words, uint32_t word_count) override {
        if (mask_words == nullptr) {
            return Status::invalid_argument("mask_words is null", __FILE__, __LINE__);
        }
        if (word_count != mask_words_) {
            return Status::invalid_argument("mask word count mismatch", __FILE__, __LINE__);
        }
        DLTensor tensor = make_bitmask_tensor(mask_words, word_count);
        int64_t shape[1] = {static_cast<int64_t>(mask_words_)};
        tensor.shape = shape;
        matcher_.FillNextTokenBitmask(&tensor);
        return Status::make_ok();
    }

    bool accept_token(int32_t token_id) override {
        return matcher_.AcceptToken(token_id);
    }

    bool is_terminated() const override { return matcher_.IsTerminated(); }

    Status fill_draft_tree_masks(const int32_t* draft_tokens, uint32_t draft_count,
                                 uint32_t* mask_rows, uint32_t word_count) override {
        if (draft_tokens == nullptr || mask_rows == nullptr) {
            return Status::invalid_argument("draft tree mask argument is null", __FILE__,
                                            __LINE__);
        }
        if (word_count != mask_words_) {
            return Status::invalid_argument("mask word count mismatch", __FILE__, __LINE__);
        }
        if (draft_count == 0u) {
            return Status::invalid_argument("draft tree requires at least one row", __FILE__,
                                            __LINE__);
        }
        const int64_t rows = static_cast<int64_t>(draft_count) + 1;
        std::vector<int64_t> next(static_cast<std::size_t>(rows), -1);
        std::vector<int64_t> sibling(static_cast<std::size_t>(rows), -1);
        std::vector<int64_t> tokens(static_cast<std::size_t>(rows), 0);
        for (int64_t i = 0; i + 1 < rows; ++i) {
            next[static_cast<std::size_t>(i)] = i + 1;
        }
        for (int64_t i = 1; i < rows; ++i) {
            tokens[static_cast<std::size_t>(i)] = draft_tokens[i - 1];
        }
        int64_t shape[2] = {0, 0};
        DLTensor next_tensor = make_index_tensor(next.data(), rows);
        DLTensor sibling_tensor = make_index_tensor(sibling.data(), rows);
        DLTensor tokens_tensor = make_index_tensor(tokens.data(), rows);
        DLTensor mask_tensor =
            make_bitmask_grid_tensor(mask_rows, rows, word_count, shape);
        int64_t next_shape[1] = {rows};
        int64_t sibling_shape[1] = {rows};
        int64_t tokens_shape[1] = {rows};
        next_tensor.shape = next_shape;
        sibling_tensor.shape = sibling_shape;
        tokens_tensor.shape = tokens_shape;
        if (!matcher_.TraverseDraftTree(&next_tensor, &sibling_tensor, &tokens_tensor,
                                        &mask_tensor)) {
            return Status::invalid_state("draft tree traversal failed", __FILE__, __LINE__);
        }
        return Status::make_ok();
    }

 private:
    std::shared_ptr<const CompiledConstraint> compiled_;
    uint32_t mask_words_ = 0;
    xgrammar::GrammarMatcher matcher_;
};

}  // namespace

struct TokenConstraintCompiler::Impl {
    xgrammar::TokenizerInfo tokenizer_info;
    xgrammar::GrammarCompiler compiler;
    uint32_t vocab_size = 0;
    uint32_t mask_words = 0;
    std::size_t cache_max_bytes = 0;
    std::size_t cache_bytes = 0;
    StopTokens generation_stop_tokens;
    std::uint64_t cache_hits = 0;
    std::uint64_t cache_misses = 0;
    std::uint64_t last_compile_us = 0;
    std::unordered_map<std::string, std::shared_ptr<const CompiledConstraint>> cache;
    std::vector<std::string> fifo;
    std::mutex mutex;

    Impl(xgrammar::TokenizerInfo info, std::size_t max_bytes, const StopTokens& stops)
        : tokenizer_info(std::move(info)),
          compiler(tokenizer_info, 1, false, -1),
          vocab_size(static_cast<uint32_t>(tokenizer_info.GetVocabSize())),
          mask_words(static_cast<uint32_t>(xgrammar::GetBitmaskSize(
              static_cast<int>(tokenizer_info.GetVocabSize())))),
          cache_max_bytes(max_bytes),
          generation_stop_tokens(stops) {}

    void evict_until_bounded() {
        while ((cache_bytes > cache_max_bytes || cache.size() > kMaxCacheEntries) &&
               !fifo.empty()) {
            const std::string key = fifo.front();
            fifo.erase(fifo.begin());
            auto it = cache.find(key);
            if (it != cache.end()) {
                cache_bytes -= it->second->bytes;
                cache.erase(it);
            }
        }
    }

    Result<std::shared_ptr<const CompiledConstraint>> get_or_compile(
        const std::string& cache_key,
        const std::function<xgrammar::CompiledGrammar()>& build) {
        std::shared_ptr<const CompiledConstraint> compiled;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = cache.find(cache_key);
            if (it != cache.end()) {
                compiled = it->second;
                ++cache_hits;
                last_compile_us = 0;
            } else {
                ++cache_misses;
            }
        }
        if (compiled == nullptr) {
            std::shared_ptr<CompiledConstraint> fresh;
            try {
                const auto start = std::chrono::steady_clock::now();
                fresh = std::make_shared<CompiledConstraint>(build());
                const auto end = std::chrono::steady_clock::now();
                std::lock_guard<std::mutex> lock(mutex);
                last_compile_us = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                        .count());
            } catch (const std::exception& exc) {
                return Status::invalid_argument(exc.what(), __FILE__, __LINE__);
            }
            compiled = fresh;
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto inserted = cache.emplace(cache_key, fresh);
                if (inserted.second) {
                    fifo.push_back(cache_key);
                    cache_bytes += fresh->bytes;
                }
                evict_until_bounded();
            }
        }
        return compiled;
    }
};

TokenConstraintCompiler::TokenConstraintCompiler() : impl_(nullptr) {}

TokenConstraintCompiler::~TokenConstraintCompiler() = default;

Result<std::unique_ptr<TokenConstraintCompiler>> TokenConstraintCompiler::create(
    const std::string& tokenizer_info_json,
    uint32_t expected_vocab_size,
    const StopTokens& generation_stop_tokens,
    std::size_t cache_max_bytes) {
    if (generation_stop_tokens.empty()) {
        return Status::invalid_argument(
            "at least one generation stop token is required", __FILE__, __LINE__);
    }
    auto parsed = xgrammar::TokenizerInfo::DeserializeJSON(tokenizer_info_json);
    if (std::holds_alternative<xgrammar::SerializationError>(parsed)) {
        return Status::invalid_argument(
            "failed to deserialize xgrammar tokenizer info", __FILE__, __LINE__);
    }
    xgrammar::TokenizerInfo info = std::move(std::get<xgrammar::TokenizerInfo>(parsed));
    if (static_cast<uint32_t>(info.GetVocabSize()) != expected_vocab_size) {
        return Status::invalid_argument(
            "tokenizer info vocab size does not match the model", __FILE__, __LINE__);
    }
    for (const int32_t token : generation_stop_tokens) {
        if (token < 0 || static_cast<uint32_t>(token) >= expected_vocab_size) {
            return Status::invalid_argument(
                "generation stop token is outside the tokenizer vocabulary", __FILE__,
                __LINE__);
        }
    }
    std::unique_ptr<TokenConstraintCompiler> compiler(new TokenConstraintCompiler());
    compiler->impl_ = std::make_unique<Impl>(std::move(info), cache_max_bytes,
                                             generation_stop_tokens);
    return compiler;
}

Result<std::unique_ptr<TokenConstraintState>> TokenConstraintCompiler::create_gbnf_state(
    const std::string& gbnf) {
    if (impl_ == nullptr) {
        return Status::invalid_state("constraint compiler not initialized", __FILE__,
                                     __LINE__);
    }
    auto compiled_result = impl_->get_or_compile(
        std::string("G\0", 2) + gbnf,
        [this, &gbnf]() { return impl_->compiler.CompileGrammar(gbnf, "root"); });
    if (!compiled_result.ok()) {
        return compiled_result.status();
    }
    auto state = std::make_unique<XGrammarConstraintState>(
        compiled_result.release(), impl_->mask_words, impl_->generation_stop_tokens);
    return std::unique_ptr<TokenConstraintState>(std::move(state));
}

Result<std::unique_ptr<TokenConstraintState>>
TokenConstraintCompiler::create_structural_tag_state(
    const std::string& structural_tag_json) {
    if (impl_ == nullptr) {
        return Status::invalid_state("constraint compiler not initialized", __FILE__,
                                     __LINE__);
    }
    auto compiled_result = impl_->get_or_compile(
        std::string("S\0", 2) + structural_tag_json,
        [this, &structural_tag_json]() {
            return impl_->compiler.CompileStructuralTag(structural_tag_json);
        });
    if (!compiled_result.ok()) {
        return compiled_result.status();
    }
    auto state = std::make_unique<XGrammarConstraintState>(
        compiled_result.release(), impl_->mask_words, impl_->generation_stop_tokens);
    return std::unique_ptr<TokenConstraintState>(std::move(state));
}

uint32_t TokenConstraintCompiler::vocab_size() const noexcept {
    return impl_ != nullptr ? impl_->vocab_size : 0u;
}

uint32_t TokenConstraintCompiler::mask_words() const noexcept {
    return impl_ != nullptr ? impl_->mask_words : 0u;
}

std::uint64_t TokenConstraintCompiler::cache_hits() const noexcept {
    return impl_ != nullptr ? impl_->cache_hits : 0u;
}

std::uint64_t TokenConstraintCompiler::cache_misses() const noexcept {
    return impl_ != nullptr ? impl_->cache_misses : 0u;
}

std::uint64_t TokenConstraintCompiler::last_compile_us() const noexcept {
    return impl_ != nullptr ? impl_->last_compile_us : 0u;
}

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
