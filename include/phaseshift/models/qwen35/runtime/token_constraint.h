#pragma once

#include <phaseshift/core/status.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ps {
namespace qwen35 {
namespace runtime {

inline bool constraint_token_allowed(
    const uint32_t* mask_words, uint32_t vocab_size, uint32_t token_id) {
    if (token_id >= vocab_size) {
        return false;
    }
    return ((mask_words[token_id >> 5u] >> (token_id & 31u)) & 1u) != 0u;
}

inline uint32_t constraint_allowed_count(
    const uint32_t* mask_words, uint32_t word_count, uint32_t vocab_size) {
    if (mask_words == nullptr || word_count == 0u || vocab_size == 0u) {
        return 0u;
    }
    uint32_t count = 0u;
    const uint32_t full_words = vocab_size / 32u;
    const uint32_t words = full_words < word_count ? full_words : word_count;
    for (uint32_t i = 0u; i < words; ++i) {
        count += static_cast<uint32_t>(__builtin_popcount(mask_words[i]));
    }
    const uint32_t remainder = vocab_size % 32u;
    if (remainder != 0u && words < word_count) {
        count += static_cast<uint32_t>(
            __builtin_popcount(mask_words[words] & ((1u << remainder) - 1u)));
    }
    return count;
}

class TokenConstraintState {
 public:
  virtual ~TokenConstraintState() = default;

  TokenConstraintState(const TokenConstraintState&) = delete;
  TokenConstraintState& operator=(const TokenConstraintState&) = delete;

  virtual Status fill_next_mask(uint32_t* mask_words, uint32_t word_count) = 0;
  virtual bool accept_token(int32_t token_id) = 0;
  virtual bool is_terminated() const = 0;

  virtual Status fill_draft_tree_masks(const int32_t* draft_tokens, uint32_t draft_count,
                                       uint32_t* mask_rows, uint32_t word_count) {
    (void)draft_tokens;
    (void)draft_count;
    (void)mask_rows;
    (void)word_count;
    return Status::unsupported("draft tree masks are not supported", __FILE__, __LINE__);
  }

 protected:
  TokenConstraintState() = default;
};

class TokenConstraintCompiler {
 public:
  static Result<std::unique_ptr<TokenConstraintCompiler>> create(
      const std::string& tokenizer_info_json,
      uint32_t expected_vocab_size,
      int32_t generation_stop_token,
      std::size_t cache_max_bytes = 64ull * 1024ull * 1024ull);

  TokenConstraintCompiler(const TokenConstraintCompiler&) = delete;
  TokenConstraintCompiler& operator=(const TokenConstraintCompiler&) = delete;
  ~TokenConstraintCompiler();

  Result<std::unique_ptr<TokenConstraintState>> create_gbnf_state(const std::string& gbnf);
  Result<std::unique_ptr<TokenConstraintState>> create_structural_tag_state(
      const std::string& structural_tag_json);

  uint32_t vocab_size() const noexcept;
  uint32_t mask_words() const noexcept;

  std::uint64_t cache_hits() const noexcept;
  std::uint64_t cache_misses() const noexcept;
  std::uint64_t last_compile_us() const noexcept;

 private:
  TokenConstraintCompiler();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
