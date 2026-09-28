#pragma once

#include <phaseshift/core/status.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace ps::quantization::fpx {

struct LogitCacheKey {
    std::string model_fingerprint;
    std::string corpus_sha256;
    std::size_t vocab_size = 0;
    std::size_t window = 0;
    std::size_t stride = 0;
    std::size_t prediction_count = 0;
    std::string runtime_config_hash;
};

struct LogitCacheWrite {
    std::string path;
    LogitCacheKey key;
    Status write_begin();
    Status write_row(const float* logits, std::size_t vocab_size);
    Status write_end();

    ~LogitCacheWrite();
    LogitCacheWrite() = default;
    LogitCacheWrite(LogitCacheWrite&&) noexcept;
    LogitCacheWrite& operator=(LogitCacheWrite&&) noexcept;
    LogitCacheWrite(const LogitCacheWrite&) = delete;
    LogitCacheWrite& operator=(const LogitCacheWrite&) = delete;

    struct Impl;
    Impl* impl_ = nullptr;
};

struct LogitCacheReader {
    std::string path;
    LogitCacheKey key;
    Status open();
    Status read_row(float* logits, std::size_t vocab_size);
    void close();
    bool is_open() const { return impl_ != nullptr; }

    ~LogitCacheReader();
    LogitCacheReader() = default;
    LogitCacheReader(LogitCacheReader&&) noexcept;
    LogitCacheReader& operator=(LogitCacheReader&&) noexcept;
    LogitCacheReader(const LogitCacheReader&) = delete;
    LogitCacheReader& operator=(const LogitCacheReader&) = delete;

    struct Impl;
    Impl* impl_ = nullptr;
};

inline const char* kLogitCacheMagic = "PSKLDLOG";
constexpr uint32_t kLogitCacheVersion = 2;

Result<bool> logit_cache_matches(const std::string& path, const LogitCacheKey& key);

}
