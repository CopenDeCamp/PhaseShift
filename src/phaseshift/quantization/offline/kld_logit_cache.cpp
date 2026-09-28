#include <phaseshift/quantization/offline/kld_logit_cache.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unistd.h>
#include <vector>

namespace ps::quantization::fpx {

namespace {

struct CacheHeader {
    char magic[8];
    uint32_t version;
    char model_fingerprint[32];
    char corpus_sha256[32];
    uint64_t vocab_size;
    uint64_t window;
    uint64_t stride;
    uint64_t prediction_count;
    uint32_t runtime_config_len;
};

std::string sha_str_to_bin(const std::string& hex) {
    std::string bin;
    for (std::size_t i = 0; i + 1 < hex.size() && bin.size() < 32; i += 2) {
        auto nib = [](char c) -> unsigned char {
            if (c >= '0' && c <= '9') return static_cast<unsigned char>(c - '0');
            if (c >= 'a' && c <= 'f') return static_cast<unsigned char>(c - 'a' + 10);
            if (c >= 'A' && c <= 'F') return static_cast<unsigned char>(c - 'A' + 10);
            return 0;
        };
        bin.push_back(static_cast<char>((nib(hex[i]) << 4) | nib(hex[i + 1])));
    }
    return bin;
}

bool is_sha256_hex(const std::string& value) {
    if (value.size() != 64) return false;
    for (char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
            return false;
        }
    }
    return true;
}

bool cache_size_matches(FILE* f, const CacheHeader& header) {
    const uint64_t max = std::numeric_limits<uint64_t>::max();
    if (header.prediction_count != 0 && header.vocab_size > max / header.prediction_count) return false;
    const uint64_t elements = header.prediction_count * header.vocab_size;
    if (elements > max / sizeof(float)) return false;
    const uint64_t payload_bytes = elements * sizeof(float);
    const uint64_t header_bytes = sizeof(CacheHeader) + static_cast<uint64_t>(header.runtime_config_len);
    if (header_bytes < sizeof(CacheHeader) || payload_bytes > max - header_bytes) return false;
    if (std::fseek(f, 0, SEEK_END) != 0) return false;
    const long file_size = std::ftell(f);
    if (file_size < 0) return false;
    return static_cast<uint64_t>(file_size) == header_bytes + payload_bytes;
}

}

struct LogitCacheWrite::Impl {
    FILE* f = nullptr;
    std::size_t written = 0;
    std::string temporary_path;
};

struct LogitCacheReader::Impl {
    FILE* f = nullptr;
    std::size_t read = 0;
};

Status cleanup_writer(LogitCacheWrite::Impl*& impl) {
    if (!impl) return Status::make_ok();
    Status first = Status::make_ok();
    if (impl->f) {
        FILE* file = impl->f;
        impl->f = nullptr;
        if (std::fclose(file) != 0) {
            first = Status::invalid_state("cache close cleanup failed", __FILE__, __LINE__);
        }
    }
    Status remove_status = Status::make_ok();
    if (!impl->temporary_path.empty() && std::remove(impl->temporary_path.c_str()) != 0 && errno != ENOENT) {
        remove_status = Status::invalid_state("cache temporary cleanup failed", __FILE__, __LINE__);
    }
    delete impl;
    impl = nullptr;
    if (!first.ok() && !remove_status.ok()) {
        const std::string message = first.message() + "; " + remove_status.message();
        return Status::invalid_state(message.c_str(), __FILE__, __LINE__);
    }
    if (first.ok()) return remove_status;
    return first;
}

Status cleanup_writer_after(Status primary, LogitCacheWrite::Impl*& impl) {
    const Status cleanup = cleanup_writer(impl);
    if (primary.ok()) return cleanup;
    if (cleanup.ok()) return primary;
    const std::string message = primary.message() + "; cache cleanup failed: " + cleanup.message();
    return Status::invalid_state(message.c_str(), __FILE__, __LINE__);
}

LogitCacheWrite::~LogitCacheWrite() {
    (void)cleanup_writer(impl_);
}

LogitCacheWrite::LogitCacheWrite(LogitCacheWrite&& other) noexcept {
    impl_ = other.impl_;
    other.impl_ = nullptr;
}

LogitCacheWrite& LogitCacheWrite::operator=(LogitCacheWrite&& other) noexcept {
    if (this != &other) {
        (void)cleanup_writer(impl_);
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

LogitCacheReader::~LogitCacheReader() {
    if (impl_) {
        if (impl_->f) std::fclose(impl_->f);
        delete impl_;
        impl_ = nullptr;
    }
}

LogitCacheReader::LogitCacheReader(LogitCacheReader&& other) noexcept {
    impl_ = other.impl_;
    other.impl_ = nullptr;
}

LogitCacheReader& LogitCacheReader::operator=(LogitCacheReader&& other) noexcept {
    if (this != &other) {
        if (impl_) {
            if (impl_->f) std::fclose(impl_->f);
            delete impl_;
        }
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

Status LogitCacheWrite::write_begin() {
    if (impl_) return Status::invalid_state("write already begun", __FILE__, __LINE__);
    if (!is_sha256_hex(key.model_fingerprint) || !is_sha256_hex(key.corpus_sha256)) {
        return Status::invalid_argument("cache key SHA-256 must be 64 hexadecimal characters", __FILE__, __LINE__);
    }
    if (key.runtime_config_hash.size() > std::numeric_limits<uint32_t>::max()) {
        return Status::invalid_argument("cache runtime config is too large", __FILE__, __LINE__);
    }
    impl_ = new Impl();
    const std::string temporary_template = path + ".tmp.XXXXXX";
    std::vector<char> temporary_buffer(temporary_template.begin(), temporary_template.end());
    temporary_buffer.push_back('\0');
    const int descriptor = mkstemp(temporary_buffer.data());
    if (descriptor < 0) {
        delete impl_;
        impl_ = nullptr;
        return Status::invalid_argument(("cannot open cache for write: " + path).c_str(), __FILE__, __LINE__);
    }
    impl_->temporary_path = temporary_buffer.data();
    impl_->f = fdopen(descriptor, "wb");
    if (!impl_->f) {
        const int saved_errno = errno;
        ::close(descriptor);
        const Status primary = Status::invalid_argument(
            ("cannot open cache for write: " + std::string(std::strerror(saved_errno))).c_str(), __FILE__, __LINE__);
        return cleanup_writer_after(primary, impl_);
    }

    CacheHeader header;
    std::memset(&header, 0, sizeof(header));
    std::memcpy(header.magic, kLogitCacheMagic, 8);
    header.version = kLogitCacheVersion;
    const std::string fp_bin = sha_str_to_bin(key.model_fingerprint);
    std::memcpy(header.model_fingerprint, fp_bin.data(), std::min<std::size_t>(32, fp_bin.size()));
    const std::string sha_bin = sha_str_to_bin(key.corpus_sha256);
    std::memcpy(header.corpus_sha256, sha_bin.data(), std::min<std::size_t>(32, sha_bin.size()));
    header.vocab_size = key.vocab_size;
    header.window = key.window;
    header.stride = key.stride;
    header.prediction_count = key.prediction_count;
    const std::string runtime_config = key.runtime_config_hash;
    header.runtime_config_len = static_cast<uint32_t>(runtime_config.size());

    if (std::fwrite(&header, 1, sizeof(header), impl_->f) != sizeof(header)) {
        return cleanup_writer_after(Status::invalid_state("cache header write failed", __FILE__, __LINE__), impl_);
    }
    if (!runtime_config.empty() && std::fwrite(runtime_config.data(), 1, runtime_config.size(), impl_->f) != runtime_config.size()) {
        return cleanup_writer_after(Status::invalid_state("cache runtime-config write failed", __FILE__, __LINE__), impl_);
    }
    return Status::make_ok();
}

Status LogitCacheWrite::write_row(const float* logits, std::size_t vocab_size) {
    if (!impl_ || !impl_->f) return Status::invalid_state("cache not open", __FILE__, __LINE__);
    if (vocab_size != key.vocab_size) {
        return cleanup_writer_after(Status::invalid_argument("cache row width mismatch", __FILE__, __LINE__), impl_);
    }
    if (std::fwrite(logits, sizeof(float), vocab_size, impl_->f) != vocab_size) {
        return cleanup_writer_after(Status::invalid_state("cache row write failed", __FILE__, __LINE__), impl_);
    }
    ++impl_->written;
    return Status::make_ok();
}

Status LogitCacheWrite::write_end() {
    if (!impl_ || !impl_->f) return Status::invalid_state("cache not open", __FILE__, __LINE__);
    if (impl_->written != key.prediction_count) {
        return cleanup_writer_after(Status::invalid_state("cache row count mismatch on close", __FILE__, __LINE__), impl_);
    }
    if (std::fflush(impl_->f) != 0) {
        return cleanup_writer_after(Status::invalid_state("cache flush failed", __FILE__, __LINE__), impl_);
    }
    FILE* file = impl_->f;
    impl_->f = nullptr;
    if (std::fclose(file) != 0) {
        return cleanup_writer_after(Status::invalid_state("cache close failed", __FILE__, __LINE__), impl_);
    }
    if (std::rename(impl_->temporary_path.c_str(), path.c_str()) != 0) {
        return cleanup_writer_after(Status::invalid_state("cache publish failed", __FILE__, __LINE__), impl_);
    }
    delete impl_;
    impl_ = nullptr;
    return Status::make_ok();
}

Result<bool> logit_cache_matches(const std::string& path, const LogitCacheKey& key) {
    if (!is_sha256_hex(key.model_fingerprint) || !is_sha256_hex(key.corpus_sha256)) return false;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    CacheHeader header;
    const std::size_t got = std::fread(&header, 1, sizeof(header), f);
    if (got != sizeof(header)) {
        std::fclose(f);
        return false;
    }
    if (std::memcmp(header.magic, kLogitCacheMagic, 8) != 0 || header.version != kLogitCacheVersion) {
        std::fclose(f);
        return false;
    }
    if (header.vocab_size != key.vocab_size || header.window != key.window ||
        header.stride != key.stride || header.prediction_count != key.prediction_count) {
        std::fclose(f);
        return false;
    }
    if (!cache_size_matches(f, header)) {
        std::fclose(f);
        return false;
    }
    if (std::fseek(f, sizeof(header), SEEK_SET) != 0) {
        std::fclose(f);
        return false;
    }
    const std::string fp_bin = sha_str_to_bin(key.model_fingerprint);
    if (std::memcmp(header.model_fingerprint, fp_bin.data(), 32) != 0) {
        std::fclose(f);
        return false;
    }
    const std::string sha_bin = sha_str_to_bin(key.corpus_sha256);
    if (std::memcmp(header.corpus_sha256, sha_bin.data(), 32) != 0) {
        std::fclose(f);
        return false;
    }
    std::string runtime_config(header.runtime_config_len, '\0');
    if (header.runtime_config_len > 0 && std::fread(runtime_config.data(), 1, runtime_config.size(), f) != runtime_config.size()) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    return runtime_config == key.runtime_config_hash;
}

Status LogitCacheReader::open() {
    if (impl_) return Status::invalid_state("reader already open", __FILE__, __LINE__);
    auto matches = logit_cache_matches(path, key);
    if (!matches.ok()) return matches.status();
    if (!matches.value()) {
        return Status::invalid_state("cache reuse rejected: key mismatch", __FILE__, __LINE__);
    }
    impl_ = new Impl();
    impl_->f = std::fopen(path.c_str(), "rb");
    if (!impl_->f) {
        delete impl_;
        impl_ = nullptr;
        return Status::invalid_argument(("cannot open cache: " + path).c_str(), __FILE__, __LINE__);
    }
    CacheHeader header;
    const std::size_t got = std::fread(&header, 1, sizeof(header), impl_->f);
    if (got != sizeof(header)) {
        close();
        return Status::invalid_state("cache header read failed", __FILE__, __LINE__);
    }
    if (header.runtime_config_len > 0) {
        std::fseek(impl_->f, header.runtime_config_len, SEEK_CUR);
    }
    return Status::make_ok();
}

Status LogitCacheReader::read_row(float* logits, std::size_t vocab_size) {
    if (!impl_ || !impl_->f) return Status::invalid_state("reader not open", __FILE__, __LINE__);
    if (vocab_size != key.vocab_size) return Status::invalid_argument("cache row width mismatch", __FILE__, __LINE__);
    if (std::fread(logits, sizeof(float), vocab_size, impl_->f) != vocab_size) {
        return Status::invalid_state("cache row read failed", __FILE__, __LINE__);
    }
    ++impl_->read;
    return Status::make_ok();
}

void LogitCacheReader::close() {
    if (impl_) {
        if (impl_->f) std::fclose(impl_->f);
        delete impl_;
        impl_ = nullptr;
    }
}

}
