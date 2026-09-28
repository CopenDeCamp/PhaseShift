#include <phaseshift/quantization/offline/token_corpus.h>
#include <openssl/sha.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>

namespace ps::quantization::fpx {

namespace {

uint32_t le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t le64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

}

std::string sha256_hex_str(const std::string& data) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(data.data()), data.size(), hash);
    std::ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        char buf[3];
        std::snprintf(buf, sizeof(buf), "%02x", hash[i]);
        oss << buf;
    }
    return oss.str();
}

Result<std::string> sha256_hex_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return Status::invalid_argument(("cannot open file for sha256: " + path).c_str(), __FILE__,
                                        __LINE__);
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (!f && !f.eof())
        return Status::invalid_argument(("cannot read file for sha256: " + path).c_str(), __FILE__,
                                        __LINE__);
    return sha256_hex_str(data);
}

Result<TokenCorpus> read_token_corpus(const std::string& path) {
    TokenCorpus corpus;
    auto sha_res = sha256_hex_file(path);
    if (!sha_res.ok()) return sha_res.status();
    corpus.sha256_hex = sha_res.release();
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return Status::invalid_argument(("cannot open corpus " + path).c_str(), __FILE__, __LINE__);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (data.size() < 8 + 4 + 8)
        return Status::invalid_argument("corpus too small", __FILE__, __LINE__);
    if (std::memcmp(data.data(), "PSKLDTOK", 8) != 0)
        return Status::invalid_argument("bad corpus magic", __FILE__, __LINE__);
    if (le32(data.data() + 8) != 1)
        return Status::invalid_argument("bad corpus version", __FILE__, __LINE__);
    const uint64_t count = le64(data.data() + 12);
    const std::size_t expect = 8 + 4 + 8 + static_cast<std::size_t>(count) * 4;
    if (data.size() < expect)
        return Status::invalid_argument("corpus truncated", __FILE__, __LINE__);
    corpus.tokens.reserve(static_cast<std::size_t>(count));
    for (uint64_t i = 0; i < count; ++i)
        corpus.tokens.push_back(static_cast<int32_t>(
            le32(data.data() + 20 + static_cast<std::size_t>(i) * 4)));
    return corpus;
}

}
