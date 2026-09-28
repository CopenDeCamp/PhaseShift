#pragma once
#include <phaseshift/core/status.h>
#include <cstdint>
#include <string>
#include <vector>

namespace ps::quantization::fpx {

struct TokenCorpus {
    std::vector<int32_t> tokens;
    std::string sha256_hex;
};

std::string sha256_hex_str(const std::string& data);

Result<std::string> sha256_hex_file(const std::string& path);

Result<TokenCorpus> read_token_corpus(const std::string& path);

}
