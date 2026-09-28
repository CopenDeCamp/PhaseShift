#include <phaseshift/quantization/offline/model_fingerprint.h>
#include <openssl/evp.h>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace ps::quantization::fpx {

namespace fs = std::filesystem;

std::string compute_model_fingerprint(const std::string& model_dir) {
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len = 0;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
    auto update = [&](const void* data, std::size_t size) {
        EVP_DigestUpdate(ctx, data, size);
    };
    auto update_file = [&](const std::string& p) {
        std::ifstream f(p, std::ios::binary);
        std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        update(data.data(), data.size());
    };
    const std::string canon = fs::canonical(model_dir).string();
    update(canon.data(), canon.size());
    update("\0", 1);

    const fs::path cfg = fs::path(model_dir) / "config.json";
    if (fs::exists(cfg)) update_file(cfg.string());
    update("\0", 1);

    const fs::path idx = fs::path(model_dir) / "model.safetensors.index.json";
    if (fs::exists(idx)) update_file(idx.string());
    update("\0", 1);

    std::vector<std::string> shards;
    for (const auto& e : fs::directory_iterator(model_dir)) {
        if (!e.is_regular_file()) continue;
        const std::string name = e.path().filename().string();
        if (name.find(".safetensors") != std::string::npos) {
            shards.push_back(e.path().string());
        }
    }
    std::sort(shards.begin(), shards.end());
    for (const auto& shard : shards) {
        update(shard.data(), shard.size());
        update("\0", 1);
        std::error_code ec;
        const auto ftime = fs::last_write_time(shard, ec);
        const auto size = fs::file_size(shard, ec);
        const long long ftime_count = static_cast<long long>(ftime.time_since_epoch().count());
        const std::string meta = std::to_string(size) + ":" + (ec ? "0" : std::to_string(ftime_count));
        update(meta.data(), meta.size());
        update("\0", 1);
    }
    EVP_DigestFinal_ex(ctx, hash, &hash_len);
    EVP_MD_CTX_free(ctx);
    std::ostringstream oss;
    for (unsigned int i = 0; i < hash_len; ++i) {
        char buf[3];
        std::snprintf(buf, sizeof(buf), "%02x", hash[i]);
        oss << buf;
    }
    return oss.str();
}

}