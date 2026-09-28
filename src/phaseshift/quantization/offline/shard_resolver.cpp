#include <phaseshift/quantization/offline/shard_resolver.h>
#include <phaseshift/io/safetensors_reader.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>

namespace ps::quantization::fpx {

namespace fs = std::filesystem;

Result<ShardResolver> ShardResolver::open(const std::string& model_dir) {
    ShardResolver r;
    r.model_dir_ = model_dir;
    const fs::path index_path = fs::path(model_dir) / "model.safetensors.index.json";
    if (fs::exists(index_path)) {
        std::ifstream f(index_path);
        nlohmann::json idx;
        try {
            f >> idx;
        } catch (const nlohmann::json::exception& e) {
            return Status::invalid_argument("bad safetensors index json", __FILE__, __LINE__);
        }
        if (!idx.contains("weight_map") || !idx["weight_map"].is_object()) {
            return Status::invalid_argument("safetensors index missing weight_map", __FILE__, __LINE__);
        }
        for (auto it = idx["weight_map"].begin(); it != idx["weight_map"].end(); ++it) {
            const std::string tensor = it.key();
            const std::string shard = it.value().get<std::string>();
            r.weight_map_[tensor] = shard;
        }
        for (const auto& kv : r.weight_map_) {
            r.tensor_names_.push_back(kv.first);
        }
        for (const auto& kv : r.weight_map_) {
            const fs::path shard = fs::path(model_dir) / kv.second;
            bool seen = false;
            for (const auto& s : r.shard_files_) seen = seen || (s == shard.string());
            if (!seen) r.shard_files_.push_back(shard.string());
        }
        std::sort(r.shard_files_.begin(), r.shard_files_.end());
        r.has_index_ = true;
        return r;
    }

    std::vector<std::string> st_files;
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            st_files.push_back(entry.path().string());
        }
    }
    if (st_files.size() != 1) {
        return Status::invalid_argument(
            "expected model.safetensors.index.json or exactly one .safetensors file",
            __FILE__, __LINE__);
    }
    auto reader_result = ps::io::SafetensorsReader::open(st_files[0]);
    if (!reader_result.ok()) {
        return reader_result.status();
    }
    auto reader = reader_result.release();
    auto list_result = reader.list_tensors();
    if (!list_result.ok()) {
        return list_result.status();
    }
    for (const auto& t : list_result.value()) {
        r.weight_map_[t] = fs::path(st_files[0]).filename().string();
        r.tensor_names_.push_back(t);
    }
    r.shard_files_.push_back(st_files[0]);
    return r;
}

Result<std::string> ShardResolver::shard_path(const std::string& tensor_name) const {
    auto it = weight_map_.find(tensor_name);
    if (it == weight_map_.end()) {
        return Status::invalid_argument("tensor not in weight_map", __FILE__, __LINE__);
    }
    return (fs::path(model_dir_) / it->second).string();
}

}  // namespace ps::quantization::fpx
