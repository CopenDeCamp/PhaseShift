#pragma once
#include <phaseshift/core/status.h>
#include <map>
#include <string>
#include <vector>

namespace ps::quantization::fpx {

class ShardResolver {
public:
    static Result<ShardResolver> open(const std::string& model_dir);

    const std::vector<std::string>& tensor_names() const { return tensor_names_; }
    const std::vector<std::string>& shard_files() const { return shard_files_; }
    bool has_index() const { return has_index_; }
    Result<std::string> shard_path(const std::string& tensor_name) const;

private:
    std::string model_dir_;
    bool has_index_ = false;
    std::map<std::string, std::string> weight_map_;
    std::vector<std::string> shard_files_;
    std::vector<std::string> tensor_names_;
};

}  // namespace ps::quantization::fpx
