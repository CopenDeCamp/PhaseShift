#include <phaseshift/models/qwen35/dflash2/draft_vocab_profile.h>

#include <nlohmann/json.hpp>
#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace ps::qwen35::dflash2 {

namespace {

namespace fs = std::filesystem;
using Json = nlohmann::json;

constexpr const char* kVocabFile = "dflash2-draft-vocab.u32";
constexpr const char* kMetadataFile = "dflash2-draft-vocab.json";
constexpr const char* kSchema = "phaseshift-dflash2-vocab-v1";

Status profile_error(const std::string& message) {
    return Status::invalid_argument(message.c_str(), __FILE__, __LINE__);
}

bool valid_sha256(const std::string& value) {
    if (value.size() != 64u)
        return false;
    return std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
               (c >= 'A' && c <= 'F');
    });
}

std::string sha256_hex(const std::vector<uint8_t>& data) {
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256(data.data(), data.size(), digest.data());
    std::ostringstream out;
    for (unsigned char byte : digest) {
        char text[3];
        std::snprintf(text, sizeof(text), "%02x", byte);
        out << text;
    }
    return out.str();
}

Result<std::vector<uint8_t>> read_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return profile_error("dflash2 draft vocabulary file cannot be opened: " +
                             path.string());
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    if (!file.eof() && file.fail())
        return profile_error("dflash2 draft vocabulary file read failed: " + path.string());
    return data;
}

uint32_t read_le32(const uint8_t* bytes) {
    return static_cast<uint32_t>(bytes[0]) |
           (static_cast<uint32_t>(bytes[1]) << 8u) |
           (static_cast<uint32_t>(bytes[2]) << 16u) |
           (static_cast<uint32_t>(bytes[3]) << 24u);
}

Result<Json> read_metadata(const fs::path& path) {
    std::ifstream file(path);
    if (!file)
        return profile_error("dflash2 draft vocabulary metadata cannot be opened: " +
                             path.string());
    Json metadata;
    try {
        file >> metadata;
    } catch (const Json::exception& error) {
        return profile_error("dflash2 draft vocabulary metadata is invalid: " +
                             std::string(error.what()));
    }
    return metadata;
}

Result<uint32_t> required_uint(const Json& metadata, const char* key) {
    if (!metadata.contains(key) || !metadata[key].is_number_unsigned())
        return profile_error(std::string("dflash2 draft vocabulary metadata missing uint: ") + key);
    const uint64_t value = metadata[key].get<uint64_t>();
    if (value == 0u || value > std::numeric_limits<uint32_t>::max())
        return profile_error(std::string("dflash2 draft vocabulary metadata uint out of range: ") +
                             key);
    return static_cast<uint32_t>(value);
}

Result<std::string> required_string(const Json& metadata, const char* key) {
    if (!metadata.contains(key) || !metadata[key].is_string() ||
        metadata[key].get<std::string>().empty()) {
        return profile_error(std::string("dflash2 draft vocabulary metadata missing string: ") +
                             key);
    }
    return metadata[key].get<std::string>();
}

Result<uint32_t> parse_head_mode(const std::optional<std::string>& value) {
    if (!value.has_value() || value->empty())
        return 0u;
    if (*value == "0")
        return 0u;
    if (*value == "1")
        return 1u;
    if (*value == "2")
        return 2u;
    return profile_error("PHASESHIFT_DFLASH2_INT2_HEAD must be 0, 1, or 2");
}

Result<bool> parse_auto_mode(const std::optional<std::string>& value) {
    if (!value.has_value() || value->empty())
        return true;
    if (*value == "0")
        return false;
    if (*value == "1")
        return true;
    return profile_error("PHASESHIFT_DFLASH2_DRAFT_VOCAB must be 0 or 1");
}

}

DraftVocabResolveOptions draft_vocab_resolve_options_from_environment() {
    DraftVocabResolveOptions options;
    if (const char* value = std::getenv("PHASESHIFT_DFLASH2_DRAFT_VOCAB"))
        options.draft_vocab = std::string(value);
    if (const char* value = std::getenv("PHASESHIFT_DFLASH2_DRAFT_VOCAB_FILE"))
        options.draft_vocab_file = std::string(value);
    if (const char* value = std::getenv("PHASESHIFT_DFLASH2_INT2_HEAD"))
        options.int2_head = std::string(value);
    return options;
}

Result<DraftVocabResolution> resolve_draft_vocab_profile(
    const std::string& model_dir,
    uint32_t target_vocab_size,
    uint32_t target_hidden_size,
    const DraftVocabResolveOptions& options) {
    auto mode_result = parse_head_mode(options.int2_head);
    if (!mode_result.ok())
        return mode_result.status();
    const uint32_t requested_mode = mode_result.release();
    const bool explicit_head = options.int2_head.has_value() && !options.int2_head->empty();
    auto auto_result = parse_auto_mode(options.draft_vocab);
    if (!auto_result.ok())
        return auto_result.status();
    const bool auto_enabled = auto_result.release();
    const bool has_explicit_file = options.draft_vocab_file.has_value() &&
                                   !options.draft_vocab_file->empty();

    if (has_explicit_file && explicit_head && requested_mode != 1u) {
        return profile_error(
            "PHASESHIFT_DFLASH2_DRAFT_VOCAB_FILE conflicts with INT2_HEAD=0 or 2");
    }
    if (has_explicit_file && !auto_enabled) {
        return profile_error(
            "PHASESHIFT_DFLASH2_DRAFT_VOCAB_FILE conflicts with DRAFT_VOCAB=0");
    }

    DraftVocabResolution resolution;
    resolution.int2_head_mode = requested_mode;
    if (has_explicit_file) {
        resolution.vocab_file = *options.draft_vocab_file;
        resolution.int2_head_mode = 1u;
        resolution.use_subset = true;
        resolution.explicit_file = true;
        resolution.reason = "explicit vocabulary file";
        return resolution;
    }

    if (!auto_enabled || (explicit_head && (requested_mode == 0u || requested_mode == 2u))) {
        resolution.reason = !auto_enabled ? "automatic vocabulary disabled"
                                          : "explicit INT2 mode ignores automatic vocabulary";
        return resolution;
    }

    const fs::path directory(model_dir);
    const fs::path vocab_path = directory / kVocabFile;
    const fs::path metadata_path = directory / kMetadataFile;
    const bool vocab_exists = fs::exists(vocab_path);
    const bool metadata_exists = fs::exists(metadata_path);
    if (!vocab_exists && !metadata_exists) {
        resolution.reason = "standard vocabulary profile absent";
        return resolution;
    }
    if (!vocab_exists || !metadata_exists) {
        return profile_error("dflash2 draft vocabulary profile is incomplete");
    }

    auto metadata_result = read_metadata(metadata_path);
    if (!metadata_result.ok())
        return metadata_result.status();
    const Json& metadata = metadata_result.value();
    if (!metadata.is_object() ||
        !metadata.contains("schema_version") || !metadata["schema_version"].is_string() ||
        metadata["schema_version"].get<std::string>() != kSchema) {
        return profile_error("dflash2 draft vocabulary metadata schema mismatch");
    }
    auto profile_id = required_string(metadata, "profile_id");
    if (!profile_id.ok()) return profile_id.status();
    auto vocab_file = required_string(metadata, "vocab_file");
    if (!vocab_file.ok()) return vocab_file.status();
    if (vocab_file.value() != kVocabFile)
        return profile_error("dflash2 draft vocabulary metadata file name mismatch");
    auto vocab_count = required_uint(metadata, "vocab_count");
    if (!vocab_count.ok()) return vocab_count.status();
    auto profile_vocab = required_uint(metadata, "target_vocab_size");
    if (!profile_vocab.ok()) return profile_vocab.status();
    auto profile_hidden = required_uint(metadata, "target_hidden_size");
    if (!profile_hidden.ok()) return profile_hidden.status();
    auto vocab_sha = required_string(metadata, "vocab_sha256");
    if (!vocab_sha.ok() || !valid_sha256(vocab_sha.value()))
        return profile_error("dflash2 draft vocabulary metadata vocab_sha256 invalid");
    auto tokenizer_sha = required_string(metadata, "tokenizer_sha256");
    if (!tokenizer_sha.ok() || !valid_sha256(tokenizer_sha.value()))
        return profile_error("dflash2 draft vocabulary metadata tokenizer_sha256 invalid");

    auto vocab_data_result = read_file(vocab_path);
    if (!vocab_data_result.ok()) return vocab_data_result.status();
    const std::vector<uint8_t>& vocab_data = vocab_data_result.value();
    if (vocab_data.size() != static_cast<std::size_t>(vocab_count.value()) * 4u ||
        (vocab_count.value() % 16u) != 0u) {
        return profile_error("dflash2 draft vocabulary payload size or geometry is invalid");
    }
    if (sha256_hex(vocab_data) != vocab_sha.value())
        return profile_error("dflash2 draft vocabulary payload sha256 mismatch");
    uint32_t previous = 0u;
    for (uint32_t index = 0u; index < vocab_count.value(); ++index) {
        const uint32_t token = read_le32(vocab_data.data() + static_cast<std::size_t>(index) * 4u);
        if (token >= profile_vocab.value() || (index != 0u && token <= previous))
            return profile_error("dflash2 draft vocabulary payload ids are invalid");
        previous = token;
    }

    if (profile_vocab.value() != target_vocab_size ||
        profile_hidden.value() != target_hidden_size) {
        resolution.reason = "standard vocabulary profile target shape mismatch";
        return resolution;
    }
    const fs::path tokenizer_path = directory / "tokenizer.json";
    if (!fs::exists(tokenizer_path)) {
        resolution.reason = "standard vocabulary profile tokenizer is absent";
        return resolution;
    }
    auto tokenizer_data_result = read_file(tokenizer_path);
    if (!tokenizer_data_result.ok()) return tokenizer_data_result.status();
    if (sha256_hex(tokenizer_data_result.value()) != tokenizer_sha.value()) {
        resolution.reason = "standard vocabulary profile tokenizer hash mismatch";
        return resolution;
    }

    resolution.vocab_file = vocab_path.string();
    resolution.int2_head_mode = 1u;
    resolution.use_subset = true;
    resolution.reason = "standard vocabulary profile matched";
    return resolution;
}

}
