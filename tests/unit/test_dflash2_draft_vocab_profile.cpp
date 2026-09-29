#include <phaseshift/models/qwen35/dflash2/draft_vocab_profile.h>

#include <openssl/sha.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using ps::qwen35::dflash2::DraftVocabResolveOptions;
using ps::qwen35::dflash2::DraftVocabResolution;
using ps::qwen35::dflash2::draft_vocab_resolve_options_from_environment;
using ps::qwen35::dflash2::resolve_draft_vocab_profile;

namespace {

std::string sha256_hex(const std::vector<uint8_t>& data) {
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256(data.data(), data.size(), digest.data());
    char text[3];
    std::string out;
    out.reserve(64u);
    for (unsigned char byte : digest) {
        std::snprintf(text, sizeof(text), "%02x", byte);
        out += text;
    }
    return out;
}

std::vector<uint8_t> ids() {
    std::vector<uint8_t> bytes;
    for (uint32_t i = 0u; i < 16u; ++i) {
        const uint32_t id = i * 2u;
        bytes.push_back(static_cast<uint8_t>(id));
        bytes.push_back(static_cast<uint8_t>(id >> 8u));
        bytes.push_back(static_cast<uint8_t>(id >> 16u));
        bytes.push_back(static_cast<uint8_t>(id >> 24u));
    }
    return bytes;
}

void write_bytes(const fs::path& path, const std::vector<uint8_t>& data) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
}

void write_metadata(const fs::path& path, const std::vector<uint8_t>& payload,
                    const std::string& tokenizer_sha, uint32_t target_vocab = 32u,
                    uint32_t target_hidden = 8u) {
    std::ofstream file(path);
    file << "{\"schema_version\":\"phaseshift-dflash2-vocab-v1\","
            "\"profile_id\":\"test\","
            "\"vocab_file\":\"dflash2-draft-vocab.u32\","
         << "\"vocab_count\":16,\"target_vocab_size\":" << target_vocab
         << ",\"target_hidden_size\":" << target_hidden
         << ",\"vocab_sha256\":\"" << sha256_hex(payload)
         << "\",\"tokenizer_sha256\":\"" << tokenizer_sha << "\"}";
}

bool resolved(const ps::Result<DraftVocabResolution>& result, uint32_t mode,
              bool subset) {
    return result.ok() && result.value().int2_head_mode == mode &&
           result.value().use_subset == subset;
}

}

int main() {
    const fs::path root = fs::temp_directory_path() / "phaseshift-dflash2-profile-test";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const std::vector<uint8_t> payload = ids();
    const std::vector<uint8_t> tokenizer{'t', 'o', 'k', 'e', 'n'};
    write_bytes(root / "tokenizer.json", tokenizer);
    write_bytes(root / "dflash2-draft-vocab.u32", payload);
    write_metadata(root / "dflash2-draft-vocab.json", payload, sha256_hex(tokenizer));

    int failures = 0;
    auto check = [&](bool condition, const char* label) {
        if (!condition) {
            std::printf("FAIL: %s\n", label);
            ++failures;
        }
    };

    check(resolved(resolve_draft_vocab_profile(root.string(), 32u, 8u), 1u, true),
          "matched standard profile enables subset and INT2");
    fs::remove(root / "dflash2-draft-vocab.u32");
    fs::remove(root / "dflash2-draft-vocab.json");
    check(resolved(resolve_draft_vocab_profile(root.string(), 32u, 8u), 0u, false),
          "no standard files preserve legacy mode");
    DraftVocabResolveOptions missing_head1;
    missing_head1.int2_head = "1";
    check(resolved(resolve_draft_vocab_profile(root.string(), 32u, 8u, missing_head1), 1u, false),
          "HEAD=1 without profile keeps full INT2");
    write_bytes(root / "dflash2-draft-vocab.u32", payload);
    write_metadata(root / "dflash2-draft-vocab.json", payload, sha256_hex(tokenizer));
    DraftVocabResolveOptions head0;
    head0.int2_head = "0";
    check(resolved(resolve_draft_vocab_profile(root.string(), 32u, 8u, head0), 0u, false),
          "HEAD=0 ignores automatic profile");
    DraftVocabResolveOptions head2;
    head2.int2_head = "2";
    check(resolved(resolve_draft_vocab_profile(root.string(), 32u, 8u, head2), 2u, false),
          "HEAD=2 ignores automatic profile");
    check(resolved(resolve_draft_vocab_profile(root.string(), 31u, 8u), 0u, false),
          "target shape mismatch preserves legacy mode");

    DraftVocabResolveOptions head1;
    head1.int2_head = "1";
    check(resolved(resolve_draft_vocab_profile(root.string(), 32u, 8u, head1), 1u, true),
          "HEAD=1 accepts matched profile");
    check(resolved(resolve_draft_vocab_profile(root.string(), 31u, 8u, head1), 1u, false),
          "HEAD=1 shape mismatch falls back to full INT2");

    DraftVocabResolveOptions off;
    off.draft_vocab = "0";
    off.int2_head = "1";
    check(resolved(resolve_draft_vocab_profile(root.string(), 32u, 8u, off), 1u, false),
          "DRAFT_VOCAB=0 disables automatic subset");
    setenv("PHASESHIFT_DFLASH2_DRAFT_VOCAB", "0", 1);
    setenv("PHASESHIFT_DFLASH2_INT2_HEAD", "1", 1);
    check(resolved(resolve_draft_vocab_profile(root.string(), 32u, 8u,
                                               draft_vocab_resolve_options_from_environment()),
                   1u, false),
          "environment DRAFT_VOCAB=0 disables automatic subset");
    unsetenv("PHASESHIFT_DFLASH2_DRAFT_VOCAB");
    unsetenv("PHASESHIFT_DFLASH2_INT2_HEAD");

    DraftVocabResolveOptions explicit_file;
    explicit_file.draft_vocab_file = (root / "dflash2-draft-vocab.u32").string();
    check(resolved(resolve_draft_vocab_profile(root.string(), 32u, 8u, explicit_file), 1u, true),
          "explicit file enables subset without metadata");
    setenv("PHASESHIFT_DFLASH2_DRAFT_VOCAB_FILE", explicit_file.draft_vocab_file->c_str(), 1);
    check(resolved(resolve_draft_vocab_profile(root.string(), 32u, 8u,
                                               draft_vocab_resolve_options_from_environment()),
                   1u, true),
          "environment explicit file enables subset");
    unsetenv("PHASESHIFT_DFLASH2_DRAFT_VOCAB_FILE");

    DraftVocabResolveOptions explicit_zero = explicit_file;
    explicit_zero.int2_head = "0";
    check(!resolve_draft_vocab_profile(root.string(), 32u, 8u, explicit_zero).ok(),
          "explicit file conflicts with HEAD=0");
    DraftVocabResolveOptions explicit_two = explicit_file;
    explicit_two.int2_head = "2";
    check(!resolve_draft_vocab_profile(root.string(), 32u, 8u, explicit_two).ok(),
          "explicit file conflicts with HEAD=2");

    fs::remove(root / "dflash2-draft-vocab.u32");
    check(!resolve_draft_vocab_profile(root.string(), 32u, 8u).ok(),
          "partial standard profile is an error");
    write_bytes(root / "dflash2-draft-vocab.u32", payload);
    write_bytes(root / "dflash2-draft-vocab.json", std::vector<uint8_t>{'{', 'b', 'a', 'd'});
    check(!resolve_draft_vocab_profile(root.string(), 32u, 8u).ok(),
          "broken metadata is an error");
    write_metadata(root / "dflash2-draft-vocab.json", payload, "00000000000000000000000000000000"
                   "00000000000000000000000000000000");
    check(resolved(resolve_draft_vocab_profile(root.string(), 32u, 8u), 0u, false),
          "tokenizer hash mismatch preserves legacy mode");
    write_metadata(root / "dflash2-draft-vocab.json", payload, sha256_hex(tokenizer));
    std::vector<uint8_t> corrupt = payload;
    corrupt[0] ^= 1u;
    write_bytes(root / "dflash2-draft-vocab.u32", corrupt);
    check(!resolve_draft_vocab_profile(root.string(), 32u, 8u).ok(),
          "payload hash mismatch is an error");

    fs::remove_all(root, ec);
    return failures == 0 ? 0 : 1;
}
