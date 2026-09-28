#pragma once
#include <phaseshift/core/status.h>
#include <cstdint>
#include <string>
#include <vector>

namespace ps {
namespace quantization {
namespace imatrix {

struct PsimEntryDesc {
    std::string name;
    uint64_t k = 0;
    uint64_t count = 0;
    std::string dtype = "f64_sum_sq";
    uint64_t data_offset = 0;
    uint64_t data_bytes = 0;
    std::string crc32;
};

struct PsimModel {
    std::string fingerprint;
    std::string config_sha256;
    std::string architecture;
};

struct PsimCalibration {
    std::string corpus_sha256;
    uint64_t corpus_token_count = 0;
    uint64_t calibration_position_count = 0;
    uint64_t window = 0;
    uint64_t stride = 0;
    std::string runtime_config;
};

struct PsimComponent {
    std::string corpus_sha256;
    uint64_t corpus_token_count = 0;
    uint64_t calibration_position_count = 0;
};

struct PsimHeader {
    std::string format = "phaseshift-imatrix";
    uint64_t version = 1;
    PsimModel model;
    PsimCalibration calibration;
    std::vector<PsimEntryDesc> entries;
    std::vector<PsimComponent> components;
};

constexpr char kPsimMagic[8] = {'P', 'S', 'I', 'M', 'A', 'T', '0', '1'};

struct PsimFile {
    PsimHeader header;
    std::vector<uint8_t> payload;
};

std::string serialize_header(const PsimHeader& h);
Result<PsimHeader> parse_header(const std::string& json_text);
Result<PsimFile> read_psim(const std::string& path);
Status write_psim(const PsimHeader& h, const std::vector<uint8_t>& payload, const std::string& path);
Result<PsimHeader> validate_header(const PsimHeader& h, uint64_t payload_size);
Result<PsimFile> merge_psim(const PsimFile& a, const PsimFile& b);
std::string compute_merged_corpus_sha256(const std::vector<PsimComponent>& components);

}
}
}