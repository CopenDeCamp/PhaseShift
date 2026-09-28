#include <phaseshift/quantization/offline/imatrix_format.h>
#include <phaseshift/quantization/fpx/crc32.h>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>

namespace ps {
namespace quantization {
namespace imatrix {

namespace {
uint64_t read_u64_le(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
    return v;
}

void write_u64_le(uint64_t v, uint8_t* p) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFFu);
}

bool valid_fingerprint(const std::string& s) {
    if (s.size() != 64) return false;
    for (char c : s) {
        const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        if (!ok) return false;
    }
    return true;
}

bool valid_hex8(const std::string& s) {
    if (s.size() != 8) return false;
    for (char c : s) {
        const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        if (!ok) return false;
    }
    return true;
}

std::string crc_hex8(const uint8_t* data, std::size_t len) {
    const uint32_t crc = ps::quantization::fpx::crc32_iso_hdlc(data, len);
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%08x", crc);
    return buf;
}
}  // namespace

std::string serialize_header(const PsimHeader& h) {
    nlohmann::json j;
    j["format"] = h.format;
    j["version"] = h.version;
    j["model"] = nlohmann::json::object();
    j["model"]["fingerprint"] = h.model.fingerprint;
    j["model"]["config_sha256"] = h.model.config_sha256;
    j["model"]["architecture"] = h.model.architecture;
    j["calibration"] = nlohmann::json::object();
    j["calibration"]["corpus_sha256"] = h.calibration.corpus_sha256;
    j["calibration"]["corpus_token_count"] = h.calibration.corpus_token_count;
    j["calibration"]["calibration_position_count"] = h.calibration.calibration_position_count;
    j["calibration"]["window"] = h.calibration.window;
    j["calibration"]["stride"] = h.calibration.stride;
    j["calibration"]["runtime_config"] = h.calibration.runtime_config;
    nlohmann::json entries = nlohmann::json::array();
    for (const auto& e : h.entries) {
        nlohmann::json je;
        je["name"] = e.name;
        je["k"] = e.k;
        je["count"] = e.count;
        je["dtype"] = e.dtype;
        je["data_offset"] = e.data_offset;
        je["data_bytes"] = e.data_bytes;
        je["crc32"] = e.crc32;
        entries.push_back(je);
    }
    j["entries"] = entries;
    if (!h.components.empty()) {
        nlohmann::json comps = nlohmann::json::array();
        for (const auto& c : h.components) {
            nlohmann::json jc;
            jc["corpus_sha256"] = c.corpus_sha256;
            jc["corpus_token_count"] = c.corpus_token_count;
            jc["calibration_position_count"] = c.calibration_position_count;
            comps.push_back(jc);
        }
        j["components"] = comps;
    }
    return j.dump();
}

Result<PsimHeader> parse_header(const std::string& json_text) {
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(json_text);
    } catch (...) {
        return Status::invalid_argument("imatrix header: bad JSON", __FILE__, __LINE__);
    }
    if (j.value("format", std::string()) != "phaseshift-imatrix") {
        return Status::invalid_argument("imatrix header: bad format", __FILE__, __LINE__);
    }
    if (j.value("version", 0) != 1) {
        return Status::invalid_argument("imatrix header: bad version", __FILE__, __LINE__);
    }
    PsimHeader h;
    h.format = j["format"].get<std::string>();
    h.version = j["version"].get<uint64_t>();
    h.model.fingerprint = j["model"].value("fingerprint", std::string());
    h.model.config_sha256 = j["model"].value("config_sha256", std::string());
    h.model.architecture = j["model"].value("architecture", std::string());
    h.calibration.corpus_sha256 = j["calibration"].value("corpus_sha256", std::string());
    h.calibration.corpus_token_count = j["calibration"].value("corpus_token_count", 0);
    h.calibration.calibration_position_count = j["calibration"].value("calibration_position_count", 0);
    h.calibration.window = j["calibration"].value("window", 0);
    h.calibration.stride = j["calibration"].value("stride", 0);
    h.calibration.runtime_config = j["calibration"].value("runtime_config", std::string());
    for (const auto& e : j["entries"]) {
        PsimEntryDesc ed;
        ed.name = e.at("name").get<std::string>();
        ed.k = e.at("k").get<uint64_t>();
        ed.count = e.at("count").get<uint64_t>();
        ed.dtype = e.value("dtype", std::string("f64_sum_sq"));
        ed.data_offset = e.at("data_offset").get<uint64_t>();
        ed.data_bytes = e.at("data_bytes").get<uint64_t>();
        ed.crc32 = e.at("crc32").get<std::string>();
        h.entries.push_back(ed);
    }
    if (j.contains("components") && j["components"].is_array()) {
        for (const auto& c : j["components"]) {
            PsimComponent pc;
            pc.corpus_sha256 = c.at("corpus_sha256").get<std::string>();
            pc.corpus_token_count = c.at("corpus_token_count").get<uint64_t>();
            pc.calibration_position_count = c.at("calibration_position_count").get<uint64_t>();
            h.components.push_back(pc);
        }
    }
    return h;
}

Result<PsimHeader> validate_header(const PsimHeader& h, uint64_t payload_size) {
    auto fail = [&](const char* msg) {
        return Status::invalid_argument(msg, __FILE__, __LINE__);
    };
    if (h.calibration.calibration_position_count == 0) {
        return fail("imatrix: calibration_position_count == 0");
    }
    if (!valid_fingerprint(h.model.fingerprint)) {
        return fail("imatrix: bad model fingerprint format");
    }
    if (h.model.architecture.empty()) {
        return fail("imatrix: empty architecture");
    }
    std::map<std::string, int> seen;
    uint64_t prev_end = 0;
    for (const auto& e : h.entries) {
        if (e.name.empty()) return fail("imatrix: empty entry name");
        if (seen.count(e.name)) return fail("imatrix: duplicate tensor name");
        seen[e.name] = 1;
        if (e.k == 0) return fail("imatrix: entry k == 0");
        if (e.count == 0) return fail("imatrix: entry count == 0");
        if (e.count != h.calibration.calibration_position_count) {
            return fail("imatrix: entry count != calibration_position_count");
        }
        if (e.data_bytes != e.k * sizeof(double)) {
            return fail("imatrix: data_bytes != k * sizeof(double)");
        }
        if (e.data_offset > payload_size || e.data_bytes > payload_size - e.data_offset) {
            return fail("imatrix: data bounds");
        }
        if (e.data_offset < prev_end) {
            return fail("imatrix: payload entry ranges overlap");
        }
        prev_end = e.data_offset + e.data_bytes;
        if (!valid_hex8(e.crc32)) return fail("imatrix: bad crc32 format");
    }
    return h;
}

Result<PsimFile> read_psim(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return Status::invalid_argument("imatrix: cannot open file", __FILE__, __LINE__);
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (bytes.size() < 16) {
        return Status::invalid_argument("imatrix: file too small", __FILE__, __LINE__);
    }
    if (std::memcmp(bytes.data(), kPsimMagic, 8) != 0) {
        return Status::invalid_argument("imatrix: bad magic", __FILE__, __LINE__);
    }
    const uint64_t header_len = read_u64_le(bytes.data() + 8);
    if (header_len > bytes.size() - 16) {
        return Status::invalid_argument("imatrix: header exceeds file", __FILE__, __LINE__);
    }
    const uint64_t header_end = 16 + header_len;
    const std::string json_text(
        reinterpret_cast<const char*>(bytes.data() + 16),
        static_cast<std::size_t>(header_len));
    auto parsed = parse_header(json_text);
    if (!parsed.ok()) return parsed.status();
    PsimFile file;
    file.header = parsed.release();
    file.payload.assign(bytes.begin() + static_cast<std::ptrdiff_t>(header_end), bytes.end());
    auto vh = validate_header(file.header, file.payload.size());
    if (!vh.ok()) return vh.status();
    for (const auto& e : file.header.entries) {
        const std::string actual = crc_hex8(
            file.payload.data() + e.data_offset, static_cast<std::size_t>(e.data_bytes));
        if (e.crc32 != actual) {
            return Status::invalid_argument("imatrix: crc mismatch", __FILE__, __LINE__);
        }
        const double* d = reinterpret_cast<const double*>(file.payload.data() + e.data_offset);
        for (uint64_t j = 0; j < e.k; ++j) {
            if (!std::isfinite(d[j])) {
                return Status::invalid_argument("imatrix: non-finite sum_sq", __FILE__, __LINE__);
            }
            if (d[j] < 0.0) {
                return Status::invalid_argument("imatrix: negative sum_sq", __FILE__, __LINE__);
            }
        }
    }
    return file;
}

Status write_psim(const PsimHeader& h, const std::vector<uint8_t>& payload, const std::string& path) {
    auto vh = validate_header(h, payload.size());
    if (!vh.ok()) return vh.status();
    for (const auto& e : h.entries) {
        const std::string actual = crc_hex8(
            payload.data() + e.data_offset, static_cast<std::size_t>(e.data_bytes));
        if (e.crc32 != actual) {
            return Status::invalid_argument("imatrix: crc mismatch on write", __FILE__, __LINE__);
        }
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return Status::invalid_argument("imatrix: cannot open output", __FILE__, __LINE__);
    }
    const std::string json_text = serialize_header(h);
    f.write(kPsimMagic, 8);
    const uint64_t header_len = static_cast<uint64_t>(json_text.size());
    uint8_t len_buf[8];
    write_u64_le(header_len, len_buf);
    f.write(reinterpret_cast<const char*>(len_buf), 8);
    f.write(json_text.data(), static_cast<std::streamsize>(json_text.size()));
    if (!payload.empty()) {
        f.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    }
    if (!f) {
        return Status::invalid_argument("imatrix: write failed", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

std::string compute_merged_corpus_sha256(const std::vector<PsimComponent>& components) {
    std::vector<PsimComponent> sorted = components;
    std::sort(sorted.begin(), sorted.end(), [](const PsimComponent& a, const PsimComponent& b) {
        return a.corpus_sha256 < b.corpus_sha256;
    });
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& c : sorted) {
        nlohmann::json jc;
        jc["corpus_sha256"] = c.corpus_sha256;
        jc["corpus_token_count"] = c.corpus_token_count;
        jc["calibration_position_count"] = c.calibration_position_count;
        arr.push_back(jc);
    }
    const std::string canon = arr.dump();
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(canon.data()), canon.size(), hash);
    std::ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        char buf[3];
        std::snprintf(buf, sizeof(buf), "%02x", hash[i]);
        oss << buf;
    }
    return oss.str();
}

Result<PsimFile> merge_psim(const PsimFile& a, const PsimFile& b) {
    auto fail = [&](const char* msg) {
        return Status::invalid_argument(msg, __FILE__, __LINE__);
    };
    const PsimHeader& ha = a.header;
    const PsimHeader& hb = b.header;
    if (ha.model.fingerprint != hb.model.fingerprint) return fail("imatrix merge: fingerprint mismatch");
    if (ha.model.architecture != hb.model.architecture) return fail("imatrix merge: architecture mismatch");
    if (ha.calibration.runtime_config != hb.calibration.runtime_config) return fail("imatrix merge: runtime_config mismatch");
    if (ha.calibration.window != hb.calibration.window) return fail("imatrix merge: window mismatch");
    if (ha.calibration.stride != hb.calibration.stride) return fail("imatrix merge: stride mismatch");
    if (ha.entries.size() != hb.entries.size()) return fail("imatrix merge: entry count mismatch");
    for (std::size_t i = 0; i < ha.entries.size(); ++i) {
        if (ha.entries[i].name != hb.entries[i].name) return fail("imatrix merge: tensor name mismatch");
        if (ha.entries[i].k != hb.entries[i].k) return fail("imatrix merge: k mismatch");
        if (ha.entries[i].dtype != hb.entries[i].dtype) return fail("imatrix merge: dtype mismatch");
    }
    if (ha.calibration.calibration_position_count > std::numeric_limits<uint64_t>::max() - hb.calibration.calibration_position_count) {
        return fail("imatrix merge: calibration_position_count overflow");
    }
    if (ha.calibration.corpus_token_count > std::numeric_limits<uint64_t>::max() - hb.calibration.corpus_token_count) {
        return fail("imatrix merge: corpus_token_count overflow");
    }

    PsimFile out;
    out.header = ha;
    out.header.calibration.calibration_position_count =
        ha.calibration.calibration_position_count + hb.calibration.calibration_position_count;
    out.header.calibration.corpus_token_count =
        ha.calibration.corpus_token_count + hb.calibration.corpus_token_count;

    std::vector<PsimComponent> comps;
    if (ha.components.empty()) {
        comps.push_back({ha.calibration.corpus_sha256, ha.calibration.corpus_token_count,
                         ha.calibration.calibration_position_count});
    } else {
        comps = ha.components;
    }
    if (hb.components.empty()) {
        comps.push_back({hb.calibration.corpus_sha256, hb.calibration.corpus_token_count,
                         hb.calibration.calibration_position_count});
    } else {
        comps.insert(comps.end(), hb.components.begin(), hb.components.end());
    }
    out.header.components = comps;
    out.header.calibration.corpus_sha256 = compute_merged_corpus_sha256(comps);

    std::vector<uint8_t> out_payload(a.payload.size() + b.payload.size());
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < ha.entries.size(); ++i) {
        const uint64_t k = ha.entries[i].k;
        const uint64_t bytes = k * sizeof(double);
        const double* pa = reinterpret_cast<const double*>(a.payload.data() + ha.entries[i].data_offset);
        const double* pb = reinterpret_cast<const double*>(b.payload.data() + hb.entries[i].data_offset);
        double* pd = reinterpret_cast<double*>(out_payload.data() + cursor);
        for (uint64_t j = 0; j < k; ++j) {
            const double s = pa[j] + pb[j];
            if (!std::isfinite(s)) {
                return fail("imatrix merge: non-finite merged sum_sq");
            }
            pd[j] = s;
        }
        PsimEntryDesc ne = ha.entries[i];
        ne.count = out.header.calibration.calibration_position_count;
        ne.data_offset = cursor;
        ne.crc32 = crc_hex8(out_payload.data() + cursor, static_cast<std::size_t>(bytes));
        out.header.entries[i] = ne;
        cursor += static_cast<std::size_t>(bytes);
    }
    out.payload = std::move(out_payload);
    return out;
}

}
}
}