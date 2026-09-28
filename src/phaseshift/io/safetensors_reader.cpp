#include <phaseshift/io/safetensors_reader.h>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <cstdint>
#include <limits>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <nlohmann/json.hpp>

namespace ps {
namespace io {

namespace {

struct MappedFile {
    int fd = -1;
    void* data = MAP_FAILED;
    std::size_t size = 0;

    MappedFile() = default;

    ~MappedFile() {
        close();
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept
        : fd(other.fd), data(other.data), size(other.size) {
        other.fd = -1;
        other.data = MAP_FAILED;
        other.size = 0;
    }

    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            close();
            fd = other.fd;
            data = other.data;
            size = other.size;
            other.fd = -1;
            other.data = MAP_FAILED;
            other.size = 0;
        }
        return *this;
    }

    void close() {
        if (data != MAP_FAILED) {
            munmap(data, size);
            data = MAP_FAILED;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }

    bool ok() const {
        return data != MAP_FAILED && fd >= 0 && size > 0;
    }
};

Result<MappedFile> mmap_file(const std::string& path) {
    MappedFile mf;
    mf.fd = ::open(path.c_str(), O_RDONLY);
    if (mf.fd < 0) {
        return Status::invalid_argument("cannot open file", __FILE__, __LINE__);
    }

    struct stat st{};
    if (fstat(mf.fd, &st) < 0) {
        ::close(mf.fd);
        mf.fd = -1;
        return Status::invalid_argument("cannot stat file", __FILE__, __LINE__);
    }

    if (st.st_size == 0) {
        ::close(mf.fd);
        mf.fd = -1;
        return Status::invalid_argument("file is empty", __FILE__, __LINE__);
    }

    mf.size = static_cast<std::size_t>(st.st_size);

    mf.data = mmap(nullptr, mf.size, PROT_READ, MAP_PRIVATE, mf.fd, 0);
    if (mf.data == MAP_FAILED) {
        ::close(mf.fd);
        mf.fd = -1;
        return Status::invalid_argument("mmap failed", __FILE__, __LINE__);
    }

    return mf;
}

}



struct SafetensorsReader::Impl {
    std::string path;
    std::uint64_t data_base_offset;
    std::map<std::string, StTensorSpec> tensors;
    std::map<std::string, std::string> metadata;
    MappedFile file_map;
};

Result<SafetensorsReader> SafetensorsReader::open(const std::string& path) {
    auto mf_result = mmap_file(path);
    if (!mf_result.ok()) {
        return mf_result.status();
    }
    MappedFile mf = mf_result.release();

    if (!mf.ok()) {
        return Status::invalid_argument("mmap failed", __FILE__, __LINE__);
    }

    std::size_t file_size = mf.size;
    if (file_size < 8) {
        return Status::invalid_argument("file too small", __FILE__, __LINE__);
    }

    const char* base = static_cast<const char*>(mf.data);
    std::uint64_t header_len = 0;
    std::memcpy(&header_len, base, 8);

    if (header_len == 0 || header_len + 8 > file_size) {
        return Status::invalid_argument("invalid header length", __FILE__, __LINE__);
    }

    nlohmann::json header;
    try {
        header = nlohmann::json::parse(base + 8, base + 8 + header_len);
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            ("bad safetensors header: " + std::string(e.what())).c_str(), __FILE__, __LINE__);
    }

    if (!header.is_object()) {
        return Status::invalid_argument("bad safetensors header: not an object", __FILE__, __LINE__);
    }

    std::map<std::string, StTensorSpec> tensors;
    std::map<std::string, std::string> metadata;
    try {
        for (auto it = header.begin(); it != header.end(); ++it) {
            const nlohmann::json& spec_json = it.value();
            if (!spec_json.is_object()) {
                return Status::invalid_argument("bad tensor entry", __FILE__, __LINE__);
            }
            if (it.key() == "__metadata__") {
                for (auto mit = spec_json.begin(); mit != spec_json.end(); ++mit) {
                    metadata[mit.key()] = mit.value().is_string()
                        ? mit.value().get<std::string>() : mit.value().dump();
                }
                continue;
            }
            StTensorSpec spec;
            spec.dtype = SType::F32;
            spec.shape = {};
            spec.data_begin = 0;
            spec.data_end = 0;

            std::string dt = spec_json.value("dtype", std::string());
            if (dt == "BF16") spec.dtype = SType::BF16;
            else if (dt == "F32") spec.dtype = SType::F32;
            else if (dt == "F16") spec.dtype = SType::F16;
            else if (dt == "I8") spec.dtype = SType::I8;
            else if (dt == "U8") spec.dtype = SType::U8;
            else if (dt == "I32") spec.dtype = SType::I32;
            else return Status::invalid_argument("unsupported dtype", __FILE__, __LINE__);

            if (spec_json.contains("shape") && spec_json["shape"].is_array()) {
                for (const auto& d : spec_json["shape"]) {
                    if (!d.is_number_unsigned()) {
                        return Status::invalid_argument("bad shape dimension", __FILE__, __LINE__);
                    }
                    const std::size_t dim = d.get<std::size_t>();
                    if (dim == 0) {
                        return Status::invalid_argument("zero dimension", __FILE__, __LINE__);
                    }
                    spec.shape.push_back(dim);
                }
            } else {
                return Status::invalid_argument("missing shape", __FILE__, __LINE__);
            }

            if (!spec_json.contains("data_offsets") || !spec_json["data_offsets"].is_array() ||
                spec_json["data_offsets"].size() != 2 ||
                !spec_json["data_offsets"][0].is_number_unsigned() ||
                !spec_json["data_offsets"][1].is_number_unsigned()) {
                return Status::invalid_argument("data_offsets needs 2 values", __FILE__, __LINE__);
            }
            spec.data_begin = spec_json["data_offsets"][0].get<std::uint64_t>();
            spec.data_end = spec_json["data_offsets"][1].get<std::uint64_t>();
            if (spec.data_end < spec.data_begin) {
                return Status::invalid_argument("invalid data_offsets ordering", __FILE__, __LINE__);
            }

            std::size_t expected_bytes = 1;
            std::size_t elem_size = 2;
            switch (spec.dtype) {
                case SType::F16: elem_size = 2; break;
                case SType::F32: elem_size = 4; break;
                case SType::BF16: elem_size = 2; break;
                case SType::I8: elem_size = 1; break;
                case SType::U8: elem_size = 1; break;
                case SType::I32: elem_size = 4; break;
            }
            for (const std::size_t d : spec.shape) {
                if (d != 0 && expected_bytes > (std::numeric_limits<std::size_t>::max() / d)) {
                    return Status::invalid_argument("shape overflow", __FILE__, __LINE__);
                }
                expected_bytes *= d;
            }
            if (expected_bytes > (std::numeric_limits<std::size_t>::max() / elem_size)) {
                return Status::invalid_argument("element count overflow", __FILE__, __LINE__);
            }
            expected_bytes *= elem_size;
            if (expected_bytes != spec.data_end - spec.data_begin) {
                return Status::invalid_argument("shape/byte count mismatch", __FILE__, __LINE__);
            }

            if (spec.data_end > file_size - (8 + header_len)) {
                return Status::invalid_argument("data out of file bounds", __FILE__, __LINE__);
            }

            if (tensors.find(it.key()) != tensors.end()) {
                return Status::invalid_argument("duplicate tensor name", __FILE__, __LINE__);
            }
            tensors[it.key()] = spec;
        }
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            ("bad safetensors header: " + std::string(e.what())).c_str(), __FILE__, __LINE__);
    }

    auto pimpl = std::make_unique<Impl>();
    pimpl->path = path;
    pimpl->data_base_offset = 8 + header_len;
    pimpl->tensors = std::move(tensors);
    pimpl->metadata = std::move(metadata);
    pimpl->file_map = std::move(mf);

    SafetensorsReader reader;
    reader.pimpl_ = pimpl.release();
    return reader;
}

SafetensorsReader::~SafetensorsReader() noexcept {
    delete pimpl_;
}

SafetensorsReader::SafetensorsReader(SafetensorsReader&& other) noexcept : pimpl_(other.pimpl_) {
    other.pimpl_ = nullptr;
}

SafetensorsReader& SafetensorsReader::operator=(SafetensorsReader&& other) noexcept {
    if (this != &other) {
        delete pimpl_;
        pimpl_ = other.pimpl_;
        other.pimpl_ = nullptr;
    }
    return *this;
}

Result<std::vector<std::string>> SafetensorsReader::list_tensors() const {
    std::vector<std::string> out;
    out.reserve(pimpl_->tensors.size());
    for (const auto& kv : pimpl_->tensors) {
        out.push_back(kv.first);
    }
    return out;
}

Result<StTensorSpec> SafetensorsReader::tensor_spec(const std::string& name) const {
    auto it = pimpl_->tensors.find(name);
    if (it == pimpl_->tensors.end()) {
        return Status::invalid_argument("tensor not found", __FILE__, __LINE__);
    }
    return it->second;
}

std::uint64_t SafetensorsReader::data_base_offset() const {
    return pimpl_->data_base_offset;
}

const std::map<std::string, std::string>& SafetensorsReader::metadata() const {
    return pimpl_->metadata;
}

Status SafetensorsReader::read_tensor(const std::string& name, void* dst, std::size_t dst_bytes) const {
    auto spec_result = tensor_spec(name);
    if (!spec_result.ok()) return spec_result.status();
    const auto& spec = spec_result.value();

    std::uint64_t expected_bytes = spec.data_end - spec.data_begin;
    if (expected_bytes != dst_bytes) {
        return Status::invalid_argument("tensor size mismatch", __FILE__, __LINE__);
    }

    std::uint64_t file_offset = pimpl_->data_base_offset + spec.data_begin;
    if (file_offset + expected_bytes > pimpl_->file_map.size) {
        return Status::invalid_argument("tensor data out of range", __FILE__, __LINE__);
    }

    std::memcpy(dst,
                static_cast<const char*>(pimpl_->file_map.data) + file_offset,
                expected_bytes);
    return Status::make_ok();
}

Result<const void*> SafetensorsReader::tensor_data(const std::string& name, std::size_t& out_bytes) const {
    auto spec_result = tensor_spec(name);
    if (!spec_result.ok()) return spec_result.status();
    const auto& spec = spec_result.value();

    std::uint64_t expected_bytes = spec.data_end - spec.data_begin;
    std::uint64_t file_offset = pimpl_->data_base_offset + spec.data_begin;
    if (file_offset + expected_bytes > pimpl_->file_map.size) {
        return Status::invalid_argument("tensor data out of range", __FILE__, __LINE__);
    }

    out_bytes = static_cast<std::size_t>(expected_bytes);
    return static_cast<const char*>(pimpl_->file_map.data) + file_offset;
}

}
}
