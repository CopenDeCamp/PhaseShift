#pragma once
#include <phaseshift/core/status.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ps {
namespace io {

enum class SType {
    F16 = 0,
    F32 = 1,
    BF16 = 2,
    I8 = 3,
    U8 = 4,
    I32 = 5,
};

struct StTensorSpec {
    SType dtype;
    std::vector<std::size_t> shape;
    std::uint64_t data_begin;
    std::uint64_t data_end;
};

class SafetensorsReader {
 public:
    static Result<SafetensorsReader> open(const std::string& path);

    ~SafetensorsReader() noexcept;
    SafetensorsReader(SafetensorsReader&& other) noexcept;
    SafetensorsReader& operator=(SafetensorsReader&& other) noexcept;
    SafetensorsReader(const SafetensorsReader&) = delete;
     SafetensorsReader& operator=(const SafetensorsReader&) = delete;

      Result<std::vector<std::string>> list_tensors() const;

      Result<StTensorSpec> tensor_spec(const std::string& name) const;

      Status read_tensor(const std::string& name, void* dst, std::size_t dst_bytes) const;

      Result<const void*> tensor_data(const std::string& name, std::size_t& out_bytes) const;

      // Absolute file offset where tensor payloads begin (8 + header_len).
      std::uint64_t data_base_offset() const;

      // Parsed "__metadata__" header entry (empty if absent).
      const std::map<std::string, std::string>& metadata() const;

 private:
     SafetensorsReader() noexcept : pimpl_(nullptr) {}

 private:
    struct Impl;
    Impl* pimpl_;
};

}
}
