#include <phaseshift/weights/weight_loader.h>

#include <phaseshift/core/memory/types.h>
#include <phaseshift/quantization/fpx/runtime_resolution.h>
#include <phaseshift/quantization/psq/quant_preshuffle.h>
#include <phaseshift/quantization/fp8/block128_native.h>
#include <phaseshift/quantization/mxfp4/mxfp4_native.h>
#include <phaseshift/weights/canonical_partition.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace ps::weights {
namespace {

using ps::quantization::fpx::QuantizedEncoding;
using ps::quantization::fpx::QuantizedModelReader;
using ps::quantization::fpx::QuantizedTensorView;
using Tensor = gpu::Tensor;
namespace fs = std::filesystem;

Tensor alloc_tensor(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& strides, gpu::GpuArena& arena) {
    std::size_t n = 1;
    for (auto d : shape) n *= d;
    auto v = arena.allocate_aligned(n * sizeof(std::uint16_t), 16);
    if (!v.ok()) return {};
    auto t = Tensor::create<std::uint16_t>(v.release(), shape, strides);
    if (!t.ok()) return {};
    return t.release();
}

Tensor alloc_tensor_f32(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& strides, gpu::GpuArena& arena) {
    std::size_t n = 1;
    for (auto d : shape) n *= d;
    auto v = arena.allocate_aligned(n * sizeof(float), 16);
    if (!v.ok()) return {};
    auto t = Tensor::create<float>(v.release(), shape, strides);
    if (!t.ok()) return {};
    return t.release();
}

gpu::Tensor alloc_bytes(
    std::size_t bytes,
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides,
    gpu::GpuArena& arena)
{
    auto v = arena.allocate_aligned(bytes, 16);
    if (!v.ok()) return {};
    auto t = gpu::Tensor::create<uint8_t>(v.release(), shape, strides);
    if (!t.ok()) return {};
    return t.release();
}

gpu::Tensor alloc_bytes16(
    std::size_t bytes,
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides,
    gpu::GpuArena& arena)
{
    auto v = arena.allocate_aligned(bytes, 16);
    if (!v.ok()) return {};
    auto t = gpu::Tensor::create<std::uint16_t>(v.release(), shape, strides);
    if (!t.ok()) return {};
    return t.release();
}

Status copy_to_device(void* dst, const void* src, std::size_t bytes, hipStream_t stream) {
    hipError_t e = hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream);
    if (e != hipSuccess) {
        return Status::hip_error("hipMemcpyAsync", hipGetErrorString(e), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status expect_bytes(const std::string& name, const quantization::fpx::ByteSpan& span, std::size_t expected) {
    if ((expected > 0 && span.data == nullptr) || span.size != expected) {
        return Status::invalid_argument(
            ("quantized tensor byte count mismatch: " + name).c_str(), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

std::vector<std::size_t> contiguous_strides(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t i = shape.size(); i-- > 1u;) strides[i - 1u] = strides[i] * shape[i];
    return strides;
}

Status upload_tensor_direct(const SafetensorsCollection& reader, const std::string& name, void* dst, hipStream_t stream) {
    std::size_t bytes = 0;
    auto ptr_result = reader.tensor_data(name, bytes);
    if (!ptr_result.ok()) return ptr_result.status();
    const void* src = ptr_result.value();

    hipError_t e = hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream);
    if (e != hipSuccess) {
        return Status::hip_error("hipMemcpyAsync", hipGetErrorString(e), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status upload_tensor_f32_to_bf16(const SafetensorsCollection& reader, const std::string& name, void* dst, std::size_t n, hipStream_t stream) {
    std::size_t bytes = 0;
    auto ptr_result = reader.tensor_data(name, bytes);
    if (!ptr_result.ok()) return ptr_result.status();
    const void* src = ptr_result.value();

    std::vector<bf16_t> converted(n);
    const float* src_f = static_cast<const float*>(src);
    for (std::size_t i = 0; i < n; ++i) {
        converted[i].data = f32_to_bf16_rne(src_f[i]);
    }

    hipError_t e = hipMemcpyAsync(dst, converted.data(), n * sizeof(bf16_t), hipMemcpyHostToDevice, stream);
    if (e != hipSuccess) {
        return Status::hip_error("hipMemcpyAsync", hipGetErrorString(e), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Result<MatrixWeight> load_bf16_matrix_raw(
    const SafetensorsCollection& reader,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream)
{
    auto spec_result = reader.tensor_spec(name);
    if (!spec_result.ok()) return spec_result.status();
    const auto& spec = spec_result.value();
    const auto shape = spec.shape;
    if (shape.size() != 2) {
        return Status::invalid_argument(
            ("matrix weight must be 2D: " + name).c_str(), __FILE__, __LINE__);
    }

    MatrixWeight w;
    w.encoding = MatrixEncoding::Bf16;
    w.rows = static_cast<uint32_t>(shape[0]);
    w.cols = static_cast<uint32_t>(shape[1]);
    w.k_padded = w.cols;

    std::vector<std::size_t> strides = {shape[1], 1};
    w.bf16 = alloc_tensor(shape, strides, arena);
    if (w.bf16.ndim() == 0) {
        return Status::insufficient_memory(
            ("matrix allocation failed: " + name).c_str(), __FILE__, __LINE__);
    }
    if (!upload_tensor_direct(reader, name, w.bf16.data<void>(), stream).ok()) {
        return Status::invalid_state(
            ("matrix upload failed: " + name).c_str(), __FILE__, __LINE__);
    }
    return w;
}

Result<MatrixWeight> load_quantized_matrix_impl(
    const QuantizedModelReader& reader,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const WeightLoadOptions& options)
{
    auto view_result = reader.resolve(name);
    if (!view_result.ok()) return view_result.status();

    OwnedCanonicalTensor partitioned_payload;
    QuantizedTensorView partitioned_view;
    const QuantizedTensorView* resolved = &view_result.value();
    const TensorPartitionDesc* applied_partition = nullptr;
    if (options.partition_plan != nullptr) {
        auto it = options.partition_plan->tensors.find(resolved->name);
        if (it != options.partition_plan->tensors.end()) {
            Status partition_status = materialize_rank_local_canonical(
                it->second, *resolved, partitioned_payload, partitioned_view);
            if (!partition_status.ok()) return partition_status;
            resolved = &partitioned_view;
            applied_partition = &it->second;
        }
    }

    const QuantizedTensorView& v = *resolved;

    if (v.logical_shape.size() != 2) {
        return Status::invalid_argument(
            ("quantized matrix weight must be 2D: " + name).c_str(), __FILE__, __LINE__);
    }
    const uint64_t m = static_cast<uint64_t>(v.logical_shape[0]);
    const uint64_t k = static_cast<uint64_t>(v.logical_shape[1]);
    if (m == 0 || k == 0) {
        return Status::invalid_argument(
            ("quantized matrix weight must be non-empty: " + name).c_str(), __FILE__, __LINE__);
    }
    const uint64_t kp = static_cast<uint64_t>(v.k_padded);

    MatrixWeight w;
    w.rows = static_cast<uint32_t>(m);
    w.cols = static_cast<uint32_t>(k);
    if (applied_partition != nullptr) w.partition = *applied_partition;

    if (v.encoding == QuantizedEncoding::Bf16) {
        if (v.codes.data != nullptr || v.metadata1.data != nullptr) {
            return Status::invalid_argument(
                ("quantized bf16 tensor must not carry codes/scales: " + name).c_str(), __FILE__, __LINE__);
        }
        auto st = expect_bytes(name + " data", v.data, m * k * 2);
        if (!st.ok()) return st;
        w.encoding = MatrixEncoding::Bf16;
        w.k_padded = static_cast<uint32_t>(k);
        std::vector<std::size_t> shape = {m, k};
        std::vector<std::size_t> strides = {k, 1};
        w.bf16 = alloc_bytes(m * k * 2, shape, strides, arena);
        if (w.bf16.ndim() == 0) {
            return Status::insufficient_memory(
                ("quantized matrix allocation failed: " + name).c_str(), __FILE__, __LINE__);
        }
        st = copy_to_device(w.bf16.data<void>(), v.data.data, v.data.size, stream);
        if (!st.ok()) return st;
        apply_default_compute_spec(w);
        return w;
    }

    if (kp < k || kp % 32 != 0) {
        return Status::invalid_argument(
            ("quantized matrix k_padded invalid: " + name).c_str(), __FILE__, __LINE__);
    }
    if (v.data.data != nullptr) {
        return Status::invalid_argument(
            ("quantized tensor must not carry bf16 data: " + name).c_str(), __FILE__, __LINE__);
    }
    w.k_padded = static_cast<uint32_t>(kp);

    if (v.encoding == QuantizedEncoding::Psq4 || v.encoding == QuantizedEncoding::Psq8) {
        const bool psq4 = v.encoding == QuantizedEncoding::Psq4;
        const uint64_t codes_row_bytes = psq4 ? (kp / 2u) : kp;
        auto st = expect_bytes(name + " codes", v.codes, m * codes_row_bytes);
        if (!st.ok()) return st;
        st = expect_bytes(name + " scales", v.metadata1, m * kp / 16u);
        if (!st.ok()) return st;

        quantization::psq::CanonicalQuantView cv;
        cv.format_id = psq4 ? quantization::QuantFormatId::Psq4 : quantization::QuantFormatId::Psq8;
        cv.desc = quantization::quant_format_desc(cv.format_id);
        cv.rows = m;
        cv.padded_k = kp;
        cv.codes = static_cast<const uint8_t*>(v.codes.data);
        cv.codes_bytes = v.codes.size;
        cv.metadata1 = static_cast<const uint8_t*>(v.metadata1.data);
        cv.metadata1_bytes = v.metadata1.size;
        if (!cv.validate()) {
            return Status::invalid_argument(
                ("quantized matrix canonical payload invalid: " + name).c_str(), __FILE__, __LINE__);
        }
        w.encoding = psq4 ? MatrixEncoding::Psq4 : MatrixEncoding::Psq8;
        if (!options.preshuffle) {
            std::vector<std::size_t> cshape = {m, codes_row_bytes};
            std::vector<std::size_t> cstrides = {codes_row_bytes, 1};
            w.codes = alloc_bytes(v.codes.size, cshape, cstrides, arena);
            if (w.codes.ndim() == 0) {
                return Status::insufficient_memory(
                    ("quantized matrix codes allocation failed: " + name).c_str(), __FILE__, __LINE__);
            }
            st = copy_to_device(w.codes.data<void>(), v.codes.data, v.codes.size, stream);
            if (!st.ok()) return st;
            const uint64_t scale_row_bytes = kp / 16u;
            std::vector<std::size_t> sshape = {m, scale_row_bytes};
            std::vector<std::size_t> sstrides = {scale_row_bytes, 1};
            w.scales = alloc_bytes(v.metadata1.size, sshape, sstrides, arena);
            if (w.scales.ndim() == 0) {
                return Status::insufficient_memory(
                    ("quantized matrix scales allocation failed: " + name).c_str(), __FILE__, __LINE__);
            }
            st = copy_to_device(w.scales.data<void>(), v.metadata1.data, v.metadata1.size, stream);
            if (!st.ok()) return st;
            w.storage_scale_stride_bytes = static_cast<uint32_t>(scale_row_bytes);
            w.weight_scale_group = 32;
            w.codes_row_stride_bytes = static_cast<uint32_t>(codes_row_bytes);
            apply_default_compute_spec(w);
            return w;
        }
        quantization::psq::NativeQuantHost native;
        if (!quantization::psq::preshuffle_native(cv, native)) {
            return Status::invalid_argument(
                ("quantized matrix format not supported for arch: " + name).c_str(), __FILE__, __LINE__);
        }
        std::vector<std::size_t> cshape = {native.rows_padded / 16u, native.codes_row_stride_bytes};
        std::vector<std::size_t> cstrides = {native.codes_row_stride_bytes, 1};
        w.codes = alloc_bytes(native.codes.size(), cshape, cstrides, arena);
        if (w.codes.ndim() == 0) {
            return Status::insufficient_memory(
                ("quantized matrix codes allocation failed: " + name).c_str(), __FILE__, __LINE__);
        }
        st = copy_to_device(w.codes.data<void>(), native.codes.data(), native.codes.size(), stream);
        if (!st.ok()) return st;
        std::vector<std::size_t> sshape = {native.rows_padded / 16u, native.metadata1_stride_bytes};
        std::vector<std::size_t> sstrides = {native.metadata1_stride_bytes, 1};
        w.scales = alloc_bytes(native.metadata1.size(), sshape, sstrides, arena);
        if (w.scales.ndim() == 0) {
            return Status::insufficient_memory(
                ("quantized matrix scales allocation failed: " + name).c_str(), __FILE__, __LINE__);
        }
        st = copy_to_device(w.scales.data<void>(), native.metadata1.data(), native.metadata1.size(), stream);
        if (!st.ok()) return st;
        w.storage_scale_stride_bytes = native.metadata1_stride_bytes;
        w.weight_scale_group = 32;
        w.codes_row_stride_bytes = native.codes_row_stride_bytes;
        w.preshuffled = true;
        apply_default_compute_spec(w);
        return w;
    }

    if (v.encoding == QuantizedEncoding::Fp8Block128) {
        if ((kp % 128u) != 0u) {
            return Status::invalid_argument(
                ("fp8 block128 k_padded must be a multiple of 128: " + name).c_str(),
                __FILE__, __LINE__);
        }
        const uint64_t scale_n = (m + 127u) / 128u;
        const uint64_t scale_k = kp / 128u;
        auto st = expect_bytes(name + " codes", v.codes, m * kp);
        if (!st.ok()) return st;
        st = expect_bytes(name + " scales", v.metadata1, scale_n * scale_k * sizeof(float));
        if (!st.ok()) return st;

        quantization::fp8::Fp8BlockCanonicalView cv;
        cv.batch = 1;
        cv.n = m;
        cv.padded_k = kp;
        cv.codes = static_cast<const uint8_t*>(v.codes.data);
        cv.codes_bytes = v.codes.size;
        cv.scales = reinterpret_cast<const float*>(v.metadata1.data);
        cv.scales_bytes = v.metadata1.size;
        cv.scale_n = scale_n;
        cv.scale_k = scale_k;
        if (!cv.validate()) {
            return Status::invalid_argument(
                ("fp8 block128 canonical payload invalid: " + name).c_str(), __FILE__, __LINE__);
        }
        w.encoding = MatrixEncoding::Fp8Block128;
        w.weight_scale_group = 128;

        if (!options.preshuffle) {
            std::vector<std::size_t> cshape = {m, kp};
            std::vector<std::size_t> cstrides = {kp, 1};
            w.codes = alloc_bytes(v.codes.size, cshape, cstrides, arena);
            if (w.codes.ndim() == 0) {
                return Status::insufficient_memory(
                    ("quantized matrix codes allocation failed: " + name).c_str(), __FILE__, __LINE__);
            }
            st = copy_to_device(w.codes.data<void>(), v.codes.data, v.codes.size, stream);
            if (!st.ok()) return st;
            std::vector<std::size_t> sshape = {scale_n, scale_k};
            std::vector<std::size_t> sstrides = {scale_k, 1};
            w.scales = alloc_tensor_f32(sshape, sstrides, arena);
            if (w.scales.ndim() == 0) {
                return Status::insufficient_memory(
                    ("quantized matrix scales allocation failed: " + name).c_str(), __FILE__, __LINE__);
            }
            st = copy_to_device(w.scales.data<void>(), v.metadata1.data, v.metadata1.size, stream);
            if (!st.ok()) return st;
            w.storage_scale_stride_bytes = static_cast<uint32_t>(scale_k * sizeof(float));
            w.codes_row_stride_bytes = static_cast<uint32_t>(kp);
            w.scale_row_stride_bytes = static_cast<uint32_t>(scale_k * sizeof(float));
            apply_default_compute_spec(w);
            return w;
        }

        quantization::fp8::Fp8Block128NativeHost native;
        if (!quantization::fp8::preshuffle_fp8_block128_native(cv, native)) {
            return Status::invalid_argument(
                ("fp8 block128 payload not supported for arch: " + name).c_str(), __FILE__, __LINE__);
        }
        const uint64_t nb = kp / 32u;
        std::vector<std::size_t> cshape = {native.rows / 16u, native.codes_row_stride_bytes};
        std::vector<std::size_t> cstrides = {native.codes_row_stride_bytes, 1};
        w.codes = alloc_bytes(native.codes.size(), cshape, cstrides, arena);
        if (w.codes.ndim() == 0) {
            return Status::insufficient_memory(
                ("quantized matrix codes allocation failed: " + name).c_str(), __FILE__, __LINE__);
        }
        st = copy_to_device(w.codes.data<void>(), native.codes.data(), native.codes.size(), stream);
        if (!st.ok()) return st;
        std::vector<std::size_t> sshape = {native.scale_n, native.scale_k};
        std::vector<std::size_t> sstrides = {native.scale_k, 1};
        w.scales = alloc_tensor_f32(sshape, sstrides, arena);
        if (w.scales.ndim() == 0) {
            return Status::insufficient_memory(
                ("quantized matrix scales allocation failed: " + name).c_str(), __FILE__, __LINE__);
        }
        st = copy_to_device(w.scales.data<void>(), native.scales.data(),
                            native.scales.size() * sizeof(float), stream);
        if (!st.ok()) return st;
        w.storage_scale_stride_bytes = static_cast<uint32_t>(native.scale_k * sizeof(float));
        w.codes_row_stride_bytes = static_cast<uint32_t>(nb * 512u);
        w.scale_row_stride_bytes = static_cast<uint32_t>(native.scale_k * sizeof(float));
        w.preshuffled = true;
        apply_default_compute_spec(w);
        return w;
    }

    if (v.encoding == QuantizedEncoding::Mxfp4) {
        const uint64_t nb = kp / 32u;
        auto st = expect_bytes(name + " codes", v.codes, m * nb * 16u);
        if (!st.ok()) return st;
        st = expect_bytes(name + " scales", v.metadata1, m * nb);
        if (!st.ok()) return st;

        quantization::mxfp4::Mxfp4CanonicalView cv;
        cv.rows = m;
        cv.padded_k = kp;
        cv.codes = static_cast<const uint8_t*>(v.codes.data);
        cv.codes_bytes = v.codes.size;
        cv.scales = static_cast<const uint8_t*>(v.metadata1.data);
        cv.scales_bytes = v.metadata1.size;
        if (!cv.validate()) {
            return Status::invalid_argument(
                ("mxfp4 canonical payload invalid: " + name).c_str(), __FILE__, __LINE__);
        }
        w.encoding = MatrixEncoding::Mxfp4;
        w.weight_scale_group = 32;

        if (!options.preshuffle) {
            std::vector<std::size_t> cshape = {m, nb * 16u};
            std::vector<std::size_t> cstrides = {nb * 16u, 1};
            w.codes = alloc_bytes(v.codes.size, cshape, cstrides, arena);
            if (w.codes.ndim() == 0) {
                return Status::insufficient_memory(
                    ("quantized matrix codes allocation failed: " + name).c_str(), __FILE__, __LINE__);
            }
            st = copy_to_device(w.codes.data<void>(), v.codes.data, v.codes.size, stream);
            if (!st.ok()) return st;
            std::vector<std::size_t> sshape = {m, nb};
            std::vector<std::size_t> sstrides = {nb, 1};
            w.scales = alloc_bytes(v.metadata1.size, sshape, sstrides, arena);
            if (w.scales.ndim() == 0) {
                return Status::insufficient_memory(
                    ("quantized matrix scales allocation failed: " + name).c_str(), __FILE__, __LINE__);
            }
            st = copy_to_device(w.scales.data<void>(), v.metadata1.data, v.metadata1.size, stream);
            if (!st.ok()) return st;
            w.storage_scale_stride_bytes = static_cast<uint32_t>(nb);
            w.codes_row_stride_bytes = static_cast<uint32_t>(nb * 16u);
            w.scale_row_stride_bytes = static_cast<uint32_t>(nb);
            apply_default_compute_spec(w);
            return w;
        }

        quantization::mxfp4::Mxfp4NativeHost native;
        if (!quantization::mxfp4::preshuffle_mxfp4_native(cv, native)) {
            return Status::invalid_argument(
                ("mxfp4 payload not supported for arch: " + name).c_str(), __FILE__, __LINE__);
        }
        std::vector<std::size_t> cshape = {native.rows / 16u, native.codes_row_stride_bytes};
        std::vector<std::size_t> cstrides = {native.codes_row_stride_bytes, 1};
        w.codes = alloc_bytes(native.codes.size(), cshape, cstrides, arena);
        if (w.codes.ndim() == 0) {
            return Status::insufficient_memory(
                ("quantized matrix codes allocation failed: " + name).c_str(), __FILE__, __LINE__);
        }
        st = copy_to_device(w.codes.data<void>(), native.codes.data(), native.codes.size(), stream);
        if (!st.ok()) return st;
        std::vector<std::size_t> sshape = {m, nb};
        std::vector<std::size_t> sstrides = {nb, 1};
        w.scales = alloc_bytes(native.scales.size(), sshape, sstrides, arena);
        if (w.scales.ndim() == 0) {
            return Status::insufficient_memory(
                ("quantized matrix scales allocation failed: " + name).c_str(), __FILE__, __LINE__);
        }
        st = copy_to_device(w.scales.data<void>(), native.scales.data(), native.scales.size(), stream);
        if (!st.ok()) return st;
        w.storage_scale_stride_bytes = static_cast<uint32_t>(nb);
        w.codes_row_stride_bytes = static_cast<uint32_t>(nb * 256u);
        w.scale_row_stride_bytes = static_cast<uint32_t>(nb);
        w.preshuffled = true;
        apply_default_compute_spec(w);
        return w;
    }

    return Status::invalid_argument(
        ("quantized matrix unknown encoding: " + name).c_str(), __FILE__, __LINE__);
}

}

Result<SafetensorsCollection> SafetensorsCollection::open(const std::string& model_dir) {
    SafetensorsCollection col;
    col.model_dir_ = model_dir;

    std::string index_path = model_dir + "/model.safetensors.index.json";
    std::ifstream index_f(index_path);
    bool has_index = index_f.is_open();
    if (has_index) {
        nlohmann::json index_json;
        try {
            index_f >> index_json;
        } catch (const nlohmann::json::exception& e) {
            return Status::invalid_argument(
                ("bad safetensors index: " + std::string(e.what())).c_str(),
                __FILE__, __LINE__);
        }
        if (!index_json.contains("weight_map") || !index_json["weight_map"].is_object()) {
            return Status::invalid_argument(
                "bad safetensors index: missing weight_map", __FILE__, __LINE__);
        }
        for (auto it = index_json["weight_map"].begin();
             it != index_json["weight_map"].end(); ++it) {
            col.weight_map_[it.key()] = it.value().get<std::string>();
        }
    } else {
        const fs::path single = fs::path(model_dir) / "model.safetensors";
        if (fs::exists(single)) {
            col.single_shard_ = "model.safetensors";
        }
    }

    return col;
}

Result<io::StTensorSpec> SafetensorsCollection::tensor_spec(const std::string& name) const {
    auto reader_result = reader_for(name);
    if (!reader_result.ok()) return reader_result.status();
    return reader_result.value()->tensor_spec(name);
}

Result<const void*> SafetensorsCollection::tensor_data(const std::string& name, std::size_t& out_bytes) const {
    auto reader_result = reader_for(name);
    if (!reader_result.ok()) return reader_result.status();
    return reader_result.value()->tensor_data(name, out_bytes);
}

std::string SafetensorsCollection::shard_for(const std::string& name) const {
    if (weight_map_.empty()) {
        if (!single_shard_.empty()) return single_shard_;
        return "model.safetensors-00001-of-00001.safetensors";
    }
    auto it = weight_map_.find(name);
    if (it == weight_map_.end()) return "";
    return it->second;
}

Result<const io::SafetensorsReader*> SafetensorsCollection::reader_for(const std::string& name) const {
    std::string shard = shard_for(name);
    if (shard.empty()) {
        return Status::invalid_argument(
            ("tensor not found: " + name).c_str(), __FILE__, __LINE__);
    }
    return open_shard(shard);
}

Result<const io::SafetensorsReader*> SafetensorsCollection::open_shard(const std::string& shard) const {
    auto existing = shard_readers_.find(shard);
    if (existing != shard_readers_.end()) {
        return &existing->second;
    }

    std::string path = model_dir_ + "/" + shard;
    auto reader_result = io::SafetensorsReader::open(path);
    if (!reader_result.ok()) return reader_result.status();

    auto inserted = shard_readers_.emplace(shard, reader_result.release());
    return &inserted.first->second;
}

Result<std::vector<std::string>> SafetensorsCollection::tensor_names() const {
    std::vector<std::string> shards;
    if (weight_map_.empty()) {
        shards.push_back(single_shard_.empty()
                             ? "model.safetensors-00001-of-00001.safetensors"
                             : single_shard_);
    } else {
        std::map<std::string, bool> seen;
        for (const auto& kv : weight_map_) seen[kv.second] = true;
        shards.reserve(seen.size());
        for (const auto& kv : seen) shards.push_back(kv.first);
    }

    std::vector<std::string> names;
    for (const auto& shard : shards) {
        auto reader_result = open_shard(shard);
        if (!reader_result.ok()) return reader_result.status();
        auto list = reader_result.value()->list_tensors();
        if (!list.ok()) return list.status();
        for (auto& n : list.value()) names.push_back(std::move(n));
    }
    return names;
}

bool is_quantized_model_dir(const std::string& model_dir) {
    const fs::path dir(model_dir);
    if (fs::exists(dir / quantization::fpx::kQuantizedMetadataFile)) {
        return true;
    }
    const fs::path config_path = dir / "config.json";
    if (fs::exists(config_path)) {
        std::ifstream cf(config_path, std::ios::binary);
        if (cf) {
            nlohmann::json config;
            try {
                cf >> config;
            } catch (const nlohmann::json::exception&) {
                return false;
            }
            if (config.is_object() && config.contains("phaseshift_quantization")) {
                return true;
            }
        }
    }
    return false;
}

Status validate_quantized_model(const QuantizedModelReader& reader) {
    using quantization::fpx::QuantizedManifest;
    using quantization::fpx::validate_quantized_manifest_contract;

    const QuantizedManifest& manifest = reader.manifest();

    Status contract = validate_quantized_manifest_contract(manifest);
    if (!contract.ok()) {
        return contract;
    }

    for (const auto& kv : manifest.aliases) {
        auto target = reader.resolve_alias(kv.first);
        if (!target.ok()) {
            return target.status();
        }
        if (manifest.tensors.find(target.value()) == manifest.tensors.end()) {
            return Status::invalid_argument(
                ("quantized manifest alias target missing: " + target.value()).c_str(),
                __FILE__, __LINE__);
        }
    }

    return Status::make_ok();
}

Result<Tensor> load_bf16_tensor(
    const SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream)
{
    auto spec_result = collection.tensor_spec(name);
    if (!spec_result.ok()) return spec_result.status();
    const auto& spec = spec_result.value();
    const auto shape = spec.shape;
    const auto strides = contiguous_strides(shape);

    std::size_t n = 1;
    for (auto d : shape) n *= d;
    Tensor t = alloc_tensor(shape, strides, arena);
    if (t.ndim() == 0) {
        return Status::insufficient_memory(
            ("tensor allocation failed: " + name).c_str(), __FILE__, __LINE__);
    }
    Status st = (spec.dtype == io::SType::F32)
        ? upload_tensor_f32_to_bf16(collection, name, t.data<void>(), n, stream)
        : upload_tensor_direct(collection, name, t.data<void>(), stream);
    if (!st.ok()) {
        return Status::invalid_state(
            ("tensor upload failed: " + name).c_str(), __FILE__, __LINE__);
    }
    return t;
}

Result<Tensor> load_bf16_or_f32_tensor(
    const SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream)
{
    auto spec_result = collection.tensor_spec(name);
    if (!spec_result.ok()) return spec_result.status();
    const auto& spec = spec_result.value();
    const auto shape = spec.shape;
    const auto strides = contiguous_strides(shape);

    Tensor t = (spec.dtype == io::SType::BF16 || spec.dtype == io::SType::F16)
        ? alloc_tensor(shape, strides, arena)
        : alloc_tensor_f32(shape, strides, arena);
    if (t.ndim() == 0) {
        return Status::insufficient_memory(
            ("tensor allocation failed: " + name).c_str(), __FILE__, __LINE__);
    }
    Status st = upload_tensor_direct(collection, name, t.data<void>(), stream);
    if (!st.ok()) {
        return Status::invalid_state(
            ("tensor upload failed: " + name).c_str(), __FILE__, __LINE__);
    }
    return t;
}

Result<MatrixWeight> load_bf16_matrix(
    const SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const WeightLoadOptions& options)
{
    if (options.partition_plan != nullptr) {
        return Status::unsupported(
            "tensor partition is not supported for raw safetensors matrix load",
            __FILE__, __LINE__);
    }
    (void)options;
    return load_bf16_matrix_raw(collection, name, arena, stream);
}

Result<MatrixWeight> load_quantized_matrix(
    const QuantizedModelReader& reader,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const WeightLoadOptions& options)
{
    return load_quantized_matrix_impl(reader, name, arena, stream, options);
}

Result<gpu::Tensor> load_quantized_small(
    const QuantizedModelReader& reader,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream)
{
    auto view_result = reader.resolve(name);
    if (!view_result.ok()) return view_result.status();
    const QuantizedTensorView& v = view_result.value();

    if (v.encoding != QuantizedEncoding::Bf16) {
        return Status::invalid_argument(
            ("non-matrix tensor must be bf16: " + name).c_str(), __FILE__, __LINE__);
    }
    if (v.codes.data != nullptr || v.metadata1.data != nullptr) {
        return Status::invalid_argument(
            ("non-matrix tensor must not carry codes/scales: " + name).c_str(), __FILE__, __LINE__);
    }
    if (v.logical_shape.size() < 1 || v.logical_shape.size() > 3) {
        return Status::invalid_argument(
            ("non-matrix tensor ndim unsupported: " + name).c_str(), __FILE__, __LINE__);
    }

    std::size_t n = 1;
    for (auto d : v.logical_shape) {
        if (d <= 0) {
            return Status::invalid_argument(
                ("non-matrix tensor shape invalid: " + name).c_str(), __FILE__, __LINE__);
        }
        n *= static_cast<std::size_t>(d);
    }
    auto st = expect_bytes(name + " data", v.data, n * 2);
    if (!st.ok()) return st;

    std::vector<std::size_t> shape;
    shape.reserve(v.logical_shape.size());
    for (auto d : v.logical_shape) shape.push_back(static_cast<std::size_t>(d));
    std::vector<std::size_t> strides = contiguous_strides(shape);

    gpu::Tensor t = alloc_bytes16(n * 2, shape, strides, arena);
    if (t.ndim() == 0) {
        return Status::insufficient_memory(
            ("non-matrix tensor allocation failed: " + name).c_str(), __FILE__, __LINE__);
    }
    st = copy_to_device(t.data<void>(), v.data.data, v.data.size, stream);
    if (!st.ok()) return st;
    return t;
}

}  // namespace ps::weights
