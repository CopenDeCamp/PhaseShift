#include <phaseshift/quantization/fpx/quantized_model_reader.h>
#include <phaseshift/quantization/mxfp4/mxfp4.h>
#include <phaseshift/quantization/psq/quant_canonical.h>
#include <phaseshift/quantization/psq/quant_preshuffle.h>
#include <phaseshift/weights/canonical_partition.h>
#include <phaseshift/weights/tensor_partition.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <vector>

using ps::Status;
using ps::weights::materialize_rank_local_canonical;
using ps::weights::OwnedCanonicalTensor;
using ps::weights::TensorPartitionDesc;
using ps::weights::TensorPartitionRange;
using ps::weights::tensor_partition_local_shape;
using ps::weights::validate_tensor_partition;
using namespace ps::quantization;
using namespace ps::quantization::fpx;
using namespace ps::quantization::psq;

static int g_fail = 0;

static void fail(const std::string& msg) {
    g_fail++;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static TensorPartitionRange rng(uint64_t offset, uint64_t extent) {
    TensorPartitionRange r;
    r.global_offset = offset;
    r.extent = extent;
    return r;
}

static TensorPartitionDesc desc_of(int32_t axis, std::vector<int64_t> shape,
                                   uint32_t index, uint32_t count,
                                   std::vector<TensorPartitionRange> ranges) {
    TensorPartitionDesc d;
    d.axis = axis;
    d.index = index;
    d.count = count;
    d.global_shape = std::move(shape);
    d.ranges = std::move(ranges);
    return d;
}

static QuantFormatId format_of(QuantizedEncoding enc) {
    switch (enc) {
        case QuantizedEncoding::Psq4: return QuantFormatId::Psq4;
        case QuantizedEncoding::Psq8: return QuantFormatId::Psq8;
        case QuantizedEncoding::Mxfp4: return QuantFormatId::Mxfp4;
        case QuantizedEncoding::Fp8Block128: return QuantFormatId::Fp8Block128;
        default: return QuantFormatId::None;
    }
}

static uint64_t rows_of(const std::vector<int64_t>& shape) {
    uint64_t rows = 1;
    for (std::size_t i = 0; i + 1 < shape.size(); ++i) {
        rows *= static_cast<uint64_t>(shape[i]);
    }
    return rows;
}

static QuantizedTensorView make_quant_view(const CanonicalQuantStore& store,
                                           QuantizedEncoding enc,
                                           std::vector<int64_t> shape) {
    QuantizedTensorView v;
    v.name = "w";
    v.encoding = enc;
    v.logical_shape = std::move(shape);
    v.k_padded = static_cast<int64_t>(store.padded_k);
    v.codes = ByteSpan{store.codes.data(), store.codes.size()};
    v.metadata1 = ByteSpan{store.metadata1.data(), store.metadata1.size()};
    return v;
}

static QuantizedTensorView make_bf16_view(const std::vector<uint8_t>& bytes,
                                          std::vector<int64_t> shape) {
    QuantizedTensorView v;
    v.name = "w";
    v.encoding = QuantizedEncoding::Bf16;
    v.logical_shape = std::move(shape);
    v.k_padded = static_cast<int64_t>(v.logical_shape.back());
    v.data = ByteSpan{bytes.data(), bytes.size()};
    return v;
}

static CanonicalQuantView to_canonical(const QuantizedTensorView& v) {
    CanonicalQuantView cv;
    cv.format_id = format_of(v.encoding);
    cv.desc = quant_format_desc(cv.format_id);
    cv.rows = rows_of(v.logical_shape);
    cv.padded_k = static_cast<uint64_t>(v.k_padded);
    cv.codes = static_cast<const uint8_t*>(v.codes.data);
    cv.codes_bytes = v.codes.size;
    cv.metadata1 = static_cast<const uint8_t*>(v.metadata1.data);
    cv.metadata1_bytes = v.metadata1.size;
    cv.metadata2 = static_cast<const uint8_t*>(v.metadata2.data);
    cv.metadata2_bytes = v.metadata2.size;
    cv.metadata3 = static_cast<const uint8_t*>(v.metadata3.data);
    cv.metadata3_bytes = v.metadata3.size;
    cv.metadata4 = static_cast<const uint8_t*>(v.metadata4.data);
    cv.metadata4_bytes = v.metadata4.size;
    return cv;
}

static void check_bytes(const std::string& tag, const std::vector<uint8_t>& got,
                        const std::vector<uint8_t>& want) {
    if (got.size() != want.size()) {
        fail(tag + " size " + std::to_string(got.size()) + " vs " + std::to_string(want.size()));
        return;
    }
    if (!got.empty() && std::memcmp(got.data(), want.data(), got.size()) != 0) {
        fail(tag + " bytes differ");
    }
}

static void check_bytes_span(const std::string& tag, const std::vector<uint8_t>& got,
                             const uint8_t* want, std::size_t want_size) {
    if (got.size() != want_size) {
        fail(tag + " size " + std::to_string(got.size()) + " vs " + std::to_string(want_size));
        return;
    }
    if (!got.empty() && std::memcmp(got.data(), want, got.size()) != 0) {
        fail(tag + " bytes differ");
    }
}

static void check_floats(const std::string& tag, const float* got, const float* want,
                         std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0) {
            fail(tag + " at " + std::to_string(i));
            return;
        }
    }
}

static std::vector<float> make_source(uint64_t rows, uint64_t k, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    std::vector<float> src(rows * k);
    for (float& f : src) f = dist(rng);
    return src;
}

static CanonicalQuantStore quantize_rows(QuantFormatId id, const float* src,
                                         uint64_t rows, uint64_t k,
                                         uint64_t src_stride_k, uint64_t src_col_begin) {
    CanonicalQuantStore store;
    store.init(id, rows, k, k);
    std::vector<float> row(k);
    for (uint64_t r = 0; r < rows; ++r) {
        std::memcpy(row.data(), src + r * src_stride_k + src_col_begin, k * sizeof(float));
        if (!store.quantize_row(static_cast<uint32_t>(r), std::span<const float>(row.data(), k))) {
            fail("reference quantize_row");
        }
    }
    return store;
}

static void check_preshuffle_equivalent(const std::string& tag,
                                        const QuantizedTensorView& materialized,
                                        const CanonicalQuantStore& reference) {
    NativeQuantHost got;
    NativeQuantHost want;
    if (!preshuffle_native(to_canonical(materialized), got)) {
        fail(tag + " preshuffle materialized");
        return;
    }
    if (!preshuffle_native(reference.view(), want)) {
        fail(tag + " preshuffle reference");
        return;
    }
    check(got.rows == want.rows, tag + " native rows match");
    check_bytes(tag + " native codes", got.codes, want.codes);
    check_bytes(tag + " native metadata1", got.metadata1, want.metadata1);
    check_bytes(tag + " native metadata2", got.metadata2, want.metadata2);
    check_bytes(tag + " native metadata3", got.metadata3, want.metadata3);
    check_bytes(tag + " native metadata4", got.metadata4, want.metadata4);
}

static CanonicalQuantStore build_global(QuantFormatId id, const std::vector<float>& src,
                                        uint64_t rows, uint64_t k) {
    CanonicalQuantStore store;
    store.init(id, rows, k, k);
    for (uint64_t r = 0; r < rows; ++r) {
        if (!store.quantize_row(static_cast<uint32_t>(r),
                                std::span<const float>(src.data() + r * k, k))) {
            fail("global quantize_row");
        }
    }
    if (!store.view().validate()) fail("global canonical validate");
    return store;
}

static void test_bf16_partitions() {
    {
        const uint64_t rows = 4, k = 8;
        std::vector<uint8_t> bytes(rows * k * 2);
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<uint8_t>(i * 7 + 3);
        }
        QuantizedTensorView g = make_bf16_view(bytes, {4, 8});

        OwnedCanonicalTensor owned;
        QuantizedTensorView local;
        Status st = materialize_rank_local_canonical(
            desc_of(0, {4, 8}, 0, 2, {rng(0, 2)}), g, owned, local);
        check(st.ok(), "bf16 axis0 materialize");
        if (st.ok()) {
            check(local.logical_shape == std::vector<int64_t>({2, 8}), "bf16 axis0 local shape");
            check(local.k_padded == 8, "bf16 axis0 local k_padded");
            const std::vector<uint8_t> want(bytes.begin(), bytes.begin() + 2 * k * 2);
            check_bytes("bf16 axis0 rank0", owned.data, want);
        }

        st = materialize_rank_local_canonical(
            desc_of(0, {4, 8}, 1, 2, {rng(2, 2)}), g, owned, local);
        check(st.ok(), "bf16 axis0 rank1 materialize");
        if (st.ok()) {
            const std::vector<uint8_t> want(bytes.begin() + 2 * k * 2, bytes.end());
            check_bytes("bf16 axis0 rank1", owned.data, want);
        }

        st = materialize_rank_local_canonical(
            desc_of(1, {4, 8}, 1, 2, {rng(3, 5)}), g, owned, local);
        check(st.ok(), "bf16 axis1 materialize");
        if (st.ok()) {
            check(local.logical_shape == std::vector<int64_t>({4, 5}), "bf16 axis1 local shape");
            std::vector<uint8_t> want;
            for (uint64_t r = 0; r < rows; ++r) {
                const uint8_t* row = bytes.data() + r * k * 2 + 3 * 2;
                want.insert(want.end(), row, row + 5 * 2);
            }
            check_bytes("bf16 axis1 slice", owned.data, want);
        }
    }
    {
        std::vector<uint8_t> bytes(16 * 2);
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<uint8_t>(i * 11 + 5);
        }
        QuantizedTensorView g = make_bf16_view(bytes, {16});
        OwnedCanonicalTensor owned;
        QuantizedTensorView local;
        Status st = materialize_rank_local_canonical(
            desc_of(0, {16}, 1, 2, {rng(4, 4)}), g, owned, local);
        check(st.ok(), "bf16 1D materialize");
        if (st.ok()) {
            check(local.logical_shape == std::vector<int64_t>({4}), "bf16 1D local shape");
            const std::vector<uint8_t> want(bytes.begin() + 8, bytes.begin() + 16);
            check_bytes("bf16 1D slice", owned.data, want);
        }
    }
    {
        const uint64_t e = 2, n = 3, k = 4;
        std::vector<uint8_t> bytes(e * n * k * 2);
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<uint8_t>(i * 13 + 1);
        }
        QuantizedTensorView g = make_bf16_view(bytes, {2, 3, 4});
        OwnedCanonicalTensor owned;
        QuantizedTensorView local;
        Status st = materialize_rank_local_canonical(
            desc_of(0, {2, 3, 4}, 0, 2, {rng(1, 1)}), g, owned, local);
        check(st.ok(), "bf16 3D axis0 materialize");
        if (st.ok()) {
            check(local.logical_shape == std::vector<int64_t>({1, 3, 4}),
                  "bf16 3D local shape");
            const std::size_t row_bytes = n * k * 2;
            const std::vector<uint8_t> want(bytes.begin() + row_bytes,
                                            bytes.begin() + 2 * row_bytes);
            check_bytes("bf16 3D row slice", owned.data, want);
        }
    }
    {
        const uint64_t rows = 6, k = 4;
        std::vector<uint8_t> bytes(rows * k * 2);
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<uint8_t>(i);
        }
        QuantizedTensorView g = make_bf16_view(bytes, {6, 4});
        OwnedCanonicalTensor owned;
        QuantizedTensorView local;
        Status st = materialize_rank_local_canonical(
            desc_of(0, {6, 4}, 0, 2, {rng(0, 2), rng(4, 2)}), g, owned, local);
        check(st.ok(), "bf16 segmented materialize");
        if (st.ok()) {
            check(local.logical_shape == std::vector<int64_t>({4, 4}),
                  "bf16 segmented local shape");
            std::vector<uint8_t> want;
            const std::size_t row_bytes = k * 2;
            want.insert(want.end(), bytes.begin(), bytes.begin() + 2 * row_bytes);
            want.insert(want.end(), bytes.begin() + 4 * row_bytes, bytes.end());
            check_bytes("bf16 segmented slice", owned.data, want);
        }
    }
}

static void test_psq_n_partition(QuantFormatId id, QuantizedEncoding enc,
                                 const std::string& tag) {
    const uint64_t rows = 32, k = 64;
    const std::vector<float> src = make_source(rows, k, 1000u + static_cast<uint32_t>(id));
    CanonicalQuantStore global_store = build_global(id, src, rows, k);
    QuantizedTensorView g = make_quant_view(global_store, enc, {32, 64});
    const uint64_t codes_row = (k / 32u) * quant_format_desc(id)->code_bytes_per_block;
    const uint64_t scale_row = (k / 32u) * quant_format_desc(id)->meta[0].element_bytes;

    for (uint32_t rank = 0; rank < 2; ++rank) {
        const uint64_t local_rows = 16;
        OwnedCanonicalTensor owned;
        QuantizedTensorView local;
        Status st = materialize_rank_local_canonical(
            desc_of(0, {32, 64}, rank, 2, {rng(rank * local_rows, local_rows)}),
            g, owned, local);
        check(st.ok(), tag + " N rank materialize");
        if (!st.ok()) continue;
        check(local.logical_shape == std::vector<int64_t>({16, 64}),
              tag + " N local logical shape");
        check(local.k_padded == 64, tag + " N local k_padded");

        check_bytes_span(tag + " N rank" + std::to_string(rank) + " codes", owned.codes,
                         global_store.codes.data() + rank * local_rows * codes_row,
                         local_rows * codes_row);
        check_bytes_span(tag + " N rank" + std::to_string(rank) + " metadata1",
                         owned.metadata1,
                         global_store.metadata1.data() + rank * local_rows * scale_row,
                         local_rows * scale_row);

        std::vector<float> global_deq;
        dequantize_canonical(global_store.view(), global_deq);
        std::vector<float> local_deq;
        dequantize_canonical(to_canonical(local), local_deq);
        check(local_deq.size() == local_rows * k, tag + " N dequant size");
        check_floats(tag + " N dequant rank" + std::to_string(rank), local_deq.data(),
                     global_deq.data() + rank * local_rows * k, local_rows * k);

        CanonicalQuantStore reference = quantize_rows(
            id, src.data() + rank * local_rows * k, local_rows, k, k, 0);
        check_preshuffle_equivalent(tag + " N rank" + std::to_string(rank), local,
                                    reference);
    }
}

static void test_psq_k_partition(QuantFormatId id, QuantizedEncoding enc,
                                 const std::string& tag) {
    const uint64_t rows = 16, k = 128;
    const std::vector<float> src = make_source(rows, k, 5000u + static_cast<uint32_t>(id));
    CanonicalQuantStore global_store = build_global(id, src, rows, k);
    QuantizedTensorView g = make_quant_view(global_store, enc, {16, 128});
    const uint64_t local_k = 64;

    for (uint32_t rank = 0; rank < 2; ++rank) {
        OwnedCanonicalTensor owned;
        QuantizedTensorView local;
        Status st = materialize_rank_local_canonical(
            desc_of(1, {16, 128}, rank, 2, {rng(rank * local_k, local_k)}),
            g, owned, local);
        check(st.ok(), tag + " K rank materialize");
        if (!st.ok()) continue;
        check(local.logical_shape == std::vector<int64_t>({16, 64}),
              tag + " K local logical shape");
        check(local.k_padded == 64, tag + " K local k_padded");

        std::vector<float> global_deq;
        dequantize_canonical(global_store.view(), global_deq);
        std::vector<float> local_deq;
        dequantize_canonical(to_canonical(local), local_deq);
        check(local_deq.size() == rows * local_k, tag + " K dequant size");
        for (uint64_t r = 0; r < rows; ++r) {
            check_floats(tag + " K dequant rank" + std::to_string(rank) + " row" +
                             std::to_string(r),
                         local_deq.data() + r * local_k,
                         global_deq.data() + r * k + rank * local_k, local_k);
        }

        CanonicalQuantStore reference =
            quantize_rows(id, src.data(), rows, local_k, k, rank * local_k);
        check_preshuffle_equivalent(tag + " K rank" + std::to_string(rank), local,
                                    reference);
    }
}

static void test_k_block_alignment_reject() {
    const uint64_t rows = 8, k = 64;
    const std::vector<float> src = make_source(rows, k, 77);
    CanonicalQuantStore global_store = build_global(QuantFormatId::Psq4, src, rows, k);
    QuantizedTensorView g = make_quant_view(global_store, QuantizedEncoding::Psq4, {8, 64});

    OwnedCanonicalTensor owned;
    QuantizedTensorView local;
    Status offset_status = materialize_rank_local_canonical(
        desc_of(1, {8, 64}, 0, 2, {rng(16, 32)}), g, owned, local);
    check(!offset_status.ok(), "PSQ4 K offset 16 rejected");
    Status extent_status = materialize_rank_local_canonical(
        desc_of(1, {8, 64}, 0, 2, {rng(0, 48)}), g, owned, local);
    check(!extent_status.ok(), "PSQ4 K extent 48 rejected");
    Status ok_status = materialize_rank_local_canonical(
        desc_of(1, {8, 64}, 0, 2, {rng(0, 32)}), g, owned, local);
    check(ok_status.ok(), "PSQ4 K extent 32 accepted");

    QuantizedTensorView g8;
    CanonicalQuantStore psq8 = build_global(QuantFormatId::Psq8, src, rows, k);
    g8 = make_quant_view(psq8, QuantizedEncoding::Psq8, {8, 64});
    Status psq8_status = materialize_rank_local_canonical(
        desc_of(1, {8, 64}, 1, 2, {rng(32, 16)}), g8, owned, local);
    check(!psq8_status.ok(), "PSQ8 K extent 16 rejected");
}

static void test_segmented_gdn() {
    const uint64_t qk = 64, value = 64;
    const uint64_t conv = 2 * qk + value;
    const uint64_t k = 32;
    const std::vector<float> src = make_source(conv, k, 4242);
    CanonicalQuantStore global_store = build_global(QuantFormatId::Psq8, src, conv, k);
    QuantizedTensorView g = make_quant_view(global_store, QuantizedEncoding::Psq8,
                                            {static_cast<int64_t>(conv),
                                             static_cast<int64_t>(k)});

    const std::vector<TensorPartitionRange> rank0 = {
        rng(0, 32), rng(64, 32), rng(128, 32)};
    OwnedCanonicalTensor owned;
    QuantizedTensorView local;
    Status st = materialize_rank_local_canonical(
        desc_of(0, {static_cast<int64_t>(conv), static_cast<int64_t>(k)}, 0, 2, rank0),
        g, owned, local);
    check(st.ok(), "segmented GDN materialize");
    if (!st.ok()) return;
    check(local.logical_shape == std::vector<int64_t>({96, 32}),
          "segmented GDN local shape");

    std::vector<float> global_deq;
    dequantize_canonical(global_store.view(), global_deq);
    std::vector<float> local_deq;
    dequantize_canonical(to_canonical(local), local_deq);

    const uint64_t expected_rows[3] = {0, 64, 128};
    for (uint64_t seg = 0; seg < 3; ++seg) {
        check_floats("segmented GDN segment" + std::to_string(seg),
                     local_deq.data() + seg * 32 * k,
                     global_deq.data() + expected_rows[seg] * k, 32 * k);
    }

    std::vector<float> reference_src;
    for (uint64_t seg = 0; seg < 3; ++seg) {
        const float* begin = src.data() + expected_rows[seg] * k;
        reference_src.insert(reference_src.end(), begin, begin + 32 * k);
    }
    CanonicalQuantStore reference =
        quantize_rows(QuantFormatId::Psq8, reference_src.data(), 96, k, k, 0);
    check_preshuffle_equivalent("segmented GDN", local, reference);
}

static void test_expert_3d_partition() {
    const uint64_t e = 8, n = 16, k = 32;
    const uint64_t rows = e * n;
    std::vector<float> src(rows * k);
    std::mt19937 gen(9001);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    for (float& f : src) f = dist(gen);

    CanonicalQuantStore global_store = build_global(QuantFormatId::Psq4, src, rows, k);
    QuantizedTensorView g = make_quant_view(
        global_store, QuantizedEncoding::Psq4,
        {static_cast<int64_t>(e), static_cast<int64_t>(n), static_cast<int64_t>(k)});

    OwnedCanonicalTensor owned;
    QuantizedTensorView local;
    Status st = materialize_rank_local_canonical(
        desc_of(0, {8, 16, 32}, 1, 2, {rng(4, 4)}), g, owned, local);
    check(st.ok(), "3D expert axis0 materialize");
    if (!st.ok()) return;
    check(local.logical_shape == std::vector<int64_t>({4, 16, 32}),
          "3D expert local shape");

    std::vector<float> global_deq;
    dequantize_canonical(global_store.view(), global_deq);
    std::vector<float> local_deq;
    dequantize_canonical(to_canonical(local), local_deq);
    check(local_deq.size() == 4 * n * k, "3D expert dequant size");
    for (uint64_t en = 0; en < 4 * n; ++en) {
        check_floats("3D expert row" + std::to_string(en), local_deq.data() + en * k,
                     global_deq.data() + (4 * n + en) * k, k);
    }

    TensorPartitionDesc gate_up = desc_of(
        0, {512, 1280, 2560}, 1, 2, {rng(256, 256)});
    auto gate_up_local = tensor_partition_local_shape(gate_up);
    check(gate_up_local.ok(), "expert gate_up plan computes");
    if (gate_up_local.ok()) {
        check(gate_up_local.value() == std::vector<int64_t>({256, 1280, 2560}),
              "expert gate_up local shape");
    }
    TensorPartitionDesc down = desc_of(0, {512, 2560, 640}, 0, 2, {rng(0, 256)});
    auto down_local = tensor_partition_local_shape(down);
    check(down_local.ok() && down_local.value() == std::vector<int64_t>({256, 2560, 640}),
          "expert down local shape");
    check(validate_tensor_partition(gate_up).ok(), "expert gate_up descriptor valid");
}

static void test_mxfp4_partition() {
    const uint64_t rows = 16, k = 64;
    mxfp4::Mxfp4CanonicalStore store;
    store.init(rows, k, k);
    std::vector<float> src_store(k);
    std::mt19937 gen(31337);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    for (uint64_t r = 0; r < rows; ++r) {
        for (float& f : src_store) f = dist(gen);
        if (!store.quantize_row(static_cast<uint32_t>(r),
                                std::span<const float>(src_store.data(), k))) {
            fail("mxfp4 quantize_row");
        }
    }
    if (!store.view().validate()) fail("mxfp4 global validate");

    QuantizedTensorView g;
    g.name = "w";
    g.encoding = QuantizedEncoding::Mxfp4;
    g.logical_shape = {16, 64};
    g.k_padded = 64;
    g.codes = ByteSpan{store.codes.data(), store.codes.size()};
    g.metadata1 = ByteSpan{store.scales.data(), store.scales.size()};

    const uint64_t codes_row = (k / 32u) * 16u;
    const uint64_t scale_row = k / 32u;
    OwnedCanonicalTensor owned;
    QuantizedTensorView local;
    Status st = materialize_rank_local_canonical(
        desc_of(0, {16, 64}, 1, 2, {rng(8, 8)}), g, owned, local);
    check(st.ok(), "mxfp4 N materialize");
    if (!st.ok()) return;
    check(local.logical_shape == std::vector<int64_t>({8, 64}), "mxfp4 N local shape");
    check_bytes_span("mxfp4 N codes", owned.codes, store.codes.data() + 8 * codes_row,
                     8 * codes_row);
    check_bytes_span("mxfp4 N metadata1", owned.metadata1,
                     store.scales.data() + 8 * scale_row, 8 * scale_row);

    Status k_status = materialize_rank_local_canonical(
        desc_of(1, {16, 64}, 1, 2, {rng(32, 32)}), g, owned, local);
    check(k_status.ok(), "mxfp4 K materialize");
    if (k_status.ok()) {
        check(local.k_padded == 32, "mxfp4 K local k_padded");
        std::vector<uint8_t> want;
        for (uint64_t r = 0; r < rows; ++r) {
            const uint8_t* block = store.codes.data() + r * codes_row + codes_row / 2u;
            want.insert(want.end(), block, block + codes_row / 2u);
        }
        check_bytes("mxfp4 K codes", owned.codes, want);
        std::vector<uint8_t> want_scales;
        for (uint64_t r = 0; r < rows; ++r) {
            const uint8_t* scale = store.scales.data() + r * scale_row + scale_row / 2u;
            want_scales.insert(want_scales.end(), scale, scale + scale_row / 2u);
        }
        check_bytes("mxfp4 K metadata1", owned.metadata1, want_scales);
    }
}

static void test_reject_cases() {
    const uint64_t rows = 8, k = 64;
    const std::vector<float> src = make_source(rows, k, 55);
    CanonicalQuantStore store = build_global(QuantFormatId::Psq4, src, rows, k);
    QuantizedTensorView g = make_quant_view(store, QuantizedEncoding::Psq4, {8, 64});

    OwnedCanonicalTensor owned;
    QuantizedTensorView local;

    Status shape_mismatch = materialize_rank_local_canonical(
        desc_of(0, {8, 32}, 0, 2, {rng(0, 4)}), g, owned, local);
    check(!shape_mismatch.ok(), "global_shape mismatch rejected");

    QuantizedTensorView truncated = g;
    truncated.codes = ByteSpan{store.codes.data(), store.codes.size() - 1};
    Status bad_size = materialize_rank_local_canonical(
        desc_of(0, {8, 64}, 0, 2, {rng(0, 4)}), truncated, owned, local);
    check(!bad_size.ok(), "truncated codes rejected");

    QuantizedTensorView missing = g;
    missing.codes = ByteSpan{};
    Status missing_codes = materialize_rank_local_canonical(
        desc_of(0, {8, 64}, 0, 2, {rng(0, 4)}), missing, owned, local);
    check(!missing_codes.ok(), "missing codes rejected");

    QuantizedTensorView fp8;
    fp8.name = "w";
    fp8.encoding = QuantizedEncoding::Fp8Block128;
    fp8.logical_shape = {8, 64};
    fp8.k_padded = 128;
    Status fp8_status = materialize_rank_local_canonical(
        desc_of(0, {8, 64}, 0, 2, {rng(0, 4)}), fp8, owned, local);
    check(!fp8_status.ok() && fp8_status.code() == Status::Code::unsupported,
          "fp8 partition rejected as unsupported");

    QuantizedTensorView bad_bf16 = make_quant_view(store, QuantizedEncoding::Psq4, {8, 64});
    bad_bf16.encoding = QuantizedEncoding::Bf16;
    Status bf16_codes = materialize_rank_local_canonical(
        desc_of(0, {8, 64}, 0, 2, {rng(0, 4)}), bad_bf16, owned, local);
    check(!bf16_codes.ok(), "bf16 view carrying codes rejected");
}

int main() {
    test_bf16_partitions();
    test_psq_n_partition(QuantFormatId::Psq4, QuantizedEncoding::Psq4, "psq4");
    test_psq_n_partition(QuantFormatId::Psq8, QuantizedEncoding::Psq8, "psq8");
    test_psq_k_partition(QuantFormatId::Psq4, QuantizedEncoding::Psq4, "psq4");
    test_psq_k_partition(QuantFormatId::Psq8, QuantizedEncoding::Psq8, "psq8");
    test_k_block_alignment_reject();
    test_segmented_gdn();
    test_expert_3d_partition();
    test_mxfp4_partition();
    test_reject_cases();

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
