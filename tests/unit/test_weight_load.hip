#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/io/safetensors_writer.h>
#include <phaseshift/models/qwen35/weights/model_weights.h>
#include <phaseshift/quantization/fpx/crc32.h>
#include <phaseshift/quantization/fpx/quantized_manifest.h>
#include <phaseshift/quantization/fpx/quantized_model_reader.h>
#include <phaseshift/quantization/psq/quant_canonical.h>
#include <phaseshift/quantization/psq/quant_preshuffle.h>
#include <phaseshift/quantization/fp8/block128.h>
#include <phaseshift/quantization/fp8/block128_native.h>
#include <phaseshift/quantization/mxfp4/mxfp4.h>
#include <phaseshift/quantization/mxfp4/mxfp4_native.h>
#include <phaseshift/weights/matrix_weight.h>
#include <phaseshift/weights/weight_loader.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using namespace ps;

static int g_fail = 0;

static void fail(const std::string& msg) {
    ++g_fail;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static void check_eq_u64(uint64_t got, uint64_t want, const std::string& msg) {
    if (got != want) fail(msg + " got=" + std::to_string(got) + " want=" + std::to_string(want));
}

static std::vector<uint8_t> d2h(const void* dev, std::size_t bytes) {
    std::vector<uint8_t> host(bytes);
    if (bytes == 0) return host;
    const hipError_t e = hipMemcpy(host.data(), dev, bytes, hipMemcpyDeviceToHost);
    if (e != hipSuccess) {
        fail(std::string("d2h ") + hipGetErrorString(e));
        host.assign(bytes, 0);
    }
    return host;
}

static std::string crc_hex(const std::vector<uint8_t>& bytes) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%08x",
                  quantization::fpx::crc32_iso_hdlc(bytes.data(), bytes.size()));
    return buf;
}

static void write_text(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << text;
}

static std::vector<uint8_t> bf16_bytes(uint64_t n, uint32_t seed) {
    std::vector<uint8_t> out(n * 2u);
    auto* u = reinterpret_cast<uint16_t*>(out.data());
    for (uint64_t i = 0; i < n; ++i) {
        u[i] = static_cast<uint16_t>(0x3F80u + ((seed + static_cast<uint32_t>(i)) % 64u));
    }
    return out;
}

static std::vector<uint8_t> f32_bytes(uint64_t n, uint32_t seed) {
    std::vector<uint8_t> out(n * 4u);
    auto* f = reinterpret_cast<float*>(out.data());
    for (uint64_t i = 0; i < n; ++i) {
        f[i] = 0.25f * static_cast<float>((seed + static_cast<uint32_t>(i)) % 251u) - 31.0f;
    }
    return out;
}

static bool bf16_convert_matches(const std::vector<uint8_t>& got, const std::vector<uint8_t>& src) {
    const auto* sf = reinterpret_cast<const float*>(src.data());
    const uint64_t n = src.size() / 4u;
    for (uint64_t i = 0; i < n; ++i) {
        const uint16_t want = f32_to_bf16_rne(sf[i]);
        if (std::memcmp(got.data() + i * 2, &want, 2) != 0) return false;
    }
    return true;
}

struct Planned {
    std::string name;
    io::SType dtype;
    std::vector<std::size_t> shape;
    std::vector<uint8_t> bytes;
};

static uint64_t numel(const std::vector<std::size_t>& shape) {
    uint64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

static bool write_shard(const std::string& path, std::vector<Planned>& tensors,
                        const std::vector<std::pair<std::string, std::string>>& metadata) {
    auto w = io::SafetensorsWriter::create(path);
    if (!w.ok()) { fail("writer create"); return false; }
    io::SafetensorsWriter writer = w.release();
    for (auto& t : tensors) {
        if (!writer.plan_tensor(t.name, t.dtype, t.shape).ok()) { fail("plan " + t.name); return false; }
    }
    for (auto& kv : metadata) {
        if (!writer.set_metadata(kv.first, kv.second).ok()) { fail("metadata " + kv.first); return false; }
    }
    auto hdr = writer.write_header();
    if (!hdr.ok()) { fail("write_header"); return false; }
    for (auto& t : tensors) {
        if (!writer.write_tensor(t.name, t.bytes.data(), t.bytes.size()).ok()) { fail("write " + t.name); return false; }
    }
    if (!writer.finish().ok()) { fail("finish"); return false; }
    return true;
}

static std::string config_json(uint32_t layers, const std::string& layer_types) {
    return std::string("{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":false,\"text_config\":{") +
        "\"model_type\":\"qwen3_5_text\",\"vocab_size\":40,\"hidden_size\":32,\"intermediate_size\":32,\"" +
        "num_hidden_layers\":" + std::to_string(layers) + ",\"full_attention_interval\":2,\"layer_types\":[" +
        layer_types + "]}}";
}

static fs::path g_root;

static fs::path make_case_dir(const std::string& name) {
    const fs::path dir = g_root / name;
    std::error_code ec;
    fs::create_directories(dir, ec);
    return dir;
}

static void write_index(const std::string& dir, const std::string& shard,
                        const std::vector<Planned>& tensors) {
    std::string j = "{\"weight_map\":{";
    bool first = true;
    for (auto& t : tensors) {
        if (!first) j += ",";
        first = false;
        j += "\"" + t.name + "\":\"" + shard + "\"";
    }
    j += "},\"metadata\":{\"total_size\":0}}";
    write_text(dir + "/model.safetensors.index.json", j);
}

struct QTensor {
    std::string physical;
    io::SType dtype;
    std::vector<std::size_t> shape;
    std::vector<uint8_t> bytes;
};

struct QMatrixPlan {
    quantization::fpx::QuantizedTensorMetadata meta;
    std::vector<QTensor> physical;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> metadata1;
};

static std::vector<uint8_t> as_bytes(const std::vector<float>& v) {
    std::vector<uint8_t> out(v.size() * sizeof(float));
    if (!out.empty()) std::memcpy(out.data(), v.data(), out.size());
    return out;
}

static quantization::QuantFormatId qfmt(quantization::fpx::QuantizedEncoding enc) {
    using namespace quantization;
    switch (enc) {
        case fpx::QuantizedEncoding::Psq4: return QuantFormatId::Psq4;
        case fpx::QuantizedEncoding::Psq8: return QuantFormatId::Psq8;
        case fpx::QuantizedEncoding::Fp8Block128: return QuantFormatId::Fp8Block128;
        case fpx::QuantizedEncoding::Mxfp4: return QuantFormatId::Mxfp4;
        default: return QuantFormatId::None;
    }
}

static QMatrixPlan make_matrix(quantization::fpx::QuantizedEncoding enc, uint64_t rows,
                               uint64_t k, const std::string& logical) {
    using namespace quantization;
    using fpx::QuantizedEncoding;
    QMatrixPlan plan;
    plan.meta.encoding = enc;
    plan.meta.role = "weight";
    plan.meta.logical_shape = {static_cast<int64_t>(rows), static_cast<int64_t>(k)};

    if (enc == fpx::QuantizedEncoding::Bf16) {
        plan.meta.k_padded = k;
        QTensor data;
        data.physical = logical + ".__phaseshift_data";
        data.dtype = io::SType::BF16;
        data.shape = {static_cast<size_t>(rows), static_cast<size_t>(k)};
        data.bytes = bf16_bytes(rows * k, static_cast<uint32_t>(rows * 7 + k));
        fpx::QuantizedTensorRef ref;
        ref.tensor = data.physical;
        ref.dtype = "BF16";
        ref.crc32 = crc_hex(data.bytes);
        plan.meta.data = ref;
        plan.codes = data.bytes;
        plan.physical.push_back(std::move(data));
        return plan;
    }

    const auto add_stream = [&](const std::string& suffix, const std::vector<uint8_t>& stream,
                                uint64_t row_bytes, std::optional<fpx::QuantizedTensorRef>& slot) {
        if (stream.empty()) return;
        QTensor t;
        t.physical = logical + suffix;
        t.dtype = io::SType::U8;
        t.shape = {static_cast<size_t>(rows), static_cast<size_t>(row_bytes)};
        t.bytes = stream;
        fpx::QuantizedTensorRef ref;
        ref.tensor = t.physical;
        ref.dtype = "U8";
        ref.crc32 = crc_hex(t.bytes);
        slot = ref;
        plan.physical.push_back(std::move(t));
    };

    if (enc == fpx::QuantizedEncoding::Fp8Block128) {
        const uint64_t fkp = (k + 127u) / 128u * 128u;
        plan.meta.k_padded = fkp;
        fp8::Fp8BlockCanonicalStore fstore;
        fstore.init(1, rows, k, fkp);
        std::mt19937 rng(static_cast<uint32_t>(rows * 131 + k + 5));
        std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
        std::vector<float> src(rows * k);
        for (auto& v : src) v = dist(rng);
        if (!fstore.quantize_matrix(0, src.data(), k)) fail("fp8 quantize_matrix " + logical);
        if (!fstore.view().validate()) fail("fp8 store invalid " + logical);
        plan.codes = fstore.codes;
        plan.metadata1 = as_bytes(fstore.scales);

        QTensor ct;
        ct.physical = logical + ".__phaseshift_codes";
        ct.dtype = io::SType::U8;
        ct.shape = {static_cast<std::size_t>(rows), static_cast<std::size_t>(fkp)};
        ct.bytes = fstore.codes;
        fpx::QuantizedTensorRef cref;
        cref.tensor = ct.physical;
        cref.dtype = "U8";
        cref.crc32 = crc_hex(ct.bytes);
        plan.meta.codes = cref;
        plan.physical.push_back(std::move(ct));

        QTensor st;
        st.physical = logical + ".__phaseshift_metadata1";
        st.dtype = io::SType::F32;
        st.shape = {static_cast<std::size_t>((rows + 127u) / 128u),
                    static_cast<std::size_t>(fkp / 128u)};
        st.bytes = plan.metadata1;
        fpx::QuantizedTensorRef sref;
        sref.tensor = st.physical;
        sref.dtype = "F32";
        sref.crc32 = crc_hex(st.bytes);
        plan.meta.metadata1 = sref;
        plan.physical.push_back(std::move(st));
        return plan;
    }

    if (enc == fpx::QuantizedEncoding::Mxfp4) {
        const uint64_t mkp = (k + 31u) / 32u * 32u;
        plan.meta.k_padded = mkp;
        mxfp4::Mxfp4CanonicalStore mstore;
        mstore.init(rows, k, mkp);
        std::mt19937 rng(static_cast<uint32_t>(rows * 131 + k + 9));
        std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
        std::vector<float> src(k);
        for (uint64_t r = 0; r < rows; ++r) {
            for (uint64_t i = 0; i < k; ++i) src[i] = dist(rng);
            if (!mstore.quantize_row(static_cast<uint32_t>(r), std::span<const float>(src.data(), k)))
                fail("mxfp4 quantize_row " + logical);
        }
        if (!mstore.view().validate()) fail("mxfp4 store invalid " + logical);
        plan.codes = mstore.codes;
        plan.metadata1 = mstore.scales;
        const uint64_t mnb = mkp / 32u;
        add_stream(".__phaseshift_codes", mstore.codes, mnb * 16u, plan.meta.codes);
        add_stream(".__phaseshift_metadata1", mstore.scales, mnb, plan.meta.metadata1);
        return plan;
    }

    const uint64_t kp = (k + 31u) / 32u * 32u;
    plan.meta.k_padded = kp;
    plan.meta.codebook = (enc == fpx::QuantizedEncoding::Psq4) ? "cb10" : "";

    psq::CanonicalQuantStore store;
    store.init(qfmt(enc), rows, k, kp);
    std::mt19937 rng(static_cast<uint32_t>(rows * 131 + k));
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    std::vector<float> src(k);
    for (uint64_t r = 0; r < rows; ++r) {
        for (uint64_t i = 0; i < k; ++i) src[i] = dist(rng);
        if (!store.quantize_row(static_cast<uint32_t>(r), std::span<const float>(src.data(), k))) {
            fail("quantize_row " + logical);
        }
    }
    if (!store.view().validate()) fail("store invalid " + logical);

    const uint64_t nb = kp / 32u;
    const uint64_t codes_row = nb * ((enc == fpx::QuantizedEncoding::Psq4) ? 16u : 32u);

    plan.codes = store.codes;
    plan.metadata1 = store.metadata1;

    add_stream(".__phaseshift_codes", store.codes, codes_row, plan.meta.codes);
    add_stream(".__phaseshift_metadata1", store.metadata1, nb * 2u, plan.meta.metadata1);
    return plan;
}

static quantization::fp8::Fp8Block128NativeHost expect_native_fp8(
    const std::vector<uint8_t>& codes, const std::vector<uint8_t>& scales_bytes,
    uint64_t rows, uint64_t kp) {
    quantization::fp8::Fp8BlockCanonicalView cv;
    cv.batch = 1;
    cv.n = rows;
    cv.padded_k = kp;
    cv.codes = codes.data();
    cv.codes_bytes = codes.size();
    cv.scales = reinterpret_cast<const float*>(scales_bytes.data());
    cv.scales_bytes = scales_bytes.size();
    cv.scale_n = (rows + 127u) / 128u;
    cv.scale_k = kp / 128u;
    quantization::fp8::Fp8Block128NativeHost native;
    if (!quantization::fp8::preshuffle_fp8_block128_native(cv, native)) fail("expect_native_fp8");
    return native;
}

static quantization::mxfp4::Mxfp4NativeHost expect_native_mxfp4(
    const std::vector<uint8_t>& codes, const std::vector<uint8_t>& scales,
    uint64_t rows, uint64_t kp) {
    quantization::mxfp4::Mxfp4CanonicalView cv;
    cv.rows = rows;
    cv.padded_k = kp;
    cv.codes = codes.data();
    cv.codes_bytes = codes.size();
    cv.scales = scales.data();
    cv.scales_bytes = scales.size();
    quantization::mxfp4::Mxfp4NativeHost native;
    if (!quantization::mxfp4::preshuffle_mxfp4_native(cv, native)) fail("expect_native_mxfp4");
    return native;
}



static QTensor make_small(uint64_t n, const std::string& logical, std::vector<std::size_t> shape) {
    QTensor t;
    t.physical = logical + ".__phaseshift_data";
    t.dtype = io::SType::BF16;
    t.shape = std::move(shape);
    t.bytes = bf16_bytes(n, static_cast<uint32_t>(n * 3 + 11));
    return t;
}

static quantization::fpx::QuantizedTensorMetadata small_meta(const QTensor& t, const std::string& role) {
    using namespace quantization::fpx;
    QuantizedTensorMetadata m;
    m.role = role;
    m.encoding = QuantizedEncoding::Bf16;
    for (auto d : t.shape) m.logical_shape.push_back(static_cast<int64_t>(d));
    m.k_padded = static_cast<uint64_t>(t.shape.back());
    QuantizedTensorRef ref;
    ref.tensor = t.physical;
    ref.dtype = "BF16";
    ref.crc32 = crc_hex(t.bytes);
    m.data = ref;
    return m;
}

static quantization::psq::NativeQuantHost expect_native(quantization::QuantFormatId fmt,
                                                        const std::vector<uint8_t>& codes,
                                                        const std::vector<uint8_t>& metadata1,
                                                        uint64_t rows, uint64_t kp) {
    quantization::psq::CanonicalQuantView cv;
    cv.format_id = fmt;
    cv.desc = quantization::quant_format_desc(fmt);
    cv.rows = rows;
    cv.padded_k = kp;
    cv.codes = codes.data();
    cv.codes_bytes = codes.size();
    cv.metadata1 = metadata1.empty() ? nullptr : metadata1.data();
    cv.metadata1_bytes = metadata1.size();
    quantization::psq::NativeQuantHost native;
    if (!quantization::psq::preshuffle_native(cv, native)) {
        fail("expect_native preshuffle");
    }
    return native;
}

static std::size_t tensor_span_bytes(const gpu::Tensor& t) {
    std::size_t span = 0;
    if (!t.required_span_bytes(span).ok()) {
        fail("required_span_bytes");
        return 0;
    }
    return span;
}

static void check_matrix_bytes(const weights::MatrixWeight& w, const std::vector<uint8_t>& want_codes,
                               const std::vector<uint8_t>& want_scales, const std::string& tag) {
    check_eq_u64(want_codes.size(), tensor_span_bytes(w.codes), tag + " codes size");
    if (!want_codes.empty()) {
        const auto got = d2h(w.codes.data<void>(), want_codes.size());
        check(got == want_codes, tag + " codes bytes");
    }
    check_eq_u64(want_scales.size(), tensor_span_bytes(w.scales), tag + " scale size");
    if (!want_scales.empty()) {
        const auto got = d2h(w.scales.data<void>(), want_scales.size());
        check(got == want_scales, tag + " scales bytes");
    }
}

static void test_bf16_matrix_load(gpu::GpuArena& arena, hipStream_t stream) {
    const fs::path dir = make_case_dir("bf16_single");
    std::vector<Planned> t;
    Planned mat;
    mat.name = "x.weight";
    mat.dtype = io::SType::BF16;
    mat.shape = {32, 32};
    mat.bytes = bf16_bytes(32 * 32, 55);
    t.push_back(std::move(mat));
    write_index(dir.string(), "model.safetensors-00001-of-00001.safetensors", t);
    if (!write_shard((dir / "model.safetensors-00001-of-00001.safetensors").string(), t, {})) return;

    auto cr = weights::SafetensorsCollection::open(dir.string());
    if (!cr.ok()) { fail("collection open"); return; }
    weights::SafetensorsCollection coll = cr.release();

    check(!weights::is_quantized_model_dir(dir.string()), "bf16 dir not quantized");
    check(!coll.tensor_spec("missing.weight").ok(), "missing spec");

    weights::WeightLoadOptions raw_options;
    raw_options.preshuffle = false;
    auto raw = weights::load_bf16_matrix(coll, "x.weight", arena, stream, raw_options);
    if (!raw.ok()) { fail("raw load"); return; }
    weights::MatrixWeight m = raw.release();
    check_eq_u64(m.rows, 32, "raw rows");
    check_eq_u64(m.cols, 32, "raw cols");
    check_eq_u64(m.k_padded, 32, "raw k_padded");
    check(!m.preshuffled, "raw not preshuffled");
    check(m.quant_spec == quantization::QuantSpecId::BF16_V1, "raw quant_spec");
    check(m.compute_spec == quantization::ComputeSpecId::BF16_F32_BF16, "raw compute_spec");
    {
        const auto got = d2h(m.bf16.data<void>(), t[0].bytes.size());
        check(got == t[0].bytes, "raw bytes");
    }

    auto pre = weights::load_bf16_matrix(coll, "x.weight", arena, stream,
                                         weights::WeightLoadOptions{});
    if (!pre.ok()) { fail("preshuffle load"); return; }
    weights::MatrixWeight p = pre.release();
    check(!p.preshuffled, "bf16 never preshuffled");
    const auto got = d2h(p.bf16.data<void>(), t[0].bytes.size());
    check(got == t[0].bytes, "bf16 bytes");
}

static bool build_bf16_model(const fs::path& dir) {
    std::vector<Planned> t;
    const auto add = [&](const std::string& name, io::SType dt, std::vector<std::size_t> shape, uint32_t seed) {
        Planned p;
        p.name = name;
        p.dtype = dt;
        p.shape = std::move(shape);
        const uint64_t n = numel(p.shape);
        p.bytes = (dt == io::SType::F32) ? f32_bytes(n, seed) : bf16_bytes(n, seed);
        t.push_back(std::move(p));
    };

    add("model.language_model.embed_tokens.weight", io::SType::BF16, {40, 32}, 1);
    add("model.language_model.lm_head.weight", io::SType::BF16, {40, 32}, 2);
    add("model.language_model.norm.weight", io::SType::BF16, {32}, 3);

    add("model.language_model.layers.0.input_layernorm.weight", io::SType::F32, {32}, 4);
    add("model.language_model.layers.0.post_attention_layernorm.weight", io::SType::BF16, {32}, 5);
    add("model.language_model.layers.0.mlp.gate_proj.weight", io::SType::BF16, {32, 32}, 6);
    add("model.language_model.layers.0.mlp.up_proj.weight", io::SType::BF16, {32, 32}, 7);
    add("model.language_model.layers.0.mlp.down_proj.weight", io::SType::BF16, {32, 32}, 8);
    add("model.language_model.layers.0.linear_attn.in_proj_a.weight", io::SType::BF16, {8, 32}, 9);
    add("model.language_model.layers.0.linear_attn.in_proj_b.weight", io::SType::BF16, {8, 32}, 10);
    add("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", io::SType::BF16, {24, 32}, 11);
    add("model.language_model.layers.0.linear_attn.in_proj_z.weight", io::SType::BF16, {8, 32}, 12);
    add("model.language_model.layers.0.linear_attn.conv1d.weight", io::SType::F32, {8, 1, 4}, 13);
    add("model.language_model.layers.0.linear_attn.out_proj.weight", io::SType::BF16, {32, 32}, 14);
    add("model.language_model.layers.0.linear_attn.norm.weight", io::SType::F32, {32}, 15);
    add("model.language_model.layers.0.linear_attn.dt_bias", io::SType::BF16, {8}, 16);
    add("model.language_model.layers.0.linear_attn.A_log", io::SType::F32, {8}, 17);

    add("model.language_model.layers.1.input_layernorm.weight", io::SType::BF16, {32}, 18);
    add("model.language_model.layers.1.post_attention_layernorm.weight", io::SType::BF16, {32}, 19);
    add("model.language_model.layers.1.mlp.gate_proj.weight", io::SType::BF16, {32, 32}, 20);
    add("model.language_model.layers.1.mlp.up_proj.weight", io::SType::BF16, {32, 32}, 21);
    add("model.language_model.layers.1.mlp.down_proj.weight", io::SType::BF16, {32, 32}, 22);
    add("model.language_model.layers.1.self_attn.q_proj.weight", io::SType::BF16, {32, 32}, 23);
    add("model.language_model.layers.1.self_attn.k_proj.weight", io::SType::BF16, {8, 24}, 24);
    add("model.language_model.layers.1.self_attn.v_proj.weight", io::SType::BF16, {8, 24}, 25);
    add("model.language_model.layers.1.self_attn.o_proj.weight", io::SType::BF16, {32, 32}, 26);
    add("model.language_model.layers.1.self_attn.q_norm.weight", io::SType::BF16, {32}, 27);
    add("model.language_model.layers.1.self_attn.k_norm.weight", io::SType::BF16, {32}, 28);

    add("mtp.pre_fc_norm_embedding.weight", io::SType::BF16, {32}, 29);
    add("mtp.pre_fc_norm_hidden.weight", io::SType::BF16, {32}, 30);
    add("mtp.norm.weight", io::SType::BF16, {32}, 31);
    add("mtp.fc.weight", io::SType::BF16, {32, 64}, 32);
    add("mtp.layers.0.input_layernorm.weight", io::SType::BF16, {32}, 33);
    add("mtp.layers.0.post_attention_layernorm.weight", io::SType::BF16, {32}, 34);
    add("mtp.layers.0.mlp.gate_proj.weight", io::SType::BF16, {32, 32}, 35);
    add("mtp.layers.0.mlp.up_proj.weight", io::SType::BF16, {32, 32}, 36);
    add("mtp.layers.0.mlp.down_proj.weight", io::SType::BF16, {32, 32}, 37);
    add("mtp.layers.0.self_attn.q_proj.weight", io::SType::BF16, {32, 32}, 38);
    add("mtp.layers.0.self_attn.k_proj.weight", io::SType::BF16, {8, 32}, 39);
    add("mtp.layers.0.self_attn.v_proj.weight", io::SType::BF16, {8, 32}, 40);
    add("mtp.layers.0.self_attn.o_proj.weight", io::SType::BF16, {32, 32}, 41);
    add("mtp.layers.0.self_attn.q_norm.weight", io::SType::BF16, {32}, 42);
    add("mtp.layers.0.self_attn.k_norm.weight", io::SType::BF16, {32}, 43);

    if (!write_shard((dir / "model.safetensors-00001-of-00001.safetensors").string(), t, {})) {
        return false;
    }
    write_index(dir.string(), "model.safetensors-00001-of-00001.safetensors", t);
    write_text((dir / "config.json").string(), config_json(2, "\"linear_attention\",\"full_attention\""));
    return true;
}

static void test_bf16_model_load(gpu::GpuArena& arena, hipStream_t stream) {
    const fs::path dir = make_case_dir("bf16_model");
    if (!build_bf16_model(dir)) return;

    qwen35::Qwen35LoadOptions options;
    auto wr = qwen35::load_qwen35_weights_from_safetensors(dir.string(), arena, stream, options);
    if (!wr.ok()) { fail("bf16 model load: " + wr.status().message()); return; }
    qwen35::Qwen35ModelWeights w = wr.release();

    check_eq_u64(w.layers.size(), 2, "model layers");
    check(w.layers[0].is_gdn, "layer0 gdn");
    check(!w.layers[1].is_gdn, "layer1 attn");
    check(w.mtp.present, "mtp present");
    check(!w.lm_head_tied, "lm_head untied");

    check_eq_u64(w.embed_tokens.rows, 40, "embed rows");
    check_eq_u64(w.embed_tokens.cols, 32, "embed cols");
    check_eq_u64(w.embed_tokens.k_padded, 32, "embed k_padded");
    check(!w.embed_tokens.preshuffled, "embed not preshuffled");

    check_eq_u64(w.layers[1].attn_q_proj.rows, 32, "q rows");
    check_eq_u64(w.layers[1].attn_k_proj.cols, 24, "k cols");
    check_eq_u64(w.layers[1].attn_k_proj.k_padded, 24, "k k_padded fallback");
    check(!w.layers[1].attn_k_proj.preshuffled, "k not preshuffled");

    {
        const auto got = d2h(w.layers[0].attn_A_log.data<void>(), 8 * 2);
        check(bf16_convert_matches(got, f32_bytes(8, 17)), "A_log f32->bf16 conversion");
    }
    {
        const auto got = d2h(w.layers[0].input_layernorm_weight.data<void>(), 32 * 4);
        check(got == f32_bytes(32, 4), "input_layernorm f32 preserved");
    }
    {
        const auto got = d2h(w.layers[0].attn_norm_weight.data<void>(), 32 * 2);
        check(bf16_convert_matches(got, f32_bytes(32, 15)), "gdn norm f32->bf16 conversion");
    }
    {
        const auto got = d2h(w.layers[0].attn_conv1d_weight.data<void>(), 32 * 2);
        check(bf16_convert_matches(got, f32_bytes(32, 13)), "conv1d f32->bf16 conversion");
        check_eq_u64(w.layers[0].attn_conv1d_weight.ndim(), 3, "conv1d ndim");
    }
    {
        const auto got = d2h(w.final_norm_weight.data<void>(), 64);
        check(got == bf16_bytes(32, 3), "final_norm bytes");
    }
    check(w.mtp.fc.encoding == weights::MatrixEncoding::Bf16, "mtp fc encoding");
    check_eq_u64(w.mtp.fc.rows, 32, "mtp fc rows == H");
    check_eq_u64(w.mtp.fc.cols, 64, "mtp fc cols == 2H");
    check(!w.mtp.layer.is_gdn, "mtp layer attn");

    qwen35::Qwen35LoadOptions no_pre;
    no_pre.weights.preshuffle = false;
    auto wr2 = qwen35::load_qwen35_weights_from_safetensors(dir.string(), arena, stream, no_pre);
    if (!wr2.ok()) { fail("bf16 model load skip_preshuffle"); return; }
    qwen35::Qwen35ModelWeights w2 = wr2.release();
    check(!w2.layers[1].attn_q_proj.preshuffled, "skip_preshuffle q raw");
}

static void test_standalone_quantized_reader(gpu::GpuArena& arena, hipStream_t stream) {
    using namespace quantization;
    using fpx::QuantizedEncoding;
    using fpx::QuantizedModelReader;

    const fs::path dir = make_case_dir("quantized_standalone");

    QMatrixPlan m_psq4 = make_matrix(QuantizedEncoding::Psq4, 32, 48, "q4");
    QMatrixPlan m_psq8 = make_matrix(QuantizedEncoding::Psq8, 16, 32, "q8");
    QMatrixPlan m_fp8 = make_matrix(QuantizedEncoding::Fp8Block128, 130, 200, "fp8");
    QMatrixPlan m_mx = make_matrix(QuantizedEncoding::Mxfp4, 40, 100, "mx");
    QMatrixPlan m_bf = make_matrix(QuantizedEncoding::Bf16, 8, 24, "bf");
    QTensor small = make_small(32, "vec", {32});

    std::vector<Planned> shard;
    fpx::QuantizedManifest manifest;
    manifest.architecture = "qwen35_dense";
    manifest.preset = "psq";
    manifest.source_dtype = "BF16";
    const auto add_matrix = [&](const std::string& logical, QMatrixPlan& plan) {
        for (auto& p : plan.physical) {
            Planned pl;
            pl.name = p.physical;
            pl.dtype = p.dtype;
            pl.shape = p.shape;
            pl.bytes = p.bytes;
            shard.push_back(std::move(pl));
        }
        manifest.tensors[logical] = plan.meta;
    };
    add_matrix("q4", m_psq4);
    add_matrix("q8", m_psq8);
    add_matrix("fp8", m_fp8);
    add_matrix("mx", m_mx);
    add_matrix("bf", m_bf);
    {
        Planned pl;
        pl.name = small.physical;
        pl.dtype = small.dtype;
        pl.shape = small.shape;
        pl.bytes = small.bytes;
        shard.push_back(std::move(pl));
        manifest.tensors["vec"] = small_meta(small, "norm");
    }
    manifest.aliases["vec_alias"] = "vec";
    manifest.aliases["dangling"] = "nope";

    auto ms = fpx::serialize_quantized_manifest(manifest);
    if (!ms.ok()) { fail("serialize manifest"); return; }
    if (!write_shard((dir / "model.safetensors").string(), shard,
                     {{"phaseshift.format", fpx::kQuantizedSafetensorsFormat},
                      {"phaseshift.format_version", std::to_string(fpx::kQuantizedSafetensorsFormatVersion)},
                      {"phaseshift.quantization_metadata", fpx::kQuantizedMetadataFile}})) {
        return;
    }
    write_text((dir / fpx::kQuantizedMetadataFile).string(), ms.value());

    check(weights::is_quantized_model_dir(dir.string()), "is_quantized_model_dir(standalone)");

    auto rr = QuantizedModelReader::open(dir.string(), true);
    if (!rr.ok()) { fail("reader open: " + rr.status().message()); return; }
    QuantizedModelReader reader = rr.release();

    check(!weights::validate_quantized_model(reader).ok(), "dangling alias must fail validation");
    manifest.aliases.erase("dangling");
    auto ms2 = fpx::serialize_quantized_manifest(manifest);
    if (!ms2.ok()) { fail("serialize manifest 2"); return; }
    write_text((dir / fpx::kQuantizedMetadataFile).string(), ms2.value());
    auto rr2 = QuantizedModelReader::open(dir.string(), true);
    if (!rr2.ok()) { fail("reader reopen: " + rr2.status().message()); return; }
    QuantizedModelReader good = rr2.release();
    check(weights::validate_quantized_model(good).ok(), "validate_quantized_model");

    {
        auto w = weights::load_quantized_matrix(good, "q4", arena, stream, weights::WeightLoadOptions{});
        if (!w.ok()) { fail("load q4: " + w.status().message()); }
        else {
            weights::MatrixWeight m = w.release();
            check_eq_u64(static_cast<uint32_t>(m.encoding), static_cast<uint32_t>(weights::MatrixEncoding::Psq4), "q4 encoding");
            check_eq_u64(m.rows, 32, "q4 rows");
            check_eq_u64(m.cols, 48, "q4 cols");
            check_eq_u64(m.k_padded, 64, "q4 k_padded");
            check_eq_u64(m.weight_scale_group, 32, "q4 wsg");
            check(m.preshuffled, "q4 preshuffled");
            check(m.quant_spec == QuantSpecId::PSQ4_S32_BF16_V1, "q4 quant_spec");
            check(m.compute_spec == ComputeSpecId::PSQ4_W4A8_F32_BF16, "q4 compute_spec");
            const auto n = expect_native(QuantFormatId::Psq4, m_psq4.codes, m_psq4.metadata1, 32, 64);
            check_matrix_bytes(m, n.codes, n.metadata1, "q4");
        }
    }
    {
        auto w = weights::load_quantized_matrix(good, "q8", arena, stream, weights::WeightLoadOptions{});
        if (!w.ok()) { fail("load q8: " + w.status().message()); }
        else {
            weights::MatrixWeight m = w.release();
            check_eq_u64(static_cast<uint32_t>(m.encoding), static_cast<uint32_t>(weights::MatrixEncoding::Psq8), "q8 encoding");
            check_eq_u64(m.k_padded, 32, "q8 k_padded");
            check(m.preshuffled, "q8 preshuffled");
            check(m.quant_spec == QuantSpecId::PSQ8_S32_BF16_V1, "q8 quant_spec");
            check(m.compute_spec == ComputeSpecId::PSQ8_W8A8_F32_BF16, "q8 compute_spec");
            const auto n = expect_native(QuantFormatId::Psq8, m_psq8.codes, m_psq8.metadata1, 16, 32);
            check_matrix_bytes(m, n.codes, n.metadata1, "q8");
        }
    }
    {
        auto w = weights::load_quantized_matrix(good, "fp8", arena, stream, weights::WeightLoadOptions{});
        if (!w.ok()) { fail("load fp8: " + w.status().message()); }
        else {
            weights::MatrixWeight m = w.release();
            check_eq_u64(static_cast<uint32_t>(m.encoding),
                         static_cast<uint32_t>(weights::MatrixEncoding::Fp8Block128), "fp8 encoding");
            check_eq_u64(m.rows, 130, "fp8 rows");
            check_eq_u64(m.cols, 200, "fp8 cols");
            check_eq_u64(m.k_padded, 256, "fp8 k_padded");
            check_eq_u64(m.weight_scale_group, 128, "fp8 wsg");
            check(m.preshuffled, "fp8 preshuffled");
            check(m.quant_spec == QuantSpecId::FP8_E4M3_S128X128_F32_V1, "fp8 quant_spec");
            check(m.compute_spec == ComputeSpecId::FP8_W8A8_F32_BF16, "fp8 compute_spec");
            const auto n = expect_native_fp8(m_fp8.codes, m_fp8.metadata1, 130, 256);
            check_matrix_bytes(m, n.codes, as_bytes(n.scales), "fp8");
        }
    }
    {
        weights::WeightLoadOptions no_pre;
        no_pre.preshuffle = false;
        auto w = weights::load_quantized_matrix(good, "fp8", arena, stream, no_pre);
        if (!w.ok()) { fail("load fp8 canonical: " + w.status().message()); }
        else {
            weights::MatrixWeight m = w.release();
            check(!m.preshuffled, "fp8 canonical not preshuffled");
            check_eq_u64(m.codes_row_stride_bytes, 256, "fp8 canonical codes stride");
            check_eq_u64(m.storage_scale_stride_bytes, 8, "fp8 canonical scale stride");
            check_matrix_bytes(m, m_fp8.codes, m_fp8.metadata1, "fp8 canonical");
        }
    }
    {
        auto w = weights::load_quantized_matrix(good, "mx", arena, stream, weights::WeightLoadOptions{});
        if (!w.ok()) { fail("load mx: " + w.status().message()); }
        else {
            weights::MatrixWeight m = w.release();
            check_eq_u64(static_cast<uint32_t>(m.encoding),
                         static_cast<uint32_t>(weights::MatrixEncoding::Mxfp4), "mx encoding");
            check_eq_u64(m.rows, 40, "mx rows");
            check_eq_u64(m.cols, 100, "mx cols");
            check_eq_u64(m.k_padded, 128, "mx k_padded");
            check_eq_u64(m.weight_scale_group, 32, "mx wsg");
            check(m.preshuffled, "mx preshuffled");
            check(m.quant_spec == QuantSpecId::MXFP4_E2M1_S32_E8M0_V1, "mx quant_spec");
            check(m.compute_spec == ComputeSpecId::MXFP4_W4A8_F32_BF16, "mx compute_spec");
            const auto n = expect_native_mxfp4(m_mx.codes, m_mx.metadata1, 40, 128);
            check_matrix_bytes(m, n.codes, n.scales, "mx");
        }
    }
    {
        weights::WeightLoadOptions no_pre;
        no_pre.preshuffle = false;
        auto w = weights::load_quantized_matrix(good, "mx", arena, stream, no_pre);
        if (!w.ok()) { fail("load mx canonical: " + w.status().message()); }
        else {
            weights::MatrixWeight m = w.release();
            check(!m.preshuffled, "mx canonical not preshuffled");
            check_eq_u64(m.codes_row_stride_bytes, 64, "mx canonical codes stride");
            check_eq_u64(m.storage_scale_stride_bytes, 4, "mx canonical scale stride");
            check_matrix_bytes(m, m_mx.codes, m_mx.metadata1, "mx canonical");
        }
    }
    {
        weights::WeightLoadOptions no_pre;
        no_pre.preshuffle = false;
        auto w = weights::load_quantized_matrix(good, "q4", arena, stream, no_pre);
        if (!w.ok()) { fail("load q4 canonical: " + w.status().message()); }
        else {
            weights::MatrixWeight m = w.release();
            check(!m.preshuffled, "q4 canonical not preshuffled");
            check_eq_u64(m.k_padded, 64, "q4 canonical k_padded");
            check_eq_u64(m.codes_row_stride_bytes, 32, "q4 canonical codes stride");
            check_eq_u64(m.storage_scale_stride_bytes, 4, "q4 canonical scale stride");
            check_matrix_bytes(m, m_psq4.codes, m_psq4.metadata1, "q4 canonical");
        }
    }
    {
        auto w = weights::load_quantized_matrix(good, "bf", arena, stream, weights::WeightLoadOptions{});
        if (!w.ok()) { fail("load bf: " + w.status().message()); }
        else {
            weights::MatrixWeight m = w.release();
            check_eq_u64(static_cast<uint32_t>(m.encoding), static_cast<uint32_t>(weights::MatrixEncoding::Bf16), "bf encoding");
            check_eq_u64(m.rows, 8, "bf rows");
            check_eq_u64(m.cols, 24, "bf cols");
            check_eq_u64(m.k_padded, 24, "bf k_padded");
            check(!m.preshuffled, "bf not preshuffled");
            check(m.quant_spec == QuantSpecId::BF16_V1, "bf quant_spec");
            const auto got = d2h(m.bf16.data<void>(), m_bf.codes.size());
            check(got == m_bf.codes, "bf bytes");
        }
    }
    {
        auto tv = weights::load_quantized_small(good, "vec", arena, stream);
        if (!tv.ok()) { fail("load small vec: " + tv.status().message()); }
        else {
            const gpu::Tensor v = tv.release();
            check_eq_u64(v.ndim(), 1, "vec ndim");
            const auto got = d2h(v.data<void>(), small.bytes.size());
            check(got == small.bytes, "vec bytes");
        }
        auto ta = weights::load_quantized_small(good, "vec_alias", arena, stream);
        check(ta.ok(), "vec alias resolve");
        auto miss = weights::load_quantized_small(good, "nope", arena, stream);
        check(!miss.ok(), "missing tensor must fail");
    }
}

static std::vector<float> make_psq_source(uint64_t rows, uint64_t k) {
    std::mt19937 rng(static_cast<uint32_t>(rows * 131 + k));
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    std::vector<float> source(rows * k);
    for (float& f : source) f = dist(rng);
    return source;
}

static quantization::psq::CanonicalQuantStore quantize_source_rows(
    quantization::QuantFormatId id, const std::vector<float>& source,
    uint64_t rows, uint64_t k, uint64_t src_k, uint64_t row_begin, uint64_t col_begin) {
    quantization::psq::CanonicalQuantStore store;
    store.init(id, rows, k, k);
    std::vector<float> row(k);
    for (uint64_t r = 0; r < rows; ++r) {
        std::memcpy(row.data(), source.data() + (row_begin + r) * src_k + col_begin,
                    k * sizeof(float));
        if (!store.quantize_row(static_cast<uint32_t>(r),
                                std::span<const float>(row.data(), k))) {
            fail("reference quantize_row");
        }
    }
    if (!store.view().validate()) fail("reference canonical validate");
    return store;
}

static void test_partition_plan_load(gpu::GpuArena& arena, hipStream_t stream) {
    using namespace quantization;
    using fpx::QuantizedEncoding;
    const fs::path dir = make_case_dir("partition_plan");

    QMatrixPlan m_psq4 = make_matrix(QuantizedEncoding::Psq4, 32, 64, "q4");
    QMatrixPlan m_psq8 = make_matrix(QuantizedEncoding::Psq8, 32, 64, "q8");
    QMatrixPlan m_bf = make_matrix(QuantizedEncoding::Bf16, 8, 16, "bf");

    fpx::QuantizedManifest manifest;
    manifest.architecture = "qwen35_dense";
    manifest.preset = "psq";
    manifest.source_dtype = "BF16";

    std::vector<Planned> shard;
    const auto add_matrix = [&](const std::string& logical, QMatrixPlan& plan) {
        for (auto& p : plan.physical) {
            Planned pl;
            pl.name = p.physical;
            pl.dtype = p.dtype;
            pl.shape = p.shape;
            pl.bytes = p.bytes;
            shard.push_back(std::move(pl));
        }
        manifest.tensors[logical] = plan.meta;
    };
    add_matrix("q4", m_psq4);
    add_matrix("q8", m_psq8);
    add_matrix("bf", m_bf);

    auto ms = fpx::serialize_quantized_manifest(manifest);
    if (!ms.ok()) { fail("partition plan serialize: " + ms.status().message()); return; }
    if (!write_shard((dir / "model.safetensors").string(), shard,
                     {{"phaseshift.format", fpx::kQuantizedSafetensorsFormat},
                      {"phaseshift.format_version",
                       std::to_string(fpx::kQuantizedSafetensorsFormatVersion)},
                      {"phaseshift.quantization_metadata", fpx::kQuantizedMetadataFile}})) {
        return;
    }
    write_text((dir / fpx::kQuantizedMetadataFile).string(), ms.value());

    auto rr = fpx::QuantizedModelReader::open(dir.string(), true);
    if (!rr.ok()) { fail("partition plan reader open: " + rr.status().message()); return; }
    fpx::QuantizedModelReader reader = rr.release();
    check(weights::validate_quantized_model(reader).ok(), "partition plan model validates");

    const auto make_desc = [](int32_t axis, std::vector<int64_t> shape,
                              std::vector<weights::TensorPartitionRange> ranges) {
        weights::TensorPartitionDesc d;
        d.axis = axis;
        d.index = 0;
        d.count = 2;
        d.global_shape = std::move(shape);
        d.ranges = std::move(ranges);
        return d;
    };
    weights::TensorPartitionPlan plan;
    plan.tp_size = 2;
    plan.tp_rank = 0;
    plan.tensors["q4"] = make_desc(0, {32, 64}, {{0, 16}});
    plan.tensors["q8"] = make_desc(1, {32, 64}, {{0, 32}});
    plan.tensors["bf"] = make_desc(0, {8, 16}, {{0, 4}});
    check(weights::validate_tensor_partition_plan(plan).ok(), "load plan validates");

    const std::vector<float> source = make_psq_source(32, 64);
    quantization::psq::CanonicalQuantStore ref_q4 =
        quantize_source_rows(QuantFormatId::Psq4, source, 16, 64, 64, 0, 0);
    quantization::psq::CanonicalQuantStore ref_q8 =
        quantize_source_rows(QuantFormatId::Psq8, source, 32, 32, 64, 0, 0);
    const std::vector<uint8_t> bf_source = bf16_bytes(8 * 16, 8 * 7 + 16);
    const std::vector<uint8_t> ref_bf(bf_source.begin(), bf_source.begin() + 4 * 16 * 2);

    {
        weights::WeightLoadOptions opts;
        opts.partition_plan = &plan;
        auto wr = weights::load_quantized_matrix(reader, "q4", arena, stream, opts);
        if (!wr.ok()) { fail("partition q4 load: " + wr.status().message()); }
        else {
            weights::MatrixWeight m = wr.release();
            check_eq_u64(m.rows, 16, "partition q4 local rows");
            check_eq_u64(m.cols, 64, "partition q4 local cols");
            check_eq_u64(m.k_padded, 64, "partition q4 local k_padded");
            check_eq_u64(m.weight_scale_group, 32, "partition q4 wsg");
            check(m.preshuffled, "partition q4 preshuffled");
            check(m.partition.has_value(), "partition q4 keeps partition");
            if (m.partition) {
                check(*m.partition == plan.tensors.at("q4"), "partition q4 equals plan");
            }
            const auto n = expect_native(QuantFormatId::Psq4, ref_q4.codes,
                                         ref_q4.metadata1, 16, 64);
            check_matrix_bytes(m, n.codes, n.metadata1, "partition q4 native");
        }
    }
    {
        weights::WeightLoadOptions opts;
        opts.partition_plan = &plan;
        opts.preshuffle = false;
        auto wr = weights::load_quantized_matrix(reader, "q4", arena, stream, opts);
        if (!wr.ok()) { fail("partition q4 canonical load: " + wr.status().message()); }
        else {
            weights::MatrixWeight m = wr.release();
            check(!m.preshuffled, "partition q4 canonical not preshuffled");
            check_eq_u64(m.rows, 16, "partition q4 canonical local rows");
            check_eq_u64(m.codes_row_stride_bytes, 32, "partition q4 canonical stride");
            check_matrix_bytes(m, ref_q4.codes, ref_q4.metadata1,
                               "partition q4 canonical");
        }
    }
    {
        weights::WeightLoadOptions opts;
        opts.partition_plan = &plan;
        auto wr = weights::load_quantized_matrix(reader, "q8", arena, stream, opts);
        if (!wr.ok()) { fail("partition q8 load: " + wr.status().message()); }
        else {
            weights::MatrixWeight m = wr.release();
            check_eq_u64(m.rows, 32, "partition q8 local rows");
            check_eq_u64(m.cols, 32, "partition q8 local cols");
            check_eq_u64(m.k_padded, 32, "partition q8 local k_padded");
            check(m.preshuffled, "partition q8 preshuffled");
            check(m.partition.has_value(), "partition q8 keeps partition");
            if (m.partition) {
                check(*m.partition == plan.tensors.at("q8"), "partition q8 equals plan");
            }
            const auto n = expect_native(QuantFormatId::Psq8, ref_q8.codes,
                                         ref_q8.metadata1, 32, 32);
            check_matrix_bytes(m, n.codes, n.metadata1, "partition q8 native");
        }
    }
    {
        weights::WeightLoadOptions opts;
        opts.partition_plan = &plan;
        opts.preshuffle = false;
        auto wr = weights::load_quantized_matrix(reader, "bf", arena, stream, opts);
        if (!wr.ok()) { fail("partition bf load: " + wr.status().message()); }
        else {
            weights::MatrixWeight m = wr.release();
            check_eq_u64(m.rows, 4, "partition bf local rows");
            check_eq_u64(m.cols, 16, "partition bf local cols");
            check_eq_u64(m.k_padded, 16, "partition bf local k_padded");
            check(!m.preshuffled, "partition bf not preshuffled");
            check(m.partition.has_value(), "partition bf keeps partition");
            const auto got = d2h(m.bf16.data<void>(), ref_bf.size());
            check(got == ref_bf, "partition bf bytes");
        }
    }
    {
        weights::WeightLoadOptions opts;
        auto wr = weights::load_quantized_matrix(reader, "q4", arena, stream, opts);
        if (!wr.ok()) { fail("global q4 load: " + wr.status().message()); }
        else {
            weights::MatrixWeight m = wr.release();
            check_eq_u64(m.rows, 32, "plan nullptr keeps global rows");
            check(!m.partition.has_value(), "plan nullptr keeps no partition");
        }
    }
    {
        weights::TensorPartitionPlan other;
        other.tp_size = 2;
        other.tp_rank = 0;
        other.tensors["q4"] = make_desc(0, {32, 64}, {{0, 16}});
        weights::WeightLoadOptions opts;
        opts.partition_plan = &other;
        auto wr = weights::load_quantized_matrix(reader, "q8", arena, stream, opts);
        if (!wr.ok()) { fail("unplanned q8 load: " + wr.status().message()); }
        else {
            weights::MatrixWeight m = wr.release();
            check_eq_u64(m.cols, 64, "tensor without plan entry stays global");
            check(!m.partition.has_value(), "tensor without plan entry has no partition");
        }
    }
}

static void test_qwen35_quantized_model_load(gpu::GpuArena& arena, hipStream_t stream) {
    using namespace quantization;
    using fpx::QuantizedEncoding;
    const fs::path dir = make_case_dir("quantized_model");

    std::vector<Planned> shard;
    fpx::QuantizedManifest manifest;
    manifest.architecture = "qwen35_dense";
    manifest.preset = "psq";
    manifest.source_dtype = "BF16";

    const auto add_matrix = [&](const std::string& logical, QMatrixPlan plan) {
        for (auto& p : plan.physical) {
            Planned pl;
            pl.name = p.physical;
            pl.dtype = p.dtype;
            pl.shape = p.shape;
            pl.bytes = p.bytes;
            shard.push_back(std::move(pl));
        }
        manifest.tensors[logical] = plan.meta;
    };
    const auto add_small = [&](const std::string& logical, std::vector<std::size_t> shape) {
        const uint64_t n = numel(shape);
        QTensor t = make_small(n, logical, shape);
        Planned pl;
        pl.name = t.physical;
        pl.dtype = t.dtype;
        pl.shape = t.shape;
        pl.bytes = t.bytes;
        shard.push_back(std::move(pl));
        manifest.tensors[logical] = small_meta(t, "norm");
    };

    const auto L = [](const char* s) { return std::string("model.language_model.") + s; };

    add_small(L("layers.0.input_layernorm.weight"), {32});
    add_small(L("layers.0.post_attention_layernorm.weight"), {32});
    add_matrix(L("layers.0.mlp.gate_proj.weight"), make_matrix(QuantizedEncoding::Psq4, 32, 32, "L0_g"));
    add_matrix(L("layers.0.mlp.up_proj.weight"), make_matrix(QuantizedEncoding::Psq4, 32, 32, "L0_u"));
    add_matrix(L("layers.0.mlp.down_proj.weight"), make_matrix(QuantizedEncoding::Psq8, 32, 32, "L0_d"));
    add_matrix(L("layers.0.linear_attn.in_proj_a.weight"), make_matrix(QuantizedEncoding::Bf16, 8, 32, "L0_ia"));
    add_matrix(L("layers.0.linear_attn.in_proj_b.weight"), make_matrix(QuantizedEncoding::Bf16, 8, 32, "L0_ib"));
    add_matrix(L("layers.0.linear_attn.in_proj_qkv.weight"), make_matrix(QuantizedEncoding::Psq4, 24, 32, "L0_iqkv"));
    add_matrix(L("layers.0.linear_attn.in_proj_z.weight"), make_matrix(QuantizedEncoding::Bf16, 8, 32, "L0_iz"));
    add_small(L("layers.0.linear_attn.conv1d.weight"), {8, 1, 4});
    add_matrix(L("layers.0.linear_attn.out_proj.weight"), make_matrix(QuantizedEncoding::Psq8, 32, 32, "L0_op"));
    add_small(L("layers.0.linear_attn.norm.weight"), {32});
    add_small(L("layers.0.linear_attn.dt_bias"), {8});
    add_small(L("layers.0.linear_attn.A_log"), {8});

    add_small(L("layers.1.input_layernorm.weight"), {32});
    add_small(L("layers.1.post_attention_layernorm.weight"), {32});
    add_matrix(L("layers.1.mlp.gate_proj.weight"), make_matrix(QuantizedEncoding::Psq4, 32, 32, "L1_g"));
    add_matrix(L("layers.1.mlp.up_proj.weight"), make_matrix(QuantizedEncoding::Psq4, 32, 32, "L1_u"));
    add_matrix(L("layers.1.mlp.down_proj.weight"), make_matrix(QuantizedEncoding::Psq8, 32, 32, "L1_d"));
    add_matrix(L("layers.1.self_attn.q_proj.weight"), make_matrix(QuantizedEncoding::Psq4, 32, 32, "L1_q"));
    add_matrix(L("layers.1.self_attn.k_proj.weight"), make_matrix(QuantizedEncoding::Psq8, 8, 32, "L1_k"));
    add_matrix(L("layers.1.self_attn.v_proj.weight"), make_matrix(QuantizedEncoding::Psq8, 8, 32, "L1_v"));
    add_matrix(L("layers.1.self_attn.o_proj.weight"), make_matrix(QuantizedEncoding::Psq4, 32, 32, "L1_o"));
    add_small(L("layers.1.self_attn.q_norm.weight"), {32});
    add_small(L("layers.1.self_attn.k_norm.weight"), {32});

    add_matrix(L("embed_tokens.weight"), make_matrix(QuantizedEncoding::Bf16, 40, 32, "embed"));
    add_matrix(L("lm_head.weight"), make_matrix(QuantizedEncoding::Bf16, 40, 32, "lmhead"));
    add_small(L("norm.weight"), {32});

    add_small("mtp.pre_fc_norm_embedding.weight", {32});
    add_small("mtp.pre_fc_norm_hidden.weight", {32});
    add_small("mtp.norm.weight", {32});
    add_matrix("mtp.fc.weight", make_matrix(QuantizedEncoding::Bf16, 32, 64, "mtp_fc"));
    add_small("mtp.layers.0.input_layernorm.weight", {32});
    add_small("mtp.layers.0.post_attention_layernorm.weight", {32});
    add_matrix("mtp.layers.0.mlp.gate_proj.weight", make_matrix(QuantizedEncoding::Psq4, 32, 32, "mtp_e_g"));
    add_matrix("mtp.layers.0.mlp.up_proj.weight", make_matrix(QuantizedEncoding::Psq4, 32, 32, "mtp_e_u"));
    add_matrix("mtp.layers.0.mlp.down_proj.weight", make_matrix(QuantizedEncoding::Psq8, 32, 32, "mtp_e_d"));
    add_matrix("mtp.layers.0.self_attn.q_proj.weight", make_matrix(QuantizedEncoding::Psq4, 32, 32, "mtp_e_q"));
    add_matrix("mtp.layers.0.self_attn.k_proj.weight", make_matrix(QuantizedEncoding::Psq8, 8, 32, "mtp_e_k"));
    add_matrix("mtp.layers.0.self_attn.v_proj.weight", make_matrix(QuantizedEncoding::Psq8, 8, 32, "mtp_e_v"));
    add_matrix("mtp.layers.0.self_attn.o_proj.weight", make_matrix(QuantizedEncoding::Psq4, 32, 32, "mtp_e_o"));
    add_small("mtp.layers.0.self_attn.q_norm.weight", {32});
    add_small("mtp.layers.0.self_attn.k_norm.weight", {32});

    auto ms = fpx::serialize_quantized_manifest(manifest);
    if (!ms.ok()) { fail("serialize model manifest"); return; }
    if (!write_shard((dir / "model.safetensors").string(), shard,
                     {{"phaseshift.format", fpx::kQuantizedSafetensorsFormat},
                      {"phaseshift.format_version", std::to_string(fpx::kQuantizedSafetensorsFormatVersion)},
                      {"phaseshift.quantization_metadata", fpx::kQuantizedMetadataFile}})) {
        return;
    }
    write_text((dir / fpx::kQuantizedMetadataFile).string(), ms.value());
    write_text((dir / "config.json").string(), config_json(2, "\"linear_attention\",\"full_attention\""));

    check(weights::is_quantized_model_dir(dir.string()), "is_quantized_model_dir(model)");

    qwen35::Qwen35LoadOptions options;
    options.verify_quantized_payload_crc = true;
    auto rr = qwen35::load_qwen35_weights_from_safetensors(dir.string(), arena, stream, options);
    if (!rr.ok()) { fail("qwen35 quantized load: " + rr.status().message()); return; }
    qwen35::Qwen35ModelWeights w = rr.release();

    check_eq_u64(w.layers.size(), 2, "quant model layers");
    check(w.layers[0].is_gdn, "quant layer0 gdn");
    check(!w.layers[1].is_gdn, "quant layer1 attn");
    check(w.mtp.present, "quant mtp");
    check(!w.lm_head_tied, "quant lm_head untied");

    check_eq_u64(w.embed_tokens.rows, 40, "quant embed rows");
    check_eq_u64(w.embed_tokens.cols, 32, "quant embed cols");
    check(w.embed_tokens.encoding == weights::MatrixEncoding::Bf16, "quant embed encoding");

    check_eq_u64(static_cast<uint32_t>(w.layers[0].mlp_gate_proj.encoding),
                 static_cast<uint32_t>(weights::MatrixEncoding::Psq4), "quant gate encoding");
    check(w.layers[0].mlp_gate_proj.preshuffled, "quant gate preshuffled");
    check(w.layers[0].mlp_down_proj.compute_spec == quantization::ComputeSpecId::PSQ8_W8A8_F32_BF16,
          "quant down compute_spec");
    check(w.layers[0].attn_in_proj_a.encoding == weights::MatrixEncoding::Bf16, "in_proj_a bf16 matrix");
    check_eq_u64(static_cast<uint32_t>(w.layers[1].attn_q_proj.encoding),
                 static_cast<uint32_t>(weights::MatrixEncoding::Psq4), "quant q encoding");
    check(w.mtp.fc.encoding == weights::MatrixEncoding::Bf16, "mtp fc encoding bf16 (MTP not quantized)");
    check_eq_u64(w.mtp.fc.rows, 32, "mtp fc rows == H");
    check_eq_u64(w.mtp.fc.cols, 64, "mtp fc cols == 2H");

    {
        weights::TensorPartitionPlan guard_plan;
        guard_plan.tp_size = 2;
        guard_plan.tp_rank = 0;
        qwen35::Qwen35LoadOptions guard_options;
        guard_options.weights.partition_plan = &guard_plan;
        auto blocked = qwen35::load_qwen35_weights_from_safetensors(
            dir.string(), arena, stream, guard_options);
        check(!blocked.ok() && blocked.status().code() == Status::Code::unsupported,
              "qwen35 load path rejects a tensor partition plan");
    }

    {
        auto rdr = fpx::QuantizedModelReader::open(dir.string(), false);
        if (!rdr.ok()) { fail("reopen reader"); return; }
        auto view = rdr.value().resolve("model.language_model.layers.1.self_attn.q_proj.weight");
        if (!view.ok()) { fail("resolve q"); return; }
        const auto& qv = view.value();
        const std::vector<uint8_t> codes_src(static_cast<const uint8_t*>(qv.codes.data),
                                            static_cast<const uint8_t*>(qv.codes.data) + qv.codes.size);
        const std::vector<uint8_t> metadata1_src(static_cast<const uint8_t*>(qv.metadata1.data),
                                          static_cast<const uint8_t*>(qv.metadata1.data) + qv.metadata1.size);
        const auto n = expect_native(quantization::QuantFormatId::Psq4, codes_src, metadata1_src, 32, 32);
        check_matrix_bytes(w.layers[1].attn_q_proj, n.codes, n.metadata1, "quant layer1 q bytes");

        auto norm_view = rdr.value().resolve(L("norm.weight"));
        if (!norm_view.ok()) { fail("resolve final norm"); return; }
        const auto& nv = norm_view.value();
        const std::vector<uint8_t> norm_src(static_cast<const uint8_t*>(nv.data.data),
                                            static_cast<const uint8_t*>(nv.data.data) + nv.data.size);
        const auto got = d2h(w.final_norm_weight.data<void>(), norm_src.size());
        check(got == norm_src, "final norm bytes");
    }
}

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    auto ar = gpu::GpuArena::create(0, 512ull * 1024ull * 1024ull);
    if (!ar.ok()) { fail("arena"); return 1; }
    gpu::GpuArena arena = ar.release();

    std::error_code ec;
    g_root = fs::temp_directory_path(ec) / ("phaseshift_weight_load_" + std::to_string(::getpid()));
    fs::create_directories(g_root, ec);

    test_bf16_matrix_load(arena, stream);
    test_bf16_model_load(arena, stream);
    test_standalone_quantized_reader(arena, stream);
    test_partition_plan_load(arena, stream);
    test_qwen35_quantized_model_load(arena, stream);

    fs::remove_all(g_root, ec);
    arena.shutdown();
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));

    std::printf("test_weight_load: %s (fail=%d)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
