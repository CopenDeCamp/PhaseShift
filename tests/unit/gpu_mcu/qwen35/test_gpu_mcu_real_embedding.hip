#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"
#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/embedding.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>
#include <phaseshift/runtime/program/device_program.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace {

namespace mcu = ps::runtime::gpu_mcu;
namespace kernel = ps::kernel;
namespace chain = gpu_mcu_chain;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kThreads = 256u;
constexpr uint32_t kGuardBytes = 64u;

struct EmbedCase {
    const char* name;
    uint32_t vocab;
    uint32_t hidden;
    uint32_t rows;
    uint32_t out_stride;
    std::vector<int32_t> tokens;
    bool invalid;
};

const EmbedCase kCases[] = {
    {"rows1", 64u, 2560u, 1u, 2560u, {7}, false},
    {"rows4", 64u, 2560u, 4u, 2560u, {0, 63, 5, 31}, false},
    {"edge0", 32u, 2560u, 2u, 2560u, {0, 31}, false},
    {"pad", 32u, 2560u, 4u, 2576u, {5, 2, 9, 0}, false},
    {"wide5120", 32u, 5120u, 3u, 5120u, {0, 31, 15}, false},
    {"neg", 32u, 128u, 4u, 128u, {0, -1, 2, 3}, true},
    {"upper", 32u, 128u, 4u, 128u, {0, 32, 2, 3}, true},
};
constexpr uint32_t kCaseCount =
    static_cast<uint32_t>(sizeof(kCases) / sizeof(kCases[0]));

uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

uint32_t h32(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<uint32_t>(x);
}

bool wait_drained(const mcu::GpuMcuAqlQueue& queue, uint64_t target,
                  uint32_t timeout_ms) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (queue.host_consumed_index() < target) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL: stream\n");
        return 1;
    }

    auto queue_res = mcu::GpuMcuAqlQueue::create(0, mcu::AqlQueueOwner::Host,
                                                 mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) {
        std::printf("FAIL: queue: %s\n", queue_res.status().message().c_str());
        return 1;
    }
    auto queue = queue_res.release();
    auto code_res = mcu::GpuMcuAqlCodeObject::load(
        queue, PHASESHIFT_GPU_MCU_EMBEDDING_HSACO,
        kernel::kEmbeddingBf16Symbol);
    if (!code_res.ok()) {
        std::printf("FAIL: embedding code object: %s\n",
                    code_res.status().message().c_str());
        return 1;
    }
    std::unique_ptr<mcu::GpuMcuAqlCodeObject> code(
        new mcu::GpuMcuAqlCodeObject(code_res.release()));
    const mcu::GpuAqlKernelMetadata meta = code->metadata();
    std::printf(
        "embedding hsaco: kernel_object=%p kernarg=%u sizeof(args)=%zu "
        "hidden_fits=%d\n",
        reinterpret_cast<void*>(meta.kernel_object), meta.kernarg_segment_size,
        sizeof(kernel::EmbeddingBf16Args),
        static_cast<int>(mcu::aql_hidden_args_fits(
            sizeof(kernel::EmbeddingBf16Args), meta.kernarg_segment_size)));
    check(meta.kernel_object != 0, "kernel_object resolved");
    check(meta.kernarg_segment_size >= sizeof(kernel::EmbeddingBf16Args),
          "kernarg segment covers EmbeddingBf16Args");

    const uint32_t slot_stride =
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size);
    auto region_res = queue.allocate_kernarg(slot_stride, kCaseCount);
    if (!region_res.ok()) {
        std::printf("FAIL: kernarg: %s\n", region_res.status().message().c_str());
        return 1;
    }
    auto region = region_res.release();

    constexpr uint32_t kMaxVocab = 64u;
    constexpr uint32_t kMaxHidden = 5120u;
    constexpr uint32_t kMaxRows = 4u;
    const size_t table_bytes =
        static_cast<size_t>(kMaxVocab) * kMaxHidden * 2u;
    std::vector<uint8_t> h_table(table_bytes, 0u);
    for (uint32_t t = 0; t < kMaxVocab; ++t) {
        for (uint32_t h = 0; h < kMaxHidden; ++h) {
            const float v = (static_cast<int32_t>(
                                 h32(static_cast<uint64_t>(t) * 2654435761u ^ h) %
                                 1001) -
                             500) /
                            500.0f;
            const uint16_t bits = bf16_bits(v);
            const size_t off = (static_cast<size_t>(t) * kMaxHidden + h) * 2u;
            h_table[off] = static_cast<uint8_t>(bits & 0xFFu);
            h_table[off + 1] = static_cast<uint8_t>(bits >> 8);
        }
    }

    void* table = nullptr;
    if (hipMalloc(&table, table_bytes) != hipSuccess ||
        hipMemcpy(table, h_table.data(), table_bytes, hipMemcpyHostToDevice) !=
            hipSuccess) {
        std::printf("FAIL: table upload\n");
        return 1;
    }

    std::vector<void*> tokens(kCaseCount, nullptr);
    std::vector<void*> host_out(kCaseCount, nullptr);
    std::vector<void*> aql_out(kCaseCount, nullptr);
    std::vector<void*> host_errors(kCaseCount, nullptr);
    std::vector<void*> aql_errors(kCaseCount, nullptr);

    int result = 0;
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const EmbedCase& c = kCases[i];
        const size_t tok_bytes = static_cast<size_t>(c.rows) * sizeof(int32_t);
        const size_t out_bytes = static_cast<size_t>(c.rows) * c.out_stride * 2u;
        if (hipMalloc(&tokens[i], tok_bytes) != hipSuccess ||
            hipMalloc(&host_out[i], out_bytes + kGuardBytes) != hipSuccess ||
            hipMalloc(&aql_out[i], out_bytes + kGuardBytes) != hipSuccess ||
            hipMalloc(&host_errors[i], sizeof(uint32_t)) != hipSuccess ||
            hipMalloc(&aql_errors[i], sizeof(uint32_t)) != hipSuccess) {
            std::printf("FAIL: alloc %s\n", c.name);
            return 1;
        }
        if (hipMemcpy(tokens[i], c.tokens.data(), tok_bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            std::printf("FAIL: token upload %s\n", c.name);
            return 1;
        }

        (void)hipMemset(host_out[i], 0xAB, out_bytes + kGuardBytes);
        (void)hipMemset(aql_out[i], 0xAB, out_bytes + kGuardBytes);
        (void)hipMemset(host_errors[i], 0, sizeof(uint32_t));
        (void)hipMemset(aql_errors[i], 0, sizeof(uint32_t));

        const hipError_t herr = kernel::launch_embedding_bf16(
            static_cast<const ps::bf16_t*>(table),
            static_cast<const int32_t*>(tokens[i]),
            static_cast<ps::bf16_t*>(host_out[i]), c.rows, c.vocab, c.hidden,
            c.out_stride, static_cast<uint32_t*>(host_errors[i]), stream);
        if (herr != hipSuccess || hipStreamSynchronize(stream) != hipSuccess) {
            std::printf("FAIL: host launch %s\n", c.name);
            result = 1;
            continue;
        }
        uint32_t h_err = 0;
        (void)hipMemcpy(&h_err, host_errors[i], sizeof(uint32_t),
                        hipMemcpyDeviceToHost);

        unsigned char* slot =
            region.base + static_cast<size_t>(i) * region.slot_stride;
        std::memset(slot, 0, region.slot_stride);
        kernel::EmbeddingBf16Args args{};
        args.table = static_cast<const ps::bf16_t*>(table);
        args.token_ids = static_cast<const int32_t*>(tokens[i]);
        args.output = static_cast<ps::bf16_t*>(aql_out[i]);
        args.error_word = static_cast<uint32_t*>(aql_errors[i]);
        args.rows = c.rows;
        args.vocab_size = c.vocab;
        args.hidden_size = c.hidden;
        args.output_row_stride = c.out_stride;
        std::memcpy(slot, &args, sizeof(args));

        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = meta.kernel_object;
        desc.global_work_items_x = c.rows * kThreads;
        desc.global_work_items_y = 1u;
        desc.global_work_items_z = 1u;
        desc.workgroup_size_x = static_cast<uint16_t>(kThreads);
        desc.workgroup_size_y = 1u;
        desc.workgroup_size_z = 1u;
        desc.private_segment_size = meta.private_segment_size;
        desc.group_segment_size = meta.static_group_segment_size;
        desc.kernarg_address = slot;
        mcu::build_aql_hidden_args(slot, sizeof(kernel::EmbeddingBf16Args),
                                   meta.kernarg_segment_size,
                                   mcu::AqlHiddenArgsPolicy::IfFits, desc);
        mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, c.rows,
                                       1u, 1u, kThreads);

        const mcu::GpuAqlPacketTemplate tmpl =
            mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
        check(queue.submit_host(&tmpl, 1).ok(), "host aql submit");
        check(wait_drained(queue, queue.host_queued_index(), 20000),
              "aql queue drained");

        std::vector<uint8_t> h_out(out_bytes + kGuardBytes, 0u);
        std::vector<uint8_t> a_out(out_bytes + kGuardBytes, 0u);
        (void)hipMemcpy(h_out.data(), host_out[i], out_bytes + kGuardBytes,
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(a_out.data(), aql_out[i], out_bytes + kGuardBytes,
                        hipMemcpyDeviceToHost);
        uint32_t a_err = 0;
        (void)hipMemcpy(&a_err, aql_errors[i], sizeof(uint32_t),
                        hipMemcpyDeviceToHost);

        size_t first = 0;
        size_t diffs = 0;
        if (c.invalid) {
            check(h_err == 3u && a_err == 3u, "invalid token error word");
            check(chain::bytes_equal(h_out.data(), a_out.data(),
                                     out_bytes + kGuardBytes, &first, &diffs),
                  "invalid token output byte exact");
        } else {
            check(chain::bytes_equal(h_out.data(), a_out.data(), out_bytes,
                                     &first, &diffs),
                  "host vs aql embedding byte exact");
            check(a_out[out_bytes] == 0xABu &&
                      a_out[out_bytes + kGuardBytes - 1u] == 0xABu,
                  "aql output sentinel intact");
        }
        std::printf("  [%s] v=%u h=%u rows=%u os=%u invalid=%d err=%u/%u mm=%zu\n",
                    c.name, c.vocab, c.hidden, c.rows, c.out_stride,
                    static_cast<int>(c.invalid), h_err, a_err, diffs);
    }

    (void)code->shutdown();
    check(queue.shutdown().ok(), "queue shutdown");

    std::vector<std::vector<uint8_t>> h_host(kCaseCount);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const size_t out_bytes =
            static_cast<size_t>(kCases[i].rows) * kCases[i].out_stride * 2u;
        h_host[i].assign(out_bytes + kGuardBytes, 0u);
        (void)hipMemcpy(h_host[i].data(), host_out[i], out_bytes + kGuardBytes,
                        hipMemcpyDeviceToHost);
    }

    using gpu_mcu_test::Rig;
    Rig rig;
    if (!rig.open(0, kCaseCount + 3u, kCaseCount + 1u, 8u, 0u, false,
                  mcu::kAqlDefaultQueueSize, mcu::kMcuMaxDebugRecords,
                  meta.kernarg_segment_size)) {
        std::printf("FAIL: rig open\n");
        return 1;
    }
    mcu::GpuAqlKernelMetadata mcu_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_EMBEDDING_HSACO,
                             kernel::kEmbeddingBf16Symbol, mcu_meta))
        return 1;

    std::vector<mcu::McuKernelVariantDesc> variants;
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        variants.push_back(mcu::make_kernel_variant(
            static_cast<uint16_t>(i), mcu_meta,
            mcu::kMcuKernargRecipeEmbeddingBf16, kThreads, kCases[i].rows,
            mcu::kMcuAgentAqlPolicy, mcu::AqlHiddenArgsPolicy::IfFits, 1u, 1u));
    }
    variants.push_back(mcu::make_probe_variant(static_cast<uint16_t>(kCaseCount),
                                               rig.meta, 64u, 1u,
                                               mcu::kMcuSystemAqlPolicy));

    std::vector<mcu::McuPlanNode> nodes(kCaseCount + 3u);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        nodes[i].variant_id = static_cast<uint16_t>(i);
        nodes[i].kernarg_recipe = mcu::kMcuKernargRecipeEmbeddingBf16;
        nodes[i].completion_slot = 0;
        nodes[i].invocation_index = i;
        nodes[i].flags = mcu::kMcuNodeDispatch;
        nodes[i].next = static_cast<uint16_t>(i + 1u);
    }
    nodes[kCaseCount].variant_id = static_cast<uint16_t>(kCaseCount);
    nodes[kCaseCount].kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    nodes[kCaseCount].completion_slot = 0;
    nodes[kCaseCount].input_slot = mcu::kMcuNoInput;
    nodes[kCaseCount].flags = mcu::kMcuNodeDispatch;
    nodes[kCaseCount].next = static_cast<uint16_t>(kCaseCount + 1u);
    nodes[kCaseCount + 1u].completion_slot = 0;
    nodes[kCaseCount + 1u].flags = mcu::kMcuNodeWait;
    nodes[kCaseCount + 1u].next = static_cast<uint16_t>(kCaseCount + 2u);
    nodes[kCaseCount + 2u].flags = mcu::kMcuNodeEnd;
    if (!rig.upload(nodes, variants)) return 1;

    std::vector<mcu::McuEmbeddingBf16Invocation> invocations(kCaseCount);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const EmbedCase& c = kCases[i];
        (void)hipMemset(aql_out[i], 0xAB,
                        static_cast<size_t>(c.rows) * c.out_stride * 2u +
                            kGuardBytes);
        (void)hipMemset(aql_errors[i], 0, sizeof(uint32_t));
        invocations[i] = mcu::McuEmbeddingBf16Invocation{};
        invocations[i].table = reinterpret_cast<uint64_t>(table);
        invocations[i].token_ids = reinterpret_cast<uint64_t>(tokens[i]);
        invocations[i].output = reinterpret_cast<uint64_t>(aql_out[i]);
        invocations[i].error_word = reinterpret_cast<uint64_t>(aql_errors[i]);
        invocations[i].rows = c.rows;
        invocations[i].vocab_size = c.vocab;
        invocations[i].hidden_size = c.hidden;
        invocations[i].output_row_stride = c.out_stride;
    }
    if (!rig.set_embedding_invocations(invocations.data(), kCaseCount))
        return 1;
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x454d4244u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.kernarg_slot_count = 0u;
    config.doorbell_mode = static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    if (!rig.start(config)) {
        std::printf("FAIL: fsm start\n");
        return 1;
    }
    check(rig.fsm.request_run().ok(), "mcu plan requested");
    check(rig.fsm.wait_plans(1u, 30000).ok(), "mcu plan completed");
    check(rig.drain_queue(30000), "queue drained");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.dispatches() == static_cast<uint64_t>(kCaseCount) + 1u,
          "dispatch count");

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const EmbedCase& c = kCases[i];
        const size_t out_bytes = static_cast<size_t>(c.rows) * c.out_stride * 2u;
        std::vector<uint8_t> m_out(out_bytes + kGuardBytes, 0u);
        rig.device_to_host(m_out.data(), aql_out[i], out_bytes + kGuardBytes);
        size_t first = 0;
        size_t diffs = 0;
        check(chain::bytes_equal(h_host[i].data(), m_out.data(),
                                 out_bytes + kGuardBytes, &first, &diffs),
              "mcu embedding matches the host launch");
        if (c.invalid) {
            uint32_t m_err = 0;
            rig.device_to_host(&m_err, aql_errors[i], sizeof(uint32_t));
            check(m_err == 3u, "mcu invalid token error word");
        }
        std::printf("  fsm[%s] mm=%zu\n", c.name, diffs);
    }
    check(rig.close(), "rig teardown");

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        if (tokens[i]) (void)hipFree(tokens[i]);
        if (host_out[i]) (void)hipFree(host_out[i]);
        if (aql_out[i]) (void)hipFree(aql_out[i]);
        if (host_errors[i]) (void)hipFree(host_errors[i]);
        if (aql_errors[i]) (void)hipFree(aql_errors[i]);
    }
    (void)hipFree(table);
    (void)hipStreamDestroy(stream);

    if (g_failed != 0 || result != 0) {
        std::printf("test_gpu_mcu_real_embedding: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_embedding: PASS\n");
    return 0;
}
