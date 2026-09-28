#include <phaseshift/runtime/parallel/rccl_transport.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

using ps::Status;
using ps::runtime::CommGroup;
using ps::runtime::ParallelConfig;
using ps::runtime::RcclDataType;
using ps::runtime::RcclTransport;

int g_failed = 0;

void fail(const char* name, const std::string& detail) {
    std::printf("FAIL: %s: %s\n", name, detail.c_str());
    ++g_failed;
}

void check(bool cond, const char* name) {
    if (!cond) fail(name, "condition is false");
}

void check_status(const Status& st, const char* name) {
    if (!st.ok()) fail(name, st.message());
}

Status set_device(int device) {
    hipError_t e = hipSetDevice(device);
    if (e != hipSuccess)
        return Status::hip_error("hipSetDevice", hipGetErrorString(e), __FILE__, __LINE__);
    return Status::make_ok();
}

Status sync_stream(hipStream_t stream) {
    hipError_t e = hipStreamSynchronize(stream);
    if (e != hipSuccess)
        return Status::hip_error("hipStreamSynchronize", hipGetErrorString(e), __FILE__,
                                 __LINE__);
    return Status::make_ok();
}

Status alloc_words(uint32_t** out, uint64_t words) {
    hipError_t e = hipMalloc(reinterpret_cast<void**>(out), words * sizeof(uint32_t));
    if (e != hipSuccess)
        return Status::hip_error("hipMalloc", hipGetErrorString(e), __FILE__, __LINE__);
    return Status::make_ok();
}

Status free_words(uint32_t* ptr) {
    hipError_t e = hipFree(ptr);
    if (e != hipSuccess)
        return Status::hip_error("hipFree", hipGetErrorString(e), __FILE__, __LINE__);
    return Status::make_ok();
}

Status fill_pattern(uint32_t* device_words, uint64_t words, uint32_t pattern) {
    std::vector<uint32_t> host(static_cast<std::size_t>(words), pattern);
    hipError_t e = hipMemcpy(device_words, host.data(), words * sizeof(uint32_t),
                             hipMemcpyHostToDevice);
    if (e != hipSuccess)
        return Status::hip_error("hipMemcpy H2D", hipGetErrorString(e), __FILE__, __LINE__);
    return Status::make_ok();
}

Status verify_pattern(const uint32_t* device_words, uint64_t words, uint32_t expected,
                      const char* name, uint32_t rank) {
    std::vector<uint32_t> host(static_cast<std::size_t>(words));
    hipError_t e = hipMemcpy(host.data(), device_words, words * sizeof(uint32_t),
                             hipMemcpyDeviceToHost);
    if (e != hipSuccess)
        return Status::hip_error("hipMemcpy D2H", hipGetErrorString(e), __FILE__, __LINE__);
    uint64_t mismatches = 0;
    for (uint64_t i = 0; i < words; ++i) {
        if (host[static_cast<std::size_t>(i)] != expected) ++mismatches;
    }
    if (mismatches != 0)
        fail(name, "rank=" + std::to_string(rank) + " mismatches=" +
                       std::to_string(mismatches) + " of " + std::to_string(words));
    return Status::make_ok();
}

constexpr uint64_t kWords = (64ull << 10) / sizeof(uint32_t);

void test_tensor_groups(const ParallelConfig& config) {
    auto created = RcclTransport::create(config);
    if (!created.ok()) {
        fail("4gpu transport create", created.status().message());
        return;
    }
    RcclTransport& transport = *created.value();

    constexpr uint32_t kPatterns[4] = {0x3F800000u, 0x40000000u, 0x40400000u, 0x40800000u};
    constexpr uint32_t kExpected[4] = {0x40400000u, 0x40400000u, 0x40E00000u, 0x40E00000u};

    std::vector<hipStream_t> streams(config.world_size(), nullptr);
    std::vector<uint32_t*> send(config.world_size(), nullptr);
    std::vector<uint32_t*> recv(config.world_size(), nullptr);
    for (uint32_t rank = 0; rank < config.world_size(); ++rank) {
        check_status(set_device(config.devices[rank]), "set device (tensor setup)");
        hipError_t he = hipStreamCreate(&streams[rank]);
        check(he == hipSuccess, "tensor stream create");
        check_status(alloc_words(&send[rank], kWords), "alloc send");
        check_status(alloc_words(&recv[rank], kWords), "alloc recv");
        check_status(fill_pattern(send[rank], kWords, kPatterns[rank]), "fill send");
        check_status(fill_pattern(recv[rank], kWords, 0u), "clear recv");
    }

    std::vector<Status> results(config.world_size(), Status::make_ok());
    std::vector<std::thread> workers;
    for (uint32_t rank = 0; rank < config.world_size(); ++rank) {
        workers.emplace_back([&, rank] {
            Status st = set_device(config.devices[rank]);
            if (!st.ok()) {
                results[rank] = st;
                return;
            }
            results[rank] = transport.all_reduce_sum(rank, CommGroup::Tensor, send[rank],
                                                     recv[rank], kWords, RcclDataType::FP32,
                                                     streams[rank]);
            if (results[rank].ok()) results[rank] = sync_stream(streams[rank]);
        });
    }
    for (auto& w : workers) w.join();

    for (uint32_t rank = 0; rank < config.world_size(); ++rank) {
        check_status(results[rank], "4gpu tensor all_reduce");
        check_status(set_device(config.devices[rank]), "set device (tensor verify)");
        if (results[rank].ok())
            check_status(verify_pattern(recv[rank], kWords, kExpected[rank],
                                        "tensor group sum", rank),
                         "tensor group sum verify");
        check_status(free_words(send[rank]), "free send");
        check_status(free_words(recv[rank]), "free recv");
    }

    check_status(transport.shutdown(), "4gpu transport shutdown");
    for (uint32_t rank = 0; rank < config.world_size(); ++rank) {
        check_status(set_device(config.devices[rank]), "set device (stream destroy)");
        if (streams[rank] != nullptr) {
            hipError_t e = hipStreamDestroy(streams[rank]);
            if (e != hipSuccess)
                fail("stream destroy", hipGetErrorString(e));
        }
    }
}

void test_pipeline_lanes(const ParallelConfig& config) {
    auto created = RcclTransport::create(config);
    if (!created.ok()) {
        fail("4gpu pipeline transport create", created.status().message());
        return;
    }
    RcclTransport& transport = *created.value();

    std::vector<hipStream_t> streams(config.world_size(), nullptr);
    std::vector<uint32_t*> buffer(config.world_size(), nullptr);
    std::vector<uint32_t*> incoming(config.world_size(), nullptr);
    for (uint32_t rank = 0; rank < config.world_size(); ++rank) {
        check_status(set_device(config.devices[rank]), "set device (pipeline setup)");
        hipError_t he = hipStreamCreate(&streams[rank]);
        check(he == hipSuccess, "pipeline stream create");
        check_status(alloc_words(&buffer[rank], kWords), "alloc send buffer");
        check_status(alloc_words(&incoming[rank], kWords), "alloc recv buffer");
        check_status(fill_pattern(buffer[rank], kWords, 0x40000000u + rank * 0x00400000u),
                     "fill send buffer");
        check_status(fill_pattern(incoming[rank], kWords, 0u), "clear recv buffer");
    }

    std::vector<Status> results(config.world_size(), Status::make_ok());
    std::vector<std::thread> workers;
    for (uint32_t rank = 0; rank < config.world_size(); ++rank) {
        workers.emplace_back([&, rank] {
            Status st = set_device(config.devices[rank]);
            if (!st.ok()) {
                results[rank] = st;
                return;
            }
            const uint32_t pp = config.pp_rank_of(rank);
            const uint32_t peer = config.global_rank(1u - pp, config.tp_rank_of(rank));
            if (pp == 0) {
                results[rank] = transport.send(rank, CommGroup::Pipeline, peer, buffer[rank],
                                               kWords, RcclDataType::FP32, streams[rank]);
                if (results[rank].ok())
                    results[rank] = transport.recv(rank, CommGroup::Pipeline, peer,
                                                   incoming[rank], kWords, RcclDataType::FP32,
                                                   streams[rank]);
            } else {
                results[rank] = transport.recv(rank, CommGroup::Pipeline, peer, incoming[rank],
                                               kWords, RcclDataType::FP32, streams[rank]);
                if (results[rank].ok())
                    results[rank] = transport.send(rank, CommGroup::Pipeline, peer,
                                                   buffer[rank], kWords, RcclDataType::FP32,
                                                   streams[rank]);
            }
            if (results[rank].ok()) results[rank] = sync_stream(streams[rank]);
        });
    }
    for (auto& w : workers) w.join();

    for (uint32_t rank = 0; rank < config.world_size(); ++rank) {
        check_status(results[rank], "4gpu pipeline send/recv");
        check_status(set_device(config.devices[rank]), "set device (pipeline verify)");
        const uint32_t pp = config.pp_rank_of(rank);
        const uint32_t peer = config.global_rank(1u - pp, config.tp_rank_of(rank));
        const uint32_t expected = 0x40000000u + peer * 0x00400000u;
        if (results[rank].ok())
            check_status(verify_pattern(incoming[rank], kWords, expected, "pipeline transfer",
                                        rank),
                         "pipeline transfer verify");
        check_status(free_words(buffer[rank]), "free send buffer");
        check_status(free_words(incoming[rank]), "free recv buffer");
    }

    check_status(transport.shutdown(), "4gpu pipeline shutdown");
    for (uint32_t rank = 0; rank < config.world_size(); ++rank) {
        check_status(set_device(config.devices[rank]), "set device (stream destroy)");
        if (streams[rank] != nullptr) {
            hipError_t e = hipStreamDestroy(streams[rank]);
            if (e != hipSuccess) fail("stream destroy", hipGetErrorString(e));
        }
    }
}

void report(const ParallelConfig& config) {
    for (uint32_t rank = 0; rank < config.world_size(); ++rank) {
        std::printf("rank=%u device=%d pp_rank=%u tp_rank=%u\n", rank, config.devices[rank],
                    config.pp_rank_of(rank), config.tp_rank_of(rank));
    }
    for (uint32_t pp = 0; pp < config.pp_size; ++pp) {
        std::printf("tp_group[%u] devices=", pp);
        for (int d : config.tensor_group_devices(pp)) std::printf("%d ", d);
        std::printf("\n");
    }
    for (uint32_t tp = 0; tp < config.tp_size; ++tp) {
        std::printf("pp_lane[%u] devices=", tp);
        for (int d : config.pipeline_group_devices(tp)) std::printf("%d ", d);
        std::printf("\n");
    }
}

}

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 4) {
        std::printf("SKIP: need 4 GPUs, have %d\n", device_count);
        return 77;
    }

    ParallelConfig config;
    config.pp_size = 2;
    config.tp_size = 2;
    config.devices = {0, 1, 2, 3};
    Status vs = config.validate();
    if (!vs.ok()) {
        std::printf("SKIP: %s\n", vs.message().c_str());
        return 77;
    }
    report(config);

    test_tensor_groups(config);
    test_pipeline_lanes(config);

    if (g_failed != 0) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("4gpu rccl topology ok\n");
    return 0;
}
