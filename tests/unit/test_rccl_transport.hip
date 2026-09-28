#include <phaseshift/runtime/parallel/rccl_transport.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
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

Status create_stream(hipStream_t* out) {
    hipError_t e = hipStreamCreate(out);
    if (e != hipSuccess)
        return Status::hip_error("hipStreamCreate", hipGetErrorString(e), __FILE__, __LINE__);
    return Status::make_ok();
}

Status destroy_stream(hipStream_t stream) {
    hipError_t e = hipStreamDestroy(stream);
    if (e != hipSuccess)
        return Status::hip_error("hipStreamDestroy", hipGetErrorString(e), __FILE__, __LINE__);
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
                      const char* name, int rank) {
    std::vector<uint32_t> host(static_cast<std::size_t>(words));
    hipError_t e = hipMemcpy(host.data(), device_words, words * sizeof(uint32_t),
                             hipMemcpyDeviceToHost);
    if (e != hipSuccess)
        return Status::hip_error("hipMemcpy D2H", hipGetErrorString(e), __FILE__, __LINE__);
    uint64_t mismatches = 0;
    for (uint64_t i = 0; i < words; ++i) {
        if (host[static_cast<std::size_t>(i)] != expected) ++mismatches;
    }
    if (mismatches != 0) {
        fail(name, "rank=" + std::to_string(rank) + " mismatches=" +
                       std::to_string(mismatches) + " of " + std::to_string(words));
    }
    return Status::make_ok();
}

struct RankStreams {
    std::vector<hipStream_t> streams;

    explicit RankStreams(const ParallelConfig& config) : streams(config.world_size(), nullptr) {
        for (uint32_t rank = 0; rank < config.world_size(); ++rank) {
            Status st = set_device(config.devices[rank]);
            if (!st.ok()) {
                fail("rank stream create", st.message());
                return;
            }
            Status cs = create_stream(&streams[rank]);
            if (!cs.ok()) fail("rank stream create", cs.message());
        }
    }

    void release(const ParallelConfig& config) {
        for (uint32_t rank = 0; rank < config.world_size(); ++rank) {
            if (streams[rank] == nullptr) continue;
            Status st = set_device(config.devices[rank]);
            if (st.ok()) st = destroy_stream(streams[rank]);
            streams[rank] = nullptr;
            check_status(st, "rank stream destroy");
        }
    }
};

struct DtypeCase {
    RcclDataType dtype;
    uint64_t elements_per_word;
    uint32_t rank0_pattern;
    uint32_t rank1_pattern;
    uint32_t expected_pattern;
};

const std::vector<DtypeCase>& all_reduce_cases() {
    static const std::vector<DtypeCase> cases = {
        {RcclDataType::FP32, 1, 0x3F800000u, 0x40000000u, 0x40400000u},
        {RcclDataType::BF16, 2, 0x3F803F80u, 0x40004000u, 0x40404040u},
    };
    return cases;
}

void test_all_reduce(const ParallelConfig& config) {
    auto created = RcclTransport::create(config);
    if (!created.ok()) {
        fail("rccl transport create (tensor)", created.status().message());
        return;
    }
    RcclTransport& transport = *created.value();
    RankStreams streams(config);

    const std::vector<uint64_t> sizes = {
        4ull << 10, 64ull << 10, 1ull << 20, 16ull << 20, 64ull << 20};

    for (const DtypeCase& dc : all_reduce_cases()) {
        for (uint64_t bytes : sizes) {
            const uint64_t words = bytes / sizeof(uint32_t);
            const uint64_t elements = words * dc.elements_per_word;
            std::vector<uint32_t*> send(2, nullptr);
            std::vector<uint32_t*> recv(2, nullptr);
            for (uint32_t rank = 0; rank < 2; ++rank) {
                check_status(set_device(config.devices[rank]), "set device (all_reduce)");
                check_status(alloc_words(&send[rank], words), "alloc send");
                check_status(alloc_words(&recv[rank], words), "alloc recv");
                check_status(fill_pattern(send[rank], words,
                                          rank == 0 ? dc.rank0_pattern : dc.rank1_pattern),
                             "fill send pattern");
            }

            std::vector<Status> results(2, Status::make_ok());
            std::vector<std::thread> workers;
            for (uint32_t rank = 0; rank < 2; ++rank) {
                workers.emplace_back([&, rank] {
                    Status st = set_device(config.devices[rank]);
                    if (!st.ok()) {
                        results[rank] = st;
                        return;
                    }
                    results[rank] = transport.all_reduce_sum(rank, CommGroup::Tensor,
                                                             send[rank], recv[rank], elements,
                                                             dc.dtype, streams.streams[rank]);
                    if (results[rank].ok()) results[rank] = sync_stream(streams.streams[rank]);
                });
            }
            for (auto& w : workers) w.join();

            for (uint32_t rank = 0; rank < 2; ++rank)
                check_status(results[rank], "all_reduce_sum");

            for (uint32_t rank = 0; rank < 2; ++rank) {
                check_status(set_device(config.devices[rank]), "set device (all_reduce verify)");
                if (results[rank].ok()) {
                    check_status(verify_pattern(recv[rank], words, dc.expected_pattern,
                                                "all_reduce value", static_cast<int>(rank)),
                                 "all_reduce verify");
                }
                check_status(free_words(send[rank]), "free send");
                check_status(free_words(recv[rank]), "free recv");
            }
        }
    }

    check_status(transport.shutdown(), "tensor transport shutdown");
    streams.release(config);
}

void test_send_recv(const ParallelConfig& config) {
    auto created = RcclTransport::create(config);
    if (!created.ok()) {
        fail("rccl transport create (pipeline)", created.status().message());
        return;
    }
    RcclTransport& transport = *created.value();
    RankStreams streams(config);

    const std::vector<RcclDataType> dtypes = {
        RcclDataType::FP32, RcclDataType::BF16, RcclDataType::FP16, RcclDataType::INT32};

    for (RcclDataType dtype : dtypes) {
        const uint64_t bytes = 64ull << 10;
        const uint64_t words = bytes / sizeof(uint32_t);
        const uint64_t elements = bytes / ps::runtime::rccl_dtype_bytes(dtype);
        std::vector<uint32_t*> buffer(2, nullptr);
        std::vector<uint32_t*> incoming(2, nullptr);
        for (uint32_t rank = 0; rank < 2; ++rank) {
            check_status(set_device(config.devices[rank]), "set device (send_recv)");
            check_status(alloc_words(&buffer[rank], words), "alloc send buffer");
            check_status(alloc_words(&incoming[rank], words), "alloc recv buffer");
            check_status(fill_pattern(buffer[rank], words,
                                      rank == 0 ? 0x40E00000u : 0x41100000u),
                         "fill send buffer");
            check_status(fill_pattern(incoming[rank], words, 0u), "clear recv buffer");
        }

        std::vector<Status> results(2, Status::make_ok());
        std::vector<std::thread> workers;
        for (uint32_t rank = 0; rank < 2; ++rank) {
            workers.emplace_back([&, rank] {
                Status st = set_device(config.devices[rank]);
                if (!st.ok()) {
                    results[rank] = st;
                    return;
                }
                const uint32_t peer = 1u - rank;
                if (rank == 0) {
                    results[rank] = transport.send(rank, CommGroup::Pipeline, peer,
                                                   buffer[rank], elements, dtype,
                                                   streams.streams[rank]);
                    if (results[rank].ok())
                        results[rank] = transport.recv(rank, CommGroup::Pipeline, peer,
                                                       incoming[rank], elements, dtype,
                                                       streams.streams[rank]);
                } else {
                    results[rank] = transport.recv(rank, CommGroup::Pipeline, peer,
                                                   incoming[rank], elements, dtype,
                                                   streams.streams[rank]);
                    if (results[rank].ok())
                        results[rank] = transport.send(rank, CommGroup::Pipeline, peer,
                                                       buffer[rank], elements, dtype,
                                                       streams.streams[rank]);
                }
                if (results[rank].ok()) results[rank] = sync_stream(streams.streams[rank]);
            });
        }
        for (auto& w : workers) w.join();

        for (uint32_t rank = 0; rank < 2; ++rank)
            check_status(results[rank], "send_recv");

        for (uint32_t rank = 0; rank < 2; ++rank) {
            check_status(set_device(config.devices[rank]), "set device (send_recv verify)");
            const uint32_t expected = rank == 0 ? 0x41100000u : 0x40E00000u;
            if (results[rank].ok()) {
                check_status(verify_pattern(incoming[rank], words, expected, "send_recv value",
                                            static_cast<int>(rank)),
                             "send_recv verify");
            }
            check_status(free_words(buffer[rank]), "free send buffer");
            check_status(free_words(incoming[rank]), "free recv buffer");
        }
    }

    check_status(transport.shutdown(), "pipeline transport shutdown");
    streams.release(config);
}

void test_repeated_iterations(const ParallelConfig& config) {
    auto created = RcclTransport::create(config);
    if (!created.ok()) {
        fail("rccl transport create (iterations)", created.status().message());
        return;
    }
    RcclTransport& transport = *created.value();
    RankStreams streams(config);

    constexpr int kIterations = 1000;
    const uint64_t words = (4ull << 10) / sizeof(uint32_t);
    std::vector<uint32_t*> send(2, nullptr);
    std::vector<uint32_t*> recv(2, nullptr);
    for (uint32_t rank = 0; rank < 2; ++rank) {
        check_status(set_device(config.devices[rank]), "set device (iterations)");
        check_status(alloc_words(&send[rank], words), "alloc send");
        check_status(alloc_words(&recv[rank], words), "alloc recv");
        check_status(fill_pattern(send[rank], words,
                                  rank == 0 ? 0x3F800000u : 0x40000000u),
                     "fill send");
    }

    std::vector<int> rccl_errors(2, 0);
    std::vector<int> data_errors(2, 0);
    std::vector<Status> results(2, Status::make_ok());
    std::vector<std::thread> workers;
    for (uint32_t rank = 0; rank < 2; ++rank) {
        workers.emplace_back([&, rank] {
            Status st = set_device(config.devices[rank]);
            if (!st.ok()) {
                results[rank] = st;
                return;
            }
            for (int it = 0; it < kIterations; ++it) {
                Status op = transport.all_reduce_sum(rank, CommGroup::Tensor, send[rank],
                                                     recv[rank], words, RcclDataType::FP32,
                                                     streams.streams[rank]);
                if (!op.ok()) {
                    ++rccl_errors[rank];
                    results[rank] = op;
                    break;
                }
                Status sy = sync_stream(streams.streams[rank]);
                if (!sy.ok()) {
                    ++rccl_errors[rank];
                    results[rank] = sy;
                    break;
                }
                Status ve = verify_pattern(recv[rank], words, 0x40400000u, "iteration value",
                                           static_cast<int>(rank));
                if (!ve.ok()) {
                    ++data_errors[rank];
                    results[rank] = ve;
                    break;
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    for (uint32_t rank = 0; rank < 2; ++rank) {
        check_status(results[rank], "repeated all_reduce");
        if (rccl_errors[rank] != 0)
            fail("repeated rccl errors", std::to_string(rccl_errors[rank]));
        if (data_errors[rank] != 0)
            fail("repeated data errors", std::to_string(data_errors[rank]));
    }
    check(transport.issued_operations() == static_cast<uint64_t>(kIterations) * 2u,
          "repeated operation count");

    for (uint32_t rank = 0; rank < 2; ++rank) {
        check_status(set_device(config.devices[rank]), "set device (iterations free)");
        check_status(free_words(send[rank]), "free send");
        check_status(free_words(recv[rank]), "free recv");
    }
    check_status(transport.shutdown(), "iteration transport shutdown");
    streams.release(config);
}

void test_device_guard(const ParallelConfig& config) {
    auto created = RcclTransport::create(config);
    if (!created.ok()) {
        fail("rccl transport create (guard)", created.status().message());
        return;
    }
    RcclTransport& transport = *created.value();

    uint32_t* send = nullptr;
    uint32_t* recv = nullptr;
    const uint64_t words = 16;
    check_status(set_device(config.devices[0]), "set device (guard)");
    check_status(alloc_words(&send, words), "alloc send (guard)");
    check_status(alloc_words(&recv, words), "alloc recv (guard)");
    check_status(fill_pattern(send, words, 0x3F800000u), "fill send (guard)");

    Status wrong_device = Status::make_ok();
    {
        std::thread worker([&] {
            Status st = set_device(config.devices[1]);
            if (!st.ok()) {
                wrong_device = st;
                return;
            }
            hipStream_t stream = nullptr;
            Status cs = create_stream(&stream);
            if (!cs.ok()) {
                wrong_device = cs;
                return;
            }
            wrong_device = transport.all_reduce_sum(0, CommGroup::Tensor, send, recv, words,
                                                    RcclDataType::FP32, stream);
            (void)destroy_stream(stream);
        });
        worker.join();
    }
    check(!wrong_device.ok(), "device guard rejects mismatched current device");
    if (!wrong_device.ok())
        check(wrong_device.code() == ps::Status::Code::invalid_state,
              "device guard reports invalid_state");

    check_status(set_device(config.devices[0]), "set device (guard free)");
    check_status(free_words(send), "free send (guard)");
    check_status(free_words(recv), "free recv (guard)");
    check_status(transport.shutdown(), "guard transport shutdown");
}

}

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) {
        std::printf("SKIP: need 2 GPUs, have %d\n", device_count);
        return 77;
    }

    ParallelConfig tensor_config;
    tensor_config.pp_size = 1;
    tensor_config.tp_size = 2;
    tensor_config.devices = {0, 1};
    Status vs = tensor_config.validate();
    if (!vs.ok()) {
        std::printf("SKIP: %s\n", vs.message().c_str());
        return 77;
    }

    int version = 0;
    if (ncclGetVersion(&version) != ncclSuccess) {
        std::printf("SKIP: ncclGetVersion failed\n");
        return 77;
    }
    std::printf("rccl version: %d\n", version);

    test_all_reduce(tensor_config);
    test_repeated_iterations(tensor_config);
    test_device_guard(tensor_config);

    ParallelConfig pipeline_config;
    pipeline_config.pp_size = 2;
    pipeline_config.tp_size = 1;
    pipeline_config.devices = {0, 1};
    test_send_recv(pipeline_config);

    if (g_failed != 0) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("rccl transport ok\n");
    return 0;
}
