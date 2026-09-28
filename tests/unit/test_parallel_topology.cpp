#include <phaseshift/runtime/parallel/parallel_config.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

using ps::Status;
using ps::runtime::ParallelConfig;

int g_failed = 0;

void check(bool cond, const char* name) {
    if (!cond) {
        std::printf("FAIL: %s\n", name);
        ++g_failed;
    }
}

void check_message(const Status& status, ps::Status::Code code, const char* name) {
    if (status.ok() || status.code() != code) {
        std::printf("FAIL: %s: %s\n", name,
                    status.ok() ? "unexpected success" : status.message().c_str());
        ++g_failed;
    }
}

ParallelConfig make(uint32_t pp, uint32_t tp, std::vector<int> devices) {
    ParallelConfig config;
    config.pp_size = pp;
    config.tp_size = tp;
    config.devices = std::move(devices);
    return config;
}

void test_valid_shapes() {
    check(make(1, 1, {0}).validate().ok(), "world_size 1 accepted");
    check(make(1, 2, {0, 1}).validate().ok(), "world_size 2 tp accepted");
    check(make(2, 1, {0, 1}).validate().ok(), "world_size 2 pp accepted");
    check(make(2, 2, {0, 1, 2, 3}).validate().ok(), "world_size 4 accepted");
}

void test_invalid_shapes() {
    check_message(make(0, 1, {0}).validate(), ps::Status::Code::invalid_argument,
                  "pp_size 0 rejected");
    check_message(make(1, 0, {0}).validate(), ps::Status::Code::invalid_argument,
                  "tp_size 0 rejected");
    check_message(make(1, 2, {0}).validate(), ps::Status::Code::invalid_argument,
                  "device count mismatch rejected");
    check_message(make(2, 2, {0, 1, 2}).validate(), ps::Status::Code::invalid_argument,
                  "device count mismatch (4 ranks) rejected");
    check_message(make(2, 2, {0, 1, 1, 3}).validate(), ps::Status::Code::invalid_argument,
                  "duplicate device rejected");
    check_message(make(2, 2, {0, 1, 2, -1}).validate(), ps::Status::Code::invalid_argument,
                  "negative device rejected");
    check_message(make(1, 3, {0, 1, 2}).validate(), ps::Status::Code::unsupported,
                  "world_size 3 rejected");
    check_message(make(2, 4, {0, 1, 2, 3, 4, 5, 6, 7}).validate(),
                  ps::Status::Code::unsupported, "world_size 8 rejected");
}

void test_rank_mapping() {
    const ParallelConfig config = make(2, 2, {0, 1, 2, 3});
    check(config.world_size() == 4u, "world_size 4");
    check(config.global_rank(0, 0) == 0u, "global_rank(0,0)");
    check(config.global_rank(0, 1) == 1u, "global_rank(0,1)");
    check(config.global_rank(1, 0) == 2u, "global_rank(1,0)");
    check(config.global_rank(1, 1) == 3u, "global_rank(1,1)");
    for (uint32_t rank = 0; rank < 4; ++rank) {
        const uint32_t pp = config.pp_rank_of(rank);
        const uint32_t tp = config.tp_rank_of(rank);
        check(config.global_rank(pp, tp) == rank, "rank mapping round trip");
    }
}

void test_group_devices() {
    const ParallelConfig config = make(2, 2, {10, 11, 12, 13});
    const std::vector<int> tp0 = config.tensor_group_devices(0);
    const std::vector<int> tp1 = config.tensor_group_devices(1);
    const std::vector<int> pp0 = config.pipeline_group_devices(0);
    const std::vector<int> pp1 = config.pipeline_group_devices(1);

    check(tp0 == std::vector<int>({10, 11}), "tensor group stage 0");
    check(tp1 == std::vector<int>({12, 13}), "tensor group stage 1");
    check(pp0 == std::vector<int>({10, 12}), "pipeline group lane 0");
    check(pp1 == std::vector<int>({11, 13}), "pipeline group lane 1");

    check(config.same_tensor_group(0, 1), "ranks 0/1 share tensor group");
    check(!config.same_tensor_group(0, 2), "ranks 0/2 do not share tensor group");
    check(config.same_pipeline_group(0, 2), "ranks 0/2 share pipeline group");
    check(!config.same_pipeline_group(0, 1), "ranks 0/1 do not share pipeline group");
    check(!config.same_tensor_group(0, 9), "out of range rank rejected");
}

void test_single_stage_groups() {
    const ParallelConfig tensor = make(1, 2, {5, 6});
    check(tensor.tensor_group_devices(0) == std::vector<int>({5, 6}), "tp-only tensor group");
    check(tensor.pipeline_group_devices(0) == std::vector<int>({5}), "tp-only lane 0");
    check(tensor.pipeline_group_devices(1) == std::vector<int>({6}), "tp-only lane 1");

    const ParallelConfig pipeline = make(2, 1, {5, 6});
    check(pipeline.tensor_group_devices(0) == std::vector<int>({5}), "pp-only stage 0");
    check(pipeline.tensor_group_devices(1) == std::vector<int>({6}), "pp-only stage 1");
    check(pipeline.pipeline_group_devices(0) == std::vector<int>({5, 6}), "pp-only lane 0");
}

void test_report() {
    const ParallelConfig config = make(2, 2, {0, 1, 2, 3});
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
        std::printf("pp_group[%u] devices=", tp);
        for (int d : config.pipeline_group_devices(tp)) std::printf("%d ", d);
        std::printf("\n");
    }
}

}

int main() {
    test_valid_shapes();
    test_invalid_shapes();
    test_rank_mapping();
    test_group_devices();
    test_single_stage_groups();
    test_report();

    if (g_failed != 0) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("parallel topology ok\n");
    return 0;
}
