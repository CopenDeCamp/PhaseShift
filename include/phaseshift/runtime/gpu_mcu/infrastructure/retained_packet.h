#pragma once

#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>

#include <cstddef>
#include <cstdint>

namespace ps::runtime::gpu_mcu {

struct alignas(16) GpuMcuRetainedPacket {
    uint32_t publish_dw0 = 0;
    uint32_t at4 = 0;
    uint64_t at8 = 0;
    AqlU32x4 at16{};
    AqlU32x4 at32{};
    AqlU32x4 at48{};
};

static_assert(sizeof(GpuMcuRetainedPacket) == kAqlPacketBytes);
static_assert(alignof(GpuMcuRetainedPacket) == 16);
static_assert(offsetof(GpuMcuRetainedPacket, publish_dw0) == kAqlDw0Offset);
static_assert(offsetof(GpuMcuRetainedPacket, at4) == kAqlOffWgX);
static_assert(offsetof(GpuMcuRetainedPacket, at8) == 8);
static_assert(offsetof(GpuMcuRetainedPacket, at16) == 16);
static_assert(offsetof(GpuMcuRetainedPacket, at32) == 32);
static_assert(offsetof(GpuMcuRetainedPacket, at48) == 48);

inline constexpr uint32_t kRetainedPatchKernargAddress = kAqlOffKernargAddress;

GpuMcuRetainedPacket make_retained_packet(const GpuAqlPacketTemplate& templ);

__device__ __forceinline__ void expand_retained_packet(
    unsigned char* packet,
    const GpuMcuRetainedPacket& templ,
    uint64_t kernarg_address,
    uint32_t workgroup_count_x = 0u,
    uint32_t workgroup_count_y = 0u,
    uint32_t workgroup_count_z = 0u) {
    AqlU32x4 at32 = templ.at32;
    if (kernarg_address != 0u) {
        at32.z = static_cast<uint32_t>(kernarg_address);
        at32.w = static_cast<uint32_t>(kernarg_address >> 32);
    }
    uint64_t at8 = templ.at8;
    AqlU32x4 at16 = templ.at16;
    if (workgroup_count_x != 0u) {
        const uint32_t workgroup_x = templ.at4 & 0xffffu;
        at8 = (at8 & 0xffffffffull) |
              (static_cast<uint64_t>(workgroup_count_x * workgroup_x) << 32);
    }
    if (workgroup_count_y != 0u) {
        const uint32_t workgroup_y = (templ.at4 >> 16) & 0xffffu;
        at16.x = workgroup_count_y * workgroup_y;
    }
    if (workgroup_count_z != 0u) {
        const uint32_t workgroup_z = static_cast<uint32_t>(templ.at8 & 0xffffu);
        at16.y = workgroup_count_z * workgroup_z;
    }
    __atomic_store_n(
        reinterpret_cast<uint32_t*>(packet + kAqlDw0Offset),
        kAqlInvalidDw0,
        __ATOMIC_RELAXED);
    __builtin_memcpy(packet + 4, &templ.at4, sizeof(templ.at4));
    __builtin_memcpy(packet + 8, &at8, sizeof(at8));
    __builtin_memcpy(packet + 16, &at16, sizeof(at16));
    __builtin_memcpy(packet + 32, &at32, sizeof(at32));
    __builtin_memcpy(packet + 48, &templ.at48, sizeof(templ.at48));
}

__device__ __forceinline__ GpuMcuStagedPacket gpu_mcu_aql_stage_retained_packet(
    const DeviceAqlQueueView& queue,
    uint64_t slot,
    const GpuMcuRetainedPacket& templ,
    uint64_t kernarg_address,
    uint32_t workgroup_count_x = 0u,
    uint32_t workgroup_count_y = 0u,
    uint32_t workgroup_count_z = 0u) {
    unsigned char* pkt = aql_packet_at(queue, slot);
    expand_retained_packet(pkt, templ, kernarg_address, workgroup_count_x,
                           workgroup_count_y, workgroup_count_z);
    GpuMcuStagedPacket staged{};
    staged.packet = pkt;
    staged.publish_dw0 = templ.publish_dw0;
    return staged;
}

}  // namespace ps::runtime::gpu_mcu
