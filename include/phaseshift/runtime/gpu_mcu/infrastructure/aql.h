#pragma once

#include <phaseshift/core/status.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ps::runtime::gpu_mcu {

enum class AqlFenceScope : uint8_t {
    None = 0,
    Agent = 1,
    System = 2,
};

struct AqlMemoryPolicy {
    AqlFenceScope acquire_scope = AqlFenceScope::System;
    AqlFenceScope release_scope = AqlFenceScope::System;
    bool producer_visibility_fence = true;
};

constexpr AqlMemoryPolicy kDefaultAqlPolicy{
    .acquire_scope = AqlFenceScope::System,
    .release_scope = AqlFenceScope::System,
    .producer_visibility_fence = true,
};

constexpr AqlMemoryPolicy kMcuAqlFastPolicy{
    .acquire_scope = AqlFenceScope::None,
    .release_scope = AqlFenceScope::None,
    .producer_visibility_fence = false,
};

enum class AqlQueueOwner : uint8_t {
    Host,
    GpuMcu,
};

constexpr uint32_t kAqlPacketTypeKernelDispatch = 2;
constexpr uint32_t kAqlDw0BarrierBit = 8;
constexpr uint32_t kAqlDw0AcquireBit = 9;
constexpr uint32_t kAqlDw0ReleaseBit = 11;
constexpr uint32_t kAqlDw0DimensionsBit = 16;
constexpr uint32_t kAqlInvalidDw0 = 1u;

constexpr uint32_t make_kernel_dispatch_dw0_ex(
    AqlFenceScope acquire,
    AqlFenceScope release,
    bool barrier,
    uint32_t dimensions = 1u) noexcept {
    return kAqlPacketTypeKernelDispatch |
           (barrier ? (1u << kAqlDw0BarrierBit) : 0u) |
           (static_cast<uint32_t>(acquire) << kAqlDw0AcquireBit) |
           (static_cast<uint32_t>(release) << kAqlDw0ReleaseBit) |
           ((dimensions & 3u) << kAqlDw0DimensionsBit);
}

constexpr uint32_t make_kernel_dispatch_dw0(
    AqlFenceScope acquire,
    AqlFenceScope release,
    uint32_t dimensions = 1u) noexcept {
    return make_kernel_dispatch_dw0_ex(acquire, release, true, dimensions);
}

constexpr uint32_t aql_dispatch_dimensions(uint32_t grid_y, uint32_t grid_z) noexcept {
    return grid_z > 1u ? 3u : (grid_y > 1u ? 2u : 1u);
}

static_assert(make_kernel_dispatch_dw0(AqlFenceScope::None, AqlFenceScope::None) == 0x10102u);
static_assert(make_kernel_dispatch_dw0(AqlFenceScope::Agent, AqlFenceScope::Agent) == 0x10b02u);
static_assert(make_kernel_dispatch_dw0(AqlFenceScope::System, AqlFenceScope::System) == 0x11502u);

constexpr uint32_t kAqlPacketBytes = 64;
constexpr uint32_t kAqlDw0Offset = 0;
constexpr uint32_t kAqlBodyBytes = 60;

constexpr uint32_t kAqlOffWgX = 4;
constexpr uint32_t kAqlOffWgY = 6;
constexpr uint32_t kAqlOffWgZ = 8;
constexpr uint32_t kAqlOffGridX = 12;
constexpr uint32_t kAqlOffGridY = 16;
constexpr uint32_t kAqlOffGridZ = 20;
constexpr uint32_t kAqlOffPrivateSegment = 24;
constexpr uint32_t kAqlOffGroupSegment = 28;
constexpr uint32_t kAqlOffKernelObject = 32;
constexpr uint32_t kAqlOffKernargAddress = 40;
constexpr uint32_t kAqlOffReserved1 = 48;
constexpr uint32_t kAqlOffCompletionSignal = 56;

constexpr uint32_t kAqlBatchMaxPackets = 64;
constexpr uint32_t kAqlDefaultQueueSize = 1024;

struct AqlCuMaskView {
    const uint32_t* words = nullptr;
    uint32_t bit_count = 0;
};

bool aql_cu_mask_is_wgp_paired(const uint32_t* words, uint32_t cu_count) noexcept;

Result<uint32_t> gpu_mcu_hsa_compute_unit_count(int device);

struct DeviceAqlQueueView {
    void* queue_ptr = nullptr;
    void* ring_base = nullptr;
    uint64_t doorbell_handle = 0;
    uint32_t size = 0;
    volatile uint64_t* write_index = nullptr;
    volatile uint64_t* read_index = nullptr;
    AqlQueueOwner owner = AqlQueueOwner::Host;
};

struct GpuAqlDispatchDesc {
    uint64_t kernel_object = 0;
    uint32_t global_work_items_x = 0;
    uint32_t global_work_items_y = 1;
    uint32_t global_work_items_z = 1;
    uint16_t workgroup_size_x = 0;
    uint16_t workgroup_size_y = 1;
    uint16_t workgroup_size_z = 1;
    uint16_t reserved0 = 0;
    uint32_t private_segment_size = 0;
    uint32_t group_segment_size = 0;
    void* kernarg_address = nullptr;
    uint64_t completion_signal = 0;
};

constexpr uint32_t aql_workgroup_count(uint32_t global_work_items,
                                       uint32_t workgroup_size) noexcept {
    return workgroup_size == 0u ? 0u : global_work_items / workgroup_size;
}

constexpr uint32_t aql_workgroup_remainder(uint32_t global_work_items,
                                           uint32_t workgroup_size) noexcept {
    return workgroup_size == 0u ? 0u : global_work_items % workgroup_size;
}

struct alignas(64) GpuAqlPacketTemplate {
    std::byte bytes[kAqlPacketBytes];
    uint32_t publish_dw0 = 0;
    uint32_t reserved = 0;
};

static_assert(sizeof(((GpuAqlPacketTemplate*)nullptr)->bytes) == kAqlPacketBytes);
static_assert(alignof(GpuAqlPacketTemplate) == kAqlPacketBytes);

GpuAqlPacketTemplate build_aql_packet_template(
    const GpuAqlDispatchDesc& desc,
    const AqlMemoryPolicy& policy);

struct alignas(16) AqlU32x4 {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
    uint32_t w = 0;
};

static_assert(sizeof(AqlU32x4) == 16);
static_assert(alignof(AqlU32x4) == 16);

struct alignas(16) AqlPacketBodyBundle {
    uint32_t at4 = 0;
    uint32_t pad0 = 0;
    uint64_t at8 = 0;
    AqlU32x4 at16{};
    AqlU32x4 at32{};
    AqlU32x4 at48{};
};

AqlPacketBodyBundle make_body_bundle(const std::byte image[kAqlPacketBytes]);

constexpr std::size_t aql_launch_metadata_offset(std::size_t kernarg_segment_size) noexcept {
    return (kernarg_segment_size + 15u) / 16u * 16u;
}

constexpr std::size_t aql_kernarg_slot_stride(std::size_t kernarg_segment_size) noexcept {
    return aql_launch_metadata_offset(kernarg_segment_size) + 16u;
}

constexpr uint32_t kAqlMetadataWorkgroupCountX = 0;
constexpr uint32_t kAqlMetadataWorkgroupCountY = 4;
constexpr uint32_t kAqlMetadataWorkgroupCountZ = 8;
constexpr uint32_t kAqlMetadataWorkgroupSizeX = 12;

__host__ __device__ inline void build_aql_launch_metadata(
    void* kernarg_slot,
    std::size_t kernarg_segment_size,
    uint32_t workgroup_count_x,
    uint32_t workgroup_count_y,
    uint32_t workgroup_count_z,
    uint32_t workgroup_size_x) noexcept {
    auto* base = reinterpret_cast<unsigned char*>(kernarg_slot) +
                 aql_launch_metadata_offset(kernarg_segment_size);
    auto* meta = reinterpret_cast<uint32_t*>(base);
    meta[kAqlMetadataWorkgroupCountX / 4] = workgroup_count_x;
    meta[kAqlMetadataWorkgroupCountY / 4] = workgroup_count_y;
    meta[kAqlMetadataWorkgroupCountZ / 4] = workgroup_count_z;
    meta[kAqlMetadataWorkgroupSizeX / 4] = workgroup_size_x;
}

constexpr uint32_t kAqlHiddenBlockCountX = 0;
constexpr uint32_t kAqlHiddenBlockCountY = 4;
constexpr uint32_t kAqlHiddenBlockCountZ = 8;
constexpr uint32_t kAqlHiddenGroupSizeX = 12;
constexpr uint32_t kAqlHiddenGroupSizeY = 14;
constexpr uint32_t kAqlHiddenGroupSizeZ = 16;
constexpr uint32_t kAqlHiddenRemainderX = 18;
constexpr uint32_t kAqlHiddenRemainderY = 20;
constexpr uint32_t kAqlHiddenRemainderZ = 22;

constexpr std::size_t aql_hidden_args_offset(
    std::size_t explicit_args_bytes) noexcept {
    return (explicit_args_bytes + 7u) / 8u * 8u;
}

constexpr uint32_t kAqlHiddenArgsBytes = 24u;

enum class AqlHiddenArgsPolicy : uint32_t {
    None = 0,
    IfFits = 1,
    Required = 2,
};

constexpr std::size_t aql_hidden_args_end(
    std::size_t explicit_args_bytes) noexcept {
    return aql_hidden_args_offset(explicit_args_bytes) + kAqlHiddenArgsBytes;
}

constexpr bool aql_hidden_args_fits(std::size_t explicit_args_bytes,
                                    std::size_t kernarg_segment_size) noexcept {
    return aql_hidden_args_end(explicit_args_bytes) <= kernarg_segment_size;
}

__host__ __device__ inline bool build_aql_hidden_args(
    void* kernarg_slot,
    std::size_t explicit_args_bytes,
    std::size_t kernarg_segment_size,
    AqlHiddenArgsPolicy policy,
    const GpuAqlDispatchDesc& desc) noexcept {
    if (policy == AqlHiddenArgsPolicy::None) return false;
    if (!aql_hidden_args_fits(explicit_args_bytes, kernarg_segment_size)) {
        return false;
    }
    auto* base = reinterpret_cast<unsigned char*>(kernarg_slot) +
                 aql_hidden_args_offset(explicit_args_bytes);
    const uint32_t gx = desc.global_work_items_x;
    const uint32_t gy = desc.global_work_items_y;
    const uint32_t gz = desc.global_work_items_z;
    const uint32_t sx = desc.workgroup_size_x;
    const uint32_t sy = desc.workgroup_size_y;
    const uint32_t sz = desc.workgroup_size_z;
    auto* block_count = reinterpret_cast<uint32_t*>(base);
    block_count[kAqlHiddenBlockCountX / 4] = aql_workgroup_count(gx, sx);
    block_count[kAqlHiddenBlockCountY / 4] = aql_workgroup_count(gy, sy);
    block_count[kAqlHiddenBlockCountZ / 4] = aql_workgroup_count(gz, sz);
    auto* group_size = reinterpret_cast<uint16_t*>(base);
    group_size[kAqlHiddenGroupSizeX / 2] = static_cast<uint16_t>(sx);
    group_size[kAqlHiddenGroupSizeY / 2] = static_cast<uint16_t>(sy);
    group_size[kAqlHiddenGroupSizeZ / 2] = static_cast<uint16_t>(sz);
    auto* remainder = reinterpret_cast<uint16_t*>(base);
    remainder[kAqlHiddenRemainderX / 2] =
        static_cast<uint16_t>(aql_workgroup_remainder(gx, sx));
    remainder[kAqlHiddenRemainderY / 2] =
        static_cast<uint16_t>(aql_workgroup_remainder(gy, sy));
    remainder[kAqlHiddenRemainderZ / 2] =
        static_cast<uint16_t>(aql_workgroup_remainder(gz, sz));
    return true;
}

__device__ __forceinline__ unsigned char* aql_packet_at(
    const DeviceAqlQueueView& queue,
    uint64_t slot) {
    return static_cast<unsigned char*>(queue.ring_base) +
           static_cast<uint32_t>(slot % queue.size) * kAqlPacketBytes;
}

__device__ __forceinline__ void aql_set_packet_invalid(unsigned char* pkt) {
    __atomic_store_n(
        reinterpret_cast<uint32_t*>(pkt + kAqlDw0Offset),
        kAqlInvalidDw0,
        __ATOMIC_RELAXED);
}

__device__ __forceinline__ void copy_aql_packet_body(
    unsigned char* dst,
    const std::byte* src) {
    __builtin_memcpy(dst + 4, src + 4, kAqlBodyBytes);
}

__device__ __forceinline__ void construct_aql_packet_nonvolatile(
    unsigned char* pkt,
    const GpuAqlPacketTemplate& template_) {
    aql_set_packet_invalid(pkt);
    copy_aql_packet_body(pkt, template_.bytes);
}

__device__ __forceinline__ void aql_publish_dw0(
    unsigned char* pkt,
    uint32_t dw0) {
    __atomic_store_n(
        reinterpret_cast<uint32_t*>(pkt + kAqlDw0Offset),
        dw0,
        __ATOMIC_RELEASE);
}

struct AmdSignalDoorbell {
    long long kind;
    union {
        volatile long long value;
        volatile unsigned long long* hardware_doorbell_ptr;
    };
    unsigned long long event_mailbox_ptr;
    unsigned int event_id;
    unsigned int reserved1;
    unsigned long long start_ts;
    unsigned long long end_ts;
    union {
        void* queue_ptr;
        unsigned long long reserved2;
    };
    unsigned int reserved3[2];
};

__device__ __forceinline__ void ring_aql_doorbell(
    const DeviceAqlQueueView& queue,
    uint64_t last_valid_packet) {
    auto* sig = reinterpret_cast<AmdSignalDoorbell*>(queue.doorbell_handle);
    if (sig->kind == -1) {
        __atomic_store_n(
            reinterpret_cast<volatile unsigned int*>(sig->hardware_doorbell_ptr),
            static_cast<unsigned int>(last_valid_packet),
            __ATOMIC_RELEASE);
    } else {
        __atomic_store_n(
            &sig->value,
            static_cast<long long>(last_valid_packet),
            __ATOMIC_RELEASE);
    }
}

enum class GpuMcuAqlStatus : uint32_t {
    COMPLETE = 0,
    INVALID_PARAMS = 1,
    RESERVE_TIMEOUT = 2,
};

const char* gpu_mcu_aql_status_name(uint32_t status);

constexpr uint32_t kAqlQueueFullSpinLimit = 50000000u;
constexpr uint32_t kAqlQueueCaughtUpSpinLimit = 50000000u;

__device__ __forceinline__ bool aql_reserve_packets(
    const DeviceAqlQueueView& queue,
    uint32_t count,
    uint64_t* base_slot,
    uint32_t spin_limit) {
    if (count == 0 || count > queue.size) {
        return false;
    }
    uint32_t spins = 0;
    while ((*queue.write_index - *queue.read_index) + count > queue.size) {
        __builtin_amdgcn_s_sleep(8);
        if (++spins >= spin_limit) {
            return false;
        }
    }
    *base_slot = __atomic_fetch_add(
        queue.write_index,
        static_cast<unsigned long long>(count),
        __ATOMIC_SEQ_CST);
    return true;
}

__device__ __forceinline__ bool aql_wait_queue_caught_up(
    const DeviceAqlQueueView& queue,
    uint32_t spin_limit) {
    uint32_t spins = 0;
    while (*queue.read_index < *queue.write_index) {
        __builtin_amdgcn_s_sleep(8);
        if (++spins >= spin_limit) {
            return false;
        }
    }
    return true;
}

struct GpuMcuPublishBatchArgs {
    DeviceAqlQueueView queue{};
    const GpuAqlPacketTemplate* packet_templates = nullptr;
    uint32_t packet_count = 0;
};

__device__ __forceinline__ GpuMcuAqlStatus gpu_mcu_aql_publish_batch(
    const GpuMcuPublishBatchArgs& args) {
    if (args.queue.queue_ptr == nullptr || args.queue.ring_base == nullptr ||
        args.queue.write_index == nullptr || args.queue.read_index == nullptr ||
        args.queue.doorbell_handle == 0 || args.queue.size == 0 ||
        (args.queue.size & (args.queue.size - 1)) != 0 ||
        (reinterpret_cast<uintptr_t>(args.queue.ring_base) & 63u) != 0u ||
        args.queue.owner != AqlQueueOwner::GpuMcu ||
        args.packet_templates == nullptr ||
        args.packet_count == 0 ||
        args.packet_count > kAqlBatchMaxPackets ||
        args.packet_count > args.queue.size) {
        return GpuMcuAqlStatus::INVALID_PARAMS;
    }

    uint64_t base_slot = 0;
    if (!aql_reserve_packets(args.queue, args.packet_count, &base_slot,
                             kAqlQueueFullSpinLimit)) {
        return GpuMcuAqlStatus::RESERVE_TIMEOUT;
    }

    for (uint32_t i = 0; i < args.packet_count; ++i) {
        unsigned char* pkt = aql_packet_at(args.queue, base_slot + i);
        construct_aql_packet_nonvolatile(pkt, args.packet_templates[i]);
    }

    for (uint32_t i = 0; i < args.packet_count; ++i) {
        unsigned char* pkt = aql_packet_at(args.queue, base_slot + i);
        aql_publish_dw0(pkt, args.packet_templates[i].publish_dw0);
    }

    ring_aql_doorbell(args.queue, base_slot + args.packet_count - 1);
    return GpuMcuAqlStatus::COMPLETE;
}

// Single-producer streaming publish used by the persistent MCU. The caller owns
// the queue write index (it is the only producer), so no read of write_index /
// read_index happens per batch and the descriptor update is relaxed. The
// barrier bit selects overlap (0) or forced serialization (1) per batch.
__device__ __forceinline__ GpuMcuAqlStatus gpu_mcu_aql_publish_streaming(
    const DeviceAqlQueueView& queue,
    const GpuAqlPacketTemplate* packet_templates,
    uint32_t packet_count,
    uint64_t base_slot,
    bool barrier) {
    if (queue.ring_base == nullptr || queue.write_index == nullptr ||
        queue.doorbell_handle == 0 || queue.size == 0 ||
        (queue.size & (queue.size - 1)) != 0 ||
        (reinterpret_cast<uintptr_t>(queue.ring_base) & 63u) != 0u ||
        packet_templates == nullptr || packet_count == 0 ||
        packet_count > kAqlBatchMaxPackets || packet_count > queue.size) {
        return GpuMcuAqlStatus::INVALID_PARAMS;
    }
    for (uint32_t i = 0; i < packet_count; ++i) {
        unsigned char* pkt = aql_packet_at(queue, base_slot + i);
        aql_set_packet_invalid(pkt);
        copy_aql_packet_body(pkt, packet_templates[i].bytes);
    }
    for (uint32_t i = 0; i < packet_count; ++i) {
        unsigned char* pkt = aql_packet_at(queue, base_slot + i);
        uint32_t dw0 = packet_templates[i].publish_dw0;
        if (!barrier) {
            dw0 &= ~(1u << kAqlDw0BarrierBit);
        }
        aql_publish_dw0(pkt, dw0);
    }
    __atomic_store_n(queue.write_index, base_slot + packet_count, __ATOMIC_RELAXED);
    ring_aql_doorbell(queue, base_slot + packet_count - 1u);
    return GpuMcuAqlStatus::COMPLETE;
}

struct GpuMcuStagedPacket {
    unsigned char* packet = nullptr;
    uint32_t publish_dw0 = 0;
};

__device__ __forceinline__ GpuMcuStagedPacket gpu_mcu_aql_stage_packet_body(
    const DeviceAqlQueueView& queue,
    uint64_t slot,
    const std::byte* body,
    uint64_t kernarg_address) {
    unsigned char* pkt = aql_packet_at(queue, slot);
    aql_set_packet_invalid(pkt);
    copy_aql_packet_body(pkt, body);
    if (kernarg_address != 0u) {
        __builtin_memcpy(pkt + kAqlOffKernargAddress, &kernarg_address,
                         sizeof(kernarg_address));
    }
    GpuMcuStagedPacket staged{};
    staged.packet = pkt;
    return staged;
}

__device__ __forceinline__ GpuMcuStagedPacket gpu_mcu_aql_stage_packet(
    const DeviceAqlQueueView& queue,
    uint64_t slot,
    const GpuAqlPacketTemplate& templ,
    uint64_t kernarg_address) {
    GpuMcuStagedPacket staged = gpu_mcu_aql_stage_packet_body(
        queue, slot, templ.bytes, kernarg_address);
    staged.publish_dw0 = templ.publish_dw0;
    return staged;
}

__device__ __forceinline__ uint32_t gpu_mcu_aql_commit_packet(
    const DeviceAqlQueueView& queue,
    const GpuMcuStagedPacket& staged,
    uint64_t new_write_index,
    bool publish_write_index,
    bool barrier,
    bool signal_doorbell) {
    __threadfence_system();
    uint32_t dw0 = staged.publish_dw0;
    if (!barrier) {
        dw0 &= ~(1u << kAqlDw0BarrierBit);
    }
    aql_publish_dw0(staged.packet, dw0);
    if (publish_write_index) {
        __scoped_atomic_store_n(queue.write_index, new_write_index,
                                __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }
    if (signal_doorbell) {
        ring_aql_doorbell(queue, new_write_index - 1u);
    }
    return dw0;
}

struct GpuAqlKernelMetadata {
    uint64_t kernel_object = 0;
    uint32_t private_segment_size = 0;
    uint32_t static_group_segment_size = 0;
    uint32_t kernarg_segment_size = 0;
};

struct GpuMcuKernargRegion {
    unsigned char* base = nullptr;
    std::size_t requested_slot_bytes = 0;
    std::size_t granule = 0;
    std::size_t actual_size = 0;
    std::size_t slot_stride = 0;
    uint32_t slot_count = 0;

    unsigned char* slot(std::size_t index) const noexcept {
        return base + index * slot_stride;
    }
};

class GpuMcuAqlQueue {
public:
    GpuMcuAqlQueue() = default;
    ~GpuMcuAqlQueue() noexcept;

    GpuMcuAqlQueue(GpuMcuAqlQueue&& other) noexcept;
    GpuMcuAqlQueue& operator=(GpuMcuAqlQueue&& other) noexcept;

    GpuMcuAqlQueue(const GpuMcuAqlQueue&) = delete;
    GpuMcuAqlQueue& operator=(const GpuMcuAqlQueue&) = delete;

    static Result<GpuMcuAqlQueue> create(
        int device,
        AqlQueueOwner owner,
        uint32_t size = kAqlDefaultQueueSize);

    static Result<GpuMcuAqlQueue> create_with_mask(
        int device,
        AqlQueueOwner owner,
        uint32_t size,
        AqlCuMaskView mask);

    bool valid() const noexcept;
    const DeviceAqlQueueView& device_queue_view() const noexcept;
    const char* arch_name() const noexcept;
    uint32_t size() const noexcept;
    AqlQueueOwner owner() const noexcept;
    uint64_t hsa_agent_handle() const noexcept;

    Result<std::vector<uint32_t>> requested_mask() const;
    Result<std::vector<uint32_t>> readback_mask() const;

    Result<GpuMcuKernargRegion> allocate_kernarg(
        std::size_t slot_bytes,
        uint32_t slot_count);
    Result<GpuMcuKernargRegion> allocate_log_region(std::size_t bytes);
    Result<std::size_t> kernarg_granule() const;

    Status submit_host(const GpuAqlPacketTemplate* packets, uint32_t count);
    uint64_t host_queued_index() const noexcept;
    uint64_t host_consumed_index() const noexcept;

    Status shutdown() noexcept;

private:
    struct Internal;
    static Result<GpuMcuAqlQueue> create_impl(
        int device,
        AqlQueueOwner owner,
        uint32_t size,
        AqlCuMaskView mask);
    explicit GpuMcuAqlQueue(Internal* impl) noexcept : impl_(impl) {}
    void move_from(GpuMcuAqlQueue&& other) noexcept;
    Internal* impl_ = nullptr;
};

class GpuMcuAqlCodeObject {
public:
    GpuMcuAqlCodeObject() = default;
    ~GpuMcuAqlCodeObject() noexcept;

    GpuMcuAqlCodeObject(GpuMcuAqlCodeObject&& other) noexcept;
    GpuMcuAqlCodeObject& operator=(GpuMcuAqlCodeObject&& other) noexcept;

    GpuMcuAqlCodeObject(const GpuMcuAqlCodeObject&) = delete;
    GpuMcuAqlCodeObject& operator=(const GpuMcuAqlCodeObject&) = delete;

    static Result<GpuMcuAqlCodeObject> load(
        const GpuMcuAqlQueue& queue,
        const char* hsaco_path,
        const char* symbol_name);

    static Result<GpuMcuAqlCodeObject> load_memory(
        const GpuMcuAqlQueue& queue,
        const void* code_object,
        std::size_t code_object_size,
        const char* symbol_name);

    const GpuAqlKernelMetadata& metadata() const noexcept;

    Status shutdown() noexcept;

private:
    struct Internal;
    explicit GpuMcuAqlCodeObject(Internal* impl) noexcept : impl_(impl) {}
    void move_from(GpuMcuAqlCodeObject&& other) noexcept;
    Internal* impl_ = nullptr;
};

}  // namespace ps::runtime::gpu_mcu
