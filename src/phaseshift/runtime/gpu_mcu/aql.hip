#include <phaseshift/runtime/gpu_mcu/aql.h>
#include <phaseshift/runtime/gpu_mcu/retained_packet.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <string>
#include <utility>
#include <vector>

namespace ps::runtime::gpu_mcu {

namespace {

constexpr uint64_t kAmdQueueWriteIndexOffset = 56;
constexpr uint64_t kAmdQueueReadIndexOffset = 128;

static_assert(offsetof(hsa_queue_t, base_address) == 8);
static_assert(offsetof(hsa_queue_t, doorbell_signal) == 16);
static_assert(offsetof(hsa_queue_t, size) == 24);

static_assert(offsetof(hsa_kernel_dispatch_packet_t, full_header) == kAqlDw0Offset);
static_assert(offsetof(hsa_kernel_dispatch_packet_t, workgroup_size_x) == kAqlOffWgX);
static_assert(offsetof(hsa_kernel_dispatch_packet_t, workgroup_size_y) == kAqlOffWgY);
static_assert(offsetof(hsa_kernel_dispatch_packet_t, workgroup_size_z) == kAqlOffWgZ);
static_assert(offsetof(hsa_kernel_dispatch_packet_t, grid_size_x) == kAqlOffGridX);
static_assert(offsetof(hsa_kernel_dispatch_packet_t, grid_size_y) == kAqlOffGridY);
static_assert(offsetof(hsa_kernel_dispatch_packet_t, grid_size_z) == kAqlOffGridZ);
static_assert(offsetof(hsa_kernel_dispatch_packet_t, private_segment_size) ==
              kAqlOffPrivateSegment);
static_assert(offsetof(hsa_kernel_dispatch_packet_t, group_segment_size) ==
              kAqlOffGroupSegment);
static_assert(offsetof(hsa_kernel_dispatch_packet_t, kernel_object) == kAqlOffKernelObject);
static_assert(offsetof(hsa_kernel_dispatch_packet_t, kernarg_address) ==
              kAqlOffKernargAddress);
static_assert(offsetof(hsa_kernel_dispatch_packet_t, completion_signal) ==
              kAqlOffCompletionSignal);
static_assert(sizeof(hsa_kernel_dispatch_packet_t) == kAqlPacketBytes);

Status hsa_status_error(hsa_status_t s, const char* what) {
    if (s == HSA_STATUS_SUCCESS) return Status::make_ok();
    const char* name = nullptr;
    (void)hsa_status_string(s, &name);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s failed: %s", what,
                  name != nullptr ? name : "unknown hsa error");
    return Status::invalid_state(buf, __FILE__, __LINE__);
}

uint32_t parse_bdfid(const std::string& pci) {
    unsigned domain = 0;
    unsigned bus = 0;
    unsigned dev = 0;
    unsigned func = 0;
    if (std::sscanf(pci.c_str(), "%x:%x:%x.%x", &domain, &bus, &dev, &func) != 4) {
        return 0;
    }
    return (domain << 16) | (bus << 8) | (dev << 3) | func;
}

struct AgentMatch {
    hsa_agent_t agent{};
    char name[128] = {};
    bool found = false;
    int hip_device = 0;
};

hsa_status_t agent_match_cb(hsa_agent_t agent, void* data) {
    auto* match = static_cast<AgentMatch*>(data);
    hsa_device_type_t dev = HSA_DEVICE_TYPE_CPU;
    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev) != HSA_STATUS_SUCCESS ||
        dev != HSA_DEVICE_TYPE_GPU) {
        return HSA_STATUS_SUCCESS;
    }
    uint32_t features = 0;
    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_FEATURE, &features) !=
            HSA_STATUS_SUCCESS ||
        (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH) == 0) {
        return HSA_STATUS_SUCCESS;
    }
    char name[128] = {};
    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name) != HSA_STATUS_SUCCESS) {
        return HSA_STATUS_SUCCESS;
    }
    if (std::strstr(name, "gfx1201") == nullptr) {
        return HSA_STATUS_SUCCESS;
    }
    uint32_t bdfid = 0;
    if (hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID),
                           &bdfid) != HSA_STATUS_SUCCESS) {
        return HSA_STATUS_SUCCESS;
    }
    char pci[64] = {};
    if (hipDeviceGetPCIBusId(pci, sizeof(pci), match->hip_device) != hipSuccess) {
        return HSA_STATUS_SUCCESS;
    }
    const uint32_t hip_bdf = parse_bdfid(pci);
    if (hip_bdf == 0 || bdfid != hip_bdf) {
        return HSA_STATUS_SUCCESS;
    }
    match->agent = agent;
    match->found = true;
    std::strncpy(match->name, name, sizeof(match->name) - 1);
    return HSA_STATUS_INFO_BREAK;
}

hsa_status_t kernarg_region_cb(hsa_region_t region, void* data) {
    auto* out = static_cast<hsa_region_t*>(data);
    hsa_region_segment_t segment = HSA_REGION_SEGMENT_GLOBAL;
    if (hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment) !=
            HSA_STATUS_SUCCESS ||
        segment != HSA_REGION_SEGMENT_GLOBAL) {
        return HSA_STATUS_SUCCESS;
    }
    uint32_t flags = 0;
    if (hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags) !=
        HSA_STATUS_SUCCESS) {
        return HSA_STATUS_SUCCESS;
    }
    bool alloc_allowed = false;
    if (hsa_region_get_info(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
                            &alloc_allowed) != HSA_STATUS_SUCCESS ||
        !alloc_allowed) {
        return HSA_STATUS_SUCCESS;
    }
    if ((flags & HSA_REGION_GLOBAL_FLAG_KERNARG) != 0) {
        *out = region;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

void extract_arch_name(const char* agent_name, char* out, size_t out_size) {
    const char* gfx = std::strstr(agent_name, "gfx");
    std::snprintf(out, out_size, "%s", gfx != nullptr ? gfx : "unknown");
}

std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
    if (alignment <= 1) return value;
    return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

GpuAqlPacketTemplate build_aql_packet_template(
    const GpuAqlDispatchDesc& desc,
    const AqlMemoryPolicy& policy) {
    GpuAqlPacketTemplate out{};
    auto* raw = reinterpret_cast<unsigned char*>(out.bytes);
    const uint32_t invalid_dw0 = kAqlInvalidDw0;
    std::memcpy(raw + kAqlDw0Offset, &invalid_dw0, sizeof(invalid_dw0));
    std::memcpy(raw + kAqlOffWgX, &desc.workgroup_size_x, sizeof(desc.workgroup_size_x));
    std::memcpy(raw + kAqlOffWgY, &desc.workgroup_size_y, sizeof(desc.workgroup_size_y));
    std::memcpy(raw + kAqlOffWgZ, &desc.workgroup_size_z, sizeof(desc.workgroup_size_z));
    std::memcpy(raw + kAqlOffGridX, &desc.global_work_items_x,
                sizeof(desc.global_work_items_x));
    std::memcpy(raw + kAqlOffGridY, &desc.global_work_items_y,
                sizeof(desc.global_work_items_y));
    std::memcpy(raw + kAqlOffGridZ, &desc.global_work_items_z,
                sizeof(desc.global_work_items_z));
    std::memcpy(raw + kAqlOffPrivateSegment, &desc.private_segment_size,
                sizeof(desc.private_segment_size));
    std::memcpy(raw + kAqlOffGroupSegment, &desc.group_segment_size,
                sizeof(desc.group_segment_size));
    std::memcpy(raw + kAqlOffKernelObject, &desc.kernel_object, sizeof(desc.kernel_object));
    std::memcpy(raw + kAqlOffKernargAddress, &desc.kernarg_address,
                sizeof(desc.kernarg_address));
    std::memcpy(raw + kAqlOffCompletionSignal, &desc.completion_signal,
                sizeof(desc.completion_signal));
    const uint64_t reserved1 = 0;
    std::memcpy(raw + kAqlOffReserved1, &reserved1, sizeof(reserved1));
    out.publish_dw0 =
        make_kernel_dispatch_dw0(policy.acquire_scope, policy.release_scope,
                                 aql_dispatch_dimensions(
                                     desc.global_work_items_y,
                                     desc.global_work_items_z));
    return out;
}

GpuMcuRetainedPacket make_retained_packet(const GpuAqlPacketTemplate& templ) {
    const AqlPacketBodyBundle body = make_body_bundle(templ.bytes);
    GpuMcuRetainedPacket out{};
    out.publish_dw0 = templ.publish_dw0;
    out.at4 = body.at4;
    out.at8 = body.at8;
    out.at16 = body.at16;
    out.at32 = body.at32;
    out.at48 = body.at48;
    return out;
}

AqlPacketBodyBundle make_body_bundle(const std::byte image[kAqlPacketBytes]) {
    AqlPacketBodyBundle b{};
    std::memcpy(&b.at4, image + kAqlOffWgX, sizeof(b.at4));
    std::memcpy(&b.at8, image + 8, sizeof(b.at8));
    std::memcpy(&b.at16, image + 16, sizeof(b.at16));
    std::memcpy(&b.at32, image + 32, sizeof(b.at32));
    std::memcpy(&b.at48, image + 48, sizeof(b.at48));
    return b;
}

const char* gpu_mcu_aql_status_name(uint32_t status) {
    switch (status) {
        case 0: return "COMPLETE";
        case 1: return "INVALID_PARAMS";
        case 2: return "RESERVE_TIMEOUT";
        default: return "UNKNOWN";
    }
}

bool aql_cu_mask_is_wgp_paired(const uint32_t* words, uint32_t cu_count) noexcept {
    if (words == nullptr || cu_count == 0u || (cu_count & 1u) != 0u) {
        return false;
    }
    const auto enabled = [&](uint32_t index) {
        return (words[index / 32u] & (1u << (index % 32u))) != 0u;
    };
    for (uint32_t cu = 0; cu < cu_count; cu += 2u) {
        if (enabled(cu) != enabled(cu + 1u)) {
            return false;
        }
    }
    return true;
}

Result<uint32_t> gpu_mcu_hsa_compute_unit_count(int device) {
    int hip_device = device;
    if (hip_device < 0 && hipGetDevice(&hip_device) != hipSuccess) {
        return Status::invalid_state("hipGetDevice failed", __FILE__, __LINE__);
    }
    if (hsa_init() != HSA_STATUS_SUCCESS) {
        return Status::invalid_state("hsa_init failed", __FILE__, __LINE__);
    }
    AgentMatch match{};
    match.hip_device = hip_device;
    const hsa_status_t it = hsa_iterate_agents(agent_match_cb, &match);
    if (it != HSA_STATUS_SUCCESS && it != HSA_STATUS_INFO_BREAK) {
        return Status::invalid_state("hsa_iterate_agents failed", __FILE__, __LINE__);
    }
    if (!match.found) {
        return Status::invalid_state("no gfx1201 HSA agent matching the HIP device",
                                     __FILE__, __LINE__);
    }
    uint32_t cu_count = 0;
    const hsa_status_t st = hsa_agent_get_info(
        match.agent,
        static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT),
        &cu_count);
    if (st != HSA_STATUS_SUCCESS) {
        return Status::invalid_state("hsa_agent_get_info(COMPUTE_UNIT_COUNT) failed",
                                     __FILE__, __LINE__);
    }
    return cu_count;
}

struct GpuMcuAqlQueue::Internal {
    hsa_agent_t agent{};
    hsa_region_t kernarg_region{};
    hsa_queue_t* queue = nullptr;
    char arch_name[32] = {};
    DeviceAqlQueueView view{};
    bool created = false;
    std::vector<uint32_t> requested_mask;
    uint32_t mask_bit_count = 0;
    std::vector<std::pair<void*, std::size_t>> kernarg_allocations;

    Status shutdown_noexcept() noexcept {
        Status first_error = Status::make_ok();
        auto note = [&](Status s) {
            if (!s.ok() && first_error.ok()) first_error = s;
        };
        for (auto& alloc : kernarg_allocations) {
            if (alloc.first) {
                note(hsa_status_error(hsa_memory_free(alloc.first), "hsa_memory_free"));
                alloc.first = nullptr;
            }
        }
        kernarg_allocations.clear();
        if (queue) {
            note(hsa_status_error(hsa_queue_destroy(queue), "hsa_queue_destroy"));
            queue = nullptr;
        }
        created = false;
        return first_error;
    }
};

GpuMcuAqlQueue::GpuMcuAqlQueue(GpuMcuAqlQueue&& other) noexcept {
    move_from(std::move(other));
}

GpuMcuAqlQueue& GpuMcuAqlQueue::operator=(GpuMcuAqlQueue&& other) noexcept {
    if (this != &other) {
        (void)shutdown();
        move_from(std::move(other));
    }
    return *this;
}

void GpuMcuAqlQueue::move_from(GpuMcuAqlQueue&& other) noexcept {
    impl_ = other.impl_;
    other.impl_ = nullptr;
}

GpuMcuAqlQueue::~GpuMcuAqlQueue() noexcept {
    if (impl_) {
        (void)impl_->shutdown_noexcept();
        delete impl_;
        impl_ = nullptr;
    }
}

bool GpuMcuAqlQueue::valid() const noexcept {
    return impl_ != nullptr && impl_->created;
}

const DeviceAqlQueueView& GpuMcuAqlQueue::device_queue_view() const noexcept {
    static const DeviceAqlQueueView empty{};
    if (!valid()) return empty;
    return impl_->view;
}

const char* GpuMcuAqlQueue::arch_name() const noexcept {
    if (!valid()) return "none";
    return impl_->arch_name;
}

uint32_t GpuMcuAqlQueue::size() const noexcept {
    if (!valid()) return 0;
    return impl_->view.size;
}

AqlQueueOwner GpuMcuAqlQueue::owner() const noexcept {
    if (!valid()) return AqlQueueOwner::Host;
    return impl_->view.owner;
}

uint64_t GpuMcuAqlQueue::hsa_agent_handle() const noexcept {
    if (!valid()) return 0;
    return impl_->agent.handle;
}

Result<std::vector<uint32_t>> GpuMcuAqlQueue::requested_mask() const {
    if (!valid()) {
        return Status::invalid_state("aql queue not created", __FILE__, __LINE__);
    }
    return impl_->requested_mask;
}

Result<std::vector<uint32_t>> GpuMcuAqlQueue::readback_mask() const {
    if (!valid()) {
        return Status::invalid_state("aql queue not created", __FILE__, __LINE__);
    }
    if (impl_->mask_bit_count == 0u) {
        return std::vector<uint32_t>{};
    }
    std::vector<uint32_t> actual(impl_->mask_bit_count / 32u, 0u);
    if (const Status s = hsa_status_error(
            hsa_amd_queue_cu_get_mask(impl_->queue, impl_->mask_bit_count, actual.data()),
            "hsa_amd_queue_cu_get_mask"); !s.ok()) {
        return s;
    }
    return actual;
}

Result<std::size_t> GpuMcuAqlQueue::kernarg_granule() const {
    if (!valid()) {
        return Status::invalid_state("aql queue not created", __FILE__, __LINE__);
    }
    std::size_t granule = 0;
    if (const Status st = hsa_status_error(
            hsa_region_get_info(impl_->kernarg_region,
                                HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE, &granule),
            "hsa_region_get_info(granule)"); !st.ok()) {
        return st;
    }
    return granule;
}

Result<GpuMcuKernargRegion> GpuMcuAqlQueue::allocate_kernarg(
    std::size_t slot_bytes,
    uint32_t slot_count) {
    if (!valid()) {
        return Status::invalid_state("aql queue not created", __FILE__, __LINE__);
    }
    if (slot_bytes == 0 || slot_count == 0) {
        return Status::invalid_argument("kernarg slot size and count must be nonzero",
                                        __FILE__, __LINE__);
    }
    auto granule_res = kernarg_granule();
    if (!granule_res.ok()) {
        return granule_res.status();
    }
    const std::size_t granule = granule_res.value();
    const std::size_t stride = align_up(slot_bytes, 16);
    const std::size_t raw = stride * static_cast<std::size_t>(slot_count);
    const std::size_t actual = align_up(raw, granule > 16 ? granule : 16);

    void* ptr = nullptr;
    if (const Status st = hsa_status_error(
            hsa_memory_allocate(impl_->kernarg_region, actual, &ptr),
            "hsa_memory_allocate"); !st.ok()) {
        return st;
    }
    std::memset(ptr, 0, actual);
    impl_->kernarg_allocations.emplace_back(ptr, actual);

    GpuMcuKernargRegion region{};
    region.base = static_cast<unsigned char*>(ptr);
    region.requested_slot_bytes = slot_bytes;
    region.granule = granule;
    region.actual_size = actual;
    region.slot_stride = stride;
    region.slot_count = slot_count;
    return region;
}

Status GpuMcuAqlQueue::submit_host(
    const GpuAqlPacketTemplate* packets,
    uint32_t count) {
    if (!valid()) {
        return Status::invalid_state("aql queue not created", __FILE__, __LINE__);
    }
    if (packets == nullptr || count == 0 || count > impl_->view.size) {
        return Status::invalid_argument("invalid host submit batch", __FILE__, __LINE__);
    }
    uint32_t spins = 0;
    while ((hsa_queue_load_write_index_scacquire(impl_->queue) -
            hsa_queue_load_read_index_scacquire(impl_->queue)) + count >
           impl_->view.size) {
        if (++spins >= 50000000u) {
            return Status::invalid_state("host submit queue full", __FILE__, __LINE__);
        }
        __builtin_ia32_pause();
    }
    const uint64_t base =
        hsa_queue_add_write_index_scacq_screl(impl_->queue, count);
    auto* ring = static_cast<unsigned char*>(impl_->queue->base_address);
    for (uint32_t i = 0; i < count; ++i) {
        unsigned char* pkt =
            ring + static_cast<uint32_t>((base + i) % impl_->view.size) * kAqlPacketBytes;
        const uint32_t invalid = kAqlInvalidDw0;
        std::memcpy(pkt + kAqlDw0Offset, &invalid, sizeof(invalid));
        std::memcpy(pkt + 4, packets[i].bytes + 4, kAqlBodyBytes);
    }
    std::atomic_thread_fence(std::memory_order_release);
    for (uint32_t i = 0; i < count; ++i) {
        unsigned char* pkt =
            ring + static_cast<uint32_t>((base + i) % impl_->view.size) * kAqlPacketBytes;
        std::memcpy(pkt + kAqlDw0Offset, &packets[i].publish_dw0,
                    sizeof(packets[i].publish_dw0));
    }
    std::atomic_thread_fence(std::memory_order_release);
    hsa_signal_store_screlease(impl_->queue->doorbell_signal, base + count - 1u);
    return Status::make_ok();
}

uint64_t GpuMcuAqlQueue::host_queued_index() const noexcept {
    if (!valid()) return 0;
    return hsa_queue_load_write_index_scacquire(impl_->queue);
}

uint64_t GpuMcuAqlQueue::host_consumed_index() const noexcept {
    if (!valid()) return 0;
    return hsa_queue_load_read_index_scacquire(impl_->queue);
}

Status GpuMcuAqlQueue::shutdown() noexcept {    if (!impl_) return Status::make_ok();
    auto* impl = impl_;
    impl_ = nullptr;
    const Status st = impl->shutdown_noexcept();
    delete impl;
    return st;
}

Result<GpuMcuAqlQueue> GpuMcuAqlQueue::create(
    int device,
    AqlQueueOwner owner,
    uint32_t size) {
    return create_impl(device, owner, size, AqlCuMaskView{});
}

Result<GpuMcuAqlQueue> GpuMcuAqlQueue::create_with_mask(
    int device,
    AqlQueueOwner owner,
    uint32_t size,
    AqlCuMaskView mask) {
    return create_impl(device, owner, size, mask);
}

Result<GpuMcuAqlQueue> GpuMcuAqlQueue::create_impl(
    int device,
    AqlQueueOwner owner,
    uint32_t size,
    AqlCuMaskView mask) {
    int hip_device = device;
    if (hip_device < 0 && hipGetDevice(&hip_device) != hipSuccess) {
        return Status::invalid_state("hipGetDevice failed", __FILE__, __LINE__);
    }

    auto* impl = new Internal();
    if (hsa_init() != HSA_STATUS_SUCCESS) {
        delete impl;
        return Status::invalid_state("hsa_init failed", __FILE__, __LINE__);
    }
    auto fail = [&](const std::string& msg) -> Result<GpuMcuAqlQueue> {
        (void)impl->shutdown_noexcept();
        delete impl;
        return Status::invalid_state(msg.c_str(), __FILE__, __LINE__);
    };

    AgentMatch match{};
    match.hip_device = hip_device;
    const hsa_status_t it = hsa_iterate_agents(agent_match_cb, &match);
    if (it != HSA_STATUS_SUCCESS && it != HSA_STATUS_INFO_BREAK) {
        return fail("hsa_iterate_agents failed");
    }
    if (!match.found) {
        return fail("no gfx1201 HSA agent matching the HIP device PCI BDF");
    }
    impl->agent = match.agent;
    extract_arch_name(match.name, impl->arch_name, sizeof(impl->arch_name));

    hsa_region_t kernarg_region{};
    const hsa_status_t rit = hsa_agent_iterate_regions(
        impl->agent, kernarg_region_cb, &kernarg_region);
    if (rit != HSA_STATUS_SUCCESS && rit != HSA_STATUS_INFO_BREAK) {
        return fail("hsa_agent_iterate_regions failed");
    }
    if (!kernarg_region.handle) {
        return fail("no kernarg region on gfx1201 agent");
    }
    impl->kernarg_region = kernarg_region;

    if (hsa_queue_create(
            impl->agent, size, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr,
            UINT32_MAX, UINT32_MAX, &impl->queue) != HSA_STATUS_SUCCESS) {
        return fail("hsa_queue_create failed");
    }

    if (mask.words != nullptr && mask.bit_count != 0u) {
        if ((mask.bit_count & 31u) != 0u) {
            return fail("AQL mask bit count must be a multiple of 32");
        }
        if (!aql_cu_mask_is_wgp_paired(mask.words, mask.bit_count)) {
            return fail("AQL CU mask violates WGP pairwise enable");
        }
        const hsa_status_t set_st = hsa_amd_queue_cu_set_mask(
            impl->queue, mask.bit_count, mask.words);
        if (set_st == static_cast<hsa_status_t>(HSA_STATUS_CU_MASK_REDUCED)) {
            return fail("hsa_amd_queue_cu_set_mask reduced");
        }
        if (set_st != HSA_STATUS_SUCCESS) {
            const char* name = nullptr;
            (void)hsa_status_string(set_st, &name);
            return fail(std::string("hsa_amd_queue_cu_set_mask failed: ") +
                        (name != nullptr ? name : "unknown"));
        }
        impl->requested_mask.assign(mask.words, mask.words + mask.bit_count / 32u);
        impl->mask_bit_count = mask.bit_count;
    }

    impl->view.queue_ptr = impl->queue;
    impl->view.ring_base = impl->queue->base_address;
    impl->view.doorbell_handle = impl->queue->doorbell_signal.handle;
    impl->view.size = impl->queue->size;
    impl->view.write_index = reinterpret_cast<volatile uint64_t*>(
        reinterpret_cast<char*>(impl->queue) + kAmdQueueWriteIndexOffset);
    impl->view.read_index = reinterpret_cast<volatile uint64_t*>(
        reinterpret_cast<char*>(impl->queue) + kAmdQueueReadIndexOffset);
    impl->view.owner = owner;
    impl->created = true;

    return GpuMcuAqlQueue(impl);
}

struct GpuMcuAqlCodeObject::Internal {
    int fd = -1;
    hsa_code_object_reader_t reader{};
    hsa_executable_t executable{};
    GpuAqlKernelMetadata meta{};
    bool valid = false;

    Status shutdown_noexcept() noexcept {
        Status first_error = Status::make_ok();
        auto note = [&](Status s) {
            if (!s.ok() && first_error.ok()) first_error = s;
        };
        if (executable.handle) {
            note(hsa_status_error(hsa_executable_destroy(executable),
                                  "hsa_executable_destroy"));
            executable = hsa_executable_t{};
        }
        if (reader.handle) {
            note(hsa_status_error(hsa_code_object_reader_destroy(reader),
                                  "hsa_code_object_reader_destroy"));
            reader = hsa_code_object_reader_t{};
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        valid = false;
        return first_error;
    }

    Result<GpuAqlKernelMetadata> resolve_symbol(
        uint64_t agent_handle, const char* symbol_name) {
        hsa_agent_t agent{};
        agent.handle = agent_handle;
        hsa_executable_symbol_t symbol{};
        hsa_status_t s = hsa_executable_get_symbol_by_name(
            executable, symbol_name, &agent, &symbol);
        if (s != HSA_STATUS_SUCCESS) {
            const std::string kd = std::string(symbol_name) + ".kd";
            s = hsa_executable_get_symbol_by_name(executable, kd.c_str(), &agent, &symbol);
        }
        if (s != HSA_STATUS_SUCCESS) {
            struct IterateData {
                std::string name;
                hsa_executable_symbol_t symbol{};
                bool found = false;
            } data{symbol_name, {}, false};
            hsa_executable_iterate_symbols(
                executable,
                [](hsa_executable_t, hsa_executable_symbol_t sym, void* d) {
                    auto* p = static_cast<IterateData*>(d);
                    hsa_symbol_kind_t kind = HSA_SYMBOL_KIND_VARIABLE;
                    if (hsa_executable_symbol_get_info(
                            sym, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind) !=
                        HSA_STATUS_SUCCESS) {
                        return HSA_STATUS_SUCCESS;
                    }
                    if (kind != HSA_SYMBOL_KIND_KERNEL) return HSA_STATUS_SUCCESS;
                    uint32_t len = 0;
                    hsa_executable_symbol_get_info(
                        sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &len);
                    std::vector<char> name(len + 1);
                    hsa_executable_symbol_get_info(
                        sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data());
                    name[len] = 0;
                    if (p->name == name.data()) {
                        p->symbol = sym;
                        p->found = true;
                        return HSA_STATUS_INFO_BREAK;
                    }
                    return HSA_STATUS_SUCCESS;
                },
                &data);
            if (!data.found) {
                char buf[256];
                std::snprintf(buf, sizeof(buf), "kernel symbol not found: %s", symbol_name);
                return Status::invalid_state(buf, __FILE__, __LINE__);
            }
            symbol = data.symbol;
        }

        GpuAqlKernelMetadata meta{};
        if (const Status st = hsa_status_error(
                hsa_executable_symbol_get_info(
                    symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                    &meta.kernel_object),
                "symbol kernel_object"); !st.ok()) {
            return st;
        }
        if (const Status st = hsa_status_error(
                hsa_executable_symbol_get_info(
                    symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                    &meta.kernarg_segment_size),
                "symbol kernarg segment size"); !st.ok()) {
            return st;
        }
        if (const Status st = hsa_status_error(
                hsa_executable_symbol_get_info(
                    symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                    &meta.static_group_segment_size),
                "symbol group segment size"); !st.ok()) {
            return st;
        }
        if (const Status st = hsa_status_error(
                hsa_executable_symbol_get_info(
                    symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                    &meta.private_segment_size),
                "symbol private segment size"); !st.ok()) {
            return st;
        }
        return meta;
    }
};

GpuMcuAqlCodeObject::GpuMcuAqlCodeObject(GpuMcuAqlCodeObject&& other) noexcept {
    move_from(std::move(other));
}

GpuMcuAqlCodeObject& GpuMcuAqlCodeObject::operator=(
    GpuMcuAqlCodeObject&& other) noexcept {
    if (this != &other) {
        (void)shutdown();
        move_from(std::move(other));
    }
    return *this;
}

void GpuMcuAqlCodeObject::move_from(GpuMcuAqlCodeObject&& other) noexcept {
    impl_ = other.impl_;
    other.impl_ = nullptr;
}

GpuMcuAqlCodeObject::~GpuMcuAqlCodeObject() noexcept {
    if (impl_) {
        (void)impl_->shutdown_noexcept();
        delete impl_;
        impl_ = nullptr;
    }
}

const GpuAqlKernelMetadata& GpuMcuAqlCodeObject::metadata() const noexcept {
    static const GpuAqlKernelMetadata empty{};
    if (!impl_ || !impl_->valid) return empty;
    return impl_->meta;
}

Status GpuMcuAqlCodeObject::shutdown() noexcept {
    if (!impl_) return Status::make_ok();
    auto* impl = impl_;
    impl_ = nullptr;
    const Status st = impl->shutdown_noexcept();
    delete impl;
    return st;
}

Result<GpuMcuAqlCodeObject> GpuMcuAqlCodeObject::load(
    const GpuMcuAqlQueue& queue,
    const char* hsaco_path,
    const char* symbol_name) {
    if (!queue.valid()) {
        return Status::invalid_state("aql queue not created", __FILE__, __LINE__);
    }
    const uint64_t agent_handle = queue.hsa_agent_handle();
    hsa_agent_t agent{};
    agent.handle = agent_handle;
    auto* impl = new Internal();
    impl->fd = ::open(hsaco_path, O_RDONLY);
    if (impl->fd < 0) {
        delete impl;
        return Status::invalid_state("cannot open hsaco", __FILE__, __LINE__);
    }
    auto fail = [&](const std::string& msg) -> Result<GpuMcuAqlCodeObject> {
        (void)impl->shutdown_noexcept();
        delete impl;
        return Status::invalid_state(msg.c_str(), __FILE__, __LINE__);
    };
    if (const Status s = hsa_status_error(
            hsa_code_object_reader_create_from_file(impl->fd, &impl->reader),
            "code object reader from file"); !s.ok()) {
        return fail(s.message());
    }
    if (const Status s = hsa_status_error(
            hsa_executable_create_alt(HSA_PROFILE_BASE,
                                      HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR, nullptr,
                                      &impl->executable),
            "hsa_executable_create_alt"); !s.ok()) {
        return fail(s.message());
    }
    if (const Status s = hsa_status_error(
            hsa_executable_load_agent_code_object(impl->executable, agent,
                                                  impl->reader, nullptr, nullptr),
            "load agent code object"); !s.ok()) {
        return fail(s.message());
    }
    if (const Status s = hsa_status_error(
            hsa_executable_freeze(impl->executable, nullptr),
            "hsa_executable_freeze"); !s.ok()) {
        return fail(s.message());
    }
    auto meta = impl->resolve_symbol(agent_handle, symbol_name);
    if (!meta.ok()) {
        return fail(meta.status().message());
    }
    impl->meta = meta.release();
    if (impl->meta.kernel_object == 0) {
        return fail("aql kernel_object resolved to zero");
    }
    impl->valid = true;
    return GpuMcuAqlCodeObject(impl);
}

Result<GpuMcuAqlCodeObject> GpuMcuAqlCodeObject::load_memory(
    const GpuMcuAqlQueue& queue,
    const void* code_object,
    std::size_t code_object_size,
    const char* symbol_name) {
    if (!queue.valid()) {
        return Status::invalid_state("aql queue not created", __FILE__, __LINE__);
    }
    if (code_object == nullptr || code_object_size == 0) {
        return Status::invalid_argument("empty code object image", __FILE__, __LINE__);
    }
    const uint64_t agent_handle = queue.hsa_agent_handle();
    hsa_agent_t agent{};
    agent.handle = agent_handle;
    auto* impl = new Internal();
    auto fail = [&](const std::string& msg) -> Result<GpuMcuAqlCodeObject> {
        (void)impl->shutdown_noexcept();
        delete impl;
        return Status::invalid_state(msg.c_str(), __FILE__, __LINE__);
    };
    if (const Status s = hsa_status_error(
            hsa_code_object_reader_create_from_memory(code_object, code_object_size,
                                                      &impl->reader),
            "code object reader from memory"); !s.ok()) {
        return fail(s.message());
    }
    if (const Status s = hsa_status_error(
            hsa_executable_create_alt(HSA_PROFILE_BASE,
                                      HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR, nullptr,
                                      &impl->executable),
            "hsa_executable_create_alt"); !s.ok()) {
        return fail(s.message());
    }
    if (const Status s = hsa_status_error(
            hsa_executable_load_agent_code_object(impl->executable, agent,
                                                  impl->reader, nullptr, nullptr),
            "load agent code object"); !s.ok()) {
        return fail(s.message());
    }
    if (const Status s = hsa_status_error(
            hsa_executable_freeze(impl->executable, nullptr),
            "hsa_executable_freeze"); !s.ok()) {
        return fail(s.message());
    }
    auto meta = impl->resolve_symbol(agent_handle, symbol_name);
    if (!meta.ok()) {
        return fail(meta.status().message());
    }
    impl->meta = meta.release();
    if (impl->meta.kernel_object == 0) {
        return fail("aql kernel_object resolved to zero");
    }
    impl->valid = true;
    return GpuMcuAqlCodeObject(impl);
}

}  // namespace ps::runtime::gpu_mcu
