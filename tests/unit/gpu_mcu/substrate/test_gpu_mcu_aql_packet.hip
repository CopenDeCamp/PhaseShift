#include <phaseshift/runtime/gpu_mcu/aql.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

namespace aql = ps::runtime::gpu_mcu;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

void test_dw0_fields() {
    using S = aql::AqlFenceScope;
    const uint32_t none = aql::make_kernel_dispatch_dw0(S::None, S::None);
    const uint32_t agent = aql::make_kernel_dispatch_dw0(S::Agent, S::Agent);
    const uint32_t system = aql::make_kernel_dispatch_dw0(S::System, S::System);
    check(none == 0x10102u, "dw0 NONE/NONE == 0x10102");
    check(agent == 0x10b02u, "dw0 AGENT/AGENT == 0x10b02");
    check(system == 0x11502u, "dw0 SYSTEM/SYSTEM == 0x11502");
    for (const uint32_t dw0 : {none, agent, system}) {
        check((dw0 & 0x3u) == aql::kAqlPacketTypeKernelDispatch, "dw0 type == 2");
        check((dw0 & (1u << aql::kAqlDw0BarrierBit)) != 0, "dw0 barrier == 1");
        check((dw0 & (1u << aql::kAqlDw0DimensionsBit)) != 0, "dw0 dims == 1");
    }
    check((none & (0x3u << aql::kAqlDw0AcquireBit)) == 0, "NONE acquire == 0");
    check((none & (0x3u << aql::kAqlDw0ReleaseBit)) == 0, "NONE release == 0");
}

void test_policy_values() {
    check(aql::kDefaultAqlPolicy.acquire_scope == aql::AqlFenceScope::System,
          "default acquire System");
    check(aql::kDefaultAqlPolicy.release_scope == aql::AqlFenceScope::System,
          "default release System");
    check(aql::kDefaultAqlPolicy.producer_visibility_fence, "default producer fence");
    check(aql::kMcuAqlFastPolicy.acquire_scope == aql::AqlFenceScope::None,
          "fast acquire None");
    check(aql::kMcuAqlFastPolicy.release_scope == aql::AqlFenceScope::None,
          "fast release None");
    check(!aql::kMcuAqlFastPolicy.producer_visibility_fence, "fast no producer fence");
}

void test_template_layout() {
    aql::GpuAqlDispatchDesc desc{};
    desc.kernel_object = 0xDEADBEEFCAFEBABEull;
    desc.global_work_items_x = 4096;
    desc.global_work_items_y = 1;
    desc.global_work_items_z = 1;
    desc.workgroup_size_x = 256;
    desc.workgroup_size_y = 1;
    desc.workgroup_size_z = 1;
    desc.private_segment_size = 128;
    desc.group_segment_size = 14340;
    desc.kernarg_address = reinterpret_cast<void*>(0x1234567890ull);
    desc.completion_signal = 0;

    const aql::GpuAqlPacketTemplate t =
        aql::build_aql_packet_template(desc, aql::kMcuAqlFastPolicy);
    const auto* raw = reinterpret_cast<const unsigned char*>(t.bytes);
    check(*reinterpret_cast<const uint32_t*>(raw + aql::kAqlDw0Offset) ==
              aql::kAqlInvalidDw0,
          "template dw0 INVALID");
    check(*reinterpret_cast<const uint16_t*>(raw + aql::kAqlOffWgX) == 256, "wg.x");
    check(*reinterpret_cast<const uint16_t*>(raw + aql::kAqlOffWgY) == 1, "wg.y");
    check(*reinterpret_cast<const uint16_t*>(raw + aql::kAqlOffWgZ) == 1, "wg.z");
    check(*reinterpret_cast<const uint32_t*>(raw + aql::kAqlOffGridX) == 4096, "grid.x");
    check(*reinterpret_cast<const uint32_t*>(raw + aql::kAqlOffGridY) == 1, "grid.y");
    check(*reinterpret_cast<const uint32_t*>(raw + aql::kAqlOffGridZ) == 1, "grid.z");
    check(*reinterpret_cast<const uint32_t*>(raw + aql::kAqlOffPrivateSegment) == 128,
          "private segment");
    check(*reinterpret_cast<const uint32_t*>(raw + aql::kAqlOffGroupSegment) == 14340,
          "group segment");
    check(*reinterpret_cast<const uint64_t*>(raw + aql::kAqlOffKernelObject) ==
              desc.kernel_object,
          "kernel object");
    check(*reinterpret_cast<const uint64_t*>(raw + aql::kAqlOffKernargAddress) ==
              reinterpret_cast<uint64_t>(desc.kernarg_address),
          "kernarg address");
    check(*reinterpret_cast<const uint64_t*>(raw + aql::kAqlOffReserved1) == 0,
          "reserved1 zero");
    check(t.publish_dw0 == 0x10102u, "fast publish dw0 NONE/NONE");
}

void test_template_policy_switch() {
    aql::GpuAqlDispatchDesc desc{};
    desc.workgroup_size_x = 64;
    desc.global_work_items_x = 1;
    const auto fast = aql::build_aql_packet_template(desc, aql::kMcuAqlFastPolicy);
    const auto safe = aql::build_aql_packet_template(desc, aql::kDefaultAqlPolicy);
    check(fast.publish_dw0 == 0x10102u, "fast policy dw0");
    check(safe.publish_dw0 == 0x11502u, "default policy dw0");
}

void test_body_bundle() {
    aql::GpuAqlDispatchDesc desc{};
    desc.kernel_object = 0xA5A5A5A5A5A5A5A5ull;
    desc.global_work_items_x = 1234;
    desc.workgroup_size_x = 256;
    desc.group_segment_size = 777;
    const aql::GpuAqlPacketTemplate t =
        aql::build_aql_packet_template(desc, aql::kMcuAqlFastPolicy);
    const aql::AqlPacketBodyBundle b = aql::make_body_bundle(t.bytes);
    check(b.at4 == *reinterpret_cast<const uint32_t*>(t.bytes + 4), "bundle at4");
    check((b.at4 & 0xffffu) == 256u, "bundle at4 low16 == wg.x");
    check(b.at8 == *reinterpret_cast<const uint64_t*>(t.bytes + 8), "bundle at8");
    check(b.at16.x == *reinterpret_cast<const uint32_t*>(t.bytes + 16), "bundle at16.x");
    check(b.at48.w == *reinterpret_cast<const uint32_t*>(t.bytes + 52), "bundle at48.w");
}

void test_launch_metadata() {
    check(aql::aql_launch_metadata_offset(48) == 48, "meta offset(48) == 48");
    check(aql::aql_launch_metadata_offset(56) == 64, "meta offset(56) == 64");
    check(aql::aql_kernarg_slot_stride(48) == 64, "slot stride(48) == 64");
    check(aql::aql_kernarg_slot_stride(32) == 48, "slot stride(32) == 48");
    alignas(64) uint8_t slot[64] = {};
    aql::build_aql_launch_metadata(slot, 48, 16, 1, 1, 256);
    const auto* meta = reinterpret_cast<const uint32_t*>(slot + 48);
    check(meta[0] == 16, "meta workgroup count x");
    check(meta[1] == 1, "meta workgroup count y");
    check(meta[2] == 1, "meta workgroup count z");
    check(meta[3] == 256, "meta workgroup size x");
    check(slot[47] == 0 && slot[63] == 0, "meta does not clobber kernarg bytes");
}

void test_hidden_args() {
    check(aql::aql_hidden_args_offset(48) == 48, "hidden offset(48) == 48");
    check(aql::aql_hidden_args_offset(56) == 56, "hidden offset(56) == 56");
    check(aql::aql_hidden_args_end(48) == 72, "hidden end(48) == 72");
    check(aql::aql_hidden_args_fits(48, 72), "hidden fits(48, 72)");
    check(!aql::aql_hidden_args_fits(48, 48), "hidden does not fit(48, 48)");
    alignas(64) uint8_t slot[256] = {};
    aql::GpuAqlDispatchDesc desc{};
    desc.global_work_items_x = 130;
    desc.global_work_items_y = 1;
    desc.global_work_items_z = 1;
    desc.workgroup_size_x = 64;
    desc.workgroup_size_y = 1;
    desc.workgroup_size_z = 1;
    check(aql::build_aql_hidden_args(slot, 48, 256,
                                     aql::AqlHiddenArgsPolicy::IfFits, desc),
          "hidden args written when they fit");
    const std::size_t base = aql::aql_hidden_args_offset(48);
    const auto* words = reinterpret_cast<const uint32_t*>(slot + base);
    const auto* halfs = reinterpret_cast<const uint16_t*>(slot + base);
    const uint32_t block_count = words[aql::kAqlHiddenBlockCountX / 4];
    const uint32_t group_size = halfs[aql::kAqlHiddenGroupSizeX / 2];
    const uint32_t remainder = halfs[aql::kAqlHiddenRemainderX / 2];
    check(block_count == 2, "hidden block count x");
    check(group_size == 64, "hidden group size x");
    check(remainder == 2, "hidden remainder x");
    check(words[aql::kAqlHiddenBlockCountY / 4] == 1, "hidden block count y");
    check(words[aql::kAqlHiddenBlockCountZ / 4] == 1, "hidden block count z");
    check(halfs[aql::kAqlHiddenGroupSizeY / 2] == 1, "hidden group size y");
    check(halfs[aql::kAqlHiddenRemainderY / 2] == 0, "hidden remainder y");
    check(block_count * group_size + remainder == desc.global_work_items_x,
          "global work items == block count * group size + remainder");
    check(slot[47] == 0, "hidden args do not clobber explicit args");
    for (std::size_t i = base + 24; i < 128; ++i) {
        check(slot[i] == 0, "hidden args stay inside the declared block");
    }
    std::memset(slot, 0, sizeof(slot));
    check(aql::build_aql_hidden_args(slot, 56, 256,
                                     aql::AqlHiddenArgsPolicy::IfFits, desc),
          "hidden args written for explicit size 56");
    const auto* shifted = reinterpret_cast<const uint16_t*>(
        slot + aql::aql_hidden_args_offset(56));
    check(shifted[aql::kAqlHiddenGroupSizeX / 2] == 64,
          "hidden args follow the explicit argument size");
    check(slot[55] == 0, "hidden args offset 56 leaves byte 55 alone");

    std::memset(slot, 0, sizeof(slot));
    check(!aql::build_aql_hidden_args(slot, 48, 48,
                                      aql::AqlHiddenArgsPolicy::IfFits, desc),
          "IfFits skips when the segment is too small");
    for (std::size_t i = 48; i < 128; ++i) {
        check(slot[i] == 0, "IfFits never writes past the segment");
    }

    std::memset(slot, 0, sizeof(slot));
    check(!aql::build_aql_hidden_args(slot, 48, 48,
                                      aql::AqlHiddenArgsPolicy::Required, desc),
          "Required returns false when the segment is too small");
    for (std::size_t i = 48; i < 128; ++i) {
        check(slot[i] == 0, "Required never writes past the segment");
    }

    std::memset(slot, 0xCD, sizeof(slot));
    check(!aql::build_aql_hidden_args(slot, 48, 256,
                                      aql::AqlHiddenArgsPolicy::None, desc),
          "None reports no write");
    bool untouched = true;
    for (std::size_t i = 48; i < 72; ++i) untouched = untouched && slot[i] == 0xCD;
    check(untouched, "None leaves the slot untouched");
}

void test_status_names() {
    check(std::strcmp(aql::gpu_mcu_aql_status_name(0), "COMPLETE") == 0, "status 0");
    check(std::strcmp(aql::gpu_mcu_aql_status_name(1), "INVALID_PARAMS") == 0, "status 1");
    check(std::strcmp(aql::gpu_mcu_aql_status_name(2), "RESERVE_TIMEOUT") == 0, "status 2");
    check(std::strcmp(aql::gpu_mcu_aql_status_name(99), "UNKNOWN") == 0, "status unknown");
}

}  // namespace

int main() {
    test_dw0_fields();
    test_policy_values();
    test_template_layout();
    test_template_policy_switch();
    test_body_bundle();
    test_launch_metadata();
    test_hidden_args();
    test_status_names();
    if (g_failed != 0) {
        std::printf("test_gpu_mcu_aql_packet: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_aql_packet: PASS\n");
    return 0;
}
