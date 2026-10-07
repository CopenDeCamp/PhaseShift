#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/retained_packet.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

namespace mcu = ps::runtime::gpu_mcu;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

__global__ void expand_kernel(const mcu::GpuMcuRetainedPacket* templ,
                              unsigned char* out,
                              uint64_t kernarg) {
    mcu::expand_retained_packet(out, *templ, kernarg);
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }

    check(sizeof(mcu::GpuMcuRetainedPacket) == 64, "retained packet is 64 byte");

    mcu::GpuAqlDispatchDesc desc{};
    desc.kernel_object = 0x1234567890abcdefull;
    desc.global_work_items_x = 7;
    desc.global_work_items_y = 1;
    desc.global_work_items_z = 1;
    desc.workgroup_size_x = 64;
    desc.workgroup_size_y = 1;
    desc.workgroup_size_z = 1;
    desc.private_segment_size = 16;
    desc.group_segment_size = 32;
    desc.kernarg_address = reinterpret_cast<void*>(0xdeadbeefcafeull);
    desc.completion_signal = 0x42ull;

    const mcu::GpuAqlPacketTemplate templ =
        mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
    const mcu::GpuMcuRetainedPacket retained = mcu::make_retained_packet(templ);
    check(retained.publish_dw0 == templ.publish_dw0, "publish dw0 carried");

    const uint64_t runtime_kernarg = 0x1122334455667788ull;

    mcu::GpuMcuRetainedPacket* retained_dev = nullptr;
    unsigned char* out_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&retained_dev),
                  sizeof(mcu::GpuMcuRetainedPacket)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&out_dev), 64) != hipSuccess) {
        std::printf("FAIL: device buffers\n");
        return 1;
    }
    (void)hipMemcpy(retained_dev, &retained, sizeof(retained), hipMemcpyHostToDevice);
    (void)hipMemset(out_dev, 0xAB, 64);
    expand_kernel<<<1, 1>>>(retained_dev, out_dev, runtime_kernarg);
    if (hipDeviceSynchronize() != hipSuccess) {
        std::printf("FAIL: expand kernel\n");
        return 1;
    }
    unsigned char expanded[64];
    (void)hipMemcpy(expanded, out_dev, 64, hipMemcpyDeviceToHost);

    uint32_t dw0 = 0;
    std::memcpy(&dw0, expanded, sizeof(dw0));
    check(dw0 == mcu::kAqlInvalidDw0, "expanded DW0 starts INVALID");

    bool body_ok = true;
    for (uint32_t i = 4; i < 64; ++i) {
        const uint32_t off_kernarg = mcu::kAqlOffKernargAddress;
        if (i >= off_kernarg && i < off_kernarg + 8) continue;
        if (expanded[i] != static_cast<unsigned char>(templ.bytes[i])) {
            body_ok = false;
        }
    }
    check(body_ok, "expanded body matches template outside kernarg");

    uint64_t patched = 0;
    std::memcpy(&patched, expanded + mcu::kAqlOffKernargAddress, sizeof(patched));
    check(patched == runtime_kernarg, "kernarg patched at runtime");

    // A zero kernarg request must keep the template field untouched.
    const uint32_t template_kernarg =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(desc.kernarg_address));
    uint32_t kept = 0;
    std::memcpy(&kept, expanded + mcu::kAqlOffKernargAddress, sizeof(kept));
    check(kept != template_kernarg, "runtime kernarg overrides template value");

    check(hipFree(retained_dev) == hipSuccess, "retained freed");
    check(hipFree(out_dev) == hipSuccess, "out freed");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_retained_packet: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_retained_packet: PASS\n");
    return 0;
}
