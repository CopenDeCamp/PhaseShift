#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/runtime/gpu_mcu/execution/persistent_mcu.h>

namespace ps::qwen35::runtime {

// Hands the KV addressing of the model to the persistent MCU.
//
// The MCU allocates KV pages itself, so it has to publish the page ids into the
// table the attention and kv append kernels read, and the page size it counts
// has to be the one the model addresses with. A physical sequence slot is one to
// one with the model slot, so the resolved slot is the block table row.
//
// The session that owns the persistent MCU calls this before it starts the MCU,
// so the device resource manager writes into the model's own table and the
// kernels never learn that a resource manager exists.
inline Status bind_mcu_kv_addressing(
    ::ps::runtime::gpu_mcu::GpuMcuPersistentMcu& persistent,
    const SequenceBlockTableDeviceView& block_table,
    const PagedKVPoolBF16View& kv_bf16,
    const PagedKVPoolFP8View& kv_fp8) {
    if (block_table.block_tables == nullptr ||
        block_table.block_table_stride == 0u) {
        return Status::invalid_state("the model has no sequence block table",
                                     __FILE__, __LINE__);
    }
    const uint32_t page_tokens =
        kv_bf16.page_tokens != 0u ? kv_bf16.page_tokens : kv_fp8.page_tokens;
    if (page_tokens == 0u) {
        return Status::invalid_state("the model has no paged kv pool", __FILE__,
                                     __LINE__);
    }
    return persistent.configure_kv_blocks(block_table.block_tables,
                                          block_table.block_table_stride,
                                          page_tokens);
}

}  // namespace ps::qwen35::runtime
