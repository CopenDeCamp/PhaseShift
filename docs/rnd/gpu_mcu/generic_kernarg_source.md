> Status: historical R&D record（現在の仕様ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# Generic kernarg source 化

- 目的: Generic GPU-MCU executor から kernel / model semantics を完全に追い出す
- 基点: branch `exp/gpu-mcu-phase3`（`d7c239b3`）
- 対象外: continuous batching 改修、persistent controller lifetime、
  Host submission contract、output ring の production 統合、KV allocator 再設計

## 1. 移行前の問題

`mcu_build_kernarg()` が recipe switch で 26 種の `mcu_write_*_kernarg()` を呼んでおり、
それぞれが `McuRmsNormInvocation` / `McuPagedAttentionInvocation` 等の
kernel 固有型を直接 read していた。

`GpuMcuFsmState` と `GpuMcuFsmConfig` は typed invocation pointer/count を 25 対ずつ持ち、
`micro_fsm_host.h` はその 25 対それぞれに validation を置いていた。

backend は `McuDecodeRuntime::prepare_plan()` で typed table を upload 済みであり、
upload 後に GPU-MCU がもう一度 recipe を見てどの table を読むか判断していたことが重複だった。

## 2. 導入した contract

`include/phaseshift/runtime/gpu_mcu/binding/kernarg_source_contract.h`:

```cpp
struct alignas(16) McuKernargSourceDesc {
    uint64_t source;
    uint32_t explicit_args_bytes;
    uint32_t flags;
};
```

| flag | 意味 |
|---|---|
| `kMcuKernargSourcePrepared` | `source` が完成済みの explicit kernarg bytes を指す |
| `kMcuKernargSourceSupervisorProbe` | MCU 内部の supervisor probe。`source` は null |

- `binding/` に置いた理由: logical invocation → physical GPU memory 上の kernarg source への
  binding だから。`execution/` は descriptor を消費するだけである。
- kernel 固有 field を持たせない。`rmsnorm_rows` / `attention_heads` 等は禁止。
- descriptor index = node index。`kernarg_sources[node_index]` がその node の source になる。

## 3. migration の順序

| 段階 | 内容 |
|---|---|
| 1 | `McuKernargSourceDesc` contract を追加 |
| 2 | generic prepared copy path（`mcu_write_prepared_kernarg`）を追加。legacy path は温存 |
| 3 | Qwen35 `prepare_plan()` が RMSNorm のみ descriptor を組み立てる |
| 4 | RMSNorm の parity test |
| 5 | test rig（`gpu_mcu_fsm_test_util.h`）が全 recipe の descriptor を組み立てる |
| 6-8 | Qwen35 backend を simple → attention / verify → GDN の順に移行 |
| 9 | recipe ごとの writer 26 種と legacy switch を削除 |
| 10 | `GpuMcuFsmState` / `GpuMcuFsmConfig` から typed invocation pointer を削除 |
| 11 | `mcu_recipe_explicit_args_bytes()` と `mcu_kernarg_recipe_supported()` を削除 |
| 12 | `invocation_abi.h` / `kernarg_recipe.h` を `models/qwen35/runtime/gpu_mcu/` へ移動 |
| 13 | GDN reset kernel を `models/qwen35/kernels/optimized/gdn/` へ移動、`model_hooks/` を廃止 |
| 14 | model-independent executor boundary の boundary test 追加 |

段階の分け方: production 経路（`McuDecodeRuntime::prepare_plan()`）は
doc の推奨順（RMSNorm → simple → attention / verify → GDN）に従い、
test rig は段階 5 で一括移行した。
test rig は 87 件の gpu_mcu テストすべてを流すため、
descriptor 機構に対する coverage を最初に最大化できるためである。

## 4. 採否判断

### 4.1 embedding の `token_ids` runtime override

移行前 embedding writer は dispatch 時に `state->batch_input` で
`token_ids` を上書きしていた。単純な byte copy では再現できない。

採用: `McuInvocationPatch` に `width` と `null_guard` を追加し、
`McuInvocationPatchSource::BatchTokenIds` を追加した。
Qwen35 backend が embedding node ごとに pointer 幅 patch を発行し、
executor は generic patch と generic byte copy のみを行う。

不採用候補: 侵襲ゼロだが batch 値の意味を executor に残す
「descriptor に field offset を持たせる」方式。kernel 固有 knowledge を
descriptor に持ち込むため不採用。

### 4.2 explicit args の copy 幅

4 byte 単位の copy loop では `test_gpu_mcu_real_rmsnorm_feed` の
queue-empty counter が 1000 → 1328 に悪化した（limit 1000）。
16 byte 単位の `__builtin_memcpy` に変えたら 1000 で baseline と一致した。

採用: 16 byte → 4 byte → byte の 3 段階 copy。
typed struct copy と同等の vectorized load/store が生成される。

### 4.3 supervisor probe の扱い

`CompletionMarker` は GPU-MCU 内部 kernel であり、prepared source には統合しない。
`kMcuKernargSourceSupervisorProbe` flag として executor 内部で組み立てる。
`source == 0` を host validation で強制する。

### 4.4 `kernarg_recipe` field

layout 変更を同時に行わないため `McuPlanNode::kernarg_recipe` は削除しない。
executor が読まないようにし、backend が recipe → invocation table の対応で
descriptor を組み立てるだけにした。default 値は `0` に変更した
（`fsm_contract.h` が recipe enum を include しなくなったため）。
dispatch node はすべて明示的に recipe を設定していることを確認済み。

## 5. 移行後の状況

| 対象 | 移行前 | 移行後 |
|---|---|---|
| `micro_fsm_kernarg.h` | 989 行 / writer 26 種 | 348 行 / prepared + probe の 2 種 |
| `GpuMcuFsmState` typed pointer | 25 対 | 0 |
| `GpuMcuFsmConfig` typed pointer | 25 対 | 0 |
| `micro_fsm_host.h` family validation | 25 対 | 0 |
| `runtime/gpu_mcu/**` の model invocation type | `invocation_abi.h` の 24 構造体 + `gdn_reset.h` の 1 | 0 |
| `runtime/gpu_mcu/**` の recipe id | `kernarg_recipe.h` の 30 entry | 0 |
| `model_hooks/` | 隔離先として存在 | 廃止 |
