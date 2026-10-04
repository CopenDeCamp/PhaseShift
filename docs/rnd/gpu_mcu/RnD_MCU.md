> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# GPU-MCU Low-Level Substrate 移植レポート

PhaseShift-main（`~/wk/PhaseShift`）から PhaseNonShift へ、
GPU-MCU の低レイヤー実行基盤のみを移植した作業の記録。

- 目的: GPU-MCU Backend 本体ではなく、再利用可能・再現可能な低レイヤー基盤の復活
- 対象外: GPU-MCU の command ABI、Program materialization、Host/MCU 責務、
  FSM、Qwen3.5 との接続
- 詳細契約: `docs/developer/gpu_mcu/low_level.md`
- 測定結果: 本ドキュメント第 14 節（bench が SoT）

## 1. 基点とマージ

| 項目 | 値 |
|---|---|
| 作業開始 base | `940d10b92158cc71371cb0747ff74279a97517d1` |
| rebase 後 base | `c5a64470`（作業中に main が進行したため） |
| 作業 branch | `port/ps_gpumcu` |
| コミット | `6dd5859d` feat / `c9d54964` test / `4560bb65` bench |
| マージ | `b0d5eadf Merge pull request 'port/ps_gpumcu' (#25)` |
| Source | `~/wk/PhaseShift`（HEAD `6aa88b0`） |

## 2. Source から参照したファイル

`gpu_aql_fastpath.h/.hip`, `gpu_control_plane.h/.hip`,
`control/control_ring.h/.hip`, `control/control_record.h`,
`control/control_event.h`, `gpu_mcu_worker_image.h/.cpp`,
`gpu_control_sequencer.hip`,
`tests/unit/test_gpu_mcu_aql_worker_mask.hip`,
`test_gpu_mcu_worker_code_object.hip`, `test_gpu_control_ring.hip`,
`test_gpu_control_plane.hip`,
`tests/experiments/mcu_aql_fastpath/test_mcu_aql_fastpath.hip`,
`tests/kernels/common/test_grid_barrier_cost.hip`,
`cmake/targets.cmake`, `cmake/tests.cmake`,
`docs/developer/gpu_mcu/m0/CURRENT_IMPLEMENTATION.md`, `m0/BLOCKERS.md`

Source の設計は正とせず、Target の現行 architecture と AGENTS.md を正とした。

## 3. 移植した低レイヤー機構

- AQL packet layout（64 byte、DW0 offset、`hsa_kernel_dispatch_packet_t` との
  static_assert 照合）
- `AqlFenceScope` / `AqlMemoryPolicy`（SYSTEM/SYSTEM と NONE/NONE の区別）
- `DeviceAqlQueueView` / dispatch descriptor / packet template
- packet body copy / DW0 INVALID 化 / DW0 publish / doorbell write
- queue reservation / queue caught-up wait
- 単一 producer 用 streaming publish（reserve 省略、relaxed descriptor 更新）
  と DW0 barrier bit の選択（overlap 許可 / 強制直列）
- GPU 側 batch publication
- HSA queue create・destroy / kernarg allocation / code object load /
  symbol resolve（`.kd` fallback・symbol iteration fallback）/
  kernel_object・segment size 取得
- mapped coherent SPSC ring transport（Vyukov bounded sequence、
  device system-scope acquire/release、host `std::atomic_ref`、
  `hipHostMallocMapped | hipHostMallocCoherent`、host native atomic check）
- persistent kernel の start / heartbeat / shutdown
- worker HSACO の build / embed / memory load

## 4. 意図的に移植しなかった旧上位機構

`GpuMcuProgram` / `GpuKernelBinding` / `GpuDispatchBinding` /
`PhysicalKernelPlan` / `gpu_mcu_command.h` / 旧 generic `KernelConfig` /
旧 `DeviceExecutionContextEntry` / 旧 ExecutionContext ABI /
`RequestControlState` / `ControlFSM` / `OperationToken` / `BoundaryEvent` /
`integrated_control_state` / `gpu_control_sequencer.hip` /
`gpu_control_emitter.hip` / 旧 `GpuControlPlane` class /
旧 `gpu_mcu_runtime.hip` walker / 旧 Qwen3.5 GPU-MCU runtime state /
旧 Program lowering・command lowering / 旧 M1-M3 上位制御契約 /
`DispatchExecutionPath` と Graph fallback policy / `McuAqlFastContract` /
`ControlRecordBody`・`ControlEventBody` の意味論。

`valid_wgp_pair_mask()` は無条件には移植せず、HSA queue CU mask の
WGP pairwise 規則として CU 数付きで再定義した。

## 5. 新しい file tree

```text
include/phaseshift/runtime/gpu_mcu/
    aql.h
    completion.h
    control_ring.h
    cu_partition.h
    persistent_mcu.h
    worker_image.h
src/phaseshift/runtime/gpu_mcu/
    aql.hip
    cu_partition.hip
    control_ring.hip
    persistent_mcu.hip
    worker_probe.hip
    worker_image.cpp
src/apps/bench/
    gpu_dispatch.hip
    gpu_sync.hip
    gpu_memory.hip
    gpu_ext_dispatch.hip
tests/unit/
    test_gpu_mcu_aql_packet.hip
    test_gpu_mcu_aql_queue.hip
    test_gpu_mcu_cu_partition.hip
    test_gpu_mcu_worker_code_object.hip
    test_gpu_mcu_device_enqueue.hip
    test_gpu_mcu_control_ring.hip
    test_gpu_mcu_persistent_mcu.hip
    test_gpu_mcu_persistent_emit.hip
docs/developer/gpu_mcu/
    low_level.md
```

新 target `phaseshift_gpu_mcu`（STATIC）。依存は
`phaseshift_core` / `phaseshift_gpu` / `hip::host` / `hsa-runtime64::hsa-runtime64`
のみで、Qwen3.5 を一切知らない。`phaseshift` aggregate へは追加していない。
link するのは低レイヤーテストと `phaseshift-bench` の GPU-MCU RnD subcommand のみ。

## 6. HSA / HIP agent mapping

- HIP device → `hipDeviceGetPCIBusId` で PCI BDF を取得
- HSA agent を iterate し、`gfx1201` かつ kernel dispatch 可能で、
  `HSA_AMD_AGENT_INFO_BDFID` が HIP の BDF と完全一致する agent のみ採用
- 一致しない場合は fail-closed。「最初に見つかった gfx1201」は使用しない
- 本機は gfx1201 ×4 + gfx1036 iGPU ×1 が見えるため、BDF 完全一致は
  TP=2 / multi-GPU の前提条件

## 7. CU / WGP mask semantics（実測）

- `hipDeviceProp_t::multiProcessorCount` = 32（R9700）
- HIP stream mask は **1 bit = 1 WGP**。full mask = 32 bit（`0xffffffff`）で
  物理 64 CU 全体を覆う。`hipExtStreamGetCUMask` は要求 mask をそのまま返す
- HSA queue mask は **1 bit = 1 CU**。`HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT`
  = 64。even-index contiguous pairwise 必須（`0x3` / `0xfffffffc` は受理、
  `0x1` / `0x2` / `0xfffffffe` は `HSA_STATUS_ERROR_INVALID_ARGUMENT`）
- `hsa_amd_queue_cu_get_mask` は queue 制限ではなく agent-wide mask を返す
- mask unit 数スケーリング（32/16/8/4/1）:
  0.25 / 0.44 / 0.80 / 1.52 / 5.99 ms

| 層 | unit | bit 数 | pairwise |
|---|---|---|---|
| HIP stream mask | WGP | 32 | 不要 |
| HSA queue mask | CU | 64 | 必須（even-index 2 連続） |

この差のため `GpuMcuCuPartition` は HIP 用 mask と HSA 用 CU mask を別々に持つ。

## 8. Control WGP / Worker reservation

- physical: 32 WGP / 64 CU
- control: 1 WGP = 2 CU（HIP bit0 / HSA CU0-1）
- worker: 31 WGP = 62 CU（HIP bit1-31 / HSA CU2-63）

重複なし・和集合が全体・HSA mask が pairwise・HIP stream mask が readback 一致、
を `create()` で検証する。

## 9. Worker HSACO

- `worker_probe.hip` を `hipcc --genco` で standalone HSACO 化し、
  `worker_image.cpp` に embed
- `--genco` 出力は 4096 byte の `__CLANG_OFFLOAD_BUNDLE__` header + bare ELF。
  `dd bs=4096 skip=1` で抽出（CMake 内に局所化、build directory 下に生成）
- `worker_image.h` は data pointer と byte 数のみ公開し、file path を
  runtime requirement にしない
- `kernel_object != 0` を検証（kernarg segment = 288 byte）

## 10. GPU producer → AQL → worker qualification

`test_gpu_mcu_device_enqueue`（4 回反復）:

- Host: worker HSA queue（worker CU mask）/ worker image load / kernarg pre-stage /
  packet template pre-stage
- GPU: control stream 上の producer kernel が packet reserve → body → publish →
  doorbell、その後 queue caught-up を確認
- Worker: probe worker 起動 → output と completion 更新
- 検証: expected output、packet 消費、completion state/sequence/detail、
  queue corruption なし、timeout なし → PASS

## 11. Persistent MCU qualification

`test_gpu_mcu_persistent_mcu`:

- control stream = control WGP mask、worker stream = worker mask を readback 確認
- create / start / heartbeat 進行 / worker stream 上で独立 probe 実行 /
  request_stop / bounded 停止 / shutdown を 3 回反復、recreate も確認
- persistent 稼働中の GPU 操作は明示した control / worker stream 上で行い、
  legacy/default stream 操作を避ける（旧 PhaseShift でも観測された待ちの回避）
- 空の create / shutdown も確認

`test_gpu_mcu_persistent_emit`（persistent kernel → AQL → doorbell）:

- control WGP 上の persistent kernel が mapped な `submit_request` を見て
  worker queue へ 1 packet ずつ publish + doorbell する
- Source の `gpu_control_emitter.hip` と同じ構造。FSM / command ABI は
  持ち込まない（単調カウンタのみ）
- 検証: published 数、publish failure 0、全 worker completion、output 値、
  worker start timestamp 単調、heartbeat 進行、create/start/emit/stop を 2 回反復
  （barrier=1 と barrier=0 を各 1 回、いずれも streaming publish）
- device 側 publish は `gpu_mcu_aql_publish_batch`（`gpu_aql_fastpath` の
  device publish 本体）と `gpu_mcu_aql_publish_streaming`（単一 producer 用の
  reserve 省略経路）を使い分ける

## 12. Control ring stress

`test_gpu_mcu_control_ring`:

- mapped coherent SPSC ring、64 byte slot（sequence + 56 byte opaque payload）
- 1,000,000 records、4.1 s、約 0.24 M/s
- lost = 0 / duplicate = 0 / reorder = 0 / torn = 0、wrap around 正常
- device system-scope atomic 非対応時は volatile へ fallback せず fail-closed
- host native atomic 非対応も fail-closed

## 13. Completion / error

- `GpuMcuCompletion`（64 byte）: `sequence` / `state`
  (idle/running/complete/error) / `detail` / `aux`
- device は system-scope release、host は mapped coherent memory を acquire load
- `sequence` で stale completion を識別
- モデル固有 status / KernelId / command ABI は入れない

## 14. RnD 測定（観測値、設計定数ではない）

`phaseshift-bench` の GPU-MCU RnD subcommand で再現可能。値は観測値であり、
`constexpr` threshold や hard-coded policy へはまだ入れない。

測定条件:

- gfx1201 / AMD Radeon AI PRO R9700 ×4（測定は device 0）
- ROCm 7.14、clang 23
- Release build、clock 固定なし（自動 clock 変更は行わない）

### 14.1 gpu-dispatch

HIP empty kernel chain（`--variant hip --n 64 --iterations 100 --work 0`）:

```text
chain: nodes_ran=64/64 start_monotonic=yes
chain_total_us p50=182.80 p95=194.40 min=179.76
per_kernel_us  p50=0.920  p95=1.120  min=0.720
inter_kernel_gap_us p50=1.840 p95=2.040 min=1.760
start_to_start_period_us p50=2.800 p95=3.040 min=2.560
```

HIP Graph empty kernel（`--variant hip-graph --n 64 --iterations 100 --work 0`）:

```text
graph: nodes_ran=64/64 start_monotonic=yes
replay_total_us p50=158.52 p95=168.80 min=155.44
per_node_us     p50=0.200  p95=0.240  min=0.120
inter_node_gap_us p50=2.200 p95=2.440 min=1.800
start_to_start_period_us p50=2.360 p95=2.640 min=1.960
```

- `per_node_us 0.20 us` は **in-kernel body のみ**。node コストではない。
  node コストは start-to-start period の約 **2.36 us**（chain は 2.80 us）。
- body の chain/graph 差 ~0.8 us は end timestamp の観測位置のずれ。
  `--work 2000` で chain body 14.32 / graph body 13.56 と差は一定のままで、
  真の速度差ではない。
- graph は chain より node あたり約 15% 速い程度。dispatch は sub-us ではない。
- 64/64 node 実行・start 単調・work 比例を各条件で確認している。
- `--work N` はカーネル内 FMA 反復回数で、body 測定が work に比例することを
  確認するためのノブ。

GPU producer -> AQL worker（`--variant device-aql --batch 1 --iterations 300`）。
GPU producer と worker を同一 GPU timestamp domain で測定する。

```text
producer_submit_us            p50=5.320  p95=5.760  min=5.000
producer_to_worker_start_us   p50=23.520 p95=24.840 min=22.680
worker_duration_us            p50=0.200  p95=0.240  min=0.200
stage breakdown p50: reserve=3.600 body=0.400 publish=0.240 doorbell=1.040
                     doorbell_to_worker=18.200
```

1 packet を publish して毎回 host が同期する構成では worker queue が idle に
戻るため初回 dispatch が 18 us かかる。これは「kick latency」ではなく
**cold queue の初回遅延**である。

GPU producer -> AQL worker 連続 kick
（`--variant device-aql-chain --n 64 --iterations 300`）:

```text
cold_doorbell_to_first_worker_us        p50=18.360 p95=19.080 min=17.720
hot_prev_worker_end_to_next_start_us    p50=3.000  p95=3.280  min=2.840
worker_duration_us                      p50=0.240  p95=0.240  min=0.160
```

- **GPU-MCU からの kick は hot path で約 3.0 us**。18 us は idle からの初回のみ。
- HIP kernel chain の inter-kernel gap 1.88 us と同オーダー。

persistent emitter 経路（`--variant device-aql-plc`、`test_gpu_mcu_persistent_emit`）:

control WGP 上の persistent kernel が、mapped な `submit_request` を見て
worker queue へ 1 packet ずつ publish + doorbell する。transient producer の
ように毎回 kernel launch しない。Source の `gpu_control_emitter.hip` と同じ
構造（FSM / command ABI は持ち込まない）。

streaming publish（`gpu_mcu_aql_publish_streaming`）は単一 producer 前提で

- write_index / read_index を per-packet に読まない
- write_index を MCU がローカル保持し、descriptor 更新は relaxed store
- body write -> DW0 publish（barrier は引数）-> doorbell

を行う。p50（n=64、`--barrier`）:

```text
barrier=1（強制直列）:
  publish_to_doorbell_us     2.880
  worker_start_interval_us   4.600
barrier=0（overlap 可）:
  publish_to_doorbell_us     2.800
  worker_start_interval_us   1.040
```

- enqueue コストは barrier bit に依存しない（どちらも ~2.8 us）。
  transient stage の relaxed 経路 1.8 us との差は、persistent loop の
  heartbeat / submit_request の sysmem 更新が同じ loop に乗るため。
- **barrier=0 は worker 間隔を 4.60 -> 1.04 us に短縮**する。worker が独立なら
  overlap させ、依存があるものだけ barrier=1 にすればよい。これが
  「LLM の逐次性を崩す」現実的な手段（EXT_KERNEL_DISPATCH は本環境で不可）。
- MCU が write index を保持するため、queue 溢れ防止の `read_index` 確認は
  batch 境界で行う（per-packet では行わない）。

Host producer AQL（`--variant host-aql --iterations 200`）:

```text
HOST_AQL_SUBMIT_COST_us             p50=0.171 p95=0.331 min=0.160
HOST_AQL_SUBMIT_TO_COMPLETION_us    p50=103.295 p95=103.505 min=62.599
```

CPU clock と GPU clock を引いて「GPU dispatch latency」とは呼ばない。wall は
host polling の粒度に支配されるため submit cost と混同しない。

queue memory placement（`--variant queue-mem`）:

`hsa_queue_create` の AQL descriptor は system memory 上にある。GPU からの
cold load（L2 flush 後）:

```text
hipMalloc VRAM            163.4 ns
hipHostMallocMapped sys   816.8 ns
hipHostMalloc pinned sys  816.8 ns
hsa_queue ring_base       816.8 ns
hsa_queue write_index     816.8 ns
hsa_queue read_index      816.8 ns
hsa_queue_t               857.6 ns
```

producer の reserve（write_index 読み + read_index 読み + atomic fetch_add）が
~3.5 us かかるのはこのため。これは「enqueue を ROCr の CPU API で行っている」
という意味ではない。enqueue 自体は **GPU 上の persistent kernel が device 側で
行う**（`gpu_mcu_aql_publish_batch`）。コストの出どころは producer が触る
queue descriptor が system memory 上にあることだけである。

Device 側 publish の stage 内訳（`--variant device-aql-stage`、transient
producer、`--work` は bit0=read read_index / bit1=read write_index /
bit2=seq_cst、p50 us、reps=64）:

```text
mode=7 (read W + read R + seq_cst): readW 0.80  readR 0.80  fetch_add 1.00  total 4.32
mode=6 (read W          + seq_cst): readW 0.80  readR 0.04  fetch_add 1.00  total 3.52
mode=4 (                  seq_cst): readW 0.04  readR 0.00  fetch_add 1.00  total 2.72
mode=0 (                  relaxed): readW 0.04  readR 0.04  fetch_add 0.04  total 1.80
```

- `__ATOMIC_SEQ_CST` の fetch_add は ~1.0 us、`__ATOMIC_RELAXED` は ~0.04 us（25 倍差）。
  queue descriptor への seq_cst atomic が sysmem coherent の往復を強制している。
- read_write_index / read_read_index はそれぞれ ~0.8 us。
- **MCU が write index を保持し、per-packet の read と seq_cst を外すと
  hot path は total enqueue ≈ 1.76-1.80 us**
  （body 1.08 + publish 0.24 + doorbell 0.36 + relaxed fetch_add 0.04）。
  これは「body 0.40 + publish 0.24 + doorbell 1.00 ≒ 1.64 us」という
  hot path 像と一致する範囲である（stage 境界のぶれを含む）。
- relaxed fetch_add が安全なのは単一 producer（MCU）の場合のみ。host producer
  経路は seq_cst のままにする。
- 逐次性は barrier bit で決まる。MCU が「現在実行中の kernel 群の最後の種類」を
  見て、先行計算してよいなら barrier=0、依存があるなら barrier=1 で発射する。
  enqueue コスト自体は barrier に依存しない。

Source PhaseShift の reference（`experiments/amdgpu_device_dispatch` Phase 6、
`__clock64()` tick ≈ 0.352 ns = 2.84 GHz を本機で実測校正）:

```text
reserve (fetch_add)     110 ticks ≈ 39 ns
packet (packet 書込)     85 ticks ≈ 30 ns
publish (header store)   22 ticks ≈  8 ns
doorbell                 35 ticks ≈ 12 ns
total enqueue           297 ticks ≈ 105 ns
doorbell->child        1200 ticks ≈ 422 ns
```

Source は `__builtin_amdgcn_queue_ptr()` で **自分が走っている queue** を取得して
そこへ enqueue し、`read_dispatch_id` は読まず `write_dispatch_id` の fetch_add
のみを行っている。本移植は dedicated worker queue に対して write/read 両方を
読むため、descriptor access が Source より多い。また `doorbell->child` の
422 ns は独立 dispatch（barrier 依存を前提としない）側の値であり、逐次依存の
ある処理の指標にはそのまま使えない。

`HSA_ALLOCATE_QUEUE_DEV_MEM=1`（要 large BAR、本機は 32 GB BAR 有効）で
**ring buffer のみ** VRAM 化できる:

```text
ring_base  817 -> 245 ns
write_index / read_index / hsa_queue_t  857 ns のまま
reserve    3.5 us のまま / producer->worker 23.5 us のまま
```

設計上の帰結（Step 4 で判断）:

- producer 側は **batch 化**で reserve/doorbell を償却する。
- producer 自身の write_index を VRAM / レジスタに保持し、CP の read_index は
  batch 単位で 1 回だけ読む。per-packet の sysmem アクセスを削れる。
- 自 queue への enqueue（`__builtin_amdgcn_queue_ptr()`、Source 方式）と
  dedicated worker queue への enqueue は descriptor の hot さが異なる。
  どちらを採るかは command ABI / ownership と合わせて判断する。
- worker kick 自体は hot で 3.0 us であり、sysmem コストは producer の enqueue
  側に限定される。

GPU timestamp 校正: `wall_clock64()` は約 97.9 ticks/us（1 tick ≈ 10.2 ns、
約 100 MHz）。`/100` の us 換算は hipEvent 比較で誤差 2% 以内。絶対値はこの校正に依る。

### 14.2 gpu-sync（`--n 1000 --iterations 20`）

```text
loop_overhead          total_us=   8.320 per_iter_ns=  8.32
syncthreads            total_us=  19.320 per_iter_ns= 19.32
lds_syncthreads        total_us=  50.720 per_iter_ns= 50.72
threadfence            total_us=   8.400 per_iter_ns=  8.40
threadfence_system     total_us=  21.040 per_iter_ns= 21.04
grid_sync              total_us= 607.601 per_iter_ns=607.60
hipEventSynchronize    p50_us=20.108
hipStreamSynchronize   p50_us=21.090
hsa_signal_wait(satisfied) p50_us=0.040
```

- 空 loop との差し引きで `__syncthreads` は約 11 ns、LDS を跨ぐ barrier は
  約 42 ns、`__threadfence` は空 loop と区別できない範囲、
  `__threadfence_system` は約 13 ns。
- grid barrier は resident grid（mp × per_mp = 256 blocks, 256 threads）で
  約 608 ns/iter。resident 条件を満たす構成でのみ実行する。
- host 側 sync は約 20 us の固定 overhead を持つ。

### 14.3 gpu-memory

global working-set sweep（copy、read+write を計上）:

```text
bytes        vec    median_us    GB/s
4096         1        10.000      0.8
4096         4         9.560      0.9
262144       1        10.400     50.4
262144       4         9.800     53.5
4194304      1        30.000    279.6
4194304      4        12.600    665.8
33554432     1        56.200   1194.1
33554432     4        54.840   1223.7
134217728    1       516.801    519.4
134217728    4       463.201    579.5
536870912    1      1947.443    551.4
536870912    4      1879.084    571.4
```

- 32 MB working set（L3 64 MB 内）は約 1.2 TB/s。
- 128 MB 以上は約 550-580 GB/s で DRAM 領域。
- L2（8 MB）と L3（64 MB）の境界は今後の sweep で詰める。

LDS probe（stride = lane 間隔、one access/thread/iteration）:

```text
kernel  stride   median_us   GB/s
lds32   1        24.560      2668.4
lds32   2        24.800      2642.6
lds32   4        27.720      2364.2
lds32   8        46.520      1408.8
lds32   16       84.640       774.3
lds32   32      161.121       406.8
lds128  1        27.320      9595.3
lds128  2        46.200      5674.1
lds128  4        84.360      3107.4
lds128  8       160.440      1633.9
lds128  16      160.600      1632.3
lds128  32      161.080      1627.4
```

- stride 1-4 はほぼ peak、stride 8/16/32 で段階的に低下する。
- LDS bank topology は決め打ちせず、この実測 crossover を記録するに留める。
- 4 byte / 16 byte の access width 差（約 2.7 vs 9.6 TB/s）が支配的。

### 14.4 再現方法

```sh
cmake --build build --target phaseshift-bench --parallel
./build/phaseshift-bench gpu-dispatch --variant hip
./build/phaseshift-bench gpu-dispatch --variant hip --work 2000
./build/phaseshift-bench gpu-dispatch --variant hip-graph
./build/phaseshift-bench gpu-dispatch --variant hip-graph --work 2000
./build/phaseshift-bench gpu-dispatch --variant device-aql
./build/phaseshift-bench gpu-dispatch --variant device-aql-stage
./build/phaseshift-bench gpu-dispatch --variant device-aql-chain --n 64
./build/phaseshift-bench gpu-dispatch --variant device-aql-plc --n 64 --barrier 1
./build/phaseshift-bench gpu-dispatch --variant device-aql-plc --n 64 --barrier 0
./build/phaseshift-bench gpu-dispatch --variant host-aql
./build/phaseshift-bench gpu-dispatch --variant queue-mem
./build/phaseshift-bench gpu-sync
./build/phaseshift-bench gpu-memory
./build/phaseshift-bench gpu-ext-dispatch
```

bench 自体が SoT であり、過去の `experiments/.../results/` は移植していない。

## 15. explicit dependency packet Gate 0

目的: LLM の逐次実行を崩すため、dispatch の完了 signal で次を自動発射する
explicit dependency packet が使えるかを確認する。

### header capability

- `hsa/hsa_ext_amd.h` に `HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH = 3` と
  `hsa_amd_ext_kernel_dispatch_packet_t`（64 byte、`dep_signal` offset 48、
  `completion_signal` offset 56）が存在
- `HSA_AMD_PACKET_TYPE_BARRIER_VALUE = 2` と
  `hsa_amd_barrier_value_packet_t`（64 byte、`signal` / `value` / `mask` / `cond`）
  も存在
- したがって **HEADER: AVAILABLE**

### agent capability（実測）

```text
HSA_AMD_AGENT_INFO_KERNEL_CLUSTER_MAX_DIM  dim=(2147483647,65535,65535)
HSA_AMD_AGENT_INFO_CLUSTER_MAX_DIM         dim=(1,1,1)
HSA_AMD_AGENT_INFO_CLUSTER_MAX_SIZE        = 1
```

gfx1201 は **cluster 非対応（cluster size = 1）**。EXT packet は non-clustered 形
のみ可能。

### runtime acceptance（実測）

`phaseshift-bench gpu-ext-dispatch` で最小 packet を投入した。

| packet | header | dep_signal | 結果 |
|---|---|---|---|
| EXT (format=3) | vendor + barrier + acq + rel | created signal | INVALID_PACKET_FORMAT |
| EXT (format=3) | vendor + barrier + acq + rel | handle 0 | INVALID_PACKET_FORMAT |
| EXT (format=3) | vendor のみ | created signal | INVALID_PACKET_FORMAT |
| EXT (format=3) | vendor のみ | handle 0 | INVALID_PACKET_FORMAT |
| BARRIER_VALUE (format=2) | vendor のみ | handle 0 | INVALID_PACKET_FORMAT |

いずれも runtime が queue を inactivate し process を abort する。
**RUNTIME: UNAVAILABLE_AT_RUNTIME**。

補強: `libhsa-runtime64.so` に cluster 実装を示す文字列が無く、
`CLUSTER_MAX_SIZE = 1` であることと整合する。同一の手動 publish 経路で
通常の KERNEL_DISPATCH（format=2、header type=2）は正常に dispatch できる
（`test_gpu_mcu_device_enqueue`）ので、publish 経路自体は正しい。

### 結論と影響

- explicit dependency packet による「完了 signal で次を自動発射」は、
  本環境（ROCr 7.14 / gfx1201）では**実現できない**。
- したがって dependent dispatch の PoC（Worker A → dep_signal → Worker B）は
  未実施。NONE/NONE policy は dependent dispatch へ流用しない。
- LLM の逐次性を崩す現実的な手段は当面:
  - MCU（persistent kernel）が device 側で次 packet を publish する PLC 方式
  - DW0 の barrier bit を依存に応じて 0/1 選択する（overlap 許可 / 強制直列）
  - これらは `--variant device-aql-stage` の relaxed 経路（total enqueue ~1.8 us）
    と組み合わせられる
- 将来 ROCr が vendor packet dispatch（cluster / explicit dependency）を実装
  した場合に再評価する。header は存在するので、Gate 0 を再実行するだけでよい。

## 16. acceptance / regression

- `ctest -L gpu_mcu`: 7/7 PASS
- `ctest -L required` / `run_required_acceptance.py`:
  rebase 後 **83/83 PASS、Skipped 0、ACCEPTANCE PASSED**
- Qwen3.5-4B Host Backend E2E: `phaseshift-compute` が oracle `raw_compute` と
  完全一致（`GENERATED_IDS=220,19,11`）。GPU-MCU は Qwen runtime へ未接続
- static grep gate: 新 tree に `GpuMcuProgram` / `GpuKernelBinding` /
  `PhysicalKernelPlan` / `gpu_mcu_program` / `gpu_kernel_binding` /
  `qwen35` は 0 件。`models/qwen35/runtime/` に GPU-MCU 分岐なし

## 17. 境界

完了時点は次の状態:

```text
Host Backend -> current Program -> current kernel runtime   （従来どおり）
GPU-MCU Low-Level Substrate                                  （独立・未接続）
```

現行 Program / DispatchBinding / KernelId / KernelConfig / Selector /
Host Backend / Qwen3.5 runtime は変更していない。
接続方式と command ABI の決定は次の設計段階で行う。

## 18. known limitations

- dependent dispatch は未採用（EXT Gate 0 が runtime 非対応のため）
- `hsa_amd_queue_cu_get_mask` は queue 制限を返さないため、placement は
  mask unit 数スケーリングで補助観測
- persistent MCU は command semantics を持たない（上位 FSM は未実装）
- `phaseshift_gpu_mcu` は `phaseshift` aggregate 未追加（意図的）

## 19. Source からの再利用 / 破棄（developer からの移管記録）

`docs/developer/gpu_mcu/low_level.md` の current contract 化に伴い、
そこにあった移植履歴をここへ移した。§3 / §4 と重複する項目は要約である。

再利用した低レイヤー機構:

- AQL packet layout（64 byte、DW0 offset、`hsa_kernel_dispatch_packet_t` 照合）
- `AqlFenceScope` / `AqlMemoryPolicy` と SYSTEM/SYSTEM・NONE/NONE の区別
- `DeviceAqlQueueView` / dispatch descriptor / packet template
- packet body copy、DW0 INVALID 化、DW0 publish、doorbell write
- queue reservation、queue caught-up wait
- GPU 側 batch publication
- HSA queue create/destroy、kernarg allocation、code object load、
  symbol resolve、kernel_object / segment size 取得
- mapped coherent SPSC ring の transport 機構（Vyukov bounded sequence、
  device system-scope acquire/release、host `std::atomic_ref`、
  `hipHostMallocMapped | hipHostMallocCoherent`、host native atomic
  capability check）
- persistent kernel の start / heartbeat / shutdown 構造

意図的に移植しなかった上位機構:

- `GpuMcuProgram` / `GpuKernelBinding` / `GpuDispatchBinding` /
  `PhysicalKernelPlan` / `gpu_mcu_command.h`
- `DispatchExecutionPath` / HipGraph fallback policy / `DispatchOwner` /
  `DispatchVisibilityDomain` / `DispatchDependencyKind` /
  `McuAqlFastContract` / `resolve_execution_path()`
- 旧 generic `KernelConfig` / 旧 `DeviceExecutionContextEntry` /
  旧 ExecutionContext ABI
- `RequestControlState` / `ControlFSM` / `OperationToken` /
  `BoundaryEvent` / `integrated_control_state`
- `gpu_control_sequencer.hip` / `gpu_control_emitter.hip` /
  旧 `GpuControlPlane` class
- 旧 `gpu_mcu_runtime.hip` の walker
- `ControlRecordBody` / `ControlEventBody` の意味論
- 旧 Qwen3.5 GPU-MCU runtime state、Program lowering、M1/M2/M3 上位制御契約

旧 `valid_wgp_pair_mask()` は無条件には移植していない。HSA queue CU mask の
WGP pairwise 規則として、正しい CU 数で再定義した。
