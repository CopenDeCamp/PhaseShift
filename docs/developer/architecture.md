# Architecture

## Layer構成

```text
apps / quantizer
     ↓
Qwen35 runtime
    Executor / ContinuousBatcher / scheduler / program_executor
     ↓                    ↓
Qwen35 model        Qwen35 kernels
    config / weights    kernels/correctness/ (1 TU + detail/*.inc) / kernels/optimized/ (family per file)
    lower_to_primitives
     ↓                    ↓
Generic weights loader
    safetensors / quantized safetensors
    ↓
canonical validation
    ↓
optional logical partition
    ↓
native preshuffle
    ↓
GPU upload
     ↓                    ↓
Qwen35 state  ────────────┤
     ↓                    ↓
Quantization / FPX format
     ↓
Runtime core
    PrimitiveGraph / Program / ProgramSet / DeviceBatchContext / workspace layouts
```

`phaseshift_qwen35_kernels` は `phaseshift_runtime` にlinkする一方、
`phaseshift_runtime` は `phaseshift_core` / `phaseshift_gpu` だけを使う。

Tensor Parallel を使う場合の scheduler / rank runtime の関係（TP-Exec-1）:

```text
ContinuousBatcher (global 1個)
      |
  TpCoordinator
   /      \
Rank0    Rank1          (1 rank = 1 device-local runtime: arena / stream /
  |        |              rank-local weight / KV pool / GDN state / Executor)
Executor Executor
```

詳細は [tensor_parallel_execution.md](tensor_parallel_execution.md) を参照。

weights loader（`phaseshift_weights`）の依存は下方向のみ:

```text
phaseshift_weights -> phaseshift_fpx_format / phaseshift_quant_reference / phaseshift_io / phaseshift_core / phaseshift_gpu
phaseshift_qwen35  -> phaseshift_weights
```

weight の canonical → partition → preshuffle → upload の順序 contract は
[tensor_partition.md](tensor_partition.md) を参照。

CMake targetの一覧は `cmake/targets.cmake` を参照。

## GPU-MCU

GPU-MCU は 3 つの target に分かれる。production architecture の正本は
[gpu_mcu/architecture.md](gpu_mcu/architecture.md) を参照。

```text
phaseshift_gpu_mcu         substrate（Qwen3.5 非依存）
    -> phaseshift_core / phaseshift_gpu / hip::host / hsa-runtime64
phaseshift_qwen35_gpu_mcu  Qwen3.5 kernel HSACO blob の所有者
    -> phaseshift_gpu_mcu
phaseshift_qwen35_runtime  Qwen3.5 runtime
    -> phaseshift_gpu_mcu, phaseshift_qwen35_gpu_mcu
```

- `include/phaseshift/runtime/gpu_mcu/**` と
  `src/phaseshift/runtime/gpu_mcu/**` は Qwen3.5 を一切参照しない
  （`test_architecture_boundaries` が runtime core に適用する規則と同じ）。
- `phaseshift` aggregate の明示リストには `phaseshift_gpu_mcu` を追加しない。
  しかし `phaseshift_qwen35_runtime` が PUBLIC で link するため、
  production binary は substrate と `hsa-runtime64` を実 link する。
  production decode が HSA に依存するのはこのためである。
- Qwen3.5 固有の kernel HSACO blob は substrate に置かず、
  `phaseshift_qwen35_gpu_mcu` が所有する。blob の消費者は
  `mcu_kernel_registry.hip` のみ。
- substrate の low-level 契約と実測は `docs/developer/gpu_mcu/low_level.md`。
- 現行 Program / DispatchBinding / KernelConfig / Host Backend は substrate が
  消費しない。plan への変換は Qwen3.5 backend 側で行う。


## 依存ルール（enforced by `test_architecture_boundaries`）

```text
runtime core -> Qwen35            : forbidden
weights / fpx / psq -> Qwen35     : forbidden（models/ の include 不可）
Qwen35 model -> Qwen35 runtime    : forbidden
Qwen35 kernels -> Qwen35 runtime  : forbidden
```

- `include/phaseshift/runtime/**` と `src/phaseshift/runtime/**` は
  Qwen35を一切参照しない（`qwen35` / `Qwen35` / `phaseshift/models/` が存在しない）。
- `include/phaseshift/weights/**` と `src/phaseshift/weights/**` も同様に
  Qwen35を一切参照しない（model非依存のweight loader）。
- `quantization/fpx/**` / `quantization/psq/**` / `core/**` / `io/**` は
  `phaseshift/models/` をincludeしない。
  （manifest schema上の architecture 文字列は format契約でありコード依存ではない）
- model layerはweights binding / config / primitive loweringのみ。
  weights loaderのformat分岐（BF16 / PSQ4 / PSQ8）はmodel側に持たない。
  runtime（Executor / scheduler / state ownership）をincludeしない。
- kernels layerはdevice correctness実装のみ。
  runtime（HostExecutionContext / Executor）をincludeしない。

## New model rule

新モデルが既存Primitiveで表現可能なら、
`include/phaseshift/runtime/**` と `src/phaseshift/runtime/**` を変更しない。

新model向けの汎用抽象（IModel / Backend / Registry / Factory / Plugin interface）は
作成しない。必要になった実装はmodel-local
（`include/phaseshift/models/<model>/**`）に置く。

## Server layer

`phaseshift-server` はserving productであり、inference engineではない。
aiohttp で OpenAI Chat Completions の subset を提供し、
`phaseshift-compute --serve-stdio` へ直接 JSONL stdio で接続する。

```text
OpenAI client
     │ HTTP / SSE
     ▼
phaseshift-server
     ├── aiohttp
     ├── openai_protocol.py   (request validation / wire encoding)
     ├── chat_service.py      (HF messages -> chat template -> compute -> response parser)
     └── compute_client.py    (asyncio native JSONL client)
     │ persistent JSONL stdio
     ▼
phaseshift-compute --serve-stdio   (token IDs in / token IDs out)
```

境界:

- `phaseshift-compute` はtoken IDのin/outのみ。HTTP・OpenAI schema・messages・
  roles・tools・JSON Schema・tool parser・chat template・tokenizerを持たない。
- server側（`src/apps/server/**`）だけがOpenAI schema・tokenizer・chat template・
  tool parserに依存する。
- `src/phaseshift/**` と `include/phaseshift/**` はserving固有の技術
  （HTTP / OpenAI schema / protobuf / gRPC / xgrammar）を参照しない。
  `test_architecture_boundaries` の `check_no_server_deps` がこれを静的に検査する。
- tokenizerとQwen chat templateの唯一のsource of truthはmodel directoryの
  Hugging Face processorである。server codecは `src/apps/common/phaseshift_chat/`
  に置き、`phaseshift-cli` とserverが共有する。

### compute control plane

`phaseshift-compute --serve-stdio` はGPU runtime ownerである単一thread上で、
`poll(2)` によりstdinを non-blockingに読む。generation中でも `ping` / `cancel` /
`shutdown` を処理でき、1 stepごとにcontrol commandを確認する。
`{"op":"cancel","request_id":N}` は対象requestを `finish_reason="cancelled"` で終了させ、
terminal eventは `done` の1個だけである。GPU runtimeを複数threadから触らない。

複数requestは同一の `ContinuousBatcher` へ投入され、1回の `runtime.step()` で
`ScheduledBatch` として co-batchされる。GPU runtime ownerは1 threadのままである。

```text
aiohttp handlers
    ↓
AsyncComputeClient per-request mailboxes
    ↓
JSONL multiplex (1 stdin writer / 1 stdout reader)
    ↓
single compute service loop (1 owner thread)
    ↓
ContinuousBatcher
    ↓
ScheduledBatch
    ↓
GPU
```

論理concurrency `N` (`--max-concurrent-requests`) とphysical KV budget
(`--kv-cache-capacity-tokens`) は別契約である。`SequenceSlotPool` /
`GdnStatePool` / `max_scheduled_requests` はNに一致させ、physical KV pagesは
budgetから決める。budget不足時はKV BankerがQueued requestをActive完了後に
admissionする。

serverの対応surfaceは [../user/server.md](../user/server.md) を参照。
