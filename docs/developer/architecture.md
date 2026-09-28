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
    BF16・PSQ4 / PSQ8 の
    format解釈 / canonical validation / preshuffle / GPU upload
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

weights loader（`phaseshift_weights`）の依存は下方向のみ:

```text
phaseshift_weights -> phaseshift_fpx_format / phaseshift_quant_reference / phaseshift_io / phaseshift_core / phaseshift_gpu
phaseshift_qwen35  -> phaseshift_weights
```

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
LocalAI v4.10.0をhidden API frontendとして起動し、そのexternal gRPC backend
（`src/apps/server/backend/`）が `phaseshift-compute --serve-stdio` を所有する。

```text
OpenAI client
     │ HTTP
     ▼
phaseshift-server (launcher)
     │
     ▼
LocalAI v4.10.0
     │ private gRPC
     ▼
phaseshift backend (Python)
     │ persistent JSONL stdio
     ▼
phaseshift-compute --serve-stdio   (token IDs in / token IDs out)
```

境界:

- `phaseshift-compute` はtoken IDのin/outのみ。HTTP・OpenAI schema・messages・
  roles・tools・JSON Schema・tool parser・chat template・tokenizerを持たない。
- server側（`src/apps/server/**`）だけがLocalAI・gRPC・protobuf・OpenAI schema・
  tokenizer・chat template・tool parserに依存する。
- `src/phaseshift/**` と `include/phaseshift/**` はLocalAI / gRPC / protobuf /
  OpenAIを参照しない。`test_architecture_boundaries` の `check_no_server_deps` が
  これを静的に検査する。
- LocalAI protobuf stubは `vendor/localai/backend.proto` から生成し、
  `src/apps/server/localai_proto/` へ保持する。
- tokenizerとQwen3.5 chat templateの唯一のsource of truthはmodel directoryの
  Hugging Face processorである。server codecは `src/apps/common/phaseshift_chat/`
  に置き、`phaseshift-cli` とbackendが共有する。

### compute control plane

`phaseshift-compute --serve-stdio` はGPU runtime ownerである単一thread上で、
`poll(2)` によりstdinを non-blockingに読む。generation中でも `ping` / `cancel` /
`shutdown` を処理でき、1 stepごとにcontrol commandを確認する。
`{"op":"cancel","request_id":N}` は対象requestを `finish_reason="cancelled"` で終了させ、
terminal eventは `done` の1個だけである。GPU runtimeを複数threadから触らない。

複数requestは同一の `ContinuousBatcher` へ投入され、1回の `runtime.step()` で
`ScheduledBatch` として co-batchされる。GPU runtime ownerは1 threadのままである。

```text
HTTP threads
    ↓
LocalAI max-concurrent-backend-requests N
    ↓
backend gRPC workers
    ↓
ComputeClient per-request mailboxes
    ↓
JSONL multiplex
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

### grammar constraints

Grammar-constrained decodeはGPU runtime ownerである単一thread上で完結する。

責務境界は2つに分かれる。

```
server / control plane:
    OpenAI format
        ↓
    LocalAI
        ↓
    GBNF

runtime:
    GBNF
        ↓
    XGrammar matcher
        ↓
    token bitmask
        ↓
    sampling
```

```
LocalAI response_format / text.format
    ↓ (JSON Schema -> GBNF)
backend request.Grammar
    ↓ (structured text / tool / composition)
compute JSONL grammar or structural_tag field
    ↓
TokenConstraintCompiler (model lifetime)
    ↓
TokenConstraintState (request単位 matcher)
    ↓
compressed token bitmask (CPU)
    ↓ (1 batched H2D / step)
PhaseShift HIP sampling
```

computeが理解するのはtoken IDs、sampling、GBNF grammar、Structural Tagまでである。
OpenAI / JSON Schema / Responses semanticsはcontrol-planeに留め、computeへ入れない。
tool requestの制約はbackendがStructural Tagへ変換し、structured textとtoolsの同時指定は
1つのStructural Tagへ合成する。computeへは常にgrammarかstructural_tagの片方だけを渡す。

XGrammar v0.2.5は `vendor/xgrammar/` に固定vendorし、native C++ static libraryとしてbuildする。
TokenizerInfo sidecar は Python xgrammar 0.2.5.post1 が生成する。
詳細は [structured_generation.md](structured_generation.md) を参照。serverの対応surfaceは
[../user/server.md](../user/server.md) を参照。
