# Parallel runtime

RCCL を用いた multi-GPU 実行の現在の contract。

対象は同一 process 内で複数 GPU を駆動する single-process / multi-GPU 構成のみ。
利用できる構成は `world_size = pp_size * tp_size` が 1 / 2 / 4 の場合に限る。

## ParallelConfig

`include/phaseshift/runtime/parallel/parallel_config.h`

```cpp
struct ParallelConfig {
    uint32_t pp_size = 1;
    uint32_t tp_size = 1;
    std::vector<int> devices;
};
```

rank mapping は固定:

```text
global_rank = pp_rank * tp_size + tp_rank
pp_rank     = global_rank / tp_size
tp_rank     = global_rank % tp_size
```

`validate()` は次を要求する。いずれも満たさない設定は疎通前に拒否される。

- `pp_size >= 1` / `tp_size >= 1`、いずれも `kParallelMaxWorldSize` 以下
- `world_size` は 1 / 2 / 4 のいずれか
- `devices.size() == pp_size * tp_size`
- device index が負でなく、重複がない

## Communicator topology

communicator は clique 単位で独立に作る。world communicator を作って全操作を
全 rank collective に丸めることはない。

```text
tensor_group_devices(pp_rank)    = { devices[pp_rank*tp_size + t] | t < tp_size }
pipeline_group_devices(tp_rank)  = { devices[p*tp_size + tp_rank] | p < pp_size }
```

PP=2 / TP=2（`devices = {0,1,2,3}`）の例:

| group | 成員 |
|---|---|
| TP stage0 | GPU0, GPU1 |
| TP stage1 | GPU2, GPU3 |
| PP lane0 | GPU0, GPU2 |
| PP lane1 | GPU1, GPU3 |

`CommGroup::Tensor` の操作は同じ `pp_rank` の rank 間でのみ許可され、
`CommGroup::Pipeline` の操作は同じ `tp_rank` の rank 間でのみ許可される。
member でない peer への send/recv は `invalid_argument` で拒否される。

## RcclTransport

`include/phaseshift/runtime/parallel/rccl_transport.h`

- `RcclTransport::create(config)` が clique を初期化する。clique 内部の
  communicator 作成はこの class の内部に閉じる（外部から handle を触らない）。
- API は `all_reduce_sum` / `send` / `recv` / `broadcast` の4種。
  rank は **global rank** で渡し、group 内の local rank は transport が変換する。
- PhaseShift dtype から RCCL datatype への変換は
  `rccl_dtype_from_value()`（ValueDType → RcclDataType）と
  transport 内部の一箇所（RcclDataType → `ncclDataType_t`）に集約する。
- 全 API は `Status` を返し、`ncclResult_t` を必ず検査する。失敗メッセージには
  `operation` / `seq` / `device` / `pp_rank` / `tp_rank` / `rank` / `peer` /
  `ncclGetErrorString()` が入る。
- `PHASESHIFT_RCCL_DEBUG=1` で `rank=… seq=… op=… begin|enqueued` を stderr に
  出力する。デフォルトは off。
- `shutdown()` が明示的な終了経路。destructor は fallback で、失敗しても残りの
  communicator を破棄し続ける。

`launch_rccl_communication()`（`parallel/comm_bridge.h`）が
`CommDescriptor` を transport の API へ変換する bridge になる。

## stream contract

`hipStream_t` を RCCL に渡したら、その stream は
**当該 `RcclTransport` が shutdown されるまで破棄してはならない。**

この制約を破ると、次の collective の enqueue 時に
`ncclLaunchPrepare` → `hipEventRecord` で segfault する。
RCCL が使ったことのない stream の create/destroy は影響しない。
したがって rank ごとに永続 stream を1本作り、compute と RCCL を同じ stream で
実行する。stream の分離と event 依存の追加は性能 Gate で行う。

## Program communication node

Program は phase 固有の別 graph を持たない。通信も既存の
`PrimitiveGraph` → `Program` → `DispatchBinding` の経路に乗る。

- `PrimitiveKind::COMM_SEND` / `COMM_RECV`（`CommSendNode` / `CommRecvNode`）
- `KernelId::COMM_SEND` / `COMM_RECV`
- `Program::comms`（`CommDescriptor` の表）と
  `Program::dispatch_comm`（dispatch と1対1、該当なしは `kNoComm`）

```cpp
struct CommDescriptor {
    CommGroup group;
    CommOperation operation;
    uint32_t peer;      // global rank
    ValueDType dtype;
    ValueId input;
    ValueId output;
};
```

transfer count は descriptor に保存しない。実行時に
`rows * row_stride`（`row_stride` は element 単位）として解決する。
送信側と受信側は同じ batch metadata と同じ `ProgramBuildOptions` を使うため、
両者が同じ count を導く。

`program_executor` は `COMM_*` を physical kernel として launch せず、
`HostExecutionContext::comm_launch` 経由で RCCL API を enqueue する。
`comm_launch` が未接続のまま `COMM_*` を実行した場合は
`invalid_state` で失敗する（fail-closed）。
graph capture 中に `COMM_*` に到達した場合も `invalid_state` で失敗する。

`Executor::comm_self` / `Executor::comm_launch` は parallel runtime の側で
接続する。`create_model_executor` は通信を設定しない。

## ModelPartition

`include/phaseshift/models/qwen35/model/lower_to_primitives.h`

```cpp
struct ModelPartition {
    uint32_t layer_begin;
    uint32_t layer_end;      // 0 は最終 layer まで
    bool owns_embedding;
    bool owns_lm_head;
};
```

lowering は partition を検証してから graph を組む。

- `owns_embedding == (layer_begin == 0)` でなければならない
- `owns_lm_head == (layer_end == num_hidden_layers)` でなければならない
- hidden tap は partition の外には置けない

先頭 stage でないものは graph 先頭に `COMM_RECV` を置き、
末尾 stage でないものは graph 末尾に `COMM_SEND` を置く。
`peer` は `Qwen35LowerOptions::pipeline_peer_rank` で与える。

## stage ごとの external slot

`build_host_execution_context` の external output は **index 固定の対応** で
割り当てる。

| index | buffer |
|---|---|
| 0 | `output_hidden` |
| 1 | `logits` |
| 2 | `sampled_tokens` |
| 3 | `token_hidden` |
| 4+ | `dflash_target_hidden[tap]` |

したがって head を持たない stage は external output を持たない
（`external_output_count == 0`）こと。head を持つ stage の対応は通常の
program と同じ4つになる。`external_input_count` は embedding を持たない
stage で 0 になる。

## state index の割り当て

`next_gdn_state_index` / `next_attention_state_index` は、partition 内で
lowering した layer を走査する連番である。したがって PP で分割した stage は
全体の layer index ではなく stage 内の連番を state index として使う。
stage はそれぞれ独立した `GdnStatePool` / `PagedKVPool` を持つため、
stage 内で書込みと読出しの両方が同じ index を指す限り整合する。

`GdnStatePool::create` は `GdnStatePoolLayout::from_text_config` で作った
layout を使わなければならず、`PagedKVPool::create` の
`num_attention_layers` は `layer_types` 中の full attention 数と一致させる。

## 各 rank の所有物

各 rank は独立して次を所有する。

- `GpuArena`
- `SequenceSlotPool` / `GdnStatePool` / `PagedKVPool`
- `Executor`（`ProgramSet` を含む）
- 実行用の `hipStream_t`
- `RcclTransport` 内の communicator handle（transport は rank 間で共有）

同じ `Executor` を複数の rank / thread で共有しない。
`DeviceBatchContext` の row position / sequence slot / block table / visibility
などの logical batch metadata は全 rank で同一にし、
local layer range / local state index のみ rank 固有にする。

## Tests

| test | label | 内容 |
|---|---|---|
| `test_parallel_topology` | `cpu;required` | rank mapping / communicator topology |
| `test_qwen35_pp_boundary` | `gpu1;required` | stage 分割と communication node の contract |
| `test_qwen35_synthetic_e2e` | `gpu1;required` | synthetic model の prefill / decode 再現性 |
| `test_rccl_transport` | `gpu2;rccl;rccl_2gpu` | AllReduce / Send / Recv / 連続利用 |
| `test_rccl_topology` | `gpu4;rccl;rccl_4gpu` | PP2TP2 の communicator 構成 |
| `test_qwen35_pp2_e2e` | `gpu2;rccl;rccl_2gpu` | PP2 と 1GPU の一致比較 |

synthetic model は `tests/support/synthetic_qwen35_model.h` が
`config.json` と `model.safetensors` を書き出し、通常の
`Qwen35Model::load_from_safetensors` で読み込む。
実行ハーネスは `tests/support/qwen35_synthetic_runner.h` にある。
