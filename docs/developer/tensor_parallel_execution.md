# Tensor Parallel Execution

Qwen3.8-27B Dense の Tensor Parallel (TP) execution の source of truth である。
weight storage 側の partition（`TensorPartitionDesc` / manifest / materialization）は
[tensor_partition.md](tensor_partition.md) が正本であり、本ドキュメントは
runtime execution semantics を扱う。

対象 Gate は TP-Exec-1。目的は correctness であり、性能最適化ではない。

## 1. 全体構造

```text
ContinuousBatcher (global, 1個)
      |
  ScheduledBatch  ── 同じ batch を全 rank へ
      |
  TpCoordinator
   ├── worker thread rank0 ── TpRankRuntime rank0 (HIP device 0)
   │                             arena / stream / rank-local weights /
   │                             rank-local KV pool / rank-local GDN state /
   │                             existing Executor (tp schedule + barrier hook)
   └── worker thread rank1 ── TpRankRuntime rank1 (HIP device 1)
                                 (同上)
```

- **1 rank = 1 device-local runtime**（`TpRankRuntime`）。既存の single-GPU
  `Executor` を rank-local execution engine としてそのまま再利用し、
  multi-device な巨大 Executor には作り替えていない。
- `TpCoordinator` の責務は only: rank runtime 生成、同一 batch の配布、
  program segment の同期（barrier）、collective 呼び出し、rank failure の集約、
  rank0 の結果（sampling で確定した token）の返却。quantization実装 /
  weight slicing / preshuffle / attention math は持たない。
- 各 worker thread は起動時に `ScopedDevice` で自分の rank device を select する。
  current device の暗黙依存を持たない。
- `TpRankRuntime` / `TpCoordinator` は
  `include/phaseshift/models/qwen35/runtime/tensor_parallel.h`（qwen35 runtime 層）。
  transport は model 非依存の runtime core（`phaseshift/runtime/tp/**`）、
  partition descriptor は weights 層、境界は維持している。

- `TpCoordinator::debug_report()` が rank ごとの quantized weight bytes /
  KV bytes / GDN state bytes を出力する（memory report 用。常時ログではない）。

TP=1 では `TpCoordinator` を使わない。`WeightLoadOptions.partition_plan == nullptr`
の既存 single-GPU path（現行 Executor / ContinuousBatcher）がそのまま走る。

## 2. weight materialization（storage 側の要点）

```text
global canonical → logical TensorPartition → rank-local canonical
    → rank-local preshuffle → GPU native weight
```

`preshuffle → partition` には絶対に戻らない。詳細と field 定義は
[tensor_partition.md](tensor_partition.md) を参照。execution 側から見ると:

- `load_qwen35_weights_tensor_parallel_rank(model_dir, tp_size, tp_rank, ...)`
  が rank-local weight をロードする（`build_qwen35_tensor_partition_plan` で
  plan を作り、既存 loader に渡す）。同一 global canonical artifact を
  `tp_size`/`tp_rank` だけ変えて各 rank が読む。
- `MatrixWeight.rows/cols/k_padded` は local geometry。
  `MatrixWeight::partition` が small metadata として partition を保持する。
- 現行 manifest は `format_version = 3` の global canonical のままで、
  TP degree を disk artifact へ焼き込まない。

## 3. rank-local Qwen geometry

`Qwen35TensorParallelContext`（`models/qwen35/model/tensor_parallel_context.h`）:

```text
tp_size / tp_rank
local_intermediate_size   = intermediate_size / tp_size
local_attention_heads     = num_attention_heads / tp_size
local_key_value_heads     = num_key_value_heads / tp_size
local_gdn_key_heads       = linear_num_key_heads / tp_size
local_gdn_value_heads     = linear_num_value_heads / tp_size
```

- `hidden_size` は replicated のまま（partition しない）。
- 今回は均等分割のみ。以下のいずれかが割り切れない構成は `unsupported`:
  `intermediate_size` / `num_attention_heads` / `num_key_value_heads` /
  `linear_num_key_heads` / `linear_num_value_heads`。
- lowering（`lower_qwen35_to_primitives` の `Qwen35LowerOptions::tp`）は
  local geometry で `AttentionShapeKey` / `GdnShapeKey` / SwiGLU shape /
  weight validation を組み立てる。TensorPartitionDesc には model geometry を
  持たせない。

## 4. 段階別の parallel 化

### MLP（column / row parallel）

```text
gate/up: column parallel（local_I = intermediate_size / tp_size、axis=0）
down:    row parallel（K = local_I、axis=1）
   ↓
mlp.down_proj 後に SUM → [tokens, hidden] を全 rank 同一に
```

### Full attention

```text
q_proj: column parallel（連続する local query head group、q/gate interleaved head pair を壊さない）
k/v:    column parallel（連続する local KV head group）
        KV cache は rank-local kv_heads のみ保持
o_proj: row parallel（K = local query features、axis=1）
   ↓
o_proj 後に SUM → [tokens, hidden] を全 rank 同一に
```

GQA は q head 数・KV head 数の双方を tp_size で割る（片方だけの一致では
不十分）。rank0 は q head `[0, Q/2)` + kv head `[0, KV/2)`、rank1 は残りを
連続して所有するため query/KV の対応関係は保たれる。

### GDN

```text
in_proj_qkv: segmented ranges [local Q | local K | local V]（axis=0、3 ranges）
in_proj_z / a / b: axis=0（value feature / value head 単位）
conv1d:           axis=0（Q/K/V 同じ segmented ranges、BF16 generic partition）
recurrence state: rank-local（local key heads / local value heads / local conv state）
                  GPU 間同期は一切しない
out_proj:         row parallel（axis=1）
   ↓
out_proj 後に SUM → [tokens, hidden] を全 rank 同一に
```

runtime は rank-local qkv を `[local Q | local K | local V]` として読む。
global offset を kernel へ持ち込まない。

### replicated

RMSNorm / residual add / embedding / lm_head / final norm は全 rank 同一計算。
norm parameter・embedding・lm_head は replicated（partition plan に entry なし）。
MTP は partition せず（plan に entry なし。分散実行は別 Gate）。

## 5. execution schedule と collective boundary

lowering は row-parallel 出力 node（`L{n}.o_proj` / `L{n}.gdn_out` /
`L{n}.mlp_down`）に `PrimitiveGraphNode::tp_combine` annotation を付ける。
`create_model_executor` が `build_tp_execution_schedule(graph, program)` で
`Program.dispatch_source_node` から segment を機械的に構築する
（dispatch index の手書き hard-code はない）。

```text
TpProgramSegment { dispatch_begin, dispatch_end, barrier_after, combine_value }
TpBarrierKind { None, SumHidden }
```

- barrier は layer ごとに最大2回（row-parallel projection 後 + down 後）。
- segment の境界で `execute_program_range(begin, end)` を呼ぶ
  （既存 API を再利用。PrimitiveGraph に AllReduce/RCCL primitive は足していない）。
- combine 対象は segment 末尾 dispatch の出力 value（`resolve_host_value` で解決）。
- Primitive IR 全体は変更していない。長期的には debug_name 文字列ではなく
  `tp_combine` annotation で判別する（今回の仕組みがそれ）。

`tp_size == 1` では schedule を作らず、collective を一切呼ばない。

## 6. TpTransport

```cpp
class TpTransport {
    virtual const char* name() const noexcept = 0;
    virtual Status verify();                      // capability self-test
    virtual Status sum_hidden(const TpSumInvocation&);  // 全 rank の partial を combine
};
```

- model knowledge なし（buffer 群 + streams + ready events だけを扱う）。
- reduction は全 rank の partial を **f32 で累積し、最後に 1 回だけ bf16 へ丸める**。
  rank 数によらず丸めは 1 回で、peer 経路と host 経路が同じ contract を共有する
  （`tp_reduce_f32_accumulate` / `test_tp_reduction`）。
- 実装:
  - `HipPeerTpTransport`（**peer write ベースの対称 push reduce**）
    - rank r は自分の partial を、自分以外の各 rank の inbox へ kernel からの
      **peer write** で送る（送り先 inbox は受信側 rank の device 上にある）。
    - 各 rank は受け取った inbox と自分の partial を **local で** f32 加算し、
      1 回だけ bf16 へ丸めて自分の buffer を更新する。他 rank のメモリは read しない。
    - leader を置かず全 rank が同じ 2 kernel（send / add）を並列に実行する。
      各 rank は自分の compute stream 上で send を enqueue し、送信完了 event を
      record する。受信側は他 rank の送信完了 event だけを `hipStreamWaitEvent` で
      待ってから add する。host / device sync は使わない。
    - inbox は受信側 device に確保し、barrier ごとに parity を交互にして
      send / add のオーバーラップと再利用 hazard の回避を両立する。
    - `ready` event には依存しない（stream 順序で segment 完了が保証される）。
  - `HostMediatedTpTransport`（reference / fallback）— pinned staging へ D2H し
    host 上で f32 加算して H2D する。
- **capability self-test**: `verify()` は runtime buffer と同じ条件
  （8 KiB / 128 KiB / 1 MiB を各 4 round）で peer write send + local add を
  実際に実行し、1 回でも不一致なら `unsupported` を返す。
  `TpCoordinator::initialize` は rank runtime 作成後に `verify()` を呼び、
  失敗時は stderr に明示してから `HostMediatedTpTransport` へ切り替える
  （silent fallback ではない）。
- `PHASESHIFT_TP_TRANSPORT=auto|p2p|host` で backend を強制できる（既定 auto）。
- RCCL は必須 dependency にしていない。将来 `RcclTpTransport` を追加できる。

### R9700 における peer read の制約

本ホスト（R9700 x2、PCIe switch 経由、large BAR 32G、`pcie_p2p=Y`）では
`hipDeviceCanAccessPeer` は両方向 1 を返すが、**kernel からの peer load は
正しいデータを返さない**（ゼロまたは別データ。サイズ・実行ごとに変動）。
`hipMemcpyPeerAsync` も不安定で、`HSA_FORCE_FINE_GRAIN_PCIE=1` は改善しない。
peer READ は source を H2D で初期化しても `__threadfence_system` を付けても
D2D copy を経由しても壊れるため、L2 writeback ではなく PCIe remote-read path
自体の問題である。SDMA も PUSH（source device stream = local read + posted peer
write）は安定し、PULL（destination device stream = remote read）は壊れる。
一方 **kernel からの peer store は 4 KiB〜1 MiB / 両方向 / 1000 回で再現性を
もって正常**である。本 transport は peer store のみで構成している。
切り分けの記録は [docs/rnd/tp_exec2_p2p.md](../rnd/tp_exec2_p2p.md)。

### transport microbenchmark

```sh
phaseshift-bench tp-reduce [--rows 1,2,...] [--features 5120] [--iters N]
                           [--backend auto|p2p|host] [--check]
```

帯域定義は `2 * payload_bytes / elapsed`。`--check` は backend ごとの目標
（hip-p2p: 大 payload で >= 20.0 GB/s かつ rows=1 で <= 150 us、
host-mediated: 大 payload で >= 1.2 GB/s かつ rows=1 で <= 400 us）を判定する。
実測値と再現条件は R&D record
[docs/rnd/tp_exec2_p2p.md](../rnd/tp_exec2_p2p.md) を参照する。

対称 push 型は leader 直列型より小 payload の latency と大 payload の帯域の
両方で優る。小 payload（rows=1）でも host 経由を上回るため、decode の
per-step barrier でも peer 経路をそのまま使える。

## 7. scheduler / sampling

- `ContinuousBatcher` は **global に1個だけ**。rank ごとに作りつけない。
- `ScheduledBatch` をそのまま全 rank へ渡す（sequence count / token rows /
  slot / ordering / token position / decode-prefill 分類が全 rank 一致）。
  rank 独自 schedule は禁止。
  - rank0 の `PagedSequenceState` が正本。admit/finish のタイミングで
    `TpBatchHook` が rank1 の mirror sequence を同じ slot/generation で
    作成/解放する（slot 割当が rank 間で乖離しないことを検証する）。
  - rank ごとの batch は `ScheduledRequest.sequence` を mirror に向けて
    置き換えたコピーだけ。他は同一。
- **sampling owner は rank0**: batcher は hook から返る rank0 の
  `sampled_tokens` のみを commit する。rank1 の sampling 結果は使われない。
  next token は batcher（global request state）を介して次の step の batch に
  入るため、rank 間で token を持ち越さない。constraint / grammar state も
  batcher 側で1か所だけ進む。
- lm_head は replicated（両 rank 同じ計算。結果は identical、消費は rank0 のみ）。
- prefix cache は TP では使わない（rank0 だけが KV を restore すると
  妥当性が壊れるため。distributed prefix cache は非目標）。

## 8. 失敗伝播と shutdown

- 1 rank でも失敗したらその worker は `TpBarrierGroup::abort()` で他 rank の
  wait を解放し、`TpCoordinator` は step 全体を failure として返す。
  deadlock しない。
- `TpRankRuntime` の destructor は executor shutdown → stream / arena 解放を
  順に行う RAII。初期化途中の失敗も同じ経路で解放される。
  peer access / event / mirror sequence も coordinator 側で明示的に解放する。

## 9. numerical 互換性（TP=1 vs TP=2）

partition された GEMM の出力（bf16）と barrier の SUM（bf16）で丸めが入るため、
TP=1 と TP=2 は bit-identical ではない。Qwen3.8-27B（PSQ4/PSQ8、64層、
prefill5 tokens / greedy16 tokens）で実測した layer ごとの相対 drift:

```text
layer  0: rel=0.6%
layer  3: rel=2.1%
layer  7: rel=2.2%
layer 15: rel=1.4%
layer 31: rel=1.3%
layer 62: rel=2.7%
final hidden: rel=4.4%   prefill logits max abs diff=0.70 (argmax 一致)
```

相対 drift は全層で 0.6〜2.7% に収まり、特定層での跳躍はない（＝partition /
collective の実装 bug ではなく bf16 丸めの累積）。
rank0 と rank1 の出力（final hidden / logits）は SUM 以降 replicated なため
**bit-identical** である。

greedy token 列は、上位2値が近い token で丸め差により分岐し得る
（実測: margin の小さい natural prompt では7 token 目で分岐した例がある）。
`tests/unit/test_qwen35_tp_e2e.cpp` の acceptance prompt は margin が確保
できる prompt を選び、greedy16 token の一致を assert する。
prompt は `PHASESHIFT_TP_PROMPT` で差し替えられる。

## 10. 現在の対応範囲

- execution の正式サポートは **tp_size = 2**（`TpCoordinator` は3枚以上を
  `unsupported` で拒否する）。descriptor / plan builder は任意の tp_size を
  表現できるが、execution runtime まで tp_size=4 対応済みとは表現しない。
- TP=2 の local geometry は optimized selector の validated allowlist に含める。
  現在の対象は Linear（`linear_selector`）/ Paged Attention
  （`paged_attention_selector`）/ GDN Recurrence（`gdn_recurrence_selector`）。
  TP=2 で新たに必要になった shape は optimized kernel test で検証してから
  個別に追加し、未知 shape を自動許可しない。shape と family の対応は
  [../rnd/tp_exec2_shape_coverage.md](../rnd/tp_exec2_shape_coverage.md) を参照。
- device 不足は明示エラー（bench）または skip（test runner 77）。
- 異種 GPU は未サポート（同一 RDNA4 を前提。異なる場合は unsupported 方針）。

### 未対応 / 非目標

Multi-GPU Executor の一般化、RCCL backend（AllReduce / AllGather / AllToAll）、
通信と計算の本格的な overlap（各 barrier 内の send / add は並列だが、segment の
計算との overlap は未対応）、
graph capture との併用（`PHASESHIFT_HIP_GRAPH` は TP schedule と非併用）、
DFlash2 + TP、MTP 分散実行、vocab parallel、distributed sampling、prefix cache、
structured generation / server 統合、Expert Parallel / MoE、
PSQ3、pipeline parallel、TP=4 production support。

## 11. tests

| test | label | 内容 |
| --- | --- | --- |
| `test_qwen35_tp_context` | cpu;required | TP local geometry validation / schedule・collective boundary 生成 |
| `test_tp_reduction` | cpu;required | f32 accumulate reduction の single-round 丸め contract |
| `test_qwen35_tensor_parallel_plan` | cpu;required | Qwen35 plan builder（local geometry / head pair / GDN coverage） |
| `test_tensor_partition` / `test_canonical_partition` | cpu;required | weight storage 側（[tensor_partition.md](tensor_partition.md) 参照） |
| `test_weight_load` | gpu1;required | 既存 single-GPU 回帰 + partition plan なし path |
| `test_qwen35_tp_weight_load` | gpu1;required | rank-local weight load（PSQ4/PSQ8 local rows / segmented ranges / replicated） |
| `test_tp_transport` | gpu2;optional | `sum_hidden` の CPU reference oracle（実装 backend 全体） |
| `test_qwen35_tp_execution` | gpu2;optional | tiny model: Stage A/B + full forward + prefill/decode + greedy 一致 + rank0/rank1 identical + layer別 drift |
| `test_qwen35_tp_e2e` | gpu2;optional;external_files | Qwen3.8-27B 実モデル TP=1 vs TP=2 greedy（`PHASESHIFT_TP_MODEL_DIR`） |

GPU2 テストは `ctest -L gpu2` で実行する。GPU 不足時は runner が exit 77 で
skip し、required acceptance（`-L required`）には含まれない。
