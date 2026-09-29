> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/parallel_runtime.md`、現在の性能は `docs/perf/` を参照）

# RCCL parallel runtime Gate

PhaseNonShift に RCCL を用いた multi-GPU runtime を追加する Gate の記録。
PP=2 / TP=1、PP=1 / TP=2、PP=2 / TP=2 を同一 runtime から選択可能にし、
performance 改善より先に correctness と deadlock-free を確定させることが目的。

対象は Prefill / Decode / GDN layer / Standard Attention layer / MLP /
Paged KV / GDN recurrent state / quantized LINEAR。`DecodeBackend::Host` を固定し、
GPU-MCU との統合は次 Gate に送る。

## R0: 環境 probe

コードを書く前に実環境を確認した。

| 項目 | 値 |
|---|---|
| GPU0-3 | AMD Radeon AI PRO R9700 / gfx1201 / 34208743424 B / multiProcessorCount 32 |
| GPU4 | AMD Radeon Graphics / gfx1036 / 33190195200 B（iGPU、対象外） |
| `hipDeviceCanAccessPeer` | 5×5 すべて 1（全 mesh） |
| RCCL | `ncclGetVersion() = 23004`（NCCL 2.30.4 相当）、header/ library 一致 |
| nccl.h | `<ROCm>/include/nccl.h` |
| librccl.so | `<ROCm>/lib/librccl.so`（gfx1201 を含む） |

ROCm は `/opt/rocm` にではなく、Python パッケージ内の rocm-sdk root
（`rocm-sdk path --root`）に存在する。RCCL の CMake config package はなく、
header/library の明示探索で解決した。

PCIe topology（`lspci -tv`）:

```text
GPU0 0000:07:00.0
GPU1 0000:0a:00.0   → GPU0/GPU1 は [04-0a] 配下の同一スイッチ
GPU2 0000:0f:00.0
GPU3 0000:12:00.0   → GPU2/GPU3 は [0c-12] 配下の同一スイッチ
```

物理構成が TP clique の期待（GPU0-GPU1、GPU2-GPU3）と一致する。
PP lane はスイッチを跨ぐ（GPU0-GPU2、GPU1-GPU3）。

## R1: standalone RCCL transport

`test_rccl_transport`（2 GPU）で次を確認し、PASS まで Qwen へ入らないと決めた。

- AllReduce: GPU0=1 / GPU1=2 → 両方 3。
  4 KiB / 64 KiB / 1 MiB / 16 MiB / 64 MiB × FP32 / BF16。
- Send/Recv: 双方向。FP32 / BF16 / FP16 / INT32。
- 1000 iteration の communicator 再利用。hang 0 / data error 0 / RCCL error 0。
- 現在の HIP device と rank の device が一致しない呼び出しを拒否すること。

### 発見: RCCL が使った stream の破棄は crash を起こす

最初の test は `hipStreamCreate` / `hipStreamDestroy` を iteration ごとに
繰り返す実装だったが、3回目で `ncclLaunchPrepare` → `hipEventRecord` の中で
`pthread_mutex_lock(mutex=0x20)` に到達して segfault した。

切り分け:

| 構成 | 結果 |
|---|---|
| 永続 stream 1本（100 iteration / 最大64 MiB） | OK |
| iteration 毎に create して **destroy しない** | OK |
| 無関係な scratch stream を create/destroy | OK |
| 使用済み stream を destroy して次 iteration で新規作成 | **crash**（3回目） |
| 使用 stream を pool 化（3本 / 8本を回して destroy しない） | OK |

サイズ依存ではなく、**RCCL に渡した stream を communicator 生存中に
破棄すること**が原因と判断した。この Gate では
`rank ごとに永続 stream を1本、compute と RCCL を同じ stream で使う`
という方針（Gate 仕様 §49）と一致する形で test を書き換え、
`test_rccl_transport` は連続5回 PASS した。

現在の contract は `docs/developer/parallel_runtime.md` の stream contract 参照。

## R2: ParallelConfig / communicator topology

- `ParallelConfig::validate()` が world_size 1/2/4 以外、device 重複、
  負の device、件数不一致を拒否することを test で固定した。
- `test_rccl_topology`（4 GPU）で PP2TP2 の clique を実際に初期化する:

  ```text
  rank=0 device=0 pp_rank=0 tp_rank=0
  rank=1 device=1 pp_rank=0 tp_rank=1
  rank=2 device=2 pp_rank=1 tp_rank=0
  rank=3 device=3 pp_rank=1 tp_rank=1
  tp_group[0] devices=0 1
  tp_group[1] devices=2 3
  pp_lane[0] devices=0 2
  pp_lane[1] devices=1 3
  ```

  TP 全体 AllReduce（{0,1} / {2,3} の2 clique）と PP lane の
  send/recv（0↔2 / 1↔3）を同時並行で走らせ、値を検証した。

## PP1: stage 分割と communication node

### 実装した接点

lowering / graph / program / executor の既存経路をそのまま使い、
partition と communication node を足した。

- `ModelPartition{layer_begin, layer_end, owns_embedding, owns_lm_head}`
- `PrimitiveKind::COMM_SEND` / `COMM_RECV`、`KernelId::COMM_SEND` / `COMM_RECV`
- `Program::comms` + `Program::dispatch_comm`
- `HostExecutionContext::comm_launch`（function pointer）+
  `RcclCommEndpoint` + `launch_rccl_communication`

RCCL を `phaseshift_qwen35_runtime` に依存させないため、
transport 側は function pointer で渡す形にした。

### synthetic model

実 Qwen の外部 model に依存せず E2E を回すため、
`tests/support/synthetic_qwen35_model.h` が `config.json` と
`model.safetensors` を書き出すようにした。
`Qwen35Model` は private constructor なので
`load_from_safetensors` を通常どおり通す。
合成 weight は deterministic な LCG 由来の小さい BF16 値。

`full_attention_interval` は GDN と full attention の出現比を決める
（0 は 3:1、N は N-1:1）。`layer_types` と
`weights.layers[i].is_gdn` の一致は `validate_qwen35_weights` が検査するため、
config と weight 生成は同じ判定式を共有する。

### PP2 比較で最初に失敗した理由

`test_qwen35_pp2_e2e` は最初、prefill は一致するが decode が
一致しないとして失敗した。切り分けの結果、原因は runtime ではなく
test harness にあった。

- `stage0` は head を持たないため `sample = false` にした。
  runner はこのとき `*sampled` を書き込まないのに、
  decode の入力 token にその未更新の変数（0）を使っていた。
- 正しい PP の振る舞いは、**stage1 の sampled token を coordinator が受け取り、
  次の step で stage0 に渡す**こと。

そのため runner に step 単位の barrier と
`shared_tokens`（stage1 が書き、stage0 が読む）を導入した。
これは後の ParallelRuntime の coordinator contract にもそのまま効く。

途中で使った診断（stage0 の境界 hidden と baseline の hidden tap を比較する、
stage1 の受信値を value trace で採取する）のうち、後者は workspace 再利用で
採取値が汚れるため使えなかった。境界比較は
`target_hidden_taps` で program の external output として確定値を
取り出す方法だけが有効だった。

### 途中で除外した仮説

- GDN state index のずれ: `full_attention_interval = 1`（GDN なし）でも
  同じ症状が出たので却下。
- page境界: `prefill_tokens = 4`（decode 中は page0 内）でも症状が出るので却下。
- optimized dispatch: `PHASESHIFT_QWEN35_KERNEL_MODE=correctness` でも
  同じ症状なので却下。
- state の増加位置: position / block 数は baseline と全 step で一致。

つまり上記はいずれも原因ではなく、最終的には harness の token 供給に
帰着した。

## Gate TP1: synthetic Column / Row Parallel LINEAR

`MatrixShardSpec{axis, rank, world_size}` を導入し、BF16 weight を
output feature 方向（Column）と input feature 方向（Row）に分割する。

- Column: 元 tensor の row view をそのまま使う。追加コピーはない。
- Row: **view では切らない。** `launch_gemm_bf16_baseline` は weight / input を
  連続 row-major として読むため、stride を持つ view を渡すと行がずれる。
  連続な local matrix へ `hipMemcpy2DAsync` でコピーする。
  packed quantized weight で同じことが起きる（Gate §34）のと同一の問題。

### Column Parallel（`test_qwen35_tp_column_linear`）

2 GPU で各 rank が `out/2` 行の GEMM を走らせ、concat して1GPU の full GEMM と比較。

```text
column parallel: vs 1GPU gpu-gemm mismatches=0/448 max_abs=0 max_rel=0
1GPU gpu-gemm vs CPU reference: mismatches=0/448 max_abs=0 max_rel=0
```

**bit exact**。collective がないため一致は保証される。

### Row Parallel（`test_qwen35_tp_row_linear`）

各 rank が `X[:, rank*K/2]` と `W[:, rank*K/2]` で partial GEMM を作り、
RCCL `all_reduce_sum`（BF16）で合算して1GPU と比較。

```text
row parallel: vs 1GPU gpu-gemm mismatches=165/448
  max_abs=0.0078125  scale=1.89062  max_abs/scale=0.00413223
  max_rel=0.0971429  max_rel_active=0.0548523
1GPU gpu-gemm vs CPU reference: mismatches=0/448 max_abs=0 max_rel=0
```

- 1GPU の GEMM は CPU reference（FP32 accumulate → BF16 cast）と bit exact。
- TP の側は partial を BF16 に丸めてから AllReduce するため、
  **差は最大でも出力 scale の 0.41%**（BF16 の1 ULP は相対 2^-8 = 0.39%）。
  これは Gate §35 が予告した「partial を BF16 へ丸めてから AllReduce すると
  bit exact にならない」ことの実測値である。
- `max_rel` は0近傍の要素で 9.7%、`max_rel_active`（scale の1%以上の要素）でも
  5.5% になる。これは cancellation が起きる要素で、
  **絶対誤差は scale 依存ではなく一定量**残るためで、相対指標は使えない。
  判定は `max_abs / scale <= 1%` で行う。

### 残課題

Row Parallel を bit exact にするには Gate §35 の推奨どおり
`FP32 partial output → ncclFloat32 SUM → BF16 cast` が必要。
現状の `launch_gemm_bf16_baseline` は BF16 出力固定のため、
FP32 partial を出す launcher が要る。次 Gate で判断する。



- PP2 の E2E 一致確認（harness 修正後の再実行）
- Gate PP2（stage-local weight / state の割当）
- Gate TP1-TP4（Column/Row Parallel LINEAR、Attention、GDN、全 layer）
- Gate C1 / C2（PP2 × TP2）
- Gate P0（1GPU / PP2 / TP2 / PP2TP2 の比較計測）

## Gate TP2: standard attention の TP 化

### KV head offset の導入

`AttentionShapeKey` に `kv_head_offset` を加え、
`DispatchBinding` に `kv_heads`（local）と `kv_head_offset` を載せた。
KV pool はまだ full の head 数のままなので（Gate §28 の方針）、
rank は `offset` から始まる head 範囲だけを読み書きする。

- `exec_kv_append` / `exec_paged_attention` の入力側は local head 番号
- pool 側の index（`blocks_per_token`、`elems_per_token` 等）は full のまま、
  head の開始位置だけ offset を足す
- GQA の group は `num_q / local_kv_heads` で計算する。pool の head 数で
  計算すると rank ごとに group が変わり head 対応がずれる

`bf16` の KV append は従来 head 単位ではなく `elems_per_token` 全体を
memcpy していた。local head になったため、head 単位のコピーへ変更した。

### 抓到了 bug: offset = 0 の rank で local が full に戻る

最初の実装は `local_kv_heads = kv.kv_heads - b.kv_head_offset` と導出していた。
`kv_head_offset == 0` の rank（tp_rank = 0）ではこれが **pool の full head 数**に
なってしまい、入力の `features` と不一致になって `INVALID_BINDING` になる。
その rank が AllReduce に到達しないため他 rank が待ち、E2E は hang した。
local head 数を導出ではなく明示的に渡す形へ変更して解消した。

### 結果

`full_attention_interval = 2` の synthetic model（GDN / attention / GDN / attention）で、
attention のみ sharded、GDN と MLP は replicated にした。

```text
baseline tokens: 44 44 44 44 44
tp2 rank0 tokens: 44 44 44 44 44
tp2 rank1 tokens: 44 44 44 44 44
tp2 vs 1GPU logits: mismatches=248/320
  max_abs=0.0078125  scale=1.78906  max_abs/scale=0.00436681
TP2_ATTENTION: PASS
```

- 両 TP rank の sampled token と logits が完全一致
- 1GPU との logits 差は **出力 scale の 0.44%**。Gate TP1 の Row Parallel と同じ
  原因（partial を BF16 に丸めてから AllReduce）で、BF16 の1 ULP と同オーダー
- RCCL error 0、deadlock 0

### optimized path は未対応

`kv_append_dispatch.hip` / `paged_attention_dispatch.hip` はまだ
`kv_head_offset` を受け取らないため、`features != kv_heads*head_dim` で
reject する。この Gate は `PHASESHIFT_QWEN35_KERNEL_MODE=correctness` で
固定して検証した。optimized への対応は Gate TP4 で行う。

## 現在への影響

- parallel runtime の contract は `docs/developer/parallel_runtime.md`。
- RCCL に渡す stream の寿命制約は runtime 全体の制約として残る。
- coordinator は step 単位で全 rank を揃え、生成 token を rank 間で
  共有する。この形が ParallelRuntime の出発点になる。

## 未解決

- Gate PP2（stage-local weight / state の割当）
- Gate TP3（GDN one layer の TP 化。GDN の conv / recurrence state にも
  head offset が必要になる）
- Gate TP4（全 layer の TP 化と collective 数の照合、optimized path の対応）
- Gate C1 / C2（PP2 × TP2）
- Gate P0（1GPU / PP2 / TP2 / PP2TP2 の比較計測）
