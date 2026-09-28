> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# Async MCU Submission / Generation Completion Gate（進行中）

> Status: R&D record。仕様 §1-§14 に対する検証記録。
> baseline は D10/D11.1-2 までの実装を含む時点。

## 1. 対象と前提

対象は rows=1 decode の production 経路（`Executor::submit_co_batch` →
`enqueue_batch` → `launch_host_backend_range` / `execute_mcu_hybrid` →
`complete_batch`）と、その下の `McuDecodeRuntime` / `stream_bridge`。

すでに成立していること（D10 の成果）:

- GPU-visible signal → consumer stream wait → next kernel は **prefix →
  `hipStreamWriteValue32(start, epoch)` → `hipStreamWaitValue32(done, epoch)` →
  suffix** として実装済み。Host は待たない。
- plan は毎 token CPU compile し、署名キャッシュで upload / FSM restart を抑止する。

足りていないこと:

- `submit_batch` は既に「publish までで return」しているが、**nonblocking な完了観測
  手段が無い**ため、呼び出し側は `complete_batch`（= `hipStreamSynchronize`）で
  blocking に待つしかない。
- 完了は generation 付きで GPU から publish されていない（`done_signal` は MCU
  内部の epoch であり、Host batch を含む batch 完了の generation ではない）。

## 2. Sync inventory（§1）

rows=1 decode の hot path に入る同期点を分類する。
A=HOST_REQUIRED / B=MOVE_TO_MCU / C=GPU_DEPENDENCY / D=DEBUG_OR_SETUP。

| location | current sync | reason | class | action |
|---|---|---|---|---|
| `submit_co_batch` tokens/requests | `hipMemcpyAsync` H2D | 入力 upload（stream 順序） | D | 維持。pageable 入力だと host が staging 分だけ止まるので計測対象 |
| `submit_co_batch` batch_context | `hipMemcpyWithStream` H2D | device descriptor upload | D | 同上 |
| `submit_co_batch` status staging | `hipMemcpyAsync` D2H（pinned） | 最終 status | A | 維持。完了と同じ stream 順序で覆う |
| `submit_co_batch` error path | `hipStreamSynchronize(stream)` | `hipEventRecord` 失敗時の cleanup | D | 維持 |
| `complete_batch` | **`hipStreamSynchronize(stream)`** | **decode 進行の待ち** | **C** | **`poll` に置換**。CPU が最終結果を使う瞬間だけ A として残す |
| `PendingBatch::destroy` | `hipEventSynchronize(done)` | 未消費 pending の破棄 | D | 維持（未完了 batch を破棄すると待ちに入る点は記録） |
| `keepalive_stop` | `hipStreamSynchronize(ka)` | CP keepalive 停止 | D | 維持（decode hot path 外） |
| `McuDecodeRuntime::prepare_plan` | `hipMemcpy` H2D ×17 | plan / invocation upload | C | **signature 変化時のみ**。stream 化は次段 |
| `McuDecodeRuntime::clear_plan` | `FsM::wait_stopped` | FSM 停止待ち（host poll） | D | 維持（plan switch / shutdown 時のみ） |
| `McuFsm::shutdown` | `hipStreamSynchronize(control_stream_)` | FSM 停止 | D | 維持 |
| `GpuMcuFsm::wait_plans` | host poll | test 用 | D | 維持（production 未使用） |
| `create_model_executor` | `hipMemcpy` H2D / `hipStreamSynchronize(upload_stream)` | setup | D | 維持 |

`stream_bridge` の `hipStreamWriteValue32` / `hipStreamWaitValue32` は host 同期では
なく stream に dependency を登録するだけなので、この表の対象外（＝目標の機構）。

### 2.1 hot path の同期数

| 種別 | 数 |
|---|---|
| 毎 batch 無条件に入るもの | **1**（`complete_batch` の `hipStreamSynchronize`） |
| signature 変化時のみ | 1（`prepare_plan` の同期 H2D upload） |
| 入力 upload（host stall の可能性あり、要計測） | 2 |

## 3. 完了 publish の設計（§3-§5）

`GpuMcuDeviceCompletion` は `hipMalloc` の device-only メモリなので host から
poll できない（D2H コピーが要る = 同期が戻ってくる）。§2 のリストのうち
host-readable な `GpuMcuSignalMemory` を使い、**stream が generation を書く**
形にする。

```
submit:  ... prefix ... [mcu run/wait] ... suffix ...
         hipMemcpyAsync(status_staging, D2H, stream)
         hipStreamWriteValue32(completion_signal, generation)   <- GPU が publish
         hipEventRecord(done, stream)
return:  ticket{generation}                                   <- host は待たない

poll:    published = completion_signal を host から 1 回読む
         published != generation -> Pending
         published == generation -> status_staging / mcu result_code を読んで Complete/Error
```

- generation は `Executor::next_submission_id`（単調増加、0 を飛ばす）を使う。
- `poll` は sleep も spin もしない。1 回 observe して即 return する。
- stale completion は generation の一致で排除する。
- `Executor` は単一 outstanding（`active_submission_id`）なので、generation の
  照合は一意に決まる。**32bit に truncate して signal に載せるため wrap の
  安全性は保証しない（§13 の STOP 条件として報告）**。
