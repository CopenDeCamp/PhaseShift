> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# Autonomous MCU Execution Contract Gate（STOP 報告）

> Status: R&D record。契約「GPU-MCU を Host 非参加の自律推論 runtime とする」に対する
> **ギャップ調査と STOP 報告**。実装は未着手。

## 1. 契約が要求する形

```
Host:  START / OBSERVE(nonblocking) / CONTROL / DESTROY
MCU:   while running { next step -> publish output -> check control ring }
```

- Host は各 decode step の進行に参加しない。
- `decode_blocking()` / `decode_async()` の二重 API を作らない。
- output は producer-consumer ring。`try_pop_output()` / `poll_output()` は
  sleep / spin / stream sync / wait_for を内部で行わない。
- completion は「Host を起こすもの」ではなく「GPU 側でどこまで publish 済みか」の
  observation primitive。
- 通常進行に Host→MCU の毎 token command を送らない。

## 2. 現在の実装とのギャップ

現在の production MCU 経路は **Host が token ごとに駆動する**形である。

```
Host:  prefix を host stream に launch
       hipStreamWriteValue32(start_signal, epoch)
       hipStreamWaitValue32(done_signal, epoch)
       suffix を host stream に launch
       complete_batch で hipStreamSynchronize
```

これは契約 §2 が明示的に禁止する

```
Host submit token N -> wait -> token N complete -> submit token N+1
```

そのものである。**契約と現在の実装は別の execution model**であり、パラメータ調整で
埋まる差ではない。

## 3. 調査結果（既存 primitive の再利用可能性）

| 要素 | 現状 | 契約への適合 |
|---|---|---|
| control ring | `control_ring.h`。256 slot、slot 64 byte、`sequence` による SPSC、`try_push` / `try_pop` は **nonblocking**、`__ATOMIC_RELEASE/ACQUIRE` + `__MEMORY_SCOPE_SYSTEM` | **そのまま使える**（§7 の channel） |
| plan の loop | `pc = node.next == kMcuNoNext ? pc + 1u : node.next`。`next` は後方も指せる | **plan レベルの loop は表現可能** |
| START trigger | `external_start_signal`（epoch）を FSM が poll し、epoch ごとに plan を 1 回実行 | START は使える。**epoch による毎 token kick は契約違反** |
| output | `output_base` = node index 単位の scratch（`kMcuMaxNodes` × 4 byte、device）。`completions_` は `hipMalloc` の **device-only** 完了配列 | **token 出力 ring は無い**。device-only なので Host から poll できず、D2H コピーが要る（= 同期が戻る） |
| completion の意味 | `done_signal` = Host を起こすための epoch | 意味の変更が必要（§6） |
| output ring 満杯時 | 該当する契約が無い | **未定義。§5 は「勝手に決めず調査して報告」と指示** → 未定義であることを報告する |

## 4. Sync audit（§14）

現在の rows=1 decode hot path:

| location | sync | 判定 |
|---|---|---|
| `complete_batch` | `hipStreamSynchronize(stream)` | **FAIL**（decode 進行の待ち） |
| `submit_co_batch` | 毎 token の `hipStreamWriteValue32` / `WaitValue32` | Host は待たないが、**Host が token ごとに起動している**（§2 違反） |
| `enqueue_batch` | `hipMemcpyAsync` / `hipMemcpyWithStream` H2D | 入力 upload |
| `prepare_plan` | 同期 `hipMemcpy` H2D ×17 | signature 変化時のみ |
| `poll_decode`（新規） | なし（host-mapped word を 1 回読むだけ） | OK |

decode hot path の Host blocking synchronization = **1**（`complete_batch`）。
ただし本質的な非適合は同期の数ではなく、**Host が token ごとに submit する model**
であること。

## 5. 自律化に必要な最小設計（未実装）

1. **plan に loop を持たせる**。`next` の後方参照で body を周回させ、step ごとに
   (a) 出力を output ring へ publish、(b) control ring を `try_pop` して
   CANCEL / SHUTDOWN を確認、を node として埋め込む。
2. **output ring を新設**する。`ControlRing` と同じ SPSC + sequence +
   SYSTEM release/acquire の型を再利用し、**host-readable**（`hipHostMalloc` mapped）に
   置く。Host は `try_pop` 相当を 1 回 observe するだけ。
3. **FSM の停止条件を control ring に移す**。`stop_requested` の host 直接 store では
   なく control command として扱い、safe point で停止 → `STOPPED` を publish。
4. **START のみ signal を使う**。epoch による毎 token kick を削除。
5. **output ring 満杯時の contract を決める**（backpressure / drop / bounded stop）。
   これは §5 が明示的に「勝手に決めるな」と指示しているため、**別途決定が必要**。

## 6. STOP 条件（該当）

| 条件 | 該当 |
|---|---|
| production contract の大規模変更が必要 | **該当**（execution model の変更） |
| Host acknowledgement を control ring が要求する | 非該当（try_push/try_pop は nonblocking） |
| completion の memory ordering が async observation を保証できない | 非該当（SYSTEM scope の release/acquire が既にある） |
| `hipStreamWaitValue32` と MCU signal memory の整合性 | 非該当 |
| GPU producer → HIP stream consumer の visibility | 非該当（D0-D11 で確認済み） |
| generation wrap / ABA の安全性 | **該当**（§3 で 32bit truncate を報告済み。自律 loop では step 数が増えるため必須の検討項目） |

## 7. 現時点の結論

- 既存 primitive（control ring、SYSTEM scope の順序、back-edge 可能な plan）は
  自律化に**再利用できる**。
- 欠けているのは **output ring（host-readable、token 単位）** と
  **plan 内 loop + control check**、および **output ring 満杯時の contract**。
- 現在の per-token host-driven 経路は契約 §2 に違反しており、その上に
  「async API を足す」ことは §2 が禁じる二重 API になる。
