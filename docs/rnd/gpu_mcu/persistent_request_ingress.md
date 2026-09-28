> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# Persistent Request Ingress / Scheduler Boundary Gate（CB2-R）

> Status: R&D record。CB2 で STOP した 4 項目を実装・検証して CB2 を PASS にした。
> 現在の contract は `docs/developer/gpu_mcu/low_level.md` を正本とする。

## 1. 前提

CB0（`69f09b62` Device Slot Table）と CB1（`49819344` Request Ingress /
Device Slot Admission）は PASS 済み。CB2-R は CB0/CB1 の semantics を変えず、
additive に persistent MCU へ接続した。

## 2. 5 段階で実装した

| stage | commit | 内容 |
|---|---|---|
| R1 | `beb47e2b` | CB1 の device command 処理を `request_ingress.h` の `detail` namespace へ共有化。command application と transport を分離し、`GpuMcuCommandApplyResult{event, has_event, scheduler_dirty}` を返す |
| R2 | `20738386` | `GpuMcuPersistentState` へ request runtime state を追加。`configure_request_runtime()` は start 前のみ有効。`start()` の手動 save/restore を `GpuMcuPersistentStartConfig` の capture/restore に置換し、legacy と request runtime の pointer を 1 箇所で管理 |
| R3 | `181634c7` | boundary 冒頭で pending event を flush。push 失敗時は `pending_event` に stage して `pending_event_valid` を立て、**その間は新規 command を consume しない**。command の再実行なし |
| R4 | `725df3b4` | `cancel_requested → terminal_reason = CANCELLED`（phase は維持）、dirty 時のみ runnable snapshot を再構築、`RequestHandle(slot+generation)` を書いて fence → `runnable_count` → release で `scheduler_epoch` を publish |
| R5 | `7e948ebd` | persistent loop の legacy emit の前に boundary を接続。progress が無いときだけ `s_sleep` |

## 3. 途中で踏んだ罠（2 件とも test 側）

| 症状 | 原因 | 対処 |
|---|---|---|
| test が 545 秒ハング、GFX 100% | persistent kernel が `create()` に渡した stream を占有しているのに、**同じ stream** に test kernel を launch | 専用 work stream に分離 |
| test が 136 秒ハング、GFX 100% | persistent wave 稼働中の**同期 D2H `hipMemcpy`** | lint で禁止し、`hipMemcpyAsync` + `hipStreamSynchronize(stream)` へ変換 |

2 件目は当初「loop 接続で `progressed` が毎回真になり sleep しない」と誤診した。
真因は test の D2H で、**lint で禁止した後は loop 接続がそのまま動作した**
（136 秒 → 1.88 秒）。誤診を記録として残す。

## 4. 再発防止（build 時の gate）

`tools/check_mcu_sync.py` のスコープに **persistent MCU を起動する test**
（`test_gpu_mcu_persistent_ingress.hip` / `test_gpu_mcu_persistent_mcu.hip` /
`test_gpu_mcu_runtime_lifecycle.hip`）を追加した。同期 D2H があると
`phaseshift_mcu_sync_check` が build を失敗させる（22 files を走査）。
`request_stop()` / `wait_stopped()` が失敗したら readback せず即 fail する
fail-fast も入れた。

## 5. 検証

| suite | 結果 |
|---|---|
| `test_gpu_mcu_persistent_ingress` | **PASS**（1.88 s） |
| `gpu_mcu` | **52 / 52 PASS**（rc=0） |
| required | **177 / 177 PASS ×2**（95s / 94s） |
| CPU runtime diff | **0**（`include/`・`src/` の qwen35 runtime / state に差分なし） |

## 6. CB2 で未検証だった 4 項目のクローズ

| # | 項目 | 結果 | 根拠 |
|---|---|---|---|
| 1 | Host が何もしなくても persistent MCU が SUBMIT を consume | **YES** | loop 経由で `commands_consumed` / `runnable_count` が進む（R5） |
| 3 | boundary で runnable set を確定 | **YES** | `runnable_count` と `RequestHandle` の snapshot（R4） |
| 4 | CANCEL が次 batch から slot を除外 | **YES** | `terminal_reason = CANCELLED`、phase 維持、自動 release なし（R4/R5） |
| 5 | CLAIMED event を backpressure 時にも失わない | **YES** | pending event staging + 再開時の publish、二重実行なし（R3） |

## 7. 次 Gate へ

- ingress backlog（`no_idle_slot` は暫定挙動）
- `GpuMcuSlotState` に `sampling_params_handle` / `stop_conditions_handle` が無い
  問題（CB1 からの持ち越し。descriptor recycle 後の sampling/stop 解決方法）
- Slot → DeviceRequestDescriptor planning
- Output Ring / Prefix・Radix / autonomous inference loop
