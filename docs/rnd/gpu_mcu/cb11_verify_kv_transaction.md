> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# CB11-VT SPEC_VERIFY KV Transaction Foundation（Gate 記録）

> Status: R&D record。SPEC_VERIFY の resource transaction primitive を
> 完成させた bounded sub-gate の記録。現在も有効な contract は
> [docs/developer/gpu_mcu/low_level.md](../../developer/gpu_mcu/low_level.md) を参照。

## 1. 目的と範囲

VERIFY 実行前に candidate 1..32 token 分の KV capacity を一時予約し、accepted count に
応じて accepted boundary まで KV を残し、rejected speculative tail を Host なしで
GPU 上で返却できることを証明する。

今回証明するもの:

```text
reserve(candidate) + commit(accepted) + rollback(rejected tail)
```

今回対象外（DONE 扱いしない）:

- actual SPEC_VERIFY model execution
- device-side accepted_count 生成
- mixed PREFILL / DECODE / VERIFY execution
- persistent dynamic program plan bridge
- GDN/recurrent accepted-boundary transaction
- `scheduled_batch.cpp` の VERIFY guard 解除

## 2. revision

| | |
|---|---|
| baseline | `6a3f4b6a`（CB11 mixed-production 系の最新） |
| worktree | `.worktrees/gpu-mcu-cb11-mixed-production` |

## 3. transaction metadata

新しい sidecar は作らず、`GpuMcuSlotRuntimeState`
（`include/phaseshift/runtime/gpu_mcu/commit/slot_runtime.h`）へ置く。予約フィールドを明示 field へ
置換し `sizeof == 64` / `alignof == 64` を維持する。

| field | 意味 |
|---|---|
| `verify_txn_active` | transaction が進行中か |
| `verify_base_page_count` | VERIFY 開始前から所有していた page 数（rollback 不可） |
| `verify_reserved_page_count` | reserve 後 page count - base |
| `verify_txn_epoch` | transaction 識別 epoch（scheduler epoch） |

## 4. reservation

`gpu_mcu_resource_reserve_pass` は `GpuMcuSlotPhase::verify` のとき

```text
next_position = committed_position
next_rows     = verify_candidate_count   (1..32、範囲外なら reserve しない)
required      = ceil((committed_position + verify_candidate_count) / rows_per_page)
```

を要求する。reserve が成功して transaction が新規なら `base = reserve 前 page count` を
記録し、`verify_txn_active = 1`、`verify_txn_epoch = scheduler_epoch` とする。
**active な transaction では base を再設定しない**（repeated boundary で speculative tail が
元からあった page に見えるのを防ぐ）。reserve 失敗は all-or-nothing で、新規 transaction なら
metadata を clear し `resource_blocked = 1`、既 active なら予約済み page を戻さず維持する。

## 5. tail shrink primitive

`gpu_mcu_resource_shrink_kv_pages(mgr, pool, blocks, handle, sequence_slot, keep_pages, out_freed)`
（`sequence_resource.h`）を追加した。

- **検証を全件先に行い、1 件でも不正なら mutation 0。**
  handle resolve、`keep_pages <= current`、block table 有効、
  `current <= block_table_stride`、`free_count + to_free <= capacity`、
  tail block の `page < capacity` と `pool.owner[page] == handle` を全件確認する。
- mutation は検証後のみ。tail を `owner=0` → `free_stack` push → `block_table=0` の順で処理し、
  最後に `entry.kv_page_count = keep_pages`、`__threadfence_system()`、
  `*pool.free_count = cursor`。
- page id 0 は有効な page。empty 判定に `block_table entry == 0` を使わず、
  ownership は `pool.owner[page] == handle` のみで判定する。

## 6. finalize / abort

`batch_commit.h` に device helper を追加した。

- `gpu_mcu_verify_resource_commit(slot, meta, mgr, pool, blocks, accepted_count)`
  - `txn_active != 0`、`slot.phase == verify`、`accepted <= verify_candidate_count`、
    `accepted <= 32`、handle resolve 成功を検証。
  - `keep = max(verify_base_page_count, ceil((committed_position + accepted) / rows_per_page))`。
  - `current > keep` なら shrink。成功後 transaction metadata を clear。
  - `accepted == candidate` でも metadata を clear。shrink 0 枚も正常。
- `gpu_mcu_verify_resource_abort(slot, meta, mgr, pool, blocks)`
  - `keep = verify_base_page_count` まで tail を release。invalid accepted / execution failure 用。

## 7. batch commit integration

`GpuMcuBatchCommitView` に optional な `slot_runtime` / `resources_enabled` /
`resources` / `kv_pages` / `kv_blocks` を追加した。`resources_enabled == 0` は従来 behavior を
維持する。

SPEC_VERIFY の順序は **resource finalize → logical commit** に固定した。finalize が失敗、
または accepted が invalid（`> row_count` / `> 32`）なら、transaction を abort し slot を
ERROR にして logical progress を一切書かない。二重 finalize は `txn_active == 0` で refuse し
double free しない。

## 8. qualification

新規 required test `test_gpu_mcu_verify_kv_transaction`（`gpu1;gpu_mcu;required`）:

- prefix 15 / candidate 32 / accepted 0, 1, 2, 17, 18, 32 の page boundary。
- capacity 不足（all-or-nothing）。
- stale resource handle（mutation 0）。
- tail owner mismatch（mutation 0）。
- duplicate finalize（double free なし）。
- base over-reservation（prefix 31 で base 3 pages、accepted 0 でも 3 pages 維持）。
- 2 slot isolation（他 slot の owner / block table / page count 不変）。
- physical page id 0 を kept region に置き、clear sentinel と混同しない。
- abort で base page count ちょうどへ戻す。
- persistent reserve pass が `committed + 32` 分を予約し、repeated boundary で base を
  書き換えない。

`test_gpu_mcu_sequence_resource` の shrink case も `shrink` 名・検証強化へ更新した。

## 9. sync audit

production helper への `hipDeviceSynchronize` / `hipStreamSynchronize` / D2H / blocking HSA は
追加していない。全処理 device-side primitive。`tools/check_mcu_sync.py` clean。

## 10. この Gate が証明したこと

> VERIFY が最大 32 row 先まで KV を一時的に伸ばしても、accepted A token だけを残し、
> rejected tail を Host なしで安全に free stack へ戻せること。base pages は決して
> rollback されず、stale handle・owner mismatch・duplicate finalize・page id 0 のいずれでも
> pool invariant（`free + owned == capacity`）が保たれる。

次は `CB11 Dynamic Mixed Execution Binding`（per-loop DeviceBatchContext → static Program
skeleton + GPU-side dynamic physical dispatch binding）と、actual SPEC_VERIFY execution
（accepted_count の device 生成、GDN transaction、`scheduled_batch` guard 解除）。
