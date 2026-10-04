> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# CB9〜CB10 Continuous Turnover と Device Resource Ownership（Gate 記録）

> Status: R&D record。request の連続入れ替え（CB9）と、device 側が request の
> resource を所有する機構（CB10）の実装と Gate 検証の記録。現在も有効な contract は
> [docs/developer/gpu_mcu/low_level.md](../../developer/gpu_mcu/low_level.md) を参照。

## 1. revision

| | |
|---|---|
| baseline | `1f0e65b5`（CB8 Autonomous Closed Loop） |
| final | `bfbf4044` |
| worktree | `.worktrees/gpu-mcu-program-plan-one-layer` |

| commit | 内容 |
|---|---|
| `0433e9ec` | terminal output record の明示 |
| `3e56f0eb` | slot generation を release で進める |
| `42b04aca` | terminal 配送後の slot release |
| `c263e31d` | admission backlog の state |
| `34643d38` | slot が空いたら queued request を admit |
| `d6030be6` | CB9 turnover の qualification |
| `15eecbdf` | autonomous loop を 8 token まで駆動 |
| `810ea36f` | device 側 KV page allocator |
| `341b745c` | device sequence resource manager |
| `8a7711ce` | release call site の更新 |
| `e841fa0f` | 必要な KV page だけを incremental に確保 |
| `dd35dd66` | loop 内で KV page を確保・返却 |
| `17c454c5` | resource lifecycle の実測（telemetry + capacity exhaustion） |
| `d1cffc37` | CB9 / CB10 の Gate 記録 |
| `a9cf3dae` | batch row へ physical sequence slot を publish |
| `0895c0c3` | model の block table へ page id を publish・clear |
| `f262adff` | GDN reset の統合点を記録 |
| `bfbf4044` | GDN state reset を MCU-issued kernel で |

CB9 の `d6030be6` 時点で gpu_mcu 60/60、CB10 の各段で 61→62→63 と増え、
`bfbf4044` で 64/64 / required 189/189。

## 2. CB9 — Continuous Request Turnover

### 2.1 terminal は output ring で観測する

terminal slot は release 後に速やかに recycle されるため、停止後に slot table を
readback しても terminal を観測できない。**terminal は output ring の record で観測する**
ことを contract にした。

`GpuMcuOutputRecord.flags` が record kind を持つ。token record は kind 0 のまま、
terminal flag は terminal record だけが立てる。これにより「token を出さずに終わる」
経路（cancel、pre token error、`max_new_tokens == 0`）でも request あたり丁度 1 個の
terminal record が出る。

`GpuMcuSlotRuntimeState`（64 byte、slot id で index）は、固定 256 byte の slot state に
入らない lifecycle bookkeeping を持つ: terminal publish 状態、release pending、
resource blocked、scheduler epoch。slot generation に自己 bind するため、recycle された
slot は claim path に触れずに初期化される。

### 2.2 slot generation は release で進める

CB9 で generation の前進位置を **claim 時から release 時へ**移した（`3e56f0eb`）。
request が終わった瞬間に古い handle が stale になる。

- release は generation を進める。`UINT32_MAX` からの前進は拒否するため、stale handle が
  一致しうる generation で slot が戻ることはない。
- claim は slot が既に持つ generation を使い、初回 claim だけ invalid 値から seed する。

### 2.3 terminal publish → release の順序

boundary の先頭は次の順で走る。

```text
output flush → terminal publish → release pass → admission → control ingest
→ reconcile → resource reserve → snapshot → plan → binding
```

release eligibility（`gpu_mcu_slot_release_eligible`）は次をすべて要求する。

- `terminal_reason != none`
- `pending_count == 0`（未配送の output が残っていない）
- `terminal_published != 0`（terminal record が ring に入った）
- `release_pending != 0`
- resource を管理している場合は KV / GDN handle を解放してよい。tree / prefix handle は
  **常に拒否**する（所有する subsystem がまだ無い）。
- resource を管理していない場合、KV / GDN handle が残っていれば release しない
  （管理されていない resource を黙って捨てない）。

release は `gpu_mcu_binding_detach` で durable binding を切り離す。ring full で terminal を
push できない slot は `terminal_publish_pending` のまま release されない（global stop は
作らない）。

### 2.4 admission backlog

slot が全部埋まっている submit を `no_idle_slot` で拒否せず、device 上に queue して
slot が空いた時点で admit する。

- `GpuMcuPendingAdmission` は request id と descriptor handle / generation の組を持つ固定
  容量 FIFO。ingress は ready と claimed の間に queued 状態を持つ。
- `gpu_mcu_claim_entry` に claim 手順を切り出し、直接 submit と queued descriptor の
  admit が同じ手順を使う。
- admission は release pass の直後に backlog 先頭を 1 件だけ admit する。
- backlog が満杯なら `admission_backlog_full` で reject する（黙って捨てない）。
- cancel は admission 前でも効く（backlog から request id で取り除く）。cancel は
  走行中の current transaction を途中で破棄せず、次 loop から除外して cancelled
  terminal を出す。
- `apply_*` の signature は増やさず、戻り値の `slot_exhausted` で deferral を伝える。

### 2.5 control drain budget

`configure_control_drain(max_commands_per_boundary)`（start 前のみ）で 1 boundary に
ingest する command 数の上限を決める。既定は 1。`0` は invalid。

### 2.6 CB9 qualification

`test_gpu_mcu_continuous_turnover`: 20 request / 2 slot。Host は **slot も batch も指定
しない**。descriptor は CLAIMED 後に recycle し、あふれた request は admission backlog で
待つ。検証項目:

- 全 request が admitted
- request ごとに terminal record が丁度 1 個
- slot reuse ごとに generation が単調増加
- backlog が完全に drain する
- `batches_failed == 0`

## 3. CB10 — Device Resource Ownership

CB10 の目的は「request の resource lifecycle を device（persistent MCU）が所有する」
こと。Host は resource を確保・解放しない。

### 3.1 KV page allocator（`kv_page_allocator.h`）

page の所有権が device へ移る。page は **tag** が所有する。tag は slot に限らず、CB10 の
loop 統合以降は resource handle そのものを使う。

- `gpu_mcu_kv_page_alloc_tag` は free stack から pop し、渡した page すべてに tag を書く。
- `gpu_mcu_kv_page_free_tag` は **その tag が所有する page だけ**を返す。stale generation
  や duplicate release は 1 枚も所有しないため 1 枚も返さない（拒否分岐ではなく tag
  不一致で構造的に安全）。
- 不変条件は `free_count + owned == capacity`。stale / duplicate free はこの不変条件を
  壊せない。

### 3.2 sequence resource manager（`sequence_resource.h`）

slot は **opaque な 64 bit handle** だけを持ち、handle を解釈するのは manager だけ。
slot 側は handle を index として解釈しない。

```text
handle = (resource_generation << 32) | (resource_index + 1)
0 は「resource なし」
```

- **resource generation は request slot generation と別管理**。release で resource
  generation が進むため、slot generation が動く前に古い handle が無効化される。
- `gpu_mcu_resource_resolve` は entry が active かつ generation が一致する場合のみ受理。
- physical な sequence / GDN slot は初期実装では resource index と 1:1。Radix / prefix
  共有が入っても `GpuMcuSlotState` の ABI を変えずに分離できる構造にしてある。

### 3.3 resource lifecycle 接続

- CLAIMED 時に `gpu_mcu_resource_allocate` して handle を slot へ書く
  （`gpu_mcu_resource_commit_claim`）。claim の直接経路と admission 経路の両方が通る。
- release 時に **page を返してから resource を返す**。resource が戻った時点で page が
  残っていない。
- `test_gpu_mcu_sequence_resource` / `test_gpu_mcu_kv_page_allocator` は handle の
  resolve / stale / incremental growth / balanced 不変条件を実測する。

### 3.4 incremental page reservation

`DeviceSequenceResourceEntry.kv_page_count` を持ち、**足りない分だけ**確保する。
`test_gpu_mcu_sequence_resource` が incremental growth、再確保の冪等性、失敗時に何も
変えないこと、release が確保した分だけ返すことを検証している。

### 3.5 loop 統合

boundary に **reserve pass** を入れる（snapshot の前）。

```text
全 live slot について必要 page を計算
  required = ceil((next_position + next_rows) / kGpuMcuKvRowsPerPage)
  PREFILL: next_position = prefill_position, next_rows = 残り chunk 行
  DECODE : next_position = sequence_length,  next_rows = 1
reserve 成功 → resource_blocked を解除
reserve 失敗 → **該当 slot のみ** resource_blocked = 1（他 slot は継続）
→ scheduler_dirty = 1 → snapshot 再構築が resource_blocked の slot を除外する
```

OOM を runtime 全体の failure にしない。`kv_pages.capacity == 0`（未設定）なら reserve
pass は無効で、既存の挙動を変えない。

### 3.6 telemetry

`GpuMcuExecutionTelemetry` に resource lifecycle の計測を足した。これは「loop 統合が
実際に動いている」ことを証明するために必要だった（§4）。

| field | 意味 |
|---|---|
| `resources_allocated` | claim で resource を確保した回数 |
| `resources_released` | release で resource を返した回数 |
| `kv_pages_reserved` | reserve pass が実際に増やした page 数 |
| `kv_pages_released` | release pass が返した page 数 |
| `resource_blocked_events` | slot が resource_blocked へ遷移した回数 |
| `resource_unblocked_events` | resource_blocked が解除された回数 |

### 3.7 GDN state reset（MCU-issued kernel）

CPU runtime は `GdnStatePool::reset()`（`src/phaseshift/models/qwen35/state/gdn_state_pool.cpp`）
で `hipMemsetAsync` を 2 回（conv history の bf16、recurrent の f32）発行する。MCU では
これを device から発行する。

- `gdn_reset.h` の `GpuMcuGdnResetInvocation`（48 byte）: conv base / recurrent base /
  batch context / slot stride / slot count。conv は bf16 を `uint16_t*` として見る。
- kernel は `row_positions[row] == 0` の row だけを対象にする。これは新 sequence の
  最初の row であり、同一 request では繰り返さないため、**毎 loop dispatch しても冪等**。
  対象 row の `row_sequence_slots[row]`（adapter が publish した physical sequence slot）
  の conv / recurrent 範囲を grid-stride で zero 化する。
- recipe / invocation table / kernarg writer / variant kind / registry entry が揃い、
  通常の primitive として plan から dispatch できる。
- 実測（`test_gpu_mcu_gdn_reset`）: conv 64/64・recurrent 256/256 が zero 化され、
  隣接 sequence は poison（`0xabcd` / `0xdeadbeef`）のまま。plan は
  `probe → wait → reset → probe → wait → end` の 3 dispatch で完走。

layer あたり MB 規模（recurrent は `num_v_heads * head_k * head_v * 4 byte`）なので、
persistent loop の 32 thread で zero 化する方式は採らない。必ず worker kernel として
dispatch する。

## 4. 「assertion が何も証明していなかった」件

CB10 の loop 統合後、capacity exhaustion の test を書いたが `resource_blocked` が
一度も発生せず FAIL した。調査の結果、**production code ではなく test の駆動方法**が
原因だった。この過程で、既存の assertion が弱いことも判明した。

### 4.1 症状と真因

| 症状 | 真因 |
|---|---|
| 1 page pool でも block しない | test が 1 反復につき 1 件だけ submit し、間に 1ms sleep を入れていた。1 request は µs で完走するため **slot が一度も重ならず**、page の競合が起きない |
| pool の balance assertion が通る | 1 枚も確保・返却しなくても `free == capacity` / `owner == 0` は成立する |

### 4.2 対策と実測

telemetry を入れたうえで、駆動を burst 投入（window を常に埋める）に変更した。

```
test_gpu_mcu_continuous_turnover:  alloc=20   rel=20   resv=20   freed=20   blocked=0 unblocked=0
test_gpu_mcu_kv_page_turnover:     alloc=2000 rel=2000 resv=2000 freed=2000 blocked=3 unblocked=3
```

`test_gpu_mcu_kv_page_turnover` は **4 slot が 1 page pool を共有**する構成で、
最初の reserve pass で 3 slot が block される。それでも 2000 request は全て terminal へ
到達し、page が戻ると 3 slot が解除され、resource も page も leak しない（約 2 秒）。

> 教訓: 「成功した」assertion は、**失敗しうる assertion** でなければ検証ではない。
> 計測を先に入れ、0 でないことを確認してから acceptance を書く。

### 4.3 統合時に確定した FSM contract 2 件

GDN reset の統合で 12 時間規模のハングを踏み、次の 2 点が contract として確定した。

1. **real primitive は completion を自分で書かない。** FSM は wait node で
   `completions[slot]` の generation 一致を spin する（`micro_fsm.hip:1116`）。この
   completion を書くのは probe（supervisor probe image）だけであり、AQL packet の
   `completion_signal` は 0 のまま（`make_kernel_variant` が 0 を入れ、FSM は kernarg
   address しか patch しない: `kRetainedPatchKernargAddress` = offset 40）。したがって
   **real primitive の plan は probe で挟む**。既存の real primitive test もこの形。
   - 症状は `supervisor=running / fault=0 / plans=0 / dispatch=0`。`plans_started` と
     `dispatches_committed` は plan 末尾か定期 publish でしか host に見えないため、
     **`dispatch=0` は「dispatch していない」ことを意味しない**。
   - 切り分けは debug record（`McuDispatchRecord`）→ AQL packet dump → metadata 比較 →
     invocation 実値の順で行うと 1 段で確定する。実測では probe 312 byte / reset 304 byte /
     conv1d 376 byte の kernarg で、`PKT` の `grid`・`object`・`kernarg` はすべて正常だった。
2. **device readback は wave を止めてから。** persistent wave 稼働中の D2H は返らない
   （`docs/developer/gpu_mcu/low_level.md` の stream 契約）。test では
   `request_stop()` + `wait_stopped()` を先に呼ぶ。plan は完走していたのに readback で
   止まり、「plan が固まっている」と誤診しやすい。

## 5. sync audit

production hot path の同期は **0**。

- `hipDeviceSynchronize` / 同期 D2H `hipMemcpy`: 追加なし（`tools/check_mcu_sync.py` clean）
- `hipStreamSynchronize` は `GpuMcuPersistentMcu::shutdown()` 等の lifecycle 経路のみ
- Host は slot も batch も指定しない。`try_pop` を呼ばなくても capacity 内で progress する
- test の readback は `request_stop()` + `wait_stopped()` の後だけ

## 6. tests

| label | 件数 |
|---|---|
| gpu_mcu | 64 / 64 PASS |
| required | 189 / 189 PASS |

CB9 / CB10 で追加・強化した test:

- `test_gpu_mcu_continuous_turnover`（CB9 acceptance + CB10 resource 検証）
- `test_gpu_mcu_kv_page_allocator`
- `test_gpu_mcu_sequence_resource`
- `test_gpu_mcu_kv_page_turnover`（capacity exhaustion / 2000 claim-release / leak 0）
- `test_gpu_mcu_gdn_reset`（MCU dispatch した reset の zero 化と隣接 sequence の保護）

### 6.1 負荷依存の既知 flake

`test_gpu_mcu_continuous_ring_wrap` は「backpressure を観測した（`full > 0`）」を assertion
にしている。これは FSM の pacing と consumer の drain 速度の比で決まるため、`-j4` の並列
実行では観測できないことがある（実測: 並列時 `wraps=156 full=0 empty=7` で FAIL、単独時
`wraps=156 full=1 empty=27` で PASS）。同じ commit で required を 2 回回し、1 回目のみ
FAIL、2 回目は 189/189 PASS だった。assertion 自体（backpressure を観測できること）は
意味があるので残すが、FAIL したときは単独実行で再確認する。

## 7. known deferrals

- **block table adapter**: 既存 `SequenceBlockTableDeviceView` / `PagedKV*View` /
  `GdnStatePoolDeviceView` をそのまま adapter として使い、resolve 済みの
  `sequence_slot` を既存 execution path へ供給する。ResourceManager の存在を
  PagedAttention / KV Append kernel に見せない。
- **GDN state reset の model 側 plan への組み込み**: `compile_mcu_plan` が GDN state を
  触る range の先頭に reset node を出し、`McuDecodeRuntime::prepare_plan` が
  `plan.gdn_reset` を upload する。§3.7 の kernel と合わせて production path で完結する。
- **CB11**: production mixed PREFILL + DECODE + VERIFY physical batch qualification。
- **CB12**: stop / cancel / failure exhaustive qualification。
- **CB13**: PhaseShift-Server production integration。
- **Radix R0-R8**: GPU Radix、refcount、eviction、Tree Sparse。`kv_sequence_handle` は
  opaque のまま維持するため、slot ABI を変えずに載せられる。

## 8. この Gate が証明したこと

> Host が request を渡した後、slot の割り当て、terminal の判定、slot の reuse、
> **KV page と sequence resource の確保・解放まで**を GPU MCU だけで閉じられること。
> Host は per-token / per-batch の介入をしない。

device 側 resource ownership の中核は成立した。残るのは「既存 kernel への供給配線」
（block table adapter）、GDN state reset、および大規模 mixed batch の qualification。
