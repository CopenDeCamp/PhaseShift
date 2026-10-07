> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# Full Transformer Serial Static Region / Paged Attention Split-Reduce Gate

## 1. Revision

| 項目 | 値 |
|---|---|
| baseline | `bc5c657282bae6a94e4e4a70d8f1f39e93b144fb`（Standard Attention One Layer Gate の完了地点） |
| branch | `poc/gpu-mcu-program-plan-one-layer` |
| worktree | `.worktrees/gpu-mcu-program-plan-one-layer` |
| 対象モデル | Qwen3.5（`layers=32`, `attention=8`, `linear=24`, `hidden=2560`） |
| 計算単位 | decode（`rows=1`） |

## 2. Gate 結果

| Gate | 内容 | 結果 |
|---|---|---|
| B0 | persistent MCU の spill 除去 | PASS |
| B1–B3 | full transformer body fixture と plan compile | PASS |
| B4–B7 | Host reference / MCU static region / 出力・state 一致 | PASS |
| B8 | mid-region queue starvation の分離計測 | PASS |
| B9 | 100 iteration 全一致 + 1000 iteration endurance | PASS |
| B10–B11 | GPU-only timing / resource 再確認 | PASS |
| C0–C3 | paged attention split/reduce の variant 固定と stable entrypoint | PASS |
| C4–C6 | split / reduce 単体 qualification と 2-physical chain | PASS |
| C7 | compiler の logical 1 → physical 2 展開 | PASS |
| C8–C10 | Standard Attention @4096 / body compile @4096 / body E2E @4096 | PASS |
| C11 | split/reduce recipe 追加後の resource 再確認 | PASS |

## 3. Resource Hardening（B0 / B11）

### 3.1 最初の spill

baseline の persistent supervisor:

| | sgpr | sgpr spill | vgpr | vgpr spill | scratch | code |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 106 | 0 | 191 | **12** | 52 | 72156 |

kernel 単体を同じ compile command で再現し、変更を順に試した。

| 変更 | sgpr | sgpr spill | vgpr | vgpr spill | scratch | code |
|---|---:|---:|---:|---:|---:|---:|
| A: 全 `mcu_write_*` + `mcu_build_kernarg` を noinline | 107 | **4** | 192 | 0 | 80 | 25388 |
| C: `mcu_write_*` のみ noinline | 107 | **2** | 191 | 0 | 64 | 29112 |
| **B: `mcu_build_kernarg` のみ noinline（採用）** | **107** | **0** | **191** | **0** | **64** | **25328** |

- 原因は recipe writer 群がすべて control loop へ inline されていたこと。`mcu_build_kernarg`
  を 1 段 out-of-line にすると recipe 機構全体が loop の register allocation から切り離される。
- writer 側を個別に noinline にすると逆に SGPR spill が出る（call site が増えるため）。
- **scratch 64 は call-frame 由来**である。resource metadata の `sgpr_spill_count` /
  `vgpr_spill_count` はいずれも 0 で、spill 由来ではあり得ない。ISA 上も 24 個の
  scratch 命令が `mcu_build_kernarg` 呼び出し前後の退避・復帰に対応する。
  1 block × 32 thread のため総量は 2 KB で実害はない。

### 3.2 paged split/reduce recipe 追加後

split/reduce を追加した直後は SGPR spill が再発した。原因は recipe 述語
（`mcu_kernarg_recipe_supported`）が if 連鎖のまま hot loop へ inline されていたこと。

| | sgpr | sgpr spill | vgpr | vgpr spill | scratch | code |
|---|---:|---:|---:|---:|---:|---:|
| recipe 追加直後（if 連鎖 inline） | 107 | **6** | 192 | 0 | 64 | 26128 |
| 述語を noinline | 85 | 0 | 191 | 0 | 64 | 22092 |
| **述語を 2 比較の inline に置換（採用）** | **85** | **0** | **191** | **0** | **64** | **21992** |

`mcu_kernarg_recipe_supported` は「recipe id が 1..16 の範囲か」だけを見る
2 比較へ置き換えた。invocation table の非 null と index 範囲は各 writer が既に検査し、
失敗時は `kernarg_build_failed` で fault するため二重検査は不要だった。
これにより register を消費せず、call も発生しない。

noinline 版（SGPR 85 / spill 0）も正しさは同じだが、feeder の per-packet に
device call の遅延が乗るため、同じ GPU を共有する並列 test で queue が空になる
境界が動きやすかった。2 比較の inline 版に置き換えて解消した。

**最終の persistent supervisor は SGPR 85 / SGPR spill 0 / VGPR 191 / VGPR spill 0。**

## 4. Gate B: Full Transformer Body @512

### 4.1 Plan

```
max_visible=512 block_table_stride=32 num_pages=32
transformer body range=[1,697) dispatches=696
plan: nodes=699 variants=25 logical=696 physical=696 kernarg_slots=697 invocations=696
plan: rms=104 e4m3=0 elementwise=216 rope=16 kv_append=8 paged=8 bf16=248
      l2=48 conv1d=24 recurrence=24 psq4=0
```

- 入口は embedding の出力（Program の value binding で解決）、出口は `L31.output`。
- `L31.output` は Program 上は external output であるため、test は実バッファを与える。
- 容量: `variants=25/64`、`nodes=699/16384`、`kernarg_slots=697`。
  `kernarg_slots_required >= physical + marker` を満たす（slot 再利用はしない）。
- 実サイズ: `node_bytes=19572`（699 × `sizeof(McuPlanNode)=28`）、
  `invocation_bytes=47552`、`variant_bytes=2800`。
  @4096 では `node_bytes=19796`（707 × 28）、`invocation_bytes=48704`、
  `variant_bytes=2912`。

### 4.2 Queue

```
queue: depth=4 doorbell=per_packet doorbell_count=766700 refill=766695
       empty=1100 mid_region_empty=0 backpressure=170109 max_ahead=4 wrap=748
```

- `doorbell_count = 697 × 1100` → 全 packet で per-packet doorbell。
- `empty=1100` は **1 region あたり 1 回**で、region 先頭（前 region の drain 直後）に対応する。
- **`mid_region_queue_empty=0`** → region 内部の starvation はない。
- `max_ahead=4` → `queue_ahead_depth=4` を遵守。

### 4.3 Correctness

- Host: 696 dispatch すべて `OptimizedLaunchKind::Launched`。**correctness fallback 0**。
- MCU: region を 1100 回実行し、`fault=0`。
- **full correctness 100 反復のすべてで一致**:
  - `L0.output` … `L31.output` の **32 layer boundary すべて byte-exact**
  - **KV cache（K pool / V pool 全領域）byte-exact**
  - **GDN conv state / recurrence state 全領域 byte-exact**
- endurance 1000 反復で fault / hang / corruption / stale kernarg は 0。

各反復は input / GDN state / KV state を同一 snapshot へ戻してから実行する
（Host 実行後の state を MCU の初期値に使わない）。

### 4.4 Timing

| 経路 | 値 |
|---|---|
| Host HIP body（696 個別 launch） | 22.90 ms |
| MCU GPU-only region（696 dispatch + marker） | **16.739 ms** |
| MCU host-observed（endurance、state reset を含む） | 17.48 ms/region |

GPU-only region は MCU 自身が `region_start_ts` / `region_end_ts` を `wall_clock64()` で
記録した値で、Host の polling 粒度を含まない。性能は本 Gate の GO/NO-GO 条件ではない。

**注意（後日修正）**: 本節の初版は `wall_clock64()` の tick を 1 GHz 前提で
`ticks / 1.0e6` により ms へ換算しており、**10 倍小さい値**（1.67 ms）を記録していた。
正しい換算は device の wall clock rate を使う
`ticks / wall_clock_rate_khz`（実測 100000 kHz）で、上の表は修正後の値である。
詳細は §4.5 と、Production Decode Executor Integration Gate の Timing Truth Gate。

### 4.5 Timing Truth（D0）

`wall_clock64()` は tick を返し、その rate は device attribute で取得する。
本 Gate の初版は rate を仮定して `ticks / 1.0e6` で ms にしていたため 10 倍小さい値を
記録していた。修正後:

| 項目 | 値 |
|---|---|
| `hipDeviceAttributeWallClockRate` | **100000 kHz**（100 MHz） |
| 換算 | `ms = ticks / wall_clock_rate_khz` |
| 旧換算（誤り） | `ticks / 1.0e6` → ms（10 倍小さい） |
| GPU region（@512、696 dispatch + marker） | ticks = 1673900 → **16.739 ms** |
| GPU region（@4096、704 dispatch + marker） | ticks = 1672500 → **16.725 ms** |
| HIP event cross-check（専用 probe、`test_gpu_mcu_wall_clock_conversion`） | device/event = **0.9992**（差 0.08%） |

cross-check は十分長い kernel（event 25.7 ms）で行う。短い kernel では固定
オーバーヘッドが乗り 0.87 程度になるが、これは換算誤差ではない。
換算 helper は `include/phaseshift/runtime/gpu_mcu/infrastructure/wall_clock.h` に置き、
rate が 0 以下なら fail-closed とする。

修正前は「GPU region 1.67 ms」と「host-observed 17.4 ms」が 10 倍乖離していた。
この乖離自体が換算誤りの兆候であり、修正後は 16.74 ms と 17.48 ms で整合する。

---

## 5. Paged Attention Split / Reduce（Gate C）

### 5.1 条件の固定

| 項目 | 実測 |
|---|---|
| q_heads / kv_heads / q_per_kv | 16 / 4 / **4** |
| head_dim / full_head | 256 / **true** |
| rows | 1 |
| kv_dtype | BF16 |
| splits | **16**（`max_visible >= kPagedAttentionSplitMinVisible=2048`） |
| `partials` | `rows × kv_heads × splits × q_per_kv × (2+head_dim)` = 66048 float = **264192 B** |

### 5.2 variant の確定

`PhysicalPagedAttentionVariant` を物理 kernel 単位に具体化した。
**Host / MCU のどちらも 1 個の resolver（`resolve_paged_attention_physical`）を使い、
MCU 側で `q_per_kv` / `full_head` / `qh_per_wg` から kernel を選び直さない。**

reduce の経路は production と同じ規則で決める:

```
qh_per_wg = paged_attention_split_reduce_qh_per_wg()   // env, 既定 1
qh_route  = qh_per_wg < q_per_kv && q_per_kv % qh_per_wg == 0
```

`q_per_kv=4` では `qh_per_wg = 1, 2` が qh 経路、`3, 4, 6` は base 経路になる。

| 物理 kernel | stable symbol | args | kernarg | hidden | grid | workgroup |
|---|---|---:|---:|---|---:|---:|
| split q4 full | `phaseshift_qwen35_attention_paged_bf16_split_q4_full` | 144 | 144 | fit せず → 両側 skip | `rows×kv_heads×splits` = **64** | 128 |
| reduce q4 full qh1 | `..._reduce_q4_full_qh1` | 128 | 128 | fit せず → 両側 skip | `rows×kv_heads×(q_per_kv/1)` = **16** | 128 |
| reduce q4 full qh2 | `..._reduce_q4_full_qh2` | 128 | 128 | fit せず → 両側 skip | **8** | 128 |
| reduce q4 full base | `..._reduce_q4_full_base` | 128 | 128 | fit せず → 両側 skip | `rows×kv_heads` = **4** | 128 |

`qh_per_wg = 1, 2` と base の 3 経路を stable 化したので、実モデルでは
env をどう設定しても production の選択がすべて stable entrypoint に落ちる。

kernel body は既存 template を `__device__` 関数へ抽出して共有し、コピーしていない。
production launcher も同じ条件では stable kernel を使用する（Host と MCU が同一 kernel）。

### 5.3 split 単体 / reduce 単体 / 2-physical chain

`test_gpu_mcu_real_paged_attention_split` が 3 経路（Host production launcher /
Host-produced AQL / MCU chain）を比較する。env で qh_per_wg を 1 / 2 / 4 に切り替えて
3 回実行し、**partials 264192 B と output 16384 B がいずれも byte-exact**。

```
qh1: split_grid=64 reduce_grid=16  chain: partials=264192 B output=16384 B ok=1
qh2: split_grid=64 reduce_grid=8   chain: partials=264192 B output=16384 B ok=1
base: split_grid=64 reduce_grid=4  chain: partials=264192 B output=16384 B ok=1
```

MCU chain は split → reduce → marker の 3 dispatch で実行し、`fault=0`。
reduce は split の partials を読むため、両方 `barrier=1` / Agent acquire / Agent release。

### 5.4 Logical → Physical expansion

`compile_mcu_plan()` は `push_dispatch()` で node を積み、
`logical_dispatch_count = dispatch_end - dispatch_begin`、
`physical_dispatch_count = push_dispatch 回数` を別々に報告する。

@4096 の body で:

```
logical=696 physical=704 nodes=707 variants=26 kernarg_slots=705
paged=0 paged_split=8 paged_reduce=8
```

Standard Attention 8 層の `PAGED_ATTENTION` がそれぞれ 1 logical → 2 physical に展開され、
**696 + 8 = 704** になる。`physical == logical + paged_reduce.size()` を test が検証する。

## 6. Gate C: Full Transformer Body @4096（production 既定）

### 6.1 Plan

```
max_visible=4096 block_table_stride=256 num_pages=256
plan: nodes=707 variants=26 logical=696 physical=704 kernarg_slots=705 invocations=704
plan: rms=104 e4m3=0 elementwise=216 rope=16 kv_append=8 paged=0
      paged_split=8 paged_reduce=8 bf16=248 l2=48 conv1d=24 recurrence=24 psq4=0
```

### 6.2 Queue

```
queue: depth=4 doorbell=per_packet doorbell_count=775500 refill=775495
       empty=1100 mid_region_empty=0 backpressure=167635 max_ahead=4 wrap=757
```

split → reduce の physical expansion 境界を含めて **`mid_region_queue_empty=0`**。
continuous feeder は expansion を跨いでも途切れない。

### 6.3 Correctness

- Host: 696 dispatch、fallback 0。
- MCU: regions=1100（full 100 + endurance 1000）、dispatches=775500、**fault=0**。
- **100/100 反復で 32 layer boundary + KV state（128 MB）+ GDN state（50 MB）が byte-exact。**
- endurance 1000 反復で fault / hang / corruption 0。

### 6.4 Timing

| 経路 | @512 | @4096 |
|---|---:|---:|
| Host HIP body（696 個別 launch） | 22.90 ms | 22.47 ms |
| MCU GPU-only region | **16.739 ms** | **16.725 ms** |
| MCU host-observed（endurance、state reset 込み） | 17.48 ms | 21.55 ms |
| physical dispatch 数 | 696 | 704 |

split/reduce の 8 組（+8 physical dispatch）を追加しても GPU-only region の差は
0.014 ms（0.08%）で、body 全体の dispatch 時間に対して支配的ではない。

## 7. 最終 Acceptance 対応

| # | 条件 | 結果 |
|---:|---|---|
| 1 | persistent MCU SGPR spill 0 | PASS |
| 2 | persistent MCU VGPR spill 0 | PASS |
| 3 | max_visible=512 で body compile | PASS（logical=physical=696） |
| 4 | L0〜L31 を 1 static region で実行 | PASS |
| 5 | layer 境界 WAIT 0 | PASS（WAIT は末尾 marker 後の 1 個のみ） |
| 6 | Host primitive intervention 0 | PASS |
| 7 | 32 layer outputs 一致 | PASS |
| 8 | final body output 一致 | PASS（= `L31.output`） |
| 9 | 全 GDN state 一致 | PASS（conv + recurrence 全領域） |
| 10 | 全 Attention KV state 一致 | PASS（K/V pool 全領域） |
| 11 | body 中の queue starvation 0 | PASS（mid_region_empty=0） |
| 12 | 100 iteration full correctness | PASS |
| 13 | 1000 iteration endurance で fault/hang/corruption 0 | PASS |
| 14 | split stable AQL entrypoint | PASS |
| 15 | reduce stable AQL entrypoint | PASS |
| 16 | split partials Host/AQL 一致 | PASS |
| 17 | reduce output Host/AQL 一致 | PASS |
| 18 | logical 1 → physical 2 を表現 | PASS（696 → 704） |
| 19 | Standard Attention @4096 | PASS |
| 20 | body @4096 compile | PASS |
| 21 | body @4096 で output/state 一致 | PASS |
| 22 | body @4096 でも layer 境界 WAIT 0 | PASS |
| 23 | body @4096 でも mid-region starvation 0 | PASS |
| 24 | spill 0 を維持 | PASS |
| 25 | required acceptance 全 PASS | 本文 §8 |

## 8. 検証した test

新規:

| test | 内容 |
|---|---|
| `test_gpu_mcu_full_transformer_body_plan` | Gate B / Gate C の本体（512 / 4096 を `PHASESHIFT_BODY_MAX_VISIBLE` で切替） |
| `test_gpu_mcu_real_paged_attention_split` | Gate C4/C5/C6（3 経路、全 qh 経路） |

既存 regression:

- required acceptance 全件、`ctest -L gpu_mcu` 全件
- `test_paged_attention_split_reduce_exact`、`test_paged_attention`、
  `test_qwen35_paged_attention_selector`（production paged attention 経路）
- `test_gpu_mcu_attention_one_layer_plan`、`test_gpu_mcu_qwen_one_layer_plan`
- `test_gpu_mcu_continuous_refill` / `test_gpu_mcu_continuous_ring_wrap` /
  `test_gpu_mcu_real_primitive_chain`（counter 改名の影響）

## 9. Failures / Surprises

1. **supervisor の spill は recipe writer の inline が原因だった。**
   recipe 数そのものではなく、`mcu_build_kernarg` が control loop へ inline されることが
   原因。out-of-line にすると spill が消え、code も 3 分の 1 以下になった。
   追加 recipe のたびに recipe 述語を loop の外へ出す必要がある。
   **ただし out-of-line 化は feeder の per-packet 遅延を増やす。** 述語のように
   「安価だが inline すると register を食う」処理は、if 連鎖をやめて
   比較数個の inline 形へ書き換えるほうが両方の条件を満たす。
2. **`test_gpu_mcu_real_rmsnorm_feed` の queue 判定は並列 test 負荷に敏感。**
   この test は `queue_empty_count <= plans` を要求する。同一 GPU を共有する状態で
   -j4 実行すると feeder がわずかに遅れ、marker packet の publish 時にも queue が
   空になることがあり 1 回だけ超過した。単独実行では 3/3 で `empty == plans`。
   上記の述語 inline 化で feeder 遅延を戻したところ、gpu_mcu suite を -j4 で
   3 連続実行しても再発しなくなった。
2. **scratch を spill と混同しない。** metadata の spill count と
   `private_bytes` は別物で、noinline 化で scratch が増えても spill 0 なら退行ではない。
3. **`L31.output` は Program 上 external output。** stub ポインタ（0x2000）では
   Host reference が memory fault する。実バッファを与える必要がある。
4. **`hipMemset`（null stream）と AQL hardware queue は順序付けられない。**
   前 Gate で発見済み。state を zero した直後に packet を submit する test は
   必ず `hipDeviceSynchronize()` を挟む。
5. **`dispatch_seq` は FSM 全体の通し番号。** 「region 先頭の dispatch」を判定するには
   `run_issues` を使う。`dispatch_seq != 0` では 2 回目以降の region 先頭を誤って数える。
6. **queue counter は live range を増やすと spill する。**
   region 内の starvation 計測は accumulator を持たず、
   発生時に atomic で直接 publish することで register を消費せずに実装した。
7. **reduce の kernel 選択は env 依存。** `PHASESHIFT_PAGED_ATTENTION_SPLIT_REDUCE_QH` が
   1 / 2 のとき qh 経路、3 / 4 / 6 では base 経路。resolver はこの規則を
   production と共有し、MCU 側で再実装していない。
8. **`wall_clock64()` の tick を ns として扱っていた。** device の wall clock rate は
   実測 100000 kHz で、`ticks / 1.0e6` は ms を与えない（10 倍小さい）。
   「GPU region と host-observed が 10 倍違う」という症状が検出の手掛かりになった。
   換算は必ず `wall_clock_rate_khz` を使う。合成 probe の cross-check では
   十分長い kernel を使わないと固定オーバーヘッドで偽の不一致が出る。
9. **persistent FSM は plan と node_count を kernel 起動時に cache する。**
   test の途中で `Rig::upload()` により plan を差し替えても反映されず、
   古い node の残骸を実行して hang する。marker-only の計測は専用 Rig を
   用意する（`test_gpu_mcu_production_w4a8_chain` はそうしている）。
10. **`queue_full_count` は ahead-depth backpressure の回数**を意味していたため
   `backpressure_wait_count` に改名し、region 内 starvation 用に
   `mid_region_queue_empty_count` を新設した。

## 10. 次 Gate へ

Gate B / Gate C ともに成功した。production 既定 `max_visible_tokens=4096` で

    Embedding output
        ↓
    L0 … L31（1 static region、layer 境界 WAIT なし）
        ↓
    Final Marker / WAIT / END

が Host primitive intervention なしで通り、32 layer output と全 state が一致する。

**Case D1: Full Body @4096 PASS** に相当するため、次は
**Production Decode Executor Integration** を候補とする。初期構成は

    Host     : Embedding
    GPU MCU  : Transformer body
    Host     : LM Head / Sampling

でよい。Embedding / LM Head / Sampling の GPU-MCU 化はその後の個別 Gate とする。

今回の対象外（継続）:

- prefill
- Q/K/V / Gate-Up の parallel issue
- Barrier AND/OR、EXT_KERNEL_DISPATCH、dependency signal
- MoE、multi-GPU、multi-request scheduler
- production runtime の既定切替
- embedding / LM head / sampling の MCU 化
