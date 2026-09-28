> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# Production Program Integration Gate — One Layer

- 目的: `PhaseShift Program → MCU Plan Compiler → GPU-MCU → production kernel` の経路を、**机上ではなく実モデルの1層**で成立させる
- 対象: qwen35 の physical dispatch を production の selector 経由で解決し、MCU plan に compile して GPU 上で実行、Host 実行と比較する
- 対象外: production executor の MCU 切り替え、prefill 対応、性能の GO/NO-GO 判定、kernel/packet fusion、embedding / MoE / multi-GPU
- 性能の GO/NO-GO 判定は行わない。**成立可否と制約の記録**が目的
- 現在の仕様: `docs/developer/gpu_mcu/` / `docs/developer/models/qwen35/`

## 1. 基点

| 項目 | 値 |
|---|---|
| base revision | `1f700ca012699f00fbdab9b2ea84b0e83b923a15` |
| worktree | `.worktrees/gpu-mcu-program-plan-one-layer` |
| branch | `poc/gpu-mcu-program-plan-one-layer` |
| 対象モデル | Qwen3.5-4B（`hidden=2560`, `gdn_layers=24`, `conv_dim=8192`, `head_k=head_v=128`） |
| 計算単位 | decode（`rows=1`）。prefill は §9 |

## 2. Gate 結果

| Unit | 内容 | 結果 |
|---|---|---|
| S0 | baseline 確認（revision / acceptance / gpu_mcu） | PASS |
| S1 | hidden kernarg contract（`Required` は fit しないと fail-closed） | PASS |
| S2 | production physical resolver の共有化（selector を Host/MCU で1個に） | PASS |
| S3 | stable entrypoint 化（e4m3 K5120） | PASS |
| T0–T4 | GPU-only region timing（marker-only を Host polling から分離） | 実測済み |
| U0–U4 | `compile_mcu_plan` / layer range 解決 | PASS |
| V0 | L0 inventory（dispatch と resolve の可視化） | PASS |
| Unit 1 | RMSNorm F32 / PerGroup / Direct | PASS |
| Unit 2 | elementwise 4 種（SCALE / MUL / RESIDUAL_ADD / SWIGLU） | PASS |
| Unit 2b | SILU の selector gap（parity 実測 → rule 追加） | PASS |
| Unit 3 | LINEAR_BF16 ExactRows | PASS |
| Unit 4 | L2_NORMALIZE | PASS |
| Unit 5a | GDN conv1d（3D grid） | PASS |
| Unit 5b | GDN recurrence `Wmma` | PASS |
| W1 | One Layer plan の resource audit | PASS |
| **W2** | **One Layer end-to-end（Host vs MCU）** | **PASS** |

## 3. MCU 対応 dispatch 数の推移

分母は実モデル lower 後の **701 dispatch**。

| 時点 | `mcu_supported` | L0 (index 1..22) |
|---|---:|---:|
| Gate 開始 | 65 (9.3%) | 2 / 22 |
| Unit 1（RMSNorm F32/Pg） | 89 | 3 / 22 |
| Unit 2（elementwise 4 種） | 241 | 10 / 22 |
| Unit 2b（SILU selector） | 289 | 10 / 22 |
| Unit 3（LINEAR_BF16） | 538 | 18 / 22 |
| Unit 4（L2_NORMALIZE） | 586 | 20 / 22 |
| Unit 5a（GDN conv1d） | 610 | 21 / 22 |
| **Unit 5b（GDN recurrence `Wmma`）** | **634 (90.4%)** | **22 / 22** |

完全覆蓋した KernelId（実モデルの全 dispatch が対応）:

| KernelId | dispatches |
|---|---:|
| LINEAR_BF16 | 249 |
| RESIDUAL_ADD | 64 |
| RMS_NORM | 89（`q_norm` / `k_norm` の16件は3種目 variant 未対応） |
| L2_NORMALIZE | 48 |
| SILU | 48 |
| SWIGLU | 32 |
| MUL | 32 |
| SCALE | 24 |
| STATEFUL_CAUSAL_CONV1D | 24 |
| GDN_RECURRENCE | 24 |

L0 の残り `no` は **index 0 の `embedding` のみ**。これは layer の手前（`find_dispatch_range_for_graph_prefix(..., "L0")` は index 1..23 を拾う）なので、**L0 layer proper は 22/22**。

## 4. end-to-end 結果（W2）

```
L0 range=[1,23) dispatches=22
plan: nodes=25 variants=17 kernarg_slots=23 invocations=22
layer input: F=2560 dtype=BF16   layer output: F=2560 dtype=BF16 bytes=5120
host L0: launched 22 dispatches
mcu: dispatches=23 fault=0
L0 host vs mcu: 5120 bytes byte-exact
```

- **Host 側は fallback 不要**: `try_launch_optimized` × 22 が全て `Launched`。片方だけ correctness path に落ちていれば byte-exact にはならない
- **`mcu: dispatches=23`** = 22 dispatch + 1 marker、**`fault=0`**
- 比較対象は `L0.output`（`F=2560` / BF16 = 5,120 byte）

### state の同一初期化が成立の条件

L0 には **stateful な conv1d と recurrence** が含まれるため、Host run と MCU run の間で state を揃えないと必ず食い違う。

- `GdnStatePool::initialize(slot)` は **1回のみ**（2回目は `invalid_state: slot already initialized`）
- → 初回 `initialize`、2回目以降は `reset`（どちらも同一の `hipMemsetAsync` でクリア）

この順序を取り違えると `mcu state diff` が出る（conv1d の test で実測済み。`requests` を全 case で共有して `row_count` が末尾の値に残った際にも同じ症状が出た）。

## 5. One Layer plan の resource audit（W1）

| 項目 | 実測 | 上限 | 余裕 |
|---|---:|---:|---|
| dispatch range | [1, 23) = 22 | — | L0 layer 完全 |
| nodes | 25（22 + marker/wait/end） | 16,384 | 655× |
| **variants** | **17** | **64** | **47（27% 使用）** |
| kernarg_slots | 23 | nodes = 25 | — |
| invocations | 22（= dispatch 数、1:1） | — | — |

plan 全体のバイト数:

| 構成 | 計算 | bytes |
|---|---|---:|
| plan nodes | 25 × 28 | 700 |
| variants（retained packet 64B 含む） | 17 × 112 | 1,904 |
| invocations | 3×48 + 7×88 + 8×48 + 2×32 + 1×120 + 1×184 | 1,512 |
| kernarg region | 23 × 368 | 8,464 |
| **合計** | | **12,580（≈12.3 KiB）** |

variants 17 の内訳は事前に試算した「bf16 5 + elementwise 6 + l2 1 + rmsnorm 2 + conv1d 1 + recurrence 1 + probe 1」と**完全一致**。`kMcuMaxVariants = 64` には 47 スロットの余裕がある。

## 6. 発見した不具合と修正

Gate 中に見つけて修正したもの。

### 6.1 AQL dispatch の `dimensions` フィールドが 1D 固定

```cpp
constexpr uint32_t kAqlDw0DimensionsBit = 16;
... (1u << kAqlDw0DimensionsBit);   // 常に 1（1 dimension）
```

- **症状**: `grid_y` / `grid_z` が packet に正しく書かれているのに `blockIdx.y` / `.z` が動かない
- **実測**: `rows=4` の L2 で packet `grid=(512,4,1)` を確認しつつ、`first=8192`（= row 1 開始）から未書き込み `abababab`。Host AQL / MCU とも同一症状
- **原因**: 既存 test が**全件 `global_work_items_y = 1`** だったため一度も露出していない
- **修正**: `make_kernel_dispatch_dw0*` に `dimensions` 引数、`build_aql_packet_template` が grid から導出（`aql_dispatch_dimensions(y, z)` = `z>1 ? 3 : (y>1 ? 2 : 1)`）
- **検証**:
  - conv1d `rows=64` → packet `grid=(2560,1,2)` / **`dims_field=3`** を assert、出力と state とも byte-exact（`rows=1` では `dims_field=1` を assert）
  - L2 `rows=4`（`grid_y=4`）が byte-exact → `grid_y` が無視されていれば row 1..3 が未書き込みで必ず失敗するため、`dims ≥ 2` が効いていることの証明
  - recurrence / bf16 は `dims_field=1`（1D）を assert
- **未了**: L2 test に `dims_field == 2` の assert を追加する（今回は byte-exact の間接証明のみ）

### 6.2 variant dedupe key に `grid_y` / `grid_z` が無かった

`find_variant` は `(kind, grid, workgroup)` のみで、conv1d の 3D grid（`dim3(blocks_x, num_requests, tiles)`）を区別できなかった。`grid_y` / `grid_z` を追加。

### 6.3 `Rig::open` の kernarg region が probe の segment で決まっており大きい kernel を覆えない

- `slot_bytes = aql_kernarg_slot_stride(probe.kernarg_segment_size)` = **336**
- elementwise の segment は **344** → `configure` が `kernarg slot stride does not cover a variant segment` で **fail-closed**
- **修正**: `Rig::open(..., min_kernarg_segment)` を追加。One Layer plan では **376（conv1d が最大、probe は 312）** が必要だった

### 6.4 selector の coverage gap（model の実値が rule に無い）

| 箇所 | gap | 解決 |
|---|---|---|
| RMSNorm | `gdn_norm` = F32/PerGroup/F=4096/G=128 が rule 外 → `Unsupported` | Unit 1 で stable entrypoint を追加 |
| elementwise | `SiluF32ToBf16` の rule が `F=10240` のみ、`SiluBf16ToF32` が `F=6144` のみ。model は 8192 / 4096 | **`test_elementwise` に parity case を追加して byte-exact を実測してから** rule を追加 |
| GDN recurrence | `gdn_recurrence_decode1_supported` に **`num_v_heads == 48u`** がハードコード。model は **32** → `Wmma` に落ちる | **production が実行している `Wmma` 側に MCU entrypoint を追加**（Option B、production 挙動は不変） |
| GDN conv1d | rule に `conv_dim=8192, history=3` が**既に存在** | gap なし（確認のみ） |

**判断の指針**: production が実行している kernel と MCU が実行する kernel を一致させる。rule を足して production の kernel を変える場合は parity を実測し、変更しない方が良ければ **MCU 側を production に合わせる**。

### 6.5 `GdnRecurrenceArgs` は 44 フィールド

field 単位の mirror はタイポで ABI が静かに壊れるため、**size/align の static_assert とコンパイラ側 `sizeof` の cross-assert 付き blob**（184B）で表現した。`kernarg = 184 = sizeof(GdnRecurrenceArgs)` が実測で一致していることが根拠。

## 7. hidden kernarg

`kAqlHiddenArgsBytes = 24`。`IfFits` では fit しない場合に **Host AQL / MCU 両側が同じく skip** し、`Required` のみ fail-closed（S1 の契約）。

| kernel | args | segment | hidden(24B) |
|---|---:|---:|:-:|
| rmsnorm | 48 | 304 | fit |
| elementwise | 88 | 344 | fit |
| gdn conv1d | 120 | 376 | fit |
| **bf16 exact_rows** | 48 | **48** | **不 fit → skip** |
| **l2 normalize** | 32 | **32** | **不 fit → skip** |
| **gdn recurrence** | 184 | **184** | **不 fit → skip** |

`build_aql_launch_metadata` は `aql_launch_metadata_offset(seg)` = **segment の外側・slot の内側**に書くため、**OOB write は発生しない**。不 fit の3種も AQL / MCU とも byte-exact で実証済み。

## 8. 検証した test

すべて Host production launch / Host AQL / MCU の byte-exact を確認。

| test | 検証内容 |
|---|---|
| `test_gpu_mcu_real_rmsnorm_f32_pg` | F32/PerGroup variant（Unit 1） |
| `test_gpu_mcu_real_elementwise` | 6 profile ×（AQL + MCU） |
| `test_gpu_mcu_real_bf16_exact_rows` | L0 の6 shape ×（AQL + MCU） |
| `test_gpu_mcu_real_l2_normalize` | `rows=1` と `rows=4`（2D grid） |
| `test_gpu_mcu_real_gdn_conv1d` | `rows=1` と `rows=64`（3D grid、`dims_field=3`）、**state の byte-exact** |
| `test_gpu_mcu_real_gdn_recurrence` | decode1@48 と **wmma@model shape(32)**、**state の byte-exact** |
| `test_gpu_mcu_qwen_one_layer_plan` | **L0 全22 dispatch の Host vs MCU** |
| `test_elementwise` | SILU の Optimized vs Correctness parity（`F=8192` / `F=4096`, `rows=1`） |

## 9. 制約と未解決事項

### 9.1 Prefill は未対応

inventory の `ctx.actual_rows` を変えて実測。

| | `rows=1`（decode） | `rows=8` |
|---|---:|---:|
| `mcu_supported` | **634 / 701 = 90.4%** | 362 / 701 = 51.6%（※下記） |
| L0 | 22 / 22 | 13 / 23 |

※ `rows=8` は `66231746` 時点の測定で、当時は GDN_RECURRENCE が 0/24。Unit 5b 後は +24 で **386 / 701 = 55.1%** と推定される（同一条件での再計測を推奨）。

落ちる主因は **`LINEAR_BF16` が 249 → 1**。`select_bf16_gemm_config` は `rows == 1` のみ `ExactRows`（MCU entrypoint あり）を選び、`rows ≥ 2` は `Wmma` / `WmmaWide` / `WmmaKPartition`（`verify_exact` 時のみ `ExactRows`）へ逃げるため。

- **構造（AQL geometry）は対応済み**: `grid_y` / `grid_z` / `dimensions` 修正で 2D・3D が byte-exact（L2 `rows=4`、conv1d `rows=64`）
- **selector / variant 層が未対応**: LINEAR_BF16 の prefill variant に entrypoint が無い
- **検証が decode 専用**: required acceptance も `rows=1` 中心
- **計測 harness の制約**: inventory の workspace は `RowBucket::R16` 前提のため `rows > 16` は value alias 判定で落ち、**有効な比較点は `rows ≤ 16`**

### 9.2 embedding

`EMBEDDING_LOOKUP`（index 0、L0 の手前）は未対応。ただし **MCU compiler に case が無い以前に、production が optimized kernel を選んでいない**:

```
emb[0] weight_index=0 encoding=0 rows=248320 cols=2560 out_dtype=BF16 out_F=2560
```

- `encoding=0` = **BF16 storage**
- `embedding_selector` の rule は **PSQ8 のみ**（`{Psq8, 2560}` / `{Psq8, 5120}`）→ BF16 は1件もマッチしない → `Correctness` → production も correctness path

§6.4 と同じ種類の gap。有効化するには (1) rule 追加 + parity 実測、(2) stable entrypoint と MCU plumbing、の2段階。

さらに **前提確認が未了**: `resolve_embedding` は `input_slots[0]` を **dtype 検査せず `const int32_t*`（token ids）にキャスト**しているが、inventory 上その value の dtype は `0 = BF16` に見える。optimized を有効化する前に、`lower_to_primitives` の `em.emit(PrimitiveNode{n}, {tokens}, ...)` で `tokens = em.value(0)` の dtype が本当に I32 かを確認する必要がある。

**L0 のゴールには不要**（index 0 は layer range の外）。

### 9.3 retained template の grid は per-dispatch patch 未対応

`expand_retained_packet` が patch するのは **`kernarg_address` の8 byte のみ**。`grid_size_*` と `publish_dw0` の dimensions は template に焼き込まれる。

そのため template 数は **`(kind, grid, grid_y, grid_z, workgroup)` の組ごと**。実測（decode, rows=1）では全586 dispatch で **16 種**（bf16 7 / elementwise 6 / l2 1 / rmsnorm 2）、L0 では **17（+ probe）** で `kMcuMaxVariants = 64` に余裕がある。

しかし **rows が変わると grid が変わり template が増える**（elementwise / l2 は `grid_y = rows`、conv1d は `grid_y = num_requests`・`grid_z = tiles`）。batch 可変 decode と prefill では 64 を超えうる。

**緩和策（未実施・deferred）**: `kernarg_address` と同様に `grid_size_*` と `publish_dw0` を per-dispatch patch すれば、template は **kind 数（≈10）固定**になる。コストは `expand_retained_packet` の dispatch あたり u32 3本追加（現状は5個の memcpy がある）。hot path を触るため、`template_source` の A/B と合わせて計測してから判断する。

### 9.4 LDS template 経路は維持 / A/B

`test_gpu_mcu_qwen_one_layer_plan` は `PHASESHIFT_MCU_LDS_TEMPLATES=1` で `template_source=1`（LDS 常駐）に切り替えられ、反復数は `PHASESHIFT_MCU_PLAN_ITERS`（既定 **1000**）。**VRAM / LDS を同一反復数で比較**した:

| | VRAM（`template_source=0`） | LDS（`template_source=1`） |
|---|---:|---:|
| 反復数 | 1000 | 1000 |
| dispatch 数 | **23,000** | **23,000** |
| fault | 0 | 0 |
| **1 反復あたり** | **0.7160 ms** | **0.7165 ms** |
| 実行 wall | 4 s | 4 s |
| L0 出力 vs Host | byte-exact | byte-exact |

**差は 0.07%（ノイズ圏）→ 差なし**。§9.3 で挙げた「LDS が有利になる条件（variant 数 / template サイズ / reuse 回数）」について、**variant 17・template 64B・reuse 23,000 回**という規模でも差が出なかったことを実測で追記した。`docs/rnd/gpu_mcu/micro_fsm_retained_aql.md` の M9（variant 1 時点で差なし）と整合する。

**1000 plan の安定性**: 23,000 dispatch 連続で `fault=0`、最終回まで byte-exact 維持。


`docs/rnd/gpu_mcu/micro_fsm_retained_aql.md` M9 で **LDS vs VRAM は差なし**（variant 1 / 64B 時点）と実測済み。「LDS が有利になる条件（variant 数 / template サイズ / reuse 回数）」は同ドキュメントに未解決として記録されている。

既存 test `test_gpu_mcu_lds_templates` がこの経路を検証しており、**「既存の GPU-MCU test を削除しない」**の制約から LDS 経路の削除はできない。採否は §9.3 の patch 検討と合わせる。

## 10. Failures / surprises

- **`rig.start()` の後で null stream の同期 API を呼ぶと永久にブロックする**（本 Gate で実際にはまった）
  - `fill_input()` を生の host 書き込みから **同期 `hipMemcpy`** に変えた直後から、1000 / 100 / **1 反復すべて**で `rc=137`（SIGKILL = timeout）
  - 原因: `GpuMcuFsm::create()` が起動する supervisor kernel は **無限ループ**で、blocking stream 上を走り続ける。同期 `hipMemcpy` / `hipMemset` は **null stream 経由**で、null stream は全 blocking stream の完了を待つ → **supervisor の完了を永遠に待つ**
  - 修正: `hipMemcpyAsync(..., stream)` + `hipStreamSynchronize(stream)` にして **null stream を一切通さない**。修正後は **1 反復 3 秒**（`avg_iter ≈ 0.72 ms`）
  - 教訓: FSM 起動後の host 側 API は **stream-scoped のみ**。これは §10 の「per-dispatch 往復」とは別の独立したハザード
- **`wait_plans(count)` は絶対値待ち**: `plans_started >= count`。反復ループで `wait_plans(1)` を繰り返すと 2 回目以降が即 return し、**plan 2 が未実行のまま読み出し** → `dispatches=23`（期待46）と出力不一致。`wait_plans(it + 1)` にする必要があった
- **`cmd | tail; echo rc=$?` は `tail` の終了コードを見る**: 出力がループ中の print が無いため止まって見えるのを「クラッシュ」と誤認した（実際は timeout）

- **`dimensions` フィールドが 1D 固定だった**: packet の grid は正しく書かれているのに実行されない、という「メモリには正しくあるのに動かない」型の問題。`LastTest.log` のタイムスタンプが**分単位解像度**で、当初「8 テスト × 60 秒」を遅いテストと誤認した経緯がある（撤回済み）。正しい特定は `ctest` の `Passed X sec` 出力の収集による
- **`GdnStatePool::initialize` は2回目が失敗**: `reset` に置き換えないと end-to-end で state が揃わない
- **`requests` を全 case で共有**: 末尾の `row_count` が残り、`rows=1` の case だけ state が不一致（`diff=5108 first=10241` = 3番目の history slot）。原因特定に `first` が byte offset か word index かの確認が必要だった
- **`publish_dw0` は `bytes[]` の外**: `tmpl.bytes[0..3]` は `kAqlInvalidDw0` で、実際の dw0 は別フィールド。`dims_field=0` と誤読した
- **`kernarg_segment_size` の `+256`**: rmsnorm/elementwise/conv1d は `args + 256` だが、bf16 / l2 / recurrence は `args` ちょうど。差異の理由は未解明（layout 一致には影響しない）
- **テスト時間の大半は GPU-MCU Gate と無関係**: required 450.9 秒のうち `test_dflash2_psq4_shapes` 1件で 83.2 秒、`≥5s` の10件で 52.9%、`<0.1s` の59件は合計 1.1 秒。`test_qwen35_mtp_*`（ホストランタイム系）は合計 12.6 秒 = 2.8%

## 11. 次 Gate へ

1. **prefill**: LINEAR_BF16 の prefill variant（`Wmma` / `WmmaWide` / `WmmaKPartition`）の stable entrypoint、`RowBucket` を揃えた計測 harness
2. **batch 可変 decode**: `kMcuMaxVariants = 64` の壁と §9.3 の grid patch
3. **embedding**: §9.2 の前提確認 → rule + parity → entrypoint
4. **RMS_NORM の3種目**（`q_norm` / `k_norm`、F=4096/G=256 と F=1024/G=256）: standard attention layer のみで L0 には不要
5. **`template_source` の A/B**（§9.3 と同時計測）
6. production executor の MCU 切り替え可否（本 Gate では対象外）
