> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# Standard Attention One Layer Gate — R&D record

この文書は Standard Attention を1層完全に GPU-MCU で実行する Gate の
R&D 記録である。現在有効な contract は `docs/developer/` に、
外部 fact は `docs/references/` に置く。

## 1. 基点

| 項目 | 値 |
|---|---|
| baseline | `626db15992a426e1745544f2130e8086cc0314fb` |
| branch | `poc/gpu-mcu-program-plan-one-layer` |
| worktree | `.worktrees/gpu-mcu-program-plan-one-layer` |
| 開始時 `git status` | clean |

開始要件の実測:

| 項目 | 結果 |
|---|---|
| required acceptance | **162 / 162 PASS**（457 秒、skip 0） |
| `ctest -L gpu_mcu` | **39 / 39 PASS**（90 秒） |
| 指定 5 test | **5 / 5 PASS**（4 GPU 並列 6 秒） |

指定 5 test = `test_gpu_mcu_qwen_one_layer_plan` / `test_gpu_mcu_qwen_one_layer_inventory` /
`test_gpu_mcu_program_plan_three_primitive` / `test_gpu_mcu_continuous_refill` /
`test_gpu_mcu_continuous_ring_wrap`。

GDN L0 One Layer は `test_gpu_mcu_qwen_one_layer_plan` の PASS により
**引き続き Host vs MCU byte-exact** である。

GPU は **0〜3 の4枚**を `PHASESHIFT_TEST_GPUS=0,1,2,3` と `ctest -j4` で並列使用する。
`phaseshift-gpu-test-runner` が VRAM budget（既定 24GB/GPU）で GPU を予約し、
`HIP_VISIBLE_DEVICES` を子プロセスへ渡す。

### 既存の dispatch 総数

`graph_nodes=701 dispatches=701`、`mcu_supported=634 / unsupported=67`（GDN inventory の再確認）。

---

## 2. Gate A0 — Standard Attention Layer Inventory

test: `test_gpu_mcu_attention_layer_inventory`

layer は手書きせず、`config.layer_types` の中で最初に `full_attention`
となる layer を test 内で取得する。

```
first_attention_layer=3 prefix=L3.
attention range=[67,88) dispatches=21
```

Qwen3.5-4B の `text_config.layer_types` は 32 件
（`linear_attention` 24 / `full_attention` 8、`full_attention_interval=4`）。
GDN L0 が 22 dispatch だったのに対し、**Standard Attention L3 は 21 dispatch**。

---

## 3. Gate A0.2 — Attention shape（実測）

`models/Qwen3.5-4B/config.json` の `text_config` が source of truth。

| 項目 | 実測値 | 根拠 |
|---|---:|---|
| hidden_size | **2560** | `hidden_size` |
| q_features | **4096** | `num_attention_heads × head_dim` = 16 × 256 |
| kv_features | **1024** | `num_key_value_heads × head_dim` = 4 × 256 |
| q_heads | **16** | `num_attention_heads` |
| kv_heads | **4** | `num_key_value_heads` |
| head_dim | **256** | `head_dim` |
| **rotary_dim** | **64** | `head_dim × partial_rotary_factor` = 256 × 0.25 |
| rope_theta | **1e7** | `rope_parameters.rope_theta` |
| partial_rotary_factor | **0.25** | `rope_parameters.partial_rotary_factor` |
| actual_rows | **1** | decode（`RowBucket::R16`, `ExecutionClass::DECODE`） |
| page_tokens | **16** | `paged_attention_selector` の全 rule が 16 を要求 |
| max_visible_tokens | 4096（既定、env `PHASESHIFT_ATTN_MAX_VISIBLE` で変更可） | |
| KV dtype | **BF16**（`kv_dtype=0`） | inventory が作る pool の実測 |
| output dtype of attention | **F32**（`out_dtype=1`） | `attn_core` の出力 |
| attention layer 数 | **8** | `layer_types` 中の `full_attention` 個数 |
| intermediate_size | 9216 | |
| num_hidden_layers | 32 | |

`rotary_dim=64` は **head_dim 256 の4分の1**であり、RoPE は rotary 64 桁だけを回す。

---

## 4. Gate A0.1 — dispatch inventory（L3 全21件）

列: `index / debug_name / KernelId / in / out / weight / MCU / physical resolver`。
`in`・`out` は `dtype/F(feature)/S(row_stride)`、dtype は `0=BF16 1=F32`。

| idx | debug_name | KernelId | in | out | weight | MCU | physical resolver |
|---:|---|---|---|---|---:|---|---|
| 67 | `L3.input_norm` | RMS_NORM | BF16/F2560 | BF16/F2560 | — | **yes** | `Bf16PfOnePlus` wlayout=1 mode=0 F=2560 G=2560 grid=1 wg=256 |
| 68 | `L3.q_proj` | LINEAR_BF16 | BF16/F2560 | BF16/F8192 | 25 | **yes** | `ExactRows` rows=1 out=8192 k=2560 grid=1024,1 wg=256 |
| 69 | `L3.k_proj` | LINEAR_BF16 | BF16/F2560 | BF16/F1024 | 26 | **yes** | `ExactRows` rows=1 out=1024 k=2560 grid=128,1 wg=256 |
| 70 | `L3.v_proj` | LINEAR_BF16 | BF16/F2560 | BF16/F1024 | 27 | **yes** | `ExactRows` rows=1 out=1024 k=2560 grid=128,1 wg=256 |
| 71 | `L3.q_gate_split` | **SPLIT** | BF16/F8192 | BF16/F4096 ×2 | — | **no** | `SplitBf16InterleavedHeads` rows=1 F=4096 aux(head_dim)=256 grid=4,1 wg=256 |
| 72 | `L3.q_norm` | RMS_NORM | BF16/F4096 | **F32**/F4096 | — | **no** | `variant=Unsupported` wlayout=**2(PER_GROUP)** mode=**0(ONE_PLUS)** F=4096 G=256 grid=0 wg=0 |
| 73 | `L3.q_rope` | **ROPE** | F32/F4096 | BF16/F4096 | — | **no** | physical resolver **未分離**（A4） |
| 74 | `L3.k_norm` | RMS_NORM | BF16/F1024 | **F32**/F1024 | — | **no** | `variant=Unsupported` wlayout=2 mode=0 F=1024 G=256 grid=0 wg=0 |
| 75 | `L3.k_rope` | **ROPE** | F32/F1024 | BF16/F1024 | — | **no** | physical resolver **未分離**（A4） |
| 76 | `L3.kv_append` | **KV_APPEND** | BF16/F1024 ×2 | （stateへ） | — | **no** | resolve=ok kv_dtype=BF16 **impl=Optimized** layer=0 rows=1 kv_heads=4 head_dim=256 page_tokens=16 |
| 77 | `L3.attn_core` | **PAGED_ATTENTION** | BF16/F4096 | F32/F4096 | — | **no** | **optimized=1 use_prefill=0 splits=16** q_heads=16 kv_heads=4 head_dim=256 page_tokens=16 layer=0 |
| 78 | `L3.attn_gate` | **SIGMOID** | BF16/F4096 | F32/F4096 | — | **no** | `SigmoidBf16ToF32` rows=1 F=4096 grid=4,1 wg=256 |
| 79 | `L3.attn_gated` | MUL | F32/F4096 ×2 | BF16/F4096 | — | **yes** | `MulF32F32Bf16` rows=1 F=4096 grid=4,1 wg=256 |
| 80 | `L3.o_proj` | LINEAR_BF16 | BF16/F4096 | BF16/F2560 | 28 | **yes** | `ExactRows` rows=1 out=2560 k=4096 grid=320,1 wg=256 |
| 81 | `L3.residual_attn` | RESIDUAL_ADD | BF16/F2560 ×2 | BF16/F2560 | — | **yes** | `ResidualBf16` rows=1 F=2560 grid=3,1 wg=256 |
| 82 | `L3.post_norm` | RMS_NORM | BF16/F2560 | BF16/F2560 | — | **yes** | `Bf16PfOnePlus` wlayout=1 mode=0 F=2560 G=2560 grid=1 wg=256 |
| 83 | `L3.mlp_gate` | LINEAR_BF16 | BF16/F2560 | BF16/F9216 | 29 | **yes** | `ExactRows` rows=1 out=9216 k=2560 grid=1152,1 wg=256 |
| 84 | `L3.mlp_up` | LINEAR_BF16 | BF16/F2560 | BF16/F9216 | 30 | **yes** | `ExactRows` rows=1 out=9216 k=2560 grid=1152,1 wg=256 |
| 85 | `L3.mlp_swiglu` | SWIGLU | BF16/F9216 ×2 | BF16/F9216 | — | **yes** | `SwigluBf16` rows=1 F=9216 grid=9,1 wg=256 |
| 86 | `L3.mlp_down` | LINEAR_BF16 | BF16/F9216 | BF16/F2560 | 31 | **yes** | `ExactRows` rows=1 out=2560 k=9216 grid=320,1 wg=256 |
| 87 | `L3.output` | RESIDUAL_ADD | BF16/F2560 ×2 | BF16/F2560 | — | **yes** | `ResidualBf16` rows=1 F=2560 grid=3,1 wg=256 |

```
range mcu_supported=13 mcu_unsupported=8 mcu_error=0
attention plan: compile failed: unsupported: SPLIT   ← 全range compile は先頭の未対応で停止
```

### 未対応 8 件の理由分類

`compile_mcu_plan` は理由を明示して fail-closed する。2種類に分かれる。

| 区分 | dispatch | 理由 |
|---|---|---|
| **compiler に case が無い** | `q_gate_split`(71), `attn_gate`(78) | `mcu_plan_compiler.hip:518` の default → `unsupported: SPLIT` / `unsupported: SIGMOID`。production resolver は既に `Applicable` |
| | `q_rope`(73), `k_rope`(75) | `unsupported: ROPE`。physical resolver 自体が無い（resolve と launch が同居） |
| | `kv_append`(76) | `unsupported: KV_APPEND`。resolve はあるが physical launch に未整理 |
| | `attn_core`(77) | `unsupported: PAGED_ATTENTION`。resolve/plan はあるが physical plan が無い |
| **stable entrypoint が無い** | `q_norm`(72), `k_norm`(74) | `mcu_plan_compiler.hip:232` → `rmsnorm physical variant has no mcu entrypoint`。`resolve=Applicable` だが `variant=Unsupported` |

---

## 5. Gate A0.3 — Paged Attention plan

`resolve_paged_attention_args()` + `plan_paged_attention()` の実測。

```
optimized=1  use_prefill=0  splits=16  kv_dtype=BF16  out_dtype=F32
rows=1  q_heads=16  kv_heads=4  head_dim=256  page_tokens=16  layer=0
```

### splits の依存関係（sweep 実測）

`max_visible_tokens` を変えた実測:

| max_visible_tokens | optimized | use_prefill | **splits** |
|---:|---:|---:|---:|
| 1 | 1 | 0 | **1** |
| 1024 | 1 | 0 | **1** |
| 2047 | 1 | 0 | **1** |
| **2048** | 1 | 0 | **16** |
| 4096 | 1 | 0 | **16** |
| 32768 | 1 | 0 | **16** |

**境界は `kPagedAttentionSplitMinVisible = 2048`。**
`max_visible_tokens >= 2048` で **splits=16**、それ未満で **splits=1**。
条件は `kv_dtype == BF16 && !use_prefill && ctx.decode_attn_partials != nullptr`
`&& max_visible >= 2048 && q_heads % kv_heads == 0 && need <= decode_attn_partial_bytes`。
`need = rows × kv_heads × 16 × q_per_kv × (2 + 256) × 4 = 66048` bytes（rows=1）で、
`kDecodeAttnPartialBytes = 8MB` のため常に収まる。

`use_prefill` は `rows=1` では `rows < kPrefillKernelMinRows` のため常に 0。

### Gate への影響（Gate A6.2）

- **logical dispatch = 1**（`PAGED_ATTENTION` は Program 上1つ）
- **physical dispatch = 2**（`paged_attention_split` → `paged_attention_split_reduce`）
  - これは `max_visible >= 2048` のとき
  - `max_visible < 2048` なら physical = 1

したがって **「1 Program dispatch = 1 McuPlanNode」という前提は維持できない**。
`compile_mcu_plan()` は 1 logical dispatch から複数 node を出せる構造にしなければならない（A6.3）。

---

## 6. RMSNorm q_norm / k_norm の実態（A3 の根拠）

`resolve_rmsnorm_physical` は **2つの独立した判定**をしている。

1. **`select_rmsnorm_implementation`（selector）** — optimized かどうか
2. **`out.variant`** — `extern "C"` stable entrypoint が存在するか

実測では selector は q_norm / k_norm に **Optimized ルールを一致させている**
（`rmsnorm_selector.cpp` に `{BF16, F32, BF16, PerGroup, mode=0, F=4096, G=256}` と
`{BF16, F32, BF16, PerGroup, mode=0, F=1024, G=256}` がある）。
よって `resolve` は **`Applicable`(0)** を返す。

一方 `variant` の判定は

```cpp
bf16_pf_one_plus = in==BF16 && dst==BF16 && wdt==BF16 && layout==PER_FEATURE && mode==ONE_PLUS
f32_pg_direct    = in==F32  && dst==F32  && wdt==BF16 && layout==PER_GROUP  && mode==DIRECT
```

の2通りしか無く、q_norm / k_norm の組合せ
**`BF16 → F32 / weight BF16 / PER_GROUP / ONE_PLUS`** はどちらにも該当しない → `variant=Unsupported`。

**production が実際に何を実行しているか:**

`try_launch_rmsnorm`（`optimized_dispatch.hip`）は **`variant` を一切参照せず**
`ps::kernel::launch_rmsnorm(...)` を呼ぶ。`launch_rmsnorm` は dtype/layout/mode で
template を展開し、この組合せは

```cpp
launch_one<BF16, F32, BF16, PER_GROUP, ONE_PLUS> → rmsnorm_impl<...>
```

を起動する。

つまり:

- **production は optimized physical path を実行している**（correctness fallback ではない）
- **`variant` フィールドは「optimized の可否」ではなく「stable entrypoint の有無」を表す**
- **`extern "C"` の kernel body だけが欠けている**

よって A3 でやるべきことは、**production が選ぶ exact 組合せ
`BF16→F32 / PER_GROUP / ONE_PLUS` の stable entrypoint を1つ追加する**ことで、
`q_norm`(F=4096,G=256) と `k_norm`(F=1024,G=256) は**同じ entrypoint**で表現できる
（F / G は args の中身）。

`grid` / `workgroup` は既存の
`rmsnorm_bf16_pf_oneplus_grid(rows, F, G) = rows × (F/G)` /
`rmsnorm_bf16_pf_oneplus_block_size(G)` が一般に使える:
- q_norm: grid = 1 × 16 = 16、block = chunks(64) → **64**
- k_norm: grid = 1 × 4 = 4、block = **64**

---

## 7. Gate 割当（A0 で確定した残作業）

| Gate | 対象 dispatch | 実測値（A0 から） |
|---|---|---|
| A1 SPLIT | `q_gate_split` (71) | `SplitBf16InterleavedHeads`, F=4096, aux=256, grid=4,1, wg=256 |
| A2 SIGMOID | `attn_gate` (78) | `SigmoidBf16ToF32`, F=4096, grid=4,1, wg=256 |
| A3 Q/K RMSNorm | `q_norm` (72), `k_norm` (74) | BF16→F32 / PER_GROUP / ONE_PLUS、F=4096・1024 / G=256 |
| A4 RoPE | `q_rope` (73), `k_rope` (75) | in=F32 → out=BF16、F=4096・1024、head_dim=256、rotary=64、theta=1e7 |
| A5 KV_APPEND | `kv_append` (76) | kv_dtype=**BF16**、impl=**Optimized**、layer=0、rows=1、page_tokens=16 |
| A6 PAGED_ATTENTION | `attn_core` (77) | optimized=1、use_prefill=0、**splits=16**（mv≥2048）→ **logical 1 → physical 2** |
| A7 compiler coverage | 上記6種 | 全 range compile は現状 `unsupported: SPLIT` で停止 |

---

## 8. A1 SPLIT / A2 SIGMOID

**方針**: production の `resolve_elementwise_physical` に SPLIT / SIGMOID の判定が
既にあり、selector も `Applicable` を返す。欠けていたのは **`extern "C"` stable entrypoint
とコンパイラ case** だけなので、selector を新設せず既存 body を共有した。

`elementwise.hip` の変更:

- `op_sigmoid_bf16_f32::impl` / `op_split_bf16::impl` の body を
  `__device__` 関数へ抽出し、stable entrypoint
  `phaseshift_qwen35_sigmoid_bf16_f32` / `phaseshift_qwen35_split_bf16` を追加。
  **body はコピーせず 1 個に統合**した。
- `launch_sigmoid_bf16_f32` / `launch_split_bf16` は同じ stable kernel を
  launch するよう変更 → Host と MCU が同一 kernel を通る。
- SPLIT の layout は `ElementwiseArgs.reserved`（0=Halves / 1=InterleavedHeads）、
  `head_dim` は `aux` に載せる。`McuElementwiseInvocation` は
  `reserved` を含め struct ごとコピーされるため追加構造体は不要。

| 項目 | q_gate_split（実測） | attn_gate（実測） |
|---|---|---|
| profile | `SplitBf16InterleavedHeads` | `SigmoidBf16ToF32` |
| stable symbol | `phaseshift_qwen35_split_bf16` | `phaseshift_qwen35_sigmoid_bf16_f32` |
| args size | 88 (`ElementwiseArgs`) | 88 |
| kernarg segment | 344 | 344 |
| hidden args | **fit する**（両側で書き込み） | fit する |
| grid / workgroup | 4,1 / 256 | 4,1 / 256 |
| rows / features | 1 / 4096（aux=256） | 1 / 4096 |

検証: `test_gpu_mcu_real_split`（InterleavedHeads と Halves の 2 case を
Host production / Host AQL / MCU の 3 経路で比較し、**q 出力と gate 出力の両方 byte-exact**）。
SIGMOID は elementwise の 1 profile なので `test_gpu_mcu_real_elementwise` に
case を追加して同じ 3 経路比較を行った（新規 test を増やさない判断）。

---

## 9. A3 q_norm / k_norm

§6 のとおり、**production は既に optimized physical path を実行している**
（`try_launch_rmsnorm` は `variant` を見ず `launch_rmsnorm` を呼ぶ）。
欠けていたのは `BF16→F32 / PER_GROUP / ONE_PLUS` の stable entrypoint だけ。

- `rmsnorm.hip` に `phaseshift_qwen35_rmsnorm_bf16_f32_pg_oneplus` を追加。
  body は既存の `rmsnorm_block_body<BF16,F32,BF16,PER_GROUP,ONE_PLUS>` を
  そのまま呼ぶ（`rmsnorm_impl` は同関数の `__global__` wrapper なので同一計算）。
- `launch_one` の分岐に該当組合せを追加 → `launch_rmsnorm` が stable kernel を通る。
- `PhysicalRmsNormVariant::Bf16F32PgOnePlus` を追加し、
  `resolve_rmsnorm_physical` の variant 判定に第3の条件を追加
  （`grid` / `workgroup` は既存の `rmsnorm_bf16_pf_oneplus_grid/block_size` を再利用）。

| 項目 | q_norm（実測） | k_norm（実測） |
|---|---|---|
| in / out / weight | BF16 / **F32** / BF16 | 同左 |
| weight layout / mode | PER_GROUP(2) / ONE_PLUS(0) | 同左 |
| features / group | 4096 / 256 | 1024 / 256 |
| stable symbol | `phaseshift_qwen35_rmsnorm_bf16_f32_pg_oneplus` | 同左（F/G は args） |
| grid / workgroup | **16 / 64** | **4 / 64** |

検証: `test_gpu_mcu_real_attention_rmsnorm`（2 case、3 経路 byte-exact）。

---

## 10. A4 RoPE

`rope_dispatch.hip` は resolve と launch が同居していたため、他 resolver と同じ形へ分離した。

- `PhysicalRopeVariant` / `PhysicalRopeLaunch` / `resolve_rope_physical` /
  `launch_rope_physical` を追加。`try_launch_rope` は resolve → launch に。
- variant 判定 `pair` は `rope_pair_shape_supported()` と `ctx.rope_inv_freq != nullptr`
  の production 判定を**そのまま**使う（MCU 側でコピーしない）。
- `kRopePairBlock=128 / kRopePairHeadDim=256 / kRopePairRotary=64` を
  header へ公開し、`.hip` 側は `static_assert` で drift を防ぐ。
- stable entrypoint は **pair のみ**（`phaseshift_qwen35_rope_f32_bf16_pair`）。
  generic は production のまま動くが MCU は fail-closed（理由を明示）。

| 項目 | q_rope（実測） | k_rope（実測） |
|---|---|---|
| variant | **F32Bf16Pair** | F32Bf16Pair |
| in → out | F32 → BF16 | F32 → BF16 |
| F / head_dim / rotary_dim | 4096 / 256 / 64 | 1024 / 256 / 64 |
| theta | 1e7 | 1e7 |
| grid / workgroup | 1 / 128 | 1 / 128 |
| args size | 56 | 56 |
| kernarg segment | **56（= sizeof(args) ちょうど）** | 56 |
| hidden args | **fit しない → IfFits で Host AQL / MCU とも skip** | 同左 |

`kPairHeadDim=256 / kPairRotary=64` は実モデルの attention と完全一致するため、
実モデルは常に pair 経路を通る。

検証: `test_gpu_mcu_real_rope`（rows=1 position 0、rows=3 positions {0,31,1024} の 3 case、
3 経路 byte-exact）。position > 0 を含む。

---

## 11. A5 KV_APPEND

`resolve_kv_append_args` は残し、その上に physical 層を足した。

- `PhysicalKvAppendVariant` / `PhysicalKvAppendLaunch` / `resolve_kv_append_physical` /
  `launch_kv_append_physical` を追加。`try_launch_kv_append` は resolve → launch に。
- stable entrypoint は **BF16 のみ**（`phaseshift_qwen35_kv_append_bf16`）。
  FP8 / PSQ2 / PSQ4 は production では動くが MCU は fail-closed。
  （PSQ2 はその後リポジトリから削除。FP8 / PSQ4 は現行。）
- `resolve_kv_append_args` / `resolve_paged_attention_args` の `ctx` を
  **`const` に統一**（いずれも読み取りのみで、compiler から const ctx で呼ぶため）。

| 項目 | 実測 |
|---|---|
| kv_dtype | **BF16**（`impl=Optimized`） |
| layer / rows | 0 / 1 |
| kv_heads / head_dim / page_tokens | 4 / 256 / 16 |
| grid / workgroup | **4 / 64** |
| args size / kernarg | 112 / **112（= sizeof ちょうど）** |
| hidden args | fit しない → 両側 skip |

検証: `test_gpu_mcu_real_kv_append`（positions **0 / 15 / 16** で page 境界を跨ぐ、
3 経路、**K state と V state の両 pool を byte-exact 比較**）。

### 発見した不具合（test 側）

- `hipMemset`（null stream）と **AQL hardware queue** の完了順序は保証されない。
  pool を zero した直後に packet を submit すると、初期の packet の書き込みが
  後から zero で消えることがある（症状: 位置 0 の V だけ 0）。zero の後に
  `hipDeviceSynchronize()` を入れて解決した。以後の test ではこの順序を守る。

---

## 12. A6 PAGED_ATTENTION

- `PhysicalPagedAttentionVariant` / `PhysicalPagedAttentionKernel`（1 physical kernel 分）/
  `PhysicalPagedAttentionPlan`（`physical_count` + `kernels[2]`）を追加。
- `resolve_paged_attention_physical` は `resolve_paged_attention_args` +
  `plan_paged_attention` + `should_use_paged_attention_prefill` をそのまま使い、
  **MCU 用 selector を新設しない**。
- stable entrypoint は **BF16 full-head のみ**（`phaseshift_qwen35_attention_paged_bf16`）。
  partial-head / FP8 / PSQ2 / PSQ4 / prefill は fail-closed。（PSQ2 は削除済み。）

### logical → physical expansion（A6.2 / A6.3 / A6.4）

`McuCompiledPlan` に `logical_dispatch_count` と `physical_dispatch_count` を追加し、
- `dispatch_count` は従来どおり logical（互換）
- `kernarg_slots_required = physical_dispatch_count + 1`
- `compile_mcu_plan()` は `push_dispatch()` 経由で node を積み、
  **1 logical dispatch から複数 node を出せる構造**にした

split 経路では `physical_count=2`（split + reduce）を plan が返す。
**本 Gate では `splits > 1` を fail-closed** とし、拒否理由を
`paged attention split path is not supported on the mcu` と明示する（A6.5 の選択 B）。
理由: split/reduce は `(q_per_kv ∈ {4,6}) × (full_head ∈ {T,F}) × (qh_per_wg ∈ {1,2,3,4,6})`
の組合せで kernel instantiation が分かれ、reduce 側の経路は
`attention_split_reduce_qh_per_wg()`（env 依存）で実行時に決まる。
**compiler 側の multi-node 構造は実装済み**なので、次 Gate は variant を足すだけで済む。

### splits の確定（A0.3 の再掲）

`kPagedAttentionSplitMinVisible = 2048`。
`ctx.max_visible_tokens >= 2048` で splits=16、未満で splits=1。

| max_visible | optimized | use_prefill | splits | physical |
|---:|---:|---:|---:|---:|
| 1 / 1024 / 2047 | 1 | 0 | 1 | 1 |
| 2048 / 4096 / 32768 | 1 | 0 | **16** | **2** |

本 Gate の E2E は decode 条件として `max_visible_tokens = 512` を使う
（= production が splits==1 を選ぶ条件）。

検証: `test_gpu_mcu_real_paged_attention`
（`physical_count=1 splits=1 grid=16 wg=256`、position=100、3 経路 byte-exact）。

---

## 13. A7 compiler coverage

`compile_mcu_plan()` に SPLIT / SIGMOID / ROPE / KV_APPEND / PAGED_ATTENTION を追加し、
RMS_NORM に `Bf16F32PgOnePlus` を追加した。

| 条件 | supported |
|---|---|
| `max_visible_tokens = 4096`（既定） | **20 / 21**（`attn_core` のみ splits=16 で fail-closed） |
| `max_visible_tokens = 512`（decode 条件） | **21 / 21**（unsupported 0） |

`test_gpu_mcu_attention_layer_inventory` が両方を実測して表示する。

---

## 14. A8 / A9 / A10 / A11 — Attention One Layer E2E

test: `test_gpu_mcu_attention_one_layer_plan`

```
attention layer=L3.
L3. range=[67,88) dispatches=21
plan: nodes=24 variants=16 kernarg_slots=22 logical=21 physical=21 invocations=21
layer F=2560 dtype=0 out_bytes=5120
traces=20
host L3.: launched 21 dispatches (21 physical launches)
mcu: iterations=100 dispatches=2200 fault=0 physical=21 elapsed=82.818 ms (0.8282 ms/iter) mismatch_iters=0
intermediates compared=20
```

- **A9**: Host reference は全 21 dispatch が `OptimizedLaunchKind::Launched`。
  correctness fallback は **0 件**。
- **A10**: layer 内部に WAIT を入れず、末尾の marker + WAIT + END だけ。
  `depth=4` / per-packet doorbell を維持。
- **A11**: range 内の**出力を持つ全 20 dispatch** の出力を Host と MCU で比較し
  **すべて byte-exact**（`input_norm` / `q_proj` / `k_proj` / `v_proj` /
  `q_gate_split` / `q_norm` / `q_rope` / `k_norm` / `k_rope` / `attn_core` /
  `attn_gate` / `attn_gated` / `o_proj` / `residual_attn` / `post_norm` /
  `mlp_gate` / `mlp_up` / `mlp_swiglu` / `mlp_down` / `output`）。
- **A11.1**: KV cache は K pool / V pool の**全領域**を byte-exact 比較。
- **A8.1**: Host と MCU は毎回同じ初期状態（K/V pool を同一内容で再充填）から開始。

---

## 15. A12 反復

既定 100 反復（`PHASESHIFT_ATTN_ITERS` で変更可）。
実測 `0.83 ms/iter`、`mismatch_iters=0`、`fault=0`。
反復ごとに K/V pool と入力を同一内容へ戻す。

---

## 16. A13 queue 挙動

`GpuMcuFsm` の counter を E2E test（100 反復）で取得した実測:

```
queue: depth=4 doorbell=per_packet doorbell_count=2200 refill=2195 empty=100
       backpressure=305 append_ahead=2100 max_ahead=4 wrap=2
```

- `doorbell_count = 2200 = 22 packet × 100 反復` → **per-packet doorbell** が全 packet で機能
- `max_ahead = 4` → `queue_ahead_depth = 4` を遵守（超過なし）
- `wrap = 2` → ring の wrap も発生（ring が小さいことの確認）
- `empty = 100` → **1 反復あたり 1 回**。plan 末尾の marker + WAIT 後の drain に対応し、
  **layer 内部での starvation ではない**（layer 内部で詰まれば 1 反復に複数回出る）
- `backpressure = 305` → ahead 上限に当たった回数（depth=4 の設計どおり）

---

## 17. A14 resources

`tools/extract_kernel_resources.py` で実測。

| kernel | sgpr | sgpr spill | vgpr | vgpr spill | scratch | LDS | kernarg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `phaseshift_qwen35_split_bf16` | 21 | **0** | 15 | **0** | 0 | 0 | 344 |
| `phaseshift_qwen35_sigmoid_bf16_f32` | 15 | **0** | 26 | **0** | 0 | 0 | 344 |
| `phaseshift_qwen35_rmsnorm_bf16_f32_pg_oneplus` | 30 | **0** | 21 | **0** | 0 | 32 | 304 |
| `phaseshift_qwen35_rope_f32_bf16_pair` | 31 | **0** | 34 | **0** | 0 | 0 | 56 |
| `phaseshift_qwen35_kv_append_bf16` | 32 | **0** | 5 | **0** | 0 | 16 | 112 |
| `phaseshift_qwen35_attention_paged_bf16` | 42 | **0** | 61 | **0** | 0 | 8280 | 128 |
| persistent supervisor `gpu_mcu_fsm_kernel` | 106 | **0** | 191 | **12** | 52 | 0 | 264 |

**新規・変更 kernel はすべて SGPR/VGPR spill = 0。**

### A14.1 persistent supervisor の spill は本 Gate の退行ではない

`gpu_mcu_fsm_kernel` に VGPR spill が出ているため、**Gate 開始時点の commit
（`626db159`）の `micro_fsm.hip` を同じ compile command で単体ビルドして比較**した:

| | sgpr | sgpr spill | vgpr | vgpr spill | scratch | code |
|---|---:|---:|---:|---:|---:|---:|
| **baseline（626db159）** | 101 | **0** | 191 | **12** | 52 | 60240 |
| 本 Gate 後 | 105 | **0** | 191 | **12** | 52 | 68320 |
| 増分 | +4 | ±0 | **±0** | **±0** | ±0 | +8080 |

**VGPR spill = 12 は Gate 開始時点で既に存在**し、本 Gate の変更では
**VGPR も spill も一切変化していない**（SGPR +4 と code +8080 のみ）。
由来は前 Gate（retained packet / per-packet doorbell）と推定される。

`__noinline__` を新規 kernarg writer 3 個に適用する局所対処も試したが
spill は変化しなかったため採用せず、他の writer と同じ `__forceinline__` に戻した。
§A14.1 の指示どおり大規模リファクタは行わず、**次 Gate の改善項目**として記録する。

その後 Full Transformer Serial Static Region Gate で原因を特定し解消した
（`mcu_build_kernarg` を control loop から out-of-line 化。詳細は
[full_transformer_static_region.md](full_transformer_static_region.md) §3）。
現在の persistent supervisor は SGPR spill 0 / VGPR spill 0。

---

## 18. A15 Transformer body 全体 compile

`test_gpu_mcu_attention_layer_inventory` にて prefix `L`（全 transformer body）を
1 個の plan として compile-only した実測:

```
body compile: logical=696 physical=696 nodes=699 variants=25 kernarg_slots=697 invocations=696
body compile: ok variants_vs_limit=25/64 nodes_vs_limit=699/16384
```

- **unsupported = 0**（embedding と head を除く body 全体）
- variants **25 / 64**、nodes **699 / 16384**、kernarg slots 697 — いずれも容量内

---

## 19. A16 次の Gate

**Case A**: Attention One Layer PASS、Full Transformer compile が
unsupported = 0 かつ容量内。次 Gate は
**Full Transformer Serial Static Region**（embedding output → L0..L31 → Final Marker → WAIT、
layer 境界に Marker/WAIT を置かない）。

繰り越し事項:
1. paged attention の **split/reduce** 物理 kernel 対応（compiler 構造は実装済み）
2. embedding / LM head の production physical path の coverage gap
3. prefill（`LINEAR_BF16` Wmma 系）
4. RMSNorm の attention 以外の未対応組合せ（`q_norm` 用は解消済み）
5. parallel issue / Gate-Up parallel / Barrier AND-OR / EXT_KERNEL_DISPATCH
