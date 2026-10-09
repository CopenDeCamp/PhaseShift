# Next Task: TP=2 Host Dispatch の改善候補

2026-10-09 時点。`docs/now_task.md` の後継。Attribution（Step 1）は完了済みで、
ここに**効果の大きい順の改善候補**を残す。Step 2（実装）は未着手。

> lifecycle が定義された `docs/user/` `docs/developer/` `docs/perf/` `docs/rnd/`
> `docs/references/` には属さない、作業中のメモである。
> 数値はいずれも**1回計測・未 qualified**。`docs/perf/current.md` の正本を代替しない。

## 計測条件

| 項目 | 値 |
| --- | --- |
| revision | `20d5b199`（`exp/gpu-mcu`） |
| build | `build-perf` / `RelWithDebInfo` / `-O2 -g -fno-omit-frame-pointer` / gfx1201 |
| 対象 | `build-perf/tests/test_qwen35_tp_e2e`（Qwen3.8-27B-PSQ、prompt 5 token + greedy 16 token） |
| profiler | `perf` 7.0.14、`-e cycles:u -F 4999 --call-graph fp` |
| worker | tid 25453 / 25454（活動 span 0.505秒 = decode 窓） |
| worker 総 cycles | rank0 581,488,604 / rank1 585,738,077 |

分解能は `perf script` の **period 合計**で取得（`perf report` の self% は 0.01% 刻みの
丸めで worker 内の内訳が使えなかった）。

---

## 改善候補 1: `resolve_psq4_physical` のバリデーションを program build 時へ移動

**効果 46〜49% / リスク 低 / rank 内 1 位**

| rank | leaf（inlined 8 複製の合計） | 内部分 `find_value` | 総体 |
| --- | ---: | ---: | ---: |
| 0 | 42.46% | 3.88% | **46.34%** |
| 1 | 43.99% | 5.17% | **49.16%** |

### 呼び出し経路

```
execute_program_range
  ← execute_program_tp_ranges        (TP 経由であることを確認)
    ← try_launch_optimized
      ← try_launch_linear_psq4       (optimized_dispatch.hip:216 が呼び出し)
        ← resolve_psq4_physical      (physical_launch.hip:212-274)
```

### 内容

`resolve_psq4_physical` は **約20個のバリデーション if を毎 dispatch 実行**している:

- io count / compute_spec / weight_index / workspace_slot
- encoding / codes / scales / rows / cols / k_padded
- weight_scale_group / storage_scale_stride / activation_kp
- activation_code_stride / activation_scale_stride / activation_max_rows
- workspace / `Int8ActivationWorkspaceLayout::make` と3項目の一致

これらは **`DispatchBinding` と `WeightSlot` にしか依存しない**。両者は program
作成後は不変なので、program build 時に1回検証すれば済む。

### 手順

- 既存の検証パス `src/phaseshift/runtime/program/program.cpp:770-774`
  （`find_value` による slot 検証）に統合するのが自然
- `resolve_psq4_physical` 側の if を外し、到達不能になることを build 検証で担保

### リスクと検証

- **リスク: 低**。検証は移すだけで、実行時の解決結果は変わらない
- 検証: `test_qwen35_tp_e2e` の `PASS`（greedy 16 token の TP1/TP2 一致、
  `tp1 gate_rows=17408 tp2 gate_rows=8704`、rank0/rank1 の final hidden bit-identical）
- **効果測定は Release ビルドで**（`build-perf` は RelWithDebInfo で最適化設定が異なる）

---

## 改善候補 2: `Program::find_value` を O(1) 化

**効果 17〜20% / リスク 低〜中 / rank 内 2 位**

| rank | cycles | rank 内% |
| --- | ---: | ---: |
| 0 | 99,871,860 | **17.18%** |
| 1 | 120,213,629 | **20.52%** |

### 呼び出し経路

```
resolve_host_value              (program_executor.hip:205)
  ← program.find_value(id)      ← O(n) 線形探索
```

### 内容

```cpp
// include/phaseshift/runtime/program/program.h:147
const ValueBinding* find_value(ValueId v) const noexcept {
    for (const auto& b : values) {
        if (b.logical_value.id == v.id) return &b;
    }
    return nullptr;
}
```

**線形探索が全 dispatch 種別にわたって発生**している。`Program::find_value` 総量の
caller 内訳（rank0）:

| find_value 内% | 経路 |
| ---: | --- |
| 22.6% | `resolve_psq4_physical` |
| 11.3% | `resolve_bf16_physical` |
| 8.4% | `resolve_binary_bf16` → `resolve_elementwise_physical` |
| 7.6% | `resolve_rmsnorm_physical` |
| 7.5% | `resolve_gdn_recurrence_args` → `resolve_gdn_recurrence_physical` |
| 6.9% | `resolve_silu_desc` → `resolve_elementwise_physical` |
| 5.5% | `resolve_activation_quantize_physical` |

つまり候補2 単体の 17〜20% に加え、**候補1 の中にも同種のコストが入っている**。
両者は独立ではなく重なる。

### 手順

- `ValueId` → `values` index の対応表を program build 時に作る
- **`values` の並び順は変えない**（lookup 結果が不変であることの担保）
- 密な `ValueId` なら直接インデックス、そうでなければ sorted + binary search か hash

### リスクと検証

- **リスク: 低〜中**。lookup 結果（`ValueBinding*`）が不変なら呼び出し側は無変更
- 同上 `test_qwen35_tp_e2e` で正しさ確認

---

## 対象外: kernel launch 数の削減（fusion）

perf で `hipLaunchKernel` が decode 窓の 26.5%（119.8 ms / 452 ms、1件あたり 3.9 µs ×
30,548回）と大きいが、**リスク高**のため候補から外した。

- kernel 開発が必要で、AGENTS.md の benchmark ルールが全ての変更にかかる
- HIP graph は `tensor_parallel_execution.md` §10 の通り **TP schedule と併用不可**
- host 側（候補1・2）を先に削り、launch が支配的になってから検討するのが順当

---

## 計測の注意点（次回も有効）

1. **`try_launch_optimized` の 5.05 µs/件 には HIP launch API が含まれる**
   （手順書の警告どおり）。候補1・2 は host 計算側、launch 側は別物として評価する。
2. **`perf` の self% は 0.01% 刻みの丸め**。worker は全体の 0.31% しか占めないので
   `perf report` では内訳が出ない。`perf script` の period 合計で絶対値を取る。
3. **`perf_event_paranoid=4`** で一般ユーザーは `perf` 不可。sudoers 追加後、
   `sudo sysctl -w kernel.perf_event_paranoid=2` で user-space 計測が可能。
   永続化は不要（`-w` のみ）。
4. **模型ロードの CRC 検証が全体の 88%** を占める。`verify_quantized_payload_crc` は
   既定 false、`phaseshift-compute --verify-weights` 既定 0、ON はテストのみ。
   切ると report が読みやすくなるが、**decode 窓のサンプル数は増えない**
   （サンプリングは実時間ベース）。

## 再現手順と失敗学び

### 手順

```bash
# 計測専用ビルド
cmake -S . -B build-perf -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -fno-omit-frame-pointer" \
  -DCMAKE_HIP_FLAGS_RELWITHDEBINFO="-O2 -g -fno-omit-frame-pointer" \
  -DPHASESHIFT_ROCM_ROOT="$(rocm-sdk path --root)" \
  -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$(rocm-sdk path --root)" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake --build build-perf --target test_qwen35_tp_e2e --parallel

# 計測（--call-graph fp が必須）
PHASESHIFT_TP_MODEL_DIR=models/Qwen3.8-27B-PSQ \
  perf record -o perf.data -e cycles:u -F 4999 --call-graph fp -- \
  ./build-perf/tests/test_qwen35_tp_e2e
perf script -i perf.data > script.txt
```

worker の特定は「活動 span が 0.5秒前後」（main は112秒）。`tid` は毎回変化するので
**毎回実データから拾う**こと。

### 失敗学び

| 失敗 | 原因と対処 |
| --- | --- |
| `--call-graph dwarf,8192` で worker の leaf が0件 | **worker スレッドの DWARF unwind 失敗**。`fp` に変えれば解決（data も 4.3GB → 93MB） |
| `perf report --sort ...tid...` がエラー | report の sort key に `tid` は無い（`pid` を使う）。`perf script` の field としては `tid` が在る |
| `perf script -F comm,tid,time,period,sym` で `sym` が空・`tid` が main に潰れる | `-F` 指定時の perf 7.0.14 の挙動。デフォルトフォーマット（callchain 出力）を使う |
| 集約タイマーの出力が来ない | 名前空間スコープの `const bool = lambda()`（動的初期化）が HIP の `.hip` で機能しない。**function-local static**（`perf_stats_enabled()` パターン）にする |
| phase 名が逆に出た | `ExecutionRole::Decode = 0` なのに `kPhaseName[0] = "prefill"`。**enum の値と name 配列の対応を必ず確認** |
| 内訳合計と total が合わない | `continue` が3経路あるため、running mark（`t_mark`）方式で連続計測し、`post = total - A - B - C` で検算を取る |

---

## 現在の未コミット状態

`src/phaseshift/models/qwen35/runtime/program_executor.hip`（+183 / -12）に
**Step 4 の集約タイマーが入ったまま**。`PHASESHIFT_DISPATCH_PROFILE=<path>` で有効、
未設定時は既定 OFF（`dispatch_profile_enabled()` の function-local static で判定）。

出力 CSV: `rank,phase,section,count,total_us,avg_us`（section は
`pre_dispatch` / `try_launch_optimized` / `launch_host_binding` / `post_dispatch` /
`post_branch`〜`post_advance` の8区分 / `route_*` の3種）。

`docs/now_task.md` の §1 残課題（build target の食い違い、`required` label の穴3件、
acceptance の除外オプション無し、`testing.md` の記述乖離）は**未解決のまま**。

## 未完了タスク

- [ ] 改善候補1 の実装（`resolve_psq4_physical` のバリデーション移動）
- [ ] 改善候補2 の実装（`find_value` の O(1) 化）
- [ ] 両者を Release ビルドで効果測定（`docs/perf/methodology.md` に従う）
- [ ] 集約タイマーの扱い（revert するか、残すか）
- [ ] `docs/now_task.md` §1 の残課題5件
