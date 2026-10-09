# TP=2 Host Dispatch の Attribution と改善候補の採否

Qwen3.8-27B TP=2 の decode について、host 側 dispatch 構築がどれだけ e2e に効くかを
Attribution で絞り込み、改善候補 2 件を実装・実測して**両方とも不採用**とした記録である。

性能値の正本は [../perf/current.md](../perf/current.md) の「TP=2 decode」であり、
本ドキュメントの数値は比較判断の過程を示す R&D 記録である。

| 項目 | 値 |
| --- | --- |
| date | 2026-10-10 |
| model | `models/Qwen3.8-27B-PSQ` |
| GPU | AMD Radeon AI PRO R9700 (gfx1201) |
| 計測口 | `build/tests/test_qwen35_tp_e2e`（prompt 5 token + greedy 16 token） |
| baseline revision | `f83c8e1e`（`exp/gpu-mcu`） |

## 0. 結論

1. **改善候補1（`resolve_psq4_physical` のバリデーション移動）は不採用。**
   validation の if 群10箇所を全除去しても e2e は TP1 +0.05% / TP2 +0.27%。
   いずれも run 間 spread 内 = ノイズ。
2. **改善候補2（`Program::find_value` の O(1) 化）は不採用。**
   TP2 で −0.09%、TP1 で −0.06%。spread（TP2 2.55%）内で判定不能。
3. **`perf` の worker cycles 内占有率は e2e の改善余地を表さない。**
   host 時間は GPU busy と重なって潰れる。host 側の Attribution で見積もった
   「46〜49% / 17〜20%」は e2e では 0 だった。
4. **`rocprofv3` ありの attribution は律速を host 側へ逆転させる。**
   `hipLaunchKernel` 等の HIP API が tracer で膨張し、GPU idle が実より大きく見える。
5. 残る実質候補は **kernel launch 数の削減**のみ。ただし HIP graph は TP schedule と
   併用不可（[../developer/tensor_parallel_execution.md](../developer/tensor_parallel_execution.md) §10）で、
   kernel 開発が必要。host 側では潰せない。
6. `now_task.md` §4 が最有力とした **`ScopedDevice` の arch 検証キャッシュは打ち切り。**
   decode 寄与 0.6%（換算 2.4 ms）で判定閾値 9.5 ms に届かない。

## 1. 判定基準

baseline は [../perf/current.md](../perf/current.md) の 5-run p50。

| 指標 | TP=1 | TP=2 |
| --- | ---: | ---: |
| decode_ms p50 | 535.196 | 370.994 |
| decode tok/s p50 | 28.03 | 40.43 |
| run 間 spread | 0.25% | 2.55% |

TP2 の spread 2.55% を **判定閾値**とする。つまり e2e で **9.5 ms（= 371 ms × 2.55%）
未満の改善は測定で検出できない**。微小な改善を採用しても正しく判定できない。

## 2. Attribution 方法と限界

### 手法

`perf 7.0.14`、`-e cycles:u -F 4999 --call-graph fp`。対象は
`build-perf/tests/test_qwen35_tp_e2e`（RelWithDebInfo / `-O2 -g -fno-omit-frame-pointer`）。
worker は tid 25453 / 25454（活動 span 0.505 秒 = decode 窓）。
worker 総 cycles は rank0 581,488,604 / rank1 585,738,077。

分解能は `perf script` の **period 合計**で取る。`perf report` の self% は
0.01% 刻みの丸めで worker 内の内訳が出ない（worker は全体の 0.31%）。

### 見積もり値

| 候補 | rank0 | rank1 | 備考 |
| --- | ---: | ---: | --- |
| 1: `resolve_psq4_physical` のバリデーション | 46.34% | 49.16% | leaf 42.46 / 43.99% + 内部 `find_value` 3.88 / 5.17% |
| 2: `Program::find_value` 総量 | 17.18% | 20.52% | psq4 22.6% / bf16 11.3% / elementwise 8.4% / rmsnorm 7.6% / gdn 7.5% / silu 6.9% / act_quant 5.5% |

両者は重なる（候補1 の中に `find_value` が含まれる）。

### 限界（本記録の中心的教訓）

これらは **worker 総 cycles 内の占有率**であって、e2e の改善余地ではない。
GPU が動いている間に host が仕事をしているなら、その host 時間は wall に効かない。

次節の ceiling 実験がこれを実証する。

## 3. 改善候補と採否

### 候補1: `resolve_psq4_physical` のバリデーションを program build 時へ移動

`DispatchBinding` と `WeightSlot` は program 作成後不変なので、毎 dispatch 実行する
約20個の validation if を build 検証に統合できる。リスクは低い。

**ceiling 実験**: 実装の前に、validation 専用の文10箇所を `#if 0` で全除去して
上限を測った。機能（`WeightSlot e` / `wkp` / `sstride` / `layout` の宣言、
`resolve_host_value`、`rows_for`、selector）は残す。

| 指標 | baseline p50 | ceiling p50 | delta | spread |
| --- | ---: | ---: | ---: | ---: |
| TP1 decode_ms | 535.196 | 535.482 | **+0.05%** | 0.25 / 0.28% |
| TP2 decode_ms | 370.994 | 372.003 | **+0.27%** | 2.55 / 2.16% |
| TP1 tok/s | 28.030 | 28.010 | −0.07% | |
| TP2 tok/s | 40.430 | 40.320 | −0.27% | |

5/5 PASS（greedy 16 token の TP1/TP2 完全一致）。

**判定: 不採用。** 87% を消しても e2e は動かない。移動しても上限である。
測定後に `git checkout` で revert 済み（`#if 0` は production に残していない）。

### 候補2: `Program::find_value` を O(1) 化

`ValueId` → `values` index の対応表を program build 時に作り、線形探索を消す。
`values` の並び順は変えない（lookup 結果不変の担保）。

| 指標 | baseline | 変更後 | delta |
| --- | ---: | ---: | ---: |
| TP2 decode_ms | 371.171 | 370.824 | **−0.09%** |
| TP1 decode_ms | 535.942 | 535.609 | **−0.06%** |
| `find_value` cycles | 99,871,860 | 0 | −100% |
| worker 総 cycles | — | — | **−16.8%** |

**判定: 不採用。** worker 総 cycles が 16.8% 下がっても e2e は動かない。
変更は `/tmp/opencode/candidate2_value_index.patch` に退避済みで、ツリーには入れていない。

## 4. `rocprofv3` attribution の非転嫁

`rocprofv3 --kernel-trace --hip-runtime-trace` で測った decode 窓 452.0 ms では
GPU busy が 300.7 ms（66.5%）、GPU idle が 151.3 ms で、host 分解は次の通りだった。

| 区分 | 時間 | 率 | GPU idle 中 |
| --- | ---: | ---: | ---: |
| dispatch 構築（host 計算） | 178.3 ms | 39.4% | **46.5 ms** |
| `hipLaunchKernel` API 本体 | 119.8 ms | 26.5% | 推計 66〜74 ms |
| `complete_batch` 全 stream sync | 85.9 ms | 19.0% | 14.3 ms |
| barrier cv wait | 58.9 ms | 13.0% | 16.1 ms |
| その他 API | 8.4 ms | 1.9% | — |

この share を unprofiled の 371 ms に換算し、判定閾値 9.5 ms と比べる。

| 項目 | 換算 | 判定 |
| --- | ---: | --- |
| `hipLaunchKernel` の GPU idle | 57.5 ms | 検出可 |
| dispatch 構築の GPU idle | 38.2 ms | 検出可（**だが ceiling で 0**） |
| barrier cv wait idle | 13.2 ms | 検出可 |
| `complete_batch` sync idle | 11.7 ms | 検出可 |
| `ScopedDevice` arch 検証（decode） | 2.4 ms | **検出不可** |

「dispatch 構築の GPU idle 38.2 ms が検出可能」でありながら ceiling が 0 だった
ことは、**この表が unprofiled の実像ではない**ことを示す。tracer が `hipLaunchKernel`
等の host API を膨張させ、host が律速に見えるようになっている。

したがって **host 側の Attribution で改善余地を見積もってはいけない。**
上限は必ず `#if 0` 等で除去した ceiling 実験で取る。

## 5. 残る候補

| 候補 | 換算効果 | リスク | 判定 |
| --- | ---: | --- | --- |
| kernel launch 数の削減（fusion） | 57.5 ms | 高 | 唯一の実質候補。kernel 開発 + benchmark 必須。**HIP graph は TP と併用不可** |
| barrier 待ち中の先行 dispatch 構築 | 13.2 ms | 中〜高 | collective の順序を変える。正しさ検証が要る |
| `complete_batch` を event sync に | 11.7 ms | 低 | `done` event は status コピー後に record 済みで**ほぼ等価** |
| prefill の staging upload sync | 25 ms | 中 | 初回 prefill のみ。長時間運用では無視可 |
| `ScopedDevice` arch 検証キャッシュ | 2.4 ms | 極小 | **検出不可のため打ち切り** |
| dispatch 構築の並列化 | 46.5 ms | 高 | ceiling 実験により効果なしと判明済み |

## 6. 再現手順

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

# Attribution（--call-graph fp が必須）
PHASESHIFT_TP_MODEL_DIR=models/Qwen3.8-27B-PSQ \
  perf record -o perf.data -e cycles:u -F 4999 --call-graph fp -- \
  ./build-perf/tests/test_qwen35_tp_e2e
perf script -i perf.data > script.txt

# e2e 比較（profiler なし。これを主指標とする）
PHASESHIFT_TP_MODEL_DIR=models/Qwen3.8-27B-PSQ ./build/tests/test_qwen35_tp_e2e
```

worker の特定は「活動 span が 0.5 秒前後」（main は 112 秒）。`tid` は毎回変化するので
毎回実データから拾う。

## 7. 失敗学び

| 失敗 | 原因と対処 |
| --- | --- |
| `--call-graph dwarf,8192` で worker の leaf が 0 件 | worker スレッドの DWARF unwind 失敗。`fp` に変えれば解決（data 4.3GB → 93MB） |
| `perf report --sort ...tid...` がエラー | report の sort key に `tid` は無い（`pid` を使う）。`perf script` の field としては `tid` が在る |
| `perf script -F comm,tid,time,period,sym` で `sym` が空 | `-F` 指定時の perf 7.0.14 の挙動。デフォルトフォーマット（callchain）を使う |
| 集約タイマーの出力が来ない | 名前空間スコープの `const bool = lambda()` が HIP の `.hip` で動的初期化にならない。function-local static にする |
| phase 名が逆に出た | `ExecutionRole::Decode = 0` なのに `kPhaseName[0] = "prefill"`。enum の値と name 配列の対応を必ず確認 |
| 内訳合計と total が合わない | `continue` が 3 経路あるため、running mark（`t_mark`）方式で連続計測し `post = total - A - B - C` で検算する |
| **TP2 の tok/s が既存記録より 6.7% 高く出た** | `ContinuousBatcher` の prefill step が token0 をサンプルするため decode 窓は「生成 token 数 − 1」step。生成 token 数で割ると分母が 1 過剰。`decode_steps` で割る（`TP_TIMING` が両方を出す） |
| host 側の Attribution で効果を試算した | **GPU busy と重なる。** 上限は ceiling 実験で取る（本記録 §4） |
