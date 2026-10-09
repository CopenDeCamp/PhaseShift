# 現在のタスクメモ

2026-10-09 時点。作業中の計測結果と検討事項を残す。

> これは lifecycle が定義された `docs/user/` `docs/developer/` `docs/perf/`
> `docs/rnd/` `docs/references/` には属さない、作業中のメモである。
> 数値はすべて未 qualified の1回計測であり、`docs/perf/current.md` の正本を
> 決して代替しない。

---

## 1. GPU-MCU テストの一時除外（OSハング対策）

### 背景

GPU-MCU テストに OS ハングの可能性があるため、一旦実行対象から外す。

### 変更内容（`cmake/gpu_mcu/tests.cmake`）

label の付け替えのみ。テストの統合・削除・rename は行っていない。

| 対象 | 件数 | 変更前 | 変更後 |
| --- | ---: | --- | --- |
| GPU 実行する GPU-MCU | 86 | `gpu1;gpu_mcu;required` | `gpu1;gpu_mcu`（`required` 解除） |
| GPU 実行（optional / external） | 6 | `gpu1;optional;external_files;resident` | `gpu1;gpu_mcu;optional;external_files;resident` |
| CPU のみ（GPU 非実行） | 5 | `cpu;required` | `cpu;gpu_mcu;required`（`required` 維持） |
| 既に `required` 無し | 1 | `gpu1;gpu_mcu` | 変更なし |

CPU のみの5件（`plan_cache` / `static_attention_plan_class` / `decode_backend_policy` /
`plan_sampling_gate` / `layer_dispatch_range`）は `hip*` / `__global__` /
`GpuMcu*::create` / `hsaco` の参照がすべて0件で、plan 構築と policy 判定のみ。
ハングしないので `required` を残した。

### 変更後の件数（再 configure 済み）

| label | 件数 |
| --- | ---: |
| 全 test | 223 |
| `gpu_mcu` | 98（`test_gpu_mcu_*` 全98件と一致） |
| `required` | 123 |
| `required` ∩ `gpu_mcu` | 5（CPU のみの5件） |
| `-LE gpu_mcu` の残数 | 125 |
| `-E '^test_gpu_mcu_'` の残数 | 125（`-LE gpu_mcu` と同数 → label 完全） |

`-LE gpu_mcu` だけでは11件漏れていたのが、変更前は `test_gpu_mcu_*` 98件に対し
`gpu_mcu` label が87件しか付いていなかったため。**変更後は両者が一致し、
`-LE gpu_mcu` 単独で完全に除外できる。**

### 実行方法

```bash
# 全 test から GPU-MCU を除外
ctest --test-dir build -LE gpu_mcu --output-on-failure

# required acceptance から GPU-MCU GPU 実行を除外
ctest --test-dir build -L required -LE gpu_mcu --output-on-failure
```

### 残課題

1. **`phaseshift-required-tests` build target に66件の GPU-MCU が残っている**
   `PS_GPU_MCU_REQUIRED_TESTS`（68件 = GPU 66 + CPU 2）は変更していない。
   acceptance は**ビルドするが実行しない**状態。ビルドは hsaco のコンパイル
   のみで GPU 実行を伴わないためハングはしないが、`docs/developer/testing.md`
   の「required の集合は `required` label と `phaseshift-required-tests`
   target が正本」という記述と集合が食い違っている。
   → 今回は「実行だけ止めてビルドの回帰検知は維持、復帰は label 1語」を優先し
   て未修正。**集合を一致させるかは未決定。**

2. **3件が build target に無い（既存の穴）**
   `test_gpu_mcu_plan_cache` / `test_gpu_mcu_static_attention_plan_class` /
   `test_gpu_mcu_decode_backend_policy` は `cpu;required` label を持つが
   `PS_GPU_MCU_REQUIRED_TESTS` に入っていない（`ninja -n phaseshift-required-tests`
   で3件ともビルドされないことを確認）。clean build から
   `tests/run_required_acceptance.py` を走らせると ctest が「実行不可能」と
   出て acceptance が落ちる。**今回の変更とは無関係に既存**。未修正。

3. **`tests/run_required_acceptance.py` は除外オプションを持たない**
   `ctest -L required` をハードコードしており、`--build-dir` / `--no-build` /
   `--arch` のみ。GPU-MCU を含む required がそのまま走る。ローカルで外す場合は
   `ctest` を直接叩く必要がある。**未修正。**

4. **`docs/developer/testing.md` の記述が実態と乖離**
   「required は …GPU-MCU substrate test… からなる」「2つの正本」の記述を
   実態に合わせる必要がある。**未更新**（「一旦」の暫定措置のため保留）。

5. AGENTS.md の「required acceptance では skip を失敗として扱う」に照らすと、
   公式 acceptance から除外するのは想定外。**ローカルの安全確認用**として扱い、
   acceptance 結果として公表しないこと。

---

## 2. host バックエンドでの pp / tg 計測

### 結論

`phaseshift-bench` の `pp` / `tg` はともに `--decode-backend TYPE`
（`host` | `gpu-mcu`）を受け取り、**既定値が `host`**。フラグを付けない、
または `--decode-backend host` と明示すれば GPU-MCU を介さずに計測できる。

| 箇所 | 内容 |
| --- | --- |
| `src/apps/bench/pp.hip:44-45` | `decode_backend = DecodeBackend::Host`（既定） |
| `src/apps/bench/pp.hip:63` | usage: `--decode-backend TYPE host \| gpu-mcu (default host)` |
| `src/apps/bench/pp.hip:305` | `exec_config.backend = opts.decode_backend` |
| `src/apps/bench/tg.hip:49-50` / `:73` / `:381` | 同上 |
| `src/phaseshift/models/qwen35/runtime/decode_backend.cpp:21` | `host` / `gpu-mcu` を case-insensitive で受理 |

`host` 指定時は `decide_decode_backend` が reason `BackendIsHost` を返して
`mcu_selected = false` になり、`ensure_mcu_state()` が呼ばれない
（=`MCU persistent 資源自体が作られない`）、`launch_host_backend_range()`
（`executor.hip:1562-1564`）が全 dispatch を実行する。preffill も decode も
純粋な host 経路。

逆に `gpu-mcu` 指定時は `persistent_ready` 時に prefill も MCU hybrid に乗るため、
`docs/perf/current.md:73-83` のように pp・tg で値が分かれる。

### build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPHASESHIFT_ROCM_ROOT="$(rocm-sdk path --root)" \
  -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$(rocm-sdk path --root)" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake --build build --parallel
```

682/682 成功（revision `5d8a7b91`、`exp/gpu-mcu`）。warning は
`test_gpu_mcu_program_plan_three_primitive.hip:202` の `-Wswitch` 1件のみ。
成果物 `build/phaseshift-bench` / `build/phaseshift-compute`。
`build` 目録は作業途中で消えていたため再 configure。

### 計測コマンド（host）

```bash
# pp2048
./build/phaseshift-bench pp --model-dir models/Qwen3.8-27B-PSQ \
  --prompt-tokens 2048 --mode forward --page-tokens 16 --arena-gib 24 \
  --runs 3 --warmup 1 --device 1

# tg128 / ctx2048
./build/phaseshift-bench tg --model-dir models/Qwen3.8-27B-PSQ --mode greedy \
  --context 2048 --tokens 128 --prefill-chunk 2048 --compute-logits 1 \
  --page-tokens 16 --arena-gib 24 --warmup 2 --device 1
```

**pp / tg の新規計測は未実施**（依頼は host で「計測できるか」の確認まで）。

---

## 3. TP=2 の時刻計測

### 入口

TP は `TpCoordinator` API 経由のみで、`phaseshift-bench` / `phaseshift-compute` /
server からは未露出（`docs/developer/tensor_parallel_execution.md` §10）。
動かせるのは次の2つ。

| binary | 内容 | 必要条件 |
| --- | --- | --- |
| `build/tests/test_qwen35_tp_e2e` | Qwen3.8-27B TP=1 vs TP=2 greedy 一致 | `PHASESHIFT_TP_MODEL_DIR`、2 GPU |
| `build/tests/test_qwen35_tp_execution` | tiny model の prefill / decode | 2 GPU |

素の実行は **PASS**（greedy 16 token が TP1/TP2 一致、
`tp1 gate_rows=17408 tp2 gate_rows=8704`、rank0/rank1 の final hidden も
bit-identical）。壁時間は約 114 秒（model load が大半）。

### 計測方法

```bash
PHASESHIFT_TP_MODEL_DIR=models/Qwen3.8-27B-PSQ \
  rocprofv3 --kernel-trace --hip-runtime-trace -f csv -d <dir> -- \
  ./build/tests/test_qwen35_tp_e2e
```

kernel と HIP API を**同一実行**で取得して correlation id で突き合わせるのが必須
（別プロセスだとタイムラインが揃わず、段の切り分けを誤る）。

- **compute 窓の特定**: `hipLaunchKernel` > 5000 かつ活動 span < 5s の
  スレッド2本 = rank worker。main thread は `hipLaunchKernel` 26,547回だが
  span 111s なので除外。`hipEventRecord` を3万回呼んでいたスレッドは
  **keepalive**（`executor.hip:1767`、`PHASESHIFT_KEEPALIVE` 既定 ON）で
  compute と無関係。

- **段分け**: gap > 200ms。tp1 / tp2 / weight load が混ざるので、
  最大 gap で切ると段を取り違える。span と busy が説明できない時は
  分割をやり直す。

- **`--kernel-trace` 単体だと host 側が出ない**。`--hip-runtime-trace` が必須。

### 計測結果（1回計測・未 qualified）

条件: `test_qwen35_tp_e2e`（prompt 5 token + greedy 16 token、1 request、
`max_scheduled_tokens=16`）。**production 条件ではない。**

**compute 窓 = 605.7 ms**（prefill 153.7 ms + decode 16 step 452.0 ms）

GPU busy（両 agent union）:

| 区間 | busy | 率 |
| --- | ---: | ---: |
| prefill 153.7 ms | 26.4 ms | 17.2% |
| decode 452.0 ms | 300.7 ms | 66.5% |

transport kernel（`tp_copy_bf16` / `tp_add_peers_bf16`）は **7.6〜8.0 ms**、
窓の 1.3% のみ。

#### decode 452.0 ms 区間の host 分解（rank worker 1本）

| 区分 | 時間 | 率 | GPU busy 中 | **GPU idle 中** |
| --- | ---: | ---: | ---: | ---: |
| dispatch 構築（host 計算） | 178.3 ms | 39.4% | 131.8 ms | **46.5 ms** |
| `hipLaunchKernel` API 本体 | 119.8 ms | 26.5% | 推計 45〜54 ms | **推計 66〜74 ms** |
| `complete_batch` 全 stream sync（16回） | 85.9 ms | 19.0% | 71.7 ms (83%) | **14.3 ms** |
| barrier cv wait（約900回） | 58.9 ms | 13.0% | 42.8 ms (73%) | **16.1 ms** |
| その他 API | 8.4 ms | 1.9% | — | — |
| 計 | 451.3 ms | 100% | | **151.3 ms** |

「GPU busy 中」= その host 待ちの間に GPU が動いていた＝**正当な GPU 待ち（hidden）**。
「GPU idle 中」= **GPU が止まっているのに host が詰まっている＝実ストール**。

#### prefill 153.7 ms 区間

| 区分 | 時間 | GPU busy 中 | GPU idle 中 |
| --- | ---: | ---: | ---: |
| `hipMemcpyAsync`（staging pool upload 1400回） | 64.1 ms | — | — |
| staging upload sync（1397回） | 28.5 ms | 3.6 ms (13%) | **25.0 ms** |
| dispatch 構築 | 30.6 ms | 5.6 ms (18%) | **25.0 ms** |
| barrier cv wait | 22.4 ms | 1.7 ms (8%) | **20.7 ms** |

#### host 待ちの発生箇所（コード）

| 区分 | 箇所 |
| --- | --- |
| `complete_batch` 全 stream sync | `src/phaseshift/models/qwen35/runtime/executor.hip:3047`（`hipStreamSynchronize`）。毎バッチ1回、1回 5.4 ms |
| barrier cv wait | `src/phaseshift/runtime/tp/tp_barrier_group.cpp:106`（`cv_.wait`）。rank 間 rendezvous |
| staging upload sync | `src/phaseshift/models/qwen35/runtime/program_executor.hip:324-330` |
| transport の device-side 同步 | `hip_peer_tp_transport.hip:320`（`hipEventRecord`）/ `:332`（`hipStreamWaitEvent`）。host は介さない |

### 重要な identity

```
GPU idle 151.3 ms ≈ host 総時間 451.9 ms − GPU busy 300.7 ms = 151.2 ms
```

**一致する。** host スレッドは窓内 100% 何かやっている（idle な瞬間が無い）ので、
GPU busy の 300.7 ms は必然的に host 作業と重なっている。
つまり**「GPU busy 中の host 待ちを別の場所へ移して埋める」余地は既に無く**、
idle を減らすには **host スレッドの総経過時間を削る**しかない。

（両 GPU union では busy 342.2 ms / 両方同時 idle 109.8 ms /
片方だけ busy 41.5 ms。rank 独立なのでボトルネックは各 rank の host 側）

### GPU idle 151.3 ms の attribution（decode、rank A）

| 要因 | GPU idle | 割合 |
| --- | ---: | ---: |
| dispatch 構築 | 46.5 ms | 31% |
| `hipLaunchKernel` API 本体（差し引き推計） | 65〜74 ms | 43〜49% |
| barrier cv wait | 16.1 ms | 11% |
| `complete_batch` sync | 14.3 ms | 9% |

### 理論下限

host の**実作業**（dispatch 構築 178.3 + launch 119.8 + その他 9.0 = **307.1 ms**）
は GPU busy **300.7 ms** とほぼ同率。両者が完全に重なった場合の下限は
**≈307 ms（現在 452 ms 比で −32%）**。ただし barrier のスケューと rank0 の
commit 依存は不可避なので、現実的な到達点は **330〜360 ms** がライン。

### 意味のある知見

1. **collective のコストは kernel ではなく host の rendezvous 側**
   transport kernel は 7.6 ms（窓の 1.3%）だが、barrier の host 待ちは 58.9 ms。
   `docs/rnd/tp_exec2_p2p.md` の「collective の 81% が host 側」という既存計測と一致。

2. **`complete_batch` sync の 83% は GPU busy 中**＝正当な待ち。実ストールは
   14.3 ms しかない。ここを攻めても効果は小さい。

3. **`hipLaunchKernel` が 1件あたり 3.9 µs × 30,548回**。GPU idle の最大要因。
   削るには kernel 発行数を減らす（fusion）しかないが、
   **HIP graph は TP schedule と併用不可**（`tensor_parallel_execution.md` §10）。

4. **host 側の black box**: `hipGetLastError` → 次の `hipLaunchKernel` 間が
   **8.3 µs/dispatch × 24,117 = 199.54 ms**。これが dispatch 構築の実体だが、
   中身（`fill_staging_image` / `try_launch_optimized` の selector 解決等）の
   Attribution は**未実施**。`perf` 等での host 側 profile が必要。

---

## 4. 改善候補のリスク / 成功率評価

### 最有力（リスク最低・成功判定が trace で機械的できる）

**`ScopedDevice::create` の arch 検証を hot path から外す**

コードの矛盾:

```cpp
// include/phaseshift/core/gpu/gfx_arch.h:9-11
// architecture is a build/startup contract: it is validated once when the
// device is acquired and never queried on the hot path.
inline bool is_gfx1201(uint32_t device = 0) {
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, device) != hipSuccess) return false;
```

`is_gfx1201` の唯一の呼び出し元は `src/phaseshift/core/gpu/scoped_device.cpp:28`
で、これは `ScopedDevice::create` の中身。そして `ScopedDevice::create` は
`src/phaseshift/runtime/tp/hip_peer_tp_transport.hip:313/328` の `enqueue_sum`
ループ内で**毎 barrier 4回**呼ばれている。

実測（compute 窓 605.7 ms）:

| 項目 | rank A | rank B |
| --- | ---: | ---: |
| `ScopedDevice::create` 回数 | 5,450 | 5,538 |
| 列 API time（union） | 6.28 ms | 6.31 ms |
| + 直前/直後 gap | 9.76 ms | 10.12 ms |
| **総コスト** | **16.04 ms (2.65%)** | **16.43 ms (2.71%)** |
| うち GPU idle 中 | **12.25 ms** | **12.64 ms** |
| 内 `is_gfx1201` の `hipGetDeviceProperties` | 0.98 ms（5,450回、0.18 µs/call） | 0.97 ms |
| **decode 452 ms 区間のみ** | **6.22 ms (1.38%)**、GPU idle 2.91 ms | 6.40 ms、GPU idle 3.08 ms |

- **リスク**: 極小。arch は実行中に変わらない。per-device の static な
  検証済みフラグを挟むだけ。エラーメッセージは初回検証で維持。
- **成功判定**: 修正後に `hipGetDeviceProperties` の回数が hot path で 0 に
  なることを再計測すれば機械的に確認できる。
- **限界**: decode で 6.2 ms（1.38%）。劇的にはならない。

### 次点以下

| 候補 | GPU idle 効果量 | リスク | 判定 |
| --- | ---: | --- | --- |
| per-dispatch の static guard 系（`kv_calib_dump_maybe` 等） | <1 ms | ゼロに近い | 効果が小さすぎて割に合わない |
| `complete_batch` を event sync に | 14.3 ms | 低 | **ほぼ等価**（`done` event は status コピー後に record 済み）。効かない |
| prefill の staging upload sync（double-buffer 化） | 25 ms | 中 | `uploaded[]` が恒久なので**初回 prefill のみ**。長時間運用では無視可 |
| barrier 待ち中の先行 dispatch 構築 | 16.1 ms | 中〜高 | collective の順序を変える。正しさ検証が要る |
| dispatch 構築の並列化 | 46.5 ms | 高 | ownership / 決定論的実行を変える |
| kernel fusion で launch 数削減 | **66〜74 ms（最大）** | **高** | HIP graph は TP と併用不可。kernel 開発 + benchmark 必須 |

### 推奨する順番

1. **host 側 8.3 µs/dispatch の Attribution（`perf` 等）** — 情報価値が高い。
   本命がどこか分かるまで手を出さない
2. ScopedDevice の arch 検証キャッシュ — リスク・検証・成功判定が揃っている
3. 以降は 1 の結果を見て決める

---

## 5. 未完了タスク

- [ ] `docs/now_task.md` の残課題（§1 の5件）の処理判断
- [ ] host バックエンドでの pp2048 / tg128 計測の実行（§2、コマンド準備済み）
- [ ] host 側 8.3 µs/dispatch の Attribution（§3、本命の black box）
- [ ] ScopedDevice の arch 検証キャッシュ実装（§4、リスク最低の改善）
- [ ] `docs/developer/testing.md` の required 構成の記述更新（§1 残課題4）

## 6. 環境

| 項目 | 値 |
| --- | --- |
| revision | `5d8a7b91`（`exp/gpu-mcu`、作業前は clean） |
| GPU | AMD Radeon AI PRO R9700（gfx1201）× 4 可視、計測時は idle |
| build | Release / gfx1201 / `build` / benchmarks ON / optional tests OFF |
| profiler | `rocprofv3` 1.3.5 |
| model | `models/Qwen3.8-27B-PSQ`（18 GB） |
| 作業スクリプト | `/tmp/opencode/tp2_prof/`（`analyze_kt.py` / `segments.py` / `detail.py` / `union.py` / `hip.py` / `hip_thread.py` / `correlate.py` / `host_breakdown.py` / `host_wait.py` / `host_wait2.py` / `wait_gpu.py` / `scoped.py` / `scoped2.py`） |

`stall.py` は重複除去が O(n²) のまま残っていて遅いため中断済み。
必要な数値は `wait_gpu.py` の出力に揃っている。
