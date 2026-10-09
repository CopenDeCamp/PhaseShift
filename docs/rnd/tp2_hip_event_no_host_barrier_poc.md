# TP=2 Host Barrier 排除 PoC（HIP Event 方式）

PhaseShift の TP=2 において、Host の condition_variable による rank 間 rendezvous を
使わず、既存 HIP Event と Peer Write Kernel だけで AllReduce が成立するかを検証した
PoC の記録である。**既存の TP 実装は変更していない。**

## 判定: **FAIL**

HIP Event の API セマンティクス上、**Host rendezvous なしでは安全な rank-local
AllReduce が成立しない**。根拠は下記 Step 1 / Step 3 / Step 4。

---

## 0. 条件

| 項目 | 値 |
| --- | --- |
| revision | `20d5b199`（`exp/gpu-mcu`）+ 本 PoC のテスト追加のみ |
| GPU | AMD Radeon AI PRO R9700 (gfx1201) × 2、`pcie_p2p=Y` |
| build | Release / gfx1201 / `build` |
| transport | `hip-p2p`（`peer_access_available=1`、`verify()` OK） |
| payload | BF16 `elems=16384`（32768 bytes）/ Step 1 は `kCount=4096`（8192 bytes） |
| watchdog | `PHASESHIFT_POC_TIMEOUT_S`（既定 600 秒）でプロセス強制終了 |

### 使用したコードとリポジトリへの残し方

PoC 用に次のテストを一時作成して計測した。**いずれも採否判断が済んだため
リポジトリには残していない**（AGENTS.md「非アクティブな実装を
リポジトリ内のarchiveとして保存しない」「単発 Gate 終了後の script は残さない」）。

| 一時ファイル（未保存） | 役割 |
| --- | --- |
| `tests/unit/test_tp_hip_event_order.hip` | Step 1: HIP Event の挙動検証 |
| `tests/unit/test_tp2_event_poc.hip` | Step 2〜4: 最小 AllReduce / 連続世代 / 性能比較 |
| `cmake/tests.cmake` の CTest 登録行（`gpu2;optional`） | 上記2件の登録。ファイルごと revert 済み |

既存 TP 実装（`src/phaseshift/runtime/tp/**` / `include/phaseshift/runtime/tp/**`）は
**変更していない**（`git diff` で無変更を確認済み）。本ドキュメントがこの PoC の
唯一の成果物である。

判定 FAIL のため、**再実装する価値があるのは §8 の doorbell 検討時のみ**。
その際は §再現手順 の仕様に沿って作り直す。

---

## 1. HIP Event の仕様と実機検証結果

### 仕様（公式ドキュメント）

`hip_runtime_api.h:3126-3151` の `hipStreamWaitEvent` の記載は次の通りのみ。

> This function inserts a wait operation into the specified stream.
> All future work submitted to stream will wait until event reports
> completion before beginning execution.

**次の2点はドキュメントに記載が無い**。

1. **event が未記録（`hipEventRecord` 未実行）の場合の挙動**
2. **cross-device wait の可否**

「event reports completion」が未記録 event に対して何を意味するかは未定義であり、
**実機で確認する必要があった**。

### 実機検証結果

一時テスト `tests/unit/test_tp_hip_event_order.hip`（§0 の通り未保存）で確認した。
GPU0 が `peer write` で GPU1 の inbox に
sentinel (`0x3C3C`) を書き込み、GPU1 の verify kernel が全要素を照合する。
事前に inbox は poison (`0xDEAD`) で埋めてあるため、**同期不成立は必ず検出される**。

| ケース | 順序 | 結果 | 証拠 |
| --- | --- | --- | --- |
| **A** | Record → Wait | **SYNC_OK** | `mismatches=0 observed=0x3c3c`、completion: GPU0 write 先行 (dev0@2µs / dev1@38µs) |
| **B** | Wait → Record（event 未記録） | 0/30 broken → SYNC_OK | completion: GPU0 write 先行 (dev0@17µs / dev1@63µs)。**GPU0 が偶然先行しただけ** |
| **B2** | B + `hipEventDisableTiming`（production flag） | SYNC_OK | 同上。`wait_api=2.00µs` |
| **C** | Wait → Record、**GPU0 送信を遅延** | **SYNC_BROKEN** | `mismatches=4096/4096 observed=0xdead`、completion: GPU1 verify 先行 (dev1@66µs / dev0@7310µs) |
| **D1** | Record →（host sleep 20ms）→ Wait | SYNC_OK | 遅延 Wait は正常動作。`wait_api=11.34µs` |
| **D2** | Wait →（host sleep 20ms）→ Record | **SYNC_BROKEN** | `mismatches=4096 observed=0xdead`、completion: GPU1 verify 先行 (dev1@352µs / dev0@454µs) |
| **E** | 同一 event を連続世代で再利用 | **2/4 broken** | 下記 §3 |

**API の host ブロック計測**: `hipStreamWaitEvent` は `2.00〜35.72µs`（初回3回目の
19.05µs を含む）。`hipEventRecord` は `0.77〜1.55µs`。いずれも **host をブロックしない**
（enqueue のみ）。ただし D1 の初回のみ 35.72µs を示し、冷たい path で数十µs が掛かる。

**結論**: 未記録 event への Wait は **`hipSuccess` を返すが、同期として機能しない**。
ケース B が通ったのは GPU0 の send が GPU1 の verify より先に完了した偶然によるもので、
**B は同期の証明にはならない**。C / D2 で決定的に壊れる。

```
EVENT_SYNC_VERDICT: UNRECORDED_WAIT_NOT_A_SYNC
```

---

## 2. Cross-device Wait の結果

- `hipStreamWaitEvent`（device 1 の stream、device 0 で作成した event）→ **`hipSuccess`**
- テストの summary: `cross_device_wait_supported=1`
- 既存実装も同じ構造を使っている（`hip_peer_tp_transport.hip:332`）:
  `sent_[q][r]` は `ScopedDevice(devices[q])` 下で作成され、device r の stream で wait される
- `verify()`（3 sizes × 4 rounds）も通る

**cross-device wait 自体は本環境で問題ない。** 問題は §1 の「未記録 event」にある。

なお既存実装では `TpBarrierGroup::arrive`（`tp_barrier_group.cpp:74-108`）が
**全 rank の到着を host 側で待ってから** `enqueue_sum()` を呼ぶため、
**必ず「Record 済み」の状態で Wait が発行される**。これが現状の安全性の根拠である。

---

## 3. Record / Wait 順序の検証（世代跨ぎ）

ケース E: 単一 event を4世代で再利用し、parity double buffer（inbox[2]）を使う。
**各世代の期待値を変える**ことで、別世代のデータを読むことを検出する
（`expect[g] = 0x1000 + g`）。奇数世代は Wait 先行、偶数世代は Record 先行。

| 世代 | parity | expected | observed | mismatches | 判定 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0x1000 | 0x1000 | **3200** | **BROKEN**（部分読込 = 前世代への早期上書き） |
| 1 | 1 | 0x1001 | **0x1003** | **4096** | **BROKEN**（別世代を読んだ） |
| 2 | 0 | 0x1002 | 0x1002 | 0 | OK |
| 3 | 1 | 0x1003 | 0x1003 | 0 | OK |

**`E: 2/4 generations with SYNC_BROKEN`**

- gen0 の `mismatches=3200`（4096 中）は **同一世代内で書き込みが途中までしか
  及んでいない**ことを意味する → 前世代（または同一世代の他 rank）の inbox への
  早期上書き
- gen1 の `observed=0x1003` は **gen3 の値** → 世代跨ぎの誤読

**parity double buffer だけでは安全性を証明できない。** バッファの世代分割は
「どの領域を使うか」を決めても、「相手の書き込みが完了したか」は決めない。

---

## 4. 最小 AllReduce の実装（Step 2）

一時テスト `tests/unit/test_tp2_event_poc.hip`（§0 の通り未保存）に実装した。

### rank-local の enqueue 経路（host barrier なし）

各 rank は**独立した host thread と HIP stream**を持ち、次を自己完結で enqueue する。

```
fill(g, rank)          → part[me] に世代依存データを device 上で生成
copy(part[me] → inbox[peer])  → peer write（GPU Kernel）
record(sent[me])       → 自 rank の event を記録
wait(sent[peer])       → 相手の event を cross-device wait ← ここが未記録になり得る
add(part[me] += inbox[me])    → local read のみ
check(g, part[me])     → 世代別 verdict に誤答数を累積
```

### 制約の遵守状況

| 制約 | 状況 |
| --- | --- |
| Peer Read 禁止 | **遵守**。copy は local read + peer write、add は local read のみ |
| GPU Kernel による Peer Write | **遵守**（`tp_poc_copy_kernel`） |
| CPU による tensor データ転送禁止 | **遵守**。`tp_poc_fill_kernel` が device 上で世代データ生成 |
| Host の condition_variable / barrier 禁止 | **遵守**（B モードは `TpBarrierGroup` を使わない） |
| `hipDeviceSynchronize` による中間同期禁止 | **遵守**（不使用） |
| 最終検証時の `hipStreamSynchronize` のみ許可 | **遵守** |
| 既存の BF16 変換・加算処理の再利用 | device 側は `hip_peer_tp_transport.hip` の `tp_f32_to_bf16` と同一アルゴリズム、host 参照は既存 `ps::f32_to_bf16_rne` |
| **同期が不成立なら誤ったデータを加算させない** | **満たせない**（下記） |
| rank の処理開始時刻をずらしても正しく動く | **満たせない**（§5） |

### 「同期が不成立なら誤って加算させない」を満たせない理由

host は `hipStreamWaitEvent` の戻り値から**同期が成立したかを知る手段を持たない**
（`hipSuccess` は未記録 event でも返る）。したがって:

- host が gate を設けるには **相手の Record 完了を host が知る必要がある**
  → それ自体が host rendezvous
- GPU 上で gate するには add kernel が相手の状態を**確認できる必要がある**
  → これは HIP Event ではなく GPU-side polling（Doorbell）の領域

**HIP Event だけでは、この制約を満たす実装が存在しない。**

---

## 5. 連続世代の検証（Step 3）

`PHASESHIFT_POC_GENS=1,100,1000,10000` × `PHASESHIFT_POC_SKEW_US=0,100,1000,10000`
の16通り。世代ごとに異なる入力、parity double buffer、**世代別 verdict**
（`verdict[g]`）で全世代の結果を独立に検証。

計数は **rank × generation** 単位（最大 `2 × gens`）。

### A: 既存 `TpBarrierGroup` + `HipPeerTpTransport`

| gens | skew=0 | skew=100µs | skew=1000µs | skew=10000µs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 0 |
| 100 | 0 | 0 | 0 | 0 |
| 1000 | 0 | 0 | 0 | 0 |
| 10000 | 0 | 0 | 0 | 0 |

**16/16 ケースで broken=0。skew 10ms でも正答。**

### B: host barrier なし

| gens | skew=0 | skew=100µs | skew=1000µs | skew=10000µs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 / 2 | **0 / 2** | **0 / 2** | **0 / 2** |
| 100 | 103 / 200 | 198 / 200 | 197 / 200 | 196 / 200 |
| 1000 | 1897 / 2000 | 1998 / 2000 | 1994 / 2000 | 1968 / 2000 |
| 10000 | **19755 / 20000** | **19731 / 20000** | **19688 / 20000** | **19731 / 20000** |

**合計 87,457 rank-generation が誤答。**

- gens=10000 で **98.4〜98.8% 誤答**
- gens=10000 skew=0 の `bad_elems = 323,662,912` / 327,680,000 = **98.8%**
- **gens=1 では skew≥100µs で pass する**（その回は偶然順序が保たれた）
  → **単発成功は成立の証明にならない**。手順書の警告どおり

### skew の影響

skew を変えても誤答率はほぼ同じ（98.4〜98.8%）。**skew を入れても、入れなくても
成立しない。** 逆に gens=1 で skew を入れると通ってしまうため、**skew ありの単発
テストは危険**である。

### Double Buffer だけでは安全性が証明できない

§3 のケース E と §5 の結果がそれを示す。**追加の依存関係（相手の書き込み完了の
確認手段）が必要。**

---

## 6. Host Barrier 有無の性能比較（Step 4）

### 6-1. profiler なし（主指標）

`bench_gens=2000`、同じ GPU / 同じデータ量 / 同じ回数。5回〜15回計測。

**reps=5**

| mode | e2e 中央値 (ms) | e2e 平均 (ms) | µs/collective | spread | host_wall (ms) | host_api (µs) | host_wait (µs) | skew (µs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_existing_barrier | **47.601** | 47.630 | **23.80** | 0.6% | 32.182 | 54,030.1 | 0.0 | 10.9 |
| B_no_host_barrier | **54.319** | 54.503 | **27.16** | 4.8% | 16.247 | 13,345.8 | 2,901.5 | 16.6 |

**reps=15**

| mode | e2e 中央値 (ms) | e2e 平均 (ms) | µs/collective | spread | host_wall (ms) | host_api (µs) | host_wait (µs) | skew (µs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_existing_barrier | **47.469** | 47.956 | **23.73** | 15.5% | 32.829 | 55,549.7 | 0.0 | 42.2 |
| B_no_host_barrier | **53.667** | 54.145 | **26.83** | 18.8% | 16.408 | 13,784.4 | 2,884.8 | 19.2 |

**speedup (A/B) = 0.876x 〜 0.885x → PoC の方が約 12〜13% 遅い。**

**reps=15 / `bench_gens=200`（浅い queue）**

| mode | e2e 中央値 (ms) | spread |
| --- | ---: | ---: |
| A_existing_barrier | 4.445 | 6.9% |
| B_no_host_barrier | 4.896 | **61.0%** |

`speedup = 0.908x`。**B は spread が 61% と極めて不安定**（A の 6.9%）。
barrier が無いことで rank の進行が host スケジューリングに振り回される。

### 6-2. GPU 側の待機時間・kernel 時間（event 計測、profiler なし）

`event instrumented, first 64 gens`（rank0）。

| mode | fill | collective | copy | wait | add |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.009 ms | 0.037 ms | （内包） | （内包） | （内包） |
| B | 0.008 ms | 0.031 ms | 0.009 ms | **0.013 ms** | 0.009 ms |

A の collective は `pre → post` のため **GPU 側の wait を内包**している。
B で分解すると wait = 0.013 ms / collective と、**GPU 側の待機は小さくない**。

### 6-3. GPU idle（`rocprofv3 --kernel-trace`、profiler あり）

collective kernel のみを cluster 化し、verify probe / verdict readback を除外。
`bench_gens=2000` の区間。

| mode | span | Agent1 busy / idle | Agent2 busy / idle | inter-kernel gaps |
| --- | ---: | ---: | ---: | ---: |
| A | 96.848 ms | 13.987 ms (14.4%) / **82.861 ms (85.6%)** | 16.241 ms (16.8%) / **80.607 ms (83.2%)** | 72.599 ms (75.0%) |
| B | 86.209 ms | 15.375 ms (17.8%) / **70.834 ms (82.2%)** | 16.450 ms (19.1%) / **69.760 ms (80.9%)** | 58.236 ms (67.6%) |

kernel 内訳（両 agent 合算）:

| mode | 内訳 |
| --- | --- |
| A | add(existing) 4016 / 9.522 ms、fill 4000 / 7.490 ms、copy(existing) 4016 / 6.996 ms、check 4000 / 6.220 ms |
| B | check 4000 / 8.970 ms、copy(poc) 4000 / 8.428 ms、add(poc) 4000 / 7.121 ms、fill 4000 / 6.606 ms |

**profiler ありでは B の方が span が短く（86.2 vs 96.8 ms）、GPU util も高い。**
しかし profiler なしでは B の方が遅い。この矛盾は profiler が host の
`hipLaunchKernel` コストを大きく増幅するためと考えられる。

### 6-4. profiler ありでの e2e（参考）

| mode | e2e (ms) | host_wall (ms) | host_api (µs) | host_wait (µs) |
| --- | ---: | ---: | ---: | ---: |
| A | 88.040 | 87.942 | 132,832.9 | 0.0 |
| B | 78.446 | 77.299 | 18,960.5 | 58,338.8 |

A は `host_wall ≈ e2e`（host が律速）で、`host_api` が profiler-off の 2.4倍
（54.0ms → 132.8ms、両 rank 合算）。A は HIP API 呼び出しが B の約4倍
（54.0ms vs 13.4ms）なので、**profiler の per-call オーバーライトが A を強く罰する**。
→ **profiler ありの数値で優劣を判断できない。手順書どおり profiler なしを主指標とする。**

### 考察

- B は host 時間を大幅に減らす（host_wall 32.2→16.2 ms、host_api 54.0→13.4 ms = −75%）
- それでも e2e が **12〜13% 悪化**する
- e2e − host_wall で見ると、GPU 側の drain 時間が A=15.4ms / B=38.1ms と B の方が長い
- GPU busy は A/B とも約16ms なので、**B は host が enqueue を終えた後も GPU が
  idle な時間が長い**。deep queue になった cross-device wait の連鎖が
  2 GPU の進行を lockstep 化し、bubble を生んでいる可能性がある
  （**未検証の推測**。本 PoC の範囲では因果を確定していない）

---

## 7. 成功・部分成功・失敗の判定

### **FAIL**

手順書の FAIL 定義は「HIP Event の API セマンティクスや実機制約により、
Host rendezvous なしでは成立しない」に完全に合致する。

| 判定材料 | 結果 |
| --- | --- |
| HIP Event の API セマンティクス | **未記録 event への Wait は同期しない**（§1 C / D2 で実機証明） |
| 連続世代の安全性 | **98.4〜98.8% 誤答**（§5、gens=10000） |
| 任意の rank arrival skew | **skew の有無で不変に誤答**（§5）。逆に gens=1+skew で偶然 pass する |
| Double Buffer での安全性 | **証明不可**（§3 ケース E で 2/4 世代が誤答） |
| 「誤ったデータを加算させない」 | **実装不能**（host が同期成立を知る手段が無い、§4） |
| 性能（profiler なし） | **0.876x 〜 0.908x（PoC の方が遅い）**、spread も B が悪い（61% vs 6.9%） |

**既存実装（`TpBarrierGroup` + `HipPeerTpTransport`）は変更しない。**
Step 3 の A 経路が16/16ケースで skew 10ms まで正答していることが、
現行 host barrier の妥当性を再確認する結果でもある。

### Handout への報告事項（事前 Record の要否）

> 「もし Event の事前 Record が必須なら、その事実を報告する」

**報告: 事前 Record（または同等の順序保証）が必須である。**

host 側で `enqueue_sum` を呼ぶ前に、**相手 rank が Record 済みであることを
host が確認する**しか手段がない。これは host そのものであり、
「Host が全 rank の Record を待つ必要がある方式」= **今回の最終目標を満たさない**。

---

## 8. 今後の推奨方式

### 推奨: GPU-side Doorbell（今回の PoC では非実装）

rank-local enqueue を成立させるには、**add kernel が GPU 上で「相手の書き込みが
完了したか」を確認できる**必要がある。HIP Event は host の `record` に依存するため
これに適さない。doorbell なら次の形が取れる。

```
rank r の add kernel:
    while (peer_generation[r] != expected) { /* local spin */ }
    add(...)
```

- `peer_generation[r]` は **device r 上のローカルメモリ**に置く
- 相手 rank が copy 完了時に **peer write で世代番号を書き込む**
- add kernel は **local read で spin** する → peer read を回避できる
  （peer read が不安定であることは `docs/rnd/tp_exec2_p2p.md` で確認済み）

ただし次の未解決点を持つ。

1. **spin の消費電力 / SM 占有** — 大きな payload では spin が長い
2. **世代のラップアラウンド** — double buffer の parity では偽一致し得る
   （§3 の gen0/gen1 のような部分読込を再現しうる）。3世代以上や明示的な
   generation counter が必要
3. **GPU ハング時の検出不能** — spin が回り続けると watchdog でしか捕捉できない
4. **2 GPU 同時 spin による互相ロック**（双方が待つ対象が揃わないケース）の設計

### 現実的な中間案

doorbell を入れるまでは、**現行 `TpBarrierGroup` の host 待ちを短くする**方向が
安全である。前節の計測で、A の `host_wait` は profiler-off で **0.0µs**（既に
host は event 依存でブロックしておらず、cv wait も2000 collective で実測上消えている）
と出ている。つまり **現行実装の host 待ちは既にほぼゼロ**で、さらに削る余地は
`host_api`（54.0ms / 2000 collective × 2 rank = 27µs/collective）の側にある。

`host_api` の大半は `ScopedDevice::create` と `launch_*` の HIP API 呼び出しであり、
`docs/next_task.md` の改善候補1・2（`resolve_psq4_physical` / `find_value`）と
重なる。**先に host dispatch を削り、残った rendezvous コストを見直す方が順当。**

---

## 再現手順

**テストソースはリポジトリに残していない**（§0 の通り）。再実施する場合は
§0 の一時ファイルの仕様に沿って作り直し、`cmake/tests.cmake` に
`gpu2;optional` で登録する。手順は次の通り。

```bash
# Step 1（HIP Event 検証）
PHASESHIFT_POC_TIMEOUT_S=300 ./build/tests/test_tp_hip_event_order

# Step 2/3/4（フル）
PHASESHIFT_POC_TIMEOUT_S=600 ./build/tests/test_tp2_event_poc

# 性能のみ・mode 指定
PHASESHIFT_POC_MODE=A PHASESHIFT_POC_BENCH_ONLY=1 \
PHASESHIFT_POC_BENCH_GENS=2000 PHASESHIFT_POC_BENCH_REPS=15 \
./build/tests/test_tp2_event_poc

# GPU idle（kernel-trace）
rocprofv3 --kernel-trace -f csv -d <dir> -- \
  env PHASESHIFT_POC_MODE=B PHASESHIFT_POC_BENCH_ONLY=1 \
      PHASESHIFT_POC_BENCH_GENS=2000 PHASESHIFT_POC_BENCH_REPS=1 \
  ./build/tests/test_tp2_event_poc
```

### 一時テストの仕様（作り直す際の要点）

- `tests/unit/test_tp_hip_event_order.hip`
  - GPU0 が peer write で GPU1 の inbox に sentinel (`0x3C3C`)、事前に poison (`0xDEAD`) 埋め
  - GPU1 の verify kernel が全4096要素を照合し、不一致数を `atomicAdd` で集計
  - ケース A/B/C/D1/D2/E（§1 の表）を、fresh event / `hipEventDisableTiming` /
    GPU 遅延 kernel / host sleep で切り替えて実行
  - watchdog: `PHASESHIFT_POC_TIMEOUT_S`（既定600秒）で `std::_Exit(3)`
- `tests/unit/test_tp2_event_poc.hip`
  - rank ごとに独立 host thread + HIP stream。開始は `std::atomic<bool>` の release、
    rank1 のみ sleep で skew を作る（**このスキャフォールドは PoC の設計には含めない**）
  - B の enqueue 経路: `fill → copy(peer write) → record → wait(peer) → add → check`
  - fill kernel が device 上で世代依存データを生成（CPU の tensor 転送をしない）
  - 期待値: `((gen + rank*3 + i*7) % 128) - 64`。両 rank の和は `[-128, 126]` の
    整数で **bf16 で厳密に表現可能**（|x| ≤ 256 が8bit mantissa で正確）
    → rounding 差異による誤検出を排除
  - 世代別 `verdict[g]`（`g = 0..gens-1`）+ parity double buffer（`inbox[2]`）
  - A 経路は既存 `TpBarrierGroup` + `TpTransport::sum_hidden` をそのまま使用
  - 最終検証時のみ `hipStreamSynchronize`。中間は `hipDeviceSynchronize` 不使用

環境変数: `PHASESHIFT_POC_GENS`（既定 `1,100,1000,10000`）、
`PHASESHIFT_POC_SKEW_US`（既定 `0,100,1000,10000`）、
`PHASESHIFT_POC_ELEMS`（既定 16384）、
`PHASESHIFT_POC_BENCH_GENS`（既定 2000）、
`PHASESHIFT_POC_BENCH_REPS`（既定 5）、
`PHASESHIFT_POC_MODE`（`A`/`B`）、
`PHASESHIFT_POC_BENCH_ONLY`、
`PHASESHIFT_POC_TIMEOUT_S`（既定 600）。

## 制約と注意

- 本計測は **1〜2 GPU 環境の 1 回実行〜15 回実行**であり、`docs/perf/current.md`
  の qualified baseline ではない。
- payload は 32768 bytes（Step 1 は 8192 bytes）。実運用の collective size とは異なる。
- `rocprofv3 --kernel-trace` は host の HIP API コストを大きく増幅するため、
  **profiler ありの e2e 数値で優劣を判断してはならない**（§6-4）。
- 世代数 10000 での誤答率は、rank × generation 単位（最大 20000）での計数である。
