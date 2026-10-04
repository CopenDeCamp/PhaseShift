> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# TP-Exec-2: R9700 PCIe P2P の切り分けと peer-write transport の採用

## 目的

TP-Exec-1 の `HipPeerTpTransport` は R9700 x2 上で capability self-test に失敗し、
常に host-mediated fallback になっていた。原因が PhaseShift 側の実装か、
下層（PCIe / amdgpu / ROCm / HIP）かを切り分け、
安定して使える P2P 経路があるかを確定する。

## 条件（環境）

- GPU: R9700 (gfx1201) x2（GPU0 `07:00.0`, GPU1 `0a:00.0`）。Large BAR 32G。
- 接続: PCIe switch 経由（`lspci -tv` で同一 upstream switch の別 downstream port）。
- ROCm/HIP: HIP 7.15.26333 / clang 23。kernel `iommu=pt`。
- `amdgpu` module: `pcie_p2p=Y`, `use_xgmi_p2p=1`。
- `hipDeviceCanAccessPeer(0,1)=1`, `(1,0)=1`。
- 検証プログラムは PhaseShift / RCCL / hipMemcpyPeer を一切使わず、
  `dst[i] = src[i]` のみの micro-kernel で切り分けた（`p2p_micro` / `p2p_reduce` /
  `p2p_stream_reduce` 相当を一時的に作成して実行）。

## 結果

### Gate 3: local kernel

local fill/verify は GPU0/GPU1 とも PASS（P2P 以前の問題ではない）。

### Gate 4/5/8: direction × operation（repeat=100、size 4 KiB / 64 KiB / 1 MiB）

| direction | READ (kernel peer load) | WRITE (kernel peer store) |
|---|---|---|
| 0 -> 1 | **FAIL 100/100** | PASS 100/100 |
| 1 -> 0 | **FAIL 100/100** | PASS 100/100 |

- READ の不一致はゼロ値と「別データ」が混在し、不一致数は size・iteration ごとに変動。
  単純な一定オフセット（シフト）では説明できなかった。
- WRITE は全 size・両方向・100/100 で完全一致。

### Gate 10: hipMemcpyPeerAsync

`hipMemcpyPeerAsync` は 4 KiB で約 50%、64 KiB / 1 MiB で 100% 不一致。
DMA copy path は read 側の欠陥を引きずる。

### Gate 11: peer READ の source provenance

peer READ の source データの生成方法を変えて比較（4 KiB、repeat=100）。

| source の作り方 | peer READ |
|---|---|
| GPU1 kernel fill | FAIL 100/100 |
| GPU1 kernel fill + `__threadfence_system()` | FAIL 100/100 |
| GPU1 へ H2D (`hipMemcpy` HostToDevice) | FAIL 100/100 |
| GPU1 kernel fill → local D2D copy → その copy を read | FAIL 100/100 |

kernel が書いた L2 dirty line が原因なら H2D 初期化や D2D copy で改善するはずだが、
全 provenance で FAIL。**gfx1201 の L2 / system-scope writeback 説は否定**され、
PCIe remote-read path そのものが本命となる。

### Gate 12: SDMA の PUSH と PULL

`hipMemcpyPeerAsync` は stream がどの device に属するかで「local read + peer write」
になるか「peer read + local write」になるかが変わる。

| mode | stream | 経路 | 結果 |
|---|---|---|---|
| PUSH | source device | local read + posted peer write | **PASS 100/100**（両方向） |
| PULL | destination device | peer read + local write | **FAIL 100/100**（両方向） |

shader の peer-store が安定し peer-load が壊れるのと同じ構図。

### Gate 9: HSA_FORCE_FINE_GRAIN_PCIE

`HSA_FORCE_FINE_GRAIN_PCIE=1` を付けても READ の不一致は改善しなかった
（1 MiB で約 96% mismatch のまま）。

### Gate 7: corruption の性格

- READ: 不一致数は iteration ごとに変化（intermittent）。
- 一部ゼロ、一部は別のデータ。一定 index で再現しない。
- WRITE: 1000 回連続（10 KiB / 160 KiB / 1 MiB、両方向）で不一致ゼロ。
  leader 側であらかじめ scratch を書いて L2 に載せる coherency stress を加えても
  不一致ゼロ。

### stream + cross-device event での WRITE reduction

host sync を各 barrier で使わず、`hipStreamWaitEvent`（cross-device 可、
record は同 device event で行う）だけで peer write → local reduce → peer broadcast を
連結した場合も、10 KiB / 160 KiB / 10 MiB / 20 MiB で PASS。

### Gate 13: leader 直列型から対称 push 型へ

TP-Exec-2 の初期実装は、leader (rank0) に partial を集めて local reduce し、
broadcast する **leader 直列型 peer-write** だった。これは peer write のみを使う点で
健全だが、critical path が「peer の copy → leader の reduce → leader の broadcast」と
leader を経由して直列化し、barrier ごとの kernel 数と cross-device event 往復も多い。

peer read が使えない以上、reduction は「local read + posted peer write」だけで
構成するのが本質である。そこで leader を廃し、全 rank が自分の partial を
各 peer の inbox へ送り、受け取った inbox を各 rank が local で加算する
**対称 push 型**へ作り直した。

```text
rank r : partial_r --(peer write)--> rank q の inbox      (q != r)   全 rank 並列
rank r : local f32 add(partial_r, 受信 inbox) -> partial_r           全 rank 並列
```

- 各 rank は自分の compute stream 上で send を enqueue し、送信完了 event を record する。
- 受信側は他 rank の送信完了 event だけを `hipStreamWaitEvent` で待って add する。
- `ready` event には依存しない（stream 順序で segment 完了が保証される）。
- inbox は受信側 device に確保し、barrier ごとに parity を交互にする。

leader 直列型と対称 push 型の実測比較（features=5120, bf16, iters=50、同一ホスト）:

```text
方式           rows   payload        measured
leader 直列型  1      10240 B        185 us
leader 直列型  2048   20971520 B     1710 us / 24.5 GB/s
対称 push 型   1      10240 B        88 us
対称 push 型   2048   20971520 B     955 us  / 43.9 GB/s
```

小 payload の latency と大 payload の帯域の両方が改善したため、対称 push 型を採用する。

## 判断

判定表（READ / WRITE / hipMemcpyPeer）:

| peer READ | peer WRITE | hipMemcpyPeer | 本ホスト |
|---|---|---|---|
| FAIL | PASS | FAIL | peer load 側が壊れている |

- 原因は kernel の **peer load** 経路（および DMA copy path）にあり、
  PhaseShift 側の実装ではない。`HSA_FORCE_FINE_GRAIN_PCIE` では回復しない。
- H2D 初期化・`__threadfence_system`・D2D copy のいずれでも peer READ は壊れるため、
  L2 / system-scope writeback ではなく **PCIe の Memory Read Request →
  Completion With Data 経路そのもの**が成立していない。
- SDMA でも PUSH（local read + posted peer write）は安定し、PULL（remote read）は
  壊れる。**「他 GPU のメモリを読む経路」だけが異常**で、「他 GPU のメモリへ
  posted write する経路」は正常である。
- **kernel の peer store は安定**しているため、peer store のみで reduction を
  構成すれば正しい P2P が成立する。

採用した構成（`HipPeerTpTransport`）:

```text
rank r : partial_r --(peer write)--> rank q の inbox      (q != r)   全 rank 並列
rank r : local f32 add(partial_r, 受信 inbox) -> partial_r           全 rank 並列
```

- kernel peer load は使わない。受信側は自分の inbox を local read するだけ。
- reduction は f32 で累積し、最後に 1 回だけ bf16 へ丸める。
- 各 rank は他 rank の送信完了 event のみ待って add する。host / device sync は使わない。
- self-test は同じ peer store send + local add を 8 KiB / 128 KiB / 1 MiB ×
  4 round で実行して検証する。

## 現在への影響

- `docs/developer/tensor_parallel_execution.md` §6 が transport の現在 contract。
- `HipPeerTpTransport` は本ホストで self-test に通り、TP runtime が実際に
  P2P 経路で動作する（tiny model TP execution / Qwen3.8-27B TP=2 E2E とも PASS）。
- `phaseshift-bench tp-reduce` の実測（features=5120, bf16, iters=50）:

```text
backend        rows   payload        measured
hip-p2p        1      10240 B        88 us
hip-p2p        2048   20971520 B     955 us  / 43.9 GB/s
host-mediated  1      10240 B        130 us
host-mediated  2048   20971520 B     22700 us / 1.85 GB/s
```

- 目標（`--check`）: hip-p2p は大 payload で >= 20.0 GB/s かつ rows=1 で
  <= 150 us、host-mediated は大 payload で >= 1.2 GB/s かつ rows=1 で <= 400 us。
- 対称 push 型は小 payload（rows=1 の decode barrier）でも host 経由を上回る。
  大 payload（prefill / 大きい batch）では約 23 倍高速。
- RCCL backend は引き続き未導入。本ホストの P2P 制約は driver 依存であり、
  安定した peer read が得られる環境では同じ interface でより良い backend を
  追加できる。

## step5: TP combine の実測総時間（collective の実コスト）

`128 回/token × 88 µs ≒ 11.3 ms ≒ 42%` という推計を**実測で置き換える**ため、
GPU kernel-trace と transport bypass の 2 経路で測定した。

### 方法

1. **`rocprofv3 --kernel-trace`**（`--context 64 --tokens 128`、127 decode steps）
   transport kernel を `tp_copy_bf16_kernel` / `tp_add_peers_bf16_kernel` で識別し、
   prefill（`Grid_Size_X = 262144`）と decode（`Grid_Size_X = 5120`）を分離。
2. **transport bypass A/B**（profiler なし、5反復 interleaved）
   `TpBarrierGroup::arrive` で `transport_->sum_hidden()` のみをスキップする
   計測専用フラグを一時的に設け、同一バイナリを env で切り替えて交互に測定。
   bypass は合計されないため**正しくない実行であり、速度計測のみに使用し計測後に revert 済み**。
   bypass 実行は garbage token が EOS に当たり **`decode_steps` が 15 に早期終了**するため、
   per-step は必ず `decode_ms / decode_steps` で算出する（固定 31 と仮定すると過大になる）。

### 結果

GPU kernel-trace（127 steps）:

```text
collective 回数              : 128 / token（想定どおり、transport kernel = 512/step = 128 × 4）
transport kernel GPU 時間    : 0.464 ms / token（その agent の busy の 1.7%）
  内訳 Agent1 58.72 ms / Agent2 58.98 ms over 127 steps
```

bypass A/B（5反復 interleaved、`ms/token = decode_ms / decode_steps`）:

```text
config     transport ON   BYPASS    marginal cost   share     同方向
ctx64        26.523 ms   24.013 ms    2.509 ms      9.5%      5/5
ctx2048      27.161 ms   24.697 ms    2.465 ms      9.1%      5/5

1 collective あたり : 19.6 µs（ctx64） / 19.3 µs（ctx2048）
```

### 解釈

| 要素 | ms/token | collective 内の比率 |
| --- | ---: | ---: |
| transport kernel（GPU busy） | 0.464 | 19% |
| host enqueue + barrier + 対 rank 待ち | 2.03 | **81%** |
| 合計 | 2.49 | 100% |

- **collective のコストは帯域でも kernel でもなく、host 側が支配的**（81%）。
  payload は 20 KB で 0.5 µs 分しかなく、GPU 上の kernel も 1.9 µs/本。
- コストが ctx64 と ctx2048 でほぼ同一（2.51 / 2.47 ms）であることから、
  attention の長さには依存しない**固定の 1 回あたりコスト**である。
- 旧推計の 88 µs は `phaseshift-bench tp-reduce` が `sum_hidden` に
  **host 側 `hipStreamSynchronize` を 2 回**挟んで測った値であり、
  実行時の marginal cost 19.5 µs の約 4.5 倍だった。

### 含意（通信回数を減らす価値）

decode の wall に対する上限は以下で、**通信回数の削減は最大戦略ではない**。

| 手法 | 改善上限（decode tok/s への上乗せ） |
| --- | ---: |
| collective を全消 | +9.1%（36.8 → 40.5 tok/s） |
| **collective を半減**（2/layer → 1/layer、schedule 再構成が要る） | **約 +4.6%** |
| **host 側コストのみ除去**（schedule 不変、数行の修正） | **約 +7.7%** |

つまり **「回数を減らす」より先に「1 回あたりの host コストを下げる」の方が
見返りが大きく、改修も軽い**。具体的な候補:

- `TpBarrierGroup::arrive` が毎 barrier で `hipEventRecord(ready_)` を行うが、
  `HipPeerTpTransport::enqueue_sum` は `invocation.ready` を参照しない
  （対称 push 化で stream 順序で足りる）→ **未使用の event record 2 回/barrier**
- `arrive` と `sum_hidden` の `ScopedDevice::create`（`hipGetDevice` + `hipSetDevice`
  が計 6 回/barrier）。呼び出し元の worker thread は thread 入りで device 設定済み

### 次

- host 側コストの削減を A/B で実測（code は数行、correctness は token 一致で確認）
- それでも足りなければ overlap（`docs/developer/tensor_parallel_execution.md` §10 の
  「通信と計算の本格的な overlap は未対応」）を検討
- 通信回数そのものの削減（2 collective/layer → 1）は上記を先に済ませてから

## step6-A: 未使用の ready event とその device switch を削除

step5 の結果（collective 19.5 µs/回のうち **81% が host 側**）を受けて、
hot path から**証明済みの死んでいる作業**だけを削除した。

### 変更

`TpBarrierGroup::arrive`（毎 barrier × 毎 rank）:

```text
- hipEventRecord(ready_[index], stream)
- それを伴う ScopedDevice::create
```

安全根拠: `invocation.ready` を読む実装は
`tp_transport_internal.h:94` の **size 検証のみ**で、
`HipPeerTpTransport` / `HostMediatedTpTransport` とも実イベントを参照しない。
`arrive` の残りは host 処理と `sum_hidden`（内部で独自に device を切る）だけなので
device switch も不要になる。

削除分は 1 barrier × 2 rank = `hipEventRecord` 1 本 + `hipGetDevice` / `hipSetDevice` 各 2 本 = **6 HIP API 呼び出し**。

### A/B（同一バイナリを 2 種に分け、5反復 interleaved、`decode_ms / decode_steps`）

| config | metric | base p50 | cand p50 | delta | spread(base/cand) |
| --- | --- | ---: | ---: | ---: | --- |
| ctx64 | decode | 26.494 ms/token | **25.848** | **+2.44%** | 0.3% / 0.3% |
| ctx2048 | decode | 27.148 ms/token | **26.487** | **+2.44%** | 0.2% / 0.2% |
| ctx64 | prefill | 58.241 ms | 57.930 | +0.53% | 7.3% / 3.5%（ノイズ） |
| ctx2048 | prefill | 574.812 ms | 572.946 | +0.32% | 0.6% / 0.4% |

- **decode は両 config で +2.44%、paired 5/5 同方向、spread 0.3%** → 有意。
- 削減量は **0.645〜0.661 ms/token = 5.0〜5.2 µs per collective**。

### 何が効いたか（step6-B での切り分け）

`ON − bypass = sum_hidden` の値は **2.465 ms のまま**で、step6-A 前後で変化しない
（`arrive` の作業は bypass 時にも残るため、この差分には入らない）。
つまり step6-A が削ったのは `sum_hidden` ではなく **`arrive` 側の overhead 0.661 ms/token** であり、
削減量は

```text
0.661 ms/token ÷ 128 barrier ÷ 2 rank = 2.6 µs / rank / barrier
```

（1 barrier × 2 rank × `hipEventRecord` 1本 + `ScopedDevice` の
`hipGetDeviceCount` / `hipGetDeviceProperties` / `hipGetDevice`）。

step6-B（下記）で `hipGetDeviceCount` と `hipGetDeviceProperties` のみを
`sum_hidden` の hot path から除いたところ **効果は 0%** だったため、切り分けは次のとおり:

| 削除対象 | 効果 |
| --- | --- |
| `hipEventRecord`（`arrive` 内、2本/barrier） | **約 2.6 µs/本 ≒ 0.66 ms/token（+2.44%）** |
| `hipGetDeviceCount` / `hipGetDeviceProperties` | **0（cache 済みで実質無料）** |

よって **`hipEventRecord` が host 側の支配的コスト**で、他の HIP API 呼び出しは安い。

### step6-B: ScopedDevice を最小 switch に置換（**REJECT**）

`ScopedDevice::create` は毎回 `hipGetDeviceCount` + `hipGetDeviceProperties`
（`gcnArchName` 評価）+ `hipGetDevice` + `hipSetDevice` + destructor での復元を実行し、
`HipPeerTpTransport::enqueue_sum` は 1 barrier で 4 回呼ぶ。
transport は `create()` 時に device 数と gfx1201 を検証済みなので、
hot path の再検証は不要と考え、検証済み前提の最小 switch `TpDeviceSwitch` に置換した
（1 barrierあたり 8 HIP API 呼び出しを削除できるはずだった）。

5反復 interleaved の結果:

| config | metric | step6-A | step6-B | delta |
| --- | --- | ---: | ---: | ---: |
| ctx64 | decode | 25.800 ms/token | 25.810 | **−0.04%** |
| ctx2048 | decode | 26.484 ms/token | 26.471 | **+0.05%** |

paired は 1/5・3/5 で**ノイズ帯（spread 0.2〜0.3%）**。→ **REJECT して revert 済み**。
`hipGetDeviceCount` / `hipGetDeviceProperties` は cache されており無料だった。

### step6 後の collective コスト（現状）

```text
sum_hidden（transport 本体）      : 2.45 ms/token ≒ 19.1 µs/collective ≒ wall の 9.2%
  内訳のうち判明している大物      : hipEventRecord 2本 ≒ 5.2 µs/collective
arrive の overhead                : 0（step6-A で除去済み）
```

`sum_hidden` の残り 14 µs は、kernel launch 4本 + `hipEventRecord` 2本 +
device switch + condvar で、**cross-device 順序のための `hipEventRecord` は構造上不可避**。
つまり **host 側の削減は限界に近い**。

### gate

- correctness: `test_tp_reduction` / `test_tp_transport` / `test_qwen35_tp_execution` /
  `test_qwen35_tp_e2e` 4/4 PASS
- required acceptance: **124/124 PASS**（failure 0 / skip 0 / error 0）

## step7: collective の host critical path を分解し、overlap 対象を特定する

### 方法

`sum_hidden` と `arrive` に一時 timestamp を入れて実測（計測後 revert 済み）。
対象は TP2 / ctx2048 / decode。

```text
TPSUM     calls=8192   validate=0.02  ensure=0.92  phase1_copy=4.89
                          phase2_waitadd=5.47   total=11.31 µs/call
TPARRIVE  sum_calls=8192  sum=11.36 µs          wait_calls=8192  wait=22.50 µs
TPARRIVE  skew=7.82 µs（最初と2番目の rank の arrive 入口の差）
```

### 分解

1 barrier = 128 barrier/token（64 層 × 2）。`wait` 22.50 µs は

```text
skew（2 番目の rank が到着するまで）      7.82 µs  (35%)
peer の sum_hidden（両 stream 分の enqueue）11.52 µs  (51%)
condvar の wakeup / mutex 再獲得            3.78 µs  (16%)
```

に分解できる。`sum_hidden` 自体は HIP API 8 本（launch_copy×2 +
hipEventRecord×2 + hipStreamWaitEvent×2 + launch_add×2）≒ 10.5 µs と
`ensure_buffers` 0.97 µs で、**1 API ≒ 1.3 µs** の呼び出しコストが実体。
`validate` は 0.02 µs で無視できる。

### host 時間は wall に1:1で乗る（摂動実験）

`arrive` の入口に sleep を入れて decode を測った（64 tokens / 2 反復）:

| perturb | 実効 sleep | decode_ms |
| ---: | ---: | ---: |
| 0 µs | − | 1590.6 / 1591.6 |
| 5 µs | 約 11 µs | 1770.5 / 1778.6（**+180 ms**） |
| 10 µs | 約 16.6 µs | 1862.6 / 1863.4（**+272 ms**） |

64 tokens × 128 barrier × 2 rank = 16384 個の `arrive` 呼び出しで
`16384 × 1 µs = 16.4 ms` がそのまま増分に一致する。
つまり **`arrive` 内の host 時間は wall に 1:1 で加算される**。

`arrive` の host 時間は 1 barrier あたり 11.36 + 22.50 = 33.9 µs、
1 token あたり約 4.3 ms（wall 25.4 ms の 17%）。
**つまり collective の host 削減量はそのまま decode 改善量になる。**

### 残っている削減対象と試算

| 手法 | 削減/barrier | 1 token | wall 比 | 段階 |
| --- | ---: | ---: | ---: | --- |
| `sum_hidden` を prepare / finalize に分割し、
barrier を prepare 後へ前倒し（各 rank が自分の stream だけ enqueue） | 約 9 µs | 約 1.2 ms | **約 4.7%** | 未着手 |
| condvar → spin-then-block barrier（wakeup 3.78 µs を除去） | 約 3.3 µs | 約 0.4 ms | 約 1.7% | 未着手 |
| `ensure_buffers` を barrier 毎に呼ばない | 約 1.0 µs | 約 0.1 ms | 約 0.5% | 未着手 |
| skew 7.82 µs の原因解明（rank 間の enqueue 直列化） | 最大 7.8 µs | 最大 1.0 ms | 最大 3.9% | 未着手 |

skew は「両 host スレッドが HIP enqueue で直列化している」可能性が高い
（16 core / load 1.27 で affinity 設定なし、`sum_hidden` の HIP API 8 本が
その直列化区間にあたる）。仮説の段階であり未検証。

### 既存の結論との関係

step6-B は「`hipEventRecord` が支配的、host 側の削減は限界に近い」としたが、
これは **`sum_hidden` 内部**だけを見た结论だった。
barrier 全体（`wait` 22.5 µs を含む）で見ると、**支配的なのは peer の
`sum_hidden` を待つ側の時間**であり、構造を変える（各 rank が自分で enqueue）ことで
約 9 µs/barrier が取れる。すなわち「限界」は現構造に対してのみ成り立つ。

## step12: `sum_hidden` を prepare / finalize に分割（**REJECT**）

step7 の分解結果（`wait` 22.50 µs の 51% が「peer の `sum_hidden` を待つ時間」）に基づき、
collective の host 作業を各 rank に配る分割を 2 案実装して A/B した。**どちらも効果ゼロだった。**

### 実装した 2 案

`TpTransport` に `sum_prepare(invocation, rank)` / `sum_finalize(invocation, rank)` を追加し、
`TpBarrierGroup::arrive` を「prepare → barrier → finalize」に組み替える。
parity は transport 内部カウンタから `TpSumInvocation::barrier_index`（`open_barrier_`）へ移設した。

- **案 A**: prepare / finalize 両方を barrier のロック内で実行
- **案 B**: prepare のみロック内、**finalize をロック外へ**

`sum_hidden` は base class の非仮想関数（prepare 全 → finalize 全）として残し、
bench / test からは従来どおり呼べる。`HostMediatedTpTransport` も同じ分割で実装した。

### 計測（案 B、TP2 / ctx2048 / decode）

```text
旧 : sum_hidden=11.36 µs  wait=22.50 µs  skew=7.82 µs   → A span 22.50 µs
新 : prepare=2.85 wait=14.09 finalize=2.98 skew=11.01   → A span 19.92 µs
                （prepare・finalize は各 16384 call、wait は 8192 call）
```

`A span` は 22.50 → 19.92 µs（−2.58 µs）に縮んだように見える。
しかし **両 rank の own work 合計は 11.36 → 11.66 µs でむしろ微増**している。
`wait` が縮んだ分だけ `skew`（7.82 → 11.01）が増え、**作業を減らさず再配分しただけ**だった。
（案 B では `ensure_buffers` が barrier ごとに 2 回呼ばれる点も増加要因。）

### A/B（baseline = step8 状態、5反復 interleaved）

| 案 | ctx64 decode | ctx2048 decode | spread |
| --- | ---: | ---: | --- |
| A | +0.04%（2/5） | −0.18%（3/5） | 0.1〜0.5% |
| B | −0.08%（4/5） | +0.06%（2/5） | 0.1〜2.1% |

**いずれも run 間変動の範囲内 = ノイズ**。設計時の試算（−2.58 µs/barrier ≒ −1.3%）は出ない。

### 摂動実験の再実行（案 B 上で）

| perturb | decode_ms（64 tokens × 2 反復） |
| ---: | --- |
| 0 µs | 1596.3 / 1625.0 |
| 5 µs | 1770.7 / 1772.1（+165 ms） |
| 10 µs | 1843.0 / 1844.4（+235 ms） |

感度は 1:1 のまま維持している。つまり **host 時間は wall に効くが、削減ではなく追加に対してだけ**。

### コストモデルの訂正

step7 で「`wait` の 51% が peer の `sum_hidden` 待ちなので分割すれば取れる」としたが、
これは **待ち時間の構造**だけを見て、**barrier ごとの実作業量**を見ていない誤りだった。

- 摂動実験が1:1に出るのは、`arrive` に **実作業を追加**しているから。
- 分割は作業量を変えないので効かない。wall の律速は「何 µs 待つか」ではなく
  **1 barrier あたりの HIP API 呼び出し数**（両 rank 合計 約 11.4 µs ≒ 12 呼び出し:
  `launch_copy`×2 / `hipEventRecord`×2 / `hipStreamWaitEvent`×2 / `launch_add`×2 /
  `ScopedDevice`×4 + `validate`/`ensure_buffers`）。
- したがって overlap 構造の再編では取れず、**呼び出し数そのものを減らす**しかない。

### 残っている削減候補（未着手）

| 手法 | 削減/barrier | wall 比 | リスク |
| --- | ---: | ---: | --- |
| `hipEventRecord`×2 + `hipStreamWaitEvent`×2 を除去し、
copy kernel の flag 書き込み + `hipStreamWaitValue32` に置換 | 約 2.6 µs | 約 1.3% | 低（`stream_write_value32` の使用実績あり） |
| 同上を kernel 側 polling で完全に除去（host 呼び出し 0） | 約 5.2 µs | 約 2.6% | 中（spin 中の SM 占有、peer 停滞時の挙動） |
| `ensure_buffers` を barrier 毎呼ばない | 約 0.97 µs | 約 0.5% | 低 |

### gate と最終状態

- 案 B 段階で `test_tp_reduction` / `test_tp_transport` / `test_qwen35_tp_execution` /
  `test_qwen35_tp_e2e` **4/4 PASS**（E2E の TP1/TP2 token 一致含む）。
  効果ゼロのため required acceptance は実施せず **revert**。
- 最終状態は HEAD `a91dd715`（本 step の変更なし）。prototype は R&D 記録のみ残す。
