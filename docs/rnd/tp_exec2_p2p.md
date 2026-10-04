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
