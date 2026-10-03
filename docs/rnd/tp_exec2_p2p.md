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

## 判断

判定表（READ / WRITE / hipMemcpyPeer）:

| peer READ | peer WRITE | hipMemcpyPeer | 本ホスト |
|---|---|---|---|
| FAIL | PASS | FAIL | peer load 側が壊れている |

- 原因は kernel の **peer load** 経路（および DMA copy path）にあり、
  PhaseShift 側の実装ではない。`HSA_FORCE_FINE_GRAIN_PCIE` では回復しない。
- **kernel の peer store は安定**しているため、peer store のみで reduction を
  構成すれば正しい P2P が成立する。

採用した構成（`HipPeerTpTransport`）:

```text
rank r>0 : partial --(peer store)--> leader scratch
leader   : local f32 reduce(partial, scratch) -> out / broadcast buffer
leader   : broadcast buffer --(peer store)--> rank r buffer
```

- kernel peer load は使わない。
- reduction は f32 で累積し、最後に 1 回だけ bf16 へ丸める。
- leader は reduction 完了 event のみ待ち、broadcast は transport stream 上で
  overlap する。
- self-test は同じ peer store + reduce + broadcast を 8 KiB / 128 KiB / 1 MiB ×
  4 round で実行して検証する。

## 現在への影響

- `docs/developer/tensor_parallel_execution.md` §6 が transport の現在 contract。
- `HipPeerTpTransport` は本ホストで self-test に通り、TP runtime が実際に
  P2P 経路で動作する（tiny model TP execution / Qwen3.8-27B TP=2 E2E とも PASS）。
- `phaseshift-bench tp-reduce` の実測（features=5120, bf16）:

```text
backend        rows   payload        measured
hip-p2p        1      10240 B        185 us
hip-p2p        2048   20971520 B     1710 us  / 24.5 GB/s
host-mediated  1      10240 B        127 us
host-mediated  2048   20971520 B     22000 us / 1.9 GB/s
```

- 目標（`--check`）: hip-p2p は大 payload で >= 8.0 GB/s かつ rows=1 で
  <= 250 us、host-mediated は大 payload で >= 1.2 GB/s かつ rows=1 で <= 400 us。
- 小 payload（rows=1 の decode barrier）では P2P の kernel 数と
  cross-device event 往復のため host 経由と同程度〜やや遅い。大 payload
  （prefill / 大きい batch）では約 13 倍高速。
- RCCL backend は引き続き未導入。本ホストの P2P 制約は driver 依存であり、
  安定した peer read が得られる環境では同じ interface でより良い backend を
  追加できる。
