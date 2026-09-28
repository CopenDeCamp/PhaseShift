# ffn_k256_break_even

Qwen3.8-27B FFN `down_proj` decode1 (N=5120, K=17408, M=1) について、
N64 × K256 の K-group skip を **実際に weight load / decode / WMMA から消した** ときの
latency saving と、selector に使える break-even budget を測る benchmark。

production kernel は変更しない。benchmark 専用 kernel を別途持ち、DENSE / IDEAL_STATIC_SKIP /
RUNTIME_MASK_SKIP / MASK_READ_ONLY の 4 経路を比較する。

## 構成

| file | 役割 |
| --- | --- |
| `bench_sparse_down.hip` | PSQ4 / PSQ8 decode1 の dense と sparse 変種 + timing driver |
| `build.sh` | `bench_sparse_down` を hipcc (clang++) で単体ビルド |
| `gen_oracle_masks.py` | 前回 Oracle の PAIR_GREEDY_SWAP で layer ごとの N64 mask を dump |
| `analyze_break_even.py` | microbench JSON から 64 layer / E2E projection と budget を算出 |

## kernel 設計

- 1 block = N16 output tile (32 threads, 2 k_group)。N64 tile は blockIdx.x>>2。
- K256 = 8 × K32。`for g in 0..67` の外側で mask を 1 回判定し、KEEP 時のみ内側 8 反復を実行。
- mask は N64 tile ごとに 3 × uint32 (68 bit)。runtime 経路は block 先頭で 3 word を load。
- SKIP 時は weight code / scale load・decode・WMMA を本当に実行しない。zero weight を読む方式ではない。
- IDEAL は mask load なしの contiguous prefix skip（branch のみ）。

## working set

同じ 50MB weight を毎 launch 再利用すると L3 に載り、DRAM 帯域ではなく L3 帯域を測ってしまう。
そのため weight buffer を `--copies` 個用意し、launch ごとに別 copy を参照して working set を
L3 (64MB) より大きくする。既定 copies=4（PSQ4 で約 200MB、PSQ8 で約 380MB）。

## correctness

skip 対象 group の weight code / scale を 0 にした buffer を production kernel で計算したものを
reference とし、RUNTIME_MASK_SKIP の出力と比較する（zero code は寄与 0 なので「指定 group を
除去した」厳密な reference になる）。全 skip / pattern で max abs diff = 0。

## 実行

```bash
bash tools/rnd/ffn_k256_break_even/build.sh
ROCM=/opt/zen/.venv/lib/python3.12/site-packages/_rocm_sdk_devel
LD_LIBRARY_PATH=$ROCM/lib HIP_VISIBLE_DEVICES=2 \
  ./tools/rnd/ffn_k256_break_even/bench_sparse_down --kind psq4 --copies 4 \
  --out artifacts/ffn_k256_break_even/dense_psq4.json \
  --skips 0,7,10,14,16,17,20,27,28,30
```

`--kind psq8` で PSQ8。`--mask-dir <dir>` を付けると `oracle_mask_k<k>.txt` を読む
`oracle` pattern が加わる。
