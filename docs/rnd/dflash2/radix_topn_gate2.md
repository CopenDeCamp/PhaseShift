# Radix Select Top-N selector（DFlash2 coarse Top-N の置換）

> Status: R&D record（Gate の経緯・採否判断。現在の contract は
> `docs/developer/dflash2.md`、現在の性能値は `docs/perf/current.md` を正本とする）

対象は Gate 2 である。前提となる Gate 1（INT2 proposal head の再 qualification）は
[../dflash2/int2_head_gate1.md](../dflash2/int2_head_gate1.md) を参照。
計測条件も同文書の §1 を共有する。

## 3. Gate 2: Radix Select Top-N

### 3.1 実装したもの

- `src/phaseshift/models/qwen35/kernels/dflash2/radix_topn.hip`
  - `launch_dflash2_radix_topn()`（新規 public launcher）
  - `dflash2_radix_topn_scratch_bytes()`（init 時にサイズ決定、runtime で hipMalloc しない）
  - `dflash2_radix_topn_preferred(pool)`（切替判断）
- `tests/unit/test_dflash2_radix_topn.hip`（correctness、required）
- `tests/unit/test_dflash2_radix_topn_perf.hip`（性能専用、label `gpu1;perf`）
- 現行 `launch_dflash2_coarse_topn()` は削除せず oracle / fallback として残置。

構成は full sort ではなく **threshold selection → exact N candidate extraction →
small N sort**。

1. float32 ordered bits の 4 pass radix（bits 31:24 → 7:0）。各 pass は
   `rows × partitions` の histogram kernel + `rows` の select kernel。
   prefix を満たす element だけを histogram する。
2. 4 pass 後に N 番目の exact float32 threshold `T` が確定する。
3. `count_gt`（T より大）と `count_eq`（T と同値）を出し、`need = N - count_gt`。
   `count_eq > need` の場合のみ threshold と同一 float の element を対象に
   **token ID の radix selection** を行う（1 走査で vocab 幅の bitmap を作り、
   `need` 番目に小さい ID を選ぶ）。同値が起きない通常ケースでは追加 pass が発生しない。
4. `score > T`、および `score == T` かつ ID threshold を通過した element のみを
   candidate buffer（= 出力 buffer）へ atomic 追加で compact。件数は正確に N。
5. `next_pow2(N)` 個を bitonic で key 降順に sort して出力。

ordering contract は `detail::topn_key()` と完全互換:
score 降順、同値は token ID 昇順、`+0/-0` は同値、NaN は現行と同じ最下位、
`-INFINITY` は現行と同じく除外、不足分の tail は `(id=0, -INFINITY)`。

### 3.2 correctness

- `test_dflash2_radix_topn` = **432/432 PASS**
  - CPU oracle（`topn_key` 相当の順序）と IDs exact / logits bit exact
  - legacy `launch_dflash2_coarse_topn()` と一致
  - 出力 unique、2 回実行で決定論的
  - pool 16/24/32/48/64/80/128 × vocab 1024/4096/248320 × rows 1/3/7 × random/tie
  - special: all equal / +0 -0 mix / duplicate max / repeated threshold /
    `-INFINITY` 混入 / NaN 混入 / vocab < partition size
- `ctest -L required` = **125/125 PASS**（GPU3 に限定して実行。GPU0 は別 workload が
  VRAM を占有しており、これは環境要因）

### 3.3 standalone performance（batch 方式: warmup 100、200 launch/batch、24 batch、median）

`vocab=248320`、random 入力、GPU3、単位 µs。p10/p90 の開きは全行 1% 未満
（`gate2_standalone_perf.txt` 参照）。

rows=1（Target Decode の主戦場）:

| pool | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 30.8 | 40.1 | 49.0 | 66.2 | 84.5 | 121.2 | 157.8 |
| radix | 84.0 | 82.8 | 87.1 | 88.0 | **82.4** | 89.7 | 88.2 |
| speedup | 0.37x | 0.48x | 0.56x | 0.75x | **1.03x** | 1.35x | 1.79x |

rows=8（DFlash2 の主戦場）:

| pool | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 84.2 | 107.9 | 132.1 | 178.7 | 226.7 | 323.8 | 421.0 |
| radix | 93.1 | 93.8 | 94.2 | 95.3 | **95.2** | 203.9 | 213.5 |
| speedup | 0.90x | **1.15x** | 1.40x | 1.88x | **2.38x** | 1.59x | 1.97x |

観測:

- radix の cost は pool に対してほぼ一定（rows=1 で 82〜89 µs、rows=8 で 93〜95 µs）。
  legacy は pool に比例して増える。
- **crossover は rows 依存**: rows=1 では pool 64、rows=8 では pool 24 付近。
- rows≥4 かつ pool≥96 の行で radix が 200 µs 級に跳ねるが、これは random 入力に
  rank 96 / 128 の位置へ実在した同値があり、ID tie-break（1 block による vocab 走査）が
  発動したためである（CPU 上で入力を再現して確認済み）。
- tie-heavy 入力では全行 190〜211 µs。**tie path の追加コストは約 105 µs**。
  実 pipeline の INT2 coarse logits では tie はほぼ起きない（§3.4）。

### 3.4 DFlash2 実 pipeline での selector latency（`INT2_TIMING=1`、rows=7）

| pool | legacy topn | radix topn | pipe legacy | pipe radix |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.120 ms | 0.118 ms | 1.165 ms | 1.165 ms |
| 64 | 0.204 ms | **0.120 ms** | 1.251 ms | **1.169 ms** |
| 80 | 0.246 ms | **0.121 ms** | 1.300 ms | **1.174 ms** |

radix の latency が pool によらず 0.118〜0.121 ms で一定であることから、
**実行中の INT2 coarse logits では tie path が発生していない**（発生していれば
平均が 0.1 ms 以上跳ねる）。

### 3.5 DFlash2 E2E（interleaved、5 rep、GPU3）

| tag | tok/s 中央値 | rounds | accepted | IDS |
| --- | ---: | ---: | ---: | --- |
| legacy pool64 | 66.67 | 84 | 171 | `47aebe55d048` |
| **radix pool64** | **66.80** | 84 | 171 | `47aebe55d048` |
| legacy pool32 | 66.00 | 85 | 170 | `47aebe55d048` |
| radix pool32 | 66.02 | 85 | 170 | `47aebe55d048` |

- pool64 は **5/5 rep で radix が legacy を上回る**（+0.19%）。
- pool32 は同値（selector latency も 120 vs 118 µs で同値）。
- rounds / accepted / GENERATED_IDS は legacy と完全一致 → **E2E regression なし**。

### 3.6 切替 rule

`dflash2_radix_topn_preferred(pool)`:

- `PHASESHIFT_DFLASH2_RADIX_TOPN=0` → 常に legacy
- `PHASESHIFT_DFLASH2_RADIX_TOPN=1` → 常に radix
- 未指定 → `pool >= kDflash2RadixTopnCrossoverPool`（= 64）で radix

`kDflash2RadixTopnCrossoverPool = 64` は rows=1 の crossover（legacy 84.5 µs vs
radix 82.4 µs）に基づく。rows=8 では pool≥24 で radix が勝つが、DFlash2 の
BEST_POOL=64 はいずれの根拠でも radix 側である。

### 3.7 Gate 2 判定

- CPU oracle exact / legacy exact / required PASS / DFlash2 IDS 一致 / E2E regression なし /
  実用 pool で selector latency 改善 → **GO**
- ただし rows・pool で勝者が分かれるため結論は **HYBRID**:
  - rows=1（Target Decode）: pool ≤ 48 は legacy、pool ≥ 64 が radix
  - rows=7/8（DFlash2）: pool ≥ 24 で radix

### 3.8 未実施・既知の制約

- tie path は 1 block による vocab 走査（+105 µs）。実 workload で tie が頻出する
  model が出た場合は、compact 段で eq element を収集して解決する方式へ
 再設計する必要がある。
- coarse head と Radix の fusion は行わない（過去に悪化履歴あり、本 Gate でも対象外）。
