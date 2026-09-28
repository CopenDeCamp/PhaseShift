# GPU clock residency PoC — Gate 0–2 and CP-keepalive isolation

CP KeepAlive による Prefill 改善が「GPU の clock / power residency 維持」なのか
「CP/queue 側の power state」なのかを切り分ける PoC の記録である。branch
`poc/gpu-clock-residency`。環境は
[gpu_clock_residency_environment.md](gpu_clock_residency_environment.md) を参照。

方針は「KeepAlive 実装より idle threshold の特定を優先」である。本記録は Gate 0–2 と、
CP だけを起こしてコアを idle にしたままにする分離実験（Gate 4 の前段）までを含む。

## Revision

| 項目 | 値 |
| --- | --- |
| baseline | `3b8d6f6d` (main) |
| branch | `poc/gpu-clock-residency` |
| bench | `/opt/zen/wk/PhaseNonShift/build/phaseshift-bench` |
| probe | `tools/poc/clock_residency/clock_probe.hip`（standalone, hipcc gfx1201） |
| measurement | 2026-09-25 |

## 1. 追加した probe harness

`clock_probe` は既存 runtime / CMake を変更せず、`hipcc` で直接ビルドする独立 PoC である。

```sh
hipcc -O3 --offload-arch=gfx1201 -Wno-unused-value \
  -o clock_probe tools/poc/clock_residency/clock_probe.hip
```

- `--mode timing`: prime → `--idle-us N` 待機 → probe を実行し、GPU event で
  `gap_us`（prime 終了〜probe 開始）と `probe_us`（probe 開始〜終了）、in-kernel
  `exec_us`（probe kernel 内の `wall_clock64()` 差分）を測る。idle 中の GPU work はゼロ。
- `--mode observe`: 各 idle 直後に `freq1_input`(sclk) / `freq2_input`(mclk) /
  `power1_average` を読む。
- `--keepalive-us N`: idle を含む全期間、別 stream に `hipEventRecord` を N µs ごとに流す
  （カーネル無し。CP packet + timestamp write のみ）。

測定量の意味:

- `gap_us - idle` = CPC/queue の wake-up penalty（固定費）。
- `probe_us - exec_us` = dispatch 費。
- `exec_us` = kernel 実行そのもの（clock 感度の直接指標）。

## 2. Gate 2: idle gap sweep

### 2.1 exec（実行時間）は clock に依存しない

device 1 (card2)。idle 0 と 100 ms の比較（`exec_us` median）:

| probe | idle 0 | idle 100 ms | 差 |
| --- | ---: | ---: | ---: |
| compute 512 iters | 4.84 µs | 5.08 µs | +5% |
| compute 2048 iters | 18.08 | 18.64 | +3% |
| compute 16384 iters | 135.0 | 135.7 | +0.5% |
| mem 128 MB | 51.84 | 51.80 | 0% |
| mem 512 MB | 207.2 | 208.8 | +0.8% |

idle 100 ms では sclk が ~0.1 GHz まで落ちている（§2.3）。それでも **kernel 実行時間は
変化しない**。wake/dispatch の間に clock が復帰するためである。

### 2.2 gap（wake-up penalty）は 1–5 ms で立ち上がり飽和する

`gap_us - idle`（device 1, compute 512、`hwm` 無し）:

| idle | gap − idle |
| ---: | ---: |
| 0 | ~28 µs（host enqueue 込みの基準） |
| 500 µs | +27 |
| 1 ms | +38 |
| 2 ms | +138 |
| 5 ms | +305 |
| 8 ms | +304 |
| 12 ms | +304 |
| 20 ms | +306 |
| 50 ms | +304 |
| 100 ms | +308 |

onset は **1–2 ms**、飽和は **5 ms**。probe 種別（compute 512 / 2048 / 16384、
mem 128 MB / 512 MB）で同じ形状である。`probe_us - exec_us`（dispatch 費）も
1 ms 以降で +10 µs 程度増えるが、exec には現れない。

### 2.3 sclk / mclk residency

`observe`（device 3, compute、idle 直後の sclk median 近傍）:

| idle | sclk |
| ---: | ---: |
| 1 ms | 2.35–2.44 GHz |
| 2–5 ms | 2.82–3.03 GHz |
| 6 ms | 2.70–2.93 GHz |
| 8 ms | 2.31–2.43 GHz |
| 12 ms | 2.07–2.12 GHz |
| 20 ms | 1.75–1.79 GHz |
| 30 ms | 1.53–1.55 GHz |
| 100 ms | 0.10–1.09 GHz（run 依存） |

- **GFX/SCLK は 5–8 ms から下がり始め、10–30 ms かけて徐々に低下する。**
- MCLK: memory load をかけると 1258 MHz に上がり、その後 **100 ms idle でも 1258 MHz のまま**
  （drop の時定数 > 100 ms）。
- 1–5 ms の gap penalty は sclk がまだ ~2.9 GHz のうちに飽和している。したがって
  **penalty は sclk DPM 低下では説明できない**。

### 2.4 検出した threshold

| domain | 変化 | threshold |
| --- | --- | --- |
| CP/queue wake penalty | gap 増加 | onset 1–2 ms, 飽和 5–8 ms |
| GFX/SCLK | sclk 低下 | onset 5–8 ms, 100 ms で ~0.1–1.1 GHz |
| MCLK/UCLK | mclk 低下 | >100 ms（mem load 後） |
| kernel exec | 変化なし | — |

## 3. CP 以外のコアを idle にしたままの分離実験（device 3）

`--keepalive-us` はカーネルを一切流さず、event record（CP packet + timestamp write）だけを
周期的に発行する。CU は動かさない。

`gap_us - idle`:

| idle | keepalive off | 500 µs | 1000 µs |
| ---: | ---: | ---: | ---: |
| 1 ms | +37 | +28 | +31 |
| 2 ms | +133 | +30 | +31 |
| 5 ms | +398 | +30 | +33 |
| 8 ms | +405 | +30 | +31 |
| 20 ms | +393 | +31 | +32 |
| 100 ms | +398 | +33 | +38 |

**~400 µs の wake penalty が完全に消える**。`exec_us` は全条件で ~5.6 µs と平坦。

さらに keepalive ON では idle 中の sclk が高いまま維持される（device 3, observe）:

| idle | keepalive off | keepalive 1000 µs |
| ---: | ---: | ---: |
| 5 ms | ~2.6 GHz | ~2.9 GHz |
| 20 ms | ~1.8 GHz | ~2.93 GHz |
| 100 ms | ~1.09 GHz | ~2.94 GHz |

すなわち **event record だけで sclk 低下と wake penalty の両方が同時に防げる**。CU load は
不要であり、「別途 Clock KeepAlive カーネルを実装する」必要はない。

## 4. Gate 1: CP KeepAlive の対照実験

`phaseshift-bench pp --prompt-tokens 2048 --mode forward --warmup 20 --runs 100`。
CP KA OFF は `PHASESHIFT_KEEPALIVE=0`、ON は executor 既定（1000 µs）。

device 1 (card2、interleaved off/on/on/off):

| case | prefill median | p10 | p90 | stddev | t/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| dynamic_cp_off_1 | 916.27 ms | 907.88 | 952.80 | 18.89 | 2235.14 |
| dynamic_cp_on_1 | 908.02 | 905.43 | 909.80 | 1.73 | 2255.46 |
| dynamic_cp_on_2 | 909.42 | 907.24 | 911.33 | 1.60 | 2251.98 |
| dynamic_cp_off_2 | 921.15 | 918.94 | 955.84 | 22.59 | 2223.31 |

- CP KA ON は +0.9%（pair1）/ +1.3%（pair2）、平均 **+1.1%**。
- 効果以上に **分散が劇的に小さい**（stddev 19–23 → 1.6–1.7）。OFF は run 途中で
  ~919 ms から ~955 ms へ段状に悪化する二峰性を示す。
- 本セッションの +1.1% は `docs/rnd/runtime/execution_overhead.md` §7.89 の +5.45%
  より小さい。GPU の power/clock 状態が異なる（本機は load 時 ~200 W / sclk ~3.3 GHz）。

`performance level` 固定（C/D, `power_dpm_force_performance_level=high`）は root が必要なため
`tools/poc/clock_residency/gate1_fixed.sh` を用意したが、§3 の分離実験で機構を特定できたため
本 PoC の判定には使用しない（未実施）。

## 5. 判定

Acceptance criteria との対応:

- **Result E に該当する。** idle sweep に clock-related な cliff は無い。gap の cliff は
  1–2 ms で立ち上がり 5 ms で飽和し、sclk 低下（5–8 ms 以降）と時間スケールが合わない。
- CP のみを起こす event-record keepalive で penalty が完全に消える。CU は不要。
- exec は clock level に依存しない。

つまり「Prefill の gap 悪化は、コア（GFX/SCLK）が冷えることの復帰コスト」ではなく、
**CP/queue が低活動状態から復帰する固定 wake latency** である。GFX/SCLK 低下は
同じ低活動状態の相関した症状であり、CP を起こし続ければ同時に防げる。

Clock/power residency hypothesis: **NOT SUPPORTED**（penalty の主因としては）。
Primary suspected domain: **CP/queue**。Observed idle threshold: CP/queue 1–2 ms（飽和 5–8 ms）、
GFX/SCLK 5–8 ms、MCLK >100 ms。

## 6. Recommendation

- **GO（CP 側のみ）**。既存 executor の event-record CP KeepAlive で足りる。Clock KeepAlive
  カーネルは実装しない。
- Gate 3（GFX/MEM 分離）は MCLK が 100 ms で冷えないこと、event-record のみで penalty が
  消えることから不要。
- Gate 1 C/D（performance level 固定）は分離実験で機構が特定できたため不要。

## 7. Raw result

`artifacts/poc_clock_residency/`（gitignore 対象、ローカル）:

```
gate2/compute_tiny.csv        gate2/compute_2048.csv
gate2/compute_medium.csv      gate2/mem_short.csv        gate2/mem_medium.csv
gate4/tiny_ka0.csv            gate4/tiny_ka500.csv       gate4/tiny_ka1000.csv
gate1/dynamic_cp_off_1.txt ...（device 1）
gate1_dev3/...（device 3）
```

再現:

```sh
tools/poc/clock_residency/clock_probe --mode timing --probe compute --size short \
  --compute-iters 512 --device 1 --idle-us 0,10,25,50,100,250,500,1000,2000,5000,8000,12000,16000,20000,30000,50000,100000 \
  --reps 100 --warmup 4 --prime 8 --output artifacts/poc_clock_residency/gate2/compute_tiny.csv

tools/poc/clock_residency/clock_probe --mode timing --probe compute --size short \
  --compute-iters 512 --device 3 --idle-us 0,1000,2000,5000,8000,20000,100000 \
  --reps 60 --warmup 4 --prime 8 --keepalive-us 1000
```
