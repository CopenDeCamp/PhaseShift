> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# PSQ8 W32 KV Cache PoC（E4M3 code + per-32-block scale）

FP8 E4M3 KV と同じ 8 bit を、weight PSQ8 と同じ codec（**E4M3 8-bit code +
per-32-block BF16 scale**）で持つ `PSQ8_W32` KV cache を追加した記録。

## 目的と動機

- FP8_E4M3 KV は code が E4M3 で scale が per-(token, kv_head)（256 値に 1 個）。
  PSQ8_W32 は scale が per-(token, 32 値)（8 個/head）で、同じ 8 bit でも粒度が 8× 細かい。
- KV cache を PSQ 系（PSQ2/PSQ4/PSQ8）に統一する。
- FP8_E4M3 は削除せず併存させる。

## format

| 項目 | 値 |
| --- | --- |
| code | E4M3 8bit（weight PSQ8 と同じ） |
| block | 32 value（`kPsq8BlocksPerHead = 8`, head_dim 256） |
| scale | per block BF16 |
| 容量 | 8.5 bpw = 2176 B/token/layer（fp8 2080、psq4 1152、bf16 4096） |
| scale estimator | MAXABS / LSQ1（`PHASESHIFT_KV_PSQ8_ESTIMATOR=maxabs`） |

## 実装範囲

- `kv_cache_types.h`: `PSQ8_W32` + `kPsq8*` 定数 + `Psq8ScaleEstimator`
- `psq8_kv.h`: `psq8_maxabs_encode_block` / `psq8_lsq1_encode_block`（host/device 両用）
- `paged_kv_pool`: `PagedKVPoolPSQ8View` / allocation / `k_view` / `v_view` / `copy_page_from`
- `kv_append`: `launch_kv_append_psq8`（1 warp = 1 block、8-bit code をそのまま書く）
- `paged_decode`: `v_decode_psq8` + `launch_attention_paged_psq8`
- `paged_prefill`: `attention_paged_prefill_bf16_impl` の `kMode == 3` + launcher
- correctness: `detail/attention.inc` の append / reference
- runtime: paged attention dispatch / selector / executor / kv append 配線
- bench / CLI: `paged-attention --kv-dtype psq8`、`--kv-cache-dtype psq8`

## 計測（device 1、Qwen3.8-27B-PSQ、qh=24 kvh=4 hd=256 pt=16）

### 正しさ

- bench `--check` PASS: prefill rows256 vis256/1024/2048、decode rows8 vis256/2048/8192
- split 経路 `--check` PASS: rows1 vis8192/32768/65536（splits16）
- `test_paged_attention` 21/21、`test_qwen35_psq_kv_pool` 39/39
- e2e（prose 31 token、greedy 16 token、ctx512）: **GENERATED_IDS が bf16 と完全一致**
  （psq4 は token 2 で分岐）

### software e4m3 decode の除去（LDS LUT）

gfx12 では `e4m3_decode_f32` が分岐付きソフトウェア展開になる。256 entry の bf16 LUT を
block 先頭で構築し、decode / prefill の load loop を LDS 参照 1 回に置き換えた。

| | LUT 前 | LUT 後 |
| --- | ---: | ---: |
| decode rows1 vis8192 | 1964 µs | 443 µs |
| prefill rows2048 vis2048 | 8144 µs | 4523 µs |

### KV split/reduce の追加（psq4 にも適用）

bf16 の flash-decoding 型 2 kernel を psq4/psq8 にも用意した。kernel A は page を S 分割して
code pool から K/V を decode し partial (m, P, A) を書く。kernel B は dtype 非依存の既存
reduce を再利用する。

decode rows1 splits16（p50）:

| visible | bf16 | psq4 | psq8 |
| ---: | ---: | ---: | ---: |
| 8192 | 96 µs | 118 µs | 116 µs |
| 32768 | 360 µs | 398 µs | 423 µs |
| 65536 | 705 µs | 859 µs | 838 µs |
| 131072 | 1619 µs | 1903 µs | 1872 µs |

split 前は vis131072 で psq8 9203 µs（bf16 比 5.7×）、psq4 14460 µs（9.0×）だった。
split + scale vector load により bf16 比 約 1.16〜1.21×（psq8）、約 1.18〜1.25×（psq4）まで縮小した。

### per-element scale load の削減（psq8）

`psq8_value` は要素ごとに `scale_pool[block_index]` を global load していた。ablation で
vis65536 の decode コスト 255 µs のうち scale load が 209 µs、LDS LUT が 46 µs、残り
72 µs が W32 blocked layout の index 計算と判明した。8 block の scale は連続 bf16 なので
uint4 1 回の vector load（`bf16x8_get`）に置き換えた（psq8 のみ）。

| visible | vector 前 | vector 後 |
| ---: | ---: | ---: |
| 8192 | 139 µs | 116 µs |
| 32768 | 517 µs | 423 µs |
| 65536 | 1032 µs | 838 µs |
| 131072 | 2226 µs | 1872 µs |

psq4 は同じ変更で悪化した（vis65536 859 → 1080〜1330 µs）ため revert した。psq4 は
cb10 decode と scale load の codegen が既に良好で、uint4 load が register pressure を
上げたとみられる。

### prefill（rows2048 vis2048）

| dtype | p50 |
| --- | ---: |
| bf16 | 3242 µs |
| psq4 | 4737 µs |
| psq8 | 4523 µs |

psq8 prefill は PV の hi/lo 2 回 WMMA を使わず（bf16 prefill と同じ 1 回）CHECK PASS。

## 判断

PoC 成立。PSQ8_W32 を FP8_E4M3 と併存で採用する。長 context decode の残差（bf16 比
約 1.16〜1.21×）は code decode と W32 layout の index コストで、split + scale vector load 済み。

## 現在への影響

- 対応 KV dtype は BF16 / FP8_E4M3 / PSQ2_SHAPE_W32 / PSQ4_W32 / PSQ8_W32。
- 長 context decode の split/reduce は BF16 に加えて PSQ4_W32 / PSQ8_W32 でも発火する。
- prefix cache は PSQ8_W32 を fail-closed で拒否する。MTP KV state は FP8/BF16 のみ。
- 実長文プロンプトでの e2e 検証は未実施（手元に 31 token の実プロンプトのみ）。

## Revision

| 項目 | 値 |
| --- | --- |
| base | `b60a8375`（main） |
| branch | `poc/psq8-kv` |
| build | Release / gfx1201 / benchmarks ON / optional tests OFF |
| device | 1 |

## 計測コマンド

```bash
# correctness（split）
./build/phaseshift-bench paged-attention --variant optimized --kv-dtype psq8 --splits 16 \
  --rows 1 --visible 65536 --q-heads 24 --kv-heads 4 --head-dim 256 --page-tokens 16 \
  --out-dtype f32 --check --warmup 2 --samples 1 --launches 1 --device 1

# perf
./build/phaseshift-bench paged-attention --variant optimized --kv-dtype psq8 --splits 16 \
  --rows 1 --visible 65536 --q-heads 24 --kv-heads 4 --head-dim 256 --page-tokens 16 \
  --out-dtype bf16 --warmup 10 --samples 10 --launches 5 --device 1

# e2e
./build/phaseshift-compute --model-dir models/Qwen3.8-27B-PSQ \
  --input-ids-file <prompt_ids> --max-new-tokens 16 --max-seq-len 512 \
  --arena-gib 26 --temperature 0 --device 1 --kv-cache-dtype psq8
```
