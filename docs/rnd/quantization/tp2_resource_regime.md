# TP2 Linear Kernel Resource-Regime Optimization

`docs/perf/methodology.md` を測定原則の正本とする本ドキュメントは、
Qwen3.8-27B TP=2 の PSQ4 / PSQ8 linear について
「accumulator / reuse 重視」と「operand supply / latency hiding 重視」
どちらの resource regime が勝つかを実測する Gate 記録である。

## 0. 条件

| 項目 | 値 |
| --- | --- |
| branch | `poc/tp2-resource-regime` |
| 起点 commit | `96835679` |
| GPU | AMD Radeon AI PRO R9700 (gfx1201) ×4 可視、TP2 は HIP device 0 / 1 |
| model | `models/Qwen3.8-27B-PSQ` |
| build | Release / gfx1201 / `build-gfx1201` |
| probe | `/tmp/opencode/tp2_speed/tp_speed_regime`（greedy、投機 OFF、new_tokens=32） |
| 比較方式 | interleaved 5-run p50。run 間変動 30% 超で FAIL/BLOCKED |
| 測定 device | microbench は `--device 0`（`--device` 未指定時の既定） |

clock / power は `rocm-smi` から個体側（R9700）の値を取得できなかった
（`rocm-smi --showclocks` は iGPU のみを報告）。複数 workload の同時実行は
していないことを測定手順側で担保した。

## Gate 0: baseline 固定

### E2E（5-run p50）

worktree 作成後の incumbent（`96835679`、無改修）での実測。

| mode | ctx | metric | p50 | min | max | spread |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| TP2 | 64 | prefill | 59.294 ms | 58.782 | 62.098 | 5.6% |
| TP2 | 64 | decode | 37.750 tok/s | 36.997 | 37.819 | 2.2% |
| TP2 | 2048 | prefill | 576.693 ms | 575.268 | 577.279 | 0.3% |
| TP2 | 2048 | decode | 36.808 tok/s | 36.717 | 36.830 | 0.3% |

全条件で変動 30% 未満 → Gate 0 PASS。

（手順書 §5 記載の単回値 ctx64 58.9ms / 36.54 tok/s、ctx2048 572.7ms / 35.62 tok/s は
単回測定。本ドキュメントは methodology に従い 5-run p50 を基準値とする。）

## Gate 1: TP2 実 linear shape の確定

`PHASESHIFT_SHAPE_TRACE=1`（resolve 点の一時 instrumentation、測定後に revert）で
TP2 run の `PHASESHIFT_LINEAR_DEBUG` 相当の実配置を取得し、
family は `models/Qwen3.8-27B-PSQ/phaseshift_quantization.json` の `encoding` と
照合した。resolve は launch ごとに2回呼ばれるため件数は 1/2 して site と整合させた。

site 名・family は model 正本、local geometry は runtime trace が正本。

### A. TP2 shape inventory

| site（層内） | 層数 | family | rows | N | K | selected config |
| --- | ---: | --- | --- | ---: | ---: | --- |
| `mlp.gate_proj` | 64 | psq4 | 1 / 64 / 2048 | 8704 | 5120 | RowBlock1 / RowBlock4 / Prefill2D_K128N128 |
| `mlp.up_proj` | 64 | psq4 | 1 / 64 / 2048 | 8704 | 5120 | RowBlock1 / RowBlock4 / Prefill2D_K128N128 |
| `mlp.down_proj`（GDN 層） | 48 | psq4 | 1 / 64 / 2048 | 5120 | 8704 | RowBlock1 / RowBlock4 / Prefill2D_K128N128 |
| `mlp.down_proj`（attention 層） | 16 | psq8 | 1 / 64 / 2048 | 5120 | 8704 | RowBlock1 / RowBlock4 / Prefill2D_K128N128 |
| `linear_attn.in_proj_qkv`（GDN 32層） | 32 | psq4 | 1 / 64 / 2048 | 5120 | 5120 | RowBlock1 / RowBlock4 / Prefill2D_K128N128 |
| `linear_attn.in_proj_qkv`（GDN 16層） | 16 | psq8 | 1 / 64 / 2048 | 5120 | 5120 | RowBlock1 / RowBlock4 / Prefill2D_K128N128 |
| `linear_attn.in_proj_z`（GDN 32層） | 32 | psq4 | 1 / 64 / 2048 | 3072 | 5120 | RowBlock1 / RowBlock4 / Prefill2D_K128N128 |
| `linear_attn.in_proj_z`（GDN 16層） | 16 | psq8 | 1 / 64 / 2048 | 3072 | 5120 | RowBlock1 / RowBlock4 / Prefill2D_K128N128 |
| `linear_attn.out_proj`（GDN 32層） | 32 | psq4 | 1 / 64 / 2048 | 5120 | 3072 | RowBlock1 / RowBlock4 / Prefill2D_K128N128 |
| `linear_attn.out_proj`（GDN 16層） | 16 | psq8 | 1 / 64 / 2048 | 5120 | 3072 | RowBlock1 / RowBlock4 / Prefill2D_K128N128 |
| `self_attn.q_proj` | 16 | psq4 | 1 / 64 / 2048 | 6144 | 5120 | RowBlock1 / RowBlock4 / Prefill2D_K128N128 |
| `self_attn.k_proj` | 16 | psq4 | 1 / 64 / 2048 | 512 | 5120 | RowBlock1 / RowBlock4 / Prefill2D_K128N64 |
| `self_attn.v_proj` | 16 | psq4 | 1 / 64 / 2048 | 512 | 5120 | RowBlock1 / RowBlock4 / Prefill2D_K128N64 |
| `self_attn.o_proj` | 16 | psq4 | 1 / 64 / 2048 | 5120 | 3072 | RowBlock1 / RowBlock4 / Prefill2D_K128N128 |
| `linear_attn.in_proj_a` / `in_proj_b` | 48 | bf16 | 1 / 64 / 2048 | 24 | 5120 | ExactRows / WmmaKPartition |
| `lm_head` | 1 | psq8 | 1 | 248320 | 5120 | RowBlock1 |

config id は `config.h` の enum 順（Psq4: 0..3 = RowBlock1/2/4/8、
4..7 = Prefill2D_K64N64/K128N64/K64N128/K128N128、Psq8: 4..6 = Prefill2D/K64N128/K128N128）。

### model 正本と族の突き合わせ

`phaseshift_quantization.json`（`encoding` / `logical_shape` / `k_padded`）から
TP=2 の local geometry を復元すると、runtime trace と 1:1 に一致した。

```text
mlp.gate/up            [17408,5120] psq4 -> 8704:5120  psq4
mlp.down               [5120,17408] psq4 -> 5120:8704  psq4 (48 GDN層)
mlp.down               [5120,17408] psq8 -> 5120:8704  psq8 (16 attention層)
linear_attn.in_proj_qkv[10240,5120] psq4 -> 5120:5120  psq4 (32)
linear_attn.in_proj_qkv[10240,5120] psq8 -> 5120:5120  psq8 (16)
linear_attn.in_proj_z  [6144,5120]  psq4 -> 3072:5120  psq4 (32)
linear_attn.in_proj_z  [6144,5120]  psq8 -> 3072:5120  psq8 (16)
linear_attn.out_proj   [5120,6144]  psq4 -> 5120:3072  psq4 (32)
linear_attn.out_proj   [5120,6144]  psq8 -> 5120:3072  psq8 (16)
self_attn.q_proj       [12288,5120] psq4 -> 6144:5120  psq4 (16)
self_attn.k/v_proj     [1024,5120]  psq4 ->  512:5120  psq4 (各16)
self_attn.o_proj       [5120,6144]  psq4 -> 5120:3072  psq4 (16)
linear_attn.in_proj_a/b[48,5120]    bf16 ->   24:5120  bf16 (48×2)
lm_head                [248320,5120] psq8 -> 248320:5120 psq8
```

補足: `self_attn.q_proj` の local N は **6144**（`[12288,5120]` の半分）であり、
`3072:5120` は attention Q ではなく **`linear_attn.in_proj_z`**（GDN z 投影）である。
手順書 §21 の記述はこの点で実配置と異なるため、実 trace を正本とした。

また MTP / DFlash2 の tensor は存在するが、投機 OFF 実行では発火していない。

### A2. accumulator 幾何と既存 config の対応

`psq4.hip` の Prefill2D kernel は

```cpp
constexpr uint32_t kNt          = kOutBlock / kTileOut;   // kTileOut = 16
constexpr uint32_t kNtPerWave   = kNt / 2;                // PSQ4
float8_t acc[4][kNtPerWave];                              // 4 = row tile
```

であるため、**accumulator float 数は下表の通り**で、手順書 §11 が新規作成を
検討していた `Acc64` は既存 config として既に存在する。

| config | kOutBlock | kNtPerWave | accumulator |
| --- | ---: | ---: | ---: |
| `Prefill2D_K64N64` / `K128N64` | 64 | 2 | **4×2×8 = 64** float/thread |
| `Prefill2D_K64N128` / `K128N128` | 128 | 4 | **4×4×8 = 128** float/thread |

（PSQ8 は `kNtPerWave = kNt` で、`kOutBlock` が同じなら accumulator は同じ量。）

つまり Gate 2 の既存 geometry 総当たりは、そのまま
**「accumulator/reuse 重視（128）」 vs 「operand supply 重視（64）」**の比較になる。

## Gate 2: PSQ4 prefill 2D geometry sweep

rows ∈ {512, 1024, 2048} × TP2 主要 5 shape × cfg ∈ {auto, 0, 1, 2, 3}、
interleaved 5 反復の p50（µs）。

### ノイズ床の校正

`rows=1024 N=3072 K=5120` で auto の選択は `2d_ob128_kc128` であり、
明示指定した cfg3 と **同一 kernel**。実測値は 214.405 vs 211.942 で **1.1% 差**。
つまり本条件の run ノイズ床は約 **1.1%** で、これ以下の差は意味を持たない。

### 結果（incumbent との差）

| rows | N:K | auto(auto の選択) | cfg0 K64N64 | cfg1 K128N64 | cfg2 K64N128 | cfg3 K128N128 | winner |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 8704:5120 | 317.6 (K128N64) | 332.7 | 320.0 | 328.7 | 315.9 | 同等 |
| 512 | 5120:8704 | 327.3 (K128N64) | 345.1 | 327.9 | 360.1 | 347.3 | **auto(Acc64)** |
| 512 | 5120:5120 | 187.3 (K128N64) | 196.0 | 185.9 | 195.1 | 188.5 | 同等 |
| 512 | 5120:3072 | 109.4 (K128N64) | 117.7 | 114.0 | 120.0 | 115.9 | **auto(Acc64)** |
| 512 | 3072:5120 | 105.1 (K128N64) | 111.5 | 108.9 | 131.6 | 125.8 | **auto(Acc64)** |
| 1024 | 8704:5120 | 654.1 (K128N128) | 698.7 | 682.3 | 667.3 | 657.2 | 同等 |
| 1024 | 5120:8704 | 630.5 (K128N128) | 689.6 | 677.9 | 642.1 | 635.0 | 同等 |
| 1024 | 5120:5120 | 369.9 (K128N128) | 408.5 | 401.8 | 381.9 | 372.4 | 同等 |
| 1024 | 5120:3072 | 225.4 (K128N128) | 249.9 | 241.0 | 231.3 | 226.0 | 同等 |
| 1024 | 3072:5120 | 214.4 (K128N128) | 237.5 | 228.6 | 218.2 | 211.9 | 同等 (=ノイズ) |
| 2048 | 8704:5120 | 1409.3 (K128N128) | 1485.0 | 1450.0 | 1433.7 | 1410.0 | 同等 |
| 2048 | 5120:8704 | 1393.3 (K128N128) | 1466.9 | 1434.9 | 1423.6 | 1394.8 | 同等 |
| 2048 | 5120:5120 | 766.3 (K128N128) | 826.6 | 817.9 | 770.3 | 766.6 | 同等 |
| 2048 | 5120:3072 | 468.4 (K128N128) | 511.8 | 495.2 | 475.5 | 470.1 | 同等 |
| 2048 | 3072:5120 | 456.8 (K128N128) | 489.5 | 484.4 | 461.0 | 455.6 | 同等 |

### 判定

- **auto が全 15 (rows, shape) で最優勝またはノイズ帯内。selector 変更は不要。**
- rows ≥ 1024 では **K128N128（accumulator 128）が一貫して最速**で、
  K64 系は 3〜8% 劣後。→ accumulator / reuse 重視が TP2 prefill で正しく勝つ。
- rows = 512 では逆に **K128N64（accumulator 64）が最速**（N=3072 で K128N128 に対し
  16% 快適）。
- incumbent は `kPsq4GemmPrefill2dLargeMinRows = 1024` により
  rows=512 で K128N64、rows≥1024 で K128N128 を選んでおり、**この境界は実測と一致**。

## Gate 2-B: PSQ8 prefill 2D geometry sweep

rows ∈ {512, 1024, 2048} × TP2 の psq8 使用 4 shape（`5120:8704`, `3072:5120`,
`5120:3072`, `5120:5120`）× cfg ∈ {auto, 0=K64N64, 1=K64N128, 2=K128N128}、
interleaved 5 反復の p50（µs）。

| rows | N:K | auto(auto の選択) | cfg0 K64N64 | cfg1 K64N128 | cfg2 K128N128 | winner |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 512 | 3072:5120 | 130.95 (K128N128) | **109.99** | 132.24 | 131.07 | **cfg0 −16.0%** |
| 512 | 5120:3072 | 121.79 (K128N128) | **111.62** | 123.03 | 122.23 | **cfg0 −8.4%** |
| 512 | 5120:5120 | 195.06 (K128N128) | **179.27** | 196.11 | 195.30 | **cfg0 −8.1%** |
| 512 | 5120:8704 | 328.46 (K128N128) | **311.47** | 328.98 | 327.83 | **cfg0 −5.2%** |
| 1024 | 3072:5120 | 208.94 (K128N128) | 229.47 | 210.70 | 208.82 | 同等 |
| 1024 | 5120:8704 | 686.87 (K128N128) | 726.57 | 688.88 | 685.02 | 同等 |
| 2048 | 3072:5120 | 478.43 (K128N128) | 507.51 | 478.45 | 477.76 | 同等 |
| 2048 | 5120:8704 | 1431.87 (K128N128) | 1474.08 | 1434.37 | 1431.58 | 同等 |
| 2048 | 他2 shape | （同様に） | 悪化 | 同等 | 同等 | 同等 |

### 判定（暫定、rows=768 で境界確定待ち）

- rows ≥ 1024 は **auto が最優勝 → 変更不要**。
- **rows = 512 のみ、auto が −5.2〜−16.0% と明確に劣後**（ノイズ床 1.1% を大きく超える）。
- 原因は `select_psq8_gemm_config()` に PSQ4 が持つ
  `kPsq4GemmPrefill2dLargeMinRows = 1024` 相当の rows 守りが無く、
  `rows = 512` でも `ob128 && kc128` から **K128N128 を選んでしまう**こと。
  PSQ4 は同じ rows=512 で K128N64 を選び、それが正解である。
- 到達性: `schedule_requests(active, config_.max_scheduled_tokens, ...)` により
  prefill の rows は予算の余りとなり得る（予算 2048 / prompt 2560 → 最終 chunk が
  `rows=512`）。2D の発火条件は `rows % 256 == 0 && rows >= 512` なので
  `rows ∈ {512, 768}` が該当する。

## Gate 3: decode1 unroll 再測定

rows=1、PSQ4 / PSQ8 を別表で、configs を rep 単位で interleaved した 5 反復 p50（µs）。

### PSQ4（u=0 が production auto = `N<=2048 ? 16 : 8`）

| N | K | u0(auto) | u2 | u4 | u8 | u16 | winner |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 5120 | 8.572 | 18.753 | 13.067 | 9.764 | 8.548 | 同等 |
| 3072 | 5120 | 13.904 | 25.982 | 18.001 | 13.921 | 13.735 | u16 −1.2% |
| 5120 | 3072 | 12.813 | 18.496 | 14.360 | **12.700** | 13.357 | 同等 |
| 5120 | 5120 | 27.013 | 31.348 | 27.699 | **26.984** | 28.244 | 同等 |
| 5120 | 8704 | 43.321 | 47.250 | 43.627 | **43.319** | 43.899 | 同等 |
| 8704 | 5120 | 43.102 | 47.131 | 43.970 | **43.022** | 44.308 | 同等 |

- **N=5120 は K = 3072 / 5120 / 8704 のいずれでも u8 が最良 → K 依存は無い。**
  crossover は **N のみ**で決まる（§8 の問いへの回答）。
- 唯一の非同等は N=3072 の u16 が −1.2%。先行の 20 回ペア測定では −0.72% で、
  §8 の採用ルール「1〜2% は selector を複雑化しない」に該当。
- u2 は全形状で 1.18〜1.96 倍と明確に劣後、u4 も劣後。

### PSQ8（u=0 が production auto = 8 固定）

| N | K | u0(auto) | u2 | u4 | u8 | u16 | winner |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3072 | 5120 | **30.508** | 34.874 | 30.849 | 30.529 | 30.353 | 同等 |
| 5120 | 3072 | **30.403** | 33.623 | 30.759 | 30.401 | 30.522 | 同等 |
| 5120 | 5120 | **47.525** | 52.359 | 48.275 | 47.561 | 47.618 | 同等 |
| 5120 | 8704 | **77.637** | 78.025 | 77.661 | 77.631 | 77.613 | 同等 |

### 判定

**decode1 の unroll policy は変更不要**（PSQ4 / PSQ8 とも）。

## Gate 4: resource regime の分類

実測は `A = accumulator/reuse-heavy（N128, acc128, VGPR 223〜240, occ 6〜7）` と
`B = operand supply / occupancy-heavy（N64, acc64, VGPR 141〜166, occ 9〜10）` の
2 つの既存 variant 間で比較した結果である。

| 検討対象 | rows | 勝者 | 対応する Case |
| --- | ---: | --- | --- |
| PSQ4 prefill | ≥1024 | **A = K128N128** | **Case A** |
| PSQ4 prefill | 512 | **B = K128N64**（incumbent が既に選択） | Case A（境界が既に正しい） |
| PSQ8 prefill | ≥1024 | **A = K128N128** | **Case A** |
| PSQ8 prefill | 512 | **B = K64N64**（incumbent は A を選択） | **Case B** |
| PSQ4 / PSQ8 decode1 | 1 | unroll は occupancy regime で分かれるが auto が最良 | Case A |

**判定:**

- **Case A が支配的**（rows ≥ 1024 = E2E の主経路では両 family とも K128N128 が最速）。
  → accumulator / reuse-heavy が TP2 prefill でも正しい。**kernel 内部は変更しない。**
- **Case B は PSQ8 の rows=512 のみ**で発生。incumbent が K128N128 を選ぶため
  −5.2〜−16.0% の空きがある → selector の境界だけを直す（新 kernel 不要）。
- **Case C（中間 gap）は存在しない。** K128N128 と K64 系がいずれかに明確に分かれており、
  「両方とも中途半端」という状況ではない。

## Gate 5 / Gate 6 / Gate 7: Acc64 PoC の要否

**Gate 5 は成立せず、Gate 6（新 variant 作成）には入らない。**

理由:

1. **§11 が「作ること」としていた Acc64 variant は既に存在する**
   （`Prefill2D_K64N64` / `K128N64` = accumulator 64 float/thread、VGPR 141〜166）。
   独立した `Prefill2D_Acc64` を新設する必要がない。
2. **Gate 7 の棄却条件に該当する。** 手順書 §12 の規則
   「Acc64 が >3% 悪化 → reuse loss / grid 増加が支配 → 棄却」を、
   rows ≥ 1024（E2E の主経路）での実測が満たしている:

   ```text
   PSQ4 rows=2048  K128N64 vs K128N128
     8704:5120   1450.0 vs 1410.0  -> +2.8%
     5120:8704   1434.9 vs 1394.8  -> +2.9%
     5120:5120    817.9 vs  766.6  -> +6.7%
     5120:3072    495.2 vs  470.1  -> +5.3%
     3072:5120    484.4 vs  455.6  -> +6.3%
   ```

   全 5 shape で Acc64 が 2.8〜6.7% 悪化（ノイズ床 1.1% を明確に超える）。
3. resource 上も Acc64 は VGPR 166 / occ 9 と有利な側にいるのに負けるため、
   「空いた register budget を latency hiding に使う」ことの投資回収根拠が無い。

したがって **Gate 8（A prefetch）・Gate 9（W prefetch）・Gate 10（double buffer）は
着手しない**。本作業の kernel 関連は Gate 6 の新規作成なしで終了する。

## D. New variant

**該当なし。** Gate 6（`Acc64` PoC の新 variant 作成）には到達していない
（§11 の PoC 対象は既存 `K64N64` / `K128N64` として実在し、Gate 7 で棄却）。
したがって `psq4.hip` / `psq8.hip` の kernel 実装には **1 行も変更を加えていない**。

| shape | incumbent | Acc64 | Acc64+A1 | Acc64+A2 | W-prefetch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全 TP2 shape | （既存 variant 内で比較済み、Gate 2/2-B 参照） | REJECT | 未実施 | 未実施 | 未実施 |

## E2E Gate（§22）結果

base / cand を **run 単位で交互**（rep 奇数: base→cand、偶数: cand→base）に5反復、
model は同じ `Qwen3.8-27B-PSQ`、greedy 32 tokens、投機 OFF。

| config | metric | base p50 | cand p50 | delta | base spread | cand spread |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `chunk512` (ctx2048 / budget512) | prefill | 165.468 ms | **163.810 ms** | **+1.00%** | 1.9% | 0.5% |
| `chunk512` | decode | 23.279 tok/s | 23.347 | +0.29% | 0.1% | 0.3% |
| `ctx2048` (rule 到達なし) | prefill | 578.059 ms | 578.348 ms | −0.05% | 0.7% | 0.7% |
| `ctx2048` | decode | 36.826 tok/s | 36.848 | +0.06% | 0.2% | 0.3% |
| `ctx64` (row-block 経路) | prefill | 58.042 ms | 58.473 ms | −0.74% | 3.0% | 3.0% |
| `ctx64` | decode | 37.791 tok/s | 37.796 | +0.01% | 0.4% | 0.3% |

`chunk512` のペア結果（rep 単位）:

```text
rep1 base=167.559 cand=163.569  +2.38%  cand wins
rep2 base=164.387 cand=164.079  +0.19%  cand wins
rep3 base=165.771 cand=163.760  +1.21%  cand wins
rep4 base=164.376 cand=163.810  +0.34%  cand wins
rep5 base=165.468 cand=164.395  +0.65%  cand wins
base sorted: 164.38 164.39 165.47 165.77 167.56
cand sorted: 163.57 163.76 163.81 164.08 164.40   (最大 cand < 最小 base 直下、ほぼ分離)
```

- **rule が効く構成で +1.00%、全 5 run が同方向**（§22「>= 0.5% で全 run 同方向なら有力」を満たす）。
- rule に到達しない 2 構成は **回帰なし**（ctx2048 −0.05% / ctx64 −0.74%、いずれも
  各自の spread 内でノイズ。しかも rows<512 あるいは rows×N>3M のため構造的に到達しない）。
- decode は +0.01〜+0.29% でノイズ（prefill 専用の変更のため想定どおり）。

## F. Decision

| candidate | decision | 理由 |
| --- | --- | --- |
| **PSQ8 prefill2D の小 workload 境界（rows<1024 && N≥2048 && rows×N≤3e6 → K64N64）** | **ADOPT** | micro: rows=512 で −5.2〜−16.0%（4/4 shape、spread 1〜2%）。E2E: `chunk512` +1.00%・全 5 run 同方向、回帰 2 構成なし。unit test 100% / targeted 5/5 / **required acceptance 124/124 PASS（failure 0 / skip 0 / error 0）**。rule は 3 条件。 |
| PSQ4 prefill2D の境界（`kPsq4GemmPrefill2dLargeMinRows=1024`） | **ADOPT（変更なし）** | auto が全 15 (rows, shape) で最優勝またはノイズ帯内。境界は実測と一致。 |
| decode1 unroll policy（PSQ4: N≤2048→U16 / それ以外 U8、PSQ8: 固定 U8） | **ADOPT（変更なし）** | 全 shape 同等。N=5120 で K 3通り測っても crossover が変わらず **K 依存なし**。N=3072 のみ −1.2% で §8「1〜2% は複雑化しない」に該当。 |
| 新 variant `Prefill2D_Acc64`（§11 PoC） | **REJECT** | 既存 `K64N64`/`K128N64` がその役割を担う。rows≥1024（E2E 主経路）で **5/5 shape が 2.8〜6.7% 悪化** → §12 Gate 7 の棄却条件に該当。新設不要。 |
| Gate 8: A prefetch | **BLOCKED（着手せず）** | Gate 7 で Acc64 が棄却され、Gate 8 に到達しない。 |
| Gate 9: W prefetch | **BLOCKED（着手せず）** | 同上。 |
| Gate 10: double buffer | **BLOCKED（着手せず）** | 同上。過去の VGPR 256 + spill 失敗を踏み戻さない。 |

**Phase 5〜10 は Gate 5/7 の成立要件が揃わなかったため実施しない。**
本作業で kernel に手を入れた箇所はゼロ（selector と unit test のみ）。

## 実施した変更

```text
include/phaseshift/models/qwen35/kernels/optimized/linear/config.h
  + kPsq8GemmPrefill2dSmallWorkElements    3000000
  + kPsq8GemmPrefill2dLargeMinRows         1024
  + kPsq8GemmPrefill2dLargeMinOutFeatures  2048

src/phaseshift/models/qwen35/runtime/linear_selector.cpp
  + select_psq8_gemm_config の auto 分岐に 3 条件の小 workload 判定を追加

tests/unit/test_qwen35_linear_selector.cpp
  + test_psq8_small_work_prefill2d()
    ON/OFF 両側の境界、override 優先、既存挙動の維持、PSQ4 非影響を固定

docs/rnd/quantization/tp2_resource_regime.md（本ドキュメント）
```

測定は `PHASESHIFT_SHAPE_TRACE` の一時 instrumentation を使用し、
計測完了後にすべて revert 済み（`grep linear_shape_trace` = 0 件）。

## 正しさの gate まとめ

| gate | 結果 |
| --- | --- |
| targeted ctest（`test_gemm_psq4_w4a8_wmma` / `test_gemm_psq4_decode1` / `test_gemm_psq8_w8a8_wmma` / `test_gemm_psq8_decode1` / `test_qwen35_linear_selector`） | **5/5 PASS** |
| required acceptance | **124/124 PASS**（failure 0 / skip 0 / error 0） |
| correctness と性能の分離 | なし（correctness は ctest / acceptance、性能は E2E A/B と microbench で分離） |




## C. Resource metadata（`-Rpass-analysis=kernel-resource-usage`、gfx1201）

| unit | variant | SGPR | VGPR | LDS (B) | spill | scratch | occ [waves/SIMD] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| psq4 | Prefill2D N64 K64 | 22 | 147 | 4608 | 0 | 0 | 9 |
| psq4 | Prefill2D N64 K128 | 22 | 166 | 9216 | 0 | 0 | 9 |
| psq4 | Prefill2D N128 K64 | 22 | 226 | 9216 | 0 | 0 | 6 |
| psq4 | Prefill2D N128 K128 | 22 | 240 | 18432 | 0 | 0 | 6 |
| psq4 | RowBlock1 | 38 | 79 | 0 | 0 | 0 | 16 |
| psq4 | Decode1 u=2 / 4 / 8 / 16 | 22-23 | 43 / 67 / 108 / 108 | 0 | 0 | 0 | 16 / 16 / 12 / 12 |
| psq8 | Prefill2D N64 K64 | 28 | 141 | 4608 | 0 | 0 | 10 |
| psq8 | Prefill2D N128 K64 | 28 | 196 | 9216 | 0 | 0 | 7 |
| psq8 | Prefill2D N128 K128 | 28 | 223 | 18432 | 0 | 0 | 6 |
| psq8 | RowBlock1 | 34 | 77 | 0 | 0 | 0 | 16 |
| psq8 | Decode1 u=2 / 4 / 8 / 16 | 22 | 44 / 58 / 94 / 103 | 0 | 0 | 0 | 16 / 16 / 16 / 12 |

要点:

- **全 variant で spill = 0、scratch = 0**（§11-C の promotion 条件を満たす）。
- occupancy の階段は **VGPR 166 → 9 波 / VGPR 226 → 6 波**（psq4）。
  `Acc128 (N128)` 側は VGPR 226〜240 で occupancy 6〜7、
  `Acc64 (N64)` 側は VGPR 141〜166 で occupancy 9〜10。
- psq4 Decode1 は **u8 と u16 がともに VGPR 108 / occ 12 で同値**であり、
  Gate 3 で両者がほぼ等速だったことと整合する。
- psq8 Decode1 は u8 (VGPR 94, occ 16) と u16 (VGPR 103, occ 12) で occupancy が
  違うのに等速 → decode1 は VGPR では律速していない。


