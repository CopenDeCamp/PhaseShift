> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# PSQ / GEMM / quantization カーネル最適化履歴

`docs/rnd/optimization_findings.md` から分離した、PSQ4/PSQ8 GEMM・preshuffle・scale・codebook・iMatrix・W4A8/W8A8・FP8/MXFP4 の実験履歴。見出し番号（7.x）は元 dump の通し番号を保持する。各節はその時点の条件・実測・採否判断を含み、採否（採用 / 却下 / revert / 未着手）は本文中に記載がある。

## 収録セクション

- 7.1 weight code plane の fragment-order 化（runtime preshuffle）
- 7.1b FP8 code plane の fragment-order 化
- 7.1c activation code plane の fragment-order 化
- 7.1d weight scale plane の fragment-order 化
- 7.2 MB（複数 row-tile / block）で weight 再利用を上げる
- 7.3 MB の副作用（decode 回帰）と shape 別 instantiation
- 7.5 fragment-order layout の制約（テスト含む）
- 7.6 operand swap（A=activation, B=weight）
- 7.7 MB=2 の現状（revert 済み）
- 7.8 GEMV（Small-M compute kind）の削除
- 7.9 prefill GEMM の実測（WMMA 天井と律速の切り分け）
- 7.10 prefill GEMM の構造解: scale を operand へ畳む（LUT + fp8 WMMA）
- 7.10b fold 精度の追試: pow2 グローバルスケールと scale 粒度
- 7.11 実装結果: LUT 方式は不成立、perm 展開 + epilogue FMA で 1.5 倍
- 7.12 MB は効かない（実装して否定）
- 7.20 int8 (W8A8) GEMM の次の一手（MB tiling 移植）
- 7.21 全最適化カーネルの ISA 走査と k-loop の命令削減
- 7.22 E4M0 + codebook 形状の実データ比較
- 7.23 PSQ4 W4A8 の scale fold 実測と「epilogue 1.82x」の訂正
- 7.24 scale 保存形式の再測定（bf16 / f16 / e5m2 / e4m3）と cb10 vs e2m1
- 7.25 decode PSQ4 GEMM の in-block split-K
- 7.26 e4m3 activation quantize の producer 融合
- 7.27 GEMM 帯域フェーズの負結果とカーネル下限（HIP Graph 評価は runtime へ分離）
- 7.28 residual_add の GEMM epilogue 融合
- 7.30 fused e4m3 量子化カーネルの latency 改善（unroll / thread 数）
- 7.31 lm_head (Output role) の PSQ8 量子化
- 7.32 PSQ GEMM k-loop の unroll 2 -> 4
- 7.33 見送り: swiglu/attngate の per-tile scale 融合、gdn_prepare の GEMM epilogue 融合
- 7.34 量子化レシピの感度: psq8 を減らす
- 7.35 PP2048: §7.32 の unroll 4 が MB=8 で spill を起こしていた
- 7.36 iMatrix collector 実装と量子化レシピへの適用
- 7.37 iMatrix で G8 レシピの品質を回復
- 7.40 BF16 の preshuffle（tile layout）: GEMV と WMMA GEMM の layout トレードオフ
- 7.41 MTP（M>1）での preshuffle 検証（将来機能用）
- 7.42 BF16 の preshuffle をロード時に採用
- 7.44 BF16 preshuffle GEMM (tile kernel) の repair と correctness path
- 7.45 gfx1201 decode bf16 GEMV: rows=1 カーネル移植
- 7.46 gfx1201 prefill bf16 GEMM: 64-row wide tile カーネル（rows>=65）
- 7.47 fused PSQ4 GEMM の wave 量子化
- 7.49 PSQ4 prefill 2D LDS tiled GEMM 実験（負結果・production 化せず）
- 7.50 PSQ4 prefill 2D v2 と production 化
- 7.52 PSQ4 prefill 2D の BK / BN 拡張
- 7.55 PSQ4 prefill GEMM の律速はどこか
- 7.68 PSQ4 GEMM の M 依存と batch-invariant 化
- 7.78 PSQ4 W4A8 の BF16 scale を v_dot2_f32_bf16 へ置換する PoC（不成立）
- 7.80 FP8 block128 / MXFP4 WMMA の初回計測
- 7.81 PSQ4 decode GEMV: codebook 簡約 + rows=1 専用 kernel + unroll 再調整
- 7.85 PSQ4 prefill_2d: 既存 4 variant の sweep と Phase 1 冗長除去
- 7.86 PSQ4 prefill_2d: Phase 3（software pipelining）は棄却、律速は WMMA 側
- 7.91 PSQ8 prefill 2D GEMM の幾何拡張（BN=128 / BK=128）
- 7.92 PSQ8 prefill GEMM の operand supply 分解（ablation）
- 7.93 PSQ8 prefill GEMM の A 重複除去（wave マッピング変更）
- 7.94 PSQ4 prefill GEMM への A 重複除去の移植（負結果）

---

## 7.1 weight code plane の fragment-order 化（runtime preshuffle）

FP4 の code plane `[N][K/2]` では、16出力tileの各laneが `(K/2)` バイト間隔の
アドレスを読み、**1命令で16本のcache lineに散る**。load時に次の形へ並べ替える。

```text
byte(o, ib, j) = (o>>4)*(K*8) + (o&15)*16 + ib*256 + j
```
- 32 lane × 8byte = 1 k step あたり連続256B（16出力×16byte）→ **1〜2 line/命令**。
- **モデルファイルは変更しない**。loader が arena へ上げる時にCPUで1回 permute する。
- 前提: `m%16==0 && kp%32==0`。非該当は error（旧layoutへのfallbackは持たない）。
- 消費側は同じ式を使う: ordinary/fused の fp4 GEMM・GEMV、correctness参照。
- 実測（rows=128, 4B shapes）: mlp_gate_up 350→280us、mlp_down 306→266、attn_o 163→128、gdn_qkv 302→254（-13〜-21%）。

**白箱テストへの波及**: カーネルテストは `N=1,15,17,...` の非16倍も検証する。
テスト側で **padded fragment-order copy を upload**（`ceil(N/16)` tile、端数tileは0埋め）し、
参照計算はファイルlayoutのまま読めばよい。そうしないと `tiles = N/16 = 0` で全ゼロになる。

**一般則（alignment と coalescing は別物）**: 速さの本質は
**「warp の1ロード命令が何本の cache line を踏むか」**。16byte塊は**アラインを保ったまま**並べ替える（詰め直さない）。

```text
row-major: lanes 0-15 が K/2 バイト間隔の16行を読む → 16 line を踏む（8/64 = 12.5%使用）
frag     : lanes 0-15 が連続256Bを読む             → 2〜4 line（使用率100%）
```
alignment は必要条件であって原因ではない。row-major でも alignment は満たしている。
「命令数」「ロードするバイト数」は両者で同じで、**address pattern だけ**が違う。


---

## 7.1b FP8 code plane の fragment-order 化

FP8 は fp4 と違い **loader が permute していなかった**（`[N][kp]` のまま）。
GEMM の lane は16出力行を跨ぐため、1命令で16 line（8/64 = 12.5%）を踏んでいた。

```text
byte(o, blk, half, jj) = (o>>4)*(nb*512) + blk*512 + half*256 + (o&15)*16 + jj
  k = blk*32 + half*16 + jj        (nb = kp/32, 512 = 32k × 2half × 8B)
```
- 1 k-block(32k) = 行あたり 16byte × 2 half。16行分が連続 → 1命令 = 連続256B。
- 消費側はすべて frag 式に変更: fp8 GEMM/GEMV・fused GEMM/GEMV・embedding・correctness参照。
- **メタデータ(scale)は別プレーンのまま**（16byte塊に入れない。入れると17/33byteになりアラインが崩れる）。
- 実測（4B shapes, `--check` mismatch=0）:
  mlp_gate_up 263.4→177.6 (r128) / 1594.6→902.9 (r512) / 6686→4122us (r2048)、
  mlp_down(2560×9216) 205.8→165.0 (r128, -20%)。
- end-to-end 4B-BALANCED: PP128 1696→1718、PP512 2960→3066、TG 43.5→47.5 tok/s。
- GEMV は元から coalesced（1 warp = 1行を担当）なので、frag 化で 2 line/subgroup 程度（軽微）。


---

## 7.1c activation code plane の fragment-order 化

GEMM の A 断片は lane l が活性行 (l&15) を跨ぐため、row-major の活性面では
1命令で16行 × stride を踏んでいた（捨て実験で **-12〜-20%** 分）。

```text
act(m, blk, half, jj) = (m>>4)*(code_stride*16) + blk*512 + half*256 + (m&15)*16 + jj
  k = blk*32 + half*16 + jj
```
- `code_row_stride_bytes` の**意味は変えない**（tile stride = 16×code_stride）。
  `make()` の codes 領域だけ `ceil(max_rows/16)*16*code_stride` に拡大（rows%16!=0 対策）。
  `a8_ws_scales_offset` も同じ丸めに合わせる。
- producer は2箇所: `activation_quantize_a8`（optimized）と
  correctness の `exec_activation_quantize_a8`（**片方だけ直すと一致しなくなる**）。
- 実測（bit-exact, 4B shapes）: fp4 mlp_gate_up 172.6→138.2 (r128) / 740.6→605.8 (r512)、
  fp8 177.6→171.3 / 902.9→759.6。mlp_down r128: fp4 176.4→171.3、fp8 165.0→153.9。
- end-to-end 4B-BALANCED: PP128 1718→1792、PP512 3066→3246 tok/s。
  **TG 47.5→46.7 (-1.6%)**: GEMV は1行しか読まないので frag だと chunk が分かれて不利。
  GEMM の利得（prefill）が大きいので frag を採用。

**落とし穴**: `uint8_t*` 経由で int8 コードを読むと符号拡張が落ちる。
参照と GEMV が同じバグで一致してしまい、**GEMM だけ落ちる**という分かりにくい形で出た
（probe で kernel 単体・act_quant 単体を検証して切り分けた）。


---

## 7.1d weight scale plane の fragment-order 化

GEMM の scale は lane が出力行 (l&15) ごとに読むため、row-major だと1命令で16行×stride。
fp4/fp8 どちらもここが**最大の残りコスト**だった（捨て実験で fp4 -7〜-25%、fp8 -36〜-49% の見込み）。

```text
fp4 (u8, per-16):   s(o, ib, half) = (o>>4)*(nb*32) + ib*32 + half*16 + (o&15)
fp8 (bf16, per-32): s(o, blk)      = (o>>4)*(nb*32) + blk*32 + (o&15)*2
```
- `storage_scale_stride_bytes` / `scale_row_stride_bytes` の意味を **tile stride (nb*32)** に変更。
- loader が permute（fp4 は storage scales、fp8 は compute_scales_bf16）。
  fp8 の storage scales (u8, embedding 用) は row-major のまま。
- 実測（bit-exact）: fp4 gate_up 138.2→130.7 (r128) / 605.8→**453.2** (r512, -25%)、
  fp8 171.3→171.4 (r128) / 759.6→**532.3** (r512, -30%)。
  **rows=128 の fp8 は伸びない**（別の律速に隠れている）。
- end-to-end: PP128 1792→1795、PP512 3246→**3442 (+6.0%)**、TG 46.8→46.6。


---

## 7.2 MB（複数 row-tile / block）で weight 再利用を上げる

1 block が 16行tileを MB 個処理し、**同じ weight fragment を MB 回使う**。
- grid の行次元は `ceil(rows / (16*MB))` に減る。
- 実測（rows=128, MB=2）: mlp_gate_up 280→199us、gdn_qkv 254→157、attn_q 114→81、attn_o 128→97（-24〜-38%）。
- end-to-end 4B: PP128 1321→1626、PP512 2239→2781、PP2048 6428→6976 tok/s。


---

## 7.3 MB の副作用（decode回帰）と shape別instantiation

MB=2 を全shapeに適用すると **rows≤16 では2つ目の row-tile が無効でも WMMA 4本を発行**する
（+VGPR 77）。decode（rows=1）は丸損で、実測 **TG 41.8→38.1 tok/s（-9%）**。

一般則:

```text
M を増やす最適化は M>=32 を改善し、M=1 を悪化させ得る。
→ kernel を M で instantiate し、launch 時に shape で選ぶ。
   (libr4d: W4A8_MIN_M / DECODE_MAX_M / TN4_MIN_M と同じ設計)
```
n=32 のような固定費支配のshapeも同じ dispatch で MB=1 に落とす（実測 36→43us の微増も解消）。


---

## 7.5 fragment-order layout の制約（テスト含む）

**layout 自体の制約**

```text
code plane は 16出力 tile 単位（byte(o,ib,j) = (o>>4)*(K*8) + (o&15)*16 + ib*256 + j）
  → 格納上、出力次元は「16行tile」で扱われる
  → k_padded は 32 の倍数が必要（quantizer が付与する k_padded が保証）
```

**runtime** は制約を吸収する:

```text
loader が ceil(m/16)*16 行へゼロ埋めしてから permute する
  → 任意の m を受け付ける（m%16!=0 でも可）
  → codes の確保サイズも padded_m*(kp/2) に揃える
```

**カーネルテスト** も同じ吸収が必要（これが「16の倍数でないとテストできない」の実体）:

```text
テストは device へ「padded fragment-order copy」を上げる:
  tiles = ceil(N/16)、端数 tile はゼロ埋め
  fp4_frag_pack(wc, N, k_padded) 相当のヘルパを各テストに置く
参照計算は従来どおりファイル layout（[N][K/2]）を読めばよい。
padding しないと tiles = N/16 = 0 となり全ゼロになる（= 失敗する）。
```

対象テスト: `test_gemm_rocmfp4_w4a8_wmma` / `test_gemv_rocmfp4_w4a8` / `test_fused_fp4fp8`
（`test_quantized_w4_scale_layout` は runtime loader 経由なので loader の padding で足りる）。

**検証の注意**: テストバイナリは必ず**フルリビルド**してから ctest すること。
インクリメンタルビルドでは relink されず、古いlayoutのままPASSして見える（今回実際に起きた）。


---

## 7.6 operand swap（A=activation, B=weight）

iu8 WMMA の A/B を入れ替える。**load は変えず、wmma の引数順と store の対応だけ**変える。

```text
C 断片(不変)      : element e of lane l = C[8*(l>>4)+e][l&15]
A=activation 断片 : byte e of lane l = A[row=l&15][k=8*(l>>4)+e]  ← 現行の a0/a1 load のまま
B=weight 断片     : byte e of lane l = B[k=8*(l>>4)+e][col=l&15]  ← 現行の w0/w1 load のまま
store             : output[(row_start+8*k_group+e)][out_start+m_lane]
```

利得:

```text
- weight scale が lane-local になり __shfl が 16→8 本（weight分が消え、activation分8本のみ）
- store が coalesced（旧: 1命令で16行に散る、新: 同じ行の連続16出力）
```

実測（rows=128, 同一条件A/B, 4B shapes）:

```text
mlp_gate_up 184.9→172.6us  attn_q 79.0→66.4  mlp_down 197.3→176.4   (vs ca474fc)
PP128 1653.9→1696.4  PP512 2905.0→2959.6  TG 42.5→43.5 tok/s
```

**踏んだ罠**: activation scale は要素ごとに違う行（element e は row `8*(l>>4)+e`）が要る。
これを scalar の strided load 8本にすると **-45% の大悪化**（各loadが2 lineに散る）。
`__shfl` で配る方が速い。さらに伸ばすなら activation scale を `[k-block][row]` に転置して
float4×2 で読む（次の段）。


---

## 7.7 MB=2 の現状（revert 済み）

MB=2（2 row-tile/block、weight fragment 再利用）は計測上良好だった:
fp4 GEMM rows=128 で -24〜-38%、end-to-end PP128/512/2048 が +23/24/9%。

ただし **rows が 32 の倍数でないとき 2つ目の row-tile が 0 を返す**（例 rows=63, N=1 で got=0）。
production prefill を壊し得るため revert した（現在の kernel は MB=1 = f44eb4e の状態）。
再land条件: 2つ目の tile の境界（`r_self2` / `row_valid2` / store の `tile<MB`、
activation の `cstride` 境界）を直し、**rows%32!=0 を含むテスト**を通すこと。
`MB` のテンプレート化と `rows<=16 → MB=1` の dispatch は、TG 回帰（-9%）対策として同時に入れる。


---

## 7.8 GEMV（Small-M compute kind）の削除§7.3 の「M で instantiate して launch 時に選ぶ」を、**そもそも選ばせない**方向で畳んだ。

```text
理由: MTP + 継続バッチでは decode の rows が 1 でなくなる。
      実測でも rows>=2 で GEMM が勝つ shape が支配的:
        lm_head bf16 248320x5120  rows=2  GEMM 5317us / GEMV 8193us (GEMM が 35% 速い)
      GEMV が勝つのは rows=1 の細い shape だけ、しかも ~1.2x:
        1024x5120 rows=1  47.1 vs 55.6 / 48x5120 46.9 vs 55.4
      32x2560 rows=1 は GEMM の方が速い (31.3 vs 38.1)。
```

**「GEMM は R が細いと空 WMMA で損」は誤り**。GEMM は weight 帯域律速で R に依らず一定:

```text
fp4 10240x5120  GEMM rows=1/2/4/8/16 = 62.5/62.0/61.0/60.4/66.5us  (一定)
                GEMV rows=1/2/4/8/16 = 248/506/1000/2093/4053us     (R に比例)
```

削除で消えたもの: gemv kernel 3本 + header 3本 + 専用 test 3本 + fused-gemv 9本 +
detail 3本 + 専用 bench + `select_linear_topology` と crossover 表 → 計 **-2951 行**。
selector は `Correctness / Optimized` の coverage 表だけになった。

```text
許容したコスト（rows=1 のみ）: 4B TG 48.5→43.5 (-10%) / 27B TG 8.48→7.84 (-8%)
PP は不変（4B 2922/3095、27B 469/PP512）。required 75/75。
4B の greedy トークンは変化（bf16 丸め差。27B は不変）。
```


---

## 7.9 prefill GEMM の実測（WMMA 天井と律速の切り分け）

27B prefill は 463 t/s (PP2048, TP1)。同じ 27B を vllm-radiance は TP2 で 4444 t/s
(single-stream, prompt 3024 tokens, MXFP4/W4A8 + FP8 KV + DFlash K7) = **per-GPU で約 4.8 倍差**。
compute bound は `2*27e9*512/157TF = 176ms/pass → 2900 t/s` なので、我々は 18%、相手は 72%。

### WMMA 単体の天井（マイクロベンチ, R9700 32CU@2350MHz）

```text
iu8  (int8xint8->int32)  209.8 TFLOPS
fp8  (e4m3xe4m3->f32)    210.1 TFLOPS   ← f32 accumulate。__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12
bf16 (bf16xbf16->f32)    113.9 TFLOPS
```

実効: fp4 GEMM 44.6 / fp8 GEMM 27.8 / bf16 GEMM 9.1 TFLOPS（いずれも 17408x5120 r2048）。
= fp4 は天井の 21%。**WMMA は律速ではない。**

### 律速の切り分け（fp4 17408x5120 r2048 = 8172us を基準に全部実測）

| 施策 | 結果 |
|---|---|
| MB=1 | 9375us (+15%) ← weight 再利用は効く |
| MB=2（現行） | 8172us |
| MB=4 | 8206us ±0 |
| OB=4（out-tile 4本/block、A traffic 1/4） | 13332us **+63%** ✗ |
| `__launch_bounds__(32,4)` | 8164us ±0 |
| shuffle 除去（16本/iter） | 7877us -4% |
| fp4 expand 除去 | 8195us ±0 |
| shfl を c0/c1 で共有 + UE4M3 LDS LUT を arithmetic decode 化 | 8169us ±0（bit-exact） |

**どれも支配項ではない。** VGPR 74 / SGPR 32 / spill 0。

### 残る唯一の consistent な説明

```text
per 2 WMMA (k=32, 16x16 出力): epilogue 16 shfl+16 cvt+16 FMA / expand ~10 / load ~5
WMMA 発行 ~48 cycle に対し aux ~63 命令 → 命令数だけなら ~78 TFLOPS のはずが 44.6
総トラフィック: A = 10.5MB × 1088 out-tile = 11.4GB、W = 44.6MB × 64 = 2.85GB
               計 14.3GB / 8.17ms = 1.75 TB/s  ← L2 帯域クラスで rows に依らず一定
```

→ **L2 帯域が濃厚**だが、A を 1/4 にするはずの OB=4 が register 圧で負けたため未確定。
（OB=4 は acc が 8 本 = 64 VGPR になり occupancy/spill で相殺された可能性が高い）

### 結論と次の手

現行 GEMM は「1 warp/block・16x16 tile・global から直接 fragment」の設計で、
A 断片を out-tile ごとに読み直す（1088回）。LDS staging の無いこれが 21% の正体。

```text
1. L2 帯域をマイクロベンチで実測（1.75 TB/s が天井かを確定）
2. A 断片を LDS に staging して block 内の複数 out-tile で再利用（register を増やさない形）
   = OB=4 を acc を増やさずに実装し直す
3. multi-warp + LDS staging の tiled GEMM へ（hipfire/vLLM 型）
4. software pipeline（ib+1 の load/decode を WMMA の裏へ）
5. なお bf16 GEMM は重み 178MB > L2 64MB で weight 再読が本物 → MB 拡張が直接効く
```

### フォーマット側（別ブランチ `exp/rocmfpx-scale-factorization`）

epilogue を消すには scale を operand に畳む必要があり、それには 2の冪 scale (E8M0) が要る。
実測（27B の bf16 lm_head を正解にした重み再構成誤差）:

```text
ROCMFP4 Codebook10 + UE4M3/16 = 0.0984   (4.5 bpw)
MXFP4   E2M1 + E8M0/32        = 0.1894   (4.25 bpw)  ← 1.9倍悪い
INT8    per-16                = 0.0048   (8.5 bpw)
shared mantissa/32 の factorize = +11% RMS / per-64 = +14% → 精度は MXFP4 同等、速度はそれ未満で支配される
```

= **MXFP4 の ~3 倍は「重み誤差 1.9 倍」との交換**。ROCmFP4 の精度を守るなら 3 倍は取れない。


---

## 7.10 prefill GEMM の構造解: scale を operand へ畳む（LUT + fp8 WMMA）

§7.9 で「source-level ablation では律速を特定できない」ところまで書いたが、
**原因も解も特定できた**。結論から:

```text
犯人 = WMMA 直後にぶら下がる「16要素ぶんの scale 適用（mul + shfl）+ I32→F32 cvt」
       = per-16 UE4M3 scale が k-block ごとに変わるため、C を最後に一括スケールできない
解決 = 4KB の固定表 (code 4bit × scale 8bit → e4m3) を LDS に置き、
       表引き + pack で「scale 込みの fp8 オペランド」を作り、fp8 WMMA（f32 累算）へ渡す
       → epilogue が消える。MB を上げて再構築を割り勘にする
```

### PSQ8 への適用

PSQ8 は production 到達可能（27B-PSQ の `phaseshift_quantization.json` で
ffn_down 16 / gdn_out 16 / gdn_qkvza 32 / output / token_embedding の計 66 tensor
が psq8）であり、同じ 2D 構造を PSQ8 kernel にも適用した。

- 差分は W staging のみ: PSQ4 は packed 8B → CB10 decode が必要だが、
  PSQ8 は w0 = 8B 直 load、w1 = +256B の 8B 直 load で decode 不要。
  したがって PSQ8 の利得源は W global request の半減（legacy 2× → 1×）と
  A の半減（4× → 2×）だけになる（scale FMA / A layout / wmma / epilogue は共通）。
- resources: VGPR 154 / spill 0 / LDS 4608B。
- dispatch 条件は PSQ4 と同一。test_gemm_psq8_w8a8_wmma に同じ disp2d ケースを追加。

計測（interleaved 3 ラウンド、per-op、rows 256〜4096、7 shape、Δ は PSQ8 legacy 比）:

| shape | rows | legacy µs / TF | 2D Δ / TF |
|---|---|---|---|
| gate_up 17408×5120 | 2048 | 3737.4 / 97.7 | **−23.0% / 126.8** |
| mlp_down 5120×17408 | 2048 | 3379.3 / 108.0 | **−30.1% / 154.5** |
| attn_q 6144×5120 | 1024 | 422.3 / 152.6 | **−7.9% / 165.6** |
| attn_kv 1024×5120 | 2048 | 159.4 / 134.7 | **−12.3% / 153.6** |
| attn_o 5120×6144 | 2048 | 857.9 / 150.2 | **−8.4% / 163.9** |
| gdn_qkv 10240×5120 | 4096 | 3516.9 / 122.1 | **−18.8% / 150.4** |

集計: **WIN 29 / TIE 0 / LOSS 1**（唯一の負けは attn_kv r512 +11.5%）。
rows=256 と gdn_ba(n=48) は両バイナリとも ±1% で dispatch 正常。
PSQ4 より利得が大きいのは W バイト数が 2 倍で W request 削減の効きが強いため。

### 決定的だった3つの実測

**(1) Experiment A: epilogue を完全に消すと 2.44 倍（実カーネル・実メモリアクセス）**

| variant | r512 | r2048 | TFLOPS |
|---|---|---|---|
| 現行: iu8 + W4A8 epilogue | 2036us | 7514us | 48.6 |
| **fp8 WMMA + epilogue ゼロ** | **777us** | **3078us** | **118.6** |

天井 209.8 の 56%。vllm-radiance の ~120 TFLOPS/GPU と一致。
**epilogue が 2.44 倍の正体**だった（ロードではない: ロードを定数化しても -22% だけ）。

**(2) 分解: scale 適用 1.53x / cvt 1.60x**

| variant | r2048 | TFLOPS |
|---|---|---|
| 現行 | 7514us | 48.6 |
| iu8 + cvt だけ（scale 適用を除去） | 4912us | 74.3 |
| fp8 + epilogue ゼロ（cvt も消える） | 3078us | 118.6 |

`16 shfl + 16 FMA` が 1.53x、`16 cvt` が 1.60x。**どちらも効く。**

**(3) LUT 案（4KB 固定表 + gather/pack + fp8 WMMA）は 2.23 倍まで届く**

PoC は表の中身をダミーにした **timing のみ**の測定（数値はゴミ）。
`(MB,OB)` を振ると、再構築コストが `1/MB` で割り勘になることが実測で出る:

| (MB,OB) | r512 | r2048 | TFLOPS | 倍率 |
|---|---|---|---|---|
| 現行 (iu8+epilogue, 2,1) | 2036us | 7514us | 48.6 | 1.00x |
| LUT (2,1) | 1478us | 5479us | 66.6 | 1.37x |
| LUT (4,1) | 997us | 4139us | 88.2 | 1.82x |
| **LUT (8,1)** | **761us** | **3374us** | **108.2** | **2.23x** |
| LUT (4,2) | 841us | 3620us | 100.8 | 2.08x |
| 参考: 8bpw 焼き込み (2,4) | 777us | 3078us | 118.6 | 2.44x |

**8bpw 版（2.44倍）の 91% の速度を、メモリ 56% で取れている。**

```text
重みオペランドの再構築 = (ib, out-tile) ごとに1回、WMMA 数 = 2·MB·OB
→ 1 WMMA あたり ∝ 1/MB  （MB=2→4→8 で 66.6→88.2→108.2）
※ 現行 epilogue 版で MB=4 が ±0 だったのは、epilogue が「要素数」比例だから。
   LUT 版は要素数に依存しないので MB が素直に効く。
```

### 精度（27B の bf16 lm_head [248320,5120] を正解にした実測）

| scheme | weight 誤差 | bpw |
|---|---|---|
| ROCMFP4 CB10 + UE4M3/16（現行） | 0.0984 | 4.5 |
| **CB10+UE4M3/16 → scale を e4m3 へ畳んだ値（= LUT 案の operand）** | **0.1013 (+3%)** | **4.5** |
| CB10 → bf16 へ畳み | 0.0984（無損失） | — |
| plain E4M3（scale なし） | 0.0403 | 8.0 |
| MXFP4 E2M1 + E8M0/32 | 0.1894 | 4.25 |
| INT8 per-16 | 0.0048 | 8.5 |

**scale を e4m3 へ畳むコストは +3%（9.84% → 10.13%）**。
e4m3 の 3bit 仮数は「小整数 code × UE4M3 scale」に対して十分。
（なお `shared mantissa per-32/64` の factorize は +11%/+14% で、MXFP4 同等精度・それ以下の速度
 → 支配される。詳細は `tools/quantization/analyze_scale_factorization.py`）

### 残る唯一のコスト: 活性側

fp8 WMMA は両オペランド fp8 が要るので、活性を e4m3 に置き換える（int8 の追加ではなく置換でOK）。

```text
現行 A8: int8 + per-block scale → 0.48%
e4m3   : 3bit 仮数             → 4.03%
合計   : 重み 10.1% が支配的なので 9.8% → 約11%
```

### 到達点まとめ

```text
案                                    速度      bpw   重み誤差   TG        prefill(27B)
現行                                   48.6      4.5   9.8%       7.84      463 t/s
★ LUT + fp8 WMMA + MB=8               108.2     4.5   10.1%      7.84維持  ~1000 t/s
plain FP8（重みに焼き込み）            118.6     8.0   4.0%       ~4.4      ~1100 t/s
MXFP4                                  ~118      4.25  18.9%      7.84      ~1100 t/s
```

**TG を犠牲にせず、精度もほぼ維持したまま prefill 2.2 倍**を取れる唯一の案。

### 実装順（未着手）

```text
1. 4KB 折り畳み表 (code 4bit × scale 8bit → e4m3) を1回生成して device に置く
2. activation 量子化を int8 → e4m3 へ置換（内部 workspace。weight format は無変更）
3. kernel: LDS に表をコピー（4KB/block、8 store/lane）→ gather/pack で fp8 operand 生成
   → fp8 WMMA（f32 累算）+ MB=8（境界は MB テンプレで担保）
4. bench `--check` → モデル greedy token 一致
```

PoC の記録: branch `exp/rocmfpx-scale-factorization`
（`tools/quantization/analyze_format_accuracy.py` が上の精度表、
`analyze_scale_factorization.py` が factorize 案の否決）。
timing PoC のカーネル差分は本節の表のとおりで、実装時に再構成する。

### 7.10c 実装チェックリスト（step 2+3、opt ブランチ）

**設計（実測で確定）**

```text
重み : LUT(code, UE4M3/16 scale) -> e4m3、行 E8M0 で pow2 シフト
       精度 9.83%（現行 9.84% と同等）
活性 : e4m3 コード + per-row scale（f32 1個/行）
       精度 2.64%（per-32 なら 2.38% = +11% 相対のコスト）
GEMM : fp8 WMMA(f32 累算) + MB=8、epilogue は per-row 補正の1要素1回のみ
       速度 fp4 108.2 / fp8 118.6 TFLOPS（現行 48.6 / 27.8）
合計 : sqrt(9.83² + 2.64²) = 10.18%（現行 9.85% 比 +3.4%）
```

**活性 scale の粒度が本質**: per-32 のままだと k ブロックごとの補正が必須で
epilogue が消せない。per-row に落として初めて「1要素1回」になる。
e4m3 自身の指数が行内レンジを吸収するので、per-row でも 2.64% に収まる。

**落とし穴（fold LUT のテストで判明済み）**

```text
- e4m3 の上限 448 を超える (code × scale) は飽和 → 量子化機が per-16 scale を
  「ブロック最大 ≤ 448」に保つこと
- UE4M3 subnormal scale (指数部 0) は指数シフトが線形でない → per-16 scale を
  正規域 0x08..0x7e に保つこと
- 行スケールの下方向シフトは e4m3 の subnormal に落ちる → 小さいブロックを
  持ち上げる方向にのみ使う
```

**変更ファイル（1〜4 は同時に入れないと required が落ちる）**

```text
前提（実装済み）
  include/phaseshift/quantization/rocmfpx/fp4/fold_lut.h        (781ee58)
  src/phaseshift/quantization/rocmfpx/fp4/fold_lut.cpp
  include/phaseshift/quantization/rocmfpx/activation/e4m3_row.h (2d46791)
  src/phaseshift/quantization/rocmfpx/activation/e4m3_row.cpp

1. 量子化器（producer 2箇所を同時に）
   src/phaseshift/models/qwen35/kernels/quantization/activation_quantize_a8.gfx1201.hip
   model_dispatch_correctness.hip 内の exec_activation_quantize_a8
   → e4m3 出力 + per-row scale 返却（int8 化は削除）
2. workspace
   include/phaseshift/runtime/program/int8_activation_workspace.h
   scales を per-32ブロック(f32) → per-row(f32) に縮小
3. GEMM（fp4 経路）
   kernels/linear/gemm_rocmfp4_w4a8_wmma.gfx1201.hip
     LDS に 4KB 表 → gather/pack で e4m3 オペランド → fp8 WMMA → MB=8
     per-row 補正は store 時に1回
   kernels/linear/detail/fused_gemm_rocmfp4_wmma_detail.h（fused 3本が共用）
4. 参照側 GEMM
   correctness の fp4 GEMM: value = e4m3_decode(code) × row_scale
5. テスト
   test_activation_quantize_a8 / test_gemm_rocmfp4_w4a8_wmma / test_fused_*
6. fp8 層（別ステップ、LUT 不要）
   重みを e4m3(+per-row E8M0) に再量子化して素で読む = 118.6 TFLOPS
   精度 0.48% → 3.4%（感度の高い層なので層ごとに判断）
```

**検証**

```text
- bench `gemm --variant gemm --dtype rocmfp4 --check`（境界 rows を 1..2048 で）
- bench `gdn-recurrence`/`fused_linear` の check
- モデル greedy token 一致（4B / 27B）
- required 77/77
```



---

### 7.10b fold 精度の追試: pow2 グローバルスケールと scale 粒度

`e4m3 へ畳むと +3%` の内訳を追った結果、**ロスは仮数3bitではなく e4m3 の subnormal 領域**だった。
e4m3 は正規最小 2^-6、subnormal 下限 2^-9 なので、scale が小さい塊は fold 後の値が
subnormal に落ちて仮数ビットを失う。**出力行ごとの pow2 スケール**で持ち上げると消える。

| scheme | w_rel | 現行per-16比 | bpw |
|---|---|---|---|
| CB10 + UE4M3/16（現行・厳密 f32 scale） | 0.09835 | — | 4.5 |
| CB10+UE4M3/16 → e4m3（素の fold） | 0.10131 | +3.0% | 4.5 |
| **fold → e4m3 + pow2/16** | **0.09829** | **±0%** | **4.5** |
| CB10 + UE4M3/32（厳密） | 0.10193 | +3.6% | 4.25 |
| fold → e4m3 + pow2/32 | 0.10187 | +3.6% | 4.25 |
| CB10 + UE4M3/64（厳密） | 0.10706 | +8.9% | 4.125 |
| fold → e4m3 + pow2/64 | 0.10701 | +8.8% | 4.125 |

```text
pow2 スケールは粒度に関係なく fold のロスを完全に打ち消す:
  粒度16: 0.10131 → 0.09829 / 粒度32: 0.10520 → 0.10187 / 粒度64: 0.11013 → 0.10701
→ 「e4m3 へ畳む」操作は pow2 スケール付きならどの粒度でも無料

粒度を上げたぶんだけが残るロス:
  per-16 → per-32 : 重み誤差 +3.6% / メモリ -5.6%（27B 15.2GB → 14.4GB）/ TG ~+5%
  per-16 → per-64 : +8.8% / -8.3%
```

**実装メモ**: pow2 なので補正は指数の整数加算で済み、LUT の引数は `(code, scale)` のまま
（行スケールぶん指数をずらすだけ）。補正は**ストア時に1要素1回**なので epilogue は増えない
（per-k-block の補正にはならないことに注意。per-token/per-out の粒度でしか無料にできない）。
なお上の PoC は単一 Codebook10 のモデルで測っている。実際の `rocm_fp4_dual`
（16重みごとに codebook を選ぶ精度優先型）は絶対値が良くなるが、粒度間の相対差はそのまま乗る。



---

## 7.11 実装結果: LUT 方式は不成立、perm 展開 + epilogue FMA で 1.5 倍

§7.10c の「LUT + fp8 WMMA」を実装して実測した結果、**LUT 方式は成立しない**。

| 構成（17408x5120 r2048, gfx1201 32CU） | 時間 | TFLOPS |
|---|---|---|
| 現行（iu8 + per-k-block epilogue） | 7514us | 48.6 |
| 4KB LDS fold LUT + gather（MB=2） | 12027us | 30.2 |
| LUT を外して operand を定数に（構造の上限確認） | 3517us | 118.6 |
| **perm 展開 + epilogue FMA（採用）** | **5164us** | **70.7** |

**なぜ LUT が遅いか**: 1 出力タイル・1 k-block で 16 回の LDS バイト gather が要る
（operand 8 byte × 2 本）。LDS はバンク競合するので 1 gather ≈ 4 cycle、16 回 = 64 cycle
= WMMA 2 本（約 6 cycle）の 10 倍。LUT 方式の追加コストは実測 8.6ms = 全体の 70%。
operand はレジスタで作らないと間に合わない。

**PoC の 108.2 TFLOPS は計測不成立だった**: PoC の表は `lut[i] = i`（線形関数）で
作っており、コンパイラが gather を算術へ畳んでいた。LUT のコストは一度も計測に
入っていない。§7.10 の「scale を operand へ畳めばほぼ無料」はこの artifact の上に
立っていた。

**採用した方式**（`781ee58`/`2d46791` の fold LUT と `activation/e4m3_row` は削除）:

```text
重み operand = perm で fp4 ニブル -> e4m3 コードブックバイト
  旧 fp4_expand_low_high と同じ 4 perm + mask 構造で、定数 4 個だけ差替え。
  コードブック {0,1,2,3,4,6,8,10} は e4m3 で厳密に表現できるので重み誤差は現行と同一。
活性 operand = e4m3（per-row scale）。量子化器は op で分岐し、fp8 op 側は int8 のまま。
WMMA = fp8（f32 累算）で i32->f32 の cvt を削除。per-16 UE4M3 scale は
  epilogue で 1 FMA/要素/k-block、活性 per-row scale は最後に 1 乗算/要素。
```

**確定した限界**: fp8 operand が scale を運ばない限り per-element FMA は消えない。
scale を operand に入れる手段は (a) 実行時 fold（上記のとおり遅い）か (b) 8bpw への
焼き込みのみ。**4.5bpw のまま 2.2 倍は取れない。**

**実測（4B prefill, PP）**

| | baseline | 本変更 | 比 |
|---|---|---|---|
| PP512 | 167.75ms | 136.39ms | 1.23x |
| PP2048 | 932.29ms | 808.62ms | 1.15x |
| fp4 GEMM micro | 7514us / 48.6 | 5164us / 70.7 | 1.45x |

**精度**: 重み誤差は現行と同じ 9.83%。活性が 0.48% -> 2.64%（per-row e4m3）。
合計 9.85% -> 10.2%（+3.5% 相対）。4B e2e の oracle ゲート（GENERATED_IDS /
logits argmax / logits finite）は bf16・fp8 両方 PASS。`cli:interactive:turn2` の
不一致は baseline でも同じく落ちる既存問題。

**残るレバー**

```text
- MB=8: operand 生成を 8 タイルで償却。FMA は償却されないので fp4 GEMM で +10〜15%。
- fp8 層を e4m3 へ: 既に 8bpw なのでメモリ増ゼロで 27.8 -> ~118 TFLOPS (4.3x)。
- prefill の非 GEMM 成分: 4B PP2048 の線形仕事は ~28%（225ms / 805ms）。
  GEMM だけでは e2e 1.3 倍が上限で、attention / GDN / norm 側が本丸。
```

**教訓**: マイクロベンチの表は実際に使うデータで作る。ダミーの線形表はコンパイラに
畳まれて、存在しない速度を報告する。


---

## 7.12 MB は効かない（実装して否定）

§7.11 の「残るレバー: MB=8」と、そこで書いた「PoC の (2,1)=66.6 → (8,1)=108.2 は
LUT ではなく MB の効果」という見立ては、**実装して測った結果どちらも間違いだった**。

実装: fp4 カーネルを `template <int MB>` で MB ∈ {1,2,8} に拡張（重み operand 生成を
MB タイルで償却、行スケールは epilogue 側で shuffle）。正しさは 1267/1267 を確認済み。

| 構成（17408x5120 r2048） | 時間 | TFLOPS |
|---|---|---|
| MB=2（現行） | 5160us | 70.7 |
| MB=8（実装・検証済み） | 7320us | 50.1 |

**MB=8 の方が 1.42 倍遅い。** 重み operand 生成の償却より、acc 8 本（64 VGPR）+
`arow[8]`（16 VGPR）によるレジスタ圧と spill のほうが大きい。メモリ律速の見積り
（重み 44.5MB × 64 パス = 2.85GB → 1.9ms = 実測 5.16ms の 37%）も、MB で取れる分が
小さいことを示していた。

```text
結論: PoC の (2,1)/(4,1)/(8,1) 系列もダミー表 artifact の疑いが強く、
      「MB で償却すれば速くなる」は我々のカーネルでは成立しない。
      実測で効いたのは e4m3 operand 化（perm 展開 + per-16 FMA）の 1.45 倍だけ。
```

**残っている本当のコスト**

```text
- fp4 epilogue: perm 展開 + per-16 FMA = 1.65ms/5.16ms (32%)。epilogue ゼロなら 118.6。
  消すには scale を operand に入れる必要があり、実行時 fold は gather が遅く、
  8bpw 焼き込みはメモリ +78%。4.5bpw では消せない。
- fp8 層 (int8): 39% の GEMM 時間。epilogue が重い（i32->f32 cvt + 2 scale で 32 op/2WMMA）。
  形式を e4m3 にすると 4.3 倍期待できるが、この層は感度が高く量子化機が int8 を
  意図的に選んでいるため oracle が落ちる（実測: [220,17,1324] vs [220,19,11]）。
- prefill の非 GEMM 成分: 4B PP2048 の線形仕事は ~28%。attention / GDN / norm が本丸。
```

**追記: MB=4 は効く（MB=8 は overshoot）**

同一 shape（17408x5120 r2048）で MB だけを切り替えて直接比較:

| MB | min | p50 | TFLOPS (min) |
|---|---|---|---|
| 2（旧） | 5160us | 5531us | 70.7 |
| **4（採用）** | **4878us** | **5184us** | **74.8** |
| 8 | 7320us | 7564us | 50.1 |

MB=4 は acc 4 本（32 VGPR）+ arow[4] で spill せず、重み operand 生成の償却が勝つ。
MB=8 は acc 8 本（64 VGPR）で spill して負ける。採用条件は rows>=64 で MB=4、
rows>16 で MB=2、それ以下は MB=1。

4B prefill への寄与は ~+2%（fp4 GEMM は prefill の ~30% なので）で、e2e では
測定誤差内。他の shape でも 60-76 TFLOPS で正しさは 1267/1267。

**追記2: MB=6 も試した（MB=4 が最適点）**

| MB | min | p50 | 備考 |
|---|---|---|---|
| 2 | 5160us | 5531us | |
| **4（採用）** | **4755us** | **5179us** | VGPR 108 / spill 0 |
| 6 | 5177us | 5506us | 4 より遅い |
| 8 | 7320us | 7564us | VGPR 160 / spill 0 |

VGPR は **spill していない**（`-Rpass-analysis=kernel-resource-usage` で Spill: 0）。
遅くなる理由は spill ではなく **VGPR 数そのものによる occupancy 低下**（108 → 160）。
MB=4 は償却の利得が occupancy コストを上回る最後の点で、6 はもう越えている。

**追記3: なぜ gfx1201 で LUT しか手が無いのか（builtin 全洗い）**

`clang/docs/AMDGPUBuiltinReference` と `BuiltinsAMDGPU.inc` を全部見て、
gfx1201 での可用性を 1 つずつコンパイル確認した:

| builtin | gfx1201 | 備考 |
|---|---|---|
| `perm` (v_perm_b32) | OK | 8 byte 選択（2 レジスタ）。codebook は引ける |
| `cvt_pk_fp8_f32` / `cvt_f32_fp8` | OK | 素の pack / unpack |
| `permlane16/32`, `ds_bpermute` | OK | cross-lane。~128 byte の register file を address |
| `cvt_scalef32_pk_fp8_f32` | ✗ | `fp8-cvt-scale-insts` 無し |
| `cvt_scalef32_pk_fp4_f16/f32` | ✗ | `fp4-cvt-scale-insts` 無し |
| `cvt_scale_pk8_*_fp4` / `_fp8` | ✗ | `gfx1250-insts` 無し |
| `cvt_scalef32_pk8_fp8_f32` | ✗ | `gfx1250-insts` 無し |

**block scale 付きの変換命令（本命だったもの）は gfx1250 以降**。これがあれば
fold は `cvt_scale_pk8_f16_fp4` + `cvt_scalef32_pk8_fp8_f16` の 2 命令/8 byte で
済み、per-element FMA も perm 展開も消えて 118 TFLOPS に届く。gfx1201 には
1 つも無い。

したがって gfx1201 では `(scale × code) = 4096` を address できるユニットが
**LDS しか存在しない**。LDS は実測でレジスタ版に負ける（§7.11）ので、
「4.5bpw で scale を operand に畳む」はこの hardware では閉じている。

gfx1250 移植時の第一候補として `cvt_scale_pk8_*_fp4` 系を記録しておく。


---

## 7.20 int8 (W8A8) GEMM の次の一手（MB tiling 移植）

### 実測（`gemm --dtype rocmfp8 --variant gemm --rows 2048 --n 17408 --k 5120`）

```text
rocmfp8 (int8 iu8 WMMA) : 12346.8us = 29.6 TFLOPS
rocmfp4 (fp8  WMMA)     :  4713.7us = 77.4 TFLOPS
```

int8 はピークの ~20%。prefill の ~11%（27B で ~235ms）を占める。

### 診断

カーネル構造の差は 1 点:

```text
gemm_rocmfp8_w8a8_wmma.gfx1201.hip:
  __global__ void __launch_bounds__(32) gemm_rocmfp8_w8a8_wmma_kernel(...)
  → template 無し = 1 block が 1 タイル（16 行）。重み fragment の生成が M 方向に償却されない

gemm_rocmfp4_w4a8_wmma.gfx1201.hip:
  template <int MB> ... float8_t acc[MB]
  const uint32_t row_start = blockIdx.y * (MB * kTileRow);
  → 重み fragment を 1 回ロードして MB 個の M タイルで使い回す（MB=4/2/1 を shape で起動）
```

W8A8 の重みは 1 byte/要素（fp4 は 0.5）なので、この非償却がそのまま operand 生成コストに
出る。fp4 は同じ手（MB=4 のレジスタタイル）で 48.6 → 77.4 TFLOPS（+59%）になった。

### やること

1. `gemm_rocmfp8_w8a8_wmma_kernel` を `template <int MB>` 化
   - `acc[MB]`（iu8 の i32 アキュムレータ）、A operand は per-M-tile、重み fragment は k-step ごとに 1 回
   - `row_start = blockIdx.y * (MB * kTileRow)`、epilogue の scale 適用も per-M-tile
2. launcher で MB=4/2/1 を shape で切り替え（fp4 と同じ判定・同じ grid の y 次元計算）
3. 計測: 29.6 → 60-75 TFLOPS を狙う（fp4 の 77.4 が目安）
4. 検証: `test_gemm_rocmfp8_w8a8_wmma`（required）、E2E、`--dtype rocmfp8` の TFLOPS

補足: W8A8 は epilogue で scale を掛ける契約なので、MB 化してもその順序を変えないこと。

### 実測: MB タイル化は逆効果だった（2026-08-14、revert 済み）

`template <int MB>`（MB=4/2/1）で重み fragment を 4 M タイルで償却する版を実装・計測:

```text
rows=2048 n=17408 k=5120, gemm variant
  before (MB なし) : 12346.8us = 29.6 TFLOPS
  after  (MB=4)    : 18311.7us = 20.0 TFLOPS   <- 遅くなった
```

つまり **重み operand の生成は int8 のボトルネックではない**。重みタイルは row-tile 方向の
ブロック間で同じものを読むので L1/L2 に乗っており、in-block での償却ではほとんど得しない。
一方で `float8_t acc[MB]` + per-tile の `int32x8_t c` でレジスタが増え、1 warp/block の
占有が落ちる分がそのまま出る（fp4 版が MB で勝つのは、あちらが acc を増やしても
それ以上に operand 生成（fp4 の 4→8bit 展開）が重いため）。

### 訂正: shfl fold 仮説も外れた（ablation 実測）

`const float ws = __shfl(ws_self, src_lane);` を `ws = ws_self;` に置き換える ablation
（値は狂うが命令数だけ見る）:

```text
baseline          : 12346.8us = 29.6 TFLOPS
shfl 除去 ablation: 14994.3us         <- 遅くなった
```

shfl を 8 個/k-step 消して遅くなる = **fold の shfl は律速ではない**。MB=4 の逆効果と
合わせて、これまで 2 つの仮説（重み operand 生成 / fold の shfl）が両方外れた。
推測を重ねるのは止めて、まず計測する。

### 実測: WMMA のピークは ~230 TFLOPS（i32 == fp8）

`/tmp/opencode/wmma_peak.hip`（1 warp/block、独立アキュムレータ 8 本、64CU×4=256 SIMD）:

```text
i32_iu8     1 warp/blk: 1.467 ms  228.8 TFLOPS
f32_fp8_fp8 1 warp/blk: 1.450 ms  231.4 TFLOPS
i32_iu8     4 warp/blk: 1.421 ms  236.1 TFLOPS
f32_fp8_fp8 4 warp/blk: 1.397 ms  240.2 TFLOPS
```

**`v_wmma_i32_16x16x16_iu8` は `v_wmma_f32_16x16x16_fp8_fp8` と同等**。int8 が遅いのは
命令種のせいではない。ピーク 230 に対し int8 GEMM は 27 TFLOPS = **11%**、fp4 は 70 = 30%。

### 原因: 独立なアキュムレータ鎖が 1 本だけ（レイテンシ律速）

カーネルは 1 warp = 16x16 タイル 1 枚。k-step ごとに

```text
c = wmma(w0,a0,c=0); c = wmma(w1,a1,c);   // 2 本が直列
8 x cvt_f32_i32(c[j]);                     // c を読み切るまで次の wmma が始まらない
```

なので per warp のループは「WMMA 2 本ぶんのレイテンシ」に縛られる。マイクロベンチは
8 本の独立鎖で 230 TFLOPS 出るので、鎖を増やせば伸びる。

**MB タイル化が効く理由**がこれ。以前 MB=4 を試して 18311us（29.6→20.0）と悪化したのは、
`__shfl(ws_self, ...)` を t ループの内側に置いて **shfl を 4 倍にしていた**ため。shfl を
k ループ側に 1 回だけ巻き上げると逆転する。

### 実装: `template <bool EXACT, int MB>` + shfl 巻き上げ

`gemm_rocmfp8_w8a8_wmma.gfx1201.hip`:

```text
- template <bool EXACT, int MB> 化
- EXACT は rows%16==0 && out_features%16==0 && logical_k%32==0 で、
  load_i8x8_frag の byte 単位フォールバックと out_valid/row_valid をコンパイル時に消す
- acc[MB] / arow[MB] / asrow[MB] / row_valid[MB]
- ws_reg[8] = __shfl(ws_self, 8*k_group + j) を t ループの外へ（k-step ごとに 1 回）
- launcher: tiles>=8 で MB=4、tiles>=2 で MB=2、それ以外 MB=1
```

計測:

```text
rows=2048 n=17408 k=5120
  before : 12346.8us = 29.6 TFLOPS
  after  :  8700.0us = 42.5 TFLOPS   (1.42x)
```

`--check` は 6 shape（exact / 非exact / 極小）で mismatch=0 max_rel=0。required 75/75、e2e 2/2。

### 負の結果（全部 revert 済み）

```text
MB=8              : 9049us（VGPR 192 で占有低下）      -> MB=4 が最適
grid 軸入替        : 9680us（weight 再利用を狙ったが逆）  -> 元の (out, row) のまま
ws を 16B ロードに  : 17002us（shfl をメモリロードに置換） -> shfl の方が速い
EXACT のみ         : 13374us（単体では効かない、MB と併せて意味が出る）
```

### 本番は fused 版だった（pp は動かない）

`launch_gemm_rocmfp8_w8a8_wmma_gfx1201` は最適化 dispatch の片方の経路でしかなく、
モデルは `fused_gemm_rocmfp8_wmma_detail.h` 経由の
`gemm_qkv_/gemm_gdn_proj_/gemm_gate_up_rocmfp8_w8a8_wmma` も使う。同じ MB=4 を
そちらへも移植した（`template <int MB, bool EXACT>`、3 launcher は rows>=64 で MB=4）。

ただし:

```text
27B pp 2048 : 変更前 2094.5ms -> 変更後 2096.4ms（変化なし）
```

**int8 GEMM 単体が 1.42x になっても prefill は動かない** = 27B prefill における int8 W8A8 の
比率は小さい（以前の「int8 ~11%」は fused 前の見積もりで、実際の本番寄与はこれより小さい）。
次のレバーは fp4 GEMM（43%、ピーク比 30%）と GDN。

### 次の一手: まずプロファイル

```text
1. ISA / occupancy を見る（compile_commands.json + --save-temps=obj）
   - vgpr_count / sgpr_count / group_segment_fixed_size / scratch spill
   - 1 warp/block（32 threads）なので、ブロック数と wave 数の比も確認
2. L1 トランザクションを見る（rocprof が使えるなら wave/L1/TA のカウンタ）
   - 重み/活性とも 8B の gather（load_i8x8_frag）で、lane ごとにアドレスが散る。
     これが L1 トランザクション律速なら、4B/8B gather の coalescing 改善が本筋
3. MB=2 は「レジスタ圧を下げつつ重みを償却」だが、上記 2 仮説が外れている以上
   先にプロファイルしてから判断する（MB は fold にも L1 にも効かない）


k-step ごとの **スカラー fold** が重い:

```text
per k-step (32 k) あたり、1 lane で:
  iu8 WMMA ×2（8192 MAC/warp）
  load  a0/a1（8B gather ×2）+ w0/w1（同）×2 + scale 2
  fold: j=0..7 で __shfl(ws_self, 8*k_group + j) ×8 + f32 FMA ×8
        + i32 -> f32 変換 ×8
→ WMMA 1 回あたり ~12 命令が fold 側に乗る（fp4 は同じ fold を LUT 済み ws で回しており
  shfl がない）
```

やること:

```text
1. `__shfl(ws_self, 8*k_group + j)` を消す。重み scale は lane ごとに静的に決まるので
   fp4 の `ue4m3_lut[wsrow[...]]` と同様、k-step の外で per-lane に前計算してレジスタに持つ
2. fold の i32->f32 変換と FMA を per-j でなくベクトル化できる形にまとめる
3. 計測は 29.6 -> 40+ TFLOPS を狙う（fp4 の 77.4 は W8A8 の構造では届かない可能性が高い）
```

revert 済み。MB 案は docs に負の結果として残す。


---

## 7.21 全最適化カーネルの ISA 走査と k-loop の命令削減（2026-09-14）

### 手法

`build-v/compile_commands.json` の全 `.gfx1201.hip` を `--save-temps=obj` で再コンパイルし、
`.s` から (a) `.vgpr_count` / `.vgpr_spill_count` / `.amdhsa_group_segment_fixed_size`、
(b) 最内ループ本体を抜き出して命令クラス別に集計した。ランタイムは rocprofv3 のカーネル別
集計（busy 1524ms）と突き合わせた。

デバイス実測（`hipDeviceProp_t`）:

```text
CUs=32  maxThreadsPerCU=2048  regsPerCU=196608(DWORD)  sharedMemPerCU=65536B  clock=2.35GHz
```

LDS は WGP で 128kB / CU で 64kB（ISA 資料 §1.2.2.1）。つまり 50kB を取る
`gdn_recurrence_wmma` は 1 block/CU に固定される。

### 判明した無駄と対応

| kernel | ms | × | VGPR | LDS | 最内ループの実態 | 対応 |
|---|---:|---:|---:|---:|---|---|
| `gemm_rocmfp4_w4a8` | 826 | 336 | 108 | 1KB | 180 命令 / 8 WMMA。22 命令が `out_valid` ガード (`s_and_saveexec`+`s_cbranch_execz`+ゼロ初期化 `v_mov`)、24 が 64bit ポインタ前進、28 が fp4 展開、46 が scale FMA | 済 |
| `gemm_rocmfp8_w8a8` | 312 | 64 | 109 | 0 | 163 命令 / 8 WMMA。32 `v_cvt_f32_i32` + 34 f32 scale + 20 ptr + 8 `ds_bpermute` | 済 |
| `gdn_recurrence_wmma` | 106 | 48 | 234 | **50180B** | 2361 命令 / 42 WMMA。508 FMA + 321 LDS load + 215 `s_wait_dscnt` | 未 |
| `gdn_prepare` | 76 | 48 | 25 | 80B | 214 命令/行。70 stall + 41 SALU。有効 block 80 = 1.25/CU | 未 |
| `gdn_norm_gate` | 62 | 48 | 16 | 4B | tid0 の 128 要素直列和 | 済 |
| `attention_paged_prefill_bf16` | 35 | 16 | 121 | 29208B | 1853 命令 / 12 WMMA。225 個の 64bit アドレス演算 + 76 `s_mov` + 53 `v_dual_mov` + 43 exec マスク分岐 | 未 |
| `activation_quantize_e4m3` | 34 | 208 | 12 | 32B | 2 pass + `global_load_u16`（336GB/s = 帯域 52%） | 未 |
| `swiglu` | 22 | 64 | 32 | 0 | `v_div_scale_f32`×16 だが 623GB/s で DRAM 律速 | 不要 |
| `activation_quantize_a8` | 16 | 48 | 14 | 0 | 1 lane = 1 要素、2byte ロード（314GB/s） | 未 |
| `residual_add` | 12 | 128 | 18 | 0 | bf16 RNE エミュ ~10 命令/2 要素。677GB/s で DRAM 律速 | 不要 |
| `rmsnorm` | 5.6 | 129 | 17 | 32B | 要素ごとに `global_load_u16` + 64bit アドレス | 未 |

### 効いた手（すべて実測付き）

```text
1. out_valid / rv を block 一様な full_tile で k-loop 外へ出す（template lambda 2 版）
   fp4 rows=2048 n=17408 k=5120: 5213.6 -> 4190.1us (-19.6%)
   ※ full_tile は lane 依存の out_valid を使ってはいけない（N が 16 の倍数でない shape で破綻）
2. k_group*8 をポインタ生成時に hoist（毎 k-step の 64bit 加算 5 ペアを削除）
3. k-loop を 2 段 unroll（load -> 展開 -> wmma -> scale の鎖を重ねる）
4. MB=8 タイル（W 側コストを 8 M タイルで償却。MB=16 は 256 VGPR + spill で逆効果）
   fp4: 5213.6 -> 3619.3us (70.1 -> 101.5 TFLOPS)
5. fp8 にも 2+3 を適用。MB=8 は i32 acc 8x8 で溢れるため不採用
   fp8 rows=2048 n=6144 k=17408: 11695 -> 8614us (37.2 -> 50.9 TFLOPS)
6. gdn_norm_gate: block を 4 行分にし、tid0 の直列和を 4 本 interleave（加算順序は reference と
   同一なので bit-exact を維持）。61.6 -> ~46ms
```

27B PP2048: **1914.6 -> 1421.3ms（1069.7 -> 1441.0 t/s、-25.8% / +34.7%）**。
WMMA ピーク 230 TFLOPS（§7.20）に対し fp4 は 101.5 = **44%**。

### 残っているもの

```text
1. fp4 の per-16k UE4M3 scale 適用 = カーネルの 22%（ablation 実測 3619 -> 2823us）
   1 出力要素 × k-16 ごとに FMA 1 本が本質。消すには scale を重みへ畳む format 変更が要る（精度と引き換え）
2. gdn_recurrence 106ms: LDS 50kB -> 1 block/CU。32kB 以下への削減は kWv 64->32 でも ~39kB で届かない
3. gdn_prepare 76ms: q/k は行分割可（出力が別 f32 バッファ）。v は in-place エイリアスのため
   conv_out への raw ステージ等が要る
4. attention prefill 35ms: ページコピーループの i/kChunksPerRow 除算エミュレーション除去
5. quantize 50ms: b128 ロード化（LSU 律速）
```


---

### gdn_norm_gate の bit-exact 制約（重要）

`test_gdn_norm_gate` は optimized と reference の **bf16 完全一致**（`mm == 0`）を要求する。
reference は tid0 の線形 float 和なので、warp 縮約に置き換えると 1 要素だけ ULP がずれて落ちる
（実測 mm=1）。`gdn_prepare` が同じ置き換えで通るのは、そちらの縮約結果が f32 出力（相対 2e-2
許容）にしか効かないため。順序を変えずに ILP だけ上げるのが正解。

## 7.22 E4M0 + codebook 形状の実データ比較（4B weight、exp）

「UE4M3 を捨てて E4M0 にできるか」「codebook 形状は何がいいか」を実データで測った。
`tools/quantization/analyze_e4m0_codebook.py`（4B の bf16 ソースから 6 tensor × 24 行、
43008 block / 688128 weight、16 weight ごとの scale を 256 候補で全探索）。

scale 候補を全バリアントで同じ 256 個に揃えて比較（E4M0 だけは定義上 16 個）。

| scheme | rms rel err | mean blk rel |
|---|---|---|
| cb10 + ue4m3（現行 ROCmFP4 dual-scale） | 0.08951 | 0.09679 |
| cb10 + e4m0（16 指数のみ） | 0.10657 | 0.10316 |
| **cb10 + e4m0 + sub4（1/16 octave）** | **0.07671** | 0.07523 |
| cb10 + free scale（上限） | 0.07584 | 0.07435 |
| learned CB + ue4m3 | 0.08795 | 0.09404 |
| **learned CB + e4m0 + sub4** | **0.07578** | 0.07369 |
| learned CB + free scale | 0.07493 | 0.07280 |
| nf4 形状 + e4m0 + sub4 | 0.10845 | 0.10775 |

`|w|/(max/10)` のヒストグラム（0-1:17% / 1-2:16% / ... / 9-10:9%）で、block 内は
ほぼ一様。つまり **誤差の支配項は codebook の形ではなく scale の表現精度**。

結論:

1. **E4M0 単独（8bit 中 4bit が octave）は現行より 19% 悪い**。1 octave 粒度は粗すぎる。
2. **E4M0 + 4bit log subscale（= 8bit log scale、1/16 octave）は現行 UE4M3 より
   14% 良い**（0.0767 対 0.0895）。metadata は同じ 8bit/16weight。UE4M3 の
   mantissa 3bit（≒6.25% 粒度）より 1/16 octave（4.4%）が細かいため。
   上限（free float scale）との差は 1% しかなく、ほぼ最適。
3. **codebook 形状の効果は約 2%**（k-means で学習しても 0.0895→0.0880、
   0.0767→0.0758）。NF4 形状はこの正規化ではむしろ悪化。**形状を追う価値は薄い**。
4. したがって「ROCmFP を捨てる」判断の根拠を精度に求めるのは無理がある。
   残るコストは「16 weight ごとの scale を演算で適用すること」で、これは
   scale が log でも linear でも同じく per-element の積（または LDS lookup）が要る。
   lookup に逃がす案は §7.13 で既に負けている。
5. 唯一の無料の改善は **指数部の整数加算**（E4M0 を FP8 operand の exponent に足す）。
   sub-octave 分は依然として積が要るので、現行の UE4M3 fetch+cvt+FMA から
   「exponent add + sub4 の積 1 本」に置き換わる分だけが得。要計測。

### 7.22.1 「scale を足し算にして掛け算を消せるか」の答え

FP8 E4M3 の exponent フィールドは biased なので、**pow2 部分は整数加算**で掛けられる
（4 byte まとめて packed add、zero/subnormal/overflow は select が要る）。
問題は sub-octave (1/16 octave) で、これは pow2 ではないので加算にできない。
pow2 だけにした場合の実測:

| scheme | rms rel err |
|---|---|
| cb10 + ue4m3（現行） | 0.08951 |
| cb10 + e4m0（pow2 のみ） | 0.10657 |
| learned CB + e4m0（pow2 のみ） | 0.10417 |
| cb10 + e4m0+sub4 | 0.07671 |

**pow2 のみ = 掛け算ゼロは達成できるが +16% 精度劣化**。codebook を pow2 制約下で
学習しても 0.10417 までしか戻らない。つまり「掛け算を消す」と「現行精度」は両立しない。

一方で構造上の事実: 現行 kernel の WMMA は `wmma_f32_16x16x16_fp8_fp8_w32_gfx12`
（k=16）で、ROCmFP4 dual-scale の scale block も 16。**k-step と scale block が一致**
しているので、scale は operand 側ではなく **accumulator 側**に適用できる
（`Σ_b s_b (A_b·W_b)`、k-step ごとに acc を 1 回スケール）。ただし acc 要素数も
operand byte 数も per-lane では同じ 8 なので、**命令数は同じ**。利点は
(a) f32 側なので subnormal/overflow/zero の例外処理が不要、
(b) pow2 部分は f32 exponent の整数加算で済む、の 2 点のみ。

結論: **scale の適用は 1 要素あたり 1 演算が下限**で、現行 ROCmFP4 は既にその
下限を払っている。フォーマット側の変更（UE4M3 → log scale）で実行時間は減らない。
精度は +14% 良くなる（0.0767 対 0.0895）ので、余った開発予算を精度側に回す意味はある。

### 7.22.2 channel scaling (su) は pow2 scale の損失を消せない

§5 の Matrix-Level Channel Scaling を実データで判定。`su[i]=amax_i^a`（入力
チャネル毎、正規化）、block 化の前に割って reconstruction を元ドメインで評価
（`tools/quantization/analyze_su_channel_scaling.py`、4B、8 tensor、43008 block）。

| scheme | su=none | su=amax^0.5 | su=amax^1.0 |
|---|---|---|---|
| cb10+ue4m3/16 (4.50bpw, 現行) | 0.08951 | 0.08628 | 0.08924 |
| cb10+ue4m3/32 (4.25) | 0.09846 | 0.09245 | 0.09499 |
| **cb10+bf16scale/32 (4.50)** | 0.08952 | **0.08169** | 0.08171 |
| cb10+f16scale/32 (4.50) | 0.09253 | 0.08467 | 0.08539 |
| **cb10+e8m0/32 (4.25, pow2)** | 0.11453 | **0.10672** | 0.11720 |

結論:

1. su は全 scheme で効く（amax^0.5 が最良、3〜12% 改善）。
2. **pow2 scale の損失は su では消えない**。e8m0/32 は su を入れても 0.10672 で、
   同じ 4.5bpw の bf16scale/32 + su（0.08169）より 24% 悪い。乗法的な octave 誤差は
   block の中身を変えても動かないため。
3. **最良は cb10 + per-32 16bit scale + su = 0.08169**。現行（per-16 UE4M3）の
   0.08951 より 8.7% 良い。metadata は同じ 4.5bpw、しかも scale 適用は半減し
   UE4M3 LUT も不要。

→ §5 は「pow2 を救う道具」ではなく「16bit scale を伸ばす道具」。順序は
scale 解像度の確保が先。


---

## 7.23 PSQ4 W4A8 の scale fold 実測と「epilogue 1.82x」の訂正（2026-09-14）

対象: `gemm_psq4_w4a8_wmma.gfx1201.hip`。GPU3 (R9700)、rows=2048、warmup20/samples30。

### 前提だった数字が誤りだった

以前 §7.10 あたりで使った「per-32 scale FMA を消すと 1.82x（2950→1618us）」は
**DCE（dead code elimination）による偽計測**だった。ablation を

```cpp
for (j) acc[t][j] = c[j];   // 上書き。累算しない
```

としていたため、`ib` の最後の 1 回しか結果に効かず、k-loop 本体ごと消えていた。
「累算」を消した測定であって「scale 適用」を消した測定ではない。

### 正しい天井の実測

scale 適用を fold ではなく「完全に消す」probe（数値不正・timing only、
WMMA の C で全 K を累算、LUT は固定）:

| 構成 | n=17408 k=5120 | n=5120 k=17408 | 比 |
|---|---|---|---|
| 現行（per-32 bf16 scale + 抽出 FMA） | 2908us | 2851us | 1.00x |
| scale 作業ゼロ probe（天井） | 2590us | 2535us | **1.12x** |
| FOLD E8M0（固定 LUT + 指数 shift + WMMA 累算） | 2644us | 2593us | 1.10x |
| FOLD eighth（cvt で per-block LUT 生成） | 3467us | — | 0.83x |
| accumulator 2 本化（MB=8） | 3173us | 3031us | 0.92x（spill） |

asm 内訳（FOLD eighth、MB=8）: `v_perm` 42 / wmma 48 / cvt 30 / lshl 164 / or 142 / mul 70。
重いのは cvt ではなく **LUT のビット packing**。FMA（`v_dual_fmac`）は 0 になった。

### 判定

- PSQ4 fp8 WMMA 経路では **scale 適用はカーネルの 11% しか無い**。
  fold・per-row・per-128 など scale 粒度の変更は全部この 11% の取り分話で、
  上限は 1.12x。format 変更（quantizer / loader / validator / manifest）を
  正当化できない。
- 精度コストとの比（27B、5 tensor RMS、bf16/32=0.08857 を基準）:
  - FOLD eighth/32 E4M3 = +6.7%、見合う速度ゲインはよくて 9% → 採算 1.3
  - FOLD E8M0 = +26.1%、ゲイン 10% → 採算 1.0 未満
  - per-row scale = +24.5%、天井 12% → 却下
  ※以前の「採算 12.2」は 1.82x 前提の上の値なので無効。
- W4 の LUT を fp32 経由でなく固定表 + 指数演算で作る道（phase 表 perm + byte 加算、
  scale に 2^B bias を入れて下位バイトの subnormal を回避）は成立するが、
  出ても 1.10x 台なので着手しない。

→ **PSQ4 の scale fold 路線は却下。現行フォーマット（cb10 + per-32 scale）を維持。**
律速は k-loop 本体（decode perm + WMMA issue）なので、次に行くならそこ。


---

## 7.24 scale 保存形式の再測定（bf16 / f16 / e5m2 / e4m3）と cb10 vs e2m1（2026-09-14）

`analyze_scale_schemes.py` に `sfmt`（実際の IEEE 丸め + 2段階 scale 探索）と
codebook 引数 `cb10` を追加。27B・rows=24・alpha=0・5 tensors。

### scale 保存形式（codebook ごと）

| scheme (per-32) | e2m1 | cb10 |
|---|---|---|
| free（1/64 octave grid、格納制約なし） | 0.08859 | 0.08578 |
| **bf16（探索込み・現行実装）** | **0.08867** | **0.08585** |
| f16（探索込み） | 0.08874 | 0.08594 |
| e5m2（1byte） | 0.09511 | 0.09231 |
| e4m3（1byte） | 0.10222 | 0.10748 |

- bf16 = free と実質同値。**スケール保存Formats の解像度はもう律速ではない**
  （quantizer が bf16 表現値を ±0.5 octave 探索しているため、§7.9〜7.22 の
  baseline 数値は実質 bf16 実装と同じ値だった）。
- f16 は mantissa 10bit あるが bf16 と差ゼロ（+0.1% はノイズ）。
  逆に f16 は exponent 5bit なので scale 下限で subnormal に落ちうる。
  → **bf16 のままが正。f16 にする理由はない。**
- e4m3（1byte）が e5m2 より悪いのは scale 下限（2^-9=0.002）で
  27B の下位 octave が subnormal/ゼロに沈むため。1byte 化するなら e5m2 側。
- 注: §7.22 の「bf16 > f16」は band_pow2 近似（log2 一様grid）の偽差。
  実丸めでは両者同値。

### codebook: cb10 = {0,1,2,3,4,6,8,10}

- e2m1 比 **-3.2%** RMS（alpha=0）、su=0.5 併用で **-6.0%**。bpw は同じ 4.50。
- 8 振幅構成は同じなので **GEMM の decode 命令数は完全に同一**
  （`psq4_e2m1_low_high` の LUT 定数 4 本を
  0x44403800/0x52504C48/0xC4C0B880/0xD2D0CCC8 に差し替えるだけ）。
  CB10 の全値は E4M3 で厳密（8=0x50, 10=0x52）。
- 代价はフォーマット変更そのもの: quantizer の code 写像・validator・
  tests・formats doc・**27B 再量子化**。実行時コストはゼロ。

→ 速度狙いの fold 路線（§7.23）は却下だが、**cb10 化は「無料の精度 3〜6%」**として
成立する。次のフォーマット変更（例: W8 側整理）に便乗で回すのが安い。

### CB10 移行の実装完了（2026-09-15）

quantizer / kernel LUT / manifest codebook ガードを CB10 へ移行し、
27B を再量子化して実ペイロードを検証:

- 実ペイロード dequant RMS（6 tensors、全行）= **0.08612**（予測 0.0859、E2M1 実測 0.0886）
- `test_gemm_psq4_w4a8_wmma` 1267/1267、4B quantizer e2e 35 gates 通過
- 速度無変化（GEMM p50 2854us vs baseline 2908us、pp 1785 t/s vs 1778 t/s）
- 旧 E2M1 モデルは manifest の codebook 必須チェックで明確に拒否される


---

## 7.25 decode PSQ4 GEMM の in-block split-K（2026-09-15）

### 前提の是正

- 本機の HBM read peak は **~670GB/s**（256MB cold read、grid2048×block512）。
  以前「~640GB/s」としていたが誤差範囲。128MB を read で flush しても
  L3（**R9700 は L3=64MB を持つ**）を完全追い出せているかは shape 依存。
  microbench は **2×256MB 交互 flush** で計測すること。
  1×128MB flush のままだと weights が L3 残留し **1TB/s 超の偽高速**を
  観測する（S16 系で 2.5 倍の差が出た）。
- 27B の per-token psq4 GEMM shape（safetensors 実 shape）:
  gate/up **17408×5120** ×128（gate_proj / up_proj が別々に 17408）、
  down 5120×17408 ×48（psq4 分）、qkv 10240×5120 ×32、
  z 6144×5120 ×32、out 5120×6144 ×32、full q **12288×5120** ×16
  （q_proj は q 6144 + output gate 6144 の 12288）、
  full o 5120×6144 ×16、k/v 1024×5120 ×32。
  合計 336 GEMM/token。
  gate/up の 17408×5120（1088 blocks）と full q の 12288×5120
  （768 blocks）は split-K 全 variant が PROD（cold 97.4us / 72.8us、
  ~514 / ~486GB/s）に勝てないため従来パス。
  per-token psq4 重み read 合計 ~11.9GB。
- rocprofv3 の kernel_trace の grid_size フィールドは work-item 数
  （grid × block）で報告される（grid の実値 = /block.x）。

### 設計

- rows==1（decode）かつ特定 (n,k) の場合、`gemm_psq4_w4a8_wmma` の
  launcher が in-block split-K カーネル `gemm_psq4_w4a8_wmma_block_splitk_kernel<S>`
  を選ぶ。block=32S threads、wave w が k の [w·nb/S, (w+1)·nb/S) を担当。
- 還元は **block 内 shared memory**（`red[S][16]`、rows==1 だと
  k_group==0 lane の acc[0] のみ有効）。第 2 カーネルも partials
  バッファも不要。bf16 splitk カーネル（a/b GEMM）と同じパターン。
- 採用 S: down(5120×17408)→4、out/o(5120×6144)/qkv(10240×5120)/
  z(6144×5120)→8、k/v(1024×5120)→16。gate/up(17408×5120) と
  full q(12288×5120) はどの S も負けるため従来パス維持。
- 非 NT 採用（NT は +0.6〜1.0us/shape 程度で、NT 版は FMA 调度が変わり
  結果が非 bit-identical になる）。

### 結果（2×256MB 交互 flush、p50、cold）

| shape | PROD | split-K | 削減/token |
|---|---|---|---|
| k/v 1024×5120 | 32.1us | S16 15.4us | 534us |
| down 5120×17408 | 103.1us | S4 94.3us | 422us |
| out 5120×6144 | 48.2us | S8 43.4us | 230us |
| qkv 10240×5120 | 62.8us | S8 61.1us | 54us |
| z 6144×5120 | 43.0us | S8 41.9us | 35us |
| gate/up 17408×5120 | 97.4us | —（従来） | 0 |
| full q 12288×5120 | 72.8us | —（従来、S8 は 72.3 で差なし） | 0 |

- e2e（tg、交互 A/B 3 ペア）: **44.34–44.52 → 42.65–42.87 ms/tok**
  （21.1 → 23.4 tok/s、−1.65ms/tok）。
- 正しさ: `test_gemm_psq4_w4a8_wmma` 通過。tg の最終 argmax 一致（271）。
  split-K は f32 の和の結合順が変わるため最終 logits は bit 非一致
  （max|Δlogit|=3.3 / 248320、top-3 一致、top-50 重複 39）。
  e4m3 量子化で 1ulp 差が増幅されるのが原因で、並列 K 還元の本質的挙動。
  greedy トークン一致（argmax）は維持。


---

## 7.26 e4m3 activation quantize の producer 融合（2026-09-15）

### 前提

- decode 1 tok あたり **258 回**の standalone `e4m3` activation quantize
  （per-row、grid=1 block × 256 thr、2 pass）が ~16us/回 = 4.1ms/tok
  を占める（rocprofv3 kernel_trace 実測）。
- 直前 producer は rmsnorm（136）/ swiglu（64）/ gdn_norm_gate（48）/
  attention_gate（16）の 4 種。各カーネルの y（bf16）は shared memory
  に既に存在するため、quantize を producer 側に折叠できる。

### 設計

- 新カーネル群 `fused_e4m3_quant.gfx1201.hip`（rmsnorm / swiglu /
  gdn_norm_gate / attention_gate × e4m3 各 1 個、per-row block 256 thr +
  dynamic shared `y_shared[k_padded]` bf16）。
- standalone チェーンと **bit-identical** を保証するため op 順序を完全再現:
  - rmsnorm: `wave_sum_down`（shfl_down 総和）で rms 計算、
    y = x·inv·w の bf16 出力を必ず書く（GDN 入力 norm の bf16 出力は
    a/b の BF16 linear が消費するため dead ではない）。
  - e4m3 tail は共通 `e4m3_row_tail`（peak は shfl_xor、scale = peak/448、
    swizzled code 書込）。
  - gdn fused の per-group inv は「スレッド g が自分の group を逐次総和」
    （standalone thread-0 総和と同一順序）で bit-identical。
- graph matcher（`act_quant_matcher`）: producer カーネル + 隣接
  activation_quantize の 2 dispatch（gdn / attention は 3 dispatch）を
  パターン認識し、fused dispatch に置換。quantize は **producer+1 隣接**
  必須（executor は consumed 分だけ dispatch を前進させるため、
  中間 dispatch を再実行できない）。
- recognizer に RmsNormQuant / SwigluQuant パターン追加、
  trace カウンタ `act_quant_fused` 追加。

### 結果

- 258/tok の quantize が全融合（trace `act_quant_fused=25856` / 101 step、
  standalone quantize = 0）。LOGITS bit-identical
  （argmax=271 max=10.836942 sum=-845382.688）。
- e2e（tg、交互 A/B 3 ペア）: **37.62–37.77 → 36.37–36.47 us/tok**
  （26.47–26.58 → **27.42–27.50 tok/s**、−1.28ms/tok）。
- テスト: ctest qwen35 19 件 + gemm/rmsnorm/quant/fusion GPU 9 件全通過。


---

## 7.27 GEMM 帯域フェーズの負結果とカーネル下限（2026-09-15）

### 到達上限の修正

- HBM pure-read 上限は shape 依存: **256MB で 670GB/s、2.53GB で
  636.6GB/s**（同 flush 方式）。2.5GB 超の weight stream の評価は
  後者の数字で見る。

### lm_head bf16 GEMM: 16B load は microbench 効くが e2e 効かない（却下）

- `gemm_bf16_wmma` の rows==1 経路（lm_head 248320×5120 のみ）で
  8B×2/load を 16B×2/load + unroll4 に変更（W16U4、bit-identical）。
- cold microbench（2×256MB 交互 flush、p50）: **487.8 → 551.4 GB/s**
  （5324 → 4611 us、−713us）。D2（2 行/lane）は 298GB/s で大敗。
- e2e 交互 A/B 3 ペア: 36.27–36.35 vs 36.29–36.42 ms/tok。
  **差なし（むしろ +22〜67us 劣）**。
- 判定: e2e で利得が出ていないため却下・revert。
  microbench の利得は e2e の clock / power 状態では転移しない
  （wide load の瞬時帯域増加 → 省電力降下クロックの相殺と推測）。
  以降の GEMM 微調整は **e2e A/B を受理基準**にする。

### psq4 gate/up 17408×5120: 新 variant 4 種も全滅（再確認）

- cold microbench（p50）: PROD 97.1us / P8U4（unroll4）160.0 /
  P16U4（16B load + unroll4）128.2 / P16U2 123.2 /
  S2（in-block split-K 2 warp）97.6。全て PROD 以下。
- 1088 block の短い stream では unroll 増が occupancy を落とし、
  16B load の partner 冗長分が L1 読み出し量を増やすため不利。
- gate/up / full q の従来パス維持を最終確定。

### fused elementwise は cold 下限到達

- fused カーネルの cold microbench 下限: swiglu 20.2us / gdn 25.2us /
  attngate 18.9us。e2e 実測（21us 台）とほぼ一致 → L3 干渉が原因では
  なく、1 block × 2 pass + global scale 往復 + swizzled byte 書込の
  構造自体が下限。
- 残る唯一の尾は tail の global scale 往復（thread0 が global 書込 →
  全スレッド global 読込）を shared memory 読みに置換（bit-identical、
  値は同一 f32、global 書込は GEMM 用のため維持）。
  e2e 交互 A/B 3 ペア: 36.29–36.45 → 36.26–36.41 ms/tok
  （3/3 ペア改善、平均 −25us/tok ≒ 258 回 × 0.1us で期待値と一致）。
  LOGITS bit-identical。採用。

### a/b bf16 GEMM（in_proj_a/b 48×5120）: unroll4 も e2e 効かない（却下）

- a/b は 1 GEMM 491KB、grid 3 block（out 48 = 16×3）、S8 split-K。
  cold microbench 18.2us（27GB/s）で HBM レイテンシ律速。
- variant（cold p50、全て bit-identical）: PROD 18.20 / U4 14.68 /
  U8 14.68 / W16U2 14.80 / W16U4 14.68。unroll4 が −3.5us。
- e2e 交互 A/B 3 ペア: 36.25–36.37 vs 36.31–36.42 ms/tok。
  **差なし（2/3 ペアで +50us 劣）**。revert。
- 原因: e2e では a/b 重み（491KB × 48 layer）が L2 残留し
  10.8us（cold 18.2us より 7us 速い）で実行されており、
  unroll が救う cold HBM レイテンシが e2e に存在しない。
  lm_head W16U4 と同型の「cold microbench 利得が e2e に転移しない」
  ケース。以降、この類の GEMM 微調整は cold microbench ではなく
  **e2e A/B だけで受理判断**する。


---

## 7.28 residual_add の GEMM epilogue 融合（2026-09-15）

### 前提

- decode 1 tok あたり `residual_add_bf16` が **128 回**（exec 1.94us +
  カーネル間 gap を含む実効コスト計測で ~2.8us/回 = 0.36ms/tok）。
- 直前 producer は必ず GEMM（out_proj / o_proj / down）で、出力は
  タイル内で完結するため epilogue 融合が bit-identical 可能:
  `bf16(bf16(acc) + f32(residual))` は standalone
  `bf16(f32(bf16 GEMM out) + f32(residual))` と同一。
- 27B（CB10）では対象 GEMM は全て **LINEAR_PSQ8**（out/down）。
  psq4 の splitk も併せて対応した。

### 設計

- `gemm_psq4_w4a8_wmma`（main + splitk）/ `gemm_psq8_w8a8_wmma`（main）に
  `residual` / `residual_row_stride` 引数を追加（nullptr = 従来の動作）。
  epilogue の BF16 書込直前に加法。residual の 16B 行は L2 hot で
  追加コストは無視できる。
- launcher は impl 分離で `launch_*_residual_gfx1201` を追加
  （既存呼び出し側・テストは不変）。
- 新 matcher `linear_residual_matcher`: LINEAR_PSQ4/PSQ8 + 隣接
  RESIDUAL_ADD（RA 入力 0 が GEMM 出力、GEMM 出力の consumer = 1）
  を認識。recognizer に `FusionPattern::LinearResidual` 追加。
  recognizer の case 列挙に LINEAR_PSQ8 が無かったため追加
  （psq8 GEMM は multi-linear fusion 対象外だが残存融合の対象）。
- fused 時は GEMM が RA の出力 value（新 residual バッファ）へ直接書込、
  consumed=2 で RA dispatch を消費。dtype は BF16 限定
  （不整合なら NotApplicable で従来パスへ fallback）。

### 結果

- trace: `residual_add_bf16` dispatch **0 件**（128/tok 消滅）。
  LOGITS bit-identical（argmax=271 max=10.836942 sum=-845382.688）。
- e2e（tg、交互 A/B 3 ペア）: **36.22–36.36 → 35.88–36.00 ms/tok**
  （27.50–27.61 → **27.78–27.87 tok/s**、−0.36ms/tok）。
- 注: 事前見積は gap 4.5us/回込みで −0.82ms だったが実効は 2.8us/回。
  GEMM 直後の小 kernel の gap は trace の平均値より小さい。
- テスト: ctest 77 件全通過。


---

## 7.30 fused e4m3 量子化カーネルの latency 改善（unroll / thread 数）（2026-09-15）

### 背景

§7.29 の per-kernel 実測（rocprofv3, 101 step 平均）で、融合 e4m3 量子化
カーネル群が **1 block（grid = rows = 1）の latency-bound** であることが
判明した。実要素数に対し 5〜10 倍遅い:

| kernel | launches/tok | mean | per-tok |
|---|---|---|---|
| fused_rmsnorm | 128 | 13.22us | 1.69ms |
| fused_swiglu | 64 | 20.62us | 1.32ms |
| fused_gdn_norm_gate | 48 | 20.61us | 0.99ms |
| fused_attention_gate | 16 | 11.94us | 0.19ms |

いずれも 256 thread 単一 block で、要素ループが `i += kThreads` の逐次
global load 依存になっており、1 block = 8 wave では latency を隠せない。

### 実装 1: `#pragma unroll 4`（bit-identical）

`fused_e4m3_quant.gfx1201.hip` の elementwise ループと共有 e4m3_row_tail
（peak / encode）に `#pragma unroll 4` を追加。1 thread あたり 4 独立 load を
in-flight にし、逐次依存を解消する。演算順序は不変。

結果（trace, per-tok）:

| kernel | before | after |
|---|---|---|
| fused_swiglu | 1.32ms (20.62us) | 0.90ms (14.10us) |
| fused_gdn_norm_gate | 0.99ms (20.61us) | 0.69ms (14.29us) |
| fused_attention_gate | 0.19ms (11.94us) | 0.10ms (6.33us) |
| fused_rmsnorm | 1.69ms (13.22us) | 1.62ms (12.69us) |

e2e A/B 3 ペア（交互・同一ウィンドウ）:
- base: 40.78 / 40.87 / 40.94 ms/tok
- unroll: 39.89 / 39.99 / 40.02 ms/tok
- 差 **−0.90ms/tok（−2.2%）**、LOGITS bit-identical
  （argmax=271 max=10.836942 sum=-845382.688）。

fused_rmsnorm は unroll が効かない（sum 還元 + bf16 write + weight load が
支配的）ため、thread 数側で別途対応する。

### 実装 2: kThreads 256 -> 512

1 block あたりの wave 数を 8 -> 16 に倍増し latency hiding を改善する。
`kThreads` はファイル内の 4 kernel と e4m3_row_tail が共有するため
1 定数変更で全 kernel に効く。

結果（trace, per-tok）:

| kernel | §7.30 前 | unroll 後 | +512 thread |
|---|---|---|---|
| fused_rmsnorm | 1.69ms (13.22us) | 1.62ms (12.69us) | **1.03ms (8.06us)** |
| fused_swiglu | 1.32ms (20.62us) | 0.90ms (14.10us) | **0.62ms (9.67us)** |
| fused_gdn_norm_gate | 0.99ms (20.61us) | 0.69ms (14.29us) | **0.46ms (9.55us)** |
| fused_attention_gate | 0.19ms (11.94us) | 0.10ms (6.33us) | **0.07ms (4.50us)** |

1 block の wave 数が 8 -> 16 になり latency hiding が改善。
trace 合計は 38.8 -> 36.89 ms/tok。GEMM 群（psq4 main / psq8 / splitk）は
両 trace で同一（±0.1%）で、変化はこの 4 カーネルのみ。

e2e A/B 3 ペア（交互・同一ウィンドウ、tg_plain = §7.30 前の元バイナリ）:
- base: 40.87 / 40.91 / 40.97 ms/tok
- new（unroll + kThreads512）: 38.84 / 38.90 / 38.95 ms/tok
- 差 **−2.02ms/tok（−4.9%）**。内訳は unroll −0.90ms + kThreads −1.12ms。

数値: kThreads 変更で rmsnorm の sum 還元順序が変わるため bit-identical では
ない。2 token 比較で argmax 一致（198）、max/sum は 0.15%/0.4% 差
（e4m3 量子化ノイズ増幅で説明できる範囲）。96 token greedy では途中分岐するが
最終 LOGITS argmax=271（production 基準）は一致するため §7.26 と同じ基準で受容。


---

## 7.31 lm_head (Output role) の PSQ8 量子化（2026-09-15）

### 背景

§7.29/§7.30 の per-kernel 実測で LM head が **5.01ms/tok（1 launch）** と
最大の単一カーネルだった。理由は Output (lm_head.weight) が BF16 のまま
だったため（248320x5120 = 2.54GB を毎 token 読む）。

### 変更

1. quantizer: `fpx/profile.cpp` の `psq()` で `case TensorRole::Output:` を
   BF16 群から PSQ8 群へ移動（`--preset psq` の lm_head が psq8 になる）。
   再量子化: `Qwen3.8-27B-PSQ-CB10-OUT8`
   （payload 19.50GB -> 18.31GB、bpw 5.711 -> 5.362、psq8 tensor 65 -> 66）。
2. runtime: `linear_selector.cpp` の shape rule に
   `{Psq8W8A8, 248320, 5120}` を追加。**これが無いと selector が
   Correctness を返し、PSQ8 化した lm_head が reference kernel に落ちて
   逆に遅くなる**（rule は family x out_features x k の完全一致）。

### 実測

同一バイナリ・同一ウィンドウ・3 ペア交互（model dir のみ差し替え）:

| | ms/tok | tok/s |
|---|---|---|
| Qwen3.8-27B-PSQ-CB10 (lm_head bf16) | 38.83 / 38.91 / 38.90 | 25.72 |
| Qwen3.8-27B-PSQ-CB10-OUT8 (psq8) | 35.93 / 35.95 / 35.99 | **27.81** |

**−2.93ms/tok（−7.5%）**。LM head GEMM は weight 2.54GB -> 1.27GB になり
bf16 の ~507GB/s から psq8 の ~460GB/s へ移っても実時間は約半分になる。

LOGITS は argmax=271 で不変、max 12.510947 -> 12.512733、
sum -742006.375 -> -742399.938（8bit 化の影響はこの程度）。

### 累積（本セッション）

| 段階 | ms/tok | tok/s |
|---|---|---|
| セッション開始（低速ウィンドウ） | 40.92 | 24.44 |
| §7.30 unroll + 512 thread | 38.90 | 25.71 |
| §7.31 lm_head PSQ8 | 35.95 | 27.81 |


---

## 7.32 PSQ GEMM k-loop の unroll 2 -> 4（2026-09-15）

### 背景

§7.31 後の trace で最大項目は psq4 main GEMM
（`gemm_psq4_w4a8_wmma_kernel<1>`, 145.9 launch/tok, mean 85.5us = 12.5ms/tok =
step の ~35%）。重み 22MB/launch を 85us = 259GB/s でしか流せていない。
一方 psq8（`gemm_psq8_w8a8_wmma_kernel<1>`）は 44.5MB/96us = 464GB/s。

差は **1 iteration あたりの独立 load 数**（psq4: 8B/thread、psq8: 2x8B/thread）。
k-loop の `#pragma unroll 2` を 4 に上げ、in-flight load を倍にする。

### 実装

- `gemm_psq4_w4a8_wmma.gfx1201.hip`: main kernel と splitk kernel の k-loop を
  `#pragma unroll 2` -> `#pragma unroll 4`。
- 演算順序は不変（accumulator ごとの加算順は同じ）で LOGITS bit-identical。

### 実測（OUT8 モデル、3 ペア交互）

| | ms/tok |
|---|---|
| unroll 2 | 35.94 / 35.99 / 36.02 |
| unroll 4 | 35.42 / 35.47 / 35.48 |

**−0.53ms/tok（−1.5%）**、LOGITS bit-identical。

### 追記: psq8 k-loop も unroll 2 -> 4

`gemm_psq8_w8a8_wmma.gfx1201.hip` の同ループも unroll 4 に。
A/B 3 ペア: 35.37-35.46 -> 35.18-35.28 ms/tok（**−0.19ms**）、bit-identical。

なお 27B の形状を整理すると（`phaseshift_quantization.json`）:
- gate/up 17408x5120 psq4 x64 each → psq4 main (unroll 4 で ~520GB/s)
- ffn_down 5120x17408: psq8 x16 + psq4 x48
- gdn_qkvza 10240/6144x5120: psq8 x32 + psq4 x64、gdn_out 5120x6144: psq8 x16 + psq4 x32
- attn_q 12288x5120 psq4 x16 は splitk rule が無く main 経路（k=5120 で nb=160）

1 token あたりの重みストリームは約 14.3GB（step 35.2ms → 平均 ~406GB/s）。
per-kernel では psq4 main / splitk4 が ~520-533GB/s、psq8 が ~417GB/s、
pure streaming の実測上限は 623GB/s（§7.19）。よって GEMM 側の残り
ヘッドルームは 15〜20% 程度。


---

## 7.33 見送り: swiglu/attngate の per-tile scale 融合、gdn_prepare の GEMM epilogue 融合（2026-09-15）

### swiglu/attngate (当初計画の ②)

当初は swiglu (64 launch/tok) と attngate (16 launch/tok) を producer GEMM の
epilogue に吸収し、e4m3 量子化を per-k-tile scale に変える計画だった
（当時の trace 値: swiglu 20.6us / attngate 11.9us）。

実装前に関連カーネルを読んで consumer 側のコストを見積もった結果、**見送り**:

- psq8/psq4 GEMM の acc fragment は「列 = m_lane、行 = 8*k_group+j」で、
  activation scale は **トークン行方向** に効く。per-row scale は k-loop の外で
  8 shfl すれば済む（現行実装）。
- per-k-tile scale にすると、k-tile ごとに acc の 8 行へ scale を配る必要があり
  1 tile あたり 8 shfl + 8 fma が k-loop 内に入る。down GEMM (k=17408, 272 tile)
  では k-loop の ~30 op/iter に対し +16 op/iter ≈ **+50%（≈ +3ms/tok）**。
- per-ib の weight scale に activation tile scale を fold する案は
  `wsa = ws * as[ib]` の 1 load + 1 mul で済むが、weight は全 row 共有なので
  **rows==1 のときだけ正しい**（multi-request decode / prefill で破綻）。
- 収支: 融合で消えるのは swiglu 64×(9.7us+gap) + attngate 16×(4.5us+gap)
  ≈ 0.8ms。consumer 側 +0.4ms 以上、rows==1 限定の条件付き。ネット ≤0.4ms。

同じ ~1ms は §7.30 (unroll/thread 数)、§7.31 (lm_head PSQ8)、§7.32 (GEMM unroll)
でより安く取れた（合計 −3.6ms）ため、優先度を下げた。

### gdn_prepare (当初計画の #2)

- gdn_prepare は 3.25us × 48 launch = **0.16ms/tok** しかなく、上限が小さい。
- このカーネルは conv1d + silu + q/k の L2 正規化 + q_scale を融合しており、
  L2 正規化は **head_dim=128 を跨ぐ**。GEMM epilogue は 16-wide タイルなので
  そのままでは閉じない（128-wide タイルの専用カーネルが必要）。
- ROI が見合わないため見送り。

### 追記: unroll 8 は悪化（不採用）

同じ k-loop を unroll 8 にすると 35.19-35.25 -> 35.28-35.34 ms/tok で
**+0.09ms 悪化**（3 ペアとも同符号）。VGPR 圧が上がりスピルするため。
unroll 4 を維持。


---

## 7.34 量子化レシピの感度: psq8 を減らす（OUT8 / G8）（2026-09-15）

quantizer の `fpx/profile.cpp` `psq()` の `li % 4 == 0` 条件を動かした 2 モデル。

| モデル | 変更 | payload | psq8 tensor | bpw |
|---|---|---|---|---|
| PSQ-CB10 (元) | lm_head bf16 | 19.50GB | 65 | 5.711 |
| **OUT8** | `Output` を PSQ8 群へ | 18.31GB | 66 | 5.362 |
| **G8** | さらに GdnQkvza / GdnOut を `li%8` | 17.85GB | 42 | 5.227 |

同一バイナリ A/B 3 ペア:

| | ms/tok | tok/s |
|---|---|---|
| PSQ-CB10 | 38.83 / 38.91 / 38.90 | 25.72 |
| OUT8 | 35.93 / 35.95 / 35.99 | 27.81 |
| G8 (vs OUT8) | 34.85 / 34.43 / 34.46 | **28.92** |

- lm_head PSQ8 (OUT8): **−2.93ms**（weight 2.54GB → 1.27GB）
- GdnQkvza/GdnOut の psq8 を 1/8 層に (G8): **−0.77ms**
  （psq8 2.56GB → 1.28GB だが、置換先の psq4 は splitk<8> 経路で
   ~400GB/s なので実効は半分程度）
- LOGITS は G8 で argmax=271 不変、max 12.512733 → 12.436271、
  sum −742399.938 → −757292.562（4bit 増の影響。KLD 評価は未実施）。

### 累積（本セッション、すべて同一ウィンドウ内 A/B）

| 段階 | ms/tok | tok/s |
|---|---|---|
| 開始 | 40.92 | 24.44 |
| §7.30 unroll + 512 thread | 38.90 | 25.71 |
| §7.31 lm_head PSQ8 (OUT8) | 35.95 | 27.81 |
| §7.32 psq4/psq8 GEMM unroll | 35.35 | 28.29 |
| §7.34 G8 | 34.58 | 28.92 |

合計 **−6.34ms/tok（−15.5%）、24.44 → 28.92 tok/s**。

### テスト

本セッションの変更後 `ctest --test-dir build`: **77/77 passed**（GEMM psq4/psq8、
fused gate_up、embedding、e2e 4B oracle を含む。GEMM k-loop unroll は
argmax/bit 比較の対象テストを通過）。


---

## 7.35 PP2048: §7.32 の unroll 4 が MB=8 で spill を起こしていた（2026-09-15）

### 背景

main（a5c2b0b）での PP2048 実測が 1189-1201ms（1705-1721 t/s）。
`rocprofv3 --kernel-trace` の内訳は psq4 GEMM が 67.4%（1080 launch / 3 pass、
1884us/launch）で単独最大。

`extract_kernel_resources.py` で ISA メタデータを見ると、§7.32 で k-loop を
`#pragma unroll 2 -> 4` にした影響が **MB=8（rows>=128 = prefill 経路）だけ
spill 化**していた。

| kernel（main a5c2b0b） | VGPR | vgpr_spill | scratch | code size |
|---|---|---|---|---|
| `gemm_psq4_w4a8_wmma_kernel<8>` | 256 | **265** | **504B** | 45892 |
| `gemm_psq8_w8a8_wmma_kernel<8>` | 256 | **199** | **480B** | 44916 |
| 同 `<4>` / `<2>` / `<1>` | 211 / 129 / 87 | 0 | 0 | — |

§4 で「PSQ4 MB=8 は VGPR 255・spill 0」と
記録した状態から、unroll 4 で VGPR が上限を割って spill に転落した。
decode 経路（MB=1/2/4・splitk）は spill のままで、tg だけが改善し
prefill が悪化していた形になる。

### 実装

`gemm_psq4/psq8` の k-loop unroll を MB 依存にした（decode 側の利得は維持）。

```text
constexpr int kUnroll = (MB >= 8) ? 2 : 4;
```

結果（MB=8）:

| kernel | VGPR | vgpr_spill | scratch | code size |
|---|---|---|---|---|
| psq4 `<8>` | 256 | **3** | **16B** | 34264 |
| psq8 `<8>` | 254 | **0** | **0** | 32560 |

### 実測（G8 27B、device 3、交互 3 ペア）

PP2048（`--runs 3 --warmup 1`）:

| | gpu_ms_median | t/s |
|---|---|---|
| unroll 4（main） | 1189.8 / 1195.2 / 1201.1 | 1721 / 1714 / 1705 |
| MB=8 unroll 2 | **1040.6 / 1045.7 / 1048.0** | **1968 / 1959 / 1954** |

**−149ms（−12.5%）、+14.4%**。3 ペアとも同符号。

TG128（context128 / tokens128、交互 2 ペア）は無変化で decode 利得を維持:

| | us/tok |
|---|---|
| unroll 4（main） | 34464.8 / 34495.3 |
| MB=8 unroll 2 | 34467.6 / 34514.5 |

unroll 1 にすると 1062.0 / 1069.9ms で unroll 2 より **+17ms 悪化**（2 ペア）。
in-flight load が減るため。MB=8 は unroll 2 を採用。

数値: LOGITS bit-identical（argmax=271 max=12.838641 sum=-700806.188）。
unroll 段数は accumulator ごとの加算順を変えない。

### テスト

`test_gemm_psq4_w4a8_wmma` / `test_gemm_psq8_w8a8_wmma` 通過（1267 ケース系）。

### 教訓

**unroll / thread 数系の micro 最適化は「全 template instantiation の
vgpr_spill_count を見直す」までが作業。** decode だけ trace すると
prefill 側の instantiation を踏んだまま merge する。


---

## 7.36 iMatrix collector 実装と量子化レシピへの適用（2026-09-16）

### 実装

- `phaseshift-quantizer imatrix`: 校正corpusをmodelに流し、per-channel Σx² を収集して
  PSIM fileを書く。runtimeは `LinearNode` 系dispatchの入力activationを `ImatrixProbe`
  としてprogramへ登録し、collectorがattachされている時だけ蓄積する
  （productionは collector == nullptr の1分岐のみ）。
- 7 site（full-attn qkv / attn o_proj / GDN projection / GDN out_proj / MLP gate+up /
  MLP down / lm_head）を `lower_to_primitives` でtagし、site→tensor名は
  `model_sites.cpp` が持つ。lm_headは最終norm出力（TOKEN_ROWS）を観測する。
- 校正中は fused elementwise path を無効化して全siteのactivationをmaterializeする
  （非fusion時のkernel出力を使う）。既定で有効、`--with-fusion` で戻せる。
- `phaseshift-quantizer ppl`: teacher-forced NLL/perplexity。quality比較用。

### 校正

`tools/quantization/build_phaseshift_imatrix_corpus.py`（PhaseShift repo由来、
1.51M chars）→ `prepare_kld_corpus.py` でPSKLDTOK化（545,809 tokens）。

| 項目 | 値 |
|---|---|
| tokens / windows | 545,809 / 534（window=1024, stride=1024） |
| 所要 | 12m48s（27B, PSQ-CB10-OUT8 を校正モデルに使用） |
| entries | 401（7 site × 64 layer + lm_head、期待値と一致） |
| 未収集 | mtp層 / token_embedding（best-effortでunweightedへfallback） |

### 精度（27B, 同一bpw 5.3623・速度同一）

teacher-forced NLL（4096 tokens, teacher forcing decode）:

| 評価corpus | OUT8 | OUT8-IM | Δ NLL |
|---|---|---|---|
| 校正corpus | 1.292532 | 1.273124 | **−1.50%** |
| held-out（site-packages のPython、校正に未使用） | 0.797455 | 0.791148 | **−0.79%** |

- 512tokenブロック8本すべてで同符号。
- 速度: 35.12 / 35.16 / 35.16 ms vs 35.15 / 35.19 / 35.19 ms（A/B 3ペア、有意差なし）。
- 校正corpusでの改善が大きいのは当然（過大評価）で、held-outの −0.79% が実力値。

### KLD（4B, balanced preset, held-out 8192 tokens）

校正corpusは同じPhaseShift corpus（4B bf16で再校正、545,809 tokens）、評価は
held-out（site-packages のPython、SHA 27e7831a...）。baselineは元bf16（shadow経由）。

| モデル | mean KLD | p50 | p90 | p99 | PPL(bf16) | PPL(QDQ) | 劣化 |
|---|---|---|---|---|---|---|---|
| 4B BAL（imatrix無し） | 0.054071 | 0.009735 | 0.122905 | 0.593856 | 3.5687 | 3.7390 | +4.77% |
| 4B BAL-IM | **0.047272** | 0.008350 | 0.105852 | 0.528614 | 3.5687 | 3.6973 | **+3.60%** |

- mean KLD **−12.6%**、p50/p90/p99 すべて改善、bpw は 5.5185 で同一
- 量子化によるperplexity劣化（bf16比）が **+4.77% → +3.60%（−24.6%）**
- 27B の KLD は baseline/candidate とも bf16 shadow（52GB）が必要で本機材では不可。
  27B は `ppl` による相対比較のみ（上記 −0.79%）。


---

## 7.37 iMatrix で G8 レシピ（GDN psq8 を li%8）の品質を回復（2026-09-16）

§7.34 の G8（GdnQkvza/GdnOut を psq8 にする層を 1/4 → 1/8、−0.77ms、payload −0.46GB）は
品質を落とす。同一校正（`out8_full.psim`、7 site）を G8 に適用して比較した。

held-out（site-packages のPython、4096 tokens、teacher-forced NLL）:

| モデル | bpw | payload | mean NLL | ppl | ms/tok | tok/s |
|---|---|---|---|---|---|---|
| PSQ-CB10-OUT8 | 5.362 | 18.31GB | 0.797455 | 2.2199 | 35.33 | 28.30 |
| PSQ-CB10-OUT8-IM | 5.362 | 18.31GB | 0.791148 | 2.2059 | 35.33 | 28.30 |
| PSQ-CB10-OUT8-G8 | 5.227 | 17.85GB | 0.801085 | 2.2280 | 34.56 | 28.93 |
| **PSQ-CB10-OUT8-G8-IM** | 5.227 | 17.85GB | **0.794195** | 2.2127 | **34.56** | **28.93** |

- G8 化の劣化は +0.46%（OUT8比）。imatrix で **OUT8 より −0.41% 良い NLL** まで戻る。
- 速度 A/B（3ペア、OUT8-IM vs G8-IM）: 35.31/35.34ms vs 34.58/34.60ms（pair1 は warm-up 由来の
  37.5ms外れ値を除く）＝ **−0.77ms**、§7.34 と一致。
- 結論として `psq` preset は GdnQkvza/GdnOut を `li % 8` に変更（`--imatrix` 併用が前提）。
- 採用モデルはこの G8-IM で、`models/Qwen3.8-27B-PSQ-IM` として配置（17.85GB / 5.227 bpw）。

---

## 7.40 BF16 の preshuffle（tile layout）: GEMV と WMMA GEMM の layout トレードオフ（2026-09-17）

gfx1151 (Strix Halo / Radeon 8060S、40 CU / LPDDR5X) の BF16 モデルで、
量子化（PSQ4/8）がロード時に強制する **preshuffle（16 行 tile native layout）** を
BF16 に適用する案を検証。**却下（natural layout 維持）**。

### 前提: BF16 は preshuffle していない（自然 row-major）

- BF16 weight は `load_bf16_matrix` で **natural row-major、padding なし**（`strides={cols,1}`、`k_padded=cols`、`upload_tensor_direct`）。
- 量子化は `preshuffle_native` で canonical `[rows][nb][block]` → native **`[tile16][nb][16 行 × block]`**（16 行 tile、k-block ごと 16 行が連続）。
- WMMA 16×16 tile は A=weight を 16 行×16 k の tile として読む。natural layout だと 16 行が `k*2` bytes 間隔で **32B チャンク・5KB スライドの散乱 load** → 64B セクタ効率 ~50% で律速（GEMV も GEMM も同じ病）。

### 仮説テスト（同一 WMMA カーネル、weight layout だけ切替）

| rows | natural | tile | 比 |
|---|---|---|---|
| 1 (decode) | 144 GB/s | 188.6 GB/s | 1.31× |
| 16 | 118.6 GB/s | 175.2 GB/s | 1.48× |
| 128 (gate_up 9216×2560) | 20.6 GB/s (2.6 TF) | 49.5 GB/s (6.3 TF) | **2.40×** |
| 128 (down 2560×9216) | 58.0 GB/s (7.4 TF) | 123.5 GB/s (15.8 TF) | **2.13×** |

- **GEMV は row-major 有利、WMMA GEMM は preshuffle 有利**（`docs/tegaki.md` の記述通り）。
- tile WMMA decode (188.6 GB/s) は専用 GEMV (221 GB/s) の **~85%**。
  差の要因: WMMA は 30 VGPR（専用 GEMV は 20）→ occupancy 低下。
  16 feature/warp 計算なので 16 個の累加器が必須 → 低 VGPR 化不可。

### E2E A/B（実機、interleaved、Qwen3.5-4B BF16）

| 指標 | natural (base) | preshuffle (cand) | 比 |
|---|---|---|---|
| PP128 (prefill) | 329 ms | 129 ms | **2.55× 改善** |
| TG (decode) | 44.2 ms/tok | 53.8 ms/tok | **22% 劣化** |

crossover は **P ≈ 6×G**（長 prompt では preshuffle 有利、長生成では natural 有利）。

### 却下の決め手: 共有 weight の正しさ崩壊（2× memory 必須）

- **greedy token 不一致**: base 14168 / cand 579340（破綻）。
- 原因: `tie_word_embeddings` で **embedding と LM head が同じ weight（`embed_tokens.weight` [vocab][hidden]）を共有**。
  embedding lookup は weight を **natural `[vocab][hidden]`** として読むが、preshuffle が tile layout に並替える → embedding が散乱・誤読。
- 修正には **共有 weight の 2× copy（1.27GB）**が必要（メモリ哲学に反する）。
  per-layer weight は GEMM のみで使用（tile 可）なので共有 weight だけの問題だが、2× copy は非現実的。

### 教訓

- **1 つの weight layout で GEMV（1 行連続）と WMMA GEMM（16 行 tile 連続）は両立しない**。natural / tile は排他的。
- preshuffle は prefill を 2.5× 上げるが、decode を 22% 落とし、しかも **共有 weight（embedding）の正しさを壊す**。
- 反復コスト（decode per-token）が支配する primary（chat、P < G）では **natural layout（GEMV）が net win**。
  preshuffle は long-prompt（P > 6G）専用 + 共有 weight 2× memory + decode 劣化の代償。
- **「少数 rows の tiled GEMM ≈ row-major GEMV」は ~85% まで**（VGPR/occupancy の差）。等速は期待できない。


---

## 7.41 MTP（M>1）での preshuffle 検証: 将来機能用として価値を確認（2026-09-17）

§7.40 で「primary（chat、M=1）では natural layout が net win」と結論した preshuffle。
**MTP / 投機実行で forward に検証する token 数 M が 1 → 2,4,8,... に増える**
シナリオについて検証。**M>1 では preshuffle（tile）が明確に価値を持つ**ことを確認。

### 検証方法（M 軸の microbench）

同一 shape（gate_up n=9216 k=2560）で、`gemm_bf16_wmma1151_tile`（preshuffle / tile 布局）と
**weight 再利用 SmallGEMM**（行優先、1 wavefront = 1 output feature × M 行、
`w = load_weight()` を 1 回読んで M 行で再利用）を M = 2,4,8,16 で計測。
weight 読み取り回数は両者とも **1 回**（SG も WM も）。

### 結果（weight 再利用 SG vs tile WMMA、weight は両者とも 1 回読み）

| M | SG 時間 | SG GB/s | WM 時間 | WM GB/s | 勝者 |
|---:|---:|---:|---:|---:|:--|
| 2  | 237 µs | **198.8** | 261 µs | 180.6 | **SG**（1.10×） |
| 4  | 338 µs | 139.8 | 259 µs | **182.0** | **WM**（1.30×） |
| 8  | 471 µs | 100.2 | 269 µs | **175.6** | **WM**（1.75×） |
| 16 | 604 µs | 78.1  | 274 µs | **172.3** | **WM**（2.21×） |

- **WM の時間は M=1–16 でほぼ一定（~260–274 µs）**: weight tile 1 回読み、input tile（16 行）も 1 回読んで 16 feature で再利用（grid.y = ceil(M/16) = 1）。
- **SG は M に線形に劣化（237→604 µs）**: weight は 1 回読み（再利用）だが、**input の読み取りが M×**。
- **crossover は M=2→4**（理論値 M≈1.2 ではなく、計測値 M=2→4）。SG は M=2 だけ僅差（1.10×）で勝つ、M≥4 は WM 優位（M=16 で 2.21×）。

### SG（weight 再利用）が M 増で劣化する理由: input load が M×

weight 読み取りは両者とも 1 回（SG は `w` を 1 回読んで M 行で再利用）なので、
差は **weight ではなく input の読み取りパターン**:

- **SG**（行優先、1 feature/wavefront）: N 個の feature ごとに M 行の input を再読み →
  **input トラフィック = M×**（M=16 で 737MB の L2 traffic）。L2 に収まるが ~368µs の L2 仕事 → 律速。
- **WM**（tile、16 feature/block）: 16 行の input tile を **1 回**読んで wmma fragment で
  16 feature すべてで再利用 → **input トラフィック = 1×**（46MB、一定）。

**weight 再利用だけでは不十分。input load の再利用（tile が 16 feature で 1 回）が本質的な差**。
SG（行優先）は feature ごとに input を再読み（M×）し、WM（tile）は input を 16 feature で共有（1×）。
したがって M≥4 では WM が優位。

### 対 M>1 の本番比較

| 方式 | M>1 の GB/s |
|---|---|
| natural WMMA GEMM（現在の M>1 経路） | 144 |
| **preshuffle WMMA GEMM（tile）** | **180（1.25× 高速）** |
| weight 再利用 SG（行優先） | 198.8（M=2 のみ）/ 78–140（M≥4、劣化） |

M>1 では preshuffle WMMA が最良（180 GB/s、M に不依存）。

### dispatch 設計（計測値による）

```cpp
if (M == 1)  launch_native_gemv(...);       // 221 GB/s
else if (M == 2) launch_smallgemm_reuse(...); // 198.8 GB/s（僅差 1.10×、任意）
else          launch_preshuffled_wmma(...);  // M>=4、180 GB/s、M に不依存
```

`rows == 1` の二値判定ではなく M による切替。M=4 以上は preshuffle WMMA（一定 180 GB/s）。
M=2 は SG が僅差で勝つが、MTP は通常 M=4,8 を使うので実用上 M≥2 → WMMA で十分。

### 採用判断: 将来機能として記録、今は実装しない

- 代償: GEMV（行優先）+ preshuffle WMMA（tile）は **2 通りの weight layout = 2× copy（2.54GB）**を要する（load 時一次性、40GB arena に収まる）。
  ただし SG（weight 再利用）は行優先（natural layout）で動くので、**M=2 のみ SG を使うなら 2× copy は不要**（natural の 1 つで GEMV+SG）。
- **MTP は現在の設計範囲外**（M>1 の workload が無い）。仮定の将来機能を軸にアーキテクチャを再設計しない方針に従い、
  **今は preshuffle を実装せず、本検証結果を記録**。将来 MTP を導入する際に本節の数値（M≥4 で preshuffle WMMA 1.3–2.2× 高速、M 不依存）を根拠に直接採用可能。
- **教訓: weight 再利用（1 回読み）だけでは不十分。input load の再利用（tile が 16 feature で 1 回）が本質的な差**。SG（行優先）は M 増で input M× 読み → 劣化。WM（tile）は input 1×（一定）。

### 4bit weight とモデルサイズ: SG の欠点が相対的に巨大化（定量）

時間モデル（bf16 計測で裏付け済み、n=9216 k=2560 M=8、DRAM 256GB/s、L2 2TB/s）:

```
SG: T = (N·K·w_bytes)/DRAM + (N·M·K·2)/L2
WM: T = (N·K·w_bytes)/DRAM + (N·M·K·2/16)/L2
```

**SG の input トラフィックは WM の 16 倍**（SG は feature ごとに input を再読み、WM は 16 feature で 1 回共有）。この 16 倍は **N, K, M に依存しない**（定数）。

weight を軽くする（`w_bytes` 減少）と、**weight time が減り、input time（不変、SG は 16 倍）が相対的に巨大化**:

| weight | SG | WM | WM/SG |
|---|---|---|---|
| bf16 (2B) | 373 µs | 196 µs | **1.90×**（計測 1.75×） |
| 4bit (0.5B) | 235 µs | 58 µs | **4.05×** |

- **weight が軽いほど SG の「activation を feature ごとに何度も読む」欠点が相対的に巨大化** → WMMA の優位が広がる（bf16 1.9× → 4bit 4.05×）。
- **PSQ4/8（4bit/8bit）でこの効果が発動**。量子化モデルでは preshuffle WMMA の優位がさらに大きくなる。

#### 「より大きなモデル」について（正直な整理）

16 倍の input 比は **N, K に依存しない**（定数）。input/weight の時間比 = `(M·2·DRAM)/(w_bytes·L2)` も N, K に依存しない。
したがって「大きいモデル」単独では比は変わらない。大きいモデルで WMMA がより有利になるのは:

1. **4bit を使う**（メモリに収めるため）→ 上記の 4bit 効果が発動 → WMMA がより有利。
2. **input（M·K·2）が L2 に収まらなくなると**、SG の input 読みが **DRAM 読み**に（より遅い）→ SG の欠点がさらに巨大化。

つまり「より大きなモデルで WMMA がより有利」は **単独ではなく「4bit（+L2 超過）との組み合わせ」で成立**する。


---

## 7.42 BF16 の preshuffle をロード時に採用（2026-09-17）

§7.40 で却下した BF16 preshuffle（tile layout）を**採用した**。却下の根拠だった
「primary は chat（M=1 decode）」は **TG 最適化時点での派生であり、実際の対象ではない**。
実際の対象は **M≥2（MTP）/ prefill** で、preshuffle（tile）が有利（§7.41）。

### 実装: 量子化と同一 preshuffle API（namespace 統一）

BF16 baseline と PSQuant は**区別しない**。preshuffle は `quant_preshuffle.h` の
`preshuffle_native` / `inverse_preshuffle`（`CanonicalQuantView` → `NativeQuantHost`）
を共用し、**区別は `format_id` 内で行う**（BF16 専用 path でない）。

- `QuantFormatId::Bf16`（block_elements=16、code_bytes_per_block=32、meta なし）を追加。
- `preshuffle_native` / `inverse_preshuffle` / `validate` を `desc->block_elements` 化
  （32 ハードコード廃止）→ BF16（16）と PSQ（32）の両方に対応。
- BF16 ロード（`load_bf16_matrix_preshuffled`）は **PSQ と同一経路**:
  `CanonicalQuantView`（format_id=Bf16）構築 → `preshuffle_native` → H2D → GPU native のみ。
  canonical（natural）は GPU に上げない（二重常駐なし、§14 準拠）。
- linear weight（attn/MLP）を preshuffle。**embedding / LM head は natural 維持**
  （`tie_word_embeddings` で共有。embedding lookup は natural を読むため）。
- dispatch は `native_layout_id == Gfx1151`（tile）のとき tile WMMA、それ以外は natural
  （GEMV / natural WMMA）。

### E2E A/B（実機、interleaved、Qwen3.5-4B BF16）

| 指標 | natural (base) | preshuffle (cand) | 比 |
|---|---|---|---|
| PP128 (prefill) | 329 ms | 129.9 ms | **2.53× 改善** |
| PP2048 (prefill) | 4077 ms | 2828 ms | **1.44× 改善** |
| TG (decode, M=1) | 44.2 ms/tok | 53.8 ms/tok | **19% 劣化**（GEMV 221→tile WMMA 180） |

- **正しさ**: base と cand の greedy token が**同一**（`12,1234,6027,1234,11,1234,1234,7890`）。
  preshuffle は純 permutation なので結果不変。
- required 74/74 通過（2 件 fail は既存の gfx12xx 必須 skip、本変更と無関係）。

### 教訓

- **preshuffle は optional でなく、GPU native representation を成立させる変換**（§11）。
  BF16 も量子化と同一 API（`preshuffle_native`）で、`format_id` で区別する。
- **M=1 最適化（GEMV）は TG 時点の派生**。実際対象（M≥2 / prefill）では preshuffle が net win。
- 代償: decode(M=1) 19% 劣化 + linear weight の tile layout（embedding/LM head は natural 維持で 2× copy 不要）。


---

## 7.44 BF16 preshuffle GEMM (tile kernel) の repair と correctness path の native layout 対応（2026-09-17）

### tile kernel (preshuffle GEMM) の load 修正

- 旧 `load_bf16x16` は `short16_t` 一括 vector load を使っていたが、
  RDNA3 unsuffixed WMMA builtin が期待する fragment 並びと 256bit load の並びが一致しない。
  要素ごと読みに変更し、CPU 参照（double）で rows 1..256 × N ≤ 248320 × K ≤ 5120、
  BF16/F32 出力の全 shape が一致。
- 併せて現在の production 形状（rows=1 decode, rows=128 prefill slice, LM head）を
  E2E 実バッファ dumpy + in-process CPU 検証で 196 呼び出し全て PASS を確認。

### correctness mode (PHASESHIFT_QWEN35_KERNEL_MODE=correctness) が静かに壊れていた

- correctness は preshuffle 導入（§7.42, gfx1201 と同時対応）以前に作られた row-major 読み。
  gfx1151 では weight が常に preshuffle-native でロードされるため、
  correctness mode の BF16 linear は **native buffer を row-major に読むスクランブル GEMM** になり、
  E2E が「auto≠correctness」で見かけ上 broken に見えていた。
- 実際には auto 側（tile kernel）の出力が真で、gfx1201 実機（preshuffle 無し, raw weight）の
  auto/correctness 双方と E2E greedy トークンが一致（248046,198,248045）。
- `exec_linear_weight` に native layout（layout_id==1）読み分けを追加し、
  correctness mode を gfx1151 でも真の参照として復活させた。

### gfx1151 の reference 整合

- prompt = token-1×256, ctx=256 の greedy:
  - gfx1151 auto（tile + 新 GDN）= `248046,198,248045`
  - gfx1151 correctness（修正後）= `248046,198,248045`
  - gfx1201 auto / correctness = `248046,198,248045`
  の 3 Way 一致。既知の「auto vs correctness トークン不一致」は本件が原因で、
  optimized kernel 側の lossy 誤差ではなかった。


---

## 7.45 gfx1201 decode bf16 GEMV: gfx1151 の dedicated rows=1 カーネル移植（1 warp = 1 行、128bit streaming）（2026-09-18）

### 経緯

- LINEAR_BF16 は 21.5-21.9ms/tok で decode の ~88%。per-shape 有効 BW は
  431-454GB/s（nominal 640GB/s の 67-71%）。
- 旧 WMMA kernel の disassembly: 8B load × 16 行（5KB stride）、pipeline depth 1
  （wmma → 次イテレーション 2×8B load → 分岐）、in-flight ~32B/warp、VGPR 26。
  bench（同一 47MB weight の繰り返し読み）でも 431GB/s のみで、
  L3 帯域（§7.45 末尾の計測: ~1.3TB/s）から大きく離れている
  = in-flight byte / latency 不足がボトルネック。
- gfx1151 は rows==1 に dedicated GEMV（fd4ca68）を持ちほぼ全帯域を稼いでいた。
  WMMA fragment layout は gfx1151 と gfx1201 で異なるが、
  この GEMV カーネルは arch 依存 intrinsic を含まない
  （16B load + fma + shfl tree）ため gfx1201/gfx1200 にそのまま移植できた。

### 実装

- gfx1201/gfx1200（byte-identical）に `gemv_bf16_kernel` を追加:
  - 1 warp = 1 output row（256スレッド block = 8 行）。
  - lane あたり 16B（`short8_t`）連続ロード、warp = 512B/イテレーションで完全 coalesced。
  - shfl_down tree 集約（LDS なし、`__syncthreads` なし、barrier なし）。
  - gate: `rows==1 && (k&7)==0 && weight/input 16B aligned`。
  - rows==1 の全 linear で WMMA パスを置換（rows>1 prefill は不変）。
- カーネル資源: VGPR 19、spill 0、`global_load_b128` ×2/イテレーション（w+x）、
  `v_dual_fmac_f32`（RDNA4 dual-issue）。
  32768/(32×19)=54 → 32 wf/CU 上限で full occupancy。

### 実測

- per-shape（bench gemm, rows=1, mean us。同一 weight の繰り返し読みのため
  47MB 以下の shape は L3 64MB 再利用成分を含む）:

| shape (n, k) | 旧 WMMA | 新 GEMV | 変化 |
|---|---|---|---|
| mlp_gate_up (9216, 2560) | 46.26 | 26.83 | -42% |
| mlp_down (2560, 9216) | 116.96 | 28.48 | -76% |
| attn_q (4096, 2560) | 37.49 | 29.37 | -22% |
| attn_kv (1024, 2560) | 37.86 | 6.18 | -84% |
| attn_o (2560, 4096) | 57.51 | 16.05 | -72% |
| gdn_qkv (8192, 2560) | 45.00 | 38.09 | -15% |
| gdn_ba (32, 2560) | 7.30 | 4.92 | -33% |

- full model（tg forward, ctx=128, 64 tok）: **23,533 → 17,781 us/tok（-24.4%）**。
- greedy 64 tok: bit 一致（SUM=1337166、先頭 8 トークン同一）。
- bench gemm --check（production correctness path 対比）:
  全 shape max_rel ≤ 4.8e-3 ≤ 2e-2、mismatch ≤ 4 要素。
- ctest: 75/76（1 は既存の gfx1151 arch-gated skip）。

### 数値挙動

- 集約順序が WMMA の固定 tile 順序から lane 別 partial sum の shfl tree に変わった。
  GEMV 1 回あたり bf16 要素の ~0.1% が 1 ULP flip（correctness path 対比で確認）。
- モデルレベルの logit 差（16 step forward dump: mean ~0.04、max 0.29）は
  その 1-ULP flip が 32 層 × 15 step を経由した cascade 増幅で、
  以前調査した split-K 変更と同一クラス。
- 新旧バイナリとも run 間 bit 一致（diff=0）で決定論的。
  greedy 64 tok サンプルでは argmax flip 0 件。


---

## 7.46 gfx1201 prefill bf16 GEMM: 64-row wide tile カーネル（rows≥65）（2026-09-18）

### 経緯

- prefill (rows≥2) は 16×16 WMMA tile カーネル（1 block = 16 out × 16 rows）。
  block 数 = ceil(n/16) × ceil(rows/16) で、同時実行スロット
  （2048 block 相当）を超えると wave が分かれ、
  wave 1 のキャッシュ traffic が L3 の weight を追い出し、
  wave 2 以降は weight を DRAM から再ストリーミングする
  wave 量子化 pathology があった。
- 例: rows=128, n=9216, k=2560（weight 47.2MB）:
  4608 block = 2.25 wave → 617µs（intra-wave 共有の理想値の ~2×）。
- 対象は plain GEMM だけでなく gate_up / qkv / gdn_proj の 3 系 fused GEMM
  （共通 body）も同一構造だった。

### 実装

- `gemm_bf16_wmma_wide_kernel` を新規追加（gfx1201/gfx1200）。
  1 block = 16 out × 64 rows（row tile 4 個）。
  k-tile あたり a フラグメントを 1 回だけ読み、4 つの row tile で再利用
  （weight traffic が rows/16 → rows/64 に減る）。
- row tile の有効数 `active`（1-4、warp uniform）で switch 分岐し、
  各 case は分岐なしの展開済み k ループ。
  - k-tile 内の uniform guard 4 本（`rt < active` 系の unroll）は
    rows≤64 系で ~20% の issue 増を計測 → switch 方式に置き換え。
  - dynamic ループ（unroll 除去）は load/wmma が serial 化し
    さらに遅い（rows=16, n=9216: 49.6 → 63.9µs）→ 採用せず。
- fused 側も同様に `fused_gemm_bf16_wmma_wide_body` を追加
  （detail ヘッダ、3 系 launcher が共用）。
- launcher 分岐: rows ≥ 65 → wide、rows ≤ 64 → 従来 16-row kernel。
  rows≤64（MTP verify / batch decode / 短 prompt）は
  block 数減による並列性低下と 1-row-tile overhead を避けるため変更なし。
- split-K（out≤256, rows≥2）と GEMV（rows=1）は変更なし。

### 実測

- per-shape（bench gemm, mean us, L3-warm）:

| shape (n, k) | rows | 旧 | 新 | 変化 |
|---|---|---|---|---|
| mlp_gate_up (9216, 2560) | 128 | 617.3 | 196.2 | 3.15× |
| mlp_gate_up (9216, 2560) | 256 | 1419.6 | 507.2 | 2.80× |
| mlp_gate_up (9216, 2560) | 512 | 2524.3 | 1006.5 | 2.51× |
| mlp_gate_up (9216, 2560) | 1024 | 4427.5 | 2079.5 | 2.13× |
| mlp_down (2560, 9216) | 128 | 355.6 | 340.0 | 1.05× |
| mlp_down (2560, 9216) | 256 | 970.8 | 438.0 | 2.22× |
| attn_q (4096, 2560) | 128 | 148.2 | 118.9 | 1.25× |
| lm_head (248320, 2560) | 128 | 18051 | 9113 | 1.98× |

- rows ≤ 64: 旧 kernel のまま（数値・性能とも不変）。
- lm_head rows=128 の改善は weight 1.27GB（≫ L3）を
  row block 8 → 2 への traffic 削減（10.2GB → 2.5GB DRAM）が主因。
- full model prefill（bench pp, forward, mean us/prompt-token）:

| prompt | 旧 | 新 | 変化 |
|---|---|---|---|
| 128 | 645.1 | 411.2 | 1.57× |
| 512 | 725.4 | 371.4 | 1.95× |
| 2048 | 713.2 | 395.2 | 1.80× |

- decode 影響なし: tg forward 17,781 → 17,782 µs/tok（rows=1 GEMV 経路）。
- greedy 64 tok: bit 一致（SUM=1337166）。
- bench gemm --check（rows 2..256、全 shape、境界 rows=17/48/65 込み）:
  max_rel ≤ 7.6e-3 ≤ 2e-2。
- ctest: 75/76（1 は既存の gfx1151 arch-gated skip）。

### 残余

- M ≥ 1024 域は bf16 WMMA の compute 下限
  （~18-19 TFLOPS 相当）に近づく。さらに上げるには
  2D tile の拡大（multi-warp 64×64 等、FLOP/B 向上）が必要。
  未実施（別プロジェクト扱い）。


---

## 7.47 fused PSQ4 GEMM の wave 量子化: pathology は実在だが 27B production では未使用（2026-09-19）

### 経緯

- `docs/now.md`（2026-09-18 中断タスク）: 「PSQ4 prefill に BF16 同型の
  wave 量子化 pathology が無いか」の調査。中間 finding 1 で fused PSQ4
  カーネル（`gemm_fused_psq4_w4a8_wmma.gfx1201.hip`、`kPsq4TileRow = 16u`
  固定、MB 機構なし）に構造上の pathology と推定、Step 1 で直接計測する
  手順が書かれていた。

### Step 1 計測（standalone harness、27B gate_up 実 shape n=17408×2, k=5120）

- payload / projection = codes 44.6MB + scales 5.5MB ≈ 50.1MB、×2 = 100.2MB。

| rows | blocks | waves | p50 µs | µs/wave |
|---|---|---|---|---|
| 16 | 2176 | 1.06 | 238.9 | 225 |
| 64 | 8704 | 4.25 | 785.2 | 185 |
| 128 | 17408 | 8.50 | 1715.3 | 202 |
| 512 | 69632 | 34.00 | 6071.6 | 179 |
| 1024 | 139264 | 68.00 | 11874.8 | 175 |
| 2048 | 278528 | 136.00 | 23531.0 | 173 |

- **wave 当たり 173µs = 100.2MB/173µs ≈ 579 GB/s ≈ DRAM 読取ピーク
  （560 GB/s 実測）+ L2**。つまり全 wave が weight を DRAM から
  全量再ストリーミングしている（16-row tile かつ MB なし）。
- rows=128 で既に 8.5 wave（単一 pass 179µs 対 1715µs = 5×）。
- now.md の判定式 fused(1024)/(fused(128)×8) = 0.865 は「線形」に見えるが、
  両端とも既に re-stream 域にあるため pathology の有無を判定できない
  （式設計の前提「rows=128 ≒ 1 wave」が誤り）。

### production 到達性の確認（本調査の決定的 finding）

- `runtime/linear_fusion_selector.cpp` の fusion rule は 2 行のみ、
  どちらも **k=2560（4B の hidden）**:
  Qkv（rows 1..64）、GdnProj（rows 1..128）。
- 27B-PSQ の linear は全て k=5120 → rule 非一致 → `Unfused`。
  **27B の production では fused PSQ4 カーネルが 1 回も launch されない**
  （`optimized_dispatch.hip` の `try_launch_fusion_psq4` が
  selector 判定で NotApplicable になり plain GEMM に落ちる）。
- よって pp 2048（1165.8ms）の 27B-PSQ 線形レイヤは全て
  plain PSQ4（rows≥128 で MB=8 wide tile）であり、
  now.md「中間 findings 1」の「pp M=128 の 83ms 中の ~50ms が
  fused 由来 pathology」推定は 27B には当てはまらない。

### wide tile 追加（Step 2）の費用対効果: 無し

- wave 数 = out_tiles × ceil(rows/tile) は fused / plain で同型。
  rows=2048 で plain gate+up（2 launch）= 8.5+8.5 = 17 wave、
  fused wide 128-row tile = 2176×16 = 17 wave（同数）、
  fused wide 64-row tile = 34 wave（plain の 2 倍）。
- 即ち **fused wide tile は plain MB=8 を超えない**（launch 1 回分の
  節約のみ）。production rule も無いため 27B への追加価値は 0。

### 結論

- カーネル変更なしで締結。fused PSQ4 カーネルの pathology は
  「rule が存在した瞬間に爆発する地雷」であり（16-row 固定で
  rows=2048 が 136 wave）、将来的に 27B fusion rule を追加する場合は
  先に wide tile（128-row 相当）を実装する必要がある。
- 27B pp の残るボトルネックは plain PSQ4 の M≥1024 での
  compute 下限 + 12-18% super-linearity（now.md 関連 defer 1/2、
  別プロジェクト扱い）のみ。


---

## 7.49 PSQ4 prefill 2D LDS tiled GEMM 実験（負結果・production 化せず）（2026-09-19）

### 目的

§7.9 で prefill GEMM は WMMA 天井の 21%（44.6/210 TFLOPS）で
「A を out tile ごとに再読取し、総トラフィック ~1.75TB/s ≒ L3 帯域クラスで律速」
と切り分けた。本実験は、1 wave の算術密度を壊さずに
**A を N 方向で、W を M 方向で共有する 2D tiled kernel** を追加し、
A の global（L3）再読取を削減できるかを検証するもの。
MB（1 wave の M）は 8 のまま固定し、wave を 2D に並べる（MB=16 禁止）。

### 設計

- Phase 1（`2d_4w`）: BM=128 / BN=64 / BK=64、TM=1 / TN=4、128 threads。
  1 wave = M=128（MB=8 と同一）× N=16。A（activation codes）のみ
  fragment layout のまま LDS に stage（8KiB、16B cooperative copy、
  row tail は zero-fill）。W は従来通り global（packed q4 + bf16 scale、
  CB10 decode 不変）。
- Phase 2（`2d_8w`）: BM=256 / BN=64 / BK=64、TM=2 / TN=4、256 threads。
  A 16KiB + W decoded fragment（w0/w1/ws、5120B）+ activation scale
  （1KiB）を stage。W の global 読取と CB10 decode は wave_m=0 の
  4 wave が 1 回だけ行い、wave_m=1 も同じ decoded fragment を読む。
- 両カーネルとも arithmetic order は legacy と同一
  （ib 昇順、ib ごとに wmma(a0,w0)→wmma(a1,w1)→fma(acc,ws)、
  epilogue の shfl/乗算/BF16 変換も同一）。operand の格納場所
  （global→LDS）の変更のみ → **bit-identical が正しさの基準**。
- `nb % 2 != 0`（K が 64 非倍数）は legacy へ fallback。

### 正しさ

bench `--check`（legacy 出力 vs 2D 出力、別 buffer、bit 比較）:

- 7 Qwen shapes（n=17408/5120/6144/1024/48、k=5120/6144/17408）
  × rows {65,127,128,129,255,256,511,512,1024,2048} = 70 case
  × {4w, 8w}、BF16 / F32 / residual なし / residual あり 全て **bit-identical**。
- 小 N/K 境界（n∈{17,65}、k∈{64,5120,17408} 等）24 case も全て OK。
- 計 94/94 PASS。arithmetic order 維持の目標は達成しており、
  2D カーネルは「正しく、かつ遅い」。

検証の過程で **bench 側の既存 bug 2 件**を発見・修正
（legacy カーネルは byte-identical 維持、production は非該当）:

1. legacy full-tile path（`kGuarded=false`）は A の row edge をガードしない。
   最終 row block が `row_start..row_start+MB*16-1` まで読むため、
   rows 17..127（MB=2/4）で activation workspace 末尾越えの OOB read。
   production は row bucket 分の workspace があるため fault しないが、
   bench の rows 厳密割り当てでは memory fault
   （rows=65, n=5120, k=17408 で再現）。
   bench 側で workspace を 128 行単位で割り当てる回避を適用。
   legacy カーネル本体のガード追加は本タスク範囲外（decode/rows<=64 不変の
   制約）として残課題。
2. bench の weight codes 割当が `n*(kp/2)` で、preshuffle plane が必要とする
   `((n+15)/16)*16*(kp/2)` を n%16!=0 で割っていた（OOB read）。修正済み。

### resources（`extract_kernel_resources.py`、同一 binary）

| kernel | SGPR | VGPR | vgpr_spill | scratch | LDS | code size |
|---|---|---|---|---|---|---|
| `gemm_psq4_w4a8_wmma_kernel<8>`（legacy prefill 経路） | 44 | 256 | 3 | 16B | 0 | 34264 |
| `gemm_psq4_w4a8_wmma_2d_4w_kernel` | 36 | 166 | 0 | 0 | 8192 | 26240 |
| `gemm_psq4_w4a8_wmma_2d_8w_kernel` | 34 | 142 | 0 | 0 | 22528 | 25760 |

register 圧は legacy MB=8 より**良い**（spill 0 / scratch 0、VGPR 低減）。
敗因は register ではなく occupancy:

- legacy: LDS 0、256 VGPR → 32 wave/CU（上限）まで resident 可能。
- 4w: 8KiB LDS → 4 block/CU = **16 wave/CU（50%）**。
- 8w: 22.5KiB LDS → 1 block/CU = **8 wave/CU（25%）**。

### 計測

計測モード: bench `gemm --dtype psq4`、warmup 30 / samples 30、
**variant 交互 3 ラウンド・per-op event（launches=1）**。
（launches=20 の batch event 値も同符号だが、§8.4 の非決定論により
表値は per-op を採用。）

p50（3 ラウンド平均、µs）と legacy 比（>1 = 遅い）:

| shape (n, k) | rows | legacy | 4w | 4w/leg | 8w | 8w/leg |
|---|---|---|---|---|---|---|
| mlp_gate_up (17408,5120) | 128 | 168.2 | 204.7 | 1.22× | 360.6 | 2.14× |
| 同上 | 512 | 608.8 | 764.9 | 1.26× | 783.4 | 1.29× |
| 同上 | 1024 | 1382.6 | 1538.0 | 1.11× | 1623.7 | 1.17× |
| 同上 | 2048 | 3001.3 | 3181.3 | **1.06×** | 3319.4 | 1.11× |
| mlp_down (5120,17408) | 128 | 201.1 | 449.6 | 2.24× | 500.2 | 2.49× |
| 同上 | 512 | 622.9 | 1007.5 | 1.62× | 784.0 | 1.26× |
| 同上 | 1024 | 1295.7 | 1714.5 | 1.32× | 1542.0 | 1.19× |
| 同上 | 2048 | 2952.1 | 3398.6 | 1.15× | 3304.0 | 1.12× |
| attn_q (6144,5120) | 128 | 72.7 | 142.7 | 2.0× | 161.5 | 2.2× |
| 同上 | 2048 | 941.9 | 1040.9 | 1.11× | 1055.5 | 1.12× |
| attn_kv (1024,5120) | 128 | 47.4 | 132.6 | 2.80× | 108.0 | 2.28× |
| 同上 | 2048 | 157.8 | 184.7 | 1.17× | 186.6 | 1.18× |
| attn_o (5120,6144) | 128 | 83.4 | 164.8 | 1.98× | 188.8 | 2.26× |
| 同上 | 2048 | 884.8 | 1040.6 | 1.18× | 1063.3 | 1.20× |
| gdn_qkv (10240,5120) | 128 | 103.8 | 149.6 | 1.44× | 209.1 | 2.02× |
| 同上 | 2048 | 1722.6 | 1829.0 | **1.06×** | 1943.2 | 1.13× |
| gdn_ba (48,5120) | 2048 | 26.4 | 133.0 | 5.0× | 129.3 | 4.9× |

- **全 7 shape × 全 rows で 4w・8w の両方が legacy 比で劣化**。
  crossover は存在しない（4w の最良でも gate_up/gdn_qkv r2048 の +6%）。
- 劣化幅は N が小さいほど、rows が小さいほど大きい
  （gdn_ba n=48 で 5-7×: BN=64 固定のため grid が
  n/64 = 1 column だけになり parallelism が崩壊）。
- rocprofv3 kernel-only（gate_up r2048、同一条件の連続 launch）:
  legacy `<8>` ~2447µs、4w ~2665µs（+9%）で bench と同符号。
  grid は legacy 1088×16 block（32 threads）、4w 272×16 block（128 threads）。

### 敗因（なぜ利得が出なかったか）

期待利得は「A の global 再読取が TN=4 / TM×TN=8 分の 1 に減る」ことだが、

1. **legacy の A 再読取は既に安かった**。r2048 の A working set は
   10.5MB ≪ L3 64MB で L3 常駐、かつ同一 CU に同時 resident する
   32 wave（block）が同じ A fragment を短い ib 間隔で読むため、
   L1/L3 上に滞留したまま再利用される。2D 化で「L3 ヒットした global read」を
   「1 回の global read + LDS write + 4〜8 回の LDS read」に変えただけで、
   global トラフィック削減の対価として LDS round-trip を純増させた。
2. **occupancy が 1/2〜1/4 に落ちた**（LDS 制約、上の表）。
   in-flight memory request の数は resident wave 数に比例するため、
   W の DRAM streaming（4w では量不変）の latency hiding が弱まり、
   実効帯域が下がった。これは resources が良好（spill 0）にもかかわらず
   劣化する唯一の構造的説明であり、rocprof の kernel time 増とも整合する。
3. **single buffer の串行化**。BK=64 ごとに `__syncthreads()`×2 で
   staging と compute が完全に直列。次の BK の A global load は
   直前 compute 完了まで発行できない（software pipeline は本実験の
   採用基準「single buffer が明確に勝った後だけ」を満たさなかったため未実施）。
4. **BN=64 固定の small-N 崩壊**。n=1024/48 は 16/1 column だけ。
   legacy の 16-wide N tile は 64/3 column で grid が維持される。

### 決定

- 中止条件（4-wave A staging が負ける、8-wave true 2D も legacy に勝てない）
  に該当 → **production 化せず**。
- experimental kernel・launcher・bench variant（`psq4-2d-4/8`、`--check` A/B）
  は production source から削除し、実装は git history にのみ残す。
- bench の bug 修正 2 件（workspace 128 行割当、weight plane 割当）は有効なため維持。
- production selector は変更なし（rows=1 / split-K / rows<=64 / out<256 /
  nb 奇数 の既存経路は全て不変）。e2e PP / decode 回帰計測は不要
  （production のバイナリ内容が不変）。

### 却下した variant と理由

| variant | 結果 | 却下理由 |
|---|---|---|
| Phase 1: 4w（BM128/BN64/BK64、A-only） | 全 shape +6%〜+280% 遅 | occupancy 50% + 串行化、A 削減利得不発 |
| Phase 2: 8w（BM256/BN64/BK64、A+W staging） | 全 shape +4%〜+118% 遅 | occupancy 25% で上記がより悪化 |
| BN=128（TN=8） | 未実装 | 前提「BN=64 が勝つ」を満たさなかったため実施せず |
| software pipeline（LDS double buffer） | 未実施 | 前提「single buffer が明確に勝つ」を満たさなかったため実施せず |

### 残 bottleneck

- §7.9 の prefill GEMM 律速（WMMA 天井の 21%、L3 帯域クラス）は未解決。
  本実験は「A の tile 内共有」では効かないことを示した（L3 が既に
  再読取を吸収しているため）。次の一手は A の global/L3 トラフィック削減
  ではなく、§7.9 の aux 命令（shfl/cvt/FMA）発行や W streaming の
  in-flight 深さ維持の方向が候補。

### 訂正（2026-09-19、occupancy 実測と barrier asmgen）

上記の occupancy 推定（「legacy 32 / 4w 16 / 8w 8 wave/CU」）は
LDS 容量のみから推計したもので、**実測と一致しない**。
max-resident block probe（カーネル入口/出口の per-thread カウンタ、
peak = threads×blocks−1 の平たん値を確認）で gate_up r2048 を計測:

| kernel | resident blocks | 実測 waves/CU（÷64 CU） | 旧推定 |
|---|---|---|---|
| legacy `<8>`（256 VGPR、LDS 0） | 640 | **10** | 32（誤り） |
| `2d_4w`（166 VGPR、LDS 8KiB） | 288 | **18** | 16（誤り） |
| `2d_8w`（142 VGPR、LDS 22.5KiB） | 160 | **20** | 8（誤り） |

- legacy の律速は LDS ではなく **256 VGPR**（10 wave/CU）。
  2D カーネルは legacy より **1.8〜2 倍の resident wave** を持っていた。
  敗因 2 の「occupancy 低下」の主張は**撤回**する。
- driver の occupancy API（per-CU ソフトウェアモデル）は
  legacy 24 / 4w 8 / 8w 2 block/CU と返し、ハードウェア実測と乖離する
  （R9700 は WGP スケジューリング、`multiProcessorCount` = 32 = WGP 数）。
- device 実測値: `sharedMemPerMultiprocessor` = 64KB（per WGP、
  実効 32KB/CU）、`regsPerMultiprocessor` = 196608、
  `maxThreadsPerMultiProcessor` = 2048（per WGP）。
- 本ツールチェーン（LLVM 23 / gfx1201）で `__syncthreads()` の asmgen を確認:
  `s_wait_loadcnt 0x0`（**global load 全 drain**）+ `s_wait_dscnt 0x0`
  + `s_barrier_signal/wait` + `global_inv scope:SCOPE_SE`。
  2D カーネルは BK=64 ごとにこれを 2 回（k=5120 で **1 block 160 回**）
  発行しており、staging/compute の直列化と in-flight 深度の打ち切りの
  直接コストになる。
  local-only バリア
  （`s_waitcnt lgkmcnt(0); s_barrier; s_waitcnt lgkmcnt(0)`）は
  global drain と `global_inv` を伴わない。
- 敗因の修正版: ① バリアごとの global drain + SCOPE_SE invalidation
  による直列化、② A の global（L3 ヒット）read を
  「global read + LDS write + 複数回 LDS read」へ変えたこと、
  ③ BN=64 固定の small-N grid 崩壊。


---

## 7.50 PSQ4 prefill 2D v2（WM4/WN2・A direct + W-only LDS 共有）と production 化（2026-09-19）

### 目的

§7.49 の訂正（occupancy 実測: legacy 10 / 2D 18〜20 wave/CU、legacy の律速は
256 VGPR。`__syncthreads()` の asmgen: global load 全 drain + `global_inv`）と、
外部プロジェクト（Radiance mxfp4）の production A-tiled kernel 構造
（BM256×BN64、WM=4/WN=2、A は fragment-order で register 直読み、
W だけ LDS staging、local-only barrier）を PSQ4 に適用する。

分離実験として 3 段階を実装した:

| variant | A | W | buffer | barrier |
|---|---|---|---|---|
| v2a | global 直読み | global 直読み（WM=4 で重複） | - | なし |
| v2b | global 直読み | block で 1 回 global load + CB10 decode → LDS 共有 | 1 | local-only ×2/iter |
| v2c | global 直読み | 同上 | 2（staging を compute と重畳） | local-only ×1/iter |

v2a は「W direct は WM 方向の全 wave が decode を重複するため負ける」
（Radiance の記録と同じ傾向）ことの negative control。

### 設計

- block = BM256 × BN64 × BK64、256 スレッド（8 wave）。
- wave 配置: `wave_m = wave & 3`（M 64 行）× `wave_n = wave >> 2`（N 32 列）。
  1 wave = M64 × N32 = 8 個の 16×16 accumulator（§7.49 / legacy と同数）。
- A: §7.1c の fragment-order layout をそのまま global/L3 → register 直読み（LDS 不使用）。
- W: 8 wave が 1 つずつ (nt, ib) スロットを担当し、packed PSQ4 8B + bf16 scale 2B を
  1 回だけ load → CB10 decode → LDS へ store。全 wave が自分の wave_n 側
  (2 nt × 2 ib) を LDS から読む。LDS 4608B。
- barrier: `s_wait_dscnt 0` + `s_barrier_signal -1` + `s_barrier_wait -1` + `s_wait_dscnt 0`。
  `s_barrier` は gfx1201 で encodable でない（アセンブラが拒否）。
  この列は global drain も `global_inv` も含まない（asm 確認済み）。
- scale: BF16 per-32 を維持（fold しない）、FMA も維持。ws も LDS 経由。
- epilogue / guard / nb 奇数 fallback は legacy と同一の意味論。

### 正しさ

- `--check`（legacy 比 bit-exact、BF16/F32/residual 3 経路）:
  v2a / v2b / v2c 各 118/118（Qwen 7 shape × rows 10 + 小 N/K 境界 48、legacy-fallback 含む）。
- production 化後: test_gemm_psq4_w4a8_wmma に dispatch 対象 shape
  （rows=512 N=512 K=256 / rows=512 N=1024 K=2560）を追加し CPU reference 比で PASS
  （同テスト 1269 passed / 0 failed、ctest 新規 failure 0）。

### resources

| kernel | VGPR | spill | LDS |
|---|---|---|---|
| legacy `<8>` | 256 | 3 | 0 |
| v2a | 185 | 0 | 0 |
| v2b（= production 2D） | 148 | 0 | 4608 |
| v2c | 250 | 0 | 9216 |

### 計測（interleaved 3 ラウンド、per-op event、p50）

TFLOPS = 2·rows·n·k/t。Δ は legacy 比。

| shape | rows | legacy µs / TF | v2a Δ | v2b Δ / TF | v2c Δ |
|---|---|---|---|---|---|
| gate_up 17408×5120 | 2048 | 2903.5 / 125.7 | +1.4% | **−13.8% / 145.9** | −4.9% |
| mlp_down 5120×17408 | 2048 | 2593.6 / 140.8 | −3.3% | **−16.0% / 167.6** | −3.8% |
| attn_q 6144×5120 | 1024 | 402.4 / 160.1 | +8.9% | **−4.1% / 167.0** | +9.1% |
| attn_kv 1024×5120 | 2048 | 158.0 / 135.9 | +3.1% | **−7.5% / 147.0** | −1.0% |
| attn_o 5120×6144 | 2048 | 811.1 / 158.8 | +9.0% | **−3.3% / 164.3** | +11.4% |
| gdn_qkv 10240×5120 | 4096 | 3338.6 / 128.6 | +0.5% | **−16.9% / 154.8** | −5.6% |
| gdn_ba 48×5120 | 2048 | 26.5 / 38.0 | +535% | +491% | +476% |
| gate_up 17408×5120 | 128 | 168.8 / 135.2 | +231% | +197% | +194% |

集計（35 shape×rows）: v2a WIN 2 / TIE 3 / LOSS 30、**v2b WIN 22 / TIE 1 / LOSS 12**、
v2c WIN 7 / LOSS 28。
v2b の損失 12 件は全て構造要因（rows=128 の BM256 空転、n=48 の BN64 grid 崩壊）で、
rows≥512 かつ n≥1024 の全ケースで勝利（−2〜−17%）。

### 敗因・勝因

- v2a: W global load + CB10 decode を WM=4 の 4 wave が重複実行する
  （legacy 比 W global 2→4、decode 2→4）。A 重複 ×2 の改善では償えない。
  W を削らずに geometry だけ 2D にしても勝てないことの確認。
- v2b: W の global load と decode を block あたり 1 回に落とし、A も WN=2 で
  legacy の半分 → A / W global / W decode を全て legacy 未満にできる
  （legacy A=4/W=2/decode=2 に対し v2b A=2/W=1/decode=1）。
  これで符号が反転し、r2048 で −3〜−16%。
- v2c: staging register を compute 跨ぎで保持するため VGPR 148→250。
  staging 隠蔽（barrier 160→80 回）より occupancy 低下が上回った。
- 残差: v2b でも Radiance mxfp4 の 215〜220 TFLOPS には届かない
  （本実験の最大は r2048 mlp_down 167.6、r1024 attn_q 167.0）。
  残りは per-32 scale FMA（§7.23 で scale 全除去 1.12x）、W stock load の
  coalescing、epilogue、BK/BN 形状（BN=128/TN=4、BK=128）が候補。

### production 化

- dispatch（`launch_impl_gemm_psq4_w4a8_wmma`）:
  `k%64==0 && rows%256==0 && n%64==0 && rows>=512 && n>=512` のとき v2b
  （unguarded 版、VGPR 148）。それ以外は legacy
  （r128 の空転、n=48 の grid 崩壊、nb 奇数、k%64≠0 を含む）。
- micro A/B（dispatch 有効/無効の 2 バイナリ、interleaved 3 ラウンド、rows 256〜4096）:

  dispatch 対象（rows≥512, n≥512）: **WIN 29 / TIE 0 / LOSS 1**
  （唯一の負けは attn_kv r512 の +11.3%）。
  rows=256 と gdn_ba(n=48) は両バイナリとも ±0.6% = dispatch が意図通り legacy を
  選んでいることの確認。legacy 122〜158 TF → 2D 140〜166 TF。

- e2e PP（27B-PSQ、prompt 2048、mode forward、--runs 5 --warmup 2、4 ラウンド、
  per-invocation min。tok/s = 2048 / min ms として併記）:

  | bin | per-invocation min (ms) | tok/s |
  |---|---|---|
  | legacy | 1156.4 / 1165.5 / 1166.5 / 1169.8 | 1771 |
  | PSQ4 2D dispatch | 1079.7 / 1080.3 / 1082.8 / 1081.5 | 1897 |
  | PSQ4 + PSQ8 2D dispatch | 1056.5 / 1060.4 / 1060.9 / 1061.3 | 1938 |

  legacy → PSQ4 2D で **−6.6%（+7.1% tok/s）**、PSQ8 追加で **−1.5%（+1.7% tok/s）**、
  合計 **−9.3%（+9.7% tok/s）**。
  prompt 4096 は runtime の row bucket 契約（rows ≤ 2048）により両バイナリとも
  実行不可（本変更とは無関係の既存制約）。

  bench 側は pp / tg とも `gpu_tokens_per_sec` / `wall_tokens_per_sec` を
  summary と pp の run_detail 各行に併記するようにした（us/token だけでは
  直感的でないため）。

### 決定

- v2b を production prefill 経路として採用（dispatch 条件付き）。
  測定で勝利した領域のみを 2D に回し、非対象は legacy のままなので退行なし。
- v2a / v2c は git history にのみ残す（production source から削除、bench variant も撤去）。

### 残 bottleneck

- §7.9 の prefill GEMM 律速は未解決。本実験は
  「W の global load + decode の削減」が唯一効いた因子であることを示した。
- 次の候補: BN=128/TN=4 または BK=128（Radiance の M≥2048 設定）、
  W stock load の coalesced vector 化、epilogue fast path、
  `__builtin_amdgcn_sched_barrier(0)` による fragment lifetime 制御
  （gfx1201 で使用可能なことは確認済み）。


---

## 7.52 PSQ4 prefill 2D の BK / BN 拡張（BK=128・BN=128 と幾何ルール化）（2026-09-19）

### 目的

§7.50 の 2D カーネルは BM256×BN64×BK64 固定だった。外部プロジェクト
（Radiance mxfp4）の M 依存設定（M<2048 は BK=128、M>=2048 は TN=4/BN=128 相当）に
倣い、BK=128 / BN=128 を測って幾何ルール化する。

### 実装

- `gemm_psq4_w4a8_wmma_prefill_2d_kernel<kGuarded, kOutBlock, kKChunk>` に一般化。
  staging は「1 ラウンド = (kOutBlock/16) × (kKChunk/32) スロットを 8 wave で分担」
  （どの組合せでも割り切れる）。LDS は `[kNt][kIbc][32]`。
- dispatcher は幾何で config を選ぶ:
  `rows>=1024 && out_features>=2048 && k%128==0 && n%128==0` → BN128/BK128、
  else `k%128==0` → BN64/BK128、else `n%128==0` → BN128/BK64、else BN64/BK64。
  非対象（rows<512 / n<512 / k%64≠0 など）は従来どおり legacy。
- 診断用 override: `PHASESHIFT_PSQ_PREFILL_2D=bk64bn64|bk128|bn128|bk128bn128`
  （4 config はルール自体が全て instantiate するため実装コスト増なし）。

### resources

| config | VGPR | spill | LDS |
|---|---|---|---|
| BN64/BK64（従来 production） | 147 | 0 | 4 608 |
| BN64/BK128 | 166 | 0 | 9 216 |
| BN128/BK64 | 226 | 0 | 9 216 |
| BN128/BK128 | 240 | 0 | 18 432 |

### 計測（interleaved 3 ラウンド、per-op p50、6 shape × rows 512〜4096 = 24 ケース）

Δ は BN64/BK64 比。代表:

| shape | rows | base µs | BK128 | BN128 | BK128+BN128 |
|---|---|---|---|---|---|
| gate_up 17408×5120 | 2048 | 2523.0 | −3.4% | −10.9% | **−14.9%** |
| gate_up | 4096 | 5300.6 | −1.8% | −6.2% | −7.7% |
| mlp_down 5120×17408 | 2048 | 2236.6 | −3.3% | −2.5% | −5.5% |
| attn_q 6144×5120 | 2048 | 803.3 | −3.7% | −0.1% | −4.4% |
| attn_kv 1024×5120 | 512 | 69.3 | **−15.4%** | +51.9% | +34.5% |
| attn_kv | 1024 | 88.6 | −6.8% | +26.3% | +11.7% |
| gdn_qkv 10240×5120 | 4096 | 2803.8 | +0.4% | −7.1% | **−10.2%** |

集計: BK128 **WIN 22 / TIE 1 / LOSS 1（−1.96%）**、BN128 9/6/9（−3.05%）、
BK128+BN128 19/0/5（−6.23%）。

- BK=128 は barrier が 80→40 回/block になり **ほぼ全ケースで安定した小勝ち**。
- BN=128 は block あたり固定費（barrier/prologue/epilogue）が半減し
  **rows>=1024・n>=2048 では −4〜−15%** と大きいが、小 rows / 小 n では
  grid 崩壊（attn_kv n=1024 は r512 で +34%）と occupancy 低下で負ける。

### production ルール適用と e2e

- 幾何ルール適用時（BK128+BN128 を rows>=1024 && n>=2048 のみ、他は BK128）:
  e2e PP（prompt 2048、3 ラウンド交互、per-invocation min）
  **1056.5ms/1937 tok/s → 1017.2ms/2010 tok/s（−3.7% / +3.8% tok/s）**。
- §7.51 の lossy recurrence と併用: **979.2ms / 2085.6 tok/s**。
  legacy 比（1164.6ms / 1758.6 tok/s）で **+19.0% tok/s**。

### 正しさ

- 4 config すべて `test_gemm_psq4_w4a8_wmma`（disp2d ケース = production dispatcher
  経由、CPU reference 比）を PASS。128/128 を通すため `rows=1024 N=2048 K=256` の
  ケースを追加（4 config を網羅）。
- 全 ctest で新規 failure 0（既知の gfx1151 halo skip 2 件のみ）。

### 決定

- 幾何ルールを production 既定として採用（測定済みの勝ち領域のみ config を上げる）。
- 診断用 env override は残す（全 config instantiate 済みで追加コストなし）。

### 残

- PSQ8 側は BN64/BK64 のまま（同じ BK/BN 拡張が可能、未実施）。
- 次の候補: GEMM の W load coalescing（32bit offset / vector load）、epilogue fast path、
  `sched_barrier` による fragment lifetime 制御、HIP graph 有効化（gap 43ms）。

### PSQ8 への横展開（BK=128 / BN=128）は負け（2026-09-19）

PSQ4 と同じ BK/BN 拡張を PSQ8 の 2D カーネルにも適用して測ったが、
**全 config で劣化**した（interleaved 3 ラウンド、24 ケース、base = PSQ8 BN64/BK64）:

| config | 集計 | WIN / TIE / LOSS |
|---|---|---|
| BN64/BK128 | +5.75% | 0 / 9 / 15 |
| BN128/BK64 | +3.10% | 0 / 1 / 23 |
| BN128/BK128 | +0.55% | 0 / 17 / 7（r4096 は tie） |

- 原因: PSQ8 は W が 8bit 直読み（CB10 decode なし）で staging 自体が軽く、
  BK=128 による barrier 半減（80→40 回/block）の利得がない。BN=128 は
  VGPR 149→240 の occupancy 低下と small-N grid 崩壊
  （attn_kv n=1024 で +20〜80%）が上回る。
- 決定: **PSQ8 は BN64/BK64 を維持**（production 変更なし）。実験実装
  （template 化 + 幾何ルール + env override）は git history にのみ残す。
- 正しさ: 4 config すべて CPU reference PASS（128/128 用ケース追加）だったが、
  性能が負けるため production 化しない。
- 示唆: PSQ4 で BK/BN が効いたのは「CB10 decode を伴う staging が重く、
  barrier 回数が律速していた」ためで、decode のない PSQ8 では同じ手が効かない。


---

## 7.55 PSQ4 prefill GEMM の律速はどこか: W load と epilogue は律速ではない（2026-09-19）

PSQ4 prefill GEMM（PP 内 527ms）の W load coalescing / epilogue 改善の可否を、
ablation と実測ピーク参照で判定した。結論は **どちらも投資対象ではない**。

### 計測条件

- `phaseshift-bench gemm --dtype psq4 --rows 2048 --n 17408 --k 5120`（モデル最大形状の mlp_gate_up）
- 2D kernel `<false, 128, 128>`（§7.52 の幾何ルールで選択）、min of samples を採用
- 共有 GPU のため他プロセスが走ると 2.3× まで遅くなる（実測 4790us の例）。
  比較は同一セッションで back-to-back に取り直した baseline に対して行った。

### config 別（同一形状、両方 2 の倍数）

| config | BN/BK | min us |
|---|---|---|
| bk64bn64 | 64/64 | 2159.0 |
| bk128 | 64/128 | 2108.4 |
| bn128 | 128/64 | 2133.0 |
| bk128bn128 | 128/128 | **2065.8** |

§7.52 の幾何ルール（大形状は BN128/BK128）を単体 kernel でも確認。BN128/BK128 は
VGPR 240 / occupancy 6 waves/SIMD（BN64/BK64 は VGPR 166 / 9 waves）で、occupancy は低いが速い。

### ablation（baseline 2052us）

| 内容 | min us | 差分 |
|---|---|---|
| baseline | 2052 | - |
| W 経路を全削除（global load + fp4 decode + LDS store + LDS read を定数化） | 2064 | **±0%** |
| A load を削除（定数化） | 1848 | -10% |
| scale FMA を 1/8 に削減（同時に accumulator 依存鎖も切断） | 1842 | -10% |

W 経路を丸ごと消しても速度が変わらない = **W の load/decode/LDS は WMMA の裏に完全に隠蔽**されている。
epilogue は block あたり 1 回で、K ループ（40 round × 128 WMMA/wave）に対して命令数の 1-2% 程度しかない。

### 実測ピークとの比較

fp8 WMMA の実効ピークを自作 microbench（`wmma_f32_16x16x16_fp8_fp8_w32_gfx12` を
register 上で連続発行、512 block × 64 thread = 1024 wave、boost 時）で測ると **約 314 TFLOPS**
（`iters=2e6, blocks=512` で 402.7 TFLOP / 1.283s。block 内 2 wave を掛けて補正）。

- GEMM の持続 clock は **約 2.2GHz**（`rocm-smi --showclocks` 実測）、
  microbench は **約 3.36GHz** まで boost する。
- clock 補正後の実効ピーク = 314 × 2.2/3.36 ≈ **206 TFLOPS**。
- GEMM は 365.1 GFLOP / 2052us = **177.9 TFLOPS = 約 86%**。

つまり「52%（無補正）」ではない。持続 clock で見れば **約 86%** であり、
残る余地（A load ~10%、FMA/依存 ~10%）の合計と整合する。

### 結論

- **W load coalescing**: 改善余地なし（削除しても 0%）。
- **epilogue**: 命令数の 1-2% で対象外。
- 追加投資するなら A operand の prefetch / software pipelining（上限 ~10%、
  PP 換算で約 -50ms）だが、clock 補正後 86% のため期待値は小さい。
- PSQ4 prefill GEMM への追加最適化はここで打ち切り、他カテゴリ
  （gdn_prepare 75ms など）へ回すのが妥当。

### 参考: 実測ピーク microbench

`/tmp` の使い捨てコード（register 上で WMMA 連続発行、memory traffic なし）。
本リポジトリには残さない（production 到達不能な placeholder を作らない方針）。


---

## 7.68 PSQ4 GEMM の M 依存と batch-invariant 化（2026-09-20）

bf16 と同じ問題が PSQ4 にも存在する。

- `gemm_psq4_w4a8_wmma` は `rows==1` のとき block split-K カーネル
  （K を S 分割し、wave ごとに `nb/S` ブロックを逐次 FMA、最後に S 個を逐次加算）。
- `rows>=2` は非 split の `gemm_psq4_w4a8_wmma_kernel<mb>`（K を 1 パスで逐次）。
- K 集約順序が異なるため、M=1 decode と M=K+1 verify で数値がずれる。

`gemm_psq4_halo` / `gemm_psq8_*` / fused PSQ には `rows==1` 特殊経路が無く M 非依存。

`PHASESHIFT_PSQ4_BATCH_INVARIANT=1`（または `PHASESHIFT_VERIFY_EXACT=1`）で
M=1 の split-K fast path を無効化し、全 M を非 split カーネルへ統一する。

### 検証（Qwen3.5-4B-PSQ、PSQ4 + MTP、4 prompt × GEN64、optimized）

| mode | token_match | spec_off ms/token |
| --- | --- | --- |
| default (FAST) | 1–2/4 | 8.84 |
| VERIFY_EXACT + PSQ4_BATCH_INVARIANT | **4/4** | 8.86 |

split-K を外しても 4B-PSQ の decode は実質変わらない（8.84 → 8.86 ms/token）。
PSQ4 は非 split カーネルでも M=1 のコストが同等。

27B-PSQ（PSQ4/PSQ8 混在、18GB）向けの batch-invariant 条件はこれで揃う。

### 27B-PSQ での検証

`Qwen3.8-27B-PSQ`（18GB、PSQ4/PSQ8 混在、64 layer、num_v_heads=48、MTP あり）。

- 単発 target forward（M=1 vs M=K+1）は layer boundary bit-exact。
  greedy token 列でもランダム token 列（draft が外れた場合を模擬）でも
  `FIRST_MISMATCH=none`。
- KV/GDN replay も bit-exact（`test_qwen35_mtp_gate3_replay` PASS）。
- ただし **`PHASESHIFT_VERIFY_EXACT=1` が必須**。
  `PHASESHIFT_PSQ4_BATCH_INVARIANT=1` だけでは verify が Fast のまま
  （GDN recurrence が非 serial）で token が一致しない。
  逆に `PHASESHIFT_VERIFY_EXACT=1` は PSQ4 batch-invariant を自動で有効化する。
- 両条件 + fresh decoder で list prompt K=1/2/4/8 が token 一致。

**注意**: ベンチ harness で MtpKvState / SpecDecoder を多数 case で使い回すと
27B では token 不一致が出る。production と同じく case ごとに fresh に作ること
（`test_qwen35_mtp_gate4e` を修正済み）。4B では使い回しでも問題が出なかったため
27B の geometry 依存の副作用と見られる。

---

---

## 7.78 PSQ4 W4A8 の BF16 scale を `v_dot2_f32_bf16` へ置換する PoC（不成立）（2026-09-20）

対象: `gemm_psq4_w4a8_wmma.gfx1201.hip` の 2D prefill
（`gemm_psq4_w4a8_wmma_prefill_2d_kernel`、代表 config BK128/BN128）。
基準 commit `ab7ea9b8`、worktree `.worktrees/poc-psq4-bf16-dot2`、branch `poc/psq4-bf16-dot2`。

### 動機

現行の per-w32 BF16 scale 適用は、WMMA の F32 partial `c[8]` に対し

```text
for j=0..7: acc[j] = fma(c[j], float(scale_bf16), acc[j])
```

である。2 個の連続 w32 block を

```text
acc[j] = dot2_bf16( pack_bf16(c0[j], c1[j]), pack_bf16(s0, s1), acc[j] )
```

に置換すれば、同一 accumulator への 2 段の依存鎖を 1 段へ圧縮できる。
F32 partial の BF16 化は丸めず上位 16bit を取る truncate とし、
`__builtin_amdgcn_perm` 1 命令で 2 値分を pack する（§7.27 の
`__float2bfloat16` が約 5 命令に展開される問題を回避する狙い）。

### Gate 0: pack + dot2 の ISA 確認（PASS）

最小 probe（`poc/psq4_bf16_dot2/gate0_probe.hip`）を gfx1201 実機で実行:

| 項目 | 結果 |
| --- | --- |
| `pack_f32x2_to_bf16_trunc(1.0, -2.5)` | `0xC0203F80`（low16=bf16(1.0), high16=bf16(-2.5)） |
| selector | `perm(u1, u0, 0x07060302)` で上記の並びになることを確認 |
| `dot2([3,4],[5,6],0)` | `39`（厳密） |
| `dot2([1.0,-2.5],[5,6],0)` | `-10`（厳密） |
| `dot2(..., acc=1.5)` | `40.5` |
| ISA | `v_perm_b32` 1 命令 + `v_dot2_f32_bf16` 1 命令、cvt 系なし |

`__builtin_amdgcn_fdot2_f32_bf16(short2, short2, float, false)` は
`short2_t = short __attribute__((ext_vector_type(2)))` で使用できた。

### Gate 1: scale reducer 単体 microbench（FAIL）

`poc/psq4_bf16_dot2/gate1_microbench.hip`。8 accumulator を独立に持ち、
2 block 分の reduce を 1 iteration とする。partial をループ不変にすると
perm が hoist されるため、scale の bit を partial の一方に流用して
ループ依存を保つ（per-lane 定数 + uniform scale で VALU を強制）。
threads=262144、nblk=2048/8192/16384、samples=40、median。

| variant | ns/block | BASE 比 |
| --- | --- | --- |
| BASE（F32 `fma` ×16 / 2 block） | 0.00063 | 1.00x |
| **DOT2-TRUNC**（`v_perm_b32` ×8 + `v_dot2_f32_bf16` ×8 / 2 block） | 0.00086–0.00091 | **1.36–1.44x（+36〜44%）** |
| DOT2-RNE（`__float2bfloat16` 経由） | 0.00117–0.00120 | 1.84–1.91x（+84〜91%） |

nblk を変えても比は安定（TRUNC は常に 1.4x 前後）。VGPR は BASE 25 / TRUNC 25 / RNE 19、spill なし。

ループ本体の ISA:

| variant | 2 block あたりの scale 命令 |
| --- | --- |
| BASE | `v_dual_fmac_f32` ×8（= FMA 16 を 2/命令へ融合） |
| DOT2-TRUNC | `v_perm_b32` ×8 + `v_dot2_f32_bf16` ×8 = 16 |
| DOT2-RNE | 変換（`s_cvt`/`s_pack` 等）が支配、dot2 は 5 命令のみ |

**RDNA4 は `v_dual_fmac_f32` で FMA を 2 個/命令に束ねる。**
そのため「FMA 16 → DOT2 8」という前提が崩れ、DOT2 は 1 perm + 1 dot2 per j が
必要で命令数では 2 倍負ける。§7.27 の BF16 dot2 と同じ結論に、truncate-pack でも到達した。

### production kernel の実 ISA（BK128/BN128）

`clang++ --cuda-device-only -S` で出力した `gemm_psq4_w4a8_wmma_prefill_2d_kernel<false,128,128>`:

| 項目 | 値 |
| --- | --- |
| VGPR | 240 |
| SGPR | 26 |
| scratch / spill | 0 |
| WMMA (`v_wmma_f32_16x16x16_fp8_fp8`) | 128 |
| `v_dual_fmac_f32` | 440（= FMA 880） |
| `v_fmac_f32_e32` | 72 |

scale 部分は既にほぼ dual issue 済み（952 FMA 中 880 が dual）。
さらに 1 thread あたりの accumulator は `4 t × kNtPerWave(4) × 8 j = 128` 個あり、
scale 適用の依存鎖は元々律速ではない。よって「依存鎖を 1 段に圧縮する」取り分が無い。

### 判定

- **Gate 0 PASS / Gate 1 FAIL（明確な負け）**。plan の打ち切り条件に該当するため
  production kernel への実装（Gate 2）は行わない。
- 不成立の理由:
  1. `v_dual_fmac_f32` により現行 FMA は 2/命令。DOT2 は perm + dot2 の 2 命令/j。
  2. production の accumulator 数 128 で依存鎖は律速でない。
  3. scale 適用はカーネルの 11%（§7.23）しかなく、たとえ高速化しても上限 1.12x。
- PSQ4 の canonical format・quantizer・CB10・activation quantize は一切変更しない。

### 再現方法

```bash
cd .worktrees/poc-psq4-bf16-dot2/poc/psq4_bf16_dot2
hipcc -O3 --offload-arch=gfx1201 gate0_probe.hip -o gate0_probe && ./gate0_probe
hipcc -O3 --offload-arch=gfx1201 gate1_microbench.hip -o gate1_microbench -std=c++17
./gate1_microbench 8192 40
# ISA は --cuda-device-only -S で出力し v_dual_fmac_f32 / v_perm_b32 / v_dot2_f32_bf16 を数える
```


---

## 7.80 FP8 block128 / MXFP4 WMMA の初回計測（2026-09-20）

`phaseshift-bench gemm` に `--dtype fp8|mxfp4` を追加し、gfx1201 の
RowBlock WMMA 経路を PSQ8 と比較した。weight を再利用するため L3 に
収まる shape では命令律速、L3 を超える shape では DRAM 律速になる。

### L3 に収まる shape（n=5120, k=5120、fp8 26MB / mxfp4 13MB）

| dtype | rows=1 p50 | rows=16 p50 | rows=128 p50 | rows=512 p50 |
| --- | --- | --- | --- | --- |
| psq8 | 18.7us | 20.2us | 59.5us | 172.5us |
| fp8 | 17.1us | 22.6us | 56.1us | 182.4us |
| mxfp4 | 19.2us | 19.7us | 63.9us | 209.5us |

MXFP4 の E2M1→E4M3 展開を PSQ4 と同じ even/odd nibble perm へ置き換えた
（後述）ことで、命令律速の decode 側は FP8 と同等、PSQ8 とほぼ同等まで
詰めた。rows=512 では weight bytes が半分でもタイル再利用が効かず
PSQ8 / FP8 に劣る。

### L3 を超える shape（n=34816, k=5120、fp8 178MB / mxfp4 89MB）

| dtype | rows=1 min/p50 | 実効帯域 |
| --- | --- | --- |
| fp8 | 289.8 / 290.1us | 614 GB/s |
| mxfp4 | 152.4 / 153.0us | 585 GB/s |

両者とも DRAM 天井 640GB/s 付近に飽和し、**MXFP4 は FP8 の約 1.9x**
（weight bytes 半減どおり）。実モデルの decode は全 weight が L3 を
大きく超えるため、ここが代表性のある数値になる。

### Qwen3.5-4B E2E decode（context 128, 32 token, tg）

| model | gpu us/token | tok/s |
| --- | --- | --- |
| Qwen3.5-4B-PSQ (既存) | 19203.7 | 52.07 |
| MXFP4 (text-only) | 19997.0 | 50.01 |
| FP8 (text-only) | 22612.4 | 44.22 |

context 128 では重み traffic が固定 overhead（attention / GDN / norm /
dispatch）に埋もれるため差は小さいが、MXFP4 は FP8 より 13% 速い。
重み traffic が支配的になる長 context / 大 model では上の比に近づく。

### prefill 2D 経路

`rows % 256 == 0 && rows >= 512 && out % 64 == 0 && k % 64 == 0` のとき
256 (rows) x 64 (out) タイルを 8 wave で処理する 2D kernel を選ぶ。
各 wave が 16 output x 64 K の weight タイルを shared に staging し、
16 行で再利用する（MXFP4 は E4M3 展開も staging 時に 1 回だけ行う）。

Qwen3.5-4B prefill（1024 token, forward）:

| model | us/token | tok/s |
| --- | --- | --- |
| Qwen3.5-4B-PSQ (既存) | 105.5 | 9481 |
| FP8 | 108.6 | 9212 |
| MXFP4 | 110.4 | 9060 |

1024 token では attention / GDN の O(context^2) 項が支配的なため差は
小さい。行 tile の再利用で weight traffic は減るが、MXFP4 は展開 ALU が
残る分わずかに FP8 へ劣る。

### E2E 比較の注意（preset で role→format が違う）

上の 3 モデルは同じ model ではない。`psq` preset は
`token_embedding`（0.636B params）を **PSQ8** に、`fp8`/`mxfp4` preset は
**BF16** にする。lm_head は embedding への alias。よって

| model | payload | bpw |
| --- | --- | --- |
| psq (PSQ4+PSQ8+BF16) | 3.194GB | 5.91 |
| mxfp4 (MXFP4+BF16) | 3.416GB | 6.32 |
| fp8 (FP8+BF16) | 5.088GB | 9.41 |

となり、PSQ4 モデルは「4bit の割に」軽い。E2E の差は format の
カーネル効率だけでなくこの配分差を含む。

### kernel 単体の prefill GEMM（同一 shape, rows=1024 n=9216 k=2560）

| dtype | p50 us | weight bytes |
| --- | --- | --- |
| psq4 (K128N128 2D) | 285.9 | 13.3MB |
| psq8 (2D) | 298.5 | 25.1MB |
| fp8 (2D 64out) | 299.0 | 23.6MB |
| mxfp4 (2D 64out) | 313.0 | 12.5MB |

MXFP4 の展開を even/odd nibble perm へ置き換えたことで 316.4us → 313.0us。
残る差は PSQ4 の 2D が 128x128 tile であるのに対し、新規 FP8/MXFP4 2D は
64-output tile で weight 再利用が劣ることによる。128x128 tile 化が次の
改善対象。

### E2M1→E4M3 展開の実装

`{0,0.5,1,1.5,2,3,4,6}` → `{00,30,38,3C,40,44,48,4C}` の変換を、
配列 lookup（局所メモリに落ちる）→ `v_perm_b32` + shift/or → PSQ4 と
同じ even/odd nibble perm、と段階的に置き換えた。n=5120,k=5120,rows=1 で
42.3us → 22.8us → 19.2us（計 2.2x）。

当初の `v_perm_b32` 実装は、canonical（OCP）の even/odd packing に合わせて
nibble を **byte へ spread** してから selector に使っていた。selector は
byte 単位でしか読まれないため、この spread が必要だった。

PSQ4 の cb10 展開は同じ `v_perm_b32` を使いながら spread を持たない。
理由は native layout にあり、PSQ4 は 32-K block の 16 byte を
**byte b = code[k=b] (低 nibble), code[k=16+b] (高 nibble)** として格納する。
これにより

- 低 nibble は 1 byte につき 1 個 → `q_even = q4` がそのまま selector になる
- 高 nibble は `q_odd = q4 >> 4` で bit0-2 / bit3 がそのまま使える
  （隣 byte の混入は `0x07070707` / `0x08080808` mask で落ちる）

MXFP4 の native layout をこの形へ変更し（`preshuffle_mxfp4_native` が
canonical から repack する）、展開を

1. `q_even & 0x07070707` / `q_odd & 0x07070707` を selector に、
   `{mags0, mags1}`（magnitude 0..3 / 4..7）を `v_perm_b32` で gather
2. `{negs0, negs1}` も同様に gather
3. `(q & 0x08080808) >> 1` を selector の bit2 に足し、正負の表を選ぶ

とした（`mxfp4_low_high`）。`w0 = {lo[0], hi[0]}` が k=0..7、
`w1 = {lo[1], hi[1]}` が k=16..23 に対応し、PSQ4 と同一の組み立てになる。
canonical は OCP のまま保存し、native への repack は load 時に一度だけ行う。

#### 落とし穴（表の typo、実モデルで発覚）

負の magnitude 表 `negs1` を `0xCCC8C4C0` とすべきところ `0xCCC8C480` と
書いていた。最下位 byte（magnitude index 4 = 2.0 の負値）が `0x80`(-0.0)
になり、その code を持つ weight だけが -0.0 として寄与を失った。
random 重みのテストは誤差が閾値内に収まり通ってしまうため、**全 16-bit
入力を scalar reference と総当たり照合**して 0 mismatch を確認した。
展開単体は通常の GEMM テストより広い入力空間を持つため、この種の表の
typo は総当たり probe でしか確実に検出できない。

### 正しさ

`test_gemm_fp8_block128_optimized`（259 cases）と
`test_gemm_mxfp4_optimized`（322 cases）が CPU canonical codec から作った
厳密な host reference と一致。MXFP4 は worst relative error 4.2e-4
（FP32 累加順序差）、FP8 は O(1e-5)。`test_gemm_block_scaled_dispatch`
（74 cases）は correctness dispatch と optimized kernel を同一
activation workspace で走らせ、両者を host reference と照合する。
`mxfp4_low_high` 単体は全 65536 通りの 16-bit 入力で scalar reference と
総当たり照合し 0 mismatch を確認した。Qwen3.5-4B MXFP4 の E2E greedy
生成も会話的な文章を出力する。

### 再現方法

```bash
./build/phaseshift-bench gemm --dtype mxfp4 --rows 1,16,128,512 --n 5120 --k 5120 \
    --warmup 20 --samples 20 --launches 20
./build/phaseshift-bench gemm --dtype fp8 --rows 1,16 --n 34816 --k 5120 \
    --warmup 10 --samples 15 --launches 10
./build/tests/test_gemm_fp8_block128_optimized
./build/tests/test_gemm_mxfp4_optimized
```

---


---

## 7.81 PSQ4 decode GEMV: codebook 簡約 + rows=1 専用 kernel + unroll 再調整（2026-09-20）

27B-PSQ の decode で最大の費目（21.07 ms/token、帯域 88.4%）である
`gemm_psq4_w4a8_wmma_kernel<1>` を、RoPE / activation_quantize / PSQ8 と同じ
手順（bit-exact を維持したまま 1 Phase ずつ A/B）で最適化した。

production shape は K=5120→N={17408,12288,10240,6144,1024}、
K=6144→N=5120、K=17408→N=5120。weight bytes は
`N * K_padded * 0.5625`（code 0.5 B/weight + BF16 scale 0.0625 B/weight）。

### Phase 1: codebook 展開の代数簡約

cb10 の negative table は positive table に FP8 sign bit を OR しただけ:

```
negs0 == mags0 | 0x80808080   (mags0=0x44403800, negs0=0xC4C0B880)
negs1 == mags1 | 0x80808080   (mags1=0x52504C48, negs1=0xD2D0CCC8)
```

positive/negative の 2 回 gather + sign-select perm を、
magnitude gather 1 回 + sign bit OR に置換した（`psq4_cb10_expand`）。

- `v_perm_b32` 120 → 40（helper 1 call あたり 6 → 2、packed b64 あたり 12 → 4）
- 総命令 1124 → 1018、negative table 定数が ISA から消滅
- VGPR 87 → 81（MB=1）。**MB=8 の private 16 B / spill 3 が消えた**
- 単体テスト: 全 16 code × 8 位置、各 byte 0x00..0xFF 全探索、262144 random
  を独立 host reference と照合し 0 mismatch

性能は code 律速ではなく中立（−0.1〜−0.75%、N=1024 のみ −4.8%）。

### Phase 2: rows=1 専用 decode1 kernel

`gemv_psq4_w4a8_decode1_kernel<output_type, unroll>` を追加し、dispatch を
`_auto` に切り替えた。generic MB=1/2/4/8 と prefill 2D は fallback として残す。

- row 0 の activation のみロード、WMMA は convergent な制御フローのまま
- `acc[8]` → `acc0 = fma(c[0], ws, acc0)`（K 分割・reassociation なし）
- epilogue の `rs[8]` shfl、row guard、runtime dtype branch を削除（dtype は template）
- full 16-wide tile 前提（launcher が rows==1 / out%16==0 / k%32==0 を保証）

ISA（generic MB=1 → decode1 U=4）: VGPR 81 → 67、SGPR 34 → 23、
分岐 41 → 9、`ds_bpermute_b32` 8 → 0、kernarg 13 → 7 引数、spill 0。

構造特化単体の性能は中立（U=4 では generic と同等）。真の効果は
register 余裕による **unroll の上限解放**。

### Phase 3: unroll 再探索

`nb = K/32` は 160/192/544 と十分大きい。U=2/4/8/16 を production 7 shape で比較:

| K/N | U=2 | U=4 | U=8 | U=16 |
| --- | ---: | ---: | ---: | ---: |
| 5120/17408 | 89.72 | 83.92 | **83.99** | 83.93 |
| 17408/5120 | 91.38 | 83.68 | **82.68** | 83.86 |
| 5120/12288 | 64.58 | 60.60 | **59.83** | 61.18 |
| 5120/10240 | 54.55 | 51.19 | **50.08** | 51.61 |
| 6144/5120 | 39.12 | 35.54 | **32.04** | 33.29 |
| 5120/6144 | 35.13 | 32.11 | **31.63** | 33.07 |
| 5120/1024 | 16.89 | 11.62 | 8.83 | **7.90** |

U=8 が既定。ISA は U=8 で VGPR 108 / SGPR 23 / spill 0、U=16 も同じ 108。

U=8 のループは 8 iteration 分の weight/activation load（`global_load_b64`
×24）を先頭で一括発行し、`s_wait_loadcnt` の待ちを
`v_perm_b32` / bit ops / WMMA で埋める構造になっている。手書き inline asm は
不要だった（compiler schedule で十分）。

register file は device query で per-WGP 196608（= per-SIMD 24576 dwords、
thread 上限 8 wave32/SIMD）。U=8 は VGPR 108 で 7 wave/SIMD 程度に落ちるが、
それでも高速化する。**PSQ4 decode は occupancy ではなく K-chain の
メモリレイテンシ律速**である。

### Phase 4: shape 別 unroll selector

K=5120 で N を振ると U=8 / U=16 の crossover は N≈2048-3072:

| N | U=8 | U=16 |
| --- | ---: | ---: |
| 256 | 8.74 | **7.75** |
| 512 | 8.79 | **7.79** |
| 1024 | 8.81 | **7.85** |
| 2048 | 8.96 | **8.34** |
| 3072 | **13.58** | 13.71 |
| 4096 | **22.19** | 22.89 |
| 5120 | **26.73** | 27.95 |

launcher 側で `out_features <= 2048 → U=16`、それ以外 `U=8`。hot path に
runtime 分岐はない。

### Phase 5: 8-output/wave 実験 → 棄却

N=1024 は 64 wave しかなく帯域 374 GB/s（58%）なので、1 wave = 8 useful
output（lane 8..15 は lane 0..7 と同一アドレスを読んで結果を捨てる、
weight bytes 総量は不変）で 128 wave にする実験を行った。bit-exact は維持
できたが、**全 shape で悪化**:

| N | full U=16 | half U=16 |
| --- | ---: | ---: |
| 256 | 7.75 | 8.69 (+12%) |
| 512 | 7.79 | 8.65 (+11%) |
| 1024 | 7.85 | 8.71 (+11%) |
| 2048 | 8.34 | 9.57 (+15%) |

N=256〜2048 で時間がほぼ flat（8〜9 us、最小 work の N=256 と同等）である
事実からも、small-N の律速は wave 数ではなく K-chain のレイテンシである。
追加された WMMA / activation load が利得を上回るため即撤回した。

### 最終性能（warmup 100 / samples 500 / launches 20、p50、DRAM 条件）

| K | N | generic | **最終** | delta | code+scale GB/s | 天井比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5120 | 17408 | 84.04 | **83.97** | -0.1% | 597 | 93.9% |
| 17408 | 5120 | 84.06 | **82.69** | -1.6% | 606 | 95.3% |
| 5120 | 12288 | 60.70 | **59.82** | -1.4% | 592 | 93.1% |
| 5120 | 10240 | 51.31 | **50.07** | -2.4% | 589 | 92.6% |
| 6144 | 5120 | 35.85 | **32.04** | -10.6% | 552 | 86.8% |
| 5120 | 6144 | 32.27 | **31.63** | -2.0% | 560 | 88.1% |
| 5120 | 1024 | 12.05 | **7.88** | -34.6% | 374 | 58.8% |

per-token 換算（launch 数 128/48/16/32/48/32/32）で
**PSQ4 20.54 → 20.08 ms/token（-0.47 ms、-2.3%）**。

### E2E（27B-PSQ、context 2048 / tokens 128 / warmup 2）

| 段階 | ms/token | t/s |
| --- | ---: | ---: |
| 変更前 | 39.23 / 39.25 / 39.30 | 25.49 / 25.48 / 25.44 |
| Phase 2 (U=4) | 38.79 / 38.85 | 25.78 / 25.74 |
| Phase 3 (U=8) | 38.24 / 38.25 | 26.15 / 26.15 |
| Phase 4（最終） | 38.22 / 38.23 / 38.27 | 26.17 / 26.16 / 26.13 |

`GREEDY_TOKEN_SUM=2649738`、fallback 0、pp2048 2128 t/s、
required acceptance 全通過。decode は **-1.0 ms/token（-2.6%）**。

### 教訓

PSQ4 decode は **ALU でも occupancy でもなく DRAM レイテンシ律速**だった。
codebook の命令削減（-9.4%）と rows=1 特化（VGPR -17）はどちらも単体では
性能に効かず、後者が可能にした **unroll 拡大**だけが効いた。
N=1024 は K-chain レイテンシの下限（約 7.8 us）に張り付いており、
wave を増やす方向（Phase 5）では解けない。

### 再現方法

```bash
./build/phaseshift-bench gemm --dtype psq4 --rows 1 --k 5120 --n 17408 \
    --warmup 100 --samples 500 --launches 20 --psq4-decode1
./build/phaseshift-bench gemm --dtype psq4 --rows 1 --k 5120 --n 1024 \
    --warmup 100 --samples 500 --launches 20 --psq4-decode1 --psq4-unroll 16
./build/tests/test_psq4_codebook
./build/tests/test_gemm_psq4_decode1
```

---


---

## 7.85 PSQ4 prefill_2d: 既存 4 variant の sweep と Phase 1 冗長除去（2026-09-21）

§7.82 で PSQ4 prefill_2d `<false,128,128>` が pp2048 の 50.3%、WMMA peak の 73% と
判明した。残り 27% に着手する前に、既存 4 variant を production shape で総当たりし、
低リスクの冗長 load/store 除去（Phase 1）を試した。

### 測定環境とノイズ床

- ホストに R9700 が 4 枚ある。bench は `--device` で固定し、他セッションとの競合を避けて
  device 1 を使った。
- A/B は interleaved（base と candidate を交互に実行）、同一 shape で 9 reps。
- **同一ソースを再ビルドした base↔base の A/B は mean +0.03%**（5 shape、各 5 reps）。
  run 間の p50 ばらつきも 0.3% 未満。したがって以降の ±1% オーダーの差は有意として扱う。

### occupancy を実測で確定させる

`hipOccupancyMaxActiveBlocksPerMultiprocessor` はこの GPU では実機と一致しなかった
（LDS を 64 KB/MPU と見積もり、18432 B で 3 blocks と返す）。spin kernel による
co-residency 実測（block が `go` フラグを待つ間に同時 resident 数を数える）で確定させた。

| 資源 | 実測上限（MPU あたり） |
| --- | ---: |
| thread | 8 blocks（2048 threads、block 256 threads のとき） |
| LDS | 128 KB |
| VGPR | 196608 dwords |

`sharedMemPerMultiprocessor` は 65536 を返すが、実際に同居できるのはその 2 倍の
128 KB である（`lds=18432` で 7 blocks、`lds=36864` で 3 blocks）。

実カーネルの同時 resident blocks（256 threads/block）:

| variant | LDS | VGPR | blocks/MPU | waves/MPU |
| --- | ---: | ---: | ---: | ---: |
| `<false,128,128>` | 18432 | 240 | 3 | 24 |
| `<false,64,128>` | 9216 | 166 | 4 | 32 |
| `<false,128,64>` | 9216 | 226 | 3 | 24 |
| `<false,64,64>` | 4608 | 147 | 5 | 40 |

**`<128,128>` は既に 3 blocks/MPU 同居しており、「VGPR 240 なので 1 block/CU」ではない。**
LDS が 128 KB/MPU あるため、**staging を double buffer しても occupancy は落ちない**
（`<128,128>` で 36864 B → 3 blocks、`<64,128>` で 18432 B → 4 blocks のまま）。
これは §7.82 で「double buffer は LDS を 2 倍にするので危険」と考えていた前提を覆す。

### Phase 0: 既存 4 variant の production shape sweep

`--psq4-prefill2d 0..3` で強制。rows ∈ {512, 1024, 2048}、production K/N 8 種、
数値は p50（us）。太字が最速。

| rows | K | N | ob64/kc64 | ob64/kc128 | ob128/kc64 | ob128/kc128 | 最速 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 5120 | 1024 | 63.3 | **53.1** | 92.1 | 81.7 | ob64/kc128 |
| 512 | 5120 | 5120 | 174.3 | **167.6** | 192.7 | 186.6 | ob64/kc128 |
| 512 | 5120 | 6144 | 204.1 | 199.8 | 199.7 | **194.9** | ob128/kc128 |
| 512 | 5120 | 10240 | 339.2 | 328.7 | 337.4 | **322.7** | ob128/kc128 |
| 512 | 5120 | 12288 | 404.9 | 394.2 | 401.8 | **388.8** | ob128/kc128 |
| 512 | 5120 | 17408 | 574.4 | **560.9** | 584.8 | 572.1 | ob64/kc128 |
| 512 | 6144 | 5120 | 210.6 | **201.9** | 231.0 | 222.7 | ob64/kc128 |
| 512 | 17408 | 5120 | 599.7 | **571.4** | 648.0 | 627.8 | ob64/kc128 |
| 1024 | 5120 | 1024 | 77.4 | **74.5** | 96.1 | 86.7 | ob64/kc128 |
| 1024 | 5120 | 5120 | 334.9 | 328.1 | 333.6 | **321.5** | ob128/kc128 |
| 1024 | 5120 | 6144 | 399.6 | 392.5 | 396.9 | **385.3** | ob128/kc128 |
| 1024 | 5120 | 10240 | 663.8 | 655.2 | 657.4 | **635.3** | ob128/kc128 |
| 1024 | 5120 | 12288 | 795.5 | 781.3 | 786.9 | **763.0** | ob128/kc128 |
| 1024 | 5120 | 17408 | 1412.1 | 1413.5 | 1353.6 | **1333.3** | ob128/kc128 |
| 1024 | 6144 | 5120 | 402.0 | 391.4 | 399.4 | **386.0** | ob128/kc128 |
| 1024 | 17408 | 5120 | 1411.6 | 1390.2 | 1351.9 | **1317.2** | ob128/kc128 |
| 2048 | 5120 | 1024 | 138.4 | 136.8 | 140.8 | **134.5** | ob128/kc128 |
| 2048 | 5120 | 5120 | 661.0 | 652.6 | 653.1 | **635.3** | ob128/kc128 |
| 2048 | 5120 | 6144 | 792.3 | 785.1 | 784.5 | **761.4** | ob128/kc128 |
| 2048 | 5120 | 10240 | 1642.5 | 1627.4 | 1585.3 | **1561.8** | ob128/kc128 |
| 2048 | 5120 | 12288 | 1925.4 | 1912.1 | 1886.1 | **1837.6** | ob128/kc128 |
| 2048 | 5120 | 17408 | 2750.4 | 2725.3 | 2617.6 | **2566.5** | ob128/kc128 |
| 2048 | 6144 | 5120 | 788.4 | 781.4 | 782.5 | **759.2** | ob128/kc128 |
| 2048 | 17408 | 5120 | 2668.5 | 2658.8 | 2580.8 | **2523.4** | ob128/kc128 |

幾何平均（baseline = `ob128/kc128` 比）と 24 shape の p50 単純和:

| variant | geomean ratio | 24 shape 合計 |
| --- | ---: | ---: |
| ob64/kc64 | 1.0126 | 19.434 ms |
| ob64/kc128 | 0.9864 | 19.184 ms |
| ob128/kc64 | 1.0370 | 19.094 ms |
| ob128/kc128 | 1.0000 | **18.605 ms** |

shape 別の勝者:

- rows=2048（production pp2048）: **`ob128/kc128` が 8/8 で最速**。
- rows=1024: `ob128/kc128` が 7/8（N=1024 のみ `ob64/kc128`）。
- rows=512: `ob64/kc128` が 6/8。

**DRAM-cold でも順位は変わらない。** `--weight-copies 8`（footprint 118 MB 以上）で
rows ∈ {512, 2048} × 7 shape を回した結果、geomean ratio は ob64/kc64 1.0310 /
ob64/kc128 0.9893 / ob128/kc64 1.0276 / ob128/kc128 1.0000 で、rows=2048 は
`ob128/kc128` が 7/7 最速。bench の weight-copy 巡回量に結論は依存しない。

### Phase 1: 冗長除去はすべて棄却、dtype 特殊化のみ採用

| # | 内容 | mean delta | 判定 |
| --- | --- | ---: | --- |
| 1-A | weight scale staging を half-wave（`k_group == 0`）のみに | **+6.29%** | 棄却 |
| 1-C | epilogue の activation scale load を half-wave のみに | **+3.13%** | 棄却 |
| 1-A+1-C | 上記 2 つ同時 | +4.99% | 棄却 |
| 1-B | weight scale を LDS に BF16 のまま保持 | **+3.84%** | 棄却 |
| 1-D | output dtype を compile-time 特殊化 | **-0.82%** | 採用 |

1-A〜1-C は「重複 load/store を消す」変更だが、いずれも悪化した。
`if (k_group == 0u)` による divergence が staging / epilogue の schedule を崩すためで、
重複 load は実害が無い（lane 0 と 16 は同一 address を読むので memory 側で coalesce される）。
1-B も `__bfloat162float` を hot loop 側へ移すことで VGPR が 240 → 243 に増え、
schedule が崩れて悪化した。**`<128,128>` の main loop は静的 773 命令で、
これ以上削る余地は命令数ではなく schedule 側にある。**

1-D は epilogue の runtime branch を消す。VGPR は不変（240）、SGPR が 28 → 22。
5 shape × 9 reps で全 shape が改善（-0.31% 〜 -1.27%）。F32 経路は
`optimized_dispatch.hip` から到達可能なので、特殊化は placeholder ではない。

### Phase 2: `<64,128>` は production の主力にならない

`<64,128>` は VGPR 166 / LDS 9216 で 4 blocks/MPU と occupancy は高いが、
rows=2048 では `<128,128>` に 3.8% 劣る（24 shape 合計で +3.1%）。
**occupancy は律速ではない。** `<128,128>` の方が WMMA 1 命令あたりの
overhead 命令が少ない（main loop 773 命令 / 128 WMMA = 6.04 に対し、
`<64,128>` は 433 命令 / 64 WMMA = 6.77）。

現在の production selector（`rows >= 1024 && N >= 2048 && k%128==0 && N%128==0` で
`<128,128>`、それ以外は `k%128==0` なら `<64,128>`）は sweep 結果とほぼ一致しており、
**selector の変更は不要**。唯一 N=1024 × rows=2048 で `<128,128>` が 1.8% 勝つが、
production の N=1024 は `<64,64>` が選ばれる（§7.82）。

### ISA: 残り 27% の所在

`<128,128>` の main loop（静的 773 命令、WMMA 128、barrier 2）を phase 別に見ると:

| phase | 内容 |
| --- | --- |
| staging | `global_load_b64`×4 + `global_load_u16`×4 を発行 → `s_wait_loadcnt 0x7..0x0` で逐次待ち → `psq4_cb10_expand` → `ds_store_2addr`×6 |
| barrier | `s_wait_dscnt` / `s_barrier_signal` / `s_barrier_wait` / `s_wait_dscnt`（1 round に 2 回） |
| compute | `global_load_b64`×24（activation）→ `ds_load_2addr`×24 → WMMA 128（`s_wait_loadcnt` 27 + `s_wait_dscnt` 21 + `s_delay_alu` 103 を挟む） |

**staging の 8 global load を発行してから最初の `ds_store` まで約 25 命令しかない。**
DRAM load のレイテンシ（数百 cycle）を埋めるには足りず、各 round の先頭で
load レイテンシが露出している。1 round・1 wave の実測は約 3676 cycle で、
WMMA 128 本の下限（peak 240 TFLOP/s 換算で約 2624 cycle）との差 1052 cycle の
主因はここにあると考えられる。

したがって次の一手は、register 上で次 round の compressed weight（`uint64` = 2 VGPR）と
raw BF16 scale（1 VGPR）を先読みし、current round の WMMA の背後で
`psq4_cb10_expand` と `ds_store` を済ませる **software pipelining** である。
LDS が 128 KB/MPU あるため ping-pong buffer にしても occupancy は落ちない
（上記の co-residency 表）。

### E2E（27B-PSQ、device 1、4 run の中央値）

1-D は prefill 専用なので pp にだけ効く。プロンプトが短いほど epilogue の
相対コストが上がるため、効果は pp512 で最大になる。

| 指標 | base | 1-D | delta |
| --- | ---: | ---: | ---: |
| pp512 t/s | 2165.2 | 2184.9 | **+0.91%** |
| pp1024 t/s | 2075.5 | 2090.4 | **+0.72%** |
| pp2048 t/s | 2058.1 | 2065.2 | **+0.34%** |
| tg128 t/s | 26.11 | 26.09 | ±0（prefill 非依存） |
| GREEDY_TOKEN_SUM | 2649738 | 2649738 | 一致 |

`test_gemm_psq4_w4a8_wmma` 1270/1270、`test_gemm_psq4_decode1` 84/84、
`test_psq4_codebook` 5/5、required acceptance 96/96。

### 再現方法

```bash
# variant 強制つき sweep（device 1）
./build/phaseshift-bench gemm --dtype psq4 --rows 2048 --k 5120 --n 17408 \
  --warmup 20 --samples 30 --launches 5 --device 1 --psq4-prefill2d 3
# production selector の env override でも同じ
PHASESHIFT_PSQ_PREFILL_2D=1 ./build/phaseshift-bench gemm ...
```


---

## 7.86 PSQ4 prefill_2d: Phase 3（software pipelining）は棄却、律速は WMMA 側（2026-09-21）

§7.85 の ISA 読みから「staging の global load latency が各 round 先頭で露出している」を
主因と推定し、Phase 3 として register 先読み + LDS ping-pong + 1 barrier/round の
software pipeline を実装した。結論から書くと **棄却**で、その過程で律速の見立てが
間違っていたことが判明した。

### 実装した内容（棄却）

| 項目 | 内容 |
| --- | --- |
| LDS | `w0_lds[2][kNt][kIbc][32]` / `w1_lds` / `ws_lds` の ping-pong（`<128,128>` で 18432 → 36864 B） |
| 先読み | 次 round の compressed weight（`uint64` = 2 VGPR）と raw BF16 scale（1 VGPR）を register 保持 |
| schedule | `prefetch(round+1)` → `compute(round, cur)` → `store(round+1, cur^1)` → barrier |
| barrier | steady-state で 2/round → **1/round**（buffer 再利用は barrier で隔離） |
| expand | codebook 展開は global load の wait 後（WMMA の背後） |

spill の実測（BF16、256 threads）:

| variant | VGPR | SGPR | vgpr_spill | private | LDS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `<64,64>` | 253 | 21 | 0 | 0 | 9216 |
| `<64,128>` | 256 | 22 | 133 | 500 | 18432 |
| `<128,64>` | 256 | 21 | 205 | 784 | 18432 |
| `<128,128>` | 256 | 21 | **489** | **1864** | 36864 |

baseline の VGPR 240 に対し、先読み register（`<128,128>` で 4 slot × 3 = 12 VGPR）で
255 上限に張り付き、3/4 の variant が spill した。repository ルール（spill を許容して
採用しない）により不採用。

### そもそも律速ではなかった（L3 常駐テスト）

採用判断の前に、仮説そのものを検証した。weight を L3 常駐させる（`--weight-copies 1`、
footprint 50 MB < L3 64 MB）と、DRAM-cold（copies 2〜16）に対して:

| weight-copies | footprint | p50 (rows=2048, K5120, N17408) |
| ---: | ---: | ---: |
| 1 | 50 MB（L3 常駐） | **2469.0 us** |
| 2 | 100 MB | 2545.1 us |
| 4 | 200 MB | 2543.0 us |
| 8 | 401 MB | 2546.9 us |
| 16 | 802 MB | 2545.1 us |

**L3 常駐にしても 3.0% しか速くならない。** weight load のレイテンシ／帯域は
残り 27% の主因ではない。ISA の見た目（load→use 距離が短い）に引っ張られた誤診だった。

### 他のレバーも効かない

| 施策 | 内容 | mean delta | 判定 |
| --- | --- | ---: | --- |
| kKChunk 128→256 | barrier 半減、round あたりの独立作業を倍増 | -0.26%（1-D 込み -1.08%） | 効果なし |
| row tile 256→128（`t` 4→2） | accumulator 128→64 VGPR、VGPR 162 / spill 0、4 blocks/MPU（6→8 waves/SIMD） | **+5.01%** | 棄却 |

row tile 半減は occupancy を 3→4 blocks/MPU に上げたが、weight 再読が 2 倍
（row-block が 8→16、44.6 MB × 16 = 714 MB/dispatch）になり差し引きで負けた。
**wave を増やしても実効 IPC は 0.36 のまま**で、latency 隠蔽では説明できない。

### 律速の見立て（更新）

`<128,128>` の実測から:

```text
1 wave-round = 773 命令 / WMMA 128
1 wave-round の実測 = 約 2199 SIMD-cycle（128 SIMD、32 CU@2350MHz）
行列 pipe = 128 × 10.7 = 1370 cycle 相当 → 稼働率 62%
issue     = 773 / 2199 = 0.35 IPC（6 wave/SIMD 同居でも 0.36）
```

issue も行列 pipe も半分程度しか埋まっていない。wave を増やしても改善しないので、
**WMMA の operand 供給側**が濃厚である。マイクロベンチで operand を可変にすると
スループットが落ちる（同一条件で 1 operand set 211-230 → 可変 174-225 TFLOP/s、
ただし run 間のばらつきが大きく絶対値は信頼できない）。

実効 169-173 TFLOP/s は、operand 可変時の実測レンジのほぼ上限に位置する。
**この design（1 wave / 16x16 tile / LDS staging 無しの operand 直読み）は
すでに実用上の天井に近い**と考えられる。

### 次に試す価値があるもの

1. **prefill chunk の見直し（caller 側）**: activation working set は rows × k で、
   rows=2048 / k=5120 では 10.5 MB > L2 8 MB。E2E では pp512 が 2184.9 t/s、
   pp2048 が 2065.2 t/s で、**小さい chunk の方が 5.8% 効率が良い**。
   chunk を 512〜1024 にすると weight 再読が増えるが、L2 に収まる利点がある。
2. **operand 供給の削減**: 現状は 1 WMMA あたり a0/a1（activation 2 本）と
   w0/w1（weight 2 本）を読む。B operand は `t` 方向に 4 回再利用、A operand は
   `nt` 方向に 4 回再利用済み。これ以上の削減は tile 形状の変更を要する。
3. 別 design（multi-wave + LDS staging の tiled GEMM、§7.9 の項 3）への移行。
   operand 直読みをやめるので、今回の天井を超えられる可能性がある。

---

## 7.91 PSQ8 prefill 2D GEMM の幾何拡張（BN=128 / BK=128）（2026-09-21）

300 W の pp2048 で `gemm_psq8_w8a8_wmma_prefill_2d` は 75.44 ms（9.9%、64 dispatch）で、
PSQ4 に次ぐ単一カーネルだった。PSQ4 prefill 2d は §7.52 / §7.55 で OB（出力ブロック）と
KC（K チャンク）を 128 まで拡張したのに、PSQ8 は 64×64 固定のままだった。本節はそれを
PSQ8 へ移植した記録で、**pp2048 +0.53%**。

### 改善余地の存在

rows=2048, n=17408, k=5120（mlp_gate_up、単体 microbench）:

| kernel | p50 us | TFLOP/s |
| --- | ---: | ---: |
| PSQ8 64×64（変更前） | 約 3206（3196.8 / 3204.2 / 3217.7） | 114 |
| PSQ4 64×64（強制） | 2790.6 | 131 |
| PSQ4 128×128（production） | 2602.7 | 140 |

同じ FLOPs で PSQ8 が PSQ4 より同一幾何でも約 14% 遅い。幾何の差（PSQ4 が 64×64→128×128
で −6.7%）を加えると、PSQ8 にも移植余地があるはずと分かった。

### 幾何スイープ（rows=2048, n=17408, k=5120、interleaved 2 rep）

| OB×KC | p50 us 平均 | 64×64比 |
| --- | ---: | ---: |
| 64×64 | 3170.8 | — |
| 64×128（KC128・OB64） | 3176.6 | ±0% |
| 128×64（K64N128） | 3007.2 | −5.2% |
| 128×128（K128N128） | **2970.0** | **−6.3%** |

効果が大きいのは N 方向（OB128）。KC128 は OB128 と組み合わせたときだけ追加 −1.2%、
OB64 のままだと効果ゼロ（wmma 1 本あたりの A global load 回数が OB で決まるため）。
よって **KC128・OB64 のみ（K128N64）の variant は作成しない**。

### 形状別（production 形状、rows=2048、64×64 → K128N128 の p50）

| shape | n×k | 64×64 | K128N128 | 差 |
| --- | --- | ---: | ---: | ---: |
| mlp_gate_up | 17408×5120 | 3167.4 | 2962.4 | −6.5% |
| mlp_down | 5120×17408 | 3066.5 | 2840.9 | −7.4% |
| gdn_qkv | 10240×5120 | 1730.6 | 1612.9 | −6.8% |
| attn_q | 6144×5120 | 898.1 | 807.1 | −10.1% |
| attn_o | 5120×6144 | 843.6 | 768.6 | −8.9% |
| attn_kv | 1024×5120 | 141.2 | 134.7 | −4.6% |

### K64N128 と K128N128 の細部（interleaved、attn は 3 rep）

| shape | K64N128 | K128N128 | K128N128 の差 |
| --- | ---: | ---: | ---: |
| mlp_gate_up | 2987.2 | 2977.1 | −0.34% |
| mlp_down | 2834.9 | 2867.1 | **+1.14%** |
| gdn_qkv | 1647.1 | 1629.3 | −1.08% |
| attn_q | 933.8 | 925.6 | −0.87% |
| attn_o | 926.1 | 917.8 | −0.90% |
| attn_kv | 141.9 | 134.8 | −5.0% |

`mlp_down`（k=17408 のみ大きい）だけ KC128 が 1.1% 損をするが、層合計では K128N128 が
正味有利なので規則は特別扱いしない（K64N128 側に倒すと gate_up / gdn_qkv / attn が損）。

### 実装

- `gemm_psq8_w8a8_wmma_prefill_2d_kernel` を `<kGuarded, kOutBlock, kKChunk>` の
  template 化し、staging を PSQ4 と同じ slot ループ（`kSlotsPerWave = kNt*kIbc/8`）に
  一般化した。instantiation は K64N64 / K64N128 / K128N128 の 3 つ。
- selector（`select_psq8_gemm_config`）: 2D 適用条件を満たせば
  `ob128 && kc128 → K128N128`、`ob128 → K64N128`、`でなければ K64N64`、
  %64 自体が崩れるなら row-block へ。A/B 用に `PHASESHIFT_PSQ8_PREFILL_2D`
  （`bk64bn64` / `bn128` / `bk128bn128`）と bench の `--psq8-prefill2d N` を追加
  （PSQ4 と同じ構造）。
- ISA: VGPR は 64×64=149、K64N128=226、K128N128=**240（spill 0）**。
  K128N128 の LDS は 18432 B（w0/w1 各 8192 + ws 2048）。PSQ4 の BN128/BK128 が
  VGPR 240（§7.55）なので同一 occupancy レジーム。

### 正しさ（bit-exact）

k チャンクの進行順序（`ibg = round*kIbc + ib`）と wmma→FMA の順序は幾何に依存しない
ため、全 variant が逐 bit 一致するはず。これをテストにした:

- `test_gemm_psq8_w8a8_wmma` に `disp2d_exact` を追加。同一入力で 3 幾何を launch し
  f32 出力を **bitwise 比較**。rows=512, N=512/1024, K=256/2560 で一致
  （1271 passed / 0 failed）。
- `test_qwen35_linear_selector` を更新（K128N128 選択、`k_padded%128!=0 → K64N128`）。68 passed。
- required acceptance **97/97**（skip なし）。

### 効果

e2e pp2048（`--device 1`、interleaved、旧挙動は `PHASESHIFT_PSQ8_PREFILL_2D=bk64bn64`）。
実行間に約 4% の machine drift（absolute t/s は 2270 台→2170 台へ下降）があるため対で読む:

| batch | 旧 平均 | 新 平均 | 対差 |
| --- | ---: | ---: | --- |
| A（6 対） | 2172.4 | 2193.1 | +3.0〜+62.1、中央値 +11.6（+0.53%）。rep1 の +2.9% は warm-up 外れ値 |
| B（6 対） | 2169.2 | 2180.9 | +9.4〜+15.0、+0.54% |

**12/12 対で改善、pp2048 +0.53%**。

カーネルレベル（drift で absolute ms は比較できないため、同じ profile 内の
psq8 / psq4 比で見る。psq4 は変更されていない基準カーネル）:

| profile | psq8 ms | psq4 ms | ratio |
| --- | ---: | ---: | ---: |
| §7.90（300 W、変更前） | 75.44 | 447.83 | 0.1685 |
| 変更後 | 79.64 | 512.99 | **0.1553** |

比で **−7.9%**（形状別の −4.6〜−10.1% と整合）。

### 残る課題

PSQ8 は同一 64×64 幾何でも PSQ4 より約 14% 遅い（3170.8 vs 2790.6 µs）。構造差は
重みが 8 bit（W バイト 2 倍）か unpack 有無だけだが、§7.55 の ablation は PSQ4 で
W 経路が wmma に完全に隠蔽されることを示しており、2 倍の W traffic が 14% を説明できるか
は不明。K128N128 でも 2970 µs は wmma throughput 理論値 1772 µs の 1.68 倍で、約 40% が
stall。次は W load / barrier の ablation で stall の実体（L2 latency か、 LDS 競合か、
wave 不足か）を切り分けるべき。→ **§7.92 で実施した**（結論: 実体は供給経路のレイテンシ
露出で、差の主因は W のバイト数。隠す手段は 240 VGPR の壁で塞がっている）。

### 計測コマンド

```bash
# microbench（幾何強制 / auto）
./build/phaseshift-bench gemm --dtype psq8 --variant gemm --psq8-prefill2d 2 \
  --rows 2048 --n 17408 --k 5120 --warmup 20 --samples 30 --launches 20 --device 1

# e2e A/B（旧挙動 = 64×64 強制）
PHASESHIFT_PSQ8_PREFILL_2D=bk64bn64 ./build/phaseshift-bench pp \
  --model-dir models/Qwen3.8-27B-PSQ --prompt-tokens 2048 --mode forward \
  --page-tokens 16 --arena-gib 24 --runs 1 --warmup 0 --device 1
```


---

## 7.92 PSQ8 prefill GEMM の operand supply 分解（ablation、2026-09-21）

§7.91 の「残る課題」（同一幾何で PSQ8 が PSQ4 より約 14% 遅い）を切り分けた記録。
結論は **差の主因は W 供給路の露出**、そして **A は PSQ8/PSQ4 共通の最大コストだが、
その実体は帯域ではなく load→wmma の依存チェーンに露出するレイテンシ**である。
隠す手段は 240 VGPR の壁で塞がっている。

### 方法

`gemm_psq8_w8a8_wmma_prefill_2d_kernel` / `gemm_psq4_w4a8_wmma_prefill_2d_kernel` に
一時的な ablation 用 template パラメータを入れ、W global load / A global load /
barrier / scale FMA / footprint / load 幅を個別に除去して計測した（計測後に足場は撤去し、
production は §7.91 のまま）。各 variant の VGPR と spill は device ELF の metadata で確認し、
occupancy が変わる variant は artifact として切り分けた。

形状は rows=2048, n=17408, k=5120（mlp_gate_up、K128N128）、`--device 1`、interleaved 2 rep。

### PSQ8 の分解（K128N128、baseline 2945 µs）

| variant | p50 µs | 差 | VGPR / spill | 備考 |
| --- | ---: | ---: | --- | --- |
| baseline | 2945 | — | 240 / 0 | |
| W global load を定数化（LDS store/read は維持） | 2405 | **−18.3%** | 240 / 0 | W 露出 |
| A global load を定数化 | 1643 | −44.2% | 173 / 0 | occupancy 3→4 blocks が混入 |
| barrier 除去 | 3021 | +2.2% | 236 / 0 | 同期は無料 |
| A を 4 slot 窓（L1 常駐）から load（命令数同一） | 2858 | **−3.0%** | 236 / 0 | **A は footprint 非依存** |
| A を b128 化（命令半減・バイト同一） | 3621 | +22.9% | **241** / 0 | 240 の境界を 1 超え 2 blocks/MPU の artifact |
| W を 8 slot 窓から load（命令数同一） | 2563 | **−13.0%** | 238 / 0 | W は footprint 依存 |
| W double buffer（r+1 の load を compute r の裏へ） | 26275 | +790% | 256 / **spill 206** | 成立せず |
| 同上（K128N128） | 45376 | +1440% | 256 / **spill 503** | 成立せず |
| scale FMA を 1/8 に | 3292 | +12% | 256 / **spill 11** | FMA 本数は測れず |

### PSQ4 との比較（同一形状・同一幾何、baseline 2582 µs）

| variant | p50 µs | 差 |
| --- | ---: | ---: |
| baseline | 2582 | — |
| W global load を定数化 | 2431 | **−5.9%** |
| A global load を定数化 | 1110 | −57.0% |
| barrier 除去 | 2643 | +2.4% |

### 解釈

1. **W**: 命令数を変えず footprint だけ縮めると −13.0%、load 自体を消すと −18.3%。
   W のコストは供給経路のレイテンシであり、バイト数に比例する。PSQ8 は W8（fp8, 89 MB）
   で PSQ4 は W4（fp4, 45 MB）なので、露出が −18.3% と −5.9% に分かれる。
   **この約 12% の差が、同一幾何での PSQ8−PSQ4 差（約 14%）の主因**である。
2. **A**: footprint を L1 常駐窓にしても −3.0% しか動かない。一方 load 自体を消すと
   −44%（PSQ4 では −57%）。すなわち A は**バイト帯域ではなく、使用直前に発行される
   load が wmma の依存チェーンに露出させるレイテンシ**で時間を食っている。
3. **barrier**: 除去すると逆に 2% 遅い。round 境界の同期はコストではない。
4. **scale FMA**: 1/8 に減らすと +12% 遅くなるが、これは VGPR 256 / spill 11 の
   codegen artifact であり、FMA 本数の寄与は測れていない。

### レジスタの壁

- A の b128 化（命令数半減）は **VGPR 241** となり、240 の境界を 1 超えただけで
  2 blocks/MPU に落ちる（§7.88 の GDN で見た境界と同一）。命令数削減そのものの効果は
  測れないままである。
- W の double buffer は **VGPR 256 / spill 206〜503** で成立しない。acc（4 tile × 4 nt ×
  8 float = 128 VGPR）と staging レジスタを同時に 240 へ収める余地が無い。
- つまり **acc 128 VGPR が A/W の先読み深さを奪っている**。PSQ4 が BN128 で止まった理由も
  これで説明できる（OB256 は acc が 256 VGPR になり 9.2 ms へ崩壊した）。

### 次に試すなら

acc を削って先読みへ回す方向のみが残る。row tile を 4→2（block 128 rows）にすると
acc が 64 VGPR に減り、A を 1〜2 round 先読みする余地が生まれる（A の global バイト/出力は
不変、grid.y が倍、wave の持つ row 範囲が半減）。§7.55 と同じく「operand supply が実用天井」
という結論だが、その中身が帯域ではなくレイテンシ露出であることは本節で確定した。

### 計測コマンド

```bash
# ablation は一時的な --psq8-ablate / --psq4-ablate（測定後に撤去）で実行した
./build/phaseshift-bench gemm --dtype psq8 --variant gemm --rows 2048 \
  --n 17408 --k 5120 --warmup 20 --samples 30 --launches 20 --device 1
```


---

## 7.93 PSQ8 prefill GEMM の A 重複除去（wave マッピング変更、2026-09-21）

§7.92 の分解で「A の load が最大コスト」と分かったので、A の**重複 load を消す**方向で
改善した記録である。結果は **pp2048 +0.25%（6/6 対で改善）**、カーネル単体では
形状あたり −0.8〜−2.5%。併せて、A の先読みと W staging の double buffer は負結果だった。

### 仮説

従来の wave マッピングは 8 wave を「4 row グループ × 2 nt グループ」に割っていた。
すると同じ row 範囲を持つ 2 wave（wave_n=0/1）が**同じ A を重複して load** する。
A の global load 量はブロックあたり 64 KB/round（ユニークは 32 KB）で、ちょうど 2 倍である。

### 変更

wave を「8 row グループ × 1 nt グループ」に割り直す:

| | 従来 | 変更後 |
| --- | --- | --- |
| wave あたり row tile | 4（64 行） | 2（32 行） |
| wave あたり nt | 4（kNtPerWave = kNt/2） | 8（kNtPerWave = kNt、全 nt） |
| acc | 4 × 4 × 8 = 128 VGPR | 2 × 8 × 8 = 128 VGPR（不変） |
| A global load / block round | 64 KB（2 倍の重複） | **32 KB（重複ゼロ）** |
| VGPR（実測） | 240（K128N128） | **206** |

W の staging（slot 数 = kNt×kIbc/8）は変わらず、LDS も 18432 B のまま。
変更は 1 ファイル +13/−16 行で、template 引数の追加も無い。

### 正しさ（bit-exact）

出力要素ごとの k 進行順序と wmma→FMA の順序は不変なので、ビット単位で同一になる。
`disp2d_exact`（3 幾何の f32 出力を bitwise 比較）が PASS、`test_gemm_psq8_w8a8_wmma`
1271/1271、required acceptance 97/97。

### 形状別（rows=2048、p50 µs）

| shape | n×k | 従来 | 変更後 | 差 |
| --- | --- | ---: | ---: | ---: |
| mlp_gate_up | 17408×5120 | 2957.0 | 2927.5 | −1.0% |
| mlp_down | 5120×17408 | 2838.5 | 2766.9 | **−2.5%** |
| gdn_qkv | 10240×5120 | 1633.4 | 1603.7 | −1.8% |
| attn_q | 6144×5120 | 766.8 | 761.0 | −0.8% |
| attn_o | 5120×6144 | 767.2 | 756.4 | −1.4% |
| attn_kv | 1024×5120 | 132.5 | 130.3 | −1.6% |

層合計で約 −1.6%。

### e2e（pp2048、interleaved 6 対、新旧バイナリの直接比較）

| | 1 | 2 | 3 | 4 | 5 | 6 | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 従来 | 2186.80 | 2181.28 | 2193.59 | 2180.45 | 2179.10 | 2177.40 | 2183.1 |
| 変更後 | 2180.24 | 2180.28 | 2179.32 | 2174.83 | 2175.88 | 2175.36 | **2177.7** |

**pp2048 +0.25%（6/6 対で改善）**。

### 負結果（同時に試して棄却したもの）

1. **A のレジスタ先読み**（round r+1 の A を compute r の裏で load、K chunk 32/64/128）:
   +1.0〜+5.6% 遅い。VGPR は 193 / 220 / 256（KC128 は spill 6）で occupancy の artifact ではなく、
   **A のコストは先読みで隠れるレイテンシではない**ことを示す。
2. **W staging の double buffer**（LDS 2 面 + 1 barrier/round）: 2 通りの実装とも負け。
   - load をレジスタに保持して compute の後に store: **VGPR 256 / spill 192〜499**（成立せず）
   - stage(r+1) を compute(r) の前に置く: VGPR 215（KC64）/ 243（KC128）、spill 0 だが +1.3〜+2.1% 遅い。
   W の load を先に流すと、compute の A load と LSU を奪い合うためと考えられる。

つまり A は「重複 load を消す」方向にだけ反応し、レイテンシ隠蔽には反応しない。
残る構造的な重複は W の 2 倍バイト（fp8 形式）だけで、これは形式変更なしには消せない。

### 計測コマンド

```bash
# 形状別（新旧の比較は、変更前後のバイナリを交互に実行して取った）
./build/phaseshift-bench gemm --dtype psq8 --variant gemm --rows 2048 \
  --n 17408 --k 5120 --warmup 20 --samples 30 --launches 20 --device 1

# e2e
./build/phaseshift-bench pp --model-dir models/Qwen3.8-27B-PSQ \
  --prompt-tokens 2048 --mode forward --page-tokens 16 --arena-gib 24 \
  --runs 1 --warmup 0 --device 1
```


---

## 7.94 PSQ4 prefill GEMM への A 重複除去の移植（負結果、2026-09-21）

§7.93 の wave マッピング変更（A 重複除去）を、最大カーネルである PSQ4 prefill 2d に
そのまま移植して測った記録。**PSQ4 では効かない**（gate_up 中立、mlp_down は悪化）。

### 計測（rows=2048、新旧バイナリの交互実行、3 対）

| shape | n×k | 従来 | 移植後 | 差 |
| --- | --- | ---: | ---: | ---: |
| mlp_gate_up | 17408×5120 | 2582.6 | 2582.1 | ±0% |
| mlp_down | 5120×17408 | 2539.5 | 2558.8 | **+0.76%** |

VGPR は 240 → 223（spill 0、3 blocks/MPU 維持）で occupancy は変わらない。
`attn_kv` だけ −1.4% と改善したが、他は中立〜悪化のため採用しない（撤回済み）。

### なぜ PSQ8 では効いて PSQ4 では効かないのか

計測から絞れる事実:

1. PSQ4 の A load を**全部消す**と −57%（2582 → 1110、ただし VGPR 240 → 173 で
   3 → 4 blocks になる）。
2. PSQ4 の A load を**半分にしても 0%**（本節）。PSQ8 では同じ操作が −1.6%。
3. PSQ4 の W は fp4 で、load は 1 slot あたり b64 1 本（PSQ8 は 2 本）。

つまり A のコストは load 数・バイト数に比例しない。PSQ8 で効いたのは、W のバイト数が
2 倍（fp8）で A と供給路を奪い合っている状態を緩めたためと考えられる。PSQ4 は W が
半分なので、A を減らしても供給路に余裕が生まれない。

### 試して駄目だった検証

`__launch_bounds__(k2dThreads, 4)` で 4 blocks/MPU を狙わせても **VGPR は 240 のまま**
（acc 128 + operand で下限が決まる）で、occupancy 仮説はこの方法では検証できない。
occupancy を上げるには acc を削る設計変更（row tile を減らす = block 行数を減らす）が必要で、
それは W の再読を増やす方向と表裏になる（§7.92 の「次に試すなら」と同じ論点）。

### 結論

A 重複除去は「W が重い PSQ8」に固有の改善であり、W が軽い PSQ4 には移植できない。
PSQ4 prefill の残るレバーは、W の 2 倍バイトを消す（形式変更）か、acc を削って
occupancy を上げる設計変更のどちらかで、どちらも本節の範囲外である。
