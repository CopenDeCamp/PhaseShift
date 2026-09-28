> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# Attention カーネル最適化履歴

`docs/rnd/optimization_findings.md` から分離した、paged attention decode / prefill・KV split/reduce の実験履歴。見出し番号（7.x）は元 dump の通し番号を保持する。KV page pruning 系の PoC は同 directory の既存文書を参照。

## 収録セクション

- 7.16 paged attention の現状と書き換え設計
- 7.17 paged attention bf16/fp8 の書き換え（bf16 3.73x / fp8 2.02x）
- 7.19 paged attention prefill カーネル（query tile + bf16 WMMA、10.5x）
- 7.48 paged attention decode の KV split + reduce（flash-decoding 型 2 カーネル）
- 7.96 prefill attention の性格と、残る最大レバー
- 7.97 paged prefill bf16: K を LDS から外して末尾 barrier を削除
- 7.98 paged prefill fp8: ソフトウェア e4m3 デコードの除去

---

## 7.16 ② paged attention の現状と書き換え設計（27B 36.6% / 4B 47.5%）

### 実測（`phaseshift-bench paged-attention`、4B 形状 qh=16 kvh=4 hd=256 pt=16）

`rows=2048` 固定、visible を振る:

| visible | p50 | 比 |
|---|---|---|
| 256 | 12.05ms | 1.00 |
| 512 | 23.05ms | 1.91 |
| 1024 | 45.79ms | 3.80 |
| 2048 | 91.51ms | 7.60 |

visible に完全比例（token-visit 数に完全比例）。
2048×16 head×2048 token = 67.1M token-visit → **1.36ns/token-visit**。

### 現カーネルの構造と律速

`attention_paged_bf16.gfx1201.hip`:

```text
grid = rows * q_heads （1 block = (query 1 行, q_head 1 個)）
8 warp が「8 token」を分担して、token ごとに:
  lane0 が resolve_paged_token（page 割り算）→ __shfl ×2 で broadcast
  dot 8FMA → wave_reduce_sum（5 shfl + 5 add）→ lane0 で exp → __shfl
  PV 8FMA（acc0..acc7 に j で分岐）
これを max 用と sum/PV 用の 2 周（K を 2 回読む）
最後に acc_shared[8*256] を LDS 経由で合算
```

marginal 1.36ns/token-visit ≒ 2 cycle/token-visit @1.5GHz。token-visit あたり
~35 warp 命令（dot/reduce/exp/PV）なので **issue 率 ~4.5%**。
つまり帯域でも FMA でもなく **依存チェーン（resolve→shfl→reduce→exp→shfl）の
ストール律速**。1 block 1 行なので K/V の再利用もゼロ（L2 で拾うだけ）。

### 書き換え設計（prefill 用 flash 構成）

`attention_paged_prefill_bf16` を追加し、selector で `rows` により decode 版と分ける。

```text
1 block = (BTQ=32 query 行, q_head 1 個)、256 threads
page (16 token) 単位でループ:
  K_page[16][256] / V_page[16][256] を LDS に staged（協調ロード、16B vector）
  S[32][16] = Q_tile(32×256, LDS or reg) · K_pageᵀ
      → 1 thread あたり (r,t) ペアを 2×2 or 4×1 で持ち、内側 256 は LDS read を
        register 再利用（1 FMA あたり LDS 0.5 回）。shuffle ゼロ
  causal mask（page_start + t <= row_position[r]）と online softmax を
  行ごとに 16 lane で縮約（4 shfl、token あたりでなく page あたり）
  O[32][256] += P(32×16) · V_page → 1 thread 16 dim 保持で 16 FMA/出力
```

根拠: 4B prefill の attention = 8 layer × 2048 行 × 平均 1024 token。
現状 token-visit あたり ~2 cycle が、この構成なら
`S`(256 FMA/thread/page) + `PV`(256 FMA/thread/page) ≈ 512 FMA/thread/page の
連続 FMA になり、shuffle も per-token チェーンも消える。
K/V の実読みも 1 block = 32 行分で 32 倍の再利用。

```text
期待: attention 現状 365ms(4B PP2048) / 1138ms(27B) を数分の1
      → e2e prefill 1.3〜1.5 倍
注意: q/k は契約どおり pre-normalized。page boundary と causal mask、
      BF16 出力（output_dtype）の両対応が必要。decode(rows 小) は現行のまま。
```


---

## 7.17 ② paged attention bf16/fp8 の書き換え（実装、bf16 3.73x / fp8 2.02x）

§7.16 の診断（依存チェーンのストール律速）に対して、まずブロック構成を変えずに
チェーンを削る書き換えを行った。`attention_paged_bf16.gfx1201.hip` のみ変更。

### 変更点

```text
before:
  token ごとに lane0 が resolve_paged_token（p/page_tokens の整数除算、block_table の
  グローバルロード）→ __shfl ×2 で broadcast
  q_shared を LDS から読む（1 FMA あたり LDS 1）
  dot 8FMA → wave_reduce_sum(shfl_down ×5、結果は lane0 のみ) → lane0 で exp → __shfl
  PV 8FMA
  max 用 / sum・PV 用の 2 周（K/V を 2 回読む）

after:
  page 単位で block_table を 1 回だけロード（全 warp 同一アドレス = broadcast）。
  token アドレスは page_base + off*elems_per_token で算術的に導出（除算なし、shfl なし）
  q はレジスタ 8 本に保持（q_row[lane+32j] を最初に 1 回だけロード）
  online softmax の単一パス:
      score = warp_sum_all(partial)*scale     # __shfl_xor 蝶 = 全 lane が同じ値
      next_max = fmax(run_max, score)
      alpha = __expf(run_max - next_max); weight = __expf(score - next_max)
      run_sum = run_sum*alpha + weight
      acc[j] = acc[j]*alpha + weight*v[j]     # 8 本
  warp ごとの (run_max, run_sum, acc[8]) を最後に flash 流にマージ:
      w = exp(run_max_w - M), total = Σ acc_w*w / Σ run_sum_w*w
```

`wave_reduce_sum`（shfl_down）を `warp_sum_all`（shfl_xor 蝶）に変えたことで
全 lane が同一の score を持ち、exp と状態更新が全 lane で一様にできる（broadcast 不要）。

### 実測（gfx1201: 64 CU / 2 SIMD/CU / 32 wave/CU / 2.35GHz）

`phaseshift-bench paged-attention --check`（4B 形状 qh=16 kvh=4 hd=256 pt=16）:

| rows | visible | before | after | 比 |
|---|---|---|---|---|
| 1 | 2048 | 608.6us | 392.7us | 1.55x |
| 64 | 2048 | 2872.5us | 1690.7us | 1.70x |
| 512 | 2048 | 22591.5us | 13482.9us | 1.68x |
| 2048 | 2048 | 90594.7us | 53682.6us | 1.69x |

check は全形状 PASS（rel tol 2e-3 対 correctness カーネル）。

e2e prefill:

| | before | after | 比 |
|---|---|---|---|
| 27B PP2048 | 3200.2ms (640 t/s) | 2765.7ms (740.5 t/s) | 1.157x |
| 4B PP2048 | 808.6ms | 653.6ms | 1.237x |

27B の差分 -434.5ms は attention の -465ms と整合（attention は prefill の 36.6%）。

### 追加で効いた: head_dim 256 特化 + K/V ロードの束ね（3.73x）

この版の ISA を数えると、内側ループ 320 命令のうち **156 命令がスカラー/プレディ
ケーション**だった:

```text
s_and_saveexec_b32 31 / s_cbranch_execz 17 / s_wait_loadcnt 16 / s_or_b32 33
v_lshlrev_b32 17 / v_add_co_u32 18 / global_load_u16 16 / v_fmac_f32 15
```

原因は 3 つ:

1. `d < head_dim` ガードを unroll ループ内に置いていたため、要素ごとに
   `s_and_saveexec_b32` + `s_cbranch_execz` のペアが出ていた（ガード付きロード）。
   本モデルの head_dim は 256 固定（selector も 256 のみ）なので、
   `template <bool kFullHead>` で 256 の場合の predicate を全部消した。
2. `for (off = wave; off < page_tokens; off += kWaves)` の内側の
   `if (p >= visible) break;` が token ごとに分岐を作っていた →
   page ごとに `tokens_here` を計算してループ境界にした。
3. K と V を別々に読んでいたため `s_wait_loadcnt` がロードごとに挟まっていた →
   token の K/V スライス（各 8 本）を先にレジスタへロードしてから dot / PV を回す。

| | before(初回書き換え) | after | 元の実装比 |
|---|---|---|---|
| rows=2048 vis=2048 | 53.7ms | 24.3ms | 3.73x (90.6ms) |
| rows=1 vis=2048 | 392.7us | 92.4us | 6.6x (608.6us) |
| 27B PP2048 | 2765.7ms | 2392.1ms | 1.338x (3200.2ms) |
| 4B PP2048 | 653.6ms | 530.2ms | 1.525x (808.6ms) |

### 否定した仮説: L2 リクエスト帯域（1 block = (row, kv_head) 共有）

初回書き換え版で達成していた L1→L2 リクエストは 67.1M visit × 1KB / 53.7ms ≒
1.28 TB/s で、L2 帯域が律速という仮説を立てた。これを検証するため
「1 block = (row, kv_head)、1 warp = 1 q_head」にして 1 block 内で同じ K/V を
共有する版を実装した（L2 リクエストが q_per_kv=4〜6 分の 1 になるはず）:

| | block=(row,q_head) 8 warp | block=(row,kv_head) warp=q_head |
|---|---|---|
| rows=1 vis=2048 | 392.7us | 3210.8us |
| rows=64 vis=2048 | 1690.7us | 3183.5us |
| rows=2048 vis=2048 | 53682.6us | 52622.0us |
| 27B PP2048 | 2765.7ms | 2775.5ms |

大形状ではほぼ差がなく（1.02x）、小形状では並列度が落ちて大幅悪化。**L2 帯域は
律速ではない**と結論（K/V 共有の効果は出ない）。この版は revert した。

### 次のレバー

効いたのは「スカラー命令とロード待ちを削る」方向だったので、残りも同じ方向:

1. **fp8 版の同型移植（完了、2.02x）**。fp8 版はさらに token ごとに lane0 が
   scale をグローバルロードし `__shfl` で配る構造だったので、bf16 版と同じ
   page 単位の一様アドレス + scale 添字算術 + 単一パス化 + 256 特化を入れた。
   paged-attention bench（rows=512, vis=2048, fp8 KV）: 41442.7 → 20485.4us。
   まだ bf16 版（同形状 rows=2048 で 24.4ms 対 fp8 81.7ms）より重いのは、
   要素ごとの fp8→f16→f32 decode 2 命令と k/v scale 乗算が残っているため。
2. K/V ロードの更なる束ね: `lane` ごとに 2 次元ずつ持てば uint32 ロード 4 本で
   済む（現在 8 本）。`__bfloat162float` も 2 要素まとめて変換できる。
3. §7.16 の query tile 型 flash（softmax 縮約まで page 単位に償却する）。K/V の
   実読みが行方向に 32 倍になり、shuffle が per-token から per-page に落ちる。

### 検証

- `--check`（correctness カーネル比 rel 2e-3）: bf16 rows 1/64/512/2048 × vis 2048、
  fp8 rows 1/64/512/2048 すべて PASS。
- required acceptance 75/75 PASS、optional E2E 2/2 PASS
  （4B inference・quantizer、FP8 KV の exact 検証込み）。


---

## 7.19 ② paged attention prefill カーネル（query tile + bf16 WMMA、10.5x）

§7.16 の設計（query tile 型 flash）を WMMA で実装した。decode 形状は §7.17 の
現行カーネルのままにし、`rows >= 128` かつ single request の bf16 KV だけ新しい
`attention_paged_prefill_bf16.gfx1201.hip` へ dispatcher で振り分ける。

### 構成

```text
1 block = (32 query 行, q_head)、256 threads (8 warp)
1 page = 16 token。K/V を LDS に staged（bf16, row pitch 264）
Q の A-fragment は block 先頭でレジスタへ（warp ごと 4 k-step 分 = 16 u32）
S = Q·Kᵀ: warp = (M-tile 2) × (k-quarter 4)。k-step ごとに
    __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12 を 1 命令
    → 部分和を LDS s_part[2][4][16][17] に書き、softmax が 4 本を足す
online softmax: 全 warp が (row, token 2 本) を担当し、
    8 lane の butterfly で row max / row sum を取る（per-page に償却）
P · V: P を bf16 hi/lo の 2 項に分けて WMMA を 2 回（量子化誤差 1e-2 → 1e-5）
    O は 4 N-tile × 8 f32 = 32 f32/lane をレジスタに保持
```

causal mask は `page_start + t < row_positions[row] + 1`、tile 内のページ数は
tile 最終行の visible で決める（行ごとの visible 差は mask で落とす）。

### 実測（gfx1201、`phaseshift-bench paged-attention`、4B 形状 qh=16 kvh=4 hd=256 pt=16）

rows=2048, visible=2048, `--check` PASS:

| | before（§7.17 版） | after | 比 |
|---|---|---|---|
| rows=64 | 688.5us | 143.5us | 4.8x |
| rows=512 | 6331.4us | 553.9us | 11.4x |
| rows=2048 | 24003.3us | 2286.0us | 10.5x |

e2e prefill（`phaseshift-bench pp --prompt-tokens 2048 --runs 3`）:

| | before | after | 比 |
|---|---|---|---|
| 27B PP2048 | 2392.1ms (856.2 t/s) | 2094.5ms (977.8 t/s) | 1.142x |
| 4B PP2048 | 530.2ms (3943 t/s) | 430.7ms (4755.0 t/s) | 1.231x |

### 検証

- `--check`（correctness カーネル比 rel 2e-3）: rows 1/16/32/33/64/128/512/2048 ×
  visible 1/16/17/64/100/1024/2048/4096、qh=16/24、f32 出力すべて PASS。
  bf16 出力は現行カーネルも 2e-3 に収まらない（bench 側の閾値が f32 前提）ため
  比較対象外。
- 256 token prompt の `phaseshift-compute` で
  `PHASESHIFT_QWEN35_KERNEL_MODE=auto`（prefill 版が動く）と `=correctness` の
  GENERATED_IDS が一致。
- required acceptance 75/75 PASS、optional E2E 2/2 PASS。

### fp8 e4m3 KV 版

同じ構成で fp8 KV 版も入れた（`attention_paged_prefill_fp8_e4m3.gfx1201.hip`）。
staging で fp8→bf16 に変換して LDS に置く（e4m3→bf16 は仮数 3→7 bit で厳密）。
k_scale は softmax で `dot * scale * k_scale[t]`、v_scale は P に畳んでから
hi/lo 分割する（run_sum には入れない。現行カーネルと同じ規約）。

`--kv-dtype fp8 --check` PASS（rows 1/32/128/512/2048 × vis 1/17/100/2048/4096）:

| rows=2048 vis=2048 | before（§7.17 fp8 版） | after | 比 |
|---|---|---|---|
| fp8 KV | 80733.2us | 5519.3us | 14.6x |

bf16 版（2286.0us）より 2.4 倍重いのは、page あたり 8192 要素の
fp8→bf16 変換（要素あたり 3 命令）が staging に乗るため。

### fp8 16x16x16 WMMA の fragment layout（probe 実測、2026-09-14）

`__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` の operand/accumulator 配置を
単体 probe で確定した（`fp8_probe.hip`: fragment の (lane, byte) 1 ヶ所だけ 1 を置き、
相手 operand を all-ones にして C の非零パターン、および両側 single-1 の衝突を見る）。

```text
C accumulator: bf16 と同じ。element e <=> (row = 8*(lane>>4)+e, col = lane&15)
A operand    : row = lane&15、k = 8*(lane>>4) + byte
B operand    : col = lane&15、k = 8*(lane>>4) + byte
                -> 両 operand とも「連続 8 byte が k 8 個」で、k の上位半分 (8..15) は
                   lane 16-31 が持つ。bf16 の {4g, 4g+8}+0..3 とは別物
積の向き     : bf16 と同じ A·Bᵀ（A = [m][k]、B = [n][k]）
```

probe の生データ: A byte i (lane 0) に衝突する B は `(lane 0..15, byte i)` の 16 個だけ。
つまり byte index がそのまま k の下位 3 bit、lane>>4 が k の bit3。

これを踏まえた kernel 側の注意: k-quarter で warp を分けるとき、**B operand の k base にも
`kq * 64` を足す**必要がある（A 側だけ直すと全 warp の B が同じ k を読んで無言で壊れる。
これを踏んで fp8 QK 版は一度 check FAIL した）。

### fp8 KV の結論: gfx1201 では速度にならない（参照実装の実測、2026-08-14）

一次情報を読んだ: llama.cpp-rdna-lab D098（Qwen3.6-27B / gfx1201 / 2x RX 9070 XT、
native E4M3 KV、rocWMMA、`v_wmma_f32_16x16x16_fp8_fp8`）。

彼らの設計は我々の §7.19 の案と**同じ**:

```text
G3a: Q を kernel 内で E4M3 化（scale は MMA の後）、K は raw F8 KV から直接、
     P*V は f16 のまま  <- 我々の item 0 と一致
G3b: V も raw F8、P を E4M3 に再量子化（fp8 x fp8 -> fp32）、VKQ は fp32
利用: rocWMMA の float8_t fragment（layout は rocWMMA 任せ。混在 f16 x fp8 は reject）
P の注意: softmax 値は E4M3 の subnormal floor (2^-9) を下回るので **P を 128 倍**して
          normal 域に置き、VKQ merge で割り戻す
精度: 1e-3 NMSE を許容（f16 経路は 5e-4）。実測 ~7e-4
```

**そして速度の結論が我々の状況と一致する**:

```text
"FP8 KV on gfx1201 does not beat f16 at any lane; only the prefill KQ phase is
 cost-neutral. FP8 KV remains a memory-saving option (half the cache), not a
 speed option, on this backend."

（その後 spill-free 8-wave で f16 とほぼ同等まで詰めたが、勝ち幅は q8_0 比で
  +5% 程度。f16 比では同等。つまり fp8 経路の利得は「KV が半分になる」だけ）
```

我々の実測も同じ形:

```text
bf16 KV prefill flash : 2286us   <- 変換なし + bf16 WMMA
fp8  KV prefill       : 5642us   <- K を fp8->bf16 変換して bf16 WMMA
fp8 QK 版（試作/撤回）: 4291us   <- それでも bf16 KV 版に負ける
```

= **gfx1201 で fp8 KV の attention は bf16 KV より速くならない**（変換・scale・
operand 生成のコストが KV 読み出しの帯域削減を食い尽くす）。KV メモリを半分に
したいときだけ意味がある。なので fp8 QK/PV の最適化は追わない（fidelity も落ちる）。

このとき rocWMMA を使う理由が明確になった: 生 builtin の fp8 operand layout は
probing で値レベルまで確定できず（0x38=1.0 を入れて 2060 が返る = e4m3 として
読まれていない）、rocWMMA の fragment API に任せるのが正解。我々の bf16 経路は
GDN カーネルから layout が分かっているので生 builtin で問題ない。

**決定**:
1. attention は **bf16 KV を主経路**として維持（現状の 2286us / 10.5x）。
2. fp8 KV は「メモリ節約オプション」として K→bf16 版（5642us / 14.3x、--check PASS、
   token 一致）を維持。fp8 QK/PV は追わない。
3. 次のレバーは int8 GEMM（prefill の ~11%、29.6 TFLOPS = ピークの ~20%）と
   multi-request batch。

### 未着手

1. multi-request batch（descriptor + grid を (max_chunks, num_requests, q_heads)）。
2. int8 GEMM の fp4 並み最適化。


---

## 7.48 paged attention decode の KV split + reduce（flash-decoding 型 2 カーネル）（2026-09-19）

### 経緯

- 27B decode の長コンテキスト崩壊（todo「visible 4096 の壁」実施順 1）:
  旧カーネルは grid = rows×q_heads（bs=1 で 24 block）、1 block が
  全 page を直列 walk。visible 比例で latency が増え、
  405.6 µs/層 @8k → 2.15 ms/層 @32k → 14.1 ms/層 @262k。
  64 CU に対して 24 block（37.5%）で並列化が頭打ち。
- 設計候補 A「GQA group のみ」（block = (row, kv_head)、
  グループ内全 q_heads を 1 block で処理し K/V 重複読取を 6× → 1× にする）
  は実測で却下: rows=1 vis 32768 で 5384 µs（旧 2145 µs の 2.5-3× 劣化）。
  bs=1 だと 4 block × 4 wave = 16 wave しか立たず、
  DRAM in-flight 深度が不足（旧は 24 block × 8 wave = 192 wave）。
  spill 無し（NumVgprs 166、Scratch 0）を確認済み。
- 採用: **GQA group + KV split の 2 カーネル**。
  - kernel A `attention_paged_split_bf16_impl`: grid = rows×kv_heads×S、
    block = (row, kv_head, split)。page を S 分割し
    （per = ceil(page_count/S)）、グループ内全 q_heads（27B: 6、4B: 4）の
    online softmax を走らせる。4 wave（128 threads）、K/V をトークン毎 1 回だけ読取。
    epilogue で 4-wave merge して partial (m, P, A[256]) を global 書込
    （正規化しない、weight = exp(score - m) の積和）。
  - kernel B `attention_paged_split_reduce_bf16_impl`: grid = rows×kv_heads。
    g = max_s m_s、W_s = exp(m_s - g) として
    out = Σ_s W_s·A_s / Σ_s W_s·P_s（P_s は A_s の合計 weight であり
    out の正規化項）。
  - partial レイアウト（f32 平坦、(row, kv_head, split) ごとに
    q_per_kv×(2+256)×4B、27B で 1 split 6192B）:
    [0..qpk-1]=m、[qpk..2qpk-1]=P、[2qpk + qh×256 + d]=A。
  - 空 split は m=-FLT_MAX / P=0 / A=0 の dummy として自然に寄与 0。
  - 旧カーネル（`attention_paged_bf16_impl`、8 wave）は git main から
    byte-identical に復元し維持（visible < 2048 と prefill と
    batch prefill は旧カーネルのままで、ベースライン不変）。
  - 初回 GQA-only 試作は実測で却下したため、本カーネルは group + split
    を最初から組み合わせた形（split が occupancy 不足を解消する）。

### dispatch / workspace

- `paged_attention_dispatch.hip` の `try_launch_paged_attention` 内 strategy:
  BF16・非 prefill・`ctx.decode_attn_partials != nullptr`・
  `ctx.max_visible_tokens >= 2048`・`q_heads % kv_heads == 0` の全てで **S=16 固定**。
  当初の 8192→4 / 16384→8 / ≥65536→16 の段階 S 案は計測で
  S=16 が全 visible で最良（下表）のため廃止。selector / args 変更なし。
- partials は execution workspace 末尾の scratch 直後に **8MiB 固定**
  （`runtime::kDecodeAttnPartialBytes`、arena +0.4%）。
  必要量 rows×kv_heads×16×q_per_kv×(2+256)×4B を超過すると（27B で rows>21）
  旧カーネルへ自動 fallback。
- fp8 KV は旧カーネル維持（decode 2.75× 遅延の異常が未調査のため凍結候補）。
- `record_physical_launch(2)`（2 launch 分を記録）。

### per-shape 計測（bench paged-attention、27B profile q24/kv4/hd256/pt16、rows=1、p50 µs）

| visible | 旧カーネル | S=2 | S=4 | S=8 | S=16 | S=16 比 |
|---|---|---|---|---|---|---|
| 512 | 27.8 | - | - | - | 40.6 | 0.68× |
| 1024 | 50.2 | - | - | - | 44.9 | 1.12× |
| 2048 | 93.7 | - | - | 55.9 | 52.6 | 1.78× |
| 4096 | 205.8 | 277.3 | 148.3 | - | 70.7 | 2.91× |
| 8192 | 362.0 | - | 279.8 | 155.0 | 106.0 | 3.41× |
| 16384 | 730.6 | - | - | 290.2 | 186.8 | 3.91× |
| 32768 | 2027.4 | - | - | 653.4 | 376.9 | 5.38× |
| 65536 | 3842.4 | - | - | - | 708.3 | 5.43× |
| 262144 | 14425.6 | - | - | - | 3081.3 | 4.68× |

- S が大きいほど常に有利（block 数 = MLP 深度が律速）。
- クロスオーバーは 1024-2048 の間。閾値は margin 確保のため **2048** に設定。
- rows=8: 8192 で 640.1 → 401.5（1.60×）、32768 で 2850.0 → 1948.9（1.46×、
  33.5MB×8 の BW 下限 1.91ms に到達）。

### E2E A/B（27B-PSQ、bench tg greedy 64 tok、device 0、arena 24GiB）

| context | 旧（base） | 新 | 変化 | GREEDY_TOKEN_SUM |
|---|---|---|---|---|
| 128 | 26.79 t/s | 26.80 t/s | 不変（経路不変） | 186677 |
| 2048 | 25.67 t/s | 26.03 t/s | 1.01× | 1612327（bit 一致） |
| 8192 | 22.42 t/s | 25.39 t/s | **1.13×** | 849103（bit 一致） |
| 32768 | 14.33 t/s | 22.65 t/s | **1.58×** | 280260（下記） |

- 32768 の 25.7ms/token 削減 = 16 層 × (2027-377)µs の attention 差分と整合。
- 32768 の token 分岐について: 旧カーネル固定ビルド（dispatch 閾値を
  無効化）はベースラインと **bit 一致**（SUM=464326、先頭 8 トークン同一）。
  split ビルドは token 7 で分岐（271,40,5125,421,678,1876,5435,264 vs
  …,16661,11040）。原因はカーネル差そのもの: 同一 input で
  旧カーネルと split カーネルの attention 出力を直接比較し
  **max_abs = 4.3e-6、max_rel = 8.3e-7**（f32 丸めのみ、softmax 計算順序差）。
  248320 vocab の greedy で top-2 logit の near-tie が 64 step 中に
  1 箇所以上現れるのは統計的に当然であり、実エラーではない。
  acceptance 判定は bench `--check`（correctness reference 比 2e-3）で
  判定し、これは 2400× margin で PASS。
- 2048 / 8192 で SUM が bit 一致したのは、短ctx では attention の
  peaky な分布が f32 丸め差を消したため。

### correctness（bench paged-attention --check、correctness reference 比 2e-3）

- 27B（q24/kv4/hd256）: rows {1,8} × visible {128, 2048, 8192, 16384,
  32768, 65536, 262144} × splits {2,4,8,16} 全て PASS。
- 4B（q16/kv4/hd256、q_per_kv=4 の別 template）: rows {1,8} ×
  visible {128, 2048, 8192, 32768} × {S=1 旧カーネル, S=16} 全て PASS。
- 旧カーネルの byte-identical 復元を確認（diff main は
  `kHeadLimit = 256u` → `kHeadLimit = kPagedAttentionHeadLimit`（=256u）の
  定数参照化のみ、カーネル本体は変更なし）。
- ctest: 76/78（残 2 は既存の gfx1151 arch-gated skip、本変更と無関係）。
- 4B-PSQ decode（ctx=128、split 未発火経路）: tg forward 8,940 µs/tok、
  greedy SUM=1240072。§7.46 の 17,782 は psq4 GEMM 時点の stale 値
  （以降の fusion 群で 8.9ms 台まで改善済み、本変更の前後で差なし）。

### 残 backlog

- fp8 KV decode の 2.75× 遅延（旧カーネル自体の異常）は未調査。
  本 split は BF16 のみ有効。
- 262k の production A/B（tg）は未実施（prefill が数分かかるため）。
  カーネルは bench CHECK（262144、S=16）で検証済み。


---

## 7.96 prefill attention の性格と、残る最大レバー（2026-09-21）

conv1d の次に prefill attention（`v_prefill_bf16::attention_paged_prefill`）を調べた記録。
結論は **attention は staging 律速ではなく wmma/softmax/barrier 構造律速**で、
現時点で残る最大のレバーは **keep-alive の production 化（+6.0%）** である。

### attention の構成

27B は 64 層の hybrid で、4 層ごとに full attention（16 層）、残り 48 層が GDN
（`layer_types` の `full_attention_interval = 4`）。attention は
q_heads=24 / kv_heads=4（GQA 6:1）/ head_dim=256 / page_tokens=16。

`v_prefill` は 30.77 ms（16 dispatch、3.6%）。カーネルは 32 行 × 1 head / block、8 warp、
page（16 token）ごとに「K/V staging → QK^T wmma → softmax → PV wmma」を回し、
**1 page あたり 4 回の barrier** を打つ構造。

### 計測（rows=2048、210 W）

| visible | p50 | FLOPs | 実効 |
| ---: | ---: | ---: | ---: |
| 256 | 454 µs | 12.9 GFLOP | 28.4 TFLOP/s |
| 1024 | 1782 µs | 51.5 GFLOP | 28.9 TFLOP/s |

参考として bf16 GEMM（`gemm --dtype bf16 --rows 2048 --n 5120 --k 5120`）は
107.4 GFLOP / 4719 µs = **22.8 TFLOP/s** で、attention はそれ以上の実効レートを出している。

### staging 律速ではない（ablation）

K/V のページ offset を外して常に同じ 16 KB（L1 常駐）から読ませた（interleaved 3 対）:

| visible | 通常 | L1 常駐 | 差 |
| ---: | ---: | ---: | ---: |
| 256 | 453 µs | 424 µs | −6.4% |
| 1024 | 1781 µs | 1741 µs | **−2.3%** |

つまり時間のほぼ全部が staging 以外（wmma、softmax、barrier、warp 間の受け渡し）にある。
改善するなら 1 page あたり 4 barrier の削減（K/V の double buffer で末尾 barrier を消す、
2 page を 1 反復にまとめる等）だが、online softmax の max/sum 更新順序が bit-exact の
制約になるため、単純な page 統合は不可。

### 残る最大レバー: keep-alive の production 化

pp2048 で bench の `--gap-keepalive 1000` を今の 210 W で測ると:

| | 1 | 2 | 3 | 平均 |
| --- | ---: | ---: | ---: | ---: |
| off | 2220.31 | 2218.26 | 2220.09 | 2219.6 |
| on | 2352.12 | 2352.15 | 2355.06 | **2353.1** |

**+6.0%（3/3 対）**。§7.89 で機構（CP の power-gating、長い kernel の後に 235〜265 µs の
wake-up）と最適間隔（950〜1100 µs）、event flag（`hipEventDisableSystemFence`）は特定済みで、
残っているのは executor への組み込みだけである。カーネル単位の残り（attention で
+0.7〜1.0% 程度、elementwise は天井）より一桁大きい。

### 計測コマンド

```bash
./build/phaseshift-bench paged-attention --variant prefill --actual-qwen-profiles \
  --model-dir models/Qwen3.8-27B-PSQ --page-tokens 16 --arena-gib 24 \
  --warmup 10 --samples 20 --launches 10 --device 1
./build/phaseshift-bench pp --model-dir models/Qwen3.8-27B-PSQ --prompt-tokens 2048 \
  --mode forward --page-tokens 16 --arena-gib 24 --runs 1 --warmup 0 --device 1 \
  --gap-keepalive 1000
```

---

## 7.97 paged prefill bf16: K を LDS から外して末尾 barrier を削除（2026-09-27）

§7.96 で「prefill attention の残るレバーは 1 page 4 barrier の削減」と整理した点の追試。
`v_prefill_bf16::attention_paged_prefill_bf16_impl<0u>`（pp2048 で 16 dispatch、3.6%）を
対象に、低リスク項目から 1 項目ずつ独立計測した。

### 条件

- base revision `b60a8375`、branch `poc/paged-prefill-barrier`
- device 1 / Release / HIP graph OFF / fusion OFF / `Qwen3.8-27B-PSQ`
- microbench `phaseshift-bench paged-attention --variant prefill`、
  rows=2048 / q_heads=24 / kv_heads=4 / head_dim=256 / page_tokens=16
- baseline と candidate は交互 A/B（5 round、samples 30 / launches 10、p50 比較）
- barrier ablation は正しさを捨てた上限測定

### barrier コストの上限

page loop の 4 barrier を全て除去した ablated build（不正）: 3264 → 2960 µs = **−9.3%**。
1 barrier あたり ~2.3%。ここが page loop barrier の上限。

### 低リスク 3 項目（vis2048 bf16 p50）

| item | 内容 | baseline | candidate | 判断 |
| --- | --- | ---: | ---: | --- |
| 1 | 最終 page の barrier 削除 | 3255 µs | 3290 µs（+1.0%） | 棄却 |
| 2 | `s.row_alpha[tid]=0` 初期化削除 | 3265 µs | 3267 µs（±0.2%） | 効果なし |
| 3 | full-page / tail の load path 分離 | 3269 µs | 3232 µs（−1.1%） | 採用 |

- item 1: 512 barrier 中 1 回しか削れず、条件分岐コストが上回る。ISA は barrier 命令が
  条件分岐内に残り命令数 +5。
- item 2: kernel 先頭で 1 回のみの dead store。
- item 3: full page では `tokens_here==kPageTokens` が常に成り立つため、K/V load loop から
  zero-fill と tail predicate を除去。出力は byte-identical。

### barrier 削減の試行（不採用）

| 方式 | 結果 | 判断 |
| --- | --- | --- |
| v_page 2 面化 + 末尾 barrier 削除 | LDS 28160 → 36608 B、2 → 1 block/WGP、+4.1% | 棄却 |
| `s_part` f16 化 + v_page 2 面化 | `test_paged_attention` 20/21 FAIL、`max_rel=1.2e-1` | 棄却 |
| `global_inv` を落とした軽量 barrier | `test_paged_attention` 20/21 FAIL | 棄却 |

- v_page 2 面化はデータフロー上正しい（21/21 PASS、byte-identical）が、barrier 削減 ~2.3% に対し
  占有率半減の損失が上回る。
- gfx12 は L1 と LDS が統合されており、`__syncthreads()` が吐く `global_inv` は LDS 書き込みの
  可視化に必要。除去すると壊れる。
- `sharedMemPerMultiprocessor = 65536 B`。2 block/WGP 維持には 1 block ≤ 32768 B が必要。

### 採用: mode 0 で K を global 直読み + v_page 2 面化 + 末尾 barrier 削除

mode 0（bf16 KV pool）では K を LDS に置かず、QK が global の K を `frag_nk` で直接読む。
store loop は V のみを書く。`k_page`（8448 B）を削除し v_page を 2 面化（+8448 B）することで
**LDS は 28160 B のまま 2 block/WGP を維持**し、page loop 末尾の barrier を除去する。
`k_page` / `v_page` を持つ `PrefillShared` は kMode ごとに struct を分離。

| shape（rows=2048 bf16） | baseline | 採用後 | 差 |
| --- | ---: | ---: | ---: |
| vis2048 | 3243 µs | 2852 µs | **−12.1%** |
| vis1024 | 1737 µs | 1540 µs | −11.3% |
| vis256 | 404 µs | 349 µs | −13.5% |
| vis2048 f32 | 3289 µs | 2924 µs | −11.1% |
| vis2048 rows1024 | 1737 µs | 1535 µs | −11.6% |

- 正しさ: `test_paged_attention` 21/21 PASS、F32 出力は baseline と byte-identical。
- e2e pp2048: 2352 → 2361 gpu_tokens_per_sec（**+0.38%**、4 round 交互）。
- ISA（`impl<0u>`）: `s_barrier_signal` 6 → 5、`global_inv` 6 → 5、命令数 2006 → 2072、
  VGPR 117 / SGPR 42（spill 0）、code object 11448 → 11768 B。

ablation 上限 9.3% に対し実測 12% が出た差の ~3% は、store loop から K の global load と
LDS write を消した分である（K の global read 回数は変わらず、LDS 往復だけが消える）。

### 判断

採用。`poc/paged-prefill-barrier` に 2 commit:

- full-page / tail load path 分離
- mode 0 の K global 直読み + v_page 2 面化 + 末尾 barrier 削除

### 現在への影響

- mode 0 の QK は K を global（L2）から読む。`k_page` は modes 1/2（psq2 / psq4 KV）のみ。
- 出力 dtype の runtime 分岐、`q_per_kv` の runtime 除算、`1/row_sum` の full division は未着手。
- ISA は最適化前（base revision）の `docs/paged_prefill_bf16_impl_0.gfx1201.s` ほか 3 kernel 分を参照。

### 計測コマンド

```bash
# kernel A/B
./build/phaseshift-bench paged-attention --variant prefill \
  --rows 2048 --visible 2048 --q-heads 24 --kv-heads 4 --head-dim 256 \
  --page-tokens 16 --out-dtype bf16 --warmup 20 --samples 30 --launches 10 --device 1

# correctness
./build/tests/test_paged_attention

# e2e
./build/phaseshift-bench pp --model-dir models/Qwen3.8-27B-PSQ \
  --prompt-tokens 2048 --mode forward --page-tokens 16 --arena-gib 24 \
  --runs 1 --warmup 0 --device 1
```

---

## 7.98 paged prefill fp8: ソフトウェア e4m3 デコードの除去（2026-09-27）

§7.48 の backlog「fp8 KV の 2.75× 遅延」の追試。実際は decode だけでなく
**prefill（`attention_paged_prefill_fp8_impl`）も bf16 の約 2.7× 遅い**。

### 再現（rows=2048 qh=24 kvh=4 hd=256、p50）

| visible | bf16 | fp8 |
| ---: | ---: | ---: |
| 256 | 576 µs | 1115 µs |
| 1024 | 1631 µs | 4288 µs |
| 2048 | 3152 µs | 8509 µs |

### 原因

`e4m3_decode_f32`（`__hip_cvt_fp8_to_halfraw`）は gfx12 でハードウェア変換に落ちず、
LLVM が NaN/Inf 判定の分岐（`v_cmpx` + `s_cbranch_execz`）、`v_clz_i32_u32`、
shift/or 連鎖を含む ~25 命令のソフトウェア E4M3→f16 デコードを展開していた。
load も 1 バイトの `global_load_d16_u8` が 16 本で、各々 64-bit アドレス計算が付く。

baseline fp8 prefill ISA: 2817 命令、`global_load_d16_u8` ×16、`v_cvt_f32_f16` ×16。

### 対策: LDS の 256 entry bf16 LUT

`fp8_lut[256]`（512 B）を block 先頭で現行デコードにより 1 回だけ構築し、load loop を
「8 バイト `uint2` ロード + バイト抽出 + LDS 参照 1 回」に置き換える。

### 結果（fp8 prefill、p50）

| visible | baseline | LUT | 差 |
| ---: | ---: | ---: | ---: |
| 2048 | 8589 µs | 4484 µs | −47.8% |
| 1024 | 4376 µs | 2373 µs | −45.8% |
| 256 | 1250 µs | 571 µs | −54.3% |

- ISA 2817 → 2162 命令、`global_load_d16_u8` 16 → 0、`v_cvt_f32_f16` 16 → 1
- VGPR 113 / SGPR 42
- 出力は baseline と byte-identical
- bf16 prefill（§7.97 適用後 2852 µs）比で **2.7× → 1.57×**

### 判断

採用。残差 ~1.57× は PV を `p_tile`(hi) + `p_lo`(lo) の 2 回 bf16 WMMA で回す分と
scale 乗算。

### 現在への影響

- gfx12 には fp8 WMMA（`__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`）が
  あり、psq4 / psq8 で使用済み。fp8 prefill は Q / P を fp8 化すれば K/V を
  デコードせず fp8 tensor core を使える（別 PoC）。
- `test_paged_attention` の prefill ケースは bf16 のみで、fp8 prefill は required test の
  対象外。数値検証は bench の dump 比較に依存している。

### 計測コマンド

```bash
./build/phaseshift-bench paged-attention --variant prefill --kv-dtype fp8 \
  --rows 2048 --visible 2048 --q-heads 24 --kv-heads 4 --head-dim 256 \
  --page-tokens 16 --out-dtype bf16 --warmup 20 --samples 20 --launches 10 --device 1
```

### fp8 prefill の後続検討

LUT 版は「fp8 を bf16 に落として以降は bf16 WMMA」だが、bf16 prefill とは別カーネルで、
per-token scale と PV の hi/lo 2 回 WMMA が残る。この差分を詰めた記録。

#### full-page fast path（採用）

bf16 と同じく full page / tail page を分離し、full page から zero-fill と tail predicate を
除去。vis2048 4484 → 4439 µs（**−1.0%**）、出力は byte-identical。

#### 末尾 barrier の条件化（不採用）

最終 page のみ末尾 barrier を省くと **+1.3%**。bf16 の item 1 と同じく、1 kernel あたり
1 回しか削れず分岐コストが上回る。

#### PV の hi/lo 削除（不採用）

PV の `p_lo` 項を落として 1 回 WMMA にすると **−6.6%**（4439 → 4148 µs）だが、
f32 reference 比 `max_rel` が 7.8e-3 → 2.5e-2 に悪化し、prefill 許容 1e-2 を超える。

#### fp8 WMMA QK（NO-GO）

gfx12 には fp8 WMMA（`__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`、psq4 / psq8 で
使用）があるため、K を fp8 のまま LDS に置き Q を E4M3 量子化して QK を fp8 tensor core で
回す版を実装した。fragment layout は単体 probe で
`C[m][n] = Σ_k A[m][k]·B[n][k]`、A/B とも lane =（row `l&15`, col `(l>>4)*8`）と確定。

| Q の表現 | vis16 max_rel | vis2048 max_rel |
| --- | ---: | ---: |
| per-row E4M3 | 2.4 | 7.0 |
| hi/lo 2 項 fp8（~6 bit） | 0.13 | 0.23 |

Q の E4M3 量子化による score 誤差が softmax で増幅され、bf16 QK（f32 比 1.5e-2）に対して
約 10× の誤差になる。2 項 fp8 でも許容に届かず不採用。
