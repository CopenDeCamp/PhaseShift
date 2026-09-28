> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

> Historical R&D record. GPU-MCU runtime 再構成のため、production の Fusion 実装は
> 撤去された。本文書に登場する fusion flag・kernel・bench は現在の production には
> 存在しない。

# Gate 11J Series Progress（C ～ K）

Safe/Exact launch fusion の連続 program。各 Gate は独立 compile flag を持ち、default OFF。

baseline: Gate 11J-B final `d79f5c92`（branch `gate/11j-c-k-fusion-series`）

cumulative baseline（ADOPT 済みのみ）:

| flag | state | 根拠 |
| --- | --- | --- |
| `PHASESHIFT_FUSE_RESIDUAL_RMSNORM_QUANT` | ON | Gate 11J-A ADOPT |
| `PHASESHIFT_FUSE_TARGET_SWIGLU_QUANT` | ON | Gate 11J-B ADOPT |
| `PHASESHIFT_FUSE_DFLASH2_SWIGLU_QUANT` | OFF | Gate 11J-B PERF-NEUTRAL（default OFF 維持） |

| Gate | family | Target/DFlash | decision | commit |
| --- | --- | --- | --- | --- |
| 11J-C | residual + RMSNorm + activation quantize | DFlash | **ADOPT** | |
| 11J-D | GDN prepare | Target | **ADOPT** | |
| 11J-E | GDN post | Target | **ADOPT** | |
| 11J-F | attention gate + quantize | Target | **ADOPT** | |
| 11J-G | Q postprocess | Both | Target **REVERT** / DFlash2 **PERF-NEUTRAL** | |
| 11J-H | K postprocess | Both | Target 実装せず / DFlash2 **PERF-NEUTRAL** | |
| 11J-I | conv boundaries | DFlash | 未着手 | |
| 11J-J | INT2 coarse + TopN | DFlash head | 未着手 | |
| 11J-K | rerank + Top16/selector | DFlash head | 未着手 | |

### Gate 11J-C メモ

- 対象 chain（actual code から確認）: A = `residual_add → rmsnorm → quantize`（attention tail、
  3 → 1）、B = `rmsnorm → quantize`（layer 先頭、2 → 1）。quantize は
  `dflash2_grouped_conv_prepare` 内の conv kernel projection linear の最初の launch で、
  rmsnorm 直後に隣接する。5 layer × (A + B) = 10 fusion / draft forward、-15 launch。
- `mlp_prepared` / Q/K/V の quantize は dynamic conv を挟むため非隣接 → 対象外。
- 専用 device body（`kernels/dflash2/detail/rmsnorm_device.h`）を standalone と共有。
  intrinsic・演算順・geometry を同一に保つため split-only build は byte-identical。
- build: `build-g11jck-a`（cumulative baseline）/ `build-g11jck-b`（+ 本 Gate）。
- focused test 21/21 byte-exact、primitive 1.12-1.15x、proposer backbone -0.82% /
  proposer -0.64%（3/3）、`dflash2_rrq_hits=1750`。
- required A 129/129・B 130/130（skip 0）、DFlash2 ids == target-only greedy（128/128）、
  gate11c failed=0、E2E median +0.01%（非劣化）。
- decision: **ADOPT**（proposer の一貫した改善 + launch -15/draft forward + byte-exact）。

Gate 11J-D 以降の cumulative baseline は 11J-C を ON にした build を基準とする。

### Gate 11J-D メモ

- 対象 chain（actual dispatch 順）: `SILU(F32→BF16, 10240) → L2_NORMALIZE(Q view 2048, g128) →
  L2_NORMALIZE(K view 2048) → SCALE(2048)` が連続 4 dispatch。48 GDN layer。
- Q/K は同一 `act` の view なので 1 kernel にまとめてよい（別 branch ではない）。
- `kernels/optimized/gdn/detail/gdn_prepare_device.h` に silu / l2norm の device body を抽出し、
  standalone の `op_silu_f32_bf16` / `l2_normalize_bf16_f32_kernel` と共有（同一 intrinsic・同一順）。
- fused kernel: grid=rows、256 thread、`act`/`q_norm`/`k_norm`/`q_scaled` を materialize。
  scratch 0、SGPR26/VGPR34/LDS0。
- matcher は workspace の部分 overlap を拒否するため 47/48 layer が対象。1 layer は allocator が
  `k_norm` を `conv` buffer 先頭に置くため split へフォールバック。
- focused test 18/18 byte-exact、primitive 1.70-1.76x、verify wall M1 -0.90% / M3 -0.90% /
  M8 -0.75%（すべて 4/4 pair 改善）、physical -141/update（= 3 × 47）。
- decision: **ADOPT**（verify wall M1/M3 -0.90%、M8 -0.75%、すべて 4/4 pair 改善、primitive 1.70-1.76x、physical -141）。
- required A 130/130・B 132/132（skip 0）、DFlash2 ids == target greedy、gate11c failed=0、
  E2E median +0.80%（15/15 改善）。

Gate 11J-E 以降の cumulative baseline は 11J-D を ON にした build を基準とする。

### Gate 11J-E 事前調査メモ（実装前）

- GDN post chain（dispatch 順）: `GDN_RECURRENCE → RMS_NORM(gdn_norm, F32→F32, group 128,
  DIRECT weight) → SILU(gdn_z_silu, v_z BF16→F32, 6144) → MUL(v_rms, v_gated_z → v_gated BF16)
  → ACTIVATION_QUANTIZE(out linear) → LINEAR`。48 layer。
- z の SiLU は producer が直前ではない（consumer side fusion）。
- RMS は J-A で抽出済みの `detail/rmsnorm_row_body<F32,F32,BF16,PerGroup,DIRECT>` を
  standalone と共有できる。standalone は group_size=128 のとき block=32（1 warp）で
  (groups, rows) grid。
- `rmsnorm_row_body` は `threadIdx.x`/`blockDim.x` を直接使うため、block=32 の
  (groups, rows) topology で呼ぶのが唯一そのまま共有できる形。
- したがって本 Gate は `RMS + SiLU + MUL` を 3→1 にする（quantize は別 dispatch のまま）。
  4→1 にすると row 全体を 1 block に置く必要があり、(a) quantize を 32 thread で回すと
  VGPR spill（k=6144 で kNChunk=24）、(b) norm body を per-warp 用に書き換える必要がある
  （過去 Gate の共有 body を変更することになる）ため採らない。
- 予定 saving: 3→1 で -2 launch / layer = -96 / update。
- 実測: focused 14/14 byte-exact、primitive 2.47-2.56x、verify wall M1 -0.74% / M3 -0.79% /
  M8 -0.76%（各 4/4 改善）、physical -96/update（48/48 layer 発火）、E2E median +0.73%。
- required A 133/133・B 134/134（skip 0）、gate11c failed=0、split-only greedy ids 不変。
- decision: **ADOPT**。
- silu は Gate 11J-D で共有化済みの `detail::gdn_silu_f32` を `op_silu_bf16_f32` にも使わせる。
  MUL は `op_mul_f32_f32_bf16` から device body を抽出して共有。

### Gate 11J-G 結果

- Target 側: `RMS_NORM → ROPE(pair)` の 2→1 を実装・実測したが、共有 body の制約で row 内 head を
  逐次処理する形になり primitive が **0.36x**（q）/ 1.03x（k）→ **REVERT**（code は commit しない）。
- DFlash2 側: norm+rope を 2→1。focused 25/25 byte-exact、primitive 1.44-1.63x、
  proposer backbone -0.19%（6/7）/ proposer -0.13%（4/7）、E2E median +0.02%（8/15）→
  **PERF-NEUTRAL**（flag default OFF で保持）。
- required A 137/137・B 138/138（skip 0）、gate11c failed=0、dflash2 refactor ids 不変。

### Gate 11J-F 結果

- 実測: focused 13/13 byte-exact、primitive 1.66-1.76x、verify wall M1 -0.33% / M3 -0.28% /
  M8 -0.30%（各 4/4 改善）、physical -32/update（16/16 layer 発火）、E2E median +0.22%。
- required A 135/135・B 136/136（skip 0）、gate11c failed=0、split-only greedy ids 不変。
- decision: **ADOPT**。

### Gate 11J-F / G / H 事前調査メモ

Target full attention layer の dispatch 順（`lower_to_primitives.cpp` 800-890 行）:

```
q_gate_split → q_norm(RMS) → q_rope → k_norm(RMS) → k_rope → kv_append
             → attn_core(PAGED_ATTENTION) → attn_gate(SIGMOID) → attn_gated(MUL)
             → o_proj(LINEAR + activation quantize) → residual_attn → post_norm → MLP
```

- Gate 11J-F: 対象は `SIGMOID(v_gate) → MUL(v_ctx, v_gs) → ACTIVATION_QUANTIZE` の 3 dispatch
  （16 full attention layer、q_features=6144）。`SigmoidBf16ToF32`/`MulF32F32ToBf16` は
  ともに 6144 の Optimized rule あり。`PAGED_ATTENTION` は触らない。saving −2/layer = −32。
  J-B の fused_swiglu_quant と同型（2 入力 → BF16 中間 → quantize）なので同じ段取りで作れる。
- Gate 11J-G: `q_norm(RMS, group 256, ONE_PLUS) → q_rope` の 2 dispatch（Target: BF16→F32→BF16）。
  DFlash2 側は `dflash2_rope_bf16` を使っており contract が別なので個別実装・個別判定。
- Gate 11J-H: `k_norm → k_rope`（2 dispatch）。Target は直後に `kv_append` が独立 dispatch なので
  Candidate A（norm+rope）を先に評価し、問題なければ Candidate B（+kv_append）を試す。
  KV cache の page layout / position / head indexing は変更禁止。

計測環境（Gate 11J-B から継続）:

- `ROCR_VISIBLE_DEVICES=N` は rocm-smi の GPU index と一致（PCI bus id で確認済み）。
- 各 campaign 開始時に `hipMemGetInfo` で 31GB 以上の空きを確認。実行中は free VRAM を
  サンプルし、自分の arena のみで推移することを確認する。
- 本機の rocm-smi KFD table は process → GPU 帰属が信頼できないため、in-window の帰属は
  free VRAM で判定する。
- マシンは他セッションと共有。A/B は同一 GPU 上の paired 比較とし、可能な範囲で実行順を
  反転する。他プロセスと同時実行になった campaign は統合値から除外する。

### Gate 11J-H 結果

- Target 側: 同一 kernel の primitive が k geometry で 1.03-1.07x（neutral）→ 実装せず。
- DFlash2 側: cached backbone の k_noise pair を 2→1。primitive 1.49-1.60x、
  proposer backbone -0.08%（4/7）/ proposer -0.01%（3/7）、E2E median -0.01%（7/15）
  → **PERF-NEUTRAL**（flag default OFF で保持）。
- required A 137/137・B 138/138（skip 0）、CLI ids/acceptance 一致、gate11c failed=0。
- DFlash2 G/H はどちらも -1 launch/layer であり、proposer A/B（7 pairs、分解能 ~±0.15%）では
  利得を確定できない。cumulative baseline には入れない。

### Gate 11J-I / J / K の状況（本 session 時点）

Gate 11J-H までを実施。I / J / K は未実施であり、以下を記録しておく（次 session で継続可能）。

#### Gate 11J-I（DFlash2 conv boundaries）discovery

| boundary | launches/draft-forward | 隣接性 | candidate |
| --- | ---: | --- | --- |
| `dynamic_conv(phase 0) → activation quantize`（attention / mlp の 2 箇所） | 4 | exact（conv 出力 buffer がそのまま quantize 入力） | **fused conv + quantize**（本命） |
| `o_proj → dynamic_conv(phase 1) → residual_add` | 3 | exact | `conv(phase 1) + residual_add` は Gate 11J-C の `residual+rmsnorm+quantize` 窓を壊すため不可 |

- conv kernel は `blocks = ceil(rows*hidden/256)` の elementwise grid-stride。各行の出力は
  入力の row / row-1 のみに依存するため、`grid = rows` / `block = 256` の row 単位 kernel に
  置き換えても出力は同一（stencil は入力側のみ）。
- 設計: conv の要素式を `dflash2/detail/grouped_conv_device.h` へ抽出し、fused kernel は
  各 block が 1 row の conv を計算 → `__syncthreads()` → `QuantVecRow<5120,5120,256>` で
  codes/scales を生成（peak は max なので thread 数非依存）。`act_valid_k/rows` を更新。
- 実装対象: cached backbone の `attention_prepared` / `mlp_prepared` の 2 サイト
  （`*_quantized` variant を追加し、後続の `dflash2_quantize_a8` を skip する flag を持たせる）。
- 予想: -2 launch / layer = -10 / draft-forward。Gate 11J-C の -15/forward が proposer -0.64%
  だったことから **-0.4% 程度**を見込む。conv+quantize の primitive は 2→1。
- 本 session では未実装（call-site が 4 箇所に及び、未完のまま commit しない判断）。
  検証は Gate 11J-C/D と同じ手順（focused split-vs-fused byte-exact + proposer A/B + E2E）。

#### Gate 11J-J（INT2 coarse + TopN）基準値

Gate 11I 実測（`docs/rnd/dflash2/dflash2.md` の Gate 11I section）より:

| stage | cost |
| --- | ---: |
| INT2 coarse full vocab logits | ~0.98 ms |
| pool32 coarse Top-N | ~0.235 ms |
| 合計 | ~1.215 ms |

fused coarse-topN が ~1.0 ms 以下に近づくかを見る。correctness は vocabulary 248320、
pool 16/24/32/48/64/80/128 の全候補集合・順序が既存 kernel と完全一致すること。
WG-local top-L（L=32/48/64）→ partial merge → exact top-N の段階構成。

#### Gate 11J-K（PSQ8 rerank + Top16 / selector）基準値

| stage | cost |
| --- | ---: |
| pool32 PSQ8 rerank | ~71 us |
| rerank Top16 | ~49 us |

絶対値が小さいため、complexity が増えるなら PERF-NEUTRAL / REVERT も許容。

#### 最終 gate までの本 session の cumulative 結果

- Target Verify（M=8 ctx2048）: Gate 11J-D の baseline 40598 us → Gate 11J-F の B 39802 us
  = **-1.96%**（D -0.90% / E -0.76% / F -0.30% の積に一致）。
- Target physical launches（M=8、1 update）: 1353 → 1084 = **-269**
  （D -141 / E -96 / F -32。logical 1654 は不変）。
- DFlash2 proposer: Gate 11J-C の **-0.64%**（-15 launch/draft-forward）。G/H の DFlash2 側は
  PERF-NEUTRAL（cumulative baseline には入れない）。

#### Gate 11J-J 結果（REVERT）

fused int2 coarse + exact TopN を実装し、**byte-exact（204/204）** と scratch 0 を確認したが、
in-context で coarse が 1.028 → 2.210 ms（+115%）となり proposer が **+15.34%** regression したため
**REVERT**。詳細は `docs/rnd/dflash2/dflash2.md` の Gate 11J-J section。

- 残したもの: `detail/coarse_head_device.h` / `detail/coarse_topn_device.h` の共有 device helper と
  `launch_dflash2_coarse_topn_merge` の分割（挙動不変、standalone test 93/93・15/15）。
- 削除したもの: fused kernel / header / launcher / flag / test / counter。
- 学び: (1) full coarse logits の write+read は 6.95 MB × 2 ≒ 22 us（640 GB/s）で支配項ではなく、
  coarse 自体は DRAM-bound のまま。(2) 16-lane `shfl_down` は group 外 lane を読むため、
  reduction 結果を leader から broadcast しないと各 lane が自分の tuple で自己無効化する。
  (3) 1 warp / 1 tile の legacy coarse に対して、複数 tile を 1 CTA に束ねる構成は
  barrier + staging + 候補 list（VGPR 102）で occupancy が落ち、tile 単価が約 2x になる。
- 次 Gate（11J-K）への示唆: rerank + Top16 + remap は 1 候補あたりの作業が小さく
  （pool32 で 71 + 49 us）、coarse のような DRAM-bound の並列性低下を伴わないため、
  同じ fusion でも成立する可能性がある。

#### Gate 11J-K 結果（ADOPT）

rerank + Top16 + remap を 1 kernel family に融合。**exact（147/147）**、scratch 0（VGPR 42 /
SGPR 40 / LDS 528）、launch 4 → 1（-3 / draft forward）。

- in-context（3 pairs、ctx=2048）: rerank 0.079 → 0.088 ms（+11.4%）、top16 0.045 → 0.005 ms
  （-88.9%）、rerank+top16 **-25.0%**、full cached proposer **-1.90%**（3/3 paired 同符号）。
- in-context exactness: PSQ4 + INT2 head で `GENERATED_IDS` / rounds / accepted / reruns が A/B 一致。
- K2（CandidateSelector 融合）は **NO-GO**: selector は 1 block が draft row を順に処理する
  predecessor 依存 chain を持ち、row 並列の rerank block から連結するには grid 全体の完了依存が
  必要。合法な grid-wide barrier が無く、単一 block 化は rerank の row 並列性を失うため実装しない。
- 判定: ADOPT（K1 独立、K2 は report のみ）。

#### J / K 完了時点の cumulative（DeepFusion へ入る前の checkpoint）

| 項目 | 値 |
| --- | --- |
| Target Verify（M=8 ctx2048） | 40598 → 39802 us（**-1.96%**、Gate 11J-C～F） |
| Target physical launches（M=8） | 1353 → 1084（**-269**） |
| DFlash2 proposer（ctx=2048） | Gate 11J-C -0.64%、Gate 11J-J REVERT、Gate 11J-K **-1.90%** |
| 11J-K 後 proposer 絶対値 | ~5870 us |
| 11J-J | REVERT（性能 regression、production source 削除、refactor のみ残置） |
| 11J-K | ADOPT（K1）、K2 NO-GO |

#### DeepFusion 11K-A ～ 11K-E の結果（本 session）

- **11K-A DFlash2: ADOPT**。`grouped_dynamic_conv` の epilogue に residual を統合
  （flag `PHASESHIFT_DEEPFUSE_DFLASH2_MLP_A`、counter `dflash2_mlp_tail`）。36/36 bit exact、
  in-context で `GENERATED_IDS`/rounds/accepted/reruns 一致、proposer **-0.39%**（3 pairs）、
  launch -1/layer。Target 側（Down GEMM + Residual）は共有 PSQ GEMM core の epilogue 化が必要で
  未実装。
- **11K-B / 11K-C / 11K-D: NO-GO（未実装）**。Gate/Up/Down はいずれも
  `select_psq4_gemm_config` / `select_psq8_gemm_config` が選ぶ共有 GEMM であり、dual projection、
  SwiGLU epilogue、on-the-fly code 生成はいずれも GEMM core の trait 化を要求する。11J-J の実測
  （複数 tile を 1 CTA に束ねると occupancy 低下で tile 単価 ~2x）から regression リスクが高く、
  残 budget で exactness を保証した実装・検証（required PASS / skip 0 / E2E）を完了できないと
  判断した。11K-C の naive 1 kernel 化は grid-wide 同期が必須で禁止事項に抵触する。
- **11K-E**: 現構成を final topology として確定。logical operation の owner 重複 0
  （`docs/rnd/dflash2/dflash2.md` の `Active Fusion Ownership`）。

#### Final 実測値（本 session 終了時点）

| 項目 | 値 |
| --- | --- |
| Target Verify（M=8 ctx2048、final build） | **39658.14 us**（initial 40598 → **-2.32%**） |
| Target physical launches（M=8） | 1353 → **1084**（-269） |
| DFlash2 proposer（final、INT2 head pool32） | ~5837 us（C -0.64% / K -1.90% / 11K-A -0.39%） |
| 11J-K E2E（n=8 paired、A = C-F baseline / B = +K） | median **+0.78%**、ids_exact all True |
| required（final B build） | **139/139 PASS**、skip 0 |
| HIP_GRAPH=OFF（primary） | `gate11c_e2e` 3 case とも `ids_match=yes` / failed=0（全 exactness・acceptance・E2E はこの構成で検証） |
| HIP_GRAPH=ON | branch 単体では base が main の Gate HG（`38cfbe3e`、PR #40）より前のため `gate11c_e2e` が hist/ref 不一致。**main merge で修正を取り込み、merge 後は graph ON でも 3 case `ids_match=yes` / `failed=0`（rerun 数も OFF と一致）**。focused exact test も graph build で 147/147・36/36 PASS |
