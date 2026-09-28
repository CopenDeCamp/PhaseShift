> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# PSQ4 W32 Hadamard KV Cache PoC

既存 PSQ4 W32 KV cache（4bit CB10 + BF16 scale/W32 + LSQ1 = 4.5 bpw、`docs` 外の PoC で成立済み）
に対し、W32 単位の normalized Hadamard rotation（plain H32）を追加し、W32 内 outlier の分散で

- PSQ4 K/V reconstruction error
- block p95 / p99
- logit rel_l2 / KLD
- top1 / top16
- greedy generation

が改善するかを検証した。KV 容量は 1152 B/token/layer のまま変えない。

本 PoC は **コードを revert 済み**であり、結果のみをここに記録する（後述 Revision の
`experiment commit` を参照）。production format / loader / GEMM / manifest は変更していない。

---

## Revision

| 項目 | 値 |
| --- | --- |
| baseline | `f0f84290`（`poc/psq4-kv-cache`、PSQ4 W32 KV cache 成立版） |
| final | `PENDING` |
| branch | `poc/psq4-kv-cache` |
| worktree | `.worktrees/poc-psq4-kv` |
| experiment commit | `8b0ce0fefb9f93161e32ac6810f817326510e4f0`（revert 済み。`git show`/`cherry-pick` で復元可能） |

Hadamard 実装・テスト・比較 tool は experiment commit に含まれる。本レポートのみを
baseline に積む。追加したもの:

- `include/phaseshift/quantization/psq/hadamard32.h`
- `Psq4KvRotation`（`kv_cache_types.h`）
- append / decode / prefill / correctness の rotation 対応
- `tests/unit/test_psq4_hadamard.hip`
- bench `--kv-psq4-rotation`、CLI `--kv-psq4-rotation`
- `tools/rnd/analyze_psq4_kv.py`（rotation 対応）、比較/集計 tool 群

---

## Environment

| 項目 | 値 |
| --- | --- |
| GPU | AMD Radeon AI PRO R9700（gfx1201） |
| ROCm / HIP | `rocm-sdk` 同梱 clang（`_rocm_sdk_devel`）、gfx1201 |
| model | `models/Qwen3.5-4B`（8 full-attention + 24 GDN、q_heads=16、kv_heads=4、head_dim=256、page_tokens=16） |
| build | Release、gfx1201 |
| 対象 | KV cache（K/V）のみ。weight PSQ4 は対象外 |

---

## Format / Rotation

```
block   : W32
code    : 4bit CB10 { 0, ±1, ±2, ±3, ±4, ±6, ±8, ±10 }
scale   : BF16 / W32
shape   : none
rotation: normalized H32（block diagonal: diag(H32, ..., H32)） 1 head = 8 block
bpw     : 4.5（rotation で不変）
memory  : 1152 B/token/layer（rotation で不変）
```

H32 は `H32 * H32^T = I`、`H32^-1 = H32^T = H32`。各 W32 に独立適用する。

配置:

| 対象 | 位置 | 回転回数 |
| --- | --- | --- |
| K | post-RoPE, pre-PSQ4（append 内） | rows × kv_heads × head_dim |
| Q | post-RoPE, pre-dot（decode は in-register） | rows × q_heads × head_dim |
| V | pre-PSQ4（append 内） | rows × kv_heads × head_dim |
| V inverse | post-attention（decode は store 直前、prefill は shared 上） | rows × q_heads × head_dim |

attention score は `Q·H(K) = H(Q)·K`、V は `(A·Vrot)` を最後に H で戻す。行列は対称なので
inverse も同じ helper で行う。training / calibration はなし。

---

## Gate 1: H32 primitive

- involution: 8 入力パターン（random / all-zero / one-hot / constant / alternating / outlier /
  positive / negative）で `H32(H32(x))` の rel_l2 ≤ 1e-6。GPU shfl 版が host array 版と一致。
- dot-product invariance: GPU で同一 reduction の `dot(q,k)` と `dot(H32(q),H32(k))` を比較。
  さらに host double で 256 dim = 8×H32 の不変性を確認。
- **PASS**

実装上の注意: 素朴な ascending-stride butterfly で bit-set lane に `x - partner` を入れると、
直交だが対称でない行列になり involution にならない。正しくは `partner - x`。テストで検出した。

---

## Gate 2: Offline reconstruction（holdout 55,194 W32 block/side、original BF16 domain）

### K

| method | MSE | rel_l2 | cosine | p95 | p99 | peak/RMS mean |
| --- | --- | --- | --- | --- | --- | --- |
| PSQ4 none | 0.021344 | 0.0983 | 0.99514 | 0.045886 | 0.068358 | 2.821 |
| PSQ4 H32 | 0.017876 | 0.0900 | 0.99594 | 0.034687 | 0.047426 | 2.319 |

MSE ×0.8375、p99 ×0.6938、peak/RMS mean −17.8%。

### V

| method | MSE | rel_l2 | cosine | p95 | p99 | peak/RMS mean |
| --- | --- | --- | --- | --- | --- | --- |
| PSQ4 none | 0.022214 | 0.1039 | 0.99458 | 0.106148 | 0.213754 | 2.847 |
| PSQ4 H32 | 0.016255 | 0.0889 | 0.99604 | 0.078976 | 0.150249 | 2.314 |

MSE ×0.7317、p99 ×0.7029、peak/RMS mean −18.7%。**V p99 は −29.7%**。

extreme code（magnitude 10）率: K 0.0434 → 0.0518、V 0.0438 → 0.0521。回転で分布が
Gaussian 化し最大 code が使われやすくなるため微増するが、MSE/p99 は改善する。

---

## Gate 3–8: Kernel correctness

| 対象 | 結果 |
| --- | --- |
| append none / K / V / KV | PASS（`test_kv_append` 105/105、CPU reference と code/scale/reconstruction 一致） |
| decode K / V / KV | PASS（`test_paged_attention` 33/33、CPU PSQ4 rotated reference と rel ≤ 2e-3） |
| prefill K / V / KV | PASS（rel ≤ 1e-2） |
| correctness fallback | PASS（実モデル correctness mode で none/K/V/KV が BF16 と同一 greedy 4 token） |

修正した不具合:

- per-lane H32 の符号規約（Gate 1）。
- prefill の shared H32 で fill と H32 の間に barrier がなく race（ctx2048 で NaN → greedy
  失敗）。`__syncthreads()` を追加。追加 barrier は「回転ありの PSQ4」に限定し、他経路は
  1 本のまま。

---

## Gate 9–11: E2E quality

### Canonical contexts（BF16 reference、rel_l2 / KLD / top16 / gen exact）

| ctx | mode | rel_l2 | KLD | top16 | gen |
| --- | --- | --- | --- | --- | --- |
| 128 | FP8 | 0.0208 | 2.50e-4 | 1.000 | 9 |
| 128 | none | 0.0845 | 9.04e-4 | 1.000 | 21 |
| 128 | K | 0.1066 | 2.31e-3 | 1.000 | 6 |
| 128 | V | 0.0774 | 1.03e-3 | 1.000 | 32 |
| 128 | KV | 0.0540 | 4.24e-4 | 1.000 | 32 |
| 512 | FP8 | 0.0395 | 3.15e-3 | 1.000 | 32 |
| 512 | none | 0.0900 | 8.66e-3 | 1.000 | 32 |
| 512 | K | 0.1078 | 2.85e-3 | 1.000 | 32 |
| 512 | V | 0.0880 | 1.11e-2 | 0.9375 | 32 |
| 512 | KV | 0.0865 | 2.92e-3 | 1.000 | 32 |
| 2048 | FP8 | 0.0192 | 1.92e-5 | 1.000 | 32 |
| 2048 | none | 0.0561 | 1.76e-4 | 0.9375 | 32 |
| 2048 | K | 0.0473 | 1.02e-4 | 0.9375 | 32 |
| 2048 | V | 0.0381 | 2.56e-4 | 0.9375 | 32 |
| 2048 | KV | 0.0398 | 5.0e-5 | 0.9375 | 32 |
| 8192 | FP8 | 0.0165 | 4.28e-6 | 0.9375 | 4 |
| 8192 | none | 0.0481 | 4.83e-5 | 0.9375 | 4 |
| 8192 | K | 0.0445 | 4.9e-5 | 0.9375 | 4 |
| 8192 | V | 0.0320 | 1.2e-5 | 1.000 | 4 |
| 8192 | KV | 0.0379 | 7.0e-6 | 0.9375 | 4 |

canonical の top1 は全 mode 一致（FP8 含む）。

### Additional prompts（32 prompt、112–1536 token、BF16 reference）

| mode | top1 | top16 mean | top16 min | rel_l2 mean | rel_l2 max | KLD mean | KLD max | gen exact /32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP8 | 0.969 | 0.9863 | 0.9375 | 0.0202 | 0.0317 | 0.00038 | 0.00197 | 25.2 |
| PSQ4 none | 0.938 | 0.9453 | 0.8125 | 0.0659 | 0.1352 | 0.00386 | 0.02084 | 18.8 |
| PSQ4 K | 0.938 | 0.9492 | 0.8750 | 0.0612 | 0.1068 | 0.00318 | 0.01885 | 22.2 |
| PSQ4 V | 0.938 | 0.9512 | 0.7500 | 0.0637 | 0.1694 | 0.00384 | 0.01401 | 21.3 |
| PSQ4 KV | 0.938 | 0.9590 | 0.8750 | 0.0578 | 0.1747 | 0.00326 | 0.02615 | 19.8 |

**top1 は 32 prompt で PSQ4 全 mode が 0.938 で同一**（none と変わらない）。12 prompt 時点では
KV が 1.000 だったが、20 prompt 追加で 0.900 に落ち、少数標本の見かけだったことが判明した。

### Generation（canonical exact match 128/512/2048/8192）

- FP8: 9 / 32 / 32 / 4
- PSQ4 none: 21 / 32 / 32 / 4
- PSQ4 K: 6 / 32 / 32 / 4（ctx128 で悪化）
- PSQ4 V: 32 / 32 / 32 / 4
- PSQ4 KV: 32 / 32 / 32 / 4

---

## Gate 12–14: Memory / Performance

memory: BF16 4096 / FP8 2080 / PSQ4 none 1152 / PSQ4 H32 1152 B/token/layer。
**rotation による追加メモリは 0**（cache レイアウト・bpw 不変）。

### Append（p50 us）

| rows | FP8 | none | K | V | KV |
| --- | --- | --- | --- | --- | --- |
| 1 | 5.9 | 5.5 | 5.5 | 5.5 | 5.7 |
| 16 | 5.8 | 5.6 | 5.6 | 5.7 | 5.7 |
| 128 | 6.2 | 23.4 | 23.8 | 23.9 | 24.3 |
| 512 | 24.6 | 22.5 | 24.4 | 24.4 | 26.0 |
| 2048 | 29.3 | 81.2 | 87.0 | 86.9 | 92.5 |

H32 cost: +7%（K/V）、+14%（KV、rows=2048）で絶対値 ~11us 以内。

### Decode qh16（p50 us）

| visible | BF16 | FP8 | none | K | V | KV |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 11.50 | 25.82 | 16.16 | 16.43 | 16.21 | 16.47 |
| 512 | 30.60 | 84.35 | 47.40 | 48.90 | 48.44 | 45.89 |
| 2048 | 106.18 | 289.85 | 156.89 | 158.41 | 159.01 | 159.01 |
| 8192 | 364.79 | 1376.37 | 624.12 | 629.70 | 632.86 | 631.14 |
| 32768 | 2065.38 | 5389.62 | 2698.69 | 2716.44 | 2715.35 | 2719.66 |

decode の H32 cost は **<1%** で context 非依存（Q を 1 回、出力を 1 回のみ）。FP8 に
対する約 2× の速度優位は全 mode で維持。

### Prefill attention kernel（rows × vis=4096、p50 us）

| rows | BF16 | none | K | V | KV |
| --- | --- | --- | --- | --- | --- |
| 128 | 351.66 | 598.19 | 1507.71 | 1550.28 | 2045.52 |
| 512 | 1045.34 | 1740.34 | 3791.49 | 3858.63 | 5160.78 |
| 2048 | 4379.95 | 6452.96 | 14373.10 | 14651.03 | 19566.74 |

**回転ありは ×2.2–3.0**。原因は「shared に dequant → shared 上で inverse H32 → WMMA」に
するため、ページごとに barrier が 1 本増え、shared round-trip が入ること。E2E の
`PREFILL_MS` では該当が 8/32 layer のみのため影響は小さく、

| ctx | BF16 | FP8 | none | K | V | KV |
| --- | --- | --- | --- | --- | --- | --- |
| 2048 | 789.0 | 804.7 | 795.3 | 813.3 | 817.2 | 825.6 |
| 8192 | 785.4 | 801.3 | 792.0 | 814.4 | 810.5 | 831.3 |

で +2.3〜3.8%（8192 KV で +5%）に留まる。

---

## Acceptance

| 項目 | 判定 |
| --- | --- |
| FORMAT | **GO**（容量・bpw・layout 不変、HF32 は追加メモリ 0） |
| HADAMARD QUALITY | **GO（条件付き / weak）** |
| BEST MODE | **V** |
| KERNEL PERFORMANCE | **NEEDS OPTIMIZATION**（decode/append は GO。prefill attention が ×2.2–3.0） |

---

## Conclusion

1. **W32 内 outlier は均されたか**: 均された。block peak/RMS mean は K 2.82→2.32、V 2.85→2.31。
   extreme code 率は微増するが MSE/p99 は改善。
2. **K reconstruction**: 改善。MSE −16.3%、p99 −30.6%。
3. **V reconstruction（p95/p99）**: 改善。MSE −26.8%、p95 0.106→0.079、p99 0.214→0.150（−29.7%）。
4. **E2E logit への波及**: 部分的。canonical は V/KV が全 context で rel_l2 を改善。K は長 context
   で改善するが short（128/512）で悪化。32 prompt の rel_l2 mean は KV 0.0578 / K 0.0612 /
   V 0.0637 と none 0.0659 を改善。
5. **K/V/KV どれが効果的か**: 指標依存。canonical は V/KV が全 context 改善 + generation 改善
  （ctx128 21→32）。extra 32 prompt では K が worst-case 最良（rel_l2 max 0.1068、KLD mean
  0.00318、gen 22.2）、V が KLD tail 最良（0.0140）、KV が mean 最良（rel 0.0578、top16 0.9590）
  だが worst-case 最悪（rel max 0.1747、KLD max 0.0262）。
6. **rel_l2 max ≤ 0.08**: 未達。canonical 最良 KV 0.0865、extra 最良 K 0.1068（none 0.1352）。
7. **additional prompt top1**: 改善せず（32 prompt で none = K = V = KV = 0.938、FP8 0.969）。
8. **generation divergence**: 改善。canonical で V/KV が ctx128 を 21→32、extra 平均 exact は
   none 18.8 → K 22.2 / V 21.3 / KV 19.8。
9. **1152 B/token/layer**: 維持。追加メモリ 0。
10. **decode の FP8 優位**: 維持。32768 で FP8 5330–5390us に対し PSQ4 2713–2720us（約 2×）。
11. **継続価値**: ある。再構成と canonical 品質は明確に改善し、容量・decode/append 速度を損なわない。
    ただし top1 の頑健改善と rel_l2 ≤ 0.08 は未達で、gain は「平均・tail の改善」に留まる。
12. **非対称 KV へ進むべきか**: 本 PoC は NO-GO ではないが、rel_l2 ≤ 0.08 と top1 改善が未達のため、
    次の一手は H32 の延長ではなく `K=PSQ4 / V=FP8`（または `K=FP8 / V=PSQ4`）の非対称 KV を推奨。
    V tail が支配的という前 PoC の観測と、V-only が最も consistent だった本結果は整合する。

**推奨 mode は V-only**（canonical 全 context 改善、canonical generation 満点、KLD tail 最良、
cost は 1 transform で KV の半分）。次点 K、KV は mean/top16 最良だが worst-case が悪化。

---

## 未解決 / 次にやること

### prefill 最適化（入手余地は大きい）

現状の prefill はページごとにキャッシュを逆回転しており、回転量が `O(rows × visible)`。
数学的には `Q·H(K) = H(Q)·K` なので、

- K: Q をタイルごとに 1 回回す（`O(rows)`、visible 非依存）
- V: 出力 accumulator を最後に 1 回逆回転（`O(rows)`）

に置き換えれば、**×2.2–3.0 → ほぼ ×1.0** が狙える。barrier も追加不要。RoPE 後の natural
layout で Q を回し、decode/prefill で規約を一本化するのが clean。

妥協案として、psq4 prefill の chunk fill が **1 warp = 1 token** である性質を使い、shared に
書く前にレジスタで H32（stride 1/2/4 は lane 内、stride 8/16 のみ 4 lane 間 `shfl_xor`）すれば、
barrier と shared round-trip は消せる（計算量は残るため戻りは ×1.3–1.6 程度）。

### 他セッションとの整合

本変更は `attention/paged_prefill.hip`、`paged_decode.hip`、`kv_append.hip` と launch 署名、
`ModelDispatchStateView` / `ExecutorConfig` に触れる。既定 `--kv-psq4-rotation none` では
ランタイム挙動は baseline と同一。kernel 融合と衝突するため、回転の prefill 実装は融合側で
書き換えた後に「Q 1 回 / 出力 1 回」方式で載せ直すのが安全。

---

## 参照

- 実験実装: commit `8b0ce0fefb9f93161e32ac6810f817326510e4f0`（revert 済み）
- 前段 PSQ4 W32 KV cache: PSQ4 W32 KV Cache PoC（4.5 bpw、training-free）
- 実測データ: `/tmp/psq4-h32/`（canonical / 32 prompt logits、summaries、perf log）、
  `/tmp/psq4-eval/psq4_rotation_analysis.json`
