> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# PSQ4-VQ4 成立性 PoC（FFN / Qwen3.8-27B-PSQ）

Qwen3.8-27B-PSQ の既存 PSQ4（CB10 16 値 + per-32 BF16 scale）を維持したまま、
K 方向に連続する 4 weight を 1 つの vector symbol として代表 pattern へ丸め、
256 / 1024 / 4096 pattern で 2.5 / 3.0 / 3.5 bpw を成立させられるかを CPU のみで検証した。

本 PoC は **lossy** である。前回の PSQ4-E（lossless permutation + entropy coding、
`docs/rnd/quantization/psq4_entropy_poc.md`）とは目的も手法も独立している。production format /
loader / GEMM / manifest schema は一切変更していない。

---

## Revision

| 項目 | 値 |
| --- | --- |
| baseline | `ef98918c38cf7def1b63bddb845801280476beb5`（main） |
| final | `PENDING` |
| branch | `poc/psq4-vq4` |
| worktree | `.worktrees/psq4-vq4` |

---

## Environment

| 項目 | 値 |
| --- | --- |
| CPU | AMD Ryzen 7 7800X3D 8-Core（16 thread） |
| GPU | AMD Radeon AI PRO R9700（gfx1201） |
| ROCm / HIP | 7.15.26333（`_rocm_sdk_devel` 10.0.0） |
| model | `models/Qwen3.8-27B-PSQ`（`phaseshift_quantization.json` を source of truth） |
| baseline PSQ model | `models/Qwen3.8-27B-PSQ` |
| build | Release、gfx1201、`PHASESHIFT_BUILD_BENCHMARKS=OFF`、`PHASESHIFT_BUILD_TESTS=ON` |
| 実装 | C++20。GPU kernel は追加していない |

---

## Existing PSQ4

```text
CB10 code values : { 0, ±1, ±2, ±3, ±4, ±6, ±8, ±10 }   (16 codes)
block            : 32 weights
codes            : 16 bytes / 32 weights (4.0 bpw)
metadata1        :  2 bytes / 32 weights (BF16 scale, 0.5 bpw)
total            : 18 bytes / 32 weights = 4.5 bpw
```

canonical file の nibble 順は「byte `j` の low = block 内 `j`、high = block 内 `16+j`」
（`quantize_psq4_row` / `dequantize_psq4_row` と一致）。tuple はこの logical 順で
`[0..3] [4..7] ... [28..31]` の 8 個に分割し、block 境界は跨がない。

対象は FFN の PSQ4 tensor のみ（`ffn_gate` / `ffn_up` / PSQ4 の `ffn_down`）。
PSQ8 の `ffn_down`（64 layer 中 16 layer）は対象外。176 tensor、15,686,696,960 weight。

---

## Tuple statistics（Gate V0）

27B FFN 全 PSQ4 から tuple histogram を集計（再量子化なし）。

| 指標 | 値 |
| --- | ---: |
| total tuples | 3,921,674,240 |
| distinct tuples | **65,536 / 65,536（全 pattern が出現）** |
| top16 coverage | 0.145% |
| top64 coverage | 0.523% |
| top256 coverage | 1.851% |
| top1024 coverage | 6.287% |
| top4096 coverage | 19.977% |

role 別もほぼ同一（`ffn_down`/`ffn_gate`/`ffn_up` の top1024 は 6.36 / 6.24 / 6.28%）。
最頻 pattern でも出現率 0.1% 未満で、**頻度辞書（Top-K）による近似は成立しない**。
したがって 256 / 1024 / 4096 の意味は頻度ではなく、**value 空間の代表 pattern への
クラスタリング（Discrete Lloyd）**としてのみ成立する。

---

## Local reconstruction（Gate V1）

objective は `psq-delta`（既存 PSQ4 再構成値を target とし、VQ 化で追加される誤差のみ）。

`additional SSE = Σ_t W_t · Σ_i (v(t_i) − v(p_i))²`、`W_t = Σ s²`。
`relative MSE = additional SSE / Σ s²·Σ_i v(t_i)²`（PSQ4 再構成信号エネルギー比）。

| entries | scope | objective | additional SSE | relative MSE | codebook bytes | effective bpw |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 256 | global | psq-delta | 258,298.4 | 12.712% | 512 | 2.5000 |
| 256 | role | psq-delta | 260,075.7 | 12.799% | 1,536 | 2.5000 |
| 1024 | global | psq-delta | 143,405.0 | 7.058% | 2,048 | 3.0000 |
| 1024 | role | psq-delta | 141,943.7 | 6.985% | 6,144 | 3.0000 |
| 4096 | global | psq-delta | 103,169.6 | 5.077% | 8,192 | 3.5000 |
| 4096 | role | psq-delta | 103,143.2 | 5.076% | 24,576 | 3.5000 |
| 1024 | layer-role | psq-delta | 145,438.7 | 7.152% | 360,448 | 3.0002 |

`effective bpw` は code index（8 index × 8/10/12 bit = 8/10/12 byte per 32 weight）
+ scale 2 byte + codebook metadata を、対象 weight 数で割った値。理論値
2.5 / 3.0 / 3.5 bpw と一致する（codebook metadata 込み）。

role-1024 の誤差分布:

| role | relative MSE | p50 | p90 | p99 | p99.9 | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ffn_down | 6.93% | 0.00134 | 0.00500 | 0.00903 | 0.01312 | 0.165 |
| ffn_gate | 6.99% | 0.00160 | 0.00504 | 0.00928 | 0.01367 | 0.103 |
| ffn_up | 7.03% | 0.00163 | 0.00504 | 0.00925 | 0.01361 | 0.087 |

BF16 MSE ratio（`BF16→VQ4 MSE / BF16→PSQ4 MSE`）は **N/A**。ローカルに 27B の
BF16 元モデルが存在しないため測定していない（`--bf16-model` 未指定）。

所見:

- codebook を 256 → 4096 に 16 倍しても relative MSE は 12.8% → 5.1%（2.5 倍改善）
  止まりで、明確な収穫逓減。4 weight × 16 値の pattern 空間（65,536）を 4,096 点に
  落とすこと自体が誤差の支配要因であり、codebook 容量では解消しない。
- global と role はほぼ同一（1024 で 7.058% vs 6.985%）。role 分離の利得は 1% 未満。
- layer-role（oracle、176 個の個別 codebook、metadata 360 KB）は **7.152% と逆に悪化**。
  1 group あたりの標本が減るためで、codebook を細分化しても精度は上がらない。

---

## KLD（Gate V3）

**未実施。** `phaseshift-quantizer kld` は BF16 元モデルを必須とするが、本環境に
27B BF16 モデルが存在しない。BF16 参照が用意でき次第、baseline PSQ と
shadow bundle を同一 corpus / window / stride で比較する。ここでは PPL を代理指標と
する（同一 token、同一 max-tokens）。

---

## PPL（Gate V4）

`phaseshift-quantizer ppl`（token 2,049、`--max-tokens 2048`、docs / code 系 corpus を
27B tokenizer で符号化）。shadow bundle はすべて通常の PSQ4（4.5 bpw）で、
code のみ VQ 制約へ投影したもの。

| variant | VQ storage bpw（仮想） | mean_nll | perplexity | PPL 劣化 |
| --- | ---: | ---: | ---: | ---: |
| baseline PSQ4 | 4.5 | 1.638943 | 5.149722 | — |
| VQ4-4096 role | 3.5 | 1.799165 | 6.044599 | **+17.38%** |
| VQ4-1024 global | 3.0 | 1.870751 | 6.493168 | **+26.09%** |
| VQ4-1024 role | 3.0 | 1.894722 | 6.650702 | **+29.15%** |
| VQ4-256 role | 2.5 | 2.220305 | 9.210143 | **+78.85%** |

PPL 劣化は VQ の additional relative MSE と単調に対応する
（5.1% → +17%、7.0% → +26〜29%、12.8% → +79%）。

所見:

- **4096 / 3.5 bpw でも PPL +17.4%**。本 PoC の最低限の sanity check に既に失敗している。
- global は role よりわずかに良好（+26.1% vs +29.1%）。codebook の role 分離は
  PPL でも利得がない。
- layer-role oracle は local reconstruction で悪化するため PPL は実施していない。

---

## Layer outliers（Gate §33）

role-1024 codebook を layer 別 histogram に適用した relative MSE:

| 順位 | layer | relative MSE |
| ---: | ---: | ---: |
| 1（最悪） | 0 | 0.07034 |
| 2 | 12 | 0.07028 |
| 3 | 8 | 0.07026 |
| 4 | 4 | 0.07024 |
| 5 | 16 | 0.07024 |
| 6 | 20 | 0.07022 |
| 7 | 56 | 0.07019 |
| 8 | 60 | 0.07012 |
| 9 | 40 | 0.07008 |
| 10 | 24 | 0.07008 |

最良は layer 34 の 0.06938。**layer 間のばらつきは 1% 未満**で、極端に敏感な
single layer は存在しない。「sensitive layer だけ PSQ4、残り VQ4」という mixed policy
に根拠はない（候補としても報告しない）。

---

## 実装と再現

RnD executable（`cmake/rnd.cmake`、product entrypoint ではない）:

```text
tools/rnd/psq4_vq/psq4_vq.{h,cpp}    tuple / pack / codebook / projection（CPU のみ）
tools/rnd/psq4_vq/main.cpp           analyze / train / project / report
tools/rnd/psq4_vq/summarize.py       集計
tools/rnd/psq4_vq/make_corpus.py     PSKLDTOK corpus 生成
```

再現コマンド:

```text
BUILD=.worktrees/psq4-vq4/build
TOOL=$BUILD/phaseshift-psq4-vq-poc
M=/path/to/Qwen3.8-27B-PSQ

$TOOL analyze --psq-model $M --output /tmp/vq/analyze
$TOOL train --histograms /tmp/vq/analyze/histograms.bin \
    --entries 1024 --scope role --objective psq-delta --seed 1 \
    --psq-model $M --output /tmp/vq/train-role-1024
$TOOL project --psq-model $M \
    --codebook /tmp/vq/train-role-1024/codebook-role-1024.cbk \
    --output /tmp/vq/shadow-role-1024
phaseshift-quantizer verify /tmp/vq/shadow-role-1024
phaseshift-quantizer ppl --model-dir /tmp/vq/shadow-role-1024 \
    --tokens /tmp/vq/eval.pskldtok --max-tokens 2048
```

codebook training は 65,536 pattern のみを対象とした Discrete Lloyd（最大 20 iter、
assignment 不変または相対 loss 改善 < 1e-6 で終了）。update は 4 coordinate を独立に
16 code で総当たりし、代表 pattern は常に PSQ4 の 16 値の組み合わせに留まる。
duplicate entry は決定論的に除去し、reconstruction loss 最大の tuple で補充する。
seed 固定で完全に再現可能（同一 seed で同一 code / 同一 loss 列）。

shadow bundle は baseline PSQ を別 directory へ byte copy し、対象 `codes` の
byte のみを in-place 書換える。metadata1（scale）/ PSQ8 / BF16 / その他 asset は
byte-exact。manifest は `crc32` のみを原文へ外科的に置換する。

### 検証

- `phaseshift-quantizer verify` → 27B shadow は `Validation OK: 866 logical tensors resolved`。
- 追加 unit test（`cpu;required`）4 本すべて PASS:
  `test_psq4_vq_tuple` / `test_psq4_vq_pack` / `test_psq4_vq_codebook` /
  `test_psq4_vq_projection`（K=256/1024/4096 の index pack/unpack、tuple encode/decode、
  codebook 代表値が 0..15、padded K、deterministic training、duplicate 処理、
  all-zero / all-one-pattern / random 65536、projection の byte 整合）。
- 4B-PSQ で `analyze → train → project → verify` を end-to-end 検証。対象外 tensor が
  byte-exact であること、変更 tensor が対象 `codes` のみであることを byte 比較で確認。
- required acceptance: `PENDING`

---

## Conclusion

```text
NO-GO
```

理由（数値）:

| 判定軸 | 結果 |
| --- | --- |
| storage（3.0 bpw 成立） | 成立（codebook metadata 込み 3.0000 bpw） |
| 1024 / 3.0 bpw の精度 | PPL +29.15%（baseline 5.150 → 6.651）、additional relMSE 6.99% |
| 4096 / 3.5 bpw の精度 | PPL +17.38%。§31 の最低限 sanity check に失敗 |
| 256 / 2.5 bpw の精度 | PPL +78.85%。可能性なし |
| global vs role | global で十分。role は local/PPL とも利得なし |
| layer-role oracle | local reconstruction が逆に悪化（7.15%）。細分化は無意味 |
| layer outlier | ばらつき 1% 未満で mixed policy の根拠なし |

4 weight × 16 値の pattern 空間を 256 / 1024 / 4096 点へ落とす誤差は、codebook 容量や
scope の細分化では解消できない。最も有利な 4096 / 3.5 bpw でも PPL が +17% 悪化し、
本 PoC の精度ゲート（1024 で baseline 近傍）には遠く届かない。同一 weight 予算なら
既存 PSQ4（4.5 bpw）を維持する方が明確に優れる。

### この PoC が答える質問（§38）

1. **PSQ4 の 16 値を維持したまま 3.0 bpw へ落とせるか** → storage は可能。精度は不可。
2. **1024 pattern の既存 PSQ4 比の劣化** → PPL +29.15%、additional relMSE 6.99%（KLD 未実施）。
3. **global で十分か、role が必要か** → global で十分。role / layer-role の利得はない。
4. **4096 ですら精度が落ちるか** → 落ちる（PPL +17.4%）。
5. **256 / 2.5 bpw に可能性があるか** → ない（PPL +78.9%）。
6. **将来 GPU kernel を作る価値があるか** → 現時点でない。GPU 実装の前提条件
   「少なくとも 1024-entry / 約 3.0 bpw が十分な精度を示す」を満たさない。

### 未実施（明示）

- KLD（BF16 元モデル不在）。
- `objective = bf16-fixed-scale`（§20）。BF16 元モデルが無く、`psq-delta` の結果が
  明確に NO-GO のため試行していない。
- iMatrix-aware objective（§21）。
- GPU VQ decoder / VQ GEMM / M=1 benchmark（§35）。精度不成立のため進めない。
