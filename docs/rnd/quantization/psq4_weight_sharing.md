> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# PSQ4 weight sharing 実現性調査

Qwen3.8-27B-PSQ の PSQ4 weight に共有可能な冗長性があるかを実データで測る。
PSQ4-E（lossless permutation + entropy coding）が compression NO-GO だったため、
次に「重みの共有」を検討する。production format は変更しない。

---

## Revision / Environment

| 項目 | 値 |
| --- | --- |
| baseline SHA | `2c6130d9`（main） |
| worktree | `.worktrees/psq4-entropy-storage` |
| model | `models/Qwen3.8-27B-PSQ` |
| tool | `tools/rnd/psq4_weight_sharing.py` |
| GPU | 不使用（CPU / numpy、`rocm` 非依存） |

対象: `ffn_gate` / `ffn_up` / `ffn_down`（PSQ4 layer）。dequantize は
`kPsq4Magnitudes = {0,1,2,3,4,6,8,10}` + sign bit を使う。

---

## 1. Level 1 — lossless な完全一致共有（block 単位）

1 block = 32 weights = 16 byte codes + 2 byte scale。

| tensor | blocks | distinct code blocks | duplicate ratio | distinct scale values |
| --- | ---: | ---: | ---: | ---: |
| `layers.0.mlp.gate_proj` | 2,785,280 | **2,785,280** | **0.0** | 541 |

code block は 128 bit 空間で、2.78M block がすべて相異なる。
完全一致 block を辞書化する lossless 共有は **不可能**。

- 参考: 理論的にも 16^32 ≈ 3.4e38 に対し 2.78e6 block なので衝突期待値 ≈ 0。

---

## 2. Level 2 — near-duplicate block（delta 共有）

32-weight block 間の Hamming 距離（16 byte = 128 bit、sample 8,000 block）。

| 指標 | 値 |
| --- | ---: |
| 最小 | 31 bit |
| 中央値 | 42 bit |
| 平均 | 41.6 bit |

最良の隣接 block でも 128 bit 中 31 bit 異なる。
「近い block + 小さい差分」で共有する余地は無い（分布も 31〜50 bit に集中）。

---

## 3. Level 3 — row / 部分空間の共有

### 3.1 row レベルの類似（gate layer 0、3,000 row sample）

| 指標 | 値 |
| --- | ---: |
| 他 row への最大 cos の最小 | 0.0605 |
| 同 中央値 | 0.0978 |
| 同 p90 | 0.1856 |

row は相互にほぼ無相関。近並行な row 対は存在せず、共有 row + 差分の余地は無い。

### 3.2 cross-layer row 類似（layer 8 の row vs layer 0 の row）

| 指標 | 値 |
| --- | ---: |
| 最大 cos の最小 | 0.0381 |
| 同 中央値 | 0.0508 |
| 同 p90 | 0.0583 |

layer 間で row を共有する余地は無い。

### 3.3 cross-layer 部分空間（rank 256 基底への射影）

layer 0 の row space の top-256 基底に、他 layer の weight を射影したときの
エネルギー捕捉率:

| 対象 layer | captured ratio |
| --- | ---: |
| 8 | 0.0196 |
| 16 | 0.0172 |
| 32 | 0.0176 |

約 2%。layer 間で共通基底を共有する設計は成立しない。

---

## 4. Level 4 — 低ランク共有（spectral）

`ffn_gate` layer 0（17408×5120）の特異値エネルギー捕捉率:

| rank | energy | rank | energy |
| ---: | ---: | ---: | ---: |
| 16 | 0.0308 | 128 | 0.1546 |
| 32 | 0.0517 | 256 | 0.2618 |
| 64 | 0.0892 | 512 | 0.4115 |

top-512（K 次元 5120 の 10%）でも 41% しか捕捉できない。
低ランク分解で共有（基底 + 係数）するには rank をほぼ全次元に近づける必要があり、
10 倍圧縮のような共有は成立しない。

---

## 5. Level 5 — 既存の構造的共有 / tying

| 項目 | 値 | 共有 |
| --- | --- | --- |
| `num_attention_heads` / `num_key_value_heads` | 24 / 4 | K/V head を 6:1 で既に共有 |
| `tie_word_embeddings` | false | — |
| `embed_tokens` vs `lm_head` の実体 | codes/metadata1 の sha256 が不一致 | tie されていない |
| DFlash2 drafter | target の embed_tokens / lm_head を参照 | 既に共有済み |

---

## 6. 副次的な発見（共有ではないが lossless で有効）

scale stream（BF16、2 byte / 32 weights = 0.5 bpw）は冗長である。

`ffn_gate` layer 0:

| 項目 | 値 |
| --- | ---: |
| distinct scale 値 | 541 |
| scale 16-bit 値の H0 | 7.394 bit |
| scale 換算 | **0.2310 bpw**（raw 0.5 bpw） |
| 内訳: 下位 byte H | 7.371 bit |
| 内訳: 上位 byte H | 0.760 bit（指数部がほぼ一定） |
| 上位 scale 値の占有率 | 1.57%, 1.44%, 1.43%, ... |

scale を entropy coding すると **約 0.27 bpw（raw 4.5 bpw の約 5%）** を
**追加の量子化誤差なしに** 削れる。code stream（H0=3.89 bpw、削減 0.1 bpw）より
削減余地が大きい。ただし 25〜35% には届かない。

---

## 7. 結論

```
NO-GO（weight sharing）
```

測った全経路で共有可能な冗長性は存在しない:

| 経路 | 結果 |
| --- | --- |
| 完全一致 block 共有（lossless） | 重複 0 / 2.78M |
| near-duplicate block delta 共有 | 最良 NN でも 31/128 bit 相違 |
| row 共有（intra-layer） | 最大 cos ≤ 0.19 |
| row 共有（cross-layer） | 最大 cos ≤ 0.06 |
| cross-layer 共通基底（rank 256） | 捕捉 ≈ 2% |
| 低ランク共有 | top-512 で 41% |
| embedding tying | tie なし |

PSQ4（4-bit codebook + per-32 bf16 scale）は既に「共有可能な冗長性を
ほぼ残さない」設計であり、code/scale/row のいずれにも共有構造が無い。

### 代わりに価値がある方向

1. **scale stream の lossless entropy coding**（約 0.27 bpw、~5% 削減、
   量子化誤差ゼロ）。code stream より優先度が高い。
2. 0.5 bpw の scale 自体の再設計（per-32 を粗くする等）は量子化誤差を
   増やすため、本 PoC の制約外。
3. MoE 化（`Qwen3.8 Flash Next`）の expert sharing は本 dense モデルの
   計測対象外。将来そちらで再評価する。

---

## 8. 再現コマンド

```
python3 tools/rnd/psq4_weight_sharing.py \
  --model-dir models/Qwen3.8-27B-PSQ --role ffn_gate --layers 0 \
  --cross-layers 0,8,16,32 --out /tmp/ws_gate0.json
```

scale entropy / tying の確認は同 doc 記載の手順（safetensors を直接読む）。
