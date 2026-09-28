> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# Qwen Sparse Attention から見える PSQ Proxy / Long-Context Pruning の研究方向

## 1. 前提

現在の PhaseNonShift の対象モデルは **Qwen3.8 27B** である。

したがって、本レポートで扱う **Qwen Sparse Attention (QSA)** は、
現在対象モデルにそのまま存在する機構として扱うものではない。

QSAは、今後の

- Long-Context Attention高速化
- Query-aware KV pruning
- PSQ2 / PSQ4をselector proxyとして利用する設計

を考える上で、非常に参考になる既存アーキテクチャとして整理する。

---

## 2. 発端

PhaseNonShiftでは、PSQ8由来の低bit proxyについて、

```text
DFlash2 INT2 pruning
candidate pool = 32
```

で今回の評価対象を**全件正解**した。

これは、

```text
低bit値を最終結果として使わなくても、
重要候補を落とさないselectorとして使える
```

という方向を強く示している。

この考え方をlong-context attentionへ応用する案として、

```text
全KV
  ↓
cheap selector
  ↓
重要KV blockだけ選択
  ↓
元の高精度KVでexact attention
```

という **KV block pruning** を検討した。

---

## 3. これはKV Cache圧縮ではない

ここで考えている方式は、

```text
KV CacheそのものをPSQ2へ圧縮する
```

方式ではない。

KV Cache本体は保持する。

高速化対象は、

```text
decode時に読むKV量
QK計算量
softmax対象
V weighted sum
```

である。

概念的には、

```text
Full KV Cache
     │
     ├── cheap selector
     │       ↓
     │   selected pages
     │       ↓
     └── exact K/V read
             ↓
        exact attention
```

となる。

---

## 4. SWAとの違い

見た目はSliding Window Attention (SWA)に近いが、本質は異なる。

### SWA

```text
現在token
   ↓
直近W tokenのみ見る
```

距離ベースで固定的に切る。

古いtokenは内容に関係なく見えなくなる。

### Query-aware pruning

```text
現在Query
   ↓
過去context全体をcheapに評価
   ↓
重要なblockだけ選択
```

距離ではなく、現在Queryとの関連度で選ぶ。

したがって、

```text
recent context
+
sparse global context
```

という構造になる。

遠い過去でも重要なら選択できる。

---

## 5. Qwen Sparse Attentionとの関係

Qwen Sparse Attention (QSA) は、この発想に非常に近い。

QSAでは概念的に、

```text
long context
    ↓
lightweight learned indexer
    ↓
重要micro-blockを選択
    ↓
selected tokensのみ
full attention
```

という二段構造を採る。

重要なのは、

```text
indexer自体が最終attentionを計算するわけではない
```

という点である。

indexerの役割は、

```text
重要候補を取りこぼさないこと
```

であり、最終的なattentionは選択されたtokenに対して本来のQ/K/Vを用いて行う。

この構造は、PhaseNonShiftでDFlash2に使っているlow-bit candidate pruningと非常に近い。

---

## 6. DFlash2との対応関係

```text
DFlash2

248K vocab
   ↓
cheap low-bit proxy
   ↓
candidate pool
   ↓
PSQ8 exact rerank
```

に対して、

```text
Sparse Attention

大量のKV blocks
   ↓
cheap indexer / proxy
   ↓
selected blocks
   ↓
exact attention
```

となる。

共通点は、

```text
proxyの絶対値精度より、
重要候補をcandidate setから落とさないことが重要
```

という点である。

---

## 7. PSQとの相性

今回PSQ8由来INT2で、

```text
candidate pool = 32
```

で全件正解したことから、

PSQ系は単なる量子化形式ではなく、

```text
高忠実度なlow-bit ranking proxy
```

として利用できる可能性がある。

QSA型のindexerや独自KV block selectorに対して、

```text
PSQ8
  ↓
PSQ4
  ↓
PSQ2 Shape
```

と低bit化し、

```text
重要blockのrankingだけ維持する
```

という方向が考えられる。

---

## 8. PSQ2をKV本体に直接使う必要はない

単純PSQ2 KVが高精度attentionに耐えない場合でも、
selector用途なら要求精度は大きく異なる。

### KV本体としてPSQ2を使う場合

```text
Q × PSQ2(K)
  ↓
approx attention logits
  ↓
softmax
  ↓
PSQ2(V)
```

量子化誤差が最終attentionへ直接入る。

### SelectorとしてPSQ2を使う場合

```text
Q × cheap proxy
  ↓
Top-N blocks
  ↓
元のFP8/BF16 K/V
  ↓
exact attention
```

PSQ2に必要なのは、

```text
重要blockをTop-Nから落とさない
```

ことだけである。

この差は非常に大きい。

---

## 9. Candidate Budgetで誤差を吸収できる

selectorの精度が多少低下しても、

```text
Top-512
↓
Top-576
↓
Top-640
```

のようにcandidate budgetを少し増やすことで、
重要blockのrecallを回復できる可能性がある。

これはDFlash2で、

```text
proxy精度
vs
candidate pool size
```

を調整する考え方と同じである。

---

## 10. PSQ2 Shapeとの関係

検討中のPSQ2 Shapeは、

```text
2bit index × 32
BF16 scale / W32
4bit shape ID / W32
```

で、

```text
2 + 16/32 + 4/32
= 2.625 bpw
```

となる。

この方式をattention本体へ直接適用するのではなく、

```text
attention indexer
KV page summary
block scoring metadata
```

などのselector部分へ利用する方が、
より安全かつ効果が出る可能性がある。

---

## 11. QSAから得られる設計上の示唆

### 11.1 Selectorは専用に設計してよい

QSAが示しているのは、

```text
full attentionと同じ表現を
cheap selectorに使う必要はない
```

ということでもある。

つまりPhaseNonShiftでも、

```text
KV本体の表現
```

と、

```text
KV block selection用の表現
```

を分離してよい。

---

### 11.2 Selector専用weight / metadataを持てる

候補として、

```text
K min/max
centroid
radius
learned index embedding
compressed key representation
```

などが考えられる。

さらにそれらを、

```text
W32 BF16 scale
+
PSQ2 / PSQ4
```

で表現することも可能。

---

### 11.3 Exact Attentionを残せる

この方式の大きな利点は、

```text
selected candidate内では
元のattention精度を維持できる
```

点である。

誤差源を、

```text
selectorによるcandidate miss
```

へ集中できる。

これは検証もしやすい。

---

## 12. 現在のQwen3.8 27Bでの位置付け

現在の対象はQwen3.8 27Bであり、
QSAをそのまま実装することが目的ではない。

現時点では、

```text
QSA = 参考アーキテクチャ
```

として扱う。

PhaseNonShiftで検討する場合は、

```text
既存paged attention
    ↓
後付けquery-aware page selector
    ↓
selected KV pages
    ↓
既存exact attention
```

としてPoC可能。

その後、

```text
BF16 selector
↓
PSQ8 selector
↓
PSQ4 selector
↓
PSQ2 Shape selector
```

と研究を進める。

---

## 13. PoC方針への影響

以前検討したLong-Context KV block pruning PoCは依然として有効である。

ただしQSAの存在から、

```text
query-aware sparse attention
```

という方向自体は既に合理性のある設計であることが分かる。

したがってPoCでは、

### Gate 1

Oracleで、

```text
理想的に重要pageを選べた場合、
何%のKV pageを削減できるか
```

を確認する。

### Gate 2

BF16 min/maxなどのcheap selectorを試す。

### Gate 3

GPU selectorを実装する。

### Gate 4

selected pageだけを処理するsparse paged attentionを実装する。

### Gate 5

selector metadataをPSQ8 / PSQ4 / PSQ2 Shapeへ圧縮する。

という流れが妥当。

---

## 14. PSQ研究への新しい評価軸

従来は、

```text
PSQ2 / PSQ4 / PSQ8
    ↓
本計算をどこまで正確にできるか
```

を重視していた。

今回のDFlash2結果とQSA型アーキテクチャを合わせると、

```text
PSQを使って
どれだけ正確に「計算しなくてよい場所」を判定できるか
```

という別の評価軸が重要になる。

具体的には、

```text
Top-K recall
full containment
ranking preservation
important block recall
retained attention mass
candidate budget
```

を測るべきである。

---

## 15. 研究仮説

### 実測事実

DFlash2において、

```text
PSQ8由来INT2
candidate pool = 32
```

で今回の評価対象を全件正解した。

### 仮説A

W32ごとのBF16 scaleにより、
元BF16 weightの局所的な振幅構造が低bit化後も残りやすい。

### 仮説B

その結果、

```text
絶対値精度
```

以上に、

```text
ranking preservation
```

が高い可能性がある。

### 仮説C

この性質は、

```text
LM head pruning
attention block selection
MoE expert selection
FFN channel selection
```

のような候補選択問題で特に有効。

### 仮説D

PSQ2 Shape 2.625 bpwでも、
本計算ではなくselector用途であれば、
十分なcandidate recallを得られる可能性がある。

---

## 16. 重要な方向転換

PSQの高速化を、

```text
FP8より速いGEMMを作る
```

だけで考えない。

新しい方向として、

```text
cheap PSQ proxy
      ↓
重要候補だけ選択
      ↓
高精度本計算
```

を正式な研究対象とする。

すなわち、

```text
高速化 = 1回の計算を速くする
```

だけではなく、

```text
高速化 = 不要な計算そのものを消す
```

という方向である。

---

## 17. 結論

Qwen Sparse Attentionは、
今回PhaseNonShiftで考えていたLong-Context KV block pruningと
非常に近い思想を持つ。

ただし現在対象のQwen3.8 27Bへ
QSAそのものを移植することが直近目的ではない。

重要なのは、

```text
cheap selector
→ candidate pruning
→ exact compute
```

という構造が、
LM head以外にも有効であること。

特にPSQ8由来INT2がcandidate pool=32で全件正解した結果から、

PSQ2 / PSQ4 / PSQ8を、

```text
main compute format
```

だけではなく、

```text
high-fidelity low-bit selector format
```

として研究する価値が高い。

今後のLong-Context Attention PoCでは、

```text
Oracle pruning
→ BF16 selector
→ sparse exact attention
→ PSQ selector
```

の順で評価し、

最終的には、

```text
PSQ2 Shape 2.625 bpw
```

をattention block selectorへ適用できるか検証する。
