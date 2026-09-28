> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# W32ごとのBF16 Scaleが持つ特徴と可能性

## 1. 概要

PSQ系量子化では、32 weight（W32）ごとにBF16 scaleを保持する設計を採用している。

この構造は単に量子化誤差を抑えるためだけではなく、元のBF16 weightが持つ局所的な振幅・相対強度・分布形状をかなり細かい粒度で保持できる可能性がある。

今回、DFlash2のINT2枝刈りにおいて、

- FP8由来のINT2 proxyでは候補数80程度が必要だった
- PSQ8由来のINT2 proxyでは、候補数32で今回の評価対象を全件正解した

という観測が得られた。

この結果から、PSQ8由来の低bit proxyが今回の評価条件では非常に高い候補保持性能を示したことは実測事実である。一方、その主因がW32ごとのBF16 scaleによる元BF16層の「重要度順位」や「局所構造」の保持である、という説明は現時点では仮説である。

---

## 2. PSQの基本構造

概念的には、W32ごとに以下の形でweightを表現する。

```text
32 weights
   │
   ├─ quantized code × 32
   │
   └─ BF16 scale × 1
```

各weightは概ね、

```text
w_i ≈ scale_w32 × code_i
```

として復元される。

PSQ8の場合は8bit codeを用いるため、

```text
8.0 bit   : weight code
0.5 bit   : BF16 scale / 32 weights
--------------------------------
8.5 bpw
```

となる。

W32単位でscaleを持つことで、より大きなblock単位の量子化と比べ、weightの局所的なレンジ変化を保持しやすい。

---

## 3. W32 BF16 Scaleが保持している可能性がある情報

### 3.1 局所的な振幅構造

大きなblockで1つのscaleを共有すると、block内部の局所的な振幅差が圧縮されやすい。

一方、W32ごとにscaleを持つ場合、

```text
BF16 original

[ small ][ large ][ medium ][ very small ][ large ]
    ↓        ↓        ↓           ↓          ↓
   S0       S1       S2          S3         S4
```

のように、32 weight単位の局所的なenergy mapをscale列として保持できる。

これはcode自体を2bitや4bitに削減しても残る情報である。

---

### 3.2 dot-product寄与度の順位

推論において重要なのは、必ずしも各weightの絶対誤差だけではない。

候補選択や枝刈りでは、

```text
score_A > score_B
```

という順位関係が維持されることが重要になる。

W32 BF16 scaleによって局所的な振幅が保存されるなら、低bit codeに落とした場合でもdot-productの相対順位が維持されやすい可能性がある。

今回のDFlash2 INT2枝刈りでは、PSQ8由来INT2の32候補で全件正解しており、少なくとも今回の評価範囲では高い順位保存性能が実測された。

---

### 3.3 元BF16層の局所的な特徴

PSQ8では、

```text
BF16 weight
   ↓
W32単位で正規化
   ↓
quantized code
```

という構造になる。

そのため、低bit化によって細かい値そのものを失っても、

- そのW32 blockが強いか弱いか
- 正負の分布
- 大きいweightと小さいweightの局所比
- dot-productへの寄与が大きそうな領域

といった情報が残る可能性がある。

---

## 4. 今回のDFlash2 INT2枝刈りとの関係

### 4.1 実測結果

今回の評価では、PSQ8由来のINT2 proxyについて、

```text
candidate pool = 32
```

で評価対象を**全件正解**した。

これは「32候補でも十分そう」という推定ではなく、今回の評価範囲における実測結果である。

比較対象では、FP8由来のINT2 proxyはより大きな候補数を必要としていたため、PSQ8由来proxyの候補保持性能には明確な差が観測されている。

ただし、

```text
なぜPSQ8由来INT2が32候補で全件正解できたのか
```

という因果説明はまだ確定していない。

現時点の主要仮説は、

```text
W32ごとのBF16 scaleが
元BF16 weightの局所的な振幅・energy・順位構造を保持し、
2bit proxy化しても重要候補の順位が壊れにくい
```

というものである。

したがって、以下を明確に分離して扱う。

```text
実測事実:
PSQ8由来INT2はcandidate pool=32で全件正解した。

研究仮説:
W32 BF16 scaleによる局所構造保持が、その高い候補保持性能の主因である。
```


DFlash2のlm_head枝刈りでは、低精度proxyで全語彙を粗く評価し、その後、少数候補のみ高精度で再計算する。

```text
全語彙
  │
  ▼
low-bit proxy
  │
  ▼
Top-R candidates
  │
  ▼
PSQ8 / FP8 exact rerank
  │
  ▼
final Top-K
```

重要なのは、low-bit proxyのlogit値そのものが高精度である必要はない点である。

必要なのは、

```text
本当に必要な候補がTop-Rに含まれていること
```

である。

したがって評価指標としては、

- Weight MSE
- KLD
- PPL

だけでなく、

- exact Top-K recall
- exact Top-K full containment
- ranking preservation
- final selected token coverage

が重要になる。

---

## 5. PSQ8がINT2 proxyに強い可能性

今回の評価では、

```text
FP8 → INT2 proxy
    Top-R ≈ 80必要

PSQ8 → INT2 proxy
    Top-R = 32で全件正解
```

という明確な差が観測された。

これは少なくとも今回の評価条件では、PSQ8からlow-bit化した際の候補順位・候補包含性能が非常に高かったことを意味する。

つまりPSQ8は、

```text
Exact Compute Format
```

としてだけでなく、

```text
High-Fidelity Proxy Source
```

として価値を持つ可能性がある。

---

## 6. 「PSQ Dual-View / Multi-View」という考え方

PSQ weightを単一精度で扱うのではなく、同じW32 scaleを共有する複数のviewとして扱う。

```text
                BF16 original
                      │
                      ▼
             W32 BF16 scale
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
        PSQ2         PSQ4        PSQ8
       coarse       medium       exact
```

例えば、

```text
PSQ2:
2bit code + same W32 BF16 scale

PSQ4:
4bit code + same W32 BF16 scale

PSQ8:
8bit code + same W32 BF16 scale
```

という階層が考えられる。

この場合、

```text
PSQ2で粗く探索
   ↓
候補のみPSQ4
   ↓
さらに候補のみPSQ8
```

というhierarchical refinementが可能になる。

---

## 7. PSQ2の可能性

### 7.1 基本PSQ2

W32ごとにBF16 scaleを持つ単純なPSQ2なら、

```text
2bit × 32 weights = 64bit
BF16 scale         = 16bit
--------------------------
total              = 80bit / 32
                   = 2.5 bpw
```

となる。

通常のINT2よりmetadata overheadは増えるが、局所scaleを保持できる。

---

### 7.2 Learned Shape付きPSQ2

4択の値そのものを固定せず、複数の4-level shapeを事前学習しておき、W32ごとにshape IDを持たせる方式も考えられる。

```text
w_i ≈ scale_b × C_shape[q_i]
```

where:

```text
q_i      : 2bit index
scale_b  : BF16 scale / W32
shape    : learned 4-level codebook ID
```

shape IDが4bitなら、

```text
2bit × 32 weights = 64bit
BF16 scale         = 16bit
shape ID           =  4bit
--------------------------
total              = 84bit / 32
                   = 2.625 bpw
```

となる。

これは非常に興味深い。

---

## 8. Learned Shapeが重要になる理由

2bit量子化の問題は、単純に「4値しかない」ことだけではない。

より重要なのは、

```text
4個の代表値をどこに配置するか
```

である。

固定INT2では全blockに同じ4値を使う。

一方、Learned Shape方式では例えば、

```text
Shape 0:
[-1.00, -0.33, +0.33, +1.00]

Shape 1:
[-1.00, -0.18, +0.11, +0.72]

Shape 2:
[-0.76, -0.08, +0.24, +1.00]
```

のような複数shapeを用意できる。

W32ごとに最も近いshapeを選択すれば、

```text
2bit index
+
local BF16 scale
+
4bit shape selector
```

だけで、単純INT2よりかなり高い表現力を得られる可能性がある。

---

## 9. PSQ2 Shapeのbit効率

PSQ2 Shapeが2.625 bpwの場合、

```text
PSQ8          8.5 bpw
PSQ4          4.5 bpw
PSQ2 Shape    2.625 bpw
```

となる。

PSQ4比では、

```text
2.625 / 4.5 ≈ 0.5833
```

つまりweight trafficを約41.7%削減できる。

decodeがweight bandwidth boundである場合、この差は非常に大きい。

---

## 10. 応用候補

### 10.1 LM Head Candidate Pruning

最も実績がある対象。

```text
vocab全体
   ↓ PSQ2 / INT2 proxy
少数candidate
   ↓ PSQ8 exact
Top-K
```

今回のDFlash2観測が直接対応する。

---

### 10.2 FFN Channel Pruning

SwiGLUなどのintermediate channelを低精度proxyで粗く評価し、寄与の大きいchannelのみexact計算する。

```text
all FFN channels
      ↓
PSQ2 coarse score
      ↓
important channels
      ↓
PSQ8 exact compute
```

成功すれば削減可能なFLOPsは非常に大きい。

ただしLM headと違い、FFN channelは本来すべて出力に寄与するため、近似誤差評価が必要。

---

### 10.3 KV Block / Page Selection

長context attentionで、全KV blockをexact計算する前にlow-bit proxyで重要blockを選択する。

```text
all KV blocks
     ↓
low-bit score
     ↓
Top-N blocks
     ↓
exact attention
```

長contextほど効果が増える可能性がある。

---

### 10.4 MoE Expert Preselection

router後にさらにcheap proxyを用いてexpert寄与を評価し、本当に必要なexpertのみexact実行する案。

```text
router candidates
      ↓
cheap PSQ proxy
      ↓
selected experts
      ↓
full expert compute
```

ただしモデル挙動を変化させるため、LM head pruningより難易度は高い。

---

### 10.5 Dynamic Precision Refinement

単純なskipだけでなく、

```text
PSQ2
 ↓ uncertain
PSQ4
 ↓ still uncertain
PSQ8
```

という動的精度昇格も可能。

この場合、W32 BF16 scaleを共通化できれば、複数精度view間で構造を共有しやすい。

---

## 11. 適用に向く計算と向かない計算

### 向く

以下のような「多数から少数を選ぶ」処理。

```text
大量候補
   ↓
score / ranking
   ↓
Top-K / threshold
   ↓
少数のみexact compute
```

例:

- LM head candidate pruning
- KV block selection
- FFN sparse channel selection
- MoE expert candidate refinement

---

### 向かない / 慎重に扱うべき

以下は誤差が将来へ蓄積しやすい。

- dense residual path
- recurrent state
- GDN state update
- 長時間保持されるhidden state

特にGDNでは、一度捨てた情報が将来tokenで復元できないため、低精度skipは慎重に扱う必要がある。

---

## 12. PSQの評価軸を拡張する

従来:

```text
1. Weight accuracy
2. PPL / KLD
3. GEMM throughput
```

今後:

```text
1. Exact Compute Quality
   - BF16との差

2. Proxy Quality
   - ranking preservation
   - Top-K recall
   - candidate containment

3. Exact Throughput
   - PSQ2 / PSQ4 / PSQ8 GEMM速度

4. Pruned Effective Throughput
   - proxy + pruning込みで
     実際にどれだけ本計算を消せたか

5. Memory Efficiency
   - bpw
   - bandwidth
   - cache residency
```

特にPSQでは、

```text
Exact GEMMがFP8より遅い
```

だけでは不十分な評価になる可能性がある。

低bit proxyによって本計算の大部分をskipできるなら、最終runtimeではFP8を上回る可能性がある。

---

## 13. 検証すべき仮説

### Hypothesis A

W32 BF16 scaleは元BF16 weightの局所的なenergy structureを保持している。

### Hypothesis B

そのためPSQ8からPSQ2へ落としても、dot-productのrankingが比較的壊れにくい。

### Hypothesis C

固定INT2より、

```text
2bit index + BF16/W32 + learned shape
```

の方が大幅に高いranking preservationを持つ。

### Hypothesis D

PSQ2 Shapeは2.625 bpwでも、candidate pruning用途では十分な精度を得られる可能性がある。

### Hypothesis E

PSQ2 → PSQ4 → PSQ8のhierarchical refinementにより、PSQ8単体より高い実効スループットを得られる可能性がある。

---

## 14. 次のPoC候補

まずモデル全体をPSQ2化する必要はない。

BF16 W32 blockを大量にサンプリングし、以下を比較する。

```text
A. Fixed symmetric INT2

B. W32 BF16 scale付き fixed INT2

C. W32 BF16 scale + Shape8

D. W32 BF16 scale + Shape16

E. W32 BF16 scale + Shape32

F. block-local optimal 4 centroids
   （理論上限の参考）
```

評価:

```text
Weight MSE
Cosine similarity
Dot-product NRMSE
Top-K ranking recall
Top-K full containment
Candidate-selected coverage
Logit KLD
PPL
```

特に重要なのは、

```text
Top-K full containment
```

と、

```text
最終selectorが必要とする候補を落とさない率
```

である。

---

## 15. 研究上の重要な見方

今回の結果から、PSQは単なる量子化方式としてではなく、

```text
元BF16層の局所構造を保持した
multi-resolution weight representation
```

として扱える可能性がある。

つまりPSQの価値は、

```text
BF16に近い値を少ないbitで再現する
```

だけではなく、

```text
低bit viewでも
「どこが重要か」をかなり正しく判断できる
```

ことにあるかもしれない。

もしこの仮説が成立すれば、

```text
高速化 = 1 GEMMを速くする
```

だけでなく、

```text
高速化 = 計算しなくてよい場所を
         低bit proxyで正しく判断する
```

という別の最適化軸が得られる。

これはPSQ2 / PSQ4 / PSQ8を単独フォーマットとして設計するよりも、

```text
PSQ2 coarse
PSQ4 refinement
PSQ8 exact
```

という一体型runtimeへ発展できる可能性を持つ。

---

## 16. 現時点の結論

W32ごとのBF16 scaleは、単なる量子化scale以上の役割を持っている可能性がある。

特に、

- 局所的なweight振幅
- blockごとのenergy
- dot-product寄与度
- candidate ranking
- BF16層の局所的特徴

を低bit化後も保持する情報源として利用できる可能性がある。

DFlash2ではPSQ8由来INT2がcandidate pool=32で全件正解しており、これは仮説ではなく今回の評価で得られた実測結果である。W32 BF16 scaleがその主因であるかどうかは、今後切り分けるべき仮説である。

次段階では、

```text
W32 BF16 scale
+
2bit index
+
learned 4-level shape
```

によるPSQ2 Shape（2.625 bpw）を、通常のINT2とは別の量子化方式として評価する価値が高い。

PSQの研究方向は、

```text
「FP8より速い8bit GEMMを作る」
```

だけではなく、

```text
「BF16の重要構造を保持したlow-bit proxyで
  本計算そのものを削減する」
```

方向まで拡張して検討すべきである。
