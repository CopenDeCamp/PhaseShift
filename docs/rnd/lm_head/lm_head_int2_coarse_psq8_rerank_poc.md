> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# lm_head INT2 coarse + PSQ8 exact rerank PoC（Target / Verify 正本）

## 0. 目的と設計

Target / Verify の production `lm_head` の greedy argmax を、

```
final_norm
    ↓
INT2 coarse lm_head（全語彙スキャン = 枝刈り）
    ↓
coarse Top-N
    ↓
original PSQ8 candidate exact rerank（BF16 相当）
    ↓
candidate argmax
    ↓
next token
```

へ置き換える。**certificate は使わない。**

近似源は **INT2 枝刈りだけ** である。最終 token は production と同一の PSQ8 weight /
scale / activation 量子化で rerank して決めるため、候補集合に入っていれば
production の PSQ8 head をそのまま再現する。したがって

```
true PSQ8 Top1 ∈ INT2 coarse Top-N   →   full PSQ8 と完全一致
true PSQ8 Top1 ∉ INT2 coarse Top-N   →   full PSQ8 と異なる token を返し得る
```

数学的 lossless ではなく、**qualification 済み lossy fast path** として扱う。

---

## 1. 数値契約

| 項目 | 内容 |
| --- | --- |
| 参照（oracle） | production と同一の full PSQ8 `lm_head` + greedy argmax |
| PSQ8 の位置づけ | BF16 相当の精度。INT2 codebook も「元 BF16 model」ではなく **target PSQ8 head そのもの**から導出する |
| 近似源 | INT2 coarse による Top-N 枝刈りのみ |
| 主指標 | `containment`（true PSQ8 Top1 ∈ coarse Top-N）。正常なら `containment == rerank_match` |
| 辞書順 | value 降順、同値は token id 昇順（production greedy と同一） |

`containment == rerank_match` は rerank が候補集合内で exact であること（候補順位の
取り違えが無いこと）を意味する。実測では全 pool で完全一致した（§5）。

---

## 2. 最重要ルール

candidate 選択には必ず **`coarse` logits** を使う。

```
coarse → Top-N → PSQ8 rerank     (正しい)
upper  → Top-N → PSQ8 rerank     (誤り)
```

`upper = coarse + activation_l2 * error_l2` は certificate 研究専用であり、
candidate generator へ混入させてはならない。real hidden で `upper` を候補選択に使うと
mismatch **0.72** まで悪化する（実測）。`error_l2` / `activation_l2` / `upper logits` /
`omitted upper` / `certificate gap` は本経路では計算しない。

---

## 3. 実装

既存 kernel を再利用し、Target / Verify 専用に複製しない。

| 段 | 内容 | kernel |
| --- | --- | --- |
| 1 | activation E4M3 quantization | `launch_activation_quantize_e4m3` |
| 2 | INT2 coarse lm_head | `launch_dflash2_int2_coarse_head` |
| 3 | coarse Top-N | `launch_dflash2_coarse_topn`（branchless key + cursor + bitonic） |
| 4 | PSQ8 candidate exact rerank | `launch_dflash2_psq8_candidate_rerank` |
| 5 | candidate argmax | `launch_dflash2_target_certified_argmax`（best_ids のみ使用） |

Gate:

| Gate | 内容 | test |
| --- | --- | --- |
| T0 | baseline | `ctest -L required` |
| T1/T2 | corpus static qualification | `tests/unit/test_target_lm_coarse_rerank.hip` |
| T3 | closed-loop generation | `tests/unit/test_target_lm_closed_loop.hip` |
| T4 | isolated performance | `tests/unit/test_target_lm_proxy_real_perf.hip` |
| V | Verify candidate generator | `tests/unit/test_verify_lm_proxy_real_verify.hip` |

共通 harness は `tests/support/target_lm_proxy_harness.h`。`topn_on(logits, ...)` で
入力 logits を明示する（`topn()` は certificate 用の `upper` を入力にするため
candidate 選択には使わない）。

---

## 4. 分岐元 / 環境

| 項目 | 値 |
| --- | --- |
| 分岐元 revision | `d282fc29` |
| branch | `poc/target-lm-head-coarse-rerank` |
| worktree | `.worktrees/lm-head-certified-proxy` |
| GPU | AMD Radeon AI PRO R9700 (gfx1201)、測定は物理 GPU3 のみ |
| target model | `models/Qwen3.8-27B-PSQ`（PSQ8 preshuffle W32、hidden 5120、vocab 248320） |
| draft model | `models/Qwen3.8-27B-DFlash2-PSQ4` |
| corpus | `/tmp/opencode/psq2_corpus`（KVPrune 構成、約 197K token） |

---

## 5. Target — Gate T0 / T1 / T2

### T0 baseline

```
cmake --build build --parallel                         : 成功
ctest --test-dir build -L required --output-on-failure : 138/138 PASS（real 312.2 s）
```

### T2 static qualification

corpus は KVPrune と同一構成（Moby Dick / repo code / docs / JSON / needle）。
source の連続領域を document 単位で Train / Val / Qual へ分離する。

| split | tokens |
| --- | ---: |
| train | 137,944 |
| val | 29,593 |
| qual | 29,593 |

episode 4,096 token で prefill し、各 row で
`oracle = full PSQ8 Top1`、`candidate = coarse Top-N → PSQ8 rerank Top1` を比較する。
全 split で `sample_argmax_mismatch = 0`（oracle が argmax であることを確認）。

**train（137,944 rows）**

| N | containment | mismatch | mismatch_rate |
| ---: | ---: | ---: | ---: |
| 16 | 0.994679 | 734 | 5.321e-3 |
| 24 | 0.997985 | 278 | 2.015e-3 |
| 32 | 0.999000 | 138 | 1.000e-3 |
| 48 | 0.999667 | 46 | 3.335e-4 |
| 64 | 0.999833 | 23 | 1.667e-4 |
| 80 | 0.999920 | 11 | 7.974e-5 |
| 96 | 0.999964 | 5 | 3.625e-5 |
| 127 | 0.999993 | 1 | 7.250e-6 |

**val（29,593 rows）**

| N | containment | mismatch | mismatch_rate |
| ---: | ---: | ---: | ---: |
| 16 | 0.990876 | 270 | 9.124e-3 |
| 24 | 0.996452 | 105 | 3.548e-3 |
| 32 | 0.998243 | 52 | 1.757e-3 |
| 48 | 0.999324 | 20 | 6.758e-4 |
| 64 | 0.999628 | 11 | 3.717e-4 |
| 80 | 0.999797 | 6 | 2.028e-4 |
| 96 | 0.999966 | 1 | 3.379e-5 |
| 127 | 1.000000 | 0 | 0 |

**qual（29,593 rows）**

| N | containment | mismatch | mismatch_rate |
| ---: | ---: | ---: | ---: |
| 16 | 0.993208 | 201 | 6.792e-3 |
| 24 | 0.997601 | 71 | 2.399e-3 |
| 32 | 0.998479 | 45 | 1.521e-3 |
| 48 | 0.999561 | 13 | 4.393e-4 |
| 64 | 0.999763 | 7 | 2.365e-4 |
| 80 | 0.999899 | 3 | 1.014e-4 |
| 96 | 0.999899 | 3 | 1.014e-4 |
| 127 | 0.999932 | 2 | 6.758e-5 |

`containment == rerank_match` は全 pool で完全一致した。

### T2 held-out（追加分布）

204K corpus（Moby / repo code / docs / JSON / needle）に加え、distribution shift の検証用に
日本語・実コード・structured（JSON）・math を用意した。各 source は repo 内テキストから
tokenizer で tokenize した（PSKLDTOK、`/tmp/opencode/lm_head_heldout`）。

| source | tokens | N=16 | N=32 | N=64 | N=127 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ja（日本語 docs） | 12,000 | 0.995833 | 0.999083 | 0.999917 | 1.000000 |
| code（src） | 12,000 | 0.999167 | 0.999833 | 0.999917 | 1.000000 |
| json（structured） | 6,378 | 0.995453 | 0.999843 | 1.000000 | 1.000000 |
| math（数値・式） | 12,000 | 0.998750 | 0.999833 | 1.000000 | 1.000000 |
| **hient（高エントロピー生成）** | 12,000 | **0.746083** | **0.849667** | **0.918167** | **0.959667** |

（数値は containment = rerank_match。N=64 の mismatch は ja 1 / code 1 / json 0 / math 0 /
hient 982、N=127 は ja/code/json/math 0 / hient 484。`sample_argmax_mismatch = 0`。）

`hient` は temperature 1.3 / top_p 0.98 で生成した 12,000 token で、decode すると多言語の断片が
混ざった劣化テキストになる。production distribution そのものではないが、モデルの予測
エントロピーが高い状態を作る **stress set** である。**N を増やしても containment が 0.96 止まり**で、
自然分布（N=127 で 0）とは明確に挙動が異なる。この反例への対応は
`lm_head_margin_fallback_poc.md`（high-entropy fallback）で扱う。

**N の上限**: `kDflash2Int2MaxPool = 128` のため本 PoC で測定できる pool は 127 までである
（harness は `pool < kDflash2Int2MaxPool` を要求）。N=128/192/256 を測るには partition
kernel の LDS 容量（`kWarps * kDflash2Int2MaxPool`）を増やす必要があり、occupancy を
悪化させる。N=127 で train mismatch 7.25e-6（1 / 137,944）に達しており、accuracy 面で
それ以上の N を選ぶ動機は小さい。

**zero-event の 95% upper bound**（≈ 3 / N）: val N=127 は約 1.0e-4。これは数学的保証ではなく、
「その N で観測しなかった」ことの上限にすぎない。0 件を lossless とは呼ばない。

---

## 6. Target — Gate T3 closed-loop generation

corpus 先頭から等間隔に取った prompt に対し、`temperature=0` の greedy で
full PSQ8 と proxy を **それぞれ独立に** 生成し、生成列を比較する
（`prompt_tokens=128`、`generated=96`）。

| pool | prompts | generated tokens | divergent prompts | divergent tokens | first divergence | proxy local mismatch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 12 | 1,152 | 0 | 0 | — | 0 |
| 64 | 12 | 1,152 | 0 | 0 | — | 0 |

`sequence_match_rate = 1.000000`。`proxy local mismatch` は proxy trajectory 上で
proxy token が full PSQ8 token と異なった step 数であり、これも 0 だった。

held-out 分布（16 prompt × 128 token = 2,048 token / 本、pool 64 / 127）:

| dataset | pool | divergent tokens | first divergence | proxy local mismatch |
| --- | ---: | ---: | ---: | ---: |
| ja | 64 | 1 | 26 | 1 |
| ja | 127 | 0 | — | 0 |
| code | 64 / 127 | 0 | — | 0 |
| json | 64 / 127 | 0 | — | 0 |
| math | 64 / 127 | 0 | — | 0 |
| hient | 64 / 127 | 0 | — | 0 |

`hient` は static では N=64 で 8% mismatch だが、greedy closed-loop では 2,048 token × 2 pool で
divergence 0 だった。temperature 1.3 の生成 token 列はモデルの予測エントロピーが高い状態を作るが、
そこから greedy decoding すると速やかに低エントロピー領域へ入るためである。すなわち hient の
static mismatch は **ranking の stress test** であり、greedy production でそのまま発生する
ものではない。ただし「モデルが自信を失う局面」は原理的に起こり得るため、
`lm_head_margin_fallback_poc.md` で cheap な検出を検討する。

T2 の mismatch 率（N=32 で 1.0e-3）からは約 1 件の divergence が期待されるが、
1,152 token では観測されなかった。lossless ではなく、**観測範囲で divergence 0** と記録する。

---

### On-policy greedy shadow（production qualification）

static corpus は stress qualification であり、production 判断は **on-policy** で行う。
baseline を full PSQ8 greedy とし、実際の次 token には必ず full PSQ8 の oracle を使い、
同じ final_norm hidden から proxy を計算して比較する（proxy は診断のみ、trajectory は汚染しない）。
詳細は `lm_head_onpolicy_greedy_shadow_poc.md`。

539,611 decisions（100,000 級 run 完走 + 1,000,000 級を 437,211 で停止、合算）:

| pool | mismatch | rate | natural のみ（hient 除外 462,687） |
| ---: | ---: | ---: | ---: |
| 64 | 68 | 1.260e-4 | 47 / 462,687 = 1.016e-4 |
| 127 | 15 | 2.780e-5 | 7 / 462,687 = 1.513e-5 |

static との差は大きく（N=64、hient: 6.73e-2 → 2.73e-4、mixed204k/train: 5.80e-4 → 1.30e-4、
math: 6.67e-4 → 5.20e-5）、**static corpus には greedy production では到達しにくい hidden が
大量に含まれる**。N=127 の 15 件はすべて full PSQ8 Top1 が coarse Top-127 の外側にある
**prune miss**（rerank の誤りではない）である。

---

## 7. Target — Gate T4 isolated performance

real hidden、full path は production と同一の `select_psq8_gemm_config` で測定（us、median）。

### rows=1

| N | quant | coarse | topn | rerank | argmax | total | full | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 10.3 | 912.3 | (noise) | 71.6 | 5.1 | 980.6 | 2145.1 | **2.19x** |
| 32 | 10.4 | 914.1 | 3.4 | 70.6 | 2.2 | 1000.7 | 2144.4 | **2.14x** |
| 64 | 10.3 | 913.3 | 48.1 | 78.9 | 3.7 | 1054.3 | 2143.2 | 2.03x |
| 127 | 10.3 | 911.9 | 134.7 | 78.0 | 2.8 | 1137.8 | 2141.2 | 1.88x |

### rows=8

| N | coarse | topn | rerank | total | full | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 927.8 | 47.8 | 69.4 | 1058.2 | 2208.8 | 2.09x |
| 32 | 929.5 | 98.9 | 73.4 | 1115.0 | 2207.1 | 1.98x |
| 64 | 927.1 | 201.0 | 75.6 | 1214.1 | 2204.7 | 1.82x |
| 127 | 925.4 | 403.7 | 94.0 | 1433.3 | 2201.9 | 1.54x |

観測:

- **coarse が支配的**（total の 82〜92%）。rows=1 で 318 MB を 912 us で読んでおり、
  実効帯域は約 **348 GB/s**（streaming read 実測 636 GB/s の 55%）。最大の改善余地。
- rerank は N=32 で約 71 us、N=127 でも約 78 us。**pool 増加に極めて軽い**。
- topn は rows=1 で N=32 が 3 us、N=127 が 135 us。
- N=32 と N=64 の total 差は約 54 us（rows=1）。accuracy を 1 桁改善する対価として小さい。

---

## 8. Verify — candidate generator

Verify の target lm_head も同じ candidate generator で置き換えられるかを、
real verify hidden と実 DFlash2 draft で測る。

### 8.1 real verify hidden

`test_verify_lm_proxy_real_verify.hip`。Moby 256 token prompt、ctx 256、
1024 rounds、**7,168 draft rows**、候補生成は `topn_on(coarse)`。

| N | candidate mismatch | rate | false_accept | false_reject |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0 / 7,168 | 0.000000 | 0 | 0 |
| 127 | 0 / 7,168 | 0.000000 | 0 | 0 |

0/7,168 の 95% CI 上限は約 0.042%。

### 8.2 実 DFlash2 draft での E2E

`Qwen3.8-27B-DFlash2-PSQ4` を draft、`Qwen3.8-27B-PSQ` を target とし、INT2 draft head
（pool=32）で spec decode を実行（prompt 256 token、max-new 256）。

```
DFLASH2_ROUNDS=76            DFLASH2_FULL_ACCEPTS=24
DFLASH2_PARTIAL_ACCEPTS=51   DFLASH2_ACCEPTED_DRAFTS=179
DFLASH2_MEAN_ACCEPTED=2.355  DFLASH2_EMITTED_PER_ROUND=3.355
DECODE_TOKENS_PER_SEC=71.84
```

E2E の verify は target が full PSQ8 のため correctness counter は自明に 0。
上記は draft head（INT2、pool=32）の受理率である。

---

## 9. 判定

| 項目 | 判定 |
| --- | --- |
| Target static qualification | containment は N とともに単調改善。N=32 で 1.0e-3、N=64 で 1.7e-4、N=127 で 7.3e-6（train） |
| Target closed-loop | 1,152 token（N=32 / N=64）で divergent 0 |
| Target 性能 | rows=1 N=32 で **2.14x**、N=64 で 2.03x |
| Verify candidate generator | 7,168 row で mismatch 0、false accept / reject 0 |
| **candidate N の候補** | accuracy / runtime の両面から **N=32 を第一候補、N=64 を accuracy 重視の第二候補** |
| production integration | 未実施（common object 抽出 → runtime shadow → LINEAR+SAMPLING fusion が次段） |

production GO の条件（未達成分）:

- closed-loop を prompt / token 数ともに増やす（現在 1,152 token）。
- 日本語 / 実コード / structured generation / math / 高エントロピー生成の held-out set を追加する。
  204K corpus は Moby / repo code / docs / JSON / needle であり、この範囲は未評価。
- N=32 の 1.0e-3（約 1000 token に 1 回）を許容するか、N=64 を採るかの判断。

---

## 10. 不採用の記録

### 10.1 certificate（上界による全域証明）— NO-GO

`upper = coarse + activation_l2 * error_l2` により「この argmax は数学的に正しい」と
証明できた行だけ proxy 結果を使い、証明できない行は full PSQ8 へ fallback する方式。

- real target prefill hidden 12,032 decisions、pool 16〜127、margin A〜E のすべてで
  **cert_rate = 0**。すなわち 1 行も証明できない。
- 原因は L2 Cauchy–Schwarz 上界が次元 K=5120 に対し sqrt(K)≈72 倍緩いこと。
  real hidden 実測 `||A||=137.6, ||D||=0.474, global bound=65.2, W32 group=59.3,
  actual=0.611`（bound/actual ≈ 100）。
- cert_rate=0 では全行 fallback となり、コストは `proxy + full`。
  isolated performance も full 2.1449 ms に対し expected 3.3176 ms（**0.65x**）で
  full 単体より遅い。
- 結論: certificate によって full PSQ8 を回避するという旧来の目的は成立しない。
  **NO-GO。**

### 10.2 residual-corrected certificate — NO-GO

`D = W_psq8 - W_int2` の構造を利用して上界を締め直す方式。

- `D` は sparse でも W32 group 集中でも low-rank でもない。
  Top-M256（K の 5%）で residual energy 22.7%、Top-W32 G32（K の 20%）で 33.3%、
  low-rank r=64 で 0.19%（白色雑音）。
- corrected upper でも certificate rate は全方式 / 全 pool で 0。
  bound slack は 75.7 → 50.1 までしか縮まない。
- 結論: exact certified pruning を停止する。**NO-GO。**

### 10.3 learned PSQ2 / Rank Shape16 — 不採用（production は INT2）

coarse 段を learned PSQ2（16 shapes × 4 levels）で置き換える案。

- Phase A（204K corpus）の INT2 baseline は **N=16 で約 99%、N=24〜32 で 99.6〜99.8%**。
  旧 synthetic の N=32 = 0.936 は OOD 由来で過度に難しかった。
- 凍結比較で Shape 系の明確な優位は **N=4〜8 のみ**
  （val N=4: INT2 0.890 → Shape16 0.972、N=8: 0.962 → 0.995）。N>=16 では飽和して差が消える。
  Global Rank4 は INT2 とほぼ同等。
- PSQ2 Shape は coarse 段（全候補を必ずスキャンする第一段）を複雑化し、
  N=16→8 の節約をそこで食い潰す可能性が高い。
- 結論: **production は INT2（N=16/24/32）**。PSQ2-AW / Rank Shape16 は N<16 が
  必要になったときだけ使う R&D 切り札として残す。Global Rank4 は落とす。
- PSQ2 の実装（`tools/psq2`, `test_psq2_ranking_proxy`）は削除済み。
- 将来 PSQ2 coarse へ差し替える場合も、比較は **同じ accuracy を達成する設定同士**
  （例: INT2 N=64 対 PSQ2 N=32）で行う。

---

## 11. INT2 kernel の現状と効率化の記録

candidate path の性能は ① INT2 coarse head と ② coarse Top-N で決まる。

- **① coarse head**: rows 1〜8 で時間がほぼフラット = memory-bound。命令削減
  （unroll / prefetch / LUT 除去）は効かず、**block geometry** が効いた。
  1 block = 32×W waves（W=16、512 threads、出力 256/block）で **970 → 890 us**、
  実効帯域 410 → 447 GB/s（ピークの約 70%）。`test_dflash2_int2_coarse_head: passed=15 failed=0`。
- **② coarse Top-N**: **branchless 64-bit key + per-thread cursor（16 要素再探索の廃止）+
  partition ごとの k-way merge + bitonic sorting network** に書き換え。
  exec-mask 命令 20 → 4、VGPR 148 → 90、occupancy 9 → 16。
  topn **232 → 146 us**（N=32）、merge 41 → 28 us、partition 150 → 103 us。
  `test_dflash2_coarse_topn: passed=93 failed=0`。
- 合成後（rows=8, N=32）: total **1384 → 1171 us**（coarse 890 + topn 146 + exact 63 + certify 3）。
- **負の結果**（再試行しないこと）: coarse の full-tile 特化（out_valid 除去）は
  命令数 477 → 474 でも実測 858-887 → 886-915 us と 3% 悪化。coarse 専用 FP8 scale は
  40 MB 削減でも 859-888 → 861-890 us と完全に同一。**coarse head は byte 律速ではなく**、
  メモリ要求数 / latency / LUT 依存で律速している（multi-wave block のみが効いた事実と整合）。
- 残る最大レバーは coarse head。**ただし帯域ではない**（下記）。
- **roofline 切り分け（決定的測定）**: coarse head と同一のアクセスパターン（2 B/lane、
  128 B/warp、連続）で 397 MB（INT2 codes + scales）を計算なしで読むと **667 us / 624 GB/s**。

  | access pattern | time | bandwidth |
  | --- | ---: | ---: |
  | uint4（16 B/thread） | 663 us | 627 GB/s |
  | uint2（8 B/thread） | 666 us | 625 GB/s |
  | coarse head と同一（2 B/lane, 128 B/warp） | 667 us | 624 GB/s |
  | uint16（2 B/thread 連続） | 758 us | 549 GB/s |

  実測 914 us（434 GB/s）に対し床は 667 us なので、**約 247 us はメモリ以外**である。
  FP8 scale no-op（40 MB 削減で無変化）はこれと整合する（メモリ時間が compute/latency と
  重なっており、メモリだけ削っても縮まない）。つまり coarse head は bandwidth-bound では
  なく、**メモリと非メモリ（issue / latency）の co-limited**。
- 未着手のレバーはこの非メモリ約 250 us のみ。上限は 914 → 667 us
  （candidate total 1000 → 753 us、2.14x → 2.85x）。

---

## 12. 再現

```
# T0
cmake --build build --parallel
ctest --test-dir build -L required --output-on-failure

# T1/T2（split ごと）
HIP_VISIBLE_DEVICES=3 PHASESHIFT_MODEL_DIR_DFLASH2_TARGET=models/Qwen3.8-27B-PSQ \
  PHASESHIFT_TARGET_LM_CORPUS=/tmp/opencode/psq2_corpus/train.pskldtok \
  PHASESHIFT_TARGET_LM_SEG=4096 \
  ./build/tests/test_target_lm_coarse_rerank

# T3
HIP_VISIBLE_DEVICES=3 PHASESHIFT_MODEL_DIR_DFLASH2_TARGET=models/Qwen3.8-27B-PSQ \
  PHASESHIFT_TARGET_LM_CORPUS=/tmp/opencode/psq2_corpus/train.pskldtok \
  PHASESHIFT_TARGET_LM_CL_PROMPTS=12 PHASESHIFT_TARGET_LM_CL_TOKENS=96 \
  PHASESHIFT_TARGET_LM_PROXY_POOL=32 \
  ./build/tests/test_target_lm_closed_loop

# T4
HIP_VISIBLE_DEVICES=3 PHASESHIFT_MODEL_DIR_DFLASH2_TARGET=models/Qwen3.8-27B-PSQ \
  ./build/tests/test_target_lm_proxy_real_perf

# coarse head の roofline 切り分け（§11）
/opt/zen/.venv/bin/hipcc --offload-arch=gfx1201 -O3 /tmp/opencode/read_bench.hip \
  -o /tmp/opencode/read_bench
HIP_VISIBLE_DEVICES=3 /tmp/opencode/read_bench 416284672   # 397 MB = codes + scales
```

corpus は KVPrune と同一構成（Moby Dick / repository code / docs / JSON / needle、約 197K token）で、
source ごとの連続領域を document 単位に Train 70% / Val 15% / Qual 15% へ分割して作る。
隣接 chunk を split 間に interleave しない。生成済み corpus は
`/tmp/opencode/psq2_corpus/{train,val,qual}.pskldtok`（PSKLDTOK: magic + version + count +
uint32 tokens）にある。`PHASESHIFT_TARGET_LM_CORPUS` にこのパスを渡す。

---

## 13. 次段（未実施）

手順書 §25〜§39 の production integration。

1. common object `LmHeadCandidateProxy` の抽出（candidate backend を INT2 / PSQ2 で
   差し替え可能にする。certificate metadata は持たない）。
2. executor への optional proxy ownership 追加。
3. `LINEAR(lm_head) + SAMPLING` の fused dispatch（`ExecutionRole::Decode` かつ
   greedy / constraint なし / PSQ8 preshuffle W32 のときのみ適用）。
4. `PHASESHIFT_TARGET_LM_HEAD_PROXY=0|1|2`（off / fast / shadow）と
   `..._POOL` / `..._DIAG`。
5. production shadow 100k+ decisions、E2E fast path 再測定
   （既知 baseline: DFlash2 71.84 tok/s）。

Stochastic sampling（temperature > 0、top_p < 1、top_k）と full logits を要求する経路、
constraint 付き sampling は initial scope 外であり、既存 full PSQ8 path へ fallback する。
