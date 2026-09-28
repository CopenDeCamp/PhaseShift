> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# DFlash2 / speculative verify 最適化履歴

`docs/rnd/optimization_findings.md` から分離した、DFlash2 drafter・speculative verify・INT2 coarse head の実験履歴。見出し番号（7.x）は元 dump の通し番号を保持する。DFlash2 全体の Gate 記録は同 directory の [dflash2.md](dflash2.md) を参照。

## 収録セクション

- 7.98 speculative verify の小 kernel は launch floor にある
- 7.100 INT2 coarse draft head の pool 選定と、近似源を exact 側に揃える原則
- lm_head candidate proxy（INT2 coarse + PSQ8 exact rerank）— 正本

---

## 7.98 speculative verify の小 kernel は launch floor にある（2026-09-22）

Gate 11H（DFlash2 target verify）の profile で、verify の非 linear primitive は
**仕事量を増やしても時間が増えない**ことが分かった。rows を 16 倍にしても p50 は
1.4 倍程度しか増えない。

| kernel | rows=2 | rows=8 | rows=32 | rows=128 |
| --- | ---: | ---: | ---: | ---: |
| activation quantize e4m3（K=5120） | - | 4.15 | 4.30 | 5.68 |
| residual add bf16（F=5120） | 3.82 | 3.97 | 4.14 | 5.64 |
| RMSNorm bf16（F=5120） | 5.45 | 5.46 | 5.06 | 18.58 |

したがって verify の非 linear family（activation quantize / RMSNorm / elementwise /
BF16 helper）は **内部最適化の headroom が無い**。改善は launch 数削減
（fusion / graph / persistent）でしか出ない。

### bit-exact を崩さずに効いた唯一の型: work partition

同じ Gate 11H で、attention の split reduce（1 WG = (row, kv_head)、rows=8 で
grid 32 WG、M=1..8 で 607-622us と M 非依存）を q head 単位に分割した
（grid = rows × kv_heads × q_per_kv = 192 WG）。演算順を変えないので bit-exact のまま
**31.09 → 7.82us/call（3.98x）**、Verify wall -0.33ms。

低 occupancy の kernel では「同じ順序で WG を増やす」ことが最も安い高速化であり、
演算そのものを削っても意味は無い。判定順序は次の通り:

1. kernel の時間が M（行数）に依存しない → latency / occupancy 律速
2. grid が WG 数として小さい（数十）→ work partition 変更で直る
3. 単体 kernel の rows 掃引が平坦 → launch floor、内部最適化は無価値

### 計測コマンド

```bash
./build/phaseshift-bench activation-quantize --variant e4m3-vec --rows 8,128 --k 5120 \
  --samples 30 --warmup 30
./build/phaseshift-bench elementwise --variant optimized --profile residual \
  --features 5120 --rows 2,8,128 --samples 30 --warmup 30
./build/phaseshift-bench rmsnorm --variant optimized --input-dtype bf16 \
  --output-dtype bf16 --weight-dtype bf16 --mode direct --features 5120 \
  --rows 8,128 --samples 30 --warmup 30
```


---

## 7.100 INT2 coarse draft head の pool 選定と、近似源を exact 側に揃える原則（2026-09-23）

DFlash2 proposer の full PSQ8 lm_head（2.2ms）を INT2 coarse（全語彙）→ coarse Top-N →
PSQ8 exact rerank → top16 に置き換えた。追加 resident は INT2 codes 303MiB のみで、
weight scale は target PSQ8 の BF16 / block32 をそのまま共有する。

### 一般則: coarse の近似源を exact 側と揃える

INT2 codebook は「元 BF16 model」ではなく **target PSQ8 head そのもの**から導出し、
block scale も PSQ8 のものを共有する。すると coarse の誤差は codebook 近似だけになり、
rerank は同一 weight の exact 計算になるため coarse top-N と rerank top16 の整合が取れる。

| pool | top1 coverage | avg top16 recall | selector coverage | proposal token match |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.9107 | 0.5921 | 0.8036 | 0.6875 |
| 32 | 0.9821 | 0.7606 | 0.8750 | 0.8036 |
| 64 | 0.9911 | 0.8677 | 0.9464 | 0.8393 |
| 128 | 1.0000 | 0.9470 | 0.9911 | 0.8929 |

（prose, real hidden, full PSQ8 top16 を oracle とする実測）

### pool は recall ではなく E2E と head 費用で選ぶ

head は pool にほぼ線形に高くつく（topN が pool 線形）: pool16 1.267ms / pool32 1.377ms /
pool48 1.50ms / pool64 1.64ms / pool80 1.85ms / pool128 2.50ms
（full PSQ8 head は 2.32ms。pool128 は full より遅い）。

E2E（prose K7 256, interleaved 3 対, median）: full 63.80 / pool32 65.00 / pool80 64.29
tok/s。code +3.5%、ctx2048 +2.2%（pool32）。ids と acceptance は全 pool で full と一致。
→ **pool32 を既定**（recall の余裕と head 費用のバランス）。

### standalone と in-context の乖離（Gate 11H の counter-example と対比）

coarse kernel は standalone で DRAM floor の 63%（405GB/s）しか出ていないが、
head total 1.68x は E2E でも再現した（+1.3-1.9%）。一方 Gate 11H の BF16 ExactRows は
standalone -24% が in-context で消えた。**kernel 単体の数字だけで採否を決めない。**


---

## lm_head candidate proxy（INT2 coarse + PSQ8 exact rerank）— 正本

詳細は [lm_head_int2_coarse_psq8_rerank_poc.md](../lm_head/lm_head_int2_coarse_psq8_rerank_poc.md)。

- 設計: `final_norm → INT2 coarse lm_head（枝刈り）→ coarse Top-N → original PSQ8 candidate
  exact rerank（BF16 相当）→ candidate argmax`。**certificate は使わない。**
- 近似源は INT2 枝刈りのみ。候補集合に入っていれば production の PSQ8 head を再現する。
  実測で `containment == rerank_match` が全 pool で一致し、rerank が候補集合内で exact で
  あることを確認した。
- **candidate 選択は必ず `coarse` を使う。** `upper`（certificate 用の上界）を候補選択に
  使うと real hidden で mismatch 0.72 まで悪化する（実測）。`upper` / `error_l2` /
  `activation_l2` / `omitted upper` は candidate path では計算しない。

### Target（204K corpus: train 137,944 / val 29,593 / qual 29,593 rows）

| 指標 | N=16 | N=32 | N=64 | N=127 |
| --- | ---: | ---: | ---: | ---: |
| containment（train） | 0.994679 | 0.999000 | 0.999833 | 0.999993 |
| mismatch（train） | 734 | 138 | 23 | 1 |
| containment（val） | 0.990876 | 0.998243 | 0.999628 | 1.000000 |
| containment（qual） | 0.993208 | 0.998479 | 0.999763 | 0.999932 |

- closed-loop（greedy、12 prompt × 96 token、N=32 / N=64）: divergent **0 / 1,152**、
  proxy local mismatch 0。
- isolated performance（real hidden、median）: rows=1 で N=32 は total 1000.7 us 対
  full 2144.4 us（**2.14x**）、N=64 で 2.03x、N=127 で 1.88x。rows=8 でも N=32 が 1.98x。
- coarse head が total の 82〜92% を占め、実効帯域は 348 GB/s（ピーク 636 GB/s の 55%）。
  これが残る最大の改善余地。
- N の測定上限は 127（`kDflash2Int2MaxPool = 128`）。N=128 以上は partition kernel の
  LDS 容量を増やす必要があり occupancy を悪化させる。

### Verify

- real verify hidden 7,168 row で **mismatch 0 / false accept 0 / false reject 0**
  （95% CI 上限 0.042%）。
- 実 DFlash2 draft（`Qwen3.8-27B-DFlash2-PSQ4`）+ INT2 draft head（pool=32）の E2E:
  rounds=76、mean accepted **2.355** / 7、emitted/round **3.355**、71.84 tok/s。

### 不採用の記録（履歴）

- **certificate（上界による全域証明）: NO-GO**。real target 12,032 decisions、pool 16〜127、
  margin A〜E のすべてで cert_rate = 0。L2 Cauchy–Schwarz 上界が次元 K=5120 に対し
  sqrt(K)≈72 倍緩い（bound/actual ≈ 100）。cert_rate=0 では全行 fallback となり、
  isolated では full に対し **0.65x**（proxy + full を払う）。
- **residual-corrected certificate: NO-GO**。`D = W_psq8 - W_int2` は sparse でも
  W32 group 集中でも low-rank でもない（low-rank r=64 で residual energy 0.19%、白色雑音）。
  corrected upper でも cert_rate は全方式 / 全 pool で 0（bound slack 75.7 → 50.1）。
- **learned PSQ2 / Rank Shape16: 不採用（production は INT2）**。204K corpus では INT2 が
  N=16 で約 99%、N=24〜32 で 99.6〜99.8%。Shape の明確な優位は N=4〜8 のみで N>=16 では
  飽和し、Global Rank4 は INT2 とほぼ同等。PSQ2 Shape は coarse 段を複雑化するため
  production は INT2 を本命とし、PSQ2 は N<16 用の R&D 切り口として残す。
  `tools/psq2` / `test_psq2_ranking_proxy` は削除済み。
- INT2 kernel 効率化（coarse head W=16、topn の branchless key + cursor + bitonic）と
  負の結果（full-tile 特化 / FP8 scale）は §7.100 と正本 §11 を参照。
