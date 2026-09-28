# Qwen3.8-Flash-Next Gate 0 / Gate 0.5 記録

> Status: R&D record（Gate 0 / Gate 0.5 の検証と採否判断。現在の contract は
> `docs/developer/qwen4exp.md`、現在の性能値ではない）

## 1. 目的と範囲

Qwen3.8-Flash-Next (`qwen4_exp`) を PhaseShift に載せる前段として、
「何を計算するモデルか」をコードを書く前に固定し (Gate 0)、
R9700 x2 で重みが入るかを model code 着手前に判定する (Gate 0.5)。

今回の範囲は architecture inspection・tensor manifest・documentation・
小規模 utility のみである。QSA / MoE / PLE kernel と Qwen4Exp executor は追加しない。

## 2. revision

| 項目 | 値 |
| --- | --- |
| PhaseShift baseline | `b60a83753b8ae2ba422e9af08be79650915df64f` |
| worktree | `.worktrees/qwen4exp-gate0` (branch `poc/qwen4exp-gate0`) |
| model checkpoint | `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| reference implementation | Transformers `main` `src/transformers/models/qwen4_exp/` |
| reference commit | `f324707307757d9c0b8dac1c4462eceff911fa2f` (2026-09-23) |
| 取得日 | 2026-09-26 |

参照実装の確認点は `docs/references/qwen4exp.md`、contract は
`docs/developer/qwen4exp.md` に記録した。

## 3. 方法

`tools/inspect_qwen4exp.py` を追加した。
`model.safetensors.index.json` と各 shard の safetensors header のみを読み
(HTTP Range、GPU allocation なし)、tensor name / dtype / shape / shard を列挙する。

```sh
python3 tools/inspect_qwen4exp.py check
python3 tools/inspect_qwen4exp.py inventory
python3 tools/inspect_qwen4exp.py budget
python3 tools/inspect_qwen4exp.py manifest --out tensor_manifest.json
```

`check` は config から required tensor を生成し、checkpoint と照合する
(dry-run weight resolver)。

## 4. tensor inventory

checkpoint: 131 shard、tensor 1658、`total_size` 359999963128 byte。

`check` の結果:

```text
present=1658
required=1294 missing_required=0
optional_present(mtp)=31
ignored(vision)=333
unknown=0
RESULT: OK
```

- text backbone required: 1294
- MTP (optional): 31 tensor
- vision (ignored): 333 tensor
- 分類不能 tensor: 0

tensor 総数を超える required はない。全 text tensor を 15 category に分類した。
pattern と shape の全列挙は `docs/developer/qwen4exp.md` section 12 にある。

## 5. REUSE / ADAPT / NEW matrix

| feature | 判定 | 根拠 |
| --- | --- | --- |
| token embedding | REUSE | `EMBEDDING_LOOKUP`。feature 方向 `hc_count` 複製のみ追加 |
| RMSNorm (`1+w`, grouped) | REUSE | `RmsNormNode.group_size` と `ONE_PLUS` |
| RoPE (partial 64, theta 1e7) | REUSE | `RoPENode.rotary_dim`。text では mRoPE は no-op |
| dense linear | REUSE | `LinearNode`。既存 format をそのまま使う |
| GDN conv / recurrence / l2norm | ADAPT | 既存 GDN primitive。geometry と head repeat の差分 |
| GDN output gated norm | ADAPT | 既存は silu、qwen4exp は sigmoid |
| gated query attention (QSA main) | ADAPT | `q_proj` 2x + sigmoid gate と partial rotary は既存 |
| sparse attention mask | NEW | selected-token mask を causal mask に合成 |
| QSA indexer | NEW | index_qk_proj / block pooling / block topk / indexer key cache |
| MoE router | NEW | fp32 softmax、top-10、再正規化 |
| routed expert GEMM | NEW | fused 3D `[E, 2I, H]` / `[E, H, I]`、grouped GEMM |
| shared expert | ADAPT | 通常 MLP は既存、`shared_expert_gate` (sigmoid) は NEW |
| Gated Residual / HyperConnection | NEW | read gate / branch ごと write gate / 4-stream |
| PLE n-gram embedding | NEW | host-resident table、hash、gather |
| PLE projection / gate / conv | NEW | gated value + dilated depthwise conv |
| lm_head | REUSE | `LinearNode`。final norm なし |
| MTP | NEW | reference forward 未実装 |
| vision | IGNORE | text support と別 Gate |

## 6. current PhaseShift との gap

`docs/developer/qwen4exp.md` section 14 に 11 項目を記載した。
実装ブロッカーとして重要な順に:

1. MoE primitive / weight / model code が存在しない (全 48 層)。
2. fused 3D expert tensor の loader / quantizer role がない。
3. QSA indexer primitive と indexer key state がない。
4. sparse attention mask がない。
5. Gated Residual (multi-stream residual) がない。
6. PLE がない。
7. PSQ3 (3.5bpw) weight format が存在しない (memory budget のブロッカー)。現行は
   PSQ4 (4.5bpw) 以上のみ。
8. GDN output gate が silu 固定。
9. MTP の正本 forward がない。
10. config / weights は dense 前提。

再利用できるもの: GDN conv / recurrence / l2norm、grouped RMSNorm (`ONE_PLUS`)、
partial RoPE、gated query attention、通常 MLP、`LinearNode` / `EMBEDDING_LOOKUP`。

## 7. memory budget (Gate 0.5)

parameter 数と、format ごとの byte 数 (scale overhead 込み)。
1 次元パラメータ (norm / `A_log` / `dt_bias`) は BF16 固定とした。
format の payload density は `docs/developer/psq_canonical_soa_payload.md` に従う
(BF16 = 2 B/w、PSQ8 = 8.5 bpw、PSQ4 = 4.5 bpw、PSQ3 = 3.5 bpw、
MXFP4 = 4.25 bpw、FP8 block128 ≈ 8 bpw)。

### 7.1 category 別

| category | params (B) | BF16 | PSQ8 | PSQ4 | PSQ3 | MXFP4 | FP8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| token_embedding | 0.636 | 1.27 | 0.68 | 0.36 | 0.28 | 0.34 | 0.64 |
| lm_head | 0.636 | 1.27 | 0.68 | 0.36 | 0.28 | 0.34 | 0.64 |
| routed_experts | 120.796 | 241.59 | 128.35 | 67.95 | 52.85 | 64.17 | 120.83 |
| shared_expert | 0.236 | 0.47 | 0.25 | 0.13 | 0.10 | 0.13 | 0.24 |
| router | 0.063 | 0.13 | 0.07 | 0.04 | 0.03 | 0.03 | 0.06 |
| gdn | 2.087 | 4.17 | 2.23 | 1.18 | 0.92 | 1.11 | 2.13 |
| qsa_attention | 0.598 | 1.20 | 0.64 | 0.34 | 0.26 | 0.32 | 0.60 |
| qsa_indexer | 0.020 | 0.04 | 0.02 | 0.01 | 0.01 | 0.01 | 0.02 |
| hyper_connection_layer | 0.634 | 1.27 | 0.67 | 0.36 | 0.28 | 0.34 | 0.70 |
| hyper_connection_final | 0.007 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.01 |
| ple_ngram_table | 51.200 | 102.40 | 54.40 | 28.80 | 22.40 | 27.20 | 81.94 |
| ple_other | 0.033 | 0.07 | 0.04 | 0.02 | 0.01 | 0.02 | 0.03 |
| mtp | 2.607 | 5.21 | 2.77 | 1.47 | 1.14 | 1.39 | 2.61 |
| vision (ignored) | 0.449 | 0.90 | 0.48 | 0.25 | 0.20 | 0.24 | 0.46 |
| **total** | **180.000** | **360.00** | **191.27** | **101.26** | **80.80** | **95.63** | **210.90** |

単位は GB。値は 1e9 byte 換算である。

### 7.2 residency 別

| residency | params (B) | BF16 | PSQ8 | PSQ4 | PSQ3 | MXFP4 | FP8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main_backbone (no PLE/MTP/vision) | 125.711 | 251.42 | 133.58 | 70.72 | 55.00 | 66.79 | 125.85 |
| main_plus_mtp | 128.318 | 256.64 | 136.35 | 72.19 | 56.15 | 68.18 | 128.46 |
| routed_experts_only | 120.796 | 241.59 | 128.35 | 67.95 | 52.85 | 64.17 | 120.83 |
| ple_table_only | 51.200 | 102.40 | 54.40 | 28.80 | 22.40 | 27.20 | 81.94 |
| ple_other_only | 0.033 | 0.07 | 0.04 | 0.02 | 0.01 | 0.02 | 0.03 |

### 7.3 判定

- R9700 x1 = 32 GB。main_backbone は最小 format (PSQ3) でも 55.00 GB。
  routed experts だけでも 52.85 GB。x1 常駐は全 format で不可。
- R9700 x2 = 64 GB。main_backbone は PSQ4/MXFP4 で 66.8～70.7 GB > 64 GB。
  PSQ3 では 55.00 GB となり総量では収まる。
- expert-parallel (EP=2) で experts を半分ずつ持つ場合:
  - PSQ4: 33.98 GB/GPU → 予算超過。
  - MXFP4: 32.09 GB/GPU → 予算超過。
  - PSQ3: 26.48 GB/GPU → 非 expert と state を含めて境界内。
- したがって 4-bit 以上では R9700 x2 に main model が入らない。
  これは Gate 0.5 の NO-GO 条件に該当し、PSQ3 への方針変更で解消する。

### 7.4 expert offload 変種 (FreeToken 方式)

expert の一部を host RAM に置き、VRAM 常駐を `(1 - offload)` とする。
EP=2 で常駐 expert を 2 GPU に分散する。
この変種は KV / context を大きく伸ばすための lever である。

前提: experts PSQ3 (合計 52.95 GB)、non-expert / embedding / lm_head PSQ8、
1 次元 BF16、usable 30.5 GB/GPU、workspace 2.0 GB と GDN state 0.12 GB を確保、
KV は QSA 12 層 + indexer のみ。

KV の 1 token あたりサイズ:

| KV / indexer | B/token | 256K context の KV |
| --- | ---: | ---: |
| BF16 / BF16 | 27648 | 7.25 GB |
| FP8 / BF16 | 15360 | 4.03 GB |
| FP8 / FP8 | 13824 | 3.62 GB |

(indexer key: 12 層 × 1 head × 128 dim。GDN は固定 state であり伸びない。)

offload 量ごとの per-GPU 予算と最大 context:

| offload | weights/GPU | KV budget | BF16 KV 最大 | FP8 KV 最大 |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 29.79 GB | -1.41 GB | 0 | 0 |
| 12.5% | 26.48 GB | 1.90 GB | 67k tok | 121k tok |
| **25%** | **23.17 GB** | **5.21 GB** | **184k tok** | **331k tok** |
| 37.5% | 19.86 GB | 8.52 GB | 301k tok | 542k tok |
| 50% | 16.55 GB | 11.83 GB | 418k tok | 752k tok |
| 75% | 9.93 GB | 18.45 GB | 652k tok | 1.17M tok |

- offload 0% では KV budget が負になり、32 GB に weight + state + workspace は入らない。
- **offload 25% + FP8 KV** で KV は約 5.2 GB となり、native 262144 context
  (4.03 GB) が収まる。BF16 KV では約 184k token が上限である。
- 1M context (FP8 KV で約 16.1 GB) には offload 75% が必要である。

host RAM の追加要求は expert の 25% で約 13.2 GB である。
PLE table は SSD 常駐とする (section 7.6)。

decode コスト: 一様 routing を仮定すると 1 層あたり 10 × 0.25 = 2.5 個、
48 層で 120 expert が host から来る。PSQ3 expert 1 個は約 2.15 MB なので
**258 MB/token** を host→device へ移す。

- 直列換算: 258 MB / 55 GB/s = **4.69 ms/token** (plain step の約 51%)。
- 層単位で prefetch すれば 1 層あたり 5.38 MB = 0.098 ms であり、
  層の compute 0.192 ms の裏に隠せる。
- したがって offload 率は「層単位 prefetch を実装できるか」との tradeoff である。
  prefetch が無いと decode は大幅に落ちる。

### 7.5 offload した expert を CPU 推論する場合

offload 分を host へ転送せず CPU で計算する。PCIe の weight 転送 258 MB/token は
消え、GPU <-> CPU のやり取りは hidden と出力だけになる
(M=1 で 1 層あたり約 18 KB、token 全体で約 0.9 MB)。

前提: 1 expert = 9.83 MFLOP / 2.15 MB、host RAM 80 GB/s、
GPU <-> CPU sync 0.02 ms/層。GPU と CPU は同一層内で並列に走ると仮定し、
層時間 = max(GPU, CPU) + sync とする。

plain decode (M=1):

| offload f | GPU ms/層 | CPU ms/層 (200 GFLOP/s) | plain tok/s (CPU 100 / 200 / 400 GFLOP/s) |
| ---: | ---: | ---: | ---: |
| 0.125 | 0.196 | 0.095 | 96.5 / 96.5 / 96.5 |
| 0.25 | 0.192 | 0.190 | 62.6 / 98.4 / 98.4 |
| 0.5 | 0.183 | 0.380 | 32.3 / 52.1 / 75.1 |

- f=0.125 では CPU が GPU の裏に隠れ、plain は約 96 tok/s (sync 分の tax 約 9%)。
- f=0.25 は CPU 200 GFLOP/s 以上で GPU-bound の約 98 tok/s、100 GFLOP/s では
  CPU-bound で約 63 tok/s。
- sync 0.02 ms/層が token あたり約 1 ms 乗る。fused schedule で削る必要がある。

MTP K=2 (M=3, expert union 29.4/10 = 2.94):

| offload f | tok/s (CPU 100 / 200 / 400 GFLOP/s) |
| ---: | ---: |
| 0.125 | 79.3 / 120.6 / 130.4 |
| 0.25 | 42.4 / 66.8 / 93.9 |
| 0.5 | 21.9 / 35.3 / 50.8 |

- verify の union 増大が CPU 側に乗るため、MTP ありでは offload を下げる必要がある。
  f=0.125 + CPU 400 GFLOP/s で約 130 tok/s が上限側。
- f=0.25 では CPU-bound になり、plain より遅くなる。

留意:

- host RAM bandwidth が CPU 側の下限を作る (f=0.25 で 5.4 MB/層)。
- PhaseShift の CPU 実装は `phaseshift_quant_reference` (reference math) のみで、
  最適化された CPU expert kernel は存在しない。新規実装が必要である。

100 tok/s を出すのに必要な host 側リソース (M=1):

| offload f | offloaded expert read / token | 必要 CPU GFLOP/s | 必要 host GB/s |
| ---: | ---: | ---: | ---: |
| 0.125 | 60 | 59 | 12.9 |
| 0.25 | 120 | 118 | 25.8 |
| 0.5 | 240 | 236 | 51.6 |

- **PCIe 転送方式は CPU 性能を要求しない**。PCIe 5.0 x16 (実効 55 GB/s) に対し
  f=0.25 で 25.8 GB/s であり、層単位 prefetch で足りる。通常の CPU で成立する。
- CPU 推論方式は CPU GFLOP/s が律速である。f=0.125 なら現行 desktop CPU でも
  隠せるが、f=0.25 以上や MTP union (M=3 で約 3x) では
  Threadripper / EPYC 級 (多コア + 多チャネル DDR5) が現実的である。
- host RAM 容量は PLE table を SSD 常駐にすることで解消する (section 7.6)。
  残る host RAM は offload expert (f=0.25 で 13.2 GB) + runtime で 32～64 GB に収まる。
- まとめ: FreeToken 本来の PCIe 転送 + PLE SSD は普通の CPU / 通常 RAM で成立する。
  「offload 分を CPU 推論」にすると CPU 依存になり、f を上げるほど
  high-end CPU が必要になる。

### 7.6 PLE を SSD 常駐にする場合

PLE は 51.2B パラメータ (BF16 で 102 GB) であるが、1 token あたりの参照は
`ngram_heads = 16` row × 160 dim × BF16 = **5.0 KiB** だけである。

| 局面 | 参照量 | rate |
| --- | ---: | ---: |
| decode 100 tok/s | 5.0 KiB/token | 0.51 MB/s、1600 IOPS |
| prefill T=2048 | 32768 row | 10.5 MB (1 回) |
| prefill T=8192 | 131072 row | 41.9 MB (1 回) |

- decode の NVMe 負荷は 0.5 MB/s であり、latency も layer 0 compute (0.19 ms) の
  裏に隠せる。n-gram ID は input token 履歴だけで決まるため、token 生成直後に
  prefetch できる。
- prefill は 1 回 10～42 MB であり、NVMe でも無視できる。
- したがって PLE table に host RAM 102 GB は不要である。SSD mmap + page cache で
  成立し、hot n-gram は page cache が自然に吸収する。
- GPU 常駐・host RAM 常駐のどちらも不要であり、residency budget から除外できる。

## 8. 推奨 weight format (Gate 0.5 の決定)

Qwen3.8-Flash-Next は **PSQ3 (3.5bpw) と PSQ8 (8.5bpw)** で構築する。

**最初の実モデル weight configuration**:

| component | params (B) | format | 配置 |
| --- | ---: | --- | --- |
| routed experts + shared expert | 121.03 | **PSQ3 (3.5bpw = 3-bit code + BF16/W32 scale)** | EP=2 で 2 GPU に分散 |
| non-expert linear (GDN / QSA / indexer / router / HC) | 3.65 | **PSQ8 (8.5bpw)** | 層 partition で分割 |
| embedding + lm_head | 1.27 | **PSQ8** | 複製 |
| norm / conv / `A_log` / `dt_bias` | ~0.07 | BF16 | 複製 |
| PLE n-gram table | 51.20 | SSD mmap (BF16) | GPU / RAM 常駐しない |
| MTP | 2.61 | PSQ3 (Gate 10 まで除外) | — |

per-GPU の重み:

| 内訳 | GB/GPU |
| --- | ---: |
| experts PSQ3 (EP=2) | 26.48 |
| non-expert PSQ8 (層 partition) | 1.81 |
| embedding + lm_head PSQ8 (複製) | 1.35 |
| norm / 1-D BF16 | ~0.15 |
| **weight 合計** | **~29.8** |

state (`fp8` KV / GDN recurrence) と workspace を加えて約 31～33 GB/GPU。
32 GB に収めるには次を前提とする:

- KV cache を FP8 にする。
- MTP を除外する (Gate 10 まで)。
- PLE table を SSD 常駐にする。
- 初回 bring-up は 32K 程度の context に限定する (262K は x2 では不可)。

余裕を増やす場合の lever は non-expert の PSQ3 化であり、
そのとき weight は約 27.8 GB/GPU になる。
ただし non-expert は本 directive では PSQ8 とする。
KV / context を伸ばす lever は expert offload であり、section 7.4 に試算を置く
(expert 25% offload + FP8 KV で native 262K context が収まる)。

副次的な方針:

- PLE table は GPU 常駐を前提にしない。SSD mmap + selected row gather を第一候補とする
  (section 7.6)。1 token あたり 16 row = 5.0 KiB であり、host RAM 102 GB は不要。
- MTP は最初の residency budget から除外する。
- PSQ3 format は未実装である。Gate 7 で codebook / scale layout / kernel を
  決めるまで R9700 x2 の実 weight residency は成立しない。PSQ8 は既存。

これは quantization 方針の変更である。Gate 1 以降の model code は
BF16 の tiny synthetic config で correctness を進められるため、
この変更は Gate 1 をブロックしない。

### 8.1 机上 decode 速度推測 (未計測)

batch=1 greedy decode を bandwidth-bound と仮定した roofline。
R9700 の実測 DRAM 636 GB/s、weight は 1 token に 1 回参照される。

1 token あたり weight traffic (expert 1/4 offload 後):

| component | activated (B) | format | GB/token | 比率 |
| --- | ---: | --- | ---: | ---: |
| routed experts (10/512) | 2.359 | PSQ3 | 1.032 | 18% |
| shared expert | 0.236 | PSQ3 | 0.103 | 2% |
| GDN (36 層) | 2.087 | PSQ8 | 2.218 | 38% |
| QSA + indexer (12 層) | 0.618 | PSQ8 | 0.657 | 11% |
| HyperConnection | 0.641 | PSQ8 | 0.681 | 12% |
| router | 0.063 | PSQ8 | 0.067 | 1% |
| embedding + lm_head | 1.272 | PSQ8 | 1.351 | 22% |
| **合計** | — | — | **5.851** | 100% |

offload 前は 6.109 GB/token であり、1/4 offload で VRAM traffic は 5.851 GB/token
(約 4% 減) になる。host 転送は 258 MB/token (4.69 ms 直列) であり、
層単位 prefetch で隠せるかが分岐点になる。

| 条件 | ms/token | ideal tok/s |
| --- | ---: | ---: |
| single-stream (x1 相当) | 9.20 | 109 |
| x2 layer partition (逐次) | 9.20 | 109 |
| x2 EP=2 (attention 複製) | 8.51 | 118 |
| x2 TP=2 (全行列分割、通信除外) | 4.60 | 217 |

現行 Qwen3.8-27B-PSQ の tg128 実測 29.36 tok/s と PSQ4 の roofline 41.9 tok/s から
実効率は約 0.70 である。これを当てると:

- x2 layer partition: 約 76 tok/s。
- x2 TP/EP: 理想 118～217 に対し MoE all-to-all / reduce の分だけ低下。約 90～150 tok/s。
- 未融合 (expert ごとに launch、48 層 × 約 15 dispatch、2 us floor) の bring-up では
  追加で 1 ms 超となり、約 40～70 tok/s。

長 context の追加 traffic (indexer が全 key を scan する):

| context | 追加 GB/token | base 比 |
| ---: | ---: | ---: |
| 32K | 0.151 | +2.5% |
| 256K | 0.856 | +14% |
| 1M | 3.272 | +54% |

所見:

- decode traffic の 59% は非 expert (GDN/QSA/HC) PSQ8 であり、expert は 19% に過ぎない。
  **decode 速度に対しては non-expert の精度選択の方が支配的**である。non-expert を
  PSQ3 にすると base は約 3.18 GB/token となり、ideal は約 200 tok/s に上がる。
- expert 1/4 offload は VRAM traffic を約 4% 減らすが、host 転送 258 MB/token を
  層単位で prefetch して隠せることが条件である。隠せなければ 4.69 ms/token が
  直列に乗り、plain decode は約 2/3 に落ちる。
- 256K context でも長文 read の追加は約 14% に留まる。GDN が固定 state のため
  context 伸長の主コストは indexer の全 key scan である。

これは机上推測であり、実測ではない。確定には Gate 9 の計測が要る。

### 8.2 MTP ありの机上 decode 速度推測 (未計測)

MTP を native drafter として greedy speculative decode する。
draft は 1 step ごとに target の lm_head を共有する
(27B 実測で draft 1 step の 55～58% が lm_head の weight read)。
acceptance は Qwen3.5 MTP の実測 (27B Gate 4E) を流用し、emitted/update を
K1 1.60 / K2 2.01 / K4 2.52 / K8 2.58 とする。

MoE では verify が M = K+1 行を同時に処理し、**選択 expert の union** を読む。
512 expert / top-10 では M 行の distinct expert 数は
`512 x (1 - (1 - 10/512)^M)` で増える:

| M | 1 | 2 | 3 | 5 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| distinct | 10.0 | 19.8 | 29.4 | 48.1 | 83.3 |
| vs M=1 | 1.0x | 2.0x | 2.9x | 4.8x | 8.3x |

verify traffic = 非 expert 等 5.076 GB (M に依らず 1 回) + `union/10 x 0.774 GB`
(VRAM 常駐 expert)。draft = `K x 0.731 GB` (lm_head 0.676 + MTP 層 0.055)。
expert offload の host 転送は層単位 prefetch で隠せると仮定し、時間に含めない。

| K | verify GB | draft GB | ms/update | emitted | tok/s | vs plain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (plain) | 5.85 | 0.00 | 9.20 | 1.00 | 108.7 | 1.00x |
| 1 | 6.61 | 0.73 | 11.54 | 1.60 | 138.8 | 1.28x |
| 2 | 7.35 | 1.46 | 13.86 | 2.01 | **145.1** | **1.33x** |
| 4 | 8.80 | 2.92 | 18.43 | 2.52 | 136.5 | 1.26x |
| 8 | 11.52 | 5.85 | 27.31 | 2.58 | 94.5 | 0.87x |

**K=2 が最適で約 1.33x**。K を伸ばすと expert union と draft lm_head が増えて逆効果。

x2 の並列化を当てると、plain 80～150 tok/s に対し MTP ありは
**約 100～190 tok/s**、未融合 bring-up では 50～90 tok/s。

留意:

- 27B (dense) の native MTP 実測は break-even、DFlash2 (専用 drafter) は 2.4x。
  Flash-Next は 6B activated の bandwidth-bound のため dense 27B より利得は出やすいが、
  MoE expert union と draft lm_head が上限を作る。
- draft lm_head を candidate selection で圧縮すると draft は 0.731 GB/step から
  約 0.15 GB/step に落ち、K=4 の利得も上がる (将来 lever)。
- MTP は Qwen4Exp では reference forward が未実装 (section 10) であり、acceptance と
  draft 構成は未確定である。本推測は Qwen3.5 MTP の acceptance を仮定した上限側の目安。
- offload を CPU 推論にした場合、verify の expert union が CPU に乗るため
  MTP ありでは offload 率を下げる必要がある (section 7.5)。f=0.125 + 強い CPU で
  約 130 tok/s、f=0.25 では CPU-bound で plain を下回る。

## 9. Gate 1 の具体的変更ファイル候補

Gate 1 は config parser / skeleton / weight metadata / dry-run resolver のみで、
全 weight を GPU にロードしない。

新規 (qwen4exp namespace):

- `include/phaseshift/models/qwen4exp/model/qwen4exp_config.h`
- `src/phaseshift/models/qwen4exp/model/qwen4exp_config.cpp`
- `include/phaseshift/models/qwen4exp/model/qwen4exp_model.h`
- `src/phaseshift/models/qwen4exp/model/qwen4exp_model.cpp`
- `include/phaseshift/models/qwen4exp/weights/model_weights.h`
- `src/phaseshift/models/qwen4exp/weights/model_weights.cpp`
- `tests/unit/test_qwen4exp_config.cpp` (real config.json の主要 field assert)
- `tests/unit/test_qwen4exp_weight_resolver.cpp` (dry-run resolver)

更新:

- `cmake/targets.cmake` — `phaseshift_qwen4exp` target 追加
- `cmake/tests.cmake` — 上記 test の required 登録
- `docs/developer/README.md` — qwen4exp contract へのリンク
- `docs/rnd/README.md` — 本記録へのリンク
- `docs/references/README.md` — 外部参照へのリンク

既存変更なし:

- `Qwen35TextConfig` に qwen4exp field を追加しない。
- `qwen35` namespace / directory を rename しない。

`tools/inspect_qwen4exp.py` は今回追加済みで、Gate 1 の resolver の期待値生成に使う。

## 10. 未確定事項

1. MTP forward。`pre_fc_norm_hidden` が `[10240]`、`fc_hidden` が `[2560, 2560]` で
   あり、4-stream 入力の扱いが確定していない。Transformers は `^mtp.*` を無視する。
   Gate 10 で Qwen 実装 / tech report を追跡する。
2. QSA の数値契約。top-k tie の安定順序、block 整列、tail の扱い、
   indexer key cache の dtype を fixture で固定する。
3. grouped RMSNorm の PhaseShift mapping。`group_size = hidden_size` の
   `ONE_PLUS` が既存 `RmsNormNode` でそのまま表現できるかの検証。
4. QSA state の `PrimitiveStateKind` 表現。indexer key cache は新規 kind 候補。
   PLE n-gram context はトークン ID を保持するため conv state とは dtype が異なる。
5. PLE host gather の throughput。decode batch=1 で必要 row 数と
   staging latency を測る (Gate 3)。
6. PSQ3 (3.5bpw) weight format の design。codebook / scale layout / kernel は未定。
7. multi-GPU expert-parallel の実現性 (Gate 8/9)。semantics が単 GPU で
   一致するまで着手しない。
8. vision tower (Gate 11)。今回は ignored のままとした。

## 11. GO / NO-GO

| Gate | 判定 | 理由 |
| --- | --- | --- |
| Gate 0 | **GO** | config 値が reference と一致。全 text tensor を分類。missing required = 0、unknown = 0。REUSE/ADAPT/NEW 決定済み |
| Gate 0.5 | **条件付き GO** | 4-bit 以上では R9700 x2 に main model が入らない。expert を PSQ3 (3.5bpw)、non-expert / embedding / lm_head を PSQ8 (8.5bpw) にする方針変更で成立する。PSQ3 format を Gate 7 までに用意する |
| Gate 1 着手 | **GO** | model code は BF16 synthetic config の correctness から始められ、quantization 方針変更にブロックされない |

Gate 0 acceptance:

- config 値が reference と一致: 満たす
- 全 text-model tensor を分類: 満たす (unknown = 0)
- tensor shape 不明が 0: 満たす
- operator 不明が 0: 満たす (MTP を除く。MTP は section 10-1 に明示)
- REUSE / ADAPT / NEW 決定済み: 満たす
- VRAM / RAM 概算表が存在: 満たす (section 7)
- production kernel 未追加: 満たす
