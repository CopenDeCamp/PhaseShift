# NgramTail Gate 3 — production 統合

> Status: R&D record（Gate の経緯・採否判断・計測）。現在の contract は
> `docs/developer/dflash2.md` と `docs/user/compute.md`、現在の性能値は
> `docs/perf/current.md` を正本とする。

## 0. 結論

Gate 2（[ngram_tail_gate2.md](ngram_tail_gate2.md)）の推奨構成
`DFlash2 K=7 + NgramTail T=8 / n=5 / window=2048 / Exact` を **production option として
統合**した。ただし **既定は 0（opt-in）**である。

production の挙動を実際に変えたのは次の 2 点である。

1. target lm_head proxy を DFlash2 有効時に停止（env 未指定時のみ）。**これが既定の変更**
2. GDN history guard を 2.25 GiB へ引き上げ、NgramTail 有効構成（`K=7 + T=8` = 15 row）を
   history path で動かせるようにした。NgramTail 無効の既定では確保量は従来どおり 1.00 GiB

NgramTail を既定 ON にしなかった理由は §4.2 の計測である。production 条件
（history mode）での 8 workload paired A/B で **6 workload が 0.6〜3.9% 悪化、平均 -1.10%** だった。
Gate 2 の GO は「json PP512 のみ」の条件付き GO であり、勝つ workload は corpus / model に依存する。

対象外: `n = 8`（json のみ +6.76%）、`T >= 16`（不採用）、continuous batch による
DFlash2 同時リクエスト、GPU n-gram。

## 1. Gate 2 から引き継ぐ結論

| 項目 | 結論 |
| --- | --- |
| 推奨構成 | `K=7 + T=8 / n=5 / window=2048`、Exact verify |
| 不採用 | `T >= 16`（json PP512 で +0.42%、他 workload は悪化） |
| GO の範囲 | json PP512 のみ +5.94%（n=8 で +6.76%）。残り 5 workload は -0.32〜-9.75% |
| 測定条件 | `PHASESHIFT_TARGET_LM_HEAD_PROXY=0`、`PHASESHIFT_DFLASH2_GDN_RERUN_REFERENCE=1` |

Gate 2B の性能値は **rerun reference mode** で測られている点が production 統合の最大の論点である（§2.1）。

## 2. 統合で解決した 3 点

### 2.1 GDN history guard の引き上げ（rerun mode へは切り替えない）

27B geometry の GDN history は 1 row = 153,944,064 B（146.8 MiB）で、旧 guard
1.5 GiB は **rows ≤ 10** しか許さない。`K=7 + T=8 = 15 row`（2.10 GiB）は guard を超えるため、
旧 guard のままでは NgramTail 有効時に decoder create が `insufficient_memory` で失敗する。

| 候補 | 判断 |
| --- | --- |
| rerun reference mode へ自動切替（= Gate 2B の条件） | 却下。rerun_ms は 22〜36 ms/round で round の 26〜37% を占め、production 現行の history path（rerun なし）から大きく劣化する |
| history rows を guard 上限 10 に抑え、深い partial だけ rerun に逃がす hybrid | 却下。毎 round の snapshot（slot 146.8 MiB）と稀な rerun 経路の分岐が必要で、未計測の新ロジックになる |
| **guard を 2.25 GiB へ引き上げ history path を維持** | **採用**。rows ≤ 15 = K7+T8 にちょうど収まり、既存 path をそのまま使う |

変更は `kDFlash2SpecHistoryBytesMax`（1.5 GiB → 2.25 GiB）と error message のみ。
超過は従来どおり create 時に明示 error（silent fallback なし）。
確保量は decoder 1 個あたり 1.00 GiB（NgramTail 無効）/ 2.15 GiB（T=8）。
DFlash2 は `max_concurrent_requests = 1` が強制されるため同時存在は 1 decoder 分のままである。

`K=7 + T=8 + anchor = 16` 行は **R16 bucket** で、`VerifyNumericMode::Exact` の保証範囲内に収まる
（Gate 2 の R64 gap は `rows >= 17` の話であり、採用構成には該当しない）。

### 2.2 target lm_head proxy を DFlash2 有効時に停止

Gate 2 は `PHASESHIFT_TARGET_LM_HEAD_PROXY=0` 前提で測定した（Gate 2 §4.1）。
production の既定は Verify のみ Fast proxy で、decode（M=1）と verify（M>1）で logits が
一致しないため生成列が target-only greedy から分岐する。

`phaseshift-compute` は DFlash2 有効時、env が未指定・空文字のときだけ `0` を設定して起動する。
明示指定した値は尊重する。library の mode 判定は変更していない
（`docs/developer/qwen35.md` §7.1 に例外として記載）。

コストは §4.1 の通り **-2.21%**（verify GPU 時間 +2.67%）。これは NgramTail ではなく
`GENERATED_IDS == target-only greedy` 契約を回復するための代償である。

### 2.3 option と validation

`--dflash2-ngram-tail` / `--dflash2-ngram-n`（ともに既定 **0** = 無効）を追加。
有効化は両方正の値を指定して行う。次の 3 種類は model load 前に exit 2:

- `--dflash2-model-dir` なしでの ngram option 指定
- tail / n のどちらか一方だけが 0
- `--dflash2-drafts + --dflash2-ngram-tail > 63`（verify capacity）

runtime 側も `create_dflash2_spec_decoder()` で同じペア条件を error として検証する
（片側だけ正の値は silent disable しない）。

`--dflash2-stats 1` は `DFLASH2_NGRAM_N` / `_TAIL` / `_WINDOW`、`DFLASH2_GDN_MODE`、
`DFLASH2_NGRAM_HIT_ROUNDS` / `_PROPOSED` / `_ACCEPTED`、
`DFLASH2_TAIL_REACHED_ROUNDS` / `_BLOCKED_ROUNDS` を追加出力する。

## 3. correctness

| 検証 | 条件 | 結果 |
| --- | --- | --- |
| Gate 2 harness `k7_t8`（新規 case） | history path、n=5、window 2048、GEN 256、4 category × ctx 512/2048 | **parity 8/8、failed=0**、hit 78 round、tail accepted 86、history bytes 2,309,160,960（15 row） |
| Gate 2 harness 既存 case（`k1_t6` / `k3_t4` / `k7_t0`） | 同上 | **parity 24/24、failed=0** |
| paired A/B（`k7_t0` + `k7_t8`、RUNS=3） | 同上 | **parity 48/48、failed=0** |
| required acceptance | `tests/run_required_acceptance.py` | **131/131 PASS、skip 0** |
| `test_dflash2_gate10_cli` | production binary vs target-only greedy | **token exact=yes、failed=0** |
| `phaseshift-compute` A/B | config A / B / C と target-only | `GENERATED_IDS` sha1 が全て一致 |

per-round assertion（`total_k == dflash_k + tail_k`、capacity 超過、future leakage、
context position invariant）は 0 違反。全 run が `gdn_mode=history`（`rerun_ms = 0`）。

harness 側に追加したもの: `k7_t8` case、proxy 既定の setenv（env 未指定時のみ `0`）、
control case で `ngram_n = 0` を明示（pairing 検証との整合）。

corpus は既存 `tests/data/mtp_perf/*.psktok`（468〜629 token）では ctx 2048 を満たせないため、
`tools/gen_token_corpus.py --base-psktok` で延長した。Gate 1 の corpus
（`docs/rnd/spec_decode/ngram_tail_gate1.md` §4）とは生成テキストが異なるため、
**Gate 1 / Gate 2 の数値との直接比較はできない**（§4.3）。

| file | tokens | sha256（先頭16桁） |
| --- | ---: | --- |
| `code.psktok` | 11,605 | `50e01ea02cc8ead9` |
| `json.psktok` | 7,055 | `21afb91ab793a799` |
| `prose.psktok` | 3,909 | `f62dc989ad0c7116` |
| `reasoning.psktok` | 128,193 | `f9ce5880224b77b2` |

## 4. 性能

計測条件: AMD Radeon AI PRO R9700（gfx1201）、`--device 1`、Release build、
target `models/Jackrong/Qwopus3.8-27B-Flash-V2-PSQ`、
draft `models/z-lab/Qwen3.8-27B-DFlash2-PSQ`、2026-09-29。

### 4.1 production 既定の変更（proxy 停止）

`phaseshift-compute`、prose prompt 31 token → 256 token、3 rep 交互（A→C→B / C→B→A / B→A→C）、median:

| config | 内容 | tok/s | verify GPU ms | rounds / accepted |
| --- | --- | ---: | ---: | --- |
| A | 現行既定（ngram 無効、proxy Fast） | **58.43** | 3779.1 | 94 / 161 |
| B | 新コード、ngram 無効（= 現行の production 既定） | **57.14** | 3879.9 | 94 / 161 |
| C | 新コード、`--dflash2-ngram-tail 8 --dflash2-ngram-n 5` | **57.11** | 3877.1 | 94 / 161 |
| target-only | 参照 | 28.37 | — | — |

- A → B = **-2.21%**（verify GPU +2.67%）。これは proxy 停止のみの差で、
  NgramTail は有効でない（`NGRAM_HIT_ROUNDS=0`）ため C と B は同値。
- 4 run すべて `GENERATED_IDS` sha1 = `1373b0e8357b`（target-only と一致）。
- C の `DFLASH2_GDN_HISTORY_BYTES=2309160960`（15 row）/ `DFLASH2_GDN_MODE=history`。

### 4.2 NgramTail opt-in の workload 別効果（production 条件）

Gate 2 harness、`CASES=k7_t0,k7_t8`、`RUNS=3`、GEN 256、history path、proxy 停止、
4 category × ctx 512/2048。`k7_t0` と `k7_t8` を同一 sample 内で交互実行した paired 比較。
ratio の 3 run は全て 0.1% 以内（安定）。

| workload | k7_t0 tok/s | k7_t8 tok/s | paired Δ | hit/round | tail acc/round |
| --- | ---: | ---: | ---: | ---: | ---: |
| code 512 | 103.30 | 105.76 | **+2.39%** | 0.167 | 0.271 |
| code 2048 | 110.96 | 112.15 | **+1.07%** | 0.262 | 0.452 |
| json 512 | 81.14 | 80.26 | -1.07% | 0.046 | 0.015 |
| json 2048 | 111.66 | 107.24 | -3.93% | 0.442 | 0.581 |
| prose 512 | 87.69 | 84.42 | -3.73% | 0.183 | 0.000 |
| prose 2048 | 80.66 | 78.11 | -3.18% | 0.230 | 0.049 |
| reasoning 512 | 59.93 | 59.59 | -0.57% | 0.023 | 0.000 |
| reasoning 2048 | 73.95 | 72.63 | -1.76% | 0.149 | 0.373 |

8 workload 平均 **-1.10%**（改善 2 / 悪化 6）。正しさは全 workload で完全。

内訳（code 512、平均 round ms）: `E/round` 5.000 → 5.312（+6.2%）に対して
round 48.42 → 50.24 ms（+3.8%）なので +2.39%。逆に prose 512 は `hit = 0.183` ながら
`tail accepted = 0` で、verify 幅増だけが残って -3.73% である。
つまり Gate 2 と同型の `E_ratio > L_ratio` の成立可否が workload で分かれる。

### 4.3 Gate 2 との差異

Gate 2 は json PP512 で +5.94% だったが、本環境の json corpus では
`hit = 0.046/round`（Gate 2 は 0.489）まで落ち、結果は -1.07% だった。
原因は corpus と model が Gate 2 のものと異なることであり、
「反復構造のある workload で勝つ」という Gate 2 の構造的結論は
（勝った workload が json から code に変わった形で）再現している。
**この差を無視して既定 ON にすると、非反復 workload で 0.6〜3.9% の回帰を既定負荷にできる。**
これが既定 OFF（opt-in）の根拠である。

### 4.4 docs/perf/current.md の扱い

`docs/perf/current.md` の正本 model `models/Qwen3.8-27B-PSQ` は本計測環境に存在せず、
prompt（31 token、tokenizer 同一）を揃えても現行コードの acceptance
（rounds 94 / accepted 161）が記載値（rounds 84 / accepted 171）と一致しない。
よって **数値は書き換えず**、DFlash2 セクションに「Gate 3 以前の値であり現行既定を保証しない」
という注記のみ追加した（`docs/perf/` の未計測値は更新しないルール）。

## 5. Gate 判定

**Gate 3: GO（opt-in 統合）**

- 正しさ: required 131/131、`k7_t8` parity 8/8、既存 case 24/24、paired 48/48、
  gate10 token exact、compute 全 config が target-only と一致 → **条件を満たす**
- performance: opt-in としての統合に性能上の障害なし（既定は変更しない）
- 既定 ON: **NO-GO**。production 条件の 8 workload で平均 -1.10%、6 workload 悪化。
  Gate 2 の条件付き GO（json PP512 のみ）を既定負荷に一般化できない。
- proxy 停止: **GO**。target-only 一致契約を回復し、コストは -2.21%。

残課題:

1. composite seed の改善（Gate 2 §9-3）。K を減らすと seed に committed token が
   入って hit が増えるが `E/round` が下がる。
2. history path での `rows >= 16` の capture overhead（M=8 で +1.94 ms/verify、
   Gate 11C 計測）を压し込む。NgramTail 有効時の verify 幅増がそのまま cost になる。
3. continuous batch による DFlash2 同時リクエスト。有効化する場合、GDN history は
   decoder 単位で確保されるため arena 容量を `2.15 GiB × N` で設計する必要がある。

## 6. artifacts と再現手順

計測出力は `/tmp/opencode/psgate3/`（本セッションの一時ディレクトリ、リポジトリ外）。
corpus のみ `artifacts/ngram_tail_gate1/corpus/`（`/artifacts/` は gitignore 対象）。

```bash
cmake --build build-gfx1201 --parallel

# correctness（production 既定構成の option を含む全 case）
HIP_VISIBLE_DEVICES=1 \
PHASESHIFT_MODEL_DIR_DFLASH2=models/z-lab/Qwen3.8-27B-DFlash2-PSQ \
PHASESHIFT_MODEL_DIR_DFLASH2_TARGET=models/Jackrong/Qwopus3.8-27B-Flash-V2-PSQ \
PHASESHIFT_NGRAM_TAIL_GATE2_MODE=correctness PHASESHIFT_NGRAM_TAIL_GATE2_GEN=256 \
PHASESHIFT_NGRAM_TAIL_GATE2_CASES=k1_t6,k3_t4,k7_t0,k7_t8 \
PHASESHIFT_NGRAM_TAIL_GATE2_DIR=artifacts/ngram_tail_gate3 \
./build-gfx1201/tests/test_dflash2_ngram_tail_gate2

# production 既定（NgramTail 無効）
./build-gfx1201/phaseshift-compute --model-dir <target> --dflash2-model-dir <draft> \
  --dflash2-drafts 7 --input-ids-file prompt_ids.txt --max-new-tokens 256 \
  --max-seq-len 4096 --arena-gib 26 --temperature 0 --dflash2-stats 1 --device 1

# NgramTail 有効化（Gate 2 推奨値）
... --dflash2-ngram-tail 8 --dflash2-ngram-n 5 ...

# 回帰
python3 tests/run_required_acceptance.py --build-dir build-gfx1201
HIP_VISIBLE_DEVICES=1 PHASESHIFT_MODEL_DIR_DFLASH2=<draft> \
  PHASESHIFT_MODEL_DIR_DFLASH2_TARGET=<target> \
  ./build-gfx1201/tests/test_dflash2_gate10_cli
```
