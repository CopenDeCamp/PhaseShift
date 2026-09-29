# NgramTail Gate 2 — DFlash2 + NgramTail live speculative verify

## 0. 起点と範囲

Gate 1（`docs/rnd/spec_decode/ngram_tail_gate1.md`）は target-only oracle replay で
NgramTail の候補品質を確認し、GO / strong GO を出した。

Gate 2 はそれを実際の speculative decode path に接続する。

    DFlash2 draft
        +
    NgramTail extension
        +
    target Exact verify

Gate 2 が答える質問:

> DFlash2 の draft の後ろに NgramTail を追加したとき、
> 実際の target verify を含めて emitted/update と tok/s が改善するか？

production CLI 追加、DFlash checkpoint / block_size 変更、MTP 復活、GPU n-gram、
GDN 圧縮、新的 Exact kernel はこの Gate の範囲外である。

## 1. Gate 1 から引き継ぐ事実

| 条件 | accepted/position | ideal E/update |
| --- | --- | --- |
| code PP512 / n=5 / tail=32 (TG512) | 1.504 | 2.504 |
| code PP2048 / n=5 / tail=32 (TG512) | 2.267 | 3.267 |
| json PP512 / n=5 / tail=32 (TG512) | 5.792 | 6.792 |
| prose PP2048 / n=5 / tail=32 (TG512) | 10.625 | 11.625 |
| reasoning PP2048 / n=5 / tail=32 (TG512) | 4.300 | 5.300 |
| json PP512 / n=8 / tail=32 (TG256, best) | 8.262 | 9.262 |

ただし Gate 1 は standalone oracle であり、DFlash2 prefix が先に全 accept されることを
要求していない。Gate 2 はこの差を測る。

## 2. live composite の構成

linear proposal:

    DFlash2:      d0 d1 ... d(K-1)
    NgramTail:    n0 n1 ... n(T-1)
    composite:    d0 ... d(K-1) n0 ... n(T-1)

target acceptance が DFlash prefix の途中で止まった場合、tail は 1 token も利益を
生まない。よって以下を必ず分離して測る。

- `dflash_accepted`（prefix 受容）
- `tail_reached`（`accepted >= dflash_k`）
- `tail_accepted`（`accepted - dflash_k`、正のときのみ）

## 3. runtime 契約の変更点

### 3.1 verify capacity を DFlash block capacity から分離

`dflash2::kMaxBlockSize = 16`（DFlash2 model config の `block_size = 8` を上限とする
drafter 側の定数）は変更しない。runtime 側に独立した定数を置いた。

```cpp
constexpr uint32_t kDFlash2SpecMaxVerifyRows = 64u;
constexpr uint32_t kDFlash2SpecMaxVerifyDrafts = kDFlash2SpecMaxVerifyRows - 1u;
```

R64 bucket の上限に合わせた 64 rows（= draft 63 + anchor 1）。
以下は全て verify capacity 前提へ変更した。

- `verify_token_ids_device` / `decision_staging_device` の確保サイズ
- host 側 `drafts` / `sampled` / `decision_host` / `verify_host` / `emitted` / `prefix`
- `DFlash2SpecIterationOutput::emitted`

DFlash2 drafter の `proposal_tokens` buffer と `draft_cfg.max_rows` は
`block_size` のまま（= 8）である。

### 3.2 NgramTail 設定

`DFlash2SpecDecoderConfig` に `ngram_n` / `ngram_max_tail` / `ngram_window` を追加した。
既定は `n = 0`（disabled）で、既存 runtime 動作は変更しない。
Gate 1 の `NgramTailConfig` は重複実装せず、step 内でこれを作る。

### 3.3 token history invariant

`DFlash2SpecDecoder::token_history` は target が確定済みの token 列のみを保持する。

- prefill 後: `prompt + pending_token`
- step 後: `out.emitted[0:emitted_count]`（EOS truncate 後）のみ追加
- error / abort では変更しない
- DFlash proposal token も Ngram proposal token も proposal 時点では追加しない

### 3.4 composite lookup の query

candidate corpus は `token_history` のみ。
`read_only_suffix`（DFlash draft token K 個）は seed の一部としてのみ使い、
continuation source には使わない。

    seed = 末尾 n token of (token_history + DFlash drafts)
    candidate = token_history 内の一致位置
    continuation = token_history[candidate_end ...]

Gate 1 の helper は削除・複製せず、API を拡張した。

```cpp
Status propose_ngram_tail_into(span<const int32_t> committed_history,
                               span<const int32_t> read_only_suffix,
                               const NgramTailConfig& config,
                               span<int32_t> output,
                               NgramTailMatch& out_match);
```

`read_only_suffix` が empty なら Gate 1 の standalone 動作と完全一致する
（既存 Gate 1 unit test は全 PASS）。
`_into` は allocation-free で、runtime は固定 buffer を使う。

future leakage contract は Gate 1 と同じ:

    candidate_start  < committed_history.size()
    candidate_end   <= committed_history.size()
    candidate_end + count <= committed_history.size()

### 3.5 CPU NgramTail と proposal D2H

DFlash proposal は device resident だが、CPU NgramTail は proposal token を必要とする。
Ngram enabled のみ、`dflash2_propose_cached()` の後に

    stream sync + proposal D2H（`ngram_seed_wait_ms`）
        → CPU lookup（`ngram_lookup_ms`）
        → tail のみ H2D（`ngram_tail_h2d_ms`）
        → composite verify

を行う。DFlash 部分は従来どおり device D2D のままであり、
composite 全体を host 経由で再アップロードはしない。
これはこの Gate の意図した PoC コストである。

### 3.6 GDN state

- Gate 2A: `history_rows = num_drafts + ngram_max_tail`（`kDFlash2SpecMaxVerifyDrafts` で
  上限）。1.5 GiB guard は維持。guard 超過は create 時に error（確保前へ移動した）。
- Gate 2B: `PHASESHIFT_DFLASH2_GDN_RERUN_REFERENCE=1` の snapshot + restore + accepted
  prefix rerun を使う。Tail16/32 分の full history は guardを超えるため作らない。

### 3.7 Exact rows

現行 optimized Exact path は `actual_rows ∈ {2,4,8}` のときだけ
DecodeRowsExact を使う。Gate 2A は `total verify rows <= 8`（= composite drafts <= 7）を
守ることでこの path を維持する。Gate 2B の rows > 8 は既知の測定条件として記録し、
この Gate では最適化しない。

### 3.8 bf16 GEMM の exact rows 拡張（R64 gap の 1 つ目）

`VerifyNumericMode::Exact` でも bf16 GEMM の exact 経路は
`kBf16GemmExactRowsMax = 16` までしか選ばれず、rows ≥ 17 では WMMA（lossy）へ
落ちていた。K 還元順が M=1 decode と一致しないため、R64 bucket（rows 17..40）で
target parity が崩れる。

- `kBf16GemmExactRowsMax` を 64 に拡大し、`kBf16GemmExactRowsKernelMax = 16`（
  template kernel の per-launch 上限）を分離した。
- `launch_gemm_bf16_exact_rows()` は rows > 16 のとき 16 行ずつ chunk launch する。
  `gemm_bf16_exact_rows_body<R>` の K 還元ループと lane 減算は row 数に依存しないため、
  chunk 化しても M=1 と bit-exact である。weight 再利用も chunk 単位で維持され、
  template 実体は 1..16 のままなのでコンパイル時間は増えない。

効果（`C9` = K7 + T9、max rows 17）: 修正前 NO → 修正後 **parity 恢復**。
rows 18..24（`C16` / `C32`）には別要因が残り、それは §3.9 で扱う。

### 3.9 R64 gap の残り: attention split fallback

bf16 GEMM 修正後も rerun mode で rows ≥ 17 の divergence が残った。原因は decode
attention の split fallback である。

decode attention は `max_visible_tokens >= 2048` で S=16 の split カーネルを使うが、
partial workspace が 8 MiB 固定のため必要量
`rows×kv_heads×16×q_per_kv×(2+256)×4B` を超えると旧カーネルへ fallback する。
27B geometry（kv_heads=4, q_per_kv=6, head_dim=256）では **rows > 21** がこれに当たる。
split と旧カーネルは f32 丸めが一致せず（softmax 計算順序差）、verify だけ旧カーネルに
落ちると M=1 decode（split）と target parity が崩れる。

- context 512（split 無効）では rows 40 まで parity OK、context 2048 では rows ≥ 22 で NG。
- `runtime::kDecodeAttnPartialBytes` を 8 MiB → **32 MiB** に拡張（rows 64 まで split 維持）。
- 効果: 既定 lossy のまま `A / C8 / C16 / C32` が全 6 workload で parity 合格
  （4 config × 6 workload × 2 phase = **48/48**、`failed=0`）。
- 詳細は `docs/rnd/attention/optimization_history.md` §7.99。

注意: 当初 GDN recurrence の lossy 経路を原因と誤認したが、bf16 修正と本 attention 修正を
入れると **既定 lossy のままで parity が回復**する。`PHASESHIFT_GDN_RECURRENCE_EXACT=1` は
parity には不要である（exact のコストは §6.5）。

#### 参考: exact 経路の GDN history capture 欠落（別件）

`PHASESHIFT_GDN_RECURRENCE_EXACT=1` を使う場合には別の不具合があった。lossy off では
`gdn_recurrence_decode1_supported()` が false になり verify variant が `WmmaSerial` へ
変わるが、GDN spec history の capture は decode1 kernel の `row_override >= 0` 条件付き
ブロックにのみ実装されていた。`WmmaSerial` は `row_override` を設定しないため history が
書かれず、history mode（Gate 2A）では restore が未初期化行を読んで generation が崩壊した
（parity 0/24）。

修正として `gdn_recurrence_wmma_body<kLossy, kSerialRows>` の serial chunk 終端で
state 更新後の `Sacc` を history 行 `ci` へ書く（条件・レイアウトは
`gdn_recurrence_wmma_decode_rows_exact_impl` と同一）。詳細は
`docs/rnd/gdn/optimization_history.md` §7.96。

- `PHASESHIFT_GDN_RECURRENCE_EXACT=1` + `PHASESHIFT_TARGET_LM_HEAD_PROXY=0` の
  Gate 2A（correctness, history mode）: parity **24/24**、violations **0**。
- 既定 lossy の Gate 2A は従来どおり parity 24/24。
- 回帰テスト `test_gdn_recurrence_exact_history`（required）で exact serial と
  sequential exact M=1 の state / output / history が bit-exact。

## 4. 測定

### 4.1 target parity と lm_head proxy

Gate 2 は `VerifyNumericMode::Exact` 固定で測る（Fast は測らない）。
しかし Exact だけでは target parity は達成できなかった。

既存 `test_dflash2_gate9_e2e` は **Gate 2 の変更を含まない baseline コードでも**
`json_k7_ctx32` で `token[1]` から diverge して FAIL する
（`git stash` して変更前コードで同一実行、`GATE9_BASELINE_EXIT=1`）。
原因は `PHASESHIFT_TARGET_LM_HEAD_PROXY`（既定 ON）の target lm_head proxy で、
decode（M=1）と verify（M>1）で logits が一致しないためである。

- proxy ON（既定）: gate9 `failed=1`（baseline / Gate 2 変更後とも同一）
- proxy OFF: gate9 `failed=0`（token parity K1 / K3 / K7 すべて一致）

よって **Gate 2 の全測定は `PHASESHIFT_TARGET_LM_HEAD_PROXY=0` で実施する**。
これは runtime コードの変更ではなく環境変数であり、baseline / candidate 双方に
同じ条件を適用するため比較は公平である。
verify 時間への影響は gate9 の計測で約 +2%（例: prose K7 verify 1648 → 1682 ms）。

### 4.2 harness

`tests/unit/test_dflash2_ngram_tail_gate2.hip`（optional test）。

| 環境変数 | 既定 |
| --- | --- |
| `PHASESHIFT_MODEL_DIR_DFLASH2` / `_TARGET` | 必須 |
| `PHASESHIFT_NGRAM_TAIL_GATE2_MODE` | `correctness`（`perf` で Gate 2B） |
| `PHASESHIFT_NGRAM_TAIL_GATE2_CORPUS` | `artifacts/ngram_tail_gate1/corpus`（無ければ `tests/data/mtp_perf`） |
| `PHASESHIFT_NGRAM_TAIL_GATE2_DIR` | `artifacts/ngram_tail_gate2` |
| `PHASESHIFT_NGRAM_TAIL_GATE2_WORKLOADS` | correctness: 4 カテゴリ × 512/2048、perf: 6 workload |
| `PHASESHIFT_NGRAM_TAIL_GATE2_CASES` | correctness: `k1_t6,k3_t4,k7_t0`、perf: `A..C` 5 本 |
| `PHASESHIFT_NGRAM_TAIL_GATE2_GEN` | correctness 256 / perf 512 |
| `PHASESHIFT_NGRAM_TAIL_GATE2_RUNS` | correctness 1 / perf 5 |
| `PHASESHIFT_NGRAM_TAIL_GATE2_N` / `_WINDOW` | 5 / 2048 |

出力: `correctness.csv` / `raw_perf.csv` / `aggregate_perf.csv` / `environment.json`。

各 round で次の assertion を行い、違反すればその run を FAIL とする。

- `total_k == dflash_k + tail_k`
- `total_k <= case capacity`（Gate 2A では 7）
- `accepted <= total_k`
- `tail accepted <= tail_k`、`tail accepted > 0 ⇒ accepted >= dflash_k + 1`
- `ngram candidate_end <= committed_history size（lookup 時点）`
- `context.next_position == sequence.position`

## 5. Gate 2A — bounded-width live integration

条件: `PHASESHIFT_TARGET_LM_HEAD_PROXY=0`、Exact、GDN history mode、TG256、
1 prompt / category / context、n=5、window=2048、workload = 4 カテゴリ × PP512 / PP2048、
case = `k1_t6` / `k3_t4` / `k7_t0`（control、ngram disabled）。
verify rows は全 case で 8 以下（`max_total_drafts = 7`）。

| category | context | case | rounds | ngram hits | tail accepted | token exact | violations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| code | 512 | k1_t6 | 135 | 3 | 3 | yes | 0 |
| code | 512 | k3_t4 | 81 | 2 | 2 | yes | 0 |
| code | 512 | k7_t0 | 56 | 0 | 0 | yes | 0 |
| code | 2048 | k1_t6 | 100 | 26 | 56 | yes | 0 |
| code | 2048 | k3_t4 | 57 | 22 | 39 | yes | 0 |
| code | 2048 | k7_t0 | 39 | 0 | 0 | yes | 0 |
| json | 512 | k1_t6 | 75 | 31 | 125 | yes | 0 |
| json | 512 | k3_t4 | 61 | 29 | 81 | yes | 0 |
| json | 512 | k7_t0 | 53 | 0 | 0 | yes | 0 |
| json | 2048 | k1_t6 | 129 | 11 | 21 | yes | 0 |
| json | 2048 | k3_t4 | 89 | 8 | 14 | yes | 0 |
| json | 2048 | k7_t0 | 78 | 0 | 0 | yes | 0 |
| prose | 512 | k1_t6 | 131 | 7 | 12 | yes | 0 |
| prose | 512 | k3_t4 | 84 | 6 | 7 | yes | 0 |
| prose | 512 | k7_t0 | 66 | 0 | 0 | yes | 0 |
| prose | 2048 | k1_t6 | 126 | 9 | 30 | yes | 0 |
| prose | 2048 | k3_t4 | 90 | 9 | 17 | yes | 0 |
| prose | 2048 | k7_t0 | 80 | 0 | 0 | yes | 0 |
| reasoning | 512 | k1_t6 | 146 | 2 | 0 | yes | 0 |
| reasoning | 512 | k3_t4 | 101 | 2 | 0 | yes | 0 |
| reasoning | 512 | k7_t0 | 88 | 0 | 0 | yes | 0 |
| reasoning | 2048 | k1_t6 | 99 | 24 | 80 | yes | 0 |
| reasoning | 2048 | k3_t4 | 72 | 21 | 49 | yes | 0 |
| reasoning | 2048 | k7_t0 | 60 | 0 | 0 | yes | 0 |

合計: `GATE2 parity=24/24`、`ngram_hit_rounds=212`、`ngram_accepted=536`、
per-round assertion 違反 0、future leakage 0、context position invariant 0 違反。

観察:

- `tail_blocked` は全 case で 0〜4 round（proposal はあったが DFlash prefix で reject）。
- json PP512 / k1_t6 は 75 round 中 31 hit、tail accepted 125
  （= 1.67 token/round）で、Gate 1 の signal が live でも現れた。
- control（k7_t0）は ngram disabled でも parity が一致しており、
  NgramTail が spec 出力に影響していないことを確認した。
- reasoning PP512 は hit しても tail accepted 0（Gate 1 で最も弱かったカテゴリと整合）。

**Gate 2A: PASS**（§39 の全条件を満たす）。

## 6. Gate 2B — long-tail live performance

### 6.1 条件

- K = 7、n = 5、window = 2048、T ∈ {0, 8, 16, 32}
- baseline A = `k7 t0`（device token bridge ON）、baseline B = `k7 t0` + host proposal D2H、
  candidate C = `k7 + t8 / t16 / t32`
- TG512、5 paired samples（sample 単位で A→B→C8→C16→C32 の順に交互実行）
- workload: code PP512 / code PP2048 / json PP512 / json PP2048 / prose PP2048 /
  reasoning PP2048
- Exact、`PHASESHIFT_TARGET_LM_HEAD_PROXY=0`、`PHASESHIFT_DFLASH2_GDN_RERUN_REFERENCE=1`
- smoke（TG256 × 1 sample）→ perf（TG512 × 5 sample）

### 6.2 target parity と verify rows の壁

smoke を含む全 run の parity:

| config | max verify rows | row bucket | parity |
| --- | --- | --- | --- |
| A (t0) | 8 | R16 | 30/30 |
| B (t0 + host D2H) | 8 | R16 | 30/30 |
| **C8** | **16** | **R16** | **30/30** |
| C16 | 24 | R64 | 20/30（code2048 / json512 / prose2048 / reasoning2048 で FAIL） |
| C32 | 40 | R64 | 20/30（同一 workload・同一 idx で FAIL） |

閾値の切り分けとして `t9`（max rows = 17）を prose PP2048 で追加測定した。

| config | max rows | bucket | parity | diverge idx |
| --- | --- | --- | --- | --- |
| A | 8 | R16 | yes | — |
| C8 | 16 | R16 | yes | — |
| **C9** | **17** | **R64** | **NO** | 18 |
| C16 | 24 | R64 | NO | 18 |

⇒ **verify rows が 16 を超えた瞬間（`resolve_row_bucket` が R16 → R64 へ変わる）に
target parity が崩れる。** `VerifyNumericMode::Exact` の保証は R16 bucket の経路に対して
しか成立していない。これは §54 の hard stop（`Exact mode で baseline と generation
divergence`）に該当するため、**C16 / C32 の性能値は無効として破棄する。**

歴史的経緯との関係: `docs/rnd/mtp/optimization_history.md` §7.64 の VERIFY_EXACT は
K ≤ 8（rows ≤ 9）で検証されており、`docs/rnd/dflash2/dflash2.md` も Exact を既定にしている。
Gate 2 が初めて rows 17..40 の verify を通したため、この gap が露呈した。
GDN history mode では 1.5 GiB guard（rows ≤ 10）により wide rows をそもそも作れないため、
この問題は rerun mode でのみ観測される（§3.6）。

**2026-09-29 追記（R64 gap 解消）**: 上記 C16 / C32 の hard stop は §3.8 / §3.9 で解消した。
内訳は (a) bf16 GEMM の exact rows 上限（§3.8）と (b) attention split fallback（§3.9）である。
**既定 lossy のまま** `A / C8 / C16 / C32` が全 6 workload で target parity に合格する
（4 config × 6 workload × 2 phase = 48/48）。したがって **T = 16 / 32 の性能値は再測定により
有効化できる**（本節の旧値は破棄のまま）。`PHASESHIFT_GDN_RECURRENCE_EXACT=1` は parity には
不要である。

### 6.3 性能（R64 修正後の全 matrix）

§3.8 / §3.9 の修正後、`A / B / C8 / C16 / C32` を 6 workload × 5 sample で再測定した
（`artifacts/ngram_tail_gate2/wide_tail/`、parity **174/174**、failed=0。T=16/32 を含む）。

paired median（A に対する tok/s 比、n=5）:

| workload | B | C8 | C16 | C32 |
| --- | --- | --- | --- | --- |
| code PP512 | -0.08% | -2.69% | -3.39% | -4.87% |
| code PP2048 | -0.12% | -5.47% | -9.95% | -16.69% |
| **json PP512** | -0.10% | **+5.94%** | **+0.42%** | -4.28% |
| json PP2048 | -0.08% | -9.75% | -11.54% | -15.89% |
| prose PP2048 | -0.09% | -0.32% | -1.72% | -4.13% |
| reasoning PP2048 | -0.08% | -6.98% | -9.02% | -14.85% |

break-even（`E_ratio > L_ratio`、§47）は json PP512 の C8（1.227 > 1.159）と C16
（1.241 > 1.236）でのみ成立する。C16 の実測は +0.42% とほぼ横ばい、C32 は
E_ratio 1.286 < L_ratio 1.343 で不成立。

⇒ **tail を伸ばすほど verify 幅（`rows/round`）と rerun prefix のコストが受容増を
上回る。** T=8 が最良で、T=16 は json PP512 でかろうじて横ばい、T=32 は全 workload で
悪化する。Gate 1 が示した「tail 32 まで伸ばす」価値は、この corpus では確認できない。

baseline B は全 workload で A と ±0.12% 以内 → **proposal D2H（seed wait）は
コストとして観測されない**。先に stream を sync しているだけであり、
decision sync の時間と相殺される。

主要内訳（median tok/s、mean/round）:

| workload | config | tok/s | E/round | tail acc/round | rows/round | round_ms | verify_gpu_ms | rerun_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| code PP512 | A | 70.76 | 5.059 | 0.000 | 8.0 | 71.50 | 40.90 | 23.76 |
| code PP512 | C8 | 68.85 | 5.110 | 0.060 | 8.4 | 74.21 | 41.42 | 25.85 |
| code PP512 | C16 | 68.36 | 5.110 | 0.060 | 8.9 | 74.75 | 41.95 | 25.86 |
| code PP512 | C32 | 67.32 | 5.110 | 0.060 | 9.6 | 75.91 | 43.10 | 25.88 |
| code PP2048 | A | 111.50 | 7.000 | 0.000 | 8.0 | 62.77 | 42.04 | 12.53 |
| code PP2048 | C8 | 105.40 | 8.111 | 1.317 | 10.8 | 76.95 | 45.69 | 22.88 |
| code PP2048 | C16 | 100.40 | 8.517 | 1.700 | 14.2 | 84.81 | 50.13 | 26.28 |
| code PP2048 | C32 | 92.89 | 8.810 | 2.052 | 19.1 | 94.85 | 59.51 | 26.91 |
| **json PP512** | A | 64.96 | 4.731 | 0.000 | 8.0 | 72.84 | 40.95 | 25.03 |
| **json PP512** | **C8** | **68.82** | **5.807** | **1.534** | 11.7 | 84.38 | 45.24 | 32.12 |
| json PP512 | C16 | 65.24 | 5.874 | 1.667 | 14.1 | 90.03 | 48.25 | 34.74 |
| json PP512 | C32 | 62.17 | 6.083 | 1.940 | 17.5 | 97.84 | 54.06 | 36.73 |
| json PP2048 | A | 55.94 | 4.294 | 0.000 | 8.0 | 76.76 | 41.99 | 26.58 |
| json PP2048 | C8 | 50.48 | 4.368 | 0.248 | 9.4 | 86.50 | 43.76 | 34.44 |
| json PP2048 | C16 | 49.49 | 4.368 | 0.248 | 10.7 | 88.26 | 45.49 | 34.47 |
| json PP2048 | C32 | 47.05 | 4.368 | 0.248 | 13.1 | 92.83 | 50.01 | 34.52 |
| prose PP2048 | A | 57.00 | 4.405 | 0.000 | 8.0 | 77.28 | 42.01 | 27.09 |
| prose PP2048 | C8 | 56.82 | 4.867 | 0.790 | 9.7 | 85.64 | 44.20 | 33.12 |
| prose PP2048 | C16 | 56.02 | 5.059 | 1.149 | 11.2 | 90.31 | 46.14 | 35.82 |
| prose PP2048 | C32 | 54.65 | 5.162 | 1.333 | 12.9 | 94.45 | 49.42 | 36.68 |
| reasoning PP2048 | A | 67.08 | 4.913 | 0.000 | 7.9 | 73.25 | 41.99 | 23.08 |
| reasoning PP2048 | C8 | 62.40 | 5.615 | 0.879 | 11.7 | 89.99 | 46.73 | 34.93 |
| reasoning PP2048 | C16 | 61.03 | 5.874 | 1.299 | 14.8 | 96.24 | 51.02 | 36.85 |
| reasoning PP2048 | C32 | 57.12 | 5.874 | 1.299 | 18.5 | 102.82 | 57.54 | 36.92 |

CPU Ngram lookup は 1 round あたり 0.001〜0.003 ms（linear scan）で無視できる。
`ngram_seed_wait_ms` は B と同水準（6.0〜7.4 ms/round）で、A の decision sync と相殺される。
`ngram_tail_h2d_ms` は 0.001 ms 以下。

### 6.4 secondary（§43: json PP512, n=8, T=8）

primary 終了後に n=8 で json PP512 のみ追加測定（`artifacts/ngram_tail_gate2/secondary_n8/`）。

- parity 12/12
- paired median **+6.76%**（n=5 の +5.94% より良い）
- E/round 5.742（A 4.731）、round_ms 82.73（A 72.74）、tail acc/round 1.427、hit 0.348

### 6.5 GDN recurrence exact のコスト（参考計測）

parity には不要だが、`PHASESHIFT_GDN_RECURRENCE_EXACT=1` を選ぶ場合のコストを
A_k7_t0 で同一条件測定した
（TG512、n=5、6 workload、`artifacts/ngram_tail_gate2/gdn_exact_timing/`）。

| 指標 | lossy 既定 | exact | 差 |
| --- | --- | --- | --- |
| verify GPU ms/round | 40.9–42.0 | 47.2–49.1 | **+6.8–7.1 ms（+17%、全 workload 一定）** |
| round_ms | 62.7–77.2 | 70.3–92.8 | +1.7–15.6 |
| tok/s | 基準 | — | **-24.8%〜+32.2%** |

tok/s が大きく振れるのは GDN の数値差が DFlash2 drafter の target tap に伝播し、
draft 受容（E/round）が ±26〜28% 変わるためで、計算コスト差ではない。純粋な
recurrence コストは verify の **+約 7 ms/round（+17%）**である。prefill 側の既存計測は
`docs/rnd/gdn/optimization_history.md` §7.51（e2e PP で lossy が +3.7% tok/s）を参照。

## 7. Gate 1 oracle との差

§48 の通り、Gate 1 の `accepted_per_position` を Gate 2 の期待値にしてはならない。

| workload（n=5, TG512） | Gate 1 accepted/position（standalone oracle） | Gate 2 tail_accepted/round（K=7 + T=8） |
| --- | --- | --- |
| code PP512 | 1.504 | 0.060 |
| code PP2048 | 2.267 | 1.317 |
| json PP512 | 5.792 | 1.534 |
| json PP2048 | 1.020 | 0.248 |
| prose PP2048 | 10.625 | 0.790 |
| reasoning PP2048 | 4.300 | 0.879 |

低下の理由は 2 つある。

1. **DFlash prefix 全 accept が必要**: `reach/round` は 0.04〜0.29 にとどまる。
2. **composite seed が DFlash draft で埋まる**: K=7 ≥ n=5 のため、seed の末尾 5 token は
   全て DFlash proposal になり、その 5-gram が過去 history に存在して初めて hit する。
   結果 hit rate は workload 依存（code PP512 0.060、json PP512 0.489）。

`tail_acc/reached` は 1.5〜5.9 と Gate 1 の `accepted/hit` より低いが同水準で、
到達さえすれば tail 自体は有効に働く。

## 8. Gate 判定

### 8.1 Gate 2A: **PASS**

§39 の全条件を満たす（§5）。

### 8.2 Gate 2B: **DIRECT GO（strong）— 条件付き**

§55 の条件を json PP512 で満たす。

- correctness: Gate 2 の T = 8 / 16 / 32 を含む全 run で target parity 100%
  （wide-tail 再測定 174/174）
- candidate tok/s ≥ baseline + 2%: **C8 = +5.94%**（n=8 で +6.76%）paired median
- `tail_accepted_per_round` = 1.534 > 0
- +5% 以上 → **strong direct GO**

ただし次の 2 点がこの GO の範囲を狭める。

1. **GO は json PP512 のみ**。残り 5 workload は -0.32% 〜 -9.75% で、
   `E_ratio > L_ratio` を満たさない。反復構造が強い workload 限定の GO である。
2. **tail を伸ばしても改善しない**。R64 gap 解消（§3.8 / §3.9）後に T = 16 / 32 を
   再測定した結果、json PP512 の C16 は +0.42% とほぼ横ばい、C32 は -4.28%。
   他 workload は T が伸びるほど悪化する（§6.3）。**T = 8 が最良**であり、
   Gate 1 が示した「tail 32 まで伸ばす」価値はこの corpus では確認できない。

NO-GO 条件（§57）には該当しない（json PP512 で改善があり、
`emitted_per_round` gain は +21〜23%）。

### 8.3 Primary bottleneck

    target verify width

根拠:

- proposal D2H（seed wait）は baseline B で A と ±0.13% → 汎用オーバーヘッドではない。
- CPU Ngram lookup は 0.003 ms/round 未満 → 無視できる。
- C8 の劣化は全て `rows/round` 増（8.0 → 8.4..11.7）に伴う
  `verify_gpu_ms`（+0.7..+4.8 ms）と `rerun_ms`（+2.3..+11.8 ms）で説明できる。
- json PP512 だけが `E_ratio`(1.227) > `L_ratio`(1.160) を満たすのは、
  tail 受容（1.534 token/round）が verify 幅増のコストを上回る唯一の workload であるため。

補助因: **low Ngram acceptance**（K=7 の composite seed により hit が workload 依存、
code PP512 は 0.060/round）と **GDN rerun**（A でも round の 34% を占め、
C8 では tail 受容分だけ rerun prefix が長くなる）。

## 9. 次の action

1. **R64 bucket の Exact parity gap を解消する**（Gate 2B 再開の必須条件）。
   `rows > 16` の verify が M=1 decode と一致しないため、T ≥ 9 の評価ができない。
   scope は `docs/rnd/mtp/optimization_history.md` §7.64 / §7.65 の続編として
   verify role の exact 保証を R64 bucket まで広げること。
   → **2026-09-29 解消**（§3.8 / §3.9）。bf16 GEMM exact rows を 64 へ拡張し、
   attention split partials を 32 MiB へ拡張する。既定 lossy のままで parity が成立。
   → **wide-tail 再測定完了**（§6.3、parity 174/174）。結論は **T = 8 が最良**で、
   T = 16 / 32 は改善しない。
2. **json PP512（反復構造）を対象に Gate 3（production 統合）へ進む**。
   推奨 parameter は `DFlash2 K=7 + NgramTail T=8 / n=5`（n=8 も候補）、window 2048、
   Exact、`PHASESHIFT_TARGET_LM_HEAD_PROXY=0` のまま。T = 16 以上は採用しない。
   → **統合済み**。経緯と計測は [ngram_tail_gate3.md](ngram_tail_gate3.md)。
3. **composite seed の改善を検討する**（次 Gate）。K を減らす（K=3 + T=16 のような構成）と
   seed に committed token が混じって hit が増えるが、`E/round` が下がるトレードオフが
   ある。Gate 2A の `k3_t4`（json PP512 で hit 29/61 round、tail 81 token）は有望。
4. production CLI 追加は行わない（§59）。DIRECT GO でも production contract は Gate 3 で決める。
   → Gate 3 で `--dflash2-ngram-tail` / `--dflash2-ngram-n` を追加したが、
   既定は 0（opt-in）。理由は [ngram_tail_gate3.md](ngram_tail_gate3.md)。

## 10. artifacts と再現手順

出力（`artifacts/ngram_tail_gate2/`）:

| file | 内容 |
| --- | --- |
| `environment.json` | mode / model / corpus / workload / config 一覧 |
| `gate2a_correctness.csv` | Gate 2A の全 24 run（parity / hits / tail accepted / violations） |
| `correctness.csv` | Gate 2B smoke の parity |
| `raw_perf.csv` | Gate 2B 主要 matrix（150 run、TG512 × 5 sample × 5 config × 6 workload） |
| `aggregate_perf.csv` | config 別集計（median tok/s、E/round、tail 指標、rows、ms） |
| `secondary_n8/` | json PP512 の n=8 追加測定 |
| `wide_tail/` | R64 修正後の再測定（A / B / C8 / C16 / C32、150 run、parity 174/174） |
| `diag_rows/` | rows 閾値切り分け（A / C8 / C9 / C16） |
| `diag_history/` | GDN history guard の確認（rows=23 が 1.5 GiB guard で reject） |

再現:

```bash
cmake --build build-gfx1201 --parallel

PHASESHIFT_TARGET_LM_HEAD_PROXY=0 \
PHASESHIFT_MODEL_DIR_DFLASH2=models/Qwen3.8-27B-DFlash2-PSQ \
PHASESHIFT_MODEL_DIR_DFLASH2_TARGET=models/Qwen3.8-27B-PSQ \
PHASESHIFT_NGRAM_TAIL_GATE2_MODE=correctness PHASESHIFT_NGRAM_TAIL_GATE2_GEN=256 \
PHASESHIFT_NGRAM_TAIL_GATE2_DIR=artifacts/ngram_tail_gate2 \
./build-gfx1201/tests/test_dflash2_ngram_tail_gate2

PHASESHIFT_TARGET_LM_HEAD_PROXY=0 \
PHASESHIFT_MODEL_DIR_DFLASH2=models/Qwen3.8-27B-DFlash2-PSQ \
PHASESHIFT_MODEL_DIR_DFLASH2_TARGET=models/Qwen3.8-27B-PSQ \
PHASESHIFT_NGRAM_TAIL_GATE2_MODE=perf PHASESHIFT_NGRAM_TAIL_GATE2_GEN=512 \
PHASESHIFT_NGRAM_TAIL_GATE2_RUNS=5 \
PHASESHIFT_NGRAM_TAIL_GATE2_DIR=artifacts/ngram_tail_gate2 \
./build-gfx1201/tests/test_dflash2_ngram_tail_gate2
```

回帰:

```bash
ctest --test-dir build-gfx1201 --output-on-failure \
  -R "test_ngram_tail|test_qwen35_spec_verify|test_gdn_spec_history"
python3 tests/run_required_acceptance.py --build-dir build-gfx1201
PHASESHIFT_TARGET_LM_HEAD_PROXY=0 ./build-gfx1201/tests/test_dflash2_gate9_e2e
```

