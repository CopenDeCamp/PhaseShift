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

### 6.3 性能（parity が成立する C8 のみ有効）

paired median（A に対する tok/s 比）:

| workload | B vs A | **C8 vs A** | E_ratio (C8/A) | L_ratio (C8/A) | break-even |
| --- | --- | --- | --- | --- | --- |
| code PP512 | -0.13% | **-3.02%** | 1.010 | 1.041 | 否 |
| code PP2048 | -0.11% | **-5.50%** | 1.159 | 1.226 | 否 |
| **json PP512** | -0.04% | **+5.84%** | 1.227 | 1.160 | **成立** |
| json PP2048 | -0.11% | **-9.70%** | 1.017 | 1.126 | 否 |
| prose PP2048 | -0.08% | **-0.32%** | 1.105 | 1.108 | 否 |
| reasoning PP2048 | -0.06% | **-7.00%** | 1.143 | 1.229 | 否 |

`E_ratio > L_ratio` が break-even の必要条件（§47）で、json PP512 のみ成立し、
実測 tok/s も一致して増加している。

baseline B は全 workload で A と ±0.13% 以内 → **proposal D2H（seed wait）は
コストとして観測されない**。先に stream を sync しているだけであり、
decision sync の時間と相殺される。

内訳（median tok/s、mean/round）:

| workload | config | tok/s | E/round | tail acc/round | tail acc/reached | hit/round | reach/round | rows/round | round_ms | verify_gpu_ms | rerun_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| code PP512 | A | 70.78 | 5.059 | 0.000 | — | — | — | 8.0 | 71.97 | 41.16 | 23.89 |
| code PP512 | C8 | 68.36 | 5.110 | 0.060 | 1.500 | 0.060 | 0.040 | 8.4 | 75.14 | 41.88 | 26.13 |
| code PP2048 | A | 111.65 | 7.000 | 0.000 | — | — | — | 8.0 | 62.73 | 42.00 | 12.51 |
| code PP2048 | C8 | 105.48 | 8.111 | 1.317 | 4.611 | 0.365 | 0.286 | 10.8 | 76.88 | 45.63 | 22.85 |
| **json PP512** | A | 65.03 | 4.731 | 0.000 | — | — | — | 8.0 | 72.75 | 40.89 | 24.98 |
| **json PP512** | C8 | **68.88** | **5.807** | **1.534** | 5.870 | 0.489 | 0.261 | 11.7 | 84.34 | 45.20 | 32.08 |
| json PP2048 | A | 56.00 | 4.294 | 0.000 | — | — | — | 8.0 | 76.69 | 41.95 | 26.54 |
| json PP2048 | C8 | 50.49 | 4.368 | 0.248 | 1.812 | 0.179 | 0.137 | 9.4 | 86.64 | 43.82 | 34.48 |
| prose PP2048 | A | 57.07 | 4.405 | 0.000 | — | — | — | 8.0 | 77.20 | 41.96 | 27.04 |
| prose PP2048 | C8 | 56.90 | 4.867 | 0.790 | 5.188 | 0.219 | 0.152 | 9.7 | 85.57 | 44.16 | 33.07 |
| reasoning PP2048 | A | 67.19 | 4.913 | 0.000 | — | — | — | 7.9 | 73.15 | 41.92 | 23.03 |
| reasoning PP2048 | C8 | 62.49 | 5.615 | 0.879 | 3.333 | 0.473 | 0.264 | 11.7 | 89.87 | 46.67 | 34.86 |

CPU Ngram lookup は 1 round あたり 0.001〜0.003 ms（linear scan）で無視できる。
`ngram_seed_wait_ms` は B と同水準（6.0〜7.4 ms/round）で、A の decision sync と相殺される。
`ngram_tail_h2d_ms` は 0.001 ms 以下。

### 6.4 secondary（§43: json PP512, n=8, T=8）

primary 終了後に n=8 で json PP512 のみ追加測定（`artifacts/ngram_tail_gate2/secondary_n8/`）。

- parity 12/12
- paired median **+6.76%**（n=5 の +5.84% より良い）
- E/round 5.742（A 4.731）、round_ms 82.73（A 72.74）、tail acc/round 1.427、hit 0.348

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

- correctness: Gate 2A PASS、A / B / C8 は全 30 run で target parity 100%
- candidate tok/s ≥ baseline + 2%: **C8 = +5.84%**（n=8 で +6.76%）paired median
- `tail_accepted_per_round` = 1.534 > 0
- +5% 以上 → **strong direct GO**

ただし次の 2 点がこの GO の範囲を狭める。

1. **GO は json PP512 のみ**。残り 5 workload は -0.32% 〜 -9.70% で、
   `E_ratio > L_ratio` を満たさない。反復構造が強い workload 限定の GO である。
2. **T = 16 / 32 は測定不能（hard stop）**。verify rows > 16 の R64 bucket で
   target parity が崩れるため、§54 に従い性能値を破棄した。
   Gate 1 が示した「tail 32 まで伸ばす」価値は、この gap が解消するまで検証できない。

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
2. **json PP512（反復構造）を対象に Gate 3（production 統合）へ進む**。
   推奨 parameter は `DFlash2 K=7 + NgramTail T=8 / n=5`（n=8 も候補）、window 2048、
   Exact、`PHASESHIFT_TARGET_LM_HEAD_PROXY=0` のまま。
3. **composite seed の改善を検討する**（次 Gate）。K を減らす（K=3 + T=16 のような構成）と
   seed に committed token が混じって hit が増えるが、`E/round` が下がるトレードオフが
   ある。Gate 2A の `k3_t4`（json PP512 で hit 29/61 round、tail 81 token）は有望。
4. production CLI 追加は行わない（§59）。DIRECT GO でも production contract は Gate 3 で決める。

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

