# NgramTail Gate 1 — Qwen3.8-27B committed-history tail extension PoC

## 0. 目的

Qwen3.8-27B の target-only greedy 出力に対して、committed history のみを参照する
NgramTail proposer を offline replay し、speculative draft の「末尾延長」として
有効な過去 continuation が存在するかを判定する。

この Gate は性能 Gate ではない。答える質問は 1 つだけである:

> Qwen3.8-27B の実際の出力には、target verify 幅を広げてまで利用する価値のある
> 過去 continuation が存在するか？

production runtime への統合、DFlash2 / MTP runtime / GDN transaction の変更、
GPU n-gram lookup は全てこの Gate の範囲外である。

## 1. 現行実装との違い

Gate 1 では 2 種類の n-gram を混同しない。

    Current implementation（補充型）
        dynamic MTP early-stop
              ↓
        Ngram fills unused slots
              ↓
        total draft <= MTP Kmax

    Gate 1 target（tail extension 型）
        committed-history NgramTail
              ↓
        arbitrary tail extension
              ↓
        total draft may exceed base MTP K

現行の `append_ngram_tail()` は MTP の不足分を補う実装である
（`MTP Kmax = 8` / early-stop = 3 のとき最大 5 token 追加、total draft は 8 以下）。
Gate 1 は「MTP K + NgramTail T = 20 token」のような延長を対象とする。
`docs/rnd/mtp/optimization_history.md` に記録された既存の改善は前者
（Kmax 内の空き枠補充）であり、後者の潜在能力は未測定である。

## 2. 実装

propose_ngram_tail() を純粋 CPU 関数として切り出した。

- `include/phaseshift/models/qwen35/runtime/ngram_tail.h`
- `src/phaseshift/models/qwen35/runtime/ngram_tail.cpp`

```cpp
struct NgramTailConfig {
    uint32_t n = 4u;
    uint32_t max_tail = 16u;
    uint32_t window = 2048u;
};

struct NgramTailProposal {
    bool hit = false;
    uint32_t match_position = 0u;
    uint32_t seed_length = 0u;
    uint32_t candidate_start = 0u;
    uint32_t candidate_end = 0u;
    uint32_t current_position = 0u;
    std::vector<int32_t> tokens;
};

Result<NgramTailProposal> propose_ngram_tail(
    std::span<const int32_t> committed_history,
    const NgramTailConfig& config);
```

- fixed-n exact match、linear scan。
- candidate 探索範囲は `[max(0, size - window), size - n - 1]`。
  current suffix 自身は candidate にしない。
- tie-break は「最も最近の exact match」（後方から探索して最初に一致したもの）。
- proposal は `candidate_end` 以降かつ `committed_history` 内に存在する token のみ。
  future token は参照しない。
- `candidate_end + tokens.size() <= current_position` を assert で固定
  （self-extension 禁止）。
- `n == 0` は `Status::invalid_argument`。
  `n > history` / `max_tail == 0` / `window == 0` は no-hit。

### 現行 `append_ngram_tail()` への接続

`src/phaseshift/models/qwen35/runtime/spec_decoder.cpp` の検索コードを
`propose_ngram_tail()` で置換した。draft 追加は `k_eff` 以下で打ち切るため
`total drafts <= k_eff` という現行制限は維持される。

探索窓の基点だけは `hist`（committed history）の末尾から
`seq = hist + pending + MTP drafts` の末尾へ移動した。
差分は `1 + MTP drafts` （最大 9 token）で、候補が**追加**されることはない
（新しい探索範囲は常に旧探索範囲の部分集合）。
実装置換の等価性は乱数 40,000 ケースで確認した。

| 区分 | ケース | 差分 |
| --- | --- | --- |
| 探索窓の基点が一致（`history + 1 + drafts <= window` など） | 40,000 中 39,648 | 0 |
| 探索窓の基点がずれる（history が window を超過） | 352 | 窓端の最大 9 起点分のみ（狭い側） |

## 3. 測定方法

target-only greedy generation で得た token stream を正解列とし、
各 generation position `p` で

    history = prompt + generated[0:p]

を `propose_ngram_tail()` に渡す。proposal の実在未来と比較し、

    accepted = proposal と actual の longest-prefix exact match 長

を数える。proposal のうち generation 末尾で照合できない部分は proposed から
除外する（有限長 oracle の制約であり、実際の verify では継続生成されるため）。

future leakage は replay 中に毎位置 assert する。

    candidate_start  < current_position
    candidate_end   <= current_position
    candidate_end + tokens.size() <= current_position
    tokens == committed_history[candidate_end ...]

違反が出た場合は測定を中断して FAIL とする。

## 4. corpus

既存 `tests/data/mtp_perf/{code,json,prose,reasoning}.psktok` は約 470〜630 token しかなく
PP512 / PP2048 を満たせないため、既存 corpus を先頭に保持したまま同一カテゴリの
実テキストで延長した（`tools/gen_token_corpus.py --base-psktok`）。
先頭 token は既存 corpus と同一なので、短い context での prompt は既存 corpus と一致する。

| カテゴリ | 既存 | 延長後 | 延長テキスト |
| --- | --- | --- | --- |
| code | 516 | 13,313 | PhaseShift `spec_decoder.cpp` / `spec_decode.cpp` / `scheduled_batch.cpp` / `run_required_acceptance.py` |
| json | 629 | 6,993 | `models/Qwen3.8-27B-PSQ/` の `config.json` / `tokenizer_config.json` / `generation_config.json` / `preprocessor_config.json` / `video_preprocessor_config.json` |
| prose | 468 | 23,098 | `models/Qwen3.8-27B-PSQ/README.md` |
| reasoning | 481 | 18,477 | DFlashInfinity の英語 experiment report 3 件 |

生成物は `artifacts/ngram_tail_gate1/corpus/*.psktok`（sha256）:

| ファイル | sha256（先頭 16 桁） |
| --- | --- |
| `code.psktok` | `bca1068c7d9390f3` |
| `json.psktok` | `94b0be419e36349e` |
| `prose.psktok` | `00f3e4043abe652a` |
| `reasoning.psktok` | `d0a0aff02e09c02c` |

## 5. harness

`tests/unit/test_qwen35_ngram_tail_gate1.hip`（optional test、`gpu1;optional;external_files`）。

- target-only generation（MTP forward は実行しない）
- 全 condition を同一 stream に対して offline replay
- `raw.csv` / `aggregate.csv` / `environment.json` 出力

環境変数:

| 変数 | 既定 |
| --- | --- |
| `PHASESHIFT_MODEL_DIR_MTP` | 必須（`PHASESHIFT_MODEL_DIR_4B` に fallback） |
| `PHASESHIFT_NGRAM_TAIL_GATE1_CORPUS` | `tests/data/mtp_perf` |
| `PHASESHIFT_NGRAM_TAIL_GATE1_DIR` | `artifacts/ngram_tail_gate1` |
| `PHASESHIFT_NGRAM_TAIL_GATE1_CONTEXTS` | `512` |
| `PHASESHIFT_NGRAM_TAIL_GATE1_N` | `3,4,5,8` |
| `PHASESHIFT_NGRAM_TAIL_GATE1_TAIL` | `8,16,32` |
| `PHASESHIFT_NGRAM_TAIL_GATE1_WINDOW` | `2048` |
| `PHASESHIFT_NGRAM_TAIL_GATE1_GEN` | `256` |
| `PHASESHIFT_NGRAM_TAIL_GATE1_PROMPTS` | `2` |
| `PHASESHIFT_NGRAM_TAIL_GATE1_ONLY` | 全カテゴリ |

### 5.1 correctness と既存回帰

| 検証 | 結果 |
| --- | --- |
| unit test `test_ngram_tail`（CPU required、8 ケース / 65 check） | PASS |
| required acceptance `python3 tests/run_required_acceptance.py --build-dir build-gfx1201` | **ACCEPTANCE PASSED**（Total 125 / Passed 125 / Failed 0 / Skipped 0） |
| future leakage assert（replay 全 98,304 位置 = 8,192 位置 × 12 条件） | 違反 0 件（違反時は測定を中断し FAIL） |
| 既存 `append_ngram_tail()` との等価性（乱数 40,000 ケース） | 探索窓基点が一致する 39,648 件で差分 **0**。残り 352 件は窓基点の差（旧 `hist` 末尾 / 新 `seq` 末尾、最大 9 起点・狭い側）のみ |
| gate4e 回帰（ngram ON、`PHASESHIFT_VERIFY_EXACT=1`、PP512、TG128、16 run） | `GATE4E_TOKEN_MATCH=16/16` |
| gate4e ngram ON / OFF 比較（同一 corpus / 同一条件） | diverge の idx / spec / target が全 16 run で一致 → **NgramTail は spec 出力に影響しない** |

### 5.2 Fast numeric mode による token mismatch は既存問題

gate4e の既定（`VerifyNumericMode::Fast`）では spec ON/OFF とも `GATE4E_TOKEN_MATCH=1/16`
（既存 `tests/data/mtp_perf` corpus でも `1/8`）と、target-only 参照と一致しない。
原因は M（verify 行数）依存の kernel 選択で、`rows == 1` の GEMV と `rows >= 2` の WMMA GEMM で
K 還元順序が異なり 1 ULP の差が argmax を反転させるためである。

- `docs/rnd/mtp/optimization_history.md` §7.64 / §7.65（VERIFY_EXACT の導入と verify role 限定）
- `docs/rnd/dflash2/dflash2.md`（Fast では 6/6 mismatch、Exact を既定）

MTP 固有の問題ではなく DFlash2 path も同じ挙動である。
Gate 1 はこの問題に触れておらず、`PHASESHIFT_VERIFY_EXACT=1` で既存の exactness が
維持されることを確認した。

## 6. 結果

### 6.1 測定条件

| 項目 | 値 |
| --- | --- |
| revision | baseline `1613544f` → final `0d7093e8`（branch `poc/ngram-tail-gate1`） |
| GPU | AMD Radeon AI PRO R9700（gfx1201） |
| model | Qwen3.8-27B-PSQ（`PHASESHIFT_MODEL_DIR_MTP`） |
| quantization | PSQ（`phaseshift_quantization.json` あり） |
| sampling | greedy（`VerifyNumericMode::Fast` の target-only） |
| corpus | `artifacts/ngram_tail_gate1/corpus/*.psktok`（§4） |
| generation length | TG256（§6.6 に TG512） |
| context | PP32 / PP128 / PP512 / PP2048 |
| prompts / category / context | 2 |
| parameter sweep | n ∈ {3,4,5,8} × max_tail ∈ {8,16,32}、window = 2048（12 条件） |
| raw artifact | `artifacts/ngram_tail_gate1/{raw.csv,aggregate.csv,environment.json}` |
| TG512 artifact | `artifacts/ngram_tail_gate1/tg512/`（PP512 / PP2048） |
| window sweep artifact | `artifacts/ngram_tail_gate1/window_sweep/`（window 512 / 2048 / 8192） |
| 初回測定 artifact | `artifacts/ngram_tail_gate1/initial/`（PP512 のみ） |

positions = 生成位置数 = 4 カテゴリ × 4 context × 2 prompt × 256 = **8,192 / 条件**。
raw.csv は全 12 条件を束ねた **98,304 行**（`GATE1_STREAMS=32 ROWS=98304 AGGS=192`）。

### 6.2 主要結果（n=5 / max_tail=32 / window=2048、TG256）

| category | context | hit_rate | acceptance_rate | accepted/pos | accepted/hit | P(>=4\|hit) | P(>=8\|hit) | P(>=16\|hit) | ideal E/update |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| code | PP32 | 0.191 | 0.098 | 0.287 | 1.500 | 0.122 | 0.020 | 0.000 | 1.287 |
| code | PP128 | 0.568 | 0.466 | 7.498 | 13.192 | 0.656 | 0.512 | 0.392 | 8.498 |
| code | PP512 | 0.250 | 0.278 | 1.139 | 4.555 | 0.578 | 0.234 | 0.000 | 2.139 |
| code | PP2048 | 0.328 | 0.162 | 1.438 | 4.381 | 0.375 | 0.196 | 0.071 | 2.438 |
| json | PP32 | 0.131 | 0.060 | 0.166 | 1.269 | 0.104 | 0.000 | 0.000 | 1.166 |
| json | PP128 | 0.150 | 0.066 | 0.217 | 1.442 | 0.143 | 0.000 | 0.000 | 1.217 |
| json | PP512 | 0.855 | 0.582 | 7.688 | 8.986 | 0.692 | 0.352 | 0.178 | 8.688 |
| json | PP2048 | 0.344 | 0.135 | 1.361 | 3.960 | 0.369 | 0.153 | 0.045 | 2.361 |
| prose | PP32 | 0.264 | 0.268 | 1.896 | 7.193 | 0.341 | 0.215 | 0.178 | 2.896 |
| prose | PP128 | 0.020 | 0.028 | 0.018 | 0.900 | 0.000 | 0.000 | 0.000 | 1.018 |
| prose | PP512 | 0.135 | 0.071 | 0.176 | 1.304 | 0.087 | 0.000 | 0.000 | 1.176 |
| prose | PP2048 | 0.580 | 0.561 | 9.408 | 16.219 | 0.714 | 0.603 | 0.502 | 10.408 |
| reasoning | PP32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| reasoning | PP128 | 0.020 | 0.061 | 0.027 | 1.400 | 0.100 | 0.000 | 0.000 | 1.027 |
| reasoning | PP512 | 0.076 | 0.064 | 0.109 | 1.436 | 0.154 | 0.000 | 0.000 | 1.109 |
| reasoning | PP2048 | 0.529 | 0.277 | 4.008 | 7.572 | 0.513 | 0.347 | 0.151 | 5.008 |

`ideal E/update` は verify cost を無視した上限値であり、実測性能ではない。
（`ideal = accepted_per_position + 1`。no-hit 位置は 1 扱い。）

### 6.3 cell ごとの最良条件（accepted/pos 最大）

| category | context | n | tail | hit_rate | accepted/pos | accepted/hit | P(>=8\|hit) | ideal E/update |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| code | PP32 | 3 | 16 | 0.441 | 0.693 | 1.571 | 0.018 | 1.693 |
| code | PP128 | 5 | 32 | 0.568 | 7.498 | 13.192 | 0.512 | 8.498 |
| code | PP512 | 3 | 16 | 0.438 | 1.684 | 3.848 | 0.232 | 2.684 |
| code | PP2048 | 3 | 32 | 0.600 | 1.705 | 2.844 | 0.101 | 2.705 |
| json | PP32 | 3 | 8 | 0.346 | 0.436 | 1.260 | 0.000 | 1.436 |
| json | PP128 | 3 | 16 | 0.395 | 0.559 | 1.416 | 0.020 | 1.559 |
| json | PP512 | 8 | 32 | 0.801 | 8.262 | 10.317 | 0.424 | 9.262 |
| json | PP2048 | 3 | 32 | 0.455 | 1.525 | 3.352 | 0.120 | 2.525 |
| prose | PP32 | 8 | 32 | 0.191 | 2.332 | 12.184 | 0.418 | 3.332 |
| prose | PP128 | 3 | 8 | 0.084 | 0.062 | 0.744 | 0.000 | 1.062 |
| prose | PP512 | 3 | 8 | 0.273 | 0.441 | 1.614 | 0.000 | 1.441 |
| prose | PP2048 | 8 | 32 | 0.535 | 9.709 | 18.142 | 0.693 | 10.709 |
| reasoning | PP32 | 3 | 8 | 0.020 | 0.006 | 0.300 | 0.000 | 1.006 |
| reasoning | PP128 | 3 | 8 | 0.074 | 0.076 | 1.026 | 0.000 | 1.076 |
| reasoning | PP512 | 3 | 8 | 0.205 | 0.299 | 1.457 | 0.010 | 1.299 |
| reasoning | PP2048 | 5 | 32 | 0.529 | 4.008 | 7.572 | 0.347 | 5.008 |

### 6.4 観察

- **hit 率は履歴量より prompt / 生成の反復性に支配される。**
  同一 context 内でも prompt によって accepted/pos が 1 桁以上異なる
  （例: code PP512 は prompt0 = 0.016 / prompt1 = 2.262、
  json PP512 は 7.633 / 7.742）。context を増やしても反復のない生成では増えない。
- max_tail を伸ばすと `acceptance_rate`（proposed に対する比率）は下がるが、
  `accepted_per_position` は増える。判定に効くのは後者である。
- n を大きくすると hit 率は必ず下がるが、hit 時の `accepted/hit` は上がる
  （偽陽性が減るため）。json PP512 では n=8 が最良。
- prose / reasoning は PP128・PP512 で弱いが、NO-GO 条件は満たさない
  （§7）。反復構造に強い方式という前提と整合する。

### 6.5 lookup CPU 時間（参考値）

linear scan のままでも 1 位置あたり次のように収まる（GO 条件にはしない）。

| context | mean | min | max |
| --- | --- | --- | --- |
| PP32 | 75 ns | 57 ns | 88 ns |
| PP128 | 111 ns | 72 ns | 150 ns |
| PP512 | 201 ns | 59 ns | 286 ns |
| PP2048 | 550 ns | 355 ns | 763 ns |

### 6.6 TG512（長い generation）

TG256 と同一条件で generation を 512 に延長したもの（PP512 / PP2048、
`artifacts/ngram_tail_gate1/tg512/`）。generation が長いほど履歴再利用が増えるため、
code / prose / reasoning は TG256 より改善する。
json は生成後半の内容に依存して TG256 を下回る（例: json PP512 は 7.688 → 5.792）。

n=5 / max_tail=32 / window=2048:

| category | context | hit_rate | acceptance_rate | accepted/pos | accepted/hit | P(>=4\|hit) | P(>=8\|hit) | P(>=16\|hit) | ideal E/update |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| code | PP512 | 0.334 | 0.249 | 1.504 | 4.503 | 0.556 | 0.249 | 0.000 | 2.504 |
| code | PP2048 | 0.405 | 0.187 | 2.267 | 5.593 | 0.431 | 0.241 | 0.116 | 3.267 |
| json | PP512 | 0.688 | 0.482 | 5.792 | 8.413 | 0.660 | 0.353 | 0.155 | 6.792 |
| json | PP2048 | 0.340 | 0.106 | 1.020 | 3.000 | 0.293 | 0.095 | 0.023 | 2.020 |
| prose | PP512 | 0.216 | 0.074 | 0.339 | 1.570 | 0.122 | 0.000 | 0.000 | 1.339 |
| prose | PP2048 | 0.693 | 0.533 | 10.625 | 15.324 | 0.714 | 0.587 | 0.458 | 11.625 |
| reasoning | PP512 | 0.142 | 0.098 | 0.365 | 2.579 | 0.331 | 0.041 | 0.000 | 1.365 |
| reasoning | PP2048 | 0.593 | 0.249 | 4.300 | 7.254 | 0.489 | 0.331 | 0.153 | 5.300 |

GO 条件の充足（code / json・PP512 以上、各 cell 12 条件中 A / B / C が成立した条件数）:

| cell | TG256 A | TG256 B | TG256 C | TG512 A | TG512 B | TG512 C |
| --- | --- | --- | --- | --- | --- | --- |
| code PP512 | 9 | 9 | 9 | 9 | 9 | 9 |
| code PP2048 | 11 | 3 | 11 | 12 | 6 | 12 |
| json PP512 | 12 | 12 | 12 | 12 | 12 | 12 |
| json PP2048 | 11 | 3 | 11 | 6 | 0 | 6 |

A（`accepted_tokens_per_position >= 1.0`）と C（`ideal E/update >= 2.0`）は
`ideal = accepted_per_position + 1` という定義から同値である。
B（`P(accepted >= 8 | hit) >= 20%`）は `n = 8` と PP512 で安定して成立し、
PP2048 では hit 率が下がるため成立条件が限られる。

TG512 の最良条件（accepted/pos 最大）:

| category | context | n | tail | hit_rate | accepted/pos | accepted/hit | P(>=8\|hit) | ideal E/update |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| code | PP512 | 3 | 16 | 0.509 | 2.188 | 4.301 | 0.261 | 3.188 |
| code | PP2048 | 3 | 32 | 0.652 | 2.451 | 3.757 | 0.139 | 3.451 |
| json | PP512 | 8 | 32 | 0.612 | 6.272 | 10.244 | 0.450 | 7.272 |
| json | PP2048 | 3 | 32 | 0.515 | 1.344 | 2.611 | 0.078 | 2.344 |

### 6.7 window sweep（512 / 2048 / 8192、TG256）

`artifacts/ngram_tail_gate1/window_sweep/`。n=5 / max_tail=32 の例:

| category | context | window | hit_rate | accepted/pos | accepted/hit | P(>=8\|hit) | ideal E/update |
| --- | --- | --- | --- | --- | --- | --- | --- |
| code | PP512 | 512 | 0.250 | 1.139 | 4.555 | 0.234 | 2.139 |
| code | PP512 | 2048 | 0.250 | 1.139 | 4.555 | 0.234 | 2.139 |
| code | PP512 | 8192 | 0.250 | 1.139 | 4.555 | 0.234 | 2.139 |
| code | PP2048 | 512 | 0.270 | 1.115 | 4.138 | 0.174 | 2.115 |
| code | PP2048 | 2048 | 0.328 | 1.438 | 4.381 | 0.196 | 2.438 |
| code | PP2048 | 8192 | 0.328 | 1.438 | 4.381 | 0.196 | 2.438 |
| json | PP2048 | 512 | 0.293 | 1.223 | 4.173 | 0.167 | 2.223 |
| json | PP2048 | 2048 | 0.344 | 1.361 | 3.960 | 0.153 | 2.361 |
| json | PP2048 | 8192 | 0.344 | 1.361 | 3.960 | 0.153 | 2.361 |
| reasoning | PP2048 | 512 | 0.367 | 1.426 | 3.883 | 0.213 | 2.426 |
| reasoning | PP2048 | 2048 | 0.529 | 4.008 | 7.572 | 0.347 | 5.008 |
| reasoning | PP2048 | 8192 | 0.529 | 4.008 | 7.572 | 0.347 | 5.008 |

- PP512 では window の差が出ない（履歴 768 token に対し直近 512 で足りる）。
- PP2048 では window=512 が明確に劣化し、**2048 と 8192 は同一結果**。
  履歴は最大 2,304 token で、window=2048 が外すのは prompt 先頭の 256 token のみで、
  そこには有効な候補が無かった。
- **既定の `window = 2048` で十分**である（8192 へ広げても改善なし、検索コストだけ増える）。

## 7. GO / NO-GO

### 7.1 判定: **GO**（候補品質）

GO 条件 A / B / C を満たす parameter set が存在する（成立条件数は §6.6。
code / json・PP512 以上の 48 条件で、A = C は TG256 / TG512 で 43 / 39、B は 27 / 27）。

- **A**（`accepted_tokens_per_position >= 1.0`）:
  code / json の PP512・PP2048 全 4 cell に成立条件が存在する。
  成立数が最少なのは TG512 の json PP2048（12 条件中 6、最大 1.344）で、
  ここは `n = 3`（全 tail）と `n = 5 / tail = 32` が成立する。
- **B**（`P(accepted >= 8 | hit) >= 20%`）:
  json PP512 は 12/12（最大 0.424 / 0.450）、code PP512 は 9/12（最大 0.256 / 0.269）。
  PP2048 では code のみ成立（TG256 3/12・最大 0.364、TG512 6/12・最大 0.354）で、
  json PP2048 は TG512 で最大 0.186 にとどまる。
- **C**（`ideal E/update >= 2.0`）は A と同値である（`ideal = accepted_per_position + 1`）。

GO は A / B / C の**いずれか**の成立で判定するため、この結果は GO に十分である。

**強い GO**（`ideal >= 3.0` かつ `P(accepted >= 8 | hit)` が高い）も
`json PP512 / n=8 / tail=32`（ideal 9.26、P(>=8|hit) 0.424、TG256）と
`json PP512 / n=8 / tail=32`（ideal 7.27、P(>=8|hit) 0.450、TG512）で満たす。

**NO-GO 条件は該当なし。** prose / reasoning は PP128・PP512 で弱いが、
NO-GO の定義（全カテゴリ・長文脈で hit が極端に低く `accepted_tokens_per_position < 0.25`、
tail を伸ばしても accepted が増えない）を満たさない。
prose / reasoning も PP2048 の最良条件では `accepted_per_position` が
それぞれ 10.63 / 4.30（TG512、n=5 / tail=32）に達しており、
履歴が足りれば反復構造以外のカテゴリでも再利用が起きる。

### 7.2 この Gate で言えること / 言えないこと

- **言える**: Qwen3.8-27B の実際の出力には、target verify 幅を広げてまで使う価値のある
  過去 continuation が存在する。提案品質としては実 verify へ進む価値がある。
- **言えない**: 速度向上。`ideal E/update` は verify cost を無視した上限値であり、
  実際には追加した T 行の target verify コストを伴う（§29）。

### 7.3 Gate 2 推奨構成

Gate 2 の drafter は **DFlash2** を使う（AGENTS.md の現行方針、`docs/developer/dflash2.md`）。

| 項目 | 推奨 |
| --- | --- |
| baseline | DFlash2 K（現行） |
| candidate | DFlash2 K + NgramTail T |
| Ngram n | 5 |
| Ngram tail | 16（A/B で 32 も測定） |
| window | 2048 |
| verify numeric mode | **Exact**（既定を維持） |

根拠:

- NgramTail は committed history だけを見る proposer であり drafter に依存しない。
  Gate 1 は target-only oracle で MTP forward を一切実行していないため、
  得られた proposer の有効性はそのまま DFlash2 へ移せる。
- production の performance target の drafter は DFlash2 である。
- MTP の draft 品質は既存計測で static K1 のみが優位、dynamic policy は劣化、
  n-gram 補充も PP512 で E/update +5.8% にとどまる
  （`docs/rnd/mtp/optimization_history.md` §7.77）。verify 幅を T だけ広げる用途には
  接受率の高い drafter を使う方が有利。
- **verify numeric mode は Exact のままにする。** Fast は M（verify 行数）依存の
  数値差で token mismatch を起こす（§5.2）。Gate 2 で verify 幅が `K + T` へ
  広がることを考えると、Fast の影響は Gate 1 より大きくなる。
  DFlash2 path はすでに Exact を既定としている。

**現行の `append_ngram_tail()`（MTP 空き枠補充型）は、DFlash2 + NgramTail extension が
実装された時点で production からの扱いを改めて判断する。**

## 8. 再現手順

```bash
cmake -S . -B build-gfx1201 -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DPHASESHIFT_ROCM_ROOT=$ROCM \
    -DPHASESHIFT_BUILD_TESTS=ON -DPHASESHIFT_BUILD_OPTIONAL_TESTS=ON

cmake --build build-gfx1201 --parallel
ctest --test-dir build-gfx1201 --output-on-failure -R test_ngram_tail
python3 tests/run_required_acceptance.py --build-dir build-gfx1201
```

corpus（既存 `tests/data/mtp_perf/*.psktok` を先頭に保持して延長、延長テキストは §4）:

```bash
python3 tools/gen_token_corpus.py --model-dir models/Qwen3.8-27B-PSQ \
    --base-psktok tests/data/mtp_perf/code.psktok \
    --text-file <延長テキスト...> \
    --output artifacts/ngram_tail_gate1/corpus/code.psktok
```

測定（GPU は他 workload と共有しない）:

```bash
HIP_VISIBLE_DEVICES=0 \
PHASESHIFT_MODEL_DIR_MTP=models/Qwen3.8-27B-PSQ \
PHASESHIFT_NGRAM_TAIL_GATE1_CORPUS=artifacts/ngram_tail_gate1/corpus \
PHASESHIFT_NGRAM_TAIL_GATE1_DIR=artifacts/ngram_tail_gate1 \
PHASESHIFT_NGRAM_TAIL_GATE1_CONTEXTS=32,128,512,2048 \
./build-gfx1201/tests/test_qwen35_ngram_tail_gate1
```

TG512 は `PHASESHIFT_NGRAM_TAIL_GATE1_GEN=512` と
`PHASESHIFT_NGRAM_TAIL_GATE1_DIR=.../tg512`、
window sweep は `PHASESHIFT_NGRAM_TAIL_GATE1_WINDOW=512,2048,8192` と
`PHASESHIFT_NGRAM_TAIL_GATE1_DIR=.../window_sweep` を指定する。

既存実装との回帰確認（gate4e）:

```bash
HIP_VISIBLE_DEVICES=0 PHASESHIFT_MODEL_DIR_MTP=models/Qwen3.8-27B-PSQ \
PHASESHIFT_MTP_GATE4E_DIR=<root> PHASESHIFT_MTP_GATE4E_LENGTHS=512 \
PHASESHIFT_MTP_GATE4E_GEN=128 PHASESHIFT_MTP_GATE4E_PROMPTS_PER_BUCKET=1 \
PHASESHIFT_MTP_NGRAM_N=4 PHASESHIFT_MTP_NGRAM_TAIL=8 \
PHASESHIFT_VERIFY_EXACT=1 \
./build-gfx1201/tests/test_qwen35_mtp_gate4e
```

`<root>/corpus/*.psktok` に corpus を置く。
`PHASESHIFT_VERIFY_EXACT=1` なしでは既知の Fast numeric divergence で
`GATE4E_TOKEN_MATCH` が一致しない（§5.2）。
