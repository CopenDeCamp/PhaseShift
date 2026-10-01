# Constraint-aware LM Head Hybrid（Gate 4）

> Status: R&D record（Gate の経緯・採否判断・計測値）。現在の contract は
> `docs/developer/qwen35.md` §7.1、mask と allowed count の受け渡しは
> `docs/developer/runtime.md`、trace 出力は
> `docs/developer/structured_generation.md` を正本とする。

## 0. 目的

constraint 付き greedy decode では、allowed token が 3 件でも 20 件でも
full vocab（248320）の PSQ8 LM Head を計算していた。これを

- **Case A（ExactCandidates）**: allowed token を列挙して PSQ8 をその候補だけ評価する
- **Case B（MaskedCoarse）**: INT2 coarse full-vocab → constraint mask → Radix Top-N →
  PSQ8 exact rerank

に分岐し、LM Head の計算量を allowed token 数まで下げる。

前提は Gate 1（INT2 coarse）/ Gate 2（Radix Top-N）/ Gate 3（Decode greedy proxy と
shadow）。

## 1. 実装した内容

### 1.1 allowed token count の追跡

mask 生成直後に host 側で `std::popcount` を合算し、vocab 末尾の padding bit は
数えない。GPU への転送・D2H・追加同期は無い。

- `ScheduledRequest::constraint_allowed_count`（unconstrained は `UINT32_MAX`）
- `Executor::constraint_allowed_counts_host`（`max_scheduled_output_rows` 分を事前確保）
- `HostExecutionContext::constraint_allowed_counts`（output row 単位）

### 1.2 分岐

`select_lm_head_constrained(...)` が 1 箇所で決める。

- `role == Decode`、`stochastic_outputs == 0`、`sampled == outputs`
- 全 output row が constraint 付き、全 row の allowed count が 1 以上
- `max(allowed) <= small_threshold` → ExactCandidates
- `min(allowed) > small_threshold` かつ `min(allowed) >= coarse_pool` → MaskedCoarse
- それ以外（mixed な constrained / unconstrained、small / large 混在、allowed 0）は
  full path へ fallback

`small_threshold` は `PHASESHIFT_TARGET_LM_HEAD_PROXY_CONSTRAINT_THRESHOLD`
（既定 128 = `kDflash2Int2MaxPool`）。`coarse_pool` 条件は、Top-N の候補が埋まらないと
未使用スロットに不正な ID が入るため（radix は `id=0`、legacy は `id=-1`）。

### 1.3 ExactCandidates

`launch_constraint_mask_to_candidates` が mask を token ID 昇順の候補へ展開し、
残りを `candidate_ids[0]` で埋める（既存 PSQ8 rerank の fixed pool をそのまま再利用）。
`launch_dflash2_psq8_candidate_rerank` → pool argmax。**INT2 coarse は計算しない。**

候補集合が全 allowed token を含む限り full PSQ8 の constrained argmax と同一。

### 1.4 MaskedCoarse

`launch_dflash2_apply_constraint_mask`（DFlash2 と共有）で許可外を `-INFINITY` に
潰してから Radix Top-N / legacy Top-N → PSQ8 exact rerank → argmax。
INT2 は candidate 生成にのみ使用し、final score は常に PSQ8。

### 1.5 shadow

mode=2 では同じ経路の候補を `shadow_ids_` へ書き、production output は full PSQ8 側。
decisions / mismatches / mismatch_rate は Gate 3 の shadow 機構をそのまま使う。
計測 instrument は `PHASESHIFT_TARGET_LM_HEAD_PROXY_TIMING`（quant / extract / rerank /
argmax / total）。

### 1.6 test

- `tests/unit/test_lm_head_proxy_path.cpp`（cpu;required、35 check）
  分岐条件・境界・fallback 全種
- `tests/unit/test_constraint_candidates.hip`（gpu1;required、76 check）
  mask → candidate 展開（allowed 1〜128、sparse / contiguous / vocab 末尾 / padding 有無）、
  full PSQ8 との token exact、tie での下位 token 優先、masked Top-N の candidate
  containment（radix / legacy 双方、allowed >= pool）
- `tests/unit/test_lm_head_constraint_perf.hip`（gpu1;optional;external_files）
  実モデルでの計測

## 2. 測定条件（Small / Large path 比較）

| 項目 | 値 |
| --- | --- |
| GPU | R9700（gfx1201）1 枚 |
| モデル | `Qwen3.8-27B-PSQ` |
| vocab / cols | 248320 / 5120 |
| rows | 4 |
| 既定 pool | 32 |
| warmup / iters | 20 / 100 |
| 比較 A | activation quantize + full PSQ8 GEMM + constraint 付き argmax |
| 比較 B | `select_constrained_exact` / `select_constrained_masked` |

## 3. 結果

### 3.1 ExactCandidates（allowed <= 128）

| allowed | full PSQ8 (us) | exact candidate (us) | speedup | token |
| --- | --- | --- | --- | --- |
| 1 | 2360 | 261 | 9.0× | exact |
| 4 | 2362 | 284 | 8.3× | exact |
| 8 | 2362 | 283 | 8.4× | exact |
| 16 | 2362 | 268 | 8.8× | exact |
| 32 | 2362 | 282 | 8.4× | exact |
| 48 | 2362 | 292 | 8.1× | exact |
| 64 | 2361 | 293 | 8.1× | exact |
| 96 | 2362 | 295 | 8.0× | exact |
| 128 | 2362 | 295 | 8.0× | exact |

全 allowed count で full PSQ8 と token exact。cost は allowed count にほぼ依存しない
（candidate extraction が 128 件で止まるため）。

### 3.2 MaskedCoarse（allowed > 128、pool=32）

| allowed | full PSQ8 (us) | masked path (us) | speedup | token |
| --- | --- | --- | --- | --- |
| 129 | 2362 | 1160 | 2.0× | match |
| 256 | 2362 | 1160 | 2.0× | match |
| 512 | 2363 | 1163 | 2.0× | match |
| 1024 | 2369 | 1165 | 2.0× | match |
| 4096 | 2429 | 1166 | 2.1× | match |
| 16384 | 2537 | 1177 | 2.2× | **2/4 行が mismatch** |

`allowed = 16384` で INT2 Top-32 に PSQ8 の真の argmax が入らない miss が発生。
**pool=64 と pool=128 では同じ条件で全件 match**（miss は pool=32 のみ）。

### 3.3 crossover

Exact path の cost（0.26〜0.30ms）は Masked path（1.16ms）より常に低いが、
候補数が 128 を超えると Exact path は使えない。つまり

- allowed <= 128 → Exact path が最速で、かつ mathematically exact
- allowed > 128 → Masked path は full PSQ8 に対して約 2 倍。ただし INT2 miss の
  qualification が必要

既定 `small_threshold = 128` は測定と整合する。

### 3.4 実 grammar での allowed count

`phaseshift-compute --serve-stdio` に GBNF grammar（`YES`/`NO`、JSON 固定、
300 文字固定、三択）を送り、`CONSTRAINT_ALLOWED_COUNT` を取った。

| grammar | p10 | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| YES/NO | 1 | 2 | 5 | 5 |
| JSON 固定 / 長い固定文字列 | 1 | 2 | 5 | 7 |

token 境界に依存するが、この種の grammar はすべて Small exact の範囲（<= 128）に
収まる。生成列が短い（ASCII 数文字）ため、後半 step では allowed が 1 になる。

### 3.5 E2E（proxy OFF / FAST / shadow）

`test_compute_constraints.py` を `PHASESHIFT_TARGET_LM_HEAD_PROXY` = 0 / 1 / 2 で
実行した。

| mode | 結果 |
| --- | --- |
| 0（OFF） | 18 PASS / 0 FAIL |
| 1（FAST） | 18 PASS / 0 FAIL。生成列は OFF と完全一致 |
| 2（shadow） | 18 PASS / 0 FAIL。生成列は OFF と完全一致 |

shadow 統計は `decisions=24 mismatches=0 mismatch_rate=0`。つまり
constraint 付き greedy の全 decode step で、Exact path が full PSQ8 と同じ token を
選んだ。

### 3.6 統合時に見つかった実装欠陥

E2E で constrained hybrid が一切発火しないことが判明した。原因は Gate 4 で
`Executor` に追加した `constraint_allowed_counts_host` が
`Executor` の move constructor と move assignment のどちらでも移動されていなかった
こと。`create_model_executor` の戻り値は move assignment でメンバへ入るため、
production の executor ではこの vector が常に空で、`select_lm_head_constrained` が
必ず `None` を返していた。

kernel 単体テストと bench は proxy を直接叩くためこの欠落を検出できない。
`docs/developer/qwen35.md` の E2E で初めて現れる。
修正は `fix: carry constraint and radix workspaces through executor move`
（同じ欠陥が Gate 5 の radix workspace にもあった）。再発防止として
`test_executor_move_assignment` を move の両経路と追加メンバを検証するよう拡張した。

## 4. 判断

| 項目 | 判断 |
| --- | --- |
| Small exact（allowed <= 128） | **GO** — token exact、8〜9倍、実 grammar の allowed count がこの範囲に収まる、E2E shadow mismatch 0 |
| Large INT2（allowed > 128） | **HOLD** — pool=32 で miss。shadow mismatch の定量化と pool の再検討が要る |

Gate 4 全体を NO-GO にする必要はない。既定設定では Large path は
`min(allowed) > 128` のときだけ使われる。既定運用で主に使われる Small exact は
GO。

## 5. 未実施

- Large path の shadow mismatch 定量化（複数 seed / 複数 grammar）
