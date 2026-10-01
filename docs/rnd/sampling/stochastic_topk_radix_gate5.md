# Stochastic Top-K Radix Active Set（Gate 5）

> Status: R&D record（Gate の経緯・採否判断・計測値）。現在の contract は
> `docs/developer/sampling.md` を正本とする。

## 0. 目的

`top_k > 0` の stochastic sampling は、Top-K 集合を事前に作らずに
「full vocab から Gumbel-Max → Top-K 外なら reject → 再試行」を繰り返していた。
Top-K mass が小さいと rejection 概率が上がり、full vocab scan を何十回も行う。

Gate 5 の目的は sort を速くすることではなく、**rejection loop を無くすこと**である。

## 1. 実装した内容

### 1.1 経路

```
exact F32 Radix Top-K（score 降順、同値なら token ID 昇順）
    ↓
full-softmax prefix mass で active set を決定
    ↓
active set 上で 1 回だけ Gumbel-Max（attempt = 0）
```

- Top-K selection は常に exact F32。**INT2 は使わない。**
- `top_k` 単独（`top_p == 1`）では full vocab の `Z` を計算しない。
  row max の finite 検査と NaN 検査だけ full vocab を 1 回走査する。
- `top_k` + `top_p` では full vocab の `Z` を 1 回だけ計算し、
  Top-K 内の prefix が `top_p * Z` 未満の位置までを active set にする。
  prefix が 0 の token は必ず入る。
- RNG key は `(seed, sample_index, attempt = 0, token_id)`。
  token ID を key にするため batch 位置や Radix の内部順に依存しない。

### 1.2 eligibility（host 側で計算）

`submit_co_batch` が `ScheduledRequest::sampling` から判定し、
`HostExecutionContext::stochastic_topk_eligible` / `stochastic_top_k` で渡す。
device 上の params を D2H して判断しない。

- 全 output row が stochastic（`sampled == stochastic == outputs`）
- 全 row の `top_k` が同じ値で `1 <= top_k <= 128`
- constraint が無い
- radix workspace が確保済み

成立した batch のみ `DeviceSamplingParams.reserved[0]` の top-k path フラグ
（`kSamplingTopKPathFlag = 4`）を全 row に立てる。correctness backend も同じフラグを
見て reference（simple exact Top-K）へ切り替えるため、両 backend が同じ方式を使う。

それ以外（`top_k = 0`、`top_k > 128`、`top_k` が row ごとに異なる、stochastic 混在、
constraint 付き）は従来の `StochasticSingleBlock`（rejection path）へ fallback する。

### 1.3 invalid logits

従来契約と同じ結果になる。

- row max が finite でない / `lmax <= -3.4e38`（全 `-Inf`）→ `DISPATCH_FAILED`
- NaN を含む → `DISPATCH_FAILED`
- `+Inf` を含む → `DISPATCH_FAILED`
- `top_p < 1` では `Z` の finite 検査も従来どおり実施

### 1.4 seed と生成列

active-set path と従来 path は最終分布を同じにできるが、**同じ seed でも token 列は
一致しない**。従来は reject した attempt の RNG を消費してから次へ進むが、
新方式は `attempt = 0` だけで決まるため。

維持する互換性は「同じ実装 + 同じ seed + 同じ sample_index + 同じ logits なら bit exact」
「batch 内の位置が変わっても同じ」「分布が同じ」の 3 点のみ。
Gate 導入前の historical な seed → token 列は保証しない。

### 1.5 test

- `tests/unit/test_qwen35_sampling_selector.cpp`（cpu;required、31 check）
  eligibility / workspace / `top_k` 範囲 / mixed の fallback
- `tests/kernels/optimized/test_stochastic_topk.hip`（gpu1;required、47 check）
  - `top_k = 1 / 2 / 16 / 50 / 128` の membership
  - `top_k >= vocab` でも Top-K 制限が効かないこと
  - 同値 logits での下位 token 優先
  - `top_k` + `top_p` の積集合
  - determinism（100 回 bit exact）・seed 依存
  - batch independence
  - NaN / `+Inf` / `-Inf` → `DISPATCH_FAILED`
  - **reference（simple exact Top-K）と Radix が token bit exact**
  - active set が full-softmax 定義（`full Z` / `better_count` / `better_mass`）と一致
  - 既知分布（0.60 / 0.25 / 0.10 / 0.05）で ±3%
- `tests/kernels/optimized/test_stochastic_sampling.hip`（従来 path、regression なし）

## 2. 測定条件

| 項目 | 値 |
| --- | --- |
| GPU | R9700（gfx1201）1 枚 |
| vocab | 248320 |
| logit pattern | random |
| temperature | 0.7 |
| warmup / samples / launches | 20 / 5 / 10 |
| variant | `stochastic-rejection` / `stochastic-radix` |

`phaseshift-bench sampling` に `stochastic-rejection`（legacy）と `stochastic-radix`
（新）を追加。legacy は average attempts、radix は active_count を出力する。

## 3. 結果（p50、us）

### 3.1 `top_k` 単独（`top_p = 1.0`）

| top_k | out | rejection | radix | speedup |
| --- | --- | --- | --- | --- |
| 20 | 1 | 4700 | 431 | 10.9× |
| 20 | 8 | 255915 | 438 | 585× |
| 50 | 1 | 4699 | 438 | 10.7× |
| 50 | 8 | 77470 | 443 | 175× |
| 64 | 1 | 4664 | 418 | 11.2× |
| 64 | 8 | 61105 | 446 | 137× |

### 3.2 `top_k` + `top_p`（`top_p = 0.9`）

| top_k | out | rejection | radix | speedup |
| --- | --- | --- | --- | --- |
| 20 | 1 | 4670 | 615 | 7.6× |
| 20 | 8 | 255950 | 624 | 410× |
| 50 | 1 | 4715 | 619 | 7.6× |
| 50 | 8 | 77654 | 632 | 123× |
| 64 | 1 | 4694 | 600 | 7.8× |
| 64 | 8 | 61097 | 634 | 96× |

### 3.3 観測

- rejection の average attempts は out=1 で 12、out=8 では
  `top_k=20` で 261、`top_k=50` で 102、`top_k=64` で 72。
  out=8 の `top_k=50` は R&D 記録（`sampling.md`）の
  「約 63863us / attempts 約 102」と同じ傾向。
- radix の cost は `top_k` と `top_p` にほぼ依存せず 418〜634us。
  `top_p = 1.0` のとき 418〜446us、`top_p = 0.9` のとき 600〜634us
  （full vocab の `Z` scan が加わる分）。
- random logits では `top_k = 20..64` の mass が小さく、
  `top_p = 0.9` でも active set は `top_k` 全件だった
  （Top-K の mass が 0.9Z に達しないため）。
- 特に `out=8 / top_k=20` の rejection は 256ms に達しており、
  これが Gate 5 の主な改善対象。

### 3.4 E2E（radix path の選択確認と deterministic repeat）

`phaseshift-compute` に自然文 prompt（"Write a short poem about the ocean."）を
送り、`temperature=1.0 / top_p=0.95 / top_k=20 / seed=42` で 64 token を生成した。

- 同じ seed の 2 回実行は生成列が完全一致（deterministic repeat）
- seed を 43 に変えると生成列が変わる（RNG が実際に効いている）
- 生成結果は 1 token に集中せず、多様な token 列になる

さらに Gate 5 の selector が radix path を選んでいることを別途確認した
（一時計測で全 64 step が `radix`）。詳細は 3.5 を参照。

### 3.5 統合時に見つかった実装欠陥

E2E の最初の実行では radix path が一度も選ばれず、常に rejection path へ
fallback していた。原因は Gate 5 で `Executor` に追加した `stochastic_topk_*` が
`Executor` の move constructor と move assignment のどちらでも移動されておらず、
production の executor では radix workspace が常に空だったこと。

bench は kernel を直接起動するためこの欠落を検出できない。selector の unit test も
`SamplingSelectorInput` に workspace 有無を注入するだけで、executor の実 workspace を
見ない。compute 経路で初めて現れる。

修正は `fix: carry constraint and radix workspaces through executor move`。
`test_executor_move_assignment` を拡張し、move の両経路で
`stochastic_topk_*` が移動されることを検証する。

## 4. 判断

| 項目 | 判断 |
| --- | --- |
| 性能 | **GO** — common top_k（20 / 50 / 64）全範囲で 7.6〜585倍。top_k=50 の病理ケースも大幅改善 |
| 正しさ | **GO** — membership / ties / distribution / determinism / batch independence / invalid logits / backend 一致 が PASS |
| E2E | **GO** — compute 経路で radix path が選択され、同一 seed の生成列が repeat する |

## 5. 未実施

- `top_k` と `top_p` の crossover 検討（現状は全条件で radix が優位）
- constraint 付き stochastic（Gate 5 の scope 外、別 Gate）
- `top_k` が row ごとに異なる batch の radix 化（初期版は all rows same top_k のみ）
