# DFlash2 constraint / stochastic / prefix cache 対応

## 目的

DFlash2 speculative decoding 経路で拒否されていた次の3つを解く。

- constraint（`grammar` / `structural_tag` = structured output / tool calling）
- stochastic（`temperature > 0`）
- target prefix cache（`--prefix-cache-capacity-tokens` / `prefix_cache_checkpoint_position`）

いずれも実装未着手であって構造的な不可能ではなく、拒否は silent fallback 禁止方針に
沿った fail-closed だった。

## 方式

### constraint

`TokenConstraintState::fill_draft_tree_masks()` を追加し、内部で
`xgrammar::GrammarMatcher::TraverseDraftTree()` を呼ぶ。linear chain として
`draft_tokens = [_, d0, ..., d_{K-1}]` を渡すと、row 0 は matcher の現状態、
row i は `draft[0..i-1]` を accept した状態の許可集合が得られ、行数は
`K + 1` で verify row 数と一致する。matcher は呼び出し後に rollback で元の状態に
戻るため request の matcher を消費しない。

正しさは `spec_greedy_accept()` の仮定と実際の採否の一致で担保する。
correction は row `accepted`、bonus は row `K` の結果であり、
どちらも「それまでの draft が全て accepted」という仮定が実際と一致するため、
commit される token は必ず grammar 準拠になる。draft が grammar 外を提案した場合は
reject が増え、生成結果は変わらない。

- draft が親 row の mask に無い場合、および matcher が terminate した後の row は
  `TraverseDraftTree` が全 0 で埋める。全 0 の row で sampling kernel が
  許可 0 件エラーにならないよう、DFlash verify batch のみ
  `ScheduledRequest::constraint_allow_empty` を立てて -1 を返させる
  （`DeviceSamplingParams.reserved[0]` の bit 1）。無制約経路の fail-closed 契約は変えない。
- row 0 が全 0（dead end）は request error。
- matcher が terminate して返る -1 は生成終了として扱い、`emitted` から除外したうえで
  `pending_token` に EOS を設定する。

提案側は `dflash2_select_draft_tokens()` が `constraint_mask` を受け、
`launch_dflash2_apply_constraint_mask()` で許可外の logit を `-INFINITY` に潰してから
top-k / INT2 coarse topn を実行する。既存の top-k は `-INFINITY` を候補外として扱うため
kernel のシグネチャは変えずに済んだ。proposal の row は現在の matcher 状態
（draft 未確定）の mask を全 row に使う近似であり、row 0 のみ厳密である。

### stochastic

方式は sample-and-compare。`spec_greedy_accept()` は target の sample 結果と draft を
比較するだけなので、関数も `SpecVerifyResult` も変更していない。
`DFlash2SpecDecoder` に `SamplingConfig` と `sample_index` を持ち、
`make_verify_batch()` / `make_prefill_batch()` へ渡した。
round ごとに行消費 row 数（`total_k + 1` / prefill は 1）だけ `sample_index` を進め、
RNG key の再利用を避けている。これにより conditioning `s_0 = d_0` が
`s_1` の分布を汚さず、出力分布は target の分布と一致する。

### prefix cache

`PrefixCheckpoint` に `dflash` / `dflash_slot` / `dflash_length` /
`dflash_next_position` を追加し、`PrefixCache::enable_dflash_ring()` で
`max_entries` 個の `DFlash2ContextState` を ring pool として作る。
save / restore は `k_ring` / `v_ring` の D2D copy と metadata 3 行だけで、
新規 kernel は作っていない。ring の無い checkpoint に対して DFlash session の
restore を要求した場合は error にする。

checkpoint は prompt boundary（prompt 全体）で保存する。
`dflash2_spec_prefill()` は `restored_tokens` を受け、復元済み位置から
残りの prefill だけを実行する。context は `next_position == restored_tokens` を
事前検証する。

## 条件

- GPU: AMD Radeon AI PRO R9700 (gfx1201) / ROCm / HIP / Linux
- model: `models/Qwen3.8-27B-PSQ` + `models/Qwen3.8-27B-DFlash2-PSQ`
- KV dtype: bf16 / psq4（acceptance）、計測は bf16
- device 1、`--max-seq-len 512`、`--arena-gib 31`
- grammar: `root ::= "A" * 300`、prompt は敵対長文、`max_new_tokens 300`
- `PHASESHIFT_DFLASH2_SERVE_STATS=1` で session 終了時に `DFLASH2_*` を stderr へ出す
- A/B: `PHASESHIFT_DFLASH2_CONSTRAINT_PROPOSAL=0` で提案側 mask を無効化

## 結果

### proposal mask の効果（bf16、同一 grammar / prompt）

| 構成 | mean_accepted | full_accept_rate | rounds |
| --- | ---: | ---: | ---: |
| proposal mask on | 5.375 | 0.688 | 16 |
| proposal mask off | 4.611 | 0.611 | 18 |

同じ 300 文字（102 token）を 16 round で終え、mean_accepted は +16.6%、
full accept rate は +12.6% である。draft が grammar 外を提案していた場合の
低下を、提案側 mask が解消している。

### 受入

- `tests/server/test_compute_dflash2.py`: 54 項目すべて PASS
  （grammar 準拠、structural_tag、constraint 下での target-only 一致、
  stochastic、stochastic + constraint、prefix cache の save/restore、A/B 計測）
- `tests/server/test_server_dflash2.py`: 17 項目すべて PASS
  （tool calling、structured output、responses structured、stochastic、
  prefix cache 起動表示）
- required acceptance: 132 項目すべて PASS（`test_dflash2_constraint_mask` 含む）

## 判断

- proposal mask を採用する。verify 側 mask が正しさを担保し、提案側 mask が
  受け入れ率を上げるという役割分担を維持する。
- proposal の row ごとの mask は「draft 未確定の現在状態」の近似に留める。
  draft prefix 依存の厳密な値を得るには 2 pass の再選択が必要になるが、
  上記の改善率ではその複雑化は正当化されない。必要になったら再度計測する。
- `--dump-logits` と draft vocab profile + constraint は非対応のままとする。

## 現在への影響

- contract は `docs/developer/dflash2.md`（Constraint 節 / Scope / Known limitations）、
  `docs/developer/prefix_cache.md`（DFlash2 有効時）、
  `docs/developer/runtime.md`（DFlash serve 経路）、
  `docs/developer/sampling.md` に反映済み（structured generation は削除済み）。
- user docs は `docs/user/server.md` / `docs/user/compute.md` / `docs/user/cli.md`。
- 計測値の正本は本節であり、`docs/perf/current.md` は未計測のため触っていない。
