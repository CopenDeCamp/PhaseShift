# Next Task: TP=2 Host Dispatch の改善候補

2026-10-10 時点。`docs/now_task.md` の後継。

> lifecycle が定義された `docs/user/` `docs/developer/` `docs/perf/` `docs/rnd/`
> `docs/references/` には属さない、作業中のメモである。
> 結論と測定の正本は [rnd/tp2_host_dispatch.md](rnd/tp2_host_dispatch.md)、
> 性能値は [perf/current.md](perf/current.md) を参照すること。

## 状態: 改善候補1・2 はいずれも不採用

Attribution（Step 1）で効果の大きい順に挙げた 2 件を実測したが、**両方 e2e 効果ゼロ**。

| 候補 | 効果見積もり | 実測（e2e） | 判定 |
| --- | ---: | ---: | --- |
| 1: `resolve_psq4_physical` のバリデーション移動 | rank 内 46〜49% | TP1 +0.05% / TP2 +0.27% | 不採用 |
| 2: `Program::find_value` の O(1) 化 | rank 内 17〜20% | TP1 −0.06% / TP2 −0.09% | 不採用 |

**見積もりの前提が誤っていた。** 46〜49% / 17〜20% は `perf` の worker 総 cycles 内の
占有率であって、e2e の改善余地ではない。host 時間は GPU busy と重なって潰れる。
詳細と ceiling 実験の結果は [rnd/tp2_host_dispatch.md](rnd/tp2_host_dispatch.md) §3・§4。

判定閾値は TP2 の run 間 spread **2.55%**（= 371 ms に対して 9.5 ms）。それ未満の
改善は測定で検出できない。

- 候補2 の実装差分は `/tmp/opencode/candidate2_value_index.patch` に退避済みで、
  ツリーには入れていない。
- 候補1 の ceiling 実験（`#if 0` で validation 全除去）は測定後に revert 済み。

## 残る候補

kernel launch 数の削減（換算 57.5 ms）が唯一の実質候補。ただし HIP graph は TP schedule
と併用不可で kernel 開発が要る。host 側では潰せない。
`now_task.md` §4 の `ScopedDevice` arch 検証は換算 2.4 ms で判定閾値未満のため打ち切り。

一覧とリスクは [rnd/tp2_host_dispatch.md](rnd/tp2_host_dispatch.md) §5。

## 未完了タスク

- [ ] kernel launch 数の削減の要否判断（リスク高。benchmark が必須）
- [ ] `docs/now_task.md` §1 の残課題5件（build target の食い違い、`required` label の穴3件、
      acceptance の除外オプション無し、`testing.md` の記述乖離、GPU-MCU の required 復帰）
