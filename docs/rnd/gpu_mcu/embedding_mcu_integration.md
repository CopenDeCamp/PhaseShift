> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# Embedding MCU Integration Gate

- 目的: Host 側に残っていた Embedding dispatch を GPU-MCU plan へ取り込み、
  `Host → MCU: Embedding → L0 ... L31` の execution region を成立させる。
- 対象: qwen35 family。BF16 embedding の stable ABI / selector / standalone HSACO /
  MCU invocation / plan compile / plan cache fingerprint / production range 拡張。
- 対象外: PSQ8 embedding の MCU 化、LM Head / Sampling の MCU 化、correctness backend の
  削除、prefill 専用最適化、bucket-max grid。
- 性能の GO/NO-GO 判定は行わない。成立可否と制約の記録が目的。
- 現在の仕様: `docs/developer/gpu_mcu/low_level.md`、`docs/developer/kernels.md`

## 1. 基点

| 項目 | 値 |
|---|---|
| base revision | `4ef0a38542d7eb7cc443e011192f89e869fdfd3a` |
| worktree | `.worktrees/gpu-mcu-embedding` |
| branch | `feat/gpu-mcu-embedding` |
| 対象モデル | Qwen3.5-4B（`hidden=2560`）、Qwen3.8-27B-PSQ（`hidden=5120`） |
| 計算単位 | decode（1 batch、`rows=1` の full body parity） |

## 2. Gate 結果

| Gate | 内容 | 結果 |
|---|---|---|
| 0 | token input を graph 上 `I32` に固定（main / MTP lowering、resolver で検証） | PASS |
| 1 | BF16 embedding stable symbol / `EmbeddingBf16Args` / selector rule（2560 / 5120） | PASS |
| 2 | standalone embedding HSACO / raw AQL parity（stable symbol lookup） | PASS |
| 3 | `McuEmbeddingBf16Invocation` / `kMcuKernargRecipeEmbeddingBf16` / FSM 経路 | PASS |
| 4 | plan compiler / registry / `McuDecodeRuntime` upload / fingerprint | PASS |
| 5 | production range を dispatch 0 まで拡張（`region_begin`） | PASS |

commit 列: `41b26287` → `870eb7fd` → `474a8856` → `ee2d6fc5` → `b178cc10` → `36580999`。

## 3. 実装要点

- BF16 embedding の stable entrypoint `phaseshift_qwen35_embedding_bf16(EmbeddingBf16Args)`。
  既存 Host launcher も同じ entrypoint を launch する。
- selector は Host / MCU で `resolve_embedding_physical` を共有し、
  storage / encoding / shape / selector 条件を二重実装しない。
- invocation の `token_ids` は compile 時に resolve した安定 device pointer を既定とし、
  kernarg 生成時に `state->batch_input != 0` なら batch context の token buffer で上書きする。
- `rows` / `grid.x` は actual launch rows。stable kernel に `row >= args.rows` guard を残す。
- executor は `ProgramMcuRange::region_begin` を持ち、先頭 dispatch が MCU 可能な
  BF16 embedding のときだけ MCU region を 1 つ手前へ広げる。`region_begin == 0` のとき
  Host prefix launch を発行しない。
- plan builder は先頭の BF16 embedding node を GDN reset より前に置く。

## 4. 検証結果

- 2560 / 5120 の BF16 embedding が optimized selector 対象になり、
  Host optimized と correctness / raw AQL / FSM が byte exact。
- 5120 は legacy `test_embedding`、2560 は同 test と 4B full body で確認。
- `test_gpu_mcu_full_transformer_body_plan` を `[dispatch 0, body_end)` へ拡張。
  Qwen3.5-4B decode 1 row で embedding を含む 697 logical dispatch（+ GDN reset + marker）
  を MCU 実行し、全 32 layer 出力・KV state・GDN state が Host と byte exact。
- 同 test で MCU region 実行中に Host embedding dispatch が 0 件であることを
  `embedding_optimized_count()` で確認。Host fallback も 0。

## 5. 制約・留意

- PSQ8 embedding は MCU 対象外。BF16 / selector 非対象 shape は Host prefix に残る。
- correctness fallback 側の Embedding 直後 sync は維持する（別 Gate で扱う）。
- 先頭に極短の embedding が入るため、full body plan test の
  `mid_region_queue_empty_count` は厳密 0 に収まらない run がある。これは FSM 制御 loop の
  per-node overhead が 1 row region の GPU 実行時間に近づくためで、correctness には影響しない。
  test は bounded 判定へ変更した。
- HostStream 側で device batch context からの row-global invocation patch を統一するまで、
  `grid = bucket max` / `args.rows = ActualRows` には変えない。
- production executor の MCU E2E（`test_gpu_mcu_production_decode` の `PHASESHIFT_MCU_E2E=1`）は
  既存の persistent body plan blocker の影響で既定では走らない。embedding を含む region の
  exact parity は `test_gpu_mcu_full_transformer_body_plan` で確認する。
  continuous batching の request turnover / rows 1,2,8,16 は今後の Gate で扱う。
