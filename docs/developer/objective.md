# Objective

PhaseShift が最適化する対象は 1 つだけである。

> AMD RDNA4 (gfx1201) 上で、単一カードの推論性能を最大化する。

汎用性・複数プラットフォーム対応は目標ではない。性能が目標である。

## Current performance target

外部の参照実装 `GGZ14/vllm-mxfp4`（MXFP4 / DFlash2 drafter）の
**One card (TP=1)** プロファイルの水準を、PhaseShift の単一カード構成で
超えることを目標とする。

参照側の decode 数値（combined 137.7 t/s 等）は speculative decode 込みである。
PhaseShift は同一条件で比較できないため、次の比較可能な指標に分解して設定する。

| 指標 | 意味 |
| --- | --- |
| Prefill t/s | 量子化・speculation に非依存で直接比較できる |
| non-spec decode t/s（ctx 別） | target の verify cost に対応する |
| speculative decode t/s | DFlash2 drafter 込みのスループット |
| 並行 aggregate t/s | scheduler / batching 能力 |
| キャパシティ | 単一カード上の KV token 数・context 長 |

数値の比較可能性（sampling 条件、context 長、power cap、量子化 bit 幅の差）は
参照側 audit に整理してある。詳細と経緯は
`docs/rnd/objective_vllm_mxfp4.md` / `docs/rnd/vllm_mxfp4_gap.md` を参照。

## Current target model

コード上の runtime family は `src/phaseshift/models/qwen35/`（Qwen3.5 系）である。
現在の主要 performance target は **Qwen3.8-27B-PSQ**、drafter は **DFlash2** である。

## Current measured baseline

現在再現可能な performance baseline は `docs/perf/current.md` にある。
測定方法は `docs/perf/methodology.md` を参照。
