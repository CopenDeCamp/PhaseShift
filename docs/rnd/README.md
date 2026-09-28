# docs/rnd

このdirectoryは **R&D 記録**（PoC、Gate 検証、採否判断、実験ログ、過去 baseline）を置く。

- ここに置く文書は **現在の仕様でも現在の性能値でもない**。判断の履歴である。
- 現在の contract は `docs/developer/`、現在の性能値は `docs/perf/` を参照する。
- 外部資料・外部 repository・ISA は `docs/references/` を参照する。
- 過去のコードは Git history に任せる。archive directory は作らない。
- 各研究は「目的 → 条件 → 結果 → 判断 → 現在への影響」が読めることを目標とする。
  実装 diff の逐語説明や大量の途中ログは Git history へ任せる。

## 横断的な最適化知見

- [optimization_findings.md](optimization_findings.md) — cross-cutting findings と各 topic への索引。
  測定原則の正本は `docs/perf/methodology.md`。

## topic 別の最適化履歴

- [quantization/kernel_optimization_history.md](quantization/kernel_optimization_history.md) — PSQ / GEMM / preshuffle / scale / codebook / iMatrix / FP8・MXFP4 の検証履歴
- [gdn/optimization_history.md](gdn/optimization_history.md) — GDN recurrence / conv / chunked scan / WMMA 数値挙動
- [attention/optimization_history.md](attention/optimization_history.md) — paged attention decode / prefill、KV split/reduce
- [runtime/execution_overhead.md](runtime/execution_overhead.md) — prefill・decode コスト内訳、kernel gap、HIP Graph / keep-alive
- [runtime/rccl_parallel_gate.md](runtime/rccl_parallel_gate.md) — RCCL multi-GPU Gate（R0 環境、R1/R2、PP stage 分割）の検証記録
- [kernel/optimization_history.md](kernel/optimization_history.md) — RoPE 等の単独 kernel
- [mtp/optimization_history.md](mtp/optimization_history.md) — MTP の correctness / acceptance / Gate 検証
- [dflash2/optimization_history.md](dflash2/optimization_history.md) — DFlash2 / speculative verify / INT2 coarse head

## DFlash2 / speculative decode

- [dflash2/dflash2.md](dflash2/dflash2.md) — DFlash2 drafter の Gate 別検証記録（正本）
- [dflash2/spec_decode_loops_adoption.md](dflash2/spec_decode_loops_adoption.md) — draft loop 設計を PhaseShift へどう適用したか
- [dflash2/external_facts_adoption.md](dflash2/external_facts_adoption.md) — 外部 DFlash2 実装 fact を PhaseShift でどう検証・採用したか
- [mtp/mtp.md](mtp/mtp.md) — MTP（内蔵 drafter）の Gate 記録（正本）

## Qwen4Exp (Qwen3.8-Flash-Next)

- [qwen4exp/gate0.md](qwen4exp/gate0.md) — Gate 0 (architecture contract) / Gate 0.5 (memory feasibility) の記録

## attention / KV

- [attention/kv_page_pruning_poc.md](attention/kv_page_pruning_poc.md)
- [attention/kv_page_pruning_layer_profile.md](attention/kv_page_pruning_layer_profile.md)
- [attention/kv_page_pruning_psq2_proxy_poc.md](attention/kv_page_pruning_psq2_proxy_poc.md)
- [attention/kv_page_pruning_psq2_gpu_selector.md](attention/kv_page_pruning_psq2_gpu_selector.md)
- [attention/kv_page_pruning_psq2_dtype_compatibility.md](attention/kv_page_pruning_psq2_dtype_compatibility.md)
- [attention/paged_attention_decode_linear_degradation_report.md](attention/paged_attention_decode_linear_degradation_report.md)

## quantization

- [quantization/w4a4_RnD.md](quantization/w4a4_RnD.md)
- [quantization/PSQ_W32_BF16_Scale_Features_and_Potential.md](quantization/PSQ_W32_BF16_Scale_Features_and_Potential.md)
- [quantization/QSA_PSQ_Proxy_LongContext_Report.md](quantization/QSA_PSQ_Proxy_LongContext_Report.md)
- [quantization/psq2_k_shape_v0_freeze.md](quantization/psq2_k_shape_v0_freeze.md)
- [quantization/psq4_entropy_poc.md](quantization/psq4_entropy_poc.md)
- [quantization/psq4_kv_hadamard_poc.md](quantization/psq4_kv_hadamard_poc.md)
- [quantization/psq4_next_format_design.md](quantization/psq4_next_format_design.md)
- [quantization/psq8_kv_poc.md](quantization/psq8_kv_poc.md)
- [quantization/psq4_vq4_poc.md](quantization/psq4_vq4_poc.md)
- [quantization/psq4_weight_sharing.md](quantization/psq4_weight_sharing.md)
- [quantization/quaternion_psq_poc.md](quantization/quaternion_psq_poc.md)

## ffn / lm_head

- [ffn/ffn_int2_prune_gate.md](ffn/ffn_int2_prune_gate.md)
- [ffn/ffn_psq4_vs_int2_psq8.md](ffn/ffn_psq4_vs_int2_psq8.md)
- [lm_head/lm_head_int2_coarse_psq8_rerank_poc.md](lm_head/lm_head_int2_coarse_psq8_rerank_poc.md)
- [lm_head/lm_head_margin_fallback_poc.md](lm_head/lm_head_margin_fallback_poc.md)
- [lm_head/lm_head_onpolicy_greedy_shadow_poc.md](lm_head/lm_head_onpolicy_greedy_shadow_poc.md)

## server

- [server/server_history.md](server/server_history.md) — Server の Gate 系譜と Closure / compatibility 判断
- [server/reasoning.md](server/reasoning.md) — reasoning の測定・検証記録

## fusion

- [fusion/gate11j-series-progress.md](fusion/gate11j-series-progress.md) — Fusion / DeepFusion Gate 11J-11K の進捗

## baseline / objective / competitive audit

- [primitive_baseline.md](primitive_baseline.md) — 過去の primitive / E2E baseline 断面
- [objective_vllm_mxfp4.md](objective_vllm_mxfp4.md) — vllm-mxfp4 を目標とした比較・過去判断
- [vllm_mxfp4_gap.md](vllm_mxfp4_gap.md) — vllm-mxfp4 competitive gap audit

## sampling

- [sampling/sampling.md](sampling/sampling.md)
