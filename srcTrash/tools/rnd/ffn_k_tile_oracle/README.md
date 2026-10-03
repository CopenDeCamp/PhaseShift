# ffn_k_tile_oracle

Qwen3.8-27B FFN `down_proj` の decode1 を対象に、**token × output tile × K256 group**
ごとに独立な mask を持つ dynamic K-group skipping の **exact Oracle** を測る PoC。

predictor は実装しない。activation-only / activation×weight-norm / INT2 estimator /
sparse production kernel / runtime selector は対象外。

## 構成

| file | 役割 |
| --- | --- |
| `oracle_down.py` | exact group partial / score / mask / apply の pure function |
| `run_oracle.py` | 27B streaming load + MLP patch + PPL sweep + stats の CLI |
| `stream_load.py` | safetensors shard streaming loader（旧 `poc/ffn-int2-prune` harness から再利用） |
| `summarize.py` | 結果 JSON から table を生成 |
| `test_oracle.py` | GPU 不要の smoke test |

`oracle_down.py` が数値 core で、`group_partials` / `group_scores` / `select_keep_mask` /
`apply_mask` を個別に検証できる。

## 定義

```text
c[tile, g]      = W_down[tile, g*256:(g+1)*256] @ h[g*256:(g+1)*256]
score[tile, g]  = sum(c[tile, g]^2)        (squared L2, primary)
keep[tile]      = score 上位 (ng - skip_count) group
y[tile]         = sum_{g in keep[tile]} c[tile, g]
```

* mask は必ず token ごと・output tile ごとに独立。全 output 共通 mask へ退化させない。
* `tile_width = 5120` は全 output 共通 mask 相当の sanity mode（既存 global G256 と比較用）。
* partial accumulation は float32。weight / activation は BF16 source。

## smoke test

```bash
python3 tools/rnd/ffn_k_tile_oracle/test_oracle.py
```

skip=0 の full GEMM 相当、指定 group だけが消えること、token/tile ごとに mask が異なること、
N64 mask が隣接 64 output にのみ作用すること、global mask でないことを確認する。

## 実行

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python3 tools/rnd/ffn_k_tile_oracle/run_oracle.py \
    --model-dir models/Qwen3.8-27B \
    --corpus artifacts/ffn_prune/corpus.psktok \
    --k-group 256 --tile-width 64 --score l2 --max-tokens 2048 \
    --skip-fracs 0,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6 \
    --out artifacts/ffn_k_tile_oracle/oracle_n64.json
```

`--tile-width 16` が N16、`--tile-width 5120` が global sanity。
`--chunk-tokens` は token chunk 長で、GPU memory peak が高い場合に下げる。
OOM 時に CPU へ自動 fallback はしない。

## Cancellation-aware subset Oracle

同一の K256 group 数 k を skip するとき、`||Σ_{g∈S} c_g||²` を最小化する subset を
選ぶ PoC。個々の `||c_g||²` の小さい順に選ぶ SMALL_L2 の改良。

| file | 役割 |
| --- | --- |
| `subset_select.py` | SMALL_L2 / RESIDUAL_GREEDY / PAIR_SEEDED_GREEDY / PAIR_GREEDY_SWAP / exact subset |
| `run_cancel.py` | Stage A diagnostic（pair 統計 + local error 比較）と Stage B full PPL |
| `test_cancel.py` | 合成 cancellation test + 小規模 exhaustive validation |
| `summarize_cancel.py` | diagnostic / pair / PPL の table 生成 |

selector はすべて 1 行 = 1 (token, N64 tile) の `C [R, ng, tile]` に対する torch 演算で、
group ごとの GPU sync をしない。`pair_greedy_swap` は pair seed から greedy 拡張し、
最後に 1-swap local refinement を `max_swap_iters` 回まで行う。

```bash
# Stage A diagnostic
CUDA_VISIBLE_DEVICES=0,1,2 python3 tools/rnd/ffn_k_tile_oracle/run_cancel.py \
    --stage diagnostic --model-dir models/Qwen3.8-27B \
    --corpus artifacts/ffn_prune/corpus.psktok \
    --k-group 256 --tile-width 64 --layers 0,16,28,32,48,63 \
    --skip-groups 7,10,14,17,20,27 --out-dir artifacts/ffn_k_tile_cancel_oracle
# Stage B full PPL
CUDA_VISIBLE_DEVICES=0,1,2 python3 tools/rnd/ffn_k_tile_oracle/run_cancel.py \
    --stage ppl --model-dir models/Qwen3.8-27B \
    --corpus artifacts/ffn_prune/corpus.psktok \
    --k-group 256 --tile-width 64 --methods small_l2,pair_greedy_swap \
    --skip-groups 0,7,10,14,17,20,27 --out-dir artifacts/ffn_k_tile_cancel_oracle
```

