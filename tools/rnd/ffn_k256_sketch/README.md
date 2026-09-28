# ffn_k256_sketch

Qwen3.8-27B FFN `down_proj` の N64 × K256 sparse down 向けに、exact 64D contribution
`c_g = W_tile,g x_g` を計算せず、低次元 linear sketch `z_g = R c_g ∈ R^d` だけで
cancellation-aware な skip mask を生成できるかを確認する PoC。

固定 deterministic projection（64×64 Walsh-Hadamard、per (layer,tile) hash permutation +
sign、d は nested）を使い、学習はしない。

## 構成

| file | 役割 |
| --- | --- |
| `projection.py` | Hadamard basis / deterministic per-tile R / apply |
| `gateA_diag.py` | exact c を capture し、sketch selector の J_exact を exact selector と比較（情報 Gate） |
| `gateA_ppl.py` | sketch-only selector で 2048-token PPL（float sketch） |
| `test_sketch.py` | projection の直交性・決定性・nested・apply 一致 |
| `analyze.py` | diagnostic / PPL から table と headroom を生成 |

selector は `tools/rnd/ffn_k_tile_oracle/subset_select.py` を再利用。sketch selector には
`z` のみを渡し、64D の `c` は参照しない（`gateA_ppl.py` の `choose` で構造的に分離）。

## 実行

```bash
# 情報 Gate
CUDA_VISIBLE_DEVICES=2,3 python3 tools/rnd/ffn_k256_sketch/gateA_diag.py \
    --layers 0,16,28,32,48,63 --n-tokens 128 --dims 2,4,8,16 \
    --skips 10,14,16,17,20,24,27,28 --seed 0 \
    --out artifacts/ffn_k256_sketch/diagnostic.json

# full PPL（float sketch）
CUDA_VISIBLE_DEVICES=2,3 python3 tools/rnd/ffn_k256_sketch/gateA_ppl.py \
    --configs exact_pair:0,small_l2:0,sk_residual:4,sk_pair4:4,sk_residual:8,sk_pair4:8,sk_pair4:16 \
    --skips 16,24,27,28 --out artifacts/ffn_k256_sketch/ppl_float_sketch.json
```

## Gate A の判定基準（`docs/perf/ffn_k256_sketch.md`）

exact PAIR_GREEDY_SWAP の mask と一致する必要はない。d<=8 の sketch selector が
35〜40% skip で ΔPPL <= +5% を維持できるかで判定する。
