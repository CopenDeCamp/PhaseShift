# ffn_k256_candidate_selector

Qwen3.8-27B FFN `down_proj`（N64 × K256、68 group）で、
cheap scalar prefilter により skip 候補 pool を 32〜48 group へ絞り、
その候補だけで cancellation-aware selector を実行する方式を検証する。

前 Gate `ffn_k256_sketch`（全 68 group 一律 low-D sketch）が NO-GO だったため、
「候補を絞ってから濃い direction 情報を使う」route の成立性を、
段階を混ぜずに確認する。

## 段階

| Stage | 内容 | 停止条件 |
| --- | --- | --- |
| A | cheap prefilter で pool を作り、pool 内 exact 64D pair で headroom を測る | NO-GO なら以降を実装しない |
| B | pool 内を d16/24/32 の sketch へ圧縮 | NO-GO なら INT2/GPU へ進まない |
| C | projected weight `B = R W` の生成、INT2 量子化 | traffic Gate で判定 |
| D | GPU prefilter / sketch / select / fused benchmark | performance Gate で判定 |

## prefilter

| 名前 | score |
| --- | --- |
| `exact_l2` | `q_g = ||c_g||²`（production 候補ではない上限） |
| `act_energy` | `q_g = Σ x_j²`（68 scalar、全 tile 共通） |
| `act_wnorm` | `q_g = a_g × ||W[tile,g]||_F²`（weight norm は offline metadata） |

pool 外 group は強制 KEEP。

## 構成

| file | 役割 |
| --- | --- |
| `prefilter.py` | score / pool / restricted selector |
| `gateA_pool_diag.py` | pool 情報 Gate（J、recall、rank、overlap） |
| `gateA_pool_ppl.py` | pool 制限 exact pair での 2048-token PPL |
| `analyze_pool.py` | diag / PPL の table と headroom |
| `test_pool.py` | pool size、C>=k、pool 外 KEEP、C68 一致、recall、決定性、mask、tile 独立性 |

## 実行

```bash
CUDA_VISIBLE_DEVICES=2,3 python3 tools/rnd/ffn_k256_candidate_selector/gateA_pool_diag.py \
    --layers 0,16,28,32,48,63 --n-tokens 128 --pools 32,40,48,56,68 \
    --skips 16,20,24,27,28 --out artifacts/ffn_k256_candidate_selector/pool_diag.json

CUDA_VISIBLE_DEVICES=2,3 python3 tools/rnd/ffn_k256_candidate_selector/gateA_pool_ppl.py \
    --configs unrestricted:68,exact_l2:32,exact_l2:40,exact_l2:48,act_wnorm:32,act_wnorm:40,act_wnorm:48,act_energy:40,act_energy:48 \
    --skips 16,24,27,28 --out artifacts/ffn_k256_candidate_selector/pool_ppl.json

python3 tools/rnd/ffn_k256_candidate_selector/analyze_pool.py
```
