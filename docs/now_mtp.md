# MTP 調査レポート

MTP first-position acceptance の原因調査と、調査中に発見した
`LmHeadCandidateProxy` の不具合に関する記録。

- 対象モデル: `models/Qwen3.8-27B-PSQ`
- corpus: `tests/data/mtp_perf/prose.psktok`
- GPU: GPU 1
- 分岐: `fix/mtp-bench-spec`
- 詳細な計測根拠は [rnd/mtp/mtp.md](rnd/mtp/mtp.md) §9-10 が正本

---

## 1. 結論

1. **MTP の実装は壊れていない。** MTP 層の forward は vLLM semantic reference と
   cos 0.999949 で一致する。state contract（token shift / RoPE position / logical KV）、
   ONE_PLUS、fc concat 順も全て正しい。
2. first-position acceptance は ctx256 で **0.6623**、ctx32 で **0.7647**。
   「現状 0.58」という前提は 31 round の小標本由来であり、実測値として使わないこと。
3. acceptance の低さは **PSQ 化された target を、BF16 checkpoint 用に訓練された MTP が
   予測している**ことによる。MTP 側の修正では改善しない。
4. 調査中に **別不具合**を発見し修正した。`LmHeadCandidateProxy` が
   spec ON/OFF の greedy equivalence（必須要件）を破っていた。
5. MTP linear 8本（fc / attn / ffn）を **PSQ4** にすると draft が
   **−20〜22%** 高速化し、acceptance は標本誤差内で変化しない。詳細は §8。
   MTP の較正 / 訓練は**中止**した。

---

## 2. 変更ファイル

| ファイル | 内容 |
| --- | --- |
| `include/phaseshift/models/qwen35/runtime/lm_head_proxy.h` | 証明用バッファと `select()` の `out_certified` を追加 |
| `src/phaseshift/models/qwen35/runtime/lm_head_proxy.hip` | 証明付き pipeline に組み替え、既定を exact に変更 |
| `include/phaseshift/models/qwen35/runtime/spec_decode.h` | `MtpDraftPolicy::chain_post_norm`（既定 false） |
| `src/phaseshift/models/qwen35/runtime/spec_decode.cpp` | draft chain の `hidden_out` / `hidden_out_normed` 切替 |
| `src/apps/bench/mtp.hip` | `--mtp-hidden` / `--mtp-chain` / `--mtp-probe` / `--mtp-fixture`、accepted histogram・conditional 統計、rank・margin・cos 診断 |
| `docs/developer/qwen35.md` | `lm_head_proxy` の contract を更新 |
| `docs/rnd/mtp/mtp.md` | 調査記録 §9-10 |

既定値は全て従来と同一。`spec_decode` / bench の変更は診断用で、既定経路には影響しない。

---

## 3. 調査結果

### 3.1 target hidden の PRE / POST A/B

target graph は `layer_input → OUTPUT_GATHER → selected_hidden → final RMSNorm → logits`
であり、external output の `selected_hidden` / `layer_input` は final RMSNorm **前**。
MTP の入力にどちらを渡すかを A/B した。

| 条件 | PRE（現行） | POST（final RMSNorm 後） |
| --- | ---: | ---: |
| ctx256 / TG256 | 0.6623 (102/154) | 0.6842 (104/152) |
| ctx32 / TG150 | 0.7647 (65/85) | 0.7442 (64/86) |

差は ±2pt で ctx により**符号が反転**する。**原因ではない。**

### 3.2 (b)〜(i) の切り分け

| 項目 | 結論 | 根拠 |
| --- | --- | --- |
| (b) prompt MTP KV sync | OK | `logical_length == context-1`、初回 draft で連続 |
| (c) token shift `(hidden[p], token[p+1])` | OK | vLLM `set_inputs_first_pass` と一致 |
| (d) absolute RoPE position = p | OK | `seq.position` と abs position が整合 |
| (e) MTP logical KV position | OK | 累積位置欠損の影響も計測で確認なし |
| (f) RMSNorm ONE_PLUS | OK | `pre_fc_norm_hidden` mean −0.157 / `pre_fc_norm_embedding` mean −0.461。plain 形なら負倍率 |
| (g) fc concat 順 | OK | `test_qwen35_mtp_lowering` / `_primitives`（負コントロール付き） |
| (h) MTP logits / lm_head | OK | reference と一致（§3.3） |
| (i) PSQ quantumization | 主因 | 下記 §3.3 |

### 3.3 vLLM semantic reference との数値 A/B

`tools/reference/export_qwen35_mtp_gate1.py` は vLLM Qwen3.5 MTP semantics の reference。
同 exporter の attention は KV 履歴のない T=1 forward なので、`--context 1`
（MTP KV が空）で同一 `(hidden, token, position=0)` を与えて比較した。31 サンプル。

| 項目 | 結果 |
| --- | ---: |
| cos(reference `final_residual`, PhaseShift `hidden_out`) | 0.999949（min 0.999825） |
| cos(reference `final_norm`, PhaseShift `hidden_out_normed`) | 0.999937（min 0.999822） |
| draft == reference top1 | 30/31 |
| P(draft == target) | 11/31 |
| P(reference top1 == target) | 11/31 |

**最初に diverge する tensor は存在しない。** MTP 側の PSQ8 `embed_tokens` / `lm_head` も
target 一致率を変えていない。

参考事実:

- `mtp.*` 15 tensor は `Qwen3.8-27B` と `Qwen3.8-27B-PSQ` で bit-identical（MTP 層は非量子化）
- `tie_word_embeddings: False` で lm_head は独立 tensor
- Gate1 exporter の `target_hidden` は `make_target_hidden()` の合成ランダム値であり、
  pre / post を裁けない

### 3.4 draft chain の A/B

per-position は**同一 run の accepted histogram から**算出した。

| target hidden | chain | P(d1) | P(d2 \| d1) |
| --- | --- | ---: | ---: |
| PRE | `hidden_out`（現行） | 0.6612 | 0.6875 |
| PRE | `hidden_out_normed` | 0.6423 | 0.6835 |
| POST | `hidden_out`（現行） | 0.6557 | 0.6750 |
| POST | `hidden_out_normed` | 0.6557 | 0.6750 |

K=3（POST）: chain pre `0.6729 / 0.6806 / 0.5714`、chain post `0.6667 / 0.6806 / 0.5510`。

**`hidden_out_normed` は劣化するため採用しない。** vLLM と同じく `hidden_out` を維持する。

### 3.5 失敗モード

ctx256 / K=1 / n=154 における miss 時の「target token の MTP logits 内 rank」:

| rank | 2 | 3 | 4 | 5 | 6-10 | >50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 件数 | 28 | 7 | 4 | 4 | 4 | 5 |

83% が rank 2-5。MTP margin は match 3.67 / miss 0.93 であり、意味的破綻ではなく
near-tie の順位反転である。

### 3.6 原因判定

同一条件（prose / ctx32 / K=1）の比較:

| drafter | first-position |
| --- | ---: |
| MTP | 0.7647 |
| DFlash2（このモデル用に訓練済み） | 0.841 |

DFlash2 が同じ PSQ target で 0.841 を出せることから、量子化された target は
予測可能な水準にある。差の 8pt は「BF16 checkpoint 用に訓練された MTP」による。
残る 0.84 から公開値 0.94-0.97 までの差は比較基準
（checkpoint・量子化度・acceptance の定義）の違いによるものと判断する。

---

## 4. 発見した不具合: `LmHeadCandidateProxy`

### 症状

```
$ ./phaseshift-bench mtp --context 1 --steps 4 --tokens-offset 345 --spec --spec-only
reference (tokens[context+1+i]): 279 4220 3065 303
emitted                          : 279 1245 1208 303
greedy equivalence: FAIL
  divergence index 1: got 1245 expected 4220
```

`spec OFF の greedy target decode と spec ON の token stream が完全一致`するという
必須要件を破っていた。

### 原因

`LmHeadCandidateProxy::select()` は

```
INT2 coarse head → top-P 候補 → PSQ8 rerank → argmax
```

という**近似**であり、真の argmax が INT2 coarse の top-P 外だと取りこぼす。
`test_target_lm_proxy_error_bound` が保証しているのは per-logit の
Cauchy-Schwarz 誤差境界であって、**argmax 保存ではない**。
`docs/developer/qwen35.md` にも保証内容の記述がなかった。

### 修正

`test_target_lm_proxy_fallback` と同じ形の厳密証明へ組み替えた。

1. activation を E4M3 に量子化する
2. `coarse + activation_l2 * error_l2` を全語彙の **upper logits** とする
3. upper logits の上位 `pool+1` を取り、先頭 `pool` を候補とする
4. 候補だけ PSQ8 で厳密に rerank する
5. 最良候補の厳密 logit が **残り 1 つ目の upper logits より大きい**場合のみ certified。
   満たさない場合は単体 launcher（exact 経路）へ fallback する

最初に試した `coarse + act_l2 * max_error` という一括上界は締まりが悪く
fallback が常時発生し、逆に遅くなったため、上位 `pool+1` の upper logits を
`omitted_upper` にする方式に変更した。

### 結果

- `--tokens-offset 345` を含む **31/31 offset で greedy equivalence PASS**
- `test_target_lm_proxy_{certificate,fallback,upper_topn,error_bound}` 全て PASS
- K=1 acceptance は 0.662 で変化なし（acceptance には影響していなかった）

### 既定を exact にした理由

ctx256 / steps256 / K=1、verify 1 forward あたりの計測:

| 経路 | ms/forward |
| --- | ---: |
| exact（proxy 無効、既定） | 37.4 |
| proxy（証明なし） | 36.5 |
| proxy（証明あり） | 57.7 |

proxy 自体の利得は **2.4%** にとどまる一方、証明の判定に forward ごとの D2H が必要で
CPU/GPU の重なりを失わせて +20 ms を生む。よって **既定は exact 経路**
（`PHASESHIFT_TARGET_LM_HEAD_PROXY=0`）とした。

証明を device 側で完結させれば proxy を有効化できるが、利得 2.4% に対して
複雑化が大きいため実施しない。

---

## 5. 環境変数

| 変数 | 既定 | 意味 |
| --- | --- | --- |
| `PHASESHIFT_TARGET_LM_HEAD_PROXY` | `0` | `0` で exact 経路。`1` / `2` で proxy を有効化 |
| `PHASESHIFT_TARGET_LM_HEAD_PROXY_POOL` | `32` | 候補数。上限 `kDflash2Int2MaxPool - 1` |
| `PHASESHIFT_TARGET_LM_HEAD_PROXY_CERTIFY` | `1` | `0` で証明と fallback を省略。**greedy equivalence を満たさない** |

---

## 6. 診断ツール（`phaseshift-bench mtp`）

| フラグ | 内容 |
| --- | --- |
| `--mtp-hidden pre\|post` | MTP に渡す target hidden を final RMSNorm 前後で切替 |
| `--mtp-chain pre\|post` | step>=2 の入力 hidden を `hidden_out` / `hidden_out_normed` で切替 |
| `--mtp-probe` | 更新ごとの draft / target / top1 / top2 / margin / target margin / target の rank / cos を行出力 |
| `--mtp-fixture FILE` | position 0 の MTP 入出力をバイナリ dump（reference 比較用） |
| `--verify-mode exact\|fast` | `SpecDecoder` の verify numeric mode |

`--mtp-probe` 有効時に accepted histogram と
`P(A>=k)` / `P(dk | prev)` が報告へ追加される。per-position acceptance は
**同一 run の histogram から**算出すること。K の異なる run を引き算しないこと。

---

## 7. 残課題

- 必須 acceptance (`tests/run_required_acceptance.py`) は **124/124 PASS**
  （0 fail / 0 skip）。MTP 実モデル gate も `weight_real` / `spec_real` /
  `gate3_replay` / `verify_divergence` が PASS。
  `forward_real` は baseline（非量子化 model）でも同じく失敗する既存の環境問題
  （arena 不足 + `artifacts/mtp_gate1` 欠落）であり回帰ではない。
- acceptance を 0.84 以上へ上げる方針は未決定。選択肢:
  - 公開値 0.94-0.97 の測定条件を確認し、目標値の妥当性を先に確定する
  - MTP embedding / lm_head の BF16 化は計測上効果ゼロのため**実装しない**
  - MTP の較正 / 訓練は要件から**中止**
- 診断フラグ `--mtp-fixture` の比較スクリプトは使い捨てであり
  `tools/` には置いていない（再実行する場合は本ファイル §3.3 の手順を参照）。
- `models/Qwen3.8-27B-PSQ-mtpq4`（MTP 量子化版）は既存
  `models/Qwen3.8-27B-PSQ` と別に保持している。旧 model へ置き換えると
  `docs/perf/current.md` の数値が別 model の測定値になるため、
  **置換するなら perf を再計測する**。

---

## 8. MTP quantumization の A/B

MTP（425M params、BF16 で 0.79 GiB）を量子化して draft を速くする検討。
`--preset psq` の profile（`src/phaseshift/quantization/fpx/profile.cpp`）を変更し、
`phaseshift-quantizer quantize --preset psq --backend hip --scope text-only` で
再生成して比較した。詳細は [rnd/mtp/mtp.md](rnd/mtp/mtp.md) §11 が正本。

非 MTP tensor は **1253 本すべてビット一致**であり、差分は MTP のみ。

### 8.1 encoding の決定

| `TensorRole` | 実体 | params | 初期案 | 採用 |
| --- | --- | ---: | --- | --- |
| `MtpOutput` | `mtp.fc.weight` (5120, 10240) | 52M | PSQ8 | **PSQ4** |
| `MtpAttention` | `self_attn.{q,k,v,o}_proj` | 68M | PSQ4 | PSQ4 |
| `MtpFfn` | `mlp.{gate,up,down}_proj` | 267M | PSQ4 | PSQ4 |
| `MtpNorm` | norm 7本 | 1M | PSQ4 | **BF16**（`load_quantized_small` が 1 次元の量子化を拒否） |
| `Output` | `lm_head.weight` | 1271M | PSQ8 | PSQ8（従来どおり） |

### 8.2 Round 1: `mtp.fc` = PSQ8 — 否決

draft が **+30〜34% 遅化**（4.46 → 5.79 / 8.16 → 10.95 ms/update）。

`PHASESHIFT_QWEN35_KERNEL_TRACE=1` の `KERNEL_TRACE_FALLBACK LINEAR_PSQ8`
（base は fallback 0）で原因を特定。`linear_selector.cpp` の `kLinearShapeRules` は
`{ 5120, 10240, kFamilyBf16 | kFamilyPsq4 }` で **PSQ8 が未登録**、
`select_psq8_gemm_config` が `nullopt` を返して
`launch_model_dispatch_correctness`（参照実装）へ退避していた。
rocprofv3 では `model_dispatch_kernel` 89 dispatch / 207 ms（2.33 ms/draft）。

**遅延の原因は PSQ4 化ではなく、1 本だけが参照経路に落ちていたこと。**

### 8.3 Round 2: MTP linear 8本すべて PSQ4 — 採用

fallback 0、`greedy equivalence` 全条件 PASS。

| 条件 | acceptance base / new | draft ms/update base / new |
| --- | --- | --- |
| ctx32 K1 | 0.765 / 0.786 | 4.453 / **3.552**（−20%） |
| ctx32 K2 | 1.141 / 1.141 | 8.252 / **6.442**（−22%） |
| ctx256 K1 | 0.662 / 0.652 | 4.471 / **3.570**（−20%） |
| ctx256 K2 | 1.116 / 1.151 | 8.281 / **6.484**（−22%） |

- acceptance の差は 4 条件すべて**標本誤差内**
  （n = 85〜154 rounds、標準誤差 約 ±4〜5pt に対し −1.0〜+3.5pt）
- draft 速度: **−20〜22%**
- メモリ: MTP linear 387M が 774 MiB → 約 218 MiB（約 556 MiB 削減）

### 8.4 検証

| 対象 | 結果 |
| --- | --- |
| required acceptance | **124/124 PASS** |
| `test_dflash2_quantization_adapter`（新契約 assert 4件追加） | **74/74 PASS** |
| `test_qwen35_mtp_weight_real` | **PASS**（encoding-agnostic） |
| `test_qwen35_mtp_spec_real` | **PASS**（`MTP_GATE3`） |
| `test_qwen35_mtp_gate3_replay` | **PASS**（7/7） |
| `test_qwen35_mtp_verify_divergence` | **PASS** |
| `test_qwen35_mtp_forward_real` | baseline と同一失敗（既存の環境問題、回帰ではない） |

`tools/` / `docs/` の BF16 前提の記述は `docs/developer/mtp.md`（loader 契約）を
書き換えた。

### 8.5 計測条件

`phaseshift-bench mtp --device 1 --arena-gib 24 --draft-k {1,2} --spec --spec-only`、
corpus は `tests/data/mtp_perf/prose.psktok`、`--verify-mode` 既定（Exact）。
