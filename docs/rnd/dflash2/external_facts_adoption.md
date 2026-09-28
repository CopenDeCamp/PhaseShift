> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は docs/developer/、現在の性能は docs/perf/ を参照）

# DFlash2 外部 fact の検証・採用記録

`docs/references/dflash2.md` に記録した公式実装（z-lab/dflash）の external fact を、
PhaseShift がどう検証し採用したかの記録。外部 fact 自体は references 側にある。

- 日付: 2026-09-21
- 確認対象 revision: `07ebd93db9f472af339b644bb70221ad8428328a`
- 外部 reference: `docs/references/dflash2.md`

---

## target tap（Gate 2）

- PhaseNonShift の Gate 2 tap（graph の `L<layer>.output`）は、公式の
  `hidden_states[layer_id + 1]`（residual 込み final output）と一致する。

## conv / RoPE の dtype（Gate 3 / Gate 4）

- 公式の `_grouped_dynamic_convolve` は bf16 tensor 同士の演算で、積と和は
  offset ごとに bf16 へ丸められる。
- PhaseNonShift の kernel は Gate 3 では f32 で積和していたため、公式とは
  ~1e-3 程度の差が出る。Gate 3 の reference は f32 積和で実装し、この差を
  `docs/rnd/dflash2/dflash2.md` に記録した。
- Gate 4 で公式の丸め列（conv / RoPE / SwiGLU）をそのまま再現した。その結果、
  `attention_prepared` は公式とほぼ一致（rel_l2 1.4e-4）し、conv 単体の
  synthetic 比較は bit-exact になった。
- 公式の RoPE は cos/sin が入力 dtype（bf16）になるため bf16 の積和で丸められる
  （Gate 4 で実測）。

## 活性の大きさ（Gate 5.1）

- 内部活性の大きさは **入力が学習分布内かどうか** に強く依存する。
  - 分布外入力（ランダム noise / tap）: layer0 の layer_output absmax は 1.8e6 に達する
    （公式 f32 でも `mlp_down` 2.5e5）。bf16 格納の量子化が層ごとに増幅し、
    5 layer で final hidden の rel_l2 が 5e-2 程度になる。
  - 実 target 入力（prefill の tap + target embed_tokens の noise block）:
    layer0 の layer_output absmax は 4.0 程度で、final hidden の rel_l2 は
    1e-2 以下に収まる。
- reference 比較では入力の dynamic range を必ず記録する。

## rotary inv_freq の dtype（Gate 5.1）

- `from_pretrained` は `rotary_emb.inv_freq` を float32 で保持する
  （`persistent=False` で checkpoint に含まれないため）。診断用に model 全体へ
  `to(torch.bfloat16)` を掛けると inv_freq まで丸まり、RoPE が実装と無相関になる。
- fixture 生成では rotary embedder を config から作り直すこと。

## candidate selector（Gate 6）

- `torch.topk(..., sorted=False)` の並び順は未規定なので、Gate 6 の reference は
  `(logit 降順, token id 昇順)` の tie rule で並べ直し、candidate 集合と
  `select()` の出力が一致することを確認してから使う。
- 公式 `select()` は `compute_logits()` の出力を受け取る。PhaseNonShift は
  target lm_head を production 経路（PSQ8）で再利用し logits を F32 で出すため、
  reference も PSQ8 production 契約で logits を再計算してから比較する。
- target が PSQ8 のとき `program.cpp` は `e4m3_act = quant_weight` により
  **e4m3** の活性量子化（`launch_activation_quantize_e4m3`、行ごとの `peak/448`
  scale）を選ぶ。int8 量子化（`i8_row`）ではない。e4m3 の code をそのまま
  fp8 WMMA へ渡すため、int8 で量子化すると NaN になる。
- PSQ8 weight は e4m3 codebook と 32 要素ごとの BF16 scale で表現される
  （checkpoint では `weight.__phaseshift_codes` と
  `weight.__phaseshift_metadata1`）。

## 当時の未確認

- candidate selector の path 確定手順（Gate 6 で確認する予定だった）。
