> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は docs/developer/、現在の性能は docs/perf/ を参照）

# vLLM / SGLang の draft loop 設計の採用記録

`docs/references/spec_decode_loops.md` の外部実装調査（vLLM / SGLang の draft loop）を
受けて、PhaseShift が何を採用したか、実装への含意を記録する。
外部 fact 自体は `docs/references/spec_decode_loops.md` にある。

- 日付: 2026-09-21
- 外部調査: `docs/references/spec_decode_loops.md`

---

## PhaseNonShift への含意

現状の `mtp_executor.hip:404-597`（`run_mtp_rows_on_state`）は 1 draft step あたり
`hipStreamSynchronize` 1 + D2H 3 + H2D 5 を行う。原因は次の 3 つ。

1. draft token（次 step 入力）を Host 経由で渡している。
2. top1/top2 logit（dynamic / discard の margin）を Host で見ている。
3. execution_status を毎 step Host で確認している。

参照実装に合わせる最小の変更:

- sampling kernel の出力先を次 step の embed 入力（`batch_storage.token_ids`）へ向け、
  draft 中は token を Host に戻さない。
- K step 分の descriptor / rope / position を事前に 1 回の H2D で用意する
  （row=1 なら決定的な漸増）。
- `hipStreamSynchronize` と status / sampled / top2 の読みを draft ループの後ろ 1 回へ。
- dynamic 幅は iteration 間（verify 後）の統計で決める。
- `MtpExecutor` にも target と同じ HIP graph / event 非同期を導入する
  （`executor.hip:668-753` に既存実装がある）。

これは `docs/rnd/mtp/mtp.md` §3.4 の「per-step 往復の除去」に対応する。
§3.1 の partial-accept rerun 除去とは独立の改善。

### DFlash2 での実施（Gate 11I-2）

DFlash2 speculative decode では上記の「draft を Host に戻さない」を実装済みである。
DFlash proposal を `proposal_tokens` に残したまま device 上で verify 入力
（`[pending, draft_0, .., draft_{K-1}]`）を組み立て、target verify を
`TokenIdsLocation::Device` で実行する。round の Host boundary は verify 後の
acceptance 用 1 回だけである。

MTP の draft ループ（自己回帰 K step）は対象が異なり、上記の最小変更は未実施のまま。
詳細は `docs/developer/dflash2.md` の「Device-resident path」と
`docs/rnd/dflash2/dflash2.md` の Gate 11I-2 を参照。

---

## 当時の参照

- 目標リファレンス: `docs/rnd/objective_vllm_mxfp4.md` §9
- 外部調査: `docs/references/spec_decode_loops.md`
