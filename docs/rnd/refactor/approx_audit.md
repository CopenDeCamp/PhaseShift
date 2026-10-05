# Approx / heuristic / proxy / coarse / shadow 経路監査

> Status: historical R&D record（server 簡素化・constraint / LocalAI 削除 **前** の状態を記録している。現在の仕様ではない。現在の仕様は docs/developer/ を参照）


PhaseShift に存在する「Exact full computation を省略する経路」の全棚卸し。
本ドキュメントは read-only 棚卸しの記録であり、実装変更は伴わない。

## 判定基準

本監査での用語定義:

- **Exact**: full-vocab PSQ8 GEMM + 全語彙 argmax / sampling(参照実装 `LINEAR_PSQ8` → `SAMPLING` の3連続 dispatch)に対して、返す token / top-1 が数学的に一致することが保証されるもの。
- **確率的近似**: 候補集合を近似スコアで切り、最終 rerank だけ exact でも、候補集合に真の argmax が含まれる保証がなければ **非 exact** と判定する。「final rerank が exact だから経路全体が exact」は判定に用いない。
- **提案側近似**: drafter のみが近似で、target verify が full-vocab exact を実行する経路は「最終結果 exact(verify が担保)」として区別する。

数値はすべて既存 docs(`docs/rnd/`・`docs/perf/current.md`)からの引用であり、新規 benchmark は実施していない。

---

## 1. Target LM head candidate proxy — Fast(非制約 greedy)

- **証拠**: `src/phaseshift/models/qwen35/runtime/lm_head_proxy.hip:397-480`(run_select)、`:758-892`(try_launch_lm_head_proxy_fusion)、`include/phaseshift/models/qwen35/runtime/lm_head_proxy.h:113-126`、呼び出し `src/phaseshift/models/qwen35/runtime/program_executor.hip:479`
1. **Exact path**: `ACTIVATION_QUANTIZE_W4A8 → LINEAR_PSQ8 → SAMPLING` の full-vocab PSQ8 GEMM + 全語彙 argmax。
2. **近似対象**: activation を e4m3 量子化 → INT2 codebook による coarse 全語彙近似 logits → coarse Top-N(既定 pool=32)→ PSQ8 candidate rerank → pool 内 argmax。full vocab logits を計算しない。
3. **最終結果 exact性**: **非 exact(確率的)**。候補集合は INT2 近似スコアで prune され、production に cert/rerank-to-exact の保証がない。greedy equivalence 破りの実測記録あり(`docs/rnd/mtp/mtp.md:756-764`)。
4. **既定**: Decode は既定 off(`target_lm_head_proxy_decode_mode()` が env 未設定なら 0、`lm_head_proxy.hip:723-728`)。Verify 用 `target_lm_head_proxy_mode()` の既定は **1**(`lm_head_proxy.hip:711-721`)だが、`phaseshift-compute` は DFlash2 有効時に env を `0` へ強制(`src/apps/compute/main.hip:1853-1857`)。→ **現行 production 既定では OFF**(env 明示時のみ)。
5. **削除可否**: 可。exact 経路(単体 launcher)は常に存在し到達可能。
6. **削除時性能影響**: 既定 OFF のため現行既定への影響はゼロ。有効時の効果は既存docs値で DFlash2 verify proxy 有効 65.20 tok/s / 無効 63.55 tok/s(`docs/rnd/mtp/mtp.md:820-839`、`docs/perf/current.md` も proxy 停止で約 2.4% 変化と記載)。
7. **複雑度**: `lm_head_proxy.hip` 894行 + `lm_head_proxy.h` 172行。依存 kernel: `draft_head_int2.hip`(1072) / `coarse_topn.hip`(266) / `radix_topn.hip`(442) / `detail/psq8_rerank_device.h`(71)。

## 2. Target LM head proxy — Shadow(mode=2、計測専用)

- **証拠**: `lm_head_proxy.hip:392-395, 695-709, 873-882`、`program_executor.hip:547-558`
1. **Exact path**: 同上。
2. **近似対象**: 近似結果を proxy 自有 buffer に書き、production 出力は full path が書き戻す。差分だけ device counter に集計。
3. **最終結果 exact性**: **exact**(production 出力は proxy 非依存)。
4. **既定**: env のみ(`PHASESHIFT_TARGET_LM_HEAD_PROXY=2`)。
5. **削除可否**: 可。
6. **性能影響**: 既存docs値なし → NEEDS_INVESTIGATION。
7. **複雑度**: 約100行。

## 3. 制約付き LM head — ExactCandidates

- **証拠**: `lm_head_proxy.h:139-170`(select_lm_head_constrained、`max(allowed) <= 128`)、`lm_head_proxy.hip:482-578`、`src/phaseshift/models/qwen35/kernels/optimized/constraint_candidates.hip:11-71`
1. **Exact path**: constrained full PSQ8(mask を潰した全語彙 GEMM + argmax)。
2. **近似対象**: なし。constraint mask から許可 token を全列挙 → PSQ8 candidate rerank → argmax。
3. **最終結果 exact性**: **exact**。候補集合 = 全許可集合(`constraint_candidates.hip:26-51`)。token exact 検証は `docs/rnd/lm_head/constraint_lm_head_hybrid_gate4.md:95-110`。
4. **既定**: env-only / 既定 OFF(`role == Decode && mode != 0` のみ、decode mode 既定 0)。
5. **削除可否**: 可(full path が fallback)。
6. **性能影響**: 既存docs値 — full PSQ8 2362us → 0.26〜0.30ms、**8〜9倍**(`constraint_lm_head_hybrid_gate4.md:97-107`)。既定 OFF。
7. **複雑度**: `lm_head_proxy.hip` 内 約100行 + `constraint_candidates.hip` 73行。

## 4. 制約付き LM head — MaskedCoarse

- **証拠**: `lm_head_proxy.hip:580-693`、選択条件 `lm_head_proxy.h:166-168`
1. **Exact path**: constrained full PSQ8。
2. **近似対象**: INT2 coarse 全語彙 → mask で許可外を `-INFINITY` → Top-N(既定32)で候補 prune → PSQ8 rerank。
3. **最終結果 exact性**: **非 exact(確率的)**。`allowed = 16384 / pool=32` で 2/4 行が mismatch、pool 64/128 なら全件 match(`constraint_lm_head_hybrid_gate4.md:112-124`)。
4. **既定**: env-only / 既定 OFF。
5. **削除可否**: 可。ただし `allowed > 128` の constrained greedy の唯一の高速経路。
6. **性能影響**: 既存docs値 — full 2362us → 1160us(約2倍)。既定 OFF。
7. **複雑度**: 約115行。

## 5. 認証(upper-bound / certified argmax)カーネル群

- **証拠**: `kernels/dflash2/draft_head_int2.hip:437-537, 1044-1071`、宣言 `include/.../draft_head_int2.h:183-278`
1. **Exact path**: full-vocab argmax。
2. **近似対象**: Cauchy-Schwarz 上界で「候補外に勝る token は無い」ことを証明してから採用する機構。
3. **最終結果 exact性**: 設計上 exact(証明失敗時は exact へ fallback)。**本番では不採用**。
4. **既定**: **test-only**。呼び出しは `tests/unit/test_target_lm_proxy_*.hip` と `tests/support/target_lm_proxy_harness.h` のみ。`PHASESHIFT_TARGET_LM_CERT_RATE` / `PHASESHIFT_TARGET_LM_ROWS` も test のみ。
5. **削除可否**: production からは可。ただし required tests が依存(`cmake/tests.cmake:124-128`)→ tests と一体でしか消せない。
6. **性能影響**: 不採用理由が既存docs値(`docs/rnd/mtp/mtp.md:774-801`): 証明は実データで一度も通らず、GPU busy +3.0%、D2H sync で wall 63.55 → 43.99 tok/s(−31%)。削除時の性能影響 = なし。
7. **複雑度**: kernel 約250行 + harness 348行 + tests 約1100行。

## 6. DFlash2 INT2 coarse head(drafter 提案の近似)

- **証拠**: `dflash2/executor.hip:126-136, 515-523, 1995-2015, 2447-2493`、kernel `draft_head_int2.hip:264-310, 808-836`
1. **Exact path**: drafter の full PSQ8 lm_head GEMM + 全語彙 top-K(`executor.hip:2453-2493`)。
2. **近似対象**: INT2 codebook coarse 全語彙 → Top-N(既定32)→ PSQ8 candidate rerank → top-16。full logits を materialize しない。
3. **最終結果 exact性**: 提案側は非 exact。**最終出力は target verify が担保**(提案 token は pool=64 でも 8% が full 提案と異なるが最終 IDS 一致、既存docs値 `docs/rnd/dflash2/int2_head_gate1.md:86-92`)。
4. **既定**: env/config-only、既定 OFF(`PHASESHIFT_DFLASH2_INT2_HEAD` 既定0)。
5. **削除可否**: 可(full PSQ8 経路が本体)。
6. **性能影響**: 既存docs値 — full PSQ8 64.90 → INT2 66.45 tok/s(+2.4%、pool64)。既定 OFF。
7. **複雑度**: 約2700行(`draft_head_int2.*` 1350 + `coarse_topn` 266 + `radix_topn` 442 + `psq8_rerank_device` 71 + `executor.hip` int2 部分 ≈ 600)。

## 7. DFlash2 固定語彙(draft vocab profile)

- **証拠**: `dflash2/executor.hip:138-208, 511-566, 1793-1810`、`draft_vocab_profile.cpp:134-138`
1. **Exact path**: 全語彙(vocab=248320)での提案探索。
2. **近似対象**: 固定 token id 列(縮小語彙)に提案候補を構造的に制限。
3. **最終結果 exact性**: 提案側のみの近似。最終出力は target verify が担保。
4. **既定**: opt-in のみ(`PHASESHIFT_DFLASH2_DRAFT_VOCAB` / `_FILE` 既定 off。自動既定適用は撤回済み `docs/rnd/dflash2/fixed_vocab_default.md:135-137`)。
5. **削除可否**: 可。
6. **性能影響**: 既存docs値 — coverage を保つ入力で約 +4%、入力依存のため既定化は却下。
7. **複雑度**: 約400行 + `tools/quantization/prepare_draft_vocab.py`。

## 8. DFlash2 candidate selector(top-K 上の greedy 選択)

- **証拠**: `kernels/dflash2/candidate_selector.hip:22-87`、`include/.../candidate_selector.h:10-25`、呼び出し `executor.hip:2045-2090, 2489-2493`
1. **Exact path**: なし(DFlash2 モデル自身の定義。「全語彙中的な提案」の参考は `full_vocab_top16`、`executor.hip:2094-2120`)。
2. **近似対象**: 提案候補を top-16 に truncate し、greedy で 1 token/row 確定(決定論的、tie は token id 昇順)。
3. **最終結果 exact性**: selector 自体は定義に対して exact。**top-16 truncate が構造的近似**。最終出力は target verify が exact を担保。
4. **既定**: **DFlash2 の中核(production default で使用)**。
5. **削除可否**: **不可**(DFlash2 提案の本体)。
6. **性能影響**: 該当なし(提案品質 = acceptance に直結)。
7. **複雑度**: 119行 + ヘッダ31行。

## 9. DFlash2 提案側 constraint mask の近似

- **証拠**: `runtime/dflash2_spec_decoder.cpp:732-753`(row0 の mask を全 row に複製)、env `PHASESHIFT_DFLASH2_CONSTRAINT_PROPOSAL` 既定 true
1. **Exact path**: draft prefix ごとに厳密な許可集合を計算する方法(未実装)。
2. **近似対象**: 提案側が適用する許可集合(draft 未確定の現行 grammar 状態を全 row に使用)。
3. **最終結果 exact性**: **exact(target verify の mask が担保)**(`docs/developer/dflash2.md:1146-1148`)。
4. **既定**: 既定 ON(`env_flag_enabled(..., true)`)。
5. **削除可否**: 機能としては削除可(verify が担保)。ただし提案の constraint 適合性(accept 率)は未計測。
6. **性能影響**: 既存docs値なし → NEEDS_INVESTIGATION。
7. **複雑度**: 約20行。

## 10. NgramTail(speculative n-gram tail 提案)

- **証拠**: `runtime/ngram_tail.cpp:8-110`、統合 `spec_decoder.cpp:191-219, 343` / `dflash2_spec_decoder.cpp:805-840`、CLI `main.hip:1803-1814`
1. **Exact path**: なし(drafter 置き換え)。final 判定は target verify。
2. **近似対象**: n-gram 一致ヒューリスティックで draft を延長(提案のみ)。
3. **最終結果 exact性**: exact(verify が担保)。
4. **既定**: opt-in / 既定 OFF(`--dflash2-ngram-tail` 未指定 = 0)。既定 ON は NO-GO 決定(`docs/rnd/spec_decode/ngram_tail_gate3.md:183-188`)。
5. **削除可否**: 可。
6. **性能影響**: 既存docs値 — 8 workload 平均 −1.10%(改善2/悪化6、最悪 −3.93%)。
7. **複雑度**: 約200行。

## 11. MTP draft policy(dynamic / discard margin)

- **証拠**: `include/.../spec_decode.h:129-136`(既定 false/false)、`spec_decode.cpp:368-383`
1. **Exact path**: 固定 K の MTP draft。
2. **近似対象**: top1-top2 margin による draft 数の早期打ち切り/破棄(提案側)。
3. **最終結果 exact性**: exact(verify が担保)。
4. **既定**: 既定 OFF。設定は bench(`src/apps/bench/mtp.hip:1209-1213`)と tests のみ。**`phaseshift-compute` に MTP 経路は存在しない**。
5. **削除可否**: 可(既定で到達不能)。
6. **性能影響**: 既存docs値なし → NEEDS_INVESTIGATION。
7. **複雑度**: 約30行。

## 12. GDN recurrence lossy(既定 ON)

- **証拠**: `kernels/optimized/gdn/recurrence.hip:1190-1199`(`gdn_recurrence_lossy_enabled()` は `PHASESHIFT_GDN_RECURRENCE_EXACT` 未設定なら true)、`kLossy` 分岐 `:152-210, 342-471, 522-539`、dispatch `gdn_recurrence_dispatch.hip:235-261`
1. **Exact path**: `phaseshift_qwen35_gdn_recurrence_wmma_exact` / `gdn_recurrence_f32_impl`(env `PHASESHIFT_GDN_RECURRENCE_EXACT=1` で復帰)。
2. **近似対象**: bf16 hi/lo 分解の4項(hh/hl/lh/ll)を hh のみ1 WMMA に縮減。**「exact」命名の `decode_rows_exact` も `gdn_decomp8<true>` で算術は lossy**。
3. **最終結果 exact性**: **非 exact(数値近似)**。correctness reference 比 max_rel 4.5e-02 / state 8.7e-02(exact は 5.4e-05 / 1.3e-04)(既存docs値 `docs/rnd/gdn/optimization_history.md:351-359`)。
4. **既定**: **production default ON**。
5. **削除可否**: 可(exact 経路・f32 参照が残り env で到達可能)。
6. **性能影響**: 既存docs値 — kernel p50 731.9→526.8us(−28.0%)、e2e PP 1947→2019 tok/s(+3.7%)。
7. **複雑度**: `recurrence.hip` 1353行(lossy/exact はテンプレート分岐で共有)。

## 13. GDN compact commit(既定 ON)

- **証拠**: `dflash2_spec_decoder.cpp:238-243`(既定 true)、commit kernel `recurrence.hip:812-920`、contract `docs/developer/dflash2.md:809-832`
1. **Exact path**: (a) history path(conv+recurrent を row 分保持、`COMPACT_COMMIT=0`)、(b) rerun reference(`PHASESHIFT_DFLASH2_GDN_RERUN_REFERENCE=1`)。
2. **近似対象**: 部分 reject 時の state 復元を {a,k,δ} compact log からの逐次 replay で代替(full state 保持・full rerun を省略)。
3. **最終結果 exact性**: 数式は厳密・逐次 replay は bit-exact と PoC は記載。ただし e2e 統合で parity 失敗 → revert の記録があり(`docs/now_ngram.md`)、現行 developer doc は「生成列一致」と記載 → **NEEDS_INVESTIGATION(再現計測の裏付けは docs にのみ)**。
4. **既定**: **production default ON**。
5. **削除可否**: 可(history path が残る)。
6. **性能影響**: 既存docs値 — メモリ 311.1 MiB(compact) vs 1027.8 MiB(history、+716.8 MiB)、round 時間 compact +2.1%。
7. **複雑度**: 約260行 + history 連携。

## 14. Stochastic radix top-k sampling(active-set path)

- **証拠**: `sampling_dispatch.hip:70-121`、eligibility `sampling_selector.cpp:23-34`、`executor.hip:623-665`、`kernels/dflash2/radix_topn.hip:384-440`
1. **Exact path**: `StochasticSingleBlock`(rejection sampling)/ `Correctness` 参照実装。
2. **近似対象**: 近似ではなく **exact top-K 選択(F32 Radix)→ top-K 内 sampling への実装置換**。INT2 は不使用(`docs/developer/sampling.md:156`)。
3. **最終結果 exact性**: top-K 選択は exact、分布は従来 path と同定義。ただし**同じ seed でも返す token 列は従来 path と一致しない**(RNG 消費順差、`sampling.md:164-168`)。
4. **既定**: 条件成立時 default 有効(`top_k ∈ (0, kSamplingMaxTopK]`、全 row stochastic、workspace あり、constraint なし)。env 不要。
5. **削除可否**: 可(`StochasticSingleBlock` / `Correctness` へ戻る)。
6. **性能影響**: 既存docs値 — common top_k(20/50/64)で 7.6〜585倍の改善、病理ケース `out=8/top_k=20` の 256ms 解消(`docs/rnd/sampling/stochastic_topk_radix_gate5.md:139-173`)。削除で top-k 設定時に大幅な遅化。
7. **複雑度**: `radix_topn.hip` 442行 + sampling 側 約125行。

## 15. Verify numeric mode Fast(bf16 GEMM 還元順)

- **証拠**: `runtime/execution/execution_types.h:18-21`、`linear_selector.cpp:181-199`、`gdn_recurrence_dispatch.hip:236-254`、`spec_decoder.cpp:145-149`(`PHASESHIFT_VERIFY_EXACT`)、compute は常に Exact(`main.hip`)
1. **Exact path**: `VerifyNumericMode::Exact`(bf16 linear を `rows <= 16` で M=1 と bit-exact な `ExactRows` に固定)。
2. **近似対象**: Fast 時は WMMA 系(row ごとに還元順が M=1 と異なり丸めも不一致)。
3. **最終結果 exact性**: 数値差はあるが「target verification が正しければ最終出力を変えない」。proxy Fast と併用時は §1 の非 exact 性が上書きされる。
4. **既定**: DFlash2 = Exact(production)。MTP SpecDecoder の config 既定は Fast。**MTP は production アプリから未到達(bench/tests のみ)**。
5. **削除可否**: 可(Exact のみ残す。compute は既に Exact 固定)。
6. **性能影響**: 既存docs値なし → NEEDS_INVESTIGATION。
7. **複雑度**: 約40行。

## 16. Top-N / Top-K の実装代替(radix vs legacy、reference vs optimized)

- **証拠**: `radix_topn.hip:361-373`(pool>=64 で radix)、`coarse_topn.hip:198-266`、`executor.hip:2102-2114`(`PHASESHIFT_DFLASH2_TOPK_REFERENCE`)
1. **Exact path**: legacy `coarse_topn` / reference `topk_f32`。
2. **近似対象**: なし。**どちらも exact 選択**(CPU oracle と IDs exact、`docs/rnd/dflash2/radix_topn_gate2.md:44,124`)。
3. **最終結果 exact性**: exact。
4. **既定**: radix は pool >= 64 のとき既定採用(proxy/DFlash2 の既定 pool=32 なので legacy のまま)。
5. **削除可否**: 相互代替可(env 切替)。
6. **性能影響**: 既存docs値 — pool64 で radix が legacy を +0.19%。
7. **複雑度**: 708行(radix 442 + coarse 266)。

## 17. Prefix cache(cache であって近似ではない)

- **証拠**: `runtime/prefix_cache.cpp`(507行、trace env のみ)、`docs/developer/prefix_cache.md:32,125`
1. **Exact path**: 再 prefill。
2. **近似対象**: なし。完全一致 prefix の KV / GDN state を D2D で再利用。
3. **最終結果 exact性**: exact(hash だけで hit にせず exact full prefix comparison)。
4. **既定**: compute は `--prefix-cache-capacity-tokens` 0 = disabled、server は既定 16384/8。
5. **削除可否**: 可。
6. **性能影響**: `docs/perf/current.md` に prefix cache 計測なし → NEEDS_INVESTIGATION。
7. **複雑度**: 約560行。

## 18. KLD / shadow model(offline のみ)

- **証拠**: `quantization/offline/kld_shadow_model.cpp`(357行)、利用は `src/apps/quantizer/commands/kld.cpp`
- quantizer の `kld --keep-shadow` 専用。production 推論には無関係。削除可(推論は無影響)。複雑度 357行 + 周辺。

## 19. test-only の env / 調査経路

| env / 経路 | 実装箇所 | 扱い |
| --- | --- | --- |
| `PHASESHIFT_VERIFY_SHADOW_*` | `tests/unit/test_verify_lm_proxy_real_verify.hip` | test-only |
| `PHASESHIFT_RESIDUAL_*` | `tests/unit/test_lm_head_residual_survey.hip` | test-only |
| `PHASESHIFT_REPLAY_*` / `PHASESHIFT_DIVERGENCE_*` | `tests/unit/test_qwen35_mtp_gate3_replay.hip` / `test_qwen35_mtp_verify_divergence.hip` | test-only |
| `PHASESHIFT_TARGET_LM_ROWS` / `_CERT_RATE` | `tests/unit/test_target_lm_proxy_*.hip` | test-only |
| `PHASESHIFT_DFLASH2_GDN_RERUN_REFERENCE` / `TOPK_REFERENCE` / `ATTENTION_REFERENCE` | **production src**(`dflash2_spec_decoder.cpp:241`、`dflash2/executor.hip:116-124`) | reference 切替(既定 off) |
| `PHASESHIFT_QWEN35_KERNEL_MODE=correctness` | `optimized_dispatch.hip:674-683` | 参照 kernel 強制(既定 auto) |
| `PHASESHIFT_PA_PROBE` | `paged_attention_dispatch.hip:30-250` | option 既定 OFF、**call site 0 件(ON でも未使用)** |

いずれも production の数値出力を近似することはない(reference / 計測専用)。

## 20. production に到達していない近似提案(R&D のみ)

`src/` にコードが存在しない(= production 経路ではない)もの:

- FFN k256 sketch / int2 prune / candidate selector / k-tile oracle: `tools/rnd/ffn_k256_*`、`tools/rnd/ffn_k_tile_oracle`(`src/` grep 0 件)
- KV page pruning: `src/apps/bench/paged_prune.cpp`(642行、PoC 明記)+ `docs/rnd/attention/*`。runtime には未統合
- lm_head margin fallback PoC / on-policy greedy shadow PoC: `docs/rnd/lm_head/` のみ(未採用)

---

## サマリー表

| 経路 | Exact対応 | 近似対象 | 最終結果exact性 | production default | OpenCode経路 | ゲート条件 | 削除可否 | 削除時性能影響 | 複雑度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LM head proxy Fast | full PSQ8 + 全語彙 argmax | INT2 coarse→Top32 prune | **非 exact** | OFF | 無効 | env=1 + PSQ8 preshuffled + greedy | 可 | 既定ゼロ(有効化で +2.4〜3.3%、既存docs値) | ~1100行+kernel依存 ~1900行 |
| LM head proxy Shadow | 同上 | 同上(出力は full path) | exact | OFF | 無効 | env=2 のみ | 可 | docs値なし | ~100行 |
| ExactCandidates | constrained full PSQ8 | なし | exact | OFF | 無効 | env=1 + Decode + allowed≤128 | 可 | 既定ゼロ(有効時 8〜9×、既存docs値) | ~190行 |
| MaskedCoarse | constrained full PSQ8 | INT2 Top-32 prune | **非 exact** | OFF | 無効 | env=1 + allowed>128 | 可 | 既定ゼロ(有効時 約2×) | ~115行 |
| 認証カーネル群 | full-vocab argmax | 上界証明による短絡 | 設計上 exact(本番不採用) | **test-only** | 不可 | required tests のみ | production から可(tests と併走) | ゼロ(不採用理由: wall −31%、既存docs値) | ~250行+tests ~1100行 |
| DFlash2 INT2 coarse head | drafter full PSQ8 top-K | INT2 coarse→Top-N→rerank | 提案は非 exact / 最終は verify が担保 | OFF | 無効 | env=1 または固定語彙 | 可 | 既定ゼロ(有効化 +2.4%、既存docs値) | ~2700行 |
| DFlash2 固定語彙 | 全語彙提案探索 | 語彙縮小 | 提案のみ / 最終 exact | OFF | 無効 | env 明示(INT2=1 必須) | 可 | 既定ゼロ(約+4%、既定化却下) | ~400行+tool |
| DFlash2 candidate selector | (DFlash2 定義の中核) | top-16 truncate | 定義上 exact / 最終 verify | **ON(中核)** | **有効** | DFlash2 使用時 | **不可** | — | 150行 |
| 提案側 constraint mask | draft prefix ごとの厳密集合 | 全 row 複製 | exact(verify 担保) | ON | 有効(constraint 使用時) | constraint + DFlash2 | 機能としては可 | docs値なし | ~20行 |
| NgramTail | なし | n-gram 提案延長 | exact(verify 担保) | OFF | 無効 | CLI 明示 | 可 | 既存docs値 平均 −1.10%(既定ON却下) | ~200行 |
| MTP draft policy | 固定 K draft | margin 打ち切り | exact(verify 担保) | OFF | 不可(compute に MTP なし) | bench/tests のみ | 可 | docs値なし | ~30行 |
| **GDN recurrence lossy** | exact WMMA / f32 参照 | 4→1 WMMA 算術縮減 | **非 exact(max_rel 4.5e-02)** | **ON** | **有効** | `GDN_RECURRENCE_EXACT=1` で exact 化 | 可 | exact 化で PP −3.7%(既存docs値) | ~1353行(分岐共有) |
| GDN compact commit | history path / rerun | full state・full rerun の省略 | 数式 exact、parity 記録は要確認 | **ON** | **有効** | `COMPACT_COMMIT=0` / `RERUN_REFERENCE=1` | 可 | メモリ +716.8 MiB・round −2.1%(既存docs値) | ~260行 |
| Stochastic radix top-k | rejection / correctness sampling | 実装置換(top-K は exact) | exact(分布同定義、seed 列は非互換) | 条件成立時 ON | 有効(temp>0 + top_k) | 全 row stochastic + top_k≤128 + constraint なし | 可 | 削除で 7.6〜585倍の遅化リスク(既存docs値) | ~570行 |
| Verify numeric Fast | VerifyNumericMode::Exact | bf16 還元順 | 数値差あり / verify 正しさが担保 | DFlash2=Exact、MTP は Fast(production 未到達) | Exact(compute は Exact 固定) | `PHASESHIFT_VERIFY_EXACT` / bench flag | 可 | docs値なし | ~40行 |
| radix/legacy topn | 互いに exact | なし(実装代替) | exact | radix は pool≥64 のみ | legacy(pool=32) | env 切替 | 相互代替可 | pool64 で +0.19%(既存docs値) | 708行 |
| Prefix cache | 再 prefill | なし | exact | server=ON / compute=OFF | 有効(server) | capacity 設定 | 可 | docs値なし | ~560行 |
| KLD shadow model | 全層 KLD | shadow モデル指標近似 | 指標近似(offline のみ) | offline のみ | 不可 | quantizer `kld` | 可(推論無影響) | なし | 357行+周辺 |

---

## 削除シナリオ別まとめ

1. **「proxy 系(LM head Fast / Shadow / constrained)だけ削除」**: 既定 OFF のため現行既定の性能・正しさへの影響はゼロ。`lm_head_proxy.hip`/`.h`、`constraint_candidates.hip`、証明カーネル群と対応 tests が一体で削除対象になる。**ただし §注記の通り、`executor.hip:1828` の初期化条件は Verify 用 mode(既定1)を見るため、非 spec の PSQ8 モデルでは proxy を使わないのにバッファを割り当てる可能性がある**(NEEDS_INVESTIGATION)。
2. **「提案側近似(DFlash2 INT2 / 固定語彙 / NgramTail)だけ削除」**: いずれも既定 OFF で、削除しても DFlash2 本体(full PSQ8 top-K + candidate selector + verify)は残る。
3. **「既定 ON の近似(GDN lossy / compact commit / radix top-k)を exact 一本に戻す」**: ここにのみ既定性能への実影響がある(数値は既存docs値、上表参照)。正しさ的には exact 経路が残っているため戻すことは可能。
4. **test-only 資産(証明カーネル・survey・replay)**: production から切り離せるが、required/optional tests と一体。

## NEEDS_INVESTIGATION

1. GDN compact commit の現行 parity(`docs/now_ngram.md` の revert 記録と `docs/developer/dflash2.md` の一致記載の矛盾、現行 revision での再計測なし)。
2. VerifyNumericMode::Fast の性能影響(既存docsに数値なし)。
3. Prefix cache の性能影響(`docs/perf/current.md` に計測なし)。
4. 提案側 constraint mask 近似の acceptance 影響(既存docsに数値なし)。
5. `target_lm_head_proxy_mode()` 既定1 と docs「既定 1→0」記述の不整合(env 未設定 + Verify role で Fast proxy が発火し得る経路の有無)。
6. 非 DFlash2 の `phaseshift-compute` 既定での proxy バッファ割当量(`executor.hip:1828` の条件で PSQ8 preshuffled 時に発生し得る)。
7. stochastic radix top-k path が現行 workload で実際に選択される比率(既存docsに数値なし)。
