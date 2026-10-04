# TP2 での DFlash2（drafter 非分割・verify のみ TP2）

`tensor_parallel_execution.md` §10 の未対応項目「DFlash2 + TP」について、
**drafter を分割せず、target verify だけを TP2 で回す**設計の調査と実装記録。

## 1. 現状の断絶

DFlash2 と TP はコード上**完全に非接続**（相互参照ゼロ、明示的な拒否も無し）。

| 点 | 状態 |
| --- | --- |
| DFlash2 → TP 参照 | `dflash2/`・`dflash2_spec_decoder.cpp` に `tp_size` / `TpCoordinator` の参照なし |
| TP → DFlash2 参照 | `tensor_parallel.cpp` に `dflash` の参照なし |
| 統合点 | `ContinuousBatcher::set_tp_batch_hook` → `TpCoordinator::on_execute`（**DFlash2 は素通り**） |
| verify の呼び出し | `dflash2_spec_step` が `execute_batch(*decoder.target, ...)` を直接呼ぶ |

## 2. 事前実測（判定材料）

### 2.1 draft と verify の時間配分

`--dflash2-drafts 7 --max-new-tokens 256 --dflash2-stats 1`（GPU1、2反復、差 <0.1%）:

```text
DFLASH2_DRAFT_GPU_MS   = 189.0   →  5.1 ms/round  (37 rounds)
DFLASH2_VERIFY_GPU_MS  = 1525.3  → 41.2 ms/round
DFLASH2_ROUND_MS       = 1759.5  → 47.6 ms/round
```

docs の既知値（84 rounds / 63.15 tok/s）とも整合する:
4.05 s ÷ 84 = **48.2 ms/round**（round 単価は prompt によらず同一）。
つまり **verify が GPU 時間の 89%** で、これが TP2 の対象になる。

drafter 側の host 時間は `DRAFT_MS = 9.99` に対し `DRAFT_GPU_MS = 189`
であり、**draft は verify とパイプラインで重なっている**。

### 2.2 device token bridge のコスト

`PHASESHIFT_DFLASH2_DEVICE_TOKEN_BRIDGE`（既定 1）の A/B（3反復、方向 3/3）:

| 設定 | ROUND_MS（中央値） | `PROPOSAL_WAIT_MS` | `VERIFY_MS` |
| --- | ---: | ---: | ---: |
| 1（既定） | **1760.0** | 0.00 | 1748.4 |
| 0 | **1763.7** | 202.68 | 1547.5 |
| 差 | **+3.7 ms / 37 rounds = 0.21%** | +202.68 | −200.9 |

`PROPOSAL_WAIT` の 202.7 ms は `VERIFY_MS` の同額減と相殺され、
**wall に効くのは 3.7 ms**。TP2 が要求する `TokenIdsLocation::Host` を
強制する代償は約 0.2% で無視できる。

副次的な事実: `need_host_seed = host_proposal_visibility || constraint != nullptr`
（`dflash2_spec_decoder.cpp`）のため、**constraint が有効な場合は bridge があっても
毎 round D2H している**。制約付き生成では TP2 の Host 必須と元々の挙動が一致する。

## 3. 契約調査の結果

| # | 契約 | 判定 | 根拠 |
| --- | --- | --- | --- |
| **A** | **per-rank `ExecuteBatchOptions`** | **必須（最大の変更）** | view は生の device ポインタ + pool geometry。`create_gdn_spec_history(arena, pool, ...)` は既に両方を引数で受け、消費者（`gdn_conv` / `gdn_recurrence`）は local slice を書くため **rank 間同期は不要**。`PHASESHIFT_DFLASH2_GDN_COMPACT_COMMIT` の既定が 1 で history と compact の**両方が必須** |
| **B** | `TokenIdsLocation::Host` | **必須・撤回不可**（コスト 0.21% で解決） | `executor.hip` の `upload_tokens` が `Device` のとき `hipMemcpyDeviceToDevice` → 跨 device の D2D = peer read = 本ホスト（R9700）で破損が実証済み |
| **C** | rank 間の同値性 | **4/4 コード確認** | ① tap は `lower_to_primitives.cpp:944-956` で `tp_combine = 1` の**後**の `layer_output` → 両 rank 同値 ② sampling は `on_execute` が rank0 のみ正準 ③ constraint mask / `verify_numeric_mode` / `ExecutionRole` は `ScheduledBatch` に載る ④ GDN は per-rank local slice で同期不要 |
| **D** | stream identity | **非ブロッカー** | `complete_batch` が `hipStreamSynchronize` を呼ぶため `on_execute` は**完全同期関数**。decoder 側の `:872` などの sync はすべて `on_execute` の外側 |
| **E** | **sequence mirror** | **必須（最小）** | `on_sequence_created` の呼び出し元は `continuous_batcher.cpp:307` の**ただ1箇所**。DFlash2 は `create_paged_sequence_state` を直接呼ぶため mirror が無く `on_execute` が `tp mirror sequence missing` で即 error |
| **F** | guard の撤回 | A/E 解決後 | `tensor_parallel.cpp` の `if (batch.speculative_verify) return unsupported`（`19fbb428` から存在） |

### 既に用意されていたもの

- `TpCoordinatorConfig` に `target_hidden_taps` / `target_hidden_tap_count` /
  `max_scheduled_output_rows` が**既に存在**
- `tensor_parallel.cpp` が tap 設定を rank executor へ配布済み
- TP schedule は `for (const auto& prog : built_set.programs)` で**全 8 row bucket**に構築 →
  verify の `N = block_size + 1` も網羅
- prefill-class batch は TP2 で動作済み（`tp_speed` の `prefill_us` 計測）

### deadlock 評価

`on_execute` = `submit_batch` + `complete_batch`（内部で `hipStreamSynchronize`）。
barrier の condvar は **同一 batch・同一 program を走らせる両 rank worker の間**で
発生するためペアリングは必然。DFlash2 driver 側の sync
（`:754` proposal / `:872` decision / `:1028` `:1061` compact）は**すべて `on_execute`
の外側**で、barrier を保持したまま host wait しない。
→ **構造上 deadlock しない**。

## 4. 実装（3 段階、各段階で単 GPU 挙動を検証）

### Stage 1: GDN artifacts を rank-binding vector に畳む（`e37224c7`）

`DFlash2SpecDecoder` の 14 個の GDN field を
`std::vector<DFlash2GdnRankState> gdn_ranks`（size 1 で従来と同一挙動）へ。
init / shutdown / snapshot / options / restore / commit / 統計を rank loop 化。
diag 比較ブロックは rank0 専用とし `gdn_ranks[0]` 参照に固定（既定 OFF）。

**検証**: DFlash2 ctest 17/17、E2E 指標
（`ROUNDS=37` / `ACCEPTED_DRAFTS=218` / `MEAN_ACCEPTED=5.892` /
`EMITTED_PER_ROUND=6.892` / `VERIFY_GPU_MS≈1525.5`）が refactor 前と一致、
`GENERATED_IDS` の sha1 が `908d2190e5724a29` で**前後一致**、required acceptance 124/124。

### Stage 2: hook 経由の verify 配管（`0b0232e3`）

- `TpBatchHook::on_execute(batch, const ExecuteBatchOptions* per_rank)` を追加し、
  `TpCoordinator` は `Worker::options` 経由で rank ごとの options を配る
- guard を `speculative_verify && per_rank == nullptr` のときだけ拒否へ緩和
- `DFlash2SpecDecoder` に `TpBatchHook* tp_hook` と
  `create_dflash2_spec_decoder(..., const std::vector<DFlash2GdnRankBinding>*, TpBatchHook*)` を追加
- `build_gdn_options()` / `run_target_batch()` を作り、**target 呼び出し4箇所**
  （prefill / `run_single_target` / main verify / rerun）を hook 経由へ
- **`tp_hook != nullptr` のとき `device_token_bridge = false`** → batch が自動的に
  Host になる（契約 B を1行で解決）

**検証**: 単 GPU（`tp_hook == nullptr` → else 分岐）で
DFlash2 ctest 17/17・E2E sha1 `908d2190e5724a29` **完全一致**。

### Stage 3: harness と test

- `tests/support/dflash2_tp_harness.h` — coordinator 構築 → DFlash2 weight/draft →
  sequence 作成 → **`on_sequence_created`（契約 E）** → per-rank bindings →
  `create_dflash2_spec_decoder(..., &bindings, coordinator.get())` → prefill/step ループ
- `tests/unit/test_dflash2_tp_e2e.cpp` + `cmake/tests.cmake` 登録
  （`gpu2;optional;external_files`、GPU_COUNT 2）

## 4.3 正しさ oracle の設計（`tp1 spec == tp2 spec` は採らない）

契約 G 修正後、TP2 は **18 rounds / accepted=29 / 48 tokens まで完走**し、
同一実行で**完全に決定的**だった（2回の実行で token 列が一致）。
しかし **TP1 と TP2 の token 列が index 19 から diverge** した。

```text
tp1: ... 713 304 18 283 2454 304 18 26 ...
tp2: ... 713 304 16 283 2454 304 16 26 ...
        ↑ index19: 18 vs 16（以降オフセット、後に再収束）
```

出力は**完全に coherent**（同じ反復パターン、`19 22 735 220` で再一致）で、
破損ではなく**系統的な差**と判定できる。

### なぜ TP1 と TP2 で DFlash2 の結果が異なり得るか

drafter は **target の hidden taps を入力に取る**（`collect_target_taps`）。
`test_qwen35_tp_e2e` が既に示すとおり、taps は TP1 vs TP2 で
**bit 一致ではなく相対差 `<= 0.10` の許容**がある（`tp_combine` 後の full-width
hidden でも数値差は残る）。

→ **drafter の入力が backend 依存**のため、draft の内容・accept 判定が
backend ごとに変わり得る。これはバグではなく仕様である。

### 採用した oracle

「TP1 の結果と TP2 の結果が一致するか」ではなく、
**各 backend において「speculative decode ≡ target-only greedy」が成り立つか**
を問う（`docs/developer/dflash2.md` の Gate9 と同じ考え方）。

| assert | 意味 |
| --- | --- |
| `tp1_spec == tp1_plain` | 1 GPU で DFlash2 が greedy を保存する（Gate9 相当） |
| **`tp2_spec == tp2_plain`** | **TP2 で DFlash2 が greedy を保存する（本件の本命）** |
| `tp1_plain == tp2_plain` | backend 間で greedy が一致（既存 `test_qwen35_tp_e2e` と同根） |
| `tp1_spec` vs `tp2_spec` | **assert せず prefix 長さのみ print**（taps が backend 依存のため） |

`VerifyNumericMode`（DFlash2 = Exact / plain = Fast）の違いも、Gate9 が
1 GPU で成立させているのと同根の前提に載る。

## 4.6 E2E 結果: DFlash2 は正しく、divergence は既知の TP1/TP2 丸め

4 run 比較（tp1 spec / tp1 greedy / tp2 greedy / tp2 spec、各 48 tokens、
tp2 は `rounds=18 accepted=29`）:

```text
tp1 speculative == tp1 greedy   完全一致（48 tokens）
tp2 speculative == tp2 greedy   完全一致（48 tokens）  ← 本命の E2E 意味は成立
tp1 greedy        != tp2 greedy index19 で diverge（18 vs 16）
```

### 当初の仮説（drafter taps）は棄却

当初は「drafter が backend 依存の target taps を入力にするため」と推測したが、
**plain greedy（drafter 不使用）でも同じ index19 で diverge する**ため、
DFlash2 も taps も原因ではない。原因は **target の TP1 vs TP2 の数値差**だった。

`docs/developer/tensor_parallel_execution.md` §9 が明文でこれを説明している:

> partition された GEMM の出力（bf16）と barrier の SUM（bf16）で丸めが入るため、
> TP=1 と TP=2 は bit-identical ではない。
> greedy token 列は、上位2値が近い token で丸め差により分岐し得る
> （実測: margin の小さい natural prompt では7 token 目で分岐した例がある）。
> `test_qwen35_tp_e2e.cpp` の acceptance prompt は margin が確保できる prompt を選び、
> **greedy16 token の一致を assert する**。

本 test の prompt は同物だが、`max_new_tokens=48` で保証範囲（16 token）を
超えたため index19 で既知の挙動に当たった。

### 結論

- **`tp2_spec == tp2_plain`（48 tokens 完全一致）が本件の E2E 証明**。
  `rounds=18 / accepted=29` は mean 1.6/round で、**partial reject が大半** →
  GDN の capture / compact commit / restore が TP2 で **18回** 走り、
  その全回で target-only greedy と一致している。
- TP1/TP2 の greedy 一致は **documented margin（16 tokens）** に限定して assert し、
  超過分は print のみとする（§9 の契約どおり）。
- 「drafter taps が backend 依存」という推測は **E2E が棄却した**。
  推測ではなく oracle で問うたことで確認できた。

## 4.4 契約 G: mirror の host 側 mutation 追随（実測で発見）

Stage 3 の2回目の実行で、allowlist 修正後に次のエラー:

```text
FAIL tp2 run: dflash2 step: scheduled_batch.cpp:94:
     invalid argument: prefix_tokens != sequence.position
```

### 現象の構造

- `commit_sequence_append` は executor の実行経路で `seq.position += n` を行うため、
  **両 rank は batch 実行ごとに同じ量だけ進む**（ここまでは同期する）。
- しかし DFlash2 は verify の後に **source（rank0 の sequence）だけ**を host で弄る:
  - `decoder.sequence->position = position + accepted + 1u;`（3 箇所）
  - `rollback_sequence_append(*decoder.sequence, required, stream)`（4 箇所）
- mirror（rank1）はこれが知られず、**position / block_table が source と乖離**する。
- `scheduled_batch.cpp:94` が `req.prefix_tokens != req.sequence->position` を
  **エラー**にするため、次の verify で停止する（silent corruption ではなく loud failure）。

`rollback_sequence_append` は GPU 側 `seq_pool_->clear_blocks` と
`kv_pool_->release`（**mirror 自身の PageId**）を伴うため、
mirror の `block_table` を source の中身で単純コピーすることはできない。
**件数だけ揃え、PageId は mirror のものを保持する**のが正しい。

### 修正

`TpCoordinator::on_execute` の**入口**で mirror を source に同期する
（`sync_mirror_state(batch)`）。これにより DFlash2 側の変更は不要で、
所有箇所を1箇所に集約できる。

```text
mirror.block_table.size() > source.size()  → rollback_sequence_append(mirror, source.size(), rank.stream())
                                              （mirror の PageId を release し件数を揃える）
mirror.position != source.position         → mirror.position = source.position
```

ContinuousBatcher 経路では常に同期済みのため実質 no-op（比較のみ）。

### 教訓

DFlash2 は **sequence の所有権を host 側に持つ**（position / block table を自分で管理する）
一方、ContinuousBatcher は executor に委ねる。`TpBatchHook` は後者の想定で設計されており、
前者の runtime を載せるには **state の伝播契約が別途必要**だった。
`on_sequence_created` / `on_sequence_released` に続く3つ目の契約である。

## 4.5 実装中に見つかった shape allowlist（`activation_quantize` と同種）

Stage 3 の最初の実行で、TP1 は成功し TP2 は次のエラーで停止した:

```text
FAIL tp2 run: dflash2 step: gdn_spec_history.hip:209: HIP error
     (commit_gdn_compact_log launch): operation not supported
```

`launch_gdn_recurrence_f32_wmma_commit` の前置条件
`gdn_recurrence_commit_supported()` が **TP1 専用の幾何をハードコード**していた:

```cpp
return args.head_k == gdngfx12::kWk && args.head_v == 128u &&
       args.key_heads == 16u && args.num_v_heads == 48u;   // ← 48 は TP1 のみ
```

- `num_v_heads` は `GdnStatePoolLayout::from_tensor_parallel_config(tc, tp_size)` で
  **TP 調整される**ため、TP2 では 24 → 拒否 → `hipErrorNotSupported`。
- `key_heads == 16u` は **この commit kernel では未使用**（`a.key_heads` を参照しない）。
  同ファイルの他 launcher が使うのは `repeat = num_v_heads / key_heads` で、
  それらは自前の `num_v_heads % key_heads == 0` 検証を持つ。
- kernel 本体は `num_v_heads` を grid / index / vdim のすべてで**一般的に使う**
  （`v_head = (blockIdx.x / vgroups) % a.num_v_heads`）ため、ハードコードは保守的な
  「この幾何しか試していない」ゲートだった。

**対応**: `head_k == kWk && head_v == 128`（WMMA tile 制約、TP 非依存）は残し、
`key_heads == 16u`（未使用）と `num_v_heads == 48u`（TP1 専用）を除去。
正しさは `test_dflash2_tp_e2e` の TP1/TP2 greedy token 一致で担保する。

これは `activation_quantize`（step7/8 の `vec_supported` の k allowlist）と
**同一の失敗パターン**である: shape をハードコードした allowlist が TP2 の local 幾何を
拒否し、結果が「正しく動かない」のではなく「**エラーまたは静かに遅くなる**」。
`docs/rnd/quantization/tp2_resource_regime.md` step9 で分類した第3層の穴に相当する。

## 5. 残る検証（本 step の gate）

- `test_dflash2_tp_e2e`: TP1 と TP2 の **greedy token 列一致**
- 既存 gate: `test_qwen35_tp_e2e`（rank0 vs rank1 の hidden / logits 同値を既に検証）
- required acceptance 124/124

## 5.5 性能: TP2 で DFlash2 が +30.5%

`tests/support/dflash2_tp_harness.h` の報告値
（`dflash2_prefill_plus_decode_ms` / `dflash2_tok_per_s`、model load は含まない）。
同一プロセス内で `tp1_spec → tp1_greedy → tp2_greedy → tp2_spec` の順に実行、
5 反復。prompt は 5 token、`max_new_tokens=48`、`--dflash2-drafts 7` 相当。

| mode | n | tok/s p50 | min | max | spread | rounds | accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp1 | 5 | **55.03** | 54.91 | 55.04 | **0.2%** | 17 | 30 |
| tp2 | 5 | **71.83** | 71.81 | 71.96 | **0.2%** | 18 | 29 |

**TP2 / TP1 = 1.305（+30.5%）**。`rounds` / `accepted` は 5 回とも完全に一致
（決定的）で、run 間変動 0.2% は判定基準（30%）を大きく下回る。

### round 単価への分解

```text
tp1: 874.2 ms / 17 rounds = 51.4 ms/round   (accepted 30 → 1.76/round)
tp2: 668.3 ms / 18 rounds = 37.1 ms/round   (accepted 29 → 1.61/round)
```

- **各 DFlash2 round は 51.4 / 37.1 = 1.385× 速い**
- 全体が 1.305× なのは **round が 1 つ多い**（18 vs 17）ためで、
  `1.385 × 17/18 = 1.309` が実測 1.305 と一致する

round 数の差は index19 以降の token 列分岐に伴う**生成内容の違い**であり、
§9 の bf16 丸めが受け渡した結果である（`accepted` 30 vs 29、5 回とも一貫）。
「TP2 で accept 率が系統的に下がる」とは言えない（n=1 の prompt、方向は一貫だが
prompt 依存の可能性を排除できない）。

### 事前推定との対照

事前の分解（`DFLASH2_VERIFY_GPU_MS = 1525` / `DRAFT_GPU_MS = 189` = **verify が
GPU 時間の 89%**）から、verify が半減すれば理想は 1.80×、
barrier と replicated 部分を織り込んで **1.4〜1.6×** を推定していた。
実測の **round 単価 1.385×** はその下限付近。差の主因は

- verify の GPU 時間は計算が半分になっても **barrier（128 回/round）と replicated op が残る**
- TP2 の wall は step8 で確認したとおり **GPU 時間 + host の barrier 時間が加算**される

ことによる（`docs/rnd/tp_exec2_p2p.md` step7 の cost model と整合）。

### 留意（比較の可否）

- 本測定は**同一プロセス・同一 prompt・同一バイナリ内の TP1 vs TP2** のみ有効。
- `docs/perf/current.md` の DFlash2 63.15 tok/s とは prompt が異なる
  （同 doc の prose K7 / 84 rounds 対し本測定は 5 token / 17 rounds）ため**直接比較不可**。
- run 順は常に tp1 → tp2 のため、tp2 は3回のモデルロード後に走る。
  熱的に不利な側が +30.5% を示しているため**保守的な数値**。

## 6. 未検証・リスク

- **tap の同値は既存 `test_qwen35_tp_e2e` が担保**する構造だが、DFlash2 の
  block verify（`N = block_size + 1`）で同値かは本 test で初確認
- `verify_numeric_mode = Exact` 時の accept/reject 判断が rank 間で一致するかは
  token 列一致で间接確認するのみ
- compact の host 比較 diag（`gdn_compact_compare_remaining`）は rank0 専用のため、
  TP2 では **diag を無効化する前提**（既定 OFF）
- TP2 化の性能効果は本 step では未計測
