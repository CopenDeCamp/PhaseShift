> Status: 作業中の進捗レポート。現在の contract は `docs/developer/parallel_runtime.md`、
> 現在の性能値の正本は `docs/perf/current.md`、Gate の経緯は
> `docs/rnd/runtime/rccl_parallel_gate.md` を参照。

# RCCL parallel runtime 本番 model 計測の現在地

Gate P0（§51）の本番 model 計測の進行中状態を記録する。
対象 model・条件は §51 の同一条件（model / prompt / output length / sampling /
KV dtype / quant format）で比較する。

## 計測条件

| 項目 | 値 |
|---|---|
| model | `models/Qwen3.8-27B-PSQ`（18 GB、safetensors 5 分割） |
| geometry | `hidden_size=5120`、`num_hidden_layers=64`、`num_attention_heads=24`、`num_key_value_heads=4`、`intermediate_size=17408`、`vocab_size=248320`、`tie_word_embeddings=false`、`full_attention_interval=4` |
| prefill / decode | 64 tokens / 64 steps（固定） |
| sampling | greedy（`sample=true`） |
| arena | 22 GB |
| GPU | GPU2, GPU3（2 枚） |
| harness | `tests/unit/test_qwen35_parallel_bench.hip` |

`attention_head_dim` は config に存在しないが、実行時の geometry 解決には影響しない。

## 計測手段

`test_qwen35_parallel_bench` は env で本番 model を指定できる。

| env | 意味 |
|---|---|
| `PHASESHIFT_BENCH_MODEL_DIR` | 本番 model の path。未指定なら synthetic model を生成 |
| `PHASESHIFT_BENCH_CONFIGS` | 実行する構成（例 `1gpu,pp2,tp2`）。既定は全 4 構成 |
| `PHASESHIFT_BENCH_PREFILL` / `PHASESHIFT_BENCH_DECODE` | prefill 長 / decode steps |
| `PHASESHIFT_BENCH_ARENA_GB` | arena 容量（GB） |

partition 層数・expected collectives は synthetic の spec ではなく
`read_qwen35_text_config` から計算する。出力は prefill/decode tok/s、
p50/p95、per-rank 通信 bytes、bytes/token、VRAM、生成 token 列を含む。

`max_scheduled_tokens` と `max_seq_len` は runner 内で
`prefill_tokens` / `decode_steps` から導出する。

## 計測結果（取得済み）

| 構成 | prefill | decode | decode p50 | p95 | bytes/token | VRAM |
|---|---|---|---|---|---|---|
| 1GPU（auto） | 1083.3 tok/s | 27.1 tok/s | 36.1 ms | 37.0 ms | 0 | 22.21 GB |
| 1GPU（`PHASESHIFT_QWEN35_KERNEL_MODE=correctness`） | 4.4 tok/s | 0.7 tok/s | 1347 ms | 1379 ms | 0 | 22.21 GB |
| TP2（auto、allowlist 追加前） | 9.3 tok/s | 1.5 tok/s | 649 ms | 679 ms | 1,310,720 B | 22.86 GB |
| PP2 | 未取得（ハング） | 未取得 | — | — | 未取得 | — |
| PP2TP2 | 未取得（GPU 2 枚では不可） | — | — | — | — | — |

- TP2 の bytes/token は decode 64 steps のとき 167,772,160 B、8 steps のとき
  94,371,840 B で、いずれも 1,310,720 B/token に一致する。
- 1GPU の auto と correctness の差は prefill で約 246 倍、decode で約 39 倍。
  optimized path と correctness path の差であり、model そのものの性能差ではない。
- 合成 model（4 層、hidden 128）の参照値は既に Gate P0 で取得済みで、
  bytes は PP2 = 5120 B、TP2 = 40,960 B、PP2TP2 = 25,600 B（いずれも理論値と一致）。

## 計測中に判明した問題

### 1. lm_head proxy の `max_rows` 制約

`LmHeadCandidateProxy::init` は `max_rows > 64` で `invalid_argument` を返す。
`max_rows` は `max_scheduled_tokens` に由来し、これが
`max(prefill_tokens, decode_steps)` になるため、prefill 256 では初期化に失敗する。
prefill を 64 に落として制約内に収め、proxy は有効（mode 1）のまま計測している。

### 2. `gpu-test-state` の taint

`gpu-test-runner` の timeout による SIGKILL 後に
`build/gpu-test-state/{lock,taint,usage}` が残ると、次回実行で
test が起動すらせず無出力のまま timeout になる。
計測前にこれらのファイルを削除する必要がある。

### 3. TP2 が correctness kernel に fallback していた

TP2 の local shape が `kLinearShapeRules` allowlist に無く、
`LINEAR_SELECT_MISS` が発生していた。miss した shape は以下のとおり。

```text
family=0 (Bf16) : {3072,5120} {512,5120} {5120,3072} {5120,5120} {5120,8704} {8704,5120}
family=1 (Psq4) : {3072,5120} {5120,3072} {5120,5120} {5120,8704} {8704,5120}
family=2 (Psq8) : {24,5120}
```

1GPU を correctness で測定すると 4.4 / 0.7 tok/s となり TP2 の 9.3 / 1.5 と
ほぼ同値であることから、TP2 の遅さの原因は allowlist 欠落と確定した。
上記を `linear_selector.cpp` に追加済み。
**allowlist 追加後の TP2 の再計測は未実施**（3 構成一括実行が PP2 でハングしたため）。

## 未解決: PP2 のハング

本番 model で PP2 を実行すると、rank1（stage1）が
`create_model_executor` → `build_program_set` → `build_program` の
bucket=0 の node ループで停止する。

特定できた事実:

1. rank0（stage0、`nodes=698`）は全 bucket を通過し PREFILL を開始する。
   相手の RCCL を待って正常停止している。
2. rank1（stage1、`nodes=701`）は bucket=0 の `nodes=701` で停止する。
3. 停止は `ni=5`（`L32.gdn_a`）の `emit_node` 完了後、次の iteration の冒頭。
   `emit_begin` / `emit_done` は 6 反復とも一致し、error return でもない。
4. GPU を反転（`PHASESHIFT_TEST_GPUS=3,2`）しても dev=1（rank1）で発生する。
   GPU 固有ではなく partition 固有。
5. 1GPU と TP2（ともに full partition）では再現しない。
6. `owns_lm_head` の無効化（`PHASESHIFT_TARGET_LM_HEAD_PROXY=0`）では解消しない。

つまり **`owns_embedding=false` + `owns_lm_head=true` の stage1 graph に対して
`build_program` がハングしている**。node ループと `emit_node` の各 case は
いずれも有限に見えるため、メモリ破損（UB）が有力な仮説
（`note_use` の `lifetimes[v.id]` 越界、`graph.nodes` の破損など）。
rank0 と rank1 の graph が 3 node 不同である事実とも整合する。

## 一時的な debug 出力

原因調査用の env-gated print が 3 ファイルに入っている。
原因特定後に revert する。

| env | 出力 |
|---|---|
| `PHASESHIFT_RUNNER_DEBUG=1` | runner の段階 mark（`arena` / `model_load_*` / `pools_begin` / `executor_begin`）と経過 ms |
| `PHASESHIFT_EXECUTOR_DEBUG=1` | `EXECDBG`（executor の段階）、`LOWERDBG`（lowering の段階）、`BUILDDBG`（bucket の開始/完了）、`NODEDBG`（node ループ。`nodes >= 700` の graph のみ） |
| `PHASESHIFT_LINEAR_DEBUG=1` | `LINEAR_SELECT_MISS`（既存） |

一時コードの所在:

- `src/phaseshift/models/qwen35/model/lower_to_primitives.cpp`
- `src/phaseshift/models/qwen35/runtime/executor.hip`
- `src/phaseshift/runtime/program/program.cpp`
- `tests/support/qwen35_synthetic_runner.h`（mark 部分）

## 次の選択肢

| # | 手 | コスト |
|---|---|---|
| A | `lifetimes` の境界チェック（`v.id >= size` を検出して node 名を print）で破損する ValueId を特定 | 小 |
| B | ASAN ビルドで実行して破損箇所を直接特定 | 中 |
| C | PP2 の計測を保留し、1GPU / TP2 の計測値を `docs/perf/current.md` に先に記録 | 小 |
| D | PP2TP2 は GPU 4 枚が使える時点で追加計測 | — |

## 計測コマンド

```bash
PHASESHIFT_BENCH_MODEL_DIR=models/Qwen3.8-27B-PSQ \
PHASESHIFT_BENCH_CONFIGS="1gpu,pp2,tp2" \
PHASESHIFT_BENCH_PREFILL=64 PHASESHIFT_BENCH_DECODE=64 \
PHASESHIFT_TEST_GPUS=2,3 \
./build/tests/phaseshift-gpu-test-runner \
  --gpu-count 2 --cost-gb 24 --timeout 840 --state-dir build/gpu-test-state \
  -- ./build/tests/test_qwen35_parallel_bench
```

計測前に `rm -f build/gpu-test-state/{lock,taint,usage}` を実行する。
GPU は必ず `PHASESHIFT_TEST_GPUS` で明示する。
