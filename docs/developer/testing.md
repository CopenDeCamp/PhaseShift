# Testing

## required acceptance

自己完結: repository + ROCm + 1 GPUのみ。外部ファイル不要。
canonical source は `tests/run_required_acceptance.py` である。runner は
`phaseshift-required-tests` target を build し、`required` label の CTest を実行する。

required の集合は `cmake/tests.cmake` の `required` label と
`phaseshift-required-tests` target が正本である。件数は build 構成
（benchmark build の有無）で変わるため、ここに固定しない。

一覧の取得:

```bash
ctest --test-dir build -L required -N
```

実行:

```bash
python3 tests/run_required_acceptance.py --build-dir build
```

**skip / failure / error はすべて acceptance failure である。** GPU 不足などで
test binary が exit 77 を返した場合も required acceptance では失敗として扱う。

required は CPU contract test（safetensors writer、FPX layout、量子化 payload /
codec、selector、architecture boundary 等）と、gfx1201 限定の GPU kernel /
runtime test、GPU-MCU substrate test、bench `--help` smoke からなる。
arch 非適合の kernel test は非適合 arch では CTest レベルで DISABLED にし、
plain `ctest` が誤った FAIL を出さないようにしている（test binary 側は exit 77 を
返すが、CTest に skip 機構を置かない）。

## GPU-MCU テスト

GPU-MCU テストは `tests/unit/gpu_mcu/` の下に、**何を保証しているか**で分類する。
テストの統合・削除・rename はしない。

| ディレクトリ | 保証する内容 |
|---|---|
| `tests/unit/gpu_mcu/substrate/` | AQL packet / queue / stage-commit、control・output ring、slot、request ingress、completion、KV page allocator、sequence resource、batch binding・planner・dispatch・commit、worker code object、CU partition、kernarg region、LDS template |
| `tests/unit/gpu_mcu/controller/` | persistent MCU、micro FSM、continuous refill / ring wrap / turnover、append while running、autonomous loop、prepared dispatch、dynamic plan binding、external persistent driver、runtime lifecycle |
| `tests/unit/gpu_mcu/qwen35/` | Qwen3.5 単体 primitive の AQL 実行（`real_*`）、kernel registry、plan compiler / cache、attention / GDN / verify の統合、decode backend policy、layer dispatch range |
| `tests/unit/gpu_mcu/acceptance/` | production decode、mixed batch 実行、plan binder、multirow / gdn / linear-attention の chain bridge、one layer および full body plan |

- 共有 fixture（`gpu_mcu_fsm_test_util.h`、`gpu_mcu_*_fixture.h`）は
  `tests/unit/gpu_mcu/` 直下に置く。4 分類すべてから参照されるため、
  いずれかのカテゴリへは入れない。
- 登録の正本は `cmake/gpu_mcu/tests.cmake`。`phaseshift-required-tests` の
  GPU-MCU 分は同ファイルの `PS_GPU_MCU_REQUIRED_TESTS` が正本。
- 一覧:

```bash
ctest --test-dir build -L gpu_mcu -N
```

- 件数は build 構成（optional / external の有無）で変わるため、ここには固定しない。
- `optional;external_files` label の GPU-MCU テストは外部 model が必要である。
  model が無い環境では exit で返るが、required acceptance の対象外である。
  required acceptance では skip を失敗として扱う。

## optional E2E

Qwen3.5-4B full application E2E は `PHASESHIFT_BUILD_OPTIONAL_TESTS=ON` で有効になる。
manifest は `cmake/tests.cmake` が正本であり、件数は固定しない。

- 外部model: `PHASESHIFT_MODEL_DIR_4B`（既定 `models/Qwen3.5-4B`）
- Oracle: `tests/e2e/fixtures/qwen35_4b_oracle.json`
  （Hugging Face Transformers + PyTorch FP32 eager greedy、PhaseShift非依存）
- 契約: exact greedy token ID
  - compute: BF16 KV / FP8 KV 双方で `GENERATED_IDS == Oracle`（3way exact）
    + `--dump-logits`（argmax == GENERATED_IDS / NaN 0 / Inf 0）
  - cli: single-shot / interactive multi-turn のdecoded reply exact
  - bench: pp / tg deterministic + gemm correctness
  - quantizer: quantize / verify / kld / imatrix / ppl（self）

実行:

```bash
ctest --test-dir build -L e2e --output-on-failure
```

`cli:interactive:turn2==oracle` は **KNOWN-LOSSY**（GDN recurrence の bf16 staging による
言い回しの差。文脈は保持。詳細 `docs/rnd/gdn/optimization_history.md` §7.15）。
gate は通すが `[KNOWN-LOSSY]` として理由付きで出力される。

## server regression

server / compute E2E は `tests/server/` の Python suite で検証する。
`phaseshift-server` は `phaseshift-compute` を subprocess として起動するため、
追加の frontend は要らない。model directory は `PHASESHIFT_MODEL_DIR` で指定する。

canonical runner は `tests/server/run_server_regression.py` である。glob ではなく明示 manifest を
持ち、group 単位で subprocess として順次実行する。non-zero return は失敗として扱い、
skip で通さない（自己判定した skip は exit 77 で報告される）。

```bash
PHASESHIFT_MODEL_DIR=models/Qwen3.8-27B-PSQ \
  python3 tests/server/run_server_regression.py --all
python3 tests/server/run_server_regression.py --group tools --group agent
python3 tests/server/run_server_regression.py --list
```

group と test 一覧は `tests/server/run_server_regression.py` の `GROUPS` が正本である。
現在の group は `core` / `tools` / `unsupported` / `concurrency-cancel` / `agent`。

prefix cache の JSONL contract は
`tests/server/test_compute_prefix_contract.py` が固定する。

## 実行層

2 層を混同しない。

```
required acceptance   self-contained C++ / GPU suite（tests/run_required_acceptance.py）
server regression      model + compute E2E（tests/server/run_server_regression.py）
```

`support.py` は共有 helper であり test ではない。manifest へ入れない。
