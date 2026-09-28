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
runtime test、bench `--help` smoke からなる。
arch 非適合の kernel test は非適合 arch では CTest レベルで DISABLED にし、
plain `ctest` が誤った FAIL を出さないようにしている（test binary 側は exit 77 を
返すが、CTest に skip 機構を置かない）。

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

server / LocalAI / backend / compute E2E は `tests/server/` の Python suite で検証する。
LocalAI runtime は `PHASESHIFT_LOCALAI_BINARY`（`vendor/localai/patches/` 適用済み）を指す。

canonical runner は `tests/server/run_server_regression.py` である。glob ではなく明示 manifest を
持ち、group 単位で subprocess として順次実行する。non-zero return（prerequisite 不足を含む）は
失敗として扱い、skip で通さない。

```bash
python3 tests/server/run_server_regression.py --all
python3 tests/server/run_server_regression.py --group reasoning --group reasoning-tools
python3 tests/server/run_server_regression.py --list
```

group と test 一覧は `tests/server/run_server_regression.py` の `GROUPS` が正本である。
現在の group は `core` / `constraints` / `tools` / `structured` / `composition` /
`prefix-cache` / `reasoning` / `reasoning-tools` / `concurrency-cancel` / `capability` /
`agent`。

## GPU 割り当て

`GPU_COUNT > 0` の test は `phaseshift-gpu-test-runner` にラップされる。
runner は次のことを行う。

- 使用 GPU の予約と VRAM budget の管理（`PHASESHIFT_TEST_GPU_BUDGET_GB`）
- `HIP_VISIBLE_DEVICES` を test binary へ伝える
- `--timeout` に対する hard deadline（超過で SIGKILL、state を taint）

予約状態は `<build>/gpu-test-state/` に置かれる。**この state は build dir 単位**
であり、別の worktree や別のリポジトリで走っているプロセスの GPU 使用は反映されない。
結果として、他者が大量の VRAM を使っていても runner はそれを空きと判断し、
同じ GPU を選ぶことがある。

同じ GPU を避けたい場合は、候補 GPU を明示する。`PHASESHIFT_TEST_GPUS` は
`candidate_gpus()` の最優先で読まれる。

```bash
PHASESHIFT_TEST_GPUS=2,3 ctest --test-dir build -R test_rccl
```

指定した GPU が足りない場合、runner は `SKIP: insufficient GPUs` を返して
exit 77 になる。

## 実行層

3 層を混同しない。

```
required acceptance   self-contained C++ / GPU suite（tests/run_required_acceptance.py）
server regression      model + LocalAI + compute E2E（run_server_regression.py）
closure soak           manual 長時間 stability test（test_server_closure_soak.py）
```

`support.py` は共有 helper であり test ではない。`test_server_closure_soak.py` と
`tools/bench_server_reasoning_constraints.py` は manifest へ入れない。

`test_structural_tool_constraint`（required）に reasoning envelope の CPU 試験を含む。
