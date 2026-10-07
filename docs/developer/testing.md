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
external model 不在・GPU 不足などによる exit 77 は `SKIP_RETURN_CODE 77` で
CTest の Skipped として扱って、plain `ctest` が誤った FAIL を出さないようにしている。
Skipped は開発者ローカル向けの表示にすぎず、required acceptance は JUnit の
`<skipped>` を failure として数えるため通らない。

## GPU 予約

`GPU_COUNT > 0` のテストは `phaseshift-gpu-test-runner` で wrap され、
GPU を論理 VRAM で予約してから実行する。予約できないままで
`PHASESHIFT_TEST_GPU_RESERVE_TIMEOUT`（既定 900 秒）を経過すると失敗する。

- 予約量は `GPU_COST_GB`（既定 1 GiB）。上限は各 GPU の実 VRAM
  （`/sys/class/drm/<node>/device/mem_info_vram_total`）から求める。
  明示指定は `--budget-gb` と `PHASESHIFT_TEST_GPU_BUDGET_GB`。
- iGPU は候補から外す。判定は `/sys/class/drm/<node>/device/firmware_node` の有無で、
  iGPU のみの環境では外した後の空リストを避けて iGPU を候補にする。
- `PHASESHIFT_TEST_GPUS` / `HIP_VISIBLE_DEVICES` が指定されている場合は
  明示指定として扱い、iGPU 除外を適用しない。

### GPU 負荷をかけている間の監視制約

GPU に負荷をかけている間は **SMU metrics を取得してはならない**。
`rocm-smi` / `amd-smi metric` / `/sys/class/drm/*/device/gpu_metrics` の読み出しは
SMU へ metrics export を要求し、応答が無いと次が出る。

```text
SMU: No response msg_reg: 12 resp_reg: 0
Failed to export SMU metrics table!
```

このとき得られた異常値から `GPU over temperature range(SW CTF)` が宣言され、
`System is going to shutdown due to GPU SW CTF!` でマシンが落ちる事故を観測している。

- 温度は **負荷実行の直前と直後だけ** で取る。実行中は打たない。
- 温度取得の正本は `amd-smi metric -t`（root 不要。EDGE / HOTSPOT / MEM を返す）。
  `rocm-smi --showtemp` は dGPU を返さない（iGPU のみ）ため正本にしない。

```bash
amd-smi metric -t        # 実行前
<負荷テスト>
amd-smi metric -t        # 実行後
```

### GPU ハング時の収手順

GPU が wedge して `devcoredump` が作られたら、**再起動する前に** 取る。再起動で消える。
`card` は PCI アドレスで選ぶ。

| HIP device | PCI | drm |
|---|---|---|
| 0 | `0000:07:00.0` | `card1` |
| 1 | `0000:0a:00.0` | `card2` |
| 2 | `0000:0f:00.0` | `card3` |
| 3 | `0000:12:00.0` | `card4` |
| iGPU | `0000:83:00.0` | `card5` |

```bash
for c in /sys/class/drm/card*/device; do printf '%s -> ' "$c"; readlink -f "$c"; done
sudo cat /sys/class/drm/card<N>/device/devcoredump/data \
  > amdgpu-devcoredump-$(date +%Y%m%d-%H%M%S).bin
```

あわせて kernel log を保存する。再起動後は `journalctl -k -b -1` で前 boot のログが読める。

```bash
journalctl -k -b -1 > amdgpu-kernel-prev-boot.log
```

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
  model directory は `PHASESHIFT_MODEL_DIR` で指定する。model size には依存せず、
  指定が無い・directory が無い場合は exit 77 で skip する。
  required acceptance の対象外であり、required acceptance では skip を失敗として扱う。

## optional tests

`PHASESHIFT_BUILD_OPTIONAL_TESTS=ON` で external model を要する heavy test
（DFlash2 gate、GPU-MCU acceptance、TP、MTP weight contract 等）が build される。
manifest は `cmake/tests.cmake` が正本であり、件数は固定しない。

external test の model directory はすべて環境変数で指定する。
`PHASESHIFT_MODEL_DIR`（Qwen3.5 系の既定）、`PHASESHIFT_MODEL_DIR_MTP`、
`PHASESHIFT_MODEL_DIR_DFLASH2`、`PHASESHIFT_MODEL_DIR_DFLASH2_TARGET`、
`PHASESHIFT_TP_MODEL_DIR`。cmake option に model directory は持たない。

Qwen3.5-4B full application E2E は削除済みである。外部 model が無い環境で
skip を垂れ流さないため、実行できない suite は repository に残さない。
`tests/fixtures/qwen35_4b_oracle.json` は generation stop token contract
（`test_server_eos_contract.py`）の入力として残している。

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

## resident model session

実モデルを読む重い GPU テストは、resident model host 経由で重みを session 中共有できる。
仕様の正本は [resident_model.md](resident_model.md) である。

GPU テストの session runner:

```bash
PHASESHIFT_MODEL_DIR_DFLASH2_TARGET=models/Qwen3.8-27B-PSQ \
  python3 tools/run_resident_gpu_tests.py --list
python3 tools/run_resident_gpu_tests.py --group host --devices 0
python3 tools/run_resident_gpu_tests.py --group tp --devices 0,1
```

server regression の resident mode:

```bash
PHASESHIFT_MODEL_DIR=models/Qwen3.8-27B-PSQ \
  python3 tests/server/run_server_regression.py --group core --resident-model
```

通常 mode（`--resident-model` なし）は従来挙動のままである。resident mode は
test script の process isolation、`ServerHarness` / `ComputeHarness` の lifecycle、
timeout / failure isolation を変更しない。変わるのは weight の lifetime のみである。

session runner は worker の deadline 超過時に process group を kill し、
kill の後に model host の health check を行う。healthy なら次の worker を起動し、
unhealthy なら session を fail-closed で終了する。

loader correctness test は `PHASESHIFT_DISABLE_RESIDENT_MODEL=1` で
resident path を無効化できる。environment が無ければ従来 loader path が使われる。

## 実行層

2 層を混同しない。

```
required acceptance   self-contained C++ / GPU suite（tests/run_required_acceptance.py）
server regression      model + compute E2E（tests/server/run_server_regression.py）
```

`support.py` は共有 helper であり test ではない。manifest へ入れない。
