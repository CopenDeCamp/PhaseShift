# Resident Model Infrastructure

GPU model weight の lifetime を compute process から分離する基盤の検証記録。
現在の contract は `docs/developer/resident_model.md` が正本であり、
この文書は inventory・採否判断・計測の履歴である。

## 目的

model lifetime / compute lifetime / request lifetime を分離し、
compute の crash・timeout・hang が weight を巻き込まないようにする。
test session はこの基盤の最初の consumer であり、同じ重みを複数 worker から
再利用して model reload を削減する。test isolation は維持する。

当初案では model host を test support として扱っていたが、
「compute hang → compute kill → weight は残す → compute restart → attach」という
本来の要件は production / server にも同じ価値があるため、
`phaseshift-model-host` を internal runtime component として
`ModelSource`（local load / resident attach）の runtime capability に位置付け直した。

## inventory と分類（Gate 0）

機械的な検索対象: `Qwen35Model::load_from_safetensors` /
`load_qwen35_weights_from_safetensors` / `load_qwen35_weights_tensor_parallel_rank` /
`load_dflash2_weights` / `phaseshift-compute` subprocess / `ServerHarness` /
`ComputeHarness` / CMake label `external_files`。

### loader correctness（resident 化しない）

load path 自体が検証対象であるため、従来どおり safetensors を読ませる。

- `test_weight_load`
- `test_qwen35_mtp_weight_load`
- `test_qwen35_tp_weight_load`
- `test_qwen35_mtp_weight_real`
- `test_dflash2_weight_real`
- `test_dflash2_weight_contract`

### runtime / E2E / perf（resident 化した）

- GPU-MCU acceptance 実モデル 6 本（production decode、qwen one layer inventory、
  full transformer body plan、attention one layer plan、qwen one layer plan、
  attention layer inventory）
- DFlash2 gate 3 / 4 / 5 の real・perf（draft のみ）
- DFlash2 gate 6 / 7 / 8 / 9 / 11 profile（target + draft）
- DFlash2 gate 11c perf / 11d / 11h、gate 51 target fixture（target のみ）
- DFlash2 gate 51 trace（draft のみ）
- `test_dflash2_target_taps`
- `test_qwen35_tp_e2e` / `test_dflash2_tp_e2e`（TP rank attach）
- `phaseshift-compute`（serve-stdio、single-shot、server 経由）

### 対象外

- `test_dflash2_gate10_cli` / `test_dflash2_gate11c_e2e` — プロセス内では load せず、
  `phaseshift-compute` を subprocess 起動する。env 継承で attach される。
- model を読まない CPU contract / small kernel / substrate test
  （`phaseshift-required-tests` は resident 非依存のまま）。

## 採否判断

### 1. compact block の作成方法: D2D copy → host bounce を挟む

当初手順は「staging の used prefix を device-to-device copy して compact block を作る」
だったが、R9700（31.86 GiB）で 27B-PSQ（used 16.83 GiB）を実測したところ
staging と compact の同時確保が 33.7 GiB となり OOM になる。

採用: used prefix を host bounce buffer へ D2D→H2D で退避し、staging を解放してから
compact `hipMalloc` を作って H2D する。device ピークは used 近傍に収まり、
disk load は 1 回のままである。host RAM の使用は used 相当（16.83 GiB）で、
この環境の空きメモリに対して十分に小さい。

### 2. VMM shareable handle による直接 export は見送り

VMM export はコピー自体を消せるが、fd の SCM_RIGHTS transfer と worker 側の
address reserve / import / map が必要になり、初版の IPC は `hipIpcOpenMemHandle`
に統一した。bounce を挟める以上、メリットは size のみで、PoC を挟まずに
採用しなかった。

### 3. device budget の算出: host 合計 resident bytes を使う

worker が自分の attach 分だけ差し引くと、host が別プロファイルの entry を保持している
場合に VRAM を二重計上する。HELLO が返す host 合計 resident bytes を使い、
`max(自分の attach 分, host 合計)` を差し引く方式にした。

ACQUIRE の処理中に host が entry を release すると HELLO の値が古くなるため、
ACQUIRE 成功後にもう一度 HELLO して最新値を取得している。

### 4. load 失敗時の release と retry

同一 session 内で load option が異なる key（例: verify CRC の有無）が
連続した場合、既存 entry が staging の physical commit を圧迫して
quantized matrix load が OOM になる。host はこのとき既存 entry を
明示的に release して 1 回だけ retry する。LRU は作らず、
session runner が test を直列に実行することを前提とする。

## 計測

環境: AMD Radeon AI PRO R9700 (gfx1201) ×4、ROCm 7.15、
target `models/Qwen3.8-27B-PSQ`（used 16.83 GiB）、
draft `models/Qwen3.8-27B-DFlash2-PSQ`（used 1.16 GiB）。

### server regression core group

| 項目 | 変更前 | 変更後 (resident) |
|---|---|---|
| pass / fail | 7 / 0 | 7 / 0 |
| test body 合計 wall time | 187.4 s | 43.9 s |
| session wall time（host warmup・shutdown 含む） | 187.4 s | 85 s |
| model disk load count | 13（compute process 起動回数） | 4（device 2 × profile 2） |
| peak VRAM | 33,675,661,312 B | 33,970,262,016 B |

peak VRAM の差は model host の HIP context 分（約 0.3 GiB）である。

### host backend external test

| test | 変更前 | 変更後 (resident) |
|---|---|---|
| `test_dflash2_target_taps` | 12.51 s（disk load 含む） | 0.4 s（attach のみ） |
| 同 session の disk load count | 1 / test | 1 / session（変化なし） |

### GPU-MCU acceptance 6 本

この環境には `PHASESHIFT_MODEL_DIR_4B` に該当する Qwen3.5-4B が無いため、
変更前後とも exit 77 で skip する。resident session 内でも skip は skip のままであり、
PASS には数えていない。

### TP

| test | 結果 |
|---|---|
| `test_qwen35_tp_e2e` | resident session 内で PASS（rank0 / rank1 各 1 disk load） |
| `test_dflash2_tp_e2e` | key 連続による memory pressure を release + retry と HELLO refresh で解消 |

## 判断の現在への影響

- production serving contract は変更していない。
- `phaseshift-compute` の resident branch は startup のみで、decode hot path に
  branch を追加していない。
- loader correctness test は resident を使わない。

## 残課題

- Qwen3.5-4B model が無い環境では GPU-MCU acceptance 6 本の resident 実測ができない。
- VMM shareable handle による copy なし export は未検証。
- profile の自動 evict は作っていない。session runner の直列実行と
  明示 release に依存する。
