> Status: historical R&D record（現在の仕様ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# Generic GPU-MCU と Qwen35 Backend の境界調査

- 目的: contract 分離を終えた `runtime/gpu_mcu/execution/` を読み直し、
  Generic GPU-MCU executor から kernel / model semantics を追い出せるかを調査する
- 対象外: コード変更。本記録は調査結果と次フェーズの候補を残すもので、
  どちらを採用するかの判断は次フェーズで行う
- 基点: branch `refactor/gpu-mcu-contract-boundaries`

## 1. contract 分離後の execution/

| header | 行数 | 内容 | model 識別子 |
|---|---|---|---|
| `micro_fsm.h` | 382 | `GpuMcuFsmState` / `GpuMcuFsmRunContext` / `GpuMcuFsmConfig` / `GpuMcuFsm` と FSM public entry point | 容量定数にのみ |
| `fsm_contract.h` | 155 | supervisor / fault / doorbell / log / dispatch record / plan node / kernel variant / timing | **0** |
| `invocation_abi.h` | 685 | `Mcu*Invocation` 24 構造体と `static_assert` 282 件 | 208 |
| `kernarg_recipe.h` | 40 | `kMcuKernargRecipe*`（`None = 0` … `Count = 29`） | 18 |
| `binding/plan_binding_contract.h` | 55 | `McuDynamicNodeBinding` / `McuInvocationPatchSource` / `McuInvocationPatch` | 0 |

`micro_fsm.h` のトップレベル型定義は 4 件まで減り、「GPU-MCU 全 ABI の倉庫」の状態は解消した。
`static_assert` の総数は分離前後で 289 件のままである。

## 2. execution/detail/ の分離状況

| header | 行数 | 概念 | 主要 entry |
|---|---|---|---|
| `mcu_log.h` | 36 | logging | `mcu_log_fault`, `mcu_log_event` |
| `micro_fsm_control.h` | 275 | control | `mcu_control_loop`, `gpu_mcu_execution_step` |
| `micro_fsm_run_once.h` | 697 | plan execution | `mcu_run_once` |
| `micro_fsm_host.h` | 680 | host bridge | host 側 control / state 入出力 |
| `micro_fsm_kernarg.h` | 989 | kernarg build | `mcu_write_*_kernarg` 26 種 |

**5 ファイルが control / plan execution / kernarg build / logging / host bridge の
5 概念に対応しており、ファイルの再分割は必要がない。**

分割ではなく配置の問題は前フェーズで解けている。
残るのは内容であり、`micro_fsm_kernarg.h` が model semantics の最大保有者である。

### 次フェーズ候補（本フェーズでは実施しない）

- `micro_fsm_kernarg.h` の 989 行を、recipe ごとの write 関数群と
  共通の slot / queue / hidden args 処理に分けるか。
- `micro_fsm_host.h` の 680 行が host bridge 以外の責務を含まないか。

アルゴリズム変更は禁止のままとし、次のフェーズで内容を読んだ上で判断する。

## 3. micro_fsm_kernarg.h の調査

### 3.1 recipe ごとの switch は存在する

2 箇所ある。

| 位置 | 内容 |
|---|---|
| `mcu_recipe_explicit_args_bytes(uint16_t recipe)` | recipe → `sizeof(GpuMcuFsmWorkerArgs)` または `sizeof(Mcu*Invocation)` の 29 分岐 |
| `mcu_write_*_kernarg` の dispatch | recipe → `mcu_write_*_kernarg` の 29 分岐 |

`case kMcuKernargRecipe*` の総出現は 56 件、内訳は 2 箇所の 29 分岐である。

### 3.2 generic execution が kernel field の意味を直接理解している

`mcu_write_*_kernarg` は 26 関数あり、`mcu_write_rmsnorm_kernarg`,
`mcu_write_attention_paged_kernarg`, `mcu_write_gdn_conv1d_kernarg` 等が
個々の `Mcu*Invocation` の field に意味を与えて構築している。

参照している `Mcu*Invocation` 型は 25 種、
model/kernel 識別子の出現は 111 件である。

つまり **generic MCU が kernel field の意味を知っている状態が実在する。**
これが Phase 3 の主対象である。

### 3.3 plan compiler へ移せる処理

以下のものは plan compiler 側で事前に確定できる。

- `mcu_recipe_explicit_args_bytes` が返す値。
  variant の `kernarg_size` として plan 側で既に参照している箇所があり、
  recipe と invocation type の対応は plan compiler の知識である。
- `mcu_write_*_kernarg` が固定する field のうち、
  plan の内容だけで決まるもの（shape に依存しない定数、variant 選択）。

実行時でなければ決められないもの（行数・request 数に依存する値）は、
`binding/plan_binding_contract.h` の `McuInvocationPatch` として
既に dynamic binding 側に切り出されている。

### 3.4 prebuilt kernarg template

`mcu_recipe_explicit_args_bytes` が定数 `sizeof` の switch であることは、
recipe ごとに kernarg の大きさが固定であることを意味する。
template を事前に作れるかは、`mcu_write_*_kernarg` が
実行時値で上書きする field の集合を列挙した後に判断する。

本フェーズでは判断しない。

## 4. GDN の所在

`model_hooks/gdn_reset` を移動しただけでは model 依存除去は完了しない。
GDN は次の箇所に存在する。

| location | 内容 |
|---|---|
| `execution/invocation_abi.h` | `McuGdnConv1dInvocation`, `McuGdnRecurrenceInvocation`, `McuGdnSpecRestoreInvocation`, `McuGdnSpecRestoreFromCountsInvocation` |
| `execution/kernarg_recipe.h` | `kMcuKernargRecipeGdnConv1d`, `GdnRecurrence`, `GdnReset`, `GdnSpecRestore`, `GdnSpecRestoreFromCounts` |
| `src/execution/detail/micro_fsm_kernarg.h` | `mcu_write_gdn_*_kernarg` 群 |
| `src/execution/detail/micro_fsm_host.h` | GDN 相関の host 側出力 |
| `model_hooks/gdn_reset.*` | GDN state reset |

`model_hooks/` は quarantine として維持する。generic 化は本フェーズでは行わない。

## 5. 次フェーズのテーマ

`execution/invocation_abi.h`、`execution/kernarg_recipe.h`、
`src/execution/detail/micro_fsm_kernarg.h` の 3 つが
Generic GPU-MCU と Qwen35 Backend の境界問題を最も明確に示す。

テーマは次である。

> Generic GPU-MCU executor から kernel / model semantics を追い出せるか

前フェーズのディレクトリ整理、本フェーズの依存方向整理と contract 分離を経て、
「micro_fsm.h が何でも知っている」状態は解消した。
次は内容に踏み込む。
