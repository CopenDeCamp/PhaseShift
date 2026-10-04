> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/gpu_mcu/architecture.md`、現在の性能は `docs/perf/` を参照）

# GPU-MCU Real Primitive Chain PoC

- 目的: probe worker ではなく PhaseShift の実 production kernel を GPU-MCU から AQL dispatch する
- 目標: Real RMSNorm → Real Activation Quantize A8 → Real PSQ4 Decode1 を、
  GPU-MCU / variant registry / device-generated kernarg / retained AQL / continuous feeder だけで
  primitive 間 Host 介入なしに実行する
- 対象外: Qwen full layer、production executor 置換、MoE、multi-GPU、EXT_KERNEL_DISPATCH、
  packet fusion、全 variant 対応
- 性能の GO/NO-GO はしない
- 詳細契約: `docs/developer/gpu_mcu/low_level.md`

## 1. 基点

| 項目 | 値 |
|---|---|
| baseline | `a1c623f0bd8ebddac3326263b611bd37c5772e38` |
| worktree | `.worktrees/gpu-mcu-real-primitive-chain` |
| branch | `poc/gpu-mcu-real-primitive-chain` |

## 2. Gate 結果

| Gate | 内容 | 結果 |
|---|---|---|
| R0 | baseline 記録（feeder depth=4 / per-packet / Agent を default に） | 確定 |
| R1 | static region kernarg lifetime | PASS（`test_gpu_mcu_kernarg_region_lifetime`） |
| R2 | stable RMSNorm AQL entrypoint | PASS |
| R2A–R2D | BF16/PF/ONE_PLUS 単一 variant、共通 device body、同一 entrypoint、POD args | PASS |
| R3 | RMSNorm HSACO 生成・metadata 検証 | PASS |
| R4 | Direct Host-AQL Real RMSNorm | PASS（`test_gpu_mcu_real_rmsnorm_aql`、F=256/5120） |
| R5 | RMSNorm kernarg recipe + invocation descriptor | 実装済み（baseline 起点） |
| R6 | MCU Real RMSNorm 単体 dispatch | PASS（`test_gpu_mcu_real_rmsnorm_fsm`） |
| R7 | Real RMSNorm continuous feed | PASS（`test_gpu_mcu_real_rmsnorm_feed`） |
| R8 / R8A | Activation Quantize A8 stable entrypoint + Direct AQL | PASS（`test_gpu_mcu_real_activation_quantize_aql`） |
| R9 / R9A–R9D | PSQ4 Decode1 stable entrypoint + Direct AQL | PASS（`test_gpu_mcu_real_psq4_decode1_aql`） |
| R10 | 3 primitive chain | PASS（`test_gpu_mcu_real_primitive_chain`） |
| R11 | queue への先行 enqueue 証明 | PASS（first plan `ahead=3`） |
| R12 | resource audit | spill 0（§7） |
| R13 | performance observation | 計測済み（§8） |
| R14 | 3us CP floor は触らない | 捜査せず |
| R15 | code object registry | 3 種を Host load、MCU は variant_id のみ |
| R16 | variant ID | dense（0=marker、1=RMS、2=Quant、3=PSQ4） |
| R17 | 失敗時切り分け順 | 未発生（各段で PASS） |
| R18 | 新規 required tests | 3 本追加（他は既存から継続） |
| R19 | 既存 production tests | 7/7 PASS（§5） |
| R20 | docs | 本ドキュメント + `low_level.md` |

## 3. Stable kernel symbols

| variant | stable symbol | kernarg | private | group | grid | workgroup | sgpr | vgpr | spill |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 completion marker | `phaseshift_gpu_mcu_fsm_worker` | 312 | 8 | 0 | 1 | 64 | 20 | 7 | 0 |
| 1 RMSNorm BF16/PF/ONE_PLUS | `phaseshift_qwen35_rmsnorm_bf16_pf_oneplus` | 304 | 0 | 32 | 1 | 256 | 29 | 17 | 0 |
| 2 Activation Quantize A8 | `phaseshift_qwen35_activation_quantize_a8` | 304 | 0 | 0 | 20 | 256 | 14 | 14 | 0 |
| 3 PSQ4 Decode1 BF16 U16 | `phaseshift_qwen35_psq4_decode1_bf16_u16` | 48 | 0 | 0 | 64 | 32 | 23 | 108 | 0 |
| （U8 併設） | `phaseshift_qwen35_psq4_decode1_bf16_u8` | 48 | 0 | 0 | — | 32 | 23 | 108 | 0 |

- `kernel_object` は code object load 時に resolve（host が registry へ移し、MCU は variant_id のみ）。
- `sizeof(args)` は全て 48 byte。`kernarg_segment_size` は metadata の値（RMS/Quant は 304、PSQ4 は 48）。

## 4. Chain

shape: RMS `rows=1 features=5120 group=5120`、Quant `k=5120 kp=5120`、
PSQ4 `rows=1 k=5120 n=1024 unroll=16 output=BF16`。

plan:

```
RMS → Quant → PSQ4 → Completion Marker → WAIT → END
```

- 3 primitive の間に WAIT は入れない。AQL barrier=1 と Agent scope に任せる。
- memory policy: real primitive = `barrier=1 / Agent / Agent`、
  final marker = `System acquire / System release`（host handoff 境界のみ）。
- queue: `queue_ahead_depth=4`、`doorbell_mode=per_packet`、`kernarg_slot_count=8`。

kernarg ABI: plan node の `invocation_index` から
`McuRmsNormInvocation` / `McuActivationQuantizeInvocation` / `McuPsq4Decode1Invocation`
を引いて slot へ pack し、`build_aql_hidden_args` + `build_aql_launch_metadata` を書く。

## 5. Correctness

`ctest -L gpu_mcu`: **29/29 PASS**（required 28 + 探索 1）。

required acceptance: **151/151 PASS、Failed 0、Errors 0、Skipped 0**。
（`phaseshift-bench` 未 build で `test_bench_*` 22 件が skip して一回 FAILED になったため、
bench build 後に再実行して通過させた。skip は失敗扱いが正しい。）

`test_gpu_mcu_real_primitive_chain` の中間出力比較（host reference chain と対比）:

| 評価対象 | 比較方式 | 結果 |
|---|---|---|
| RMS output (bf16 5120) | byte exact | match |
| A8 codes (81920 byte) | byte exact | match |
| A8 scales (640 byte) | byte exact | match |
| PSQ4 final (bf16 1024) | byte exact | match |

long run: **1000 plan**、`dispatches=4000`、`completions=1000`、`fault=0 pc=0`、
`hang 0`、`stale kernarg 0`（`poison_all_kernarg` で 0xAB プレーンから生成）。

既存 production テスト（R19）:

| test | 結果 |
|---|---|
| `test_rmsnorm` | PASS |
| `test_activation_quantize_a8` | PASS |
| `test_activation_quantize_e4m3` | PASS |
| `test_gemm_psq4_w4a8_wmma` | PASS |
| `test_gemm_psq4_decode1` | PASS |
| `test_psq4_codebook` | PASS |
| `test_qwen35_rmsnorm_selector` | PASS |

## 6. Queue / lifetime / fence

| 指標 | 値 |
|---|---|
| `queue_ahead_depth` | 4 |
| doorbell mode | per_packet |
| `doorbell_count` | 4000（1/dispatch） |
| `max_ahead` | 4（first plan は 3） |
| `refill_count` | 3996 |
| `queue_empty_count` | 1000（plan 末尾 WAIT 中のみ、plan 内部は 1 以下） |
| `queue_full_count` | 0 |
| wrap | 3（queue size 1024 × 4000 dispatch、overwrite 0） |

- first plan `ahead=3` は、RMS 実行中に Quant / PSQ4 / Marker が ring に
  積まれていたことを示す（barrier=1 なので 1 本以外は未 consummation）。
  real kernel に timestamp を足さず、queue occupancy で先行 enqueue を証明した。
- kernarg lifetime: plan 内 `region_seq` は再利用なし、`kernarg_slot_count=8 >= 4`。
  次 plan 先頭で再利用。

## 7. Resources（R12）

| kernel | sgpr | sgpr spill | vgpr | vgpr spill | scratch | LDS | kernarg | code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| continuous feeder FSM（baseline） | 77 | 0 | 153 | 0 | 0 | 0 | 264 | 21780 |
| real-primitive FSM | 88 | 0 | 160 | 0 | 0 | 0 | 264 | 33636 |
| FSM worker | 20 | 0 | 7 | 0 | 8 | 0 | 312 | 736 |
| probe worker | 16 | 0 | 7 | 0 | 0 | 0 | 296 | 460 |

- **sgpr_spill / vgpr_spill は全て 0**。baseline からの増分は SGPR +11、VGPR +7、code +11856。
  VGPR は 153 → 160 にとどまり、200 等への急増はない。
- real kernel 側の spill も 0（§3）。

## 8. Timing（R13）

correctness 確定後に計測。性能は GO/NO-GO にしない。

計測条件: warm-up 10 回、host chain は 50 回の p50 / min、MCU は 999 plan の steady-state。
GPU 1（acceptance と物理的に分離）で実行。host・MCU ともに code object は計測前に load 済み。

| 経路 | cold | steady p50 |
|---|---:|---:|
| host reference chain（3 HIP launch + 1 sync） | 3332 µs | **39 µs**（min 38 µs） |
| MCU chain（3 実 primitive + completion marker） | 288 µs | **77 µs** |

| 指標 | 値 |
|---|---|
| MCU 1000-plan loop（999 plan） | 77 ms |
| plans/sec | 12974 |
| first plan / drain | 0 ms（1 ms 未満） |
| close | 16〜18 ms |
| total（device 初期化 約 1.6 s を含む） | 1818 ms |

- **steady-state では MCU は host の約 2.0 倍遅い**（77 µs vs 39 µs）。MCU 側は
  3 実 primitive に加えて completion marker を 1 dispatch 追加している。
- **cold の host 3332 µs は HIP の kernel 遅延 load を含む**もので、MCU 側は計測前に
  `GpuMcuAqlCodeObject::load` で load 済みのため、cold 数値どうしを比べない。
  過去に「host 3.4 ms vs MCU 293 µs」と並べたのは、cold/実装負荷の違いを混ぜた不適切な対比であった。
- MCU の 77 µs には **host 側 `wait_plans` の 20 µs 間隔ポーリング粒度**が乗っている。
  GPU 実行時間そのものはより短いはずで、この分離は次 Gate の計測項目とする。
- queue 統計（1000 plan）: `doorbells=4000`（1/dispatch）、`max_ahead=4`、
  `refill=3996`、`empty=1000`（各 plan 末尾 WAIT 中のみ）、`full=0`。
- first plan は `ahead=3`、`doorbells=4`、`refill=0`、`empty=1`。

## 9. Failures / surprises

1. **blocking `hipMemcpy` によるテスト停止（rc=137）**
   `test_gpu_mcu_real_primitive_chain` が first plan 完了後も終了せず `timeout` で
   KILL された。原因はテスト側の readback が blocking `hipMemcpy` だったこと。
   persistent MCU kernel が control stream に常駐したままだため device-wide 同期が返らない。
   既存の `Rig::device_to_host`（`hipMemcpyAsync` + worker stream sync）へ置換して解消。
   **MCU / production 実装は無変更。** GPU hang ではない。

2. **stable entrypoint 化は algorithm 複製なしで成立**
   `rmsnorm_block_body` / `activation_quantize_a8_body` / `psq4_decode1_body` を
   `__device__` に引き出し、既存 template kernel と stable entrypoint の両方が同じ body を呼ぶ。
   Host launcher も同一 entrypoint を launch するため、同一 kernel 比較が成り立つ。

3. **kernarg_segment_size の差**（RMS 304 / Quant 304 / PSQ4 48）
   `build_aql_hidden_args` は `explicit args` の直後に約 24 byte を書く。
   PSQ4 は segment が 48 のため、その書込みは宣言済み segment の外に出る。
   kernarg region は granule（page）単位で確保されるため実際の OOB は無く、
   その kernel は hidden args を参照しないため結果にも影響しない。
   ただし将来 `slot_count > 1` かつ granule が小さい構成では slot 間干渉しうる。
   次 Gate で `kernarg_segment_size >= hidden 末尾` のときだけ書くようにすべき。

4. **A8 scale と decode1 の対応**
   `activation_quantize_a8` は 32 element ごとの scale を出力するが、
   `psq4_decode1` の `CanonicalA8ActivationPolicy::row_scale(0)` は
   `scales[row]`（**1 行に 1 個**）しか読まない。
   host と MCU が同じ 2 つの kernel を使っているため本 Gate の比較は一致するが、
   chain 上は decode1 が per-block scale のうち `scales[0]` しか使わないことになる。
   production の W4A8 経路は `launch_activation_quantize_i8_row`（row scale）と
   `launch_activation_quantize_e4m3` を選ぶため、**接続すべき A8 primitive の対応を
   次 Gate で再確認する必要がある**。現状の一致は「同一 kernel 比較」によるもの。

5. **block 0 だけの completion は不完全**
   実 primitive に MCU completion を持たせない方針どおり、末尾 marker 方式にした。
   多 block grid（Quant grid=20、PSQ4 grid=64）でも marker が region 完了を正しく表す。

## 10. 次 Gate

- 判定: 実 3 primitive chain が primitive 間 Host 介入なしで 1000 plan 安定、
  中間/最終出力は host reference と一致、spill 0、**成功**。
- 継続: stable entrypoint + body 共有、invocation descriptor table、dense variant、
  static region completion marker、kernarg region lifetime、Agent 内部 / System handoff、
  depth=4 + per-packet doorbell。
- 次候補: Qwen one-layer serial plan。まずは全 primitive を barrier=1 + continuous feed で
  直列 correctness を通し、その後 Q/K/V と Gate/Up の parallel issue、join を追加。
- 未解決: A8 primitive の対応関係（§9-4）、hidden args の書込み guard（§9-3）、
  CP dispatch floor、host launch overhead の内訳。
