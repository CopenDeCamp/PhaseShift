# Roadmap

PhaseShift の現在の作業対象と、今後予定している作業の一覧である。
完了済み項目は載せない。過去の経緯は Git history と `docs/rnd/` を参照。

現行のゴールと性能目標は `docs/developer/objective.md`、現在の性能は
`docs/perf/current.md` を参照。

## 1. GPU-MCU の hang 検出と復旧

substrate と Qwen3.5 GPU-MCU backend の contract は
[gpu_mcu/architecture.md](gpu_mcu/architecture.md) に記述した。
low-level 契約は `docs/developer/gpu_mcu/low_level.md` を参照。
ここには異常時の観測と復旧だけを残す。

- MCU の hang 検出。heartbeat / stop flag は存在するが、
  stall を判定して復旧に踏み切る経路はまだない。
- 検出した場合の復旧。shutdown 経路（stop request → wait stopped →
  stream sync、destructor は最終 fallback）は実装済みで、方針は変えない。
- GPU-MCU backend への TP=2 追加は下記 4 と別件である。

## 2. Kernel / runtime 最適化

- 不要な dispatch / workspace / copy の削除
- KernelId 粒度の再確認
- state transition / dependency の簡略化
- kernel family / KernelConfig の整理

## 3. Host backend に TP=2 を追加

## 4. GPU-MCU backend に TP=2 を追加

## Server

`phaseshift-server` は Feature Complete として維持する。API 機能追加は現在の
roadmap の主対象ではない。Server の現在の surface と詳細 contract は
[server_status.md](server_status.md) を入口に各 doc を参照。
