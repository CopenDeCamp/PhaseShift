# Roadmap

PhaseShift の現在の作業対象と、今後予定している作業の一覧である。
完了済み項目は載せない。過去の経緯は Git history と `docs/rnd/` を参照。

現行のゴールと性能目標は `docs/developer/objective.md`、現在の性能は
`docs/perf/current.md` を参照。

## 1. Kernel / runtime 最適化

- 不要な dispatch / workspace / copy の削除
- KernelId 粒度の再確認
- state transition / dependency の簡略化
- kernel family / KernelConfig の整理

## 2. Host backend に TP=2 を追加

## Server

`phaseshift-server` は Feature Complete として維持する。API 機能追加は現在の
roadmap の主対象ではない。Server の現在の surface と詳細 contract は
[server_status.md](server_status.md) を入口に各 doc を参照。
