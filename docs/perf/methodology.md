# Measurement methodology

RDNA4 (gfx1201) 上で最適化を計測・採否判断するときに今後も使う測定原則の正本である。
`docs/rnd/optimization_findings.md` など R&D の記録は、測定原則について本ドキュメントを
正本として参照する。個別 Gate の時系列ログや個別カーネルの結果はここへ書かない
（`docs/rnd/` を参照）。

## 1. 隔離と環境固定

- **1 GPU = 1 bench process**。同一 GPU 上に複数 workload を重ねない
  （clock / 帯域の非対称汚染）。
- 複数 GPU を使う場合も、ある GPU では同時に 1 つだけ動かす。
- GPU0 は別 workload と共有していることがある。隔離された device を明示する。
- 同一コマンドで条件を揃える: seed / warmup / samples / launches / rows / dtype / model。

## 2. baseline / candidate

- **baseline と candidate は交互に測る**（run1: base→cand, run2: cand→base, ...）。
- **cross-GPU 比較は禁止**（個体差・clock state が乗る）。
- baseline は **production incumbent**（その時点の selector が実際に選ぶ経路）。
  「同じ dtype の類似 kernel」を代役にしない。
- **correctness / reference kernel を p50 baseline にしない**。参照実装は数桁遅く、
  比較に意味がない。
- fused candidate の比較対象は unfused production sequence（現行 selector の連結）。

## 3. warmup と再測定

- 5-run の p50 中央値 + **run 間安定性**で判定する。
- run 間で p50 が 30% 以上ぶれたら **FAIL / BLOCKED として調査**する
  （並列汚染の検出器でもある）。
- 数値を緩めて通さない。再測定で汚染を除く。
- micro kernel（4〜20µs 級）は idle clock ramp の影響が大きい。**preroll を入れる**
  （baseline 側にも同条件で）。
- 初回プロセスは GPU クロックのランプアップで遅く出るため warmup で除外する。

## 4. production reachability

- source tree にある / CMake に入る / public bench から叩ける ≠ production reachable。
- reachability は **runtime selector / matcher の rule table** から導出する。
- rule が無い経路は NON_PRODUCTION。release blocker に数えない。
- **「負けるカーネルを選ぶ rule」は promotion contract violation**（本物の FAIL）。
  勝つ側を選ぶよう rule を直すのが正解で、カーネルを微修正しても解決しない。

## 5. 測定 harness 自体を疑う

実測で見つかった harness 側の欠陥:

- device 選択が weight / workspace 確保より後 → `--device N` が効かず別 GPU で確保。
- CLI option の非対称 → usage error で perf が欠測。
- `--check` が no-op（受理されるが何も比較しない）。
- bench 入力がカーネルの前提を満たしていない。
- 複数 kernel バッチの GPU event 両端計測が wall-clock に対して非決定論的にずれる。
  その場合は per-op event か wall-clock を使う。

## 6. E2E と micro benchmark を混同しない

- micro benchmark は単体カーネルの ceiling と律速（帯域 / compute / launch）を切り分ける。
- E2E（pp / tg / speculative decode）は selector 連結を含む現実の性能を示す。
- 片方の改善を他方の結論として流用しない。
- **token 一致を確認する**。正しさを確認せずに速度だけを見ない。
- correctness path の出力と production incumbent の出力を混同しない。

## 7. launch floor と cache residency

- 小カーネルは launch 律速になりうる。1 dispatch あたりの固定費を先に疑う。
- decode は per-token weight traffic が L3 容量を超えるなら DRAM 帯域律速。
  L3 に収まる状態（state 等）では compute 律速に切り替わる。どちらかを取り違えない。
- 「ISA で説明できるから速くなる」は誤り。**必ず同一条件で A/B を取る**。

## 8. 報告

- 性能に関する主張には計測条件（revision / commit / GPU / コマンド）と数値を添える。
- 負結果も残す。採否判断と理由を記録する。
- 新しい計測をしていない数値を「更新」しない。
