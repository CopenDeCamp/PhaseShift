# INT2と固定draft語彙の組合せ・既定選択

2026-09-29、`1613544fa0f1ec96aae27298fa8ecb7bd11ce579` を基準に試作し、
PR候補を上流 `80ef4bf` のradix selectorへ統合した。
現在の契約は[developer文書](../../developer/dflash2.md#固定語彙profile)、
配置操作は[user文書](../../user/compute.md)を正とする。

## 採用範囲

対応する標準profileがtarget modelへ配置されている場合、INT2＋固定語彙を既定選択する。
profileがないモデルは従来経路を使う。明示offを尊重し、壊れたprofileはエラーにする。
DFlash2自体の有効化、target head、Exact verifyは変更しない。

選択行だけを元のPSQ8からINT2へpackする。codebookは従来と同じ全headから作り、
Top-N後にlocal IDをglobal IDへ戻して、元のPSQ8 headでrerankする。
以前の縮小PSQ8単独PoCとCRC/ZLIB実験差分は候補から外した。

## 配布用profile

98,304個の数値ID、metadata、NOTICEの3ファイルをまとめる。
元の会話・記事・source code・token文字列・文書別統計・model重みは含まない。
7 domainの相対token頻度を等重みで合算し、33 special IDを保持した集合である。
SWE-chatだけから生成したprofileではない。利用者は生成済みprofileを使い、元のcorpusを取得しない。

| 成果物 | 寸法・SHA-256 |
| --- | --- |
| uint32 payload | 393,216 bytes（384 KiB）; `24bff6b41785a7729bff183dfea7997e6446173e0df7254cc5761a7519fdebd0` |
| 配布用archive | 139,983 bytes（約137 KiB）; `b7bd94c9131c3ef3573c27c92a512b5463bc9676c57a1a70ac0223fc2f8df246` |
| canonical token→ID digest | `1ea47dc65bdc42fb4bfacda3f3f2b60ca6e996c8a71331edafc9c5e7d0dd2a2a` |

archiveは[contributorのfork](https://github.com/jyohukuchan/PhaseShift/releases/download/draft-vocab-qwen38-v1/qwen38-draft-vocab-98304-v1.tar.gz)で提供する。installはtoken→ID対応を確認してから、
配置先tokenizer.jsonのSHAへmetadataを結び直す。既存の別profileは明示overwriteを要求する。
置換中の処理エラーでは旧3ファイルを復旧し、復旧失敗時はbackupの場所を示す。
metadataは最後に置換するが、電源断を含む複数ファイル全体のatomic更新は保証しない。

## 配布条件の確認範囲

生成元と固定revision・license URIは[NOTICE](../../references/qwen38-draft-vocab-notice.txt)に記録した。
SWE-chatとFineWebの公開表示はODC-Byであり、元の個々のContentsの権利は別に扱う。
数値profileについてcontributorが保有する権利の範囲をODC-By 1.0として表示し、
元corpusをMITへ再ライセンスしたとは扱わない。tool/runtimeのコードはMITのままである。

元の文章・コード表現を含まない機械的な集計成果物として扱うのは、成果物の構造に基づく判断である。
Wikipediaや各repositoryの条件が一般に消える、元の全コンテンツを再配布できる、と一般化しない。
公開条件には本件の数値集計成果物を明示的に配布禁止とする条項を見つけなかったが、
新規HFアカウントで確認したユーザーの報告と提供画像では、アクセス申請時の同意事項は
メールアドレスとユーザー名の共有で、追加の再配布禁止は表示されていなかった。
初回調査の「認証後画面が未確認」という留保は解消した。
元の全入力ファイルの権利関係について、独立した法的な認定を行ったものではない。

## 先行PoCの測定

Qwen3.8-27B PSQ＋DFlash2 PSQ、R9700/gfx1201、ROCm 7.14、greedy、BF16 KV、幅7、pool32。
語彙は248,320から98,304へ削減。以下は同じ語彙と数値経路を用いた先行PoCであり、
既定選択の追加・上流radix selector統合後に全性能比較を再測定した数値ではない。
最新mainのradixによって全語彙側も高速化しているため、以下の速度差を最新main比の改善率とは扱わない。

| 指標 | INT2単独 | INT2＋固定語彙 |
| --- | ---: | ---: |
| INT2 codes | 303.125 MiB | 120 MiB |
| 追加compact scale＋ID map | なし | 30.375 MiB |
| 候補選択時間（別診断、32 token・4 round平均） | 1.140 ms | 0.629 ms |
| 日本語生成速度（5回中央値） | 73.78 tok/s | 74.70 tok/s |
| 英文生成速度（5回中央値） | 54.10 tok/s | 54.80 tok/s |
| コード生成速度（追加AB/BA、各2回中央値） | 163.44 tok/s | 165.07 tok/s |

logits/scratchを含むarena内使用領域の寸法差は160.8916015625 MiB。
予約arena容量やprocess peak VRAMが同量減ったという測定ではない。
元の既定full PSQ8と比較すると、compact codes/scales/mapとして約150 MiBを追加する。

日本語・英文は最大256 token、コードはEOSまで212 tokenで測定した。
主比較には大きな外れ値と英文1回の逆転があり、物理原因は未特定。
コード主比較の+5.76%は効果の推定に使わず、追加AB/BAでは約1.00%だった。
候補選択の約45%短縮を生成全体の改善率とは扱わない。

主比較30回と追加AB/BAで生成列・受理数は一致した。
compact coarseとfull INT2の対応行、PSQ8 rerankと別のfull-head GEMMの対応行は最大誤差0。
短い3入力での確認であり、入力分布によって受理率や速度が悪化する可能性は残る。

## 既定選択の確認

- host resolver: 標準profile、自動/明示選択、欠落、非適合、破損、ID境界、env優先順位を確認。
- pack/install: 異なるJSON表現の同じtoken→ID対応、異なるID対応、hash不整合、
  既存ファイル保護、上書き失敗時の復旧、新規配置失敗時の後始末を確認。
- R9700実モデル: profile未配置で従来動作。配置後は自動選択、明示full PSQ8、
  全語彙INT2、明示INT2＋固定語彙の4モードで32-token生成列が一致。
- serve-stdio: 自動profile選択で生成成功。stdoutは7件のJSON eventだけで、起動診断はstderr。

これらは機能確認であり、追加の速度推定には使わない。広い入力分布での安定高速化の証明とは区別する。

## 最新mainへの統合確認

上流`80ef4bf`のradix Top-Nを維持し、compact語彙の寸法でscratchを確保するように統合した。
radixと従来selectorの双方で、PSQ8 rerank前のglobal ID復元を確認した。

- gfx1201 Release build、host resolver/installerのCTestを通過。
- R9700でradix数値テストのCPU oracle対照432項目を通過。
- 統合後もauto/full PSQ8/full INT2/明示combinedで32-token生成列が一致。
- 自動radixと従来selectorの両方でcompact coarse/full INT2、rerank/full PSQ8 GEMMの対応行が誤差0。
- serve-stdioは自動選択で4 tokenを生成し、stdoutはJSONのみ。

テストrunnerのUUID解釈で別GPUが選ばれたため、radixテストはR9700のUUIDを指定した直接実行で確認した。
今回の生成速度の値は性能推定に使わず、先行PoCの数値とも混ぜない。
