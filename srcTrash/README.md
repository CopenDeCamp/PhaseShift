# srcTrash

削除予定コードの一時保管場所。

- isolation date: 2026-10-03
- isolation 前の commit SHA: `7e27739ceb2d2fc6228728fb1fd81fb188a10fa0`
  (`docs(refactor): 全機能棚卸しの監査資料を追加する`, branch `refactor/src-trash-isolation` 起点)

## ルール

- srcTrash は削除予定コードの一時保管場所である。
- 元の relative path を維持する(`foo/bar.cpp` → `srcTrash/foo/bar.cpp`)。
- `srcTrash/` は CMake target / include path / test target に**追加しない**。
- srcTrash 内のコードをコンパイルされる状態にしない。
- restore する場合は Git history または srcTrash から戻す。
- `srcTrash/fragments/` には、独立ファイルではなく既存ファイルの一部だけを
  隔離した断片を `srcTrash/fragments/<元パス>/<feature名>.txt` として保存する。

## 隔離一覧

各項目に reason / original path / fallback path / moved tests / removed CLI・env /
measured result / restore note を記載する。

(Phase 実施後に記入)
