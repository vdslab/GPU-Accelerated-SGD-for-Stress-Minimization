# 論文ワークスペース

GPU-Accelerated SGD for Stress Minimization の研究完了と論文執筆を、1 つのフォルダで管理するためのワークスペースです。

このフォルダは、最終的に中身だけを独立した論文リポジトリへ移せるように、研究実装のディレクトリ構成へ依存しない形で管理します。

## 最初に見るファイル

- [STATUS.md](STATUS.md): 現在地と、直近で行うこと
- [ROADMAP.md](ROADMAP.md): 研究完了から投稿までのチェックリスト
- [docs/research-question.md](docs/research-question.md): 研究目的、仮説、貢献
- [docs/prior-work-experiments.md](docs/prior-work-experiments.md): 先行研究の実験設計
- [docs/experiment-plan.md](docs/experiment-plan.md): 比較対象、データセット、評価指標
- [docs/measurement-policy.md](docs/measurement-policy.md): 時間とstressの計測・報告方法
- [docs/experiment-output-format.md](docs/experiment-output-format.md): 1 runのJSON出力とfield定義
- [docs/experiment-runbook.md](docs/experiment-runbook.md): manifestによる自動実行・再開・検証手順
- [docs/experiment-log.md](docs/experiment-log.md): 実験結果と再現情報
- [docs/decisions.md](docs/decisions.md): 研究・執筆上の意思決定
- [manuscript/outline.md](manuscript/outline.md): 論文の章立てと本文の下書き

## ディレクトリ構成

```text
paper/
├── README.md
├── STATUS.md
├── ROADMAP.md
├── docs/                 # 研究計画、実験記録、意思決定
├── manuscript/           # 論文本文
├── figures/              # 論文で採用する図と、その生成情報
├── tables/               # 論文で採用する表と、その元データ
└── references/           # BibTeX と文献メモ
```

`output/` の生ログや `data/` の大規模データセットはここへ丸ごとコピーしません。論文に採用する成果物、集計済みデータ、再現に必要なメタデータだけを保存します。

## 日々の使い方

1. 作業開始時に `STATUS.md` の「直近の作業」を 1〜3 個に絞る。
2. 実験前に `docs/experiment-plan.md` へ条件を固定する。
3. 実験後すぐ `docs/experiment-log.md` に commit、コマンド、環境、結果を記録する。
4. 結論に影響する判断を `docs/decisions.md` に残す。
5. 確定した図表だけを `figures/` と `tables/` へ移す。
6. 得られた根拠を待たずに、書ける節から `manuscript/outline.md` に書く。

## 独立リポジトリへの切り出し

このフォルダの履歴を保持して別リポジトリへ移す場合は、現在のリポジトリのルートで次を実行します。

```bash
git subtree split --prefix=paper -b paper-only
git remote add paper-origin <論文リポジトリのURL>
git push paper-origin paper-only:main
```

新しいリポジトリでは、この `paper/` の中身がルートになります。実行前に作業ツリーがクリーンであることと、送信先 URL を確認してください。
