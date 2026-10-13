# v2結果の保存範囲

Gitには実験条件、測定JSONL、各実行のJSON・コマンド・ログ、実行時ソースのスナップショット、CSV、図（PNG／SVG／PDF）、レポート、検証結果を保存する。

初期／最終座標と頂点対応表（`*.initial.txt`、`*.final.txt`、`*.vertex-map.txt`）は合計約3.8GBのローカル生成物として保持し、Gitには含めない。入力グラフも既存の`data/`の管理方針に従う。測定記録には入力・座標ファイルのhashと保存先を残している。

保存済みのCSV・図・レポートはそのまま閲覧できる。座標の一致検証や描画を含む再集計にはローカルの入力・座標ファイルが必要。再実験の手順は[実験概要](../../experiments/docs/README.md)を参照。

[正式結果](report/RESULTS.md) · [50／200 pivotsの時間差の確認](validation/timing-audit-p50-p200/README.md)
