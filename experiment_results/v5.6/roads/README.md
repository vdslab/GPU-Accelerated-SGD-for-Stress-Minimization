# v5.6 Italy / Germany

[結果](REPORT.md) · [集計JSON](tables/summary.json) · [検証](validation/results.json)

p50、seed0〜4各1回、15反復、epsilon0.1、CPU前処理、W128。正式10件すべて成功、warmup2件除外。最終Stressは固定64始点・評価seed0。autoは公開対象外。今回の公開は完了済みp50のスナップショットで、p100/p200を完了したものではありません。

数値・native JSON・コマンド・標準エラー・資源ログ・可逆圧縮manifest・再現用ソースを保存。大容量座標の圧縮ファイルと実行バイナリは既存の.gitignore方針に従いGitへ含めず、ローカルのexperiment_results/v5.6/largeに保持。raw内の実行時絶対パスは証跡としてそのまま保存。
