# v5.5 EX2：coPapersCiteseer容量拡張補測

当初の1,395件と15件の容量未実行記録は変更せず保持。4×256 MiB（計1 GiB）への拡張後に、coPapersCiteseer / p=200 / W128 / 15反復 / seed0〜4×3回を補測し、15/15成功した。epsilon0.1、auto BFS、timestamp有効、256固定始点・評価seed0の推定Full Stress、描画条件も元EX2と同じ。warmup seed999と確認実行は集計外。

全15件の初期座標・前処理fingerprint・保存最終座標・Stressが前回EX2と一致。小規模dwt_1005でも拡張前後の一致を確認。最新main cafb96704へ反映したビルドでも、40テストと小規模・coPapersCiteseer実行の一致確認が成功した。元1,395件は拡張前のバイナリ、補測15件は容量拡張後のバイナリであり、source/binary hashを区別して保持する。

coPapersCiteseerの階層集計（各seed3回中央値→5seed中央値）：SGD 468.493 ms、総計算 23833.755 ms。前回/今回の対応速度比：SGD 0.9929、総計算 1.2328。Stress比 1.0000。異なる日時の比較で、温度・常駐アプリ等は専用固定していない。

- [補測を含むEX2全体の結果・図表](reports/results.md)
- [最新mainの確認](validation/latest-main.json)
- [40テストのログ](logs/latest-main-tests.log)
- [元EX2の記録](../ex2-rr128-speed-stress-20261110/README.md)

統合ビューはRR Full570件＋RR Sparse840件＝1,410件。CPU・Atomicの過去2,820件と合わせ4,230件。過去記録は変更せず、元のnot_run_capacityをsupersedes_run_idで参照する追加記録とした。

実行：`python3 experiments/run_v55_copapers.py`。集計：`python3 experiments/report_v55_copapers.py`。最新mainの確認：`python3 experiments/verify_v55_main.py --source-root /path/to/main --binary /path/to/main/sparse-sgd-gpu`。凍結バイナリと大容量座標はローカル保持し、既存.gitignoreの方針どおりGitには測定JSON・コマンド・ログ・hash・source snapshot・集計・PNG/PDFを保存する。
