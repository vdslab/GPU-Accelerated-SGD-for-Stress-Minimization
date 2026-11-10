# v5.5 Atomic Full 全38グラフ

570/570成功。[速度・Stressの結果](reports/results.md) · [38グラフ集計](tables/summary.csv) · [seed対応比較](tables/paired-seeds.csv) · [全測定](tables/runs.csv)

`python3 experiments/run_v55_atomic_full.py`で再開。完了条件を重複測定しない。主速度指標はGPU完了待ちを含む総計算時間。予備実験は別保存し流用しない。EX5は中断したまま。

[論文形式のv5・v5.5比較チャート](reports/paper-style-comparison.md)（辺数順・対数軸・マーカーと5seed範囲、PNG/PDF/SVG）。

[CPU・Atomic・RRのFull Complete比較](reports/full-complete-methods.md)：v5.5で使用する3方式、全38グラフ。CPUは過去v5参照、Atomic/RRはv5.5測定。
