## USPowerGrid 論文再現結果

### 実行条件

- Dataset: `data/USpowerGrid.mtx`
- Nodes: 4,941
- Edges: 6,594
- Pivots: 200
- Iterations: 15
- Epsilon: 0.1
- Seeds: 0〜24
- Pivot selection: `max-min-random-sp-distance-proportional`
- Weight model: `ortmann-region-directed-weight`
- Stress: `gpu_visualizer/src/stress.rs` と同じ全到達可能頂点対の full stress
- Figure 14 読み取り基準: `720,000`
- 合格範囲: `576,000`〜`864,000`（基準値の±20%）

### 25-run stress

| Seed | Full stress |
|---:|---:|
| 0 | 722,575.45 |
| 1 | 724,994.10 |
| 2 | 726,306.25 |
| 3 | 733,761.03 |
| 4 | 725,382.06 |
| 5 | 724,430.60 |
| 6 | 721,187.07 |
| 7 | 729,088.33 |
| 8 | 732,976.28 |
| 9 | 720,514.68 |
| 10 | 729,182.50 |
| 11 | 721,030.37 |
| 12 | 726,454.66 |
| 13 | 731,886.86 |
| 14 | 724,066.47 |
| 15 | 729,451.63 |
| 16 | 725,192.48 |
| 17 | 721,536.83 |
| 18 | 732,563.91 |
| 19 | 722,021.03 |
| 20 | 722,605.71 |
| 21 | 728,988.18 |
| 22 | 722,317.62 |
| 23 | 724,657.80 |
| 24 | 727,993.86 |

### 集計

- 平均: `726,046.63`
- 最小: `720,514.68`（seed 9）
- 最大: `733,761.03`（seed 3）
- 平均の論文基準値に対する相対誤差: `0.84%`
- 判定: 合格

### gpu_visualizer による代表 run の確認

論文基準値に最も近い seed 9 を2048×2048で描画しました。

- Result: `output/baseline-sparse-sgd-non-gpu-USpowerGrid-seed9-20260713_021153_173-1.txt`
- PNG: `output/baseline-sparse-sgd-non-gpu-USpowerGrid-seed9-20260713_021153_173-1.png`
- gpu_visualizer stress: `720,514.68`
- 座標数: 4,941
- NaN/Inf: なし
- PNG: 2048×2048 RGBA

25-run 平均と代表 run はともに仕様の許容範囲に入り、論文 Figure 14 の200-pivot SGD と近い full stress を再現しました。
