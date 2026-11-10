# 論文形式：v5とv5.5のAtomic Full比較

論文図6のFullパネルと同じ38グラフを、同じ辺数昇順・グラフ名順で表示する。対数縦軸、線で結ばないマーカー、5 seed中央値の最小〜最大、配色・寸法・縦向きラベルを揃えた。橙△はv5、青○はv5.5。各点はseed内3反復中央値→5seed中央値。

## 速度

![速度比較](../figures/paper-style-v5-v55-speed.png)

上段は `sgd_time_ms`、15反復のiteration合計時間。読み戻しを足さず、15でも割らない。新mainは非同期submit後のGPU完了待ちがreadback側にあるため、上段の値はその待機を含まず、GPUの全計算時間や両版の同一範囲の実時間とは解釈しない。下段のtotal_compute_time_msは、CPUの全pairハッシュ・shuffleとGPU完了待ちを含む入力からCPU上の座標が揃うまでの時間。Stress評価・座標保存・描画は除外されるので、描画の直前までの全処理時間ではない。

[速度PDF](../figures/paper-style-v5-v55-speed.pdf) / [速度SVG](../figures/paper-style-v5-v55-speed.svg) / [総計算のみPDF](../figures/paper-style-v5-v55-total.pdf)

## Stress

![Stress比較](../figures/paper-style-v5-v55-stress.png)

8,000頂点以下は厳密Full Stress、それ以外は固定256始点・評価seed0。全38グラフを表示し、品質が悪化した対象や速度が遅い対象も省略しない。v5は過去測定との比較で、whiskerは信頼区間ではない。小差から統計的有意差を判断しない。

[Stress PDF](../figures/paper-style-v5-v55-stress.pdf) / [全グラフの集計と考察](results.md) / [描画数値CSV](../tables/paper-style-chart-points.csv)
