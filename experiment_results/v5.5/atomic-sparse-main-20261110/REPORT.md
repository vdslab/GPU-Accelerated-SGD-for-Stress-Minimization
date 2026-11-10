# Atomic Sparse 最新main 速度・Stress（web-Stanfordまで）

2026-11-10T20:23:34+09:00。成功780/780、52グラフ×5 seed×3回、各グラフseed999 warmup除外。15反復、epsilon0.1、p=min(200,n)。

commit: `839346091717f3928be83bbcdba8cf8783a48e4e`。GPU既定更新順、CPU shuffle無効。GPU timestamp無効。n<=8000厳密Full Stress、超過は256始点・評価seed0の推定Full Stress。速度とStressは同一最終座標。座標は検証済み可逆圧縮。

各runで対応CPU比を計算し、seed内3回中央値→5seed中央値・最小最大。グラフ間は等重み幾何平均。CPUは過去セッションの正式測定。ユーザー指定でweb-Stanfordまでの52グラフで終了し、com-DBLP・web-NotreDame・com-Amazon・coPapersCiteseerは今回未測定。温度・電源・常駐アプリは統制していない。

SGD+readback速度比: 0.603726、総計算速度比: 0.787151、Stress比: 1.005524。

SGD単独のnative値も保存するが、GPU完了待ちを含む主指標はsgd_time_ms+readback_time_ms。総計算はCPU最終座標完成まで、Stress評価・保存・描画は含まない。比較対象は過去CPUであり同時測定ではない。
