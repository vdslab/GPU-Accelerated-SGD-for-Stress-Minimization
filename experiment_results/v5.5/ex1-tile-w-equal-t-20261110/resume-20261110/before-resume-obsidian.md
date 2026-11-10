# EX1 v5.5：タイル実験（W=T）

RR Fullのみ新規1,650実行。前回のCPU150件は厳密Stress・参考速度の基準として参照し、前回データは変更しない。

10グラフ：email、add20、CA-GrQc、p2p-Gnutella08、PGPgiantcompo、dwt_1005、dwt_2680、3elt、bcspwr10、fe_4elt2。T=W=1,2,4,8,16,32,64,128,256,512,1024。seed10〜14×各3回、15反復、epsilon0.1、shared、内外shuffle有効、phase_chunks1、auto BFS。全グラフ厳密Full Stress（速度計時後）、GPU timestamp無効。graph/Tごとのseed999ウォームアップを除外し、逐次測定。時間打切りなし。次の実行前に空きディスク4GiB・利用可能RAM2GiBを確認。

## 実行状況

2026-11-10T12:04:22+09:00 状態=running、成功741/1650、失敗0、残り909。実行中：v55-ex1-PGPgiantcompo-rr_sgd-t16-s12-r1（4秒）。EX2は未開始。

正本：`/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization/experiment_results/v5.5/ex1-tile-w-equal-t-20261110`。結果・図表：[[EX1_v5.5_タイル実験(W=T)_結果と考察]]。同seed3回中央値→5seedの中央値・最小〜最大で集計。CPU比は過去測定との参考比較と明記。旧EX1のW=Tと今回を対応比較する。RR SparseはEX2で実施予定。

## 停止・EX2への切替

ユーザー指示によりEX1を742/1650成功・失敗0で停止。完了データと中断実行の証跡は保持するが、EX2の最終集計へ混ぜない。以後は[[EX2_v5.5_速度ストレス実験]]を実施。
