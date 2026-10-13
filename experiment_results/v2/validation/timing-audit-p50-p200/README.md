# Sparse RRの過去ログとv2の時間差の確認

2026-10-13。v2と同じreleaseバイナリ・同じ入力・seed=0・15反復・epsilon=0.1で、pivot数だけを50／200に変えて再測定した。正式2260件のデータは変更していない。

| Graph | pivots | constraints | rounds | dispatches / iteration | compute全15反復 ms | ms / iteration |
|---|---:|---:|---:|---:|---:|---:|
| USpowerGrid | 50 | 252239 | 57 | 58 | 56.577000 | 3.771800 |
| USpowerGrid | 200 | 974214 | 207 | 208 | 157.637792 | 10.509186 |
| luxembourg_osm | 50 | 5848237 | 52 | 53 | 86.678875 | 5.778592 |
| luxembourg_osm | 200 | 23018952 | 203 | 204 | 301.649250 | 20.109950 |

貼られた過去ログは50 pivotsの実行と整合する。USpowerGridの制約数244420+7819=252239、rounds=57、dispatches=58／反復、Luxembourgの制約数5727346+120891=5848237、rounds=52、dispatches=53／反復が再測定と一致。

過去ログの計算は正しい：63.802959÷15=4.2535306 ms／反復、116.798041÷15=7.7865361 ms／反復。今回の再測定も、全15反復を囲むhostタイマーから算出した。GPU投入後の完了待ちが時間区間内にある。前処理・スケジュール作成・アップロード・読み戻しはcomputeから除外。

v2は200 pivotsで、更新対象の制約数が約4倍、ラウンド数も増える。今回の同一seedの時間差は約2.79倍／3.48倍。50 pivotsの高速な結果と200 pivotsの品質・速度を同じ条件として比較しない。

前の回答のUSpowerGrid約25ms／反復はFull RR。貼られた約4ms／反復はSparse RRで、方式も異なる。Sparse RRのv2中央値はUSpowerGrid約11.78ms／反復、Luxembourg約20.58ms／反復。

この診断は各条件1回の確認実行で、中央値や信頼区間の再評価ではない。CPUに対する高速化を50 pivotsで示すにはCPU側も同じpivot数・seed・入力に揃える。

バイナリSHA256：`6437a788babc4667069980afdaaf7624fe211e25004c46f64d76ce290ac4db77`。正式v2と現在のreleaseバイナリのSHAは一致し、GPU・schedule・shader・mainの保存ソースも一致した。

[診断結果JSON](comparison.json)に測定値、各サブディレクトリにコマンド・stdout・stderr・座標を保存。
