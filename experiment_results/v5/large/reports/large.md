# v5 超大規模グラフ実験・結果と考察

更新: 2026-11-07T13:49:22+09:00

14グラフ・5pivot・5seedの350条件を確定した。実測成功130、GPU上限の未実行220、本測定の実測失敗0。別に保全した再試行前の試行は1件。通常6030件とは別集計で、CPU/Atomicは測定していない。

| グラフ | p | 成功/5 | SGD15反復 s（中央値） | 総計算 s（中央値） | 推定Stress（中央値） | RSS GiB（中央値） |
|---|---:|---:|---:|---:|---:|---:|
| auto | 1000 | 5/5 | 2.023 | 62.11 | 7.41e+09 | 7.953 |
| auto | 500 | 5/5 | 0.9689 | 31.46 | 7.435e+09 | 4.755 |
| auto | 200 | 5/5 | 0.3815 | 13.55 | 7.896e+09 | 2.25 |
| auto | 100 | 5/5 | 0.2489 | 7.8 | 7.91e+09 | 1.399 |
| auto | 50 | 5/5 | 0.09717 | 4.762 | 8.037e+09 | 0.997 |
| NACA0015 | 1000 | 0/5 | — | — | — | — |
| NACA0015 | 500 | 5/5 | 2.173 | 61.65 | 1.463e+10 | 9.228 |
| NACA0015 | 200 | 5/5 | 0.8243 | 25.27 | 1.341e+10 | 4.504 |
| NACA0015 | 100 | 5/5 | 0.5116 | 13.66 | 1.34e+10 | 2.574 |
| NACA0015 | 50 | 5/5 | 0.1489 | 7.644 | 1.639e+10 | 1.609 |
| belgium_osm | 1000 | 0/5 | — | — | — | — |
| belgium_osm | 500 | 0/5 | — | — | — | — |
| belgium_osm | 200 | 5/5 | 1.117 | 28.52 | 2.89e+10 | 5.325 |
| belgium_osm | 100 | 5/5 | 0.6909 | 14.85 | 2.964e+10 | 3.11 |
| belgium_osm | 50 | 5/5 | 0.1731 | 7.69 | 3.028e+10 | 1.775 |
| Flan_1565 | 1000 | 0/5 | — | — | — | — |
| Flan_1565 | 500 | 0/5 | — | — | — | — |
| Flan_1565 | 200 | 5/5 | 1.595 | 85.59 | 6.637e+10 | 10.49 |
| Flan_1565 | 100 | 5/5 | 1.104 | 63.44 | 6.929e+10 | 8.466 |
| Flan_1565 | 50 | 5/5 | 0.5609 | 54.25 | 7.48e+10 | 7.573 |
| netherlands_osm | 1000 | 0/5 | — | — | — | — |
| netherlands_osm | 500 | 0/5 | — | — | — | — |
| netherlands_osm | 200 | 5/5 | 1.713 | 44.06 | 6.308e+10 | 7.221 |
| netherlands_osm | 100 | 5/5 | 1.027 | 22.72 | 6.617e+10 | 3.917 |
| netherlands_osm | 50 | 5/5 | 0.2377 | 11.85 | 6.712e+10 | 2.26 |
| com-Orkut | 1000 | 0/5 | — | — | — | — |
| com-Orkut | 500 | 0/5 | — | — | — | — |
| com-Orkut | 200 | 0/5 | — | — | — | — |
| com-Orkut | 100 | 5/5 | 3.523 | 232.8 | 8.186e+11 | 13.2 |
| com-Orkut | 50 | 5/5 | 2.391 | 180.6 | 9.087e+11 | 14.17 |
| AS365 | 1000 | 0/5 | — | — | — | — |
| AS365 | 500 | 0/5 | — | — | — | — |
| AS365 | 200 | 0/5 | — | — | — | — |
| AS365 | 100 | 5/5 | 1.883 | 53.8 | 4.686e+11 | 7.398 |
| AS365 | 50 | 5/5 | 0.5025 | 30.27 | 4.863e+11 | 4.674 |
| com-LiveJournal | 1000 | 0/5 | — | — | — | — |
| com-LiveJournal | 500 | 0/5 | — | — | — | — |
| com-LiveJournal | 200 | 0/5 | — | — | — | — |
| com-LiveJournal | 100 | 5/5 | 2.635 | 101.3 | 1.699e+12 | 9.746 |
| com-LiveJournal | 50 | 5/5 | 1.177 | 65.4 | 1.636e+12 | 6.703 |
| italy_osm | 1000 | 0/5 | — | — | — | — |
| italy_osm | 500 | 0/5 | — | — | — | — |
| italy_osm | 200 | 0/5 | — | — | — | — |
| italy_osm | 100 | 0/5 | — | — | — | — |
| italy_osm | 50 | 5/5 | 0.7073 | 35.71 | 9.26e+11 | 6.59 |
| great-britain_osm | 1000 | 0/5 | — | — | — | — |
| great-britain_osm | 500 | 0/5 | — | — | — | — |
| great-britain_osm | 200 | 0/5 | — | — | — | — |
| great-britain_osm | 100 | 0/5 | — | — | — | — |
| great-britain_osm | 50 | 5/5 | 0.8271 | 42.86 | 6.204e+11 | 7.639 |
| germany_osm | 1000 | 0/5 | — | — | — | — |
| germany_osm | 500 | 0/5 | — | — | — | — |
| germany_osm | 200 | 0/5 | — | — | — | — |
| germany_osm | 100 | 0/5 | — | — | — | — |
| germany_osm | 50 | 0/5 | — | — | — | — |
| asia_osm | 1000 | 0/5 | — | — | — | — |
| asia_osm | 500 | 0/5 | — | — | — | — |
| asia_osm | 200 | 0/5 | — | — | — | — |
| asia_osm | 100 | 0/5 | — | — | — | — |
| asia_osm | 50 | 0/5 | — | — | — | — |
| road_usa | 1000 | 0/5 | — | — | — | — |
| road_usa | 500 | 0/5 | — | — | — | — |
| road_usa | 200 | 0/5 | — | — | — | — |
| road_usa | 100 | 0/5 | — | — | — | — |
| road_usa | 50 | 0/5 | — | — | — | — |
| europe_osm | 1000 | 0/5 | — | — | — | — |
| europe_osm | 500 | 0/5 | — | — | — | — |
| europe_osm | 200 | 0/5 | — | — | — | — |
| europe_osm | 100 | 0/5 | — | — | — | — |
| europe_osm | 50 | 0/5 | — | — | — | — |

## 適用可能性

- auto: 5seed完了した最大pは1000。
- NACA0015: 5seed完了した最大pは500。
- belgium_osm: 5seed完了した最大pは200。
- Flan_1565: 5seed完了した最大pは200。
- netherlands_osm: 5seed完了した最大pは200。
- com-Orkut: 5seed完了した最大pは100。
- AS365: 5seed完了した最大pは100。
- com-LiveJournal: 5seed完了した最大pは100。
- italy_osm: 5seed完了した最大pは50。
- great-britain_osm: 5seed完了した最大pは50。
- germany_osm: 全pがGPU上限により未実行であり、実測の成功・失敗は評価していない。
- asia_osm: 全pがGPU上限により未実行であり、実測の成功・失敗は評価していない。
- road_usa: 全pがGPU上限により未実行であり、実測の成功・失敗は評価していない。
- europe_osm: 全pがGPU上限により未実行であり、実測の成功・失敗は評価していない。

今回5seed完了を確認した最大入力はgreat-britain_osm（7,733,822頂点）だった。これは現機材・実装・指定pの観測結果で、GPU一般の最大規模ではない。

GPU事前判定はone-sided配列8np bytesと実測照会したstorage binding上限4,294,967,295 bytes、頂点dispatch上限65,535 workgroups（256頂点/group）に基づく。p50でも千万頂点級は配列上限を超える。CPUを測っていないのでCPUで不可能・CPU比高速とは結論しない。

## 時間と品質の交換関係

- auto: p1000/p50の総計算時間中央値比13.045、Stress中央値比0.9221。測定範囲でStress中央値が最小だったのはp1000。初期配置・pivot選定も変わるため、pだけの因果効果や最適pの確定とは扱わない。
- NACA0015: p500/p50の総計算時間中央値比8.065、Stress中央値比0.8924。測定範囲でStress中央値が最小だったのはp100。初期配置・pivot選定も変わるため、pだけの因果効果や最適pの確定とは扱わない。
- belgium_osm: p200/p50の総計算時間中央値比3.709、Stress中央値比0.9545。測定範囲でStress中央値が最小だったのはp200。初期配置・pivot選定も変わるため、pだけの因果効果や最適pの確定とは扱わない。
- Flan_1565: p200/p50の総計算時間中央値比1.578、Stress中央値比0.8873。測定範囲でStress中央値が最小だったのはp200。初期配置・pivot選定も変わるため、pだけの因果効果や最適pの確定とは扱わない。
- netherlands_osm: p200/p50の総計算時間中央値比3.718、Stress中央値比0.9397。測定範囲でStress中央値が最小だったのはp200。初期配置・pivot選定も変わるため、pだけの因果効果や最適pの確定とは扱わない。
- com-Orkut: p100/p50の総計算時間中央値比1.289、Stress中央値比0.9009。測定範囲でStress中央値が最小だったのはp100。初期配置・pivot選定も変わるため、pだけの因果効果や最適pの確定とは扱わない。
- AS365: p100/p50の総計算時間中央値比1.778、Stress中央値比0.9636。測定範囲でStress中央値が最小だったのはp100。初期配置・pivot選定も変わるため、pだけの因果効果や最適pの確定とは扱わない。
- com-LiveJournal: p100/p50の総計算時間中央値比1.549、Stress中央値比1.0388。測定範囲でStress中央値が最小だったのはp50。初期配置・pivot選定も変わるため、pだけの因果効果や最適pの確定とは扱わない。
- italy_osm: p50のみ5seed成功した。ほかのpはGPU上限で未実行なので、p間の時間・品質の交換関係は評価できない。
- great-britain_osm: p50のみ5seed成功した。ほかのpはGPU上限で未実行なので、p間の時間・品質の交換関係は評価できない。

成功130実行でのSGD/総計算時間の比率は中央値2.42%、最小1.03%、最大8.18%だった。この全runを同じ重みで集約した記述量からも、SGD以外の入力・前処理・準備等を含む総計算時間を併記する必要がある。

上記の記述比は中央値同士の比較。図のStress比はgraph・seedごとに最大の5seed成功pへ対応付けてから5seedを集計する。SGD時間だけでなく、入力・前処理・GPU準備等を含むtotal_compute_time_msを見る。Stress評価・ファイル保存・圧縮・描画は総計算時間の外側であり、実際の待ち時間はprocess_wall_seconds＋archive_seconds＋shared_repack_seconds等も確認する。

## 批判的な確認と限界

- com-LiveJournal p100の総計算時間は中央値101.33秒、最大436.61秒だった。遅い実行も除外せずmin-maxへ含めた。実行環境・処理内容のどの要因による差かをこの測定だけで特定していない。
- com-OrkutのRSS中央値はp100で13.20GiB、p50で14.17GiBだった。配列の理論サイズとプロセスRSSは異なり、pを減らすと実測RSSも必ず単調に下がるとは結論できない。
- 保存処理中の管理プロセス中断から1件の正常なnative出力を復旧した。レイアウトを再実行せず、元のJSON・初期/最終座標・圧縮復元SHA256を確認した。子プロセス終了コードと終了直後のメモリは未保存のため欠測として残し、process_wall_secondsは/usr/bin/timeの0.01秒精度値を用いた。停止原因は特定できず、中断中の待ち時間はnative計算時間へ加算しない。証拠は各runのrecovery_evidenceを参照する。
- 初期座標fingerprint: 複数pが成功したgraph×seedの40組中40組で異なる初期座標を確認した。同seedだけで初期配置一致とみなさず、nativeの乱数消費とpivot選定を保持した結果として比較する。
- 各seed1回のため通常比較の各seed3回とは時間再現性の評価水準が異なる。5seedのmin-maxはレイアウトseedによる分布で、反復測定の信頼区間ではない。
- 64始点・評価seed0を固定した最終推定Full Stressを用いる。評価源ごとの寄与や推定標準誤差はnative出力にないため、5seedの幅をStress推定誤差と解釈せず小差を断定しない。通常比較は256始点なので絶対Stressの直接混合も避ける。
- p降順・グラフ順の逐次測定で温度・OSキャッシュ・swap等の順序効果が残る。Codex常駐のためOSのみの最小メモリ環境ではない。RSSはプロセス指標で、統合メモリ上のGPU割当を単独で厳密に表さない。
- 異なる国の道路網やSNS・メッシュは構造と辺数も異なる。同じグラフの頂点数だけを増やした純粋なスケーリング実験ではなく、SNSのスケールフリー性をここで実証したわけではない。
- ディスク容量停止を挟み、保存形式にはgzip・検証済み可逆XZを使い、com-Orkutは全byte同一の標準gzip連結memberをAPFSで物理共有して容量を節約した。共有対象は保存用の辺データで、GPU配列の容量条件は従来どおり。レイアウト・Stressのバイナリは不変で、保存・圧縮処理はnative計時の外。中断試行は本測定へ混ぜず別表に残した。Git objectの複製抑止と保全圧縮も測定停止中のみ実行した。
- レイアウト図はseed0・5seed成功した最大pを採用し、頂点/辺の決定的間引きと独立スケールを明示する。細部や密度の厳密比較に使わない。

## 図表と生データ

正本: `/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization/experiment_results/v5/large`。全350状態と全成功runのnative値はtables/runs.csv、各pの集計はsummary.csv、対応seedの品質比はpaired-pivot-ratios.csv、再試行前記録はretried-attempts.csv。

![large-completion](../figures/large-completion.png)

![large-p-sgd](../figures/large-p-sgd.png)

![large-p-total](../figures/large-p-total.png)

![large-p-rss](../figures/large-p-rss.png)

![large-p-stress-ratio](../figures/large-p-stress-ratio.png)

![large-p-stress](../figures/large-p-stress.png)

![large-scale-sgd](../figures/large-scale-sgd.png)

![large-scale-total](../figures/large-scale-total.png)

![large-scale-rss](../figures/large-scale-rss.png)

![large-layout-auto](../figures/large-layout-auto.png)

![large-layout-NACA0015](../figures/large-layout-NACA0015.png)

![large-layout-belgium_osm](../figures/large-layout-belgium_osm.png)

![large-layout-Flan_1565](../figures/large-layout-Flan_1565.png)

![large-layout-netherlands_osm](../figures/large-layout-netherlands_osm.png)

![large-layout-com-Orkut](../figures/large-layout-com-Orkut.png)

![large-layout-AS365](../figures/large-layout-AS365.png)

![large-layout-com-LiveJournal](../figures/large-layout-com-LiveJournal.png)

![large-layout-italy_osm](../figures/large-layout-italy_osm.png)

![large-layout-great-britain_osm](../figures/large-layout-great-britain_osm.png)
