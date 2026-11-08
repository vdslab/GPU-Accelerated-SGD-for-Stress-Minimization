---
tags: [area/大学院研究, 実験結果]
updated: 2026-11-08
status: 測定完了
---

# EX2：RR128 速度・Stress実験の結果と考察

[[EX2_速度ストレス実験]] / [[9_計測時間]]

RR Full 38グラフ×5seed×3回＝570実行、RR Sparse 56グラフ×5seed×3回＝840実行。計1,410実行すべて成功。
FullはT=W=128、shared・内外shuffle・phase_chunks1・全件厳密Full Stress。SparseはW128、p=min(200,n)、phase_chunks1・従来のpivot順序、n≤8000は厳密、それ以上は256始点・評価seed0の推定Full Stress。
15反復・epsilon0.1。同seed3回中央値→5seed中央値・最小〜最大。9項目＋追加⑩を全件取得し、④は15反復分、⑨はその合計として検証した。

描画は2048×2048、全辺、ノード半径1px、blue。⑦は結果取得Fから最終配置の最初のGPU描画完了Gまでで、Stress評価・保存/読込・起動/準備を含む。⑧も同じ待ちを含む。⑦を純GPU描画時間と解釈せず、品質評価時間は別途保存した。PNGのエンコード・保存はG以後で除外した。

## 前回との速度比較

前回RR/今回RRを同じgraph・seed・repetitionで求めてから階層集計。1より大きい値は今回が高速。異なる時期の測定で、今回はGPU timestampと描画が有効なため、差の全てをW/T変更だけの効果に帰属しない。

| 方式 | グラフ数 | SGD速度比の幾何平均 | 最小〜最大（グラフ間） |
|---|---:|---:|---:|
| rr_sgd | 38 | 1.592 | 0.598〜3.406 |
| rr_sparse_sgd | 56 | 1.155 | 0.864〜1.810 |

## 解釈と制約

FullとSparseは目的関数・更新対象・初期配置が異なるため、速度差やStress差の全てをスケジューリングだけの効果とはしない。SparseのW128が他の全サイズより最速であることを示す探索実験ではない。
Fullは今回厳密Stress、前回は大きいグラフで推定Stressだった。評価方式を揃えない品質比を作らず、ここでは今回の絶対Stressを報告した。CPU・Atomicは今回再測定しておらず、新規測定の同時期の基準として扱わない。

## 図表

![[EX2_v5_図表/full-all-timings.png]]

![[EX2_v5_図表/full-previous-speedup.png]]

![[EX2_v5_図表/full-stress.png]]

![[EX2_v5_図表/full-speed-stress.png]]

![[EX2_v5_図表/sparse-all-timings.png]]

![[EX2_v5_図表/sparse-previous-speedup.png]]

![[EX2_v5_図表/sparse-stress.png]]

![[EX2_v5_図表/sparse-speed-stress.png]]

## 保存先

`experiment_results/v5/ex2-rr128-speed-stress-20261108/`

- raw/：各実行のコマンド・native出力・rendererログ・記録・復元検証済み座標。
- tables/runs.csv、hierarchical.json、summary.csv：生データ・階層集計。
- validation/preflight-complete.json：timestamp有無の座標一致、Sparse W256/W128の一致、代表条件での10項目検証。
- validation/timing-coverage.json：全記録の計時取得数。
- figures/：チャートPNG・PDF。source-snapshot/・bin/・config/：再現情報。

前回のv5・EX1記録は保持し、開始前と完了後にハッシュの不変を確認した。

## 総評（2026-11-08）

今回の実測上、RR FullのT=W128を共通設定として採用する判断は妥当。RR SparseのW128も、従来W256に対して品質値を維持した速度改善が観測された。ただし全入力での最速設定や、W/T変更だけによる因果効果を確定した実験ではない。

| 方式 | 前回比SGD速度（幾何平均） | 前回比描画なし総計算速度（幾何平均） | SGDが速かったグラフ |
|---|---:|---:|---:|
| RR Full（前回T=W1024） | 1.592倍 | 1.195倍 | 31/38 |
| RR Sparse（前回W256） | 1.155倍 | 1.116倍 | 45/56 |

比は対応するgraph/seed/repetitionごとに求め、seed内3回中央値→5seed中央値とした後、グラフ間を同じ重みで幾何平均した記述値。統計的有意差の検定ではない。FullとSparseの対象集合は異なるため、この集計値同士を方式間の直接速度比較には使わない。

FullのSGD改善は特にfe_4elt2（3.406倍）、bcsstk17（3.340倍）、PGPgiantcompo（3.292倍）で大きい。一方、grid17（0.598倍）など遅くなった例もある。共通設定128は全グラフで最速という意味ではない。

品質は、Sparseでは全840実行で前回の同条件のStress値と完全一致した。Fullは厳密値同士で比較できる34グラフ・510実行を比較し、今回/前回Stress比の幾何平均は0.9944だったが、1%以上改善したのは4グラフ、1%以上悪化したのは10グラフ。例えばwiki-Voteは約23.4%低下、emailは約7.9%増加した。1%は変化量を数えるための記述的区切りであり、品質の合否基準ではない。PGPgiantcompo・bcsstk17・ca-HepTh・fe_4elt2は前回が推定、今回が厳密なので、この品質比の集計から除外した。品質が一律に改善したとは言わない。

前処理時間/総計算時間のグラフ間中央値はFull約47%、Sparse約62%。SGDの高速化が総時間では小さく見えることと整合し、今後は前処理も改善候補になる。ただし個別区間の合計と総経過時間は同じ定義ではなく、厳密な加算分解としては扱わない。

前回と測定時期が異なり、今回だけGPU timestampが有効。予備確認では小さいSparse入力で計測有効時の時間増加も観測した。したがって、上の速度差は実測された新旧構成の差であり、すべてをW/T変更単独の効果には帰属しない。論文ではFullの速度改善と品質のグラフ依存性、SparseのStress一致を併せて報告する。
