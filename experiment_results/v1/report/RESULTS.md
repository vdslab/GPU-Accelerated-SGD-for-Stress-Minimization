# 実験結果 v1（初回評価）

正式実行：536件。成功 536件、失敗 0件。

## 比較条件
- 前処理を除いた15反復のhost実時間を主速度指標とする。同期・シャッフルは含む。
- 小・中規模10 seed、大規模3 seed。品質25 seedの正式判定に先立つ暫定結果。
- exact full stressと64始点のsampled stressを混ぜず、各グラフ内の対応seedで比較。
- 全手法へ同じ無向・自己ループなし・最大連結成分の入力を渡す。

## グラフ別の結果
| Graph | Method | SGD speedup | Full-stress ratio | Paired seeds |
|---|---|---:|---:|---:|
| dwt_419 | Atomic SGD | 0.03x | 1.004 | 10 |
| dwt_419 | RR-SGD | 0.53x | 1.010 | 10 |
| dwt_419 | RR-Sparse SGD | 0.06x | 0.999 | 10 |
| qh882 | Atomic SGD | 0.10x | 1.006 | 10 |
| qh882 | RR-SGD | 2.34x | 1.004 | 10 |
| qh882 | RR-Sparse SGD | 0.14x | 1.002 | 10 |
| 1138_bus | Atomic SGD | 0.14x | 1.115 | 10 |
| 1138_bus | RR-SGD | 1.71x | 1.028 | 10 |
| 1138_bus | RR-Sparse SGD | 0.19x | 1.005 | 10 |
| dwt_1005 | Atomic SGD | 0.10x | 1.012 | 10 |
| dwt_1005 | RR-SGD | 3.23x | 1.001 | 10 |
| dwt_1005 | RR-Sparse SGD | 0.15x | 1.002 | 10 |
| cage8 | Atomic SGD | 0.12x | 1.026 | 10 |
| cage8 | RR-SGD | 2.96x | 1.000 | 10 |
| cage8 | RR-Sparse SGD | 0.16x | 1.005 | 10 |
| poli | Atomic SGD | 0.69x | 1.131 | 10 |
| poli | RR-SGD | 4.66x | 1.039 | 10 |
| poli | RR-Sparse SGD | 0.43x | 1.018 | 10 |
| dwt_2680 | Atomic SGD | 0.79x | 1.006 | 10 |
| dwt_2680 | RR-SGD | 5.89x | 0.955 | 10 |
| dwt_2680 | RR-Sparse SGD | 0.52x | 1.013 | 10 |
| EVA | Atomic SGD | 2.40x | 1.186 | 10 |
| EVA | RR-SGD | 9.76x | 1.160 | 10 |
| EVA | RR-Sparse SGD | 0.45x | 1.004 | 10 |
| 3elt | Atomic SGD | 2.63x | 1.287 | 10 |
| 3elt | RR-SGD | 11.46x | 1.012 | 10 |
| 3elt | RR-Sparse SGD | 1.17x | 1.001 | 10 |
| USpowerGrid | Atomic SGD | 2.81x | 1.037 | 10 |
| USpowerGrid | RR-SGD | 12.36x | 1.019 | 10 |
| USpowerGrid | RR-Sparse SGD | 1.20x | 1.003 | 10 |
| G65 | RR-Sparse SGD | 2.54x | 1.012 | 3 |
| fe_4elt2 | RR-Sparse SGD | 3.44x | 1.017 | 3 |
| bcsstk31 | RR-Sparse SGD | 7.73x | 1.005 | 3 |
| venkat50 | RR-Sparse SGD | 14.96x | 1.002 | 3 |
| luxembourg_osm | RR-Sparse SGD | 19.84x | 1.002 | 3 |
| web-NotreDame | RR-Sparse SGD | 3.92x | 1.171 | 3 |

## 解釈と限界
- Atomic SGD：対応比較可能な10グラフ中、中央値で高速化は3グラフ。stress比の暫定95% CI上限が1.05以下は5グラフ。
- RR-SGD：対応比較可能な10グラフ中、中央値で高速化は9グラフ。stress比の暫定95% CI上限が1.05以下は8グラフ。
- RR-Sparse SGD：対応比較可能な16グラフ中、中央値で高速化は8グラフ。stress比の暫定95% CI上限が1.05以下は13グラフ。
- 全グラフをまとめた単一の速度向上率で代表させない。グラフの構造・規模に依存する。
- GPU実装はf32、CPUはf64。差には精度と更新順の両方が含まれる。
- 初期配置の一致はFull群・Sparse群のそれぞれの中で確認。両群は初期配置生成が異なるため、群をまたぐ対応seed比は使わない。
- web-NotreDameではRR-Sparseが10,721ラウンド、15反復で160,830 dispatchを使う。他の大規模グラフは約200ラウンドである。同期・投入コストが性能を制限する可能性があるが、device時間の詳細測定なしには原因を断定しない。
- この結果はSGD部分の性能であり、初回レイアウト生成全体の高速化率ではない。
- 大規模3 seedから信頼性の高い速度・品質の結論を出さない。
- 同一seedの時間は1回ずつ。将来は固定seedの繰返しで機器の時間変動とseed差を分離する。
- Sparse atomicが未実装のためSparseの競合方式同士の比較は未完成。
- 欠測や失敗は図でゼロにせず、failures.csvおよび元stderrに理由を残す。

## 検証
- 比較整合性エラー：0件。
- 検証詳細：../tables/validation.json。正式掲載可否はfalse（標本数不足）。

## 次に必要な実験
1. 小・中規模を25 seedへ拡張し、品質の非劣性を判定する。
2. 大規模を最低10 seedへ拡張し、特にweb-NotreDameの約17%のsampled stress悪化を詳しく確認する。
3. Sparse atomicを追加し、同じ制約でRRと比較する。
4. 代表3グラフのiterationごとのstress収束を別診断実行で記録する（固定15反復の学習率を維持）。

## 図の読み方
- 01：CPU基準のstress比。点は平均、誤差棒は対応seedのbootstrap 95% CI。
- 02：SGD部分の速度向上率。点は対応比の中央値、誤差棒はbootstrap 95% CI。
- 03：SGD時間。棒は中央値、誤差棒はIQR。
- 04：最終exact full stressの分布。
- 05：品質と速度の両立。右側ほど速く、下側ほどstressが低い。
- 06：Sparseの規模拡大。丸は10 seed、四角は3 seed。
- 07：CPU full-SGDのstress中央値に近いseedを全5手法で共有した代表配置。
- 08：RR-Sparseのラウンド数と速度向上率。構造によるラウンド増加を確認する。

## 論文の体裁との対応
- SGD論文Figure 7・14の品質分布、Figure 10の品質と時間を同時に示す考え方を参考にした。
- DRGraph Table 2・5のグラフ別比較とFigure 6の規模に応じた速度評価を参考にした。
- DRGraphは尺度最適化したnormalized stressを使う。本結果のraw full stressと掲載数値は直接比較しない。
- PNGは確認用、SVGは論文で編集・拡大するための版。収束曲線は未取得。
