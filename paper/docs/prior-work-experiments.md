# 先行研究の実験設計

対象文献:

> Jonathan X. Zheng, Samraat Pawar, Dan F. M. Goodman, “Graph Drawing by Stochastic Gradient Descent.”

確認したPDF: `/Users/kose/master/paper/Graph Drawing by Stochastic Gradient Descent.pdf`（全12ページ）

## 先行研究が主張したこと

- SGDはmajorizationより少ない反復で低いstressへ到達し、初期配置に対して結果が安定している。
- 15 iterationsの固定スケジュールでも改善の大部分が得られる。
- Sparse Stress近似へSGDを直接適用することで、10万頂点規模まで適用できる。

本研究はSGDとmajorizationの優劣を再検証するのではなく、先行研究のSGDをCPU基準として、GPU並列化後もstressを保ちながら高速化できるかを検証する。

## 共通条件

| 項目 | 先行研究の条件 | 出典 |
|---|---|---|
| 初期配置 | 1×1正方形内の一様乱数 | p.3、Algorithm 1 |
| グラフ | 原則として無重み | p.7、Figure 10 |
| データ源 | SuiteSparse Matrix Collectionの対称疎行列 | pp.3, 5 |
| 重み | `w_ij = d_ij^-2` | p.1 |
| 固定反復 | 15 iterations | pp.3-4 |
| 固定反復のepsilon | 0.1 | pp.3-4 |
| 学習率 | `eta_max = 1/w_min` から `eta_min = epsilon/w_max` への指数減衰 | p.3 |
| 順序 | random reshuffling（毎iterationで制約順をshuffle） | pp.4-5 |
| 実装環境 | C# / Visual Studio、Intel Core i7-4790、RAM 16GB | p.3 |

先行研究では、全制約を1回ずつ処理することを1 iteration、1制約の更新を1 stepと定義している。

## 実験量の全体像

| 実験 | グラフ数 | 手法・条件 | 反復run | 概算algorithm runs |
|---|---:|---|---:|---:|
| Full Stress品質 | 243 | SGD / majorization、15 iterations / convergence | 25 | 1 scheduleあたり12,150、2 schedulesなら24,300 |
| ランダム化方式 | 5 | 5種類の順序 | 50 | 1,250 |
| 詳細時間 | 8 | SGD + majorization 3実装 | 10 | 320 |
| Sparse pivot感度 | 3 | 10/50/200/full、SGD / majorization | 25 | 最大600 |
| 大規模Sparse例 | 6 | 200 pivots | 記載なし | 記載なし |

Full Stress品質の2 schedulesが完全に独立したrunか、途中値を共有したかは本文だけでは断定できないため、概算を分けて示す。いずれにしても、先行研究は少数の代表例だけでなく、243グラフと複数初期配置で品質の安定性を確認している。

## 1. Annealing scheduleの選定

- スケジュール形状の比較は複数グラフの平均最終stressで評価。
- Figure 4では、代表グラフ上で各スケジュールを25 runs、通常の表示は15 iterations、長時間挙動の近似は500 iterationsで比較。
- Figure 5では、Section 3の全グラフを対象に、`t_max` と `epsilon` を変えて25 runsの平均stressを比較。
- その結果、固定時間向けに15 iterations、epsilon 0.1、指数減衰を採用。
- 収束まで実行する場合は、初期に指数減衰、その後 `1/t` 型へ切り替える混合スケジュールとし、`max ||Delta X|| < 0.03` を停止条件に採用。局所解回避のため初期側を30 iterationsへ延長。

本研究の主比較ではGPU並列方式以外の差を作らないため、先行研究の固定15 iterations・epsilon 0.1を全手法で共有する。

## 2. 制約順のランダム化

- 約1000頂点の5グラフを選び、異なる初期配置から50 runsを実施。
- indexだけをshuffle、復元抽出、1つのshuffle順を固定、2つのshuffle順を交互利用、毎iteration reshuffleの5方式を比較。
- 毎iterationのrandom reshufflingが最良品質だった一方、順序どおりの処理に比べ1 iterationが最大60%遅くなった。

この結果は、本研究のround-robin方式で「順序変更によるstress差」と「dispatch/同期コスト」を別々に測る必要があることを示す。

## 3. Full Stress SGDの品質比較

- SuiteSparseの対称行列のうち、1000頂点以下の全243グラフを使用。
- SGDとmajorizationを、各グラフ25 runsで比較。
- 15 iterations後と収束後の2条件を評価。
- 各runは1×1正方形内のランダム初期配置から開始。
- stressは、両手法の全runで得た最小値により正規化。
- 平均、最小、最大、および変動係数を報告。
- 243グラフ中、majorizationのほうが低stressだったのは `dwt_307` の1例。
- 収束までの平均iterationsは、majorization 237、SGD 106。

先行研究と同じ243グラフの完全再実験は強い再現性評価になるが、本研究の主題はGPU高速化であるため、まず代表グラフでpaired comparisonを行い、必要に応じて対象を拡張する。

## 4. Full Stress SGDの時間比較

- 代表8グラフを使用:
  `btree9`、`qh882`、`1138_bus`、`dwt_1005`、`poli`、`dwt_2680`、`USPowerGrid`、`3elt`。
- SGDとmajorization 3実装（localized、conjugate gradient、Cholesky）を比較。
- 各グラフ10 runs。
- SGDは15-iteration schedule。
- stress対実時間の軌跡を描き、線はiterationごとのstressと時間の平均を通す。
- 初期化時間を含めて比較し、初回iterationの初期化コストも議論。

本研究でも、最終時間だけでなく「wall-clock timeに対するfull stressの収束曲線」を主要図とする。

## 5. Sparse SGDのpivot数と品質

- `USPowerGrid`、`EVA`、`3elt` の3グラフを使用。
- pivot数を10、50、200、full stressで比較。
- 各条件25 runs。
- SGDは15 iterations、majorizationは100 iterations。
- 各pivot数についてstressの平均と範囲、および最小stressのレイアウトを表示。
- `EVA` は低diameter・高degreeでSparse近似が難しく、`3elt` はmesh状で近似しやすいというトポロジー差を議論。
- PivotMDSによる初期化は使わず、他実験と同じランダム初期配置を使用。

本研究のSparse系主比較は200 pivotsで行い、10/50/200 pivotsの感度分析を別実験とする。

## 6. 大規模Sparse SGD

先行研究は200 pivots、15 iterationsで次の6グラフを描画した。

| グラフ | 頂点数 |
|---|---:|
| `pesa` | 11,738 |
| `bcsstk31` | 35,588 |
| `commanche_dual` | 7,920 |
| `finance256` | 37,376 |
| `bcsstk32` | 44,609 |
| `luxembourg_osm` | 114,599 |

ここではレイアウト例が中心で、Figure 14のような反復比較は報告されていない。本研究では大規模グラフを速度・メモリ・完走可否の評価に使い、stress品質は計算可能な範囲または共通の評価器で報告する。

## 本研究へ引き継ぐ条件

- 15 iterations、epsilon 0.1、同一seed、同一初期配置。
- 品質評価は25 seedsを基本とし、最低でも平均・最小・最大を先行研究に合わせて報告する。
- 時間評価は最低10 runsとし、median/IQRも追加する。
- `USPowerGrid`、200 pivots、25 seedsをSparse SGD再現の基準ケースにする。
- 大規模ケースは`luxembourg_osm`を先行研究最大規模の再現、`web-Stanford`を先行研究を超える規模の評価として使う。
- CPU/GPU間では同じfull stress評価器を使用し、実装固有の近似stressだけで品質を判定しない。
