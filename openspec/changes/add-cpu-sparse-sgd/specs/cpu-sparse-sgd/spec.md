## ADDED Requirements

### Requirement: 独立した CPU Sparse SGD ベースライン
システムは、Matrix Market 形式の連結な無向グラフを読み込み、GPU を使用せず論文 Algorithm 2 の Sparse SGD を実行できる独立した Rust crate `baseline-sparse-sgd-non-gpu` を提供しなければならない（MUST）。

#### Scenario: 正常な連結グラフを使用した実行
- **WHEN** 有効な Matrix Market 入力と実行パラメータを指定したとき
- **THEN** グラフを読み込み、CPU 処理だけで Sparse SGD を完了する

#### Scenario: 非連結グラフを拒否する
- **WHEN** 入力グラフが複数の連結成分を持つとき
- **THEN** 無限距離を用いた pivot 確率や制約を生成せず、連結グラフが必要であることを示すエラーを返す

### Requirement: 論文準拠の MaxMinRandomSP pivot 選択
システムは、最初の pivot を全頂点から一様ランダムに選択し、以降の各 pivot を、未選択頂点から選択済み pivot 集合までの最小最短路距離に比例した確率で重複なく選択しなければならない（MUST）。

#### Scenario: 距離比例で次の pivot を選択する
- **WHEN** 1個以上の pivot が選択済みで、未選択頂点ごとの最近傍 pivot 距離が得られているとき
- **THEN** 距離0の選択済み頂点を除外し、各未選択頂点の選択確率をその最近傍距離に比例させる

#### Scenario: 固定 seed で pivot 選択を再現する
- **WHEN** 同じグラフ、pivot 数、乱数 seed を2回指定したとき
- **THEN** 同じ順序の重複しない pivot 列を生成する

#### Scenario: 頂点数を超える pivot を要求する
- **WHEN** 要求された pivot 数が頂点数を超えるとき
- **THEN** pivot 数を頂点数に制限し、すべての頂点を最大1回だけ選択する

### Requirement: pivot 領域と補正係数
システムは、各頂点を最短路距離が最小となる pivot の領域 `R(p)` へ割り当て、各 pivot `p` と非近傍頂点 `i` に対して `s_ip = |{j ∈ R(p) : 2 d_pj ≤ d_pi}|` を計算しなければならない（MUST）。

#### Scenario: 頂点を最近傍 pivot へ割り当てる
- **WHEN** 全 pivot から全頂点への最短路距離が得られたとき
- **THEN** 各頂点を距離が最小の pivot 領域へ1回だけ割り当て、同距離の場合は pivot 選択順が早いものを使用する

#### Scenario: 領域補正係数を計算する
- **WHEN** pivot `p` の領域と頂点 `i` までの距離 `d_pi` が与えられたとき
- **THEN** `R(p)` 内で `2 d_pj ≤ d_pi` を満たす頂点数を `s_ip` とする

### Requirement: 論文準拠の方向別疎制約
システムは、各制約について目標距離 `d_ij` と方向別重み `w'_ij`、`w'_ji` を保持し、論文の疎ストレス近似を構築しなければならない（MUST）。

#### Scenario: 非近傍の頂点と pivot の制約を構築する
- **WHEN** pivot `p` と、その近傍でも自己でもない到達可能な頂点 `i` を処理するとき
- **THEN** `w'_ip = s_ip / d_ip²` とし、逆方向 `w'_pi` は別の pivot 関係から設定されない限り0とする

#### Scenario: 2つの pivot 間の制約を構築する
- **WHEN** 制約の両端点が pivot であり、グラフ辺ではないとき
- **THEN** `w'_pq` と `w'_qp` を各 pivot 領域から独立して計算し、異なる値を許容する

#### Scenario: グラフ辺の制約を構築する
- **WHEN** `{i,j}` がグラフ辺であるとき
- **THEN** 目標距離を1、方向別重みを `w'_ij = w'_ji = 1` とし、pivot 制約との重複を残さない

### Requirement: 非対称 Sparse SGD 座標更新
システムは、各制約で `r = ((||X_i-X_j||-d_ij)/(2||X_i-X_j||))(X_i-X_j)`、`μ_i=min(w'_ij η,1)`、`μ_j=min(w'_ji η,1)` を計算し、`X_i←X_i-μ_i r`、`X_j←X_j+μ_j r` と更新しなければならない（MUST）。

#### Scenario: 方向別重みが異なる制約を処理する
- **WHEN** `w'_ij` と `w'_ji` が異なる有限な非負値であるとき
- **THEN** 各端点をそれぞれの `μ` で更新し、変位量が非対称になることを許容する

#### Scenario: 一方向の重みが0である
- **WHEN** `w'_ij > 0` かつ `w'_ji = 0` であるとき
- **THEN** 頂点 `i` だけを更新し、頂点 `j` の座標を変更しない

#### Scenario: 一致する座標を処理する
- **WHEN** 制約の両端点が同一座標にあるとき
- **THEN** 0除算を回避し、更新後の全座標を有限値に保つ

### Requirement: 論文準拠のアニーリング
システムは、正の方向別重みの最小値と最大値から `η_max=1/w_min`、`η_min=epsilon/w_max` を求め、指定反復数で `η_max` から `η_min` へ指数減衰する学習率列を生成しなければならない（MUST）。

#### Scenario: Figure 14 の実行条件を準備する
- **WHEN** 反復数15、`epsilon=0.1` を指定したとき
- **THEN** 15個の学習率を生成し、最初を `η_max`、最後を `η_min` とする

#### Scenario: 制約順序をランダム化する
- **WHEN** 各アニーリング反復を開始するとき
- **THEN** すべての辺制約と頂点-pivot 制約を、指定 seed から得られる再現可能なランダム順序に並べ替えて1回ずつ処理する

### Requirement: 再現可能な実行設定と結果ファイル
実行ファイルは、入力パス、pivot 数、反復数、epsilon、乱数 seed を指定可能とし、初期・処理後レイアウトへ実験条件を記録しなければならない（MUST）。

#### Scenario: 同じ seed で実行する
- **WHEN** 同じ入力と全パラメータで2回実行したとき
- **THEN** pivot 列、初期座標、制約順序、最終座標が一致する

#### Scenario: 結果を保存する
- **WHEN** Sparse SGD が正常に完了したとき
- **THEN** `-0.txt` と `-1.txt` の結果へ dataset、pivot 数と列、反復数、epsilon、seed、pivot 選択方式、重み方式、辺、座標を記録する

### Requirement: USPowerGrid による論文再現検証
システムは、`USpowerGrid.mtx` を200 pivots、15 iterations、`epsilon=0.1`、一様な `[0,1)²` 初期配置で25個の固定 seed により実行し、`gpu_visualizer` と同じ全点対 stress を使って論文 Figure 14 に近い品質を確認しなければならない（MUST）。

#### Scenario: 25 runs の stress を集計する
- **WHEN** seed 0から24までの25 runs が完了したとき
- **THEN** 各 run の full stress と、その平均・最小・最大を記録する

#### Scenario: Figure 14 の stress に近いことを判定する
- **WHEN** 25 runs の平均 full stress を算出したとき
- **THEN** Figure 14 から読み取った200-pivot SGD の基準値 `7.2×10⁵` に対する相対誤差を20%以下とする

#### Scenario: gpu_visualizer で代表 run を描画する
- **WHEN** 25 runs から基準値 `7.2×10⁵` に最も近い run を選択したとき
- **THEN** `gpu_visualizer` が同じ full stress を表示し、有限な座標を持つ PNG を生成する
