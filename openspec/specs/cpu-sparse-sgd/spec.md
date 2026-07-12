# CPU Sparse SGD

## Purpose

論文Algorithm 2に準拠するCPU Sparse SGDを、再現可能な基準実装として定義する。

## Requirements

### Requirement: 独立したCPU Sparse SGDベースライン

システムは、Matrix Market形式の連結な無向グラフを読み込み、GPUを使用せず論文Algorithm 2のSparse SGDを実行できる独立したRust crate `baseline-sparse-sgd-non-gpu` を提供しなければならない（MUST）。

#### Scenario: 正常な連結グラフを使用した実行

- **WHEN** 有効なMatrix Market入力と実行パラメータを指定したとき
- **THEN** グラフを読み込み、CPU処理だけでSparse SGDを完了する

### Requirement: 論文準拠のMaxMinRandomSP pivot選択

システムは、最初のpivotを全頂点から一様ランダムに選択し、以降の各pivotを未選択頂点から選択済みpivot集合までの最小最短路距離に比例した確率で重複なく選択しなければならない（MUST）。

#### Scenario: 距離比例で次のpivotを選択する

- **WHEN** 1個以上のpivotが選択済みで、未選択頂点ごとの最近傍pivot距離が得られているとき
- **THEN** 距離0の選択済み頂点を除外し、各未選択頂点の選択確率をその最近傍距離に比例させる

#### Scenario: 固定seedでpivot選択を再現する

- **WHEN** 同じグラフ、pivot数、乱数seedを2回指定したとき
- **THEN** 同じ順序の重複しないpivot列を生成する

#### Scenario: 頂点数を超えるpivotを要求する

- **WHEN** 要求されたpivot数が頂点数を超えるとき
- **THEN** pivot数を頂点数に制限し、すべての頂点を最大1回だけ選択する

### Requirement: pivot領域と補正係数

システムは、各頂点を最短路距離が最小となるpivotの領域`R(p)`へ割り当て、各pivot `p` と非近傍頂点 `i` に対して `s_ip = |{j ∈ R(p) : 2 d_pj ≤ d_pi}|` を計算しなければならない（MUST）。

#### Scenario: 頂点を最近傍pivotへ割り当てる

- **WHEN** 全pivotから全頂点への最短路距離が得られたとき
- **THEN** 各頂点を距離が最小のpivot領域へ1回だけ割り当て、同距離の場合はpivot選択順が早いものを使用する

#### Scenario: 領域補正係数を計算する

- **WHEN** pivot `p` の領域と頂点 `i` までの距離 `d_pi` が与えられたとき
- **THEN** `R(p)` 内で `2 d_pj ≤ d_pi` を満たす頂点数を `s_ip` とする

### Requirement: 方向別Sparse SGD制約と座標更新

システムは、各制約について目標距離 `d_ij` と方向別重み `w'_ij`、`w'_ji` を保持し、`r = ((||X_i-X_j||-d_ij)/(2||X_i-X_j||))(X_i-X_j)`、`μ_i=min(w'_ij η,1)`、`μ_j=min(w'_ji η,1)` により座標を更新しなければならない（MUST）。

#### Scenario: 非近傍の頂点とpivotの制約を構築する

- **WHEN** pivot `p` と、その近傍でも自己でもない到達可能な頂点 `i` を処理するとき
- **THEN** `w'_ip = s_ip / d_ip²` とし、逆方向 `w'_pi` は別のpivot関係から設定されない限り0とする

#### Scenario: グラフ辺の制約を構築する

- **WHEN** `{i,j}` がグラフ辺であるとき
- **THEN** 目標距離を1、方向別重みを `w'_ij = w'_ji = 1` とし、pivot制約との重複を残さない

#### Scenario: 一方向の重みが0である

- **WHEN** `w'_ij > 0` かつ `w'_ji = 0` であるとき
- **THEN** 頂点`i`だけを更新し、頂点`j`の座標を変更しない

#### Scenario: 一致する座標を処理する

- **WHEN** 制約の両端点が同一座標にあるとき
- **THEN** 0除算を回避し、更新後の全座標を有限値に保つ

### Requirement: 論文準拠のアニーリング

システムは、正の方向別重みの最小値と最大値から `η_max=1/w_min`、`η_min=epsilon/w_max` を求め、指定反復数で`η_max`から`η_min`へ指数減衰する学習率列を生成しなければならない（MUST）。

#### Scenario: 15iterationの実行条件を準備する

- **WHEN** 反復数15、`epsilon=0.1`を指定したとき
- **THEN** 15個の学習率を生成し、最初を`η_max`、最後を`η_min`とする

#### Scenario: 制約順序をランダム化する

- **WHEN** 各アニーリング反復を開始するとき
- **THEN** すべての制約を、指定seedから得られる再現可能なランダム順序に並べ替えて1回ずつ処理する

### Requirement: 再現可能な実行設定と結果ファイル

実行ファイルは、入力パス、pivot数、反復数、epsilon、乱数seedを指定可能とし、初期・処理後レイアウトへ実験条件を記録しなければならない（MUST）。

#### Scenario: 同じseedで実行する

- **WHEN** 同じ入力と全パラメータで2回実行したとき
- **THEN** pivot列、初期座標、制約順序、最終座標が一致する

#### Scenario: 結果を保存する

- **WHEN** Sparse SGDが正常に完了したとき
- **THEN** `-0.txt` と `-1.txt` の結果へdataset、pivot数と列、反復数、epsilon、seed、pivot選択方式、重み方式、辺、座標を記録する

### Requirement: USPowerGridによる論文再現検証

システムは、`USpowerGrid.mtx` を200 pivots、15 iterations、`epsilon=0.1`、一様な`[0,1)²`初期配置で固定seedにより実行し、全点対stressで論文Figure 14に近い品質を確認しなければならない（MUST）。

#### Scenario: Figure 14のstressに近いことを判定する

- **WHEN** 固定seed群の平均full stressを算出したとき
- **THEN** Figure 14から読み取った200-pivot SGDの基準値`7.2×10⁵`に対する相対誤差を20%以下とする
