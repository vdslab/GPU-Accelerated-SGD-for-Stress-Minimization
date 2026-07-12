## 背景と目的

既存の `baseline-sparse-sgd-non-gpu` は pivot 数を `O(nh)` に抑える CPU ベースラインとして動作する一方、論文「Graph Drawing by Stochastic Gradient Descent」の Algorithm 2 と異なり、pivot を最大距離候補から選び、領域補正を行わない対称重み `1/d²` で両端点を更新しています。その結果、USPowerGrid・200 pivots の full stress が `1.536×10⁶` に留まり、論文 Figure 14 の約 `7.2×10⁵` を再現できていません。

## 変更内容

- `MaxMinRandomSP` を、各頂点から選択済み pivot 集合への最小距離に比例した確率で次の pivot を選ぶ方式へ修正します。
- 各頂点を最近傍 pivot の領域 `R(p)` へ割り当て、論文の `s_ip` と方向別の補正重み `w'_ip = s_ip w_ip` を構築します。
- 各制約に端点ごとの方向別重みを保持し、`μ_i` と `μ_j` を個別に計算する非対称 Sparse SGD 更新へ修正します。
- 入力、pivot 数、反復回数、epsilon、乱数 seed を明示的に指定できるようにし、実験を再現可能にします。
- 出力へ seed、選択方式、重み方式を記録し、結果が論文準拠条件で生成されたことを追跡可能にします。
- USPowerGrid を200 pivots、15 iterations、`epsilon=0.1`、25個の固定 seed で実行し、`gpu_visualizer` と同じ full stress を集計して論文 Figure 14 に近いことを検証します。
- 論文値に近い run を `gpu_visualizer` で描画し、stress 表示と PNG の生成を確認します。

## 機能

### 新規機能

- `cpu-sparse-sgd`: 論文 Algorithm 2 に準拠した、CPU のみで動作する pivot ベース Sparse SGD と再現可能な品質検証。

### 変更する既存機能

なし。

## 影響範囲

- `baseline-sparse-sgd-non-gpu/src/graph.rs` の pivot 選択、領域割当、疎制約データを変更します。
- `baseline-sparse-sgd-non-gpu/src/algorithm.rs` の更新を方向別重みに対応させます。
- `baseline-sparse-sgd-non-gpu/src/main.rs` の実行設定と出力メタデータを再現可能な形式へ拡張します。
- `gpu_visualizer` の full stress 定義は変更せず、論文再現の評価器として使用します。
- 既存の `baseline-sgd-non-gpu` と GPU 実装は変更しません。
