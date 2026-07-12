## 背景

現在の CPU Sparse SGD は、200 pivots で USPowerGrid の full stress を初期値 `1.143×10⁷` から `1.536×10⁶` まで下げていますが、論文 Figure 14 の200-pivot SGD の中心値約 `7.2×10⁵` には届いていません。調査により、現行実装は Algorithm 2 の3点を満たしていないことが分かりました。

1. `MaxMinRandomSP` を距離比例確率ではなく最大距離候補から選んでいる。
2. pivot 領域 `R(p)` と補正係数 `s_ip` を構築せず、重みを単純な `1/d²` としている。
3. 論文の方向別重み `w'_ij`、`w'_ji` を単一の対称重みへ置き換えている。

`gpu_visualizer` は全到達可能頂点対について Equation (1) の full stress を計算しており、論文再現の品質評価器としてそのまま利用できます。

## 目標と対象外

**目標:**

- 論文「Graph Drawing by Stochastic Gradient Descent」の最終版 Algorithm 2 を CPU 上で再現します。
- pivot 選択から最終座標までを seed で再現可能にします。
- `O(h(n+m))` の BFS 前処理と `O(nh+m)` の制約規模を維持します。
- USPowerGrid・200 pivots の25-run 平均 full stress を Figure 14 の約 `7.2×10⁵` に近づけます。
- 代表 run を既存 `gpu_visualizer` で描画し、表示 stress と PNG を確認します。

**対象外:**

- GPU Sparse SGD の修正。
- `baseline-sgd-non-gpu` の変更。
- Figure 14 の majorization 曲線の再現。
- 論文の図から読み取れない未丸めの原データと完全一致すること。
- 重み付きグラフおよび非連結グラフの Sparse SGD。

## 設計上の判断

### MaxMinRandomSP を距離比例サンプリングとして実装する

最初の pivot は一様ランダムに選択します。pivot を追加するたびに BFS の結果で各頂点の最近傍 pivot 距離 `δ_i=min_{p∈P} d_ip` を更新し、次の pivot を `Pr(i)=δ_i/Σ_j δ_j` で選択します。選択済み pivot は `δ_i=0` なので再選択されません。連結グラフに限定することで、無限距離を確率へ変換する曖昧さを排除します。

乱択には呼び出し元から渡された seed 付き RNG だけを使用します。整数距離の累積和を用いた weighted sampling により、浮動小数点の丸めに依存しない再現性を保ちます。

### pivot 距離と領域 R(p) を一度の前処理で構築する

各 pivot の BFS 距離を `h×n` の配列として保持します。各頂点は距離が最小の pivot へ割り当て、同距離なら選択順が早い pivot を採用します。この規則は論文で未指定の tie をテスト可能にするための実装上の決定です。

各領域について `d_pj` の距離ヒストグラムと累積和を作ります。頂点 `i` の補正係数は `floor(d_pi/2)` までの累積数として `O(1)` で取得でき、`2d_pj≤d_pi` を整数演算で正確に評価します。

### 制約に方向別重みを保持する

制約型を、単一の `wij` から次の概念へ変更します。

```text
SparseConstraint {
    i, j,
    dij,
    weight_i_to_j,  // w'_ij: i を動かす係数
    weight_j_to_i,  // w'_ji: j を動かす係数
}
```

非近傍の頂点 `i` と pivot `p` について `w'_ip=s_ip/d_ip²` を設定し、逆方向は0のままとします。両端が pivot の場合は、もう一方の pivot 関係から逆方向が独立に設定されます。グラフ辺は pivot 制約を置き換え、`d=1`、両方向重み1とします。

端点を canonical pair で一意化すること自体は維持しますが、挿入方向に応じて対応する方向別フィールドだけを更新します。これにより pivot-pivot 制約の非対称性を失いません。

### SGD 更新で μ_i と μ_j を分離する

変位ベクトルは論文 Equation (3) と同じ半変位を使用します。各端点の `μ` を方向別重みから個別に求め、一方の重みが0ならその端点を固定します。座標一致時の微小ランダム方向は seed 付き RNG から生成し、有限値を保証します。

学習率の `w_min` と `w_max` は、実際に `μ` の計算へ使用する全方向別重みのうち正の有限値から求めます。これにより最初の反復では全ての正重み `μ` が1に制限され、最後は最大重みの `μ` が `epsilon` になります。

### 再現実験をコマンドから設定可能にする

実行ファイルは少なくとも `--input`、`--pivots`、`--iterations`、`--epsilon`、`--seed` を受け取ります。結果ファイルには、これらに加えて選択 pivot、`max-min-random-sp-distance-proportional`、`ortmann-region-directed-weight` を記録します。

### Figure 14 の比較は25-run 平均 full stress で行う

論文と同じく USPowerGrid、200 pivots、15 iterations、`epsilon=0.1`、ランダムな `[0,1)²` 初期配置を使用し、seed 0〜24を実行します。各最終結果は `gpu_visualizer/src/stress.rs` と同じ全点対 stress

```text
Σ_{i<j} d_ij^-2 (||X_i-X_j||-d_ij)²
```

で評価します。Figure 14 は数値表ではなく対数軸の図であるため、読み取り値 `7.2×10⁵` に±20%の許容範囲を設けます。すなわち25-run 平均の合格範囲は `5.76×10⁵` 以上 `8.64×10⁵` 以下です。

25 runs のうち基準値に最も近い結果を `gpu_visualizer` へ渡し、表示された stress が集計値と一致し、PNG が生成されることを確認します。

## リスクとトレードオフ

- [Figure 14 に未丸めの数値がない] → 図からの読み取り値を明記し、相対誤差20%で評価します。
- [25 runs は実行時間が長い] → seed を固定し、各 run の結果を保持して再計算を避けます。
- [方向別重みで既存制約型とテストが変わる] → 辺、頂点-pivot、pivot-pivot、一方向0を個別にテストします。
- [tie の扱いが論文に明記されていない] → pivot 選択順で決定する規則を採用し、出力へ pivot 順を保存します。
- [単一 run は局所最小値に左右される] → 25-run 平均で品質を判定し、代表画像は基準値に最も近い run を使います。

## 導入手順

1. pivot 選択と制約型を論文準拠へ変更します。
2. 領域・方向別重み・非対称更新の単体テストを通します。
3. seed 指定と実験メタデータを追加します。
4. USPowerGrid を25 seeds で実行し、full stress の統計を作成します。
5. 基準値に最も近い run を `gpu_visualizer` で描画します。
6. 合格範囲へ入らない場合はタスクを完了にせず、pivot、重み、stress 定義の差分を再調査します。

## 未解決事項

著者の旧公開リポジトリは現在取得できないため、最終版論文 Algorithm 2 と、その基礎となる Ortmann et al. の定義を仕様の正本とします。
