## 1. crate の準備とグラフ入力

- [x] 1.1 CPU 向け依存だけを持ち、`main`、`graph`、`algorithm` モジュールへ分割した `baseline-sparse-sgd-non-gpu` Rust crate を作成する
- [x] 1.2 Matrix Market 形式のグラフ読み込み、自己ループと重複辺の除外、無向隣接リストを実装する
- [x] 1.3 入力グラフの連結性を検証し、非連結グラフを明確なエラーで拒否する

## 2. 論文準拠の疎前処理

- [x] 2.1 到達不能を明示的に表す重みなし単一始点 BFS を実装する
- [x] 2.2 `MaxMinRandomSP` を最近傍 pivot 距離に比例する weighted sampling へ変更し、seed で再現可能にする
- [x] 2.3 全 pivot の距離から各頂点の領域 `R(p)` を構築し、距離 tie を pivot 選択順で解決する
- [x] 2.4 領域ごとの距離ヒストグラムと累積和から `s_ip=|{j∈R(p):2d_pj≤d_pi}|` を計算する
- [x] 2.5 制約型を方向別重みへ変更し、非近傍頂点-pivot、pivot-pivot、グラフ辺の各規則を実装する
- [x] 2.6 正の方向別重みから指数減衰アニーリングスケジュールを生成する
- [x] 2.7 固定 seed pivot、領域割当、`s_ip`、方向別重み、重複排除、学習率端点の単体テストを追加する

## 3. 非対称 CPU Sparse SGD

- [x] 3.1 seed 付き RNG によるランダム二次元配置と中心化を実行全体へ接続する
- [x] 3.2 `μ_i=min(w'_ijη,1)` と `μ_j=min(w'_jiη,1)` を個別に計算する非対称更新を実装する
- [x] 3.3 一方向重み0、方向別重み差、座標一致時の有限値、反復ごとの再現可能なシャッフルをテストする
- [x] 3.4 小規模グラフで同一 seed の pivot、初期座標、最終座標が完全一致する統合テストを追加する

## 4. 再現可能な実行と出力

- [x] 4.1 `--input`、`--pivots`、`--iterations`、`--epsilon`、`--seed` の実行引数と値検証を追加する
- [x] 4.2 初期・処理後レイアウトを既存ベースライン互換のテキスト形式で保存する
- [x] 4.3 seed、pivot 順、pivot 選択方式、領域重み方式を両結果ファイルへ記録する

## 5. コード品質

- [x] 5.1 `cargo fmt --check`、`cargo clippy --all-targets -- -D warnings`、`cargo test` を実行して全て成功させる
- [x] 5.2 既存 `baseline-sgd-non-gpu`、GPU 実装、`gpu_visualizer` の stress 定義に意図しない変更がないことを確認する

## 6. USPowerGrid 論文再現

- [x] 6.1 `USpowerGrid.mtx` を200 pivots、15 iterations、`epsilon=0.1`、seed 0〜24で実行する
- [x] 6.2 各 run の最終結果を `gpu_visualizer` と同じ full stress 式で評価し、25件の stress と平均・最小・最大を記録する
- [x] 6.3 25-run 平均が基準 `7.2×10⁵` の±20%（`5.76×10⁵`〜`8.64×10⁵`）に入ることを確認する
- [x] 6.4 基準値に最も近い run を `gpu_visualizer` で描画し、同じ stress の表示と PNG 生成を確認する
- [x] 6.5 合格しない場合は完了扱いにせず、論文の pivot 選択、方向別重み、stress 定義との差分を記録する
