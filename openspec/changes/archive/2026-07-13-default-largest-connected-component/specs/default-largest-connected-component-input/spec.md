## ADDED Requirements

### Requirement: 最大連結成分の既定選択
CPU Sparse SGDおよびGPU Sparse SGDは、Matrix Market入力から最大連結成分を常に選択してレイアウトしなければならない（MUST）。ユーザーは追加の成分選択オプションを指定せず、既存の`cargo run`引数だけで実行できなければならない（MUST）。

#### Scenario: 非連結入力を通常実行する
- **WHEN** 非連結Matrix Marketファイルを位置引数または`--input`で指定して実行したとき
- **THEN** 最大連結成分を抽出・再番号付けしてSparse SGDを開始する

#### Scenario: 連結入力を通常実行する
- **WHEN** 連結Matrix Marketファイルを通常実行したとき
- **THEN** 全頂点・全辺を採用し、従来のSGDパラメータ指定をそのまま使用する

#### Scenario: 同率最大成分を選択する
- **WHEN** 同じ頂点数の最大連結成分が複数あるとき
- **THEN** 最小の元頂点番号を含む成分を決定的に選択する

### Requirement: 最大連結成分の端末表示
システムは、CPU版・GPU版の各通常実行で、入力グラフと採用した最大連結成分の統計をターミナルへ出力しなければならない（MUST）。

#### Scenario: 成分統計を表示する
- **WHEN** Matrix Market入力の読み込みと成分選択が完了したとき
- **THEN** 元頂点数、元辺数、連結成分数、採用頂点数、採用辺数、頂点採用率、辺採用率を表示する

#### Scenario: ほぼ孤立した行列を表示する
- **WHEN** 最大連結成分の頂点採用率が小さい入力を実行したとき
- **THEN** 実際の採用頂点数と採用率を省略せず表示する

### Requirement: 縮約グラフの追跡可能な出力
システムは、選択後の内部頂点番号と元Matrix Market頂点番号の対応、および成分統計を結果出力へ記録しなければならない（MUST）。

#### Scenario: 初期・最終結果を保存する
- **WHEN** CPU版またはGPU版が結果を保存したとき
- **THEN** 両方のTXTへ元／採用頂点数・辺数、連結成分数、採用率、vertex mapパスを記録する

#### Scenario: vertex mapを保存する
- **WHEN** 結果prefixを生成したとき
- **THEN** prefixごとに1つの`-vertex-map.txt`を`local_id original_id`の0始まり形式で保存する

### Requirement: 実行不能な最大成分の明確な失敗
システムは、最大連結成分がSparse SGD制約を形成できない場合、選択統計を含むエラーを返し、結果ファイルを生成してはならない（MUST）。

#### Scenario: 自己ループだけの入力を実行する
- **WHEN** 全成分が1頂点または有効な非自己ループ辺を持たないとき
- **THEN** 採用頂点数・採用辺数・成分数を含むエラーを返す
