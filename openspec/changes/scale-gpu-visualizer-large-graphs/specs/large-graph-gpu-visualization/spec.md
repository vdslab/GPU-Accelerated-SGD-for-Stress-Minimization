## ADDED Requirements

### Requirement: 規模に応じたストレス評価

visualizerは `auto`、`exact`、`sampled`、`off` のストレス評価モードを提供しなければならない（SHALL）。引数を指定しない通常実行は `auto` を使用し、8,000頂点以下では厳密値、それを超える入力では有界な標本化推定を使用しなければならない（MUST）。

#### Scenario: 小規模入力を通常実行する

- **WHEN** 8,000頂点以下の有効な結果TXTをモード指定なしで読み込む
- **THEN** visualizerは従来と同じ定義の厳密ストレスを計算し、`Stress` として表示する

#### Scenario: 大規模入力を通常実行する

- **WHEN** 8,000頂点を超える有効な結果TXTをモード指定なしで読み込む
- **THEN** visualizerは全頂点始点の厳密計算を実行せず、既定64始点の標本化ストレスを計算する
- **AND** `Approx stress`、標本数、seed、計算時間を表示する

#### Scenario: 厳密計算を明示する

- **WHEN** 利用者が `--stress exact` を指定する
- **THEN** visualizerは規模にかかわらず厳密ストレスを計算する
- **AND** 8,000頂点を超える場合は計算開始前に高コストであることを警告する

#### Scenario: ストレス計算を省略する

- **WHEN** 利用者が `--stress off` を指定する
- **THEN** visualizerはBFSによるストレス計算を行わず、計算を省略したことを表示して描画を続行する

### Requirement: 再現可能な標本化ストレス

標本化ストレスは、指定された標本数とseedに対して重複しない始点集合を決定的に選び、無向頂点対の厳密ストレス総和を推定しなければならない（SHALL）。計算量は標本数を `K` として `O(K(|V|+|E|))`、BFS作業メモリは `O(|V|)` に収めなければならない（MUST）。

#### Scenario: 同じ条件で繰り返す

- **WHEN** 同じ入力、標本数、seedで `sampled` を2回実行する
- **THEN** 選択始点と表示される近似ストレスは一致する

#### Scenario: 標本数を変更する

- **WHEN** 利用者が `--stress-samples K` で有効な正整数を指定する
- **THEN** visualizerは `min(K, |V|)` 個の重複しない始点を使用し、実際に使用した個数を表示する

#### Scenario: 全頂点を標本にする

- **WHEN** 連結した小規模グラフで標本数が頂点数以上に設定される
- **THEN** 標本化式の結果は浮動小数点許容誤差内で厳密ストレスと一致する

### Requirement: インデックスによるGPU辺描画

visualizerは正規化済みノード座標を共有vertex bufferへ一度格納し、辺端点を `u32` index bufferへ格納してindexed line-listとして描画しなければならない（SHALL）。辺ごとに両端の浮動小数点座標を複製したCPU/GPUバッファを作成してはならない（MUST NOT）。

#### Scenario: 全辺を通常描画する

- **WHEN** 有効な結果TXTを辺数上限なしで描画する
- **THEN** visualizerは入力の全辺をindex bufferへ格納し、全辺を描画する

#### Scenario: GPU制限を超える

- **WHEN** 必要なvertex bufferまたはindex bufferサイズが選択GPUの上限を超える
- **THEN** visualizerはGPU描画開始前に、必要サイズと上限を含む診断エラーを返す

#### Scenario: 不正な辺端点を検出する

- **WHEN** 辺が存在しない頂点indexを参照している
- **THEN** visualizerは該当する辺と端点を示す入力エラーを返し、GPUへ不正データを送信しない

### Requirement: 大規模グラフ向けLOD

visualizerは出力画素数と頂点数に応じてノード半径を自動選択し、大規模入力で固定4pxによる過度な重なりを避けなければならない（SHALL）。利用者はノード半径を明示的に上書きできなければならない（MUST）。

#### Scenario: 25万頂点を自動設定で描画する

- **WHEN** 2048×2048画像へ25万頂点を自動ノード半径で描画する
- **THEN** visualizerは4pxより小さい半径を選び、選択値を表示する

#### Scenario: ノード半径を上書きする

- **WHEN** 利用者が有効な `--node-radius` を指定する
- **THEN** visualizerは自動値ではなく指定値を使用する

#### Scenario: 辺数上限を明示する

- **WHEN** 利用者が総辺数より小さい `--max-edges N` を指定する
- **THEN** visualizerは決定的に選択したN本以下の辺を描画する
- **AND** 描画辺数、総辺数、間引きを行った事実を表示する

### Requirement: 入力検証と段階別診断

visualizerは既存の結果TXT形式を読み込み、利用可能なnode/edge countを事前確保へ使用し、座標数、座標の有限性、辺端点をGPU初期化前に検証しなければならない（SHALL）。また主要処理段階の所要時間と処理規模を表示しなければならない（MUST）。

#### Scenario: 既存結果を読み込む

- **WHEN** Sparse SGDまたはRR-SGDが出力した既存形式の有効なTXTを指定する
- **THEN** 形式変換なしで読み込み、頂点数と辺数を表示してPNGを生成する

#### Scenario: 段階別時間を報告する

- **WHEN** 描画が正常に完了する
- **THEN** 読込、ストレス評価、GPU準備・転送/描画/読戻し、PNG保存、合計の時間を区別して表示する

### Requirement: 25万頂点級の受入性能

visualizerは25万頂点・約200万辺の連結グラフを通常設定で処理するとき、厳密全点対ストレス計算を実行せず、辺座標を展開せず、GPU validation errorまたはメモリ不足を起こさず2048×2048 PNGを生成できなければならない（SHALL）。

#### Scenario: 論文規模相当のfixtureを描画する

- **WHEN** 25万頂点・190万辺以上200万辺以下の決定的fixtureを `--stress auto` で対応GPU上に描画する
- **THEN** 標本化ストレス経路とindexed edge drawingを使用してPNG生成を完了する
- **AND** 全段階の時間、実際の描画頂点数、描画辺数、ストレス条件を記録する

#### Scenario: 小規模互換性を確認する

- **WHEN** 既存の小規模回帰fixtureを新旧の厳密ストレス実装で評価する
- **THEN** 厳密ストレス値は定めた浮動小数点許容誤差内で一致する
- **AND** 出力PNGの寸法と描画対象の頂点数・辺数は一致する
