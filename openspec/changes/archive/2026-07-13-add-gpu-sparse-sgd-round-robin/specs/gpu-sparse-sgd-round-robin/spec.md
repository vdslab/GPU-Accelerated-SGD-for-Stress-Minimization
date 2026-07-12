## ADDED Requirements

### Requirement: 独立した WGPU Sparse SGD crate
システムは、`sparse-sgd-gpu` という独立した Rust crate として、論文準拠 Sparse SGD を WGPU compute 上で実行する土台を提供しなければならない（MUST）。

#### Scenario: GPU Sparse SGD を起動する
- **WHEN** 有効な連結 Matrix Market グラフと実行パラメータを指定したとき
- **THEN** WGPU adapter と device を初期化し、GPU Sparse SGD を実行して初期座標と最終座標を返す

#### Scenario: GPU を利用できない
- **WHEN** compute 対応 adapter または必要な device limits を取得できないとき
- **THEN** 原因を含むエラーを返し、未初期化の結果ファイルを生成しない

### Requirement: CPU 論文準拠前処理との一致
システムは、pivot 選択、最短路、領域 `R(p)`、補正係数 `s_ip`、方向別重み、学習率、初期座標を `baseline-sparse-sgd-non-gpu` と同じアルゴリズムと seed から生成しなければならない（MUST）。

#### Scenario: 同一 seed の前処理を比較する
- **WHEN** CPU版とGPU版へ同じ入力、pivot数、反復数、epsilon、seedを指定したとき
- **THEN** pivot列、制約の端点・距離・方向別重み、学習率列、初期座標が浮動小数点変換前に一致する

### Requirement: 方向別制約の分類
システムは、Sparse SGD 制約を、pivotを読み取り専用とし非pivot端点だけを書き込む一方向制約と、両端点を書き込むpivot–pivot制約およびグラフ辺制約へ重複なく分類しなければならない（MUST）。

#### Scenario: 非pivot頂点とpivotの制約を分類する
- **WHEN** 制約の非pivot側重みが正でpivot側重みが0であるとき
- **THEN** その制約を非pivot頂点の一方向制約リストへ1回だけ格納する

#### Scenario: 両端更新制約を分類する
- **WHEN** 制約の両方向重みが正であるとき
- **THEN** その制約を両端更新スケジュールへ1回だけ格納する

#### Scenario: 全制約を照合する
- **WHEN** 分類が完了したとき
- **THEN** 元の全制約と分類後制約の集合が一致し、欠落と重複がない

### Requirement: 一方向 pivot phase の競合回避
システムは、1 GPU invocationを1非pivot頂点へ割り当て、その頂点の全一方向 pivot 制約を invocation 内で逐次処理し、他の invocation が同じ頂点またはpivot座標へ書き込まない状態で実行しなければならない（MUST）。

#### Scenario: 一方向 phase をdispatchする
- **WHEN** 非pivot頂点ごとのCSR形式制約リストをGPUへdispatchしたとき
- **THEN** 各 invocation は担当頂点だけを書き込み、pivot座標は読み取り専用とし、atomic操作やlockを使用しない

#### Scenario: pivot処理順をランダム化する
- **WHEN** 新しいannealing iterationを開始したとき
- **THEN** seedから再現可能なpivot順列と頂点別rotationを使用し、各一方向制約を1回ずつ処理する

### Requirement: 両端更新制約の競合しないラウンドロビン
システムは、pivot–pivot制約をcircle methodでラウンドロビン化し、隣接辺制約を端点が空いているラウンドへ充填したうえで、残余辺をmatchingへ分解しなければならない（MUST）。

#### Scenario: pivot–pivot制約をスケジュールする
- **WHEN** `h` 個のpivotがあるとき
- **THEN** 全pivot対を、偶数 `h` では `h-1`、奇数 `h` ではdummyを使った `h` ラウンド以内のmatchingへ分解する

#### Scenario: 隣接辺を既存ラウンドへ充填する
- **WHEN** pivotラウンド内で辺の両端点が未使用であるとき
- **THEN** その辺をラウンドへ追加しても全端点の出現回数を最大1に保つ

#### Scenario: 残余辺をスケジュールする
- **WHEN** pivotラウンドへ追加できなかった隣接辺があるとき
- **THEN** 残余辺を追加のmatchingラウンドへ分解し、全辺をちょうど1回含める

#### Scenario: round内競合を検証する
- **WHEN** 両端更新スケジュールが完成したとき
- **THEN** 各roundの全制約について端点集合が互いに素であることをCPU検証器が確認する

### Requirement: pivot数程度のラウンド数
システムは、両端更新ラウンド数と総dispatch数を記録し、pivot数を支配項とするスケジュールを生成しなければならない（MUST）。

#### Scenario: 一般グラフの上限を検証する
- **WHEN** pivot数を `h`、元グラフの最大次数を `Δ_E` としたとき
- **THEN** 両端更新ラウンド数を `h + 2Δ_E` 以下とし、超過時は実行前にスケジューラエラーを返す

#### Scenario: USPowerGridをスケジュールする
- **WHEN** USPowerGridを200 pivotsで前処理したとき
- **THEN** 両端更新ラウンド数を220以下とし、一方向phaseを含む1 iteration当たりのdispatch数を221以下とする

### Requirement: GPU dispatch間の可視性と再現可能性
システムは、書き込みを伴う各phaseおよびroundの間にWGPUが保証するstorage buffer可視性を確保し、同一roundを同時実行しなければならない（MUST）。

#### Scenario: 複数roundを実行する
- **WHEN** 1 iterationのdispatch列をcommand encoderへ記録するとき
- **THEN** dynamic uniform offsetまたは同等のround別パラメータを使用し、round順に実行して前roundの座標更新を次roundから可視にする

#### Scenario: 同じseedで再実行する
- **WHEN** 同じadapter、入力、全パラメータ、seedで2回実行したとき
- **THEN** スケジュール、phase順、round順、pivot順と最終座標が一致する

### Requirement: 論文準拠のGPU座標更新
GPU shaderは、CPU版と同じ半変位、方向別 `μ_u` と `μ_v`、clamp、座標一致時の有限値処理をf32で実行しなければならない（MUST）。

#### Scenario: 一方向制約を更新する
- **WHEN** pivot側重みが0の制約を処理するとき
- **THEN** 非pivot座標だけを更新し、pivot座標を変更しない

#### Scenario: 両端制約を更新する
- **WHEN** 両方向重みが正の制約を処理するとき
- **THEN** 各端点を対応する方向別 `μ` で更新する

#### Scenario: 座標が一致する
- **WHEN** 制約両端の現在座標が一致するとき
- **THEN** 決定的な微小方向または同等の安全処理を使用し、NaNとInfを生成しない

### Requirement: rr_gpu互換の入出力と計測
実行ファイルは、`rr_gpu` と同様にGPU情報、前処理時間、スケジュール統計、GPU時間、iteration平均を表示し、`gpu_visualizer` が読める初期・最終結果を保存しなければならない（MUST）。

#### Scenario: 実行条件を指定する
- **WHEN** `--input`、`--pivots`、`--iterations`、`--epsilon`、`--seed` を指定したとき
- **THEN** 指定値を前処理とGPU実行へ使用し、結果メタデータへ記録する

#### Scenario: 結果を保存する
- **WHEN** GPU実行が正常に完了したとき
- **THEN** `sparse-sgd-gpu` を含む `-0.txt` と `-1.txt` を出力し、辺、座標、pivot、seed、round数、GPU adapterを記録する

### Requirement: CPU版と同程度のstress品質
システムは、GPU固有の並列順序を許容しつつ、`baseline-sparse-sgd-non-gpu` と同程度のfull stressを達成しなければならない（MUST）。

#### Scenario: 小規模グラフでCPU版と比較する
- **WHEN** 複数の小規模連結グラフを同じ入力条件とseedでCPU版・GPU版へ実行したとき
- **THEN** 全座標が有限であり、GPU版full stressをCPU版full stressの10%以内とする

#### Scenario: USPowerGridで25 runsを比較する
- **WHEN** 200 pivots、15 iterations、`epsilon=0.1`、seed 0〜24でGPU版を実行したとき
- **THEN** 平均full stressをCPU基準 `726,046.63` の5%以内かつ論文許容範囲 `576,000`〜`864,000` にする

#### Scenario: gpu_visualizerで代表runを描画する
- **WHEN** GPU 25 runsからCPU基準に最も近いrunを選んだとき
- **THEN** `gpu_visualizer` が同じstressを表示し、2048×2048 PNGを生成する
