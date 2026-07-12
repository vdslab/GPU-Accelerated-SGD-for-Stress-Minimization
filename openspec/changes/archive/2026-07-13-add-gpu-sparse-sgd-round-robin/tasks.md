## 1. GPU crate と CPU 前処理の土台

- [x] 1.1 `sparse-sgd-gpu` crateを作成し、`rr_gpu`に合わせたWGPU 27、bytemuck、pollster、rand、chrono、anyhow、Matrix Market入力の依存を追加する
- [x] 1.2 `baseline-sparse-sgd-non-gpu`と同じ論文準拠のグラフ読み込み、連結性検証、pivot選択、`R(p)`、`s_ip`、方向別制約、学習率、seed付き初期配置を実装する
- [x] 1.3 同じ小規模グラフとseedでCPU版・GPU版前処理のpivot、制約、学習率、初期座標が一致するテストを追加する
- [x] 1.4 `--input`、`--pivots`、`--iterations`、`--epsilon`、`--seed`を持つCLIと値検証を追加する

## 2. 制約分類と競合なしスケジューラ

- [x] 2.1 全方向別制約を非pivot頂点ごとのone-sided CSRとtwo-sided制約へ重複なく分類する
- [x] 2.2 pivot–pivot完全グラフをcircle methodで偶数`h`は`h-1`、奇数`h`は`h` matching roundsへ分解する
- [x] 2.3 pivot–pivotの隣接辺を既存slotへ統合し、その他の隣接辺をseed付きfirst-fitでpivot roundsの空きへ充填する
- [x] 2.4 残余辺をfirst-fit matching roundsへ分解し、flattened constraints、round offsets、round countsを生成する
- [x] 2.5 制約の完全性・一意性、各roundの端点排他、`h+2Δ_E`上限を検証するCPU validatorを実装する
- [x] 2.6 偶数・奇数pivot、pivot隣接辺、非pivot辺、重複辺、空きround充填、spill roundsの単体テストを追加する
- [x] 2.7 USPowerGrid・200 pivotsのスケジュールを生成し、two-sided roundsが220以下であることを確認する

## 3. WGPU データと one-sided pipeline

- [x] 3.1 `rr_gpu`を参考にadapter/deviceを初期化し、adapter情報と必要storage buffer size/limitsを検証する`GpuContext`を実装する
- [x] 3.2 positions、pivots、one-sided CSR、two-sided constraints、round metadata、permutation、dynamic uniformsのPOD型とGPU bufferを定義する
- [x] 3.3 1 invocationが1非pivot頂点だけを書き、CSR内のpivot制約を逐次処理するone-sided WGSL entry pointを実装する
- [x] 3.4 seed付き共通pivot permutationとvertex別rotationを生成・uploadし、各一方向制約を1回処理することをテストする

## 4. two-sided round pipeline と実行制御

- [x] 4.1 1 invocationが1two-sided制約を処理し、方向別`μ_u`、`μ_v`で両端更新するWGSL entry pointを実装する
- [x] 4.2 WGSLの半変位、clamp、一方向0、決定的な座標一致処理をCPU式と照合するテストを追加する
- [x] 4.3 one-sided/two-sided pipelines、bind groups、256-byte aligned dynamic uniform offsetsを作成する
- [x] 4.4 iterationごとにphase順、two-sided round順、pivot順をseedから再現可能にシャッフルする実行制御を実装する
- [x] 4.5 1 iteration分のdispatch列を1 command encoderへ順序通りに記録し、dispatch境界で座標更新を可視にする
- [x] 4.6 最終positionsをreadbackし、必要な中心化と全座標のNaN/Inf検証を行う

## 5. GPU 正しさと再現性

- [x] 5.1 WGSL moduleと両compute pipelineがWGPU validation errorなしで作成できるテストを追加する
- [x] 5.2 小規模グラフで全scheduled constraintが実行され、競合検出canaryとguard領域が破壊されないことを確認する
- [x] 5.3 同じadapter・入力・seedの2 runsでschedule、phase順、round順、最終座標が一致することを確認する
- [x] 5.4 複数の小規模連結グラフでGPU full stressがCPU版の10%以内、全座標が有限であることを確認する
- [x] 5.5 one-sided一括phaseで品質基準を満たさない場合はpivot chunk interleaveを実装し、満たす場合は不要であることを検証記録へ残す

## 6. 入出力と計測

- [x] 6.1 `rr_gpu`互換の初期・最終テキスト出力を実装し、辺、座標、pivot、seed、adapter、round数、dispatch数を記録する
- [x] 6.2 CPU前処理、schedule生成、GPU upload、GPU iteration、readback、合計時間とiteration平均を計測して表示する
- [x] 6.3 出力を`gpu_visualizer`で読み込み、stress計算とPNG生成ができるスモークテストを追加する

## 7. 品質とパフォーマンス検証

- [x] 7.1 `cargo fmt --check`、`cargo clippy --all-targets -- -D warnings`、`cargo test`を実行し、全て成功させる
- [x] 7.2 既存`rr_gpu`、`baseline-sparse-sgd-non-gpu`、`gpu_visualizer`へ意図しない変更がないことを確認する
- [x] 7.3 USPowerGridを200 pivots、15 iterations、`epsilon=0.1`、seed 0〜24でGPU実行する
- [x] 7.4 25-runのfull stress平均をCPU基準`726,046.63`の5%以内かつ論文範囲`576,000`〜`864,000`にする
- [x] 7.5 USPowerGridのtwo-sided roundsを220以下、総dispatch数/iterationを221以下とし、GPU時間とCPU版時間を記録する
- [x] 7.6 CPU基準に最も近いGPU runを`gpu_visualizer`で2048×2048描画し、stress表示とPNGを検証する
- [x] 7.7 25件のstress、平均・最小・最大、round統計、timing、代表ファイルをOpenSpec verification reportへ記録する
