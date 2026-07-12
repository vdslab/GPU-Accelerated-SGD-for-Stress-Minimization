## 背景と目的

論文準拠の CPU Sparse SGD は USPowerGrid・200 pivots で論文に近い stress を再現できましたが、約 `O(nh+m)` 個の制約を逐次処理するため、大規模グラフの反復時間が課題です。GPU 化では単純な並列実行による座標書き込み競合を避けながら、CPU 版に近いレイアウト品質と、pivot 数程度のラウンド数を両立する必要があります。

## 変更内容

- 新しい Rust/WGPU crate `sparse-sgd-gpu` を作成し、GPU Sparse SGD の土台を追加します。
- グラフ入力、GPU 初期化、結果出力、時間計測は `rr_gpu` の WGPU 27 実装とファイル形式を参考にします。
- pivot 選択、領域補正、方向別重み、学習率、seed は `baseline-sparse-sgd-non-gpu` と同じ論文準拠の CPU 前処理を使用します。
- 制約を「非pivot頂点から pivot への一方向制約」と「両端を書き込む pivot–pivot／隣接辺制約」に分類します。
- 一方向制約は1頂点につき1 GPU invocationを割り当て、その invocation 内で pivot を逐次処理することで、pivotを読み取り専用にして競合を避けます。
- 両端更新制約は pivot–pivot の circle method を基礎とするラウンドロビンへ隣接辺を詰め、各ラウンドを matching にして atomic や lock を使用せず競合を避けます。
- ラウンド順、pivot順、phase順を seed から再現可能にランダム化し、GPU 固有の固定順序バイアスを抑えます。
- スケジュールの完全性、重複なし、同一ラウンド内の頂点競合なし、ラウンド数を CPU テストで検証します。
- CPU Sparse SGD と同じ入力・seed・実験条件で full stress を比較し、USPowerGrid・200 pivots で CPU 版と同程度の精度を確認します。

## 機能

### 新規機能

- `gpu-sparse-sgd-round-robin`: 論文準拠 Sparse SGD を、方向別制約分類と競合しないラウンドロビンで WGPU 上へ実行する機能。

### 変更する既存機能

なし。

## 影響範囲

- `sparse-sgd-gpu/` に Cargo manifest、CPU 前処理、スケジューラ、WGPU 実行、WGSL shader、CLI、テストを追加します。
- `rr_gpu` は参照元として使用し、既存コードは変更しません。
- `baseline-sparse-sgd-non-gpu` は正解基準として使用し、既存コードは変更しません。
- 出力は `gpu_visualizer` でそのまま読み込み、full stress と PNG を生成できる形式にします。
- Metal、Vulkan、DX12 など WGPU が提供する compute adapter を実行時に必要とします。
