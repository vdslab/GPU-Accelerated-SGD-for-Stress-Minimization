# 通常のf32だけでのCPU比較

ユーザーの指定により、丸め誤差を補償する診断経路を撤去した。
GPUの座標・更新は単一f32、CPU更新はf64のまま。同じ反復数で比較する。

撤去したもの：high/low座標、double-single演算、重み・学習率の残差保持、
丸め境界を強制する処理、反復中の再中心化。また、更新式を変えるstep cap診断も撤去した。
Uniformsは32 bytes、座標はvec2<f32>の通常経路に戻した。
`quality_search`は`standard`と`shuffle`だけを受け付け、検証runnerの既定値は`shuffle / 4@0`。

標準CLIは以前から補償を使っていない。頂点別pivot順序と4分割の相実行は維持した。
これはCPUの全制約シャッフルとは異なるRRの更新順序なので、比較条件として明記する。
誤差補償は使わないが、全てのstress差を丸め誤差だけに帰属することもしない。
共通のf64 stress評価と従来の出力時中心合わせは維持する。
過去の診断結果は履歴として残し、現在の比較対象と区別する。

## 検証

Stanford seed 0 / 200 pivots / 15反復を再実行。
撤去前の標準経路と初期座標SHA、前処理SHA、最終座標SHA、64始点stressが完全一致した。
stressは4,924,176,872.802872。生記録は`stanford-s0.jsonl`。
これは品質がCPUを上回るという主張ではなく、撤去で標準結果が変わらないことの確認。

- release all-targetsテスト：66件成功（補償用テストを削除、CPU診断exampleの既存テストを含む）
- release all-targets clippy `-D warnings` 成功
- 削除した5モード（compensated, extended, coefficients, recenter, cap=0.9）がエラーになることを確認
- fmt・diffチェック成功

```sh
implementations/sparse/sparse-sgd-gpu/target/release/examples/quality_search data/web-Stanford.mtx 0 '4@0' 64 200 shuffle no-cpu 15
```
