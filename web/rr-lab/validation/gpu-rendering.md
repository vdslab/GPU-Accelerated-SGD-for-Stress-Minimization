# GPU描画への変更と実測

2026-11-09。単発実行版とWeb版のコードを調査し、kron_g500-logn17で確認しました。

## 変更前の状態

- 単発実行サイト: `experiments/run_single.py` が `tools/gpu_visualizer` を起動。Rust/wgpuのrender pipelineで辺と円形ノードをGPU描画し、生成したPNGをブラウザへ返します。
- 直前のWeb版: SGDだけがWebGPU。描画WorkerはCanvas 2Dで、OffscreenCanvasに `willReadFrequently: true` を指定していました。Worker化はUIの停止対策であり、GPU描画への変更ではありませんでした。
- 全辺表示では約511万辺のCPU描画が残り、Beforeの描画がAfterの待機列も塞ぎ得る状態でした。SGDが15/15まで進んでも、描画待ちの間はSGDのステータスが残っていました。

旧CPU版のコピーに同じkronファイルを全辺表示で読み込ませると、20秒後も「グラフを描画中」のため検証を打ち切りました。この20秒は入力開始からの経過時間で、純粋な描画時間や下表との速度比を意味しません。

## 変更後

- `render/gpu-renderer.js` と `render/graph.wgsl` にブラウザ用WebGPU rendererを実装。
- `OffscreenCanvas.getContext('webgpu')` → 頂点・フラグメントshader → render pass。線幅付きの辺quad、円形ノードquadをGPUで描画します。
- 座標・辺端点・BFS色をGPUバッファに保持。カメラ操作では座標配列を再送せずuniformを更新。
- 描画を短いGPU submissionに分割し、1回の長いコマンドでGPUを占有しないようにしました。
- 同一初期配置のBefore／Afterは1つのGPU画像を共有。同じseedで計算を開始するときにBeforeを再描画しません。
- Afterを優先し、ドラッグ中は完成済みフレームを表示しながら待機列を最新のカメラにまとめます。
- 描画待ちは「SGD完了・描画中」。実際の描画方式を画面・結果・JSONに記録します。
- 既定は**全辺・全頂点**。GPU描画では頂点の間引き・画素集約をしません。任意で「辺：自動」に変更可能。
- Canvas 2Dは完成したGPU画像の合成と、WebGPU非対応環境のフォールバックに限定します。

## ローカル実測

Chrome 156、Apple / Metal、ソフトウェアフォールバックなし。seed 0、epsilon 0.1、4096 × 4096、全辺表示、15反復。1回の動作確認であり、端末をまたぐ性能保証や統計的な比較実験ではありません。

| データ           |       頂点 |         辺 | pivots | BeforeのGPU描画 | AfterのGPU描画 | SGD後の描画完了待ち |
| ---------------- | ---------: | ---------: | -----: | --------------: | -------------: | ------------------: |
| kron_g500-logn17 |    107,901 |  5,113,981 |     48 |      3,568.8 ms |       750.6 ms |            780.5 ms |
| germany_osm      | 11,548,845 | 12,369,181 |      5 |      8,143.2 ms |       232.9 ms |            394.0 ms |

描画時間は、GPU座標更新・GPU描画完了待ち・ImageBitmap生成を含むCPU単調時計の実測です。GPU timestampによる純粋なGPU時間ではなく、初回のpipeline／グラフバッファ準備は含みません。SGD後の待ちは、最後の座標を受けてから残った描画キューが画面のCanvasへ反映されるまでの待機です。単発実行版の `draw_time_ms` はプロセス起動なども含むため、上表と同じ区間ではありません。

Beforeはランダム配置なので長い辺が画面を大量に横切ります。Afterは辺長や重なりが異なるため、同じグラフでも描画時間が異なります。

render passの実際の提出から記録した頂点・辺数も入力と照合しました。kronは41 draw calls、germany_osmは140 draw calls。PNGの4096 × 4096実寸、拡大後のズーム／全体表示、512と4096を同じサイズに縮小したときの視認性も確認しています。

## 再現

`web/rr-lab` で実行します。`RR_TEST_URL=https://rr.skmtech.jp` を付けると公開サイトを対象にできます。

```sh
GRAPH_FILE=../../data/kron_g500-logn17.mtx PIVOTS=48 ITERATIONS=15 EDGE_LIMIT=0 npm run test:large
GRAPH_FILE=../../data/germany_osm.mtx ITERATIONS=15 EDGE_LIMIT=0 npm run test:large
```

各Canvasの `data-renderer`, `data-render-ms`, `data-draw-calls`, `data-drawn-edges`, `data-drawn-nodes`, `data-gpu-vendor`, `data-gpu-fallback` に実際の経路と計測値を記録します。検証スクリプトはWebGPUの使用、ソフトウェアフォールバックではないこと、全辺数、画像が白くないことも検証します。

## 公開サイトでの確認

同じブラウザから `https://rr.skmtech.jp` に読み込み、両グラフの全辺・全頂点、15反復、拡大操作、4096 × 4096 PNGを確認しました。描画Workerが返す実際のadapter情報は `vendor=apple`, `architecture=metal-3`, `isFallbackAdapter=false` です。

- kron: After描画633.7 ms、SGD後の描画完了待ち974.4 ms、全5,113,981辺・107,901頂点。
- germany_osm: After描画244.0 ms、SGD後の描画完了待ち274.2 ms、全12,369,181辺・11,548,845頂点。

この公開先の確認は別々のChromeプロセスで並行実行しています。単独実行のローカル測定との厳密な速度比較には使用しません。
