# RR-SGD

正式表記: **RR-SGD: WebGPU-Accelerated Graph Layout with Round-Robin SGD**

公開URL: https://vdslab.github.io/GPU-Accelerated-SGD-for-Stress-Minimization/

画面の見出しは「RR-SGD — Round-Robin SGD」。見出し横のⓘから概要、Full／Sparseの違い、使い方を確認できます。

MTXファイルの前処理、RR Full / RR Sparse SGD、stress評価をブラウザ内で実行する静的サイトです。計算にはMDNで説明されている[WebGPU API](https://developer.mozilla.org/ja/docs/Web/API/WebGPU_API)を直接使用します。`navigator.gpu.requestAdapter()` → `adapter.requestDevice()` → WGSL compute pipeline → `dispatchWorkgroups()` の経路です。描画もWebGPUのrender pipelineを使い、専用Worker上のOffscreenCanvasへ辺と円形ノードをGPU描画します。Canvas 2Dは完成した画像の画面への合成と、WebGPU非対応環境のフォールバックにのみ使用します。

「現在のグラフ」は左側の読み込み欄に表示し、サンプル／アップロードの種別・名前・使用頂点数・辺数を明記します。サンプル候補や読み込み元のラジオボタンを変えただけでは、読み込み済みのグラフは切り替わりません。読み込み成功後にまとめて切り替え、失敗時は直前のグラフと結果を保持します。

画面は研究リポジトリの `tools/single_run` に合わせています。このフォルダも元の研究リポジトリで一括管理し、独立したGitリポジトリは使用しません。

## 実行

```sh
npm run dev
```

http://localhost:4173 を開きます。アプリ自体の依存パッケージ・ビルドは不要です。WebGPU対応ブラウザとHTTPS（ローカルではlocalhost）が必要です。

サンプルを選択して「読み込む」、または `.mtx` を選びます。この時点ではグラフの読み込みと規模確認だけを行い、初期配置の生成・GPU描画・SGDは開始しません。手法・反復回数・epsilon・seed・Sparseのpivot数を設定して「レイアウトを計算」を押すと、初期配置を表示し、SGDから最終描画までを実行します。計算後はリセットせずにパラメータを編集して再計算できます。再計算は初期配置から開始し、前の結果は計算ボタンを押すまで保持します。一時停止中に同じ設定で再開すると続きから、設定を変更して計算すると初期配置から再計算します。一時停止・再開、前処理の中止、Before / Afterの拡大、結果コピー、座標JSON / PNGの保存に対応します。描画・保存の解像度は512／1024／2048／4096の正方形から選べ、既定は4096 × 4096です。CSS表示サイズや画面のピクセル比から独立した実ビットマップ解像度で、拡大画像・PNGにもそのまま適用します。計算後に解像度を変更しても座標や反復結果は変わりません。ノード径・エッジ幅は論理座標と画面上の最低サイズから決め、高解像度の縮小表示でも視認性を保ちます。PNG保存ボタンはAfter画像の見出し横にあります。Before／Afterの拡大画面でも全体表示・拡大縮小・ドラッグ移動に対応し、拡大画面の操作は元の表示や計算結果を変更しません。

## サンプル

| グラフ         |    頂点 |  無向辺 | 読み込み時の選択手法     |
| -------------- | ------: | ------: | ------------------------ |
| Community      |     240 |     550 | Full                     |
| USpowerGrid    |   4,941 |   6,594 | Full（Sparseも選択可能） |
| Luxembourg OSM | 114,599 | 119,666 | Sparse                   |

`dist/samples/` のMTXはリポジトリの `data/` から原ファイルをコピーしています。頂点の切り詰めやダウンサンプリングはしません。出典・帰属は [samples/README.md](dist/samples/README.md) を参照してください。読み込み時には全入力に共通して最大連結成分を抽出します。

## 責務と処理の流れ

```text
dist/app.js                 画面操作と計算状態の調停
  ui/                       設定、結果表示・保存、Workerクライアント
  render/                   WebGPU描画・WGSL頂点／フラグメントシェーダー
  workers/graph-worker.js   読み込み・前処理・評価のWorker入口
    graph/                  MTX、最大連結成分、CSR/BFS、サンプル、stress評価
    preprocess/full.js      全点対距離と重み範囲
    preprocess/sparse.js    pivot選択、領域、方向別重み、競合なしmatching
  sgd/context.js            WebGPU初期化、buffer所有権、座標の読戻し
  sgd/initialization.js     seed付き初期座標
  sgd/learning-rate.js      学習率スケジュール
  sgd/full-schedule.js      Fullの2段階RRスケジュール
  sgd/full-engine.js        Full GPU投入
  sgd/sparse-engine.js      Sparse GPU投入
  sgd/full.wgsl             Full座標更新
  sgd/sparse.wgsl           Sparse座標更新
```

前処理・評価にDOMやGPU依存はありません。SGDエンジンは読み込み・画面操作を知りません。Workerの通信は `ui/graph-client.js` に閉じ込めています。大きいCPU前処理バッファはGPU転送後に解放します。

## FullとSparse

Fullは `implementations/full/rr_gpu` のタイル間・タイル内RRを移植しています。tile 128、workgroup 64、shared座標。全頂点対を反復ごとに1回更新し、外側・内側の順をシャッフルします。

Sparseは `implementations/sparse/sparse-sgd-gpu` の前処理と2相更新を移植しています。

- 距離に比例したpivotサンプリング。領域の同距離tieは先に選ばれたpivotを優先。
- 方向別重み `s_ip / d_ip²`。元グラフの辺は両方向とも重み1で上書き。
- 片側相は1 invocationが1非pivot頂点を所有し、pivotを読むだけ。pivot行は書き戻さない。
- 両側相はpivotのcircle matchingと、元グラフ辺をfirst-fitで詰めたmatching。各dispatchの両端書き込みは競合しない。
- 頂点ごとのpivot順序はnative版と同じ4段Feistel＋cycle walking。相順・matching順・共通pivot順もシャッフル。
- 学習率は `1 / w_min` から `epsilon / w_max` へ指数減衰。座標更新はf32、atomic・lockは使用しない。

乱数生成器・浮動小数点処理・辺の格納順がRust版と異なるため、Rust版と同じseedでも座標の完全一致は保証しません。SparseをFullと呼び替えたり、辺だけのSGDで代用したりはしていません。

## 規模とメモリ

グラフ読み込み後、距離行列の必要量とGPU上限を比較し、Fullが使えなければFullの選択肢を無効化してSparseへ切り替えます。必要メモリと上限も表示し、小さなグラフに切り替えるとFullを再び選択できます。Sparseのpivot上限も表示します。

4,096頂点の固定制限は撤去しました。実行前にアダプターの制限を確認し、単一GPUバッファは最大1 GiB（アダプター上限が小さい場合はその値）まで要求します。

- Fullの距離行列は `4 × n²` bytes。USpowerGridは約93.1 MiB。LuxembourgのFullは約48.9 GiB必要なので、割り当て前にSparseの利用を案内します。
- Sparseの片側表は `8 × n × pivots` bytes。Luxembourg・200 pivotsは約174.9 MiB。前処理中はさらにpivot距離 `4 × n × pivots` bytes、隣接表等を保持します。
- 大きなグラフの読み込み時は、既存のpivot数がGPU上限を超えていれば収まる値へ変更し、入力欄に実際の値を表示します。読み込み後に指定した範囲外の値はエラー表示します。
- MTXの64 MiB・200万頂点・500万行の固定制限を撤去。File.stream()から行を逐次解析し、端点をUint32Arrayに格納、連結成分をUnion-Find、隣接をCSR、matchingを型付き配列で構築します。実行可能な規模はGPU・ブラウザ・メモリ容量に依存します。
- 既定は全辺・全頂点をWebGPUで描画します。「辺：自動」を選ぶと50万辺以上を最大10万辺に規則的に間引き、表示数／全辺数を明記します。GPU描画では頂点を間引き・画素集約せず、全頂点を円形quadのインスタンスとして描画します。計算には常に全辺を使用します。
- 描画座標・端点・色はGPUバッファに保持し、ズーム／移動時は48-byteのuniformを更新します。初期配置のBefore／Afterは同一GPU画像を共有します。非対応環境ではCanvas 2Dへフォールバックし、実際の描画方式を画面・JSONに記録します。
- 評価距離の保持を128 MiB以内に抑え、始点数を結果に明記します（germany_osmは固定2始点からの推定）。座標JSON保存も小さなBlobに分割し、巨大な単一文字列を作りません。

無向・非重み付きとして処理し、自己ループ・重複辺を除去します。行列の値は使用しません。ファイルはサーバーに送信しません。

## 評価・時間の意味

Full stressは `Σ(i<j) ((||x_i−x_j||−d_ij)/d_ij)²`。

512頂点以下はexact、それ以上は固定・独立seedで一様に選んだ始点を使用します。10,000頂点以下は64始点、それより大きい場合は最大16始点。距離の保持量が128 MiBを超える場合は始点数を減らします。各始点から全頂点までを評価し、`n / (2 × 始点数)` 倍して全点対総和を推定します。BeforeとAfterには同じ始点を使い、画面・JSONにexact / sampledと始点数を明記します。異なるグラフやexactの値と無条件に比較しないでください。

反復経過時間はGPU実行・座標転送・After描画要求のCPU経過時間で、GPU timestampの純計算時間ではありません。前処理・GPU準備・stress計算・フレーム待ち・別Workerでの描画完了待ちは除きます。前処理時間、GPU準備時間、描画完了待ち（renderWaitMs）は別に記録します。

## 検証と整形

```sh
npm install
npm test
npm run test:browser
npm run format:check
npm run format
```

`npm test` はMTX・BFS・Full全点対被覆、Sparseの独立した全点対定義との重み照合、matchingの競合不在、Feistelの全単射性、メモリ上限、stress評価を検証します。

`npm run test:browser` はインストール済みGoogle Chromeをheadlessで起動し、実WebGPUを使用します。必要なら `CHROME_PATH` で実行ファイルを指定します。Sparseの独立CPU更新との座標照合、USpowerGrid Full、全Luxembourg Sparse、Before保持、一時停止・再開、前処理中止・復帰、モバイル表示を確認します。スクリーンショットは既定で `/tmp/rr-graph-lab-qa/`、`SCREENSHOT_DIR` で変更できます。

2026-11-09のローカル実GPU確認では、USpowerGrid Full 15反復とLuxembourg Sparse 200 pivotsの実行を確認。小規模SparseのCPU/GPU座標RMSEは約10⁻⁷〜10⁻⁶でした。性能と対応メモリは端末・ブラウザに依存します。

## GitHub Pagesへの配置

`.github/workflows/pages.yml` が `main` のサイト更新時にアルゴリズムのテストを実行し、`dist/` だけをGitHub Pagesへ公開します。実行時の依存サービスや認証はありません。Node.jsのテストは標準ライブラリのみで実行でき、デプロイ時のnpm installやビルドは不要です。

HTML・ES Modules・Web Workers・WGSL・サンプルのURLは相対パスで解決するため、リポジトリ名を含むPagesのサブパスでも動作します。公開先のホスト名・URLをJavaScriptへ固定する必要はありません。独自ドメインはGitHub Pagesの設定から後で追加できます。

```sh
RR_TEST_BASE_PATH=/GPU-Accelerated-SGD-for-Stress-Minimization/ npm run test:browser
RR_TEST_BASE_PATH=/GPU-Accelerated-SGD-for-Stress-Minimization/ npm run test:load
```

既存のCloudflare配信設定は `wrangler.jsonc` に残しています。`npm run deploy` は従来のCloudflare公開先を更新するコマンドです。GitHub Pagesは上記Actionsワークフローから公開します。

大規模グラフ検証: `GRAPH_FILE=/path/to/germany_osm.mtx ITERATIONS=15 npm run test:large`。既定入力はリポジトリ内のweb-Stanford。データを切り詰めず、読み込み・Sparse計算・Before/Afterの非白色画素を検証します。

大規模実データ確認（2026-11-09、Chrome／実WebGPU）: web-StanfordはLCC 255,265頂点・1,941,926辺、200 pivots・15反復。germany_osmは11,548,845頂点・12,369,181辺、5 pivots・15反復。入力全体を保持し、描画・拡大操作・4096×4096 PNGを確認。GPUバッファ上限128 MiBでもgermany_osmを1 pivotで実行し、二側matchingのバッファ分割を検証しました。

GPU描画の調査と検証条件は [validation/gpu-rendering.md](validation/gpu-rendering.md) を参照。単発実行版は `experiments/run_single.py` からRust/wgpu版の `tools/gpu_visualizer` を呼びます。Web版はブラウザのWebGPUで同じGPU描画の構成を実装しています。

`npm run test:load` はファイル選択・ドロップ・計算後のファイル切り替えで読み込み要求以外が発生しないことをWorker通信で確認し、明示的なキーボード／マウス操作でだけ計算・描画が始まることを検証します。
