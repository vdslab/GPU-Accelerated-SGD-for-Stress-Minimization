# Sparse SGD pivot距離のCUDA前処理

CPU正解実装の基準commit: `419c89d01cb229b498cb1a5ce87c544a354884bd`．

## 実装

`pivot-preprocess`は既存CPU Sparse SGDのgraph/algorithmを共有する追加CLI．
通常の`baseline-sparse-sgd-non-gpu`も引き続きCPUで動作する．
距離は`u32`，未到達は`u32::MAX`．
Matrix Marketの無向化・自己ループ除去・重複除去・最大連結成分選択は既存コードを共有．
全source検証では最大連結成分を選ばず，非連結グラフと孤立点も比較する．

CUDA backendはGraphBLAST型の疎frontier BFSを専用実装する．GraphBLASTへの依存はない．
CSRをcontext生成時に一度転送し，distance/frontier/next/countを再利用する．
1 warpでfrontierの1頂点の隣接行を展開し，`atomicCAS(INF, level+1)`の成功時だけ次frontierへ追加する．
各levelの終了を同期してからcountを読み戻す．到達不能距離はINFのまま．
各pivotのBFSを逐次実行し，CPU上の既存weighted sampling，nearest_distance更新，領域・重み・制約生成を使う．
通常実行の距離メモリは`h × n`．全点対行列は小グラフ検証のみ．

## ビルドと小グラフ検証

NVIDIA GPU，対応ドライバ，CUDA Toolkit（nvcc，cudart），Rust，C++コンパイラ，arが必要．
`CUDA_HOME`はToolkitのrootを指定する（既定 `/usr/local/cuda`）．`NVCC`も指定可能．
共有ライブラリの探索が必要なら`LD_LIBRARY_PATH`に`$CUDA_HOME/lib64`を追加する．
リポジトリrootから実行する．

```bash
cargo test --release --locked --manifest-path baseline-sparse-sgd-non-gpu/Cargo.toml
CUDA_HOME=/usr/local/cuda cargo test --release --locked --features cuda --manifest-path baseline-sparse-sgd-non-gpu/Cargo.toml
CUDA_HOME=/usr/local/cuda cargo build --release --locked --features cuda --manifest-path baseline-sparse-sgd-non-gpu/Cargo.toml --bin pivot-preprocess
baseline-sparse-sgd-non-gpu/target/release/pivot-preprocess --mode validate --backend cuda
```

小グラフはパス，分岐，サイクル，非連結，孤立点，重複辺・自己ループ，固定seedの乱数グラフ21件．
全sourceのCPU BFSとの完全一致・対角0・対称性・INFを検査する．
CUDAのintegration testでは同じcontextで2回全sourceを実行し，連結fixtureではpivot前処理とSGD最終座標・stressも完全比較する．
GPUが使えない場合は失敗する．CPUへの自動fallbackやGPUテストのskipは行わない．

## 中規模比較

データセットファイルは利用者の既存`bcspwr10.mtx`または`USPowerGrid.mtx`を指定する．
このリポジトリにはデータセットを追加していない．
`compute-sanitizer`をPATHに置き，リポジトリrootから次を実行する．

```bash
python3 scripts/benchmark_cuda_pivots.py --input data/bcspwr10.mtx --seed 0 --repeats 10 --output results/bcspwr10-cuda-pivots
```

スクリプトは順に，small graphのmemcheck/initcheck，実データのCPU/CUDA完全比較，両backendの捨てるwarmup実行，最低10回ずつの計測を行う．
GPU・sanitizerの欠如や検証失敗で測定を中止する．
各CLI実行内でも，そのcontextでsource=0のBFSを一度warmupし，pivot/SGD用RNGは消費しない．
CPU/CUDAの測定順はtrialごとに交互に変える．
最終座標hashを含むfingerprintが全trialで一致しなければ結果を保存しない．

`--mode verify`は全`h × n`距離，全制約，pivot列，初期座標，学習率を直接比較し，CPU上のSGDを両方で実行する．
同じ全stress評価器 `sum(u<v) ((||x_u-x_v||/d_uv)-1)^2`を用いる．
全stressは全source CPU BFSを要するので，中規模では時間がかかる．この検証と評価はtimed trialsの外で一度だけ行う．

出力は`trials.csv`，`summary.json`，`verification.txt`．
summaryはmedian/mean/stdev/min/max，GPU/CPU/toolchain情報，データセットSHA256を記録する．
主speedupはCPU前処理時間のmedian / CUDA前処理時間のmedian．

## 時間の定義

全てhost wall time（秒）．`pivot_distance_s`は純粋なkernel event時間ではなく，初期化，launch，同期，host処理を含むBFS時間から転送時間を引いた値．
`transfer_s`は同期済みの明示的cudaMemcpyのhost時間の合計で，CSR upload，各levelのcount readback，距離行readbackを含む．

| CSV列 | 対象 |
|---|---|
| load_csr_s | 読み込み，既存グラフ構築，最大成分選択，CUDA利用時のCSR変換 |
| setup_nontransfer_s | context/device buffer確保などの時間からCSR転送を除く |
| pivot_selection_s | 連結性チェックと，BFSを除くpivot選択・nearest更新・距離行検査 |
| pivot_distance_s | pivot BFS，転送を除く |
| transfer_s | CSRおよびpivot BFSのCPU-GPU転送，warmup転送を除く |
| constraints_s | 領域・重み・制約，学習率，初期座標生成 |
| sgd_s | 既存CPU Sparse SGD本体 |
| preprocess_s | load + setup + 前処理実測（warmupを除く） |
| pipeline_s | 読み込み開始からSGD終了，warmupを除く |
| warmup_s | 同じcontext上の1 BFS，前処理speedupには含めない |
| cli_s | Pythonから測るCLI全プロセス時間，warmup・起動・出力・解放も含む |

GPU context初期化はsetupに含め，転送や確保を速度比較から隠さない．
CPUには不要なCSR変換を課さない．厳密な比較はhashに加えて`verify`の配列直接比較で行う．

## 今回の実行結果

[VERIFICATION-CUDA.md](VERIFICATION-CUDA.md)を参照．GPU実行の正確性とspeedupは未確認．
