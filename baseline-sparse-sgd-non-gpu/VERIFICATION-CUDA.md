# 検証結果（2026-10-06 JST）

基準commit: `419c89d01cb229b498cb1a5ce87c544a354884bd`．
実行環境: Linux x86_64，Rust 1.99.0，CUDA compiler 12.9.86 / cudart 12.9.79．
NVIDIA GPUデバイスと利用可能なCUDAドライバはない．

| 検証 | 結果 |
|---|---|
| CPU release tests | 51件成功（library 20，既存CLI 24，integration 7） |
| 変更前CPUの固定hashとの比較 | 6頂点パス，seed 0/1/42/999でpivot・距離・制約・初期座標・学習率一致 |
| CPU small graph全source検証 | 21グラフ成功 |
| CPU新CLI，3頂点パス | 成功，SGD 15 iterations，CSV区間時間/hash出力 |
| CUDA release binary build | nvccとRust FFIのコンパイル・リンク成功 |
| CUDA release tests --no-run | 全テストバイナリのコンパイル・リンク成功 |
| clippy，CPU/CUDA all targets | 警告をエラー扱いにして成功 |
| Python benchmark syntax | 成功 |
| 共有graph.rsを使う既存sparse-sgd-gpuのrelease test compile check | 成功 |
| benchmark回数の下限（9回指定） | エラーで停止することを確認 |
| CUDA small graph実行 | 環境により停止（下記） |
| GPU memcheck/initcheck | 未実行 |
| 中規模CPU/CUDA比較 | 小グラフCUDA検証が通っていないため未実行 |

CUDA CLIの実行結果:

```text
Error: CUDA BFS: CUDA driver version is insufficient for CUDA runtime version
```

CUDAが使えない場合にCPUへ切り替えず，非ゼロ終了になることを確認した．
コンパイル成功はGPU上での正確性や安全性の確認を意味しない．
現時点では要件1のGPU検証は未合格で，speedupの数値は存在しない．
GPU環境ではREADMEのrelease CUDA testsとbenchmarkスクリプトを実行する必要がある．
