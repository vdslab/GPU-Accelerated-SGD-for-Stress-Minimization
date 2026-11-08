## 完了待ち位置の対照実験

同じseed・入力・更新順序で、標準GPUコードの完了待ちを外側ラウンドごとに戻した場合と、現行の反復末尾待ちを比較。各seedでwarmup後5回ずつ、交互の順序で測定。

| seed | ラウンドごと待機 ms/反復 | 反復末尾待機 ms/反復 |
| --- | ---: | ---: |
| 0 | 26.909 | 18.669 |
| 1 | 34.098 | 17.291 |
| 2 | 25.954 | 17.206 |

各seedで最終stressは全測定で一致した。旧記録との差は同じseed内でも存在する。

加えて、旧測定時とSHAが一致する実バイナリをseed 0で再実行した（各条件1回、参考測定）。

current-cli: 17.001794 ms/iteration, stress=6142905.898276006, binary=/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization/implementations/full/rr_gpu/target/release/rr-gpu

old-no-trace: 35.615364 ms/iteration, stress=6142905.898276006, binary=/Users/kose/master/atomic-scheduling-investigation/target/v3-release/release/rr-gpu

old-trace: 38.315297 ms/iteration, stress=6142905.898276006, binary=/Users/kose/master/atomic-scheduling-investigation/target/v3-release/release/rr-gpu

旧測定のseed 0は39.479 ms/反復。今回の旧バイナリ再実行でも当時の時間とは差があり、待機位置の変更だけですべての差を説明できない。実行時負荷・電力状態・warmupなど残る測定条件の寄与は未分離。CPU比24.8倍は旧CPU記録と今回のGPU測定を組み合わせた参考値として扱う。
