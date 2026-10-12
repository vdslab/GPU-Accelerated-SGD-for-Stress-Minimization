# Full Stress比較

| 実装 | Method ID | 役割 |
|---|---|---|
| [baseline-sgd-non-gpu](baseline-sgd-non-gpu/) | `sgd` | CPU逐次Full SGD |
| [vram-lock-native](vram-lock-native/) | `atomic_sgd` | Metal Atomic Full SGD |
| [rr_gpu](rr_gpu/) | `rr_sgd` | WGPU Round-robin Full SGD |

実験条件と結果は [実験一覧](../../experiment_results/README.md)、計測・評価の共通処理は [shared](../../shared/README.md)。
