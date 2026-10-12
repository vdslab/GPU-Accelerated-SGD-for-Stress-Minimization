# Sparse Stress比較

| 実装 | Method ID | 役割 |
|---|---|---|
| [baseline-sparse-sgd-non-gpu](baseline-sparse-sgd-non-gpu/) | `sparse_sgd` | CPU逐次Sparse SGD |
| [vram-lock-sparse](vram-lock-sparse/) | `atomic_sparse_sgd` | Metal Atomic Sparse SGD |
| [sparse-sgd-gpu](sparse-sgd-gpu/) | `rr_sparse_sgd` | WGPU Round-robin Sparse SGD |

実験条件と結果は [実験一覧](../../experiment_results/README.md)、計測・評価の共通処理は [shared](../../shared/README.md)。
