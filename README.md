# GPU-Accelerated SGD for Stress Minimization

Full / Sparse SGDのGPU並列化と評価。CPU・Atomic・Round-robin方式の比較。

## 実験結果

[v1](experiment_results/v1/README.md) — 16グラフ・6手法・654正式実行。

暫定評価: 小・中規模10 seed、大規模3 seed。`publication_ready=false`。
速度指標: Iteration time。

[考察](experiment_results/v1/report/RESULTS.md) · [集計表](experiment_results/v1/tables/results.md) · [図](experiment_results/v1/figures/) · [実験一覧](experiment_results/README.md)

## 実装

| 群 | 手法 | Method ID | 実装 |
|---|---|---|---|
| Full | CPU SGD | `sgd` | [baseline-sgd-non-gpu](implementations/full/baseline-sgd-non-gpu/) |
| Full | Metal Atomic SGD | `atomic_sgd` | [vram-lock-native](implementations/full/vram-lock-native/) |
| Full | WGPU RR-SGD | `rr_sgd` | [rr_gpu](implementations/full/rr_gpu/) |
| Sparse | CPU Sparse SGD | `sparse_sgd` | [baseline-sparse-sgd-non-gpu](implementations/sparse/baseline-sparse-sgd-non-gpu/) |
| Sparse | Metal Atomic Sparse SGD | `atomic_sparse_sgd` | [vram-lock-sparse](implementations/sparse/vram-lock-sparse/README.md) |
| Sparse | WGPU RR-Sparse SGD | `rr_sparse_sgd` | [sparse-sgd-gpu](implementations/sparse/sparse-sgd-gpu/README.md) |

共通処理: [experiment-common](shared/experiment-common/) — CLI・seed・stress評価・結果記録。

## ディレクトリ構造

```text
.
├── implementations/
│   ├── full/
│   │   ├── baseline-sgd-non-gpu/
│   │   ├── vram-lock-native/
│   │   └── rr_gpu/
│   └── sparse/
│       ├── baseline-sparse-sgd-non-gpu/
│       ├── vram-lock-sparse/
│       └── sparse-sgd-gpu/
├── shared/
│   └── experiment-common/
├── experiments/
│   ├── docs/
│   ├── datasets/
│   ├── manifests/
│   ├── schema/
│   └── tests/
├── experiment_results/
│   └── v1/
│       ├── config/
│       ├── raw/
│       │   ├── v1-quality-timing/
│       │   ├── v1-large-sparse/
│       │   ├── v1-atomic-sparse-quality-timing/
│       │   └── v1-atomic-sparse-large-sparse/
│       ├── tables/
│       ├── figures/
│       └── report/
├── data/
│   └── v1_lcc/
├── paper/
├── tools/
│   ├── gpu_visualizer/
│   └── diagnostics/
│       └── mtx-import-test/
├── archive/
│   ├── implementations/
│   │   ├── vram-lock/
│   │   └── python-sgd/
│   ├── tools/
│   │   └── measure-m4-memory-capacity/
│   └── experiments/
│       ├── legacy-output/
│       └── e0-tables/
├── output/
│   └── <graph>/
│       └── <日時>_<method>/
├── openspec/
│   ├── specs/
│   └── changes/
└── .codex/
    └── skills/
```

`data/`・`output/`・大容量の座標データはGit管理外。

## 実行

リポジトリ直下での単発実行:

```sh
python3 experiments/run_single.py --input data/USpowerGrid.mtx --method rr_sparse_sgd
```

[実験手順](experiments/docs/runbook.md) · [実験概要](experiments/docs/README.md)

## 資料

[論文](paper/README.md) · [データセット](experiments/datasets/README.md) · [移動記録](archive/relocations.json)
