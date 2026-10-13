# GPU-Accelerated SGD for Stress Minimization

Full / Sparse SGDのGPU並列化と評価。CPU・Atomic・Round-robin方式の比較。

## 実験結果

現行の [v2](experiment_results/v2/README.md) — **14グラフ・6手法・2,260正式実行、全件成功**。
小・中規模の主比較は25 seed、大規模・RR設定比較は10 seed。
速度・stress・shared memory・タイルサイズ・workgroupサイズを比較し、9図をPNG / SVG / PDFで保存。

[結果と考察](experiment_results/v2/report/RESULTS.md) · [集計表](experiment_results/v2/tables/summary.csv) · [図](experiment_results/v2/figures/) · [検証](experiment_results/v2/tables/validation.json)

以前の [v1](experiment_results/v1/README.md) — 16グラフ・6手法・654正式実行の暫定評価（10 / 3 seed）。保存済み結果は維持。
[実験一覧](experiment_results/README.md)

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
│   ├── v2/
│   │   ├── config/
│   │   ├── raw/
│   │   ├── tables/
│   │   ├── figures/
│   │   ├── report/
│   │   └── validation/
│   └── v1/                 以前の実験結果
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

ブラウザでの単発実験: [ローカル実行ページ](tools/single_run/README.md)。

```sh
python3 experiments/serve_single.py --open
```

Live Serverで `tools/single_run/index.html` を開く場合は、別途 `python3 experiments/serve_single.py` を起動する。接続設定内のアイコンで起動コマンドをコピーできる。

リポジトリ直下での単発実行:

```sh
python3 experiments/run_single.py --input data/USpowerGrid.mtx --method rr_sparse_sgd
```

[実験手順](experiments/docs/runbook.md) · [実験概要](experiments/docs/README.md)

## 資料

[論文](paper/README.md) · [データセット](experiments/datasets/README.md) · [移動記録](archive/relocations.json)
