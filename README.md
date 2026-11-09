# GPU-Accelerated SGD for Stress Minimization

Full / Sparse SGDのGPU並列化と評価。CPU・Atomic・Round-robin方式の比較。

## ブラウザで実行

[**RR-SGD — Round-Robin SGD**](https://vdslab.github.io/GPU-Accelerated-SGD-for-Stress-Minimization/): MTXを読み込み、Full／Sparse SGDと描画をブラウザのWebGPUで実行するサイトです。[ソースと使い方](web/rr-lab/README.md)。

## 実験結果

現行の [v5 実験総覧](experiment_results/v5/README.md) — **正式測定10,180成功、実測失敗0、GPU上限による未実行220条件**。
初回6,030件に加え、大規模・grid/workgroup・EX1（W=T）・EX2（RR128）まで完了。

[RR128とCPU/Atomicの図表](experiment_results/v5/ex-chart-collection-rr128-20261108/README.md) · [EX2の結果と考察](experiment_results/v5/ex2-rr128-speed-stress-20261108/reports/RESULTS.md) · [128対1024](experiment_results/v5/rr128-vs1024-charts-20261108/README.md)

旧結果：[v2](experiment_results/v2/README.md)（2,260正式実行）・[v1](experiment_results/v1/README.md)（654正式実行）。[実験一覧](experiment_results/README.md)。
容量整理後の座標データの所在と再生成上の制約は [v5の保存状態](experiment_results/v5/README.md#現在の保存状態2026-11-09確認) を参照。

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
│   ├── v5/                 現行：初回・大規模・追加実験・EX1・EX2
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
