# GPU-Accelerated-SGD-for-Stress-Minimization

実験用のMatrix Marketファイルは `prod_data/` に配置します（Git管理対象外）。
取得元・checksum・取得状況は [データ一覧](experiments/datasets/README.md) を参照してください。

実験結果は [experiment_results/v1](experiment_results/v1/README.md) に保存しています。
元データ、集計表、図、考察を分け、次の条件変更ではv2を作ります。

Atomic Sparse SGDは [vram-lock-sparse](vram-lock-sparse/README.md) に実装しています。
既存の実験runnerから `atomic_sparse_sgd` として実行できます。
