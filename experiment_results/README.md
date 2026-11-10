# 実験結果

| 実験 | 内容 | 状態 |
|---|---|---|
| [v5](v5/README.md) | 初回6,030＋大規模130＋追加grid810＋EX1 1,800＋EX2 1,410成功 | 10,180成功、実測失敗0、GPU上限未実行220 |
| [v2](v2/README.md) | 14グラフ・6手法・2,260正式実行、shared / タイル / workgroup比較、9図 | 全件成功（主比較25 seed、他10 seed） |
| [v1](v1/README.md) | 16グラフ・6手法・654正式実行 | 暫定評価（10 / 3 seed） |
| [旧実験](../archive/experiments/legacy-output/) | 開発・検証の記録 | アーカイブ |

[実験概要](../experiments/docs/README.md) · [実験手順](../experiments/docs/runbook.md)

```text
<experiment>/
├── config/     条件
├── raw/        生ログ・座標
├── tables/     集計
├── figures/    図
├── report/     考察
└── validation/ 検証ログ・実験後の変更記録（v2）
```

2026-11-08の容量整理で座標・キャッシュ等は削除済み。数値結果・集計・図表は保持。[保存状態と削除証跡](storage-cleanup-20261108/README.md)。

- [v5.5：RR Full/Sparse EX2再実験とcoPapersCiteseer容量拡張補測](v5.5/README.md)
