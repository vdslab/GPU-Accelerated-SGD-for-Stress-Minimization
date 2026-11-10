# v5.5：RR・Atomic Full実験結果

**EX2完了：RR Full570件＋RR Sparse840件＝1,410件成功、失敗0件。** 当初の1,395件に、容量拡張後のcoPapersCiteseer / p=200の15件を追加した。CPU・Atomicは取り直さず、過去2,820件を参照する。

- [最終結果・図表（補測を含む）](ex2-copapers-capacity-supplement-20261110/reports/results.md)
- [容量拡張補測と検証](ex2-copapers-capacity-supplement-20261110/README.md)
- [当初の測定結果（変更せず保持）](ex2-rr128-speed-stress-20261110/reports/results.md)
- [版の記録](manifest.json)

新規1,410件すべてのStress値と保存座標digestが前回EX2に一致。最新mainへ容量拡張を反映したビルドでも40テストと小規模・大規模入力の一致確認が成功。元1,395件と容量拡張後15件の実装variantは個別に記録する。

EX1は742件での停止後、同じ新Fullの凍結バイナリで残り908件を再開し、1,650/1,650件が成功。元742件を保持し、EX2集計へは混ぜない。

- [EX1：タイルサイズ実験の結果・考察](ex1-tile-w-equal-t-20261110/reports/tile.md)
- [EX1：図表](ex1-tile-w-equal-t-20261110/figures/)
- [EX1：使用グラフ10件の情報](ex1-tile-w-equal-t-20261110/tables/graph-catalog.csv)

大容量の座標・実行バイナリはローカル保持。Gitには既存方針どおり記録・条件・hash・source snapshot・ログ・集計・図表を保存する。

## 最新main Atomic Full 全グラフ

[Atomic Full 全38グラフ・速度とStress](atomic-full-main-20261110/reports/results.md)。38グラフ×5 seed×3回の570/570件成功。v5比のグラフ等重み幾何平均で総計算2.811倍、Stress比0.8502。総計算35/38グラフで高速化、Stress32/38グラフで低下。1万頂点帯の3グラフでは総計算が遅い。GPU完了待ちを含む総計算を主指標とし、段階計時の範囲差も記載。4グラフの予備結果は流用せず全件新規測定。
