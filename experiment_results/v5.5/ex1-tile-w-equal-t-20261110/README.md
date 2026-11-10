# v5.5 EX1：タイル実験完了（T=W）

2026-11-10 14:48 JST、1650/1650件成功・失敗0。前回停止時の742件を保持し、同じPR最適化済みFull凍結バイナリで残り908件を再開・完了した。中断した1件の証跡はresume-20261110/interrupted/に保持。全1650件のStressが対応する前回EX1と一致。CPU150件は過去参照であり新規実行していない。EX2は別途完了済みで再実行していない。

共通Tの正規化時間の幾何平均はT=W128が最小（各グラフ最速比1.0512）。T64は1.0616で128との差は約1%。128が各グラフ単独でも最速だったのは5/10件。単一機材・異なる日時の測定で、僅差を統計的優位と断定しない。

- [結果と考察](reports/tile.md)
- [タイル別の集計](tables/tile-cross-graph.json)
- [再開・保持の証跡](resume-20261110/manifest.json)
- [完了検証](validation/final-comparison.json)

以下は開始時の計画説明。EX2未開始という記述は開始時点の履歴であり、現在の状態は上記のとおり。

---

# v5.5 EX1: RR Full W=T

Only RR Full is newly measured:10 graphs x11 T=W x5 seeds x3 repetitions=1650. CPU150 exact-EX1 measurements are historical references, not newly measured. Previous v5 EX1/EX2 files are hashed and never written. EX2 is not started.

Same input bytes, seeds10..14, repetitions0..2,15 iterations,epsilon0.1,shared,both shuffles,phase_chunks1,auto BFS,exact Full Stress. GPU timestamps off, no drawing during measurement. Warmup seed999 per graph/T per session excluded. Native solver binary freshly built from PR head and frozen with source/binary hashes. Sequential, no time cutoff. Stop between runs at <4GiB disk/<2GiB available RAM. Lossless coordinates gzip after measurement. Strict native record validation uses a copied schema extended only for PR BFS metadata.

Run/resume: `python3 experiments/run_v55_ex1.py --run`. Progress: progress.json/PROGRESS.md and the new v5.5 Obsidian note. Reports and PNG/PDF figures generated automatically after all1650 runs; no automatic EX2 transition.
