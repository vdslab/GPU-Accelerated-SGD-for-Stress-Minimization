"""Write the additional large-graph results into the existing research note."""
import csv,json
from pathlib import Path
p=Path(__file__).resolve().parent
rows=list(csv.DictReader((p/'summary.csv').open()))
datasets=list(dict.fromkeys(r['dataset'] for r in rows))
lookup={(r['dataset'],r['comparison_method']):r for r in rows}
text='''\n\n## 追加実験：13万〜52万頂点（2026-10-27）

10万頂点を超える規模を確認するため、delaunay_n17、loc-Gowalla、delaunay_n18、com-Amazon、delaunay_n19へ拡大した。M4 Pro、15反復・200 pivots・epsilon=0.1は維持。早期の結論を優先し、今回はseed=0を各方式1回（計15実行）とした。実行順は各グラフでCPU→GPUシャッフルあり→GPUシャッフルなし、並行実行なし。専用warm-upなし。

8,000頂点を超えるためstressは共通実装の64始点サンプリングによる全頂点対stressの推定値を使った。評価seed=0、同一グラフでは全方式で同じ始点を使う。前回のexact評価とは区別する。初期配置・前処理・制約・入力のハッシュ一致、全制約更新完了、評価条件一致を全15実行で確認した。

### 反復時間とstress

時間は15反復合計の秒、stressは低いほど良い。今回は各条件1回なので中央値・平均ではなく単回値である（CSVの共通列名はmedian/meanだが1標本）。

| グラフ | 頂点数 | 制約数 | CPU 秒 / stress | GPU shuffleあり 秒 / stress | GPU shuffleなし 秒 / stress | GPUありのCPU比速度倍率 | なし÷あり時間 | なしのstress増加（あり比） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''
for d in datasets:
 a,b,c=[lookup[d,m] for m in ['cpu','gpu_shuffle','gpu_no_shuffle']]
 vals=[f"{float(r['iteration_ms_median'])/1000:.3f} / {float(r['stress_mean']):,.0f}" for r in [a,b,c]]
 text+=f"| {d} | {int(a['nodes']):,} | {int(a['constraints']):,} | {' | '.join(vals)} | {float(a['iteration_ms_median'])/float(b['iteration_ms_median']):.2f}倍 | {float(c['iteration_ms_median'])/float(b['iteration_ms_median']):.2f}倍 | {(float(c['stress_mean'])/float(b['stress_mean'])-1)*100:+.2f}% |\n"
text+='''
### 前処理・GPU初期化込み時間

`algorithm_time_cold_ms`を秒へ換算。前処理・初期化・転送・15反復・readback・centeringを含み、入力・stress評価・結果ファイル書出しは含まない。

| グラフ | CPU 秒 | GPU shuffleあり 秒 | GPU shuffleなし 秒 | dispatch数 あり→なし |
|---|---:|---:|---:|---:|
'''
for d in datasets:
 rs=[lookup[d,m] for m in ['cpu','gpu_shuffle','gpu_no_shuffle']]
 text+=f"| {d} | "+' | '.join(f"{float(r['algorithm_cold_ms_median'])/1000:.3f}" for r in rs)+f" | {rs[1]['dispatches_median']} → {rs[2]['dispatches_median']} |\n"
text+='''
単一seed・単回の速報であり、GPU実行順・seed・熱状態の変動を含む。stressは推定値なので小差の有意性は主張しない。CPUはf64、GPUはf32という既存差も維持されている。今回の規模拡大による結論はこの条件の範囲に限る。

生データ・CSV・実行コマンド・バイナリハッシュ・ソース差分・初期最終座標はリポジトリの `experiment_results/v2/validation/atomic-sparse-no-shuffle-xlarge-20261027/` に保存。再現は同ディレクトリの `run.py`（既存結果を上書きするのでコピー後に実行）。実装は前回から変更せず、commit・pushなし。
'''
report=Path('/Users/kose/Library/Mobile Documents/iCloud~md~obsidian/Documents/sakamoti55/11_大学院研究/01_研究フェーズ/実験/Atomic SparseのCPUシャッフル除去比較.md')
s=report.read_text().split('\n\n## 追加実験：13万〜52万頂点')[0]
report.write_text(s+text)
print(text)
