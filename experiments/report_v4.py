#!/usr/bin/env python3
"""Write the measured v4 result report only after the complete planned cohort."""
from pathlib import Path
import csv,json,statistics
BASE=Path(__file__).resolve().parents[1]/'experiment_results/v4'
def read(name):return list(csv.DictReader((BASE/'tables'/name).open()))
def number(r,k):return float(r[k])
def main():
 audit=json.loads((BASE/'tables/audit.json').read_text());assert audit['completed']==audit['planned'] and audit['convergence_evaluated']
 summary=read('summary.csv');pairs=read('paired-ratios.csv');runs=read('runs.csv');arrivals=read('arrivals.csv')
 rr=[r for r in summary if r['method']=='rr_sparse_sgd' and int(r['paired_seeds'])];atomic=[r for r in summary if r['method']=='atomic_sparse_sgd' and int(r['paired_seeds'])]
 def description(rows):
  if not rows:return 'CPUとの対応成功条件がなく、速度比・stress比は算出しない。'
  speed=[number(r,'speedup_median') for r in rows];quality=[number(r,'stress_ratio_median') for r in rows]
  return f"比較可能{len(rows)}グラフ中{sum(x>1 for x in speed)}グラフでCPUより高速。速度比中央値の範囲は{min(speed):.3f}〜{max(speed):.3f}倍、stress比中央値は{min(quality):.3f}〜{max(quality):.3f}。速度比>1かつstress比≤1.05は{sum(s>1 and q<=1.05 for s,q in zip(speed,quality))}グラフ。"
 text=f'''# v4 Sparse SGD 論文図の再実験結果

RR Sparseのみ現行実装で再測定し、CPU・Atomic Sparseは既存v3の結果を参照した。Fullの数値もv3を継承する。
標準GPUは単一f32、CPUはf64。丸め誤差の補償、疑似倍精度、追加反復、グラフ別の方式変更は使っていない。

## 実験範囲と条件

図5の既存37グラフと図7のdelaunay_n19を対象にRRのみ5 seed、図6のSparse 3グラフをRRのみ10 seedで測定した。
RR計{audit['planned']}条件の記録が完了し、状態内訳は{audit['statuses']}。CPU・Atomicの参照は{audit['reference_records']}条件で、比較用データ全体の状態内訳は{audit['comparison_statuses']}。
当初3手法を再測定した後、ユーザー指示でRRのみへ変更した。変更前のCPU・Atomic {audit['excluded_auxiliary_v4_records']}条件は補助記録として保存し、以下の比較には混ぜない。
これは以前図に掲載された38グラフの追試であり、v3の78大規模候補すべての追試ではない。
入力はv3と同じ前処理済み最大連結成分をSHA照合した。

全て15反復、epsilon=0.1、pivot=200。基本比較seedは0〜4、収束seedは0〜9。
RR v4は頂点ごとのpivot置換と4分割相実行。参照CPU v3は全制約を毎反復shuffleし、参照Atomic v3もCPUでペアIDを事前shuffleする測定版である。
CPU・Atomicは2026年10月24〜25日の既存測定、RRは10月30日の測定であり、同一セッションの計測ではない。単一機種でも実行時の負荷などによる時間差は残る。Atomicの値は現行既定実装の再測定結果とは呼ばない。
GPUとCPUでは精度と更新順序が異なるため、stress差の全てを丸め誤差だけに帰属しない。
共通f64評価器を使い、8,000頂点以下はexact、それより大きい場合は固定256始点・評価seed 0のfull stress推定値。
今回の38グラフはいずれも推定評価であり、微小な優劣や全頂点対での順位を保証しない。

資源監視はプロセス120秒・観測RSS 8 GiB・空きディスク8 GiB。各設定で事前試行1回を行い、失敗時は正式seedを未実行として記録した。
RSS監視の間隔は約200 msで、終了後の最大RSSも併記する。厳密な瞬時上限ではない。
通常計測には収束checkpoint保存を入れず、収束診断の座標保存・readbackは反復時間から除いた。
図5の1反復平均は15反復の合計÷15、誤差棒は対応seed間の最小〜最大であり、信頼区間ではない。

## CPU v3との時間・stress比較

RR Sparse：{description(rr)}

Atomic Sparse（既存v3参照）：{description(atomic)}

| グラフ | CPU/RR対応seed数 | CPU 1反復 ms | RR 1反復 ms | CPU/RR速度比 | RR/CPU stress比 |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
 times=read('fig05_iteration_times.csv')
 for r in sorted(rr,key=lambda r:r['graph']):
  cpu=next(t for t in times if t['graph']==r['graph'] and t['method']=='CPU');gpu=next(t for t in times if t['graph']==r['graph'] and t['method']=='RR')
  text+=f"| {r['graph']} | {r['paired_seeds']} | {float(cpu['median_ms_per_iteration']):.3f} | {float(gpu['median_ms_per_iteration']):.3f} | {float(r['speedup_median']):.3f} | {float(r['stress_ratio_median']):.3f} |\n"
 text+='\n速度比・stress比は同seedで比を求めてから中央値を取る。時間の中央値同士の比とは一般に一致しない。\n\n![図5 CPUとRRの1反復平均時間](../figures/fig05_comparison.png)\n\n## stress推移\n\n15反復用の学習率を維持し、0・1・2・3・5・8・10・15反復の配置を評価した。\n線は同手法のstress中央値を対応するCPU最終stress中央値で割った値であり、単一seedの軌跡ではない。\n\n| グラフ | 手法 | CPU対応成功数 | 1.05倍目標への到達観測 | 最終時点でも目標内 | 初回到達観測の累積SGD時間中央値 s |\n| --- | --- | ---: | ---: | ---: | ---: |\n'
 for g in ['luxembourg_osm','kron_g500-logn18','coAuthorsCiteseer']:
  for m in ['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd']:
   rs=[r for r in arrivals if r['graph']==g and r['method']==m];yes=[r for r in rs if r['reached']=='True'];final=sum(r['final_within_target']=='True' for r in rs);ts=statistics.median(float(r['first_observed_ms'])/1000 for r in yes) if yes else None
   text+=f"| {g} | {m} | {len(rs)} | {len(yes)} | {final} | {ts:.3f} |\n" if ts is not None else f"| {g} | {m} | {len(rs)} | {len(yes)} | {final} | — |\n"
 text+='\n保存点間の到達・再上昇は観測していないため、厳密な初到達時間ではない。\n\n![図6 stress推移](../figures/fig06_convergence.png)\n\n## 大規模配置：delaunay_n19\n\n| 手法 | 正式完了数 | 15反復SGD時間 s | 入力・評価・保存を含むプロセス時間 s | 観測最大RSS GiB |\n| --- | ---: | ---: | ---: | ---: |\n'
 for m in ['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd']:
  rs=[r for r in runs if r['study']=='basic' and r['graph']=='delaunay_n19' and r['method']==m and r['status']=='success']
  def span(k,unit=1):
   values=[float(r[k])/unit for r in rs];return f'{min(values):.3f}〜{max(values):.3f}' if values else '—'
  text+=f"| {m} | {len(rs)}/5 | {span('iteration_time_ms',1000)} | {span('process_wall_seconds')} | {span('peak_rss_bytes',1024**3)} |\n"
 missing=read('missing.csv')
 text+='\n## 非成功条件\n\n| 実験 | グラフ | 手法 | 状態 | 件数 |\n| --- | --- | --- | --- | ---: |\n'
 from collections import Counter
 for (study,g,m,status),count in sorted(Counter((r['study'],r['graph'],r['method'],r['status']) for r in missing).items()):
  text+=f'| {study} | {g} | {m} | {status} | {count} |\n'
 text+='\n## 速度と品質の全対象比較\n\n![Sparse v4 speed and stress](../figures/sparse_speed_stress.png)\n\n同seedのCPUに対する比の中央値と最小〜最大。上段は大きいほど高速、下段は小さいほどstressが低い。下段の点線は1.05。×は対応成功seedがない条件を表す。全38対象を表示し、成功したGPUだけの選抜はしない。\n'
 text+='''
図7は全ての正式完了RR配置を示す。CPUに結果が得られた場合は「CPUでは処理不能」とは表現しない。
Full delaunay_n14の上段はv3の既存結果である。

![図7 大規模グラフの配置](../figures/fig07_large_layouts.png)

## データの所在

`config/`に計画・入力・ソース・バイナリSHA、`raw/results.jsonl`に全条件、`warmups/`に事前試行、
`tables/missing.csv`に欠測理由、`tables/paired-ratios.csv`にseedごとの対応比較、
`tables/convergence.csv`にcheckpointのstress、`figures/`にPNG/PDF/SVGを保存した。
Fullの出所は`reference/full-v3/`、Sparse CPU・Atomicの出所は`reference/sparse-v3/`。
新規RRの計画は`config/rr-only-plan.json`、変更前の計画は`config/plan.json`、
比較から除いた途中までのCPU・Atomic v4は`tables/excluded-auxiliary-v4.csv`に保存した。
'''
 (BASE/'report').mkdir(exist_ok=True);(BASE/'report/RESULTS.md').write_text(text)
 print('Wrote',BASE/'report/RESULTS.md')
if __name__=='__main__':main()
