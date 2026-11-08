"""Build a manuscript-oriented gallery from completed measurements, without new runs."""
import collections
import csv
import gzip
import hashlib
import json
import lzma
import math
from pathlib import Path
import shutil
import statistics as stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import numpy as np

from run_v5 import ROOT, VAULT, save, now
from speed_plot_style import iteration_figure, speedup_figure

BASE=ROOT/'experiment_results/v5'
OUT=BASE/'paper'
VFIG=VAULT/'v5_図表/paper'
CPU={'sgd','sparse_sgd'}
LABELS={'sgd':'CPU Full SGD','atomic_sgd':'Atomic Full SGD','rr_sgd':'RoundRobin Full SGD',
        'sparse_sgd':'CPU Sparse SGD','atomic_sparse_sgd':'Atomic Sparse SGD','rr_sparse_sgd':'RoundRobin Sparse SGD'}
COLORS={'sgd':'#404040','sparse_sgd':'#404040','atomic_sgd':'#d55e00','atomic_sparse_sgd':'#d55e00','rr_sgd':'#0072b2','rr_sparse_sgd':'#0072b2'}
INTERVALS=[('①','前処理','preprocess_time_ms'),('②','GPU準備','gpu_prepare_time_ms'),
 ('③','初期転送','upload_time_ms'),('④','各反復の計算時間','iteration_times_ms'),
 ('⑤','SGD経過時間','sgd_time_ms'),('⑥','結果取得','readback_time_ms'),
 ('⑦','描画','draw_time_ms'),('⑧','描画込み総時間','total_with_draw_time_ms'),
 ('⑨','反復計算時間合計','iteration_total_time_ms'),('⑩','描画なし総計算時間','total_compute_time_ms')]


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def distribution(values):return dict(median=stats.median(values),min=min(values),max=max(values))
def csvwrite(path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
 OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True);(OUT/'data').mkdir(exist_ok=True);VFIG.mkdir(parents=True,exist_ok=True)
 rows=[]
 for stage,folder in [('tile',BASE/'raw/tile'),('comparison',BASE/'raw/comparison'),('large',BASE/'large/raw')]:
  for p in folder.glob('*/record.json'):
   r=read(p)
   if r['status']=='success':rows.append(dict(r,stage=stage,record_path=str(p)))
 assert collections.Counter(r['stage'] for r in rows)=={'tile':1800,'comparison':4230,'large':130}
 normal=[r for r in rows if r['stage']=='comparison'];large=[r for r in rows if r['stage']=='large']
 hierarchical=read(BASE/'tables/comparison-hierarchical.json');tile=read(BASE/'tables/tile-hierarchical.json')
 cross=read(BASE/'tables/comparison-cross-graph.json');tilecross=read(BASE/'tables/tile-cross-graph.json')
 summary=read(BASE/'large/tables/summary.json');graphs=read(BASE/'large/config/datasets.json')
 audit=[]
 for number,label,key in INTERVALS:
  item=dict(number=number,name=label,field=key)
  for group,subset in [('cpu',[r for r in rows if r['method'] in CPU]),('gpu',[r for r in rows if r['method'] not in CPU])]:
   present=[r for r in subset if r.get(key) is not None]
   if key=='iteration_times_ms':assert all(len(r[key])==15 for r in present)
   item[group+'_measured']=len(present);item[group+'_total']=len(subset)
  audit.append(item)
 save(OUT/'data/timing-coverage.json',dict(at=now(),successful_runs=6160,unexecuted_gpu_limit=220,intervals=audit))
 csvwrite(OUT/'data/timing-coverage.csv',audit)
 plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 figures=[]
 def export(fig,name,title,caption,data,priority):
  fig.savefig(OUT/'figures'/(name+'.png'),dpi=190,bbox_inches='tight');fig.savefig(OUT/'figures'/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
  shutil.copy2(OUT/'figures'/(name+'.png'),VFIG/(name+'.png'))
  save(OUT/'data'/(name+'.json'),data)
  figures.append(dict(name=name,title=title,caption=caption,priority=priority))

 # F1: do not claim statistically established scale-free classes.
 names=sorted({r['graph'] for r in tile});fig,axes=plt.subplots(1,2,figsize=(11,3.7));data=[]
 for i,name in enumerate(names):
  rs=sorted([r for r in tile if r['graph']==name and r['method']=='rr_sgd'],key=lambda r:r['tile_size']);best=min(r['iteration_time_ms']['median'] for r in rs);color=plt.cm.tab10(i)
  for r in rs:data.append(dict(graph=name,tile=r['tile_size'],normalized_sgd=r['iteration_time_ms']['median']/best,stress_ratio=r['stress_ratio']['median']))
  axes[0].plot([r['tile_size'] for r in rs],[r['iteration_time_ms']['median']/best for r in rs],'.-',lw=1,color=color,label=name)
  axes[1].plot([r['tile_size'] for r in rs],[r['stress_ratio']['median'] for r in rs],'.-',lw=1,color=color)
 for ax in axes:
  ax.set_xscale('log',base=2);ax.set_xticks([1,4,16,64,128,256,1024],[1,4,16,64,128,256,1024]);ax.set_xlabel('Tile size T');ax.grid(alpha=.2);ax.axvline(1024,color='#777',ls=':',lw=.8)
 axes[0].set_yscale('log');axes[0].set_ylabel('SGD time / best T for each graph');axes[1].set_ylabel('Final Stress / same-run CPU Stress');axes[1].axhline(1,color='#777',lw=.8,ls='--')
 fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,fontsize=7);fig.tight_layout(rect=(0,.18,1,1))
 export(fig,'f01-tile','タイルサイズの速度・品質の交換関係','10グラフ、T=1〜1024、seed10〜14×各3回。seed内3回中央値→5seed中央値。左は各グラフの最速Tで正規化、右は対応CPU比を先に計算して集計。T1024の点線は通常比較に採用した条件。線は傾向を読むためで、スケールフリー性の実証やTの因果効果の分離ではない。',data,'本文候補')

 # Absolute mean iteration time, using the same 15-iteration measurements.
 data=[dict(graph=r['graph'],nodes=r['nodes'],method=r['method'],
            median_ms=r['iteration_time_ms']['median']/15,
            min_ms=r['iteration_time_ms']['min']/15,
            max_ms=r['iteration_time_ms']['max']/15) for r in hierarchical]
 export(iteration_figure(hierarchical),'f00-iteration-runtime','平均1反復の実行時間',
        'Full38件・Sparse56件。SGD15反復の経過時間を15で割った平均1反復時間。CPUは×、Atomicは△、RoundRobinは○。点はseed内3回中央値→5seed中央値、縦棒は5seedの最小〜最大。両対数軸、測定点間の接続なし。',data,'本文候補')

 # F2: all graphs, including GPU slowdowns, same-family CPU denominator.
 fig=speedup_figure(hierarchical)
 data=[dict(graph=r['graph'],nodes=r['nodes'],method=r['method'],metric=metric,**r[metric])
       for r in hierarchical if r['method'] not in CPU
       for metric in ['sgd_speedup','total_speedup']]
 export(fig,'f02-runtime','SGDと総計算時間の速度比較','Full38件、Sparse56件をすべて表示。縦軸は同族CPU時間/GPU時間、1より上で高速。graph・seed・repetition対応比→seed内3回中央値→5seedの中央値と最小最大。総計算時間は入力読込からCPU上の最終座標までで、Stress評価・保存・描画は含めない。両対数軸。小規模の低速化を省略していない。',data,'本文候補')

 fig,axes=plt.subplots(1,2,figsize=(11,3.6));data=[]
 for ax,family in zip(axes,['full','sparse']):
  methods=['atomic_sgd','rr_sgd'] if family=='full' else ['atomic_sparse_sgd','rr_sparse_sgd']
  for method in methods:
   rs=sorted([r for r in hierarchical if r['method']==method],key=lambda r:r['nodes'])
   ax.errorbar([r['nodes'] for r in rs],[r['stress_ratio']['median'] for r in rs],yerr=[[r['stress_ratio']['median']-r['stress_ratio']['min'] for r in rs],[r['stress_ratio']['max']-r['stress_ratio']['median'] for r in rs]],fmt='o' if method.startswith('rr') else '^',markersize=3,capsize=1,alpha=.8,color=COLORS[method],label=LABELS[method])
   for r in rs:data.append(dict(graph=r['graph'],nodes=r['nodes'],method=method,**r['stress_ratio']))
  ax.set_xscale('log');ax.axhline(1,color='#777',ls='--',lw=.8);ax.set_xlabel('Vertices (different graphs)');ax.set_ylabel('Final Stress / same-family CPU Stress');ax.set_title(family.capitalize());ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.tight_layout()
 export(fig,'f03-quality','同じ実行から取得した最終Stress','速度と同じ4230実行。対応比→seed内3回中央値→5seed中央値と最小最大。n≤8000は厳密Full Stress、それ以外は256始点・評価seed0の推定Full Stress。1より上はStress増加。Full/Sparseの分母が異なるので両パネルの値を直接比べない。固定15反復後の値であり、収束到達や最適解の品質ではない。',data,'本文候補')

 repeats=collections.defaultdict(list)
 for r in normal:repeats[(r['graph'],r['method'],r['seed'])].append(r['stress_value'])
 repeat_data=[];fig,axes=plt.subplots(1,2,figsize=(10,3.5))
 for ax,methods in zip(axes,[['sgd','atomic_sgd','rr_sgd'],['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd']]):
  for method in methods:
   values=[]
   for (graph,m,seed),v in repeats.items():
    if m==method:
     assert len(v)==3;value=(max(v)-min(v))/stats.median(v);values.append(value);repeat_data.append(dict(graph=graph,method=m,seed=seed,relative_stress_range=value))
   x=np.sort(np.array(values)*100);y=np.arange(1,len(x)+1)/len(x);ax.step(x,y,where='post',color=COLORS[method],label=LABELS[method],ls=':' if method in CPU else '-' if method.startswith('rr') else '--')
  ax.set_xscale('symlog',linthresh=.001);ax.set_xlabel('Within-seed Stress range / median (%)');ax.set_ylabel('Fraction of graph-seed groups');ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.tight_layout()
 export(fig,'f04-repeatability','同seed内3回のStressの再現性','Full各方式190群、Sparse各方式280群。(最大−最小)/中央値の経験累積分布。CPU・RoundRobinは全群でStressが3回一致し、Atomicは全群で変動した。これは観測したStressの再現性であり、全GPUや全機材のビット単位決定性の保証ではない。ゼロを表示できる対称対数軸。',repeat_data,'補足・本文の再現性主張用')

 shares=collections.defaultdict(list)
 for r in normal:shares[(r['graph'],r['method'],r['seed'])].append(100*r['sgd_time_ms']/r['total_compute_time_ms'])
 share_data=[];fig,axes=plt.subplots(1,2,figsize=(10,3.5))
 for ax,family in zip(axes,['full','sparse']):
  methods=['sgd','atomic_sgd','rr_sgd'] if family=='full' else ['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd']
  for method in methods:
   rs=sorted([r for r in hierarchical if r['method']==method],key=lambda r:r['nodes']);ys=[]
   for r in rs:
    v=[stats.median(shares[(r['graph'],method,seed)]) for seed in range(5)];d=dict(graph=r['graph'],nodes=r['nodes'],method=method,**distribution(v));share_data.append(d);ys.append(d['median'])
   ax.scatter([r['nodes'] for r in rs],ys,s=12,color=COLORS[method],marker='o' if method.startswith('rr') else '^' if method.startswith('atomic') else 'x',label=LABELS[method])
  ax.set_xscale('log');ax.set_ylim(0,105);ax.set_xlabel('Vertices (different graphs)');ax.set_ylabel('SGD elapsed / total compute (%)');ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.tight_layout()
 export(fig,'f05-bottleneck','SGD高速化が処理全体へ及ぼす範囲','各runの⑤SGD経過時間/⑩総計算時間を求め、seed内3回中央値→5seed中央値。純GPU時間ではない。残りの時間をGPU待ち・転送など特定の要因へ一括帰属しない。複数の重複したタイマーを足した積み上げ棒グラフにはしていない。',share_data,'補足・前処理ボトルネックの説明用')

 pivots=[1000,500,200,100,50];si={(r['graph'],r['pivots']):r for r in summary};fig,ax=plt.subplots(figsize=(9,5.6));matrix=[]
 for g in graphs:matrix.append([1 if si[(g['name'],p)]['success']==5 else 0 for p in pivots])
 ax.imshow(matrix,cmap=ListedColormap(['#dddddd','#b5e7ce']),vmin=0,vmax=1,aspect='auto')
 for i,g in enumerate(graphs):
  for j,p in enumerate(pivots):ax.text(j,i,'5/5' if matrix[i][j] else 'GPU limit',ha='center',va='center',fontsize=7)
 ax.set_xticks(range(5),pivots);ax.set_yticks(range(14),[g['name'] for g in graphs]);ax.set_xlabel('Pivot count');ax.set_title('RoundRobin Sparse SGD: measured success / 5 seeds')
 fig.tight_layout()
 export(fig,'f06-feasibility','大規模実験の成功とGPU上限','14グラフ×5p×5seedの350条件。130実測成功、220は照会済みstorage binding/dispatch上限で確保前に未実行。灰色は実行失敗ではない。CPU/Atomicはこの規模で測定していないため、CPUの一般的不可能性やCPU速度比を示さない。',summary,'本文候補')

 idx={(r['graph'],r['pivots'],r['seed']):r for r in large};fig,axes=plt.subplots(1,3,figsize=(13,4));data=[]
 for i,g in enumerate(graphs):
  ps=sorted(p for p in pivots if si[(g['name'],p)]['success']==5)
  if not ps:continue
  bp=max(ps);d=[]
  for p in ps:
   vals=[]
   for seed in range(5):
    r=idx[(g['name'],p,seed)];ref=idx[(g['name'],bp,seed)];a=dict(graph=g['name'],pivots=p,seed=seed,reference_pivots=bp,total_ratio=r['total_compute_time_ms']/ref['total_compute_time_ms'],stress_ratio=r['stress_value']/ref['stress_value'],rss_gib=r['peak_rss_bytes']/2**30);data.append(a);vals.append(a)
   d.append({key:distribution([a[key] for a in vals]) for key in ['total_ratio','stress_ratio','rss_gib']})
  for ax,key in zip(axes,['total_ratio','stress_ratio','rss_gib']):
   med=[a[key]['median'] for a in d];ax.errorbar(ps,med,yerr=[[a[key]['median']-a[key]['min'] for a in d],[a[key]['max']-a[key]['median'] for a in d]],fmt='.-',lw=.8,capsize=2,color=plt.cm.tab10(i),label=g['name'])
 for ax,label in zip(axes,['Total compute / same-seed reference','Stress / same-seed reference','Peak process RSS (GiB)']):
  ax.set_xscale('log');ax.set_xticks([50,100,200,500,1000],[50,100,200,500,1000]);ax.minorticks_off();ax.set_xlabel('Pivot count');ax.set_ylabel(label);ax.grid(alpha=.2)
 axes[0].axhline(1,color='#777',lw=.8,ls='--');axes[1].axhline(1,color='#777',lw=.8,ls='--');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,fontsize=7);fig.tight_layout(rect=(0,.2,1,1))
 export(fig,'f07-large-pivots','大規模のpivot数と時間・品質・RSS','各グラフで5seed成功した最大pを基準に、同seedの比を先に計算し、5seedの中央値と最小最大を表示。Stressは64始点・評価seed0。p間では初期座標fingerprintも変わるため、p単独の因果効果とは断定しない。RSSは専用GPUメモリ量ではない。各seed1回。',data,'本文候補')

 # F8: deliberately show worst graph-median RoundRobin quality case per family.
 def sample(record):
  path=Path(record['final_positions_path']);opener=gzip.open if path.suffix=='.gz' else lzma.open if path.suffix=='.xz' else open
  n=record['nodes'];m=record['edges'];points=np.empty((n,2));edges=[];stride=max(1,math.ceil(m/20000));state='';ei=vi=0
  with opener(path,'rt') as f:
   for line in f:
    if line.startswith('# Edges'):state='edges';continue
    if line.startswith('# Positions'):state='points';continue
    if not line.strip() or line.startswith('#'):continue
    if state=='edges':
     if ei%stride==0:edges.append(tuple(map(int,line.split())))
     ei+=1
    elif state=='points':points[vi]=[float(x) for x in line.split()];vi+=1
  assert ei==m and vi==n and np.isfinite(points).all()
  return points,np.asarray(edges,dtype=int),stride
 fig,axes=plt.subplots(2,3,figsize=(10,6.7));data=[]
 for i,(rr,methods) in enumerate([('rr_sgd',['sgd','atomic_sgd','rr_sgd']),('rr_sparse_sgd',['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd'])]):
  worst=max((r for r in hierarchical if r['method']==rr),key=lambda r:r['stress_ratio']['median']);graph=worst['graph'];rs=[next(r for r in normal if r['graph']==graph and r['method']==method and r['seed']==0 and r['repetition']==0) for method in methods]
  assert len({r['initial_positions_sha256'] for r in rs})==1
  draws=[sample(r) for r in rs];lo=np.min([pts.min(axis=0) for pts,_,_ in draws],axis=0);hi=np.max([pts.max(axis=0) for pts,_,_ in draws],axis=0);pad=np.maximum(hi-lo,1e-8)*.03
  for ax,r,(pts,edges,stride) in zip(axes[i],rs,draws):
   ax.add_collection(LineCollection(pts[edges],colors='#999999',lw=.15,alpha=.25,rasterized=True));vstride=max(1,math.ceil(len(pts)/50000));p=pts[::vstride];ax.scatter(p[:,0],p[:,1],s=.2,color='#0072b2',rasterized=True)
   ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1]);ax.set_aspect('equal');ax.axis('off');ax.set_title(LABELS[r['method']]+'\n'+graph,fontsize=8)
   ax.text(.02,-.02,f'Total {r["total_compute_time_ms"]/1000:.3g} s; Stress / CPU {r["stress_value"]/rs[0]["stress_value"]:.3f}',transform=ax.transAxes,fontsize=7)
   data.append(dict(graph=graph,method=r['method'],seed=0,repetition=0,run_id=r['run_id'],stress_ratio=r['stress_value']/rs[0]['stress_value'],vertices_drawn=len(p),vertex_stride=vstride,edges_drawn=len(edges),edge_stride=stride,selection='Worst graph-median RoundRobin Stress ratio in each family; fixed seed0 repetition0, not selected for appearance',scale='Same coordinate scale across three methods within row; no rotation/alignment'))
 fig.tight_layout()
 export(fig,'f08-layout-comparison','品質差が大きかった入力の配置比較','Full/SparseそれぞれRoundRobinのグラフ別Stress比中央値が最大だった入力を選び、seed0・repetition0を固定。表示した時間・比はこの1実行の値で、主図の5seed中央値とは異なる。同族3方式の初期配置一致を確認、行内で共通スケール、回転や位置合わせなし。50k頂点/20k辺以下の決定的間引き。見栄えのよい入力だけを選んだ図ではない。',data,'本文候補・定性的な限界の確認')

 selected=['auto','Flan_1565','com-Orkut','great-britain_osm'];fig,axes=plt.subplots(2,2,figsize=(9,9))
 for ax,name in zip(axes.ravel(),selected):ax.imshow(plt.imread(BASE/'large/figures'/('large-layout-'+name+'.png')));ax.axis('off')
 fig.tight_layout(pad=.2)
 export(fig,'f09-large-layouts','大規模入力の描画例','auto、Flan_1565、com-Orkut、great-britain_osm。各グラフで5seed成功した最大pのseed0。保存済みレイアウト図を配置し、測定・再レイアウトは行わない。100k頂点/50k辺以下に決定的間引き、全座標のextentを保持し、グラフ間は独立スケール。CPUとの品質比較ではない。',[a for a in read(BASE/'large/tables/layout-sampling.json') if a['graph'] in selected],'補足・大規模の実例')

 # Tables supporting the concise synthesis, all graphs retained.
 csvwrite(OUT/'data/cross-graph-summary.csv',[dict(method=r['method'],graphs=r['graphs'],sgd_speedup_gmean=r['sgd_speedup']['gmean'],total_speedup_gmean=r['total_speedup']['gmean'],stress_ratio_gmean=r['stress_ratio']['gmean']) for r in cross if r['method'] not in CPU])
 save(OUT/'manifest.json',dict(at=now(),new_layout_runs=0,normal_runs=4230,tile_runs=1800,large_successes=130,large_gpu_unexecuted=220,figures=figures,
  sources=[dict(path=str(p),sha256=digest(p)) for p in [BASE/'tables/comparison-hierarchical.json',BASE/'tables/tile-hierarchical.json',BASE/'tables/comparison-cross-graph.json',BASE/'large/tables/summary.json']],
  timing_audited_successes=6160,aggregation='Normal/tile: paired run ratios, median of 3 repetitions within seed, then median and min/max across 5 seeds. Large: one run per seed.'))
 write_notes(figures,audit,cross,tilecross)
 print(json.dumps(dict(figures=len(figures),timing_successes_audited=len(rows),new_layout_runs=0,obsidian=str(VAULT/'5_実験結果まとめと論文用チャート.md')),ensure_ascii=False))


def write_notes(figures,audit,cross,tilecross):
 title='# 実験結果の要点と論文用チャート\n\n'
 intro='通常比較4,230実行・タイル1,800実行・大規模130実行が成功。大規模の220条件はGPU/現実装の上限による未実行として区別する。今回の図の作成で新規レイアウト実行は0。旧実験の図・本文へ置き換え可能な実測図を集めた。\n\n'
 text=title+intro+'## 研究として言えそうなこと\n\n'
 text+='1. **RoundRobinによる更新スケジュールは、十分な仕事量のある入力でSGD更新を高速化できる。** ただし小規模では固定費が勝ち、処理全体の一律高速化は言えない。全グラフを残したF2を主結果にする。\n2. **速度と最終品質の交換関係を示せる。** FullのTとSparseのpの変更に応じて時間・最終Stressが変化した。ただし固定15反復、異なる初期配置等の影響が残るため、最適解品質や収束速度とは言わない。\n3. **同seedの繰り返しで最終Stressが安定したという観測がある。** CPU/RoundRobinでは3回一致、Atomicでは全graph×seed群で変動。ただしビット単位決定性や原因がlock競合だけであることの証明ではない。\n4. **単一GPU・指定条件で数百万頂点を処理できた。** 最大は英国道路網7,733,822頂点、p50、5seed成功。CPU比較をしていない大規模段階でCPU速度比やCPUの一般的不可能性は言えない。\n5. **SGD以外が主要なコストとして残る。** 大規模のSGD/総計算時間はrun単位比の中央値2.42%。更新カーネル高速化を、レイアウト全体の同じ倍率の高速化と取り違えない。\n\n'
 text+='| 方式 | 対象数 | SGDのCPU速度比・幾何平均 | 総計算のCPU速度比・幾何平均 | 最終Stress/CPU・幾何平均 |\n|---|---:|---:|---:|---:|\n'
 for r in cross:
  if r['method'] not in CPU:text+=f'| {LABELS[r["method"]]} | {r["graphs"]} | {r["sgd_speedup"]["gmean"]:.3f} | {r["total_speedup"]["gmean"]:.3f} | {r["stress_ratio"]["gmean"]:.4f} |\n'
 text+='\n速度比は1より大きいと高速、Stress比は1より大きいと悪化。各グラフを同じ重みで集約した記述量であり、有意差や同等品質の保証ではない。FullはCPU Full、SparseはCPU Sparseが分母。Sparse専用18件ではRoundRobinのSGD35.40倍・総計算4.16倍・Stress比1.0377だが、この部分集合だけで全対象の結論を置き換えない。\n\n'
 text+='## 気になるところ・主張の限界\n\n'
 text+='- **Fullの品質差が大きい入力がある。** RoundRobin FullのStress/CPUは全38件で幾何平均1.0656、最大wiki-Voteの1.6710。Sparseも最大email-EuAllで1.1552。「品質を維持した高速化」を無条件に主張しない。\n- **総計算の全対象平均は控えめ。** RoundRobin Fullは0.630倍、Sparseは1.021倍。n≥1000ではそれぞれ3.32倍・2.61倍だが、これは事後の補足集計で、都合のよい切り取りにしない。\n- **通常比較のT1024は性能最適値ではない。** タイル実験ではT128〜256が速度上良好。T128は各グラフ最速より幾何平均2.9%増、T1024は104.0%増。ユーザー指定条件の結果であり、最適Tを採用した性能とは書かない。\n- **Tに対する構造差の説明は仮説。** 5対5は構造候補で、add20は回路グラフ。厳密なスケールフリー分類や次数分布の因果実験とは呼ばない。\n- **初期条件と目的関数を区別。** 同族CPU/Atomic/RoundRobinの初期配置は一致、Sparse前処理も一致。Full対Sparse、および大規模のp間では同seedだけで初期配置は一致せず、複数pが成功した40 graph×seed群すべてでfingerprintが異なった。アルゴリズム・設定全体の比較として読む。\n- **純GPU時間と描画時間は欠測。** 取得状況は次表。⑤のホスト経過時間を⑨のGPU計算pass合計と呼ばない。\n- **固定15反復・5初期条件。** 同seed3回を15独立seedとして扱わない。推定Stressの標準誤差は出力されておらず、小差を統計的に確定しない。通常は256始点、大規模は64始点。途中収束・time-to-qualityは未測定。\n- **機材・実装への依存。** Apple M4 Pro 24GiBの観測。GPU上限220条件は現バッファ設計/dispatch上限で、GPU一般の不可能性ではない。RSSを専用GPUメモリ量と呼ばない。容量中断・再ウォームアップ・常駐Codexを含む。\n- **復旧と再試行は分離。** 保存時容量停止の再試行前1件を本測定に混ぜていない。成功出力を復旧したLiveJournalの1件は終了code/終了直後RAMが欠測で、timeのwallは0.01秒精度。native計算値・Stressは保持した。\n\n'
 text+='方式比較はスケジュールだけを差し替えた切り分け実験ではなく、各実装全体の比較である。CPUと今回のAtomic以外の全GPU手法に対する優位性も、この比較からは主張できない。\n\n'
 text+='## 事前に話した計時9項目の取得状況\n\n**全9項目を取得できている、とは言えない。** 定義は既存の「9_計測時間」の①〜⑨と追加⑩に対応する。GPU timestamp確保エラーを避けるため全GPU本測定で無効化し、描画も本測定と切り離した。CPUの②③⑥は対象外であり欠測として失敗扱いしない。\n\n| 区間 | 保存field | CPU（1560成功） | GPU（4600成功） |\n|---|---|---:|---:|\n'
 for a in audit:
  cpu='対象外（GPU区間）' if a['number'] in ['②','③','⑥'] else f'{a["cpu_measured"]}/{a["cpu_total"]}'
  text+=f'| {a["number"]} {a["name"]} | `{a["field"]}` | {cpu} | {a["gpu_measured"]}/{a["gpu_total"]} |\n'
 text+='\n④はCPUではシャッフル・座標更新を含む15個のCPU反復時間。GPUでは純GPU pass時間なので、⑤/15で代用できない。⑦⑧は後から作ったPNGの生成時間でも代用できない。③や②は実装の対応区間であり、処理順が概念図と異なる場合がある。⑩は直接計測し、内訳の単純な足し算に置き換えない。全6160成功を監査し、実行していない220条件を分母に含めていない。\n\n'
 text+='## 論文への組み込み順\n\n本文候補はF1（T）、F2（時間）、F3（最終品質）、F6（大規模の完了可否）、F7（pの交換関係）、F8（品質差の配置例）。F4/F5/F9は補足または紙幅に応じて置換。現稿には旧16分割・Sparse4分割・収束10seed・120秒打切り・旧40pivot道路実験が残るため、今回の条件と混在させず実験節の改稿が必要。ここでは論文本文・旧結果を変更していない。主結果の最悪値や小規模の低速化も掲載する。\n\n'
 for i,f in enumerate(figures,1):
  text+=f'## F{i}：{f["title"]}\n\n**掲載位置：{f["priority"]}**\n\n{f["caption"]}\n\n![[v5_図表/paper/{f["name"]}.png]]\n\n'
 text+='## 正本・再生成\n\n実験repoの `experiment_results/v5/paper/` にPNG/PDF、各図の基データJSON、計時カバレッジCSV/JSON、出典SHA256を保存。`python3 experiments/report_v5_paper.py` で測定なしに再生成する。通常の詳細考察、全グラフ別図、大規模全19図は既存ノートとレポートを保持。\n'
 note=VAULT/'5_実験結果まとめと論文用チャート.md'
 note.write_text('---\ntags: [area/大学院研究, 実験考察, 論文]\nupdated: '+now()[:10]+'\nstatus: 作成済み\n---\n\n[[0_概要]] / [[9_計測時間]] / [[4_大規模グラフ実験考察]]\n\n'+text)
 repo=text
 for f in figures:repo=repo.replace(f'![[v5_図表/paper/{f["name"]}.png]]',f'![{f["title"]}](figures/{f["name"]}.png)')
 (OUT/'README.md').write_text(repo)
 index=VAULT/'0_概要.md';s=index.read_text();link='[[5_実験結果まとめと論文用チャート]]'
 if link not in s:s=s.replace('# 実験概要\n','# 実験概要\n\n結果の要点・研究上の主張と限界・計時取得状況・論文用チャート：'+link+'。\n',1);index.write_text(s)
 timing=VAULT/'9_計測時間.md';s=timing.read_text().replace('\n\n| ⑩ |','\n| ⑩ |').replace('updated: 2026-11-05','updated: '+now()[:10],1)
 head='<!-- MEASURED_TIMING_COVERAGE_START -->';end='<!-- MEASURED_TIMING_COVERAGE_END -->'
 audit_text=text.split('## 事前に話した計時9項目の取得状況\n\n',1)[1].split('## 論文への組み込み順',1)[0]
 block=head+'\n\n## 2026-11-07：実測の取得状況\n\n'+audit_text+'詳細と図：'+link+'。\n\n'+end
 if head in s:before,tail=s.split(head,1);_,after=tail.split(end,1);s=before+block+after
 else:s=s.rstrip()+'\n\n'+block+'\n'
 timing.write_text(s)


if __name__=='__main__':main()
