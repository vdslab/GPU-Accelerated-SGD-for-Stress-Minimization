#!/usr/bin/env python3
"""Verify the final retained scope; generate paired, hierarchical v5 reports offline."""
from pathlib import Path
import collections,csv,datetime,hashlib,json,math,statistics,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from speed_plot_style import SPEED_METRICS, runtime_points, runtime_axes
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiment_results/v5';VAULT=Path('/Users/kose/Documents/ChatGPT/Obsidian タスク管理/sakamoti55/12_大学院研究/実験')
METHODS={'full':['sgd','atomic_sgd','rr_sgd'],'sparse':['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd']}
LABEL={'sgd':'CPU Full','atomic_sgd':'Atomic Full','rr_sgd':'RR Full','sparse_sgd':'CPU Sparse','atomic_sparse_sgd':'Atomic Sparse','rr_sparse_sgd':'RR Sparse'}
COLOR=dict(zip(LABEL,['#4b5563','#f59e0b','#2563eb','#9ca3af','#ea580c','#059669']))
MED=statistics.median

def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def csvwrite(path,rows):
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
def hier(rows,field):
 seedvalues=[]
 for seed in range(5):
  group=[r for r in rows if r['seed']==seed]
  assert sorted(r['repetition'] for r in group)==[0,1,2]
  vals=[r.get(field) for r in group]
  if any(v is None for v in vals):return dict(median=None,min=None,max=None,seeds=[])
  seedvalues.append(MED(vals))
 return dict(median=MED(seedvalues),min=min(seedvalues),max=max(seedvalues),seeds=seedvalues)
def gmean(vals):return math.exp(statistics.mean(math.log(v) for v in vals))
def fmt(v):return 'N/A' if v is None else f'{v:.3f}'

def main():
 plan=json.loads((BASE/'config/plan.json').read_text());jobs=[j for j in json.loads((BASE/'config/jobs.json').read_text()) if j['stage']=='comparison'];graphs={g['name']:g for g in json.loads((BASE/'config/datasets.json').read_text())};env=json.loads((BASE/'config/environment.json').read_text())
 paths=list((BASE/'raw/comparison').glob('*/record.json'));rows=[json.loads(p.read_text()) for p in paths];byid={r['run_id']:r for r in rows}
 assert len(rows)==len(byid)==len(jobs)==4230 and set(byid)=={j['run_id'] for j in jobs}
 assert len(graphs)==56
 archive_files=0
 for path,r in zip(paths,rows):
  g=graphs[r['graph']]
  assert r['status']=='success' and r['archive_status']=='success'
  assert r['iterations']==15 and r['epsilon']==.1 and r['seed'] in range(5) and r['repetition'] in range(3)
  assert r['input_sha256']==g['sha256'] and r['nodes']==g['nodes']
  assert r['binary_sha256']==env['binary_sha256'][r['method']]
  assert r['stress_value']>=0 and r['total_compute_time_ms']>0 and r['iteration_time_ms']>0
  assert r['gpu_device_time_ms'] is None
  if r['family']=='sparse':assert r['pivots']==min(200,r['nodes'])
  if r['method'].startswith('rr_'):assert r['rr_phase_chunks']==1
  if r['method']=='rr_sgd':assert r['rr_tile_size']==1024 and r['rr_workgroup_size']==1024 and r['rr_inner_shuffle'] and r['rr_outer_shuffle']
  if 'atomic' in r['method']:assert r['attempted_updates']==r['completed_updates'] and r['retry_failures']==0
  if r['nodes']<=8000:assert r['stress_kind']=='exact'
  else:assert r['stress_kind']=='sampled' and r['stress_seed']==0 and r['stress_samples']==256
  ar=json.loads((path.parent/'artifact-archive.json').read_text());assert not ar['errors']
  for file in ar['files']:
   p=Path(file['compressed_path']);assert p.exists() and p.stat().st_size==file['compressed_bytes'] and file['original_sha256']==file['decompressed_sha256'] and file['gzip_crc_validated'];archive_files+=1
 index={(r['graph'],r['method'],r['seed'],r['repetition']):r for r in rows}
 initial_mismatches=[];preprocess_mismatches=[]
 for g in graphs:
  families=['full','sparse'] if graphs[g]['set']=='common' else ['sparse']
  for family in families:
   for seed in range(5):
    rs=[index[(g,m,seed,rep)] for m in METHODS[family] for rep in range(3)]
    if len({r['initial_positions_sha256'] for r in rs})!=1:initial_mismatches.append([g,family,seed])
    if family=='sparse' and len({r['preprocess_sha256'] for r in rs})!=1:preprocess_mismatches.append([g,seed])
 # Ratios are computed for matched graph / seed / repetition before aggregation.
 paired=[];derived=[]
 for r in rows:
  r=dict(r);c=index[(r['graph'],METHODS[r['family']][0],r['seed'],r['repetition'])]
  r['sgd_speedup']=c['iteration_time_ms']/r['iteration_time_ms'];r['total_speedup']=c['total_compute_time_ms']/r['total_compute_time_ms'];r['stress_ratio']=r['stress_value']/c['stress_value'] if c['stress_value']>0 else None
  r['stress_ratio_full']=None
  if graphs[r['graph']]['set']=='common':
   full=index[(r['graph'],'sgd',r['seed'],r['repetition'])]
   if full['stress_value']>0:r['stress_ratio_full']=r['stress_value']/full['stress_value']
  if r['method'].startswith('rr_'):
   atomic=index[(r['graph'],METHODS[r['family']][1],r['seed'],r['repetition'])]
   r['rr_vs_atomic_sgd_speedup']=atomic['iteration_time_ms']/r['iteration_time_ms'];r['rr_vs_atomic_total_speedup']=atomic['total_compute_time_ms']/r['total_compute_time_ms']
  derived.append(r)
  paired.append({k:r.get(k) for k in ['run_id','graph','family','method','seed','repetition','sgd_speedup','total_speedup','stress_ratio','stress_ratio_full','rr_vs_atomic_sgd_speedup','rr_vs_atomic_total_speedup']})
 csvwrite(BASE/'tables/comparison-runs.csv',derived);csvwrite(BASE/'tables/comparison-paired-ratios.csv',paired)
 grouped=collections.defaultdict(list)
 for r in derived:grouped[(r['graph'],r['method'])].append(r)
 metrics=['iteration_time_ms','total_compute_time_ms','stress_value','stress_eval_time_ms','peak_rss_bytes','sgd_speedup','total_speedup','stress_ratio','stress_ratio_full','rr_vs_atomic_sgd_speedup','rr_vs_atomic_total_speedup']
 summary=[]
 for (g,m),rs in grouped.items():
  assert len(rs)==15
  d=dict(graph=g,method=m,family=rs[0]['family'],nodes=graphs[g]['nodes'],edges=graphs[g]['edges'],success_runs=15,planned_runs=15,complete_seeds=5,stress_kind=rs[0]['stress_kind'])
  for field in metrics:d[field]=hier(rs,field)
  d['within_seed_stress_relative_ranges']=[(max(x['stress_value'] for x in rs if x['seed']==s)-min(x['stress_value'] for x in rs if x['seed']==s))/MED(x['stress_value'] for x in rs if x['seed']==s) for s in range(5)]
  summary.append(d)
 assert len(summary)==282
 idx={(r['graph'],r['method']):r for r in summary}
 write(BASE/'tables/comparison-hierarchical.json',summary)
 flat=[]
 for r in summary:
  d={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
  for field in metrics:
   for stat in ['median','min','max']:d[field+'_'+stat]=r[field][stat]
  flat.append(d)
 csvwrite(BASE/'tables/comparison-summary.csv',flat)
 # Keep the longstanding summary filename readable with precise total timing.
 write(BASE/'tables/comparison-summary.json',flat)
 aggregate=[]
 for family,methods in METHODS.items():
  ns=sorted([g for g in graphs if family=='sparse' or graphs[g]['set']=='common'],key=lambda g:(graphs[g]['nodes'],g))
  for m in methods:
   vals=[idx[(g,m)] for g in ns]
   out=dict(family=family,method=m,graphs=len(vals),faster_than_cpu=sum(r['sgd_speedup']['median']>1 for r in vals),total_faster_than_cpu=sum(r['total_speedup']['median']>1 for r in vals),stress_lower_than_cpu=sum(r['stress_ratio']['median'] is not None and r['stress_ratio']['median']<1 for r in vals))
   for field in ['sgd_speedup','total_speedup','stress_ratio','rr_vs_atomic_sgd_speedup','rr_vs_atomic_total_speedup']:
    vs=[r[field]['median'] for r in vals if r[field]['median'] is not None and r[field]['median']>0]
    out[field]=dict(gmean=gmean(vs),median=MED(vs),min=min(vs),max=max(vs)) if vs else None
   aggregate.append(out)
 write(BASE/'tables/comparison-cross-graph.json',aggregate)
 # Shared colors, 5 seed ranges; each family's CPU is the paired denominator.
 charts=[('iteration_time_ms','Mean time per SGD iteration (ms)',lambda x:x/15,True,'sgd-time'),('total_compute_time_ms','Input to final CPU coordinates (s)',lambda x:x/1000,True,'total-time'),('sgd_speedup','Paired CPU / method SGD time',lambda x:x,True,'sgd-speedup'),('total_speedup','Paired CPU / method total time',lambda x:x,True,'total-speedup'),('stress_value','Final Full Stress',lambda x:x,True,'stress-absolute'),('stress_ratio','Final Stress / paired family CPU',lambda x:x,False,'stress-ratio')]
 for family,methods in METHODS.items():
  names=sorted([g for g in graphs if family=='sparse' or graphs[g]['set']=='common'],key=lambda g:(graphs[g]['nodes'],g));x=list(range(len(names)))
  for metric,ylabel,scale,log,suffix in charts:
   fig,ax=plt.subplots(figsize=(8,4) if metric in SPEED_METRICS else (22,7),layout='constrained')
   for m in methods:
    vals=[idx[(g,m)][metric] for g in names];y=[scale(v['median']) if v['median'] is not None else float('nan') for v in vals];lo=[scale(v['min']) if v['min'] is not None else float('nan') for v in vals];hi=[scale(v['max']) if v['max'] is not None else float('nan') for v in vals]
    if metric in SPEED_METRICS:
     runtime_points(ax,[idx[(g,m)] for g in names],metric,m,LABEL[m],scale)
    else:
     ax.plot(x,y,'o-',color=COLOR[m],ms=3,label=LABEL[m]);ax.fill_between(x,lo,hi,color=COLOR[m],alpha=.12)
   if metric in ['sgd_speedup','total_speedup','stress_ratio']:ax.axhline(1,color='#111827',ls='--',lw=1)
   if log:ax.set_yscale('log')
   if metric in SPEED_METRICS:runtime_axes(ax)
   else:ax.set_xticks(x,labels=names,rotation=70,ha='right',fontsize=8);ax.set_xlabel('Graphs ordered by number of vertices')
   ax.set_ylabel(ylabel);ax.set_title(f'{family.capitalize()} — median and min–max across 5 seeds');ax.grid(alpha=.2);ax.legend(ncol=3)
   for ext in ['png','pdf']:fig.savefig(BASE/f'figures/{family}-{suffix}.{ext}',dpi=160)
   plt.close(fig)
  fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
  for m in methods[1:]:
   vals=[idx[(g,m)] for g in names];ax.scatter([r['sgd_speedup']['median'] for r in vals],[r['stress_ratio']['median'] for r in vals],c=COLOR[m],marker='^' if m.startswith('rr_') else 'o',alpha=.75,label=LABEL[m])
  ax.axhline(1,color='#111827',ls='--');ax.axvline(1,color='#111827',ls='--');ax.set_xscale('log');ax.set_xlabel('Paired CPU / method SGD time');ax.set_ylabel('Final Stress / paired family CPU');ax.set_title(f'v5 {family.capitalize()} — speed and final Stress');ax.legend();ax.grid(alpha=.2)
  for ext in ['png','pdf']:fig.savefig(BASE/f'figures/{family}-speed-stress.{ext}',dpi=170)
  plt.close(fig)
 names=sorted([g for g in graphs if graphs[g]['set']=='common'],key=lambda g:(graphs[g]['nodes'],g));fig,ax=plt.subplots(figsize=(21,7),layout='constrained')
 for m in LABEL:
  vals=[idx[(g,m)]['stress_ratio_full'] for g in names];ax.plot(range(len(names)),[v['median'] for v in vals],'o-',ms=3,color=COLOR[m],label=LABEL[m])
 ax.axhline(1,color='#111827',ls='--');ax.set_xticks(range(len(names)),labels=names,rotation=70,ha='right',fontsize=8);ax.set_ylabel('Final Stress / paired CPU Full');ax.set_title('v5 common 38 graphs — shared CPU Full reference');ax.legend(ncol=3);ax.grid(alpha=.2)
 for ext in ['png','pdf']:fig.savefig(BASE/f'figures/common-full-sparse-stress.{ext}',dpi=160)
 plt.close(fig)
 audit=dict(formal_comparison_runs=len(rows),tile_runs=1800,total_formal_runs=6030,success_runs=4230,failed_runs=0,retained_graphs=56,conditions=282,seeds_per_condition=5,repetitions_per_seed=3,missing_runs=0,exact_stress_runs=sum(r['stress_kind']=='exact' for r in rows),sampled_stress_runs=sum(r['stress_kind']=='sampled' for r in rows),archive_files_checked_for_existence_and_size=archive_files,archive_original_and_decompressed_hashes_match=True,archive_crc_verified_at_creation=True,all_binaries_match_frozen_hashes=True,initial_fingerprint_mismatches_within_family=initial_mismatches,sparse_preprocessing_mismatches=preprocess_mismatches,large_graphs_not_executed=plan['large_excluded'],deferred_graphs=[g['name'] for g in json.loads((BASE/'config/deferred-to-large.json').read_text())['graphs']],gpu_pass_timestamps=False)
 write(BASE/'validation/comparison-completion.json',audit)
 # Reports are authored separately for speed and quality, with one shared source table.
 intro='''## 実施と検証

通常比較は共通38件×6方式＋Sparse専用18件×3方式＝282設定、各5seed×3回の **4,230実行すべて成功**。タイル1,800実行と合わせ **6,030実行すべて成功**。通常比較はseed0〜4、15反復、epsilon0.1、Sparse p=min(200,n)。RR Fullはユーザー指定T1024・W1024・shared・両shuffleあり、RR Full/Sparseともphase_chunks1。coPapersCiteseerまでで終了し、大規模14件は今回は実行していない。

全282設定で15実行・完全な5seedが揃い、欠測は0。入力・バイナリハッシュ、反復・seed・p・RR設定、Atomicの更新完了とretry失敗0、Stress評価条件、圧縮検証記録を照合した。圧縮ファイルは存在とサイズも再確認した。全ファイルのCRCを今回改めて読み直したわけではなく、作成時のCRC・展開SHA照合を記録として確認した。

比は同じgraph・seed・repetitionのCPU値と対応させて求め、seed内3回の中央値→5seedの中央値・最小〜最大で集計する。下表の幾何平均は各グラフを同じ重みで扱う記述的要約であり、独立15seedや統計的有意差の検定ではない。Fullの分母はCPU Full、SparseはCPU Sparse。

'''
 speed='# v5 速度比較：結果と考察\n\n'+intro+'## 速度の要約\n\n| 方式 | グラフ数 | SGD速度比・幾何平均 | SGD速度比の範囲 | 総計算時間の速度比・幾何平均 | SGDでCPUより速いグラフ数 |\n|---|---:|---:|---:|---:|---:|\n'
 stress='# v5 最終Stress比較：結果と考察\n\n'+intro+'## 品質の要約\n\n最終Stressは速度と同じ実行・同じ最終座標から取得し、追加レイアウト実行は0。3060実行は厳密評価、1170実行は評価seed0・256始点の推定Full Stress。Sparseの学習目的関数ではなく、共通のFull Stressを比較する。\n\n| 方式 | グラフ数 | Stress/同族CPU・幾何平均 | グラフ別比の範囲 | CPUよりStressが小さいグラフ数 |\n|---|---:|---:|---:|---:|\n'
 for r in aggregate:
  if r['method'] in ['sgd','sparse_sgd']:continue
  speed+=f"| {LABEL[r['method']]} | {r['graphs']} | {r['sgd_speedup']['gmean']:.3f} | {r['sgd_speedup']['min']:.3f}〜{r['sgd_speedup']['max']:.3f} | {r['total_speedup']['gmean']:.3f} | {r['faster_than_cpu']}/{r['graphs']} |\n"
  stress+=f"| {LABEL[r['method']]} | {r['graphs']} | {r['stress_ratio']['gmean']:.4f} | {r['stress_ratio']['min']:.4f}〜{r['stress_ratio']['max']:.4f} | {r['stress_lower_than_cpu']}/{r['graphs']} |\n"
 speed+='\n## グラフ別の結果\n\nSGD時間は15反復合計、総計算時間は入力読込開始〜CPU上の最終座標完成の直接計測total_compute_time_ms。process_wall_secondsやalgorithm_time_cold_msとは区別する。\n\n| グラフ | 方式 | SGD秒 | 総計算秒 | CPU/SGD | CPU/総計算 | peak RSS GiB |\n|---|---|---:|---:|---:|---:|---:|\n'
 stress+='\n## グラフ別の結果\n\n比が1より小さいほどCPU基準よりStressが小さい。Stress=0による未定義比はN/Aとし、品質の適格閾値は設けない。\n\n| グラフ | 方式 | 最終Full Stress | Stress/同族CPU | 5seedの比の最小〜最大 | 評価 |\n|---|---|---:|---:|---|---|\n'
 for g in sorted(graphs,key=lambda g:(graphs[g]['nodes'],g)):
  for m in LABEL:
   if (g,m) not in idx:continue
   r=idx[(g,m)]
   speed+=f"| {g} | {LABEL[m]} | {r['iteration_time_ms']['median']/1000:.4f} | {r['total_compute_time_ms']['median']/1000:.4f} | {fmt(r['sgd_speedup']['median'])} | {fmt(r['total_speedup']['median'])} | {r['peak_rss_bytes']['median']/2**30:.3f} |\n"
   stress+=f"| {g} | {LABEL[m]} | {r['stress_value']['median']:.6g} | {fmt(r['stress_ratio']['median'])} | {fmt(r['stress_ratio']['min'])}〜{fmt(r['stress_ratio']['max'])} | {r['stress_kind']} |\n"
 def agg(m):return next(r for r in aggregate if r['method']==m)
 speed+='\n## 考察\n\n'
 for m in ['rr_sgd','rr_sparse_sgd']:
  r=agg(m);speed+=f"- {LABEL[m]}のSGD速度比は幾何平均{r['sgd_speedup']['gmean']:.2f}倍、総計算時間では{r['total_speedup']['gmean']:.2f}倍。SGD単体の高速化と、前処理・初期化・転送を含む処理全体の高速化は分けて評価する。Atomicに対するSGD速度比は幾何平均{r['rr_vs_atomic_sgd_speedup']['gmean']:.2f}倍。\n"
 for m in ['atomic_sgd','rr_sgd','atomic_sparse_sgd','rr_sparse_sgd']:
  r=agg(m);speed+=f"- {LABEL[m]}は{r['faster_than_cpu']}/{r['graphs']}グラフでCPUよりSGDが速かった。遅くなる入力も隠さず、小規模では固定費が相対的に大きい可能性を考慮する。\n"
 co=[idx[('coPapersCiteseer',m)] for m in METHODS['sparse']]
 speed+='- coPapersCiteseerの15反復SGD時間中央値は、'+ '、'.join(f"{LABEL[r['method']]} {r['iteration_time_ms']['median']/1000:.2f}秒" for r in co)+'。読込・前処理・Stress評価を含むプロセス全体の待ち時間とは異なる。\n'
 speed+='- T1024はユーザー指定。タイル評価で速度上良好だった128〜256とは異なる条件であり、今回のRR Full結果をT最適化後の性能と呼ばない。\n'
 stress+='\n## 考察\n\n'
 for m in ['atomic_sgd','rr_sgd','atomic_sparse_sgd','rr_sparse_sgd']:
  r=agg(m);vals=[idx[(g,m)] for g in graphs if (g,m) in idx];best=min(vals,key=lambda x:x['stress_ratio']['median']);worst=max(vals,key=lambda x:x['stress_ratio']['median'])
  stress+=f"- {LABEL[m]}のStress/CPUは幾何平均{r['stress_ratio']['gmean']:.4f}。グラフ別には{best['graph']}の{best['stress_ratio']['median']:.4f}〜{worst['graph']}の{worst['stress_ratio']['median']:.4f}で、平均値だけでは入力ごとの品質差を表せない。\n"
 stress+='- RR FullはT1024固定の結果。タイル評価で確認したTによる更新順と品質の変化も考慮し、大きいTの速度と品質のトレードオフとして読む。\n- FullとSparseは学習対象・重みが異なる。Full/Sparse比較図では分母を共通CPU Fullへ統一し、異なる分母の比を直接比較しない。\n- 同seed3回のStress・座標の一致/変動は生データから追跡できるが、独立な初期配置は5つ。固定15反復の結果であり、収束後や最適解の品質を示すものではない。\n'
 caveats='''
## 解釈上の制約

- 同じ機材・固定バイナリ・逐次実行で測ったが、測定中の全バックグラウンド活動や温度を連続記録した対照実験ではない。大きい時間の変動をすべて方式の差に帰属しない。ディスク容量不足による中断・キャッシュ整理・再ウォームアップの履歴をlogsに保存し、ウォームアップは集計から除外した。
- メモリ記録のswap_in_bytes/swap_out_bytesはpsutil由来のOSカウンターであり、ファイルページングを含む可能性がある。厳密なスワップI/O量や「スワップなし」の証明として使わない。peak RSSは共有GPUメモリの厳密な占有量ではない。
- GPU passタイムスタンプは確保エラー回避のため全GPU方式で無効。GPU計算pass時間は未測定で、SGD実経過時間との混同を避ける。
- 推定Stressは固定256始点。始点ごとの寄与・推定誤差は今回の出力に保存されておらず、小差を統計的に有意な優劣と断定しない。
- 乱数やpivotを実験のために改変していない。Full/Sparse間は初期配置が共通ではないため、その比較は各アルゴリズムの実装全体の比較として扱う。同族内の初期配置とSparse前処理fingerprintの照合結果はvalidation/comparison-completion.jsonに保存した。
- 移管6件は今回の失敗・欠測ではない。AS365・great-britain_osm・asia_osmは前処理メモリ下限を理由に移管、auto・NACA0015・Flan_1565はユーザーの範囲指定で移管した。元の大規模8件と合わせ14件は今回未実行で、GPUでの実行成功やCPUの一般的不可能性を実証していない。

## 保存先

- `tables/comparison-runs.csv`：全4,230実行の生データと対応比。
- `tables/comparison-paired-ratios.csv`：run単位の対応CPU比。
- `tables/comparison-summary.csv` / `comparison-hierarchical.json`：282設定の階層集計と5seedの値。
- `tables/comparison-cross-graph.json`：グラフに等しい重みを与えた記述的要約。
- `validation/comparison-completion.json`：件数・条件・保存の照合。
- `raw/comparison/`：各実行のJSON・コマンド・ログ・圧縮座標。
- `config/`：最終の対象、実行順、移管理由、変更前の予定。
'''
 # Additional descriptive strata and within-seed variation; no new measurements.
 strata=[]
 for m in ['rr_sgd','rr_sparse_sgd']:
  for label,selected in [('全対象',[r for r in summary if r['method']==m]),('n≥1000（事後の補足集計）',[r for r in summary if r['method']==m and r['nodes']>=1000]),('Sparse専用18件',[r for r in summary if r['method']==m and graphs[r['graph']]['set']=='sparse_only'])]:
   if selected:strata.append(dict(method=m,group=label,graphs=len(selected),sgd_speedup_gmean=gmean([r['sgd_speedup']['median'] for r in selected]),total_speedup_gmean=gmean([r['total_speedup']['median'] for r in selected]),stress_ratio_gmean=gmean([r['stress_ratio']['median'] for r in selected])))
 csvwrite(BASE/'tables/comparison-size-strata.csv',strata)
 extra='\n## 規模別の補足集計\n\n全対象の要約には非常に小さいグラフも含まれる。n≥1000の区切りは結果を読むための事後分析であり、事前に固定した合否基準ではない。グラフ構造も異なるため純粋なサイズ効果とは呼ばない。\n\n| 方式 | 集合 | 件数 | SGD速度比・幾何平均 | 総計算速度比・幾何平均 | Stress/CPU・幾何平均 |\n|---|---|---:|---:|---:|---:|\n'
 for r in strata:extra+=f"| {LABEL[r['method']]} | {r['group']} | {r['graphs']} | {r['sgd_speedup_gmean']:.3f} | {r['total_speedup_gmean']:.3f} | {r['stress_ratio_gmean']:.4f} |\n"
 extra+='\nRR Fullの全38件では総計算時間の幾何平均速度比が1未満であり、全入力で処理全体が速くなったとは結論しない。RR Sparseも全56件での総計算速度比は約1.02倍で、SGD単体の速度比や大きいグラフ群の結果とは明確に区別する。\n'
 speed+=extra;stress+=extra
 variation=[];repro=[]
 for (g,m),rs in grouped.items():
  for seed in range(5):
   selected=[r for r in rs if r['seed']==seed];item=dict(graph=g,method=m,seed=seed)
   # The record has no final coordinate fingerprint. Use the stored Stress
   # values and preserve coordinate artifacts for later coordinate inspection.
   for field in ['iteration_time_ms','total_compute_time_ms','stress_value']:
    vals=[r[field] for r in selected];item[field+'_min']=min(vals);item[field+'_median']=MED(vals);item[field+'_max']=max(vals)
   item['relative_stress_range']=(item['stress_value_max']-item['stress_value_min'])/item['stress_value_median'];variation.append(item)
 for m in LABEL:
  selected=[r for r in variation if r['method']==m];worst=max(selected,key=lambda r:r['relative_stress_range']);repro.append(dict(method=m,graph_seed_groups=len(selected),variable_stress_groups=sum(r['relative_stress_range']>0 for r in selected),max_relative_stress_range=worst['relative_stress_range'],worst_graph=worst['graph'],worst_seed=worst['seed']))
 csvwrite(BASE/'tables/comparison-within-seed.csv',variation);write(BASE/'tables/comparison-repeatability.json',repro)
 fig,ax=plt.subplots(figsize=(21,7),layout='constrained');names=sorted(graphs,key=lambda g:(graphs[g]['nodes'],g))
 for m in ['atomic_sgd','atomic_sparse_sgd']:
  yy=[max([r['relative_stress_range'] for r in variation if r['graph']==g and r['method']==m],default=float('nan'))*100 for g in names];ax.plot(range(len(names)),yy,'o-',color=COLOR[m],ms=3,label=LABEL[m])
 ax.axhline(0,color='#4b5563',ls='--',label='CPU / RR: zero in every observed group');ax.set_xticks(range(len(names)),labels=names,rotation=70,ha='right',fontsize=8);ax.set_ylabel('Worst same-seed Stress range / median (%)');ax.set_title('v5 within-seed variation across 3 repetitions (maximum over 5 seeds)');ax.legend();ax.grid(alpha=.2)
 for ext in ['png','pdf']:fig.savefig(BASE/f'figures/repetition-stress-range.{ext}',dpi=160)
 plt.close(fig)
 text='\n## 同seed内3回の再現性\n\n範囲は(最大Stress−最小Stress)/中央値で、信頼区間ではない。全runを残したまま集計し、変動の大きい実行を除外していない。\n\n| 方式 | graph×seed群 | Stressが3回で変動した群 | 最大の相対範囲 | 該当グラフ・seed |\n|---|---:|---:|---:|---|\n'
 for r in repro:text+=f"| {LABEL[r['method']]} | {r['graph_seed_groups']} | {r['variable_stress_groups']} | {100*r['max_relative_stress_range']:.2f}% | {r['worst_graph']}・{r['worst_seed']} |\n"
 text+='\nCPU・RRでは今回の全graph×seed群でStressが3回一致した。Atomicは全群で変動を観測し、一部で非常に大きい。同じ初期配置・前処理fingerprintと更新完了数が一致していても品質の再現性は保証されない。GPU更新順の非決定性などが候補だが、今回の出力だけで原因を確定しない。seed内中央値へまとめることでこの変動が見えにくくなるため、範囲表と図も併記する。座標自体の完全な同一性まで今回の照合で主張しているわけではない。\n'
 speed+=text;stress+=text
 audit['within_seed_stress_repeatability']=repro;write(BASE/'validation/comparison-completion.json',audit)
 speed+=caveats;stress+=caveats
 if initial_mismatches or preprocess_mismatches:
  mismatch=f'\n同族内照合の不一致：初期配置{len(initial_mismatches)}件、Sparse前処理{len(preprocess_mismatches)}件。対応比はseedで対応した実装全体の比較とし、完全に同一の前処理条件とは主張しない。\n';speed+=mismatch;stress+=mismatch
 else:
  sentence='\n同族内照合では、全graph・seedでCPU/Atomic/RRの初期配置fingerprintが一致し、Sparseの前処理fingerprintも一致した。FullとSparse間の一致を意味しない。\n';speed+=sentence;stress+=sentence
 for name,text in [('speed',speed),('stress',stress)]:
  (BASE/f'reports/{name}.md').write_text(text)
  (VAULT/('2_速度比較_考察.md' if name=='speed' else '3_Stress比較_考察.md')).write_text(text)
 (BASE/'reports/comparison.md').write_text('# v5 通常比較の確定結果\n\n4,230/4,230成功、56グラフ・282設定・欠測0。速度と最終Stressは同じ実行。\n\n- [速度の結果・考察](speed.md)\n- [Stressの結果・考察](stress.md)\n- [タイルの結果・考察](tile.md)\n\n全体6,030/6,030成功。大規模14件は今回未実行。\n')
 assets=VAULT/'v5_図表';assets.mkdir(exist_ok=True)
 for p in (BASE/'figures').glob('*.png'):shutil.copy2(p,assets/p.name)
 for name,theme,images in [('2_速度比較.md','速度',['full-sgd-time','sparse-sgd-time','full-total-time','sparse-total-time','full-sgd-speedup','sparse-sgd-speedup','full-total-speedup','sparse-total-speedup']),('3_Stress比較.md','最終Stress',['full-stress-ratio','sparse-stress-ratio','common-full-sparse-stress','full-speed-stress','sparse-speed-stress','full-stress-absolute','sparse-stress-absolute','repetition-stress-range'])]:
  p=VAULT/name;s=p.read_text().replace('status: 計画案','status: 測定完了');s=s.replace('共通T=1024に確定し、速度・Stress共用4,230実行を開始。','**測定完了：4,230/4,230成功、欠測0。** RR Fullの共通T=1024。')
  heading='## v5確定結果';body=heading+'\n\n通常比較4,230実行がすべて成功。56グラフ・282設定で各5seed×3回が揃った。結果と考察は[['+('2_速度比較_考察' if theme=='速度' else '3_Stress比較_考察')+']]。移管14グラフの実行は含まない。\n\n'+'\n\n'.join(f'![[v5_図表/{image}.png]]' for image in images)+'\n'
  s=s[:s.index(heading)].rstrip()+'\n\n'+body if heading in s else s.rstrip()+'\n\n'+body;p.write_text(s)
 p=VAULT/'0_概要.md';s=p.read_text().replace('status: 計画案','status: 通常実験完了');heading='## v5 実行状況（自動更新）';body=heading+'\n\n**2026-11-06：通常実験の測定・集計・図表・考察の保存完了。**\n\n| 実験 | 成功/予定 | 状態 |\n|---|---:|---|\n| タイルサイズ | 1,800/1,800 | 結果・図表・考察保存済み |\n| 速度・最終Stress共用 | 4,230/4,230 | 56グラフ・282設定、欠測0 |\n| 今回の合計 | 6,030/6,030 | 全件成功 |\n| 後日の超大規模 | 0/最大350 | 14グラフ、今回未実行 |\n\n結果と考察：[[1_tilesize_考察]]・[[2_速度比較_考察]]・[[3_Stress比較_考察]]。結果の正本はリポジトリ `experiment_results/v5/`。通常比較はcoPapersCiteseerで終了した。\n'
 assert heading in s;p.write_text(s[:s.index(heading)].rstrip()+'\n\n'+body)
 p=BASE/'README.md';s=p.read_text();s+='\n## 測定・報告の完了\n\n2026-11-06、タイル1,800＋通常比較4,230＝6,030実行が全件成功。全対象の集計・図表・考察とObsidianへの保存を完了。速度はreports/speed.md、最終Stressはreports/stress.md、タイルはreports/tile.mdを参照。図はPNG/PDFでfiguresに保存。validation/comparison-completion.jsonに照合結果を保存。大規模14グラフは今回未実行。\n';p.write_text(s)
 print(json.dumps(dict(audit=audit,aggregate=aggregate),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
