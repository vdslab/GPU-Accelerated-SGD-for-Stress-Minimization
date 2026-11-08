#!/usr/bin/env python3
"""Per-graph RR128 vs real RR1024 comparisons from retained measurements."""
from pathlib import Path
import collections,csv,datetime,hashlib,json,math,os,re,shutil,statistics,tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import build_ex_chart_collection as exact
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiment_results/v5'
BASE=OLD/'rr128-vs1024-charts-20261108'
VAULT=Path('/Users/kose/Documents/ChatGPT/Obsidian タスク管理/sakamoti55/12_大学院研究/実験')
COLORS={1024:'#d55e00',128:'#0072b2'}
FOOTER='Previous RR1024 vs EX2 RR128: different sessions; timestamps enabled only for EX2.\nMedians of 5 seed medians (3 repeats/seed); error bars: min–max of 5 seed medians, not confidence intervals.'

def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic_note(p,s):
 with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=p.parent,prefix='.rr1024-',suffix='.tmp',delete=False) as f:f.write(s);f.flush();os.fsync(f.fileno());temp=f.name
 os.replace(temp,p)
def hierarchy(rs,fun):
 values=[]
 for seed in range(5):
  sub=sorted([r for r in rs if r['seed']==seed],key=lambda r:r['repetition']);assert [r['repetition'] for r in sub]==[0,1,2]
  values.append(statistics.median(fun(r) for r in sub))
 return dict(median=statistics.median(values),min=min(values),max=max(values),seed_medians=values)

def main():
 for d in ['figures','tables','reports','validation','exact-stress/cache','source-snapshot']:(BASE/d).mkdir(parents=True,exist_ok=True)
 exact.BASE=BASE
 previouspaths=list((OLD/'raw/comparison').glob('*/record.json'))+list((OLD/'additional-grid-workgroup/raw').glob('*/record.json'))
 currentpaths=list((OLD/'ex2-rr128-speed-stress-20261108/raw').glob('*/record.json'))
 protected={str(p.relative_to(ROOT)):sha(p) for p in previouspaths+currentpaths}
 save(BASE/'validation/measurement-record-sha256.json',protected)
 new=[json.loads(p.read_text()) for p in currentpaths];idx={(r['graph'],r['method'],r['seed'],r['repetition']):r for r in new}
 full=[dict(json.loads(p.read_text())) for p in (OLD/'raw/comparison').glob('*/record.json') if 'rr_sgd' in p.parent.name and 'rr_sparse' not in p.parent.name]
 assert len(full)==570 and all(r['rr_tile_size']==r['rr_workgroup_size']==1024 for r in full)
 grid=[json.loads(p.read_text()) for p in (OLD/'additional-grid-workgroup/raw').glob('*/record.json')]
 available=[dict(r) for r in grid if r['stage']=='sparse' and r['workgroup_size']==1024 and r['pivots']==200 and (r['graph'],'rr_sparse_sgd',r['seed'],r['repetition']) in idx]
 assert len(available)==45
 for r in available:r['family']='sparse'
 old=full+available
 strict=[]
 for r in old:
  r['source_run_id']=r['run_id']
  if r['family']=='full' and r['stress_kind']!='exact':
   value=exact.evaluate(r);strict.append(dict(run_id=r['run_id'],graph=r['graph'],source_stress_value=r['stress_value'],source_stress_kind=r['stress_kind'],exact_stress_value=value['stress_value'],coordinates_key=value['coordinates_key']))
   r['stress_value']=value['stress_value'];r['stress_kind']='exact'
   if len(strict)%15==0:print(f'Full1024 exact stress aligned {len(strict)}/60',flush=True)
 save(BASE/'tables/full1024-exact-stress-overlay.json',strict)
 groups=collections.defaultdict(list)
 for r in old:groups[r['family'],r['graph']].append(r)
 comparisons=[];runpairs=[]
 for (family,graph),rs in sorted(groups.items()):
  assert len(rs)==15
  method='rr_sgd' if family=='full' else 'rr_sparse_sgd'
  ns=[idx[graph,method,r['seed'],r['repetition']] for r in rs]
  for r,n in zip(rs,ns):
   assert r['input_sha256']==n['input_sha256'] and r['initial_positions_sha256']==n['initial_positions_sha256']
   assert r['iterations']==n['iterations']==15 and r['epsilon']==n['epsilon']==.1 and r['rr_phase_chunks']==n['rr_phase_chunks']==1
   assert r['stress_kind']==n['stress_kind']
   if family=='sparse':assert r['pivots']==n['pivots']==200 and r['preprocess_sha256']==n['preprocess_sha256'] and r['rr_pivot_order']==n['rr_pivot_order']
   else:assert r['rr_memory_mode']==n['rr_memory_mode']=='shared' and r['rr_inner_shuffle']==n['rr_inner_shuffle'] and r['rr_outer_shuffle']==n['rr_outer_shuffle']
   runpairs.append(dict(family=family,graph=graph,seed=r['seed'],repetition=r['repetition'],source1024_run_id=r['run_id'],source128_run_id=n['run_id'],sgd_speedup=r['sgd_time_ms']/n['sgd_time_ms'],total_speedup=r['total_compute_time_ms']/n['total_compute_time_ms'],stress_ratio=n['stress_value']/r['stress_value'],sgd1024_ms=r['sgd_time_ms']/15,sgd128_ms=n['sgd_time_ms']/15,stress1024=r['stress_value'],stress128=n['stress_value']))
  s=dict(family=family,graph=graph,nodes=rs[0]['nodes'],stress_kind=rs[0]['stress_kind'])
  for size,xs in [(1024,rs),(128,ns)]:
   s[str(size)]={f:hierarchy(xs,lambda r,f=f:r[f]) for f in ['sgd_time_ms','total_compute_time_ms','stress_value']}
  ps=[p for p in runpairs if p['family']==family and p['graph']==graph]
  for f in ['sgd_speedup','total_speedup','stress_ratio']:s[f]=hierarchy(ps,lambda r,f=f:r[f])
  comparisons.append(s)
 assert len(comparisons)==41
 save(BASE/'tables/hierarchical.json',comparisons)
 with (BASE/'tables/paired-runs.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(runpairs[0]));w.writeheader();w.writerows(runpairs)
 figures=[]
 plt.rcParams.update({'font.size':9,'axes.titlesize':10,'pdf.fonttype':42,'ps.fonttype':42})
 for family in ['full','sparse']:
  ss=sorted([s for s in comparisons if s['family']==family],key=lambda s:(s['nodes'],s['graph']));n=len(ss);width=20 if family=='full' else 9
  titleprefix='Full: T=W128 vs T=W1024' if family=='full' else 'Sparse: W128 vs W1024, p=200 (3 matched graphs only)'
  for field,label,scale,name in [('sgd_time_ms','Mean elapsed time per SGD iteration (ms)',1/15,'sgd-time'),('total_compute_time_ms','Total compute, input to CPU coordinates (s)',.001,'total-time')]:
   fig,ax=plt.subplots(figsize=(width,5.8))
   for size,off,marker in [(1024,-.12,'^'),(128,.12,'o')]:
    vals=[s[str(size)][field] for s in ss];y=[v['median']*scale for v in vals];err=[[max(0,(v['median']-v['min'])*scale) for v in vals],[max(0,(v['max']-v['median'])*scale) for v in vals]]
    ax.errorbar([i+off for i in range(n)],y,yerr=err,fmt=marker,linestyle='none',color=COLORS[size],markerfacecolor='none',capsize=2,ms=4,label=str(size))
   ax.set_yscale('log');ax.set_ylabel(label);ax.set_xticks(range(n),[s['graph'] for s in ss],rotation=65,ha='right',fontsize=8);ax.set_xlabel('Graphs ordered by number of vertices');ax.grid(alpha=.2);ax.legend(title='T=W' if family=='full' else 'W',frameon=False)
   fig.subplots_adjust(bottom=.34,top=.88,left=.08,right=.99);fig.suptitle(titleprefix);fig.text(.5,.012,FOOTER,ha='center',fontsize=8)
   fname=f'{family}-pergraph-{name}'
   for ext in ['png','pdf']:fig.savefig(BASE/f'figures/{fname}.{ext}',dpi=200)
   plt.close(fig);figures.append(dict(name=fname,title=titleprefix+' — '+label))
  for field,label,name in [('sgd_speedup','SGD time1024 / time128 (higher = 128 faster)','sgd-speedup'),('stress_ratio','Final Stress128 / Stress1024 (lower = 128 better)','stress-ratio')]:
   fig,ax=plt.subplots(figsize=(width,5.8));vals=[s[field] for s in ss]
   ax.errorbar(range(n),[v['median'] for v in vals],yerr=[[max(0,v['median']-v['min']) for v in vals],[max(0,v['max']-v['median']) for v in vals]],fmt='o',linestyle='none',color=COLORS[128],capsize=2,ms=4)
   ax.axhline(1,color=COLORS[1024],ls='--',lw=.8,label='1024 reference');ax.set_ylabel(label);ax.set_xticks(range(n),[s['graph'] for s in ss],rotation=65,ha='right',fontsize=8);ax.set_xlabel('Graphs ordered by number of vertices');ax.grid(alpha=.2);ax.legend(frameon=False)
   if field=='sgd_speedup':ax.set_yscale('log')
   quality='Full: exact Stress for both sizes' if family=='full' else 'Sparse: same 256 sources, seed0, both sizes'
   fig.subplots_adjust(bottom=.34,top=.88,left=.08,right=.99);fig.suptitle(titleprefix+'\n'+(quality if field=='stress_ratio' else 'paired SGD speedup'));fig.text(.5,.012,FOOTER,ha='center',fontsize=8)
   fname=f'{family}-pergraph-{name}'
   for ext in ['png','pdf']:fig.savefig(BASE/f'figures/{fname}.{ext}',dpi=200)
   plt.close(fig);figures.append(dict(name=fname,title=titleprefix+' — '+label))
 individual=[]
 for s in comparisons:
  family=s['family'];graph=s['graph'];fig,axs=plt.subplots(1,3,figsize=(11,4.2))
  for ax,(field,label,scale) in zip(axs,[('sgd_time_ms','SGD elapsed / iteration (ms)',1/15),('total_compute_time_ms','Total without frame (s)',.001),('stress_value','Final Full Stress',1)]):
   for i,size in enumerate([1024,128]):
    v=s[str(size)][field];ax.bar(i,v['median']*scale,width=.55,color=COLORS[size],alpha=.65);ax.errorbar(i,v['median']*scale,yerr=[[max(0,(v['median']-v['min'])*scale)],[max(0,(v['max']-v['median'])*scale)]],fmt='none',color='black',capsize=3,lw=.8)
   ax.set_xticks([0,1],['1024','128']);ax.set_xlabel('T=W' if family=='full' else 'W');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
  fig.suptitle(f"{graph} — {family.capitalize()}, n={s['nodes']:,} — Stress: {s['stress_kind']}")
  fig.subplots_adjust(bottom=.27,top=.84,left=.065,right=.98,wspace=.4);fig.text(.5,.015,FOOTER,ha='center',fontsize=7)
  safe=re.sub(r'[^A-Za-z0-9_-]','_',graph);name=f'{family}-detail-{safe}'
  for ext in ['png','pdf']:fig.savefig(BASE/f'figures/{name}.{ext}',dpi=180)
  plt.close(fig);individual.append(dict(graph=graph,family=family,name=name))
 for rel,h in protected.items():assert sha(ROOT/rel)==h
 assets=VAULT/'EX_128対1024_図表';assets.mkdir(exist_ok=True)
 for f in figures+individual:shutil.copy2(BASE/f"figures/{f['name']}.png",assets/f"{f['name']}.png")
 text='''---
tags: [area/大学院研究, 論文, チャート]
updated: 2026-11-08
status: 作成完了
---

# EX_チャート集：128と1024のグラフ別比較

[[EX_チャート集]] / [[EX2_速度ストレス実験]]

## 比較条件

- **Full：38グラフすべてでT=W1024（前回v5）対T=W128（EX2）。** shared・内外shuffle・phase_chunks1、15反復、epsilon0.1、seed0〜4×3回。初期配置一致を確認。
- **Sparse：W1024（追加grid/workgroup実験）対W128（EX2）。** 入力・p200・初期配置・前処理・seed・反復が一致するluxembourg_osm、la2010、coPapersCiteseerの3グラフだけを比較する。SparseにFullと同じ意味のTはない。
- 前回通常比較のSparseはW256であり、1024として扱わない。W1024のデータがない残り53グラフや、RR128未測定のgrid入力は比較図へ入れない。新規レイアウト実行は行っていない。
- FullのStressは両サイズとも厳密。前回推定だった4グラフのRR1024計60記録は保存済み座標から厳密再評価し、元recordを変更せず評価オーバーレイに保存した。
- Sparseの3グラフは両サイズとも同一の256始点・評価seed0による推定Full Stress。Fullの厳密値と混同しない。
- 点・棒は同seed3回中央値→5seed中央値。誤差棒は5seed中央値の最小〜最大で、信頼区間ではない。速度比・Stress比は対応するrunごとに求めてから集計する。
- 測定時期が異なり、RR128のみGPU timestampが有効。差をT/W変更だけの因果効果とは断定しない。主速度はSGD経過時間/15と描画なし総計算時間。

## 一覧図

SGD速度比は1024/128で**1より大きいほど128が高速**。Stress比は128/1024で**1より小さいほど128のStressが低い**。

'''
 for f in figures:text+=f"### {f['title']}\n\n![[EX_128対1024_図表/{f['name']}.png]]\n\n"
 text+='## グラフ別の数値と個別図\n\n| 族 | グラフ | SGD速度比1024/128 | Stress比128/1024 | 個別図（時間・総時間・Stress） |\n|---|---|---:|---:|---|\n'
 for s in sorted(comparisons,key=lambda s:(s['family'],s['nodes'],s['graph'])):
  detail=next(f for f in individual if f['family']==s['family'] and f['graph']==s['graph'])
  text+=f"| {s['family']} | {s['graph']} | {s['sgd_speedup']['median']:.3f} | {s['stress_ratio']['median']:.5f} | [[EX_128対1024_図表/{detail['name']}.png\\|開く]] |\n"
 text+='''
## 保存先

`experiment_results/v5/rr128-vs1024-charts-20261108/`

- `figures/`：一覧8図＋個別41図。各PNG・ベクトルPDF。
- `tables/paired-runs.csv`：615対応組（Full570＋Sparse45）。`hierarchical.json`：グラフ別の階層集計。
- `tables/full1024-exact-stress-overlay.json`、`exact-stress/cache/`：Full1024の厳密再評価の出典と値。
- `validation/`：元の実験記録のハッシュと完了検証。

前回v5・追加grid実験・EX2の記録は保持した。
'''
 (BASE/'reports/COMPARISON.md').write_text(text);atomic_note(VAULT/'EX_チャート集_128と1024比較.md',text)
 indexnote=VAULT/'EX_チャート集.md';current=indexnote.read_text();heading='## 128対1024の追加比較'
 if heading not in current:current+='\n\n'+heading+'\n\n[[EX_チャート集_128と1024比較]]：Full38グラフ、SparseはW1024の比較条件が揃う3グラフ。一覧図とグラフ別の個別図を収録。\n'
 atomic_note(indexnote,current)
 save(BASE/'figures/index.json',dict(overview=figures,individual=individual))
 save(BASE/'validation/completion.json',dict(full_graphs=38,sparse_graphs=3,matched_run_pairs=615,full_strict_overlay=60,overview_charts=8,individual_charts=41,prior_records_unchanged=True,paired_parameters_match=True))
 shutil.copy2(Path(__file__),BASE/'source-snapshot'/Path(__file__).name)
 readme='# RR128 vs RR1024, per graph\n\n[Japanese chart book](reports/COMPARISON.md). Full38 graphs; Sparse3 graphs with actual matched W1024/p200 data. Other Sparse W1024 cases are unmeasured, not replaced with W256.\n\n'
 for f in figures+individual:readme+=f"- {f.get('title',f.get('graph'))}: [PNG](figures/{f['name']}.png), [PDF](figures/{f['name']}.pdf)\n"
 (BASE/'README.md').write_text(readme)
 print('DONE: 8 overview + 41 per-graph charts; Obsidian updated.',flush=True)

if __name__=='__main__':main()
