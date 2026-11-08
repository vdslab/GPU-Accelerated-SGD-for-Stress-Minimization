#!/usr/bin/env python3
"""Paper chart collection: historical CPU/Atomic + EX2 RR128, paired quality."""
from pathlib import Path
import collections,csv,datetime,gzip,hashlib,json,lzma,math,os,shutil,statistics,subprocess,tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from speed_plot_style import method_style,runtime_axes
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiment_results/v5'
EX2=OLD/'ex2-rr128-speed-stress-20261108'
BASE=OLD/'ex-chart-collection-rr128-20261108'
VAULT=Path('/Users/kose/Documents/ChatGPT/Obsidian タスク管理/sakamoti55/12_大学院研究/実験')
EVALUATOR=ROOT/'shared/experiment-common/target/release/exact-layout-stress'
FAMILIES={'full':['sgd','atomic_sgd','rr_sgd'],'sparse':['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd']}
LABEL={'sgd':'CPU (previous)','atomic_sgd':'Atomic (previous)','rr_sgd':'RR128 (EX2)','sparse_sgd':'CPU (previous)','atomic_sparse_sgd':'Atomic (previous)','rr_sparse_sgd':'RR128 (EX2)'}
FOOTER='CPU/Atomic: previous v5; RR128: EX2. Different sessions; GPU timestamps enabled only for EX2 RR.\nPoints: median of 5 seed medians (3 repeats/seed); whiskers: min–max of those 5 medians, not confidence intervals.'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def atomic_note(path,text):
 with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix='.ex-charts-',suffix='.tmp',delete=False) as f:
  f.write(text);f.flush();os.fsync(f.fileno());temp=f.name
 os.replace(temp,path)
def read_layout(path):
 opener=gzip.open if path.suffix=='.gz' else lzma.open if path.suffix=='.xz' else open
 with opener(path,'rb') as f:return f.read()
def evaluate(row,force=False):
 path=Path(row['final_positions_path']);data=read_layout(path)
 positions=data.split(b'# Positions (x y)',1)[1]
 precision='f64' if row['method']=='sgd' else 'f32'
 key=hashlib.sha256((row['input_sha256']+precision).encode()+positions).hexdigest()
 cache=BASE/'exact-stress/cache'/f'{key}.json'
 if cache.exists():value=json.loads(cache.read_text())
 else:
  proc=subprocess.run([str(EVALUATOR),precision],input=data,capture_output=True)
  if proc.returncode:raise RuntimeError(proc.stderr.decode())
  value=json.loads(proc.stdout);assert value['nodes']==row['nodes'] and value['edges']==row['edges']
  value.update(coordinates_key=key,input_sha256=row['input_sha256'],source_layout=str(path),source_decompressed_sha256=hashlib.sha256(data).hexdigest(),evaluator_sha256=sha(EVALUATOR))
  save(cache,value)
 if force:
  assert math.isclose(value['stress_value'],row['stress_value'],rel_tol=1e-12,abs_tol=1e-9),(row['method'],value['stress_value'],row['stress_value'])
 return value

def hierarchy(rows,field):
 vals=[]
 for seed in range(5):
  rs=sorted([r for r in rows if r['seed']==seed],key=lambda r:r['repetition'])
  assert [r['repetition'] for r in rs]==[0,1,2]
  vals.append(statistics.median(r[field] for r in rs))
 return dict(median=statistics.median(vals),min=min(vals),max=max(vals),seed_medians=vals)
def gm(vs):return math.exp(statistics.mean(math.log(v) for v in vs))

def points(ax,stats,metric,method,scale=1,x=None):
 stats=sorted(stats,key=lambda r:(r['nodes'],r['graph']))
 y=[s[metric]['median']*scale for s in stats]
 bounds=[[max(0,(s[metric]['median']-s[metric]['min'])*scale) for s in stats],[max(0,(s[metric]['max']-s[metric]['median'])*scale) for s in stats]]
 ax.errorbar([s['nodes'] for s in stats] if x is None else x,y,yerr=bounds,fmt=method_style(method)['marker'],color=method_style(method)['color'],linestyle='none',ms=4,capsize=1.5,elinewidth=.7,markerfacecolor='none',markeredgewidth=.8,label=LABEL[method],alpha=.9)
 ax.grid(alpha=.2)

def main():
 for d in ['tables','figures','reports','validation','exact-stress/cache','source-snapshot']:(BASE/d).mkdir(parents=True,exist_ok=True)
 oldpaths=sorted((OLD/'raw/comparison').glob('*/record.json'));newpaths=sorted((EX2/'raw').glob('*/record.json'))
 assert len(oldpaths)==4230 and len(newpaths)==1410
 protected={str(p.relative_to(ROOT)):sha(p) for p in oldpaths+newpaths};save(BASE/'validation/input-records-sha256.json',protected)
 old=[json.loads(p.read_text()) for p in oldpaths];new=[json.loads(p.read_text()) for p in newpaths]
 rows=[dict(r,measurement_source='previous_v5') for r in old if not r['method'].startswith('rr_')]+[dict(r,measurement_source='EX2_RR128') for r in new]
 assert len(rows)==4230
 # Confirm saved coordinate precision reproduces native stress before applying to large inputs.
 controls=[]
 for method in ['sgd','atomic_sgd']:
  r=next(r for r in rows if r['method']==method and r['graph']=='dwt_1005')
  controls.append(dict(run_id=r['run_id'],native_stress=r['stress_value'],reevaluated=evaluate(r,force=True)['stress_value']))
 save(BASE/'validation/exact-stress-controls.json',controls)
 overlay=[]
 for r in rows:
  assert r['status']=='success' and r['iterations']==15 and r['epsilon']==.1
  if r['family']=='full' and r['stress_kind']!='exact':
   assert r['method'] in ['sgd','atomic_sgd']
   value=evaluate(r)
   overlay.append(dict(run_id=r['run_id'],method=r['method'],graph=r['graph'],seed=r['seed'],repetition=r['repetition'],original_stress_value=r['stress_value'],original_stress_kind=r['stress_kind'],exact_stress_value=value['stress_value'],coordinates_key=value['coordinates_key']))
   r['native_stress_value']=r['stress_value'];r['native_stress_kind']=r['stress_kind'];r['stress_value']=value['stress_value'];r['stress_kind']='exact';r['chart_stress_source']='reevaluated_saved_coordinates'
   if len(overlay)%15==0:print(f'Exact Stress alignment {len(overlay)}/120 records; {len(list((BASE/"exact-stress/cache").glob("*.json")))} cached layouts',flush=True)
  else:r['chart_stress_source']='native'
 assert len(overlay)==120
 save(BASE/'tables/full-exact-stress-overlay.json',overlay)
 index={(r['graph'],r['method'],r['seed'],r['repetition']):r for r in rows};assert len(index)==4230
 names={family:sorted({r['graph'] for r in rows if r['family']==family},key=lambda g:(next(r['nodes'] for r in rows if r['graph']==g),g)) for family in FAMILIES}
 assert len(names['full'])==38 and len(names['sparse'])==56
 for family,methods in FAMILIES.items():
  for g in names[family]:
   for seed in range(5):
    ss=[index[g,m,seed,rep] for m in methods for rep in range(3)]
    assert len({r['initial_positions_sha256'] for r in ss})==1
    assert len({r['input_sha256'] for r in ss})==1
    if family=='sparse':assert len({r['preprocess_sha256'] for r in ss})==1
    assert len({r['stress_kind'] for r in ss})==1
 # Ratios are computed per matching run, before either aggregation level.
 for r in rows:
  cpu=index[r['graph'],FAMILIES[r['family']][0],r['seed'],r['repetition']]
  r['sgd_speedup']=cpu['sgd_time_ms']/r['sgd_time_ms']
  r['total_speedup']=cpu['total_compute_time_ms']/r['total_compute_time_ms']
  assert cpu['stress_value']>0
  r['stress_ratio']=r['stress_value']/cpu['stress_value']
 groups=collections.defaultdict(list)
 for r in rows:groups[r['family'],r['method'],r['graph']].append(r)
 summaries=[]
 for (family,method,g),rs in sorted(groups.items()):
  assert len(rs)==15
  s=dict(family=family,method=method,graph=g,nodes=rs[0]['nodes'],edges=rs[0]['edges'],stress_kind=rs[0]['stress_kind'],measurement_source=rs[0]['measurement_source'],successful_runs=15,complete_seeds=5)
  for field in ['sgd_time_ms','total_compute_time_ms','stress_value','sgd_speedup','total_speedup','stress_ratio']:s[field]=hierarchy(rs,field)
  summaries.append(s)
 save(BASE/'tables/hierarchical.json',summaries)
 fields=['graph','family','method','run_id','seed','repetition','nodes','measurement_source','sgd_time_ms','total_compute_time_ms','stress_value','stress_kind','chart_stress_source','sgd_speedup','total_speedup','stress_ratio']
 with (BASE/'tables/paired-runs.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 aggregate=[]
 for family,methods in FAMILIES.items():
  for m in methods:
   ss=[s for s in summaries if s['method']==m]
   aggregate.append(dict(family=family,method=m,graphs=len(ss),sgd_speedup_gmean=gm([s['sgd_speedup']['median'] for s in ss]),total_speedup_gmean=gm([s['total_speedup']['median'] for s in ss]),stress_ratio_gmean=gm([s['stress_ratio']['median'] for s in ss])))
 save(BASE/'tables/cross-graph.json',aggregate)
 plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})
 figures=[]
 def export(fig,name,title,caption):
  fig.subplots_adjust(bottom=.20,top=.87,left=.10,right=.98)
  fig.suptitle(title,fontsize=11,y=.98)
  fig.text(.5,.015,FOOTER,ha='center',va='bottom',fontsize=7)
  for ext in ['png','pdf']:fig.savefig(BASE/f'figures/{name}.{ext}',dpi=200)
  plt.close(fig);figures.append(dict(name=name,title=title,caption=caption))
 for family,methods in FAMILIES.items():
  subset={m:sorted([s for s in summaries if s['method']==m],key=lambda s:(s['nodes'],s['graph'])) for m in methods}
  count=len(names[family]);quality='Exact Full Stress for all methods' if family=='full' else 'Exact n<=8000; same 256 sources (seed0) otherwise'
  for metric,label,scale,suffix in [('sgd_time_ms','Mean elapsed time per SGD iteration (ms)',1/15,'sgd-time'),('total_compute_time_ms','Total compute, input to CPU coordinates (s)',.001,'total-time'),('sgd_speedup','Paired CPU / method SGD time',1,'sgd-speedup'),('total_speedup','Paired CPU / method total compute time',1,'total-speedup')]:
   fig,ax=plt.subplots(figsize=(8.5,4.8))
   for m in methods:points(ax,subset[m],metric,m,scale)
   runtime_axes(ax);ax.set_ylabel(label)
   if 'speedup' in metric:ax.axhline(1,color='gray',ls='--',lw=.8)
   ax.legend(frameon=False)
   export(fig,f'{family}-{suffix}',f'{family.capitalize()} — CPU / Atomic / RR128 ({count} graphs)',f'{label}。低いほど高速。' if 'speedup' not in metric else f'{label}。1より大きいほどCPUより高速。')
  for metric,label,suffix in [('stress_ratio','Final Stress / paired family CPU','stress-ratio'),('stress_value','Final Full Stress (absolute)','stress-absolute')]:
   fig,ax=plt.subplots(figsize=(20,5.8))
   for m in methods:points(ax,subset[m],metric,m,x=list(range(count)))
   ax.set_xticks(range(count),names[family],rotation=65,ha='right',fontsize=8);ax.set_xlabel('Graphs ordered by number of vertices');ax.set_ylabel(label)
   if metric=='stress_ratio':ax.axhline(1,color='gray',ls='--',lw=.8)
   else:ax.set_yscale('log')
   ax.legend(frameon=False,ncol=3,loc='upper left')
   # Leave more space for graph names without clipping the source footnote.
   fig.subplots_adjust(bottom=.34,top=.87,left=.065,right=.99);fig.suptitle(f'{family.capitalize()} — {quality}',fontsize=12,y=.98);fig.text(.5,.012,FOOTER,ha='center',fontsize=8)
   name=f'{family}-{suffix}'
   for ext in ['png','pdf']:fig.savefig(BASE/f'figures/{name}.{ext}',dpi=200)
   plt.close(fig);figures.append(dict(name=name,title=f'{family.capitalize()} — {label}',caption=quality+'。Stress比は1より小さいほどCPUより低い。絶対Stressの大小で異なるグラフ間の品質順位を判断しない。'))
  fig,ax=plt.subplots(figsize=(8.5,5.2))
  for m in methods[1:]:
   ss=subset[m];style=method_style(m)
   ax.scatter([s['sgd_speedup']['median'] for s in ss],[s['stress_ratio']['median'] for s in ss],marker=style['marker'],facecolors='none',edgecolors=style['color'],label=LABEL[m],s=22,alpha=.85)
  ax.set_xscale('log');ax.axvline(1,color='gray',ls='--',lw=.8);ax.axhline(1,color='gray',ls='--',lw=.8);ax.set_xlabel('Paired CPU / method SGD time');ax.set_ylabel('Final Stress / paired family CPU');ax.grid(alpha=.2);ax.legend(frameon=False)
  export(fig,f'{family}-speed-stress',f'{family.capitalize()} — speed and final quality',quality+'。右ほど高速、下ほど低Stress。同じ対応run集合から集計。')
 # Compact speed overview for the chat and the top of the paper-chart notebook.
 fig,axs=plt.subplots(1,2,figsize=(12,4.8))
 for ax,(family,methods) in zip(axs,FAMILIES.items()):
  for m in methods:points(ax,[s for s in summaries if s['method']==m],'sgd_time_ms',m,1/15)
  runtime_axes(ax);ax.set_ylabel('Mean elapsed time per SGD iteration (ms)');ax.set_title(f'{family.capitalize()} ({len(names[family])} graphs)');ax.legend(frameon=False)
 export(fig,'overview-sgd','CPU / Atomic (previous v5) vs RR128 (EX2)','Full/Sparseを別パネルで表示。RR FullはT=W128、RR SparseはW128。')
 fig,axs=plt.subplots(1,2,figsize=(12,4.8))
 for ax,(family,methods) in zip(axs,FAMILIES.items()):
  for m in methods:points(ax,[s for s in summaries if s['method']==m],'sgd_speedup',m)
  runtime_axes(ax);ax.set_ylabel('Paired CPU / method SGD time');ax.set_title(f'{family.capitalize()} ({len(names[family])} graphs)');ax.axhline(1,color='gray',ls='--',lw=.8);ax.legend(frameon=False)
 export(fig,'overview-speedup','CPU speedup — historical CPU / Atomic vs RR128','1倍基準線。FullはCPU Full、SparseはCPU Sparseを分母にする。')
 for rel,h in protected.items():assert sha(ROOT/rel)==h
 save(BASE/'validation/completion.json',dict(completed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),historical_cpu_atomic_records=2820,ex2_rr_records=1410,total=4230,conditions=282,full_graphs=38,sparse_graphs=56,strict_full_stress_overlay_records=120,prior_records_unchanged=True,paired_before_aggregation=True,initial_positions_and_sparse_preprocessing_match=True,figures=len(figures)))
 shutil.copy2(Path(__file__),BASE/'source-snapshot'/Path(__file__).name)
 for p in (ROOT/'tools/diagnostics/exact-layout-stress').rglob('*'):
  if p.is_file() and 'target' not in p.parts:
   d=BASE/'source-snapshot/exact-layout-stress'/p.relative_to(ROOT/'tools/diagnostics/exact-layout-stress');d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
 save(BASE/'source-snapshot/source-and-binary-sha256.json',{'chart_generator':sha(Path(__file__)),'evaluator':sha(EVALUATOR),'shared_stress_source':sha(ROOT/'shared/experiment-common/src/stress.rs')})
 assets=VAULT/'EX_チャート集_図表';assets.mkdir(exist_ok=True)
 for item in figures:shutil.copy2(BASE/f"figures/{item['name']}.png",assets/f"{item['name']}.png")
 text='''---
tags: [area/大学院研究, 論文, チャート]
updated: 2026-11-08
status: 作成完了
---

# EX_チャート集：CPU・Atomic・RR128

[[EX2_速度ストレス実験]] / [[EX2_速度ストレス実験_結果と考察]] / [[9_計測時間]]

## 収録データと読み方

- CPU・Atomic：前回v5通常比較の保存済み2,820実行。レイアウトは再測定していない。
- RR：EX2の1,410実行。FullはT=W128、SparseはW128・p=min(200,n)、15反復、epsilon0.1、seed0〜4×3回。
- Fullは38グラフ、Sparseは56グラフ。各族でCPU・Atomic・RRのグラフ、入力、seed、repetitionを対応させる。初期座標、Sparseの前処理fingerprint一致を確認した。
- 点は同seed3回中央値→5seed中央値。誤差棒は5seed中央値の最小〜最大で、信頼区間ではない。
- CPU＝灰色×、Atomic＝橙△、RR128＝青○。異なるグラフの測定点を線で結ばない。
- CPU速度比・Stress比は対応するrunごとに比を求めてから集計する。FullはCPU Full、SparseはCPU Sparseを分母とする。

## 論文に記載する計測条件差

CPU・Atomicは前回測定、RR128は今回測定であり、同時期の比較ではない。前回GPU timestamp無効、EX2 RRは有効。主速度図は⑤SGD経過時間/15と⑩描画なし総計算時間を使い、純GPU pass時間と混同しない。Stress評価・描画は主速度の計測終了後である。これらの差を図注・実験条件に明記し、速度差の全てをT/W変更だけの効果に帰属しない。

## 品質評価を揃えた方法

Fullは3方式すべて厳密Full Stressで比較する。前回推定値だった4グラフ（ca-HepTh、bcsstk17、PGPgiantcompo、fe_4elt2）のCPU・Atomic計120記録は、保存済み最終座標から共通評価器で厳密に再評価した。同一の座標はキャッシュ再利用し、レイアウト・速度は再実行していない。CPUはf64、Atomicはf32として保存座標を復元し、既存の小グラフ厳密値との一致を確認した。過去recordは書き換えず、このチャート集用の評価オーバーレイを保存した。

Sparseは3方式とも従来どおり、n≤8000は厳密、n>8000は同じ固定256始点・評価seed0の推定Full Stress。Sparseの最適化目的関数値とは区別する。FullとSparseの品質図を、そのまま共通評価精度の比較とは呼ばない。

## 速度比較の概要

![[EX_チャート集_図表/overview-sgd.png]]

![[EX_チャート集_図表/overview-speedup.png]]

## CPU基準の記述的集計

| 族 | 方式 | SGD速度比（幾何平均） | 総計算速度比（幾何平均） | Stress/CPU（幾何平均） |
|---|---|---:|---:|---:|
'''
 for item in aggregate:text+=f"| {item['family']} | {LABEL[item['method']]} | {item['sgd_speedup_gmean']:.3f} | {item['total_speedup_gmean']:.3f} | {item['stress_ratio_gmean']:.5f} |\n"
 text+='\nグラフを等しい重みで要約した記述値であり、有意差検定ではない。FullとSparseの対象集合が異なるため、幾何平均同士を方式間の直接比較に使わない。\n'
 for family in FAMILIES:
  text+=f"\n## {family.capitalize()}：論文用の個別図\n\n"
  for item in figures:
   if item['name'].startswith(family+'-'):text+=f"### {item['title']}\n\n{item['caption']}\n\n![[EX_チャート集_図表/{item['name']}.png]]\n\n"
 text+='''## ファイルと再現情報

リポジトリ正本：`experiment_results/v5/ex-chart-collection-rr128-20261108/`

- `figures/`：各図のPNG（200 dpi）・PDF（ベクトル）。論文にはPDFを利用可能。
- `tables/paired-runs.csv`：出典と評価方法を含む4,230記録の対応表。`hierarchical.json`：282設定の階層集計。
- `tables/full-exact-stress-overlay.json`：過去の推定Stressを保存座標から厳密に再評価した120記録の対応。
- `exact-stress/cache/`：座標・入力・精度をキーにした厳密再評価結果と出典・ハッシュ。
- `validation/`：保存座標の精度確認、初期状態の対応、旧記録ハッシュ、完了検証。
- `source-snapshot/`：図生成・評価器ソースとハッシュ。

前回v5、EX1、EX2の結果・チャートは保持し、このチャート集を追加保存した。前回には存在しないGPU反復時間・描画時間をCPU/Atomicの図へ捏造して補完しない。
'''
 (BASE/'reports/CHARTS.md').write_text(text)
 atomic_note(VAULT/'EX_チャート集.md',text)
 save(BASE/'figures/index.json',figures)
 print('DONE',len(figures),'charts and Obsidian EX_チャート集.md',flush=True)

if __name__=='__main__':main()
