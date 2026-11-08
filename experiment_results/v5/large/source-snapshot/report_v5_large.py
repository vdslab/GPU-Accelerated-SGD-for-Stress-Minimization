#!/usr/bin/env python3
"""Offline v5 large report. Refuses to draw until every planned condition is recorded."""
import collections,csv,gzip,json,lzma,math,shutil,statistics
from pathlib import Path
import numpy as np
from run_v5_large import BASE,VAULT,now,save

PIVOTS=[1000,500,200,100,50]
FIELDS=['iteration_time_ms','total_compute_time_ms','stress_value','peak_rss_bytes']

def csvwrite(path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def distribution(values):
 return dict(median=statistics.median(values),min=min(values),max=max(values)) if values else dict(median=None,min=None,max=None)

def layout_sample(path,n,m):
 """Keep all coordinates for sampled edge endpoints; never retain every edge."""
 opener=lzma.open if path.suffix=='.xz' else gzip.open if path.suffix=='.gz' else open
 positions=np.empty((n,2),dtype=np.float64);edges=[];edge_stride=max(1,math.ceil(m/50000));vertex_stride=max(1,math.ceil(n/100000))
 state=None;ei=0;vi=0
 with opener(path,'rt') as f:
  for line in f:
   if line.startswith('# Edges'):state='edges';continue
   if line.startswith('# Positions'):state='positions';continue
   if not line.strip() or line.startswith('#'):continue
   if state=='edges':
    if ei%edge_stride==0:edges.append(tuple(map(int,line.split())))
    ei+=1
   elif state=='positions':
    positions[vi]=[float(x) for x in line.split()];vi+=1
 assert ei==m and vi==n and np.isfinite(positions).all(),(path,ei,vi)
 return positions,np.array(edges,dtype=np.int64),vertex_stride,edge_stride

def main():
 jobs=json.loads((BASE/'config/jobs.json').read_text());graphs=json.loads((BASE/'config/datasets.json').read_text())
 paths=list((BASE/'raw').glob('*/record.json'));rows=[json.loads(p.read_text()) for p in paths]
 assert len(rows)==len({r['run_id'] for r in rows})==len(jobs)==350,'All 350 outcomes must be recorded before offline reporting'
 assert {r['run_id'] for r in rows}=={j['run_id'] for j in jobs}
 gmap={g['name']:g for g in graphs};expected=json.loads((BASE/'config/environment.json').read_text())['binary_sha256'];archive_count=0
 for path,row in zip(paths,rows):
  if row['status']=='success':
   assert row['binary_sha256']==expected and row['input_sha256']==gmap[row['graph']]['sha256']
   assert row['iterations']==15 and row['epsilon']==.1 and row['rr_phase_chunks']==1
   assert row['seed'] in range(5) and row['repetition']==0
   assert row['stress_samples']==64 and row['stress_seed']==0 and row['stress_kind']=='sampled'
   assert row['total_compute_time_ms']>0 and row['iteration_time_ms']>0 and math.isfinite(row['stress_value'])
   assert row['archive_status']=='success'
   manifest=json.loads((path.parent/'artifact-archive.json').read_text());assert not manifest['errors']
   for entry in manifest['files']:
    p=Path(entry['compressed_path']);assert p.is_file() and p.stat().st_size==entry['compressed_bytes']
    assert entry['original_sha256']==entry['decompressed_sha256']
    assert entry.get('gzip_crc_validated') or entry.get('xz_crc_validated');archive_count+=1
  elif row['status']=='preflight_unavailable':
   assert not row['executed'] and row['preflight']['hard_limit_reasons']
 success=[r for r in rows if r['status']=='success'];groups=collections.defaultdict(list)
 for r in rows:groups[(r['graph'],r['pivots'])].append(r)
 summary=[];summary_index={}
 for g in graphs:
  for p in PIVOTS:
   rs=groups[(g['name'],p)];assert sorted(r['seed'] for r in rs)==list(range(5))
   good=[r for r in rs if r['status']=='success']
   item=dict(graph=g['name'],nodes=g['nodes'],edges=g['edges'],pivots=p,success=len(good),planned=5,statuses=dict(collections.Counter(r['status'] for r in rs)),aggregation='one run per seed; median/min/max across successful seeds only')
   for field in FIELDS:
    for name,value in distribution([r[field] for r in good]).items():item[field+'_'+name]=value
   summary.append(item);summary_index[(g['name'],p)]=item
 baseline={g['name']:max([p for p in PIVOTS if summary_index[(g['name'],p)]['success']==5],default=None) for g in graphs}
 index={(r['graph'],r['pivots'],r['seed']):r for r in success};ratios=[]
 for r in success:
  bp=baseline[r['graph']]
  if bp is None:continue
  base=index[(r['graph'],bp,r['seed'])]
  ratios.append(dict(graph=r['graph'],pivots=r['pivots'],seed=r['seed'],reference_pivots=bp,stress_ratio=r['stress_value']/base['stress_value'] if base['stress_value'] else None,initial_positions_match=r['initial_positions_sha256']==base['initial_positions_sha256']))
 for item in summary:
  vals=[r['stress_ratio'] for r in ratios if r['graph']==item['graph'] and r['pivots']==item['pivots'] and r['stress_ratio'] is not None]
  for k,v in distribution(vals).items():item['paired_stress_ratio_'+k]=v
 retry_paths=list((BASE/'validation/retried').glob('*/record.json'));retries=[dict(json.loads(p.read_text()),record_path=str(p)) for p in retry_paths]
 csvwrite(BASE/'tables/runs.csv',rows);csvwrite(BASE/'tables/summary.csv',summary);csvwrite(BASE/'tables/paired-pivot-ratios.csv',ratios);csvwrite(BASE/'tables/retried-attempts.csv',retries)
 save(BASE/'tables/summary.json',summary)
 initial=[]
 for g in graphs:
  for seed in range(5):
   rs=[r for r in success if r['graph']==g['name'] and r['seed']==seed]
   if len(rs)>1:initial.append(dict(graph=g['name'],seed=seed,pivots=[r['pivots'] for r in rs],distinct_initial_positions=len({r['initial_positions_sha256'] for r in rs})))
 save(BASE/'tables/initial-position-check.json',initial)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.collections import LineCollection
 from matplotlib.colors import ListedColormap
 from matplotlib.lines import Line2D
 plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
 figdir=BASE/'figures';figdir.mkdir(exist_ok=True);figures=[]
 def export(fig,name):
  fig.savefig(figdir/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight');plt.close(fig);figures.append(name)
 fig,ax=plt.subplots(figsize=(9,7));matrix=np.zeros((14,5),int);labels=[]
 for i,g in enumerate(graphs):
  rowlabels=[]
  for j,p in enumerate(PIVOTS):
   item=summary_index[(g['name'],p)];s=item['success'];states=item['statuses']
   matrix[i,j]=0 if states.get('preflight_unavailable')==5 else 3 if s==5 else 2 if s else 1
   rowlabels.append('GPU limit' if matrix[i,j]==0 else f'{s}/5')
  labels.append(rowlabels)
 ax.imshow(matrix,cmap=ListedColormap(['#d1d5db','#fecaca','#fde68a','#bbf7d0']),vmin=0,vmax=3,aspect='auto')
 for i in range(14):
  for j in range(5):ax.text(j,i,labels[i][j],ha='center',va='center',fontsize=8)
 ax.set_xticks(range(5),PIVOTS);ax.set_yticks(range(14),[g['name'] for g in graphs]);ax.set_xlabel('Pivot count');ax.set_title('RR Sparse: success / 5 seeds; GPU limit = not executed')
 export(fig,'large-completion')
 metrics=[('iteration_time_ms',.001,'SGD (15 iterations), s','sgd'),('total_compute_time_ms',.001,'Total compute, s','total'),('peak_rss_bytes',1/2**30,'Peak process RSS, GiB','rss'),('paired_stress_ratio',1,'Stress / same-seed reference (largest p with 5/5)','stress-ratio'),('stress_value',1,'Estimated final Full Stress (64 sources)','stress')]
 for field,scale,ylabel,name in metrics:
  fig,axes=plt.subplots(4,4,figsize=(15,11));axes=axes.ravel()
  for ax,g in zip(axes,graphs):
   items=sorted([a for a in summary if a['graph']==g['name'] and a['success'] and a[field+'_median'] is not None],key=lambda a:a['pivots'])
   if items:
    x=[a['pivots'] for a in items];v=[a[field+'_median']*scale for a in items];low=[(a[field+'_median']-a[field+'_min'])*scale for a in items];high=[(a[field+'_max']-a[field+'_median'])*scale for a in items]
    ax.errorbar(x,v,yerr=[low,high],marker='o',color='#059669',capsize=3);ax.set_xscale('log');ax.set_xticks(sorted(PIVOTS),sorted(PIVOTS));ax.minorticks_off()
    if field=='paired_stress_ratio':ax.axhline(1,color='#9ca3af',ls='--',lw=.8)
    else:ax.set_yscale('log')
   else:
    graph_rows=[a for a in rows if a['graph']==g['name']]
    message='GPU limit: not executed' if all(a['status']=='preflight_unavailable' for a in graph_rows) else 'No complete reference' if field=='paired_stress_ratio' and any(a['status']=='success' for a in graph_rows) else 'No successful measurement'
    ax.text(.5,.5,message,ha='center',va='center',transform=ax.transAxes,fontsize=8)
   ax.set_title(g['name']);ax.grid(alpha=.2);ax.set_xlabel('p')
  for ax in axes[len(graphs):]:ax.axis('off')
  fig.suptitle(ylabel+'; median and min-max of successful seeds',fontsize=13);fig.tight_layout(rect=(0,0,1,.96));export(fig,'large-p-'+name)
 for field,scale,ylabel,name in metrics[:3]:
  fig,ax=plt.subplots(figsize=(8,5))
  colors={p:plt.cm.viridis(i/4) for i,p in enumerate(reversed(PIVOTS))}
  markers={'Road':'o','SNS':'s','Mesh / FE':'^'}
  def family(graph):return 'Road' if 'osm' in graph or graph=='road_usa' else 'SNS' if graph.startswith('com-') else 'Mesh / FE'
  for p in reversed(PIVOTS):
   for group,marker in markers.items():
    items=sorted([a for a in summary if a['pivots']==p and a['success'] and family(a['graph'])==group],key=lambda a:a['nodes'])
    if items:
     med=[a[field+'_median']*scale for a in items];lo=[(a[field+'_median']-a[field+'_min'])*scale for a in items];hi=[(a[field+'_max']-a[field+'_median'])*scale for a in items]
     ax.errorbar([a['nodes'] for a in items],med,yerr=[lo,hi],fmt=marker,color=colors[p],linestyle='none',capsize=2,alpha=.85)
  handles=[Line2D([],[],marker='o',ls='',color=colors[p],label=f'p={p}') for p in reversed(PIVOTS)]
  handles += [Line2D([],[],marker=marker,ls='',color='#4b5563',label=group) for group,marker in markers.items()]
  ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Vertices (different graphs)');ax.set_ylabel(ylabel);ax.legend(handles=handles,ncol=2,fontsize=8);ax.grid(alpha=.2);export(fig,'large-scale-'+name)
 layout_info=[]
 for g in graphs:
  bp=baseline[g['name']]
  if bp is None:continue
  row=index[(g['name'],bp,0)];points,edges,vstride,estride=layout_sample(Path(row['final_positions_path']),g['nodes'],g['edges'])
  fig,ax=plt.subplots(figsize=(7,7));ax.add_collection(LineCollection(points[edges],colors='#9ca3af',linewidths=.12,alpha=.25,rasterized=True))
  sample=points[::vstride];ax.scatter(sample[:,0],sample[:,1],s=.15,c='#047857',alpha=.65,rasterized=True)
  for dimension,setter in [(0,ax.set_xlim),(1,ax.set_ylim)]:
   lo,hi=float(points[:,dimension].min()),float(points[:,dimension].max());pad=max(hi-lo,1e-9)*.02;setter(lo-pad,hi+pad)
  ax.set_aspect('equal');ax.set_axis_off()
  ax.set_title(f'{g["name"]}, p={bp}, seed=0\n{len(sample):,}/{g["nodes"]:,} vertices; {len(edges):,}/{g["edges"]:,} edges sampled',fontsize=9)
  name='large-layout-'+g['name'];export(fig,name)
  layout_info.append(dict(graph=g['name'],pivots=bp,seed=0,vertex_stride=vstride,edge_stride=estride,vertices_drawn=len(sample),edges_drawn=len(edges),rule='Every ceil(n/100000)-th vertex and ceil(m/50000)-th edge in native order; edge endpoints use all coordinates. Independent full extents per graph. No cropping.'))
 save(BASE/'tables/layout-sampling.json',layout_info)
 statuses=dict(collections.Counter(r['status'] for r in rows));failed=sum(v for k,v in statuses.items() if k not in ['success','preflight_unavailable'])
 text=f'# v5 超大規模グラフ実験・結果と考察\n\n更新: {now()}\n\n14グラフ・5pivot・5seedの350条件を確定した。実測成功{len(success)}、GPU上限の未実行{statuses.get("preflight_unavailable",0)}、本測定の実測失敗{failed}。別に保全した再試行前の試行は{len(retries)}件。通常6030件とは別集計で、CPU/Atomicは測定していない。\n\n'
 text+='| グラフ | p | 成功/5 | SGD15反復 s（中央値） | 総計算 s（中央値） | 推定Stress（中央値） | RSS GiB（中央値） |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for a in summary:
  values=[a[k+'_median'] for k in FIELDS];display=['—' if v is None else f'{v*factor:.4g}' for v,factor in zip(values,[.001,.001,1,1/2**30])]
  text+=f'| {a["graph"]} | {a["pivots"]} | {a["success"]}/5 | '+ ' | '.join(display)+' |\n'
 text+='\n## 適用可能性\n\n'
 for g in graphs:
  bp=baseline[g['name']];text+=f'- {g["name"]}: '+(f'5seed完了した最大pは{bp}。' if bp else '全pがGPU上限により未実行であり、実測の成功・失敗は評価していない。')+'\n'
 if success:
  largest=max((g for g in graphs if baseline[g['name']] is not None),key=lambda g:g['nodes'],default=None)
  if largest:text+=f'\n今回5seed完了を確認した最大入力は{largest["name"]}（{largest["nodes"]:,}頂点）だった。これは現機材・実装・指定pの観測結果で、GPU一般の最大規模ではない。\n'
 text+='\nGPU事前判定はone-sided配列8np bytesと実測照会したstorage binding上限4,294,967,295 bytes、頂点dispatch上限65,535 workgroups（256頂点/group）に基づく。p50でも千万頂点級は配列上限を超える。CPUを測っていないのでCPUで不可能・CPU比高速とは結論しない。\n\n## 時間と品質の交換関係\n\n'
 for g in graphs:
  complete=[a for a in summary if a['graph']==g['name'] and a['success']==5]
  if len(complete)<2:continue
  high=max(complete,key=lambda a:a['pivots']);low=min(complete,key=lambda a:a['pivots'])
  best=min(complete,key=lambda a:a['stress_value_median'])
  text+=f'- {g["name"]}: p{high["pivots"]}/p{low["pivots"]}の総計算時間中央値比{high["total_compute_time_ms_median"]/low["total_compute_time_ms_median"]:.3f}、Stress中央値比{high["stress_value_median"]/low["stress_value_median"]:.4f}。測定範囲でStress中央値が最小だったのはp{best["pivots"]}。初期配置・pivot選定も変わるため、pだけの因果効果や最適pの確定とは扱わない。\n'
 for g in graphs:
  complete=[a for a in summary if a['graph']==g['name'] and a['success']==5]
  if len(complete)==1:
   a=complete[0]
   text+=f'- {g["name"]}: p{a["pivots"]}のみ5seed成功した。ほかのpはGPU上限で未実行なので、p間の時間・品質の交換関係は評価できない。\n'
 shares=[r['iteration_time_ms']/r['total_compute_time_ms'] for r in success]
 text+=f'\n成功130実行でのSGD/総計算時間の比率は中央値{statistics.median(shares)*100:.2f}%、最小{min(shares)*100:.2f}%、最大{max(shares)*100:.2f}%だった。この全runを同じ重みで集約した記述量からも、SGD以外の入力・前処理・準備等を含む総計算時間を併記する必要がある。\n'
 text+='\n上記の記述比は中央値同士の比較。図のStress比はgraph・seedごとに最大の5seed成功pへ対応付けてから5seedを集計する。SGD時間だけでなく、入力・前処理・GPU準備等を含むtotal_compute_time_msを見る。Stress評価・ファイル保存・圧縮・描画は総計算時間の外側であり、実際の待ち時間はprocess_wall_seconds＋archive_seconds＋shared_repack_seconds等も確認する。\n\n## 批判的な確認と限界\n\n'
 mismatches=sum(a['distinct_initial_positions']>1 for a in initial)
 livejournal=summary_index.get(('com-LiveJournal',100))
 if livejournal and livejournal['success']==5:
  text+=f'- com-LiveJournal p100の総計算時間は中央値{livejournal["total_compute_time_ms_median"]/1000:.2f}秒、最大{livejournal["total_compute_time_ms_max"]/1000:.2f}秒だった。遅い実行も除外せずmin-maxへ含めた。実行環境・処理内容のどの要因による差かをこの測定だけで特定していない。\n'
 orkut_high=summary_index.get(('com-Orkut',100));orkut_low=summary_index.get(('com-Orkut',50))
 if orkut_high and orkut_low and orkut_high['success']==orkut_low['success']==5:
  text+=f'- com-OrkutのRSS中央値はp100で{orkut_high["peak_rss_bytes_median"]/2**30:.2f}GiB、p50で{orkut_low["peak_rss_bytes_median"]/2**30:.2f}GiBだった。配列の理論サイズとプロセスRSSは異なり、pを減らすと実測RSSも必ず単調に下がるとは結論できない。\n'
 recovered=[r for r in success if r.get('recovery_evidence')]
 if recovered:
  text+=f'- 保存処理中の管理プロセス中断から{len(recovered)}件の正常なnative出力を復旧した。レイアウトを再実行せず、元のJSON・初期/最終座標・圧縮復元SHA256を確認した。子プロセス終了コードと終了直後のメモリは未保存のため欠測として残し、process_wall_secondsは/usr/bin/timeの0.01秒精度値を用いた。停止原因は特定できず、中断中の待ち時間はnative計算時間へ加算しない。証拠は各runのrecovery_evidenceを参照する。\n'
 text+=f'- 初期座標fingerprint: 複数pが成功したgraph×seedの{len(initial)}組中{mismatches}組で異なる初期座標を確認した。同seedだけで初期配置一致とみなさず、nativeの乱数消費とpivot選定を保持した結果として比較する。\n'
 text+='- 各seed1回のため通常比較の各seed3回とは時間再現性の評価水準が異なる。5seedのmin-maxはレイアウトseedによる分布で、反復測定の信頼区間ではない。\n- 64始点・評価seed0を固定した最終推定Full Stressを用いる。評価源ごとの寄与や推定標準誤差はnative出力にないため、5seedの幅をStress推定誤差と解釈せず小差を断定しない。通常比較は256始点なので絶対Stressの直接混合も避ける。\n- p降順・グラフ順の逐次測定で温度・OSキャッシュ・swap等の順序効果が残る。Codex常駐のためOSのみの最小メモリ環境ではない。RSSはプロセス指標で、統合メモリ上のGPU割当を単独で厳密に表さない。\n- 異なる国の道路網やSNS・メッシュは構造と辺数も異なる。同じグラフの頂点数だけを増やした純粋なスケーリング実験ではなく、SNSのスケールフリー性をここで実証したわけではない。\n- ディスク容量停止を挟み、保存形式にはgzip・検証済み可逆XZを使い、com-Orkutは全byte同一の標準gzip連結memberをAPFSで物理共有して容量を節約した。共有対象は保存用の辺データで、GPU配列の容量条件は従来どおり。レイアウト・Stressのバイナリは不変で、保存・圧縮処理はnative計時の外。中断試行は本測定へ混ぜず別表に残した。Git objectの複製抑止と保全圧縮も測定停止中のみ実行した。\n- レイアウト図はseed0・5seed成功した最大pを採用し、頂点/辺の決定的間引きと独立スケールを明示する。細部や密度の厳密比較に使わない。\n\n## 図表と生データ\n\n正本: `'+str(BASE)+'`。全350状態と全成功runのnative値はtables/runs.csv、各pの集計はsummary.csv、対応seedの品質比はpaired-pivot-ratios.csv、再試行前記録はretried-attempts.csv。\n'
 for name in figures:text+='\n!['+name+'](../figures/'+name+'.png)\n'
 (BASE/'reports/large.md').write_text(text)
 vaultfig=VAULT/'v5_図表/large';vaultfig.mkdir(parents=True,exist_ok=True)
 for name in figures:shutil.copy2(figdir/(name+'.png'),vaultfig/(name+'.png'))
 vault_text=text
 for name in figures:vault_text=vault_text.replace('!['+name+'](../figures/'+name+'.png)','![[v5_図表/large/'+name+'.png]]')
 (VAULT/'4_大規模グラフ実験考察.md').write_text('---\ntags: [area/大学院研究, 実験考察]\nupdated: '+now()[:10]+'\nstatus: 測定完了\n---\n\n[[4_大規模グラフ実験]] / [[0_概要]]\n\n'+vault_text)
 certificate=dict(at=now(),planned=350,statuses=statuses,retried_attempts=len(retries),archive_files_verified_by_manifest_and_size=archive_count,initial_mismatch_groups=mismatches,figures=figures,report='reports/large.md',obsidian_saved=True)
 save(BASE/'validation/report-completion.json',certificate)
 def section(path,body):
  start='<!-- V5_LARGE_REPORT_START -->';end='<!-- V5_LARGE_REPORT_END -->';content=path.read_text()
  block=start+'\n\n'+body+'\n\n'+end
  if start in content:
   before,tail=content.split(start,1);_,after=tail.split(end,1);content=before+block+after
  else:content=content.rstrip()+'\n\n'+block+'\n'
  path.write_text(content)
 completed=f'## 超大規模v5の測定・保存完了\n\n更新: {now()}。350条件のうち成功{len(success)}、GPU上限の未実行{statuses.get("preflight_unavailable",0)}、実測失敗{failed}、保全した再試行前試行{len(retries)}。結果・{len(figures)}図のPNG/PDF・考察をv5/largeとObsidianへ保存。通常6030件の結果は保持。[[4_大規模グラフ実験考察]]参照。'
 section(BASE/'README.md',completed.replace('[[4_大規模グラフ実験考察]]','[結果と考察](reports/large.md)'))
 for path in [VAULT/'4_大規模グラフ実験.md',VAULT/'0_概要.md']:section(path,completed)
 state=json.loads((BASE/'progress.json').read_text());state.update(reporting_complete=True,reporting_completed_at=now(),report='reports/large.md');save(BASE/'progress.json',state)
 print(json.dumps(certificate,ensure_ascii=False))

if __name__=='__main__':main()
