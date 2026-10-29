#!/usr/bin/env python3
"""Paper figures with unchanged v3 Full measurements and v4 Sparse measurements."""
from pathlib import Path
import csv,json,gzip,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1]/'experiment_results/v4'
OUT=BASE/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
def read(path):return list(csv.DictReader(path.open()))
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=260,bbox_inches='tight',pad_inches=.06)
 plt.close(fig)
def positions(r):
 coords=[];active=False;p=Path(r['final_positions_path'])
 local=BASE/'reference/full-v3/coordinates'/p.name
 if local.exists():p=local
 with gzip.open(p,'rt') if p.suffix=='.gz' else p.open() as f:
  for line in f:
   if line.startswith('# Positions'):active=True;continue
   if active and line.strip() and not line.startswith('#'):
    v=line.split()
    if len(v)==2:coords.append([float(x) for x in v])
 xy=np.array(coords);assert len(xy)==int(r['nodes']);return xy

def main():
 audit=json.loads((BASE/'tables/audit.json').read_text());assert audit['completed']==audit['planned'] and audit['convergence_evaluated']
 times=[r for r in read(BASE/'reference/full-v3/fig05_iteration_times.csv') if r['family']=='full']+read(BASE/'tables/fig05_iteration_times.csv')
 fullgraphs=sorted({r['graph'] for r in times if r['family']=='full'})
 sparsegraphs=sorted(g['name'] for g in json.loads((BASE/'config/datasets.json').read_text()))
 vals=[float(r[k]) for r in times if r['family']=='sparse' for k in ['min_ms_per_iteration','max_ms_per_iteration']]
 lo=10**math.floor(math.log10(min(vals)));hi=10**math.ceil(math.log10(max(vals)))
 f,axs=plt.subplots(3,1,figsize=(7.2,8.7))
 for ax,(fam,graphs,title) in zip(axs,[('full',fullgraphs,'(a) Full (v3, unchanged)'),('sparse',sparsegraphs[:19],'(b) Sparse: CPU v3 / RR v4, graphs 1–19'),('sparse',sparsegraphs[19:],'(c) Sparse: CPU v3 / RR v4, graphs 20–38')]):
  for x,g in enumerate(graphs):
   group=[r for r in times if r['family']==fam and r['graph']==g]
   if not group:ax.text(x,.04,'N/A',rotation=90,ha='center',fontsize=6,transform=ax.get_xaxis_transform());continue
   ax.plot([x,x],[float(r['median_ms_per_iteration']) for r in group],c='.75',lw=.6,zorder=1)
   for r in group:
    med=float(r['median_ms_per_iteration']);bottom=float(r['min_ms_per_iteration']);top=float(r['max_ms_per_iteration']);cpu=r['method']=='CPU'
    ax.errorbar(x,med,yerr=[[med-bottom],[top-med]],fmt='s' if cpu else 'o',c='.4' if cpu else 'black',mfc='white' if cpu else 'black',ms=4,elinewidth=.7,capsize=1.8)
  ax.set_xticks(range(len(graphs)),graphs,rotation=55,ha='right',fontsize=7);ax.set_xlim(-.65,len(graphs)-.35);ax.set_yscale('log');ax.set_ylim((1,2000) if fam=='full' else (lo,hi));ax.grid(axis='y',which='major',c='.88',lw=.6);ax.set_ylabel('Mean time per iteration (ms)',fontsize=8);ax.set_title(title,loc='left',fontsize=9)
 f.legend([plt.Line2D([],[],ls='',marker='s',c='.4',mfc='white'),plt.Line2D([],[],ls='',marker='o',c='black')],['CPU','RR'],ncol=2,frameon=False,loc='upper center')
 f.subplots_adjust(top=.94,bottom=.14,hspace=1.07,left=.105,right=.99);save(f,'fig05_comparison')
 conv=read(BASE/'reference/full-v3/convergence.csv')+read(BASE/'tables/convergence.csv')
 f,axs=plt.subplots(2,3,figsize=(7.2,4.25))
 panels=[('dwt_2680','full'),('wiki-Vote','full'),('EVA','full'),('luxembourg_osm','sparse'),('kron_g500-logn18','sparse'),('coAuthorsCiteseer','sparse')]
 coverage=[]
 for ax,(g,fam) in zip(axs.flat,panels):
  methods=['sgd','atomic_sgd','rr_sgd'] if fam=='full' else ['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd']
  rows=[r for r in conv if r['graph']==g and r['family']==fam]
  cpu_final={int(r['seed']):float(r['stress']) for r in rows if r['method']==methods[0] and int(r['epoch'])==15}
  counts=[]
  for m,ls,col,mark in zip(methods,[':','--','-'],['.4','.55','black'],['^','s','o']):
   seeds={int(r['seed']) for r in rows if r['method']==m and int(r['epoch'])==15}&cpu_final.keys()
   points=[r for r in rows if r['method']==m and int(r['seed']) in seeds];xs=[];ys=[]
   for epoch in [0,1,2,3,5,8,10,15]:
    values=[float(r['stress']) for r in points if int(r['epoch'])==epoch]
    if values:xs.append(epoch);ys.append(np.median(values)/np.median([cpu_final[s] for s in seeds]))
   ax.plot(xs,ys,ls=ls,c=col,marker=mark,ms=3,lw=1);counts.append(len(seeds));coverage.append(dict(graph=g,method=m,seeds=len(seeds),normalization='method median / corresponding CPU final median'))
  ax.axhline(1,c='.65',lw=.5);ax.set_yscale('log');ax.set_title(g,fontsize=8.5);ax.set_xticks([0,5,10,15]);ax.tick_params(labelsize=7)
  if counts!=[10,10,10]:ax.text(.98,.95,'n='+ '/'.join(map(str,counts)),transform=ax.transAxes,ha='right',va='top',fontsize=6)
 for ax in axs[:,0]:ax.set_ylabel('Stress / CPU final median',fontsize=8)
 for ax in axs[1]:ax.set_xlabel('Iteration')
 f.legend([plt.Line2D([],[],c=c,ls=l,marker=m) for c,l,m in zip(['.4','.55','black'],[':','--','-'],['^','s','o'])],['CPU','Atomic','RR'],ncol=3,loc='upper center',frameon=False,fontsize=8)
 f.text(.99,.01,'Top: Full v3. Bottom: CPU/Atomic v3; RR v4.',ha='right',fontsize=6.5)
 f.subplots_adjust(top=.85,bottom=.12,left=.1,right=.99,hspace=.48,wspace=.27);save(f,'fig06_convergence')
 (BASE/'tables/figure06_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
 runs=read(BASE/'reference/full-v3/runs.csv')+read(BASE/'tables/runs.csv')
 selected=[r for r in runs if r['study']=='basic' and r['status']=='success' and (r['graph'],r['method']) in [('delaunay_n14','rr_sgd'),('delaunay_n19','rr_sparse_sgd')]]
 ncols=max(4,max(sum(r['graph']==g for r in selected) for g in ['delaunay_n14','delaunay_n19']))
 f,axs=plt.subplots(2,ncols,figsize=(7.2,4.5))
 counts=[]
 for row,g in enumerate(['delaunay_n14','delaunay_n19']):
  rs=sorted([r for r in selected if r['graph']==g],key=lambda r:int(r['seed']));arrays=[positions(r) for r in rs];counts.append(len(rs))
  if arrays:
   bounds=np.vstack([np.array([x.min(axis=0),x.max(axis=0)]) for x in arrays]);low=bounds.min(axis=0);high=bounds.max(axis=0)
  for col,ax in enumerate(axs[row]):
   ax.axis('off')
   if col>=len(rs):continue
   xy=arrays[col];r=rs[col];ax.scatter(xy[:,0],xy[:,1],s=.13 if row==0 else .025,c='black',linewidths=0,rasterized=True);ax.set_aspect('equal');ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1]);ax.set_title(f"seed {r['seed']} | {float(r['iteration_time_ms'])/1000:.3f} s",fontsize=7)
  if not arrays:axs[row,0].text(.5,.5,'No completed runs',ha='center')
  axs[row,0].text(-.08,.5,g,rotation=90,ha='center',va='center',transform=axs[row,0].transAxes,fontsize=9)
 f.text(.52,.96,f'Full RR (v3): {counts[0]}/5 completed (16,384 vertices)',ha='center',fontsize=10)
 f.text(.52,.48,f'Sparse RR (v4): {counts[1]}/5 completed (524,288 vertices)',ha='center',fontsize=10)
 f.text(.5,.01,'All completed RR layouts; vertices only; common scale within each graph. Times: 15 iterations.',ha='center',fontsize=7)
 f.subplots_adjust(top=.87,bottom=.06,hspace=.55,wspace=.15,left=.08,right=.99);save(f,'fig07_large_layouts')
 # Supplement: quality and speed for every planned Sparse graph, including Atomic.
 pairrows=read(BASE/'tables/paired-ratios.csv')
 f,axs=plt.subplots(2,2,figsize=(10,7.2),sharex='col')
 for col,graphs in enumerate([sparsegraphs[:19],sparsegraphs[19:]]):
  for x,g in enumerate(graphs):
   for method,offset,marker,color in [('atomic_sparse_sgd',-.12,'s','.55'),('rr_sparse_sgd',.12,'o','black')]:
    rows=[r for r in pairrows if r['study']=='basic' and r['graph']==g and r['method']==method]
    for row,key in [(0,'speedup'),(1,'stress_ratio')]:
     ax=axs[row,col]
     if not rows:
      ax.text(x+offset,.03,'×',transform=ax.get_xaxis_transform(),ha='center',fontsize=7,color=color);continue
     values=[float(r[key]) for r in rows];med=np.median(values)
     ax.errorbar(x+offset,med,yerr=[[med-min(values)],[max(values)-med]],fmt=marker,c=color,mfc='white' if method=='atomic_sparse_sgd' else color,ms=3,capsize=1,elinewidth=.6)
  for row in range(2):
   ax=axs[row,col];ax.axhline(1,c='.6',lw=.7);ax.grid(axis='y',c='.9',lw=.5);ax.set_xlim(-.7,len(graphs)-.3)
   ax.set_xticks(range(len(graphs)),graphs,rotation=60,ha='right',fontsize=7)
  axs[0,col].set_yscale('log');axs[1,col].axhline(1.05,c='.6',ls=':',lw=.7)
  axs[0,col].set_title(f'Sparse v4: graphs {1+19*col}–{19+19*col}',fontsize=10)
 axs[0,0].set_ylabel('CPU / GPU iteration time');axs[1,0].set_ylabel('GPU / CPU final stress')
 for row in range(2):
  low=min(axs[row,c].get_ylim()[0] for c in range(2));high=max(axs[row,c].get_ylim()[1] for c in range(2))
  for c in range(2):axs[row,c].set_ylim(low,high)
 f.legend([plt.Line2D([],[],ls='',marker='s',c='.55',mfc='white'),plt.Line2D([],[],ls='',marker='o',c='black')],['Atomic (v3)','RR (v4)'],ncol=2,frameon=False,loc='upper center')
 f.text(.99,.01,'Baseline: CPU v3. Points: paired-seed medians; bars: min–max. Stress: fixed 256-source estimate.',ha='right',fontsize=6.5)
 f.subplots_adjust(top=.91,bottom=.23,left=.08,right=.99,hspace=.13,wspace=.14);save(f,'sparse_speed_stress')
 print('Generated figures 5–7 and Sparse quality supplement; Full data inherited from v3')
if __name__=='__main__':main()
