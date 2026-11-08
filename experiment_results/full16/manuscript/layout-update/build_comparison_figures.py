#!/usr/bin/env python3
"""Paper figures from audited Full segment measurements and retained Sparse results."""
from pathlib import Path
import csv,json,gzip,math,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization/experiment_results/full16/paper-data')
OUT=Path(__file__).resolve().parents[1]/'figures';OUT.mkdir(exist_ok=True)
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
 # Pair each benchmark run with the sequential run for the same graph and seed.
 full = [r for r in read(BASE/'reference/full-v3/runs.csv')
         if r['study']=='basic' and r['status']=='success']
 cpu = {(r['graph'],r['seed']):r for r in full if r['method']=='sgd'}
 ratios = []
 for r in full:
  key = (r['graph'],r['seed'])
  if r['method']=='rr_sgd' and key in cpu:
   ref = cpu[key]
   ratios.append((r['graph'],float(ref['iteration_time_ms'])/float(r['iteration_time_ms']),
                  float(r['stress_value'])/float(ref['stress_value'])))
 sparse = [(r['graph'],float(r['speedup']),float(r['stress_ratio']))
           for r in read(BASE/'tables/paired-ratios.csv')
           if r['study']=='basic' and r['method']=='rr_sparse_sgd']
 fig, axes = plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained')
 for ax,rows,title in zip(axes,[ratios,sparse],['Full','Sparse']):
  ax.axhline(1.05,color='.4',ls='--',lw=.9)
  ax.axvline(1,color='.4',ls=':',lw=.9)
  for graph in sorted({r[0] for r in rows}):
   pairs=[r for r in rows if r[0]==graph]
   x,y=np.median([r[1:] for r in pairs],axis=0)
   ax.scatter(x,y,s=18,c='black',alpha=.75)
   if graph in ('wiki-Vote','fe_4elt2'):
    ax.annotate(graph,(x,y),xytext=(-5,3 if graph=='wiki-Vote' else -10),
                textcoords='offset points',ha='right',fontsize=7)
  ax.set(xscale='log',xlim=(.8,65),xlabel='Iteration speedup over CPU',title=title)
  ax.margins(y=.18)
 axes[0].set_ylabel('Final full stress / CPU final full stress')
 save(fig,'full_sparse_quality')
 conv=read(BASE/'reference/full-v3/convergence.csv')+read(BASE/'tables/convergence.csv')
 fig,axes=plt.subplots(1,2,figsize=(7.2,2.1),layout='constrained',sharey=True)
 for ax,graph in zip(axes,['fe_4elt2','wiki-Vote']):
  rows=[r for r in conv if r['family']=='full' and r['graph']==graph]
  baseline={r['seed']:float(r['stress']) for r in rows if r['method']=='sgd' and r['epoch']=='15'}
  for method,label,style in [('sgd','CPU','--'),('rr_sgd','RoundRobin Full SGD','-')]:
   for seed,final in baseline.items():
    points=sorted((float(r['elapsed_ms'])/1000,float(r['stress'])/final) for r in rows
                  if r['method']==method and r['seed']==seed and int(r['epoch'])>0)
    ax.plot(*zip(*points),ls=style,color='.45' if method=='sgd' else 'black',
            lw=.65,alpha=.65,label=label if seed==next(iter(baseline)) else None)
  ax.axhline(1.05,color='#b25326',lw=1)
  ax.set(xscale='log',yscale='log',title=graph,xlabel='Elapsed SGD time (s)')
 axes[0].set_ylabel('Stress / CPU final stress',fontsize=8)
 axes[1].set_ylabel('Stress / CPU final stress',fontsize=8)
 for ax in axes:
  ax.tick_params(labelsize=7)
  ax.xaxis.label.set_size(8)
  ax.title.set_size(9)
 axes[1].legend(frameon=False,fontsize=7)
 save(fig,'full_time_to_quality')
 f,axs=plt.subplots(2,3,figsize=(7.2,5.15))
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
    if values:xs.append(epoch);ys.append(np.median(values))
   ax.plot(xs,ys,ls=ls,c=col,marker=mark,ms=3,lw=1);counts.append(len(seeds));coverage.append(dict(graph=g,method=m,seeds=len(seeds),normalization='none; absolute stress median'))
  ax.set_yscale('log');ax.set_title(g,fontsize=8.5);ax.set_xticks([0,5,10,15]);ax.tick_params(labelsize=7)
  if counts!=[10,10,10]:ax.text(.98,.95,'n='+ '/'.join(map(str,counts)),transform=ax.transAxes,ha='right',va='top',fontsize=6)
 for ax in axs[:,0]:ax.set_ylabel('Stress',fontsize=8)
 for ax in axs[1]:ax.set_xlabel('Iteration')
 for row,family in enumerate(['Full','Sparse']):
  axs[row,1].legend([plt.Line2D([],[],c=c,ls=l,marker=m) for c,l,m in zip(['.4','.55','black'],[':','--','-'],['^','s','o'])],['CPU','Atomic',f'RoundRobin {family} SGD'],ncol=3,loc='lower center',bbox_to_anchor=(.5,1.16),frameon=False,fontsize=7)
 f.subplots_adjust(top=.85,bottom=.12,left=.1,right=.99,hspace=.7,wspace=.42);save(f,'fig06_convergence')


if __name__=='__main__':main()
