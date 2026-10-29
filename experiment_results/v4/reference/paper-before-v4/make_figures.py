from pathlib import Path
import csv, json, gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
ROOT=Path(__file__).resolve().parent
DATA=Path('/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
def save(fig,name):
 for ext in ['pdf','svg','png']: fig.savefig(ROOT/f'{name}.{ext}',dpi=260,bbox_inches='tight',pad_inches=.06)
 plt.close(fig)
def plain(ax): ax.set(xlim=(0,10),ylim=(0,6)); ax.axis('off')
def txt(ax,x,y,s,fs=9,**kw): ax.text(x,y,s,fontsize=fs,ha='center',va='center',**kw)
def box(ax,x,y,w,h,s,fc='white',fs=9):
 ax.add_patch(Rectangle((x,y),w,h,facecolor=fc,edgecolor='black',lw=.8));txt(ax,x+w/2,y+h/2,s,fs)
def arr(ax,x1,y1,x2,y2,style='-'): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->',mutation_scale=10,lw=.9,linestyle=style,color='black'))
# 1 Atomic schedule: example, not trace.
f,axs=plt.subplots(1,2,figsize=(7.2,3.2),gridspec_kw={'width_ratios':[1,1.5]});a,b=axs
plain(a);plain(b)
a.set_title('(a) Shared endpoint'); b.set_title('(b) Locks serialize conflicting updates')
for x,y,l in [(2,4.8,'i'),(5,3,'j'),(8,4.8,'k'),(2,1,'u'),(8,1,'v')]:
 a.scatter([x],[y],s=400,fc='white',ec='black',lw=1,zorder=3);txt(a,x,y,l,10,zorder=4)
for x,y,X,Y in [(2.4,4.5,4.6,3.3),(5.4,3.3,7.6,4.5),(2.5,1,7.5,1)]:a.plot([x,X],[y,Y],color='black',lw=1)
txt(a,3,5.8,'A = (i, j)');txt(a,7,5.8,'B = (j, k)');txt(a,5,.15,'C = (u, v): independent')
for y,label in [(4.8,'A'),(3,'B'),(1.2,'C')]:txt(b,.25,y,label,10)
for x,w,s in [(1,1.65,'Lock\ni, j'),(2.65,3.1,'Read / update'),(5.75,1.4,'Unlock')]: box(b,x,4.35,w,.9,s,fs=7)
b.set_xlim(0,11.5)
box(b,1,2.55,6.15,.9,'j locked: release partial lock;\nretry until available',fc='.92',fs=7)
box(b,7.15,2.55,1.4,.9,'Lock\nj, k',fs=7);box(b,8.55,2.55,2.55,.9,'Update /\nunlock',fs=7)
box(b,1,.75,1.65,.9,'Lock\nu, v',fs=7);box(b,2.65,.75,3.1,.9,'Read / update',fs=7);box(b,5.75,.75,1.4,.9,'Unlock',fs=7)
b.plot([7.15,7.15],[3.5,4.35],ls=':',color='black');arr(b,1,.3,9.6,.3);txt(b,5,-.15,'Execution order (schematic)',8)
f.subplots_adjust(wspace=.25,top=.88,bottom=.13);save(f,'fig01_atomic')
# 2 outer / inner full RR.
f=plt.figure(figsize=(7.2,3.75));gs=f.add_gridspec(1,3,width_ratios=[1.05,1.1,1.2],wspace=.36);a,b,c=[f.add_subplot(gs[0,i]) for i in range(3)]
a.set_title('(a) Block-pair matrix');a.set(xlim=(0,4),ylim=(4,0),aspect='equal');a.xaxis.tick_top();a.set_xticks(np.arange(4)+.5,[r'$B_1$',r'$B_2$',r'$B_3$',r'$B_4$']);a.set_yticks(np.arange(4)+.5,[r'$B_1$',r'$B_2$',r'$B_3$',r'$B_4$']);a.tick_params(length=0)
rnd={(0,1):1,(2,3):1,(0,2):2,(1,3):2,(0,3):3,(1,2):3}
for i in range(4):
 for j in range(4):
  a.add_patch(Rectangle((j,i),1,1,ec='.35',fc=('.94' if j<i else 'white'),lw=.6))
  if i==j:
   a.plot([j,j+1],[i,i+1],c='.45',lw=.6);a.text(j+.68,i+.32,'S',ha='center',va='center')
  elif j>i:a.text(j+.5,i+.5,f'R{rnd[i,j]}',ha='center',va='center')
plain(b);b.set_title('(b) Outer rounds')
for y,s in [(4.8,'R1: (B1,B2), (B3,B4)'),(3.5,'R2: (B1,B3), (B2,B4)'),(2.2,'R3: (B1,B4), (B2,B3)'),(.9,'S: all self tiles')]:box(b,.1,y-.4,9.8,.8,s,fs=8)
txt(b,5,5.65,'Disjoint blocks run in parallel',8)
c.set_title('(c) Inside tile (B1, B2)');c.set(xlim=(-.5,4.5),ylim=(5.5,-.8),aspect='equal');c.axis('off')
for i in range(4):
 for j in range(4):
  c.add_patch(Rectangle((j,i),1,1,ec='.5',fc='white',lw=.5));c.text(j+.5,i+.5,str((j-i)%4),ha='center',va='center',fontsize=9)
 c.text(-.3,i+.5,str(i+1),ha='right',va='center');c.text(i+.5,-.3,str(i+5),ha='center',va='center')
c.text(2,4.5,'Entry = inner round r (T = 4)',ha='center',fontsize=8)
c.text(2,5.1,'Same r: one cell per row / column',ha='center',fontsize=7.5)
f.text(.5,.055,'Load coordinates once  →  inner matching  →  barrier  →  next matching  →  store once',ha='center',fontsize=8)
f.subplots_adjust(top=.87,bottom=.16);save(f,'fig02_rr_full')
# 3 Sparse algorithmic intended design with measured caveat.
f,axs=plt.subplots(1,3,figsize=(7.2,3.25),gridspec_kw={'width_ratios':[.8,1.3,1.2]});a,b,c=axs
for ax in axs:plain(ax)
a.set_title('(a) Partition terms');box(a,.4,3.4,9.2,1.2,'Non-edge\nnon-pivot / pivot',fs=8);box(a,.4,1.1,9.2,1.2,'Edges +\npivot pairs',fs=8)
b.set_title('(b) One-sided phase');txt(b,5,5.5,'Fixed pivot coordinates',8)
for x,p in [(2,'p1'),(8,'p2')]:box(b,x-.6,4.2,1.2,.75,p,fc='.88')
for x,l in [(1.5,'u'),(5,'v'),(8.5,'w')]:
 b.scatter([x],[2.1],s=300,fc='white',ec='black',zorder=3);txt(b,x,2.1,l,9,zorder=4)
 for px in [2,8]:arr(b,px,4.1,x,2.6,':')
txt(b,5,.9,'One thread owns each non-pivot',8);txt(b,5,.2,'Load → local updates → store',8)
c.set_title('(c) Two-sided phase')
box(c,.15,3.9,9.7,1,'Round 1: (p1,p2), (u,v)',fs=8);box(c,.15,2.3,9.7,1,'Round 2: (u,p1), (v,w)',fs=8);arr(c,5,3.85,5,3.35)
txt(c,5,1.4,'Disjoint endpoints in each round',8);txt(c,5,.7,'Complete a phase before the next',8)
f.text(.5,.025,'Design shown: pivots are read-only in (b). The measured v3 code also rewrites unchanged pivot coordinates.',ha='center',fontsize=6.7)
f.subplots_adjust(wspace=.27,top=.86,bottom=.15);save(f,'fig03_rr_sparse')
# Read small metadata only.
def rows(name):return list(csv.DictReader((DATA/'tables'/name).open()))
summary=rows('summary.csv'); pairs=rows('paired-ratios.csv'); runs=rows('runs.csv')
# 4 RR-only tile-size sensitivity: unnormalized final stress.
graphs=['USpowerGrid','dwt_2680','bcsstk16','fe_4elt2','EVA','wiki-Vote']
f,axs=plt.subplots(2,3,figsize=(7.2,4.25),sharex=True)
tile_rows=[]
for ax,g in zip(axs.flat,graphs):
 rr=[r for r in summary if r['graph']==g and r['method']=='rr_sgd' and r['study']=='tiling' and int(r['success'])==5]
 rr.sort(key=lambda r:int(r['tile_size']));T=[int(r['tile_size']) for r in rr]
 med=np.array([float(r['stress_value_median']) for r in rr]);lo=np.array([float(r['stress_value_min']) for r in rr]);hi=np.array([float(r['stress_value_max']) for r in rr])
 ax.errorbar(T,med,yerr=[med-lo,hi-med],fmt='-o',c='black',ms=3,elinewidth=.6,capsize=1.5)
 ax.set_xscale('log',base=2);ax.set_ylim(0,max(hi)*1.12);ax.set_title(g+(' (estimated)' if g=='fe_4elt2' else ''),fontsize=9)
 ax.set_ylabel('Final stress',fontsize=8);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
 ax.tick_params(labelsize=7);ax.yaxis.get_offset_text().set_fontsize(7)
 ax.set_xticks([1,4,16,64,256,1024],['1','4','16','64','256','1024']);ax.grid(axis='y',color='.9',lw=.6)
 for v in rr:tile_rows.append({'graph':g,'tile_size':int(v['tile_size']),'seeds':5,'stress_kind':'sampled_256_sources' if g=='fe_4elt2' else 'exact','stress_median':float(v['stress_value_median']),'stress_min':float(v['stress_value_min']),'stress_max':float(v['stress_value_max']),'mean_iteration_ms_median':float(v['iteration_time_ms_median'])/15})
for ax in axs[1]:ax.set_xlabel('Tile size T')
f.subplots_adjust(top=.91,bottom=.12,left=.09,right=.98,hspace=.57,wspace=.53);save(f,'fig04_tile_size')
with (ROOT/'fig04_tile_values.csv').open('w') as z:
 writer=csv.DictWriter(z,fieldnames=list(tile_rows[0]));writer.writeheader();writer.writerows(tile_rows)
# 5 mean time per iteration, paired successful seeds; graph names on x-axis.
comparison_rows=[]
large={r['graph'] for r in runs if r['nodes'] and int(r['nodes'])>=100000}
graph_sets={}
for fam in ['full','sparse']:
 cpu,rr=('sgd','rr_sgd') if fam=='full' else ('sparse_sgd','rr_sparse_sgd')
 pp=[r for r in pairs if r['study']=='basic' and r['family']==fam and r['method']==rr and (fam=='full' or r['graph'] in large)]
 gg=sorted({r['graph'] for r in pp});graph_sets[fam]=gg
 for g in gg:
  seeds={r['seed'] for r in pp if r['graph']==g}
  for m in [cpu,rr]:
   selected=[r for r in runs if r['study']=='basic' and r['status']=='success' and r['graph']==g and r['method']==m and r['seed'] in seeds]
   assert len(selected)==len(seeds),(g,m,len(selected),len(seeds))
   assert all(int(r['iterations'])==15 for r in selected)
   # The misleadingly named source field is a SUM across all iterations.
   vv=[float(r['iteration_time_ms'])/int(r['iterations']) for r in selected]
   comparison_rows.append({'family':fam,'graph':g,'method':'CPU' if m==cpu else 'RR','paired_seeds':len(seeds),'iterations_per_run':15,'median_ms_per_iteration':np.median(vv),'min_ms_per_iteration':min(vv),'max_ms_per_iteration':max(vv)})
f,axs=plt.subplots(3,1,figsize=(7.2,8.7))
panels=[('full',graph_sets['full'],'(a) Full: 19 graphs'),('sparse',graph_sets['sparse'][:19],'(b) Sparse: large graphs 1–19'),('sparse',graph_sets['sparse'][19:],'(c) Sparse: large graphs 20–37')]
for ax,(fam,gg,title) in zip(axs,panels):
 for x,g in enumerate(gg):
  paired_medians=[next(r['median_ms_per_iteration'] for r in comparison_rows if r['family']==fam and r['graph']==g and r['method']==m) for m in ['CPU','RR']]
  ax.plot([x,x],paired_medians,color='.75',lw=.6,zorder=1)
  for m,mark,col,face in [('CPU','s','.4','white'),('RR','o','black','black')]:
   r=next(r for r in comparison_rows if r['family']==fam and r['graph']==g and r['method']==m)
   med=r['median_ms_per_iteration'];lo=r['min_ms_per_iteration'];hi=r['max_ms_per_iteration']
   ax.errorbar(x,med,yerr=[[med-lo],[hi-med]],fmt=mark,c=col,mfc=face,ms=4,elinewidth=.7,capsize=1.8,zorder=3)
 ax.set_xticks(range(len(gg)),gg,rotation=55,ha='right',fontsize=7)
 ax.set_xlim(-.65,len(gg)-.35);ax.set_yscale('log')
 ticks=[1,10,100,1000] if fam=='full' else [10,100,1000]
 ax.set_ylim((1,2000) if fam=='full' else (10,4000));ax.set_yticks(ticks,[f'{t:,}' for t in ticks])
 ax.tick_params(axis='y',labelsize=8);ax.tick_params(axis='x',length=2,pad=3)
 ax.grid(axis='y',which='major',color='.88',lw=.6);ax.set_axisbelow(True)
 ax.set_ylabel('Mean time per iteration (ms)',fontsize=8)
 ax.set_title(title,loc='left',fontsize=9,pad=6)
f.legend([plt.Line2D([],[],ls='',marker='s',c='.4',mfc='white'),plt.Line2D([],[],ls='',marker='o',c='black')],['CPU','RR'],ncol=2,frameon=False,loc='upper center',fontsize=9)
f.subplots_adjust(top=.94,bottom=.14,hspace=1.07,left=.105,right=.99)
save(f,'fig05_comparison')
with (ROOT/'fig05_iteration_times.csv').open('w') as z:
 writer=csv.DictWriter(z,fieldnames=list(comparison_rows[0]));writer.writeheader();writer.writerows(comparison_rows)
# 6 checkpoint convergence: all seed medians per method, CPU final median normalization.
conv=rows('convergence.csv');f,axs=plt.subplots(2,3,figsize=(7.2,4.25))
for ax,(g,fam) in zip(axs.flat,[('dwt_2680','full'),('wiki-Vote','full'),('EVA','full'),('luxembourg_osm','sparse'),('kron_g500-logn18','sparse'),('coAuthorsCiteseer','sparse')]):
 methods=['sgd','atomic_sgd','rr_sgd'] if fam=='full' else ['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd']
 p=[r for r in conv if r['graph']==g and r['family']==fam]
 ref=np.median([float(r['stress']) for r in p if r['method']==methods[0] and r['epoch']=='15'])
 for m,ls,col,mark in zip(methods,[':','--','-'],['.4','.55','black'],['^','s','o']):
  ys=[];xs=[]
  for ep in [0,1,2,3,5,8,10,15]:
   rr=[r for r in p if r['method']==m and int(r['epoch'])==ep]
   if rr: xs.append(ep);ys.append(np.median([float(r['stress']) for r in rr])/ref)
  ax.plot(xs,ys,ls=ls,c=col,marker=mark,ms=3,lw=1)
 ax.axhline(1,c='.65',lw=.5);ax.set_yscale('log');ax.set_title(f'{g} ({fam})',fontsize=8.5);ax.set_xticks([0,5,10,15]);ax.tick_params(labelsize=7)
for ax in axs[:,0]:ax.set_ylabel('Stress / CPU final median',fontsize=8)
for ax in axs[1]:ax.set_xlabel('Iteration')
f.legend([plt.Line2D([],[],c=c,ls=l,marker=m) for c,l,m in zip(['.4','.55','black'],[':','--','-'],['^','s','o'])],['CPU','Atomic','RR'],ncol=3,loc='upper center',frameon=False,fontsize=8)
f.subplots_adjust(top=.85,bottom=.12,left=.1,right=.99,hspace=.48,wspace=.27);save(f,'fig06_convergence')
# 7 all available successful RR layouts without CPU reference, n19 node-only avoids edge sampling distortion.
selected=[r for r in runs if r['status']=='success' and r['study']=='basic' and ((r['graph']=='delaunay_n14' and r['method']=='rr_sgd') or (r['graph']=='delaunay_n19' and r['method']=='rr_sparse_sgd'))]
f,axs=plt.subplots(2,4,figsize=(7.2,4.5))
for row,g in enumerate(['delaunay_n14','delaunay_n19']):
 rs=sorted([r for r in selected if r['graph']==g],key=lambda r:int(r['seed'])); arrays=[]
 for r in rs:
  with gzip.open(r['final_positions_path'],'rt') as z:
   head=z.readline()
   # native files begin with vertex count followed by x y
   z.seek(0);lines=z.readlines()
  coords=[]
  in_positions=False
  for line in lines:
   if line.startswith('# Positions'): in_positions=True;continue
   if not in_positions:continue
   v=line.split()
   if len(v)==2:
    try:coords.append([float(v[0]),float(v[1])])
    except ValueError:pass
  xy=np.asarray(coords);assert len(xy)==int(r['nodes']),(r['graph'],head,len(xy));arrays.append(xy)
 bounds=np.vstack([np.array([x.min(axis=0),x.max(axis=0)]) for x in arrays]);lo=bounds.min(axis=0);hi=bounds.max(axis=0)
 for col,ax in enumerate(axs[row]):
  ax.axis('off')
  if col>=len(rs):continue
  xy=arrays[col];r=rs[col];ax.scatter(xy[:,0],xy[:,1],s=.13 if row==0 else .025,c='black',linewidths=0,rasterized=True)
  ax.set_aspect('equal');ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_title(f"seed {r['seed']} | {float(r['iteration_time_ms'])/1000:.3f} s",fontsize=8)
 axs[row,0].text(-.08,.5,g,rotation=90,ha='center',va='center',transform=axs[row,0].transAxes,fontsize=9)
f.text(.52,.96,'Full RR: 4/5 completed (16,384 vertices)',ha='center',fontsize=10)
f.text(.52,.48,'Sparse RR (v3): 2/5 completed (524,288 vertices)',ha='center',fontsize=10)
f.text(.5,.01,'All completed layouts are shown. Vertices only; common coordinate scale within each graph.',ha='center',fontsize=8)
f.subplots_adjust(top=.87,bottom=.06,hspace=.55,wspace=.15,left=.08,right=.99);save(f,'fig07_large_layouts')
print('Generated 7 figures in PDF / SVG / PNG')
