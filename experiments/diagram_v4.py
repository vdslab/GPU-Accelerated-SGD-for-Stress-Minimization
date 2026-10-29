#!/usr/bin/env python3
"""Regenerate only the Sparse method schematic for the measured v4 schedule."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
BASE=Path(__file__).resolve().parents[1]/'experiment_results/v4'
OUT=BASE/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
def txt(ax,x,y,s,fs=8):ax.text(x,y,s,fontsize=fs,ha='center',va='center')
def box(ax,x,y,w,h,s,fc='white',fs=8):
 ax.add_patch(Rectangle((x,y),w,h,facecolor=fc,edgecolor='black',lw=.8));txt(ax,x+w/2,y+h/2,s,fs)
def arr(ax,x,y,X,Y,style='-'):ax.add_patch(FancyArrowPatch((x,y),(X,Y),arrowstyle='->',mutation_scale=9,lw=.8,linestyle=style,color='black'))
f,axs=plt.subplots(1,3,figsize=(7.2,3.5),gridspec_kw={'width_ratios':[.8,1.3,1.2]});a,b,c=axs
for ax in axs:ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
a.set_title('(a) Partition terms',fontsize=9);box(a,.4,3.4,9.2,1.2,'Non-edge\nnon-pivot / pivot');box(a,.4,1.1,9.2,1.2,'Edges +\npivot pairs')
b.set_title('(b) One-sided chunk',fontsize=9);txt(b,5,5.5,'Read-only pivot coordinates')
for x,p in [(2,'p1'),(8,'p2')]:box(b,x-.6,4.2,1.2,.75,p,fc='.88')
for x,l in [(1.5,'u'),(5,'v'),(8.5,'w')]:
 b.scatter([x],[2.1],s=300,fc='white',ec='black',zorder=3);b.text(x,2.1,l,ha='center',va='center',zorder=4)
 for px in [2,8]:arr(b,px,4.1,x,2.6,':')
txt(b,5,.9,'One thread per non-pivot');txt(b,5,.2,'Vertex-specific pivot permutation')
c.set_title('(c) Two-sided chunk',fontsize=9)
box(c,.15,3.9,9.7,1,'Round 1: (p1,p2), (u,v)');box(c,.15,2.3,9.7,1,'Round 2: (u,p1), (v,w)');arr(c,5,3.85,5,3.35)
txt(c,5,1.4,'Disjoint endpoints in each round');txt(c,5,.7,'Shuffled round order')
f.text(.5,.10,'One iteration: O1 → B1 → O2 → B2 → O3 → B3 → O4 → B4 (or B first)',ha='center',fontsize=8)
f.text(.5,.035,'O: one-sided chunk; B: two-sided chunk. Each term is visited once; updates complete before the next chunk.',ha='center',fontsize=7)
f.subplots_adjust(wspace=.27,top=.87,bottom=.24)
for ext in ['png','pdf','svg']:f.savefig(OUT/f'fig03_rr_sparse.{ext}',dpi=260,bbox_inches='tight',pad_inches=.06)
plt.close(f)
