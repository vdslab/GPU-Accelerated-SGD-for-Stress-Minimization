#!/usr/bin/env python3
"""Compare Atomic Full versions using the paper Figure 6 visual style."""
from pathlib import Path
import csv,hashlib,json,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiment_results/v5.5/atomic-full-main-20261110'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    paths=[BASE/'tables/summary.json',BASE/'config/datasets.json',BASE/'config/environment.json']
    protected={str(p.relative_to(ROOT)):sha(p) for p in paths}
    summary={s['graph']:s for s in json.loads(paths[0].read_text())}
    graphs={g['name']:g for g in json.loads(paths[1].read_text())}
    names=sorted(summary,key=lambda g:(graphs[g]['edges'],g))
    assert len(names)==38 and set(names)==set(graphs)
    reference=ROOT.parent/'sgd-v55-main/experiment_results/v5.5/final-paper-charts-20261110/tables/plotted-graphs.json'
    if reference.exists():assert names==json.loads(reference.read_text())['full']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    styles=[('v5','Atomic Full v5','^','#bb520e'),('v55','Atomic Full v5.5','o','#126c92')]
    point_rows=[]
    specs=[('sgd_time_ms',1,'Iteration total time (ms)',(1,10000),'(a) Atomic Full: iteration total (15 iterations)'),
           ('total_compute_time_ms',1,'Total compute time (ms)',(10,20000),'(b) Atomic Full: total compute')]
    def panel(ax,metric,scale,ylabel,limits,title):
        x=np.arange(len(names))
        for version,label,marker,color in styles:
            center=np.array([summary[g][version+'_'+metric] for g in names])*scale
            low=np.array([summary[g][version+'_'+metric+'_seed_min'] for g in names])*scale
            high=np.array([summary[g][version+'_'+metric+'_seed_max'] for g in names])*scale
            assert (low>0).all() and (low>=limits[0]).all() and (high<=limits[1]).all()
            ax.errorbar(x,center,yerr=[center-low,high-center],fmt=marker,linestyle='none',
                color=color,markersize=3.2,markerfacecolor='none',markeredgewidth=.85,
                elinewidth=.7,capsize=1.1,label=label)
            for g,mn,md,mx in zip(names,low,center,high):
                point_rows.append(dict(graph=g,nodes=graphs[g]['nodes'],undirected_edges=graphs[g]['edges'],version=version,metric=metric,scale=scale,median=float(md),seed_min=float(mn),seed_max=float(mx)))
        ax.set_title(title,loc='left',fontsize=9.5,pad=23)
        ax.legend(loc='lower right',bbox_to_anchor=(1,1.005),ncol=2,frameon=False,fontsize=7.3,handlelength=1.1,columnspacing=1.1)
        ax.set_xticks(x,names,rotation=90,fontsize=7)
        ax.tick_params(axis='x',length=2,pad=2)
        ax.set_xlim(-.6,len(names)-.4);ax.set_yscale('log');ax.set_ylim(*limits)
        ax.set_yticks(10.**np.arange(np.ceil(np.log10(limits[0])),np.floor(np.log10(limits[1]))+1))
        ax.tick_params(axis='y',labelsize=7.5);ax.set_ylabel(ylabel,fontsize=8.5)
        ax.grid(axis='y',which='major',color='#ddd',linewidth=.6);ax.set_axisbelow(True)
    def export(fig,name):
        for ext in ['png','pdf','svg']:
            p=BASE/'figures'/f'{name}.{ext}'
            fig.savefig(p,dpi=240,bbox_inches='tight',pad_inches=.035)
            if ext=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
        plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(7.1,4.95),layout='constrained')
    for ax,spec in zip(axes,specs):panel(ax,*spec)
    export(fig,'paper-style-v5-v55-speed')
    fig,ax=plt.subplots(figsize=(7.1,2.7),layout='constrained')
    panel(ax,*specs[1][:-1],'Atomic Full: total compute (38 graphs)')
    export(fig,'paper-style-v5-v55-total')
    fig,ax=plt.subplots(figsize=(7.1,2.7),layout='constrained')
    panel(ax,'stress_value',1,'Final Full Stress',(10,20000000),'Atomic Full: final Stress (38 graphs)')
    export(fig,'paper-style-v5-v55-stress')
    # Single-panel exports duplicate their plotted data; retain one row per point.
    unique={(r['graph'],r['version'],r['metric']):r for r in point_rows}
    rows=sorted(unique.values(),key=lambda r:(names.index(r['graph']),r['metric'],r['version']))
    with (BASE/'tables/paper-style-chart-points.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (BASE/'tables/paper-style-graph-order.json').write_text(json.dumps(names,indent=2)+'\n')
    metadata=dict(reference='Paper Figure 6: fig06-iteration-comparison',scope='Atomic Full v5 versus newly measured v5.5 only',graphs=38,formal_runs_per_version=570,new_solver_runs=0,
        x_axis='Increasing undirected edge count, then graph name; identical to paper Full panel',
        style='7.1-inch width; marker-only; open triangles/orange and circles/blue; log y axis; 90-degree graph labels; seed min/max whiskers',
        aggregation='median of 3 repetitions within each seed, then median and range of 5 seed medians',
        iteration_metric='sgd_time_ms; total of the 15-iteration host interval; no readback added and no division by 15. New main excludes GPU completion wait from this field; not pure GPU time.',
        total_metric='total_compute_time_ms; includes CPU pair hashing/shuffling and GPU completion; excludes Stress evaluation, saving, drawing',
        stress_metric='Exact at <=8000 vertices; otherwise sampled256/evaluation seed0, identical to v5',
        input_sha256=protected)
    (BASE/'validation/paper-style-charts.json').write_text(json.dumps(metadata,indent=2)+'\n')
    text='''# 論文形式：v5とv5.5のAtomic Full比較

論文図6のFullパネルと同じ38グラフを、同じ辺数昇順・グラフ名順で表示する。対数縦軸、線で結ばないマーカー、5 seed中央値の最小〜最大、配色・寸法・縦向きラベルを揃えた。橙△はv5、青○はv5.5。各点はseed内3反復中央値→5seed中央値。

## 速度

![速度比較](../figures/paper-style-v5-v55-speed.png)

上段は `sgd_time_ms`、15反復のiteration合計時間。読み戻しを足さず、15でも割らない。新mainは非同期submit後のGPU完了待ちがreadback側にあるため、上段の値はその待機を含まず、GPUの全計算時間や両版の同一範囲の実時間とは解釈しない。下段のtotal_compute_time_msは、CPUの全pairハッシュ・shuffleとGPU完了待ちを含む入力からCPU上の座標が揃うまでの時間。Stress評価・座標保存・描画は除外されるので、描画の直前までの全処理時間ではない。

[速度PDF](../figures/paper-style-v5-v55-speed.pdf) / [速度SVG](../figures/paper-style-v5-v55-speed.svg) / [総計算のみPDF](../figures/paper-style-v5-v55-total.pdf)

## Stress

![Stress比較](../figures/paper-style-v5-v55-stress.png)

8,000頂点以下は厳密Full Stress、それ以外は固定256始点・評価seed0。全38グラフを表示し、品質が悪化した対象や速度が遅い対象も省略しない。v5は過去測定との比較で、whiskerは信頼区間ではない。小差から統計的有意差を判断しない。

[Stress PDF](../figures/paper-style-v5-v55-stress.pdf) / [全グラフの集計と考察](results.md) / [描画数値CSV](../tables/paper-style-chart-points.csv)
'''
    (BASE/'reports/paper-style-comparison.md').write_text(text)
    shutil.copy2(__file__,BASE/'source-snapshot/experiments'/Path(__file__).name)
    for name,h in protected.items():assert sha(ROOT/name)==h
    print('Created paper-style charts: speed, total compute, Stress; no measurements changed.')

if __name__=='__main__':main()
