#!/usr/bin/env python3
"""Paper-style complete-time comparison: historical CPU, v5.5 Atomic and RR."""
from pathlib import Path
import csv,hashlib,json,math,shutil,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
V55=ROOT/'experiment_results/v5.5'
BASE=V55/'atomic-full-main-20261110'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(name,rows):
    with (BASE/'tables'/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    cpu_path=V55/'ex2-copapers-capacity-supplement-20261110/tables/historical-controls-view.json'
    atomic_paths=list((BASE/'raw').glob('*/record.json'))
    rr_paths=list((V55/'ex2-rr128-speed-stress-20261110/raw').glob('*/record.json'))
    rr=[(p,json.loads(p.read_text())) for p in rr_paths]
    rr=[(p,r) for p,r in rr if r['method']=='rr_sgd']
    protected={str(p.relative_to(ROOT)):sha(p) for p in [cpu_path,*atomic_paths,*[p for p,r in rr]]}
    methods={'sgd':[r for r in json.loads(cpu_path.read_text()) if r['method']=='sgd'],
             'atomic_sgd':[json.loads(p.read_text()) for p in atomic_paths],
             'rr_sgd':[r for p,r in rr]}
    assert all(len(rows)==570 and all(r['status']=='success' for r in rows) for rows in methods.values())
    datasets={g['name']:g for g in json.loads((BASE/'config/datasets.json').read_text())}
    names=sorted(datasets,key=lambda g:(datasets[g]['edges'],g))
    assert len(names)==38 and names==json.loads((BASE/'tables/paper-style-graph-order.json').read_text())
    index={m:{(r['graph'],r['seed'],r['repetition']):r for r in rows} for m,rows in methods.items()}
    assert set(index['sgd'])==set(index['atomic_sgd'])==set(index['rr_sgd'])
    initial_equal=0
    for k,cpu in index['sgd'].items():
        rows=[index[m][k] for m in methods]
        for r in rows:
            assert all(r[f]==cpu[f] for f in ['input_sha256','nodes','seed','iterations','epsilon'])
            assert r['iterations']==15 and r['epsilon']==.1
        initial_equal+=all(r['initial_positions_sha256']==cpu['initial_positions_sha256'] for r in rows)
    points=[];seeds=[];ratios=[]
    for method in methods:
        for graph in names:
            values=[]
            for seed in range(5):
                group=[index[method][graph,seed,rep] for rep in range(3)]
                val=statistics.median(r['total_compute_time_ms'] for r in group)
                values.append(val);seeds.append(dict(graph=graph,method=method,seed=seed,total_compute_time_ms=val))
                if method!='sgd':
                    speedup=statistics.median(index['sgd'][graph,seed,rep]['total_compute_time_ms']/index[method][graph,seed,rep]['total_compute_time_ms'] for rep in range(3))
                    ratios.append(dict(graph=graph,method=method,seed=seed,cpu_over_method=speedup))
            points.append(dict(graph=graph,nodes=datasets[graph]['nodes'],undirected_edges=datasets[graph]['edges'],method=method,median_ms=statistics.median(values),seed_min_ms=min(values),seed_max_ms=max(values)))
    write_csv('full-complete-method-points.csv',points);write_csv('full-complete-method-seeds.csv',seeds);write_csv('full-complete-paired-speedups.csv',ratios)
    lookup={(p['method'],p['graph']):p for p in points}
    summaries=[]
    for graph in names:
        s=dict(graph=graph,nodes=datasets[graph]['nodes'],edges=datasets[graph]['edges'])
        for method in methods:s[method+'_total_ms']=lookup[method,graph]['median_ms']
        for method in ['atomic_sgd','rr_sgd']:s[method+'_cpu_speedup']=statistics.median(r['cpu_over_method'] for r in ratios if r['method']==method and r['graph']==graph)
        summaries.append(s)
    write_csv('full-complete-method-summary.csv',summaries)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7.1,2.7),layout='constrained')
    styles=[('sgd','CPU (v5 reference)','x','#343434'),('atomic_sgd','Atomic (v5.5)','^','#bb520e'),('rr_sgd','RR (v5.5, T128)','o','#126c92')]
    for method,label,marker,color in styles:
        vals=[lookup[method,g] for g in names]
        mid=np.array([v['median_ms'] for v in vals]);lo=np.array([v['seed_min_ms'] for v in vals]);hi=np.array([v['seed_max_ms'] for v in vals])
        assert (lo>.05).all() and (hi<50000).all()
        ax.errorbar(np.arange(38),mid,yerr=[mid-lo,hi-mid],fmt=marker,linestyle='none',color=color,markersize=4 if marker=='x' else 3.2,markerfacecolor='none',markeredgewidth=.85,elinewidth=.7,capsize=1.1,label=label)
    ax.set_title('Full SGD: total compute (38 graphs)',loc='left',fontsize=9.5,pad=23)
    ax.legend(loc='lower right',bbox_to_anchor=(1,1.005),ncol=3,frameon=False,fontsize=7.1,handlelength=1.1,columnspacing=.9)
    ax.set_xticks(np.arange(38),names,rotation=90,fontsize=7);ax.tick_params(axis='x',length=2,pad=2);ax.set_xlim(-.6,37.6)
    ax.set_yscale('log');ax.set_ylim(.05,50000);ax.set_yticks(10.**np.arange(-1,5));ax.tick_params(axis='y',labelsize=7.5)
    ax.set_ylabel('Total compute time (ms)',fontsize=8.5);ax.grid(axis='y',which='major',color='#ddd',linewidth=.6);ax.set_axisbelow(True)
    for ext in ['png','pdf','svg']:
        p=BASE/'figures'/f'paper-style-v55-full-complete-methods.{ext}';fig.savefig(p,dpi=240,bbox_inches='tight',pad_inches=.035)
        if ext=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig)
    aggregate={method:math.exp(statistics.mean(math.log(s[method+'_cpu_speedup']) for s in summaries)) for method in ['atomic_sgd','rr_sgd']}
    metadata=dict(graphs=38,records_per_method=570,total_records=1710,new_solver_runs=0,initial_coordinates_equal_triplets=initial_equal,primary_metric='total_compute_time_ms; input to CPU result readiness; excludes Stress evaluation, saving, drawing',cpu='Historical v5 CPU Full records reused by v5.5; no CPU rerun',atomic='v5.5 merged-main745b38ff9 Atomic Full new570 records; timestamps off',rr='v5.5 EX2 RR Full new570 records; T=W128/shared; timestamps on',caveat='Measured in different sessions; RR timestamps enabled while Atomic disabled; not a simultaneous or instrumentation-identical experiment',aggregation='3-run median per seed, median/range of 5 seed medians; CPU speedup ratios paired by graph/seed/repetition',source_sha256=protected,cpu_speedup_graph_equal_geomean=aggregate)
    (BASE/'validation/full-complete-methods.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    text='# v5.5 Full：CPU・Atomic・RRのComplete時間比較\n\n'
    text+='![Complete比較](../figures/paper-style-v55-full-complete-methods.png)\n\n論文図6と同じ38グラフ・辺数昇順・対数縦軸・マーカーと5seed範囲。黒×CPU、橙△Atomic、青○RR。値は各seed内3反復中央値→5seed中央値、whiskerは5seedの最小〜最大。\n\nCompleteはtotal_compute_time_ms。入力からCPU座標が揃うまでで、前処理・GPU準備・GPU完了待ち・読み戻しを含み、Stress評価・保存・描画は含まない。描画直前までの全処理時間ではない。\n\nCPUはv5.5で参照している過去v5の570記録、Atomicは今回のマージ済みmainの新570記録、RRはv5.5 EX2の新570記録（T=W128/shared）。全方式で同一入力・15反復・epsilon0.1・seed0〜4×3回。RRはGPU timestamp有効、Atomicは無効。測定時期と計測設定が異なるため、同時・同一instrumentationで取り直した比較とは区別する。\n\n'
    text+=f"対応seed比を用いた38グラフ等重み幾何平均のCPU速度比：Atomic {aggregate['atomic_sgd']:.3f}倍、RR {aggregate['rr_sgd']:.3f}倍。初期座標一致：{initial_equal}/570組。\n\n"
    text+='| graph | CPU ms | Atomic ms | RR ms |\n|---|---:|---:|---:|\n'
    for s in summaries:text+=f"| {s['graph']} | {s['sgd_total_ms']:.2f} | {s['atomic_sgd_total_ms']:.2f} | {s['rr_sgd_total_ms']:.2f} |\n"
    text+='\n[PDF](../figures/paper-style-v55-full-complete-methods.pdf) / [SVG](../figures/paper-style-v55-full-complete-methods.svg) / [数値CSV](../tables/full-complete-method-summary.csv)\n'
    (BASE/'reports/full-complete-methods.md').write_text(text)
    shutil.copy2(__file__,BASE/'source-snapshot/experiments'/Path(__file__).name)
    for rel,h in protected.items():assert sha(ROOT/rel)==h
    print(json.dumps(dict(cpu_speedup_geomean=aggregate,triplets_initial_equal=initial_equal,records=1710),ensure_ascii=False))

if __name__=='__main__':main()
