"""Offline report for added grid and Sparse workgroup experiments."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics as stats
import shutil

import psutil

from prepare_v5_grid_extension import BASE
from run_v5 import ROOT, VAULT, now, save


def read(path):return json.loads(path.read_text())
def dist(values):return dict(median=stats.median(values),min=min(values),max=max(values)) if values else dict(median=None,min=None,max=None)
def csvwrite(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    jobs=read(BASE/'config/jobs.json');records=[read(p) for p in (BASE/'raw').glob('*/record.json') if read(p).get('archive_status')=='success']
    if not args.allow_partial:assert len(records)==810,'Do not finalize before all formal outcomes exist'
    for p in psutil.process_iter(['cmdline']):
        argv=p.info['cmdline'] or []
        assert not (argv and (argv[0].endswith('/release/rr-gpu') or '/additional-grid-workgroup/bin/' in argv[0])),'Never draw during native measurement'
    graphs={g['name']:g for g in read(BASE/'config/datasets.json')};summary=[];grouped=collections.defaultdict(list);planned=collections.Counter()
    for j in jobs:planned[(j['stage'],j['graph'],j['tile_size'],j['workgroup_size'],j['pivots'])]+=1
    for r in records:grouped[(r['stage'],r['graph'],r['tile_size'],r['workgroup_size'],r['pivots'])].append(r)
    metric_fields=['sgd_mean_iteration_ms','gpu_execution_ms','gpu_execution_with_upload_ms','total_compute_time_ms','stress_value','peak_rss_bytes']
    for key,count in planned.items():
        stage,graph,tile,width,pivots=key;rows=grouped[key];good=[r for r in rows if r['status']=='success'];complete=[s for s in range(5) if sum(r['seed']==s for r in good)==3]
        item=dict(stage=stage,graph=graph,nodes=graphs[graph]['nodes'],tile_size=tile,workgroup_size=width,pivots=pivots,planned=count,success=len(good),failed=len(rows)-len(good),pending=count-len(rows),complete_seeds=len(complete))
        for field in metric_fields:
            values=[stats.median(r[field] for r in good if r['seed']==s) for s in complete]
            for suffix,value in dist(values).items():item[field+'_'+suffix]=value
        summary.append(item)
    save(BASE/'tables/summary.json',summary);csvwrite(BASE/'tables/summary.csv',summary);csvwrite(BASE/'tables/runs.csv',records)
    index={(r['stage'],r['graph'],r['tile_size'],r['workgroup_size'],r['pivots'],r['seed'],r['repetition']):r for r in records if r['status']=='success'}
    paired=[];fingerprints=[]
    for r in records:
        if r['status']!='success' or r['stage']!='sparse' or r['workgroup_size']!=1024:continue
        ref=index.get(('sparse',r['graph'],None,256,r['pivots'],r['seed'],r['repetition']))
        if ref is None:continue
        assert r['initial_positions_sha256']==ref['initial_positions_sha256'] and r['preprocess_sha256']==ref['preprocess_sha256']
        paired.append(dict(graph=r['graph'],pivots=r['pivots'],seed=r['seed'],repetition=r['repetition'],sgd_speedup_w256_over_w1024=ref['sgd_time_ms']/r['sgd_time_ms'],execution_speedup_w256_over_w1024=ref['gpu_execution_ms']/r['gpu_execution_ms'],stress_ratio_w1024_over_w256=r['stress_value']/ref['stress_value'],same_stress=r['stress_value']==ref['stress_value']))
    csvwrite(BASE/'tables/sparse-paired-ratios.csv',paired)
    for graph in read(BASE/'config/plan.json')['full_graphs']:
        for seed in range(5):
            rs=[r for r in records if r['status']=='success' and r['stage']=='full' and r['graph']==graph and r['seed']==seed]
            if rs:assert len({r['initial_positions_sha256'] for r in rs})==1
    for graph in read(BASE/'config/plan.json')['sparse_graphs']:
        for seed in range(5):
            rs=[r for r in records if r['status']=='success' and r['stage']=='sparse' and r['graph']==graph and r['seed']==seed]
            fingerprints.append(dict(graph=graph,seed=seed,distinct_p_initials=len({r['initial_positions_sha256'] for r in rs})))
    save(BASE/'tables/fingerprints.json',fingerprints)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    outputs=[]
    def export(fig,name):
        fig.savefig(BASE/'figures'/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(BASE/'figures'/(name+'.pdf'),bbox_inches='tight');plt.close(fig);outputs.append(name)
    for field,ylabel,name in [('sgd_mean_iteration_ms','Mean SGD elapsed per iteration (ms)','full-grid-iteration'),('stress_value','Final Full Stress','full-grid-stress')]:
        fig,axes=plt.subplots(2,3,figsize=(12,7))
        for ax,graph in zip(axes.ravel(),read(BASE/'config/plan.json')['full_graphs']):
            rs=sorted([r for r in summary if r['stage']=='full' and r['graph']==graph and r['complete_seeds']==5],key=lambda r:r['tile_size'])
            if rs:
                ax.errorbar([r['tile_size'] for r in rs],[r[field+'_median'] for r in rs],yerr=[[r[field+'_median']-r[field+'_min'] for r in rs],[r[field+'_max']-r[field+'_median'] for r in rs]],fmt='o-',capsize=3,color='#0072b2')
                for r in rs:
                    if r['tile_size']==1024:ax.scatter([1024],[r[field+'_median']],color='#d55e00',zorder=4)
            else:ax.text(.5,.5,'No complete 5-seed result',transform=ax.transAxes,ha='center',fontsize=8)
            ax.set_xscale('log',base=2);ax.set_xticks([64,128,256,512,1024],[64,128,256,512,1024]);ax.set_title(f'{graph}\nn={graphs[graph]["nodes"]:,}',fontsize=9);ax.set_xlabel('Tile size T');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
        fig.suptitle('RoundRobin Full SGD: median and min-max over five seeds'+(' (partial)' if len(records)<810 else ''));fig.tight_layout(rect=(0,0,1,.95));export(fig,name)
    for pivots in [200,1000]:
        fig,axes=plt.subplots(2,3,figsize=(12,7))
        for ax,graph in zip(axes.ravel(),read(BASE/'config/plan.json')['sparse_graphs']):
            rs=sorted([r for r in summary if r['stage']=='sparse' and r['graph']==graph and r['pivots']==pivots and r['complete_seeds']==5],key=lambda r:r['workgroup_size'])
            if rs:ax.errorbar([r['workgroup_size'] for r in rs],[r['sgd_mean_iteration_ms_median'] for r in rs],yerr=[[r['sgd_mean_iteration_ms_median']-r['sgd_mean_iteration_ms_min'] for r in rs],[r['sgd_mean_iteration_ms_max']-r['sgd_mean_iteration_ms_median'] for r in rs]],fmt='o-',capsize=3,color='#0072b2')
            else:ax.text(.5,.5,'No complete 5-seed result',transform=ax.transAxes,ha='center',fontsize=8)
            ax.set_xticks([256,1024]);ax.set_xlim(128,1152);ax.set_xlabel('Threads per workgroup W');ax.set_ylabel('Mean SGD elapsed per iteration (ms)');ax.set_title(graph,fontsize=9);ax.grid(alpha=.2)
        fig.suptitle(f'RoundRobin Sparse SGD, p={pivots}: median and min-max over five seeds');fig.tight_layout(rect=(0,0,1,.95));export(fig,f'sparse-workgroup-p{pivots}')
    successful=sum(r['status']=='success' for r in records);text=f'# 追加実験：gridとSparse workgroup\n\n更新: {now()}。成功{successful}、失敗{len(records)-successful}、未測定{810-len(records)}。\n\n'
    winners=[]
    for graph in read(BASE/'config/plan.json')['full_graphs']:
        rs=[r for r in summary if r['stage']=='full' and r['graph']==graph and r['complete_seeds']==5]
        if len(rs)==5:
            best=min(rs,key=lambda r:r['sgd_mean_iteration_ms_median']);winners.append(best['tile_size']);text+=f'- {graph}: 最速T={best["tile_size"]}、平均1反復{best["sgd_mean_iteration_ms_median"]:.3f}ms。\n'
        else:text+=f'- {graph}: 全Tの5seedが揃っていないため最速Tの結論は保留。\n'
    text+=f'\n全Tが揃った{len(winners)}グラフ中、T1024が最速だったのは{winners.count(1024)}件。選定した全グラフを表示し、勝った条件だけで一般的優位性を主張しない。\n\n'
    for graph in read(BASE/'config/plan.json')['sparse_graphs']:
        for p in [200,1000]:
            rs=[r for r in paired if r['graph']==graph and r['pivots']==p];complete=[s for s in range(5) if sum(r['seed']==s for r in rs)==3]
            if len(complete)==5:
                ratio=stats.median(stats.median(r['sgd_speedup_w256_over_w1024'] for r in rs if r['seed']==s) for s in complete)
                text+=f'- Sparse {graph}, p{p}: W256/W1024の対応速度比中央値{ratio:.4f}。1より大きいとW1024が高速。\n'
    text+='\n## 解釈と制約\n\n主指標は15反復のSGD経過時間/15。各seed3回中央値→5seed中央値・最小最大。GPU準備・転送・読戻し込みも生表へ保存。純GPU pass時間は未計測。FullのTとSparseのWは別の変数であり、SparseにFullと同じtile実験があるとは扱わない。入力・サイズ・条件は追加測定前に固定した。Fullの全頂点対処理ではgrid形状だけでなくnと並列度が効くため、構造だけの因果効果とは断定しない。SparseはW間の初期配置と制約一致を確認する一方、p間では初期配置も変わる可能性がある。容量停止の未測定条件を0時間や高速成功として埋めない。既存v5の結果とは別の追加実験で、既存バイナリを変更していない。\n\n'
    for name in outputs:text+=f'![{name}](../figures/{name}.png)\n\n'
    (BASE/'reports/results.md').write_text(text)
    folder=VAULT/'v5_図表/additional-grid-workgroup';folder.mkdir(exist_ok=True)
    vault_text=text
    for name in outputs:shutil.copy2(BASE/'figures'/(name+'.png'),folder/(name+'.png'));vault_text=vault_text.replace(f'![{name}](../figures/{name}.png)',f'![[v5_図表/additional-grid-workgroup/{name}.png]]')
    (VAULT/'8_追加実験_gridとSparse設定_考察.md').write_text('---\ntags: [area/大学院研究, 実験考察]\nupdated: '+now()[:10]+'\n---\n\n[[8_追加実験_gridとSparse設定]] / [[論文図表_確認用]]\n\n'+vault_text)
    save(BASE/'validation/report-completion.json',dict(at=now(),success=successful,failed=len(records)-successful,pending=810-len(records),figures=outputs,full_complete_graphs=len(winners),full_winner_counts=dict(collections.Counter(winners)),sparse_paired_runs=len(paired),sparse_equal_stress_pairs=sum(r['same_stress'] for r in paired),visual_review_required=True))
    print(json.dumps(dict(success=successful,pending=810-len(records),figures=outputs),ensure_ascii=False))


if __name__=='__main__':main()
