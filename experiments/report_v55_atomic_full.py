#!/usr/bin/env python3
"""Paired hierarchical v5 versus v5.5 Atomic Full report, all 38 graphs."""
import collections,csv,json,math,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_v55_atomic_full import BASE,OLD,ROOT,integrity
from run_v5 import now,save

METRICS=['total_compute_time_ms','sgd_plus_readback_ms','sgd_time_ms','readback_time_ms','preprocess_time_ms','stress_value']

def csvfile(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (BASE/'tables'/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    integrity(True)
    new=[json.loads(p.read_text()) for p in (BASE/'raw').glob('*/record.json')]
    old=json.loads((BASE/'config/v5-controls.json').read_text())
    assert len(new)==len(old)==570 and all(r['status']=='success' for r in new)
    env=json.loads((BASE/'config/environment.json').read_text())
    key=lambda r:(r['graph'],r['seed'],r['repetition'])
    before={key(r):r for r in old};after={key(r):r for r in new}
    assert len(before)==len(after)==570 and set(before)==set(after)
    metrics_equal=['input_sha256','nodes','seed','iterations','epsilon','stress_kind','stress_samples','stress_seed']
    pairs=[]
    for k,r in after.items():
        o=before[k]
        assert all(r[f]==o[f] for f in metrics_equal),(k,[(f,r[f],o[f]) for f in metrics_equal if r[f]!=o[f]])
        assert r['git_commit']==env['merged_main_commit'] and not r['git_dirty']
        assert r['attempted_updates']==r['completed_updates']==15*r['nodes']*(r['nodes']-1)//2
        assert r['archive_status']=='success'
        for row in [r,o]:row['sgd_plus_readback_ms']=row['sgd_time_ms']+row['readback_time_ms']
        pair=dict(graph=r['graph'],seed=r['seed'],repetition=r['repetition'],initial_equal=r['initial_positions_sha256']==o['initial_positions_sha256'])
        for metric in METRICS:
            assert math.isfinite(r[metric]) and r[metric]>0
            pair['v5_'+metric]=o[metric];pair['v55_'+metric]=r[metric]
        pair.update(total_speedup=o['total_compute_time_ms']/r['total_compute_time_ms'],sgd_readback_speedup=o['sgd_plus_readback_ms']/r['sgd_plus_readback_ms'],stress_ratio=r['stress_value']/o['stress_value'])
        pairs.append(pair)
    groups=collections.defaultdict(list)
    for r in new:groups[r['graph']].append(r)
    seeds=[];summary=[]
    for graph,rows in sorted(groups.items(),key=lambda item:(item[1][0]['nodes'],item[0])):
        assert len(rows)==15 and {(r['seed'],r['repetition']) for r in rows}=={(s,r) for s in range(5) for r in range(3)}
        local=[]
        for seed in range(5):
            n=[r for r in rows if r['seed']==seed];o=[r for r in old if r['graph']==graph and r['seed']==seed]
            row=dict(graph=graph,seed=seed)
            for label,data in [('v5',o),('v55',n)]:
                for metric in METRICS:row[label+'_'+metric]=statistics.median(r[metric] for r in data)
            row.update(total_speedup=row['v5_total_compute_time_ms']/row['v55_total_compute_time_ms'],sgd_readback_speedup=row['v5_sgd_plus_readback_ms']/row['v55_sgd_plus_readback_ms'],stress_ratio=row['v55_stress_value']/row['v5_stress_value'])
            local.append(row);seeds.append(row)
        s=dict(graph=graph,nodes=rows[0]['nodes'],stress_kind=rows[0]['stress_kind'],runs=15,seeds=5,initial_equal=all(r['initial_equal'] for r in pairs if r['graph']==graph))
        for metric in [label+'_'+m for label in ['v5','v55'] for m in METRICS]+['total_speedup','sgd_readback_speedup','stress_ratio']:
            vals=[r[metric] for r in local];s[metric]=statistics.median(vals);s[metric+'_seed_min']=min(vals);s[metric+'_seed_max']=max(vals)
        s['stress_worse_seeds']=sum(r['stress_ratio']>1 for r in local)
        summary.append(s)
    csvfile('runs.csv',[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in new])
    csvfile('paired-runs.csv',pairs);csvfile('paired-seeds.csv',seeds);csvfile('summary.csv',summary)
    save(BASE/'tables/summary.json',summary)
    geomean=lambda values:math.exp(statistics.mean(math.log(v) for v in values))
    aggregate=dict(graphs=38,runs=570,total_speedup_geomean=geomean(s['total_speedup'] for s in summary),sgd_readback_speedup_geomean=geomean(s['sgd_readback_speedup'] for s in summary),stress_ratio_geomean=geomean(s['stress_ratio'] for s in summary),faster_graphs=sum(s['total_speedup']>1 for s in summary),lower_stress_graphs=sum(s['stress_ratio']<1 for s in summary),worse_stress_over_5pct=sum(s['stress_ratio']>1.05 for s in summary),all_initial_equal=sum(r['initial_equal'] for r in pairs),exact_runs=sum(r['stress_kind']=='exact' for r in new),sampled_runs=sum(r['stress_kind']=='sampled' for r in new))
    save(BASE/'tables/aggregate.json',aggregate)
    figures=[]
    def finish(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(BASE/'figures'/f'{name}.{ext}',dpi=170)
        plt.close(fig);figures.append(name)
    for part,data in enumerate([summary[:19],summary[19:]],1):
        fig,axes=plt.subplots(1,3,figsize=(15,9),sharey=True)
        y=np.arange(len(data));labels=[s['graph'] for s in data]
        for ax,metric,title in zip(axes,['total_speedup','sgd_readback_speedup','stress_ratio'],['Total compute: v5 / new','SGD + readback: v5 / new','Stress: new / v5']):
            med=np.array([s[metric] for s in data]);lo=np.array([s[metric+'_seed_min'] for s in data]);hi=np.array([s[metric+'_seed_max'] for s in data])
            ax.errorbar(med,y,xerr=[med-lo,hi-med],fmt='o',capsize=3);ax.axvline(1,color='black',lw=1);ax.set_xscale('log');ax.set_title(title);ax.grid(axis='x',alpha=.25)
        axes[0].set_yticks(y,labels);axes[0].invert_yaxis();finish(fig,f'paired-comparison-{part}')
    fig,axes=plt.subplots(2,1,figsize=(15,9),sharex=True)
    x=np.arange(38);labels=[s['graph'] for s in summary]
    for ax,metric,label in zip(axes,['total_compute_time_ms','stress_value'],['Total compute time (ms)','Full stress (exact or sampled256)']):
        for version in ['v5','v55']:ax.plot(x,[s[version+'_'+metric] for s in summary],marker='o',ms=3,label=version)
        ax.set_yscale('log');ax.set_ylabel(label);ax.legend();ax.grid(axis='y',alpha=.25)
    axes[-1].set_xticks(x,labels,rotation=65,ha='right');finish(fig,'absolute-time-stress')
    fig,ax=plt.subplots(figsize=(9,6))
    ax.scatter([s['total_speedup'] for s in summary],[s['stress_ratio'] for s in summary],c=np.log10([s['nodes'] for s in summary]),cmap='viridis')
    for s in summary:
        if s['stress_ratio']>1.05 or s['total_speedup']<1:ax.annotate(s['graph'],(s['total_speedup'],s['stress_ratio']),fontsize=7)
    ax.axhline(1,color='black',lw=1);ax.axvline(1,color='black',lw=1);ax.set_xscale('log');ax.set_xlabel('Total compute speedup: v5 / new');ax.set_ylabel('Stress ratio: new / v5');finish(fig,'speed-quality')
    text='# v5.5：最新main Atomic Full 全対象の速度・Stress比較\n\n'
    text+=f"完了：{now()}。マージ済みmain `{env['merged_main_commit']}`を凍結ビルドし、**38グラフ×5 seed×3回＝570/570件成功、失敗0件**。4グラフの予備実験を流用せず全件新規測定。EX5は中断を維持。\n\n"
    text+='## 条件と計時\n\nv5のAtomic Full本比較570条件を抽出し、同じ入力SHA256・seed0〜4・各3回・15反復・epsilon 0.1を使用。元jobs中のAtomic Full相対順序を維持し逐次実行。各graph/sessionでseed999をウォームアップし除外。GPU timestamp無効。8,000頂点以下は厳密Full Stress、超過は固定256始点・評価seed0の推定Full Stress。既定のchunk-size=0（bank単位）を使用し、実験側でアルゴリズム・精度・実行設定を変更しない。時間打切りなし。\n\n'
    text+='最新実装はGPU反復を非同期submitし、GPU完了待ちがreadback_time_msへ移っている。**主速度指標はGPU完了待ちを含むtotal_compute_time_ms**。入力・前処理・GPU準備・SGD・読み戻しを含み、Stress評価・座標保存・描画・保存後圧縮は含まない。補助指標はsgd_time_ms+readback_time_ms。rawのsgd_time_ms単独はGPU全体の高速化として解釈しない。\n\n'
    text+='## 集計と全体傾向\n\n各seed内3反復の中央値を取ってから、対応seedの旧/新時間比・新/旧Stress比を計算。その5 seedの中央値をgraph代表値とし、最小〜最大を図の範囲として示す。グラフ全体は38グラフ等重みの幾何平均。単純な570回平均やグラフサイズによる加重平均ではない。\n\n'
    text+=f"- 総計算の高速化（グラフ等重み幾何平均）：**{aggregate['total_speedup_geomean']:.3f}倍**。高速化したグラフ：{aggregate['faster_graphs']}/38。\n- SGD+読み戻しの高速化（同）：{aggregate['sgd_readback_speedup_geomean']:.3f}倍。\n- 新/旧Stress比（同）：**{aggregate['stress_ratio_geomean']:.4f}**。Stress低下：{aggregate['lower_stress_graphs']}/38、5%超悪化：{aggregate['worse_stress_over_5pct']}/38。\n- 初期座標SHA一致：{aggregate['all_initial_equal']}/570。厳密評価{aggregate['exact_runs']}件、256始点評価{aggregate['sampled_runs']}件。全570件で15反復分の全pair更新完了を検証。\n\n"
    text+='## グラフ別結果\n\n時間とStressの値はseed内中央値→5seed中央値。速度比とStress比は対応seed比の中央値なので、表の中央値同士の商と完全一致するとは限らない。速度比は1超が高速、Stress比は1未満が低下。\n\n'
    text+='| graph | n | 評価 | v5総計算 ms | v5.5総計算 ms | 高速化 | v5 Stress | v5.5 Stress | 新/旧Stress |\n|---|---:|---|---:|---:|---:|---:|---:|---:|\n'
    for s in summary:text+=f"| {s['graph']} | {s['nodes']} | {s['stress_kind']} | {s['v5_total_compute_time_ms']:.2f} | {s['v55_total_compute_time_ms']:.2f} | {s['total_speedup']:.3f} | {s['v5_stress_value']:.6g} | {s['v55_stress_value']:.6g} | {s['stress_ratio']:.4f} |\n"
    text+='\n## 解釈と保存\n\nv5は過去の実測であり、常駐プロセスやswap等の環境変動を含む参考比較。Atomicの実行順序による座標・Stress変動を含み、5seedの範囲は信頼区間ではない。品質と速度を分けて判断し、速度向上から品質維持を推測しない。Stressが悪化したグラフも除外せず表・図・全測定に残す。CPU・RR・Sparseは今回新規測定しない。\n\nconfig/に全570jobs・入力hash・元570記録とhash・実行条件、raw/に新測定・コマンド・資源履歴・座標圧縮manifest、warmups/に除外したwarmup、source-snapshot/に測定ソースを保存。各初期・最終座標は可逆XZで復元hashと長さを検証して保持。巨大座標・実行バイナリはローカル、記録・ログ・hash・図表はGitに保存する。\n\n'
    slower=[s for s in summary if s['total_speedup']<1]
    text+='総計算が遅くなった対象：'+ '、'.join(f"{s['graph']}（旧/新時間{s['total_speedup']:.3f}、新/旧Stress{s['stress_ratio']:.4f}）" for s in slower)+'。この3グラフもSGD+読み戻しは高速化しているが、それ以外の処理を含む総計算は遅い。\n\n最新mainのmain.rsではfull_preprocess_sha256による全pairのハッシュ計算とCPUの全pair shuffleがcommon_preprocess_time_msの計測後に実行され、total_compute_time_msには含まれる。したがって、記録された前処理・GPU準備・SGD・読み戻しの単純合計は総計算と一致しない。前処理時間はsummary.csvにも保持するが、ハッシュ計算とshuffleの個別寄与は今回分離計測しておらず、速度低下の全量をどちらかへ帰属しない。\n\n5%超の悪化件数は対応5seed比の中央値に対する判定であり、すべてのseedや実行の品質維持を意味しない。予備実験でdwt_198のseed0に見られたばらつきも、全5seedの結果へ一般化しない。\n\n'
    for name in figures:text+=f'![{name}](../figures/{name}.png)\n\n'
    (BASE/'reports/results.md').write_text(text.rstrip()+'\n')
    (BASE/'README.md').write_text('# v5.5 Atomic Full 全38グラフ\n\n570/570成功。[速度・Stressの結果](reports/results.md) · [38グラフ集計](tables/summary.csv) · [seed対応比較](tables/paired-seeds.csv) · [全測定](tables/runs.csv)\n\n`python3 experiments/run_v55_atomic_full.py`で再開。完了条件を重複測定しない。主速度指標はGPU完了待ちを含む総計算時間。予備実験は別保存し流用しない。EX5は中断したまま。\n')
    save(BASE/'validation/report.json',dict(at=now(),success=570,conditions_verified=True,all_previous_hashes_unchanged=True,all_pair_updates_complete=True,figures=len(figures),aggregate=aggregate))
    print(json.dumps(aggregate,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
