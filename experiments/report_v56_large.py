#!/usr/bin/env python3
"""Report scoped v5.6 measurements; retain incomplete/resource outcomes explicitly."""
import collections
import csv
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiment_results/v5.6/large'
SCOPE=['germany_osm','italy_osm','auto']


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    jobs=json.loads((BASE/'config/jobs.json').read_text())
    old={r['run_id']:r for r in json.loads((BASE/'config/previous-records.json').read_text())}
    rows=[json.loads(p.read_text()) for p in (BASE/'raw').glob('*/record.json')]
    rows=[r for r in rows if r['graph'] in SCOPE]
    groups=collections.defaultdict(list)
    for r in rows:groups[r['graph'],r['pivots']].append(r)
    summary=[];pairs=[]
    for graph in SCOPE:
        for p in sorted({j["pivots"] for j in jobs if j["graph"]==graph},reverse=True):
            group=groups[graph,p];good=[r for r in group if r['status']=='success']
            d=dict(graph=graph,pivots=p,planned=5,recorded=len(group),success=len(good),
                statuses=dict(collections.Counter(r['status'] for r in group)))
            for m in ['preprocess_time_ms','sgd_time_ms','total_compute_time_ms','stress_value','peak_rss_bytes']:
                vals=[r[m] for r in good]
                for label,fn in [('median',statistics.median),('min',min),('max',max)]:
                    d[m+'_'+label]=fn(vals) if vals else None
            ps=[]
            for r in good:
                ref=old[r['previous_run_id']]
                if ref['status']!='success':continue
                checks=['input_sha256','seed','pivots','iterations','epsilon','stress_kind','stress_samples','stress_seed','rr_phase_chunks']
                assert all(r[k]==ref[k] for k in checks),r['run_id']
                q=dict(graph=graph,pivots=p,seed=r['seed'],
                    preprocess_speedup=ref['preprocess_time_ms']/r['preprocess_time_ms'],
                    sgd_speedup=ref['sgd_time_ms']/r['sgd_time_ms'],
                    total_speedup=ref['total_compute_time_ms']/r['total_compute_time_ms'],
                    stress_ratio=r['stress_value']/ref['stress_value'],
                    initial_equal=r['initial_positions_sha256']==ref['initial_positions_sha256'],
                    preprocess_equal=r['preprocess_sha256']==ref['preprocess_sha256'])
                ps.append(q);pairs.append(q)
            d['v5_pairs']=len(ps)
            for k in ['preprocess_speedup','sgd_speedup','total_speedup','stress_ratio']:
                d[k]=statistics.median(r[k] for r in ps) if ps else None
            summary.append(d)
    save(BASE/'tables/summary.json',summary);save(BASE/'tables/paired-v5.json',pairs)
    for filename,data in [('summary.csv',summary),('paired-v5.csv',pairs)]:
        if not data:continue
        with (BASE/'tables'/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator="\n");w.writeheader();w.writerows(data)
    success=sum(r['status']=='success' for r in rows)
    def fmt(x):return '—' if x is None else f'{x:.3f}'
    lines=['# v5.6 大規模実験：'+' / '.join(SCOPE),'',
        f'公開対象の正式測定：成功 {success}/{len(jobs)}。warmupは除外。','',
        '前処理は既存CPU経路を`--bfs-engine cpu`で指定。15反復、epsilon0.1、phase_chunks1、頂点別pivot順序、W128。各pivotでseed0〜4各1回。Stressは固定64始点・評価seed0の推定Full Stress。GPU timestampは無効。', '',
        'SGD・前処理・総計算はnativeタイマーで、Stress評価・保存・圧縮を含まない。中央値と最小〜最大をJSON/CSVに保存。v5比は同一入力・pivot・seedの双方成功した組の旧/新（1超で高速）。Stress比は新/旧。旧v5はW256の歴史データであり、同時測定ではない。','',
        '| 入力 | pivot | 成功/5 | 前処理 ms | SGD ms | 総計算 ms | v5総計算速度比 | v5 SGD速度比 | Stress比 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d in summary:
        lines.append('| '+' | '.join([d['graph'],str(d['pivots']),str(d['success'])+'/5',
            fmt(d['preprocess_time_ms_median']),fmt(d['sgd_time_ms_median']),fmt(d['total_compute_time_ms_median']),
            fmt(d['total_speedup']),fmt(d['sgd_speedup']),fmt(d['stress_ratio'])])+' |')
    lines+=['','未実行・容量停止・実測失敗を0秒として集計しない。germanyと追加pivotにはv5の成功値がなく、速度比を算出しない。GPUハード上限・RSS18GiB・空きRAM512MiB未満10秒・ディスク空き1GiBの条件を保持する。','',
        '## 実装の範囲','',
        'main `20f530952` を基に、既存の4バッファ構成の各容量をadapter上限へ拡張し、dispatchを2次元化。大容量転送の一時stagingを64MiBに制限。先に試作した頂点行タイル方式は不使用。今回の追加変更はこの実験のためにCodexが加えた変更であり、likr氏の変更そのものとして扱わない。ソース・差分・テストログ・実行バイナリhashを保存。', '',
        '## 入力復元','',
        'italyの入力がローカルに存在しなかったため、SuiteSparseのUniversity of Floridaミラーから元データを再取得し、従来のnormalizeで復元。正規化後SHA256はv5記録と完全一致。証跡はvalidation/italy-input-restoration.json。','']
    cpu_path=BASE/'tables/auto-cpu-reference.json'
    if cpu_path.exists():
        cpu=json.loads(cpu_path.read_text())
        lines+=['## autoの過去CPU記録との比較','',
            f"p200/seed0：過去CPU SGD {cpu['cpu_sgd_ms']/1000:.6f}秒 ÷ 今回RR SGD {cpu['rr_v56_sgd_ms']/1000:.6f}秒 = **{cpu['cpu_over_rr_v56']:.3f}倍**。初期座標・前処理fingerprint・頂点辺数・pivot・反復・epsilon一致。CPUは2026-10-30の単一実行で、今回CPUを再測定した値ではない。",'',
            f"ユーザー報告68.18倍との数値比較では{cpu['new_over_reported']:.4f}倍。ただし68.18倍そのものの出典は未照合のため、この比較は暫定。v5 RRに対する速度比と混同しない。",'']
    (BASE/'REPORT.md').write_text('\n'.join(lines).rstrip()+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric,label in [('total_compute_time_ms','Total compute (s)'),('sgd_time_ms','SGD elapsed (s)'),('stress_value','Sampled full stress')]:
        fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
        for ax,graph in zip(axes,SCOPE):
            current=[d for d in summary if d['graph']==graph and d['success']]
            ax.plot([d['pivots'] for d in current],[d[metric+'_median']/(1 if metric=='stress_value' else 1000) for d in current],'-o',label='v5.6')
            reference=[]
            for p in [50,100,200,500,1000]:
                rs=[r for r in old.values() if r['graph']==graph and r['pivots']==p and r['status']=='success']
                if rs:reference.append((p,statistics.median(r[metric] for r in rs)/(1 if metric=='stress_value' else 1000)))
            ax.plot([p for p,v in reference],[v for p,v in reference],'--x',label='v5')
            ax.set_title(graph);ax.set_xlabel('Pivots');ax.set_ylabel(label);ax.set_xscale('log');ax.set_xticks([50,100,200,500,1000],[50,100,200,500,1000]);ax.set_ylim(bottom=0);ax.grid(alpha=.2)
        axes[0].legend();(BASE/'figures').mkdir(exist_ok=True)
        for ext in ['png','pdf']:fig.savefig(BASE/'figures'/f'{metric}.{ext}',dpi=160)
        plt.close(fig)
    print('Saved report:',BASE/'REPORT.md')


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,default=BASE)
    ap.add_argument('--graphs',nargs='+',default=SCOPE)
    args=ap.parse_args()
    BASE=args.base.resolve();SCOPE=args.graphs
    main()
