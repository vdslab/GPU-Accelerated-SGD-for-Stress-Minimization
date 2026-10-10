"""Create reproducible v1 tables, static paper figures, and an evidence-based report."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'experiment_results/v1'
METHODS = ['sgd', 'atomic_sgd', 'rr_sgd', 'sparse_sgd', 'rr_sparse_sgd']
LABELS = {'sgd': 'CPU SGD', 'atomic_sgd': 'Atomic SGD', 'rr_sgd': 'RR-SGD',
          'sparse_sgd': 'CPU Sparse SGD', 'rr_sparse_sgd': 'RR-Sparse SGD'}
COLORS = {'sgd': '#555555', 'atomic_sgd': '#d55e00', 'rr_sgd': '#0072b2',
          'sparse_sgd': '#555555', 'rr_sparse_sgd': '#009e73'}
FAMILIES = {'full': METHODS[:3], 'sparse': METHODS[3:]}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelsize': 10, 'legend.fontsize': 8, 'svg.fonttype': 'none',
                     'savefig.dpi': 240})


def dump_csv(path, records):
    if not records:
        return
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values, statistic='mean'):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return [None, None, None]
    function = np.mean if statistic == 'mean' else np.median
    center = float(function(x))
    if len(x) == 1:
        return [center, center, center]
    rng = np.random.default_rng(20261011)
    samples = function(x[rng.integers(0, len(x), (10000, len(x)))], axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return [center, float(low), float(high)]


def save_figure(fig, name):
    fig.savefig(BASE / 'figures' / (name + '.png'), bbox_inches='tight')
    fig.savefig(BASE / 'figures' / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)


def load_data():
    records, sources = [], []
    expectations = {}
    for folder in sorted((BASE / 'raw').glob('v1-*')):
        path = folder / 'results.jsonl'
        if not path.exists():
            continue
        manifest = json.loads((folder / 'manifest.json').read_text())
        for dataset in manifest['datasets']:
            for method in manifest['methods']:
                expectations[(dataset['name'], method)] = len(manifest['seeds'])
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
        # Keep retry history in raw; use only latest attempt for each run id.
        latest = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row['experiment_id'] = folder.name
            latest[row['run_id']] = row
        records.extend(latest.values())
    return records, sources, expectations


def main():
    for name in ['tables', 'figures', 'report']:
        (BASE / name).mkdir(exist_ok=True)
    rows, sources, expectations = load_data()
    if not rows:
        raise SystemExit('No v1 results; run experiments first.')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['method'])].append(row)
    available = {(r['dataset'], r['method'], r['seed']): r for r in rows if r['status'] == 'success'}
    datasets = json.loads((BASE / 'config/datasets.json').read_text())
    names = [item['name'] for item in datasets if any((item['name'], m) in groups for m in METHODS)]
    small = [item['name'] for item in datasets[:10] if item['name'] in names]
    large = [item['name'] for item in datasets[10:] if item['name'] in names]
    dataset_info = {item['name']: item for item in datasets}
    paired = defaultdict(list)
    validation = {'errors': [], 'warnings': [], 'formal_publication_ready': False}
    for r in available.values():
        baseline = 'sgd' if r['family'] == 'full' else 'sparse_sgd'
        ref = available.get((r['dataset'], baseline, r['seed']))
        if ref is None:
            continue
        for key in ['input_sha256', 'initial_positions_sha256', 'nodes', 'edges', 'constraints',
                    'iterations', 'epsilon', 'pivots', 'git_commit', 'cpu_model']:
            if r[key] != ref[key]:
                validation['errors'].append(f"{r['run_id']}: mismatched {key}")
        if r['family'] == 'sparse' and r['preprocess_sha256'] != ref['preprocess_sha256']:
            validation['errors'].append(f"{r['run_id']}: mismatched sparse preprocessing")
        if r['stress_kind'] != ref['stress_kind'] or r['stress_seed'] != ref['stress_seed'] or r['stress_samples'] != ref['stress_samples']:
            validation['errors'].append(f"{r['run_id']}: mismatched stress evaluator")
        if r['git_dirty']:
            validation['errors'].append(f"{r['run_id']}: dirty source tree")
        if r.get('retry_failures', 0):
            validation['errors'].append(f"{r['run_id']}: {r['retry_failures']} incomplete atomic updates")
        if not (np.isfinite(r['stress_value']) and r['iteration_time_ms'] > 0):
            validation['errors'].append(f"{r['run_id']}: invalid measurement")
        paired[(r['dataset'], r['method'])].append({
            'seed': r['seed'], 'stress_ratio': r['stress_value'] / ref['stress_value'],
            'iteration_speedup': ref['iteration_time_ms'] / r['iteration_time_ms']})
    summaries = []
    for name in names:
        for method in METHODS:
            expected = expectations.get((name, method))
            if expected is None:
                continue
            measured = [r for r in groups[(name, method)] if r['status'] == 'success']
            failed = [r for r in groups[(name, method)] if r['status'] != 'success']
            pairs = paired[(name, method)]
            times = [r['iteration_time_ms'] for r in measured]
            stress = [r['stress_value'] for r in measured]
            quality_ci = interval([r['stress_ratio'] for r in pairs])
            speed_ci = interval([r['iteration_speedup'] for r in pairs], 'median')
            summaries.append(dict(dataset=name, method=method, family='full' if method in METHODS[:3] else 'sparse',
                nodes=dataset_info[name]['nodes'], edges=dataset_info[name]['edges'],
                constraints=measured[0]['constraints'] if measured else None,
                expected=expected, success=len(measured), failures=len(failed), missing=expected-len(measured)-len(failed),
                stress_kind=measured[0]['stress_kind'] if measured else None,
                iteration_median_ms=float(np.median(times)) if times else None,
                iteration_q1_ms=float(np.quantile(times,.25)) if times else None,
                iteration_q3_ms=float(np.quantile(times,.75)) if times else None,
                stress_mean=float(np.mean(stress)) if stress else None,
                stress_min=min(stress) if stress else None, stress_max=max(stress) if stress else None,
                stress_ratio_mean=quality_ci[0], stress_ratio_ci_low=quality_ci[1], stress_ratio_ci_high=quality_ci[2],
                paired_iteration_speedup_median=speed_ci[0], speedup_ci_low=speed_ci[1], speedup_ci_high=speed_ci[2],
                paired_seeds=len(pairs)))
    lookup = {(s['dataset'],s['method']):s for s in summaries}
    for s in summaries:
        if s['success'] < s['expected']:
            validation['warnings'].append(f"{s['dataset']} {s['method']}: {s['success']}/{s['expected']} successful")
    validation['warnings'].extend(['10-seed quality is exploratory; planned final quality requires 25 seeds.',
                                   'Large graph timing has only 3 seeds.', 'Sparse atomic is not implemented.',
                                   'btree9 and block_2000 are unavailable in this collection.'])
    dump_csv(BASE/'tables/runs.csv',rows)
    dump_csv(BASE/'tables/summary.csv',summaries)
    flat_pairs=[dict(dataset=d,method=m,**item) for (d,m),items in paired.items() for item in items]
    dump_csv(BASE/'tables/paired-ratios.csv',flat_pairs)
    (BASE/'tables/validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    (BASE/'tables/summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    table=['| Graph | Method | n | Success | Stress | SGD time, ms (median [Q1, Q3]) | Speedup | Stress ratio |',
           '|---|---|---:|---:|---|---|---|---|']
    latex=['% v1 exploratory results: 10 seeds for small graphs, 3 for large graphs',
           r'\begin{tabular}{llrrrr}',r'\hline',r'Graph & Method & Runs & Stress & SGD time (ms) & Speedup \\',r'\hline']
    for s in summaries:
        if s['success']:
            t=f"{s['iteration_median_ms']:.2f} [{s['iteration_q1_ms']:.2f}, {s['iteration_q3_ms']:.2f}]"
            sp=f"{s['paired_iteration_speedup_median']:.2f}" if s['paired_seeds'] else 'N/A'
            qr=f"{s['stress_ratio_mean']:.3f}" if s['paired_seeds'] else 'N/A'
            quality=f"{s['stress_mean']:.2f} ({s['stress_kind']})"
            graph_label = s['dataset'].replace('_', '\\_')
            latex.append(f"{graph_label} & {LABELS[s['method']]} & {s['success']} & {s['stress_mean']:.2f} & {s['iteration_median_ms']:.2f} & {sp} \\")
        else:
            t=sp=qr=quality='N/A'
        table.append(f"| {s['dataset']} | {LABELS[s['method']]} | {s['nodes']} | {s['success']}/{s['expected']} | {quality} | {t} | {sp} | {qr} |")
    latex.extend([r'\hline',r'\end{tabular}'])
    (BASE/'tables/results.md').write_text('\n'.join(table)+'\n')
    (BASE/'tables/results.tex').write_text('\n'.join(latex)+'\n')

    # Paired means/medians with seeded bootstrap confidence intervals; no pooled graph averages.
    for kind, field, lowfield, highfield, ylabel, filename in [
        ('quality','stress_ratio_mean','stress_ratio_ci_low','stress_ratio_ci_high','Full-stress ratio to family CPU baseline','01_stress_ratio'),
        ('speed','paired_iteration_speedup_median','speedup_ci_low','speedup_ci_high','Host SGD iteration speedup (CPU / GPU)','02_iteration_speedup')]:
        fig, axes=plt.subplots(2,1,figsize=(11,6.4),sharex=True)
        for ax,(family,methods) in zip(axes,FAMILIES.items()):
            ax.axhline(1,color='#666666',linestyle='--',linewidth=.8)
            if kind=='quality':ax.axhline(1.05,color='#999999',linestyle=':',linewidth=1,label='5% margin (exploratory)')
            for mi,m in enumerate(methods[1:]):
                xpos=[]; values=[]; lows=[]; highs=[]
                for j,d in enumerate(small):
                    s=lookup.get((d,m))
                    if s and s[field] is not None:
                        xpos.append(j+(mi-.5)*.18 if family=='full' else j)
                        values.append(s[field]);lows.append(max(0,s[field]-s[lowfield]));highs.append(max(0,s[highfield]-s[field]))
                ax.errorbar(xpos,values,yerr=[lows,highs],fmt='o' if mi==0 else 's',color=COLORS[m],
                            label=LABELS[m],capsize=3,markersize=4,linewidth=1)
            ax.set_ylabel(ylabel);ax.set_title(('(a) Full SGD' if family=='full' else '(b) Sparse SGD')+'; 10 seeds',loc='left')
            if kind=='speed':ax.set_yscale('log')
            ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,loc='best')
        axes[-1].set_xticks(range(len(small)),small,rotation=30,ha='right')
        fig.tight_layout();save_figure(fig,filename)

    fig,axes=plt.subplots(2,1,figsize=(11,6.4),sharex=True)
    for ax,(family,methods) in zip(axes,FAMILIES.items()):
        for mi,m in enumerate(methods):
            x=[];y=[];lo=[];hi=[]
            for j,d in enumerate(small):
                s=lookup.get((d,m))
                if s and s['success']:
                    x.append(j+(mi-(len(methods)-1)/2)*.22);y.append(s['iteration_median_ms'])
                    lo.append(y[-1]-s['iteration_q1_ms']);hi.append(s['iteration_q3_ms']-y[-1])
            ax.bar(x,y,width=.2,color=COLORS[m],label=LABELS[m],yerr=[lo,hi],capsize=2)
        ax.set_yscale('log');ax.set_ylabel('15-iteration host time (ms)')
        ax.set_title(f'{family.capitalize()} SGD: median and IQR',loc='left');ax.legend(frameon=False);ax.grid(axis='y',alpha=.18)
    axes[-1].set_xticks(range(len(small)),small,rotation=30,ha='right')
    fig.tight_layout();save_figure(fig,'03_iteration_time')

    fig,axes=plt.subplots(2,5,figsize=(13,5.6))
    for ax,d in zip(axes.flat,small):
        values=[[r['stress_value'] for r in groups[(d,m)] if r['status']=='success'] for m in METHODS]
        nonempty=[(i,v) for i,v in enumerate(values) if v]
        if nonempty:
            box=ax.boxplot([v for i,v in nonempty],positions=[i for i,v in nonempty],patch_artist=True,widths=.6,
                           showfliers=True,medianprops={'color':'black','linewidth':1})
            for artist,(i,v) in zip(box['boxes'],nonempty):artist.set_facecolor(COLORS[METHODS[i]]);artist.set_alpha(.6)
        ax.set_title(d);ax.set_xticks(range(5),['CPU','AT','RR','S-CPU','S-RR'],rotation=45)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(axis='y',alpha=.15)
    axes[0,0].set_ylabel('Exact full stress');axes[1,0].set_ylabel('Exact full stress')
    fig.tight_layout();save_figure(fig,'04_final_stress_distribution')

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(family,methods) in zip(axes,FAMILIES.items()):
        for m in methods[1:]:
            for d in small:
                s=lookup.get((d,m))
                if s and s['paired_seeds']:
                    ax.scatter(s['paired_iteration_speedup_median'],s['stress_ratio_mean'],c=COLORS[m],s=22)
                    ax.annotate(d,(s['paired_iteration_speedup_median'],s['stress_ratio_mean']),fontsize=6,xytext=(3,3),textcoords='offset points')
            ax.plot([],[],color=COLORS[m],marker='o',ls='',label=LABELS[m])
        ax.axvline(1,color='#666666',linestyle='--',linewidth=.8);ax.axhline(1.05,color='#999999',linestyle=':',linewidth=1)
        ax.set_xscale('log');ax.set_xlabel('Iteration speedup (CPU / GPU)');ax.set_ylabel('Mean full-stress ratio')
        ax.set_title(family.capitalize()+' SGD');ax.legend(frameon=False);ax.grid(alpha=.15)
    fig.tight_layout();save_figure(fig,'05_quality_speed_tradeoff')

    if large:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for mi,m in enumerate(METHODS[3:]):
            x=[];y=[];lo=[];hi=[]
            for j,d in enumerate(large):
                s=lookup.get((d,m))
                if s and s['success']:
                    x.append(j+(mi-.5)*.24);y.append(s['iteration_median_ms'])
                    lo.append(y[-1]-s['iteration_q1_ms']);hi.append(s['iteration_q3_ms']-y[-1])
            axes[0].bar(x,y,width=.22,color=COLORS[m],label=LABELS[m],yerr=[lo,hi],capsize=2)
        axes[0].set_xticks(range(len(large)),large,rotation=35,ha='right');axes[0].set_yscale('log')
        axes[0].set_ylabel('15-iteration host time (ms)');axes[0].legend(frameon=False);axes[0].grid(axis='y',alpha=.15)
        for d in small+large:
            s=lookup.get((d,'rr_sparse_sgd'))
            if s and s['paired_seeds']:
                axes[1].scatter(s['nodes'],s['paired_iteration_speedup_median'],color=COLORS['rr_sparse_sgd'],marker='s' if d in large else 'o',s=25)
                if d in large:axes[1].annotate(d,(s['nodes'],s['paired_iteration_speedup_median']),fontsize=6,xytext=(3,3),textcoords='offset points')
        axes[1].set_xscale('log');axes[1].set_yscale('log');axes[1].axhline(1,color='#666666',linestyle='--',linewidth=.8)
        axes[1].set_xlabel('Nodes in largest connected component');axes[1].set_ylabel('RR-Sparse iteration speedup');axes[1].grid(alpha=.15)
        fig.suptitle('Sparse scaling: exploratory large-graph results (3 seeds)');fig.tight_layout();save_figure(fig,'06_sparse_scaling')

    # Same seed across all methods, picked using CPU full-SGD median stress only.
    layout_selections=[]
    fig,axes=plt.subplots(3,5,figsize=(12,7))
    for ri,d in enumerate(['USpowerGrid','3elt','EVA']):
        refs=[r for r in groups[(d,'sgd')] if r['status']=='success']
        if not refs:continue
        median=np.median([r['stress_value'] for r in refs])
        ref=min(refs,key=lambda r:abs(r['stress_value']-median));seed=ref['seed']
        for ci,m in enumerate(METHODS):
            ax=axes[ri,ci];r=available.get((d,m,seed))
            ax.set_axis_off();ax.set_aspect('equal')
            if ri==0:ax.set_title(LABELS[m])
            if r is None:ax.text(.5,.5,'Unavailable',ha='center',transform=ax.transAxes);continue
            edges=[];positions=[];section=None
            for line in Path(r['final_positions_path']).read_text().splitlines():
                if line.startswith('# Edges'):section='edges'
                elif line.startswith('# Positions'):section='positions'
                elif line.strip() and not line.startswith('#'):
                    if section=='edges':edges.append([int(v) for v in line.split()[:2]])
                    elif section=='positions':positions.append([float(v) for v in line.split()[:2]])
            xy=np.asarray(positions);ed=np.asarray(edges)
            ax.add_collection(LineCollection(xy[ed],colors='#333333',linewidths=.12,alpha=.45,rasterized=True));ax.autoscale()
            ax.text(.02,.02,f'{d}; seed {seed}\nStress {r["stress_value"]:.0f}',transform=ax.transAxes,fontsize=7)
            layout_selections.append({'dataset':d,'method':m,'seed':seed,'path':r['final_positions_path'],'sha256':sha(Path(r['final_positions_path']))})
    fig.tight_layout();save_figure(fig,'07_representative_layouts')
    (BASE/'tables/layout-selection.json').write_text(json.dumps(layout_selections,indent=2)+'\n')

    failures=[r for r in rows if r['status']!='success']
    dump_csv(BASE/'tables/failures.csv',failures)
    report=['# 実験結果 v1（初回評価）','',
            f'正式実行：{len(rows)}件。成功 {len(available)}件、失敗 {len(failures)}件。',
            '', '## 比較条件',
            '- 前処理を除いた15反復のhost実時間を主速度指標とする。同期・シャッフルは含む。',
            '- 小・中規模10 seed、大規模3 seed。品質25 seedの正式判定に先立つ暫定結果。',
            '- exact full stressと64始点のsampled stressを混ぜず、各グラフ内の対応seedで比較。',
            '- 全手法へ同じ無向・自己ループなし・最大連結成分の入力を渡す。',
            '', '## グラフ別の結果',
            '| Graph | Method | SGD speedup | Full-stress ratio | Paired seeds |',
            '|---|---|---:|---:|---:|']
    for s in summaries:
        if s['method'] in ['atomic_sgd','rr_sgd','rr_sparse_sgd'] and s['paired_seeds']:
            report.append(f"| {s['dataset']} | {LABELS[s['method']]} | {s['paired_iteration_speedup_median']:.2f}x | {s['stress_ratio_mean']:.3f} | {s['paired_seeds']} |")
    report+=['','## 解釈と限界']
    for m in ['atomic_sgd','rr_sgd','rr_sparse_sgd']:
        subset=[s for s in summaries if s['method']==m and s['paired_seeds']]
        if subset:
            fast=sum(s['paired_iteration_speedup_median']>1 for s in subset)
            stable=sum(s['stress_ratio_ci_high']<=1.05 for s in subset)
            report.append(f'- {LABELS[m]}：対応比較可能な{len(subset)}グラフ中、中央値で高速化は{fast}グラフ。stress比の暫定95% CI上限が1.05以下は{stable}グラフ。')
    report+=['- 全グラフをまとめた単一の速度向上率で代表させない。グラフの構造・規模に依存する。',
             '- GPU実装はf32、CPUはf64。差には精度と更新順の両方が含まれる。',
             '- この結果はSGD部分の性能であり、初回レイアウト生成全体の高速化率ではない。',
             '- 大規模3 seedから信頼性の高い速度・品質の結論を出さない。',
             '- 同一seedの時間は1回ずつ。将来は固定seedの繰返しで機器の時間変動とseed差を分離する。',
             '- Sparse atomicが未実装のためSparseの競合方式同士の比較は未完成。',
             '- 欠測や失敗は図でゼロにせず、failures.csvおよび元stderrに理由を残す。',
             '', '## 検証', f'- 比較整合性エラー：{len(validation["errors"])}件。',
             '- 検証詳細：../tables/validation.json。正式掲載可否はfalse（標本数不足）。',
             '', '## 次に必要な実験',
             '1. 小・中規模を25 seedへ拡張し、品質の非劣性を判定する。',
             '2. 大規模を最低10 seedへ拡張する。失敗したGPU入力はメモリ制約を解消してから再測定。',
             '3. Sparse atomicを追加し、同じ制約でRRと比較する。',
             '4. 代表3グラフのiterationごとのstress収束を別診断実行で記録する（固定15反復の学習率を維持）。',
             '', '## 図の読み方',
             '- 01：CPU基準のstress比。点は平均、誤差棒は対応seedのbootstrap 95% CI。',
             '- 02：SGD部分の速度向上率。点は対応比の中央値、誤差棒はbootstrap 95% CI。',
             '- 03：SGD時間。棒は中央値、誤差棒はIQR。',
             '- 04：最終exact full stressの分布。',
             '- 05：品質と速度の両立。右側ほど速く、下側ほどstressが低い。',
             '- 06：Sparseの規模拡大。丸は10 seed、四角は3 seed。',
             '- 07：CPU full-SGDのstress中央値に近いseedを全5手法で共有した代表配置。',
             '', '## 論文の体裁との対応',
             '- SGD論文Figure 7・14の品質分布、Figure 10の品質と時間を同時に示す考え方を参考にした。',
             '- DRGraph Table 2・5のグラフ別比較とFigure 6の規模に応じた速度評価を参考にした。',
             '- DRGraphは尺度最適化したnormalized stressを使う。本結果のraw full stressと掲載数値は直接比較しない。',
             '- PNGは確認用、SVGは論文で編集・拡大するための版。収束曲線は未取得であり図を捏造しない。']
    (BASE/'report/RESULTS.md').write_text('\n'.join(report)+'\n')
    metadata={'sources':sources,'script_sha256':sha(Path(__file__)),
              'config_sha256':sha(BASE/'config/datasets.json'),'bootstrap_resamples':10000,
              'bootstrap_seed':20261011,'publication_ready':False,
              'commits':sorted({r['git_commit'] for r in rows}),
              'dirty_runs':sum(bool(r['git_dirty']) for r in rows),
              'figures':{p.name:sha(p) for p in (BASE/'figures').glob('*') if p.is_file()}}
    (BASE/'tables/provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'{len(rows)} runs; {len(available)} success; {len(failures)} failures; {len(validation["errors"])} validation errors')
    if validation['errors']:
        raise SystemExit('Comparison validation failed; figures are drafts, inspect validation.json.')


if __name__ == '__main__':
    main()
