"""Summarize paired observations without mixing sample sizes or historical CPU runs."""
import csv
import json
import math
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT/'experiment_results/validation/atomic-sparse-20261106'

def main():
    statuses = [json.loads(s) for s in (BASE/'paired-status.jsonl').read_text().splitlines()]
    rows = []
    for status in statuses:
        graph, seed, samples = status['key']
        row = dict(graph=graph, seed=seed, stress_samples=samples, status=status['status'])
        if status['status'] == 'success':
            results = {r['mode']: r for r in map(json.loads, (ROOT/status['output']).read_text().splitlines())}
            cpu, gpu = results['cpu'], results['production']
            assert cpu['constraints'] == gpu['constraints']
            assert gpu['completed'] == gpu['constraints']*15 and gpu['dispatches'] == 15
            row.update(cpu_ms_per_iteration=cpu['iteration_ms']/15,
                       atomic_ms_per_iteration=gpu['iteration_ms']/15,
                       iteration_speedup=cpu['iteration_ms']/gpu['iteration_ms'],
                       setup_inclusive_speedup=cpu['iteration_ms']/(gpu['iteration_ms']+gpu['setup_ms']),
                       preprocessing_inclusive_speedup=(cpu['preprocess_ms']+cpu['iteration_ms'])/(gpu['preprocess_ms']+gpu['setup_ms']+gpu['iteration_ms']),
                       atomic_setup_ms=gpu['setup_ms'],
                       cpu_stress=cpu['stress'], atomic_stress=gpu['stress'],
                       stress_ratio=gpu['stress']/cpu['stress'], constraints=gpu['constraints'],
                       gpu_ms_per_iteration=gpu['gpu_ms']/15,
                       peak_rss_bytes=status['peak_rss_bytes'])
        rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (BASE/'paired-summary.csv').open('w') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
    basic = [r for r in rows if r['seed'] == 0 and r['status'] == 'success']
    extra = [r for r in rows if r['seed'] != 0 and r['status'] == 'success']
    summary = dict(completed=len(rows), planned=48, basic_success=len(basic),
                   basic_failures=[r for r in rows if r['seed'] == 0 and r['status'] != 'success'],
                   extra_success=len(extra))
    for name, group in [('basic', basic), ('heldout', extra)]:
        if not group: continue
        for field in ['iteration_speedup', 'setup_inclusive_speedup', 'preprocessing_inclusive_speedup', 'stress_ratio']:
            vals = [r[field] for r in group]
            summary[f'{name}_{field}'] = dict(min=min(vals), median=statistics.median(vals),
                max=max(vals), geometric_mean=math.exp(statistics.mean(map(math.log, vals))))
    followup = []
    for status_path in sorted((BASE/'quality-followup').glob('*.status.json')):
        path = status_path.with_name(status_path.name.replace('.status.json', '.jsonl'))
        data = {r['mode']:r for r in map(json.loads, path.read_text().splitlines())}
        cpu, old, new = data['cpu'], data['v3-each'], data['production']
        followup.append(dict(graph=cpu['graph'], seed=cpu['seed'], stress_samples=cpu['stress_samples'],
            cpu_ms=cpu['iteration_ms']/15, old_ms=old['iteration_ms']/15, new_ms=new['iteration_ms']/15,
            cpu_over_new=cpu['iteration_ms']/new['iteration_ms'],
            old_over_new=old['iteration_ms']/new['iteration_ms'],
            old_stress_over_cpu=old['stress']/cpu['stress'],
            new_stress_over_cpu=new['stress']/cpu['stress'],
            new_stress_over_old=new['stress']/old['stress']))
    summary['quality_followup'] = followup
    if followup:
        with (BASE/'quality-followup-summary.csv').open('w') as f:
            w = csv.DictWriter(f, list(followup[0])); w.writeheader(); w.writerows(followup)
    (BASE/'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# 測定結果', '', f"完了 {len(rows)}/48 ケース。主比較成功 {len(basic)}/38、追加seed成功 {len(extra)}/10。", '']
    if basic:
        speed = summary['basic_iteration_speedup']
        setup = summary['basic_setup_inclusive_speedup']
        quality = summary['basic_stress_ratio']
        lines += [f"主比較の反復速度比は中央値 **{speed['median']:.2f}倍**、範囲 {speed['min']:.2f}–{speed['max']:.2f}倍。",
                  f"Atomicの初回準備・転送を含めた比は中央値 **{setup['median']:.2f}倍**。",
                  f"推定stress比（Atomic/CPU）は中央値 {quality['median']:.4f}、範囲 {quality['min']:.4f}–{quality['max']:.4f}。", '',
                  '反復時間はホスト待機・完了検証込み。stressは主比較64始点、追加seed256始点の共通サンプルによる推定値。',
                  '差が小さいことはexact full stressの同等性や全seedでの同等性を保証しない。', '',
                  '![CPU comparison](comparison.png)', '',
                  '| グラフ | CPU ms/反復 | Atomic ms/反復 | CPU/Atomic | 初回準備込み | stress比 |',
                  '|---|---:|---:|---:|---:|---:|']
        for r in sorted((r for r in rows if r['seed'] == 0), key=lambda r:r['graph']):
            if r['status'] != 'success':
                lines.append(f"| {r['graph']} | — | {r['status']} | — | — | — |")
            else:
                lines.append(f"| {r['graph']} | {r['cpu_ms_per_iteration']:.1f} | {r['atomic_ms_per_iteration']:.1f} | {r['iteration_speedup']:.2f} | {r['setup_inclusive_speedup']:.2f} | {r['stress_ratio']:.4f} |")
    if extra:
        lines += ['', '## 未使用seedでの追加検証', '', '| グラフ | seed | CPU/Atomic | stress比 |', '|---|---:|---:|---:|']
        for r in sorted(extra, key=lambda r:(r['graph'],r['seed'])):
            lines.append(f"| {r['graph']} | {r['seed']} | {r['iteration_speedup']:.2f} | {r['stress_ratio']:.4f} |")
    if followup:
        lines += ['', '## 品質悪化例の旧Atomicとの再比較', '',
                  '主比較でCPU比のstress上昇が大きかった2グラフを選び、seed 0,1・256共通始点で再評価。',
                  '旧Atomicはv3 shaderと毎反復CPUシャッフルの再現対照。以下は主比較の64始点と別の測定。', '',
                  '| グラフ | seed | CPU ms/反復 | 旧Atomic ms/反復 | 新Atomic ms/反復 | 旧stress/CPU | 新stress/CPU | 新stress/旧 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in followup:
            lines.append(f"| {r['graph']} | {r['seed']} | {r['cpu_ms']:.1f} | {r['old_ms']:.1f} | {r['new_ms']:.1f} | {r['old_stress_over_cpu']:.4f} | {r['new_stress_over_cpu']:.4f} | {r['new_stress_over_old']:.4f} |")
    lines += ['', '[全CSV](paired-summary.csv) · [条件と原因調査](README.md)', '']
    (BASE/'RESULTS.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    basic.sort(key=lambda r:r['graph'])
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    x = range(len(basic))
    axes[0].bar(x, [r['iteration_speedup'] for r in basic], color='#207c9e', label='SGD iterations')
    axes[0].plot(x, [r['setup_inclusive_speedup'] for r in basic], 'o', ms=3, color='#d87222', label='Including Atomic setup/upload')
    axes[0].set_ylabel('CPU / Atomic time'); axes[0].legend()
    axes[0].set_ylim(0, max(r['iteration_speedup'] for r in basic)*1.25)
    axes[1].plot(x, [r['stress_ratio'] for r in basic], 'o', ms=4, color='#207c9e')
    axes[1].set_ylabel('Atomic / CPU sampled full stress')
    for ax in axes:
        ax.axhline(1, color='black', ls='--', lw=1); ax.grid(axis='y', alpha=.25)
    axes[1].set_xticks(list(x), [r['graph'] for r in basic], rotation=90)
    fig.suptitle(f'Atomic Sparse SGD: paired CPU comparison | M4 Pro | 15 iterations, 200 pivots, seed 0\n{len(basic)}/38 graph pairs completed; 64 shared stress sources; timing includes host waits')
    fig.tight_layout()
    fig.savefig(BASE/'comparison.png', dpi=160)
    fig.savefig(BASE/'comparison.svg')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
