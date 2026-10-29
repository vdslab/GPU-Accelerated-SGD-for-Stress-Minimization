#!/usr/bin/env python3
"""Quick paired-seed comparison; run from repository root after release builds."""
import csv, hashlib, json, statistics, subprocess, time
from pathlib import Path
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
DATASETS = ['dwt_419', '1138_bus', 'dwt_2680', '3elt', 'USpowerGrid']
METHODS = {
    'cpu': ('implementations/sparse/baseline-sparse-sgd-non-gpu/target/release/baseline-sparse-sgd-non-gpu', []),
    'gpu_shuffle': ('implementations/sparse/vram-lock-sparse/target/release/vram-lock-sparse', ['--cpu-shuffle']),
    'gpu_no_shuffle': ('implementations/sparse/vram-lock-sparse/target/release/vram-lock-sparse', []),
}
metadata = {'datasets': DATASETS, 'seeds': [0,1,2], 'iterations':15, 'pivots':200, 'epsilon':0.1,
            'git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'binary_sha256': {k:hashlib.sha256((ROOT/v[0]).read_bytes()).hexdigest() for k,v in METHODS.items()}}
(OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
(OUT/'source.diff').write_text(subprocess.check_output(['git','diff'],cwd=ROOT,text=True))
rows=[]
for dataset in DATASETS:
    for seed in [0,1,2]:
        group=[]
        # Rotate order to reduce systematic order/thermal bias, never run concurrently.
        methods=list(METHODS)
        methods=methods[seed:]+methods[:seed]
        for method in methods:
            binary, extra=METHODS[method]
            run_id=f'{dataset}-s{seed}-{method}'
            cmd=[str(ROOT/binary),'--input',str(ROOT/'data'/f'{dataset}.mtx'),
                 '--pivots','200','--iterations','15','--epsilon','0.1','--seed',str(seed),
                 '--run-id',run_id,'--output-format','json','--output-dir',str(OUT/'artifacts')]+extra
            start=time.perf_counter()
            result=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=180)
            (OUT/f'{run_id}.stderr').write_text(result.stderr)
            result.check_returncode()
            record=json.loads(result.stdout)
            record.update(comparison_method=method,command=cmd,process_wall_ms=(time.perf_counter()-start)*1000)
            (OUT/f'{run_id}.json').write_text(json.dumps(record,indent=2)+'\n')
            assert record['attempted_updates']==record['completed_updates']
            assert record['stress_kind']=='exact'
            rows.append(record);group.append(record)
            print(f"{run_id}: iteration={record['iteration_time_ms']:.2f}ms stress={record['stress_value']:.3f}",flush=True)
        for key in ['initial_positions_sha256','preprocess_sha256','input_sha256','constraints','nodes','edges']:
            assert len({r[key] for r in group})==1,(dataset,seed,key)
fields=['dataset','comparison_method','nodes','edges','constraints','iteration_ms_median','algorithm_cold_ms_median','algorithm_warm_ms_median','stress_mean','stress_min','stress_max','dispatches_median']
with (OUT/'summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for dataset in DATASETS:
        for method in METHODS:
            rs=[r for r in rows if r['dataset']==dataset and r['comparison_method']==method]
            row={k:rs[0][k] for k in ['dataset','comparison_method','nodes','edges','constraints']}
            for dest,src in [('iteration_ms_median','iteration_time_ms'),('algorithm_cold_ms_median','algorithm_time_cold_ms'),('algorithm_warm_ms_median','algorithm_time_warm_ms')]:
                row[dest]=statistics.median(r[src] for r in rs)
            row.update(stress_mean=statistics.mean(r['stress_value'] for r in rs),stress_min=min(r['stress_value'] for r in rs),stress_max=max(r['stress_value'] for r in rs),dispatches_median=statistics.median(r['dispatches'] or 0 for r in rs))
            writer.writerow(row)
print('DONE: matched preprocessing and initial positions; all constraints completed; exact stress.',flush=True)
