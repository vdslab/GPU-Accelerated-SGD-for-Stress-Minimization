#!/usr/bin/env python3
import csv,hashlib,json,subprocess,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
METHODS={
 'cpu':(ROOT/'implementations/sparse/baseline-sparse-sgd-non-gpu/target/release/baseline-sparse-sgd-non-gpu',[]),
 'before_shuffle':(OUT/'binaries/vram-lock-sparse-before',['--cpu-shuffle']),
 'before_fixed':(OUT/'binaries/vram-lock-sparse-before',[]),
 'fenced_shuffle':(ROOT/'implementations/sparse/vram-lock-sparse/target/release/vram-lock-sparse',['--cpu-shuffle']),
 'fenced_fixed':(ROOT/'implementations/sparse/vram-lock-sparse/target/release/vram-lock-sparse',[]),
}
GROUPS=[('delaunay_n17',0),('delaunay_n17',1),('delaunay_n17',2),('smallworld',0)]
AUDIT=ROOT/'tools/diagnostics/sparse-audit/target/release/sparse-audit'
metadata={'groups':GROUPS,'iterations':15,'pivots':200,'epsilon':0.1,'stress_samples':4096,'stress_seed':20261027,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'binary_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,(p,_) in METHODS.items()},'audit_sha256':hashlib.sha256(AUDIT.read_bytes()).hexdigest()}
(OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
(OUT/'source.diff').write_text(subprocess.check_output(['git','diff'],cwd=ROOT,text=True))
rows=[]
for index,(dataset,seed) in enumerate(GROUPS):
 group={}
 order=list(METHODS);shift=index%len(order);order=order[shift:]+order[:shift]
 for method in order:
  binary,extra=METHODS[method];rid=f'{dataset}-s{seed}-{method}'
  cmd=[str(binary),'--input',str(ROOT/'data'/f'{dataset}.mtx'),'--pivots','200','--iterations','15','--epsilon','0.1','--seed',str(seed),'--run-id',rid,'--output-format','json','--output-dir',str(OUT/'artifacts')]+extra
  started=time.perf_counter();res=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=600)
  (OUT/f'{rid}.stderr').write_text(res.stderr);res.check_returncode();r=json.loads(res.stdout)
  r.update(comparison_method=method,command=cmd,process_wall_ms=(time.perf_counter()-started)*1000)
  (OUT/f'{rid}.json').write_text(json.dumps(r,indent=2)+'\n')
  assert r['attempted_updates']==r['completed_updates'];group[method]=r
  print(f'{rid}: iterations={r["iteration_time_ms"]/1000:.3f}s dispatches={r["dispatches"]}; stress4096 pending',flush=True)
 for k in ['initial_positions_sha256','preprocess_sha256','input_sha256','constraints','nodes','edges']:
  assert len({r[k] for r in group.values()})==1,(dataset,seed,k)
 eval_order=['fenced_shuffle','cpu','before_shuffle','before_fixed','fenced_fixed']
 paths=[group[m]['final_positions_path'] for m in eval_order]
 cmd=[str(AUDIT),'paired',str(ROOT/'data'/f'{dataset}.mtx')]+paths
 res=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=600);res.check_returncode()
 (OUT/f'{dataset}-s{seed}-evaluation.jsonl').write_text(res.stdout)
 (OUT/f'{dataset}-s{seed}-evaluation-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 evaluations=[json.loads(l) for l in res.stdout.splitlines()];assert len(evaluations)==5
 for method,e in zip(eval_order,evaluations):
  r=group[method];assert e['file']==r['final_positions_path']
  row={k:r[k] for k in ['dataset','seed','nodes','constraints','iteration_time_ms','algorithm_time_cold_ms','dispatches']}
  row.update(method=method,stress4096=e['stress'],delta_vs_fenced_shuffle=e['paired_difference'],delta_ci_low=e['paired_difference_95ci'][0],delta_ci_high=e['paired_difference_95ci'][1]);rows.append(row)
  print(f'{dataset}-s{seed}-{method}: stress4096={e["stress"]:.3f}',flush=True)
 with (OUT/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('DONE: all 20 runs and paired 4096-source evaluations complete.',flush=True)
