#!/usr/bin/env python3
"""Sequential RR/CPU benchmark. Run from repository root; baseline binaries are explicit."""
import argparse, hashlib, json, pathlib, subprocess, time
p=argparse.ArgumentParser()
p.add_argument('--baseline-dir',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--suite',choices=['full','sparse'],required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def run(tag,cmd):
 out=a.output/(tag+'.jsonl');err=a.output/(tag+'.stderr')
 if out.exists() and out.stat().st_size:
  try:
   existing=[json.loads(l) for l in out.read_text().splitlines()]
   expected=2 if 'cpu' in cmd else 1
   if len(existing)==expected: return
  except ValueError: pass
 manifest={'command':[str(v) for v in cmd],'binary_sha256':hashlib.sha256(pathlib.Path(cmd[0]).read_bytes()).hexdigest()}
 (a.output/(tag+'.command.json')).write_text(json.dumps(manifest,indent=2)+'\n')
 print('START',tag,flush=True);t=time.monotonic()
 with out.open('w') as f,err.open('w') as e:
  r=subprocess.run([str(v) for v in cmd],stdout=f,stderr=e)
 if r.returncode: raise RuntimeError((tag,r.returncode))
 rows=[json.loads(l) for l in out.read_text().splitlines()]
 for row in rows:
  if row.get('final_positions_path'):
   b=pathlib.Path(row['final_positions_path']).read_bytes().split(b'# Positions',1)[1]
   row['final_coordinate_text_sha256']=hashlib.sha256(b).hexdigest()
  row['process_wall_seconds']=time.monotonic()-t
 out.write_text(''.join(json.dumps(row)+'\n' for row in rows))
 print('DONE',tag,[(r.get('iteration_time_ms',r.get('iteration_seconds')),r.get('stress_value',r.get('stress'))) for r in rows],flush=True)
if a.suite=='full':
 for graph in ['fe_4elt2','delaunay_n14']:
  for seed in [0,1,3]:
   variants=[('before',a.baseline_dir/'rr-gpu'),('after',pathlib.Path('implementations/full/rr_gpu/target/release/rr-gpu')),('cpu',pathlib.Path('implementations/full/baseline-sgd-non-gpu/target/release/baseline-sgd-non-gpu'))]
   if seed==1: variants.reverse()
   for label,binary in variants:
    tag=f'full-{graph}-s{seed}-{label}'
    cmd=[binary,'--input',f'data/{graph}.mtx','--iterations','15','--seed',str(seed),'--output-format','json','--output-dir',a.output/'artifacts','--run-id',tag]
    if label!='cpu':cmd+=['--tile-size','1024','--workgroup-size','1024','--coordinate-memory','shared']
    run(tag,cmd)
else:
 graphs=[('luxembourg_osm',200),('web-NotreDame',200),('coAuthorsCiteseer',200),('delaunay_n20',64)]
 for graph,h in graphs:
  for seed in ([0] if graph=='delaunay_n20' else [0,1,3]):
   source=pathlib.Path('data')/(graph+'.mtx')
   for label,binary in [('before',a.baseline_dir/'sparse-probe'),('after',pathlib.Path('implementations/sparse/sparse-sgd-gpu/target/release/examples/quality_probe'))]:
    run(f'sparse-{graph}-h{h}-s{seed}-{label}',[binary,source,str(h),str(seed),'cpu' if label=='after' else 'no-cpu','1'])
