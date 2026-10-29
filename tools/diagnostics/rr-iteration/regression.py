#!/usr/bin/env python3
"""Additional sparse graph shapes: compare current GPU to saved pre-change hashes."""
import json,pathlib,subprocess,sys
out=pathlib.Path(sys.argv[1]); old=pathlib.Path('experiment_results/v2/validation/rr-common-optimization-20261030');results=[]
for graph in ['EVA','USpowerGrid','preferentialAttachment','smallworld','web-Stanford']:
 for seed in [0,1,3]:
  tag=f'regression-{graph}-s{seed}'
  path=out/(tag+'.jsonl')
  cmd=['implementations/sparse/sparse-sgd-gpu/target/release/examples/quality_probe',f'data/{graph}.mtx','200',str(seed),'no-cpu','1']
  print(tag,flush=True)
  with path.open('w') as f, (out/(tag+'.stderr')).open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
  a=json.loads(path.read_text());b=json.loads((old/f'optimized-{graph}-s{seed}.jsonl').read_text().splitlines()[0])
  for key in ['initial_sha256','preprocess_sha256','final_sha256','stress']:assert a[key]==b[key],(tag,key)
  results.append({'graph':graph,'seed':seed,'exact_match':True,'before_s':b['iteration_seconds'],'after_s':a['iteration_seconds']})
(out/'additional-regression.json').write_text(json.dumps(results,indent=2)+'\n')
# Full: odd tiles, partial final blocks, shared/global coordinates and W<T.
import hashlib
full=[]
for graph,tile,workgroup,memory in [('USpowerGrid',127,64,'shared'),('EVA',32,32,'global'),('1138_bus',1024,256,'shared')]:
 for seed in [0,3]:
  records=[]
  for label,binary in [('before',str(pathlib.Path(sys.argv[2])/'rr-gpu')),('after','implementations/full/rr_gpu/target/release/rr-gpu')]:
   tag=f'regression-full-{graph}-s{seed}-{label}'
   cmd=[binary,'--input',f'data/{graph}.mtx','--seed',str(seed),'--tile-size',str(tile),'--workgroup-size',str(workgroup),'--coordinate-memory',memory,'--output-format','json','--output-dir',str(out/'artifacts'),'--run-id',tag]
   print(tag,flush=True)
   with (out/(tag+'.jsonl')).open('w') as f,(out/(tag+'.stderr')).open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
   r=json.loads((out/(tag+'.jsonl')).read_text());r['coordinate_sha256']=hashlib.sha256(pathlib.Path(r['final_positions_path']).read_bytes().split(b'# Positions',1)[1]).hexdigest();records.append(r)
  for key in ['coordinate_sha256','initial_positions_sha256','stress_value']:assert records[0][key]==records[1][key],(graph,seed,key)
  full.append({'graph':graph,'seed':seed,'tile':tile,'workgroup':workgroup,'memory':memory,'exact_match':True})
(out/'full-additional-regression.json').write_text(json.dumps(full,indent=2)+'\n')
# Default P=200 at one million vertices: compare with the previous safe CLI run.
tag='million-p200-regression'
cmd=['implementations/sparse/sparse-sgd-gpu/target/release/sparse-sgd-gpu','--input','data/delaunay_n20.mtx','--pivots','200','--seed','0','--output-format','json','--output-dir',str(out/'artifacts'),'--run-id',tag]
print(tag,flush=True)
with (out/(tag+'.json')).open('w') as f,(out/(tag+'.stderr')).open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
a=json.loads((out/(tag+'.json')).read_text());b=json.loads((old/'million-h200.json').read_text())
for key in ['initial_positions_sha256','preprocess_sha256','stress_value']:assert a[key]==b[key],(tag,key)
def coords(r):return hashlib.sha256(pathlib.Path(r['final_positions_path']).read_bytes().split(b'# Positions',1)[1]).hexdigest()
assert coords(a)==coords(b),tag
(out/'million-p200-exact.json').write_text(json.dumps({'nodes':a['nodes'],'pivots':200,'exact_match':True,'final_coordinate_sha256':coords(a),'iteration_time_ms':a['iteration_time_ms']},indent=2)+'\n')
