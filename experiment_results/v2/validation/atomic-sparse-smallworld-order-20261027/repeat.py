import json,subprocess,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent; R=O.parents[3]; prev=O.parent/'atomic-sparse-fenced-20261027'
records=[]
for rep in range(3):
 for method in (['before','fenced'] if rep%2==0 else ['fenced','before']):
  binary=prev/'binaries/vram-lock-sparse-before' if method=='before' else R/'implementations/sparse/vram-lock-sparse/target/release/vram-lock-sparse'
  rid=f'smallworld-s0-{method}-repeat{rep}'
  cmd=[str(binary),'--input',str(R/'data/smallworld.mtx'),'--pivots','200','--iterations','15','--epsilon','0.1','--seed','0','--run-id',rid,'--output-format','json','--output-dir',str(O/'artifacts'),'--cpu-shuffle']
  p=subprocess.run(cmd,text=True,capture_output=True,check=True);d=json.loads(p.stdout);d.update(command=cmd,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest());(O/f'{rid}.json').write_text(json.dumps(d,indent=2));records.append(d);print(rid,d['stress_value'],flush=True)
assert all(len({d[k] for d in records})==1 for k in ['preprocess_sha256','initial_positions_sha256','input_sha256'])
cmd=[str(R/'tools/diagnostics/sparse-audit/target/release/sparse-audit'),'paired',str(R/'data/smallworld.mtx')]+[d['final_positions_path'] for d in records]
p=subprocess.run(cmd,text=True,capture_output=True,check=True);(O/'repeat-evaluation.jsonl').write_text(p.stdout);print(p.stdout,flush=True)
