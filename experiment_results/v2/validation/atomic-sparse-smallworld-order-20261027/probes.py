import os,json,subprocess,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[3];binary=R/'tools/diagnostics/sparse-audit/target/release/order-probe'
configs=[('fenced-trace',1,20477886,1),('before-batch65536',0,65536,0),('fenced-batch65536',1,65536,0),('before-batch16384',0,16384,0),('fenced-batch16384',1,16384,0)]
(O/'probe-metadata.json').write_text(json.dumps({'configs':configs,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()},indent=2))
for name,fence,batch,trace in configs:
 cmd=[str(binary),str(R/'data/smallworld.mtx'),str(O/name)]
 env=dict(os.environ,PROBE_FENCE=str(fence),PROBE_BATCH=str(batch),PROBE_TRACE=str(trace))
 with (O/f'{name}.stderr').open('w') as err:
  p=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=err);(O/f'{name}.json').write_text(p.stdout);p.check_returncode()
 print(name,p.stdout,flush=True)
paths=[str(O/f'{name}.positions.txt') for name,_,_,_ in configs]
cmd=[str(R/'tools/diagnostics/sparse-audit/target/release/sparse-audit'),'paired',str(R/'data/smallworld.mtx')]+paths
p=subprocess.run(cmd,text=True,capture_output=True,check=True);(O/'probe-evaluation.jsonl').write_text(p.stdout);print(p.stdout,flush=True)
