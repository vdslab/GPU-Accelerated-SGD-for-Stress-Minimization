import os,json,subprocess,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[3];binary=R/'tools/diagnostics/sparse-audit/target/release/order-probe'
for fence in [0,1]:
 name=f'full-fence{fence}';cmd=[str(binary),str(R/'data/smallworld.mtx'),str(O/name)]
 with (O/f'{name}.stderr').open('w') as err:
  p=subprocess.run(cmd,env=dict(os.environ,PROBE_FENCE=str(fence),PROBE_BATCH='20477886',PROBE_TRACE='0'),text=True,stdout=subprocess.PIPE,stderr=err);(O/f'{name}.json').write_text(p.stdout);p.check_returncode()
 print(name,p.stdout,flush=True)
cmd=[str(R/'tools/diagnostics/sparse-audit/target/release/sparse-audit'),'paired',str(R/'data/smallworld.mtx')]+[str(O/f'full-fence{fence}.positions.txt') for fence in [0,1]]
p=subprocess.run(cmd,text=True,capture_output=True,check=True);(O/'full-probe-evaluation.jsonl').write_text(p.stdout);print(p.stdout,flush=True)
