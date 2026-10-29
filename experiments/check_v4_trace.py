import subprocess,pathlib,json,os,hashlib,sys
sys.path.insert(0,'experiments')
from run_v4 import BINARIES
out=pathlib.Path('experiment_results/v4/validation/trace-smoke');out.mkdir(parents=True,exist_ok=True)
results=[]
for method,binary in BINARIES.items():
 runs=[]
 for trace in [False,True]:
  folder=(out/(method+('-trace' if trace else '-plain'))).resolve();folder.mkdir(exist_ok=True)
  env=dict(os.environ,SGD_STRESS_SAMPLES='256');env.pop('SGD_TRACE_DIR',None)
  if trace:env.update(SGD_TRACE_DIR=str(folder/'trace'),SGD_TRACE_CHECKPOINTS='0,1,2,3,5,8,10,15')
  cmd=[binary,'--input','data/USpowerGrid.mtx','--seed','0','--iterations','15','--pivots','200','--output-format','json','--output-dir',str(folder)]
  p=subprocess.run(cmd,env=env,capture_output=True,text=True,check=True);r=json.loads(p.stdout)
  assert r['status']=='success'
  layout=pathlib.Path(r['final_positions_path']).read_text().split('# Positions (x y)')[-1]
  runs.append((r,layout))
  (folder/'record.json').write_text(json.dumps(r,indent=2)+'\n')
  if trace:
   t=[json.loads(x) for x in (folder/'trace/trace.jsonl').read_text().splitlines()]
   assert [x['iteration'] for x in t if x['snapshot']]==[0,1,2,3,5,8,10,15]
 if method!='atomic_sparse_sgd':assert runs[0][1]==runs[1][1],method
 else:assert all(r['attempted_updates']==r['completed_updates'] and r['retry_failures']==0 for r,_ in runs)
 results.append(dict(method=method,trace_checkpoints=8,identical_final_coordinates=runs[0][1]==runs[1][1],atomic_note='lock order is nondeterministic' if method=='atomic_sparse_sgd' else None))
 print(method,'passed',flush=True)
(out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
