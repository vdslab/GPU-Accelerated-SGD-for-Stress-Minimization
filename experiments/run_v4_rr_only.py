#!/usr/bin/env python3
"""Continue only RR jobs after the user's scope reduction; retain earlier records."""
from pathlib import Path
import collections,datetime,hashlib,importlib.util,json,subprocess
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiment_results/v4'
spec=importlib.util.spec_from_file_location('v4_original',ROOT/'experiments/run_v4.py');original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
def main():
 jobs=json.loads((BASE/'config/rr-only-plan.json').read_text());graphs={g['name']:g for g in json.loads((BASE/'config/datasets.json').read_text())};limits=json.loads((BASE/'config/limits.json').read_text());frozen=json.loads((BASE/'config/execution.json').read_text())
 assert original.sha(ROOT/'experiments/run_v4.py')==frozen['runner_sha256']
 evidence=dict(started=datetime.datetime.now().astimezone().isoformat(),policy='RR-only continuation requested by user; earlier CPU/Atomic v4 records retained as excluded auxiliary measurements. Comparisons use labeled v3 CPU/Atomic references.',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),driver_sha256=original.sha(__file__),plan_sha256=original.sha(BASE/'config/rr-only-plan.json'),binary_sha256=frozen['binary_sha256'])
 p=BASE/'config/rr-only-execution.json'
 if p.exists():
  previous=json.loads(p.read_text())
  for k in ['driver_sha256','plan_sha256','binary_sha256']:assert previous[k]==evidence[k],k
 else:original.save(p,evidence)
 results=BASE/'raw/results.jsonl';prior={r['run_id']:r for r in map(json.loads,results.read_text().splitlines())};rr={k:r for k,r in prior.items() if r['method']=='rr_sparse_sgd'}
 for j in jobs:
  if j['run_id'] in rr:continue
  for method,binary in original.BINARIES.items():assert original.sha(ROOT/binary)==frozen['binary_sha256'][method],method
  key=f"warmup-{j['graph']}-rr_sparse_sgd-p200";w=original.execute(BASE,dict(j,run_id=key,seed=0,study='warmup'),graphs[j['graph']],limits,True)
  if w['status']!='success':r=dict(j,status='warmup_failed',failure_reason=w['status'],warmup_run_id=key,executed=False)
  else:r=original.execute(BASE,j,graphs[j['graph']],limits)
  rr[j['run_id']]=r
  with results.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
  counts=dict(collections.Counter(r['status'] for r in rr.values()));original.save(BASE/'checkpoint.json',dict(completed=len(rr),planned=len(jobs),counts=counts,last_run=j['run_id'],scope='RR only',earlier_auxiliary_records=len(prior)-sum(r['method']=='rr_sparse_sgd' for r in prior.values())))
  print(len(rr),'/',len(jobs),j['run_id'],r['status'],round(r.get('process_wall_seconds',0),2),counts,flush=True)
 print('Completed RR-only plan',len(rr),flush=True)
if __name__=='__main__':main()
