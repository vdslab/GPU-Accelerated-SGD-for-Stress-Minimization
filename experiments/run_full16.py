"""Repeat the manuscript Full RR studies with resumable, bounded measurements."""
import collections, csv, importlib.util, json, pathlib, shutil, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=ROOT/'experiment_results/full16'
LEGACY=pathlib.Path('/Users/kose/master/atomic-scheduling-investigation')
spec=importlib.util.spec_from_file_location('frozen_helper',BASE/'config/runner_helper.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
helper.ROOT=ROOT
binary=ROOT/'target/full16-release/release/rr-gpu'
helper.METHODS['rr_sgd']=str(binary)
graphs=json.loads((LEGACY/'experiment_results/v3/config/datasets.json').read_text())
for g in graphs:
 if g.get('path'):g['path']=str(LEGACY/g['path'])
graphmap={g['name']:g for g in graphs}
jobs=[j for j in helper.plan(graphs) if j['method']=='rr_sgd' and j['study'] in ('basic','convergence')]
for j in jobs:
 j['run_id']=j['run_id'].replace('v3-','full16-',1)
 if j.get('reuse'):j['reuse']=j['reuse'].replace('v3-','full16-',1)
limits=json.loads((LEGACY/'experiment_results/v3/config/limits.json').read_text())
frozen={'binary_sha256':helper.sha(binary),'runner_sha256':helper.sha(__file__),'helper_sha256':helper.sha(BASE/'config/runner_helper.py'),'jobs':jobs,'limits':limits}
p=BASE/'config/execution-speed-stress.json'
if p.exists():assert json.loads(p.read_text())==frozen,'Measurement inputs changed'
else:
 helper.save(p,frozen);helper.save(BASE/'config/plan.json',jobs);helper.save(BASE/'config/datasets.json',graphs)
 for path in (ROOT/'implementations/full/rr_gpu/src').glob('*'):
  dst=BASE/'config/source-snapshot'/path.name;dst.parent.mkdir(exist_ok=True);shutil.copy2(path,dst)
resultfile=BASE/'raw/results.jsonl';resultfile.parent.mkdir(parents=True,exist_ok=True)
prior={r['run_id']:r for r in map(json.loads,resultfile.read_text().splitlines())} if resultfile.exists() else {}
old=list(csv.DictReader((ROOT/'experiment_results/v4/reference/full-v3/runs.csv').open()))
estimates={(r['study'],r['graph'],r['tile_size']):float(r.get('process_wall_seconds') or 10) for r in old if r['method']=='rr_sgd'}
for job in jobs:
 if job['run_id'] in prior:continue
 assert helper.sha(binary)==frozen['binary_sha256']
 if job.get('reuse') in prior:
  row=dict(prior[job['reuse']]);row.update(job);row['reused_from']=job['reuse']
 else:
  key=f"warmup-{job['graph']}-t{job['tile_size']}"
  warmup=helper.execute(BASE,dict(job,run_id=key,seed=0,study='warmup'),graphmap[job['graph']],limits,True)
  if warmup['status']!='success':row=dict(job,status='warmup_failed',failure_reason=warmup['status'],warmup_run_id=key,executed=False)
  else:row=helper.execute(BASE,job,graphmap[job['graph']],limits)
 prior[job['run_id']]=row
 with resultfile.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
 remaining=[j for j in jobs if j['run_id'] not in prior and not j.get('reuse')]
 estimated=sum(estimates.get((j['study'],j['graph'],str(j['tile_size'])),15) for j in remaining)
 checkpoint=dict(completed=len(prior),planned=len(jobs),counts=dict(collections.Counter(r['status'] for r in prior.values())),last_run=job['run_id'],estimated_remaining_minutes_from_old_measurements=estimated/60)
 helper.save(BASE/'checkpoint.json',checkpoint)
 print(json.dumps(checkpoint),flush=True)
print('COMPLETE',flush=True)
