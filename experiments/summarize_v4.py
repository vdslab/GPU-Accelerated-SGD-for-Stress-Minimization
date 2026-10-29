#!/usr/bin/env python3
"""Audit and summarize v4 Sparse; inherited Full rows are separately labeled."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,math,statistics
from collections import defaultdict,Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiment_results/v4'
METHODS=['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd']
def read_jsonl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def dump_csv(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def sample_indices(n,count,seed=0):
 values=np.arange(n,dtype=np.int64);state=seed;mask=(1<<64)-1
 for i in range(min(n,count)):
  state=(state+0x9e3779b97f4a7c15)&mask;state=((state^(state>>30))*0xbf58476d1ce4e5b9)&mask;state=((state^(state>>27))*0x94d049bb133111eb)&mask;state^=state>>31
  j=i+state%(n-i);values[i],values[j]=values[j],values[i]
 return values[:min(n,count)]
def checkpoint_stress(xy,dist,sources):
 total=0.;n=len(xy)
 for start in range(0,len(sources),8):
  ss=sources[start:start+8];dd=np.asarray(dist[start:start+8],dtype=np.float64)
  dx=xy[ss,0,None]-xy[None,:,0];dy=xy[ss,1,None]-xy[None,:,1]
  actual=np.sqrt(dx*dx+dy*dy);valid=dd>0
  np.subtract(actual,dd,out=actual);np.divide(actual,dd,out=actual,where=valid);actual[~valid]=0
  total+=float(np.sum(actual*actual))
 return total*n/(2*len(sources))
def convergence(base,datasets,records):
 from scipy.io import mmread
 from scipy.sparse.csgraph import shortest_path
 dest=base/'tables/convergence.csv'
 previous=list(csv.DictReader(dest.open())) if dest.exists() else []
 cache={(r.get('input_sha256'),r['snapshot_sha256']):r for r in previous};out=[]
 for g in datasets:
  group=[r for r in records if r['study']=='convergence' and r['graph']==g['name'] and r['status']=='success']
  if not group:continue
  n=g['nodes'];sources=sample_indices(n,256);dist=None
  for r in group:
   for entry in read_jsonl(Path(r['trace_dir'])/'trace.jsonl'):
    if not entry['snapshot']:continue
    p=Path(r['trace_dir'])/(entry['snapshot']+'.gz');digest=hashlib.sha256(p.read_bytes()).hexdigest()
    key=(g['sha256'],digest)
    if key in cache:
     row=cache[key];row={k:float(v) if k in ['stress','elapsed_ms'] else int(v) if k in ['epoch','seed','stress_samples','stress_seed'] else v for k,v in row.items()}
     # Equal coordinates can occur in separate runs; retain each run's identity/timing.
     row.update(graph=r['graph'],method=r['method'],seed=r['seed'],epoch=entry['iteration'],elapsed_ms=entry['cumulative_algorithm_seconds']*1000,snapshot=str(p))
    else:
     if dist is None:
      matrix=mmread(g['path']).tocsr();matrix=matrix.maximum(matrix.T)
      cachepath=base/'cache'/f"{g['name']}-{g['sha256'][:12]}-256.u32";cachepath.parent.mkdir(exist_ok=True)
      complete=cachepath.with_suffix('.complete')
      if not complete.exists() or complete.read_text()!=g['sha256']:
       d=np.memmap(cachepath,dtype='<u4',mode='w+',shape=(256,n))
       for start in range(0,256,8):
        distances=shortest_path(matrix,directed=False,unweighted=True,indices=sources[start:start+8]);assert np.isfinite(distances).all();d[start:start+8]=distances.astype(np.uint32)
       d.flush();del d;complete.write_text(g['sha256'])
      dist=np.memmap(cachepath,dtype='<u4',mode='r',shape=(256,n))
     with gzip.open(p,'rb') as f:xy=np.frombuffer(f.read(),dtype='<f8').reshape(-1,2)
     assert len(xy)==n and np.isfinite(xy).all()
     row=dict(graph=r['graph'],family='sparse',method=r['method'],seed=r['seed'],epoch=entry['iteration'],elapsed_ms=entry['cumulative_algorithm_seconds']*1000,stress=checkpoint_stress(xy,dist,sources),stress_kind='sampled',stress_samples=256,stress_seed=0,snapshot=str(p),snapshot_sha256=digest,input_sha256=g['sha256'])
    if row['epoch']==15:assert math.isclose(row['stress'],r['stress_value'],rel_tol=1e-10),(r['run_id'],row['stress'],r['stress_value'])
    cache[key]=row
    out.append(row);dump_csv(dest,out)
   print('Evaluated convergence',r['run_id'],flush=True)
  del dist
 return out

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results-dir',type=Path,default=BASE);ap.add_argument('--allow-partial',action='store_true');ap.add_argument('--skip-convergence',action='store_true');a=ap.parse_args();base=a.results_dir
 plan=json.loads((base/'config/plan.json').read_text());datasets=json.loads((base/'config/datasets.json').read_text());records=read_jsonl(base/'raw/results.jsonl');byid={r['run_id']:r for r in records}
 assert len(byid)==len(records)
 if not a.allow_partial:assert set(byid)=={j['run_id'] for j in plan},'Measurements incomplete'
 for r in records:r['family']='sparse';r['experiment_version']='v4'
 (base/'tables').mkdir(exist_ok=True);dump_csv(base/'tables/runs.csv',records)
 failures=[dict(run_id=j['run_id'],graph=j['graph'],method=j['method'],seed=j['seed'],study=j['study'],status=byid.get(j['run_id'],{}).get('status','pending'),reason=byid.get(j['run_id'],{}).get('failure_reason',byid.get(j['run_id'],{}).get('error',''))) for j in plan if byid.get(j['run_id'],{}).get('status')!='success'];dump_csv(base/'tables/missing.csv',failures)
 successes={(r['study'],r['graph'],r['method'],r['seed']):r for r in records if r['status']=='success'}
 pairs=[]
 for r in successes.values():
  if r['method']=='sparse_sgd':continue
  cpu=successes.get((r['study'],r['graph'],'sparse_sgd',r['seed']))
  if not cpu:continue
  for k in ['initial_positions_sha256','preprocess_sha256','input_sha256','iterations','epsilon','pivots','stress_kind','stress_samples','stress_seed']:assert r[k]==cpu[k],(r['run_id'],k)
  pairs.append(dict(study=r['study'],graph=r['graph'],family='sparse',method=r['method'],seed=r['seed'],speedup=cpu['iteration_time_ms']/r['iteration_time_ms'],stress_ratio=r['stress_value']/cpu['stress_value'],cpu_iteration_ms=cpu['iteration_time_ms'],gpu_iteration_ms=r['iteration_time_ms'],cpu_stress=cpu['stress_value'],gpu_stress=r['stress_value'],algorithm_speedup=cpu['algorithm_time_cold_ms']/r['algorithm_time_cold_ms'],process_speedup=cpu['process_wall_seconds']/r['process_wall_seconds']))
 dump_csv(base/'tables/paired-ratios.csv',pairs)
 summaries=[]
 for g in datasets:
  for m in METHODS:
   rs=[r for r in records if r['graph']==g['name'] and r['study']=='basic' and r['method']==m];ok=[r for r in rs if r['status']=='success'];pp=[r for r in pairs if r['graph']==g['name'] and r['study']=='basic' and r['method']==m]
   s=dict(graph=g['name'],method=m,planned=5,success=len(ok),paired_seeds=len(pp),nodes=g['nodes'])
   for k in ['iteration_time_ms','algorithm_time_cold_ms','process_wall_seconds','stress_value','peak_rss_bytes']:
    values=[r[k] for r in ok]
    if values:s.update({k+'_median':statistics.median(values),k+'_min':min(values),k+'_max':max(values)})
   for k in ['speedup','stress_ratio','algorithm_speedup','process_speedup']:
    values=[r[k] for r in pp]
    if values:s.update({k+'_median':statistics.median(values),k+'_min':min(values),k+'_max':max(values)})
   summaries.append(s)
 dump_csv(base/'tables/summary.csv',summaries)
 times=[]
 for g in datasets:
  pp=[r for r in pairs if r['study']=='basic' and r['graph']==g['name'] and r['method']=='rr_sparse_sgd']
  if not pp:continue
  for m,k in [('CPU','cpu_iteration_ms'),('RR','gpu_iteration_ms')]:
   values=[r[k]/15 for r in pp];times.append(dict(family='sparse',graph=g['name'],method=m,paired_seeds=len(pp),iterations_per_run=15,median_ms_per_iteration=statistics.median(values),min_ms_per_iteration=min(values),max_ms_per_iteration=max(values),experiment_version='v4'))
 dump_csv(base/'tables/fig05_iteration_times.csv',times)
 conv=[] if a.skip_convergence else convergence(base,datasets,records)
 if a.skip_convergence and (base/'tables/convergence.csv').exists():conv=list(csv.DictReader((base/'tables/convergence.csv').open()))
 arrivals=[]
 for r in records:
  if r['study']!='convergence' or r['status']!='success':continue
  cpu=successes.get(('convergence',r['graph'],'sparse_sgd',r['seed']))
  if not cpu:continue
  points=[p for p in conv if p['graph']==r['graph'] and p['method']==r['method'] and int(p['seed'])==r['seed']];good=[p for p in points if float(p['stress'])<=1.05*cpu['stress_value']]
  if not a.skip_convergence:assert {int(p['epoch']) for p in points}=={0,1,2,3,5,8,10,15},r['run_id']
  arrivals.append(dict(graph=r['graph'],method=r['method'],seed=r['seed'],observed_points=len(points),reached=bool(good),first_observed_epoch=min((int(p['epoch']) for p in good),default=None),first_observed_ms=min((float(p['elapsed_ms']) for p in good),default=None),final_within_target=r['stress_value']<=1.05*cpu['stress_value']))
 dump_csv(base/'tables/arrivals.csv',arrivals)
 summary=dict(convergence_evaluated=not a.skip_convergence,planned=len(plan),completed=len(records),statuses=dict(Counter(r['status'] for r in records)),basic_graphs=38,convergence_graphs=3,rr_comparable=sum(s['paired_seeds']>0 for s in summaries if s['method']=='rr_sparse_sgd'),rr_faster=sum(s.get('speedup_median',0)>1 for s in summaries if s['method']=='rr_sparse_sgd'),rr_quality_within_5pct=sum(s.get('stress_ratio_median',math.inf)<=1.05 for s in summaries if s['method']=='rr_sparse_sgd'))
 (base/'tables/audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
if __name__=='__main__':main()
