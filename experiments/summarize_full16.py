"""Reconcile Full chunk experiments with archived baselines; verify all paired inputs."""
from pathlib import Path
import collections,csv,gzip,hashlib,importlib.util,json,math,shutil,statistics,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiment_results/full16';OLD=ROOT/'experiment_results/v4';PAPER=BASE/'paper-data'
sys.path.insert(0,str(Path('/Users/kose/master/atomic-scheduling-investigation/experiments')))
import summarize_v3 as v3
from summarize_v4 import checkpoint_stress, sample_indices, dump_csv
for p in [BASE/'tables',BASE/'figures',BASE/'cache',PAPER/'reference/full-v3',PAPER/'tables',PAPER/'config']:p.mkdir(exist_ok=True,parents=True)
def readcsv(p):return list(csv.DictReader(p.open()))
def jsonl(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def typed(r):
 r=dict(r)
 for k in ['seed','iterations','epsilon','pivots','tile_size','workgroup_size','stress_samples','stress_seed','iteration_time_ms','stress_value','algorithm_time_cold_ms','algorithm_time_warm_ms','process_wall_seconds','common_preprocess_time_ms','method_setup_time_ms','peak_rss_bytes']:
  if k in r:r[k]=json.loads(r[k]) if r[k] else None
 for k,v in r.items():
  if v=='':r[k]=None
 return r
while True:
 c=json.loads((BASE/'checkpoint.json').read_text())
 d=json.loads((BASE/'degree/checkpoint.json').read_text()) if (BASE/'degree/checkpoint.json').exists() else {}
 if c['completed']==160 and d.get('completed')==90:break
 time.sleep(2)
new=jsonl(BASE/'raw/results.jsonl');assert len(new)==160
old=[typed(r) for r in readcsv(OLD/'reference/full-v3/runs.csv') if r['method'] in ['sgd','atomic_sgd'] and r['study'] in ['basic','convergence']]
rows=old+new
pairs,excluded=v3.pair_rows(rows);bad=[r for r in excluded if 'Mismatch' in r['reason']];assert not bad,bad
summary=v3.aggregate(rows)
dump_csv(BASE/'tables/runs.csv',rows);dump_csv(BASE/'tables/summary.csv',summary);dump_csv(BASE/'tables/paired-ratios.csv',pairs);dump_csv(BASE/'tables/excluded.csv',excluded)
for f in ['runs.csv','summary.csv']:shutil.copy2(BASE/'tables'/f,PAPER/'reference/full-v3'/f)
for name in ['runs.csv','summary.csv','paired-ratios.csv','convergence.csv','audit.json']:
 shutil.copy2(OLD/'tables'/name,PAPER/'tables'/name)
for name in ['sparse-v3']:
 dst=PAPER/'reference'/name
 if not dst.exists():dst.symlink_to(OLD/'reference'/name,target_is_directory=True)
# Evaluate each saved Full checkpoint with the same full-stress evaluator.
from scipy.io import mmread
from scipy.sparse.csgraph import shortest_path
datasets=json.loads((BASE/'config/datasets.json').read_text());conv=[];stress_audit=[]
cachefile=BASE/'tables/convergence-new.csv'
previous=readcsv(cachefile) if cachefile.exists() else []
cache={(r['input_sha256'],r['snapshot_sha256']):r for r in previous}
for g in datasets:
 group=[r for r in new if r['study']=='convergence' and r['graph']==g['name'] and r['status']=='success']
 if not group:continue
 n=g['nodes'];count=n if n<=8000 else 256;sources=np.arange(n) if n<=8000 else sample_indices(n,count);dist=None
 for r in group:
  entries=jsonl(Path(r['trace_dir'])/'trace.jsonl');assert {e['iteration'] for e in entries if e.get('snapshot')}=={0,1,2,3,5,8,10,15}
  for e in entries:
   if not e.get('snapshot'):continue
   p=Path(r['trace_dir'])/(e['snapshot']+'.gz');digest=hashlib.sha256(p.read_bytes()).hexdigest();key=(g['sha256'],digest)
   if key in cache and e['iteration']!=15:stress=float(cache[key]['stress'])
   else:
    if dist is None:
     signature=hashlib.sha256((g['sha256']+f':{n}:{count}:uint32:v1').encode()).hexdigest()[:20]
     path=BASE/'cache'/f'distance-{signature}.u32';complete=path.with_suffix('.complete')
     prior=Path('/Users/kose/master/atomic-scheduling-investigation/experiment_results/v3/cache')/path.name
     if not complete.exists() and prior.exists() and prior.with_suffix('.complete').exists():
      if not path.exists():path.symlink_to(prior)
      complete.write_text(signature)
     if not complete.exists():
      a=mmread(g['path']).tocsr();a=a.maximum(a.T);d=np.memmap(path,dtype='<u4',mode='w+',shape=(count,n))
      for start in range(0,count,16):
       dd=shortest_path(a,directed=False,unweighted=True,indices=sources[start:start+16]);assert np.isfinite(dd).all();d[start:start+16]=dd.astype(np.uint32)
      d.flush();del d;complete.write_text(signature)
     dist=np.memmap(path,dtype='<u4',mode='r',shape=(count,n))
    with gzip.open(p,'rb') as f:xy=np.frombuffer(f.read(),dtype='<f8').reshape(-1,2)
    assert len(xy)==n and np.isfinite(xy).all();stress=checkpoint_stress(xy,dist,sources)
   row=dict(graph=g['name'],family='full',method='rr_sgd',seed=r['seed'],epoch=e['iteration'],elapsed_ms=e['cumulative_algorithm_seconds']*1000,stress=stress,stress_kind=r['stress_kind'],stress_samples=count,stress_seed=0,snapshot=str(p),snapshot_sha256=digest,input_sha256=g['sha256'])
   if e['iteration']==15:
    # Saved checkpoints precede the native final f32 centering operation.
    f32=xy.astype(np.float32)
    means=np.array([np.cumsum(f32[:,k],dtype=np.float32)[-1]/np.float32(n) for k in range(2)])
    centered=(f32-means).astype(np.float64);native_check=checkpoint_stress(centered,dist,sources)
    assert math.isclose(native_check,r['stress_value'],rel_tol=1e-9),(r['run_id'],native_check,r['stress_value'])
    stress_audit.append(dict(run_id=r['run_id'],checkpoint_stress=stress,native_stress=r['stress_value'],recentered_stress=native_check,relative_precenter_difference=abs(stress-r['stress_value'])/r['stress_value'],relative_recentered_difference=abs(native_check-r['stress_value'])/r['stress_value']))
   conv.append(row);cache[key]=row
  dump_csv(cachefile,conv);print('Evaluated',r['run_id'],flush=True)
 del dist
dump_csv(BASE/'tables/convergence-stress-audit.csv',stress_audit)
oldconv=[{**r,'seed':int(r['seed']),'epoch':int(r['epoch']),'elapsed_ms':float(r['elapsed_ms']),'stress':float(r['stress'])} for r in readcsv(OLD/'reference/full-v3/convergence.csv') if r['method'] in ['sgd','atomic_sgd']]
conv=oldconv+conv;dump_csv(PAPER/'reference/full-v3/convergence.csv',conv)
arrivals=[]
for graph in sorted({r['graph'] for r in conv}):
 for seed in range(10):
  cpu=next(r for r in conv if r['graph']==graph and r['method']=='sgd' and r['seed']==seed and r['epoch']==15)
  for m in ['sgd','atomic_sgd','rr_sgd']:
   points=[r for r in conv if r['graph']==graph and r['method']==m and r['seed']==seed];assert len(points)==8
   good=[r for r in points if r['stress']<=1.05*cpu['stress']];last=next(r for r in points if r['epoch']==15)
   arrivals.append(dict(graph=graph,method=m,seed=seed,reached=bool(good),end=last['stress']<=1.05*cpu['stress'],first_epoch=min((r['epoch'] for r in good),default=None),first_seconds=min((r['elapsed_ms']/1000 for r in good),default=None)))
dump_csv(BASE/'tables/arrivals.csv',arrivals)
# Supplemental same-period CPU and unsplit-versus-split comparisons.
degree=jsonl(BASE/'degree/results.jsonl');assert len(degree)==90 and all(r['status']=='success' for r in degree),collections.Counter(r['status'] for r in degree)
dp=[]
for r in degree:
 if r['method']=='sgd':continue
 ref=next(x for x in degree if x['graph']==r['graph'] and x['seed']==r['seed'] and x['method']=='sgd')
 for k in ['input_sha256','initial_positions_sha256','iterations','epsilon','stress_kind','stress_samples','stress_seed']:assert r[k]==ref[k],(r['run_id'],k)
 dp.append(dict(graph=r['graph'],method=r['method'],seed=r['seed'],speedup=ref['iteration_time_ms']/r['iteration_time_ms'],stress_ratio=r['stress_value']/ref['stress_value'],iteration_ms=r['iteration_time_ms']/15,cpu_iteration_ms=ref['iteration_time_ms']/15))
dump_csv(BASE/'tables/degree-pairs.csv',dp)
def med(vals):return statistics.median(vals)
gs=[]
for graph in sorted({r['graph'] for r in pairs if r['study']=='basic'}):
 for m in ['rr_sgd','atomic_sgd']:
  pp=[r for r in pairs if r['study']=='basic' and r['graph']==graph and r['method']==m]
  if pp:gs.append(dict(graph=graph,method=m,n=len(pp),speedup=med([r['speedup'] for r in pp]),stress_ratio=med([r['stress_ratio'] for r in pp])))
dump_csv(BASE/'tables/graph-pairs.csv',gs)
arrival_summary=[]
for graph in sorted({r['graph'] for r in arrivals}):
 rr=[r for r in arrivals if r['graph']==graph and r['method']=='rr_sgd'];cpu=[r for r in arrivals if r['graph']==graph and r['method']=='sgd']
 good=[r['first_seconds'] for r in rr if r['reached']]
 arrival_summary.append(dict(graph=graph,hit=sum(r['reached'] for r in rr),end=sum(r['end'] for r in rr),cpu_seconds=med([r['first_seconds'] for r in cpu]),rr_seconds=med(good) if good else None))
dump_csv(BASE/'tables/arrival-summary.csv',arrival_summary)
dsummary=[]
for g in json.loads((BASE/'degree/config/datasets.json').read_text()):
 row={k:g[k] for k in ['name','nodes','edges','max_degree','degree_cv']}
 for m,label in [('rr_sgd','split'),('rr_sgd_chunks1','unsplit')]:
  pp=[r for r in dp if r['graph']==g['name'] and r['method']==m]
  for k in ['speedup','stress_ratio','iteration_ms','cpu_iteration_ms']:row[label+'_'+k]=med([r[k] for r in pp])
  row[label+'_within_target']=sum(r['stress_ratio']<=1.05 for r in pp)
 rr={r['seed']:r for r in dp if r['graph']==g['name'] and r['method']=='rr_sgd'}
 row['split_unsplit_time_ratio']=med([rr[r['seed']]['iteration_ms']/r['iteration_ms'] for r in dp if r['graph']==g['name'] and r['method']=='rr_sgd_chunks1'])
 dsummary.append(row)
dump_csv(BASE/'tables/degree-summary.csv',dsummary)
rr=[r for r in gs if r['method']=='rr_sgd']
audit=dict(completed=160,planned=160,convergence_evaluated=True,core_statuses=dict(collections.Counter(r['status'] for r in new)),additional_runs=90,pair_mismatches=bad,comparable_graphs=len(rr),faster=sum(r['speedup']>1 for r in rr),quality_within_5pct=sum(r['stress_ratio']<=1.05 for r in rr),both=sum(r['speedup']>1 and r['stress_ratio']<=1.05 for r in rr),full_arrivals=arrival_summary,degree_summary=dsummary,full_graphs=rr)
save(BASE/'tables/audit.json',audit);save(PAPER/'full-audit.json',audit)
print(json.dumps(audit,indent=2),flush=True)
