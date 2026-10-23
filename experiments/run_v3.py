#!/usr/bin/env python3
"""Frozen v3 seven-study plan; sequential resource-monitored runs and resumable evidence."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,re,signal,subprocess,sys,time,shutil
import psutil
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiment_results/v3'
METHODS={'sgd':'baseline-sgd-non-gpu','atomic_sgd':'vram-lock-native','rr_sgd':'rr-gpu','sparse_sgd':'baseline-sparse-sgd-non-gpu','atomic_sparse_sgd':'vram-lock-sparse','rr_sparse_sgd':'sparse-sgd-gpu'}
SMALL_REPS=['USpowerGrid','dwt_2680','EVA','fe_4elt2','bcsstk16','wiki-Vote']
LARGE_REPS=['luxembourg_osm','roadNet-CA','delaunay_n17','delaunay_n20','web-Stanford','web-NotreDame','coAuthorsCiteseer','coPapersDBLP','preferentialAttachment','smallworld','kron_g500-logn18','bmw7st_1']
PIVOT_REPS=['luxembourg_osm','delaunay_n20','web-Stanford','coAuthorsCiteseer','preferentialAttachment','bmw7st_1']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');q.replace(p)
def plan(rows):
 jobs=[]
 def add(study,g,m,seed,p=200,t=1024,reuse=None):
  name=g['name'];j=dict(study=study,graph=name,dataset=name,method=m,seed=seed,iterations=15,epsilon=.1,pivots=p if 'sparse' in m else None,tile_size=t if m=='rr_sgd' else None,workgroup_size=1024 if m=='rr_sgd' else None)
  j['run_id']=f"v3-{study}-{name}-{m}-s{seed}-p{j['pivots']}-t{j['tile_size']}";j['reuse']=reuse;jobs.append(j);return j['run_id']
 basic={}
 for g in sorted(rows,key=lambda g:(g['family']!='small',g.get('nodes') or g.get('source_nodes') or 999999999)):
  for seed in range(5):
   for m in list(METHODS) if g['family']=='small' else list(METHODS)[3:]:basic[g['name'],m,seed]=add('basic',g,m,seed)
 for g in rows:
  if g['name'] in SMALL_REPS+LARGE_REPS:
   for seed in range(10):
    for m in list(METHODS) if g['name'] in SMALL_REPS else list(METHODS)[3:]:add('convergence',g,m,seed)
  if g['name'] in SMALL_REPS:
   for t in [1,2,4,8,16,32,64,128,256,512,1024]:
    for seed in range(5):add('tiling',g,'rr_sgd',seed,t=t,reuse=basic[g['name'],'rr_sgd',seed] if t==1024 else None)
  if g['name'] in PIVOT_REPS:
   for p in [50,100,200,400]:
    for seed in range(5):
     for m in list(METHODS)[3:]:add('pivots',g,m,seed,p=p,reuse=basic[g['name'],m,seed] if p==200 else None)
 return jobs

def execute(base,j,g,limits,warmup=False):
 folder=base/('warmups' if warmup else 'raw/artifacts')/j['run_id'];folder.mkdir(parents=True,exist_ok=True);dest=folder/'record.json'
 if dest.exists():return json.loads(dest.read_text())
 row=dict(j,status='unavailable',nodes=g.get('nodes'),edges=g.get('edges'),artifact_dir=str(folder),source_status=g.get('status'),run_mode='benchmark')
 if g['status']!='ready' or not g.get('eligible',True):row['error']=g.get('reason',g.get('error','input unavailable'));save(dest,row);return row
 binary=ROOT/'target/v3-release/release'/METHODS[j['method']];cmd=[str(binary),'--input',g['path'],'--seed',str(j['seed']),'--iterations','15','--epsilon','0.1','--output-format','json','--run-id',j['run_id'],'--run-mode','benchmark','--output-dir',str(folder)]
 if j['pivots']:cmd+=['--pivots',str(j['pivots'])]
 if j['method']=='rr_sgd':
  cmd+=['--tile-size',str(j['tile_size']),'--workgroup-size','1024','--coordinate-memory','shared']
  if j['tile_size']==1:cmd+=['--no-inner-shuffle']
 trace=folder/'trace';env=dict(os.environ,SGD_STRESS_SAMPLES='256',SGD_TRACE_DIR=str(trace),SGD_TRACE_CHECKPOINTS='0,1,2,3,5,8,10,15' if j['study']=='convergence' else '')
 save(folder/'command.json',dict(argv=cmd,environment={k:env[k] for k in ['SGD_STRESS_SAMPLES','SGD_TRACE_DIR','SGD_TRACE_CHECKPOINTS']},limits=limits))
 row.update(binary_sha256=sha(binary),input_sha256=g['sha256'],trace_dir=str(trace));start=time.monotonic();peak=0;reason=None
 with (folder/'stdout.json').open('w') as out,(folder/'stderr.log').open('w') as err:
  proc=subprocess.Popen(['/usr/bin/time','-l',*cmd],stdout=out,stderr=err,env=env,cwd=ROOT,start_new_session=True)
  while proc.poll() is None:
   try:
    children=psutil.Process(proc.pid).children(recursive=True);rss=sum(p.memory_info().rss for p in children if p.is_running());peak=max(peak,rss)
   except psutil.Error:pass
   if time.monotonic()-start>limits['timeout_seconds']:reason='timeout'
   elif peak>limits['max_rss_bytes']:reason='memory_limit'
   elif shutil.disk_usage(base).free<limits.get('min_free_disk_bytes',8*1024**3):reason='disk_limit'
   if reason:
    try:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
    except (ProcessLookupError,subprocess.TimeoutExpired):
     try:os.killpg(proc.pid,signal.SIGKILL)
     except ProcessLookupError:pass
    proc.wait();break
   time.sleep(.2)
 wall=time.monotonic()-start;stderr=(folder/'stderr.log').read_text(errors='replace');match=re.search(r'(\d+)\s+maximum resident set size',stderr)
 if match:peak=max(peak,int(match[1]))
 row.update(status=reason or 'error',process_wall_seconds=wall,peak_rss_bytes=peak,returncode=proc.returncode)
 if not reason and proc.returncode==0:
  try:
   r=json.loads((folder/'stdout.json').read_text());assert r['status']=='success' and r['input_sha256']==g['sha256'] and r['seed']==j['seed']
   if 'atomic' in j['method']:assert r['attempted_updates']==r['completed_updates'] and r['retry_failures']==0
   row.update(r);row.update(j);row['status']='success';row['reported_iteration_time_ms']=r['iteration_time_ms']
   overhead=json.loads((trace/'overhead.json').read_text());row['iteration_time_ms']=overhead['cumulative_algorithm_seconds']*1000;row['trace_overhead_seconds']=overhead['trace_overhead_seconds']
   row['timing_definition']='sum of per-iteration algorithm intervals; trace IO/copy excluded; native record timing preserved in reported_iteration_time_ms'
  except Exception as e:row['status']='error';row['error']=repr(e)
 if row['status']!='success':row.setdefault('error',stderr[-2500:])
 save(dest,row);return row

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results-dir',type=Path,default=BASE);p.add_argument('--pilot',action='store_true');p.add_argument('--timeout-seconds',type=float,default=60);p.add_argument('--max-rss-gib',type=float,default=8);p.add_argument('--study',choices=['basic','convergence','tiling','pivots']);args=p.parse_args();base=args.results_dir.resolve();rows=json.loads((base/'config/datasets.json').read_text());graphs={g['name']:g for g in rows};jobs=plan(rows)
 if args.pilot:
  chosen=['USpowerGrid','fe_4elt2','luxembourg_osm','web-Stanford','delaunay_n20'];jobs=[j for j in jobs if j['study']=='basic' and j['seed']==0 and j['graph'] in chosen and (j['method'] in list(METHODS)[:3] if graphs[j['graph']]['family']=='small' else True)]
  base=base/'pilot';base.mkdir(exist_ok=True);limits=dict(timeout_seconds=args.timeout_seconds,max_rss_bytes=int(args.max_rss_gib*1024**3))
 else:
  limits=json.loads((base/'config/limits.json').read_text());save(base/'config/plan.json',jobs)
  if args.study:jobs=[j for j in jobs if j['study']==args.study]
 raw=base/'raw';raw.mkdir(parents=True,exist_ok=True)
 # Verify binaries once and freeze source version before any measurements.
 binary_hashes={m:sha(ROOT/'target/v3-release/release'/b) for m,b in METHODS.items()};provenance={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'binary_sha256':binary_hashes,'limits':limits}
 evidence=base/'config/execution.json'
 if evidence.exists():
  previous=json.loads(evidence.read_text());assert previous['binary_sha256']==binary_hashes and previous['limits']==limits,'Binary/limits changed; use a new experiment version'
 else:
  save(evidence,provenance)
  tracked=subprocess.check_output(['git','ls-files','experiments','implementations','shared'],cwd=ROOT,text=True).splitlines();source_hashes={}
  for name in tracked:
   source=ROOT/name
   if source.is_file():
    target=base/'config/source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);source_hashes[name]=sha(source)
  save(base/'config/source-hashes.json',source_hashes)
 for m,h in binary_hashes.items():assert h
 prior={}
 resultsfile=raw/'results.jsonl'
 if resultsfile.exists():prior={r['run_id']:r for r in map(json.loads,resultsfile.read_text().splitlines())}
 for index,j in enumerate(jobs,1):
  if j['run_id'] in prior:continue
  if j.get('reuse') and j['reuse'] in prior:
   r=dict(prior[j['reuse']]);r.update(j);r['reused_from']=j['reuse']
  else:
   if not args.pilot and graphs[j['graph']]['status']=='ready':
    key=f"warmup-{j['graph']}-{j['method']}-p{j['pivots']}-t{j['tile_size']}";wj=dict(j,run_id=key,seed=0,study='warmup');execute(base,wj,graphs[j['graph']],limits,True)
   r=execute(base,j,graphs[j['graph']],limits)
  prior[j['run_id']]=r
  with resultsfile.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
  counts={s:sum(x['status']==s for x in prior.values()) for s in {x['status'] for x in prior.values()}}
  save(base/'checkpoint.json',dict(completed=len(prior),planned=len(jobs),counts=counts,last_run=j['run_id']))
  print(index,'/',len(jobs),j['run_id'],r['status'],round(r.get('process_wall_seconds',0),2),counts,flush=True)
 print('Completed plan',len(prior),flush=True)
if __name__=='__main__':main()
