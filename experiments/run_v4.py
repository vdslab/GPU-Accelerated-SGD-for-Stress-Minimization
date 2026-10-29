#!/usr/bin/env python3
"""v4 Sparse paper rerun: frozen 38 graphs, 5 basic seeds, 3 convergence graphs x 10 seeds."""
from pathlib import Path
import argparse,datetime,gzip,hashlib,json,os,re,signal,subprocess,sys,time,shutil,tempfile
import psutil
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiment_results/v4'
BINARIES={
 'sparse_sgd':'implementations/sparse/baseline-sparse-sgd-non-gpu/target/release/baseline-sparse-sgd-non-gpu',
 'atomic_sparse_sgd':'implementations/sparse/vram-lock-sparse/target/release/vram-lock-sparse',
 'rr_sparse_sgd':'implementations/sparse/sparse-sgd-gpu/target/release/sparse-sgd-gpu'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');q.replace(p)
def execute(base,j,g,limits,warmup=False):
 folder=base/('warmups' if warmup else 'raw/artifacts')/j['run_id'];folder.mkdir(parents=True,exist_ok=True);dest=folder/'record.json'
 if dest.exists():return json.loads(dest.read_text())
 row=dict(j,status='unavailable',nodes=g.get('nodes'),edges=g.get('edges'),artifact_dir=str(folder),source_status=g.get('status'),run_mode='benchmark')
 if g['status']!='ready' or not g.get('eligible',True):row['error']=g.get('reason',g.get('error','input unavailable'));save(dest,row);return row
 binary=ROOT/BINARIES[j['method']];cmd=[str(binary),'--input',g['path'],'--seed',str(j['seed']),'--iterations','15','--epsilon','0.1','--output-format','json','--run-id',j['run_id'],'--run-mode','benchmark','--output-dir',str(folder)]
 if j['pivots']:cmd+=['--pivots',str(j['pivots'])]
 if j['method']=='rr_sgd':
  cmd+=['--tile-size',str(j['tile_size']),'--workgroup-size','1024','--coordinate-memory','shared']
  if j['tile_size']==1:cmd+=['--no-inner-shuffle']
 trace=folder/'trace';env=dict(os.environ,SGD_STRESS_SAMPLES='256');env.pop('SGD_TRACE_DIR',None);env.pop('SGD_TRACE_CHECKPOINTS',None)
 if j['study']=='convergence' and not warmup:env.update(SGD_TRACE_DIR=str(trace),SGD_TRACE_CHECKPOINTS='0,1,2,3,5,8,10,15')
 save(folder/'command.json',dict(argv=cmd,environment={k:env[k] for k in ['SGD_STRESS_SAMPLES','SGD_TRACE_DIR','SGD_TRACE_CHECKPOINTS'] if k in env},limits=limits))
 row.update(binary_sha256=sha(binary),input_sha256=g['sha256'],trace_dir=str(trace));start=time.monotonic();peak=0;reason=None
 with (folder/'stdout.json').open('w') as out,(folder/'stderr.log').open('w') as err:
  proc=subprocess.Popen(['/usr/bin/time','-l',*cmd],stdout=out,stderr=err,env=env,cwd=ROOT,start_new_session=True)
  try:
   while proc.poll() is None:
    try:
     children=psutil.Process(proc.pid).children(recursive=True);rss=sum(p.memory_info().rss for p in children if p.is_running());peak=max(peak,rss)
    except psutil.Error:pass
    if time.monotonic()-start>limits['timeout_seconds']:reason='timeout'
    elif peak>limits['max_rss_bytes']:reason='memory_limit'
    elif shutil.disk_usage(base).free<limits.get('min_free_disk_bytes',8*1024**3):reason='disk_limit'
    if reason:
     stop_process_group(proc,folder/'termination-error.json');break
    time.sleep(.2)
  except BaseException:
   # A termination failure already has its own evidence; do not mask it with
   # a second attempt against a group that may have disappeared.
   if reason is None:stop_process_group(proc,folder/'termination-error.json')
   raise
 wall=time.monotonic()-start;stderr=(folder/'stderr.log').read_text(errors='replace');match=re.search(r'(\d+)\s+maximum resident set size',stderr)
 if match:peak=max(peak,int(match[1]))
 row.update(status=reason or 'error',process_wall_seconds=wall,peak_rss_bytes=peak,peak_rss_source='macOS time -l + polling' if match else 'process-tree RSS polling (observed peak)',returncode=proc.returncode)
 if not reason and proc.returncode==0:
  try:
   r=json.loads((folder/'stdout.json').read_text());assert r['status']=='success' and r['input_sha256']==g['sha256'] and r['seed']==j['seed']
   if 'atomic' in j['method']:assert r['attempted_updates']==r['completed_updates'] and r['retry_failures']==0
   row.update(r);row.update(j);row['status']='success';row['reported_iteration_time_ms']=r['iteration_time_ms']
   if j['study']=='convergence' and not warmup:
    overhead=json.loads((trace/'overhead.json').read_text());row['iteration_time_ms']=overhead['cumulative_algorithm_seconds']*1000;row['trace_overhead_seconds']=overhead['trace_overhead_seconds']
   row['timing_definition']='native SGD interval without checkpoint instrumentation' if j['study']!='convergence' or warmup else 'diagnostic per-iteration sum excluding snapshot copy and IO'
   if j['method']=='rr_sparse_sgd':assert r['rr_arithmetic']=='f32' and r['rr_phase_chunks']==4
   assert r['iterations']==15 and r['pivots']==200
   assert r['stress_kind']=='exact' or (r['stress_samples']==256 and r['stress_seed']==0)
  except Exception as e:row['status']='error';row['error']=repr(e)
 if row['status']!='success':row.setdefault('error',stderr[-2500:])
 archive_text_artifacts(folder,row)
 save(dest,row);return row


def archive_text_artifacts(folder,row):
 """Losslessly archive text layouts and binary checkpoints after measurement."""
 started=time.monotonic();manifest_path=folder/'artifact-archive.json'
 manifest=dict(format='gzip',compresslevel=1,measurement_excluded=True,files=[],errors=[])
 row['archive_manifest_path']=str(manifest_path)
 if manifest_path.exists():
  row.update(archive_seconds=time.monotonic()-started,archive_status='partial_error',
             archive_error=f'Archive manifest already exists; originals and existing evidence retained: {manifest_path}')
  return
 sources=sorted([*folder.glob('*.txt'),*(folder/'trace').glob('positions-*.f64le')])
 for source in sources:
  target=source.with_name(source.name+'.gz');temporary=None;created_target=False
  try:
   if source.is_symlink():raise ValueError(f'Refusing to archive symlink: {source}')
   if target.exists():raise FileExistsError(f'Archive already exists; original retained: {target}')
   before=source.stat();original_hash=hashlib.sha256();original_bytes=0
   with tempfile.NamedTemporaryFile(prefix=target.name+'.',suffix='.tmp',dir=folder,delete=False) as output:
    temporary=Path(output.name)
    with source.open('rb') as original,gzip.GzipFile(filename='',mode='wb',compresslevel=1,fileobj=output,mtime=0) as compressed:
     for chunk in iter(lambda:original.read(1024*1024),b''):
      original_hash.update(chunk);original_bytes+=len(chunk);compressed.write(chunk)
   # Reading through EOF validates gzip's CRC32 and uncompressed-size trailer.
   decoded_hash=hashlib.sha256();decoded_bytes=0
   with gzip.open(temporary,'rb') as decoded:
    for chunk in iter(lambda:decoded.read(1024*1024),b''):
     decoded_hash.update(chunk);decoded_bytes+=len(chunk)
   if decoded_bytes!=original_bytes or decoded_hash.digest()!=original_hash.digest():
    raise ValueError(f'Archive decompressed SHA256/length mismatch: {source}')
   after=source.stat()
   if original_bytes!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
    raise ValueError(f'Original changed while archiving; original retained: {source}')
   entry=dict(original_path=str(source),compressed_path=str(target),
              original_sha256=original_hash.hexdigest(),decompressed_sha256=decoded_hash.hexdigest(),
              compressed_sha256=sha(temporary),original_bytes=original_bytes,
              compressed_bytes=temporary.stat().st_size,gzip_crc_validated=True)
   # Atomic no-overwrite publication on the same filesystem.
   os.link(temporary,target);created_target=True;temporary.unlink();temporary=None
   manifest['files'].append(entry)
  except Exception as error:
   if temporary is not None:
    try:temporary.unlink(missing_ok=True)
    except OSError:pass
   if created_target:
    try:target.unlink(missing_ok=True)
    except OSError:pass
   manifest['errors'].append(dict(original_path=str(source),error=repr(error)))
 # Persist every verified archive and its hashes before removing any original.
 # A compression or manifest failure never changes the native measurement status.
 try:save(manifest_path,manifest)
 except Exception as error:
  manifest['errors'].append(dict(original_path=None,error=f'manifest write: {error!r}'))
 else:
  for entry in manifest['files']:
   source=Path(entry['original_path']);target=Path(entry['compressed_path'])
   try:
    redirects=[field for field in ['final_positions_path','initial_positions_path','vertex_map_path']
               if row.get(field) and Path(row[field]).resolve()==source.resolve()]
    source.unlink()
   except Exception as error:
    manifest['errors'].append(dict(original_path=str(source),error=repr(error)))
   else:
    for field in redirects:
     row['native_'+field]=row[field];row[field]=str(target)
 # No filesystem writes follow deletion: the verified manifest already exists.
 row['archive_seconds']=time.monotonic()-started
 row['archive_status']='partial_error' if manifest['errors'] else 'success'
 if manifest['errors']:row['archive_error']='; '.join(error['error'] for error in manifest['errors'])


def process_group_state(pgid):
 """Inspect actual processes; killpg(pgid, 0) can return EPERM after macOS exit."""
 members=[];inaccessible=[]
 for process in psutil.process_iter(['pid','name','status','create_time'],ad_value=None):
  try:
   if os.getpgid(process.pid)!=pgid:continue
   info=process.info
   # Zombies have stopped execution and cannot consume GPU/CPU resources.
   if info.get('status')!=psutil.STATUS_ZOMBIE:members.append(info)
  except (ProcessLookupError,psutil.NoSuchProcess):continue
  except (PermissionError,psutil.AccessDenied):inaccessible.append(process.pid)
 return dict(pgid=pgid,members=members,inaccessible_pids=inaccessible)

def stop_process_group(proc,incident_path=None):
 """Reap the wrapper and prove its process group has no live members."""
 def state():
  proc.poll()
  return process_group_state(proc.pid)
 def report(message,snapshot,error=None):
  evidence=dict(stage='process_group_termination',message=message,
                process_group=snapshot,wrapper_returncode=proc.returncode)
  if error is not None:evidence['error']=repr(error)
  if incident_path is not None:save(incident_path,evidence)
  print(json.dumps(evidence,ensure_ascii=False),file=sys.stderr,flush=True)
 def empty(snapshot):
  return not snapshot['members'] and not snapshot['inaccessible_pids']
 def send(sig):
  try:os.killpg(proc.pid,sig)
  except ProcessLookupError:pass
  except PermissionError as error:
   snapshot=state()
   # A disappearing macOS group can produce EPERM rather than ESRCH.
   # Ignore it only after an independent process scan establishes absence.
   if not empty(snapshot):
    report(f'Permission denied sending signal {sig} to an extant or unverified process group',snapshot,error)
    raise
 def wait_empty(seconds):
  deadline=time.monotonic()+seconds
  while True:
   snapshot=state()
   if empty(snapshot) or time.monotonic()>=deadline:return snapshot
   time.sleep(.05)
 snapshot=state()
 if not empty(snapshot):
  send(signal.SIGTERM)
  snapshot=wait_empty(5)
  if not empty(snapshot):
   send(signal.SIGKILL)
   snapshot=wait_empty(5)
  if not empty(snapshot):
   message='Process group still has live or inaccessible members after SIGKILL'
   report(message,snapshot)
   raise RuntimeError(message)
 proc.wait(timeout=5)


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--results-dir',type=Path,default=BASE)
 args=parser.parse_args();base=args.results_dir.resolve()
 jobs=json.loads((base/'config/plan.json').read_text());datasets=json.loads((base/'config/datasets.json').read_text());graphs={g['name']:g for g in datasets}
 limits=json.loads((base/'config/limits.json').read_text())
 evidence=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
   binary_sha256={m:sha(ROOT/b) for m,b in BINARIES.items()},runner_sha256=sha(__file__),
   plan_sha256=sha(base/'config/plan.json'),datasets_sha256=sha(base/'config/datasets.json'),limits=limits)
 path=base/'config/execution.json'
 if path.exists():
  before=json.loads(path.read_text())
  for field in ['binary_sha256','runner_sha256','plan_sha256','datasets_sha256','limits']:assert before[field]==evidence[field],f'{field} changed'
 else:
  for g in datasets:assert sha(g['path'])==g['sha256'],g['name']
  save(path,evidence)
  files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','implementations/sparse','shared','experiments'],cwd=ROOT,text=True).splitlines()
  hashes={}
  for name in sorted(set(files)):
   source=ROOT/name
   if source.is_file():
    target=base/'config/source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);hashes[name]=sha(source)
  save(base/'config/source-hashes.json',hashes)
 results=base/'raw/results.jsonl';results.parent.mkdir(parents=True,exist_ok=True)
 prior={r['run_id']:r for r in map(json.loads,results.read_text().splitlines())} if results.exists() else {}
 for index,j in enumerate(jobs,1):
  if j['run_id'] in prior:continue
  for m,b in BINARIES.items():assert sha(ROOT/b)==evidence['binary_sha256'][m],f'binary changed: {m}'
  key=f"warmup-{j['graph']}-{j['method']}-p200"
  warmup=execute(base,dict(j,run_id=key,seed=0,study='warmup'),graphs[j['graph']],limits,True)
  if warmup['status']!='success':r=dict(j,status='warmup_failed',failure_reason=warmup['status'],warmup_run_id=key,executed=False)
  else:r=execute(base,j,graphs[j['graph']],limits)
  prior[j['run_id']]=r
  with results.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
  counts={s:sum(x['status']==s for x in prior.values()) for s in {x['status'] for x in prior.values()}}
  save(base/'checkpoint.json',dict(completed=len(prior),planned=len(jobs),counts=counts,last_run=j['run_id']))
  print(index,'/',len(jobs),j['run_id'],r['status'],round(r.get('process_wall_seconds',0),2),counts,flush=True)
 print('Completed plan',len(prior),flush=True)
if __name__=='__main__':main()
