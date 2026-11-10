#!/usr/bin/env python3
"""EX2: sequential RR128 timing, quality and actual first-frame measurements."""
from pathlib import Path
import argparse,collections,csv,fcntl,gzip,hashlib,json,math,os,re,shutil,statistics,subprocess,sys,tempfile,time
import psutil
from experiment_plan import Dataset,RunDefinition
from run_experiments import parse_and_validate_success,git_environment,file_sha256
from run_v5 import now,save,memory
from run_single import attach_render_timing
from share_v5_large_edges import convert_layout,sha
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiment_results/v5/ex2-rr128-speed-stress-20261108'
OLD=ROOT/'experiment_results/v5'
VAULT=Path('/Users/kose/Documents/ChatGPT/Obsidian タスク管理/sakamoti55/12_大学院研究/実験')
NOTE=VAULT/'EX2_速度ストレス実験.md'
FIELDS=['preprocess_time_ms','gpu_prepare_time_ms','upload_time_ms','iteration_times_ms','sgd_time_ms','readback_time_ms','draw_time_ms','total_with_draw_time_ms','iteration_total_time_ms','total_compute_time_ms']
BIN={'rr_sgd':BASE/'bin/rr-full','rr_sparse_sgd':BASE/'bin/rr-sparse-w128'}
RENDER=BASE/'bin/gpu-visualizer'
_old=None
_last_note_sync=0

def sync_note_progress(msg,force=False):
 global _last_note_sync
 if not force and time.monotonic()-_last_note_sync<30:return
 _last_note_sync=time.monotonic()
 temporary=None
 try:
  before=NOTE.stat()
  text=NOTE.read_text(encoding='utf-8')
  a='<!-- EX2_PROGRESS_START -->';b='<!-- EX2_PROGRESS_END -->'
  body=a+'\n## 実行状況（自動更新）\n\n'+msg+f'\n\n正本：`{BASE}`。結果・図表：[[EX2_速度ストレス実験_結果と考察]]。\n'+b
  if a in text:
   start=text.index(a);end=text.index(b,start)+len(b);text=text[:start]+body+text[end:]
  else:text=text.rstrip()+'\n\n'+body+'\n'
  with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=NOTE.parent,prefix='.ex2-progress-',suffix='.tmp',delete=False) as f:
   temporary=Path(f.name);f.write(text);f.flush();os.fsync(f.fileno())
  after=NOTE.stat()
  if (before.st_mtime_ns,before.st_size)!=(after.st_mtime_ns,after.st_size):
   raise OSError('Note changed concurrently; skip this progress update')
  os.replace(temporary,NOTE);temporary=None
 except (OSError,UnicodeError,ValueError) as error:
  # Repository progress is authoritative; note sync must not stop measurement.
  with (BASE/'logs/note-sync-errors.jsonl').open('a') as f:
   f.write(json.dumps(dict(at=now(),error=repr(error)),ensure_ascii=False)+'\n')
 finally:
  if temporary is not None:temporary.unlink(missing_ok=True)

def old_records():
 global _old
 if _old is None:
  _old={}
  for p in (OLD/'raw/comparison').glob('*/record.json'):
   r=json.loads(p.read_text())
   if r['method'] in BIN:_old[r['graph'],r['method'],r['seed'],r['repetition']]=r
 return _old

def progress(active=None,elapsed=0,state=None):
 jobs=json.loads((BASE/'config/jobs.json').read_text());rows=records();done={r['run_id'] for r in rows};remaining=[j for j in jobs if j['run_id'] not in done]
 times=collections.defaultdict(list)
 for r in rows:times[r['graph'],r['method']].append(r['process_wall_seconds']+r.get('render_wall_seconds',0)+r.get('archive_seconds',0))
 eta=sum(statistics.median(times[j['graph'],j['method']]) for j in remaining if (j['graph'],j['method']) in times)
 unknown=sum((j['graph'],j['method']) not in times for j in remaining)
 s=dict(updated=now(),state=state or ('running' if active else 'in_progress'),planned=1410,finished=len(rows),success=sum(r['status']=='success' for r in rows),failed=sum(r['status']!='success' for r in rows),remaining=len(remaining),active=active,active_elapsed_seconds=elapsed,eta_measured_conditions_seconds=eta,eta_unmeasured_runs=unknown,memory=memory(),free_disk_bytes=shutil.disk_usage(BASE).free)
 save(BASE/'progress.json',s)
 msg=f"{now()} 状態={s['state']} 完了{len(rows)}/1410、成功{s['success']}、失敗{s['failed']}。実行中={active['run_id'] if active else 'なし'} ({elapsed:.0f}秒)。実測済み条件の残り{eta/3600:.2f}時間、未推定{unknown}実行。"
 print(msg,flush=True);(BASE/'PROGRESS.md').write_text('# EX2進捗\n\n'+msg+'\n')
 sync_note_progress(msg,force=state in ('completed','failed','resource_stop'))
 return s

def records():return [json.loads(p.read_text()) for p in (BASE/'raw').glob('*/record.json')]

def freeze():
 if (BASE/'config/jobs.json').exists():return
 jobs=[dict(j,stage='ex2',tile_size=128 if j['method']=='rr_sgd' else None,workgroup_size=128,run_id=j['run_id'].replace('v5-comparison-','ex2-rr128-')) for j in json.loads((OLD/'config/jobs.json').read_text()) if j['stage']=='comparison' and j['method'] in BIN]
 assert len(jobs)==1410 and len({j['run_id'] for j in jobs})==1410
 names={j['graph'] for j in jobs};datasets=[g for g in json.loads((OLD/'config/datasets.json').read_text()) if g['name'] in names]
 for g in datasets:assert file_sha256(Path(g['path']))==g['sha256']
 save(BASE/'config/jobs.json',jobs);save(BASE/'config/datasets.json',datasets)
 with (BASE/'config/execution-order.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(jobs[0]));w.writeheader();w.writerows(jobs)
 sources=[Path(__file__),ROOT/'experiments/report_ex2_rr128.py',ROOT/'experiments/run_experiments.py',ROOT/'experiments/experiment_plan.py',ROOT/'experiments/run_single.py',ROOT/'experiments/share_v5_large_edges.py']
 for directory in ['implementations/full/rr_gpu/src','shared/gpu-timing','shared/experiment-common/src','tools/gpu_visualizer/src']:
  sources+=list((ROOT/directory).rglob('*.rs'))+list((ROOT/directory).rglob('*.wgsl'))
 hashes={}
 for p in sources:
  rel=p.resolve().relative_to(ROOT);target=BASE/'source-snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);hashes[str(rel)]=file_sha256(p)
 for p in (BASE/'source-snapshot/sparse-w128').rglob('*'):
  if p.is_file():hashes[str(p.relative_to(ROOT))]=file_sha256(p)
 env=git_environment();env.update(created=now(),binary_sha256={m:file_sha256(p) for m,p in BIN.items()},renderer_sha256=file_sha256(RENDER),source_sha256=hashes,memory=memory(),render=dict(width=2048,height=2048,node_radius=1,palette='blue',edges='all'),stress=dict(rr_sgd='exact',rr_sparse_sgd='auto, max_n8000, samples256, seed0'),timing_fields=FIELDS)
 save(BASE/'config/environment.json',env)
 (BASE/'source-snapshot/git-diff.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT))
 previous=[OLD/'config/plan.json',OLD/'config/jobs.json']+sorted((OLD/'raw/comparison').glob('*/record.json'))+sorted((OLD/'ex1-tile-w-equal-t-20261107/raw/tile').glob('*/record.json'))
 previous+=list((OLD/'reports').glob('*.md'))+list((OLD/'ex1-tile-w-equal-t-20261107/reports').glob('*.md'))
 save(BASE/'config/previous-records-sha256.json',{str(p.relative_to(ROOT)):file_sha256(p) for p in previous})


def integrity():
 e=json.loads((BASE/'config/environment.json').read_text())
 for m,p in BIN.items():assert file_sha256(p)==e['binary_sha256'][m]
 assert file_sha256(RENDER)==e['renderer_sha256']
 for rel,h in e['source_sha256'].items():assert file_sha256(ROOT/rel)==h,f'Source changed: {rel}'
 for rel,h in json.loads((BASE/'config/previous-records-sha256.json').read_text()).items():assert file_sha256(ROOT/rel)==h,f'Previous record modified: {rel}'


def coverage(r):
 for field in FIELDS:
  assert r.get(field) is not None,f'Missing required timing: {field}'
  values=r[field] if isinstance(r[field],list) else [r[field]]
  assert all(isinstance(v,(float,int)) and math.isfinite(v) and v>=0 for v in values),field
 assert len(r['iteration_times_ms'])==15
 assert math.isclose(sum(r['iteration_times_ms']),r['iteration_total_time_ms'],rel_tol=1e-10,abs_tol=1e-5)
 assert math.isclose(r['total_with_draw_time_ms'],r['total_compute_time_ms']+r['draw_time_ms'],rel_tol=1e-10,abs_tol=1e-5)
 assert r['iteration_total_time_ms']<=r['sgd_time_ms']+1,'GPU pass sum exceeds elapsed SGD'


def position_digest(path):
 opener=gzip.open if str(path).endswith('.gz') else open
 h=hashlib.sha256();active=False
 with opener(path,'rb') as f:
  for line in f:
   if line.startswith(b'# Positions'):active=True;continue
   if active and line.strip() and not line.startswith(b'#'):h.update(line)
 return h.hexdigest()


def archive(folder,row):
 start=time.monotonic();entries=[];originals=[]
 for field in ['initial_positions_path','final_positions_path','vertex_map_path']:
  if not row.get(field):continue
  p=Path(row[field]);digest,size=sha(p)
  if field!='vertex_map_path':entry=convert_layout(p,p,row,BASE/'shared-edges'/row['method'],dict(original_sha256=digest,original_bytes=size))
  else:
   target=p.with_suffix(p.suffix+'.gz')
   with p.open('rb') as src,target.open('xb') as raw:
    with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0) as dst:shutil.copyfileobj(src,dst,1024*1024)
   assert sha(target,gzip.open)==(digest,size)
   encoded,encoded_size=sha(target);entry=dict(original_path=str(p),compressed_path=str(target),original_sha256=digest,original_bytes=size,compressed_sha256=encoded,compressed_bytes=encoded_size,gzip_crc_validated=True)
  row['native_'+field]=str(p);row[field]=entry['compressed_path'];entries.append(entry);originals.append(p)
 row.update(archive_status='success',archive_format='shared-gzip',archive_seconds=time.monotonic()-start,archive_manifest_path=str(folder/'artifact-archive.json'))
 save(folder/'artifact-archive.json',dict(format='shared-gzip',measurement_excluded=True,files=entries,errors=[]))
 save(folder/'record.json',row)
 # The durable manifest and record point at independently validated lossless files.
 for p in originals:p.unlink()


def execute(j,g,kind='raw',timestamps=True,binary=None):
 folder=BASE/kind/j['run_id'];folder.mkdir(parents=True,exist_ok=True);dest=folder/'record.json'
 if dest.exists():return json.loads(dest.read_text())
 assert not (folder/'command.json').exists(),f'Incomplete prior attempt needs inspection: {folder}'
 binary=binary or BIN[j['method']]
 run=RunDefinition('ex2-rr128',j['run_id'],'benchmark',j['method'],binary,Dataset(g['name'],Path(g['path']),g['sha256']),j['seed'],15,.1,min(200,g['nodes']) if j['method']=='rr_sparse_sgd' else None,j['repetition'])
 cmd=[str(binary),'--input',g['path'],'--run-id',j['run_id'],'--run-mode','benchmark','--iterations','15','--epsilon','0.1','--seed',str(j['seed']),'--output-format','json','--output-dir',str(folder)]
 if j['method']=='rr_sgd':cmd+=['--tile-size','128','--workgroup-size','128','--coordinate-memory','shared','--phase-chunks','1']
 else:cmd+=['--pivots',str(run.pivots)]
 values=dict(SGD_GPU_TIMESTAMPS='1' if timestamps else '0',SGD_STRESS_MODE='exact' if j['method']=='rr_sgd' else 'auto',SGD_STRESS_SAMPLES='256')
 env=dict(os.environ,**values);env.pop('SGD_TRACE_DIR',None);env.pop('SGD_TRACE_CHECKPOINTS',None)
 save(folder/'command.json',dict(argv=cmd,environment=values,binary_sha256=file_sha256(binary),started=now(),memory_before=memory(),kind=kind))
 row=dict(j,status='error',warmup=kind!='raw',binary_sha256=file_sha256(binary),memory_before=memory(),configured_workgroup_size=j['workgroup_size'])
 start=time.monotonic();peak=0
 with (folder/'stdout.json').open('w') as out,(folder/'stderr.log').open('w') as err:
  proc=subprocess.Popen(['/usr/bin/time','-l',*cmd],stdout=out,stderr=err,env=env,cwd=ROOT)
  last=0
  while proc.poll() is None:
   try:peak=max(peak,sum(c.memory_info().rss for c in psutil.Process(proc.pid).children(recursive=True)))
   except psutil.Error:pass
   if time.monotonic()-last>60:progress(j,time.monotonic()-start,state='validating' if kind=='validation' else None);last=time.monotonic()
   time.sleep(.2)
 row['process_wall_seconds']=time.monotonic()-start
 stderr=(folder/'stderr.log').read_text();match=re.search(r'(\d+)\s+maximum resident set size',stderr)
 if match:peak=max(peak,int(match[1]))
 row['peak_rss_bytes']=peak
 try:
  native=parse_and_validate_success(run,subprocess.CompletedProcess(cmd,proc.returncode,(folder/'stdout.json').read_text(),stderr));row.update(native);row.update(j)
  assert row['nodes']==g['nodes'] and row['rr_phase_chunks']==1
  if j['method']=='rr_sgd':assert row['rr_tile_size']==row['rr_workgroup_size']==128 and row['rr_memory_mode']=='shared' and row['rr_inner_shuffle'] and row['rr_outer_shuffle'] and row['stress_kind']=='exact'
  else:
   assert row['rr_pivot_order']=='per-vertex-feistel4-cyclewalk'
   assert row['stress_kind']==('exact' if g['nodes']<=8000 else 'sampled')
   if g['nodes']>8000:assert row['stress_samples']==256 and row['stress_seed']==0
  if kind=='raw':
   previous=old_records()[g['name'],j['method'],j['seed'],j['repetition']]
   assert row['initial_positions_sha256']==previous['initial_positions_sha256']
   if j['method']=='rr_sparse_sgd':assert row['preprocess_sha256']==previous['preprocess_sha256']
  # Render immediately after native output: the shared monotonic boundary is F→G.
  rendercmd=[str(RENDER),str(row['final_positions_path']),str(folder/'after.png'),'--size','2048x2048','--stress','off','--node-radius','1','--palette','blue']
  row['render_command']=rendercmd;renderstart=time.monotonic()
  rendered=subprocess.run(rendercmd,cwd=ROOT,capture_output=True,text=True)
  row['render_wall_seconds']=time.monotonic()-renderstart
  (folder/'renderer.log').write_text(rendered.stdout+'\n'+rendered.stderr)
  if rendered.returncode:raise RuntimeError('Renderer failed: '+rendered.stderr[-1500:])
  attach_render_timing(row,rendered.stdout)
  if timestamps:coverage(row)
  else:assert row['iteration_times_ms'] is None
  row['final_coordinate_text_sha256']=position_digest(Path(row['final_positions_path']))
  initial=list(folder.glob('*.initial.txt'));assert len(initial)==1
  row['initial_positions_path']=str(initial[0])
  png=folder/'after.png';row['rendered_png_sha256']=file_sha256(png);row['rendered_png_bytes']=png.stat().st_size
  row['render_image_retention']='First-frame timing and image SHA retained; per-run PNG omitted; source coordinates retained losslessly; report charts retained'
  row.update(status='success',completed_at=now(),memory_after=memory(),timing_complete=timestamps)
  archive(folder,row);png.unlink()
 except Exception as error:
  row.update(status='error',error=repr(error),completed_at=now());save(dest,row);raise
 return row


def preflight(graphs):
 if (BASE/'validation/preflight-complete.json').exists():return
 checks=[]
 for method,name in [('rr_sgd','dwt_1005'),('rr_sgd','fe_4elt2'),('rr_sparse_sgd','dwt_1005'),('rr_sparse_sgd','coPapersCiteseer')]:
  pair=[]
  for timestamps in [False,True]:
   j=dict(graph=name,method=method,tile_size=128 if method=='rr_sgd' else None,workgroup_size=128,seed=999,repetition=0,run_id=f'preflight-{method}-{name}-timestamps{int(timestamps)}')
   pair.append(execute(j,graphs[name],kind='validation',timestamps=timestamps))
  assert pair[0]['final_coordinate_text_sha256']==pair[1]['final_coordinate_text_sha256'],'Timestamp enablement changed coordinates'
  checks.append(dict(method=method,graph=name,outputs_identical=True,timing_fields=10,timestamps_on_over_off_sgd=pair[1]['sgd_time_ms']/pair[0]['sgd_time_ms']))
 # The sparse W change must retain the independent vertex update result.
 j=dict(graph='dwt_1005',method='rr_sparse_sgd',tile_size=None,workgroup_size=256,seed=999,repetition=0,run_id='preflight-sparse-w256-reference')
 ref=execute(j,graphs['dwt_1005'],kind='validation',timestamps=True,binary=BASE/'bin/rr-sparse-w256-reference')
 compare=json.loads((BASE/'validation/preflight-rr_sparse_sgd-dwt_1005-timestamps1/record.json').read_text())
 assert ref['initial_positions_sha256']==compare['initial_positions_sha256'] and ref['preprocess_sha256']==compare['preprocess_sha256'] and ref['final_coordinate_text_sha256']==compare['final_coordinate_text_sha256']
 save(BASE/'validation/preflight-complete.json',dict(completed=now(),checks=checks,sparse_w128_vs_w256_identical=True,formal_runs=0))


def summary_csv():
 rows=records()
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (BASE/'tables/runs.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');a=ap.parse_args();freeze();integrity()
 if not a.run:progress(state='prepared');return
 lock=(BASE/'runner.lock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('EX2 runner already active')
 lock.write(str(os.getpid()));lock.flush();save(BASE/'process.json',dict(pid=os.getpid(),started=now(),argv=sys.argv))
 graphs={g['name']:g for g in json.loads((BASE/'config/datasets.json').read_text())}
 progress(state='validating');preflight(graphs);integrity()
 jobs=json.loads((BASE/'config/jobs.json').read_text());warmed=set();session=now().replace(':','').replace('+','_')
 for j in jobs:
  p=BASE/'raw'/j['run_id']/'record.json'
  if p.exists():assert json.loads(p.read_text())['status']=='success';continue
  if shutil.disk_usage(BASE).free<4*1024**3 or memory()['available_bytes']<2*1024**3:
   progress(state='resource_stop');save(BASE/'resource-stop.json',dict(at=now(),memory=memory(),free_disk_bytes=shutil.disk_usage(BASE).free));return
  key=(j['graph'],j['method'])
  if key not in warmed:
   warm=dict(j,seed=999,repetition=0,run_id=f"warmup-{session}-{j['graph']}-{j['method']}")
   execute(warm,graphs[j['graph']],kind='warmups');warmed.add(key)
  progress(j);execute(j,graphs[j['graph']]);progress()
  if len(records())%30==0:summary_csv()
 summary_csv();integrity()
 subprocess.run([sys.executable,str(ROOT/'experiments/report_ex2_rr128.py')],cwd=ROOT,check=True)
 progress(state='completed');save(BASE/'completion.json',dict(completed=now(),success=1410,previous_results_unchanged=True,reports_complete=True))

if __name__=='__main__':
 try:main()
 except Exception as error:
  save(BASE/'failure.json',dict(at=now(),error=repr(error)))
  if (BASE/'config/jobs.json').exists():progress(state='failed')
  raise
