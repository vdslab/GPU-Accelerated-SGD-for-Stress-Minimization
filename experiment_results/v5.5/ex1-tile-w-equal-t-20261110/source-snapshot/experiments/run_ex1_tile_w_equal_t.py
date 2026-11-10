#!/usr/bin/env python3
"""EX1 additional tile experiment: W=T; preserves v5 outputs and binaries."""
from pathlib import Path
import argparse,collections,csv,fcntl,hashlib,json,os,random,re,shutil,statistics,subprocess,sys,time
import psutil
from experiment_plan import METHODS,Dataset,RunDefinition
from run_experiments import parse_and_validate_success,file_sha256,git_environment
from run_v4 import archive_text_artifacts
from run_v5 import now,memory,save
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiment_results/v5'
BASE=OLD/'ex1-tile-w-equal-t-20261107'
VAULT=Path('/Users/kose/Documents/ChatGPT/Obsidian タスク管理/sakamoti55/12_大学院研究/実験')
NOTE=VAULT/'EX1_タイル実験(W=T).md'

_OLD_REFERENCES=None
def old_references():
 global _OLD_REFERENCES
 if _OLD_REFERENCES is None:
  _OLD_REFERENCES={}
  for p in (OLD/'raw/tile').glob('*/record.json'):
   r=json.loads(p.read_text())
   if r['method']=='rr_sgd':_OLD_REFERENCES[r['graph'],r['seed'],r['repetition'],r['tile_size']]=r
 return _OLD_REFERENCES

def prepare():
 if (BASE/'config/plan.json').exists():return
 for d in ['config','raw/tile','warmups','tables','reports','figures','validation','source-snapshot','bin','logs']:(BASE/d).mkdir(parents=True,exist_ok=True)
 oldplan=json.loads((OLD/'config/plan.json').read_text())
 datasets=[g for g in json.loads((OLD/'config/datasets.json').read_text()) if g['name'] in oldplan['tile_graphs']]
 assert len(datasets)==10
 for g in datasets:assert file_sha256(Path(g['path']))==g['sha256']
 jobs=[dict(j,run_id=j['run_id'].replace('v5-tile-','ex1-wt-'),workgroup_size=j['tile_size'] if j['method']=='rr_sgd' else None) for j in json.loads((OLD/'config/jobs.json').read_text()) if j['stage']=='tile']
 assert len(jobs)==1800 and len({j['run_id'] for j in jobs})==1800
 oldenv=json.loads((OLD/'config/environment.json').read_text())
 env=git_environment();env.update(created=now(),memory=memory(),binary_sha256={},binary_paths={})
 for m in ['sgd','rr_sgd']:
  original=ROOT/METHODS[m].binary
  previous=BASE/'bin/previous-v5'/original.name
  assert file_sha256(previous)==oldenv['binary_sha256'][m], 'Previous v5 binary archive mismatch'
  frozen=BASE/'bin'/original.name;shutil.copy2(original,frozen)
  env['binary_sha256'][m]=file_sha256(frozen);env['binary_paths'][m]=str(frozen)
 hashes={}
 sources=[Path(__file__),ROOT/'experiments/report_ex1_tile_w_equal_t.py',ROOT/'experiments/run_experiments.py',ROOT/'experiments/experiment_plan.py',ROOT/'experiments/run_v4.py',ROOT/'experiments/run_v5.py',ROOT/'experiments/artifact_paths.py']
 for directory in ['implementations/full/rr_gpu/src','implementations/full/baseline-sgd-non-gpu/src','shared/gpu-timing','shared/experiment-common/src']:
  sources+=list((ROOT/directory).rglob('*.rs'))+list((ROOT/directory).rglob('*.wgsl'))
 for p in sources:
  rel=p.resolve().relative_to(ROOT);dest=BASE/'source-snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);hashes[str(rel)]=file_sha256(p)
 env['source_sha256']=hashes
 env['previous_v5_binary_sha256']=oldenv['binary_sha256']
 env['build_change']='Optional SGD_STRESS_MODE override in shared evaluation; SGD algorithms/shaders unchanged'
 (BASE/'source-snapshot/git-diff.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT))
 shutil.copy2(NOTE,BASE/'config/obsidian-plan-original.md')
 shutil.copy2(VAULT/'11_共通条件と出典.md',BASE/'config/common-conditions-original.md')
 save(BASE/'config/environment.json',env);save(BASE/'config/datasets.json',datasets);save(BASE/'config/jobs.json',jobs)
 with (BASE/'config/execution-order.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(jobs[0]));writer.writeheader();writer.writerows(jobs)
 oldfiles=[OLD/'config/jobs.json',OLD/'config/plan.json',OLD/'reports/tile.md',OLD/'tables/tile-hierarchical.json']+sorted((OLD/'raw/tile').glob('*/record.json'))
 save(BASE/'config/old-results-sha256.json',{str(p.relative_to(ROOT)):file_sha256(p) for p in oldfiles})
 (BASE/'README.md').write_text('# EX1: W=T additional tile experiment\n\nPrevious v5 is read-only. Same datasets and execution order; binaries rebuilt for optional exact evaluation override, seeds10–14, three repetitions, 15 iterations, epsilon0.1; only RR W changes from 1024 to T. CPU150 and RR1650 are freshly measured. shared, inner/outer shuffle on, phase_chunks1, GPU timestamps off. Exact Full Stress for all ten graphs (SGD_STRESS_MODE=exact); evaluation time excluded from SGD/total-compute intervals. Warmup per graph/method/T per session. No time cutoff; stop between runs on <4GiB disk or <2GiB available RAM.\n\nRun/resume: `python3 experiments/run_ex1_tile_w_equal_t.py --run`. Progress: progress.json and PROGRESS.md. Reports generated automatically after all runs. Source and binary snapshots retained.\n')
 plan=dict(created=now(),experiment_id='ex1-tile-w-equal-t-20261107',tile_graphs=oldplan['tile_graphs'],tiles=[2**i for i in range(11)],seeds=list(range(10,15)),repetitions=3,iterations=15,epsilon=.1,workgroup_policy='W=T',coordinate_memory='shared',outer_shuffle=True,inner_shuffle=True,phase_chunks=1,stress_mode='exact',stress_samples=None,stress_seed=None,gpu_timestamps=False,total_formal_runs=1800,warmup_seed=999,execution_order='identical to v5 tile; independent new runs',timeout_seconds=None,previous_results=str(OLD),state='prepared')
 save(BASE/'config/plan.json',plan)

def note_progress(body):
 s=NOTE.read_text();start='<!-- EX1_PROGRESS_START -->';end='<!-- EX1_PROGRESS_END -->'
 section=start+'\n## 実行状況（自動更新）\n\n'+body+'\n'+end
 if start in s:
  a=s.index(start);b=s.index(end,a)+len(end);s=s[:a]+section+s[b:]
 else:s=s.rstrip()+'\n\n'+section+'\n'
 NOTE.write_text(s)

def records():return [json.loads(p.read_text()) for p in (BASE/'raw/tile').glob('*/record.json')]
def progress(active=None,active_elapsed=0):
 jobs=json.loads((BASE/'config/jobs.json').read_text());rows=records();done={r['run_id'] for r in rows};remaining=[j for j in jobs if j['run_id'] not in done]
 timings=collections.defaultdict(list)
 for r in rows:timings[r['graph'],r['method'],r['tile_size']].append(r['process_wall_seconds'])
 eta=0;unknown=0
 for j in remaining:
  ts=timings.get((j['graph'],j['method'],j['tile_size']))
  if ts:eta+=statistics.median(ts)
  else:unknown+=1
 state=dict(updated=now(),state='running' if active else ('completed' if len(rows)==1800 and all(r['status']=='success' for r in rows) else 'in_progress'),planned=1800,finished=len(rows),success=sum(r['status']=='success' for r in rows),failed=sum(r['status']!='success' for r in rows),remaining=len(remaining),active=active,active_elapsed_seconds=active_elapsed,eta_measured_conditions_seconds=eta,eta_unmeasured_runs=unknown,memory=memory(),free_disk_bytes=shutil.disk_usage(BASE).free)
 save(BASE/'progress.json',state)
 msg=f"{now()} 完了{len(rows)}/1800、成功{state['success']}、失敗{state['failed']}、残り{len(remaining)}。実行中：{active['run_id'] if active else 'なし'}（{active_elapsed:.0f}秒）。実測済み条件の残り約{eta/60:.1f}分、未推定{unknown}実行。"
 print(msg,flush=True);(BASE/'PROGRESS.md').write_text('# EX1 W=T 実験進捗\n\n'+msg+'\n')
 note_progress(msg+f'\n\n結果の正本：`{BASE}`。前回v5は保持。完了後は[[EX1_タイル実験(W=T)_結果と考察]]へ保存。')
 return state

def summarize_partial():
 rows=records()
 fields=list(dict.fromkeys(k for r in rows for k in r))
 if rows:
  with (BASE/'tables/tile-runs.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def assert_integrity():
 env=json.loads((BASE/'config/environment.json').read_text())
 for m in ['sgd','rr_sgd']:
  assert file_sha256(Path(env['binary_paths'][m]))==env['binary_sha256'][m]
  METHODS[m]=METHODS[m].__class__(env['binary_paths'][m],METHODS[m].family,METHODS[m].sparse)
 for rel,h in env['source_sha256'].items():assert file_sha256(ROOT/rel)==h,f'Source changed: {rel}'
 for g in json.loads((BASE/'config/datasets.json').read_text()):assert file_sha256(Path(g['path']))==g['sha256']
 for rel,h in json.loads((BASE/'config/old-results-sha256.json').read_text()).items():assert file_sha256(ROOT/rel)==h,f'Previous result modified: {rel}'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--max-runs',type=int);a=ap.parse_args();prepare();assert_integrity()
 if not a.run:progress();return
 lock=(BASE/'runner.lock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('EX1 runner already active')
 lock.write(str(os.getpid()));lock.flush()
 save(BASE/'process.json',dict(pid=os.getpid(),started=now(),argv=sys.argv))
 jobs=json.loads((BASE/'config/jobs.json').read_text());graphs={g['name']:g for g in json.loads((BASE/'config/datasets.json').read_text())};warmed=set();count=0
 session=now().replace(':','').replace('+','_')
 for j in jobs:
  dest=BASE/'raw/tile'/j['run_id']/'record.json'
  if dest.exists():
   assert json.loads(dest.read_text())['status']=='success','Previous failed run requires inspection'
   continue
  if shutil.disk_usage(BASE).free<4*1024**3 or memory()['available_bytes']<2*1024**3:
   progress();save(BASE/'resource-stop.json',dict(at=now(),memory=memory(),disk_free=shutil.disk_usage(BASE).free));raise SystemExit('Stopped between runs for resources; resumable')
  key=(j['graph'],j['method'],j['tile_size'])
  if key not in warmed:
   warm=dict(j,seed=999,repetition=0,run_id=f"warmup-{session}-{j['graph']}-{j['method']}-t{j['tile_size'] or 'cpu'}")
   row=execute(warm,graphs[j['graph']],True)
   if row['status']!='success':raise RuntimeError(f"Warmup failed: {row.get('error')}")
   warmed.add(key)
  progress(j);row=execute(j,graphs[j['graph']]);progress();count+=1
  if row['status']!='success' or row['archive_status']!='success':summarize_partial();raise RuntimeError(f"Run failed: {row.get('error',row.get('archive_error'))}")
  if count%30==0:summarize_partial()
  if a.max_runs and count>=a.max_runs:summarize_partial();return
 summarize_partial();assert_integrity()
 subprocess.run([sys.executable,str(ROOT/'experiments/report_ex1_tile_w_equal_t.py')],check=True,cwd=ROOT)
 state=progress();assert state['success']==1800
 note_progress(f"**測定・集計完了：1800/1800成功。** {now()}\n\n結果・図表：[[EX1_タイル実験(W=T)_結果と考察]]。保存先：`{BASE}`。前回v5は保持し、記録ハッシュの不変を確認した。")
 save(BASE/'completion.json',dict(completed=now(),success=1800,previous_results_unchanged=True,reports_complete=True))

def execute(j,g,warmup=False):
 folder=BASE/('warmups' if warmup else 'raw/'+j['stage'])/j['run_id'];folder.mkdir(parents=True,exist_ok=True)
 dest=folder/'record.json'
 if dest.exists():return json.loads(dest.read_text())
 if (folder/'command.json').exists():raise RuntimeError(f'Incomplete prior attempt retained: {folder}; inspect before rerunning')
 spec=METHODS[j['method']];pivots=min(200,g['nodes']) if spec.sparse else None
 run=RunDefinition('ex1-w-equal-t',j['run_id'],'benchmark',j['method'],ROOT/spec.binary,Dataset(g['name'],Path(g['path']),g['sha256']),j['seed'],15,.1,pivots,j['repetition'])
 cmd=[str(run.binary),'--input',g['path'],'--run-id',j['run_id'],'--run-mode','benchmark','--iterations','15','--epsilon','0.1','--seed',str(j['seed']),'--output-format','json','--output-dir',str(folder)]
 if pivots:cmd+=['--pivots',str(pivots)]
 tile=j['tile_size']
 if j['method']=='rr_sgd':
  assert tile is not None
  cmd+=['--tile-size',str(tile),'--workgroup-size',str(tile),'--coordinate-memory','shared','--phase-chunks','1']
 env=dict(os.environ,SGD_STRESS_MODE='exact',SGD_GPU_TIMESTAMPS='0');env.pop('SGD_STRESS_SAMPLES',None);env.pop('SGD_TRACE_DIR',None);env.pop('SGD_TRACE_CHECKPOINTS',None)
 assert file_sha256(run.binary)==json.loads((BASE/'config/environment.json').read_text())['binary_sha256'][j['method']], 'binary changed during measurement'
 save(folder/'command.json',dict(argv=cmd,environment={'SGD_STRESS_MODE':'exact','SGD_GPU_TIMESTAMPS':'0'},started_at=now(),memory_before=memory(),warmup=warmup))
 start=time.monotonic();peak=0
 with (folder/'stdout.json').open('w') as out,(folder/'stderr.log').open('w') as err:
  proc=subprocess.Popen(['/usr/bin/time','-l',*cmd],cwd=ROOT,env=env,stdout=out,stderr=err)
  while proc.poll() is None:
   try:peak=max(peak,sum(p.memory_info().rss for p in psutil.Process(proc.pid).children(recursive=True)))
   except psutil.Error:pass
   if time.monotonic()-execute.last_progress>60:
    progress(j,active_elapsed=time.monotonic()-start);execute.last_progress=time.monotonic()
   time.sleep(.2)
 wall=time.monotonic()-start
 match=re.search(r'(\d+)\s+maximum resident set size',(folder/'stderr.log').read_text())
 if match:peak=max(peak,int(match[1]))
 row=dict(j,binary_sha256=file_sha256(run.binary),status='error',process_wall_seconds=wall,peak_rss_bytes=peak,memory_after=memory(),completed_at=now(),warmup=warmup)
 try:
  completed=subprocess.CompletedProcess(cmd,proc.returncode,(folder/'stdout.json').read_text(),(folder/'stderr.log').read_text())
  native=parse_and_validate_success(run,completed)
  assert native['nodes']==g['nodes']
  if j['method'].startswith('rr_'):assert native['rr_phase_chunks']==1
  if j['method']=='rr_sgd':assert native['rr_tile_size']==tile and native['rr_workgroup_size']==tile and native['rr_outer_shuffle'] and native['rr_inner_shuffle']
  if 'atomic' in j['method']:assert native['attempted_updates']==native['completed_updates'] and native['retry_failures']==0
  assert native['stress_kind']=='exact' and native['stress_samples'] is None and native['stress_seed'] is None
  if j['method']=='rr_sgd' and not warmup:
   old=old_references()[j['graph'],j['seed'],j['repetition'],tile]
   assert native['initial_positions_sha256']==old['initial_positions_sha256'], 'initial positions differ from v5'
  row.update(native);row.update(j);row['status']='success'
 except Exception as e:row['error']=repr(e)
 archive_text_artifacts(folder,row)
 save(dest,row)
 return row
execute.last_progress=0

if __name__=='__main__':main()
