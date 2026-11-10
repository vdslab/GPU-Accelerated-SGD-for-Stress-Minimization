#!/usr/bin/env python3
"""v5.5 EX1: RR-only W=T rerun; previous measurements are read-only."""
from pathlib import Path
import argparse,collections,csv,fcntl,json,math,os,shutil,statistics,subprocess,sys,time
import run_ex1_tile_w_equal_t as backend
import run_experiments as validation
from experiment_plan import METHODS
from run_experiments import file_sha256 as sha,git_environment
from run_v5 import now,memory,save

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiment_results/v5.5/ex1-tile-w-equal-t-20261110'
OLD=ROOT/'experiment_results/v5/ex1-tile-w-equal-t-20261107'
OLD_EX2=ROOT/'experiment_results/v5/ex2-rr128-speed-stress-20261108'
VAULT=Path('/Users/kose/Documents/ChatGPT/Obsidian タスク管理/sakamoti55/12_大学院研究/実験')
NOTE=VAULT/'EX1_v5.5_タイル実験(W=T).md'
FIELDS=['bfs_engine','bfs_init_time_ms','bfs_kernel_time_ms','bfs_compaction_time_ms']

def previous_rows():return [json.loads(p.read_text()) for p in (OLD/'raw/tile').glob('*/record.json')]
def records():return [json.loads(p.read_text()) for p in (BASE/'raw/tile').glob('*/record.json')]
def prepare():
 if (BASE/'config/plan.json').exists():return
 for d in ['config','raw/tile','warmups','tables','reports','figures','validation','source-snapshot','bin','logs']:(BASE/d).mkdir(parents=True,exist_ok=True)
 oldplan=json.loads((OLD/'config/plan.json').read_text());datasets=json.loads((OLD/'config/datasets.json').read_text())
 original=previous_rows();assert len(original)==1800 and all(r['status']=='success' for r in original)
 cpu=[r for r in original if r['method']=='sgd'];assert len(cpu)==150 and all(r['stress_kind']=='exact' for r in cpu)
 refidx={(r['graph'],r['seed'],r['repetition']):r for r in cpu}
 jobs=[]
 for j in json.loads((OLD/'config/jobs.json').read_text()):
  if j['method']!='rr_sgd':continue
  jobs.append(dict(j,run_id=j['run_id'].replace('ex1-wt-','v55-ex1-'),experiment_version='v5.5',experiment_id='v5.5-ex1-w-equal-t',reference_rr_run_id=j['run_id'],reference_cpu_run_id=refidx[j['graph'],j['seed'],j['repetition']]['run_id']))
 assert len(jobs)==1650 and len({j['run_id'] for j in jobs})==1650
 for g in datasets:assert sha(Path(g['path']))==g['sha256']
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 assert commit==subprocess.check_output(['git','rev-parse','origin/feat/wgpu-optimization-atomic-gpu-bfs'],cwd=ROOT,text=True).strip()
 assert not subprocess.check_output(['git','diff','HEAD','--','implementations','shared'],cwd=ROOT)
 external=ROOT.parent/'webgpu-bfs';assert not subprocess.check_output(['git','-C',str(external),'status','--porcelain'])
 oldfiles=[p for folder in [OLD,OLD_EX2] for p in folder.rglob('*') if p.is_file() and p.name!='.DS_Store']
 save(BASE/'config/previous-results-sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in oldfiles})
 target=BASE/'build';binary=BASE/'bin/rr-gpu'
 cmd=['cargo','build','--release','--locked','--manifest-path',str(ROOT/'implementations/full/rr_gpu/Cargo.toml'),'--target-dir',str(target)]
 save(BASE/'logs/build-command.json',dict(argv=cmd,source_commit=commit,started=now()))
 print(now(),'BUILD v5.5 EX1 RR Full',flush=True)
 with (BASE/'logs/build.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError((BASE/'logs/build.log').read_text()[-4000:])
 shutil.copy2(target/'release/rr-gpu',binary)
 sources=[Path(__file__),ROOT/'experiments/report_v55_ex1.py',ROOT/'experiments/run_ex1_tile_w_equal_t.py',ROOT/'experiments/run_experiments.py',ROOT/'experiments/run_v4.py',ROOT/'experiments/run_v5.py',ROOT/'experiments/experiment_plan.py',ROOT/'experiments/artifact_paths.py',ROOT/'experiments/record_metadata.py',ROOT/'experiments/schema/experiment-record-v1.json']
 names=subprocess.check_output(['git','ls-files','implementations/full/rr_gpu','shared/experiment-common','shared/gpu-timing'],cwd=ROOT,text=True).splitlines()
 sources+= [ROOT/p for p in names if (ROOT/p).is_file()]
 hashes={}
 for p in sorted(set(sources)):
  rel=p.relative_to(ROOT);dest=BASE/'source-snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);hashes[str(rel)]=sha(p)
 with (BASE/'source-snapshot/external-bfs.tar.gz').open('wb') as out:subprocess.run(['git','-C',str(external),'archive','--format=tar.gz','HEAD','crates','Cargo.toml','Cargo.lock'],stdout=out,check=True)
 schema=json.loads(validation.SCHEMA_PATH.read_text())
 schema['properties'].update(bfs_engine={'type':'string','enum':['auto','cpu','gpu']},**{f:{'type':'number','minimum':0} for f in FIELDS[1:]})
 save(BASE/'config/experiment-record-schema-v55.json',schema)
 save(BASE/'config/datasets.json',datasets);save(BASE/'config/jobs.json',jobs);save(BASE/'config/cpu-reference-records.json',cpu)
 save(BASE/'config/previous-rr-reference-records.json',[r for r in original if r['method']=='rr_sgd'])
 shutil.copy2(OLD/'config/plan.json',BASE/'config/previous-ex1-plan.json')
 shutil.copy2(VAULT/'EX1_タイル実験(W=T).md',BASE/'config/previous-obsidian-ex1.md')
 shutil.copy2(VAULT/'11_共通条件と出典.md',BASE/'config/common-conditions.md')
 with (BASE/'config/execution-order.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(jobs[0]));w.writeheader();w.writerows(jobs)
 environment=git_environment();environment.update(created=now(),version='v5.5',source_sha256=hashes,binary_sha256={'rr_sgd':sha(binary)},binary_paths={'rr_sgd':str(binary)},external_bfs_commit=subprocess.check_output(['git','-C',str(external),'rev-parse','HEAD'],text=True).strip(),memory=memory(),previous_ex1=str(OLD),previous_cpu_measured_runs=150,cpu_rerun=False,raw_json_schema_extension=FIELDS)
 save(BASE/'config/environment.json',environment)
 plan=dict(oldplan,created=now(),version='v5.5',experiment_id='v5.5-ex1-w-equal-t',methods=['rr_sgd'],total_formal_runs=1650,reused_cpu_reference_runs=150,previous_results=str(OLD),cpu_reference='previous EX1 exact Full Stress; historical timing reference',bfs_engine='auto',execution_order='previous EX1 order filtered to RR Full; no CPU/Atomic rerun',ex2_state='not_started')
 (BASE/'.gitignore').write_text('build/\n')
 (BASE/'README.md').write_text('''# v5.5 EX1: RR Full W=T

Only RR Full is newly measured:10 graphs x11 T=W x5 seeds x3 repetitions=1650. CPU150 exact-EX1 measurements are historical references, not newly measured. Previous v5 EX1/EX2 files are hashed and never written. EX2 is not started.

Same input bytes, seeds10..14, repetitions0..2,15 iterations,epsilon0.1,shared,both shuffles,phase_chunks1,auto BFS,exact Full Stress. GPU timestamps off, no drawing during measurement. Warmup seed999 per graph/T per session excluded. Native solver binary freshly built from PR head and frozen with source/binary hashes. Sequential, no time cutoff. Stop between runs at <4GiB disk/<2GiB available RAM. Lossless coordinates gzip after measurement. Strict native record validation uses a copied schema extended only for PR BFS metadata.

Run/resume: `python3 experiments/run_v55_ex1.py --run`. Progress: progress.json/PROGRESS.md and the new v5.5 Obsidian note. Reports and PNG/PDF figures generated automatically after all1650 runs; no automatic EX2 transition.
''')
 (BASE.parent/'README.md').write_text('# v5.5 RR再実験\n\nEX1：[ex1-tile-w-equal-t-20261110](ex1-tile-w-equal-t-20261110/README.md)。RR Fullのみ新規1,650件。CPU・Atomicの前回結果は保持して参照。EX2（RR Full/Sparse）は未開始。\n')
 save(BASE/'config/plan.json',plan)

def integrity(full=False):
 env=json.loads((BASE/'config/environment.json').read_text())
 assert sha(Path(env['binary_paths']['rr_sgd']))==env['binary_sha256']['rr_sgd']
 for rel,h in env['source_sha256'].items():assert sha(ROOT/rel)==h,f'Source changed: {rel}'
 for g in json.loads((BASE/'config/datasets.json').read_text()):assert sha(Path(g['path']))==g['sha256']
 if full:
  for rel,h in json.loads((BASE/'config/previous-results-sha256.json').read_text()).items():assert sha(ROOT/rel)==h,f'Previous result changed: {rel}'

_rows=[];_jobs=[];_last_note=0
def progress(active=None,active_elapsed=0,state=None,force=False):
 global _last_note
 done={r['run_id'] for r in _rows};remaining=[j for j in _jobs if j['run_id'] not in done]
 times=collections.defaultdict(list)
 for r in _rows:times[r['graph'],r['tile_size']].append(r['process_wall_seconds'])
 known=sum(statistics.median(times[j['graph'],j['tile_size']]) for j in remaining if (j['graph'],j['tile_size']) in times)
 unknown=sum((j['graph'],j['tile_size']) not in times for j in remaining)
 s=dict(updated=now(),version='v5.5',state=state or ('running' if active else 'in_progress'),planned=1650,finished=len(_rows),success=sum(r['status']=='success' for r in _rows),failed=sum(r['status']!='success' for r in _rows),remaining=len(remaining),active=active,active_elapsed_seconds=active_elapsed,eta_known_conditions_seconds=known,eta_unmeasured_runs=unknown,memory=memory(),free_disk_bytes=shutil.disk_usage(BASE).free,ex2_state='not_started')
 save(BASE/'progress.json',s)
 msg=f"{now()} 状態={s['state']}、成功{s['success']}/1650、失敗{s['failed']}、残り{len(remaining)}。実行中：{active['run_id'] if active else 'なし'}（{active_elapsed:.0f}秒）。EX2は未開始。"
 (BASE/'PROGRESS.md').write_text('# v5.5 EX1進捗\n\n'+msg+'\n')
 if force or time.monotonic()-_last_note>60:
  print(msg,flush=True);_last_note=time.monotonic()
  NOTE.write_text('# EX1 v5.5：タイル実験（W=T）\n\nRR Fullのみ新規1,650実行。前回のCPU150件は厳密Stress・参考速度の基準として参照し、前回データは変更しない。\n\n10グラフ：email、add20、CA-GrQc、p2p-Gnutella08、PGPgiantcompo、dwt_1005、dwt_2680、3elt、bcspwr10、fe_4elt2。T=W=1,2,4,8,16,32,64,128,256,512,1024。seed10〜14×各3回、15反復、epsilon0.1、shared、内外shuffle有効、phase_chunks1、auto BFS。全グラフ厳密Full Stress（速度計時後）、GPU timestamp無効。graph/Tごとのseed999ウォームアップを除外し、逐次測定。時間打切りなし。次の実行前に空きディスク4GiB・利用可能RAM2GiBを確認。\n\n## 実行状況\n\n'+msg+f'\n\n正本：`{BASE}`。結果・図表：[[EX1_v5.5_タイル実験(W=T)_結果と考察]]。同seed3回中央値→5seedの中央値・最小〜最大で集計。CPU比は過去測定との参考比較と明記。旧EX1のW=Tと今回を対応比較する。RR SparseはEX2で実施予定。\n')
 return s

def main():
 global _rows,_jobs
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true');ap.add_argument('--max-runs',type=int);a=ap.parse_args()
 BASE.mkdir(parents=True,exist_ok=True)
 lock=(BASE/'runner.lock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('v5.5 EX1 runner already active')
 lock.write(str(os.getpid()));lock.flush()
 if a.prepare or not (BASE/'config/plan.json').exists():prepare()
 integrity(True)
 environment=json.loads((BASE/'config/environment.json').read_text())
 spec=METHODS['rr_sgd'];METHODS['rr_sgd']=spec.__class__(environment['binary_paths']['rr_sgd'],spec.family,spec.sparse)
 backend.BASE=BASE;backend.NOTE=NOTE;backend.progress=progress
 backend._OLD_REFERENCES={(r['graph'],r['seed'],r['repetition'],r['tile_size']):r for r in json.loads((BASE/'config/previous-rr-reference-records.json').read_text())}
 validation.SCHEMA_PATH=BASE/'config/experiment-record-schema-v55.json'
 _jobs=json.loads((BASE/'config/jobs.json').read_text());_rows=records()
 assert all(r['status']=='success' and r['archive_status']=='success' for r in _rows),'Inspect retained failure before resuming'
 if not a.run:progress(force=True);return
 save(BASE/'process.json',dict(pid=os.getpid(),started=now(),argv=sys.argv))
 graphs={g['name']:g for g in json.loads((BASE/'config/datasets.json').read_text())}
 warmed=set();done={r['run_id'] for r in _rows};session=now().replace(':','').replace('+','_');count=0
 backend.execute.last_progress=time.monotonic()
 for j in _jobs:
  if j['run_id'] in done:continue
  if shutil.disk_usage(BASE).free<4*1024**3 or memory()['available_bytes']<2*1024**3:
   progress(state='resource_stop',force=True);save(BASE/'resource-stop.json',dict(at=now(),memory=memory(),disk=shutil.disk_usage(BASE).free));return
  key=(j['graph'],j['tile_size'])
  if key not in warmed:
   integrity()
   warm=dict(j,seed=999,repetition=0,run_id=f'warm-v55-{session}-{j["graph"]}-t{j["tile_size"]}')
   progress(warm,state='warming_up')
   r=backend.execute(warm,graphs[j['graph']],True)
   if r['status']!='success' or r['archive_status']!='success':progress(state='warmup_failed',force=True);raise RuntimeError(r.get('error',r.get('archive_error')))
   warmed.add(key)
  progress(j)
  r=backend.execute(j,graphs[j['graph']]);_rows.append(r);done.add(j['run_id']);count+=1
  if r['status']!='success' or r['archive_status']!='success':progress(state='failed',force=True);raise RuntimeError(r.get('error',r.get('archive_error')))
  assert r['bfs_engine']=='auto' and r['rr_memory_mode']=='shared' and r['gpu_backend']=='Metal'
  assert all(isinstance(r[f],(int,float)) and not isinstance(r[f],bool) and math.isfinite(r[f]) and r[f]>=0 for f in FIELDS[1:])
  assert r['iteration_times_ms'] is None and r['iteration_total_time_ms'] is None
  if count%50==0:integrity()
  progress()
  if a.max_runs and count>=a.max_runs:progress(state='limited_run_finished',force=True);return
 assert len(_rows)==1650
 integrity(True);progress(state='reporting',force=True)
 subprocess.run([sys.executable,str(ROOT/'experiments/report_v55_ex1.py')],cwd=ROOT,check=True)
 integrity(True)
 save(BASE/'completion.json',dict(completed=now(),version='v5.5',new_rr_runs=1650,reused_cpu_reference_runs=150,previous_ex1_ex2_hashes_unchanged=True,source_and_binary_hashes_unchanged=True,reports_complete=True,ex2_started=False))
 progress(state='completed',force=True)

if __name__=='__main__':main()
