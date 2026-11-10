#!/usr/bin/env python3
"""v5.5 EX2: RR Full/Sparse only; historical CPU/Atomic references unchanged."""
from pathlib import Path
import argparse,collections,csv,fcntl,json,os,shutil,statistics,subprocess,sys,time
import run_ex2_rr128 as backend
import run_experiments as validation
from run_experiments import file_sha256 as sha,git_environment
from run_v5 import now,memory,save

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiment_results/v5.5/ex2-rr128-speed-stress-20261110'
OLD=ROOT/'experiment_results/v5/ex2-rr128-speed-stress-20261108'
OLD_V5=ROOT/'experiment_results/v5'
VAULT=Path('/Users/kose/Documents/ChatGPT/Obsidian タスク管理/sakamoti55/12_大学院研究/実験')
NOTE=VAULT/'EX2_v5.5_速度ストレス実験.md'
BIN={'rr_sgd':BASE/'bin/rr-full','rr_sparse_sgd':BASE/'bin/rr-sparse-w128'}
RENDER=BASE/'bin/gpu-visualizer'
_rows=[];_jobs=[];_last_note=0

def records():return [json.loads(p.read_text()) for p in (BASE/'raw').glob('*/record.json')]
def prepare():
 if (BASE/'config/plan.json').exists():return
 for d in ['config','raw','warmups','validation','reports','tables','figures','source-snapshot','bin','logs']:(BASE/d).mkdir(parents=True,exist_ok=True)
 datasets=json.loads((OLD/'config/datasets.json').read_text())
 jobs=[dict(j,run_id=j['run_id'].replace('ex2-rr128-','v55-ex2-'),version='v5.5',experiment_id='v5.5-ex2-rr128',previous_rr_run_id=j['run_id']) for j in json.loads((OLD/'config/jobs.json').read_text())]
 assert len(jobs)==1410 and len(datasets)==56
 for g in datasets:assert sha(Path(g['path']))==g['sha256']
 graphs={g['name']:g for g in datasets}
 unavailable=[j for j in jobs if j['method']=='rr_sparse_sgd' and graphs[j['graph']]['nodes']*min(200,graphs[j['graph']]['nodes'])>4*16_777_216]
 assert len(unavailable)==15 and {j['graph'] for j in unavailable}=={'coPapersCiteseer'}
 oldrr=[json.loads(p.read_text()) for p in (OLD/'raw').glob('*/record.json')];assert len(oldrr)==1410
 controls=[json.loads(p.read_text()) for p in (OLD_V5/'raw/comparison').glob('*/record.json')]
 controls=[r for r in controls if r['method'] in ['sgd','atomic_sgd','sparse_sgd','atomic_sparse_sgd']];assert len(controls)==2820
 overlay_path=OLD_V5/'ex-chart-collection-rr128-20261108/tables/full-exact-stress-overlay.json'
 overlay=json.loads(overlay_path.read_text())
 protect=[p for p in OLD.rglob('*') if p.is_file() and p.name!='.DS_Store']
 protect+=list((OLD_V5/'raw/comparison').glob('*/record.json'))+[overlay_path]
 protect+=list((OLD_V5/'ex1-tile-w-equal-t-20261107/raw/tile').glob('*/record.json'))
 save(BASE/'config/previous-files-sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in protect})
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 assert head==subprocess.check_output(['git','rev-parse','origin/feat/wgpu-optimization-atomic-gpu-bfs'],cwd=ROOT,text=True).strip()
 assert not subprocess.check_output(['git','diff','HEAD','--','implementations','shared'],cwd=ROOT)
 external=ROOT.parent/'webgpu-bfs';assert not subprocess.check_output(['git','-C',str(external),'status','--porcelain'])
 for method,crate,binary in [('rr_sgd','implementations/full/rr_gpu','rr-gpu'),('rr_sparse_sgd','implementations/sparse/sparse-sgd-gpu','sparse-sgd-gpu')]:
  command=['cargo','build','--release','--locked','--manifest-path',str(ROOT/crate/'Cargo.toml'),'--target-dir',str(BASE/'build')]
  save(BASE/'logs'/f'build-{method}-command.json',dict(argv=command,commit=head,started=now()))
  print(now(),'BUILD',method,flush=True)
  with (BASE/'logs'/f'build-{method}.log').open('w') as log:p=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  if p.returncode:raise RuntimeError((BASE/'logs'/f'build-{method}.log').read_text()[-4000:])
  shutil.copy2(BASE/'build/release'/binary,BIN[method])
 shutil.copy2(OLD/'bin/gpu-visualizer',RENDER)
 assert sha(RENDER)==json.loads((OLD/'config/environment.json').read_text())['renderer_sha256']
 schema=json.loads(validation.SCHEMA_PATH.read_text());schema['properties'].update(bfs_engine={'type':'string','enum':['auto','cpu','gpu']},**{k:{'type':'number','minimum':0} for k in ['bfs_init_time_ms','bfs_kernel_time_ms','bfs_compaction_time_ms']})
 save(BASE/'config/schema-v55.json',schema)
 save(BASE/'config/datasets.json',datasets);save(BASE/'config/jobs.json',jobs);save(BASE/'config/unsupported-jobs.json',unavailable)
 save(BASE/'config/previous-rr-records.json',oldrr);save(BASE/'config/cpu-atomic-reference-records.json',controls);save(BASE/'config/full-exact-stress-overlay.json',overlay)
 with (BASE/'config/execution-order.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(jobs[0]));w.writeheader();w.writerows(jobs)
 source_files=[Path(__file__),ROOT/'experiments/report_v55_ex2.py',ROOT/'experiments/run_ex2_rr128.py',ROOT/'experiments/run_experiments.py',ROOT/'experiments/record_metadata.py',ROOT/'experiments/run_single.py',ROOT/'experiments/run_v5.py',ROOT/'experiments/share_v5_large_edges.py',ROOT/'experiments/experiment_plan.py',ROOT/'experiments/artifact_paths.py',ROOT/'experiments/schema/experiment-record-v1.json']
 names=subprocess.check_output(['git','ls-files','implementations/full/rr_gpu','implementations/sparse/sparse-sgd-gpu','shared/experiment-common','shared/gpu-timing','tools/gpu_visualizer'],cwd=ROOT,text=True).splitlines()
 source_files+=[ROOT/name for name in names if (ROOT/name).is_file()]
 hashes={}
 for p in sorted(set(source_files)):
  rel=p.relative_to(ROOT);dest=BASE/'source-snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);hashes[str(rel)]=sha(p)
 with (BASE/'source-snapshot/external-bfs.tar.gz').open('wb') as f:subprocess.run(['git','-C',str(external),'archive','--format=tar.gz','HEAD','crates','Cargo.toml','Cargo.lock'],stdout=f,check=True)
 env=git_environment();env.update(created=now(),version='v5.5',binary_sha256={m:sha(p) for m,p in BIN.items()},binary_paths={m:str(p) for m,p in BIN.items()},renderer_sha256=sha(RENDER),source_sha256=hashes,external_bfs_commit=subprocess.check_output(['git','-C',str(external),'rev-parse','HEAD'],text=True).strip(),memory=memory())
 save(BASE/'config/environment.json',env)
 for j in unavailable:
  n=graphs[j['graph']]['nodes'];save(BASE/'raw'/j['run_id']/'record.json',dict(j,status='not_run_capacity',layout_status='not_started',stress_status='not_evaluated',nodes=n,pivots=200,reason=f'Current PR requires dense n*p <=67108864 one-sided entries; {n}*200={n*200} exceeds compiled 4-bank512MiB guard. Mathematical precheck; no allocation/run attempted; pivots not reduced.',measurement_available=False))
 plan=dict(version='v5.5',experiment='EX2',created=now(),graphs=56,full_conditions=38,sparse_conditions=56,requested_runs=1410,runnable_runs=1395,capacity_not_run=15,seeds=list(range(5)),repetitions=3,iterations=15,epsilon=.1,full_tile=128,full_workgroup=128,sparse_workgroup=128,pivots='min(200,n)',phase_chunks=1,gpu_timestamps=True,full_stress='exact',sparse_stress='exact<=8000; sampled256/eval-seed0 otherwise',warmup_seed=999,timeout_seconds=None,previous_rr=str(OLD),cpu_atomic_new_runs=0,historical_cpu_atomic_records=2820,ex1_state='stopped_at_user_request;742successful runs retained separately',auto_start_ex1=False)
 save(BASE/'config/plan.json',plan)
 (BASE/'.gitignore').write_text('build/\n')
 (BASE/'README.md').write_text('''# v5.5 EX2 RR-only rerun

User changed scope to EX2 only. EX1 stopped after742 successful runs and remains a separate incomplete dataset. EX2 requests1410 RR jobs, of which15 coPapersCiteseer Sparse p200 jobs exceed the current PR compiled 4-bank512MiB cap. Preserve parameters and mark not_run_capacity, never zero-fill. Run1395 supported RR Full/Sparse jobs sequentially. CPU/Atomic2820 historical records are references; no reruns. Previous v5 results are read-only and hashed.

Same EX2 inputs/order,seeds0..4 x3 repetitions,15 iterations,epsilon0.1,Full T=W128/shared/both shuffles,Sparse W128/p=min200n,phase_chunks1,auto BFS,timestamps enabled. Full exact Stress; Sparse exact<=8000 otherwise256 fixed sources/eval seed0. Each graph/method warmup999 excluded. Representative small/large timestamp preflight. No time cutoff. Resource stop between runs below4GiB disk or2GiB available RAM. Actual After GPU render2048x2048 blue radius1. Per-run PNG hashed then omitted; coordinates stored losslessly with APFS-shared edge gzip members. Reports/PNG/PDF automatically generated after supported measurements finish.

Run/resume: `python3 experiments/run_v55_ex2.py --run`. Progress in progress.json and new v5.5 Obsidian notes. Legacy CPU/Atomic exact Full Stress overlay reused only in a new report-side copy, never altering original records. Historical timing comparisons are not contemporaneous.
''')
 (BASE.parent/'README.md').write_text('# v5.5：EX2 RR再実験\n\n依頼をEX2のみへ変更。EX1は742/1650成功で停止し、途中データを別フォルダに保持。最終EX2集計に混ぜない。\n\nEX2：`ex2-rr128-speed-stress-20261110/`。RRのみ1,410件計画、現実装の容量上限で15件未実行、1,395件を測定。CPU・Atomicは過去データ参照。v5の既存データは変更しない。\n')

def integrity(full=False):
 e=json.loads((BASE/'config/environment.json').read_text())
 for m,p in BIN.items():assert sha(p)==e['binary_sha256'][m]
 assert sha(RENDER)==e['renderer_sha256']
 for rel,h in e['source_sha256'].items():assert sha(ROOT/rel)==h,f'Source changed: {rel}'
 for g in json.loads((BASE/'config/datasets.json').read_text()):assert sha(Path(g['path']))==g['sha256']
 if full:
  for rel,h in json.loads((BASE/'config/previous-files-sha256.json').read_text()).items():assert sha(ROOT/rel)==h,f'Previous file changed: {rel}'

def progress(active=None,elapsed=0,state=None,force=False):
 global _last_note
 done={r['run_id'] for r in _rows};remaining=[j for j in _jobs if j['run_id'] not in done]
 success=sum(r['status']=='success' for r in _rows);unavailable=sum(r['status']=='not_run_capacity' for r in _rows);failed=len(_rows)-success-unavailable
 s=dict(updated=now(),version='v5.5',state=state or ('running' if active else 'in_progress'),planned=1410,runnable=1395,finished=len(_rows),success=success,capacity_not_run=unavailable,failed=failed,remaining=len(remaining),active=active,active_elapsed_seconds=elapsed,memory=memory(),free_disk_bytes=shutil.disk_usage(BASE).free,ex1_state='stopped_at_user_request')
 save(BASE/'progress.json',s)
 msg=f"{now()} 状態={s['state']}、成功{success}/1395、容量上限の未実行{unavailable}/15、失敗{failed}、残り{len(remaining)}。実行中：{active['run_id'] if active else 'なし'}（{elapsed:.0f}秒）。"
 (BASE/'PROGRESS.md').write_text('# v5.5 EX2進捗\n\n'+msg+'\n')
 if force or time.monotonic()-_last_note>60:
  _last_note=time.monotonic();print(msg,flush=True)
  NOTE.write_text('# EX2 v5.5：RR Full／Sparse速度・Stress比較\n\nEX1からEX2のみへ切替。EX1の742件は別フォルダに途中結果として保持し、今回のEX2へ混ぜない。\n\n前回EX2と同じ56入力、Full38条件＋Sparse56条件。RRのみ新規1,410件計画。coPapersCiteseer Sparse p200の15件は現PRのdense n*p・4bank512MiB上限で事前に未実行と記録し、パラメータを減らさない。対応可能1,395件を逐次測定する。CPU・Atomic2,820件は過去データを参照し、取り直さない。過去データは変更しない。\n\nseed0〜4×各3回、15反復、epsilon0.1、Full T=W128/shared/両shuffle/Sparse W128・p=min(200,n)、phase_chunks1、auto BFS。GPU timestamp有効。Full全件厳密Stress、Sparse n<=8000厳密・それ以外256始点/評価seed0の推定Full Stress。graph/method seed999ウォームアップを除外。After実GPU描画2048×2048、blue、半径1px。CPU最終座標取得→Stress評価・座標I/O・描画準備を含むF→Gと、SGD・描画なし総計算は区別する。\n\n## 実行状況\n\n'+msg+f'\n\n正本：`{BASE}`。結果・図表：[[EX2_v5.5_速度ストレス実験_結果と考察]]。\n')
 return s

def preflight(graphs):
 if (BASE/'validation/preflight-complete.json').exists():return
 checks=[]
 for method,g in [('rr_sgd','dwt_1005'),('rr_sgd','fe_4elt2'),('rr_sparse_sgd','dwt_1005'),('rr_sparse_sgd','com-Amazon')]:
  pair=[]
  for timestamps in [False,True]:
   j=dict(stage='validation',graph=g,method=method,tile_size=128 if method=='rr_sgd' else None,workgroup_size=128,seed=999,repetition=0,run_id=f'preflight-v55-{method}-{g}-timestamps{int(timestamps)}',version='v5.5')
   progress(j,state='validating')
   pair.append(backend.execute(j,graphs[g],kind='validation',timestamps=timestamps))
  assert pair[0]['initial_positions_sha256']==pair[1]['initial_positions_sha256'] and pair[0]['final_coordinate_text_sha256']==pair[1]['final_coordinate_text_sha256']
  checks.append(dict(method=method,graph=g,outputs_identical=True,complete_timing_fields=10,timestamp_sgd_on_over_off=pair[1]['sgd_time_ms']/pair[0]['sgd_time_ms']))
 save(BASE/'validation/preflight-complete.json',dict(completed=now(),checks=checks,coPapersCiteseer_capacity_precheck=True,formal_runs=0))

def main():
 global _rows,_jobs
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true');a=ap.parse_args()
 BASE.mkdir(parents=True,exist_ok=True);lock=(BASE/'runner.lock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('v5.5 EX2 runner already active')
 lock.write(str(os.getpid()));lock.flush()
 if a.prepare or not (BASE/'config/plan.json').exists():prepare()
 integrity(True)
 backend.BASE=BASE;backend.BIN=BIN;backend.RENDER=RENDER;backend.progress=progress
 backend._old={(r['graph'],r['method'],r['seed'],r['repetition']):r for r in json.loads((BASE/'config/previous-rr-records.json').read_text())}
 validation.SCHEMA_PATH=BASE/'config/schema-v55.json'
 _jobs=json.loads((BASE/'config/jobs.json').read_text());_rows=records()
 assert all(r['status'] in ['success','not_run_capacity'] for r in _rows),'Inspect retained failure before resuming'
 if not a.run:progress(state='prepared',force=True);return
 save(BASE/'process.json',dict(pid=os.getpid(),started=now(),argv=sys.argv))
 graphs={g['name']:g for g in json.loads((BASE/'config/datasets.json').read_text())}
 preflight(graphs);integrity();warmed=set();done={r['run_id'] for r in _rows};session=now().replace(':','').replace('+','_');count=0
 for j in _jobs:
  if j['run_id'] in done:continue
  if shutil.disk_usage(BASE).free<4*1024**3 or memory()['available_bytes']<2*1024**3:progress(state='resource_stop',force=True);return
  key=j['graph'],j['method']
  if key not in warmed:
   integrity();warm=dict(j,seed=999,repetition=0,run_id=f'warm-v55-{session}-{j["graph"]}-{j["method"]}')
   progress(warm,state='warming_up');backend.execute(warm,graphs[j['graph']],kind='warmups');warmed.add(key)
  progress(j);r=backend.execute(j,graphs[j['graph']]);_rows.append(r);done.add(j['run_id']);count+=1
  assert r['bfs_engine']=='auto' and r['gpu_backend']=='Metal'
  if count%50==0:integrity()
  progress()
 assert len(_rows)==1410 and sum(r['status']=='success' for r in _rows)==1395
 integrity(True);progress(state='reporting',force=True)
 subprocess.run([sys.executable,str(ROOT/'experiments/report_v55_ex2.py')],cwd=ROOT,check=True)
 integrity(True)
 save(BASE/'completion.json',dict(completed=now(),version='v5.5',requested_rr_runs=1410,success=1395,not_run_capacity=15,cpu_atomic_new_runs=0,historical_cpu_atomic_records=2820,previous_files_unchanged=True,reports_complete=True,ex1_state='stopped_at_user_request'))
 progress(state='completed_with_capacity_exclusion',force=True)

if __name__=='__main__':
 try:main()
 except Exception as e:
  save(BASE/'failure.json',dict(at=now(),error=repr(e)))
  if _jobs:progress(state='failed',force=True)
  raise
