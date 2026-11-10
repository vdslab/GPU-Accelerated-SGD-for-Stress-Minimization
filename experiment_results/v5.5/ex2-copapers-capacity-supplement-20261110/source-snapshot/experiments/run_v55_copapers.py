#!/usr/bin/env python3
"""Append-only v5.5 EX2 capacity supplement; never rewrite original run records."""
from pathlib import Path
import json,os,shutil,subprocess,time
import run_ex2_rr128 as backend
import run_experiments as validation
from run_experiments import file_sha256 as sha
from run_v5 import now,save
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'experiment_results/v5.5/ex2-rr128-speed-stress-20261110'
BASE=ROOT/'experiment_results/v5.5/ex2-copapers-capacity-supplement-20261110'
BINARY=BASE/'bin/rr-sparse-w128'

def progress(active=None,elapsed=0,state=None):
 rows=list((BASE/'raw').glob('*/record.json'))
 s=dict(updated=now(),state=state or 'running',success=sum(json.loads(p.read_text())['status']=='success' for p in rows),planned=15,active=active,active_elapsed_seconds=elapsed)
 save(BASE/'progress.json',s)
 print(now(),s['state'],f"{s['success']}/15",active['run_id'] if active else '',round(elapsed),flush=True)

def main():
 for d in ['config','bin','logs','validation','warmups','raw','reports','tables','figures','source-snapshot']:(BASE/d).mkdir(parents=True,exist_ok=True)
 envpath=BASE/'config/environment.json'
 if not envpath.exists():
  shutil.copy2(ROOT/'implementations/sparse/sparse-sgd-gpu/target/release/sparse-sgd-gpu',BINARY)
  original_hashes={str(p.relative_to(ROOT)):sha(p) for p in ORIGINAL.rglob('*') if p.is_file() and 'build' not in p.relative_to(ORIGINAL).parts and p.name!='.DS_Store'}
  save(BASE/'config/original-files-sha256.json',original_hashes)
  source_paths=['implementations/sparse/sparse-sgd-gpu/src/gpu.rs','implementations/sparse/sparse-sgd-gpu/src/shader.wgsl','experiments/run_v55_copapers.py']
  source_hashes={}
  for rel in source_paths:
   dest=BASE/'source-snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest);source_hashes[rel]=sha(dest)
  (BASE/'source-snapshot/capacity.patch').write_bytes(subprocess.check_output(['git','diff','--','implementations/sparse/sparse-sgd-gpu'],cwd=ROOT))
  save(envpath,dict(created=now(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),binary_sha256=sha(BINARY),source_sha256=source_hashes,original_environment=str(ORIGINAL/'config/environment.json'),bank_bytes=256*1024**2,bank_count=4,parameters_changed=False))
  shutil.copy2(ORIGINAL/'config/unsupported-jobs.json',BASE/'config/original-unsupported-jobs.json')
  (BASE/'.gitignore').write_text('build/\n')
 env=json.loads(envpath.read_text());assert sha(BINARY)==env['binary_sha256']
 for rel,h in env['source_sha256'].items():assert sha(ROOT/rel)==h
 graphs={g['name']:g for g in json.loads((ORIGINAL/'config/datasets.json').read_text())};g=graphs['coPapersCiteseer'];assert sha(Path(g['path']))==g['sha256']
 backend.BASE=BASE;backend.BIN={'rr_sparse_sgd':BINARY};backend.RENDER=ORIGINAL/'bin/gpu-visualizer';backend.progress=progress
 backend._old={(r['graph'],r['method'],r['seed'],r['repetition']):r for r in json.loads((ORIGINAL/'config/previous-rr-records.json').read_text())}
 validation.SCHEMA_PATH=ORIGINAL/'config/schema-v55.json'
 jobs=[dict(j,run_id=j['run_id'].replace('v55-ex2-','v55-ex2-capacity-'),supersedes_run_id=j['run_id'],implementation_variant='four-256MiB-banks') for j in json.loads((BASE/'config/original-unsupported-jobs.json').read_text())]
 save(BASE/'config/jobs.json',jobs)
 check=BASE/'validation/small-regression.json'
 if not check.exists():
  j=dict(jobs[0],graph='dwt_1005',run_id='capacity-small-regression')
  r=backend.execute(j,graphs['dwt_1005'],kind='validation')
  oldpath=next(p for p in (ORIGINAL/'raw').glob('*/record.json') if (lambda q:q.get('graph')=='dwt_1005' and q.get('method')=='rr_sparse_sgd' and q.get('seed')==0 and q.get('repetition')==0)(json.loads(p.read_text())))
  old=json.loads(oldpath.read_text())
  assert all(r[k]==old[k] for k in ['final_coordinate_text_sha256','stress_value','preprocess_sha256','initial_positions_sha256'])
  save(check,dict(passed=True,unchanged=['saved coordinates','stress','preprocess fingerprint','initial positions']))
 warm=dict(jobs[0],seed=999,run_id='capacity-copapers-warmup')
 backend.execute(warm,g,kind='warmups')
 for j in jobs:
  r=backend.execute(j,g)
  assert r['status']=='success' and r['bfs_engine']=='auto' and r['gpu_backend']=='Metal'
  previous=backend._old[g['name'],j['method'],j['seed'],j['repetition']]
  assert r['final_coordinate_text_sha256']==previous['final_coordinate_text_sha256']
  assert r['stress_value']==previous['stress_value']
  progress()
 for rel,h in json.loads((BASE/'config/original-files-sha256.json').read_text()).items():assert sha(ROOT/rel)==h,rel
 save(BASE/'completion.json',dict(completed=now(),success=15,failed=0,original_files_unchanged=True,all_coordinates_and_stress_equal_previous_rr=True,small_regression_passed=True,combined_rr_success=1410))
 progress(state='completed')
if __name__=='__main__':main()
