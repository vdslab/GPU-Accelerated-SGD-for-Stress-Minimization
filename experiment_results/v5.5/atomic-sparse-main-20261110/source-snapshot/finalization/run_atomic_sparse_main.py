#!/usr/bin/env python3
"""All 780 v5 Atomic Sparse conditions, freshly measured on merged main."""
import argparse,collections,fcntl,json,os,re,shutil,signal,subprocess,sys,time
from pathlib import Path
import psutil
from experiment_plan import Dataset,RunDefinition
from run_experiments import file_sha256 as sha,parse_and_validate_success
from archive_v5_large import archive_text_artifacts
from run_v5 import ROOT,now,save,memory

SOURCE=ROOT.parent/'sgd-v55-main'
BASE=ROOT/'experiment_results/v5.5/atomic-sparse-main-20261110'
OLD=ROOT/'experiment_results/v5'
BINARY=BASE/'bin/atomic-sparse'

def records():return [json.loads(p.read_text()) for p in (BASE/'raw').glob('*/record.json')]

def progress(active=None,elapsed=0,state='running'):
    rows=records();counts=collections.Counter(r['status'] for r in rows)
    s=dict(at=now(),state=state,planned=780,finished=len(rows),statuses=dict(counts),remaining=780-len(rows),active=active,elapsed=elapsed,memory=memory())
    save(BASE/'progress.json',s)
    (BASE/'PROGRESS.md').write_text('# v5.5 Atomic Sparse\n\n'+json.dumps(s,ensure_ascii=False,indent=2)+'\n')
    print(now(),state,len(rows),'/780',active or '',round(elapsed),flush=True)

def prepare():
    if (BASE/'config/environment.json').exists():return
    for d in ['config','bin','logs','raw','warmups','validation','source-snapshot','reports','tables','figures']:(BASE/d).mkdir(parents=True,exist_ok=True)
    (BASE/'.gitignore').write_text('*.tmp\n')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    assert head==subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','diff','HEAD','--','implementations','shared'],cwd=SOURCE)
    target=ROOT/'implementations/sparse/vram-lock-sparse/target'
    cmd=['cargo','build','--release','--locked','--manifest-path',str(SOURCE/'implementations/sparse/vram-lock-sparse/Cargo.toml'),'--target-dir',str(target)]
    save(BASE/'logs/build-command.json',dict(argv=cmd,source_commit=head,at=now()))
    with (BASE/'logs/build.log').open('w') as f:subprocess.run(cmd,cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,check=True)
    shutil.copy2(target/'release/vram-lock-sparse',BINARY)
    jobs=[dict(j,run_id=j['run_id'].replace('v5-comparison-','v55-atomic-sparse-'),previous_run_id=j['run_id']) for j in json.loads((OLD/'config/jobs.json').read_text()) if j['stage']=='comparison' and j['method']=='atomic_sparse_sgd' and j['graph'] not in {'com-DBLP','web-NotreDame','com-Amazon','coPapersCiteseer'}]
    assert len(jobs)==780 and len({j['graph'] for j in jobs})==52
    graphs=[g for g in json.loads((OLD/'config/datasets.json').read_text()) if g['name'] in {j['graph'] for j in jobs}]
    assert len(graphs)==52
    for g in graphs:assert sha(Path(g['path']))==g['sha256']
    controls=[]
    for j in jobs:
        p=OLD/'raw/comparison'/j['previous_run_id']/'record.json';r=json.loads(p.read_text());assert r['status']=='success'
        controls.append(dict(r,reference_record=str(p),reference_record_sha256=sha(p)))
    save(BASE/'config/jobs.json',jobs);save(BASE/'config/datasets.json',graphs);save(BASE/'config/v5-controls.json',controls)
    sources=subprocess.check_output(['git','ls-files','implementations/sparse/vram-lock-sparse','shared/experiment-common','shared/gpu-timing'],cwd=SOURCE,text=True).splitlines()
    hashes={}
    for rel in sources:
        dst=BASE/'source-snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/rel,dst);hashes[rel]=sha(dst)
    for name in ['run_atomic_sparse_main.py','report_atomic_sparse_main.py','run_experiments.py','experiment_plan.py','record_metadata.py','archive_v5_large.py','run_v5.py','schema/experiment-record-v1.json']:
        dst=BASE/'source-snapshot/experiments'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'experiments'/name,dst)
    save(BASE/'config/plan.json',dict(version='v5.5',graphs=52,runs=780,seeds=list(range(5)),repetitions=3,iterations=15,epsilon=.1,stress='auto exact <=8000; sampled256 otherwise; evaluation seed0',gpu_timestamps=False,warmup='graph/session seed999 excluded',timeout_seconds=None,update_order='main default GPU permutation; cpu_shuffle=false', bfs='auto (flat CSR CPU)', pivots=200,fresh_measurements=True,pilot_records_reused=0,ex5='paused',resource_policy='sequential; between-run >=2GiB available RAM and >=4GiB free disk; running RSS<=18GiB, available RAM>=512MiB (10s sustained), disk>=1GiB',primary_timing='total_compute_time_ms; asynchronous GPU completion wait included',secondary_timing='sgd_time_ms + readback_time_ms; raw sgd_time_ms measures submissions in new main'))
    save(BASE/'config/environment.json',dict(at=now(),source_root=str(SOURCE),merged_main_commit=head,binary_sha256=sha(BINARY),source_sha256=hashes,memory=memory(),machine=subprocess.check_output(['system_profiler','SPHardwareDataType','SPDisplaysDataType'],text=True)))

def integrity(inputs=False):
    env=json.loads((BASE/'config/environment.json').read_text());assert sha(BINARY)==env['binary_sha256']
    for rel,h in env['source_sha256'].items():assert sha(SOURCE/rel)==h
    if inputs:
        for g in json.loads((BASE/'config/datasets.json').read_text()):assert sha(Path(g['path']))==g['sha256']
        for r in json.loads((BASE/'config/v5-controls.json').read_text()):assert sha(Path(r['reference_record']))==r['reference_record_sha256']

def execute(j,g,warm=False):
    folder=BASE/('warmups' if warm else 'raw')/j['run_id']
    if (folder/'record.json').exists():return json.loads((folder/'record.json').read_text())
    folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()):
        dest=BASE/'validation/interrupted'/f"{j['run_id']}-{time.time_ns()}";dest.parent.mkdir(parents=True,exist_ok=True);folder.rename(dest);folder.mkdir()
    env=dict(os.environ,SGD_STRESS_MODE='auto',SGD_STRESS_SAMPLES='256',SGD_GPU_TIMESTAMPS='0')
    for k in ['SGD_TRACE_DIR','SGD_TRACE_CHECKPOINTS']:env.pop(k,None)
    cmd=[str(BINARY),'--input',g['path'],'--iterations','15','--epsilon','0.1','--seed',str(j['seed']),'--run-id',j['run_id'],'--run-mode','benchmark','--output-format','json','--output-dir',str(folder),'--pivots',str(min(200,g['nodes']))]
    save(folder/'command.json',dict(argv=cmd,at=now(),environment={k:env[k] for k in ['SGD_STRESS_MODE','SGD_STRESS_SAMPLES','SGD_GPU_TIMESTAMPS']},memory_before=memory()))
    run=RunDefinition('v55-atomic-sparse',j['run_id'],'benchmark','atomic_sparse_sgd',BINARY,Dataset(g['name'],Path(g['path']),g['sha256']),j['seed'],15,.1,min(200,g['nodes']),j['repetition'])
    start=time.monotonic();last=start;peak=0;low=None;reason=None
    with (folder/'stdout.json').open('w') as out,(folder/'stderr.log').open('w') as err,(folder/'resources.jsonl').open('w') as resources:
        p=subprocess.Popen(['/usr/bin/time','-l',*cmd],cwd=SOURCE,env=env,stdout=out,stderr=err,start_new_session=True)
        save(BASE/'active-process.json',dict(pid=p.pid,run=j,warmup=warm,at=now()))
        try:
            while p.poll() is None:
                rss=0
                try:rss=sum(c.memory_info().rss for c in psutil.Process(p.pid).children(recursive=True));peak=max(peak,rss)
                except psutil.Error:pass
                current=time.monotonic();mem=memory()
                if mem['available_bytes']<512*1024**2:low=low or current
                else:low=None
                if rss>18*1024**3:reason='rss_capacity_stop'
                elif low and current-low>10:reason='system_memory_capacity_stop'
                elif shutil.disk_usage(BASE).free<1024**3:reason='disk_capacity_stop'
                if current-last>30:
                    resources.write(json.dumps(dict(elapsed=current-start,rss=rss,**mem))+'\n');resources.flush();progress(j['run_id'],current-start);last=current
                if reason:
                    os.killpg(p.pid,signal.SIGTERM);p.wait();break
                time.sleep(.2)
        except BaseException:
            os.killpg(p.pid,signal.SIGTERM);p.wait();raise
    stderr=(folder/'stderr.log').read_text()
    row=dict(j,version='v5.5',status=reason or 'error',warmup=warm,process_wall_seconds=time.monotonic()-start,peak_rss_sampled_bytes=peak,returncode=p.returncode,at=now(),binary_sha256=sha(BINARY))
    try:
        if reason:raise RuntimeError(reason)
        native=parse_and_validate_success(run,subprocess.CompletedProcess(cmd,p.returncode,(folder/'stdout.json').read_text(),stderr))
        expected='exact' if native['nodes']<=8000 else 'sampled'
        assert native['stress_kind']==expected and native['gpu_backend']=='Metal'
        if expected=='sampled':assert native['stress_samples']==256 and native['stress_seed']==0
        assert native['attempted_updates']==native['completed_updates']==15*native['constraints']
        assert native['retry_failures']==0
        assert native['git_commit']==json.loads((BASE/'config/environment.json').read_text())['merged_main_commit']
        assert not native['git_dirty']
        assert native['nodes']==g['nodes'] and native['edges']==g['edges']
        if not warm:
            previous=json.loads((OLD/'raw/comparison'/j['previous_run_id']/'record.json').read_text())
            for field in ['input_sha256','initial_positions_sha256','preprocess_sha256','nodes','edges','constraints','stress_kind','stress_samples','stress_seed']:
                assert native[field]==previous[field],(field,native[field],previous[field])
        row.update(native);row.update(j)
        row['completed_sgd_readback_ms']=native['sgd_time_ms']+(native.get('readback_time_ms') or 0)
        row['initial_positions_path']=str(next(folder.glob('*.initial.txt')))
        row['layout_status']='success';row['stress_status']='success'
        match=re.search(r'(\d+)\s+maximum resident set size',stderr)
        if match:row['peak_rss_bytes']=int(match[1])
    except Exception as e:row['error']=repr(e)
    save(folder/'record.json',row)
    if row['status']=='success':archive_text_artifacts(folder,row)
    save(BASE/'active-process.json',dict(pid=None,last_run=j,at=now()))
    assert row['status']=='success',row
    return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare();integrity(True)
        if a.prepare:progress(state='prepared');return
        graphs={g['name']:g for g in json.loads((BASE/'config/datasets.json').read_text())}
        jobs=json.loads((BASE/'config/jobs.json').read_text());done={r['run_id'] for r in records()}
        assert all(r['status']=='success' for r in records()),'Inspect retained failed measurement before resume'
        warmed=set();session=str(time.time_ns())
        for j in jobs:
            if j['run_id'] in done:continue
            if shutil.disk_usage(BASE).free<4*1024**3 or memory()['available_bytes']<2*1024**3:
                progress(state='resource_stop');raise RuntimeError('Resource reserve unavailable')
            g=graphs[j['graph']]
            if g['name'] not in warmed:
                integrity();w=dict(j,seed=999,run_id=f"warm-v55-atomic-sparse-{g['name']}-{session}")
                progress(w['run_id'],state='warming_up');execute(w,g,True);warmed.add(g['name'])
            progress(j['run_id']);execute(j,g)
        integrity(True);progress(state='reporting')
        subprocess.run([sys.executable,str(ROOT/'experiments/report_v55_atomic_full.py')],cwd=ROOT,check=True)
        integrity(True)
        save(BASE/'completion.json',dict(at=now(),success=780,failed=0,graphs=52,previous_records_unchanged=True,reports_complete=True,source_commit=json.loads((BASE/'config/environment.json').read_text())['merged_main_commit']))
        progress(state='completed')

if __name__=='__main__':main()
