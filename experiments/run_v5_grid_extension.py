"""Sequential added grid/workgroup measurements. Frozen scope, no time deadline."""
import argparse
import collections
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time

import psutil
import shutil

from archive_v5_large import archive_text_artifacts, digest
from experiment_plan import Dataset, METHODS, RunDefinition
from run_experiments import file_sha256, parse_and_validate_success
from run_v5 import ROOT, VAULT, now, save, memory
from prepare_v5_grid_extension import BASE

GIB=2**30


def read(path):return json.loads(path.read_text())
def completed():
    return [read(p) for p in (BASE/'raw').glob('*/record.json') if read(p).get('archive_status')=='success']


def progress(active=None,elapsed=0,**extra):
    jobs=read(BASE/'config/jobs.json');rows=completed();done={r['run_id'] for r in rows};pending=[j for j in jobs if j['run_id'] not in done]
    times=collections.defaultdict(list)
    for r in rows:
        if r['status']=='success':times[(r['stage'],r['graph'],r['tile_size'],r['workgroup_size'],r['pivots'])].append(r['process_wall_seconds']+r.get('archive_seconds',0)+r.get('shared_repack_seconds',0))
    known=0;unknown=0
    for j in pending:
        values=times.get((j['stage'],j['graph'],j['tile_size'],j['workgroup_size'],j['pivots']))
        if values:known+=statistics.median(values)
        else:unknown+=1
    state=dict(updated=now(),planned=810,finished=len(rows),success=sum(r['status']=='success' for r in rows),failed=sum(r['status']!='success' for r in rows),remaining=len(pending),remaining_settings=len({(j['stage'],j['graph'],j['tile_size'],j['workgroup_size'],j['pivots']) for j in pending}),
               active=active,active_elapsed_seconds=elapsed,eta_known_seconds=known,eta_unknown_runs=unknown,memory=memory(),disk_free_bytes=shutil.disk_usage(BASE).free,**extra)
    save(BASE/'progress.json',state)
    message=f'{state["updated"]}: 成功{state["success"]}・実測失敗{state["failed"]}／810。残り{len(pending)}実行・{state["remaining_settings"]}設定。実測済み条件の残り{known/60:.1f}分、未推定{unknown}実行。現在 {active["run_id"] if active else "なし"} ({elapsed:.0f}秒)。'
    print(message,flush=True);(BASE/'PROGRESS.md').write_text('# 追加実験の進捗\n\n'+message+'\n')
    note=VAULT/'8_追加実験_gridとSparse設定.md';text=note.read_text();start='<!-- GRID_EXTENSION_PROGRESS_START -->';end='<!-- GRID_EXTENSION_PROGRESS_END -->';block=start+'\n\n## 実行状況\n\n'+message+'\n\n'+end
    if start in text:a,b=text.split(start,1);_,c=b.split(end,1);text=a+block+c
    else:text=text.rstrip()+'\n\n'+block+'\n'
    note.write_text(text)
    return state


def binary(job):
    if job['stage']=='full':return ROOT/METHODS['rr_sgd'].binary,read(BASE/'config/plan.json')['full_binary_sha256']
    v=next(v for v in read(BASE/'config/binaries.json') if v['workgroup_size']==job['workgroup_size'])
    return Path(v['binary']),v['sha256']


def archive(folder,row):
    # Recover publication interrupted between completed XZ member and manifest.
    import lzma
    manifest=read(folder/'artifact-archive.json') if (folder/'artifact-archive.json').exists() else dict(files=[],errors=[])
    existing={e['original_path'] for e in manifest['files']}
    redundant=[]
    for source in folder.glob('*.txt'):
        target=Path(str(source)+'.xz')
        if not target.exists():continue
        raw_sha,length=digest(source);assert digest(target,lzma.open)==(raw_sha,length)
        if str(source) not in existing:
            h,size=digest(target);manifest['files'].append(dict(original_path=str(source),compressed_path=str(target),original_sha256=raw_sha,decompressed_sha256=raw_sha,original_bytes=length,compressed_bytes=size,compressed_sha256=h,compression='xz',preset=0,xz_crc_validated=True))
        redundant.append(source)
    if manifest['files']:save(folder/'artifact-archive.json',manifest)
    archive_text_artifacts(folder,row)
    for source in redundant:source.unlink(missing_ok=True)
    return finish_storage(folder)


def finish_storage(folder):
    row=read(folder/'record.json')
    if row['status']=='success' and row['graph']=='coPapersCiteseer':
        # This is storage after native timing: independently decodable gzip files
        # physically share only immutable edge-list blocks on APFS.
        from share_v5_large_edges import repack
        repack(folder,pool=BASE/'shared-edges')
        row=read(folder/'record.json')
    return row


def execute(job,graph,warmup=False):
    folder=BASE/('warmups' if warmup else 'raw')/job['run_id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'record.json').exists() and read(folder/'record.json').get('archive_status')=='success':return finish_storage(folder)
    if (folder/'measurement.json').exists():
        row=read(folder/'measurement.json');return archive(folder,row)
    if any(folder.iterdir()):
        target=BASE/'validation/interrupted'/f'{job["run_id"]}-{time.time_ns()}';target.parent.mkdir(parents=True,exist_ok=True);folder.rename(target);folder.mkdir()
    exe,expected=binary(job);assert file_sha256(exe)==expected
    cmd=[str(exe),'--input',graph['path'],'--run-id',job['run_id'],'--run-mode','benchmark','--iterations','15','--epsilon','0.1','--seed',str(job['seed']),'--output-format','json','--output-dir',str(folder)]
    if job['stage']=='full':cmd+=['--tile-size',str(job['tile_size']),'--workgroup-size','1024','--coordinate-memory','shared','--phase-chunks','1']
    else:cmd+=['--pivots',str(job['pivots'])]
    env=dict(os.environ,SGD_GPU_TIMESTAMPS='0',SGD_STRESS_SAMPLES='256');env.pop('SGD_TRACE_DIR',None);env.pop('SGD_TRACE_CHECKPOINTS',None)
    save(folder/'command.json',dict(argv=cmd,binary_sha256=expected,workgroup_size=job['workgroup_size'],started_at=now(),memory_before=memory(),warmup=warmup,environment={'SGD_GPU_TIMESTAMPS':'0','SGD_STRESS_SAMPLES':'256'}))
    started=time.monotonic();peak=0;reason=None;low=None;last=-31
    with (folder/'stdout.json').open('w') as out,(folder/'stderr.log').open('w') as err,(folder/'resources.jsonl').open('w') as resource:
        proc=subprocess.Popen(['/usr/bin/time','-l',*cmd],cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
        save(BASE/'active-process.json',dict(pid=proc.pid,job=job,warmup=warmup,started_at=now()))
        try:
            while proc.poll() is None:
                rss=0
                try:
                    for child in psutil.Process(proc.pid).children(recursive=True):
                        try:rss+=child.memory_info().rss
                        except psutil.Error:pass
                except psutil.Error:pass
                peak=max(peak,rss);t=time.monotonic();mem=memory();free=shutil.disk_usage(BASE).free
                low=(low or t) if mem['available_bytes']<512*2**20 else None
                if rss>18*GIB:reason='rss_capacity_limit'
                elif low and t-low>=10:reason='available_ram_capacity_limit'
                elif free<GIB:reason='disk_capacity_limit'
                if t-last>=30:
                    resource.write(json.dumps(dict(elapsed=t-started,rss_bytes=rss,disk_free_bytes=free,**mem))+'\n');resource.flush();progress(dict(job,warmup=warmup),t-started);last=t
                if reason:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    break
                time.sleep(.5)
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM);proc.wait()
            raise
    wall=time.monotonic()-started;stderr=(folder/'stderr.log').read_text(errors='replace');match=re.search(r'(\d+)\s+maximum resident set size',stderr)
    if match:peak=max(peak,int(match[1]))
    row=dict(job,status=reason or 'error',warmup=warmup,process_wall_seconds=wall,peak_rss_bytes=peak,returncode=proc.returncode,binary_sha256=expected,completed_at=now(),memory_after=memory())
    if not reason:
        try:
            definition=RunDefinition('v5-grid-extension',job['run_id'],'benchmark',job['method'],exe,Dataset(graph['name'],Path(graph['path']),graph['sha256']),job['seed'],15,.1,job['pivots'],job['repetition'])
            native=parse_and_validate_success(definition,subprocess.CompletedProcess(cmd,proc.returncode,(folder/'stdout.json').read_text(),stderr))
            assert native['nodes']==graph['nodes'] and native['rr_phase_chunks']==1
            assert native['stress_kind']==('exact' if graph['nodes']<=8000 else 'sampled')
            if graph['nodes']>8000:assert native['stress_samples']==256 and native['stress_seed']==0
            if job['stage']=='full':assert native['rr_workgroup_size']==1024 and native['rr_tile_size']==job['tile_size']
            row.update(native);row.update(job);row['status']='success';row['sgd_mean_iteration_ms']=native['sgd_time_ms']/15
            row['gpu_execution_ms']=native['gpu_prepare_time_ms']+native['sgd_time_ms']+native['readback_time_ms'];row['gpu_execution_with_upload_ms']=row['gpu_execution_ms']+native['upload_time_ms']
        except Exception as error:row['error']=repr(error)
    if row['status']!='success':row.setdefault('error',reason or stderr[-4000:])
    save(folder/'measurement.json',row);row=archive(folder,row)
    save(BASE/'active-process.json',dict(pid=None,completed_at=now(),last_job=job))
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['all','full','sparse'],default='all');ap.add_argument('--progress',action='store_true');ap.add_argument('--max-runs',type=int);args=ap.parse_args()
    lock=(BASE/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
    if args.progress:progress();return
    assert read(BASE/'validation/workgroup-equivalence.json')['status']=='passed'
    known_binaries={str(ROOT/METHODS['rr_sgd'].binary),str(ROOT/METHODS['rr_sparse_sgd'].binary)}|{v['binary'] for v in read(BASE/'config/binaries.json')}
    for process in psutil.process_iter(['cmdline']):
        argv=process.info['cmdline'] or []
        assert not (argv and argv[0] in known_binaries),f'Native GPU process still active: {process.pid}'
    save(BASE/'process.json',dict(pid=os.getpid(),started_at=now(),stage=args.stage,log=str(BASE/'logs/runner.log')))
    jobs=read(BASE/'config/jobs.json');graphs={g['name']:g for g in read(BASE/'config/datasets.json')};session=str(time.time_ns());warmed=set();verified=set();blocked=set();deferred=[];count=0
    for job in jobs:
        if args.stage!='all' and job['stage']!=args.stage:continue
        path=BASE/'raw'/job['run_id']/'record.json'
        if path.exists() and read(path).get('archive_status')=='success':
            if job['graph']=='coPapersCiteseer':finish_storage(path.parent)
            continue
        if (BASE/'pause-request.json').exists():progress(paused_reason='Pause requested between native runs');return
        key=(job['stage'],job['graph'])
        if key in blocked:deferred.append(job);continue
        if memory()['available_bytes']<2*GIB or shutil.disk_usage(BASE).free<4*GIB:
            progress(paused_reason='Pre-run resource reserve: RAM2GiB / disk4GiB');return
        graph=graphs[job['graph']]
        if job['graph'] not in verified:assert file_sha256(Path(graph['path']))==graph['sha256'];verified.add(job['graph'])
        config=(job['stage'],job['graph'],job['tile_size'],job['workgroup_size'],job['pivots'])
        if config not in warmed:
            warm=dict(job,seed=999,repetition=0,run_id=job['run_id']+'-warmup-'+session);row=execute(warm,graph,True)
            if row['status']!='success':
                blocked.add(key);deferred.append(job);save(BASE/'validation'/f'deferred-{session}.json',dict(at=now(),reason='Warmup failed; formal jobs remain pending, not inferred failed',blocked=list(blocked),first_job=job,warmup_record=str(BASE/'warmups'/warm['run_id']/'record.json')));continue
            warmed.add(config)
        if memory()['available_bytes']<2*GIB or shutil.disk_usage(BASE).free<4*GIB:
            progress(paused_reason='Post-warmup resource reserve: RAM2GiB / disk4GiB');return
        row=execute(job,graph);progress();count+=1
        if row['status']!='success':
            blocked.add(key);save(BASE/'validation'/f'formal-failure-{job["run_id"]}.json',row)
        if args.max_runs and count>=args.max_runs:progress(paused_reason='Requested maximum count reached between runs');return
    state=progress(measurement_complete=not bool(deferred),deferred_jobs=len(deferred),deferred_graphs=list(blocked))
    if state['remaining']:progress(paused_reason='Remaining jobs pending after failed warmups or resource failures',deferred_jobs=len(deferred),deferred_graphs=list(blocked))


if __name__=='__main__':main()
