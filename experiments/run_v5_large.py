#!/usr/bin/env python3
"""Sequential, resumable v5 large RR Sparse experiment; no time deadline."""
import argparse
import collections
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import time

import psutil
from experiment_plan import Dataset, METHODS, RunDefinition
from run_experiments import file_sha256, git_environment, parse_and_validate_success
from archive_v5_large import archive_text_artifacts
from run_v5 import LARGE, ROOT, VAULT, memory, now, save

BASE = ROOT / 'experiment_results/v5/large'
GIB = 1024**3
BINARY = ROOT / METHODS['rr_sparse_sgd'].binary


def preflight(g, p, limits):
    n, m = g['nodes'], g['edges']
    dense = 8*n*p
    # The graph edges are two-sided constraints. Pivot-only pairs may add
    # at most p(p-1)/2 entries; avoid calling that upper bound a proven failure.
    two_min = 24*m
    reasons = []
    cap = min(limits['max_buffer_size'], limits['max_storage_buffer_binding_size'])
    for name, size in [('one_sided', dense), ('two_sided_lower_bound', two_min), ('positions', 8*(n+2))]:
        if size > cap:
            reasons.append(f'{name}: {size} bytes > storage buffer limit {cap}')
    groups = (n+255)//256
    if groups > limits['max_compute_workgroups_per_dimension']:
        reasons.append(f'vertex dispatch: {groups} > {limits["max_compute_workgroups_per_dimension"]}')
    return dict(graph=g['name'], pivots=p, nodes=n, edges=m,
                one_sided_bytes=dense, two_sided_lower_bound_bytes=two_min,
                vertex_workgroups=groups, hard_limit_reasons=reasons,
                estimated_peak_bytes=192*m+192*n+16*n*p,
                estimate_definition='192m + 192n + 16np; conservative planning estimate, not proven minimum; does not itself skip runs')


def prepare():
    if (BASE/'config/plan.json').exists():
        return
    graphs = sorted([dict(g, path=str(ROOT/'data/v5'/g['path']))
                     for g in json.loads((ROOT/'data/v5/manifest.json').read_text())['graphs']
                     if g['name'] in LARGE], key=lambda g: (g['nodes'], g['name']))
    assert len(graphs) == 14
    limits = json.loads((BASE/'config/device-limits.json').read_text())
    jobs = [dict(graph=g['name'], method='rr_sparse_sgd', pivots=p, seed=s, repetition=0,
                 run_id=f'v5-large-{g["name"]}-p{p}-s{s}-r0')
            for g in graphs for p in [1000,500,200,100,50] for s in range(5)]
    assert len(jobs) == 350
    expected = json.loads((ROOT/'experiment_results/v5/config/environment.json').read_text())['binary_sha256']['rr_sparse_sgd']
    assert file_sha256(BINARY) == expected
    save(BASE/'config/datasets.json', graphs)
    save(BASE/'config/jobs.json', jobs)
    checks = [preflight(g,p,limits) for g in graphs for p in [1000,500,200,100,50]]
    save(BASE/'config/preflight.json', checks)
    save(BASE/'config/plan.json', dict(created=now(), graphs=14, planned_runs=350,
         pivots=[1000,500,200,100,50], seeds=list(range(5)), repetitions=1,
         iterations=15, epsilon=.1, stress_samples=64, stress_seed=0,
         phase_chunks=1, timeout_seconds=None, max_rss_bytes=18*GIB,
         min_available_bytes=512*1024**2, emergency_disk_bytes=GIB,
         pre_run_disk_reserve_bytes=4*GIB,
         memory_policy='Only proven GPU limits skipped before allocation; estimates recorded only. Stop current process on RSS >18 GiB, available RAM <512 MiB sustained 10s, or disk <1 GiB. No wall-clock kill. Each seed independent.',
         warmup='seed999 once per graph/p/session, excluded from formal results; a failed warmup does not replace or skip formal seeds',
         execution_context='User requested Codex execution instead of previously planned manual terminal run. Codex remains resident; environment is not a minimal OS-only state.',
         initial_positions='Native algorithm retained: same seed across p does not guarantee identical initial coordinates because preprocessing consumes RNG. Fingerprints retained; pivot lists not forced to align.'))
    env=git_environment()
    env.update(binary_sha256=expected, memory=memory(), disk_free_bytes=shutil.disk_usage(BASE).free,
               processes=[dict(pid=p.pid, name=p.info['name'], rss=p.info['memory_info'].rss)
                          for p in psutil.process_iter(['name','memory_info']) if p.info['memory_info']])
    save(BASE/'config/environment.json',env)
    snapshots=BASE/'source-snapshot'; snapshots.mkdir(exist_ok=True)
    for source in [Path(__file__), ROOT/'experiments/run_v4.py', ROOT/'experiments/run_experiments.py',
                   ROOT/'implementations/sparse/sparse-sgd-gpu/src/main.rs',
                   ROOT/'implementations/sparse/sparse-sgd-gpu/src/gpu.rs']:
        shutil.copy2(source,snapshots/source.name)
    (snapshots/'git-diff.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT))
    for j in jobs:
        c=next(c for c in checks if c['graph']==j['graph'] and c['pivots']==j['pivots'])
        if c['hard_limit_reasons']:
            save(BASE/'raw'/j['run_id']/'record.json',dict(j,status='preflight_unavailable',
                 executed=False, measured=False, completed_at=now(), preflight=c,
                 error='; '.join(c['hard_limit_reasons'])))


def records():
    return [json.loads(p.read_text()) for p in (BASE/'raw').glob('*/record.json')]


def progress(active=None, elapsed=0, **extra):
    jobs=json.loads((BASE/'config/jobs.json').read_text()); rows=records()
    done={r['run_id'] for r in rows}; remaining=[j for j in jobs if j['run_id'] not in done]
    times=collections.defaultdict(list)
    for r in rows:
        if r['status']=='success': times[(r['graph'],r['pivots'])].append(r['process_wall_seconds']+r.get('archive_seconds',0)+r.get('shared_repack_seconds',0))
    known=sum(statistics.median(times[(j['graph'],j['pivots'])]) for j in remaining if (j['graph'],j['pivots']) in times)
    unknown=sum((j['graph'],j['pivots']) not in times for j in remaining)
    state=dict(updated=now(), total=350, finished=len(rows), success=sum(r['status']=='success' for r in rows),
               preflight_unavailable=sum(r['status']=='preflight_unavailable' for r in rows),
               failed=sum(r['status'] not in ('success','preflight_unavailable') for r in rows),
               remaining=len(remaining), remaining_conditions=len({(j['graph'],j['pivots']) for j in remaining}),
               eta_known_seconds=known, eta_unknown_runs=unknown, active=active,
               active_elapsed_seconds=elapsed, memory=memory(), disk_free_bytes=shutil.disk_usage(BASE).free,
               retried_attempts=len(list((BASE/'validation/retried').glob('*/record.json'))), **extra)
    save(BASE/'progress.json',state)
    msg=(f'{state["updated"]}: 成功{state["success"]}、GPU上限による未実行{state["preflight_unavailable"]}、'
         f'実行失敗{state["failed"]}／計350。残り{len(remaining)}実行・{state["remaining_conditions"]}設定。'
         f'実測済み条件の残り{known/3600:.2f}時間、未推定{unknown}実行。'
         f'現在 {active["run_id"] if active else "なし"}（{elapsed:.0f}秒）。')
    print(msg,flush=True)
    (BASE/'PROGRESS.md').write_text('# v5 超大規模実験の進捗\n\n'+msg+'\n\nGPU上限の未実行は成功・実測失敗と区別する。通常6030件は完了済み。\n')
    for name in ['0_概要.md','4_大規模グラフ実験.md']:
        path=VAULT/name; text=path.read_text()
        begin='<!-- V5_LARGE_PROGRESS_START -->'; end='<!-- V5_LARGE_PROGRESS_END -->'
        block=begin+'\n\n## v5 超大規模の実行状況\n\n'+msg+'\n\n正本: `'+str(BASE)+'`。通常比較の完了分は再実行しない。\n\n'+end
        if begin in text:
            a,b=text.split(begin,1); _,c=b.split(end,1);text=a+block+c
        else:text=text.rstrip()+'\n\n'+block+'\n'
        path.write_text(text)
    return state


def stop(proc):
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def execute(j,g,warmup=False):
    folder=BASE/('warmups' if warmup else 'raw')/j['run_id'];folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'record.json').exists()
    # Preserve evidence of any interrupted prior invocation before retrying.
    if any(folder.iterdir()):
        old=BASE/'validation/interrupted'/f'{j["run_id"]}-{time.time_ns()}'
        old.parent.mkdir(parents=True,exist_ok=True);folder.rename(old);folder.mkdir()
    assert file_sha256(BINARY)==json.loads((BASE/'config/environment.json').read_text())['binary_sha256']
    run=RunDefinition('v5-large',j['run_id'],'benchmark','rr_sparse_sgd',BINARY,
                      Dataset(g['name'],Path(g['path']),g['sha256']),j['seed'],15,.1,j['pivots'],0)
    cmd=[str(BINARY),'--input',g['path'],'--run-id',j['run_id'],'--run-mode','benchmark',
         '--iterations','15','--epsilon','0.1','--seed',str(j['seed']),'--pivots',str(j['pivots']),
         '--output-format','json','--output-dir',str(folder)]
    env=dict(os.environ,SGD_STRESS_SAMPLES='64',SGD_GPU_TIMESTAMPS='0')
    env.pop('SGD_TRACE_DIR',None);env.pop('SGD_TRACE_CHECKPOINTS',None)
    save(folder/'command.json',dict(argv=cmd,started_at=now(),memory_before=memory(),warmup=warmup,
          environment={'SGD_STRESS_SAMPLES':'64','SGD_GPU_TIMESTAMPS':'0'}))
    started=time.monotonic();peak=0;reason=None;low_since=None;last=0
    plan=json.loads((BASE/'config/plan.json').read_text())
    with (folder/'stdout.json').open('w') as out,(folder/'stderr.log').open('w') as err, (folder/'resources.jsonl').open('w') as log:
        proc=subprocess.Popen(['/usr/bin/time','-l',*cmd],cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
        save(BASE/'active-process.json',dict(pid=proc.pid,run=j,warmup=warmup,started_at=now()))
        try:
            while proc.poll() is None:
                rss=0
                try:
                    for child in psutil.Process(proc.pid).children(recursive=True):
                        try:rss+=child.memory_info().rss
                        except psutil.Error:pass
                except psutil.Error:pass
                peak=max(peak,rss);current=time.monotonic();mem=memory()
                if mem['available_bytes'] < plan['min_available_bytes']:
                    low_since=low_since or current
                else:low_since=None
                if rss>plan['max_rss_bytes']:reason='rss_capacity_limit'
                elif low_since and current-low_since>=10:reason='system_memory_capacity_limit'
                elif shutil.disk_usage(BASE).free<plan['emergency_disk_bytes']:reason='disk_capacity_limit'
                if current-last>=30:
                    log.write(json.dumps(dict(elapsed=current-started,rss=rss,**mem))+'\n');log.flush()
                    progress(dict(j,warmup=warmup),current-started);last=current
                if reason:stop(proc);break
                time.sleep(.5)
        except BaseException:
            if proc.poll() is None:stop(proc)
            raise
    wall=time.monotonic()-started; stderr=(folder/'stderr.log').read_text(errors='replace')
    match=re.search(r'(\d+)\s+maximum resident set size',stderr)
    if match:peak=max(peak,int(match[1]))
    row=dict(j,status=reason or 'error',executed=True,measured=True,warmup=warmup,
             process_wall_seconds=wall,peak_rss_bytes=peak,returncode=proc.returncode,
             completed_at=now(),memory_after=memory(),binary_sha256=file_sha256(BINARY))
    if not reason:
        try:
            native=parse_and_validate_success(run,subprocess.CompletedProcess(cmd,proc.returncode,(folder/'stdout.json').read_text(),stderr))
            assert native['nodes']==g['nodes'] and native['pivots']==j['pivots']
            assert native['rr_phase_chunks']==1 and native['iterations']==15
            assert native['stress_kind']=='sampled' and native['stress_samples']==64 and native['stress_seed']==0
            assert native['total_compute_time_ms']>0
            row.update(native);row.update(j);row['status']='success'
        except Exception as error:row['error']=repr(error)
    if row['status']!='success':row.setdefault('error',reason or stderr[-3000:])
    archive_text_artifacts(folder,row);save(folder/'record.json',row)
    save(BASE/'active-process.json',dict(pid=None,completed_at=now(),last_run=j))
    if row.get('archive_status')!='success':
        if row.get('archive_error'):raise RuntimeError(row['archive_error'])
    # Dense Orkut layouts repeat the same >1 GiB edge text in every artifact.
    # Preserve standalone gzip recovery while APFS shares that immutable member.
    # Both archival stages are after native measurement, including Stress.
    if j['graph']=='com-Orkut' and row['status']=='success':
        from share_v5_large_edges import repack
        repack(folder)
        row=json.loads((folder/'record.json').read_text())
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--progress',action='store_true')
    ap.add_argument('--reuse-recent-warmups',action='store_true',help='Capacity-only resume: reuse validated same-binary warmup completed within 30 minutes')
    args=ap.parse_args()
    BASE.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'runner.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
    prepare()
    if args.prepare or args.progress:progress();return
    graphs={g['name']:g for g in json.loads((BASE/'config/datasets.json').read_text())}
    jobs=json.loads((BASE/'config/jobs.json').read_text()); verified=set();warmed=set();session=str(time.time_ns())
    if args.reuse_recent_warmups:
        expected=json.loads((BASE/'config/environment.json').read_text())['binary_sha256'];reused=[]
        for path in (BASE/'warmups').glob('*/record.json'):
            row=json.loads(path.read_text())
            age=datetime.fromisoformat(now()).timestamp()-datetime.fromisoformat(row['completed_at']).timestamp()
            if (0<=age<=1800 and row['status']=='success' and row.get('archive_status')=='success'
                and row.get('binary_sha256')==expected and row['seed']==999
                and row.get('iterations')==15 and row.get('stress_samples')==64
                and row.get('stress_seed')==0 and row.get('rr_phase_chunks')==1):
                warmed.add((row['graph'],row['pivots']))
                reused.append(dict(graph=row['graph'],pivots=row['pivots'],source_record=str(path),age_seconds=age))
        save(BASE/'validation'/f'warmup-reuse-{session}.json',dict(at=now(),reason='Capacity-only resume; measurement binary and parameters unchanged',reused=reused))
    save(BASE/'process.json',dict(pid=os.getpid(),started_at=now(),log=str(BASE/'logs/runner.log')))
    deferred=[];blocked_graphs=set()
    def defer(job,reason,reserve):
        blocked_graphs.add(job['graph'])
        deferred.append(dict(run=job,reason=reason,required_disk_bytes=reserve,
                             available_disk_bytes=shutil.disk_usage(BASE).free,memory=memory(),at=now()))
        save(BASE/'validation'/f'schedule-deferred-{session}.json',dict(at=now(),policy='Pending only; preserve each graph pivot/seed order, process other approved graphs if resource reserve fits',jobs=deferred))
    for j in jobs:
        if (BASE/'raw'/j['run_id']/'record.json').exists():continue
        g=graphs[j['graph']]
        # Reserve for uncompressed initial/final coordinates, vertex map and XZ.
        # Each native layout contains every graph edge, both initial and final.
        # Include dense graph edge text, not only vertex coordinates.
        reserve=4*GIB+160*g['nodes']+48*g['edges']
        if g['name'] in blocked_graphs:
            defer(j,'earlier condition of this graph deferred; retain pivot/seed order',reserve);continue
        if shutil.disk_usage(BASE).free<reserve or memory()['available_bytes']<2*GIB:
            defer(j,'pre_run_resource_reserve',reserve);continue
        if g['name'] not in verified:
            assert file_sha256(Path(g['path']))==g['sha256'];verified.add(g['name'])
        key=(g['name'],j['pivots'])
        if key not in warmed:
            w=dict(j,seed=999,run_id=j['run_id']+f'-warmup-{session}')
            execute(w,g,warmup=True);warmed.add(key)
        if shutil.disk_usage(BASE).free<reserve or memory()['available_bytes']<2*GIB:
            defer(j,'post_warmup_resource_reserve',reserve);continue
        execute(j,g);progress()
    if deferred:progress(paused_reason='remaining graphs deferred for resource reserves',deferred_jobs=len(deferred),deferred_graphs=sorted(blocked_graphs))
    else:progress(measurement_complete=True)


if __name__=='__main__': main()
