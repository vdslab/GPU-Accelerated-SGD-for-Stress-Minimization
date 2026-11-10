#!/usr/bin/env python3
"""v5.6 large graphs: existing four banks enlarged, CPU preprocessing, EX5 protocol."""
import argparse
import collections
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import psutil
import run_v5_large as backend
import run_experiments as validation
from run_v5 import now, save
from run_experiments import file_sha256 as sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiment_results/v5.6/large'
OLD=ROOT/'experiment_results/v5/large'
BINARY=BASE/'bin/rr-sparse'
GIB=1024**3
LAST_PRINT=0
PREVIOUS={}
SCOPE={"germany_osm", "italy_osm"}


def records():
    return [r for p in (BASE/'raw').glob('*/record.json') if (r:=json.loads(p.read_text()))['graph'] in SCOPE]


def prepare():
    if (BASE/'config/plan.json').exists():return
    for d in ['bin','config','raw','warmups','logs','tables','reports','source-snapshot','validation']:
        (BASE/d).mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'implementations/sparse/sparse-sgd-gpu/target/release/sparse-sgd-gpu',BINARY)
    limits=json.loads((BASE/'logs/device-limits.json').read_text())
    cap=min(limits['max_buffer_size'],limits['max_storage_buffer_binding_size'])
    bank_entries=cap//8
    graphs=[g for g in json.loads((OLD/'config/datasets.json').read_text()) if g['name'] in SCOPE]
    old_jobs=[j for j in json.loads((OLD/'config/jobs.json').read_text()) if j['graph'] in SCOPE]
    jobs=[dict(j,run_id=j['run_id'].replace('v5-large-','v56-large-'),
        previous_run_id=j['run_id'],version='v5.6',experiment='large',bfs_engine='cpu') for j in old_jobs]
    checks=[]
    for g in graphs:
        for p in [1000,500,200,100,50]:
            n,m=g['nodes'],g['edges'];reasons=[]
            if n*p>4*bank_entries:reasons.append('one-sided table exceeds four adapter-sized banks')
            if 20*m>cap:reasons.append('two-sided edge lower bound exceeds adapter binding')
            if 8*(n+2)>cap:reasons.append('positions exceed adapter binding')
            groups=(n+127)//128;dim=limits['max_compute_workgroups_per_dimension']
            if groups>dim*dim:reasons.append('vertex dispatch exceeds adapter 2D capacity')
            # These two dense arrays coexist in prepare_with_backend; this is an
            # exact lower bound, not a speculative peak-RSS estimate.
            cpu_dense_lower_bound=12*n*p
            status='preflight_unavailable' if reasons else 'preflight_resource_unavailable' if cpu_dense_lower_bound>18*GIB else None
            checks.append(dict(graph=g['name'],pivots=p,nodes=n,edges=m,one_sided_bytes=8*n*p,
                cpu_dense_arrays_lower_bound_bytes=cpu_dense_lower_bound,
                resource_reason='coexisting u32 distances + 8-byte one-sided table alone exceed 18 GiB RSS policy' if status=='preflight_resource_unavailable' else None,
                hard_limit_reasons=reasons,status=status))
    check_by_key={(c['graph'],c['pivots']):c for c in checks}
    for j in jobs:
        c=check_by_key[j['graph'],j['pivots']]
        if c['status']:
            save(BASE/'raw'/j['run_id']/'record.json',dict(j,status=c['status'],executed=False,measured=False,preflight=c,completed_at=now()))
    # Verify the graph requested by the user first; retain original descending-p
    # order for all remaining graph/config combinations after this probe group.
    priority=[j for j in jobs if j['graph']=='germany_osm' and j['pivots']==50]
    ordered=priority+[j for j in jobs if j not in priority]
    save(BASE/'config/jobs.json',ordered)
    save(BASE/'config/datasets.json',graphs)
    save(BASE/'config/preflight.json',checks)
    save(BASE/'config/device-limits.json',dict(limits,bank_entries=bank_entries,four_bank_capacity_bytes=32*bank_entries))
    save(BASE/'config/schema.json',json.loads((ROOT/'experiment_results/v5.6/rr-sparse-lux-stanford-20261110/config/schema.json').read_text()))
    files=subprocess.check_output(['git','ls-files','implementations/sparse/sparse-sgd-gpu','shared'],cwd=ROOT,text=True).splitlines()
    files+=['experiments/run_v56_large.py','experiments/run_v5_large.py','experiments/archive_v5_large.py','experiments/share_v5_large_edges.py','experiments/run_experiments.py']
    hashes={}
    for rel in files:
        src=ROOT/rel
        if not src.is_file():continue
        dest=BASE/'source-snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);hashes[rel]=sha(src)
    (BASE/'source-snapshot/implementation.patch').write_bytes(subprocess.check_output(['git','diff','--binary','--','implementations/sparse/sparse-sgd-gpu'],cwd=ROOT))
    shutil.copy2('/tmp/rr-sparse-v56-four-bank-tests.log',BASE/'validation/cargo-test.log')
    previous_paths=list((OLD/'raw').glob('*/record.json'))
    save(BASE/'config/previous-records-sha256.json',{str(p):sha(p) for p in previous_paths})
    save(BASE/'config/previous-records.json',[json.loads(p.read_text()) for p in previous_paths])
    external=ROOT.parent/'webgpu-bfs'
    save(BASE/'config/environment.json',dict(created=now(),version='v5.6',
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        implementation='existing four-bank layout expanded to adapter limits; 2D dispatch; bounded staging; no vertex-row tiling',
        binary_sha256=sha(BINARY),source_sha256=hashes,
        external_bfs_commit=subprocess.check_output(['git','-C',str(external),'rev-parse','HEAD'],text=True).strip(),
        external_bfs_used=False,memory_before=backend.memory(),disk_free_bytes=shutil.disk_usage(BASE).free))
    plan=dict(version='v5.6',created=now(),graphs=len(graphs),planned_runs=len(jobs),pivots=[1000,500,200,100,50],
        seeds=list(range(5)),repetitions=1,iterations=15,epsilon=.1,phase_chunks=1,
        bfs_engine='cpu',workgroup_size=128,gpu_timestamps=False,stress_samples=64,stress_seed=0,
        timeout_seconds=None,max_rss_bytes=18*GIB,min_available_bytes=512*1024**2,
        emergency_disk_bytes=GIB,pre_run_disk_reserve_bytes=4*GIB,
        pre_run_disk_formula='4 GiB + 160n + 48m bytes',
        warmup='seed999 per graph/p/session; excluded; failed warmup does not substitute formal seeds',
        execution_order='germany_osm p50 first, then original graph/p descending/seed order',
        exact_resource_precheck='12*n*p bytes for coexisting distance/one-sided arrays; skip if these alone exceed RSS policy',
        reference=str(OLD),report_interval_seconds=120)
    save(BASE/'config/plan.json',plan)
    (BASE/'PLAN.md').write_text('''# v5.6 大規模グラフ実験

指定Obsidian「4_大規模グラフ実験」の14入力、pivot1000/500/200/100/50、seed0〜4各1回、15反復、epsilon0.1、phase_chunks1、頂点別pivot順序を使用。前処理は既存CPU経路を--bfs-engine cpuで明示。W128、GPU timestamp無効。Stressは最終座標から固定64始点、評価seed0。SGD・前処理・総計算を記録し、Stress・保存・圧縮は除外。warmup999はgraph/p/session単位で除外。時間打切りなし。

実装差分は既存4バッファの各容量をadapter binding上限へ拡張、dispatchを2D化、転送時の一時stagingを64MiBに制限。先の頂点行タイル実装は不使用。4バッファという構成とGPUハードウェア・RAM上限は残る。小規模実GPUで4バンク境界と2D化による最終座標の完全一致、全クレートテストを確認。

ユーザー指定のgermany_osm p50を最初に測定後、残りは旧順序。350候補を保存し、確定GPU上限による未実行、前処理で同時生存する2密配列だけで18GiBを超える確定メモリ下限、実行中の容量停止、実測成功を区別。メモリ推定値だけでは除外しない。

RSS18GiB超、空きRAM512MiB未満10秒継続、ディスク空き1GiB未満で容量停止。開始前は空きRAM2GiB、ディスク4GiB+160n+48mを確保。資源不足なら未測定のまま保留し、他の収まるグラフへ進む。native座標は従来の検証済み可逆圧縮・Orkut辺共有保存を使用。過去v5データは変更しない。

2分ごとにprogress.json、PROGRESS.md、tables/paired-v5.jsonを更新。比較は同じgraph/p/seedで両方成功したもののみ。旧v5はW256の歴史データであり、同時測定ではない。ETAは新実測で補正した旧実行時間または同条件新実測。新規条件で根拠がない分は未推定件数として別表示。
''')


def integrity():
    e=json.loads((BASE/'config/environment.json').read_text())
    assert sha(BINARY)==e['binary_sha256']
    for rel,h in e['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    for rel,h in json.loads((BASE/'config/previous-records-sha256.json').read_text()).items():assert sha(rel)==h,rel


def progress(active=None,elapsed=0,**extra):
    global LAST_PRINT
    jobs=json.loads((BASE/'config/jobs.json').read_text());rs=records();done={r['run_id'] for r in rs}
    remaining=[j for j in jobs if j['run_id'] not in done]
    counts=collections.Counter(r['status'] for r in rs)
    good=[r for r in rs if r['status']=='success']
    paired=[];times=collections.defaultdict(list);relative=[]
    for r in good:
        measured=r['process_wall_seconds']+r.get('archive_seconds',0)+r.get('shared_repack_seconds',0)
        times[r['graph'],r['pivots']].append(measured)
        old=PREVIOUS.get(r['previous_run_id'])
        if old and old['status']=='success':
            assert all(r[k]==old[k] for k in ['input_sha256','seed','pivots','iterations','epsilon','stress_samples','stress_seed','stress_kind','rr_phase_chunks'])
            paired.append(dict(graph=r['graph'],pivots=r['pivots'],seed=r['seed'],
                preprocess_speedup=old['preprocess_time_ms']/r['preprocess_time_ms'],
                sgd_speedup=old['sgd_time_ms']/r['sgd_time_ms'],
                total_speedup=old['total_compute_time_ms']/r['total_compute_time_ms'],
                stress_ratio=r['stress_value']/old['stress_value'],
                initial_equal=r['initial_positions_sha256']==old['initial_positions_sha256'],
                preprocess_equal=r['preprocess_sha256']==old['preprocess_sha256']))
            old_wall=old.get('process_wall_seconds',0)+old.get('archive_seconds',0)+old.get('shared_repack_seconds',0)
            if old_wall:relative.append(measured/old_wall)
    known=0;unknown=0;conditions=set()
    for j in remaining:
        key=j['graph'],j['pivots'];old=PREVIOUS.get(j['previous_run_id'])
        if times[key]:estimate=statistics.median(times[key])
        elif old and old['status']=='success':estimate=(old['process_wall_seconds']+old.get('archive_seconds',0)+old.get('shared_repack_seconds',0))*(statistics.median(relative) if relative else 1)
        else:unknown+=1;continue
        known+=estimate;conditions.add(key)
    # Approximate one warmup per remaining condition not yet observed this session.
    warmup_estimate=sum(statistics.median(times[k]) for k in conditions if times[k])
    s=dict(updated=now(),version='v5.6',planned=len(jobs),finished=len(rs),success=len(good),
        statuses=dict(counts),remaining=len(remaining),active=active,active_elapsed_seconds=elapsed,
        eta_estimated_seconds=known+warmup_estimate,eta_unknown_runs=unknown,
        eta_includes_complete_warmup_model=False,
        v5_pairs=len(paired),memory=backend.memory(),free_disk_bytes=shutil.disk_usage(BASE).free,**extra)
    if paired:
        s['v5_comparison']={k:statistics.median(p[k] for p in paired) for k in ['preprocess_speedup','sgd_speedup','total_speedup','stress_ratio']}
    save(BASE/'progress.json',s);save(BASE/'tables/paired-v5.json',paired)
    msg=f"{s['updated']} 成功{len(good)}、確定未実行{counts['preflight_unavailable']}、確定RAM下限{counts['preflight_resource_unavailable']}、残り{len(remaining)}/{len(jobs)}。現在 {active['run_id'] if active else 'なし'} {elapsed:.0f}秒。見積可能分の残り約{(known+warmup_estimate)/3600:.2f}時間＋未推定{unknown}件。v5比較{len(paired)}件。"
    (BASE/'PROGRESS.md').write_text('# v5.6 大規模実験の進捗\n\n'+msg+'\n\n```json\n'+json.dumps(s,ensure_ascii=False,indent=2)+'\n```\n')
    if time.monotonic()-LAST_PRINT>=120 or active is None:
        print(msg,flush=True)
        if paired:print(json.dumps(s['v5_comparison']),flush=True)
        LAST_PRINT=time.monotonic()
    return s


def main():
    global PREVIOUS
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true');args=ap.parse_args()
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare();integrity()
        PREVIOUS={r['run_id']:r for r in json.loads((BASE/'config/previous-records.json').read_text())}
        backend.BASE=BASE;backend.BINARY=BINARY;backend.progress=progress;backend.EXTRA_ARGS=('--bfs-engine','cpu')
        validation.SCHEMA_PATH=BASE/'config/schema.json'
        if not args.run:progress(state='prepared');return
        save(BASE/'process.json',dict(pid=os.getpid(),started=now()))
        jobs=json.loads((BASE/'config/jobs.json').read_text());graphs={g['name']:g for g in json.loads((BASE/'config/datasets.json').read_text())}
        warmed=set();verified=set();deferred=[];session=str(time.time_ns())
        for j in jobs:
            if (BASE/'raw'/j['run_id']/'record.json').exists():continue
            g=graphs[j['graph']];reserve=4*GIB+160*g['nodes']+48*g['edges']
            if shutil.disk_usage(BASE).free<reserve or backend.memory()['available_bytes']<2*GIB:
                deferred.append(dict(j,reason='resource_reserve',memory=backend.memory(),free_disk=shutil.disk_usage(BASE).free));continue
            if g['name'] not in verified:
                assert sha(g['path'])==g['sha256'];verified.add(g['name'])
            key=g['name'],j['pivots']
            if key not in warmed:
                w=dict(j,seed=999,run_id=j['run_id']+'-warmup-'+session)
                progress(w,state='warmup');backend.execute(w,g,warmup=True);warmed.add(key)
            if shutil.disk_usage(BASE).free<reserve or backend.memory()['available_bytes']<2*GIB:
                deferred.append(dict(j,reason='post_warmup_resource_reserve'));continue
            progress(j,state='running');backend.execute(j,g);progress()
        integrity();save(BASE/'config/deferred.json',deferred)
        s=progress(state='resource_deferred' if deferred else 'completed',deferred=len(deferred))
        if not deferred:save(BASE/'completion.json',dict(completed=now(),statuses=s['statuses'],previous_records_unchanged=True))


if __name__=='__main__':main()
