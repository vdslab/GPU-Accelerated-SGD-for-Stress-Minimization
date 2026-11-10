#!/usr/bin/env python3
"""Sequential EX3/EX4 feasibility pilot; never launches the full experiment."""
from pathlib import Path
import argparse, csv, datetime, hashlib, io, json, os, random, shutil
import statistics, subprocess, sys, tarfile, time
import psutil
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiment_results/v6/ex3-ex4-pr15-pilot-20261110'
PRE = '40b2ef21a'
POST = '70e945730'
GRAPHS = ['dwt_1005', '3elt', 'email']
def graph_path(graph):
    return ROOT/'data/v5/common/email.mtx' if graph=='email' else ROOT/'data'/f'{graph}.mtx'
CRATES = {
    'sgd': ('full/baseline-sgd-non-gpu', 'baseline-sgd-non-gpu'),
    'atomic_sgd': ('full/vram-lock-native', 'vram-lock-native'),
    'rr_sgd': ('full/rr_gpu', 'rr-gpu'),
    'sparse_sgd': ('sparse/baseline-sparse-sgd-non-gpu', 'baseline-sparse-sgd-non-gpu'),
    'atomic_sparse_sgd': ('sparse/vram-lock-sparse', 'vram-lock-sparse'),
    'rr_sparse_sgd': ('sparse/sparse-sgd-gpu', 'sparse-sgd-gpu'),
}

def now():
    return datetime.datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()

def sha(p):
    return hashlib.file_digest(Path(p).open('rb'), 'sha256').hexdigest()

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def prepare():
    BASE.mkdir(parents=True, exist_ok=True)
    src = BASE / 'source-pre'
    if not (src/'SNAPSHOT_READY').exists():
        src.mkdir(exist_ok=True)
        data = subprocess.check_output(['git', 'archive', PRE, 'implementations', 'shared'], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tf.extractall(src)
        # EX2 already used W128; isolate PR effects from a W256 -> W128 change.
        gpu = src / 'implementations/sparse/sparse-sgd-gpu/src/gpu.rs'
        shader = gpu.with_name('shader.wgsl')
        gpu.write_text(gpu.read_text().replace('const WORKGROUP_SIZE: u32 = 256;', 'const WORKGROUP_SIZE: u32 = 128;'))
        shader.write_text(shader.read_text().replace('@workgroup_size(256)', '@workgroup_size(128)'))
        (src/'SNAPSHOT_READY').write_text(PRE+'; sparse W128 variant\n')
    for version, source, target in [('pre', src, BASE/'build-pre'), ('post', ROOT, ROOT/'implementations/full/rr_gpu/target')]:
        for method, (crate, binary) in CRATES.items():
            frozen = BASE / 'bin' / version / method
            if frozen.exists():
                continue
            cmd = ['cargo', 'build', '--release', '--locked', '--manifest-path', str(source/'implementations'/crate/'Cargo.toml'), '--target-dir', str(target)]
            logfile = BASE/'logs'/f'build-{version}-{method}.log'
            logfile.parent.mkdir(exist_ok=True)
            print(now(), 'BUILD', version, method, flush=True)
            with logfile.open('w') as log:
                p = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if p.returncode:
                raise RuntimeError(logfile.read_text()[-4000:])
            frozen.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target/'release'/binary, frozen)
    renderer = BASE/'bin/gpu-visualizer'
    if not renderer.exists():
        shutil.copy2(ROOT/'tools/gpu_visualizer/target/release/gpu-visualizer', renderer)
    environment = dict(created=now(), pre_commit=subprocess.check_output(['git','rev-parse',PRE],cwd=ROOT,text=True).strip(), post_commit=subprocess.check_output(['git','rev-parse',POST],cwd=ROOT,text=True).strip(), external_bfs_commit=subprocess.check_output(['git','-C',str(ROOT.parent/'webgpu-bfs'),'rev-parse','HEAD'],text=True).strip(), external_bfs_origin='https://github.com/likr-sandbox/webgpu-bfs', pre_adjustment='RR Sparse: host WORKGROUP_SIZE and both WGSL @workgroup_size changed 256->128 to match EX2; no other source changes', machine=subprocess.check_output(['system_profiler','SPHardwareDataType','SPDisplaysDataType'],text=True), binaries={str(p.relative_to(BASE)):sha(p) for p in (BASE/'bin').rglob('*') if p.is_file()}, inputs={g:dict(path=str(graph_path(g)),sha256=sha(graph_path(g))) for g in GRAPHS}, policy='Sequential, no layout timeout, 15 iterations, epsilon0.1, pivots200, exact full stress, frozen binaries; graph/config warmups seed999 excluded; pilot only')
    save(BASE/'environment.json',environment)

def jobs():
    result = []
    for graph in GRAPHS:
        for version in (['pre','post'] if graph!='email' else []):
            for tile in [None] + [2**i for i in range(11)]:
                result.append(dict(stage='ex3', graph=graph, version=version, method='sgd' if tile is None else 'rr_sgd', tile=tile, bfs='auto', seed=10, repetition=0, timestamps=False))
        for repetition in range(3):
            group = [dict(stage='ex4', graph=graph, version=v, method=m, tile=128 if m=='rr_sgd' else None, bfs='auto', seed=0, repetition=repetition, timestamps=True) for v in ['pre','post'] for m in CRATES]
            random.Random(20261107 + repetition).shuffle(group)
            result.extend(group)
        for method in (['rr_sgd','rr_sparse_sgd'] if graph!='email' else []):
            for repetition in range(3):
                for bfs in ['cpu','gpu']:
                    result.append(dict(stage='bfs', graph=graph, version='post', method=method, tile=128 if method=='rr_sgd' else None, bfs=bfs, seed=0, repetition=repetition, timestamps=True))
    for j in result:
        j['id'] = '-'.join(str(j[k]) for k in ['stage','graph','version','method','tile','bfs','seed','repetition'])
    return result

def execute(j, warmup=False):
    folder = BASE/('warmups' if warmup else 'raw')/j['id']
    dest = folder/'record.json'
    if dest.exists():
        return json.loads(dest.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    binary = BASE/'bin'/j['version']/j['method']
    env = dict(os.environ, SGD_STRESS_MODE=j.get('stress_mode','exact'), SGD_STRESS_SAMPLES='256', SGD_GPU_TIMESTAMPS='1' if j['timestamps'] else '0')
    env.pop('SGD_TRACE_DIR',None)
    env.pop('SGD_TRACE_CHECKPOINTS',None)
    cmd = [str(binary), '--input', str(graph_path(j['graph'])), '--iterations','15','--epsilon','0.1','--seed',str(j['seed']), '--run-id',j['id'], '--run-mode','benchmark','--output-format','json','--output-dir',str(folder)]
    if 'sparse' in j['method']:
        cmd += ['--pivots','200']
    if j['method']=='rr_sgd':
        cmd += ['--tile-size',str(j['tile']),'--workgroup-size',str(j['tile']),'--coordinate-memory','shared','--phase-chunks','1']
    if j['version']=='post' and j['method'].startswith('rr_'):
        cmd += ['--bfs-engine',j['bfs']]
    cmd += j.get('extra_args',[])
    save(folder/'command.json',dict(argv=cmd, environment={k:env[k] for k in ['SGD_STRESS_MODE','SGD_STRESS_SAMPLES','SGD_GPU_TIMESTAMPS']}, binary_sha256=sha(binary), started=now()))
    row = dict(j, warmup=warmup, pilot_status='error', started=now(), memory_before=psutil.virtual_memory()._asdict(), swap_before=psutil.swap_memory()._asdict())
    start = time.monotonic()
    with (folder/'stdout.json').open('w') as out, (folder/'stderr.log').open('w') as err:
        p = subprocess.Popen(['/usr/bin/time','-l',*cmd],cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
        peak=0
        last=start
        interrupted=False
        while p.poll() is None:
            try:
                peak=max(peak,sum(c.memory_info().rss for c in psutil.Process(p.pid).children(recursive=True)))
            except psutil.Error:
                pass
            if time.monotonic()-last>45:
                print(now(),'RUNNING',j['id'],round(time.monotonic()-start),'seconds',flush=True)
                last=time.monotonic()
            if not interrupted and j.get('timeout_seconds') and time.monotonic()-start>j['timeout_seconds']:
                import signal
                os.killpg(p.pid,signal.SIGTERM)
                interrupted=True
                row['pilot_interruption']=f"Feasibility limit {j['timeout_seconds']}s; not a completed benchmark"
            time.sleep(.2)
    row.update(process_wall_seconds=time.monotonic()-start, returncode=p.returncode, peak_rss_sampled_bytes=peak, memory_after=psutil.virtual_memory()._asdict(), swap_after=psutil.swap_memory()._asdict())
    try:
        if p.returncode:
            raise RuntimeError((folder/'stderr.log').read_text()[-3000:])
        native = json.loads((folder/'stdout.json').read_text())
        assert native['status']=='success'
        assert native['stress_kind']==j.get('expected_stress_kind','exact')
        assert native['iterations']==15
        if native.get('attempted_updates') is not None and native.get('completed_updates') is not None:
            assert native['attempted_updates']==native['completed_updates']
        row.update(native)
        import re
        stderr=(folder/'stderr.log').read_text()
        match=re.search(r'(\d+)\s+maximum resident set size',stderr)
        if match: row['peak_rss_bytes']=int(match[1])
        for label, path in [('final',Path(native['final_positions_path'])), ('initial',Path(native['final_positions_path']).with_suffix('.txt'))]:
            if label=='initial':
                path=Path(str(path).replace('.final.txt','.initial.txt'))
            if path.exists():
                text=path.read_text().split('# Positions (x y)')[-1]
                coordinates=[tuple(map(float,line.split())) for line in text.splitlines() if line.strip() and not line.startswith('#')]
                row[label+'_coordinate_sha256']=hashlib.sha256(json.dumps(coordinates,separators=(',',':')).encode()).hexdigest()
        if j['stage'] in ['ex4','bfs']:
            render = [str(BASE/'bin/gpu-visualizer'),native['final_positions_path'],str(folder/'after.png'),'--size','2048x2048','--stress','off','--node-radius','1','--palette','blue']
            r = subprocess.run(render,cwd=ROOT,capture_output=True,text=True)
            log=r.stdout+'\n'+r.stderr
            (folder/'renderer.log').write_text(log)
            row['render_command']=render
            assert r.returncode==0, log[-1500:]
            from run_single import attach_render_timing
            attach_render_timing(row,log)
        row['pilot_status']='success'
    except Exception as e:
        row['pilot_error']=str(e)
    row['completed']=now()
    save(dest,row)
    print(now(),j['id'],row['pilot_status'],f"{row['process_wall_seconds']:.2f}s",flush=True)
    return row

def run(stages, graphs=None):
    plan=[j for j in jobs() if j['stage'] in stages and (graphs is None or j['graph'] in graphs)]
    save(BASE/'jobs.json',jobs())
    warmed=set()
    blocked={}
    for index,j in enumerate(plan):
        key=tuple(j[k] for k in ['stage','graph','version','method','tile','bfs','timestamps'])
        if (BASE/'raw'/j['id']/'record.json').exists():
            continue
        if shutil.disk_usage(BASE).free < 4*1024**3 or psutil.virtual_memory().available < 2*1024**3:
            raise RuntimeError('Less than 4GiB disk or 2GiB available memory; stopped between runs')
        save(BASE/'progress.json',dict(at=now(),active=j,stage_index=index,stage_planned=len(plan)))
        if key in blocked:
            save(BASE/'raw'/j['id']/'record.json',dict(j,pilot_status='blocked_by_warmup',warmup_error=blocked[key]))
            continue
        if key not in warmed:
            warm=dict(j, seed=999, id='warm-'+j['id'])
            w=execute(warm,True)
            warmed.add(key)
            if w['pilot_status']!='success':
                blocked[key]=w.get('pilot_error')
                # Keep a reviewable failed record for each requested repetition.
                failure=dict(j,pilot_status='blocked_by_warmup',warmup_error=w.get('pilot_error'),warmup_record=str(BASE/'warmups'/warm['id']/'record.json'))
                save(BASE/'raw'/j['id']/'record.json',failure)
                continue
        execute(j)
    save(BASE/'progress.json',dict(at=now(),active=None,stages_finished=stages))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--stages',nargs='+',choices=['ex3','ex4','bfs'])
    parser.add_argument('--graphs',nargs='+',choices=GRAPHS)
    a=parser.parse_args()
    if a.prepare: prepare()
    if a.stages: run(a.stages,a.graphs)
