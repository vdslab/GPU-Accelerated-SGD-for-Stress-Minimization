"""Freeze inputs and isolated Sparse workgroup variants before added measurements."""
import hashlib
import itertools
import json
from pathlib import Path
import random
import shutil

from experiment_plan import METHODS
from run_v5 import ROOT, VAULT, now, save

BASE=ROOT/'experiment_results/v5/additional-grid-workgroup'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def make_grid(shape):
    name='grid'+str(len(shape))+'d_'+'x'.join(map(str,shape))
    n=1
    for length in shape:n*=length
    strides=[];stride=1
    for length in reversed(shape):strides.append(stride);stride*=length
    strides.reverse()
    edges=[]
    for point in itertools.product(*(range(s) for s in shape)):
        u=sum(a*b for a,b in zip(point,strides))
        for axis,length in enumerate(shape):
            if point[axis]+1<length:edges.append((u,u+strides[axis]))
    path=BASE/'data'/(name+'.mtx');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        f.write('%%MatrixMarket matrix coordinate pattern general\n% Deterministic unweighted connected grid; one entry per undirected edge, row-major IDs.\n')
        f.write(f'{n} {n} {len(edges)}\n')
        for u,v in edges:f.write(f'{u+1} {v+1}\n')
    return dict(name=name,path=str(path),nodes=n,edges=len(edges),sha256=sha(path),family='grid',shape=shape,generator='Nearest neighbors along each axis, open boundaries, row-major IDs, no graph RNG')


def main():
    BASE.mkdir(parents=True,exist_ok=True)
    assert not (BASE/'config/plan.json').exists(),'Do not replace a frozen plan'
    grids=[make_grid(shape) for shape in [(64,64),(96,96),(128,128),(128,192),(16,16,16),(24,24,24)]]
    original=json.loads((ROOT/'experiment_results/v5/config/datasets.json').read_text())
    sparse=[g for g in grids if g['name'] in ['grid2d_64x64','grid2d_128x128','grid3d_24x24x24']]
    sparse.extend(next(g for g in original if g['name']==name) for name in ['luxembourg_osm','la2010','coPapersCiteseer'])
    all_graphs={g['name']:g for g in grids+sparse}
    save(BASE/'config/datasets.json',list(all_graphs.values()))
    frozen=json.loads((ROOT/'experiment_results/v5/config/environment.json').read_text())['binary_sha256']
    for method in ['rr_sgd','rr_sparse_sgd']:assert sha(ROOT/METHODS[method].binary)==frozen[method]
    source=ROOT/'implementations/sparse/sparse-sgd-gpu';variants=[]
    for w in [256,1024]:
        dest=BASE/'source-snapshot'/f'sparse-w{w}';shutil.copytree(source/'src',dest/'src')
        package=f'sparse-workgroup-w{w}'
        cargo=(source/'Cargo.toml').read_text().replace('path = "../../../shared/experiment-common"','path = '+json.dumps(str(ROOT/'shared/experiment-common'))).replace('name = "sparse-sgd-gpu"',f'name = "{package}"',1)
        cargo+='\n[lib]\nname = "sparse_sgd_gpu"\npath = "src/lib.rs"\n'
        (dest/'Cargo.toml').write_text(cargo)
        (dest/'Cargo.lock').write_text((source/'Cargo.lock').read_text().replace('name = "sparse-sgd-gpu"',f'name = "{package}"'))
        lib=dest/'src/lib.rs';s=lib.read_text().replace('../../baseline-sparse-sgd-non-gpu/src/graph.rs','graph_reference.rs').replace('../../baseline-sparse-sgd-non-gpu/src/algorithm.rs','cpu_reference.rs');lib.write_text(s)
        shutil.copy2(ROOT/'implementations/sparse/baseline-sparse-sgd-non-gpu/src/graph.rs',dest/'src/graph_reference.rs')
        shutil.copy2(ROOT/'implementations/sparse/baseline-sparse-sgd-non-gpu/src/algorithm.rs',dest/'src/cpu_reference.rs')
        gpu=dest/'src/gpu.rs';s=gpu.read_text().replace('../../../../shared/gpu-timing/wgpu.rs','timing.rs');assert s.count('const WORKGROUP_SIZE: u32 = 256;')==1;s=s.replace('const WORKGROUP_SIZE: u32 = 256;',f'const WORKGROUP_SIZE: u32 = {w};');gpu.write_text(s)
        shutil.copy2(ROOT/'shared/gpu-timing/wgpu.rs',dest/'src/timing.rs')
        shader=dest/'src/shader.wgsl';s=shader.read_text();assert s.count('@workgroup_size(256)')==2;shader.write_text(s.replace('@workgroup_size(256)',f'@workgroup_size({w})'))
        variants.append(dict(workgroup_size=w,manifest=str(dest/'Cargo.toml'),binary=str(BASE/'bin'/f'rr-sparse-w{w}'),package_name=package,
                             code_changes='Workgroup size constant and two shader declarations only; frozen reference/timing sources copied locally; p, arithmetic and update schedule unchanged'))
    save(BASE/'config/variants-to-build.json',variants)
    rng=random.Random(20261108);jobs=[]
    for graph in grids:
        for seed in range(5):
            for rep in range(3):
                tiles=[64,128,256,512,1024];rng.shuffle(tiles)
                for tile in tiles:jobs.append(dict(stage='full',graph=graph['name'],method='rr_sgd',tile_size=tile,workgroup_size=1024,pivots=None,seed=seed,repetition=rep,run_id=f'grid-full-{graph["name"]}-t{tile}-s{seed}-r{rep}'))
    for graph in sorted(sparse,key=lambda g:g['nodes']):
        for pivots in [1000,200]:
            for seed in range(5):
                for rep in range(3):
                    widths=[256,1024];rng.shuffle(widths)
                    for width in widths:jobs.append(dict(stage='sparse',graph=graph['name'],method='rr_sparse_sgd',tile_size=None,workgroup_size=width,pivots=pivots,seed=seed,repetition=rep,run_id=f'grid-sparse-{graph["name"]}-w{width}-p{pivots}-s{seed}-r{rep}'))
    assert len(jobs)==810
    save(BASE/'config/jobs.json',jobs)
    plan=dict(created=now(),full_graphs=[g['name'] for g in grids],sparse_graphs=[g['name'] for g in sparse],full_tiles=[64,128,256,512,1024],full_workgroup=1024,sparse_workgroups=[256,1024],sparse_pivots=[1000,200],seeds=list(range(5)),repetitions=3,iterations=15,epsilon=.1,phase_chunks=1,stress_samples=256,stress_seed=0,gpu_timestamps=False,timeout_seconds=None,order_seed=20261108,planned_full_runs=450,planned_sparse_runs=360,planned_runs=810,
              primary_metric='SGD elapsed / 15 ms per iteration; also GPU prepare + SGD + readback, and plus upload',
              selection='Grid families and sizes fixed before added measurements. Full shape/n and Sparse workgroup/pivot comparisons are distinct. Keep every outcome, not only W/T1024 wins.',
              warmup='seed999 once per graph/configuration/session; excluded',resource_policy='Sequential only; stop on RSS18GiB, available RAM512MiB sustained10s, disk1GiB. Pause before next run if RAM<2GiB or disk<4GiB. No wall-clock kill.',full_binary_sha256=frozen['rr_sgd'],production_sparse_binary_sha256=frozen['rr_sparse_sgd'],sparse_variants=variants)
    save(BASE/'config/plan.json',plan)
    for path in [BASE/'raw',BASE/'warmups',BASE/'validation',BASE/'logs',BASE/'tables',BASE/'figures',BASE/'reports',BASE/'bin']:path.mkdir(exist_ok=True)
    exclude=ROOT/'.git/info/exclude';s=exclude.read_text()
    for suffix in ['raw','warmups','validation']:
        entry='/experiment_results/v5/additional-grid-workgroup/'+suffix+'/'
        if entry not in s:s+='\n'+entry+'\n'
    exclude.write_text(s)
    text='# 追加検証：FullのgridとSparseのworkgroup\n\n更新: '+now()+'\n\n既存v5とは別の追加実験。結果を見てグラフを追加・除外せず全結果を保存する。\n\n| 項目 | 条件 |\n|---|---|\n| Full | 2D grid 64×64, 96×96, 128×128, 128×192、3D grid 16³, 24³ |\n| Full T / W | T=64,128,256,512,1024、W1024固定 |\n| Sparse | 2D grid 64×64,128×128、3D grid24³、luxembourg_osm、la2010、coPapersCiteseer |\n| Sparse W / p | W256と1024、p1000と200 |\n| 実行 | 各5seed×3回、15反復、epsilon0.1、分割なし |\n| 評価 | 同じ最終配置のFull Stress、n≤8000厳密、それ以上256始点seed0 |\n| 計時 | 主指標は平均1反復のSGD経過時間。準備・転送・読戻し込みも別に集計 |\n| 予定 | Full450＋Sparse360＝810実行、warmupは除外 |\n\nSparseのWはFullのTとは別の設定。既存のCPU/Atomicや本測定バイナリは変更しない。実験用Sparseバイナリを別にビルドし、W256との数値一致を検証してから測定する。1024が最速という結論は先に置かない。Fullは全頂点対を扱うためgrid構造そのものの効果と頂点数の効果を混同しない。大きいFullはRAM制約に触れる可能性があり、容量停止・未実行と速度結果を区別する。\n'
    (BASE/'README.md').write_text(text)
    (VAULT/'8_追加実験_gridとSparse設定.md').write_text('---\ntags: [area/大学院研究, 実験計画]\nupdated: '+now()[:10]+'\nstatus: 準備中\n---\n\n[[0_概要]]\n\n'+text)
    print(json.dumps(dict(full=450,sparse=360,total=810,base=str(BASE)),ensure_ascii=False))


if __name__=='__main__':main()
