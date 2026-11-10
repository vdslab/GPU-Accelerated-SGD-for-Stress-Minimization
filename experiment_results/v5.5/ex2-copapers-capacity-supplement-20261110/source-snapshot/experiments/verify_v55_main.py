import argparse,sys,json,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description='Verify capacity expansion against the latest source checkout')
ap.add_argument('--source-root',type=Path,required=True)
ap.add_argument('--binary',type=Path)
args=ap.parse_args()
MAIN=args.source_root.resolve()
sys.path.insert(0,str(ROOT/'experiments'))
import run_v55_copapers as supplement
import run_ex2_rr128 as backend
import run_experiments as validation
from run_experiments import file_sha256 as sha
from run_v5 import save,now
BASE=supplement.BASE;ORIGINAL=supplement.ORIGINAL
binary=BASE/'bin/rr-sparse-main';shutil.copy2(args.binary or MAIN/'implementations/sparse/sparse-sgd-gpu/target/release/sparse-sgd-gpu',binary)
backend.BASE=BASE;backend.ROOT=MAIN;backend.RENDER=ORIGINAL/'bin/gpu-visualizer';backend.progress=supplement.progress
validation.SCHEMA_PATH=ORIGINAL/'config/schema-v55.json'
graphs={g['name']:g for g in json.loads((ORIGINAL/'config/datasets.json').read_text())}
results=[]
for graph in ['dwt_1005','coPapersCiteseer']:
 j=dict(graph=graph,method='rr_sparse_sgd',tile_size=None,workgroup_size=128,seed=0,repetition=0,run_id='main-capacity-verification-'+graph,implementation_variant='latest-main-four-256MiB-banks')
 r=backend.execute(j,graphs[graph],kind='validation',binary=binary)
 comparison=next(json.loads(p.read_text()) for p in (BASE/('raw' if graph=='coPapersCiteseer' else 'validation')).glob('*/record.json') if (lambda q:q['graph']==graph and q['seed']==0 and q['repetition']==0 and q['run_id']!=j['run_id'])(json.loads(p.read_text())))
 keys=['final_coordinate_text_sha256','stress_value','preprocess_sha256','initial_positions_sha256']
 assert all(r[k]==comparison[k] for k in keys),(graph,keys)
 results.append(dict(graph=graph,status='success',coordinates_stress_and_fingerprints_equal=True,run_id=j['run_id']))
paths=['implementations/sparse/sparse-sgd-gpu/src/gpu.rs','implementations/sparse/sparse-sgd-gpu/src/shader.wgsl','implementations/sparse/sparse-sgd-gpu/Cargo.toml','implementations/sparse/sparse-sgd-gpu/Cargo.lock']
for rel in paths:
 dest=BASE/'source-snapshot/latest-main'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(MAIN/rel,dest)
(BASE/'source-snapshot/latest-main.patch').write_bytes(subprocess.check_output(['git','diff','--','implementations/sparse/sparse-sgd-gpu'],cwd=MAIN))
save(BASE/'validation/latest-main.json',dict(completed=now(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=MAIN,text=True).strip(),binary_sha256=sha(binary),source_sha256={p:sha(MAIN/p) for p in paths},results=results,measurements_in_main_aggregation=False))
print('LATEST MAIN VERIFICATION PASSED',flush=True)
