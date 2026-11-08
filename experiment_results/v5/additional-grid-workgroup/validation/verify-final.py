import json,hashlib,math,statistics,csv,time,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'experiments'))
from run_v5 import ROOT,save,now
from prepare_v5_grid_extension import BASE
from experiment_plan import METHODS
started=time.monotonic()
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
 return h.hexdigest()
plan=read(BASE/'config/plan.json');jobs=read(BASE/'config/jobs.json');graphs={g['name']:g for g in read(BASE/'config/datasets.json')};rows=[read(p) for p in (BASE/'raw').glob('*/record.json')]
assert len(rows)==810 and {r['run_id'] for r in rows}=={r['run_id'] for r in jobs}
index={r['run_id']:r for r in rows};binaries={}
for name,expected in read(ROOT/'experiment_results/v5/config/environment.json')['binary_sha256'].items():
 actual=sha(ROOT/METHODS[name].binary);assert actual==expected;binaries[name]=actual
variants={v['workgroup_size']:v for v in read(BASE/'config/binaries.json')}
for w,v in variants.items():assert sha(Path(v['binary']))==v['sha256'];binaries['added_w'+str(w)]=v['sha256']
for g in graphs.values():assert sha(Path(g['path']))==g['sha256']
archives=0;compressed_bytes=0
for j in jobs:
 r=index[j['run_id']];folder=BASE/'raw'/j['run_id'];assert r['status']=='success' and r['archive_status']=='success'
 for k,v in j.items():assert r[k]==v,(j['run_id'],k)
 assert r['input_sha256']==graphs[r['graph']]['sha256']
 assert r['binary_sha256']==(plan['full_binary_sha256'] if r['stage']=='full' else variants[r['workgroup_size']]['sha256'])
 assert r['returncode']==0
 assert math.isclose(r['sgd_mean_iteration_ms'],r['sgd_time_ms']/15,abs_tol=1e-9)
 assert math.isclose(r['gpu_execution_ms'],r['gpu_prepare_time_ms']+r['sgd_time_ms']+r['readback_time_ms'],abs_tol=1e-9)
 assert math.isclose(r['gpu_execution_with_upload_ms'],r['gpu_execution_ms']+r['upload_time_ms'],abs_tol=1e-9)
 before=read(folder/'measurement.json');allowed={'initial_positions_path','final_positions_path','vertex_map_path'}
 for k,v in before.items():
  if k not in allowed:assert r[k]==v,(j['run_id'],'measurement field changed',k)
 manifest=read(folder/'artifact-archive.json');assert not manifest['errors'];assert len(manifest['files']) in (2,3)
 if len(manifest['files'])==3:assert any(e['original_path'].endswith('.vertex-map.txt') for e in manifest['files'])
 assert all(any(e['original_path'].endswith('.'+stage+'.txt') for e in manifest['files']) for stage in ['initial','final'])
 for e in manifest['files']:
  p=Path(e['compressed_path']);assert p.is_file() and p.stat().st_size==e['compressed_bytes']
  assert sha(p)==e['compressed_sha256'],p
  assert e['original_sha256']==e['decompressed_sha256']
  assert e.get('xz_crc_validated') or e.get('gzip_crc_validated')
  archives+=1;compressed_bytes+=p.stat().st_size
 for k in ['initial_positions_path','final_positions_path']:
  if r.get(k):assert Path(r[k]).is_file()
summary=read(BASE/'tables/summary.json');assert len(summary)==54
for s in summary:
 group=[r for r in rows if all(r[k]==s[k] for k in ['stage','graph','tile_size','workgroup_size','pivots'])]
 assert len(group)==15 and len({(r['seed'],r['repetition']) for r in group})==15
 for field in ['sgd_mean_iteration_ms','gpu_execution_ms','gpu_execution_with_upload_ms','total_compute_time_ms','stress_value','peak_rss_bytes']:
  values=[statistics.median(r[field] for r in group if r['seed']==seed) for seed in range(5)]
  assert s[field+'_median']==statistics.median(values) and s[field+'_min']==min(values) and s[field+'_max']==max(values)
with (BASE/'tables/runs.csv').open() as f:assert len(list(csv.DictReader(f)))==810
pairs=read(BASE/'tables/sparse-paired-summary.json');assert sum(r['paired_runs'] for r in pairs)==180 and all(r['same_stress_pairs']==15 for r in pairs)
audit=dict(at=now(),status='passed',success=810,failed=0,pending=0,settings=54,full=450,sparse=360,measurement_binaries=binaries,input_hashes={g['name']:g['sha256'] for g in graphs.values()},checked_archives=archives,checked_compressed_logical_bytes=compressed_bytes,all_compressed_sha_match=True,all_native_measurement_scalars_unchanged=True,derived_timing_formulas_checked=True,hierarchical_aggregation_independently_checked=True,sparse_fingerprint_and_stress_pairs=180,seconds=time.monotonic()-started)
save(BASE/'validation/final-data-audit.json',audit)
print(json.dumps({k:audit[k] for k in ['status','success','settings','checked_archives','seconds']}))
