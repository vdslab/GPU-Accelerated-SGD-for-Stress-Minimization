from pathlib import Path
import csv,json,shutil,hashlib,subprocess,platform
root=Path('/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization')
old=Path('/Users/kose/master/atomic-scheduling-investigation')
vault=Path('/Users/kose/Library/Mobile Documents/iCloud~md~obsidian/Documents/sakamoti55/11_大学院研究')
base=root/'experiment_results/v4'
def save(name,data):
 p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
names={r['graph'] for r in csv.DictReader((vault/'02_論文フェーズ/図表/fig05_iteration_times.csv').open()) if r['family']=='sparse'}|{'delaunay_n19'}
assert len(names)==38
all_data=json.loads((old/'experiment_results/v3/config/datasets.json').read_text());recovered={g['name']:g for g in json.loads((old/'experiment_results/v3/snap-origin-recovery/config/datasets.json').read_text())};all_data=[recovered.get(g['name'],g) for g in all_data];data=[]
for g in all_data:
 if g['name'] not in names:continue
 g=dict(g);src=old/g['path'];dst=root/'data/v4_lcc'/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():shutil.copy2(src,dst)
 assert hashlib.sha256(dst.read_bytes()).hexdigest()==g['sha256']
 g.update(v3_prepared_path=str(src),path=str(dst))
 data.append(g)
assert len(data)==38
save('config/datasets.json',data)
limits=json.loads((old/'experiment_results/v3/config/limits.json').read_text());limits['rationale']='Keep v3 resource monitoring unchanged; rerun only Sparse graphs shown in paper figures. Basic runs have no trace instrumentation; convergence traces are separate.'
save('config/limits.json',limits)
jobs=[]
for study in ['basic','convergence']:
 for g in sorted(data,key=lambda g:g['nodes']):
  if study=='convergence' and g['name'] not in ['luxembourg_osm','kron_g500-logn18','coAuthorsCiteseer']:continue
  for seed in range(5 if study=='basic' else 10):
   for m in ['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd']:
    jobs.append(dict(run_id=f"v4-{study}-{g['name']}-{m}-s{seed}-p200",study=study,graph=g['name'],dataset=g['name'],method=m,seed=seed,iterations=15,epsilon=.1,pivots=200))
save('config/plan.json',jobs)
save('config/machine.json',dict(platform=platform.platform(),cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),os=subprocess.check_output(['sw_vers'],text=True),date_timezone='Asia/Tokyo'))
full=base/'reference/full-v3';full.mkdir(parents=True,exist_ok=True)
for name in ['summary.csv','runs.csv','convergence.csv']:
 source=old/'experiment_results/v3/tables'/name
 rows=list(csv.DictReader(source.open()));keep=[r for r in rows if r.get('family')=='full' or (r.get('method') in ['sgd','atomic_sgd','rr_sgd'])]
 with (full/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(keep)
for name in ['fig05_iteration_times.csv','fig04_tile_size.png','fig04_tile_size.pdf','fig04_tile_size.svg']:
 shutil.copy2(vault/'02_論文フェーズ/図表'/name,full/name)
save('reference/full-v3/provenance.json',dict(source=str(old/'experiment_results/v3'),policy='Inherited Full only; no new Full experiments',source_tables={n:hashlib.sha256((old/'experiment_results/v3/tables'/n).read_bytes()).hexdigest() for n in ['runs.csv','summary.csv','convergence.csv']}))
(base/'.gitignore').write_text('**/*.txt.gz\n**/*.f64le\n**/*.f64le.gz\n**/*.npy\n**/*.npz\nconfig/source-snapshot/\n')
print('Prepared',len(data),'graphs;',len(jobs),'formal runs; free GiB',shutil.disk_usage(base).free/1024**3)
