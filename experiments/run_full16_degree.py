"""Degree-heterogeneous size study; frozen graphs and sequential matched measurements."""
from pathlib import Path
import collections, hashlib, importlib.util,json,platform,shutil,time
import networkx as nx
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'experiment_results/full16/degree'; BASE.mkdir(parents=True,exist_ok=True)
SPEC=importlib.util.spec_from_file_location('helper',ROOT/'experiment_results/full16/config/runner_helper.py');h=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(h);h.ROOT=ROOT
binary=ROOT/'target/full16-release/release/rr-gpu';cpu=Path('/Users/kose/master/atomic-scheduling-investigation/target/v3-release/release/baseline-sgd-non-gpu')
wrapper=BASE/'config/rr-unsplit';wrapper.parent.mkdir(exist_ok=True);wrapper.write_text('#!/bin/sh\nexec '+str(binary)+' --phase-chunks 1 "$@"\n');wrapper.chmod(0o755)
h.METHODS.update(sgd=str(cpu),rr_sgd=str(binary),rr_sgd_chunks1=str(wrapper))
graphs=[]
for n in (2000,4000,8000):
 for gs in (0,1):
  name=f'BA-{n}-g{gs}';g=nx.barabasi_albert_graph(n,5,seed=gs);p=BASE/'inputs'/f'{name}.mtx';p.parent.mkdir(exist_ok=True)
  if not p.exists():
   with p.open('w') as f:
    f.write('%%MatrixMarket matrix coordinate pattern symmetric\n% NetworkX BA m=5 graph_seed='+str(gs)+'\n'+f'{n} {n} {g.number_of_edges()}\n')
    for u,v in sorted(g.edges()):f.write(f'{u+1} {v+1}\n')
  deg=np.array([d for _,d in g.degree()]);graphs.append(dict(name=name,family='small',status='ready',eligible=True,path=str(p),sha256=h.sha(p),nodes=n,edges=g.number_of_edges(),graph_seed=gs,m=5,max_degree=int(deg.max()),mean_degree=float(deg.mean()),degree_cv=float(deg.std()/deg.mean())))
jobs=[]
for g in graphs:
 for seed in range(5):
  # Reverse method order on alternating initial-layout seeds.
  methods=['sgd','rr_sgd_chunks1','rr_sgd']
  if seed%2:methods=methods[::-1]
  for m in methods:jobs.append(dict(run_id=f"degree-{g['name']}-{m}-s{seed}",study='degree',graph=g['name'],dataset=g['name'],method=m,seed=seed,iterations=15,epsilon=.1,pivots=None,tile_size=1024 if m!='sgd' else None,workgroup_size=1024 if m!='sgd' else None))
limits=dict(timeout_seconds=120,max_rss_bytes=8*1024**3,min_free_disk_bytes=8*1024**3)
frozen=dict(graphs=graphs,jobs=jobs,binary_sha256={m:h.sha(Path(h.METHODS[m])) for m in ['sgd','rr_sgd','rr_sgd_chunks1']},runner_sha256=h.sha(__file__),networkx=nx.__version__,limits=limits)
p=BASE/'config/execution.json'
if p.exists():assert json.loads(p.read_text())==frozen
else:h.save(p,frozen);h.save(BASE/'config/datasets.json',graphs)
# Avoid simultaneous benchmark workloads.
while True:
 c=json.loads((ROOT/'experiment_results/full16/checkpoint.json').read_text())
 if c['completed']==160 and c['planned']==160:break
 time.sleep(2)
results=BASE/'results.jsonl';prior={r['run_id']:r for r in map(json.loads,results.read_text().splitlines())} if results.exists() else {};start=time.monotonic();initial=len(prior)
for j in jobs:
 if j['run_id'] in prior:continue
 g=next(g for g in graphs if g['name']==j['graph']);m=j['method'];assert h.sha(Path(h.METHODS[m]))==frozen['binary_sha256'][m];assert h.sha(binary)==frozen['binary_sha256']['rr_sgd']
 w=h.execute(BASE,dict(j,run_id=f"warmup-{g['name']}-{m}",seed=0,study='warmup'),g,limits,True)
 row=h.execute(BASE,j,g,limits) if w['status']=='success' else dict(j,status='warmup_failed',failure_reason=w['status'])
 prior[j['run_id']]=row
 with results.open('a') as f:f.write(json.dumps(row)+'\n')
 c=dict(completed=len(prior),planned=len(jobs),counts=dict(collections.Counter(r['status'] for r in prior.values())),last_run=j['run_id'],elapsed_seconds=time.monotonic()-start)
 h.save(BASE/'checkpoint.json',c);print(json.dumps(c),flush=True)
print('COMPLETE',flush=True)
