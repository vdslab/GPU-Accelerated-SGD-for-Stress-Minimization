#!/usr/bin/env python3
"""Fail on an RR coordinate/stress regression; report CPU quality separately."""
import csv,json,pathlib,sys,statistics
root=pathlib.Path(sys.argv[1]); rows=[]
for before in sorted(root.glob('*-before.jsonl')):
 if not before.name.startswith(('full-', 'sparse-')): continue
 after=before.with_name(before.name.replace('-before','-after'))
 if not after.exists() or not after.stat().st_size: continue
 b=json.loads(before.read_text().splitlines()[0]); lines=[json.loads(l) for l in after.read_text().splitlines()];a=lines[0]
 sparse=before.name.startswith('sparse-')
 if sparse:
  for key in ['initial_sha256','preprocess_sha256','final_sha256','stress']: assert a[key]==b[key],(before,key)
  if len(lines)<2:continue
  c=lines[1]
  for key in ['initial_sha256','preprocess_sha256']: assert a[key]==c[key],(after,key)
  t='iteration_seconds';stress='stress';unit=1
 else:
  cpu=before.with_name(before.name.replace('-before','-cpu'))
  if not cpu.exists() or not cpu.stat().st_size:continue
  c=json.loads(cpu.read_text().splitlines()[0])
  for key in ['input_sha256','initial_positions_sha256','final_coordinate_text_sha256','stress_value']:assert a[key]==b[key],(before,key)
  for key in ['input_sha256','initial_positions_sha256','nodes','iterations','epsilon','stress_kind','stress_samples','stress_seed']:assert a[key]==c[key],(after,key)
  t='iteration_time_ms';stress='stress_value';unit=.001
 rows.append(dict(family='Sparse' if sparse else 'Full',graph=pathlib.Path(a['graph'] if sparse else a['input_path']).stem,nodes=a['nodes'],seed=a['seed'],pivots=a.get('pivots'),before_s=b[t]*unit,after_s=a[t]*unit,cpu_s=c[t]*unit,speedup_before=b[t]/a[t],speedup_cpu=c[t]/a[t],rr_stress=a[stress],cpu_stress=c[stress],stress_ratio=a[stress]/c[stress],coordinates_equal=True))
assert rows,'no complete pairs'
with (root/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for family,graph in sorted(set((r['family'],r['graph']) for r in rows)):
 rr=[r for r in rows if (r['family'],r['graph'])==(family,graph)]
 s={k:rr[0][k] for k in ['family','graph','nodes','pivots']};s['seeds']=[r['seed'] for r in rr]
 for key in ['before_s','after_s','cpu_s','speedup_before','speedup_cpu','stress_ratio']:s[key]=statistics.median(r[key] for r in rr)
 s['max_stress_ratio']=max(r['stress_ratio'] for r in rr);summary.append(s)
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'pairs':len(rows),'all_coordinates_equal':True,'summary':summary},indent=2))
