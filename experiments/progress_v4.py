#!/usr/bin/env python3
"""Read-only progress estimate using v3 graph durations, calibrated to finished v4 jobs."""
from pathlib import Path
import collections,datetime,json,statistics
import psutil
BASE=Path(__file__).resolve().parents[1]/'experiment_results/v4'
def main():
 reference=json.loads((BASE/'reference/v3-duration-estimates.json').read_text())
 planpath=BASE/'config/rr-only-plan.json';rr_only=planpath.exists()
 plan=json.loads((planpath if rr_only else BASE/'config/plan.json').read_text());done=[json.loads(s) for s in (BASE/'raw/results.jsonl').read_text().splitlines() if s.strip()];done=[r for r in done if not rr_only or r['method']=='rr_sparse_sgd'];ids={r['run_id'] for r in done}
 def key(r):return '|'.join(str(r[k]) for k in ['study','graph','method'])
 ratios=collections.defaultdict(list)
 for r in done:
  ref=reference.get(key(r))
  if ref and r['status']=='success':ratios[r['method']].append(r['process_wall_seconds']/ref['median_seconds'])
 factors={m:statistics.median(v) for m,v in ratios.items()};remaining=0.;warmups=set();cached={}
 for j in plan:
  if j['run_id'] in ids:continue
  warmup_id=f"warmup-{j['graph']}-{j['method']}-p200";p=BASE/'warmups'/warmup_id/'record.json'
  if warmup_id not in cached:cached[warmup_id]=json.loads(p.read_text()) if p.exists() else None
  if cached[warmup_id] and cached[warmup_id]['status']!='success':continue
  ref=reference.get(key(j));seconds=(ref['median_seconds'] if ref else 120)*factors.get(j['method'],1)
  remaining+=min(120,seconds)
  if cached[warmup_id] is None and warmup_id not in warmups:
   remaining+=min(120,seconds);warmups.add(warmup_id)
 archive=statistics.median([r.get('archive_seconds',0) for r in done]) if done else 0
 remaining+=archive*(len(plan)-len(done)+len(warmups))
 # Broad estimate: graph-dependent failures, diagnostic evaluation and publication remain uncertain.
 low=max(0,remaining*.8/3600)+.5;high=remaining*1.2/3600+1.5
 processes=json.loads((BASE/'processes.json').read_text()) if (BASE/'processes.json').exists() else {}
 health={k:psutil.pid_exists(v) for k,v in processes.items() if k.endswith('_pid')}
 now=datetime.datetime.now().astimezone()
 print(json.dumps(dict(process_health=health,now=now.isoformat(),completed=len(done),planned=len(plan),statuses=dict(collections.Counter(r['status'] for r in done)),native_plus_archive_hours=round(remaining/3600,2),remaining_total_hours=[round(low,1),round(high,1)],estimated_completion_local=[(now+datetime.timedelta(hours=low)).strftime('%H:%M'),(now+datetime.timedelta(hours=high)).strftime('%H:%M')],method_duration_factors=factors),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
