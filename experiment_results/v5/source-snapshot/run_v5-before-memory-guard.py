#!/usr/bin/env python3
"""Sequential, resumable v5 measurements. No wall-clock deadline, no tracing."""
from pathlib import Path
import argparse, collections, csv, datetime, fcntl, hashlib, json, os, random, re, shutil, statistics, subprocess, sys, time
import psutil
from experiment_plan import METHODS, Dataset, RunDefinition
from run_experiments import parse_and_validate_success, file_sha256, git_environment
from run_v4 import archive_text_artifacts
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiment_results/v5'
VAULT=Path('/Users/kose/Documents/ChatGPT/Obsidian タスク管理/sakamoti55/12_大学院研究/実験')
LARGE={'belgium_osm','netherlands_osm','italy_osm','germany_osm','road_usa','europe_osm','com-LiveJournal','com-Orkut'}
TILES=['email','add20','CA-GrQc','p2p-Gnutella08','PGPgiantcompo','dwt_1005','dwt_2680','3elt','bcspwr10','fe_4elt2']

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')
def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def memory():
 v=psutil.virtual_memory();s=psutil.swap_memory()
 return dict(available_bytes=v.available,used_bytes=v.used,swap_used_bytes=s.used,swap_in_bytes=s.sin,swap_out_bytes=s.sout)
def config():return json.loads((BASE/'config/plan.json').read_text())
def prepare():
 if (BASE/'config/plan.json').exists():return
 graphs=json.loads((ROOT/'data/v5/manifest.json').read_text())['graphs']
 datasets=[]
 for g in graphs:
  if g['name'] in LARGE:continue
  p=ROOT/'data/v5'/g['path']
  assert g['status']=='ready' and file_sha256(p)==g['sha256'],g['name']
  datasets.append(dict(g,path=str(p)))
 assert len(datasets)==62
 save(BASE/'config/datasets.json',datasets)
 save(BASE/'config/plan.json',dict(created=now(),seeds_tile=list(range(10,15)),seeds_comparison=list(range(5)),repetitions=3,iterations=15,epsilon=.1,pivots=200,stress_samples=256,stress_seed=0,order_seed=20261107,timeout_seconds=None,normal_rr_full_tile=None,total_formal_runs=6300,large_excluded=sorted(LARGE),tile_graphs=TILES,tile_classification='5 high-degree-heterogeneity candidates and 5 non-scale-free candidates; scale-free property not statistically established; add20 is a circuit graph',memory_policy='sequential; log RSS and swap; pause between runs if available memory < 2 GiB or disk free < 4 GiB; no wall-clock kill'))
 env=git_environment();env['binary_sha256']={m:file_sha256(ROOT/s.binary) for m,s in METHODS.items()};env['memory']=memory();env['started_at']=now();save(BASE/'config/environment.json',env)
 (BASE/'source-snapshot/git-diff.patch').write_text(subprocess.check_output(['git','diff','--binary'],cwd=ROOT,text=True))
 for p in [Path(__file__),ROOT/'experiments/run_experiments.py',ROOT/'experiments/experiment_plan.py',ROOT/'experiments/record_metadata.py',ROOT/'experiments/run_v4.py',ROOT/'experiments/schema/experiment-record-v1.json']:
  shutil.copy2(p,BASE/'source-snapshot'/p.name)
 jobs=[];rng=random.Random(20261107)
 for stage,seeds in [('tile',range(10,15)),('comparison',range(5))]:
  gs=[g for name in TILES for g in datasets if g['name']==name] if stage=='tile' else sorted(datasets,key=lambda g:(g['nodes'],g['name']))
  for g in gs:
   for seed in seeds:
    for rep in range(3):
     choices=[('sgd',None)]+[('rr_sgd',2**i) for i in range(11)] if stage=='tile' else [(m,None) for m in METHODS if g['set']=='common' or METHODS[m].sparse]
     rng.shuffle(choices)
     for method,tile in choices:
      jobs.append(dict(stage=stage,graph=g['name'],method=method,tile_size=tile,seed=seed,repetition=rep,run_id=f'v5-{stage}-{g["name"]}-{method}-t{tile or "fixed"}-s{seed}-r{rep}'))
 assert collections.Counter(j['stage'] for j in jobs)=={'tile':1800,'comparison':4500}
 save(BASE/'config/jobs.json',jobs)
 with (BASE/'config/execution-order.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(jobs[0]));w.writeheader();w.writerows(jobs)
 (BASE/'README.md').write_text('''# v5 実験結果\n\n大規模8件を除く62グラフ。過去の結果は再利用せず新規測定する。\n\n| 保存先 | 内容 |\n|---|---|\n| config/ | 入力ハッシュ、全6,300実行の順序、方式・seed・反復・評価設定、機材・バイナリハッシュ |\n| raw/tile/ | タイル1,800実行のJSON、ログ、圧縮座標 |\n| raw/comparison/ | 速度と最終Stress共用4,500実行 |\n| warmups/ | graph×method×設定のseed999、集計対象外 |\n| tables/ | 生データCSV・seed内3回→5seedの階層集計 |\n| figures/ | 完了後の図 |\n| reports/ | 結果と考察・Obsidianと同期 |\n| logs/ | ビルド・実行ログ |\n| source-snapshot/ | 実行管理コードとdirty差分 |\n| progress.json / PROGRESS.md | 現在の条件、残り実行数と残り設定数、推定時間 |\n\n`python3 experiments/run_v5.py --stage tile`で開始・再開。完了記録は上書きしない。\n通常比較のRR Fullの共通Tはタイル実験後に決め、config/plan.jsonのnormal_rr_full_tileに固定してから`--stage comparison`。\n\n15反復、epsilon0.1、RR分割なし、通常Sparse p=min(200,n)、評価seed0・256始点（n≤8000は厳密）。時間打切りなし。測定中は描画・ビルドをしない。圧縮はプロセス終了後。\nタイルの構造群は候補名を維持するが、スケールフリー性の実証とは扱わない。\n''')

def execute(j,g,warmup=False):
 folder=BASE/('warmups' if warmup else 'raw/'+j['stage'])/j['run_id'];folder.mkdir(parents=True,exist_ok=True)
 dest=folder/'record.json'
 if dest.exists():return json.loads(dest.read_text())
 spec=METHODS[j['method']];pivots=min(200,g['nodes']) if spec.sparse else None
 run=RunDefinition('v5-'+j['stage'],j['run_id'],'benchmark',j['method'],ROOT/spec.binary,Dataset(g['name'],Path(g['path']),g['sha256']),j['seed'],15,.1,pivots,j['repetition'])
 cmd=[str(run.binary),'--input',g['path'],'--run-id',j['run_id'],'--run-mode','benchmark','--iterations','15','--epsilon','0.1','--seed',str(j['seed']),'--output-format','json','--output-dir',str(folder)]
 if pivots:cmd+=['--pivots',str(pivots)]
 tile=j['tile_size'] or config()['normal_rr_full_tile']
 if j['method']=='rr_sgd':
  assert tile is not None
  cmd+=['--tile-size',str(tile),'--workgroup-size','1024','--coordinate-memory','shared','--phase-chunks','1']
 env=dict(os.environ,SGD_STRESS_SAMPLES='256',SGD_GPU_TIMESTAMPS='0');env.pop('SGD_TRACE_DIR',None);env.pop('SGD_TRACE_CHECKPOINTS',None)
 assert file_sha256(run.binary)==json.loads((BASE/'config/environment.json').read_text())['binary_sha256'][j['method']], 'binary changed during measurement'
 save(folder/'command.json',dict(argv=cmd,environment={'SGD_STRESS_SAMPLES':'256','SGD_GPU_TIMESTAMPS':'0'},started_at=now(),memory_before=memory(),warmup=warmup))
 start=time.monotonic();peak=0
 with (folder/'stdout.json').open('w') as out,(folder/'stderr.log').open('w') as err:
  proc=subprocess.Popen(['/usr/bin/time','-l',*cmd],cwd=ROOT,env=env,stdout=out,stderr=err)
  while proc.poll() is None:
   try:peak=max(peak,sum(p.memory_info().rss for p in psutil.Process(proc.pid).children(recursive=True)))
   except psutil.Error:pass
   if time.monotonic()-execute.last_progress>60:
    progress(j,active_elapsed=time.monotonic()-start);execute.last_progress=time.monotonic()
   time.sleep(.2)
 wall=time.monotonic()-start
 match=re.search(r'(\d+)\s+maximum resident set size',(folder/'stderr.log').read_text())
 if match:peak=max(peak,int(match[1]))
 row=dict(j,binary_sha256=file_sha256(run.binary),status='error',process_wall_seconds=wall,peak_rss_bytes=peak,memory_after=memory(),completed_at=now(),warmup=warmup)
 try:
  completed=subprocess.CompletedProcess(cmd,proc.returncode,(folder/'stdout.json').read_text(),(folder/'stderr.log').read_text())
  native=parse_and_validate_success(run,completed)
  assert native['nodes']==g['nodes']
  if j['method'].startswith('rr_'):assert native['rr_phase_chunks']==1
  if j['method']=='rr_sgd':assert native['rr_tile_size']==tile and native['rr_workgroup_size']==1024 and native['rr_outer_shuffle'] and native['rr_inner_shuffle']
  if native['stress_kind']=='sampled':assert native['stress_seed']==0 and native['stress_samples']==256
  if 'atomic' in j['method']:assert native['attempted_updates']==native['completed_updates'] and native['retry_failures']==0
  row.update(native);row.update(j);row['status']='success'
 except Exception as e:row['error']=repr(e)
 archive_text_artifacts(folder,row)
 save(dest,row)
 return row
execute.last_progress=0

def records(stage):return [json.loads(p.read_text()) for p in (BASE/'raw'/stage).glob('*/record.json')]
def progress(active=None,active_elapsed=0):
 jobs=json.loads((BASE/'config/jobs.json').read_text());rows=records('tile')+records('comparison');done={r['run_id'] for r in rows};remaining=[j for j in jobs if j['run_id'] not in done]
 groups=collections.defaultdict(list)
 for r in rows:groups[(r['stage'],r['graph'],r['method'],r['tile_size'])].append(r['process_wall_seconds'])
 eta=0;unknown=0
 for j in remaining:
  times=groups.get((j['stage'],j['graph'],j['method'],j['tile_size']))
  if times:eta+=statistics.median(times)
  else:unknown+=1
 state=dict(updated=now(),total=6300,finished=len(rows),success=sum(r['status']=='success' for r in rows),failed=sum(r['status']!='success' for r in rows),remaining=len(remaining),remaining_conditions=len({(j['stage'],j['graph'],j['method'],j['tile_size']) for j in remaining}),eta_measured_conditions_seconds=eta,eta_unmeasured_runs=unknown,active=active,active_elapsed_seconds=active_elapsed,memory=memory())
 save(BASE/'progress.json',state)
 msg=f"{now()} 完了{len(rows)}/6300・残り{len(remaining)}実行／{state['remaining_conditions']}設定。実測済み条件の残り約{eta/3600:.2f}時間、未推定{unknown}実行。実行中: {active['run_id'] if active else 'なし'} ({active_elapsed:.0f}秒)"
 print(msg,flush=True);(BASE/'PROGRESS.md').write_text('# v5 実験進捗\n\n'+msg+'\n\n全体の残り時間は、未測定条件が多いため未確定。設定間の所要時間差が大きいため単純な件数比例では推定しない。\n')
 sync_progress(state,msg)
 return state

def sync_progress(state,msg):
 p=VAULT/'0_概要.md';s=p.read_text();head='## v5 実行状況（自動更新）'
 body=f'{head}\n\n{msg}\n\n結果の正本: `{BASE}`。大規模8件は今回実行しない。速度とStressは同一実行から取得する。\n'
 if head in s:s=s[:s.index(head)].rstrip()+'\n\n'+body
 else:s=s.rstrip()+'\n\n'+body
 p.write_text(s)

def summarize(stage):
 rows=records(stage);out=BASE/'tables'/f'{stage}-runs.csv'
 fields=['run_id','graph','method','tile_size','seed','repetition','status','iteration_time_ms','algorithm_time_cold_ms','stress_value','stress_kind','process_wall_seconds','peak_rss_bytes','error']
 with out.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 grouped=collections.defaultdict(list)
 for r in rows:grouped[(r['graph'],r['method'],r['tile_size'])].append(r)
 summary=[]
 for (graph,method,tile),rs in grouped.items():
  item=dict(graph=graph,method=method,tile_size=tile,success_runs=sum(r['status']=='success' for r in rs),planned_runs=15)
  seeds=collections.defaultdict(list)
  for r in rs:
   if r['status']=='success':seeds[r['seed']].append(r)
  complete=[v for v in seeds.values() if len(v)==3];item['complete_seeds']=len(complete)
  for field in ['iteration_time_ms','algorithm_time_cold_ms','stress_value','peak_rss_bytes']:
   vals=[statistics.median(r[field] for r in v) for v in complete]
   for suffix,fun in [('median',statistics.median),('min',min),('max',max)]:item[field+'_'+suffix]=fun(vals) if vals else None
  summary.append(item)
 save(BASE/'tables'/f'{stage}-summary.json',summary)
 successes=sum(r['status']=='success' for r in rows)
 text=f'# v5 {stage} 結果と考察\n\n更新: {now()}\n\n完了記録{len(rows)}、成功{successes}。測定値は `{out}`、階層集計は `{BASE}/tables/{stage}-summary.json`。\n\n'
 text+='同seed3回の中央値を取り、完全な5seedの分布を比較する。成功率と欠測を併記し、失敗を0で埋めない。\n\n'
 if stage=='tile':
  for graph in TILES:
   rr=[x for x in summary if x['graph']==graph and x['method']=='rr_sgd' and x['complete_seeds']==5]
   if rr:
    fastest=min(rr,key=lambda x:x['iteration_time_ms_median']);text+=f"- {graph}: SGD時間の中央値が最小のTは{fastest['tile_size']}、15反復{fastest['iteration_time_ms_median']:.3f} ms、最終Stress {fastest['stress_value_median']:.6g}。\n"
  text+='\nTは時間とStressを併せて比較し、通常比較の共通Tは結果確認後に固定する。構造群は候補の比較であり、スケールフリー性や未知グラフへの一般化は実証していない。\n'
 (BASE/'reports'/f'{stage}.md').write_text(text)
 name='1_tilesize_考察.md' if stage=='tile' else '2_速度比較_考察.md'
 (VAULT/name).write_text(text)
 if stage=='comparison':(VAULT/'3_Stress比較_考察.md').write_text(text.replace('# v5 comparison 結果と考察','# v5 最終Stress 結果と考察'))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['tile','comparison','prepare','progress','summary'],default='tile');ap.add_argument('--max-runs',type=int);a=ap.parse_args();prepare()
 if a.stage=='prepare':progress();return
 if a.stage=='progress':progress();return
 if a.stage=='summary':summarize('tile');summarize('comparison');return
 lock=(BASE/'runner.lock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('別のv5ランナーが実行中です')
 lock.write(str(os.getpid()));lock.flush()
 stage=a.stage
 if stage=='comparison' and config()['normal_rr_full_tile'] is None:raise SystemExit('タイル実験後に共通Tをplan.jsonへ固定してください')
 graphs={g['name']:g for g in json.loads((BASE/'config/datasets.json').read_text())};jobs=[j for j in json.loads((BASE/'config/jobs.json').read_text()) if j['stage']==stage]
 session=now().replace(':','').replace('+','_');warmed=set();count=0
 for j in jobs:
  if (BASE/'raw'/stage/j['run_id']/'record.json').exists():continue
  if shutil.disk_usage(BASE).free<4*1024**3 or memory()['available_bytes']<2*1024**3:
   progress();raise SystemExit('容量確保のため次の実行前で停止。実行中の処理は打ち切っていません。再開可能です。')
  key=(j['graph'],j['method'],j['tile_size'])
  if key not in warmed:
   warm=dict(j,seed=999,repetition=0,run_id=f'warmup-{session}-{j["graph"]}-{j["method"]}-t{j["tile_size"] or "fixed"}')
   row=execute(warm,graphs[j['graph']],True)
   if row['status']!='success':raise SystemExit(f'ウォームアップ失敗: {row.get("error")}')
   warmed.add(key)
  progress(j);row=execute(j,graphs[j['graph']]);progress();count+=1
  if row['status']!='success':summarize(stage);raise SystemExit(f'出力確認に失敗。条件を変更せず停止: {row.get("error")}')
  if count%30==0:summarize(stage)
  if a.max_runs and count>=a.max_runs:break
 summarize(stage)
if __name__=='__main__':main()
