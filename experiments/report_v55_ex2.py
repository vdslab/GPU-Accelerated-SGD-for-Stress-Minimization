#!/usr/bin/env python3
"""v5.5 EX2 integrated report; only RR is newly measured."""
from pathlib import Path
import collections,csv,json,math,statistics,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_v55_ex2 import ROOT,BASE,OLD,VAULT,now,save
from run_ex2_rr128 import FIELDS,coverage

LABEL={'sgd':'CPU Full (historical)','atomic_sgd':'Atomic Full (historical Metal)','rr_sgd':'RR Full v5.5','sparse_sgd':'CPU Sparse (historical)','atomic_sparse_sgd':'Atomic Sparse (historical Metal)','rr_sparse_sgd':'RR Sparse v5.5'}
STYLE={'sgd':('#555555','s'),'sparse_sgd':('#555555','s'),'atomic_sgd':('#d97706','D'),'atomic_sparse_sgd':('#d97706','D'),'rr_sgd':('#2563eb','o'),'rr_sparse_sgd':('#2563eb','o')}

def hierarchy(rows,fun):
 values=[]
 for seed in range(5):
  rs=sorted([r for r in rows if r['seed']==seed],key=lambda r:r['repetition'])
  assert [r['repetition'] for r in rs]==[0,1,2]
  values.append(statistics.median(fun(r) for r in rs))
 return dict(median=statistics.median(values),min=min(values),max=max(values),seed_medians=values)

def csv_write(path,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def points(ax,stats,field,method,label=None):
 if not stats:return
 color,marker=STYLE[method];y=[s[field]['median'] for s in stats]
 ax.errorbar([s['nodes'] for s in stats],y,yerr=[[s[field]['median']-s[field]['min'] for s in stats],[s[field]['max']-s[field]['median'] for s in stats]],fmt=marker,color=color,capsize=2,ms=4,linestyle='none',label=label or LABEL[method]);ax.grid(alpha=.2)

def main():
 records=[json.loads(p.read_text()) for p in (BASE/'raw').glob('*/record.json')]
 rr=[r for r in records if r['status']=='success'];unavailable=[r for r in records if r['status']=='not_run_capacity']
 assert len(records)==1410 and len(rr)==1395 and len(unavailable)==15
 assert all(r['method'] in ['rr_sgd','rr_sparse_sgd'] and r['archive_status']=='success' and r['timing_complete'] for r in rr)
 for r in rr:coverage(r)
 previous=json.loads((BASE/'config/previous-rr-records.json').read_text());old={(r['graph'],r['method'],r['seed'],r['repetition']):r for r in previous}
 controls=json.loads((BASE/'config/cpu-atomic-reference-records.json').read_text())
 overlay={r['run_id']:r for r in json.loads((BASE/'config/full-exact-stress-overlay.json').read_text())}
 reference=[]
 for r in controls:
  q=dict(r,record_origin='historical_v5',newly_measured=False)
  if q['run_id'] in overlay:
   e=overlay[q['run_id']];assert e['graph']==q['graph'] and e['method']==q['method'] and e['seed']==q['seed'] and e['repetition']==q['repetition']
   q.update(native_historical_stress_value=q['stress_value'],native_historical_stress_kind=q['stress_kind'],native_historical_stress_eval_time_ms=q.get('stress_eval_time_ms'),stress_value=e['exact_stress_value'],stress_kind='exact',stress_samples=None,stress_seed=None,stress_eval_time_ms=None,stress_source='existing historical exact re-evaluation overlay; original record unchanged; exact evaluation duration not available')
  reference.append(q)
 current=[dict(r,record_origin='new_v5.5_rr',newly_measured=True) for r in rr]
 combined=reference+current
 assert len(reference)==2820 and len(combined)==4215
 save(BASE/'tables/historical-controls-view.json',reference)
 cpu={(r['graph'],r['family'],r['seed'],r['repetition']):r for r in reference if r['method'] in ['sgd','sparse_sgd']}
 grouped=collections.defaultdict(list)
 for r in combined:grouped[r['graph'],r['method']].append(r)
 summaries=[];paired=[];quality_missing=[]
 for (graph,method),rs in grouped.items():
  assert len(rs)==15
  s=dict(graph=graph,method=method,family=rs[0]['family'],nodes=rs[0]['nodes'],edges=rs[0]['edges'],stress_kind=rs[0]['stress_kind'],successful_runs=15,complete_seeds=5,record_origin=rs[0]['record_origin'])
  for field in ['sgd_time_ms','total_compute_time_ms','stress_value','peak_rss_bytes']:
   s[field]=hierarchy(rs,lambda r,f=field:r[f])
  for field in ['sgd_time_ms','total_compute_time_ms']:
   s['cpu_speedup_'+field]=hierarchy(rs,lambda r,f=field:cpu[r['graph'],r['family'],r['seed'],r['repetition']][f]/r[f])
  compatible=all((r['stress_kind'],r.get('stress_samples'),r.get('stress_seed'))==(cpu[r['graph'],r['family'],r['seed'],r['repetition']]['stress_kind'],cpu[r['graph'],r['family'],r['seed'],r['repetition']].get('stress_samples'),cpu[r['graph'],r['family'],r['seed'],r['repetition']].get('stress_seed')) for r in rs)
  if compatible:s['cpu_stress_ratio']=hierarchy(rs,lambda r:r['stress_value']/cpu[r['graph'],r['family'],r['seed'],r['repetition']]['stress_value'])
  else:quality_missing.append(dict(graph=graph,method=method,reason='Evaluation protocols differ; CPU stress ratio omitted'))
  if method in ['rr_sgd','rr_sparse_sgd']:
   for r in rs:
    o=old[graph,method,r['seed'],r['repetition']]
    assert r['input_sha256']==o['input_sha256'] and r['initial_positions_sha256']==o['initial_positions_sha256']
    if method=='rr_sparse_sgd':assert r['preprocess_sha256']==o['preprocess_sha256']
    paired.append(dict(graph=graph,method=method,seed=r['seed'],repetition=r['repetition'],previous_run_id=o['run_id'],previous_over_new_sgd=o['sgd_time_ms']/r['sgd_time_ms'],previous_over_new_total=o['total_compute_time_ms']/r['total_compute_time_ms'],stress_new_over_previous=r['stress_value']/o['stress_value']))
   for field in FIELDS+['stress_eval_time_ms']:
    s[field]=hierarchy(rs,lambda r,f=field:statistics.mean(r[f]) if isinstance(r[f],list) else r[f])
   p=[r for r in paired if r['graph']==graph and r['method']==method]
   for field in ['previous_over_new_sgd','previous_over_new_total','stress_new_over_previous']:s[field]=hierarchy(p,lambda r,f=field:r[f])
  summaries.append(s)
 save(BASE/'tables/hierarchical.json',summaries);save(BASE/'tables/paired-previous-rr.json',paired);save(BASE/'validation/quality-comparability.json',dict(omitted_cpu_stress_ratios=quality_missing))
 cross_graph=[]
 for method in ['rr_sgd','rr_sparse_sgd']:
  ss=[s for s in summaries if s['method']==method]
  cross_graph.append(dict(method=method,graphs=len(ss),aggregation='equal-weight geometric mean of graph-level hierarchical medians of paired run ratios',sgd_speedup_geomean=math.exp(statistics.mean(math.log(s['previous_over_new_sgd']['median']) for s in ss)),total_speedup_geomean=math.exp(statistics.mean(math.log(s['previous_over_new_total']['median']) for s in ss)),stress_ratio_min=min(s['stress_new_over_previous']['median'] for s in ss),stress_ratio_max=max(s['stress_new_over_previous']['median'] for s in ss)))
 save(BASE/'tables/cross-graph-summary.json',cross_graph)
 csv_write(BASE/'tables/rr-runs.csv',rr);csv_write(BASE/'tables/combined-runs.csv',combined);csv_write(BASE/'tables/paired-previous-rr.csv',paired);csv_write(BASE/'tables/not-run-capacity.csv',unavailable)
 flat=[]
 for s in summaries:
  q={k:v for k,v in s.items() if not isinstance(v,dict)}
  for k,v in s.items():
   if isinstance(v,dict):
    for stat in ['median','min','max']:q[k+'_'+stat]=v[stat]
  flat.append(q)
 csv_write(BASE/'tables/summary.csv',flat)
 coverage_info={m:{f:sum(r.get(f) is not None for r in rr if r['method']==m) for f in FIELDS} for m in ['rr_sgd','rr_sparse_sgd']}
 save(BASE/'validation/timing-coverage.json',coverage_info)
 figures=[]
 for family,rrmethod,methods in [('full','rr_sgd',['sgd','atomic_sgd','rr_sgd']),('sparse','rr_sparse_sgd',['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd'])]:
  for metric,ylabel,suffix in [('cpu_speedup_sgd_time_ms','SGD speedup / historical CPU','sgd-speedup'),('cpu_speedup_total_compute_time_ms','Total-compute speedup / historical CPU','total-speedup'),('cpu_stress_ratio','Full Stress / protocol-matched historical CPU','stress-ratio'),('stress_value','Full Stress (exact / sampled by protocol)','stress-absolute'),('total_compute_time_ms','Total compute without frame (ms)','total-time')]:
   fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
   for method in methods:
    ss=sorted([s for s in summaries if s['method']==method and metric in s],key=lambda s:s['nodes']);points(ax,ss,metric,method)
   ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Number of vertices');ax.set_ylabel(ylabel)
   if metric.startswith('cpu_'):ax.axhline(1,color='#aaa',ls=':')
   ax.set_title(f'v5.5 {family}: RR new; CPU / Atomic historical; 5-seed median and range');ax.legend(fontsize=8)
   name=family+'-'+suffix;figures.append(name)
   for ext in ['png','pdf']:fig.savefig(BASE/f'figures/{name}.{ext}',dpi=160)
   plt.close(fig)
  ss=sorted([s for s in summaries if s['method']==rrmethod],key=lambda s:s['nodes'])
  fig,axs=plt.subplots(5,2,figsize=(13,18),layout='constrained')
  labels=['1 Preprocessing','2 GPU preparation','3 Upload','4 GPU compute / iteration','5 SGD elapsed','6 Readback','7 Result-to-frame','8 Total including frame','9 GPU compute total','10 Total excluding frame']
  for ax,field,label in zip(axs.flat,FIELDS,labels):
   points(ax,ss,field,rrmethod);ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Number of vertices');ax.set_ylabel('ms');ax.set_title(label)
  fig.suptitle(f'v5.5 RR {family}: 10 measured intervals; first-frame timing includes handoff')
  name=family+'-all-timings';figures.append(name)
  for ext in ['png','pdf']:fig.savefig(BASE/f'figures/{name}.{ext}',dpi=160)
  plt.close(fig)
  fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
  for ax,field,label in zip(axs,['previous_over_new_sgd','previous_over_new_total','stress_new_over_previous'],['Previous RR128 / new SGD time','Previous RR128 / new total time','New / previous Full Stress']):
   points(ax,ss,field,rrmethod);ax.axhline(1,color='gray',ls=':');ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Number of vertices');ax.set_ylabel(label)
  fig.suptitle(f'v5.5 {family}: same W128, previous EX2 comparison (cross-session)')
  name=family+'-previous-comparison';figures.append(name)
  for ext in ['png','pdf']:fig.savefig(BASE/f'figures/{name}.{ext}',dpi=160)
  plt.close(fig)
 text='''# v5.5 EX2：RR Full／Sparse速度・Stress比較

**EX2のみ実施。新規RR測定1,395件成功、容量上限で15件未実行。CPU・Atomicの新規実行0件、過去2,820件を参照。EX1は742件で停止した途中結果として別フォルダに保持し、このEX2集計へ混ぜない。**

前回EX2と同じ56入力、Full38条件・Sparse56条件、seed0〜4×各3回。15反復、epsilon0.1、RR Full T=W128/shared/内外shuffleあり、RR Sparse W128・p=min(200,n)、phase_chunks1、auto BFS。GPU timestamp有効。各graph/methodのseed999ウォームアップを除外し、逐次測定した。時間打切りなし。代表小/大入力のtimestamp有無で保存座標一致、全10計時項目の取得を事前確認した。

## 容量上限の未実行

現PRのSparse one-sided配列はdense n*p、8bytes/entry、4bank各128MiBで最大67,108,864項目。coPapersCiteseer n=434,102・p=200は86,820,400項目となり、ソースの明示的上限を超える。該当15件は数学的事前確認でnot_run_capacityとし、割当・SGD・評価を開始していない。pを減らして別条件を混ぜず、時間・Stressを0補完しない。RR Full570件、RR Sparse825件が成功。旧版で完了していた入力が現PRでは上限外になったことを制約として残す。

## 評価・集計

Fullは全38入力で厳密Full Stress。Sparseはn<=8,000で厳密、それ以上は256固定始点・評価seed0の推定Full Stress（Sparse目的関数ではない）。初期座標hashとSparse前処理fingerprintを前回EX2と照合。同じ最終座標から速度計時終了後にStress評価・Afterの実GPU描画を行った。2048×2048、blue、半径1px。F→Gは中心化・Stress評価・座標保存/読込・プロセス起動・描画準備を含み、純粋な描画kernel時間ではない。PNGエンコード・保存はF→G外。PNGはhash/サイズを記録して省略し、初期/最終座標を復元検証付きで保存した。

同seed3回中央値→5seedの中央値・最小〜最大。比は対応するgraph/method/seed/repetitionごとに計算して階層集計。誤差棒はseed中央値の範囲で、信頼区間ではない。欠測と未実行を除外し成功数を併記。10計時項目は重複を含み、単純に積み上げて総時間としない。

## 過去CPU・Atomicの参照

CPU/Atomicはv5通常比較の過去2,820件を読み取り専用で使用。Atomic参照は旧Metal実装であり、今回のPRのwgpu Atomicを測り直したものではない。過去のFull sampled Stressは、既存の保存座標から作成済みの厳密再評価overlayがあるrunだけ、新しい参照ビュー内で厳密値を使用した。元のrecordは変更しない。評価方式が揃わないCPU比は未算出とし、validation/quality-comparability.jsonへ記録した。

RRの前回比は、前回EX2のRR128・timestamp有効の1,410件を基準にする。W1024・timestamp無効の旧v5 RRと混同しない。CPU/Atomicとの速度比は異なる日時・計測条件の過去参照であり同時測定ではない。温度・電源・常駐アプリを専用に固定しておらず、小さい差の断定は避ける。

## 図表

'''
 text+='前回EX2 RR128に対する、グラフ別・対応実行比の階層中央値を各グラフ等重みで幾何平均した参考集計：\n\n| 方式 | 比較グラフ数 | SGD速度比（前回/今回） | 総計算速度比 | Stress比のグラフ別中央値範囲 |\n|---|---:|---:|---:|---:|\n'
 for d in cross_graph:text+=f"| {d['method']} | {d['graphs']} | {d['sgd_speedup_geomean']:.3f} | {d['total_speedup_geomean']:.3f} | {d['stress_ratio_min']:.5f}–{d['stress_ratio_max']:.5f} |\n"
 text+='\n既存厳密Stress overlayの評価時間は不明のため、新しい参照ビューのstress_eval_time_msを未計測とし、元の推定評価時間はnative_historical_stress_eval_time_msとして保持する。CPUの追加実行・再評価は行っていない。\n\n'
 for name in figures:text+=f'![[EX2_v5.5_図表/{name}.png]]\n\n'
 text+=f'## 保存先\n\n`{BASE}`\n\nconfig/に入力・条件・順序・参照・source/binary SHA、raw/に測定JSON・コマンド・ログ・圧縮座標、validation/に予備検証・計時充足・比較可能性、tables/に統合CSV/JSON・対応する前回比、figures/にPNG/PDF。過去のv5 EX1/EX2・CPU/Atomic recordのハッシュ不変を完了時に確認。v5.5のEX1途中データは別保管。\n'
 (BASE/'reports/results.md').write_text(text)
 (VAULT/'EX2_v5.5_速度ストレス実験_結果と考察.md').write_text(text)
 assets=VAULT/'EX2_v5.5_図表';assets.mkdir(exist_ok=True)
 for p in (BASE/'figures').glob('*.png'):shutil.copy2(p,assets/p.name)
 save(BASE/'validation/completion-validation.json',dict(requested=1410,rr_success=1395,capacity_not_run=15,full_success=570,sparse_success=825,cpu_new=0,atomic_new=0,historical_controls=2820,combined_available_records=4215,timing_fields=10,initial_fingerprints_match_previous=True,sparse_preprocess_matches_previous=True,complete_seeds_per_condition=5,repetitions=3,figure_names=figures))
 print('v5.5 EX2 integrated report complete',flush=True)

if __name__=='__main__':main()
