"""Arithmetic-only comparison requested by the author; no plots or measurements."""
import collections
import csv
import hashlib
import json
import math
import statistics as stats

from run_v5 import ROOT, VAULT, now, save

BASE=ROOT/'experiment_results/v5'
OUT=BASE/'execution-cost'
CPU={'sgd','sparse_sgd'}
NAMES={'sgd':'CPU Full','atomic_sgd':'Atomic Full','rr_sgd':'RoundRobin Full',
       'sparse_sgd':'CPU Sparse','atomic_sparse_sgd':'Atomic Sparse','rr_sparse_sgd':'RoundRobin Sparse'}


def read(p):return json.loads(p.read_text())
def csvwrite(path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def dist(v):return dict(median=stats.median(v),min=min(v),max=max(v))
def gmean(v):return math.exp(stats.mean(math.log(x) for x in v))


def main():
 OUT.mkdir(exist_ok=True);records=[]
 for stage,path in [('tile',BASE/'raw/tile'),('comparison',BASE/'raw/comparison'),('large',BASE/'large/raw')]:
  for p in path.glob('*/record.json'):records.append(dict(read(p),stage=stage,record_path=str(p)))
 assert len(records)==6380
 cpu={(r['stage'],r['graph'],r['method'],r['seed'],r['repetition']):r for r in records if r['status']=='success' and r['method'] in CPU}
 runs=[]
 for r in records:
  row={k:r.get(k) for k in ['stage','graph','method','nodes','pivots','tile_size','seed','repetition','run_id','status','iterations','record_path']}
  row.update(is_cpu=r['method'] in CPU)
  if r['status']!='success':runs.append(row);continue
  assert r['iterations']==15
  sgd=r['sgd_time_ms'];assert math.isclose(sgd,r['iteration_time_ms'],rel_tol=1e-12)
  if row['is_cpu']:
   requested=with_upload=sgd
  else:
   for key in ['gpu_prepare_time_ms','readback_time_ms','upload_time_ms']:assert isinstance(r.get(key),(float,int)) and r[key]>=0
   requested=r['gpu_prepare_time_ms']+sgd+r['readback_time_ms']
   with_upload=requested+r['upload_time_ms']
  row.update(sgd_time_ms=sgd,gpu_prepare_time_ms=r.get('gpu_prepare_time_ms'),readback_time_ms=r.get('readback_time_ms'),upload_time_ms=r.get('upload_time_ms'),
             requested_compute_ms=requested,compute_with_upload_ms=with_upload,
             amortized_requested_per_iteration_ms=requested/15,amortized_with_upload_per_iteration_ms=with_upload/15,
             common_preprocess_time_ms=r.get('common_preprocess_time_ms'),preprocess_time_ms=r.get('preprocess_time_ms'),method_setup_time_ms=r.get('method_setup_time_ms'))
  if r['stage']!='large':
   method='sparse_sgd' if r['family']=='sparse' else 'sgd';ref=cpu[(r['stage'],r['graph'],method,r['seed'],r['repetition'])]
   assert r['input_sha256']==ref['input_sha256'] and r['initial_positions_sha256']==ref['initial_positions_sha256']
   if r['family']=='sparse':assert r['preprocess_sha256']==ref['preprocess_sha256']
   row.update(cpu_sgd_time_ms=ref['sgd_time_ms'],requested_cpu_speedup=ref['sgd_time_ms']/requested,with_upload_cpu_speedup=ref['sgd_time_ms']/with_upload,sgd_cpu_speedup=ref['sgd_time_ms']/sgd)
  runs.append(row)
 csvwrite(OUT/'runs.csv',runs)
 fields=['sgd_time_ms','gpu_prepare_time_ms','readback_time_ms','upload_time_ms','requested_compute_ms','compute_with_upload_ms',
         'amortized_requested_per_iteration_ms','amortized_with_upload_per_iteration_ms','cpu_sgd_time_ms','requested_cpu_speedup','with_upload_cpu_speedup','sgd_cpu_speedup','common_preprocess_time_ms','preprocess_time_ms','method_setup_time_ms']
 grouped=collections.defaultdict(list)
 for r in runs:grouped[(r['stage'],r['graph'],r['method'],r['tile_size'],r['pivots'])].append(r)
 summary=[];within=[]
 for (stage,graph,method,tile,pivots),rs in grouped.items():
  good=[r for r in rs if r['status']=='success'];item=dict(stage=stage,graph=graph,method=method,tile_size=tile,pivots=pivots,nodes=rs[0]['nodes'],success=len(good),planned=len(rs),status_counts=dict(collections.Counter(r['status'] for r in rs)))
  if good:
   seeds=sorted({r['seed'] for r in good});assert len(seeds)==5
   for field in fields:
    values=[]
    for seed in seeds:
     seed_rows=[r for r in good if r['seed']==seed];assert len(seed_rows)==(1 if stage=='large' else 3)
     v=[r.get(field) for r in seed_rows]
     if all(x is not None for x in v):
      med=stats.median(v);values.append(med);within.append(dict(stage=stage,graph=graph,method=method,tile_size=tile,pivots=pivots,seed=seed,field=field,median=med))
     else:assert all(x is None for x in v)
    if values:
     for name,value in dist(values).items():item[field+'_'+name]=value
   assert item['compute_with_upload_ms_median']>=item['requested_compute_ms_median']>=item['sgd_time_ms_median']
  summary.append(item)
 summary.sort(key=lambda r:(r['stage'],r['nodes'] or 0,r['graph'],r['method'],r['tile_size'] or 0,r['pivots'] or 0))
 assert len(summary)==472
 csvwrite(OUT/'summary.csv',summary);save(OUT/'summary.json',summary);csvwrite(OUT/'within-seed.csv',within)
 cross=[]
 for method in ['atomic_sgd','rr_sgd','atomic_sparse_sgd','rr_sparse_sgd']:
  rs=[r for r in summary if r['stage']=='comparison' and r['method']==method];a=dict(method=method,graphs=len(rs))
  for metric in ['sgd_cpu_speedup','requested_cpu_speedup','with_upload_cpu_speedup']:
   vals=[r[metric+'_median'] for r in rs];a[metric+'_gmean']=gmean(vals);a[metric+'_faster_graphs']=sum(v>1 for v in vals)
  cross.append(a)
 save(OUT/'cross-graph.json',cross);csvwrite(OUT/'cross-graph.csv',cross)
 write_report(summary,cross)
 save(OUT/'validation.json',dict(at=now(),status='passed',input_records=6380,successful_records=6160,unexecuted_gpu_limit=220,
                               matched_normal_gpu_runs=2820,matched_tile_gpu_runs=1650,large_gpu_runs_without_cpu_reference=130,
                               summary_groups=472,formula='gpu_prepare_time_ms + sgd_time_ms + readback_time_ms; alternative also adds upload_time_ms; CPU baseline=sgd_time_ms',
                               aggregation='Per-run sums and paired ratios first, median of repetitions per seed, then median/min/max of five seeds',plots_created=0,new_measurements=0))
 print(json.dumps(dict(cross=cross,luxembourg=[r for r in summary if r['stage']=='comparison' and r['graph']=='luxembourg_osm']),ensure_ascii=False,indent=2))


def write_report(summary,cross):
 text='# GPU準備・SGD・読み戻しを含む比較\n\n'
 text+='ユーザー指定の合計A＝GPU準備＋SGD経過時間＋読み戻し。比較値B＝A＋初期転送。CPU側は同族CPUのSGD経過時間を使う。全て15反復の1実行あたりで、GPU準備・読み戻しは各1回加算し15倍しない。前処理・Stress評価・保存・描画は含めない。表の÷15は準備等を15反復に配賦した平均であり、毎反復GPUを初期化する意味ではない。\n\n'
 text+='合計・CPU比をrunごとに先に計算し、同seed3回の中央値→5seedの中央値・最小最大で集計。**各成分の中央値を足した値ではない。** 大規模はseed各1回。再実行・新しい図の作成は行っていない。\n\n'
 text+='## 前処理について\n\n同族内の初期配置・入力、およびSparseのpivot・距離・制約等のfingerprintは一致する。CPU/Atomic Sparseは共有の参照実装で制約一覧を作り、RoundRobin Sparseは同等の制約を省メモリ形式に組み立てる独立実装である。計算内容・結果の同等性と、実装・前処理時間の同一性は区別する。GPU固有のスケジュール/データ構築（method_setup_time_ms）は今回のA/Bには含めていないため、全GPU追加費用や処理全体を包含した値とは呼ばない。描画はこの実験では未計測で、同時間であると実証していない。\n\n'
 text+='## 通常比較の要約\n\n各グラフを同じ重みで扱う幾何平均。速度比はCPU/GPUで、1より大きいと高速。\n\n| 方式 | グラフ数 | SGDのみ | A：準備＋SGD＋読戻し | B：さらに初期転送込み | Aで高速な件数 |\n|---|---:|---:|---:|---:|---:|\n'
 for r in cross:text+=f'| {NAMES[r["method"]]} | {r["graphs"]} | {r["sgd_cpu_speedup_gmean"]:.3f}倍 | {r["requested_cpu_speedup_gmean"]:.3f}倍 | {r["with_upload_cpu_speedup_gmean"]:.3f}倍 | {r["requested_cpu_speedup_faster_graphs"]}/{r["graphs"]} |\n'
 for family,methods in [('Full',['sgd','atomic_sgd','rr_sgd']),('Sparse',['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd'])]:
  text+='\n## '+family+'：全グラフ\n\n時間は15反復1実行の中央値（ms）。CPU列はCPU SGD。GPU列は合計Aと対応CPU比。\n\n| グラフ | CPU ms | Atomic A ms | CPU/Atomic | RoundRobin A ms | CPU/RoundRobin |\n|---|---:|---:|---:|---:|---:|\n'
  selected=[r for r in summary if r['stage']=='comparison' and r['method'] in methods];idx={(r['graph'],r['method']):r for r in selected}
  for r in selected:
   if r['method']!=methods[0]:continue
   a=idx[(r['graph'],methods[1])];rr=idx[(r['graph'],methods[2])]
   text+=f'| {r["graph"]} | {r["sgd_time_ms_median"]:.3f} | {a["requested_compute_ms_median"]:.3f} | {a["requested_cpu_speedup_median"]:.3f} | {rr["requested_compute_ms_median"]:.3f} | {rr["requested_cpu_speedup_median"]:.3f} |\n'
 text+='\n## luxembourg_osmの内訳\n\n各列はそれぞれの階層中央値なので、成分中央値の和と合計中央値には小差が生じうる。全てms。\n\n| 方式 | GPU準備 | SGD15反復 | 読み戻し | 合計A | 初期転送 | 合計B | A÷15 | CPU/A | CPU/B |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for method in ['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd']:
  r=next(r for r in summary if r['stage']=='comparison' and r['graph']=='luxembourg_osm' and r['method']==method)
  fields=['gpu_prepare_time_ms','sgd_time_ms','readback_time_ms','requested_compute_ms','upload_time_ms','compute_with_upload_ms','amortized_requested_per_iteration_ms','requested_cpu_speedup','with_upload_cpu_speedup']
  text+='| '+NAMES[method]+' | '+' | '.join('対象外' if r.get(k+'_median') is None else f'{r[k+"_median"]:.3f}' for k in fields)+' |\n'
 text+='\n## 全結果ファイル\n\n- runs.csv：全6380条件の状態、合計A/B、対応CPU比。220未実行は空欄。\n- summary.csv / summary.json：通常282設定・タイル120設定・大規模70設定の5seed集計。転送込みB、平均1反復換算、min/maxも保存。\n- within-seed.csv：seed内中央値。\n- cross-graph.csv / cross-graph.json：通常全グラフの方式別要約。\n\n大規模130成功には対応CPUがないため、A/Bは計算したがCPU比は空欄とした。GPU上限による未実行220条件も成功の測定値で埋めない。タイル分も別stageとして全Tを計算し、通常比較と混ぜて幾何平均しない。\n'
 (OUT/'README.md').write_text(text)
 note=VAULT/'6_GPU準備と読み戻し込みの比較.md';note.write_text('---\ntags: [area/大学院研究, 実験考察]\nupdated: '+now()[:10]+'\n---\n\n[[0_概要]] / [[5_実験結果まとめと論文用チャート]]\n\n'+text+'\n正本：`'+str(OUT)+'`。\n')
 index=VAULT/'0_概要.md';s=index.read_text();link='[[6_GPU準備と読み戻し込みの比較]]'
 if link not in s:index.write_text(s.rstrip()+'\n\nGPU準備＋SGD＋読み戻しの全グラフ比較表：'+link+'。\n')


if __name__=='__main__':main()
