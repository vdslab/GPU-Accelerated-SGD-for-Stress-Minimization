#!/usr/bin/env python3
"""Offline EX2 report; no plotting while native measurements are running."""
from pathlib import Path
import collections,csv,json,math,os,shutil,statistics,tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiment_results/v5/ex2-rr128-speed-stress-20261108'
VAULT=Path('/Users/kose/Documents/ChatGPT/Obsidian タスク管理/sakamoti55/12_大学院研究/実験')
METRICS=[('preprocess_time_ms','1: Preprocessing'),('gpu_prepare_time_ms','2: GPU preparation'),('upload_time_ms','3: Initial upload'),('iteration_times_ms','4: GPU compute / iteration'),('sgd_time_ms','5: SGD elapsed'),('readback_time_ms','6: Readback'),('draw_time_ms','7: Result-to-frame elapsed'),('total_with_draw_time_ms','8: Total including frame'),('iteration_total_time_ms','9: GPU compute total'),('total_compute_time_ms','10: Total without frame')]
METHODS=['rr_sgd','rr_sparse_sgd']
LABELS={'rr_sgd':'RR Full, T=W=128, exact Stress','rr_sparse_sgd':'RR Sparse, W=128, previous Stress protocol'}

def atomic_note(path,text):
 temporary=None
 try:
  with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix='.ex2-report-',suffix='.tmp',delete=False) as f:
   temporary=Path(f.name);f.write(text);f.flush();os.fsync(f.fileno())
  os.replace(temporary,path);temporary=None
 finally:
  if temporary is not None:temporary.unlink(missing_ok=True)

def hierarchy(rows,fun):
 per=[]
 for seed in range(5):
  rs=sorted([r for r in rows if r['seed']==seed],key=lambda r:r['repetition'])
  assert [r['repetition'] for r in rs]==[0,1,2]
  per.append(statistics.median(fun(r) for r in rs))
 return dict(median=statistics.median(per),min=min(per),max=max(per),seed_medians=per)

def errorbar(ax,x,stats,color='tab:blue',marker='o',label=None):
 y=[s['median'] for s in stats];low=[s['median']-s['min'] for s in stats];high=[s['max']-s['median'] for s in stats]
 ax.errorbar(x,y,yerr=[low,high],fmt=marker,color=color,capsize=2,ms=4,label=label,linestyle='none')
 ax.grid(alpha=.2)

def main():
 rows=[json.loads(p.read_text()) for p in (BASE/'raw').glob('*/record.json')]
 assert len(rows)==1410 and len({r['run_id'] for r in rows})==1410
 assert all(r['status']=='success' and r['archive_status']=='success' and r['timing_complete'] for r in rows)
 assert all(len(r['iteration_times_ms'])==15 for r in rows)
 old={}
 for p in (ROOT/'experiment_results/v5/raw/comparison').glob('*/record.json'):
  r=json.loads(p.read_text())
  if r['method'] in METHODS:old[r['graph'],r['method'],r['seed'],r['repetition']]=r
 groups=collections.defaultdict(list)
 for r in rows:groups[r['method'],r['graph']].append(r)
 summaries=[]
 fields=[k for k,label in METRICS]+['stress_value','stress_eval_time_ms','peak_rss_bytes']
 for (method,graph),rs in sorted(groups.items()):
  s=dict(method=method,graph=graph,nodes=rs[0]['nodes'],edges=rs[0]['edges'],stress_kind=rs[0]['stress_kind'],complete_seeds=5,successful_runs=15)
  assert all(r['initial_positions_sha256']==old[graph,method,r['seed'],r['repetition']]['initial_positions_sha256'] for r in rs)
  if method=='rr_sgd':assert all(r['rr_tile_size']==r['rr_workgroup_size']==128 and r['stress_kind']=='exact' for r in rs)
  else:assert all(r['configured_workgroup_size']==128 and r['preprocess_sha256']==old[graph,method,r['seed'],r['repetition']]['preprocess_sha256'] for r in rs)
  for field in fields:
   s[field]=hierarchy(rs,lambda r,f=field:statistics.mean(r[f]) if isinstance(r[f],list) else r[f])
  for field in ['sgd_time_ms','total_compute_time_ms']:
   s['previous_over_new_'+field]=hierarchy(rs,lambda r,f=field:old[graph,method,r['seed'],r['repetition']][f]/r[f])
  summaries.append(s)
 (BASE/'tables/hierarchical.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
 flat=[]
 for s in summaries:
  o={k:v for k,v in s.items() if not isinstance(v,dict)}
  for k,v in s.items():
   if isinstance(v,dict):
    for stat in ['median','min','max']:o[k+'_'+stat]=v[stat]
  flat.append(o)
 with (BASE/'tables/summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 # Actual timing coverage, including the per-iteration array.
 coverage={method:{k:sum(r.get(k) is not None for r in rows if r['method']==method) for k,label in METRICS} for method in METHODS}
 (BASE/'validation/timing-coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
 figure_names=[]
 for method in METHODS:
  stats=sorted([s for s in summaries if s['method']==method],key=lambda s:(s['nodes'],s['graph']));x=[s['nodes'] for s in stats];prefix='full' if method=='rr_sgd' else 'sparse'
  fig,axs=plt.subplots(5,2,figsize=(13,18),layout='constrained')
  for ax,(field,label) in zip(axs.flat,METRICS):
   errorbar(ax,x,[s[field] for s in stats]);ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Number of vertices');ax.set_ylabel('ms');ax.set_title(label)
  fig.suptitle(LABELS[method]+' — 10 measured intervals; 5-seed median/min–max')
  name=prefix+'-all-timings';figure_names.append(name);fig.savefig(BASE/f'figures/{name}.png',dpi=160);fig.savefig(BASE/f'figures/{name}.pdf');plt.close(fig)
  fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
  for ax,field,label in zip(axs,['sgd_time_ms','total_compute_time_ms'],['SGD elapsed','Total without frame']):
   errorbar(ax,x,[s['previous_over_new_'+field] for s in stats]);ax.axhline(1,color='gray',ls='--');ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Number of vertices');ax.set_ylabel('Previous RR time / RR128 time');ax.set_title(label)
  fig.suptitle(LABELS[method]+' — cross-session comparison, timestamps now enabled')
  name=prefix+'-previous-speedup';figure_names.append(name);fig.savefig(BASE/f'figures/{name}.png',dpi=160);fig.savefig(BASE/f'figures/{name}.pdf');plt.close(fig)
  fig,ax=plt.subplots(figsize=(max(11,len(stats)*.24),5),layout='constrained')
  errorbar(ax,list(range(len(stats))),[s['stress_value'] for s in stats]);ax.set_xticks(range(len(stats)),[s['graph']+(' [sampled]' if s['stress_kind']=='sampled' else '') for s in stats],rotation=75,ha='right');ax.set_yscale('log');ax.set_ylabel('Final Full Stress (absolute)');ax.set_title(LABELS[method]+' — exact / sampled explicitly labeled')
  name=prefix+'-stress';figure_names.append(name);fig.savefig(BASE/f'figures/{name}.png',dpi=160);fig.savefig(BASE/f'figures/{name}.pdf');plt.close(fig)
  fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
  for kind,marker in [('exact','o'),('sampled','^')]:
   ss=[s for s in stats if s['stress_kind']==kind]
   if ss:ax.scatter([s['previous_over_new_sgd_time_ms']['median'] for s in ss],[s['stress_value']['median'] for s in ss],marker=marker,label=kind)
  ax.set_xscale('log');ax.set_yscale('log');ax.axvline(1,color='gray',ls='--');ax.set_xlabel('Previous RR SGD time / RR128 time');ax.set_ylabel('Final Full Stress (absolute)');ax.legend();ax.set_title(LABELS[method])
  name=prefix+'-speed-stress';figure_names.append(name);fig.savefig(BASE/f'figures/{name}.png',dpi=160);fig.savefig(BASE/f'figures/{name}.pdf');plt.close(fig)
 text='''---
tags: [area/大学院研究, 実験結果]
updated: DATE
status: 測定完了
---

# EX2：RR128 速度・Stress実験の結果と考察

[[EX2_速度ストレス実験]] / [[9_計測時間]]

RR Full 38グラフ×5seed×3回＝570実行、RR Sparse 56グラフ×5seed×3回＝840実行。計1,410実行すべて成功。
FullはT=W=128、shared・内外shuffle・phase_chunks1・全件厳密Full Stress。SparseはW128、p=min(200,n)、phase_chunks1・従来のpivot順序、n≤8000は厳密、それ以上は256始点・評価seed0の推定Full Stress。
15反復・epsilon0.1。同seed3回中央値→5seed中央値・最小〜最大。9項目＋追加⑩を全件取得し、④は15反復分、⑨はその合計として検証した。

描画は2048×2048、全辺、ノード半径1px、blue。⑦は結果取得Fから最終配置の最初のGPU描画完了Gまでで、Stress評価・保存/読込・起動/準備を含む。⑧も同じ待ちを含む。⑦を純GPU描画時間と解釈せず、品質評価時間は別途保存した。PNGのエンコード・保存はG以後で除外した。

## 前回との速度比較

前回RR/今回RRを同じgraph・seed・repetitionで求めてから階層集計。1より大きい値は今回が高速。異なる時期の測定で、今回はGPU timestampと描画が有効なため、差の全てをW/T変更だけの効果に帰属しない。

| 方式 | グラフ数 | SGD速度比の幾何平均 | 最小〜最大（グラフ間） |
|---|---:|---:|---:|
'''
 for method in METHODS:
  ss=[s for s in summaries if s['method']==method];vs=[s['previous_over_new_sgd_time_ms']['median'] for s in ss];gm=math.exp(statistics.mean(map(math.log,vs)))
  text+=f"| {method} | {len(ss)} | {gm:.3f} | {min(vs):.3f}〜{max(vs):.3f} |\n"
 text+='''
## 解釈と制約

FullとSparseは目的関数・更新対象・初期配置が異なるため、速度差やStress差の全てをスケジューリングだけの効果とはしない。SparseのW128が他の全サイズより最速であることを示す探索実験ではない。
Fullは今回厳密Stress、前回は大きいグラフで推定Stressだった。評価方式を揃えない品質比を作らず、ここでは今回の絶対Stressを報告した。CPU・Atomicは今回再測定しておらず、新規測定の同時期の基準として扱わない。

## 図表

'''
 for name in figure_names:text+=f'![[EX2_v5_図表/{name}.png]]\n\n'
 text+='''## 保存先

`experiment_results/v5/ex2-rr128-speed-stress-20261108/`

- raw/：各実行のコマンド・native出力・rendererログ・記録・復元検証済み座標。
- tables/runs.csv、hierarchical.json、summary.csv：生データ・階層集計。
- validation/preflight-complete.json：timestamp有無の座標一致、Sparse W256/W128の一致、代表条件での10項目検証。
- validation/timing-coverage.json：全記録の計時取得数。
- figures/：チャートPNG・PDF。source-snapshot/・bin/・config/：再現情報。

前回のv5・EX1記録は保持し、開始前と完了後にハッシュの不変を確認した。
'''
 from run_v5 import now
 text=text.replace('DATE',now()[:10]);(BASE/'reports/RESULTS.md').write_text(text);atomic_note(VAULT/'EX2_速度ストレス実験_結果と考察.md',text)
 assets=VAULT/'EX2_v5_図表';assets.mkdir(exist_ok=True)
 for name in figure_names:shutil.copy2(BASE/f'figures/{name}.png',assets/f'{name}.png')
 note=VAULT/'EX2_速度ストレス実験.md';s=note.read_text();marker='## 確定結果（自動出力）'
 body=marker+'\n\n**測定・集計完了：1,410/1,410成功。** 詳細：[[EX2_速度ストレス実験_結果と考察]]。\n\n'+''.join(f'![[EX2_v5_図表/{name}.png]]\n\n' for name in figure_names)
 if marker in s:s=s[:s.index(marker)].rstrip()+'\n\n'+body
 else:s=s.rstrip()+'\n\n'+body
 atomic_note(note,s);print(text)

if __name__=='__main__':main()
