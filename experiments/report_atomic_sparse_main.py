#!/usr/bin/env python3
"""Summarize complete new Atomic Sparse measurements against historical matched CPU."""
import collections,csv,json,math,statistics
from pathlib import Path
from run_v5 import save,now
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiment_results/v5.5/atomic-sparse-main-20261110'
def hierarchy(rows,value):
    values=[]
    for seed in range(5):
        rs=sorted([r for r in rows if r['seed']==seed],key=lambda r:r['repetition'])
        assert [r['repetition'] for r in rs]==[0,1,2]
        values.append(statistics.median(value(r) for r in rs))
    return dict(median=statistics.median(values),min=min(values),max=max(values),seed_medians=values)
def main():
    rows=[json.loads(p.read_text()) for p in sorted((BASE/'raw').glob('*/record.json'))]
    assert len(rows)==780 and all(r['status']=='success' for r in rows)
    cpu={}
    for p in (ROOT/'experiment_results/v5/raw/comparison').glob('*-sparse_sgd-*/record.json'):
        r=json.loads(p.read_text());cpu[r['graph'],r['seed'],r['repetition']]=r
    grouped=collections.defaultdict(list)
    for r in rows:
        c=cpu[r['graph'],r['seed'],r['repetition']]
        for field in ['input_sha256','initial_positions_sha256','preprocess_sha256','stress_kind','stress_samples','stress_seed']:
            assert r[field]==c[field],(r['run_id'],field)
        r['sgd_speedup']=(c['sgd_time_ms']+(c.get('readback_time_ms') or 0))/r['completed_sgd_readback_ms']
        r['total_speedup']=c['total_compute_time_ms']/r['total_compute_time_ms']
        r['stress_ratio']=r['stress_value']/c['stress_value']
        grouped[r['graph']].append(r)
    result=[]
    fields=['sgd_time_ms','readback_time_ms','completed_sgd_readback_ms','total_compute_time_ms','stress_value','sgd_speedup','total_speedup','stress_ratio','peak_rss_sampled_bytes']
    for graph,rs in grouped.items():
        assert len(rs)==15
        item=dict(graph=graph,runs=15,complete_seeds=5,nodes=rs[0]['nodes'],edges=rs[0]['edges'])
        for field in fields:item[field]=hierarchy(rs,lambda r,f=field:r[f])
        result.append(item)
    save(BASE/'tables/records.json',rows);save(BASE/'tables/hierarchical.json',result)
    flat=[{**{k:v for k,v in r.items() if not isinstance(v,dict)},**{f'{f}_{k}':v[k] for f,v in r.items() if isinstance(v,dict) for k in ['median','min','max']}} for r in result]
    with (BASE/'tables/summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator="\n");w.writeheader();w.writerows(flat)
    gm={f:math.exp(statistics.mean(math.log(r[f]['median']) for r in result)) for f in ['sgd_speedup','total_speedup','stress_ratio']}
    save(BASE/'tables/geometric-means.json',gm)
    text=f"# Atomic Sparse 最新main 速度・Stress（web-Stanfordまで）\n\n{now()}。成功780/780、52グラフ×5 seed×3回、各グラフseed999 warmup除外。15反復、epsilon0.1、p=min(200,n)。\n\ncommit: `{rows[0]['git_commit']}`。GPU既定更新順、CPU shuffle無効。GPU timestamp無効。n<=8000厳密Full Stress、超過は256始点・評価seed0の推定Full Stress。速度とStressは同一最終座標。座標は検証済み可逆圧縮。\n\n各runで対応CPU比を計算し、seed内3回中央値→5seed中央値・最小最大。グラフ間は等重み幾何平均。CPUは過去セッションの正式測定。ユーザー指定でweb-Stanfordまでの52グラフで終了し、com-DBLP・web-NotreDame・com-Amazon・coPapersCiteseerは今回未測定。温度・電源・常駐アプリは統制していない。\n\nSGD+readback速度比: {gm['sgd_speedup']:.6f}、総計算速度比: {gm['total_speedup']:.6f}、Stress比: {gm['stress_ratio']:.6f}。\n\nSGD単独のnative値も保存するが、GPU完了待ちを含む主指標はsgd_time_ms+readback_time_ms。総計算はCPU最終座標完成まで、Stress評価・保存・描画は含まない。比較対象は過去CPUであり同時測定ではない。\n"
    (BASE/'REPORT.md').write_text(text)
    print(json.dumps(gm),flush=True)
if __name__=='__main__':main()
