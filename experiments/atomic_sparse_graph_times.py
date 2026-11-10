#!/usr/bin/env python3
"""Refresh graph wall times with matched historical Atomic and current RR references."""
import collections,csv,datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiment_results/v5.5/atomic-sparse-main-20261110'
def read(p):return json.loads(p.read_text())
def stamp(s):return datetime.datetime.fromisoformat(s)
def duration(value):
    if value is None:return '—'
    s=round(value);return f'{s//60}分{s%60:02d}秒'
def seconds(rows):return sum(r['process_wall_seconds']+r.get('archive_seconds',0) for p,r in rows)
def main():
    datasets={g['name']:g for g in read(BASE/'config/datasets.json')}
    jobs=read(BASE/'config/original-jobs.json' if (BASE/'config/original-jobs.json').exists() else BASE/'config/jobs.json');names=list(dict.fromkeys(j['graph'] for j in jobs))
    formal=collections.defaultdict(list);warm=collections.defaultdict(list);legacy=collections.defaultdict(list)
    for kind,index in [('raw',formal),('warmups',warm)]:
        for p in (BASE/kind).glob('*/record.json'):
            r=read(p)
            if r['status']=='success':index[r['graph']].append((p.parent,r))
    for kind,pattern in [('raw/comparison','*-atomic_sparse_sgd-*/record.json'),('warmups','*atomic_sparse_sgd*/record.json')]:
        for p in (ROOT/'experiment_results/v5'/kind).glob(pattern):
            r=read(p)
            if r['status']=='success' and r.get('stage')=='comparison':legacy[r['graph']].append((p,r))
    rr={};rrsources={};hashes={}
    # Later complete cohorts replace earlier cohorts for the same input.
    for source in ['v5.5/ex2-rr128-speed-stress-20261110','v5.5/ex2-copapers-capacity-supplement-20261110','v5.6/rr-sparse-lux-stanford-20261110']:
        folder=ROOT/'experiment_results'/source;groups=collections.defaultdict(list)
        for kind in ['raw','warmups']:
            for p in (folder/kind).glob('*/record.json'):
                r=read(p)
                if r['status']=='success' and r['method']=='rr_sparse_sgd':groups[r['graph']].append((p,r))
        for graph,rs in groups.items():
            measured=[r for p,r in rs if not r.get('warmup')]
            if len(measured)!=15:continue
            assert {(r['seed'],r['repetition']) for r in measured}=={(s,p) for s in range(5) for p in range(3)}
            assert all(r['input_sha256']==datasets[graph]['sha256'] for p,r in rs)
            rr[graph]=rs;rrsources[graph]=source
    assert set(rr)==set(names)
    result=[];now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    for graph in names:
        runs=formal[graph];warmups=warm[graph];folders=[p for p,r in runs+warmups]
        folders += [p.parent for kind,pattern in [('raw',f'*-{graph}-atomic_sparse_sgd-*/command.json'),('warmups',f'warm-v55-atomic-sparse-{graph}-*/command.json')] for p in (BASE/kind).glob(pattern) if p.parent not in folders]
        starts=[stamp(read(p/'command.json')['at']) for p in folders]
        complete=len(runs)==15
        # record is written before compression; require all archive manifests too.
        archives=[p/'artifact-archive.json' for p,r in runs]
        complete=complete and all(p.exists() for p in archives)
        ends=[stamp(read(p)['converted_at']) for p in archives if p.exists()]
        finish=max(ends) if complete and ends else now
        elapsed=(finish-min(starts)).total_seconds() if starts else None
        old=legacy[graph];old_formal=[r for p,r in old if not r.get('warmup')];old_warm=[r for p,r in old if r.get('warmup')]
        assert len(old_formal)==15
        rrrows=rr[graph]
        result.append(dict(graph=graph,vertices=datasets[graph]['nodes'],edges=datasets[graph]['edges'],completed_runs=len(runs),planned_runs=15,warmup_runs=len(warmups),
            state='completed' if complete else ('running' if starts else ('excluded' if graph in {'com-DBLP','web-NotreDame','com-Amazon','coPapersCiteseer'} else 'pending')),
            started_at=min(starts).isoformat(timespec='seconds') if starts else None,
            finished_at=finish.isoformat(timespec='seconds') if complete else None,
            graph_elapsed_seconds=elapsed,graph_elapsed_minutes=elapsed/60 if elapsed is not None else None,
            formal_process_seconds=sum(r['process_wall_seconds'] for p,r in runs),
            formal_archive_seconds=sum(r.get('archive_seconds',0) for p,r in runs),
            warmup_process_and_archive_seconds=seconds(warmups),
            v5_total_seconds=seconds(old),v5_formal_runs=len(old_formal),v5_warmup_runs=len(old_warm),v5_source_records=[str(p) for p,r in old],
            rr_sparse_total_seconds=seconds(rrrows),rr_sparse_formal_runs=15,
            rr_sparse_warmup_runs=sum(bool(r.get('warmup')) for p,r in rrrows),
            rr_sparse_source=rrsources[graph],rr_sparse_source_records=[str(p) for p,r in rrrows]))
        for p,r in old+rrrows:hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    target=BASE/'tables';target.mkdir(exist_ok=True)
    (target/'graph-times.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (target/'graph-times-source-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with (target/'graph-times.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]),lineterminator="\n");w.writeheader();w.writerows(result)
    lines=['# グラフ別所要時間','',
        '各グラフのwarmup開始から、本測定15件目の座標保存・可逆圧縮完了までの実経過時間。warmup・Stress評価・保存・圧縮・実行間の待ちを含む。SGD性能指標とは別。vは頂点数、eは無向辺数（正規化・最大連結成分抽出後の実験入力）。タイムスタンプの精度は1秒。途中経過は観測時点まで。未開始の合計所要時間は「—」。','',
        'v5・RR Sparse列は、成功した本測定15件＋warmup（再開時の追加warmupも含む）のprocess_wall_secondsとarchive_secondsの合計。他方式の実行・待ち時間とRR固有の描画処理は含めない。現行Atomic列はグラフ全体の実経過時間で、実行間の短い管理時間も含む。圧縮方式等も異なるため、この列からSGD性能比を計算しない。','',
        'RR SparseはW128・p=min(200,n)、15反復・epsilon0.1、seed0〜4×各3回。v5.5 EX2を基本に、coPapersCiteseerは容量拡張補測、luxembourg_osm/web-Stanfordは最新mainのv5.6正式測定を採用。同時測定した値ではない。参照元のrun一覧・SHA256はtables/graph-times.jsonおよびgraph-times-source-sha256.jsonに保存。','',
        '| グラフ | v（頂点数） | e（辺数） | 完了/予定 | 状態 | 合計所要時間 | v5 所要時間 | RR Sparse 所要時間 |',
        '|---|---:|---:|---:|---|---:|---:|---:|']
    for r in result:
        lines.append(f"| {r['graph']} | {r['vertices']:,} | {r['edges']:,} | {r['completed_runs']}/15 | {r['state']} | {duration(r['graph_elapsed_seconds'])} | {duration(r['v5_total_seconds'])} | {duration(r['rr_sparse_total_seconds'])} |")
    (BASE/'GRAPH_TIMES.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([{k:r[k] for k in ['graph','state','completed_runs','graph_elapsed_seconds','v5_total_seconds','rr_sparse_total_seconds']} for r in result if r['state']=='running' or (r['state']=='completed' and r['graph'] in names[max(0,len(formal)-3):len(formal)])],ensure_ascii=False))
if __name__=='__main__':main()
