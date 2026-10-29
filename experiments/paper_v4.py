#!/usr/bin/env python3
"""Prepare a reviewable local manuscript; publish separately after figure inspection."""
from pathlib import Path
import csv,json,statistics,re
BASE=Path(__file__).resolve().parents[1]/'experiment_results/v4'
def read(n):return list(csv.DictReader((BASE/'tables'/n).open()))
def main():
 audit=json.loads((BASE/'tables/audit.json').read_text());assert audit['completed']==audit['planned'] and audit['convergence_evaluated']
 summary=read('summary.csv');runs=read('runs.csv');arrivals=read('arrivals.csv');pairs=read('paired-ratios.csv')
 rr=[r for r in summary if r['method']=='rr_sparse_sgd' and int(r['paired_seeds'])];at=[r for r in summary if r['method']=='atomic_sparse_sgd' and int(r['paired_seeds'])]
 text=(BASE/'reference/paper-before-v4/論文_日本語草稿.md').read_text()
 def replace(prefix,new):
  nonlocal text
  paragraphs=text.split('\n\n');matches=[i for i,p in enumerate(paragraphs) if p.startswith(prefix)];assert len(matches)==1,(prefix,matches);paragraphs[matches[0]]=new;text='\n\n'.join(paragraphs)
 def stats(rows):
  ss=[float(r['speedup_median']) for r in rows];qq=[float(r['stress_ratio_median']) for r in rows]
  return f'比較可能{len(rows)}グラフ中{sum(s>1 for s in ss)}グラフでCPUより高速だった。速度比中央値の範囲は{min(ss):.3f}〜{max(ss):.3f}倍、stress比中央値は{min(qq):.3f}〜{max(qq):.3f}で、{sum(s>1 and q<=1.05 for s,q in zip(ss,qq))}グラフが高速化とstress比1.05以下を両立した。'
 nfast=sum(float(r['speedup_median'])>1 for r in rr)
 replace('本研究では、グラフ描画におけるSGD',f'本研究では、グラフ描画におけるSGDのストレス最小化を高速化するため、ミニバッチSGDに着想を得た二つのGPU実行方式を提案する。Round-robin（RR）によって独立な更新をまとめ、Full SGDではタイル内の座標を再利用し、Sparse SGDでは片側更新と両側更新を分離する。Apple M4 Pro上で、Full RRのv3測定は比較可能19グラフすべて、Sparse RRのv4測定は比較可能{len(rr)}グラフ中{nfast}グラフでCPUより短い反復時間を得た。GPUは単一f32、CPUはf64とし、15反復で比較した。速度、配置品質、資源制限下の完了率を併せて評価する。')
 text=text.replace('評価には実装間の数値精度や順序の違いがあり、特にv3のSparse実装と読取り専用設計との差を明示する。','評価では実装間の数値精度と更新順序の違いを明示し、FullとCPU・Atomic Sparseの既存v3結果と、RR Sparseの再測定v4結果を区別する。')
 text=text.replace('実験版ではFullに明示シャッフルはなく、Sparseは毎反復ペアIDをシャッフルする。','比較に使うFull Atomic v3には明示shuffleがなく、Sparse Atomic v3は毎反復CPUでペアIDをshuffleする。Sparse Atomicは既存v3の参照値であり、現行実装の再測定値ではない。')
 text=text.replace('意図する手順では片側更新中にpivotへ書き込まない。','v4実装はpivot頂点を片側更新の対象から除き、この相ではpivotへ書き込まない。')
 text=text.replace('関連するpivot項を順に適用して最後に書き戻す。','その区間に割り当てたpivot項を順に適用し、区間の最後に書き戻す。')
 replace('**図3：**','**図3：** (a)更新対象を分割する。(b)片側相では各非pivot頂点を一つのスレッドが担当し、pivotを読取り専用で参照して局所座標を更新する。点線矢印は座標の読取りを表す。(c)両側相では端点を共有しないmatchingを順に処理する。v4では各相を4分割して交互に実行し、1反復で全対象を一巡する。')
 replace('片側更新の段階を片側相','片側更新の段階を片側相、両側更新の段階を両側相と呼び、前の段階の更新が完了してから次へ進む。v4では各反復で共通のpivot順列と両側ラウンド順をシャッフルし、頂点ID・反復・seedに基づく置換で頂点ごとのpivot処理順を構成する。片側のpivot走査と両側のラウンド列をそれぞれ4分割し、片側区間と両側区間を交互に実行する。各反復でどちらの相から始めるかを選び、全項を1回ずつ処理する。更新式、重み、学習率、移動率の上限は変更しない。ただし全項を任意にシャッフルするCPUの順序とは異なり、同じ最終配置を保証しない。')
 replace('設計とv3測定版には','旧v3で片側相に残っていたpivot座標の同値書戻しは、v4では除かれている。GPUの更新計算には通常の単一f32を用い、丸め誤差の補償や疑似倍精度は用いない。グラフ構造に応じた方式の切替えや、CPUより反復数を増やす調整も行わない。')
 replace('実験では、タイルサイズ','実験では、タイルサイズ、CPU・Atomicとの比較、大規模グラフへの適用の三つを評価する。Fullは保存済みのv3結果を継承し、SparseはRRのみ現行実装によるv4再測定を用い、CPU・Atomicは既存v3の参照値を用いる。新旧の出所を図・表で明示する。CPU・Atomicは2026年10月24〜25日、RR v4は10月30日の測定であり、同一セッションの同時計測ではない。時間比には実行時の負荷などの違いが影響し得る。')
 replace('データはSuiteSparse','データはSuiteSparse Matrix Collection[6]の収録グラフを基準とする。Full v3は小中規模20グラフを対象とした。RR Sparse v4は以前の論文図5に掲載した37グラフと図7のdelaunay_n19の計38グラフを対象とし、収束診断はluxembourg_osm、kron_g500-logn18、coAuthorsCiteseerの3グラフで行った。これは旧v3の大規模78候補すべての再測定ではない。入力はv3と同じ前処理済み最大連結成分をSHAで照合した。com-Amazon、com-DBLP、loc-Gowallaはv3の復旧コホートと同じ公式SNAP原配布を用いる。SuiteSparse配布ファイルとのバイト・辺単位の同一性は主張しない。方向・重み・数値ゼロ・自己ループ・重複辺を除去した入力で、成功条件数と欠測を併記する。')
 replace('**図5：**',f'**図5：** CPUとRRの1反復あたりの平均時間。各実行の15反復合計を15で割り、同じ成功seed集合の中央値を点、最小〜最大を縦の誤差棒で示す。縦軸はmsの対数目盛で、白抜き四角がCPU、黒丸がRRである。Fullは既存v3の19グラフ・各5対応seed。SparseはCPU v3とRR v4の38対象中{len(rr)}グラフ・計{sum(int(r["paired_seeds"]) for r in rr)}対応seedで、比較できない対象はN/Aとする。Sparseを2段に分け、共通の縦軸範囲を用いる。個々の反復時間の最小・最大や信頼区間ではない。')
 replace('大規模Sparse RRは','Sparse RR（v4）は'+stats(rr))
 replace('Sparse Atomicの速度改善は','Sparse Atomic（既存v3参照）は'+stats(at)+' 小中規模Sparseの旧v3補助比較は今回再測定しておらず、このv4結果から小規模の性能を結論づけない。')
 arr=[r for r in arrivals if r['method']=='rr_sparse_sgd'];reached=sum(r['reached']=='True' for r in arr);final=sum(r['final_within_target']=='True' for r in arr)
 replace('固定反復後の比較を補うため',f'固定反復後の比較を補うため、反復0、1、2、3、5、8、10、15の配置からstressの推移を求めた（図6）。15反復用の学習率を保ち、同seedのCPU最終stressの1.05倍以下を目標とした。Full RRの既存v3診断では評価可能60実行中36実行で到達を観測した。今回のSparse RRの3グラフではCPU v3との対応成功{len(arr)}実行中{reached}実行で到達を観測し、最終時点でも目標以下だったのは{final}実行だった。')
 replace('**図6：**','**図6：** 各保存反復の手法別stress中央値を、対応するCPUの最終stress中央値で割って示す。各パネルの計画は10 seedで、欠測時はCPU/Atomic/RRの対応数を図中に明記する。線は単一実行の軌跡ではなく、観測点を結んだ中央値である。上段はFull v3、下段はCPU・Atomicがv3、RRがv4。下段のstressは固定256始点の推定値であり、全頂点対の厳密評価ではない。')
 desc=[]
 for g in ['luxembourg_osm','kron_g500-logn18','coAuthorsCiteseer']:
  rs=[r for r in arr if r['graph']==g];yes=[r for r in rs if r['reached']=='True'];ts=statistics.median(float(r['first_observed_ms'])/1000 for r in yes) if yes else None
  desc.append(f'{g}のRRはCPU対応{len(rs)}実行中{len(yes)}実行で到達を観測し、最終時点の目標内は{sum(r["final_within_target"]=="True" for r in rs)}実行だった。'+(f'到達観測時のSGD累積時間中央値は{ts:.3f}秒だった。' if ts is not None else ''))
 replace('品質目標までの時間にも',''.join(desc)+' 保存点間は観測していないため、厳密な初到達時刻とはしない。途中保存とGPUからの座標取得は診断の反復時間から除き、通常の基本計測とは分けて扱う。')
 text=text.replace('RR-Sparseはv3のpivot書戻しを含むため、差をスケジュール固有の性質と結論づけない。','v4はpivot書戻しを除いているが、CPUとGPUの精度と順序は異なるため、差を一つの原因だけに帰属しない。')
 vals=[statistics.median(float(r[k]) for r in rr) for k in ['speedup_median','algorithm_speedup_median','process_speedup_median']]
 replace('反復の高速化に比べ','反復速度と全体時間を分けて評価する。同じ成功集合で求めたグラフ別速度比をグラフ等重みで中央値にすると、Full RRの既存v3はSGD、前処理などを含むアルゴリズム、プロセス全体で7.384、3.769、2.597倍だった。'+f'Sparse RR v4はそれぞれ{vals[0]:.3f}、{vals[1]:.3f}、{vals[2]:.3f}倍だった。描画を得るまでの待ち時間には反復以外も含まれる。')
 replace('大規模実験では、資源監視','大規模実験では、資源監視条件下での完了率とRR配置を示す。Full delaunay_n14はv3、Sparse delaunay_n19はCPU・Atomicがv3参照、RRがv4の結果である。CPU結果がない場合は速度比・品質比を推定せず、CPU結果がある場合は対応seedによる比較を用いる。GPUだけが原理的に処理できる規模という意味ではない。')
 def completed(m):return [r for r in runs if r['study']=='basic' and r['graph']=='delaunay_n19' and r['method']==m and r['status']=='success']
 cpu,atomic,gpu=[completed(m) for m in ['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd']]
 def span(rs,k,unit=1):
  vv=[float(r[k])/unit for r in rs];return f'{min(vv):.3f}〜{max(vv):.3f}' if vv else '—'
 text=re.sub(r'^\| delaunay_n19・Sparse .*$',f'| delaunay_n19・Sparse v4 | 524,288 | {len(cpu)}/5 | {len(atomic)}/5 | {len(gpu)}/5 | {span(gpu,"iteration_time_ms",1000)}秒 | {span(gpu,"process_wall_seconds")}秒 |',text,flags=re.M)
 fail=[r for r in runs if r['study']=='basic' and r['graph']=='delaunay_n19' and r['status']!='success']
 counts={m:{status:sum(r['method']==m and r['status']==status for r in fail) for status in sorted({r['status'] for r in fail if r['method']==m})} for m in ['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd']}
 labels={'sparse_sgd':'CPU v3','atomic_sparse_sgd':'Atomic v3','rr_sparse_sgd':'RR v4'};states={'memory_limit':'RSS上限停止','timeout':'時間上限停止','warmup_failed':'事前試行失敗による未起動','error':'実行エラー','disk_limit':'ディスク空き上限停止'}
 failures_text='、'.join(f'{labels[m]}：'+ '・'.join(f'{states.get(k,k)}{n}件' for k,n in cc.items()) for m,cc in counts.items() if cc) or 'なし'
 replace('**表1：**','**表1：** 資源監視条件下の完了数と時間。Full delaunay_n14のCPUは正式5実行がRSS監視で停止し、Atomicは事前試行がRSSで停止した。Full RRの残り1実行はRSSで停止した。Sparseの非成功状態は'+failures_text+'。Sparse CPU・Atomicの事前試行停止理由は既存v3のRSS上限である。全体時間は入力・前処理・評価・保存を含み、終了後の圧縮は含めない。')
 replace('Sparse RRは524,288',f'Sparse RR v4は524,288頂点・1,572,823辺のdelaunay_n19で{len(gpu)}/5 seedが完了した。SGDの15反復合計は{span(gpu,"iteration_time_ms",1000)}秒、全体時間は{span(gpu,"process_wall_seconds")}秒だった。CPUは{len(cpu)}/5、Atomicは{len(atomic)}/5が完了した。'+('CPU対応成功seedによる速度・stress比較は図5と基本比較に含める。' if cpu else 'CPUの正式成功がないため、速度・品質の優劣は評価しない。')+'stressの絶対値を異なるグラフ間の品質比較には用いない。')
 text=text.replace('![図7 CPUの対応結果がない大規模グラフの配置]','![図7 資源監視条件下の大規模グラフ配置]')
 replace('**図7：**',f'**図7：** 完了して保存されたRR配置を全て示す。上段はFull v3のseed 1〜4、下段はSparse v4のseed {", ".join(r["seed"] for r in gpu) if gpu else "なし"}である。良好なGPU seedの選抜は行わない。頂点だけを描き、同じグラフ内で共通の座標尺度を使う。見出しの秒数はSGDの15反復合計で、全体時間は表1を参照する。')
 text=text.replace('補助実験では1,048,576頂点','旧v3の補助実験では1,048,576頂点').replace('規模拡大にはGPUだけでなく近似の強さとデータ構造も関わる。','このpivot変更実験は今回のv4範囲には含まれず、現行実装の100万頂点での結果とはみなさない。')
 replace('以上は、今回の条件でCPU','これらは今回の資源監視条件下での観測であり、普遍的な容量優位や安定した完了を保証しない。単一機種、RSS監視の粒度、固定反復、推定stressという条件の下で、全工程の時間、最大メモリ、完了率を併せて解釈する。')
 replace('v3ではFull RRが',f'Full RRのv3測定では比較可能19グラフすべて、Sparse RRのv4測定では比較可能{len(rr)}グラフ中{nfast}グラフでCPUより速かった。速度と最終stressは別々に評価する必要があり、同じ更新式でも順序と数値精度が異なれば同じ最終配置は保証されない。タイルサイズによる品質の変化も残り、有効性は速度・品質・完了率を併せて判断する。')
 replace('今後は、RR-Sparseのpivot','今後は、精度と更新順の影響を分離し、同一品質までの全体時間、前処理とメモリの改善、異なるGPUでの再現性を検証する。Sparse v4では通常のf32とCPUと同じ反復数で比較したが、固定256始点によるstress推定値であり、微小な優劣や全頂点対での順位を保証するものではない。')
 out=BASE/'report/論文_日本語草稿_v4.md';out.parent.mkdir(exist_ok=True);out.write_text(text)
 from merge_latest_paper_v4 import merge
 merge(BASE)
 print(out)
if __name__=='__main__':main()
