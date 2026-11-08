from pathlib import Path
import json,re
base=Path('/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization/experiment_results/full16');a=json.loads((base/'tables/audit.json').read_text());rr={r['graph']:r for r in a['full_graphs']};arr={r['graph']:r for r in a['full_arrivals']};ds=a['degree_summary'];assets=json.loads(Path('/tmp/full16-esa-assets.json').read_text());s=Path('/tmp/esa9028-before-full16.md').read_text()
# Preserve the abstract verbatim in the Markdown source.
abstract=s[s.index('## Abstract'):s.index('## 1 はじめに')]
s=s.replace('4.4節ではイタリア','4.3節ではイタリア')
s=s.replace('本研究では、この簡潔な更新を基盤として速度と品質を併せて評価する。','')
s=s.replace('Full SGDでは頂点ブロック間とタイル内の2段階でRRを構成し、独立な更新の並列実行と座標の再利用を組み合わせる。Sparse SGDでは片側更新と両側更新を分離し、それぞれの読取り・書込みの範囲に合わせて処理を割り当てる。','Full SGDでは頂点ブロック間とタイル内の2段階でRRを構成し、独立な更新の並列実行と座標の再利用を組み合わせる。内側ラウンドを区間に分割して外側のスケジュールを巡回し、同じ相手ブロックへの連続更新を抑える。Sparse SGDでは、pivot対のRRラウンドに端点が重ならない実辺の更新を加える。この両側更新のスケジュールを区間に分け、非pivotの更新先ごとにまとめた片側更新の区間と交互に実行する。各片側区間ではpivotを読取り専用とする。')
old='15反復後にstress比が1.05以下で反復が速かったのは、Sparseでは比較可能な37グラフ中36グラフだった。Fullは19グラフすべてで反復時間を短縮したが、9グラフではstress比が1.05を超えた。そこで、同じ品質目標に達するまでの時間も調べた。また、pivot数を減らした道路網で1,100万頂点を超える配置の計算時間を評価した。'
new=f"15反復後、RoundRobin Full SGDは比較可能な19グラフ中{a['faster']}グラフで反復時間を短縮し、{a['quality_within_5pct']}グラフでstress比の中央値が1.05以下となった。両条件を満たしたのは{a['both']}グラフである。RoundRobin Sparse SGDは37グラフ中36グラフで両条件を満たした。共通の品質目標への到達時間を測定し、規模の異なる優先的選択グラフでFullの分割前後を比較する。さらに、pivot数を減らした道路網で1,100万頂点を超える配置の計算時間を評価する。"
assert old in s;s=s.replace(old,new)
s=s.replace('対角の自己タイルSは別の一つのラウンドにまとめ、下三角の重複は処理しない。','対角の自己タイルSはブロックが重ならない追加ラウンドにまとめ、下三角の重複は処理しない。')
s=s.replace('図中のラウンド順は説明用で、実行時には並べ替える。','図中のラウンド順は説明用で、実行時には並べ替える。内側列を区間に分割し、全外側ラウンドを巡回してから次の区間へ進む。')
start=s.index('座標はタイル処理中に共有メモリへ保持し');end=s.index('タイルサイズは座標の再利用量と並列度',start)
s=s[:start]+r'''各反復で、自己タイルのラウンドを含む外側ラウンド順と、各外側ラウンドの内側matching順をシャッフルする。同じ外側ラウンドの同種・同サイズのタイルでは内側順を共有する。一つのタイルを最後まで処理してから別の外側ラウンドへ移ると、同じ相手ブロックへの更新が集中する。そこで、シャッフル済みの内側列を$C=\min(16,T)$個の連続区間に分割し、第0区間について全外側ラウンド、第1区間について全外側ラウンド、という順に巡回する。長さ$L$の列に対し、区間$c$は位置$\lfloor cL/C\rfloor$から$\lfloor(c+1)L/C\rfloor-1$を含む。この分割は標本抽出や更新の省略を行わず、1反復で同じ学習率の下、各頂点対を一度ずつ処理する。変わるのは外側ラウンド間の処理の交ぜ方である。16区間はwiki-Voteでの予備調査から選び、評価では全グラフに共通の値として固定する。

各タイルの各区間を、共有メモリとバリア同期を使えるスレッド群であるworkgroupに割り当てる。対象ブロックの座標を読み、区間内で共有メモリに保持しながら各matchingを並列に処理する。matchingの後は全スレッドがバリアに参加し、更新済み座標を次へ渡す。区間の最後にグローバルメモリへ書き戻してから、別の外側ラウンドがその座標を使う。$T$に満たないブロックは仮の頂点で補い、実在する両端の対だけを更新する。dispatchはqueue順に実行し、CPUは各反復の末尾でGPUの完了を待つ。

区間内の座標再利用によってグローバルメモリの読取りと書戻しを減らす。両ブロックが$T$頂点の場合、$T^2$対の両端を毎回読み書きすると読取り・書込みはそれぞれ$2T^2$回となる。$C$区間の各々で$2T$座標を読み書きする場合は、それぞれ$2CT$回となる。1頂点の2次元座標を1アクセスと数える。$T=1024,C=16$では対ごとのアクセスに比べて64分の1となる一方、分割前のタイル処理よりdispatchと座標転送の回数は増える。距離行列の参照は残るため、この比率がそのまま時間短縮率を表すわけではない。

''' .replace(chr(92)*2,chr(92))+s[end:]
# Replace the evaluation up to the retained road-network section.
start=s.index('## 4 評価実験');road=s.index('### 4.4 数百万頂点の道路網への適用',start);end=s.index('## 5 結論',road);roadtext=s[road:end].replace('### 4.4','### 4.3')
w=rr['wiki-Vote'];f=rr['fe_4elt2'];hits=sum(v['hit'] for v in arr.values());ends=sum(v['end'] for v in arr.values())
evaltext=f'''## 4 評価実験

[更新版の英語論文PDF（2026-11-05）]({assets['main_pdf']})

頂点対更新の高速化によって、同程度のstressを持つ配置を得るまでの時間を短縮できるかを評価する[11]。評価対象は、15反復後の実行時間と最終stressの逐次CPU・ロック方式GPUとの比較、共通のstress目標への到達時間、優先的選択グラフにおける分割前後の比較、および数百万頂点の道路網に対する前処理を含む実行時間である。

各実装は、統合メモリ24 GiB、macOS 15.7.2のApple M4 Pro上で個別に実行した。CPUはf64による逐次Rust実装、Atomicはnative Metal、RoundRobinの両方式はRust・wgpu・WGSL[21]によるMetalバックエンドのf32実装である。速度差にはスケジューリングに加え、精度と実行基盤の違いも含まれる。同じ系列内で入力・seed・初期配置・学習率・Sparse前処理をそろえる。今回はRoundRobin Full SGDを再測定し、同じ実験機で保存したCPU・Atomic・Sparseの測定を再利用した。したがって主比較は測定日の異なる値を含み、一時的なシステム負荷の影響を分離していない。追加の優先的選択グラフではCPUとFull GPUの両方式を同じ測定期間に実行した。

Zhengらの学習率設定[49]で$\\epsilon=0.1$、15反復とする。基本比較は5 seed、収束診断は10 seed。Fullは$T=1024$・16区間に固定し、Sparseは各相4区間の交互実行を維持する。タイルサイズ感度は今回の評価対象外である。Sparseの比較は共通の200 pivotとし、同じ近似目的関数を最適化する。FullとSparseのworkgroupはそれぞれ1024、256スレッドで構成する。

SuiteSparseに基づくデータセット[8]を用い、Fullは20グラフ、Sparseは38グラフで評価した。com-Amazon、com-DBLP、loc-Gowallaでは公式SNAP入力を共通に使う。方向・重み・数値ゼロ・自己ループ・重複辺を除去し、最大連結成分をSHAで照合する。プロセス120秒、観測RSS 8 GiBを停止基準とし、約200 ms間隔でRSSを取得する。事前試行が停止した条件では正式実行を行わず、失敗・未起動・欠測は対応比率から除外する。観測RSSは厳密な瞬間最大値やGPU専用メモリの上限ではない。

SGD時間はシャッフル・dispatch・完了待ち・再投入を含み、前処理・初期化・stress評価・保存を除く。基本比較は15反復の合計を15で割り、道路網では合計を示す。アルゴリズム時間はさらに前処理・GPU準備・転送・後処理を含み、プロセス時間は入力・評価・出力も含む。品質は式(1)のfull stressで、8,000頂点以下は厳密、それより大きい場合はpivotと独立した固定256始点による推定とする。同seedで両手法が成功したとき、CPU時間/GPU時間を速度比、GPU stress/CPU stressをstress比とし、グラフごとの中央値とseed間の最小・最大を示す。

### 4.1 CPU・Atomicとの比較

Fullは$T=1024$、Sparseは$p=200$で、平均反復時間と最終stressを比較する。

![図5 各グラフの平均反復時間]({assets['fig05_comparison']})

**図5　平均反復時間。** 15反復のSGD時間を15で割った値。記号は3手法が共通して成功したseedの中央値、バーは最小・最大である（Full 95組、Sparse 179組）。共通seedのないグラフはN/A。縦軸は対数目盛。

RoundRobin Full SGDは19グラフ中{a['faster']}グラフで高速となり、速度比の中央値は{min(v['speedup'] for v in rr.values()):.3f}〜{max(v['speedup'] for v in rr.values()):.3f}倍だった。CPUより遅い2グラフはdwt_1005と1138_busである。fe_4elt2は速度比{f['speedup']:.3f}倍・stress比{f['stress_ratio']:.3f}、wiki-Voteは速度比{w['speedup']:.3f}倍・stress比{w['stress_ratio']:.3f}だった。18グラフでstress比の中央値が1.05以下となり、16グラフで高速化とこの品質基準を両立した。品質差が残ったbcsstk17の比は{rr['bcsstk17']['stress_ratio']:.3f}である。Atomicは19グラフ中13グラフで速く、速度比0.064〜3.365倍、stress比1.001〜2.921だった。Atomicの更新順序・更新前座標の記録費用も含むため、RoundRobinとの差がロック費用のみを示すわけではない。

![図6 Full SGDの経過時間とstress]({assets['full_time_to_quality']})

**図6　Full SGDの経過時間とstress。** 各手法10 seedについて反復1〜15の保存点をseedごとに結んだ。stressは同seedのCPU最終stressで正規化し、水平線は1.05の目標を示す。

fe_4elt2とwiki-Voteではそれぞれ{arr['fe_4elt2']['hit']}/10、{arr['wiki-Vote']['hit']}/10 seedが目標へ到達した。RoundRobinの到達時間中央値は{arr['fe_4elt2']['rr_seconds']:.3f}秒と{arr['wiki-Vote']['rr_seconds']:.3f}秒で、CPUは{arr['fe_4elt2']['cpu_seconds']:.3f}秒と{arr['wiki-Vote']['cpu_seconds']:.3f}秒だった。

RoundRobin Sparse SGDは37比較可能グラフすべてで速く、速度比6.136〜55.053倍、stress比0.987〜1.067だった。36グラフで高速化とstress比1.05以下を両立した。Atomicは37グラフ中2グラフで速く、速度比0.869〜1.042倍、stress比0.971〜1.139。高速化と品質基準の両立は1グラフだった。

Fullの各手法100予定条件のうち、CPUは95完了・5メモリ停止、Atomicは95完了・事前失敗による5見送り、RoundRobinは100条件すべて完了した。Sparseは各190条件中CPU 181、Atomic 180、RoundRobin 190が完了した。CPUとAtomicはともに事前失敗による5見送りと3メモリ停止があり、さらにCPU 1件・Atomic 2件の時間超過があった。本文の比率にはCPUと比較するGPUの両者が成功した全対応seedを用い、図5では3手法共通のseedを用いる。

### 4.2 品質目標への収束

Sparseの収束確認でも基本比較と同じ200 pivotを使い、目的関数を維持する。学習率を保って反復0、1、2、3、5、8、10、15の配置を保存する。目標は同seedのCPU最終stressの1.05倍以下とし、CPUにも同じ基準を適用する。表1は最初に目標以下となった保存点を集計する。保存と途中の座標読戻しは時間から除く。

Fullの6グラフでは{hits}/60実行が目標へ到達し、15反復後には{ends}/60が目標内だった。Sparseは30実行すべてが到達し、15反復後も目標内だった。保存点による最初の観測を示しており、保存点間の正確な通過時刻は特定していない。

**次数の偏りとグラフサイズ。** NetworkXのBarabási–Albertモデルで6グラフを生成した。頂点数は2,000・4,000・8,000、新規頂点当たり5辺、各サイズのグラフseedは0・1。辺数は9,975・19,975・39,975で、最大次数は143〜311だった。各グラフで初期配置seed 0〜4をそろえ、15反復と厳密full stressで評価した。CPU・分割前・16分割を逐次計測し、各設定で事前試行を行い、初期配置seedごとに手法の実行順を交互に入れ替えた。GPU両方式のCPU完了待ちは反復末尾に統一し、以前の待機位置変更と分割の影響を分ける。

6グラフで16分割の速度比中央値は{min(v['split_speedup'] for v in ds):.2f}〜{max(v['split_speedup'] for v in ds):.2f}倍だった。分割前のstress比中央値は{min(v['unsplit_stress_ratio'] for v in ds):.3f}〜{max(v['unsplit_stress_ratio'] for v in ds):.3f}、16分割後は{min(v['split_stress_ratio'] for v in ds):.3f}〜{max(v['split_stress_ratio'] for v in ds):.3f}となった。30配置のうちCPU差5%以内だったのは分割前{sum(v['unsplit_within_target'] for v in ds)}件、16分割後{sum(v['split_within_target'] for v in ds)}件だった。分割後の反復時間は分割前の{min(v['split_unsplit_time_ratio'] for v in ds):.2f}〜{max(v['split_unsplit_time_ratio'] for v in ds):.2f}倍（グラフごとの中央値）。このグラフ族では更新を交ぜることによる時間と品質の変化を確認できるが、次数分布だけの効果や全ネットワークでの挙動を示すものではない。

![図7 優先的選択グラフの速度とstress]({assets['full_degree_scaling']})

**図7　優先的選択グラフでのFull SGD。** 各頂点数で2グラフseedを使い、点は5初期配置の中央値、バーは最小・最大。(a)対数軸の平均反復時間。(b)同seedのCPUに対する厳密最終stress比。水平線は1.05。GPU両方式の更新則・タイルサイズ・反復末尾の完了待ちは共通。

**表1　15反復以内の品質目標への到達。** 各行10対応seed。「到達」は保存点のいずれかで到達したRoundRobin実行数、「最終」は15反復後に目標内の数。時間は初回到達の累積SGD秒数の中央値で、RoundRobinは到達したseedだけを用いる。CPUは全seedで到達した。

|系列|グラフ|到達|最終|CPU (s)|RoundRobin (s)|
|---|---|---:|---:|---:|---:|
'''
for g in ['USpowerGrid','dwt_2680','bcsstk16','fe_4elt2','EVA','wiki-Vote']:
 r=arr[g];rt=f"{r['rr_seconds']:.3f}" if r['rr_seconds'] is not None else '—';evaltext+=f"|Full|{g}|{r['hit']}|{r['end']}|{r['cpu_seconds']:.3f}|{rt}|\n"
evaltext+='|Sparse|luxembourg_osm|10|10|6.722|0.149|\n|Sparse|kron_g500-logn18|10|10|2.138|0.919|\n|Sparse|coAuthorsCiteseer|10|10|13.925|0.391|\n\n'
s=s[:start]+evaltext+roadtext+s[end:]
# Preserve unchanged Sparse conclusion and hardware outlook, replace Full conclusion.
start=s.index('Fullでは19グラフすべてで反復が速くなったものの');end=s.index('各ラウンドに十分な独立頂点対があれば',start)
s=s[:start]+f'''Fullでは、内側列を16区間に分けて外側ラウンドを巡回する方式により、19グラフ中{a['faster']}グラフで反復時間を短縮し、{a['quality_within_5pct']}グラフでstress比中央値が1.05以下となった。品質診断は{hits}/60実行が15反復以内に目標へ到達した。wiki-Voteの到達時間中央値は{arr['wiki-Vote']['rr_seconds']:.3f}秒で、CPUは{arr['wiki-Vote']['cpu_seconds']:.3f}秒だった。追加の優先的選択グラフでも、分割前に比べて反復時間を要する一方で品質が改善した。小さい2グラフはCPUより遅く、bcsstk17には品質差が残る。用途に応じたstress目標への到達時間は、反復時間を補う指標として引き続き必要である。

'''+s[end:]
s=s.replace('https://img.esa.io/uploads/production/attachments/8704/2026/11/05/164555/533641ad-5ba7-471f-8e78-4fd430265e9a.png',assets['fig02_rr_full'])
assert s[s.index('## Abstract'):s.index('## 1 はじめに')]==abstract
# Editorial note is explicitly outside the manuscript and abstract.
s+='\n\n## 2026-11-05 更新記録（論文本文外）\n\n添付した英語PDFを正として本文・図・数値を更新した。Fullは16分割、Sparseは従来の4分割。タイル実験と100 seed探索は今回は外し、6個のBAグラフを追加した。アブストラクトは依頼どおり据え置きのため、そこに残る旧Full数値は更新後の本文と一致していない。実験条件・生記録・変更の詳細は[作業ログ #9051](https://vdslab.esa.io/posts/9051)を参照。\n'
Path('/tmp/esa9028-full16-final.md').write_text(s)
print('Japanese draft',len(s),'chars; abstract retained verbatim')
