# esa向け：v3実験3・4の考察案（最終有効cohort）

大規模80候補のうち78件が入力条件を満たした。Sparse basic1,200条件は成功565、正式RSS停止27、正式timeout3、warmup_failed590、未取得15だった。warmup_failedにはサイズ除外15と、115件の実warmup停止による575件の未起動が含まれる。公式SNAP6件を補完し、Gowalla・DBLP・Amazonの45正式条件が成功した一方、YouTube・LiveJournal・Pokecの45正式条件は9件のwarmup RSS停止で未起動となった。これらを元のunavailable90件と二重計上しない。

同じ入力・初期配置・前処理・評価定義を満たす同seed成功組で比較すると、Atomicは39 graph/187組、RRは37 graph/181組が対応した。各graph内の対応比中央値を等重みで集計したSGD速度比中央値はAtomic0.944倍、RR25.116倍、stress比中央値は1.001145、1.006002だった。RRで速度改善とstress+5%以内を両立したのは21/37 graph、+1%以内は19/37 graph。全5seedが揃った35 graphではそれぞれ20/35、18/35で、成功部分集合での結果である。

RRのSGD25.116倍を処理全体の高速化とは呼ばない。同じ対応集合のcold algorithm比中央値は2.192倍、入力・stress評価・座標保存も含む全プロセスwall比は1.743倍だった。追加3件でもRRはDBLP29.70倍、Amazon22.02倍、Gowalla1.48倍と速いが、stressはそれぞれ+16.50%、+11.64%、+9.63%。追加によって5%品質基準を満たす高速graph数は増えなかった。

Luxやrgg18は高速化と近い品質を両立したが、coAuthorsCiteseerやsmallworldではRRのstressが約19%悪化し、kron17/18とweb-StanfordではRR自体がCPUより遅かった。RRのround・dispatchが多い例は時間増加と整合する一方、入力最大次数17・CV0.134のsmallworldでも品質が悪化するため、次数分布だけでは説明できない。頂点別の予定更新・lock対象数は補助指標で、実store、衝突率、retry、待ち時間の実測とは区別する。

出典と限界：core3210＋補完90原記録を保持した有効3210条件。追加6件はSNAP公式原配布から再構成し、SuiteSparse MTXとの辺・行番号の直接照合は未実施。CPU f64 / GPU f32、更新順・微小距離処理・実装経路も異なるため、品質差をlock又は更新順だけの原因へ帰属しない。小20補助は別集計で、Atomic/RRのSGD比中央値0.511/0.804倍、300条件全成功だった。
