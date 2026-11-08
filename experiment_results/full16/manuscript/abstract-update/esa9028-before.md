**論文タイトル**　Round-Robin SGD: Accelerating Stress Minimization for Graph Drawing on GPUs




**著者**　Kohsei Mochizuki、Yosuke Onoue（日本大学）




![図1 ベルギー・ドイツ・イタリアの道路網](https://img.esa.io/uploads/production/attachments/8704/2026/11/05/164555/712a8237-0a52-4dbd-9301-260e07df9e2f.png)




**図1　道路網の配置。** RoundRobin Sparse SGDによる (a) ベルギー、(b) ドイツ、(c) イタリアの道路網。最大のドイツは11,548,845頂点、12,369,181無向辺を持つ。色は固定した始点からのホップ距離を正規化した値で、頂点の位置は計算で得た配置を表す。いずれも15反復、seed 0で、pivot数は順に200、40、50。大きい2グラフでは距離表のメモリを抑えるためpivot数を減らした。4.3節ではイタリアとドイツをともに40 pivotで評価する。




## Abstract




本研究では、確率的勾配降下法（SGD）によるストレス最小化を用いたグラフ描画のために、頂点対の更新をGPU上で並列実行する方式を提案する。ミニバッチSGDが複数の目的関数項をまとめて並列に処理する考え方に着想を得て、各頂点対の更新則を保ちながら、同時に実行できる更新をまとめる。RoundRobin Full SGDは、端点を共有しない頂点対を同時に更新し、タイル内で座標を再利用する。RoundRobin Sparse SGDは、片側更新を更新先の頂点ごとにまとめ、両側更新は端点が重ならない組に分ける。Apple M4 Pro上で初期配置と反復数をそろえ、逐次CPU実装およびロックを用いるGPU実装と、反復時間と15反復後のストレスを比較した。両方式とも、比較可能なすべてのグラフで逐次CPU実装より反復時間が短かった。一方、最終ストレスの対CPU比を同じ初期配置ごとに求めると、Fullでは19グラフ中9グラフでその中央値が1.05を超え、Sparseでは37グラフ中36グラフで1.05以下だった。Sparseのグラフ別速度向上率の中央値は、反復部分で44.49倍、前処理などを含むアルゴリズム全体で3.77倍だった。さらに、40個のピボットを用いたRoundRobin Sparse SGDは、約1,155万頂点の道路網の配置を80.32秒で計算した。このうち15反復に要した時間は4.48秒だった。




**Keywords:** Graph drawing, stress minimization, stochastic gradient descent, GPU parallelization, round-robin scheduling




**Index Terms:** Human-centered computing—Visualization—Visualization techniques—Graph drawings




## 1 はじめに








確率的勾配降下法（SGD）は、グラフ描画におけるストレス最小化を効率よく行うための有力な手法である。グラフ描画では、ネットワークの構造を読み取れるように頂点を配置することが重要であり、グラフ上の距離を描画空間で再現するストレス最小化が用いられる[20, 31]。Zhengら[49]は頂点対を一組ずつ更新するSGDを提案し、実験でstress majorizationに比べて低いストレスへ速く、安定して到達できることを報告した。一方、追試では最終stressの優位が常に成立するわけではないことも示されている[3]。




SGDは、全頂点対を扱う描画から、更新対象を削減した大規模グラフの描画まで適用できる。頂点座標を$X$、頂点間の最短経路距離を$d_{ij}$、重みを$w_{ij}=d_{ij}^{-2}$とすると、stressは次式で定義される。




$$
S(X)=\sum_{i<j}w_{ij}\bigl(\lVert x_i-x_j\rVert-d_{ij}\bigr)^2.\tag{1}
$$




本論文では、全頂点対を更新する方式をFull SGDと呼ぶ。Sparse SGDは、実辺と少数の代表頂点（pivot）に関する項で近似するSparse StressをSGDで最適化する方式であり、より大規模なグラフを扱える[36, 49]。




更新対象を減らしても、残った頂点対を逐次処理する実装には並列化の余地がある。Fullでは1反復に$n(n-1)/2$対を扱い、Sparseでも多数の更新を繰り返す。本研究のCPU比較実装はこれらを逐次実行する。複数の項を同時に処理できれば時間短縮が期待できるが、同じ頂点を共有する対の読取りと書込みが競合する。Atomic操作を用いたロックは一つの解決策となるものの、取得の再試行と座標アクセスの費用が生じる。




本研究では、round-robin（RR）スケジューリングを用いてFull SGDとSparse SGDをGPU上で高速化する手法を提案する。RRは、総当たりの組合せを、同じ対象が一度だけ現れるラウンドへ分ける方式である。Full SGDでは頂点ブロック間とタイル内の2段階でRRを構成し、独立な更新の並列実行と座標の再利用を組み合わせる。内側ラウンドを区間に分割して外側のスケジュールを巡回し、同じ相手ブロックへの連続更新を抑える。Sparse SGDでは、pivot対のRRラウンドに端点が重ならない実辺の更新を加える。この両側更新のスケジュールを区間に分け、非pivotの更新先ごとにまとめた片側更新の区間と交互に実行する。各片側区間ではpivotを読取り専用とする。競合を避けるスケジュール自体は既存の並列SGDでも用いられており[30, 37, 51]、本研究はグラフ描画の二つの更新構造に合わせた実行方式を検討する。




並列化による時間短縮が、同じ配置品質への到達時間の短縮を意味するとは限らない。RRによる更新順序の変更や実装間の数値精度の違いは、最終配置にも影響し得る。そこで、対応するCPU実装およびAtomic実装と比較し、反復時間、一定反復後のstress、品質目標への到達状況を評価する。また、提案方式をwgpuとWGSLで実装する。




15反復後、RoundRobin Full SGDは比較可能な19グラフ中17グラフで反復時間を短縮し、18グラフでstress比の中央値が1.05以下となった。両条件を満たしたのは16グラフである。RoundRobin Sparse SGDは37グラフ中36グラフで両条件を満たした。共通の品質目標への到達時間を測定し、規模の異なる優先的選択グラフでFullの分割前後を比較する。さらに、pivot数を減らした道路網で1,100万頂点を超える配置の計算時間を評価する。








## 2 関連研究




### 2.1 ストレス最小化とSGDによるグラフ描画




力に基づく描画手法は、頂点間の幾何的な相互作用でグラフ構造を表す。Eadesのばねモデル[15]、Fruchterman–Reingold法[17]、ForceAtlas2[29]がその例である。Kamada–Kawai法[31]はグラフの最短経路長から理想距離を求める。距離の再現という観点では、古典的多次元尺度構成法（MDS）[22, 43]、非計量的尺度構成法[34]、非線形の距離保存写像[42]とも関係する。stress majorization[9, 20]は距離に関する目的関数の補助関数を反復的に最適化する。




確率近似[40]は反復最適化法の広い系統を支える。グラフ描画ではZhengら[49]が頂点対ごとの簡潔な更新則と学習率の設定を示した。Börsigら[3]はその実用上の頑健性を確認し、最終stressでmajorizationを上回るかは条件によることを示した。本研究はこの更新則の並列実行を評価する。




描画の目的や制約を変える方法もある。焼きなまし法は複数の描画基準を組み合わせる[7]。制約付き描画では、分離制約、距離制約、重なり除去などを扱う[12, 13, 14]。近年の例として、多基準の(SGD)²[1]とUNICON[48]がある。本研究では既存の更新を実行する順序と並列化の方法を扱う。




### 2.2 大規模グラフの描画




多段階法は粗いグラフを構築し、その配置を段階的に精緻化することで計算を速める[18, 26, 45]。階層的な力の近似[2]やポテンシャル場を用いる方法[25]は遠方の相互作用を求める費用を抑える。Hu[28]は多段階処理と力の近似を組み合わせた。これらの方法は、すべての頂点対を陽に処理する費用を避ける。




距離に基づく手法には高次元埋め込み[27]、PivotMDS[6]、低ランクstress majorization[32]、Maxent-Stress[19]がある。Sparse Stress[36]は実辺とpivotの代表領域で重み付けした項に目的関数を縮約し、Zhengら[49]はこの近似をSGDで最適化した。本研究のSparse方式は同じ近似を採用し、項数削減と、残った項の並列化を区別する。




次元削減を使うtsNET[33, 44]はt-SNEをグラフ描画に適用する。DRGraph[50]は疎な距離、負例サンプリング、多段階処理を組み合わせる。CoRe-GD[23]はグラフニューラルネットワークと粗視化の階層を使い、stressを考慮した配置を学習する。本研究では既存のstress更新を直接実行する。公開グラフ集[8, 41]と近年の評価指針[11]は、異なるグラフ群で実行時間と品質を再現可能な形で評価する基盤となる。




### 2.3 GPU実行と更新の競合




GPUによる描画には、多段階のグラフ描画[16]、GPU向けのt-SNE最適化[38]、不規則なメモリアクセスを考慮したパンゲノム描画[35]がある。GPU上のstress majorizationでは、制約付き描画を共役勾配法で解く[46]。頂点対SGDでは、一つの更新が両端を読み、その両方を動かし得る。SGDの最適化には学習率や標本順序が影響し[4, 5]、ミニバッチは並列に計算できる項をまとめる[10]。グラフ描画では、共有座標を更新する際の調整も必要になる。




Hogwild!は疎性の仮定の下で非同期の競合を許し[39]、CYCLADESは独立した更新を構成する[37]。行列分解の並列化では局所性と負荷分散も重視される[51]。Kaleemら[30]はGPU上でロックとマッチングを比較し、競合を解消しても高速化するとは限らないことを示した。そこで本研究では、独立した更新とタイル内の座標再利用を組み合わせる。




Wangら[47]はグラフ描画のSGDで更新先が共通する対を同じスレッドに割り当て、Sparse Stressにも適用した。本研究のSparseの片側更新も更新先でまとめる。この相では、各スレッドが一つの非pivot頂点に書き込み、終了までpivotは読取り専用とする。実辺とpivot対では更新前の座標に基づいて両端を動かすため、端点が重ならないマッチングに分け、ラウンド間で同期する。Fullでは連続するマッチング間で両ブロックの座標を共有メモリに保持する。本研究では逐次SGDとロック方式を比較対象とした。Wangらの方式との直接比較は今後の課題である。RRによる組合せ構成とランダムな再シャッフルでは更新順序が異なり、後者の収束保証[24]から、本方式での非凸なstressの挙動を直接導くことはできない。




## 3 ストレス最小化のためのラウンドロビンGPUスケジューリング








本研究では、Full SGDとSparse SGDをGPU上で高速化するRoundRobin Full SGDとRoundRobin Sparse SGDを提案する。以下では、SGDとAtomic方式、ミニバッチSGD、Full SGD、Sparse SGDの順に説明する。




### 3.1 SGDとAtomic方式




SGDは、グラフ上の距離と配置上の距離のずれを、頂点対ごとの更新によって減らす。連結な無向・重みなしグラフ$G=(V,E)$について、頂点数を$n$、辺数を$m$、頂点$i$の2次元座標を$x_i$、最短経路距離を$d_{ij}$とする。Zhengら[49]のSGDは頂点対を逐次更新し、本論文では全頂点対を対象とする方式をFull SGDと呼ぶ。更新対象を一巡する処理を1反復と呼び、距離計算や項の生成は前処理として区別する。




頂点対の移動量は、Zhengら[49]のSGDに従い、現在の距離と目標距離の差から求める。$x_i\ne x_j$のとき、更新前の座標から次のベクトルを計算する。




$$
r_{ij}=\frac{\lVert x_i-x_j\rVert-d_{ij}}{2}\frac{x_i-x_j}{\lVert x_i-x_j\rVert}.\tag{2}
$$




両端に対応する重みを$w_{ij}^{(i)},w_{ij}^{(j)}$、反復$t$の学習率を$\eta_t$とすると、更新は次式で表される。




$$
\begin{aligned}
\mu_i&=\min\{1,\eta_t w_{ij}^{(i)}\},&x_i&\leftarrow x_i-\mu_i r_{ij},\\
\mu_j&=\min\{1,\eta_t w_{ij}^{(j)}\},&x_j&\leftarrow x_j+\mu_j r_{ij}.
\end{aligned}\tag{3}
$$




現在距離が目標より長ければ両端を近づけ、短ければ遠ざける。Full SGDでは両端に重み$d_{ij}^{-2}$を用い、同じ更新前の$r_{ij}$から両端を更新する。距離が極めて小さい場合は実装ごとの数値処理を行う。




Sparse SGDは、実辺と少数のpivotを介して遠方の距離を表す項からなるSparse Stressを、SGDで最適化する既存手法である[36, 49]。pivot集合を$P$、数を$p$とし、各頂点を最も近いpivotの領域に割り当て、領域の情報から重みを求める。全頂点対を扱うFull SGDに対し、項数を$O(m+np)$に抑え、全対距離の保持を避ける。




Sparse SGDでは式(3)に端点別の重みを用い、重み0の端点は固定する。非pivot・pivotの非辺対は非pivot側のみを動かし、実辺とpivot同士は両端を動かす。前者を片側更新、後者を両側更新と呼ぶ。両側更新では同じ更新前の$r_{ij}$を両端に用いる。実辺の項を優先して重複を除く。




CPU版は更新対象を毎反復シャッフルし、直前の更新を反映した座標を次の対に用いる。$(i,j)$の後に$(j,k)$を処理すれば、後者は更新済みの$x_j$を読む。両者をそのまま並列実行すると、書込みの競合や、両端の座標を一貫した状態で読めない問題が生じる。座標の加算だけをatomic操作にしても、頂点対更新全体の排他性は得られない。




Atomic比較実装はMetal APIを直接用いて構築した。Atomic方式は頂点対の両端をロックし、座標の読取りから書戻しまでを排他的に実行する[30]。IDの小さい端点から取得し、両端を確保してから式(2)、(3)を適用し、書込み後に解放する。片方しか取得できなければ取得済みのロックを解放して再試行する。Atomic Fullではロック取得後と解放前にデバイススコープのメモリフェンスを置き、座標成分にもatomic操作を使う。さらに、relaxed atomicによるロック操作だけでは座標更新の可視性を保証できないため、Atomic Sparseでは両端のロック取得後と解放前にdevice scopeのメモリフェンスを配置し、前のロック保持者による更新を次の保持者へ引き継ぐ。図2の更新AとBは$j$を共有するため順に進み、独立な更新Cは並列に実行できる。




![図2 Atomic方式の処理](https://img.esa.io/uploads/production/attachments/8704/2026/11/05/164555/9e8b9aad-fdbc-4c36-a5f9-16f954e19cd5.png)




**図2：** Atomic方式では両端のロックを保持したまま座標を読み、更新し、解放する。共有頂点を持つ更新はロック取得によって順序づけられる。灰色は再試行が必要な区間を表し、長さは待ち時間の測定値を表さない。




FullとSparseのAtomic実装は、完了の確認方法と更新順が異なる。Fullはロック取得まで再試行し、GPU処理後に更新件数を確認する。Sparseは未完了項をCPU側で集めて再投入する。片側更新でも、参照中のpivotが他の更新で変わることを避けるため両端をロックする。Atomic Fullでは頂点対を明示的にシャッフルせずにGPUへ投入し、Atomic Sparseでは各反復の開始時にペアIDをシャッフルしてから投入する。どちらの実装でも、同じ頂点を共有する二つの更新のうち、両端のロックを先に取得したスレッドが先に座標を更新する。投入順が同じでも、ロック取得の試行や再試行のタイミングによって、どちらが先に更新するかは変わり得る。この仕組みは、すべての頂点対の更新順が一様ランダムな順列になることを保証しない。




### 3.2 ミニバッチSGD




ミニバッチSGDは、複数の項を一つのバッチにまとめ、それらの勾配を用いてパラメータを更新する方法である[5]。一項ずつ処理するSGDに対して、バッチ内の計算を並列化できる点がGPUの利用につながる。一般の目的関数を項$f_a(\theta)$の和または平均として表すと、選択した集合$B_t$に対する更新は次のように書ける。




$$
\theta_{t+1}=\theta_t-\frac{\alpha_t}{|B_t|}\sum_{a\in B_t}\nabla f_a(\theta_t).\tag{4}
$$




ここで$\alpha_t$は学習率である。バッチ内では同じ$\theta_t$から各項の勾配を計算し、集約して一度に更新する。このように、複数の項をまとめてGPU上で処理することで、逐次処理に含まれる並列化の機会を取り出せる。




本研究は、この「複数の項をまとめて並列処理する」という考えから、グラフ描画の頂点対更新をまとめてGPUへ割り当てる着想を得た。ただし、同じ頂点を共有する更新をそのまま同時実行すると、座標の読取り・書込みが競合する。そこで、同時実行できる更新の組を構成し、各頂点対には式(2)、(3)の更新則を適用する。式(4)の勾配平均を導入するのではなく、既存の頂点対更新を並列に進められるまとまりを作ることが、提案方式の基本方針である。




### 3.3 RoundRobin Full SGD




グラフ描画では、端点を共有しない頂点対を集めれば、勾配を集約せずに独立に更新できる。$(i,j)$が参照・変更するのは$x_i,x_j$だけなので、端点集合が交わらない$(u,v)$とは互いの入力を変えない。このような対の集合をmatchingと呼ぶ。同一matching内の対に同じ更新則を適用する限り、逐次処理と同時処理の間に数学的な依存関係は生じず、共有頂点のロックなしで並列化できる。




Round-robinは総当たりの組合せを、各対象が一度だけ現れるラウンドに分ける。4頂点なら$(1,2),(3,4)$、$(1,3),(2,4)$、$(1,4),(2,3)$の3ラウンドで全6対を一度ずつ扱える。対象数が偶数なら一つ少ない数のラウンドを用い、奇数なら仮の対象を追加してそれを含む組を除く。ラウンド内を並列化し、次のラウンドへ進む前に完了を待つ。




同一ラウンド内の独立性は、異なるラウンド順でも同じ最終配置を得ることを保証しない。RRは全対を任意の順序でシャッフルするCPU方式とは異なる順序制約を導入するため、反復時間と配置品質の両方を実験で確認する。




RoundRobin Full SGDは、ブロック間とブロック内の2段階で独立な更新を構成する。頂点をID順に最大$T$個ずつまとめ、$T$をタイルサイズとする。異なる二つのブロック間の全頂点対をタイル、同じブロック内の対を自己タイルと呼ぶ。まず同時に処理するタイルを決め、その内部で同時に更新する対を決める。




外側のRRでは、各ブロックが同じラウンド内で高々一つのタイルに属するように組合せを作る。図3(a)の上三角は異ブロックのタイルであり、同じR番号のタイルはブロックを共有せず並列に処理できる。対角の自己タイルSはブロックが重ならない追加ラウンドにまとめ、下三角の重複は処理しない。これにより全頂点対を重複なく含める。




タイル内部では、各頂点が一つの相手だけと組になるmatchingを作る。異ブロックの頂点を$0,\ldots,T-1$で番号づけ、内側ラウンド$r$で頂点$a$を他方の$(a+r)\bmod T$と組み合わせる。$r=0,\ldots,T-1$で全$T^2$対を一度ずつ扱える。図3(c)の同じ数は各行・各列に一度だけ現れる。異ブロックの表なので対角の組も有効である。自己タイルには実頂点のRRを用いて自己対と重複を除く。




![図3 RoundRobin Full SGDの2段階スケジュール](https://img.esa.io/uploads/production/attachments/8704/2026/11/05/164555/20b96721-2bbd-4c17-8cab-54e3a88355a4.png)




**図3：** (a)全頂点対をブロック単位に分割する。(b)ブロックを共有しないタイルを同じ外側ラウンドで処理する。(c)異ブロックのタイル内を巡回matchingに分ける。例は$T=4$で、行の頂点1〜4と列の頂点5〜8は別の集合である。自己タイルは別のRRで処理する。図中のラウンド順は説明用で、実行時には並べ替える。内側列を区間に分割し、全外側ラウンドを巡回してから次の区間へ進む。




各反復で、自己タイルのラウンドを含む外側ラウンド順と、各外側ラウンドの内側matching順をシャッフルする。同じ外側ラウンドの同種・同サイズのタイルでは内側順を共有する。一つのタイルを最後まで処理してから別の外側ラウンドへ移ると、同じ相手ブロックへの更新が集中する。そこで、シャッフル済みの内側列を$C=\min(16,T)$個の連続区間に分割し、第0区間について全外側ラウンド、第1区間について全外側ラウンド、という順に巡回する。長さ$L$の列に対し、区間$c$は位置$\lfloor cL/C\rfloor$から$\lfloor(c+1)L/C\rfloor-1$を含む。この分割は標本抽出や更新の省略を行わず、1反復で同じ学習率の下、各頂点対を一度ずつ処理する。変わるのは外側ラウンド間の処理の交ぜ方である。16区間はwiki-Voteでの予備調査から選び、評価では全グラフに共通の値として固定する。


各タイルの各区間を、共有メモリとバリア同期を使えるスレッド群であるworkgroupに割り当てる。対象ブロックの座標を読み、区間内で共有メモリに保持しながら各matchingを並列に処理する。matchingの後は全スレッドがバリアに参加し、更新済み座標を次へ渡す。区間の最後にグローバルメモリへ書き戻してから、別の外側ラウンドがその座標を使う。$T$に満たないブロックは仮の頂点で補い、実在する両端の対だけを更新する。dispatchはqueue順に実行し、CPUは各反復の末尾でGPUの完了を待つ。


区間内の座標再利用によってグローバルメモリの読取りと書戻しを減らす。両ブロックが$T$頂点の場合、$T^2$対の両端を毎回読み書きすると読取り・書込みはそれぞれ$2T^2$回となる。$C$区間の各々で$2T$座標を読み書きする場合は、それぞれ$2CT$回となる。1頂点の2次元座標を1アクセスと数える。$T=1024,C=16$では対ごとのアクセスに比べて64分の1となる一方、分割前のタイル処理よりdispatchと座標転送の回数は増える。距離行列の参照は残るため、この比率がそのまま時間短縮率を表すわけではない。


タイルサイズは座標の再利用量と並列度の両方に影響する。大きな$T$は外側ラウンド数を減らす一方、タイル内の更新・同期を増やし、同時実行できるタイル数を減らす。更新順も変わるため品質への影響があり得る。全対更新と距離行列の$O(n^2)$の規模は変わらず、本方式は処理のまとめ方と座標アクセスを改善する。




### 3.4 RoundRobin Sparse SGD




RoundRobin Sparse SGDは、3.1節で述べたSparse SGDの片側更新と両側更新を分離し、それぞれに応じて並列実行する。CPU・Atomic・RRは共通前処理の距離と端点別重みを用いる。




片側更新は、変更する頂点ごとにまとめて独立に処理する。図4(b)では各非pivotに一つのスレッドを割り当て、座標を局所変数へ読み、その区間に割り当てたpivot項を順に適用し、区間の最後に書き戻す。pivotを変更する項は別の段階で処理するため、RoundRobin Sparse SGDの実装はpivot頂点を片側更新の対象から除き、この相ではpivotへ書き込まない。複数スレッドによる共通pivotの参照と、担当頂点の座標再利用を両立させる。




両側更新は、pivot対のRRと実辺のmatchingによって同時書込みを避ける。pivot対のRRラウンドを作った後、残った実辺を並べ替え、両端が未使用の最初のラウンドへ追加する。追加できなければ新設する。実辺でもあるpivot対は一度だけ含める。各ラウンドはmatchingなのでロックなしで更新できるが、実辺が特定の頂点に集中すると多くのラウンドを要する。




![図4 RoundRobin Sparse SGDの更新分離](https://img.esa.io/uploads/production/attachments/8704/2026/11/05/164555/f87c1ee3-5fd3-4b49-b871-ecea845df003.png)




**図4：** (a)更新対象を分割する。(b)片側相では各非pivot頂点を一つのスレッドが担当し、pivotを読取り専用で参照して局所座標を更新する。点線矢印は座標の読取りを表す。(c)両側相では端点を共有しないmatchingを順に処理する。RoundRobin Sparse SGDでは各相を4分割して交互に実行し、1反復で全対象を一巡する。




片側更新の段階を片側相、両側更新の段階を両側相と呼び、前の段階の更新が完了してから次へ進む。RoundRobin Sparse SGDでは各反復で共通のpivot順列と両側ラウンド順をシャッフルし、頂点ID・反復・seedに基づく置換で頂点ごとのpivot処理順を構成する。片側のpivot走査と両側のラウンド列をそれぞれ4分割し、片側区間と両側区間を交互に実行する。各反復でどちらの相から始めるかを選び、全項を1回ずつ処理する。更新式、重み、学習率、移動率の上限は変更しない。ただし全項を任意にシャッフルするCPUの順序とは異なり、同じ最終配置を保証しない。








## 4 評価実験


[更新版の英語論文PDF（2026-11-05）](https://esa-storage-tokyo.s3-ap-northeast-1.amazonaws.com/uploads/production/attachments/8704/2026/11/05/164555/774a1fb8-1624-40e1-9e02-872ca1ce7fa6.pdf)


頂点対更新の高速化によって、同程度のstressを持つ配置を得るまでの時間を短縮できるかを評価する[11]。評価対象は、15反復後の実行時間と最終stressの逐次CPU・ロック方式GPUとの比較、共通のstress目標への到達時間、優先的選択グラフにおける分割前後の比較、および数百万頂点の道路網に対する前処理を含む実行時間である。


各実装は、統合メモリ24 GiB、macOS 15.7.2のApple M4 Pro上で個別に実行した。CPUはf64による逐次Rust実装、Atomicはnative Metal、RoundRobinの両方式はRust・wgpu・WGSL[21]によるMetalバックエンドのf32実装である。速度差にはスケジューリングに加え、精度と実行基盤の違いも含まれる。同じ系列内で入力・seed・初期配置・学習率・Sparse前処理をそろえる。今回はRoundRobin Full SGDを再測定し、同じ実験機で保存したCPU・Atomic・Sparseの測定を再利用した。したがって主比較は測定日の異なる値を含み、一時的なシステム負荷の影響を分離していない。追加の優先的選択グラフではCPUとFull GPUの両方式を同じ測定期間に実行した。


Zhengらの学習率設定[49]で$\epsilon=0.1$、15反復とする。基本比較は5 seed、収束診断は10 seed。Fullは$T=1024$・16区間に固定し、Sparseは各相4区間の交互実行を維持する。タイルサイズ感度は今回の評価対象外である。Sparseの比較は共通の200 pivotとし、同じ近似目的関数を最適化する。FullとSparseのworkgroupはそれぞれ1024、256スレッドで構成する。


SuiteSparseに基づくデータセット[8]を用い、Fullは20グラフ、Sparseは38グラフで評価した。com-Amazon、com-DBLP、loc-Gowallaでは公式SNAP入力を共通に使う。方向・重み・数値ゼロ・自己ループ・重複辺を除去し、最大連結成分をSHAで照合する。プロセス120秒、観測RSS 8 GiBを停止基準とし、約200 ms間隔でRSSを取得する。事前試行が停止した条件では正式実行を行わず、失敗・未起動・欠測は対応比率から除外する。観測RSSは厳密な瞬間最大値やGPU専用メモリの上限ではない。


SGD時間はシャッフル・dispatch・完了待ち・再投入を含み、前処理・初期化・stress評価・保存を除く。基本比較は15反復の合計を15で割り、道路網では合計を示す。アルゴリズム時間はさらに前処理・GPU準備・転送・後処理を含み、プロセス時間は入力・評価・出力も含む。品質は式(1)のfull stressで、8,000頂点以下は厳密、それより大きい場合はpivotと独立した固定256始点による推定とする。同seedで両手法が成功したとき、CPU時間/GPU時間を速度比、GPU stress/CPU stressをstress比とし、グラフごとの中央値とseed間の最小・最大を示す。


### 4.1 CPU・Atomicとの比較


Fullは$T=1024$、Sparseは$p=200$で、平均反復時間と最終stressを比較する。


![図5 各グラフの平均反復時間](https://img.esa.io/uploads/production/attachments/8704/2026/11/05/164555/a1789403-c58c-4b35-a333-7e6a0840f3b4.png)


**図5　平均反復時間。** 15反復のSGD時間を15で割った値。記号は3手法が共通して成功したseedの中央値、バーは最小・最大である（Full 95組、Sparse 179組）。共通seedのないグラフはN/A。縦軸は対数目盛。


RoundRobin Full SGDは19グラフ中17グラフで高速となり、速度比の中央値は0.855〜23.469倍だった。CPUより遅い2グラフはdwt_1005と1138_busである。fe_4elt2は速度比23.469倍・stress比1.010、wiki-Voteは速度比16.928倍・stress比1.026だった。18グラフでstress比の中央値が1.05以下となり、16グラフで高速化とこの品質基準を両立した。品質差が残ったbcsstk17の比は1.208である。Atomicは19グラフ中13グラフで速く、速度比0.064〜3.365倍、stress比1.001〜2.921だった。Atomicの更新順序・更新前座標の記録費用も含むため、RoundRobinとの差がロック費用のみを示すわけではない。


![図6 Full SGDの経過時間とstress](https://img.esa.io/uploads/production/attachments/8704/2026/11/05/164555/7701b65f-8fba-40ec-aedf-b498402ad52b.png)


**図6　Full SGDの経過時間とstress。** 各手法10 seedについて反復1〜15の保存点をseedごとに結んだ。stressは同seedのCPU最終stressで正規化し、水平線は1.05の目標を示す。


fe_4elt2とwiki-Voteではそれぞれ10/10、10/10 seedが目標へ到達した。RoundRobinの到達時間中央値は0.463秒と0.385秒で、CPUは11.593秒と6.691秒だった。


RoundRobin Sparse SGDは37比較可能グラフすべてで速く、速度比6.136〜55.053倍、stress比0.987〜1.067だった。36グラフで高速化とstress比1.05以下を両立した。Atomicは37グラフ中2グラフで速く、速度比0.869〜1.042倍、stress比0.971〜1.139。高速化と品質基準の両立は1グラフだった。


Fullの各手法100予定条件のうち、CPUは95完了・5メモリ停止、Atomicは95完了・事前失敗による5見送り、RoundRobinは100条件すべて完了した。Sparseは各190条件中CPU 181、Atomic 180、RoundRobin 190が完了した。CPUとAtomicはともに事前失敗による5見送りと3メモリ停止があり、さらにCPU 1件・Atomic 2件の時間超過があった。本文の比率にはCPUと比較するGPUの両者が成功した全対応seedを用い、図5では3手法共通のseedを用いる。


### 4.2 品質目標への収束


Sparseの収束確認でも基本比較と同じ200 pivotを使い、目的関数を維持する。学習率を保って反復0、1、2、3、5、8、10、15の配置を保存する。目標は同seedのCPU最終stressの1.05倍以下とし、CPUにも同じ基準を適用する。表1は最初に目標以下となった保存点を集計する。保存と途中の座標読戻しは時間から除く。


Fullの6グラフでは55/60実行が目標へ到達し、15反復後には55/60が目標内だった。未到達の5件はdwt_2680の4 seedとEVAの1 seedであり、分割による全初期配置での目標到達は保証されない。Sparseは30実行すべてが到達し、15反復後も目標内だった。保存点による最初の観測を示しており、保存点間の正確な通過時刻は特定していない。


**次数の偏りとグラフサイズ。** NetworkXのBarabási–Albertモデルで6グラフを生成した。頂点数は2,000・4,000・8,000、新規頂点当たり5辺、各サイズのグラフseedは0・1。辺数は9,975・19,975・39,975で、最大次数は143〜311だった。各グラフで初期配置seed 0〜4をそろえ、15反復と厳密full stressで評価した。CPU・分割前・16分割を逐次計測し、各設定で事前試行を行い、初期配置seedごとに手法の実行順を交互に入れ替えた。GPU両方式のCPU完了待ちは反復末尾に統一し、以前の待機位置変更と分割の影響を分ける。


6グラフで16分割の速度比中央値は2.96〜22.95倍だった。分割前のstress比中央値は1.170〜1.519、16分割後は1.007〜1.038となった。30配置のうちCPU差5%以内だったのは分割前0件、16分割後30件だった。分割後の反復時間は分割前の1.16〜1.42倍（グラフごとの中央値）。このグラフ族では更新を交ぜることによる時間と品質の変化を確認できるが、次数分布だけの効果や全ネットワークでの挙動を示すものではない。


![図7 優先的選択グラフの速度とstress](https://img.esa.io/uploads/production/attachments/8704/2026/11/05/164555/e0adf921-6edc-4cef-aca7-8c56ce810648.png)


**図7　優先的選択グラフでのFull SGD。** 各頂点数で2グラフseedを使い、点は5初期配置の中央値、バーは最小・最大。(a)対数軸の平均反復時間。(b)同seedのCPUに対する厳密最終stress比。水平線は1.05。GPU両方式の更新則・タイルサイズ・反復末尾の完了待ちは共通。


**表1　15反復以内の品質目標への到達。** 各行10対応seed。「到達」は保存点のいずれかで到達したRoundRobin実行数、「最終」は15反復後に目標内の数。時間は初回到達の累積SGD秒数の中央値で、RoundRobinは到達したseedだけを用いる。CPUは全seedで到達した。


|系列|グラフ|到達|最終|CPU (s)|RoundRobin (s)|
|---|---|---:|---:|---:|---:|
|Full|USpowerGrid|10|10|2.046|0.199|
|Full|dwt_2680|6|6|0.807|0.160|
|Full|bcsstk16|10|10|2.907|0.285|
|Full|fe_4elt2|10|10|11.593|0.463|
|Full|EVA|9|9|1.678|0.281|
|Full|wiki-Vote|10|10|6.691|0.385|
|Sparse|luxembourg_osm|10|10|6.722|0.149|
|Sparse|kron_g500-logn18|10|10|2.138|0.919|
|Sparse|coAuthorsCiteseer|10|10|13.925|0.391|


### 4.3 数百万頂点の道路網への適用




RoundRobin Sparse SGDを、italy_osmとgermany_osmに適用した。頂点数はそれぞれ6,686,493、11,548,845、無向辺数は7,013,978、12,369,181である。両実行とも同じGPU上で、40 pivots、15反復、$\epsilon=0.1$、seed 0を用い、比較実験の停止基準は適用しない。同じメモリ条件で両道路網の実行時間を比べるため、両者とも40個にそろえた。ドイツの11,548,845頂点では、各頂点から各pivotまでの距離を32ビットで保持する表だけで、40個なら1.72 GiB、200個なら8.60 GiB、1024個なら44.06 GiBを要する。1024個では距離表だけで実験機の統合メモリ24 GiBを超える。この規模でのCPU・Atomicの対応実行はない。




![図8 イタリア・ドイツの道路網](https://img.esa.io/uploads/production/attachments/8704/2026/11/05/164555/0607cb8b-b32f-45f5-8da3-1df38d45c1d3.png)




**図8　40 pivotで得た道路網の配置。** RoundRobin Sparse SGDを15反復、seed 0で実行し、イタリアとドイツの全頂点・全辺を描画した。各パネルは縦横比を保ち、独立の縮尺を用いる。色は最小IDの頂点を始点とするBFS距離を正規化した値を表す。配置の計算には地理座標を用いない。




**表2　道路網の実行時間（秒）。** 各グラフを1回ずつ測定した。SGD時間は15反復の合計、アルゴリズム時間はさらに前処理、GPU準備、転送、後処理を含む。入力読込み、stress評価、保存、画像描画は含めない。




| グラフ | SGD (s) | アルゴリズム (s) |
|---|---:|---:|
| italy_osm | 2.11 | 43.45 |
| germany_osm | 4.48 | 80.32 |




両グラフで配置計算が完了した。Germanyは15反復に4.48秒、前処理などを含むアルゴリズム全体に80.32秒を要し、単一GPUで1,000万頂点を超える配置を計算できた。確認した最大入力は処理能力の上限を示すものではなく、40 pivotsによる実行から200 pivotsと同等の品質やCPUに対する速度比は判断できない。全体時間では前処理とデータ処理が支配的となり、主要な改善対象となる。




この規模ではSGD時間とアルゴリズム時間の違いが特に大きい。SGDがアルゴリズム時間に占める割合はItalyで約4.9%、Germanyで約5.6%だった。このため、頂点対更新カーネルをさらに高速化しても、現在の座標生成時間の一部にしか効果が及ばない。共通前処理だけでItalyは37.50秒、Germanyは68.51秒を要した。両実行では、別途64始点からfull stressを推定し、それぞれ$8.93\times10^{11}$、$1.46\times10^{12}$を得た。これらは異なるグラフの絶対値であり、pivot数削減による品質変化を定量化するものではない。両配置の記録にはピークメモリ測定値が含まれていない。




pivotを増やすと、より多くの代表頂点から遠距離の関係を取り込める。メモリに収まるグラフでpivot数を変えれば、追加コストに見合うfull stressの改善があるかを検証できる。$p$を固定すると、項数$O(m+np)$は辺数と頂点数に対して線形に増える。$p$を200から40へ減らすと、実辺の項を保ちながら頂点–pivot項の規模を5分の1にできる。一方で、記憶量や計算量とともに近似目的関数も変わるため、この実験は指定条件での実行可能性を示す。2件の完了実行は1,000万頂点を超える具体的な実行条件を示しており、最大処理規模を求めるには、グラフ構造、pivot数、メモリ使用量を系統的に調べる必要がある。




## 5 結論








Sparseでは比較可能な37グラフすべてで反復時間を短縮し、36グラフで15反復後のstress比が1.05以下だった。前処理などを含む速度向上の中央値は3.77倍だった。40 pivotでは、1台のApple M4 Proで約1,155万頂点の配置が80.32秒で完了した。




Fullでは、内側列を16区間に分けて外側ラウンドを巡回する方式により、19グラフ中17グラフで反復時間を短縮し、18グラフでstress比中央値が1.05以下となった。品質診断は55/60実行が15反復以内に目標へ到達した。wiki-Voteの到達時間中央値は0.385秒で、CPUは6.691秒だった。追加の優先的選択グラフでも、分割前に比べて反復時間を要する一方で品質が改善した。小さい2グラフはCPUより遅く、bcsstk17には品質差が残る。用途に応じたstress目標への到達時間は、反復時間を補う指標として引き続き必要である。


各ラウンドに十分な独立頂点対があれば、演算ユニットの多いGPUはより多くの対を同時に処理できる可能性がある。一方で、ラウンド間の同期と前処理の費用は残る。この予測を検証するには他機種での実測と、更新先ごとに処理する方式[47]との同一GPU上での比較が必要である。




## 参考文献




[1] R. Ahmed, F. De Luca, S. Devkota, S. Kobourov, M. Li. [Multicriteria Scalable Graph Drawing via Stochastic Gradient Descent, (SGD)²](https://doi.org/10.1109/TVCG.2022.3155564). IEEE Transactions on Visualization and Computer Graphics, 28(6), 2388–2399, 2022.




[2] Josh Barnes, Piet Hut. [A hierarchical O(N log N) force-calculation algorithm](https://doi.org/10.1038/324446a0). Nature, 324(6096), 446–449, 1986.




[3] K. Börsig, U. Brandes, B. Pasztor. [Stochastic Gradient Descent Works Really Well for Stress Minimization](https://doi.org/10.1007/978-3-030-68766-3_2). Graph Drawing and Network Visualization (GD 2020), 18–25, 2020.




[4] L. Bottou. [Stochastic Gradient Descent Tricks](https://doi.org/10.1007/978-3-642-35289-8_25). Neural Networks: Tricks of the Trade, 2nd ed., 421–436, 2012.




[5] L. Bottou, F. E. Curtis, J. Nocedal. [Optimization Methods for Large-Scale Machine Learning](https://doi.org/10.1137/16M1080173). SIAM Review, 60(2), 223–311, 2018.




[6] U. Brandes, C. Pich. [Eigensolver Methods for Progressive Multidimensional Scaling of Large Data](https://doi.org/10.1007/978-3-540-70904-6_6). Graph Drawing (GD 2006), LNCS 4372, 42–53, 2007.




[7] Ron Davidson, David Harel. [Drawing graphs nicely using simulated annealing](https://doi.org/10.1145/234535.234538). ACM Transactions on Graphics, 15(4), 301–331, 1996.




[8] T. A. Davis, Y. Hu. [The University of Florida Sparse Matrix Collection](https://doi.org/10.1145/2049662.2049663). ACM Transactions on Mathematical Software, 38(1), 2011.




[9] J. de Leeuw. [Convergence of the Majorization Method for Multidimensional Scaling](https://doi.org/10.1007/BF01897162). Journal of Classification, 5(2), 163–180, 1988.




[10] O. Dekel, R. Gilad-Bachrach, O. Shamir, L. Xiao. [Optimal Distributed Online Prediction Using Mini-Batches](https://jmlr.org/papers/v13/dekel12a.html). Journal of Machine Learning Research, 13, 165–202, 2012.




[11] Sara Di Bartolomeo, Tarik Crnovrsanin, David Saffo, Eduardo Puerta, Connor Wilson, Cody Dunne. [Evaluating Graph Layout Algorithms: A Systematic Review of Methods and Best Practices](https://doi.org/10.1111/cgf.15073). Computer Graphics Forum, 43(6), e15073, 2024.




[12] Tim Dwyer. [Scalable, Versatile and Simple Constrained Graph Layout](https://doi.org/10.1111/j.1467-8659.2009.01449.x). Computer Graphics Forum, 28(3), 991–998, 2009.




[13] Tim Dwyer, Yehuda Koren, Kim Marriott. [IPSep-CoLa: An Incremental Procedure for Separation Constraint Layout of Graphs](https://doi.org/10.1109/tvcg.2006.156). IEEE Transactions on Visualization and Computer Graphics, 12(5), 821–828, 2006.




[14] Tim Dwyer, Kim Marriott, Peter J. Stuckey. [Fast Node Overlap Removal](https://doi.org/10.1007/11618058_15). Graph Drawing, 3843, 153–164, 2006.




[15] P. Eades. [A Heuristic for Graph Drawing](https://www.cs.ubc.ca/~will/536E/papers/Eades1984.pdf). Congressus Numerantium, 42, 149–160, 1984.




[16] Y. Frishman, A. Tal. [Multi-Level Graph Layout on the GPU](https://doi.org/10.1109/TVCG.2007.70580). IEEE Transactions on Visualization and Computer Graphics, 13(6), 1310–1317, 2007.




[17] T. M. J. Fruchterman, E. M. Reingold. [Graph Drawing by Force-Directed Placement](https://doi.org/10.1002/spe.4380211102). Software: Practice and Experience, 21(11), 1129–1164, 1991.




[18] Pawel Gajer, Michael T. Goodrich, Stephen G. Kobourov. [A Multi-dimensional Approach to Force-Directed Layouts of Large Graphs](https://doi.org/10.1007/3-540-44541-2_20). Graph Drawing, 1984, 211–221, 2001.




[19] E. R. Gansner, Y. Hu, S. North. [A Maxent-Stress Model for Graph Layout](https://doi.org/10.1109/TVCG.2012.299). IEEE Transactions on Visualization and Computer Graphics, 19(6), 927–940, 2013.




[20] E. R. Gansner, Y. Koren, S. North. [Graph Drawing by Stress Majorization](https://doi.org/10.1007/978-3-540-31843-9_25). Graph Drawing (GD 2004), LNCS 3383, 239–250, 2005.




[21] gfx-rs developers. [wgpu: portable graphics library for Rust](https://wgpu.rs/). Project documentation. 2026年10月30日閲覧。




[22] J. C. GOWER. [Some distance properties of latent root and vector methods used in multivariate analysis](https://doi.org/10.1093/biomet/53.3-4.325). Biometrika, 53(3-4), 325–338, 1966.




[23] F. Grötschla, J. Mathys, R. Veres, R. Wattenhofer. [CoRe-GD: A Hierarchical Framework for Scalable Graph Visualization with GNNs](https://arxiv.org/abs/2402.06706). International Conference on Learning Representations, 2024.




[24] M. Gürbüzbalaban, A. Ozdaglar, P. A. Parrilo. [Why Random Reshuffling Beats Stochastic Gradient Descent](https://doi.org/10.1007/s10107-019-01440-w). Mathematical Programming, 186, 49–84, 2021.




[25] Stefan Hachul, Michael Jünger. [Drawing Large Graphs with a Potential-Field-Based Multilevel Algorithm](https://doi.org/10.1007/978-3-540-31843-9_29). Graph Drawing, 3383, 285–295, 2005.




[26] David Harel, Yehuda Koren. [A Fast Multi-Scale Method for Drawing Large Graphs](https://doi.org/10.7155/jgaa.00051). Journal of Graph Algorithms and Applications, 6(3), 179–202, 2002.




[27] D. Harel, Y. Koren. [Graph Drawing by High-Dimensional Embedding](https://doi.org/10.7155/jgaa.00089). Journal of Graph Algorithms and Applications, 8(2), 195–214, 2004.




[28] Y. Hu. [Efficient, High-Quality Force-Directed Graph Drawing](https://yifanhu.net/PUB/graph_draw.pdf). The Mathematica Journal, 10(1), 37–71, 2005.




[29] Mathieu Jacomy, Tommaso Venturini, Sebastien Heymann, Mathieu Bastian. [ForceAtlas2, a Continuous Graph Layout Algorithm for Handy Network Visualization Designed for the Gephi Software](https://doi.org/10.1371/journal.pone.0098679). PLoS ONE, 9(6), e98679, 2014.




[30] R. Kaleem, S. Pai, K. Pingali. [Stochastic Gradient Descent on GPUs](https://doi.org/10.1145/2716282.2716289). Proceedings of the 8th Workshop on General Purpose Processing using GPUs (GPGPU-8), 81–89, 2015.




[31] T. Kamada, S. Kawai. [An Algorithm for Drawing General Undirected Graphs](https://doi.org/10.1016/0020-0190(89)90102-6). Information Processing Letters, 31(1), 7–15, 1989.




[32] M. Khoury, Y. Hu, S. Krishnan, C. Scheidegger. [Drawing Large Graphs by Low-Rank Stress Majorization](https://doi.org/10.1111/j.1467-8659.2012.03090.x). Computer Graphics Forum, 31(3pt1), 975–984, 2012.




[33] J. F. Kruiger, P. E. Rauber, R. M. Martins, A. Kerren, S. Kobourov, A. C. Telea. [Graph Layouts by t-SNE](https://doi.org/10.1111/cgf.13187). Computer Graphics Forum, 36(3), 283–294, 2017.




[34] J. B. Kruskal. [Multidimensional Scaling by Optimizing Goodness of Fit to a Nonmetric Hypothesis](https://doi.org/10.1007/BF02289565). Psychometrika, 29(1), 1–27, 1964.




[35] Jiajie Li, Jan-Niklas Schmelzle, Yixiao Du, Simon Heumos, Andrea Guarracino, Giulia Guidi, Pjotr Prins, Erik Garrison, Zhiru Zhang. [Rapid GPU-Based Pangenome Graph Layout](https://doi.org/10.1109/sc41406.2024.00035). International Conference for High Performance Computing, Networking, Storage and Analysis (SC), 1–19, 2024.




[36] M. Ortmann, M. Klimenta, U. Brandes. [A Sparse Stress Model](https://doi.org/10.7155/jgaa.00440). Journal of Graph Algorithms and Applications, 21(5), 791–821, 2017.




[37] X. Pan, M. Lam, S. Tu, D. Papailiopoulos, C. Zhang, M. I. Jordan, K. Ramchandran, C. Ré. [Cyclades: Conflict-Free Asynchronous Machine Learning](https://papers.nips.cc/paper_files/paper/2016/hash/28e209b61a52482a0ae1cb9f5959c792-Abstract.html). Advances in Neural Information Processing Systems, 29, 2016.




[38] N. Pezzotti, J. Thijssen, A. Mordvintsev, T. Höllt, B. van Lew, B. P. F. Lelieveldt, E. Eisemann, A. Vilanova. [GPGPU Linear Complexity t-SNE Optimization](https://doi.org/10.1109/TVCG.2019.2934307). IEEE Transactions on Visualization and Computer Graphics, 26(1), 1172–1181, 2020.




[39] B. Recht, C. Ré, S. Wright, F. Niu. [Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://papers.nips.cc/paper/2011/hash/218a0aefd1d1a4be65601cc6ddc1520e-Abstract.html). Advances in Neural Information Processing Systems, 24, 2011.




[40] Herbert Robbins, Sutton Monro. [A Stochastic Approximation Method](https://doi.org/10.1214/aoms/1177729586). The Annals of Mathematical Statistics, 22(3), 400–407, 1951.




[41] R. A. Rossi, N. K. Ahmed. [The Network Data Repository with Interactive Graph Analytics and Visualization](https://doi.org/10.1609/aaai.v29i1.9277). Proceedings of the AAAI Conference on Artificial Intelligence, 29(1), 4292–4293, 2015.




[42] J.W. Sammon. [A Nonlinear Mapping for Data Structure Analysis](https://doi.org/10.1109/t-c.1969.222678). IEEE Transactions on Computers, C-18(5), 401–409, 1969.




[43] Warren S. Torgerson. [Multidimensional Scaling: I. Theory and Method](https://doi.org/10.1007/bf02288916). Psychometrika, 17(4), 401–419, 1952.




[44] L. van der Maaten, G. Hinton. [Visualizing Data using t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html). Journal of Machine Learning Research, 9, 2579–2605, 2008.




[45] Chris Walshaw. [A Multilevel Algorithm for Force-Directed Graph-Drawing](https://doi.org/10.7155/jgaa.00070). Journal of Graph Algorithms and Applications, 7(3), 253–285, 2003.




[46] Y. Wang, Y. Wang, Y. Sun, L. Zhu, K. Lu, C. Fu, M. Sedlmair, O. Deussen, B. Chen. [Revisiting Stress Majorization as a Unified Framework for Interactive Constrained Graph Visualization](https://doi.org/10.1109/TVCG.2017.2745919). IEEE Transactions on Visualization and Computer Graphics, 24(1), 489–499, 2018.




[47] Z. Wang, M. Xue, Y. Wang, F. Zhong, Y. Wang. [Graph Drawing by Conflict-Free Parallel Stochastic Gradient Descent](https://doi.org/10.3724/SP.J.1089.2023-00627). Journal of Computer-Aided Design & Computer Graphics, 37(6), 1063–1072, 2025.




[48] J. Yu, Y. Hu, X. Yuan. [UNICON: A UNIform CONstraint Based Graph Layout Framework](https://doi.org/10.1109/PacificVis53943.2022.00015). Proceedings of the IEEE Pacific Visualization Symposium (PacificVis), 61–70, 2022.




[49] J. X. Zheng, S. Pawar, D. F. M. Goodman. [Graph Drawing by Stochastic Gradient Descent](https://doi.org/10.1109/TVCG.2018.2859997). IEEE Transactions on Visualization and Computer Graphics, 25(9), 2738–2748, 2019.




[50] M. Zhu, W. Chen, Y. Hu, Y. Hou, L. Liu, K. Zhang. [DRGraph: An Efficient Graph Layout Algorithm for Large-scale Graphs by Dimensionality Reduction](https://doi.org/10.1109/TVCG.2020.3030447). IEEE Transactions on Visualization and Computer Graphics, 27(2), 1666–1676, 2021.




[51] Y. Zhuang, W.-S. Chin, Y.-C. Juan, C.-J. Lin. [A Fast Parallel SGD for Matrix Factorization in Shared Memory Systems](https://doi.org/10.1145/2507157.2507164). Proceedings of the ACM Conference on Recommender Systems (RecSys), 249–256, 2013.




---




## 執筆メモ（論文本文外・旧版）




<details><summary>Introductionの流れ（段落ごとのトピックセンテンス）</summary>




| 節 | トピックセンテンス |
|---|---|
| 導入 | 確率的勾配降下法（SGD）は、グラフ描画におけるストレス最小化を効率よく行うための有力な手法である。 |
| 導入 | SGDは、全頂点対を扱う描画から、更新対象を削減した大規模グラフの描画まで適用できる。 |
| 導入 | 本論文では、全頂点対を更新する方式をFull SGDと呼ぶ。 |
| 導入 | 更新対象を減らしても、残った頂点対を逐次処理する実装には並列化の余地がある。 |
| 導入 | 本研究では、頂点対更新の依存関係に基づいて処理を分割し、Full SGDとSparse SGDをGPU上で高速化する。 |
| 導入 | 並列化による時間短縮が、同じ配置品質への到達時間の短縮を意味するとは限らない。 |
| 導入 | 本論文の貢献は以下の3点である。 |
| 導入 | 1. Full SGDとSparse SGDの高速化。 |
| 導入 | 第2章では関連研究を整理し、第3章では更新則、ミニバッチSGDとの関係、RRによる並列化、および実装を説明する。 |




</details>




<details><summary>Related Workの流れ（段落ごとのトピックセンテンス）</summary>




| 節 | トピックセンテンス |
|---|---|
| 導入 | 本研究に関係する研究を、stressとSGDによる描画、大規模グラフへの近似、GPU上の並列更新の三つに整理する。 |
| 2.1 Stress最小化とSGDによる描画 | 距離誤差を最小化する描画は、頂点間に力を設定する描画法の中でも、目的関数を明示して配置を求める方法である。 |
| 2.1 Stress最小化とSGDによる描画 | 頂点対を一組ずつ処理するSGDは、連立方程式を解くことなくstressを改善する方法である。 |
| 2.1 Stress最小化とSGDによる描画 | SGDによる描画は、単一のstress以外の目的や制約にも拡張されている。 |
| 2.2 大規模グラフに対する近似 | 大規模化には、更新項を減らす方法と、グラフ表現を段階的に変える方法がある。 |
| 2.2 大規模グラフに対する近似 | Sparse Stressは、実辺とpivotに関連する項で全体の距離関係を近似する。 |
| 2.2 大規模グラフに対する近似 | 次元削減に基づく大規模描画も、比較上重要な別の系統である。 |
| 2.3 GPU並列化と更新競合 | GPUによる描画高速化では、計算を並列実行しやすい単位に分けることが重要となる。 |
| 2.3 GPU並列化と更新競合 | 並列SGDでは、競合を許容する方式と、独立な更新を構成する方式が研究されてきた。 |
| 2.3 GPU並列化と更新競合 | グラフ描画のSGD自体をGPUで並列化した直接の先行研究として、Wangらがある。 |
| 2.3 GPU並列化と更新競合 | 更新順序の理論と、本研究のRRの効果は分けて扱う必要がある。 |




</details>




<details><summary>Round-Robin GPU Scheduling for Stress Minimizationの流れ（段落ごとのトピックセンテンス）</summary>




| 節 | トピックセンテンス |
|---|---|
| 導入 | 本研究では、Full SGDとSparse SGDをGPU上で高速化するRR-Full SGDとRR-Sparse SGDを提案する。 |
| 3.1 SGDとAtomic方式 | SGDは、グラフ上の距離と配置上の距離のずれを、頂点対ごとの更新によって減らす。 |
| 3.1 SGDとAtomic方式 | 頂点対の移動量は、Zhengら[28]のSGDに従い、現在の距離と目標距離の差から求める。 |
| 3.1 SGDとAtomic方式 | 両端に対応する重みを$w_{ij}^{(i)},w_{ij}^{(j)}$、反復$t$の学習率を$\eta_t$とすると、更新は次式で表される。 |
| 3.1 SGDとAtomic方式 | 現在距離が目標より長ければ両端を近づけ、短ければ遠ざける。 |
| 3.1 SGDとAtomic方式 | Sparse SGDは、実辺と少数のpivotを介して遠方の距離を表す項からなるSparse Stressを、SGDで最適化する既存手法である[20, 28]。 |
| 3.1 SGDとAtomic方式 | Sparse SGDでは式(3)に端点別の重みを用い、重み0の端点は固定する。 |
| 3.1 SGDとAtomic方式 | CPU版は更新対象を毎反復シャッフルし、直前の更新を反映した座標を次の対に用いる。 |
| 3.1 SGDとAtomic方式 | Atomic比較実装はMetal APIを直接用いて構築した。 |
| 3.1 SGDとAtomic方式 | 図1： Atomic方式では両端のロックを保持したまま座標を読み、更新し、解放する。 |
| 3.1 SGDとAtomic方式 | FullとSparseのAtomic実装は、完了の確認方法と更新順が異なる。 |
| 3.2 ミニバッチSGD | ミニバッチSGDは、複数の項を一つのバッチにまとめ、それらの勾配を用いてパラメータを更新する方法である[4]。 |
| 3.2 ミニバッチSGD | ここで$\alpha_t$は学習率である。 |
| 3.2 ミニバッチSGD | 本研究は、この「複数の項をまとめて並列処理する」という考えから、グラフ描画の頂点対更新をまとめてGPUへ割り当てる着想を得た。 |
| 3.3 Full SGD | グラフ描画では、端点を共有しない頂点対を集めれば、勾配を集約せずに独立に更新できる。 |
| 3.3 Full SGD | Round-robinは総当たりの組合せを、各対象が一度だけ現れるラウンドに分ける。 |
| 3.3 Full SGD | 同一ラウンド内の独立性は、異なるラウンド順でも同じ最終配置を得ることを保証しない。 |
| 3.3 Full SGD | RR-Full SGDは、ブロック間とブロック内の2段階で独立な更新を構成する。 |
| 3.3 Full SGD | 外側のRRでは、各ブロックが同じラウンド内で高々一つのタイルに属するように組合せを作る。 |
| 3.3 Full SGD | タイル内部では、各頂点が一つの相手だけと組になるmatchingを作る。 |
| 3.3 Full SGD | 図2： (a)全頂点対をブロック単位に分割する。 |
| 3.3 Full SGD | 座標はタイル処理中に共有メモリへ保持し、複数の更新で再利用する。 |
| 3.3 Full SGD | 座標の再利用により、グローバルメモリへの読取りと書戻しを削減する。 |
| 3.3 Full SGD | 各反復で、自己タイルを含む外側ラウンド順と、タイル内のmatching順をシャッフルする。 |
| 3.3 Full SGD | タイルサイズは座標の再利用量と並列度の両方に影響する。 |
| 3.4 Sparse SGD | RR-Sparse SGDは、3.1節で述べたSparse SGDの片側更新と両側更新を分離し、それぞれに応じて並列実行する。 |
| 3.4 Sparse SGD | 片側更新は、変更する頂点ごとにまとめて独立に処理する。 |
| 3.4 Sparse SGD | 両側更新は、pivot対のRRと実辺のmatchingによって同時書込みを避ける。 |
| 3.4 Sparse SGD | 図3： (a)更新対象を分割する。 |
| 3.4 Sparse SGD | 片側更新の段階を片側相、両側更新の段階を両側相と呼び、前の段階の更新が完了してから次へ進む。 |
| 3.4 Sparse SGD | Sparse化による項数の削減と、RRによる並列化は異なる役割を持つ。 |




</details>




<details><summary>Conclusionの流れ（段落ごとのトピックセンテンス）</summary>




| 節 | トピックセンテンス |
|---|---|
| 導入 | 本研究では、頂点対更新の依存関係と座標再利用に着目し、RR-Full SGDとRR-Sparse SGDを構成した。 |
| 導入 | Full RRは比較可能19グラフすべて、Sparse RRは比較可能37グラフすべてでCPUより速かった。 |
| 導入 | 今後は、前処理とメモリ使用量の改善、異なるGPUでの再現性を検証する。 |
| 導入 | 提案方式をwgpuとWGSLで実装したが、本実験はMetalバックエンド上の単一機種での評価であり、他の実行環境への性能の一般化は今後の検証課題である。 |




</details>




<details><summary>2026-10-30 図の編集メモ（論文本文外）</summary>




- 添付の4096×4096画像3枚を英語論文のタイトル・著者情報の下に配置し、頂点数・無向辺数を併記した。最大規模はItalyの6,686,493頂点・7,013,978辺。
- 本文4.3と図7は、Italy・Germanyを40 pivot・15反復・seed 0で再測定した大規模適用実験へ更新した。
- 手法差の配置比較図は次の改善対象。まずluxembourg_osmでCPU / Atomic / RRの同一入力・seed・pivots・反復数をそろえ、同じ頂点色・描画設定で並べる。stressと実行時間も併記し、良い配置だけを選ばない。今回の3枚はいずれもRRの例なので、手法比較図としては使用しない。




### 先輩論文に合わせた図の構成（2026-10-30）




伊藤先輩の *Constrained Graph Drawing by Stochastic Gradient Descent* のFigure 1・Figure 3を参照。英語論文の冒頭図は同じ幅・下端揃えの3パネルにし、(a) belgium_osm、(b) netherlands_osm、(c) italy_osmと対応する|V|・|E|を示す構成へ修正した。キャプションでは何の図か、最大規模、色の意味を説明し、詳細条件は本文4.3にまとめた。




本文の配置比較図は先輩のFigure 3にならい、行をグラフ・条件、列をCPU / Atomic / RRにそろえ、定量結果の直後にVisual Comparisonとして配置する方針。比較データの選定・再描画は未実施。
</details>


## 2026-11-05 更新記録（論文本文外）


添付した英語PDFを正として本文・図・数値を更新した。Fullは16分割、Sparseは従来の4分割。タイル実験と100 seed探索は今回は外し、6個のBAグラフを追加した。アブストラクトは依頼どおり据え置きのため、そこに残る旧Full数値は更新後の本文と一致していない。実験条件・生記録・変更の詳細は[作業ログ #9051](https://vdslab.esa.io/posts/9051)を参照。

