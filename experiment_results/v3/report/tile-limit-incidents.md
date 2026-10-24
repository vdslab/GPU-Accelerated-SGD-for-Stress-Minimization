# v3 タイル感度実験：小Tのwarmup時間上限停止

正式core 3,210条件の一巡完了後に更新した事故メモ。対象はFull RR GPU、W=1024、shared座標である。No6全330条件のうち新規正式実起動275件はすべて成功、basic再利用30条件も成功、新規未起動25条件は以下のwarmup失敗に由来する。66個のグラフ・T条件では61条件が5seed完備、5条件が全欠測で、部分成功条件はなかった。後続のSNAP原配布補完90条件は扱わない。

## core最終値で確認した停止

| Graph | T | 実warmup失敗回数 | 停止wall秒 | 観測RSS GiB | 最終完了反復 | 完了分累積SGD秒 | 正式未起動枠 |
|---|---:|---:|---:|---:|---:|---:|---:|
| USpowerGrid | 1 | 1 | 120.143 | 1.098 | 13 | 116.432 | 5 |
| bcsstk16 | 1 | 1 | 120.152 | 1.026 | 14 | 118.506 | 5 |
| fe_4elt2 | 1 | 1 | 120.181 | 4.440 | 5 | 102.889 | 5 |
| fe_4elt2 | 2 | 1 | 120.168 | 3.368 | 11 | 109.619 | 5 |
| wiki-Vote | 1 | 1 | 120.177 | 1.949 | 11 | 117.981 | 5 |

5実行ともstatusは`timeout`で、memory_limitではない。T=1は4グラフでwarmup各1実行が失敗し、dwt_2680とEVAではwarmup・正式5seedとも成功した。fe_4elt2はT=2もwarmup1実行が失敗した。**timeoutの実warmupは計5件、その影響で未起動の正式条件は25件**である。未完了実行の最終stressや15反復完了時間は補わない。成功条件の同seed CPU比・最短T・非劣位集合は[No6/7結果メモ](sensitivity-observations.md)を参照。

## USpowerGrid T=1で観測した事象

- T=1、seed 0の**warmup 1実行**が、nativeプロセス全体の120秒上限で停止した。記録は`timeout`、wall 120.142770秒、終了コード−15、観測RSS最大1,178,812,416 bytes（約1.10 GiB）。今回の停止理由は時間上限である。
- 同じwarmupの失敗を受け、本測定seed 0〜4の5条件は起動されなかった。**実際のtimeoutは1件、未起動の条件は5件**と分けて数える。
- traceは反復13まで完了を記録し、その累積SGD区間は116.432325秒だった。直近の完了反復10〜13はそれぞれ8.880859、8.870006、8.844988、8.927944秒。13反復分の進行を確認した上での時間上限停止である。
- 停止した正確なouter roundや命令は記録されていない。反復13完了後から反復14完了記録前の段階までしか絞れない。15反復完了時の最終stress・完了時間は欠測であり、部分進捗から補わない。

wallは入力読込・前処理・GPU初期化・SGD・最終stress評価・出力を含む上限の時計である。traceの累積SGD秒は反復区間だけで、両者の差をそのまま前処理時間とはみなさない。差には、まだ完了記録のない反復の処理が含まれる可能性がある。監視間隔が約200 msなので、停止wallが120秒を少し超えること自体は設定と整合する。

## TとWが変える処理

1. 頂点を大きさTのブロックへ分ける。ブロック数は`B = ceil(N/T)`。
2. CPU側で、同じouter round内に同一ブロックが重複しないブロック対を並べる。最後に各ブロックのself-tileも別roundへ詰める。
3. 各反復でouter roundの順をshuffleする。各outer roundごとに、CPUでタイル一覧とinner permutationを作り、GPUのbufferへ書く。
4. **1 outer roundにつき1 compute pass・1 `dispatch_workgroups`・1 `queue.submit`・1 `poll(wait_indefinitely)`**を呼ぶ。現在の実装はこの完了確認を終えてから次roundへ進む。
5. 1 workgroupが1タイルを担当する。異ブロックタイルではT回のinner matchingを順番に処理し、各matchingの間はworkgroup内のbarrierを使う。

sharedモードでは、担当ブロックの座標をglobal bufferからworkgroup領域へ読み、内部のmatchingを終えてからglobalへ書き戻す。workgroup内barrierは同じworkgroupの同期である。別workgroupを含むouter roundの完了と、次roundが前roundの更新済みglobal座標を使う区切りは、host側のsubmit/pollで作っている。同一outer roundのworkgroup間は担当ブロックが重複しない。

**Wは1 workgroupのinvocation数と、タイル内の仕事を分担するstrideを決める。B・outer round数はTとNから決まり、Wでは減らない。** W=1024でも、T=1のcross-tileでSGD更新を計算するのは`tx=0`の1 invocationだけ。他のinvocationもworkgroup内barrierには到達する。T=2/4なら1 inner matchingで更新を担当し得るのは最大2/4 invocationである。これはshader上の仕事の割当であり、実GPUのoccupancyや待ち時間を測定した値ではない。

T=1ではrunnerが`--no-inner-shuffle`を付ける。cross-tileのinner matchingは1つだけで、outer shuffleは有効のまま。self-tileは1頂点なのでSGDの点対更新をしないが、scheduleには残り、dispatchとsubmit/pollを通る。

## ソースから求められる予定回数

`B >= 2`のとき、cross outer roundはBが偶数ならB−1、奇数ならB。self outer roundは`ceil(B / max(floor(B/2), 1))`。B=1はselfの1 roundだけである。

USpowerGridについて以下を算定できる。**SGD本体のouter roundだけ**の回数で、初期化・最終readback・診断checkpoint等の追加処理は含めない。

| T | B | cross round / 反復 | self round / 反復 | submit/poll呼出し / 反復 | 15反復の予定dispatch数 |
|---:|---:|---:|---:|---:|---:|
| 1 | 4,941 | 4,941 | 3 | 4,944 | 74,160 |
| 2 | 2,471 | 2,471 | 3 | 2,474 | 37,110 |
| 4 | 1,236 | 1,235 | 2 | 1,237 | 18,555 |
| 1024 | 5 | 5 | 3 | 8 | 120 |

T=2/4の`rounds`と`dispatches`は実際の成功recordでも上表と一致した。T=1の74,160は完走した場合の**予定数**で、停止実行の実測総数ではない。反復13までの完了から、少なくとも64,272回のSGD outer-round処理が終わったことは算定できる。途中反復での呼出し数は不明である。

ここでのdispatch数はGPU workgroup数とは異なる。1 dispatchでそのroundの複数タイルを処理する。T=1では1反復にcrossの12,204,270 workgroupとselfの4,941 workgroupを実行する予定で、各cross workgroupが1点対を更新する。点対更新総数`N(N−1)/2 = 12,204,270`はTを変えても同じだが、それを何個のworkgroupとhost同期要求へ分けるかは変わる。poll呼出し数は、すべての呼出しで実待機が同じ時間発生したことを意味しない。

## 近接するTで確認した完了結果

各値はcore完了後に確認した個別recordの範囲。warmupは以下の成功本測定の集計に混ぜない。T=2・T=4ともseed 0〜4の正式5実行が完了している。

| T | 照合対象 | 成功した15反復のSGD秒 | native process wall秒 | 最終stress |
|---:|---|---:|---:|---:|
| 1 | warmup seed 0 timeout、本測定5条件未起動 | 未完了 | warmup 120.143で停止 | 欠測 |
| 2 | 本測定seed 0〜4、5件成功 | 54.393〜64.705 | 55.126〜65.587 | 702,049.6〜702,165.2 |
| 4 | 本測定seed 0〜4、5件成功 | 26.198〜26.725 | 27.097〜27.534 | 702,199.1〜702,258.7 |

## 調査結果と残る範囲

小さいTではouter roundごとのhost側準備・GPU投入・完了確認が増え、W=1024に対するタイル内のSGD担当invocation数も少なくなる。T=1の4,944回/反復、T=2の2,474回/反復、T=4の1,237回/反復という実装上の分割と、T=1の直近4反復が約8.845〜8.928秒で完了を記録し、T=2/4が時間内に完了した観測は整合している。

ただし、CPUのschedule/permutation準備、buffer書込、command encoder生成、driver投入、GPUの計算・barrier、hostのpoll待ちを個別には計測していない。今回の結果から遅さを特定の1成分へ断定したり、回数比だけから実時間比を決めたりはしない。T=1の正確な15反復完了時間や最終stressは外挿せず欠測として残す。時間上限を変更した再測定は、この正式一巡の結果へ混ぜない。

## 根拠

- 小さな実行metadata：`warmups/warmup-{USpowerGrid,bcsstk16,fe_4elt2,wiki-Vote}-rr_sgd-pNone-t1/record.json`、`warmups/warmup-fe_4elt2-rr_sgd-pNone-t2/record.json`、各`command.json`・`trace/trace.jsonl`。
- 個別の成功metadata：`raw/artifacts/v3-tiling-USpowerGrid-rr_sgd-s{0..4}-pNone-t2/record.json`、同`s{0..4}-pNone-t4/record.json`。
- 固定source commit `118d3ec22b350fcc7522527154f3d03131215a25`：[outer schedule](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/118d3ec22b350fcc7522527154f3d03131215a25/implementations/full/rr_gpu/src/gpu.rs#L196)、[shared/global座標helper](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/118d3ec22b350fcc7522527154f3d03131215a25/implementations/full/rr_gpu/src/gpu.rs#L108)、[outer roundの投入と完了確認](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/118d3ec22b350fcc7522527154f3d03131215a25/implementations/full/rr_gpu/src/gpu.rs#L603)、[shaderのinner matching](https://github.com/vdslab/GPU-Accelerated-SGD-for-Stress-Minimization/blob/118d3ec22b350fcc7522527154f3d03131215a25/implementations/full/rr_gpu/src/shader.wgsl#L32)。

このメモの作成ではsourceとrecord/trace・正式JSONL metadataだけを読み、実験・コンパイル・テスト・描画・座標や大きな行列の解析は行っていない。
