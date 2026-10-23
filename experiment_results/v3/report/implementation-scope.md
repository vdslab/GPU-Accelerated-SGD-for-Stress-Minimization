# v3：6実装の比較範囲

更新アルゴリズムはcommit `3f57b38dd551573de85f7ce1d13f24909f4029f4`で固定し、commit `a5910d9`で保存用bufferとflushだけを修正した。正式測定のsource/binary版はconfig/execution.jsonに保存する。以下はコードを読んで確認した仕様であり、追加の性能実験や原因切り分けの結果ではない。

## 共通の条件

- 同じfamily内では、入力グラフ・seed・反復数・epsilonを揃える。v3は15反復、epsilon=0.1。Sparseの基本条件はpivot=200。
- Fullは全到達可能ペアの最短路距離dと重みw=1/d²を使う。初期座標は3実装とも共通の`seeded_positions`で作る。
- Sparseの3実装はCPU版の`graph.rs`を直接共有する。pivot選択、領域、ペア集合、距離、両端の方向別重みは同じ前処理から生成する。GPU向けには値をf32へ変換する。
- 学習率は、正の重みの最小・最大からeta_max=1/w_min、eta_min=epsilon/w_maxを作り、15反復で指数的に下げる。同一family・同一Sparse pivot条件で同じf64列を作り、GPUではf32へ変換する。
- 各ペアの基本式はr=((現在距離−d)/(2×現在距離))×座標差。Fullはmu=min(w×eta,1)、Sparseはmu_u=min(w_u×eta,1)、mu_v=min(w_v×eta,1)を方向別に計算し、両端へ逆向きに適用する。Sparseで重み0の端点は動かさない。
- FullとSparseでは制約集合・重み・初期乱数の使い方が異なる。同じ数値のseedでも、familyをまたいで初期座標が同じとは限らない。

## 実装ごとの違い

| 実装 | 更新の精度 | 順序・シャッフル | 座標の扱いと実行系 |
|---|---|---|---|
| Full CPU | f64 | 毎反復、全ペア配列をseed付きRNGでshuffleして逐次更新 | CPUの座標配列を直接更新 |
| Full Atomic | f32 | **明示shuffleなし**。u<vの辞書順ペア配列を転送し、workgroup番号で参照。実際の更新順はGPUのlock獲得に依存 | native Metal。32スレッドのうち1つが1ペアを担当。global座標を両端lock中にatomic加算 |
| Full RR | f32 | 外側のtileラウンドと、内側のmatching順序を毎反復shuffle。CPUの全ペアshuffleとは異なる乱数消費・順序 | WGSL→wgpu。v3基本条件はtile=1024、workgroup=1024、shared座標。matching間にbarrier、ラウンドごとにGPU完了待ち |
| Sparse CPU | f64 | pivot選択・初期座標生成に使ったRNGを引き継ぎ、毎反復ペア全体をshuffle | CPUで逐次更新 |
| Sparse Atomic | f32（出力時f64化） | CPUと同じ前処理後RNGで、全ペアIDを毎反復shuffle。最終的な更新順はlock獲得と再投入に依存 | native Metal。両端lockとatomic加算。1回のdispatchで未完了のペアだけCPUが再投入 |
| Sparse RR | f32（出力時f64化） | 独立したseed派生RNGでpivot順・matchingラウンド順をshuffle。one-sided phaseを先/後に置く選択と、頂点ごとのpivot巡回開始位置も変える | WGSL→wgpu。256スレッド/workgroup。one-sidedは各頂点の座標をlocalに保持し、そのphase中のpivot座標を読む。two-sidedは頂点が重ならないmatchingで更新 |

今回取得済みのGPU metadataは4実装ともApple M4 Pro / Metal。native Metal APIとwgpu経由の実装は、同じGPU・backendでも同じコンパイル経路ではない。

## 数値計算にも差がある

- CPUは距離・重み・学習率・座標をf64で扱う。GPUはf32。初期座標hashが同じでも、GPU更新開始時にはf32への丸めが入る。f64に戻して保存しても失われた精度は戻らない。
- Full RRはshaderでw=1/(d×d)をf32計算する。Full AtomicはCPUで作ったf64のwをf32へ丸めて渡す。この2つも一般にはbit単位で同じ値とは限らない。
- ゼロに近い距離（1e-12未満）の処理も異なる。CPU2種は乱数角度の微小方向、Atomic2種は固定方向(1e-6,1e-6)、Sparse RRはseed等を使うhash角度を使う。Full RRは距離だけを1e-12で下限処理し、方向ベクトル自体は差し替えないため、完全一致点の差ベクトルは0のまま。
- 数学的に同じ式でも、`((dist-d)/2)×(diff/dist)`と`((dist-d)/(2×dist))×diff`など評価順が違う。GPUのlengthやコンパイラ最適化も含め、演算結果の完全一致を前提にしない。

## 同期・完了・時間の解釈

- Atomicは両端のlock取得後に座標を読み、書込み後に解放する。lockにはdevice範囲のfenceがある。Full Atomicのlock再試行は無制限。Sparse Atomicはshader内128回の後に未完了ペアをhostへ戻し、進捗なし又は1024pass以上ならエラーにする。
- Full Atomicは反復のMetal完了状態と更新件数を確認。Sparse Atomicは完了flagを使い、pendingが空になるまで進める。v3成功判定でも予定更新数=完了更新数、最終未完了数=0を確認する。これらはlock競合の発生回数を数える指標ではない。
- **Full Atomicは各ペアの更新順IDと更新前4座標を常時記録する。** そのbuffer確保・GPU書込みも現在実装の負荷である。Sparse Atomicの完了flag・再投入、RRのschedule準備やdispatch数も異なる。速度差をlock命令だけの費用と解釈しない。
- v3の主なSGD時間は各反復のhost経過時間の合計。shuffle・dispatch・GPU完了待ち・Sparse Atomic再投入を含む。座標snapshotの読出し・保存、stress評価は除外する。GPU kernelだけのdevice実行時間とは異なる。
- checkpointは最終重心移動前に保存する。最終出力の重心移動は別処理。距離の理論値は平行移動に不変だが、f32での重心移動は丸め誤差を加え得る。

## No.7：pivot数を変える比較の交絡

Sparseの初期座標は**同じseedでもpivot数Pに依存する**。SparseはFull用の`seeded_positions`を使わず、`StdRng::seed_from_u64(seed)`で作った1本のRNGを次の順で使う。

`pivot選択（最初の1件＋残りの距離比例sampling） → 制約・学習率の生成 → 同じRNGで初期座標生成 → CPU/Atomicでは更新順shuffleにも継続利用`

- pivot選択の反復回数は`min(P,頂点数)`で変わる。初期座標を作る前の乱数消費が変わるため、P=50/100/200/400の同じseed同士で初期座標の一致は保証されない。全て同じ実効pivot数にclampされた場合は、この理由による差は生じない。
- 同じP・同じseedの3手法では共通前処理と初期座標が一致する。Pをまたぐ比較とは分ける。
- Pを変えると制約・方向別重みも変わり、そこから導出する学習率列も変わり得る。CPU/Atomicのshuffle開始時のRNG状態にも影響する。
- No.7は各Pで現在の処理を行った速度・品質の比較である。「同一初期配置でpivot数だけを変えた因果比較」とは解釈しない。品質差の全量を制約数の増減だけへ帰属できない。初期配置を独立RNGへ分ける対照実験は、今回の固定済みv3には追加していない。

根拠：[Sparse前処理・pivot選択・初期座標生成](/Users/kose/master/atomic-scheduling-investigation/implementations/sparse/baseline-sparse-sgd-non-gpu/src/graph.rs:314)、[RNG生成と引継ぎ](/Users/kose/master/atomic-scheduling-investigation/implementations/sparse/baseline-sparse-sgd-non-gpu/src/main.rs:159)。ソースの静的確認であり、追加実行による確認ではない。

## 結果から言える範囲

v3は、各実装をこの条件で動かした速度と最終品質の比較である。**品質差の寄与を「更新順だけ」と特定する実験ではない。** 更新順、並列化単位、f64/f32、式の評価順、微小距離処理、コンパイル経路が同時に異なる。

同じfamily内のペア・重みの共通性と、実行記録の入力/初期座標/preprocess hashは比較の前提を確認できる。ただし、全更新の数学的正しさ・lockの形式的な正当性や、品質差に対する各要因の寄与率をv3だけで証明することはできない。精度や更新順を個別に固定した対照実験は今回の比較と区別する。

## 参照ソース

- [共通Full seed生成](/Users/kose/master/atomic-scheduling-investigation/shared/experiment-common/src/seed.rs)
- [Full CPU更新](/Users/kose/master/atomic-scheduling-investigation/implementations/full/baseline-sgd-non-gpu/src/algorithm.rs)、[Full前処理](/Users/kose/master/atomic-scheduling-investigation/implementations/full/baseline-sgd-non-gpu/src/graph.rs)
- [Full Atomic host](/Users/kose/master/atomic-scheduling-investigation/implementations/full/vram-lock-native/src/metal.rs)、[shader](/Users/kose/master/atomic-scheduling-investigation/implementations/full/vram-lock-native/src/shader.metal)
- [Full RR host](/Users/kose/master/atomic-scheduling-investigation/implementations/full/rr_gpu/src/gpu.rs)、[shader](/Users/kose/master/atomic-scheduling-investigation/implementations/full/rr_gpu/src/shader.wgsl)
- [Sparse共通前処理](/Users/kose/master/atomic-scheduling-investigation/implementations/sparse/baseline-sparse-sgd-non-gpu/src/graph.rs)、[CPU更新](/Users/kose/master/atomic-scheduling-investigation/implementations/sparse/baseline-sparse-sgd-non-gpu/src/algorithm.rs)
- [Sparse Atomic host](/Users/kose/master/atomic-scheduling-investigation/implementations/sparse/vram-lock-sparse/src/metal.rs)、[shader](/Users/kose/master/atomic-scheduling-investigation/implementations/sparse/vram-lock-sparse/src/shader.metal)
- [Sparse RR schedule](/Users/kose/master/atomic-scheduling-investigation/implementations/sparse/sparse-sgd-gpu/src/schedule.rs)、[host](/Users/kose/master/atomic-scheduling-investigation/implementations/sparse/sparse-sgd-gpu/src/gpu.rs)、[shader](/Users/kose/master/atomic-scheduling-investigation/implementations/sparse/sparse-sgd-gpu/src/shader.wgsl)
