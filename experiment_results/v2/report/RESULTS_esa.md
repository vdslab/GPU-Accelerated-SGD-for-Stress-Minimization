# v2 実験結果：GPU版SGDの速度・品質評価

> **実行結果:** 2,260件を記録し、成功2,260件、失敗0件、未実行0件。比較整合性エラーは0件。
>
> **実験環境:** CPU Apple M4 Pro、GPU Apple M4 Pro。
>
> **評価範囲:** 主比較は代表グラフ14個。SGD論文の243グラフ全体を再現・網羅した評価ではない。

## 要旨

- **Full RR:** CPUより高速だったのは10グラフ中9グラフ。速度比は0.46〜10.12倍で、最大はUSpowerGrid。小規模グラフではCPUより遅い場合がある。
- **Sparse RR:** CPUより高速だったのは14グラフ中6グラフ。速度比は0.05〜30.15倍で、最大はluxembourg_osm。全グラフで速いわけではない。
- **品質も速度と併せて評価する必要がある。** 平均stress比はEVA / Full RRで1.138、web-NotreDame / Sparse RRで1.244となった。高速化だけでは品質を含む有効性を結論づけられない。
- **タイルサイズTの効果はグラフ依存。** EVAではT=32から1024にするとSGD時間中央値が3.873秒から0.365秒へ短縮した一方、平均CPU stress比は1.027から1.143へ増加した。Tは速度と更新順の両方に影響する。
- **shared memoryによる一貫した高速化は確認できなかった。** 24条件中21条件で速度比の95%区間が1を含む。shared/globalの座標は240/240ペアで完全一致した。Workgroupサイズ変更時も120/120ペアで完全一致した。

## 実験条件

| 実験 | グラフ数 | 比較内容 | seed数 / グラフ | 記録数 |
|---|---:|---|---:|---:|
| quality-timing | 10 | Full 3手法 + Sparse 3手法 | 25 | 1,500 |
| large-sparse | 4 | Sparse 3手法 | 10 | 120 |
| rr-memory-tiling | 4 | Full RR、T=32〜1024、W=256、shared/global | 10 | 480 |
| rr-workgroup | 4 | Full RR、T=1024、shared、W=128/256/512/1024 | 10 | 160 |

- 反復数15、ε=0.1、Sparse pivot数200。
- n≤8,000の評価はexact full stress。これを超える4グラフは64起点のsampled full stress。
- 入力は無向・単純グラフの最大連結成分。
- 時間は前処理を除くSGD反復のhost実時間で、同期・shuffle・dispatch投入を含む。各seedの時間測定は1回。
- Tはブロック当たりの頂点数（CLIの`tile-size`）。2ブロックの処理タイルでは最大2T頂点を扱う。Wは1 workgroup当たりのスレッド数。
- 入力の規模・由来・hashは`experiment_results/v2/config/datasets.json`を参照。

## 手法別の結果

| 手法 | CPUより高速だったグラフ数 | 速度比の範囲 | 平均stress比の範囲 |
|---|---:|---:|---:|
| atomic_sgd | 3 / 10 | 0.03〜2.64倍 | 1.004〜1.482 |
| rr_sgd | 9 / 10 | 0.46〜10.12倍 | 0.986〜1.138 |
| atomic_sparse_sgd | 5 / 14 | 0.28〜1.56倍 | 0.984〜1.044 |
| rr_sparse_sgd | 6 / 14 | 0.05〜30.15倍 | 0.999〜1.244 |

グラフごとの差を主結果として扱う。速度比は同じグラフ・同じseedでのCPU時間÷各手法時間、stress比は同じfamilyのCPU stressに対する各手法の比。速度比が1より大きいと高速、stress比が1より小さいとstressが低い。

## タイルサイズとshared memory

### shared/globalの速度比

値はglobal時間÷shared時間。1より大きいとsharedが高速。

| グラフ | T=32 | T=64 | T=128 | T=256 | T=512 | T=1024 |
|---|---:|---:|---:|---:|---:|---:|
| 3elt | 1.00× | 1.00× | 1.00× | 1.00× | 1.07× | 0.99× |
| EVA | 1.00× | 1.00× | 1.00× | 0.99× | 1.12× | 1.00× |
| USpowerGrid | 1.00× | 1.00× | 1.00× | 1.02× | 0.93× | 1.00× |
| dwt_2680 | 1.01× | 1.00× | 1.00× | 0.99× | 1.02× | 0.93× |

同一schedule・f32での座標は許容誤差内240/240ペア、完全一致240ペア。Tを変えた実験ではWを固定した。Tが大きいほど同時workgroup数は減る一方、座標再利用が増えdispatch数が減る。速度とstressへの影響はグラフにより異なり、単調な傾向ではない。T=1でも完全な全ペアshuffleにはならない。

### WorkgroupサイズW

以下はT=1024固定時に、今回試したWの範囲で最短だった中央値。

| グラフ | 最短のW | 時間中央値 |
|---|---:|---:|
| 3elt | 256 | 356.41 ms |
| EVA | 256 | 363.61 ms |
| USpowerGrid | 512 | 361.21 ms |
| dwt_2680 | 1024 | 175.69 ms |

W=256との座標比較は許容誤差内120/120ペアで一致した。shared容量は16T bytesでWに依存しない。Wを増やしても常に速くなるわけではなく、担当仕事量と同期効率によって変わる。

### タイル割り当て

各T・Wのブロック数、端数ブロックの頂点数・充足率、shared bytes、ラウンド数、dispatch数、1ラウンド当たりの最大workgroup数は`experiment_results/v2/tables/rr-geometry.csv`を参照。この値はNとスケジュールからの計算値であり、GPUの実測occupancyではない。

`B=ceil(N/T)`、最大workgroup数は`max(1, floor(B/2))`。Tを増やすとBと並列workgroup数が減る。Wだけを変えてもBとshared容量は変わらない。端数充足率が低いと最後のタイルに空きが増える。これらは時間差を考える材料だが、原因の断定にはハードウェア測定が必要。

## 図

図ファイルは`experiment_results/v2/figures/`に保存。

| 図 | 内容 | ファイル名 |
|---|---|---|
| 01 | Full/Sparse各3手法のSGD高速化 | `01_iteration_speedup` |
| 02 | 速度比較と同じseedでの最終品質 | `02_final_stress_ratio` |
| 03 | seedごとの品質分布と外れ値 | `03_stress_ratio_distributions` |
| 04 | Fullの前処理・初期化・SGD時間内訳 | `04_time_breakdown_full` |
| 04 | Sparseの時間内訳と全体の速度 | `04_time_breakdown_sparse` |
| 05 | タイルTによる速度・品質の変化 | `05_rr_tile_sensitivity` |
| 06 | 同一RR順序でのshared/global比較 | `06_shared_memory_ablation` |
| 07 | 数値差が最終描画に与える影響 | `07_representative_layouts` |
| 08 | T固定時のworkgroupサイズWの効果 | `08_workgroup_sensitivity` |

図の読み方：

- **01:** 点は対応比の中央値、棒はseed単位で再標本化した95% bootstrap区間。
- **02・03:** 最終stressのCPU比。02は平均と95%区間、03はseedごとの分布。03はpanelごとに縦軸範囲が異なるため、箱の高さをグラフ間で直接比較しない。未測定Full条件は図中に明示。
- **04:** 上段のcoldアルゴリズム時間は前処理・初期化を含む。下段warmはGPU初期化時間を差し引いた値。積み上げ棒は各成分の平均。
- **05:** W固定でTを変更。上段は時間（中央値・IQR）、下段はstress（中央値・最小〜最大）。
- **06:** T・W・更新順を揃えてshared/globalを比較。
- **07:** CPU fullのstress中央値に近いseedを選び、同じseedの6手法を並べた。最良例の選抜ではない。
- **08:** T=1024・shared固定でWを変更し、時間とstressを比較。

## 解釈上の注意

- 対応比は同じfamily内で入力・初期配置・制約・評価条件を照合し、不一致の組み合わせは除外。Full/Sparseをまたぐ対応比は計算していない。
- CPUはf64、GPUはf32。6手法比較には精度差と更新順の差が含まれる。shared/global比較は同じf32・同じRR順序。
- v2のFull RRではペア順列と端数タイルの実装を修正した。v1との差をタイル調整だけの効果として説明しない。
- sampled stressのグラフはexactのグラフと絶対値で混ぜず、同じ評価ペア・seedでCPU比を比較する。
- 各seed内の繰返しは中央値に集約し、bootstrapはseed単位。各seedの時間測定は1回のため、機器の時間変動とseed差を分離できない。
- 少数seed、欠測、失敗はCSVと`experiment_results/v2/tables/validation.json`に記録。今回の集計だけでは性能・品質の保証や正式掲載可否を認定しない。
- Full RRのGPU timestamp区間は保存済み。ただし最初から最後のdispatch間には投入間の待ち時間も含むため、kernel単独時間とは扱わない。cache miss・occupancy・物理globalアクセス回数は未計測。sharedの効果にはアクセス方法と同期方法の違いも含まれる。
- 反復回数を固定した結果であり、同じstressに到達するまでの時間や途中の収束曲線は未取得。

## 検証状況

- 未commitの実行ソースについて、凍結snapshotとrunごとの入力・binary・source hashを検証済み。
- bcsstk31、fe_4elt2、luxembourg_osm、web-NotreDameはsampled full stressであり、全ペアexact stressではない。
- 全件のvalidation記録は`experiment_results/v2/tables/validation.json`。

:::details グラフ別の全測定値（展開）
| Graph |Family | Method | 成功 / 計画 | 評価 | 対応seed | 時間 ms（中央値） | 高速化 | stress比（平均） |
|---|---|---|---:|---|---:|---:|---:|---:|
| dwt_419 | full | CPU | 25/25 | exact | 25 | 12.36 | 1.00× | 1.000 |
| dwt_419 | full | Atomic GPU | 25/25 | exact | 25 | 427.92 | 0.03× | 1.004 |
| dwt_419 | full | RR GPU | 25/25 | exact | 25 | 27.19 | 0.46× | 1.000 |
| dwt_419 | sparse | CPU | 25/25 | exact | 25 | 9.57 | 1.00× | 1.000 |
| dwt_419 | sparse | Atomic GPU | 25/25 | exact | 25 | 34.30 | 0.28× | 1.000 |
| dwt_419 | sparse | RR GPU | 25/25 | exact | 25 | 176.92 | 0.05× | 0.999 |
| qh882 | full | CPU | 25/25 | exact | 25 | 62.02 | 1.00× | 1.000 |
| qh882 | full | Atomic GPU | 25/25 | exact | 25 | 658.51 | 0.09× | 1.006 |
| qh882 | full | RR GPU | 25/25 | exact | 25 | 27.32 | 2.25× | 1.000 |
| qh882 | sparse | CPU | 25/25 | exact | 25 | 25.31 | 1.00× | 1.000 |
| qh882 | sparse | Atomic GPU | 25/25 | exact | 25 | 58.90 | 0.43× | 1.001 |
| qh882 | sparse | RR GPU | 25/25 | exact | 25 | 181.42 | 0.14× | 1.002 |
| dwt_1005 | full | CPU | 25/25 | exact | 25 | 80.71 | 1.00× | 1.000 |
| dwt_1005 | full | Atomic GPU | 25/25 | exact | 25 | 723.00 | 0.11× | 1.012 |
| dwt_1005 | full | RR GPU | 25/25 | exact | 25 | 28.75 | 2.80× | 1.000 |
| dwt_1005 | sparse | CPU | 25/25 | exact | 25 | 29.67 | 1.00× | 1.000 |
| dwt_1005 | sparse | Atomic GPU | 25/25 | exact | 25 | 70.06 | 0.42× | 1.002 |
| dwt_1005 | sparse | RR GPU | 25/25 | exact | 25 | 183.19 | 0.16× | 1.001 |
| cage8 | full | CPU | 25/25 | exact | 25 | 82.52 | 1.00× | 1.000 |
| cage8 | full | Atomic GPU | 25/25 | exact | 25 | 728.91 | 0.11× | 1.026 |
| cage8 | full | RR GPU | 25/25 | exact | 25 | 30.35 | 2.71× | 1.000 |
| cage8 | sparse | CPU | 25/25 | exact | 25 | 30.02 | 1.00× | 1.000 |
| cage8 | sparse | Atomic GPU | 25/25 | exact | 25 | 71.25 | 0.42× | 1.001 |
| cage8 | sparse | RR GPU | 25/25 | exact | 25 | 182.42 | 0.16× | 1.005 |
| 1138_bus | full | CPU | 25/25 | exact | 25 | 109.05 | 1.00× | 1.000 |
| 1138_bus | full | Atomic GPU | 25/25 | exact | 25 | 781.68 | 0.14× | 1.121 |
| 1138_bus | full | RR GPU | 25/25 | exact | 25 | 76.17 | 1.43× | 1.012 |
| 1138_bus | sparse | CPU | 25/25 | exact | 25 | 34.17 | 1.00× | 1.000 |
| 1138_bus | sparse | Atomic GPU | 25/25 | exact | 25 | 76.50 | 0.45× | 1.004 |
| 1138_bus | sparse | RR GPU | 25/25 | exact | 25 | 179.54 | 0.19× | 1.004 |
| poli | full | CPU | 25/25 | exact | 25 | 690.34 | 1.00× | 1.000 |
| poli | full | Atomic GPU | 25/25 | exact | 25 | 1066.46 | 0.66× | 1.122 |
| poli | full | RR GPU | 25/25 | exact | 25 | 174.62 | 4.00× | 1.035 |
| poli | sparse | CPU | 25/25 | exact | 25 | 81.32 | 1.00× | 1.000 |
| poli | sparse | Atomic GPU | 25/25 | exact | 25 | 125.87 | 0.64× | 1.002 |
| poli | sparse | RR GPU | 25/25 | exact | 25 | 192.34 | 0.43× | 1.015 |
| dwt_2680 | full | CPU | 25/25 | exact | 25 | 942.00 | 1.00× | 1.000 |
| dwt_2680 | full | Atomic GPU | 25/25 | exact | 25 | 1307.07 | 0.72× | 1.015 |
| dwt_2680 | full | RR GPU | 25/25 | exact | 25 | 228.21 | 4.11× | 0.986 |
| dwt_2680 | sparse | CPU | 25/25 | exact | 25 | 96.76 | 1.00× | 1.000 |
| dwt_2680 | sparse | Atomic GPU | 25/25 | exact | 25 | 140.45 | 0.71× | 0.984 |
| dwt_2680 | sparse | RR GPU | 25/25 | exact | 25 | 179.40 | 0.54× | 1.030 |
| EVA | full | CPU | 25/25 | exact | 25 | 3123.42 | 1.00× | 1.000 |
| EVA | full | Atomic GPU | 25/25 | exact | 25 | 1410.31 | 2.22× | 1.203 |
| EVA | full | RR GPU | 25/25 | exact | 25 | 364.83 | 8.43× | 1.138 |
| EVA | sparse | CPU | 25/25 | exact | 25 | 191.53 | 1.00× | 1.000 |
| EVA | sparse | Atomic GPU | 25/25 | exact | 25 | 202.81 | 0.94× | 1.024 |
| EVA | sparse | RR GPU | 25/25 | exact | 25 | 409.40 | 0.48× | 1.044 |
| 3elt | full | CPU | 25/25 | exact | 25 | 3370.61 | 1.00× | 1.000 |
| 3elt | full | Atomic GPU | 25/25 | exact | 25 | 1414.92 | 2.40× | 1.482 |
| 3elt | full | RR GPU | 25/25 | exact | 25 | 367.36 | 9.17× | 1.011 |
| 3elt | sparse | CPU | 25/25 | exact | 25 | 211.07 | 1.00× | 1.000 |
| 3elt | sparse | Atomic GPU | 25/25 | exact | 25 | 214.00 | 0.99× | 1.000 |
| 3elt | sparse | RR GPU | 25/25 | exact | 25 | 176.00 | 1.20× | 1.000 |
| USpowerGrid | full | CPU | 25/25 | exact | 25 | 3836.16 | 1.00× | 1.000 |
| USpowerGrid | full | Atomic GPU | 25/25 | exact | 25 | 1448.91 | 2.64× | 1.037 |
| USpowerGrid | full | RR GPU | 25/25 | exact | 25 | 372.26 | 10.12× | 1.019 |
| USpowerGrid | sparse | CPU | 25/25 | exact | 25 | 223.15 | 1.00× | 1.000 |
| USpowerGrid | sparse | Atomic GPU | 25/25 | exact | 25 | 222.82 | 1.00× | 1.000 |
| USpowerGrid | sparse | RR GPU | 25/25 | exact | 25 | 176.66 | 1.26× | 1.004 |
| fe_4elt2 | sparse | CPU | 10/10 | sampled | 10 | 633.60 | 1.00× | 1.000 |
| fe_4elt2 | sparse | Atomic GPU | 10/10 | sampled | 10 | 468.54 | 1.36× | 1.003 |
| fe_4elt2 | sparse | RR GPU | 10/10 | sampled | 10 | 193.12 | 3.27× | 1.009 |
| bcsstk31 | sparse | CPU | 10/10 | sampled | 10 | 2610.29 | 1.00× | 1.000 |
| bcsstk31 | sparse | Atomic GPU | 10/10 | sampled | 10 | 1675.68 | 1.56× | 1.002 |
| bcsstk31 | sparse | RR GPU | 10/10 | sampled | 10 | 311.64 | 8.38× | 1.004 |
| luxembourg_osm | sparse | CPU | 10/10 | sampled | 10 | 9371.85 | 1.00× | 1.000 |
| luxembourg_osm | sparse | Atomic GPU | 10/10 | sampled | 10 | 6537.35 | 1.44× | 1.044 |
| luxembourg_osm | sparse | RR GPU | 10/10 | sampled | 10 | 308.77 | 30.15× | 1.001 |
| web-NotreDame | sparse | CPU | 10/10 | sampled | 10 | 19852.58 | 1.00× | 1.000 |
| web-NotreDame | sparse | Atomic GPU | 10/10 | sampled | 10 | 18449.17 | 1.08× | 1.001 |
| web-NotreDame | sparse | RR GPU | 10/10 | sampled | 10 | 6269.67 | 3.27× | 1.244 |

:::
