# v2 実験結果

記録 2260 件、成功 2260 件、失敗 0 件。未実行 0 件。比較整合性エラー 0 件。

機器：CPU Apple M4 Pro、GPU Apple M4 Pro。主比較は 14 個の代表グラフ。SGD論文の243グラフによる網羅評価と同じ範囲を検証した結果ではない。

速度は前処理を除くSGD反復のhost実時間。同期・シャッフル・投入を含む。速度比は同一グラフ・同一seedのCPU時間÷各手法時間、stress比は各手法÷同じfamilyのCPU。1より大きい速度比は高速、1より小さいstress比は良好。

## 今回わかったこと

- Full RRは最大 10.12×（USpowerGrid）、Sparse RRは最大 30.15×（luxembourg_osm）。小グラフでCPUより遅い例もあり、常に速いわけではない。
- 品質も同時に確認する。web-NotreDame / sparse RRの平均stress比 1.244、EVA / full RRの平均stress比 1.138と悪化した例があり、高速化だけで有効性を結論づけない。
- EVAではT=32→1024でSGD時間中央値 3.873→0.365 秒、平均CPU stress比 1.027→1.143。Tは速度と更新順の両方を変える。傾向はグラフ依存で、単調に変わるとは限らない。
- sharedの一貫した高速化は見られず、24条件中21条件で95%区間が1を含む。shared/global座標は240/240完全一致。WG変更でも120/120完全一致。

## 実験条件

| 実験 | グラフ数 | 比較 | seed / グラフ | 記録数 |
|---|---:|---|---:|---:|
| quality-timing | 10 | Full 3手法 + Sparse 3手法 | 25 | 1500 |
| large-sparse | 4 | Sparse 3手法 | 10 | 120 |
| rr-memory-tiling | 4 | Full RR; T=32,64,128,256,512,1024; W=256; shared/global | 10 | 480 |
| rr-workgroup | 4 | Full RR; T=1024; shared; W=128,256,512,1024 | 10 | 160 |

反復数 15、ε 0.1、Sparse pivot数 200。今回の評価はn≤8,000でexact full stress、それを超える4グラフは64起点のsampled full stress。各seedの時間測定は1回。入力は無向・単純グラフの最大連結成分。
Tはブロック当たりの頂点数（CLIのtile-size）。異なる2ブロックの処理タイルは最大2T頂点を扱う。Wは1WGのスレッド数。
入力の規模・由来・hash：[datasets.json](../config/datasets.json)。

## すぐ見る図

| 図 | 示すこと | ファイル |
|---|---|---|
| 01 | Full/Sparse各3手法のSGD高速化 | [PNG](../figures/01_iteration_speedup.png) · [PDF](../figures/01_iteration_speedup.pdf) |
| 02 | 速度比較と同じseedでの最終品質 | [PNG](../figures/02_final_stress_ratio.png) · [PDF](../figures/02_final_stress_ratio.pdf) |
| 03 | 品質のseed差と外れた例 | [PNG](../figures/03_stress_ratio_distributions.png) · [PDF](../figures/03_stress_ratio_distributions.pdf) |
| 04 | Fullの前処理・初期化・SGDの時間内訳 | [PNG](../figures/04_time_breakdown_full.png) · [PDF](../figures/04_time_breakdown_full.pdf) |
| 04 | Sparseの時間内訳と全体の速度 | [PNG](../figures/04_time_breakdown_sparse.png) · [PDF](../figures/04_time_breakdown_sparse.pdf) |
| 05 | タイルTによる速度・品質の変化 | [PNG](../figures/05_rr_tile_sensitivity.png) · [PDF](../figures/05_rr_tile_sensitivity.pdf) |
| 06 | 同一RR順序でsharedが効くか | [PNG](../figures/06_shared_memory_ablation.png) · [PDF](../figures/06_shared_memory_ablation.pdf) |
| 07 | 数値差が最終描画にどう現れるか | [PNG](../figures/07_representative_layouts.png) · [PDF](../figures/07_representative_layouts.pdf) |
| 08 | T固定でWGスレッド数Wの効果 | [PNG](../figures/08_workgroup_sensitivity.png) · [PDF](../figures/08_workgroup_sensitivity.pdf) |

## この結果で示せること

- atomic_sgd：10グラフ中3グラフでCPUより高速。速度比 0.03–2.64×、平均stress比 1.004–1.482。グラフ別の差を主結果とする。
- rr_sgd：10グラフ中9グラフでCPUより高速。速度比 0.46–10.12×、平均stress比 0.986–1.138。グラフ別の差を主結果とする。
- atomic_sparse_sgd：14グラフ中5グラフでCPUより高速。速度比 0.28–1.56×、平均stress比 0.984–1.044。グラフ別の差を主結果とする。
- rr_sparse_sgd：14グラフ中6グラフでCPUより高速。速度比 0.05–30.15×、平均stress比 0.999–1.244。グラフ別の差を主結果とする。

## Shared memory とタイル

- 3elt / rr_sgd / W=256：global時間÷shared時間は T=32: 1.00×、T=64: 1.00×、T=128: 1.00×、T=256: 1.00×、T=512: 1.07×、T=1024: 0.99×。
- EVA / rr_sgd / W=256：global時間÷shared時間は T=32: 1.00×、T=64: 1.00×、T=128: 1.00×、T=256: 0.99×、T=512: 1.12×、T=1024: 1.00×。
- USpowerGrid / rr_sgd / W=256：global時間÷shared時間は T=32: 1.00×、T=64: 1.00×、T=128: 1.00×、T=256: 1.02×、T=512: 0.93×、T=1024: 1.00×。
- dwt_2680 / rr_sgd / W=256：global時間÷shared時間は T=32: 1.01×、T=64: 1.00×、T=128: 1.00×、T=256: 0.99×、T=512: 1.02×、T=1024: 0.93×。
- 同一schedule・f32の座標一致：許容誤差内 240/240 ペア、完全一致 240 ペア。
- Tを変える実験はWGサイズWを固定。Tが大きいほど同時WG数は減る一方、座標再利用が増え、dispatchが減る。速度の優劣とstressへの影響はFigure 05で確認する。T=1でも完全な全ペアshuffleにはならない。

## Workgroup の割り当て

- 3elt：今回の範囲で最短の中央値は W=256、356.41 ms（T=1024固定）。
- EVA：今回の範囲で最短の中央値は W=256、363.61 ms（T=1024固定）。
- USpowerGrid：今回の範囲で最短の中央値は W=512、361.21 ms（T=1024固定）。
- dwt_2680：今回の範囲で最短の中央値は W=1024、175.69 ms（T=1024固定）。
- W=256との座標比較：許容誤差内 120/120 ペア。shared容量は16T bytesでWに依存しない。Wを増やしても常に速くなるとは限らず、担当仕事量と同期の効率で変わる。

## タイル割り当ての数値

- [rr-geometry.csv](../tables/rr-geometry.csv)に各T・Wのブロック数、端数ブロックの頂点数・充足率、shared bytes、ラウンド数、dispatch数、1ラウンド当たりの最大WG数を保存。Nとスケジュールからの計算値で、GPUの実測occupancyではない。
- B=ceil(N/T)、最大WG数=max(1, floor(B/2))。Tを増やすとBと並列WG数が減り、Wだけを変えてもB・shared容量は変わらない。端数充足率が低いと最後のタイルに空きが増える。これらは時間差の説明に使えるが、原因の断定にはハードウェア測定が必要。

## 図の読み方

- 01：full/sparse各3手法の高速化。点は対応比の中央値、棒はseedを再標本化した95% bootstrap区間。
- 02・03：最終stressのCPU比。02は平均と95%区間、03は各seedの分布。
- 03の各panelは縦軸範囲が異なるため、箱の見かけの高さをグラフ間で直接比較しない。未測定Full条件は明示する。
- 04：coldアルゴリズム時間の内訳は前処理・初期化も含む。下段warmはGPU初期化時間を差し引いた値。積み上げ棒は各成分の平均。
- 05：WGサイズ固定でタイルTを変更。上は時間（中央値・IQR）、下はstress（中央値・最小〜最大）。
- 06：同一T・W・更新順でshared/globalを比較。1を超えるとsharedが速い。
- 07：CPU fullの中央値に近いseedを選び、同じseedの6手法を並べた描画。最良例の選抜ではない。
- 08：T=1024・shared固定でWGサイズWを変更。時間とstressを並べ、スレッド割り当ての影響を示す。
- 生成済み：01_iteration_speedup, 02_final_stress_ratio, 03_stress_ratio_distributions, 04_time_breakdown_full, 04_time_breakdown_sparse, 05_rr_tile_sensitivity, 06_shared_memory_ablation, 07_representative_layouts, 08_workgroup_sensitivity。存在しない条件の図は作らない。

## 解釈の範囲

- 同一family内で入力・初期配置・制約・評価条件を照合し、不一致の対応比は除外。Full/Sparseをまたぐ対応比は計算しない。
- 実行ソースに未commitの変更を含む場合は、ソースsnapshot・binary hash・各実行前後のhash一致を検証。Git commitだけで実行版を表さない。
- CPUはf64、GPUはf32。6手法比較には精度差と更新順の差が含まれる。shared/global比較は同じf32と同じRR順序。
- v2のFull RRはペア順列と端数タイルを修正した実装。v1との差をタイル調整だけによる高速化と説明しない。
- sampled stressのグラフはexactと混ぜて絶対値を比較しない。同じ評価ペア・seedでCPU比を取る。
- 各seed内の繰返しを中央値でまとめ、bootstrapはseed単位。反復時間1回だけのseedは機器の時間変動とseed差を分離できない。
- 少数seed、欠測、失敗はCSVとvalidation.jsonに残す。性能・品質の保証や正式掲載可否をこの集計だけで認定しない。
- Full RRのGPU timestamp区間は保存済み。最初〜最後のdispatch間には投入間の待ちも含むためkernel単独時間とは扱わない。cache miss・occupancy・物理globalアクセス回数は未計測。sharedの効果にはアクセスと同期方法の違いも含む。
- 固定反復後の結果であり、同じstressまで到達する時間や途中の収束曲線は未取得。

## 確認事項

- Uncommitted execution source: frozen snapshot and per-run input/binary/source hashes verified
- bcsstk31: sampled full stress; not an exact all-pairs score
- fe_4elt2: sampled full stress; not an exact all-pairs score
- luxembourg_osm: sampled full stress; not an exact all-pairs score
- web-NotreDame: sampled full stress; not an exact all-pairs score
- 全件：../tables/validation.json。

## グラフ別の数値（詳細）

| Graph | Family | Method | 成功 / 計画 | 評価 | 対応seed | 時間 ms（中央値） | 高速化 | stress比（平均） |
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
