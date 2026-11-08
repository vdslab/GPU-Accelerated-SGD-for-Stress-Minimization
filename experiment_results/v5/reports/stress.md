# v5 最終Stress比較：結果と考察

## 実施と検証

通常比較は共通38件×6方式＋Sparse専用18件×3方式＝282設定、各5seed×3回の **4,230実行すべて成功**。タイル1,800実行と合わせ **6,030実行すべて成功**。通常比較はseed0〜4、15反復、epsilon0.1、Sparse p=min(200,n)。RR Fullはユーザー指定T1024・W1024・shared・両shuffleあり、RR Full/Sparseともphase_chunks1。coPapersCiteseerまでで終了し、大規模14件は今回は実行していない。

全282設定で15実行・完全な5seedが揃い、欠測は0。入力・バイナリハッシュ、反復・seed・p・RR設定、Atomicの更新完了とretry失敗0、Stress評価条件、圧縮検証記録を照合した。圧縮ファイルは存在とサイズも再確認した。全ファイルのCRCを今回改めて読み直したわけではなく、作成時のCRC・展開SHA照合を記録として確認した。

比は同じgraph・seed・repetitionのCPU値と対応させて求め、seed内3回の中央値→5seedの中央値・最小〜最大で集計する。下表の幾何平均は各グラフを同じ重みで扱う記述的要約であり、独立15seedや統計的有意差の検定ではない。Fullの分母はCPU Full、SparseはCPU Sparse。

## 品質の要約

最終Stressは速度と同じ実行・同じ最終座標から取得し、追加レイアウト実行は0。3060実行は厳密評価、1170実行は評価seed0・256始点の推定Full Stress。Sparseの学習目的関数ではなく、共通のFull Stressを比較する。

| 方式 | グラフ数 | Stress/同族CPU・幾何平均 | グラフ別比の範囲 | CPUよりStressが小さいグラフ数 |
|---|---:|---:|---:|---:|
| Atomic Full | 38 | 1.1818 | 0.9678〜2.9056 | 1/38 |
| RR Full | 38 | 1.0656 | 0.9544〜1.6710 | 10/38 |
| Atomic Sparse | 56 | 1.0083 | 0.9710〜1.0855 | 8/56 |
| RR Sparse | 56 | 1.0171 | 0.9819〜1.1552 | 11/56 |

## グラフ別の結果

比が1より小さいほどCPU基準よりStressが小さい。Stress=0による未定義比はN/Aとし、品質の適格閾値は設けない。

| グラフ | 方式 | 最終Full Stress | Stress/同族CPU | 5seedの比の最小〜最大 | 評価 |
|---|---|---:|---:|---|---|
| karate | CPU Full | 39.2682 | 1.000 | 1.000〜1.000 | exact |
| karate | Atomic Full | 39.964 | 1.018 | 0.978〜1.042 | exact |
| karate | RR Full | 38.8662 | 0.990 | 0.962〜1.063 | exact |
| karate | CPU Sparse | 39.8625 | 1.000 | 1.000〜1.000 | exact |
| karate | Atomic Sparse | 40.4464 | 0.998 | 0.957〜1.026 | exact |
| karate | RR Sparse | 40.3189 | 0.993 | 0.951〜1.181 | exact |
| dolphins | CPU Full | 114.291 | 1.000 | 1.000〜1.000 | exact |
| dolphins | Atomic Full | 121.908 | 1.028 | 0.988〜1.116 | exact |
| dolphins | RR Full | 119.934 | 1.013 | 0.995〜1.099 | exact |
| dolphins | CPU Sparse | 116.223 | 1.000 | 1.000〜1.000 | exact |
| dolphins | Atomic Sparse | 121.738 | 1.027 | 0.996〜1.063 | exact |
| dolphins | RR Sparse | 116.621 | 0.990 | 0.927〜1.048 | exact |
| dwt_198 | CPU Full | 31.6124 | 1.000 | 1.000〜1.000 | exact |
| dwt_198 | Atomic Full | 31.6445 | 1.001 | 1.001〜2.171 | exact |
| dwt_198 | RR Full | 31.6224 | 1.001 | 1.000〜2.186 | exact |
| dwt_198 | CPU Sparse | 31.6049 | 1.000 | 1.000〜1.000 | exact |
| dwt_198 | Atomic Sparse | 31.6551 | 1.002 | 1.001〜1.002 | exact |
| dwt_198 | RR Sparse | 31.6173 | 1.000 | 1.000〜2.156 | exact |
| dwt_72 | CPU Full | 73.7331 | 1.000 | 1.000〜1.000 | exact |
| dwt_72 | Atomic Full | 74.6297 | 1.012 | 1.011〜1.013 | exact |
| dwt_72 | RR Full | 73.6941 | 0.999 | 0.999〜1.000 | exact |
| dwt_72 | CPU Sparse | 73.7309 | 1.000 | 1.000〜1.000 | exact |
| dwt_72 | Atomic Sparse | 75.067 | 1.018 | 1.016〜1.020 | exact |
| dwt_72 | RR Sparse | 73.6575 | 1.000 | 0.998〜1.000 | exact |
| lesmis | CPU Full | 245.669 | 1.000 | 1.000〜1.000 | exact |
| lesmis | Atomic Full | 254.339 | 1.040 | 0.987〜1.069 | exact |
| lesmis | RR Full | 245.846 | 1.004 | 0.964〜1.046 | exact |
| lesmis | CPU Sparse | 248.969 | 1.000 | 1.000〜1.000 | exact |
| lesmis | Atomic Sparse | 254.159 | 1.023 | 1.014〜1.038 | exact |
| lesmis | RR Sparse | 249.03 | 1.000 | 0.989〜1.015 | exact |
| can_96 | CPU Full | 318.485 | 1.000 | 1.000〜1.000 | exact |
| can_96 | Atomic Full | 335.237 | 1.052 | 1.042〜1.069 | exact |
| can_96 | RR Full | 319.088 | 1.002 | 0.997〜1.003 | exact |
| can_96 | CPU Sparse | 318.915 | 1.000 | 1.000〜1.000 | exact |
| can_96 | Atomic Sparse | 333.375 | 1.045 | 1.024〜1.050 | exact |
| can_96 | RR Sparse | 318.856 | 1.001 | 0.996〜1.003 | exact |
| polbooks | CPU Full | 380.103 | 1.000 | 1.000〜1.000 | exact |
| polbooks | Atomic Full | 386.25 | 1.012 | 0.969〜1.017 | exact |
| polbooks | RR Full | 382.405 | 0.998 | 0.955〜1.015 | exact |
| polbooks | CPU Sparse | 388.319 | 1.000 | 1.000〜1.000 | exact |
| polbooks | Atomic Sparse | 398.246 | 1.026 | 0.977〜1.065 | exact |
| polbooks | RR Sparse | 383.094 | 0.982 | 0.959〜1.021 | exact |
| adjnoun | CPU Full | 844.645 | 1.000 | 1.000〜1.000 | exact |
| adjnoun | Atomic Full | 870.591 | 1.042 | 0.979〜1.067 | exact |
| adjnoun | RR Full | 823.326 | 0.981 | 0.945〜1.105 | exact |
| adjnoun | CPU Sparse | 852.78 | 1.000 | 1.000〜1.000 | exact |
| adjnoun | Atomic Sparse | 866.096 | 1.023 | 1.002〜1.032 | exact |
| adjnoun | RR Sparse | 855.752 | 1.003 | 0.971〜1.027 | exact |
| jazz | CPU Full | 1998.57 | 1.000 | 1.000〜1.000 | exact |
| jazz | Atomic Full | 2092.78 | 1.050 | 1.025〜1.053 | exact |
| jazz | RR Full | 2008.63 | 1.011 | 0.994〜1.036 | exact |
| jazz | CPU Sparse | 1995.78 | 1.000 | 1.000〜1.000 | exact |
| jazz | Atomic Sparse | 2015.85 | 1.014 | 1.001〜1.015 | exact |
| jazz | RR Sparse | 1995.81 | 1.000 | 0.998〜1.005 | exact |
| grid17 | CPU Full | 570.087 | 1.000 | 1.000〜1.000 | exact |
| grid17 | Atomic Full | 574.377 | 1.008 | 1.007〜1.008 | exact |
| grid17 | RR Full | 570.073 | 1.000 | 1.000〜1.000 | exact |
| grid17 | CPU Sparse | 574.69 | 1.000 | 1.000〜1.000 | exact |
| grid17 | Atomic Sparse | 576.476 | 1.004 | 1.002〜1.004 | exact |
| grid17 | RR Sparse | 574.527 | 1.000 | 0.999〜1.000 | exact |
| mesh3e1 | CPU Full | 103.819 | 1.000 | 1.000〜1.000 | exact |
| mesh3e1 | Atomic Full | 104.319 | 1.005 | 1.005〜1.005 | exact |
| mesh3e1 | RR Full | 103.822 | 1.000 | 1.000〜1.000 | exact |
| mesh3e1 | CPU Sparse | 105.14 | 1.000 | 1.000〜1.000 | exact |
| mesh3e1 | Atomic Sparse | 105.437 | 1.003 | 1.001〜1.003 | exact |
| mesh3e1 | RR Sparse | 105.079 | 1.000 | 0.997〜1.000 | exact |
| celegansneural | CPU Full | 5386.53 | 1.000 | 1.000〜1.000 | exact |
| celegansneural | Atomic Full | 6002.49 | 1.117 | 1.105〜1.139 | exact |
| celegansneural | RR Full | 5384.46 | 0.999 | 0.994〜1.002 | exact |
| celegansneural | CPU Sparse | 5547.31 | 1.000 | 1.000〜1.000 | exact |
| celegansneural | Atomic Sparse | 5610.66 | 1.001 | 0.995〜1.023 | exact |
| celegansneural | RR Sparse | 5630.56 | 1.008 | 0.997〜1.073 | exact |
| netscience | CPU Full | 4061.2 | 1.000 | 1.000〜1.000 | exact |
| netscience | Atomic Full | 4121.02 | 1.015 | 1.010〜1.059 | exact |
| netscience | RR Full | 4063.01 | 1.000 | 0.999〜1.002 | exact |
| netscience | CPU Sparse | 4130.41 | 1.000 | 1.000〜1.000 | exact |
| netscience | Atomic Sparse | 4155.05 | 1.004 | 1.001〜1.006 | exact |
| netscience | RR Sparse | 4155.37 | 1.006 | 1.001〜1.006 | exact |
| dwt_419 | CPU Full | 1372.02 | 1.000 | 1.000〜1.000 | exact |
| dwt_419 | Atomic Full | 1377.62 | 1.004 | 1.004〜1.005 | exact |
| dwt_419 | RR Full | 1371.02 | 1.000 | 0.998〜1.001 | exact |
| dwt_419 | CPU Sparse | 1406.93 | 1.000 | 1.000〜1.000 | exact |
| dwt_419 | Atomic Sparse | 1411.45 | 1.004 | 1.003〜1.006 | exact |
| dwt_419 | RR Sparse | 1404.14 | 0.999 | 0.996〜0.999 | exact |
| dwt_758 | CPU Full | 971.354 | 1.000 | 1.000〜1.000 | exact |
| dwt_758 | Atomic Full | 939.814 | 0.968 | 0.933〜0.980 | exact |
| dwt_758 | RR Full | 982.894 | 0.999 | 0.977〜1.031 | exact |
| dwt_758 | CPU Sparse | 1105.32 | 1.000 | 1.000〜1.000 | exact |
| dwt_758 | Atomic Sparse | 1057.79 | 1.011 | 0.938〜1.130 | exact |
| dwt_758 | RR Sparse | 1070.67 | 1.040 | 0.932〜1.091 | exact |
| dwt_1005 | CPU Full | 10727 | 1.000 | 1.000〜1.000 | exact |
| dwt_1005 | Atomic Full | 10849.3 | 1.011 | 1.010〜1.014 | exact |
| dwt_1005 | RR Full | 10728.7 | 1.000 | 0.999〜1.001 | exact |
| dwt_1005 | CPU Sparse | 11012.3 | 1.000 | 1.000〜1.000 | exact |
| dwt_1005 | Atomic Sparse | 11093.9 | 1.005 | 1.001〜1.010 | exact |
| dwt_1005 | RR Sparse | 11071 | 1.003 | 1.001〜1.006 | exact |
| bcsstk09 | CPU Full | 8990.9 | 1.000 | 1.000〜1.000 | exact |
| bcsstk09 | Atomic Full | 9100.47 | 1.012 | 1.012〜1.012 | exact |
| bcsstk09 | RR Full | 8990.98 | 1.000 | 1.000〜1.000 | exact |
| bcsstk09 | CPU Sparse | 9288.77 | 1.000 | 1.000〜1.000 | exact |
| bcsstk09 | Atomic Sparse | 9301.94 | 1.003 | 1.001〜1.004 | exact |
| bcsstk09 | RR Sparse | 9264.88 | 0.998 | 0.994〜1.004 | exact |
| email | CPU Full | 83613.2 | 1.000 | 1.000〜1.000 | exact |
| email | Atomic Full | 96050.4 | 1.147 | 1.133〜1.178 | exact |
| email | RR Full | 86130 | 1.027 | 1.022〜1.056 | exact |
| email | CPU Sparse | 91936.7 | 1.000 | 1.000〜1.000 | exact |
| email | Atomic Sparse | 92663.6 | 1.011 | 0.995〜1.015 | exact |
| email | RR Sparse | 93618.4 | 1.015 | 1.013〜1.037 | exact |
| 1138_bus | CPU Full | 39901.9 | 1.000 | 1.000〜1.000 | exact |
| 1138_bus | Atomic Full | 44403.9 | 1.113 | 1.105〜1.125 | exact |
| 1138_bus | RR Full | 40192.7 | 1.008 | 1.001〜1.042 | exact |
| 1138_bus | CPU Sparse | 40927.9 | 1.000 | 1.000〜1.000 | exact |
| 1138_bus | Atomic Sparse | 41416.1 | 1.015 | 0.999〜1.016 | exact |
| 1138_bus | RR Sparse | 41006.3 | 1.002 | 0.999〜1.019 | exact |
| polblogs | CPU Full | 88848.4 | 1.000 | 1.000〜1.000 | exact |
| polblogs | Atomic Full | 121771 | 1.353 | 1.333〜1.408 | exact |
| polblogs | RR Full | 97699.8 | 1.100 | 1.003〜1.196 | exact |
| polblogs | CPU Sparse | 102128 | 1.000 | 1.000〜1.000 | exact |
| polblogs | Atomic Sparse | 100905 | 0.993 | 0.984〜1.016 | exact |
| polblogs | RR Sparse | 103578 | 1.023 | 0.985〜1.040 | exact |
| yeast | CPU Full | 329092 | 1.000 | 1.000〜1.000 | exact |
| yeast | Atomic Full | 454563 | 1.382 | 1.350〜1.415 | exact |
| yeast | RR Full | 391661 | 1.190 | 1.080〜1.218 | exact |
| yeast | CPU Sparse | 388580 | 1.000 | 1.000〜1.000 | exact |
| yeast | Atomic Sparse | 394727 | 1.025 | 0.964〜1.036 | exact |
| yeast | RR Sparse | 397181 | 1.022 | 0.994〜1.068 | exact |
| add20 | CPU Full | 283522 | 1.000 | 1.000〜1.000 | exact |
| add20 | Atomic Full | 384619 | 1.355 | 1.290〜1.432 | exact |
| add20 | RR Full | 306643 | 1.082 | 1.051〜1.115 | exact |
| add20 | CPU Sparse | 309296 | 1.000 | 1.000〜1.000 | exact |
| add20 | Atomic Sparse | 312878 | 1.002 | 0.890〜1.012 | exact |
| add20 | RR Sparse | 309089 | 0.988 | 0.901〜1.014 | exact |
| dwt_2680 | CPU Full | 45398.7 | 1.000 | 1.000〜1.000 | exact |
| dwt_2680 | Atomic Full | 46118.1 | 1.016 | 0.829〜1.016 | exact |
| dwt_2680 | RR Full | 45611.1 | 1.004 | 0.818〜1.008 | exact |
| dwt_2680 | CPU Sparse | 57871.4 | 1.000 | 1.000〜1.000 | exact |
| dwt_2680 | Atomic Sparse | 58345.1 | 1.014 | 0.823〜1.367 | exact |
| dwt_2680 | RR Sparse | 53733.2 | 0.996 | 0.817〜1.125 | exact |
| data | CPU Full | 39392.6 | 1.000 | 1.000〜1.000 | exact |
| data | Atomic Full | 39998.1 | 1.016 | 1.014〜1.020 | exact |
| data | RR Full | 39824.2 | 1.010 | 1.005〜1.015 | exact |
| data | CPU Sparse | 40749.1 | 1.000 | 1.000〜1.000 | exact |
| data | Atomic Sparse | 41103.4 | 1.009 | 1.005〜1.021 | exact |
| data | RR Sparse | 40795.3 | 1.002 | 0.996〜1.016 | exact |
| bcsstk15 | CPU Full | 696971 | 1.000 | 1.000〜1.000 | exact |
| bcsstk15 | Atomic Full | 733745 | 1.053 | 1.052〜1.160 | exact |
| bcsstk15 | RR Full | 728638 | 1.045 | 1.026〜1.054 | exact |
| bcsstk15 | CPU Sparse | 712031 | 1.000 | 1.000〜1.000 | exact |
| bcsstk15 | Atomic Sparse | 712531 | 1.003 | 0.984〜1.078 | exact |
| bcsstk15 | RR Sparse | 709782 | 1.002 | 0.981〜1.023 | exact |
| CA-GrQc | CPU Full | 1.05772e+06 | 1.000 | 1.000〜1.000 | exact |
| CA-GrQc | Atomic Full | 1.5367e+06 | 1.453 | 1.310〜1.500 | exact |
| CA-GrQc | RR Full | 1.29834e+06 | 1.227 | 1.170〜1.266 | exact |
| CA-GrQc | CPU Sparse | 1.21194e+06 | 1.000 | 1.000〜1.000 | exact |
| CA-GrQc | Atomic Sparse | 1.23436e+06 | 1.012 | 0.988〜1.051 | exact |
| CA-GrQc | RR Sparse | 1.22356e+06 | 1.022 | 0.929〜1.074 | exact |
| EVA | CPU Full | 955404 | 1.000 | 1.000〜1.000 | exact |
| EVA | Atomic Full | 1.13863e+06 | 1.192 | 1.166〜1.206 | exact |
| EVA | RR Full | 1.07213e+06 | 1.122 | 1.083〜1.190 | exact |
| EVA | CPU Sparse | 1.11126e+06 | 1.000 | 1.000〜1.000 | exact |
| EVA | Atomic Sparse | 1.13238e+06 | 1.013 | 0.980〜1.184 | exact |
| EVA | RR Sparse | 1.13946e+06 | 1.027 | 0.996〜1.146 | exact |
| 3elt | CPU Full | 422898 | 1.000 | 1.000〜1.000 | exact |
| 3elt | Atomic Full | 506190 | 1.197 | 1.179〜2.131 | exact |
| 3elt | RR Full | 426680 | 1.009 | 1.008〜1.018 | exact |
| 3elt | CPU Sparse | 428793 | 1.000 | 1.000〜1.000 | exact |
| 3elt | Atomic Sparse | 430044 | 1.003 | 1.003〜1.004 | exact |
| 3elt | RR Sparse | 429267 | 1.001 | 1.000〜1.002 | exact |
| bcsstk16 | CPU Full | 180322 | 1.000 | 1.000〜1.000 | exact |
| bcsstk16 | Atomic Full | 265019 | 1.470 | 1.446〜1.569 | exact |
| bcsstk16 | RR Full | 172084 | 0.954 | 0.937〜1.031 | exact |
| bcsstk16 | CPU Sparse | 192123 | 1.000 | 1.000〜1.000 | exact |
| bcsstk16 | Atomic Sparse | 208346 | 1.085 | 0.919〜1.098 | exact |
| bcsstk16 | RR Sparse | 194447 | 1.004 | 0.930〜1.184 | exact |
| uk | CPU Full | 127020 | 1.000 | 1.000〜1.000 | exact |
| uk | Atomic Full | 135865 | 1.069 | 1.055〜1.071 | exact |
| uk | RR Full | 128687 | 1.014 | 0.993〜3.022 | exact |
| uk | CPU Sparse | 132401 | 1.000 | 1.000〜1.000 | exact |
| uk | Atomic Sparse | 133955 | 1.012 | 1.001〜1.024 | exact |
| uk | RR Sparse | 134339 | 1.012 | 0.979〜1.016 | exact |
| USpowerGrid | CPU Full | 702102 | 1.000 | 1.000〜1.000 | exact |
| USpowerGrid | Atomic Full | 731516 | 1.042 | 1.041〜1.043 | exact |
| USpowerGrid | RR Full | 712186 | 1.014 | 1.011〜1.023 | exact |
| USpowerGrid | CPU Sparse | 728875 | 1.000 | 1.000〜1.000 | exact |
| USpowerGrid | Atomic Sparse | 730885 | 1.003 | 1.001〜1.006 | exact |
| USpowerGrid | RR Sparse | 730183 | 1.003 | 1.000〜1.008 | exact |
| bcspwr10 | CPU Full | 679074 | 1.000 | 1.000〜1.000 | exact |
| bcspwr10 | Atomic Full | 679995 | 1.001 | 1.001〜1.001 | exact |
| bcspwr10 | RR Full | 679967 | 1.001 | 1.001〜1.002 | exact |
| bcspwr10 | CPU Sparse | 691693 | 1.000 | 1.000〜1.000 | exact |
| bcspwr10 | Atomic Sparse | 693989 | 1.002 | 1.001〜1.004 | exact |
| bcspwr10 | RR Sparse | 692330 | 1.000 | 0.999〜1.001 | exact |
| p2p-Gnutella08 | CPU Full | 3.01624e+06 | 1.000 | 1.000〜1.000 | exact |
| p2p-Gnutella08 | Atomic Full | 5.13671e+06 | 1.705 | 1.560〜1.762 | exact |
| p2p-Gnutella08 | RR Full | 4.38289e+06 | 1.453 | 1.272〜1.558 | exact |
| p2p-Gnutella08 | CPU Sparse | 3.49063e+06 | 1.000 | 1.000〜1.000 | exact |
| p2p-Gnutella08 | Atomic Sparse | 3.53354e+06 | 1.005 | 0.992〜1.016 | exact |
| p2p-Gnutella08 | RR Sparse | 3.68967e+06 | 1.040 | 1.027〜1.088 | exact |
| wiki-Vote | CPU Full | 3.20877e+06 | 1.000 | 1.000〜1.000 | exact |
| wiki-Vote | Atomic Full | 7.3556e+06 | 2.294 | 2.205〜2.411 | exact |
| wiki-Vote | RR Full | 5.35912e+06 | 1.671 | 1.573〜1.896 | exact |
| wiki-Vote | CPU Sparse | 3.83549e+06 | 1.000 | 1.000〜1.000 | exact |
| wiki-Vote | Atomic Sparse | 3.93822e+06 | 1.027 | 1.004〜1.038 | exact |
| wiki-Vote | RR Sparse | 4.0313e+06 | 1.035 | 1.011〜1.096 | exact |
| ca-HepTh | CPU Full | 4.88153e+06 | 1.000 | 1.000〜1.000 | sampled |
| ca-HepTh | Atomic Full | 8.32766e+06 | 1.706 | 1.685〜1.778 | sampled |
| ca-HepTh | RR Full | 6.48401e+06 | 1.328 | 1.271〜1.361 | sampled |
| ca-HepTh | CPU Sparse | 5.85686e+06 | 1.000 | 1.000〜1.000 | sampled |
| ca-HepTh | Atomic Sparse | 5.8831e+06 | 1.008 | 0.988〜1.017 | sampled |
| ca-HepTh | RR Sparse | 6.06835e+06 | 1.038 | 1.010〜1.050 | sampled |
| bcsstk17 | CPU Full | 1.75598e+06 | 1.000 | 1.000〜1.000 | sampled |
| bcsstk17 | Atomic Full | 5.26262e+06 | 2.906 | 2.492〜3.024 | sampled |
| bcsstk17 | RR Full | 2.20878e+06 | 1.255 | 1.137〜1.371 | sampled |
| bcsstk17 | CPU Sparse | 1.80776e+06 | 1.000 | 1.000〜1.000 | sampled |
| bcsstk17 | Atomic Sparse | 2.16012e+06 | 1.015 | 0.923〜1.304 | sampled |
| bcsstk17 | RR Sparse | 1.81141e+06 | 1.001 | 0.774〜1.214 | sampled |
| PGPgiantcompo | CPU Full | 5.82214e+06 | 1.000 | 1.000〜1.000 | sampled |
| PGPgiantcompo | Atomic Full | 9.42291e+06 | 1.618 | 1.480〜1.666 | sampled |
| PGPgiantcompo | RR Full | 7.51092e+06 | 1.290 | 1.265〜1.359 | sampled |
| PGPgiantcompo | CPU Sparse | 6.47662e+06 | 1.000 | 1.000〜1.000 | sampled |
| PGPgiantcompo | Atomic Sparse | 6.54219e+06 | 1.013 | 1.004〜1.028 | sampled |
| PGPgiantcompo | RR Sparse | 6.70339e+06 | 1.029 | 1.013〜1.134 | sampled |
| fe_4elt2 | CPU Full | 2.58818e+06 | 1.000 | 1.000〜1.000 | sampled |
| fe_4elt2 | Atomic Full | 2.68934e+06 | 1.039 | 1.029〜1.039 | sampled |
| fe_4elt2 | RR Full | 2.63216e+06 | 1.017 | 1.002〜1.019 | sampled |
| fe_4elt2 | CPU Sparse | 2.64095e+06 | 1.000 | 1.000〜1.000 | sampled |
| fe_4elt2 | Atomic Sparse | 2.65644e+06 | 1.005 | 0.999〜1.007 | sampled |
| fe_4elt2 | RR Sparse | 2.65371e+06 | 1.005 | 0.998〜1.016 | sampled |
| ca-AstroPh | CPU Sparse | 2.55008e+07 | 1.000 | 1.000〜1.000 | sampled |
| ca-AstroPh | Atomic Sparse | 2.57732e+07 | 1.008 | 0.980〜1.016 | sampled |
| ca-AstroPh | RR Sparse | 2.60098e+07 | 1.007 | 1.001〜1.045 | sampled |
| email-Enron | CPU Sparse | 9.76014e+07 | 1.000 | 1.000〜1.000 | sampled |
| email-Enron | Atomic Sparse | 9.76307e+07 | 1.009 | 0.947〜1.085 | sampled |
| email-Enron | RR Sparse | 1.04894e+08 | 1.075 | 0.951〜1.164 | sampled |
| bcsstk31 | CPU Sparse | 2.09887e+07 | 1.000 | 1.000〜1.000 | sampled |
| bcsstk31 | Atomic Sparse | 2.10847e+07 | 1.005 | 1.000〜1.017 | sampled |
| bcsstk31 | RR Sparse | 2.10573e+07 | 1.001 | 1.001〜1.006 | sampled |
| bcsstk32 | CPU Sparse | 4.5769e+07 | 1.000 | 1.000〜1.000 | sampled |
| bcsstk32 | Atomic Sparse | 4.57704e+07 | 1.003 | 0.981〜1.006 | sampled |
| bcsstk32 | RR Sparse | 4.57649e+07 | 1.000 | 1.000〜1.001 | sampled |
| venkat50 | CPU Sparse | 8.4237e+07 | 1.000 | 1.000〜1.000 | sampled |
| venkat50 | Atomic Sparse | 8.43534e+07 | 1.003 | 1.001〜1.003 | sampled |
| venkat50 | RR Sparse | 8.42421e+07 | 1.000 | 0.999〜1.003 | sampled |
| p2p-Gnutella31 | CPU Sparse | 3.63374e+08 | 1.000 | 1.000〜1.000 | sampled |
| p2p-Gnutella31 | Atomic Sparse | 3.62324e+08 | 0.997 | 0.974〜1.001 | sampled |
| p2p-Gnutella31 | RR Sparse | 3.71005e+08 | 1.032 | 0.968〜1.053 | sampled |
| soc-Epinions1 | CPU Sparse | 5.17769e+08 | 1.000 | 1.000〜1.000 | sampled |
| soc-Epinions1 | Atomic Sparse | 5.20837e+08 | 1.021 | 0.973〜1.053 | sampled |
| soc-Epinions1 | RR Sparse | 5.70635e+08 | 1.100 | 1.038〜1.157 | sampled |
| soc-Slashdot0811 | CPU Sparse | 5.73488e+08 | 1.000 | 1.000〜1.000 | sampled |
| soc-Slashdot0811 | Atomic Sparse | 5.57605e+08 | 0.976 | 0.927〜1.023 | sampled |
| soc-Slashdot0811 | RR Sparse | 5.9508e+08 | 1.037 | 0.863〜1.152 | sampled |
| luxembourg_osm | CPU Sparse | 1.52153e+08 | 1.000 | 1.000〜1.000 | sampled |
| luxembourg_osm | Atomic Sparse | 1.5281e+08 | 1.004 | 1.003〜1.005 | sampled |
| luxembourg_osm | RR Sparse | 1.52515e+08 | 1.002 | 1.000〜1.005 | sampled |
| fe_ocean | CPU Sparse | 3.03085e+08 | 1.000 | 1.000〜1.000 | sampled |
| fe_ocean | Atomic Sparse | 3.04071e+08 | 1.004 | 1.001〜1.005 | sampled |
| fe_ocean | RR Sparse | 3.03529e+08 | 1.001 | 1.001〜1.003 | sampled |
| la2010 | CPU Sparse | 6.56958e+08 | 1.000 | 1.000〜1.000 | sampled |
| la2010 | Atomic Sparse | 6.59858e+08 | 1.003 | 1.001〜1.007 | sampled |
| la2010 | RR Sparse | 6.58417e+08 | 1.001 | 0.999〜1.005 | sampled |
| email-EuAll | CPU Sparse | 4.7999e+09 | 1.000 | 1.000〜1.000 | sampled |
| email-EuAll | Atomic Sparse | 4.76755e+09 | 0.993 | 0.942〜1.001 | sampled |
| email-EuAll | RR Sparse | 5.55755e+09 | 1.155 | 1.024〜1.205 | sampled |
| coAuthorsCiteseer | CPU Sparse | 4.24355e+09 | 1.000 | 1.000〜1.000 | sampled |
| coAuthorsCiteseer | Atomic Sparse | 4.21668e+09 | 1.001 | 0.991〜1.009 | sampled |
| coAuthorsCiteseer | RR Sparse | 4.45184e+09 | 1.069 | 1.015〜1.200 | sampled |
| web-Stanford | CPU Sparse | 5.42823e+09 | 1.000 | 1.000〜1.000 | sampled |
| web-Stanford | Atomic Sparse | 5.24944e+09 | 0.971 | 0.908〜1.044 | sampled |
| web-Stanford | RR Sparse | 5.62251e+09 | 1.040 | 0.902〜1.254 | sampled |
| com-DBLP | CPU Sparse | 8.37946e+09 | 1.000 | 1.000〜1.000 | sampled |
| com-DBLP | Atomic Sparse | 8.4283e+09 | 0.994 | 0.990〜1.011 | sampled |
| com-DBLP | RR Sparse | 9.03616e+09 | 1.078 | 1.040〜1.271 | sampled |
| web-NotreDame | CPU Sparse | 7.31157e+09 | 1.000 | 1.000〜1.000 | sampled |
| web-NotreDame | Atomic Sparse | 7.33565e+09 | 1.004 | 0.972〜1.058 | sampled |
| web-NotreDame | RR Sparse | 7.87543e+09 | 1.047 | 0.985〜1.264 | sampled |
| com-Amazon | CPU Sparse | 7.84221e+09 | 1.000 | 1.000〜1.000 | sampled |
| com-Amazon | Atomic Sparse | 7.74482e+09 | 0.989 | 0.979〜1.022 | sampled |
| com-Amazon | RR Sparse | 7.83085e+09 | 0.990 | 0.960〜1.078 | sampled |
| coPapersCiteseer | CPU Sparse | 1.57314e+10 | 1.000 | 1.000〜1.000 | sampled |
| coPapersCiteseer | Atomic Sparse | 1.5796e+10 | 1.003 | 0.985〜1.022 | sampled |
| coPapersCiteseer | RR Sparse | 1.70792e+10 | 1.059 | 1.006〜1.145 | sampled |

## 考察

- Atomic FullのStress/CPUは幾何平均1.1818。グラフ別にはdwt_758の0.9678〜bcsstk17の2.9056で、平均値だけでは入力ごとの品質差を表せない。
- RR FullのStress/CPUは幾何平均1.0656。グラフ別にはbcsstk16の0.9544〜wiki-Voteの1.6710で、平均値だけでは入力ごとの品質差を表せない。
- Atomic SparseのStress/CPUは幾何平均1.0083。グラフ別にはweb-Stanfordの0.9710〜bcsstk16の1.0855で、平均値だけでは入力ごとの品質差を表せない。
- RR SparseのStress/CPUは幾何平均1.0171。グラフ別にはpolbooksの0.9819〜email-EuAllの1.1552で、平均値だけでは入力ごとの品質差を表せない。
- RR FullはT1024固定の結果。タイル評価で確認したTによる更新順と品質の変化も考慮し、大きいTの速度と品質のトレードオフとして読む。
- FullとSparseは学習対象・重みが異なる。Full/Sparse比較図では分母を共通CPU Fullへ統一し、異なる分母の比を直接比較しない。
- 同seed3回のStress・座標の一致/変動は生データから追跡できるが、独立な初期配置は5つ。固定15反復の結果であり、収束後や最適解の品質を示すものではない。

## 規模別の補足集計

全対象の要約には非常に小さいグラフも含まれる。n≥1000の区切りは結果を読むための事後分析であり、事前に固定した合否基準ではない。グラフ構造も異なるため純粋なサイズ効果とは呼ばない。

| 方式 | 集合 | 件数 | SGD速度比・幾何平均 | 総計算速度比・幾何平均 | Stress/CPU・幾何平均 |
|---|---|---:|---:|---:|---:|
| RR Full | 全対象 | 38 | 1.383 | 0.630 | 1.0656 |
| RR Full | n≥1000（事後の補足集計） | 23 | 11.338 | 3.320 | 1.1110 |
| RR Sparse | 全対象 | 56 | 1.871 | 1.021 | 1.0171 |
| RR Sparse | n≥1000（事後の補足集計） | 41 | 8.095 | 2.609 | 1.0229 |
| RR Sparse | Sparse専用18件 | 18 | 35.399 | 4.159 | 1.0377 |

RR Fullの全38件では総計算時間の幾何平均速度比が1未満であり、全入力で処理全体が速くなったとは結論しない。RR Sparseも全56件での総計算速度比は約1.02倍で、SGD単体の速度比や大きいグラフ群の結果とは明確に区別する。

## 同seed内3回の再現性

範囲は(最大Stress−最小Stress)/中央値で、信頼区間ではない。全runを残したまま集計し、変動の大きい実行を除外していない。

| 方式 | graph×seed群 | Stressが3回で変動した群 | 最大の相対範囲 | 該当グラフ・seed |
|---|---:|---:|---:|---|
| CPU Full | 190 | 0 | 0.00% | lesmis・0 |
| Atomic Full | 190 | 190 | 157.34% | 3elt・0 |
| RR Full | 190 | 0 | 0.00% | dolphins・0 |
| CPU Sparse | 280 | 0 | 0.00% | email・0 |
| Atomic Sparse | 280 | 280 | 117.29% | dwt_198・4 |
| RR Sparse | 280 | 0 | 0.00% | EVA・0 |

CPU・RRでは今回の全graph×seed群でStressが3回一致した。Atomicは全群で変動を観測し、一部で非常に大きい。同じ初期配置・前処理fingerprintと更新完了数が一致していても品質の再現性は保証されない。GPU更新順の非決定性などが候補だが、今回の出力だけで原因を確定しない。seed内中央値へまとめることでこの変動が見えにくくなるため、範囲表と図も併記する。座標自体の完全な同一性まで今回の照合で主張しているわけではない。

## 解釈上の制約

- 同じ機材・固定バイナリ・逐次実行で測ったが、測定中の全バックグラウンド活動や温度を連続記録した対照実験ではない。大きい時間の変動をすべて方式の差に帰属しない。ディスク容量不足による中断・キャッシュ整理・再ウォームアップの履歴をlogsに保存し、ウォームアップは集計から除外した。
- メモリ記録のswap_in_bytes/swap_out_bytesはpsutil由来のOSカウンターであり、ファイルページングを含む可能性がある。厳密なスワップI/O量や「スワップなし」の証明として使わない。peak RSSは共有GPUメモリの厳密な占有量ではない。
- GPU passタイムスタンプは確保エラー回避のため全GPU方式で無効。GPU計算pass時間は未測定で、SGD実経過時間との混同を避ける。
- 推定Stressは固定256始点。始点ごとの寄与・推定誤差は今回の出力に保存されておらず、小差を統計的に有意な優劣と断定しない。
- 乱数やpivotを実験のために改変していない。Full/Sparse間は初期配置が共通ではないため、その比較は各アルゴリズムの実装全体の比較として扱う。同族内の初期配置とSparse前処理fingerprintの照合結果はvalidation/comparison-completion.jsonに保存した。
- 移管6件は今回の失敗・欠測ではない。AS365・great-britain_osm・asia_osmは前処理メモリ下限を理由に移管、auto・NACA0015・Flan_1565はユーザーの範囲指定で移管した。元の大規模8件と合わせ14件は今回未実行で、GPUでの実行成功やCPUの一般的不可能性を実証していない。

## 保存先

- `tables/comparison-runs.csv`：全4,230実行の生データと対応比。
- `tables/comparison-paired-ratios.csv`：run単位の対応CPU比。
- `tables/comparison-summary.csv` / `comparison-hierarchical.json`：282設定の階層集計と5seedの値。
- `tables/comparison-cross-graph.json`：グラフに等しい重みを与えた記述的要約。
- `validation/comparison-completion.json`：件数・条件・保存の照合。
- `raw/comparison/`：各実行のJSON・コマンド・ログ・圧縮座標。
- `config/`：最終の対象、実行順、移管理由、変更前の予定。

同族内照合では、全graph・seedでCPU/Atomic/RRの初期配置fingerprintが一致し、Sparseの前処理fingerprintも一致した。FullとSparse間の一致を意味しない。
