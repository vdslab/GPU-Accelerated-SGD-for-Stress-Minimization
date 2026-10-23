# 本計測前の試運転と上限

全体wall 60秒・RSS 8 GiBの15試行：成功11、全体時間切れ2、メモリ制限2。FullのUSpowerGrid/fe_4elt2とSparseのluxembourg_osmは全3手法が成功した。

| Graph | Method | Status | Process wall s | Observed peak RSS GiB | SGD s (success only) |
|---|---|---|---:|---:|---:|
| USpowerGrid | sgd | success | 3.84 | 1.02 | 3.111 |
| USpowerGrid | atomic_sgd | success | 2.84 | 1.05 | 1.935 |
| USpowerGrid | rr_sgd | success | 1.30 | 0.98 | 0.384 |
| fe_4elt2 | sgd | success | 19.75 | 3.52 | 17.327 |
| fe_4elt2 | atomic_sgd | success | 7.84 | 4.06 | 5.194 |
| fe_4elt2 | rr_sgd | success | 4.06 | 4.02 | 1.304 |
| luxembourg_osm | sparse_sgd | success | 15.29 | 2.57 | 6.774 |
| luxembourg_osm | atomic_sparse_sgd | success | 15.76 | 2.56 | 7.031 |
| luxembourg_osm | rr_sparse_sgd | success | 9.37 | 2.56 | 0.268 |
| web-Stanford | sparse_sgd | success | 52.97 | 4.00 | 16.121 |
| web-Stanford | atomic_sparse_sgd | success | 55.32 | 4.94 | 18.125 |
| web-Stanford | rr_sparse_sgd | timeout | 60.02 | 4.28 | — |
| delaunay_n20 | sparse_sgd | timeout | 60.19 | 5.44 | — |
| delaunay_n20 | atomic_sparse_sgd | memory_limit | 12.50 | 8.12 | — |
| delaunay_n20 | rr_sparse_sgd | memory_limit | 12.54 | 8.10 | — |

失敗行のSGD秒は欠測とし、成功結果以外の速度比較には使わない。

web-Stanford RRはSGD15反復を23.85秒で終えた後、全体60秒制限に到達。120秒の追加試運転では全体63.54秒・RSS約6.04 GiBで正常終了した。この結果を見て本番の全体上限を120秒へ倍増し、RSS 8 GiBを維持した。正式測定の上限と失敗時方針はconfig/limits.jsonで固定する。

追加試運転のdelaunay_n20 CPUでは、制限超過に伴いSIGTERMを送った後のkillpg(pgid,0)がEPERMを返して監視が例外終了した。対象の生存プロセスは残っていなかった。実存プロセスの独立走査へ変更し、確認不能・残存を無視しない処理に修正した。この監視事故は正式実験に数えず、pilot-monitor-incident/へ証拠を保存した。トリガーとなった制限値とピークRSSは永続化前で確定できない。

修正後の同条件（delaunay_n20 CPU・seed0）再試行は全体120.07秒、観測ピークRSS約7.995 GiBでtimeoutを正常に記録し、対象プロセスを停止できた。正式版の停止処理はこの再試行で確認済み。
