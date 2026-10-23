# 初回計測の中断：テキスト書出しの負荷

初回は662条件を記録した時点で中断し、全ログを`prebuffer-aborted/`へ移した。最終v3の主集計には混ぜず、修正後の同一版で全条件を最初から測る。元のrecordが指す`v3/raw`・`v3/warmups`は、この履歴では`v3/prebuffer-aborted/raw`・`warmups`へ読み替える。

kron_g500-logn17・Sparse CPU・seed0ではnative process wall 60.76秒に対し、CLI内の入力+アルゴリズム16.39秒、stress評価3.67秒だった。差分約40.70秒には初期/最終座標・辺・頂点map保存やhash/git取得等が含まれる。初期/最終ファイルだけで各62MBあり、全6実装の9保存関数がbufferなしの細かな書込みを繰り返していた。

commit `a5910d9`で1MiBのBufWriterと明示flushへ変更。数値・行順・書式・更新アルゴリズム・shader・学習率・乱数処理は変更しない。書式処理を変更していないことを差分で照合した。

修正後の同じCPU条件は全体22.10秒、SGD7.96秒、stress 1,041,858,608.7586834。修正前とCPU stressが一致した。小規模dwt_1005では6手法すべて正常完了し、決定的なCPU2種・RR2種の最終座標が修正前と完全一致した。Atomicは実行順が変わり得るため座標一致を要求せず、予定更新全件完了を確認した。

座標と辺の重複保存で容量も増えるため、正式実行ではnative process終了後にテキストをgzip level1で可逆圧縮する。CRCと解凍SHA256一致を確認し、対応manifestを保存した後に元ファイルを置換する。圧縮時間archive_secondsは測定外で、SGD/cold total/native process wallには含めない。保存済みの座標データ自体は変えない。
