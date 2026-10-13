# v2 独立最終監査

監査日：2026-10-13（Asia/Tokyo）。正式測定の再実行はせず、raw・集計CSV・保存座標・ソースsnapshot・バイナリを読み取り専用で照合した。

- 正式記録は **2260件、成功2260件、失敗0件**。内訳は主比較1500件、大規模Sparse120件、shared/global480件、WG変更160件。
- `paired-ratios.csv` の **2260件**について、rawから `CPU反復時間÷手法反復時間` と `手法stress÷同familyのCPU stress` を再計算し、すべて一致した。family・seedの対応にも不一致はなかった。集計の速度比中央値とstress比平均も一致した。
- shared/global **240組**とWG変更 **120組**について、入力・初期配置・seed・T・shuffle設定・評価条件を照合し、保存座標を直接読み直した。全組の座標が完全一致した。前者はWも一致、後者はcoordinate memoryも一致した。
- 4実験のhashに登録されたソースsnapshotに不一致はなく、全実行の `integrity.before` と `integrity.after` が保存済み実行版に一致した。**6種類のSGDバイナリの現在のSHA-256も測定時と一致**した。
- 測定後の汎用JSON互換性修正は `post-run-change.json` と `post-run-compat.patch` で分離され、patchのSHA-256も一致した。実行時schemaの補助保存は旧SHA-256 `c77024db57e6d69c8b89466280d4a9a12aaa9405b9fba2277f9ed884b2dd1dad` を保持している。hashに登録された測定ソースは書き換えられていない。

主な数値も報告と一致した。Full RRは10グラフ中9件でCPUより速く、速度比0.46–10.12倍、平均stress比0.986–1.138。Sparse RRは14グラフ中6件で速く、速度比0.05–30.15倍、平均stress比0.999–1.244。web-NotreDameのstress比1.244は品質低下として併記する必要がある。

解釈の範囲：主な速度比は同期・shuffle・投入を含むhost反復実時間。Full RRの890件にはGPU timestamp区間も保存されているが、最初から最後のdispatchまでの待ちを含み、kernel単独時間ではない。cache missと実測occupancyは取得していない。shared/globalの速度差は概ね小さく、sharedの普遍的優位性は示していない。

比較は固定15反復で、CPUはf64、GPUはf32。Full/Sparseをまたぐ対応比は作っていない。大規模4グラフは同じsampling条件のfull-stress推定値を比較しており、全ペアのexact評価ではない。1 seedにつき時間測定1回なので機器の時間変動とseed差を分離できず、T変更も更新順・dispatch数・並列WG数を同時に変える。これらの結果だけで同品質への到達時間や高速化原因を断定しない。

監査時に見つかった報告の文言「device時間は未計測」は、Full RRのtimestamp保存状況と一致しなかったため、報告担当へ修正を依頼した。数値・座標・測定版の整合性を損なう問題は見つからなかった。
