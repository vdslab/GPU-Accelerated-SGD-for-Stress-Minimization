# 大規模グラフの実行確認

これはGPU描画化より前の実行確認です。現在の描画方式と全辺での実測は [gpu-rendering.md](gpu-rendering.md) を参照してください。

2026-11-09。Google Chrome 156、実WebGPU、seed 0、epsilon 0.1、4096 × 4096。1回ずつの動作確認で、統計的な性能比較ではありません。

| グラフ                                |   使用頂点 | 使用辺（無向・重複除去後） | pivots | 反復 | stress評価           |
| ------------------------------------- | ---------: | -------------------------: | -----: | ---: | -------------------- |
| web-Stanford                          |    255,265 |                  1,941,926 |    200 |   15 | 固定16始点からの推定 |
| germany_osm                           | 11,548,845 |                 12,369,181 |      5 |   15 | 固定2始点からの推定  |
| germany_osm（GPUバッファ上限128 MiB） | 11,548,845 |                 12,369,181 |      1 |    3 | 固定2始点からの推定  |

web-Stanfordは入力281,903頂点から最大連結成分を抽出。germany_osmは全頂点・全辺を保持しました。GPUの容量に合わせてpivot数を選んでおり、germany_osmの200 pivots対応を確認したという意味ではありません。

Before・Afterの非白色画素、拡大画面のズームと全体表示、4096 × 4096 PNG保存を確認。128 MiB制限のケースでは、二側matchingを複数のGPUバッファへ分割した実行も確認しました。

SGDとstress評価には、概要表示で省略された辺も含むグラフ全体を使用します。50万辺を超える概要表示は最大10万辺を規則的に抽出し、表示数と全辺数を併記。「辺：すべて」で全辺描画を選択できます。100万頂点を超える描画は全頂点の寄与を画素単位で集約します。

stressは始点サンプリングによる推定です。特に2始点のgermany_osmの値を、exact stressや別設定の値と無条件に比較しないでください。反復時間と、非同期描画の完了待ちも分けて記録します。

## 再現

`web/rr-lab` を作業ディレクトリとして実行します。入力データはGit管理対象外です。

```sh
PIVOTS=200 ITERATIONS=15 npm run test:large
GRAPH_FILE=../../data/germany_osm.mtx ITERATIONS=15 npm run test:large
GPU_BUFFER_MB=128 GRAPH_FILE=../../data/germany_osm.mtx ITERATIONS=3 npm run test:large
```

`RR_TEST_URL=https://rr.skmtech.jp` で公開サイトを対象に、同じローカルファイルをブラウザへ読み込んで確認できます。ファイルはサイトへアップロードされません。

## 入力データ

- [SNAP/web-Stanford](https://sparse.tamu.edu/SNAP/web-Stanford): 30,579,901 bytes。SHA-256 `a514e9693666e9d063b3f0b3535f45706f31e524239194750785af737f8ad090`
- [DIMACS10/germany_osm](https://sparse.tamu.edu/DIMACS10/germany_osm): 198,530,725 bytes（約189 MiB）。SHA-256 `a8de087a1262601537a77ca8acc0ce6d3f790f07621f0cef605a8c362303924c`

ダウンロード元は既存の `data/germany_osm.source.json` と同じSuiteSparse配布物で、ハッシュ一致を確認しています。
