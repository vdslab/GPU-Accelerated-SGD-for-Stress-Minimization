# GPU Visualizer

Sparse SGD/RR-SGDが出力した座標TXTをWGPUでPNGへ描画します。入力を省略した従来の対話実行も利用できます。

既定の出力サイズは **8192×8192**、自動ノード半径の上限は **1px** です。正方形の解像度は `-2`、`-4`、`-8`、`-16` の短縮オプションで選べます。任意サイズと半径は `--size WxH` と `--node-radius PX` で上書きできます。16K出力は数GiBの一時メモリを使用します。

```sh
cargo rr -8
cargo rr -16
```

`cargo rr`は、このcrateに設定した`cargo run --release --`の短縮形です。入力パスと出力パスを省略すると、起動後に入力ファイル名を指定でき、出力先は入力ファイルと同じディレクトリへ自動設定されます。パスをコマンドで渡す場合は `cargo rr ../output/result.txt result-4k.png -4` のように指定します。

## 大規模グラフ

通常の `--stress auto` は8,000頂点以下で厳密ストレスを計算し、それを超えると64始点の標本化推定へ切り替えます。`Approx stress` は厳密値ではありません。同じ入力、標本数、seedなら再現できます。

```sh
cargo run -- result.txt result.png --stress sampled --stress-samples 128 --stress-seed 0
cargo run -- result.txt result.png --stress off
cargo run -- result.txt result.png --stress exact   # 大規模入力では非常に高コスト
```

辺は共有頂点座標と `u32` index bufferで既定では全件描画します。ノード半径は頂点密度から自動決定されます。必要な場合だけ上書き・辺間引きを指定できます。

```sh
cargo run -- result.txt result.png --node-radius 1.0
cargo run -- result.txt result.png --max-edges 500000 --stress-seed 0
```

## 25万頂点級の受入確認

巨大fixtureはリポジトリへ保存せず、次のexampleで生成します。既定値は250,000頂点・1,941,926辺です。

```sh
cargo run --release --example generate_large_fixture -- /tmp/gpu-visualizer-250k.txt
cargo run --release -- /tmp/gpu-visualizer-250k.txt /tmp/gpu-visualizer-250k.png
```

出力には読込、ストレス、GPU準備、GPU描画/readback、PNG保存の時間、使用したノード半径、描画辺数が表示されます。

2026-07-13にApple M4 Pro（Metal）で上記fixtureを2048×2048へ全辺描画した結果は、読込131.6ms、64始点の近似ストレス573.6ms、GPU準備7.6ms、GPU描画/readback 167.0ms、PNG保存11.5ms、合計911.7msでした。使用した自動ノード半径は2.05pxです。これは性能保証値ではなく、受入確認時の測定値です。
