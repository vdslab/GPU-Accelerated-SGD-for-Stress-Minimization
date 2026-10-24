# sampling二乗和のRuntimeWarning調査

`check_v3_sampling.py:115` の `np.dot(residual, residual)` が divide by zero / overflow / invalid の3警告を出した。実データを使わない定数vectorでも同じ警告を再現し、NumPyを経由しないAccelerate `cblas_ddot` が不要な浮動小数点例外flagを残すことまで確認した。今回の出力に非有限値やstress計算の誤りを示す証拠はなかった。

## 実出力の確認

- samplingは3グラフ×3手法×3起点数の27行で完了し、JSON全体を再帰走査した非有限floatは0件。CSVにもNaN/Inf表記はない。
- 初期のdot版で保存最終座標を256起点で再評価した全9件はnative元stressと一致し、最大絶対差は **2.055666×10^−13%**（相対値では約2.06×10^−15）。0.1%の調査flagは全件未到達。
- 読込関数は全座標数と全成分の有限性を検査するため、成功した各レイアウトはこの検査を通過している。独立確認したluxembourg_osm CPU seed 0の114,599座標も全成分有限だった。
- 同lux座標の全点対のEuclidean距離上界は1,788.224。正の非重み付きgraph距離は1以上なので、正規化残差絶対値の上界は1,787.224、1起点の二乗和の上界は約3.66×10^11。float64のoverflow規模には達しない。

## 実データなしの再現

環境はPython 3.11.0 / NumPy 2.3.0 / macOS 15.7.2 / arm64。`np.show_config()` と `otool -L` でAccelerate BLASへのリンクを確認した。

```python
x = np.ones(1024, dtype=np.float64)
np.dot(x, x)                 # 1024.0、3種類のRuntimeWarning
np.einsum('i,i->', x, x)     # 1024.0、警告なし
np.sum(np.square(x))        # 1024.0、警告なし
```

定数1の二乗和1024は正確に表現でき、除算もoverflowも必要としない。NumPyを通さず `ctypes` でAccelerateの `cblas_ddot` を直接呼び、SDKのARM64 `fenv.h` 定義に従って例外flagを先にclearした場合も、返り値1024.0に対しflagが0→`0x9f`になった（invalid・divide by zero・overflow・underflow・inexact等）。長さ64/512/769ではflag 0、1023/1024/114599では再現した。これは調べた長さの観測であり、境界の一般仕様を主張しない。

長さ114599の`linspace(-1,1,n)`でもdotは3警告を出し、結果38,200.33333915078は`math.fsum(x*x)`と完全一致した。threadpool制限1/4でも変わらず、threadpoolctlはAccelerate poolを列挙しなかった。

## 保存座標を用いた小テスト

同lux座標と合成の正整数距離1〜127を使って、source_termsと同じ残差処理を作った（実graph距離・実stressの再計算ではない）。残差は全成分有限、範囲−0.989283〜1,052.679568。dot / einsum / square+sum / math.fsumの二乗和はすべて271,066,110.87437403で一致し、警告はdotだけだった。

## 判定と修正

警告の発生源は、今回の環境のAccelerate ddotが返す値と浮動小数点例外状態の不整合まで切り分けられた。Accelerate内部のどの命令がflagを立てるかは未調査。今回の全27stressは有限で、独立native照合9件に数値誤りの兆候はない。ただし、有限出力だけで任意入力の正確性を保証することはしない。

root承認後、入力形状・座標・graph距離（+infは除外用として許可）・Euclidean距離・残差・二乗和を検査し、同一数式の二乗和を非BLASの `np.einsum('i,i->', residual, residual, optimize=False)` へ変更した。0のself距離と+infの到達不能距離の寄与は従来どおり0。NaN・負距離・非有限の中間値や結果は明示的に失敗させ、実際の演算例外もraiseする。警告をignoreする処理は加えていない。

初期切り分けの機械可読の値と再現結果は[根拠JSON](sampling-dot-warning-evidence.json)。この切り分けではnative再実行・コンパイル・MTX読込・全stress再計算・作図を行わず、小さいvectorと保存座標1本で調べた。修正sourceはその後 `e92714c` で確定し、下記の保存座標サンプリング再解析を完了した。

## 修正後の小規模検証

`python3 -m unittest experiments.tests.test_v3_sampling -v` の8テストが成功。独立な3点既知値、0/+inf距離、長さ1024のvectorとmath.hypot/math.fsum対照、実保存座標の8点prefix、NaN/Inf座標・負/NaN距離・不正index/shape・中間演算/二乗和のoverflow拒否を確認した。長さ1024テストではnp.dotへの呼出しを禁止し、RuntimeWarningもエラーとして扱った。

実lux保存座標全114,599点を使った1起点・合成整数距離の小テストでも、修正後のsource_termsは271,066,110.87437403、独立なmath.hypot/math.fsumも同値で、RuntimeWarningは0件。これは実graphのstress再計算ではない。

## 全27行の再解析による最終検証

旧解析を `analysis/sampling-before-einsum/` へ保存した上で、修正後のsamplerによる3グラフ×3手法×3起点数の全27行の再解析が正常終了した。全stressは有限で、修正後ログのRuntimeWarningは**0件**。旧dot版と修正einsum版のstressを27行すべて照合した最大相対差は **3.9445607232258693×10^−16（4×10^−16以下）**だった。これは相対値であり百分率ではない。

[再解析比較の証拠JSON](sampling-einsum-recheck.json)に旧・修正後resultsのSHA256、修正scriptのSHA256、commit `e92714c`、比較行数・最大差・警告件数を保存した。保存済み最終座標のoffline再評価のみで、native SGDの再実行や正式計測recordの変更はない。

ctypesは既存libm/Accelerateを読み込んで計算しただけ。fenvのclearは短時間の診断用Python子process内だけで、環境変数・インストール済みライブラリ・OS設定・他processには変更を加えていない。
