# v3 sampled stress感度の確認

最終有効cohortから、basic seed0/P200でSparse CPU・Atomic・RRがすべて成功し、入力・初期配置が対応するLux / web-Stanford / 144の保存済み最終座標を再評価した。優先候補delaunay20はCPU成功がなく、事前の名前順fallbackで144を採用した。SGDの再実行は行っていない。

同じ9配置（3 graph×3手法）を、同じSplitMix64 seed0のnested prefixで64 / 256 / 1024始点から評価し、27結果がすべて成功した。推定式はnativeと同じn/(2K)によるfull stress推定。1024始点も厳密値ではない。

256対1024の絶対stress最大相対差は**2.668941%**、CPUに対する品質比の最大相対差は**3.908351%**だった。web-Stanford Atomic seed0の品質比は次のように変わり、+5%基準の判定を跨いだ。

| 始点数K | GPU stress / 同KのCPU stress | q<=1.05 |
|---|---:|---|
| 64 | 1.008512489 | 基準内 |
| 256 | 1.047231946 | 基準内 |
| 1024 | 1.089826174 | 基準外 |

したがって、RRの21/37 graph等の達成数は固定256始点という評価定義での記述であり、厳密stressや別の始点数でも同じ分類になる保証ではない。上の例はAtomic seed0であり、graph別5seed中央値による達成数を再分類した結果でもない。3 graphだけの比較を、全graphの誤差保証や信頼区間へ一般化しない。

native照合では元の評価定義がsampled・K=256・stress seed0であることを確認し、9配置すべてが再評価値との差0.1%未満だった。最大相対差は2.056×10⁻¹³%で、保存座標の丸め等を許す検査において整合した。bit単位一致を要求したものではない。

警告修正後の最終再計算はwarning0、旧計算との最大相対stress差3.945×10⁻¹⁶だった。native計測値・保存配置は変更していない。

根拠：[27結果CSV](../tables/sampling-sensitivity.csv)・[感度とnative照合JSON](../tables/sampling-sensitivity.json)・[警告修正後の再確認](sampling-einsum-recheck.json)。
