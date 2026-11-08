"""Collect existing figures and measured tables into one Obsidian review note."""
from pathlib import Path
import hashlib
import json
import shutil

from run_v5 import ROOT, VAULT, now, save


def main():
    base=ROOT/'experiment_results/v5'
    manuscript=Path('/Users/kose/master/vis-paper-2027-kose')
    assets=VAULT/'v5_図表/論文確認'
    assets.mkdir(parents=True,exist_ok=True)
    figures=json.loads((base/'paper/manifest.json').read_text())['figures']
    hierarchy=json.loads((base/'tables/comparison-hierarchical.json').read_text())
    costs=json.loads((base/'execution-cost/summary.json').read_text())
    note=VAULT/'論文図表_確認用.md'
    body='# 論文図表・確認用\n\n'
    body+='このノートを上から見れば、既存の手法図・実験チャート・比較表を一通り確認できる。掲載順・採用は検討用で、論文本文へ反映した状態ではない。今回は既存図の集約と数値表の掲載のみで、新規作図・再測定は行っていない。\n\n'
    body+='## 先に確認する点\n\n- **主図は平均1反復の実行時間を比較する図。現時点では未作図。** 下に数値表を置き、既存の速度比図とは区別した。\n- 平均1反復時間＝15反復のSGD経過時間÷15。GPUの純計算pass時間とは別で、この値は既存結果から計算できる。\n- GPU準備・読み戻し・初期転送を含める比較は別表。準備等は1実行につき1回足し、15倍しない。\n- 現稿のFull/Sparse手法図には旧分割の記述がある。今回の分割なし条件に合わせる修正が必要。\n\n'
    body+='## 図表の一覧\n\n| 確認順 | 内容 | 現在の状態 |\n|---|---|---|\n| 1 | 平均1反復時間：CPU / Atomic / RoundRobin | 数値表あり・主図は未作図 |\n| 2 | GPU準備・転送・読み戻し込みの比較 | 全グラフの計算済み。ここにも要約と全表を掲載 |\n| 3 | Atomic / RoundRobin Full / Sparseの手法図 | 現稿の図。Full/Sparseは旧分割の修正対象 |\n| 4 | タイルサイズと時間・Stress | 既存図あり |\n| 5 | 最終Stressと配置比較 | 既存図あり |\n| 6 | 大規模の完了可否・pivot数・描画例 | 既存図あり |\n| 7 | 速度比・再現性・時間配分 | 補足図あり |\n\n'
    body+='## 1．主結果：平均1反復の実行時間\n\n**主図は未作図。** 既存の速度比図を、この絶対時間の図の代わりにしない。以下は同seed3回中央値→5seed中央値のSGD15反復時間を15で割った値。CPU・Atomic・RoundRobinを同じグラフ・同族で比較する。単位はms/iteration。\n\n'
    for family,methods in [('Full',['sgd','atomic_sgd','rr_sgd']),('Sparse',['sparse_sgd','atomic_sparse_sgd','rr_sparse_sgd'])]:
        selected=[r for r in hierarchy if r['method'] in methods]
        idx={(r['graph'],r['method']):r for r in selected}
        body+='### '+family+'：全グラフの平均1反復時間\n\n'
        body+='| グラフ | CPU ms/反復 | Atomic ms/反復 | RoundRobin ms/反復 |\n|---|---:|---:|---:|\n'
        for r in sorted((r for r in selected if r['method']==methods[0]),key=lambda r:(r['nodes'],r['graph'])):
            body+='| '+r['graph']+' | '+' | '.join(f'{idx[(r["graph"],m)]["iteration_time_ms"]["median"]/15:.4f}' for m in methods)+' |\n'
        body+='\n'
    body+='## 2．GPU準備・転送・読み戻し込み\n\nA＝GPU準備＋SGD＋読み戻し。B＝A＋初期転送。CPU側はCPU SGD。GPU固有のスケジュール/データ構築時間はA/Bには入っていない。共通前処理・描画が実測で同じ時間だった、と仮定した表ではない。\n\n'
    cost_text=(VAULT/'6_GPU準備と読み戻し込みの比較.md').read_text()
    # Inline the actual tables instead of requiring navigation to another note.
    section=cost_text.split('## 通常比較の要約\n',1)[1].split('## 全結果ファイル',1)[0]
    section=section.replace('## Full：','### Full：').replace('## Sparse：','### Sparse：').replace('## luxembourg_osmの内訳','### luxembourg_osmの内訳')
    body+=section+'\n'
    copied=[]
    body+='## 3．手法の説明図（現稿）\n\nこの3図は実験チャートではなく、現稿の手法説明図。Full図末尾のsegment順、Sparse図末尾のO1→B1→…→O4→B4は今回の分割なし条件と一致していない。確認用にそのまま掲載し、修正済みとは扱わない。\n\n'
    for title,source,target,status in [
        ('Atomic', 'fig01_atomic.pdf','method-atomic.pdf','現稿の説明図。計測された待ち時間を表す図ではない。'),
        ('RoundRobin Full SGD','fig02_rr_full.png','method-full-current-manuscript.png','修正対象：旧segment分割の記述が残っている。'),
        ('RoundRobin Sparse SGD','fig03_rr_sparse.png','method-sparse-current-manuscript.png','修正対象：旧4分割の交互実行が描かれている。')]:
        src=manuscript/'figures'/source;dest=assets/target;shutil.copy2(src,dest)
        assert src.read_bytes()==dest.read_bytes()
        copied.append(dict(source=str(src),copy=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
        body+=f'### {title}\n\n{status}\n\n![[v5_図表/論文確認/{target}]]\n\n'
    index={f['name']:f for f in figures}
    groups=[('4．タイルサイズ',['f01-tile']),
            ('5．最終品質と配置比較',['f03-quality','f08-layout-comparison']),
            ('6．大規模実験',['f06-feasibility','f07-large-pivots','f09-large-layouts']),
            ('7．補足：速度比・再現性・時間配分',['f02-runtime','f04-repeatability','f05-bottleneck'])]
    for title,names in groups:
        body+='## '+title+'\n\n'
        for name in names:
            f=index[name];image=VAULT/'v5_図表/paper'/(name+'.png');assert image.is_file()
            body+='### '+f['title']+'\n\n'
            if name=='f02-runtime':body+='**これは速度比と総計算の補足図。平均1反復の絶対時間を比較する主図ではない。**\n\n'
            body+=f['caption']+f'\n\n![[v5_図表/paper/{name}.png]]\n\n'
    body+='## 8．今回の結果と混ぜない旧図\n\n現稿に残る収束曲線・time-to-quality・16分割のBA比較・旧40pivot道路実験の図は、今回の条件の結果ではないため、この採用候補の中へ混ぜていない。旧データとファイル自体は保持している。新しい実験では途中収束を測っていない。\n\n'
    body+='## 確認メモ\n\n以下の欄は確認時のメモ用。\n\n- 採用する図：\n- 修正したい点：\n- キャプションの変更：\n- 論文中での掲載順：\n'
    content='---\ntags: [area/大学院研究, 論文, 図表]\nupdated: '+now()[:10]+'\nstatus: 確認用\n---\n\n[[0_概要]]\n\n'+body
    assert not note.exists(), 'Preserve existing review comments; do not overwrite an existing review note'
    note.write_text(content)
    embedded=[]
    import re
    for name in re.findall(r'!\[\[([^\]]+)\]\]',content):
        p=VAULT/name;assert p.is_file(),p;embedded.append(str(p))
    assert len(embedded)==12
    assert len([r for r in hierarchy if r['method']=='sgd'])==38
    assert len([r for r in hierarchy if r['method']=='sparse_sgd'])==56
    save(base/'paper/review-note.json',dict(at=now(),note=str(note),embedded_figures=12,charts=9,manuscript_method_figures=3,
                                         copied_method_figures=copied,new_charts_created=0,new_measurements=0,
                                         iteration_main_chart='not yet plotted; all 94 graph timing tables included',all_embed_paths_verified=True))
    overview=VAULT/'0_概要.md';s=overview.read_text();link='[[論文図表_確認用]]'
    if link not in s:overview.write_text(s.replace('# 実験概要\n','# 実験概要\n\n**論文図表を1ファイルで確認：'+link+'**\n',1))
    print(str(note))


if __name__=='__main__':main()
