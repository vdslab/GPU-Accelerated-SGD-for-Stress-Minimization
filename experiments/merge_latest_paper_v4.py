#!/usr/bin/env python3
"""Apply measured v4 Sparse evidence to the latest concurrent manuscript revision."""
from pathlib import Path
import csv,re,json
BASE=Path(__file__).resolve().parents[1]/'experiment_results/v4'
def section(text,heading,level):
 start=text.index(heading);match=re.search(r'^#{1,'+str(level)+r'} ',text[start+len(heading):],re.M);end=start+len(heading)+match.start() if match else len(text);return text[start:end]
def merge(base=BASE):
 source=base/'reference/paper-latest-before-v4/論文_日本語草稿.md'
 if not source.exists():return
 fresh=source.read_text();out=base/'report/論文_日本語草稿_v4.md';generated=out.read_text();text=fresh
 def replace_section(heading,level,new):
  nonlocal text
  old=section(text,heading,level);text=text.replace(old,new.rstrip()+'\n\n',1)
 # Preserve the new six-part method structure and implementation section.
 sparse=section(generated,'### 3.4 RR-Sparse SGD',3).replace('### 3.4 RR-Sparse SGD','### 3.5 RR-Sparse SGD',1)
 sparse='\n\n'.join(p for p in sparse.split('\n\n') if not p.startswith('旧v3で片側相に残っていた'))
 replace_section('### 3.5 RR-Sparse SGD',3,sparse)
 experiments=section(generated,'## Experiments',2)
 old_full=section(experiments,'### 4.1 タイルサイズの影響と共通設定',3);new_full=section(fresh,'### 4.1 タイルサイズの影響と共通設定',3);experiments=experiments.replace(old_full,new_full,1)
 marker='**実施環境と共通条件。**';implementation='**評価対象となる実装。** RR-Sparseは、片側相でのpivotの同値書戻しを除いた現行v4を用いる。頂点ごとのpivot置換と4区間の相実行を全グラフに共通して適用し、単一f32で15反復する。丸め誤差の補償・疑似倍精度・追加反復は用いない。CPU・Atomicの比較値は既存v3の参照である。\n\n'
 assert marker in experiments;experiments=experiments.replace(marker,implementation+marker,1);replace_section('## Experiments',2,experiments)
 rr=[r for r in csv.DictReader((base/'tables/summary.csv').open()) if r['method']=='rr_sparse_sgd' and int(r['paired_seeds'])];nfast=sum(float(r['speedup_median'])>1 for r in rr)
 abstract=section(fresh,'## Abstract',2)
 old='Sparseについても測定した実装で時間短縮を観測したが、設計との差異が残るため予備的な結果として扱う。';assert old in abstract
 abstract=abstract.replace(old,f'RR-Sparseのv4測定では、比較可能{len(rr)}グラフ中{nfast}グラフで既存CPU結果より短い反復時間を得た。GPUは単一f32、CPUはf64とし、同じ15反復で比較した。').replace('Apple M4 Pro上でFull RRは','Apple M4 Pro上の既存v3測定ではFull RRは');replace_section('## Abstract',2,abstract)
 conclusion=section(generated,'## Conclusion',2);fresh_conclusion=section(fresh,'## Conclusion',2);first=fresh_conclusion.split('\n\n')[1]
 paragraphs=conclusion.split('\n\n');paragraphs[1]=first
 metal='提案方式をwgpuとWGSLで実装したが、本実験はMetalバックエンド上の単一機種での評価であり、他の実行環境への性能の一般化は今後の検証課題である。'
 conclusion='\n\n'.join(paragraphs).rstrip()+'\n\n'+metal+'\n';replace_section('## Conclusion',2,conclusion)
 # Explicit guards for unaffected concurrent revisions and Full results.
 for heading,level in [('## Introduction',2),('## Related Work',2),('### 3.2 ミニバッチSGDとの関係',3),('### 3.3 独立な頂点対更新とRound-robin',3),('### 3.4 RR-Full SGD',3),('### 3.6 wgpuを用いた実装',3),('### 4.1 タイルサイズの影響と共通設定',3),('## References',2)]:assert section(text,heading,level).strip()==section(fresh,heading,level).strip(),heading
 for prefix in ['Full RRは比較可能19グラフ','Full Atomicは19グラフ','Full RRは16,384頂点']:
  a=next(p for p in generated.split('\n\n') if p.startswith(prefix));assert a in text,prefix
 out.write_text(text);(base/'report/paper-merge-audit.json').write_text(json.dumps(dict(policy='Fresh manuscript retained; only Sparse method, v4 experiment evidence, abstract and conclusion numerical interpretation updated.',preserved=['Introduction','Related Work','New method section numbering','Full method and tile sensitivity','wgpu implementation section','References']),indent=2)+'\n');print('Merged v4 evidence into the latest six-subsection manuscript.')
if __name__=='__main__':merge()
