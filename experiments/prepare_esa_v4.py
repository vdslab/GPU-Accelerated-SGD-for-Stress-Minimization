#!/usr/bin/env python3
"""Prepare esa Markdown using uploaded URLs and freshly read existing post bodies."""
from pathlib import Path
import argparse,json,re
BASE=Path(__file__).resolve().parents[1]/'experiment_results/v4'
def normalize_prose(text):
 text=re.sub(r'<details>.*?</details>','',text,flags=re.S)
 text=re.sub(r'!\[[^\]]*\]\([^\n]+?\)','IMAGE',text)
 return re.sub(r'\s+',' ',text).strip()
def update_topics(paper,source):
 # Retain esa's topic-sentence planning aids, refreshed to match the revised paragraphs.
 for match in re.finditer(r'^(## [^\n]+)\n\s*(<details><summary>(.*?)</summary>.*?</details>)',source,re.M|re.S):
  heading,_,title=match.groups();assert heading in paper
  start=paper.index(heading)+len(heading);end=paper.find('\n## ',start);chunk=paper[start:end if end>=0 else None]
  section='導入';rows=[]
  for para in chunk.split('\n\n'):
   p=para.strip()
   if p.startswith('### '):section=p[4:];continue
   if not p or p.startswith(('#','!','$','|','<','Keywords:')):continue
   first=p.split('。',1)[0]+'。' if '。' in p else p.splitlines()[0]
   first=first.replace('**','').replace('|','\\|');rows.append(f'| {section} | {first} |')
  block='\n\n<details><summary>'+title+'</summary>\n\n| 節 | トピックセンテンス |\n|---|---|\n'+'\n'.join(rows)+'\n\n</details>'
  paper=paper[:start]+block+paper[start:]
 return paper

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--images',type=Path,required=True);ap.add_argument('--paper-source',type=Path,required=True);ap.add_argument('--results-source',type=Path,required=True);a=ap.parse_args()
 audit=json.loads((BASE/'tables/audit.json').read_text());assert audit['completed']==audit['planned'] and audit['convergence_evaluated']
 images=json.loads(a.images.read_text());paper_source=a.paper_source.read_text();results_source=a.results_source.read_text()
 baselinepath=BASE/'reference/paper-latest-before-v4/論文_日本語草稿.md'
 if not baselinepath.exists():baselinepath=BASE/'reference/paper-before-v4/論文_日本語草稿.md'
 baseline=baselinepath.read_text();assert normalize_prose(baseline)==normalize_prose(paper_source),'esa prose changed during measurements; merge it before applying generated manuscript'
 for num,name in [(1,'fig01_atomic.png'),(2,'fig02_rr_full.png'),(4,'fig04_tile_size.png')]:
  found=re.search(r'!\[図'+str(num)+r'[^\]]*\]\((https://[^\s)]+)\)',paper_source);assert found,(num,'missing existing image');images[name]=found.group(1)
 required=['fig01_atomic.png','fig02_rr_full.png','fig03_rr_sparse.png','fig04_tile_size.png','fig05_comparison.png','fig06_convergence.png','fig07_large_layouts.png','sparse_speed_stress.png'];assert all(images.get(n,'').startswith('https://img.esa.io/') for n in required)
 def replace_images(text):
  for name,url in images.items():text=text.replace('図表/'+name,url).replace('../figures/'+name,url)
  assert not re.search(r'!\[[^\]]*\]\((?:図表/|\.\./figures/)',text)
  return text
 paper=replace_images(update_topics((BASE/'report/論文_日本語草稿_v4.md').read_text(),paper_source))
 report=replace_images((BASE/'report/RESULTS.md').read_text()).replace('# v4 Sparse SGD 論文図の再実験結果','## 現在の結果：v4 RR Sparse SGD',1)
 full=[]
 for heading,next_heading in [('### 2. 小規模20グラフ：Full SGD','### 3.'),('### 6. タイルサイズ','### 7.')]:
  start=results_source.find(heading);assert start>=0,heading;end=results_source.find(next_heading,start);assert end>start;full.append(results_source[start:end].strip())
 results=report+'\n\n## Full SGD：既存v3の結果（再測定なし）\n\n'+'\n\n'.join(full)+'\n\n<details><summary>v3の実験計画・全結果（過去記録。現在のRR Sparse結果は上記v4を参照）</summary>\n\n'+results_source+'\n\n</details>\n'
 (BASE/'report/esa9028-v4.md').write_text(paper);(BASE/'report/esa9029-v4.md').write_text(results);(BASE/'report/esa-image-urls.json').write_text(json.dumps(images,ensure_ascii=False,indent=2)+'\n')
 print('Prepared esa9028-v4.md and esa9029-v4.md; no publication performed.')
if __name__=='__main__':main()
