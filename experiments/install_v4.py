#!/usr/bin/env python3
"""Install reviewed v4 paper outputs while preserving the existing Full figures."""
from pathlib import Path
import argparse,csv,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiment_results/v4'
PAPER=Path('/Users/kose/Library/Mobile Documents/iCloud~md~obsidian/Documents/sakamoti55/11_大学院研究/02_論文フェーズ')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--figures-only',action='store_true');args=ap.parse_args()
 a=json.loads((BASE/'tables/audit.json').read_text());assert a['completed']==a['planned'] and a['convergence_evaluated']
 figures=PAPER/'図表';full_before={str(p):digest(p) for n in ['fig01_atomic','fig02_rr_full','fig04_tile_size'] for p in figures.glob(n+'.*') if p.suffix in ['.png','.pdf','.svg']}
 if not args.figures_only:
  old=BASE/'reference/paper-before-v4/論文_日本語草稿.md';target=PAPER/'論文_日本語草稿.md'
  assert digest(old)==digest(target),'Local manuscript changed during measurement; merge the user changes before installation'
  shutil.copy2(BASE/'report/論文_日本語草稿_v4.md',target)
 for name in ['fig03_rr_sparse','fig05_comparison','fig06_convergence','fig07_large_layouts','sparse_speed_stress']:
  for ext in ['png','pdf','svg']:shutil.copy2(BASE/'figures'/f'{name}.{ext}',figures/f'{name}.{ext}')
 times=[r for r in csv.DictReader((BASE/'reference/full-v3/fig05_iteration_times.csv').open()) if r['family']=='full']+[r for r in csv.DictReader((BASE/'tables/fig05_iteration_times.csv').open())]
 with (figures/'fig05_iteration_times.csv').open('w') as f:
  fields=list(dict.fromkeys(k for r in times for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([dict(r,experiment_version=r.get('experiment_version','v3')) for r in times])
 (figures/'make_figures.py').write_text('''#!/usr/bin/env python3
"""Regenerate the current Sparse figures (CPU/Atomic v3 references, RR v4)."""
from pathlib import Path
import subprocess,sys
repo=Path('/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization')
for script,args in [('plot_v4.py',[]),('diagram_v4.py',[]),('install_v4.py',['--figures-only'])]:
 subprocess.run([sys.executable,str(repo/'experiments'/script),*args],cwd=repo,check=True)
''')
 (figures/'README.md').write_text('''# 論文図表

図1・2・4は既存のFull/Atomic説明図・Full v3結果。図3はRR Sparse v4の更新構成。
図5〜7はFullをv3から継承し、SparseのCPU・Atomicは既存v3、RRは再測定v4を使う。
`sparse_speed_stress`は全38対象の速度・stress比の補足。欠測は0/1に置き換えない。

全てPNG（確認用）・PDF/SVG（論文用）。図5の時間は15反復合計÷15の平均ms。
点は対応成功seedの中央値、棒は最小〜最大。図6の下段は固定256始点のstress推定値。
CPU・AtomicとRRは別日時の測定であり、同一セッションの速度比較とは区別する。

`python3 make_figures.py`でv4の図3・5・6・7と補足図を再生成する。
図1・2・4はこの処理では変更しない。旧生成コードはメイン実装リポジトリの
`experiment_results/v4/reference/paper-before-v4/make_figures.py`に保存している。
原表・欠測・出所は同リポジトリ`experiment_results/v4/`を参照。
''')
 assert full_before=={p:digest(Path(p)) for p in full_before}
 (BASE/'report/local-install.json').write_text(json.dumps(dict(full_figures_unchanged=full_before,paper=str(PAPER),figures_only=args.figures_only),indent=2)+'\n')
 print('Installed reviewed Sparse figures; existing Full figures verified unchanged.')
if __name__=='__main__':main()
