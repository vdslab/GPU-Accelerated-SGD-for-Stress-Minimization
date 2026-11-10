#!/usr/bin/env python3
"""Report-only layout/status correction; frozen measurement sources stay unchanged."""
import json
from pathlib import Path
import matplotlib.figure
import report_v55_ex1 as report
from run_experiments import file_sha256 as sha
from run_v5 import save, now

original = matplotlib.figure.Figure.suptitle

def separated_title(self, text, *args, **kwargs):
    if text.startswith('v5.5 EX1:'):
        text = ''  # Conditions are already in the report; omit overlapping global title.
    return original(self, text, *args, **kwargs)

matplotlib.figure.Figure.suptitle = separated_title
report.main()
for p in [report.BASE/'reports/tile.md',report.VAULT/'EX1_v5.5_タイル実験(W=T)_結果と考察.md']:
    p.write_text(p.read_text().replace('EX2は未開始。','EX2は別途完了済み。今回の再開でEX2を再実行していない。'))
p=report.BASE/'validation/completion-validation.json'
d=json.loads(p.read_text());d.update(resumed_from=742,new_after_resume=908,preserved_742_record_hashes_unchanged=True,ex2_already_completed=True);save(p,d)
save(report.BASE/'resume-20261110/report-only-revision.json',dict(updated=now(),measurement_values_changed=False,frozen_report_source_unchanged=True,frozen_report_sha256=sha(Path(report.__file__)),postprocessor_sha256=sha(Path(__file__)),reason='Omit redundant global title overlapping the legend; clarify EX2 already completed'))
