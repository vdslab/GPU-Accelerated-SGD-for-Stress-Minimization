# v4 completion work

The user requested rerunning only Sparse experiments currently plotted in the paper, organizing v4, then updating esa results and figures. Full remains v3. Commits authorized, no push requested. Do not spawn agents.

## Running jobs

Main repo: `/Users/kose/master/GPU-Accelerated-SGD-for-Stress-Minimization`, branch main.
- Serial benchmark: `python3 experiments/run_v4.py`, originally PID 98464, exec session 38640. Log `runner.log`, progress `checkpoint.json`.
- Completion worker: `python3 experiments/finalize_v4.py --wait-pid 98464`, exec session 85947, log `finalize.log`. It waits for the benchmark, then runs summarize, plot, report, manuscript draft and schematic scripts. It does NOT publish or install manuscript files.
- Do not rebuild binaries or change run_v4.py/config plans during measurement. Native binary SHA and runner SHA are frozen. Execution source commit 7dbf14a0cdb09acce1a73973c0b498922dbb20b6.
- Do not run other benchmarks or heavy analysis in parallel. Current benchmark is serial and may take several hours. Inspect logs/checkpoint; resume same runner only if stopped after identifying cause.
- 660 planned records: 38 graphs ×3 methods×5 seeds basic =570; 3 graphs ×3 methods×10 seeds convergence=90. Warmups separate. Failure/warmup_failed is a result, not a reason to tune parameters or selectively rerun.
- GPU plain single f32, CPU f64, all15 iterations, epsilon .1, pivots200. No compensation, extra iterations, graph-specific schemes. RR vertex-specific permutation +4 chunks. Atomic current default no CPU preshuffle (changed from v3, disclosed).
- Sampled full stress uses256 fixed sources, seed0. CPU/GPU initial/input/preprocess SHA checked. All38 are >100k nodes. RSS8GiB/process120s/minfree8GiB monitoring retained.

## On measurement completion

1. Confirm all660 unique IDs accounted for, archive_status success, no pending. Check finalize.log. If worker fails, fix only analysis and rerun needed scripts serially. Distance evaluation cache under cache/ is ignored. Recomputed final checkpoint stress must match native result to1e-10 relative; never bypass an unexplained mismatch.
2. Inspect `tables/audit.json`, `summary.csv`, `paired-ratios.csv`, `missing.csv`, `arrivals.csv`, `convergence.csv`, `figure06_coverage.json`.
3. View and verify figures in figures/: fig03_rr_sparse, fig05_comparison, fig06_convergence, fig07_large_layouts, sparse_speed_stress (PNG/PDF/SVG). Ensure labels/axes/errors aren't clipped, no incorrect N/A interpretation. Full numeric data are separately inherited in reference/full-v3. Fig4 must remain unchanged.
4. Review report/RESULTS.md and report/論文_日本語草稿_v4.md. Draft generator verified unique paragraph replacements using a temporary incomplete-data fixture; final counts must still be inspected. Delaunay_n19 CPU completion is dynamic; never assert CPU failed if it succeeded. Missing reasons need clear readable Japanese, improve if raw status dictionary is awkward.
5. Install manuscript and updated figures to `/Users/kose/Library/Mobile Documents/iCloud~md~obsidian/Documents/sakamoti55/11_大学院研究/02_論文フェーズ/`. Main draft: 論文_日本語草稿.md, figures: 図表/. Copy3/5/6/7; Full1/2/4 unchanged. Update 図表/README.md and make_figures.py so regeneration uses current v4 scripts, not overwrites v4 with old v3 Sparse. Old script saved reference/paper-before-v4/make_figures.py. Formal TeX repo is a skeleton; no need edit.
6. Update esa result page9029 and manuscript9028, including new uploaded figures. Read fresh versions first and preserve user edits. Backups from this run are reference/paper-before-v4/esa9028-before-v4.md and esa9029-before-v4.md. Full sections unchanged; the new result page can lead with the v4 report and retain existing Full sections plus explicitly folded v3 history. Clearly state v4 covers paper's38graphs, NOT all78 historical candidates/pivot sweeps. Do not silently present old Sparse charts as current. Use updated composite paperfigures5/6/7 and Sparse speed/stress supplement; methodfigure3 reflects current4chunk ordering.
7. Verify saved esa pages display current counts/images. Save final esa Markdown and image URL map under report/ and publication audit JSON with URLs/revisions.
8. Update README final status/reproduction commands. Commit scripts, small raw metadata, tables, figures, reports. Coordinate archives remain local ignored, with SHA manifests; Full4 copied coordinate archives under reference/full-v3/coordinates/ are likewise ignored. Do not push without instruction. Report results/limitations + paths + esa links + commit.

## esa access

- https://vdslab.esa.io/posts/9028 (paper) and https://vdslab.esa.io/posts/9029 (results).
- mcp__esa is configured ONLY for team spookies, not vdslab. It cannot read/write vdslab (confirmed access error). DO NOT write to spookies9028, unrelated post.
- Use existing logged-in native Chrome via mcp__cua_repl. No Chrome browser provider; cua inventory has iab/mcpapps only. Do not use HTTP credential extraction/AppleScript/UI bypass.
- Persistent Cua binding `chrome = await cua.getApp('com.google.Chrome')` exists. After compaction call `await cua.rewriteDocumentation()` before continuing.
- App binding follows active Chrome window; user has unrelated Chrome profiles. Select correct esa window through Window/ウィンドウ menu using fresh AX state, then choose title 実験結果のまとめ or 一旦書く. Don't act on unrelated Gmail/PDF windows. IDs change each observation.
- Native edit UI has text input area with complete Markdown in AX. Read full AX and extract body until next indented numbered AX element. Strip trailing ` (showing 0-100 of N items)` AX annotation if present.
- Read-only backups used cua+fs ordinary file writes. Editor opened and closed WITHOUT publishing. Current page9028 display. No edits to esa yet.
- Upload via Attach Files (up to10MB), mac file dialog path entry. User explicitly authorized esa/figure replacement, so no extra confirmation needed. Use fresh AX after actions; do not guess indices. `chrome.paste(markdown,{format:'text'})` can insert prepared file text after focusing correct editor and selecting all; verify before Ship It.
- Existing images1/2/4 can be retained by extracting their remote Markdown links from fresh9028 source; local draft has relative 図表/ paths. New3/5/6/7 should be uploaded and mapped.9028 backup currently has latest Full fig4 rawstress (not old ratio); initial cached AX had older version but edit source confirmed latest.

## Validation already done

Release builds and tests for RR/alltargets, Atomic and shared common passed; CPU release built. Three-method trace smoke passed (CPU/RR bit-identical final coordinates trace on/off, Atomic validated coverage). Small synthetic Python checkpoint evaluator checks passed. Summary partial pairing/SHA checks passed. Report generation and manuscript paragraph replacements passed in temporary fixture. No formal results claimed yet.

## Model-switch recovery (2026-10-30 05:23 JST)

The model switch terminated the original benchmark parent and waiter after59 persisted records. The isolated native child completed without parent bookkeeping; its entire folder was moved to interruptions/model-switch-* with a reason record, before quality inspection. Same frozen condition was repeated once. A detached runner PID37897 and finalizer PID37898 now continue; use processes.json and actual process state rather than old IDs. The15minute heartbeat sparse-v4-esa was PAUSED while this active chat handles completion, to avoid duplicate publication. User explicitly requested staying active through completion and including estimated remaining time in progress comments. progress_v4.py estimates remaining graph-dependent duration using historical v3 times calibrated to completed v4 runs. Initial range about6–10hours including downstream work, not a deadline.
