# RR tile-size cause investigation (2026-11-07)

- [Results and limitations](RESULTS.md)
- [Diagnostic figure](diagnosis.png)
- [All grouped measurements, including min/max](summary.json)
- [Replay instructions](../../../tools/diagnostics/rr-tile-cause/README.md)
- [Source, input, binary, and data provenance](provenance.json)
- [Instruments recording notes](profile-note.txt)

Main analysis uses 590 runs including warmups: 200 actual Full-SGD runs,
360 controlled cross-round microbenchmark runs, and 30 timestamp-disabled
Full-SGD runs. Exploratory `micro.jsonl` and profiled `profile-stdout.jsonl`
are excluded. This directory is separate from the v5 paper measurements.

The raw `metal.trace` is retained locally and ignored by Git. This recording
provided a GPU timeline but did not expose compute occupancy counters;
per-core occupancy and individual stall causes have not been measured.

The final source snapshot is for replay. Earlier example revisions used in
initial runs had the same Full-SGD path; original binary hashes were not saved.
The production GPU and shader sources were unchanged throughout.
