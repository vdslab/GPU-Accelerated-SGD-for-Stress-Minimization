# v5.5 EX1: RR Full W=T

Only RR Full is newly measured:10 graphs x11 T=W x5 seeds x3 repetitions=1650. CPU150 exact-EX1 measurements are historical references, not newly measured. Previous v5 EX1/EX2 files are hashed and never written. EX2 is not started.

Same input bytes, seeds10..14, repetitions0..2,15 iterations,epsilon0.1,shared,both shuffles,phase_chunks1,auto BFS,exact Full Stress. GPU timestamps off, no drawing during measurement. Warmup seed999 per graph/T per session excluded. Native solver binary freshly built from PR head and frozen with source/binary hashes. Sequential, no time cutoff. Stop between runs at <4GiB disk/<2GiB available RAM. Lossless coordinates gzip after measurement. Strict native record validation uses a copied schema extended only for PR BFS metadata.

Run/resume: `python3 experiments/run_v55_ex1.py --run`. Progress: progress.json/PROGRESS.md and the new v5.5 Obsidian note. Reports and PNG/PDF figures generated automatically after all1650 runs; no automatic EX2 transition.
