# Dispatch extension: reviewer replay and evidence boundary

The completed extension contains four graph/device cells (width-24 INT8 and
seven-convolution FP16 on RK3576 and RK3566), each with two sessions, forty
masks, four policies and eight blocks: 1,280 samples per cell, 5,120 total.
The masks represent compact, scattered, clustered and mixed requests at five
selected-tile counts. All four policies preserve the same selected outputs.

## Reproduce the manuscript dispatch table

From the package root, using fresh output filenames:

```powershell
python -B -m src.report.paper3_dispatch_publication_v1 --json-output dispatch-check.json --latex-output dispatch-check.tex
```

Compare these with `results/tables/paper3_dispatch_publication_v1.json` and
`docs/paper/generated/paper3_dispatch_publication_v1.tex`. The normal
`artifact/paper3_samos/reproduce.ps1` flow performs this replay and byte comparison
in a temporary directory without changing the reviewer package.

`results/manifests/paper3_dispatch_public_v1.json` contains every included
request's nanosecond service endpoints, policy/choice, mask/order, decision
overhead, temperature and control-invariant hashes, raw-output identity/hash,
session bindings and complete-journal SHA-256. It includes the frozen tensor
contracts, coefficient vectors, calibration and numerical-qualification
identities, method-file hashes, full schedule, and author-side independent-code
raw-audit outcomes. Replay recomputes mask features and decisions, exact service
durations, all policy/session means, deadline counts and paired regressions.

This is a compact **timing/policy replay**, not a rerun of a board and not a full
raw numerical replay. Complete process-rich journals, RKNN binaries and raw NPZ
arrays remain in the research archive and are not included. Their hashes and
the prior raw-audit results provide provenance, not substitutes for access to
those bytes. The author-side export first reruns independent raw-output,
schedule and control verification; reviewers of this slice can inspect the
recorded audit results but cannot independently repeat its tensor checks.

## Fixed method and boundaries

The original protocol record is
`docs/paper3_dispatch_successor_v1/protocol_candidate.md`, SHA-256
`25c4323b23433b936940f99fd07fc816949cff31aa8ccbd3b1a5a816f5ce8cee`;
its subsequent confirmation approval is
`docs/paper3_dispatch_successor_v1/orchestrator_confirmation_approval.md`,
`615a573c7313ec62bbc8928912fb2704debfbd987216fb109bab5feed4fdf6d1`.
The global frozen method/coefficient record has SHA-256
`cc7daa6a07f266be42016843bf606d3bfe2b692f5e8e24887b887b8e72c124fb`.
Those historical author-side records remain in the research archive; the
package records their identities and the operative method below rather than
presenting internal authorization notes as reviewer instructions.

The policy compares exact tiles with row spans rounded upward to a compiled
width from 64, 128, 256, 512, 576 or 640. The five cost features are an intercept,
launch count, dispatched-core megapixels, qualified tensor movement in MiB and
selected high-resolution commit megapixels. Nonnegative ridge coefficients use
2,000 fixed coordinate sweeps and lambda 1e-6, fitted only to the separate
240-request development calibration per cell. Prediction ties choose exact.
The simple heuristic chooses row spans only when launches are at most half the
selected tiles and padded core area at most 1.25 times selected area.

The included complete-service interval covers policy/worklist generation,
bicubic fallback, extraction, inference, reconstruction and commit. Numerical
validation, durable evidence writes and between-request controls are outside
the timer. Between-request activity can nevertheless affect thermal/idle
history. Captures start blocks at no more than 55 C and stop at 80 C; numerical
gates are selected MAE <= 0.001 and PSNR difference <= 0.02 dB, with repeat
identity and unchanged unselected fallback. Hash-pinned control snapshots are
point-in-time observations, not proof of continuous whole-system exclusivity.

The original RK3576 width-24 run used the reserved confirmation geometry.
RK3576 FP16 V1 stopped after 640 + 320 requests on control drift; the rejected
field was not retained, so its cause remains unresolved. That entire incomplete
campaign is separately identified and excluded from the performance table.
The complete FP16 V2 rerun uses the same frozen method and already-opened
geometry. RK3566 also uses that already-opened geometry, with separately frozen
per-target calibration. These are not fresh unseen-geometry confirmation or
calibration-free transfer. The completed sessions are never spliced with V1.

The calibrated mean is 1.35--3.02% below the best fixed policy across the four
cells, but the RK3576 FP16 first session regresses, and every cell has individual
paired timing regressions. Such differences include timing variability and
decision overhead; they are not counts of wrong organization choices. No
population significance, cross-vendor generality, universal gain, energy saving
or real-time 720p claim follows. This is a bounded practical illustration of
the fragmentation trade-off, not a novel scheduler algorithm.
