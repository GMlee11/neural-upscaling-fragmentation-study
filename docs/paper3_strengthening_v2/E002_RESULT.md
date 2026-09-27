# E002 result — empirical strengthening, not a successful new scheduler

Completed 2026-09-25. This append-only report records the frozen outcome.
No additional board measurements, fitting, or mask selection are planned.

## Evidence identity and completion

- Source: `results/paper3_strengthening_v2/E002_source_v3`; manifest SHA-256
  `0364767473f4b073f6b453057f4a13f42c8009a67d8de529b3d166f8e3ef7a95`.
- Four RK3576 cells completed: width24 and FP16, each in two physical sessions.
  Each retained 465 samples: 416 evaluation, 48 qualification and one reference.
  Total: 1,664 evaluation requests and 1,860 retained samples.
- Sixteen reserved masks form eight compact/fragmented pairs in four families.
  Each pair has identical selected count, active rows and per-row counts.
  Non-overlap was checked against 162 geometries in 12 named hashed catalogs,
  not an asserted enumeration of every historical request.
- Both reconstruction orders occur within each session; graph order reverses
  between sessions. The fixed 300-second inter-session idle and restoration
  occurred. Earlier engineering failures and all adverse outcomes remain.
- V1 stopped before any retained sample because Python 3.12/3.11 float-sum
  semantics produced different score-table hashes. V3's explicit left-fold
  accumulation was qualified locally and on the board. Masks, schedule,
  profiles, coefficients, original reservation time, planner job tables and
  all reserved-mask chosen configurations remained unchanged. V2 was a local
  builder failure, never a physical evaluation. See the portability audit.

Independent reports are `results/paper3_strengthening_v2/E002_confirmation_summary.json`
and `E002_prediction_summary.json`. They authenticate the frozen schedule,
source/profile/table identities, journals and all four pixel-replay reports.

## Passed geometry gate, failed chooser gate

The prespecified **row-span geometry interaction gate passed for both graphs**.
The quantity below is `(fragmented - compact)whole - (fragmented - compact)selected`
in complete service time. Positive values mean selected-only reconstruction
reduced the measured fragmentation penalty. These differences of differences
are not whole-request speedups.

| Pair family | Width24, ms | Gate | FP16, ms | Gate |
|---|---:|---|---:|---|
| Sparse, all six rows | 16.194 | Pass | 24.549 | Pass |
| Sparse, three rows | 5.906 | Pass | 7.470 | Pass |
| Dense, all six rows | -8.842 | Fail: session sign changes | 10.558 | Pass |
| Dense, four rows | 0.194 | Fail: small and session sign changes | 14.211 | Pass |

The gate required at least two families per graph with pooled magnitude at
least `max(2 ms, 5% of whole-compact service)` and the same sign in both sessions.
Width24 passed two sparse families; FP16 passed all four. Family pooling does
not mean every individual pair improved. Dense cases and session variation must
remain visible. The result supports a geometry-dependent host contribution, not
an attribution of all remaining cost to intrinsic NPU computation. RKNN inference
intervals include compiled-graph/runtime and host adapter work.

The **frozen chooser failed its practical performance gate on both graphs**.

| Graph | Best fixed configuration | Fixed mean, ms | Chooser mean, ms | Chooser slower | Worst mask regression |
|---|---|---:|---:|---:|---:|
| Width24 | Affine partition, selected | 84.429 | 91.928 | 8.882% | 39.259% |
| FP16 | Per-row choice, selected | 98.751 | 100.195 | 1.462% | 12.348% |

The chooser was slower than the session-specific best fixed configuration in
both sessions of both graphs. It regressed on 13/16 width24 and 11/16 FP16 masks
versus each graph's pooled best fixed configuration. It therefore failed both
the required 5% mean improvement and the no-greater-than-10% per-mask regression
criterion. Mean decision cost was 0.319/0.317 ms; removing the old online dynamic
programming overhead was not enough. Table preparation took 19.0–19.6 seconds
per cell. There is no positive amortization versus the best fixed baseline here.

The 5,000-draw stratified paired-mask bootstrap gave chooser reduction intervals
[-13.306%, -4.586%] for width24 and [-4.607%, -0.086%] for FP16. These are
descriptive sensitivity intervals with only two pairs per family, not validated
population confidence coverage or cross-device generalization. Practical gates
are not significance tests, novelty criteria or acceptance probabilities.

## Prediction and scientific boundary

The frozen ordinary call-count/element-volume model predicted the *normalization
component's* whole-versus-selected sign on 85.4%/92.7% of paired observations,
versus 66.7%/78.1% for always-selected reconstruction. Pair-difference MAE was
2.213/2.155 ms. This component result is not a deployed end-to-end scheduling
win. The frozen profiled job-cost estimate had MAE 19.235/15.096 ms and did not
yield a better complete-service chooser. No coefficients were refit.

The useful finding is that a measured fragmentation penalty is partly a property
of host reconstruction and its interaction with geometry, even with unchanged
NPU jobs and committed outputs. Stronger conventional baselines and the failed
chooser establish limits on simple launch-versus-padding prediction. Cropping,
PixelShuffle, dynamic programming, shape profiling and table compilation are
established techniques, not new algorithms introduced by this work.

This is bounded empirical strengthening of Paper 3, not a new generally superior
scheduler or accelerator mechanism. Both graphs use the same RKNN family and
synthetic input. Fresh geometry and two sessions are not a new application,
vendor or backend. Existing RK3566 results remain separate evidence. Proceed
with honest manuscript/artifact integration, not another automatic experiment.

## Numerical integrity and board release

All four `E002_*_v3_pixels.json` reports passed independent retained-pixel replay.
Each authenticates its journal, source, graph, session and 465 samples. Exact,
row-span and grouped policies have 96 same-job whole/selected pairs per cell;
every pair is byte-identical. Width24's maximum aggregate selected-output MAE
was 0.000872816 under the unchanged 0.001 gate; maximum individual-tile MAE was
0.000878018. This is not byte equality across different organizations. FP16's
same-backend selected difference was zero; its maximum error versus the pinned
independent full reference was 0.0000260799. Fallback verification is
complementary-corner cross-record consistency, not regenerated bicubic truth.

The final cell and launcher exited successfully. GDM and logind were verified
active with their prior static enabled states; no experiment worker remained.
SSH, clocks and governors were not changed. Temporary headless setup removed
uncontrolled shared device handles; such handles do not prove prior inference
contention. The read-only, digest-pinned dual-node helper remains installed under
explicit user authorization; its ownership checks are point-in-time.

All data are retrieved and hash-verified under
`C:\Research\Thesis\Visual_Upscaling\neural-upscaling-hw-sw-codesign\results\paper3_strengthening_v2`:
`E002_width24_s0_v3`, `E002_fp16_s0_v3`, `E002_fp16_s1_v3`,
`E002_width24_s1_v3`, and `E002_source_v3_headless.jsonl`.
`E002_v3_raw_inventory.json` verifies 968,059,676 retained NPZ bytes, of which
484,029,838 bytes are 120 unique files/arrays. Second-session copies are
byte-identical. No copies were deleted or replaced. A later optional
content-addressed numerical supplement can avoid those duplicate bytes.

The Radxa is finished and can be shut down normally before disconnecting power.
No commit, push, public upload or submission was performed by this work.
