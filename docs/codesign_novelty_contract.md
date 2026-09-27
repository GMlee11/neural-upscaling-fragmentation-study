# Paper 3 Novelty Contract

## Working Claim

Paper 3 is a bounded case study of two structurally different graph/precision
cells, one RKNN runtime family, and RK3576/RK3566 targets. It first uses a preregistered equal-
count/equal-row control to isolate fragmentation, repeats the control with a
separately preregistered shared-feature FP16 graph, then reports a descriptive
15-mask execution-tax vector: launch amplification, dispatched-core padding,
tensor traffic, physically exposed reconstruction/host tail, and deadline loss.
Every quantity comes from Paper 3 masks and RKNN measurements.

## What Is New Enough To Test

1. A physical equal-count/equal-row fragmentation comparison of independent-
   tile and row-span static RKNN organizations under one frozen output-demand
   contract, with all repeat distributions reported.
2. A descriptive 15-mask execution-tax vector using selected-mask geometry and
   the static RKNN shapes actually dispatched. No external accelerator
   denominator or companion result enters the accounting.
3. Exact per-mask descriptors and descriptive order/temperature sensitivity
   from already-open records, with no regression or thermal/order causal claim.
4. Six static-graph NPU interface gaps derived directly from matched RK3576
   measurements; prior interface concepts appear only as possible remedies.
5. A topology/precision transfer result from a pre-existing seven-convolution
   shared-feature FP16 graph that passed all eleven target numerical gates
   before a complete 120-sample paired capture.

## What Is Explicitly Not New

- Neural super-resolution or ESPCN.
- Tile selection, early exit, token skipping, or sparse CNN execution.
- Heterogeneous CPU/GPU/NPU scheduling by itself.
- Line buffers, PixelShuffle, quantization, or FPGA acceleration by themselves.
- Deadline-aware scheduling or cancellation by itself.
- Exact dependency closure, persistence, atomic commit, FPGA organization
  ablations, or cross-vendor FPGA portability; these are Paper 2 contributions.
- Displayed-result freshness and temporal reprojection; these are Paper 1
  contributions.

## Required Evidence

- The same checkpoint identity, resolution, mask geometry, tile geometry,
  nominal precision, and deadline are documented for every platform.
- Every backend passes its own frozen numerical reference.
- The independent-tile and row-span organizations are physical RK3576 results.
- Selected work, dispatched-core padding, tensor traffic, service time, and
  deadline outcomes are derived entirely from Paper 3 evidence.
- NPU launch reduction is reported beside added dispatched work and traffic.
- The frozen six-mask control passes every primary gate: row-span median service
  penalties are 199.665%, 130.130%, and 12.555%, exact selected-tile
  density-level medians stay within $\pm$1.961%, and all 60 deadline outcomes
  are reported (0 pass). The five paired deltas per density remain visible;
  the median bound is not misreported as a per-repeat bound.
- All 15 row-span paired deltas are positive. Pair orientation and observed
  start temperature are reported descriptively, alongside the absence of
  per-sample timestamps, independent thermal manipulation, and repeat sessions.
- Numerical integrity remains inside the preregistered MAE 0.001 and PSNR-delta
  0.02 dB thresholds.
- The V22 transfer cell reports all ten pairs per density/mode: row-span median
  penalties 156.857%, 73.339%, and 12.252%; exact medians -0.125%, 0.906%, and
  0.882%; and one retained unexplained 2.368-second exact-mode pause.
- Reconstruction and deadline loss are reported only where physical RK3576
  stage/outcome records expose them.
- The failed selector remains frozen and is not marketed as generalized.

## Prohibited Claims

- Real-time 720p sparse neural upscaling on the measured hardware.
- Physical energy efficiency or joules/frame.
- Cross-platform bit identity or equal quantization.
- A generalized mask predictor.
- A universal NPU-versus-FPGA winner.
- Novelty of the Paper 2 FPGA mechanism if that mechanism is already published.
- Successful simple temporal reprojection or a repeated Paper 1 result block.
- Tang full-frame equivalence or reuse of Tang as Paper 3 claimed evidence.
- Broad model-family, cross-runtime, or cross-device generalization from two
  graph/precision cells on one RK3576.

## Publication Separation From Paper 2

Paper 2 owns the detailed FPGA architecture contribution: exact output-mask
closure, persistent traversal, atomic private commit, organization controls,
and portability validation. Paper 3 owns the matched-mask NPU methodology,
NPU-local execution-tax accounting, and measured interface gaps. Paper 3
imports no Paper 2 result, counter, denominator, mechanism measurement, or
conclusion. Its causal controls use only Paper 3 RKNN evidence. The papers may
share a checkpoint, traces, masks, and project history; that substrate is
disclosed without making either paper necessary to reproduce the other.

The live Paper 3 package excludes the historical exact-closure projection and
its config, report, freeze, verifier, and tests. Those records remain preserved
only in the main research repository as development history. Normal venue
related-manuscript and concurrent-submission policies still apply.

## Submission and Breadth Gate

Submission requires:

1. the physical RK3576 matched-mask evidence remains reproducible;
2. the NPU-only summary regenerates from Paper 3 evidence alone;
3. the manuscript clearly distinguishes new Paper 3 contributions;
4. the target venue permits any disclosed related-manuscript relationship; and
5. the near-deadline literature refresh finds no paper with the complete
   matched physical boundary and interface contract.

The working manuscript is therefore narrowed to an RK3576/RKNN case study. A
broader architecture or cross-platform title is not authorized by the evidence.

Cross-model breadth is now limited to the original width-24 INT8 cell and the
separately preregistered V22 shared-feature FP16 cell. The earlier Progressive
A+B+C candidate remains a sealed failed qualification and supplies no timing
evidence. V22 was selected before target output, passed all eleven numerical
gates, and completed 120 physical samples without replacement. No further
RK3576 graph sweep is authorized from these opened results.

### Multi-backend replication amendment (Phase A)

The graph-breadth gate above does not prohibit a fresh, append-only replication
of the already frozen compact-versus-fragmented request on additional named
devices or backends. Such a study may test device/backend transfer of the
measured interface effect while holding the width-24 checkpoint, synthetic
input, mask geometry, dispatch definitions, reconstruction boundary, and
analysis rules fixed. It may not replace, pool with, or relabel the completed
RK3576 V1 result, and it adds no structurally distinct model-family evidence.

This amendment permits protocol preparation and non-result-bearing deployment
infrastructure only. Every new device/backend cell requires a genuinely unused
append-only experiment ID, backend-specific numerical reference, hash-pinned
software/runtime/compiler and image provenance, and a separately accepted
preregistration before conversion, board access, or measurement. Cross-device
FPS is descriptive. No Paper 2 result or normalization enters the replication.

GPU cells are eligible only if a result-free, already supported sparse
tile/row-span execution path is documented before performance is observed. No
such path exists in the current repository, so GPU work is excluded from the
Phase A matrix rather than deferred based on speed.


## Width-24 RK3566 completion (V32)

The separate V32 capture contains 60 samples in five frozen paired blocks.
All 15 row-span pair deltas are positive; median penalties are 181.937%,
102.907%, and 4.145% at 20/40/60% density. Exactly two densities exceed 5%;
the 60% effect is positive but below 5%. The original two-density rule passes.
Exact medians are 0.542%, 0.439%, and -0.138%, with equal within-pair launch
counts and tensor traffic. No sample meets 33.33 ms (0/60). Cross-mode MAE
is at most 0.000842239 and PSNR delta 0.0114562 dB; unselected change is zero,
and selected checksums match qualification. Capture control temperatures span
41.875--48.333 C. All 60 samples, five block orders and range statistics remain
available in the compact offline replay; no CI or p-value is claimed.

The four graph/target cells remain separate. Within-backend fragmentation
ratios are primary; cross-device absolute comparisons are descriptive because
host paths, quantization/compiler details, controls, sessions and five-versus-ten
repeat counts differ. V25/V29/V31 instrumentation failures remain sealed;
V32 is not pooled with or substituted for the original RK3576 V1 result.
Self NPU-FD presence is enforced by the hash-bound runner; the raw journal
does not record an independent self-FD transcript. Permission-limited process
visibility must not be described as whole-system exclusivity.

The reviewer slice contains only the compact V32 sample/metric projection,
paired analysis, generated table and offline replay code/tests. Full
process-rich operator journals remain in the historical repository, bound by
SHA-256 provenance, not in the publication folder. Qualification/capture are
complete; no permission for a rerun follows from this status.
