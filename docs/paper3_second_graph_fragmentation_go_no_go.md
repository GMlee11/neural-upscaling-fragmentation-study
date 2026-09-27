# Paper 3 Second-Graph Fragmentation Audit and Go/No-Go

Date: 2026-08-12 (reconfirmed read-only at baseline
`2aa43be2ae0a40e4b3b68903215ed2a3ea383f3c`)

Historical decision on 2026-08-12: **NO-GO FOR RK3576 AUTHORIZATION**.

Current index (2026-08-18): the implementation blockers identified below have
now been addressed offline for the frozen Progressive ESPCN A+B+C checkpoint.
The eight-shape ONNX catalog, selected-output reference, deterministic schedule,
fresh experiment IDs and local-only RKNN compilation path are governed by
`configs/paper3_progressive_multinpu_fragmentation_v1.yaml` and
`docs/paper3_offline_readiness_v1.md`. This does not rewrite the historical
audit or authorize hardware. RK3576 capture remains pending independent review;
RK3566 capture is ineligible because the one preselected image has no accepted
NPU DTB/driver/firmware closure.

This is a catalog/provenance audit and conditional protocol sketch, not an
active preregistration. It authorizes no graph conversion, board access,
warmup, sample, result file, or change to the accepted V1 experiment. No opened
or sealed trace was read. No RK3576 connection was made during the
reconfirmation.

## Scientific purpose

The accepted V1 result establishes fragmentation causality for one width-24
residual graph through one RKNN interface on one RK3576. A second capture is
worthwhile only if it tests a genuinely different operator topology under the
same compact/fragmented geometry. Repeating the experiment with only a channel
width change would increase samples without resolving the external-validity
objection.

## Read-only catalog audit

| Candidate | Structural distinction | Existing RKNN deployment | Paired sparse-mode closure | Disposition |
|---|---|---|---|---|
| Width-24 residual student | Three-convolution subpixel residual prefix | Eight-shape board-ready sparse catalog | Complete | Already measured; cannot reuse the V1 ID |
| Width-16 or width-32 residual student | Same three-convolution topology; channel width only | Full-frame RKNN graph | Absent | Not a genuinely distinct graph family and not runnable in both sparse modes |
| Progressive ESPCN A+B+C | Nine convolutions and two additions across three residual stages | Full 640x360 and 320x180 RKNN graphs | Absent | Genuinely distinct candidate, but not deployable for the paired sparse experiment |
| Independent progressive branches | Three separate stage graphs | Full-frame/180p branch graphs | Absent; one “360p” ONNX bundle also declares a 320x180 input | Not an equivalent sparse request contract |

The only sparse catalog is
`results/models/sparse_refinement_rknn_v2/catalog.json`; it is tied to the
width-24 checkpoint
`fdbaf032b50d02dc214b978ab9083e27d349240f2f5c1929974f90a88a2be1fb`
and eight fixed halo-expanded shapes. The existing sparse exporter loads that
student checkpoint and hard-codes its three-convolution prefix, so it cannot
truthfully qualify the progressive graph without a separately reviewed
implementation.

The strongest existing distinct candidate is pinned by:

- checkpoint:
  `experiments/exp066_progressive_espcn_int8/checkpoints/best_model.pth`,
  SHA-256 `9714c8d9a3994f3cd9b30a3578df69ecd97a08b191ef69a7d777af25b812a6ff`;
- full 640x360 RKNN A+B+C graph:
  `results/models/progressive_espcn_int8/rknn_stage_abc/progressive_stage_abc_subpixel_residual_int8_rk3576.rknn`,
  SHA-256 `42b75502fdc74bf5abde0d61200d3bcf1eff83b4675cf0bbea9b2ac2e61f312c`;
- 320x180 RKNN A+B+C graph:
  `results/models/progressive_espcn_int8/rknn_180p_stage_abc/progressive_180p_stage_abc_subpixel_residual_int8_rk3576.rknn`,
  SHA-256 `c6cd6a3a585c780ece798a96407f134cdacec8c95b779317bb3f927a91e12af3`.

Neither RKNN graph accepts the 66-pixel-high halo tiles or row-span widths used
by the V1 backend. Full-frame execution cannot stand in for the same two sparse
dispatch modes. The current repository therefore lacks an already available,
deployable second graph for this causal contrast.

Creating, training, exporting, or tuning a new graph only to manufacture paper
breadth is outside the accepted scope. The manuscript must remain a one-graph,
one-device case study unless a distinct, independently motivated deployment
path later exists before any result is observed.

## Gates before a real preregistration

All of the following must pass offline before requesting RK3576 access:

1. Implement and review a progressive A+B+C sparse-prefix exporter without
   reading evaluation traces or changing the checkpoint.
2. Export the exact V1 core/halo shape dictionary, prove PyTorch--ONNX
   agreement, compile every shape with the accepted RKNN toolchain, and finalize
   a new board-ready catalog with per-file hashes and tensor contracts.
3. Extend the existing backend only through a new experiment-specific catalog
   contract; preserve catalog-native quantization and expose no scale/zero-point
   knobs.
4. Prove selected-output integrity and cross-mode agreement offline. Do not use
   a physical result to choose shapes, thresholds, or graph stages.
5. Freeze a new implementation baseline and obtain independent review. Only
   then may the candidate experiment ID
   `paper3_rk3576_fragmentation_pairs_progressive_abc_v1` be preregistered.

## Conditional protocol after those gates

The future preregistration should retain the V1 compact/maximally-fragmented
mask pairs at 20%, 40%, and 60%, the `exact` and `row_spans` modes, identical
selected count and six active rows per pair, two warmups per compiled shape,
and the fixed V1 structural/service/integrity thresholds. It should use ten
paired repeats rather than five: six masks times two modes times ten repeats,
exactly 120 physical samples in deterministic randomized complete blocks.

Input must be independent of V1 and generated in memory, never from a trace:

- `R=(5x+3y+71) mod 256`
- `G=(11x+7y+43) mod 256`
- `B=(2x+13y+19) mod 256`

The 640x360 row-major HWC byte stream is 691,200 bytes and has SHA-256
`88854a9e2b5b670aa89ab09548b4d11b43a4f1c9c628b4f21a3f16177fe3a66f`.
This fixture is for timing/integrity only and supports no quality claim.

The causal decision rules must be frozen before access and unchanged from V1:
row spans must add deterministic GMAC/traffic at every density; row-span paired
service deltas must be positive at all densities and have median penalty at
least 5% at two; exact-mode structural totals must match within each pair and
absolute median service deltas must be at most 5% at every density; selected
output MAE must be at most 0.001 and PSNR delta at most 0.02 dB; every deadline
outcome must be reported. Ten repeats improve distribution visibility but do
not authorize p-values, post-hoc outlier deletion, or threshold revision.

One separately authorized capture is permitted only after a clean, hash-pinned
checkout, board/runtime identity checks, exclusive AC-powered session, thermal
and storage gates, and independent signoff. A stopped or failed capture is
preserved and is not retried under the same ID. There is no post-result
retuning, graph replacement, sample replacement, or trace access.

## Authorization verdict

The scientific study is conditionally worthwhile because it would test whether
the measured fragmentation effect transfers across operator topology, not just
another mask or width. The historical hardware verdict was therefore **NO-GO**
at the time of this audit. The later append-only offline implementation closes
the catalog prerequisite but does not retroactively authorize or claim a board
experiment.
