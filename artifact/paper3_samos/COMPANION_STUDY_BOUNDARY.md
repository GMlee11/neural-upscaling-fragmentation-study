# Shared-Substrate Independence Boundary

Status: **current, 2026-09-10**

Paper 3 shares a project origin, checkpoint, renderer traces, and frozen masks
with related studies. Those items are experimental inputs, not imported
results. The Paper 3 claims-bearing path uses only its own RKNN graphs,
measurements, numerical gates, timing records, and NPU-local accounting.

The package must not contain:

- Paper 1 timing or displayed-result records;
- Paper 2 RTL, FPGA counters, physical results, or mechanism ablations;
- the historical exact-closure denominator projection;
- a cross-platform NPU-versus-FPGA performance row; or
- development-only `paper1companion` or `paper2companion` citations.

The self-contained summary derives selected work from Paper 3 masks and
dispatched work/traffic from the RKNN tensor shapes actually submitted.
`verify_independence_boundary.py` enforces the file and manuscript boundary.

Related submissions must still be disclosed when venue policy requires it.
That disclosure is about shared substrate and adjacent subject matter; neither
paper is required to validate or reproduce the other's claim.
