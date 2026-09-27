# E001 interpretation boundaries, recorded during execution

2026-09-25. This supplements, rather than rewrites, the frozen source-V5
engineering design. It does not change the running program or its sample count.

## What the ablation can establish

The exact, row-span and contiguous-group policies have unchanged jobs across
whole-tensor and selected-only reconstruction. Their paired comparison screens
for a complete-service difference between those two host reconstruction paths
within this headless engineering session. However, the cyclic policy ordering
usually retains whole-before-selected adjacency; it is not an independently
counterbalanced reconstruction intervention. Order/cache effects remain a
possible explanation. A causal estimate requires fresh, independently
counterbalanced pairs. This screen cannot identify pure hardware-NPU
compute: `inference_ns` includes RKNN adapter/runtime handling and conversion.
Use **compiled-graph/runtime inference cost**, not intrinsic NPU compute cost.

Per-row, affine-partition and lookup-partition are calibrated independently for
each reconstruction path. Their jobs can therefore differ. Those comparisons
are complete-policy comparisons, not isolated reconstruction ablations.

The selected-only path does not avoid producing or transferring the backend's
full dispatched raw tensor. It avoids materializing a full contiguous HWC
residual after that tensor has already been returned. For the qualified NCHW12
float output, one selected 64-by-60 LR tile contains 46,080 scalar residual
values, the same count as its 128-by-120-by-3 HR output. With a fixed selected
count, selected-only materialization has the same scalar count regardless of
the job width; whole-tensor materialization scales with each job's returned
shape, including context and unselected interior. Small-array call overhead
can offset those savings. Byte counts alone cannot predict elapsed time.

The accepted whole-tensor baseline already uses contiguous HWC normalization,
not a deliberately weak strided implementation. Root verified the accepted
bundle copy `src/deployment/benchmark_rk3576_sparse.py` against current main:
SHA-256 `e9368148b40e835782c89dd761d9aa054ababbf2bc2efbe0c152fe41635a1a78`.
The test suite also exercises the current implementation. The source manifest
and accepted bundle hash bind the board implementation separately.

## Prior-art boundary

The conventional partition dynamic program, shape profiles, crop-before-shuffle
and avoiding unneeded materialization are **not claimed as novel algorithms**.

- [RegenHance, NSDI 2025](https://www.usenix.org/system/files/nsdi25-wang-weijun.pdf)
  already studies sparse region enhancement, packing and profiled end-to-end
  resource planning. Its existing contribution prevents claiming sparse-region
  packing or profiling as our invention.
- [LENS, June 2026](https://arxiv.org/abs/2606.18042) explicitly addresses opaque
  NPU/compiler latency and bucket nonlinearity, so nonlinear finite-shape costs
  and a profile lookup are not sufficient novelty by themselves. Its workload
  is LLM inference, not our fixed-output super-resolution request setting.
- [ISRLUT](https://doi.org/10.1145/3770759) is a directly relevant PixelShuffle
  and materialization prior-art lead. The ACM full-text fetch returned 403 in
  this session; do not use an inaccessible full text to assert a missing feature
  or a first-of-its-kind claim. Search-index descriptions are leads only.

## Decision after independent replay

Uniformly faster reconstruction is useful engineering but does not by itself
create the requested new scientific distinction. A potentially informative
finding would be a meaningful, reproducible change in organization ranking or
fragmentation penalty under an output-identical, fixed-job host ablation, with
an explanation that predicts where it occurs. Three-repetition means from
opened development masks are a screen, not sufficient evidence of a stable
crossover. Any reserved validation needs a new frozen design and review.

Retain regressions, ties, zero effects, calibration costs, and cases explained
adequately by the conventional affine or lookup baseline. Better timing alone
does not certify novelty or a 70% acceptance probability.

## Actual execution condition

The earlier README's no-service-change statement describes the pre-headless
plan/history. The reviewed launcher records and uses a reversible headless
condition: `gdm.service` was active and is temporarily stopped; logind remains
active because stopping GDM released all observed RKNPU handles. The launcher
restores prior active state in `finally`. A GUI-held descriptor does not establish
active inference contention. Only within-session matched contrasts are valid;
do not call a comparison against old GUI-active data a reconstruction speedup.
