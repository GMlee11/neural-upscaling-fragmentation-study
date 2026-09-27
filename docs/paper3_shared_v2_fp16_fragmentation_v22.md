# Paper 3 shared-feature second-graph protocol (V22)

V22 is a clean successor to the sealed Progressive A+B+C numerical failure.
It does not replay, repair, or replace that outcome. It asks whether the same
compact-versus-fragmented effect transfers to a different pre-existing graph.

The candidate is `exp082_progressive_espcn_v2_direct_deep`, checkpoint SHA-256
`f78730284e8199ca93348a5486f059eb7b2bd6e6b6389c96f135ff0daec7ef5a`.
It was selected before any V22 RK3576 output because its recorded training
summary passed all eight per-source monotonic checks. The graph has a shared
stem, five continued feature blocks, four residual feature additions, and a
Stage-C subpixel head. Its seven serial 3x3 convolutions require a seven-pixel
LR halo. This differs from both the original three-convolution width-24 graph
and the retired separately staged Progressive A+B+C candidate.

RKNN FP16 is frozen before compilation. This deliberately avoids introducing
a second independently calibrated INT8 approximation into a topology-transfer
test after that failure mode was observed in the retired candidate. Precision
is therefore part of the named replication cell; raw FP16-versus-INT8 speed or
quality comparisons are prohibited.

The six existing synthetic compact/fragmented mask pairs, coordinate fixture,
two dispatch modes, and paired analysis are retained. The only workload change
is the graph-required halo. Eleven static graphs are required: eight ordinary
core shapes and top/middle/bottom full-width boundary variants.

Before timing, all eleven graphs must load and every mask/mode must pass two
checksum-stable qualification repetitions. Selected-output MAE versus the
frozen FP32 reference must be at most 0.001, cross-mode PSNR delta at most
0.02 dB, and unselected output change exactly zero. A target numerical or load
failure retires V22. No precision, compiler, optimization, or candidate sweep
is allowed after target output is observed.

If qualification passes, two warmups per static graph precede ten randomized,
counterbalanced paired blocks (120 measured samples). Every result is retained;
there is no outlier removal or replacement. The result may support only a
named-graph, named-device, named-runtime topology/precision replication claim.

