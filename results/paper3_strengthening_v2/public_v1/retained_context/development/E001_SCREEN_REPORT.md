# E001 strong-baseline screen: what is established

Engineering development evidence, not fresh confirmation. All 890 retained
outputs across two existing graphs passed independent schedule/hash replay
and separate numerical replay. These requests were previously opened. The
cyclic execution order did not independently counterbalance reconstruction
modes, limiting causal attribution of mode differences.

## Complete service, not inference alone

Equal-observation mean service in milliseconds:

| Organization | Width24 whole | Width24 selected | FP16 whole | FP16 selected |
|---|---:|---:|---:|---:|
| Exact tiles | 119.974 | 112.115 | 126.721 | 119.383 |
| Row spans | 96.196 | 93.611 | 113.885 | 103.449 |
| Contiguous groups | 92.909 | 91.304 | 103.962 | 103.667 |
| Per-row choice | 94.902 | 91.805 | 110.851 | 99.583 |
| Affine partition | 95.964 | 95.052 | 110.897 | 108.307 |
| Profiled partition | 103.124 | 100.158 | 115.302 | 107.651 |

Sources: `E001_width24_v5_summary.json` and `E001_fp16_v5_summary.json`
in `results/paper3_strengthening_v2`, independently produced by the parent.
The stronger profiled partition planner did not produce a convincing win.
Its Python planning cost was approximately 6ms/request. Contiguous grouping
cost approximately0.3ms; per-row choice approximately1.4ms. Even unrealistically
subtracting all profiled planning time would not make that planner beat
grouping in three of four graph/reconstruction combinations.

Table-compiling a finite planner is an ordinary implementation improvement,
not a new algorithm. E002 includes that correction for fairness. Construction
and deployment costs must remain visible; small per-request savings do not
automatically repay preparation for short sessions.

## Outputs and numerical scope

- All72 fixed-policy whole/selected pairs per graph were byte-identical.
- Width24 meets the accepted selected-output aggregate MAE threshold0.001
  across organizations; it is not byte-identical between organizations.
- FP16 agrees exactly across same-backend organizations; selected-output MAE
  versus the separately pinned full reference is at most2.608e-5.
- Maximum per-tile error remains reported, not silently substituted for the
  previously accepted aggregate threshold.
- Fallback replay establishes cross-record consistency from complementary
  corner requests, not independently regenerated bicubic ground truth.

## Plausible explanation, not a demonstrated new predictor

Whole reconstruction materializes the returned residual, including halo and
unrequested core pixels. Selected reconstruction materializes only committed
tiles but invokes normalization more often. Exact tiles have no unrequested
core padding and still carry halo. Therefore any savings cannot all be
attributed to gaps between selected regions.

A conventional nonnegative call-plus-volume model was fitted solely to E001
qualification/profile records. Its development sign agreement was77.8% for
Width24 and88.9% for FP16; FP16 did not beat the trivial always-selected sign
prediction. It is not yet evidence for a novel or useful prediction rule.

The retained `inference_ns` interval includes RKNN host/runtime handling and
calls. It is **not pure NPU compute** or an intrinsic accelerator padding tax.
Original normalization already produced contiguous HWC data, so the observed
comparison is not against a deliberately weak strided-output implementation.

## Why E002 is bounded

E002 tests matched geometry, independent mode counterbalancing, optimized
conventional planners and the actually timed frozen chooser. It uses the same
graphs, input fixture and device; it is fresh geometry/session evidence, not
new application, input-content or vendor generality. Quantitative practical
gates were recorded before collection. Agreement, failures, near ties and
regressions must all remain.

Packing, dependency context, profiling, stitching and launch-versus-work
trade-offs already appear in RegenHance, SBNet, Mondrian and related work;
finite-shape NPU latency effects also have close prior art. The viable claim
is a sufficiently informative bounded empirical distinction, if E002 supports
it. Neither more samples nor a faster Python planner guarantees novelty or
a70% journal acceptance probability. If optimized conventional alternatives
explain the observations, retain that finding and stop algorithmic expansion.
