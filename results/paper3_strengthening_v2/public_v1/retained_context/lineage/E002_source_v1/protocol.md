# E002 draft: matched geometry and host-materialization confirmation

**Executable protocol for root/reviewer prelaunch review. This document alone
does not reserve masks or start physical evaluation.** The once-only source
builder binds this document and all method sources before mask generation.
E001 remains opened development evidence.
The Radxa is idle and its desktop/login services have been restored.

## Finite scientific question

For unchanged graph, requested outputs and inference jobs, when does omitting
uncommitted residual materialization change the complete-service geometry
penalty or preferred organization? Does a frozen ordinary cost explanation
predict those boundaries better than launch/work-only accounting and constant
reconstruction choices on reserved matched requests?

This is an empirical question, not a claim that halo handling, crop-before-
PixelShuffle, dynamic programming, table compilation or affine cost fitting are
new. A uniform speedup, near-tie rank flip, or another 1--3% mean scheduler gain
does not by itself meet the requested novelty gate.

## Development facts and proposed mechanism

E001 qualified 890 retained outputs across two graphs. Simple grouping remained
strong. Python partition planning cost 5--6ms/request and is replaceable by
3072 immutable precomputed row decisions; exhaustive tests preserve the exact
old objective, ties and jobs. Construction/preparation cost must be recorded on
the actual board, along with serialized size (not falsely labeled resident RAM).

Returned residual elements decompose into selected core, unselected core, and
halo. Exact tiles have zero unselected core, but still materialize halo. Whole
reconstruction makes one normalization call per inference job; selected-only
makes one per committed tile. Thus less materialized volume can be offset by
more small host calls. The frozen conventional explanatory form is:

`normalization_ms = a * normalization_calls + b * materialized_elements / 1e6`.

Fit only the already recorded E001 qualification/profile jobs; do not refit
from evaluation. The present development fit is not a new predictor result:
its FP16 sign accuracy equals the trivial always-selected choice. Existing
E001 per-shape costs remain the stronger conventional comparator. No target
calibration or coefficient updating is allowed after evaluation starts.

## Requests: matched interventions, not unrelated random masks

Propose **eight pairs / sixteen masks**, two independently generated pairs for
each predeclared per-row-count profile:

1. Six active rows, two selected tiles each (12 total).
2. Three active rows, four selected tiles each (12 total), including top/bottom.
3. Six active rows, six selected tiles each (36 total).
4. Four active rows, nine selected tiles each (36 total), including top/bottom.

Each pair has identical selected count, active rows and per-row selected counts.
One member uses contiguous selected runs; the other uses noncontiguous columns
with a larger span, using fixed geometric rules independent of measured latency.
Keep finite-catalog cases where both members map to the same dispatched width:
they are useful predicted-zero controls, not exclusions. Preserve all pairs.

Freeze generator, seed, row placement, geometric constraints and exclusion
catalog before generation. Exclude named development/confirmation/transfer
catalogs, packing catalogs and E001 qualification/profile masks by exact six-row
identity. Save each source path/hash, geometry count and the complete union.
Non-overlap applies to the named checked catalogs, not all possible history. Reserve
once and fail rather than silently change seed if the finite generation bound
is exhausted. The two existing graphs use the same reserved geometries and
their accepted fixed input/reference; this is not a new-input or new-device test.

## Fixed evaluation design

Two separately initialized physical sessions; two repetitions per mask/session.
No reboot, clock/governor change, thermal threshold shopping, model compilation
or parameter fitting. Apply the previously approved reversible headless setup,
authenticated dual-node ownership, <=60C start and <80C operating limit.

The twelve existing controls are exact, row-span, grouped, cached per-row,
cached affine partition, cached lookup partition, each with whole or selected
reconstruction. Independently counterbalance reconstruction order within every
mask/policy across its four repetitions: exactly two whole-first and two
selected-first, with both orders in each session. Reverse graph order in the
second session (width24/FP16, then FP16/width24); separately
rotate policy and mask-pair ordering from a fixed seed. Never optimize order
from measurements. Retain all outcomes, including failures and regressions.

A thirteenth, separately timed `frozen_choice` selects among all twelve
configurations using the E001 per-shape additive job-cost estimate. Exact
candidate order is policy-major as listed above, whole before selected; first
minimum wins ties. No obsolete online-DP planning penalty is retained. Shared
fallback cancels from this objective; actual decision and planning costs are
included in measured complete service. Preparation and serialized storage are
reported separately, with amortization. Serialized bytes are not resident RAM.

Because lookup partition minimizes the same additive job objective over a
superset of each competing planner's row actions, it is normally predicted no
worse within each reconstruction mode. Other candidates matter at ties and
as measured controls; the chooser principally adapts reconstruction mode,
not a fundamentally richer or novel scheduling mechanism. Origin-specific
extraction and whole-request interference remain unmodeled. Call this a frozen
calibrated job-cost estimate, not exact predicted end-to-end latency.

The final design is **1,664 evaluation requests** (16 masks x13 configurations
x4 repetitions x2 graphs), plus one reference and48 numerical qualification
samples per graph/session (196 total): **1,860 retained samples**. Estimated
55--65 minutes plus fixed300-second idle between restored-service sessions;
actual table preparation and control collection can extend this. Stop after
the prescribed count; do not add repetitions based on the observed result.

## Analysis and decision

- First audit full schedule, source/models, controls, raw hashes and outputs.
  Within fixed-policy reconstruction pairs require identical jobs and bytes.
  Across organizations retain the accepted .001 aggregate MAE contract; width24
  is not byte-identical between organizations. Preserve max per-tile errors and
  FP16 agreement with its independently pinned full reference.
- For each matched mask pair/policy/session, report fragmented-minus-compact
  service and normalization costs under both reconstruction paths. Their
  difference separates the observed host intervention from the unchanged jobs;
  it does not identify pure NPU compute or all cache/governor mediation.
- Test the frozen call/element model against always-whole, always-selected,
  discarded-volume-only, and the established per-shape profile. Report sign
  error, absolute prediction error and wrong-choice service penalty, including
  near-zero cases rather than treating noisy signs as ground truth.
- For planner/chooser claims, compare complete service (including online
  planning/selection) against every fixed organization/reconstruction. Report
  preparation and calibration separately and their break-even request count.
  Independent sessions and paired request families are the units of evidence;
  repeated samples are not independent new geometries.
- Practical chooser gate, separately for each graph: at least5% lower pooled
  equal-mask mean service than the best of all12 fixed configurations, positive
  gain in both sessions, and no pooled per-mask mean regression above10% versus
  that graph's pooled best fixed. Report session-specific best-fixed controls.
  These thresholds are prespecified practical stopping criteria, not laws of
  novelty or statistical-significance tests.
- Primary geometry gate uses **row spans only**. For each family calculate
  `(fragmented - compact)_whole - (fragmented - compact)_selected` in service
  time, averaging its two pairs. A consequential interaction has magnitude at
  least max(2ms,5% of whole-compact service), the same sign in both sessions,
  and occurs in at least two predeclared families per graph. Exact and grouped
  are controls, not alternative policies selected afterward to pass the gate.
  A negative interaction is not a penalty reduction; report the actual sign
  and both constituent penalties. The predeclared equal-catalog-width pair is
  retained as a null-shape control regardless of its result.
- Uncertainty:5000 draws, seed20260925, stratified matched-pair bootstrap.
  Sample two pairs with replacement within each of four families, retaining
  both members/sessions/repeats; reselect the best of12 fixed configurations
  in each draw. Report2.5/97.5 percentiles as a descriptive sensitivity
  interval, **not** validated95% population coverage (only two pairs/family).
  Crossing zero blocks "robust gain" wording even if a practical mean gate
  passes. Do not call individual repeated samples independent geometries.
- A useful result needs a consequential, repeated geometry/ranking distinction
  and a prediction that survives reserved requests against strong alternatives.
  If ordinary optimized choices explain the behavior, or only near ties/noisy
  flips remain, stop this novelty candidate. Preserve the evidence and improve
  the bounded characterization paper without declaring a new algorithm or a
  guaranteed acceptance probability.
