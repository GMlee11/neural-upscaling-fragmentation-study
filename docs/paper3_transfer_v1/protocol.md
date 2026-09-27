# Paper 3 transfer V1 — prospective protocol

Date: 2026-09-24. Author authorized the bounded transfer study. This is a new
experiment, not a rerun, replacement or reopening of prior evaluation traces.
Engineering qualification may be repaired transparently before measurement.

## Question and information boundary

Can an unchanged launch-versus-padding rule, fitted only on the original
RK3576 width-24 development calibration, improve complete service on fresh
requests on another graph and/or device without target-specific fitting?

All four graph/device cells were previously studied. Their published results
informed the motivation for this study. **No entire cell is historically unseen.**
The withheld object is target calibration in this rule, together with new mask
identities and new physical sessions. Do not claim blinded graph discovery,
cross-vendor generality or a new latency-prediction algorithm.

One donor is selected now by its role as the original baseline, not by comparing
donors' target performance: RK3576 width-24 INT8 calibration, report SHA-256
`c9bfae357925d64dd9e566b8ed0e52fc738bb36e1910b94f6ae4d346f73e0107`.
Use its five coefficients unchanged, in the existing feature order:
`[25.058020696584993, 1.7210469904490526, 544.3142267347189,
8.148333720198874e-05, 0.0]`.
Target timing/calibration is not an input to fitting, donor choice, thresholds
or decision logic. No target refit, rescaling or calibration is allowed.
Numerical qualification is allowed but its times cannot tune this rule.

The score is a transferred decision score, not a validated prediction of
absolute milliseconds on a different graph/device. Intercept and committed
pixels cancel between modes. In effect, choose row spans when saved launch
cost exceeds added dispatched-core and host-tensor cost. Exact wins ties.
The implementation keeps the previous `calibrated` policy key for compatibility;
in this experiment that key ALWAYS means **donor-calibrated, target-unfitted**.

## Workload and freeze order

1. Hash this protocol, the fixed donor report, generator, decision, analysis and
   numerical code before generating the new masks.
2. Generate once using seed 20260924031, with order seed 20260924032. Same
   pre-existing generator: compact, scattered, clustered, mixed; 6/12/24/36/48
   selected tiles; two masks/stratum = 40 masks. Exclude exact matches to the
   previous 40 development, 40 dispatch evaluation, 15 descriptive masks and
   the published paired-characterization masks. No performance-based rejection.
3. Generate numerical references without opening performance results. All masks
   must qualify; no dropping failures. Freeze exact source/model/runtime/input,
   tensor metadata, reference, controls, schedules and analysis before capture.
4. Engineering uses ONLY predecessor development masks and disjoint outputs.
   Fresh evaluation mask qualification may inspect output correctness, never
   select coefficients, masks or operating settings by its service times.

All cells reuse the same fresh geometry set. Later-device observations are
controlled transfer replication, not independently sampled geometry holdouts.
No sealed visual-quality or failed-selector traces are accessed.

## Cells and controls

Primary target: seven-convolution FP16 on RK3566 (graph/precision and device
both differ from donor). Secondary: FP16 RK3576 and width-24 RK3566. Width-24
RK3576 is donor-domain fresh-request replication, not transfer.

Two fresh process sessions/cell, each four blocks, 40 masks x four policies =
640 requests/session; 5,120 requests in the four-cell study. Each mask receives
each policy position once/session. A new process/session is not a new independent
device/day. Report session differences; do not claim robust cross-day behavior.

Policies: always exact, always row-span, unchanged >=2x launch / <=25% padding
geometry heuristic, donor-calibrated transferred score. Use accepted compiled
graphs, widths, halos and boundary handling; no compiler search or new training.
Use the original complete-service boundary INCLUDING decision, planning,
extraction, inference, reconstruction and selected-output commit. Preserve
unselected bicubic output. Keep all per-request numerical gates (MAE <= .001,
PSNR difference <= .02 dB, repeated output identity), with the existing width-24
cross-mode versus FP16 per-mode ONNX-reference distinction.

Normal governors, stable runtime/control identities, two warmups per graph,
start/block temperature <=55 C, maximum <80 C, cooldown <=300 s, owned-worker
timeout 180 s. Observe and durably retain controls before rejecting them. No
unrelated process termination, display-service changes or governor adjustments.
Occupancy must pass the installed read-only privileged helper, not just pgrep.
Budget all uncompressed possible outputs plus 2 GiB before each session.
Content-addressed lossless outputs may deduplicate repeated identities.

## Analysis and stopping rule

Primary outcome: equal-mask/equal-session complete-service difference between
transferred policy and each fixed policy in the PRIMARY cell. Improvement
requires a lower mean than BOTH fixed policies; additionally state whether
the sign agrees in both sessions. A negative or mixed result remains valid.
No success threshold is imported from a favorable observed result, and no
population p-value or narrow confidence bound is justified by two sessions.

Report all four cells, all policies, absolute means, relative differences,
decision overhead, family/density results, paired regressions, max regression,
deadline outcomes, session differences and incomplete/failing samples. Compare
against the best fixed policy over the complete declared workload, not the
best policy per selected subgroup. Report chosen-mode mismatch and its cost
using paired fixed-mode observations as a **noise-sensitive posthoc reference**,
not truth, a deployable oracle or proof of every wrong decision.

The donor's historical calibration used 240 requests. Report that cost from
its record, as well as zero additional target calibration. Where average service
savings are positive, report break-even requests for amortizing donor calibration;
include the case where no break-even exists. Never hide calibration by calling
the entire method cost-free. Engineering/qualification cost is separately listed.

Every claimed evaluation session has one immutable destination. No replacement,
adaptive repeats, splicing, source edits or coefficient updates after capture.
An engineering defect or infrastructure failure is retained and assessed before
any separately registered successor. No automatic run-until-positive loop.
All prior V1/V22/V24/V32 and dispatch evidence remains unchanged.

## Novelty boundary / sources checked September 24

- [nn-Meter](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/05/nn-Meter-Mobisys21.pdf): hardware-aware latency prediction is established.
- [LitePred](https://www.usenix.org/system/files/nsdi24spring_prepub_feng-chengquan.pdf): transferable predictors with limited adaptation are established.
- [Welder](https://www.usenix.org/conference/osdi23/presentation/shi): tile traffic cost models and memory scheduling are established.
- [MobiSR](https://arxiv.org/abs/1908.07985): mobile super-resolution patch scheduling is established.

The possible new evidence is prospective practical use (or a measured transfer
failure) of this specific geometry/launch/padding characterization under unchanged
requested outputs on static RKNN interfaces. A simple fitted rule is not claimed
as an independently novel algorithm. Manuscript/artifact updates follow verified
physical outcomes only; the current submission candidate remains intact meanwhile.
