# September 25 follow-up outcome: transfer published locally; packing screened

The completed transfer study is unchanged and independently raw-audited. Its
new table, manuscript source, rendered 11-page PDF and minimal reviewer package
are installed in `C:/Research/Paper 3`. The reviewer repository has 87 hashed
files plus its manifest; full offline reproduction and exact ESM equality pass.
Nothing has been committed, pushed, uploaded or submitted.

## E001: packing numerical qualification

Both RK3566 models completed 17 retained output checks (reference plus eight
canvas shapes in forward/reverse order). Local compressed-file hashes, arrays,
same-backend error, order invariance and recorded controls pass independent replay.
FP16 matches the exact path byte-for-byte; width24 maximum full-output MAE is
0.00044844. Reversed packing order is byte-identical in both. Original-adapter
agreement and FP16 full-reference quality are recorded target attestations;
the local E001 audit does not recreate those references. Width24 full-reference
quality and full repeatability remain gates before any confirmation.

## E002: bounded development headroom screen

Eight old development masks: four families, selected counts 12 and 36. Eleven
organizations per graph, one observation each: exact, row span, contiguous-run
grouping, and eight conventional shelf-canvas sizes. All 176 observations pass
selected-output MAE <=0.001 and byte-identical unselected fallback checks against
the original exact implementation. All raw arrays and journals were retrieved
and replayed locally. Fixed model residency and initialization/warmup are recorded;
planning, fallback, allocation/copies, inference and composition are timed.

| Mean complete service, ms | Width24 | FP16 |
|---|---:|---:|
| Exact tiles | 189.65 | 209.75 |
| Row spans | 140.72 | 175.81 |
| Contiguous grouping | 137.60 | 169.13 |
| Best fixed packing size | 149.11 | 182.15 |
| Hindsight best packing per request | 145.84 | 173.44 |
| Hindsight grouping-or-packing per request | 133.88 | 164.80 |
| Hindsight best conventional exact/span/grouped | 132.77 | 163.60 |
| Hindsight best of all eleven | 130.40 | 162.20 |

The best fixed packing sizes differ: 640x60 and 576x60 core shapes. All eight
size results remain in the audit JSON, not only the winners. Packing wins over
grouping on 3/8 width24 and 4/8 FP16 masks, mostly scattered/mixed requests.
It is wrong to conclude that packing *never* helps. But even the hindsight
group-or-pack hybrid improves these observed grouping means by only 2.70%/2.56%,
without paying for profiling or selection. Adding packing to the hindsight
best-of-three conventional set changes its observed mean by about 1.78%/0.86%.
These are descriptive oracle calculations, NOT deployable policies, inference
confidence bounds, or upper bounds on other implementations' performance.

## Decision and unresolved work

Stop before confirmation of this candidate. The original purpose was to find a
consequential unexplained trade-off beyond conventional grouping/packing, not
to manufacture another small average improvement. This screen does not establish
that contribution. Per-row profile-guided choice and a deployable profiled-packing
policy were not run; stronger conventional choices could narrow the advantage.
This is a deliberate down-selection, not completion of all proposed comparisons.

Limitations matter: only eight development masks, one timing each, one board,
ordinary governor variation, and a conservative implementation that sends true
image-boundary tiles through separate exact calls. Better boundary-aware packing
or lower host overhead could change rankings. No comparison with RegenHance's
actual implementation is claimed, and no universal NPU limitation follows.
Routine packing/context techniques remain prior art; these engineering outcomes
do not establish algorithmic novelty.

The current paper's confirmed transfer result remains valid within its declared
exact-versus-span action set. It must not claim optimal dispatch or superiority
over packing/grouped alternatives. Those limitations already remain explicit in
the manuscript. E001/E002 are retained engineering evidence, not inserted as a
new confirmed performance table in the manuscript or its reviewer artifact.

## Reproduce engineering audits

From the main project with the collected `output/paper3_packing_v1` arrays:

```
python -B scripts/paper3_packing_v1/audit_e001.py
python -B scripts/paper3_packing_v1/audit_e002.py --cell width24
python -B scripts/paper3_packing_v1/audit_e002.py --cell fp16
```

Sources and plans are sealed separately in E001/E002 source manifests and board
folders. Outputs are append-only. No prior transfer samples were altered.
Both owned board workers exited and no experiment remains running.
