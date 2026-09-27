# E002 — conventional packing headroom screen, development only

2026-09-25, fixed before this screen's timings. Existing E001 has completed.
Use eight already-open development masks: replicate 0, counts 12 and 36 in each
of compact/scattered/clustered/mixed. No transfer evaluation masks are consumed.
On each graph compare exact, row span, contiguous-run grouping and all eight
E001 shelf canvases, one observation each, deterministic rotated order. Keep
all results, not just winners. No significance, stable speedup, or transfer claim.

All loaded static graphs remain resident as in the accepted runtime. Initialization
and warmup are outside service and reported separately. Service includes planning,
bicubic fallback, allocation/copying, dispatch, unpacking and reconstruction.
Runtime output/FD checks, numerical checks and recording follow the timer equally
for all methods; implementation-specific validation inside each service remains
charged. No tracing/logging influence bound is claimed for this engineering screen.

All outputs are retained losslessly by content identity. Compare selected pixels
against the original accepted exact path at MAE <=0.001, and unselected pixels
byte-exactly against the clipped bicubic baseline. Grouped paths are additionally
checked offline by toy tests; E001 covers actual packed-canvas placement semantics.
These are NOT complete confirmatory numerical gates: full-reference quality and
repeatability requirements still apply before any held-out performance experiment.

Safety/source/ownership, 60 C starting and 80 C abort gates, 180-second owned-worker
watchdog and sealed source manifests match E001. No overwrites/retries. Sequential
graphs only. Stop on numerical failure and retain the failed output and metrics.

Decision: comparison with exact/span/grouped is only an initial necessary screen.
An after-the-fact best canvas is an ORACLE, not a deployable profiled policy. If even
that is dominated over these requests, do not automatically run confirmation.
If useful headroom exists, next add calibrated per-row choice and profiled packing,
account for profile cost, complete numerical qualification, then reserve and freeze
new requests/sessions. No acceptance/novelty claim follows merely from packing.
