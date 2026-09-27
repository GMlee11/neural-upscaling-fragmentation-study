# Target-unfitted transfer supplement — September 25, 2026

Run `python -B -m src.report.paper3_transfer_publication_v1` from the repository
root. It verifies the frozen donor coefficients, all 5,120 projected request
identities and decisions, complete session coverage, timing summaries and the
generated transfer table. `tests/test_paper3_transfer_publication_v1.py` includes
mutation rejection checks. This command is also in the artifact reproduction
script. No board access or new inference is involved.

Input: `results/manifests/paper3_transfer_public_v1.json`.
Outputs: `results/tables/paper3_transfer_publication_v1.json` and
`docs/paper/generated/paper3_transfer_publication_v1.tex`.

The donor was fitted to 240 RK3576 width-24 development requests, costing
27.24 seconds of summed service. No target fitting was permitted. Forty masks
were reserved after excluding prior geometries; the same masks were subsequently
used for graph/device replications. All four cells were historically studied.
This is not a previously unseen model-family or cross-vendor claim.

Original dated protocols and the thermal amendment are in `docs/paper3_transfer_v1/`.
The first width-24 RK3576 session used a 55 C start/block gate; after a zero-sample
FP16 temperature stop, remaining sessions used the documented 60 C amendment.
The 80 C abort was unchanged. Original failures are not replaced or pooled.

Mean savings are 2.59%, 2.72%, 4.46%, 3.55% in RK3576 width-24/FP16 and RK3566
width-24/FP16 order. Both sessions in every cell improve, but 132, 125, 118, 130
paired requests regress. All policies miss 33.33 ms. Decision overhead is included.
The posthoc best fixed policy and paired minimum are comparison summaries, not
deployable or noise-free oracles. Full donor-calibration break-even counts assume
observed savings persist and exclude loading and numerical qualification costs.

Compact replay validates reported timing/decision evidence and retained audit
identities. It does NOT repeat the original raw-tensor, device-control or RKNPU
ownership audit. Model binaries and full raw arrays remain outside this release.
The separately developing packing baseline is not a result in this manuscript.
