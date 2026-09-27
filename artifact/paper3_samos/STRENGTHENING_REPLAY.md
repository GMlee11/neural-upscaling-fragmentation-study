# Complete-service and reconstruction-placement confirmation

This compact package adds the separately frozen E002 confirmation. It does not
replace or pool the earlier geometry, dispatch, or transfer experiments.

The experiment compares six fixed execution organizations, each with whole-tensor
or selected-output host reconstruction, plus the actually executed frozen chooser.
Sixteen reserved paired masks are evaluated on two existing graphs, in two fresh
sessions and two repetitions: 1,664 evaluation requests. Each graph/session also
has one retained reference and 48 numerical qualification requests.

## What is included

`results/paper3_strengthening_v2/public_v1/` contains:

- Exact complete timing/control journals, not just favorable summary rows.
- The accepted source manifest, method, profiles, coefficients, exclusion catalog,
  masks, schedule, runner, and frozen implementation source.
- Original development/calibration journals, kept separate from evaluation.
- The completed saved-record audit and four retained-pixel numerical reports,
  bound to exact journals, sessions, source and reference identities.
- Additional append-only reservation/failure context when listed in the projection.
- A file-level projection manifest and reproducible expected replay output.

The projection explicitly omits neural models, vendor runtimes, raw output tensors
and reference arrays. Recorded output hashes are preserved, but their presence is
not equivalent to independently recomputing the omitted pixels. The numerical
reports document the author's separate private retained-pixel audit; compact
replay does **not** repeat that numerical validation.

## Replay

From the artifact root with the documented Python dependencies installed:

```powershell
python -B -m artifact.paper3_samos.build_strengthening_projection replay results/paper3_strengthening_v2/public_v1
```

This verifies the exact package and source, complete sample/control sequence,
request coverage and ordering, frozen costs, independently reconstructed job
plans and chooser tie rules, reported stage boundaries, and recorded output-hash
agreement for matched reconstruction controls. It recomputes all planned practical
gates, descriptive paired bootstrap, prediction errors, regressions and setup-cost
amortization. It performs no inference, fitting, new experiment or board access.

The full artifact `reproduce.ps1` additionally regenerates both service and
geometry tables from the exact saved-record audit, tests mutations with synthetic
records, and checks that the frozen release has not changed. The table outputs are
under `docs/paper/generated/paper3_strengthening_*_v1.*`; these are table fragments, not a
manuscript or preprint.

## Interpretation boundaries

The frozen method and masks are inspectable; hashes bind their recorded bytes,
not independently attest historical timestamps or preregistration. The original
source uses an explicit binary64 left fold for calibrated scores, preserving the
board's Python 3.11 arithmetic when replayed in Python 3.12.

NPU owner proofs are two point-in-time helper sweeps before/after a sample, not
continuous exclusion or an independently persisted backend-worker identity.
Two sessions and eight matched pairs provide bounded empirical evidence, not
validated population confidence coverage. Practical gates are not statistical
significance tests, proof of novelty, or journal-acceptance probabilities.
