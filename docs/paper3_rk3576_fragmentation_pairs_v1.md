# RK3576 Fragmentation Pairs V1 Preregistration

> **ARCHIVAL PRE-CAPTURE PROTOCOL.** The remainder of this document preserves
> the frozen preregistration and its historical access-phase wording. It is not
> the current experiment status. One authorized capture completed without
> retry; the immutable raw record is
> `results/rk3576/paper3_fragmentation_pairs_v1/paper3_rk3576_fragmentation_pairs_v1_raw.json`
> (SHA-256
> `7657b5a5ec0abe62cb693dffab97a832ebd777348cc00aca90b45630b12b535b`),
> and the current disposition is recorded in
> `results/tables/paper3_rk3576_fragmentation_pairs_v1.json` and
> `results/manifests/paper3_rk3576_fragmentation_pairs_v1.json`. No second run
> is authorized by this archival document.

Experiment ID: `paper3_rk3576_fragmentation_pairs_v1`

State: **GO / REQUIRED_FOR_FULL_PAPER / AUTHORIZED_FOR_ACCESS**

This implementation-ready preregistration is authorized for one physical
capture after every runtime gate below passes in a clean authorization
checkout. No sealed or opened trace has been read, evaluated, or used as an
input, and no physical result exists yet.

## Purpose

The current fifteen-mask table is observational: density, active rows,
horizontal runs, source frame, and content vary together. This experiment
isolates geometry. At each of 20%, 40%, and 60% selected output, it compares a
compact mask with a maximally fragmented mask while holding selected count, six
active rows, model, input pixels, device, mode, deadline, and repeat block fixed.

There are three geometric mask pairs and two execution modes, yielding six
mode-stratified causal pair cells. Five repeats of both members of every cell
produce exactly 60 measured samples.

## Frozen Stimuli

The source of truth is
`configs/paper3_rk3576_fragmentation_pairs_v1.yaml`. Bits are row-major over a
10-by-6 grid; each listed string is one left-to-right row.

| Pair | Geometry | Row repeated six times | Selected | Runs | Row-span coverage |
|---|---|---|---:|---:|---:|
| 20% | compact | `0000110000` | 12 | 6 | 12 tiles |
| 20% | maximally fragmented | `1000000001` | 12 | 12 | 60 tiles |
| 40% | compact | `0001111000` | 24 | 6 | 24 tiles |
| 40% | maximally fragmented | `1001001001` | 24 | 24 | 60 tiles |
| 60% | compact | `0011111100` | 36 | 6 | 36 tiles |
| 60% | maximally fragmented | `1010101011` | 36 | 30 | 60 tiles |

For six selected cells in ten columns, five runs per row is maximal because
four zeros can separate at most five one-runs. Every fragmented member also
touches both horizontal extremes so row-span coverage is exactly ten tiles per
active row. No row is copied, selected, or derived from a sealed trace.

All masks use one deterministic 640-by-360 RGB input generated in memory:

- `R=(3x+5y+17) mod 256`
- `G=(7x+11y+29) mod 256`
- `B=(13x+2y+47) mod 256`

The row-major HWC byte stream must hash to
`1ac106cdfffeb3a232f2ff3e5a8ffbdd854ec8f9271be34c9f55d6e6cc93f18b`.
This fixture supports timing and selected-output integrity only; it supports no
image-quality claim.

## Hypotheses

1. **Structural padding (primary):** at every density, row-span fragmentation
   increases dispatched static coverage, executed MACs, and tensor bytes per
   committed pixel.
2. **Physical service (primary):** row-span fragmented-minus-compact service is
   positive at all three densities and has a median paired penalty of at least
   5% at two of three.
3. **Independent-tile negative control (primary):** fixed selected count gives
   identical structural MAC/byte totals, and the absolute median paired service
   difference is at most 5% at every density.
4. **Mode interaction (secondary):** the row-span service penalty exceeds the
   exact-mode penalty by at least five percentage points at two densities and
   is positive at all three.

Deadline success is descriptive. No p-value, confidence interval, regression,
or model-family extrapolation is authorized at five repeats.

## Execution Schedule

- Modes: `exact`, `row_spans`.
- Repeats: five measured repeats per mask/mode.
- Warmups: two per compiled static shape, excluded from the sample count.
- Design: five deterministic randomized complete blocks.
- Each block contains all twelve mask/mode samples.
- Within each `(repeat, density, mode)` cell, compact and fragmented are
  adjacent; seed `357603` determines cell order and pair orientation.
- Pairing key: `(repeat_index, pair_id, dispatch_mode)`.
- No outlier deletion, replacement sample, adaptive ordering, or post-access
  retuning is allowed.

The Paper-3-specific runner materializes and archives the exact 60-entry
schedule before the first warmup. It accepts no dataset, trace, policy,
routing, checkpoint-replacement, or model-conversion input. The checked-in
contract is `AUTHORIZED_FOR_ACCESS`; the command below remains fail-closed
unless the clean-checkout, accepted-baseline, board/runtime, exclusive-session,
AC-power, no-other-workload, reviewer-signoff, storage, and thermal gates pass:

```text
python -m src.deployment.benchmark_rk3576_fragmentation_pairs \
  --contract configs/paper3_rk3576_fragmentation_pairs_v1.yaml \
  --output results/rk3576/paper3_fragmentation_pairs_v1/paper3_rk3576_fragmentation_pairs_v1_raw.json \
  --require-status AUTHORIZED_FOR_ACCESS
```

The runner refuses any status other than `AUTHORIZED_FOR_ACCESS`, any output other than the exact
preregistered raw-capture path, an existing output, or a nonempty output root.
It records and continuously checks the pinned board/runtime identity, CPU
governors, NPU frequency controls, storage, and thermals. A failure record is
preserved without replacement samples or reuse of this experiment ID. This
authorization permits exactly one run: any stopped or failed attempt is not
retried under this experiment ID.

The non-authorizing offline preflight command is:

```text
python -m src.deployment.prepare_paper3_rk3576_fragmentation_pairs \
  --contract configs/paper3_rk3576_fragmentation_pairs_v1.yaml \
  --catalog results/models/sparse_refinement_rknn_v2/catalog.json \
  --output results/preflight/paper3_rk3576_fragmentation_pairs_v1.json
```

That record contains fixture/mask/schedule hashes and accepted graph hashes,
but it keeps `authorized_for_samples=false`, executes zero warmups and zero
samples, and leaves exactly `ssh_authentication_not_available` and
`exclusive_session_ac_power_and_reviewer_signoff_not_attested` unresolved. An
offline generator cannot attest either physical gate.

## Required Sample Record

Every record must include block/repeat and execution order, exact mask identity,
density/count/rows/runs/coverage, mode, physical NPU calls, executed GMAC,
input/output tensor bytes, bytes per committed HR pixel, extraction/planning/NPU
inference/reconstruction/service times, deadline overrun and success, committed
pixel count, selected-output checksum, and start/end thermals. Record raw values
for every repeat; do not keep only aggregates.

## Frozen Decision Rules

### Integrity gate

- exactly 60 measured records and a complete 6-mask x 2-mode x 5-repeat matrix;
- checkpoint, catalog, input, masks, runtime, board, and accepted repository
  commit match their pinned identities;
- all timings are positive and finite;
- pair members have identical selected and committed counts;
- no unselected output is committed;
- cross-mode selected output mean absolute float difference is at most 0.001
  and PSNR delta is at most 0.02 dB.

Any integrity failure makes the run unusable.

### Causal gates

- all three row-span fragmented members must exceed compact members in executed
  GMAC and tensor bytes per pixel;
- exact-mode GMAC and tensor bytes must be identical within each pair;
- row-span paired service deltas must be positive at all densities and at least
  5% at two densities; and
- exact-mode absolute median paired service deltas must be no more than 5% at
  every density.

If structural gates pass but the physical service gate fails, publish the
adverse result and claim only deterministic padding/traffic causality. If the
exact negative control fails, make no geometry-causal timing claim.

## Device and Runtime Controls

Use one exclusive RK3576 session, NPU core setting `both`, AC power, stable
cooling, and no other user workloads. The accepted runner must capture board
identity, device-tree compatibility, OS, kernel, Python, RKNN runtime library
hash, CPU governor, NPU frequency, thermals, catalog hash, every RKNN binary
hash, and checkpoint hash.

The preregistered reference environment is RK3576/aarch64, Debian 12, kernel
`6.1.84-13-rk2410-nocsf`, Python 3.11.2, and RKNN Toolkit compiler 2.3.2. Access
authorization pins board identity
`928996c84e65dc3f119cd65f232d2ad6b4d4acd12a0d76ed7b4ecdec016b199c`, RKNN
runtime-library SHA-256
`cf6ea624489225ddc9d334b370c64083b0fd9d6a5c9166eed47d999cc85a806b`, and
accepted implementation baseline
`6adca329ef335b6d4b4bcc17412e373eb98c9f9d`. Any mismatch remains a stop
condition before measurement.

`accepted_repository_commit` denotes the independently accepted
**implementation-baseline commit**, not the later authorization commit. The
authorization checkout must be clean and descend from that baseline. Between
the baseline and authorization checkout, changes are restricted to the
contract, this preregistration, and the zero-sample preflight record; within
the contract, only the explicitly enumerated authorization fields may change.
The runner verifies every runner, analyzer, manifest builder, shell entry
point, focused test, and reused measurement primitive against its Git blob in
the baseline. It refuses a dirty checkout, non-descendant history, any other
changed path or contract field, and any executable/test drift.

Quantization is not an operator-controlled variable in this experiment. The
runner exposes no output-scale or output-zero-point option and invokes the
accepted sparse backend with catalog-native behavior (`None`/`None`) for both
values. Adding or changing either would require a new experiment contract; it
cannot be authorized as metadata.

Start each block at or below 65 C. Stop at 85 C, preserve the incomplete record,
and do not replace samples. Cool to at most 65 C before another block.

## Stop and Failure Rules

Stop before the first physical sample if any access, identity, hash, schedule,
no-overwrite, or thermal gate is unresolved. Stop on missing metrics, nonfinite
timing, numerical-integrity failure, unselected commit, runtime exception, or
thermal abort. Do not delete failed records or retry with the same experiment
ID. Diagnose offline, define new thresholds only from non-result engineering
evidence, preregister a new ID, and seek fresh authorization.

## Expected Outputs After Authorized Physical Capture

- `results/rk3576/paper3_fragmentation_pairs_v1/paper3_rk3576_fragmentation_pairs_v1_raw.json`
- `results/tables/paper3_rk3576_fragmentation_pairs_v1.json`
- `results/tables/paper3_rk3576_fragmentation_pairs_v1.md`
- `docs/paper/generated/paper3_rk3576_fragmentation_pairs_v1.tex`
- `results/manifests/paper3_rk3576_fragmentation_pairs_v1.json`

All five remain absent before physical capture. Only an independently accepted
compact summary/table and hash manifest may later enter the Paper 3 publication
folder.

## Exact Access Checklist

- [x] Orchestrator authorizes experiment ID
  `paper3_rk3576_fragmentation_pairs_v1` and changes the contract state through
  an independently reviewed write.
- [x] The implementation baseline containing the contract, runner, analyzers,
  manifests, shell entry point, focused tests, and reused measurement
  primitives is independently accepted and pinned as
  `6adca329ef335b6d4b4bcc17412e373eb98c9f9d`.
- [ ] The authorization checkout is clean, descends from that baseline, changes
  only the enumerated authorization metadata, and passes every baseline Git
  blob check.
- [x] Board identity and RKNN runtime-library SHA-256 are pinned before access.
- [ ] Exclusive RK3576 reservation, AC power, cooling, storage, and authenticated
  shell access are confirmed.
- [ ] Checkpoint, catalog, RKNN binaries, input bytes, mask rows, and 60-entry
  schedule pass offline hashes.
- [ ] Runner source contains no sealed-trace manifest, observation-log, selector,
  routing, or model-retuning path.
- [ ] Output root is empty and the runner refuses overwrite.
- [ ] Two warmups/shape, five repeats, modes, core, seed, thresholds, and stop
  rules match the preregistration.
- [ ] A second reviewer signs the preflight record before the first warmup.

Publication value: this is the smallest controlled physical study that removes
the density/content confound and gives Paper 3 a clearly owned causal result.
Estimated elapsed effort after access is 4--7 hours. Current action: **WAIT FOR
CLEAN AUTHORIZATION COMMIT AND INDEPENDENT RUNTIME-GATE ATTESTATION; THEN ONE
RUN WITH NO RETRY UNDER THIS EXPERIMENT ID**.
