# User-authorized 60 C continuation — September 24, 2026

The first width24 session completed 640 requests under the original 55 C
start/block criterion. FP16 session 0 stopped before backend creation, warmup
or sampling at 55.461 C. The outer launcher had passed its earlier check.
This is a startup-check timing race, not evidence of inference failure.

The author explicitly requested continuation with a 60 C starting criterion.
55 C was an experimental starting-condition rule, not a manufacturer safety
rating. We do not assert that temperature alone guarantees identical frequency,
performance or safety. Keep normal governors, thermal telemetry and the existing
strictly-below-80 C operational abort check; never disable kernel protections.

## Prospective continuation, not replacement

Create the separate `paper3-transfer-v1-thermal60` workspace and hash-bound
bundle/approval. Run ONLY fp16/0, width24/1, fp16/1 in that order. Preserve the
completed width24/0 and the failed zero-sample fp16/0 in the original workspace.
No repeated successful samples, mask changes, fitting or outcome selection.
The startup check now waits up to 300 seconds when above the accepted limit,
instead of failing immediately after the launcher's separate check. Block
cooldown remains bounded by 300 seconds. Both use 60 C. Actual temperatures
remain recorded around every request, with the unchanged numerical/ownership
gates, sources, NPU models, coefficients, requests and service-time boundaries.

The new code changes only lifecycle/authorization and thermal handling; the
existing accepted engineering numerical qualifications are reused by their
exact journal hashes, with their original bundle provenance retained. Local
synthetic tests cover the changed gate, wait and retained abort conditions.

## Reporting

Always disclose the 55-to-60 C amendment and the original zero-sample failure.
The first width24 session is not a homogeneous-temperature replicate of the
second by protocol. Report both separately as well as the planned aggregate,
with actual thermal/frequency ranges. Both FP16 sessions use the new limit.
No causal effect of the temperature amendment can be inferred from these runs.
Continue the original rule without inspecting intermediate performance to tune
it. The failed initial session and all successful evidence stay archived.
RK3566 may use this starting envelope only in its separately frozen controls.
