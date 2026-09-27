# Paper 3 RK3566 replication protocol v24

Status: **frozen instrumentation correction; no numerical result retained and
no performance sample collected**.

V23 successfully initialized all eleven exact RK3566/FP16 models and reached
the first inference, but its NPU-authentication gate watched only
`/dev/dri/renderD129`. The output was discarded before numerical comparison.
A value-blind, syscall-level witness then showed that RKNN Lite 2.3.2 opens
`/dev/dri/card1`, retains that descriptor, and issues successful RKNPU DRM
allocation, submission, synchronization, and buffer ioctls. Both `card1` and
`renderD129` resolve through sysfs to `/sys/bus/platform/drivers/RKNPU`.

V24 changes only that observed descriptor path. It consumes the exact V23
compiled catalog, V22 reference payload, V22 masks, V22 schedule, FP16
precision, RKNN 2.3.2 runtime, driver 0.9.8, numerical thresholds, two
qualification repetitions, two warmups, and 120 counterbalanced samples.
There is no graph, input, mask, candidate, precision, compiler, or stopping-rule
change. The qualification and timed run execute without syscall tracing; the
ordinary runner directly retains `/dev/dri/card1` and verifies its RKNPU driver
mapping before accepting output.

The rootless private mount namespace exposes the hash-pinned
`librknnrt.so` at the absolute path required by RKNN Lite without changing the
board OS or broadening its intentionally narrow sudo policy. It adds no
emulation or tracing to the measured path.

All numerical failures remain terminal. Timing remains forbidden until every
qualification gate passes. If qualification passes, one 120-sample run is
performed with no replacement, outlier removal, or tuning. V23 and its
pre-output dependency failures remain archived rather than overwritten.
