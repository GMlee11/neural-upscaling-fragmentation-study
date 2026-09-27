# Frozen V32 width-24 RK3566 result

Current state: qualification and capture completed by the operator; offline
analysis passes the frozen primary gates. The pre-capture review candidate at
`configs/paper3_width24_int8_rk3566_capture_review_v32.json` and its review note
are preserved archival records, not the current outcome or permission to rerun.

Capture SHA-256: `1b3236b61485e0e5a2d72c9c96233fd1deaa2a11882ee33c0d823eafa025c98d`
(4595914 bytes); qualification SHA-256:
`fb20fee93691ffef22fca4913307af4f858aca26c29b2f084577841a3dbc9fd1` (1769905 bytes).
Both complete hash chains pass with final success and no fallback. Capture has
137 events, 16 warmup calls, 60 samples and five cross-mode records; qualification
has 55 events, 24 numerical records and zero warmups/timing samples.

## Within-backend comparison

Median fragmented-minus-compact service percentages; cells are never pooled.

| Graph/precision and target | Paired blocks | 20% | 40% | 60% |
|---|---:|---:|---:|---:|
| Width-24 INT8 RK3576, V1 | 5 | 199.665 | 130.130 | 12.555 |
| Width-24 INT8 RK3566, V32 | 5 | 181.937 | 102.907 | 4.145 |
| Shared-feature seven-conv FP16 RK3576, V22 | 10 | 156.857 | 73.339 | 12.252 |
| Shared-feature seven-conv FP16 RK3566, V24 | 10 | 115.955 | 68.360 | 7.186 |

Sources: frozen V1 result table and V22/V24 analysis manifests. Their repeated
distributions are unchanged. V32 exact medians are 0.542%, 0.439%, -0.138%; all
15 row-span deltas are positive. Row-span ranges are 149.930--188.383%,
97.516--112.682%, 3.400--16.192%. The 60% median is below 5%, but the frozen
rule requires two densities >=5%, so it passes without revision. Exact launch
counts and input/output bytes match within each pair. Row-span tensor inflation
is 382.090%, 146.565%, 24.710%, with equal selected count and launches.

V32 service spans 102.794971--314.567363 ms; 0/60 meet 33.33 ms. Absolute times
are descriptive, not causal device rankings: host tails, compiler/quantization
paths, runtime configurations, sessions and five-versus-ten repeats differ.
All observations are retained. Five repeats support disclosed medians/ranges,
not a fabricated CI or p-value. Post-sample temperature is observed, not
manipulated, and no per-sample wall-clock timestamps exist. Both pair
orientations occur at each density/mode. Full pairs/components are in the
V32 analysis JSON; the compact projection is deterministic reviewer input.

## Integrity, provenance and publication boundary

Maximum cross-mode MAE is 0.0008422387763857841 <=0.001; maximum PSNR delta is
0.011456164703950833 dB <=0.02. Selected checksums match qualification and all
unselected changes are zero. All 67 capture control snapshots pass identity,
affinity, thread and frequency-limit checks; temperatures span 41.875--48.333 C.
RKNPU driver is 0.9.8; RKNN 2.3.2 runtime SHA is
`d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8`.
Self /dev/dri/card1 presence and eight-graph tensor checks are enforced by the
accepted source. Independent raw self-FD/ioctl transcripts are not recorded.
Permission-limited process coverage is not whole-system exclusivity.

The reviewer projection excludes operator process/FD details, retaining all
samples, schedule, exactness and control summaries with raw hash/tail provenance.
Full journals stay historical; no graph/checkpoint/deployment/credential payload
enters Publication V1. V25/V29/V31 failures stay sealed. No selector, energy,
real-time, platform-superiority or universal NPU claim follows; Paper 2 remains
cited MAC/traffic context only.

No board was contacted by this analysis task. Once the operator confirms the
completed process exited and filesystem writes flushed, Orange Pi is no longer
needed for offline work. Use normal shutdown before power disconnection; this
offline audit cannot attest live shutdown state.
