# Paper 3 Self-Contained RKNN Fragmentation Summary

This report uses only Paper 3 physical RKNN measurements and the static graph shapes actually dispatched.
It imports no FPGA counters, companion-paper results, or external accelerator denominator.

| Metric | Result |
|---|---:|
| Masks | 15 |
| Average exact launches | 20.000 |
| Average row-span launches | 3.933 |
| Row-span launch reduction | 80.33% |
| Row-span dispatched-core increase | 34.33% |
| Row-span tensor-byte increase | 24.50% |
| Row-span mean-service reduction | 10.96% |
| Exact mean-deadline success | 13.3% |
| Row-span mean-deadline success | 6.7% |

Across the frozen 15-mask RK3576 replay, row-span dispatch reduces static launches and mean service time while increasing dispatched core area and tensor traffic. These are within-backend organization effects.
