# Paper 3 RK3576 Fragmentation Pairs V1

Disposition: **PRIMARY_GATES_PASS**

| Density | Mode | Compact GMAC | Fragmented GMAC | Compact B/pix | Fragmented B/pix | Service penalty min / median / max | Fragmented second | Max paired start-temperature gap |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 20% | exact | 0.467027 | 0.467027 | 15.3398 | 15.3398 | -3.085% / -1.838% / 6.802% | 2/5 | 0.924 C |
| 20% | row_spans | 0.447011 | 2.154994 | 14.6824 | 70.7824 | 187.165% / 199.665% / 231.914% | 3/5 | 1.846 C |
| 40% | exact | 0.934053 | 0.934053 | 15.3398 | 15.3398 | -3.855% / 1.960% / 10.894% | 2/5 | 0.000 C |
| 40% | row_spans | 0.874007 | 2.154994 | 14.3537 | 35.3912 | 108.476% / 130.130% / 157.248% | 4/5 | 0.924 C |
| 60% | exact | 1.401080 | 1.401080 | 15.3398 | 15.3398 | -10.751% / -0.373% / 8.222% | 2/5 | 0.924 C |
| 60% | row_spans | 1.727998 | 2.154994 | 18.9191 | 23.5941 | 6.635% / 12.555% / 26.291% | 1/5 | 0.000 C |

## Paired repeat distributions

| Density | Mode | Repeat 1 | Repeat 2 | Repeat 3 | Repeat 4 | Repeat 5 |
|---:|---|---:|---:|---:|---:|---:|
| 20% | exact | -3.085% | -1.838% | 2.051% | 6.802% | -2.559% |
| 20% | row_spans | 202.397% | 231.914% | 189.073% | 187.165% | 199.665% |
| 40% | exact | 3.984% | 0.704% | 1.960% | 10.894% | -3.855% |
| 40% | row_spans | 108.476% | 147.696% | 123.625% | 130.130% | 157.248% |
| 60% | exact | 8.222% | -7.073% | 3.307% | -10.751% | -0.373% |
| 60% | row_spans | 22.948% | 12.555% | 6.635% | 9.795% | 26.291% |

## Descriptive order and temperature audit

All 15/15 row-span paired service deltas are positive. Both compact-first and fragmented-first orders occur at every density. Start temperature spans 45.307--50.846 C; the maximum within-pair start-temperature difference is 1.846 C.

This audit is descriptive only: the capture has no per-sample timestamps, temperature was not independently manipulated, order was frozen, and all records come from one physical session.

Authorized interpretation: For this one RK3576, one model, one synthetic input, and three frozen density pairs, within-row fragmentation causes additional static row-span padding and traffic and has the reported physical service effect.

No p-value, confidence interval, regression, selector-generalization, energy, 720p-real-time, platform-superiority, or cross-platform bit-identity claim is made.
