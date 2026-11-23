# KPI triage results (800 test examples, 11 series)

Verdicts are balanced, so always answering one verdict scores 0.25 accuracy. Higher is better everywhere.
Intervals: 95%, resampling whole series. Full match = verdict, shape and start (within a tenth of the window) all right.

| System | Verdict macro-F1 | 95% interval | Verdict accuracy | Full match | 95% interval | Attention F1 | Unreadable |
|---|---|---|---|---|---|---|---|
| llama-text-finetuned | 0.820 | 0.738 to 0.885 | 0.819 | 0.715 | 0.580 to 0.825 | 0.919 | 0% |
| rules+llama-text-finetuned | 0.813 | 0.696 to 0.888 | 0.812 | 0.708 | 0.560 to 0.811 | 0.917 | 0% |
| rules+opentslm-finetuned | 0.796 | 0.697 to 0.871 | 0.796 | 0.688 | 0.564 to 0.780 | 0.917 | 0% |
| opentslm-finetuned-values | 0.786 | 0.729 to 0.831 | 0.782 | 0.591 | 0.479 to 0.693 | 0.924 | 0% |
| rules | 0.708 | 0.619 to 0.783 | 0.720 | 0.619 | 0.510 to 0.702 | 0.917 | 0% |
| opentslm-finetuned | 0.695 | 0.638 to 0.745 | 0.694 | 0.455 | 0.394 to 0.544 | 0.867 | 3% |
| note-only | 0.539 | 0.524 to 0.554 | 0.619 | 0.233 | 0.203 to 0.281 | 0.778 | 0% |
| series-only | 0.487 | 0.419 to 0.536 | 0.550 | 0.497 | 0.400 to 0.579 | 0.917 | 0% |
| majority | 0.100 | 0.088 to 0.117 | 0.250 | 0.250 | 0.214 to 0.304 | 0.000 | 0% |
| opentslm-untuned | 0.000 | 0.000 to 0.000 | 0.000 | 0.000 | 0.000 to 0.000 | 0.000 | 100% |

## Each system against `rules`, example by example

Positive means better than the reference. If the interval includes 0, this test cannot tell them apart.

| System | Macro-F1 difference | 95% interval | Full-match difference | 95% interval |
|---|---|---|---|---|
| llama-text-finetuned | +0.112 | +0.038 to +0.179 | +0.096 | +0.027 to +0.169 |
| rules+llama-text-finetuned | +0.105 | +0.051 to +0.150 | +0.089 | +0.045 to +0.123 |
| rules+opentslm-finetuned | +0.088 | +0.067 to +0.110 | +0.069 | +0.051 to +0.084 |
| opentslm-finetuned-values | +0.078 | +0.022 to +0.129 | -0.027 | -0.082 to +0.042 |
| opentslm-finetuned | -0.013 | -0.075 to +0.058 | -0.164 | -0.248 to -0.060 |
| note-only | -0.169 | -0.245 to -0.075 | -0.386 | -0.471 to -0.273 |
| series-only | -0.221 | -0.263 to -0.169 | -0.121 | -0.157 to -0.075 |
| majority | -0.608 | -0.683 to -0.517 | -0.369 | -0.448 to -0.260 |
| opentslm-untuned | -0.708 | -0.783 to -0.619 | -0.619 | -0.702 to -0.510 |

## Full match by note

| Note | n | llama-text-finetuned | rules+llama-text-finetuned | rules+opentslm-finetuned | opentslm-finetuned-values | rules | opentslm-finetuned | note-only | series-only | majority | opentslm-untuned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| data-problem note | 159 | 0.89 | 0.74 | 0.73 | 0.49 | 0.69 | 0.41 | 0.02 | 0.30 | 0.00 | 0.00 |
| event note, matching | 200 | 0.64 | 0.53 | 0.58 | 0.65 | 0.27 | 0.46 | 0.01 | 0.00 | 0.00 | 0.00 |
| event note, opposite direction | 36 | 0.61 | 0.56 | 0.11 | 0.44 | 0.22 | 0.11 | 0.00 | 0.72 | 0.00 | 0.00 |
| no note | 207 | 0.66 | 0.77 | 0.78 | 0.61 | 0.78 | 0.50 | 0.43 | 0.78 | 0.43 | 0.00 |
| note unrelated or without effect | 198 | 0.72 | 0.82 | 0.77 | 0.62 | 0.82 | 0.51 | 0.46 | 0.82 | 0.57 | 0.00 |

## Full match by event injected

| Event injected | n | llama-text-finetuned | rules+llama-text-finetuned | rules+opentslm-finetuned | opentslm-finetuned-values | rules | opentslm-finetuned | note-only | series-only | majority | opentslm-untuned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| definition_change | 30 | 0.67 | 0.73 | 0.67 | 0.13 | 0.50 | 0.37 | 0.00 | 0.00 | 0.00 | 0.00 |
| dip | 102 | 0.35 | 0.40 | 0.42 | 0.41 | 0.49 | 0.18 | 0.03 | 0.34 | 0.00 | 0.00 |
| duplicate_load | 36 | 0.97 | 0.64 | 0.64 | 0.28 | 0.61 | 0.44 | 0.00 | 0.00 | 0.00 | 0.00 |
| flatline | 39 | 0.82 | 0.97 | 0.97 | 0.62 | 0.97 | 0.31 | 0.00 | 0.97 | 0.00 | 0.00 |
| level_down | 96 | 0.51 | 0.61 | 0.51 | 0.48 | 0.39 | 0.31 | 0.00 | 0.29 | 0.00 | 0.00 |
| level_up | 99 | 0.73 | 0.86 | 0.72 | 0.63 | 0.43 | 0.41 | 0.00 | 0.49 | 0.00 | 0.00 |
| none (normal) | 200 | 0.89 | 0.90 | 0.90 | 0.71 | 0.90 | 0.81 | 0.90 | 0.90 | 1.00 | 0.00 |
| partial_load | 34 | 0.91 | 0.74 | 0.74 | 0.68 | 0.74 | 0.29 | 0.06 | 0.00 | 0.00 | 0.00 |
| spike | 103 | 0.60 | 0.46 | 0.53 | 0.69 | 0.38 | 0.44 | 0.00 | 0.21 | 0.00 | 0.00 |
| unit_change | 33 | 0.91 | 1.00 | 1.00 | 0.70 | 1.00 | 0.36 | 0.00 | 1.00 | 0.00 | 0.00 |
| zero_outage | 28 | 0.96 | 0.50 | 0.50 | 0.89 | 0.50 | 0.21 | 0.04 | 0.50 | 0.00 | 0.00 |

## Full match by source

| Source | n | llama-text-finetuned | rules+llama-text-finetuned | rules+opentslm-finetuned | opentslm-finetuned-values | rules | opentslm-finetuned | note-only | series-only | majority | opentslm-untuned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cik | 516 | 0.65 | 0.70 | 0.69 | 0.53 | 0.61 | 0.37 | 0.19 | 0.45 | 0.21 | 0.00 |
| online_retail | 284 | 0.83 | 0.71 | 0.69 | 0.69 | 0.63 | 0.60 | 0.30 | 0.58 | 0.32 | 0.00 |

## Confusion of `llama-text-finetuned` (rows: true verdict, columns: answered)

| True \ answered | normal | business_change | explained_event | data_issue |
|---|---|---|---|---|
| normal | 178 | 22 | 0 | 0 |
| business_change | 55 | 132 | 10 | 3 |
| explained_event | 16 | 28 | 154 | 2 |
| data_issue | 0 | 9 | 0 | 191 |
