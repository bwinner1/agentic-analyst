# KPI triage results (400 test examples, 3 series)

Verdicts are balanced, so always answering one verdict scores 0.25 accuracy. Higher is better everywhere.
Intervals: 95%, resampling whole series. Full match = verdict, shape and start (within a tenth of the window) all right.

| System | Verdict macro-F1 | 95% interval | Verdict accuracy | Full match | 95% interval | Attention F1 | Unreadable |
|---|---|---|---|---|---|---|---|
| opentslm-finetuned | 0.865 | 0.835 to 0.868 | 0.863 | 0.725 | 0.667 to 0.795 | 0.953 | 0% |
| rules | 0.768 | 0.711 to 0.881 | 0.760 | 0.703 | 0.612 to 0.801 | 0.814 | 0% |
| note-only | 0.601 | 0.540 to 0.636 | 0.682 | 0.285 | 0.185 to 0.439 | 0.783 | 0% |
| series-only | 0.436 | 0.373 to 0.544 | 0.522 | 0.500 | 0.435 to 0.561 | 0.814 | 0% |
| majority | 0.100 | 0.078 to 0.139 | 0.250 | 0.250 | 0.185 to 0.383 | 0.000 | 0% |

## Each system against `rules`, example by example

Positive means better than the reference. If the interval includes 0, this test cannot tell them apart.

| System | Macro-F1 difference | 95% interval | Full-match difference | 95% interval |
|---|---|---|---|---|
| opentslm-finetuned | +0.097 | -0.046 to +0.129 | +0.022 | -0.007 to +0.054 |
| note-only | -0.167 | -0.342 to -0.101 | -0.418 | -0.616 to -0.252 |
| series-only | -0.332 | -0.338 to -0.331 | -0.203 | -0.281 to -0.131 |
| majority | -0.668 | -0.803 to -0.621 | -0.453 | -0.616 to -0.308 |

## Full match by note

| Note | n | opentslm-finetuned | rules | note-only | series-only | majority |
|---|---|---|---|---|---|---|
| data-problem note | 73 | 0.71 | 0.36 | 0.16 | 0.23 | 0.00 |
| event note, matching | 100 | 0.84 | 0.72 | 0.02 | 0.00 | 0.00 |
| event note, opposite direction | 20 | 0.60 | 0.75 | 0.00 | 0.75 | 0.00 |
| no note | 92 | 0.61 | 0.82 | 0.38 | 0.82 | 0.38 |
| note unrelated or without effect | 115 | 0.75 | 0.81 | 0.57 | 0.81 | 0.57 |

## Full match by event injected

| Event injected | n | opentslm-finetuned | rules | note-only | series-only | majority |
|---|---|---|---|---|---|---|
| dip | 33 | 0.48 | 0.15 | 0.06 | 0.09 | 0.00 |
| duplicate_load | 17 | 0.82 | 0.53 | 0.00 | 0.00 | 0.00 |
| flatline | 19 | 0.42 | 1.00 | 0.00 | 1.00 | 0.00 |
| level_down | 53 | 0.94 | 0.91 | 0.00 | 0.47 | 0.00 |
| level_up | 63 | 0.84 | 0.98 | 0.00 | 0.49 | 0.00 |
| none (normal) | 100 | 0.85 | 0.94 | 1.00 | 0.94 | 1.00 |
| partial_load | 28 | 0.57 | 0.00 | 0.18 | 0.00 | 0.00 |
| spike | 51 | 0.67 | 0.57 | 0.00 | 0.25 | 0.00 |
| unit_change | 15 | 0.33 | 1.00 | 0.00 | 1.00 | 0.00 |
| zero_outage | 21 | 0.43 | 0.00 | 0.33 | 0.00 | 0.00 |

## Full match by source

| Source | n | opentslm-finetuned | rules | note-only | series-only | majority |
|---|---|---|---|---|---|---|
| online_retail | 400 | 0.72 | 0.70 | 0.28 | 0.50 | 0.25 |

## Confusion of `opentslm-finetuned` (rows: true verdict, columns: answered)

| True \ answered | normal | business_change | explained_event | data_issue | unreadable |
|---|---|---|---|---|---|
| normal | 85 | 15 | 0 | 0 | 0 |
| business_change | 6 | 87 | 6 | 0 | 1 |
| explained_event | 0 | 1 | 99 | 0 | 0 |
| data_issue | 6 | 20 | 0 | 74 | 0 |
