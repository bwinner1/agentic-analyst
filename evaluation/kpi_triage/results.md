# KPI triage results (800 test examples, 11 series)

Verdicts are balanced, so always answering one verdict scores 0.25 accuracy. Higher is better everywhere.
Intervals: 95%, resampling whole series. Full match = verdict, shape and start (within a tenth of the window) all right.

| System | Verdict macro-F1 | 95% interval | Verdict accuracy | Full match | 95% interval | Attention F1 | Unreadable |
|---|---|---|---|---|---|---|---|
| rules | 0.708 | 0.619 to 0.783 | 0.720 | 0.619 | 0.510 to 0.702 | 0.917 | 0% |
| note-only | 0.539 | 0.524 to 0.554 | 0.619 | 0.233 | 0.203 to 0.281 | 0.778 | 0% |
| series-only | 0.487 | 0.419 to 0.536 | 0.550 | 0.497 | 0.400 to 0.579 | 0.917 | 0% |
| majority | 0.100 | 0.088 to 0.117 | 0.250 | 0.250 | 0.214 to 0.304 | 0.000 | 0% |

## Each system against `rules`, example by example

Positive means better than the reference. If the interval includes 0, this test cannot tell them apart.

| System | Macro-F1 difference | 95% interval | Full-match difference | 95% interval |
|---|---|---|---|---|
| note-only | -0.169 | -0.245 to -0.075 | -0.386 | -0.471 to -0.273 |
| series-only | -0.221 | -0.263 to -0.169 | -0.121 | -0.157 to -0.075 |
| majority | -0.608 | -0.683 to -0.517 | -0.369 | -0.448 to -0.260 |

## Full match by note

| Note | n | rules | note-only | series-only | majority |
|---|---|---|---|---|---|
| data-problem note | 159 | 0.69 | 0.02 | 0.30 | 0.00 |
| event note, matching | 200 | 0.27 | 0.01 | 0.00 | 0.00 |
| event note, opposite direction | 36 | 0.22 | 0.00 | 0.72 | 0.00 |
| no note | 207 | 0.78 | 0.43 | 0.78 | 0.43 |
| note unrelated or without effect | 198 | 0.82 | 0.46 | 0.82 | 0.57 |

## Full match by event injected

| Event injected | n | rules | note-only | series-only | majority |
|---|---|---|---|---|---|
| definition_change | 30 | 0.50 | 0.00 | 0.00 | 0.00 |
| dip | 102 | 0.49 | 0.03 | 0.34 | 0.00 |
| duplicate_load | 36 | 0.61 | 0.00 | 0.00 | 0.00 |
| flatline | 39 | 0.97 | 0.00 | 0.97 | 0.00 |
| level_down | 96 | 0.39 | 0.00 | 0.29 | 0.00 |
| level_up | 99 | 0.43 | 0.00 | 0.49 | 0.00 |
| none (normal) | 200 | 0.90 | 0.90 | 0.90 | 1.00 |
| partial_load | 34 | 0.74 | 0.06 | 0.00 | 0.00 |
| spike | 103 | 0.38 | 0.00 | 0.21 | 0.00 |
| unit_change | 33 | 1.00 | 0.00 | 1.00 | 0.00 |
| zero_outage | 28 | 0.50 | 0.04 | 0.50 | 0.00 |

## Full match by source

| Source | n | rules | note-only | series-only | majority |
|---|---|---|---|---|---|
| cik | 516 | 0.61 | 0.19 | 0.45 | 0.21 |
| online_retail | 284 | 0.63 | 0.30 | 0.58 | 0.32 |

## Confusion of `rules` (rows: true verdict, columns: answered)

| True \ answered | normal | business_change | explained_event | data_issue |
|---|---|---|---|---|
| normal | 179 | 20 | 1 | 0 |
| business_change | 36 | 145 | 17 | 2 |
| explained_event | 30 | 82 | 75 | 13 |
| data_issue | 8 | 10 | 5 | 177 |
