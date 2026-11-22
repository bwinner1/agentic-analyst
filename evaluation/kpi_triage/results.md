# KPI triage results (800 test examples, 11 series)

Verdicts are balanced, so always answering one verdict scores 0.25 accuracy. Higher is better everywhere.
Intervals: 95%, resampling whole series. Full match = verdict, shape and start (within a tenth of the window) all right.

| System | Verdict macro-F1 | 95% interval | Verdict accuracy | Full match | 95% interval | Attention F1 | Unreadable |
|---|---|---|---|---|---|---|---|
| rules | 0.681 | 0.616 to 0.756 | 0.682 | 0.545 | 0.456 to 0.626 | 0.913 | 0% |
| note-only | 0.539 | 0.524 to 0.554 | 0.619 | 0.233 | 0.203 to 0.281 | 0.778 | 0% |
| series-only | 0.458 | 0.394 to 0.520 | 0.504 | 0.427 | 0.335 to 0.531 | 0.913 | 0% |
| majority | 0.100 | 0.088 to 0.117 | 0.250 | 0.250 | 0.214 to 0.304 | 0.000 | 0% |

## Each system against `rules`, example by example

Positive means better than the reference. If the interval includes 0, this test cannot tell them apart.

| System | Macro-F1 difference | 95% interval | Full-match difference | 95% interval |
|---|---|---|---|---|
| note-only | -0.142 | -0.215 to -0.076 | -0.312 | -0.382 to -0.227 |
| series-only | -0.222 | -0.260 to -0.175 | -0.118 | -0.152 to -0.074 |
| majority | -0.581 | -0.652 to -0.520 | -0.295 | -0.359 to -0.215 |

## Full match by note

| Note | n | rules | note-only | series-only | majority |
|---|---|---|---|---|---|
| data-problem note | 159 | 0.69 | 0.02 | 0.30 | 0.00 |
| event note, matching | 200 | 0.25 | 0.01 | 0.00 | 0.00 |
| event note, opposite direction | 36 | 0.25 | 0.00 | 0.75 | 0.00 |
| no note | 207 | 0.65 | 0.43 | 0.65 | 0.43 |
| note unrelated or without effect | 198 | 0.67 | 0.46 | 0.67 | 0.57 |

## Full match by event injected

| Event injected | n | rules | note-only | series-only | majority |
|---|---|---|---|---|---|
| definition_change | 30 | 0.50 | 0.00 | 0.00 | 0.00 |
| dip | 102 | 0.49 | 0.03 | 0.35 | 0.00 |
| duplicate_load | 36 | 0.67 | 0.00 | 0.00 | 0.00 |
| flatline | 39 | 0.97 | 0.00 | 0.97 | 0.00 |
| level_down | 96 | 0.40 | 0.00 | 0.30 | 0.00 |
| level_up | 99 | 0.43 | 0.00 | 0.49 | 0.00 |
| none (normal) | 200 | 0.59 | 0.90 | 0.59 | 1.00 |
| partial_load | 34 | 0.68 | 0.06 | 0.00 | 0.00 |
| spike | 103 | 0.38 | 0.00 | 0.23 | 0.00 |
| unit_change | 33 | 1.00 | 0.00 | 1.00 | 0.00 |
| zero_outage | 28 | 0.50 | 0.04 | 0.50 | 0.00 |

## Full match by source

| Source | n | rules | note-only | series-only | majority |
|---|---|---|---|---|---|
| cik | 516 | 0.51 | 0.19 | 0.36 | 0.21 |
| online_retail | 284 | 0.60 | 0.30 | 0.55 | 0.32 |

## Confusion of `rules` (rows: true verdict, columns: answered)

| True \ answered | normal | business_change | explained_event | data_issue |
|---|---|---|---|---|
| normal | 119 | 79 | 2 | 0 |
| business_change | 13 | 166 | 19 | 2 |
| explained_event | 12 | 96 | 79 | 13 |
| data_issue | 3 | 10 | 5 | 182 |
