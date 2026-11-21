# Results: Context is Key forecasting

Generated 2026-11-21 by `python -m tslm_eval report`. Metric: RCRPS, the benchmark's own score (scaled error, half-weighted on the region the text is about, plus a penalty for breaking stated constraints; capped at 5, which is also the score of a failed output). Weighted mean over instances with CiK's task weights; interval from resampling whole tasks. All systems give point forecasts, so scores compare with each other, not with the published CiK leaderboard (which samples 25+ forecasts).

## Held-out test set: scenario groups atm, unemployment, sensor (65 instances)

| System | RCRPS (lower is better) | 95% interval | Valid outputs | Better than naive | Instances (tasks) |
|---|---|---|---|---|---|
| seasonal-naive | 0.286 | 0.236 to 0.331 | 100% | 52% | 65 (13) |
| naive | 0.330 | 0.280 to 0.370 | 100% |  | 65 (13) |

## All 355 instances

Zero-shot systems (baselines, Claude, pretrained OpenTSLM) never trained on CiK, so all 355 are unseen for them. A model fine-tuned on CiK must be read on the held-out set only.

| System | RCRPS (lower is better) | 95% interval | Valid outputs | Better than naive | Instances (tasks) |
|---|---|---|---|---|---|
| naive | 0.332 | 0.264 to 0.410 | 100% |  | 355 (71) |
| seasonal-naive | 0.352 | 0.254 to 0.485 | 100% | 33% | 355 (71) |

## By scenario group (all instances)

| Scenario group | naive | seasonal-naive |
|---|---|---|
| atm | 0.322 | 0.271 |
| causal | 0.387 | 0.387 |
| constraints | 0.778 | 1.400 |
| electricity | 0.092 | 0.087 |
| montreal-fire | 0.225 | 0.225 |
| sensor | 0.373 | 0.308 |
| solar | 0.321 | 0.322 |
| traffic | 0.276 | 0.291 |
| unemployment | 0.308 | 0.308 |
| wind-tunnel | 0.684 | 0.684 |

![Example forecast](chart_example.png)
