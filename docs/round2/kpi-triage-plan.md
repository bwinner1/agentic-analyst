# Round 2 plan: KPI triage, the step before the commentary

Sun 22 Nov 2026. Branch `dmitry/round2-kpi-triage`. Decision record: [ADR 0002](../adr/0002-triage-before-commentary.md). Terms: Triage, Review window, Analyst note, Verdict and Injected event in [CONTEXT.md](../../CONTEXT.md).

## The use case in one paragraph

Before an analyst writes the monthly KPI commentary for a Partner, someone checks each KPI: did anything happen in the latest weeks, and if so, is it real? A promotion that doubled orders is worth a sentence. A load job that ran twice and doubled orders must be fixed before anyone writes a sentence. A drop the client already told us about needs a different sentence from one nobody can explain. Today this check is done by eye, KPI by KPI, using whatever notes came with the data. **Triage** gives every Review window a Verdict (`normal`, `business_change`, `explained_event`, `data_issue`), the shape of what happened and the points it covers. Code then measures the change (ADR 0001), and the Draft is written from both.

## Why this, after Round 1

| What Round 1 showed | What Round 2 does with it |
|---|---|
| Forecasting with a 1B model was level with naive baselines: no difference clear of zero. | Ask the model for the part ADR 0001 gives it, Spotting, not for numbers. |
| The text helped where it described what the series would do (sensor tasks, a holiday dip). | Make the note matter by design: some verdicts can only be reached by reading the note against the series. |
| Real-data labels are scarce and disputed; CiK's few instances gave wide intervals. | Exact labels from Injected events in real series, so there are thousands of examples and the intervals are narrow. |
| Our pipeline works: Jonan's training on Nebius, the scoring harness with series-level intervals, the CiK connector. | Reuse all three: the same OpenTSLM-SP training, the same evaluation style, and CiK histories as part of the data. |

What makes the task need both inputs:
- **The note alone is not enough.** A note about a promotion or a double posting does not say where in the window it shows, or whether it shows at all. A note can also point the other way: "opening hours were cut" next to a step *up* is not an explanation.
- **The series alone is not enough.** A duplicated load looks exactly like a short spike, a partial load like a dip, and a changed definition like a level shift. Only the note tells them apart.
- **Neither is enough when nothing happened.** More than half of normal windows (56% in the test set) come with a note that is unrelated or says nothing changed.

## The task

Input: one KPI series (history plus Review window) and one Analyst note, which may be empty. Output, in a fixed format:

```
Verdict: data_issue
Shape: spike
Window points: 4-6
Reason: The short spike lines up with the data problem described in the note, so it reflects the data, not the business.
```

A real held-out example (`python -m kpi_triage show test-00003`): weekly orders in Germany, last 8 weeks, note "Finance flagged a double posting on 3 October 2011". Points 4 to 6 of the window are about twice the usual level. The series alone reads as a business spike; with the note it is a data issue.

OpenTSLM gets the history and the window as two time series, both scaled with the history's mean and standard deviation (stated in the text, so a level shift stays visible). The text-only baseline gets the same numbers, written out to two decimals.

## Data

**Backgrounds** (real series, nothing injected yet), `src/kpi_triage/backgrounds.py`:
- *UCI Online Retail II* (CC BY 4.0): daily revenue, orders and customers per trading day for all countries and the UK, and weekly totals for Germany, France and Ireland. The derived daily CSV is committed with attribution in `data/kpi_triage/`.
- *Context is Key* histories (Apache-2.0): ATM withdrawals (daily), highway occupancy and electricity demand (hourly). Task families whose history CiK edited are left out.

**Injected events**, `src/kpi_triage/events.py`, each at least 3 robust standard deviations (of the series' seasonal differences):

| Verdict | Events | What tells it apart |
|---|---|---|
| business_change | spike, dip, level up, level down | The series; the note is empty, unrelated, or points the other way |
| explained_event | the same four | A note that names that kind of event, at that time, in that direction |
| data_issue, visible in the series | zero outage, frozen values, unit change (x100 / x1000) | The series alone |
| data_issue, visible only through the note | duplicated load (x2), partial load (30 to 60% kept), definition change (+/-25 to 50%) | The note; the series looks like business |
| normal | none | Note empty, unrelated, or saying nothing changed |

**Notes**, `src/kpi_triage/notes.py`: 6 to 8 wordings per kind. The last 2 of each kind are used only in the test set. 70% of event and data notes name a date; the rest say "last week", "a few days ago" and so on.

**Splits**, `src/kpi_triage/generate.py`: 4,000 train, 400 validation, 800 test, verdicts exactly balanced. Test holds out whole countries (Germany, France) and a whole CiK domain (electricity), so no test series was seen in training. Validation holds out Ireland. Everything is rebuilt with one command from a fixed seed.

## Evaluation

Measures (`src/kpi_triage/score.py`), all from the answer text through one parser; an unreadable answer counts as wrong:
- **Full match** (headline): verdict, shape and start all right, with the start within a tenth of the window.
- **Verdict macro-F1**: the four verdicts are balanced, so always answering one verdict scores 0.25 accuracy.
- **Attention F1**: does it flag the windows that need flagging.
- Breakdowns by note kind, injected event and source.
- 95% intervals resample whole background series, since examples cut from one series are not independent. Comparisons are paired, example by example.

Systems:

| System | What it shows |
|---|---|
| Rules (seasonal residuals, zero / frozen / unit-change checks, level-shift search; thresholds tuned on training examples) + a note classifier fitted on training notes | What a careful analyst could script; the bar to clear |
| Rules without the note / note classifier without the series | What each input is worth alone |
| Always "normal" | The floor |
| **OpenTSLM-SP (Llama-3.2-1B), fine-tuned** | Ours |
| OpenTSLM-SP before fine-tuning | What the fine-tune adds |
| **Llama-3.2-1B fine-tuned on the same examples as text** | The key baseline from ADR 0001: does the time-series encoder help? |
| A large hosted model, few-shot (optional) | A ceiling without fine-tuning; costs API money, so only with approval |

**Decision rule, fixed before any model result:** OpenTSLM is worth building the product on for triage if, on the test set, the fine-tuned OpenTSLM beats *both* the text-only Llama and the rules on full match, with the paired 95% interval of each difference above zero. If it only beats the rules, a fine-tuned text model is the simpler choice and we say so. Secondary checks: it should beat the rules on the "data-problem note" and "opposite direction" groups, where reading the note against the series is the point.

## Results (held-out test set: 800 examples from 11 series never seen in training)

Full tables: [`evaluation/kpi_triage/results.md`](../../evaluation/kpi_triage/results.md). Every model was fine-tuned once on the same 4,000 examples on one L40S, keeping its best epoch by validation loss (OpenTSLM: epoch 2 of 4, 33 minutes of training; text-only Llama: epoch 4 of 4, 64 minutes, sharing the GPU for most of it).

| System | Full match | 95% interval | Verdict macro-F1 | Unreadable |
|---|---|---|---|---|
| **Llama-3.2-1B fine-tuned, series as text** | **0.715** | 0.580 to 0.825 | **0.820** | 0% |
| Rules + note classifier (thresholds tuned on training examples) | 0.619 | 0.510 to 0.702 | 0.708 | 0% |
| Rules, series only | 0.497 | 0.400 to 0.579 | 0.487 | 0% |
| **OpenTSLM-SP fine-tuned** | **0.455** | 0.394 to 0.544 | 0.695 | 3% |
| Always normal | 0.250 | 0.214 to 0.304 | 0.100 | 0% |
| Note classifier only | 0.233 | 0.203 to 0.281 | 0.539 | 0% |
| OpenTSLM-SP before fine-tuning | 0.000 | | 0.000 | 100% |

Paired differences, example by example:

| Comparison | Full match | 95% interval |
|---|---|---|
| Text-only Llama minus rules | +0.096 | +0.027 to +0.169 |
| OpenTSLM minus rules | -0.164 | -0.248 to -0.060 |
| OpenTSLM minus text-only Llama | -0.260 | -0.336 to -0.172 |

**Against the decision rule fixed in advance, OpenTSLM-SP fails:** it is below both the text-only Llama and the rules. The Round 1 risk named in ADR 0001 came true for this task: a text model given the same numbers spots better.

**The use case itself works.** A 1B model fine-tuned for about an hour beats a tuned rule system on series and note wordings it never saw, and it gains most exactly where the note has to be read against the series:

| Full match by note | n | Rules | OpenTSLM | Text-only Llama |
|---|---|---|---|---|
| Data-problem note | 159 | 0.69 | 0.41 | **0.89** |
| Event note that explains the change | 200 | 0.27 | 0.46 | **0.64** |
| Event note pointing the other way | 36 | 0.22 | 0.11 | **0.61** |
| No note | 207 | **0.78** | 0.50 | 0.66 |
| Note unrelated or saying nothing changed | 198 | **0.82** | 0.51 | 0.72 |

Why OpenTSLM loses, from the breakdowns:
- **It reads notes well but the series poorly.** Its verdict is right on 88% of windows whose note explains the event (rules 38%), so the language side works. But it gets frozen values fully right 31% of the time (rules 97%, text 82%), unit changes 36% (rules 100%, text 91%) and zero outages 21% (rules 50%, text 96%), and its start points are off by a median of 2 points (mean 3.3), where the rules and the text model are usually exact (median 0).
- **The likely cause, not yet tested:** OpenTSLM-SP encodes the series in patches of several points, so exact values and positions blur, while the text model sees every value. Exact zeros, repeated values and a jump by a factor of 1,000 are what triage turns on.
- **As a note reader the two models are level.** When the rules find the event and a model only gives the verdict, OpenTSLM reaches 0.688 and the text Llama 0.708 (difference -0.020, interval -0.046 to +0.011). This combination was added after seeing the first results, so it is exploratory.

**Is the encoder worth anything on top of text? No.** A further run gave OpenTSLM the same numbers as text *and* its series input (`run.sh trainvalues`; planned after the first OpenTSLM result, before its own). If it beat the text-only Llama, the encoder would add something.

| Full match, test | Score | 95% interval | OpenTSLM + values minus it |
|---|---|---|---|
| Text-only Llama | 0.715 | 0.580 to 0.825 | -0.124 (-0.172 to -0.077) |
| Rules | 0.619 | 0.510 to 0.702 | -0.027 (-0.082 to +0.042) |
| **OpenTSLM + values as text** | **0.591** | 0.479 to 0.693 | |
| OpenTSLM | 0.455 | 0.394 to 0.544 | +0.136 (+0.059 to +0.200) |

- **Lost detail was half the problem.** With the exact values, OpenTSLM gains 0.136: zero outages go from 0.21 to 0.89, frozen values from 0.31 to 0.62, and the median start error halves to 1 point.
- **With the same information, it is still 0.124 below the text-only model.** The encoder adds nothing on top of text here, and something in OpenTSLM's setup costs accuracy. The gap is now on data problems only the note reveals (duplicated loads 0.28 against 0.97, definition changes 0.13 against 0.67) and on false alarms in normal windows (0.71 against 0.89), not on what the series shows.
- **Next fix, chosen from this breakdown:** the same run with the encoder frozen, since the remaining gap is not in the series signatures the marker series target (`--with_marks`, built but not run).

**How many times the test set was used.** OpenTSLM variants scored on test so far: 3 (plain, values as text, values with the encoder frozen). All are reported here. The validation set could not be used to choose between them: it has no hourly series and only training wordings, while 65% of test is hourly CiK electricity and every test note uses a held-out wording. Fixing that split is a step for the next data version (see Next steps).

## TimeNet: the Round 2 dataset PR

Each round needs a dataset PR. TimeNet has no business KPI data yet, and Online Retail II is open (CC BY 4.0) and widely known. Proposal: a TimeNet connector for **Online Retail II daily KPIs** (revenue, orders, customers per country), built like our CiK connector (PR 788): the derived daily series as the dataset, with a forecasting task, and the triage examples as a `ClassificationTask` (verdict) plus a `TemporalLocalizationTask` (where). TimeNet's rules forbid repairing data in a connector; the injected events are a separate, labelled task built on top of the untouched series, so they stay within those rules. To check first: whether TimeNet accepts a task whose examples are generated, or wants only the raw series in the connector, with the generator kept here.

## Risks

- **Injected events are cleaner than real ones.** Results show what a model can learn to spot, not how it does on real client incidents. Next check: real labelled events (NAB, MIT licence) and the CiK tasks where the note describes a real event.
- **Notes are templated.** Held-out wordings test some generalisation, not free-form notes. A few dozen hand-written notes for the test set would make it stronger.
- **Retail has few series.** Test retail examples come from 6 German and French weekly series; intervals resample series, so they show this.
- **The rules are ours.** A stronger rule set would raise the bar; they are kept simple and documented, and every threshold is in `baselines.py`.

## Next steps

1. Finish the frozen-encoder run above, and report it whatever it shows.
2. Unless that run changes the picture: build the Round 2 demo on the fine-tuned text model, which already beats the rules and is the simplest to serve, and report OpenTSLM as a measured negative result with its breakdown.
3. Rebuild the validation split so it looks like test: hold out some hourly CiK series from training and give validation its own unseen note wordings, so future variants are chosen there and test is used once.
4. Try the other OpenTSLM variant (Flamingo, cross-attention to the series) and a finer patch size, which are the direct fixes for blurred positions, on this benchmark.
5. Hand-write 50 test notes (no templates) and re-score; check real labelled events (NAB, the CiK tasks whose notes describe events).
6. TimeNet connector for Online Retail II, and the Round 2 dataset PR.
7. Product side: wire the Verdict into the Draft (a data issue stops the Draft; an explained event is cited; a business change gets measured by code).

## How to run

```
pip install -e ".[triage,test]"
python -m kpi_triage generate      # downloads CiK once (checked by sha256), writes data/kpi_triage/generated/
python -m kpi_triage baselines     # writes evaluation/kpi_triage/runs/*.jsonl
python -m kpi_triage score         # writes evaluation/kpi_triage/results.md
pytest tests/test_kpi_triage.py
```

GPU runs (OpenTSLM and the text-only Llama): see [`opentslm-triage/README.md`](../../opentslm-triage/README.md). Copy each `predictions.jsonl` into `evaluation/kpi_triage/runs/<system>.jsonl` and score again.
