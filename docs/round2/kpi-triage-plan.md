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
| Rules (seasonal residuals, zero / frozen / unit-change checks, level-shift search) + a note classifier fitted on training notes | What a careful analyst could script; the bar to clear |
| Rules without the note / note classifier without the series | What each input is worth alone |
| Always "normal" | The floor |
| **OpenTSLM-SP (Llama-3.2-1B), fine-tuned** | Ours |
| OpenTSLM-SP before fine-tuning | What the fine-tune adds |
| **Llama-3.2-1B fine-tuned on the same examples as text** | The key baseline from ADR 0001: does the time-series encoder help? |
| A large hosted model, few-shot (optional) | A ceiling without fine-tuning; costs API money, so only with approval |

**Decision rule, fixed before any model result:** OpenTSLM is worth building the product on for triage if, on the test set, the fine-tuned OpenTSLM beats *both* the text-only Llama and the rules on full match, with the paired 95% interval of each difference above zero. If it only beats the rules, a fine-tuned text model is the simpler choice and we say so. Secondary checks: it should beat the rules on the "data-problem note" and "opposite direction" groups, where reading the note against the series is the point.

## Results so far (held-out test set, 800 examples, 11 series)

Full table: [`evaluation/kpi_triage/results.md`](../../evaluation/kpi_triage/results.md).

| System | Full match | 95% interval | Verdict macro-F1 |
|---|---|---|---|
| Rules + note classifier | 0.545 | 0.456 to 0.626 | 0.681 |
| Rules, series only | 0.427 | 0.335 to 0.531 | 0.458 |
| Always normal | 0.250 | 0.214 to 0.304 | 0.100 |
| Note classifier only | 0.233 | 0.203 to 0.281 | 0.539 |

What this already says:
- **The task needs both inputs.** Rules with the note beat rules without it by 0.12 full match (paired interval 0.07 to 0.15), and the note alone cannot place anything.
- **The rules leave plenty of room.** They get 0.25 full match on windows whose note matches the event, and 0.25 on notes that point the other way. The note reader is the weak part: on the held-out wordings it reads 78 of 126 "up" event notes as "down", because it matches words, not meaning. The series rules also flag 40% of normal windows, because real KPIs are noisy. Reading the note's meaning against the series is exactly what a fine-tuned model should do better.
- The model rows (fine-tuned OpenTSLM, untuned OpenTSLM, text-only Llama) are running on the Nebius VM and will be added to this table.

## TimeNet: the Round 2 dataset PR

Each round needs a dataset PR. TimeNet has no business KPI data yet, and Online Retail II is open (CC BY 4.0) and widely known. Proposal: a TimeNet connector for **Online Retail II daily KPIs** (revenue, orders, customers per country), built like our CiK connector (PR 788): the derived daily series as the dataset, with a forecasting task, and the triage examples as a `ClassificationTask` (verdict) plus a `TemporalLocalizationTask` (where). TimeNet's rules forbid repairing data in a connector; the injected events are a separate, labelled task built on top of the untouched series, so they stay within those rules. To check first: whether TimeNet accepts a task whose examples are generated, or wants only the raw series in the connector, with the generator kept here.

## Risks

- **Injected events are cleaner than real ones.** Results show what a model can learn to spot, not how it does on real client incidents. Next check: real labelled events (NAB, MIT licence) and the CiK tasks where the note describes a real event.
- **Notes are templated.** Held-out wordings test some generalisation, not free-form notes. A few dozen hand-written notes for the test set would make it stronger.
- **Retail has few series.** Test retail examples come from 6 German and French weekly series; intervals resample series, so they show this.
- **The rules are ours.** A stronger rule set would raise the bar; they are kept simple and documented, and every threshold is in `baselines.py`.

## Next steps

1. Add the three model rows when the VM runs finish, and check the decision rule.
2. Hand-write 50 test notes (no templates) and re-score.
3. Real-event check on NAB and the CiK tasks whose notes describe events.
4. TimeNet connector for Online Retail II, and the Round 2 dataset PR.
5. Product side: wire the Verdict into the Draft (a data issue stops the Draft; an explained event is cited; a business change gets measured by code).

## How to run

```
pip install -e ".[triage,test]"
python -m kpi_triage generate      # downloads CiK once (checked by sha256), writes data/kpi_triage/generated/
python -m kpi_triage baselines     # writes evaluation/kpi_triage/runs/*.jsonl
python -m kpi_triage score         # writes evaluation/kpi_triage/results.md
pytest tests/test_kpi_triage.py
```

GPU runs (OpenTSLM and the text-only Llama): see [`opentslm-triage/README.md`](../../opentslm-triage/README.md). Copy each `predictions.jsonl` into `evaluation/kpi_triage/runs/<system>.jsonl` and score again.
