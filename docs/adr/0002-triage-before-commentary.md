# Triage before commentary, with exact labels from injected events

Round 1 asked OpenTSLM to forecast (Context is Key). On held-out task types the fine-tuned 1B model was level with naive and seasonal-naive forecasts on the official metric (no difference clear of zero), and the text helped where a note described what the series would do (the sensor tasks, a holiday dip). Forecasting also tests the arithmetic that ADR 0001 gives to code. So Round 2 tests the part the model is meant to do: Spotting, together with reading a note. Before any Draft is written, the model gives a Verdict for the Review window (normal, business change, explained event, data issue), its shape and the points it covers. Code then measures the change, as in ADR 0001.

The examples are real KPI series (UCI Online Retail II, and the clean histories of the Context is Key tasks) with Injected events, so every Verdict, shape and span is known exactly, and no person has to label anything. Some data issues are visible only through the Analyst note (a duplicated load doubles a day the way a promotion would), and some notes point the opposite way to what the series shows. In both cases the series alone, or the note alone, is not enough.

## Considered Options

- **Label real anomalies by hand (NAB, Yahoo, client data).** Rejected for now: few labelled business KPIs exist under open licences, the labels disagree on where an event starts, and they carry no notes. Kept as a later check that results hold on events nobody injected.
- **Keep forecasting, with notes.** Rejected: it measures numbers the product never asks the model for.
- **Let a text-only model do triage.** This is the key baseline, not a design: the same base Llama-3.2-1B, fine-tuned on the same examples with the series written as text. If it does as well, OpenTSLM has no advantage here either.

## Consequences

The test set holds out whole countries and a whole CiK domain, and note wordings never seen in training, so a model cannot pass by remembering series or phrases. Results on injected events show what a model can learn to spot; they do not yet show how often each event happens on real client data, so they say nothing about real-world precision until checked on real labelled events.
