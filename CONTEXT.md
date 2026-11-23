# Agentic Analyst

A small time-series language model that lets consulting Partners ask questions about client KPI series and get verified answers, running on the firm's own hardware so client data never leaves it.

## Language

**Partner**:
A consulting partner: knows the client's industry but is not a data specialist. The user who asks Questions about client KPIs instead of waiting weeks for an analyst.
_Avoid_: user, client, non-expert (when meaning this person)

**Buyer**:
A mid-size or boutique consulting firm (roughly 20–500 consultants) that has the client-confidentiality problem but not the budget to build its own AI platform. Large firms with in-house platforms are not the target.
_Avoid_: client, customer, enterprise

**Question**:
A natural-language question a Partner asks about one or more KPI series, such as "what happened last month?" or "is this normal?".
_Avoid_: query, prompt

**Answer**:
The system's reply to a Question, stating only what the data supports up to its Explanation level.
_Avoid_: response, output, insight

**KPI commentary**:
The recurring written report that says what moved in a client's KPIs over a period. The standing Answer to "what happened this period?".
_Avoid_: insight, analysis, memo (when meaning this report)

**Draft**:
A KPI commentary written by the model, before an analyst or Partner checks and finishes it.
_Avoid_: output, generation, final report

## Verification

**Claim**:
A single statement in an Answer that the data can confirm or refute, such as a number, a direction, a date or a Driver.
_Avoid_: fact, finding, statement

**Check**:
Recomputing a Claim from the data by code before the Partner sees it. Claims that fail are removed or flagged; an Answer is never shown unchecked.
_Avoid_: validation, review, judge

**Calculation path**:
The steps from the raw series to a Claim's number (which values, which operation, which result), shown so the Partner can retrace it.
_Avoid_: reasoning, chain of thought, explanation

**Evidence**:
The data shown alongside an Answer: the figures used and a chart of the series with the relevant period highlighted.
_Avoid_: proof, source, citation

**Computed answer**:
The correct answer to a Question, derived from the data by code. The primary ground truth for scoring Answers.
_Avoid_: label, gold answer, reference

**Behaviour test**:
A test with no answer key that checks the model uses the series, for example shuffling or rescaling the series or asking a question the data cannot answer.
_Avoid_: sanity check, ablation (when meaning these)

**Forecast**:
The projected next values of a KPI series, produced by a forecasting model separate from the language model. Trusted through its stated error on held-out periods against a naive baseline, not through the Check.
_Avoid_: prediction, outlook, projection

## Explanations

**Explanation level**:
How a Draft accounts for why a KPI moved, ranked by how far the data can verify it. Levels 1–3 are in scope; level 4 is deferred until there are extra data sources; level 5 is left to people.
_Avoid_: insight, reason, root cause

**Driver** (level 1):
A component of a KPI whose change accounts for the KPI's change, because the components add up to the total (for example a region's share of revenue). Wording: "driven by".
_Avoid_: cause, factor

**Seasonal comparison** (level 2):
Judging a change against the same period in earlier years, to say whether it is normal for the time of year. Wording: "in line with", "above / below the usual".
_Avoid_: trend, seasonality adjustment

**Co-movement** (level 3):
Another series that moved at the same time as the KPI, with no claim that it caused the movement. Wording: "coincided with", never "caused by".
_Avoid_: correlation driver, cause

**Hypothesis** (level 4, deferred):
A possible outside cause taken from a cited source such as news or client documents, marked as unverified in the Draft.
_Avoid_: explanation, finding

**Expert judgement** (level 5):
Interpretation that needs industry knowledge. Never written by the model; left to the analyst or partner.

## Roles

**Spotting**:
The model's job: deciding which periods, breaks or components in a series are worth an Answer. The model spots; code computes the figures.
_Avoid_: detection, analysis, reasoning

## Triage (Round 2 use case)

**Triage**:
Deciding, before a KPI commentary is written, whether the latest stretch of a series needs attention and what kind. Every Draft starts from it: a data problem must be fixed before anything is said, and an explained event needs a different sentence from a real change.
_Avoid_: anomaly detection (when meaning the whole task), alerting

**Review window**:
The latest points of a series that the commentary will talk about (for example the last 20 trading days). Everything before it is the history the window is judged against.
_Avoid_: test period, horizon

**Analyst note**:
A short free-text remark that comes with the series, as it would from a client or an analyst: a promotion, an outage, a reload of the data, or nothing relevant. It may or may not explain what the series shows.
_Avoid_: context, metadata, prompt

**Verdict**:
The triage outcome for a review window: `normal` (nothing to flag), `business_change` (a real change no note explains), `explained_event` (a change the analyst note accounts for) or `data_issue` (the data is wrong, from the series or the note). Each verdict other than normal comes with a shape and the points it covers.
_Avoid_: label, class, anomaly type

**Injected event**:
A change written into a real series on purpose (a spike, a level shift, a duplicated load), so the right verdict, shape and points are known exactly. The series it goes into is a background.
_Avoid_: synthetic anomaly, fake data
