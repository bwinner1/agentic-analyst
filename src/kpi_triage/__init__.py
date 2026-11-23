"""KPI anomaly triage: before KPI commentary is written, decide for the review window whether
anything needs attention, and if so where, what shape, and whether it is a real business change,
an event the analyst's note explains, or a data issue.

The model spots (verdict, shape, where); code measures sizes, as in ADR 0001. Labelled examples
come from real series with events injected by code, so every label is exact. See
docs/round2/kpi-triage-plan.md.
"""
