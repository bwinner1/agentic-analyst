# Market context demo: season, market, or just them?

A partner asks: "Sales moved this month. Is that just the season, the whole market, or something specific
to the client?" The engine answers from US Census retail data, and every number is checked by code.

## What it does

The `in_context` answer type (`src/agentic_analyst/answer_types/in_context.py`) runs three checks:

1. **Change from last month.**
2. **Seasonal check:** the same month-to-month change in each of the previous 10 years, leaving out 2020
   and 2021 (pandemic closures). Within the lowest-to-highest range counts as normal for the season.
3. **Market check:** growth on the same month last year, against total US retail. Within 1.0 percentage
   point counts as in line with the market.

Every figure in the answer is a Claim that the Check recomputes, as in the "what happened" answer type.

## Data

US Census Bureau, Monthly Retail Trade, via FRED. Public domain. Not seasonally adjusted, current dollars.
Snapshot of 22 Nov 2026 in `data/census_retail/` (2014-01 to 2026-09): clothing, shoe, health and personal
care, and grocery stores, plus total retail as the market. Refresh with:

    PYTHONPATH=src python -m agentic_analyst.sources.census_retail

## Demo page

`index.html` shows `answers.json`, which `build_data.py` writes by running the engine on every store type
and month. The page computes nothing itself.

    PYTHONPATH=src python prototypes/market-context/build_data.py
    cd prototypes/market-context && python -m http.server

## Example (shoe stores, September 2026)

> Shoe store sales (US): September 2026 was 2,500 USD millions, -33.9% from August 2026. Seasonal check:
> over 8 reference years (2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025), September moved a median of
> -30.1% from August (lowest -37.6%, highest -25.1%). Market check: against September 2025, sales changed
> by +3.9%, while total US retail (excluding food services) changed by +7.0%. In short: the change from
> August is normal for the season, and over the year sales are behind the market.

## Limits

- Census data is the market, not one client. A client's own sales would be the series; Census stays the
  benchmark.
- Nominal dollars: growth includes inflation, for the business and the market alike.
- No language model yet. The text is written by code; a model can rephrase it later, and the Check still
  runs on whatever it writes (ADR 0001).
