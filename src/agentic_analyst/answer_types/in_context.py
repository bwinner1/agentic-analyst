"""'Is that normal?': is a month's change seasonal, the whole market, or specific to this business?

Three checks, all computed by code:
1. The change on the previous month.
2. Seasonal: the same change (for example August to September) in each of the previous 10 years, leaving
   out 2020 and 2021 (pandemic closures). Within the lowest-to-highest range of those years counts as
   normal for the season.
3. Market: growth on the same month last year, against the benchmark series (the market) over the same
   months. Within 1.0 percentage point counts as in line with the market.

Verdicts are decided on the percentages as shown (one decimal), so a reader can verify them from the text.
Needs at least 5 reference years. The market check is skipped when the series has no benchmark.
"""
from __future__ import annotations

from statistics import median

from ..answer import Chart, Claim, Draft, Evidence, Figure
from ..measures import MEASURES, register_measure
from ..series import MONTH_NAMES, Series

PATTERN = (r"\bnormal\b|\bseason|\busual\b|\btypical\b|\bexpected\b|\bmarket\b"
           r"|\bjust (us|them)\b|\bis it us\b|\bin context\b")
PRIORITY = 90  # before 'what happened', which this answer includes

REFERENCE_YEARS = 10
LEFT_OUT_YEARS = (2020, 2021)
MIN_REFERENCE_YEARS = 5
IN_LINE_POINTS = 1.0  # percentage points


# ---- computations (each takes the series and the period index) -------------------------------------

def _shown(x: float) -> float:
    """A share as the percentage shown in the text, rounded to one decimal."""
    return round(x * 100, 1)


def _pct(a: float, b: float) -> float | None:
    return a / b - 1 if b > 0 else None


def _previous_month(year: int, month: int) -> tuple[int, int]:
    return (year, month - 1) if month > 1 else (year - 1, 12)


def month_change(s: Series, i: int) -> float | None:
    return _pct(s.values[i], s.values[i - 1]) if i >= 1 else None


def reference_changes(s: Series, i: int) -> list[tuple[int, float]]:
    """(year, change) for the same month-on-month change in each usable reference year."""
    year, month = s.dates[i].year, s.dates[i].month
    out = []
    for y in range(year - REFERENCE_YEARS, year):
        if y in LEFT_OUT_YEARS:
            continue
        cur = s.index_of(y, month)
        prev = s.index_of(*_previous_month(y, month))
        if cur is not None and prev is not None and (c := _pct(s.values[cur], s.values[prev])) is not None:
            out.append((y, c))
    return out


def seasonal_typical(s: Series, i: int) -> float | None:
    ref = reference_changes(s, i)
    return median(c for _, c in ref) if len(ref) >= MIN_REFERENCE_YEARS else None


def seasonal_low(s: Series, i: int) -> float | None:
    ref = reference_changes(s, i)
    return min(c for _, c in ref) if len(ref) >= MIN_REFERENCE_YEARS else None


def seasonal_high(s: Series, i: int) -> float | None:
    ref = reference_changes(s, i)
    return max(c for _, c in ref) if len(ref) >= MIN_REFERENCE_YEARS else None


def seasonal_verdict(s: Series, i: int) -> str | None:
    change, low, high = month_change(s, i), seasonal_low(s, i), seasonal_high(s, i)
    if change is None or low is None:
        return None
    if _shown(change) < _shown(low):
        return "weaker than in any reference year"
    if _shown(change) > _shown(high):
        return "stronger than in any reference year"
    return "normal for the season"


def year_on_year(s: Series, i: int) -> float | None:
    j = s.index_of(s.dates[i].year - 1, s.dates[i].month)
    return _pct(s.values[i], s.values[j]) if j is not None else None


def market_year_on_year(s: Series, i: int) -> float | None:
    b = s.benchmark
    if b is None:
        return None
    cur = b.index_of(s.dates[i].year, s.dates[i].month)
    prev = b.index_of(s.dates[i].year - 1, s.dates[i].month)
    return _pct(b.values[cur], b.values[prev]) if cur is not None and prev is not None else None


def market_verdict(s: Series, i: int) -> str | None:
    own, market = year_on_year(s, i), market_year_on_year(s, i)
    if own is None or market is None:
        return None
    gap = _shown(own) - _shown(market)
    if abs(gap) <= IN_LINE_POINTS:
        return "in line with the market"
    return "ahead of the market" if gap > 0 else "behind the market"


def _as_percent(s: Series, x: float) -> str:
    return f"{x * 100:+.1f}%"


register_measure("month_change", month_change, _as_percent)
register_measure("seasonal_typical", seasonal_typical, _as_percent)
register_measure("seasonal_low", seasonal_low, _as_percent)
register_measure("seasonal_high", seasonal_high, _as_percent)
register_measure("seasonal_verdict", seasonal_verdict, lambda s, x: x)
register_measure("year_on_year", year_on_year, _as_percent)
register_measure("market_year_on_year", market_year_on_year, _as_percent)
register_measure("market_verdict", market_verdict, lambda s, x: x)


# ---- the answer ---------------------------------------------------------------------------------------

def answer(series: Series, i: int, question: str) -> Draft | None:
    if i < 1 or month_change(series, i) is None or seasonal_typical(series, i) is None:
        return None  # nothing to compare with

    def shown(measure: str) -> str:
        m = MEASURES[measure]
        return m.render(series, m.compute(series, i))

    d = series.dates[i]
    month, prev_month = MONTH_NAMES[d.month - 1], MONTH_NAMES[_previous_month(d.year, d.month)[1] - 1]
    cur, prev = series.label(i), series.label(i - 1)
    ref = reference_changes(series, i)
    years = ", ".join(str(y) for y, _ in ref)

    claims = [Claim(m, i, shown(m)) for m in
              ("level", "month_change", "seasonal_typical", "seasonal_low", "seasonal_high", "seasonal_verdict")]
    text = [
        f"{series.name}: {cur} was {shown('level')} {series.unit}, {shown('month_change')} from {prev}.",
        f"Seasonal check: over {len(ref)} reference years ({years}), {month} moved a median of "
        f"{shown('seasonal_typical')} from {prev_month} (lowest {shown('seasonal_low')}, "
        f"highest {shown('seasonal_high')}).",
    ]
    path = [
        f"{cur} = {series.values[i]:,.0f}; {prev} = {series.values[i - 1]:,.0f}; "
        f"change = {series.values[i]:,.0f} / {series.values[i - 1]:,.0f} - 1 = {shown('month_change')}",
        f"Reference years: the previous {REFERENCE_YEARS}, leaving out "
        f"{' and '.join(map(str, LEFT_OUT_YEARS))} (pandemic closures)",
    ]
    for y, c in ref:
        a, b = series.index_of(y, d.month), series.index_of(*_previous_month(y, d.month))
        path.append(f"{series.label(a)} vs {series.label(b)}: {series.values[a]:,.0f} / {series.values[b]:,.0f} - 1 "
                    f"= {_as_percent(series, c)}")
    path.append(f"median of {len(ref)} = {shown('seasonal_typical')}; lowest {shown('seasonal_low')}; "
                f"highest {shown('seasonal_high')}")
    conclusion = f"In short: the change from {prev_month} is {shown('seasonal_verdict')}"

    j = series.index_of(d.year - 1, d.month)
    highlight = [i - 1, i]
    figures = [Figure(cur, series.values[i]), Figure(prev, series.values[i - 1])]
    if market_verdict(series, i) is not None:
        b = series.benchmark
        bi, bj = b.index_of(d.year, d.month), b.index_of(d.year - 1, d.month)
        claims += [Claim(m, i, shown(m)) for m in ("year_on_year", "market_year_on_year", "market_verdict")]
        text.append(f"Market check: against {series.label(j)}, sales changed by {shown('year_on_year')}, "
                    f"while {b.name} changed by {shown('market_year_on_year')}.")
        path += [
            f"{cur} vs {series.label(j)}: {series.values[i]:,.0f} / {series.values[j]:,.0f} - 1 = {shown('year_on_year')}",
            f"{b.name}, {b.label(bi)} vs {b.label(bj)}: {b.values[bi]:,.0f} / {b.values[bj]:,.0f} - 1 "
            f"= {shown('market_year_on_year')}",
            f"gap = {_shown(year_on_year(series, i)) - _shown(market_year_on_year(series, i)):+.1f} percentage points; "
            f"within {IN_LINE_POINTS:.1f} counts as in line",
        ]
        conclusion += f", and over the year sales are {shown('market_verdict')}"
        highlight.append(j)
        figures.append(Figure(series.label(j), series.values[j]))
    else:
        text.append("Market check: not available (no market series, or no data for the same month last year).")
    text.append(conclusion + ".")
    path.append("All values are not seasonally adjusted and in current dollars (not adjusted for inflation).")

    chart = Chart(tuple(x.isoformat() for x in series.dates), series.values, tuple(sorted(highlight)))
    return Draft(" ".join(text), tuple(claims), tuple(path), Evidence(tuple(figures), chart))
