"""'Is that normal?': seasonal check and market check, on series built so every answer is known."""
from datetime import date

import pytest

from agentic_analyst import Series, ask, load_census_retail
from agentic_analyst.sources.census_retail import load_fred_csv

# September-vs-August change in each year. 2020 and 2021 are extreme on purpose: they must be left out.
REFERENCE = {2015: -0.10, 2016: -0.12, 2017: -0.08, 2018: -0.11, 2019: -0.09,
             2020: -0.90, 2021: +0.50, 2022: -0.10, 2023: -0.13, 2024: -0.07}


def store(september_2025: float, start_year: int = 2014) -> Series:
    """Every month is 100, except September, which is 100 * (1 + that year's change on August)."""
    changes = REFERENCE | {2025: september_2025}
    points = {}
    for y in range(start_year, 2026):
        for m in range(1, 13):
            points[date(y, m, 1)] = 100 * (1 + changes.get(y, 0.0)) if m == 9 else 100.0
    return Series.from_points("Store sales", "units", points)


def market(september_2025_on_2024: float) -> Series:
    """Every month is 100, except September 2025."""
    points = {date(y, m, 1): 100.0 for y in range(2014, 2026) for m in range(1, 13)}
    points[date(2025, 9, 1)] = 100 * (1 + september_2025_on_2024)
    return Series.from_points("the market", "units", points)


Q = "Sales moved in September 2025. Is that normal?"


def test_seasonal_check_uses_median_and_range_of_reference_years_without_2020_and_2021():
    answer = ask(store(-0.10), Q)

    assert answer.status == "answered"
    # Reference changes, 2020 and 2021 left out: -13 -12 -11 -10 -10 -9 -8 -7 (%). Median -10, range -13 to -7.
    assert "over 8 reference years (2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024)" in answer.text
    assert "median of -10.0%" in answer.text
    assert "(lowest -13.0%, highest -7.0%)" in answer.text
    assert "is normal for the season" in answer.text
    assert answer.all_claims_passed


@pytest.mark.parametrize("change, verdict", [(-0.20, "weaker than in any reference year"),
                                             (+0.05, "stronger than in any reference year"),
                                             (-0.13, "normal for the season")])  # the lowest year is still normal
def test_seasonal_verdict(change, verdict):
    assert f"is {verdict}" in ask(store(change), Q).text


@pytest.mark.parametrize("market_change, verdict", [(-0.030, "in line with the market"),
                                                    (+0.050, "behind the market"),
                                                    (-0.100, "ahead of the market"),
                                                    (-0.022, "in line with the market")])  # gap of exactly 1.0 point
def test_market_verdict(market_change, verdict):
    # Own growth on September 2024: 0.90 / 0.93 - 1 = -3.2%.
    answer = ask(store(-0.10).with_benchmark(market(market_change)), Q)

    assert "sales changed by -3.2%" in answer.text
    assert f"sales are {verdict}" in answer.text
    assert answer.all_claims_passed


def test_without_a_benchmark_the_market_check_is_skipped():
    answer = ask(store(-0.10), Q)

    assert "Market check: not available" in answer.text
    assert "market_verdict" not in {c.claim.measure for c in answer.claims}


def test_declines_with_fewer_than_five_reference_years():
    assert ask(store(-0.10, start_year=2022), Q).status == "declined"


def test_what_happened_questions_still_route_to_what_happened():
    answer = ask(store(-0.10), "What happened in September 2025?")

    assert "Seasonal check" not in answer.text
    assert "down 10" in answer.text


def test_real_census_data_shoe_stores_september_2026():
    """Checked by hand from data/census_retail (FRED snapshot of 22 Nov 2026):
    2,500 / 3,784 - 1 = -33.9%. September vs August, 2016-2019 and 2022-2025:
    -28.4, -29.9, -30.2, -35.5, -25.1, -28.6, -33.3, -37.6 -> median -30.1, range -37.6 to -25.1.
    2,500 / 2,407 - 1 = +3.9%; total retail 637,194 / 595,443 - 1 = +7.0%; gap -3.1 points."""
    answer = ask(load_census_retail("MRTSSM4482USN"), "Is September 2026 normal for shoe stores?")

    shown = {c.claim.measure: c.claim.shown for c in answer.claims}
    assert shown == {"level": "2,500", "month_change": "-33.9%", "seasonal_typical": "-30.1%",
                     "seasonal_low": "-37.6%", "seasonal_high": "-25.1%", "seasonal_verdict": "normal for the season",
                     "year_on_year": "+3.9%", "market_year_on_year": "+7.0%", "market_verdict": "behind the market"}
    assert answer.all_claims_passed


@pytest.mark.parametrize("rows, problem", [
    (["2024-01-01,5", "2024-03-01,6"], "gap"),
    (["2024-01-01,5", "2024-01-01,6"], "duplicate"),
    (["2024-01-01,5", "2024-02-01,."], "missing"),
    (["2024-01-01,5", "2024-02-01,0"], "non-positive"),
])
def test_loader_refuses_unclean_series(tmp_path, rows, problem):
    path = tmp_path / "X.csv"
    path.write_text("\n".join(["observation_date,X", *rows]) + "\n")

    with pytest.raises(ValueError, match=problem):
        load_fred_csv(path, "X")
