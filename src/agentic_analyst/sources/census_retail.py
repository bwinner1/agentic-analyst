"""US Census monthly retail sales by kind of business, as published on FRED.

Source: U.S. Census Bureau, Monthly Retail Trade Survey, via FRED (https://fred.stlouisfed.org).
US federal data, public domain. Values are sales in millions of US dollars, in current prices (not adjusted
for inflation) and NOT seasonally adjusted: the seasonal pattern is kept on purpose, because the
"in context" answer type compares a month with the same month in earlier years.

`data/census_retail/` holds a snapshot taken on 22 Nov 2026, one FRED CSV per series (2014-01 onwards).
`fetch_census_retail` downloads fresh copies in the same format.
"""
from __future__ import annotations

import csv
import urllib.request
from datetime import date
from pathlib import Path

from ..series import Series

UNIT = "USD millions"

# Store types a consumer goods client would compare itself with. FRED id -> series name.
CATEGORIES = {
    "MRTSSM448USN": "Clothing and accessories store sales (US)",
    "MRTSSM4482USN": "Shoe store sales (US)",
    "MRTSSM446USN": "Health and personal care store sales (US)",
    "MRTSSM4451USN": "Grocery store sales (US)",
}
# The market every category is compared with.
MARKET_ID = "RSXFSN"
MARKET_NAME = "total US retail (excluding food services)"

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "census_retail"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&cosd={start}"


def load_fred_csv(path: str | Path, name: str, unit: str = UNIT) -> Series:
    """Read one FRED CSV (columns observation_date, <id>). Refuses gaps, duplicates and non-positive values."""
    points: dict[date, float] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) != 2 or header[0] != "observation_date":
            raise ValueError(f"{path}: not a FRED CSV (header {header})")
        for row in reader:
            if not row or row[1] in ("", "."):  # FRED writes "." for a missing value
                raise ValueError(f"{path}: missing value in row {row}")
            d = date.fromisoformat(row[0])
            if d in points:
                raise ValueError(f"{path}: duplicate month {d}")
            v = float(row[1])
            if v <= 0:
                raise ValueError(f"{path}: non-positive value {v} for {d}")
            points[d] = v
    series = Series.from_points(name, unit, points)
    for a, b in zip(series.dates, series.dates[1:]):
        if (b.year * 12 + b.month) - (a.year * 12 + a.month) != 1:
            raise ValueError(f"{path}: gap between {a} and {b}")
    return series


def load_census_retail(series_id: str, data_dir: str | Path = DEFAULT_DIR) -> Series:
    """One store type, with total retail attached as its benchmark."""
    if series_id not in CATEGORIES:
        raise ValueError(f"unknown series {series_id}; choose one of {sorted(CATEGORIES)}")
    data_dir = Path(data_dir)
    market = load_fred_csv(data_dir / f"{MARKET_ID}.csv", MARKET_NAME)
    return load_fred_csv(data_dir / f"{series_id}.csv", CATEGORIES[series_id]).with_benchmark(market)


def fetch_census_retail(data_dir: str | Path = DEFAULT_DIR, start: str = "2014-01-01") -> list[Path]:
    """Download fresh FRED CSVs for every category and the market. No API key needed."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for series_id in [MARKET_ID, *CATEGORIES]:
        target = data_dir / f"{series_id}.csv"
        with urllib.request.urlopen(FRED_CSV.format(id=series_id, start=start), timeout=30) as r:
            target.write_bytes(r.read())
        load_fred_csv(target, series_id)  # fail loudly if the download is not a clean series
        written.append(target)
    return written


if __name__ == "__main__":
    for p in fetch_census_retail():
        print(f"wrote {p}")
