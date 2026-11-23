"""Real KPI series that events are injected into ("backgrounds").

Two licence-clean sources, both chosen because a consultant would recognise them as client KPIs:

- **UCI Online Retail II** (CC BY 4.0, Chen 2019, https://doi.org/10.24432/C5CG6D): every transaction
  of a UK online retailer, Dec 2009 to Dec 2011. `derive_online_retail` turns it into daily
  revenue, orders and customers per trading day (the shop does not trade on Saturdays), for the
  total and the largest countries. The small derived CSV is committed with attribution; the
  45 MB source is downloaded only to rebuild it.
- **Context is Key** (Apache-2.0, ServiceNow): the histories of its ATM withdrawal, highway traffic
  and electricity demand tasks. Only the history is used, and task families whose history was
  edited by CiK (spikes, maintenance) are left out, so a background starts without an event.
  This is the dataset our TimeNet connector (PR 788) onboards.

A background is never edited here; events are added in events.py, on a copy.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/kpi_triage")

RETAIL_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RETAIL_ZIP_SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
RETAIL_CSV = DATA_DIR / "online_retail_daily.csv"
RETAIL_COUNTRIES = ("United Kingdom", "Germany", "France", "EIRE", "Netherlands", "Spain", "Belgium", "Switzerland")

CIK_COMMIT = "32da467ade142a16d099675e78b1129745c5ef7d"
CIK_URL = f"https://huggingface.co/datasets/ServiceNow/context-is-key/resolve/{CIK_COMMIT}/data/test-00000-of-00001.parquet"
CIK_SHA256 = "f04bebe14b792bb47acfd3dccacb8469ac49f98eb601880e640407e177caad60"
CIK_PATH = DATA_DIR / "raw" / "cik-test.parquet"

# CiK task families whose series a consultant would read as an operational KPI, and their domain.
CIK_DOMAINS = (
    (re.compile(r"ATM|Withdrawal"), "atm", "ATM cash withdrawals", "normalised units", "daily", 7),
    (re.compile(r"Traffic|Sensor|Constraint", re.I), "traffic", "Highway lane occupancy", "%", "hourly", 24),
    (re.compile(r"Electricity"), "electricity", "Electricity demand", "kW", "hourly", 24),
)
# Families whose history CiK edited on purpose; a background must start clean.
CIK_EDITED_HISTORY = re.compile(r"Spike|Maintenance|TrendAccumulation")


@dataclass(frozen=True, eq=False)
class Background:
    key: str  # unique per underlying series
    source: str  # "online_retail" or "cik"
    group: str  # what splits hold out together: a country for retail, a domain for CiK
    name: str
    unit: str
    freq: str  # word used in prompts: "daily", "hourly", "trading-day"
    period: int | None  # seasonal period in points, if any
    timestamps: tuple[str, ...]
    values: np.ndarray

    def __len__(self) -> int:
        return len(self.values)


def _download(url: str, path: Path, sha256: str) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "kpi-triage-data-fetch/0.1"})
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"{path}: sha256 {digest} does not match the pinned {sha256}")
    return path


# --- Online Retail II -------------------------------------------------------------------------


def derive_online_retail(zip_path: Path | None = None, out: Path = RETAIL_CSV) -> Path:
    """Rebuild the committed daily CSV from the original file (needs openpyxl; takes minutes).

    Revenue is quantity times price, net of cancellations (invoices starting with "C"), in GBP.
    Orders are distinct non-cancelled invoices; customers are distinct customer ids on those.
    Only days on which the shop traded at all are kept, so a missing day is a closed day, not a gap.
    """
    import zipfile

    import pandas as pd

    zip_path = _download(RETAIL_URL, zip_path or DATA_DIR / "raw" / "online_retail_ii.zip", RETAIL_ZIP_SHA256)
    with zipfile.ZipFile(zip_path) as z, z.open("online_retail_II.xlsx") as f:
        sheets = pd.read_excel(f, sheet_name=None, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)
    df["Invoice"] = df["Invoice"].astype(str)
    df["day"] = pd.to_datetime(df["InvoiceDate"]).dt.strftime("%Y-%m-%d")
    df["revenue"] = df["Quantity"] * df["Price"]
    sale = ~df["Invoice"].str.startswith("C")
    trading_days = sorted(df["day"].unique())

    def daily(part: "pd.DataFrame", label: str) -> "pd.DataFrame":
        rev = part.groupby("day")["revenue"].sum()
        orders = part[sale.loc[part.index]].groupby("day")["Invoice"].nunique()
        cust = part[sale.loc[part.index]].groupby("day")["Customer ID"].nunique()
        out = pd.DataFrame({"revenue_gbp": rev, "orders": orders, "customers": cust}).reindex(trading_days).fillna(0.0)
        out.insert(0, "segment", label)
        return out.rename_axis("day").reset_index()

    frames = [daily(df, "All countries")] + [daily(df[df["Country"] == c], c) for c in RETAIL_COUNTRIES]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(out, index=False, float_format="%.2f")
    return out


RETAIL_METRICS = (("revenue_gbp", "Online revenue", "GBP"), ("orders", "Orders", "orders"), ("customers", "Customers", "customers"))
# A daily series needs most days active to read as a KPI; sparser segments are left out.
MIN_ACTIVE_SHARE = 0.85


def load_online_retail(path: Path = RETAIL_CSV) -> list[Background]:
    """Daily series where the segment trades on most days; weekly totals (weeks starting Monday)
    for the smaller countries, whose daily series are mostly empty."""
    import pandas as pd

    df = pd.read_csv(path)
    out = []
    for segment, part in df.groupby("segment", sort=False):
        part = part.sort_values("day")
        week = pd.to_datetime(part["day"]).dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
        for col, name, unit in RETAIL_METRICS:
            versions = []
            daily = part[col].to_numpy(dtype=float)
            if np.mean(daily > 0) >= MIN_ACTIVE_SHARE:
                versions.append((tuple(part["day"]), daily, "trading-day", 5))
            # The first and last weeks are partial (the data starts on a Tuesday and ends on a Friday).
            weekly = part.groupby(week.to_numpy())[col].sum().sort_index().iloc[1:-1]
            if np.mean(weekly.to_numpy() > 0) >= MIN_ACTIVE_SHARE:
                versions.append((tuple(weekly.index), weekly.to_numpy(dtype=float), "weekly", None))
            for times, v, freq, period in versions:
                out.append(Background(
                    key=f"online_retail/{segment}/{col}/{freq}", source="online_retail", group=f"retail:{segment}",
                    name=f"{name}, {segment}", unit=unit, freq=freq, period=period, timestamps=times, values=v,
                ))
    return out


# --- Context is Key ---------------------------------------------------------------------------


def load_cik(path: Path | None = None) -> list[Background]:
    """Distinct clean histories from CiK's ATM, traffic and electricity tasks."""
    import pandas as pd

    path = _download(CIK_URL, path or CIK_PATH, CIK_SHA256)
    seen, out = set(), []
    for row in pd.read_parquet(path).itertuples(index=False):
        if CIK_EDITED_HISTORY.search(row.name):
            continue
        match = next((d for d in CIK_DOMAINS if d[0].search(row.name)), None)
        if match is None:
            continue
        _, domain, name, unit, freq, period = match
        (values_by_time,) = json.loads(row.past_time).values()
        points = sorted(values_by_time.items())
        v = np.array([x for _, x in points], dtype=float)
        if len(v) < 96 or not np.all(np.isfinite(v)):
            continue
        digest = hashlib.sha1(np.round(v, 6).tobytes()).hexdigest()[:12]
        if digest in seen:  # CiK reuses one series across task variants
            continue
        seen.add(digest)
        times = tuple(t.replace("T", " ").removesuffix(".000")[:16] for t, _ in points)
        out.append(Background(
            key=f"cik/{domain}/{digest}", source="cik", group=f"cik:{domain}", name=name, unit=unit,
            freq=freq, period=period, timestamps=times, values=v,
        ))
    return out


def load_all() -> list[Background]:
    return load_online_retail() + load_cik()
