"""Write the business explanations for all 355 Context is Key instances (no API needed).

Each task family has its own hand-written explanation logic. Every instance's dates, durations,
multipliers and levels come from its own context text and its own data, so the 5 seeds of a task get
different explanations. Run:

    python build_rationales.py data/cik_test.parquet cik_rationales.jsonl
"""

import json
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def parse_ts(s):
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(s)


def day(t):
    return f"{t:%b} {t.day}"


def hour(t):
    return f"{t:%H:%M} on {day(t)}"


def month(t):
    return f"{t:%B}"


def num(x):
    x = float(x)
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}".rstrip("0").rstrip(".")
    if abs(x) >= 1:
        return f"{x:.2f}".rstrip("0").rstrip(".")
    return f"{x:.2g}"


def pct(a, b):
    return abs(a - b) / abs(b) * 100 if b else 0.0


class Inst:
    def __init__(self, row):
        self.row = row
        self.ctx = " ".join(str(row[k] or "") for k in ("background", "scenario", "constraints"))
        past, fut = json.loads(row["past_time"]), json.loads(row["future_time"])
        self.target = list(fut)[0]
        self.ft = [parse_ts(k) for k in fut[self.target]]
        self.f = np.array(list(fut[self.target].values()), float)
        p = past.get(self.target) or list(past.values())[-1]
        self.ht = [parse_ts(k) for k in p]
        self.h = np.array(list(p.values()), float)
        self.period = int(row["seasonal_period"] or -1)
        self.roi = [int(i) for i in row["region_of_interest"]]

    def peak(self):
        i = int(np.argmax(self.f))
        return self.ft[i], self.f[i]

    def trough(self):
        i = int(np.argmin(self.f))
        return self.ft[i], self.f[i]

    def mask_between(self, start, end):
        return np.array([start <= t < end for t in self.ft])

    def last_season(self):
        p = self.period if 1 < self.period <= len(self.h) else len(self.h)
        return self.h[-p:]


def first(pattern, text, cast=str, flags=re.S):
    m = re.search(pattern, text, flags)
    return cast(m.group(1)) if m else None


# --------------------------------------------------------------------------------------
# Families
# --------------------------------------------------------------------------------------


def atm_outage(x: Inst, reason):
    start = parse_ts(first(r"from (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d), for", x.ctx))
    days = int(first(r"for (\d+) days?", x.ctx))
    end = start + pd.Timedelta(days=days)
    out = x.mask_between(start, end)
    normal = x.f[~out]
    span = "that day" if days == 1 else f"those {days} days"
    return (
        f"{reason} from {day(start)} for {days} day{'s' if days > 1 else ''}, so withdrawals fall to "
        f"essentially zero for {span}. Outside that window, demand follows the machine's usual weekly rhythm "
        f"at roughly {num(np.median(normal))} withdrawals a day, with busier days reaching about {num(np.percentile(normal, 90))}."
    )


def atm_maintenance(x: Inst, explicit):
    hist_nonzero = x.h[x.h > 1]
    what = ("the regular 7-day maintenance shutdowns every 15 days" if explicit
            else "past maintenance shutdowns")
    return (
        f"The stretches of zero withdrawals in the history were caused by {what}, not by a lack of demand, "
        f"and no more maintenance is planned. The forecast therefore ignores those gaps and follows normal "
        f"demand of about {num(np.median(x.f))} withdrawals a day, in line with the {num(np.median(hist_nonzero))} "
        f"a day seen when the machine was working, rather than the lower raw average of {num(np.mean(x.h))}."
    )


def atm_carnival(x: Inst):
    start = parse_ts(first(r"from (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d), for", x.ctx))
    days = int(first(r"for (\d+) days?", x.ctx))
    mult = first(r"(\d+(?:\.\d+)?) times the number of usual", x.ctx)
    win = x.mask_between(start, start + pd.Timedelta(days=days))
    return (
        f"A carnival brings extra people to the area from {day(start)} for {days} days, and the briefing puts "
        f"withdrawals at {mult} times their usual level during that period. The forecast therefore jumps to "
        f"around {num(np.median(x.f[win]))} withdrawals a day during the carnival, against a normal level of "
        f"about {num(np.median(x.f[~win]))} before and after it."
    )


def traffic_bounds(x: Inst):
    lo = first(r"bounded below by (-?\d+(?:\.\d+)?)", x.ctx, float)
    hi = first(r"bounded above by (-?\d+(?:\.\d+)?)", x.ctx, float)
    pt, pv = x.peak()
    tt, tv = x.trough()
    bounds = " and ".join(b for b in [f"never below {num(lo)}" if lo is not None else "",
                                       f"never above {num(hi)}" if hi is not None else ""] if b)
    return (
        f"Road occupancy keeps its usual daily rhythm, quiet overnight and busier during the day. The briefing "
        f"sets firm limits on the forecast ({bounds}), so the forecast is held inside that band, peaking at "
        f"about {num(pv)}% around {pt:%H:%M} and bottoming out near {num(tv)}% around {tt:%H:%M}."
    )


def traffic_accident(x: Inst):
    share = first(r"there is (\d+(?:\.\d+)?)% of the usual traffic", x.ctx)
    start = parse_ts(first(r"traffic from (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", x.ctx))
    hrs = int(first(r"for (\d+) hours?", x.ctx))
    win = x.mask_between(start, start + pd.Timedelta(hours=hrs))
    usual = x.h[-24:][[t.hour in {u.hour for u, w in zip(x.ft, win) if w} for t in x.ht[-24:]]]
    return (
        f"An accident cuts traffic to {share}% of its usual level from {hour(start)} for {hrs} hours. "
        f"During that window occupancy is forecast to fall to around {num(np.mean(x.f[win]))}%, compared with "
        f"roughly {num(np.mean(usual))}% at the same hours on a normal day. Before and after the accident, "
        f"the road follows its normal daily pattern."
    )


def weather_events(x: Inst):
    ev = re.findall(r"At (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d), we expect that the weather will become (clear|cloudy)", x.ctx)
    return [(parse_ts(t), w) for t, w in ev]


def describe_weather(ev):
    if not ev:
        return "no change in the weather"
    parts = [f"{w} from {t:%H:%M}" for t, w in ev[:4]]
    return ", then ".join(parts)


def solar_cloud(x: Inst, kind):
    ev = weather_events(x)
    pt, pv = x.peak()
    sun = [t for t, v in zip(x.ft, x.f) if v > 1]
    rise, sett = (sun[0], sun[-1]) if sun else (pt, pt)
    if kind == "DNI":
        why = ("Direct sunlight is strongest in clear spells and collapses when clouds block the sun, so "
               "output is concentrated in the clear hours")
    else:
        why = ("Diffuse light comes from sunlight scattered by clouds and the air, so it stays moderate "
               "through the day and does not need clear sky to appear")
    return (
        f"Sunlight is zero overnight and the usable day runs from about {rise:%H:%M} to {sett:%H:%M}. "
        f"The weather outlook for the forecast day is {describe_weather(ev)}. {why}, peaking at about "
        f"{num(pv)} W/m² around {pt:%H:%M}."
    )


def clearsky(x: Inst, kind):
    pts = [(parse_ts(t), float(v)) for t, v in re.findall(r"\((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d), (-?\d+(?:\.\d+)?)\)", x.ctx)]
    day0 = x.ft[0].date()
    cs = max((v for t, v in pts if t.date() == day0), default=max(v for _, v in pts))
    pt, pv = x.peak()
    share = pv / cs * 100 if cs else 0
    if share > 85:
        verdict = "close to the clear-sky ceiling, so the forecast assumes a mostly clear day"
    elif share > 50:
        verdict = "well below the clear-sky ceiling, so the forecast allows for some cloud cover"
    else:
        verdict = "far below the clear-sky ceiling, so the forecast assumes a largely cloudy day"
    name = "direct sunlight" if kind == "DNI" else "total sunlight"
    return (
        f"The clear-sky values in the briefing set the most {name} this location can receive, about {num(cs)} W/m² "
        f"at midday on the forecast day. The forecast is zero overnight and peaks at about {num(pv)} W/m² "
        f"around {pt:%H:%M}, {verdict}."
    )


def heatwave(x: Inst, variant):
    if variant == "news":
        start = first(r"began on (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", x.ctx)
        hrs = first(r"lasted for approximately (\d+) hours?", x.ctx, int)
        mult = first(r"approximately (\d+(?:\.\d+)?) times the typical", x.ctx)
    else:
        m = re.search(r"heat wave in city A from (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) for (\d+) hours?(.*?)(\d+(?:\.\d+)?) times the usual electricity", x.ctx, re.S)
        start, hrs = m.group(1), int(m.group(2))
        if variant == "split":
            mult = first(r"usage of only (\d+(?:\.\d+)?) times", x.ctx)
            typical = first(r"typically lead to.*?(\d+(?:\.\d+)?) times", x.ctx)
        else:
            mult = m.group(4)
    start = parse_ts(start)
    win = x.mask_between(start, start + pd.Timedelta(hours=hrs))
    peak = np.max(x.f[win])
    usual = np.median(x.f[~win])
    lead = f"A heat wave from {hour(start)} lasting {hrs} hour{'s' if hrs > 1 else ''} drives air-conditioning use to about {mult} times normal."
    if variant == "split":
        lead = (f"A heat wave from {hour(start)} lasting {hrs} hour{'s' if hrs > 1 else ''} would normally push use to "
                f"{typical} times normal, but residents are conserving energy, so the spike is only about {mult} times normal.")
    elif variant == "distractor":
        lead += " The other events in the briefing (past grid issues, cold spells or festivals elsewhere) do not affect this city's demand in the forecast window."
    elif variant == "news":
        lead = (f"The news report describes a heat wave starting at {hour(start)} and lasting about {hrs} hour{'s' if hrs > 1 else ''}, "
                f"with consumption reaching roughly {mult} times typical levels.")
    return (
        f"{lead} The forecast therefore spikes to about {num(peak)} kW during those hours, then returns to the "
        f"usual daily pattern, which hovers around {num(usual)} kW."
    )


def _tod(t):
    return t.hour * 3600 + t.minute * 60 + t.second


def _segments_in_window(x: Inst, changes):
    """changes: [(seconds_of_day or None, value)] in order. Returns [(start_sec, value, truth_mean)]
    for the segments that are active during the forecast window."""
    lo, hi = _tod(x.ft[0]), _tod(x.ft[-1])
    secs = np.array([_tod(t) for t in x.ft])
    out = []
    for i, (s, v) in enumerate(changes):
        s0 = -1 if s is None else s
        e0 = changes[i + 1][0] if i + 1 < len(changes) else 10**9
        if e0 <= lo or s0 > hi:
            continue
        m = (secs >= s0) & (secs < e0)
        out.append((max(s0, lo), v, float(x.f[m].mean()) if m.any() else None))
    return out


def _hms(sec):
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _to_sec(hms):
    h, m, s = (int(p) for p in hms.split(":"))
    return h * 3600 + m * 60 + s


def pressure(x: Inst, explicit):
    pmax = first(r"maximal pressure is (\d+(?:\.\d+)?)", x.ctx, float) or 37.5
    smax = first(r"maximal fan speed is (\d+(?:\.\d+)?)", x.ctx, float) or 3000
    speeds = [float(v) for v in re.findall(r"(?:starts at|changes to) (\d+(?:\.\d+)?)", x.ctx)]
    times = [_to_sec(t) for t in re.findall(r"At (\d\d:\d\d:\d\d)", x.ctx)]
    segs = _segments_in_window(x, list(zip([None] + times, speeds)))
    parts = []
    for k, (s, v, _) in enumerate(segs):
        est = pmax * (v / smax) ** 2
        when = "at the start of the window" if k == 0 else f"from {_hms(s)}"
        parts.append(f"{num(v)} rpm {when}, which implies about {num(est)} Pa")
    law = ("By the affinity law, pressure over its 37.5 Pa maximum scales with the square of speed over the "
           "3,000 rpm maximum" if explicit else "Fan pressure rises with the square of fan speed")
    law = law.replace("37.5", num(pmax)).replace("3,000", num(smax))
    return (
        f"{law}, so each speed setting maps to a pressure level. During the forecast the fan runs at "
        + "; then ".join(parts)
        + f". Pressure moves toward each level with a short lag and some noise, so the forecast ranges from "
        f"about {num(x.f.min())} to {num(x.f.max())} Pa."
    )


def speed_from_load(x: Inst):
    sched = re.findall(r"(\d(?:\.\d+)?) (?:until|from) (\d\d:\d\d:\d\d)", x.ctx)
    # first entry is "<load> until T"; later ones are "<load> from T until ..."
    changes, prev_until = [], None
    for m in re.finditer(r"(\d(?:\.\d+)?) (?:from (\d\d:\d\d:\d\d) )?until (\d\d:\d\d:\d\d)", x.ctx):
        load, frm = float(m.group(1)), m.group(2)
        changes.append((_to_sec(frm) if frm else None, load))
    segs = _segments_in_window(x, changes)
    parts = []
    for k, (s, load, mean) in enumerate(segs):
        when = "at the start of the window" if k == 0 else f"from {_hms(s)}"
        if load == 0:
            lvl = "the fan is off, so speed falls toward zero"
        else:
            lvl = f"speed settles around {num(mean)} rpm" if mean is not None else f"about {num(load * 3000)} rpm"
        parts.append(f"load {num(load)} {when} ({lvl})")
    return (
        "Fan speed scales roughly linearly with load, up to 3,000 rpm at full load, and the fan never spins "
        "below a minimum speed unless it is switched off. During the forecast the plan is: "
        + "; then ".join(parts)
        + (". Each change shows up after a few seconds as the fan spins up or down." if len(parts) > 1
           else ", with no change during the window, so the forecast stays near that level.")
    )


def solar_plant(x: Inst, ref=False):
    pt, pv = x.peak()
    on = [t for t, v in zip(x.ft, x.f) if v > 0.05 * max(pv, 1e-9)]
    s = (f"A solar plant only produces in daylight, so output is zero overnight, rises from about {on[0]:%H:%M}, "
         f"peaks around {pt:%H:%M} at about {num(pv)} and fades to zero by about {on[-1]:%H:%M}. ")
    if ref:
        rv = first(r"was of (\d+(?:\.\d+)?)", x.ctx)
        rt = first(r"at (\d\d:\d\d:\d\d)", x.ctx)
        s += f"The reference peak of {rv} at {rt[:5]} in similar states anchors both the height and the timing of the curve."
    else:
        s += "With no history for a new plant, the forecast borrows the typical daily shape of plants in neighbouring states with a similar climate."
    return s


def halfday_solar(x: Inst, info):
    pt, pv = x.peak()
    s = (f"The forecast covers the rest of the day: sunlight continues its daily arc, reaching about {num(pv)} W/m² "
         f"around {pt:%H:%M} and falling to zero after sunset, with the overnight hours flat at zero. ")
    if info == "zenith":
        zt = first(r"on average at (\d\d:\d\d:\d\d)", x.ctx)
        s += f"The briefing places the usual daily maximum at {zt[:5]}, which pins down when the peak arrives."
    elif info == "locale":
        s += "Knowing the location (Florida) helps set realistic day length and peak strength for the season."
    else:
        s += "With no location given, the forecast relies on the shape of the morning seen so far."
    return s


def holiday_traffic(x: Inst, variant):
    days = sorted({t.date() for t in x.ft})
    means = [np.mean([v for t, v in zip(x.ft, x.f) if t.date() == d]) for d in days]
    hol = first(r"Note that (\d{4}-\d\d-\d\d) is a holiday due to ([^.]+)\.", x.ctx)
    hname = first(r"is a holiday due to ([^.]+)\.", x.ctx)
    d0 = parse_ts(str(days[0]))
    if hol:
        lead = f"{day(d0)} is a public holiday ({hname}), and traffic on this freeway drops on holidays."
    else:
        lead = (f"The first forecast day, {day(d0)}, is a public holiday, and traffic on this freeway drops on holidays; "
                f"spotting this needs the calendar, not the briefing.")
    rest = ", ".join(f"{num(m)}% on {d:%A}" for d, m in zip(days[1:], means[1:]))
    return (
        f"{lead} Average occupancy on the holiday is forecast at about {num(means[0])}%, against {rest}, "
        f"with the usual rush-hour peaks muted on the holiday."
    )


def svar(x: Inst, variant):
    segs = re.findall(r"(-?\d+\.\d+) from (\d{4}-\d\d-\d\d) to (\d{4}-\d\d-\d\d)", x.row["scenario"])
    fstart = x.ft[0]
    fut_segs = [(float(v), parse_ts(a), parse_ts(b)) for v, a, b in segs if parse_ts(b) >= fstart]
    parts = []
    for v, a, b in fut_segs:
        m = np.array([a <= t <= b for t in x.ft])
        if m.any():
            parts.append(f"X_0 at {v:g} from {day(a)} gives X_1 near {np.mean(x.f[m]):.3g}")
    rel = {"full_explicit": "The briefing gives the exact equation linking X_1 to its own past and to X_0",
           "full_implicit": "The briefing explains that X_1 depends on its own last 3 values and on X_0",
           "minimal": "X_1 is driven by its own recent values and by the covariate X_0"}[variant]
    return (
        f"{rel}, so changes in X_0 feed through to X_1 within a few steps. Over the forecast window, "
        + "; ".join(parts)
        + ", with small random noise around that path."
    )


def montreal_seasonal(x: Inst, kind, explicit):
    pt, pv = x.peak()
    total = float(np.sum(x.f))
    yearly = first(r"(?:yearly average number of incidents was|respond to) (\d+)", x.ctx)
    s = f"{kind} follow a strong yearly cycle"
    if explicit:
        pm = first(r"month with the most incidents was (\w+)", x.ctx)
        s += f", and in past years {pm} was the busiest month"
    s += (f". The forecast peaks at about {num(pv)} incidents in {month(pt)} and moves to {num(x.f[-1])} by {month(x.ft[-1])}, "
          f"about {num(total)} incidents over the {len(x.f)} months")
    if yearly:
        s += f", consistent with a typical year of around {yearly}"
    return s + "."


def montreal_confounding(x: Inst, kind, other):
    plan_month = parse_ts(first(r"starting on (\d{4}-\d\d)", x.ctx))
    pt, pv = x.peak()
    return (
        f"{kind} and {other} rise and fall together because they share common causes such as season and weather, "
        f"not because one causes the other. The Mayor's ban on {other.split(' ')[0].lower()}-related activity from "
        f"{month(plan_month)} {plan_month.year} therefore should not reduce {kind.lower()}, so the forecast keeps the normal "
        f"seasonal pattern: about {num(pv)} in {month(pt)}, easing to {num(x.f[-1])} by {month(x.ft[-1])}."
    )


def nautical_analogy(x: Inst):
    landlocked = "landlocked" in x.ctx
    name = first(r"new borough called ([^ (]+)", x.ctx)
    if landlocked:
        return (
            f"{name} is landlocked, so it behaves like the reference borough with no major bodies of water, which "
            f"records almost no nautical rescues. The forecast is therefore about {num(np.mean(x.f))} incidents a month "
            f"for the whole period."
        )
    pt, pv = x.peak()
    return (
        f"{name} borders the Saint-Lawrence River, so it should behave like the reference borough with major "
        f"bodies of water, where rescues happen in the warmer months. The forecast stays near zero until spring "
        f"and rises to about {num(pv)} in {month(pt)}, about {num(np.sum(x.f))} rescues over the six months."
    )


def sensor_maint(x: Inst):
    a, b = re.findall(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", x.ctx)[:2]
    a, b = parse_ts(a), parse_ts(b)
    win = x.mask_between(a, b)
    return (
        f"The sensor goes offline for maintenance from {hour(a)} to {b:%H:%M}, so readings drop to zero in that window "
        f"regardless of real traffic. Outside it, occupancy follows the normal daily pattern, peaking at about "
        f"{num(np.max(x.f[~win]))}%."
    )


def sensor_periodic(x: Inst):
    m = np.array([8 <= t.hour < 14 for t in x.ft])
    return (
        "The daily zeros between 08:00 and 14:00 in the history were the sensor being serviced, not empty roads, and "
        "no more maintenance is planned. The forecast therefore fills those hours with real daytime traffic, about "
        f"{num(np.mean(x.f[m]))}% on average, instead of copying the zeros."
    )


def sensor_spike(x: Inst):
    t = parse_ts(first(r"spike starting from (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", x.ctx))
    return (
        f"The spike at {hour(t)} was a sensor glitch, not real traffic, and it will not repeat. The forecast "
        f"ignores it and follows the normal daily cycle, peaking at about {num(x.f.max())}% around {x.peak()[0]:%H:%M}."
    )


def sensor_trend(x: Inst):
    rate = first(r"increases by (\d+(?:\.\d+)?) at every hour", x.ctx, float)
    a = parse_ts(first(r"starting from (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", x.ctx))
    b = parse_ts(first(r"At timestep (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", x.ctx))
    drift = rate * (b - a).total_seconds() / 3600
    return (
        f"A calibration fault added a false upward drift of {rate} per hour from {hour(a)}, which had inflated readings by "
        f"about {num(drift)} points by the time the sensor was fixed. With the fault gone, the forecast drops back by that "
        f"amount to true occupancy, averaging about {num(np.mean(x.f))}%."
    )


def unemployment(x: Inst, variant):
    last = x.h[-1]
    direction = "rise" if x.f[-1] > last + 0.15 else "fall" if x.f[-1] < last - 0.15 else "hold roughly steady"
    if variant == "single":
        ref = "The state-level series in the briefing"
    elif variant == "explicit":
        ref = "Kansas, the county's own state, is the relevant reference among the states listed, and its recent path"
    else:
        ref = "The state reference series, especially the one for the county's own state,"
    return (
        f"{ref} shows how unemployment has been moving in the wider region, and county rates tend to track their "
        f"state's direction. Starting from {num(last)}% in {month(x.ht[-1])}, the county rate is forecast to {direction}, "
        f"ending near {num(x.f[-1])}% by {month(x.ft[-1])} {x.ft[-1].year}."
    )


# --------------------------------------------------------------------------------------


def explain(row):
    n, x = row["name"], Inst(row)
    if n == "ATMBuildingClosedTask":
        return atm_outage(x, "The building that houses the ATM is closed")
    if n == "CashDepletedinATMScenarioTask":
        return atm_outage(x, "The ATM runs out of cash")
    if n.startswith("ATMUnderPeriodicMaintenance"):
        return atm_maintenance(x, explicit="LessExplicit" not in n)
    if n == "IncreasedWithdrawalScenario":
        return atm_carnival(x)
    if n in ("BoundedPredConstraintsBasedOnPredQuantilesTask", "OraclePredUnivariateConstraintsTask"):
        return traffic_bounds(x)
    if n == "DecreaseInTrafficInPredictionTask":
        return traffic_accident(x)
    if "IrradianceFromCloudStatus" in n:
        return solar_cloud(x, "DNI" if "DirectNormal" in n else "DHI")
    if n == "DirectNormalIrradianceFromClearsky":
        return clearsky(x, "DNI")
    if n == "GlobalHorizontalIrradianceFromClearsky":
        return clearsky(x, "GHI")
    if "NewsElectricity" in n:
        return heatwave(x, "news")
    if n == "ElectricityIncreaseInPredictionWithSplitContext":
        return heatwave(x, "split")
    if n.startswith("ElectricityIncreaseInPredictionWithDistractor"):
        return heatwave(x, "distractor")
    if n == "ElectricityIncreaseInPredictionTask":
        return heatwave(x, "plain")
    if n.endswith("PressureFromSpeedTask"):
        return pressure(x, explicit=n.startswith("Explicit"))
    if n == "SpeedFromLoadTask":
        return speed_from_load(x)
    if n == "SimilarLocationWithReferenceDaySolarForecastTask":
        return solar_plant(x, ref=True)
    if "SimilarLocationDaySolar" in n:
        return solar_plant(x)
    if n.endswith("HalfDaySolarForecastTask"):
        return halfday_solar(x, "zenith" if n.startswith("Zenith") else "locale" if n.startswith("Locale") else "minimal")
    if "TrafficForecastTaskwithHolidays" in n:
        return holiday_traffic(x, n)
    if n.endswith("BivarLinSVAR"):
        v = "full_explicit" if "ExplicitEquation" in n else "full_implicit" if "Full" in n else "minimal"
        return svar(x, v)
    if "CausalConfounding" in n:
        kinds = {"FieldAndGas": ("Field fires", "gas leaks"), "FieldAndTrash": ("Field fires", "trash fires"),
                 "TrashAndBicycle": ("Trash fires", "bicycle accidents"), "TrashAndNautical": ("Trash fires", "nautical rescues")}
        for k, (a, b) in kinds.items():
            if k in n:
                return montreal_confounding(x, a, b)
    if "NauticalRescueAnalogy" in n:
        return nautical_analogy(x)
    if n.endswith("ShortHistoryTask"):
        kind = {"FieldFire": "Field fires", "IceRescue": "Ice rescues", "NauticalRescue": "Nautical rescues",
                "TrashFire": "Trash fires"}[re.search(r"MontrealFire(\w+?)(?:Explicit|Implicit)", n).group(1)]
        return montreal_seasonal(x, kind, explicit="Explicit" in n)
    if n == "SensorMaintenanceInPredictionTask":
        return sensor_maint(x)
    if n == "SensorPeriodicMaintenanceTask":
        return sensor_periodic(x)
    if n == "SensorSpikeTask":
        return sensor_spike(x)
    if n == "SensorTrendAccumulationTask":
        return sensor_trend(x)
    if n.startswith("Unemployment"):
        return unemployment(x, "single" if "Single" in n else "explicit" if "Explicit" in n else "multi")
    raise KeyError(n)


def main(src, out):
    df = pd.read_parquet(src)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        failed = 0
        for _, row in df.iterrows():
            row = row.to_dict()
            try:
                text = explain(row)
            except Exception as exc:  # an unusual generated instance; it is then left out of training
                failed += 1
                if failed <= 5:
                    print(f"  could not explain {row['name']} seed {row['seed']}: {type(exc).__name__}: {exc}")
                continue
            f.write(json.dumps({"task_name": row["name"], "seed": int(row["seed"]), "rationale": text}) + "\n")
            n += 1
    print(f"wrote {n} explanations to {out}" + (f" ({failed} instances skipped)" if failed else ""))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
