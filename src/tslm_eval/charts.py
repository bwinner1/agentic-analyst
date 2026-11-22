"""Static charts for the results page and the X / LinkedIn posts (PNG, light surface).

Colours and mark specs follow the dataviz reference palette: one highlight hue for our model,
neutral bars for the rest, hairline grid, 2px lines, text in ink tokens (never series colours).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL_BAR = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3: validated all-pairs, light mode
DPI = 150

def _sans() -> str:
    """First installed system sans; DejaVu Sans ships with matplotlib."""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in ("Segoe UI", "Helvetica Neue", "Helvetica", "Arial") if f in installed), "DejaVu Sans")


plt.rcParams.update({
    "font.family": _sans(),
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": INK_2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _frame(ax, grid_axis: str) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _title(fig, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.97, title, ha="left", va="top", fontsize=14, fontweight="semibold", color=INK)
    fig.text(0.02, 0.97 - 0.075 * 6 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=10, color=INK_2)


def bar_chart(summary, path: Path, highlight: str | None = None, subtitle: str = "",
              title: str = "Forecasting with text context: error by system") -> None:
    """Horizontal bars, best (lowest RCRPS) on top, 95% interval as a thin whisker."""
    s = summary.sort_values("rcrps", ascending=False).reset_index(drop=True)  # barh draws bottom-up
    n = len(s)
    fig, ax = plt.subplots(figsize=(10, 1.9 + 0.55 * n), dpi=DPI)
    y = np.arange(n)
    colours = [SERIES[0] if name == highlight else NEUTRAL_BAR for name in s.system]
    ax.barh(y, s.rcrps, height=0.32, color=colours)
    ax.hlines(y, s.ci_low, s.ci_high, color=INK_2, linewidth=1)
    xmax = float(max(s.ci_high.max(), s.rcrps.max()))
    for yi, r in zip(y, s.itertuples()):
        ax.text(max(r.rcrps, r.ci_high) + xmax * 0.015, yi, f"{r.rcrps:.3f}", va="center", ha="left",
                color=INK, fontsize=10, fontweight="semibold" if r.system == highlight else "normal")
    labels = [f"{name}  (ours)" if name == highlight else name for name in s.system]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, xmax * 1.15)
    ax.set_xlabel("RCRPS, lower is better")
    _frame(ax, "x")
    _title(fig, title,
           "\n".join(textwrap.wrap(subtitle, 130) + ["Bars: weighted mean. Whiskers: 95% interval."]))
    fig.subplots_adjust(left=0.24, right=0.97, top=1 - 1.15 / fig.get_figheight(), bottom=0.5 / fig.get_figheight() + 0.08)
    fig.savefig(path)
    plt.close(fig)


def example_chart(inst, forecasts: dict, path: Path, scores: dict | None = None) -> None:
    """History, the true future, and each system's forecast for one instance.

    The first forecast is drawn on top. A forecast identical to an earlier one would be hidden under
    it, so its legend entry says so. `scores` (system -> RCRPS) are shown in the legend.
    """
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=DPI)
    n_past, h = len(inst.past), inst.horizon
    x_past = np.arange(n_past)
    x_future = np.arange(n_past, n_past + h)
    if len(inst.roi):
        lo, hi = n_past + inst.roi.min() - 0.5, n_past + inst.roi.max() + 0.5
        ax.axvspan(lo, hi, color=GRID, alpha=0.6, linewidth=0)
        ax.text((lo + hi) / 2, 1.0, "period the text is about", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=MUTED, fontsize=9)
    ax.axvline(n_past - 0.5, color=AXIS, linewidth=1)
    ax.plot(x_past, inst.past, color=MUTED, linewidth=2, solid_capstyle="round", label="History")
    ax.plot(x_future, inst.future, color=INK, linewidth=2, solid_capstyle="round", label="What happened")
    series = {name: np.asarray(fc, dtype=float) for name, fc in forecasts.items() if fc is not None}
    series = {name: v if v.ndim == 1 else np.median(v, axis=0) for name, v in series.items()}
    everything = np.concatenate([inst.past, inst.future, *series.values()])
    visible = 0.005 * (np.nanmax(everything) - np.nanmin(everything))  # less than this apart looks like one line
    drawn: dict[str, np.ndarray] = {}
    for k, (colour, name) in enumerate(zip(SERIES, forecasts)):
        if name not in series:
            continue
        values = series[name]
        label = name if not scores or name not in scores else f"{name} ({scores[name]:.2f})"
        twin = next((other for other, v in drawn.items() if np.max(np.abs(v - values)) <= visible), None)
        if twin:
            label += f", hidden under {twin}"
        ax.plot(x_future, values, color=colour, linewidth=2, solid_capstyle="round", label=label,
                zorder=10 - k)
        drawn[name] = values
    times = list(inst.past_times) + list(inst.future_times)
    ticks = np.linspace(0, len(times) - 1, 6).round().astype(int)
    width = 10 if all(t.endswith("00:00:00") for t in times) else 16  # date only for daily or slower data
    ax.set_xticks(ticks, [times[t][:width] for t in ticks])
    tick_labels = ax.get_xticklabels()
    tick_labels[0].set_horizontalalignment("left")
    tick_labels[-1].set_horizontalalignment("right")
    ax.set_xlim(-0.5, len(times) - 0.5)
    _frame(ax, "y")
    labels = ax.get_legend_handles_labels()[1]
    longest = max(len(t) for t in labels)
    ncol = len(labels) if sum(len(t) for t in labels) < 110 else (3 if longest <= 40 else 2)
    rows = -(-len(labels) // ncol)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=ncol, frameon=False,
              labelcolor=INK_2, handlelength=1.6)
    context = textwrap.shorten(inst.scenario or inst.background, 220, placeholder=" ...")
    wrapped = "\n".join(textwrap.wrap(f"Text given: {context}", 120))
    fig.text(0.02, 0.97, f"Example: {inst.task} (seed {inst.seed})", ha="left", va="top",
             fontsize=13, fontweight="semibold", color=INK)
    fig.text(0.02, 0.91, wrapped, ha="left", va="top", fontsize=9, color=INK_2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.12 + 0.05 * rows)
    fig.savefig(path)
    plt.close(fig)
