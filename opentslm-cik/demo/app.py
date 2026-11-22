"""Forecast demo for the OpenTSLM model finetuned on Context is Key.

Pick a CiK example or paste your own series, type a business note, and get a forecast chart plus a
short explanation written by the model.

    python app.py                                   # live model, checkpoint from demo/checkpoints/
    python app.py --ckpt path/to/best_model.pt      # a specific checkpoint
    python app.py --replay ../results/full/test_predictions.jsonl   # no model: replays saved test answers

Then open the address it prints (http://127.0.0.1:7860) in a browser.
The prompt format is the one the model was trained on (cik_core.build_example, code version in ../VERSION).
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time
from datetime import datetime

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Mac: run any op MPS lacks on the CPU instead of failing

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)  # the opentslm-cik folder: cik_core.py, VERSION, data/
sys.path.insert(0, CODE)
import cik_core  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import gradio as gr  # noqa: E402

DEFAULT_PARQUET = os.path.join(CODE, "data", "cik_test.parquet")
CKPT_DIR = os.path.join(HERE, "checkpoints")
BASE_LLM = "meta-llama/Llama-3.2-1B"
MAX_CUSTOM_HORIZON = 60

# Examples where the finetuned model clearly uses the note (from the Nebius test run), listed first.
FEATURED = [
    ("SensorPeriodicMaintenanceTask", 1),
    ("SensorSpikeTask", 1),
    ("SpeedFromLoadTask", 4),
    ("ExplicitDiffuseHorizontalIrradianceFromCloudStatus", 1),
    ("SensorPeriodicMaintenanceTask", 4),
    ("SensorSpikeTask", 4),
]

# Chart colours (validated categorical slots 1-3 of the reference palette; history and baseline stay neutral).
C_FORECAST, C_NONOTE, C_ACTUAL = "#2a78d6", "#eb6834", "#1baf7a"
C_HISTORY, C_BASELINE, C_TEXT, C_GRID = "#52514e", "#9a9892", "#0b0b0b", "#e4e3df"

DRAFT_LABEL = ("**Draft commentary, written by the model.** It can state numbers or dates that are not in "
               "the data. Check it against the chart before you use it.")


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


def pick_device(requested: str | None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Forecaster:
    """Loads Llama-3.2-1B once, then the finetuned parts (time-series encoder, projector, LoRA) from a checkpoint.

    The checkpoint only holds what training changed, so swapping checkpoints is quick and the base model stays loaded.
    """

    def __init__(self, base_llm: str, device: str):
        self.base_llm, self.device = base_llm, device
        self.model = None
        self.ckpt = None

    def load(self, ckpt: str):
        import torch

        if self.model is not None and self.ckpt == ckpt:
            return
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if self.model is None:
            from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP

            print(f"Loading {self.base_llm} on {self.device} (first time downloads about 2.5 GB)...")
            self.model = OpenTSLMSP(llm_id=self.base_llm, device=self.device)
            r = 16
            cfg = state.get("lora_config")
            if cfg:
                r = getattr(next(iter(cfg.values())), "r", r)
            if state.get("lora_enabled"):
                self.model.enable_lora(lora_r=r, lora_alpha=2 * r)
        del state
        self.model.load_from_file(ckpt)
        self.model.eval()
        self.ckpt = ckpt
        print(f"Checkpoint ready: {ckpt}")

    def run(self, example: cik_core.Example) -> tuple[str, float]:
        import torch
        from opentslm.model_config import PATCH_SIZE
        from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

        sample = {
            "pre_prompt": example.pre_prompt,
            "time_series_text": example.series_texts,
            "time_series": [np.asarray(s, dtype=np.float32) for s in example.series],
            "post_prompt": example.post_prompt,
            "answer": "",
        }
        batch = extend_time_series_to_match_patch_size_and_aggregate([sample], patch_size=PATCH_SIZE)
        h = sum(len(t["values"]) for t in example.meta["targets"])
        start = time.time()
        with torch.no_grad():
            text = self.model.generate(
                batch, max_new_tokens=10 * h + 250, min_new_tokens=4 * h,
                do_sample=False, pad_token_id=self.model.tokenizer.eos_token_id,
            )[0]
        return text, time.time() - start


# --------------------------------------------------------------------------------------
# Inputs: CiK examples and pasted series
# --------------------------------------------------------------------------------------


def load_cik(parquet: str):
    """CiK rows of the held-out test split (the model never saw these task types), featured ones first."""
    rows = cik_core.load_rows(parquet)
    _, _, test = cik_core.split_by_task(rows)
    order = {k: i for i, k in enumerate(FEATURED)}
    test.sort(key=lambda r: (order.get((r["name"], int(r["seed"])), 99), r["name"], int(r["seed"])))
    return {label(r): r for r in test}


def label(row) -> str:
    star = "★ " if (row["name"], int(row["seed"])) in FEATURED else ""
    return f"{star}{row['name']} (seed {int(row['seed'])})"


FREQS = {"Hourly": ("h", 24), "Daily": ("D", 7), "Weekly": ("W", 52), "Monthly": ("MS", 12),
         "Every 15 minutes": ("15min", 96), "Every 30 minutes": ("30min", 48)}


def parse_pasted(text: str, start: str, freq_name: str) -> tuple[list[str], np.ndarray]:
    """Accept one number per line, or 'date,value' lines (comma, tab or semicolon). Returns timestamps and values."""
    stamps, values = [], []
    for raw in text.replace(";", ",").replace("\t", ",").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]
        try:
            v = float(parts[-1])
        except ValueError:
            if not values:
                continue  # a header line such as "date,sales"
            raise gr.Error(f"Could not read a number in this line: {line!r}")
        values.append(v)
        stamps.append(parts[0] if len(parts) > 1 else None)
    if len(values) < 8:
        raise gr.Error("Paste at least 8 values (one per line).")
    if all(s is not None for s in stamps):
        idx = pd.to_datetime(stamps, errors="coerce")
        if idx.isna().any():
            raise gr.Error("Some dates could not be read. Use a format like 2024-05-31 or 2024-05-31 14:00.")
    else:
        code = FREQS.get(freq_name, ("D", 7))[0]
        try:
            first = pd.Timestamp(start or "2024-01-01")
        except ValueError:
            raise gr.Error("Start date not understood. Use a format like 2024-01-01.")
        idx = pd.date_range(first, periods=len(values), freq=code)
    return [fmt_stamp(t) for t in idx], np.asarray(values, float)


def fmt_stamp(t: pd.Timestamp) -> str:
    if (t.hour, t.minute, t.second) == (0, 0, 0):
        return t.strftime("%Y-%m-%d")
    return t.strftime("%Y-%m-%d %H:%M" if t.second == 0 else "%Y-%m-%d %H:%M:%S")


def future_stamps(stamps: list[str], horizon: int, freq_name: str) -> list[str]:
    idx = pd.to_datetime(stamps)
    freq = pd.infer_freq(idx) if len(idx) >= 3 else None
    if freq is None:
        step = idx[-1] - idx[-2]
        return [fmt_stamp(idx[-1] + step * (i + 1)) for i in range(horizon)]
    return [fmt_stamp(t) for t in pd.date_range(idx[-1], periods=horizon + 1, freq=freq)[1:]]


def custom_row(stamps, values, horizon, season, note, name="value") -> dict:
    fut = future_stamps(stamps, horizon, "")
    return {
        "name": "YourSeries", "seed": 0, "weight": "1",
        "background": "", "scenario": note or "", "constraints": "",
        "seasonal_period": int(season) if season and season > 1 else -1,
        "past_time": json.dumps({name: dict(zip(stamps, values.tolist()))}),
        # Unknown future: placeholder values. Only the count and timestamps reach the prompt.
        "future_time": json.dumps({name: {s: 0.0 for s in fut}}),
        "metric_scaling": 1.0, "region_of_interest": [],
    }


def with_note(row: dict, note: str) -> dict:
    """The CiK row with its text context replaced by the note the user typed (or left as is)."""
    r = dict(row)
    r["background"], r["scenario"], r["constraints"] = "", note or "", ""
    return r


# --------------------------------------------------------------------------------------
# Output: chart and text
# --------------------------------------------------------------------------------------


def make_chart(meta, hist_stamps, hist_values, forecast, baseline, truth=None, nonote=None, title=""):
    t = meta["targets"][0]
    h = len(t["stamps"])
    keep = int(min(len(hist_values), max(3 * h, 48), 300))
    hx = pd.to_datetime(hist_stamps[-keep:])
    fx = pd.to_datetime(t["stamps"])

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    roi = meta.get("region_of_interest") or []
    if roi and len(roi) < h:
        ax.axvspan(fx[min(roi)], fx[max(roi)], color="#2a78d6", alpha=0.07, lw=0,
                   label="Period the context is about")
    ax.plot(hx, hist_values[-keep:], color=C_HISTORY, lw=1.5, label="History")
    ax.axvline(hx[-1], color=C_GRID, lw=1.5)
    if baseline is not None:
        ax.plot(fx, baseline, color=C_BASELINE, lw=1.5, ls="--", label="Baseline: last season repeated")
    if truth is not None:
        ax.plot(fx, truth, color=C_ACTUAL, lw=2, label="What actually happened")
    if nonote is not None:
        ax.plot(fx, nonote, color=C_NONOTE, lw=2, label="Model forecast without the note")
    if forecast is not None:
        ax.plot(fx, forecast, color=C_FORECAST, lw=2.2, label="Model forecast with the note")
    ax.grid(True, color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors="#52514e", labelsize=9)
    n_items = len(ax.get_legend_handles_labels()[1])
    ax.set_title(title, loc="left", fontsize=11, color=C_TEXT, pad=14 + 13 * ((n_items + 2) // 3))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9, labelcolor=C_TEXT)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def table_md(meta, forecast, baseline, truth=None, nonote=None, limit=60) -> str:
    t = meta["targets"][0]
    cols = ["Time", "Forecast"] + (["Without note"] if nonote is not None else []) + ["Baseline"] \
        + (["Actual"] if truth is not None else [])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, s in enumerate(t["stamps"][:limit]):
        s = fmt_stamp(pd.Timestamp(s))
        cells = [s, f"{forecast[i]:.4g}"] + ([f"{nonote[i]:.4g}"] if nonote is not None else []) \
            + [f"{baseline[i]:.4g}"] + ([f"{truth[i]:.4g}"] if truth is not None else [])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def score_line(meta, forecast, baseline, truth) -> str:
    if truth is None:
        return ""
    k = meta["metric_scaling"]
    m = np.abs(forecast - truth).mean() * k
    b = np.abs(baseline - truth).mean() * k
    if abs(m - b) <= 0.02 * max(b, 1e-9):
        verdict = "about as close to what actually happened as simply repeating the last season"
    elif m < b:
        verdict = "closer to what actually happened than simply repeating the last season"
    else:
        verdict = "further from what actually happened than simply repeating the last season"
    return (f"Average error (scaled like the CiK benchmark): model **{m:.3f}**, baseline {b:.3f}. "
            f"The model was {verdict}.")


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------


def build_app(args):
    cik = load_cik(args.parquet)
    labels = list(cik)
    replay = {}
    if args.replay:
        for line in open(args.replay, encoding="utf-8"):
            d = json.loads(line)
            replay[(d["task_name"], int(d["seed"]))] = d
        labels = [lb for lb in labels if (cik[lb]["name"], int(cik[lb]["seed"])) in replay]

    forecaster = None if args.replay else Forecaster(args.base, pick_device(args.device))
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")), key=os.path.getmtime, reverse=True)
    if args.ckpt:
        ckpts = [os.path.abspath(args.ckpt)] + [c for c in ckpts if os.path.abspath(c) != os.path.abspath(args.ckpt)]
    if forecaster is not None and not ckpts:
        raise SystemExit(f"No checkpoint found. Put a .pt file in {CKPT_DIR} or pass --ckpt <file>.")
    if forecaster is not None:
        forecaster.load(ckpts[0])  # load at start, so the first click is fast

    def generate(row, note_on: bool, also_without: bool, ckpt: str | None):
        """Returns (forecast, nonote_forecast, explanation, raw, seconds, meta, ok)."""
        ex = cik_core.build_example(row, include_context=True, rationale="", max_history=512, show_baseline=True)
        if forecaster is None:
            d = replay[(row["name"], int(row["seed"]))]
            fc = np.asarray(next(iter(d["forecast_median"].values())))
            return fc, None, d["explanation"], d["raw_output"], 0.0, ex.meta, bool(d["parsed_ok"])
        forecaster.load(ckpt or ckpts[0])
        text, secs = forecaster.run(ex)
        fc, ok, expl = cik_core.parse_answer(text, ex.meta)
        nonote = None
        if also_without:
            ex2 = cik_core.build_example(row, include_context=False, rationale="", max_history=512, show_baseline=True)
            text2, secs2 = forecaster.run(ex2)
            nonote = next(iter(cik_core.parse_answer(text2, ex2.meta)[0].values()))
            secs += secs2
        return next(iter(fc.values())), nonote, expl, text, secs, ex.meta, ok

    def render(row, title, truth_known, also_without, ckpt):
        fc, nonote, expl, raw, secs, meta, ok = generate(row, True, also_without, ckpt)
        t = meta["targets"][0]
        hist = cik_core.parse_frame(row["past_time"])
        col = next(c for c in hist if c.name == t["name"])
        baseline = cik_core.seasonal_naive(meta)[t["name"]]
        truth = np.asarray(t["values"]) if truth_known else None
        fig = make_chart(meta, col.stamps, col.values, fc, baseline, truth, nonote, title)
        status = []
        if not ok:
            status.append("⚠️ The model did not write every forecast step; missing steps were filled in from their neighbours.")
        if secs:
            status.append(f"Generated in {secs:.0f} s on {forecaster.device}"
                          f" with checkpoint `{os.path.basename(forecaster.ckpt)}`.")
        elif forecaster is None:
            status.append("Replay mode: showing the answer the model gave in the Nebius test run.")
        sl = score_line(meta, fc, baseline, truth)
        if sl:
            status.append(sl)
        explanation = f"{DRAFT_LABEL}\n\n> {expl.strip() or '(the model wrote no explanation)'}"
        return fig, explanation, "\n\n".join(status), table_md(meta, fc, baseline, truth, nonote), raw

    def on_pick(lb):
        row = cik[lb]
        return cik_core.context_text(row)

    def run_cik(lb, note, also_without, ckpt):
        if not lb:
            raise gr.Error("Pick an example first.")
        row = cik[lb] if forecaster is None else with_note(cik[lb], note)
        return render(row, lb.replace("★ ", ""), True, also_without, ckpt)

    def run_custom(text, start, freq_name, horizon, season, note, also_without, ckpt):
        if forecaster is None:
            raise gr.Error("Replay mode only shows saved CiK answers. Start the app without --replay to forecast your own series.")
        stamps, values = parse_pasted(text, start, freq_name)
        horizon = int(horizon or 0)
        if not 1 <= horizon <= MAX_CUSTOM_HORIZON:
            raise gr.Error(f"Choose between 1 and {MAX_CUSTOM_HORIZON} steps to forecast.")
        row = custom_row(stamps, values, horizon, season, note)
        return render(row, "Your series", False, also_without, ckpt)

    sample_series = "\n".join(
        f"{d.strftime('%Y-%m-%d')},{v:.0f}" for d, v in zip(
            pd.date_range("2024-01-01", periods=84, freq="D"),
            200 + 40 * np.sin(np.arange(84) * 2 * np.pi / 7) + np.arange(84) * 0.8))

    version = open(os.path.join(CODE, "VERSION")).read().strip() if os.path.exists(os.path.join(CODE, "VERSION")) else "?"
    with gr.Blocks(title="Context-aware forecasting demo") as app:
        gr.Markdown(
            "# Forecasts that read the business context\n"
            "A small time-series language model (OpenTSLM on Llama 3.2 1B, finetuned on the *Context is Key* "
            "benchmark) reads a series **and** a note written in plain English, then writes a forecast and a short "
            "explanation. The dashed grey line is the simple baseline: the last season repeated.")
        with gr.Row():
            ckpt_dd = gr.Dropdown(ckpts, value=ckpts[0] if ckpts else None, label="Model checkpoint",
                                  visible=forecaster is not None and len(ckpts) > 1, scale=3)
            also = gr.Checkbox(False, label="Also forecast without the note (shows what the note changes; twice as slow)",
                               visible=forecaster is not None, scale=2)
        with gr.Tabs():
            with gr.Tab("Benchmark examples"):
                gr.Markdown("Examples from the test part of Context is Key: task types the model never saw in "
                            "training. ★ marks examples where the note clearly helps. You can edit the note to see "
                            "how the forecast reacts." if forecaster is not None else
                            "Replay mode: saved answers from the test run. Notes cannot be edited here.")
                pick = gr.Dropdown(labels, value=labels[0] if labels else None, label="Example")
                note1 = gr.Textbox(cik_core.context_text(cik[labels[0]]) if labels else "", lines=6,
                                   label="Business context (the note the model reads)", interactive=forecaster is not None)
                go1 = gr.Button("Forecast", variant="primary")
            with gr.Tab("Your own series"):
                gr.Markdown("Paste one value per line, or `date,value` lines (for example straight from a spreadsheet). "
                            "Without dates, set a start date and frequency.")
                with gr.Row():
                    series = gr.Textbox(sample_series, lines=10, label="Series", scale=2)
                    with gr.Column(scale=1):
                        start = gr.Textbox("2024-01-01", label="Start date (only if you pasted values only)")
                        freq = gr.Dropdown(list(FREQS), value="Daily", label="Frequency (only if values only)")
                        horizon = gr.Number(14, precision=0, label=f"Steps to forecast (max {MAX_CUSTOM_HORIZON})")
                        season = gr.Number(7, precision=0, label="Season length (7 for daily data with a weekly pattern, 24 for hourly; 0 = none)")
                note2 = gr.Textbox("A two-day promotion runs on the 8th and 9th day of the forecast and usually lifts sales "
                                   "by about 30%.", lines=3, label="Business note")
                go2 = gr.Button("Forecast my series", variant="primary")
        plot = gr.Plot(show_label=False)
        status = gr.Markdown()
        expl = gr.Markdown()
        with gr.Accordion("Forecast values as a table", open=False):
            table = gr.Markdown()
        with gr.Accordion("What the model wrote (raw output, normalized scale)", open=False):
            raw = gr.Textbox(lines=12, show_label=False)
        gr.Markdown(f"<small>Prompt format: {version}. Forecasts take roughly 10 to 60 seconds on a laptop, "
                    "longer for long horizons.</small>")

        outs = [plot, expl, status, table, raw]
        pick.change(on_pick, pick, note1)
        go1.click(run_cik, [pick, note1, also, ckpt_dd], outs)
        go2.click(run_custom, [series, start, freq, horizon, season, note2, also, ckpt_dd], outs)
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="Checkpoint .pt (default: newest file in demo/checkpoints/).")
    p.add_argument("--replay", default=None, help="test_predictions.jsonl: show saved answers, no model needed.")
    p.add_argument("--parquet", default=DEFAULT_PARQUET, help="CiK parquet (default: ../data/cik_test.parquet).")
    p.add_argument("--base", default=BASE_LLM, help="Base LLM the checkpoint was trained on.")
    p.add_argument("--device", default=None, help="cuda, mps or cpu (default: the best available).")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Also make a temporary public link (gradio.live) to send to others.")
    args = p.parse_args()
    build_app(args).queue(default_concurrency_limit=1).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
