"""Score every prediction file the same way and write the results table and charts."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from . import metric
from .forecasts import read_predictions
from .splits import scenario_group

REFERENCE = "naive"


def score_file(path: Path, instances) -> pd.DataFrame:
    preds = read_predictions(path, instances)
    rows = []
    for inst in instances:
        p = preds[inst.id]
        rows.append({
            "id": inst.id, "task": inst.task, "group": scenario_group(inst.task), "seed": inst.seed,
            "weight": inst.weight, "rcrps": metric.rcrps(inst, p.forecast),
            "valid": p.forecast is not None, "problem": p.problem,
        })
    return pd.DataFrame(rows)


def score_runs(runs_dir: Path, instances) -> dict[str, pd.DataFrame]:
    files = sorted(Path(runs_dir).glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no prediction files in {runs_dir}")
    return {f.stem: score_file(f, instances) for f in files}


def summarize(scored: dict[str, pd.DataFrame], ids: set[str] | None = None) -> pd.DataFrame:
    rows = []
    ref = scored.get(REFERENCE)
    for name, df in scored.items():
        d = df if ids is None else df[df.id.isin(ids)]
        agg = metric.aggregate(d.task.tolist(), d.rcrps.tolist(), d.weight.tolist())
        row = {"system": name, "rcrps": agg.mean, "ci_low": agg.ci_low, "ci_high": agg.ci_high,
               "valid_rate": float(d.valid.mean()), "n_instances": agg.n_instances, "n_tasks": agg.n_tasks,
               "better_than_naive": float("nan")}
        if ref is not None and name != REFERENCE:
            ref_scores = ref.set_index("id").rcrps.reindex(d.id).to_numpy()
            row["better_than_naive"] = float((d.rcrps.to_numpy() < ref_scores).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rcrps").reset_index(drop=True)


def by_group(scored: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Weighted mean RCRPS per scenario group (rows) and system (columns)."""
    cols = {}
    for name, df in scored.items():
        g = df.assign(ws=df.weight * df.rcrps).groupby("group")
        cols[name] = g.ws.sum() / g.weight.sum()
    return pd.DataFrame(cols)


def _meta_lines(runs_dir: Path) -> list[str]:
    lines = []
    for meta_path in sorted(Path(runs_dir).glob("*.meta.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        cost = sum(r.get("cost_usd", 0) for r in m.get("runs", []))
        lines.append(
            f"- `{meta_path.name.removesuffix('.meta.json')}`: model `{m.get('model')}`, effort `{m.get('effort')}`, "
            f"context {'given' if m.get('with_context') else 'withheld'}, prompt {m.get('prompt_version')}, "
            f"API cost so far ${cost:.2f}"
        )
    return lines


def _table(summary: pd.DataFrame) -> str:
    head = ("| System | RCRPS (lower is better) | 95% interval | Valid outputs | Better than naive | Instances (tasks) |\n"
            "|---|---|---|---|---|---|")
    rows = []
    for r in summary.itertuples():
        better = "" if pd.isna(r.better_than_naive) else f"{r.better_than_naive:.0%}"
        rows.append(f"| {r.system} | {r.rcrps:.3f} | {r.ci_low:.3f} to {r.ci_high:.3f} | {r.valid_rate:.0%} "
                    f"| {better} | {r.n_instances} ({r.n_tasks}) |")
    return "\n".join([head, *rows])


def _group_table(groups: pd.DataFrame) -> str:
    head = "| Scenario group | " + " | ".join(groups.columns) + " |\n|---|" + "---|" * len(groups.columns)
    rows = [f"| {g} | " + " | ".join(f"{v:.3f}" for v in groups.loc[g]) + " |" for g in groups.index]
    return "\n".join([head, *rows])


def write_report(runs_dir: Path, out_dir: Path, instances, test_ids: set[str] | None, test_label: str,
                 example_id: str | None = None, highlight: str | None = None) -> Path:
    from . import charts

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = score_runs(runs_dir, instances)
    if highlight is None:
        highlight = next((n for n in scored if "finetuned" in n or "fine-tuned" in n), None)

    summary_all = summarize(scored)
    summary_all.to_csv(out_dir / "summary_all.csv", index=False)
    pd.concat([df.assign(system=n) for n, df in scored.items()]).to_csv(out_dir / "per_instance.csv", index=False)
    groups = by_group(scored)
    groups.to_csv(out_dir / "by_group.csv")

    md = [f"# Results: Context is Key forecasting\n",
          f"Generated {date.today().isoformat()} by `python -m tslm_eval report`. "
          "Metric: RCRPS, the benchmark's own score (scaled error, half-weighted on the region the text is about, "
          "plus a penalty for breaking stated constraints; capped at 5, which is also the score of a failed output). "
          "Weighted mean over instances with CiK's task weights; interval from resampling whole tasks. "
          "All systems give point forecasts, so scores compare with each other, not with the published "
          "CiK leaderboard (which samples 25+ forecasts).\n"]

    if test_ids:
        summary_test = summarize(scored, test_ids)
        summary_test.to_csv(out_dir / "summary_test.csv", index=False)
        md += [f"## Held-out test set: {test_label}\n", _table(summary_test), ""]
        charts.bar_chart(summary_test, out_dir / "chart_test.png", highlight=highlight,
                         subtitle=f"Held-out test set: {test_label}")
    md += ["## All 355 instances\n",
           "Zero-shot systems (baselines, Claude, pretrained OpenTSLM) never trained on CiK, so all 355 are "
           "unseen for them. A model fine-tuned on CiK must be read on the held-out set only.\n",
           _table(summary_all), ""]
    charts.bar_chart(summary_all, out_dir / "chart_all.png", highlight=highlight,
                     subtitle="All 355 instances (71 tasks)")

    md += ["## By scenario group (all instances)\n", _group_table(groups), ""]
    meta = _meta_lines(runs_dir)
    if meta:
        md += ["## Run settings\n", *meta, ""]

    if example_id:
        inst = next(i for i in instances if i.id == example_id)
        shown = [n for n in [highlight, "claude-opus-5-5", "seasonal-naive", "naive"] if n and n in scored][:3]
        forecasts = {n: read_predictions(Path(runs_dir) / f"{n}.jsonl", instances)[inst.id].forecast
                     for n in shown}
        charts.example_chart(inst, forecasts, out_dir / "chart_example.png")
        md += ["![Example forecast](chart_example.png)\n"]

    path = out_dir / "results.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path
