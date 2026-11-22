"""Score every prediction file the same way and write the results table and charts."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from . import metric
from .forecasts import read_predictions
from .splits import scenario_group

BASELINES = ("seasonal-naive", "naive")  # the first one present is what "Beats ..." compares with
CLAUDE = "claude-opus-5-5"


def reference(scored) -> str | None:
    return next((n for n in BASELINES if n in scored), None)


def full_coverage(scored: dict[str, pd.DataFrame]) -> list[str]:
    """Systems with a prediction for every instance (the others were run on the test set only)."""
    return [n for n, df in scored.items() if not (df.problem == "missing").any()]


def default_highlight(scored) -> str | None:
    """Our main model: the shortest system name containing 'finetuned' (variants add suffixes)."""
    ours = [n for n in scored if "finetuned" in n or "fine-tuned" in n]
    return min(ours, key=len) if ours else None


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
    ref_name = reference(scored)
    ref = scored.get(ref_name) if ref_name else None
    for name, df in scored.items():
        d = df if ids is None else df[df.id.isin(ids)]
        agg = metric.aggregate(d.task.tolist(), d.rcrps.tolist(), d.weight.tolist())
        row = {"system": name, "rcrps": agg.mean, "ci_low": agg.ci_low, "ci_high": agg.ci_high,
               "valid_rate": float(d.valid.mean()), "n_instances": agg.n_instances, "n_tasks": agg.n_tasks,
               "better_than_baseline": float("nan")}
        if ref is not None and name != ref_name:
            ref_scores = ref.set_index("id").rcrps.reindex(d.id).to_numpy()
            row["better_than_baseline"] = float((d.rcrps.to_numpy() < ref_scores).mean())
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["system", "rcrps", "ci_low", "ci_high", "valid_rate", "n_instances",
                                     "n_tasks", "better_than_baseline"])
    return pd.DataFrame(rows).sort_values("rcrps").reset_index(drop=True)


def by_group(scored: dict[str, pd.DataFrame], ids: set[str] | None = None) -> pd.DataFrame:
    """Weighted mean RCRPS per scenario group (rows) and system (columns)."""
    cols = {}
    for name, df in scored.items():
        if ids is not None:
            df = df[df.id.isin(ids)]
        g = df.assign(ws=df.weight * df.rcrps).groupby("group")
        cols[name] = g.ws.sum() / g.weight.sum()
    return pd.DataFrame(cols)


def paired(scored: dict[str, pd.DataFrame], ours: str, other: str, ids: set[str] | None = None) -> dict:
    """`ours` minus `other` per instance (negative = ours is better), with a task-bootstrap interval."""
    a, b = scored[ours].set_index("id"), scored[other].set_index("id")
    common = a.index.intersection(b.index)
    if ids is not None:
        common = common[common.isin(list(ids))]
    diff = a.rcrps[common] - b.rcrps[common]
    agg = metric.aggregate(a.task[common].tolist(), diff.tolist(), a.weight[common].tolist())
    return {"other": other, "diff": agg.mean, "ci_low": agg.ci_low, "ci_high": agg.ci_high,
            "lower": float((diff < -1e-9).mean()), "higher": float((diff > 1e-9).mean()), "n": len(diff)}


def _paired_table(rows: list[dict], ours: str) -> str:
    head = (f"| {ours} compared with | Difference in RCRPS (ours minus other) | 95% interval "
            "| Ours lower (better) | Ours higher |\n|---|---|---|---|---|")
    body = [f"| {r['other']} | {r['diff']:+.3f} | {r['ci_low']:+.3f} to {r['ci_high']:+.3f} "
            f"| {r['lower']:.0%} | {r['higher']:.0%} |" for r in rows]
    return "\n".join([head, *body])


def _meta_lines(runs_dir: Path) -> list[str]:
    lines = []
    for meta_path in sorted(Path(runs_dir).glob("*.meta.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        name = meta_path.name.removesuffix(".meta.json")
        if m.get("system_type") == "opentslm":
            training = ""
            if m.get("best_epoch") and m.get("epochs_run", 0) > 1:
                training = (f"; best of {m['epochs_run']} epochs was epoch {m['best_epoch']} "
                            f"(validation loss {m.get('best_val_loss'):.3f})")
            elif m.get("best_epoch"):
                training = f"; trained 1 epoch (validation loss {m.get('best_val_loss'):.3f})"
            if m.get("extra_train"):
                training += ", with extra generated training examples"
            tuned = (f"fine-tuned (checkpoint `{m.get('checkpoint')}`)"
                     if m.get("checkpoint") not in (None, "none") else "without fine-tuning")
            lines.append(
                f"- `{name}`: OpenTSLM `{m.get('model')}` {tuned}, "
                f"context {'given' if m.get('with_context') else 'withheld in training and test'}, "
                f"median of {m.get('num_samples')} samples at temperature {m.get('temperature')}{training}. "
                f"Imported from `{m.get('source')}`."
            )
            continue
        cost = sum(r.get("cost_usd", 0) for r in m.get("runs", []))
        lines.append(
            f"- `{name}`: model `{m.get('model')}`, effort `{m.get('effort')}`, "
            f"context {'given' if m.get('with_context') else 'withheld'}, prompt {m.get('prompt_version')}, "
            f"API cost so far ${cost:.2f}"
        )
    return lines


def _table(summary: pd.DataFrame, ref_name: str | None) -> str:
    beats = f"Beats {ref_name} on" if ref_name else "Beats baseline on"
    head = (f"| System | RCRPS (lower is better) | 95% interval | Valid outputs | {beats} | Instances (tasks) |\n"
            "|---|---|---|---|---|---|")
    rows = []
    for r in summary.itertuples():
        better = "" if pd.isna(r.better_than_baseline) else f"{r.better_than_baseline:.0%}"
        rows.append(f"| {r.system} | {r.rcrps:.3f} | {r.ci_low:.3f} to {r.ci_high:.3f} | {r.valid_rate:.0%} "
                    f"| {better} | {r.n_instances} ({r.n_tasks}) |")
    return "\n".join([head, *rows])


def _group_table(groups: pd.DataFrame) -> str:
    head = "| Scenario group | " + " | ".join(groups.columns) + " |\n|---|" + "---|" * len(groups.columns)
    rows = [f"| {g} | " + " | ".join(f"{v:.3f}" for v in groups.loc[g]) + " |" for g in groups.index]
    return "\n".join([head, *rows])


def write_report(runs_dir: Path, out_dir: Path, instances, test_ids: set[str] | None, test_label: str,
                 examples: list[str] | None = None, highlight: str | None = None) -> Path:
    from . import charts

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = score_runs(runs_dir, instances)
    if highlight is None:
        highlight = default_highlight(scored)
    ref_name = reference(scored)
    complete = full_coverage(scored)
    partial = [n for n in scored if n not in complete]

    summary_all = summarize({n: scored[n] for n in complete})
    summary_all.to_csv(out_dir / "summary_all.csv", index=False)
    # A system run on the test set only gets rows for the test set, not "missing" rows for the rest.
    rows = [df.assign(system=n) if n in complete else df[df.id.isin(test_ids or set())].assign(system=n)
            for n, df in scored.items()]
    pd.concat(rows).to_csv(out_dir / "per_instance.csv", index=False)
    groups = by_group(scored, test_ids) if test_ids else by_group({n: scored[n] for n in complete})
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
        md += [f"## Held-out test set: {test_label}\n", _table(summary_test, ref_name), ""]
        charts.bar_chart(summary_test, out_dir / "chart_test.png", highlight=highlight,
                         subtitle=f"Held-out test set: {test_label}")
        if highlight in scored:
            others = [n for n in summary_test.system if n != highlight]
            rows = [paired(scored, highlight, o, test_ids) for o in others]
            pd.DataFrame(rows).to_csv(out_dir / "paired_test.csv", index=False)
            md += ["### Our model against each system, instance by instance\n",
                   "Both systems are scored on the same instances, so this is a sharper test than comparing "
                   "the two means above. A negative difference means ours is better. If the interval includes "
                   "0, this test set cannot tell the two apart.\n",
                   _paired_table(rows, highlight), ""]
    md += ["## All 355 instances\n"]
    if summary_all.empty:
        md += ["No system has predictions for every instance yet.\n"]
    else:
        md += ["Zero-shot systems (baselines, Claude, pretrained OpenTSLM) never trained on CiK, so all 355 are "
               "unseen for them. A model fine-tuned on CiK must be read on the held-out set only.\n",
               _table(summary_all, ref_name), ""]
        charts.bar_chart(summary_all, out_dir / "chart_all.png", highlight=highlight,
                         subtitle="All 355 instances (71 tasks)")
    if partial:
        md += [f"Not in this table, because they were run on the test set only: {', '.join(partial)}.\n"]

    where = "held-out test set" if test_ids else "all instances"
    md += [f"## By scenario group ({where})\n", _group_table(groups), ""]
    meta = _meta_lines(runs_dir)
    if meta:
        md += ["## Run settings\n", *meta, ""]

    if examples:
        md += ["## Examples\n",
               "Chosen by hand (`--example`) to illustrate. The tables above are the result. "
               "The number after each system is its RCRPS on that instance.\n"]
        candidates = [highlight, f"{highlight}-no-context", CLAUDE, *BASELINES]
        shown = list(dict.fromkeys(n for n in candidates if n and n in scored))[:3]
        preds = {n: read_predictions(Path(runs_dir) / f"{n}.jsonl", instances) for n in shown}
        by_id = {i.id: i for i in instances}
        for k, example_id in enumerate(examples):
            if example_id not in by_id:
                raise ValueError(f"example {example_id} is not a CiK instance id")
            inst = by_id[example_id]
            name = "chart_example.png" if k == 0 else f"chart_example_{k + 1}.png"
            forecasts = {n: preds[n][inst.id].forecast for n in shown}
            scores = {n: float(scored[n].set_index("id").rcrps[inst.id])
                      for n in shown if forecasts[n] is not None}
            charts.example_chart(inst, forecasts, out_dir / name, scores=scores)
            md += [f"![Example: {inst.id}]({name})\n"]

    path = out_dir / "results.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path
