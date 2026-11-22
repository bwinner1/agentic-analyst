"""Small synthetic CiK instances for the evaluation tests (no download needed)."""

import numpy as np

from tslm_eval.cik import Instance


def make_instance(task="ATMBuildingClosedTask", seed=1, past=(1.0, 2.0, 3.0, 4.0), future=(5.0, 6.0, 7.0),
                  roi=(), cmin=-np.inf, cmax=np.inf, var_idx=(), var_vals=(), scaling=1.0, period=-1,
                  weight=1.0, background="Daily ATM withdrawals.", scenario="The building is closed.",
                  constraints=""):
    past = np.asarray(past, dtype=float)
    future = np.asarray(future, dtype=float)
    return Instance(
        task=task, seed=seed, weight=weight, background=background, scenario=scenario,
        constraints=constraints,
        past_times=tuple(f"2024-01-{d + 1:02d} 00:00:00" for d in range(len(past))),
        past=past,
        future_times=tuple(f"2024-02-{d + 1:02d} 00:00:00" for d in range(len(future))),
        future=future, seasonal_period=period, metric_scaling=scaling,
        roi=np.asarray(roi, dtype=int), constraint_min=cmin, constraint_max=cmax,
        var_max_index=np.asarray(var_idx, dtype=int), var_max_values=np.asarray(var_vals, dtype=float),
        context_sources=("c_i",),
    )
