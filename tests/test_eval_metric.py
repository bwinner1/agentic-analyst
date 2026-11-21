import itertools

import pytest

np = pytest.importorskip("numpy")

from eval_helpers import make_instance  # noqa: E402
from tslm_eval import metric  # noqa: E402


# --- reference: the official CiK code, transcribed (crps.py and compute_rcrps_with_hf_dataset.py) ---

def _official_crps(target, samples):
    num_samples = samples.shape[0]
    num_dims = samples.ndim
    sorted_samples = np.sort(samples, axis=0)
    abs_diff = np.abs(np.expand_dims(target, axis=0) - sorted_samples).sum(axis=0) / num_samples
    beta0 = sorted_samples.sum(axis=0) / num_samples
    i_array = np.expand_dims(np.arange(num_samples), axis=tuple(range(1, num_dims)))
    beta1 = (i_array * sorted_samples).sum(axis=0) / (num_samples * (num_samples - 1))
    return abs_diff + beta0 - 2 * beta1


def _official_violation(inst, samples, scaling):
    violation = 0.0
    scaled_samples = scaling * samples
    violation += (scaling * inst.constraint_min - scaled_samples).clip(min=0).mean(axis=1)
    violation += (scaled_samples - scaling * inst.constraint_max).clip(min=0).mean(axis=1)
    if len(inst.var_max_index) > 0:
        indexed = scaled_samples[:, inst.var_max_index]
        thresholds = scaling * np.array(inst.var_max_values)
        violation += (indexed - thresholds[None, :]).clip(min=0).mean(axis=1)
    return violation


def _official_rcrps(inst, samples):
    target = inst.future
    if len(inst.roi):
        mask = np.zeros(len(target), dtype=bool)
        mask[inst.roi] = True
        roi_crps = _official_crps(target[mask], samples[:, mask]).mean()
        non_roi_crps = _official_crps(target[~mask], samples[:, ~mask]).mean()
        crps_value = 0.5 * roi_crps + 0.5 * non_roi_crps
    else:
        crps_value = _official_crps(target, samples).mean()
    violation_func = 10.0 * _official_violation(inst, samples, inst.metric_scaling)
    violation_crps = _official_crps(np.zeros(1), violation_func[:, None])[0]
    value = inst.metric_scaling * crps_value + violation_crps
    return 5.0 if value >= 5.0 else value


# --- tests ---

def test_point_forecast_crps_is_absolute_error():
    target = np.array([1.0, 2.0, 3.0])
    assert np.allclose(metric.crps(target, np.array([[2.0, 2.0, 0.0]])), [1.0, 0.0, 3.0])


def test_sample_crps_matches_the_pairwise_definition():
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(7, 4))
    target = rng.normal(size=4)
    n = samples.shape[0]
    for t in range(4):
        x, y = samples[:, t], target[t]
        pairs = sum(abs(a - b) for a, b in itertools.permutations(x, 2))
        expected = np.abs(x - y).mean() - pairs / (2 * n * (n - 1))
        assert metric.crps(target, samples)[t] == pytest.approx(expected)


@pytest.mark.parametrize("case", [
    dict(),
    dict(roi=(1, 2)),
    dict(cmin=4.0),
    dict(cmax=6.5, roi=(0,)),
    dict(var_idx=(0, 2), var_vals=(5.5, 6.0), cmin=0.0),
])
def test_rcrps_matches_the_official_code_on_sampled_forecasts(case):
    inst = make_instance(future=(5.0, 6.0, 7.0, 8.0), scaling=0.2, **case)
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=6.5, scale=2.0, size=(25, 4))
    assert metric.rcrps(inst, samples) == pytest.approx(_official_rcrps(inst, samples))


def test_perfect_forecast_scores_zero():
    inst = make_instance(roi=(1,), cmin=0.0)
    assert metric.rcrps(inst, inst.future.copy()) == 0.0


def test_region_of_interest_gets_half_the_weight():
    inst = make_instance(future=(0.0, 0.0, 0.0, 0.0), roi=(3,), scaling=1.0)
    # error 4 only inside the one-step region: 0.5 * 4 + 0.5 * 0
    assert metric.rcrps(inst, np.array([0.0, 0.0, 0.0, 4.0])) == pytest.approx(2.0)


def test_breaking_a_constraint_is_penalised_tenfold():
    inst = make_instance(future=(1.0, 1.0), cmin=0.0, scaling=0.1)
    # error: 2 at one of two steps -> mean 1, scaled 0.1 * 1 = 0.1
    # violation: 1 below the minimum at one of two steps -> scaled 0.1 * 1 / 2 = 0.05, times 10 = 0.5
    assert metric.rcrps(inst, np.array([-1.0, 1.0])) == pytest.approx(0.1 + 0.5)


@pytest.mark.parametrize("bad", [None, np.array([1.0, 2.0]), np.array([np.nan, 1.0, 2.0]), np.array([1e9, 0, 0])])
def test_failed_or_wild_forecasts_score_the_cap(bad):
    assert metric.rcrps(make_instance(), bad) == metric.CAP


def test_aggregate_is_the_task_weighted_mean():
    agg = metric.aggregate(["a", "a", "b"], [1.0, 3.0, 4.0], [0.5, 0.5, 1.0])
    assert agg.mean == pytest.approx((0.5 + 1.5 + 4.0) / 2.0)
    assert agg.ci_low <= agg.mean <= agg.ci_high
    assert (agg.n_instances, agg.n_tasks) == (3, 2)


def test_aggregate_interval_collapses_when_every_score_is_equal():
    agg = metric.aggregate(["a", "b", "c"], [0.3, 0.3, 0.3], [1.0, 1.0, 1.0])
    assert agg.ci_low == pytest.approx(0.3) and agg.ci_high == pytest.approx(0.3)
