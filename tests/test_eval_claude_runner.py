import json
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")

from eval_helpers import make_instance  # noqa: E402
from tslm_eval import claude_runner, metric  # noqa: E402
from tslm_eval.forecasts import read_predictions  # noqa: E402


class FakeClient:
    """Stands in for anthropic.Anthropic(): records requests, answers with canned text."""

    def __init__(self, reply="<forecast>\n2024-02-01,5\n2024-02-02,6\n2024-02-03,7\n</forecast>",
                 stop_reason="end_turn"):
        self.calls = []
        self.reply, self.stop_reason = reply, stop_reason
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        blocks = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=self.reply)]
        return SimpleNamespace(content=blocks, stop_reason=self.stop_reason, model=kwargs["model"],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=50))


def _quiet(*args, **kwargs):
    pass


def test_prompt_has_history_context_and_target_times_but_never_the_answer():
    inst = make_instance(future=(123.456, 789.012, 345.678))
    prompt = claude_runner.build_prompt(inst)
    assert "The building is closed." in prompt
    assert all(t in prompt for t in inst.future_times)
    assert "2024-01-04 00:00:00,4" in prompt
    assert not any(v in prompt for v in ("123.456", "789.012", "345.678"))


def test_no_context_ablation_withholds_the_text():
    prompt = claude_runner.build_prompt(make_instance(), with_context=False)
    assert "closed" not in prompt and "<context>" not in prompt


def test_run_records_answers_with_fixed_model_and_effort(tmp_path):
    insts = [make_instance(seed=s) for s in (1, 2)]
    client = FakeClient()
    out = tmp_path / "claude.jsonl"
    totals = claude_runner.run(insts, out, client=client, effort="high", workers=2, log=_quiet)
    assert totals["done"] == 2 and totals["failed"] == 0
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "high"}
    assert "temperature" not in call and "thinking" not in call
    preds = read_predictions(out, insts)
    assert all(metric.rcrps(i, preds[i.id].forecast) == 0.0 for i in insts)
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert (meta["model"], meta["effort"], meta["with_context"]) == ("claude-opus-5-5", "high", True)


def test_rerun_skips_instances_already_answered(tmp_path):
    insts = [make_instance(seed=s) for s in (1, 2, 3)]
    out = tmp_path / "claude.jsonl"
    claude_runner.run(insts[:2], out, client=FakeClient(), log=_quiet)
    client = FakeClient()
    claude_runner.run(insts, out, client=client, log=_quiet)
    assert len(client.calls) == 1
    assert len(out.read_text().splitlines()) == 3


def test_refusal_is_kept_and_scored_as_a_failed_forecast(tmp_path):
    inst = make_instance()
    out = tmp_path / "claude.jsonl"
    claude_runner.run([inst], out, client=FakeClient(reply="", stop_reason="refusal"), log=_quiet)
    rec = json.loads(out.read_text())
    assert rec["stop_reason"] == "refusal"
    assert metric.rcrps(inst, read_predictions(out, [inst])[inst.id].forecast) == metric.CAP
