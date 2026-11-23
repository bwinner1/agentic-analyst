"""CPU contract checks; run with python -B opentslm-triage/test_pipeline.py."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from common import BASE, add_repo_src, read_rows, write_json
from text_baseline import collate_text, prompt_ids, training_sample

add_repo_src()
from kpi_triage.labels import parse_answer
from kpi_triage.prompting import opentslm_example, text_prompt


@contextlib.contextmanager
def patched_modules(modules: dict):
    """Replace only test doubles without unloading newly imported torch modules."""
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def row(identifier: str = "sample") -> dict:
    return {
        "id": identifier, "name": "Revenue", "unit": "CHF", "freq": "daily",
        "timestamps": [f"t{i}" for i in range(120)],
        "values": [100.0 + i % 9 for i in range(116)] + [150.0] * 4,
        "window": 4, "note": "The campaign started recently.",
        "label": {"verdict": "explained_event", "shape": "level_up", "start": 0, "end": 3},
        "reason": "The campaign explains the sustained increase.",
    }


class CharacterTokenizer:
    """Lossless tokens make deletion boundaries observable without model downloads."""

    eos_token_id = 2
    pad_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return ([1] if add_special_tokens else []) + [ord(char) + 10 for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(token - 10) for token in ids if token >= 10)


class ContractTests(unittest.TestCase):
    def test_truncation_keeps_review_question_scale_and_newest_history(self) -> None:
        source = row()
        tokenizer = CharacterTokenizer()
        original, answer = text_prompt(source)
        ids, got_answer, dropped = prompt_ids(source, tokenizer, len(original) - 200)
        shortened = tokenizer.decode(ids)
        self.assertLessEqual(len(ids), len(original) - 200)
        self.assertGreater(dropped, 0)
        self.assertEqual(got_answer, answer)
        marker = "\nHistory before the review window: "
        self.assertEqual(shortened.partition(marker)[0], original.partition(marker)[0])
        self.assertEqual(shortened.partition("\nReview window:")[2], original.partition("\nReview window:")[2])
        history = original.partition(marker)[2].partition("\nReview window:")[0].split(", ")
        kept = shortened.partition(marker)[2].partition("\nReview window:")[0]
        self.assertEqual(kept, ", ".join(history[dropped:]))
        self.assertEqual(source, row())

    def test_untruncated_prompt_is_identical_and_small_budget_fails(self) -> None:
        tokenizer = CharacterTokenizer()
        prompt, _ = text_prompt(row())
        ids, _, dropped = prompt_ids(row(), tokenizer, 10000)
        self.assertEqual(ids, tokenizer.encode(prompt))
        self.assertEqual(dropped, 0)
        with self.assertRaisesRegex(ValueError, "review window and question exceed"):
            prompt_ids(row(), tokenizer, 50)

    def test_answer_and_padding_masks(self) -> None:
        tokenizer = CharacterTokenizer()
        sample, _ = training_sample(row(), tokenizer, 10000)
        prompt, answer = text_prompt(row())
        boundary = len(tokenizer.encode(prompt))
        self.assertEqual(sample["labels"][:boundary], [-100] * boundary)
        self.assertEqual(sample["labels"][boundary:], tokenizer.encode(answer, False) + [2])
        shorter = {"input_ids": [1, 12, 2], "labels": [-100, 12, 2]}
        batch = collate_text([sample, shorter], 2)
        self.assertEqual(batch["labels"][1, 3:].tolist(), [-100] * (len(sample["input_ids"]) - 3))
        self.assertEqual(batch["attention_mask"][1].sum().item(), 3)
        # Real EOS tokens are supervised even when EOS doubles as padding.
        self.assertEqual(batch["labels"][1, 2].item(), 2)

    def test_split_alias_limit_and_dataset_format(self) -> None:
        from TriageQADataset import TriageQADataset

        class TextPrompt:
            def __init__(self, text: str):
                self.text = text

        class SeriesPrompt:
            def __init__(self, text: str, values: list[float]):
                self.text, self.values = text, values

        class AnswerPrompt:
            def __init__(self, pre: TextPrompt, series: list[SeriesPrompt], post: TextPrompt, answer: str):
                self.sample = {"pre_prompt": pre.text, "post_prompt": post.text, "answer": answer,
                               "time_series_text": [item.text for item in series],
                               "time_series": [item.values for item in series]}

            def to_dict(self) -> dict:
                return self.sample

        modules = {}
        for name, cls in (("text_prompt", TextPrompt), ("text_time_series_prompt", SeriesPrompt),
                          ("prompt_with_answer", AnswerPrompt)):
            module = types.ModuleType(f"opentslm.prompt.{name}")
            setattr(module, {"text_prompt": "TextPrompt", "text_time_series_prompt": "TextTimeSeriesPrompt",
                             "prompt_with_answer": "PromptWithAnswer"}[name], cls)
            modules[module.__name__] = module
        dataset = object.__new__(TriageQADataset)
        dataset.EOS_TOKEN = "<eos>"
        with patched_modules(modules):
            sample = dataset._format_sample(row())
        expected = opentslm_example(row())
        self.assertEqual(sample["time_series"], expected.series)
        self.assertEqual(len(sample["time_series"]), 2)
        self.assertEqual(sample["pre_prompt"], expected.pre_prompt)
        self.assertEqual(sample["post_prompt"], expected.post_prompt)
        self.assertEqual(sample["answer"], expected.answer + "<eos>")
        self.assertEqual(sample["meta"], expected.meta)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp) / "val.jsonl"
            path.write_text("\n" + json.dumps(row("a")) + "\n" + json.dumps(row("b")) + "\n", encoding="utf-8")
            self.assertEqual([item["id"] for item in read_rows(temp, "validation", 1)], ["a"])

    def test_eval_continues_after_failure_and_reloads_training_settings(self) -> None:
        import eval_triage

        class Dataset:
            def __init__(self, split: str, EOS_TOKEN: str):
                self.rows = [row("a"), row("b"), row("c")]

            def __getitem__(self, index: int) -> dict:
                return self.rows[index]

        class Model:
            def get_eos_token(self) -> str:
                return "<eos>"

            def load_from_file(self, path: str) -> None:
                self.loaded = path

            def eval(self) -> None:
                pass

            def generate(self, batch: list[dict], **kwargs: object) -> list[str]:
                if batch[0]["id"] == "b":
                    raise RuntimeError("deliberate generation failure")
                return [opentslm_example(batch[0]).answer]

        module = types.ModuleType("TriageQADataset")
        module.TriageQADataset = Dataset
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            directory = Path(temp)
            (directory / "test.jsonl").write_text("".join(json.dumps(row(i)) + "\n" for i in ("a", "b", "c")), encoding="utf-8")
            write_json(directory / "train_log.json", {"args": {"base": "custom-sp", "lora_r": 8}})
            with patched_modules({"TriageQADataset": module}), \
                    patch.object(eval_triage, "load_model", return_value=Model()) as load, \
                    patch.object(eval_triage, "collate", side_effect=lambda batch: batch), \
                    contextlib.redirect_stdout(io.StringIO()):
                eval_triage.main(["--data_dir", temp, "--out", temp, "--ckpt", str(directory / "best_model.pt"), "--device", "cpu"])
            load.assert_called_once_with("custom-sp", "cpu", 8)
            predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([item["id"] for item in predictions], ["a", "b", "c"])
            self.assertEqual(predictions[1]["text"], "")
            self.assertIsNotNone(parse_answer(predictions[2]["text"], 4)[0])
            settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
            self.assertEqual(settings["max_new_tokens"], 96)
            self.assertEqual(settings["base"], "custom-sp")
            self.assertEqual(settings["lora_r"], 8)
            with patched_modules({"TriageQADataset": module}), \
                    patch.object(eval_triage, "load_model", return_value=Model()) as load, \
                    patch.object(eval_triage, "collate", side_effect=lambda batch: batch), \
                    contextlib.redirect_stdout(io.StringIO()):
                eval_triage.main(["--data_dir", temp, "--out", temp, "--ckpt", "none", "--device", "cpu"])
            load.assert_called_once_with(BASE, "cpu", None)

    def test_training_keeps_best_checkpoint_and_stops_with_patience(self) -> None:
        import torch
        import train_triage

        class Dataset:
            def __init__(self, split: str, EOS_TOKEN: str):
                self.split = split

            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> dict:
                return {"split": self.split}

        class Model(torch.nn.Module):
            def __init__(self, fail_validation: bool = False):
                super().__init__()
                self.encoder = torch.nn.Linear(1, 1)
                self.projector = torch.nn.Linear(1, 1)
                self.adapter = torch.nn.Parameter(torch.ones(1))
                self.validations = 0
                self.saves = 0
                self.fail_validation = fail_validation

            def get_eos_token(self) -> str:
                return "<eos>"

            def get_lora_parameters(self) -> list:
                return [self.adapter]

            def compute_loss(self, batch: list[dict]) -> torch.Tensor:
                if self.training:
                    return sum(param.square().sum() for param in self.parameters())
                if self.fail_validation:
                    raise torch.OutOfMemoryError("deliberate validation OOM")
                self.validations += 1
                return torch.tensor(float(self.validations))

            def store_to_file(self, path: str) -> None:
                self.saves += 1
                Path(path).write_text("checkpoint", encoding="utf-8")

        module = types.ModuleType("TriageQADataset")
        module.TriageQADataset = Dataset
        for fail_validation in (False, True):
            with self.subTest(fail_validation=fail_validation), \
                    tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
                model = Model(fail_validation)
                with patched_modules({"TriageQADataset": module}), \
                        patch.object(train_triage, "load_model", return_value=model), \
                        patch.object(train_triage, "collate", side_effect=lambda batch: batch), \
                        contextlib.redirect_stdout(io.StringIO()):
                    argv = ["--data_dir", temp, "--out", temp, "--epochs", "5", "--patience", "1", "--device", "cpu"]
                    if fail_validation:
                        with self.assertRaisesRegex(RuntimeError, "No validation batches succeeded"):
                            train_triage.main(argv)
                    else:
                        train_triage.main(argv)
                if fail_validation:
                    self.assertFalse((Path(temp) / "best_model.pt").exists())
                else:
                    self.assertEqual(model.saves, 1)
                    log = json.loads((Path(temp) / "train_log.json").read_text(encoding="utf-8"))
                    self.assertEqual(len(log["history"]), 2)
                    self.assertEqual(log["best_val_loss"], 1.0)


if __name__ == "__main__":
    unittest.main()
