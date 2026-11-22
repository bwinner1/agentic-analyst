"""OpenTSLM dataset wrapper for Context is Key.

Copy this file and cik_core.py into src/opentslm/time_series_datasets/cik/ of the OpenTSLM
repo (see README.md). It follows the same QADataset pattern as M4QADataset.

Configure with class attributes before first use (QADataset caches per class):
    CiKQADataset.source = "auto"     # "timenet", "hf" or "auto" (TimeNet first, Hugging Face if that fails)
    CiKQADataset.timenet_ref = "servicenow/context-is-key@1.0.0"
    CiKQADataset.timenet_registry = None   # default ~/.timenet/registry
    CiKQADataset.parquet_path = "data/cik.parquet"          # for "hf": local file or the hf:// default
    CiKQADataset.rationales_path = ".../cik/cik_rationales.jsonl"  # explanations; None = forecasts only
    CiKQADataset.include_context = False   # the ablation: same data, text context removed
QADataset formats the data once per class and caches it, so use one configuration per process
(train_cik.py and eval_cik.py set these from their command-line flags).
"""

from typing import List, Literal, Tuple

from opentslm.prompt.prompt_with_answer import PromptWithAnswer
from opentslm.prompt.text_prompt import TextPrompt
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset

from . import cik_core


class CiKQADataset(QADataset):
    source: str = "auto"
    timenet_ref: str = cik_core.TIMENET_REF
    timenet_registry: str | None = None
    parquet_path: str = cik_core.HF_PARQUET
    rationales_path: str | None = None
    include_context: bool = True
    max_history: int = 512
    show_baseline: bool = True  # put the seasonal-naive forecast in the prompt as a starting point
    max_samples: int | None = None  # set for a quick smoke test
    extra_train_path: str | None = None  # extra generated training rows (cik_extra.py), added to train only
    extra_rationales_path: str | None = None

    def __init__(self, split: Literal["train", "test", "validation"], EOS_TOKEN: str, **kwargs):
        self._rationales = cik_core.load_rationales(self.rationales_path)
        self._rationales.update(cik_core.load_rationales(self.extra_rationales_path))
        super().__init__(split, EOS_TOKEN, **kwargs)

    def _load_splits(self) -> Tuple[list, list, list]:
        rows = self._read_rows()
        train, val, test = cik_core.split_by_task(rows)
        if self.extra_train_path:
            heldout = {r["name"] for r in val + test}
            extra = [r for r in cik_core.load_rows(self.extra_train_path)
                     if r["name"] not in heldout and (r["name"], int(r["seed"])) in self._rationales]
            print(f"Adding {len(extra)} generated training examples from {self.extra_train_path}")
            train = train + extra
        if self.max_samples:
            train, val, test = train[: self.max_samples], val[: self.max_samples], test[: self.max_samples]
        return train, val, test

    def _read_rows(self) -> list:
        if self.source in ("timenet", "auto"):
            try:
                rows = cik_core.load_rows_from_timenet(self.timenet_ref, self.timenet_registry)
                print(f"Loaded {len(rows)} CiK instances from TimeNet ({self.timenet_ref})")
                return rows
            except Exception as exc:  # TimeNet not installed, or the dataset not built yet
                if self.source == "timenet":
                    raise
                print(f"TimeNet load failed ({exc!r}); reading CiK from Hugging Face instead")
        rows = cik_core.load_rows(self.parquet_path)
        print(f"Loaded {len(rows)} CiK instances from {self.parquet_path}")
        return rows

    def _example(self, row) -> cik_core.Example:
        rationale = self._rationales.get((row["name"], int(row["seed"])))
        if self.rationales_path and rationale is None:
            rationale = ""  # keep the answer format the same for rows without a rationale
        return cik_core.build_example(
            row, include_context=self.include_context, rationale=rationale, max_history=self.max_history,
            show_baseline=self.show_baseline
        )

    def _format_sample(self, row):
        ex = self._example(row)
        answer = ex.answer
        if not answer.endswith(self.EOS_TOKEN):
            answer += self.EOS_TOKEN
        sample = PromptWithAnswer(
            TextPrompt(ex.pre_prompt),
            [TextTimeSeriesPrompt(t, s) for t, s in zip(ex.series_texts, ex.series)],
            TextPrompt(ex.post_prompt),
            answer,
        ).to_dict()
        sample["meta"] = ex.meta
        return sample

    # QADataset declares these abstract; _format_sample above does the work.
    def _get_answer(self, row) -> str:
        return self._example(row).answer

    def _get_pre_prompt(self, row) -> str:
        return self._example(row).pre_prompt

    def _get_post_prompt(self, row) -> str:
        return self._example(row).post_prompt

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        ex = self._example(row)
        return [TextTimeSeriesPrompt(t, s) for t, s in zip(ex.series_texts, ex.series)]

