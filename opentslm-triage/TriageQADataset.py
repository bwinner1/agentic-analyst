"""OpenTSLM QADataset for the existing KPI triage prompt contract."""

from __future__ import annotations

from typing import Any, Literal

from common import add_repo_src, read_rows

add_repo_src()

# Keep this module inspectable on machines without the OpenTSLM installation.
try:
    from opentslm.time_series_datasets.QADataset import QADataset
except ModuleNotFoundError as exc:
    _IMPORT_ERROR: Exception | None = exc
    QADataset = object
else:
    _IMPORT_ERROR = None


class TriageQADataset(QADataset):
    """Configure data_dir and max_samples before constructing any split."""

    data_dir: str = "data/triage"
    max_samples: int | None = None
    with_values: bool = False  # also write the series values as text (kpi_triage.prompting)
    with_marks: bool = False  # add the two marker series (kpi_triage.prompting.marks)
    _configuration: tuple[str, int | None, str, bool, bool] | None = None

    def __init__(self, split: Literal["train", "test", "validation"], EOS_TOKEN: str, **kwargs: Any):
        if _IMPORT_ERROR is not None:
            raise RuntimeError("TriageQADataset requires the existing OpenTSLM environment") from _IMPORT_ERROR
        configuration = (str(self.data_dir), self.max_samples, EOS_TOKEN, self.with_values, self.with_marks)
        cls = type(self)
        if cls._configuration != configuration:
            # QADataset caches all three formatted splits on the subclass.
            for name in ("loaded", "_train_dataset", "_validation_dataset", "_test_dataset"):
                if name in cls.__dict__:
                    delattr(cls, name)
            cls._configuration = configuration
        super().__init__(split, EOS_TOKEN, **kwargs)

    def _load_splits(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        return tuple(read_rows(self.data_dir, split, self.max_samples) for split in ("train", "val", "test"))

    def _example(self, row: dict[str, Any]) -> Any:
        from kpi_triage.prompting import opentslm_example

        return opentslm_example(row, with_values=self.with_values, with_marks=self.with_marks)

    def _format_sample(self, row: dict[str, Any]) -> dict[str, Any]:
        from opentslm.prompt.prompt_with_answer import PromptWithAnswer
        from opentslm.prompt.text_prompt import TextPrompt
        from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt

        ex = self._example(row)
        answer = ex.answer if ex.answer.endswith(self.EOS_TOKEN) else ex.answer + self.EOS_TOKEN
        sample = PromptWithAnswer(
            TextPrompt(ex.pre_prompt),
            [TextTimeSeriesPrompt(text, series) for text, series in zip(ex.series_texts, ex.series, strict=True)],
            TextPrompt(ex.post_prompt),
            answer,
        ).to_dict()
        sample["meta"] = ex.meta
        return sample

    def _get_answer(self, row: dict[str, Any]) -> str:
        return self._example(row).answer

    def _get_pre_prompt(self, row: dict[str, Any]) -> str:
        return self._example(row).pre_prompt

    def _get_post_prompt(self, row: dict[str, Any]) -> str:
        return self._example(row).post_prompt

    def _get_text_time_series_prompt_list(self, row: dict[str, Any]) -> list[Any]:
        from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt

        ex = self._example(row)
        return [TextTimeSeriesPrompt(text, series) for text, series in zip(ex.series_texts, ex.series, strict=True)]
