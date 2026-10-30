from __future__ import annotations

from typing import Any

from pa_rmd.config import Config
from pa_rmd.evaluate import TestBlock, TrainBatch


def load_dataset(name: str, cfg: Config, **kwargs: Any) -> tuple[TrainBatch, list[TestBlock], dict]:
    if name == "synthetic-payoff-family":
        from pa_rmd.datasets import synthetic_payoff

        return synthetic_payoff.load(cfg, **kwargs)
    if name == "youtube-multicountry":
        from pa_rmd.datasets import youtube

        return youtube.load(cfg, **kwargs)
    raise ValueError(f"Unknown dataset: {name}")
