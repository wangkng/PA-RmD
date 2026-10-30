"""Paper methods; import only the selected implementation."""
from pa_rmd.config import Config, method_settings

METHODS = ("pa-rmd", "node", "sindy", "leads", "coda-l1", "coda-l2", "siar")


def make_model(method: str, cfg: Config, experiment: str, **overrides):
    """Select a method with the published settings for this experiment."""
    settings = method_settings(method, experiment, **overrides)
    if method == "pa-rmd":
        from pa_rmd.models.pa_rmd import PARmD
        return PARmD(cfg, **settings)
    if method == "node":
        from pa_rmd.models.node import NodePredictor
        return NodePredictor(cfg, **settings)
    if method == "sindy":
        from pa_rmd.models.sindy import SindyPredictor
        return SindyPredictor(cfg, **settings)
    if method == "leads":
        from pa_rmd.models.leads import LeadsZeroShotPredictor
        return LeadsZeroShotPredictor(cfg, **settings)
    if method in {"coda-l1", "coda-l2"}:
        from pa_rmd.models.coda import CodaPredictor
        return CodaPredictor(cfg, **settings)
    if method == "siar":
        from pa_rmd.models.siar import SIARPredictor
        return SIARPredictor(cfg, **settings)
    raise ValueError(f"Unknown method: {method}; choose from {METHODS}")
