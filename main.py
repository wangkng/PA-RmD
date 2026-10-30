"""Run the synthetic and YouTube experiments with the main methods."""
from __future__ import annotations

import argparse
import copy
import json
import math
from itertools import product
import time
from pathlib import Path
from typing import Any

from pa_rmd.config import Config
from pa_rmd.datasets import load_dataset
from pa_rmd.datasets.synthetic_payoff import ensure_synthetic_dataset
from pa_rmd.datasets.build_youtube import ensure_youtube_dataset
from pa_rmd.models import METHODS, make_model
from pa_rmd.paths import PROJECT_ROOT
from pa_rmd.utils import RESULT_SETTING_DIRS

from pa_rmd.experiment_io import (
    artifact_contract,
    fingerprint,
    input_identity,
    build_run_identity,
    completed_shard,
    json_value,
    metric_metadata,
    predictor_identity,
    prepare_run,
    publish_result,
    rebuild_exports,
    run_lock,
)
from pa_rmd.evaluate import train_and_evaluate, evaluate_fitted


def _rows_for_result(
    result: dict[str, Any], meta: dict[str, Any], data_seed: int, model_seed: int,
    *, protocol_metadata: dict, train_env_ids: list[int],
) -> list[dict]:
    rows: list[dict] = []
    common = {
        "predictor": result["predictor"],
        "label": result["label"],
        "dataset": meta.get("dataset"),
        "exp_name": meta.get("exp_name"),
        "data_seed": int(data_seed),
        "model_seed": int(model_seed),
        **protocol_metadata,
    }
    seen = set(train_env_ids)
    for block in result.get("blocks", []):
        env_id = int(block["env_id"])
        group = block.get("group_label", f"env_{env_id}")
        role = "seen" if env_id in seen else "unseen"
        for name, value in block.items():
            if name in {"env_id", "group_label"}:
                continue
            rows.append({**common, "env_id": env_id, "split_role": role,
                         "group_label": group,
                         "metric_name": name, "value": float(value)})
    return sorted(rows, key=lambda r: (r["env_id"], r["group_label"], r["metric_name"]))


def run(
    *,
    dataset_name: str,
    dataset_args: dict[str, Any],
    predictor: Any,
    cfg: Config,
    out_dir: Path | str,
    data_seed: int,
    model_seed: int,
    fit_cache: dict | None = None,
) -> None:
    out_path = Path(out_dir).resolve()
    load_t0 = time.time()
    train, test_blocks, meta = load_dataset(dataset_name, cfg, **dataset_args)
    test_blocks = list(test_blocks)
    print(f"[timing][runner] load_dataset={time.time() - load_t0:.1f}s out_dir={out_path}", flush=True)
    meta = {"dataset": dataset_name, **meta}
    identity = build_run_identity(dataset_name, dataset_args, cfg, train, test_blocks, meta, data_seed, model_seed)
    # Freeze pre-fit input identities. config_summary() after fit contains learned
    # scales/n_env etc.; those are audit outputs, never a resume input signature.
    predictor_inputs = {str(predictor.key): predictor_identity(predictor, identity)}
    contract = artifact_contract(train, test_blocks)
    config_export = json_value({"dataset": dataset_name, **dataset_args, "cfg": vars(cfg),
                               "data_seed": int(data_seed), "model_seed": int(model_seed),
                               "evaluation_protocol": identity["evaluation_protocol"]})
    with run_lock(out_path):
        manifest = prepare_run(out_path, identity, predictor_inputs, config_export)
        # Repair stale/partial exports even when PA-RmD is already complete.
        # Incomplete attempts are retained but never aggregated.
        rebuild_exports(out_path, manifest)
        key = str(predictor.key)
        if completed_shard(out_path, manifest, key, contract) is not None:
            return
        print(f"[runner] start predictor={key}", flush=True)
        cache_key = fingerprint({"train": input_identity(train),
                                 "predictor": predictor_inputs[key]["constructor_input_fingerprint"],
                                 "code": identity["code_signature"]})
        cached = fit_cache.get(cache_key) if fit_cache is not None else None
        if cached is not None:
            predictor = copy.deepcopy(cached["predictor"])
        predictor._input_cache = {
            "constructor_fingerprint": predictor_inputs[key]["constructor_input_fingerprint"],
            "prefit_summary": predictor_inputs[key]["prefit_config"], "postfit_summary": None,
        }
        try:
            if cached is None:
                result = train_and_evaluate(predictor, train, test_blocks, seed=model_seed)
                if fit_cache is not None:
                    fit_cache[cache_key] = {"predictor": copy.deepcopy(predictor), "model_seed": model_seed}
            else:
                result = evaluate_fitted(predictor, train, test_blocks, seed=model_seed)
                result.setdefault("training_details", {})["deterministic_fit_reuse"] = {
                    "source_model_seed": cached["model_seed"],
                    "independent_model_seed_replicate": False,
                }
        finally:
            # A successful fit followed by an evaluation exception must not
            # turn learned state into a new input identity on retry.
            predictor._input_cache["postfit_summary"] = json_value(predictor.config_summary())
        if result.get("predictor") != key:
            raise ValueError(f"Trainer returned a different predictor: {result.get('predictor')}")
        publish_evaluation_result(out_path, manifest, predictor, result, train, test_blocks, meta,
                                  data_seed=data_seed, model_seed=model_seed, contract=contract)


def publish_evaluation_result(
    out_path: Path, manifest: dict, predictor: Any, result: dict, train: Any,
    test_blocks: list[Any], meta: dict, *, data_seed: int, model_seed: int, contract: dict,
) -> None:
    """Publish ordinary evaluator outputs while the caller holds the run lock."""
    key = str(predictor.key)
    identity = manifest["identity"]
    save_t0 = time.time()
    provenance = metric_metadata(manifest, key)
    rows = _rows_for_result(result, meta, data_seed, model_seed,
                            protocol_metadata=provenance, train_env_ids=identity["train_env_ids"])
    postfit_config = json_value(result.get("config") or {})
    history = {
        **provenance, "predictor": key, "label": result["label"],
        "data_seed": int(data_seed), "model_seed": int(model_seed),
        "loss": [float(v) for v in (result.get("history") or [])],
        "postfit_config": postfit_config,
        "parameter_count": json_value(result.get("parameter_count")),
        "loss_components": json_value(
            predictor.loss_components() if hasattr(predictor, "loss_components") else {}
        ),
        "deterministic_fit_reuse": json_value((result.get("training_details") or {}).get("deterministic_fit_reuse")),
        "timings": json_value(result.get("timings") or {}),
        "payoff_closure_audit": {
            "type": "target-kernel" if any(b.payoff_support is not None for b in test_blocks) else "known-game",
            "target_support_transitions": {str(b.env_id): int(b.payoff_support.x_current.shape[0])
                                           for b in test_blocks if b.payoff_support is not None},
        },
    }
    publish_result(out_path, manifest, key, rows, history, result.get("predictions") or {}, contract)
    rebuild_exports(out_path, manifest)
    print(f"[timing][{key}] save_outputs={time.time() - save_t0:.1f}s", flush=True)



def parse_int_list(raw: str | None, default: list[int] | None = None) -> list[int]:
    if raw is None:
        return list(default or [])
    values = []
    for item in raw.split(","):
        if ":" in item:
            start, count = map(int, item.split(":"))
            values.extend(range(start, start + count))
        elif item.strip():
            values.append(int(item))
    if not values or len(values) != len(set(values)):
        raise ValueError("Expected a nonempty list of distinct integers")
    return values


def synthetic_cases(args):
    """The 12 experiment settings, with explicit filters for smaller runs."""
    families = args.families.split(",")
    dynamics = args.dynamics.split(",")
    dimensions = parse_int_list(args.dimensions)
    if set(families) - {"cyclic", "congestion"} or len(families) != len(set(families)):
        raise ValueError("families must select cyclic and/or congestion")
    if set(dynamics) - {"replicator", "projection", "MWU"} or len(dynamics) != len(set(dynamics)):
        raise ValueError("dynamics must select replicator, projection and/or MWU")
    if set(dimensions) - {3, 4}:
        raise ValueError("dimensions must select 3 and/or 4")
    for family, dynamic, dim in product(families, dynamics, dimensions):
        cfg = Config(N=dim, metric_type=dynamic, num_epochs=args.epochs,
                     num_trajectories=args.train_trajectories,
                     num_eval_trajectories=args.eval_trajectories,
                     trajectory_length=args.steps, delta_t=.05,
                     observation_stride=1 if dynamic == "MWU" else 2)
        n_payoff, n_train = (2, 1) if dim == 3 else (6, 2)
        data = {"sampler_name": "cyclic-isomorphism" if family == "cyclic" else "congestion",
                "num_payoff": n_payoff, "num_payoff_train": n_train,
                "payoff_train_indices": list(range(n_train)), "payoff_test_indices": list(range(n_payoff)),
                "payoff_base_family": "zero-sum" if family == "cyclic" else "potential",
                "payoff_dim": dim, "matrix_frob": None,
                "evaluation_protocol": "independent-ic-v1", "num_eval_trajectories": args.eval_trajectories}
        setting = f"{family}-k{dim}/{dynamic}"
        yield setting, cfg, data


def run_synthetic(args):
    data_seeds = parse_int_list(args.data_seeds, [123])
    model_seeds = parse_int_list(args.model_seeds, list(range(42, 47)))
    for setting, cfg, base_data in synthetic_cases(args):
        for data_seed in data_seeds:
            data = {**base_data, "data_seed": data_seed}
            if not args.dry_run:
                ensure_synthetic_dataset(cfg, **data)
            for model_seed in model_seeds:
                output = (PROJECT_ROOT / "outputs/synthetic-payoff-family" / (args.run_tag or "") /
                          setting / f"data_seed={data_seed}" / f"model_seed={model_seed}")
                for method in args.method:
                    predictor = make_model(method, cfg, "synthetic", **model_overrides(args, method))
                    if args.dry_run:
                        print(json.dumps({"experiment": "synthetic", "setting": setting,
                                          "method": method, "data_seed": data_seed, "model_seed": model_seed,
                                          "data": data, "config": json_value(vars(cfg)),
                                          "model": predictor.config_summary(), "output": str(output)}))
                    else:
                        run(dataset_name="synthetic-payoff-family", dataset_args=data,
                            predictor=predictor, cfg=cfg, out_dir=output,
                            data_seed=data_seed, model_seed=model_seed)


def run_youtube(args):
    data_seeds = parse_int_list(args.data_seeds, list(range(123, 133)))
    model_seeds = parse_int_list(args.model_seeds, list(range(42, 47)))
    q_values = parse_int_list(args.q_values)
    if any(q not in range(3, 10) for q in q_values):
        raise ValueError("YouTube q-values must be between 3 and 9")
    data_path = Path(args.data).resolve()
    if not args.dry_run:
        data_path = ensure_youtube_dataset(data_path, raw_root=args.raw_root)
    cfg = Config(N=15, num_epochs=args.epochs)
    for q in q_values:
        for data_seed in data_seeds:
            # Keep the frozen split derivation: k-train, source countries only for fit.
            data = {"npz_path": str(data_path), "target_country": "ALL",
                    "split_designs": "k-train", "k_train_values": [q], "data_seeds": [data_seed],
                    "split_index": 0, "train_frac": .7, "payoff_field": "payoff_log_views",
                    "payoff_scale": 1.0, "payoff_support_transitions": 40,
                    "rollout_start_transition": 40}
            fit_cache = {}
            for model_seed in model_seeds:
                output = (PROJECT_ROOT / "outputs/youtube-multicountry" / (args.run_tag or "") /
                          f"q={q}" / f"data_seed={data_seed}" / f"model_seed={model_seed}")
                for method in args.method:
                    predictor = make_model(method, cfg, "youtube", **model_overrides(args, method))
                    if args.dry_run:
                        print(json.dumps({"experiment": "youtube", "q": q, "method": method,
                                          "data_seed": data_seed, "model_seed": model_seed,
                                          "data": data, "config": json_value(vars(cfg)),
                                          "model": predictor.config_summary(), "output": str(output)}))
                    else:
                        run(dataset_name="youtube-multicountry", dataset_args=data,
                            predictor=predictor, cfg=cfg, out_dir=output,
                            data_seed=data_seed, model_seed=model_seed,
                            fit_cache=fit_cache if method == "siar" else None)


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return result


_SIAR_OPTIONS = {
    "siar_degree": "siar_degree",
    "siar_fi": "siar_enforce_forward_invariance",
    "siar_pc": "siar_enforce_positive_correlation",
    "siar_project_simplex": "siar_project_simplex",
    "siar_objective_scale": "siar_solver_objective_scale",
}


def model_overrides(args, method):
    if method != "siar":
        return {}
    return {key: getattr(args, option) for option, key in _SIAR_OPTIONS.items()
            if getattr(args, option) is not None}


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="experiment", required=True)
    for name in ("synthetic", "youtube"):
        sub = commands.add_parser(name)
        sub.add_argument("--method", nargs="+", choices=METHODS, default=["pa-rmd"])
        sub.add_argument("--data-seeds", help="Comma-separated integers or start:count")
        sub.add_argument("--model-seeds", help="Default: 42,43,44,45,46")
        sub.add_argument("--epochs", type=positive_int, default=8000 if name == "synthetic" else 1000,
                         help="Optimizer updates for neural methods; unused by SIAR/SINDy")
        sub.add_argument("--run-tag", help="Optional subdirectory for a separate run")
        sub.add_argument("--dry-run", action="store_true", help="Print resolved runs without reading data or fitting")
        sub.add_argument("--siar-degree", type=positive_int, help="Polynomial degree; default: synthetic 3, YouTube 2")
        sub.add_argument("--siar-fi", action=argparse.BooleanOptionalAction, default=None,
                         help="Forward invariance; default: synthetic on, YouTube off")
        sub.add_argument("--siar-pc", action=argparse.BooleanOptionalAction, default=None,
                         help="Positive correlation; default: on")
        sub.add_argument("--siar-project-simplex", action=argparse.BooleanOptionalAction, default=None,
                         help="Project each predicted Euler step; default: on")
        sub.add_argument("--siar-objective-scale", type=positive_float,
                         help="Scale the full solver objective; default: synthetic 1, YouTube 100")
        if name == "synthetic":
            sub.add_argument("--families", default="cyclic,congestion")
            sub.add_argument("--dynamics", default="replicator,projection,MWU")
            sub.add_argument("--dimensions", default="3,4")
            sub.add_argument("--train-trajectories", type=positive_int, default=16)
            sub.add_argument("--eval-trajectories", type=positive_int, default=16)
            sub.add_argument("--steps", type=positive_int, default=20, help="Observed transitions per trajectory")
        else:
            sub.add_argument("--data", default=str(PROJECT_ROOT / "data/YouTubeTrending/youtube_multicountry_common.npz"))
            sub.add_argument("--raw-root", type=Path, help="Existing raw YouTube CSV/JSON directory; missing files are downloaded")
            sub.add_argument("--q-values", default="3,4,5,6,7,8,9")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if len(args.method) != len(set(args.method)):
        parser.error("Choose each method only once")
    if args.run_tag is not None and (not args.run_tag or Path(args.run_tag).name != args.run_tag or args.run_tag in {".", ".."}):
        parser.error("run-tag must be a single directory name")
    reserved = {"tables", "figures", *RESULT_SETTING_DIRS, *set().union(*RESULT_SETTING_DIRS.values())}
    if args.run_tag in reserved:
        parser.error("run-tag must not use an experiment setting or result directory name")
    if "siar" not in args.method and any(getattr(args, name) is not None for name in _SIAR_OPTIONS):
        parser.error("SIAR options require --method siar")
    try:
        (run_synthetic if args.experiment == "synthetic" else run_youtube)(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
