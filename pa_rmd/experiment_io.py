"""Predictor-level result saving and comparison validation.

Completion manifests, not exported JSONL files, are the source of truth.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any
import uuid
import zipfile

import numpy as np

SHARD_DIR = ".runs"
ARTIFACTS = ("metrics.jsonl", "training_history.jsonl", "inference_trajectories.npz")
SEED_FIELDS = {"seed", "data_seed", "model_seed"}
DATA_CFG_FIELDS = {
    "game_type", "N", "payoff_matrix", "payoff_intercept", "metric_type",
    "num_trajectories", "trajectory_length", "delta_t", "interior_epsilon",
    "observation_stride", "evaluation_protocol", "num_eval_trajectories",
}


def json_value(value: Any) -> Any:
    """JSON snapshot including dynamically added Config fields (not asdict)."""
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return np.asarray(value).tolist()
    if hasattr(value, "__dict__"):
        return json_value(vars(value))
    raise TypeError(f"Cannot serialize protocol input {type(value).__name__}")


def canonical(value: Any) -> bytes:
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_identity(value: Any) -> Any:
    """Content identity of all arrays, including real-data/support/observed pairs."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        arr = np.ascontiguousarray(np.asarray(value))
        if arr.dtype.hasobject:
            raise TypeError("Object arrays cannot identify experiment inputs")
        return {"shape": list(arr.shape), "dtype": arr.dtype.str,
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(k): input_identity(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [input_identity(v) for v in value]
    if inspect.isfunction(value):
        return {"callable": value.__module__ + "." + value.__qualname__,
                "source": inspect.getsource(value),
                "defaults": input_identity(value.__defaults__),
                "closure": [input_identity(c.cell_contents) for c in (value.__closure__ or ())]}
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {"class": type(value).__module__ + "." + type(value).__qualname__,
                "state": input_identity(vars(value))}
    return json_value(value)


def without_seeds(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: without_seeds(v) for k, v in value.items() if k not in SEED_FIELDS}
    if isinstance(value, list):
        return [without_seeds(v) for v in value]
    return value


def code_identity() -> dict:
    # Hash this checkout independently of the data/output workspace override.
    # A source/layout change intentionally creates a new run identity.
    package_root = Path(__file__).resolve().parent
    source_root = package_root.parent
    files = {
        "pa_rmd/" + p.relative_to(package_root).as_posix(): file_sha256(p)
        for p in sorted(package_root.rglob("*.py"))
    }
    entrypoints = [source_root / "main.py", source_root / "requirements.txt"]
    entrypoints.extend(sorted((source_root / "scripts").glob("*.bash")))
    files.update({p.relative_to(source_root).as_posix(): file_sha256(p) for p in entrypoints})
    return {"signature": fingerprint(files), "files": files}


def runtime_versions() -> dict:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "jax", "jaxlib", "optax", "scipy", "cvxpy", "clarabel", "scs"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def semantic_dataset_args(dataset_name: str, args: dict) -> dict:
    """Drop only selectors of a YouTube run within a larger requested sweep.

    Actual selected train/evaluation arrays and environment IDs remain signed,
    so changing the resolved split still conflicts even when indices are omitted.
    """
    snapshot = json_value(args)
    if dataset_name == "youtube-multicountry":
        snapshot = {key: value for key, value in snapshot.items() if key not in {"data_seeds", "split_index"}}
    return snapshot


def build_run_identity(dataset_name, dataset_args, cfg, train, blocks, meta, data_seed, model_seed) -> dict:
    dataset_args = semantic_dataset_args(dataset_name, dataset_args)
    cfg_snapshot = json_value(vars(cfg))
    evaluation_protocol = meta.get("evaluation_protocol")
    if evaluation_protocol is None:
        if dataset_name.startswith("synthetic"):
            raise ValueError("Synthetic data must supply evaluation_protocol and signatures")
        evaluation_protocol = f"{dataset_name}-observed-split-v1"
    if dataset_name.startswith("synthetic"):
        if evaluation_protocol != "independent-ic-v1" or any(not meta.get(k) for k in (
                "evaluation_signature", "base_dataset_signature", "split_signature")):
            raise ValueError("Synthetic data is missing independent evaluation identity")
    train_id = fingerprint(input_identity(train))
    eval_id = fingerprint(input_identity(blocks))
    split = [{"env_id": int(b.env_id), "group_label": str(b.group_label)} for b in blocks]
    if len({b["group_label"] for b in split}) != len(split):
        raise ValueError("Test group labels must be unique for prediction artifacts")
    train_env_ids = sorted(int(v) for v in np.unique(np.asarray(train.env_id))) if train.env_id is not None else []
    # Full cfg/arguments are used for resume. Only explicitly permitted seeds
    # vary in comparison identities; cross-method data alignment omits model cfg.
    data_settings = {"dataset": dataset_name, "dataset_args": without_seeds(json_value(dataset_args)),
                     "data_cfg": {k: v for k, v in cfg_snapshot.items() if k in DATA_CFG_FIELDS},
                     "split": split, "train_env_ids": train_env_ids,
                     "payoff_closure": getattr(train, "payoff_closure", None),
                     "support_groups": [b.group_label for b in blocks if b.payoff_support is not None]}
    code = code_identity()
    return {"evaluation_protocol": evaluation_protocol,
            "dataset": dataset_name, "dataset_args": json_value(dataset_args), "cfg": cfg_snapshot,
            "cfg_input_fingerprint": fingerprint(input_identity(cfg)),
            "dataset_args_input_fingerprint": fingerprint(input_identity(dataset_args)),
            "data_seed": int(data_seed), "model_seed": int(model_seed),
            "dataset_content_signature": meta.get("dataset_content_signature"),
            "train_fingerprint": train_id, "eval_fingerprint": eval_id,
            "evaluation_signature": meta.get("evaluation_signature") or eval_id,
            "base_dataset_signature": meta.get("base_dataset_signature") or train_id,
            "split_signature": meta.get("split_signature") or fingerprint(split),
            "train_env_ids": train_env_ids, "blocks": split,
            "data_comparison_fingerprint": fingerprint(without_seeds(data_settings)),
            "code_signature": code["signature"], "code_files": code["files"],
            "runtime_versions": {**runtime_versions(), "dtype": str(train.x_current.dtype)}}


def predictor_identity(predictor: Any, run_identity: dict) -> dict:
    input_summary = getattr(predictor, "input_config_summary", None)
    if callable(input_summary):
        # The built-in predictors resolve a fresh, untrained view of current
        # constructor inputs, including after fit or a failed evaluation.
        summary = json_value(input_summary())
        constructor = summary
        constructor_fp = fingerprint(constructor)
    else:
        summary = json_value(predictor.config_summary())
        constructor = json_value({"shared_cfg": getattr(predictor, "shared_cfg", None),
                                  "kwargs": getattr(predictor, "kwargs", {})})
        constructor_fp = fingerprint(constructor)
        cached = getattr(predictor, "_input_cache", None)
        if cached is not None and cached.get("postfit_summary") is not None:
            if cached["constructor_fingerprint"] != constructor_fp or summary != cached["postfit_summary"]:
                raise ValueError("Predictor configuration conflict: construct a fresh predictor after changing inputs")
            summary = cached["prefit_summary"]
    resolved = summary.get("resolved_model_config", summary)
    training = {**summary.get("training_config", {
        k: run_identity["cfg"].get(k) for k in ("num_epochs", "learning_rate", "batch_size", "grad_clip")}),
        "model_seed": run_identity["model_seed"]}
    payload = {"predictor": str(predictor.key),
               "class": type(predictor).__module__ + "." + type(predictor).__qualname__,
               "constructor_input_fingerprint": constructor_fp,
               "prefit_config": summary, "resolved_model_config": resolved, "training_config": training,
               "backend": summary.get("backend", "unspecified"),
               "seen_env_policy": summary.get("seen_env_policy", "not-applicable"),
               "unseen_env_policy": summary.get("unseen_env_policy", "not-applicable")}
    payload["model_fingerprint"] = fingerprint(resolved)
    payload["training_fingerprint"] = fingerprint(training)
    payload["model_comparison_fingerprint"] = fingerprint(without_seeds({
        "summary": summary, "cfg": run_identity["cfg"], "constructor": constructor}))
    payload["training_comparison_fingerprint"] = fingerprint(without_seeds(training))
    return payload


def _atomic_bytes(path: Path, contents: bytes) -> None:
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(contents)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, canonical(payload) + b"\n")


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(canonical(row) + b"\n" for row in rows)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@contextmanager
def run_lock(out: Path):
    """Fail rather than interleave two writers. Lock file remains reusable."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a+b") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another runner is writing {out}") from exc
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def prepare_run(out: Path, identity: dict, predictors: dict[str, dict], config_export: dict) -> dict:
    """Check ALL identities before replacing any existing output."""
    path = out / "run_manifest.json"
    run_fp = fingerprint(identity)
    if path.exists():
        try:
            manifest = json.loads(path.read_text())
            if manifest["run_fingerprint"] != fingerprint(manifest["identity"]):
                raise ValueError("Invalid run manifest checksum")
            if manifest["run_fingerprint"] != run_fp:
                raise ValueError("Run input/config/data/code identity conflict; use a new output directory")
            for key, value in manifest["predictors"].items():
                if value["fingerprint"] != fingerprint(value["identity"]):
                    raise ValueError(f"Invalid predictor identity: {key}")
            for key, value in predictors.items():
                if key in manifest["predictors"] and manifest["predictors"][key]["fingerprint"] != fingerprint(value):
                    raise ValueError(f"Predictor configuration conflict: {key}")
            config_path = out / "config.json"
            if config_path.exists():
                stored = json.loads(config_path.read_text())
                dataset_name = identity["dataset"]
                if semantic_dataset_args(dataset_name, stored) != semantic_dataset_args(dataset_name, config_export):
                    raise ValueError("config.json conflicts with the signed run input")
        except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid existing run manifest/config; refusing to overwrite") from exc
    else:
        existing = [p for p in out.iterdir() if p.name != ".run.lock"]
        if existing:
            raise ValueError("Nonempty output without a valid run manifest (missing or interrupted initialization); use a new directory")
        manifest = {"run_fingerprint": run_fp, "identity": identity, "predictors": {}}
    for key, value in predictors.items():
        manifest["predictors"][key] = {"fingerprint": fingerprint(value), "identity": value}
    # The input manifest is published before exports; it never asserts completion.
    _atomic_json(path, manifest)
    if not (out / "config.json").exists():
        _atomic_json(out / "config.json", config_export)
    (out / SHARD_DIR).mkdir(exist_ok=True)
    return manifest


def comparison_metadata(manifest: dict, key: str) -> dict:
    """Load shared configuration once from the manifest, not every metric row."""
    run = manifest["identity"]
    model = manifest["predictors"][key]
    names = ("evaluation_protocol", "evaluation_signature",
             "base_dataset_signature", "split_signature", "train_fingerprint", "eval_fingerprint",
             "code_signature", "data_comparison_fingerprint")
    model_names = ("backend", "seen_env_policy", "unseen_env_policy",
                   "resolved_model_config", "training_config", "model_fingerprint", "training_fingerprint",
                   "model_comparison_fingerprint", "training_comparison_fingerprint")
    return {**{k: run[k] for k in names}, **{k: model["identity"][k] for k in model_names},
            "run_fingerprint": manifest["run_fingerprint"], "predictor_fingerprint": model["fingerprint"]}


def metric_metadata(manifest: dict, key: str) -> dict:
    """Minimal references tying a saved row to its run and method configuration."""
    return {"run_fingerprint": manifest["run_fingerprint"],
            "predictor_fingerprint": manifest["predictors"][key]["fingerprint"]}


def artifact_contract(train: Any, blocks: list) -> dict:
    metrics = []
    predictions = {}
    step_names = [f"one_step{suffix}" for suffix in ("_mean", "_std", "_mse_mean", "_mse_std", "_js_mean", "_js_std")]
    for b in blocks:
        env, group = int(b.env_id), str(b.group_label)
        observed_x = b.observed_x_pairs if b.observed_x_pairs is not None else b.x_pairs
        observed_next = b.observed_x_next if b.observed_x_next is not None else b.x_next
        predictions[f"ground_truth/{group}/one_step/x_current"] = list(observed_x.shape)
        predictions[f"ground_truth/{group}/one_step/x_next"] = list(observed_next.shape)
        predictions[f"{group}/one_step"] = list(observed_next.shape)
        names = list(step_names)
        learned = b.payoff_support is not None
        if learned:
            names += [n.replace("one_step", "one_step_learned_v", 1) for n in step_names]
            names += [f"payoff_{n}_{s}" for n in ("l2", "mse") for s in ("mean", "std")]
            predictions[f"ground_truth/{group}/payoff/v_observed"] = list(b.v_pairs.shape)
            predictions[f"{group}/payoff/v_learned"] = list(b.v_pairs.shape)
            predictions[f"{group}/one_step_learned_v"] = list(b.x_next.shape)
        has_rollout = False
        if (b.game is not None or learned) and b.trajectories is not None:
            for i, trajectory in enumerate(b.trajectories):
                if trajectory.shape[0] < 2:
                    continue
                predictions[f"ground_truth/{group}/rollout/{i}"] = list(trajectory.shape)
                predictions[f"{group}/rollout/{i}"] = list(trajectory.shape)
                has_rollout = True
        if has_rollout:
            names += [f"rollout_{n}_{s}" for n in ("mse", "at_T") for s in ("mean", "std")]
        metrics.extend([[env, group, n] for n in names])
    return {"metrics": sorted(metrics), "predictions": predictions}


def _validate_artifacts(directory: Path, manifest: dict, key: str, contract: dict) -> dict:
    rows = _read_jsonl(directory / ARTIFACTS[0])
    expected_meta = metric_metadata(manifest, key)
    actual = []
    allowed_blocks = {(b["env_id"], b["group_label"]) for b in manifest["identity"]["blocks"]}
    for row in rows:
        if (row["predictor"] != key or any(row.get(k) != v for k, v in expected_meta.items())
                or any(row.get(k) != manifest["identity"][k] for k in ("data_seed", "model_seed"))):
            raise ValueError("Metric identity mismatch")
        item = (row["env_id"], row["group_label"], row["metric_name"])
        if item[:2] not in allowed_blocks or not np.isfinite(row["value"]):
            raise ValueError("Unexpected block or nonfinite metric")
        expected_role = "seen" if row["env_id"] in manifest["identity"]["train_env_ids"] else "unseen"
        if row.get("split_role") != expected_role:
            raise ValueError("Incorrect seen/unseen role")
        actual.append(item)
    if len(actual) != len(set(actual)) or not set(map(tuple, contract["metrics"])).issubset(actual):
        raise ValueError("Missing or duplicate environment metrics")
    history = _read_jsonl(directory / ARTIFACTS[1])
    if len(history) != 1 or history[0].get("predictor") != key or "loss" not in history[0]:
        raise ValueError("Missing valid history (empty loss must be explicit)")
    if (any(history[0].get(k) != v for k, v in expected_meta.items())
            or any(history[0].get(k) != manifest["identity"][k] for k in ("data_seed", "model_seed"))):
        raise ValueError("History identity mismatch")
    if any(not np.all(np.isfinite(np.asarray(history[0].get(k, []), dtype=float))) for k in ("loss", "payoff_loss")):
        raise ValueError("Nonfinite training history")
    components = history[0].get("loss_components", {})
    if not isinstance(components, dict):
        raise ValueError("Invalid loss component history")
    for values in components.values():
        if len(values) != len(history[0]["loss"]) or not np.all(np.isfinite(np.asarray(values, dtype=float))):
            raise ValueError("Invalid loss component length or values")
    timings = history[0].get("timings", {})
    if not isinstance(timings, dict) or any(not np.isfinite(value) or value < 0 for value in timings.values()):
        raise ValueError("Invalid training/evaluation timings")
    with np.load(directory / ARTIFACTS[2], allow_pickle=False) as data:
        if set(data.files) != set(contract["predictions"]):
            raise ValueError("Missing or unexpected predictions")
        for name, shape in contract["predictions"].items():
            if list(data[name].shape) != shape or not np.all(np.isfinite(data[name])):
                raise ValueError(f"Invalid prediction: {name}")
    return {"metric_count": len(rows), "prediction_count": len(contract["predictions"]), "history_count": 1}


def _predictor_dir(out: Path, key: str) -> Path:
    return out / SHARD_DIR / fingerprint(key)


def completed_shard(out: Path, manifest: dict, key: str, contract: dict | None = None) -> Path | None:
    """Corrupt/missing completion is incomplete, never grounds for skipping fit."""
    parent = _predictor_dir(out, key)
    try:
        completion = json.loads((parent / "completion.json").read_text())
        if not isinstance(completion, dict):
            return None
        seal = completion.pop("completion_fingerprint")
        if seal != fingerprint(completion):
            return None
        if (completion["run_fingerprint"] != manifest["run_fingerprint"]
                or completion["predictor_fingerprint"] != manifest["predictors"][key]["fingerprint"]):
            return None
        attempt = completion["attempt"]
        if not isinstance(attempt, str) or not attempt.startswith("attempt-") or Path(attempt).name != attempt:
            return None
        directory = parent / attempt
        saved_contract = completion["contract"]
        if contract is not None and saved_contract != contract:
            return None
        if set(completion["checksums"]) != set(ARTIFACTS):
            return None
        for name, digest in completion["checksums"].items():
            if file_sha256(directory / name) != digest:
                return None
        counts = _validate_artifacts(directory, manifest, key, saved_contract)
        if counts != completion["counts"]:
            return None
        return directory
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, zipfile.BadZipFile):
        return None


def publish_result(out: Path, manifest: dict, key: str, rows: list[dict], history: dict,
                   predictions: dict, contract: dict) -> Path:
    parent = _predictor_dir(out, key)
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / ("attempt-" + uuid.uuid4().hex)
    directory.mkdir()
    _atomic_bytes(directory / ARTIFACTS[0], _jsonl(rows))
    _atomic_bytes(directory / ARTIFACTS[1], _jsonl([history]))
    with (directory / ARTIFACTS[2]).open("wb") as fh:
        np.savez_compressed(fh, **{k: np.asarray(v) for k, v in sorted(predictions.items())})
        fh.flush()
        os.fsync(fh.fileno())
    counts = _validate_artifacts(directory, manifest, key, contract)
    completion = {"run_fingerprint": manifest["run_fingerprint"],
                  "predictor_fingerprint": manifest["predictors"][key]["fingerprint"],
                  "attempt": directory.name, "contract": contract, "counts": counts,
                  "checksums": {name: file_sha256(directory / name) for name in ARTIFACTS}}
    completion["completion_fingerprint"] = fingerprint(completion)
    _atomic_json(parent / "completion.json", completion)
    return directory


def rebuild_exports(out: Path, manifest: dict) -> None:
    rows, histories, predictions = [], [], {}
    for key in sorted(manifest["predictors"]):
        directory = completed_shard(out, manifest, key)
        if directory is None:
            continue
        rows.extend(_read_jsonl(directory / ARTIFACTS[0]))
        histories.extend(_read_jsonl(directory / ARTIFACTS[1]))
        with np.load(directory / ARTIFACTS[2], allow_pickle=False) as data:
            for name in sorted(data.files):
                output_key = name if name.startswith("ground_truth/") else f"{key}/{name}"
                if output_key in predictions and not np.array_equal(predictions[output_key], data[name]):
                    raise ValueError(f"Conflicting shared ground truth: {output_key}")
                predictions[output_key] = data[name]
    rows.sort(key=lambda r: (r["predictor"], r["env_id"], r["group_label"], r["metric_name"]))
    # Exports are independently atomic and reconstructible. They are never used
    # for completion or trusted by the protocol-aware analysis loader.
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=out)
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **predictions)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(name, out / ARTIFACTS[2])
    finally:
        if os.path.exists(name):
            os.unlink(name)
    _atomic_bytes(out / ARTIFACTS[1], _jsonl(histories))
    _atomic_bytes(out / ARTIFACTS[0], _jsonl(rows))


def load_completed_rows(out: Path, artifact: str = "metrics.jsonl") -> list[dict]:
    """Read verified shards without trusting stale/partially rebuilt exports."""
    if artifact not in ARTIFACTS[:2]:
        raise ValueError("Only metrics/history JSONL can be loaded as rows")
    manifest = json.loads((out / "run_manifest.json").read_text())
    if manifest.get("run_fingerprint") != fingerprint(manifest["identity"]):
        raise ValueError("Invalid run identity checksum")
    rows = []
    for key, payload in sorted(manifest["predictors"].items()):
        if payload["fingerprint"] != fingerprint(payload["identity"]):
            raise ValueError(f"Invalid predictor identity: {key}")
        directory = completed_shard(out, manifest, key)
        if directory is None:
            continue
        metadata = comparison_metadata(manifest, key)
        rows.extend({**row, **metadata} for row in _read_jsonl(directory / artifact))
    return rows


def load_comparison_rows(out: Path, artifact: str = "metrics.jsonl") -> list[dict]:
    """Read checked results; derive YouTube sweep scope without rewriting old runs.

    A sweep may vary the country split across data seeds. Full split/array
    identities must still match across methods and model seeds at each seed.
    """
    rows = load_completed_rows(out, artifact)
    manifest = json.loads((out / "run_manifest.json").read_text())
    run = manifest["identity"]
    if run["dataset"] != "youtube-multicountry":
        return rows
    scope = {
        "dataset": run["dataset"],
        "evaluation_protocol": run["evaluation_protocol"],
        "dataset_args": without_seeds(semantic_dataset_args(run["dataset"], run["dataset_args"])),
        "data_cfg": {k: v for k, v in run["cfg"].items() if k in DATA_CFG_FIELDS},
        "source_count": len(run["train_env_ids"]),
        "target_count": len(run["blocks"]),
        "dataset_content_signature": run.get("dataset_content_signature"),
    }
    return [{**row, "_comparison_scope": fingerprint(scope),
             "_dataset_content_verified": bool(run.get("dataset_content_signature"))} for row in rows]


def validate_comparison_rows(rows: list[dict]) -> None:
    """Shared table API: reject mixed protocols/budgets/unpaired data/duplicates.

    Different methods may use different structures. Only data/model seed fields
    may vary within a method.
    """
    if not rows:
        return
    common, methods, pools, samples = set(), {}, {}, set()
    required = ("predictor", "data_seed", "model_seed", "evaluation_protocol", "evaluation_signature",
                "base_dataset_signature", "split_signature", "train_fingerprint", "eval_fingerprint",
                "data_comparison_fingerprint", "code_signature", "backend",
                "model_comparison_fingerprint", "training_comparison_fingerprint", "model_fingerprint",
                "training_fingerprint", "resolved_model_config", "training_config", "seen_env_policy", "unseen_env_policy")
    for row in rows:
        if any(k not in row for k in required):
            raise ValueError("Missing protocol/configuration metadata; summary.csv cannot establish provenance")
        if fingerprint(row["resolved_model_config"]) != row["model_fingerprint"] or fingerprint(row["training_config"]) != row["training_fingerprint"]:
            raise ValueError("Model/training configuration fingerprint mismatch")
        common.add((row["evaluation_protocol"], row.get("_comparison_scope", row["data_comparison_fingerprint"])))
        method = tuple(row[k] for k in ("code_signature", "backend",
                       "model_comparison_fingerprint", "training_comparison_fingerprint", "seen_env_policy", "unseen_env_policy"))
        method += (fingerprint(without_seeds(row["resolved_model_config"])),
                   fingerprint(without_seeds(row["training_config"])))
        key = row["predictor"]
        if key in methods and methods[key] != method:
            raise ValueError(f"Incompatible implementation/model hyperparameters/training budget: {key}")
        methods[key] = method
        pool = tuple(row[k] for k in ("evaluation_signature", "base_dataset_signature", "split_signature", "train_fingerprint", "eval_fingerprint", "data_comparison_fingerprint"))
        seed = row["data_seed"]
        if seed in pools and pools[seed] != pool:
            raise ValueError(f"Different train/evaluation pools or splits for data_seed={seed}")
        pools[seed] = pool
        if "metric_name" in row:
            if row.get("split_role") not in {"seen", "unseen"} or "env_id" not in row:
                raise ValueError("New metrics require explicit env_id and seen/unseen role")
            sample = (key, seed, row["model_seed"], row["env_id"], row["group_label"], row["metric_name"])
        else:
            sample = (key, seed, row["model_seed"])
        if sample in samples:
            raise ValueError(f"Duplicate result sample: {sample}")
        samples.add(sample)
    if len(common) != 1:
        raise ValueError("Cannot combine different evaluation protocols/data settings/splits")
