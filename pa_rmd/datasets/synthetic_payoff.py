from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from pa_rmd import data_generation

import jax
import jax.numpy as jnp
import numpy as np

from pa_rmd.config import Config
from pa_rmd.data_generation import generate_dataset
from pa_rmd.experiment_io import file_sha256
from pa_rmd.utils import derive_seed, seed_map
from pa_rmd.games import build_game
from pa_rmd.metrics import build_metric
from pa_rmd.payoffs import (
    canonical_payoff_matrix_for_config,
    make_payoff_base_matrix,
    normalize_payoff_base_family,
    payoff_family_label_for_matrix,
    to_builtin,
)
from pa_rmd.samplers import from_name, normalize_payoff_sampling_mode, payoff_sampling_mode_display
from pa_rmd.evaluate import TestBlock, TrainBatch

SYNTHETIC_ARTIFACT_VERSION = 2
from pa_rmd.paths import PROJECT_ROOT

SYNTHETIC_DATASET_DIR = PROJECT_ROOT / "data" / "synthetic_datasets"
EVALUATION_PROTOCOL = "independent-ic-v1"
EVALUATION_SCHEMA_VERSION = 1
_EVAL_MAX_RETRIES = 3


def _safe_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", "=") else "_" for ch in str(value))
    return cleaned.strip("_") or "default"


def _indices_tag(indices: list[int]) -> str:
    return "-".join(str(int(idx)) for idx in indices) if indices else "none"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2, sort_keys=True) + "\n")


def _signature(payload: dict) -> str:
    raw = json.dumps(to_builtin(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _base_matrix_and_spec(
    cfg: Config,
    *,
    payoff_base_family: str | None,
    payoff_dim: int | None,
    payoff_base_seed: int,
) -> tuple[np.ndarray, dict[str, Any], Config]:
    family = normalize_payoff_base_family(payoff_base_family)
    if family is None:
        return canonical_payoff_matrix_for_config(cfg), {"payoff_base_family": None, "payoff_base_seed": None}, cfg

    n = int(payoff_dim) if payoff_dim is not None else int(cfg.N)
    base = make_payoff_base_matrix(family=family, N=n, seed=int(payoff_base_seed))
    out_cfg = replace(cfg, game_type="custom_matrix", N=n, payoff_matrix=base, payoff_intercept=None)
    return base, {
        "payoff_base_family": family,
        "payoff_base_seed": None if family == "zero_sum" else int(payoff_base_seed),
    }, out_cfg


def _sampler_kwargs(
    mode: str,
    spec: dict[str, Any],
    base: np.ndarray,
    cfg: Config,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    if mode == "congestion":
        return {
            "num_routes": int(cfg.N),
            "target_frob": float(kwargs.get("matrix_frob") or 1.0),
            "incidence_matrix": kwargs.get("incidence_matrix"),
        }
    return {"A_base": base}


def _validate_payoff_kwargs(kwargs: dict[str, Any]) -> None:
    allowed = {"matrix_frob", "incidence_matrix"}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise TypeError(f"Unsupported synthetic dataset argument(s): {', '.join(unknown)}")


def _family_label(base_matrix: np.ndarray, family: str | None) -> str:
    if family == "zero_sum":
        return "zeroSum"
    if family == "potential":
        return "potential"
    return payoff_family_label_for_matrix(base_matrix)


def _dataset_id(base_matrix: np.ndarray, family: str | None, sampling_mode: str, payoff_dim: int) -> str:
    family_label = _family_label(base_matrix, family)
    sampling_label = payoff_sampling_mode_display(sampling_mode)
    if family_label == "zeroSum" and sampling_label == "cyclic-isomorphism":
        return f"zs-cyclic-DIM{int(payoff_dim)}"
    if family_label == "potential" and sampling_label == "congestion":
        return f"pot-congestion-DIM{int(payoff_dim)}"
    return _safe_component(f"{family_label}-{sampling_label}-DIM{int(payoff_dim)}")


def _dynamics_tag(cfg: Config) -> str:
    if cfg.metric_type == "replicator":
        return ""
    return f"-dyn={_safe_component(cfg.metric_type)}"






def _artifact_dir(dataset_id: str, data_seed: int) -> Path:
    return SYNTHETIC_DATASET_DIR / dataset_id / f"data_seed={int(data_seed)}"


def _resolve_split(
    payoff_ids: list[int],
    num_payoff_train: int,
    payoff_train_indices: list[int] | None,
    payoff_test_indices: list[int] | str | None,
) -> tuple[list[int], list[int]]:
    available = sorted(int(idx) for idx in payoff_ids)
    available_set = set(available)
    if not available:
        raise ValueError("Synthetic dataset contains no payoff operators.")

    if payoff_train_indices is None:
        train_count = min(int(num_payoff_train), max(1, len(available) - 1))
        train_ids = available[:train_count]
    else:
        train_ids = [int(idx) for idx in payoff_train_indices]
        _validate_ids(train_ids, available_set, "--payoff-train")

    if payoff_test_indices is None or payoff_test_indices == "complement":
        test_ids = [idx for idx in available if idx not in set(train_ids)]
    elif payoff_test_indices == "all":
        test_ids = list(available)
    else:
        test_ids = [int(idx) for idx in payoff_test_indices]
        _validate_ids(test_ids, available_set, "--payoff-test")

    if not train_ids:
        raise ValueError("--payoff-train resolved to an empty set.")
    if not test_ids:
        raise ValueError("--payoff-test resolved to an empty set.")
    return train_ids, test_ids


def _validate_ids(ids: list[int], available: set[int], label: str) -> None:
    missing = [idx for idx in ids if idx not in available]
    if missing:
        raise ValueError(f"{label} references payoff ids not present in the dataset: {missing}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate ids: {ids}")


def _split_id(train_ids: list[int], test_ids: list[int]) -> str:
    return f"train={_indices_tag(train_ids)}_test={_indices_tag(test_ids)}"


def _split_payload(dataset: dict, train_ids: list[int], test_ids: list[int]) -> dict:
    payload = {
        "split_id": _split_id(train_ids, test_ids),
        "dataset_id": dataset["dataset_id"],
        "dataset_signature": dataset["dataset_signature"],
        "data_seed": int(dataset["data_seed"]),
        "train": {"payoff_ids": train_ids, "trajectory_selector": "all"},
        "test": {"payoff_ids": test_ids, "trajectory_selector": "all"},
        "seen_payoff_ids": [idx for idx in test_ids if idx in set(train_ids)],
        "unseen_payoff_ids": [idx for idx in test_ids if idx not in set(train_ids)],
    }
    payload["split_signature"] = _signature(payload)
    return payload


def _pack_trajectories(trajectories) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([int(traj.shape[0]) for traj in trajectories], dtype=np.int32)
    if lengths.size == 0:
        values = np.empty((0, 0), dtype=np.float32)
    else:
        values = np.concatenate(
            [np.asarray(jax.device_get(traj), dtype=np.float32) for traj in trajectories],
            axis=0,
        )
    return values, lengths


def _unpack_trajectories(values: np.ndarray, lengths: np.ndarray) -> list[jnp.ndarray]:
    trajectories: list[jnp.ndarray] = []
    cursor = 0
    for length in np.asarray(lengths, dtype=np.int32):
        next_cursor = cursor + int(length)
        trajectories.append(jnp.asarray(values[cursor:next_cursor]))
        cursor = next_cursor
    return trajectories


def _sample_payoff_pool(
    cfg: Config,
    *,
    sampler_name: str,
    num_payoff: int,
    num_payoff_train: int,
    data_seed: int,
    shift_seed: int | None,
    payoff_train_indices: list[int] | None,
    payoff_base_family: str | None,
    payoff_dim: int | None,
    payoff_base_seed: int,
    kwargs: dict[str, Any],
):
    if shift_seed is None:
        shift_seed = derive_seed(int(data_seed), "payoff_pool")
    base, spec, cfg = _base_matrix_and_spec(
        cfg,
        payoff_base_family=payoff_base_family,
        payoff_dim=payoff_dim,
        payoff_base_seed=payoff_base_seed,
    )
    mode = normalize_payoff_sampling_mode(sampler_name)
    sampler = from_name(mode)(**_sampler_kwargs(mode, spec, base, cfg, kwargs))
    sample = sampler.sample(
        int(num_payoff_train),
        int(num_payoff) - int(num_payoff_train),
        seed=int(shift_seed),
        train_indices=payoff_train_indices,
        test_indices="all",
    )
    if len(sample.test_matrices) != int(num_payoff):
        raise ValueError(
            "Sampler did not produce a complete payoff pool; "
            f"expected {int(num_payoff)}, got train={len(sample.train_matrices)} test={len(sample.test_matrices)}"
        )
    matrices = sample.test_matrices
    intercepts = sample.test_intercepts
    payoff_ids = sample.test_indices
    return cfg, base, spec, mode, sample.metadata, [int(idx) for idx in payoff_ids], matrices, intercepts


def _artifact_identity(
    cfg: Config,
    *,
    sampler_name: str,
    payoff_base_family: str | None,
    payoff_dim: int | None,
    payoff_base_seed: int,
) -> tuple[str, Config, np.ndarray, dict[str, Any], str]:
    if cfg.metric_type not in {"replicator", "projection", "MWU"}:
        raise ValueError(f"Unsupported main-experiment dynamics: {cfg.metric_type!r}")
    base, spec, cfg = _base_matrix_and_spec(
        cfg,
        payoff_base_family=payoff_base_family,
        payoff_dim=payoff_dim,
        payoff_base_seed=payoff_base_seed,
    )
    mode = normalize_payoff_sampling_mode(sampler_name)
    dataset_id = (
        _dataset_id(base, spec.get("payoff_base_family"), mode, int(cfg.N))
        + _dynamics_tag(cfg)
    )
    return dataset_id, cfg, base, spec, mode


def _generation_config(cfg: Config) -> dict:
    return {
        "num_trajectories_per_payoff": int(cfg.num_trajectories),
        "trajectory_length": int(cfg.trajectory_length),
        "delta_t": float(cfg.delta_t),
        "observation_stride": int(cfg.observation_stride),
        "dt": float(cfg.observed_delta_t()),
        "initial_simplex_margin": float(cfg.interior_epsilon),
        "substeps": 20,
    }


def _dataset_spec(
    *,
    dataset_id: str,
    data_seed: int,
    cfg: Config,
    base: np.ndarray,
    spec: dict[str, Any],
    mode: str,
    num_payoff: int,
    sampler_metadata: dict[str, Any],
    kwargs: dict[str, Any],
) -> dict:
    payload = {
        "version": SYNTHETIC_ARTIFACT_VERSION,
        "dataset_id": dataset_id,
        "dataset_type": "synthetic-payoff-family",
        "data_seed": int(data_seed),
        "game": {
            "base_family": _family_label(base, spec.get("payoff_base_family")),
            "sampling_mode": payoff_sampling_mode_display(mode),
            "payoff_dim": int(cfg.N),
            "num_payoff": int(num_payoff),
            "metric_type": cfg.metric_type,
            "dynamics_type": cfg.metric_type,
        },
        "generation": _generation_config(cfg),
        "payoff_sampling": {
            "payoff_base_seed": spec.get("payoff_base_seed"),
            "matrix_frob": kwargs.get("matrix_frob"),
        },
        "sampler_metadata": sampler_metadata,
        "files": {
            "payoff_operators": "payoff_operators.npz",
            "trajectories": "trajectories.npz",
        },
        "splits": [],
    }
    payload["dataset_signature"] = _signature({k: v for k, v in payload.items() if k != "splits"})
    return payload


def _payoff_ids(operator_payload: dict[str, np.ndarray]) -> list[int]:
    if "payoff_ids" in operator_payload:
        return [int(idx) for idx in np.asarray(operator_payload["payoff_ids"], dtype=np.int32).tolist()]
    ids = []
    for key in operator_payload:
        if key.startswith("payoff_") and key.endswith("/matrix"):
            ids.append(int(key.split("/", 1)[0].replace("payoff_", "")))
    return sorted(set(ids))


def _validate_artifact_schema(artifact_dir: Path, requested_ids: list[int] | None = None) -> dict:
    dataset_path = artifact_dir / "dataset.json"
    operator_path = artifact_dir / "payoff_operators.npz"
    trajectory_path = artifact_dir / "trajectories.npz"
    if not dataset_path.exists() or not operator_path.exists() or not trajectory_path.exists():
        raise FileNotFoundError(
            f"Synthetic dataset artifact is incomplete at {artifact_dir}; "
            "expected dataset.json, payoff_operators.npz, and trajectories.npz."
        )
    dataset = _read_json(dataset_path)
    # Older main-experiment manifests contain an inert zero-response identity.
    # Reject appendix artifacts before any loading or cache creation.
    response = dataset.get("response_heterogeneity", {"mode": "none", "alpha": 0.0})
    if response.get("mode") != "none" or response.get("alpha") != 0.0:
        raise ValueError("Unsupported response_heterogeneity artifact; rebuild main-experiment data explicitly.")
    operators = _load_npz(operator_path)
    if any(key.endswith("/relative_response") for key in operators):
        raise ValueError("Unsupported relative_response operator in main-experiment data.")
    trajectories = _load_npz(trajectory_path)
    ids = _payoff_ids(operators)
    if requested_ids is not None:
        _validate_ids([int(idx) for idx in requested_ids], set(ids), "requested payoff ids")
    for payoff_idx in ids if requested_ids is None else requested_ids:
        prefix = f"payoff_{int(payoff_idx)}"
        required_operator = [f"{prefix}/matrix", f"{prefix}/intercept"]
        required_trajectory = [
            f"{prefix}/x_current",
            f"{prefix}/x_next",
            f"{prefix}/pair_dt",
            f"{prefix}/trajectory_values",
            f"{prefix}/trajectory_lengths",
        ]
        missing = [key for key in required_operator if key not in operators]
        missing += [key for key in required_trajectory if key not in trajectories]
        if missing:
            raise ValueError(
                f"Synthetic dataset artifact at {artifact_dir} is not in the fixed payoff schema; "
                f"missing keys for payoff_{int(payoff_idx)}: {missing}. Rebuild the dataset artifact explicitly."
            )
    return dataset


def _dataset_dynamics_type(dataset: dict) -> str:
    game = dataset.get("game", {})
    return str(game.get("dynamics_type", game.get("metric_type", "replicator")))


def _validate_artifact_dynamics(dataset: dict, cfg: Config, artifact_dir: Path) -> None:
    actual = _dataset_dynamics_type(dataset)
    expected = cfg.metric_type
    if actual != expected:
        raise ValueError(
            f"Synthetic dataset artifact at {artifact_dir} was generated with "
            f"dynamics_type={actual!r}, but the current request expects {expected!r}. "
            "Use a dynamics-specific artifact path or rebuild the dataset artifact explicitly."
        )


def _operator_intercept(operators: dict[str, np.ndarray], payoff_idx: int) -> np.ndarray | None:
    raw = operators[f"payoff_{int(payoff_idx)}/intercept"]
    if raw.size == 0:
        return None
    return np.asarray(raw, dtype=np.float32)


def _load_payoff_trajectory(
    trajectories: dict[str, np.ndarray],
    payoff_idx: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, list[jnp.ndarray]]:
    prefix = f"payoff_{int(payoff_idx)}"
    return (
        jnp.asarray(trajectories[f"{prefix}/x_current"]),
        jnp.asarray(trajectories[f"{prefix}/x_next"]),
        jnp.asarray(trajectories[f"{prefix}/pair_dt"]),
        _unpack_trajectories(trajectories[f"{prefix}/trajectory_values"], trajectories[f"{prefix}/trajectory_lengths"]),
    )


def _write_artifact(
    artifact_dir: Path,
    dataset: dict,
    cfg: Config,
    payoff_ids: list[int],
    matrices: list[np.ndarray],
    intercepts: list[np.ndarray | None],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metric = build_metric(cfg) if cfg.metric_type in {"replicator", "projection"} else None
    trajectory_seeds = seed_map(int(dataset["data_seed"]), "trajectory", payoff_ids)
    operator_payload: dict[str, np.ndarray] = {"payoff_ids": np.asarray(payoff_ids, dtype=np.int32)}
    trajectory_payload: dict[str, np.ndarray] = {"payoff_ids": np.asarray(payoff_ids, dtype=np.int32)}

    for payoff_idx, matrix, intercept in zip(payoff_ids, matrices, intercepts):
        cfg_i = replace(cfg, game_type="custom_matrix", payoff_matrix=matrix, payoff_intercept=intercept)
        game = build_game(cfg_i)
        x_current, x_next, pair_dt, trajectories = generate_dataset(
            cfg_i,
            game,
            metric,
            seed=trajectory_seeds[int(payoff_idx)],
        )
        prefix = f"payoff_{int(payoff_idx)}"
        values, lengths = _pack_trajectories(trajectories)
        operator_payload[f"{prefix}/matrix"] = np.asarray(matrix, dtype=np.float32)
        operator_payload[f"{prefix}/intercept"] = (
            np.asarray(intercept, dtype=np.float32) if intercept is not None else np.empty((0,), dtype=np.float32)
        )
        trajectory_payload[f"{prefix}/x_current"] = np.asarray(jax.device_get(x_current), dtype=np.float32)
        trajectory_payload[f"{prefix}/x_next"] = np.asarray(jax.device_get(x_next), dtype=np.float32)
        trajectory_payload[f"{prefix}/pair_dt"] = np.asarray(jax.device_get(pair_dt), dtype=np.float32)
        trajectory_payload[f"{prefix}/trajectory_values"] = values
        trajectory_payload[f"{prefix}/trajectory_lengths"] = lengths
        trajectory_payload[f"{prefix}/trajectory_seed"] = np.asarray(trajectory_seeds[int(payoff_idx)], dtype=np.int64)

    _write_json(artifact_dir / "dataset.json", dataset)
    np.savez_compressed(artifact_dir / "payoff_operators.npz", **operator_payload)
    np.savez_compressed(artifact_dir / "trajectories.npz", **trajectory_payload)


def _evaluation_protocol(cfg: Config, requested: str | None) -> str:
    requested = cfg.evaluation_protocol if requested is None else requested
    if requested != EVALUATION_PROTOCOL:
        raise ValueError(f"Unsupported synthetic evaluation_protocol={requested!r}.")
    return requested


def _eval_count(cfg: Config, requested: int | None) -> int:
    value = requested if requested is not None else getattr(cfg, "num_eval_trajectories", None)
    value = cfg.num_trajectories if value is None else value
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError("num_eval_trajectories must be a positive integer.")
    return int(value)




@contextmanager
def _cache_lock(path: Path):
    """Kernel locks survive neither exceptions nor process death; stale temp dirs are ignored."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _publish_json(path: Path, payload: dict, *, require_existing: bool = False) -> None:
    """Publish a complete immutable manifest, without overwriting a concurrent writer."""
    payload = to_builtin(payload)
    if path.exists():
        if _read_json(path) != payload:
            raise ValueError(f"Synthetic manifest conflict at {path}")
        return
    if require_existing:
        raise FileNotFoundError(f"Synthetic evaluation split is missing: {path}; call ensure_synthetic_dataset first.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".pending-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if _read_json(path) != payload:
                    raise ValueError(f"Synthetic manifest conflict at {path}")
        finally:
            temporary.unlink(missing_ok=True)


def _normalize_generation_request(request: dict, mode: str) -> dict:
    """Compare only consumed sampler inputs, including requests saved before normalization."""
    sampling = request["sampler_kwargs"]
    if mode == "cyclic_isomorphism":
        # Cyclic matrices depend on dimension, not on the zero-sum base's entries.
        base = sampling.get("A_base")
        normalized = {"sampler_kwargs": {
            "payoff_dim": len(base) if base is not None else sampling["payoff_dim"],
            "base_family": payoff_family_label_for_matrix(base) if base is not None else sampling["base_family"],
        }}
    else:
        keys = ("num_routes", "incidence_matrix", "target_frob")
        normalized = {"sampler_kwargs": {key: sampling[key] for key in keys}}
    normalized["shift_seed"] = request["shift_seed"]
    return to_builtin(normalized)


def _generation_request(cfg: Config, base: np.ndarray, spec: dict, mode: str,
                        data_seed: int, shift_seed: int | None, kwargs: dict) -> dict:
    # Only generation inputs; split selection, model seed and optimizer settings are excluded.
    return _normalize_generation_request({
        "base_matrix": base,
        "shift_seed": int(shift_seed) if shift_seed is not None else derive_seed(data_seed, "payoff_pool"),
        "sampler_kwargs": _sampler_kwargs(mode, spec, base, cfg, kwargs),
    }, mode)


def _validate_base_config(dataset: dict, cfg: Config, artifact_dir: Path, *,
                          dataset_id: str, num_payoff: int, data_seed: int,
                          base: np.ndarray, spec: dict, mode: str,
                          shift_seed: int | None, kwargs: dict) -> None:
    expected_generation = _generation_config(cfg)
    expected = {
        "dataset_id": dataset_id, "data_seed": int(data_seed),
        "game.payoff_dim": int(cfg.N), "game.num_payoff": int(num_payoff),
        "game.base_family": _family_label(base, spec.get("payoff_base_family")),
        "game.sampling_mode": payoff_sampling_mode_display(mode),
        **{f"generation.{key}": value for key, value in expected_generation.items()},
    }
    differences = []
    for key, value in expected.items():
        actual = dataset
        for part in key.split("."):
            actual = actual.get(part) if isinstance(actual, dict) else None
        if actual != value:
            differences.append(f"{key}: cached={actual!r}, requested={value!r}")
    request = _generation_request(cfg, base, spec, mode, data_seed, shift_seed, kwargs)
    if "generation_request" in dataset:
        cached_request = _normalize_generation_request(dataset["generation_request"], mode)
        if cached_request != request:
            differences.append(f"generation_request: cached={cached_request!r}, requested={request!r}")
    else:
        # Older schemas record fewer sampling inputs. Never guess an explicitly supplied
        # value that cannot be verified from those records.
        sampling = dataset.get("payoff_sampling", {})
        metadata = dataset.get("sampler_metadata", {})
        if mode == "congestion":
            actual = sampling.get("matrix_frob")
            actual = 1.0 if actual is None else actual
            expected_frob = kwargs.get("matrix_frob")
            expected_frob = 1.0 if expected_frob is None else expected_frob
            if actual != expected_frob:
                differences.append(f"payoff_sampling.matrix_frob: cached={actual!r}, requested={expected_frob!r}")
        # Validate recorded pool tags, not train_tags (which describe the original split).
        # This checks the seed request without sampling or replacing a payoff operator.
        pool_seed = request.get("shift_seed")
        tags = None
        if mode == "cyclic_isomorphism":
            tags = np.random.default_rng(pool_seed).permutation(math.factorial(int(cfg.N) - 1))[:num_payoff].tolist()
        elif mode == "congestion":
            tags = list(range(pool_seed, pool_seed + num_payoff))
        if tags is not None and metadata.get("test_tags") != tags:
            differences.append("shift_seed does not match recorded payoff pool tags")
        if mode == "congestion" and kwargs.get("incidence_matrix") is not None:
            records = [metadata.get("base_metadata", {}), *metadata.get("train_metadata", []),
                       *metadata.get("test_metadata", [])]
            recorded = [record["B"] for record in records if record.get("B") is not None]
            if not recorded:
                differences.append("incidence_matrix cannot be verified in this older base schema")
            elif any(not np.array_equal(np.asarray(value, dtype=np.float32),
                                        np.asarray(kwargs["incidence_matrix"], dtype=np.float32)) for value in recorded):
                differences.append("incidence_matrix does not match recorded congestion B")
    if dataset.get("version") not in {1, SYNTHETIC_ARTIFACT_VERSION}:
        differences.append(f"unsupported base schema version={dataset.get('version')!r}")
    if differences:
        raise ValueError(f"Synthetic base generation configuration mismatch at {artifact_dir}: " + "; ".join(differences))
    _validate_artifact_dynamics(dataset, cfg, artifact_dir)
    semantic = {key: value for key, value in dataset.items() if key not in {"dataset_signature", "splits"}}
    if dataset.get("dataset_signature") != _signature(semantic):
        raise ValueError(f"Stale or corrupt base dataset_signature at {artifact_dir}")
    if dataset.get("files") != {"payoff_operators": "payoff_operators.npz", "trajectories": "trajectories.npz"}:
        raise ValueError(f"Unexpected base artifact file references at {artifact_dir}")


def _validate_pool(payload: dict, payoff_ids: list[int], cfg: Config, count: int, label: str) -> dict[int, np.ndarray]:
    if _payoff_ids(payload) != payoff_ids:
        raise ValueError(f"{label}: trajectory payoff ids do not match operators")
    initial = {}
    dt = data_generation._MWU_PAIR_DT if cfg.metric_type == "MWU" else cfg.observed_delta_t()
    for idx in payoff_ids:
        prefix = f"payoff_{idx}"
        try:
            values = payload[f"{prefix}/trajectory_values"]
            lengths = payload[f"{prefix}/trajectory_lengths"]
            if (lengths.shape != (count,) or not np.all(lengths == int(cfg.trajectory_length) + 1)
                    or values.shape != (count * (int(cfg.trajectory_length) + 1), int(cfg.N))):
                raise ValueError("trajectory count/length/dimension mismatch")
            if not np.all(np.isfinite(values)):
                raise ValueError("non-finite trajectory values")
            trajectories = values.reshape(count, int(cfg.trajectory_length) + 1, int(cfg.N))
            for key, expected in {
                "x_current": trajectories[:, :-1].reshape(-1, int(cfg.N)),
                "x_next": trajectories[:, 1:].reshape(-1, int(cfg.N)),
                "pair_dt": np.full((count * int(cfg.trajectory_length),), dt, dtype=np.float32),
            }.items():
                if not np.array_equal(payload[f"{prefix}/{key}"], expected):
                    raise ValueError(f"{key} does not match rollout ground truth/time step")
            seed = payload[f"{prefix}/trajectory_seed"]
            if seed.shape != () or not np.issubdtype(seed.dtype, np.integer):
                raise ValueError("missing or invalid trajectory_seed")
            initial[idx] = trajectories[:, 0]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{label}, {prefix}: {exc}") from exc
    return initial


def _validate_operators(operators: dict, payoff_ids: list[int], cfg: Config) -> None:
    for idx in payoff_ids:
        matrix = operators[f"payoff_{idx}/matrix"]
        intercept = operators[f"payoff_{idx}/intercept"]
        if matrix.shape != (cfg.N, cfg.N) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"Invalid payoff_{idx} matrix")
        if intercept.shape not in {(0,), (cfg.N,)} or not np.all(np.isfinite(intercept)):
            raise ValueError(f"Invalid payoff_{idx} intercept")


def _evaluation_identity(artifact_dir: Path, dataset: dict, cfg: Config, count: int) -> dict:
    checksums = {name: file_sha256(artifact_dir / filename) for name, filename in dataset["files"].items()}
    base_semantic = {key: dataset[key] for key in ("dataset_id", "data_seed", "game", "generation", "payoff_sampling")}
    # Preserve an existing historical zero-response identity only for old caches.
    if "response_heterogeneity" in dataset:
        base_semantic["response_heterogeneity"] = dataset["response_heterogeneity"]
    base_semantic["file_checksums"] = checksums
    generation = dict(dataset["generation"], num_trajectories_per_payoff=count)
    generation["pair_dt"] = data_generation._MWU_PAIR_DT if cfg.metric_type == "MWU" else cfg.observed_delta_t()
    identity = {
        "version": EVALUATION_SCHEMA_VERSION, "evaluation_protocol": EVALUATION_PROTOCOL,
        "base_dataset_signature": _signature(base_semantic), "base_file_checksums": checksums,
        "data_seed": int(dataset["data_seed"]), "generation": generation,
        "seed_namespace": "trajectory_eval_v1", "files": {"trajectories": "trajectories.npz"},
    }
    identity["evaluation_signature"] = _signature(identity)
    return identity


def _evaluation_dir(artifact_dir: Path, identity: dict) -> Path:
    return artifact_dir / "evaluation" / EVALUATION_PROTOCOL / identity["evaluation_signature"]


def _eval_seed(data_seed: int, idx: int, retry: int) -> int:
    parts = ("trajectory_eval_v1", idx)
    return derive_seed(data_seed, *parts) if retry == 0 else derive_seed(data_seed, *parts, "retry", retry)


def _initial_rows(initial: dict[int, np.ndarray]) -> set[tuple]:
    return {tuple(row.tolist()) for values in initial.values() for row in values}


def _initial_check(values: np.ndarray, forbidden: set[tuple]) -> dict:
    rows = [tuple(row.tolist()) for row in values]
    return {"train_overlap_count": sum(row in forbidden for row in rows),
            "duplicate_count": len(rows) - len(set(rows)), "num_initial_conditions": len(rows)}


def _validate_evaluation(eval_dir: Path, identity: dict, cfg: Config, payoff_ids: list[int],
                         training: dict, initial: dict) -> tuple[dict, dict]:
    manifest_path = eval_dir / "evaluation.json"
    if not manifest_path.exists() or not (eval_dir / "trajectories.npz").exists():
        raise FileNotFoundError(f"Incomplete evaluation cache at {eval_dir}; refusing to overwrite it.")
    manifest = _read_json(manifest_path)
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError(f"Evaluation identity/signature conflict at {eval_dir}")
    if manifest.get("manifest_signature") != _signature({k: v for k, v in manifest.items() if k != "manifest_signature"}):
        raise ValueError(f"Corrupt evaluation manifest_signature at {eval_dir}")
    if manifest.get("trajectory_sha256") != file_sha256(eval_dir / "trajectories.npz"):
        raise ValueError(f"Evaluation trajectory checksum mismatch at {eval_dir}")
    payload = _load_npz(eval_dir / "trajectories.npz")
    eval_initial = _validate_pool(payload, payoff_ids, cfg, identity["generation"]["num_trajectories_per_payoff"], str(eval_dir))
    forbidden = _initial_rows(initial)
    for idx in payoff_ids:
        record = manifest.get("environments", {}).get(str(idx), {})
        retry = record.get("retry_count", -1)
        if not isinstance(retry, int) or not 0 <= retry <= _EVAL_MAX_RETRIES:
            raise ValueError(f"Invalid eval retry record for payoff_{idx}")
        expected_seed = _eval_seed(identity["data_seed"], idx, retry)
        train_seed = int(training[f"payoff_{idx}/trajectory_seed"])
        check = _initial_check(eval_initial[idx], forbidden)
        if (record.get("seed") != expected_seed or int(payload[f"payoff_{idx}/trajectory_seed"]) != expected_seed
                or expected_seed == train_seed or record.get("train_seed") != train_seed
                or record.get("initial_conditions") != check
                or check["train_overlap_count"] or check["duplicate_count"]):
            raise ValueError(f"Evaluation seed/initial-condition conflict for payoff_{idx} at {eval_dir}")
    return manifest, payload


def _build_evaluation(eval_dir: Path, identity: dict, cfg: Config, payoff_ids: list[int],
                      operators: dict, training: dict, initial: dict) -> None:
    payload = {"payoff_ids": np.asarray(payoff_ids, dtype=np.int32)}
    environments = {}
    forbidden = _initial_rows(initial)
    metric = build_metric(cfg) if cfg.metric_type in {"replicator", "projection"} else None
    for idx in payoff_ids:
        prefix = f"payoff_{idx}"
        train_seed = int(training[f"{prefix}/trajectory_seed"])
        cfg_i = replace(cfg, game_type="custom_matrix", payoff_matrix=operators[f"{prefix}/matrix"],
                        payoff_intercept=_operator_intercept(operators, idx),
                        num_trajectories=identity["generation"]["num_trajectories_per_payoff"])
        game = build_game(cfg_i)
        rejected = []
        for retry in range(_EVAL_MAX_RETRIES + 1):
            seed = _eval_seed(identity["data_seed"], idx, retry)
            if seed == train_seed:
                raise ValueError(f"Evaluation seed collides with training seed for {prefix}: {seed}")
            x_cur, x_next, pair_dt, trajectories = generate_dataset(
                cfg_i, game, metric, seed=seed,
            )
            check = _initial_check(np.asarray([np.asarray(traj[0]) for traj in trajectories]), forbidden)
            if not check["train_overlap_count"] and not check["duplicate_count"]:
                break
            rejected.append({"seed": seed, "initial_conditions": check})
        else:
            raise ValueError(f"Evaluation initial-condition collision for {prefix} after deterministic retries: {rejected}")
        values, lengths = _pack_trajectories(trajectories)
        for key, value in {"x_current": x_cur, "x_next": x_next, "pair_dt": pair_dt,
                           "trajectory_values": values, "trajectory_lengths": lengths,
                           "trajectory_seed": np.asarray(seed, dtype=np.int64)}.items():
            payload[f"{prefix}/{key}"] = np.asarray(jax.device_get(value))
        environments[str(idx)] = {"seed": seed, "train_seed": train_seed, "retry_count": retry,
                                  "rejected_attempts": rejected, "initial_conditions": check}
    _validate_pool(payload, payoff_ids, cfg, identity["generation"]["num_trajectories_per_payoff"], "new evaluation")
    np.savez_compressed(eval_dir / "trajectories.npz", **payload)
    manifest = dict(identity, environments=environments, trajectory_sha256=file_sha256(eval_dir / "trajectories.npz"))
    manifest["manifest_signature"] = _signature(manifest)
    _write_json(eval_dir / "evaluation.json", manifest)


def _evaluation_split(dataset: dict, identity: dict, train_ids: list[int], test_ids: list[int]) -> dict:
    payload = _split_payload(dataset, train_ids, test_ids)
    payload.pop("split_signature")
    payload.update({key: identity[key] for key in ("evaluation_protocol", "evaluation_signature", "base_dataset_signature")})
    payload["train"]["trajectory_file"] = "trajectories.npz"
    payload["test"]["trajectory_file"] = str(Path("evaluation") / EVALUATION_PROTOCOL / identity["evaluation_signature"] / "trajectories.npz")
    payload["split_signature"] = _signature(payload)
    return payload


def _prepare_evaluation(artifact_dir: Path, dataset: dict, cfg: Config, count: int,
                        train_ids: list[int], test_ids: list[int], *, create: bool,
                        reuse_split: bool = False) -> tuple[dict, dict, dict]:
    operators = _load_npz(artifact_dir / "payoff_operators.npz")
    training = _load_npz(artifact_dir / "trajectories.npz")
    ids = _payoff_ids(operators)
    if len(ids) != int(dataset["game"]["num_payoff"]) or len(set(ids)) != len(ids):
        raise ValueError("Base payoff ids/count do not match dataset.json")
    initial = _validate_pool(training, ids, cfg, cfg.num_trajectories, "base training pool")
    _validate_operators(operators, ids, cfg)
    identity = _evaluation_identity(artifact_dir, dataset, cfg, count)
    eval_dir = _evaluation_dir(artifact_dir, identity)
    if create:
        with _cache_lock(eval_dir.parent / ".evaluation.lock"):
            for existing in eval_dir.parent.iterdir():
                if existing.name.startswith(".") or not existing.is_dir():
                    continue
                path = existing / "evaluation.json"
                if not path.exists():
                    raise ValueError(f"Incomplete evaluation cache at {existing}; refusing to overwrite it.")
                if _read_json(path).get("base_dataset_signature") != identity["base_dataset_signature"]:
                    raise ValueError(f"Base dataset changed since evaluation cache was built at {existing}")
            if not eval_dir.exists():
                # A complete directory is the publication unit; interrupted .pending-* builds
                # cannot masquerade as caches. The lock also serializes different eval counts.
                with tempfile.TemporaryDirectory(dir=eval_dir.parent, prefix=".pending-") as temporary:
                    staging = Path(temporary) / "pool"
                    staging.mkdir()
                    _build_evaluation(staging, identity, cfg, ids, operators, training, initial)
                    if _evaluation_identity(artifact_dir, dataset, cfg, count) != identity:
                        raise ValueError("Base dataset changed while generating evaluation trajectories")
                    os.rename(staging, eval_dir)
    manifest, evaluation = _validate_evaluation(eval_dir, identity, cfg, ids, training, initial)
    split = _evaluation_split(dataset, identity, train_ids, test_ids)
    split_path = eval_dir / "splits" / f"{split['split_id']}.json"
    _publish_json(split_path, split, require_existing=not create or reuse_split)
    return manifest, evaluation, split






def ensure_synthetic_dataset(
    cfg: Config,
    *,
    sampler_name: str = "cyclic-isomorphism",
    num_payoff: int = 6,
    num_payoff_train: int = 5,
    data_seed: int = 123,
    shift_seed: int | None = None,
    payoff_train_indices: list[int] | None = None,
    payoff_test_indices: list[int] | str | None = None,
    payoff_base_family: str | None = None,
    payoff_dim: int | None = None,
    payoff_base_seed: int = 123,
    reuse_existing_split_manifest: bool = False,
    evaluation_protocol: str | None = None,
    num_eval_trajectories: int | None = None,
    **kwargs: Any,
) -> Path:
    _validate_payoff_kwargs(kwargs)
    _evaluation_protocol(cfg, evaluation_protocol)
    count = _eval_count(cfg, num_eval_trajectories)
    dataset_id, cfg, base, spec, mode = _artifact_identity(
        cfg, sampler_name=sampler_name, payoff_base_family=payoff_base_family,
        payoff_dim=payoff_dim, payoff_base_seed=payoff_base_seed,
    )
    artifact_dir = _artifact_dir(dataset_id, int(data_seed))
    with _cache_lock(artifact_dir.parent / f".{artifact_dir.name}.lock"):
        if not artifact_dir.exists():
            cfg, base, spec, mode, sampler_metadata, payoff_ids, matrices, intercepts = _sample_payoff_pool(
                cfg, sampler_name=sampler_name, num_payoff=num_payoff, num_payoff_train=num_payoff_train,
                data_seed=data_seed, shift_seed=shift_seed, payoff_train_indices=payoff_train_indices,
                payoff_base_family=payoff_base_family, payoff_dim=payoff_dim,
                payoff_base_seed=payoff_base_seed, kwargs=kwargs,
            )
            _resolve_split(payoff_ids, num_payoff_train, payoff_train_indices, payoff_test_indices)
            dataset = _dataset_spec(
                dataset_id=dataset_id, data_seed=int(data_seed), cfg=cfg, base=base, spec=spec,
                mode=mode, num_payoff=int(num_payoff), sampler_metadata=sampler_metadata,
                kwargs=kwargs,
            )
            dataset["generation_request"] = _generation_request(cfg, base, spec, mode, data_seed, shift_seed, kwargs)
            dataset["dataset_signature"] = _signature({k: v for k, v in dataset.items() if k not in {"splits", "dataset_signature"}})
            with tempfile.TemporaryDirectory(dir=artifact_dir.parent, prefix=".pending-base-") as temporary:
                staging = Path(temporary) / "base"
                _write_artifact(staging, dataset, cfg, payoff_ids, matrices, intercepts)
                os.rename(staging, artifact_dir)
        dataset = _validate_artifact_schema(artifact_dir)
        _validate_artifact_dynamics(dataset, cfg, artifact_dir)
        operators = _load_npz(artifact_dir / dataset["files"]["payoff_operators"])
        payoff_ids = _payoff_ids(operators)
        train_ids, test_ids = _resolve_split(payoff_ids, num_payoff_train, payoff_train_indices, payoff_test_indices)
        _validate_base_config(
            dataset, cfg, artifact_dir, dataset_id=dataset_id, num_payoff=num_payoff,
            data_seed=data_seed, base=base, spec=spec, mode=mode,
            shift_seed=shift_seed, kwargs=kwargs,
        )
        _prepare_evaluation(artifact_dir, dataset, cfg, count, train_ids, test_ids,
                            create=True, reuse_split=reuse_existing_split_manifest)
    return artifact_dir


def load(
    cfg: Config,
    *,
    sampler_name: str = "cyclic-isomorphism",
    num_payoff: int = 6,
    num_payoff_train: int = 5,
    data_seed: int = 123,
    shift_seed: int | None = None,
    payoff_train_indices: list[int] | None = None,
    payoff_test_indices: list[int] | str | None = None,
    payoff_base_family: str | None = None,
    payoff_dim: int | None = None,
    payoff_base_seed: int = 123,
    evaluation_protocol: str | None = None,
    num_eval_trajectories: int | None = None,
    **kwargs: Any,
) -> tuple[TrainBatch, list[TestBlock], dict]:
    _validate_payoff_kwargs(kwargs)
    protocol = _evaluation_protocol(cfg, evaluation_protocol)
    count = _eval_count(cfg, num_eval_trajectories)
    dataset_id, cfg, base, spec, mode = _artifact_identity(
        cfg,
        sampler_name=sampler_name,
        payoff_base_family=payoff_base_family,
        payoff_dim=payoff_dim,
        payoff_base_seed=payoff_base_seed,
    )
    artifact_dir = _artifact_dir(dataset_id, int(data_seed))
    dataset = _validate_artifact_schema(artifact_dir)
    _validate_artifact_dynamics(dataset, cfg, artifact_dir)
    operators = _load_npz(artifact_dir / dataset["files"]["payoff_operators"])
    trajectory_payload = _load_npz(artifact_dir / dataset["files"]["trajectories"])
    payoff_ids = _payoff_ids(operators)
    train_ids, test_ids = _resolve_split(payoff_ids, num_payoff_train, payoff_train_indices, payoff_test_indices)
    _validate_base_config(
        dataset, cfg, artifact_dir, dataset_id=dataset_id, num_payoff=num_payoff,
        data_seed=data_seed, base=base, spec=spec, mode=mode,
        shift_seed=shift_seed, kwargs=kwargs,
    )
    evaluation, test_payload, split = _prepare_evaluation(
        artifact_dir, dataset, cfg, count, train_ids, test_ids, create=False,
    )
    train_ids = [int(idx) for idx in split["train"]["payoff_ids"]]
    test_ids = [int(idx) for idx in split["test"]["payoff_ids"]]
    x_parts, v_parts, next_parts, dt_parts, env_parts = [], [], [], [], []
    train_payoff_games: dict[int, Any] = {}
    for payoff_idx in train_ids:
        matrix = np.asarray(operators[f"payoff_{int(payoff_idx)}/matrix"], dtype=np.float32)
        intercept = _operator_intercept(operators, payoff_idx)
        cfg_i = replace(cfg, game_type="custom_matrix", payoff_matrix=matrix, payoff_intercept=intercept)
        game = build_game(cfg_i)
        train_payoff_games[int(payoff_idx)] = game
        x_cur, x_next, pair_dt, _ = _load_payoff_trajectory(trajectory_payload, payoff_idx)
        x_parts.append(x_cur)
        next_parts.append(x_next)
        dt_parts.append(pair_dt)
        v_parts.append(jax.vmap(game.payoff)(x_cur))
        env_parts.append(jnp.full((x_cur.shape[0],), int(payoff_idx), dtype=jnp.int32))

    test_blocks: list[TestBlock] = []
    for payoff_idx in test_ids:
        matrix = np.asarray(operators[f"payoff_{int(payoff_idx)}/matrix"], dtype=np.float32)
        intercept = _operator_intercept(operators, payoff_idx)
        cfg_i = replace(cfg, game_type="custom_matrix", payoff_matrix=matrix, payoff_intercept=intercept)
        game = build_game(cfg_i)
        x_cur, x_next, pair_dt, trajectories = _load_payoff_trajectory(test_payload, payoff_idx)
        test_blocks.append(
            TestBlock(
                x_pairs=x_cur,
                x_next=x_next,
                v_pairs=jax.vmap(game.payoff)(x_cur),
                pair_dt=pair_dt,
                env_id=int(payoff_idx),
                group_label=f"payoff_{int(payoff_idx)}",
                trajectories=[np.asarray(jax.device_get(traj)) for traj in trajectories],
                game=game,
            )
        )

    train = TrainBatch(
        x_current=jnp.concatenate(x_parts, axis=0),
        v_current=jnp.concatenate(v_parts, axis=0),
        x_next=jnp.concatenate(next_parts, axis=0),
        pair_dt=jnp.concatenate(dt_parts, axis=0),
        env_id=jnp.concatenate(env_parts, axis=0),
        payoff_games=train_payoff_games,
    )
    return train, test_blocks, {
        "dataset_id": dataset_id,
        "artifact_dir": str(artifact_dir),
        "split_id": split["split_id"],
        "evaluation_protocol": protocol,
        "evaluation_signature": evaluation["evaluation_signature"],
        "base_dataset_signature": evaluation["base_dataset_signature"],
        "split_signature": split.get("split_signature"),
        "seen_payoff_ids": [idx for idx in test_ids if idx in set(train_ids)],
        "unseen_payoff_ids": [idx for idx in test_ids if idx not in set(train_ids)],
        "num_eval_trajectories": count,
        "sampler": dataset["game"]["sampling_mode"],
        "dynamics_type": _dataset_dynamics_type(dataset),
        "payoff_train_indices": train_ids,
        "payoff_test_indices": test_ids,
        "sampler_metadata": dataset.get("sampler_metadata", {}),
    }
