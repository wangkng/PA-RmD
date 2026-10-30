from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pa_rmd.games import PopulationGame
from pa_rmd.payoff_closure import fit_target_kernel_payoff_game, payoff_metrics


@dataclass
class TrainBatch:
    x_current: jnp.ndarray
    v_current: jnp.ndarray
    x_next: jnp.ndarray
    pair_dt: jnp.ndarray
    env_id: jnp.ndarray | None = None
    payoff_games: dict[int, PopulationGame] | None = None


@dataclass
class TestBlock:
    x_pairs: jnp.ndarray
    x_next: jnp.ndarray
    v_pairs: jnp.ndarray
    pair_dt: jnp.ndarray
    env_id: int
    group_label: str
    trajectories: list[np.ndarray] | None = None
    game: PopulationGame | None = None
    payoff_support: TrainBatch | None = None
    observed_x_pairs: jnp.ndarray | None = None
    observed_x_next: jnp.ndarray | None = None
    observed_v_pairs: jnp.ndarray | None = None
    observed_pair_dt: jnp.ndarray | None = None


def _step_metrics(pred: jnp.ndarray, target: jnp.ndarray) -> dict[str, float]:
    errors = jnp.linalg.norm(pred - target, axis=-1)
    sq_errors = jnp.sum((pred - target) ** 2, axis=-1)

    def normalize_prob(x: jnp.ndarray) -> jnp.ndarray:
        clipped = jnp.clip(x, 0.0)
        mass = jnp.sum(clipped, axis=-1, keepdims=True)
        uniform = jnp.ones_like(clipped) / clipped.shape[-1]
        return jnp.where(mass > 0.0, clipped / jnp.maximum(mass, 1e-12), uniform)

    p = normalize_prob(pred)
    q = normalize_prob(target)
    m = 0.5 * (p + q)
    eps = 1e-12
    js_values = 0.5 * jnp.sum(p * (jnp.log(p + eps) - jnp.log(m + eps)), axis=-1)
    js_values += 0.5 * jnp.sum(q * (jnp.log(q + eps) - jnp.log(m + eps)), axis=-1)
    return {
        "one_step_mean": float(jnp.mean(errors)),
        "one_step_std": float(jnp.std(errors)),
        "one_step_mse_mean": float(jnp.mean(sq_errors)),
        "one_step_mse_std": float(jnp.std(sq_errors)),
        "one_step_js_mean": float(jnp.mean(js_values)),
        "one_step_js_std": float(jnp.std(js_values)),
    }


def _rollout_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    aligned = np.asarray(pred, dtype=np.float64)
    target = np.asarray(true, dtype=np.float64)
    if aligned.shape != target.shape or target.ndim != 2 or target.shape[0] < 2:
        raise ValueError("Rollout predictions and targets must have identical (states, dimensions) shapes.")
    sq = np.sum((aligned - target) ** 2, axis=-1)
    terminal = np.sqrt(sq[-1]) if sq.size else np.nan
    return {
        "rollout_mse_mean": float(np.mean(sq)),
        "rollout_mse_std": float(np.std(sq)),
        "rollout_at_T_mean": float(terminal),
        "rollout_at_T_std": 0.0,
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row if k not in {"env_id", "group_label"}})
    out: dict[str, float] = {}
    for key in keys:
        vals = np.asarray([row[key] for row in rows if key in row], dtype=np.float64)
        if vals.size:
            out[key] = float(vals.mean())
    return out


def _rollout_block_predictions(
    predictor: Any,
    block: TestBlock,
    game: PopulationGame,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    if block.trajectories is None:
        return {}, []

    grouped: dict[int, list[tuple[int, np.ndarray]]] = {}
    for i, true_traj in enumerate(block.trajectories):
        true_arr = np.asarray(true_traj)
        if true_arr.ndim != 2 or true_arr.shape[0] < 2:
            raise ValueError("Each rollout trajectory must contain at least two states.")
        grouped.setdefault(int(true_arr.shape[0]), []).append((i, true_arr))

    predictions: dict[str, np.ndarray] = {}
    rollout_rows: list[dict[str, float]] = []
    for traj_len, items in grouped.items():
        x0_batch = jnp.asarray(np.stack([true_arr[0] for _, true_arr in items], axis=0))
        if block.pair_dt.shape[0] < traj_len - 1:
            raise ValueError("Rollout time intervals do not cover the trajectory.")
        dts = jnp.asarray(block.pair_dt[: traj_len - 1])
        pred_batch = predictor.predict_rollout_batch(x0_batch, dts, game, env_id=block.env_id)
        pred_batch_arr = np.asarray(jax.device_get(pred_batch))
        for batch_idx, (traj_id, true_arr) in enumerate(items):
            pred_arr = pred_batch_arr[batch_idx]
            predictions[f"ground_truth/{block.group_label}/rollout/{traj_id}"] = true_arr
            predictions[f"{block.group_label}/rollout/{traj_id}"] = pred_arr
            rollout_rows.append(_rollout_metrics(pred_arr, true_arr))

    return predictions, rollout_rows


def _learned_v_step_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key.replace("one_step", "one_step_learned_v", 1): value for key, value in metrics.items()}


def _predict_step(
    predictor: Any,
    x: jnp.ndarray,
    v: jnp.ndarray,
    dt: jnp.ndarray,
    *,
    env_id: int,
    game: PopulationGame | None,
) -> jnp.ndarray:
    predict_with_game = getattr(predictor, "predict_step_with_game", None)
    if predict_with_game is not None:
        return predict_with_game(x, v, dt, game=game, env_id=env_id)
    return predictor.predict_step(x, v, dt, env_id=env_id)


def train_and_evaluate(
    predictor: Any,
    train: TrainBatch,
    test_blocks: list[TestBlock],
    *,
    seed: int,
) -> dict[str, Any]:
    label = getattr(predictor, "key", predictor.__class__.__name__)
    total_t0 = time.time()
    fit_t0 = time.time()
    predictor.fit(train, seed=seed)
    fit_time = time.time() - fit_t0
    print(f"[timing][{label}] fit={fit_time:.1f}s", flush=True)
    return evaluate_fitted(predictor, train, test_blocks, seed=seed, fit_seconds=fit_time,
                           total_started_at=total_t0)


def evaluate_fitted(
    predictor: Any,
    train: TrainBatch,
    test_blocks: list[TestBlock],
    *,
    seed: int,
    fit_seconds: float = 0.0,
    total_started_at: float | None = None,
) -> dict[str, Any]:
    """Evaluate an already fitted predictor on the supplied test observations."""
    label = getattr(predictor, "key", predictor.__class__.__name__)
    total_t0 = time.time()
    fit_time = float(fit_seconds)
    block_rows: list[dict[str, float]] = []
    predictions: dict[str, np.ndarray] = {}
    one_step_time = 0.0
    rollout_time = 0.0

    for block in test_blocks:
        block_payoff_game = None
        if block.payoff_support is not None:
            block_payoff_game = fit_target_kernel_payoff_game(
                block.payoff_support.x_current,
                block.payoff_support.v_current,
            )
        rollout_game = block_payoff_game if block_payoff_game is not None else block.game
        if block.trajectories is not None and rollout_game is None:
            raise ValueError(
                f"Rollout for {block.group_label!r} requires target payoff support or a known game."
            )

        step_t0 = time.time()
        observed_x_pairs = block.observed_x_pairs if block.observed_x_pairs is not None else block.x_pairs
        observed_x_next = block.observed_x_next if block.observed_x_next is not None else block.x_next
        observed_v_pairs = block.observed_v_pairs if block.observed_v_pairs is not None else block.v_pairs
        observed_pair_dt = block.observed_pair_dt if block.observed_pair_dt is not None else block.pair_dt
        pred = _predict_step(
            predictor,
            observed_x_pairs,
            observed_v_pairs,
            observed_pair_dt,
            env_id=block.env_id,
            game=block.game,
        )
        predictions[f"ground_truth/{block.group_label}/one_step/x_current"] = np.asarray(jax.device_get(observed_x_pairs))
        predictions[f"ground_truth/{block.group_label}/one_step/x_next"] = np.asarray(jax.device_get(observed_x_next))
        predictions[f"{block.group_label}/one_step"] = np.asarray(jax.device_get(pred))
        row = {"env_id": float(block.env_id), "group_label": block.group_label, **_step_metrics(pred, observed_x_next)}
        payoff_domain_metrics = getattr(predictor, "payoff_domain_metrics", None)
        if payoff_domain_metrics is not None:
            row.update(payoff_domain_metrics(block.v_pairs))
        if block_payoff_game is not None:
            v_hat = jax.vmap(block_payoff_game.payoff)(block.x_pairs)
            pred_learned_v = _predict_step(
                predictor,
                block.x_pairs,
                v_hat,
                block.pair_dt,
                env_id=block.env_id,
                game=block_payoff_game,
            )
            predictions[f"ground_truth/{block.group_label}/payoff/v_observed"] = np.asarray(jax.device_get(block.v_pairs))
            predictions[f"{block.group_label}/payoff/v_learned"] = np.asarray(jax.device_get(v_hat))
            predictions[f"{block.group_label}/one_step_learned_v"] = np.asarray(jax.device_get(pred_learned_v))
            row.update(payoff_metrics(v_hat, block.v_pairs))
            row.update(_learned_v_step_metrics(_step_metrics(pred_learned_v, block.x_next)))
        one_step_time += time.time() - step_t0

        if block.trajectories is not None and rollout_game is not None:
            rollout_t0 = time.time()
            rollout_predictions, rollout_rows = _rollout_block_predictions(predictor, block, rollout_game)
            predictions.update(rollout_predictions)
            if rollout_rows:
                row.update(_aggregate(rollout_rows))
            rollout_time += time.time() - rollout_t0

        block_rows.append(row)

    total_time = (time.time() - total_started_at if total_started_at is not None
                  else fit_time + time.time() - total_t0)
    print(
        f"[timing][{label}] one_step={one_step_time:.1f}s "
        f"rollout={rollout_time:.1f}s total_fit_eval={total_time:.1f}s",
        flush=True,
    )

    predictor_config = predictor.config_summary()
    predictor_config["payoff_closure"] = {
        "target_support_transitions": {
            str(block.env_id): int(block.payoff_support.x_current.shape[0])
            for block in test_blocks if block.payoff_support is not None
        },
        "selection_order": ["target-support", "block-game"],
    }
    return {
        "predictor": predictor.key,
        "label": predictor.label,
        "blocks": block_rows,
        "metrics": _aggregate(block_rows),
        "history": predictor.history(),
        "training_details": getattr(predictor, "_history", {}),
        "parameter_count": predictor.parameter_count(),
        "config": predictor_config,
        "predictions": predictions,
        "timings": {
            "fit_seconds": fit_time,
            "one_step_seconds": one_step_time,
            "rollout_seconds": rollout_time,
            "total_seconds": total_time,
        },
    }
