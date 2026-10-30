from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from pa_rmd.games import PopulationGame


def _center(x: jnp.ndarray) -> jnp.ndarray:
    return x - jnp.mean(x, axis=-1, keepdims=True)


@dataclass
class TargetKernelPayoffGame(PopulationGame):
    support_x: jnp.ndarray
    support_v: jnp.ndarray
    bandwidth: float

    def payoff(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=jnp.float32)
        sq_dist = jnp.sum((self.support_x - x[None, :]) ** 2, axis=-1)
        denom = 2.0 * max(float(self.bandwidth) ** 2, 1e-12)
        weights = jax.nn.softmax(-sq_dist / denom)
        return _center(weights @ self.support_v)

    @property
    def N(self) -> int:
        return int(self.support_x.shape[1])


def _median_bandwidth(x: jnp.ndarray) -> float:
    n_data = int(x.shape[0])
    if n_data <= 1:
        return 1.0
    diffs = x[:, None, :] - x[None, :, :]
    dists = jnp.sqrt(jnp.maximum(jnp.sum(diffs**2, axis=-1), 0.0))
    tri = dists[jnp.triu_indices(n_data, k=1)]
    value = float(jnp.median(tri))
    return value if value > 1e-6 else 1.0


def fit_target_kernel_payoff_game(
    x_support: jnp.ndarray,
    v_support: jnp.ndarray,
    *,
    bandwidth: float | None = None,
) -> TargetKernelPayoffGame:
    x_support = jnp.asarray(x_support, dtype=jnp.float32)
    v_support = jnp.asarray(v_support, dtype=jnp.float32)
    if x_support.ndim != 2 or v_support.ndim != 2:
        raise ValueError("payoff support arrays must be rank-2.")
    if x_support.shape != v_support.shape:
        raise ValueError(
            f"payoff support x/v shapes must match; got {tuple(x_support.shape)} and {tuple(v_support.shape)}."
        )
    if int(x_support.shape[0]) < 1:
        raise ValueError("target payoff support must contain at least one observation.")
    bw = _median_bandwidth(x_support) if bandwidth is None else float(bandwidth)
    return TargetKernelPayoffGame(support_x=x_support, support_v=v_support, bandwidth=bw)


def payoff_metrics(pred: jnp.ndarray, target: jnp.ndarray) -> dict[str, float]:
    errors = jnp.linalg.norm(pred - target, axis=-1)
    sq_errors = jnp.sum((pred - target) ** 2, axis=-1)
    return {
        "payoff_l2_mean": float(jnp.mean(errors)),
        "payoff_l2_std": float(jnp.std(errors)),
        "payoff_mse_mean": float(jnp.mean(sq_errors)),
        "payoff_mse_std": float(jnp.std(sq_errors)),
    }
