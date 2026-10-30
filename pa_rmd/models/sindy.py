from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PolynomialLibrary:
    input_dim: int
    degree: int
    include_bias: bool
    combinations: tuple[tuple[int, ...], ...]
    padded_indices: jnp.ndarray
    active_mask: jnp.ndarray

    @classmethod
    def build(cls, input_dim: int, degree: int = 3, include_bias: bool = True) -> "PolynomialLibrary":
        input_dim = int(input_dim)
        degree = int(degree)
        if input_dim < 1:
            raise ValueError("SINDy polynomial library requires input_dim >= 1.")
        if degree < 1:
            raise ValueError("SINDy polynomial library requires degree >= 1.")

        terms: list[tuple[int, ...]] = [()] if include_bias else []
        for term_degree in range(1, degree + 1):
            terms.extend(combinations_with_replacement(range(input_dim), term_degree))

        width = max(degree, 1)
        padded = []
        masks = []
        for term in terms:
            indices = list(term) + [0] * (width - len(term))
            mask = [True] * len(term) + [False] * (width - len(term))
            padded.append(indices)
            masks.append(mask)

        return cls(
            input_dim=input_dim,
            degree=degree,
            include_bias=bool(include_bias),
            combinations=tuple(terms),
            padded_indices=jnp.asarray(padded, dtype=jnp.int32),
            active_mask=jnp.asarray(masks, dtype=bool),
        )

    @property
    def feature_count(self) -> int:
        return int(len(self.combinations))

    def transform_one(self, z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z)
        if z.shape[-1] != self.input_dim:
            raise ValueError(f"SINDy input has dim {z.shape[-1]}, expected {self.input_dim}.")
        gathered = z[self.padded_indices]
        factors = jnp.where(self.active_mask, gathered, jnp.ones_like(gathered))
        return jnp.prod(factors, axis=-1)

    def transform(self, z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z)
        if z.ndim == 1:
            return self.transform_one(z)
        return jax.vmap(self.transform_one)(z)


@dataclass(frozen=True)
class SindyFitResult:
    library: PolynomialLibrary
    coefficients: jnp.ndarray
    losses: tuple[float, ...]
    threshold: float
    ridge: float


def _ridge_solve(theta: jnp.ndarray, target: jnp.ndarray, ridge: float) -> jnp.ndarray:
    n_features = int(theta.shape[1])
    gram = theta.T @ theta
    rhs = theta.T @ target
    penalty = float(ridge) * jnp.eye(n_features, dtype=theta.dtype)
    return jnp.linalg.solve(gram + penalty, rhs)


def _masked_ridge_solve(
    theta: jnp.ndarray,
    target: jnp.ndarray,
    mask: jnp.ndarray,
    ridge: float,
) -> jnp.ndarray:
    active = jnp.nonzero(mask, size=int(mask.shape[0]), fill_value=-1)[0]
    active = active[active >= 0]
    if int(active.shape[0]) == 0:
        return jnp.zeros((theta.shape[1],), dtype=theta.dtype)
    theta_active = theta[:, active]
    coeff_active = _ridge_solve(theta_active, target, ridge).reshape(-1)
    coeff = jnp.zeros((theta.shape[1],), dtype=theta.dtype)
    return coeff.at[active].set(coeff_active)


def _mse(theta: jnp.ndarray, coefficients: jnp.ndarray, target: jnp.ndarray) -> float:
    pred = theta @ coefficients
    return float(jnp.mean(jnp.sum(jnp.square(pred - target), axis=-1)))


def fit_sindy(
    x_current: jnp.ndarray,
    v_current: jnp.ndarray,
    x_next: jnp.ndarray,
    pair_dt: jnp.ndarray,
    *,
    degree: int = 3,
    threshold: float = 1e-3,
    ridge: float = 1e-6,
    max_iter: int = 10,
    include_bias: bool = True,
) -> SindyFitResult:
    x_current = jnp.asarray(x_current, dtype=jnp.float32)
    v_current = jnp.asarray(v_current, dtype=jnp.float32)
    x_next = jnp.asarray(x_next, dtype=jnp.float32)
    pair_dt = jnp.asarray(pair_dt, dtype=jnp.float32).reshape(-1)

    if x_current.ndim != 2 or v_current.ndim != 2 or x_next.ndim != 2:
        raise ValueError("SINDy expects 2D x_current, v_current, and x_next arrays.")
    if x_current.shape != v_current.shape or x_current.shape != x_next.shape:
        raise ValueError(
            "SINDy expects x_current, v_current, and x_next to have identical shapes; "
            f"got {x_current.shape}, {v_current.shape}, {x_next.shape}."
        )
    if pair_dt.shape[0] != x_current.shape[0]:
        raise ValueError(f"SINDy pair_dt length {pair_dt.shape[0]} does not match batch size {x_current.shape[0]}.")

    z = jnp.concatenate([x_current, v_current], axis=-1)
    library = PolynomialLibrary.build(z.shape[-1], degree=degree, include_bias=include_bias)
    theta = library.transform(z)
    target = (x_next - x_current) / jnp.maximum(pair_dt[:, None], 1e-12)
    target = target - jnp.mean(target, axis=-1, keepdims=True)

    coefficients = _ridge_solve(theta, target, ridge)
    losses = [_mse(theta, coefficients, target)]
    threshold = float(threshold)

    for _ in range(int(max_iter)):
        if threshold > 0.0:
            coefficients = jnp.where(jnp.abs(coefficients) < threshold, 0.0, coefficients)
        columns = []
        for output_dim in range(int(target.shape[1])):
            active = jnp.abs(coefficients[:, output_dim]) > 0.0
            columns.append(_masked_ridge_solve(theta, target[:, output_dim], active, ridge))
        coefficients = jnp.stack(columns, axis=1)
        losses.append(_mse(theta, coefficients, target))

    if threshold > 0.0:
        coefficients = jnp.where(jnp.abs(coefficients) < threshold, 0.0, coefficients)
        losses.append(_mse(theta, coefficients, target))

    return SindyFitResult(
        library=library,
        coefficients=coefficients,
        losses=tuple(losses),
        threshold=threshold,
        ridge=float(ridge),
    )


def sindy_vector_field(library: PolynomialLibrary, coefficients: jnp.ndarray, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    z = jnp.concatenate([jnp.asarray(x), jnp.asarray(v)], axis=-1)
    features = library.transform(z)
    field = features @ coefficients
    return field - jnp.mean(field, axis=-1, keepdims=True)

from typing import Any
import jax
import jax.numpy as jnp
from pa_rmd.models.base import _BasePredictor
from pa_rmd.utils import project_simplex


class SindyPredictor(_BasePredictor):
    label = "SINDy"
    key = "sindy"
    color = "black"

    def fit(self, batch: Any, *, seed: int) -> None:
        del seed
        cfg = self._cfg()
        self._fit_cfg = cfg
        self._set_v_scale(batch, cfg)
        result = fit_sindy(
            batch.x_current,
            self._scale_v(batch.v_current),
            batch.x_next,
            batch.pair_dt,
            degree=int(getattr(cfg, "sindy_degree", 3)),
            threshold=float(getattr(cfg, "sindy_threshold", 1e-3)),
            ridge=float(getattr(cfg, "sindy_ridge", 1e-6)),
            max_iter=int(getattr(cfg, "sindy_max_iter", 10)),
            include_bias=True,
        )
        self.model = result.library
        self.params = {"coefficients": result.coefficients}
        nonzero = int(jnp.sum(jnp.abs(result.coefficients) > 0.0))
        total = int(result.coefficients.size)
        counts = {
            "params_active": nonzero,
            "params_shared": nonzero,
            "params_env_active": 0,
            "params_total_stored": total,
            "features": int(result.library.feature_count),
        }
        self._history = {"loss": list(result.losses), "parameter_counts": counts}
        self._rollout_batch_cache.clear()
        print(
            f"[sindy] features={result.library.feature_count} nonzero={nonzero}/{total} "
            f"final_loss={result.losses[-1]:.6f}",
            flush=True,
        )

    def predict_step(self, x: jnp.ndarray, v: jnp.ndarray, dt: float | jnp.ndarray, env_id: int | jnp.ndarray | None = None) -> jnp.ndarray:
        del env_id
        self._require_fit()
        x = jnp.asarray(x)
        v = self._scale_v(v)
        dt = self._batch_dt(x, dt)
        field = sindy_vector_field(self.model, self.params["coefficients"], x, v)
        return jax.vmap(project_simplex)(x + dt[:, None] * field)

    def config_summary(self) -> dict:
        summary = super().config_summary()
        if self.model is not None:
            summary["sindy_feature_count"] = int(self.model.feature_count)
        return summary

