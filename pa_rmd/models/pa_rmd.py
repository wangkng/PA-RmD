"""PA-RmD tangent transport, quadratic energy, fitting and prediction."""
from __future__ import annotations

import copy
from typing import Any, Dict

import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np

from pa_rmd.models.base import _BasePredictor
from pa_rmd.models.components import _mlp_init, _mlp_apply
from pa_rmd.config import Config
from pa_rmd.utils import project_simplex


def _compute_tangent_basis(N: int) -> jnp.ndarray:
    """Return an orthonormal basis U in R^{N x (N-1)} for {z : 1^T z = 0}."""
    ones = np.ones((N, 1), dtype=np.float32) / np.sqrt(float(N))
    Q, _ = np.linalg.qr(np.hstack([ones, np.eye(N, dtype=np.float32)]))
    return jnp.array(Q[:, 1:], dtype=jnp.float32)


class TangentCholeskyMLP:
    """Learn C_theta(x) in S_{++}^{N-1} and lift it to the simplex tangent space."""

    def __init__(
        self,
        N: int,
        hidden_dim: int,
        num_layers: int,
        init_transport_scale: float = 1.0,
        diag_floor: float = 1e-3,
    ):
        self.N = N
        self.d = N - 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.init_transport_scale = float(init_transport_scale)
        # Transport positivity uses the additive ridge below.
        self.diag_floor = float(diag_floor)
        self.ridge_epsilon = 1e-4
        self.out_dim = self.d * (self.d + 1) // 2
        self.tril_idx = jnp.tril_indices(self.d)
        self.U = _compute_tangent_basis(N)

    def init_params(self, rng_key: jax.Array) -> Dict:
        params = _mlp_init(
            rng_key,
            self.N,
            self.out_dim,
            self.hidden_dim,
            self.num_layers,
            small_output=True,
        )
        target_variance = max(self.init_transport_scale - self.ridge_epsilon, 1e-6)
        target_diag = float(np.sqrt(target_variance))
        diag_bias = float(jnp.log(jnp.exp(jnp.array(target_diag)) - 1.0))
        rows, cols = jnp.tril_indices(self.d)
        is_diag = jnp.array(rows == cols)
        last_layer = f"layer_{self.num_layers}"
        b = params[last_layer]["b"]
        b = b + jnp.where(is_diag, diag_bias, 0.0)
        params[last_layer] = {**params[last_layer], "b": b}
        return params

    def C(self, params: Dict, x: jnp.ndarray) -> jnp.ndarray:
        entries = _mlp_apply(params, x, self.num_layers, self.out_dim)
        L = jnp.zeros((self.d, self.d), dtype=entries.dtype)
        L = L.at[self.tril_idx].set(entries)
        diag_vals = jnn.softplus(jnp.diag(L))
        L = L.at[jnp.diag_indices(self.d)].set(diag_vals)
        eye = jnp.eye(self.d, dtype=entries.dtype)
        return L @ L.T + self.ridge_epsilon * eye

    def transport(self, params: Dict, x: jnp.ndarray) -> jnp.ndarray:
        C = self.C(params, x)
        return self.U @ C @ self.U.T

    def apply_transport(self, params: Dict, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return self.transport(params, x) @ v


class QuadraticEnergyFromTangentTransport:
    """Quadratic displacement energy induced by a tangent transport C_theta(x)."""

    def __init__(self, tangent_model: TangentCholeskyMLP, tau: float):
        self.tangent_model = tangent_model
        self.tau = tau

    def init_params(self, rng_key: jax.Array) -> Dict[str, Any]:
        return self.tangent_model.init_params(rng_key)


    def closed_form_displacement(
        self,
        params: Dict,
        x: jnp.ndarray,
        v: jnp.ndarray,
        step_scale: float,
    ) -> jnp.ndarray:
        z_hat = step_scale * self.tangent_model.apply_transport(params, x, v)
        return z_hat - jnp.mean(z_hat)


_MODEL_DEFAULTS = {
    "model_type": "tangent_cholesky_mlp", "solver_type": "closed_form",
    "energy_model_type": "quadratic_from_tangent_transport",
    "evaluation_protocol": "independent-ic-v1",
    "num_eval_trajectories": None, "payoff_input_normalization": "none",
}


class PARmD(_BasePredictor):
    """Fit payoff-conditioned tangent transport and predict simplex states."""

    label = "RmD_C"
    key = "rmd_c"
    color = "seagreen"
    model_type = "tangent_cholesky_mlp"
    energy_model_type = "quadratic_from_tangent_transport"

    def __init__(self, shared_cfg: Config, **kwargs: Any):
        self.shared_cfg = shared_cfg
        self.kwargs = kwargs
        self.params = None
        self.model = None
        self._history: dict[str, Any] = {}
        self._fit_cfg = shared_cfg
        self._rollout_batch_cache: dict[tuple, Any] = {}
        self._v_scale = 1.0
        self._v_scale_mode = "none"

    def _cfg(self, **overrides: Any) -> Config:
        cfg = copy.copy(self.shared_cfg)
        for key, value in _MODEL_DEFAULTS.items():
            if not hasattr(cfg, key):
                setattr(cfg, key, value)
        for key, value in {**self.kwargs, **overrides}.items():
            setattr(cfg, key, value)
        cfg.__post_init__()
        cfg.tau = cfg.observed_delta_t()
        return cfg

    def fit(self, batch: Any, *, seed: int) -> None:
        from pa_rmd.train import train

        cfg = self._cfg(model_type=self.model_type, energy_model_type=self.energy_model_type)
        self._fit_seed = int(seed)
        self._fit_cfg = cfg
        self._set_v_scale(batch, cfg)
        self.params, self.model, self._history = train(
            cfg, batch.x_current, self._scale_v(batch.v_current), batch.x_next, batch.pair_dt, seed=seed
        )
        self._rollout_batch_cache.clear()

    def predict_step(self, x: jnp.ndarray, v: jnp.ndarray, dt: float | jnp.ndarray, env_id: int | jnp.ndarray | None = None) -> jnp.ndarray:
        del env_id
        self._require_fit()
        x, v, dt = jnp.asarray(x), self._scale_v(v), self._batch_dt(jnp.asarray(x), dt)

        def one(x_t, v_t, step_dt):
            z_hat = self.model.closed_form_displacement(self.params, x_t, v_t, step_dt)
            return project_simplex(x_t + z_hat)

        return jax.vmap(one)(x, v, dt)


    def _make_rollout_batch_fn(self, *, is_affine: bool):
        if is_affine:
            def payoff_batch(x_batch, payoff_matrix, payoff_intercept):
                return x_batch @ payoff_matrix.T + payoff_intercept

            def rollout_impl(x_init, step_dts, payoff_matrix, payoff_intercept):
                def scan_step(x_batch, step_dt):
                    v_batch = payoff_batch(x_batch, payoff_matrix, payoff_intercept)
                    dt_batch = jnp.full((x_batch.shape[0],), step_dt, dtype=x_batch.dtype)
                    x_next = self.predict_step(x_batch, v_batch, dt_batch, env_id=None)
                    return x_next, x_next

                _, pred_steps = jax.lax.scan(scan_step, x_init, step_dts)
                return jnp.swapaxes(jnp.concatenate([x_init[None, :, :], pred_steps], axis=0), 0, 1)

            return jax.jit(rollout_impl)

        def rollout_impl(x_init, step_dts, payoff_fn):
            def scan_step(x_batch, step_dt):
                v_batch = jax.vmap(payoff_fn)(x_batch)
                dt_batch = jnp.full((x_batch.shape[0],), step_dt, dtype=x_batch.dtype)
                x_next = self.predict_step(x_batch, v_batch, dt_batch, env_id=None)
                return x_next, x_next

            _, pred_steps = jax.lax.scan(scan_step, x_init, step_dts)
            return jnp.swapaxes(jnp.concatenate([x_init[None, :, :], pred_steps], axis=0), 0, 1)

        return jax.jit(rollout_impl, static_argnums=(2,))

    def predict_rollout_batch(
        self,
        x0_batch: jnp.ndarray,
        dts: jnp.ndarray,
        game: Any,
        env_id: int | None = None,
    ) -> jnp.ndarray:
        x0_batch = jnp.asarray(x0_batch)
        dts = jnp.asarray(dts)
        matrix = getattr(game, "matrix", None)
        intercept = getattr(game, "intercept", None)
        is_affine = matrix is not None and intercept is not None
        static_game_key = None if is_affine else id(game)
        cache_key = (
            "affine" if is_affine else "generic",
            static_game_key,
            int(x0_batch.shape[0]),
            int(x0_batch.shape[1]),
            int(dts.shape[0]),
        )
        rollout_fn = self._rollout_batch_cache.get(cache_key)
        if rollout_fn is None:
            rollout_fn = self._make_rollout_batch_fn(is_affine=is_affine)
            self._rollout_batch_cache[cache_key] = rollout_fn

        if is_affine:
            return rollout_fn(x0_batch, dts, jnp.asarray(matrix), jnp.asarray(intercept))
        return rollout_fn(x0_batch, dts, game.payoff)
