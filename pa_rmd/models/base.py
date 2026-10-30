"""Shared fitting, configuration and rollout interface for predictors."""
from __future__ import annotations
import copy
from typing import Any, Protocol
import jax
import jax.numpy as jnp
import numpy as np
from pa_rmd.config import Config


_PREDICTOR_DEFAULTS = {
    "model_type": "cholesky_mlp", "solver_type": "closed_form",
    "energy_model_type": "quadratic_from_metric",
    "baseline_model_type": "neural_ode", "ode_solver_substeps": 1, "baseline_n_env": 1,
    "leads_shared_hidden_dim": 56, "leads_residual_hidden_dim": 24,
    "leads_regularization_lambda": 2e-4, "leads_spectral_weight": 1e-2, "leads_norm_epsilon": 1e-5,
    "coda_regularization_type": "l2", "coda_regularization_lambda": 1e-4,
    "coda_context_dim": 2, "coda_hyper_regularization_lambda": 1e-5,
    "coda_context_regularization_lambda": 1e-4, "coda_norm_epsilon": 1e-8,
    "evaluation_protocol": "independent-ic-v1",
    "num_eval_trajectories": None,
    "payoff_input_normalization": "none",
    "sindy_degree": 3, "sindy_threshold": 1e-3, "sindy_ridge": 1e-6, "sindy_max_iter": 10,
    "siar_degree": 3, "siar_coefficient_regularization": 1e-9,
    "siar_payoff_domain_margin": 1.25, "siar_enforce_forward_invariance": True,
    "siar_enforce_positive_correlation": True, "siar_solver": "CLARABEL",
    "siar_solver_eps": 1e-6,
    "siar_solver_max_iters": 500_000,
    "siar_solver_use_indirect": False,
}


def _json_safe(value: Any) -> Any:
    """Convert configuration scalars/arrays without losing their actual values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return _json_safe(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


class Predictor(Protocol):
    label: str
    key: str
    color: str

    def fit(self, batch: Any, *, seed: int) -> None: ...
    def predict_step(self, x: jnp.ndarray, v: jnp.ndarray, dt: float | jnp.ndarray, env_id: int | jnp.ndarray | None = None) -> jnp.ndarray: ...
    def predict_rollout(self, x0: jnp.ndarray, dts: jnp.ndarray, game: Any, env_id: int | None = None) -> jnp.ndarray: ...
    def predict_rollout_batch(self, x0: jnp.ndarray, dts: jnp.ndarray, game: Any, env_id: int | None = None) -> jnp.ndarray: ...
    def history(self) -> list[float]: ...
    def parameter_count(self) -> dict | None: ...
    def config_summary(self) -> dict: ...


class _BasePredictor:
    label = ""
    key = ""
    color = "black"

    def __init__(self, shared_cfg: Config, **kwargs: Any):
        self.shared_cfg = shared_cfg
        self.kwargs = kwargs
        self.params = None
        self.model = None
        self._history: dict[str, Any] = {}
        self._fit_cfg = shared_cfg
        self._env_id_map: dict[int, int] = {0: 0}
        self._rollout_batch_cache: dict[tuple, Any] = {}
        self._v_scale = 1.0
        self._v_scale_mode = "none"

    def _cfg(self, **overrides: Any) -> Config:
        cfg = copy.copy(self.shared_cfg)
        for key, value in _PREDICTOR_DEFAULTS.items():
            if not hasattr(cfg, key):
                setattr(cfg, key, value)
        for key, value in {**self.kwargs, **overrides}.items():
            setattr(cfg, key, value)
        cfg.__post_init__()
        cfg.tau = cfg.observed_delta_t()
        return cfg

    def _require_fit(self) -> None:
        if self.params is None or self.model is None:
            raise RuntimeError(f"{self.label} must be fit before prediction")

    @staticmethod
    def _batch_dt(x: jnp.ndarray, dt: float | jnp.ndarray) -> jnp.ndarray:
        dt = jnp.asarray(dt)
        return jnp.full((x.shape[0],), float(dt), dtype=x.dtype) if dt.ndim == 0 else dt

    def _set_v_scale(self, batch: Any, cfg: Config) -> None:
        mode = str(getattr(cfg, "payoff_input_normalization", "none") or "none").lower().replace("-", "_")
        values = np.asarray(batch.v_current, dtype=np.float64)
        if mode in {"none", "off", "false", "0"}:
            scale = 1.0
        elif mode in {"std", "standard_deviation"}:
            scale = float(np.nanstd(values))
        elif mode in {"max_abs", "maxabs", "linf"}:
            scale = float(np.nanmax(np.abs(values)))
        else:
            try:
                scale = float(mode)
            except ValueError as exc:
                raise ValueError(f"Unknown payoff_input_normalization={mode!r}") from exc
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        self._v_scale = float(scale)
        self._v_scale_mode = mode

    def _scale_v(self, v: Any) -> jnp.ndarray:
        return jnp.asarray(v) / float(self._v_scale)

    def _rollout_from_payoff(self, x0: jnp.ndarray, dts: jnp.ndarray, game: Any, env_id: int | None = None) -> jnp.ndarray:
        xs = [jnp.asarray(x0)]
        x = xs[0]
        for step_dt in jnp.asarray(dts):
            x = self.predict_step(x[None, :], game.payoff(x)[None, :], step_dt, env_id=env_id)[0]
            xs.append(x)
        return jnp.stack(xs, axis=0)

    def predict_rollout(self, x0: jnp.ndarray, dts: jnp.ndarray, game: Any, env_id: int | None = None) -> jnp.ndarray:
        return self._rollout_from_payoff(x0, dts, game, env_id=env_id)

    def predict_rollout_batch(self, x0: jnp.ndarray, dts: jnp.ndarray, game: Any, env_id: int | None = None) -> jnp.ndarray:
        return self._rollout_batch_from_payoff(x0, dts, game, env_id=env_id)

    def _make_rollout_batch_fn(self, *, is_affine: bool, static_env_id: int | None):
        if is_affine:
            def payoff_batch(x_batch, payoff_matrix, payoff_intercept):
                return x_batch @ payoff_matrix.T + payoff_intercept

            def rollout_impl(x_init, step_dts, payoff_matrix, payoff_intercept):
                def scan_step(x_batch, step_dt):
                    v_batch = payoff_batch(x_batch, payoff_matrix, payoff_intercept)
                    dt_batch = jnp.full((x_batch.shape[0],), step_dt, dtype=x_batch.dtype)
                    x_next = self.predict_step(x_batch, v_batch, dt_batch, env_id=static_env_id)
                    return x_next, x_next

                _, pred_steps = jax.lax.scan(scan_step, x_init, step_dts)
                return jnp.swapaxes(jnp.concatenate([x_init[None, :, :], pred_steps], axis=0), 0, 1)

            return jax.jit(rollout_impl)

        def rollout_impl(x_init, step_dts, payoff_fn):
            def scan_step(x_batch, step_dt):
                v_batch = jax.vmap(payoff_fn)(x_batch)
                dt_batch = jnp.full((x_batch.shape[0],), step_dt, dtype=x_batch.dtype)
                x_next = self.predict_step(x_batch, v_batch, dt_batch, env_id=static_env_id)
                return x_next, x_next

            _, pred_steps = jax.lax.scan(scan_step, x_init, step_dts)
            return jnp.swapaxes(jnp.concatenate([x_init[None, :, :], pred_steps], axis=0), 0, 1)

        return jax.jit(rollout_impl, static_argnums=(2,))

    def _rollout_batch_from_payoff(
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
        uses_env_id = bool(getattr(self, "use_env_id", False))
        static_env_id = int(env_id) if uses_env_id and env_id is not None else None
        cache_key = (
            "affine" if is_affine else "generic",
            static_game_key,
            static_env_id,
            int(x0_batch.shape[0]),
            int(x0_batch.shape[1]),
            int(dts.shape[0]),
        )
        rollout_fn = self._rollout_batch_cache.get(cache_key)
        if rollout_fn is None:
            rollout_fn = self._make_rollout_batch_fn(is_affine=is_affine, static_env_id=static_env_id)
            self._rollout_batch_cache[cache_key] = rollout_fn

        if is_affine:
            return rollout_fn(x0_batch, dts, jnp.asarray(matrix), jnp.asarray(intercept))
        return rollout_fn(x0_batch, dts, game.payoff)

    def history(self) -> list[float]:
        return list(self._history.get("loss", []))

    def parameter_count(self) -> dict | None:
        return self._history.get("parameter_counts")

    def input_config_summary(self) -> dict:
        """Resolve current constructor inputs without consulting fitted state."""
        fresh = type(self)(copy.copy(self.shared_cfg), **self.kwargs)
        summary = fresh.config_summary()
        # Explicitly spelling an already-resolved default is not a new model.
        fields = (
            "label", "key", "evaluation_protocol",
            "num_eval_trajectories", "backend",
            "seen_env_policy", "unseen_env_policy", "resolved_model_config",
            "training_config", "payoff_input_normalization", "payoff_input_scale",
            "env_id_map", "training_env_ids", "siar_constraint_identity",
        )
        return {name: summary[name] for name in fields if name in summary}

    def _summary_cfg(self) -> Config:
        if self.params is not None:
            return self._fit_cfg
        overrides = {
            name: getattr(self, name) for name in (
                "model_type", "energy_model_type", "baseline_model_type", "coda_regularization_type",
            ) if getattr(self, name, None) is not None
        }
        return self._cfg(**overrides)

    def config_summary(self) -> dict:
        cfg = self._summary_cfg()
        fitted = self.params is not None
        model_fields = ("N", "hidden_dim", "num_layers", "cholesky_diag_floor", "model_type", "energy_model_type")
        model_config = {name: getattr(cfg, name) for name in model_fields}
        model_config["payoff_input_normalization"] = self._v_scale_mode if fitted else cfg.payoff_input_normalization
        model_config["payoff_input_scale"] = self._v_scale if fitted else None
        training_config = {name: getattr(cfg, name) for name in ("num_epochs", "batch_size", "learning_rate", "grad_clip")}
        training_config.update(optimizer="adam", gradient_clipping="global_norm", model_seed=getattr(self, "_fit_seed", None))
        # Include method-specific controls rather than just explicitly supplied kwargs.
        siar_family = self.key == "siar"
        prefixes = ("sindy_",) if self.key == "sindy" else ("siar_",) if siar_family else ("solver_",)
        model_config.update({name: value for name, value in vars(cfg).items() if name.startswith(prefixes)})
        if self.key in {"sindy", "siar"}:
            model_config = {name: value for name, value in model_config.items() if name not in model_fields or name == "N"}
            training_config = {name: value for name, value in model_config.items() if name.startswith(prefixes)}
        backend = "numpy/cvxpy+jax" if siar_family else "jax"
        return _json_safe({
            **self.kwargs,
            "label": self.label,
            "key": self.key,
            "evaluation_protocol": cfg.evaluation_protocol,
            "num_eval_trajectories": cfg.num_eval_trajectories,
            "backend": backend,
            "seen_env_policy": "not-used",
            "unseen_env_policy": "not-used",
            "resolved_model_config": model_config,
            "training_config": training_config,
            "payoff_input_normalization": model_config["payoff_input_normalization"],
            "payoff_input_scale": model_config["payoff_input_scale"],
        })

