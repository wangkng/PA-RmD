"""Shared fitting and zero-shot environment routing."""
from __future__ import annotations
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from pa_rmd.models.base import _BasePredictor, _json_safe
from pa_rmd.models.components import neural_ode_step


class _OdePredictor(_BasePredictor):
    baseline_model_type = "neural_ode"
    coda_regularization_type = None
    use_env_id = False

    @staticmethod
    def _validated_env_ids(env_id: Any, n: int, *, training: bool = False):
        # Static Python IDs remain NumPy values even when called while tracing a
        # rollout. Dynamic JAX IDs need only static dtype/shape checks here.
        if isinstance(env_id, (list, tuple)) and any(isinstance(value, (bool, np.bool_)) for value in env_id):
            raise ValueError("env_id must contain integers, not booleans")
        raw = env_id if isinstance(env_id, jax.core.Tracer) else np.asarray(env_id)
        if raw.dtype.kind not in "iu":
            raise ValueError("env_id must contain integers (not booleans, floats or strings)")
        if raw.ndim > 1 or (training and raw.ndim != 1):
            raise ValueError("env_id must be a scalar or a one-dimensional batch")
        if raw.ndim == 1 and raw.shape[0] != n:
            raise ValueError(f"env_id batch length {raw.shape[0]} does not match x batch length {n}")
        if training and (n < 1 or np.any(raw < 0)):
            raise ValueError("Training env_id must be nonnegative and nonempty; -1 is reserved for unseen")
        return raw

    def fit(self, batch: Any, *, seed: int) -> None:
        if self.use_env_id and batch.env_id is not None:
            raw = self._validated_env_ids(batch.env_id, int(batch.x_current.shape[0]), training=True)
            train_env_ids = sorted(int(v) for v in np.unique(raw))
            self._env_id_map = {env: i for i, env in enumerate(train_env_ids)}
            compact_env_id = jnp.asarray([self._env_id_map[int(v)] for v in raw], dtype=jnp.int32)
            n_env = len(train_env_ids)
        else:
            self._env_id_map = {0: 0}
            compact_env_id = None
            n_env = 1
        overrides = {"baseline_model_type": self.baseline_model_type, "baseline_n_env": n_env}
        if self.coda_regularization_type is not None:
            overrides["coda_regularization_type"] = self.coda_regularization_type
        cfg = self._cfg(**overrides)
        self._fit_cfg = cfg
        self._fit_seed = int(seed)
        self._set_v_scale(batch, cfg)
        from pa_rmd.train_ode import train_ode_baseline

        self.params, self.model, self._history = train_ode_baseline(
            cfg, batch.x_current, self._scale_v(batch.v_current), batch.x_next, batch.pair_dt, env_id=compact_env_id, seed=seed
        )
        self._rollout_batch_cache.clear()

    def _compact_eval_env_id(self, env_id: int | jnp.ndarray | None, n: int) -> jnp.ndarray:
        if self.use_env_id:
            if set(self._env_id_map.values()) != set(range(len(self._env_id_map))):
                raise ValueError("Training environment mapping must use contiguous compact indices")
            if env_id is None:
                return jnp.full((n,), -1, dtype=jnp.int32)
            raw = self._validated_env_ids(env_id, n)
            if isinstance(raw, jax.core.Tracer):
                values = jnp.broadcast_to(raw, (n,))
                mapped = jnp.full((n,), -1, dtype=jnp.int32)
                for label, compact in self._env_id_map.items():
                    # Avoid integer overflow aliasing a large raw label to seen.
                    if np.iinfo(raw.dtype).min <= label <= np.iinfo(raw.dtype).max:
                        mapped = jnp.where(values == label, compact, mapped)
                return mapped
            values = np.broadcast_to(raw, (n,))
            return jnp.asarray([self._env_id_map.get(int(value), -1) for value in values], dtype=jnp.int32)
        return jnp.zeros((n,), dtype=jnp.int32)

    def predict_step(self, x: jnp.ndarray, v: jnp.ndarray, dt: float | jnp.ndarray, env_id: int | jnp.ndarray | None = None) -> jnp.ndarray:
        self._require_fit()
        x, v = jnp.asarray(x), self._scale_v(v)
        dt = jnp.broadcast_to(jnp.asarray(dt), (x.shape[0],)) if self.use_env_id else self._batch_dt(x, dt)
        compact_env_id = self._compact_eval_env_id(env_id, int(x.shape[0]))
        return self._predict_compact_step(self.params, x, v, dt, compact_env_id)

    def _predict_compact_step(self, params, x, v, dt, compact_env_id):
        """Numerical kernel: IDs are already compact JAX integers, v is scaled."""
        def one(x_t, v_t, step_dt, e_id):
            return neural_ode_step(
                x_t,
                v_t,
                self.model,
                params,
                delta_t=step_dt,
                num_substeps=self._fit_cfg.ode_solver_substeps,
                project_output=True,
                env_id=e_id,
            )

        return jax.vmap(one)(x, v, dt, compact_env_id)

    def predict_rollout(self, x0, dts, game, env_id=None):
        if not self.use_env_id:
            return super().predict_rollout(x0, dts, game, env_id=env_id)
        return self.predict_rollout_batch(jnp.asarray(x0)[None, :], dts, game, env_id=env_id)[0]

    def predict_rollout_batch(self, x0, dts, game, env_id=None):
        if not self.use_env_id:
            return super().predict_rollout_batch(x0, dts, game, env_id=env_id)
        self._require_fit()
        x0, dts = jnp.asarray(x0), jnp.asarray(dts)
        compact_ids = self._compact_eval_env_id(env_id, int(x0.shape[0]))
        matrix, intercept = getattr(game, "matrix", None), getattr(game, "intercept", None)
        is_affine = matrix is not None and intercept is not None
        cache_key = ("zero-shot", is_affine)
        rollout_fn = self._rollout_batch_cache.get(cache_key)
        if rollout_fn is None:
            def rollout_impl(params, x_init, step_dts, ids, scale, payoff_arg, intercept_arg):
                def scan_step(x_batch, step_dt):
                    v = x_batch @ payoff_arg.T + intercept_arg if is_affine else jax.vmap(payoff_arg)(x_batch)
                    dt_batch = jnp.full((x_batch.shape[0],), step_dt, dtype=x_batch.dtype)
                    x_next = self._predict_compact_step(params, x_batch, v / scale, dt_batch, ids)
                    return x_next, x_next

                _, steps = jax.lax.scan(scan_step, x_init, step_dts)
                return jnp.swapaxes(jnp.concatenate([x_init[None, :, :], steps], axis=0), 0, 1)

            rollout_fn = jax.jit(rollout_impl, static_argnums=() if is_affine else (5,))
            self._rollout_batch_cache[cache_key] = rollout_fn
        payoff_arg = jnp.asarray(matrix) if is_affine else game.payoff
        intercept_arg = jnp.asarray(intercept) if is_affine else None
        return rollout_fn(self.params, x0, dts, compact_ids, self._v_scale, payoff_arg, intercept_arg)

    def config_summary(self) -> dict:
        summary = super().config_summary()
        cfg = self._summary_cfg()
        context_coda = self.use_env_id and self.baseline_model_type.startswith("coda")
        leads = self.baseline_model_type.startswith("leads")
        fitted = self.params is not None
        model_config = {
            "baseline_model_type": self.baseline_model_type,
            "N": cfg.N, "num_layers": cfg.num_layers,
            "backbone": "mlp", "activation": "softplus", "small_output_init_std": 0.01,
            "use_payoff": True,
            "baseline_n_env": cfg.baseline_n_env if fitted else None,
            "payoff_input_normalization": summary["payoff_input_normalization"],
            "payoff_input_scale": summary["payoff_input_scale"],
            "integrator": "rk4", "ode_solver_substeps": cfg.ode_solver_substeps,
            "project_output": "once-per-observed-step", "payoff_substeps": "fixed-v_t",
        }
        if leads:
            model_config.update(leads_shared_hidden_dim=cfg.leads_shared_hidden_dim, leads_residual_hidden_dim=cfg.leads_residual_hidden_dim)
            if self.use_env_id:
                model_config.update(
                    leads_regularization_lambda=cfg.leads_regularization_lambda,
                    leads_spectral_weight=cfg.leads_spectral_weight,
                    leads_norm_epsilon=cfg.leads_norm_epsilon,
                    regularizer="normalized-output+sum-spectral-squared",
                    regularization_enabled=cfg.leads_regularization_lambda > 0.0,
                    regularization_reduction="sum-env-means",
                    regularization_inputs="all-training-current-inputs",
                    regularization_input_normalization="same-as-vector-field",
                    residual_output_formula="sum_env mean(||g_e(z)||_2^2 / (||z||_2 + epsilon)^2)",
                    residual_spectral_formula="sum_env sum_layer sigma_max(W_e_layer)^2",
                    regularized_parameters="residual-only; spectral-excludes-biases",
                    spectral_norm="exact-svd", regularization_schedule="constant",
                )
        else:
            model_config["hidden_dim"] = cfg.hidden_dim
        if context_coda:
            model_config.update(
                parameterization="base+linear-hypernetwork@context", hypernetwork_bias=False,
                modulated_parameters="all-layer-weights-and-biases", context_initialization="zeros",
                hypernetwork_initialization="normal/sqrt(context_dim)",
                context_mean_reduction="equal-weight-training-environments",
                coda_context_dim=cfg.coda_context_dim,
                coda_regularization_type=self.coda_regularization_type,
                hyper_regularization="row-l12" if self.coda_regularization_type == "l1" else "frobenius-squared",
                context_regularization="l2-squared", regularization_reduction="sum",
                coda_hyper_regularization_lambda=cfg.coda_hyper_regularization_lambda,
                coda_context_regularization_lambda=cfg.coda_context_regularization_lambda,
                coda_norm_epsilon=cfg.coda_norm_epsilon,
                row_norm_formula="sqrt(sum(row**2)+epsilon**2)-epsilon",
            )
        training_config = summary["training_config"]
        training_config.update(
            objective="one-step-displacement-squared-error", prediction_reduction="batch-mean",
            minibatch_sampling="without-replacement", rng_seed_offset=1,
            adam_b1=0.9, adam_b2=0.999, adam_eps=1e-8, adam_eps_root=0.0,
            observed_delta_t=cfg.observed_delta_t(), dtype="float32",
        )
        if self.use_env_id and leads:
            training_config.update(
                objective="one-step-displacement-squared-error+leads-residual-complexity",
                leads_regularization_lambda=cfg.leads_regularization_lambda,
                leads_spectral_weight=cfg.leads_spectral_weight,
                leads_norm_epsilon=cfg.leads_norm_epsilon,
                regularization_inputs="all-training-current-inputs",
                regularization_reduction="sum-env-means", spectral_norm="exact-svd",
                regularization_schedule="constant",
            )
        if self.use_env_id:
            seen, unseen = "trained-context" if context_coda else "trained-head", "mean-context" if context_coda else "shared-only"
        else:
            seen = unseen = "not-used"
        summary.update(
            resolved_model_config=model_config, training_config=training_config, backend="jax",
            seen_env_policy=seen, unseen_env_policy=unseen,
            env_id_map={str(label): compact for label, compact in self._env_id_map.items()} if fitted and self.use_env_id else {},
            training_env_ids=sorted(self._env_id_map) if fitted and self.use_env_id else [],
        )
        return _json_safe(summary)

    def loss_components(self) -> dict[str, list[float]]:
        """Weighted model-specific terms; history() returns the total loss list."""
        names = ("loss_prediction", "loss_hypernetwork", "loss_context", "loss_residual_output", "loss_residual_spectral")
        return {name: list(self._history[name]) for name in names if name in self._history}

    def parameter_count(self) -> dict | None:
        counts = super().parameter_count()
        if counts is None or not self.baseline_model_type.startswith("leads") or not self.use_env_id:
            return counts
        counts = dict(counts)
        counts.update(params_active_seen=counts["params_active"], params_active_unseen=counts["params_shared"], params_env_active_unseen=0)
        return counts
