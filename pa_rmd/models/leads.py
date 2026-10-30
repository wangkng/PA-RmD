"""LEADS shared vector field and environment residuals."""
from __future__ import annotations
from typing import Any, Dict
import jax
import jax.numpy as jnp
from pa_rmd.models.components import LearnedDynamics, _mlp_init, _mlp_apply
import numpy as np
from pa_rmd.models.components import _stack_pytrees, _take_stacked_pytree, count_tree_parameters
from pa_rmd.models.ode import _OdePredictor


class LEADSDynamics(LearnedDynamics):
    """LEADS baseline: shared vector field plus environment residual branches."""

    def __init__(
        self,
        N: int,
        shared_hidden_dim: int,
        residual_hidden_dim: int,
        num_layers: int,
        n_env: int = 1,
        use_payoff: bool = True,
    ):
        self.N = N
        self.shared_hidden_dim = int(shared_hidden_dim)
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.num_layers = int(num_layers)
        self.n_env = int(n_env)
        self.use_payoff = bool(use_payoff)
        if self.n_env < 1:
            raise ValueError("LEADSDynamics requires n_env >= 1")

    def init_params(self, rng_key: jax.Array) -> Dict[str, Any]:
        k_shared, k_residual = jax.random.split(rng_key)
        in_dim = 2 * self.N if self.use_payoff else self.N
        residual_keys = jax.random.split(k_residual, self.n_env)
        residual_params = [
            _mlp_init(
                key,
                in_dim=in_dim,
                out_dim=self.N,
                hidden_dim=self.residual_hidden_dim,
                num_layers=self.num_layers,
                small_output=True,
            )
            for key in residual_keys
        ]
        return {
            "shared": _mlp_init(
                k_shared,
                in_dim=in_dim,
                out_dim=self.N,
                hidden_dim=self.shared_hidden_dim,
                num_layers=self.num_layers,
                small_output=True,
            ),
            "residual": _stack_pytrees(residual_params),
        }

    def vector_field(
        self,
        params: Dict,
        x: jnp.ndarray,
        v: jnp.ndarray | None = None,
        env_id: int | jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if self.use_payoff:
            if v is None:
                v = jnp.zeros_like(x)
            inp = jnp.concatenate([x, v])
        else:
            inp = x
        shared = _mlp_apply(params["shared"], inp, self.num_layers, self.N)
        raw_env_id = jnp.asarray(0 if env_id is None else env_id, dtype=jnp.int32)
        safe_env_id = jnp.maximum(raw_env_id, 0)
        residual_params = _take_stacked_pytree(params["residual"], safe_env_id)
        residual = _mlp_apply(residual_params, inp, self.num_layers, self.N)
        mask = (raw_env_id >= 0).astype(residual.dtype)
        return shared + mask * residual

    def regularization_terms(
        self, params: Dict, x_current: jnp.ndarray, v_current: jnp.ndarray | None,
        env_id: jnp.ndarray, norm_epsilon: float = 1e-5,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Residual regularization adapted from the official LEADS implementation (https://github.com/yuan-yin/LEADS)."""
        
        x_current = jnp.asarray(x_current)
        if x_current.ndim != 2 or x_current.shape[1] != self.N or not x_current.shape[0]:
            raise ValueError("LEADS regularization requires nonempty [batch, N] training inputs")
        env_id = jnp.asarray(env_id)
        if env_id.shape != (x_current.shape[0],) or env_id.dtype.kind not in "iu":
            raise ValueError("LEADS regularization requires one compact integer env_id per input")
        if self.use_payoff:
            v_current = jnp.zeros_like(x_current) if v_current is None else jnp.asarray(v_current)
            if v_current.shape != x_current.shape:
                raise ValueError("LEADS regularization payoff shape must match x_current")
            inputs = jnp.concatenate([x_current, v_current], axis=-1)
        else:
            inputs = x_current
        # Reject invalid concrete IDs; traced calls are validated before jit by
        # train_ode_baseline. Never silently regularize the last head for -1.
        if not isinstance(env_id, jax.core.Tracer):
            ids = np.asarray(env_id)
            if np.any(ids < 0) or np.any(ids >= self.n_env):
                raise ValueError("LEADS regularization env_id must refer to a training head")
        residuals = jax.vmap(lambda z, e: _mlp_apply(
            _take_stacked_pytree(params["residual"], e), z, self.num_layers, self.N,
        ))(inputs, env_id)
        ratios = jnp.sum(jnp.square(residuals), axis=-1) / jnp.square(
            jnp.linalg.norm(inputs, axis=-1) + norm_epsilon
        )
        totals = jax.ops.segment_sum(ratios, env_id, num_segments=self.n_env)
        counts = jax.ops.segment_sum(jnp.ones_like(ratios), env_id, num_segments=self.n_env)
        output = jnp.sum(totals / jnp.maximum(counts, 1))
        spectral = jnp.zeros((), dtype=x_current.dtype)
        for i in range(self.num_layers + 1):
            weights = params["residual"][f"layer_{i}"]["W"]
            sigma = jnp.linalg.svd(weights, compute_uv=False)[..., 0]
            spectral = spectral + jnp.sum(jnp.square(sigma))
        return output, spectral

    def parameter_count_breakdown(self, params: Dict) -> Dict[str, int]:
        shared = count_tree_parameters(params["shared"])
        residual_active = count_tree_parameters(_take_stacked_pytree(params["residual"], 0))
        residual_total = count_tree_parameters(params["residual"])
        total = shared + residual_total
        return {
            "params_active": shared + residual_active,
            "params_shared": shared,
            "params_env_active": residual_active,
            "params_total_stored": total,
        }


class LeadsZeroShotPredictor(_OdePredictor):
    label = "LEADS-zero-shot"
    key = "leads_zero_shot"
    color = "purple"
    baseline_model_type = "leads_zero_shot"
    use_env_id = True
