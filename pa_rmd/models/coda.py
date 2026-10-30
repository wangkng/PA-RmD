"""CoDA all-parameter linear hypernetwork and training contexts."""
from __future__ import annotations
from typing import Any, Dict
import jax
import jax.numpy as jnp
from pa_rmd.models.components import LearnedDynamics, _mlp_init, _mlp_apply
import numpy as np
from pa_rmd.models.components import count_tree_parameters
from pa_rmd.models.ode import _OdePredictor


class CoDAContextDynamics(LearnedDynamics):
    """All-parameter linear hypernetwork; unknown environments use mean context.

    The parameter layout contains only Python shapes/slices, fixed at construction.
    Decoding never caches arrays or tracers and remains differentiable under JAX.
    """

    def __init__(
        self, N: int, hidden_dim: int, num_layers: int, n_env: int = 1,
        use_payoff: bool = True, context_dim: int = 2, norm_epsilon: float = 1e-8,
    ):
        self.N = int(N)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.n_env = int(n_env)
        self.use_payoff = bool(use_payoff)
        self.context_dim = int(context_dim)
        self.norm_epsilon = float(norm_epsilon)
        if self.n_env < 1 or self.context_dim < 1:
            raise ValueError("CoDAContextDynamics requires n_env and context_dim >= 1")
        min_epsilon = float(np.sqrt(np.finfo(np.float32).tiny))
        max_epsilon = float(np.sqrt(np.finfo(np.float32).max)) / 2.0
        if not np.isfinite(self.norm_epsilon) or not min_epsilon <= self.norm_epsilon <= max_epsilon:
            raise ValueError("CoDA norm_epsilon must have a finite, normal FP32 square")
        dims = [2 * self.N if self.use_payoff else self.N] + [self.hidden_dim] * self.num_layers + [self.N]
        shapes = {
            f"layer_{i}": {"W": (dims[i + 1], dims[i]), "b": (dims[i + 1],)}
            for i in range(len(dims) - 1)
        }
        layout, offset = [], 0
        # Match JAX's sorted dictionary leaf order, including for >=10 layers.
        for layer in sorted(shapes):
            for name in sorted(shapes[layer]):
                shape = shapes[layer][name]
                size = int(np.prod(shape))
                layout.append((layer, name, shape, offset, offset + size))
                offset += size
        self._parameter_layout = tuple(layout)
        self.base_parameter_count = offset

    def init_params(self, rng_key: jax.Array) -> Dict[str, Any]:
        base_key, hyper_key = jax.random.split(rng_key)
        base = _mlp_init(
            base_key, 2 * self.N if self.use_payoff else self.N, self.N,
            self.hidden_dim, self.num_layers, small_output=True,
        )
        dtype = base["layer_0"]["W"].dtype
        return {
            "base": base,
            "hyper_weight": jax.random.normal(
                hyper_key, (self.base_parameter_count, self.context_dim), dtype=dtype,
            ) / jnp.sqrt(float(self.context_dim)),
            "contexts": jnp.zeros((self.n_env, self.context_dim), dtype=dtype),
        }

    def unflatten_base(self, flat: jnp.ndarray) -> Dict:
        base = {}
        for layer, name, shape, start, stop in self._parameter_layout:
            base.setdefault(layer, {})[name] = flat[start:stop].reshape(shape)
        return base

    def decode_context(self, params: Dict, context: jnp.ndarray) -> Dict:
        if set(params) != {"base", "hyper_weight", "contexts"}:
            raise ValueError("CoDAContextDynamics requires base/hyper_weight/contexts, not a legacy checkpoint")
        if params["hyper_weight"].shape != (self.base_parameter_count, self.context_dim):
            raise ValueError("CoDA hyper_weight shape does not match the model layout")
        if params["contexts"].shape != (self.n_env, self.context_dim):
            raise ValueError("CoDA contexts must contain exactly the training environments")
        offsets = self.unflatten_base(params["hyper_weight"] @ context)
        return jax.tree_util.tree_map(lambda base, delta: base + delta.astype(base.dtype), params["base"], offsets)

    def effective_params(self, params: Dict, env_id=None) -> Dict:
        if "contexts" not in params:
            raise ValueError("CoDAContextDynamics requires base/hyper_weight/contexts, not a legacy checkpoint")
        contexts = params["contexts"]
        raw_id = jnp.asarray(-1 if env_id is None else env_id, dtype=jnp.int32)
        safe_id = jnp.clip(raw_id, 0, self.n_env - 1)
        context = jnp.where(
            (raw_id >= 0) & (raw_id < self.n_env), contexts[safe_id],
            jnp.mean(contexts, axis=0),
        )
        return self.decode_context(params, context)

    def vector_field(self, params: Dict, x: jnp.ndarray, v=None, env_id=None) -> jnp.ndarray:
        inp = jnp.concatenate([x, jnp.zeros_like(x) if v is None else v]) if self.use_payoff else x
        return _mlp_apply(self.effective_params(params, env_id), inp, self.num_layers, self.N)

    def regularization_terms(self, params: Dict, regularization_type: str | None = None):
        weight = params["hyper_weight"]
        reg_type = (regularization_type or "l2").lower()
        if reg_type == "l1":
            # Smooth row l2 norm with zero value/finite derivative at a zero row.
            eps = self.norm_epsilon
            hyper = jnp.sum(jnp.sqrt(jnp.sum(jnp.square(weight), axis=1) + eps ** 2) - eps)
        elif reg_type == "l2":
            hyper = jnp.sum(jnp.square(weight))
        else:
            raise ValueError(f"Unknown context CoDA regularization type: {regularization_type}")
        return hyper, jnp.sum(jnp.square(params["contexts"]))

    def parameter_count_breakdown(self, params: Dict) -> Dict[str, int]:
        base = count_tree_parameters(params["base"])
        hyper = count_tree_parameters(params["hyper_weight"])
        contexts = count_tree_parameters(params["contexts"])
        shared = base + hyper
        return {
            "params_active": shared + self.context_dim,
            "params_shared": shared,
            "params_env_active": self.context_dim,
            "params_total_stored": shared + contexts,
            "params_base": base,
            "params_hypernetwork": hyper,
            "params_contexts_total": contexts,
            "params_active_seen": shared + self.context_dim,
            # The pure-function mean reads every training context at inference.
            "params_active_unseen": shared + contexts,
            "params_env_active_unseen": contexts,
            "params_decoded_model": base,
        }


class CodaPredictor(_OdePredictor):
    """CoDA zero-shot predictor with L1 or L2 hypernetwork regularization."""

    color = "crimson"
    use_env_id = True

    def __init__(self, cfg, regularization_type: str = "l2", **kwargs: Any):
        if not isinstance(regularization_type, str) or regularization_type.lower() not in {"l1", "l2"}:
            raise ValueError("CoDA regularization_type must be 'l1' or 'l2'")
        regularization_type = regularization_type.lower()
        if "coda_regularization_type" in kwargs and kwargs["coda_regularization_type"] != regularization_type:
            raise ValueError("coda_regularization_type conflicts with regularization_type")
        self.coda_regularization_type = regularization_type
        self.label = f"CoDA-{regularization_type.upper()}-zero-shot"
        self.key = self.baseline_model_type = f"coda_{regularization_type}_zero_shot"
        # Preserve the constructor choice when input_config_summary rebuilds a
        # fresh instance from the shared configuration and generic kwargs.
        super().__init__(cfg, regularization_type=regularization_type, **kwargs)
