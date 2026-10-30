"""Payoff-conditioned Neural ODE baseline."""
from __future__ import annotations
from typing import Any, Dict
import jax
import jax.numpy as jnp
from pa_rmd.models.components import LearnedDynamics, _mlp_init, _mlp_apply
from pa_rmd.models.ode import _OdePredictor


class NeuralODEMLP(LearnedDynamics):
    """Neural ODE baseline with optional payoff input."""

    def __init__(self, N: int, hidden_dim: int, num_layers: int, use_payoff: bool = True):
        self.N = N
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_payoff = bool(use_payoff)

    def init_params(self, rng_key: jax.Array) -> Dict[str, Any]:
        # Start near the zero vector field so the initial rollout is close to
        # x_{t+1} ≈ x_t.  For the ODE baseline this is an intentional stability
        # choice, not a leftover from the metric/energy routes.
        return _mlp_init(
            rng_key,
            in_dim=2 * self.N if self.use_payoff else self.N,
            out_dim=self.N,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            small_output=True,
        )

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
        return _mlp_apply(params, inp, self.num_layers, self.N)


class NodePredictor(_OdePredictor):
    label = "Neural ODE"
    key = "node"
    color = "royalblue"
