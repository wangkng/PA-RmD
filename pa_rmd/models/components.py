"""Small neural vector-field components and fixed-payoff RK4 integration."""
from abc import ABC, abstractmethod
from typing import Any, Dict
import jax
import jax.numpy as jnp
import jax.nn as jnn
from pa_rmd.utils import project_simplex


def _init_linear(key: jax.Array, in_dim: int, out_dim: int) -> Dict:
    k1, k2 = jax.random.split(key)
    scale = jnp.sqrt(2.0 / in_dim)
    W = jax.random.normal(k1, (out_dim, in_dim)) * scale
    b = jnp.zeros(out_dim)
    return {"W": W, "b": b}


def _apply_linear(layer: Dict, x: jnp.ndarray) -> jnp.ndarray:
    return layer["W"] @ x + layer["b"]


def _mlp_init(
    key: jax.Array,
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    num_layers: int,
    small_output: bool = False,
) -> Dict:
    """Initialize MLP params: input -> [hidden]*num_layers -> output.

    small_output: if True, initialize output layer with near-zero weights so that
    the output-layer weights contribute very little at initialization, so the
    last-layer bias dominates the initial output. The final output is intentionally close to the zero vector field.
    """
    params = {}
    dims = [in_dim] + [hidden_dim] * num_layers + [out_dim]
    output_layer_idx = len(dims) - 2
    for i in range(len(dims) - 1):
        key, subkey = jax.random.split(key)
        if small_output and i == output_layer_idx:
            # Keep the output-layer contribution small so the bias can control
            # the initial metric scale.
            W = jax.random.normal(subkey, (dims[i + 1], dims[i])) * 0.01
            b = jnp.zeros(dims[i + 1])
            params[f"layer_{i}"] = {"W": W, "b": b}
        else:
            params[f"layer_{i}"] = _init_linear(subkey, dims[i], dims[i + 1])
    return params


def _mlp_apply(params: Dict, x: jnp.ndarray, num_layers: int, out_dim: int) -> jnp.ndarray:
    total_layers = num_layers + 1  # including output layer
    h = x
    for i in range(total_layers - 1):
        h = _apply_linear(params[f"layer_{i}"], h)
        h = jnn.softplus(h)
    h = _apply_linear(params[f"layer_{total_layers - 1}"], h)
    return h


def _stack_pytrees(pytrees):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *pytrees)


def _take_stacked_pytree(pytree, index):
    idx = jnp.asarray(0 if index is None else index, dtype=jnp.int32)
    return jax.tree_util.tree_map(lambda x: x[idx], pytree)


def count_tree_parameters(params: Any) -> int:
    """Count scalar parameters in a JAX pytree."""
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(int(jnp.size(leaf)) for leaf in leaves))


class LearnedDynamics(ABC):
    """Abstract base for black-box continuous-time dynamics baselines."""

    @abstractmethod
    def init_params(self, rng_key: jax.Array) -> Dict[str, Any]:
        pass

    @abstractmethod
    def vector_field(
        self,
        params: Dict,
        x: jnp.ndarray,
        v: jnp.ndarray | None = None,
        env_id: int | jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Returns f_theta(x, v) for dx/dt = f_theta(x, v)."""
        pass

    def parameter_count_breakdown(self, params: Dict) -> Dict[str, int]:
        total = count_tree_parameters(params)
        return {
            "params_active": total,
            "params_shared": total,
            "params_env_active": 0,
            "params_total_stored": total,
        }


def neural_ode_step(
    x_t: jnp.ndarray,
    v_t: jnp.ndarray | None,
    dynamics_model: LearnedDynamics,
    params: Dict[str, Any],
    delta_t: float,
    num_substeps: int = 4,
    project_output: bool = True,
    env_id: int | jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Fixed-step RK4 integrator for dx/dt = f_theta(x, v).

    When ``project_output=True`` the returned state is projected back onto the
    simplex exactly once at the end of the RK4 step.
    """
    h = delta_t / num_substeps

    def f(x):
        return dynamics_model.vector_field(params, x, v_t, env_id=env_id)

    def rk4_step(x, _):
        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        x_new = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_new, None

    x_hat, _ = jax.lax.scan(rk4_step, x_t, None, length=num_substeps)
    if project_output:
        x_hat = project_simplex(x_hat)
    return x_hat
