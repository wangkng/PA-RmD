"""PA-RmD velocity-matching objective and optimizer updates."""
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pa_rmd.config import Config
from pa_rmd.models.pa_rmd import QuadraticEnergyFromTangentTransport, TangentCholeskyMLP


def displacement_loss(params: dict, energy_model: QuadraticEnergyFromTangentTransport,
                      batch: tuple[jnp.ndarray, ...], config: Config) -> jnp.ndarray:
    """Mean squared velocity error before finite-step simplex projection."""
    if len(batch) == 3:
        x_current, v_current, x_next = batch
        pair_dt = jnp.full((x_current.shape[0],), float(config.observed_delta_t()), dtype=x_current.dtype)
    else:
        x_current, v_current, x_next, pair_dt = batch

    def per(x_t, v_t, x_n, dt):
        z_hat_raw = energy_model.closed_form_displacement(params, x_t, v_t, dt)
        sq_loss = jnp.sum((x_n - x_t - z_hat_raw) ** 2)
        return sq_loss / (dt ** 2 + 1e-12)

    losses = jax.vmap(per)(x_current, v_current, x_next, pair_dt)
    return jnp.mean(losses)


def train(config: Config, x_current, v_current, x_next, pair_dt=None, seed=42):
    """Fit the concrete transport model using fixed-budget Adam updates."""
    tangent_model = TangentCholeskyMLP(
        config.N,
        config.hidden_dim,
        config.num_layers,
        init_transport_scale=1.0,
        diag_floor=float(config.cholesky_diag_floor),
    )
    energy_model = QuadraticEnergyFromTangentTransport(tangent_model, config.observed_delta_t())
    params = energy_model.init_params(jax.random.PRNGKey(seed))
    optimizer = optax.chain(optax.clip_by_global_norm(config.grad_clip), optax.adam(config.learning_rate))
    opt_state = optimizer.init(params)
    n_data = x_current.shape[0]
    batch_size = min(config.batch_size, n_data)
    if pair_dt is None:
        pair_dt = jnp.full((n_data,), float(config.observed_delta_t()), dtype=x_current.dtype)
    grad_fn = jax.jit(jax.value_and_grad(
        lambda p, xc, vc, xn, dt: displacement_loss(p, energy_model, (xc, vc, xn, dt), config),
    ))
    history, rng, t0 = {"loss": []}, jax.random.PRNGKey(seed + 1), time.time()
    for epoch in range(config.num_epochs):
        rng, subkey = jax.random.split(rng)
        idx = np.array(jax.random.choice(subkey, n_data, (batch_size,), replace=False))
        loss_val, grads = grad_fn(params, x_current[idx], v_current[idx], x_next[idx], pair_dt[idx])
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        history["loss"].append(float(loss_val))
        if epoch % 100 == 0 or epoch == config.num_epochs - 1:
            print(f"  [disp|given_v] Epoch {epoch:5d}  loss={loss_val:.6f}  [{time.time() - t0:.0f}s]", flush=True)
    print(f"  Total training time: {time.time() - t0:.1f}s", flush=True)
    return params, energy_model, history
