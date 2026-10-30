"""Neural ODE fitting for the main zero-shot experiments."""
from typing import Dict, Tuple
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from pa_rmd.config import Config
from pa_rmd.models.components import LearnedDynamics, neural_ode_step
from pa_rmd.models.node import NeuralODEMLP
from pa_rmd.models.leads import LEADSDynamics
from pa_rmd.models.coda import CoDAContextDynamics


def build_dynamics_model(config):
    """Construct only the paper's Neural ODE, LEADS and CoDA variants."""
    model_type = config.baseline_model_type.replace("-", "_")
    if model_type == "neural_ode":
        return NeuralODEMLP(config.N, config.hidden_dim, config.num_layers)
    if model_type == "leads_zero_shot":
        return LEADSDynamics(config.N, config.leads_shared_hidden_dim,
                             config.leads_residual_hidden_dim, config.num_layers,
                             n_env=config.baseline_n_env)
    if model_type in {"coda_l1_zero_shot", "coda_l2_zero_shot"}:
        return CoDAContextDynamics(config.N, config.hidden_dim, config.num_layers,
                                   n_env=config.baseline_n_env,
                                   context_dim=config.coda_context_dim,
                                   norm_epsilon=config.coda_norm_epsilon)
    raise ValueError(f"Unknown baseline_model_type: {model_type}")


def single_step_ode_displacement_loss(
    params: Dict, dynamics_model: LearnedDynamics, x_t: jnp.ndarray, v_t: jnp.ndarray,
    x_next: jnp.ndarray, delta_t: float, ode_solver_substeps: int, env_id: int | jnp.ndarray | None = None,
) -> jnp.ndarray:
    x_hat = neural_ode_step(
        x_t, v_t, dynamics_model, params, delta_t=delta_t, num_substeps=ode_solver_substeps,
        project_output=True, env_id=env_id,
    )
    return jnp.sum((x_next - x_t - (x_hat - x_t)) ** 2)


def _is_leads(config: Config, dynamics_model: LearnedDynamics) -> bool:
    variant = getattr(config, "baseline_model_type", "leads_zero_shot").replace("-", "_")
    return isinstance(dynamics_model, LEADSDynamics) and variant == "leads_zero_shot"


def total_ode_displacement_loss(
    params: Dict, dynamics_model: LearnedDynamics, batch: Tuple[jnp.ndarray, ...], config: Config,
    *, return_components: bool = False,
    regularization_batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
):
    """One-step prediction plus model-specific weighted regularization.

    LEADS uses separate true training inputs for its empirical residual norm,
    so the prediction minibatch need not contain every environment. CoDA ignores ``regularization_batch``.
    """
    if len(batch) == 3:
        x_current, v_current, x_next = batch
        pair_dt = jnp.full((x_current.shape[0],), float(config.observed_delta_t()), dtype=x_current.dtype)
        env_id = jnp.zeros((x_current.shape[0],), dtype=jnp.int32)
    elif len(batch) == 4:
        x_current, v_current, x_next, pair_dt = batch
        env_id = jnp.zeros((x_current.shape[0],), dtype=jnp.int32)
    elif len(batch) == 5:
        x_current, v_current, x_next, pair_dt, env_id = batch
    else:
        raise ValueError("Neural ODE baseline expects x_current, v_current, x_next, pair_dt[, env_id].")
    per = lambda x, v, xn, dt, e: single_step_ode_displacement_loss(
        params, dynamics_model, x, v, xn, dt, config.ode_solver_substeps, env_id=e
    )
    pred_loss = jnp.mean(jax.vmap(per)(x_current, v_current, x_next, pair_dt, env_id))
    if _is_leads(config, dynamics_model):
        weight = float(config.leads_regularization_lambda)
        if weight == 0.0:
            # Exact unregularized ablation; do not evaluate then multiply by 0.
            output = spectral = jnp.zeros_like(pred_loss)
        else:
            regularization_inputs = (
                (x_current, v_current, env_id) if regularization_batch is None else regularization_batch
            )
            output, spectral = dynamics_model.regularization_terms(
                params, *regularization_inputs, norm_epsilon=config.leads_norm_epsilon,
            )
        components = {
            "loss_prediction": pred_loss,
            "loss_residual_output": weight * output,
            "loss_residual_spectral": weight * float(config.leads_spectral_weight) * spectral,
        }
        total = pred_loss if weight == 0.0 else sum(components.values())
        return (total, components) if return_components else total
    if isinstance(dynamics_model, CoDAContextDynamics):
        hyper, context = dynamics_model.regularization_terms(params, getattr(config, "coda_regularization_type", None))
        components = {
            "loss_prediction": pred_loss,
            "loss_hypernetwork": float(getattr(config, "coda_hyper_regularization_lambda", 1e-5)) * hyper,
            "loss_context": float(getattr(config, "coda_context_regularization_lambda", 1e-4)) * context,
        }
        total = components["loss_prediction"] + components["loss_hypernetwork"] + components["loss_context"]
        return (total, components) if return_components else total
    return pred_loss



def _optimizer(config: Config, params):
    optimizer = optax.chain(optax.clip_by_global_norm(config.grad_clip), optax.adam(config.learning_rate))
    return optimizer, optimizer.init(params)


def train_ode_baseline(
    config: Config, x_current: jnp.ndarray, v_current: jnp.ndarray, x_next: jnp.ndarray,
    pair_dt: jnp.ndarray | None = None, env_id: jnp.ndarray | None = None, seed: int = 42,
):
    dynamics_model = build_dynamics_model(config)
    params = dynamics_model.init_params(jax.random.PRNGKey(seed))
    optimizer, opt_state = _optimizer(config, params)
    context_coda = isinstance(dynamics_model, CoDAContextDynamics)
    regularized_leads = _is_leads(config, dynamics_model) and config.leads_regularization_lambda > 0.0
    has_components = context_coda or regularized_leads
    n_data = x_current.shape[0]
    batch_size = min(config.batch_size, n_data)
    if regularized_leads:
        raw_ids = np.zeros(n_data, dtype=np.int32) if env_id is None else np.asarray(env_id)
        if (
            raw_ids.shape != (n_data,) or raw_ids.dtype.kind not in "iu" or n_data < 1
            or set(np.unique(raw_ids).tolist()) != set(range(dynamics_model.n_env))
        ):
            raise ValueError("LEADS regularization requires compact integer labels covering all training heads")
    if pair_dt is None:
        pair_dt = jnp.full((n_data,), float(config.observed_delta_t()), dtype=x_current.dtype)
    env_id = jnp.zeros((n_data,), dtype=jnp.int32) if env_id is None else jnp.asarray(env_id, dtype=jnp.int32)
    # Full empirical regularizer: per-environment means on true training inputs.
    # Keep the prediction minibatch RNG/selection exactly as before.
    regularization_batch = (x_current, v_current, env_id) if regularized_leads else None
    loss_fn = jax.jit(lambda p, xc, vc, xn, dt, eid: total_ode_displacement_loss(
        p, dynamics_model, (xc, vc, xn, dt, eid), config,
        return_components=has_components, regularization_batch=regularization_batch,
    ))
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=has_components))
    counts = dynamics_model.parameter_count_breakdown(params)
    history = {"loss": [], "parameter_counts": counts}
    if context_coda:
        history.update({name: [] for name in ("loss_prediction", "loss_hypernetwork", "loss_context")})
    if regularized_leads:
        history.update({name: [] for name in ("loss_prediction", "loss_residual_output", "loss_residual_spectral")})
    rng = jax.random.PRNGKey(seed + 1)
    label = config.baseline_model_type.replace("_", "-")
    print(f"[{label}] parameter_counts={counts}", flush=True)
    t0 = time.time()
    for epoch in range(config.num_epochs):
        rng, subkey = jax.random.split(rng)
        idx = jax.random.choice(subkey, n_data, (batch_size,), replace=False)
        loss_val, grads = grad_fn(params, x_current[idx], v_current[idx], x_next[idx], pair_dt[idx], env_id[idx])
        if has_components:
            loss_val, components = loss_val
            for name, value in components.items():
                history[name].append(float(value))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        history["loss"].append(float(loss_val))
        if epoch % 100 == 0 or epoch == config.num_epochs - 1:
            print(f"[{label}] Epoch {epoch:4d}  loss={loss_val:.6f}  [{time.time() - t0:.0f}s]", flush=True)
    print(f"[{label}] Total training time: {time.time() - t0:.1f}s", flush=True)
    return params, dynamics_model, history
