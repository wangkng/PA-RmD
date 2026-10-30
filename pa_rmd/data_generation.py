from typing import Tuple, List, Optional
import jax
import jax.numpy as jnp

from pa_rmd.games import PopulationGame
from pa_rmd.metrics import RiemannianMetric
from pa_rmd.utils import project_simplex, sample_simplex_interior


# If these constants are changed, manually delete the corresponding MWU synthetic cache.
_MWU_ETA = 0.2
_MWU_PAIR_DT = 0.1


def rmd_vector_field(
    x: jnp.ndarray,
    game: PopulationGame,
    metric: RiemannianMetric,
) -> jnp.ndarray:
    """Compute the RmD vector field (Mertikopoulos and Sandholm, JET 2018)."""
    v = game.payoff(x)                          # (N,)
    g_inv = metric.g_inv(x)                     # (N, N)
    v_sharp = g_inv @ v                         # (N,)
    n = g_inv @ jnp.ones(x.shape[0])            # (N,)
    projection = (jnp.sum(v_sharp) / jnp.sum(n)) * n
    dx = v_sharp - projection
    return dx






def mwu_step(x: jnp.ndarray, game: PopulationGame) -> jnp.ndarray:
    """Single multiplicative-weights update on the simplex."""
    v = game.payoff(x)
    logits = jnp.log(jnp.maximum(x, 1e-12)) + _MWU_ETA * v
    return jax.nn.softmax(logits)


def population_vector_field(
    x: jnp.ndarray,
    game: PopulationGame,
    config,
    metric: Optional[RiemannianMetric] = None,
) -> jnp.ndarray:
    if config.metric_type in {"replicator", "projection"}:
        if metric is None:
            raise ValueError(f"{config.metric_type} dynamics requires a metric")
        return rmd_vector_field(x, game, metric)
    raise ValueError(f"Unknown dynamics: {config.metric_type!r}")


def rk4_step(
    x: jnp.ndarray,
    h: float,
    game: PopulationGame,
    config,
    metric: Optional[RiemannianMetric] = None,
) -> jnp.ndarray:
    """Single RK4 step.

    No projection is applied inside the step: RmD's vector field satisfies
    sum(dx/dt) = 0 and preserves the positive support for p >= 1, so the
    simplex constraints are invariant under the exact flow. Projecting after
    each substep would break the O(h^4) consistency of RK4 (the intermediate
    k2/k3 stages are allowed to leave the simplex temporarily). Numerical
    drift is controlled by choosing a small enough h (delta_t / substeps).
    """
    k1 = population_vector_field(x, game, config, metric)
    k2 = population_vector_field(x + 0.5 * h * k1, game, config, metric)
    k3 = population_vector_field(x + 0.5 * h * k2, game, config, metric)
    k4 = population_vector_field(x + h * k3, game, config, metric)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_trajectory(
    x0: jnp.ndarray,
    game: PopulationGame,
    metric: Optional[RiemannianMetric],
    config,
    T: int,
    delta_t: float,
    substeps: int = 20,
) -> jnp.ndarray:
    """
    Integrate the configured population dynamics from x0 for T steps, sampling every delta_t.
    Uses RK4 with internal step h = delta_t / substeps.
    Returns array of shape (T+1, N).
    """
    h = delta_t / substeps
    trajectory = [x0]
    x = x0

    for _ in range(T):
        for _ in range(substeps):
            x = rk4_step(x, h, game, config, metric)
        x = project_simplex(x)
        trajectory.append(x)

    return jnp.stack(trajectory, axis=0)  # (T+1, N)


def integrate_observed_trajectory(
    x0: jnp.ndarray,
    game: PopulationGame,
    metric: Optional[RiemannianMetric],
    config,
    observed_T: int,
    delta_t: float,
    observation_stride: int = 1,
    substeps: int = 20,
) -> jnp.ndarray:
    """Integrate latent dynamics and return observations every `observation_stride` steps."""
    stride = max(int(observation_stride), 1)
    if stride == 1:
        return integrate_trajectory(
            x0,
            game,
            metric,
            config,
            observed_T,
            delta_t,
            substeps=substeps,
        )

    latent_traj = integrate_trajectory(
        x0,
        game,
        metric,
        config,
        observed_T * stride,
        delta_t,
        substeps=substeps,
    )
    return latent_traj[::stride]


def integrate_mwu_observed_trajectory(
    x0: jnp.ndarray,
    game: PopulationGame,
    observed_T: int,
    observation_stride: int = 1,
) -> jnp.ndarray:
    """Generate observed MWU trajectory; one observed transition is one update round."""
    stride = max(int(observation_stride), 1)
    if stride != 1:
        raise ValueError("MWU data generation expects observation_stride=1.")

    trajectory = [x0]
    x = x0
    for _ in range(observed_T):
        x = mwu_step(x, game)
        x = project_simplex(x)
        trajectory.append(x)

    return jnp.stack(trajectory, axis=0)


def _build_training_pairs_from_trajectories(
    trajectories: List[jnp.ndarray],
    config,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, List[jnp.ndarray]]:
    x_current_list = []
    x_next_list = []
    pair_dt_list = []
    if config.metric_type == "MWU":
        dt_value = _MWU_PAIR_DT
    else:
        dt_value = float(config.observed_delta_t())

    for observed in trajectories:
        xc = observed[:-1]
        xn = observed[1:]
        pair_dt = jnp.full(
            (xc.shape[0],),
            dt_value,
            dtype=observed.dtype,
        )

        if xc.shape[0] == 0:
            continue

        x_current_list.append(xc)
        x_next_list.append(xn)
        pair_dt_list.append(pair_dt)

    if not x_current_list:
        raise ValueError("No training pairs were generated from the observed trajectories.")

    x_current = jnp.concatenate(x_current_list, axis=0)
    x_next = jnp.concatenate(x_next_list, axis=0)
    pair_dt = jnp.concatenate(pair_dt_list, axis=0)
    return x_current, x_next, pair_dt, trajectories


def generate_dataset(
    config,
    game: PopulationGame,
    metric: Optional[RiemannianMetric],
    seed: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, List[jnp.ndarray]]:
    """
    Generate M trajectories, each of observed length T.

    Returns:
        x_current: observed training pairs, shape (K, N)
        x_next:    observed training pairs, shape (K, N)
        pair_dt:   elapsed time for each pair, shape (K,)
        trajectories: list of observed trajectories
    """
    key = jax.random.PRNGKey(seed)
    M = config.num_trajectories
    T = config.trajectory_length
    N = config.N
    epsilon = config.interior_epsilon

    x0_batch = sample_simplex_interior(key, M, N, epsilon)  # (M, N)
    trajectories = []

    for i in range(M):
        x0 = x0_batch[i]
        if config.metric_type == "MWU":
            traj = integrate_mwu_observed_trajectory(
                x0,
                game,
                observed_T=T,
                observation_stride=getattr(config, 'observation_stride', 1),
            )
        else:
            traj = integrate_observed_trajectory(
                x0,
                game,
                metric,
                config,
                observed_T=T,
                delta_t=config.delta_t,
                observation_stride=getattr(config, 'observation_stride', 1),
                substeps=20,
            )
        trajectories.append(traj)

    return _build_training_pairs_from_trajectories(
        trajectories,
        config,
    )
