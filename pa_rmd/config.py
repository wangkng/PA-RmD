from dataclasses import dataclass
import numpy as np


# Constructor settings for the paper experiments; algorithms do not select datasets.
_METHOD_SETTINGS = {
    "coda-l1": {"regularization_type": "l1"},
    "coda-l2": {"regularization_type": "l2"},
}
_EXPERIMENT_SETTINGS = {
    "synthetic": {
        "siar": {
            "siar_degree": 3,
            "siar_enforce_forward_invariance": True,
            "siar_enforce_positive_correlation": True,
            "siar_project_simplex": True,
            "siar_solver_objective_scale": 1.0,
            "siar_payoff_graph_source": "known",
            "siar_payoff_scale_mode": "observed_or_graph",
            "siar_certificate_tolerance": None,
        },
    },
    "youtube": {
        "siar": {
            "siar_degree": 2,
            "siar_enforce_forward_invariance": False,
            "siar_enforce_positive_correlation": True,
            "siar_project_simplex": True,
            "siar_solver_objective_scale": 100.0,
            "siar_payoff_graph_source": "estimated",
            "siar_payoff_scale_mode": "observed",
            "siar_certificate_tolerance": 1e-5,
        },
    },
}


def method_settings(method: str, experiment: str, **overrides) -> dict:
    """Resolve explicit overrides over paper settings and model defaults."""
    if experiment not in _EXPERIMENT_SETTINGS:
        raise ValueError(f"Unknown experiment: {experiment}")
    return {**_METHOD_SETTINGS.get(method, {}),
            **_EXPERIMENT_SETTINGS[experiment].get(method, {}), **overrides}


@dataclass
class Config:
    # Game
    game_type: str = "rps"
    N: int = 3
    payoff_matrix: np.ndarray | None = None
    payoff_intercept: np.ndarray | None = None

    # Ground-truth dynamics selector. "replicator" uses the replicator metric;
    # "projection" uses the Euclidean metric; "MWU" uses discrete-time
    # multiplicative weights update.
    metric_type: str = "replicator"

    # Data generation
    num_trajectories: int = 50
    trajectory_length: int = 30
    delta_t: float = 0.05
    interior_epsilon: float = 0.05
    observation_stride: int = 1
    evaluation_protocol: str = "independent-ic-v1"
    num_eval_trajectories: int | None = None

    # Model
    hidden_dim: int = 64
    num_layers: int = 2
    cholesky_diag_floor: float = 1e-3
    # LEADS official ODE starting values; zero disables the residual penalty.
    leads_regularization_lambda: float = 2e-4
    leads_spectral_weight: float = 1e-2
    leads_norm_epsilon: float = 1e-5
    coda_context_dim: int = 2
    coda_hyper_regularization_lambda: float = 1e-5
    coda_context_regularization_lambda: float = 1e-4
    coda_norm_epsilon: float = 1e-8

    # Training
    learning_rate: float = 1e-3
    num_epochs: int = 8000
    batch_size: int = 256
    grad_clip: float = 10.0

    def observed_delta_t(self) -> float:
        return self.delta_t * self.observation_stride

    def __post_init__(self):
        if self.metric_type not in {"replicator", "projection", "MWU"}:
            raise ValueError(f"Unknown metric_type/dynamics selector: {self.metric_type!r}")
        if int(self.observation_stride) < 1:
            raise ValueError("observation_stride must be >= 1")
        if float(self.cholesky_diag_floor) < 0.0:
            raise ValueError("cholesky_diag_floor must be >= 0")
        if self.evaluation_protocol != "independent-ic-v1":
            raise ValueError(f"Unknown evaluation_protocol: {self.evaluation_protocol!r}")
        for name in ("coda_context_dim", "num_eval_trajectories"):
            value = getattr(self, name)
            if value is None and name == "num_eval_trajectories":
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "coda_hyper_regularization_lambda", "coda_context_regularization_lambda", "coda_norm_epsilon",
            "leads_regularization_lambda", "leads_spectral_weight", "leads_norm_epsilon",
        ):
            value = float(getattr(self, name))
            is_epsilon = name.endswith("norm_epsilon")
            if not np.isfinite(value) or value < 0.0 or (is_epsilon and value == 0.0):
                raise ValueError(f"{name} must be finite and {'positive' if is_epsilon else 'nonnegative'}")
            if is_epsilon:
                min_epsilon = float(np.sqrt(np.finfo(np.float32).tiny))
                max_epsilon = float(np.sqrt(np.finfo(np.float32).max)) / 2.0
                if not min_epsilon <= value <= max_epsilon:
                    raise ValueError(f"{name} must have a finite, normal FP32 square")
