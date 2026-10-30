from abc import ABC, abstractmethod
import jax.numpy as jnp


class RiemannianMetric(ABC):
    """Abstract base class for Riemannian metrics on the simplex."""

    @abstractmethod
    def g(self, x: jnp.ndarray) -> jnp.ndarray:
        """Metric tensor g(x). Shape (N, N). Must be SPD."""
        pass

    @abstractmethod
    def g_inv(self, x: jnp.ndarray) -> jnp.ndarray:
        """Inverse metric tensor g^{-1}(x). Shape (N, N)."""
        pass


class ReplicatorMetric(RiemannianMetric):
    """Shahshahani metric: g = diag(1/x), g_inv = diag(x).

    During RK4 intermediate stages x can briefly go slightly negative; we clip
    to a small positive floor in g/g_inv so the metric stays well-defined.
    """

    def g(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.diag(1.0 / jnp.maximum(x, 1e-8))

    def g_inv(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.diag(jnp.maximum(x, 0.0))


class IdentityMetric(RiemannianMetric):
    """Euclidean metric: g = I, g_inv = I."""

    def g(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.eye(x.shape[0], dtype=x.dtype)

    def g_inv(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.eye(x.shape[0], dtype=x.dtype)


def build_metric(config) -> RiemannianMetric:
    if config.metric_type == "replicator":
        return ReplicatorMetric()
    if config.metric_type == "projection":
        return IdentityMetric()
    raise ValueError(f"Unsupported metric-backed dynamics: {config.metric_type!r}.")
