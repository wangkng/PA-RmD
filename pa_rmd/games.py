from abc import ABC, abstractmethod
from typing import Optional
import jax.numpy as jnp


class PopulationGame(ABC):
    """Abstract base class for population games."""

    @abstractmethod
    def payoff(self, x: jnp.ndarray) -> jnp.ndarray:
        """v(x): population state -> payoff vector. Shape (N,)."""
        pass

    @property
    @abstractmethod
    def N(self) -> int:
        pass


class AffineGame(PopulationGame):
    """v(x) = P x + q for a given payoff operator."""

    def __init__(self, P: jnp.ndarray, q: Optional[jnp.ndarray] = None):
        self._P = P
        self._N = P.shape[0]
        if q is None:
            q = jnp.zeros((self._N,), dtype=P.dtype)
        else:
            q = jnp.asarray(q, dtype=P.dtype)
            if q.ndim != 1 or q.shape[0] != self._N:
                raise ValueError(
                    f"AffineGame intercept must have shape ({self._N},); got {tuple(q.shape)}"
                )
        self._q = q

    def payoff(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._P @ x + self._q

    @property
    def N(self) -> int:
        return self._N

    @property
    def matrix(self) -> jnp.ndarray:
        return self._P

    @property
    def intercept(self) -> jnp.ndarray:
        return self._q


class MatrixGame(AffineGame):
    """v(x) = Ax for a given payoff matrix A."""

    def __init__(self, A: jnp.ndarray):
        super().__init__(A, None)


class RPS(MatrixGame):
    """Rock-Paper-Scissors with antisymmetric payoff matrix."""

    def __init__(self):
        A = jnp.array(
            [[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=jnp.float32
        )
        super().__init__(A)


def build_game(config) -> PopulationGame:
    if config.game_type == "rps":
        return RPS()
    elif config.game_type == "custom_matrix":
        assert config.payoff_matrix is not None
        P = jnp.array(config.payoff_matrix, dtype=jnp.float32)
        q_raw = getattr(config, "payoff_intercept", None)
        if q_raw is None:
            return MatrixGame(P)
        q = jnp.array(q_raw, dtype=jnp.float32)
        return AffineGame(P, q)
    else:
        raise ValueError(f"Unknown game_type: {config.game_type}")
