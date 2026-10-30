from __future__ import annotations

from itertools import permutations
from typing import Any

import numpy as np


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def payoff_family_label_for_matrix(A) -> str:
    A = np.asarray(A, dtype=np.float32)
    if np.linalg.norm(A + A.T, ord="fro") < 1e-6:
        return "zeroSum"
    if np.linalg.norm(A - A.T, ord="fro") < 1e-6:
        return "potential"
    return "general"


def payoff_family_key_for_matrix(A) -> str:
    return {"zeroSum": "zero_sum", "potential": "potential"}.get(
        payoff_family_label_for_matrix(A), "general"
    )


def normalize_payoff_base_family(value: str | None) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower().replace("_", "-")
    if key in {"", "none", "default"}:
        return None
    aliases = {
        "zero-sum": "zero_sum",
        "zerosum": "zero_sum",
        "zero": "zero_sum",
        "antisymmetric": "zero_sum",
        "anti-symmetric": "zero_sum",
        "potential": "potential",
        "symmetric": "potential",
    }
    if key not in aliases:
        raise ValueError(f"--payoff-base-family must be one of: zero-sum, potential (got {value!r})")
    return aliases[key]


def make_fixed_potential_matrix(N: int = 3, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((int(N), int(N))).astype(np.float32)
    A = (M + M.T) / np.sqrt(2.0)
    frob = np.linalg.norm(A, ord="fro")
    if frob > 0:
        A = A / frob * np.sqrt(int(N))
    return A.astype(np.float32)


def make_fixed_zero_sum_matrix(N: int = 3, seed: int = 123) -> np.ndarray:
    _ = seed
    N = int(N)
    A = np.zeros((N, N), dtype=np.float32)
    if N == 2:
        A[0, 1], A[1, 0] = -1.0, 1.0
        return A
    for i in range(N):
        j = (i + 1) % N
        A[i, j], A[j, i] = -1.0, 1.0
    return A.astype(np.float32)


def make_payoff_base_matrix(family: str, N: int, seed: int) -> np.ndarray:
    if int(N) < 2:
        raise ValueError("--payoff-dim must be >= 2")
    if family == "zero_sum":
        return make_fixed_zero_sum_matrix(N=int(N), seed=int(seed))
    if family == "potential":
        return make_fixed_potential_matrix(N=int(N), seed=int(seed))
    raise ValueError(f"Unsupported payoff base family: {family}")


def canonical_payoff_matrix_for_config(config) -> np.ndarray:
    if config.game_type == "rps":
        return np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]], dtype=np.float32)
    if config.game_type == "custom_matrix" and config.payoff_matrix is not None:
        return np.asarray(config.payoff_matrix, dtype=np.float32)
    raise ValueError(f"Cannot derive canonical payoff matrix for game_type={config.game_type}")


def enumerate_cyclic_isomorphism_matrices(N: int) -> list[np.ndarray]:
    N = int(N)
    base = make_fixed_zero_sum_matrix(N=N)
    return [base[np.ix_([0] + list(p), [0] + list(p))].astype(np.float32) for p in permutations(range(1, N))]





def default_congestion_incidence_matrix(num_routes: int) -> np.ndarray:
    n = int(num_routes)
    if n < 2:
        raise ValueError("Congestion sampling requires payoff dimension N >= 2.")
    if n == 3:
        return np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    B = np.zeros((n, n), dtype=np.float32)
    for r in range(n):
        B[r, r], B[r, (r + 1) % n] = 1.0, 1.0
    return B


def sample_congestion_affine_payoff(
    num_routes: int,
    seed: int,
    incidence_matrix: np.ndarray | None = None,
    target_frob: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    B = default_congestion_incidence_matrix(num_routes) if incidence_matrix is None else np.asarray(incidence_matrix, dtype=np.float32)
    if B.ndim != 2 or B.shape[1] != int(num_routes):
        raise ValueError(f"incidence_matrix shape mismatch: expected (*, {int(num_routes)}), got {B.shape}")
    rng = np.random.default_rng(int(seed))
    a = rng.uniform(0.8, 2.2, size=(B.shape[0],)).astype(np.float32)
    b = rng.uniform(0.0, 0.3, size=(B.shape[0],)).astype(np.float32)
    raw = B.T @ np.diag(a) @ B
    frob = float(np.linalg.norm(raw, ord="fro"))
    target = float(target_frob)
    if target <= 0.0:
        raise ValueError("congestion target Frobenius norm must be > 0.")
    scale = target / frob if frob > 0.0 else 1.0
    P = (-scale * raw).astype(np.float32)
    q = (scale * (-(B.T @ b))).astype(np.float32)
    return P, q, {"model": "potential_congestion_affine", "seed": int(seed), "B": B, "a": a, "b": b, "P_raw_frob": frob, "P_target_frob": target, "scale": scale}
