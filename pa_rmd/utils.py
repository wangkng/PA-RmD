"""Simplex operations, deterministic seeds and output collection helpers."""
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp


def project_simplex(x: jnp.ndarray) -> jnp.ndarray:
    """
    Project x onto the probability simplex Delta_N.
    Uses the efficient sorting-based algorithm (O(N log N)).
    JAX-compatible via jax.lax.
    """
    n = x.shape[0]
    # Sort descending
    u = jnp.sort(x)[::-1]
    cssv = jnp.cumsum(u)
    rho_candidates = jnp.arange(1, n + 1, dtype=x.dtype)
    # Find largest rho such that u[rho-1] - (cssv[rho-1] - 1) / rho > 0
    cond = u - (cssv - 1.0) / rho_candidates
    # rho = last index where cond > 0
    rho = jnp.sum(cond > 0)
    rho = jnp.maximum(rho, 1)
    theta = (cssv[rho - 1] - 1.0) / rho.astype(x.dtype)
    return jnp.maximum(x - theta, 0.0)


def sample_simplex_interior(key: jax.Array, M: int, N: int, epsilon: float) -> jnp.ndarray:
    """
    Sample M points from {x in Delta_N : x_i > epsilon for all i}.
    Uses Dirichlet(alpha=1) then clips and renormalizes.
    """
    key, subkey = jax.random.split(key)
    # Dirichlet via Gamma sampling
    gamma_samples = jax.random.gamma(subkey, jnp.ones((M, N)))
    x = gamma_samples / gamma_samples.sum(axis=1, keepdims=True)
    # Clip to ensure interior margin
    x = jnp.clip(x, epsilon, 1.0)
    x = x / x.sum(axis=1, keepdims=True)
    return x


MAX_SEED = 2**31 - 1


def derive_seed(master_seed: int, *parts: object) -> int:
    """Derive a stable non-negative integer seed from a master seed and labels."""
    payload = "|".join(["DynamicsPredictionSeedV1", str(int(master_seed)), *(str(p) for p in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % MAX_SEED


def seed_map(master_seed: int, prefix: str, indices: list[int]) -> dict[int, int]:
    """Derive per-index seeds while preserving integer keys for metadata."""
    return {int(idx): derive_seed(master_seed, prefix, int(idx)) for idx in indices}


RESULT_SETTING_DIRS = {
    "synthetic-payoff-family": {f"{family}-k{dim}" for family in ("cyclic", "congestion") for dim in (3, 4)},
    "youtube-multicountry": {f"q={q}" for q in range(3, 10)},
}


def is_excluded_output(root: Path, candidate: Path) -> bool:
    """Skip nested archives and named runs when collecting default results.

    A dataset root selects its direct settings. Passing a named run or legacy
    directory explicitly still permits reading its own results.
    """
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return False
    parts = relative.parts
    if root.name in RESULT_SETTING_DIRS and parts and parts[0] not in RESULT_SETTING_DIRS[root.name]:
        return True
    if root.name == "outputs" and parts[:1] == ("legacy",):
        return True
    return any(parts[index:index + 2] == ("outputs", "legacy")
               for index in range(len(parts) - 1))
