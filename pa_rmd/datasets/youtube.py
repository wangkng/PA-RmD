"""Paper YouTube protocol: q source countries, held-out targets and S40 rollout."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from pa_rmd.config import Config
from pa_rmd.evaluate import TestBlock, TrainBatch
from pa_rmd.experiment_io import fingerprint, input_identity
from pa_rmd.paths import PROJECT_ROOT
from pa_rmd.utils import derive_seed


PAPER_SHAPE = (10, 122, 15)
SUPPORT_TRANSITIONS = 40


def _make_country_splits(
    countries: list[str],
    *,
    split_designs: str | list[str] | None = "k-train",
    target_country: str | None = "ALL",
    k_train_values: list[int] | None = None,
    data_seeds: list[int] | None = None,
    train_frac: float = 0.7,
) -> list[dict[str, Any]]:
    del train_frac  # The paper country split always uses all source transitions.
    designs = [split_designs] if isinstance(split_designs, str) else split_designs
    if designs != ["k-train"]:
        raise ValueError("The public YouTube protocol requires split_designs='k-train'.")
    if str(target_country).strip().upper() != "ALL":
        raise ValueError("The paper k-train protocol requires target_country='ALL'.")
    all_idx = list(range(len(countries)))
    splits = []
    for k in (k_train_values or [3]):
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 3 <= k <= 9 or k >= len(countries):
            raise ValueError("Paper YouTube q must be an integer from 3 to 9 with held-out countries.")
        for seed in (data_seeds or [123]):
            # Keep the original frozen k-train namespace and sampling order.
            split_seed = derive_seed(seed, "youtube_country_split", "k-train", int(k))
            train_idx = sorted(np.random.default_rng(split_seed).choice(all_idx, size=int(k), replace=False).tolist())
            splits.append({
                "design": "k-train", "label": f"ktrain_k{int(k)}_data_seed{int(seed)}",
                "train_country_idx": train_idx,
                "test_country_idx": [i for i in all_idx if i not in train_idx],
                "train_pair_mode": "all", "test_pair_mode": "all",
                "data_seed": int(seed), "train_frac": None, "split_seed": int(split_seed),
            })
    return splits


def _flatten_youtube_pairs(shares: np.ndarray, payoff: np.ndarray, countries: list[int], pairs: np.ndarray) -> dict[str, Any]:
    x_cur = shares[np.ix_(countries, pairs)]
    x_next = shares[np.ix_(countries, pairs + 1)]
    v_cur = payoff[np.ix_(countries, pairs)]
    n_countries, n_pairs, n_dim = x_cur.shape
    return {
        "x_current": jnp.asarray(x_cur.reshape(n_countries * n_pairs, n_dim), dtype=jnp.float32),
        "v_current": jnp.asarray(v_cur.reshape(n_countries * n_pairs, n_dim), dtype=jnp.float32),
        "x_next": jnp.asarray(x_next.reshape(n_countries * n_pairs, n_dim), dtype=jnp.float32),
        "pair_dt": jnp.ones((n_countries * n_pairs,), dtype=jnp.float32),
        "country_idx": np.repeat(np.asarray(countries, dtype=np.int32), n_pairs),
        "pair_idx": np.tile(np.asarray(pairs, dtype=np.int32), n_countries),
    }


def load(
    cfg: Config,
    *,
    npz_path: str | Path = PROJECT_ROOT / "data/YouTubeTrending/youtube_multicountry_common.npz",
    target_country: str | None = "ALL",
    split_designs: str | list[str] | None = "k-train",
    k_train_values: list[int] | None = None,
    data_seeds: list[int] | None = None,
    train_frac: float = 0.7,
    payoff_field: str = "payoff_log_views",
    payoff_scale: float = 1.0,
    split_index: int = 0,
    payoff_support_transitions: int = SUPPORT_TRANSITIONS,
    rollout_start_transition: int = SUPPORT_TRANSITIONS,
) -> tuple[TrainBatch, list[TestBlock], dict]:
    del cfg
    for name, value in (("payoff_support_transitions", payoff_support_transitions),
                        ("rollout_start_transition", rollout_start_transition)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value != SUPPORT_TRANSITIONS:
            raise ValueError(f"The public YouTube protocol requires {name}=40.")
    with np.load(npz_path, allow_pickle=True) as data:
        countries = [str(c) for c in data["countries"]]
        shares = np.asarray(data["shares"], dtype=np.float32)
        if payoff_field not in data.files:
            raise ValueError(f"Unknown payoff field {payoff_field!r}; available={[k for k in data.files if k.startswith('payoff_')]}")
        payoff = np.asarray(data[payoff_field], dtype=np.float32) * float(payoff_scale)
    if shares.shape != PAPER_SHAPE or payoff.shape != PAPER_SHAPE or len(countries) != 10:
        raise ValueError("The paper YouTube NPZ requires 10 countries, 122 days and 15 common categories.")
    if len(set(countries)) != len(countries):
        raise ValueError("YouTube country identifiers must be unique.")
    if not np.isfinite(shares).all() or not np.isfinite(payoff).all():
        raise ValueError("YouTube states and payoffs must be finite.")
    if np.any(shares < 0) or not np.allclose(shares.sum(axis=-1), 1.0, atol=1e-5):
        raise ValueError("YouTube states must be probability distributions.")
    dataset_content_signature = fingerprint(input_identity({
        "countries": countries, "shares": shares, "payoff": payoff,
    }))
    splits = _make_country_splits(countries, split_designs=split_designs,
        target_country=target_country, k_train_values=k_train_values,
        data_seeds=data_seeds, train_frac=train_frac)
    if not 0 <= split_index < len(splits):
        raise ValueError(f"split_index must be in [0, {len(splits)}).")
    split = splits[int(split_index)]
    pairs = np.arange(shares.shape[1] - 1, dtype=np.int32)
    train_raw = _flatten_youtube_pairs(shares, payoff, split["train_country_idx"], pairs)
    train = TrainBatch(x_current=train_raw["x_current"], v_current=train_raw["v_current"],
        x_next=train_raw["x_next"], pair_dt=train_raw["pair_dt"],
        env_id=jnp.asarray(train_raw["country_idx"], dtype=jnp.int32))
    blocks = []
    for country_idx in split["test_country_idx"]:
        raw = _flatten_youtube_pairs(shares, payoff, [country_idx], pairs)
        support = TrainBatch(x_current=raw["x_current"][:40], v_current=raw["v_current"][:40],
            x_next=raw["x_next"][:40], pair_dt=raw["pair_dt"][:40],
            env_id=jnp.full((40,), country_idx, dtype=jnp.int32))
        blocks.append(TestBlock(x_pairs=raw["x_current"], x_next=raw["x_next"],
            v_pairs=raw["v_current"], pair_dt=raw["pair_dt"], env_id=country_idx,
            group_label=f"country_{countries[country_idx]}",
            trajectories=[np.asarray(shares[country_idx, 40:], dtype=np.float32)],
            game=None, payoff_support=support,
            observed_x_pairs=raw["x_current"], observed_x_next=raw["x_next"],
            observed_v_pairs=raw["v_current"], observed_pair_dt=raw["pair_dt"]))
    return train, blocks, {
        "exp_name": "youtube_multicountry", "split": split, "countries": countries,
        "dataset_content_signature": dataset_content_signature,
        "payoff_field": payoff_field, "payoff_scale": float(payoff_scale),
        "payoff_support_transitions": 40,
        "num_splits": len(splits), "selected_split_index": int(split_index),
        "evaluation_protocol": "youtube-multicountry-full-one-step-rollout40-v2",
        "rollout_start_transition": 40,
    }
