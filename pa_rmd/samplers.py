from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pa_rmd.payoffs import (
    enumerate_cyclic_isomorphism_matrices,
    payoff_family_key_for_matrix,
    sample_congestion_affine_payoff,
)


@dataclass
class SampleResult:
    train_matrices: list[np.ndarray]
    test_matrices: list[np.ndarray]
    train_intercepts: list[np.ndarray | None]
    test_intercepts: list[np.ndarray | None]
    train_indices: list[int]
    test_indices: list[int]
    metadata: dict[str, Any]


SAMPLERS: dict[str, type[Any]] = {}


def from_name(name: str) -> type[Any]:
    key = payoff_sampling_mode_display(name)
    try:
        return SAMPLERS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown payoff sampler: {name}. Available: {sorted(SAMPLERS)}") from exc


def normalize_payoff_sampling_mode(value: str | None) -> str:
    if value is None:
        return "cyclic_isomorphism"
    key = str(value).strip().lower().replace("_", "-")
    aliases = {
        "cyclic-isomorphism": "cyclic_isomorphism",
        "cyclic": "cyclic_isomorphism",
        "isomorphism": "cyclic_isomorphism",
        "congestion": "congestion",
        "potential-congestion": "congestion",
        "pot-congestion": "congestion",
        "pot-cong": "congestion",
    }
    if key not in aliases:
        raise ValueError(
            "--payoff-sampling-mode must be cyclic-isomorphism or congestion "
            f"(got {value!r})"
        )
    return aliases[key]


def payoff_sampling_mode_display(mode: str) -> str:
    return {
        "cyclic_isomorphism": "cyclic-isomorphism",
        "congestion": "congestion",
    }[normalize_payoff_sampling_mode(mode)]






def _resolve_indices(
    num_payoff: int,
    num_train: int,
    train_indices: list[int] | None,
    test_indices: list[int] | str | None,
) -> tuple[list[int], list[int]]:
    num_payoff, num_train = int(num_payoff), int(num_train)
    if train_indices is None:
        if num_train < 1 or num_train > num_payoff:
            raise ValueError("num_payoff_train must be between 1 and num_payoff")
        if test_indices is None and num_train >= num_payoff:
            raise ValueError("num_payoff_train must be < num_payoff when --payoff-test is complement")
        train = list(range(num_train))
    else:
        train = [int(i) for i in train_indices]
        _validate_indices(train, num_payoff, "--payoff-train")
        if test_indices is None and len(train) >= num_payoff:
            raise ValueError("--payoff-train must leave at least one held-out test matrix")
    if test_indices is None or test_indices == "complement":
        test = [i for i in range(num_payoff) if i not in set(train)]
    elif test_indices == "all":
        test = list(range(num_payoff))
    else:
        test = [int(i) for i in test_indices]
        _validate_indices(test, num_payoff, "--payoff-test")
    if not test:
        raise ValueError("--payoff-test resolved to an empty set")
    return train, test


def _validate_indices(indices: list[int], total: int, label: str) -> None:
    if not indices:
        raise ValueError(f"{label} must contain at least one index")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{label} contains duplicate indices: {indices}")
    for idx in indices:
        if idx < 0 or idx >= total:
            raise ValueError(f"{label} index out of range: {idx} (valid: 0..{total - 1})")




def _select(matrices, tags, train, test):
    train_m = [np.asarray(matrices[i], dtype=np.float32) for i in train]
    test_m = [np.asarray(matrices[i], dtype=np.float32) for i in test]
    return train_m, test_m, [tags[i] for i in train], [tags[i] for i in test]




class CyclicSampler:
    name = "cyclic-isomorphism"

    def __init__(self, A_base):
        self.A_base = np.asarray(A_base, dtype=np.float32)

    def sample(self, num_train, num_test, *, seed, train_indices=None, test_indices=None) -> SampleResult:
        total = int(num_train) + int(num_test)
        train_idx, test_idx = _resolve_indices(total, int(num_train), train_indices, test_indices)
        A_base = self.A_base
        if payoff_family_key_for_matrix(A_base) != "zero_sum":
            raise ValueError("cyclic-isomorphism sampling requires a zero-sum base family")
        all_iso = enumerate_cyclic_isomorphism_matrices(A_base.shape[0])
        if total > len(all_iso):
            raise ValueError(f"cyclic-isomorphism supports at most {(len(all_iso))} payoff matrices")
        selected = [int(i) for i in np.random.default_rng(int(seed)).permutation(len(all_iso))[:total]]
        matrices, tags = [all_iso[i] for i in selected], selected
        train, test, train_tags, test_tags = _select(matrices, tags, train_idx, test_idx)
        return SampleResult(train, test, [None] * len(train), [None] * len(test), train_idx, test_idx,
                            {"train_tags": train_tags, "test_tags": test_tags})


class CongestionSampler:
    name = "congestion"

    def __init__(self, num_routes, incidence_matrix=None, target_frob=1.0):
        self.kwargs = dict(num_routes=num_routes, incidence_matrix=incidence_matrix, target_frob=target_frob)

    def sample(self, num_train, num_test, *, seed, train_indices=None, test_indices=None) -> SampleResult:
        total = int(num_train) + int(num_test)
        train_idx, test_idx = _resolve_indices(total, int(num_train), train_indices, test_indices)
        seeds = [int(seed) + i for i in range(total)]
        matrices, intercepts, metadata = [], [], []
        for s in seeds:
            P, q, meta = sample_congestion_affine_payoff(
                self.kwargs["num_routes"], s, self.kwargs.get("incidence_matrix"), self.kwargs.get("target_frob", 1.0)
            )
            matrices.append(P); intercepts.append(q); metadata.append(meta)
        train = [matrices[i] for i in train_idx]; test = [matrices[i] for i in test_idx]
        train_q = [intercepts[i] for i in train_idx]; test_q = [intercepts[i] for i in test_idx]
        return SampleResult(train, test, train_q, test_q, train_idx, test_idx, {
            "sampler": "congestion", "train_tags": [seeds[i] for i in train_idx],
            "test_tags": [seeds[i] for i in test_idx], "base_matrix": matrices[0],
            "base_intercept": intercepts[0], "train_metadata": [metadata[i] for i in train_idx],
            "test_metadata": [metadata[i] for i in test_idx], "base_metadata": metadata[0],
        })




SAMPLERS["cyclic-isomorphism"] = CyclicSampler
SAMPLERS["congestion"] = CongestionSampler
