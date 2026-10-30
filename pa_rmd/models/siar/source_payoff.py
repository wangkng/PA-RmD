"""Source-only bounded affine proxy graphs for SIAR positive correlation.

These graphs define the PC certificate domain; they never replace observed
training payoffs or the independent target kernel used for rollout.
"""
from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np

from pa_rmd.models.siar.solver import (
    AffinePayoffGraph, ProductSimplexChart, centered_payoff_coordinates,
)


RIDGE = 1e-6
BOUND_TOLERANCE = 1e-10
SOLVER_OPTIONS = {"max_iter": 200, "tol_gap_abs": 1e-10,
                  "tol_gap_rel": 1e-10, "tol_feas": 1e-10}


def source_affine_estimator_identity() -> dict[str, Any]:
    """Fixed, prefit identity for the estimator and its information boundary."""
    return {
        "estimator": "source_only_bounded_centered_affine_vertex_qp_v1",
        "inputs": ["TrainBatch.x_current", "TrainBatch.v_current", "TrainBatch.env_id"],
        "environment_fit": "independent_per_source_env_id",
        "payoff_scale": "all_source_centered_reduced_observed_max_abs_with_original_FI_zero_fallback",
        "normalization": "float64_centered_payoff_coordinates_then_restore_last_coordinate",
        "parameterization": "v_centered(x)/scale=A@x; columns_are_vertex_payoffs_including_affine_intercept",
        "centering": "each_column_sums_to_zero; last_row=-sum(first_N_minus_1_rows)",
        "independent_vertex_bound": [-1.0, 1.0],
        "objective": "mean_source_sample_ambient_squared_L2_error+ridge*sum_squares(full_A)",
        "ridge": RIDGE,
        "solver": "CLARABEL",
        "solver_options": dict(SOLVER_OPTIONS),
        "bound_numerical_tolerance": BOUND_TOLERANCE,
        "bound_correction": "only_tolerance_level_clip_then_rebuild_last_row; record_actual_postclip_objective",
        "use": "PC_domain_only; observed_training_payoffs_and_target_kernel_unchanged",
        "certificate_scope": "estimated_source_graphs_only; no_true_or_held_out_graph_guarantee",
    }


def estimate_source_affine_payoff_graphs(
    x_current: Any, v_current: Any, env_id: Any, *, payoff_scale: float,
) -> tuple[tuple[int, ...], tuple[AffinePayoffGraph, ...], dict[str, Any]]:
    """Fit each source graph using fixed scale and a centered, bounded QP."""
    x = np.array(x_current, dtype=np.float64, copy=True)
    v = np.array(v_current, dtype=np.float64, copy=True)
    if (x.ndim != 2 or v.shape != x.shape or x.shape[0] == 0 or x.shape[1] < 2
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(v))):
        raise ValueError("Source affine payoff estimation requires finite, nonempty matching state/payoff matrices.")
    if (np.any(x < -1e-6) or np.any(x > 1.0 + 1e-6)
            or not np.allclose(np.sum(x, axis=1), 1.0, atol=1e-6, rtol=0.0)):
        raise ValueError("Source affine payoff states must lie in the simplex.")
    if not np.isfinite(payoff_scale) or payoff_scale <= 0.0:
        raise ValueError("Source affine payoff estimation requires a finite positive fixed scale.")
    if env_id is None:
        raise ValueError("Source affine payoff estimation requires one source env_id per observation.")
    ids = np.asarray(env_id)
    if (ids.shape != (x.shape[0],) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0)):
        raise ValueError("Source affine payoff env_id must be a nonnegative integer vector matching observations.")
    source_env_ids = tuple(int(value) for value in np.unique(ids))
    n = x.shape[1]
    reduced = centered_payoff_coordinates(v, ProductSimplexChart((n,))) / payoff_scale
    target = np.concatenate((reduced, -np.sum(reduced, axis=1, keepdims=True)), axis=1)
    if not np.all(np.isfinite(target)):
        raise ValueError("Source affine payoff normalization produced nonfinite observations.")
    graphs = []
    environments = []
    for source_env_id in source_env_ids:
        selected = ids == source_env_id
        source_x, source_v = x[selected], target[selected]
        independent = cp.Variable((n - 1, n))
        full_matrix = cp.vstack((independent, -cp.sum(independent, axis=0, keepdims=True)))
        mse = cp.sum_squares(source_x @ full_matrix.T - source_v) / len(source_x)
        problem = cp.Problem(cp.Minimize(mse + RIDGE * cp.sum_squares(full_matrix)),
                             [independent >= -1.0, independent <= 1.0])
        problem.solve(solver="CLARABEL", **SOLVER_OPTIONS)
        if problem.status not in {"optimal", "optimal_inaccurate"} or independent.value is None:
            raise RuntimeError(f"Source affine payoff QP env_id={source_env_id} failed: {problem.status}.")
        raw = np.asarray(independent.value, dtype=np.float64)
        if not np.all(np.isfinite(raw)) or not np.isfinite(problem.value):
            raise RuntimeError(f"Source affine payoff QP env_id={source_env_id} returned nonfinite values.")
        violation = max(0.0, float(np.max(np.abs(raw))) - 1.0)
        if violation > BOUND_TOLERANCE:
            raise RuntimeError(f"Source affine payoff QP env_id={source_env_id} violates vertex bounds: {violation}.")
        bounded = np.clip(raw, -1.0, 1.0)
        normalized = np.concatenate((bounded, -np.sum(bounded, axis=0, keepdims=True)), axis=0)
        centering_residual = float(np.max(np.abs(np.sum(normalized, axis=0))))
        if (not np.all(np.isfinite(normalized)) or centering_residual > BOUND_TOLERANCE
                or np.max(np.abs(normalized[:-1])) > 1.0):
            raise RuntimeError(f"Source affine payoff QP env_id={source_env_id} failed centering/bound validation.")
        physical = payoff_scale * normalized
        if not np.all(np.isfinite(physical)):
            raise RuntimeError(f"Source affine payoff graph env_id={source_env_id} overflowed at the fixed scale.")
        residual = source_x @ normalized.T - source_v
        actual_mse = float(np.mean(np.sum(residual ** 2, axis=1)))
        actual_ridge = float(RIDGE * np.sum(normalized ** 2))
        if not np.all(np.isfinite((actual_mse, actual_ridge))):
            raise RuntimeError(f"Source affine payoff QP env_id={source_env_id} has nonfinite residual/objective.")
        graphs.append(AffinePayoffGraph(matrix=physical, intercept=None))
        environments.append({
            "env_id": source_env_id, "train_row_count": int(len(source_x)),
            "solver_status": problem.status, "solver_objective_value": float(problem.value),
            "normalized_vertex_payoffs": normalized.tolist(),
            "payoff_graph_matrix": physical.tolist(), "payoff_graph_intercept": None,
            "postclip_ambient_squared_L2_mse": actual_mse,
            "postclip_ridge_penalty": actual_ridge,
            "postclip_objective_value": actual_mse + actual_ridge,
            "independent_vertex_max_abs": float(np.max(np.abs(normalized[:-1]))),
            "column_centering_max_abs": centering_residual,
            "preclip_bound_violation": violation,
            "bound_clip_max_abs": float(np.max(np.abs(raw - bounded))),
            "bound_clip_element_count": int(np.count_nonzero(raw != bounded)),
        })
    audit = {"identity": source_affine_estimator_identity(), "payoff_scale": float(payoff_scale),
             "source_env_ids": list(source_env_ids), "total_train_row_count": int(len(x)),
             "environments": environments}
    return source_env_ids, tuple(graphs), audit
