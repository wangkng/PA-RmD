from __future__ import annotations
import time
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from pa_rmd.config import Config
from pa_rmd.models.base import _BasePredictor
from pa_rmd.utils import project_simplex
from pa_rmd.models.siar.solver import (
    AffinePayoffGraph, ProductSimplexChart, affine_payoff_graph_max_abs,
    centered_payoff_coordinates, fit_payoff_conditioned_siar,
    validate_siar_solver_objective_scale,
)


class SIARPredictor(_BasePredictor):
    """Payoff-conditioned polynomial field with independently configurable FI/PC.

    Positive correlation uses known affine training games or bounded affine
    graphs estimated only from the supplied training states, payoffs and IDs.
    """

    label = "SIAR"
    key = "siar"
    color = "darkcyan"

    def __init__(self, shared_cfg: Config, **kwargs: Any):
        super().__init__(shared_cfg, **kwargs)
        self._fit_cfg = self._cfg()

    def _cfg(self, **overrides: Any) -> Config:
        cfg = super()._cfg(**overrides)
        for name, default in (
            ("siar_payoff_graph_source", "known"),
            ("siar_payoff_scale_mode", "observed_or_graph"),
            ("siar_certificate_tolerance", None),
            ("siar_project_simplex", True),
        ):
            if not hasattr(cfg, name):
                setattr(cfg, name, default)
        for name in (
            "siar_enforce_forward_invariance", "siar_enforce_positive_correlation",
            "siar_project_simplex",
        ):
            if not isinstance(getattr(cfg, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        if (isinstance(cfg.siar_degree, (bool, np.bool_))
                or not isinstance(cfg.siar_degree, (int, np.integer))
                or cfg.siar_degree < 1):
            raise ValueError("siar_degree must be a positive integer")
        if (not isinstance(cfg.siar_payoff_graph_source, str)
                or cfg.siar_payoff_graph_source not in {"known", "estimated"}):
            raise ValueError("siar_payoff_graph_source must be 'known' or 'estimated'")
        if (not isinstance(cfg.siar_payoff_scale_mode, str)
                or cfg.siar_payoff_scale_mode not in {"observed_or_graph", "observed"}):
            raise ValueError("siar_payoff_scale_mode must be 'observed_or_graph' or 'observed'")
        tolerance = cfg.siar_certificate_tolerance
        if tolerance is not None:
            if (isinstance(tolerance, (bool, np.bool_))
                    or not isinstance(tolerance, (int, float, np.integer, np.floating))
                    or not np.isfinite(tolerance) or tolerance < 0.0):
                raise ValueError("siar_certificate_tolerance must be None or finite and nonnegative")
            cfg.siar_certificate_tolerance = float(tolerance)
        cfg.siar_solver_objective_scale = validate_siar_solver_objective_scale(
            getattr(cfg, "siar_solver_objective_scale", 1.0)
        )
        if not isinstance(cfg.siar_solver_use_indirect, (bool, np.bool_)):
            raise ValueError("siar_solver_use_indirect must be boolean")
        if cfg.siar_solver_use_indirect and str(cfg.siar_solver).upper() != "SCS":
            raise ValueError("siar_solver_use_indirect=True requires siar_solver='SCS'")
        return cfg

    def _validate_fit_result(self, result: Any) -> None:
        """Apply an optional finite numerical certificate acceptance gate."""
        tolerance = self._fit_cfg.siar_certificate_tolerance
        if tolerance is None:
            return
        if result.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"{self.label} optimization failed with status {result.status!r}.")
        diagnostics = (
            result.objective_value, result.data_mse,
            result.max_constraint_violation, result.min_gram_eigenvalue,
        )
        if result.scaled_solver_objective_value is not None:
            diagnostics += (result.scaled_solver_objective_value,)
        if not np.all(np.isfinite(diagnostics)) or not np.all(
            np.isfinite(result.vector_field.ambient_coefficients)
        ):
            raise RuntimeError(f"{self.label} solver returned nonfinite coefficients or certificate diagnostics.")
        if (result.max_constraint_violation > tolerance
                or result.min_gram_eigenvalue < -tolerance):
            raise RuntimeError(
                f"{self.label} numerical certificate rejected: "
                f"max_constraint_violation={result.max_constraint_violation:.3e}, "
                f"min_gram_eigenvalue={result.min_gram_eigenvalue:.3e}, "
                f"tolerance={tolerance:.3e}."
            )

    def _fit_payoff_graphs(self, batch: Any, *, action_dim: int,
                           observed_payoff_scale: float) -> tuple[AffinePayoffGraph, ...]:
        self._siar_pc_graph_env_ids = ()
        self._siar_source_affine_payoff_estimation = None
        if not self._fit_cfg.siar_enforce_positive_correlation:
            return ()
        if self._fit_cfg.siar_payoff_graph_source == "known":
            self._siar_pc_graph_env_ids, graphs = self._training_payoff_graphs(
                batch, action_dim=action_dim,
            )
        else:
            from pa_rmd.models.siar.source_payoff import estimate_source_affine_payoff_graphs

            # Only the training observations and their environment IDs enter this QP.
            scale = self._observed_scale(observed_payoff_scale)
            env_ids, graphs, audit = estimate_source_affine_payoff_graphs(
                batch.x_current, batch.v_current, batch.env_id, payoff_scale=scale,
            )
            self._siar_pc_graph_env_ids = env_ids
            self._siar_source_affine_payoff_estimation = audit
        return graphs

    @staticmethod
    def _observed_scale(observed: float) -> float:
        if not np.isfinite(observed):
            raise ValueError("SIAR requires a finite observed payoff scale.")
        return observed if observed > 1e-12 else 1.0

    def _fit_payoff_scale(self, observed: float, graph: float) -> float:
        if self._fit_cfg.siar_payoff_scale_mode == "observed":
            scale = self._observed_scale(observed)
            if self._siar_source_affine_payoff_estimation is not None:
                tolerance = 1e-10 * max(1.0, scale)
                if not np.isfinite(graph) or graph > scale + tolerance:
                    raise ValueError("SIAR estimated graph exceeds the fixed observed payoff scale.")
                self._siar_source_affine_payoff_estimation["graph_scale_roundoff_excess"] = max(0.0, graph - scale)
            return scale
        scale = max(observed, graph)
        return scale if np.isfinite(scale) and scale > 1e-12 else 1.0

    @staticmethod
    def _training_payoff_graphs(
        batch: Any,
        *,
        action_dim: int,
    ) -> tuple[tuple[int, ...], tuple[AffinePayoffGraph, ...]]:
        payoff_games = getattr(batch, "payoff_games", None)
        if not payoff_games:
            raise ValueError(
                "SIAR positive correlation on training payoff graphs requires "
                "TrainBatch.payoff_games."
            )

        if batch.env_id is None:
            env_ids = tuple(sorted(int(env_id) for env_id in payoff_games))
        else:
            env_ids = tuple(
                sorted(int(env_id) for env_id in np.unique(np.asarray(batch.env_id)))
            )
        graphs: list[AffinePayoffGraph] = []
        for env_id in env_ids:
            if env_id not in payoff_games:
                raise ValueError(
                    "SIAR is missing the training payoff game for "
                    f"env_id={env_id}."
                )
            game = payoff_games[env_id]
            matrix = getattr(game, "matrix", None)
            if matrix is None:
                raise ValueError(
                    "SIAR graph-restricted positive correlation currently "
                    "requires affine training payoff games exposing `matrix`; "
                    f"env_id={env_id} has {type(game).__name__}."
                )
            matrix = np.asarray(matrix, dtype=np.float64)
            if matrix.shape != (int(action_dim), int(action_dim)):
                raise ValueError(
                    f"SIAR payoff game env_id={env_id} has matrix shape "
                    f"{matrix.shape}; expected ({action_dim}, {action_dim})."
                )
            intercept = getattr(game, "intercept", None)
            graphs.append(
                AffinePayoffGraph(
                    matrix=matrix,
                    intercept=(
                        None
                        if intercept is None
                        else np.asarray(intercept, dtype=np.float64)
                    ),
                )
            )
        return env_ids, tuple(graphs)

    def fit(self, batch: Any, *, seed: int) -> None:
        del seed
        fit_started = time.perf_counter()
        cfg = self._cfg()
        self._fit_cfg = cfg
        x_current = np.asarray(batch.x_current, dtype=np.float64)
        v_current = np.asarray(batch.v_current, dtype=np.float64)
        x_next = np.asarray(batch.x_next, dtype=np.float64)
        pair_dt = np.asarray(batch.pair_dt, dtype=np.float64).reshape(-1)
        if (
            x_current.ndim != 2
            or v_current.shape != x_current.shape
            or x_next.shape != x_current.shape
        ):
            raise ValueError(
                "SIAR expects x_current, v_current, and x_next with identical "
                f"2D shapes; got {x_current.shape}, {v_current.shape}, {x_next.shape}."
            )
        if (
            pair_dt.shape[0] != x_current.shape[0]
            or np.any(pair_dt <= 0.0)
            or not all(np.all(np.isfinite(values)) for values in (
                x_current, v_current, x_next, pair_dt,
            ))
            or x_current.shape[0] == 0
        ):
            raise ValueError(
                "SIAR requires finite nonempty observations and one positive pair_dt per observation; "
                f"got pair_dt={pair_dt.shape}, samples={x_current.shape[0]}."
            )

        action_dims = (int(x_current.shape[1]),)
        target_velocity = (x_next - x_current) / pair_dt[:, None]
        centered_reduced_v = centered_payoff_coordinates(
            v_current,
            chart=self._single_simplex_chart(action_dims[0]),
        )
        observed_payoff_scale = float(np.max(np.abs(centered_reduced_v)))
        payoff_graphs = self._fit_payoff_graphs(
            batch, action_dim=action_dims[0], observed_payoff_scale=observed_payoff_scale,
        )
        graph_payoff_scale = affine_payoff_graph_max_abs(
            self._single_simplex_chart(action_dims[0]),
            payoff_graphs,
        )
        payoff_scale = self._fit_payoff_scale(observed_payoff_scale, graph_payoff_scale)
        self._siar_observed_payoff_scale = observed_payoff_scale
        self._siar_graph_payoff_scale = graph_payoff_scale

        solver = str(cfg.siar_solver).upper()
        solver_options = (
            {
                "eps": float(cfg.siar_solver_eps),
                "max_iters": int(cfg.siar_solver_max_iters),
                "use_indirect": bool(cfg.siar_solver_use_indirect),
            }
            if solver == "SCS"
            else {}
        )
        result = fit_payoff_conditioned_siar(
            x_current,
            v_current,
            target_velocity,
            action_dims=action_dims,
            payoff_graphs=payoff_graphs,
            degree=int(cfg.siar_degree),
            payoff_scale=payoff_scale,
            payoff_domain_bound=float(cfg.siar_payoff_domain_margin),
            enforce_forward_invariance=bool(cfg.siar_enforce_forward_invariance),
            enforce_positive_correlation=bool(cfg.siar_enforce_positive_correlation),
            coefficient_regularization=float(cfg.siar_coefficient_regularization),
            solver_objective_scale=cfg.siar_solver_objective_scale,
            solver=solver,
            solver_options=solver_options,
        )
        self._validate_fit_result(result)
        self._siar_fit_wall_time = time.perf_counter() - fit_started
        self.model = result.vector_field
        self.params = {
            "coefficients": jnp.asarray(
                result.vector_field.ambient_coefficients,
                dtype=jnp.float32,
            )
        }
        self._siar_exponents = jnp.asarray(result.vector_field.exponents, dtype=jnp.int32)
        self._v_scale = float(result.vector_field.payoff_scale)
        self._v_scale_mode = (
            "centered_reduced_source_observation_max_abs"
            if cfg.siar_payoff_scale_mode == "observed" and cfg.siar_payoff_graph_source == "estimated"
            else "centered_reduced_observation_max_abs"
            if cfg.siar_payoff_scale_mode == "observed"
            else "centered_reduced_observation_or_train_graph_max_abs"
        )
        self._siar_result = result
        coefficients = np.asarray(result.vector_field.reduced_coefficients)
        active = int(np.sum(np.abs(coefficients) > 1e-10))
        total = int(coefficients.size)
        self._history = {
            "loss": [float(result.data_mse)],
            "parameter_counts": {
                "params_active": active,
                "params_shared": active,
                "params_env_active": 0,
                "params_total_stored": total,
                "features": int(len(result.vector_field.exponents)),
            },
        }
        self._rollout_batch_cache.clear()
        print(
            f"[{self.key}] features={len(result.vector_field.exponents)} "
            f"coefficients={active}/{total} data_mse={result.data_mse:.8e} "
            f"status={result.status} constraint_violation="
            f"{result.max_constraint_violation:.3e} "
            f"gram_min_eigenvalue={result.min_gram_eigenvalue:.3e} "
            f"payoff_scale={self._v_scale:.6g} "
            f"payoff_domain_bound={result.vector_field.payoff_domain_bound:.3g} "
            f"pc_domain={result.positive_correlation_domain} "
            f"pc_graphs={result.num_positive_correlation_graphs}",
            flush=True,
        )

    @staticmethod
    def _single_simplex_chart(size: int) -> ProductSimplexChart:
        return ProductSimplexChart((int(size),))

    def _conditional_coordinates(
        self,
        x: jnp.ndarray,
        v: jnp.ndarray,
    ) -> jnp.ndarray:
        centered_v = v - jnp.mean(v, axis=-1, keepdims=True)
        return jnp.concatenate(
            [x[:, :-1], centered_v[:, :-1] / float(self._v_scale)],
            axis=-1,
        )

    def _vector_field(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        coordinates = self._conditional_coordinates(x, v)
        features = jnp.prod(
            coordinates[:, None, :] ** self._siar_exponents[None, :, :],
            axis=-1,
        )
        return features @ self.params["coefficients"]

    def predict_step(
        self,
        x: jnp.ndarray,
        v: jnp.ndarray,
        dt: float | jnp.ndarray,
        env_id: int | jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        del env_id
        self._require_fit()
        x = jnp.asarray(x)
        v = jnp.asarray(v)
        dt = self._batch_dt(x, dt)
        field = self._vector_field(x, v)
        prediction = x + dt[:, None] * field
        return (jax.vmap(project_simplex)(prediction)
                if self._fit_cfg.siar_project_simplex else prediction)

    def payoff_domain_metrics(self, v: Any) -> dict[str, float]:
        self._require_fit()
        reduced = centered_payoff_coordinates(
            np.asarray(v, dtype=np.float64),
            self.model.chart,
        )
        normalized_abs = np.abs(reduced / float(self._v_scale))
        sample_max = np.max(normalized_abs, axis=1)
        bound = float(self.model.payoff_domain_bound)
        return {
            "siar_payoff_domain_max_abs": float(np.max(sample_max)),
            "siar_payoff_domain_violation_fraction": float(
                np.mean(sample_max > bound + 1e-8)
            ),
        }

    def config_summary(self) -> dict:
        summary = super().config_summary()
        cfg = self._summary_cfg()
        summary["siar_constraint_identity"] = {
            "tangency": "exact_reduced_output_construction",
            "forward_invariance": (
                ("affine-face-box-exact" if cfg.siar_degree == 1
                 else "sos_simplex_faces_cross_centered_payoff_box")
                if cfg.siar_enforce_forward_invariance else "disabled"
            ),
            "positive_correlation": (
                ("sos_training_affine_payoff_graphs"
                 if cfg.siar_payoff_graph_source == "known"
                 else "sos_source_only_bounded_estimated_affine_payoff_graphs")
                if cfg.siar_enforce_positive_correlation else "disabled"
            ),
        }
        summary.update({
            "siar_payoff_graph_source": cfg.siar_payoff_graph_source,
            "siar_payoff_scale_mode": cfg.siar_payoff_scale_mode,
            "siar_certificate_tolerance": cfg.siar_certificate_tolerance,
            "siar_project_simplex": bool(cfg.siar_project_simplex),
        })
        if cfg.siar_enforce_positive_correlation and cfg.siar_payoff_graph_source == "estimated":
            from pa_rmd.models.siar.source_payoff import source_affine_estimator_identity

            summary["siar_constraint_identity"]["source_payoff_estimator"] = source_affine_estimator_identity()
        if cfg.siar_certificate_tolerance is not None:
            summary["siar_constraint_identity"]["numerical_acceptance"] = {
                "allowed_solver_status": ["optimal", "optimal_inaccurate"],
                "max_constraint_violation": cfg.siar_certificate_tolerance,
                "min_gram_eigenvalue": -cfg.siar_certificate_tolerance,
                "finite_coefficients_and_diagnostics": True,
            }
        if self.model is not None:
            if self._siar_source_affine_payoff_estimation is not None:
                summary["siar_source_affine_payoff_estimation"] = self._siar_source_affine_payoff_estimation
            result = self._siar_result
            summary.update(
                {
                    "siar_degree": int(self.model.degree),
                    "siar_feature_count": int(len(self.model.exponents)),
                    "siar_payoff_domain_bound": float(
                        self.model.payoff_domain_bound
                    ),
                    "siar_observed_payoff_scale": (
                        self._siar_observed_payoff_scale
                    ),
                    "siar_training_graph_payoff_scale": (
                        self._siar_graph_payoff_scale
                    ),
                    "siar_enforce_forward_invariance": bool(
                        self._fit_cfg.siar_enforce_forward_invariance
                    ),
                    "siar_enforce_positive_correlation": bool(
                        self._fit_cfg.siar_enforce_positive_correlation
                    ),
                    "siar_positive_correlation_domain": (
                        result.positive_correlation_domain
                    ),
                    "siar_positive_correlation_graph_env_ids": list(
                        self._siar_pc_graph_env_ids
                    ),
                    "siar_num_positive_correlation_graphs": (
                        result.num_positive_correlation_graphs
                    ),
                    "siar_training_graph_payoff_domain_max_abs": (
                        result.training_graph_payoff_domain_max_abs
                    ),
                    "siar_solver": result.solver,
                    "siar_solver_use_indirect": bool(self._fit_cfg.siar_solver_use_indirect),
                    "siar_solver_status": result.status,
                    "siar_solve_time": result.solve_time,
                    "siar_fit_wall_time": self._siar_fit_wall_time,
                    "siar_num_scalar_variables": result.num_scalar_variables,
                    "siar_num_scalar_equalities": result.num_scalar_equalities,
                    "siar_num_scalar_inequalities": result.num_scalar_inequalities,
                    "siar_objective_value": result.objective_value,
                    "siar_solver_objective_scale": result.solver_objective_scale,
                    "siar_scaled_solver_objective_value": (
                        result.scaled_solver_objective_value
                        if result.scaled_solver_objective_value is not None
                        else result.solver_objective_scale * result.objective_value
                    ),
                    "siar_data_mse": result.data_mse,
                    "siar_max_constraint_violation": result.max_constraint_violation,
                    "siar_min_gram_eigenvalue": result.min_gram_eigenvalue,
                    "siar_num_psd_gram_matrices": result.num_psd_gram_matrices,
                    "siar_forward_invariance_certificate_kind": (
                        result.forward_invariance_certificate_kind
                    ),
                }
            )
        return summary
