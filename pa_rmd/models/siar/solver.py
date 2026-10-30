from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from numbers import Real
from typing import Mapping, Sequence

import cvxpy as cp
import numpy as np
from scipy import sparse


Exponent = tuple[int, ...]
NumericPolynomial = dict[Exponent, float]
ExpressionPolynomial = dict[Exponent, cp.Expression]


@dataclass(frozen=True)
class AffinePayoffGraph:
    """Affine payoff graph ``v(x) = matrix @ x + intercept`` used by SIAR PC."""

    matrix: np.ndarray
    intercept: np.ndarray | None = None


@lru_cache(maxsize=128)
def monomial_exponents(num_variables: int, degree: int) -> tuple[Exponent, ...]:
    """Return graded-lexicographic monomial exponents of total degree <= degree."""
    if num_variables < 0:
        raise ValueError("num_variables must be nonnegative.")
    if degree < 0:
        return ()
    if num_variables == 0:
        return ((),)

    # Enumerate only weak compositions of the requested total degree. Iterating
    # each leading power in ascending order preserves the old Cartesian-product
    # filter's exact ordering without visiting (degree + 1)**num_variables terms.
    exponents: list[Exponent] = []
    powers = [0] * num_variables

    def append_compositions(variable_index: int, remaining: int) -> None:
        if variable_index == num_variables - 1:
            powers[variable_index] = remaining
            exponents.append(tuple(powers))
            return
        for power in range(remaining + 1):
            powers[variable_index] = power
            append_compositions(variable_index + 1, remaining - power)

    for total_degree in range(degree + 1):
        append_compositions(0, total_degree)
    return tuple(exponents)


def evaluate_monomials(values: np.ndarray, exponents: Sequence[Exponent]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError(f"Expected one- or two-dimensional values, got shape {values.shape}.")
    if exponents and values.shape[1] != len(exponents[0]):
        raise ValueError(
            f"Values have dimension {values.shape[1]}, but monomials use {len(exponents[0])} variables."
        )

    features = np.ones((values.shape[0], len(exponents)), dtype=np.float64)
    for feature_index, exponent in enumerate(exponents):
        for variable_index, power in enumerate(exponent):
            if power:
                features[:, feature_index] *= values[:, variable_index] ** power
    return features[0] if single else features


def _constant_polynomial(num_variables: int, value: float) -> NumericPolynomial:
    return {(0,) * num_variables: float(value)}


def _variable_polynomial(num_variables: int, variable_index: int) -> NumericPolynomial:
    exponent = [0] * num_variables
    exponent[variable_index] = 1
    return {tuple(exponent): 1.0}


def _numeric_poly_add(*polynomials: Mapping[Exponent, float]) -> NumericPolynomial:
    result: NumericPolynomial = {}
    for polynomial in polynomials:
        for exponent, coefficient in polynomial.items():
            result[exponent] = result.get(exponent, 0.0) + float(coefficient)
    return {
        exponent: coefficient
        for exponent, coefficient in result.items()
        if abs(coefficient) > 1e-14
    }


def _numeric_poly_scale(polynomial: Mapping[Exponent, float], scale: float) -> NumericPolynomial:
    return {
        exponent: float(scale) * float(coefficient)
        for exponent, coefficient in polynomial.items()
        if abs(float(scale) * float(coefficient)) > 1e-14
    }


def _numeric_poly_multiply(
    left: Mapping[Exponent, float],
    right: Mapping[Exponent, float],
) -> NumericPolynomial:
    result: NumericPolynomial = {}
    for left_exponent, left_coefficient in left.items():
        for right_exponent, right_coefficient in right.items():
            exponent = tuple(a + b for a, b in zip(left_exponent, right_exponent))
            result[exponent] = (
                result.get(exponent, 0.0)
                + float(left_coefficient) * float(right_coefficient)
            )
    return {
        exponent: coefficient
        for exponent, coefficient in result.items()
        if abs(coefficient) > 1e-14
    }


def _numeric_poly_power(
    polynomial: Mapping[Exponent, float],
    power: int,
    *,
    num_variables: int,
) -> NumericPolynomial:
    if int(power) < 0:
        raise ValueError("Polynomial powers must be nonnegative.")
    result = _constant_polynomial(num_variables, 1.0)
    for _ in range(int(power)):
        result = _numeric_poly_multiply(result, polynomial)
    return result








def _expression_poly_compose_numeric(
    expression_polynomial: _CoefficientPolynomial,
    substitutions: Sequence[NumericPolynomial],
    *,
    num_output_variables: int,
) -> ExpressionPolynomial:
    """Compose internal batched coefficients with numeric polynomials."""
    source_exponents, source_coefficients = _polynomial_coefficient_vector(
        expression_polynomial
    )
    output_exponents, coefficient_map = _numeric_composition_map(
        source_exponents,
        tuple(tuple(substitution.items()) for substitution in substitutions),
        num_output_variables,
    )
    if not output_exponents:
        return {}
    return _coefficient_vector_polynomial(
        output_exponents, coefficient_map @ source_coefficients
    )


def _polynomial_degree(polynomial: Mapping[Exponent, object]) -> int:
    return max((sum(exponent) for exponent in polynomial), default=0)


@dataclass(frozen=True)
class ProductSimplexChart:
    """Independent-coordinate chart for a product of probability simplices."""

    action_dims: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.action_dims or any(int(size) < 2 for size in self.action_dims):
            raise ValueError("Each player must have at least two actions.")

    @property
    def ambient_dim(self) -> int:
        return int(sum(self.action_dims))

    @property
    def reduced_dim(self) -> int:
        return int(sum(size - 1 for size in self.action_dims))

    @property
    def player_slices(self) -> tuple[slice, ...]:
        slices: list[slice] = []
        start = 0
        for size in self.action_dims:
            slices.append(slice(start, start + size))
            start += size
        return tuple(slices)

    @property
    def reduced_player_slices(self) -> tuple[slice, ...]:
        slices: list[slice] = []
        start = 0
        for size in self.action_dims:
            width = size - 1
            slices.append(slice(start, start + width))
            start += width
        return tuple(slices)

    def ambient_to_reduced(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float64)
        single = states.ndim == 1
        if single:
            states = states[None, :]
        if states.ndim != 2 or states.shape[1] != self.ambient_dim:
            raise ValueError(
                f"Expected ambient states with final dimension {self.ambient_dim}, got {states.shape}."
            )
        pieces = [
            states[:, player_slice.start : player_slice.stop - 1]
            for player_slice in self.player_slices
        ]
        reduced = np.concatenate(pieces, axis=1)
        return reduced[0] if single else reduced

    def ambient_coordinate_polynomials(self) -> tuple[NumericPolynomial, ...]:
        variables = self.reduced_dim
        coordinates: list[NumericPolynomial] = []
        for reduced_slice in self.reduced_player_slices:
            independent = []
            for variable_index in range(reduced_slice.start, reduced_slice.stop):
                polynomial = _variable_polynomial(variables, variable_index)
                coordinates.append(polynomial)
                independent.append(polynomial)
            last = _constant_polynomial(variables, 1.0)
            for polynomial in independent:
                last = _numeric_poly_add(last, _numeric_poly_scale(polynomial, -1.0))
            coordinates.append(last)
        return tuple(coordinates)

    def domain_generators(self) -> tuple[NumericPolynomial, ...]:
        # In this chart, nonnegativity of every ambient action probability
        # exactly describes the product of simplices.
        return self.ambient_coordinate_polynomials()




def evaluate_numeric_polynomial(
    polynomial: Mapping[Exponent, float],
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    exponents = tuple(polynomial)
    if not exponents:
        return np.zeros(values.shape[:-1], dtype=np.float64)
    features = evaluate_monomials(values, exponents)
    coefficients = np.asarray([polynomial[exponent] for exponent in exponents], dtype=np.float64)
    return features @ coefficients


def centered_payoff_coordinates(
    payoffs: np.ndarray,
    chart: ProductSimplexChart,
) -> np.ndarray:
    """Remove payoff translations and return independent coordinates per player."""
    payoffs = np.asarray(payoffs, dtype=np.float64)
    single = payoffs.ndim == 1
    if single:
        payoffs = payoffs[None, :]
    if payoffs.ndim != 2 or payoffs.shape[1] != chart.ambient_dim:
        raise ValueError(
            f"Expected ambient payoffs with final dimension {chart.ambient_dim}, "
            f"got {payoffs.shape}."
        )

    pieces = []
    for player_slice in chart.player_slices:
        player_payoffs = payoffs[:, player_slice]
        centered = player_payoffs - np.mean(player_payoffs, axis=1, keepdims=True)
        pieces.append(centered[:, :-1])
    reduced = np.concatenate(pieces, axis=1)
    return reduced[0] if single else reduced


def _normalized_centered_affine_payoff_polynomials(
    chart: ProductSimplexChart,
    graph: AffinePayoffGraph,
    *,
    payoff_scale: float,
) -> tuple[NumericPolynomial, ...]:
    """Return the centered, normalized ambient payoff graph in state coordinates."""
    matrix = np.asarray(graph.matrix, dtype=np.float64)
    if matrix.shape != (chart.ambient_dim, chart.ambient_dim):
        raise ValueError(
            "Affine payoff graph matrix must have shape "
            f"({chart.ambient_dim}, {chart.ambient_dim}); got {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Affine payoff graph matrix must contain only finite values.")

    if graph.intercept is None:
        intercept = np.zeros((chart.ambient_dim,), dtype=np.float64)
    else:
        intercept = np.asarray(graph.intercept, dtype=np.float64)
    if intercept.shape != (chart.ambient_dim,):
        raise ValueError(
            "Affine payoff graph intercept must have shape "
            f"({chart.ambient_dim},); got {intercept.shape}."
        )
    if not np.all(np.isfinite(intercept)):
        raise ValueError("Affine payoff graph intercept must contain only finite values.")

    state_coordinates = chart.ambient_coordinate_polynomials()
    payoff_polynomials: list[NumericPolynomial] = []
    for payoff_index in range(chart.ambient_dim):
        polynomial = _constant_polynomial(
            chart.reduced_dim,
            float(intercept[payoff_index]),
        )
        for state_index, state_polynomial in enumerate(state_coordinates):
            coefficient = float(matrix[payoff_index, state_index])
            if coefficient:
                polynomial = _numeric_poly_add(
                    polynomial,
                    _numeric_poly_scale(state_polynomial, coefficient),
                )
        payoff_polynomials.append(polynomial)

    centered_polynomials: list[NumericPolynomial] = []
    for player_slice in chart.player_slices:
        player_payoffs = payoff_polynomials[player_slice]
        mean_payoff: NumericPolynomial = {}
        for polynomial in player_payoffs:
            mean_payoff = _numeric_poly_add(
                mean_payoff,
                _numeric_poly_scale(polynomial, 1.0 / len(player_payoffs)),
            )
        for polynomial in player_payoffs:
            centered_polynomials.append(
                _numeric_poly_scale(
                    _numeric_poly_add(
                        polynomial,
                        _numeric_poly_scale(mean_payoff, -1.0),
                    ),
                    1.0 / float(payoff_scale),
                )
            )
    return tuple(centered_polynomials)


def affine_payoff_graph_max_abs(
    chart: ProductSimplexChart,
    payoff_graphs: Sequence[AffinePayoffGraph],
) -> float:
    """Exact centered independent-payoff maximum over affine graph domains."""
    pure_profiles = product(
        *(range(action_count) for action_count in chart.action_dims)
    )
    reduced_vertices: list[np.ndarray] = []
    for profile in pure_profiles:
        ambient_vertex = np.zeros((chart.ambient_dim,), dtype=np.float64)
        for player_slice, action in zip(chart.player_slices, profile):
            ambient_vertex[player_slice.start + int(action)] = 1.0
        reduced_vertices.append(chart.ambient_to_reduced(ambient_vertex))
    vertices = np.stack(reduced_vertices, axis=0)

    independent_indices: list[int] = []
    for player_slice in chart.player_slices:
        independent_indices.extend(
            range(player_slice.start, player_slice.stop - 1)
        )

    maximum = 0.0
    for graph in payoff_graphs:
        centered_payoffs = _normalized_centered_affine_payoff_polynomials(
            chart,
            graph,
            payoff_scale=1.0,
        )
        for ambient_index in independent_indices:
            values = evaluate_numeric_polynomial(
                centered_payoffs[ambient_index],
                vertices,
            )
            maximum = max(maximum, float(np.max(np.abs(values))))
    return maximum


def _lift_numeric_polynomial(
    polynomial: Mapping[Exponent, float],
    *,
    trailing_variables: int,
) -> NumericPolynomial:
    return {
        tuple(exponent) + (0,) * int(trailing_variables): float(coefficient)
        for exponent, coefficient in polynomial.items()
    }




def _payoff_box_generators(
    chart: ProductSimplexChart,
    bound: float,
) -> tuple[NumericPolynomial, ...]:
    num_variables = 2 * chart.reduced_dim
    generators = []
    for reduced_index in range(chart.reduced_dim):
        variable = _variable_polynomial(
            num_variables,
            chart.reduced_dim + reduced_index,
        )
        generators.append(
            _numeric_poly_add(
                _constant_polynomial(num_variables, float(bound) ** 2),
                _numeric_poly_scale(
                    _numeric_poly_multiply(variable, variable),
                    -1.0,
                ),
            )
        )
    return tuple(generators)


class _CoefficientPolynomial(dict[Exponent, cp.Expression]):
    """Internal polynomial mapping with its batched affine coefficients."""

    def __init__(self, exponents: Sequence[Exponent], coefficients: cp.Expression):
        self.exponents = tuple(exponents)
        self.coefficients = coefficients
        super().__init__(
            (exponent, coefficients[index])
            for index, exponent in enumerate(self.exponents)
        )


def _coefficient_vector_polynomial(
    exponents: Sequence[Exponent],
    coefficients: cp.Expression,
) -> _CoefficientPolynomial:
    return _CoefficientPolynomial(exponents, coefficients)


@lru_cache(maxsize=32)
def _gram_coefficient_map(
    num_variables: int,
    basis_degree: int,
) -> tuple[tuple[Exponent, ...], sparse.csr_matrix]:
    """Map a full row-major Gram matrix to coefficients of z.T @ Gram @ z."""
    basis = monomial_exponents(num_variables, basis_degree)
    exponents = monomial_exponents(num_variables, 2 * basis_degree)
    exponent_rows = {exponent: index for index, exponent in enumerate(exponents)}
    rows = []
    for left_exponent in basis:
        for right_exponent in basis:
            exponent = tuple(a + b for a, b in zip(left_exponent, right_exponent))
            rows.append(exponent_rows[exponent])
    width = len(basis) ** 2
    coefficient_map = sparse.csr_matrix(
        (np.ones(width), (rows, np.arange(width))),
        shape=(len(exponents), width),
    )
    return exponents, coefficient_map


@lru_cache(maxsize=128)
def _numeric_multiplier_map(
    exponents: tuple[Exponent, ...],
    generator_terms: tuple[tuple[Exponent, float], ...],
) -> tuple[tuple[Exponent, ...], sparse.csr_matrix]:
    """Map polynomial coefficients through multiplication by a fixed generator."""
    products = [
        (tuple(a + b for a, b in zip(exponent, generator_exponent)), index, coefficient)
        for index, exponent in enumerate(exponents)
        for generator_exponent, coefficient in generator_terms
    ]
    output_exponents = tuple(
        sorted({exponent for exponent, _, _ in products}, key=lambda exp: (sum(exp), exp))
    )
    exponent_rows = {exponent: index for index, exponent in enumerate(output_exponents)}
    coefficient_map = sparse.csr_matrix(
        (
            [coefficient for _, _, coefficient in products],
            (
                [exponent_rows[exponent] for exponent, _, _ in products],
                [index for _, index, _ in products],
            ),
        ),
        shape=(len(output_exponents), len(exponents)),
    )
    return output_exponents, coefficient_map


@lru_cache(maxsize=32)
def _numeric_composition_map(
    exponents: tuple[Exponent, ...],
    substitution_terms: tuple[tuple[tuple[Exponent, float], ...], ...],
    num_output_variables: int,
) -> tuple[tuple[Exponent, ...], sparse.csr_matrix]:
    """Compose coefficients numerically before creating a CVXPY expression.

    Every output component uses the same substitution map on a payoff graph.
    Retain the scalar composition's term order and numeric pruning, including
    terms whose *expression* coefficients later cancel to zero: their support
    still determines the declared polynomial and certificate degrees.
    """
    substitutions = tuple(dict(terms) for terms in substitution_terms)
    powers: dict[tuple[int, int], NumericPolynomial] = {}
    products: list[tuple[Exponent, int, float]] = []
    for source_index, source_exponent in enumerate(exponents):
        if len(source_exponent) != len(substitutions):
            raise ValueError(
                "Polynomial substitution dimension mismatch: "
                f"exponent has {len(source_exponent)} variables, "
                f"substitutions has {len(substitutions)}."
            )
        monomial = _constant_polynomial(num_output_variables, 1.0)
        for substitution_index, power in enumerate(source_exponent):
            if power:
                key = (substitution_index, power)
                if key not in powers:
                    powers[key] = _numeric_poly_power(
                        substitutions[substitution_index], power,
                        num_variables=num_output_variables,
                    )
                monomial = _numeric_poly_multiply(monomial, powers[key])
        products.extend(
            (exponent, source_index, coefficient)
            for exponent, coefficient in monomial.items()
        )
    output_exponents = tuple(dict.fromkeys(exponent for exponent, _, _ in products))
    output_rows = {exponent: index for index, exponent in enumerate(output_exponents)}
    coefficient_map = sparse.csr_matrix(
        (
            [coefficient for _, _, coefficient in products],
            (
                [output_rows[exponent] for exponent, _, _ in products],
                [index for _, index, _ in products],
            ),
        ),
        shape=(len(output_exponents), len(exponents)),
    )
    return output_exponents, coefficient_map


def _expression_poly_sum_products_numeric(
    terms: Sequence[tuple[Mapping[Exponent, cp.Expression | float], NumericPolynomial]],
) -> ExpressionPolynomial:
    """Sum polynomial products using one sparse affine coefficient expression."""
    blocks = []
    output_rows: dict[Exponent, int] = {}
    for polynomial, generator in terms:
        if not polynomial or not generator:
            continue
        exponents, coefficients = _polynomial_coefficient_vector(polynomial)
        product_exponents, coefficient_map = _numeric_multiplier_map(
            exponents, tuple(generator.items())
        )
        # Match the support/insertion order of the scalar multiply-and-add.
        # Do not remove zero entries created by numeric or symbolic cancellation.
        for exponent in exponents:
            for generator_exponent in generator:
                product_exponent = tuple(a + b for a, b in zip(exponent, generator_exponent))
                if product_exponent not in output_rows:
                    output_rows[product_exponent] = len(output_rows)
        blocks.append((product_exponents, coefficient_map, coefficients))
    if not blocks:
        return {}
    coefficient_map = sparse.hstack(
        [_align_coefficient_map(exponents, block_map, output_rows)
         for exponents, block_map, _ in blocks],
        format="csr",
    )
    coefficients = coefficient_map @ cp.hstack([vector for _, _, vector in blocks])
    return _coefficient_vector_polynomial(tuple(output_rows), coefficients)


def _align_coefficient_map(
    exponents: Sequence[Exponent],
    coefficient_map: sparse.spmatrix,
    output_rows: Mapping[Exponent, int],
) -> sparse.csr_matrix:
    coordinates = coefficient_map.tocoo()
    rows = np.asarray([output_rows[exponent] for exponent in exponents], dtype=np.int64)
    return sparse.csr_matrix(
        (coordinates.data, (rows[coordinates.row], coordinates.col)),
        shape=(len(output_rows), coefficient_map.shape[1]),
    )


def _polynomial_coefficient_vector(
    polynomial: _CoefficientPolynomial,
) -> tuple[tuple[Exponent, ...], cp.Expression]:
    return polynomial.exponents, polynomial.coefficients








def _putinar_nonnegative(
    target: Mapping[Exponent, cp.Expression | float],
    generators: Sequence[NumericPolynomial],
    *,
    maximum_degree: int,
    name: str,
    equality_generator_index: int | None = None,
) -> tuple[list[cp.Constraint], list[cp.Variable]]:
    """Certify target >= 0 on a simplex domain (or one of its faces)."""
    if maximum_degree < _polynomial_degree(target):
        raise ValueError(
            f"Certificate degree {maximum_degree} is below target degree {_polynomial_degree(target)}."
        )
    num_variables = len(next(iter(generators[0]))) if generators else len(next(iter(target)))
    # Assemble the same Putinar polynomial in sparse coefficient space. Keeping
    # one vector identity per certificate avoids thousands of CVXPY scalar sums
    # and equalities, especially with the full degree-three 28-variable library.
    constraints: list[cp.Constraint] = []
    auxiliary_variables: list[cp.Variable] = []
    blocks: list[tuple[tuple[Exponent, ...], sparse.spmatrix, cp.Expression]] = []

    def add_sos_block(maximum_sos_degree: int, block_name: str):
        basis_degree = int(maximum_sos_degree) // 2
        basis = monomial_exponents(num_variables, basis_degree)
        gram = cp.Variable((len(basis), len(basis)), symmetric=True, name=f"{block_name}_gram")
        auxiliary_variables.append(gram)
        constraints.append(gram >> 0)
        block_exponents, coefficient_map = _gram_coefficient_map(num_variables, basis_degree)
        return block_exponents, coefficient_map, cp.vec(gram, order="C")

    blocks.append(add_sos_block(maximum_degree, f"{name}_sigma0"))

    for generator_index, generator in enumerate(generators):
        generator_degree = _polynomial_degree(generator)
        multiplier_degree = maximum_degree - generator_degree
        if generator_index == equality_generator_index:
            multiplier_exponents = monomial_exponents(num_variables, multiplier_degree)
            coefficients = cp.Variable(
                len(multiplier_exponents), name=f"{name}_equality_coefficients"
            )
            auxiliary_variables.append(coefficients)
            multiplier_map = sparse.eye(len(multiplier_exponents), format="csr")
            multiplier_vector = coefficients
        else:
            multiplier_exponents, multiplier_map, multiplier_vector = add_sos_block(
                multiplier_degree, f"{name}_sigma{generator_index + 1}"
            )
        product_exponents, product_map = _numeric_multiplier_map(
            multiplier_exponents, tuple(generator.items())
        )
        blocks.append((product_exponents, product_map @ multiplier_map, multiplier_vector))

    output_exponents = tuple(sorted(
        set(target).union(*(block_exponents for block_exponents, _, _ in blocks)),
        key=lambda exponent: (sum(exponent), exponent),
    ))
    output_rows = {exponent: index for index, exponent in enumerate(output_exponents)}
    certificate_map = sparse.hstack(
        [_align_coefficient_map(exponents, coefficient_map, output_rows)
         for exponents, coefficient_map, _ in blocks],
        format="csr",
    )
    certificate_vector = certificate_map @ cp.hstack([vector for _, _, vector in blocks])
    if target:
        target_exponents, target_vector = _polynomial_coefficient_vector(target)
        target_map = _align_coefficient_map(
            target_exponents, sparse.eye(len(target_exponents), format="csr"), output_rows
        )
        target_vector = target_map @ target_vector
    else:
        target_vector = cp.Constant(np.zeros(len(output_exponents)))
    constraints.append(target_vector == certificate_vector)
    return constraints, auxiliary_variables


def _affine_face_box_nonnegative(
    target: Mapping[Exponent, cp.Expression | float],
    chart: ProductSimplexChart,
    *,
    face_index: int,
    payoff_domain_bound: float,
    name: str,
) -> list[cp.Constraint]:
    """Exactly certify an affine field on one simplex face times a payoff box.

    The box minimum is minus its radius times the payoff coefficient L1 norm.
    The state minimum separates across players; each affine contribution is
    minimized at a vertex, excluding the face's action for its own player.
    Lower-bound variables express this separable minimum without enumerating
    products of vertices or payoff-box corners. All constraints are linear.
    """
    num_variables = 2 * chart.reduced_dim
    if any(len(exponent) != num_variables or sum(exponent) > 1 for exponent in target):
        raise ValueError("Exact affine FI requires an affine state/payoff polynomial.")
    if not 0 <= face_index < chart.ambient_dim:
        raise ValueError("face_index must identify an ambient simplex action.")
    constant = target.get((0,) * num_variables, 0.0)
    linear = [
        target.get(tuple(int(index == variable) for index in range(num_variables)), 0.0)
        for variable in range(num_variables)
    ]
    payoff_coefficients = cp.hstack(linear[chart.reduced_dim:])
    absolute_upper = cp.Variable(chart.reduced_dim, name=f"{name}_payoff_abs")
    constraints = [
        absolute_upper >= payoff_coefficients,
        absolute_upper >= -payoff_coefficients,
    ]
    box_penalty = float(payoff_domain_bound) * cp.sum(absolute_upper)
    player_lower = (
        cp.Variable(len(chart.action_dims), name=f"{name}_player_min")
        if len(chart.action_dims) > 1 else None
    )
    for player, (ambient_slice, reduced_slice) in enumerate(zip(
        chart.player_slices, chart.reduced_player_slices,
    )):
        # Independent-coordinate vertices contribute one linear coefficient;
        # the dependent last-action vertex contributes zero.
        vertex_values = cp.hstack([*linear[reduced_slice], 0.0])
        allowed_actions = [
            action for action in range(chart.action_dims[player])
            if ambient_slice.start + action != face_index
        ]
        if player_lower is None:
            constraints.append(constant + vertex_values[allowed_actions] >= box_penalty)
        else:
            constraints.append(player_lower[player] <= vertex_values[allowed_actions])
    if player_lower is not None:
        constraints.append(constant + cp.sum(player_lower) >= box_penalty)
    return constraints






@dataclass(frozen=True)
class PayoffConditionedPolynomialVectorField:
    """Polynomial field p(x, v) using reduced state and centered-payoff coordinates."""

    chart: ProductSimplexChart
    degree: int
    exponents: tuple[Exponent, ...]
    reduced_coefficients: np.ndarray
    payoff_scale: float
    payoff_domain_bound: float

    @property
    def ambient_coefficients(self) -> np.ndarray:
        columns = []
        reduced_column = 0
        for size in self.chart.action_dims:
            player_columns = []
            for _ in range(size - 1):
                player_columns.append(self.reduced_coefficients[:, reduced_column])
                reduced_column += 1
            player_columns.append(-np.sum(np.stack(player_columns, axis=1), axis=1))
            columns.extend(player_columns)
        return np.stack(columns, axis=1)

    def coordinates(
        self,
        ambient_states: np.ndarray,
        ambient_payoffs: np.ndarray,
    ) -> np.ndarray:
        reduced_states = self.chart.ambient_to_reduced(ambient_states)
        reduced_payoffs = centered_payoff_coordinates(ambient_payoffs, self.chart)
        return np.concatenate(
            [
                np.atleast_2d(reduced_states),
                np.atleast_2d(reduced_payoffs) / float(self.payoff_scale),
            ],
            axis=1,
        )

    def __call__(
        self,
        ambient_states: np.ndarray,
        ambient_payoffs: np.ndarray,
    ) -> np.ndarray:
        ambient_states = np.asarray(ambient_states, dtype=np.float64)
        ambient_payoffs = np.asarray(ambient_payoffs, dtype=np.float64)
        single = ambient_states.ndim == 1
        if single != (ambient_payoffs.ndim == 1):
            raise ValueError("States and payoffs must both be batched or both be single vectors.")
        if not single and ambient_states.shape[0] != ambient_payoffs.shape[0]:
            raise ValueError(
                f"State/payoff batch sizes differ: {ambient_states.shape[0]} "
                f"and {ambient_payoffs.shape[0]}."
            )
        coordinates = self.coordinates(ambient_states, ambient_payoffs)
        features = evaluate_monomials(coordinates, self.exponents)
        values = features @ self.ambient_coefficients
        return values[0] if single else values


@dataclass(frozen=True)
class PayoffConditionedSIARFitResult:
    vector_field: PayoffConditionedPolynomialVectorField
    status: str
    objective_value: float
    data_mse: float
    positive_correlation_domain: str
    num_positive_correlation_graphs: int
    training_graph_payoff_domain_max_abs: float
    solver: str
    solve_time: float | None
    num_scalar_variables: int
    num_scalar_equalities: int
    num_scalar_inequalities: int
    num_psd_gram_matrices: int
    max_constraint_violation: float
    min_gram_eigenvalue: float
    solver_objective_scale: float = 1.0
    scaled_solver_objective_value: float | None = None
    forward_invariance_certificate_kind: str = "sos"


def validate_siar_solver_objective_scale(value: object) -> float:
    """Require a finite positive real factor for the entire SIAR objective."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("siar_solver_objective_scale must be finite and positive.")
    try:
        scale = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("siar_solver_objective_scale must be finite and positive.") from exc
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("siar_solver_objective_scale must be finite and positive.")
    return scale




def fit_payoff_conditioned_siar(
    x: np.ndarray,
    v: np.ndarray,
    x_dot: np.ndarray,
    *,
    action_dims: Sequence[int],
    payoff_graphs: Sequence[AffinePayoffGraph] | None = None,
    degree: int = 3,
    payoff_scale: float | None = None,
    payoff_domain_bound: float = 1.25,
    enforce_forward_invariance: bool = True,
    enforce_positive_correlation: bool = True,
    coefficient_regularization: float = 1e-9,
    solver_objective_scale: float = 1.0,
    solver: str = "SCS",
    solver_options: Mapping[str, object] | None = None,
) -> PayoffConditionedSIARFitResult:
    """Fit a shared SIAR response p(x, v) across payoff environments.

    Payoffs are centered independently for every player, so the regressor and
    side-information certificates are invariant to player-wise payoff
    translations. Tangency holds by parameterization, and forward invariance
    is certified on the product of the state simplex and the payoff box.
    Positive correlation is instead certified only after substituting every
    training payoff graph ``v = v^(n)(x)``, matching fixed-game SIAR on each
    observed game. The independent centered-payoff coordinates are normalized
    by ``payoff_scale``. ``solver_objective_scale`` multiplies the complete
    loss-plus-regularization objective, preserving its exact minimizers and
    relative weights. ``objective_value`` retains the unscaled objective.
    """
    solver_objective_scale = validate_siar_solver_objective_scale(solver_objective_scale)
    chart = ProductSimplexChart(tuple(int(size) for size in action_dims))
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    x_dot = np.asarray(x_dot, dtype=np.float64)
    expected_shape = (x.shape[0], chart.ambient_dim) if x.ndim == 2 else None
    if x.ndim != 2 or x.shape[1] != chart.ambient_dim:
        raise ValueError(f"x must have shape (samples, {chart.ambient_dim}), got {x.shape}.")
    if x.shape[0] == 0:
        raise ValueError("SIAR requires at least one observation.")
    if v.shape != expected_shape or x_dot.shape != expected_shape:
        raise ValueError(
            f"v and x_dot must match x; got x={x.shape}, v={v.shape}, x_dot={x_dot.shape}."
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(v)) or not np.all(np.isfinite(x_dot)):
        raise ValueError("x, v, and x_dot must contain only finite values.")
    if degree < 1:
        raise ValueError("degree must be at least one.")
    if coefficient_regularization < 0:
        raise ValueError("coefficient_regularization must be nonnegative.")
    if not np.isfinite(payoff_domain_bound) or payoff_domain_bound <= 0.0:
        raise ValueError("payoff_domain_bound must be finite and positive.")
    payoff_graphs = () if payoff_graphs is None else tuple(payoff_graphs)
    if enforce_positive_correlation and not payoff_graphs:
        raise ValueError(
            "Payoff-conditioned SIAR positive correlation requires at least "
            "one training affine payoff graph."
        )

    reduced_x = chart.ambient_to_reduced(x)
    reduced_v = centered_payoff_coordinates(v, chart)
    if payoff_scale is None:
        payoff_scale = float(np.max(np.abs(reduced_v)))
    payoff_scale = float(payoff_scale)
    if not np.isfinite(payoff_scale) or payoff_scale <= 1e-12:
        raise ValueError("payoff_scale must be finite and positive.")
    normalized_v = reduced_v / payoff_scale
    train_payoff_max_abs = float(np.max(np.abs(normalized_v)))
    if train_payoff_max_abs > float(payoff_domain_bound) + 1e-10:
        raise ValueError(
            "Training payoffs exceed the certified domain: "
            f"max_abs={train_payoff_max_abs:.6g}, bound={float(payoff_domain_bound):.6g}."
        )
    graph_payoff_max_abs = (
        affine_payoff_graph_max_abs(chart, payoff_graphs) / payoff_scale
        if payoff_graphs
        else 0.0
    )
    if graph_payoff_max_abs > float(payoff_domain_bound) + 1e-10:
        raise ValueError(
            "A training payoff graph exceeds the forward-invariance payoff "
            "domain: "
            f"max_abs={graph_payoff_max_abs:.6g}, "
            f"bound={float(payoff_domain_bound):.6g}. "
            "Increase payoff_scale or payoff_domain_bound."
        )

    coordinates = np.concatenate([reduced_x, normalized_v], axis=1)
    num_variables = 2 * chart.reduced_dim
    exponents = monomial_exponents(num_variables, degree)
    library = evaluate_monomials(coordinates, exponents)
    independent_coefficients = cp.Variable(
        (len(exponents), chart.reduced_dim),
        name="payoff_conditioned_siar_coefficients",
    )

    ambient_polynomials: list[ExpressionPolynomial] = []
    ambient_coefficient_columns: list[cp.Expression] = []
    reduced_column = 0
    for size in chart.action_dims:
        player_columns = []
        for _ in range(size - 1):
            column = independent_coefficients[:, reduced_column]
            polynomial = _coefficient_vector_polynomial(exponents, column)
            player_columns.append(column)
            ambient_polynomials.append(polynomial)
            reduced_column += 1
        last_column = -sum(player_columns[1:], player_columns[0])
        last_polynomial = _coefficient_vector_polynomial(exponents, last_column)
        ambient_coefficient_columns.extend([*player_columns, last_column])
        ambient_polynomials.append(last_polynomial)

    ambient_coefficient_matrix = cp.hstack(
        [
            cp.reshape(column, (len(exponents), 1), order="C")
            for column in ambient_coefficient_columns
        ]
    )
    prediction = library @ ambient_coefficient_matrix
    data_loss = cp.sum_squares(prediction - x_dot) / float(x.shape[0])
    objective = data_loss
    if coefficient_regularization:
        objective += float(coefficient_regularization) * cp.sum_squares(
            independent_coefficients
        )

    state_generators = tuple(
        _lift_numeric_polynomial(
            generator,
            trailing_variables=chart.reduced_dim,
        )
        for generator in chart.domain_generators()
    )
    payoff_generators = _payoff_box_generators(chart, float(payoff_domain_bound))
    generators = (*state_generators, *payoff_generators)
    constraints: list[cp.Constraint] = []
    auxiliary_variables: list[cp.Variable] = []

    forward_invariance_certificate_kind = "disabled"
    if enforce_forward_invariance:
        if degree == 1:
            forward_invariance_certificate_kind = "affine-face-box-exact"
            for face_index, target in enumerate(ambient_polynomials):
                constraints.extend(_affine_face_box_nonnegative(
                    target, chart, face_index=face_index,
                    payoff_domain_bound=float(payoff_domain_bound),
                    name=f"conditional_forward_face_{face_index}",
                ))
        else:
            forward_invariance_certificate_kind = "sos"
            for face_index, target in enumerate(ambient_polynomials):
                face_constraints, face_variables = _putinar_nonnegative(
                    target,
                    generators,
                    maximum_degree=degree,
                    name=f"conditional_forward_face_{face_index}",
                    equality_generator_index=face_index,
                )
                constraints.extend(face_constraints)
                auxiliary_variables.extend(face_variables)

    if enforce_positive_correlation:
        state_variables = tuple(
            _variable_polynomial(chart.reduced_dim, variable_index)
            for variable_index in range(chart.reduced_dim)
        )
        state_only_generators = chart.domain_generators()
        for graph_index, payoff_graph in enumerate(payoff_graphs):
            graph_payoffs = _normalized_centered_affine_payoff_polynomials(
                chart,
                payoff_graph,
                payoff_scale=payoff_scale,
            )
            independent_graph_payoffs: list[NumericPolynomial] = []
            for player_slice in chart.player_slices:
                independent_graph_payoffs.extend(
                    graph_payoffs[player_slice.start : player_slice.stop - 1]
                )
            substitutions = (*state_variables, *independent_graph_payoffs)
            graph_vector_field = tuple(
                _expression_poly_compose_numeric(
                    polynomial,
                    substitutions,
                    num_output_variables=chart.reduced_dim,
                )
                for polynomial in ambient_polynomials
            )

            for player, player_slice in enumerate(chart.player_slices):
                correlation = _expression_poly_sum_products_numeric(
                    tuple(
                        (
                            graph_vector_field[ambient_index],
                            graph_payoffs[ambient_index],
                        )
                        for ambient_index in range(player_slice.start, player_slice.stop)
                    )
                )
                correlation_degree = _polynomial_degree(correlation)
                certificate_degree = correlation_degree + (
                    correlation_degree % 2 == 0
                )
                correlation_constraints, correlation_variables = _putinar_nonnegative(
                    correlation,
                    state_only_generators,
                    maximum_degree=certificate_degree,
                    name=(
                        "conditional_positive_correlation_"
                        f"graph_{graph_index}_player_{player}"
                    ),
                )
                constraints.extend(correlation_constraints)
                auxiliary_variables.extend(correlation_variables)

    # Preserve the original expression and canonicalization when scale is one.
    solver_objective = (
        objective if solver_objective_scale == 1.0
        else solver_objective_scale * objective
    )
    problem = cp.Problem(cp.Minimize(solver_objective), constraints)
    options = dict(solver_options or {})
    if solver.upper() == "SCS":
        options.setdefault("eps", 1e-6)
        options.setdefault("max_iters", 200_000)
        options.setdefault("acceleration_lookback", 20)
    # The default high-accuracy CLARABEL solve can stop on numerical error for
    # some projection games. Retrying the *same* SDP with looser stopping
    # tolerances changes no model, observations, or certificate constraints.
    # SCS is a final numerical fallback; its result is still audited below.
    used_solver = str(solver).upper()
    try:
        problem.solve(solver=used_solver, **options)
    except cp.error.SolverError:
        if used_solver != "CLARABEL":
            raise
        print("[siar] CLARABEL numerical error; retrying same SDP with 1e-6 tolerances", flush=True)
        try:
            problem.solve(
                solver="CLARABEL",
                tol_gap_abs=1e-6,
                tol_gap_rel=1e-6,
                tol_feas=1e-6,
                max_iter=600,
            )
        except cp.error.SolverError:
            print("[siar] CLARABEL retry failed; falling back to SCS on same SDP", flush=True)
            used_solver = "SCS"
            problem.solve(solver="SCS", eps=1e-6, max_iters=500_000,
                          acceleration_lookback=20)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(
            f"Payoff-conditioned SIAR optimization failed with status {problem.status!r}."
        )
    if independent_coefficients.value is None:
        raise RuntimeError("SIAR solver returned no polynomial coefficients.")

    fitted_coefficients = np.asarray(independent_coefficients.value, dtype=np.float64)
    vector_field = PayoffConditionedPolynomialVectorField(
        chart=chart,
        degree=int(degree),
        exponents=tuple(exponents),
        reduced_coefficients=fitted_coefficients,
        payoff_scale=payoff_scale,
        payoff_domain_bound=float(payoff_domain_bound),
    )
    fitted_prediction = vector_field(x, v)
    fitted_mse = float(np.mean(np.sum(np.square(fitted_prediction - x_dot), axis=1)))
    metrics = problem.size_metrics
    solver_stats = problem.solver_stats
    constraint_violations = [
        float(np.max(np.asarray(constraint.violation(), dtype=np.float64)))
        for constraint in constraints
    ]
    gram_matrices = [
        np.asarray(variable.value, dtype=np.float64)
        for variable in auxiliary_variables
        if len(variable.shape) == 2
        and variable.shape[0] == variable.shape[1]
        and variable.value is not None
    ]
    gram_eigenvalues = [
        float(np.min(np.linalg.eigvalsh(0.5 * (matrix + matrix.T))))
        for matrix in gram_matrices
    ]
    return PayoffConditionedSIARFitResult(
        vector_field=vector_field,
        status=str(problem.status),
        objective_value=float(objective.value),
        solver_objective_scale=solver_objective_scale,
        scaled_solver_objective_value=float(problem.value),
        data_mse=fitted_mse,
        positive_correlation_domain=(
            "training_payoff_graphs"
            if enforce_positive_correlation
            else "disabled"
        ),
        num_positive_correlation_graphs=(
            len(payoff_graphs) if enforce_positive_correlation else 0
        ),
        training_graph_payoff_domain_max_abs=graph_payoff_max_abs,
        solver=used_solver,
        solve_time=None if solver_stats.solve_time is None else float(solver_stats.solve_time),
        num_scalar_variables=int(metrics.num_scalar_variables),
        num_scalar_equalities=int(metrics.num_scalar_eq_constr),
        num_scalar_inequalities=int(metrics.num_scalar_leq_constr),
        num_psd_gram_matrices=len(gram_matrices),
        max_constraint_violation=max(constraint_violations, default=0.0),
        min_gram_eigenvalue=min(gram_eigenvalues, default=0.0),
        forward_invariance_certificate_kind=forward_invariance_certificate_kind,
    )
