"""
XQVM XQMX Types and Operations

XQMX represents quadratic models (QUBO/Ising) with:
- mode: MODEL (for building constraints/objectives) or SAMPLE (for solutions)
- domain: BINARY [0,1], SPIN [-1,+1], or DISCRETE [0..k-1]
- Grid operations for row/column indexing
- High-level functions (HLF)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from .errors import XQMXModeError

class XQMXMode(Enum):
    """ XQMX matrix mode: model (for building) or sample (for solutions). """
    MODEL = auto()
    SAMPLE = auto()

class XQMXDomain(Enum):
    """ XQMX variable domain types. """
    BINARY = auto()    # [0, 1] - QUBO/BQM
    SPIN = auto()      # [-1, +1] - Ising
    DISCRETE = auto()  # [0..k-1] - discrete with k values

@dataclass
class XQMX:
    """
    Sparse quadratic matrix for optimization problems.

    XQMX represents a quadratic model (QUBO/Ising) with:
    - mode: MODEL (for building constraints/objectives) or SAMPLE (for solutions)
    - domain: BINARY [0,1], SPIN [-1,+1], or DISCRETE [0..k-1]
    - dimensions: size (total variables), rows, cols (for grid indexing)
    - linear: dict mapping variable index -> linear coefficient
    - quadratic: dict mapping (i, j) tuple -> coupling coefficient (i < j)

    For SAMPLE mode, linear stores the variable assignments (0/1 or -1/+1).
    """
    mode: XQMXMode
    domain: XQMXDomain
    size: int  # Total number of variables
    rows: int = 0  # Grid rows (0 if not grid-indexed)
    cols: int = 0  # Grid cols (0 if not grid-indexed)
    linear: dict[int, float] = field(default_factory=dict)
    quadratic: dict[tuple[int, int], float] = field(default_factory=dict)
    discrete_k: int = 2  # For DISCRETE domain: number of values [0..k-1]

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError(f"XQMX size must be non-negative, got {self.size}")
        if self.rows < 0 or self.cols < 0:
            raise ValueError(f"XQMX rows/cols must be non-negative")
        if self.domain == XQMXDomain.DISCRETE and self.discrete_k < 2:
            raise ValueError(f"DISCRETE domain requires k >= 2, got {self.discrete_k}")

    @classmethod
    def binary_model(cls, size: int, rows: int = 0, cols: int = 0) -> XQMX:
        """ Create a binary [0,1] model XQMX. """
        return cls(
            mode=XQMXMode.MODEL,
            domain=XQMXDomain.BINARY,
            size=size,
            rows=rows,
            cols=cols,
        )

    @classmethod
    def spin_model(cls, size: int, rows: int = 0, cols: int = 0) -> XQMX:
        """ Create a spin [-1,+1] model XQMX. """
        return cls(
            mode=XQMXMode.MODEL,
            domain=XQMXDomain.SPIN,
            size=size,
            rows=rows,
            cols=cols,
        )

    @classmethod
    def discrete_model(cls, size: int, k: int, rows: int = 0, cols: int = 0) -> XQMX:
        """ Create a discrete [0..k-1] model XQMX. """
        return cls(
            mode=XQMXMode.MODEL,
            domain=XQMXDomain.DISCRETE,
            size=size,
            rows=rows,
            cols=cols,
            discrete_k=k,
        )

    @classmethod
    def binary_sample(cls, size: int, rows: int = 0, cols: int = 0) -> XQMX:
        """ Create a binary [0,1] sample XQMX. """
        return cls(
            mode=XQMXMode.SAMPLE,
            domain=XQMXDomain.BINARY,
            size=size,
            rows=rows,
            cols=cols,
        )

    @classmethod
    def spin_sample(cls, size: int, rows: int = 0, cols: int = 0) -> XQMX:
        """ Create a spin [-1,+1] sample XQMX. """
        return cls(
            mode=XQMXMode.SAMPLE,
            domain=XQMXDomain.SPIN,
            size=size,
            rows=rows,
            cols=cols,
        )

    @classmethod
    def discrete_sample(cls, size: int, k: int, rows: int = 0, cols: int = 0) -> XQMX:
        """ Create a discrete [0..k-1] sample XQMX. """
        return cls(
            mode=XQMXMode.SAMPLE,
            domain=XQMXDomain.DISCRETE,
            size=size,
            rows=rows,
            cols=cols,
            discrete_k=k,
        )

    def is_model(self) -> bool:
        """ Check if this is a model (vs sample). """
        return self.mode == XQMXMode.MODEL

    def is_sample(self) -> bool:
        """ Check if this is a sample (vs model). """
        return self.mode == XQMXMode.SAMPLE

    def get_linear(self, i: int) -> float:
        """ Get linear coefficient/value for variable i. """
        return self.linear.get(i, 0.0)

    def set_linear(self, i: int, value: float) -> None:
        """ Set linear coefficient/value for variable i. """
        if i < 0 or i >= self.size:
            raise IndexError(f"Variable index {i} out of range [0, {self.size})")

        if value == 0.0:
            self.linear.pop(i, None)
        else:
            self.linear[i] = value

    def add_linear(self, i: int, delta: float) -> None:
        """ Add to linear coefficient for variable i. """
        if i < 0 or i >= self.size:
            raise IndexError(f"Variable index {i} out of range [0, {self.size})")

        current = self.linear.get(i, 0.0)
        new_value = current + delta

        if new_value == 0.0:
            self.linear.pop(i, None)
        else:
            self.linear[i] = new_value

    def get_quadratic(self, i: int, j: int) -> float:
        """ Get quadratic coefficient for variables i, j. """
        if i > j:
            i, j = j, i
        return self.quadratic.get((i, j), 0.0)

    def set_quadratic(self, i: int, j: int, value: float) -> None:
        """ Set quadratic coefficient for variables i, j. """
        if i < 0 or i >= self.size or j < 0 or j >= self.size:
            raise IndexError(f"Variable indices ({i}, {j}) out of range [0, {self.size})")

        if i > j:
            i, j = j, i

        if value == 0.0:
            self.quadratic.pop((i, j), None)
        else:
            self.quadratic[(i, j)] = value

    def add_quadratic(self, i: int, j: int, delta: float) -> None:
        """ Add to quadratic coefficient for variables i, j. """
        if i < 0 or i >= self.size or j < 0 or j >= self.size:
            raise IndexError(f"Variable indices ({i}, {j}) out of range [0, {self.size})")

        if i > j:
            i, j = j, i

        current = self.quadratic.get((i, j), 0.0)
        new_value = current + delta

        if new_value == 0.0:
            self.quadratic.pop((i, j), None)
        else:
            self.quadratic[(i, j)] = new_value

    def grid_index(self, row: int, col: int) -> int:
        """ Convert grid (row, col) to linear variable index. """
        if self.rows == 0 or self.cols == 0:
            raise ValueError("Grid indexing requires non-zero rows and cols")
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            raise IndexError(f"Grid position ({row}, {col}) out of range [{self.rows}, {self.cols}]")

        return row * self.cols + col

    def __repr__(self) -> str:
        return (
            f"XQMX(mode={self.mode.name}, domain={self.domain.name}, "
            f"size={self.size}, linear_terms={len(self.linear)}, "
            f"quadratic_terms={len(self.quadratic)})"
        )

# ============================================================================
# Grid Operations
# ============================================================================

def row_indices(xqmx: XQMX, row: int) -> list[int]:
    """
    Get all variable indices in a given row.

    For a grid with `cols` columns, row `r` contains indices [r*cols, (r+1)*cols).
    """
    if xqmx.rows == 0 or xqmx.cols == 0:
        raise ValueError("Grid indexing requires non-zero rows and cols")
    if row < 0 or row >= xqmx.rows:
        raise IndexError(f"Row {row} out of range [0, {xqmx.rows})")

    start = row * xqmx.cols
    return list(range(start, start + xqmx.cols))

def col_indices(xqmx: XQMX, col: int) -> list[int]:
    """
    Get all variable indices in a given column.

    For a grid with `cols` columns, column `c` contains indices [c, c+cols, c+2*cols, ...].
    """
    if xqmx.rows == 0 or xqmx.cols == 0:
        raise ValueError("Grid indexing requires non-zero rows and cols")
    if col < 0 or col >= xqmx.cols:
        raise IndexError(f"Column {col} out of range [0, {xqmx.cols})")

    return [col + r * xqmx.cols for r in range(xqmx.rows)]

def row_sum(xqmx: XQMX, row: int) -> float:
    """
    Sum all linear values in a given row.

    Used primarily with SAMPLE mode to count active variables in a row.
    """
    indices = row_indices(xqmx, row)
    return sum(xqmx.get_linear(i) for i in indices)

def col_sum(xqmx: XQMX, col: int) -> float:
    """
    Sum all linear values in a given column.

    Used primarily with SAMPLE mode to count active variables in a column.
    """
    indices = col_indices(xqmx, col)
    return sum(xqmx.get_linear(i) for i in indices)

def row_find(xqmx: XQMX, row: int, value: int) -> int:
    """
    Find the first column index where the row has the given value.

    Used primarily with SAMPLE mode to find which column is selected in a row.
    Returns -1 if no column has the given value.
    """
    if xqmx.cols == 0:
        return -1
    indices = row_indices(xqmx, row)
    for col, idx in enumerate(indices):
        if xqmx.get_linear(idx) == float(value):
            return col
    return -1

def col_find(xqmx: XQMX, col: int, value: int) -> int:
    """
    Find the first row index where the column has the given value.

    Used primarily with SAMPLE mode to find which row is selected in a column.
    Returns -1 if no row has the given value.
    """
    if xqmx.rows == 0:
        return -1
    indices = col_indices(xqmx, col)
    for row, idx in enumerate(indices):
        if xqmx.get_linear(idx) == float(value):
            return row
    return -1

# ============================================================================
# Mode Validators
# ============================================================================

def require_model_mode(xqmx: XQMX, operation: str) -> None:
    """
    Require that the XQMX is in MODEL mode.

    Raises XQMXModeError if not in MODEL mode.
    """
    if xqmx.mode != XQMXMode.MODEL:
        raise XQMXModeError(operation, xqmx.mode.name, "MODEL")

def require_sample_mode(xqmx: XQMX, operation: str) -> None:
    """
    Require that the XQMX is in SAMPLE mode.

    Raises XQMXModeError if not in SAMPLE mode.
    """
    if xqmx.mode != XQMXMode.SAMPLE:
        raise XQMXModeError(operation, xqmx.mode.name, "SAMPLE")

# ============================================================================
# High-Level Functions (HLF)
# ============================================================================

def expand_onehot(model: XQMX, indices: list[int], penalty: float) -> None:
    """
    Add a one-hot constraint: exactly one variable in indices must be 1.

    Expands (sum(x) - 1)^2 = sum(x)^2 - 2*sum(x) + 1

    For QUBO:
      - Linear terms: -penalty for each x_i (from -2*sum(x), ignoring constant)
      - Quadratic terms: +2*penalty for each pair (x_i, x_j) where i < j

    Note: x_i^2 = x_i for binary variables, so the quadratic expansion
    contributes to linear terms as well, but the net effect is:
      - linear[i] += -penalty  (the -2 + 1 from expansion simplifies)
      - quadratic[i,j] += 2*penalty
    """
    require_model_mode(model, "ONEHOT")

    # Linear terms: -penalty for each variable
    for i in indices:
        model.add_linear(i, -penalty)

    # Quadratic terms: +2*penalty for each pair
    n = len(indices)
    for a in range(n):
        for b in range(a + 1, n):
            model.add_quadratic(indices[a], indices[b], 2.0 * penalty)

def expand_exclude(model: XQMX, i: int, j: int, penalty: float) -> None:
    """
    Add an exclusion constraint: variables i and j cannot both be 1.

    Adds penalty * x_i * x_j to the model.
    This makes it energetically unfavorable for both to be 1 simultaneously.
    """
    require_model_mode(model, "EXCLUDE")

    model.add_quadratic(i, j, penalty)

def expand_implies(model: XQMX, i: int, j: int, penalty: float) -> None:
    """
    Add an implication constraint: x_i = 1 implies x_j = 1.

    Expands x_i * (1 - x_j) = x_i - x_i * x_j

    This penalizes the case where x_i = 1 and x_j = 0.

    For QUBO:
      - linear[i] += penalty
      - quadratic[i,j] += -penalty
    """
    require_model_mode(model, "IMPLIES")

    model.add_linear(i, penalty)
    model.add_quadratic(i, j, -penalty)

def compute_energy(model: XQMX, sample: XQMX) -> float:
    """
    Compute the energy of a sample with respect to a model.

    E = sum_i(linear[i] * x_i) + sum_{i<j}(quadratic[i,j] * x_i * x_j)

    Where x_i are the sample values (from sample.linear).

    The model provides the coefficients, the sample provides the variable assignments.
    """
    require_model_mode(model, "ENERGY (model)")
    require_sample_mode(sample, "ENERGY (sample)")

    if model.size != sample.size:
        raise ValueError(
            f"Model and sample size mismatch: {model.size} vs {sample.size}"
        )

    energy = 0.0

    # Linear contribution
    for i, coeff in model.linear.items():
        x_i = sample.get_linear(i)
        energy += coeff * x_i

    # Quadratic contribution
    for (i, j), coeff in model.quadratic.items():
        x_i = sample.get_linear(i)
        x_j = sample.get_linear(j)
        energy += coeff * x_i * x_j

    return energy

def expand_penalty_eq(
    model: XQMX,
    indices: list[int],
    coefficients: list[int],
    target: int,
    penalty: float,
) -> None:
    """
    Add a weighted equality penalty: P * (sum(a_k * x_k) - b)^2.

    Expands into QUBO terms for binary variables (x_k^2 = x_k):
      linear[k]   += P * a_k * (a_k - 2*b)
      quad[k,m]   += P * 2 * a_k * a_m    (for all k < m)

    This is the general form that subsumes ONEHOT (all coeffs=1, target=1),
    EXACTLY-k (all coeffs=1, target=k), and weighted constraints like
    knapsack capacity.

    Args:
        model: XQMX in MODEL mode
        indices: Variable indices [x_0, x_1, ..., x_{n-1}]
        coefficients: Corresponding coefficients [a_0, a_1, ..., a_{n-1}]
        target: Target value b
        penalty: Penalty weight P
    """
    require_model_mode(model, "PENALTY_EQ")

    n = len(indices)
    if n != len(coefficients):
        raise ValueError(
            f"PENALTY_EQ: indices length {n} != coefficients length {len(coefficients)}"
        )

    two_b = 2.0 * target

    # Linear terms: P * a_k * (a_k - 2*b)
    for k in range(n):
        a_k = float(coefficients[k])
        model.add_linear(indices[k], penalty * a_k * (a_k - two_b))

    # Quadratic terms: P * 2 * a_k * a_m (for all k < m)
    two_p = 2.0 * penalty
    for k in range(n):
        a_k = float(coefficients[k])
        for m in range(k + 1, n):
            a_m = float(coefficients[m])
            model.add_quadratic(indices[k], indices[m], two_p * a_k * a_m)

def expand_atleast(
    model: XQMX,
    indices: list[int],
    k: int,
    penalty: float,
) -> None:
    """
    Add an at-least-k constraint: at least k variables must be 1.

    Penalizes assignments where fewer than k variables are selected.
    Uses internal slack variables with negative coefficients to convert
    the inequality to an equality, then delegates to PENALTYEQ:

      P * (x_0 + x_1 + ... - s_0 - 2*s_1 - ... - k)^2

    The slack variables are allocated starting at model.size, and the
    model size is extended to accommodate them.

    Args:
        model: XQMX in MODEL mode
        indices: Variable indices that participate in the constraint
        k: Minimum number of variables that must be 1
        penalty: Penalty weight P
    """
    require_model_mode(model, "ATLEAST")

    n = len(indices)

    # General case: introduce slack variables
    # sum(x_i) >= k  →  sum(x_i) - sum(2^j * s_j) = k
    # Need slacks to represent excess: 0 to (n - k)
    max_excess = n - k
    if max_excess <= 0:
        # k >= n: all variables must be 1, use PENALTYEQ with unit coeffs
        expand_penalty_eq(model, indices, [1] * n, k, penalty)
        return

    # Compute number of slack variables
    num_slacks = 0
    temp = max_excess
    while temp > 0:
        temp //= 2
        num_slacks += 1

    # Allocate slack variables at end of model
    slack_start = model.size
    model.size += num_slacks

    # Build combined indices and coefficients
    all_indices = list(indices)
    all_coeffs = [1] * n  # unit coefficients for original variables

    # Slack variables with negative coefficients
    power = 1
    for j in range(num_slacks):
        all_indices.append(slack_start + j)
        all_coeffs.append(-power)
        power *= 2

    # Expand using PENALTYEQ
    expand_penalty_eq(model, all_indices, all_coeffs, k, penalty)

def expand_reduce(
    model: XQMX,
    var_a: int,
    var_b: int,
    penalty: float,
) -> int:
    """
    Degree reduction: replace the product x_a * x_b with an auxiliary variable w.

    Allocates a new auxiliary variable w at model.size and adds penalty terms
    that enforce w = x_a * x_b:

        P * (x_a * x_b - 2 * x_a * w - 2 * x_b * w + 3 * w)

    This penalty is 0 when w = x_a * x_b and positive otherwise (for binary variables).

    After calling this, the cubic term x_a * x_b * x_c can be replaced by
    the quadratic term w * x_c.

    IMPORTANT: The penalty must exceed the maximum coefficient that multiplies
    the auxiliary variable w in the reduced terms. Otherwise the solver can
    gain more energy by setting w incorrectly than it loses from the penalty.
    For example, if the reduced term is 4P * w * x_c, then penalty must be > 4P.

    Args:
        model: XQMX in MODEL mode
        var_a: Index of first variable
        var_b: Index of second variable
        penalty: Penalty weight for the auxiliary constraint (must exceed
                 max coefficient using w in the problem terms)

    Returns:
        Index of the new auxiliary variable w
    """
    require_model_mode(model, "REDUCE")

    # Allocate auxiliary variable
    w = model.size
    model.size += 1

    # Add constraint: P * (x_a * x_b - 2*x_a*w - 2*x_b*w + 3*w)
    model.add_quadratic(var_a, var_b, penalty)         # P * x_a * x_b
    model.add_quadratic(var_a, w, -2.0 * penalty)      # -2P * x_a * w
    model.add_quadratic(var_b, w, -2.0 * penalty)      # -2P * x_b * w
    model.add_linear(w, 3.0 * penalty)                 # 3P * w

    return w

def triu(i: int, j: int) -> int:
    """ Upper triangular index: maps (i, j) to a linear index (auto-swaps so i < j). """
    if i >= j:
        i, j = j, i
    return j * (j - 1) // 2 + i
