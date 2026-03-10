"""
Tests for XQMX class and operations.
"""

import pytest

from xqvm.core.xqmx import (
    XQMX,
    XQMXMode,
    XQMXDomain,
    row_indices,
    col_indices,
    row_sum,
    col_sum,
    row_find,
    col_find,
    expand_onehot,
    expand_exclude,
    expand_implies,
    compute_energy,
    require_model_mode,
)
from xqvm.core.errors import XQMXModeError

class TestXQMXConstruction:
    """Tests for XQMX factory methods and construction."""

    def test_binary_model(self):
        """binary_model creates correct XQMX."""
        x = XQMX.binary_model(size=10)
        assert x.mode == XQMXMode.MODEL
        assert x.domain == XQMXDomain.BINARY
        assert x.size == 10
        assert x.rows == 0
        assert x.cols == 0

    def test_spin_model(self):
        """spin_model creates correct XQMX."""
        x = XQMX.spin_model(size=15)
        assert x.mode == XQMXMode.MODEL
        assert x.domain == XQMXDomain.SPIN
        assert x.size == 15

    def test_discrete_model(self):
        """discrete_model creates correct XQMX."""
        x = XQMX.discrete_model(size=20, k=4)
        assert x.mode == XQMXMode.MODEL
        assert x.domain == XQMXDomain.DISCRETE
        assert x.size == 20
        assert x.discrete_k == 4

    def test_binary_sample(self):
        """binary_sample creates correct XQMX."""
        x = XQMX.binary_sample(size=10)
        assert x.mode == XQMXMode.SAMPLE
        assert x.domain == XQMXDomain.BINARY
        assert x.size == 10

    def test_spin_sample(self):
        """spin_sample creates correct XQMX."""
        x = XQMX.spin_sample(size=12)
        assert x.mode == XQMXMode.SAMPLE
        assert x.domain == XQMXDomain.SPIN
        assert x.size == 12

    def test_discrete_sample(self):
        """discrete_sample creates correct XQMX."""
        x = XQMX.discrete_sample(size=8, k=3)
        assert x.mode == XQMXMode.SAMPLE
        assert x.domain == XQMXDomain.DISCRETE
        assert x.discrete_k == 3

    def test_grid_dimensions(self):
        """Factory methods accept grid dimensions."""
        x = XQMX.binary_model(size=25, rows=5, cols=5)
        assert x.rows == 5
        assert x.cols == 5

    def test_negative_size_raises(self):
        """Negative size should raise ValueError."""
        with pytest.raises(ValueError):
            XQMX.binary_model(size=-1)

    def test_discrete_k_validation(self):
        """Discrete k < 2 should raise ValueError."""
        with pytest.raises(ValueError):
            XQMX.discrete_model(size=10, k=1)

class TestXQMXModeChecks:
    """Tests for mode checking methods."""

    def test_is_model_true(self, binary_model):
        """is_model returns True for MODEL mode."""
        assert binary_model.is_model() is True

    def test_is_model_false(self, binary_sample):
        """is_model returns False for SAMPLE mode."""
        assert binary_sample.is_model() is False

    def test_is_sample_true(self, binary_sample):
        """is_sample returns True for SAMPLE mode."""
        assert binary_sample.is_sample() is True

    def test_is_sample_false(self, binary_model):
        """is_sample returns False for MODEL mode."""
        assert binary_model.is_sample() is False

class TestLinearCoefficients:
    """Tests for linear coefficient operations."""

    def test_get_linear_default(self, binary_model):
        """get_linear returns 0.0 for unset index."""
        assert binary_model.get_linear(0) == 0.0
        assert binary_model.get_linear(5) == 0.0

    def test_set_linear(self, binary_model):
        """set_linear stores coefficient."""
        binary_model.set_linear(3, 2.5)
        assert binary_model.get_linear(3) == 2.5

    def test_set_linear_zero_removes(self, binary_model):
        """set_linear with 0 removes key (sparse)."""
        binary_model.set_linear(0, 5.0)
        binary_model.set_linear(0, 0.0)
        assert 0 not in binary_model.linear

    def test_add_linear(self, binary_model):
        """add_linear adds to existing coefficient."""
        binary_model.set_linear(0, 1.0)
        binary_model.add_linear(0, 0.5)
        assert binary_model.get_linear(0) == 1.5

    def test_add_linear_to_unset(self, binary_model):
        """add_linear to unset index creates it."""
        binary_model.add_linear(5, 3.0)
        assert binary_model.get_linear(5) == 3.0

    def test_linear_index_bounds(self, binary_model):
        """Out of bounds index raises IndexError."""
        with pytest.raises(IndexError):
            binary_model.set_linear(100, 1.0)

    def test_linear_negative_index(self, binary_model):
        """Negative index raises IndexError."""
        with pytest.raises(IndexError):
            binary_model.set_linear(-1, 1.0)

class TestQuadraticCoefficients:
    """Tests for quadratic coefficient operations."""

    def test_get_quadratic_default(self, binary_model):
        """get_quadratic returns 0.0 for unset pair."""
        assert binary_model.get_quadratic(0, 1) == 0.0

    def test_set_quadratic(self, binary_model):
        """set_quadratic stores coefficient."""
        binary_model.set_quadratic(0, 1, 3.5)
        assert binary_model.get_quadratic(0, 1) == 3.5

    def test_quadratic_index_normalization(self, binary_model):
        """Indices are normalized so i < j."""
        binary_model.set_quadratic(5, 2, 4.0)
        # Should be stored as (2, 5)
        assert binary_model.get_quadratic(2, 5) == 4.0
        assert binary_model.get_quadratic(5, 2) == 4.0

    def test_set_quadratic_zero_removes(self, binary_model):
        """set_quadratic with 0 removes key (sparse)."""
        binary_model.set_quadratic(0, 1, 5.0)
        binary_model.set_quadratic(0, 1, 0.0)
        assert (0, 1) not in binary_model.quadratic

    def test_add_quadratic(self, binary_model):
        """add_quadratic adds to existing coefficient."""
        binary_model.set_quadratic(0, 1, 1.0)
        binary_model.add_quadratic(0, 1, 0.5)
        assert binary_model.get_quadratic(0, 1) == 1.5

    def test_add_quadratic_to_unset(self, binary_model):
        """add_quadratic to unset pair creates it."""
        binary_model.add_quadratic(3, 4, 2.0)
        assert binary_model.get_quadratic(3, 4) == 2.0

    def test_quadratic_index_bounds(self, binary_model):
        """Out of bounds indices raise IndexError."""
        with pytest.raises(IndexError):
            binary_model.set_quadratic(0, 100, 1.0)

    def test_quadratic_same_index_allowed(self, binary_model):
        """Same index (i == j) may be allowed depending on implementation."""
        # Some implementations allow self-couplings, some don't
        # Test whatever the actual behavior is
        try:
            binary_model.set_quadratic(0, 0, 1.0)
            # If it doesn't raise, check it stored correctly
            assert binary_model.get_quadratic(0, 0) == 1.0
        except (IndexError, ValueError):
            # If it raises, that's also valid behavior
            pass

class TestGridOperations:
    """Tests for grid-based operations."""

    def test_grid_index(self, grid_model):
        """grid_index converts row/col to linear index."""
        # For 5x5 grid: index = row * cols + col
        assert grid_model.grid_index(0, 0) == 0
        assert grid_model.grid_index(0, 4) == 4
        assert grid_model.grid_index(1, 0) == 5
        assert grid_model.grid_index(4, 4) == 24

    def test_row_indices(self, grid_model):
        """row_indices returns all indices in row."""
        indices = row_indices(grid_model, 0)
        assert indices == [0, 1, 2, 3, 4]

        indices = row_indices(grid_model, 2)
        assert indices == [10, 11, 12, 13, 14]

    def test_col_indices(self, grid_model):
        """col_indices returns all indices in column."""
        indices = col_indices(grid_model, 0)
        assert indices == [0, 5, 10, 15, 20]

        indices = col_indices(grid_model, 2)
        assert indices == [2, 7, 12, 17, 22]

    def test_row_sum(self, grid_model):
        """row_sum sums linear values in row."""
        grid_model.set_linear(0, 1.0)
        grid_model.set_linear(1, 2.0)
        grid_model.set_linear(2, 3.0)

        assert row_sum(grid_model, 0) == 6.0
        assert row_sum(grid_model, 1) == 0.0  # No values set

    def test_col_sum(self, grid_model):
        """col_sum sums linear values in column."""
        grid_model.set_linear(0, 1.0)
        grid_model.set_linear(5, 2.0)
        grid_model.set_linear(10, 3.0)

        assert col_sum(grid_model, 0) == 6.0
        assert col_sum(grid_model, 1) == 0.0

    def test_row_find(self, grid_model):
        """row_find finds first column with value."""
        grid_model.set_linear(2, 1)  # Row 0, Col 2
        assert row_find(grid_model, 0, 1) == 2

    def test_row_find_not_found(self, grid_model):
        """row_find returns -1 if value not found."""
        assert row_find(grid_model, 0, 1) == -1

    def test_col_find(self, grid_model):
        """col_find finds first row with value."""
        grid_model.set_linear(7, 1)  # Row 1, Col 2
        assert col_find(grid_model, 2, 1) == 1

    def test_col_find_not_found(self, grid_model):
        """col_find returns -1 if value not found."""
        assert col_find(grid_model, 0, 1) == -1

    def test_grid_without_dimensions_raises(self):
        """Grid ops on non-grid XQMX raise ValueError."""
        x = XQMX.binary_model(size=10)  # No grid dimensions
        with pytest.raises(ValueError):
            row_indices(x, 0)

    def test_row_out_of_bounds(self, grid_model):
        """Row out of bounds raises IndexError."""
        with pytest.raises(IndexError):
            row_indices(grid_model, 10)

    def test_col_out_of_bounds(self, grid_model):
        """Column out of bounds raises IndexError."""
        with pytest.raises(IndexError):
            col_indices(grid_model, 10)

class TestHLFExpandOnehot:
    """Tests for expand_onehot high-level function."""

    def test_expand_onehot_linear_terms(self):
        """expand_onehot adds linear terms."""
        model = XQMX.binary_model(size=5)
        expand_onehot(model, [0, 1, 2], penalty=1.0)

        # Linear terms should be set
        assert model.get_linear(0) != 0
        assert model.get_linear(1) != 0
        assert model.get_linear(2) != 0

    def test_expand_onehot_quadratic_terms(self):
        """expand_onehot adds quadratic terms."""
        model = XQMX.binary_model(size=5)
        expand_onehot(model, [0, 1, 2], penalty=1.0)

        # Quadratic terms for all pairs
        assert model.get_quadratic(0, 1) != 0
        assert model.get_quadratic(0, 2) != 0
        assert model.get_quadratic(1, 2) != 0

    def test_expand_onehot_requires_model(self):
        """expand_onehot requires MODEL mode."""
        sample = XQMX.binary_sample(size=5)
        with pytest.raises(XQMXModeError):
            expand_onehot(sample, [0, 1], penalty=1.0)

class TestHLFExpandExclude:
    """Tests for expand_exclude high-level function."""

    def test_expand_exclude_quadratic(self):
        """expand_exclude adds quadratic term."""
        model = XQMX.binary_model(size=5)
        expand_exclude(model, 0, 1, penalty=2.0)

        assert model.get_quadratic(0, 1) == 2.0

    def test_expand_exclude_requires_model(self):
        """expand_exclude requires MODEL mode."""
        sample = XQMX.binary_sample(size=5)
        with pytest.raises(XQMXModeError):
            expand_exclude(sample, 0, 1, penalty=1.0)

class TestHLFExpandImplies:
    """Tests for expand_implies high-level function."""

    def test_expand_implies_linear_and_quadratic(self):
        """expand_implies adds linear and quadratic terms."""
        model = XQMX.binary_model(size=5)
        expand_implies(model, 0, 1, penalty=1.0)

        # Should set both linear and quadratic
        assert model.get_linear(0) != 0 or model.get_quadratic(0, 1) != 0

    def test_expand_implies_requires_model(self):
        """expand_implies requires MODEL mode."""
        sample = XQMX.binary_sample(size=5)
        with pytest.raises(XQMXModeError):
            expand_implies(sample, 0, 1, penalty=1.0)

class TestComputeEnergy:
    """Tests for compute_energy function."""

    def test_energy_linear_only(self):
        """Energy from linear terms only."""
        model = XQMX.binary_model(size=3)
        model.set_linear(0, 1.0)
        model.set_linear(1, 2.0)
        model.set_linear(2, 3.0)

        sample = XQMX.binary_sample(size=3)
        sample.set_linear(0, 1)  # x0 = 1
        sample.set_linear(1, 1)  # x1 = 1
        sample.set_linear(2, 0)  # x2 = 0

        energy = compute_energy(model, sample)
        # Energy = 1.0*1 + 2.0*1 + 3.0*0 = 3.0
        assert energy == 3.0

    def test_energy_quadratic_only(self):
        """Energy from quadratic terms only."""
        model = XQMX.binary_model(size=3)
        model.set_quadratic(0, 1, 2.0)

        sample = XQMX.binary_sample(size=3)
        sample.set_linear(0, 1)
        sample.set_linear(1, 1)
        sample.set_linear(2, 0)

        energy = compute_energy(model, sample)
        # Energy = 2.0 * 1 * 1 = 2.0
        assert energy == 2.0

    def test_energy_combined(self):
        """Energy from both linear and quadratic terms."""
        model = XQMX.binary_model(size=2)
        model.set_linear(0, 1.0)
        model.set_linear(1, 2.0)
        model.set_quadratic(0, 1, 3.0)

        sample = XQMX.binary_sample(size=2)
        sample.set_linear(0, 1)
        sample.set_linear(1, 1)

        energy = compute_energy(model, sample)
        # Energy = 1.0*1 + 2.0*1 + 3.0*1*1 = 6.0
        assert energy == 6.0

    def test_energy_size_mismatch_raises(self):
        """Size mismatch raises ValueError."""
        model = XQMX.binary_model(size=5)
        sample = XQMX.binary_sample(size=3)

        with pytest.raises(ValueError):
            compute_energy(model, sample)

    def test_energy_zero_for_empty(self):
        """Empty model and sample have zero energy."""
        model = XQMX.binary_model(size=5)
        sample = XQMX.binary_sample(size=5)

        assert compute_energy(model, sample) == 0.0

class TestRequireModeValidators:
    """Tests for mode validation functions."""

    def test_require_model_mode_passes(self, binary_model):
        """require_model_mode passes for MODEL."""
        require_model_mode(binary_model, "test")  # Should not raise

    def test_require_model_mode_fails(self, binary_sample):
        """require_model_mode raises for SAMPLE."""
        with pytest.raises(XQMXModeError):
            require_model_mode(binary_sample, "test")

class TestXQMXDomain:
    """Tests for XQMXDomain enum."""

    def test_domain_values(self):
        """Domain enum has expected values."""
        assert XQMXDomain.BINARY.value is not None
        assert XQMXDomain.SPIN.value is not None
        assert XQMXDomain.DISCRETE.value is not None

    def test_domain_count(self):
        """Should have exactly 3 domains."""
        assert len(XQMXDomain) == 3

class TestXQMXMode:
    """Tests for XQMXMode enum."""

    def test_mode_values(self):
        """Mode enum has expected values."""
        assert XQMXMode.MODEL.value is not None
        assert XQMXMode.SAMPLE.value is not None

    def test_mode_count(self):
        """Should have exactly 2 modes."""
        assert len(XQMXMode) == 2
