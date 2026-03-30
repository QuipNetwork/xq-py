"""
Tests for the XQSA solver backend package.
"""

import pytest

dimod = pytest.importorskip("dimod", reason="dwave-neal / dimod not installed")

from xqsa import Backend, NealBackend, SolverResult
from xqvm.core.xqmx import XQMX, XQMXMode, compute_energy

# ---------------------------------------------------------------------------
# SolverResult
# ---------------------------------------------------------------------------


class TestSolverResult:
    """Tests for the SolverResult dataclass."""

    def test_construction(self) -> None:
        """SolverResult stores sample, energy, timing, metadata."""
        sample = XQMX.binary_sample(2)
        result = SolverResult(sample=sample, energy=-5.0, timing=0.1, metadata={"k": "v"})
        assert result.sample is sample
        assert result.energy == -5.0
        assert result.timing == 0.1
        assert result.metadata == {"k": "v"}

    def test_frozen(self) -> None:
        """SolverResult is immutable."""
        sample = XQMX.binary_sample(2)
        result = SolverResult(sample=sample, energy=0.0, timing=0.0)
        with pytest.raises(AttributeError):
            result.energy = 1.0

    def test_default_metadata(self) -> None:
        """Metadata defaults to empty dict."""
        sample = XQMX.binary_sample(2)
        result = SolverResult(sample=sample, energy=0.0, timing=0.0)
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------


class TestBackend:
    """Tests for the abstract Backend base class."""

    def test_cannot_instantiate(self) -> None:
        """Backend is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Backend()

    def test_validate_rejects_sample_mode(self) -> None:
        """_validate_model rejects SAMPLE mode."""

        class DummyBackend(Backend):
            def solve(self, model, **kwargs):
                pass

        backend = DummyBackend()
        sample = XQMX.binary_sample(2)
        with pytest.raises(ValueError, match="MODEL"):
            backend._validate_model(sample)

    def test_validate_rejects_discrete_domain(self) -> None:
        """_validate_model rejects DISCRETE domain."""

        class DummyBackend(Backend):
            def solve(self, model, **kwargs):
                pass

        backend = DummyBackend()
        model = XQMX.discrete_model(2, k=3)
        with pytest.raises(ValueError, match="DISCRETE"):
            backend._validate_model(model)

    def test_validate_accepts_binary_model(self) -> None:
        """_validate_model accepts BINARY MODEL."""

        class DummyBackend(Backend):
            def solve(self, model, **kwargs):
                pass

        backend = DummyBackend()
        model = XQMX.binary_model(2)
        backend._validate_model(model)  # should not raise

    def test_validate_accepts_spin_model(self) -> None:
        """_validate_model accepts SPIN MODEL."""

        class DummyBackend(Backend):
            def solve(self, model, **kwargs):
                pass

        backend = DummyBackend()
        model = XQMX.spin_model(2)
        backend._validate_model(model)  # should not raise


# ---------------------------------------------------------------------------
# NealBackend
# ---------------------------------------------------------------------------


class TestNealBackend:
    """Tests for the DWave neal simulated annealing backend."""

    def test_default_params(self) -> None:
        """NealBackend stores default parameters."""
        backend = NealBackend()
        assert backend.num_reads == 100
        assert backend.num_sweeps == 1000
        assert backend.beta_range is None
        assert backend.seed is None

    def test_custom_params(self) -> None:
        """NealBackend accepts custom parameters."""
        backend = NealBackend(num_reads=50, num_sweeps=500, seed=42)
        assert backend.num_reads == 50
        assert backend.num_sweeps == 500
        assert backend.seed == 42

    def test_solve_trivial_binary(self) -> None:
        """Solve a trivial 2-variable QUBO: minimize x0 + x1."""
        model = XQMX.binary_model(2)
        model.set_linear(0, 1.0)
        model.set_linear(1, 1.0)

        backend = NealBackend(num_reads=10, num_sweeps=100, seed=42)
        result = backend.solve(model)

        assert isinstance(result, SolverResult)
        assert isinstance(result.sample, XQMX)
        assert result.sample.mode == XQMXMode.SAMPLE
        assert result.sample.size == 2
        assert result.energy == 0.0
        assert result.timing > 0.0
        assert "num_reads" in result.metadata

    def test_solve_antiferromagnetic(self) -> None:
        """Solve x0*x1 with positive coupling: optimal is x0 != x1."""
        model = XQMX.binary_model(2)
        model.set_quadratic(0, 1, 1.0)

        backend = NealBackend(num_reads=10, num_sweeps=100, seed=42)
        result = backend.solve(model)

        x0 = result.sample.get_linear(0)
        x1 = result.sample.get_linear(1)
        assert x0 * x1 == 0.0  # at least one must be 0

    def test_solve_preserves_grid(self) -> None:
        """Solver preserves rows/cols from the model."""
        model = XQMX.binary_model(4, rows=2, cols=2)
        model.set_linear(0, 1.0)

        backend = NealBackend(num_reads=10, num_sweeps=100, seed=42)
        result = backend.solve(model)

        assert result.sample.rows == 2
        assert result.sample.cols == 2

    def test_solve_rejects_sample_mode(self) -> None:
        """Solve rejects SAMPLE mode input."""
        sample = XQMX.binary_sample(2)
        backend = NealBackend()
        with pytest.raises(ValueError, match="MODEL"):
            backend.solve(sample)

    def test_kwargs_override(self) -> None:
        """Per-call kwargs override constructor defaults."""
        model = XQMX.binary_model(2)
        model.set_linear(0, 1.0)

        backend = NealBackend(num_reads=100, seed=1)
        result = backend.solve(model, num_reads=5, seed=99)

        assert result.metadata["num_reads"] == 5
        assert result.metadata["seed"] == 99

    def test_energy_matches_compute_energy(self) -> None:
        """Solver-reported energy matches compute_energy."""
        model = XQMX.binary_model(3)
        model.set_linear(0, -2.0)
        model.set_linear(1, -3.0)
        model.set_quadratic(0, 1, 5.0)

        backend = NealBackend(num_reads=50, num_sweeps=500, seed=42)
        result = backend.solve(model)

        expected_energy = compute_energy(model, result.sample)
        assert abs(result.energy - expected_energy) < 1e-9
