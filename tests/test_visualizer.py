"""
Tests for the XQVM ASCII visualizer.
"""

from tools.visualizer import render_info, render_matrix, render_sparsity
from xqvm.core.xqmx import XQMX, XQMXDomain, XQMXMode

# === render_info ===


class TestRenderInfo:
    """Test XQMX info summary rendering."""

    def test_binary_model(self):
        m = XQMX.binary_model(size=4, rows=2, cols=2)
        text = render_info(m)
        assert "MODEL" in text
        assert "BINARY" in text
        assert "size: 4" in text
        assert "grid: 2x2" in text

    def test_spin_sample(self):
        s = XQMX.spin_sample(size=9, rows=3, cols=3)
        text = render_info(s)
        assert "SAMPLE" in text
        assert "SPIN" in text

    def test_no_grid(self):
        m = XQMX.binary_model(size=5)
        text = render_info(m)
        assert "grid" not in text

    def test_nnz_counts(self):
        m = XQMX.binary_model(size=3)
        m.set_linear(0, 1)
        m.set_linear(1, 2)
        m.set_quadratic(0, 1, 5)
        text = render_info(m)
        assert "2 non-zero" in text  # linear
        assert "1 non-zero" in text  # quadratic


# === render_matrix ===


class TestRenderMatrix:
    """Test ASCII matrix rendering."""

    def test_empty_xqmx(self):
        m = XQMX(mode=XQMXMode.MODEL, domain=XQMXDomain.BINARY, size=0)
        assert render_matrix(m) == "(empty)"

    def test_model_diagonal_only(self):
        m = XQMX.binary_model(size=3)
        m.set_linear(0, 1)
        m.set_linear(2, 3)
        text = render_matrix(m)
        lines = text.strip().split("\n")
        assert len(lines) == 3
        # Row 0: linear[0]=1 on diagonal, rest dots
        assert "1" in lines[0]
        assert "." in lines[0]

    def test_model_with_quadratic(self):
        m = XQMX.binary_model(size=3)
        m.set_quadratic(0, 1, 5)
        text = render_matrix(m)
        lines = text.strip().split("\n")
        # (0,1) and (1,0) should both show 5
        assert "5" in lines[0]  # row 0, col 1
        assert "5" in lines[1]  # row 1, col 0

    def test_sample_grid(self):
        s = XQMX.binary_sample(size=4, rows=2, cols=2)
        s.set_linear(0, 1)
        s.set_linear(3, 1)
        text = render_matrix(s)
        lines = text.strip().split("\n")
        assert len(lines) == 2  # 2 rows
        # (0,0)=1 and (1,1)=1
        assert "1" in lines[0]
        assert "1" in lines[1]

    def test_sample_without_grid_falls_back_to_model(self):
        s = XQMX(mode=XQMXMode.SAMPLE, domain=XQMXDomain.BINARY, size=3)
        s.set_linear(0, 1)
        text = render_matrix(s)
        lines = text.strip().split("\n")
        # Falls back to NxN model-style rendering
        assert len(lines) == 3

    def test_values_are_aligned(self):
        m = XQMX.binary_model(size=3)
        m.set_linear(0, 100)
        m.set_linear(1, 1)
        text = render_matrix(m)
        lines = text.strip().split("\n")
        # All lines should have same length (right-justified columns)
        assert len(lines[0]) == len(lines[1])


# === render_sparsity ===


class TestRenderSparsity:
    """Test sparsity pattern rendering."""

    def test_empty_xqmx(self):
        m = XQMX(mode=XQMXMode.MODEL, domain=XQMXDomain.BINARY, size=0)
        assert render_sparsity(m) == "(empty)"

    def test_model_sparsity(self):
        m = XQMX.binary_model(size=3)
        m.set_linear(0, 1)
        m.set_quadratic(0, 2, 5)
        text = render_sparsity(m)
        lines = text.strip().split("\n")
        assert len(lines) == 3
        # Row 0: linear[0] on diagonal, quadratic(0,2) off-diagonal
        assert lines[0][0] == "#"  # (0,0) linear
        assert lines[0][1] == "."  # (0,1) empty
        assert lines[0][2] == "#"  # (0,2) quadratic

    def test_model_symmetric_sparsity(self):
        m = XQMX.binary_model(size=3)
        m.set_quadratic(0, 1, 5)
        text = render_sparsity(m)
        lines = text.strip().split("\n")
        # (0,1) and (1,0) should both be #
        assert lines[0][1] == "#"
        assert lines[1][0] == "#"

    def test_sample_sparsity(self):
        s = XQMX.binary_sample(size=4, rows=2, cols=2)
        s.set_linear(0, 1)
        s.set_linear(3, 1)
        text = render_sparsity(s)
        lines = text.strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "#."
        assert lines[1] == ".#"

    def test_all_zeros(self):
        m = XQMX.binary_model(size=3)
        text = render_sparsity(m)
        lines = text.strip().split("\n")
        for line in lines:
            assert all(c == "." for c in line)

    def test_dense(self):
        m = XQMX.binary_model(size=2)
        m.set_linear(0, 1)
        m.set_linear(1, 1)
        m.set_quadratic(0, 1, 1)
        text = render_sparsity(m)
        lines = text.strip().split("\n")
        assert lines[0] == "##"
        assert lines[1] == "##"
