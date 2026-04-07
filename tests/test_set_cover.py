"""
Tests for Set Cover encoding on the XQVM.

Verifies asm and asm_atleast produce identical models,
and coverage constraints are correctly enforced.
"""


import random
import pytest
from pathlib import Path

from xqvm.assembler import assemble
from xqvm.core.executor import Executor
from xqvm.core.vector import Vec
from xqvm.core.xqmx import XQMX, compute_energy


def _make_coverage_vec(n, subsets):
    v = Vec()
    for elem in range(n):
        covering = [j for j, s in enumerate(subsets) if elem in s]
        v.push(len(covering))
        for j in covering:
            v.push(j)
    return v


def _run_encoder(src, n, subsets, a=1, p=10):
    m = len(subsets)
    cv = _make_coverage_vec(n, subsets)
    path = Path(f"programs/set_cover/{src}/encoder.xqasm")
    program = assemble(path.read_text(), "encoder")
    ex = Executor()
    ex.execute(program.program, {0: n, 1: m, 2: cv, 3: a, 4: p})
    return ex.state.get_output(0)


def _generate_instance(n, m, seed=42, density=0.4):
    rng = random.Random(seed)
    subsets = []
    for _ in range(m):
        s = set()
        for elem in range(n):
            if rng.random() < density:
                s.add(elem)
        if not s:
            s.add(rng.randint(0, n - 1))
        subsets.append(s)
    for elem in range(n):
        if not any(elem in s for s in subsets):
            subsets[rng.randint(0, m - 1)].add(elem)
    return subsets


class TestSetCoverModels:
    """Tests for model equivalence between asm and asm_atleast."""

    @pytest.mark.parametrize("n, m", [(3, 4), (4, 6), (6, 8)])
    def test_asm_matches_atleast(self, n, m):
        subsets = _generate_instance(n, m)

        m_asm = _run_encoder("asm", n, subsets)
        m_alt = _run_encoder("asm_atleast", n, subsets)

        assert m_asm.size == m_alt.size
        assert m_asm.linear == m_alt.linear
        assert m_asm.quadratic == m_alt.quadratic


class TestSetCoverCorrectness:
    """Tests for QUBO formulation correctness."""

    def test_full_cover_is_valid(self):
        """Selecting all subsets should cover everything."""
        subsets = [{0, 1}, {1, 2}, {2, 3}]
        n, m = 4, 3
        model = _run_encoder("asm_atleast", n, subsets)

        sample = XQMX.binary_sample(model.size)
        for j in range(m):
            sample.set_linear(j, 1)

        cv = _make_coverage_vec(n, subsets)
        path = Path("programs/set_cover/asm/verifier.xqasm")
        program = assemble(path.read_text(), "verifier")
        ex = Executor()
        ex.execute(program.program, {0: model, 1: sample, 2: n, 3: m, 4: cv})

        assert ex.state.get_output(1) == 1  # valid
        assert ex.state.get_output(3) == n  # all elements covered

    def test_empty_selection_fails_coverage(self):
        """Selecting no subsets should fail coverage check."""
        subsets = [{0, 1}, {1, 2}]
        n, m = 3, 2
        model = _run_encoder("asm_atleast", n, subsets)

        sample = XQMX.binary_sample(model.size)  # nothing selected

        cv = _make_coverage_vec(n, subsets)
        path = Path("programs/set_cover/asm/verifier.xqasm")
        program = assemble(path.read_text(), "verifier")
        ex = Executor()
        ex.execute(program.program, {0: model, 1: sample, 2: n, 3: m, 4: cv})

        assert ex.state.get_output(3) < n  # not all covered

    def test_minimal_cover_has_lower_energy(self):
        """A cover using fewer subsets should have lower energy than using all."""
        subsets = [{0, 1, 2, 3}]  # one subset covers everything
        n, m = 4, 1
        model = _run_encoder("asm_atleast", n, subsets, a=1, p=10)

        # Select the one subset
        s1 = XQMX.binary_sample(model.size)
        s1.set_linear(0, 1)

        # Find best slack assignment
        best_e = float("inf")
        for sbits in range(2 ** (model.size - m)):
            sample = XQMX.binary_sample(model.size)
            sample.set_linear(0, 1)
            for j in range(model.size - m):
                if (sbits >> j) & 1:
                    sample.set_linear(m + j, 1)
            e = compute_energy(model, sample)
            if e < best_e:
                best_e = e

        # Energy should be negative (A*1 = 1 cost, but penalty terms dominate)
        assert best_e < 0

    @pytest.mark.parametrize("n, m", [(3, 4), (4, 6), (6, 8)])
    def test_both_versions_find_same_cover(self, n, m):
        """Both encoder versions should produce solutions covering all elements."""
        subsets = _generate_instance(n, m)

        for src in ["asm", "asm_atleast"]:
            model = _run_encoder(src, n, subsets)

            # Greedy cover
            uncovered = set(range(n))
            selected = set()
            while uncovered:
                best_j = max(range(m), key=lambda j: len(subsets[j] & uncovered))
                selected.add(best_j)
                uncovered -= subsets[best_j]

            # Verify it covers everything
            covered = set()
            for j in selected:
                covered |= subsets[j]
            assert covered == set(range(n))
