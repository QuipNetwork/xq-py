"""
Tests for REDUCE opcode: brute-force verification that QUBO energy
exactly matches the original HOBO polynomial for all 2^N assignments.

Each test exhaustively tries every binary assignment AND every auxiliary
variable combination, confirming the minimum QUBO energy equals the
HOBO polynomial value.
"""

import random
from pathlib import Path

from xqvm.core.xqmx import XQMX, expand_reduce, compute_energy
from xqvm.assembler import assemble
from xqvm.core.executor import Executor
from xqvm.core.vector import Vec


def _brute_force_cubic(n, terms, p_aux=50):
    """Verify QUBO energy == cubic polynomial for all 2^N assignments."""
    model = XQMX.binary_model(n)

    for i, j, k, c in terms:
        w = expand_reduce(model, i, j, float(p_aux))
        model.add_quadratic(w, k, float(c))

    num_aux = model.size - n

    for bits in range(2 ** n):
        x = [(bits >> i) & 1 for i in range(n)]
        cubic_val = sum(c * x[i] * x[j] * x[k] for i, j, k, c in terms)

        best_e = float("inf")
        for aux_bits in range(2 ** num_aux):
            sample = XQMX.binary_sample(model.size)
            for i in range(n):
                if x[i]:
                    sample.set_linear(i, 1)
            for a in range(num_aux):
                if (aux_bits >> a) & 1:
                    sample.set_linear(n + a, 1)
            e = compute_energy(model, sample)
            if e < best_e:
                best_e = e

        assert best_e == cubic_val, (
            f"x={x}: cubic={cubic_val} qubo={best_e}"
        )


def _brute_force_quartic(n, terms, p_aux=50):
    """Verify QUBO energy == quartic polynomial for all 2^N assignments."""
    model = XQMX.binary_model(n)

    for i, j, k, l, c in terms:
        w1 = expand_reduce(model, i, j, float(p_aux))
        w2 = expand_reduce(model, w1, k, float(p_aux))
        model.add_quadratic(w2, l, float(c))

    num_aux = model.size - n

    for bits in range(2 ** n):
        x = [(bits >> i) & 1 for i in range(n)]
        quartic_val = sum(
            c * x[i] * x[j] * x[k] * x[l] for i, j, k, l, c in terms
        )

        best_e = float("inf")
        for aux_bits in range(2 ** num_aux):
            sample = XQMX.binary_sample(model.size)
            for i in range(n):
                if x[i]:
                    sample.set_linear(i, 1)
            for a in range(num_aux):
                if (aux_bits >> a) & 1:
                    sample.set_linear(n + a, 1)
            e = compute_energy(model, sample)
            if e < best_e:
                best_e = e

        assert best_e == quartic_val, (
            f"x={x}: quartic={quartic_val} qubo={best_e}"
        )


class TestReduceCubic:
    """Brute-force verification of REDUCE on cubic terms."""

    def test_single_term(self):
        _brute_force_cubic(3, [(0, 1, 2, 3)])

    def test_duplicate_terms(self):
        """Two terms on same variables — coefficients should net."""
        _brute_force_cubic(3, [(0, 1, 2, 2), (0, 1, 2, -5)])

    def test_mixed_terms_n4(self):
        _brute_force_cubic(4, [
            (0, 1, 2, 4), (0, 1, 3, -2), (1, 2, 3, 7), (0, 2, 3, -3),
        ])

    def test_random_n5(self):
        rng = random.Random(42)
        terms = []
        for _ in range(10):
            i, j, k = sorted(rng.sample(range(5), 3))
            c = rng.randint(-10, 10)
            if c != 0:
                terms.append((i, j, k, c))
        _brute_force_cubic(5, terms)

    def test_random_n6(self):
        rng = random.Random(99)
        terms = []
        for _ in range(15):
            i, j, k = sorted(rng.sample(range(6), 3))
            c = rng.randint(-10, 10)
            if c != 0:
                terms.append((i, j, k, c))
        _brute_force_cubic(6, terms)


class TestReduceQuartic:
    """Brute-force verification of chained REDUCE on quartic terms."""

    def test_single_term(self):
        _brute_force_quartic(4, [(0, 1, 2, 3, 5)])

    def test_mixed_terms_n5(self):
        _brute_force_quartic(5, [
            (0, 1, 2, 3, 4), (0, 1, 2, 4, -3),
            (1, 2, 3, 4, 6), (0, 1, 3, 4, -2),
        ])

    def test_negative_coefficient(self):
        _brute_force_quartic(4, [(0, 1, 2, 3, -7)])

    def test_random_n5(self):
        rng = random.Random(42)
        terms = []
        for _ in range(5):
            i, j, k, l = sorted(rng.sample(range(5), 4))
            c = rng.randint(-10, 10)
            if c != 0:
                terms.append((i, j, k, l, c))
        _brute_force_quartic(5, terms)


class TestReduceMixed:
    """Brute-force verification of mixed cubic + quartic on one model."""

    def test_portfolio_style(self):
        """3 cubic + 2 quartic terms on same model, N=5."""
        n = 5
        cubic = [(0, 1, 2, 3), (1, 2, 3, -4), (0, 3, 4, 2)]
        quartic = [(0, 1, 2, 3, 5), (1, 2, 3, 4, -3)]

        model = XQMX.binary_model(n)

        for i, j, k, c in cubic:
            w = expand_reduce(model, i, j, 50.0)
            model.add_quadratic(w, k, float(c))

        for i, j, k, l, c in quartic:
            w1 = expand_reduce(model, i, j, 50.0)
            w2 = expand_reduce(model, w1, k, 50.0)
            model.add_quadratic(w2, l, float(c))

        num_aux = model.size - n

        for bits in range(2 ** n):
            x = [(bits >> i) & 1 for i in range(n)]
            true_val = (
                sum(c * x[i] * x[j] * x[k] for i, j, k, c in cubic)
                + sum(c * x[i] * x[j] * x[k] * x[l] for i, j, k, l, c in quartic)
            )

            best_e = float("inf")
            for aux_bits in range(2 ** num_aux):
                sample = XQMX.binary_sample(model.size)
                for i in range(n):
                    if x[i]:
                        sample.set_linear(i, 1)
                for a in range(num_aux):
                    if (aux_bits >> a) & 1:
                        sample.set_linear(n + a, 1)
                e = compute_energy(model, sample)
                if e < best_e:
                    best_e = e

            assert best_e == true_val, (
                f"x={x}: true={true_val} qubo={best_e}"
            )


class TestReduceEncoder:
    """Tests that asm and asm_reduce encoders produce identical models."""

    def test_cubic_model_equivalence(self, n=6):
        rng = random.Random(42)
        terms = []
        for _ in range(10):
            i, j, k = sorted(rng.sample(range(n), 3))
            c = rng.randint(-10, 10)
            if c != 0:
                terms.append((i, j, k, c))

        tv = Vec()
        for i, j, k, c in terms:
            tv.push(i); tv.push(j); tv.push(k); tv.push(c)

        for src in ["asm", "asm_reduce"]:
            path = Path(f"programs/cubic_opt/{src}/encoder.xqasm")
            prog = assemble(path.read_text(), "encoder")
            tv_copy = Vec()
            for i, j, k, c in terms:
                tv_copy.push(i); tv_copy.push(j); tv_copy.push(k); tv_copy.push(c)
            ex = Executor()
            ex.execute(prog.program, {0: n, 1: len(terms), 2: tv_copy, 3: 50})
            if src == "asm":
                m1 = ex.state.get_output(0)
            else:
                m2 = ex.state.get_output(0)

        assert m1.size == m2.size
        assert m1.linear == m2.linear
        assert m1.quadratic == m2.quadratic

    def test_quartic_model_equivalence(self, n=5):
        rng = random.Random(42)
        terms = []
        for _ in range(5):
            i, j, k, l = sorted(rng.sample(range(n), 4))
            c = rng.randint(-10, 10)
            if c != 0:
                terms.append((i, j, k, l, c))

        for src in ["asm", "asm_reduce"]:
            path = Path(f"programs/quartic_opt/{src}/encoder.xqasm")
            prog = assemble(path.read_text(), "encoder")
            tv = Vec()
            for i, j, k, l, c in terms:
                tv.push(i); tv.push(j); tv.push(k); tv.push(l); tv.push(c)
            ex = Executor()
            ex.execute(prog.program, {0: n, 1: len(terms), 2: tv, 3: 50})
            if src == "asm":
                m1 = ex.state.get_output(0)
            else:
                m2 = ex.state.get_output(0)

        assert m1.size == m2.size
        assert m1.linear == m2.linear
        assert m1.quadratic == m2.quadratic
