"""
Tests for Number Partitioning encoding on the XQVM.

Verifies asm and asm_penaltyeq produce identical models,
and the QUBO correctly minimizes partition difference.
"""


import random
import pytest
from pathlib import Path

from xqvm.assembler import assemble
from xqvm.core.executor import Executor
from xqvm.core.vector import Vec
from xqvm.core.xqmx import XQMX, compute_energy


def _run_encoder(src: str, elements: list[int], penalty: int = 100) -> XQMX:
    n = len(elements)
    v = Vec()
    for e in elements:
        v.push(e)
    path = Path(f"programs/number_partition/{src}/encoder.xqasm")
    program = assemble(path.read_text(), "encoder")
    ex = Executor()
    ex.execute(program.program, {0: n, 1: v, 2: penalty})
    return ex.state.get_output(0)


class TestNumberPartitionModels:
    """Tests for model equivalence between asm and asm_penaltyeq."""

    @pytest.mark.parametrize("n", [4, 6, 8, 10])
    def test_asm_matches_penaltyeq(self, n):
        rng = random.Random(42)
        elements = [rng.randint(1, 100) for _ in range(n)]

        m_asm = _run_encoder("asm", elements)
        m_peq = _run_encoder("asm_penaltyeq", elements)

        assert m_asm.size == m_peq.size
        assert m_asm.linear == m_peq.linear
        assert m_asm.quadratic == m_peq.quadratic


class TestNumberPartitionCorrectness:
    """Tests for QUBO formulation correctness."""

    def test_equal_partition_has_lowest_energy(self):
        """Elements [10, 10] should prefer x=[1,0] or x=[0,1] (diff=0)."""
        model = _run_encoder("asm_penaltyeq", [10, 10])

        e_split = []
        e_same = []
        for bits in range(4):
            x = [(bits >> i) & 1 for i in range(2)]
            sample = XQMX.binary_sample(model.size)
            for i in range(2):
                if x[i]:
                    sample.set_linear(i, 1)
            e = compute_energy(model, sample)
            if sum(x) == 1:
                e_split.append(e)
            else:
                e_same.append(e)

        # Split assignments should have lower energy than putting all in one subset
        assert all(es < em for es in e_split for em in e_same)

    def test_perfect_partition_zero_penalty(self):
        """Elements [3, 7, 5, 5] has perfect partition: {3,7} vs {5,5}, diff=0."""
        elements = [3, 7, 5, 5]
        model = _run_encoder("asm_penaltyeq", elements, penalty=100)
        total = sum(elements)
        target = total // 2  # 10

        # Find assignment where subset A sums to target
        best_e = float("inf")
        for bits in range(16):
            x = [(bits >> i) & 1 for i in range(4)]
            subset_sum = sum(elements[i] for i in range(4) if x[i])
            if subset_sum == target:
                sample = XQMX.binary_sample(model.size)
                for i in range(4):
                    if x[i]:
                        sample.set_linear(i, 1)
                e = compute_energy(model, sample)
                if e < best_e:
                    best_e = e

        # Any non-perfect partition should have higher energy
        for bits in range(16):
            x = [(bits >> i) & 1 for i in range(4)]
            subset_sum = sum(elements[i] for i in range(4) if x[i])
            if subset_sum != target:
                sample = XQMX.binary_sample(model.size)
                for i in range(4):
                    if x[i]:
                        sample.set_linear(i, 1)
                e = compute_energy(model, sample)
                assert e > best_e

    def test_energy_proportional_to_imbalance(self):
        """Larger partition imbalance should give higher energy."""
        elements = [1, 2, 3, 4]
        model = _run_encoder("asm_penaltyeq", elements, penalty=100)

        energies = {}
        for bits in range(16):
            x = [(bits >> i) & 1 for i in range(4)]
            subset_sum = sum(elements[i] for i in range(4) if x[i])
            diff = abs(2 * subset_sum - sum(elements))
            sample = XQMX.binary_sample(model.size)
            for i in range(4):
                if x[i]:
                    sample.set_linear(i, 1)
            e = compute_energy(model, sample)
            if diff not in energies:
                energies[diff] = e
            else:
                assert e == energies[diff]  # same diff = same energy

        # Larger diff should have higher energy
        sorted_diffs = sorted(energies.keys())
        for i in range(len(sorted_diffs) - 1):
            assert energies[sorted_diffs[i]] < energies[sorted_diffs[i + 1]]
