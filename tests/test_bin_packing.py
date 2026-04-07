"""
Tests for Bin Packing encoding on the XQVM.

Verifies asm and asm_penaltyeq produce identical models,
one-hot and capacity constraints are correct, and the
combined use of ONEHOTR + PENALTYEQ + SLACK works.
"""


import random
import pytest
from pathlib import Path

from xqvm.assembler import assemble
from xqvm.core.executor import Executor
from xqvm.core.vector import Vec
from xqvm.core.xqmx import XQMX, compute_energy


def _run_encoder(src, weights, capacity, num_bins, p_oh=200, p_cap=200):
    n = len(weights)
    wv = Vec()
    for w in weights:
        wv.push(w)
    path = Path(f"programs/bin_packing/{src}/encoder.xqasm")
    program = assemble(path.read_text(), "encoder")
    ex = Executor()
    ex.execute(program.program, {0: n, 1: num_bins, 2: wv, 3: capacity, 4: p_oh, 5: p_cap})
    return ex.state.get_output(0)


class TestBinPackingModels:
    """Tests for model equivalence between asm and asm_penaltyeq."""

    @pytest.mark.parametrize("n", [3, 4, 6])
    def test_asm_matches_penaltyeq(self, n):
        rng = random.Random(42)
        weights = [rng.randint(1, 20) for _ in range(n)]
        capacity = max(weights) + rng.randint(5, 15)
        total = sum(weights)
        num_bins = max(2, (total + capacity - 1) // capacity)

        m_asm = _run_encoder("asm", weights, capacity, num_bins)
        m_peq = _run_encoder("asm_penaltyeq", weights, capacity, num_bins)

        assert m_asm.size == m_peq.size
        assert m_asm.linear == m_peq.linear
        assert m_asm.quadratic == m_peq.quadratic


class TestBinPackingCorrectness:
    """Tests for QUBO formulation correctness."""

    def test_valid_packing_passes_verifier(self):
        """A feasible packing should pass the verifier."""
        weights = [3, 5, 4]
        capacity = 10
        num_bins = 2
        model = _run_encoder("asm_penaltyeq", weights, capacity, num_bins)

        # Put items 0,2 in bin 0 (weight=7), item 1 in bin 1 (weight=5)
        sample = XQMX.binary_sample(model.size)
        sample.set_linear(0 * 2 + 0, 1)  # item 0 -> bin 0
        sample.set_linear(1 * 2 + 1, 1)  # item 1 -> bin 1
        sample.set_linear(2 * 2 + 0, 1)  # item 2 -> bin 0

        # Set slack variables
        slacks_per_bin = capacity.bit_length()
        slack_base = 3 * 2  # N * B

        # Bin 0: weight=7, remaining=3, binary=011
        remaining_0 = capacity - 7  # 3
        for j in range(slacks_per_bin):
            if remaining_0 & (1 << j):
                sample.set_linear(slack_base + 0 * slacks_per_bin + j, 1)

        # Bin 1: weight=5, remaining=5, binary=0101
        remaining_1 = capacity - 5  # 5
        for j in range(slacks_per_bin):
            if remaining_1 & (1 << j):
                sample.set_linear(slack_base + 1 * slacks_per_bin + j, 1)

        wv = Vec()
        for w in weights:
            wv.push(w)
        path = Path("programs/bin_packing/asm/verifier.xqasm")
        program = assemble(path.read_text(), "verifier")
        ex = Executor()
        ex.execute(program.program, {0: model, 1: sample, 2: 3, 3: num_bins, 4: wv, 5: capacity})

        assert ex.state.get_output(1) == 1  # valid
        assert ex.state.get_output(2) == 2  # bins used

    def test_model_has_correct_dimensions(self):
        """Model size should be N*B + B*slacks_per_bin."""
        weights = [5, 8, 3]
        capacity = 15
        num_bins = 2
        model = _run_encoder("asm_penaltyeq", weights, capacity, num_bins)

        n = len(weights)
        slacks_per_bin = capacity.bit_length()
        expected_size = n * num_bins + num_bins * slacks_per_bin

        assert model.size == expected_size
        assert model.rows == n
        assert model.cols == num_bins

    def test_overloaded_bin_has_higher_energy(self):
        """Putting all items in one bin beyond capacity should have higher energy."""
        weights = [10, 10, 10]
        capacity = 15
        num_bins = 2
        model = _run_encoder("asm_penaltyeq", weights, capacity, num_bins)

        slacks_per_bin = capacity.bit_length()
        slack_base = 3 * 2

        # Valid: items split across bins
        s_valid = XQMX.binary_sample(model.size)
        s_valid.set_linear(0 * 2 + 0, 1)  # item 0 -> bin 0
        s_valid.set_linear(1 * 2 + 1, 1)  # item 1 -> bin 1
        s_valid.set_linear(2 * 2 + 0, 1)  # item 2 -> bin 0
        # Bin 0: weight=20 > capacity=15 -- still overloaded but less than all 3
        # Actually let's just check relative energy

        # Invalid: all in bin 0 (weight=30 >> capacity=15)
        s_invalid = XQMX.binary_sample(model.size)
        s_invalid.set_linear(0 * 2 + 0, 1)
        s_invalid.set_linear(1 * 2 + 0, 1)
        s_invalid.set_linear(2 * 2 + 0, 1)

        e_valid = compute_energy(model, s_valid)
        e_invalid = compute_energy(model, s_invalid)

        # All-in-one should have higher energy due to capacity violation
        assert e_invalid > e_valid
