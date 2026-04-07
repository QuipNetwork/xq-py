"""
Tests for PENALTYEQ, SLACK, and BITLEN opcodes.

Verifies correctness of the new high-level opcodes for weighted
equality constraints and slack variable generation.
"""

import pytest
import random
from pathlib import Path

from xqvm.core.xqmx import (
    XQMX,
    expand_onehot,
    expand_penalty_eq,
    compute_energy,
)
from xqvm.core.executor import Executor
from xqvm.core.program import Instruction, Program
from xqvm.core.opcodes import Opcode
from xqvm.core.vector import Vec
from xqvm.core.errors import XQMXModeError

# ============================================================================
# expand_penalty_eq function tests
# ============================================================================

class TestExpandPenaltyEq:
    """ Tests for the expand_penalty_eq XQMX function. """

    def test_matches_manual_expansion(self):
        """ PENALTYEQ produces identical terms to manual ADDLINE/ADDQUAD. """
        indices = [0, 1, 2]
        coeffs = [3, 7, 5]
        target = 10
        penalty = 50.0

        # Manual expansion
        model_manual = XQMX.binary_model(3)
        for k in range(3):
            a_k = float(coeffs[k])
            model_manual.add_linear(indices[k], penalty * a_k * (a_k - 2.0 * target))
        for k in range(3):
            for m in range(k + 1, 3):
                a_k = float(coeffs[k])
                a_m = float(coeffs[m])
                model_manual.add_quadratic(indices[k], indices[m], penalty * 2.0 * a_k * a_m)

        # PENALTYEQ expansion
        model_opcode = XQMX.binary_model(3)
        expand_penalty_eq(model_opcode, indices, coeffs, target, penalty)

        assert model_manual.linear == model_opcode.linear
        assert model_manual.quadratic == model_opcode.quadratic

    def test_matches_onehot(self):
        """ PENALTYEQ with unit coefficients and target=1 matches ONEHOT. """
        model_onehot = XQMX.binary_model(4)
        expand_onehot(model_onehot, [0, 1, 2, 3], 100.0)

        model_peq = XQMX.binary_model(4)
        expand_penalty_eq(model_peq, [0, 1, 2, 3], [1, 1, 1, 1], 1, 100.0)

        assert model_onehot.linear == model_peq.linear
        assert model_onehot.quadratic == model_peq.quadratic

    def test_satisfying_solution_has_zero_penalty(self):
        """ A solution matching the target has zero penalty contribution. """
        model = XQMX.binary_model(3)
        expand_penalty_eq(model, [0, 1, 2], [3, 7, 5], 10, 50.0)

        # x0=1, x1=1, x2=0: sum = 3+7 = 10 = target
        satisfying = XQMX.binary_sample(3)
        satisfying.set_linear(0, 1)
        satisfying.set_linear(1, 1)

        # x0=0, x1=0, x2=0: sum = 0, off by 10
        zero_sample = XQMX.binary_sample(3)

        e_sat = compute_energy(model, satisfying)
        e_zero = compute_energy(model, zero_sample)

        # Difference should be P * target^2 = 50 * 100 = 5000
        assert e_zero - e_sat == 5000.0

    def test_energy_differences_match_penalty(self):
        """ Energy differences between solutions match P*(sum-target)^2. """
        model = XQMX.binary_model(3)
        expand_penalty_eq(model, [0, 1, 2], [3, 7, 5], 10, 50.0)

        # Satisfying: x=[1,1,0], sum=10
        s1 = XQMX.binary_sample(3)
        s1.set_linear(0, 1)
        s1.set_linear(1, 1)
        e1 = compute_energy(model, s1)

        # Off by 2: x=[1,0,1], sum=8
        s2 = XQMX.binary_sample(3)
        s2.set_linear(0, 1)
        s2.set_linear(2, 1)
        e2 = compute_energy(model, s2)

        # Off by 5: x=[1,1,1], sum=15
        s3 = XQMX.binary_sample(3)
        s3.set_linear(0, 1)
        s3.set_linear(1, 1)
        s3.set_linear(2, 1)
        e3 = compute_energy(model, s3)

        assert e2 - e1 == 50.0 * (8 - 10) ** 2   # 200
        assert e3 - e1 == 50.0 * (15 - 10) ** 2   # 1250

    def test_single_variable(self):
        """ PENALTYEQ works with a single variable. """
        model = XQMX.binary_model(1)
        expand_penalty_eq(model, [0], [5], 3, 10.0)

        # linear[0] = P * a * (a - 2*b) = 10 * 5 * (5 - 6) = -50
        assert model.linear[0] == -50.0
        assert len(model.quadratic) == 0

    def test_zero_penalty(self):
        """ Zero penalty produces no terms. """
        model = XQMX.binary_model(3)
        expand_penalty_eq(model, [0, 1, 2], [1, 2, 3], 5, 0.0)

        assert len(model.linear) == 0
        assert len(model.quadratic) == 0

    def test_rejects_sample_mode(self):
        """ PENALTYEQ raises on sample mode. """
        sample = XQMX.binary_sample(3)
        with pytest.raises(XQMXModeError):
            expand_penalty_eq(sample, [0, 1, 2], [1, 1, 1], 1, 100.0)

    def test_mismatched_lengths_raises(self):
        """ Mismatched indices and coefficients raises ValueError. """
        model = XQMX.binary_model(3)
        with pytest.raises(ValueError):
            expand_penalty_eq(model, [0, 1, 2], [1, 2], 1, 100.0)

    def test_adds_to_existing_terms(self):
        """ PENALTYEQ adds to existing linear/quadratic terms. """
        model = XQMX.binary_model(3)
        model.set_linear(0, 100.0)
        model.set_quadratic(0, 1, 50.0)

        expand_penalty_eq(model, [0, 1], [1, 1], 1, 10.0)

        # linear[0] should be 100 + 10*1*(1-2) = 100 - 10 = 90
        assert model.linear[0] == 90.0
        # quad[0,1] should be 50 + 10*2*1*1 = 50 + 20 = 70
        assert model.quadratic[(0, 1)] == 70.0

# ============================================================================
# PENALTYEQ opcode execution tests
# ============================================================================

class TestPenaltyEqOpcode:
    """ Tests for the PENALTYEQ opcode via executor. """

    def _run_penaltyeq(self, indices, coeffs, target, penalty, model_size):
        """ Helper: run PENALTYEQ via assembly and return the model. """
        instructions = [
            # Create vecs
            Instruction(Opcode.VECI, (1,)),   # r1 = indices vec
            Instruction(Opcode.VECI, (2,)),   # r2 = coeffs vec
        ]

        # Push indices
        for idx in indices:
            instructions += [
                Instruction(Opcode.PUSH1, (idx,)),
                Instruction(Opcode.VECPUSH, (1,)),
            ]

        # Push coefficients
        for c in coeffs:
            instructions += [
                Instruction(Opcode.PUSH1, (c,)),
                Instruction(Opcode.VECPUSH, (2,)),
            ]

        instructions += [
            # Allocate model
            Instruction(Opcode.PUSH1, (model_size,)),
            Instruction(Opcode.BQMX, (0,)),   # r0 = model
            # PENALTYEQ
            Instruction(Opcode.PUSH1, (penalty,)),   # P
            Instruction(Opcode.PUSH1, (target,)),     # b
            Instruction(Opcode.PENALTYEQ, (0, 1, 2)),
            Instruction(Opcode.HALT),
        ]

        ex = Executor()
        ex.execute(Program(instructions))
        return ex.state.get_register(0)

    def test_opcode_matches_function(self):
        """ PENALTYEQ opcode produces same result as direct function call. """
        model_func = XQMX.binary_model(3)
        expand_penalty_eq(model_func, [0, 1, 2], [3, 7, 5], 10, 50.0)

        model_opcode = self._run_penaltyeq([0, 1, 2], [3, 7, 5], 10, 50, 3)

        assert model_func.linear == model_opcode.linear
        assert model_func.quadratic == model_opcode.quadratic

    def test_opcode_onehot_equivalence(self):
        """ PENALTYEQ opcode with unit coeffs matches ONEHOT. """
        model_onehot = XQMX.binary_model(3)
        expand_onehot(model_onehot, [0, 1, 2], 100.0)

        model_opcode = self._run_penaltyeq([0, 1, 2], [1, 1, 1], 1, 100, 3)

        assert model_onehot.linear == model_opcode.linear
        assert model_onehot.quadratic == model_opcode.quadratic

# ============================================================================
# SLACK opcode tests
# ============================================================================

class TestSlackOpcode:
    """ Tests for the SLACK opcode. """

    def _run_slack(self, start_index, capacity):
        """ Helper: run SLACK via assembly and return idx/coeff vecs. """
        instructions = [
            Instruction(Opcode.VECI, (0,)),   # r0 = idx vec
            Instruction(Opcode.VECI, (1,)),   # r1 = coeff vec
            Instruction(Opcode.PUSH1, (start_index,)),
            Instruction(Opcode.PUSH1, (capacity,)),
            Instruction(Opcode.SLACK, (0, 1)),
            Instruction(Opcode.HALT),
        ]
        ex = Executor()
        ex.execute(Program(instructions))
        idx_vec = ex.state.get_register(0)
        coeff_vec = ex.state.get_register(1)
        indices = [idx_vec.get(i) for i in range(idx_vec.length)]
        coeffs = [coeff_vec.get(i) for i in range(coeff_vec.length)]
        return indices, coeffs

    def test_capacity_7(self):
        """ W=7 needs 3 slacks: [1, 2, 4]. """
        indices, coeffs = self._run_slack(0, 7)
        assert coeffs == [1, 2, 4]
        assert indices == [0, 1, 2]

    def test_capacity_10(self):
        """ W=10 needs 4 slacks: [1, 2, 4, 8]. """
        indices, coeffs = self._run_slack(5, 10)
        assert coeffs == [1, 2, 4, 8]
        assert indices == [5, 6, 7, 8]

    def test_capacity_31(self):
        """ W=31 needs 5 slacks: [1, 2, 4, 8, 16]. """
        indices, coeffs = self._run_slack(10, 31)
        assert coeffs == [1, 2, 4, 8, 16]
        assert indices == [10, 11, 12, 13, 14]

    def test_capacity_32(self):
        """ W=32 needs 6 slacks: [1, 2, 4, 8, 16, 32]. """
        indices, coeffs = self._run_slack(0, 32)
        assert coeffs == [1, 2, 4, 8, 16, 32]
        assert len(indices) == 6

    def test_capacity_1(self):
        """ W=1 needs 1 slack: [1]. """
        indices, coeffs = self._run_slack(0, 1)
        assert coeffs == [1]
        assert indices == [0]

    def test_appends_to_existing_vec(self):
        """ SLACK appends to vecs that already have data. """
        instructions = [
            Instruction(Opcode.VECI, (0,)),
            Instruction(Opcode.VECI, (1,)),
            # Pre-populate with item data
            Instruction(Opcode.PUSH1, (99,)),
            Instruction(Opcode.VECPUSH, (0,)),
            Instruction(Opcode.PUSH1, (88,)),
            Instruction(Opcode.VECPUSH, (0,)),
            Instruction(Opcode.PUSH1, (77,)),
            Instruction(Opcode.VECPUSH, (1,)),
            Instruction(Opcode.PUSH1, (66,)),
            Instruction(Opcode.VECPUSH, (1,)),
            # Now SLACK appends
            Instruction(Opcode.PUSH1, (2,)),    # start_index = 2
            Instruction(Opcode.PUSH1, (7,)),    # capacity = 7
            Instruction(Opcode.SLACK, (0, 1)),
            Instruction(Opcode.HALT),
        ]
        ex = Executor()
        ex.execute(Program(instructions))

        idx_vec = ex.state.get_register(0)
        coeff_vec = ex.state.get_register(1)

        # Should have pre-existing + slack data
        all_indices = [idx_vec.get(i) for i in range(idx_vec.length)]
        all_coeffs = [coeff_vec.get(i) for i in range(coeff_vec.length)]

        assert all_indices == [99, 88, 2, 3, 4]
        assert all_coeffs == [77, 66, 1, 2, 4]

    def test_coefficients_are_powers_of_2(self):
        """ All slack coefficients are powers of 2. """
        for capacity in [3, 7, 15, 31, 63, 100, 127]:
            _, coeffs = self._run_slack(0, capacity)
            for c in coeffs:
                assert c & (c - 1) == 0, f"Coefficient {c} is not a power of 2"

    def test_coefficients_can_represent_capacity(self):
        """ Slack coefficients can represent all values from 0 to capacity. """
        for capacity in [5, 10, 20, 50, 100]:
            _, coeffs = self._run_slack(0, capacity)
            max_representable = sum(coeffs)
            assert max_representable >= capacity, (
                f"W={capacity}: max representable {max_representable} < capacity"
            )

# ============================================================================
# BITLEN opcode tests
# ============================================================================

class TestBitlenOpcode:
    """ Tests for the BITLEN opcode. """

    def _run_bitlen(self, value):
        """ Helper: run BITLEN and return result. """
        instructions = [
            Instruction(Opcode.PUSH1, (value,)),
            Instruction(Opcode.BITLEN),
            Instruction(Opcode.STOW, (0,)),
            Instruction(Opcode.HALT),
        ]
        ex = Executor()
        ex.execute(Program(instructions))
        return ex.state.get_register(0)

    def test_powers_of_2(self):
        """ BITLEN for powers of 2. """
        assert self._run_bitlen(1) == 1    # 1 bit
        assert self._run_bitlen(2) == 2    # 2 bits
        assert self._run_bitlen(4) == 3    # 3 bits
        assert self._run_bitlen(8) == 4    # 4 bits
        assert self._run_bitlen(16) == 5   # 5 bits
        assert self._run_bitlen(32) == 6   # 6 bits
        assert self._run_bitlen(64) == 7   # 7 bits

    def test_non_powers(self):
        """ BITLEN for non-powers of 2. """
        assert self._run_bitlen(3) == 2    # 11 in binary
        assert self._run_bitlen(7) == 3    # 111 in binary
        assert self._run_bitlen(10) == 4   # 1010 in binary
        assert self._run_bitlen(100) == 7  # 1100100 in binary

    def test_value_1(self):
        """ BITLEN(1) = 1. """
        assert self._run_bitlen(1) == 1

    def test_matches_slack_count(self):
        """ BITLEN matches the number of slack variables SLACK produces. """
        for capacity in [7, 10, 15, 31, 32, 100]:
            bitlen = self._run_bitlen(capacity)
            # SLACK produces one var per power of 2 up to capacity
            expected = 0
            power = 1
            while power <= capacity:
                expected += 1
                power *= 2
            assert bitlen == expected, f"W={capacity}: BITLEN={bitlen}, SLACK count={expected}"

# ============================================================================
# Knapsack integration tests
# ============================================================================

class TestKnapsackIntegration:
    """ Integration tests: asm vs asm_full Knapsack encoders produce identical models. """

    def _run_encoder(self, src, items, capacity, penalty=200):
        """ Helper: run a knapsack encoder and return the model. """
        from xqvm.assembler import assemble
        from xqvm.core.vector import Vec

        n = len(items)
        values_vec = Vec()
        weights_vec = Vec()
        for v, w in items:
            values_vec.push(v)
            weights_vec.push(w)

        src_dir = Path(__file__).parent.parent / "programs" / "knapsack" / src
        path = src_dir / "encoder.xqasm"
        source = path.read_text()
        program = assemble(source, "encoder")

        ex = Executor()
        ex.execute(program.program, {0: n, 1: values_vec, 2: weights_vec, 3: capacity, 4: penalty})
        return ex.state.get_output(0)

    @pytest.mark.parametrize("n", [4, 6, 8, 10, 15])
    def test_asm_matches_asm_full(self, n):
        """ asm and asm_full encoders produce identical models. """
        rng = random.Random(42)
        items = [(rng.randint(1, 50), rng.randint(1, 30)) for _ in range(n)]
        capacity = sum(w for _, w in items) // 2

        model_asm = self._run_encoder("asm", items, capacity)
        model_full = self._run_encoder("asm_full", items, capacity)

        assert model_asm.size == model_full.size
        assert model_asm.linear == model_full.linear
        assert model_asm.quadratic == model_full.quadratic

    @pytest.mark.parametrize("n", [4, 6, 8, 10, 15])
    def test_asm_matches_asm_penaltyeq(self, n):
        """ asm and asm_penaltyeq encoders produce identical models. """
        rng = random.Random(42)
        items = [(rng.randint(1, 50), rng.randint(1, 30)) for _ in range(n)]
        capacity = sum(w for _, w in items) // 2

        model_asm = self._run_encoder("asm", items, capacity)
        model_peq = self._run_encoder("asm_penaltyeq", items, capacity)

        assert model_asm.size == model_peq.size
        assert model_asm.linear == model_peq.linear
        assert model_asm.quadratic == model_peq.quadratic

    def test_greedy_sample_validates(self):
        """ Greedy sample passes verifier for asm_full encoder. """
        rng = random.Random(42)
        items = [(rng.randint(1, 50), rng.randint(1, 30)) for _ in range(8)]
        capacity = sum(w for _, w in items) // 2

        model = self._run_encoder("asm_full", items, capacity)

        # Build greedy sample
        n = len(items)
        num_slacks = 0
        temp = capacity
        while temp > 0:
            temp //= 2
            num_slacks += 1

        sample = XQMX.binary_sample(size=model.size)
        ranked = sorted(range(n), key=lambda i: items[i][0] / items[i][1], reverse=True)
        remaining = capacity
        for i in ranked:
            v, w = items[i]
            if w <= remaining:
                sample.set_linear(i, 1)
                remaining -= w

        for j in range(num_slacks):
            if remaining & (1 << j):
                sample.set_linear(n + j, 1)

        # Verify weight constraint
        total_weight = sum(int(sample.get_linear(i)) * items[i][1] for i in range(n))
        assert total_weight <= capacity

        # Verify slack encoding
        slack_sum = sum(int(sample.get_linear(n + j)) * (1 << j) for j in range(num_slacks))
        assert slack_sum == capacity - total_weight
