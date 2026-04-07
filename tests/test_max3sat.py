"""
Tests for Max-3-SAT encoding on the XQVM.

Verifies the HOBO degree reduction produces correct QUBO models
where energy equals the number of unsatisfied clauses.
"""


import random
import pytest
from pathlib import Path

from xqvm.assembler import assemble
from xqvm.core.executor import Executor
from xqvm.core.vector import Vec
from xqvm.core.xqmx import XQMX, compute_energy


def _make_clause_vec(clauses):
    v = Vec()
    for clause in clauses:
        for var, neg in clause:
            v.push(var)
            v.push(1 if neg else 0)
    return v


def _run_encoder(clauses, n, p=1, p_aux=10):
    m = len(clauses)
    cv = _make_clause_vec(clauses)
    path = Path("programs/max3sat/asm/encoder.xqasm")
    program = assemble(path.read_text(), "encoder")
    ex = Executor()
    ex.execute(program.program, {0: n, 1: m, 2: cv, 3: p, 4: p_aux})
    return ex.state.get_output(0)


def _evaluate_clauses(assignment, clauses):
    count = 0
    for clause in clauses:
        satisfied = False
        for var, neg in clause:
            val = (1 - assignment[var]) if neg else assignment[var]
            if val == 1:
                satisfied = True
                break
        if satisfied:
            count += 1
    return count


class TestMax3SATCorrectness:
    """Tests for QUBO formulation: energy = unsatisfied clause count."""

    def test_brute_force_small_instance(self):
        """Energy equals number of unsatisfied clauses for all 2^N assignments."""
        clauses = [
            [(0, False), (1, False), (2, True)],
            [(0, True), (2, False), (3, False)],
            [(1, False), (2, True), (3, True)],
        ]
        n, m = 4, 3
        model = _run_encoder(clauses, n)

        for bits in range(2 ** n):
            x = [(bits >> i) & 1 for i in range(n)]
            sat_count = _evaluate_clauses(x, clauses)
            unsatisfied = m - sat_count

            # Find best auxiliary assignment
            best_e = float("inf")
            for aux_bits in range(2 ** m):
                sample = XQMX.binary_sample(model.size)
                for i in range(n):
                    if x[i]:
                        sample.set_linear(i, 1)
                for j in range(m):
                    if (aux_bits >> j) & 1:
                        sample.set_linear(n + j, 1)
                e = compute_energy(model, sample)
                if e < best_e:
                    best_e = e

            assert best_e == unsatisfied

    def test_all_satisfied_has_zero_energy(self):
        """An assignment satisfying all clauses should have energy 0."""
        # All-negative clause: satisfied when all vars are 0
        clauses = [
            [(0, True), (1, True), (2, True)],
        ]
        model = _run_encoder(clauses, 3)

        # x=[0,0,0] satisfies the clause
        best_e = float("inf")
        for aux in range(2):
            sample = XQMX.binary_sample(model.size)
            if aux:
                sample.set_linear(3, 1)
            e = compute_energy(model, sample)
            if e < best_e:
                best_e = e

        assert best_e == 0

    def test_unsatisfied_has_higher_energy(self):
        """An unsatisfied assignment should have higher energy than a satisfied one."""
        clauses = [
            [(0, False), (1, False), (2, False)],
        ]
        model = _run_encoder(clauses, 3)

        # x=[0,0,0] violates the clause
        best_violated = float("inf")
        for aux in range(2):
            sample = XQMX.binary_sample(model.size)
            if aux:
                sample.set_linear(3, 1)
            e = compute_energy(model, sample)
            if e < best_violated:
                best_violated = e

        # x=[1,0,0] satisfies the clause
        best_satisfied = float("inf")
        for aux in range(2):
            sample = XQMX.binary_sample(model.size)
            sample.set_linear(0, 1)
            if aux:
                sample.set_linear(3, 1)
            e = compute_energy(model, sample)
            if e < best_satisfied:
                best_satisfied = e

        assert best_violated > best_satisfied


class TestMax3SATVerifier:
    """Tests for Max-3-SAT verifier."""

    def test_verifier_matches_independent_check(self):
        rng = random.Random(42)
        n, m = 6, 8
        clauses = []
        for _ in range(m):
            vars_chosen = rng.sample(range(n), 3)
            clause = [(v, rng.random() < 0.5) for v in vars_chosen]
            clauses.append(clause)

        model = _run_encoder(clauses, n)

        # Greedy assignment
        assignment = [0] * n
        for i in range(n):
            assignment[i] = 0
            s0 = _evaluate_clauses(assignment, clauses)
            assignment[i] = 1
            s1 = _evaluate_clauses(assignment, clauses)
            assignment[i] = 1 if s1 >= s0 else 0

        sample = XQMX.binary_sample(model.size)
        for i in range(n):
            if assignment[i]:
                sample.set_linear(i, 1)

        # Run verifier
        cv = _make_clause_vec(clauses)
        path = Path("programs/max3sat/asm/verifier.xqasm")
        program = assemble(path.read_text(), "verifier")
        ex = Executor()
        ex.execute(program.program, {0: model, 1: sample, 2: n, 3: m, 4: cv})

        vm_satisfied = ex.state.get_output(2)
        ind_satisfied = _evaluate_clauses(assignment, clauses)

        assert vm_satisfied == ind_satisfied

    @pytest.mark.parametrize("n, m", [(4, 5), (6, 10), (8, 15)])
    def test_verifier_valid_flag(self, n, m):
        rng = random.Random(42)
        clauses = []
        for _ in range(m):
            vars_chosen = rng.sample(range(n), min(3, n))
            while len(vars_chosen) < 3:
                vars_chosen.append(rng.randint(0, n - 1))
            clause = [(v, rng.random() < 0.5) for v in vars_chosen]
            clauses.append(clause)

        model = _run_encoder(clauses, n)

        sample = XQMX.binary_sample(model.size)
        for i in range(n):
            sample.set_linear(i, 1)

        cv = _make_clause_vec(clauses)
        path = Path("programs/max3sat/asm/verifier.xqasm")
        program = assemble(path.read_text(), "verifier")
        ex = Executor()
        ex.execute(program.program, {0: model, 1: sample, 2: n, 3: m, 4: cv})

        assert ex.state.get_output(1) == 1  # valid (all binary)
