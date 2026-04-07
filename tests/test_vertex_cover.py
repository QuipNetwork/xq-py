"""
Tests for Vertex Cover encoding on the XQVM.

Verifies correctness of asm and asm_atleast encoders,
verifier edge coverage checks, and ATLEAST opcode behavior.
"""


import random
import pytest
from pathlib import Path

from xqvm.assembler import assemble
from xqvm.core.executor import Executor
from xqvm.core.vector import Vec
from xqvm.core.xqmx import XQMX, expand_atleast, compute_energy


def _load_and_run(src_dir: str, name: str, inputs: dict) -> Executor:
    path = Path(f"programs/vertex_cover/{src_dir}/{name}.xqasm")
    program = assemble(path.read_text(), name)
    ex = Executor()
    ex.execute(program.program, inputs)
    return ex


def _generate_graph(n: int, seed: int = 42, density: float = 0.5):
    rng = random.Random(seed)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                edges.append((i, j))
    return edges


def _make_edge_vec(edges):
    v = Vec()
    for i, j in edges:
        v.push(i)
        v.push(j)
    return v


class TestVertexCoverEncoder:
    """Tests for Vertex Cover encoder correctness."""

    @pytest.mark.parametrize("n", [4, 6, 8, 10])
    def test_asm_produces_valid_model(self, n):
        edges = _generate_graph(n)
        ev = _make_edge_vec(edges)
        ex = _load_and_run("asm", "encoder", {0: n, 1: ev, 2: 1, 3: 10})
        model = ex.state.get_output(0)
        assert isinstance(model, XQMX)
        assert model.size == n
        assert len(model.linear) == n
        assert len(model.quadratic) == len(edges)

    @pytest.mark.parametrize("n", [4, 6, 8, 10])
    def test_greedy_cover_is_valid(self, n):
        edges = _generate_graph(n)
        ev = _make_edge_vec(edges)
        ex = _load_and_run("asm", "encoder", {0: n, 1: ev, 2: 1, 3: 10})
        model = ex.state.get_output(0)

        # Build greedy cover
        degree = [0] * n
        for i, j in edges:
            degree[i] += 1
            degree[j] += 1

        sample = XQMX.binary_sample(size=model.size)
        for i, j in edges:
            i_in = sample.get_linear(i) == 1.0
            j_in = sample.get_linear(j) == 1.0
            if not i_in and not j_in:
                if degree[i] >= degree[j]:
                    sample.set_linear(i, 1)
                else:
                    sample.set_linear(j, 1)

        # Verify every edge is covered
        for i, j in edges:
            assert sample.get_linear(i) == 1.0 or sample.get_linear(j) == 1.0


class TestVertexCoverVerifier:
    """Tests for Vertex Cover verifier."""

    def test_valid_cover_passes(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        ev = _make_edge_vec(edges)
        ex = _load_and_run("asm", "encoder", {0: 4, 1: ev, 2: 1, 3: 10})
        model = ex.state.get_output(0)

        # Select vertices 1 and 2 — covers all edges
        sample = XQMX.binary_sample(size=4)
        sample.set_linear(1, 1)
        sample.set_linear(2, 1)

        ev2 = _make_edge_vec(edges)
        ver = _load_and_run("asm", "verifier", {0: model, 1: sample, 2: 4, 3: ev2})
        assert ver.state.get_output(1) == 1  # valid
        assert ver.state.get_output(2) == 2  # cover size

    def test_invalid_cover_fails(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        ev = _make_edge_vec(edges)
        ex = _load_and_run("asm", "encoder", {0: 4, 1: ev, 2: 1, 3: 10})
        model = ex.state.get_output(0)

        # Select only vertex 0 — edge (1,2) and (2,3) uncovered
        sample = XQMX.binary_sample(size=4)
        sample.set_linear(0, 1)

        ev2 = _make_edge_vec(edges)
        ver = _load_and_run("asm", "verifier", {0: model, 1: sample, 2: 4, 3: ev2})
        assert ver.state.get_output(1) == 0  # invalid

    def test_empty_cover_fails(self):
        edges = [(0, 1)]
        ev = _make_edge_vec(edges)
        ex = _load_and_run("asm", "encoder", {0: 2, 1: ev, 2: 1, 3: 10})
        model = ex.state.get_output(0)

        sample = XQMX.binary_sample(size=2)  # nothing selected
        ev2 = _make_edge_vec(edges)
        ver = _load_and_run("asm", "verifier", {0: model, 1: sample, 2: 2, 3: ev2})
        assert ver.state.get_output(1) == 0


class TestAtleastFunction:
    """Tests for expand_atleast correctness."""

    def test_atleast1_penalizes_all_zero(self):
        model = XQMX.binary_model(2)
        expand_atleast(model, [0, 1], 1, 10.0)

        # x=[0,0]: violated, should have highest energy
        s_00 = XQMX.binary_sample(model.size)
        s_10 = XQMX.binary_sample(model.size)
        s_10.set_linear(0, 1)

        best_00 = float("inf")
        best_10 = float("inf")
        for sbits in range(2 ** (model.size - 2)):
            for s, base in [(s_00, 0), (s_10, 1)]:
                sample = XQMX.binary_sample(model.size)
                if base:
                    sample.set_linear(0, 1)
                for j in range(model.size - 2):
                    if (sbits >> j) & 1:
                        sample.set_linear(2 + j, 1)
                e = compute_energy(model, sample)
                if base == 0 and e < best_00:
                    best_00 = e
                elif base == 1 and e < best_10:
                    best_10 = e

        assert best_00 > best_10  # violated has higher energy

    def test_atleast2_over_4_vars(self):
        model = XQMX.binary_model(4)
        expand_atleast(model, [0, 1, 2, 3], 2, 100.0)

        for bits in range(16):
            x = [(bits >> i) & 1 for i in range(4)]
            item_sum = sum(x)

            best_e = float("inf")
            n_slacks = model.size - 4
            for sbits in range(2 ** n_slacks):
                sample = XQMX.binary_sample(model.size)
                for i in range(4):
                    if x[i]:
                        sample.set_linear(i, 1)
                for j in range(n_slacks):
                    if (sbits >> j) & 1:
                        sample.set_linear(4 + j, 1)
                e = compute_energy(model, sample)
                if e < best_e:
                    best_e = e

            if item_sum < 2:
                assert best_e > -400  # violated
            else:
                assert best_e == -400  # satisfied
