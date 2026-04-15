"""
Verifier tests for Maximum Independent Set.

Tests the verifier assembly with known-good (independent) and
known-bad (adjacent selected) samples.
"""

from pathlib import Path

from xqvm.assembler import assemble
from xqvm.core.executor import Executor
from xqvm.core.vector import Vec
from xqvm.core.xqmx import XQMX


def _assemble_and_run(program_path: str, inputs: dict) -> Executor:
    path = Path(program_path)
    prog = assemble(path.read_text(), path.stem)
    ex = Executor()
    ex.execute(prog.program, inputs)
    return ex


def _run_mis_pipeline(n, edges, assignment, src="asm", a=1, p=10):
    ev = Vec()
    for i, j in edges:
        ev.push(i); ev.push(j)

    enc = _assemble_and_run(
        f"programs/max_indep_set/{src}/encoder.xqasm",
        {0: n, 1: len(edges), 2: ev, 3: a, 4: p},
    )
    model = enc.state.get_output(0)

    sample = XQMX.binary_sample(model.size)
    for i, val in enumerate(assignment):
        if val:
            sample.set_linear(i, 1)

    ev2 = Vec()
    for i, j in edges:
        ev2.push(i); ev2.push(j)

    ver = _assemble_and_run(
        f"programs/max_indep_set/{src}/verifier.xqasm",
        {0: model, 1: sample, 2: n, 3: len(edges), 4: ev2},
    )
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)
    set_size = ver.state.get_output(2)
    return energy, valid, set_size


class TestMaxIndepSetVerifier:
    """Verifier tests for Maximum Independent Set."""

    def test_independent_set_valid(self):
        """Triangle graph: selecting one vertex is independent."""
        edges = [(0, 1), (1, 2), (0, 2)]
        _, valid, size = _run_mis_pipeline(3, edges, [1, 0, 0])
        assert valid == 1
        assert size == 1

    def test_non_adjacent_pair_valid(self):
        """Path 0-1-2: selecting {0, 2} is independent."""
        edges = [(0, 1), (1, 2)]
        _, valid, size = _run_mis_pipeline(3, edges, [1, 0, 1])
        assert valid == 1
        assert size == 2

    def test_adjacent_pair_invalid(self):
        """Path 0-1-2: selecting {0, 1} is NOT independent."""
        edges = [(0, 1), (1, 2)]
        _, valid, _ = _run_mis_pipeline(3, edges, [1, 1, 0])
        assert valid == 0

    def test_all_selected_triangle_invalid(self):
        """Triangle: selecting all 3 is not independent."""
        edges = [(0, 1), (1, 2), (0, 2)]
        _, valid, _ = _run_mis_pipeline(3, edges, [1, 1, 1])
        assert valid == 0

    def test_empty_set_valid(self):
        """Selecting no vertices is trivially independent."""
        edges = [(0, 1), (1, 2)]
        _, valid, size = _run_mis_pipeline(3, edges, [0, 0, 0])
        assert valid == 1
        assert size == 0

    def test_larger_independent_set(self):
        """Star graph: selecting all leaves is independent."""
        edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
        _, valid, size = _run_mis_pipeline(5, edges, [0, 1, 1, 1, 1])
        assert valid == 1
        assert size == 4

    def test_larger_invalid_set(self):
        """Star graph: selecting center + leaf is invalid."""
        edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
        _, valid, _ = _run_mis_pipeline(5, edges, [1, 1, 0, 0, 0])
        assert valid == 0
