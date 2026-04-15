"""
Verifier tests for Graph Coloring.

Tests the verifier assembly with proper and improper colorings,
checking validity, conflict counts, and one-hot constraints.
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


def _run_coloring_pipeline(n, k, edges, assignment_colors, src="asm", p_oh=100, p_ex=100, p_sym=100):
    """
    assignment_colors: list of color indices per vertex (0 to k-1).
    Internally converted to one-hot binary encoding.
    """
    ev = Vec()
    for i, j in edges:
        ev.push(i); ev.push(j)

    enc = _assemble_and_run(
        f"programs/graph_coloring/{src}/encoder.xqasm",
        {0: n, 1: k, 2: len(edges), 3: ev, 4: p_oh, 5: p_ex, 6: p_sym},
    )
    model = enc.state.get_output(0)

    sample = XQMX.binary_sample(model.size)
    for v, color in enumerate(assignment_colors):
        if color >= 0:
            sample.set_linear(v * k + color, 1)

    ev2 = Vec()
    for i, j in edges:
        ev2.push(i); ev2.push(j)

    ver = _assemble_and_run(
        f"programs/graph_coloring/{src}/verifier.xqasm",
        {0: model, 1: sample, 2: n, 3: k, 4: len(edges), 5: ev2},
    )
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)
    conflicts = ver.state.get_output(2)
    return energy, valid, conflicts


class TestGraphColoringVerifier:
    """Verifier tests for Graph Coloring."""

    def test_proper_coloring_valid(self):
        """Triangle with 3 colors: each vertex gets a different color."""
        edges = [(0, 1), (1, 2), (0, 2)]
        _, valid, conflicts = _run_coloring_pipeline(3, 3, edges, [0, 1, 2])
        assert valid == 1
        assert conflicts == 0

    def test_adjacent_same_color_invalid(self):
        """Triangle: two adjacent vertices share color 0."""
        edges = [(0, 1), (1, 2), (0, 2)]
        _, valid, conflicts = _run_coloring_pipeline(3, 3, edges, [0, 0, 1])
        assert valid == 0
        assert conflicts >= 1

    def test_all_same_color_counts_conflicts(self):
        """Triangle: all vertices color 0. Should have 3 conflicts."""
        edges = [(0, 1), (1, 2), (0, 2)]
        _, valid, conflicts = _run_coloring_pipeline(3, 3, edges, [0, 0, 0])
        assert valid == 0
        assert conflicts == 3

    def test_path_graph_2_colors(self):
        """Path 0-1-2 with 2 colors: alternating coloring is valid."""
        edges = [(0, 1), (1, 2)]
        _, valid, conflicts = _run_coloring_pipeline(3, 2, edges, [0, 1, 0])
        assert valid == 1
        assert conflicts == 0

    def test_path_graph_bad_coloring(self):
        """Path 0-1-2: adjacent same color."""
        edges = [(0, 1), (1, 2)]
        _, valid, conflicts = _run_coloring_pipeline(3, 2, edges, [0, 0, 1])
        assert valid == 0
        assert conflicts == 1

    def test_4_vertices_proper_coloring(self):
        """Complete graph K4 with 4 colors."""
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        _, valid, conflicts = _run_coloring_pipeline(4, 4, edges, [0, 1, 2, 3])
        assert valid == 1
        assert conflicts == 0

    def test_isolated_vertices_any_color(self):
        """No edges: any coloring is valid."""
        _, valid, conflicts = _run_coloring_pipeline(3, 2, [], [0, 0, 0])
        assert valid == 1
        assert conflicts == 0
