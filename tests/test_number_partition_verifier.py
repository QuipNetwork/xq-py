"""
Verifier tests for Number Partitioning.

Tests the verifier assembly with known-good and known-bad samples:
partition difference, validity, and energy ordering.
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


def _run_partition_pipeline(elements, assignment, src="asm_penaltyeq", penalty=100):
    n = len(elements)
    v = Vec()
    for e in elements:
        v.push(e)

    enc = _assemble_and_run(
        f"programs/number_partition/{src}/encoder.xqasm",
        {0: n, 1: v, 2: penalty},
    )
    model = enc.state.get_output(0)

    sample = XQMX.binary_sample(model.size)
    for i, val in enumerate(assignment):
        if val:
            sample.set_linear(i, 1)

    v2 = Vec()
    for e in elements:
        v2.push(e)
    ver = _assemble_and_run(
        f"programs/number_partition/{src}/verifier.xqasm",
        {0: model, 1: sample, 2: n, 3: v2},
    )

    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)
    difference = ver.state.get_output(2)
    return energy, valid, difference


class TestNumberPartitionVerifier:
    """Verifier tests for Number Partitioning."""

    def test_perfect_partition_valid(self):
        """[3, 7, 5, 5] with {3,7} vs {5,5} has difference 0."""
        energy, valid, diff = _run_partition_pipeline(
            [3, 7, 5, 5], [1, 1, 0, 0],
        )
        assert valid == 1
        assert diff == 0

    def test_imbalanced_partition_has_nonzero_diff(self):
        """[3, 7, 5, 5] with {3,5,5} vs {7} has difference |13-7|=6."""
        energy, valid, diff = _run_partition_pipeline(
            [3, 7, 5, 5], [1, 0, 1, 1],
        )
        assert valid == 1
        assert abs(diff) == 6

    def test_all_in_one_subset(self):
        """All elements in subset A, none in B."""
        energy, valid, diff = _run_partition_pipeline(
            [10, 20], [1, 1],
        )
        assert valid == 1
        assert abs(diff) == 30

    def test_perfect_partition_has_lowest_energy(self):
        """Perfect partition should yield lower energy than imbalanced."""
        e_perfect, _, _ = _run_partition_pipeline([3, 7, 5, 5], [1, 1, 0, 0])
        e_bad, _, _ = _run_partition_pipeline([3, 7, 5, 5], [1, 0, 1, 1])
        assert e_perfect < e_bad
