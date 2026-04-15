"""
Verifier tests for Cubic Optimization.

Tests the verifier assembly with known assignments,
checking valid flag for proper binary inputs.
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


def _run_cubic_pipeline(n, terms, assignment, src="asm_reduce", p_aux=50):
    tv = Vec()
    for i, j, k, c in terms:
        tv.push(i); tv.push(j); tv.push(k); tv.push(c)

    enc = _assemble_and_run(
        f"programs/cubic_opt/{src}/encoder.xqasm",
        {0: n, 1: len(terms), 2: tv, 3: p_aux},
    )
    model = enc.state.get_output(0)

    sample = XQMX.binary_sample(model.size)
    for i, val in enumerate(assignment):
        if val:
            sample.set_linear(i, 1)
    # Set optimal auxiliaries
    for t_idx, (ti, tj, tk, tc) in enumerate(terms):
        aux_idx = n + t_idx
        if aux_idx < model.size:
            w_val = assignment[ti] * assignment[tj] if ti < len(assignment) and tj < len(assignment) else 0
            if w_val:
                sample.set_linear(aux_idx, 1)

    ver = _assemble_and_run(
        f"programs/cubic_opt/{src}/verifier.xqasm",
        {0: model, 1: sample, 2: n},
    )
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)
    return energy, valid


class TestCubicOptVerifier:
    """Verifier tests for Cubic Optimization."""

    def test_all_zeros_valid(self):
        """All-zero assignment should be valid binary."""
        _, valid = _run_cubic_pipeline(3, [(0, 1, 2, 5)], [0, 0, 0])
        assert valid == 1

    def test_all_ones_valid(self):
        """All-ones assignment should be valid binary."""
        _, valid = _run_cubic_pipeline(3, [(0, 1, 2, 5)], [1, 1, 1])
        assert valid == 1

    def test_mixed_assignment_valid(self):
        """Mixed binary assignment should be valid."""
        _, valid = _run_cubic_pipeline(3, [(0, 1, 2, 3)], [1, 0, 1])
        assert valid == 1
