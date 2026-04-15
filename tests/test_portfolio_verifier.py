"""
Verifier tests for Portfolio Optimization.

Tests the verifier assembly with known samples, checking
validity flag and portfolio return computation.
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


def _run_portfolio_pipeline(n, data, assignment, src="asm_reduce", p_aux=50):
    returns_vec = Vec()
    for r in data["returns"]:
        returns_vec.push(r)

    cov_vec = Vec()
    for c in data["covariance"]:
        cov_vec.push(c)

    skew_vec = Vec()
    for s in data.get("skew_terms", []):
        for v in s:
            skew_vec.push(v)

    kurt_vec = Vec()
    for k in data.get("kurt_terms", []):
        for v in k:
            kurt_vec.push(v)

    num_skew = len(data.get("skew_terms", []))
    num_kurt = len(data.get("kurt_terms", []))

    enc = _assemble_and_run(
        f"programs/portfolio/{src}/encoder.xqasm",
        {0: n, 1: returns_vec, 2: cov_vec, 3: skew_vec, 4: kurt_vec,
         5: num_skew, 6: num_kurt, 7: p_aux},
    )
    model = enc.state.get_output(0)

    sample = XQMX.binary_sample(model.size)
    for i, val in enumerate(assignment):
        if val:
            sample.set_linear(i, 1)

    # Set optimal auxiliaries for REDUCE
    aux_idx = n
    for s in data.get("skew_terms", []):
        i, j, k, c = s
        w_val = assignment[i] * assignment[j] if i < len(assignment) and j < len(assignment) else 0
        if w_val:
            sample.set_linear(aux_idx, 1)
        aux_idx += 1

    for kt in data.get("kurt_terms", []):
        i, j, k, l, c = kt
        w1 = assignment[i] * assignment[j] if i < len(assignment) and j < len(assignment) else 0
        if w1:
            sample.set_linear(aux_idx, 1)
        w2 = w1 * assignment[k] if k < len(assignment) else 0
        aux_idx += 1
        if w2:
            sample.set_linear(aux_idx, 1)
        aux_idx += 1

    # Verifier
    returns_vec2 = Vec()
    for r in data["returns"]:
        returns_vec2.push(r)

    ver = _assemble_and_run(
        f"programs/portfolio/{src}/verifier.xqasm",
        {0: model, 1: sample, 2: n, 3: returns_vec2},
    )
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)
    port_return = ver.state.get_output(2)
    return energy, valid, port_return


class TestPortfolioVerifier:
    """Verifier tests for Portfolio Optimization."""

    def test_no_assets_selected_valid(self):
        """Selecting no assets is valid (unconstrained problem)."""
        data = {
            "returns": [10, 20, 30],
            "covariance": [5, 1, 2, 8, 3, 6],
            "skew_terms": [],
            "kurt_terms": [],
        }
        _, valid, ret = _run_portfolio_pipeline(3, data, [0, 0, 0])
        assert valid == 1
        assert ret == 0

    def test_all_assets_selected_valid(self):
        """Selecting all assets is valid."""
        data = {
            "returns": [10, 20, 30],
            "covariance": [5, 1, 2, 8, 3, 6],
            "skew_terms": [],
            "kurt_terms": [],
        }
        _, valid, ret = _run_portfolio_pipeline(3, data, [1, 1, 1])
        assert valid == 1
        assert ret == 60

    def test_single_asset_return(self):
        """Selecting one asset gives correct return."""
        data = {
            "returns": [10, 20, 30],
            "covariance": [5, 1, 2, 8, 3, 6],
            "skew_terms": [],
            "kurt_terms": [],
        }
        _, valid, ret = _run_portfolio_pipeline(3, data, [0, 1, 0])
        assert valid == 1
        assert ret == 20
