"""
Portfolio Optimization Runner: HOBO problem with higher moments.

H = -λ1*Σμi*xi + λ2*Σσij*xi*xj - λ3*Σsijk*xi*xj*xk + λ4*Σκijkl*xi*xj*xk*xl

Industry-relevant HOBO: skewness (cubic) and kurtosis (quartic).
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from xqvm.assembler import assemble, AssembledProgram
from xqvm.core.executor import Executor
from xqvm.core.vector import Vec
from xqvm.core.xqmx import XQMX, compute_energy

from tools.visualizer import render_info

if TYPE_CHECKING:
    from xqsa import Backend

PF_DIR = Path(__file__).parent
DEFAULT_SOURCE = "asm"
DEFAULT_P_AUX = 50


def load_program(name: str, source_dir: Path | None = None) -> AssembledProgram:
    program_dir = source_dir or (PF_DIR / DEFAULT_SOURCE)
    path = program_dir / f"{name}.xqasm"
    return assemble(path.read_text(), name)


def generate_random_portfolio(
    n: int, rng: random.Random,
    skew_density: float = 0.3,
    kurt_density: float = 0.2,
) -> dict[str, Any]:
    """Generate random portfolio optimization instance."""
    # Returns (scaled by 100)
    returns = [rng.randint(-20, 50) for _ in range(n)]

    # Covariance (upper triangle, scaled)
    covariance = []
    for i in range(n):
        for j in range(i, n):
            if i == j:
                covariance.append(rng.randint(5, 30))  # variance
            else:
                covariance.append(rng.randint(-10, 15))  # covariance

    # Skewness terms (cubic)
    skew_terms = []
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                if rng.random() < skew_density:
                    coeff = rng.randint(-5, 5)
                    if coeff != 0:
                        skew_terms.append((i, j, k, coeff))

    # Kurtosis terms (quartic)
    kurt_terms = []
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                for l in range(k, n):
                    if rng.random() < kurt_density:
                        coeff = rng.randint(1, 8)
                        kurt_terms.append((i, j, k, l, coeff))

    return {
        "returns": returns,
        "covariance": covariance,
        "skew_terms": skew_terms,
        "kurt_terms": kurt_terms,
    }


def make_greedy_sample(
    n: int, data: dict, model_size: int,
) -> XQMX:
    """Simple greedy: select assets with positive returns."""
    sample = XQMX.binary_sample(size=model_size)
    for i in range(n):
        if data["returns"][i] > 10:
            sample.set_linear(i, 1)

    # Set auxiliary variables
    assignment = [int(sample.get_linear(i)) for i in range(n)]
    aux_idx = n

    for i, j, k, c in data["skew_terms"]:
        w = assignment[i] * assignment[j]
        if w:
            sample.set_linear(aux_idx, 1)
        aux_idx += 1

    for i, j, k, l, c in data["kurt_terms"]:
        w1 = assignment[i] * assignment[j]
        w2 = w1 * assignment[k]
        if w1:
            sample.set_linear(aux_idx, 1)
        aux_idx += 1
        if w2:
            sample.set_linear(aux_idx, 1)
        aux_idx += 1

    return sample


def run_program(
    program: AssembledProgram,
    input_data: dict[int, Any],
) -> tuple[Executor, float]:
    ex = Executor()
    t0 = time.perf_counter()
    ex.execute(program.program, input_data)
    elapsed = time.perf_counter() - t0
    return ex, elapsed


def run_pipeline(
    n: int,
    data: dict[str, Any],
    p_aux: int = DEFAULT_P_AUX,
    verbose: bool = False,
    source_dir: Path | None = None,
    solver: Backend | None = None,
) -> dict[str, Any]:
    returns_vec = Vec()
    for r in data["returns"]:
        returns_vec.push(r)

    cov_vec = Vec()
    for c in data["covariance"]:
        cov_vec.push(c)

    skew_vec = Vec()
    for i, j, k, c in data["skew_terms"]:
        skew_vec.push(i)
        skew_vec.push(j)
        skew_vec.push(k)
        skew_vec.push(c)

    kurt_vec = Vec()
    for i, j, k, l, c in data["kurt_terms"]:
        kurt_vec.push(i)
        kurt_vec.push(j)
        kurt_vec.push(k)
        kurt_vec.push(l)
        kurt_vec.push(c)

    num_skew = len(data["skew_terms"])
    num_kurt = len(data["kurt_terms"])

    # Encoder
    encoder = load_program("encoder", source_dir)
    enc, t_enc = run_program(
        encoder,
        {0: n, 1: returns_vec, 2: cov_vec, 3: skew_vec, 4: kurt_vec,
         5: num_skew, 6: num_kurt, 7: p_aux},
    )
    model = enc.state.get_output(0)
    assert isinstance(model, XQMX)

    # Sample
    t_solve = 0.0
    if solver is not None:
        solve_result = solver.solve(model)
        sample = solve_result.sample
        t_solve = solve_result.timing
    else:
        sample = make_greedy_sample(n, data, model.size)

    energy = compute_energy(model, sample)

    # Verifier
    verifier = load_program("verifier", source_dir)
    ver, t_ver = run_program(
        verifier, {0: model, 1: sample, 2: n, 3: returns_vec},
    )
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)
    portfolio_return = ver.state.get_output(2)

    # Decoder
    decoder = load_program("decoder", source_dir)
    dec, t_dec = run_program(
        decoder, {0: sample, 1: n},
    )
    selected_vec = dec.state.get_output(0)
    selected = [int(selected_vec.get(i)) for i in range(selected_vec.length)]

    results = {
        "n": n,
        "num_skew": num_skew,
        "num_kurt": num_kurt,
        "encoder_time": t_enc,
        "verifier_time": t_ver,
        "decoder_time": t_dec,
        "total_time": t_enc + t_ver + t_dec,
        "solve_time": t_solve,
        "energy": energy,
        "valid": valid,
        "selected": selected,
        "portfolio_return": portfolio_return,
        "model_size": model.size,
        "model_linear": len(model.linear),
        "model_quadratic": len(model.quadratic),
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Portfolio Optimization N={n}, Skew={num_skew}, Kurt={num_kurt}")
        print(f"{'=' * 50}")
        print(render_info(model))
        print(f"Energy:     {energy}")
        print(f"Valid:      {valid} ({'PASS' if valid else 'FAIL'})")
        print(f"Selected:   {selected}")
        print(f"Return:     {portfolio_return}")
        print(f"Encoder:    {t_enc:.6f}s")
        print(f"Verifier:   {t_ver:.6f}s")
        print(f"Decoder:    {t_dec:.6f}s")
        print(f"Total:      {t_enc + t_ver + t_dec:.6f}s")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XQVM Portfolio Optimization")
    parser.add_argument("--bench", type=int, nargs="+", metavar="N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--src", type=str, default=DEFAULT_SOURCE, choices=["asm", "asm_reduce"])
    parser.add_argument("-n", type=int, default=5)
    parser.add_argument("--p-aux", type=int, default=DEFAULT_P_AUX)
    parser.add_argument("--no-solve", action="store_true")
    args = parser.parse_args()

    source_dir = PF_DIR / args.src
    sa_solver = None
    if not args.no_solve:
        from xqsa import NealBackend
        sa_solver = NealBackend(seed=args.seed)

    if args.bench:
        rng = random.Random(args.seed)
        for n in args.bench:
            data = generate_random_portfolio(n, rng)
            run_pipeline(n, data, p_aux=args.p_aux, verbose=True, source_dir=source_dir, solver=sa_solver)
    else:
        rng = random.Random(args.seed)
        data = generate_random_portfolio(args.n, rng)
        run_pipeline(args.n, data, p_aux=args.p_aux, verbose=True, source_dir=source_dir, solver=sa_solver)
