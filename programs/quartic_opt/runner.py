"""
Quartic Optimization Runner: Pure HOBO with degree-4 terms.

H = sum of c_ijkl * x_i * x_j * x_k * x_l

Tests chained REDUCE: each term needs two REDUCE calls.
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

from tools.tracer import Tracer
from tools.visualizer import render_info

if TYPE_CHECKING:
    from xqsa import Backend

QO_DIR = Path(__file__).parent
DEFAULT_SOURCE = "asm"
DEFAULT_P_AUX = 50


def load_program(name: str, source_dir: Path | None = None) -> AssembledProgram:
    program_dir = source_dir or (QO_DIR / DEFAULT_SOURCE)
    path = program_dir / f"{name}.xqasm"
    return assemble(path.read_text(), name)


def generate_random_quartic(
    n: int, num_terms: int, rng: random.Random, max_coeff: int = 10,
) -> list[tuple[int, int, int, int, int]]:
    """Generate random quartic terms (i, j, k, l, coefficient)."""
    terms = []
    for _ in range(num_terms):
        if n < 4:
            continue
        i, j, k, l = sorted(rng.sample(range(n), 4))
        coeff = rng.randint(-max_coeff, max_coeff)
        if coeff != 0:
            terms.append((i, j, k, l, coeff))
    return terms


def make_terms_vec(terms: list[tuple[int, int, int, int, int]]) -> Vec:
    v = Vec()
    for i, j, k, l, c in terms:
        v.push(i)
        v.push(j)
        v.push(k)
        v.push(l)
        v.push(c)
    return v


def evaluate_quartic(assignment: list[int], terms: list[tuple[int, int, int, int, int]]) -> int:
    total = 0
    for i, j, k, l, c in terms:
        total += c * assignment[i] * assignment[j] * assignment[k] * assignment[l]
    return total


def make_greedy_sample(
    n: int, terms: list[tuple[int, int, int, int, int]], model_size: int,
) -> XQMX:
    assignment = [0] * n
    for _ in range(3):
        for i in range(n):
            assignment[i] = 0
            e0 = evaluate_quartic(assignment, terms)
            assignment[i] = 1
            e1 = evaluate_quartic(assignment, terms)
            assignment[i] = 1 if e1 < e0 else 0

    sample = XQMX.binary_sample(size=model_size)
    for i in range(n):
        if assignment[i]:
            sample.set_linear(i, 1)

    # Set auxiliary variables
    for t, (i, j, k, l, c) in enumerate(terms):
        w1_val = assignment[i] * assignment[j]
        w2_val = w1_val * assignment[k]
        if w1_val:
            sample.set_linear(n + 2 * t, 1)
        if w2_val:
            sample.set_linear(n + 2 * t + 1, 1)

    return sample


def run_program(
    program: AssembledProgram,
    input_data: dict[int, Any],
    tracer: Tracer | None = None,
) -> tuple[Executor, float]:
    ex = Executor(tracer=tracer)
    t0 = time.perf_counter()
    ex.execute(program.program, input_data)
    elapsed = time.perf_counter() - t0
    return ex, elapsed


def run_pipeline(
    n: int,
    terms: list[tuple[int, int, int, int, int]],
    p_aux: int = DEFAULT_P_AUX,
    verbose: bool = False,
    source_dir: Path | None = None,
    solver: Backend | None = None,
) -> dict[str, Any]:
    t_count = len(terms)
    terms_vec = make_terms_vec(terms)

    encoder = load_program("encoder", source_dir)
    enc, t_enc = run_program(
        encoder, {0: n, 1: t_count, 2: terms_vec, 3: p_aux},
    )
    model = enc.state.get_output(0)
    assert isinstance(model, XQMX)

    t_solve = 0.0
    if solver is not None:
        solve_result = solver.solve(model)
        sample = solve_result.sample
        t_solve = solve_result.timing
    else:
        sample = make_greedy_sample(n, terms, model.size)

    verifier = load_program("verifier", source_dir)
    ver, t_ver = run_program(
        verifier, {0: model, 1: sample, 2: n},
    )
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)

    decoder = load_program("decoder", source_dir)
    dec, t_dec = run_program(
        decoder, {0: sample, 1: n},
    )
    assign_vec = dec.state.get_output(0)
    assert isinstance(assign_vec, Vec)
    assignment = [assign_vec.get(i) for i in range(assign_vec.length)]

    quartic_value = evaluate_quartic(assignment, terms)

    results: dict[str, Any] = {
        "n": n,
        "terms": t_count,
        "encoder_time": t_enc,
        "solve_time": t_solve,
        "verifier_time": t_ver,
        "decoder_time": t_dec,
        "total_time": t_enc + t_ver + t_dec,
        "energy": energy,
        "valid": valid,
        "assignment": assignment,
        "quartic_value": quartic_value,
        "model_size": model.size,
        "model_linear": len(model.linear),
        "model_quadratic": len(model.quadratic),
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Quartic Optimization N={n}, T={t_count}, P_aux={p_aux}")
        print(f"{'=' * 50}")
        print(render_info(model))
        print(f"Energy (QUBO):  {energy}")
        print(f"Quartic value:  {quartic_value}")
        print(f"Valid:          {valid}")
        print(f"Encoder:  {t_enc:.6f}s")
        print(f"Verifier: {t_ver:.6f}s")
        print(f"Decoder:  {t_dec:.6f}s")
        print(f"Total:    {t_enc + t_ver + t_dec:.6f}s")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XQVM Quartic Optimization")
    parser.add_argument("--bench", type=int, nargs="+", metavar="N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--src", type=str, default=DEFAULT_SOURCE, choices=["asm", "asm_reduce"])
    parser.add_argument("-n", type=int, default=6)
    parser.add_argument("-t", type=int, default=10)
    parser.add_argument("--p-aux", type=int, default=DEFAULT_P_AUX)
    parser.add_argument("--no-solve", action="store_true")
    args = parser.parse_args()

    source_dir = QO_DIR / args.src
    sa_solver = None
    if not args.no_solve:
        from xqsa import NealBackend
        sa_solver = NealBackend(seed=args.seed)

    if args.bench:
        if len(args.bench) % 2 != 0:
            parser.error("--bench requires pairs of (N, T)")
        rng = random.Random(args.seed)
        for i in range(0, len(args.bench), 2):
            n, t = args.bench[i], args.bench[i + 1]
            terms = generate_random_quartic(n, t, rng)
            run_pipeline(n, terms, p_aux=args.p_aux, verbose=True, source_dir=source_dir, solver=sa_solver)
    else:
        rng = random.Random(args.seed)
        terms = generate_random_quartic(args.n, args.t, rng)
        run_pipeline(args.n, terms, p_aux=args.p_aux, verbose=True, source_dir=source_dir, solver=sa_solver)
