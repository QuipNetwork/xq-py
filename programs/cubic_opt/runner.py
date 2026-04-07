"""
Cubic Optimization Runner: Pure HOBO problem with random cubic terms.

H = sum of c_ijk * x_i * x_j * x_k

Tests REDUCE opcode on a problem where every term needs degree reduction.
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

CO_DIR = Path(__file__).parent
DEFAULT_SOURCE = "asm"
DEFAULT_P_AUX = 50


def load_program(name: str, source_dir: Path | None = None) -> AssembledProgram:
    program_dir = source_dir or (CO_DIR / DEFAULT_SOURCE)
    path = program_dir / f"{name}.xqasm"
    return assemble(path.read_text(), name)


def generate_random_cubic(
    n: int, num_terms: int, rng: random.Random, max_coeff: int = 10,
) -> list[tuple[int, int, int, int]]:
    """Generate random cubic terms (i, j, k, coefficient)."""
    terms = []
    for _ in range(num_terms):
        i, j, k = sorted(rng.sample(range(n), 3))
        coeff = rng.randint(-max_coeff, max_coeff)
        if coeff != 0:
            terms.append((i, j, k, coeff))
    return terms


def make_terms_vec(terms: list[tuple[int, int, int, int]]) -> Vec:
    v = Vec()
    for i, j, k, c in terms:
        v.push(i)
        v.push(j)
        v.push(k)
        v.push(c)
    return v


def evaluate_cubic(assignment: list[int], terms: list[tuple[int, int, int, int]]) -> int:
    """Evaluate cubic objective directly."""
    total = 0
    for i, j, k, c in terms:
        total += c * assignment[i] * assignment[j] * assignment[k]
    return total


def make_greedy_sample(n: int, terms: list[tuple[int, int, int, int]], model_size: int) -> XQMX:
    """Try all single-variable flips, pick best."""
    assignment = [0] * n
    # Greedy: flip each variable to minimize objective
    for _ in range(3):  # multiple passes
        for i in range(n):
            assignment[i] = 0
            e0 = evaluate_cubic(assignment, terms)
            assignment[i] = 1
            e1 = evaluate_cubic(assignment, terms)
            assignment[i] = 1 if e1 < e0 else 0

    sample = XQMX.binary_sample(size=model_size)
    for i in range(n):
        if assignment[i]:
            sample.set_linear(i, 1)

    # Set auxiliary variables: w_t = x_i * x_j for each term
    for t, (i, j, k, c) in enumerate(terms):
        w_val = assignment[i] * assignment[j]
        if w_val:
            sample.set_linear(n + t, 1)

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
    terms: list[tuple[int, int, int, int]],
    p_aux: int = DEFAULT_P_AUX,
    verbose: bool = False,
    trace_verbosity: int | None = None,
    source_dir: Path | None = None,
    solver: Backend | None = None,
) -> dict[str, Any]:
    t_count = len(terms)
    terms_vec = make_terms_vec(terms)

    def make_tracer(name: str) -> Tracer | None:
        if trace_verbosity is None:
            return None
        print(f"\n--- Trace: {name} ---")
        return Tracer(verbosity=trace_verbosity)

    # Encoder
    encoder = load_program("encoder", source_dir)
    enc, t_enc = run_program(
        encoder, {0: n, 1: t_count, 2: terms_vec, 3: p_aux},
        make_tracer("encoder"),
    )
    model = enc.state.get_output(0)
    assert isinstance(model, XQMX)

    # Solver
    t_solve = 0.0
    if solver is not None:
        solve_result = solver.solve(model)
        sample = solve_result.sample
        t_solve = solve_result.timing
    else:
        sample = make_greedy_sample(n, terms, model.size)

    # Verifier
    verifier = load_program("verifier", source_dir)
    ver, t_ver = run_program(
        verifier, {0: model, 1: sample, 2: n},
        make_tracer("verifier"),
    )
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)

    # Decoder
    decoder = load_program("decoder", source_dir)
    dec, t_dec = run_program(
        decoder, {0: sample, 1: n},
        make_tracer("decoder"),
    )
    assign_vec = dec.state.get_output(0)
    assert isinstance(assign_vec, Vec)
    assignment = [assign_vec.get(i) for i in range(assign_vec.length)]

    # Independent evaluation
    cubic_value = evaluate_cubic(assignment, terms)

    results: dict[str, Any] = {
        "n": n,
        "terms": t_count,
        "p_aux": p_aux,
        "encoder_time": t_enc,
        "solve_time": t_solve,
        "verifier_time": t_ver,
        "decoder_time": t_dec,
        "total_time": t_enc + t_ver + t_dec,
        "energy": energy,
        "valid": valid,
        "assignment": assignment,
        "cubic_value": cubic_value,
        "model_size": model.size,
        "model_linear": len(model.linear),
        "model_quadratic": len(model.quadratic),
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Cubic Optimization N={n}, T={t_count}, P_aux={p_aux}")
        print(f"{'=' * 50}")
        print(f"\n--- Model ---")
        print(render_info(model))
        print(f"\n--- Results ---")
        print(f"Energy (QUBO):  {energy}")
        print(f"Cubic value:    {cubic_value}")
        print(f"Valid:          {valid} ({'PASS' if valid else 'FAIL'})")
        print(f"Assignment:     {assignment}")
        print(f"\n--- Timings ---")
        print(f"Encoder:  {t_enc:.6f}s")
        if solver is not None:
            print(f"Solver:   {t_solve:.6f}s")
        print(f"Verifier: {t_ver:.6f}s")
        print(f"Decoder:  {t_dec:.6f}s")
        print(f"Total:    {results['total_time']:.6f}s")

    return results


def benchmark(
    configs: list[tuple[int, int]],
    seed: int = 42,
    source_dir: Path | None = None,
    solver: Backend | None = None,
    p_aux: int = DEFAULT_P_AUX,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    all_results = []
    for n, t in configs:
        terms = generate_random_cubic(n, t, rng)
        results = run_pipeline(
            n, terms, p_aux=p_aux, verbose=True,
            source_dir=source_dir, solver=solver,
        )
        all_results.append(results)
    return all_results


def print_summary(all_results: list[dict[str, Any]]) -> None:
    if not all_results:
        return
    print(f"\n{'=' * 90}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 90}")
    print(
        f"{'N':>5}  {'T':>5}  {'Vars':>6}  {'Linear':>8}  {'Quad':>8}  "
        f"{'Encoder':>10}  {'Total':>10}  {'Valid':>5}"
    )
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['n']:>5}  {r['terms']:>5}  {r['model_size']:>6}  "
            f"{r['model_linear']:>8}  {r['model_quadratic']:>8}  "
            f"{r['encoder_time']:>10.6f}  {r['total_time']:>10.6f}  "
            f"{'Y' if r['valid'] else 'N':>5}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XQVM Cubic Optimization")
    parser.add_argument("--bench", type=int, nargs="+", metavar="N",
        help="Benchmark: pairs of (N, T)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--src", type=str, default=DEFAULT_SOURCE, choices=["asm", "asm_reduce"])
    parser.add_argument("-n", type=int, default=6)
    parser.add_argument("-t", type=int, default=10, help="Number of cubic terms")
    parser.add_argument("--p-aux", type=int, default=DEFAULT_P_AUX)
    parser.add_argument("--no-solve", action="store_true")
    args = parser.parse_args()

    source_dir = CO_DIR / args.src

    sa_solver = None
    if not args.no_solve:
        from xqsa import NealBackend
        sa_solver = NealBackend(seed=args.seed)

    if args.bench:
        if len(args.bench) % 2 != 0:
            parser.error("--bench requires pairs of (N, T)")
        configs = [(args.bench[i], args.bench[i+1]) for i in range(0, len(args.bench), 2)]
        results = benchmark(configs, seed=args.seed, source_dir=source_dir, solver=sa_solver, p_aux=args.p_aux)
        print_summary(results)
    else:
        rng = random.Random(args.seed)
        terms = generate_random_cubic(args.n, args.t, rng)
        run_pipeline(args.n, terms, p_aux=args.p_aux, verbose=True, source_dir=source_dir, solver=sa_solver)
