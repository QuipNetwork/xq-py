"""
TSP Runner: Assemble and execute the TSP encoder-verifier-decoder pipeline.

Benchmarks XQVM performance across multiple problem sizes using
random distance matrices with upper-triangle (triu) indexing.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from xqvm.assembler import assemble, AssembledProgram
from xqvm.core.executor import Executor
from xqvm.core.vector import Vec
from xqvm.core.xqmx import XQMX, triu

from tools.tracer import Tracer
from tools.visualizer import render_info

if TYPE_CHECKING:
    from xqsa import Backend

TSP_DIR = Path(__file__).parent
DEFAULT_SOURCE = "asm"

def load_program(name: str, source_dir: Path | None = None) -> AssembledProgram:
    """ Load and assemble a .xqasm file from the given source directory. """
    program_dir = source_dir or (TSP_DIR / DEFAULT_SOURCE)
    path = program_dir / f"{name}.xqasm"
    source = path.read_text()
    return assemble(source, name)

def make_distance_vec(distances: list[int]) -> Vec:
    """ Wrap a flat distance list as a Vec for the VM. """
    v = Vec()
    for d in distances:
        v.push(d)
    return v

def make_identity_sample(n: int) -> XQMX:
    """
    Build a known-good sample: city i at position i (identity tour).

    This simulates what an annealer would return for a trivial case.
    """
    sample = XQMX.binary_sample(size=n * n, rows=n, cols=n)
    for i in range(n):
        sample.set_linear(i * n + i, 1)
    return sample

def run_program(
    program: AssembledProgram,
    input_data: dict[int, Any],
    tracer: Tracer | None = None,
) -> tuple[Executor, float]:
    """ Execute an assembled program, returning the executor and elapsed time. """
    ex = Executor(tracer=tracer)
    t0 = time.perf_counter()
    ex.execute(program.program, input_data)
    elapsed = time.perf_counter() - t0
    return ex, elapsed

def run_pipeline(
    n: int,
    distances: list[int],
    verbose: bool = False,
    trace_verbosity: int | None = None,
    source_dir: Path | None = None,
    solver: Backend | None = None,
) -> dict[str, Any]:
    """
    Run the full TSP encoder-verifier-decoder pipeline.

    Returns a results dict with timings, energy, validity, and tour.
    """
    dist_vec = make_distance_vec(distances)

    def make_tracer(name: str) -> Tracer | None:
        if trace_verbosity is None:
            return None
        print(f"\n--- Trace: {name} ---")
        return Tracer(verbosity=trace_verbosity)

    # Encoder
    encoder = load_program("encoder", source_dir)
    enc, t_enc = run_program(encoder, {0: n, 1: dist_vec}, make_tracer("encoder"))
    model = enc.state.get_output(0)
    assert isinstance(model, XQMX)

    # Solver
    t_solve = 0.0
    if solver is not None:
        solve_result = solver.solve(model)
        sample = solve_result.sample
        t_solve = solve_result.timing
    else:
        sample = make_identity_sample(n)

    # Verifier
    verifier = load_program("verifier", source_dir)
    ver, t_ver = run_program(verifier, {0: model, 1: sample, 2: n}, make_tracer("verifier"))
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)

    # Decoder
    decoder = load_program("decoder", source_dir)
    dec, t_dec = run_program(decoder, {0: sample, 1: n}, make_tracer("decoder"))
    tour_vec = dec.state.get_output(0)
    assert isinstance(tour_vec, Vec)
    tour = [tour_vec.get(i) for i in range(tour_vec.length)]

    tour_dist = 0
    for p in range(n):
        ci, cj = tour[p], tour[(p + 1) % n]
        tour_dist += distances[triu(ci, cj)]

    results: dict[str, Any] = {
        "n": n,
        "encoder_time": t_enc,
        "solve_time": t_solve,
        "verifier_time": t_ver,
        "decoder_time": t_dec,
        "total_time": t_enc + t_ver + t_dec,
        "energy": energy,
        "valid": valid,
        "tour": tour,
        "tour_distance": tour_dist,
        "model_linear": len(model.linear),
        "model_quadratic": len(model.quadratic),
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"TSP N={n}")
        print(f"{'=' * 50}")
        print(f"\n--- Model ---")
        print(render_info(model))
        print(f"\n--- Results ---")
        print(f"Energy:         {energy}")
        print(f"Valid:          {valid} ({'PASS' if valid else 'FAIL'})")
        print(f"Tour:           {tour}")
        print(f"Tour distance:  {tour_dist}")
        print(f"\n--- Timings ---")
        print(f"Encoder:  {t_enc:.6f}s")
        if solver is not None:
            print(f"Solver:   {t_solve:.6f}s")
        print(f"Verifier: {t_ver:.6f}s")
        print(f"Decoder:  {t_dec:.6f}s")
        print(f"Total:    {results['total_time']:.6f}s")

    return results

def benchmark(
    sizes: list[int], seed: int = 42, source_dir: Path | None = None,
    solver: Backend | None = None,
) -> list[dict[str, Any]]:
    """ Run the TSP pipeline across multiple problem sizes and collect results. """
    rng = random.Random(seed)
    all_results: list[dict[str, Any]] = []

    for n in sizes:
        distances = [rng.randint(1, 100) for _ in range(n * (n - 1) // 2)]
        results = run_pipeline(n, distances, verbose=True, source_dir=source_dir, solver=solver)
        all_results.append(results)

    return all_results

def print_summary(all_results: list[dict[str, Any]]) -> None:
    """ Print a summary table of benchmark results. """
    if not all_results:
        return

    print(f"\n{'=' * 80}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"{'N':>5}  {'Vars':>7}  {'Linear':>8}  {'Quad':>8}  "
        f"{'Encoder':>10}  {'Verifier':>10}  {'Decoder':>10}  {'Total':>10}  {'Valid':>5}"
    )
    print("-" * 80)

    for r in all_results:
        n = r["n"]
        print(
            f"{n:>5}  {n*n:>7}  {r['model_linear']:>8}  {r['model_quadratic']:>8}  "
            f"{r['encoder_time']:>10.6f}  {r['verifier_time']:>10.6f}  "
            f"{r['decoder_time']:>10.6f}  {r['total_time']:>10.6f}  "
            f"{'Y' if r['valid'] else 'N':>5}"
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XQVM TSP Example")
    parser.add_argument(
        "--bench", type=int, nargs="+", metavar="N",
        help="Run benchmarks for given problem sizes (e.g. --bench 4 8 16)",
    )
    trace_group = parser.add_mutually_exclusive_group()
    trace_group.add_argument(
        "--trace", action="store_const", const=2, dest="trace",
        help="Trace execution with full detail (stack + register diffs)",
    )
    trace_group.add_argument(
        "--trace-compact", action="store_const", const=1, dest="trace",
        help="Trace execution with compact one-line output",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--src", type=str, default=DEFAULT_SOURCE,
        choices=["asm", "cp"],
        help="Program source directory (default: asm)",
    )
    parser.add_argument(
        "-n", type=int, default=5,
        help="Problem size for single run (default: 5)",
    )
    parser.add_argument(
        "--no-solve", action="store_true",
        help="Disable solver and use a hardcoded sample instead",
    )
    args = parser.parse_args()

    if args.bench and args.trace is not None:
        parser.error("--trace cannot be used with --bench")

    source_dir = TSP_DIR / args.src

    sa_solver = None
    if not args.no_solve:
        from xqsa import NealBackend
        sa_solver = NealBackend(seed=args.seed)

    if args.bench:
        results = benchmark(args.bench, seed=args.seed, source_dir=source_dir, solver=sa_solver)
        print_summary(results)
    else:
        rng = random.Random(args.seed)
        n = args.n
        distances = [rng.randint(1, 100) for _ in range(n * (n - 1) // 2)]
        run_pipeline(n, distances, verbose=True, trace_verbosity=args.trace, source_dir=source_dir, solver=sa_solver)
