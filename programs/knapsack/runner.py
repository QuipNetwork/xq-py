"""
Knapsack Runner: Assemble and execute the 0-1 Knapsack encoder-verifier-decoder pipeline.

Benchmarks XQVM performance across multiple problem sizes using
random item values and weights.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from xqvm.assembler import assemble, AssembledProgram
from xqvm.core.executor import Executor
from xqvm.core.vector import Vec
from xqvm.core.xqmx import XQMX

from tools.tracer import Tracer
from tools.visualizer import render_info, render_matrix

if TYPE_CHECKING:
    from xqsa import Backend

KNAPSACK_DIR = Path(__file__).parent
DEFAULT_SOURCE = "asm"
DEFAULT_PENALTY = 200

def load_program(name: str, source_dir: Path | None = None) -> AssembledProgram:
    """ Load and assemble a .xqasm file from the given source directory. """
    program_dir = source_dir or (KNAPSACK_DIR / DEFAULT_SOURCE)
    path = program_dir / f"{name}.xqasm"
    source = path.read_text()
    return assemble(source, name)

def make_vecs(items: list[tuple[int, int]]) -> tuple[Vec, Vec]:
    """ Split item list into value and weight vecs for the VM. """
    values = Vec()
    weights = Vec()
    for v, w in items:
        values.push(v)
        weights.push(w)
    return values, weights

def num_slack_vars(capacity: int) -> int:
    """ Compute the number of slack variables needed: floor(log2(W)) + 1. """
    if capacity <= 0:
        return 0
    count = 0
    temp = capacity
    while temp > 0:
        temp //= 2
        count += 1
    return count

def make_greedy_sample(items: list[tuple[int, int]], capacity: int) -> XQMX:
    """
    Build a known-good sample using greedy value/weight ratio.

    Selects items by descending value-to-weight ratio until capacity is full.
    Includes slack variables set to encode unused capacity in binary.
    """
    n = len(items)
    s = num_slack_vars(capacity)
    total_vars = n + s
    sample = XQMX.binary_sample(size=total_vars)

    # Sort by value/weight ratio descending
    ranked = sorted(range(n), key=lambda i: items[i][0] / items[i][1], reverse=True)

    remaining = capacity
    for i in ranked:
        v, w = items[i]
        if w <= remaining:
            sample.set_linear(i, 1)
            remaining -= w

    # Set slack variables to encode remaining capacity in binary
    for j in range(s):
        if remaining & (1 << j):
            sample.set_linear(n + j, 1)

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
    items: list[tuple[int, int]],
    capacity: int,
    penalty: int = DEFAULT_PENALTY,
    verbose: bool = False,
    trace_verbosity: int | None = None,
    source_dir: Path | None = None,
    solver: Backend | None = None,
    show_matrix: bool = False,
) -> dict[str, Any]:
    """
    Run the full Knapsack encoder-verifier-decoder pipeline.

    Args:
        items: List of (value, weight) tuples
        capacity: Knapsack capacity W
        penalty: Constraint penalty weight P
    """
    n = len(items)
    values_vec, weights_vec = make_vecs(items)

    def make_tracer(name: str) -> Tracer | None:
        if trace_verbosity is None:
            return None
        print(f"\n--- Trace: {name} ---")
        return Tracer(verbosity=trace_verbosity)

    # Encoder
    encoder = load_program("encoder", source_dir)
    enc, t_enc = run_program(
        encoder,
        {0: n, 1: values_vec, 2: weights_vec, 3: capacity, 4: penalty},
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
        sample = make_greedy_sample(items, capacity)

    # Verifier
    verifier = load_program("verifier", source_dir)
    ver, t_ver = run_program(
        verifier,
        {0: model, 1: sample, 2: n, 3: weights_vec, 4: capacity},
        make_tracer("verifier"),
    )
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)
    total_weight = ver.state.get_output(2)

    # Decoder
    decoder = load_program("decoder", source_dir)
    dec, t_dec = run_program(
        decoder,
        {0: sample, 1: n},
        make_tracer("decoder"),
    )
    selected_vec = dec.state.get_output(0)
    assert isinstance(selected_vec, Vec)
    selected = [selected_vec.get(i) for i in range(selected_vec.length)]

    total_value = sum(items[i][0] for i in selected)
    s = num_slack_vars(capacity)

    results: dict[str, Any] = {
        "n": n,
        "capacity": capacity,
        "penalty": penalty,
        "slack_vars": s,
        "total_vars": n + s,
        "encoder_time": t_enc,
        "solve_time": t_solve,
        "verifier_time": t_ver,
        "decoder_time": t_dec,
        "total_time": t_enc + t_ver + t_dec,
        "energy": energy,
        "valid": valid,
        "selected": selected,
        "total_value": total_value,
        "total_weight": total_weight,
        "model_linear": len(model.linear),
        "model_quadratic": len(model.quadratic),
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Knapsack N={n}, W={capacity}, P={penalty}")
        print(f"{'=' * 50}")
        print(f"\n--- Items ---")
        print(f"  {'#':>3}  {'Value':>7}  {'Weight':>7}")
        for i, (v, w) in enumerate(items):
            marker = " *" if i in selected else ""
            print(f"  {i:>3}  {v:>7}  {w:>7}{marker}")
        print(f"\n--- Model ---")
        print(render_info(model))
        print(f"  slack vars: {s}")
        if show_matrix:
            print(render_matrix(model))
        print(f"\n--- Results ---")
        print(f"Energy:         {energy}")
        print(f"Valid:          {valid} ({'PASS' if valid else 'FAIL'})")
        print(f"Selected:       {selected}")
        print(f"Total value:    {total_value}")
        print(f"Total weight:   {total_weight} / {capacity}")
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
    solver: Backend | None = None, penalty: int = DEFAULT_PENALTY,
) -> list[dict[str, Any]]:
    """ Run the Knapsack pipeline across multiple problem sizes. """
    rng = random.Random(seed)
    all_results: list[dict[str, Any]] = []

    for n in sizes:
        items = [(rng.randint(1, 50), rng.randint(1, 30)) for _ in range(n)]
        capacity = sum(w for _, w in items) // 2  # ~50% capacity
        results = run_pipeline(
            items, capacity, penalty=penalty, verbose=True,
            source_dir=source_dir, solver=solver,
        )
        all_results.append(results)

    return all_results

def print_summary(all_results: list[dict[str, Any]]) -> None:
    """ Print a summary table of benchmark results. """
    if not all_results:
        return

    print(f"\n{'=' * 90}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 90}")
    print(
        f"{'N':>5}  {'W':>5}  {'Vars':>6}  {'Linear':>8}  {'Quad':>8}  "
        f"{'Encoder':>10}  {'Verifier':>10}  {'Decoder':>10}  {'Total':>10}  {'Valid':>5}"
    )
    print("-" * 90)

    for r in all_results:
        print(
            f"{r['n']:>5}  {r['capacity']:>5}  {r['total_vars']:>6}  "
            f"{r['model_linear']:>8}  {r['model_quadratic']:>8}  "
            f"{r['encoder_time']:>10.6f}  {r['verifier_time']:>10.6f}  "
            f"{r['decoder_time']:>10.6f}  {r['total_time']:>10.6f}  "
            f"{'Y' if r['valid'] else 'N':>5}"
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XQVM 0-1 Knapsack Example")
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
        choices=["asm", "asm_penaltyeq", "asm_full"],
        help="Program source directory (default: asm)",
    )
    parser.add_argument(
        "-n", type=int, default=5,
        help="Number of items for single run (default: 5)",
    )
    parser.add_argument(
        "-w", type=int, default=None,
        help="Knapsack capacity (default: ~50%% of total weight)",
    )
    parser.add_argument(
        "-p", type=int, default=DEFAULT_PENALTY,
        help=f"Constraint penalty weight (default: {DEFAULT_PENALTY})",
    )
    parser.add_argument(
        "--matrix", action="store_true",
        help="Print the full XQMX coefficient matrix",
    )
    parser.add_argument(
        "--no-solve", action="store_true",
        help="Disable solver and use a greedy sample instead",
    )
    args = parser.parse_args()

    if args.bench and args.trace is not None:
        parser.error("--trace cannot be used with --bench")

    source_dir = KNAPSACK_DIR / args.src

    sa_solver = None
    if not args.no_solve:
        from xqsa import NealBackend
        sa_solver = NealBackend(seed=args.seed)

    if args.bench:
        results = benchmark(
            args.bench, seed=args.seed, source_dir=source_dir,
            solver=sa_solver, penalty=args.p,
        )
        print_summary(results)
    else:
        rng = random.Random(args.seed)
        n = args.n
        items = [(rng.randint(1, 50), rng.randint(1, 30)) for _ in range(n)]
        capacity = args.w if args.w is not None else sum(w for _, w in items) // 2
        run_pipeline(
            items, capacity, penalty=args.p, verbose=True,
            trace_verbosity=args.trace, source_dir=source_dir,
            solver=sa_solver, show_matrix=args.matrix,
        )
