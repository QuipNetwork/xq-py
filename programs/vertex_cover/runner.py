"""
Vertex Cover Runner: Assemble and execute the Vertex Cover encoder-verifier-decoder pipeline.

Benchmarks XQVM performance across multiple problem sizes using
random graphs with flat edge-pair input format.
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

VC_DIR = Path(__file__).parent
DEFAULT_SOURCE = "asm"
DEFAULT_A = 1     # vertex cost weight
DEFAULT_B = 10    # edge constraint penalty (must be > A)

def load_program(name: str, source_dir: Path | None = None) -> AssembledProgram:
    """ Load and assemble a .xqasm file from the given source directory. """
    program_dir = source_dir or (VC_DIR / DEFAULT_SOURCE)
    path = program_dir / f"{name}.xqasm"
    source = path.read_text()
    return assemble(source, name)

def make_edge_vec(edges: list[tuple[int, int]]) -> Vec:
    """ Wrap edge pairs as a flat Vec for the VM. """
    v = Vec()
    for i, j in edges:
        v.push(i)
        v.push(j)
    return v

def generate_random_graph(n: int, rng: random.Random, density: float = 0.5) -> list[tuple[int, int]]:
    """
    Generate a random graph with N vertices.

    Each possible edge is included with the given density probability.
    Edges are (i, j) with i < j.
    """
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                edges.append((i, j))
    return edges

def make_greedy_cover(n: int, edges: list[tuple[int, int]], model_size: int | None = None) -> XQMX:
    """
    Build a known-good sample using greedy vertex cover.

    For each edge, if neither endpoint is covered, add the one
    with higher degree.
    """
    size = model_size if model_size is not None else n
    sample = XQMX.binary_sample(size=size)

    # Compute degrees
    degree = [0] * n
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    covered = set()
    for i, j in edges:
        i_in = sample.get_linear(i) == 1.0
        j_in = sample.get_linear(j) == 1.0
        if not i_in and not j_in:
            # Add higher-degree vertex
            if degree[i] >= degree[j]:
                sample.set_linear(i, 1)
            else:
                sample.set_linear(j, 1)

    return sample

def verify_cover(n: int, edges: list[tuple[int, int]], cover: list[int]) -> bool:
    """ Verify that every edge has at least one endpoint in the cover. """
    cover_set = set(cover)
    for i, j in edges:
        if i not in cover_set and j not in cover_set:
            return False
    return True

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
    edges: list[tuple[int, int]],
    a: int = DEFAULT_A,
    b: int = DEFAULT_B,
    verbose: bool = False,
    trace_verbosity: int | None = None,
    source_dir: Path | None = None,
    solver: Backend | None = None,
    show_matrix: bool = False,
) -> dict[str, Any]:
    """
    Run the full Vertex Cover encoder-verifier-decoder pipeline.
    """
    edge_vec = make_edge_vec(edges)

    def make_tracer(name: str) -> Tracer | None:
        if trace_verbosity is None:
            return None
        print(f"\n--- Trace: {name} ---")
        return Tracer(verbosity=trace_verbosity)

    # Encoder
    encoder = load_program("encoder", source_dir)
    enc, t_enc = run_program(
        encoder,
        {0: n, 1: edge_vec, 2: a, 3: b},
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
        sample = make_greedy_cover(n, edges, model.size)

    # Verifier
    verifier = load_program("verifier", source_dir)
    ver, t_ver = run_program(
        verifier,
        {0: model, 1: sample, 2: n, 3: edge_vec},
        make_tracer("verifier"),
    )
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)
    cover_size = ver.state.get_output(2)

    # Decoder
    decoder = load_program("decoder", source_dir)
    dec, t_dec = run_program(
        decoder,
        {0: sample, 1: n},
        make_tracer("decoder"),
    )
    cover_vec = dec.state.get_output(0)
    assert isinstance(cover_vec, Vec)
    cover = [cover_vec.get(i) for i in range(cover_vec.length)]

    # Independent verification
    independently_valid = verify_cover(n, edges, cover)

    results: dict[str, Any] = {
        "n": n,
        "edges": len(edges),
        "a": a,
        "b": b,
        "encoder_time": t_enc,
        "solve_time": t_solve,
        "verifier_time": t_ver,
        "decoder_time": t_dec,
        "total_time": t_enc + t_ver + t_dec,
        "energy": energy,
        "valid": valid,
        "cover": cover,
        "cover_size": cover_size,
        "independently_valid": independently_valid,
        "model_size": model.size,
        "model_linear": len(model.linear),
        "model_quadratic": len(model.quadratic),
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Vertex Cover N={n}, E={len(edges)}, A={a}, B={b}")
        print(f"{'=' * 50}")
        print(f"\n--- Edges ---")
        for i, j in edges:
            i_in = i in cover
            j_in = j in cover
            marker = ""
            if i_in and j_in:
                marker = "  (both covered)"
            elif i_in:
                marker = f"  ({i} covers)"
            elif j_in:
                marker = f"  ({j} covers)"
            else:
                marker = "  *** UNCOVERED ***"
            print(f"  ({i}, {j}){marker}")
        print(f"\n--- Model ---")
        print(render_info(model))
        if show_matrix:
            print(render_matrix(model))
        print(f"\n--- Results ---")
        print(f"Energy:         {energy}")
        print(f"Valid (VM):     {valid} ({'PASS' if valid else 'FAIL'})")
        print(f"Valid (check):  {independently_valid}")
        print(f"Cover:          {cover}")
        print(f"Cover size:     {cover_size}")
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
    solver: Backend | None = None, a: int = DEFAULT_A, b: int = DEFAULT_B,
) -> list[dict[str, Any]]:
    """ Run the Vertex Cover pipeline across multiple problem sizes. """
    rng = random.Random(seed)
    all_results: list[dict[str, Any]] = []

    for n in sizes:
        edges = generate_random_graph(n, rng)
        results = run_pipeline(
            n, edges, a=a, b=b, verbose=True,
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
        f"{'N':>5}  {'Edges':>7}  {'Linear':>8}  {'Quad':>8}  "
        f"{'Encoder':>10}  {'Verifier':>10}  {'Decoder':>10}  {'Total':>10}  "
        f"{'Cover':>6}  {'Valid':>5}"
    )
    print("-" * 90)

    for r in all_results:
        print(
            f"{r['n']:>5}  {r['edges']:>7}  {r['model_linear']:>8}  {r['model_quadratic']:>8}  "
            f"{r['encoder_time']:>10.6f}  {r['verifier_time']:>10.6f}  "
            f"{r['decoder_time']:>10.6f}  {r['total_time']:>10.6f}  "
            f"{r['cover_size']:>6}  "
            f"{'Y' if r['valid'] else 'N':>5}"
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XQVM Vertex Cover Example")
    parser.add_argument(
        "--bench", type=int, nargs="+", metavar="N",
        help="Run benchmarks for given problem sizes (e.g. --bench 4 8 16)",
    )
    trace_group = parser.add_mutually_exclusive_group()
    trace_group.add_argument(
        "--trace", action="store_const", const=2, dest="trace",
        help="Trace execution with full detail",
    )
    trace_group.add_argument(
        "--trace-compact", action="store_const", const=1, dest="trace",
        help="Trace execution with compact output",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--src", type=str, default=DEFAULT_SOURCE,
        choices=["asm", "asm_atleast"],
        help="Program source directory (default: asm)",
    )
    parser.add_argument("-n", type=int, default=6, help="Vertex count (default: 6)")
    parser.add_argument("-a", type=int, default=DEFAULT_A, help=f"Vertex cost (default: {DEFAULT_A})")
    parser.add_argument("-b", type=int, default=DEFAULT_B, help=f"Edge penalty (default: {DEFAULT_B})")
    parser.add_argument("--matrix", action="store_true", help="Print XQMX matrix")
    parser.add_argument("--no-solve", action="store_true", help="Use greedy cover instead of solver")
    args = parser.parse_args()

    if args.bench and args.trace is not None:
        parser.error("--trace cannot be used with --bench")

    source_dir = VC_DIR / args.src

    sa_solver = None
    if not args.no_solve:
        from xqsa import NealBackend
        sa_solver = NealBackend(seed=args.seed)

    if args.bench:
        results = benchmark(
            args.bench, seed=args.seed, source_dir=source_dir,
            solver=sa_solver, a=args.a, b=args.b,
        )
        print_summary(results)
    else:
        rng = random.Random(args.seed)
        n = args.n
        edges = generate_random_graph(n, rng)
        run_pipeline(
            n, edges, a=args.a, b=args.b, verbose=True,
            trace_verbosity=args.trace, source_dir=source_dir,
            solver=sa_solver, show_matrix=args.matrix,
        )
