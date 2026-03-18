"""
Max-Cut Runner: Assemble and execute the Max-Cut encoder-verifier-decoder pipeline.

Benchmarks XQVM performance across multiple problem sizes using
random weighted graphs with flat edge-triple input format.
"""

import random
import time
from pathlib import Path
from typing import Any

from xqvm.assembler import assemble, AssembledProgram
from xqvm.core.executor import Executor
from xqvm.core.vector import Vec
from xqvm.core.xqmx import XQMX

from tools.tracer import Tracer
from tools.visualizer import render_info

PROGRAM_DIR = Path(__file__).parent

def load_program(name: str) -> AssembledProgram:
    """ Load and assemble a .xqasm file from the Max-Cut program directory. """
    path = PROGRAM_DIR / f"{name}.xqasm"
    source = path.read_text()
    return assemble(source, name)

def make_edge_vec(edges: list[tuple[int, int, int]]) -> Vec:
    """ Wrap edge triples as a flat Vec for the VM. """
    v = Vec()
    for i, j, w in edges:
        v.push(i)
        v.push(j)
        v.push(w)
    return v

def generate_random_graph(n: int, rng: random.Random) -> list[tuple[int, int, int]]:
    """
    Generate a random weighted graph with N nodes.

    Includes all edges (complete graph) with random weights in [1, 100].
    Edges are (i, j, w) with i < j.
    """
    edges: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            w = rng.randint(1, 100)
            edges.append((i, j, w))
    return edges

def make_bisection_sample(n: int) -> XQMX:
    """
    Build a known-good sample: bisection partition.

    Nodes [0, N/2) in set 0, nodes [N/2, N) in set 1.
    """
    sample = XQMX.binary_sample(size=n)
    for i in range(n // 2, n):
        sample.set_linear(i, 1)
    return sample

def compute_cut_value(edges: list[tuple[int, int, int]], partition: list[int]) -> int:
    """ Compute the cut value for a given partition. """
    cut = 0
    for i, j, w in edges:
        if partition[i] != partition[j]:
            cut += w
    return cut

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
    edges: list[tuple[int, int, int]],
    verbose: bool = False,
    trace_verbosity: int | None = None,
) -> dict[str, Any]:
    """
    Run the full Max-Cut encoder-verifier-decoder pipeline.

    Returns a results dict with timings, energy, validity, and partition.
    """
    edge_vec = make_edge_vec(edges)

    def make_tracer(name: str) -> Tracer | None:
        if trace_verbosity is None:
            return None
        print(f"\n--- Trace: {name} ---")
        return Tracer(verbosity=trace_verbosity)

    # Encoder
    encoder = load_program("encoder")
    enc, t_enc = run_program(encoder, {0: n, 1: edge_vec}, make_tracer("encoder"))
    model = enc.state.get_output(0)
    assert isinstance(model, XQMX)

    # Build known-good sample (bisection partition)
    sample = make_bisection_sample(n)

    # Verifier
    verifier = load_program("verifier")
    ver, t_ver = run_program(verifier, {0: model, 1: sample, 2: n}, make_tracer("verifier"))
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)

    # Decoder
    decoder = load_program("decoder")
    dec, t_dec = run_program(decoder, {0: sample, 1: n}, make_tracer("decoder"))
    part_vec = dec.state.get_output(0)
    assert isinstance(part_vec, Vec)
    partition = [part_vec.get(i) for i in range(part_vec.length)]

    cut_value = compute_cut_value(edges, partition)

    results = {
        "n": n,
        "edges": len(edges),
        "encoder_time": t_enc,
        "verifier_time": t_ver,
        "decoder_time": t_dec,
        "total_time": t_enc + t_ver + t_dec,
        "energy": energy,
        "valid": valid,
        "partition": partition,
        "cut_value": cut_value,
        "model_linear": len(model.linear),
        "model_quadratic": len(model.quadratic),
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Max-Cut N={n}, E={len(edges)}")
        print(f"{'=' * 50}")
        print(f"\n--- Model ---")
        print(render_info(model))
        print(f"\n--- Results ---")
        print(f"Energy:         {energy}")
        print(f"Valid:          {valid} ({'PASS' if valid else 'FAIL'})")
        print(f"Partition:      {partition}")
        print(f"Cut value:      {cut_value}")
        print(f"\n--- Timings ---")
        print(f"Encoder:  {t_enc:.6f}s")
        print(f"Verifier: {t_ver:.6f}s")
        print(f"Decoder:  {t_dec:.6f}s")
        print(f"Total:    {t_enc + t_ver + t_dec:.6f}s")

    return results

def benchmark(sizes: list[int], seed: int = 42) -> list[dict[str, Any]]:
    """ Run the Max-Cut pipeline across multiple problem sizes and collect results. """
    rng = random.Random(seed)
    all_results: list[dict[str, Any]] = []

    for n in sizes:
        edges = generate_random_graph(n, rng)
        results = run_pipeline(n, edges, verbose=True)
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
        f"{'N':>5}  {'Edges':>7}  {'Linear':>8}  {'Quad':>8}  "
        f"{'Encoder':>10}  {'Verifier':>10}  {'Decoder':>10}  {'Total':>10}  {'Valid':>5}"
    )
    print("-" * 80)

    for r in all_results:
        print(
            f"{r['n']:>5}  {r['edges']:>7}  {r['model_linear']:>8}  {r['model_quadratic']:>8}  "
            f"{r['encoder_time']:>10.6f}  {r['verifier_time']:>10.6f}  "
            f"{r['decoder_time']:>10.6f}  {r['total_time']:>10.6f}  "
            f"{'Y' if r['valid'] else 'N':>5}"
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XQVM Max-Cut Example")
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
    args = parser.parse_args()

    if args.bench and args.trace is not None:
        parser.error("--trace cannot be used with --bench")

    if args.bench:
        results = benchmark(args.bench, seed=args.seed)
        print_summary(results)
    else:
        rng = random.Random(args.seed)
        n = 5
        edges = generate_random_graph(n, rng)
        run_pipeline(n, edges, verbose=True, trace_verbosity=args.trace)
