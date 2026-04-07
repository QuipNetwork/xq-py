"""
Bin Packing Runner: Encode, verify, and decode the Bin Packing problem.
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

BP_DIR = Path(__file__).parent
DEFAULT_SOURCE = "asm"
DEFAULT_P_OH = 200
DEFAULT_P_CAP = 200


def load_program(name: str, source_dir: Path | None = None) -> AssembledProgram:
    program_dir = source_dir or (BP_DIR / DEFAULT_SOURCE)
    path = program_dir / f"{name}.xqasm"
    return assemble(path.read_text(), name)


def generate_random_instance(
    n: int, rng: random.Random, max_weight: int = 20, capacity: int | None = None,
) -> tuple[list[int], int, int]:
    """Generate random bin packing instance. Returns (weights, capacity, num_bins)."""
    weights = [rng.randint(1, max_weight) for _ in range(n)]
    if capacity is None:
        capacity = max(weights) + rng.randint(5, 15)
    # Estimate bins needed (ceil of total_weight / capacity)
    total = sum(weights)
    num_bins = max(2, (total + capacity - 1) // capacity)
    return weights, capacity, num_bins


def make_greedy_sample(
    weights: list[int], capacity: int, num_bins: int, model_size: int,
) -> XQMX:
    """First-fit decreasing heuristic."""
    n = len(weights)
    sample = XQMX.binary_sample(size=model_size)

    # Sort items by weight descending
    ranked = sorted(range(n), key=lambda i: weights[i], reverse=True)
    bin_remaining = [capacity] * num_bins
    assignment = [-1] * n

    for i in ranked:
        for b in range(num_bins):
            if weights[i] <= bin_remaining[b]:
                assignment[i] = b
                bin_remaining[b] -= weights[i]
                sample.set_linear(i * num_bins + b, 1)
                break

    # Set slack variables for each bin
    slacks_per_bin = capacity.bit_length()
    slack_base = n * num_bins

    for b in range(num_bins):
        remaining = bin_remaining[b]
        for j in range(slacks_per_bin):
            if remaining & (1 << j):
                sample.set_linear(slack_base + b * slacks_per_bin + j, 1)

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
    weights: list[int],
    capacity: int,
    num_bins: int,
    p_oh: int = DEFAULT_P_OH,
    p_cap: int = DEFAULT_P_CAP,
    verbose: bool = False,
    trace_verbosity: int | None = None,
    source_dir: Path | None = None,
    solver: Backend | None = None,
    show_matrix: bool = False,
) -> dict[str, Any]:
    n = len(weights)
    weights_vec = Vec()
    for w in weights:
        weights_vec.push(w)

    def make_tracer(name: str) -> Tracer | None:
        if trace_verbosity is None:
            return None
        print(f"\n--- Trace: {name} ---")
        return Tracer(verbosity=trace_verbosity)

    # Encoder
    encoder = load_program("encoder", source_dir)
    enc, t_enc = run_program(
        encoder,
        {0: n, 1: num_bins, 2: weights_vec, 3: capacity, 4: p_oh, 5: p_cap},
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
        sample = make_greedy_sample(weights, capacity, num_bins, model.size)

    # Verifier
    verifier = load_program("verifier", source_dir)
    w_vec2 = Vec()
    for w in weights:
        w_vec2.push(w)
    ver, t_ver = run_program(
        verifier,
        {0: model, 1: sample, 2: n, 3: num_bins, 4: w_vec2, 5: capacity},
        make_tracer("verifier"),
    )
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)
    bins_used = ver.state.get_output(2)

    # Decoder
    decoder = load_program("decoder", source_dir)
    dec, t_dec = run_program(
        decoder, {0: sample, 1: n, 2: num_bins},
        make_tracer("decoder"),
    )
    assign_vec = dec.state.get_output(0)
    assert isinstance(assign_vec, Vec)
    assignment = [assign_vec.get(i) for i in range(assign_vec.length)]

    results: dict[str, Any] = {
        "n": n,
        "num_bins": num_bins,
        "capacity": capacity,
        "encoder_time": t_enc,
        "solve_time": t_solve,
        "verifier_time": t_ver,
        "decoder_time": t_dec,
        "total_time": t_enc + t_ver + t_dec,
        "energy": energy,
        "valid": valid,
        "bins_used": bins_used,
        "assignment": assignment,
        "model_size": model.size,
        "model_linear": len(model.linear),
        "model_quadratic": len(model.quadratic),
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Bin Packing N={n}, B={num_bins}, C={capacity}")
        print(f"{'=' * 50}")
        print(f"\n--- Items ---")
        for i, w in enumerate(weights):
            bin_assigned = assignment[i] if i < len(assignment) else -1
            print(f"  item {i}: weight={w} -> bin {bin_assigned}")
        print(f"\n--- Bins ---")
        for b in range(num_bins):
            items_in_bin = [i for i, a in enumerate(assignment) if a == b]
            bin_weight = sum(weights[i] for i in items_in_bin)
            print(f"  bin {b}: items={items_in_bin} weight={bin_weight}/{capacity}")
        print(f"\n--- Model ---")
        print(render_info(model))
        if show_matrix:
            print(render_matrix(model))
        print(f"\n--- Results ---")
        print(f"Energy:       {energy}")
        print(f"Valid (VM):   {valid} ({'PASS' if valid else 'FAIL'})")
        print(f"Bins used:    {bins_used}")
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
    rng = random.Random(seed)
    all_results = []
    for n in sizes:
        weights, capacity, num_bins = generate_random_instance(n, rng)
        results = run_pipeline(
            weights, capacity, num_bins, verbose=True,
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
        f"{'N':>5}  {'B':>3}  {'C':>4}  {'Vars':>6}  {'Linear':>8}  {'Quad':>8}  "
        f"{'Encoder':>10}  {'Verifier':>10}  {'Total':>10}  "
        f"{'Bins':>5}  {'Valid':>5}"
    )
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['n']:>5}  {r['num_bins']:>3}  {r['capacity']:>4}  {r['model_size']:>6}  "
            f"{r['model_linear']:>8}  {r['model_quadratic']:>8}  "
            f"{r['encoder_time']:>10.6f}  {r['verifier_time']:>10.6f}  "
            f"{r['total_time']:>10.6f}  "
            f"{r['bins_used']:>5}  {'Y' if r['valid'] else 'N':>5}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XQVM Bin Packing")
    parser.add_argument("--bench", type=int, nargs="+", metavar="N")
    trace_group = parser.add_mutually_exclusive_group()
    trace_group.add_argument("--trace", action="store_const", const=2, dest="trace")
    trace_group.add_argument("--trace-compact", action="store_const", const=1, dest="trace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--src", type=str, default=DEFAULT_SOURCE, choices=["asm", "asm_penaltyeq"])
    parser.add_argument("-n", type=int, default=5)
    parser.add_argument("-b", type=int, default=None, help="Number of bins (auto if not set)")
    parser.add_argument("-c", type=int, default=None, help="Bin capacity (auto if not set)")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--no-solve", action="store_true")
    args = parser.parse_args()

    source_dir = BP_DIR / args.src

    sa_solver = None
    if not args.no_solve:
        from xqsa import NealBackend
        sa_solver = NealBackend(seed=args.seed)

    if args.bench:
        results = benchmark(args.bench, seed=args.seed, source_dir=source_dir, solver=sa_solver)
        print_summary(results)
    else:
        rng = random.Random(args.seed)
        weights, capacity, num_bins = generate_random_instance(args.n, rng, capacity=args.c)
        if args.b is not None:
            num_bins = args.b
        run_pipeline(
            weights, capacity, num_bins, verbose=True,
            trace_verbosity=args.trace, source_dir=source_dir,
            solver=sa_solver, show_matrix=args.matrix,
        )
