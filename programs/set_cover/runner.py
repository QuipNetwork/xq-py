"""
Set Cover Runner: Encode, verify, and decode the Minimum Set Cover problem.
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

SC_DIR = Path(__file__).parent
DEFAULT_SOURCE = "asm"
DEFAULT_A = 1
DEFAULT_P = 10


def load_program(name: str, source_dir: Path | None = None) -> AssembledProgram:
    program_dir = source_dir or (SC_DIR / DEFAULT_SOURCE)
    path = program_dir / f"{name}.xqasm"
    return assemble(path.read_text(), name)


def generate_random_instance(
    n: int, m: int, rng: random.Random, density: float = 0.4,
) -> list[set[int]]:
    """Generate random Set Cover instance: m subsets over universe [0, n)."""
    subsets: list[set[int]] = []
    for _ in range(m):
        s = set()
        for elem in range(n):
            if rng.random() < density:
                s.add(elem)
        if not s:
            s.add(rng.randint(0, n - 1))
        subsets.append(s)
    # Ensure all elements are covered by at least one subset
    for elem in range(n):
        if not any(elem in s for s in subsets):
            subsets[rng.randint(0, m - 1)].add(elem)
    return subsets


def make_coverage_vec(n: int, subsets: list[set[int]]) -> Vec:
    """Build coverage vec: for each element, list the subsets containing it."""
    v = Vec()
    for elem in range(n):
        covering = [j for j, s in enumerate(subsets) if elem in s]
        v.push(len(covering))
        for j in covering:
            v.push(j)
    return v


def make_greedy_sample(n: int, subsets: list[set[int]], model_size: int) -> XQMX:
    """Greedy set cover: repeatedly pick the subset covering most uncovered elements."""
    m = len(subsets)
    uncovered = set(range(n))
    selected = set()

    while uncovered:
        best_j = max(range(m), key=lambda j: len(subsets[j] & uncovered))
        selected.add(best_j)
        uncovered -= subsets[best_j]

    sample = XQMX.binary_sample(size=model_size)
    for j in selected:
        sample.set_linear(j, 1)

    # Set slack variables: figure out which slacks the model uses
    # by checking model size vs M
    total_slacks = model_size - m
    if total_slacks > 0:
        slack_start = m
        for elem in range(n):
            covering = [j for j, s in enumerate(subsets) if elem in s]
            group_size = len(covering)
            if group_size <= 1:
                continue

            excess = group_size - 1
            slack_count = excess.bit_length()

            if slack_start + slack_count > model_size:
                break

            selected_count = sum(1 for j in covering if j in selected)
            remaining = selected_count - 1

            for k in range(slack_count):
                if remaining & (1 << k):
                    sample.set_linear(slack_start + k, 1)

            slack_start += slack_count

    return sample


def verify_cover(n: int, subsets: list[set[int]], selected: list[int]) -> tuple[bool, int]:
    """Verify cover and count covered elements."""
    covered = set()
    for j in selected:
        covered |= subsets[j]
    return covered == set(range(n)), len(covered)


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
    subsets: list[set[int]],
    a: int = DEFAULT_A,
    p: int = DEFAULT_P,
    verbose: bool = False,
    trace_verbosity: int | None = None,
    source_dir: Path | None = None,
    solver: Backend | None = None,
    show_matrix: bool = False,
) -> dict[str, Any]:
    m = len(subsets)
    coverage_vec = make_coverage_vec(n, subsets)

    def make_tracer(name: str) -> Tracer | None:
        if trace_verbosity is None:
            return None
        print(f"\n--- Trace: {name} ---")
        return Tracer(verbosity=trace_verbosity)

    # Encoder
    encoder = load_program("encoder", source_dir)
    enc, t_enc = run_program(
        encoder, {0: n, 1: m, 2: coverage_vec, 3: a, 4: p},
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
        sample = make_greedy_sample(n, subsets, model.size)

    # Verifier
    verifier = load_program("verifier", source_dir)
    cov_vec2 = make_coverage_vec(n, subsets)
    ver, t_ver = run_program(
        verifier, {0: model, 1: sample, 2: n, 3: m, 4: cov_vec2},
        make_tracer("verifier"),
    )
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)
    subsets_selected = ver.state.get_output(2)
    elements_covered = ver.state.get_output(3)

    # Decoder
    decoder = load_program("decoder", source_dir)
    dec, t_dec = run_program(
        decoder, {0: sample, 1: m}, make_tracer("decoder"),
    )
    selected_vec = dec.state.get_output(0)
    assert isinstance(selected_vec, Vec)
    selected = [selected_vec.get(i) for i in range(selected_vec.length)]

    # Independent verification
    ind_valid, ind_covered = verify_cover(n, subsets, selected)

    results: dict[str, Any] = {
        "n": n,
        "m": m,
        "a": a,
        "p": p,
        "encoder_time": t_enc,
        "solve_time": t_solve,
        "verifier_time": t_ver,
        "decoder_time": t_dec,
        "total_time": t_enc + t_ver + t_dec,
        "energy": energy,
        "valid": valid,
        "subsets_selected": subsets_selected,
        "elements_covered": elements_covered,
        "selected": selected,
        "independent_valid": ind_valid,
        "independent_covered": ind_covered,
        "model_size": model.size,
        "model_linear": len(model.linear),
        "model_quadratic": len(model.quadratic),
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Set Cover N={n}, M={m}, A={a}, P={p}")
        print(f"{'=' * 50}")
        print(f"\n--- Subsets ---")
        for j, s in enumerate(subsets):
            mark = "* " if j in selected else "  "
            print(f"  {mark}S{j}: {sorted(s)}")
        print(f"\n--- Coverage ---")
        for elem in range(n):
            covering = [j for j, s in enumerate(subsets) if elem in s]
            covered = any(j in selected for j in covering)
            print(f"  e{elem}: covered by {covering} {'✓' if covered else '✗'}")
        print(f"\n--- Model ---")
        print(render_info(model))
        if show_matrix:
            print(render_matrix(model))
        print(f"\n--- Results ---")
        print(f"Energy:           {energy}")
        print(f"Valid (VM):       {valid} ({'PASS' if valid else 'FAIL'})")
        print(f"Selected:         {selected}")
        print(f"Subsets used:     {subsets_selected}")
        print(f"Elements covered: {elements_covered}/{n}")
        print(f"Cover valid:      {ind_valid}")
        print(f"\n--- Timings ---")
        print(f"Encoder:  {t_enc:.6f}s")
        if solver is not None:
            print(f"Solver:   {t_solve:.6f}s")
        print(f"Verifier: {t_ver:.6f}s")
        print(f"Decoder:  {t_dec:.6f}s")
        print(f"Total:    {results['total_time']:.6f}s")

    return results


def benchmark(
    sizes: list[tuple[int, int]],
    seed: int = 42,
    source_dir: Path | None = None,
    solver: Backend | None = None,
    a: int = DEFAULT_A,
    p: int = DEFAULT_P,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    all_results = []
    for n, m in sizes:
        subsets = generate_random_instance(n, m, rng)
        results = run_pipeline(
            n, subsets, a=a, p=p, verbose=True,
            source_dir=source_dir, solver=solver,
        )
        all_results.append(results)
    return all_results


def print_summary(all_results: list[dict[str, Any]]) -> None:
    if not all_results:
        return
    print(f"\n{'=' * 100}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 100}")
    print(
        f"{'N':>5}  {'M':>5}  {'Vars':>6}  {'Linear':>8}  {'Quad':>8}  "
        f"{'Encoder':>10}  {'Verifier':>10}  {'Decoder':>10}  {'Total':>10}  "
        f"{'Cover':>6}  {'Valid':>5}"
    )
    print("-" * 100)
    for r in all_results:
        print(
            f"{r['n']:>5}  {r['m']:>5}  {r['model_size']:>6}  "
            f"{r['model_linear']:>8}  {r['model_quadratic']:>8}  "
            f"{r['encoder_time']:>10.6f}  {r['verifier_time']:>10.6f}  "
            f"{r['decoder_time']:>10.6f}  {r['total_time']:>10.6f}  "
            f"{r['elements_covered']}/{r['n']:>3}  "
            f"{'Y' if r['valid'] else 'N':>5}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XQVM Set Cover")
    parser.add_argument(
        "--bench", type=int, nargs="+", metavar="N",
        help="Benchmark: pairs of (N, M)",
    )
    trace_group = parser.add_mutually_exclusive_group()
    trace_group.add_argument("--trace", action="store_const", const=2, dest="trace")
    trace_group.add_argument("--trace-compact", action="store_const", const=1, dest="trace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--src", type=str, default=DEFAULT_SOURCE, choices=["asm", "asm_atleast"])
    parser.add_argument("-n", type=int, default=6)
    parser.add_argument("-m", type=int, default=8)
    parser.add_argument("-a", type=int, default=DEFAULT_A)
    parser.add_argument("-p", type=int, default=DEFAULT_P)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--no-solve", action="store_true")
    args = parser.parse_args()

    source_dir = SC_DIR / args.src

    sa_solver = None
    if not args.no_solve:
        from xqsa import NealBackend
        sa_solver = NealBackend(seed=args.seed)

    if args.bench:
        if len(args.bench) % 2 != 0:
            parser.error("--bench requires pairs of (N, M)")
        sizes = [(args.bench[i], args.bench[i + 1]) for i in range(0, len(args.bench), 2)]
        results = benchmark(
            sizes, seed=args.seed, source_dir=source_dir,
            solver=sa_solver, a=args.a, p=args.p,
        )
        print_summary(results)
    else:
        rng = random.Random(args.seed)
        subsets = generate_random_instance(args.n, args.m, rng)
        run_pipeline(
            args.n, subsets, a=args.a, p=args.p, verbose=True,
            trace_verbosity=args.trace, source_dir=source_dir,
            solver=sa_solver, show_matrix=args.matrix,
        )
