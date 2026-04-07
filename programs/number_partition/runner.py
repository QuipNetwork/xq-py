"""
Number Partitioning Runner: Encode, verify, and decode the Number Partitioning problem.

Benchmarks across multiple sizes comparing existing opcodes vs PENALTYEQ.
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

NP_DIR = Path(__file__).parent
DEFAULT_SOURCE = "asm"
DEFAULT_P = 100


def load_program(name: str, source_dir: Path | None = None) -> AssembledProgram:
    program_dir = source_dir or (NP_DIR / DEFAULT_SOURCE)
    path = program_dir / f"{name}.xqasm"
    return assemble(path.read_text(), name)


def generate_random_set(n: int, rng: random.Random, max_val: int = 100) -> list[int]:
    return [rng.randint(1, max_val) for _ in range(n)]


def make_greedy_sample(elements: list[int]) -> XQMX:
    n = len(elements)
    sample = XQMX.binary_sample(size=n)
    sum_a, sum_b = 0, 0
    ranked = sorted(range(n), key=lambda i: elements[i], reverse=True)
    for i in ranked:
        if sum_a <= sum_b:
            sample.set_linear(i, 1)
            sum_a += elements[i]
        else:
            sum_b += elements[i]
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
    elements: list[int],
    p: int = DEFAULT_P,
    verbose: bool = False,
    trace_verbosity: int | None = None,
    source_dir: Path | None = None,
    solver: Backend | None = None,
    show_matrix: bool = False,
) -> dict[str, Any]:
    n = len(elements)
    elem_vec = Vec()
    for e in elements:
        elem_vec.push(e)

    def make_tracer(name: str) -> Tracer | None:
        if trace_verbosity is None:
            return None
        print(f"\n--- Trace: {name} ---")
        return Tracer(verbosity=trace_verbosity)

    # Encoder
    encoder = load_program("encoder", source_dir)
    enc_vec = Vec()
    for e in elements:
        enc_vec.push(e)
    enc, t_enc = run_program(
        encoder, {0: n, 1: enc_vec, 2: p}, make_tracer("encoder"),
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
        sample = make_greedy_sample(elements)

    # Verifier
    verifier = load_program("verifier", source_dir)
    ver_vec = Vec()
    for e in elements:
        ver_vec.push(e)
    ver, t_ver = run_program(
        verifier, {0: model, 1: sample, 2: n, 3: ver_vec},
        make_tracer("verifier"),
    )
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)
    difference = ver.state.get_output(2)

    # Decoder
    decoder = load_program("decoder", source_dir)
    dec_vec = Vec()
    for e in elements:
        dec_vec.push(e)
    dec, t_dec = run_program(
        decoder, {0: sample, 1: n, 2: dec_vec}, make_tracer("decoder"),
    )
    subset_a_vec = dec.state.get_output(0)
    subset_b_vec = dec.state.get_output(1)
    assert isinstance(subset_a_vec, Vec)
    assert isinstance(subset_b_vec, Vec)
    subset_a = [subset_a_vec.get(i) for i in range(subset_a_vec.length)]
    subset_b = [subset_b_vec.get(i) for i in range(subset_b_vec.length)]

    results: dict[str, Any] = {
        "n": n,
        "p": p,
        "encoder_time": t_enc,
        "solve_time": t_solve,
        "verifier_time": t_ver,
        "decoder_time": t_dec,
        "total_time": t_enc + t_ver + t_dec,
        "energy": energy,
        "valid": valid,
        "difference": difference,
        "subset_a": subset_a,
        "subset_b": subset_b,
        "model_size": model.size,
        "model_linear": len(model.linear),
        "model_quadratic": len(model.quadratic),
    }

    if verbose:
        total = sum(elements)
        target = total // 2
        print(f"\n{'=' * 50}")
        print(f"Number Partitioning N={n}, T={total}, target={target}, P={p}")
        print(f"{'=' * 50}")
        print(f"\n--- Elements ---")
        print(f"  {elements}")
        print(f"\n--- Model ---")
        print(render_info(model))
        if show_matrix:
            print(render_matrix(model))
        print(f"\n--- Results ---")
        print(f"Energy:       {energy}")
        print(f"Valid (VM):   {valid} ({'PASS' if valid else 'FAIL'})")
        print(f"Subset A:     {subset_a} (sum={sum(subset_a)})")
        print(f"Subset B:     {subset_b} (sum={sum(subset_b)})")
        print(f"Difference:   {difference}")
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
    solver: Backend | None = None, p: int = DEFAULT_P,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    all_results = []
    for n in sizes:
        elements = generate_random_set(n, rng)
        results = run_pipeline(
            elements, p=p, verbose=True,
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
        f"{'N':>5}  {'Vars':>5}  {'Linear':>8}  {'Quad':>8}  "
        f"{'Encoder':>10}  {'Verifier':>10}  {'Decoder':>10}  {'Total':>10}  "
        f"{'Diff':>6}  {'Valid':>5}"
    )
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['n']:>5}  {r['model_size']:>5}  {r['model_linear']:>8}  "
            f"{r['model_quadratic']:>8}  {r['encoder_time']:>10.6f}  "
            f"{r['verifier_time']:>10.6f}  {r['decoder_time']:>10.6f}  "
            f"{r['total_time']:>10.6f}  {r['difference']:>6}  "
            f"{'Y' if r['valid'] else 'N':>5}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XQVM Number Partitioning")
    parser.add_argument(
        "--bench", type=int, nargs="+", metavar="N",
        help="Benchmark for given sizes",
    )
    trace_group = parser.add_mutually_exclusive_group()
    trace_group.add_argument("--trace", action="store_const", const=2, dest="trace")
    trace_group.add_argument("--trace-compact", action="store_const", const=1, dest="trace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--src", type=str, default=DEFAULT_SOURCE,
        choices=["asm", "asm_penaltyeq"],
    )
    parser.add_argument("-n", type=int, default=6)
    parser.add_argument("-p", type=int, default=DEFAULT_P)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--no-solve", action="store_true")
    args = parser.parse_args()

    source_dir = NP_DIR / args.src

    sa_solver = None
    if not args.no_solve:
        from xqsa import NealBackend
        sa_solver = NealBackend(seed=args.seed)

    if args.bench:
        results = benchmark(
            args.bench, seed=args.seed, source_dir=source_dir,
            solver=sa_solver, p=args.p,
        )
        print_summary(results)
    else:
        rng = random.Random(args.seed)
        elements = generate_random_set(args.n, rng)
        run_pipeline(
            elements, p=args.p, verbose=True,
            trace_verbosity=args.trace, source_dir=source_dir,
            solver=sa_solver, show_matrix=args.matrix,
        )
