"""
Max-3-SAT Runner: Assemble and execute the Max-3-SAT encoder-verifier-decoder pipeline.

Benchmarks XQVM performance across multiple problem sizes using
random 3-SAT instances with degree reduction for cubic-to-quadratic conversion.
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

SAT_DIR = Path(__file__).parent
DEFAULT_SOURCE = "asm"
DEFAULT_P = 1         # clause penalty
DEFAULT_P_AUX = 10    # auxiliary constraint penalty


def load_program(name: str, source_dir: Path | None = None) -> AssembledProgram:
    """Load and assemble a .xqasm file."""
    program_dir = source_dir or (SAT_DIR / DEFAULT_SOURCE)
    path = program_dir / f"{name}.xqasm"
    source = path.read_text()
    return assemble(source, name)


def make_clause_vec(clauses: list[list[tuple[int, bool]]]) -> Vec:
    """Wrap clauses as a flat Vec: [var, neg, var, neg, var, neg, ...]."""
    v = Vec()
    for clause in clauses:
        for var, neg in clause:
            v.push(var)
            v.push(1 if neg else 0)
    return v


def generate_random_3sat(
    n: int, m: int, rng: random.Random,
) -> list[list[tuple[int, bool]]]:
    """
    Generate a random 3-SAT instance.

    Args:
        n: Number of variables
        m: Number of clauses
        rng: Random number generator

    Returns:
        List of clauses, each clause is [(var, negated), (var, negated), (var, negated)]
    """
    clauses = []
    for _ in range(m):
        # Pick 3 distinct variables
        vars_chosen = rng.sample(range(n), min(3, n))
        while len(vars_chosen) < 3:
            vars_chosen.append(rng.randint(0, n - 1))
        clause = [(v, rng.random() < 0.5) for v in vars_chosen]
        clauses.append(clause)
    return clauses


def evaluate_clauses(
    assignment: list[int], clauses: list[list[tuple[int, bool]]],
) -> tuple[int, list[bool]]:
    """Evaluate which clauses are satisfied by the assignment."""
    results = []
    for clause in clauses:
        satisfied = False
        for var, neg in clause:
            lit_val = (1 - assignment[var]) if neg else assignment[var]
            if lit_val == 1:
                satisfied = True
                break
        results.append(satisfied)
    return sum(results), results


def make_greedy_sample(
    n: int, m: int, clauses: list[list[tuple[int, bool]]],
) -> XQMX:
    """
    Build a greedy sample that tries to satisfy as many clauses as possible.

    Simple heuristic: for each variable, try both 0 and 1,
    pick the value that satisfies more clauses.
    """
    total_vars = n + m
    assignment = [0] * n

    for i in range(n):
        # Try 0
        assignment[i] = 0
        sat_0, _ = evaluate_clauses(assignment, clauses)
        # Try 1
        assignment[i] = 1
        sat_1, _ = evaluate_clauses(assignment, clauses)
        # Pick better
        assignment[i] = 1 if sat_1 >= sat_0 else 0

    # Set sample
    sample = XQMX.binary_sample(size=total_vars)
    for i in range(n):
        if assignment[i]:
            sample.set_linear(i, 1)

    # Set auxiliary variables: w_c = x_a * x_b for each clause
    for c_idx, clause in enumerate(clauses):
        a_var, a_neg = clause[0]
        b_var, b_neg = clause[1]

        # Compute complement values (what enters the cubic penalty)
        a_val = (1 - assignment[a_var]) if not a_neg else assignment[a_var]
        b_val = (1 - assignment[b_var]) if not b_neg else assignment[b_var]

        w_val = a_val * b_val
        if w_val:
            sample.set_linear(n + c_idx, 1)

    return sample


def run_program(
    program: AssembledProgram,
    input_data: dict[int, Any],
    tracer: Tracer | None = None,
) -> tuple[Executor, float]:
    """Execute an assembled program."""
    ex = Executor(tracer=tracer)
    t0 = time.perf_counter()
    ex.execute(program.program, input_data)
    elapsed = time.perf_counter() - t0
    return ex, elapsed


def run_pipeline(
    n: int,
    clauses: list[list[tuple[int, bool]]],
    p: int = DEFAULT_P,
    p_aux: int = DEFAULT_P_AUX,
    verbose: bool = False,
    trace_verbosity: int | None = None,
    source_dir: Path | None = None,
    solver: Backend | None = None,
    show_matrix: bool = False,
) -> dict[str, Any]:
    """Run the full Max-3-SAT encoder-verifier-decoder pipeline."""
    m = len(clauses)
    clause_vec = make_clause_vec(clauses)

    def make_tracer(name: str) -> Tracer | None:
        if trace_verbosity is None:
            return None
        print(f"\n--- Trace: {name} ---")
        return Tracer(verbosity=trace_verbosity)

    # Encoder
    encoder = load_program("encoder", source_dir)
    enc, t_enc = run_program(
        encoder,
        {0: n, 1: m, 2: clause_vec, 3: p, 4: p_aux},
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
        sample = make_greedy_sample(n, m, clauses)

    # Verifier
    verifier = load_program("verifier", source_dir)
    ver, t_ver = run_program(
        verifier,
        {0: model, 1: sample, 2: n, 3: m, 4: clause_vec},
        make_tracer("verifier"),
    )
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)
    satisfied = ver.state.get_output(2)

    # Decoder
    decoder = load_program("decoder", source_dir)
    dec, t_dec = run_program(
        decoder,
        {0: sample, 1: n},
        make_tracer("decoder"),
    )
    assignment_vec = dec.state.get_output(0)
    assert isinstance(assignment_vec, Vec)
    assignment = [assignment_vec.get(i) for i in range(assignment_vec.length)]

    # Independent verification
    ind_sat, clause_results = evaluate_clauses(assignment, clauses)

    results: dict[str, Any] = {
        "n": n,
        "m": m,
        "p": p,
        "p_aux": p_aux,
        "encoder_time": t_enc,
        "solve_time": t_solve,
        "verifier_time": t_ver,
        "decoder_time": t_dec,
        "total_time": t_enc + t_ver + t_dec,
        "energy": energy,
        "valid": valid,
        "satisfied": satisfied,
        "independent_satisfied": ind_sat,
        "assignment": assignment,
        "model_size": model.size,
        "model_linear": len(model.linear),
        "model_quadratic": len(model.quadratic),
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Max-3-SAT N={n}, M={m}, P={p}, P_aux={p_aux}")
        print(f"{'=' * 50}")
        print(f"\n--- Clauses ---")
        for c_idx, clause in enumerate(clauses):
            lits = []
            for var, neg in clause:
                lits.append(f"{'NOT ' if neg else ''}x{var}")
            sat_mark = "✓" if clause_results[c_idx] else "✗"
            print(f"  C{c_idx}: ({' OR '.join(lits)}) {sat_mark}")
        print(f"\n--- Model ---")
        print(render_info(model))
        if show_matrix:
            print(render_matrix(model))
        print(f"\n--- Results ---")
        print(f"Energy:           {energy}")
        print(f"Valid (VM):       {valid} ({'PASS' if valid else 'FAIL'})")
        print(f"Assignment:       {assignment}")
        print(f"Satisfied (VM):   {satisfied}/{m}")
        print(f"Satisfied (check):{ind_sat}/{m}")
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
    p: int = DEFAULT_P,
    p_aux: int = DEFAULT_P_AUX,
) -> list[dict[str, Any]]:
    """Run Max-3-SAT pipeline across multiple problem sizes."""
    rng = random.Random(seed)
    all_results: list[dict[str, Any]] = []

    for n, m in sizes:
        clauses = generate_random_3sat(n, m, rng)
        results = run_pipeline(
            n, clauses, p=p, p_aux=p_aux, verbose=True,
            source_dir=source_dir, solver=solver,
        )
        all_results.append(results)

    return all_results


def print_summary(all_results: list[dict[str, Any]]) -> None:
    """Print summary table."""
    if not all_results:
        return

    print(f"\n{'=' * 100}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 100}")
    print(
        f"{'N':>5}  {'M':>5}  {'Vars':>6}  {'Linear':>8}  {'Quad':>8}  "
        f"{'Encoder':>10}  {'Verifier':>10}  {'Decoder':>10}  {'Total':>10}  "
        f"{'Sat':>5}  {'Valid':>5}"
    )
    print("-" * 100)

    for r in all_results:
        print(
            f"{r['n']:>5}  {r['m']:>5}  {r['model_size']:>6}  "
            f"{r['model_linear']:>8}  {r['model_quadratic']:>8}  "
            f"{r['encoder_time']:>10.6f}  {r['verifier_time']:>10.6f}  "
            f"{r['decoder_time']:>10.6f}  {r['total_time']:>10.6f}  "
            f"{r['satisfied']}/{r['m']:>3}  "
            f"{'Y' if r['valid'] else 'N':>5}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XQVM Max-3-SAT Example")
    parser.add_argument(
        "--bench", type=int, nargs="+", metavar="N",
        help="Run benchmarks: pairs of (N, M) e.g. --bench 4 8 6 12 means N=4,M=8 then N=6,M=12",
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
        choices=["asm", "asm_reduce"],
        help="Program source directory (default: asm)",
    )
    parser.add_argument("-n", type=int, default=4, help="Variable count (default: 4)")
    parser.add_argument("-m", type=int, default=6, help="Clause count (default: 6)")
    parser.add_argument("-p", type=int, default=DEFAULT_P, help=f"Clause penalty (default: {DEFAULT_P})")
    parser.add_argument("--p-aux", type=int, default=DEFAULT_P_AUX, help=f"Aux penalty (default: {DEFAULT_P_AUX})")
    parser.add_argument("--matrix", action="store_true", help="Print XQMX matrix")
    parser.add_argument("--no-solve", action="store_true", help="Use greedy sample instead of solver")
    args = parser.parse_args()

    if args.bench and args.trace is not None:
        parser.error("--trace cannot be used with --bench")

    source_dir = SAT_DIR / args.src

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
            solver=sa_solver, p=args.p, p_aux=args.p_aux,
        )
        print_summary(results)
    else:
        rng = random.Random(args.seed)
        clauses = generate_random_3sat(args.n, args.m, rng)
        run_pipeline(
            args.n, clauses, p=args.p, p_aux=args.p_aux, verbose=True,
            trace_verbosity=args.trace, source_dir=source_dir,
            solver=sa_solver, show_matrix=args.matrix,
        )
