"""
Maximum Independent Set Runner.

Find the largest subset of vertices with no two adjacent vertices selected.
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

from tools.visualizer import render_info

if TYPE_CHECKING:
    from xqsa import Backend

MIS_DIR = Path(__file__).parent
DEFAULT_SOURCE = "asm"
DEFAULT_A = 1
DEFAULT_P = 10


def load_program(name: str, source_dir: Path | None = None) -> AssembledProgram:
    program_dir = source_dir or (MIS_DIR / DEFAULT_SOURCE)
    path = program_dir / f"{name}.xqasm"
    return assemble(path.read_text(), name)


def generate_random_graph(
    n: int, rng: random.Random, density: float = 0.4,
) -> list[tuple[int, int]]:
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                edges.append((i, j))
    return edges


def make_edge_vec(edges: list[tuple[int, int]]) -> Vec:
    v = Vec()
    for i, j in edges:
        v.push(i)
        v.push(j)
    return v


def make_greedy_sample(
    n: int, edges: list[tuple[int, int]], model_size: int,
) -> XQMX:
    """Greedy MIS: add vertices by degree (ascending), skip if neighbor selected."""
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    ranked = sorted(range(n), key=lambda v: len(adj[v]))
    selected = set()
    for v in ranked:
        if not any(u in selected for u in adj[v]):
            selected.add(v)

    sample = XQMX.binary_sample(size=model_size)
    for v in selected:
        sample.set_linear(v, 1)

    # Set slack variables: for edge e, s_e = 1 - x_i - x_j
    for e, (i, j) in enumerate(edges):
        xi = 1 if i in selected else 0
        xj = 1 if j in selected else 0
        s = 1 - xi - xj
        if s == 1:
            sample.set_linear(n + e, 1)

    return sample


def verify_independent(
    n: int, edges: list[tuple[int, int]], selected: list[int],
) -> bool:
    sel = set(selected)
    for i, j in edges:
        if i in sel and j in sel:
            return False
    return True


def run_program(
    program: AssembledProgram,
    input_data: dict[int, Any],
) -> tuple[Executor, float]:
    ex = Executor()
    t0 = time.perf_counter()
    ex.execute(program.program, input_data)
    elapsed = time.perf_counter() - t0
    return ex, elapsed


def run_pipeline(
    n: int,
    edges: list[tuple[int, int]],
    a: int = DEFAULT_A,
    p: int = DEFAULT_P,
    verbose: bool = False,
    source_dir: Path | None = None,
    solver: Backend | None = None,
) -> dict[str, Any]:
    e_count = len(edges)
    edge_vec = make_edge_vec(edges)

    encoder = load_program("encoder", source_dir)
    enc, t_enc = run_program(
        encoder, {0: n, 1: e_count, 2: edge_vec, 3: a, 4: p},
    )
    model = enc.state.get_output(0)
    assert isinstance(model, XQMX)

    t_solve = 0.0
    if solver is not None:
        solve_result = solver.solve(model)
        sample = solve_result.sample
        t_solve = solve_result.timing
    else:
        sample = make_greedy_sample(n, edges, model.size)

    # Verifier
    verifier = load_program("verifier", source_dir)
    ev2 = make_edge_vec(edges)
    ver, t_ver = run_program(
        verifier, {0: model, 1: sample, 2: n, 3: e_count, 4: ev2},
    )
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)
    set_size = ver.state.get_output(2)

    # Decoder
    decoder = load_program("decoder", source_dir)
    dec, t_dec = run_program(
        decoder, {0: sample, 1: n},
    )
    sel_vec = dec.state.get_output(0)
    assert isinstance(sel_vec, Vec)
    selected = [sel_vec.get(i) for i in range(sel_vec.length)]

    ind_valid = verify_independent(n, edges, selected)

    results = {
        "n": n,
        "edges": e_count,
        "encoder_time": t_enc,
        "solve_time": t_solve,
        "verifier_time": t_ver,
        "decoder_time": t_dec,
        "total_time": t_enc + t_ver + t_dec,
        "energy": energy,
        "valid": valid,
        "set_size": set_size,
        "selected": selected,
        "independent_valid": ind_valid,
        "model_size": model.size,
        "model_linear": len(model.linear),
        "model_quadratic": len(model.quadratic),
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Max Independent Set N={n}, E={e_count}, A={a}, P={p}")
        print(f"{'=' * 50}")
        print(render_info(model))
        print(f"Energy:       {energy}")
        print(f"Valid (VM):   {valid}")
        print(f"Independent:  {ind_valid}")
        print(f"Selected:     {selected} (size={set_size})")
        print(f"Encoder:      {t_enc:.6f}s")
        print(f"Verifier:     {t_ver:.6f}s")
        print(f"Decoder:      {t_dec:.6f}s")
        print(f"Total:        {t_enc + t_ver + t_dec:.6f}s")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XQVM Maximum Independent Set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--src", type=str, default=DEFAULT_SOURCE, choices=["asm", "asm_penaltyeq"])
    parser.add_argument("-n", type=int, default=6)
    parser.add_argument("--no-solve", action="store_true")
    args = parser.parse_args()

    source_dir = MIS_DIR / args.src
    sa_solver = None
    if not args.no_solve:
        from xqsa import NealBackend
        sa_solver = NealBackend(seed=args.seed)

    rng = random.Random(args.seed)
    edges = generate_random_graph(args.n, rng)
    run_pipeline(args.n, edges, verbose=True, source_dir=source_dir, solver=sa_solver)
