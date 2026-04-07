"""
Graph Coloring Runner.

Assign K colors to N vertices such that no adjacent vertices share a color.
Includes symmetry breaking: vertex 0 is fixed to color 0.
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

GC_DIR = Path(__file__).parent
DEFAULT_SOURCE = "asm"
DEFAULT_P_OH = 100
DEFAULT_P_EX = 100
DEFAULT_P_SYM = 50


def load_program(name: str, source_dir: Path | None = None) -> AssembledProgram:
    program_dir = source_dir or (GC_DIR / DEFAULT_SOURCE)
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


def greedy_coloring(
    n: int, k: int, edges: list[tuple[int, int]],
) -> list[int]:
    """Greedy coloring with vertex 0 fixed to color 0, ordered by degree."""
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    coloring = [-1] * n
    coloring[0] = 0  # symmetry breaking

    # Order remaining vertices by degree (highest first)
    order = sorted(range(1, n), key=lambda v: len(adj[v]), reverse=True)

    for v in order:
        neighbor_colors = {coloring[u] for u in adj[v] if coloring[u] >= 0}
        for c in range(k):
            if c not in neighbor_colors:
                coloring[v] = c
                break
        if coloring[v] == -1:
            coloring[v] = 0  # fallback (may be invalid)

    return coloring


def make_sample(
    n: int, k: int, coloring: list[int],
) -> XQMX:
    sample = XQMX.binary_sample(size=n * k)
    for i, c in enumerate(coloring):
        if 0 <= c < k:
            sample.set_linear(i * k + c, 1)
    return sample


def verify_coloring(
    n: int, edges: list[tuple[int, int]], coloring: list[int],
) -> tuple[bool, int]:
    conflicts = 0
    for i, j in edges:
        if coloring[i] == coloring[j]:
            conflicts += 1
    return conflicts == 0, conflicts


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
    k: int,
    edges: list[tuple[int, int]],
    p_oh: int = DEFAULT_P_OH,
    p_ex: int = DEFAULT_P_EX,
    p_sym: int = DEFAULT_P_SYM,
    verbose: bool = False,
    source_dir: Path | None = None,
    solver: Backend | None = None,
) -> dict[str, Any]:
    e_count = len(edges)
    edge_vec = make_edge_vec(edges)

    encoder = load_program("encoder", source_dir)
    enc, t_enc = run_program(
        encoder, {0: n, 1: k, 2: e_count, 3: edge_vec, 4: p_oh, 5: p_ex, 6: p_sym},
    )
    model = enc.state.get_output(0)
    assert isinstance(model, XQMX)

    t_solve = 0.0
    if solver is not None:
        solve_result = solver.solve(model)
        sample = solve_result.sample
        t_solve = solve_result.timing
    else:
        coloring = greedy_coloring(n, k, edges)
        sample = make_sample(n, k, coloring)

    # Verifier
    verifier = load_program("verifier", source_dir)
    ev2 = make_edge_vec(edges)
    ver, t_ver = run_program(
        verifier, {0: model, 1: sample, 2: n, 3: k, 4: e_count, 5: ev2},
    )
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)
    conflicts = ver.state.get_output(2)

    # Decoder
    decoder = load_program("decoder", source_dir)
    dec, t_dec = run_program(
        decoder, {0: sample, 1: n, 2: k},
    )
    col_vec = dec.state.get_output(0)
    assert isinstance(col_vec, Vec)
    coloring = [col_vec.get(i) for i in range(col_vec.length)]

    ind_valid, ind_conflicts = verify_coloring(n, edges, coloring)

    colors_used = len(set(coloring))

    results = {
        "n": n,
        "k": k,
        "edges": e_count,
        "encoder_time": t_enc,
        "solve_time": t_solve,
        "verifier_time": t_ver,
        "decoder_time": t_dec,
        "total_time": t_enc + t_ver + t_dec,
        "energy": energy,
        "valid": valid,
        "conflicts": conflicts,
        "coloring": coloring,
        "colors_used": colors_used,
        "independent_valid": ind_valid,
        "model_size": model.size,
        "model_linear": len(model.linear),
        "model_quadratic": len(model.quadratic),
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Graph Coloring N={n}, K={k}, E={e_count}")
        print(f"{'=' * 50}")
        print(render_info(model))
        print(f"Energy:       {energy}")
        print(f"Valid (VM):   {valid}")
        print(f"Conflicts:    {conflicts}")
        print(f"Coloring:     {coloring}")
        print(f"Colors used:  {colors_used}")
        print(f"Ind. valid:   {ind_valid}")
        print(f"Vertex 0:     color {coloring[0]} (should be 0 = symmetry break)")
        print(f"Encoder:      {t_enc:.6f}s")
        print(f"Verifier:     {t_ver:.6f}s")
        print(f"Decoder:      {t_dec:.6f}s")
        print(f"Total:        {t_enc + t_ver + t_dec:.6f}s")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XQVM Graph Coloring")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-n", type=int, default=6)
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--no-solve", action="store_true")
    args = parser.parse_args()

    source_dir = GC_DIR / "asm"
    sa_solver = None
    if not args.no_solve:
        from xqsa import NealBackend
        sa_solver = NealBackend(seed=args.seed)

    rng = random.Random(args.seed)
    edges = generate_random_graph(args.n, rng)
    run_pipeline(args.n, args.k, edges, verbose=True, source_dir=source_dir, solver=sa_solver)
