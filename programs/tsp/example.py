"""
TSP Example: Generic N-city Travelling Salesman Problem

Demonstrates the three-program architecture (Encoder, Verifier, Decoder)
using hand-built instruction lists with loop-based control flow.

Problem:
    N cities with a symmetric distance matrix.
    Find the shortest Hamiltonian cycle (tour visiting each city exactly once).

Grid layout (NxN BQMX):
    Rows = cities (0..N-1), Cols = positions (0..N-1)
    Variable x[i,p] = 1 means city i is at position p in the tour.

Distance input format:
    Flattened upper triangle of the distance matrix (no diagonal).
    For N cities, this is N*(N-1)/2 elements.
    Index for pair (i, j) where i < j: j*(j-1)/2 + i  (via IDXTRIU)

Register map (Encoder):
    r0  = N (city count)
    r1  = distance vec (flattened upper triangle, N*(N-1)/2 elements)
    r2  = N*N
    r4  = BQMX model
    r10 = ci (city i, loop var)
    r11 = cj (city j, loop var)
    r12 = p  (position, loop var)
    r13 = p_next ((p+1) % N)
    r15 = dist (current city-pair distance)
    r20 = col (column constraint loop var)
    r21 = i   (column constraint loop var)
    r22 = j   (column constraint loop var)
"""

import random

from xqvm.core.program import Instruction, run_program
from xqvm.core.opcodes import Opcode as Op

# =========================================================================
# Encoder
# =========================================================================
# Input slot 0: number of cities (int, N)
# Input slot 1: distances (vec<int>, N*(N-1)/2 upper triangle elements)
# Output slot 0: BQMX model (N*N variables, N rows x N cols grid)
#
# Steps:
#   1. Read N, compute N*N, allocate BQMX, set grid to NxN
#   2. Distance objective: for each city pair (ci < cj), for each
#      position p, add dist[ci,cj] as coupling between x[ci,p]-x[cj,p']
#      and x[cj,p]-x[ci,p'] where p' = (p+1) % N
#   3. Row one-hot constraints via ONEHOT (each city in exactly one position)
#   4. Column one-hot constraints via manual linear/quadratic terms
#      (each position has exactly one city)
#   5. Output model

PENALTY = 100

ENCODER = [
    # === Read inputs ===
    Instruction(Op.PUSH, (0,)),        # slot 0
    Instruction(Op.INPUT, (0,)),       # r0 = N
    Instruction(Op.PUSH, (1,)),        # slot 1
    Instruction(Op.INPUT, (1,)),       # r1 = distances vec (upper triangle)

    # === Compute N*N, allocate model, set grid ===
    Instruction(Op.LOAD, (0,)),        # push N
    Instruction(Op.DUPL),             # N, N
    Instruction(Op.MUL),              # N*N
    Instruction(Op.STOW, (2,)),        # r2 = N*N
    Instruction(Op.LOAD, (2,)),        # push N*N
    Instruction(Op.BQMX, (4,)),        # r4 = BQMX(N*N)
    Instruction(Op.LOAD, (0,)),        # push N (rows)
    Instruction(Op.LOAD, (0,)),        # push N (cols)
    Instruction(Op.RESIZE, (4,)),      # r4.grid = NxN

    # === Distance objective ===
    # for ci in [0, N-1):
    Instruction(Op.PUSH, (0,)),        # start = 0
    Instruction(Op.LOAD, (0,)),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.SUB),              # count = N - 1
    Instruction(Op.RANGE),
    Instruction(Op.LVAL, (10,)),       # r10 = ci

    #   for cj in [ci+1, N):
    Instruction(Op.LOAD, (10,)),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.ADD),              # start = ci + 1
    Instruction(Op.LOAD, (0,)),
    Instruction(Op.LOAD, (10,)),
    Instruction(Op.SUB),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.SUB),              # count = N - ci - 1
    Instruction(Op.RANGE),
    Instruction(Op.LVAL, (11,)),       # r11 = cj

    #     dist = distances[IDXTRIU(ci, cj)]
    Instruction(Op.LOAD, (10,)),       # push ci (i)
    Instruction(Op.LOAD, (11,)),       # push cj (j)
    Instruction(Op.IDXTRIU),          # upper triangle index
    Instruction(Op.VECGET, (1,)),      # push dist
    Instruction(Op.STOW, (15,)),       # r15 = dist

    #     for p in [0, N):
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.LOAD, (0,)),
    Instruction(Op.RANGE),
    Instruction(Op.LVAL, (12,)),       # r12 = p

    #       p_next = (p + 1) % N
    Instruction(Op.LOAD, (12,)),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.ADD),
    Instruction(Op.LOAD, (0,)),
    Instruction(Op.MOD),
    Instruction(Op.STOW, (13,)),       # r13 = p_next

    #       ADDQUAD r4 ci*N+p, cj*N+p_next, dist
    Instruction(Op.LOAD, (10,)),       # ci
    Instruction(Op.LOAD, (0,)),
    Instruction(Op.MUL),
    Instruction(Op.LOAD, (12,)),       # p
    Instruction(Op.ADD),              # idx_a = ci*N + p
    Instruction(Op.LOAD, (11,)),       # cj
    Instruction(Op.LOAD, (0,)),
    Instruction(Op.MUL),
    Instruction(Op.LOAD, (13,)),       # p_next
    Instruction(Op.ADD),              # idx_b = cj*N + p_next
    Instruction(Op.LOAD, (15,)),       # dist
    Instruction(Op.ADDQUAD, (4,)),

    #       ADDQUAD r4 cj*N+p, ci*N+p_next, dist
    Instruction(Op.LOAD, (11,)),       # cj
    Instruction(Op.LOAD, (0,)),
    Instruction(Op.MUL),
    Instruction(Op.LOAD, (12,)),       # p
    Instruction(Op.ADD),              # idx_c = cj*N + p
    Instruction(Op.LOAD, (10,)),       # ci
    Instruction(Op.LOAD, (0,)),
    Instruction(Op.MUL),
    Instruction(Op.LOAD, (13,)),       # p_next
    Instruction(Op.ADD),              # idx_d = ci*N + p_next
    Instruction(Op.LOAD, (15,)),       # dist
    Instruction(Op.ADDQUAD, (4,)),

    Instruction(Op.NEXT),             # end p loop
    Instruction(Op.NEXT),             # end cj loop
    Instruction(Op.NEXT),             # end ci loop

    # === Row one-hot constraints (each city in exactly one position) ===
    # for row in [0, N):
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.LOAD, (0,)),
    Instruction(Op.RANGE),
    Instruction(Op.LVAL, (10,)),       # r10 = row
    Instruction(Op.LOAD, (10,)),
    Instruction(Op.PUSH, (PENALTY,)),
    Instruction(Op.ONEHOT, (4,)),
    Instruction(Op.NEXT),

    # === Column one-hot constraints (each position has exactly one city) ===
    # ONEHOT only works on rows, so manually add column constraints.
    # One-hot(sum=1): linear[idx] += -P, quadratic[idx_i,idx_j] += 2P
    # for col in [0, N):
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.LOAD, (0,)),
    Instruction(Op.RANGE),
    Instruction(Op.LVAL, (20,)),       # r20 = col

    #   Linear terms: for i in [0, N)
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.LOAD, (0,)),
    Instruction(Op.RANGE),
    Instruction(Op.LVAL, (21,)),       # r21 = i
    #     index = i * N + col
    Instruction(Op.LOAD, (21,)),
    Instruction(Op.LOAD, (0,)),
    Instruction(Op.MUL),
    Instruction(Op.LOAD, (20,)),
    Instruction(Op.ADD),              # push index
    Instruction(Op.PUSH, (-PENALTY,)),
    Instruction(Op.ADDLINE, (4,)),
    Instruction(Op.NEXT),             # end i linear loop

    #   Quadratic terms: for i in [0, N-1)
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.LOAD, (0,)),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.SUB),              # count = N - 1
    Instruction(Op.RANGE),
    Instruction(Op.LVAL, (21,)),       # r21 = i

    #     for j in [i+1, N)
    Instruction(Op.LOAD, (21,)),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.ADD),              # start = i + 1
    Instruction(Op.LOAD, (0,)),
    Instruction(Op.LOAD, (21,)),
    Instruction(Op.SUB),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.SUB),              # count = N - i - 1
    Instruction(Op.RANGE),
    Instruction(Op.LVAL, (22,)),       # r22 = j

    #       idx_i = i * N + col
    Instruction(Op.LOAD, (21,)),
    Instruction(Op.LOAD, (0,)),
    Instruction(Op.MUL),
    Instruction(Op.LOAD, (20,)),
    Instruction(Op.ADD),
    #       idx_j = j * N + col
    Instruction(Op.LOAD, (22,)),
    Instruction(Op.LOAD, (0,)),
    Instruction(Op.MUL),
    Instruction(Op.LOAD, (20,)),
    Instruction(Op.ADD),
    Instruction(Op.PUSH, (2 * PENALTY,)),
    Instruction(Op.ADDQUAD, (4,)),

    Instruction(Op.NEXT),             # end j loop
    Instruction(Op.NEXT),             # end i loop
    Instruction(Op.NEXT),             # end col loop

    # === Output model ===
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.OUTPUT, (4,)),
    Instruction(Op.HALT),
]

# =========================================================================
# Verifier
# =========================================================================
# Input slot 0: BQMX model (from encoder output)
# Input slot 1: BSMX sample (solution to verify)
# Input slot 2: N (number of cities)
# Output slot 0: energy (int)
# Output slot 1: valid flag (1 = valid, 0 = invalid)
#
# Validation:
#   1. Each row sums to 1 (each city assigned exactly one position)
#   2. Each column sums to 1 (each position has exactly one city)
#   3. Compute energy
#
# Register map:
#   r0 = model, r1 = sample, r2 = valid flag, r3 = N
#   r4 = energy, r10 = loop var

VERIFIER = [
    # === Load inputs ===
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.INPUT, (0,)),       # r0 = model
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.INPUT, (1,)),       # r1 = sample
    Instruction(Op.PUSH, (2,)),
    Instruction(Op.INPUT, (3,)),       # r3 = N

    # === Initialize valid flag ===
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.STOW, (2,)),        # r2 = 1

    # === Check row sums = 1: for row in [0, N) ===
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.LOAD, (3,)),
    Instruction(Op.RANGE),
    Instruction(Op.LVAL, (10,)),       # r10 = row
    Instruction(Op.LOAD, (10,)),
    Instruction(Op.ROWSUM, (1,)),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.EQ),
    Instruction(Op.LOAD, (2,)),
    Instruction(Op.AND),
    Instruction(Op.STOW, (2,)),
    Instruction(Op.NEXT),

    # === Check col sums = 1: for col in [0, N) ===
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.LOAD, (3,)),
    Instruction(Op.RANGE),
    Instruction(Op.LVAL, (10,)),       # r10 = col
    Instruction(Op.LOAD, (10,)),
    Instruction(Op.COLSUM, (1,)),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.EQ),
    Instruction(Op.LOAD, (2,)),
    Instruction(Op.AND),
    Instruction(Op.STOW, (2,)),
    Instruction(Op.NEXT),

    # === Compute energy ===
    Instruction(Op.ENERGY, (0, 1)),
    Instruction(Op.STOW, (4,)),

    # === Output ===
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.OUTPUT, (4,)),       # output[0] = energy
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.OUTPUT, (2,)),       # output[1] = valid flag
    Instruction(Op.HALT),
]

# =========================================================================
# Decoder
# =========================================================================
# Input slot 0: BSMX sample
# Input slot 1: N (number of cities)
# Output slot 0: tour order as vec<int> (city at each position)
#
# For each position p in [0, N), find which city is assigned via COLFIND.
#
# Register map:
#   r0 = sample, r1 = N, r2 = tour vec, r10 = loop var

DECODER = [
    # === Load inputs ===
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.INPUT, (0,)),       # r0 = sample
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.INPUT, (1,)),       # r1 = N

    # === Create output vec ===
    Instruction(Op.VECI, (2,)),         # r2 = vec<int>

    # === For each position p in [0, N): find city ===
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.LOAD, (1,)),
    Instruction(Op.RANGE),
    Instruction(Op.LVAL, (10,)),       # r10 = p
    Instruction(Op.LOAD, (10,)),       # push col = p
    Instruction(Op.PUSH, (1,)),        # push value = 1
    Instruction(Op.COLFIND, (0,)),      # push city at position p
    Instruction(Op.VECPUSH, (2,)),      # append to tour
    Instruction(Op.NEXT),

    # === Output tour ===
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.OUTPUT, (2,)),
    Instruction(Op.HALT),
]

def triu_index(i: int, j: int) -> int:
    """ Upper triangle index for pair (i, j) where i < j. """
    if i > j:
        i, j = j, i
    return j * (j - 1) // 2 + i

# =========================================================================
# Run the pipeline
# =========================================================================

def run_tsp(n: int, distances: list[int]) -> None:
    """
    Run the full TSP encoder -> verifier -> decoder pipeline.

    Args:
        n: Number of cities
        distances: Upper-triangle distance vector (N*(N-1)/2 elements).
    """
    import time
    from xqvm.core.vector import Vec
    from xqvm.core.xqmx import XQMX

    # Wrap distance list as Vec for the VM
    dist_vec = Vec()
    for d in distances:
        dist_vec.push(d)

    # --- Encoder ---
    print("\n=== Encoder ===")
    encoder_input = {0: n, 1: dist_vec}

    t0 = time.perf_counter()
    enc = run_program(ENCODER, encoder_input)
    t_encoder = time.perf_counter() - t0

    model = enc.state.get_output(0)
    assert isinstance(model, XQMX)

    print(f"Model: size={model.size}, grid={model.rows}x{model.cols}")
    print(f"Linear terms: {len(model.linear)}, Quadratic terms: {len(model.quadratic)}")

    # --- Build known-good sample (simulate annealer) ---
    # Identity assignment: city i at position i
    sample = XQMX.binary_sample(size=n * n, rows=n, cols=n)
    for i in range(n):
        idx = i * n + i  # diagonal
        sample.set_linear(idx, 1.0)

    # --- Verifier ---
    print("\n=== Verifier ===")
    verifier_input = {0: model, 1: sample, 2: n}

    t0 = time.perf_counter()
    ver = run_program(VERIFIER, verifier_input)
    t_verifier = time.perf_counter() - t0

    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)

    print(f"Energy: {energy}")
    print(f"Valid: {valid} ({'PASS' if valid else 'FAIL'})")

    # --- Decoder ---
    print("\n=== Decoder ===")
    decoder_input = {0: sample, 1: n}

    t0 = time.perf_counter()
    dec = run_program(DECODER, decoder_input)
    t_decoder = time.perf_counter() - t0

    tour = dec.state.get_output(0)
    assert isinstance(tour, Vec)

    tour_list = [tour.get(i) for i in range(tour.length)]
    city_names = [f"C{i}" for i in range(n)]
    route = " -> ".join(city_names[c] for c in tour_list)

    print(f"Tour: {tour_list}")
    print(f"Route: {route} -> {city_names[tour_list[0]]}")

    # --- Validate ---
    assert valid == 1, "Expected valid solution"
    assert tour_list == list(range(n)), f"Expected identity tour, got {tour_list}"

    # Compute tour distance using upper triangle
    tour_dist = 0
    for p in range(n):
        ci, cj = tour_list[p], tour_list[(p + 1) % n]
        tour_dist += distances[triu_index(ci, cj)]

    print(f"\nTour distance: {tour_dist}")

    # --- Benchmarks ---
    t_total = t_encoder + t_verifier + t_decoder
    print(f"\n=== Benchmarks ===")
    print(f"Encoder:  {t_encoder:.4f}s")
    print(f"Verifier: {t_verifier:.4f}s")
    print(f"Decoder:  {t_decoder:.4f}s")
    print(f"Total:    {t_total:.4f}s")

def run_tsp_random(n_cities: int):
    # Generate random distance matrix
    random.seed(42)
    dist_triu = [random.randint(1, 100) for _ in range(n_cities * (n_cities - 1) // 2)]
    
    # Print distance matrix
    print(f"=== TSP with {n_cities} cities ===")
    print("Distance matrix (upper triangle):")
    for i in range(n_cities):
        row = []
        for j in range(n_cities):
            if i == j:
                row.append("  .")
            elif i < j:
                row.append(f"{dist_triu[triu_index(i, j)]:3d}")
            else:
                row.append(f"{dist_triu[triu_index(j, i)]:3d}")
        print(f"  C{i}: {' '.join(row)}")

    run_tsp(n_cities, dist_triu)

if __name__ == "__main__":
    run_tsp_random(200)