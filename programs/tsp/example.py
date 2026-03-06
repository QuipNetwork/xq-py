"""
TSP Example: 3-city Travelling Salesman Problem

Demonstrates the three-program architecture (Encoder, Verifier, Decoder)
using hand-built instruction lists. This is a sample program for validation
purposes — it will be replaced with proper .xqasm files later.

Problem:
    3 cities with distance matrix:
        C0-C1: 10, C0-C2: 15, C1-C2: 35
    Optimal tour: C0 -> C1 -> C2 -> C0 (distance = 60)

Grid layout (3x3 BQMX):
    Rows = cities (0,1,2), Cols = positions (0,1,2)
    Variable x[i,j] = 1 means city i is at position j
"""

from xqvm.core.program import Instruction, run_program
from xqvm.core.opcodes import Opcode as Op


# =========================================================================
# Encoder
# =========================================================================
# Input slot 0: number of cities (int, N=3)
# Input slot 1: flat distance matrix (vec<int>, N*N=9 elements)
# Output slot 0: BQMX model (N*N variables, N rows x N cols grid)
#
# Steps:
#   1. Read N from input, compute N*N
#   2. Allocate BQMX with size=N*N, set grid to NxN
#   3. Add distance objective: for each pair of positions (p, p+1),
#      add dist[ci][cj] as quadratic coupling between x[ci,p] and x[cj,p+1]
#   4. Add one-hot row constraints (each city assigned exactly one position)
#   5. Add one-hot column constraints (each position has exactly one city)
#      — done manually since ONEHOT only works on rows
#   6. Output the model

PENALTY = 100

ENCODER = [
    # --- Read N from input slot 0, store in r0 ---
    Instruction(Op.PUSH, (0,)),        # slot 0
    Instruction(Op.INPUT, (0,)),       # r0 = input[0] = N
    # --- Read distance vec from input slot 1 into r1 ---
    Instruction(Op.PUSH, (1,)),        # slot 1
    Instruction(Op.INPUT, (1,)),       # r1 = input[1] = distances vec

    # --- Compute N*N, store in r2 ---
    Instruction(Op.LOAD, (0,)),        # push N
    Instruction(Op.DUPL),             # push N, N
    Instruction(Op.MUL),              # push N*N
    Instruction(Op.STOW, (2,)),        # r2 = N*N

    # --- Allocate BQMX of size N*N into r4 ---
    Instruction(Op.LOAD, (2,)),        # push N*N
    Instruction(Op.BQMX, (4,)),        # r4 = BQMX(size=N*N)

    # --- Set grid dimensions: RESIZE pops cols, rows ---
    Instruction(Op.LOAD, (0,)),        # push N (rows)
    Instruction(Op.LOAD, (0,)),        # push N (cols)
    Instruction(Op.RESIZE, (4,)),      # r4.resize(rows=N, cols=N)

    # --- Distance objective ---
    # For each pair of cities (ci, cj) where ci < cj:
    #   For each position p in [0, N-1):
    #     coupling_a = grid_index(ci, p) and grid_index(cj, p+1)
    #     coupling_b = grid_index(cj, p) and grid_index(ci, p+1)
    #     dist = distances[ci*N + cj]
    #     ADDQUAD r4 with (coupling_a, coupling_b, dist)
    #
    # For 3 cities, ci<cj pairs: (0,1), (0,2), (1,2)
    # For 3 positions, p: 0, 1 (and wrap p=2->0)

    # -- Pair (ci=0, cj=1), dist=10 --
    # p=0: x[0,0]-x[1,1] and x[1,0]-x[0,1]
    # idx(0,0)=0, idx(1,1)=4, idx(1,0)=3, idx(0,1)=1
    Instruction(Op.PUSH, (0,)),        # i = idx(0,0) = 0
    Instruction(Op.PUSH, (4,)),        # j = idx(1,1) = 4
    Instruction(Op.PUSH, (10,)),       # delta = dist(0,1) = 10
    Instruction(Op.ADDQUAD, (4,)),     # quad[0,4] += 10
    Instruction(Op.PUSH, (3,)),        # i = idx(1,0) = 3
    Instruction(Op.PUSH, (1,)),        # j = idx(0,1) = 1
    Instruction(Op.PUSH, (10,)),       # delta = 10
    Instruction(Op.ADDQUAD, (4,)),     # quad[3,1] += 10 -> stored as quad[1,3]

    # p=1: x[0,1]-x[1,2] and x[1,1]-x[0,2]
    # idx(0,1)=1, idx(1,2)=5, idx(1,1)=4, idx(0,2)=2
    Instruction(Op.PUSH, (1,)),        # i
    Instruction(Op.PUSH, (5,)),        # j
    Instruction(Op.PUSH, (10,)),       # delta
    Instruction(Op.ADDQUAD, (4,)),
    Instruction(Op.PUSH, (4,)),        # i
    Instruction(Op.PUSH, (2,)),        # j
    Instruction(Op.PUSH, (10,)),       # delta
    Instruction(Op.ADDQUAD, (4,)),

    # p=2 (wrap to 0): x[0,2]-x[1,0] and x[1,2]-x[0,0]
    # idx(0,2)=2, idx(1,0)=3, idx(1,2)=5, idx(0,0)=0
    Instruction(Op.PUSH, (2,)),
    Instruction(Op.PUSH, (3,)),
    Instruction(Op.PUSH, (10,)),
    Instruction(Op.ADDQUAD, (4,)),
    Instruction(Op.PUSH, (5,)),
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.PUSH, (10,)),
    Instruction(Op.ADDQUAD, (4,)),

    # -- Pair (ci=0, cj=2), dist=15 --
    # p=0: idx(0,0)=0, idx(2,1)=7 / idx(2,0)=6, idx(0,1)=1
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.PUSH, (7,)),
    Instruction(Op.PUSH, (15,)),
    Instruction(Op.ADDQUAD, (4,)),
    Instruction(Op.PUSH, (6,)),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.PUSH, (15,)),
    Instruction(Op.ADDQUAD, (4,)),
    # p=1: idx(0,1)=1, idx(2,2)=8 / idx(2,1)=7, idx(0,2)=2
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.PUSH, (8,)),
    Instruction(Op.PUSH, (15,)),
    Instruction(Op.ADDQUAD, (4,)),
    Instruction(Op.PUSH, (7,)),
    Instruction(Op.PUSH, (2,)),
    Instruction(Op.PUSH, (15,)),
    Instruction(Op.ADDQUAD, (4,)),
    # p=2 (wrap): idx(0,2)=2, idx(2,0)=6 / idx(2,2)=8, idx(0,0)=0
    Instruction(Op.PUSH, (2,)),
    Instruction(Op.PUSH, (6,)),
    Instruction(Op.PUSH, (15,)),
    Instruction(Op.ADDQUAD, (4,)),
    Instruction(Op.PUSH, (8,)),
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.PUSH, (15,)),
    Instruction(Op.ADDQUAD, (4,)),

    # -- Pair (ci=1, cj=2), dist=35 --
    # p=0: idx(1,0)=3, idx(2,1)=7 / idx(2,0)=6, idx(1,1)=4
    Instruction(Op.PUSH, (3,)),
    Instruction(Op.PUSH, (7,)),
    Instruction(Op.PUSH, (35,)),
    Instruction(Op.ADDQUAD, (4,)),
    Instruction(Op.PUSH, (6,)),
    Instruction(Op.PUSH, (4,)),
    Instruction(Op.PUSH, (35,)),
    Instruction(Op.ADDQUAD, (4,)),
    # p=1: idx(1,1)=4, idx(2,2)=8 / idx(2,1)=7, idx(1,2)=5
    Instruction(Op.PUSH, (4,)),
    Instruction(Op.PUSH, (8,)),
    Instruction(Op.PUSH, (35,)),
    Instruction(Op.ADDQUAD, (4,)),
    Instruction(Op.PUSH, (7,)),
    Instruction(Op.PUSH, (5,)),
    Instruction(Op.PUSH, (35,)),
    Instruction(Op.ADDQUAD, (4,)),
    # p=2 (wrap): idx(1,2)=5, idx(2,0)=6 / idx(2,2)=8, idx(1,0)=3
    Instruction(Op.PUSH, (5,)),
    Instruction(Op.PUSH, (6,)),
    Instruction(Op.PUSH, (35,)),
    Instruction(Op.ADDQUAD, (4,)),
    Instruction(Op.PUSH, (8,)),
    Instruction(Op.PUSH, (3,)),
    Instruction(Op.PUSH, (35,)),
    Instruction(Op.ADDQUAD, (4,)),

    # --- Row one-hot constraints (each city in exactly one position) ---
    # ONEHOT pops penalty, row from stack
    Instruction(Op.PUSH, (0,)),            # row 0
    Instruction(Op.PUSH, (PENALTY,)),      # penalty
    Instruction(Op.ONEHOT, (4,)),
    Instruction(Op.PUSH, (1,)),            # row 1
    Instruction(Op.PUSH, (PENALTY,)),
    Instruction(Op.ONEHOT, (4,)),
    Instruction(Op.PUSH, (2,)),            # row 2
    Instruction(Op.PUSH, (PENALTY,)),
    Instruction(Op.ONEHOT, (4,)),

    # --- Column one-hot constraints (each position has exactly one city) ---
    # ONEHOT only works on rows, so we manually add column constraints.
    # For column c, variables are at indices c, c+N, c+2N (i.e., c, c+3, c+6)
    # One-hot: sum = 1 means linear[i] -= penalty, quad[i,j] += 2*penalty

    # Column 0: indices 0, 3, 6
    Instruction(Op.PUSH, (0,)),  Instruction(Op.PUSH, (-PENALTY,)),  Instruction(Op.ADDLINE, (4,)),
    Instruction(Op.PUSH, (3,)),  Instruction(Op.PUSH, (-PENALTY,)),  Instruction(Op.ADDLINE, (4,)),
    Instruction(Op.PUSH, (6,)),  Instruction(Op.PUSH, (-PENALTY,)),  Instruction(Op.ADDLINE, (4,)),
    Instruction(Op.PUSH, (0,)),  Instruction(Op.PUSH, (3,)),  Instruction(Op.PUSH, (2 * PENALTY,)),  Instruction(Op.ADDQUAD, (4,)),
    Instruction(Op.PUSH, (0,)),  Instruction(Op.PUSH, (6,)),  Instruction(Op.PUSH, (2 * PENALTY,)),  Instruction(Op.ADDQUAD, (4,)),
    Instruction(Op.PUSH, (3,)),  Instruction(Op.PUSH, (6,)),  Instruction(Op.PUSH, (2 * PENALTY,)),  Instruction(Op.ADDQUAD, (4,)),

    # Column 1: indices 1, 4, 7
    Instruction(Op.PUSH, (1,)),  Instruction(Op.PUSH, (-PENALTY,)),  Instruction(Op.ADDLINE, (4,)),
    Instruction(Op.PUSH, (4,)),  Instruction(Op.PUSH, (-PENALTY,)),  Instruction(Op.ADDLINE, (4,)),
    Instruction(Op.PUSH, (7,)),  Instruction(Op.PUSH, (-PENALTY,)),  Instruction(Op.ADDLINE, (4,)),
    Instruction(Op.PUSH, (1,)),  Instruction(Op.PUSH, (4,)),  Instruction(Op.PUSH, (2 * PENALTY,)),  Instruction(Op.ADDQUAD, (4,)),
    Instruction(Op.PUSH, (1,)),  Instruction(Op.PUSH, (7,)),  Instruction(Op.PUSH, (2 * PENALTY,)),  Instruction(Op.ADDQUAD, (4,)),
    Instruction(Op.PUSH, (4,)),  Instruction(Op.PUSH, (7,)),  Instruction(Op.PUSH, (2 * PENALTY,)),  Instruction(Op.ADDQUAD, (4,)),

    # Column 2: indices 2, 5, 8
    Instruction(Op.PUSH, (2,)),  Instruction(Op.PUSH, (-PENALTY,)),  Instruction(Op.ADDLINE, (4,)),
    Instruction(Op.PUSH, (5,)),  Instruction(Op.PUSH, (-PENALTY,)),  Instruction(Op.ADDLINE, (4,)),
    Instruction(Op.PUSH, (8,)),  Instruction(Op.PUSH, (-PENALTY,)),  Instruction(Op.ADDLINE, (4,)),
    Instruction(Op.PUSH, (2,)),  Instruction(Op.PUSH, (5,)),  Instruction(Op.PUSH, (2 * PENALTY,)),  Instruction(Op.ADDQUAD, (4,)),
    Instruction(Op.PUSH, (2,)),  Instruction(Op.PUSH, (8,)),  Instruction(Op.PUSH, (2 * PENALTY,)),  Instruction(Op.ADDQUAD, (4,)),
    Instruction(Op.PUSH, (5,)),  Instruction(Op.PUSH, (8,)),  Instruction(Op.PUSH, (2 * PENALTY,)),  Instruction(Op.ADDQUAD, (4,)),

    # --- Output model ---
    Instruction(Op.PUSH, (0,)),        # output slot 0
    Instruction(Op.OUTPUT, (4,)),      # output[0] = r4 (model)
    Instruction(Op.HALT),
]


# =========================================================================
# Verifier
# =========================================================================
# Input slot 0: BQMX model (from encoder output)
# Input slot 1: BSMX sample (solution to verify)
# Output slot 0: energy (int)
# Output slot 1: valid flag (1 = valid, 0 = invalid)
#
# Validation checks:
#   1. Each row sums to 1 (each city assigned exactly one position)
#   2. Each column sums to 1 (each position has exactly one city)
#   3. Compute energy
#
# Uses AND-accumulation: start with valid=1, AND each check result.

VERIFIER = [
    # --- Load model and sample ---
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.INPUT, (0,)),       # r0 = model
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.INPUT, (1,)),       # r1 = sample

    # --- Initialize valid flag: r2 = 1 ---
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.STOW, (2,)),        # r2 = 1 (valid)

    # --- Check row sums = 1 ---
    # Row 0
    Instruction(Op.PUSH, (0,)),        # row
    Instruction(Op.ROWSUM, (1,)),       # push sum of sample row 0
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.EQ),                # sum == 1?
    Instruction(Op.LOAD, (2,)),
    Instruction(Op.AND),               # valid = valid AND (sum==1)
    Instruction(Op.STOW, (2,)),
    # Row 1
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.ROWSUM, (1,)),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.EQ),
    Instruction(Op.LOAD, (2,)),
    Instruction(Op.AND),
    Instruction(Op.STOW, (2,)),
    # Row 2
    Instruction(Op.PUSH, (2,)),
    Instruction(Op.ROWSUM, (1,)),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.EQ),
    Instruction(Op.LOAD, (2,)),
    Instruction(Op.AND),
    Instruction(Op.STOW, (2,)),

    # --- Check column sums = 1 ---
    # Col 0
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.COLSUM, (1,)),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.EQ),
    Instruction(Op.LOAD, (2,)),
    Instruction(Op.AND),
    Instruction(Op.STOW, (2,)),
    # Col 1
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.COLSUM, (1,)),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.EQ),
    Instruction(Op.LOAD, (2,)),
    Instruction(Op.AND),
    Instruction(Op.STOW, (2,)),
    # Col 2
    Instruction(Op.PUSH, (2,)),
    Instruction(Op.COLSUM, (1,)),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.EQ),
    Instruction(Op.LOAD, (2,)),
    Instruction(Op.AND),
    Instruction(Op.STOW, (2,)),

    # --- Compute energy ---
    Instruction(Op.ENERGY, (0, 1)),     # push energy(model, sample)
    Instruction(Op.STOW, (3,)),         # r3 = energy

    # --- Output ---
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.OUTPUT, (3,)),       # output[0] = energy
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.OUTPUT, (2,)),       # output[1] = valid flag
    Instruction(Op.HALT),
]


# =========================================================================
# Decoder
# =========================================================================
# Input slot 0: BSMX sample
# Input slot 1: N (number of cities)
# Output slot 0: tour order as vec<int> (city index at each position)
#
# For each position p in [0, N):
#   Find which city is assigned to position p using COLFIND.
#   COLFIND pops value, col -> pushes row where col has that value.

DECODER = [
    # --- Load sample and N ---
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.INPUT, (0,)),       # r0 = sample
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.INPUT, (1,)),       # r1 = N

    # --- Create output vec ---
    Instruction(Op.VECI, (2,)),         # r2 = empty vec<int>

    # --- For each position, find the city ---
    # Position 0: COLFIND pops value, col -> push row
    Instruction(Op.PUSH, (0,)),        # col = 0
    Instruction(Op.PUSH, (1,)),        # value = 1
    Instruction(Op.COLFIND, (0,)),      # push city at position 0
    Instruction(Op.VECPUSH, (2,)),      # append to tour

    # Position 1
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.COLFIND, (0,)),
    Instruction(Op.VECPUSH, (2,)),

    # Position 2
    Instruction(Op.PUSH, (2,)),
    Instruction(Op.PUSH, (1,)),
    Instruction(Op.COLFIND, (0,)),
    Instruction(Op.VECPUSH, (2,)),

    # --- Output tour ---
    Instruction(Op.PUSH, (0,)),
    Instruction(Op.OUTPUT, (2,)),       # output[0] = tour vec
    Instruction(Op.HALT),
]


# =========================================================================
# Run the pipeline
# =========================================================================

def run_tsp_example() -> None:
    """ Run the full TSP encoder -> verifier -> decoder pipeline. """
    from xqvm.core.vector import Vec
    from xqvm.core.xqmx import XQMX

    N = 3
    distances = [
        #  C0  C1  C2
        [  0, 10, 15],  # C0
        [ 10,  0, 35],  # C1
        [ 15, 35,  0],  # C2
    ]

    # Build flat distance vec
    dist_vec = Vec()
    for row in distances:
        for d in row:
            dist_vec.push(d)

    # --- Encoder ---
    print("=== Encoder ===")
    encoder_input = {0: N, 1: dist_vec}
    enc = run_program(ENCODER, encoder_input)
    model = enc.state.get_output(0)
    assert isinstance(model, XQMX)
    print(f"Model: size={model.size}, grid={model.rows}x{model.cols}")
    print(f"Linear coefficients: {dict(model.linear)}")
    print(f"Quadratic coefficients: {dict(model.quadratic)}")

    # --- Build known-good sample (simulate annealer) ---
    # Optimal tour: C0->C1->C2->C0
    # City 0 at position 0, city 1 at position 1, city 2 at position 2
    sample = XQMX.binary_sample(size=9, rows=3, cols=3)
    assignments = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    for i, val in enumerate(assignments):
        if val:
            sample.set_linear(i, float(val))

    # --- Verifier ---
    print("\n=== Verifier ===")
    verifier_input = {0: model, 1: sample}
    ver = run_program(VERIFIER, verifier_input)
    energy = ver.state.get_output(0)
    valid = ver.state.get_output(1)
    print(f"Energy: {energy}")
    print(f"Valid: {valid} ({'PASS' if valid else 'FAIL'})")

    # --- Decoder ---
    print("\n=== Decoder ===")
    decoder_input = {0: sample, 1: N}
    dec = run_program(DECODER, decoder_input)
    tour = dec.state.get_output(0)
    assert isinstance(tour, Vec)
    tour_list = [tour.get(i) for i in range(tour.length)]
    print(f"Tour: {tour_list}")
    city_names = ["C0", "C1", "C2"]
    route = " -> ".join(city_names[c] for c in tour_list)
    print(f"Route: {route} -> {city_names[tour_list[0]]}")

    # --- Validate results ---
    assert valid == 1, "Expected valid solution"
    assert tour_list == [0, 1, 2], f"Expected [0, 1, 2], got {tour_list}"

    # Compute expected tour distance manually
    tour_dist = distances[0][1] + distances[1][2] + distances[2][0]
    print(f"\nTour distance: {tour_dist}")
    assert tour_dist == 60, f"Expected distance 60, got {tour_dist}"

    print("\nAll assertions passed.")


if __name__ == "__main__":
    run_tsp_example()
