# 0-1 Knapsack

Encode, verify, and decode the 0-1 Knapsack problem on the XQVM.

## Problem

Given N items with values and weights, and a knapsack with capacity W, select items to maximize total value without exceeding the capacity.

## QUBO Formulation

- **Variables:** N item variables + S slack variables (S = ⌊log₂(W)⌋ + 1)
- **Objective:** Minimize −Σ vᵢ·xᵢ (negated value maximization)
- **Constraint:** P·(Σ wᵢ·xᵢ + Σ 2ʲ·sⱼ − W)² (capacity via slack encoding)

## Three Encoder Versions

Three encoder implementations demonstrate the impact of new high-level opcodes. All produce identical QUBO models (verified at coefficient level, N=4 to N=15).

| Version | Opcodes Used | Encoder Lines |
|---------|-------------|---------------|
| `asm/` | Existing only (ADDLINE, ADDQUAD, loops) | 229 |
| `asm_penaltyeq/` | Existing + PENALTYEQ | 135 |
| `asm_full/` | Existing + PENALTYEQ + SLACK | 94 |

### Encoder Speedup

| N | asm | asm_penaltyeq | asm_full | Speedup (penaltyeq) | Speedup (full) |
|---|-----|---------------|----------|---------------------|----------------|
| 4 | 0.001840s | 0.000411s | 0.000350s | 4.5x | 5.3x |
| 6 | 0.002349s | 0.000446s | 0.000293s | 5.3x | 8.0x |
| 8 | 0.005873s | 0.000550s | 0.000335s | 10.7x | 17.5x |
| 10 | 0.005702s | 0.000569s | 0.000429s | 10.0x | 13.3x |
| 15 | 0.014487s | 0.000788s | 0.000846s | 18.4x | 17.1x |
| 20 | 0.017647s | 0.003120s | 0.000853s | 5.7x | 20.7x |

## Usage

```sh
# Single run (5 items, ~50% capacity, with solver)
python -m programs.knapsack.runner -n 5

# Without solver (greedy sample)
python -m programs.knapsack.runner -n 5 --no-solve

# Compare encoder versions
python -m programs.knapsack.runner -n 5 --no-solve --src asm
python -m programs.knapsack.runner -n 5 --no-solve --src asm_penaltyeq
python -m programs.knapsack.runner -n 5 --no-solve --src asm_full

# Custom capacity and penalty
python -m programs.knapsack.runner -n 8 -w 50 -p 300

# Show QUBO matrix
python -m programs.knapsack.runner -n 4 --matrix --no-solve

# Trace execution
python -m programs.knapsack.runner -n 3 --trace --no-solve

# Benchmark multiple sizes
python -m programs.knapsack.runner --bench 4 6 8 10 15 20 --no-solve --src asm_full
```

## Input Format

| Slot | Type | Description |
|------|------|-------------|
| 0 | int | N (item count) |
| 1 | vec\<int\> | Item values |
| 2 | vec\<int\> | Item weights |
| 3 | int | W (capacity) |
| 4 | int | P (penalty weight) |

## Output Format

**Encoder:** BQMX model with N + S variables

**Verifier:** energy (int), valid flag (1/0), total weight (int)

**Decoder:** vec\<int\> of selected item indices

## Tests

34 tests in `tests/test_penalty_eq.py`. Run with:

```sh
python -m pytest tests/test_penalty_eq.py -v
```

**PENALTYEQ function tests (13):** Verify the `expand_penalty_eq` function produces correct QUBO coefficients — matches manual ADDLINE/ADDQUAD expansion, matches ONEHOT as a special case (coeffs=[1,1,...], target=1), energy differences between solutions match expected penalty values P·(sum−target)², and edge cases (zero penalty, single variable, sample mode rejection, mismatched lengths).

**PENALTYEQ opcode tests (4):** Verify the opcode running through the executor produces the same results as calling the function directly. Tests the full VM path: instruction dispatch → handler → function call → model modification.

**SLACK opcode tests (8):** Verify correct slack variable generation for various capacities (W=1,7,10,31,32), confirm all coefficients are powers of 2, confirm coefficients can represent the full capacity range, and verify SLACK appends to existing vecs without overwriting.

**BITLEN opcode tests (9):** Verify correct bit length computation for powers of 2 and non-powers, and cross-check that BITLEN output matches the number of slack variables SLACK produces.

**Knapsack integration tests (12):** Run all three encoder versions (asm, asm_penaltyeq, asm_full) with identical inputs at N=4,6,8,10,15 and verify they produce identical models (same size, same linear dict, same quadratic dict). Also verify greedy sample satisfies weight capacity and slack encoding is correct.
