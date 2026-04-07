# Max-3-SAT

Encode, verify, and decode the Maximum 3-Satisfiability problem on the XQVM.

## Problem

Given a Boolean formula in CNF with M clauses, each containing 3 literals, find an assignment of N variables that satisfies the maximum number of clauses.

## Significance

This is the first HOBO (Higher-Order Binary Optimization) problem encoded on the XQVM. Each clause produces a cubic penalty term (degree 3) that must be reduced to quadratic (degree 2) using auxiliary variables before it can be stored in the XQMX model.

## QUBO Formulation

- **Variables:** N original binary variables + M auxiliary variables (one per clause)
- **Clause penalty:** For clause (l₁ ∨ l₂ ∨ l₃), the penalty for all literals being false is P · (1-l₁)(1-l₂)(1-l₃)
- **Degree reduction:** The cubic term is reduced by introducing auxiliary variable w = (1-l₁)(1-l₂), with a quadratic constraint P_aux · (ab - 2aw - 2bw + 3w) ensuring w correctly represents the product
- **Energy:** Equals the number of unsatisfied clauses (verified by brute force for N=4, M=3)

## Usage

```sh
# Single run (4 variables, 6 clauses)
python -m programs.max3sat.runner -n 4 -m 6 --no-solve

# With solver
python -m programs.max3sat.runner -n 8 -m 16

# Custom penalties
python -m programs.max3sat.runner -n 6 -m 10 -p 1 --p-aux 20

# Benchmark (pairs of N, M)
python -m programs.max3sat.runner --bench 4 6 6 12 8 16 10 30 15 50 20 80 --no-solve
```

## Input Format

| Slot | Type | Description |
|------|------|-------------|
| 0 | int | N (variable count) |
| 1 | int | M (clause count) |
| 2 | vec\<int\> | Clauses (flat: [var, neg, var, neg, var, neg, ...]) |
| 3 | int | P (clause penalty weight) |
| 4 | int | P_aux (auxiliary constraint penalty, must be >> P) |

## Output Format

**Encoder:** BQMX model with N + M variables

**Verifier:** energy (int), valid flag (1/0), satisfied count (int)

**Decoder:** vec\<int\> of variable assignments (0 or 1, original N variables only)

## Benchmark Results

| N | M | Vars | Linear | Quad | Encoder | Satisfied |
|---|---|------|--------|------|---------|-----------|
| 4 | 6 | 10 | 8 | 24 | 0.001108s | 6/6 |
| 6 | 12 | 18 | 16 | 50 | 0.001972s | 12/12 |
| 8 | 16 | 24 | 21 | 67 | 0.002564s | 16/16 |
| 10 | 30 | 40 | 38 | 125 | 0.004478s | 29/30 |
| 15 | 50 | 65 | 62 | 219 | 0.007881s | 48/50 |
| 20 | 80 | 100 | 96 | 336 | 0.012016s | 80/80 |
