# Cubic Optimization

Encode, verify, and decode a random cubic polynomial optimization on the XQVM.

## Problem

Minimize a polynomial with cubic (degree 3) terms:

H = Σ c_ijk · x_i · x_j · x_k

This is a HOBO (Higher-Order Binary Optimization) problem — the XQMX model only supports quadratic terms, so each cubic term must be reduced to quadratic using REDUCE.

## QUBO Encoding

For each cubic term c · x_i · x_j · x_k:

1. w = REDUCE(x_i, x_j) — introduces auxiliary variable w = x_i · x_j
2. Add c · w · x_k as a quadratic term (ADDQUAD)

Total variables: N original + T auxiliary (one per cubic term).

## Two Encoder Versions

Both produce coefficient-identical QUBO models.

| Version | Opcodes Used | Lines |
|---------|-------------|-------|
| `asm/` | Manual auxiliary constraint (4 ADDQUAD/ADDLINE per term) | 116 |
| `asm_reduce/` | REDUCE (1 call per term) | 80 |

## Usage

```sh
python -m programs.cubic_opt.runner -n 6 --no-solve
python -m programs.cubic_opt.runner --bench 4 5 6 8 --no-solve
```

## Input Format

| Slot | Type | Description |
|------|------|-------------|
| 0 | int | N (variable count) |
| 1 | int | T (number of cubic terms) |
| 2 | vec\<int\> | Terms (flat: [i, j, k, coeff, ...]) |
| 3 | int | P_aux (auxiliary constraint penalty) |

## Output Format

**Encoder:** BQMX model with N + T variables

**Verifier:** energy (int), valid flag (1/0), cubic polynomial value (int)

**Decoder:** vec\<int\> of variable assignments (original N variables only)
