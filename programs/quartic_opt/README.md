# Quartic Optimization

Encode, verify, and decode a random quartic polynomial optimization on the XQVM.

## Problem

Minimize a polynomial with quartic (degree 4) terms:

H = Σ c_ijkl · x_i · x_j · x_k · x_l

This is a HOBO problem requiring chained REDUCE — each quartic term needs two REDUCE calls where the output of the first feeds into the second.

## QUBO Encoding

For each quartic term c · x_i · x_j · x_k · x_l:

1. w₁ = REDUCE(x_i, x_j) — auxiliary w₁ = x_i · x_j
2. w₂ = REDUCE(w₁, x_k) — auxiliary w₂ = x_i · x_j · x_k
3. Add c · w₂ · x_l as a quadratic term (ADDQUAD)

Total variables: N original + 2T auxiliary (two per quartic term).

## Two Encoder Versions

Both produce coefficient-identical QUBO models.

| Version | Opcodes Used | Lines |
|---------|-------------|-------|
| `asm/` | Manual auxiliary constraint (8 ADDQUAD/ADDLINE per term) | 151 |
| `asm_reduce/` | REDUCE (2 chained calls per term) | 91 |

## Usage

```sh
python -m programs.quartic_opt.runner -n 6 --no-solve
python -m programs.quartic_opt.runner --bench 4 5 6 --no-solve
```

## Input Format

| Slot | Type | Description |
|------|------|-------------|
| 0 | int | N (variable count) |
| 1 | int | T (number of quartic terms) |
| 2 | vec\<int\> | Terms (flat: [i, j, k, l, coeff, ...]) |
| 3 | int | P_aux (auxiliary constraint penalty) |

## Output Format

**Encoder:** BQMX model with N + 2T variables

**Verifier:** energy (int), valid flag (1/0), quartic polynomial value (int)

**Decoder:** vec\<int\> of variable assignments (original N variables only)
