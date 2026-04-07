# Set Cover

Encode, verify, and decode the Minimum Set Cover problem on the XQVM.

## Problem

Given a universe of N elements and M subsets, find the minimum number of subsets whose union covers all elements.

## QUBO Formulation

- **Variables:** M binary variables (xⱼ = 1 means subset j is selected) + slack variables for coverage constraints
- **Objective:** A · Σ xⱼ (minimize number of selected subsets)
- **Constraint:** For each element i, at least one covering subset must be selected. This is an ATLEAST 1 constraint over the variable-size coverage group.

## Two Encoder Versions

| Version | Opcodes Used | Lines |
|---------|-------------|-------|
| `asm/` | PENALTYEQ + manual slack computation (SHL, NEG, loops) | 173 |
| `asm_atleast/` | ATLEAST (handles slacks internally) | 87 |

### Encoder Speedup

| N | M | asm | asm_atleast | Speedup |
|---|---|-----|-------------|---------|
| 4 | 6 | 0.000572s | 0.000299s | 1.9x |
| 6 | 10 | 0.001076s | 0.000567s | 1.9x |
| 8 | 12 | 0.001620s | 0.000811s | 2.0x |
| 10 | 15 | 0.002090s | 0.001129s | 1.9x |
| 15 | 25 | 0.004308s | 0.002414s | 1.8x |
| 20 | 30 | 0.008352s | 0.003868s | 2.2x |

Unlike Vertex Cover (where every ATLEAST operates on 2 variables), Set Cover has variable-size coverage groups (2–6+ subsets per element), giving ATLEAST meaningful speedup.

## Usage

```sh
# Single run
python -m programs.set_cover.runner -n 6 -m 8 --no-solve

# Compare encoders
python -m programs.set_cover.runner -n 10 -m 15 --no-solve --src asm
python -m programs.set_cover.runner -n 10 -m 15 --no-solve --src asm_atleast

# Benchmark (pairs of N, M)
python -m programs.set_cover.runner --bench 4 6 6 10 8 12 10 15 --no-solve
```

## Input Format

| Slot | Type | Description |
|------|------|-------------|
| 0 | int | N (universe size) |
| 1 | int | M (number of subsets) |
| 2 | vec\<int\> | Coverage (flat: [count, idx, idx, ..., count, idx, ...]) |
| 3 | int | A (subset cost weight) |
| 4 | int | P (coverage penalty) |
