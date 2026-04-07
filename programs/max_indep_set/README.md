# Maximum Independent Set

Encode, verify, and decode the Maximum Independent Set problem on the XQVM.

## Problem

Given a graph with N vertices and E edges, find the largest subset of vertices such that no two vertices in the subset are adjacent.

## QUBO Formulation

- **Variables:** N binary variables (x_i = 1 means vertex i is in the set) + 1 slack variable per edge
- **Objective:** -A · Σ x_i (maximize set size)
- **Constraint:** For each edge (i,j): x_i + x_j ≤ 1 (at most one endpoint). Converted to equality with a slack variable: P · (x_i + x_j + s_ij - 1)²

This is the ATMOST pattern — PENALTYEQ + SLACK with target = 1 and one slack bit per edge.

## Two Encoder Versions

Both produce coefficient-identical QUBO models.

| Version | Opcodes Used | Lines |
|---------|-------------|-------|
| `asm/` | Existing only (ADDLINE, ADDQUAD, manual slack) | 123 |
| `asm_penaltyeq/` | PENALTYEQ + SLACK | 100 |

Speedup is ~1x because each constraint group has only 3 variables (x_i, x_j, s_ij) — too small for PENALTYEQ to outperform manual expansion.

## Usage

```sh
python -m programs.max_indep_set.runner -n 6 --no-solve
python -m programs.max_indep_set.runner --bench 4 6 8 10 --no-solve
```

## Input Format

| Slot | Type | Description |
|------|------|-------------|
| 0 | int | N (vertex count) |
| 1 | vec\<int\> | Edges (flat: [i, j, i, j, ...]) |
| 2 | int | A (vertex reward weight) |
| 3 | int | P (edge constraint penalty) |

## Output Format

**Encoder:** BQMX model with N + E variables

**Verifier:** energy (int), valid flag (1/0)

**Decoder:** vec\<int\> of selected vertex indices
