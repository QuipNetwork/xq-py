# Graph Coloring

Encode, verify, and decode the Graph Coloring problem on the XQVM.

## Problem

Given a graph with N vertices and E edges, assign K colors to vertices such that no adjacent vertices share the same color.

## Two Encoding Strategies

### 1. One-Hot Encoding (`asm/`)

Standard approach: N×K binary grid, x[i,c] = 1 means vertex i gets color c.

- **ONEHOTR** per vertex (exactly one color)
- **EXCLUDE** per edge per color (no same-color adjacency)
- N×K variables, no auxiliary variables
- All constraints are quadratic — no REDUCE needed

### 2. Binary (CHROMBIN) Encoding (`asm_chrombin/`)

Compact approach: 2 bits per vertex (for K=4), encoding color as bit0 + 2×bit1.

- No one-hot constraint needed (all 4 bit patterns are valid colors)
- Adjacency constraint becomes quartic: P · δ(a₀,b₀) · δ(a₁,b₁)
- Uses **REDUCE** for cubic and quartic degree reduction
- Fewer original variables (2N vs 4N for K=4) but adds 6 auxiliary variables per edge

### Variable Count Comparison

| N | K | One-Hot | CHROMBIN (original + aux) |
|---|---|---------|--------------------------|
| 10 | 4 | 40 | 20 + 6E |
| 20 | 4 | 80 | 40 + 6E |
| 50 | 4 | 200 | 100 + 6E |

CHROMBIN uses fewer variables only when the graph is sparse (E < N·K/6).

## Symmetry Breaking

Colors are interchangeable — swapping all red/blue vertices gives an equivalent solution. Both encoders fix vertex 0 to color 0 by adding penalty terms, eliminating K! equivalent colorings.

This is done with simple ADDLINE calls (bias on one variable), not a dedicated opcode.

## REDUCE P_aux Requirement

**Critical:** When using REDUCE for degree reduction, the auxiliary constraint penalty P_aux must exceed the maximum coefficient that multiplies the auxiliary variable in the reduced terms.

In CHROMBIN, the quartic term has coefficient 4P (where P is the adjacency penalty). If P_aux < 4P, the solver can "cheat" — it saves more on the reduced term by setting the auxiliary wrong than it pays in penalty. The QUBO then gives incorrect energies.

**Rule:** P_aux > max coefficient using the auxiliary variable. When in doubt, make P_aux 5–10× your problem penalty — that's always safe.

| P (adjacency) | Min P_aux | Recommended P_aux |
|---------------|-----------|-------------------|
| 100 | 400 | 500+ |
| 50 | 200 | 250+ |

This applies to all REDUCE usage, not just CHROMBIN. Our earlier tests (cubic/quartic optimization, portfolio) didn't hit this because problem coefficients (≤10) were small relative to P_aux (50).

## Usage

```sh
# One-hot encoding (K=3)
python -m programs.graph_coloring.runner -n 6 -k 3 --no-solve

# One-hot encoding (K=4)
python -m programs.graph_coloring.runner -n 8 -k 4 --no-solve
```

## Input Format

### One-hot (`asm/`)

| Slot | Type | Description |
|------|------|-------------|
| 0 | int | N (vertex count) |
| 1 | int | K (color count) |
| 2 | int | E (edge count) |
| 3 | vec\<int\> | Edges (flat: [i, j, i, j, ...]) |
| 4 | int | P_oh (one-hot penalty) |
| 5 | int | P_ex (exclusion penalty) |
| 6 | int | P_sym (symmetry breaking penalty) |

### CHROMBIN (`asm_chrombin/`)

| Slot | Type | Description |
|------|------|-------------|
| 0 | int | N (vertex count) |
| 1 | int | E (edge count) |
| 2 | vec\<int\> | Edges (flat: [i, j, ...]) |
| 3 | int | P (adjacency penalty) |
| 4 | int | P_aux (auxiliary penalty, must be > 4×P) |
| 5 | int | P_sym (symmetry breaking penalty) |
