# Vertex Cover

Encode, verify, and decode the Minimum Vertex Cover problem on the XQVM.

## Problem

Given a graph with N vertices and E edges, find the smallest set of vertices such that every edge has at least one endpoint in the set.

## QUBO Formulation

- **Variables:** N binary variables (xᵢ = 1 if vertex i is in the cover)
- **Objective:** A · Σ xᵢ (minimize cover size)
- **Constraint:** B · (1 - xᵢ)(1 - xⱼ) for each edge (penalize uncovered edges)
- **Requirement:** B > A so constraints dominate

## Usage

```sh
# Single run (6 vertices, ~50% edge density)
python -m programs.vertex_cover.runner -n 6

# Without solver
python -m programs.vertex_cover.runner -n 6 --no-solve

# Custom weights
python -m programs.vertex_cover.runner -n 8 -a 1 -b 20

# Benchmark
python -m programs.vertex_cover.runner --bench 4 6 8 10 15
```

## Input Format

| Slot | Type | Description |
|------|------|-------------|
| 0 | int | N (vertex count) |
| 1 | vec\<int\> | Edges (flat pairs [i, j, i, j, ...]) |
| 2 | int | A (vertex cost weight) |
| 3 | int | B (edge constraint penalty) |
