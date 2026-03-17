# TSP — Travelling Salesman Problem

QUBO formulation of the Travelling Salesman Problem for N cities using the XQVM three-program architecture.

## Problem

Given N cities and a symmetric distance matrix, find the shortest Hamiltonian cycle (a tour visiting each city exactly once and returning to the start).

## QUBO Formulation

The problem is encoded as an N×N binary quadratic model (BQMX) where variable `x[i,p] = 1` means city `i` is assigned to position `p` in the tour.

- **Objective:** Minimize total tour distance via quadratic couplings between adjacent positions.
- **Row constraints:** Each city appears in exactly one position (ONEHOT per row).
- **Column constraints:** Each position has exactly one city (manual linear/quadratic penalty terms).

Distances are provided as a flattened upper-triangle vector indexed by `IDXTRIU(i, j)` with `N*(N-1)/2` elements.

## Programs

| Program | Input | Output |
|---------|-------|--------|
| `encoder.xqasm` | `0: N`, `1: distances (vec<int>)` | `0: BQMX model` |
| `verifier.xqasm` | `0: BQMX model`, `1: BSMX sample`, `2: N` | `0: energy`, `1: valid flag` |
| `decoder.xqasm` | `0: BSMX sample`, `1: N` | `0: tour (vec<int>)` |

## Running

Default run (N=5, full pipeline):

```sh
python -m programs.tsp.runner
```

Benchmark across multiple sizes:

```sh
python -m programs.tsp.runner --bench 4 8 16 32
```

Trace execution (cannot be used with `--bench`):

```sh
python -m programs.tsp.runner --trace
python -m programs.tsp.runner --trace-compact
```

Save output to a file:

```sh
python -m programs.tsp.runner --trace > tee results.txt
python -m programs.tsp.runner --trace | tee results.txt
```

```sh
python -m programs.tsp.runner --bench 4 8 16 > results.txt
python -m programs.tsp.runner --bench 4 8 16 | tee results.txt
```

Options:

```text
--bench N [N ...]   Problem sizes to benchmark
--trace             Trace with full detail (stack + register diffs)
--trace-compact     Trace with compact one-line output
--seed INT          Random seed (default: 42)
```
