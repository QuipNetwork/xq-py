# Max-Cut — Maximum Cut Problem

QUBO formulation of the Maximum Cut problem for weighted graphs using the XQVM three-program architecture.

## Problem

Given a weighted undirected graph G=(V,E), partition the vertices into two sets to maximize the total weight of edges crossing the partition.

## QUBO Formulation

The problem is encoded as a BQMX with one binary variable per node. Variable `x_i = 0` or `1` indicates which partition node `i` belongs to.

For each edge `(i, j)` with weight `w`:

- `linear[i] -= w`, `linear[j] -= w`
- `quadratic[i,j] += 2w`

This minimizes the negated cut value, equivalent to maximizing the cut.

Edges are provided as a flat vector of triples `[i, j, w, i, j, w, ...]`.

## Programs

| Program            | Input                                        | Output                        |
| ------------------ | -------------------------------------------- | ----------------------------- |
| `encoder.xqasm`    | `0: N`, `1: edges (vec<int>)`                | `0: BQMX model`              |
| `verifier.xqasm`   | `0: BQMX model`, `1: BSMX sample`, `2: N`   | `0: energy`, `1: valid flag`  |
| `decoder.xqasm`    | `0: BSMX sample`, `1: N`                     | `0: partition (vec<int>)`     |

## Running

Default run (N=5, complete random graph, bisection partition):

```sh
python -m programs.maxcut.runner
```

Benchmark across multiple sizes:

```sh
python -m programs.maxcut.runner --bench 4 8 16 32
```

Trace execution (cannot be used with `--bench`):

```sh
python -m programs.maxcut.runner --trace
python -m programs.maxcut.runner --trace-compact
```

Save output to a file:

```sh
python -m programs.maxcut.runner --trace | tee results.txt
python -m programs.maxcut.runner --bench 4 8 16 > results.txt
```

Options:

```text
--bench N [N ...]   Problem sizes to benchmark
--trace             Trace with full detail (stack + register diffs)
--trace-compact     Trace with compact one-line output
--seed INT          Random seed (default: 42)
```
