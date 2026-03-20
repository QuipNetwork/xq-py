# XQCP — X-Quadratic Constraint Programmer

A constraint programming DSL that compiles high-level problem descriptions into three XQVM assembly programs: **encoder**, **verifier**, and **decoder**.

## Overview

Instead of writing `.xqasm` by hand, define your problem in Python using symbolic expressions, loops, and model operations. XQCP records these operations and compiles them into assembly that the XQVM executor can run directly.

```python
from xqvm.cp import Problem, Types
from xqvm.core import XQMXDomain

problem = Problem("MyProblem")

n = problem.input("n", type=Types.Int)
problem.define_model(size=n, domain=XQMXDomain.BINARY)

# ... define objective, constraints, decoder ...

programs = problem.compile()
# programs.encoder   -> str (.xqasm source)
# programs.verifier  -> str
# programs.decoder   -> str
```

## Architecture

XQCP uses a **record-then-compile** pattern:

1. **Record** — DSL calls (`input`, `define_model`, `range`, `stow`, `model.add_linear`, etc.) append `Action` objects to an ordered list.
2. **Compile** — The compiler walks the action list and emits assembly for each of the three programs, with automatic register allocation and loop nesting.

### Module Structure

| Module | Purpose |
|--------|---------|
| `problem.py` | `Problem` container, action recording, register allocator |
| `expression.py` | Symbolic expression tree nodes that emit assembly |
| `symbols.py` | Symbolic references: `InputRef`, `LoopVar`, `ModelRef`, `SampleRef`, `OutputRef` |
| `compiler.py` | Three compiler functions: `compile_encoder`, `compile_verifier`, `compile_decoder` |
| `__init__.py` | Public API re-exports and `triu()` helper |

## DSL Reference

### Inputs

Declare runtime inputs with a name and type:

```python
n = problem.input("n", type=Types.Int)           # integer scalar
edges = problem.input("edges", type=Types.Vec)    # integer vector
```

`InputRef` supports arithmetic (`+`, `-`, `*`, `//`, `%`, unary `-`) and vector operations:

```python
edges.get(i)        # VECGET: access element at index i
edges.veclen()      # VECLEN: get vector length
```

### Model

Define the XQMX model the encoder will build:

```python
# 1D model (flat variables)
problem.define_model(size=n, domain=XQMXDomain.BINARY)

# 2D model (grid with rows and columns)
problem.define_model(
    size=n * n, domain=XQMXDomain.BINARY,
    rows=n, cols=n,
)
```

Model operations:

```python
# 1D coordinates (flat index)
problem.model.add_linear(i, weight)
problem.model.add_quadratic(i, j, weight)

# 2D coordinates (row, col tuples)
problem.model.add_quadratic((row_a, col_a), (row_b, col_b), weight)

# Constraints (2D models only)
problem.model.apply_onehot_row(row, penalty=100)
problem.model.apply_onehot_col(col, penalty=100)
```

### Loops

`problem.range(start, end)` emits a `RANGE` loop. Use as a context manager:

```python
with problem.range(0, n) as i:
    with problem.range(i + 1, n) as j:
        # body executes for all (i, j) pairs where i < j
```

The yielded `LoopVar` supports full arithmetic and can be used as indices, coordinates, or operands.

### Stow

Evaluate an expression and store it in a register for reuse:

```python
dist = problem.stow("dist", edges.get(triu(i, j)))
w = problem.stow("w", edges.get(offset + 2))
```

Returns a `RegLoad` that can be used in subsequent expressions without re-emitting the computation.

### Outputs

Declare decoder outputs:

```python
tour = problem.output("tour", type=Types.Vec)

with problem.range(0, n) as pos:
    tour.append(problem.sample.colfind(col=pos, value=1))   # 2D grid decoder
    # or
    tour.append(problem.sample.getline(pos))                 # 1D flat decoder
```

### Sample

`problem.sample` provides read access to the solution sample in the decoder:

```python
problem.sample.colfind(col=pos, value=1)   # find row where column has value (2D)
problem.sample.getline(index)               # read variable by index (1D)
```

### Arithmetic

All symbolic types (`InputRef`, `LoopVar`, `RegLoad`, `Literal`) support:

| Operator | Assembly | Notes |
|----------|----------|-------|
| `a + b` | `ADD` | |
| `a - b` | `SUB` | |
| `a * b` | `MUL` | |
| `a // b` | `DIV` | Integer division |
| `a % b` | `MOD` | |
| `-a` | `NEG` | Unary negation |

### Helpers

```python
from xqvm.cp import triu

idx = triu(i, j)    # upper triangular index, compiles to IDXTRIU
```

## Compiled Output

`problem.compile()` returns a `CompiledPrograms` dataclass with three `.xqasm` source strings:

- **Encoder** — reads inputs, allocates the XQMX model, emits objective terms and constraints
- **Verifier** — reads model + sample + N, checks validity (onehot or binary domain), computes energy
- **Decoder** — reads sample + N, extracts the solution into output vectors

The verifier automatically selects the right validity check:
- **ROWSUM/COLSUM** loops when onehot constraints are present
- **Binary domain** check (GETLINE + 0-or-1 test) when no onehot constraints exist

## Examples

See the working XQCP programs:

- **TSP**: `programs/tsp/cp/tsp.py` — 2D grid model, onehot constraints, COLFIND decoder
- **Max-Cut**: `programs/maxcut/cp/maxcut.py` — 1D model, edge iteration, GETLINE decoder
