# XQVM

**X-Quadratic Virtual Machine** — a specialized virtual machine for encoding, verifying, and decoding quadratic optimization models.

This is the Python prototype implementation. See [`XQVM_SPEC.md`](XQVM_SPEC.md) for the full technical specification.

## Overview

XQVM provides a unified instruction set for working with quadratic optimization models (QUBO/Ising) across binary, spin, and discrete variable domains. Each optimization problem is defined by three independent programs:

- **Encoder** — transforms problem input into a quadratic model (XQMX)
- **Verifier** — validates solution quality and constraint satisfaction
- **Decoder** — extracts a human-readable solution from a quantum sample

Programs share no state. Communication happens only through input (calldata) and output (results).

### Key Features

- 84-opcode instruction set
- Integer-only stack machine with typed registers (int, vec, xqmx)
- Sparse quadratic matrix (XQMX) with model and sample modes
- Grid operations for row/column indexing
- High-level functions: one-hot constraints, exclusion, implication, energy computation
- RANGE and ITER loop primitives
- Constraint programming DSL (XQCP) for generating programs from high-level problem descriptions
- Pluggable solver backends (XQSA) with D-Wave neal support

## Getting Started

### Installation

```sh
git clone <repo-url> && cd xqvm-py
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set up the pre-commit hook (runs lint + tests before each commit):

```sh
git config core.hooksPath .githooks
```

### Quick Example

Assemble and run a minimal program that pushes two values, adds them, and stores the result:

```python
from xqvm.assembler import assemble
from xqvm.core import Executor

source = """
PUSH 10
PUSH 5
ADD
STOW r0
HALT
"""

program = assemble(source)
executor = Executor()
executor.execute(program.program)

result = executor.state.get_register(0)
print(result)  # 15
```

### Constraint Programming DSL (XQCP)

Instead of writing assembly by hand, you can define optimization problems at a high level using the [`xqcp/`](xqcp/) package. XQCP records a symbolic problem description and compiles it into the three XQVM assembly programs automatically:

```python
from xqcp import Problem, Types, triu
from xqvm.core import XQMXDomain

problem = Problem("MyProblem")
n = problem.input("size", type=Types.Int)
problem.define_model(size=n * n, domain=XQMXDomain.BINARY, rows=n, cols=n)

# ... define objective, constraints, outputs ...

programs = problem.compile()
print(programs.encoder)   # generated .xqasm source
print(programs.verifier)
print(programs.decoder)
```

See the [`xqcp/README.md`](xqcp/README.md) for the full DSL reference.

### Example Programs

The [`programs/`](programs/) directory contains complete optimization problems implemented with the three-program architecture. Each includes both hand-written assembly (`asm/`) and XQCP-generated assembly (`cp/`), making them a good starting point for understanding how the VM and DSL work together:

- **[`programs/tsp/`](programs/tsp/)** — Travelling Salesman Problem (QUBO formulation with N*N binary variables)
- **[`programs/maxcut/`](programs/maxcut/)** — Maximum Cut Problem (weighted graph partitioning)

Each program comes with a runner supporting benchmarking, execution tracing, and matrix visualization:

```sh
# Run TSP for 5 cities with a hardcoded sample
python -m programs.tsp.runner -n 5 --no-solve

# Run Max-Cut for 8 nodes with simulated annealing
python -m programs.maxcut.runner -n 8

# Benchmark TSP across sizes with execution trace
python -m programs.tsp.runner --bench 4 8 16

# View the QUBO matrix
python -m programs.maxcut.runner -n 4 --matrix --no-solve
```

See each program's README for QUBO formulations and detailed usage. The `cp/` subdirectories contain the XQCP source (e.g., [`programs/tsp/cp/tsp.py`](programs/tsp/cp/tsp.py)) that generates the equivalent assembly.

## Project Structure

```
xqvm/              VM implementation
  core/            Executor, state, opcodes, types (Vec, XQMX)
  assembler/       Parser, validator, program assembly
xqcp/              Constraint programming DSL → assembly compiler
xqsa/              Solver backends (D-Wave neal)
tools/             Tracer, disassembler, ASCII matrix visualizer
programs/          Example optimization problems
tests/             Test suite (499 tests)
```

## Development

### Tests

```sh
python -m pytest tests/ -v
```

### Linting

Formatting and linting are enforced by [ruff](https://docs.astral.sh/ruff/) (config in `ruff.toml`):

```sh
ruff check .
ruff format .
```

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).
