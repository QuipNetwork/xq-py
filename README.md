# XQVM

**X-Quadratic Virtual Machine** — a specialized virtual machine for optimization problems and constraint programming.

This is the Python prototype implementation. See [`XQVM_SPEC.md`](XQVM_SPEC.md) for the full technical specification.

XQVM provides a unified instruction set for encoding, verifying, and decoding quadratic optimization models across binary, spin, and discrete variable domains.

## Architecture

Each optimization problem is defined by three independent programs:

- **Encoder** — transforms problem input into a quadratic model
- **Verifier** — validates solution quality and constraint satisfaction
- **Decoder** — transforms a quantum solution back into problem output

Programs share no state. Communication happens only through input (calldata) and output (results).

## Features

- 68-opcode instruction set
- Integer-only stack machine with typed registers (int, vec, xqmx)
- Sparse quadratic matrix (XQMX) with model and sample modes
- Grid operations for row/column indexing
- High-level functions: one-hot constraints, exclusion, implication, energy computation
- RANGE and ITER loop primitives
- Assembly syntax for human-readable programs

## Roadmap

**Complete:**

- Core VM: type system, machine state, 68 opcode handlers
- XQMX sparse quadratic matrix with grid ops and high-level functions
- Full test suite

**Upcoming:**

- Assembler (text `.xqasm` → instruction list)
- Execution tracer and matrix visualizer
- Example programs (TSP, Max-Cut)

## Setup

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```sh
python -m pytest tests/ -v
```
