# xq-py

**A Python implementation of the Quip Network's Quantum Virtual Machine (XQVM)**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Quip Network](https://img.shields.io/badge/Quip-Network-purple)](https://quip.network)

---

## Overview

`xq-py` is the Python implementation of **XQVM** — the Quip Network's quantum virtual machine for expressing and executing binary optimization models. It is the Python counterpart to [`xq-rs`](https://github.com/QuipNetwork/xq-rs), which is the canonical Rust implementation.

XQVM is a **stack-based virtual machine** designed to construct and execute **X-quadratic models** (QUBO/Ising formulations) targeting quantum annealers such as D-Wave hardware. It is the execution core used by the [quip-protocol](https://github.com/QuipNetwork/quip-protocol) node to power quantum proof-of-work mining.

### What is XQVM?

XQVM operates on **Ising models** — mathematical structures representing optimization problems that quantum annealers are designed to solve. The VM:

- Assembles binary optimization problems from a stack-based instruction set
- Executes programs written in **XQASM** (the X-Quadratic Assembly language)
- Produces **QUBO/Ising** model descriptions that can be submitted to D-Wave QPUs, GPU simulators, or CPU-based simulated annealing
- Is used inside Quip Network nodes to generate and validate quantum proof-of-work puzzles

### Relationship to Other Quip Repos

```
quip-protocol          ← Full network node (mining, P2P, consensus)
    └── uses xqvm for quantum PoW puzzle generation

xq-py  (this repo)    ← Python XQVM implementation
xq-rs                 ← Rust XQVM implementation (canonical)
```

---

## Architecture

The VM follows a **register + stack** model:

- **256-slot register file** — holds typed values: integers, integer vectors, QUBO/Ising models (`XqmxModel`), and candidate solutions (`XqmxSample`)
- **Loop stack** — drives `RANGE`/`ITER` iteration for constructing model coefficients
- **Instruction set** — ~76 opcodes covering arithmetic, model construction, sampling, and control flow

### Core Concepts

| Term | Description |
|------|-------------|
| `XqmxModel` | An Ising/QUBO model — the optimization problem to solve |
| `XqmxSample` | A candidate solution returned by a sampler |
| `QUBO` | Quadratic Unconstrained Binary Optimization — equivalent formulation to Ising |
| `Ising` | A spin-glass model used as the proof-of-work puzzle in Quip mining |

---

## Installation

> ⚠️ **This library is under active development and not yet production-ready.**

```bash
# Clone the repo
git clone https://github.com/QuipNetwork/xq-py.git
cd xq-py

# Create a virtual environment (Python 3.10+ required)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in editable mode
pip install -U pip setuptools wheel
pip install -e .
```

---

## Usage

> Full usage examples will be added as the implementation matures.

### Assembling and Running an XQASM Program

```python
# Coming soon — Python API to assemble and execute .xqasm programs
from xqvm import Assembler, VM

program = Assembler.from_file("program.xqasm")
vm = VM()
result = vm.execute(program)
print(result.sample)
```

### Working with Ising Models

```python
# Coming soon — Pythonic model construction API
from xqvm import IsingModel

model = IsingModel(n_qubits=64)
model.add_coupling(0, 1, weight=-1.0)
model.add_bias(0, weight=0.5)

samples = model.sample(num_reads=100)
```

---

## XQASM Instruction Set

XQASM is a human-readable assembly language that compiles to XQVM bytecode (`.xqbc`). Example:

```asm
; Push two integers and add them
PUSH 10
PUSH 32
ADD
HALT
```

The compiled `.xqbc` format is a bare instruction stream with no header. Each instruction is an **opcode byte** followed by operands in big-endian byte order.

To assemble and inspect programs (from `xq-rs`):

```bash
# Assemble
xqasm program.xqasm -o program.xqbc

# Disassemble to inspect encoding
xqdism program.xqbc

# Execute
xqvm program.xqbc
```

---

## Relation to Quip Proof-of-Work

The Quip Network uses Ising model optimization as its **quantum proof-of-work** mechanism. The mining flow is:

1. A block header hash seeds a deterministic Ising problem
2. XQVM assembles the Ising model for that problem
3. Miners (CPU/GPU/QPU) sample solutions from the model
4. Valid solutions must meet energy, diversity, and minimum-count thresholds
5. The first miner to find valid solutions wins the block

`xq-py` provides the Python-native interface for steps 1–3, making it easy to integrate XQVM into Python-based mining nodes and tooling.

---

## Development Status

| Feature | Status |
|---------|--------|
| XQASM parser | 🔲 Planned |
| Bytecode assembler | 🔲 Planned |
| Stack VM core | 🔲 Planned |
| IsingModel / QUBO builder | 🔲 Planned |
| D-Wave sampler integration | 🔲 Planned |
| CPU (simulated annealing) backend | 🔲 Planned |
| GPU (CUDA/MPS) backend | 🔲 Planned |
| Tests | 🔲 Planned |
| PyPI release | 🔲 Planned |

Contributions are welcome! See [Contributing](#contributing) below.

---

## Contributing

We welcome pull requests, issues, and discussion. Good starting points:

1. **Port core VM logic** from [`xq-rs`](https://github.com/QuipNetwork/xq-rs) — the Rust implementation is the reference
2. **Add tests** — unit tests for opcodes, model construction, and sampling
3. **Documentation** — examples, docstrings, and tutorials

Please open an issue before starting large changes so we can coordinate.

---

## Related Repositories

| Repo | Description |
|------|-------------|
| [`xq-rs`](https://github.com/QuipNetwork/xq-rs) | Rust XQVM — canonical implementation |
| [`quip-protocol`](https://github.com/QuipNetwork/quip-protocol) | Full quantum network node |
| [`quip-node-manager`](https://github.com/QuipNetwork/quip-node-manager) | GUI for managing Quip nodes |
| [`ethereum-sdk`](https://github.com/QuipNetwork/ethereum-sdk) | Smart contracts + TypeScript SDK |

---

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE) for details.

---

## About Quip Network

[Quip Network](https://quip.network) is building the world's shared quantum computer — a decentralized marketplace where quantum operators contribute spare QPU time, classical nodes verify solutions, and anyone can submit jobs to real quantum hardware.

> Run a node · Earn QUIP · Defend against quantum threats