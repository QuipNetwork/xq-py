# XQVM Technical Specification

## X-Quadratic Virtual Machine

A specialized virtual machine for encoding, verifying, and decoding quantum optimization problems. Provides a unified instruction set for manipulating quadratic models across variable domains (binary, spin, chromatic).

## Three-Program Architecture

Each optimization problem is defined by three independent programs sharing the same instruction set:

- **Encoder** — Transforms problem-specific input into an XQMX model suitable for quantum optimization.
- **Verifier** — Validates solution quality and constraint satisfaction.
- **Decoder** — Transforms quantum solution back into problem-specific output.

Programs execute independently with no shared state. Communication occurs only through calldata (`input`) and results (`output`).

---

## Machine State

```python
{
    "pc": 0,                  # Program counter
    "stack": [],              # Integer-only stack

    "registers": {            # slot (0-255) → value
        0: 42,                # int
        1: [1, 2, 3, 4],      # vec
        2: {                  # xqmx
            "mode": "model",          # "model" | "sample"
            "domain": [0, 1],         # [0,1] | [-1,1] | [0,...,k-1]
            "size": 9,
            "rows": None,
            "cols": None,
            "linear": {},             # sparse: var_index → value
            "quadratic": {}           # sparse: triu_index → value
        }
    },

    "jc": {
        "targets": {},        # target_id → pc
        "loop_stack": []      # loop state stack (LIFO)
    },

    "input": {},              # calldata: slot → value
    "output": {}              # results: slot → value
}
```

---

## Type System

### Stack

- Integer only. All primitive operations work on integers.
- Maximum depth: 8192 (2^13)

### Registers (`r0`–`r255`)

- 8-bit slot ID (0-255)
- Types: `int` | `vec` | `xqmx`
- No pointers, no type coercion
- Only `int` registers exchange with stack via `LOAD`/`STOW`
- `vec` and `xqmx` accessed only through specialized opcodes

### `vec`

- Homogeneous dynamic array: `vec<int>`, `vec<xqmx>` with support for nesting (`vec<vec<int>>`)
- Element type inferred and locked on first push, or explicit via `VECI`/`VECX` opcodes. Type validation on mutate operations
- Tracks length and capacity

### `xqmx`

- Sparse x-quadratic matrix
- **Mode:** `model` (linear & quadratic are hamiltonian coefficients) or `sample` (linear are variable assignments, quadratic is nil)
- **Domain:** `[0,1]` binary, `[-1,1]` spin, `[-k, ..., 0 , ..., k-1]` chromatic
- **Dimension:** `size` (total linear variables), optional `rows`/`cols` for grid layout
- **Storage:** Sparse tables for `linear` and `quadratic`

---

## Instruction Set (85 opcodes)

### Control Flow

| Code | Opcode | Operands | Stack | Description |
| ------ | -------- | ---------- | ------- | ------------- |
| 0x00 | `NOP` | — | — | No operation |
| 0x01 | `TARGET` | `.N` | — | Mark valid jump destination |
| 0x02 | `JUMP` | `.N` | — | Unconditional jump to target |
| 0x03 | `JUMPI` | `.N` | pop cond | Jump to target if non-zero |
| 0x04 | `NEXT` | — | — | Advance loop index, jump back or exit |
| 0x05 | `LVAL` | `<reg>` | — | Copy current loop value to register |
| 0x06 | `RANGE` | — | pop count, start | Start range loop: iterate [start, start+count) |
| 0x07 | `ITER` | `<reg>` | pop end_idx, start_idx | Start vec iteration: iterate vec[start:end] |
| 0x09 | `HALT` | — | — | Stop execution |

### Register Manipulation

| Code | Opcode | Operands | Stack | Description |
| ------ | -------- | ---------- | ------- | ------------- |
| 0x0A | `LOAD` | `<reg>` | → push reg[n] | Push int register value onto stack |
| 0x0B | `STOW` | `<reg>` | pop → reg[n] | Pop stack into int register |
| 0x0C | `DROP` | `<reg>` | — | Clear register (reset to unset) |
| 0x0E | `INPUT` | `<reg>` | pop slot → reg[n] | Load calldata slot into register |
| 0x0F | `OUTPUT` | `<reg>` | pop slot | Write register to output slot |

### Load Constant

| Code | Opcode | Operands | Stack | Description |
| ------ | -------- | ---------- | ------- | ------------- |
| 0x11 | `LDC1` | 1 × `<int>` | → push value | Load 1-byte constant |
| 0x12 | `LDC2` | 2 × `<int>` | → push value | Load 2-byte constant |
| 0x13 | `LDC3` | 3 × `<int>` | → push value | Load 3-byte constant |
| 0x14 | `LDC4` | 4 × `<int>` | → push value | Load 4-byte constant |
| 0x15 | `LDC5` | 5 × `<int>` | → push value | Load 5-byte constant |
| 0x16 | `LDC6` | 6 × `<int>` | → push value | Load 6-byte constant |
| 0x17 | `LDC7` | 7 × `<int>` | → push value | Load 7-byte constant |
| 0x18 | `LDC8` | 8 × `<int>` | → push value | Load 8-byte constant |

Operand bytes are concatenated in big-endian order (most significant byte first) and interpreted as a signed two's complement integer. For example, `LDC2 0xFF 0xFE` pushes −2.

### Stack Manipulation

| Code | Opcode | Operands | Stack | Description |
| ------ | -------- | ---------- | ------- | ------------- |
| 0x1A | `PUSH` | `<int>` | → push value | Push literal onto stack |
| 0x1B | `POP` | — | pop | Discard top of stack |
| 0x1C | `SWAP` | — | pop b, a → push a, b | Swap top two elements |
| 0x1D | `DUPL` | — | peek → push copy | Duplicate top of stack |
| 0x1E | `SCLR` | — | clear all | Empty entire stack |

### Arithmetic

| Code | Opcode | Operands | Stack | Description |
| ------ | -------- | ---------- | ------- | ------------- |
| 0x20 | `ADD` | — | pop b, a → push a+b | Add |
| 0x21 | `SUB` | — | pop b, a → push a−b | Subtract |
| 0x22 | `MUL` | — | pop b, a → push a×b | Multiply |
| 0x23 | `DIV` | — | pop b, a → push a/b | Integer divide |
| 0x24 | `MOD` | — | pop b, a → push a%b | Modulo |
| 0x25 | `SQR` | — | pop a → push a×a | Square |
| 0x26 | `ABS` | — | pop a → push \|a\| | Absolute value |
| 0x27 | `NEG` | — | pop a → push −a | Negate |
| 0x28 | `MIN` | — | pop b, a → push min(a,b) | Minimum |
| 0x29 | `MAX` | — | pop b, a → push max(a,b) | Maximum |
| 0x2A | `INC` | — | pop a → push a+1 | Increment |
| 0x2B | `DEC` | — | pop a → push a−1 | Decrement |

### Logical — Comparison

| Code | Opcode | Operands | Stack | Description |
| ------ | -------- | ---------- | ------- | ------------- |
| 0x30 | `EQ` | — | pop b, a → push a==b | Equal |
| 0x31 | `LT` | — | pop b, a → push a<b | Less than |
| 0x32 | `GT` | — | pop b, a → push a>b | Greater than |
| 0x33 | `LTE` | — | pop b, a → push a≤b | Less or equal |
| 0x34 | `GTE` | — | pop b, a → push a≥b | Greater or equal |

### Logical — Boolean

| Code | Opcode | Operands | Stack | Description |
| ------ | -------- | ---------- | ------- | ------------- |
| 0x36 | `NOT` | — | pop a → push !a | Logical NOT |
| 0x37 | `AND` | — | pop b, a → push a&&b | Logical AND |
| 0x38 | `OR` | — | pop b, a → push a\|\|b | Logical OR |
| 0x39 | `XOR` | — | pop b, a → push a^b | Logical XOR |

### Logical — Bitwise

| Code | Opcode | Operands | Stack | Description |
| ------ | -------- | ---------- | ------- | ------------- |
| 0x3A | `BAND` | — | pop b, a → push a&b | Bitwise AND |
| 0x3B | `BOR` | — | pop b, a → push a\|b | Bitwise OR |
| 0x3C | `BXOR` | — | pop b, a → push a⊕b | Bitwise XOR |
| 0x3D | `BNOT` | — | pop a → push ~a | Bitwise NOT |
| 0x3E | `SHL` | — | pop b, a → push a<<b | Shift left |
| 0x3F | `SHR` | — | pop b, a → push a>>b | Shift right |

### Allocators

| Code | Opcode | Operands | Stack | Description |
| ------ | -------- | ---------- | ------- | ------------- |
| 0x40 | `BQMX` | `<reg>` | pop size → reg[n] | Allocate binary model [0,1] |
| 0x41 | `SQMX` | `<reg>` | pop size → reg[n] | Allocate spin model [−1,1] |
| 0x42 | `XQMX` | `<reg>` | pop k, size → reg[n] | Allocate discrete model [0..k-1] |
| 0x43 | `BSMX` | `<reg>` | pop size → reg[n] | Allocate binary sample [0,1] |
| 0x44 | `SSMX` | `<reg>` | pop size → reg[n] | Allocate spin sample [−1,1] |
| 0x45 | `XSMX` | `<reg>` | pop k, size → reg[n] | Allocate discrete sample [0..k-1] |
| 0x4A | `VEC` | `<reg>` | → reg[n] | Create empty vec (type inferred) |
| 0x4B | `VECI` | `<reg>` | → reg[n] | Create empty vec\<int\> |
| 0x4C | `VECX` | `<reg>` | → reg[n] | Create empty vec\<xqmx\> |

### Vector Access

| Code | Opcode | Operands | Stack | Description |
| ------ | -------- | ---------- | ------- | ------------- |
| 0x50 | `VECPUSH` | `<reg>` | pop value | Append to vec (infers/validates type) |
| 0x51 | `VECGET` | `<reg>` | pop index → push value | Get vec[index] |
| 0x52 | `VECSET` | `<reg>` | pop value, index | Set vec[index] (validates type) |
| 0x53 | `VECLEN` | `<reg>` | → push len | Push vec length onto stack |

### Vector Math

| Code | Opcode | Operands | Stack | Description |
| ------ | -------- | ---------- | ------- | ------------- |
| 0x5A | `IDXGRID` | — | pop cols, col, row → push index | Convert (row, col) to flat index |
| 0x5B | `IDXTRIU` | — | pop j, i → push triu_index | Convert (i, j) to upper triangular index |

### XQMX Access

| Code | Opcode | Operands | Stack | Description |
| ------ | -------- | ---------- | ------- | ------------- |
| 0x60 | `GETLINE` | `<reg>` | pop i → push linear[i] | Get linear coefficient (0 if absent) |
| 0x61 | `SETLINE` | `<reg>` | pop value, i | Set linear[i] |
| 0x62 | `ADDLINE` | `<reg>` | pop delta, i | Add to linear[i] |
| 0x63 | `GETQUAD` | `<reg>` | pop j, i → push quad[i,j] | Get quadratic coefficient (0 if absent) |
| 0x64 | `SETQUAD` | `<reg>` | pop value, j, i | Set quadratic[i,j] |
| 0x65 | `ADDQUAD` | `<reg>` | pop delta, j, i | Add to quadratic[i,j] |

### XQMX Grid

| Code | Opcode | Operands | Stack | Description |
| ------ | -------- | ---------- | ------- | ------------- |
| 0x66 | `RESIZE` | `<reg>` | pop cols, rows | Set grid dimensions |
| 0x67 | `ROWFIND` | `<reg>` | pop value, row → push col | Find first col where row has value |
| 0x68 | `COLFIND` | `<reg>` | pop value, col → push row | Find first row where col has value |
| 0x69 | `ROWSUM` | `<reg>` | pop row → push sum | Sum all values in row |
| 0x6A | `COLSUM` | `<reg>` | pop col → push sum | Sum all values in column |

### XQMX High-Level

| Code | Opcode | Operands | Stack | Description |
| ------ | -------- | ---------- | ------- | ------------- |
| 0x70 | `ONEHOTR` | `<reg>` | pop penalty, row | Add one-hot constraint for row |
| 0x71 | `ONEHOTC` | `<reg>` | pop penalty, col | Add one-hot constraint for column |
| 0x72 | `EXCLUDE` | `<reg>` | pop penalty, j, i | Add exclusion constraint |
| 0x73 | `IMPLIES` | `<reg>` | pop penalty, j, i | Add implication constraint |
| 0x7F | `ENERGY` | `<model> <sample>` | → push energy | Compute energy of sample against model |

---

## Assembly Syntax

```assembly
# Comments start with #
# Registers: r0, r1, ... r255 (8-bit slot ID)
# Targets: .0, .1, .2 (dot-prefixed numeric)
# Hex literals: 0x0A, 0xFF

PUSH 0x00         # start = 0
PUSH 0x0A         # count = 10
RANGE             # range loop [0, 10)
  LVAL r0         # r0 = current loop value
  LOAD r0
  PUSH 0x05
  GT
  JUMPI .0        # break if r0 > 5
NEXT
TARGET .0
HALT
```

### `LDC` Sugar

The assembler accepts `LDC <value>` as syntactic sugar for the `LDC1`–`LDC8` family. The assembler parses the signed integer, selects the smallest `LDCn` opcode that fits, and encodes the value as big-endian signed two's complement byte operands. The desugared forms (`LDC1 0xFF`, `LDC2 0x01 0x00`, etc.) remain valid.

```assembly
LDC 42            # → LDC1 42
LDC -1            # → LDC1 0xFF
LDC 256           # → LDC2 0x01 0x00
LDC 2147483647    # → LDC4 0x7F 0xFF 0xFF 0xFF
```

---

## Control Flow Details

### Jump Targets

- Targets use integer IDs: `.0`, `.1`, `.2` in assembly syntax
- `TARGET .N` marks the current position as a valid jump destination with ID N
- Before execution, all TARGET positions are collected into `targets: dict[int, int]` (target_id → pc)
- `JUMP .N` / `JUMPI .N` resolve target N to pc and jump
- Both forward and backward jumps are supported
- TARGET is a no-op during execution (marker only, resolved in pre-scan)

### Loop Semantics

Two loop paradigms share a unified `LoopFrame`:

```python
@dataclass
class LoopFrame:
    target: int             # PC to jump back to
    values: list[Value]     # All loop values (plain list, not Vec)
    index: int = 0          # Current position in values
```

**RANGE (numeric iteration):**

- `RANGE`: Pop count, start from stack. Generate values `[start, start+1, ..., start+count-1]`.
- Push loop frame with target = current PC.
- If count <= 0, skip loop body entirely.

**ITER (vec iteration):**

- `ITER <reg>`: Pop end_idx, start_idx from stack. Copy `vec[start_idx:end_idx]` from register.
- Push loop frame with target = current PC.
- If start_idx >= end_idx, skip loop body entirely.
- Elements are copied for immutability (modifications don't affect source vec).

**Shared behavior:**

- `LVAL <reg>`: Copy `values[index]` to register.
- `NEXT`: Increment index. If index < len(values), jump back to target. Else pop loop frame, continue.
- Nested loops tracked via `jc.loop_stack` (LIFO).

### Stack Arithmetic Ordering

- Binary ops pop second operand first: `PUSH a; PUSH b; SUB` → `a - b`
- Top of stack is always the second operand.

---

## XQMX Mode Semantics

- **model** — `linear` and `quadratic` are Hamiltonian coefficients. Constraint opcodes (ONEHOTR, ONEHOTC, EXCLUDE, IMPLIES) only valid in model mode.
- **sample** — `linear` holds variable assignments. `quadratic` is empty/unused.

### Energy Computation

```plain
E = Σ_i linear_model[i] · linear_sample[i]
  + Σ_(i<j) quadratic_model[i,j] · linear_sample[i] · linear_sample[j]
```
