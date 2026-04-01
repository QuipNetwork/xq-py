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
            "domain": [0, 1],         # [0,1] | [-1,1] | [-k,...,k-1]
            "size": 9,
            "rows": None,
            "cols": None,
            "linear": {},             # sparse: var_index → value
            "quadratic": {}           # sparse: (i,j) → value, i < j
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
- **Domain:** `[0,1]` binary, `[-1,1]` spin, `[-k, ..., k-1]` chromatic
- **Dimension:** `size` (total linear variables), optional `rows`/`cols` for grid layout
- **Storage:** Sparse tables for `linear` and `quadratic`
- Constraint opcodes (ONEHOTR, ONEHOTC, EXCLUDE, IMPLIES) are only valid in model mode
- ENERGY computes the Hamiltonian energy of a sample against a model

---

## Instruction Set (84 opcodes)

### Notation

The following conventions are used throughout this section to describe opcode behaviour. They are documentation shorthand only and do not prescribe implementation details.

- **Stack diagrams** — `[..., a, b] → [..., r]`, rightmost = **top**. `b` is popped first.
- **`reg`** — the register operand encoded in the instruction.
- **Register effect** column uses three access modes:
  - **`read`** — register contents are inspected but not changed.
  - **`write`** — register is replaced wholesale with a new value.
  - **`mutate`** — register's existing value is modified in-place (e.g. appending to a vec, incrementing a coefficient).
- Assignments use `←` (register write) and `→` (stack push).
- **Boolean representation** — `0` is false, any non-zero value is true. Boolean-producing opcodes always push `0` or `1`.

### Control Flow

| Code | Mnemonic | Arguments | Stack effect | Register effect | Interpretation |
|------|----------|-----------|--------------|-----------------|----------------|
| `0x00` | `NOP` | — | `[...] → [...]` | — | No operation. |
| `0x01` | `TARGET` | `.N` | `[...] → [...]` | — | Mark a valid jump destination. Required at every label that `JUMP`/`JUMPI` may target. During pre-scan, registers the current PC as target N. No-op at runtime. Duplicate target IDs are allowed (last definition wins). |
| `0x02` | `JUMP` | `.N` | `[...] → [...]` | — | Set PC to the instruction at `targets[N]`. Unconditional. Error: `TargetNotFound` if N is undefined. |
| `0x03` | `JUMPI` | `.N` | `[..., cond] → [...]` | — | Pop `cond`. If `cond != 0`, set PC to `targets[N]`; otherwise fall through. Error: `TargetNotFound` if N is undefined and condition is non-zero. |
| `0x04` | `NEXT` | — | `[...] → [...]` | — | Advance the active loop frame index. If more values remain, set PC to `frame.target` (loop body start). Otherwise pop the frame and fall through. Error: `LoopError` if no loop frame is active. |
| `0x05` | `LVAL` | `reg` | `[...] → [...]` | `write` — `reg ← values[index]` | Copy the current loop value into `reg`. For RANGE loops: `reg ← int`. For ITER loops: `reg ← vec element` (type preserved: int or xqmx). Error: `LoopError` if no active loop. |
| `0x06` | `RANGE` | — | `[..., start, count] → [...]` | — | Pop `count`, then `start`. Generate values `[start, start+1, ..., start+count-1]`. Push a loop frame with `target = PC+1`. If `count <= 0`, the loop body is skipped entirely. |
| `0x07` | `ITER` | `reg` | `[..., start_idx, end_idx] → [...]` | `read` — validates `reg` holds `vec` | Pop `end_idx`, then `start_idx`. Read vec from `reg`. Copy elements `vec[start_idx:end_idx]` into a loop frame with `target = PC+1`. If `start_idx >= end_idx`, the loop body is skipped. Elements are copied for immutability. Error: `TypeMismatch` if `reg` is not a vec. |
| `0x09` | `HALT` | — | `[...] → [...]` | — | Stop execution immediately. |

---

### Register I/O

| Code | Mnemonic | Arguments | Stack effect | Register effect | Interpretation |
|------|----------|-----------|--------------|-----------------|----------------|
| `0x0A` | `LOAD` | `reg` | `[...] → [..., v]` | `read` — `reg` must hold `int` | Push the integer value from `reg` onto the stack. Error: `TypeMismatch` if `reg` holds vec or xqmx. Error: `RegisterNotFound` if `reg` is unset. |
| `0x0B` | `STOW` | `reg` | `[..., v] → [...]` | `write` — `reg ← int(v)` | Pop `v`. Write `reg ← v` as an integer. |
| `0x0C` | `DROP` | `reg` | `[...] → [...]` | `write` — `reg ← unset` | Clear the register, releasing any value it held. The register becomes unset. No error if already unset. |
| `0x0E` | `INPUT` | `reg` | `[..., s] → [...]` | `write` — `reg ← input[s]` | Pop `s` (slot index). Copy `input[s]` into `reg`. Any value type is transferable (int, vec, or xqmx). Returns `None` if slot is not set. |
| `0x0F` | `OUTPUT` | `reg` | `[..., s] → [...]` | `read` — `reg` value copied to `output[s]` | Pop `s` (slot index). Copy `reg`'s value into `output[s]`. Any value type is transferable. Error: `RegisterNotFound` if `reg` is unset. |

---

### Stack Manipulation

| Code | Mnemonic | Arguments | Stack effect | Register effect | Interpretation |
|------|----------|-----------|--------------|-----------------|----------------|
| `0x10` | `POP` | — | `[..., a] → [...]` | — | Discard the top of the stack. Error: `StackUnderflow` if empty. |
| `0x11` | `PUSH1` | `val: 1 × <int>` | `[...] → [..., v]` | — | Interpret `val` as a 1-byte big-endian signed two's complement integer, push `v`. |
| `0x12` | `PUSH2` | `val: 2 × <int>` | `[...] → [..., v]` | — | Same, 2 bytes. |
| `0x13` | `PUSH3` | `val: 3 × <int>` | `[...] → [..., v]` | — | Same, 3 bytes. |
| `0x14` | `PUSH4` | `val: 4 × <int>` | `[...] → [..., v]` | — | Same, 4 bytes. |
| `0x15` | `PUSH5` | `val: 5 × <int>` | `[...] → [..., v]` | — | Same, 5 bytes. |
| `0x16` | `PUSH6` | `val: 6 × <int>` | `[...] → [..., v]` | — | Same, 6 bytes. |
| `0x17` | `PUSH7` | `val: 7 × <int>` | `[...] → [..., v]` | — | Same, 7 bytes. |
| `0x18` | `PUSH8` | `val: 8 × <int>` | `[...] → [..., v]` | — | Same, 8 bytes (full-width signed integer). |
| `0x1A` | `SCLR` | — | `[...] → []` | — | Clear the entire value stack. Stack depth becomes 0. |
| `0x1B` | `SWAP` | — | `[..., a, b] → [..., b, a]` | — | Swap the top two elements. Error: `StackUnderflow` if depth < 2. |
| `0x1C` | `COPY` | — | `[..., a] → [..., a, a]` | — | Duplicate the top of the stack without consuming it. Error: `StackUnderflow` if empty. |

Operand bytes are concatenated in big-endian order (most significant byte first) and interpreted as a signed two's complement integer. For example, `PUSH2 0xFF 0xFE` pushes −2.

---

### Arithmetic

| Code | Mnemonic | Stack effect | Interpretation |
|------|----------|--------------|----------------|
| `0x20` | `ADD` | `[..., a, b] → [..., a+b]` | Addition. |
| `0x21` | `SUB` | `[..., a, b] → [..., a-b]` | Subtraction. |
| `0x22` | `MUL` | `[..., a, b] → [..., a*b]` | Multiplication. |
| `0x23` | `DIV` | `[..., a, b] → [..., a/b]` | Truncating integer division (rounds toward negative infinity). Error: `DivisionByZero` if `b == 0`. |
| `0x24` | `MOD` | `[..., a, b] → [..., a%b]` | Modulo (result has same sign as divisor). Error: `DivisionByZero` if `b == 0`. |
| `0x25` | `SQR` | `[..., a] → [..., a*a]` | Square. |
| `0x26` | `ABS` | `[..., a] → [..., \|a\|]` | Absolute value. |
| `0x27` | `NEG` | `[..., a] → [..., -a]` | Negation. |
| `0x28` | `MIN` | `[..., a, b] → [..., min(a,b)]` | Signed minimum. |
| `0x29` | `MAX` | `[..., a, b] → [..., max(a,b)]` | Signed maximum. |
| `0x2A` | `INC` | `[..., a] → [..., a+1]` | Increment. |
| `0x2B` | `DEC` | `[..., a] → [..., a-1]` | Decrement. |

Binary operations pop the second operand first: `PUSH a; PUSH b; SUB` → `a - b`. Top of stack is always the second operand.

---

### Comparison

Results are `1` (true) or `0` (false). All comparisons are signed.

| Code | Mnemonic | Stack effect | Interpretation |
|------|----------|--------------|----------------|
| `0x30` | `EQ` | `[..., a, b] → [..., a==b ? 1 : 0]` | Equality. |
| `0x31` | `LT` | `[..., a, b] → [..., a<b ? 1 : 0]` | Less-than. |
| `0x32` | `GT` | `[..., a, b] → [..., a>b ? 1 : 0]` | Greater-than. |
| `0x33` | `LTE` | `[..., a, b] → [..., a<=b ? 1 : 0]` | Less-or-equal. |
| `0x34` | `GTE` | `[..., a, b] → [..., a>=b ? 1 : 0]` | Greater-or-equal. |

---

### Logical Boolean

Operands are treated as booleans: `0` is false, any non-zero value is true. Results are `1` or `0`.

| Code | Mnemonic | Stack effect | Interpretation |
|------|----------|--------------|----------------|
| `0x36` | `NOT` | `[..., a] → [..., a==0 ? 1 : 0]` | Logical NOT. |
| `0x37` | `AND` | `[..., a, b] → [..., (a!=0 && b!=0) ? 1 : 0]` | Logical AND. |
| `0x38` | `OR` | `[..., a, b] → [..., (a!=0 \|\| b!=0) ? 1 : 0]` | Logical OR. |
| `0x39` | `XOR` | `[..., a, b] → [..., ((a!=0) ^ (b!=0)) ? 1 : 0]` | Logical XOR. True iff exactly one operand is non-zero. |

---

### Bitwise

Operate on raw integer bit patterns.

| Code | Mnemonic | Stack effect | Interpretation |
|------|----------|--------------|----------------|
| `0x3A` | `BAND` | `[..., a, b] → [..., a & b]` | Bitwise AND. |
| `0x3B` | `BOR` | `[..., a, b] → [..., a \| b]` | Bitwise OR. |
| `0x3C` | `BXOR` | `[..., a, b] → [..., a ^ b]` | Bitwise XOR. |
| `0x3D` | `BNOT` | `[..., a] → [..., ~a]` | Bitwise NOT (one's complement). |
| `0x3E` | `SHL` | `[..., a, b] → [..., a << b]` | Left shift. |
| `0x3F` | `SHR` | `[..., a, b] → [..., a >> b]` | Right shift (arithmetic, sign-extending). |

---

### Allocators

These instructions allocate typed objects into registers.

#### Model Allocators

| Code | Mnemonic | Arguments | Stack effect | Register effect | Interpretation |
|------|----------|-----------|--------------|-----------------|----------------|
| `0x40` | `BQMX` | `reg` | `[..., size] → [...]` | `write` — `reg ← xqmx(model, binary, size)` | Pop `size`. Create a binary `[0,1]` model XQMX with `size` variables and empty linear/quadratic tables. Write to `reg`. |
| `0x41` | `SQMX` | `reg` | `[..., size] → [...]` | `write` — `reg ← xqmx(model, spin, size)` | Pop `size`. Create a spin `[-1,+1]` model XQMX. Write to `reg`. |
| `0x42` | `XQMX` | `reg` | `[..., size, k] → [...]` | `write` — `reg ← xqmx(model, discrete(k), size)` | Pop `k`, then `size`. Create a discrete `[-k,...,k-1]` model XQMX. Error if `k < 2`. Write to `reg`. |

#### Sample Allocators

| Code | Mnemonic | Arguments | Stack effect | Register effect | Interpretation |
|------|----------|-----------|--------------|-----------------|----------------|
| `0x43` | `BSMX` | `reg` | `[..., size] → [...]` | `write` — `reg ← xqmx(sample, binary, size)` | Pop `size`. Create a binary `[0,1]` sample XQMX with `size` variables. Linear table stores variable assignments. Write to `reg`. |
| `0x44` | `SSMX` | `reg` | `[..., size] → [...]` | `write` — `reg ← xqmx(sample, spin, size)` | Pop `size`. Create a spin `[-1,+1]` sample XQMX. Write to `reg`. |
| `0x45` | `XSMX` | `reg` | `[..., size, k] → [...]` | `write` — `reg ← xqmx(sample, discrete(k), size)` | Pop `k`, then `size`. Create a discrete `[-k,...,k-1]` sample XQMX. Error if `k < 2`. Write to `reg`. |

#### Vec Allocators

| Code | Mnemonic | Arguments | Stack effect | Register effect | Interpretation |
|------|----------|-----------|--------------|-----------------|----------------|
| `0x4A` | `VEC` | `reg` | `[...] → [...]` | `write` — `reg ← vec(unset)` | Create an empty vec with unset element type (inferred on first push). Write to `reg`. |
| `0x4B` | `VECI` | `reg` | `[...] → [...]` | `write` — `reg ← vec<int>` | Create an empty `vec<int>`. Write to `reg`. |
| `0x4C` | `VECX` | `reg` | `[...] → [...]` | `write` — `reg ← vec<xqmx>` | Create an empty `vec<xqmx>`. Write to `reg`. |

---

### Vector Access

| Code | Mnemonic | Arguments | Stack effect | Register effect | Interpretation |
|------|----------|-----------|--------------|-----------------|----------------|
| `0x50` | `VECPUSH` | `reg` | `[..., v] → [...]` | `mutate` — appends `v` to `reg`'s vec | Pop `v`. `reg` must hold a vec. Append `v`. If vec type is unset, infer from `v`. Otherwise validate type compatibility. Error: `TypeMismatch` if `reg` is not a vec or if `v` has incompatible type. |
| `0x51` | `VECGET` | `reg` | `[..., idx] → [..., v]` | `read` — `reg` must hold `vec<int>` | Pop `idx`. `reg` must hold a vec. Bounds-check: `0 ≤ idx < len`. Push `vec[idx]`. Error: `IndexError` if out of bounds. Error: `TypeMismatch` if element is not int (cannot push non-int to stack). |
| `0x52` | `VECSET` | `reg` | `[..., idx, v] → [...]` | `mutate` — sets `reg.vec[idx] ← v` | Pop `v`, then `idx`. `reg` must hold a vec. Bounds-check: `0 ≤ idx < len`. Set `vec[idx] ← v`. Validates type compatibility. |
| `0x53` | `VECLEN` | `reg` | `[...] → [..., n]` | `read` — `reg` must hold a vec | `reg` must hold a vec. Push `len(vec)` as integer. Error: `TypeMismatch` if `reg` is not a vec. |

---

### Index Math

Utilities for mapping 2-D coordinates to flat array indices.

| Code | Mnemonic | Stack effect | Interpretation |
|------|----------|--------------|----------------|
| `0x5A` | `IDXGRID` | `[..., row, col, cols] → [..., row*cols+col]` | Row-major flat index. Pop `cols`, then `col`, then `row`. Push `row * cols + col`. |
| `0x5B` | `IDXTRIU` | `[..., i, j] → [..., j*(j-1)/2+i]` | Upper-triangular index for the pair `(i, j)`. Pop `j`, then `i`. If `i > j`, swap them. Push `j * (j - 1) / 2 + i`. |

---

### XQMX Coefficient Access

Read and write the linear (bias) and quadratic (coupling) coefficients of an XQMX register. Missing entries read as `0`; writes create the entry on first call. Zero values are removed from sparse storage to maintain sparsity. `reg` must hold an XQMX.

#### Linear Coefficients

| Code | Mnemonic | Arguments | Stack effect | Register effect | Interpretation |
|------|----------|-----------|--------------|-----------------|----------------|
| `0x60` | `GETLINE` | `reg` | `[..., i] → [..., linear[i]]` | `read` — `reg.xqmx.linear[i]` | Pop `i`. Push `linear[i]` (0 if absent). |
| `0x61` | `SETLINE` | `reg` | `[..., i, v] → [...]` | `mutate` — `reg.xqmx.linear[i] ← v` | Pop `v`, then `i`. Set `linear[i] ← v`. Error: `IndexError` if `i` out of range `[0, size)`. |
| `0x62` | `ADDLINE` | `reg` | `[..., i, δ] → [...]` | `mutate` — `reg.xqmx.linear[i] += δ` | Pop `δ`, then `i`. `linear[i] += δ`. Error: `IndexError` if `i` out of range. |

#### Quadratic Coefficients

| Code | Mnemonic | Arguments | Stack effect | Register effect | Interpretation |
|------|----------|-----------|--------------|-----------------|----------------|
| `0x63` | `GETQUAD` | `reg` | `[..., i, j] → [..., quad[i,j]]` | `read` — `reg.xqmx.quad[i,j]` | Pop `j`, then `i`. If `i > j`, swap. Push `quad[i,j]` (0 if absent). |
| `0x64` | `SETQUAD` | `reg` | `[..., i, j, v] → [...]` | `mutate` — `reg.xqmx.quad[i,j] ← v` | Pop `v`, then `j`, then `i`. If `i > j`, swap. Set `quad[i,j] ← v`. Error: `IndexError` if indices out of range `[0, size)`. |
| `0x65` | `ADDQUAD` | `reg` | `[..., i, j, δ] → [...]` | `mutate` — `reg.xqmx.quad[i,j] += δ` | Pop `δ`, then `j`, then `i`. If `i > j`, swap. `quad[i,j] += δ`. Error: `IndexError` if indices out of range. |

---

### XQMX Grid

A model can optionally be given 2-D grid dimensions so that variables are addressed as `(row, col)` with flat index `row * cols + col`. `reg` must hold an XQMX.

| Code | Mnemonic | Arguments | Stack effect | Register effect | Interpretation |
|------|----------|-----------|--------------|-----------------|----------------|
| `0x66` | `RESIZE` | `reg` | `[..., rows, cols] → [...]` | `mutate` — `reg.xqmx.rows ← rows; reg.xqmx.cols ← cols` | Pop `cols`, then `rows`. Set grid dimensions on the XQMX. |
| `0x67` | `ROWFIND` | `reg` | `[..., row, value] → [..., col]` | `read` — scans `reg.xqmx.linear` across row | Pop `value`, then `row`. Scan `linear[row*cols + c]` for `c` in `0..cols`. Push the column index of the first entry equal to `value`, or `-1` if not found. |
| `0x68` | `COLFIND` | `reg` | `[..., col, value] → [..., row]` | `read` — scans `reg.xqmx.linear` down column | Pop `value`, then `col`. Scan `linear[r*cols + col]` for `r` in `0..rows`. Push the row index of the first match, or `-1` if not found. |
| `0x69` | `ROWSUM` | `reg` | `[..., row] → [..., sum]` | `read` — sums `reg.xqmx.linear` across row | Pop `row`. Push `Σ linear[row*cols + c]` for `c` in `0..cols`. |
| `0x6A` | `COLSUM` | `reg` | `[..., col] → [..., sum]` | `read` — sums `reg.xqmx.linear` down column | Pop `col`. Push `Σ linear[r*cols + col]` for `r` in `0..rows`. |

---

### XQMX High-Level Functions

These instructions inject QUBO penalty terms for common combinatorial constraints, expanding into linear and quadratic coefficient deltas automatically. `reg` must hold an XQMX in MODEL mode. Error: `XQMXModeError` if the XQMX is in SAMPLE mode.

| Code | Mnemonic | Arguments | Stack effect | Register effect | Interpretation |
|------|----------|-----------|--------------|-----------------|----------------|
| `0x70` | `ONEHOTR` | `reg` | `[..., row, penalty] → [...]` | `mutate` — adds to linear and quadratic for row variables | Pop `penalty`, then `row`. Apply one-hot constraint over all variables in grid row `row`. Grid dimensions must be set. See expansion below. |
| `0x71` | `ONEHOTC` | `reg` | `[..., col, penalty] → [...]` | `mutate` — adds to linear and quadratic for column variables | Pop `penalty`, then `col`. Apply one-hot constraint over all variables in grid column `col`. Grid dimensions must be set. See expansion below. |
| `0x72` | `EXCLUDE` | `reg` | `[..., i, j, penalty] → [...]` | `mutate` — `reg.xqmx.quad[i,j] += penalty` | Pop `penalty`, then `j`, then `i`. Add mutual-exclusion penalty. See expansion below. |
| `0x73` | `IMPLIES` | `reg` | `[..., i, j, penalty] → [...]` | `mutate` — modifies linear and quadratic | Pop `penalty`, then `j`, then `i`. Add implication constraint `i → j`. See expansion below. |
| `0x7F` | `ENERGY` | `model sample` | `[...] → [..., E]` | `read` — both `model` and `sample` registers are read-only | The `model` register must hold an XQMX in MODEL mode. The `sample` register must hold an XQMX in SAMPLE mode. Sizes must match. Compute and push the Hamiltonian energy. See formula below. |

#### `ONEHOTR` / `ONEHOTC` Expansion

Apply the one-hot constraint over a set of variable indices (all variables in a row or column):

```
H += penalty × (Σ x_i - 1)²
```

Expanding for binary variables (`x² = x`):

```
linear[i]    += -penalty          for each i in indices
quad[i, j]   += 2 × penalty      for each pair i < j in indices
```

`ONEHOTR` uses indices `[row*cols, row*cols+1, ..., row*cols+cols-1]`.
`ONEHOTC` uses indices `[col, col+cols, col+2*cols, ..., col+(rows-1)*cols]`.

#### `EXCLUDE` Expansion

Penalise `x_i = 1` and `x_j = 1` simultaneously:

```
quad[i, j] += penalty
```

#### `IMPLIES` Expansion

Penalise `x_i = 1` with `x_j = 0` (implication `x_i → x_j`):

```
H += penalty × x_i × (1 - x_j) = penalty × x_i - penalty × x_i × x_j

linear[i]    += penalty
quad[i, j]   += -penalty
```

#### `ENERGY` Computation

```
E = Σ_i linear_model[i] × x_sample[i]
  + Σ_{i<j} quad_model[i,j] × x_sample[i] × x_sample[j]
```

Where `x_sample[i] = sample.linear[i]` (the variable assignment). Error: `ValueError` if `model.size != sample.size`.

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

### `PUSH` Sugar

The assembler accepts `PUSH <value>` as syntactic sugar for the `PUSH1`–`PUSH8` family. The assembler parses the signed integer, selects the smallest `PUSHn` opcode that fits, and encodes the value as big-endian signed two's complement byte operands. The desugared forms (`PUSH1 0xFF`, `PUSH2 0x01 0x00`, etc.) remain valid.

```assembly
PUSH 42            # → PUSH1 42
PUSH -1            # → PUSH1 0xFF
PUSH 256           # → PUSH2 0x01 0x00
PUSH 2147483647    # → PUSH4 0x7F 0xFF 0xFF 0xFF
```

---

## Binary Bytecode Format

A program is encoded as a raw byte stream with no header, length prefix, or alignment padding. Each instruction is one opcode byte followed by zero or more operand bytes. The total length of each instruction is determined solely by its opcode.

### Opcode Encoding

Each opcode occupies a single byte in the range `0x00`–`0x7F`. Bytes `0x80`–`0xFF` are reserved and must be rejected by the decoder.

### Operand Encoding

| Operand type | Width | Encoding |
|--------------|-------|----------|
| Register (`reg`) | 1 byte | Unsigned `u8`. Range: `0`–`255`. |
| Target (`.N`) | 2 bytes | Unsigned big-endian `u16`. Range: `0`–`65535`. |
| Immediate (`PUSHn`) | `n` bytes | Big-endian signed two's complement. Range depends on `n` (1–8 bytes). |

Special case: `ENERGY` has two register operands (model register followed by sample register), totalling 2 operand bytes.

### Instruction Length Table

| Length | Opcodes |
|--------|---------|
| 1 byte (opcode only) | `NOP`, `NEXT`, `RANGE`, `HALT`, `POP`, `SCLR`, `SWAP`, `COPY`, `ADD`, `SUB`, `MUL`, `DIV`, `MOD`, `SQR`, `ABS`, `NEG`, `MIN`, `MAX`, `INC`, `DEC`, `EQ`, `LT`, `GT`, `LTE`, `GTE`, `NOT`, `AND`, `OR`, `XOR`, `BAND`, `BOR`, `BXOR`, `BNOT`, `SHL`, `SHR`, `IDXGRID`, `IDXTRIU` |
| 2 bytes (opcode + 1 reg) | `LOAD`, `STOW`, `DROP`, `INPUT`, `OUTPUT`, `LVAL`, `ITER`, `PUSH1`, `VEC`, `VECI`, `VECX`, `BQMX`, `SQMX`, `XQMX`, `BSMX`, `SSMX`, `XSMX`, `VECPUSH`, `VECGET`, `VECSET`, `VECLEN`, `GETLINE`, `SETLINE`, `ADDLINE`, `GETQUAD`, `SETQUAD`, `ADDQUAD`, `RESIZE`, `ROWFIND`, `COLFIND`, `ROWSUM`, `COLSUM`, `ONEHOTR`, `ONEHOTC`, `EXCLUDE`, `IMPLIES` |
| 3 bytes (opcode + 2) | `TARGET` (1 + u16), `JUMP` (1 + u16), `JUMPI` (1 + u16), `PUSH2` (1 + 2 imm), `ENERGY` (1 + 2 reg) |
| 4 bytes | `PUSH3` |
| 5 bytes | `PUSH4` |
| 6 bytes | `PUSH5` |
| 7 bytes | `PUSH6` |
| 8 bytes | `PUSH7` |
| 9 bytes | `PUSH8` |

Note: `XQMX` and `XSMX` have 1 register operand (2 bytes total) but pop 2 values from stack. Their bytecode length is 2 bytes, not 3.

### Encoding Examples

```
NOP             → 0x00
PUSH1 42        → 0x11 0x2A
PUSH2 -1        → 0x12 0xFF 0xFF
LOAD r5         → 0x0A 0x05
JUMP .100       → 0x02 0x00 0x64
TARGET .3       → 0x01 0x00 0x03
ENERGY r0 r1    → 0x7F 0x00 0x01
BQMX r2         → 0x40 0x02
```

---

## Runtime Limits

| Limit | Value | Notes |
|-------|-------|-------|
| Stack depth | 8192 (2^13) | `StackOverflow` if exceeded. |
| Register slots | 256 (r0–r255) | 8-bit addressing. |
| Target IDs | 0–65535 | 16-bit in bytecode. |
| Loop nesting | Unbounded | Limited by available memory. |
| XQMX size | Implementation-defined | No spec limit. |
| Program length | Implementation-defined | No spec limit. |

---

## Reserved Opcodes

The following byte values within `0x00`–`0x7F` are unassigned and reserved for future use. A decoder encountering any of these in opcode position must reject the program:

`0x08`, `0x0D`, `0x19`, `0x1D`, `0x1E`, `0x1F`, `0x2C`, `0x2D`, `0x2E`, `0x2F`, `0x35`, `0x46`, `0x47`, `0x48`, `0x49`, `0x4D`, `0x4E`, `0x4F`, `0x54`, `0x55`, `0x56`, `0x57`, `0x58`, `0x59`, `0x5C`, `0x5D`, `0x5E`, `0x5F`, `0x6B`, `0x6C`, `0x6D`, `0x6E`, `0x6F`, `0x74`, `0x75`, `0x76`, `0x77`, `0x78`, `0x79`, `0x7A`, `0x7B`, `0x7C`, `0x7D`, `0x7E`

All byte values `0x80`–`0xFF` are likewise reserved and illegal.
