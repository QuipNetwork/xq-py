# New Opcodes Reference

> **Added:** 5 opcodes (84 → 89) · **Verified:** 115 tests across 13 files · **Status:** Implemented

---

## Table of Contents

- [Quick Reference](#quick-reference)
- [PENALTYEQ — Weighted Equality Penalty](#penaltyeq--weighted-equality-penalty)
- [SLACK — Generate Slack Variables](#slack--generate-slack-variables)
- [BITLEN — Bit Length](#bitlen--bit-length)
- [ATLEAST — At-Least-k Constraint](#atleast--at-least-k-constraint)
- [REDUCE — HOBO Degree Reduction](#reduce--hobo-degree-reduction)
- [Opcode Hierarchy](#opcode-hierarchy)
- [Decision Criteria: Opcode vs DSL Macro](#decision-criteria-opcode-vs-dsl-macro)
- [Problems Using These Opcodes](#problems-using-these-opcodes)
- [Benchmark Example: Knapsack](#benchmark-example-knapsack)
- [Test Suite](#test-suite)

---

## Quick Reference

| Opcode | Code | Category | Stack | Operands | Summary |
|--------|------|----------|-------|----------|---------|
| `PENALTYEQ` | `0x74` | Constraint | pop `target`, pop `penalty` | `rModel rIndices rCoeffs` | Expand P·(Σ aₖxₖ − b)² into QUBO terms |
| `BITLEN` | `0x75` | Utility | pop `value` → push `bitlen` | — | Push ⌊log₂(value)⌋ + 1 |
| `SLACK` | `0x76` | Encoding | pop `capacity`, pop `start_index` | `rIndices rCoeffs` | Append slack variable indices and binary coefficients |
| `ATLEAST` | `0x77` | Constraint | pop `k`, pop `penalty` | `rModel rIndices` | At least k variables must be 1 |
| `REDUCE` | `0x78` | Encoding | pop `var_a`, pop `var_b`, pop `P_aux` → push `aux_index` | `rModel` | HOBO degree reduction via auxiliary variable |

---

## `PENALTYEQ` — Weighted Equality Penalty

```
PUSH penalty
PUSH target
PENALTYEQ rModel rIndices rCoeffs
```

Expands P·(Σ aₖ·xₖ − b)² into QUBO linear and quadratic terms in a single instruction. This is the general form of equality constraints — `ONEHOTR`/`ONEHOTC` are special cases where all coefficients are 1 and the target is 1.

| | |
|---|---|
| **Code** | `0x74` |
| **Stack** | pop `target`, pop `penalty` |
| **Operands** | `rModel` (model), `rIndices` (vec\<int\> of variable indices), `rCoeffs` (vec\<int\> of coefficients) |

**Expansion:**

- Linear: `h[k] += P · aₖ · (aₖ − 2b)` for each variable k
- Quadratic: `J[k,m] += P · 2 · aₖ · aₘ` for each pair (k, m)

**Example — Knapsack capacity constraint:**
```
# Constraint: Σ wᵢ·xᵢ + Σ 2ʲ·sⱼ = W
PUSH 200           # penalty
PUSH 20            # target = capacity W
PENALTYEQ r4 r8 r9 # r4=model, r8=indices, r9=weights+slack coeffs
```

**Why not DSL-only?** Benchmarked at 5–22x faster than manual `ADDLINE`/`ADDQUAD` expansion because a single handler dispatch replaces hundreds of individual VM instructions.

---

## `SLACK` — Generate Slack Variables

```
PUSH start_index
PUSH capacity
SLACK rIndices rCoeffs
```

Appends slack variable entries to existing vecs for inequality-to-equality conversion. Generates indices `[start, start+1, ...]` and binary coefficients `[1, 2, 4, 8, ...]` for all powers of 2 up to `capacity`.

| | |
|---|---|
| **Code** | `0x76` |
| **Stack** | pop `capacity`, pop `start_index` |
| **Operands** | `rIndices` (vec\<int\> to append to), `rCoeffs` (vec\<int\> to append to) |

**Example — Knapsack with W=20:**
```
# After building item vecs with N=4 items:
LOAD r0             # N=4 (slack vars start at index 4)
LOAD r3             # W=20
SLACK r8 r9         # appends [4,5,6,7,8] and [1,2,4,8,16] to r8, r9
```

**Design note:** Appends directly into existing vecs rather than creating new ones. Earlier design created intermediate copies, which caused overhead.

---

## `BITLEN` — Bit Length

```
PUSH value
BITLEN
```

Pops value, pushes ⌊log₂(value)⌋ + 1 — the number of bits needed to represent the value.

| | |
|---|---|
| **Code** | `0x75` |
| **Stack** | pop `value` → push `bitlen` |
| **Operands** | — |

**Examples:**

| Input | Output | Binary |
|-------|--------|--------|
| 1 | 1 | `1` |
| 7 | 3 | `111` |
| 8 | 4 | `1000` |
| 20 | 5 | `10100` |
| 100 | 7 | `1100100` |

**Note:** Originally needed for pre-computing slack variable count. After `SLACK` redesign (which appends directly), `VECLEN` suffices for that purpose. Kept as a general utility.

---

## `ATLEAST` — At-Least-k Constraint

```
PUSH penalty
PUSH k
ATLEAST rModel rIndices
```

At least k variables from the given index set must be 1. Internally generates slack variables with negative coefficients and expands as a penalty equality.

| | |
|---|---|
| **Code** | `0x77` |
| **Stack** | pop `k`, pop `penalty` |
| **Operands** | `rModel` (model), `rIndices` (vec\<int\> of variable indices) |

**Example — Vertex Cover edge constraint:**
```
# At least one of endpoints i, j must be selected
VECI r5
LOAD r11            # i
VECPUSH r5
LOAD r12            # j
VECPUSH r5
PUSH 10             # penalty
PUSH 1              # k = 1
ATLEAST r4 r5
```

**Why a separate opcode?** Cannot be cleanly built from `PENALTYEQ` + `SLACK` because the slack coefficients need to be negative (−1, −2, −4, ...) to absorb the excess above k. `SLACK` produces positive coefficients. Building it from primitives would require a negation loop in assembly, defeating the purpose.

---

## `REDUCE` — HOBO Degree Reduction

```
PUSH P_aux
PUSH var_b
PUSH var_a
REDUCE rModel
```

Reduces a cubic term xₐ·xᵦ·xᵧ by introducing an auxiliary variable w constrained to represent xₐ·xᵦ. Adds auxiliary penalty terms to the model and pushes w's index onto the stack for chaining.

| | |
|---|---|
| **Code** | `0x78` |
| **Stack** | pop `var_a`, pop `var_b`, pop `P_aux` → push `aux_index` |
| **Operands** | `rModel` (model) |

**Constraint added:** P_aux·(xₐ·xᵦ − 2xₐ·w − 2xᵦ·w + 3w)

**Chaining for quartic terms:**
```
# Reduce x_a * x_b * x_c * x_d
# Step 1: w1 = reduce(x_a, x_b)
PUSH P_aux
PUSH var_b
PUSH var_a
REDUCE r4           # pushes w1

# Step 2: w2 = reduce(w1, x_c)
PUSH P_aux
PUSH var_c
SWAP                # w1 is already on stack
REDUCE r4           # pushes w2

# Now use w2 * x_d as a quadratic term
```

> **Critical:** P_aux must strictly exceed the maximum coefficient multiplying any reduced term. For example, CHROMBIN's quartic term carries coefficient 4P, requiring P_aux > 4P. Rule of thumb: P_aux = 5–10 × P.

---

## Opcode Hierarchy

```
PENALTYEQ  ← foundational (weighted equality, any target)
  ├── EXACTLY k  ← unweighted equality (aᵢ = 1, b = k)
  │     └── ONEHOT  ← already exists (EXACTLY with k = 1)
  ├── ATMOST k   ← buildable from PENALTYEQ + SLACK
  └── ATLEAST k  ← needs own opcode (negative slack coefficients)

SLACK      ← inequality → equality conversion
BITLEN     ← utility (bit count)
REDUCE     ← HOBO degree reduction (cubic/quartic → quadratic)
```

---

## Decision Criteria: Opcode vs DSL Macro

If a constraint pattern replaces O(N²) individual opcode calls → implement as a **VM opcode** (new instruction in the executor).

If it composes 2–3 existing opcodes in sequence → implement as a **DSL macro** (a Python function in the XQCP compiler that emits the existing opcodes).

**Example — PERMUTATION as DSL macro:**

TSP and QAP both need one-hot constraints on every row and every column. This is just `ONEHOTR` + `ONEHOTC` in a loop — two existing opcodes, no new VM instruction needed:

```python
# DSL macro, not a VM opcode
def apply_permutation(model, n, penalty):
    for i in range(n):
        model.apply_onehot_row(i, penalty)
        model.apply_onehot_col(i, penalty)
```

The compiler expands this into the same `ONEHOTR`/`ONEHOTC` assembly the programmer would write by hand. The VM never sees "PERMUTATION" — it only executes existing opcodes. The macro is purely a compiler-time shortcut.

By contrast, `PENALTYEQ` replaces hundreds of individual `ADDLINE`/`ADDQUAD` calls with a single handler dispatch — too many to be a macro, so it warrants a VM opcode.

**Not opcodes:**

| Pattern | Reason |
|---------|--------|
| ATMOST k | Covered by PENALTYEQ + SLACK |
| EXACTLY k | Covered by PENALTYEQ with unit coefficients |
| SYMBREAK | Problem-specific, 2–3 ADDLINE calls |
| CHROMBIN | Encoding strategy, not a constraint pattern |
| PERMUTATION | DSL macro (ONEHOTR + ONEHOTC loop) |

---

## Problems Using These Opcodes

| Problem | Type | Opcodes Used | Identical? | Speedup |
|---------|------|-------------|------------|---------|
| Knapsack | QUBO | `PENALTYEQ`, `SLACK` | ✓ | 5–22× |
| Number Partitioning | QUBO | `PENALTYEQ` | ✓ | 2.4–19.9× |
| Bin Packing | QUBO | `PENALTYEQ`, `SLACK`, `ONEHOTR` | ✓ | 6.5–8.2× |
| Set Cover | QUBO | `ATLEAST` | ✓ | ~2× |
| Max Independent Set | QUBO | `PENALTYEQ`, `SLACK` | ✓ | ~1× |
| Max-3-SAT | HOBO | `REDUCE` | ✓ | 1.1–2.0× |
| Cubic Optimization | HOBO | `REDUCE` | ✓ | 1.1–6.8× |
| Quartic Optimization | HOBO | `REDUCE` (chained) | ✓ | 1.6–7.4× |
| Portfolio Optimization | HOBO | `REDUCE` (mixed) | ✓ | 1.3–1.7× |
| Vertex Cover | QUBO | `ATLEAST` | diff. form. | ~1× |
| Graph Coloring | QUBO/HOBO | `ONEHOTR`, `EXCLUDE`, `REDUCE` | — | — |

---

## Benchmark Example: Knapsack

The 0-1 Knapsack problem demonstrates the impact of `PENALTYEQ` and `SLACK`. Three encoder versions produce identical QUBO models (verified at coefficient level), with increasing speedup as opcodes replace manual expansion.

### Encoder Comparison

| N | W | Vars | Linear | Quad | `asm` | `asm_penaltyeq` | `asm_full` | Speedup (peq) | Speedup (full) |
|---|---|------|--------|------|-------|-----------------|------------|---------------|----------------|
| 4 | 20 | 9 | 9 | 36 | 0.001840s | 0.000547s | 0.000350s | **3.4x** | **5.3x** |
| 6 | 26 | 11 | 11 | 55 | 0.002349s | 0.000490s | 0.000293s | **4.8x** | **8.0x** |
| 8 | 58 | 14 | 14 | 91 | 0.005873s | 0.000552s | 0.000335s | **10.6x** | **17.5x** |
| 10 | 82 | 17 | 17 | 136 | 0.005702s | 0.000593s | 0.000429s | **9.6x** | **13.3x** |
| 15 | 125 | 22 | 22 | 231 | 0.014487s | 0.000975s | 0.000846s | **14.9x** | **17.1x** |
| 20 | 163 | 28 | 28 | 378 | 0.017647s | 0.001620s | 0.000853s | **10.9x** | **20.7x** |

- `asm` — existing opcodes only (229 lines)
- `asm_penaltyeq` — PENALTYEQ only (135 lines)
- `asm_full` — PENALTYEQ + SLACK (94 lines)
- Line count reduction: 229 → 135 → 94 (59% total reduction)

### Solver Verification

All three encoders tested with D-Wave Neal simulated annealing (seed=42). Identical energies, selected items, and validity across all versions at every size — confirming the models are truly identical.

| N | Energy (all 3) | Selected (all 3) | Valid |
|---|---------------|-----------------|-------|
| 4 | -80074 | [0, 2, 3] | PASS |
| 6 | -135324 | [0, 2, 3, 4] | PASS |
| 8 | -672993 | [1, 2, 3, 4, 5] | PASS |
| 10 | -1344911 | [2, 3, 4, 5, 6, 8] | PASS |
| 15 | -3125248 | [1, 3, 4, 6, 9, 10, 12, 13, 14] | PASS |
| 20 | -5314127 | [0, 5, 6, 7, 10, 12, 13, 15, 18, 19] | PASS |

---

## Test Suite

115 tests across 13 files, all passing.

| File | Tests | Coverage |
|------|-------|---------|
| `test_penalty_eq.py` | 26 | PENALTYEQ, SLACK, BITLEN, Knapsack |
| `test_vertex_cover.py` | 7 | Encoder, verifier, ATLEAST |
| `test_number_partition.py` | 4 | Model equivalence, energy |
| `test_max3sat.py` | 5 | Brute-force QUBO, verifier |
| `test_set_cover.py` | 5 | Model equivalence, coverage |
| `test_bin_packing.py` | 4 | Model equivalence, verifier |
| `test_reduce.py` | 12 | Brute-force cubic/quartic/mixed |
| `test_number_partition_verifier.py` | 4 | Partition diff, energy ordering |
| `test_cubic_opt_verifier.py` | 3 | Binary validity checks |
| `test_quartic_opt_verifier.py` | 3 | Binary validity checks |
| `test_max_indep_set_verifier.py` | 7 | Independent/adjacent/empty sets |
| `test_graph_coloring_verifier.py` | 7 | Proper/improper colorings, conflicts |
| `test_portfolio_verifier.py` | 3 | Portfolio return, validity |

Tests verify correctness at four levels:

- **Coefficient-identical model comparisons:** Both encoder versions produce the same QUBO model — same size, same linear and quadratic dictionaries
- **Brute-force energy verification:** For all 2ᴺ assignments, QUBO energy with optimal auxiliaries equals the original polynomial value (REDUCE correctness)
- **Verifier logic validation:** Known-good samples produce Valid=1, known-bad samples produce Valid=0. All 11 problems have verifier tests
- **Opcode unit tests:** PENALTYEQ matches ONEHOT, SLACK produces correct powers of 2, REDUCE truth table holds for all 8 binary assignments
