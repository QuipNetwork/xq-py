# Bin Packing

Encode, verify, and decode the Bin Packing problem on the XQVM.

## Problem

Given N items with weights and B bins with capacity C, assign each item to exactly one bin such that no bin exceeds its capacity.

## QUBO Formulation

- **Variables:** N×B binary (x[i,b] = 1 means item i in bin b) + slack variables per bin
- **Constraint 1 (assignment):** Each item in exactly one bin — ONEHOTR per row
- **Constraint 2 (capacity):** For each bin b: Σ wᵢ · x[i,b] + slacks = C — weighted equality via PENALTYEQ

This problem uses both existing opcodes (ONEHOTR) and new opcodes (PENALTYEQ + SLACK) in the same encoder.

## Two Encoder Versions

Both produce identical QUBO models (verified at coefficient level, N=3 to N=10).

| Version | Opcodes Used | Lines |
|---------|-------------|-------|
| `asm/` | Existing + manual PENALTYEQ expansion (nested loops) | 208 |
| `asm_penaltyeq/` | Existing + PENALTYEQ + SLACK | 115 |

### Encoder Speedup

| N | B | asm | asm_penaltyeq | Speedup |
|---|---|-----|---------------|---------|
| 3 | 2 | 0.001739s | 0.000261s | 6.6x |
| 4 | 2 | 0.002050s | 0.000313s | 6.5x |
| 5 | 2 | 0.003029s | 0.000380s | 8.0x |
| 6 | 3 | 0.003776s | 0.000551s | 6.9x |
| 8 | 3 | 0.005839s | 0.000714s | 8.2x |
| 10 | 3 | 0.007269s | 0.000969s | 7.5x |

## Usage

```sh
# Single run
python -m programs.bin_packing.runner -n 5 --no-solve

# Compare encoders
python -m programs.bin_packing.runner -n 5 --no-solve --src asm
python -m programs.bin_packing.runner -n 5 --no-solve --src asm_penaltyeq

# Benchmark
python -m programs.bin_packing.runner --bench 3 4 5 6 8 10 --no-solve
```

## Input Format

| Slot | Type | Description |
|------|------|-------------|
| 0 | int | N (item count) |
| 1 | int | B (bin count) |
| 2 | vec\<int\> | Item weights |
| 3 | int | C (bin capacity) |
| 4 | int | P_oh (one-hot penalty) |
| 5 | int | P_cap (capacity penalty) |
