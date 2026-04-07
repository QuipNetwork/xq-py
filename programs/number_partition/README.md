# Number Partitioning

Encode, verify, and decode the Number Partitioning problem on the XQVM.

## Problem

Given a set of N integers, partition them into two subsets such that the difference between their sums is minimized.

## QUBO Formulation

- **Variables:** N binary variables (xᵢ = 1 means element i goes to subset A)
- **Objective/Constraint:** P · (Σ sᵢ · xᵢ - T/2)² where T = Σ sᵢ
- **No slack variables needed** — this is a pure equality constraint

This is the simplest application of PENALTYEQ: one opcode call replaces the entire nested-loop manual expansion.

## Two Encoder Versions

Both produce identical QUBO models (verified at coefficient level, N=4 to N=20).

| Version | Opcodes Used | Lines |
|---------|-------------|-------|
| `asm/` | Existing only (ADDLINE, ADDQUAD, nested loops) | 119 |
| `asm_penaltyeq/` | Existing + PENALTYEQ | 69 |

### Encoder Speedup

| N | asm | asm_penaltyeq | Speedup |
|---|-----|---------------|---------|
| 4 | 0.000276s | 0.000114s | 2.4x |
| 6 | 0.000484s | 0.000161s | 3.0x |
| 8 | 0.000686s | 0.000161s | 4.3x |
| 10 | 0.000989s | 0.000192s | 5.1x |
| 15 | 0.005578s | 0.000280s | 19.9x |
| 20 | 0.003471s | 0.000465s | 7.5x |
| 30 | 0.010273s | 0.000594s | 17.3x |

## Usage

```sh
# Single run
python -m programs.number_partition.runner -n 6 --no-solve

# Compare encoders
python -m programs.number_partition.runner -n 10 --no-solve --src asm
python -m programs.number_partition.runner -n 10 --no-solve --src asm_penaltyeq

# Benchmark
python -m programs.number_partition.runner --bench 4 6 8 10 15 20 --no-solve --src asm_penaltyeq
```

## Input Format

| Slot | Type | Description |
|------|------|-------------|
| 0 | int | N (element count) |
| 1 | vec\<int\> | Elements (the set S) |
| 2 | int | P (penalty weight) |

## Output Format

**Encoder:** BQMX model with N variables

**Verifier:** energy (int), valid flag (1/0), difference (|sum_A - sum_B|)

**Decoder:** subset_A (vec\<int\>), subset_B (vec\<int\>)
