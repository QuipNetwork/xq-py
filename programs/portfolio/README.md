# Portfolio Optimization (Higher Moments)

Encode, solve, and decode the higher-order portfolio optimization problem on the XQVM.

## Problem

Given N assets with return statistics, select a portfolio that maximizes expected return while minimizing risk across all statistical moments:

- **Return** (1st moment): linear terms
- **Variance/Covariance** (2nd moment): quadratic terms — standard Markowitz
- **Skewness** (3rd moment): cubic terms — captures asymmetric return distributions
- **Kurtosis** (4th moment): quartic terms — captures fat tails / extreme events

## HOBO Formulation

H = -λ₁·Σμᵢxᵢ + λ₂·Σσᵢⱼxᵢxⱼ - λ₃·Σsᵢⱼₖxᵢxⱼxₖ + λ₄·Σκᵢⱼₖₗxᵢxⱼxₖxₗ

Cubic terms need 1 REDUCE each. Quartic terms need 2 chained REDUCEs.

## Two Encoder Versions

Both produce identical QUBO models (verified at coefficient level).

| Version | Opcodes Used | Lines |
|---------|-------------|-------|
| `asm/` | Existing + manual degree reduction (8 ADDQUAD/ADDLINE per quartic) | 293 |
| `asm_reduce/` | Existing + REDUCE (2 calls per quartic) | 199 |

### Encoder Speedup

| N | Skew | Kurt | asm | asm_reduce | Speedup |
|---|------|------|-----|------------|---------|
| 4 | 5 | 6 | 0.000913s | 0.000585s | 1.6x |
| 5 | 5 | 15 | 0.001669s | 0.001015s | 1.6x |
| 8 | 34 | 69 | 0.006976s | 0.004005s | 1.7x |
| 10 | 71 | 122 | 0.014695s | 0.011250s | 1.3x |
| 12 | 105 | 263 | 0.032487s | 0.015968s | 2.0x |

## Industry Relevance

Standard Markowitz portfolio optimization uses only variance (quadratic), which underestimates tail risk. Adding skewness and kurtosis produces a more realistic risk model — this is an active area of research in quantitative finance and quantum computing (see IEEE QCE 2025, arXiv:2509.01496).

## Usage

```sh
python -m programs.portfolio.runner -n 6 --no-solve --src asm_reduce
python -m programs.portfolio.runner --bench 4 5 6 8 10 --no-solve
```

## Input Format

| Slot | Type | Description |
|------|------|-------------|
| 0 | int | N (asset count) |
| 1 | vec\<int\> | Returns (μᵢ, scaled) |
| 2 | vec\<int\> | Covariance (upper triangle, flat) |
| 3 | vec\<int\> | Skewness terms (flat: [i,j,k,coeff,...]) |
| 4 | vec\<int\> | Kurtosis terms (flat: [i,j,k,l,coeff,...]) |
| 5 | int | Number of skewness terms |
| 6 | int | Number of kurtosis terms |
| 7 | int | P_aux (auxiliary constraint penalty) |

## Output Format

**Encoder:** BQMX model with N + auxiliaries variables

**Verifier:** energy (int), valid flag (1/0), portfolio return (int)

**Decoder:** vec\<int\> of selected asset indices
