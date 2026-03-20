"""
XQSA — Solver backends for XQMX quadratic models.

Provides pluggable solver backends that take XQMX models and return
optimized samples. The first backend wraps DWave's neal simulated annealer.
"""

from .backend import Backend, SolverResult
from .neal import NealBackend

__all__ = [
    "Backend",
    "SolverResult",
    "NealBackend",
]
