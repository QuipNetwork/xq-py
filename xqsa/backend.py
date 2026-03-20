"""
Abstract solver backend and result types for XQMX quadratic models.

Backends implement the solve() method to find low-energy solutions
for XQMX models using different optimization strategies.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from xqvm.core.xqmx import XQMX, XQMXMode, XQMXDomain

@dataclass(frozen=True)
class SolverResult:
    """ Result from a solver backend. """
    sample: XQMX
    energy: float
    timing: float
    metadata: dict[str, Any] = field(default_factory=dict)

class Backend(ABC):
    """ Abstract solver backend for XQMX quadratic models. """

    @abstractmethod
    def solve(self, model: XQMX, **kwargs: Any) -> SolverResult:
        """ Solve a quadratic model, returning the best solution found. """
        ...

    def _validate_model(self, model: XQMX) -> None:
        """ Validate that the model is solvable. """
        if model.mode != XQMXMode.MODEL:
            raise ValueError(
                f"Expected MODEL mode, got {model.mode.name}"
            )
        if model.domain not in (XQMXDomain.BINARY, XQMXDomain.SPIN):
            raise ValueError(
                f"Unsupported domain for solving: {model.domain.name}"
            )
