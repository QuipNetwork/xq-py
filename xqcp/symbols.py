"""
Symbolic references for the constraint programming DSL.

These types represent inputs, loop variables, models, samples,
and outputs as symbolic objects that record operations for later
compilation into assembly.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, Union

from xqvm.core import XQMXDomain

from .expression import (
    Expr, _ExprOps, Types,
    RegLoad, VecGetExpr, VecLenExpr, ColFindExpr, GetLineExpr,
    coerce, line,
)

if TYPE_CHECKING:
    from .problem import Problem

# ---------------------------------------------------------------------------
# InputRef
# ---------------------------------------------------------------------------

class InputRef(Expr, _ExprOps):
    """ Symbolic reference to a declared input. """

    def __init__(self, reg: int, name: str, type_: Types) -> None:
        self.reg = reg
        self.name = name
        self.type_ = type_

    def emit(self, lines: list[str], indent: int) -> None:
        lines.append(line(f"LOAD r{self.reg}", indent))

    def get(self, index_expr: Union[Expr, int]) -> VecGetExpr:
        """ Access an element of a Vec input. """
        if self.type_ != Types.Vec:
            raise TypeError(f"Cannot index into {self.type_.value} input '{self.name}'")
        return VecGetExpr(self.reg, coerce(index_expr))

    def veclen(self) -> VecLenExpr:
        """ Get the length of a Vec input. """
        if self.type_ != Types.Vec:
            raise TypeError(f"Cannot get length of {self.type_.value} input '{self.name}'")
        return VecLenExpr(self.reg)

# ---------------------------------------------------------------------------
# LoopVar
# ---------------------------------------------------------------------------

class LoopVar(Expr, _ExprOps):
    """ Symbolic loop variable. """

    def __init__(self, reg: int, name: str) -> None:
        self.reg = reg
        self.name = name

    def emit(self, lines: list[str], indent: int) -> None:
        lines.append(line(f"LOAD r{self.reg}", indent))

# ---------------------------------------------------------------------------
# SampleRef
# ---------------------------------------------------------------------------

class SampleRef:
    """ Symbolic reference to the sample (counterpart of the model). """

    def __init__(self, reg: int) -> None:
        self.reg = reg

    def colfind(self, col: Union[Expr, int], value: int) -> ColFindExpr:
        """ Find the row where the given column has the specified value. """
        return ColFindExpr(self.reg, coerce(col), value)

    def getline(self, index: Union[Expr, int]) -> GetLineExpr:
        """ Read a sample variable by index. """
        return GetLineExpr(self.reg, coerce(index))

# ---------------------------------------------------------------------------
# ModelRef
# ---------------------------------------------------------------------------

class ModelRef:
    """ Symbolic reference to the XQMX model. """

    def __init__(
        self, problem: Problem, reg: int, domain: XQMXDomain,
        cols_reg: int | None, is_2d: bool,
    ) -> None:
        self._problem = problem
        self.reg = reg
        self.domain = domain
        self.cols_reg = cols_reg
        self.is_2d = is_2d

    def add_linear(self, coord: Any, weight: Union[Expr, int]) -> None:
        """ Add a linear term to the model. """
        self._problem._record_add_linear(self, coord, weight)

    def add_quadratic(self, coord_a: Any, coord_b: Any, weight: Union[Expr, int]) -> None:
        """ Add a quadratic coupling term to the model. """
        self._problem._record_add_quadratic(self, coord_a, coord_b, weight)

    def apply_onehot_row(self, row: Union[Expr, int], penalty: int) -> None:
        """ Apply one-hot constraint on a row. """
        self._problem._record_onehot_row(self, row, penalty)

    def apply_onehot_col(self, col: Union[Expr, int], penalty: int) -> None:
        """ Apply one-hot constraint on a column. """
        self._problem._record_onehot_col(self, col, penalty)

# ---------------------------------------------------------------------------
# OutputRef
# ---------------------------------------------------------------------------

class OutputRef:
    """ Symbolic reference to a declared output. """

    def __init__(self, problem: Problem, slot: int, reg: int, name: str, type_: Types) -> None:
        self._problem = problem
        self.slot = slot
        self.reg = reg
        self.name = name
        self.type_ = type_

    def append(self, value_expr: Union[Expr, int]) -> None:
        """ Append a value to a Vec output (valid only for Vec type). """
        if self.type_ != Types.Vec:
            raise TypeError(f"Cannot append to {self.type_.value} output '{self.name}'")
        self._problem._record_output_append(self, value_expr)
