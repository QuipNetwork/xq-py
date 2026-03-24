"""
Symbolic expression system for the constraint programming DSL.

Expressions form a tree that emits XQVM assembly instructions.
Each node type knows how to append its assembly to a line buffer.
"""

from __future__ import annotations

import enum
from typing import Any, Union

# ---------------------------------------------------------------------------
# Types enum
# ---------------------------------------------------------------------------

class Types(enum.Enum):
    """ Variable types for inputs and outputs. """
    Int = "int"
    Vec = "vec"

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_HEX_VALUES = {100: "0x64", 200: "0xC8"}

def fmt_int(value: int) -> str:
    """ Format an integer, using hex for common penalty values. """
    return _HEX_VALUES.get(value, str(value))

def line(text: str, indent: int) -> str:
    """ Format an assembly line with indentation. """
    return "  " * indent + text

def coerce(val: Any) -> Expr:
    """ Convert an int or Expr-like object to an Expr node. """
    if isinstance(val, Expr):
        return val
    if isinstance(val, int):
        return Literal(val)
    raise TypeError(f"Cannot coerce {type(val).__name__} to Expr")

def emit_flat_index(
    row_expr: Expr, col_expr: Expr, cols_reg: int,
    lines: list[str], indent: int,
) -> None:
    """ Emit row * cols + col using inline arithmetic. """
    row_expr.emit(lines, indent)
    lines.append(line(f"LOAD r{cols_reg}", indent))
    lines.append(line("MUL", indent))
    col_expr.emit(lines, indent)
    lines.append(line("ADD", indent))

def resolve_coord(coord: Any) -> tuple[Expr, Expr]:
    """ Resolve a 2D coordinate tuple to (row_expr, col_expr). """
    if not isinstance(coord, tuple) or len(coord) != 2:
        raise TypeError(f"Expected (row, col) tuple, got {coord!r}")
    return coerce(coord[0]), coerce(coord[1])

def expr_reg(expr: Expr) -> int | None:
    """ Extract the register number from a RegLoad or InputRef, or None. """
    if isinstance(expr, RegLoad):
        return expr.reg
    return None

# ---------------------------------------------------------------------------
# Expression base and mixin
# ---------------------------------------------------------------------------

class _ExprOps:
    """ Mixin providing arithmetic operators that build symbolic BinOp trees. """

    def __add__(self, other: Union[Expr, int]) -> BinOp:
        return BinOp("ADD", coerce(self), coerce(other))

    def __radd__(self, other: Union[Expr, int]) -> BinOp:
        return BinOp("ADD", coerce(other), coerce(self))

    def __sub__(self, other: Union[Expr, int]) -> BinOp:
        return BinOp("SUB", coerce(self), coerce(other))

    def __rsub__(self, other: Union[Expr, int]) -> BinOp:
        return BinOp("SUB", coerce(other), coerce(self))

    def __mul__(self, other: Union[Expr, int]) -> BinOp:
        return BinOp("MUL", coerce(self), coerce(other))

    def __rmul__(self, other: Union[Expr, int]) -> BinOp:
        return BinOp("MUL", coerce(other), coerce(self))

    def __mod__(self, other: Union[Expr, int]) -> BinOp:
        return BinOp("MOD", coerce(self), coerce(other))

    def __rmod__(self, other: Union[Expr, int]) -> BinOp:
        return BinOp("MOD", coerce(other), coerce(self))

    def __floordiv__(self, other: Union[Expr, int]) -> BinOp:
        return BinOp("DIV", coerce(self), coerce(other))

    def __rfloordiv__(self, other: Union[Expr, int]) -> BinOp:
        return BinOp("DIV", coerce(other), coerce(self))

    def __neg__(self) -> NegExpr:
        return NegExpr(coerce(self))

class Expr:
    """ Base class for symbolic expressions. """

    def emit(self, lines: list[str], indent: int) -> None:
        """ Append assembly instructions for this expression to lines. """
        raise NotImplementedError

# ---------------------------------------------------------------------------
# Expression node types
# ---------------------------------------------------------------------------

class Literal(Expr, _ExprOps):
    """ Integer literal. """

    def __init__(self, value: int) -> None:
        self.value = value

    def emit(self, lines: list[str], indent: int) -> None:
        lines.append(line(f"PUSH {fmt_int(self.value)}", indent))

class RegLoad(Expr, _ExprOps):
    """ Load from a register. """

    def __init__(self, reg: int) -> None:
        self.reg = reg

    def emit(self, lines: list[str], indent: int) -> None:
        lines.append(line(f"LOAD r{self.reg}", indent))

class BinOp(Expr, _ExprOps):
    """ Binary arithmetic operation. """

    def __init__(self, op: str, left: Expr, right: Expr) -> None:
        self.op = op
        self.left = left
        self.right = right

    def emit(self, lines: list[str], indent: int) -> None:
        # Optimize x + 1 → INC, x - 1 → DEC
        if self.op == "ADD" and isinstance(self.right, Literal) and self.right.value == 1:
            self.left.emit(lines, indent)
            lines.append(line("INC", indent))
            return
        if self.op == "ADD" and isinstance(self.left, Literal) and self.left.value == 1:
            self.right.emit(lines, indent)
            lines.append(line("INC", indent))
            return
        if self.op == "SUB" and isinstance(self.right, Literal) and self.right.value == 1:
            self.left.emit(lines, indent)
            lines.append(line("DEC", indent))
            return

        self.left.emit(lines, indent)
        self.right.emit(lines, indent)
        lines.append(line(self.op, indent))

class SqrExpr(Expr, _ExprOps):
    """ Square expression: emits SQR opcode. """

    def __init__(self, inner: Expr) -> None:
        self.inner = inner

    def emit(self, lines: list[str], indent: int) -> None:
        self.inner.emit(lines, indent)
        lines.append(line("SQR", indent))

class VecGetExpr(Expr, _ExprOps):
    """ Vector element access: VECGET r<vec>. """

    def __init__(self, vec_reg: int, index_expr: Expr) -> None:
        self.vec_reg = vec_reg
        self.index_expr = index_expr

    def emit(self, lines: list[str], indent: int) -> None:
        self.index_expr.emit(lines, indent)
        lines.append(line(f"VECGET r{self.vec_reg}", indent))

class TriuExpr(Expr, _ExprOps):
    """ Upper triangular index: IDXTRIU. """

    def __init__(self, i_expr: Expr, j_expr: Expr) -> None:
        self.i_expr = i_expr
        self.j_expr = j_expr

    def emit(self, lines: list[str], indent: int) -> None:
        self.i_expr.emit(lines, indent)
        self.j_expr.emit(lines, indent)
        lines.append(line("IDXTRIU", indent))

class ColFindExpr(Expr, _ExprOps):
    """ Column find: COLFIND r<sample>. """

    def __init__(self, sample_reg: int, col_expr: Expr, value: int) -> None:
        self.sample_reg = sample_reg
        self.col_expr = col_expr
        self.value = value

    def emit(self, lines: list[str], indent: int) -> None:
        self.col_expr.emit(lines, indent)
        lines.append(line(f"PUSH {fmt_int(self.value)}", indent))
        lines.append(line(f"COLFIND r{self.sample_reg}", indent))

class NegExpr(Expr, _ExprOps):
    """ Unary negation: NEG. """

    def __init__(self, inner: Expr) -> None:
        self.inner = inner

    def emit(self, lines: list[str], indent: int) -> None:
        self.inner.emit(lines, indent)
        lines.append(line("NEG", indent))

class VecLenExpr(Expr, _ExprOps):
    """ Vector length: VECLEN r<vec>. """

    def __init__(self, vec_reg: int) -> None:
        self.vec_reg = vec_reg

    def emit(self, lines: list[str], indent: int) -> None:
        lines.append(line(f"VECLEN r{self.vec_reg}", indent))

class GetLineExpr(Expr, _ExprOps):
    """ Read sample variable: GETLINE r<sample>. """

    def __init__(self, sample_reg: int, index_expr: Expr) -> None:
        self.sample_reg = sample_reg
        self.index_expr = index_expr

    def emit(self, lines: list[str], indent: int) -> None:
        self.index_expr.emit(lines, indent)
        lines.append(line(f"GETLINE r{self.sample_reg}", indent))
