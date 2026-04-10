"""
Symbolic expression system for the constraint programming DSL.

Expressions form a tree that emits XQVM assembly instructions.
Each node type knows how to append its assembly to a line buffer.
"""

from __future__ import annotations

import enum
from typing import Any

# ---------------------------------------------------------------------------
# Types enum
# ---------------------------------------------------------------------------


class Types(enum.Enum):
    """Variable types for inputs and outputs."""

    Int = "int"
    Vec = "vec"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_HEX_VALUES = {100: "0x64", 200: "0xC8"}


def fmt_int(value: int) -> str:
    """Format an integer, using hex for common penalty values."""
    return _HEX_VALUES.get(value, str(value))


def line(text: str, indent: int) -> str:
    """Format an assembly line with indentation."""
    return "  " * indent + text


def coerce(val: Any) -> Expr:
    """Convert an int or Expr-like object to an Expr node."""
    if isinstance(val, Expr):
        return val
    if isinstance(val, int):
        return Literal(val)
    raise TypeError(f"Cannot coerce {type(val).__name__} to Expr")


def emit_flat_index(
    row_expr: Expr,
    col_expr: Expr,
    cols_reg: int,
    lines: list[str],
    indent: int,
) -> None:
    """Emit IDXGRID for row * cols + col."""
    row_expr.emit(lines, indent)
    col_expr.emit(lines, indent)
    lines.append(line(f"LOAD r{cols_reg}", indent))
    lines.append(line("IDXGRID", indent))


def resolve_coord(coord: Any) -> tuple[Expr, Expr]:
    """Resolve a 2D coordinate tuple to (row_expr, col_expr)."""
    if not isinstance(coord, tuple) or len(coord) != 2:
        raise TypeError(f"Expected (row, col) tuple, got {coord!r}")
    return coerce(coord[0]), coerce(coord[1])


def expr_reg(expr: Expr) -> int | None:
    """Extract the register number from a RegLoad or InputRef, or None."""
    if isinstance(expr, RegLoad):
        return expr.reg
    return None


# ---------------------------------------------------------------------------
# Expression base and mixin
# ---------------------------------------------------------------------------


class _ExprOps:
    """Mixin providing arithmetic, bitwise, and comparison operators."""

    # -- Arithmetic --

    def __add__(self, other: Expr | int) -> BinOp:
        return BinOp("ADD", coerce(self), coerce(other))

    def __radd__(self, other: Expr | int) -> BinOp:
        return BinOp("ADD", coerce(other), coerce(self))

    def __sub__(self, other: Expr | int) -> BinOp:
        return BinOp("SUB", coerce(self), coerce(other))

    def __rsub__(self, other: Expr | int) -> BinOp:
        return BinOp("SUB", coerce(other), coerce(self))

    def __mul__(self, other: Expr | int) -> BinOp:
        return BinOp("MUL", coerce(self), coerce(other))

    def __rmul__(self, other: Expr | int) -> BinOp:
        return BinOp("MUL", coerce(other), coerce(self))

    def __mod__(self, other: Expr | int) -> BinOp:
        return BinOp("MOD", coerce(self), coerce(other))

    def __rmod__(self, other: Expr | int) -> BinOp:
        return BinOp("MOD", coerce(other), coerce(self))

    def __floordiv__(self, other: Expr | int) -> BinOp:
        return BinOp("DIV", coerce(self), coerce(other))

    def __rfloordiv__(self, other: Expr | int) -> BinOp:
        return BinOp("DIV", coerce(other), coerce(self))

    def __neg__(self) -> UnaryOp:
        return UnaryOp("NEG", coerce(self))

    # -- Bitwise --

    def __and__(self, other: Expr | int) -> BinOp:
        return BinOp("BAND", coerce(self), coerce(other))

    def __rand__(self, other: Expr | int) -> BinOp:
        return BinOp("BAND", coerce(other), coerce(self))

    def __or__(self, other: Expr | int) -> BinOp:
        return BinOp("BOR", coerce(self), coerce(other))

    def __ror__(self, other: Expr | int) -> BinOp:
        return BinOp("BOR", coerce(other), coerce(self))

    def __xor__(self, other: Expr | int) -> BinOp:
        return BinOp("BXOR", coerce(self), coerce(other))

    def __rxor__(self, other: Expr | int) -> BinOp:
        return BinOp("BXOR", coerce(other), coerce(self))

    def __invert__(self) -> UnaryOp:
        return UnaryOp("BNOT", coerce(self))

    def __lshift__(self, other: Expr | int) -> BinOp:
        return BinOp("SHL", coerce(self), coerce(other))

    def __rlshift__(self, other: Expr | int) -> BinOp:
        return BinOp("SHL", coerce(other), coerce(self))

    def __rshift__(self, other: Expr | int) -> BinOp:
        return BinOp("SHR", coerce(self), coerce(other))

    def __rrshift__(self, other: Expr | int) -> BinOp:
        return BinOp("SHR", coerce(other), coerce(self))

    # -- Comparison --

    def __eq__(self, other: Expr | int) -> CompareOp:  # type: ignore[override]
        return CompareOp("EQ", coerce(self), coerce(other))

    def __lt__(self, other: Expr | int) -> CompareOp:
        return CompareOp("LT", coerce(self), coerce(other))

    def __gt__(self, other: Expr | int) -> CompareOp:
        return CompareOp("GT", coerce(self), coerce(other))

    def __le__(self, other: Expr | int) -> CompareOp:
        return CompareOp("LTE", coerce(self), coerce(other))

    def __ge__(self, other: Expr | int) -> CompareOp:
        return CompareOp("GTE", coerce(self), coerce(other))

    def __hash__(self) -> int:
        return id(self)


class Expr:
    """Base class for symbolic expressions."""

    def emit(self, lines: list[str], indent: int) -> None:
        """Append assembly instructions for this expression to lines."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Expression node types
# ---------------------------------------------------------------------------


class Literal(Expr, _ExprOps):
    """Integer literal."""

    def __init__(self, value: int) -> None:
        self.value = value

    def emit(self, lines: list[str], indent: int) -> None:
        lines.append(line(f"PUSH {fmt_int(self.value)}", indent))


class RegLoad(Expr, _ExprOps):
    """Load from a register."""

    def __init__(self, reg: int) -> None:
        self.reg = reg

    def emit(self, lines: list[str], indent: int) -> None:
        lines.append(line(f"LOAD r{self.reg}", indent))


class BinOp(Expr, _ExprOps):
    """Binary operation (arithmetic, bitwise, logical, min/max)."""

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


class UnaryOp(Expr, _ExprOps):
    """Unary operation (NEG, ABS, BNOT, NOT)."""

    def __init__(self, op: str, inner: Expr) -> None:
        self.op = op
        self.inner = inner

    def emit(self, lines: list[str], indent: int) -> None:
        self.inner.emit(lines, indent)
        lines.append(line(self.op, indent))


class CompareOp(Expr, _ExprOps):
    """Comparison operation returning 0 or 1."""

    def __init__(self, op: str, left: Expr, right: Expr) -> None:
        self.op = op
        self.left = left
        self.right = right

    def emit(self, lines: list[str], indent: int) -> None:
        self.left.emit(lines, indent)
        self.right.emit(lines, indent)
        lines.append(line(self.op, indent))


class SqrExpr(Expr, _ExprOps):
    """Square expression: emits SQR opcode."""

    def __init__(self, inner: Expr) -> None:
        self.inner = inner

    def emit(self, lines: list[str], indent: int) -> None:
        self.inner.emit(lines, indent)
        lines.append(line("SQR", indent))


class VecGetExpr(Expr, _ExprOps):
    """Vector element access: VECGET r<vec>."""

    def __init__(self, vec_reg: int, index_expr: Expr) -> None:
        self.vec_reg = vec_reg
        self.index_expr = index_expr

    def emit(self, lines: list[str], indent: int) -> None:
        self.index_expr.emit(lines, indent)
        lines.append(line(f"VECGET r{self.vec_reg}", indent))


class GridExpr(Expr, _ExprOps):
    """Grid index: IDXGRID (row * cols + col)."""

    def __init__(self, row_expr: Expr, col_expr: Expr, cols_expr: Expr) -> None:
        self.row_expr = row_expr
        self.col_expr = col_expr
        self.cols_expr = cols_expr

    def emit(self, lines: list[str], indent: int) -> None:
        self.row_expr.emit(lines, indent)
        self.col_expr.emit(lines, indent)
        self.cols_expr.emit(lines, indent)
        lines.append(line("IDXGRID", indent))


class TriuExpr(Expr, _ExprOps):
    """Upper triangular index: IDXTRIU."""

    def __init__(self, i_expr: Expr, j_expr: Expr) -> None:
        self.i_expr = i_expr
        self.j_expr = j_expr

    def emit(self, lines: list[str], indent: int) -> None:
        self.i_expr.emit(lines, indent)
        self.j_expr.emit(lines, indent)
        lines.append(line("IDXTRIU", indent))


class ColFindExpr(Expr, _ExprOps):
    """Column find: COLFIND r<sample>."""

    def __init__(self, sample_reg: int, col_expr: Expr, value: int) -> None:
        self.sample_reg = sample_reg
        self.col_expr = col_expr
        self.value = value

    def emit(self, lines: list[str], indent: int) -> None:
        self.col_expr.emit(lines, indent)
        lines.append(line(f"PUSH {fmt_int(self.value)}", indent))
        lines.append(line(f"COLFIND r{self.sample_reg}", indent))


class RowFindExpr(Expr, _ExprOps):
    """Row find: ROWFIND r<sample>."""

    def __init__(self, sample_reg: int, row_expr: Expr, value: int) -> None:
        self.sample_reg = sample_reg
        self.row_expr = row_expr
        self.value = value

    def emit(self, lines: list[str], indent: int) -> None:
        self.row_expr.emit(lines, indent)
        lines.append(line(f"PUSH {fmt_int(self.value)}", indent))
        lines.append(line(f"ROWFIND r{self.sample_reg}", indent))


class RowSumExpr(Expr, _ExprOps):
    """Row sum: ROWSUM r<sample>."""

    def __init__(self, sample_reg: int, row_expr: Expr) -> None:
        self.sample_reg = sample_reg
        self.row_expr = row_expr

    def emit(self, lines: list[str], indent: int) -> None:
        self.row_expr.emit(lines, indent)
        lines.append(line(f"ROWSUM r{self.sample_reg}", indent))


class ColSumExpr(Expr, _ExprOps):
    """Column sum: COLSUM r<sample>."""

    def __init__(self, sample_reg: int, col_expr: Expr) -> None:
        self.sample_reg = sample_reg
        self.col_expr = col_expr

    def emit(self, lines: list[str], indent: int) -> None:
        self.col_expr.emit(lines, indent)
        lines.append(line(f"COLSUM r{self.sample_reg}", indent))


class GetLineExpr(Expr, _ExprOps):
    """Read sample variable: GETLINE r<sample>."""

    def __init__(self, sample_reg: int, index_expr: Expr) -> None:
        self.sample_reg = sample_reg
        self.index_expr = index_expr

    def emit(self, lines: list[str], indent: int) -> None:
        self.index_expr.emit(lines, indent)
        lines.append(line(f"GETLINE r{self.sample_reg}", indent))


class GetQuadExpr(Expr, _ExprOps):
    """Read quadratic coefficient: GETQUAD r<model>."""

    def __init__(self, model_reg: int, i_expr: Expr, j_expr: Expr) -> None:
        self.model_reg = model_reg
        self.i_expr = i_expr
        self.j_expr = j_expr

    def emit(self, lines: list[str], indent: int) -> None:
        self.i_expr.emit(lines, indent)
        self.j_expr.emit(lines, indent)
        lines.append(line(f"GETQUAD r{self.model_reg}", indent))


class VecLenExpr(Expr, _ExprOps):
    """Vector length: VECLEN r<vec>."""

    def __init__(self, vec_reg: int) -> None:
        self.vec_reg = vec_reg

    def emit(self, lines: list[str], indent: int) -> None:
        lines.append(line(f"VECLEN r{self.vec_reg}", indent))


# ---------------------------------------------------------------------------
# Backward compatibility — NegExpr is now UnaryOp("NEG", ...)
# ---------------------------------------------------------------------------

NegExpr = UnaryOp
