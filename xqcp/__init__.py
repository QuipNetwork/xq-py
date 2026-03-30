"""
Constraint Programming DSL for XQVM.

Compiles high-level problem descriptions into three XQVM assembly
programs: encoder, verifier, and decoder.
"""

from __future__ import annotations

from .expression import (
    BinOp,
    ColFindExpr,
    Expr,
    GetLineExpr,
    Literal,
    NegExpr,
    RegLoad,
    SqrExpr,
    TriuExpr,
    Types,
    VecGetExpr,
    VecLenExpr,
    coerce,
)
from .problem import Action, CompiledPrograms, Problem
from .symbols import InputRef, LoopVar, ModelRef, OutputRef, SampleRef


def triu(i: Expr | int, j: Expr | int) -> TriuExpr:
    """Upper triangular index: compiles to IDXTRIU opcode."""
    return TriuExpr(coerce(i), coerce(j))


__all__ = [
    # DSL entry points
    "Problem",
    "Types",
    "CompiledPrograms",
    "triu",
    # Expression types (for advanced use / testing)
    "Expr",
    "Literal",
    "RegLoad",
    "BinOp",
    "SqrExpr",
    "VecGetExpr",
    "TriuExpr",
    "ColFindExpr",
    "NegExpr",
    "VecLenExpr",
    "GetLineExpr",
    # Symbolic references
    "InputRef",
    "LoopVar",
    "SampleRef",
    "ModelRef",
    "OutputRef",
    # Internals
    "Action",
]
