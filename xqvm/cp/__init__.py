"""
Constraint Programming DSL for XQVM.

Compiles high-level problem descriptions into three XQVM assembly
programs: encoder, verifier, and decoder.
"""

from __future__ import annotations

from typing import Union

from .expression import (
    Types,
    Expr,
    Literal,
    RegLoad,
    BinOp,
    DuplMul,
    VecGetExpr,
    TriuExpr,
    ColFindExpr,
    NegExpr,
    VecLenExpr,
    GetLineExpr,
    coerce,
)
from .symbols import InputRef, LoopVar, SampleRef, ModelRef, OutputRef
from .problem import Problem, CompiledPrograms, Action

def triu(i: Union[Expr, int], j: Union[Expr, int]) -> TriuExpr:
    """ Upper triangular index: compiles to IDXTRIU opcode. """
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
    "DuplMul",
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
