"""
XQVM Opcode Definitions
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

class OperandType(Enum):
    """ Types of operands that opcodes can accept. """

    IMMEDIATE = auto()  # Literal integer value
    REGISTER = auto()   # Register reference (r1-r16)
    TARGET = auto()     # Jump target ID

@dataclass(frozen=True)
class OpcodeMeta:
    """ Metadata for an opcode. """
    code: int                               # Numeric opcode value
    stack_pop: int                          # Number of values popped from stack
    stack_push: int                         # Number of values pushed to stack
    operand_count: int                      # Number of operands in assembly
    operand_types: tuple[OperandType, ...]  # Types of each operand
    description: str

class Opcode(Enum):
    """ XQVM opcodes. """

    # === Opcode Table Begins ===

    # Control flow (0x00-0x0F)
    NOP = OpcodeMeta(
        0x00, 0, 0,
        0, (),
        "No operation",
    )
    TARGET = OpcodeMeta(
        0x01, 0, 0,
        1, (OperandType.TARGET,),
        "Define jump target",
    )
    JUMP = OpcodeMeta(
        0x02, 0, 0,
        1, (OperandType.TARGET,),
        "Unconditional jump to target",
    )
    JUMPI = OpcodeMeta(
        0x03, 1, 0,
        1, (OperandType.TARGET,),
        "Jump to target if top of stack is non-zero",
    )
    NEXT = OpcodeMeta(
        0x04, 0, 0,
        0, (),
        "Advance loop index, jump back or exit",
    )
    LVAL = OpcodeMeta(
        0x05, 0, 0,
        1, (OperandType.REGISTER,),
        "Copy current loop value to register",
    )
    RANGE = OpcodeMeta(
        0x06, 2, 0,
        0, (),
        "Start range loop: pop count, start → iterate [start, start+count)",
    )
    ITER = OpcodeMeta(
        0x07, 2, 0,
        1, (OperandType.REGISTER,),
        "Start vec iteration: pop end_idx, start_idx → iterate vec[start:end]",
    )
    HALT = OpcodeMeta(
        0x0F, 0, 0,
        0, (),
        "Stop execution",
    )

    # Stack & Register I/O (0x10-0x17)
    PUSH = OpcodeMeta(
        0x10, 0, 1,
        1, (OperandType.IMMEDIATE,),
        "Push immediate value onto stack",
    )
    POP = OpcodeMeta(
        0x11, 1, 0,
        0, (),
        "Pop and discard top of stack",
    )
    DUPL = OpcodeMeta(
        0x12, 1, 2,
        0, (),
        "Duplicate top of stack",
    )
    SWAP = OpcodeMeta(
        0x13, 2, 2,
        0, (),
        "Swap top two stack values",
    )
    LOAD = OpcodeMeta(
        0x14, 0, 1,
        1, (OperandType.REGISTER,),
        "Load register value onto stack",
    )
    STOW = OpcodeMeta(
        0x15, 1, 0,
        1, (OperandType.REGISTER,),
        "Store top of stack into register",
    )
    INPUT = OpcodeMeta(
        0x16, 1, 0,
        1, (OperandType.REGISTER,),
        "Load input slot into register",
    )
    OUTPUT = OpcodeMeta(
        0x17, 1, 0,
        1, (OperandType.REGISTER,),
        "Write register to output slot",
    )

    # Arithmetic (0x20-0x25)
    ADD = OpcodeMeta(
        0x20, 2, 1,
        0, (),
        "Add: push(pop() + pop())",
    )
    SUB = OpcodeMeta(
        0x21, 2, 1,
        0, (),
        "Subtract: push(second - top)",
    )
    MUL = OpcodeMeta(
        0x22, 2, 1,
        0, (),
        "Multiply: push(pop() * pop())",
    )
    DIV = OpcodeMeta(
        0x23, 2, 1,
        0, (),
        "Integer divide: push(second / top)",
    )
    MOD = OpcodeMeta(
        0x24, 2, 1,
        0, (),
        "Modulo: push(second % top)",
    )
    NEG = OpcodeMeta(
        0x25, 1, 1,
        0, (),
        "Negate top value",
    )

    # Comparison (0x26-0x2A)
    EQ = OpcodeMeta(
        0x26, 2, 1,
        0, (),
        "Equal: push(1 if second == top else 0)",
    )
    LT = OpcodeMeta(
        0x27, 2, 1,
        0, (),
        "Less than: push(1 if second < top else 0)",
    )
    GT = OpcodeMeta(
        0x28, 2, 1,
        0, (),
        "Greater than: push(1 if second > top else 0)",
    )
    LTE = OpcodeMeta(
        0x29, 2, 1,
        0, (),
        "Less or equal: push(1 if second <= top else 0)",
    )
    GTE = OpcodeMeta(
        0x2A, 2, 1,
        0, (),
        "Greater or equal: push(1 if second >= top else 0)",
    )

    # Boolean (0x30-0x33)
    NOT = OpcodeMeta(
        0x30, 1, 1,
        0, (),
        "Logical NOT: push(1 if top == 0 else 0)",
    )
    AND = OpcodeMeta(
        0x31, 2, 1,
        0, (),
        "Logical AND: push(1 if both non-zero else 0)",
    )
    OR = OpcodeMeta(
        0x32, 2, 1,
        0, (),
        "Logical OR: push(1 if either non-zero else 0)",
    )
    XOR = OpcodeMeta(
        0x33, 2, 1,
        0, (),
        "Logical XOR: push(1 if exactly one non-zero else 0)",
    )

    # Bitwise (0x34-0x39)
    BAND = OpcodeMeta(
        0x34, 2, 1,
        0, (),
        "Bitwise AND",
    )
    BOR = OpcodeMeta(
        0x35, 2, 1,
        0, (),
        "Bitwise OR",
    )
    BXOR = OpcodeMeta(
        0x36, 2, 1,
        0, (),
        "Bitwise XOR",
    )
    BNOT = OpcodeMeta(
        0x37, 1, 1,
        0, (),
        "Bitwise NOT (complement)",
    )
    SHL = OpcodeMeta(
        0x38, 2, 1,
        0, (),
        "Shift left: push(second << top)",
    )
    SHR = OpcodeMeta(
        0x39, 2, 1,
        0, (),
        "Shift right: push(second >> top)",
    )

    # Allocators (0x40-0x4A)
    BQMX = OpcodeMeta(
        0x40, 1, 0,
        1, (OperandType.REGISTER,),
        "Create binary model XQMX",
    )
    SQMX = OpcodeMeta(
        0x41, 1, 0,
        1, (OperandType.REGISTER,),
        "Create spin model XQMX",
    )
    XQMX = OpcodeMeta(
        0x42, 2, 0,
        1, (OperandType.REGISTER,),
        "Create discrete model XQMX",
    )
    BSMX = OpcodeMeta(
        0x43, 1, 0,
        1, (OperandType.REGISTER,),
        "Create binary sample XQMX",
    )
    SSMX = OpcodeMeta(
        0x44, 1, 0,
        1, (OperandType.REGISTER,),
        "Create spin sample XQMX",
    )
    XSMX = OpcodeMeta(
        0x45, 2, 0,
        1, (OperandType.REGISTER,),
        "Create discrete sample XQMX",
    )
    VEC = OpcodeMeta(
        0x4A, 0, 0,
        1, (OperandType.REGISTER,),
        "Create empty vec (type inferred on first push)",
    )
    VECI = OpcodeMeta(
        0x4B, 0, 0,
        1, (OperandType.REGISTER,),
        "Create empty vec<int>",
    )
    VECX = OpcodeMeta(
        0x4C, 0, 0,
        1, (OperandType.REGISTER,),
        "Create empty vec<xqmx>",
    )

    # Vector Access (0x50-0x53)
    VECPUSH = OpcodeMeta(
        0x50, 1, 0,
        1, (OperandType.REGISTER,),
        "Push value onto vec (infers/validates type)",
    )
    VECGET = OpcodeMeta(
        0x51, 1, 1,
        1, (OperandType.REGISTER,),
        "Get vec[index]",
    )
    VECSET = OpcodeMeta(
        0x52, 2, 0,
        1, (OperandType.REGISTER,),
        "Set vec[index] = value (validates type)",
    )
    VECLEN = OpcodeMeta(
        0x53, 0, 1,
        1, (OperandType.REGISTER,),
        "Push vec length onto stack",
    )

    # Vector Math (0x5A-0x5B)
    IDXGRID = OpcodeMeta(
        0x5A, 3, 1,
        0, (),
        "Convert (row, col) to flat index using cols",
    )
    IDXTRIU = OpcodeMeta(
        0x5B, 2, 1,
        0, (),
        "Convert (i, j) to upper triangular index",
    )

    # XQMX Access (0x60-0x65)
    GETLINE = OpcodeMeta(
        0x60, 1, 1,
        1, (OperandType.REGISTER,),
        "Get linear coefficient",
    )
    SETLINE = OpcodeMeta(
        0x61, 2, 0,
        1, (OperandType.REGISTER,),
        "Set linear coefficient",
    )
    ADDLINE = OpcodeMeta(
        0x62, 2, 0,
        1, (OperandType.REGISTER,),
        "Add to linear coefficient",
    )
    GETQUAD = OpcodeMeta(
        0x63, 2, 1,
        1, (OperandType.REGISTER,),
        "Get quadratic coefficient",
    )
    SETQUAD = OpcodeMeta(
        0x64, 3, 0,
        1, (OperandType.REGISTER,),
        "Set quadratic coefficient",
    )
    ADDQUAD = OpcodeMeta(
        0x65, 3, 0,
        1, (OperandType.REGISTER,),
        "Add to quadratic coefficient",
    )

    # XQMX Grid (0x66-0x6A)
    RESIZE = OpcodeMeta(
        0x66, 2, 0,
        1, (OperandType.REGISTER,),
        "Set grid dimensions",
    )
    ROWFIND = OpcodeMeta(
        0x67, 2, 1,
        1, (OperandType.REGISTER,),
        "Find first col where row has value",
    )
    COLFIND = OpcodeMeta(
        0x68, 2, 1,
        1, (OperandType.REGISTER,),
        "Find first row where col has value",
    )
    ROWSUM = OpcodeMeta(
        0x69, 1, 1,
        1, (OperandType.REGISTER,),
        "Sum all values in row",
    )
    COLSUM = OpcodeMeta(
        0x6A, 1, 1,
        1, (OperandType.REGISTER,),
        "Sum all values in column",
    )

    # XQMX High Level Functions (0x70-0x7F)
    ONEHOTR = OpcodeMeta(
        0x70, 2, 0,
        1, (OperandType.REGISTER,),
        "Add one-hot constraint for row",
    )
    ONEHOTC = OpcodeMeta(
        0x71, 2, 0,
        1, (OperandType.REGISTER,),
        "Add one-hot constraint for column",
    )
    EXCLUDE = OpcodeMeta(
        0x72, 3, 0,
        1, (OperandType.REGISTER,),
        "Add exclusion constraint with penalty",
    )
    IMPLIES = OpcodeMeta(
        0x73, 3, 0,
        1, (OperandType.REGISTER,),
        "Add implication constraint with penalty",
    )
    ENERGY = OpcodeMeta(
        0x7F, 0, 1,
        2, (OperandType.REGISTER, OperandType.REGISTER),
        "Compute energy",
    )

    # === Opcode Table Ends ===

    @property
    def meta(self) -> OpcodeMeta:
        """ Get the opcode metadata. """
        return self.value

    @property
    def code(self) -> int:
        """ Get the numeric opcode value. """
        return self.value.code

    @classmethod
    def from_code(cls, code: int) -> Optional[Opcode]:
        """ Look up an opcode by its numeric code. """
        for op in cls:
            if op.code == code:
                return op
        return None

    @classmethod
    def from_name(cls, name: str) -> Optional[Opcode]:
        """ Look up an opcode by name (case-insensitive). """
        try:
            return cls[name.upper()]
        except KeyError:
            return None

# Verify we have exactly 69 opcodes
assert len(Opcode) == 69, f"Expected 69 opcodes, got {len(Opcode)}"
