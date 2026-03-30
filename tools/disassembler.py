"""
XQVM Disassembler: Convert instruction lists back to readable assembly text.
"""

from __future__ import annotations

from xqvm.core.opcodes import Opcode, OperandType
from xqvm.core.program import Instruction, Program

_PUSH_OPCODES = frozenset(
    {
        Opcode.PUSH1,
        Opcode.PUSH2,
        Opcode.PUSH3,
        Opcode.PUSH4,
        Opcode.PUSH5,
        Opcode.PUSH6,
        Opcode.PUSH7,
        Opcode.PUSH8,
    }
)


def _format_operand(value: int, typ: OperandType) -> str:
    """Format a single operand value as assembly text."""
    if typ == OperandType.REGISTER:
        return f"r{value}"
    if typ == OperandType.TARGET:
        return f".{value}"
    # IMMEDIATE — use hex for values >= 16 or negative with magnitude >= 16
    if value >= 16 or value <= -16:
        if value < 0:
            return f"-0x{abs(value):02X}"
        return f"0x{value:02X}"
    return str(value)


def disassemble_instruction(instr: Instruction) -> str:
    """Convert a single Instruction to assembly text."""
    if instr.opcode in _PUSH_OPCODES:
        value = int.from_bytes(bytes(instr.operands), byteorder="big", signed=True)
        return f"PUSH {_format_operand(value, OperandType.IMMEDIATE)}"

    meta = instr.opcode.meta
    parts = [instr.opcode.name]

    for val, typ in zip(instr.operands, meta.operand_types):
        parts.append(_format_operand(val, typ))

    return " ".join(parts)


def disassemble(program: Program) -> str:
    """Convert a Program to assembly text."""
    lines: list[str] = []

    for instr in program.instructions:
        lines.append(disassemble_instruction(instr))

    return "\n".join(lines)
