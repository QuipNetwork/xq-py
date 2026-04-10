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

_JUMP_OPCODES = frozenset({Opcode.JUMP1, Opcode.JUMP2})
_JUMPI_OPCODES = frozenset({Opcode.JUMPI1, Opcode.JUMPI2})


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


def _resolve_jump_target(instr: Instruction) -> int:
    """Resolve a JUMP/JUMPI instruction's target ID from its operands."""
    if instr.opcode in (Opcode.JUMP1, Opcode.JUMPI1):
        return instr.operands[0]
    return (instr.operands[0] << 8) | instr.operands[1]


def disassemble_instruction(instr: Instruction) -> str:
    """Convert a single Instruction to assembly text."""
    if instr.opcode in _PUSH_OPCODES:
        value = int.from_bytes(bytes(instr.operands), byteorder="big", signed=True)
        return f"PUSH {_format_operand(value, OperandType.IMMEDIATE)}"

    if instr.opcode in _JUMP_OPCODES:
        return f"JUMP .{_resolve_jump_target(instr)}"

    if instr.opcode in _JUMPI_OPCODES:
        return f"JUMPI .{_resolve_jump_target(instr)}"

    meta = instr.opcode.meta
    parts = [instr.opcode.name]

    for val, typ in zip(instr.operands, meta.operand_types):
        parts.append(_format_operand(val, typ))

    return " ".join(parts)


def disassemble(program: Program) -> str:
    """Convert a Program to assembly text."""
    lines: list[str] = []
    target_counter = 0

    for instr in program.instructions:
        if instr.opcode == Opcode.TARGET:
            lines.append(f"TARGET .{target_counter}")
            target_counter += 1
        else:
            lines.append(disassemble_instruction(instr))

    return "\n".join(lines)
