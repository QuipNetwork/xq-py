"""
XQVM Assembly Validator: Static checks on parsed instruction lists.

Validates register ranges, operand counts, and target ID ranges
before execution.
"""

from __future__ import annotations

from xqvm.core.opcodes import OperandType
from xqvm.core.program import Instruction


class ValidationError(Exception):
    """Raised when static validation fails."""

    def __init__(self, message: str, line: int = 0):
        self.line = line

        prefix = f"Line {line}: " if line else ""
        super().__init__(f"{prefix}{message}")


def validate(instructions: list[Instruction]) -> None:
    """
    Run static validation checks on parsed instructions.

    Checks:
      - Register operands are in range 0-255
      - Target operands (JUMP/JUMPI) are in u8 range 0-255
      - Operand counts match opcode metadata

    Target label resolution and duplicate detection are handled
    by the parser. The validator only checks value ranges.

    Raises ValidationError on the first problem found.
    """
    for instr in instructions:
        meta = instr.opcode.meta

        # Operand count
        if len(instr.operands) != meta.operand_count:
            raise ValidationError(
                f"{instr.opcode.name} expects {meta.operand_count} operand(s), got {len(instr.operands)}",
                instr.line,
            )

        # Check each operand
        for val, typ in zip(instr.operands, meta.operand_types):
            if typ == OperandType.REGISTER and not (0 <= val <= 255):
                raise ValidationError(
                    f"Register out of range: r{val} (must be r0-r255)",
                    instr.line,
                )

            if typ == OperandType.TARGET and not (0 <= val <= 255):
                raise ValidationError(
                    f"Target ID out of range: {val} (must be 0-255)",
                    instr.line,
                )
