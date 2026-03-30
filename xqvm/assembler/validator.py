"""
XQVM Assembly Validator: Static checks on parsed instruction lists.

Validates target references, register ranges, and operand counts
before execution.
"""

from __future__ import annotations

from xqvm.core.opcodes import Opcode, OperandType
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
      - Jump targets (JUMP/JUMPI) reference defined TARGET instructions
      - Register operands are in range 0-255
      - Operand counts match opcode metadata

    Raises ValidationError on the first problem found.
    """
    # Collect all defined target IDs
    defined_targets: set[int] = set()
    for instr in instructions:
        if instr.opcode == Opcode.TARGET:
            target_id = instr.operands[0]
            if target_id in defined_targets:
                raise ValidationError(f"Duplicate target .{target_id}", instr.line)
            defined_targets.add(target_id)

    # Validate each instruction
    for instr in instructions:
        meta = instr.opcode.meta

        # Operand count
        if len(instr.operands) != meta.operand_count:
            raise ValidationError(
                f"{instr.opcode.name} expects {meta.operand_count} operand(s), got {len(instr.operands)}",
                instr.line,
            )

        # Check each operand
        for i, (val, typ) in enumerate(zip(instr.operands, meta.operand_types)):
            if typ == OperandType.REGISTER and not (0 <= val <= 255):
                raise ValidationError(
                    f"Register out of range: r{val} (must be r0-r255)",
                    instr.line,
                )

            if typ == OperandType.TARGET and instr.opcode in (Opcode.JUMP, Opcode.JUMPI):
                if val not in defined_targets:
                    raise ValidationError(f"Undefined target .{val}", instr.line)
