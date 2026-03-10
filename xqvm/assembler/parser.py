"""
XQVM Assembly Parser: Tokenize and parse .xqasm text into instructions.

Two-pass design:
  Pass 1 — Tokenize lines into (opcode_name, operand_tokens, line_number) tuples.
  Pass 2 — Resolve opcode names and parse operand tokens into Instruction objects.
"""

from __future__ import annotations

from xqvm.core.opcodes import Opcode, OperandType
from xqvm.core.program import Instruction

class ParseError(Exception):
    """ Raised when assembly source contains invalid syntax. """

    def __init__(self, message: str, line: int = 0):
        self.line = line

        prefix = f"Line {line}: " if line else ""
        super().__init__(f"{prefix}{message}")

def _strip_comment(text: str) -> str:
    """ Remove inline comment from a line. """
    idx = text.find("#")
    if idx == -1:
        return text
    return text[:idx]

def _parse_operand(token: str, expected_type: OperandType, line: int) -> int:
    """ Parse a single operand token into an integer value. """
    if expected_type == OperandType.REGISTER:
        if not token.startswith("r"):
            raise ParseError(f"Expected register (rN), got '{token}'", line)

        try:
            slot = int(token[1:])
        except ValueError:
            raise ParseError(f"Invalid register '{token}'", line)

        if not (0 <= slot <= 255):
            raise ParseError(f"Register out of range: {token} (must be r0-r255)", line)
        return slot

    if expected_type == OperandType.TARGET:
        if not token.startswith("."):
            raise ParseError(f"Expected target (.N), got '{token}'", line)

        try:
            return int(token[1:])
        except ValueError:
            raise ParseError(f"Invalid target '{token}'", line)

    # IMMEDIATE
    try:
        if token.startswith("0x") or token.startswith("0X"):
            return int(token, 16)
        if token.startswith("-0x") or token.startswith("-0X"):
            return -int(token[1:], 16)
        return int(token)
    except ValueError:
        raise ParseError(f"Invalid integer literal '{token}'", line)

def _tokenize(source: str) -> list[tuple[str, list[str], int]]:
    """
    Pass 1: Tokenize source into (opcode_name, operand_tokens, line_number) tuples.

    Blank lines and comment-only lines are skipped.
    """
    tokens: list[tuple[str, list[str], int]] = []

    for line_num, raw_line in enumerate(source.splitlines(), start=1):
        stripped = _strip_comment(raw_line).strip()
        if not stripped:
            continue

        parts = stripped.split()
        opcode_name = parts[0]
        operand_tokens = parts[1:]
        tokens.append((opcode_name, operand_tokens, line_num))

    return tokens

def parse(source: str) -> list[Instruction]:
    """
    Parse assembly source text into a list of Instructions.

    Resolves opcode names, validates operand types and counts,
    and returns a flat instruction list ready for execution.
    """
    tokens = _tokenize(source)
    instructions: list[Instruction] = []

    for opcode_name, operand_tokens, line_num in tokens:
        opcode = Opcode.from_name(opcode_name)
        if opcode is None:
            raise ParseError(f"Unknown opcode '{opcode_name}'", line_num)

        meta = opcode.meta

        if len(operand_tokens) != meta.operand_count:
            raise ParseError(
                f"{opcode.name} expects {meta.operand_count} operand(s), got {len(operand_tokens)}",
                line_num,
            )

        operands = tuple(
            _parse_operand(tok, typ, line_num)
            for tok, typ in zip(operand_tokens, meta.operand_types)
        )

        instructions.append(Instruction(opcode, operands, line_num))

    return instructions
