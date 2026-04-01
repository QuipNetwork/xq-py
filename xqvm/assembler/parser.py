"""
XQVM Assembly Parser: Tokenize and parse .xqasm text into instructions.

Three-pass design:
  Pass 1 — Tokenize lines into (opcode_name, operand_tokens, line_number) tuples.
  Pass 2 — Resolve TARGET labels: assign sequential IDs, build label map.
  Pass 3 — Build Instruction objects, resolving JUMP/JUMPI labels via the map.
"""

from __future__ import annotations

from xqvm.core.opcodes import Opcode, OperandType
from xqvm.core.program import Instruction


class ParseError(Exception):
    """Raised when assembly source contains invalid syntax."""

    def __init__(self, message: str, line: int = 0):
        self.line = line

        prefix = f"Line {line}: " if line else ""
        super().__init__(f"{prefix}{message}")


def _strip_comment(text: str) -> str:
    """Remove inline comment from a line."""
    idx = text.find("#")
    if idx == -1:
        return text
    return text[:idx]


def _parse_operand(token: str, expected_type: OperandType, line: int) -> int:
    """Parse a single operand token into an integer value."""
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


def _push_byte_count(value: int) -> int:
    """Compute the minimum byte count to represent a signed integer."""
    for n in range(1, 9):
        lo = -(1 << (8 * n - 1))
        hi = (1 << (8 * n - 1)) - 1
        if lo <= value <= hi:
            return n
    return 8


_PUSH_MIN = -(2**63)
_PUSH_MAX = 2**63 - 1


def _expand_push(
    operand_tokens: list[str],
    line_num: int,
) -> tuple[str, list[str]]:
    """
    Expand PUSH sugar into PUSHn with byte operands.

    PUSH <value> → PUSHn <b0> <b1> ... where n is the minimum byte count
    and bytes are big-endian signed two's complement.
    """
    if len(operand_tokens) != 1:
        raise ParseError(f"PUSH expects 1 operand, got {len(operand_tokens)}", line_num)

    token = operand_tokens[0]

    try:
        if token.startswith("0x") or token.startswith("0X"):
            value = int(token, 16)
        elif token.startswith("-0x") or token.startswith("-0X"):
            value = -int(token[1:], 16)
        else:
            value = int(token)
    except ValueError:
        raise ParseError(f"Invalid integer literal '{token}'", line_num)

    if not (_PUSH_MIN <= value <= _PUSH_MAX):
        raise ParseError(
            f"PUSH value out of range (must fit in 8 bytes signed): {value}",
            line_num,
        )

    n = _push_byte_count(value)
    encoded = value.to_bytes(n, byteorder="big", signed=True)
    byte_tokens = [str(b) for b in encoded]

    return (f"PUSH{n}", byte_tokens)


def _tokenize(source: str) -> list[tuple[str, list[str], int]]:
    """
    Pass 1: Tokenize source into (opcode_name, operand_tokens, line_number) tuples.

    Blank lines and comment-only lines are skipped.
    Expands PUSH sugar into the appropriate PUSHn opcode.
    """
    tokens: list[tuple[str, list[str], int]] = []

    for line_num, raw_line in enumerate(source.splitlines(), start=1):
        stripped = _strip_comment(raw_line).strip()
        if not stripped:
            continue

        parts = stripped.split()
        opcode_name = parts[0]
        operand_tokens = parts[1:]

        if opcode_name.upper() == "PUSH":
            opcode_name, operand_tokens = _expand_push(operand_tokens, line_num)

        tokens.append((opcode_name, operand_tokens, line_num))

    return tokens


def _parse_target_label(token: str, line: int) -> int:
    """Parse a `.N` target label token into its integer value."""
    if not token.startswith("."):
        raise ParseError(f"Expected target label (.N), got '{token}'", line)

    try:
        return int(token[1:])
    except ValueError:
        raise ParseError(f"Invalid target label '{token}'", line)


def _resolve_labels(
    tokens: list[tuple[str, list[str], int]],
) -> dict[int, int]:
    """
    Pass 2: Scan TARGET lines and build a label-to-sequential-ID map.

    Each TARGET with a `.N` sugar label gets the next sequential ID (0, 1, 2, ...).
    Bare TARGET (no label) also advances the counter but has no label entry.
    Duplicate labels raise ParseError.
    """
    label_map: dict[int, int] = {}
    target_counter = 0

    for opcode_name, operand_tokens, line_num in tokens:
        if opcode_name.upper() != "TARGET":
            continue

        if len(operand_tokens) == 1:
            user_label = _parse_target_label(operand_tokens[0], line_num)
            if user_label in label_map:
                raise ParseError(f"Duplicate target label .{user_label}", line_num)
            label_map[user_label] = target_counter
        elif len(operand_tokens) > 1:
            raise ParseError(
                f"TARGET expects 0 or 1 label, got {len(operand_tokens)}",
                line_num,
            )

        target_counter += 1

    return label_map


def parse(source: str) -> list[Instruction]:
    """
    Parse assembly source text into a list of Instructions.

    Resolves opcode names, validates operand types and counts,
    resolves TARGET label sugar into sequential IDs for JUMP/JUMPI,
    and returns a flat instruction list ready for execution.
    """
    tokens = _tokenize(source)
    label_map = _resolve_labels(tokens)
    instructions: list[Instruction] = []

    for opcode_name, operand_tokens, line_num in tokens:
        opcode = Opcode.from_name(opcode_name)
        if opcode is None:
            raise ParseError(f"Unknown opcode '{opcode_name}'", line_num)

        # TARGET: consume optional label sugar, emit 0-operand instruction
        if opcode == Opcode.TARGET:
            if len(operand_tokens) > 1:
                raise ParseError(
                    f"TARGET expects 0 or 1 label, got {len(operand_tokens)}",
                    line_num,
                )
            instructions.append(Instruction(opcode, (), line_num))
            continue

        # JUMP/JUMPI: resolve label to sequential target ID
        if opcode in (Opcode.JUMP, Opcode.JUMPI):
            if len(operand_tokens) != 1:
                raise ParseError(
                    f"{opcode.name} expects 1 operand, got {len(operand_tokens)}",
                    line_num,
                )
            user_label = _parse_target_label(operand_tokens[0], line_num)
            if user_label not in label_map:
                raise ParseError(f"Undefined target .{user_label}", line_num)
            seq_id = label_map[user_label]
            instructions.append(Instruction(opcode, (seq_id,), line_num))
            continue

        # All other opcodes: standard operand parsing
        meta = opcode.meta

        if len(operand_tokens) != meta.operand_count:
            raise ParseError(
                f"{opcode.name} expects {meta.operand_count} operand(s), got {len(operand_tokens)}",
                line_num,
            )

        operands = tuple(_parse_operand(tok, typ, line_num) for tok, typ in zip(operand_tokens, meta.operand_types))

        instructions.append(Instruction(opcode, operands, line_num))

    return instructions
