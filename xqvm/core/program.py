"""
Program and Instruction types, plus construction and execution helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .opcodes import Opcode


@dataclass(frozen=True)
class Instruction:
    """
    A single XQVM instruction.

    Attributes:
        opcode: The operation to perform
        operands: Tuple of operand values (immediates, register indices, target IDs)
        line: Source line number for debugging (0 if unknown)
    """
    opcode: Opcode
    operands: tuple[int, ...] = ()
    line: int = 0

@dataclass
class Program:
    """
    A complete XQVM program.

    Attributes:
        instructions: List of instructions to execute
        name: Optional program name for debugging
    """
    instructions: list[Instruction] = field(default_factory=list)
    name: str = ""

    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, index: int) -> Instruction:
        return self.instructions[index]


def make_program(instructions: list[Instruction]) -> Program:
    """ Build a Program from a list of Instructions. """
    return Program(instructions)


def run_program(instructions: list[Instruction], input_data: dict[int, Any] | None = None):
    """ Build and execute a program, returning executor for state inspection. """
    from .executor import Executor

    prog = make_program(instructions)
    ex = Executor()
    ex.execute(prog, input_data)
    return ex
