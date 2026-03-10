"""
XQVM Assembled Program: High-level assembly entry point.

Combines parsing, validation, and program construction into
a single `assemble()` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xqvm.core.program import Instruction, Program

from .parser import parse
from .validator import validate

@dataclass
class AssembledProgram:
    """
    A program produced by the assembler.

    Wraps the core Program with assembly-level metadata.
    """
    program: Program
    source_lines: int = 0
    name: str = ""

    def __len__(self) -> int:
        return len(self.program)

    def __getitem__(self, index: int) -> Instruction:
        return self.program[index]

def assemble(source: str, name: str = "") -> AssembledProgram:
    """
    Assemble source text into an executable program.

    Parses the source, validates the instruction list, and wraps
    the result in an AssembledProgram.
    """
    instructions = parse(source)
    validate(instructions)

    source_lines = len(source.splitlines())
    program = Program(instructions, name)

    return AssembledProgram(
        program=program,
        source_lines=source_lines,
        name=name,
    )
