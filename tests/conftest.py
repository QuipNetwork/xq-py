"""
Shared pytest fixtures for XQVM test suite.
"""

import pytest

from xqvm.core.state import MachineState
from xqvm.core.xqmx import XQMX
from xqvm.core.executor import Executor
from xqvm.core.program import Instruction, Program
from xqvm.core.opcodes import Opcode

@pytest.fixture
def empty_state() -> MachineState:
    """Create a fresh MachineState with no data."""
    return MachineState()

@pytest.fixture
def binary_model() -> XQMX:
    """Create an empty binary model XQMX."""
    return XQMX.binary_model(size=10)

@pytest.fixture
def binary_sample() -> XQMX:
    """Create an empty binary sample XQMX."""
    return XQMX.binary_sample(size=10)

@pytest.fixture
def grid_model() -> XQMX:
    """Create a binary model XQMX with grid dimensions."""
    return XQMX.binary_model(size=25, rows=5, cols=5)

@pytest.fixture
def executor() -> Executor:
    """Create a fresh Executor instance."""
    return Executor()

@pytest.fixture
def simple_program() -> Program:
    """Create a simple program that pushes, adds, and stores."""
    return Program([
        Instruction(Opcode.PUSH1, (10,)),
        Instruction(Opcode.PUSH1, (5,)),
        Instruction(Opcode.ADD),
        Instruction(Opcode.STOW, (0,)),
        Instruction(Opcode.HALT),
    ])
