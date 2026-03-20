"""
Core XQVM components: types, state, opcodes, executor, and errors.
"""

from .errors import (
    XQVMError,
    StackUnderflow,
    StackOverflow,
    TypeMismatch,
    RegisterNotFound,
    InvalidOpcode,
    DivisionByZero,
    TargetNotFound,
    LoopError,
    XQMXModeError,
)
from .vector import Vec, VecElem
from .xqmx import (
    XQMX,
    XQMXMode,
    XQMXDomain,
    # Grid operations
    row_indices,
    col_indices,
    row_sum,
    col_sum,
    row_find,
    col_find,
    # Mode validators
    require_model_mode,
    require_sample_mode,
    # High-level functions
    expand_onehot,
    expand_exclude,
    expand_implies,
    compute_energy,
    # Index helpers
    triu,
)
from .opcodes import Opcode, OpcodeMeta, OperandType
from .state import MachineState, JumpControl, Value
from .program import Instruction, Program, make_program, run_program
from .executor import Executor

__all__ = [
    # Errors
    "XQVMError",
    "StackUnderflow",
    "StackOverflow",
    "TypeMismatch",
    "RegisterNotFound",
    "InvalidOpcode",
    "DivisionByZero",
    "TargetNotFound",
    "LoopError",
    "XQMXModeError",
    # Vector types
    "Vec",
    "VecElem",
    # XQMX types
    "XQMX",
    "XQMXMode",
    "XQMXDomain",
    # XQMX grid operations
    "row_indices",
    "col_indices",
    "row_sum",
    "col_sum",
    "row_find",
    "col_find",
    # XQMX mode validators
    "require_model_mode",
    "require_sample_mode",
    # XQMX high-level functions
    "expand_onehot",
    "expand_exclude",
    "expand_implies",
    "compute_energy",
    # Index helpers
    "triu",
    # Opcodes
    "Opcode",
    "OpcodeMeta",
    "OperandType",
    # State
    "MachineState",
    "JumpControl",
    "Value",
    # Program
    "Instruction",
    "Program",
    "make_program",
    "run_program",
    # Executor
    "Executor",
]
