"""
Core XQVM components: types, state, opcodes, executor, and errors.
"""

from .errors import (
    DivisionByZero,
    InvalidOpcode,
    LoopError,
    RegisterNotFound,
    StackOverflow,
    StackUnderflow,
    TargetNotFound,
    TypeMismatch,
    XQMXModeError,
    XQVMError,
)
from .executor import Executor
from .opcodes import Opcode, OpcodeMeta, OperandType
from .program import Instruction, Program, make_program, run_program
from .state import JumpControl, MachineState, Value
from .vector import Vec, VecElem
from .xqmx import (
    XQMX,
    XQMXDomain,
    XQMXMode,
    col_find,
    col_indices,
    col_sum,
    compute_energy,
    expand_exclude,
    expand_implies,
    # High-level functions
    expand_onehot,
    # Mode validators
    require_model_mode,
    require_sample_mode,
    row_find,
    # Grid operations
    row_indices,
    row_sum,
    # Index helpers
    triu,
)

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
