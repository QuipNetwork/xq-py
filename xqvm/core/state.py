"""
XQVM Virtual Machine State
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

from .vector import Vec
from .xqmx import XQMX
from .errors import StackUnderflow, StackOverflow, RegisterNotFound, LoopError

# Type alias for values that can be stored in registers
Value = Union[int, Vec, XQMX]

# Maximum stack size to prevent runaway programs
MAX_STACK_SIZE = 8192  # 2^13

@dataclass
class LoopFrame:
    """ Loop iteration frame supporting RANGE and ITER paradigms. """
    target: int             # PC to jump back to
    values: list[Value]     # All loop values (plain list, not Vec)
    index: int = 0          # Current position in values

@dataclass
class JumpControl:
    """
    Jump control state: targets and loop stack.

    Targets map target IDs to instruction indices.
    The loop stack tracks nested loop state for RANGE/ITER/NEXT operations.
    """
    targets: dict[int, int] = field(default_factory=dict)
    loop_stack: list[LoopFrame] = field(default_factory=list)

    def define_target(self, target_id: int, pc: int) -> None:
        """ Define a target at the given program counter. """
        self.targets[target_id] = pc

    def resolve_target(self, target_id: int) -> int | None:
        """ Resolve a target ID to its program counter. Returns None if not found. """
        return self.targets.get(target_id)

    def push_loop_range(self, target: int, start: int, count: int) -> None:
        """ Push a RANGE loop frame. """
        values = list(range(start, start + count))
        self.loop_stack.append(LoopFrame(target=target, values=values))

    def push_loop_iter(self, target: int, vec: Vec, start_idx: int, end_idx: int) -> None:
        """ Push an ITER loop frame. Copies elements for immutability. """
        values = [vec.get(i) for i in range(start_idx, end_idx)]
        self.loop_stack.append(LoopFrame(target=target, values=values))

    def pop_loop(self) -> LoopFrame:
        """ Pop the current loop frame. Raises LoopError if no active loop. """
        if not self.loop_stack:
            raise LoopError("No active loop")
        return self.loop_stack.pop()

    def current_loop(self) -> LoopFrame | None:
        """ Get the current loop frame without removing it. """
        return self.loop_stack[-1] if self.loop_stack else None

    def current_value(self) -> Value:
        """ Get current loop value. Raises LoopError if no active loop. """
        if not self.loop_stack:
            raise LoopError("No active loop")
        frame = self.loop_stack[-1]
        return frame.values[frame.index]

    def advance_loop(self) -> bool:
        """ Advance loop index. Returns True if loop should continue. """
        if not self.loop_stack:
            raise LoopError("No active loop to advance")

        frame = self.loop_stack[-1]
        frame.index += 1

        if frame.index >= len(frame.values):
            self.loop_stack.pop()
            return False

        return True

    @property
    def in_loop(self) -> bool:
        """ Check if currently inside a loop. """
        return len(self.loop_stack) > 0

    @property
    def loop_depth(self) -> int:
        """ Get current loop nesting depth. """
        return len(self.loop_stack)

@dataclass
class MachineState:
    """
    Complete XQVM virtual machine state.

    The state includes:
    - stack: Integer-only operand stack
    - registers: Unified register file holding int, Vec, or XQMX values
    - pc: Program counter (current instruction index)
    - jc: Jump control (targets and loop stack)
    - input: Input data provided to the program
    - output: Output data produced by the program
    - halted: Whether execution has stopped
    """
    stack: list[int] = field(default_factory=list)
    registers: dict[int, Value] = field(default_factory=dict)
    pc: int = 0
    jc: JumpControl = field(default_factory=JumpControl)
    input: dict[int, Any] = field(default_factory=dict)
    output: dict[int, Any] = field(default_factory=dict)
    halted: bool = False

    # === Stack Operations ===

    def push(self, value: int) -> None:
        """ Push an integer onto the stack. """
        if len(self.stack) >= MAX_STACK_SIZE:
            raise StackOverflow(MAX_STACK_SIZE)
        self.stack.append(value)

    def pop(self) -> int:
        """ Pop an integer from the stack. """
        if not self.stack:
            raise StackUnderflow(required=1, available=0)
        return self.stack.pop()

    def peek(self, depth: int = 0) -> int:
        """ Peek at a stack value without removing it. depth=0 is top. """
        if depth >= len(self.stack):
            raise StackUnderflow(required=depth + 1, available=len(self.stack))
        return self.stack[-(depth + 1)]

    def pop_n(self, n: int) -> list[int]:
        """ Pop n values from stack. Returns in pop order (top first). """
        if len(self.stack) < n:
            raise StackUnderflow(required=n, available=len(self.stack))

        result = []
        for _ in range(n):
            result.append(self.stack.pop())

        return result

    @property
    def stack_depth(self) -> int:
        """ Current stack depth. """
        return len(self.stack)

    # === Register Operations ===

    def get_register(self, slot: int) -> Value:
        """ Get a register value. Raises RegisterNotFound if not set. """
        if slot not in self.registers:
            raise RegisterNotFound(slot)
        return self.registers[slot]

    def set_register(self, slot: int, value: Value) -> None:
        """ Set a register value. """
        self.registers[slot] = value

    def has_register(self, slot: int) -> bool:
        """ Check if a register exists. """
        return slot in self.registers

    def clear_register(self, slot: int) -> None:
        """ Clear a register. No error if it doesn't exist. """
        self.registers.pop(slot, None)

    # === I/O Operations ===

    def get_input(self, slot: int) -> Any:
        """ Get an input slot value. Returns None if not set. """
        return self.input.get(slot)

    def set_input(self, slot: int, value: Any) -> None:
        """ Set an input slot value. """
        self.input[slot] = value

    def set_output(self, slot: int, value: Any) -> None:
        """ Set an output slot value. """
        self.output[slot] = value

    def get_output(self, slot: int) -> Any:
        """ Get an output slot value. Returns None if not set. """
        return self.output.get(slot)

    # === Control Flow ===

    def advance_pc(self) -> None:
        """ Advance program counter by one. """
        self.pc += 1

    def jump_to(self, target: int) -> None:
        """ Set program counter to target instruction index. """
        self.pc = target

    def halt(self) -> None:
        """ Stop execution. """
        self.halted = True

    # === State Management ===

    def reset(self) -> None:
        """ Reset machine state to initial conditions. """
        self.stack.clear()
        self.registers.clear()
        self.pc = 0
        self.jc = JumpControl()
        self.input.clear()
        self.output.clear()
        self.halted = False

    def snapshot(self) -> dict[str, Any]:
        """ Create a snapshot of current state for debugging. """
        return {
            "stack": list(self.stack),
            "registers": {k: repr(v) for k, v in self.registers.items()},
            "pc": self.pc,
            "targets": dict(self.jc.targets),
            "loop_depth": self.jc.loop_depth,
            "halted": self.halted,
        }

    def __repr__(self) -> str:
        return (
            f"MachineState(pc={self.pc}, stack_depth={len(self.stack)}, "
            f"registers={len(self.registers)}, halted={self.halted})"
        )
