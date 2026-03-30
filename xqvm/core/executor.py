"""
XQVM Executor: Fetch-Decode-Execute loop with dispatch table.

The Executor processes XQVM programs by fetching instructions,
decoding opcodes, and dispatching to handler methods.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from .errors import (
    DivisionByZero,
    InvalidOpcode,
    TargetNotFound,
    TypeMismatch,
)
from .opcodes import Opcode
from .program import Instruction, Program
from .state import MachineState
from .vector import Vec, VecElem
from .xqmx import (
    XQMX,
    col_find,
    col_indices,
    compute_energy,
    expand_exclude,
    expand_implies,
    expand_onehot,
    row_find,
    row_indices,
)
from .xqmx import (
    col_sum as xqmx_col_sum,
)
from .xqmx import (
    row_sum as xqmx_row_sum,
)


@runtime_checkable
class TracerProtocol(Protocol):
    """
    Optional tracer interface for debugging/monitoring execution.

    Implement this protocol to receive callbacks during execution.
    """

    def on_step_begin(self, executor: Executor, instr: Instruction) -> None:
        """Called before each instruction executes."""
        ...

    def on_step_end(self, executor: Executor, instr: Instruction) -> None:
        """Called after each instruction executes."""
        ...

    def on_error(self, executor: Executor, instr: Instruction, error: Exception) -> None:
        """Called when an error occurs during execution."""
        ...

    def on_halt(self, executor: Executor) -> None:
        """Called when execution halts normally."""
        ...


class Executor:
    """
    XQVM execution engine.

    The Executor maintains machine state and processes programs through
    a fetch-decode-execute loop. Each opcode is dispatched to a handler
    method via a dispatch table.

    Usage:
        executor = Executor()
        result = executor.execute(program, input_data)

    Attributes:
        state: The current machine state
        program: The program being executed (None if not executing)
        tracer: Optional tracer for debugging
    """

    def __init__(self, tracer: TracerProtocol | None = None) -> None:
        self.state = MachineState()
        self.program: Program | None = None
        self.tracer = tracer
        self._dispatch = self._build_dispatch_table()

    def _build_dispatch_table(self) -> dict[Opcode, Callable[[Instruction], None]]:
        """Build the opcode -> handler dispatch table."""
        return {
            # Control flow
            Opcode.NOP: self._runner_NOP,
            Opcode.HALT: self._runner_HALT,
            Opcode.TARGET: self._runner_TARGET,
            Opcode.JUMP: self._runner_JUMP,
            Opcode.JUMPI: self._runner_JUMPI,
            Opcode.RANGE: self._runner_RANGE,
            Opcode.ITER: self._runner_ITER,
            Opcode.NEXT: self._runner_NEXT,
            Opcode.LVAL: self._runner_LVAL,
            # Stack & Register I/O
            Opcode.PUSH1: self._runner_PUSH,
            Opcode.PUSH2: self._runner_PUSH,
            Opcode.PUSH3: self._runner_PUSH,
            Opcode.PUSH4: self._runner_PUSH,
            Opcode.PUSH5: self._runner_PUSH,
            Opcode.PUSH6: self._runner_PUSH,
            Opcode.PUSH7: self._runner_PUSH,
            Opcode.PUSH8: self._runner_PUSH,
            Opcode.POP: self._runner_POP,
            Opcode.COPY: self._runner_COPY,
            Opcode.SWAP: self._runner_SWAP,
            Opcode.SCLR: self._runner_SCLR,
            Opcode.LOAD: self._runner_LOAD,
            Opcode.STOW: self._runner_STOW,
            Opcode.DROP: self._runner_DROP,
            Opcode.INPUT: self._runner_INPUT,
            Opcode.OUTPUT: self._runner_OUTPUT,
            # Arithmetic
            Opcode.ADD: self._runner_ADD,
            Opcode.SUB: self._runner_SUB,
            Opcode.MUL: self._runner_MUL,
            Opcode.DIV: self._runner_DIV,
            Opcode.MOD: self._runner_MOD,
            Opcode.SQR: self._runner_SQR,
            Opcode.ABS: self._runner_ABS,
            Opcode.NEG: self._runner_NEG,
            Opcode.MIN: self._runner_MIN,
            Opcode.MAX: self._runner_MAX,
            Opcode.INC: self._runner_INC,
            Opcode.DEC: self._runner_DEC,
            # Comparison
            Opcode.EQ: self._runner_EQ,
            Opcode.LT: self._runner_LT,
            Opcode.GT: self._runner_GT,
            Opcode.LTE: self._runner_LTE,
            Opcode.GTE: self._runner_GTE,
            # Boolean
            Opcode.NOT: self._runner_NOT,
            Opcode.AND: self._runner_AND,
            Opcode.OR: self._runner_OR,
            Opcode.XOR: self._runner_XOR,
            # Bitwise
            Opcode.BAND: self._runner_BAND,
            Opcode.BOR: self._runner_BOR,
            Opcode.BXOR: self._runner_BXOR,
            Opcode.BNOT: self._runner_BNOT,
            Opcode.SHL: self._runner_SHL,
            Opcode.SHR: self._runner_SHR,
            # Allocators
            Opcode.VEC: self._runner_VEC,
            Opcode.VECI: self._runner_VECI,
            Opcode.VECX: self._runner_VECX,
            Opcode.BQMX: self._runner_BQMX,
            Opcode.SQMX: self._runner_SQMX,
            Opcode.XQMX: self._runner_XQMX,
            Opcode.BSMX: self._runner_BSMX,
            Opcode.SSMX: self._runner_SSMX,
            Opcode.XSMX: self._runner_XSMX,
            # Vec Access
            Opcode.VECPUSH: self._runner_VECPUSH,
            Opcode.VECGET: self._runner_VECGET,
            Opcode.VECSET: self._runner_VECSET,
            Opcode.VECLEN: self._runner_VECLEN,
            # XQMX Access
            Opcode.GETLINE: self._runner_GETLINE,
            Opcode.SETLINE: self._runner_SETLINE,
            Opcode.ADDLINE: self._runner_ADDLINE,
            Opcode.GETQUAD: self._runner_GETQUAD,
            Opcode.SETQUAD: self._runner_SETQUAD,
            Opcode.ADDQUAD: self._runner_ADDQUAD,
            # Vector Math
            Opcode.IDXGRID: self._runner_IDXGRID,
            Opcode.IDXTRIU: self._runner_IDXTRIU,
            # XQMX Grid
            Opcode.RESIZE: self._runner_RESIZE,
            Opcode.ROWFIND: self._runner_ROWFIND,
            Opcode.COLFIND: self._runner_COLFIND,
            Opcode.ROWSUM: self._runner_ROWSUM,
            Opcode.COLSUM: self._runner_COLSUM,
            # XQMX High-level
            Opcode.ONEHOTR: self._runner_ONEHOTR,
            Opcode.ONEHOTC: self._runner_ONEHOTC,
            Opcode.EXCLUDE: self._runner_EXCLUDE,
            Opcode.IMPLIES: self._runner_IMPLIES,
            Opcode.ENERGY: self._runner_ENERGY,
        }

    def execute(
        self,
        program: Program,
        input_data: dict[int, Any] | None = None,
    ) -> dict[int, Any]:
        """
        Execute a program to completion.

        Args:
            program: The program to execute
            input_data: Optional input data keyed by slot number

        Returns:
            Output data keyed by slot number

        Raises:
            XQVMError: On execution errors
        """
        self.state.reset()
        self.program = program

        if input_data:
            for slot, value in input_data.items():
                self.state.set_input(slot, value)

        # Pre-scan: collect all TARGET definitions so forward jumps resolve
        for i, instr in enumerate(program.instructions):
            if instr.opcode == Opcode.TARGET:
                self.state.jc.define_target(instr.operands[0], i)

        while not self.state.halted and self.state.pc < len(program):
            self.step()

        return dict(self.state.output)

    def step(self) -> bool:
        """
        Execute a single instruction.

        Returns:
            True if execution should continue, False if halted or at end

        Raises:
            XQVMError: On execution errors
        """
        if self.program is None:
            return False

        if self.state.halted or self.state.pc >= len(self.program):
            return False

        instr = self.program[self.state.pc]

        if self.tracer:
            self.tracer.on_step_begin(self, instr)

        try:
            handler = self._dispatch.get(instr.opcode)
            if handler is None:
                raise InvalidOpcode(instr.opcode)

            old_pc = self.state.pc
            handler(instr)

            # Advance PC only if handler didn't modify it (via jump or halt)
            if not self.state.halted and self.state.pc == old_pc:
                self.state.advance_pc()

        except Exception as e:
            if self.tracer:
                self.tracer.on_error(self, instr, e)
            raise

        if self.tracer:
            if self.state.halted:
                self.tracer.on_halt(self)
            else:
                self.tracer.on_step_end(self, instr)

        return not self.state.halted

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_register_as_int(self, slot: int) -> int:
        """Get a register value, ensuring it's an int."""
        value = self.state.get_register(slot)
        if not isinstance(value, int):
            raise TypeMismatch("int", type(value).__name__, f"register r{slot}")
        return value

    def _get_register_as_vec(self, slot: int) -> Vec:
        """Get a register value, ensuring it's a vec."""
        value = self.state.get_register(slot)
        if not isinstance(value, Vec):
            raise TypeMismatch("Vec", type(value).__name__, f"register r{slot}")
        return value

    def _get_register_as_xqmx(self, slot: int) -> XQMX:
        """Get a register value, ensuring it's an xqmx"""
        value = self.state.get_register(slot)
        if not isinstance(value, XQMX):
            raise TypeMismatch("XQMX", type(value).__name__, f"register r{slot}")
        return value

    # =========================================================================
    # Instruction Set Runners
    # =========================================================================

    def _runner_NOP(self, instr: Instruction) -> None:
        """NOP: No operation."""
        pass

    def _runner_HALT(self, instr: Instruction) -> None:
        """HALT: Stop execution."""
        self.state.halt()

    def _runner_TARGET(self, instr: Instruction) -> None:
        """TARGET: No-op (targets pre-scanned before execution)."""
        pass

    def _runner_JUMP(self, instr: Instruction) -> None:
        """JUMP: Unconditional jump to target."""
        target_id = instr.operands[0]
        pc = self.state.jc.resolve_target(target_id)
        if pc is None:
            raise TargetNotFound(target_id)
        # Jump to target, then step() will advance to next instruction
        self.state.jump_to(pc)

    def _runner_JUMPI(self, instr: Instruction) -> None:
        """JUMPI: Jump to target if top of stack is non-zero."""
        target_id = instr.operands[0]
        condition = self.state.pop()

        if condition != 0:
            pc = self.state.jc.resolve_target(target_id)
            if pc is None:
                raise TargetNotFound(target_id)
            self.state.jump_to(pc)

    def _runner_RANGE(self, instr: Instruction) -> None:
        """RANGE: Start range loop. Pop count, start -> iterate [start, start+count)."""
        count, start = self.state.pop_n(2)
        # Store current PC + 1 as the loop target (next instruction)
        self.state.jc.push_loop_range(self.state.pc + 1, start, count)

    def _runner_ITER(self, instr: Instruction) -> None:
        """ITER: Start vec iteration. Pop end_idx, start_idx -> iterate vec[start:end]."""
        reg = instr.operands[0]
        vec = self._get_register_as_vec(reg)
        end_idx, start_idx = self.state.pop_n(2)
        # Store current PC + 1 as the loop target (next instruction)
        self.state.jc.push_loop_iter(self.state.pc + 1, vec, start_idx, end_idx)

    def _runner_NEXT(self, instr: Instruction) -> None:
        """NEXT: Advance loop index, jump back if more, else pop frame."""
        frame = self.state.jc.current_loop()
        if frame is None:
            from .errors import LoopError

            raise LoopError("NEXT outside of loop")

        if self.state.jc.advance_loop():
            # More iterations: jump back to loop start
            self.state.jump_to(frame.target)
        # else: loop finished, frame popped, continue normally

    def _runner_LVAL(self, instr: Instruction) -> None:
        """LVAL: Copy current loop value to register."""
        reg = instr.operands[0]
        value = self.state.jc.current_loop_value()
        self.state.set_register(reg, value)

    def _runner_PUSH(self, instr: Instruction) -> None:
        """PUSH1-8: Push N-byte constant (big-endian signed two's complement)."""
        value = int.from_bytes(bytes(instr.operands), byteorder="big", signed=True)
        self.state.push(value)

    def _runner_POP(self, instr: Instruction) -> None:
        """POP: Pop and discard top of stack."""
        self.state.pop()

    def _runner_COPY(self, instr: Instruction) -> None:
        """COPY: Duplicate top of stack."""
        value = self.state.peek()
        self.state.push(value)

    def _runner_SWAP(self, instr: Instruction) -> None:
        """SWAP: Swap top two stack values."""
        a, b = self.state.pop_n(2)
        self.state.push(a)
        self.state.push(b)

    def _runner_SCLR(self, instr: Instruction) -> None:
        """SCLR: Clear entire stack."""
        self.state.stack.clear()

    def _runner_LOAD(self, instr: Instruction) -> None:
        """LOAD: Load register value onto stack."""
        reg = instr.operands[0]
        value = self._get_register_as_int(reg)
        self.state.push(value)

    def _runner_STOW(self, instr: Instruction) -> None:
        """STOW: Store top of stack into register."""
        reg = instr.operands[0]
        value = self.state.pop()
        self.state.set_register(reg, value)

    def _runner_DROP(self, instr: Instruction) -> None:
        """DROP: Clear register (reset to unset)."""
        reg = instr.operands[0]
        self.state.clear_register(reg)

    def _runner_INPUT(self, instr: Instruction) -> None:
        """INPUT: Load input slot into register."""
        reg = instr.operands[0]
        slot = self.state.pop()
        value = self.state.get_input(slot)
        self.state.set_register(reg, value)

    def _runner_OUTPUT(self, instr: Instruction) -> None:
        """OUTPUT: Write register to output slot."""
        reg = instr.operands[0]
        slot = self.state.pop()
        value = self.state.get_register(reg)
        self.state.set_output(slot, value)

    def _runner_ADD(self, instr: Instruction) -> None:
        """ADD: push(pop() + pop())."""
        b, a = self.state.pop_n(2)
        self.state.push(a + b)

    def _runner_SUB(self, instr: Instruction) -> None:
        """SUB: push(second - top)."""
        b, a = self.state.pop_n(2)
        self.state.push(a - b)

    def _runner_MUL(self, instr: Instruction) -> None:
        """MUL: push(pop() * pop())."""
        b, a = self.state.pop_n(2)
        self.state.push(a * b)

    def _runner_DIV(self, instr: Instruction) -> None:
        """DIV: push(second / top)."""
        b, a = self.state.pop_n(2)
        if b == 0:
            raise DivisionByZero()
        self.state.push(a // b)

    def _runner_MOD(self, instr: Instruction) -> None:
        """MOD: push(second % top)."""
        b, a = self.state.pop_n(2)
        if b == 0:
            raise DivisionByZero()
        self.state.push(a % b)

    def _runner_NEG(self, instr: Instruction) -> None:
        """NEG: Negate top value."""
        value = self.state.pop()
        self.state.push(-value)

    def _runner_SQR(self, instr: Instruction) -> None:
        """SQR: Square top value."""
        value = self.state.pop()
        self.state.push(value * value)

    def _runner_ABS(self, instr: Instruction) -> None:
        """ABS: Absolute value."""
        value = self.state.pop()
        self.state.push(abs(value))

    def _runner_MIN(self, instr: Instruction) -> None:
        """MIN: push(min(second, top))."""
        b, a = self.state.pop_n(2)
        self.state.push(min(a, b))

    def _runner_MAX(self, instr: Instruction) -> None:
        """MAX: push(max(second, top))."""
        b, a = self.state.pop_n(2)
        self.state.push(max(a, b))

    def _runner_INC(self, instr: Instruction) -> None:
        """INC: Increment top value."""
        value = self.state.pop()
        self.state.push(value + 1)

    def _runner_DEC(self, instr: Instruction) -> None:
        """DEC: Decrement top value."""
        value = self.state.pop()
        self.state.push(value - 1)

    def _runner_EQ(self, instr: Instruction) -> None:
        """EQ: push(1 if second == top else 0)."""
        b, a = self.state.pop_n(2)
        self.state.push(1 if a == b else 0)

    def _runner_LT(self, instr: Instruction) -> None:
        """LT: push(1 if second < top else 0)."""
        b, a = self.state.pop_n(2)
        self.state.push(1 if a < b else 0)

    def _runner_GT(self, instr: Instruction) -> None:
        """GT: push(1 if second > top else 0)."""
        b, a = self.state.pop_n(2)
        self.state.push(1 if a > b else 0)

    def _runner_LTE(self, instr: Instruction) -> None:
        """LTE: push(1 if second <= top else 0)."""
        b, a = self.state.pop_n(2)
        self.state.push(1 if a <= b else 0)

    def _runner_GTE(self, instr: Instruction) -> None:
        """GTE: push(1 if second >= top else 0)."""
        b, a = self.state.pop_n(2)
        self.state.push(1 if a >= b else 0)

    def _runner_NOT(self, instr: Instruction) -> None:
        """NOT: push(1 if top == 0 else 0)."""
        value = self.state.pop()
        self.state.push(1 if value == 0 else 0)

    def _runner_AND(self, instr: Instruction) -> None:
        """AND: push(1 if both non-zero else 0)."""
        b, a = self.state.pop_n(2)
        self.state.push(1 if (a != 0 and b != 0) else 0)

    def _runner_OR(self, instr: Instruction) -> None:
        """OR: push(1 if either non-zero else 0)."""
        b, a = self.state.pop_n(2)
        self.state.push(1 if (a != 0 or b != 0) else 0)

    def _runner_XOR(self, instr: Instruction) -> None:
        """XOR: push(1 if exactly one non-zero else 0)."""
        b, a = self.state.pop_n(2)
        self.state.push(1 if ((a != 0) != (b != 0)) else 0)

    def _runner_BAND(self, instr: Instruction) -> None:
        """BAND: Bitwise AND."""
        b, a = self.state.pop_n(2)
        self.state.push(a & b)

    def _runner_BOR(self, instr: Instruction) -> None:
        """BOR: Bitwise OR."""
        b, a = self.state.pop_n(2)
        self.state.push(a | b)

    def _runner_BXOR(self, instr: Instruction) -> None:
        """BXOR: Bitwise XOR."""
        b, a = self.state.pop_n(2)
        self.state.push(a ^ b)

    def _runner_BNOT(self, instr: Instruction) -> None:
        """BNOT: Bitwise NOT (complement)."""
        value = self.state.pop()
        self.state.push(~value)

    def _runner_SHL(self, instr: Instruction) -> None:
        """SHL: push(second << top)."""
        b, a = self.state.pop_n(2)
        self.state.push(a << b)

    def _runner_SHR(self, instr: Instruction) -> None:
        """SHR: push(second >> top)."""
        b, a = self.state.pop_n(2)
        self.state.push(a >> b)

    def _runner_VEC(self, instr: Instruction) -> None:
        """VEC: Create empty vec (type inferred on first push)."""
        reg = instr.operands[0]
        self.state.set_register(reg, Vec())

    def _runner_VECI(self, instr: Instruction) -> None:
        """VECI: Create empty vec<int>."""
        reg = instr.operands[0]
        vec = Vec()
        vec.element_type = VecElem("int")
        self.state.set_register(reg, vec)

    def _runner_VECX(self, instr: Instruction) -> None:
        """VECX: Create empty vec<xqmx>."""
        reg = instr.operands[0]
        vec = Vec()
        vec.element_type = VecElem("xqmx")
        self.state.set_register(reg, vec)

    def _runner_BQMX(self, instr: Instruction) -> None:
        """BQMX: Create binary model XQMX."""
        reg = instr.operands[0]
        size = self.state.pop()
        xqmx = XQMX.binary_model(size)
        self.state.set_register(reg, xqmx)

    def _runner_SQMX(self, instr: Instruction) -> None:
        """SQMX: Create spin model XQMX."""
        reg = instr.operands[0]
        size = self.state.pop()
        xqmx = XQMX.spin_model(size)
        self.state.set_register(reg, xqmx)

    def _runner_XQMX(self, instr: Instruction) -> None:
        """XQMX: Create discrete model XQMX."""
        reg = instr.operands[0]
        k, size = self.state.pop_n(2)
        xqmx = XQMX.discrete_model(size, k)
        self.state.set_register(reg, xqmx)

    def _runner_BSMX(self, instr: Instruction) -> None:
        """BSMX: Create binary sample XQMX."""
        reg = instr.operands[0]
        size = self.state.pop()
        xqmx = XQMX.binary_sample(size)
        self.state.set_register(reg, xqmx)

    def _runner_SSMX(self, instr: Instruction) -> None:
        """SSMX: Create spin sample XQMX."""
        reg = instr.operands[0]
        size = self.state.pop()
        xqmx = XQMX.spin_sample(size)
        self.state.set_register(reg, xqmx)

    def _runner_XSMX(self, instr: Instruction) -> None:
        """XSMX: Create discrete sample XQMX."""
        reg = instr.operands[0]
        k, size = self.state.pop_n(2)
        xqmx = XQMX.discrete_sample(size, k)
        self.state.set_register(reg, xqmx)

    def _runner_VECPUSH(self, instr: Instruction) -> None:
        """VECPUSH: Push value onto vec (infers/validates type)."""
        reg = instr.operands[0]
        vec = self._get_register_as_vec(reg)
        value = self.state.pop()
        vec.push(value)

    def _runner_VECGET(self, instr: Instruction) -> None:
        """VECGET: Get vec[index]."""
        reg = instr.operands[0]
        vec = self._get_register_as_vec(reg)
        index = self.state.pop()
        value = vec.get(index)
        if isinstance(value, int):
            self.state.push(value)
        else:
            raise TypeMismatch("int", type(value).__name__, "VECGET result")

    def _runner_VECSET(self, instr: Instruction) -> None:
        """VECSET: Set vec[index] = value."""
        reg = instr.operands[0]
        vec = self._get_register_as_vec(reg)
        value, index = self.state.pop_n(2)
        vec.set(index, value)

    def _runner_VECLEN(self, instr: Instruction) -> None:
        """VECLEN: Push vec length onto stack."""
        reg = instr.operands[0]
        vec = self._get_register_as_vec(reg)
        self.state.push(vec.length)

    def _runner_GETLINE(self, instr: Instruction) -> None:
        """GETLINE: Get linear coefficient."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        index = self.state.pop()
        value = xqmx.get_linear(index)
        self.state.push(int(value))

    def _runner_SETLINE(self, instr: Instruction) -> None:
        """SETLINE: Set linear coefficient."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        value, index = self.state.pop_n(2)
        xqmx.set_linear(index, float(value))

    def _runner_ADDLINE(self, instr: Instruction) -> None:
        """ADDLINE: Add to linear coefficient."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        delta, index = self.state.pop_n(2)
        xqmx.add_linear(index, float(delta))

    def _runner_GETQUAD(self, instr: Instruction) -> None:
        """GETQUAD: Get quadratic coefficient."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        j, i = self.state.pop_n(2)
        value = xqmx.get_quadratic(i, j)
        self.state.push(int(value))

    def _runner_SETQUAD(self, instr: Instruction) -> None:
        """SETQUAD: Set quadratic coefficient."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        value, j, i = self.state.pop_n(3)
        xqmx.set_quadratic(i, j, float(value))

    def _runner_ADDQUAD(self, instr: Instruction) -> None:
        """ADDQUAD: Add to quadratic coefficient."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        delta, j, i = self.state.pop_n(3)
        xqmx.add_quadratic(i, j, float(delta))

    def _runner_IDXGRID(self, instr: Instruction) -> None:
        """IDXGRID: Convert (row, col) to flat index using cols."""
        cols, j, i = self.state.pop_n(3)  # pop cols, col, row
        index = i * cols + j
        self.state.push(index)

    def _runner_IDXTRIU(self, instr: Instruction) -> None:
        """IDXTRIU: Convert (i, j) to upper triangular index."""
        j, i = self.state.pop_n(2)
        # Ensure i < j for upper triangular
        if i > j:
            i, j = j, i
        idx = j * (j - 1) // 2 + i
        self.state.push(idx)

    def _runner_RESIZE(self, instr: Instruction) -> None:
        """RESIZE: Set grid dimensions."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        cols, rows = self.state.pop_n(2)
        xqmx.rows = rows
        xqmx.cols = cols

    def _runner_ROWFIND(self, instr: Instruction) -> None:
        """ROWFIND: Find first col where row has value."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        value, row = self.state.pop_n(2)
        col = row_find(xqmx, row, value)
        self.state.push(col)

    def _runner_COLFIND(self, instr: Instruction) -> None:
        """COLFIND: Find first row where col has value."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        value, col = self.state.pop_n(2)
        row = col_find(xqmx, col, value)
        self.state.push(row)

    def _runner_ROWSUM(self, instr: Instruction) -> None:
        """ROWSUM: Sum all values in row."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        row = self.state.pop()
        total = xqmx_row_sum(xqmx, row)
        self.state.push(int(total))

    def _runner_COLSUM(self, instr: Instruction) -> None:
        """COLSUM: Sum all values in column."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        col = self.state.pop()
        total = xqmx_col_sum(xqmx, col)
        self.state.push(int(total))

    def _runner_ONEHOTR(self, instr: Instruction) -> None:
        """ONEHOTR: Add one-hot constraint for row."""
        reg = instr.operands[0]
        model = self._get_register_as_xqmx(reg)
        penalty, row = self.state.pop_n(2)

        if model.rows == 0 or model.cols == 0:
            raise ValueError("ONEHOTR requires grid dimensions to be set")

        indices = row_indices(model, row)
        expand_onehot(model, indices, float(penalty))

    def _runner_ONEHOTC(self, instr: Instruction) -> None:
        """ONEHOTC: Add one-hot constraint for column."""
        reg = instr.operands[0]
        model = self._get_register_as_xqmx(reg)
        penalty, col = self.state.pop_n(2)

        if model.rows == 0 or model.cols == 0:
            raise ValueError("ONEHOTC requires grid dimensions to be set")

        indices = col_indices(model, col)
        expand_onehot(model, indices, float(penalty))

    def _runner_EXCLUDE(self, instr: Instruction) -> None:
        """EXCLUDE: Add exclusion constraint with penalty."""
        reg = instr.operands[0]
        model = self._get_register_as_xqmx(reg)
        penalty, j, i = self.state.pop_n(3)
        expand_exclude(model, i, j, float(penalty))

    def _runner_IMPLIES(self, instr: Instruction) -> None:
        """IMPLIES: Add implication constraint with penalty."""
        reg = instr.operands[0]
        model = self._get_register_as_xqmx(reg)
        penalty, j, i = self.state.pop_n(3)
        expand_implies(model, i, j, float(penalty))

    def _runner_ENERGY(self, instr: Instruction) -> None:
        """ENERGY: Compute energy of sample against model."""
        model_reg = instr.operands[0]
        sample_reg = instr.operands[1]
        model = self._get_register_as_xqmx(model_reg)
        sample = self._get_register_as_xqmx(sample_reg)
        energy = compute_energy(model, sample)
        self.state.push(int(energy))
