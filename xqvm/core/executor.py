"""
XQVM Executor: Fetch-Decode-Execute loop with dispatch table.

The Executor processes XQVM programs by fetching instructions,
decoding opcodes, and dispatching to handler methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from .opcodes import Opcode
from .state import MachineState
from .vector import Vec, VecElem
from .xqmx import (
    XQMX,
    row_find,
    col_find,
    row_sum as xqmx_row_sum,
    col_sum as xqmx_col_sum,
    expand_onehot,
    expand_exclude,
    expand_implies,
    compute_energy,
)
from .errors import (
    InvalidOpcode,
    DivisionByZero,
    TargetNotFound,
    TypeMismatch,
)

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
            Opcode.NOP: self._handle_nop,
            Opcode.HALT: self._handle_halt,
            Opcode.TARGET: self._handle_target,
            Opcode.JUMP: self._handle_jump,
            Opcode.JUMPI: self._handle_jumpi,
            Opcode.RANGE: self._handle_range,
            Opcode.ITER: self._handle_iter,
            Opcode.NEXT: self._handle_next,
            Opcode.LVAL: self._handle_lval,
            # Stack & Register I/O
            Opcode.PUSH: self._handle_push,
            Opcode.POP: self._handle_pop,
            Opcode.DUPL: self._handle_dupl,
            Opcode.SWAP: self._handle_swap,
            Opcode.LOAD: self._handle_load,
            Opcode.STOW: self._handle_stow,
            Opcode.INPUT: self._handle_input,
            Opcode.OUTPUT: self._handle_output,
            # Arithmetic
            Opcode.ADD: self._handle_add,
            Opcode.SUB: self._handle_sub,
            Opcode.MUL: self._handle_mul,
            Opcode.DIV: self._handle_div,
            Opcode.MOD: self._handle_mod,
            Opcode.NEG: self._handle_neg,
            # Comparison
            Opcode.EQ: self._handle_eq,
            Opcode.LT: self._handle_lt,
            Opcode.GT: self._handle_gt,
            Opcode.LTE: self._handle_lte,
            Opcode.GTE: self._handle_gte,
            # Boolean
            Opcode.NOT: self._handle_not,
            Opcode.AND: self._handle_and,
            Opcode.OR: self._handle_or,
            Opcode.XOR: self._handle_xor,
            # Bitwise
            Opcode.BAND: self._handle_band,
            Opcode.BOR: self._handle_bor,
            Opcode.BXOR: self._handle_bxor,
            Opcode.BNOT: self._handle_bnot,
            Opcode.SHL: self._handle_shl,
            Opcode.SHR: self._handle_shr,
            # Allocators
            Opcode.VEC: self._handle_vec,
            Opcode.VECI: self._handle_veci,
            Opcode.VECX: self._handle_vecx,
            Opcode.BQMX: self._handle_bqmx,
            Opcode.SQMX: self._handle_sqmx,
            Opcode.XQMX: self._handle_xqmx,
            Opcode.BSMX: self._handle_bsmx,
            Opcode.SSMX: self._handle_ssmx,
            Opcode.XSMX: self._handle_xsmx,
            # Vec Access
            Opcode.VECPUSH: self._handle_vecpush,
            Opcode.VECGET: self._handle_vecget,
            Opcode.VECSET: self._handle_vecset,
            Opcode.VECLEN: self._handle_veclen,
            # XQMX Access
            Opcode.GETLINE: self._handle_getline,
            Opcode.SETLINE: self._handle_setline,
            Opcode.ADDLINE: self._handle_addline,
            Opcode.GETQUAD: self._handle_getquad,
            Opcode.SETQUAD: self._handle_setquad,
            Opcode.ADDQUAD: self._handle_addquad,
            # Vector Math
            Opcode.IDXGRID: self._handle_idxgrid,
            Opcode.IDXTRIU: self._handle_idxtriu,
            # XQMX Grid
            Opcode.RESIZE: self._handle_resize,
            Opcode.ROWFIND: self._handle_rowfind,
            Opcode.COLFIND: self._handle_colfind,
            Opcode.ROWSUM: self._handle_rowsum,
            Opcode.COLSUM: self._handle_colsum,
            # XQMX High-level
            Opcode.ONEHOT: self._handle_onehot,
            Opcode.EXCLUDE: self._handle_exclude,
            Opcode.IMPLIES: self._handle_implies,
            Opcode.ENERGY: self._handle_energy,
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

            handler(instr)

            # Advance PC if handler didn't jump
            # Note: HALT, JUMP, JUMPI, NEXT may modify PC or halt
            if not self.state.halted:
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

    def _get_register_as_vec(self, slot: int) -> Vec:
        """Get a register value, ensuring it's a Vec."""
        value = self.state.get_register(slot)
        if not isinstance(value, Vec):
            raise TypeMismatch("Vec", type(value).__name__, f"register r{slot}")
        return value

    def _get_register_as_xqmx(self, slot: int) -> XQMX:
        """Get a register value, ensuring it's an XQMX."""
        value = self.state.get_register(slot)
        if not isinstance(value, XQMX):
            raise TypeMismatch("XQMX", type(value).__name__, f"register r{slot}")
        return value

    def _get_register_as_int(self, slot: int) -> int:
        """Get a register value, ensuring it's an int."""
        value = self.state.get_register(slot)
        if not isinstance(value, int):
            raise TypeMismatch("int", type(value).__name__, f"register r{slot}")
        return value

    # =========================================================================
    # Control Flow Handlers
    # =========================================================================

    def _handle_nop(self, instr: Instruction) -> None:
        """NOP: No operation."""
        pass

    def _handle_halt(self, instr: Instruction) -> None:
        """HALT: Stop execution."""
        self.state.halt()

    def _handle_target(self, instr: Instruction) -> None:
        """TARGET: Define a jump target at current PC."""
        target_id = instr.operands[0]
        self.state.jc.define_target(target_id, self.state.pc)

    def _handle_jump(self, instr: Instruction) -> None:
        """JUMP: Unconditional jump to target."""
        target_id = instr.operands[0]
        pc = self.state.jc.resolve_target(target_id)
        if pc is None:
            raise TargetNotFound(target_id)
        # Jump to target, then step() will advance to next instruction
        self.state.jump_to(pc)

    def _handle_jumpi(self, instr: Instruction) -> None:
        """JUMPI: Jump to target if top of stack is non-zero."""
        target_id = instr.operands[0]
        condition = self.state.pop()

        if condition != 0:
            pc = self.state.jc.resolve_target(target_id)
            if pc is None:
                raise TargetNotFound(target_id)
            self.state.jump_to(pc)

    def _handle_range(self, instr: Instruction) -> None:
        """RANGE: Start range loop. Pop count, start -> iterate [start, start+count)."""
        count, start = self.state.pop_n(2)
        # Store current PC + 1 as the loop target (next instruction)
        self.state.jc.push_loop_range(self.state.pc + 1, start, count)

    def _handle_iter(self, instr: Instruction) -> None:
        """ITER: Start vec iteration. Pop end_idx, start_idx -> iterate vec[start:end]."""
        reg = instr.operands[0]
        vec = self._get_register_as_vec(reg)
        end_idx, start_idx = self.state.pop_n(2)
        # Store current PC + 1 as the loop target (next instruction)
        self.state.jc.push_loop_iter(self.state.pc + 1, vec, start_idx, end_idx)

    def _handle_next(self, instr: Instruction) -> None:
        """NEXT: Advance loop index, jump back if more, else pop frame."""
        frame = self.state.jc.current_loop()
        if frame is None:
            from .errors import LoopError
            raise LoopError("NEXT outside of loop")

        if self.state.jc.advance_loop():
            # More iterations: jump back to loop start
            self.state.jump_to(frame.target)
        # else: loop finished, frame popped, continue normally

    def _handle_lval(self, instr: Instruction) -> None:
        """LVAL: Copy current loop value to register."""
        reg = instr.operands[0]
        value = self.state.jc.current_value()
        self.state.set_register(reg, value)

    # =========================================================================
    # Stack & Register I/O Handlers
    # =========================================================================

    def _handle_push(self, instr: Instruction) -> None:
        """PUSH: Push immediate value onto stack."""
        value = instr.operands[0]
        self.state.push(value)

    def _handle_pop(self, instr: Instruction) -> None:
        """POP: Pop and discard top of stack."""
        self.state.pop()

    def _handle_dupl(self, instr: Instruction) -> None:
        """DUPL: Duplicate top of stack."""
        value = self.state.peek()
        self.state.push(value)

    def _handle_swap(self, instr: Instruction) -> None:
        """SWAP: Swap top two stack values."""
        a, b = self.state.pop_n(2)
        self.state.push(a)
        self.state.push(b)

    def _handle_load(self, instr: Instruction) -> None:
        """LOAD: Load register value onto stack."""
        reg = instr.operands[0]
        value = self._get_register_as_int(reg)
        self.state.push(value)

    def _handle_stow(self, instr: Instruction) -> None:
        """STOW: Store top of stack into register."""
        reg = instr.operands[0]
        value = self.state.pop()
        self.state.set_register(reg, value)

    def _handle_input(self, instr: Instruction) -> None:
        """INPUT: Load input slot into register."""
        reg = instr.operands[0]
        slot = self.state.pop()
        value = self.state.get_input(slot)
        self.state.set_register(reg, value)

    def _handle_output(self, instr: Instruction) -> None:
        """OUTPUT: Write register to output slot."""
        reg = instr.operands[0]
        slot = self.state.pop()
        value = self.state.get_register(reg)
        self.state.set_output(slot, value)

    # =========================================================================
    # Arithmetic Handlers
    # =========================================================================

    def _handle_add(self, instr: Instruction) -> None:
        """ADD: push(pop() + pop())."""
        b, a = self.state.pop_n(2)
        self.state.push(a + b)

    def _handle_sub(self, instr: Instruction) -> None:
        """SUB: push(second - top)."""
        b, a = self.state.pop_n(2)
        self.state.push(a - b)

    def _handle_mul(self, instr: Instruction) -> None:
        """MUL: push(pop() * pop())."""
        b, a = self.state.pop_n(2)
        self.state.push(a * b)

    def _handle_div(self, instr: Instruction) -> None:
        """DIV: push(second / top)."""
        b, a = self.state.pop_n(2)
        if b == 0:
            raise DivisionByZero()
        self.state.push(a // b)

    def _handle_mod(self, instr: Instruction) -> None:
        """MOD: push(second % top)."""
        b, a = self.state.pop_n(2)
        if b == 0:
            raise DivisionByZero()
        self.state.push(a % b)

    def _handle_neg(self, instr: Instruction) -> None:
        """NEG: Negate top value."""
        value = self.state.pop()
        self.state.push(-value)

    # =========================================================================
    # Comparison Handlers
    # =========================================================================

    def _handle_eq(self, instr: Instruction) -> None:
        """EQ: push(1 if second == top else 0)."""
        b, a = self.state.pop_n(2)
        self.state.push(1 if a == b else 0)

    def _handle_lt(self, instr: Instruction) -> None:
        """LT: push(1 if second < top else 0)."""
        b, a = self.state.pop_n(2)
        self.state.push(1 if a < b else 0)

    def _handle_gt(self, instr: Instruction) -> None:
        """GT: push(1 if second > top else 0)."""
        b, a = self.state.pop_n(2)
        self.state.push(1 if a > b else 0)

    def _handle_lte(self, instr: Instruction) -> None:
        """LTE: push(1 if second <= top else 0)."""
        b, a = self.state.pop_n(2)
        self.state.push(1 if a <= b else 0)

    def _handle_gte(self, instr: Instruction) -> None:
        """GTE: push(1 if second >= top else 0)."""
        b, a = self.state.pop_n(2)
        self.state.push(1 if a >= b else 0)

    # =========================================================================
    # Boolean Handlers
    # =========================================================================

    def _handle_not(self, instr: Instruction) -> None:
        """NOT: push(1 if top == 0 else 0)."""
        value = self.state.pop()
        self.state.push(1 if value == 0 else 0)

    def _handle_and(self, instr: Instruction) -> None:
        """AND: push(1 if both non-zero else 0)."""
        b, a = self.state.pop_n(2)
        self.state.push(1 if (a != 0 and b != 0) else 0)

    def _handle_or(self, instr: Instruction) -> None:
        """OR: push(1 if either non-zero else 0)."""
        b, a = self.state.pop_n(2)
        self.state.push(1 if (a != 0 or b != 0) else 0)

    def _handle_xor(self, instr: Instruction) -> None:
        """XOR: push(1 if exactly one non-zero else 0)."""
        b, a = self.state.pop_n(2)
        self.state.push(1 if ((a != 0) != (b != 0)) else 0)

    # =========================================================================
    # Bitwise Handlers
    # =========================================================================

    def _handle_band(self, instr: Instruction) -> None:
        """BAND: Bitwise AND."""
        b, a = self.state.pop_n(2)
        self.state.push(a & b)

    def _handle_bor(self, instr: Instruction) -> None:
        """BOR: Bitwise OR."""
        b, a = self.state.pop_n(2)
        self.state.push(a | b)

    def _handle_bxor(self, instr: Instruction) -> None:
        """BXOR: Bitwise XOR."""
        b, a = self.state.pop_n(2)
        self.state.push(a ^ b)

    def _handle_bnot(self, instr: Instruction) -> None:
        """BNOT: Bitwise NOT (complement)."""
        value = self.state.pop()
        self.state.push(~value)

    def _handle_shl(self, instr: Instruction) -> None:
        """SHL: push(second << top)."""
        b, a = self.state.pop_n(2)
        self.state.push(a << b)

    def _handle_shr(self, instr: Instruction) -> None:
        """SHR: push(second >> top)."""
        b, a = self.state.pop_n(2)
        self.state.push(a >> b)

    # =========================================================================
    # Allocator Handlers
    # =========================================================================

    def _handle_vec(self, instr: Instruction) -> None:
        """VEC: Create empty vec (type inferred on first push)."""
        reg = instr.operands[0]
        self.state.set_register(reg, Vec())

    def _handle_veci(self, instr: Instruction) -> None:
        """VECI: Create empty vec<int>."""
        reg = instr.operands[0]
        vec = Vec()
        vec.element_type = VecElem("int")
        self.state.set_register(reg, vec)

    def _handle_vecx(self, instr: Instruction) -> None:
        """VECX: Create empty vec<xqmx>."""
        reg = instr.operands[0]
        vec = Vec()
        vec.element_type = VecElem("xqmx")
        self.state.set_register(reg, vec)

    def _handle_bqmx(self, instr: Instruction) -> None:
        """BQMX: Create binary model XQMX."""
        reg = instr.operands[0]
        size = self.state.pop()
        xqmx = XQMX.binary_model(size)
        self.state.set_register(reg, xqmx)

    def _handle_sqmx(self, instr: Instruction) -> None:
        """SQMX: Create spin model XQMX."""
        reg = instr.operands[0]
        size = self.state.pop()
        xqmx = XQMX.spin_model(size)
        self.state.set_register(reg, xqmx)

    def _handle_xqmx(self, instr: Instruction) -> None:
        """XQMX: Create discrete model XQMX."""
        reg = instr.operands[0]
        k, size = self.state.pop_n(2)
        xqmx = XQMX.discrete_model(size, k)
        self.state.set_register(reg, xqmx)

    def _handle_bsmx(self, instr: Instruction) -> None:
        """BSMX: Create binary sample XQMX."""
        reg = instr.operands[0]
        size = self.state.pop()
        xqmx = XQMX.binary_sample(size)
        self.state.set_register(reg, xqmx)

    def _handle_ssmx(self, instr: Instruction) -> None:
        """SSMX: Create spin sample XQMX."""
        reg = instr.operands[0]
        size = self.state.pop()
        xqmx = XQMX.spin_sample(size)
        self.state.set_register(reg, xqmx)

    def _handle_xsmx(self, instr: Instruction) -> None:
        """XSMX: Create discrete sample XQMX."""
        reg = instr.operands[0]
        k, size = self.state.pop_n(2)
        xqmx = XQMX.discrete_sample(size, k)
        self.state.set_register(reg, xqmx)

    # =========================================================================
    # Vec Access Handlers
    # =========================================================================

    def _handle_vecpush(self, instr: Instruction) -> None:
        """VECPUSH: Push value onto vec (infers/validates type)."""
        reg = instr.operands[0]
        vec = self._get_register_as_vec(reg)
        value = self.state.pop()
        vec.push(value)

    def _handle_vecget(self, instr: Instruction) -> None:
        """VECGET: Get vec[index]."""
        reg = instr.operands[0]
        vec = self._get_register_as_vec(reg)
        index = self.state.pop()
        value = vec.get(index)
        if isinstance(value, int):
            self.state.push(value)
        else:
            raise TypeMismatch("int", type(value).__name__, "VECGET result")

    def _handle_vecset(self, instr: Instruction) -> None:
        """VECSET: Set vec[index] = value."""
        reg = instr.operands[0]
        vec = self._get_register_as_vec(reg)
        value, index = self.state.pop_n(2)
        vec.set(index, value)

    def _handle_veclen(self, instr: Instruction) -> None:
        """VECLEN: Push vec length onto stack."""
        reg = instr.operands[0]
        vec = self._get_register_as_vec(reg)
        self.state.push(vec.length)

    # =========================================================================
    # XQMX Access Handlers
    # =========================================================================

    def _handle_getline(self, instr: Instruction) -> None:
        """GETLINE: Get linear coefficient."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        index = self.state.pop()
        value = xqmx.get_linear(index)
        self.state.push(int(value))

    def _handle_setline(self, instr: Instruction) -> None:
        """SETLINE: Set linear coefficient."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        value, index = self.state.pop_n(2)
        xqmx.set_linear(index, float(value))

    def _handle_addline(self, instr: Instruction) -> None:
        """ADDLINE: Add to linear coefficient."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        delta, index = self.state.pop_n(2)
        xqmx.add_linear(index, float(delta))

    def _handle_getquad(self, instr: Instruction) -> None:
        """GETQUAD: Get quadratic coefficient."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        j, i = self.state.pop_n(2)
        value = xqmx.get_quadratic(i, j)
        self.state.push(int(value))

    def _handle_setquad(self, instr: Instruction) -> None:
        """SETQUAD: Set quadratic coefficient."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        value, j, i = self.state.pop_n(3)
        xqmx.set_quadratic(i, j, float(value))

    def _handle_addquad(self, instr: Instruction) -> None:
        """ADDQUAD: Add to quadratic coefficient."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        delta, j, i = self.state.pop_n(3)
        xqmx.add_quadratic(i, j, float(delta))

    # =========================================================================
    # Vector Math Handlers
    # =========================================================================

    def _handle_idxgrid(self, instr: Instruction) -> None:
        """IDXGRID: Convert (row, col) to flat index using cols."""
        cols, j, i = self.state.pop_n(3)  # pop cols, col, row
        index = i * cols + j
        self.state.push(index)

    def _handle_idxtriu(self, instr: Instruction) -> None:
        """IDXTRIU: Convert (i, j) to upper triangular index."""
        j, i = self.state.pop_n(2)
        # Ensure i < j for upper triangular
        if i > j:
            i, j = j, i
        idx = j * (j - 1) // 2 + i
        self.state.push(idx)

    # =========================================================================
    # XQMX Grid Handlers
    # =========================================================================

    def _handle_resize(self, instr: Instruction) -> None:
        """RESIZE: Set grid dimensions."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        cols, rows = self.state.pop_n(2)
        xqmx.rows = rows
        xqmx.cols = cols

    def _handle_rowfind(self, instr: Instruction) -> None:
        """ROWFIND: Find first col where row has value."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        value, row = self.state.pop_n(2)
        col = row_find(xqmx, row, value)
        self.state.push(col)

    def _handle_colfind(self, instr: Instruction) -> None:
        """COLFIND: Find first row where col has value."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        value, col = self.state.pop_n(2)
        row = col_find(xqmx, col, value)
        self.state.push(row)

    def _handle_rowsum(self, instr: Instruction) -> None:
        """ROWSUM: Sum all values in row."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        row = self.state.pop()
        total = xqmx_row_sum(xqmx, row)
        self.state.push(int(total))

    def _handle_colsum(self, instr: Instruction) -> None:
        """COLSUM: Sum all values in column."""
        reg = instr.operands[0]
        xqmx = self._get_register_as_xqmx(reg)
        col = self.state.pop()
        total = xqmx_col_sum(xqmx, col)
        self.state.push(int(total))

    # =========================================================================
    # XQMX High-Level Handlers
    # =========================================================================

    def _handle_onehot(self, instr: Instruction) -> None:
        """ONEHOT: Add one-hot constraint."""
        reg = instr.operands[0]
        model = self._get_register_as_xqmx(reg)
        penalty, row = self.state.pop_n(2)

        # Get indices for the row
        if model.rows == 0 or model.cols == 0:
            raise ValueError("ONEHOT requires grid dimensions to be set")

        from .xqmx import row_indices
        indices = row_indices(model, row)
        expand_onehot(model, indices, float(penalty))

    def _handle_exclude(self, instr: Instruction) -> None:
        """EXCLUDE: Add exclusion constraint with penalty."""
        reg = instr.operands[0]
        model = self._get_register_as_xqmx(reg)
        penalty, j, i = self.state.pop_n(3)
        expand_exclude(model, i, j, float(penalty))

    def _handle_implies(self, instr: Instruction) -> None:
        """IMPLIES: Add implication constraint with penalty."""
        reg = instr.operands[0]
        model = self._get_register_as_xqmx(reg)
        penalty, j, i = self.state.pop_n(3)
        expand_implies(model, i, j, float(penalty))

    def _handle_energy(self, instr: Instruction) -> None:
        """ENERGY: Compute energy of sample against model."""
        model_reg = instr.operands[0]
        sample_reg = instr.operands[1]
        model = self._get_register_as_xqmx(model_reg)
        sample = self._get_register_as_xqmx(sample_reg)
        energy = compute_energy(model, sample)
        self.state.push(int(energy))
