"""
Tests for Executor and all 84 opcode handlers.
"""

import pytest

from xqvm.core.errors import (
    DivisionByZero,
    LoopError,
    RegisterNotFound,
    StackUnderflow,
    TargetNotFound,
    TypeMismatch,
    XQMXModeError,
)
from xqvm.core.executor import Executor
from xqvm.core.opcodes import Opcode
from xqvm.core.program import Instruction, make_program, run_program
from xqvm.core.state import MachineState
from xqvm.core.vector import Vec
from xqvm.core.xqmx import XQMX


class TestControlFlow:
    """Tests for control flow opcodes."""

    def test_nop(self):
        """NOP does nothing but advance PC."""
        ex = run_program(
            [
                Instruction(Opcode.NOP),
                Instruction(Opcode.NOP),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.halted is True

    def test_halt(self):
        """HALT stops execution."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.HALT),
                Instruction(Opcode.PUSH1, (2,)),  # Never reached
            ]
        )
        assert ex.state.stack_depth == 1
        assert ex.state.peek(0) == 1

    def test_target_and_jump(self):
        """TARGET defines label, JUMP jumps back to it."""
        # Define target first, then jump back to it
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (0,)),  # Counter
                Instruction(Opcode.STOW, (0,)),
                Instruction(Opcode.TARGET, ()),  # Target 0 - loop start
                Instruction(Opcode.LOAD, (0,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.ADD),
                Instruction(Opcode.COPY),
                Instruction(Opcode.STOW, (0,)),  # counter++
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.LT),  # counter < 3?
                Instruction(Opcode.JUMPI, (0,)),  # Jump back to target 0
                Instruction(Opcode.HALT),
            ]
        )
        # Counter should be 3
        assert ex.state.get_register(0) == 3

    def test_forward_jump(self):
        """JUMP to a TARGET defined later in the program."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (42,)),
                Instruction(Opcode.STOW, (0,)),
                Instruction(Opcode.JUMP, (0,)),  # Forward jump to target 0
                Instruction(Opcode.PUSH1, (99,)),  # Skipped
                Instruction(Opcode.STOW, (0,)),  # Skipped
                Instruction(Opcode.TARGET, ()),  # Target 0 defined after jump
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.get_register(0) == 42

    def test_forward_jumpi(self):
        """JUMPI forward to a TARGET defined later (break-out-of-loop pattern)."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (1,)),  # Non-zero condition
                Instruction(Opcode.JUMPI, (0,)),  # Forward jump
                Instruction(Opcode.PUSH1, (99,)),  # Skipped
                Instruction(Opcode.STOW, (0,)),  # Skipped
                Instruction(Opcode.TARGET, ()),
                Instruction(Opcode.HALT),
            ]
        )
        assert not ex.state.has_register(0)

    def test_jumpi_taken(self):
        """JUMPI jumps when top of stack is non-zero."""
        # Define target first, then conditionally jump to it
        ex = run_program(
            [
                Instruction(Opcode.TARGET, ()),  # Define target 0 here
                Instruction(Opcode.PUSH1, (42,)),
                Instruction(Opcode.STOW, (0,)),
                Instruction(Opcode.PUSH1, (1,)),  # Non-zero condition
                Instruction(Opcode.NOT),  # Invert to 0, so we don't jump
                Instruction(Opcode.JUMPI, (0,)),  # Won't jump (0)
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.get_register(0) == 42

    def test_jumpi_not_taken(self):
        """JUMPI doesn't jump when top of stack is zero."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (99,)),
                Instruction(Opcode.STOW, (0,)),
                Instruction(Opcode.TARGET, ()),  # Define target first
                Instruction(Opcode.PUSH1, (0,)),  # Zero condition
                Instruction(Opcode.JUMPI, (0,)),  # Should not jump (0)
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.get_register(0) == 99

    def test_range_loop(self):
        """RANGE creates loop with values [start, start+count)."""
        # Sum 0+1+2+3+4 = 10
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (0,)),  # Sum accumulator
                Instruction(Opcode.STOW, (0,)),
                Instruction(Opcode.PUSH1, (0,)),  # Start
                Instruction(Opcode.PUSH1, (5,)),  # Count
                Instruction(Opcode.RANGE),
                Instruction(Opcode.TARGET, ()),  # Loop body
                Instruction(Opcode.LVAL, (1,)),  # i -> r1
                Instruction(Opcode.LOAD, (0,)),  # Load sum
                Instruction(Opcode.LOAD, (1,)),  # Load i
                Instruction(Opcode.ADD),
                Instruction(Opcode.STOW, (0,)),  # Store sum
                Instruction(Opcode.NEXT),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.get_register(0) == 10

    def test_iter_loop(self):
        """ITER iterates over vector elements."""
        ex = run_program(
            [
                # Create vec [10, 20, 30]
                Instruction(Opcode.VECI, (0,)),
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.VECPUSH, (0,)),
                Instruction(Opcode.PUSH1, (20,)),
                Instruction(Opcode.VECPUSH, (0,)),
                Instruction(Opcode.PUSH1, (30,)),
                Instruction(Opcode.VECPUSH, (0,)),
                # Sum accumulator
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.STOW, (1,)),
                # Iterate: start=0, end=3
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.ITER, (0,)),
                Instruction(Opcode.TARGET, ()),
                Instruction(Opcode.LVAL, (2,)),  # val -> r2
                Instruction(Opcode.LOAD, (1,)),  # Load sum
                Instruction(Opcode.LOAD, (2,)),  # Load val
                Instruction(Opcode.ADD),
                Instruction(Opcode.STOW, (1,)),
                Instruction(Opcode.NEXT),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.get_register(1) == 60

    def test_lval(self):
        """LVAL copies current loop value to register."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (5,)),  # Start
                Instruction(Opcode.PUSH1, (1,)),  # Count
                Instruction(Opcode.RANGE),
                Instruction(Opcode.TARGET, ()),
                Instruction(Opcode.LVAL, (0,)),  # Store loop value in r0
                Instruction(Opcode.NEXT),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.get_register(0) == 5


class TestStackRegisterIO:
    """Tests for stack, register, and I/O opcodes."""

    def test_push(self):
        """PUSH puts immediate value on stack."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (42,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 42

    def test_pop(self):
        """POP removes top of stack."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.PUSH1, (2,)),
                Instruction(Opcode.POP),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.stack_depth == 1
        assert ex.state.peek(0) == 1

    def test_copy(self):
        """COPY duplicates top of stack."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (42,)),
                Instruction(Opcode.COPY),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.stack_depth == 2
        assert ex.state.peek(0) == 42
        assert ex.state.peek(1) == 42

    def test_swap(self):
        """SWAP exchanges top two stack values."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.PUSH1, (2,)),
                Instruction(Opcode.SWAP),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 1
        assert ex.state.peek(1) == 2

    def test_sclr(self):
        """SCLR clears entire stack."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.PUSH1, (2,)),
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.SCLR),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.stack_depth == 0

    def test_sclr_empty_stack(self):
        """SCLR on empty stack is a no-op."""
        ex = run_program(
            [
                Instruction(Opcode.SCLR),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.stack_depth == 0

    def test_load(self):
        """LOAD pushes register value onto stack."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (100,)),
                Instruction(Opcode.STOW, (5,)),
                Instruction(Opcode.LOAD, (5,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 100

    def test_stow(self):
        """STOW stores stack top in register."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (42,)),
                Instruction(Opcode.STOW, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.get_register(0) == 42
        assert ex.state.stack_depth == 0

    def test_input(self):
        """INPUT loads input slot to register."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (0,)),  # slot index on stack
                Instruction(Opcode.INPUT, (0,)),  # input[0] -> r0
                Instruction(Opcode.HALT),
            ],
            input_data={0: 42},
        )
        assert ex.state.get_register(0) == 42

    def test_output(self):
        """OUTPUT writes register to output slot."""
        prog = make_program(
            [
                Instruction(Opcode.PUSH1, (99,)),
                Instruction(Opcode.STOW, (0,)),
                Instruction(Opcode.PUSH1, (0,)),  # slot index on stack
                Instruction(Opcode.OUTPUT, (0,)),  # r0 -> output[0]
                Instruction(Opcode.HALT),
            ]
        )
        ex = Executor()
        output = ex.execute(prog)
        assert output[0] == 99

    def test_drop_int_register(self):
        """DROP clears an int register."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (42,)),
                Instruction(Opcode.STOW, (0,)),
                Instruction(Opcode.DROP, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert not ex.state.has_register(0)

    def test_drop_vec_register(self):
        """DROP clears a vec register."""
        ex = run_program(
            [
                Instruction(Opcode.VEC, (0,)),
                Instruction(Opcode.DROP, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert not ex.state.has_register(0)

    def test_drop_unset_register(self):
        """DROP on unset register is a no-op."""
        ex = run_program(
            [
                Instruction(Opcode.DROP, (5,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert not ex.state.has_register(5)

    def test_drop_then_load_raises(self):
        """LOAD after DROP raises RegisterNotFound."""
        prog = make_program(
            [
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.STOW, (0,)),
                Instruction(Opcode.DROP, (0,)),
                Instruction(Opcode.LOAD, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        with pytest.raises(RegisterNotFound):
            Executor().execute(prog)


class TestPushConstant:
    """Tests for PUSH1-PUSH8 push constant opcodes."""

    def test_ldc1_positive(self):
        """PUSH1 loads 1-byte positive constant."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (42,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 42

    def test_ldc1_negative(self):
        """PUSH1 loads 1-byte negative constant (two's complement)."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (0xFF,)),  # -1
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == -1

    def test_ldc1_zero(self):
        """PUSH1 loads zero."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0

    def test_ldc2(self):
        """PUSH2 loads 2-byte constant."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH2, (0x01, 0x00)),  # 256
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 256

    def test_ldc2_negative(self):
        """PUSH2 loads 2-byte negative constant."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH2, (0xFF, 0xFE)),  # -2
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == -2

    def test_ldc3(self):
        """PUSH3 loads 3-byte constant."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH3, (0x01, 0x00, 0x00)),  # 65536
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 65536

    def test_ldc4(self):
        """PUSH4 loads 4-byte constant."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH4, (0x00, 0x01, 0x00, 0x00)),  # 65536
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 65536

    def test_ldc4_max_positive(self):
        """PUSH4 loads max positive 4-byte value."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH4, (0x7F, 0xFF, 0xFF, 0xFF)),  # 2^31 - 1
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 2147483647

    def test_ldc4_min_negative(self):
        """PUSH4 loads min negative 4-byte value."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH4, (0x80, 0x00, 0x00, 0x00)),  # -2^31
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == -2147483648

    def test_ldc8(self):
        """PUSH8 loads 8-byte constant."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH8, (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00)),  # 256
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 256

    def test_ldc8_large_negative(self):
        """PUSH8 loads large negative value."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH8, (0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)),  # -1
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == -1


class TestArithmetic:
    """Tests for arithmetic opcodes."""

    def test_add(self):
        """ADD sums top two values."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.ADD),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 15

    def test_sub(self):
        """SUB subtracts (second - top)."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.SUB),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 7

    def test_mul(self):
        """MUL multiplies top two values."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (6,)),
                Instruction(Opcode.PUSH1, (7,)),
                Instruction(Opcode.MUL),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 42

    def test_div(self):
        """DIV performs integer division."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (20,)),
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.DIV),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 6

    def test_div_by_zero(self):
        """DIV by zero raises DivisionByZero."""
        prog = make_program(
            [
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.DIV),
                Instruction(Opcode.HALT),
            ]
        )
        with pytest.raises(DivisionByZero):
            Executor().execute(prog)

    def test_mod(self):
        """MOD computes remainder."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (17,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.MOD),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 2

    def test_mod_by_zero(self):
        """MOD by zero raises DivisionByZero."""
        prog = make_program(
            [
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.MOD),
                Instruction(Opcode.HALT),
            ]
        )
        with pytest.raises(DivisionByZero):
            Executor().execute(prog)

    def test_neg(self):
        """NEG negates top value."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (42,)),
                Instruction(Opcode.NEG),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == -42

    def test_neg_negative(self):
        """NEG on negative produces positive."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (246,)),
                Instruction(Opcode.NEG),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 10

    def test_sqr(self):
        """SQR squares top value."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (7,)),
                Instruction(Opcode.SQR),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 49

    def test_sqr_negative(self):
        """SQR of negative is positive."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (253,)),
                Instruction(Opcode.SQR),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 9

    def test_sqr_zero(self):
        """SQR of zero is zero."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.SQR),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0

    def test_abs_positive(self):
        """ABS of positive is unchanged."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.ABS),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 5

    def test_abs_negative(self):
        """ABS of negative is positive."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (251,)),
                Instruction(Opcode.ABS),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 5

    def test_abs_zero(self):
        """ABS of zero is zero."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.ABS),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0

    def test_min(self):
        """MIN pushes the smaller of two values."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.MIN),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 3

    def test_min_equal(self):
        """MIN with equal values returns that value."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.MIN),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 5

    def test_min_negative(self):
        """MIN with negative values."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (255,)),
                Instruction(Opcode.PUSH1, (251,)),
                Instruction(Opcode.MIN),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == -5

    def test_max(self):
        """MAX pushes the larger of two values."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.MAX),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 10

    def test_max_negative(self):
        """MAX with negative values."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (255,)),
                Instruction(Opcode.PUSH1, (251,)),
                Instruction(Opcode.MAX),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == -1

    def test_inc(self):
        """INC increments top value."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (41,)),
                Instruction(Opcode.INC),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 42

    def test_inc_negative(self):
        """INC on -1 produces 0."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (255,)),
                Instruction(Opcode.INC),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0

    def test_dec(self):
        """DEC decrements top value."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (43,)),
                Instruction(Opcode.DEC),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 42

    def test_dec_zero(self):
        """DEC on 0 produces -1."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.DEC),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == -1


class TestComparison:
    """Tests for comparison opcodes."""

    def test_eq_true(self):
        """EQ returns 1 when equal."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.EQ),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 1

    def test_eq_false(self):
        """EQ returns 0 when not equal."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.EQ),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0

    def test_lt_true(self):
        """LT returns 1 when second < top."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.LT),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 1

    def test_lt_false(self):
        """LT returns 0 when second >= top."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.LT),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0

    def test_gt_true(self):
        """GT returns 1 when second > top."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.GT),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 1

    def test_gt_false(self):
        """GT returns 0 when second <= top."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.GT),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0

    def test_lte_true_less(self):
        """LTE returns 1 when second < top."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.LTE),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 1

    def test_lte_true_equal(self):
        """LTE returns 1 when equal."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.LTE),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 1

    def test_lte_false(self):
        """LTE returns 0 when second > top."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (7,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.LTE),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0

    def test_gte_true_greater(self):
        """GTE returns 1 when second > top."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (7,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.GTE),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 1

    def test_gte_true_equal(self):
        """GTE returns 1 when equal."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.GTE),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 1

    def test_gte_false(self):
        """GTE returns 0 when second < top."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.GTE),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0


class TestBoolean:
    """Tests for boolean logic opcodes."""

    def test_not_true_to_false(self):
        """NOT converts non-zero to 0."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.NOT),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0

    def test_not_false_to_true(self):
        """NOT converts 0 to 1."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.NOT),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 1

    def test_and_both_true(self):
        """AND of two truthy values is 1."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.AND),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 1

    def test_and_one_false(self):
        """AND with one falsy value is 0."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.AND),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0

    def test_or_both_true(self):
        """OR of two truthy values is 1."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.OR),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 1

    def test_or_one_true(self):
        """OR with one truthy value is 1."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.OR),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 1

    def test_or_both_false(self):
        """OR of two falsy values is 0."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.OR),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0

    def test_xor_different(self):
        """XOR of different values is 1."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.XOR),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 1

    def test_xor_same(self):
        """XOR of same values is 0."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.XOR),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0


class TestBitwise:
    """Tests for bitwise opcodes."""

    def test_band(self):
        """BAND performs bitwise AND."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (0b1100,)),
                Instruction(Opcode.PUSH1, (0b1010,)),
                Instruction(Opcode.BAND),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0b1000

    def test_bor(self):
        """BOR performs bitwise OR."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (0b1100,)),
                Instruction(Opcode.PUSH1, (0b1010,)),
                Instruction(Opcode.BOR),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0b1110

    def test_bxor(self):
        """BXOR performs bitwise XOR."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (0b1100,)),
                Instruction(Opcode.PUSH1, (0b1010,)),
                Instruction(Opcode.BXOR),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0b0110

    def test_bnot(self):
        """BNOT performs bitwise NOT."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.BNOT),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == ~0

    def test_shl(self):
        """SHL shifts left."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.PUSH1, (4,)),
                Instruction(Opcode.SHL),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 16

    def test_shr(self):
        """SHR shifts right."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (16,)),
                Instruction(Opcode.PUSH1, (2,)),
                Instruction(Opcode.SHR),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 4


class TestAllocators:
    """Tests for allocator opcodes."""

    def test_vec_creates_empty(self):
        """VEC creates empty untyped vector."""
        ex = run_program(
            [
                Instruction(Opcode.VEC, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        v = ex.state.get_register(0)
        assert isinstance(v, Vec)
        assert v.length == 0

    def test_veci_creates_int_vec(self):
        """VECI creates empty int vector."""
        ex = run_program(
            [
                Instruction(Opcode.VECI, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        v = ex.state.get_register(0)
        assert isinstance(v, Vec)
        assert v.element_type.kind == "int"

    def test_vecx_creates_xqmx_vec(self):
        """VECX creates empty xqmx vector."""
        ex = run_program(
            [
                Instruction(Opcode.VECX, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        v = ex.state.get_register(0)
        assert isinstance(v, Vec)
        assert v.element_type.kind == "xqmx"

    def test_bqmx_creates_binary_model(self):
        """BQMX creates binary model XQMX."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (10,)),  # size
                Instruction(Opcode.BQMX, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        x = ex.state.get_register(0)
        assert isinstance(x, XQMX)
        assert x.is_model()
        assert x.size == 10

    def test_sqmx_creates_spin_model(self):
        """SQMX creates spin model XQMX."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (8,)),
                Instruction(Opcode.SQMX, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        x = ex.state.get_register(0)
        assert x.is_model()
        assert x.size == 8

    def test_xqmx_creates_discrete_model(self):
        """XQMX creates discrete model with k values."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (5,)),  # size
                Instruction(Opcode.PUSH1, (3,)),  # k
                Instruction(Opcode.XQMX, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        x = ex.state.get_register(0)
        assert x.is_model()
        assert x.discrete_k == 3

    def test_bsmx_creates_binary_sample(self):
        """BSMX creates binary sample XQMX."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.BSMX, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        x = ex.state.get_register(0)
        assert x.is_sample()

    def test_ssmx_creates_spin_sample(self):
        """SSMX creates spin sample XQMX."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.SSMX, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        x = ex.state.get_register(0)
        assert x.is_sample()

    def test_xsmx_creates_discrete_sample(self):
        """XSMX creates discrete sample with k values."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.PUSH1, (4,)),
                Instruction(Opcode.XSMX, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        x = ex.state.get_register(0)
        assert x.is_sample()
        assert x.discrete_k == 4


class TestVectorAccess:
    """Tests for vector access opcodes."""

    def test_vecpush(self):
        """VECPUSH appends value to vector."""
        ex = run_program(
            [
                Instruction(Opcode.VECI, (0,)),
                Instruction(Opcode.PUSH1, (42,)),
                Instruction(Opcode.VECPUSH, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        v = ex.state.get_register(0)
        assert v.length == 1
        assert v.get(0) == 42

    def test_vecget(self):
        """VECGET gets value at index."""
        ex = run_program(
            [
                Instruction(Opcode.VECI, (0,)),
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.VECPUSH, (0,)),
                Instruction(Opcode.PUSH1, (20,)),
                Instruction(Opcode.VECPUSH, (0,)),
                Instruction(Opcode.PUSH1, (1,)),  # index
                Instruction(Opcode.VECGET, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 20

    def test_vecset(self):
        """VECSET sets value at index."""
        ex = run_program(
            [
                Instruction(Opcode.VECI, (0,)),
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.VECPUSH, (0,)),
                Instruction(Opcode.PUSH1, (0,)),  # index
                Instruction(Opcode.PUSH1, (99,)),  # value
                Instruction(Opcode.VECSET, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        v = ex.state.get_register(0)
        assert v.get(0) == 99

    def test_veclen(self):
        """VECLEN pushes vector length."""
        ex = run_program(
            [
                Instruction(Opcode.VECI, (0,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.VECPUSH, (0,)),
                Instruction(Opcode.PUSH1, (2,)),
                Instruction(Opcode.VECPUSH, (0,)),
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.VECPUSH, (0,)),
                Instruction(Opcode.VECLEN, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 3


class TestVectorMath:
    """Tests for vector math opcodes."""

    def test_idxgrid(self):
        """IDXGRID converts (row, col) to flat index."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (2,)),  # row
                Instruction(Opcode.PUSH1, (3,)),  # col
                Instruction(Opcode.PUSH1, (5,)),  # cols
                Instruction(Opcode.IDXGRID),
                Instruction(Opcode.HALT),
            ]
        )
        # index = row * cols + col = 2 * 5 + 3 = 13
        assert ex.state.peek(0) == 13

    def test_idxtriu(self):
        """IDXTRIU converts (i, j) to upper triangular index."""
        # idx = j * (j - 1) // 2 + i (with i < j)
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (0,)),  # i
                Instruction(Opcode.PUSH1, (2,)),  # j
                Instruction(Opcode.IDXTRIU),
                Instruction(Opcode.HALT),
            ]
        )
        # With i=0, j=2: idx = 2 * (2-1) / 2 + 0 = 1
        assert ex.state.peek(0) == 1


class TestXQMXAccess:
    """Tests for XQMX coefficient access opcodes."""

    def test_getline(self):
        """GETLINE gets linear coefficient."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.BQMX, (0,)),
                Instruction(Opcode.PUSH1, (5,)),  # index
                Instruction(Opcode.GETLINE, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        # Default is 0
        assert ex.state.peek(0) == 0

    def test_setline(self):
        """SETLINE sets linear coefficient."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.BQMX, (0,)),
                Instruction(Opcode.PUSH1, (3,)),  # index
                Instruction(Opcode.PUSH1, (5,)),  # value (as int, will be float)
                Instruction(Opcode.SETLINE, (0,)),
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.GETLINE, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 5

    def test_addline(self):
        """ADDLINE adds to linear coefficient."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.BQMX, (0,)),
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.SETLINE, (0,)),
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.PUSH1, (2,)),
                Instruction(Opcode.ADDLINE, (0,)),
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.GETLINE, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 5

    def test_getquad(self):
        """GETQUAD gets quadratic coefficient."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.BQMX, (0,)),
                Instruction(Opcode.PUSH1, (0,)),  # i
                Instruction(Opcode.PUSH1, (1,)),  # j
                Instruction(Opcode.GETQUAD, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 0

    def test_setquad(self):
        """SETQUAD sets quadratic coefficient."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.BQMX, (0,)),
                Instruction(Opcode.PUSH1, (0,)),  # i
                Instruction(Opcode.PUSH1, (1,)),  # j
                Instruction(Opcode.PUSH1, (7,)),  # value
                Instruction(Opcode.SETQUAD, (0,)),
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.GETQUAD, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 7

    def test_addquad(self):
        """ADDQUAD adds to quadratic coefficient."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.BQMX, (0,)),
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.SETQUAD, (0,)),
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.PUSH1, (4,)),
                Instruction(Opcode.ADDQUAD, (0,)),
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.GETQUAD, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 7


class TestXQMXGrid:
    """Tests for XQMX grid opcodes."""

    def test_resize(self):
        """RESIZE sets grid dimensions."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (25,)),
                Instruction(Opcode.BQMX, (0,)),
                Instruction(Opcode.PUSH1, (5,)),  # rows
                Instruction(Opcode.PUSH1, (5,)),  # cols
                Instruction(Opcode.RESIZE, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        x = ex.state.get_register(0)
        assert x.rows == 5
        assert x.cols == 5

    def test_rowfind(self):
        """ROWFIND finds first column with value."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (25,)),
                Instruction(Opcode.BQMX, (0,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.RESIZE, (0,)),
                # Set value at row 0, col 2 (index 2)
                Instruction(Opcode.PUSH1, (2,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.SETLINE, (0,)),
                # Find in row 0
                Instruction(Opcode.PUSH1, (0,)),  # row
                Instruction(Opcode.PUSH1, (1,)),  # value
                Instruction(Opcode.ROWFIND, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 2

    def test_colfind(self):
        """COLFIND finds first row with value."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (25,)),
                Instruction(Opcode.BQMX, (0,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.RESIZE, (0,)),
                # Set value at row 2, col 0 (index 10)
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.SETLINE, (0,)),
                # Find in col 0
                Instruction(Opcode.PUSH1, (0,)),  # col
                Instruction(Opcode.PUSH1, (1,)),  # value
                Instruction(Opcode.COLFIND, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 2

    def test_rowsum(self):
        """ROWSUM sums values in row."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (25,)),
                Instruction(Opcode.BQMX, (0,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.RESIZE, (0,)),
                # Set values in row 0: indices 0, 1, 2
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.SETLINE, (0,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.PUSH1, (2,)),
                Instruction(Opcode.SETLINE, (0,)),
                Instruction(Opcode.PUSH1, (2,)),
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.SETLINE, (0,)),
                # Sum row 0
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.ROWSUM, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 6

    def test_colsum(self):
        """COLSUM sums values in column."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (25,)),
                Instruction(Opcode.BQMX, (0,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.RESIZE, (0,)),
                # Set values in col 0: indices 0, 5, 10
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.SETLINE, (0,)),
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.PUSH1, (2,)),
                Instruction(Opcode.SETLINE, (0,)),
                Instruction(Opcode.PUSH1, (10,)),
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.SETLINE, (0,)),
                # Sum col 0
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.COLSUM, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert ex.state.peek(0) == 6


class TestXQMXHighLevel:
    """Tests for XQMX high-level function opcodes."""

    def test_onehotr(self):
        """ONEHOTR adds one-hot constraint for a row."""
        ex = run_program(
            [
                # Create 3x3 grid model (size=9)
                Instruction(Opcode.PUSH1, (9,)),
                Instruction(Opcode.BQMX, (0,)),
                # Set grid dimensions
                Instruction(Opcode.PUSH1, (3,)),  # rows
                Instruction(Opcode.PUSH1, (3,)),  # cols
                Instruction(Opcode.RESIZE, (0,)),
                # Apply one-hot constraint for row 0 with penalty 1
                Instruction(Opcode.PUSH1, (0,)),  # row
                Instruction(Opcode.PUSH1, (1,)),  # penalty
                Instruction(Opcode.ONEHOTR, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        x = ex.state.get_register(0)
        # Row 0 indices are [0, 1, 2]: should have linear and quadratic terms
        assert len(x.linear) > 0 or len(x.quadratic) > 0

    def test_onehotc(self):
        """ONEHOTC adds one-hot constraint for a column."""
        ex = run_program(
            [
                # Create 3x3 grid model (size=9)
                Instruction(Opcode.PUSH1, (9,)),
                Instruction(Opcode.BQMX, (0,)),
                # Set grid dimensions
                Instruction(Opcode.PUSH1, (3,)),  # rows
                Instruction(Opcode.PUSH1, (3,)),  # cols
                Instruction(Opcode.RESIZE, (0,)),
                # Apply one-hot constraint for col 0 with penalty 1
                Instruction(Opcode.PUSH1, (0,)),  # col
                Instruction(Opcode.PUSH1, (1,)),  # penalty
                Instruction(Opcode.ONEHOTC, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        x = ex.state.get_register(0)
        # Col 0 indices are [0, 3, 6]: should have linear and quadratic terms
        assert x.linear.get(0) == -1
        assert x.linear.get(3) == -1
        assert x.linear.get(6) == -1
        assert x.quadratic.get((0, 3)) == 2
        assert x.quadratic.get((0, 6)) == 2
        assert x.quadratic.get((3, 6)) == 2

    def test_exclude(self):
        """EXCLUDE adds exclusion constraint."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.BQMX, (0,)),
                Instruction(Opcode.PUSH1, (0,)),  # i
                Instruction(Opcode.PUSH1, (1,)),  # j
                Instruction(Opcode.PUSH1, (2,)),  # penalty
                Instruction(Opcode.EXCLUDE, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        x = ex.state.get_register(0)
        assert x.get_quadratic(0, 1) == 2.0

    def test_implies(self):
        """IMPLIES adds implication constraint."""
        ex = run_program(
            [
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.BQMX, (0,)),
                Instruction(Opcode.PUSH1, (0,)),  # i
                Instruction(Opcode.PUSH1, (1,)),  # j
                Instruction(Opcode.PUSH1, (1,)),  # penalty
                Instruction(Opcode.IMPLIES, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        x = ex.state.get_register(0)
        # Should have some coefficients set
        assert len(x.linear) > 0 or len(x.quadratic) > 0

    def test_energy(self):
        """ENERGY computes energy of sample against model."""
        ex = run_program(
            [
                # Create model with linear terms
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.BQMX, (0,)),
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.SETLINE, (0,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.PUSH1, (2,)),
                Instruction(Opcode.SETLINE, (0,)),
                # Create sample with values
                Instruction(Opcode.PUSH1, (3,)),
                Instruction(Opcode.BSMX, (1,)),
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.SETLINE, (1,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.SETLINE, (1,)),
                # Compute energy
                Instruction(Opcode.ENERGY, (0, 1)),
                Instruction(Opcode.HALT),
            ]
        )
        # Energy = 1*1 + 2*1 = 3
        assert ex.state.peek(0) == 3


class TestErrorHandling:
    """Tests for error conditions."""

    def test_stack_underflow(self):
        """Stack underflow raises StackUnderflow."""
        prog = make_program(
            [
                Instruction(Opcode.POP),
                Instruction(Opcode.HALT),
            ]
        )
        with pytest.raises(StackUnderflow):
            Executor().execute(prog)

    def test_register_not_found(self):
        """Missing register raises RegisterNotFound."""
        prog = make_program(
            [
                Instruction(Opcode.LOAD, (99,)),
                Instruction(Opcode.HALT),
            ]
        )
        with pytest.raises(RegisterNotFound):
            Executor().execute(prog)

    def test_type_mismatch_vec_expected(self):
        """Type mismatch when Vec expected raises TypeMismatch."""
        prog = make_program(
            [
                Instruction(Opcode.PUSH1, (42,)),
                Instruction(Opcode.STOW, (0,)),  # r0 = int
                Instruction(Opcode.VECLEN, (0,)),  # Expects Vec
                Instruction(Opcode.HALT),
            ]
        )
        with pytest.raises(TypeMismatch):
            Executor().execute(prog)

    def test_type_mismatch_xqmx_expected(self):
        """Type mismatch when XQMX expected raises TypeMismatch."""
        prog = make_program(
            [
                Instruction(Opcode.PUSH1, (42,)),
                Instruction(Opcode.STOW, (0,)),  # r0 = int
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.GETLINE, (0,)),  # Expects XQMX
                Instruction(Opcode.HALT),
            ]
        )
        with pytest.raises(TypeMismatch):
            Executor().execute(prog)

    def test_target_not_found(self):
        """Jump to undefined target raises TargetNotFound."""
        prog = make_program(
            [
                Instruction(Opcode.JUMP, (99,)),  # Target 99 not defined
                Instruction(Opcode.HALT),
            ]
        )
        with pytest.raises(TargetNotFound):
            Executor().execute(prog)

    def test_loop_error_next_outside_loop(self):
        """NEXT outside loop raises LoopError."""
        prog = make_program(
            [
                Instruction(Opcode.NEXT),
                Instruction(Opcode.HALT),
            ]
        )
        with pytest.raises(LoopError):
            Executor().execute(prog)

    def test_loop_error_lval_outside_loop(self):
        """LVAL outside loop raises LoopError."""
        prog = make_program(
            [
                Instruction(Opcode.LVAL, (0,)),
                Instruction(Opcode.HALT),
            ]
        )
        with pytest.raises(LoopError):
            Executor().execute(prog)

    def test_xqmx_mode_error(self):
        """HLF on SAMPLE mode raises XQMXModeError."""
        prog = make_program(
            [
                Instruction(Opcode.PUSH1, (5,)),
                Instruction(Opcode.BSMX, (0,)),  # Create SAMPLE
                Instruction(Opcode.PUSH1, (0,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.EXCLUDE, (0,)),  # Requires MODEL
                Instruction(Opcode.HALT),
            ]
        )
        with pytest.raises(XQMXModeError):
            Executor().execute(prog)


class TestTracerIntegration:
    """Tests for tracer protocol integration."""

    def test_tracer_on_step_begin(self):
        """Tracer receives on_step_begin calls."""
        events = []

        class TestTracer:
            def on_step_begin(self, _executor, instr):
                events.append(("begin", instr.opcode))

            def on_step_end(self, _executor, _instr):
                pass

            def on_error(self, _executor, _instr, _error):
                pass

            def on_halt(self, _executor):
                pass

        prog = make_program(
            [
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.HALT),
            ]
        )
        ex = Executor(tracer=TestTracer())
        ex.execute(prog)

        assert ("begin", Opcode.PUSH1) in events
        assert ("begin", Opcode.HALT) in events

    def test_tracer_on_step_end(self):
        """Tracer receives on_step_end calls."""
        events = []

        class TestTracer:
            def on_step_begin(self, _executor, _instr):
                pass

            def on_step_end(self, _executor, instr):
                events.append(("end", instr.opcode))

            def on_error(self, _executor, _instr, _error):
                pass

            def on_halt(self, _executor):
                pass

        prog = make_program(
            [
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.HALT),
            ]
        )
        ex = Executor(tracer=TestTracer())
        ex.execute(prog)

        assert ("end", Opcode.PUSH1) in events

    def test_tracer_on_error(self):
        """Tracer receives on_error calls."""
        errors = []

        class TestTracer:
            def on_step_begin(self, _executor, _instr):
                pass

            def on_step_end(self, _executor, _instr):
                pass

            def on_error(self, _executor, _instr, error):
                errors.append(type(error).__name__)

            def on_halt(self, _executor):
                pass

        prog = make_program(
            [
                Instruction(Opcode.POP),  # Will fail - empty stack
                Instruction(Opcode.HALT),
            ]
        )
        ex = Executor(tracer=TestTracer())

        with pytest.raises(StackUnderflow):
            ex.execute(prog)

        assert "StackUnderflow" in errors

    def test_tracer_on_halt(self):
        """Tracer receives on_halt call."""
        halted = []

        class TestTracer:
            def on_step_begin(self, _executor, _instr):
                pass

            def on_step_end(self, _executor, _instr):
                pass

            def on_error(self, _executor, _instr, _error):
                pass

            def on_halt(self, _executor):
                halted.append(True)

        prog = make_program(
            [
                Instruction(Opcode.HALT),
            ]
        )
        ex = Executor(tracer=TestTracer())
        ex.execute(prog)

        assert halted == [True]


class TestExecutorHelpers:
    """Tests for Executor helper methods."""

    def test_execute_returns_output(self):
        """execute returns output dictionary."""
        prog = make_program(
            [
                Instruction(Opcode.PUSH1, (42,)),
                Instruction(Opcode.STOW, (0,)),
                Instruction(Opcode.PUSH1, (0,)),  # slot index
                Instruction(Opcode.OUTPUT, (0,)),  # r0 -> output[0]
                Instruction(Opcode.HALT),
            ]
        )
        ex = Executor()
        output = ex.execute(prog)
        assert output[0] == 42

    def test_execute_with_input_data(self):
        """execute accepts input data."""
        prog = make_program(
            [
                Instruction(Opcode.PUSH1, (0,)),  # input slot
                Instruction(Opcode.INPUT, (0,)),  # input[0] -> r0
                Instruction(Opcode.PUSH1, (1,)),  # output slot
                Instruction(Opcode.OUTPUT, (0,)),  # r0 -> output[1]
                Instruction(Opcode.HALT),
            ]
        )
        ex = Executor()
        output = ex.execute(prog, input_data={0: "hello"})
        assert output[1] == "hello"

    def test_step_returns_continue_flag(self):
        """step returns True to continue, False to stop."""
        prog = make_program(
            [
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.HALT),
            ]
        )
        ex = Executor()
        ex.program = prog
        ex.state = MachineState()

        assert ex.step() is True  # PUSH - continue
        assert ex.step() is False  # HALT - stop


class TestProgramAndInstruction:
    """Tests for Program and Instruction classes."""

    def test_program_length(self):
        """Program length equals instruction count."""
        prog = make_program(
            [
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.PUSH1, (2,)),
                Instruction(Opcode.HALT),
            ]
        )
        assert len(prog) == 3

    def test_program_getitem(self):
        """Program supports index access."""
        prog = make_program(
            [
                Instruction(Opcode.PUSH1, (1,)),
                Instruction(Opcode.ADD),
                Instruction(Opcode.HALT),
            ]
        )
        assert prog[0].opcode == Opcode.PUSH1
        assert prog[1].opcode == Opcode.ADD
        assert prog[2].opcode == Opcode.HALT

    def test_instruction_operands(self):
        """Instruction stores operands."""
        instr = Instruction(Opcode.PUSH1, (42,))
        assert instr.operands == (42,)

    def test_instruction_no_operands(self):
        """Instruction without operands has empty tuple."""
        instr = Instruction(Opcode.ADD)
        assert instr.operands == ()

    def test_instruction_line_number(self):
        """Instruction stores line number."""
        instr = Instruction(Opcode.NOP, line=10)
        assert instr.line == 10
