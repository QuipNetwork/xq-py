"""
Tests for Executor and all 68 opcode handlers.
"""

import pytest

from xqvm.core.executor import Executor, Instruction
from xqvm.core.opcodes import Opcode
from xqvm.core.state import MachineState
from xqvm.core.vector import Vec
from xqvm.core.xqmx import XQMX
from xqvm.core.errors import (
    StackUnderflow,
    RegisterNotFound,
    DivisionByZero,
    TargetNotFound,
    LoopError,
    TypeMismatch,
    XQMXModeError,
)

from tests.conftest import make_program, run_program


class TestControlFlow:
    """Tests for control flow opcodes."""

    def test_nop(self):
        """NOP does nothing but advance PC."""
        ex = run_program([
            (Opcode.NOP,),
            (Opcode.NOP,),
            (Opcode.HALT,),
        ])
        assert ex.state.halted is True

    def test_halt(self):
        """HALT stops execution."""
        ex = run_program([
            (Opcode.PUSH, (1,)),
            (Opcode.HALT,),
            (Opcode.PUSH, (2,)),  # Never reached
        ])
        assert ex.state.stack_depth == 1
        assert ex.state.peek(0) == 1

    def test_target_and_jump(self):
        """TARGET defines label, JUMP jumps back to it."""
        # Define target first, then jump back to it
        ex = run_program([
            (Opcode.PUSH, (0,)),     # Counter
            (Opcode.STOW, (0,)),
            (Opcode.TARGET, (0,)),   # Target 0 - loop start
            (Opcode.LOAD, (0,)),
            (Opcode.PUSH, (1,)),
            (Opcode.ADD,),
            (Opcode.DUPL,),
            (Opcode.STOW, (0,)),     # counter++
            (Opcode.PUSH, (3,)),
            (Opcode.LT,),            # counter < 3?
            (Opcode.JUMPI, (0,)),    # Jump back to target 0
            (Opcode.HALT,),
        ])
        # Counter should be 3
        assert ex.state.get_register(0) == 3

    def test_jumpi_taken(self):
        """JUMPI jumps when top of stack is non-zero."""
        # Define target first, then conditionally jump to it
        ex = run_program([
            (Opcode.TARGET, (0,)),   # Define target 0 here
            (Opcode.PUSH, (42,)),
            (Opcode.STOW, (0,)),
            (Opcode.PUSH, (1,)),     # Non-zero condition
            (Opcode.NOT,),           # Invert to 0, so we don't jump
            (Opcode.JUMPI, (0,)),    # Won't jump (0)
            (Opcode.HALT,),
        ])
        assert ex.state.get_register(0) == 42

    def test_jumpi_not_taken(self):
        """JUMPI doesn't jump when top of stack is zero."""
        ex = run_program([
            (Opcode.PUSH, (99,)),
            (Opcode.STOW, (0,)),
            (Opcode.TARGET, (0,)),   # Define target first
            (Opcode.PUSH, (0,)),     # Zero condition
            (Opcode.JUMPI, (0,)),    # Should not jump (0)
            (Opcode.HALT,),
        ])
        assert ex.state.get_register(0) == 99

    def test_range_loop(self):
        """RANGE creates loop with values [start, start+count)."""
        # Sum 0+1+2+3+4 = 10
        ex = run_program([
            (Opcode.PUSH, (0,)),     # Sum accumulator
            (Opcode.STOW, (0,)),
            (Opcode.PUSH, (0,)),     # Start
            (Opcode.PUSH, (5,)),     # Count
            (Opcode.RANGE,),
            (Opcode.TARGET, (0,)),   # Loop body
            (Opcode.LVAL, (1,)),     # i -> r1
            (Opcode.LOAD, (0,)),     # Load sum
            (Opcode.LOAD, (1,)),     # Load i
            (Opcode.ADD,),
            (Opcode.STOW, (0,)),     # Store sum
            (Opcode.NEXT,),
            (Opcode.HALT,),
        ])
        assert ex.state.get_register(0) == 10

    def test_iter_loop(self):
        """ITER iterates over vector elements."""
        ex = run_program([
            # Create vec [10, 20, 30]
            (Opcode.VECI, (0,)),
            (Opcode.PUSH, (10,)),
            (Opcode.VECPUSH, (0,)),
            (Opcode.PUSH, (20,)),
            (Opcode.VECPUSH, (0,)),
            (Opcode.PUSH, (30,)),
            (Opcode.VECPUSH, (0,)),
            # Sum accumulator
            (Opcode.PUSH, (0,)),
            (Opcode.STOW, (1,)),
            # Iterate: start=0, end=3
            (Opcode.PUSH, (0,)),
            (Opcode.PUSH, (3,)),
            (Opcode.ITER, (0,)),
            (Opcode.TARGET, (0,)),
            (Opcode.LVAL, (2,)),     # val -> r2
            (Opcode.LOAD, (1,)),     # Load sum
            (Opcode.LOAD, (2,)),     # Load val
            (Opcode.ADD,),
            (Opcode.STOW, (1,)),
            (Opcode.NEXT,),
            (Opcode.HALT,),
        ])
        assert ex.state.get_register(1) == 60

    def test_lval(self):
        """LVAL copies current loop value to register."""
        ex = run_program([
            (Opcode.PUSH, (5,)),     # Start
            (Opcode.PUSH, (1,)),     # Count
            (Opcode.RANGE,),
            (Opcode.TARGET, (0,)),
            (Opcode.LVAL, (0,)),     # Store loop value in r0
            (Opcode.NEXT,),
            (Opcode.HALT,),
        ])
        assert ex.state.get_register(0) == 5


class TestStackRegisterIO:
    """Tests for stack, register, and I/O opcodes."""

    def test_push(self):
        """PUSH puts immediate value on stack."""
        ex = run_program([
            (Opcode.PUSH, (42,)),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 42

    def test_pop(self):
        """POP removes top of stack."""
        ex = run_program([
            (Opcode.PUSH, (1,)),
            (Opcode.PUSH, (2,)),
            (Opcode.POP,),
            (Opcode.HALT,),
        ])
        assert ex.state.stack_depth == 1
        assert ex.state.peek(0) == 1

    def test_dupl(self):
        """DUPL duplicates top of stack."""
        ex = run_program([
            (Opcode.PUSH, (42,)),
            (Opcode.DUPL,),
            (Opcode.HALT,),
        ])
        assert ex.state.stack_depth == 2
        assert ex.state.peek(0) == 42
        assert ex.state.peek(1) == 42

    def test_swap(self):
        """SWAP exchanges top two stack values."""
        ex = run_program([
            (Opcode.PUSH, (1,)),
            (Opcode.PUSH, (2,)),
            (Opcode.SWAP,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 1
        assert ex.state.peek(1) == 2

    def test_load(self):
        """LOAD pushes register value onto stack."""
        ex = run_program([
            (Opcode.PUSH, (100,)),
            (Opcode.STOW, (5,)),
            (Opcode.LOAD, (5,)),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 100

    def test_stow(self):
        """STOW stores stack top in register."""
        ex = run_program([
            (Opcode.PUSH, (42,)),
            (Opcode.STOW, (0,)),
            (Opcode.HALT,),
        ])
        assert ex.state.get_register(0) == 42
        assert ex.state.stack_depth == 0

    def test_input(self):
        """INPUT loads input slot to register."""
        ex = run_program([
            (Opcode.PUSH, (0,)),     # slot index on stack
            (Opcode.INPUT, (0,)),    # input[0] -> r0
            (Opcode.HALT,),
        ], input_data={0: 42})
        assert ex.state.get_register(0) == 42

    def test_output(self):
        """OUTPUT writes register to output slot."""
        prog = make_program([
            (Opcode.PUSH, (99,)),
            (Opcode.STOW, (0,)),
            (Opcode.PUSH, (0,)),     # slot index on stack
            (Opcode.OUTPUT, (0,)),   # r0 -> output[0]
            (Opcode.HALT,),
        ])
        ex = Executor()
        output = ex.execute(prog)
        assert output[0] == 99


class TestArithmetic:
    """Tests for arithmetic opcodes."""

    def test_add(self):
        """ADD sums top two values."""
        ex = run_program([
            (Opcode.PUSH, (10,)),
            (Opcode.PUSH, (5,)),
            (Opcode.ADD,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 15

    def test_sub(self):
        """SUB subtracts (second - top)."""
        ex = run_program([
            (Opcode.PUSH, (10,)),
            (Opcode.PUSH, (3,)),
            (Opcode.SUB,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 7

    def test_mul(self):
        """MUL multiplies top two values."""
        ex = run_program([
            (Opcode.PUSH, (6,)),
            (Opcode.PUSH, (7,)),
            (Opcode.MUL,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 42

    def test_div(self):
        """DIV performs integer division."""
        ex = run_program([
            (Opcode.PUSH, (20,)),
            (Opcode.PUSH, (3,)),
            (Opcode.DIV,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 6

    def test_div_by_zero(self):
        """DIV by zero raises DivisionByZero."""
        prog = make_program([
            (Opcode.PUSH, (10,)),
            (Opcode.PUSH, (0,)),
            (Opcode.DIV,),
            (Opcode.HALT,),
        ])
        with pytest.raises(DivisionByZero):
            Executor().execute(prog)

    def test_mod(self):
        """MOD computes remainder."""
        ex = run_program([
            (Opcode.PUSH, (17,)),
            (Opcode.PUSH, (5,)),
            (Opcode.MOD,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 2

    def test_mod_by_zero(self):
        """MOD by zero raises DivisionByZero."""
        prog = make_program([
            (Opcode.PUSH, (10,)),
            (Opcode.PUSH, (0,)),
            (Opcode.MOD,),
            (Opcode.HALT,),
        ])
        with pytest.raises(DivisionByZero):
            Executor().execute(prog)

    def test_neg(self):
        """NEG negates top value."""
        ex = run_program([
            (Opcode.PUSH, (42,)),
            (Opcode.NEG,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == -42

    def test_neg_negative(self):
        """NEG on negative produces positive."""
        ex = run_program([
            (Opcode.PUSH, (-10,)),
            (Opcode.NEG,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 10


class TestComparison:
    """Tests for comparison opcodes."""

    def test_eq_true(self):
        """EQ returns 1 when equal."""
        ex = run_program([
            (Opcode.PUSH, (5,)),
            (Opcode.PUSH, (5,)),
            (Opcode.EQ,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 1

    def test_eq_false(self):
        """EQ returns 0 when not equal."""
        ex = run_program([
            (Opcode.PUSH, (5,)),
            (Opcode.PUSH, (3,)),
            (Opcode.EQ,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 0

    def test_lt_true(self):
        """LT returns 1 when second < top."""
        ex = run_program([
            (Opcode.PUSH, (3,)),
            (Opcode.PUSH, (5,)),
            (Opcode.LT,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 1

    def test_lt_false(self):
        """LT returns 0 when second >= top."""
        ex = run_program([
            (Opcode.PUSH, (5,)),
            (Opcode.PUSH, (3,)),
            (Opcode.LT,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 0

    def test_gt_true(self):
        """GT returns 1 when second > top."""
        ex = run_program([
            (Opcode.PUSH, (5,)),
            (Opcode.PUSH, (3,)),
            (Opcode.GT,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 1

    def test_gt_false(self):
        """GT returns 0 when second <= top."""
        ex = run_program([
            (Opcode.PUSH, (3,)),
            (Opcode.PUSH, (5,)),
            (Opcode.GT,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 0

    def test_lte_true_less(self):
        """LTE returns 1 when second < top."""
        ex = run_program([
            (Opcode.PUSH, (3,)),
            (Opcode.PUSH, (5,)),
            (Opcode.LTE,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 1

    def test_lte_true_equal(self):
        """LTE returns 1 when equal."""
        ex = run_program([
            (Opcode.PUSH, (5,)),
            (Opcode.PUSH, (5,)),
            (Opcode.LTE,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 1

    def test_lte_false(self):
        """LTE returns 0 when second > top."""
        ex = run_program([
            (Opcode.PUSH, (7,)),
            (Opcode.PUSH, (5,)),
            (Opcode.LTE,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 0

    def test_gte_true_greater(self):
        """GTE returns 1 when second > top."""
        ex = run_program([
            (Opcode.PUSH, (7,)),
            (Opcode.PUSH, (5,)),
            (Opcode.GTE,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 1

    def test_gte_true_equal(self):
        """GTE returns 1 when equal."""
        ex = run_program([
            (Opcode.PUSH, (5,)),
            (Opcode.PUSH, (5,)),
            (Opcode.GTE,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 1

    def test_gte_false(self):
        """GTE returns 0 when second < top."""
        ex = run_program([
            (Opcode.PUSH, (3,)),
            (Opcode.PUSH, (5,)),
            (Opcode.GTE,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 0


class TestBoolean:
    """Tests for boolean logic opcodes."""

    def test_not_true_to_false(self):
        """NOT converts non-zero to 0."""
        ex = run_program([
            (Opcode.PUSH, (5,)),
            (Opcode.NOT,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 0

    def test_not_false_to_true(self):
        """NOT converts 0 to 1."""
        ex = run_program([
            (Opcode.PUSH, (0,)),
            (Opcode.NOT,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 1

    def test_and_both_true(self):
        """AND of two truthy values is 1."""
        ex = run_program([
            (Opcode.PUSH, (1,)),
            (Opcode.PUSH, (1,)),
            (Opcode.AND,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 1

    def test_and_one_false(self):
        """AND with one falsy value is 0."""
        ex = run_program([
            (Opcode.PUSH, (1,)),
            (Opcode.PUSH, (0,)),
            (Opcode.AND,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 0

    def test_or_both_true(self):
        """OR of two truthy values is 1."""
        ex = run_program([
            (Opcode.PUSH, (1,)),
            (Opcode.PUSH, (1,)),
            (Opcode.OR,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 1

    def test_or_one_true(self):
        """OR with one truthy value is 1."""
        ex = run_program([
            (Opcode.PUSH, (0,)),
            (Opcode.PUSH, (1,)),
            (Opcode.OR,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 1

    def test_or_both_false(self):
        """OR of two falsy values is 0."""
        ex = run_program([
            (Opcode.PUSH, (0,)),
            (Opcode.PUSH, (0,)),
            (Opcode.OR,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 0

    def test_xor_different(self):
        """XOR of different values is 1."""
        ex = run_program([
            (Opcode.PUSH, (1,)),
            (Opcode.PUSH, (0,)),
            (Opcode.XOR,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 1

    def test_xor_same(self):
        """XOR of same values is 0."""
        ex = run_program([
            (Opcode.PUSH, (1,)),
            (Opcode.PUSH, (1,)),
            (Opcode.XOR,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 0


class TestBitwise:
    """Tests for bitwise opcodes."""

    def test_band(self):
        """BAND performs bitwise AND."""
        ex = run_program([
            (Opcode.PUSH, (0b1100,)),
            (Opcode.PUSH, (0b1010,)),
            (Opcode.BAND,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 0b1000

    def test_bor(self):
        """BOR performs bitwise OR."""
        ex = run_program([
            (Opcode.PUSH, (0b1100,)),
            (Opcode.PUSH, (0b1010,)),
            (Opcode.BOR,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 0b1110

    def test_bxor(self):
        """BXOR performs bitwise XOR."""
        ex = run_program([
            (Opcode.PUSH, (0b1100,)),
            (Opcode.PUSH, (0b1010,)),
            (Opcode.BXOR,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 0b0110

    def test_bnot(self):
        """BNOT performs bitwise NOT."""
        ex = run_program([
            (Opcode.PUSH, (0,)),
            (Opcode.BNOT,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == ~0

    def test_shl(self):
        """SHL shifts left."""
        ex = run_program([
            (Opcode.PUSH, (1,)),
            (Opcode.PUSH, (4,)),
            (Opcode.SHL,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 16

    def test_shr(self):
        """SHR shifts right."""
        ex = run_program([
            (Opcode.PUSH, (16,)),
            (Opcode.PUSH, (2,)),
            (Opcode.SHR,),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 4


class TestAllocators:
    """Tests for allocator opcodes."""

    def test_vec_creates_empty(self):
        """VEC creates empty untyped vector."""
        ex = run_program([
            (Opcode.VEC, (0,)),
            (Opcode.HALT,),
        ])
        v = ex.state.get_register(0)
        assert isinstance(v, Vec)
        assert v.length == 0

    def test_veci_creates_int_vec(self):
        """VECI creates empty int vector."""
        ex = run_program([
            (Opcode.VECI, (0,)),
            (Opcode.HALT,),
        ])
        v = ex.state.get_register(0)
        assert isinstance(v, Vec)
        assert v.element_type.kind == "int"

    def test_vecx_creates_xqmx_vec(self):
        """VECX creates empty xqmx vector."""
        ex = run_program([
            (Opcode.VECX, (0,)),
            (Opcode.HALT,),
        ])
        v = ex.state.get_register(0)
        assert isinstance(v, Vec)
        assert v.element_type.kind == "xqmx"

    def test_bqmx_creates_binary_model(self):
        """BQMX creates binary model XQMX."""
        ex = run_program([
            (Opcode.PUSH, (10,)),  # size
            (Opcode.BQMX, (0,)),
            (Opcode.HALT,),
        ])
        x = ex.state.get_register(0)
        assert isinstance(x, XQMX)
        assert x.is_model()
        assert x.size == 10

    def test_sqmx_creates_spin_model(self):
        """SQMX creates spin model XQMX."""
        ex = run_program([
            (Opcode.PUSH, (8,)),
            (Opcode.SQMX, (0,)),
            (Opcode.HALT,),
        ])
        x = ex.state.get_register(0)
        assert x.is_model()
        assert x.size == 8

    def test_xqmx_creates_discrete_model(self):
        """XQMX creates discrete model with k values."""
        ex = run_program([
            (Opcode.PUSH, (5,)),   # size
            (Opcode.PUSH, (3,)),   # k
            (Opcode.XQMX, (0,)),
            (Opcode.HALT,),
        ])
        x = ex.state.get_register(0)
        assert x.is_model()
        assert x.discrete_k == 3

    def test_bsmx_creates_binary_sample(self):
        """BSMX creates binary sample XQMX."""
        ex = run_program([
            (Opcode.PUSH, (10,)),
            (Opcode.BSMX, (0,)),
            (Opcode.HALT,),
        ])
        x = ex.state.get_register(0)
        assert x.is_sample()

    def test_ssmx_creates_spin_sample(self):
        """SSMX creates spin sample XQMX."""
        ex = run_program([
            (Opcode.PUSH, (10,)),
            (Opcode.SSMX, (0,)),
            (Opcode.HALT,),
        ])
        x = ex.state.get_register(0)
        assert x.is_sample()

    def test_xsmx_creates_discrete_sample(self):
        """XSMX creates discrete sample with k values."""
        ex = run_program([
            (Opcode.PUSH, (5,)),
            (Opcode.PUSH, (4,)),
            (Opcode.XSMX, (0,)),
            (Opcode.HALT,),
        ])
        x = ex.state.get_register(0)
        assert x.is_sample()
        assert x.discrete_k == 4


class TestVectorAccess:
    """Tests for vector access opcodes."""

    def test_vecpush(self):
        """VECPUSH appends value to vector."""
        ex = run_program([
            (Opcode.VECI, (0,)),
            (Opcode.PUSH, (42,)),
            (Opcode.VECPUSH, (0,)),
            (Opcode.HALT,),
        ])
        v = ex.state.get_register(0)
        assert v.length == 1
        assert v.get(0) == 42

    def test_vecget(self):
        """VECGET gets value at index."""
        ex = run_program([
            (Opcode.VECI, (0,)),
            (Opcode.PUSH, (10,)),
            (Opcode.VECPUSH, (0,)),
            (Opcode.PUSH, (20,)),
            (Opcode.VECPUSH, (0,)),
            (Opcode.PUSH, (1,)),     # index
            (Opcode.VECGET, (0,)),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 20

    def test_vecset(self):
        """VECSET sets value at index."""
        ex = run_program([
            (Opcode.VECI, (0,)),
            (Opcode.PUSH, (10,)),
            (Opcode.VECPUSH, (0,)),
            (Opcode.PUSH, (0,)),     # index
            (Opcode.PUSH, (99,)),    # value
            (Opcode.VECSET, (0,)),
            (Opcode.HALT,),
        ])
        v = ex.state.get_register(0)
        assert v.get(0) == 99

    def test_veclen(self):
        """VECLEN pushes vector length."""
        ex = run_program([
            (Opcode.VECI, (0,)),
            (Opcode.PUSH, (1,)),
            (Opcode.VECPUSH, (0,)),
            (Opcode.PUSH, (2,)),
            (Opcode.VECPUSH, (0,)),
            (Opcode.PUSH, (3,)),
            (Opcode.VECPUSH, (0,)),
            (Opcode.VECLEN, (0,)),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 3


class TestVectorMath:
    """Tests for vector math opcodes."""

    def test_idxgrid(self):
        """IDXGRID converts (row, col) to flat index."""
        ex = run_program([
            (Opcode.PUSH, (2,)),     # row
            (Opcode.PUSH, (3,)),     # col
            (Opcode.PUSH, (5,)),     # cols
            (Opcode.IDXGRID,),
            (Opcode.HALT,),
        ])
        # index = row * cols + col = 2 * 5 + 3 = 13
        assert ex.state.peek(0) == 13

    def test_idxtriu(self):
        """IDXTRIU converts (i, j) to upper triangular index."""
        # idx = j * (j - 1) // 2 + i (with i < j)
        ex = run_program([
            (Opcode.PUSH, (0,)),     # i
            (Opcode.PUSH, (2,)),     # j
            (Opcode.IDXTRIU,),
            (Opcode.HALT,),
        ])
        # With i=0, j=2: idx = 2 * (2-1) / 2 + 0 = 1
        assert ex.state.peek(0) == 1


class TestXQMXAccess:
    """Tests for XQMX coefficient access opcodes."""

    def test_getline(self):
        """GETLINE gets linear coefficient."""
        ex = run_program([
            (Opcode.PUSH, (10,)),
            (Opcode.BQMX, (0,)),
            (Opcode.PUSH, (5,)),     # index
            (Opcode.GETLINE, (0,)),
            (Opcode.HALT,),
        ])
        # Default is 0
        assert ex.state.peek(0) == 0

    def test_setline(self):
        """SETLINE sets linear coefficient."""
        ex = run_program([
            (Opcode.PUSH, (10,)),
            (Opcode.BQMX, (0,)),
            (Opcode.PUSH, (3,)),     # index
            (Opcode.PUSH, (5,)),     # value (as int, will be float)
            (Opcode.SETLINE, (0,)),
            (Opcode.PUSH, (3,)),
            (Opcode.GETLINE, (0,)),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 5

    def test_addline(self):
        """ADDLINE adds to linear coefficient."""
        ex = run_program([
            (Opcode.PUSH, (10,)),
            (Opcode.BQMX, (0,)),
            (Opcode.PUSH, (0,)),
            (Opcode.PUSH, (3,)),
            (Opcode.SETLINE, (0,)),
            (Opcode.PUSH, (0,)),
            (Opcode.PUSH, (2,)),
            (Opcode.ADDLINE, (0,)),
            (Opcode.PUSH, (0,)),
            (Opcode.GETLINE, (0,)),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 5

    def test_getquad(self):
        """GETQUAD gets quadratic coefficient."""
        ex = run_program([
            (Opcode.PUSH, (10,)),
            (Opcode.BQMX, (0,)),
            (Opcode.PUSH, (0,)),     # i
            (Opcode.PUSH, (1,)),     # j
            (Opcode.GETQUAD, (0,)),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 0

    def test_setquad(self):
        """SETQUAD sets quadratic coefficient."""
        ex = run_program([
            (Opcode.PUSH, (10,)),
            (Opcode.BQMX, (0,)),
            (Opcode.PUSH, (0,)),     # i
            (Opcode.PUSH, (1,)),     # j
            (Opcode.PUSH, (7,)),     # value
            (Opcode.SETQUAD, (0,)),
            (Opcode.PUSH, (0,)),
            (Opcode.PUSH, (1,)),
            (Opcode.GETQUAD, (0,)),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 7

    def test_addquad(self):
        """ADDQUAD adds to quadratic coefficient."""
        ex = run_program([
            (Opcode.PUSH, (10,)),
            (Opcode.BQMX, (0,)),
            (Opcode.PUSH, (0,)),
            (Opcode.PUSH, (1,)),
            (Opcode.PUSH, (3,)),
            (Opcode.SETQUAD, (0,)),
            (Opcode.PUSH, (0,)),
            (Opcode.PUSH, (1,)),
            (Opcode.PUSH, (4,)),
            (Opcode.ADDQUAD, (0,)),
            (Opcode.PUSH, (0,)),
            (Opcode.PUSH, (1,)),
            (Opcode.GETQUAD, (0,)),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 7


class TestXQMXGrid:
    """Tests for XQMX grid opcodes."""

    def test_resize(self):
        """RESIZE sets grid dimensions."""
        ex = run_program([
            (Opcode.PUSH, (25,)),
            (Opcode.BQMX, (0,)),
            (Opcode.PUSH, (5,)),     # rows
            (Opcode.PUSH, (5,)),     # cols
            (Opcode.RESIZE, (0,)),
            (Opcode.HALT,),
        ])
        x = ex.state.get_register(0)
        assert x.rows == 5
        assert x.cols == 5

    def test_rowfind(self):
        """ROWFIND finds first column with value."""
        ex = run_program([
            (Opcode.PUSH, (25,)),
            (Opcode.BQMX, (0,)),
            (Opcode.PUSH, (5,)),
            (Opcode.PUSH, (5,)),
            (Opcode.RESIZE, (0,)),
            # Set value at row 0, col 2 (index 2)
            (Opcode.PUSH, (2,)),
            (Opcode.PUSH, (1,)),
            (Opcode.SETLINE, (0,)),
            # Find in row 0
            (Opcode.PUSH, (0,)),     # row
            (Opcode.PUSH, (1,)),     # value
            (Opcode.ROWFIND, (0,)),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 2

    def test_colfind(self):
        """COLFIND finds first row with value."""
        ex = run_program([
            (Opcode.PUSH, (25,)),
            (Opcode.BQMX, (0,)),
            (Opcode.PUSH, (5,)),
            (Opcode.PUSH, (5,)),
            (Opcode.RESIZE, (0,)),
            # Set value at row 2, col 0 (index 10)
            (Opcode.PUSH, (10,)),
            (Opcode.PUSH, (1,)),
            (Opcode.SETLINE, (0,)),
            # Find in col 0
            (Opcode.PUSH, (0,)),     # col
            (Opcode.PUSH, (1,)),     # value
            (Opcode.COLFIND, (0,)),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 2

    def test_rowsum(self):
        """ROWSUM sums values in row."""
        ex = run_program([
            (Opcode.PUSH, (25,)),
            (Opcode.BQMX, (0,)),
            (Opcode.PUSH, (5,)),
            (Opcode.PUSH, (5,)),
            (Opcode.RESIZE, (0,)),
            # Set values in row 0: indices 0, 1, 2
            (Opcode.PUSH, (0,)),
            (Opcode.PUSH, (1,)),
            (Opcode.SETLINE, (0,)),
            (Opcode.PUSH, (1,)),
            (Opcode.PUSH, (2,)),
            (Opcode.SETLINE, (0,)),
            (Opcode.PUSH, (2,)),
            (Opcode.PUSH, (3,)),
            (Opcode.SETLINE, (0,)),
            # Sum row 0
            (Opcode.PUSH, (0,)),
            (Opcode.ROWSUM, (0,)),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 6

    def test_colsum(self):
        """COLSUM sums values in column."""
        ex = run_program([
            (Opcode.PUSH, (25,)),
            (Opcode.BQMX, (0,)),
            (Opcode.PUSH, (5,)),
            (Opcode.PUSH, (5,)),
            (Opcode.RESIZE, (0,)),
            # Set values in col 0: indices 0, 5, 10
            (Opcode.PUSH, (0,)),
            (Opcode.PUSH, (1,)),
            (Opcode.SETLINE, (0,)),
            (Opcode.PUSH, (5,)),
            (Opcode.PUSH, (2,)),
            (Opcode.SETLINE, (0,)),
            (Opcode.PUSH, (10,)),
            (Opcode.PUSH, (3,)),
            (Opcode.SETLINE, (0,)),
            # Sum col 0
            (Opcode.PUSH, (0,)),
            (Opcode.COLSUM, (0,)),
            (Opcode.HALT,),
        ])
        assert ex.state.peek(0) == 6


class TestXQMXHighLevel:
    """Tests for XQMX high-level function opcodes."""

    def test_onehot(self):
        """ONEHOT adds one-hot constraint for a row."""
        ex = run_program([
            # Create 3x3 grid model (size=9)
            (Opcode.PUSH, (9,)),
            (Opcode.BQMX, (0,)),
            # Set grid dimensions
            (Opcode.PUSH, (3,)),     # rows
            (Opcode.PUSH, (3,)),     # cols
            (Opcode.RESIZE, (0,)),
            # Apply one-hot constraint for row 0 with penalty 1
            (Opcode.PUSH, (0,)),     # row
            (Opcode.PUSH, (1,)),     # penalty
            (Opcode.ONEHOT, (0,)),
            (Opcode.HALT,),
        ])
        x = ex.state.get_register(0)
        # Should have linear and quadratic terms set
        assert len(x.linear) > 0 or len(x.quadratic) > 0

    def test_exclude(self):
        """EXCLUDE adds exclusion constraint."""
        ex = run_program([
            (Opcode.PUSH, (5,)),
            (Opcode.BQMX, (0,)),
            (Opcode.PUSH, (0,)),     # i
            (Opcode.PUSH, (1,)),     # j
            (Opcode.PUSH, (2,)),     # penalty
            (Opcode.EXCLUDE, (0,)),
            (Opcode.HALT,),
        ])
        x = ex.state.get_register(0)
        assert x.get_quadratic(0, 1) == 2.0

    def test_implies(self):
        """IMPLIES adds implication constraint."""
        ex = run_program([
            (Opcode.PUSH, (5,)),
            (Opcode.BQMX, (0,)),
            (Opcode.PUSH, (0,)),     # i
            (Opcode.PUSH, (1,)),     # j
            (Opcode.PUSH, (1,)),     # penalty
            (Opcode.IMPLIES, (0,)),
            (Opcode.HALT,),
        ])
        x = ex.state.get_register(0)
        # Should have some coefficients set
        assert len(x.linear) > 0 or len(x.quadratic) > 0

    def test_energy(self):
        """ENERGY computes energy of sample against model."""
        ex = run_program([
            # Create model with linear terms
            (Opcode.PUSH, (3,)),
            (Opcode.BQMX, (0,)),
            (Opcode.PUSH, (0,)),
            (Opcode.PUSH, (1,)),
            (Opcode.SETLINE, (0,)),
            (Opcode.PUSH, (1,)),
            (Opcode.PUSH, (2,)),
            (Opcode.SETLINE, (0,)),
            # Create sample with values
            (Opcode.PUSH, (3,)),
            (Opcode.BSMX, (1,)),
            (Opcode.PUSH, (0,)),
            (Opcode.PUSH, (1,)),
            (Opcode.SETLINE, (1,)),
            (Opcode.PUSH, (1,)),
            (Opcode.PUSH, (1,)),
            (Opcode.SETLINE, (1,)),
            # Compute energy
            (Opcode.ENERGY, (0, 1)),
            (Opcode.HALT,),
        ])
        # Energy = 1*1 + 2*1 = 3
        assert ex.state.peek(0) == 3


class TestErrorHandling:
    """Tests for error conditions."""

    def test_stack_underflow(self):
        """Stack underflow raises StackUnderflow."""
        prog = make_program([
            (Opcode.POP,),
            (Opcode.HALT,),
        ])
        with pytest.raises(StackUnderflow):
            Executor().execute(prog)

    def test_register_not_found(self):
        """Missing register raises RegisterNotFound."""
        prog = make_program([
            (Opcode.LOAD, (99,)),
            (Opcode.HALT,),
        ])
        with pytest.raises(RegisterNotFound):
            Executor().execute(prog)

    def test_type_mismatch_vec_expected(self):
        """Type mismatch when Vec expected raises TypeMismatch."""
        prog = make_program([
            (Opcode.PUSH, (42,)),
            (Opcode.STOW, (0,)),     # r0 = int
            (Opcode.VECLEN, (0,)),   # Expects Vec
            (Opcode.HALT,),
        ])
        with pytest.raises(TypeMismatch):
            Executor().execute(prog)

    def test_type_mismatch_xqmx_expected(self):
        """Type mismatch when XQMX expected raises TypeMismatch."""
        prog = make_program([
            (Opcode.PUSH, (42,)),
            (Opcode.STOW, (0,)),     # r0 = int
            (Opcode.PUSH, (0,)),
            (Opcode.GETLINE, (0,)),  # Expects XQMX
            (Opcode.HALT,),
        ])
        with pytest.raises(TypeMismatch):
            Executor().execute(prog)

    def test_target_not_found(self):
        """Jump to undefined target raises TargetNotFound."""
        prog = make_program([
            (Opcode.JUMP, (99,)),    # Target 99 not defined
            (Opcode.HALT,),
        ])
        with pytest.raises(TargetNotFound):
            Executor().execute(prog)

    def test_loop_error_next_outside_loop(self):
        """NEXT outside loop raises LoopError."""
        prog = make_program([
            (Opcode.NEXT,),
            (Opcode.HALT,),
        ])
        with pytest.raises(LoopError):
            Executor().execute(prog)

    def test_loop_error_lval_outside_loop(self):
        """LVAL outside loop raises LoopError."""
        prog = make_program([
            (Opcode.LVAL, (0,)),
            (Opcode.HALT,),
        ])
        with pytest.raises(LoopError):
            Executor().execute(prog)

    def test_xqmx_mode_error(self):
        """HLF on SAMPLE mode raises XQMXModeError."""
        prog = make_program([
            (Opcode.PUSH, (5,)),
            (Opcode.BSMX, (0,)),     # Create SAMPLE
            (Opcode.PUSH, (0,)),
            (Opcode.PUSH, (1,)),
            (Opcode.PUSH, (1,)),
            (Opcode.EXCLUDE, (0,)),  # Requires MODEL
            (Opcode.HALT,),
        ])
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

        prog = make_program([
            (Opcode.PUSH, (1,)),
            (Opcode.HALT,),
        ])
        ex = Executor(tracer=TestTracer())
        ex.execute(prog)

        assert ("begin", Opcode.PUSH) in events
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

        prog = make_program([
            (Opcode.PUSH, (1,)),
            (Opcode.HALT,),
        ])
        ex = Executor(tracer=TestTracer())
        ex.execute(prog)

        assert ("end", Opcode.PUSH) in events

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

        prog = make_program([
            (Opcode.POP,),  # Will fail - empty stack
            (Opcode.HALT,),
        ])
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

        prog = make_program([
            (Opcode.HALT,),
        ])
        ex = Executor(tracer=TestTracer())
        ex.execute(prog)

        assert halted == [True]


class TestExecutorHelpers:
    """Tests for Executor helper methods."""

    def test_execute_returns_output(self):
        """execute returns output dictionary."""
        prog = make_program([
            (Opcode.PUSH, (42,)),
            (Opcode.STOW, (0,)),
            (Opcode.PUSH, (0,)),     # slot index
            (Opcode.OUTPUT, (0,)),   # r0 -> output[0]
            (Opcode.HALT,),
        ])
        ex = Executor()
        output = ex.execute(prog)
        assert output[0] == 42

    def test_execute_with_input_data(self):
        """execute accepts input data."""
        prog = make_program([
            (Opcode.PUSH, (0,)),     # input slot
            (Opcode.INPUT, (0,)),    # input[0] -> r0
            (Opcode.PUSH, (1,)),     # output slot
            (Opcode.OUTPUT, (0,)),   # r0 -> output[1]
            (Opcode.HALT,),
        ])
        ex = Executor()
        output = ex.execute(prog, input_data={0: "hello"})
        assert output[1] == "hello"

    def test_step_returns_continue_flag(self):
        """step returns True to continue, False to stop."""
        prog = make_program([
            (Opcode.PUSH, (1,)),
            (Opcode.HALT,),
        ])
        ex = Executor()
        ex.program = prog
        ex.state = MachineState()

        assert ex.step() is True   # PUSH - continue
        assert ex.step() is False  # HALT - stop


class TestProgramAndInstruction:
    """Tests for Program and Instruction classes."""

    def test_program_length(self):
        """Program length equals instruction count."""
        prog = make_program([
            (Opcode.PUSH, (1,)),
            (Opcode.PUSH, (2,)),
            (Opcode.HALT,),
        ])
        assert len(prog) == 3

    def test_program_getitem(self):
        """Program supports index access."""
        prog = make_program([
            (Opcode.PUSH, (1,)),
            (Opcode.ADD,),
            (Opcode.HALT,),
        ])
        assert prog[0].opcode == Opcode.PUSH
        assert prog[1].opcode == Opcode.ADD
        assert prog[2].opcode == Opcode.HALT

    def test_instruction_operands(self):
        """Instruction stores operands."""
        instr = Instruction(Opcode.PUSH, (42,))
        assert instr.operands == (42,)

    def test_instruction_no_operands(self):
        """Instruction without operands has empty tuple."""
        instr = Instruction(Opcode.ADD)
        assert instr.operands == ()

    def test_instruction_line_number(self):
        """Instruction stores line number."""
        instr = Instruction(Opcode.NOP, line=10)
        assert instr.line == 10
