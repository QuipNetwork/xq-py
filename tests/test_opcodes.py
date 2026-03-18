"""
Tests for Opcode enum and metadata.
"""

import pytest

from xqvm.core.opcodes import Opcode, OpcodeMeta, OperandType

class TestOpcodeCount:
    """Tests for opcode count and completeness."""

    def test_total_opcode_count(self):
        """Should have exactly 69 opcodes."""
        assert len(Opcode) == 69

    def test_all_opcodes_have_metadata(self):
        """Every opcode must have valid metadata."""
        for op in Opcode:
            meta = op.meta
            assert isinstance(meta, OpcodeMeta)
            assert isinstance(meta.code, int)
            assert isinstance(meta.description, str)
            assert len(meta.description) > 0

class TestOpcodeCodeLookup:
    """Tests for from_code lookup."""

    def test_from_code_valid(self):
        """from_code returns correct opcode for valid codes."""
        assert Opcode.from_code(0x00) == Opcode.NOP
        assert Opcode.from_code(0x0F) == Opcode.HALT
        assert Opcode.from_code(0x10) == Opcode.PUSH
        assert Opcode.from_code(0x20) == Opcode.ADD

    def test_from_code_invalid(self):
        """from_code returns None for invalid codes."""
        assert Opcode.from_code(0xFF) is None
        assert Opcode.from_code(0x99) is None
        assert Opcode.from_code(-1) is None

    def test_from_code_all_opcodes(self):
        """Every opcode can be looked up by its code."""
        for op in Opcode:
            result = Opcode.from_code(op.code)
            assert result == op

class TestOpcodeNameLookup:
    """Tests for from_name lookup."""

    def test_from_name_exact(self):
        """from_name finds exact name match."""
        assert Opcode.from_name("NOP") == Opcode.NOP
        assert Opcode.from_name("HALT") == Opcode.HALT
        assert Opcode.from_name("PUSH") == Opcode.PUSH

    def test_from_name_case_insensitive(self):
        """from_name is case-insensitive."""
        assert Opcode.from_name("nop") == Opcode.NOP
        assert Opcode.from_name("Halt") == Opcode.HALT
        assert Opcode.from_name("pUsH") == Opcode.PUSH

    def test_from_name_invalid(self):
        """from_name returns None for invalid names."""
        assert Opcode.from_name("INVALID") is None
        assert Opcode.from_name("") is None
        assert Opcode.from_name("NOTANOP") is None

class TestOpcodeMetadata:
    """Tests for opcode metadata validity."""

    def test_stack_effects_non_negative(self):
        """Stack pop/push counts must be non-negative."""
        for op in Opcode:
            meta = op.meta
            assert meta.stack_pop >= 0, f"{op.name} has negative stack_pop"
            assert meta.stack_push >= 0, f"{op.name} has negative stack_push"

    def test_operand_count_matches_types(self):
        """operand_count must match operand_types length."""
        for op in Opcode:
            meta = op.meta
            assert meta.operand_count == len(meta.operand_types), (
                f"{op.name}: operand_count={meta.operand_count} "
                f"but operand_types has {len(meta.operand_types)} items"
            )

    def test_operand_types_are_valid(self):
        """All operand types must be valid OperandType enum values."""
        for op in Opcode:
            meta = op.meta
            for t in meta.operand_types:
                assert isinstance(t, OperandType), (
                    f"{op.name} has invalid operand type: {t}"
                )

class TestCodeUniqueness:
    """Tests for opcode code uniqueness."""

    def test_no_duplicate_codes(self):
        """Each opcode must have a unique code."""
        codes = [op.code for op in Opcode]
        assert len(codes) == len(set(codes)), "Duplicate opcode codes found"

    def test_no_duplicate_names(self):
        """Each opcode must have a unique name."""
        names = [op.name for op in Opcode]
        assert len(names) == len(set(names)), "Duplicate opcode names found"

class TestOpcodeGroupRanges:
    """Tests for opcode code ranges by group."""

    def test_control_flow_range(self):
        """Control flow opcodes in 0x00-0x0F range."""
        control_ops = [Opcode.NOP, Opcode.TARGET, Opcode.JUMP, Opcode.JUMPI,
                       Opcode.NEXT, Opcode.LVAL, Opcode.RANGE, Opcode.ITER, Opcode.HALT]
        for op in control_ops:
            assert 0x00 <= op.code <= 0x0F, f"{op.name} not in control range"

    def test_stack_register_range(self):
        """Stack/register opcodes in 0x10-0x1F range."""
        stack_ops = [Opcode.PUSH, Opcode.POP, Opcode.DUPL, Opcode.SWAP,
                     Opcode.LOAD, Opcode.STOW, Opcode.INPUT, Opcode.OUTPUT]
        for op in stack_ops:
            assert 0x10 <= op.code <= 0x1F, f"{op.name} not in stack/reg range"

    def test_arithmetic_range(self):
        """Arithmetic opcodes in 0x20-0x2A range."""
        arith_ops = [Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV,
                     Opcode.MOD, Opcode.NEG, Opcode.EQ, Opcode.LT,
                     Opcode.GT, Opcode.LTE, Opcode.GTE]
        for op in arith_ops:
            assert 0x20 <= op.code <= 0x2F, f"{op.name} not in arithmetic range"

    def test_boolean_bitwise_range(self):
        """Boolean/bitwise opcodes in 0x30-0x3F range."""
        bool_ops = [Opcode.NOT, Opcode.AND, Opcode.OR, Opcode.XOR,
                    Opcode.BAND, Opcode.BOR, Opcode.BXOR, Opcode.BNOT,
                    Opcode.SHL, Opcode.SHR]
        for op in bool_ops:
            assert 0x30 <= op.code <= 0x3F, f"{op.name} not in bool/bitwise range"

    def test_allocator_range(self):
        """Allocator opcodes in 0x40-0x4F range."""
        alloc_ops = [Opcode.BQMX, Opcode.SQMX, Opcode.XQMX,
                     Opcode.BSMX, Opcode.SSMX, Opcode.XSMX,
                     Opcode.VEC, Opcode.VECI, Opcode.VECX]
        for op in alloc_ops:
            assert 0x40 <= op.code <= 0x4F, f"{op.name} not in allocator range"

    def test_vector_access_range(self):
        """Vector access opcodes in 0x50-0x5F range."""
        vec_ops = [Opcode.VECPUSH, Opcode.VECGET, Opcode.VECSET, Opcode.VECLEN,
                   Opcode.IDXGRID, Opcode.IDXTRIU]
        for op in vec_ops:
            assert 0x50 <= op.code <= 0x5F, f"{op.name} not in vector range"

    def test_xqmx_range(self):
        """XQMX opcodes in 0x60-0x7F range."""
        xqmx_ops = [Opcode.GETLINE, Opcode.SETLINE, Opcode.ADDLINE,
                    Opcode.GETQUAD, Opcode.SETQUAD, Opcode.ADDQUAD,
                    Opcode.RESIZE, Opcode.ROWFIND, Opcode.COLFIND,
                    Opcode.ROWSUM, Opcode.COLSUM,
                    Opcode.ONEHOTR, Opcode.ONEHOTC, Opcode.EXCLUDE, Opcode.IMPLIES, Opcode.ENERGY]
        for op in xqmx_ops:
            assert 0x60 <= op.code <= 0x7F, f"{op.name} not in XQMX range"

class TestSpecificOpcodeMetadata:
    """Tests for specific opcode metadata values."""

    def test_nop_metadata(self):
        """NOP has no stack effect and no operands."""
        meta = Opcode.NOP.meta
        assert meta.stack_pop == 0
        assert meta.stack_push == 0
        assert meta.operand_count == 0

    def test_push_metadata(self):
        """PUSH takes immediate and pushes one value."""
        meta = Opcode.PUSH.meta
        assert meta.stack_pop == 0
        assert meta.stack_push == 1
        assert meta.operand_count == 1
        assert meta.operand_types == (OperandType.IMMEDIATE,)

    def test_add_metadata(self):
        """ADD pops two and pushes one."""
        meta = Opcode.ADD.meta
        assert meta.stack_pop == 2
        assert meta.stack_push == 1
        assert meta.operand_count == 0

    def test_stow_metadata(self):
        """STOW pops one and takes register operand."""
        meta = Opcode.STOW.meta
        assert meta.stack_pop == 1
        assert meta.stack_push == 0
        assert meta.operand_count == 1
        assert meta.operand_types == (OperandType.REGISTER,)

    def test_load_metadata(self):
        """LOAD takes register operand and pushes value."""
        meta = Opcode.LOAD.meta
        assert meta.stack_pop == 0
        assert meta.stack_push == 1
        assert meta.operand_count == 1
        assert meta.operand_types == (OperandType.REGISTER,)

    def test_jump_metadata(self):
        """JUMP takes target operand."""
        meta = Opcode.JUMP.meta
        assert meta.operand_count == 1
        assert meta.operand_types == (OperandType.TARGET,)

    def test_target_metadata(self):
        """TARGET takes target operand."""
        meta = Opcode.TARGET.meta
        assert meta.operand_count == 1
        assert meta.operand_types == (OperandType.TARGET,)

class TestOpcodeCodeProperty:
    """Tests for opcode .code property."""

    def test_code_property_returns_int(self):
        """code property returns integer value."""
        for op in Opcode:
            assert isinstance(op.code, int)

    def test_code_matches_meta_code(self):
        """code property equals meta.code."""
        for op in Opcode:
            assert op.code == op.meta.code

class TestOperandType:
    """Tests for OperandType enum."""

    def test_operand_type_values(self):
        """OperandType has expected members."""
        assert hasattr(OperandType, "IMMEDIATE")
        assert hasattr(OperandType, "REGISTER")
        assert hasattr(OperandType, "TARGET")

    def test_operand_type_count(self):
        """Should have exactly 3 operand types."""
        assert len(OperandType) == 3
