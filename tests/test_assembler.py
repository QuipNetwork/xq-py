"""
Tests for the XQVM assembler: parser, validator, program, and disassembler.
"""

import pytest

from xqvm.core.opcodes import Opcode
from xqvm.core.program import Instruction
from xqvm.assembler.parser import parse, ParseError
from xqvm.assembler.validator import validate, ValidationError
from xqvm.assembler.program import assemble
from tools.disassembler import disassemble, disassemble_instruction

# === Parser Tests ===

class TestParserBasic:
    """ Basic parsing of valid assembly. """

    def test_empty_source(self):
        assert parse("") == []

    def test_comment_only(self):
        assert parse("# this is a comment") == []

    def test_blank_lines(self):
        assert parse("\n\n\n") == []

    def test_single_nop(self):
        result = parse("NOP")
        assert len(result) == 1
        assert result[0].opcode == Opcode.NOP
        assert result[0].operands == ()

    def test_single_halt(self):
        result = parse("HALT")
        assert len(result) == 1
        assert result[0].opcode == Opcode.HALT

    def test_push_decimal(self):
        result = parse("PUSH 42")
        assert result[0].opcode == Opcode.PUSH
        assert result[0].operands == (42,)

    def test_push_hex(self):
        result = parse("PUSH 0x0A")
        assert result[0].operands == (10,)

    def test_push_hex_upper(self):
        result = parse("PUSH 0XFF")
        assert result[0].operands == (255,)

    def test_push_negative(self):
        result = parse("PUSH -5")
        assert result[0].operands == (-5,)

    def test_push_negative_hex(self):
        result = parse("PUSH -0x0A")
        assert result[0].operands == (-10,)

    def test_push_zero(self):
        result = parse("PUSH 0")
        assert result[0].operands == (0,)

    def test_register_operand(self):
        result = parse("LOAD r0")
        assert result[0].opcode == Opcode.LOAD
        assert result[0].operands == (0,)

    def test_register_high(self):
        result = parse("STOW r255")
        assert result[0].operands == (255,)

    def test_target_operand(self):
        result = parse("TARGET .0")
        assert result[0].opcode == Opcode.TARGET
        assert result[0].operands == (0,)

    def test_jump_target(self):
        result = parse("JUMP .5")
        assert result[0].operands == (5,)

    def test_two_register_operands(self):
        result = parse("ENERGY r0 r1")
        assert result[0].opcode == Opcode.ENERGY
        assert result[0].operands == (0, 1)

class TestParserLineNumbers:
    """ Line number tracking. """

    def test_line_numbers_simple(self):
        result = parse("NOP\nHALT")
        assert result[0].line == 1
        assert result[1].line == 2

    def test_line_numbers_with_blanks(self):
        result = parse("NOP\n\n\nHALT")
        assert result[0].line == 1
        assert result[1].line == 4

    def test_line_numbers_with_comments(self):
        result = parse("# header\nNOP\n# middle\nHALT")
        assert result[0].line == 2
        assert result[1].line == 4

class TestParserInlineComments:
    """ Inline comment stripping. """

    def test_inline_comment(self):
        result = parse("PUSH 10 # push ten")
        assert result[0].operands == (10,)

    def test_instruction_with_trailing_whitespace(self):
        result = parse("NOP   ")
        assert len(result) == 1
        assert result[0].opcode == Opcode.NOP

class TestParserCaseInsensitive:
    """ Opcode name case handling. """

    def test_lowercase(self):
        result = parse("nop")
        assert result[0].opcode == Opcode.NOP

    def test_mixed_case(self):
        result = parse("Push 1")
        assert result[0].opcode == Opcode.PUSH

class TestParserMultiLine:
    """ Multi-line programs. """

    def test_simple_program(self):
        source = """
        PUSH 5
        PUSH 3
        ADD
        HALT
        """
        result = parse(source)
        assert len(result) == 4
        assert result[0].opcode == Opcode.PUSH
        assert result[0].operands == (5,)
        assert result[2].opcode == Opcode.ADD
        assert result[3].opcode == Opcode.HALT

    def test_loop_program(self):
        source = """
        PUSH 0
        PUSH 5
        RANGE
          LVAL r0
        NEXT
        HALT
        """
        result = parse(source)
        assert len(result) == 6
        assert result[2].opcode == Opcode.RANGE
        assert result[3].opcode == Opcode.LVAL
        assert result[3].operands == (0,)
        assert result[4].opcode == Opcode.NEXT

class TestParserErrors:
    """ Parser error cases. """

    def test_unknown_opcode(self):
        with pytest.raises(ParseError, match="Unknown opcode"):
            parse("BOGUS")

    def test_missing_operand(self):
        with pytest.raises(ParseError, match="expects 1 operand"):
            parse("PUSH")

    def test_extra_operand(self):
        with pytest.raises(ParseError, match="expects 0 operand"):
            parse("NOP 5")

    def test_invalid_register_format(self):
        with pytest.raises(ParseError, match="Expected register"):
            parse("LOAD 5")

    def test_invalid_register_number(self):
        with pytest.raises(ParseError, match="Invalid register"):
            parse("LOAD rABC")

    def test_register_out_of_range(self):
        with pytest.raises(ParseError, match="Register out of range"):
            parse("LOAD r256")

    def test_invalid_target_format(self):
        with pytest.raises(ParseError, match="Expected target"):
            parse("JUMP 5")

    def test_invalid_target_number(self):
        with pytest.raises(ParseError, match="Invalid target"):
            parse("JUMP .abc")

    def test_invalid_integer(self):
        with pytest.raises(ParseError, match="Invalid integer"):
            parse("PUSH abc")

    def test_error_includes_line_number(self):
        with pytest.raises(ParseError) as exc_info:
            parse("NOP\nBOGUS")
        assert exc_info.value.line == 2

# === Validator Tests ===

class TestValidatorTargets:
    """ Jump target validation. """

    def test_valid_targets(self):
        instructions = parse("TARGET .0\nJUMP .0\nHALT")
        validate(instructions)  # Should not raise

    def test_undefined_target_jump(self):
        instructions = parse("JUMP .0\nHALT")
        with pytest.raises(ValidationError, match="Undefined target .0"):
            validate(instructions)

    def test_undefined_target_jumpi(self):
        instructions = parse("PUSH 1\nJUMPI .5\nHALT")
        with pytest.raises(ValidationError, match="Undefined target .5"):
            validate(instructions)

    def test_forward_reference(self):
        instructions = parse("JUMP .0\nTARGET .0\nHALT")
        validate(instructions)  # Forward refs are valid

    def test_duplicate_target(self):
        instructions = parse("TARGET .0\nTARGET .0\nHALT")
        with pytest.raises(ValidationError, match="Duplicate target .0"):
            validate(instructions)

    def test_target_definition_no_validation_needed(self):
        instructions = parse("TARGET .5\nHALT")
        validate(instructions)  # Unused target is fine

class TestValidatorOperands:
    """ Operand validation. """

    def test_valid_program(self):
        source = """
        PUSH 10
        STOW r0
        LOAD r0
        HALT
        """
        instructions = parse(source)
        validate(instructions)  # Should not raise

# === Assemble Tests ===

class TestAssemble:
    """ End-to-end assembly. """

    def test_assemble_simple(self):
        result = assemble("PUSH 1\nPUSH 2\nADD\nHALT", name="test")
        assert result.name == "test"
        assert len(result) == 4
        assert result[0].opcode == Opcode.PUSH

    def test_assemble_with_targets(self):
        source = """
        TARGET .0
        NOP
        JUMP .0
        """
        result = assemble(source)
        assert len(result) == 3

    def test_assemble_source_lines(self):
        result = assemble("NOP\nHALT")
        assert result.source_lines == 2

    def test_assemble_rejects_invalid(self):
        with pytest.raises(ParseError):
            assemble("BOGUS")

    def test_assemble_rejects_undefined_target(self):
        with pytest.raises(ValidationError):
            assemble("JUMP .0\nHALT")

# === Disassembler Tests ===

class TestDisassembleInstruction:
    """ Single instruction disassembly. """

    def test_nop(self):
        instr = Instruction(Opcode.NOP)
        assert disassemble_instruction(instr) == "NOP"

    def test_push_small(self):
        instr = Instruction(Opcode.PUSH, (5,))
        assert disassemble_instruction(instr) == "PUSH 5"

    def test_push_hex(self):
        instr = Instruction(Opcode.PUSH, (255,))
        assert disassemble_instruction(instr) == "PUSH 0xFF"

    def test_push_negative_hex(self):
        instr = Instruction(Opcode.PUSH, (-32,))
        assert disassemble_instruction(instr) == "PUSH -0x20"

    def test_register(self):
        instr = Instruction(Opcode.LOAD, (5,))
        assert disassemble_instruction(instr) == "LOAD r5"

    def test_target(self):
        instr = Instruction(Opcode.JUMP, (3,))
        assert disassemble_instruction(instr) == "JUMP .3"

    def test_two_registers(self):
        instr = Instruction(Opcode.ENERGY, (0, 1))
        assert disassemble_instruction(instr) == "ENERGY r0 r1"

class TestDisassembleProgram:
    """ Full program disassembly. """

    def test_simple_program(self):
        from xqvm.core.program import Program
        prog = Program([
            Instruction(Opcode.PUSH, (1,)),
            Instruction(Opcode.PUSH, (2,)),
            Instruction(Opcode.ADD),
            Instruction(Opcode.HALT),
        ])
        text = disassemble(prog)
        assert text == "PUSH 1\nPUSH 2\nADD\nHALT"

    def test_empty_program(self):
        from xqvm.core.program import Program
        assert disassemble(Program()) == ""

# === Round-Trip Tests ===

class TestRoundTrip:
    """ Assemble → disassemble → assemble round trips. """

    def test_arithmetic_round_trip(self):
        source = "PUSH 5\nPUSH 3\nADD\nHALT"
        prog1 = assemble(source)
        text = disassemble(prog1.program)
        prog2 = assemble(text)
        assert len(prog1) == len(prog2)
        for i in range(len(prog1)):
            assert prog1[i].opcode == prog2[i].opcode
            assert prog1[i].operands == prog2[i].operands

    def test_control_flow_round_trip(self):
        source = "TARGET .0\nPUSH 1\nJUMPI .0\nHALT"
        prog1 = assemble(source)
        text = disassemble(prog1.program)
        prog2 = assemble(text)
        assert len(prog1) == len(prog2)
        for i in range(len(prog1)):
            assert prog1[i].opcode == prog2[i].opcode
            assert prog1[i].operands == prog2[i].operands

    def test_allocator_round_trip(self):
        source = "PUSH 4\nBQMX r0\nHALT"
        prog1 = assemble(source)
        text = disassemble(prog1.program)
        prog2 = assemble(text)
        for i in range(len(prog1)):
            assert prog1[i].opcode == prog2[i].opcode
            assert prog1[i].operands == prog2[i].operands

    def test_all_operand_types_round_trip(self):
        source = "PUSH 0xFF\nSTOW r10\nTARGET .3\nJUMP .3\nENERGY r0 r1\nHALT"
        prog1 = assemble(source)
        text = disassemble(prog1.program)
        prog2 = assemble(text)
        assert len(prog1) == len(prog2)
        for i in range(len(prog1)):
            assert prog1[i].opcode == prog2[i].opcode
            assert prog1[i].operands == prog2[i].operands

# === Integration: Assemble and Execute ===

class TestAssembleAndExecute:
    """ Assemble source and run through the executor. """

    def test_add_two_numbers(self):
        from xqvm.core.executor import Executor
        source = "PUSH 3\nPUSH 7\nADD\nHALT"
        prog = assemble(source)
        ex = Executor()
        ex.execute(prog.program)
        assert ex.state.peek(0) == 10

    def test_loop_sum(self):
        from xqvm.core.executor import Executor
        source = """
        # Sum 0..4 → r1
        PUSH 0
        STOW r1       # r1 = accumulator = 0
        PUSH 0        # start
        PUSH 5        # count
        RANGE
          LVAL r0     # r0 = loop value
          LOAD r1
          LOAD r0
          ADD
          STOW r1     # r1 += r0
        NEXT
        HALT
        """
        prog = assemble(source)
        ex = Executor()
        ex.execute(prog.program)
        assert ex.state.get_register(1) == 10  # 0+1+2+3+4

    def test_forward_jump(self):
        from xqvm.core.executor import Executor
        source = """
        PUSH 42
        STOW r0
        PUSH 1
        JUMPI .0       # Forward jump — skip the overwrite
        PUSH 99
        STOW r0
        TARGET .0
        HALT
        """
        prog = assemble(source)
        ex = Executor()
        ex.execute(prog.program)
        assert ex.state.get_register(0) == 42
