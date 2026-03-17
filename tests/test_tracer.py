"""
Tests for the XQVM execution tracer.
"""

import pytest

from xqvm.core.opcodes import Opcode
from xqvm.core.program import Instruction, Program, make_program
from xqvm.core.executor import Executor
from tools.tracer import Tracer

# === Event Collection ===

class TestEventCollection:
    """ Test that tracer collects events correctly. """

    def test_silent_collects_events(self):
        tracer = Tracer(verbosity=0)
        prog = make_program([
            Instruction(Opcode.PUSH, (1,)),
            Instruction(Opcode.PUSH, (2,)),
            Instruction(Opcode.ADD),
            Instruction(Opcode.HALT),
        ])
        ex = Executor(tracer=tracer)
        ex.execute(prog)
        # 3 step events + 1 halt event (TARGET-like NOPs excluded, HALT triggers on_halt)
        step_events = [e for e in tracer.events if "opcode" in e]
        halt_events = [e for e in tracer.events if "halt" in e]
        assert len(step_events) == 3  # PUSH, PUSH, ADD (HALT triggers halt, not step_end)
        assert len(halt_events) == 1

    def test_event_has_opcode(self):
        tracer = Tracer(verbosity=0)
        prog = make_program([
            Instruction(Opcode.PUSH, (42,)),
            Instruction(Opcode.HALT),
        ])
        ex = Executor(tracer=tracer)
        ex.execute(prog)
        assert tracer.events[0]["opcode"] == "PUSH"
        assert tracer.events[0]["operands"] == (42,)

    def test_event_captures_stack(self):
        tracer = Tracer(verbosity=0)
        prog = make_program([
            Instruction(Opcode.PUSH, (10,)),
            Instruction(Opcode.PUSH, (20,)),
            Instruction(Opcode.HALT),
        ])
        ex = Executor(tracer=tracer)
        ex.execute(prog)
        # After first PUSH: stack was [] -> [10]
        assert tracer.events[0]["stack_before"] == []
        assert tracer.events[0]["stack_after"] == [10]
        # After second PUSH: stack was [10] -> [10, 20]
        assert tracer.events[1]["stack_before"] == [10]
        assert tracer.events[1]["stack_after"] == [10, 20]

    def test_event_captures_register_changes(self):
        tracer = Tracer(verbosity=0)
        prog = make_program([
            Instruction(Opcode.PUSH, (99,)),
            Instruction(Opcode.STOW, (5,)),
            Instruction(Opcode.HALT),
        ])
        ex = Executor(tracer=tracer)
        ex.execute(prog)
        # STOW event should show r5 changed
        stow_event = tracer.events[1]
        assert 5 in stow_event["changed"]
        assert stow_event["registers"][5] == 99

    def test_no_change_when_register_unchanged(self):
        tracer = Tracer(verbosity=0)
        prog = make_program([
            Instruction(Opcode.PUSH, (1,)),
            Instruction(Opcode.PUSH, (2,)),
            Instruction(Opcode.ADD),
            Instruction(Opcode.HALT),
        ])
        ex = Executor(tracer=tracer)
        ex.execute(prog)
        # ADD doesn't touch registers
        add_event = tracer.events[2]
        assert add_event["changed"] == set()

# === Halt Event ===

class TestHaltEvent:
    """ Test halt event recording. """

    def test_halt_event_recorded(self):
        tracer = Tracer(verbosity=0)
        prog = make_program([
            Instruction(Opcode.PUSH, (42,)),
            Instruction(Opcode.STOW, (0,)),
            Instruction(Opcode.HALT),
        ])
        ex = Executor(tracer=tracer)
        ex.execute(prog)
        halt = [e for e in tracer.events if "halt" in e]
        assert len(halt) == 1
        assert halt[0]["final_stack"] == []
        assert halt[0]["final_registers"] == 1

    def test_halt_event_tracks_outputs(self):
        tracer = Tracer(verbosity=0)
        prog = make_program([
            Instruction(Opcode.PUSH, (10,)),
            Instruction(Opcode.STOW, (0,)),
            Instruction(Opcode.PUSH, (0,)),
            Instruction(Opcode.OUTPUT, (0,)),
            Instruction(Opcode.HALT),
        ])
        ex = Executor(tracer=tracer)
        ex.execute(prog)
        halt = [e for e in tracer.events if "halt" in e][0]
        assert halt["output_slots"] == 1

# === Error Event ===

class TestErrorEvent:
    """ Test error event recording. """

    def test_error_event_recorded(self):
        tracer = Tracer(verbosity=0)
        prog = make_program([
            Instruction(Opcode.POP),  # Stack underflow
            Instruction(Opcode.HALT),
        ])
        ex = Executor(tracer=tracer)
        with pytest.raises(Exception):
            ex.execute(prog)
        error_events = [e for e in tracer.events if "error" in e]
        assert len(error_events) == 1
        assert "underflow" in error_events[0]["error"].lower()

# === Formatting ===

class TestFormatting:
    """ Test trace output formatting. """

    def test_compact_format(self):
        tracer = Tracer(verbosity=1)
        event = {
            "pc": 0,
            "opcode": "PUSH",
            "operands": (42,),
            "stack_before": [],
            "stack_after": [42],
            "registers": {},
            "pre_registers": {},
            "changed": set(),
        }
        text = tracer.format_event(event)
        assert "[0000] PUSH 42" == text

    def test_detailed_format_includes_stack(self):
        tracer = Tracer(verbosity=2)
        event = {
            "pc": 3,
            "opcode": "ADD",
            "operands": (),
            "stack_before": [10, 20],
            "stack_after": [30],
            "registers": {},
            "pre_registers": {},
            "changed": set(),
        }
        text = tracer.format_event(event)
        assert "[0003] ADD" in text
        assert "[10, 20] -> [30]" in text

    def test_detailed_format_includes_register_snapshot(self):
        tracer = Tracer(verbosity=2)
        event = {
            "pc": 1,
            "opcode": "STOW",
            "operands": (0,),
            "stack_before": [42],
            "stack_after": [],
            "registers": {0: 42},
            "pre_registers": {},
            "changed": {0},
        }
        text = tracer.format_event(event)
        assert "r0: int(42)" in text

    def test_format_error_event(self):
        tracer = Tracer(verbosity=1)
        event = {
            "pc": 5,
            "opcode": "POP",
            "operands": (),
            "error": "Stack underflow",
            "stack_before": [],
        }
        text = tracer.format_event(event)
        assert "ERROR" in text
        assert "POP" in text

    def test_format_halt_event(self):
        tracer = Tracer(verbosity=1)
        event = {
            "halt": True,
            "final_stack": [42],
            "final_registers": 3,
            "output_slots": 1,
        }
        text = tracer.format_event(event)
        assert "halt" in text
        assert "stack_depth=1" in text

    def test_format_trace_all_events(self):
        tracer = Tracer(verbosity=0)
        prog = make_program([
            Instruction(Opcode.PUSH, (1,)),
            Instruction(Opcode.HALT),
        ])
        ex = Executor(tracer=tracer)
        ex.execute(prog)
        text = tracer.format_trace()
        assert "PUSH" in text
        assert "halt" in text

# === Integration ===

class TestTracerIntegration:
    """ End-to-end tracer with real programs. """

    def test_loop_tracing(self):
        tracer = Tracer(verbosity=0)
        prog = make_program([
            Instruction(Opcode.PUSH, (0,)),
            Instruction(Opcode.PUSH, (3,)),
            Instruction(Opcode.RANGE),
            Instruction(Opcode.LVAL, (0,)),
            Instruction(Opcode.NEXT),
            Instruction(Opcode.HALT),
        ])
        ex = Executor(tracer=tracer)
        ex.execute(prog)
        opcodes = [e["opcode"] for e in tracer.events if "opcode" in e]
        # PUSH, PUSH, RANGE, then 3 iterations of (LVAL, NEXT), minus final NEXT exits
        assert "RANGE" in opcodes
        assert "LVAL" in opcodes
        assert "NEXT" in opcodes

    def test_tracer_does_not_affect_execution(self):
        prog = make_program([
            Instruction(Opcode.PUSH, (10,)),
            Instruction(Opcode.PUSH, (20,)),
            Instruction(Opcode.ADD),
            Instruction(Opcode.HALT),
        ])
        # Without tracer
        ex1 = Executor()
        ex1.execute(prog)
        result1 = ex1.state.peek(0)
        # With tracer
        ex2 = Executor(tracer=Tracer(verbosity=0))
        ex2.execute(prog)
        result2 = ex2.state.peek(0)
        assert result1 == result2 == 30
