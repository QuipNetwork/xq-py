"""
XQVM Exception Hierarchy
"""

from typing import Any

class XQVMError(Exception):
    """ Base exception for all XQVM errors. """
    pass

class StackUnderflow(XQVMError):
    """ Raised when attempting to pop from an empty stack. """

    def __init__(self, required: int = 1, available: int = 0):
        self.required = required
        self.available = available
        super().__init__(f"Stack underflow: need {required}, have {available}")

class StackOverflow(XQVMError):
    """ Raised when stack exceeds maximum capacity. """

    def __init__(self, max_size: int):
        self.max_size = max_size
        super().__init__(f"Stack overflow: maximum size {max_size} exceeded")

class TypeMismatch(XQVMError):
    """ Raised when an operation receives an unexpected type. """

    def __init__(self, expected: str, got: str, context: str = ""):
        self.expected = expected
        self.got = got
        self.context = context

        msg = f"Type mismatch: expected {expected}, got {got}"
        if context:
            msg += f" in {context}"

        super().__init__(msg)

class RegisterNotFound(XQVMError):
    """ Raised when accessing a non-existent register slot. """

    def __init__(self, slot: int):
        self.slot = slot
        super().__init__(f"Register not found: r{slot}")

class InvalidOpcode(XQVMError):
    """ Raised when encountering an unknown opcode. """

    def __init__(self, opcode: Any):
        self.opcode = opcode
        super().__init__(f"Invalid opcode: {opcode}")

class DivisionByZero(XQVMError):
    """ Raised when attempting to divide by zero. """

    def __init__(self):
        super().__init__("Division by zero")

class TargetNotFound(XQVMError):
    """ Raised when a jump target does not exist. """

    def __init__(self, target_id: int):
        self.target_id = target_id
        super().__init__(f"Target not found: {target_id}")

class LoopError(XQVMError):
    """ Raised for loop-related errors (e.g., NEXT outside loop). """

    def __init__(self, message: str):
        super().__init__(message)

class XQMXModeError(XQVMError):
    """ Raised when an XQMX operation is invalid for the current mode. """

    def __init__(self, operation: str, mode: str, required_mode: str):
        self.operation = operation
        self.mode = mode
        self.required_mode = required_mode

        super().__init__(
            f"XQMX mode error: {operation} requires {required_mode} mode, but matrix is in {mode} mode"
        )
