# Python Style

## Spacing

- Use single blank lines consistently everywhere:
  - After imports before first definition
  - Between top-level definitions (classes, functions)
  - Between methods within a class
  - Before section comment blocks (`# === ... ===`)

- No blank line after class docstring for dataclasses or simple classes.
- Blank line after class docstring before `__init__` when class has methods.

## Docstrings

- Use spaces inside triple quotes: `""" Text here. """`
- Module docstrings: multi-line with `"""` on separate lines
- Class/method docstrings: single line with spaces

```python
"""
Module description here.
"""

from typing import Any

class SimpleClass:
    """ Simple class with no methods. """
    pass

class ClassWithMethods:
    """ Class with methods. """

    def __init__(self, value: int):
        self.value = value

    def method(self):
        """ Method description. """
        pass
```

## Dataclasses

No blank line after docstring, fields immediately follow:

```python
@dataclass(frozen=True)
class OpcodeMeta:
    """ Metadata for an opcode. """
    code: int
    stack_pop: int
    description: str
```

## Method Bodies

- Blank line before multi-line statements
- Blank line to separate logical groups

```python
def __init__(self, expected: str, got: str, context: str = ""):
    self.expected = expected
    self.got = got
    self.context = context

    msg = f"Type mismatch: expected {expected}, got {got}"
    if context:
        msg += f" in {context}"

    super().__init__(msg)
```

## Imports

- Standard library first, then third-party, then local
- Single blank line between groups
- `from __future__ import annotations` first if used
- Remove unused imports
