"""
Problem container and action recording for the constraint programming DSL.

The Problem class accumulates declarations (inputs, model, loops,
operations, outputs) as an ordered action list. Calling compile()
delegates to the compiler module to emit three .xqasm programs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Union

from xqvm.core import XQMXDomain

from .expression import Expr, RegLoad, coerce, Types
from .symbols import InputRef, LoopVar, ModelRef, SampleRef, OutputRef

# ---------------------------------------------------------------------------
# Register allocator
# ---------------------------------------------------------------------------

class _RegisterAllocator:
    """ Simple incrementing register allocator. """

    def __init__(self, start: int = 0) -> None:
        self._next = start

    def alloc(self) -> int:
        """ Allocate the next register. """
        if self._next > 255:
            raise RuntimeError("Register overflow: exceeded r255")
        reg = self._next
        self._next += 1
        return reg

# ---------------------------------------------------------------------------
# Action recording
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """ A single recorded declaration in the problem. """
    kind: str
    data: dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Compiled output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompiledPrograms:
    """ Three compiled .xqasm program strings. """
    encoder: str
    verifier: str
    decoder: str

# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

class Problem:
    """ Container for a constraint programming problem description. """

    def __init__(self, name: str) -> None:
        self.name = name
        self._alloc = _RegisterAllocator()
        self._actions: list[Action] = []
        self._inputs: list[InputRef] = []
        self._constraints: list[Action] = []
        self._outputs: list[OutputRef] = []
        self._output_slot = 0
        self._model: ModelRef | None = None
        self._sample: SampleRef | None = None
        self._indent = 0
        self._rows_expr: Expr | None = None
        self._cols_expr: Expr | None = None

    @property
    def model(self) -> ModelRef:
        """ The declared model reference. """
        if self._model is None:
            raise RuntimeError("No model defined — call define_model() first")
        return self._model

    @property
    def sample(self) -> SampleRef:
        """ The implicit sample reference (counterpart of the model). """
        if self._sample is None:
            raise RuntimeError("No model defined — call define_model() first")
        return self._sample

    def input(self, name: str, type: Types) -> InputRef:
        """ Declare a runtime input. Returns a symbolic InputRef. """
        reg = self._alloc.alloc()
        ref = InputRef(reg, name, type)
        self._inputs.append(ref)
        self._actions.append(Action("input", {"ref": ref}))
        return ref

    def define_model(
        self, size: Union[Expr, int], domain: XQMXDomain,
        rows: Union[Expr, int, None] = None, cols: Union[Expr, int, None] = None,
    ) -> None:
        """ Declare the XQMX model the encoder will build. """
        size_expr = coerce(size)
        is_2d = rows is not None and cols is not None
        cols_reg: int | None = None

        if is_2d:
            self._rows_expr = coerce(rows)
            self._cols_expr = coerce(cols)
            cols_reg = self._alloc.alloc()

        model_reg = self._alloc.alloc()
        self._model = ModelRef(self, model_reg, domain, cols_reg, is_2d)
        self._sample = SampleRef(model_reg + 100)

        self._actions.append(Action("define_model", {
            "model_reg": model_reg,
            "domain": domain,
            "size_expr": size_expr,
            "is_2d": is_2d,
            "cols_reg": cols_reg,
            "rows_expr": self._rows_expr,
            "cols_expr": self._cols_expr,
        }))

    @contextmanager
    def range(self, start: Union[Expr, int], end: Union[Expr, int]) -> Iterator[LoopVar]:
        """ Declare a RANGE loop. Yields a symbolic LoopVar. """
        reg = self._alloc.alloc()
        start_expr = coerce(start)
        end_expr = coerce(end)
        var = LoopVar(reg, f"v{reg}")

        self._actions.append(Action("range_start", {
            "var": var,
            "start_expr": start_expr,
            "end_expr": end_expr,
        }))
        self._indent += 1
        try:
            yield var
        finally:
            self._indent -= 1
            self._actions.append(Action("range_end", {}))

    def stow(self, name: str, expr: Union[Expr, int]) -> RegLoad:
        """ Evaluate an expression and stow into a scratch register. Returns a RegLoad. """
        reg = self._alloc.alloc()
        self._actions.append(Action("stow", {
            "reg": reg,
            "name": name,
            "expr": coerce(expr),
        }))
        return RegLoad(reg)

    def output(self, name: str, type: Types) -> OutputRef:
        """ Declare a pipeline output. Returns a symbolic OutputRef. """
        reg = self._alloc.alloc()
        slot = self._output_slot
        self._output_slot += 1
        ref = OutputRef(self, slot, reg, name, type)
        self._outputs.append(ref)
        self._actions.append(Action("output_decl", {"ref": ref}))
        return ref

    # -- Internal recording methods called by ModelRef / OutputRef ----------

    def _record_add_linear(
        self, model: ModelRef, coord: Any, weight: Union[Expr, int],
    ) -> None:
        self._actions.append(Action("add_linear", {
            "model": model,
            "coord": coord,
            "weight": coerce(weight),
        }))

    def _record_add_quadratic(
        self, model: ModelRef, coord_a: Any, coord_b: Any, weight: Union[Expr, int],
    ) -> None:
        self._actions.append(Action("add_quadratic", {
            "model": model,
            "coord_a": coord_a,
            "coord_b": coord_b,
            "weight": coerce(weight),
        }))

    def _record_onehot_row(
        self, model: ModelRef, row: Union[Expr, int], penalty: int,
    ) -> None:
        action = Action("onehot_row", {
            "model": model,
            "row": coerce(row),
            "penalty": penalty,
        })
        self._actions.append(action)
        self._constraints.append(action)

    def _record_onehot_col(
        self, model: ModelRef, col: Union[Expr, int], penalty: int,
    ) -> None:
        action = Action("onehot_col", {
            "model": model,
            "col": coerce(col),
            "penalty": penalty,
        })
        self._actions.append(action)
        self._constraints.append(action)

    def _record_penalty_eq(
        self, model: ModelRef,
        indices: Any, coefficients: Any,
        target: Union[Expr, int], penalty: Union[Expr, int],
    ) -> None:
        """
        Add weighted equality constraint: P * (sum(a_k * x_k) - b)^2.

        Args:
            model: The XQMX model register.
            indices: Register or InputRef holding the variable indices vec.
            coefficients: Register or InputRef holding the coefficients vec.
            target: Target value b.
            penalty: Penalty weight P.
        """
        action = Action("penalty_eq", {
            "model": model,
            "indices": indices,
            "coefficients": coefficients,
            "target": coerce(target),
            "penalty": coerce(penalty),
        })
        self._actions.append(action)
        self._constraints.append(action)

    def _record_slack(
        self, indices: Any, coefficients: Any,
        start_index: Union[Expr, int], capacity: Union[Expr, int],
    ) -> None:
        """
        Append slack variable indices and binary coefficients to existing vecs.

        Args:
            indices: Register holding variable indices vec (will be extended).
            coefficients: Register holding coefficients vec (will be extended).
            start_index: First slack variable index.
            capacity: Maximum gap value (determines number of slack bits).
        """
        self._actions.append(Action("slack", {
            "indices": indices,
            "coefficients": coefficients,
            "start_index": coerce(start_index),
            "capacity": coerce(capacity),
        }))

    def _record_atleast(
        self, model: ModelRef, variables: Any,
        k: Union[Expr, int], penalty: Union[Expr, int],
    ) -> None:
        """
        At least k variables in the vec must be 1.

        Args:
            model: The XQMX model register.
            variables: Register holding variable indices vec.
            k: Minimum number of variables that must be 1.
            penalty: Penalty weight P.
        """
        action = Action("atleast", {
            "model": model,
            "variables": variables,
            "k": coerce(k),
            "penalty": coerce(penalty),
        })
        self._actions.append(action)
        self._constraints.append(action)

    def _record_reduce(
        self, model: ModelRef,
        var_a: Union[Expr, int], var_b: Union[Expr, int],
        penalty: Union[Expr, int],
        result_reg: int,
    ) -> None:
        """
        Degree reduction: replace x_a * x_b with auxiliary variable w.

        The auxiliary variable index is stored in result_reg.

        Args:
            model: The XQMX model register.
            var_a: Index of first variable.
            var_b: Index of second variable.
            penalty: Auxiliary constraint penalty (must exceed max coefficient).
            result_reg: Register to store the auxiliary variable index.
        """
        self._actions.append(Action("reduce", {
            "model": model,
            "var_a": coerce(var_a),
            "var_b": coerce(var_b),
            "penalty": coerce(penalty),
            "result_reg": result_reg,
        }))

    def _record_output_append(self, output: OutputRef, value_expr: Union[Expr, int]) -> None:
        self._actions.append(Action("output_append", {
            "output": output,
            "value_expr": coerce(value_expr),
        }))

    # -- Compilation --------------------------------------------------------

    def compile(self) -> CompiledPrograms:
        """ Compile the problem into three .xqasm program strings. """
        from .compiler import compile_encoder, compile_verifier, compile_decoder

        return CompiledPrograms(
            encoder=compile_encoder(self),
            verifier=compile_verifier(self),
            decoder=compile_decoder(self),
        )
