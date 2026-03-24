"""
Compiler for the constraint programming DSL.

Generates three .xqasm program strings (encoder, verifier, decoder)
from a Problem's recorded action list.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from xqvm.core import XQMXDomain

from .expression import (
    Expr, Literal, RegLoad, BinOp,
    ColFindExpr, GetLineExpr,
    line, fmt_int, coerce, emit_flat_index, resolve_coord, expr_reg,
    Types,
)
from .symbols import InputRef, LoopVar, ModelRef, OutputRef

if TYPE_CHECKING:
    from .problem import Action, Problem

# ---------------------------------------------------------------------------
# Compiler: encoder
# ---------------------------------------------------------------------------

def compile_encoder(prob: Problem) -> str:
    """ Generate the encoder .xqasm program. """
    lines: list[str] = []
    indent = 0

    # Partition actions into sections
    input_actions: list[Action] = []
    model_action: Action | None = None
    body_actions: list[Action] = []
    output_actions: list[Action] = []

    in_output_section = False
    for action in prob._actions:
        if action.kind == "input":
            input_actions.append(action)
        elif action.kind == "define_model":
            model_action = action
        elif action.kind == "output_decl":
            in_output_section = True
            output_actions.append(action)
        elif in_output_section:
            output_actions.append(action)
        else:
            body_actions.append(action)

    # --- Inputs ---
    lines.append("# === Inputs ===")
    for action in input_actions:
        ref: InputRef = action.data["ref"]
        lines.append(f"PUSH {ref.reg}")
        lines.append(f"INPUT r{ref.reg}")

    # --- Allocations ---
    lines.append("")
    lines.append("# === Allocations ===")
    if model_action is not None:
        _emit_model_allocation(model_action.data, lines, indent)

    # --- Body (objective + constraints) ---
    obj_actions, con_actions = _partition_body(body_actions)

    if obj_actions:
        lines.append("")
        lines.append("# === Objective ===")
        _emit_body_actions(obj_actions, lines, 0)

    if con_actions:
        lines.append("")
        lines.append("# === Constraints ===")
        _emit_body_actions(con_actions, lines, 0)

    # --- Output ---
    lines.append("")
    lines.append("# === Output ===")
    lines.append("PUSH 0")
    if model_action is not None:
        lines.append(f"OUTPUT r{model_action.data['model_reg']}")
    lines.append("HALT")

    return "\n".join(lines) + "\n"

def _emit_model_allocation(d: dict[str, Any], lines: list[str], indent: int) -> None:
    """ Emit model size computation, allocation, and optional grid resize. """
    size_expr: Expr = d["size_expr"]
    model_reg: int = d["model_reg"]
    domain: XQMXDomain = d["domain"]
    is_2d: bool = d["is_2d"]
    cols_reg: int | None = d["cols_reg"]

    # Emit size expression (optimize N*N → LOAD, SQR)
    if isinstance(size_expr, BinOp) and size_expr.op == "MUL":
        left_reg = expr_reg(size_expr.left) or _input_reg(size_expr.left)
        right_reg = expr_reg(size_expr.right) or _input_reg(size_expr.right)

        if left_reg is not None and left_reg == right_reg:
            lines.append(line(f"LOAD r{left_reg}", indent))
            lines.append(line("SQR", indent))
        else:
            size_expr.emit(lines, indent)
    else:
        size_expr.emit(lines, indent)

    opcode = "BQMX" if domain == XQMXDomain.BINARY else "SQMX"
    lines.append(line(f"{opcode} r{model_reg}", indent))

    if is_2d:
        rows_expr: Expr = d["rows_expr"]
        cols_expr_val: Expr = d["cols_expr"]

        cols_expr_val.emit(lines, indent)
        lines.append(line(f"STOW r{cols_reg}", indent))

        rows_expr.emit(lines, indent)
        lines.append(line(f"LOAD r{cols_reg}", indent))
        lines.append(line(f"RESIZE r{model_reg}", indent))

def _input_reg(expr: Expr) -> int | None:
    """ Extract register from an InputRef expression. """
    if isinstance(expr, InputRef):
        return expr.reg
    return None

def _partition_body(body_actions: list[Action]) -> tuple[list[Action], list[Action]]:
    """ Split body actions into objective and constraint sections by top-level block. """
    blocks: list[list[Action]] = []
    current_block: list[Action] = []
    depth = 0

    for action in body_actions:
        current_block.append(action)
        if action.kind == "range_start":
            depth += 1
        elif action.kind == "range_end":
            depth -= 1
            if depth == 0:
                blocks.append(current_block)
                current_block = []
        elif depth == 0:
            blocks.append(current_block)
            current_block = []

    if current_block:
        blocks.append(current_block)

    obj_actions: list[Action] = []
    con_actions: list[Action] = []
    for block in blocks:
        has_constraint = any(a.kind in ("onehot_row", "onehot_col") for a in block)
        if has_constraint:
            con_actions.extend(block)
        else:
            obj_actions.extend(block)

    return obj_actions, con_actions

def _emit_body_actions(actions: list[Action], lines: list[str], indent: int) -> None:
    """ Emit a sequence of body actions (loops, operations) as assembly. """
    for action in actions:
        kind = action.kind
        d = action.data

        if kind == "range_start":
            _emit_range_start(d, lines, indent)
            indent += 1

        elif kind == "range_end":
            indent -= 1
            lines.append(line("NEXT", indent))

        elif kind == "stow":
            d["expr"].emit(lines, indent)
            lines.append(line(f"STOW r{d['reg']}", indent))

        elif kind == "add_linear":
            _emit_add_linear(d, lines, indent)

        elif kind == "add_quadratic":
            _emit_add_quadratic(d, lines, indent)

        elif kind == "onehot_row":
            d["row"].emit(lines, indent)
            lines.append(line(f"PUSH {fmt_int(d['penalty'])}", indent))
            lines.append(line(f"ONEHOTR r{d['model'].reg}", indent))

        elif kind == "onehot_col":
            d["col"].emit(lines, indent)
            lines.append(line(f"PUSH {fmt_int(d['penalty'])}", indent))
            lines.append(line(f"ONEHOTC r{d['model'].reg}", indent))

def _emit_range_start(d: dict[str, Any], lines: list[str], indent: int) -> None:
    """ Emit RANGE preamble: start value and count. """
    var: LoopVar = d["var"]
    start_expr: Expr = d["start_expr"]
    end_expr: Expr = d["end_expr"]

    start_expr.emit(lines, indent)

    if isinstance(start_expr, Literal) and start_expr.value == 0:
        end_expr.emit(lines, indent)
    else:
        end_expr.emit(lines, indent)
        start_expr.emit(lines, indent)
        lines.append(line("SUB", indent))

    lines.append(line("RANGE", indent))
    lines.append(line(f"LVAL r{var.reg}", indent + 1))

def _emit_add_linear(d: dict[str, Any], lines: list[str], indent: int) -> None:
    """ Emit ADDLINE instruction with coordinate and weight. """
    model: ModelRef = d["model"]

    if model.is_2d:
        row_expr, col_expr = resolve_coord(d["coord"])
        emit_flat_index(row_expr, col_expr, model.cols_reg, lines, indent)
    else:
        coerce(d["coord"]).emit(lines, indent)

    d["weight"].emit(lines, indent)
    lines.append(line(f"ADDLINE r{model.reg}", indent))

def _emit_add_quadratic(d: dict[str, Any], lines: list[str], indent: int) -> None:
    """ Emit ADDQUAD instruction with two coordinates and weight. """
    model: ModelRef = d["model"]

    if model.is_2d:
        row_a, col_a = resolve_coord(d["coord_a"])
        emit_flat_index(row_a, col_a, model.cols_reg, lines, indent)
    else:
        coerce(d["coord_a"]).emit(lines, indent)

    if model.is_2d:
        row_b, col_b = resolve_coord(d["coord_b"])
        emit_flat_index(row_b, col_b, model.cols_reg, lines, indent)
    else:
        coerce(d["coord_b"]).emit(lines, indent)

    d["weight"].emit(lines, indent)
    lines.append(line(f"ADDQUAD r{model.reg}", indent))

# ---------------------------------------------------------------------------
# Compiler: verifier
# ---------------------------------------------------------------------------

def compile_verifier(prob: Problem) -> str:
    """ Generate the verifier .xqasm program. """
    lines: list[str] = []

    R_MODEL = 0
    R_SAMPLE = 1
    R_N = 2
    R_VALID = 3
    R_ENERGY = 4
    R_LOOP = 10

    has_row = any(c.kind == "onehot_row" for c in prob._constraints)
    has_col = any(c.kind == "onehot_col" for c in prob._constraints)
    has_onehot = has_row or has_col

    # --- Inputs ---
    lines.append("# === Inputs ===")
    lines.append(f"PUSH 0")
    lines.append(f"INPUT r{R_MODEL}")
    lines.append(f"PUSH 1")
    lines.append(f"INPUT r{R_SAMPLE}")
    lines.append(f"PUSH 2")
    lines.append(f"INPUT r{R_N}")

    # --- Initialize valid flag ---
    lines.append("")
    lines.append("# === Validity checks ===")
    lines.append(f"PUSH 1")
    lines.append(f"STOW r{R_VALID}")

    if has_row:
        lines.append("")
        lines.append("# Check row sums = 1")
        _emit_verifier_check_loop(lines, R_N, R_LOOP, "ROWSUM", R_SAMPLE, R_VALID)

    if has_col:
        lines.append("")
        lines.append("# Check col sums = 1")
        _emit_verifier_check_loop(lines, R_N, R_LOOP, "COLSUM", R_SAMPLE, R_VALID)

    if not has_onehot:
        lines.append("")
        lines.append("# Check all variables are binary")
        _emit_verifier_binary_check(lines, R_N, R_LOOP, R_SAMPLE, R_VALID)

    # --- Energy ---
    lines.append("")
    lines.append("# === Energy ===")
    lines.append(f"ENERGY r{R_MODEL} r{R_SAMPLE}")
    lines.append(f"STOW r{R_ENERGY}")

    # --- Output ---
    lines.append("")
    lines.append("# === Output ===")
    lines.append(f"PUSH 0")
    lines.append(f"OUTPUT r{R_ENERGY}")
    lines.append(f"PUSH 1")
    lines.append(f"OUTPUT r{R_VALID}")
    lines.append("HALT")

    return "\n".join(lines) + "\n"

def _emit_verifier_check_loop(
    lines: list[str], n_reg: int, loop_reg: int,
    check_op: str, sample_reg: int, valid_reg: int,
) -> None:
    """ Emit a ROWSUM or COLSUM check loop for the verifier. """
    lines.append(f"PUSH 0")
    lines.append(f"LOAD r{n_reg}")
    lines.append("RANGE")
    lines.append(f"  LVAL r{loop_reg}")
    lines.append(f"  LOAD r{loop_reg}")
    lines.append(f"  {check_op} r{sample_reg}")
    lines.append(f"  PUSH 1")
    lines.append(f"  EQ")
    lines.append(f"  LOAD r{valid_reg}")
    lines.append(f"  AND")
    lines.append(f"  STOW r{valid_reg}")
    lines.append("NEXT")

def _emit_verifier_binary_check(
    lines: list[str], n_reg: int, loop_reg: int,
    sample_reg: int, valid_reg: int,
) -> None:
    """ Emit a binary domain check loop: each variable must be 0 or 1. """
    lines.append(f"PUSH 0")
    lines.append(f"LOAD r{n_reg}")
    lines.append("RANGE")
    lines.append(f"  LVAL r{loop_reg}")
    lines.append(f"  LOAD r{loop_reg}")
    lines.append(f"  GETLINE r{sample_reg}")
    lines.append(f"  COPY")
    lines.append(f"  PUSH 0")
    lines.append(f"  EQ")
    lines.append(f"  SWAP")
    lines.append(f"  PUSH 1")
    lines.append(f"  EQ")
    lines.append(f"  OR")
    lines.append(f"  LOAD r{valid_reg}")
    lines.append(f"  AND")
    lines.append(f"  STOW r{valid_reg}")
    lines.append("NEXT")

# ---------------------------------------------------------------------------
# Compiler: decoder
# ---------------------------------------------------------------------------

def compile_decoder(prob: Problem) -> str:
    """ Generate the decoder .xqasm program. """
    lines: list[str] = []

    R_SAMPLE = 0
    R_N = 1
    next_reg = 2
    R_LOOP = 10

    # --- Inputs ---
    lines.append("# === Inputs ===")
    lines.append(f"PUSH 0")
    lines.append(f"INPUT r{R_SAMPLE}")
    lines.append(f"PUSH 1")
    lines.append(f"INPUT r{R_N}")

    # --- Collect output blocks ---
    output_blocks = _collect_output_blocks(prob._actions)

    for output_ref, block in output_blocks:
        out_reg = next_reg
        next_reg += 1

        if output_ref.type_ == Types.Vec:
            lines.append("")
            lines.append(f"# === Decode {output_ref.name} ===")
            lines.append(f"VECI r{out_reg}")
            _emit_decoder_block(block, lines, 0, R_SAMPLE, R_N, R_LOOP, out_reg)

        lines.append("")
        lines.append("# === Output ===")
        lines.append(f"PUSH {output_ref.slot}")
        lines.append(f"OUTPUT r{out_reg}")

    lines.append("HALT")
    return "\n".join(lines) + "\n"

def _collect_output_blocks(actions: list[Action]) -> list[tuple[OutputRef, list[Action]]]:
    """ Group actions into (output_ref, computation_block) pairs. """
    blocks: list[tuple[OutputRef, list[Action]]] = []
    current_output: OutputRef | None = None
    current_block: list[Action] = []
    in_output_section = False

    for action in actions:
        if action.kind == "output_decl":
            if current_output is not None:
                blocks.append((current_output, current_block))
            current_output = action.data["ref"]
            current_block = []
            in_output_section = True
        elif in_output_section:
            current_block.append(action)

    if current_output is not None:
        blocks.append((current_output, current_block))

    return blocks

def _emit_decoder_block(
    actions: list[Action], lines: list[str], indent: int,
    sample_reg: int, n_reg: int, loop_reg: int, out_reg: int,
) -> None:
    """ Emit decoder computation block actions. """
    for action in actions:
        kind = action.kind
        d = action.data

        if kind == "range_start":
            start_expr: Expr = d["start_expr"]
            end_expr: Expr = d["end_expr"]

            _emit_decoder_expr(start_expr, lines, indent, n_reg)

            if isinstance(start_expr, Literal) and start_expr.value == 0:
                _emit_decoder_expr(end_expr, lines, indent, n_reg)
            else:
                _emit_decoder_expr(end_expr, lines, indent, n_reg)
                _emit_decoder_expr(start_expr, lines, indent, n_reg)
                lines.append(line("SUB", indent))

            lines.append(line("RANGE", indent))
            indent += 1
            lines.append(line(f"LVAL r{loop_reg}", indent))

        elif kind == "range_end":
            indent -= 1
            lines.append(line("NEXT", indent))

        elif kind == "output_append":
            _emit_decoder_value_expr(d["value_expr"], lines, indent, sample_reg, loop_reg)
            lines.append(line(f"VECPUSH r{out_reg}", indent))

def _emit_decoder_expr(expr: Expr, lines: list[str], indent: int, n_reg: int) -> None:
    """ Emit an expression in decoder context, remapping InputRef to N register. """
    if isinstance(expr, Literal):
        expr.emit(lines, indent)
    elif isinstance(expr, (InputRef, RegLoad)):
        lines.append(line(f"LOAD r{n_reg}", indent))
    elif isinstance(expr, BinOp):
        _emit_decoder_expr(expr.left, lines, indent, n_reg)
        _emit_decoder_expr(expr.right, lines, indent, n_reg)
        lines.append(line(expr.op, indent))
    else:
        expr.emit(lines, indent)

def _emit_decoder_value_expr(
    expr: Expr, lines: list[str], indent: int,
    sample_reg: int, loop_reg: int,
) -> None:
    """ Emit a value expression in decoder context (inside loop). """
    if isinstance(expr, ColFindExpr):
        col = expr.col_expr
        if isinstance(col, (LoopVar, RegLoad)):
            lines.append(line(f"LOAD r{loop_reg}", indent))
        else:
            col.emit(lines, indent)
        lines.append(line(f"PUSH {fmt_int(expr.value)}", indent))
        lines.append(line(f"COLFIND r{sample_reg}", indent))
    elif isinstance(expr, GetLineExpr):
        idx = expr.index_expr
        if isinstance(idx, (LoopVar, RegLoad)):
            lines.append(line(f"LOAD r{loop_reg}", indent))
        else:
            idx.emit(lines, indent)
        lines.append(line(f"GETLINE r{sample_reg}", indent))
    else:
        expr.emit(lines, indent)
