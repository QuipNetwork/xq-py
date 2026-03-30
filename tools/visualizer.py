"""
XQVM XQMX ASCII Visualizer: Render quadratic matrices as text.
"""

from __future__ import annotations

from xqvm.core.xqmx import XQMX, XQMXMode


def render_info(xqmx: XQMX) -> str:
    """Render a summary header for an XQMX."""
    mode = "MODEL" if xqmx.mode == XQMXMode.MODEL else "SAMPLE"
    domain = xqmx.domain.name
    nnz_linear = len(xqmx.linear)
    nnz_quad = len(xqmx.quadratic)

    lines = [
        f"XQMX {mode} ({domain})",
        f"  size: {xqmx.size}",
    ]

    if xqmx.rows > 0 and xqmx.cols > 0:
        lines.append(f"  grid: {xqmx.rows}x{xqmx.cols}")

    lines.append(f"  linear:    {nnz_linear} non-zero")
    lines.append(f"  quadratic: {nnz_quad} non-zero")
    return "\n".join(lines)


def render_matrix(xqmx: XQMX) -> str:
    """
    Render an XQMX as an ASCII matrix.

    For MODEL mode: linear on diagonal, quadratic off-diagonal.
    For SAMPLE mode: linear assignments in grid layout (rows x cols).
    """
    if xqmx.size == 0:
        return "(empty)"

    if xqmx.mode == XQMXMode.SAMPLE and xqmx.rows > 0 and xqmx.cols > 0:
        return _render_sample_grid(xqmx)

    return _render_model_matrix(xqmx)


def _render_model_matrix(xqmx: XQMX) -> str:
    """Render model as NxN matrix with linear on diagonal, quadratic off-diagonal."""
    n = xqmx.size

    # Collect all values to determine column width
    cells: list[list[str]] = []
    for i in range(n):
        row: list[str] = []
        for j in range(n):
            if i == j:
                val = xqmx.linear.get(i, 0)
            elif i < j:
                val = xqmx.quadratic.get((i, j), 0)
            else:
                val = xqmx.quadratic.get((j, i), 0)

            if val == 0:
                row.append(".")
            else:
                row.append(_fmt_val(val))
        cells.append(row)

    return _format_grid(cells, n, n)


def _render_sample_grid(xqmx: XQMX) -> str:
    """Render sample as rows x cols grid of variable assignments."""
    rows, cols = xqmx.rows, xqmx.cols
    cells: list[list[str]] = []

    for r in range(rows):
        row: list[str] = []
        for c in range(cols):
            idx = r * cols + c
            val = xqmx.linear.get(idx, 0)

            if val == 0:
                row.append(".")
            else:
                row.append(_fmt_val(val))
        cells.append(row)

    return _format_grid(cells, rows, cols)


def _format_grid(cells: list[list[str]], rows: int, cols: int) -> str:
    """Format a 2D list of strings into aligned columns."""
    col_widths = []
    for c in range(cols):
        width = max(len(cells[r][c]) for r in range(rows))
        col_widths.append(max(width, 1))

    lines: list[str] = []
    for r in range(rows):
        parts = [cells[r][c].rjust(col_widths[c]) for c in range(cols)]
        lines.append(" ".join(parts))

    return "\n".join(lines)


def render_sparsity(xqmx: XQMX) -> str:
    """
    Render sparsity pattern of an XQMX.

    Shows '#' for non-zero entries, '.' for zero.
    For MODEL: NxN with linear on diagonal, quadratic off-diagonal.
    For SAMPLE: rows x cols grid.
    """
    if xqmx.size == 0:
        return "(empty)"

    if xqmx.mode == XQMXMode.SAMPLE and xqmx.rows > 0 and xqmx.cols > 0:
        rows, cols = xqmx.rows, xqmx.cols
        lines: list[str] = []

        for r in range(rows):
            row_chars: list[str] = []
            for c in range(cols):
                idx = r * cols + c
                row_chars.append("#" if idx in xqmx.linear else ".")
            lines.append("".join(row_chars))

        return "\n".join(lines)

    # MODEL: NxN
    n = xqmx.size
    lines = []

    for i in range(n):
        row_chars = []
        for j in range(n):
            if i == j:
                row_chars.append("#" if i in xqmx.linear else ".")
            elif i < j:
                row_chars.append("#" if (i, j) in xqmx.quadratic else ".")
            else:
                row_chars.append("#" if (j, i) in xqmx.quadratic else ".")
        lines.append("".join(row_chars))

    return "\n".join(lines)


def _fmt_val(val: float | int) -> str:
    """Format a numeric value for display."""
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)
