"""
Max-Cut Constraint Program (XQCP).

Compiles a Max-Cut problem description into three XQVM assembly programs
(encoder, verifier, decoder) using the constraint programming DSL.

Usage:
    python programs/maxcut/cp/maxcut.py           # compile and write .xqasm files
    python programs/maxcut/cp/maxcut.py --print   # print generated assembly to stdout
"""

from pathlib import Path

from xqvm.cp import Problem, Types
from xqvm.core import XQMXDomain

OUTPUT_DIR = Path(__file__).parent

def build() -> Problem:
    """ Build the Max-Cut constraint program. """
    problem = Problem("MaxCut")

    # --- Inputs ---
    num_nodes = problem.input("num_nodes", type=Types.Int)
    edges = problem.input("edges", type=Types.Vec)

    # --- Model: N binary variables (one per node) ---
    problem.define_model(size=num_nodes, domain=XQMXDomain.BINARY)

    # --- Objective: edge-based QUBO couplings ---
    edge_count = problem.stow("edge_count", edges.veclen() // 3)

    with problem.range(0, edge_count) as e:
        offset = e * 3
        i = problem.stow("i", edges.get(offset))
        j = problem.stow("j", edges.get(offset + 1))
        w = problem.stow("w", edges.get(offset + 2))

        problem.model.add_linear(i, -w)
        problem.model.add_linear(j, -w)
        problem.model.add_quadratic(i, j, w * 2)

    # --- Decoder output: extract partition ---
    partition = problem.output("partition", type=Types.Vec)

    with problem.range(0, num_nodes) as node:
        partition.append(problem.sample.getline(node))

    return problem

def compile_and_write() -> None:
    """ Compile the Max-Cut program and write .xqasm files. """
    problem = build()
    programs = problem.compile()

    (OUTPUT_DIR / "encoder.xqasm").write_text(programs.encoder)
    (OUTPUT_DIR / "verifier.xqasm").write_text(programs.verifier)
    (OUTPUT_DIR / "decoder.xqasm").write_text(programs.decoder)

    print(f"Wrote encoder.xqasm, verifier.xqasm, decoder.xqasm to {OUTPUT_DIR}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile Max-Cut XQCP program")
    parser.add_argument(
        "--print", action="store_true", dest="print_asm",
        help="Print generated assembly instead of writing files",
    )
    args = parser.parse_args()

    if args.print_asm:
        problem = build()
        programs = problem.compile()
        for name, source in [
            ("encoder", programs.encoder),
            ("verifier", programs.verifier),
            ("decoder", programs.decoder),
        ]:
            print(f"# {'=' * 60}")
            print(f"# {name}.xqasm")
            print(f"# {'=' * 60}")
            print(source)
    else:
        compile_and_write()
