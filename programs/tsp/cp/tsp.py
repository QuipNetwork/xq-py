"""
TSP Constraint Program (XQCP).

Compiles a TSP problem description into three XQVM assembly programs
(encoder, verifier, decoder) using the constraint programming DSL.

Usage:
    python programs/tsp/cp/tsp.py           # compile and write .xqasm files
    python programs/tsp/cp/tsp.py --print   # print generated assembly to stdout
"""

from pathlib import Path

from xqcp import Problem, Types, triu
from xqvm.core import XQMXDomain

OUTPUT_DIR = Path(__file__).parent


def build() -> Problem:
    """Build the TSP constraint program."""
    problem = Problem("TSP")

    # --- Inputs ---
    num_cities = problem.input("num_cities", type=Types.Int)
    distance_matrix = problem.input("distance_matrix", type=Types.Vec)

    # --- Model: NxN binary grid (city x position) ---
    problem.define_model(
        size=num_cities * num_cities,
        domain=XQMXDomain.BINARY,
        rows=num_cities,
        cols=num_cities,
    )

    # --- Objective: distance couplings ---
    with problem.range(0, num_cities - 1) as city_i:
        with problem.range(city_i + 1, num_cities) as city_j:
            dist = problem.stow("dist", distance_matrix.get(triu(city_i, city_j)))

            with problem.range(0, num_cities) as position:
                next_position = (position + 1) % num_cities

                problem.model.add_quadratic(
                    (city_i, position),
                    (city_j, next_position),
                    dist,
                )
                problem.model.add_quadratic(
                    (city_j, position),
                    (city_i, next_position),
                    dist,
                )

    # --- Constraints: one-hot rows and columns ---
    with problem.range(0, num_cities) as city:
        problem.model.apply_onehot_row(city, penalty=100)

    with problem.range(0, num_cities) as position:
        problem.model.apply_onehot_col(position, penalty=100)

    # --- Decoder output: extract tour ---
    tour = problem.output("tour", type=Types.Vec)

    with problem.range(0, num_cities) as position:
        tour.append(problem.sample.colfind(col=position, value=1))

    return problem


def compile_and_write() -> None:
    """Compile the TSP program and write .xqasm files."""
    problem = build()
    programs = problem.compile()

    (OUTPUT_DIR / "encoder.xqasm").write_text(programs.encoder)
    (OUTPUT_DIR / "verifier.xqasm").write_text(programs.verifier)
    (OUTPUT_DIR / "decoder.xqasm").write_text(programs.decoder)

    print(f"Wrote encoder.xqasm, verifier.xqasm, decoder.xqasm to {OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile TSP XQCP program")
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_asm",
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
