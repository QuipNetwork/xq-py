# Claude Guidelines

## Preferences

- **File Changes:** Aggregate all changes for a file and apply them together. Do not make multiple small edits to the same file in sequence.
- **Dependencies:** Never add dependencies to `requirements.txt` without explicit user approval.
- **Virtual Environment:** Always use the local virtual environment defined in `.venv`. Never install packages globally or create new environments.
- **Formatting:** All Python code must pass `ruff check .` and `ruff format --check .` (config in `ruff.toml`). After creating or modifying Python files, run `ruff check --fix <file>` and `ruff format <file>`.

## References

- **Specification (`~/XQVM_SPEC.md`):** This is the authoritative source for XQVM architecture. Read relevant sections before modifying VM behaviour. Update the spec (after confirmation) whenever changes affect implementation-independent details: opcode table, control flow rules, stack depth, type system, HLF expansions (non-exhaustive list).
- **Plan (`~/XQVM_PLAN.md`):** Reference for project structure and phase status. Update phase status and milestones after completing phase deliverables.
