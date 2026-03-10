# Claude Guidelines

## Preferences

- **File Changes:** Aggregate all changes for a file and apply them together. Do not make multiple small edits to the same file in sequence.
- **Dependencies:** Never add dependencies to `requirements.txt` without explicit user approval.
- **Virtual Environment:** Always use the local virtual environment defined in `.venv`. Never install packages globally or create new environments.

## References

- **Specification (`~/XQVM_SPEC.md`):** This is the authoritative source for XQVM architecture. Read relevant sections before modifying VM behaviour. Update the spec (after confirmation) whenever changes affect implementation-independent details: opcode table, control flow rules, stack depth, type system, HLF expansions (non-exhaustive list).
- **Style Guide (`~/STYLING.md`):** Read before writing or editing any Python file. All code — new files, edits, and generated tests — must comply with every rule in the style guide. Do not fall back to PEP 8 defaults where they conflict (e.g. PEP 8 uses double blank lines between top-level definitions; this project uses single). If new code constructs are used, ask for style convention before adding to the style guide.
- **Plan (`~/XQVM_PLAN.md`):** Reference for project structure and phase status. Update phase status and milestones after completing phase deliverables.
