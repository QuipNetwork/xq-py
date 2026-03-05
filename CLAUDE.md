# Claude Guidelines

## Preferences

- File Changes: When possible, aggregate all changes for a file before displaying and apply them all together.
- Dependecies: Ask before injecting new dependencies into `requirements.txt`
- Virtual Environment: Always use the local virtual environment defined in `.venv`

## References

- Reference `~/XQVM_SPEC.md` for technical specification for the XQVM. Update this specification after confirmation whenever changes are made to implementation independent details of the XQVM such as opcode table and behaviour, control flow rules, stack depth, type system and expansions for HLFs. (non-exhaustive list)
- Reference `~/STYLING.md` for code style conventions. Always adhere to this style guide. If new code constructs are used, ask for style convention before adding to the style guide.
