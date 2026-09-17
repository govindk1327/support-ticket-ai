"""Support Ticket AI.

Fails fast with a readable message on unsupported Python versions. Without this
guard, Python 3.9 raises a pydantic ForwardRef error deep in model construction
that does not mention the real cause, which is the `X | None` annotation syntax
introduced in 3.10.
"""

import sys

MINIMUM_PYTHON = (3, 10)

if sys.version_info < MINIMUM_PYTHON:
    raise RuntimeError(
        f"Support Ticket AI requires Python "
        f"{MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or newer, but this "
        f"interpreter is {sys.version_info.major}.{sys.version_info.minor}"
        f".{sys.version_info.micro} at {sys.executable}.\n\n"
        f"The codebase uses `X | None` type annotations (PEP 604), which "
        f"earlier versions cannot evaluate at runtime.\n\n"
        f"Fix: recreate the virtual environment with a newer interpreter.\n"
        f"  Windows:      py -3.12 -m venv .venv\n"
        f"  macOS/Linux:  python3.12 -m venv .venv\n"
        f"then reinstall with: pip install -r requirements.txt"
    )
