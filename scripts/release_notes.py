"""Print the package version, or this version's CHANGELOG section.

Used by .github/workflows/release.yml:

    python scripts/release_notes.py --version   # 0.1.0
    python scripts/release_notes.py             # release notes for v0.1.0
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def version() -> str:
    text = (ROOT / "src" / "pde_agent" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
    if not match:
        sys.exit("__version__ not found in src/pde_agent/__init__.py")
    return match.group(1)


def notes(ver: str) -> str:
    """The body of the ``## v<ver>`` section of CHANGELOG.md."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(rf"^## v{re.escape(ver)}\b[^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not match or not match.group(1).strip():
        sys.exit(f"CHANGELOG.md has no section for v{ver}; write one before releasing")
    return match.group(1).strip() + "\n"


if __name__ == "__main__":
    ver = version()
    print(ver if "--version" in sys.argv[1:] else notes(ver), end="\n" if "--version" in sys.argv[1:] else "")
