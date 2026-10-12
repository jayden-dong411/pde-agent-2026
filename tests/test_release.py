"""Every version must ship with release notes (CHANGELOG.md) before it is released."""

import subprocess
import sys
from pathlib import Path

import pde_agent

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "release_notes.py"


def run(*args: str) -> str:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=True).stdout


def test_script_reads_the_package_version():
    assert run("--version").strip() == pde_agent.__version__


def test_current_version_has_release_notes():
    notes = run()
    assert len(notes.strip()) > 20
    assert "\n## " not in notes  # only this version's section
