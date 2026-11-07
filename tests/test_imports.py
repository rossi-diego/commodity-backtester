"""The page, the scripts and the tests use only the public names of ``src``."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "app.py", *sorted(ROOT.glob("scripts/*.py")), *sorted(ROOT.glob("tests/*.py"))]


def _private_imports(source: str) -> list[str]:
    """Each ``from src... import _name`` in ``source``, as ``module._name``."""
    return [
        f"{node.module}.{alias.name}"
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (node.module == "src" or node.module.startswith("src."))
        for alias in node.names
        if alias.name.startswith("_")
    ]


def test_no_private_name_is_imported_from_src() -> None:
    found = {
        path.relative_to(ROOT).as_posix(): names
        for path in SOURCES
        if (names := _private_imports(path.read_text(encoding="utf-8")))
    }
    assert not found, found


def test_the_check_sees_a_private_import() -> None:
    source = (
        "from src.rins import load_d4, _write\n"
        "from src import _x\n"
        "from scripts.readme_figures import _p_value\n"
        "def f():\n    from src.risk import _scenarios\n"
    )
    assert sorted(_private_imports(source)) == ["src._x", "src.rins._write", "src.risk._scenarios"]
