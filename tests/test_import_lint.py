"""The genericity gate: train/ names no environment, ever.

Environments are resolved from the CDN at runtime (train/core/discovery.py)
and described by their own engines' declarations. The ONLY
Lockstep package train/ may import is ``lockstep_train`` — the generic,
environment-agnostic host. A direct import of anything else (or any
environment slug appearing in the training core) would silently re-couple
the template to one environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"

#: The one legitimate Lockstep import: the generic wasm-stepping host.
ALLOWED = {"lockstep_train"}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
    return found


def test_train_imports_only_the_generic_host():
    offenders = {}
    for path in sorted(TRAIN_DIR.rglob("*.py")):
        bad = {
            mod
            for mod in _imported_modules(path)
            if mod.split(".")[0].startswith("lockstep")
            and mod.split(".")[0] not in ALLOWED
        }
        if bad:
            offenders[str(path.relative_to(TRAIN_DIR.parent))] = sorted(bad)
    assert not offenders, (
        f"train/ imports non-generic lockstep packages: {offenders} — "
        "everything per-environment comes from the engine's own declaration"
    )


def test_train_names_no_environment():
    """No environment slug appears in train/ source — not even in comments.

    dance-off is allowed to appear in docs, examples and CI config; the
    training core is where it must never appear.
    """
    offenders = []
    for path in sorted(TRAIN_DIR.rglob("*.py")):
        text = path.read_text().lower()
        for line_no, line in enumerate(text.splitlines(), 1):
            if "dance" in line:
                offenders.append(f"{path.name}:{line_no}")
    assert not offenders, f"environment-specific references in train/: {offenders}"
