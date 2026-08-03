"""The genericity gate: train/ must not import any game package, ever.

Games are reached ONLY through entry-point discovery
(train/core/discovery.py). A direct import would silently re-couple the
template to one game — this test walks every module under train/ and fails
on any static import of a lockstep package other than the discovery
machinery's own stdlib imports. (Game packages are all named
``lockstep_<something>``; the template legitimately imports NONE of them —
not even ``lockstep_train``, which is the game wheel's dependency, not
ours.)
"""

from __future__ import annotations

import ast
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
    return found


def test_train_imports_no_game_package():
    offenders = {}
    for path in sorted(TRAIN_DIR.rglob("*.py")):
        bad = {
            mod
            for mod in _imported_modules(path)
            if mod.split(".")[0].startswith("lockstep")
        }
        if bad:
            offenders[str(path.relative_to(TRAIN_DIR.parent))] = sorted(bad)
    assert not offenders, (
        f"train/ statically imports game/platform packages: {offenders} — "
        "games must be reached through entry-point discovery only"
    )


def test_train_names_no_game():
    """No game slug appears in train/ source — not even in comments.

    dance-off is allowed to appear in docs, examples and CI fixture config;
    the training core is where it must never appear.
    """
    offenders = []
    for path in sorted(TRAIN_DIR.rglob("*.py")):
        text = path.read_text().lower()
        for line_no, line in enumerate(text.splitlines(), 1):
            if "dance" in line:
                offenders.append(f"{path.name}:{line_no}")
    assert not offenders, f"game-specific references in train/: {offenders}"
