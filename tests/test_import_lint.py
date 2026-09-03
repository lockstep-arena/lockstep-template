"""The genericity gate: this template names no environment, ever.

Environments are discovered from the platform at runtime
(train/core/discovery.py — `task envs`) and described by their own engines'
declarations; no slug is hardcoded anywhere in the repo, not as a default,
not in CI, not in a docs example. The ONLY
Lockstep package train/ may import is ``lockstep_train`` — the generic,
environment-agnostic host. A direct import of anything else (or any
environment slug appearing in the training core) would silently re-couple
the template to one environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO_DIR / "train"

#: Text files the slug lint scans, repo-wide.
LINTED_SUFFIXES = {".py", ".yml", ".yaml", ".md", ".toml", ".txt"}
#: Top-level directories that are not the template's own text: git
#: internals, the venv, build output, and researchers' own agent scaffolds.
LINT_SKIP_DIRS = {".git", ".venv", "out", "agents", "__pycache__"}

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


#: Fragments of environment slugs that must never appear in the template.
#: Assembled at runtime so this file does not itself name one.
BANNED_FRAGMENTS = ("".join(("dan", "ce")),)


def _lintable_files():
    for path in sorted(REPO_DIR.rglob("*")):
        if not path.is_file() or path.suffix not in LINTED_SUFFIXES:
            continue
        rel = path.relative_to(REPO_DIR)
        if any(part in LINT_SKIP_DIRS for part in rel.parts):
            continue
        yield path


def test_no_environment_slug_anywhere():
    """No environment slug is named anywhere in the template — not in train/,
    not as a Taskfile default, not in CI, not in a docs example or a test
    fixture. Environments come from the platform at runtime (`task envs`);
    docs say `ENV=<slug>`, CI asks discovery for one, tests mock invented
    slugs.
    """
    offenders = []
    for path in _lintable_files():
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            low = line.lower()
            if any(fragment in low for fragment in BANNED_FRAGMENTS):
                offenders.append(f"{path.relative_to(REPO_DIR)}:{line_no}")
    assert not offenders, f"environment slug named in the template: {offenders}"
