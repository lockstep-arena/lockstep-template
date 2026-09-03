"""Stage the submittable agent bundle.

The staged bundle is what the platform actually consumes — and what
``lockstep match run`` / ``lockstep agent upload`` take directly::

    <bundle>/
      lockstep.toml        declares the `policy` artifact by NAME
      component.wasm       the GENERIC ONNX agent shell (from the cached release)
      artifacts/policy.onnx

The component is NOT trained here and is not per-environment either: it is
the one generic shell every environment shares. It decodes the Lockstep-wire
seat-init, feeds each observation to the ONNX graph BY NAME (u8
images as f32/255, batch dim prepended) and maps the graph's ``action``
output from [-1, 1] onto the declared action bounds. The environment
specifics live entirely in ``policy.onnx``'s learned weights.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

#: What every bundle staged by this repo records as its `[provenance] tool`.
TOOL = "lockstep-template"


def template_version() -> str:
    """The template's own identity: its git commit when it is a checkout
    (the normal case — the README says clone it), else ``unversioned``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unversioned"
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else "unversioned"


def provenance(
    *,
    steps: int | None = None,
    num_envs: int | None = None,
    trained: bool,
    tool_version: str | None = None,
) -> dict:
    """The optional ``[provenance]`` table a bundle carries: which tool staged
    it, at what version, and — for a trained policy — the recipe (``steps``,
    ``num_envs``) and when. Rendered under the platform's "Technical details";
    it is a self-report, so it is context for an interviewer, not a check."""
    table: dict = {
        "tool": TOOL,
        "tool_version": tool_version or template_version(),
        "kind": "trained" if trained else "built",
    }
    if trained:
        if steps is not None:
            table["steps"] = int(steps)
        if num_envs is not None:
            table["num_envs"] = int(num_envs)
        table["trained_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return table


def provenance_toml(table: dict) -> str:
    """Serialize the provenance table as a TOML block (strings quoted, ints bare)."""
    lines = ["", "# Self-reported by the template: which tool staged this bundle and, for a", "# trained policy, the recipe. Shown to the employer under Technical details.", "[provenance]"]
    for k, v in table.items():
        lines.append(f"{k} = {v}" if isinstance(v, int) else f'{k} = "{v}"')
    return "\n".join(lines) + "\n"


def stage(
    slug: str,
    mode: str,
    payload_schema_version: int,
    onnx: Path,
    shell_wasm: Path,
    bundle: Path,
    provenance_table: dict | None = None,
) -> Path:
    """Write the agent bundle and return its directory.

    ``payload_schema_version`` is read from the ENGINE (the wasm's own
    descriptor, via ``lockstep_train``), never typed by hand: the api
    refuses an agent whose declared version does not match the live
    environment catalog. ``provenance_table`` (see [`provenance`]) is
    written as the optional ``[provenance]`` table when given.
    """
    if not shell_wasm.is_file():
        raise SystemExit(
            f"no agent shell at {shell_wasm} — any engine-using task refetches it (task info ENV={slug})"
        )
    (bundle / "artifacts").mkdir(parents=True, exist_ok=True)
    (bundle / "artifacts/policy.onnx").write_bytes(Path(onnx).read_bytes())
    shutil.copyfile(shell_wasm, bundle / "component.wasm")
    (bundle / "lockstep.toml").write_text(
        "# Staged by train/core/stage.py — do not hand-edit.\n"
        "#\n"
        "# `policy` is the artifact NAME the shell passes to `infer()`.\n"
        "schema_version = 2\n"
        f'environment = "{slug}"\n'
        "# Read from the engine's own descriptor, never typed here: the api\n"
        "# refuses an agent whose declared version does not match the live\n"
        "# environment catalog.\n"
        f"payload_schema_version = {payload_schema_version}\n"
        "# The ladder this agent targets (environments can have several modes).\n"
        f'mode = "{mode}"\n'
        "\n"
        "[artifacts.policy]\n"
        'kind = "onnx"\n'
        'path = "artifacts/policy.onnx"\n'
        + (provenance_toml(provenance_table) if provenance_table else ""),
        encoding="utf-8",
    )
    return bundle
