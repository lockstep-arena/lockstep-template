"""Stage the submittable agent bundle from a GameSpec.

The staged bundle is what the platform actually consumes — and what
``lockstep match run`` / ``lockstep agent upload`` take directly::

    <bundle>/
      lockstep.toml        declares the `policy` artifact by NAME
      component.wasm       the mode's prebuilt agent shell (from the wheel)
      artifacts/policy.onnx

The component is NOT trained here: it is the fixed WASM shell that feeds
observations to whatever ``policy.onnx`` you put next to it, shipped inside
the game's training wheel and version-coupled to the observation codec the
env trains against.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def stage(spec: dict, mode: str, onnx: Path, bundle: Path) -> Path:
    """Write the agent bundle for ``mode`` and return its directory."""
    payload_schema_version = spec["modes"][mode]["payload_schema_version"]
    (bundle / "artifacts").mkdir(parents=True, exist_ok=True)
    (bundle / "artifacts/policy.onnx").write_bytes(Path(onnx).read_bytes())
    shutil.copyfile(spec["agent_component_path"](mode), bundle / "component.wasm")
    (bundle / "lockstep.toml").write_text(
        "# Staged by train/core/stage.py — do not hand-edit.\n"
        "#\n"
        "# `policy` is the artifact NAME the shell passes to `infer()`.\n"
        "schema_version = 1\n"
        f'game = "{spec["slug"]}"\n'
        "# Read from the game package, never typed here: the api refuses an\n"
        "# agent whose declared version does not match the live game catalog.\n"
        f"payload_schema_version = {payload_schema_version}\n"
        "# The ladder this agent targets (games can have more than one mode).\n"
        f'mode = "{mode}"\n'
        "\n"
        "[artifacts.policy]\n"
        'kind = "onnx"\n'
        'path = "artifacts/policy.onnx"\n'
    )
    return bundle
