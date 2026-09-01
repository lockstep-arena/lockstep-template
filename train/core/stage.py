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
from pathlib import Path


def stage(
    slug: str,
    mode: str,
    payload_schema_version: int,
    onnx: Path,
    shell_wasm: Path,
    bundle: Path,
) -> Path:
    """Write the agent bundle and return its directory.

    ``payload_schema_version`` is read from the ENGINE (the wasm's own
    descriptor, via ``lockstep_train``), never typed by hand: the api
    refuses an agent whose declared version does not match the live
    environment catalog.
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
    )
    return bundle
