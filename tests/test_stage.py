"""The staged bundle: manifest v2 shape + the generic shell next to the
policy. This is the exact directory ``lockstep agent upload`` consumes."""

from __future__ import annotations

import tomllib

from train.core.stage import stage


def test_stage_writes_a_v2_manifest(tmp_path):
    onnx = tmp_path / "policy.onnx"
    onnx.write_bytes(b"onnx-bytes")
    shell = tmp_path / "agent-onnx.wasm"
    shell.write_bytes(b"shell-bytes")

    bundle = stage("panda-pick", "default", 3, onnx, shell, tmp_path / "bundle")

    manifest = tomllib.loads((bundle / "lockstep.toml").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["environment"] == "panda-pick"
    assert manifest["payload_schema_version"] == 3
    assert manifest["mode"] == "default"
    assert manifest["artifacts"]["policy"] == {
        "kind": "onnx",
        "path": "artifacts/policy.onnx",
    }
    assert (bundle / "component.wasm").read_bytes() == b"shell-bytes"
    assert (bundle / "artifacts/policy.onnx").read_bytes() == b"onnx-bytes"


def test_stage_without_the_shell_names_the_fix(tmp_path):
    onnx = tmp_path / "policy.onnx"
    onnx.write_bytes(b"x")
    import pytest

    with pytest.raises(SystemExit, match="task info ENV=panda-pick"):
        stage("panda-pick", "default", 1, onnx, tmp_path / "missing.wasm", tmp_path / "b")
