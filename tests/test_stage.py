"""The staged bundle: manifest v2 shape + the generic shell next to the
policy. This is the exact directory ``lockstep agent upload`` consumes."""

from __future__ import annotations

import tomllib

from train.core.stage import provenance, provenance_toml, stage


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


def test_stage_writes_the_provenance_table_when_given(tmp_path):
    onnx = tmp_path / "policy.onnx"
    onnx.write_bytes(b"x")
    shell = tmp_path / "agent-onnx.wasm"
    shell.write_bytes(b"s")
    table = provenance(steps=2_000_000, num_envs=8, trained=True, tool_version="abc1234")

    bundle = stage("go1-beacon", "default", 2, onnx, shell, tmp_path / "bundle", provenance_table=table)

    manifest = tomllib.loads((bundle / "lockstep.toml").read_text())
    prov = manifest["provenance"]
    assert prov["tool"] == "lockstep-template"
    assert prov["tool_version"] == "abc1234"
    assert prov["kind"] == "trained"
    assert prov["steps"] == 2_000_000 and prov["num_envs"] == 8
    assert prov["trained_at"].endswith("Z") and len(prov["trained_at"]) == 20
    # The rest of the manifest is untouched by the optional table.
    assert manifest["artifacts"]["policy"]["kind"] == "onnx"


def test_stage_without_provenance_writes_none(tmp_path):
    onnx = tmp_path / "policy.onnx"
    onnx.write_bytes(b"x")
    shell = tmp_path / "agent-onnx.wasm"
    shell.write_bytes(b"s")
    bundle = stage("go1-beacon", "default", 2, onnx, shell, tmp_path / "bundle")
    assert "provenance" not in tomllib.loads((bundle / "lockstep.toml").read_text())


def test_built_provenance_has_no_recipe():
    table = provenance(trained=False, tool_version="v")
    assert table == {"tool": "lockstep-template", "tool_version": "v", "kind": "built"}
    parsed = tomllib.loads(provenance_toml(table))
    assert parsed["provenance"] == table
