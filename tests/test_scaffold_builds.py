"""The rust/c scaffolds COMPILE to wasm components, straight from the
synthetic declaration (no network, no engine — the stubs build as
scaffolded).

Toolchain-gated locally (skip when cargo/wasm32-wasip2 or
wasi-sdk/wit-bindgen are absent); CI installs both and runs them
unconditionally.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from tests.test_scaffold import BUDGETS, cfg_for, synthetic_init
from train import agents as agents_mod
from train import scaffold as scaffold_mod
from train.toolchain import find_wasi_sdk, find_wit_bindgen


@pytest.fixture
def agent_env(tmp_path, monkeypatch):
    root = tmp_path / "agents"
    monkeypatch.setattr(agents_mod, "AGENTS_ROOT", root)
    return root


def _has_wasip2() -> bool:
    if not shutil.which("cargo"):
        return False
    rustup = shutil.which("rustup")
    if not rustup:
        return True  # non-rustup install: let the build speak
    out = subprocess.run(
        [rustup, "target", "list", "--installed"], capture_output=True, text=True
    ).stdout
    return "wasm32-wasip2" in out


@pytest.mark.skipif(not _has_wasip2(), reason="no cargo/wasm32-wasip2 — task setup LANGS=rust")
def test_rust_scaffold_compiles_to_a_component(agent_env):
    cfg = cfg_for("rust")
    scaffold_mod.scaffold_rust(cfg, synthetic_init(), BUDGETS, "t")
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--target",
            "wasm32-wasip2",
            "--manifest-path",
            str(cfg.dir / "Cargo.toml"),
        ],
        check=True,
    )
    wasm = cfg.dir / "target" / "wasm32-wasip2" / "release" / "testling.wasm"
    assert wasm.is_file()
    _assert_component(wasm)


@pytest.mark.skipif(
    find_wasi_sdk() is None or find_wit_bindgen() is None,
    reason="no wasi-sdk/wit-bindgen — task setup LANGS=c",
)
def test_c_scaffold_compiles_to_a_component(agent_env):
    from train.build import compile_c_agent

    cfg = cfg_for("c")
    scaffold_mod.scaffold_c(cfg, synthetic_init(), BUDGETS, "t")
    wasm = compile_c_agent(cfg.dir)
    assert wasm.is_file()
    _assert_component(wasm)


def _assert_component(wasm) -> None:
    """`wasm-tools component wit` succeeds only on a real component and its
    world must export the agent surface. Skipped (not failed) when
    wasm-tools isn't installed — the build succeeding is the main claim."""
    if not shutil.which("wasm-tools"):
        pytest.skip("wasm-tools not installed — component shape not double-checked")
    out = subprocess.run(
        ["wasm-tools", "component", "wit", str(wasm)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "export init" in out and "export on-tick" in out, out
