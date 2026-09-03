"""``task build AGENT=`` — build one agent into its submittable bundle.

What "build" means depends on the agent's language (its ``agent.toml``
says; you never do):

- **python** — export the agent's OWN ``policy.py`` (the hand-written
  ``ScriptedPolicy``, untrained unless you edited it) to ONNX,
  parity-check torch vs onnxruntime, and stage the bundle with the
  release's generic ONNX shell. This is the no-training path; ``task
  train`` stages the PPO-trained policy into the same bundle slot.
- **rust** — ``cargo build --release --target wasm32-wasip2`` (the crate
  the scaffolder laid out) and bundle the resulting component.
- **c** — run the agent's own ``build.sh`` (wit-bindgen c + wasi-sdk
  clang → component) and bundle the result.

Every path ends at the same place::

    agents/<name>/out/bundle/
      lockstep.toml     environment/mode/payload_schema_version from agent.toml
      component.wasm    the agent (generic ONNX shell for python)
      [artifacts/policy.onnx]   python only

which is exactly what ``task match AGENT=`` and ``task upload AGENT=``
consume.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .agents import AgentConfig, resolve_agent
from .core import utf8_output
from .core.engine import ensure_engine
from .core.stage import provenance, provenance_toml, stage


def manifest_text(cfg: AgentConfig, with_policy: bool) -> str:
    head = (
        "# Staged by `task build` — do not hand-edit.\n"
        "schema_version = 2\n"
        f'environment = "{cfg.env}"\n'
        "# Read from the engine's descriptor at scaffold time, never typed by\n"
        "# hand: the api refuses an agent whose declared version does not\n"
        "# match the live environment catalog.\n"
        f"payload_schema_version = {cfg.payload_schema_version}\n"
        f'mode = "{cfg.mode}"\n'
    )
    if with_policy:
        head += '\n[artifacts.policy]\nkind = "onnx"\npath = "artifacts/policy.onnx"\n'
    # A hand-written policy (python export, rust, c): the template built it,
    # nothing trained it — say so, so the employer's Technical details do not
    # have to guess.
    head += provenance_toml(provenance(trained=False))
    return head


def bundle_component(cfg: AgentConfig, component: Path) -> Path:
    bundle = cfg.bundle_dir
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(component, bundle / "component.wasm")
    (bundle / "lockstep.toml").write_text(manifest_text(cfg, with_policy=False), encoding="utf-8")
    return bundle


def build_python(cfg: AgentConfig) -> Path:
    import importlib

    import torch  # noqa: F401 — fail here, with the venv hint, not deeper

    from .core.export import export, verify
    from .main import engine_identity

    paths = ensure_engine(cfg.env, cfg.mode)
    mode, payload_schema_version = engine_identity(paths.engine)

    # The agent's own policy module (agents/<name>/policy.py). Loaded with
    # the agent dir on sys.path (names like `my-bot` are not importable as
    # packages); its generated sibling `interface.py` resolves the same way.
    sys.path.insert(0, str(cfg.dir.resolve()))
    try:
        importlib.invalidate_caches()
        policy_mod = importlib.import_module("policy")
    finally:
        sys.path.pop(0)

    # Signature straight from the engine's declaration, exactly as the
    # generic shell will bind it at match time.
    import numpy as np
    from lockstep_train.env import LockstepEnv

    env = LockstepEnv(engine_source=str(paths.engine))
    try:
        shapes = {n: tuple(int(d) for d in b.shape) for n, b in env.observation_space.spaces.items()}
        dtypes = {
            n: "int32" if b.dtype == np.int32 else "float32"
            for n, b in env.observation_space.spaces.items()
        }
        (action_len,) = env.action_space.shape
    finally:
        env.close()

    net = policy_mod.ScriptedPolicy(shapes, dtypes, int(action_len))
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    onnx = export(net, out / "policy.onnx")
    print(f"→ onnx: {onnx} ({onnx.stat().st_size} bytes)", file=sys.stderr)
    diff = verify(net, onnx)
    print(f"✓ torch/onnxruntime parity: max abs diff {diff:.3e}", file=sys.stderr)
    return stage(
        cfg.env,
        mode,
        payload_schema_version,
        onnx,
        paths.shell,
        cfg.bundle_dir,
        provenance_table=provenance(trained=False),
    )


def build_rust(cfg: AgentConfig) -> Path:
    crate_name = None
    for line in (cfg.dir / "Cargo.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("name = "):
            crate_name = line.split('"')[1]
            break
    if not crate_name:
        raise SystemExit(f"{cfg.dir}/Cargo.toml has no [package] name")
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
    component = (
        cfg.dir / "target" / "wasm32-wasip2" / "release" / f"{crate_name.replace('-', '_')}.wasm"
    )
    if not component.is_file():
        raise SystemExit(f"cargo succeeded but {component} is missing")
    return bundle_component(cfg, component)


def compile_c_agent(agent_dir: Path) -> Path:
    """The C build, portably (no bash — Windows resolves `bash` to the WSL
    stub): wit-bindgen world bindings, then one wasi-sdk clang line to a
    wasm32-wasip2 component. The scaffold's build.sh is the same two steps
    for running by hand on unix."""
    from .toolchain import find_wasi_sdk, find_wit_bindgen

    sdk = find_wasi_sdk()
    wit_bindgen = find_wit_bindgen()
    if sdk is None or wit_bindgen is None:
        raise SystemExit("the C toolchain is missing — run: task setup LANGS=c")
    clang = sdk / "bin" / ("clang.exe" if os.name == "nt" else "clang")
    subprocess.run(
        [wit_bindgen, "c", "wit", "--world", "agent", "--out-dir", "gen"],
        check=True,
        cwd=agent_dir,
    )
    (agent_dir / "out").mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(clang),
            "--target=wasm32-wasip2",
            "-mexec-model=reactor",
            "-O2",
            "-o",
            "out/agent.wasm",
            "agent.c",
            "wire.c",
            "gen/agent.c",
            "gen/agent_component_type.o",
        ],
        check=True,
        cwd=agent_dir,
    )
    return agent_dir / "out" / "agent.wasm"


def build_c(cfg: AgentConfig) -> Path:
    component = compile_c_agent(cfg.dir)
    if not component.is_file():
        raise SystemExit(f"the C build succeeded but {component} is missing")
    return bundle_component(cfg, component)


BUILDERS = {"python": build_python, "rust": build_rust, "c": build_c}


def main() -> None:
    utf8_output()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", default=None, help="agent name (AGENT= on the task line)")
    args = p.parse_args()
    cfg = resolve_agent(args.agent)
    bundle = BUILDERS[cfg.lang](cfg)
    print(f"→ bundle: {bundle}", file=sys.stderr)
    print(
        f"\nRun it:   task match AGENT={cfg.name}\nCompete:  task upload AGENT={cfg.name}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    sys.exit(main())
