"""train -> export -> parity-check -> stage the submittable agent bundle.

One command (``task train`` runs it). The staged bundle is what the platform
actually consumes — and what ``lockstep match run`` / ``lockstep agent
upload`` take directly::

    out/agent-bundle/
      lockstep.toml        declares the `policy` artifact by NAME
      component.wasm       the mode's prebuilt agent shell (from the wheel)
      artifacts/policy.onnx

The component is NOT trained here: it is the fixed WASM shell that feeds
observations to whatever ``policy.onnx`` you put next to it, shipped inside
the ``lockstep-game-dance-off`` wheel and version-coupled to the observation
codec the env trains against.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

from lockstep_dance_off import MODE_RAW_TORQUE, MODE_SERVO_ASSIST, _native
from lockstep_dance_off.components import agent_component_path

from .export import export, verify
from .policy import Policy
from .train import train

GAME_SLUG = "dance-off"

OUT_DIR = Path("out")
BUNDLE_DIR = OUT_DIR / "agent-bundle"


def stage(mode: str, onnx: Path, bundle: Path = BUNDLE_DIR) -> Path:
    """Write the agent bundle for ``mode`` and return its directory.

    ``bundle`` defaults to the trained agent's home; the scripted example
    stages to its own directory so the two can coexist.
    """
    (bundle / "artifacts").mkdir(parents=True, exist_ok=True)
    (bundle / "artifacts/policy.onnx").write_bytes(onnx.read_bytes())
    shutil.copyfile(agent_component_path(mode), bundle / "component.wasm")
    (bundle / "lockstep.toml").write_text(
        "# Staged by train/main.py — do not hand-edit.\n"
        "#\n"
        "# `policy` is the artifact NAME the shell passes to `infer()`.\n"
        "schema_version = 1\n"
        f'game = "{GAME_SLUG}"\n'
        "# Read from the codec, never typed here: the api refuses an agent\n"
        "# whose declared version does not match the live game catalog.\n"
        f"payload_schema_version = {_native.PAYLOAD_SCHEMA_VERSION}\n"
        "# The ladder this agent targets (dance-off has more than one mode).\n"
        f'mode = "{mode}"\n'
        "\n"
        "[artifacts.policy]\n"
        'kind = "onnx"\n'
        'path = "artifacts/policy.onnx"\n'
    )
    return bundle


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=[MODE_SERVO_ASSIST, MODE_RAW_TORQUE], required=True)
    p.add_argument(
        "--steps",
        type=int,
        default=8192,
        help="environment steps. The default is a SHORT run: it produces a real "
        "policy, not a good one. Raise it by orders of magnitude to train.",
    )
    p.add_argument("--time-limit-ticks", type=int, default=1800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--engine",
        required=True,
        help="engine wasm: a local path (task engine downloads one) or an https URL",
    )
    p.add_argument(
        "--from-weights",
        default=None,
        help="skip training and export an existing .pt (for iterating on export/stage)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="update-pass device (cuda/mps/cpu); default: best available",
    )
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_weights:
        print(f"── loading weights {args.from_weights}")
        blob = torch.load(args.from_weights, weights_only=True)
        net = Policy(blob["action_len"])
        net.load_state_dict(blob["state_dict"])
    else:
        print(f"── training [{args.mode}] for {args.steps} steps")
        net = train(
            mode=args.mode,
            steps=args.steps,
            engine=args.engine,
            time_limit_ticks=args.time_limit_ticks,
            seed=args.seed,
            device=args.device,
        )
        weights = OUT_DIR / "policy.pt"
        torch.save({"action_len": net.action_len, "state_dict": net.state_dict()}, weights)
        print(f"→ weights: {weights}")

    onnx = export(net, OUT_DIR / "policy.onnx")
    print(f"→ onnx: {onnx} ({onnx.stat().st_size} bytes)")

    diff = verify(net, onnx)
    print(f"✓ torch/onnxruntime parity: max abs diff {diff:.3e}")

    bundle = stage(args.mode, onnx)
    print(f"→ bundle: {bundle}")
    print("\nRun it:   task match\nCompete:  task upload")


if __name__ == "__main__":
    sys.exit(main())
