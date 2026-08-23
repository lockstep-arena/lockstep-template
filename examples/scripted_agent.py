"""A NON-trained agent: hand-built policy, no RL, same pipeline — for ANY
environment.

The platform doesn't care where your ONNX came from — an agent is any graph
with the right signature (one input per observation tensor, by name; one
``action`` output in [-1, 1]) placed in a bundle next to the generic shell.
This example builds the simplest correct one: a graph that plays NEUTRAL.
The shell maps ``[-1, 1]`` outputs affinely onto each action element's
declared bounds, so an all-zeros output is every element's bounds midpoint
— the same "neutral" the tensor wire itself defines for a missing input.

Everything about the graph is derived from the engine's own tensor-wire
declaration (``lockstep_train`` reads it): input names and shapes are the
declared observation tensors, the action width is the declared action
tensor. No environment is named anywhere.

It exports and parity-checks through the exact same code path as the
trained policy — any module exposing ``input_names`` / ``input_shapes`` /
``action_len`` rides the same seam — and stages its own bundle
(``out/scripted-bundle``) so it can live next to a trained one::

    task scripted ENV=<slug>
    task match ENV=<slug> BUNDLE=out/scripted-bundle

From here, make it YOURS: replace ``forward`` with any closed-form policy —
read a slice of an observation vector (the declaration names them; see the
environment's Interface page), gate on it, output whatever pose you like.
The Rust twin of this example (``examples/rust-agent``) shows the same idea
without ONNX at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

from train.core import utf8_output
from train.core.export import export, verify
from train.core.stage import stage
from train.main import OUT_DIR, SHELL_WASM, engine_identity

BUNDLE = OUT_DIR / "scripted-bundle"


class Neutral(nn.Module):
    """Every action element at its bounds midpoint, every tick.

    The graph still declares every observation tensor as an input (the
    export seam checks the signature, and the shell feeds them all); each
    contributes ``0.0 ×`` its mean so the exporter cannot prune them away.
    """

    def __init__(
        self,
        input_shapes: dict[str, tuple[int, ...]],
        input_dtypes: dict[str, str],
        action_len: int,
    ):
        super().__init__()
        self.input_names = list(input_shapes)
        self.input_shapes = dict(input_shapes)
        self.input_dtypes = dict(input_dtypes)
        self.action_len = action_len

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs[0].shape[0]
        keep_alive = sum(t.float().mean() for t in inputs) * 0.0
        return torch.zeros(batch, self.action_len) + keep_alive


def graph_signature(engine: Path) -> tuple[dict[str, tuple[int, ...]], dict[str, str], int]:
    """(input name -> declared shape, name -> graph dtype, action width)
    from the engine's wire declaration. Graph dtypes mirror the generic
    shell exactly: u8 images arrive as normalized float32, i32 tensors as
    int32, f32 as-is."""
    import numpy as np
    from lockstep_train.env import LockstepEnv

    env = LockstepEnv(engine_source=str(engine))
    try:
        shapes: dict[str, tuple[int, ...]] = {}
        dtypes: dict[str, str] = {}
        for name, box in env.observation_space.spaces.items():
            shapes[name] = tuple(int(d) for d in box.shape)
            dtypes[name] = "int32" if box.dtype == np.int32 else "float32"
        (action_len,) = env.action_space.shape
        return shapes, dtypes, int(action_len)
    finally:
        env.close()


def main() -> None:
    utf8_output()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True, help="environment slug (ENV= on the task line)")
    p.add_argument("--engine", default=str(OUT_DIR / "engine.wasm"))
    p.add_argument("--shell", default=str(SHELL_WASM))
    args = p.parse_args()

    engine = Path(args.engine)
    if not engine.is_file():
        raise SystemExit(f"no engine at {engine} — run: task engine ENV={args.env}")
    mode, payload_schema_version = engine_identity(engine)
    shapes, dtypes, action_len = graph_signature(engine)

    net = Neutral(shapes, dtypes, action_len)
    onnx = export(net, BUNDLE.parent / "scripted-policy.onnx")
    print(f"→ onnx: {onnx} ({onnx.stat().st_size} bytes)")
    diff = verify(net, onnx)
    print(f"✓ torch/onnxruntime parity: max abs diff {diff:.3e}")
    bundle = stage(args.env, mode, payload_schema_version, onnx, Path(args.shell), BUNDLE)
    print(f"→ bundle: {bundle}")
    print(f"\nRun it:   task match ENV={args.env} BUNDLE=out/scripted-bundle")


if __name__ == "__main__":
    sys.exit(main())
